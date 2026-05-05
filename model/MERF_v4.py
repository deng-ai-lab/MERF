import torch
import torch.nn as nn
import torch.nn.functional as F

from model.agent_model import BaseAgent
from model.qmixer_v3 import QMixer
from model.geo_ipa_encoder import GeoIPAEncoder

# -----------------------------------------------------------------------
# ESM2 vocabulary → AA_KEY index mapping
#
# AA_KEY used throughout this codebase: "ACDEFGHIKLMNPQRSTVWY"
# (same as Bio.PDB.Polypeptide.three_to_index alphabetical order)
#
# ESM2 token IDs for standard amino acids (from ESM2 tokenizer):
#   L=4, A=5, G=6, V=7, S=8, E=9, R=10, T=11, I=12, D=13,
#   P=14, K=15, Q=16, N=17, F=18, Y=19, M=20, H=21, W=22, C=23
#
# We build ESM2_TO_AAKEY: a LongTensor of length 33 (ESM2 vocab size)
# where ESM2_TO_AAKEY[esm2_token_id] = AA_KEY index (0-19), or -1 if not a standard AA.
# -----------------------------------------------------------------------
AA_KEY = "ACDEFGHIKLMNPQRSTVWY"
_AA_KEY_IDX = {aa: i for i, aa in enumerate(AA_KEY)}

# ESM2 token_id -> single-letter AA (only standard 20)
_ESM2_TOKEN_TO_AA = {
    4: 'L', 5: 'A', 6: 'G', 7: 'V', 8: 'S', 9: 'E', 10: 'R', 11: 'T',
    12: 'I', 13: 'D', 14: 'P', 15: 'K', 16: 'Q', 17: 'N', 18: 'F', 19: 'Y',
    20: 'M', 21: 'H', 22: 'W', 23: 'C',
}

# ESM2 vocab size is 33
_ESM2_VOCAB_SIZE = 33
_esm2_to_aakey = torch.full((_ESM2_VOCAB_SIZE,), -1, dtype=torch.long)
for tok_id, aa in _ESM2_TOKEN_TO_AA.items():
    _esm2_to_aakey[tok_id] = _AA_KEY_IDX[aa]

# Ordered list of ESM2 token IDs corresponding to AA_KEY order
# i.e., ESM2_AA_TOKEN_IDS[i] is the ESM2 token id for AA_KEY[i]
_AA_TO_ESM2_TOKEN = {aa: tok for tok, aa in _ESM2_TOKEN_TO_AA.items()}
ESM2_AA_TOKEN_IDS = [_AA_TO_ESM2_TOKEN[aa] for aa in AA_KEY]  # length 20


class PLMDdGHead(nn.Module):
    """
    Estimates ddG from WT PLM embedding only.

    Using only the WT context, the lm_head decodes a log-probability over all 20 AAs
    at each position. For a mutation wt_aa -> mut_aa, the PLM-based ddG contribution is:
        ddg_contrib = log_p(wt_aa | wt_context) - log_p(mut_aa | wt_context)
    If PLM thinks mut_aa is more likely than wt_aa, contribution is negative (favorable).
    Sum over all mutation sites gives the full PLM ddG estimate.

    Logit columns are reordered from ESM2 vocab order to AA_KEY order.
    """

    def __init__(self, plm_model):
        super().__init__()
        # lm_head from ESM2ForMaskedLM: dense(1280->1280) + layer_norm + decoder(1280->33)
        self.lm_head = plm_model.lm_head
        self.register_buffer('esm2_aa_token_ids', torch.tensor(ESM2_AA_TOKEN_IDS, dtype=torch.long))

    def forward(self, plm_embedding_wt, aa_wt, aa_mut, mutation_mask):
        """
        Args:
            plm_embedding_wt: (N, L, 1280) — WT per-residue embeddings only
            aa_wt:            (N, L)        — WT amino acid indices in AA_KEY order (0-19, 20=unknown)
            aa_mut:           (N, L)        — mutant amino acid indices in AA_KEY order
            mutation_mask:    (N, L) bool   — True at mutated positions

        Returns:
            ddg_plm: (N,) — PLM-based ddG estimate, summed over mutation sites
        """
        full_logits = self.lm_head(plm_embedding_wt)           # (N, L, 33)
        aa_logits = full_logits[:, :, self.esm2_aa_token_ids]  # (N, L, 20) in AA_KEY order
        log_p = F.log_softmax(aa_logits, dim=-1)               # (N, L, 20)

        # Gather log-probs for wt and mut amino acids at each position
        log_p_wt_aa  = log_p.gather(-1, aa_wt.clamp(max=19).unsqueeze(-1)).squeeze(-1)   # (N, L)
        log_p_mut_aa = log_p.gather(-1, aa_mut.clamp(max=19).unsqueeze(-1)).squeeze(-1)  # (N, L)

        # ddG sign convention: lower is better (favorable mutation)
        # PLM: higher log_p(mut_aa) means more favorable → subtract to get ddG direction
        per_site = log_p_wt_aa - log_p_mut_aa  # positive = destabilizing

        # Mask out non-mutated positions and unknown residues (index 20)
        valid = mutation_mask & (aa_wt < 20) & (aa_mut < 20)
        ddg_plm = (per_site * valid.float()).sum(dim=-1)  # (N,)

        return ddg_plm


class MERF(nn.Module):

    def __init__(self, args, plm_model=None):
        super().__init__()
        self.args = args

        # Structure branch (no PLM info)
        self.encoder = GeoIPAEncoder(args)
        self.agent = BaseAgent(args)
        self.mixer = QMixer(args)
        self.ln = nn.LayerNorm(args.obs_shape)

        # PLM branch
        self.use_plm_head = (plm_model is not None)
        if self.use_plm_head:
            self.plm_ddg_head = PLMDdGHead(plm_model)

        self.loss_fn = nn.MSELoss()

        # KL loss settings
        self.use_kl_loss = getattr(args, 'use_kl_loss', False)
        self.kl_loss_weight = getattr(args, 'kl_loss_weight', 0.1)

    def add_aa_id(self, batch_embed, complex_info):
        aa_onehot = F.one_hot(complex_info['aa'], 21)
        return torch.cat((batch_embed, aa_onehot), dim=2)

    def compute_kl_loss(self, qs_wt, batch, device):
        """Compute KL divergence loss between Q-value distribution and target ddG distribution.

        Args:
            qs_wt: Q-values from agent, shape [batch_size, seq_len, 20]
            batch: batch dict containing 'mutation_mask', 'kl_target', 'has_kl_target'
            device: torch device

        Returns:
            kl_loss: KL divergence loss (scalar), or 0 if no valid samples
        """
        if 'kl_target' not in batch or 'has_kl_target' not in batch:
            return torch.tensor(0.0, device=device)

        mutation_mask = batch['mutation_mask']  # [batch_size, seq_len]
        kl_target = batch['kl_target'].to(device).float()  # [batch_size, 20]
        has_kl_target = batch['has_kl_target'].to(device)  # [batch_size]

        batch_size = qs_wt.shape[0]

        # Collect Q-values at mutation positions for each sample
        qs_mutation_list = []
        for i in range(batch_size):
            # Get mutation positions for this sample
            mut_pos = mutation_mask[i].nonzero(as_tuple=True)[0]
            if len(mut_pos) > 0:
                # For single mutation, take the first (and only) mutation position
                qs_mutation_list.append(qs_wt[i, mut_pos[0]])
            else:
                # No mutation found, use zeros (will be masked out anyway)
                qs_mutation_list.append(torch.zeros(20, device=device))

        qs_mutation = torch.stack(qs_mutation_list, dim=0)  # [batch_size, 20]

        # Only compute loss for samples with valid KL targets
        valid_mask = has_kl_target  # [batch_size]
        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=device)

        # Get valid samples
        qs_valid = qs_mutation[valid_mask]  # [n_valid, 20]
        kl_target_valid = kl_target[valid_mask]  # [n_valid, 20]

        # Convert Q-values to probability distribution
        # Lower Q-value means better mutation, so we use -Q for softmax
        pred_prob = F.softmax(-qs_valid, dim=1)  # [n_valid, 20]

        # Convert target ddG to probability distribution
        # Lower ddG means better mutation, so we use -ddG for softmax
        target_prob = F.softmax(-kl_target_valid, dim=1)  # [n_valid, 20]

        # Compute KL divergence: KL(target || pred)
        # Using F.kl_div with log_target=False
        eps = 1e-8
        kl_loss = (target_prob * (torch.log(target_prob + eps) - torch.log(pred_prob + eps))).sum(dim=1).mean()

        return kl_loss

    def forward(self, batch, device):
        ddG = batch['ddG'].to(device).float()

        # ---- Structure branch (no PLM) ----
        batch_embedding_wt = self.encoder(batch['wt'], batch['mutation_mask'])
        batch_embedding_mut = self.encoder(batch['mut'], batch['mutation_mask'])
        batch_embedding_wt = self.ln(batch_embedding_wt)
        batch_embedding_mut = self.ln(batch_embedding_mut)

        batch_embedding_wt_ = self.add_aa_id(batch_embedding_wt, batch['wt'])
        qs_wt = self.agent(batch_embedding_wt_)

        q_struct, _ = self.mixer(qs_wt, batch, device, batch_embedding_wt, batch_embedding_mut)

        # ---- PLM branch ----
        if self.use_plm_head:
            plm_emb_wt = batch['wt']['plm_embedding']  # (N, L, 1280)
            ddg_plm = self.plm_ddg_head(
                plm_emb_wt, batch['wt']['aa'], batch['mut']['aa'], batch['mutation_mask']
            )
            q_tot = q_struct + ddg_plm
        else:
            q_tot = q_struct

        mse_loss = self.loss_fn(q_tot, ddG.squeeze())

        if self.use_kl_loss:
            kl_loss = self.compute_kl_loss(qs_wt, batch, device)
            total_loss = mse_loss + self.kl_loss_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=device)
            total_loss = mse_loss

        return q_tot, total_loss, mse_loss, kl_loss

    def choose_best_action(self, batch, device):
        batch_embedding_wt = self.encoder(batch['wt'], batch['mutation_mask'])
        batch_embedding_wt = self.ln(batch_embedding_wt)
        batch_embedding_wt_ = self.add_aa_id(batch_embedding_wt, batch['wt'])
        return self.agent(batch_embedding_wt_)
