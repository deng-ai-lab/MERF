import torch
import torch.nn as nn
import torch.nn.functional as F

from model.agent_model_v6 import BaseAgent
from model.qmixer import QMixer
from model.geo_ipa_encoder import GeoIPAEncoder
from utils.util import *


class MERF(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        # network init
        self.encoder = GeoIPAEncoder(args)
        self.agent = BaseAgent(args)
        self.mixer = QMixer(args)

        # loss function
        self.loss_fn = torch.nn.MSELoss()

        # self.bn = nn.BatchNorm1d(num_features=self.args.obs_shape)
        self.ln = nn.LayerNorm(self.args.obs_shape)

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

        ddG = batch['ddG'].to(device)
        ddG = ddG.to(torch.float32)

        # ----------------------- Local mutation policy generation ----------------------- #
        # feature update
        batch_embedding_wt = self.encoder(batch['wt'], batch['mutation_mask'])
        batch_embedding_mut = self.encoder(batch['mut'], batch['mutation_mask'])

        # batch_embedding_wt = self.bn(batch_embedding_wt.transpose(1, 2)).transpose(1, 2)
        # batch_embedding_mut = self.bn(batch_embedding_mut.transpose(1, 2)).transpose(1, 2)

        batch_embedding_wt = self.ln(batch_embedding_wt)
        batch_embedding_mut = self.ln(batch_embedding_mut)

        # policy making
        batch_embedding_wt_ = self.add_aa_id(batch_embedding_wt, batch['wt'])
        qs_wt = self.agent(batch_embedding_wt_)

        # ----------------------- global mutational effects estimation ----------------------- #
        q_tot, _ = self.mixer(qs_wt, batch, device, batch_embedding_wt, batch_embedding_mut)

        mse_loss = self.loss_fn(q_tot, ddG.squeeze())

        # ----------------------- KL distribution loss (optional) ----------------------- #
        if self.use_kl_loss:
            kl_loss = self.compute_kl_loss(qs_wt, batch, device)
            total_loss = mse_loss + self.kl_loss_weight * kl_loss
        else:
            kl_loss = torch.tensor(0.0, device=device)
            total_loss = mse_loss

        return q_tot, total_loss, mse_loss, kl_loss

    def choose_best_action(self, batch, device):

        batch_embedding_wt = self.encoder(batch['wt'], batch['mutation_mask'])
        
        # batch_embedding_wt = self.bn(batch_embedding_wt.transpose(1, 2)).transpose(1, 2)
        batch_embedding_wt = self.ln(batch_embedding_wt)
        
        batch_embedding_wt_ = self.add_aa_id(batch_embedding_wt, batch['wt'])

        qs_wt = self.agent(batch_embedding_wt_)
        
        return qs_wt





