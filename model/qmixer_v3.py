import torch
import torch.nn as nn
import torch.nn.functional as F
from protein.read_pdbs import _mask_dict_recursively
from model.geomattn_encoder import GeometricAttention
from model.geo_ipa_encoder import PerResidueEncoder, construct_3d_basis, get_pos_CB

ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


class QMixer(nn.Module):
    """QMixer without PLM information — structure-only global state."""

    def __init__(self, args):
        super(QMixer, self).__init__()
        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim

        # global state dim is feat_dim only (no PLM)
        self.mixer_feat_dim = args.feat_dim
        self.mixer_rel_dim = args.mixer_rel_dim
        self.max_relpos = args.max_relpos
        self.mixer_ga_layer = args.mixer_ga_layer

        self.relpos_embedding = nn.Embedding(self.max_relpos * 2 + 2, self.mixer_rel_dim)
        self.GA_blocks = nn.ModuleList([
            GeometricAttention(self.mixer_feat_dim, self.mixer_rel_dim, spatial_attn_mode='CB')
            for _ in range(self.mixer_ga_layer)
        ])

        hypernet_embed = args.hypernet_embed
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(self.mixer_feat_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim * self.n_agents)
        )
        self.hyper_w_final = nn.Sequential(
            nn.Linear(self.mixer_feat_dim, hypernet_embed), nn.ReLU(),
            nn.Linear(hypernet_embed, self.embed_dim)
        )
        self.hyper_b_1 = nn.Linear(self.mixer_feat_dim, self.embed_dim)
        self.V = nn.Sequential(
            nn.Linear(self.mixer_feat_dim, self.embed_dim), nn.ReLU(),
            nn.Linear(self.embed_dim, 1)
        )

    def focus_on_agents(self, qs_wt, batch, batch_embedding_wt, batch_embedding_mut):
        batch_size, aa_len, n_actions = qs_wt.size()
        agent_mask = batch['wt']['agent_mask']

        qs_agent = qs_wt[agent_mask].view(batch_size, self.n_agents, n_actions)
        batch_embedding_wt_agents = batch_embedding_wt[agent_mask].view(batch_size, self.n_agents, -1)
        batch_embedding_mut_agents = batch_embedding_mut[agent_mask].view(batch_size, self.n_agents, -1)

        batch_agents = {}
        batch_agents['mutation_mask'] = batch['mutation_mask'][agent_mask].view(batch_size, self.n_agents)
        batch_agents['mask'] = batch['mask'][agent_mask].view(batch_size, self.n_agents)

        batch_agents.setdefault('wt', {})['aa'] = batch['wt']['aa'][agent_mask].view(batch_size, self.n_agents)
        batch_agents.setdefault('mut', {})['aa'] = batch['mut']['aa'][agent_mask].view(batch_size, self.n_agents)
        batch_agents.setdefault('wt', {})['pos14'] = batch['wt']['pos14'][agent_mask].view(batch_size, self.n_agents, 14, 3)
        batch_agents.setdefault('mut', {})['pos14'] = batch['mut']['pos14'][agent_mask].view(batch_size, self.n_agents, 14, 3)
        batch_agents.setdefault('wt', {})['seq'] = batch['wt']['seq'][agent_mask].view(batch_size, self.n_agents)
        batch_agents.setdefault('mut', {})['seq'] = batch['mut']['seq'][agent_mask].view(batch_size, self.n_agents)
        batch_agents.setdefault('wt', {})['chain_seq'] = batch['wt']['chain_seq'][agent_mask].view(batch_size, self.n_agents)
        batch_agents.setdefault('mut', {})['chain_seq'] = batch['mut']['chain_seq'][agent_mask].view(batch_size, self.n_agents)
        batch_agents.setdefault('wt', {})['pos14_mask'] = batch['wt']['pos14_mask'][agent_mask].view(batch_size, self.n_agents, 14, 3)
        batch_agents.setdefault('mut', {})['pos14_mask'] = batch['mut']['pos14_mask'][agent_mask].view(batch_size, self.n_agents, 14, 3)

        return qs_agent, batch_agents, batch_embedding_wt_agents, batch_embedding_mut_agents

    def select_prepare_qs(self, qs, batch):
        batch_size, agent_num, _ = qs.size()
        aa_idx_wt = batch['wt']['aa']
        aa_idx_mut = batch['mut']['aa']

        qs_sele = torch.zeros(batch_size, agent_num).to(qs.device)
        for i in range(batch_size):
            for j in range(agent_num):
                if batch['mask'][i, j] == False:
                    continue
                if batch['mutation_mask'][i, j]:
                    qs_sele[i, j] = qs[i, j, aa_idx_mut[i, j]]
                else:
                    qs_sele[i, j] = qs[i, j, aa_idx_wt[i, j]]
        return qs_sele

    def forward(self, qs_wt, batch, device, batch_embedding_wt, batch_embedding_mut):
        qs_agents, batch_agents, batch_embedding_wt_agents, batch_embedding_mut_agents = self.focus_on_agents(
            qs_wt, batch, batch_embedding_wt, batch_embedding_mut
        )
        qs_input = self.select_prepare_qs(qs_agents, batch_agents)

        # global state: structure diff only, no PLM
        global_states = batch_embedding_mut_agents - batch_embedding_wt_agents

        # GA over agent residues
        res_feat_GA = global_states
        pos14 = batch_agents['wt']['pos14']
        aa = batch_agents['wt']['aa']
        seq = batch_agents['wt']['seq']
        chain = batch_agents['wt']['chain_seq']
        mask_atom = batch_agents['wt']['pos14_mask'].all(dim=-1)

        same_chain = (chain[:, None, :] == chain[:, :, None])
        relpos = (seq[:, None, :] - seq[:, :, None]).clamp(
            min=-self.max_relpos, max=self.max_relpos) + self.max_relpos
        relpos = torch.where(same_chain, relpos, torch.full_like(relpos, fill_value=self.max_relpos * 2 + 1))
        pair_feat = self.relpos_embedding(relpos)
        R = construct_3d_basis(pos14[:, :, ATOM_CA], pos14[:, :, ATOM_C], pos14[:, :, ATOM_N])
        t = pos14[:, :, ATOM_CA]
        mask_residue = mask_atom[:, :, ATOM_CA]

        for GA_block in self.GA_blocks:
            res_feat_GA = GA_block(R, t, get_pos_CB(pos14, mask_atom), res_feat_GA, pair_feat, mask_residue)

        updated_global_states = res_feat_GA.mean(dim=1)  # (batch_size, feat_dim)

        # hypernet mixing
        qs_input = qs_input.unsqueeze(1)  # (batch_size, 1, n_agents)
        w1 = torch.abs(self.hyper_w_1(updated_global_states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(updated_global_states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(qs_input, w1) + b1)

        w_final = torch.abs(self.hyper_w_final(updated_global_states)).view(-1, self.embed_dim, 1)
        b_final = self.V(updated_global_states).view(-1, 1, 1)
        q_tot = torch.bmm(hidden, w_final) + b_final

        return q_tot.squeeze(), qs_input
