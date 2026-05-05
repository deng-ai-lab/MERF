import torch
import torch.nn as nn
import torch.nn.functional as F

from model.agent_model import BaseAgent
from model.qmixer import QMixer
from model.geo_ipa_encoder import GeoIPAEncoder


class MERF(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args

        self.encoder = GeoIPAEncoder(args)
        self.agent = BaseAgent(args)
        self.mixer = QMixer(args)

        self.loss_fn = torch.nn.MSELoss()
        self.ln = nn.LayerNorm(self.args.obs_shape)

        self.use_rank_loss = getattr(args, 'use_rank_loss', False)
        self.rank_loss_weight = getattr(args, 'rank_loss_weight', 0.1)
        self.rank_tie_threshold = getattr(args, 'rank_tie_threshold', 0.0)

    def add_aa_id(self, batch_embed, complex_info):
        aa_onehot = F.one_hot(complex_info['aa'], 21)
        return torch.cat((batch_embed, aa_onehot), dim=2)

    def compute_rank_loss(self, qs_wt, batch, device):
        if 'rank_target' not in batch or 'rank_mask' not in batch or 'has_rank_target' not in batch:
            return torch.tensor(0.0, device=device)

        mutation_mask = batch['mutation_mask']
        rank_target = batch['rank_target'].to(device).float()
        rank_mask = batch['rank_mask'].to(device).bool()
        has_rank_target = batch['has_rank_target'].to(device).bool()

        sample_losses = []
        batch_size = qs_wt.shape[0]

        for i in range(batch_size):
            if not has_rank_target[i]:
                continue

            mut_pos = mutation_mask[i].nonzero(as_tuple=True)[0]
            if len(mut_pos) == 0:
                continue

            site_q = qs_wt[i, mut_pos[0]]
            observed_idx = rank_mask[i].nonzero(as_tuple=True)[0]
            if observed_idx.numel() < 2:
                continue

            site_ddg = rank_target[i, observed_idx]
            site_q = site_q[observed_idx]

            pair_losses = []
            num_observed = observed_idx.numel()
            for left in range(num_observed):
                for right in range(left + 1, num_observed):
                    ddg_left = site_ddg[left]
                    ddg_right = site_ddg[right]
                    ddg_diff = ddg_left - ddg_right

                    if torch.abs(ddg_diff) <= self.rank_tie_threshold:
                        continue

                    if ddg_left < ddg_right:
                        better_q = site_q[left]
                        worse_q = site_q[right]
                    else:
                        better_q = site_q[right]
                        worse_q = site_q[left]

                    pair_losses.append(F.softplus(-(worse_q - better_q)))

            if pair_losses:
                sample_losses.append(torch.stack(pair_losses).mean())

        if not sample_losses:
            return torch.tensor(0.0, device=device)

        return torch.stack(sample_losses).mean()

    def forward(self, batch, device):
        ddG = batch['ddG'].to(device)
        ddG = ddG.to(torch.float32)

        batch_embedding_wt = self.encoder(batch['wt'], batch['mutation_mask'])
        batch_embedding_mut = self.encoder(batch['mut'], batch['mutation_mask'])

        batch_embedding_wt = self.ln(batch_embedding_wt)
        batch_embedding_mut = self.ln(batch_embedding_mut)

        batch_embedding_wt_ = self.add_aa_id(batch_embedding_wt, batch['wt'])
        qs_wt = self.agent(batch_embedding_wt_)

        q_tot, _ = self.mixer(qs_wt, batch, device, batch_embedding_wt, batch_embedding_mut)

        mse_loss = self.loss_fn(q_tot, ddG.squeeze())

        if self.use_rank_loss:
            rank_loss = self.compute_rank_loss(qs_wt, batch, device)
            total_loss = mse_loss + self.rank_loss_weight * rank_loss
        else:
            rank_loss = torch.tensor(0.0, device=device)
            total_loss = mse_loss

        return q_tot, total_loss, mse_loss, rank_loss

    def choose_best_action(self, batch, device):
        batch_embedding_wt = self.encoder(batch['wt'], batch['mutation_mask'])
        batch_embedding_wt = self.ln(batch_embedding_wt)
        batch_embedding_wt_ = self.add_aa_id(batch_embedding_wt, batch['wt'])
        qs_wt = self.agent(batch_embedding_wt_)
        return qs_wt
