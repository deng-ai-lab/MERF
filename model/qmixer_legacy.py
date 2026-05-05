import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QMixer(nn.Module):
    def __init__(self, args):
        self.args = args
        super(QMixer, self).__init__()

        self.n_agents = args.n_agents
        self.state_dim = args.global_state_dim
        self.embed_dim = args.mixing_embed_dim

        if args.hypernet_layers == 2:
            hypernet_embed = args.hypernet_embed
            self.hyper_w_1 = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim * self.n_agents))
            self.hyper_w_final = nn.Sequential(nn.Linear(self.state_dim, hypernet_embed),
                                           nn.ReLU(),
                                           nn.Linear(hypernet_embed, self.embed_dim))
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b_1 = nn.Linear(self.state_dim, self.embed_dim)

        self.V = nn.Sequential(nn.Linear(self.state_dim, self.embed_dim),
                               nn.ReLU(),
                               nn.Linear(self.embed_dim, 1))

    def select_prepare_q(self, qs, batch, device):
        wt = 'wt'
        mut = 'mut'
        batch_size, aa_len, _ = qs.size()
        
        aa_idx_wt = batch[wt]['aa']
        aa_idx_mut = batch[mut]['aa']

        # select action-value functions for agents
        qs_sele = torch.zeros(batch_size, aa_len).to(device)
        
        for i in range(batch_size):
            for j in range(aa_len):
                if batch[wt]['agent_mask'][i, j] == False or batch['mask'][i, j] == False:
                    continue
                if batch['mutation_mask'][i, j]:
                    qs_sele[i, j] = qs[i, j, aa_idx_mut[i, j]]
                else:
                    qs_sele[i, j] = qs[i, j, aa_idx_wt[i, j]]

        return qs_sele

    def reshape_q(self, q_wt, batch, device):
        batch_size, aa_len = q_wt.size()
        aa_idx_wt = batch['wt']['aa']

        Q_mixer_input = torch.zeros(batch_size, 20).to(device)

        for i in range(batch_size):
            for j in range(aa_len):
                if batch['wt']['agent_mask'][i, j] == False or batch['mask'][i, j] == False:
                    continue
                Q_mixer_input[i, aa_idx_wt[i, j]] += q_wt[i, j]

        return Q_mixer_input

    def prepare_global_state(self, batch_embedding_wt, batch_embedding_mut, batch, global_state_len, n_agents, device):
        batch_size, aa_len, fea_num = batch_embedding_wt.size()
        global_state = torch.zeros(batch_size, n_agents, fea_num).to(device)

        aa_idx_wt = batch['wt']['aa']
        aa_idx_mut = batch['mut']['aa']

        for i in range(batch_size):
            nums = torch.zeros(20).to(device)
            for j in range(aa_len):
                if batch['wt']['agent_mask'][i, j] == False or batch['mask'][i, j] == False:
                    continue
                aa_type_wt = aa_idx_wt[i, j]
                nums[aa_type_wt] += 1
                global_state[i, aa_type_wt, :] += batch_embedding_wt[i, j, :]

                aa_type_mut = aa_idx_mut[i, j]
                nums[aa_type_mut] += 1
                global_state[i, aa_type_mut, :] += batch_embedding_mut[i, j, :]

            for j in range(20):
                if nums[j] == 0:
                    continue
                global_state[i, j, :] = global_state[i, j, :] / nums[j]
        global_state = global_state.reshape([batch_size, global_state_len])
        return global_state

    def forward(self, qs_wt, batch, device, batch_embedding_wt, batch_embedding_mut):
        # select and prepare q for each agent
        q_sele_wt = self.select_prepare_q(qs_wt, batch, device=device)
        agent_qs = self.reshape_q(q_sele_wt, batch, device)
        bs = agent_qs.size(0)

        # prepare global state
        global_states = self.prepare_global_state(batch_embedding_wt, batch_embedding_mut, batch,
                                                              global_state_len=self.args.global_state_dim,
                                                              n_agents=self.args.n_agents, device=device)

        # value-mixing with hyper network
        global_states = global_states.reshape(-1, self.state_dim)
        agent_qs = agent_qs.view(-1, 1, self.n_agents)

        # First layer
        w1 = torch.abs(self.hyper_w_1(global_states))
        b1 = self.hyper_b_1(global_states)
        w1 = w1.view(-1, self.n_agents, self.embed_dim)
        b1 = b1.view(-1, 1, self.embed_dim)
        
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)
        
        # Second layer
        w_final = torch.abs(self.hyper_w_final(global_states))
        b_final = self.V(global_states)
        w_final = w_final.view(-1, self.embed_dim, 1)
        b_final = b_final.view(-1, 1, 1)

        y = torch.bmm(hidden, w_final) + b_final

        # Reshape and return
        q_tot = y.view(bs, -1, 1)

        return q_tot, q_sele_wt