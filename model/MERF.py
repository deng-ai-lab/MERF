import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.agent_model import BaseAgent
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
        
        self.bn = nn.BatchNorm1d(num_features=self.args.obs_shape)

    def add_aa_id(self, batch_embed, complex_info):
        aa_onehot = F.one_hot(complex_info['aa'], 21)
        return torch.cat((batch_embed, aa_onehot), dim=2)
    
    def generate_positional_embedding(self, feature_dim):
        pos = torch.arange(2).unsqueeze(1)  # [seq_len, 1]
        div_term = torch.exp(torch.arange(0, feature_dim, 2) * (-math.log(10000.0) / feature_dim))
        pe = torch.zeros(2, feature_dim)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        
        return pe
        
    def add_mutation_mask(self, batch_embed, mutation_mask):
        bs, seq_len, feat_dim = batch_embed.shape
        
        # by positional encoding, no new dimension added
        
        pe = self.generate_positional_embedding(feat_dim).to(batch_embed.device)
        mutation_mask_ = mutation_mask.unsqueeze(-1).repeat(1, 1, feat_dim)

        true_encoding = pe[1, :].unsqueeze(0).unsqueeze(0).repeat(bs, seq_len, 1)  # [64, 128, 128]
        false_encoding = pe[0, :].unsqueeze(0).unsqueeze(0).repeat(bs, seq_len, 1)  # [64, 128, 128]

        selected_encoding = torch.where(mutation_mask_, true_encoding, false_encoding)
        
        output_embed = batch_embed + selected_encoding
        
        return output_embed
    
    def forward(self, batch, device):

        # get batch size
        batch_size = len(batch['wt']['name'])

        # get label
        ddG = batch['ddG'].to(device)
        ddG = ddG.to(torch.float32)

        # ----------------------- Local mutation policy generation ----------------------- #
        # feature update
        batch_embedding_wt, _ = self.encoder(batch['wt'], batch['mutation_mask'])
        batch_embedding_mut, _ = self.encoder(batch['mut'], batch['mutation_mask'])

        batch_embedding_wt = self.bn(batch_embedding_wt.transpose(1, 2)).transpose(1, 2)
        batch_embedding_mut = self.bn(batch_embedding_mut.transpose(1, 2)).transpose(1, 2)
        
        # policy making
        batch_embedding_wt_ = self.add_aa_id(batch_embedding_wt, batch['wt'])
        qs_wt = self.agent(batch_embedding_wt_)
        
        # ----------------------- global mutational effects estimation ----------------------- #
        q_tot, _ = self.mixer(qs_wt, batch, device, batch_embedding_wt, batch_embedding_mut)

        q_tot = q_tot.reshape(batch_size)
        
        loss = self.loss_fn(q_tot, ddG)

        return q_tot, loss

    def choose_best_action(self, batch, device):

        batch_embedding_wt, _ = self.encoder(batch['wt'], batch['mutation_mask'])
        
        batch_embedding_wt = self.bn(batch_embedding_wt.transpose(1, 2)).transpose(1, 2)
        
        batch_embedding_wt_ = self.add_aa_id(batch_embedding_wt, batch['wt'])

        qs_wt = self.agent(batch_embedding_wt_)
        
        return qs_wt





