import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from invariant_point_attention import InvariantPointAttention
from model.geomattn_encoder import GAEncoder


ATOM_N, ATOM_CA, ATOM_C, ATOM_O, ATOM_CB = 0, 1, 2, 3, 4


def get_pos_CB(pos14, atom_mask):
    """
    Args:
        pos14:  (N, L, 14, 3)
        atom_mask:  (N, L, 14)
    """
    N, L = pos14.shape[:2]
    mask_CB = atom_mask[:, :, ATOM_CB]  # (N, L)
    mask_CB = mask_CB[:, :, None].expand(N, L, 3)
    pos_CA = pos14[:, :, ATOM_CA]  # (N, L, 3)
    pos_CB = pos14[:, :, ATOM_CB]
    return torch.where(mask_CB, pos_CB, pos_CA)


def mask_zero(mask, value):
    return torch.where(mask, value, torch.zeros_like(value))


class PositionalEncoding(nn.Module):

    def __init__(self, num_funcs=6):
        super().__init__()
        self.num_funcs = num_funcs
        self.register_buffer('freq_bands', 2.0 ** torch.linspace(0.0, num_funcs - 1, num_funcs))

    def get_out_dim(self, in_dim):
        return in_dim * (2 * self.num_funcs + 1)

    def forward(self, x):
        """
        Args:
            x:  (..., d).
        """
        shape = list(x.shape[:-1]) + [-1]
        x = x.unsqueeze(-1)  # (..., d, 1)
        code = torch.cat([x, torch.sin(x * self.freq_bands), torch.cos(x * self.freq_bands)], dim=-1)  # (..., d, 2f+1)
        code = code.reshape(shape)
        return code


def safe_norm(x, dim=-1, keepdim=False, eps=1e-8, sqrt=True):
    out = torch.clamp(torch.sum(torch.square(x), dim=dim, keepdim=keepdim), min=eps)
    return torch.sqrt(out) if sqrt else out


def normalize_vector(v, dim, eps=1e-6):
    return v / (torch.linalg.norm(v, ord=2, dim=dim, keepdim=True) + eps)


def project_v2v(v, e, dim):
    """
    Description:
        Project vector `v` onto vector `e`.
    Args:
        v:  (N, L, 3).
        e:  (N, L, 3).
    """
    return (e * v).sum(dim=dim, keepdim=True) * e


def construct_3d_basis(center, p1, p2):
    """
    Args:
        center: (N, L, 3), usually the position of C_alpha.
        p1:     (N, L, 3), usually the position of C.
        p2:     (N, L, 3), usually the position of N.
    Returns
        A batch of orthogonal basis matrix, (N, L, 3, 3cols_index).
        The matrix is composed of 3 column vectors: [e1, e2, e3].
    """
    v1 = p1 - center  # (N, L, 3)
    e1 = normalize_vector(v1, dim=-1)

    v2 = p2 - center  # (N, L, 3)
    u2 = v2 - project_v2v(v2, e1, dim=-1)
    e2 = normalize_vector(u2, dim=-1)

    e3 = torch.cross(e1, e2, dim=-1)  # (N, L, 3)

    mat = torch.cat([
        e1.unsqueeze(-1), e2.unsqueeze(-1), e3.unsqueeze(-1)
    ], dim=-1)  # (N, L, 3, 3_index)
    return mat


def local_to_global(R, t, p):
    """
    Description:
        Convert local (internal) coordinates to global (external) coordinates q.
        q <- Rp + t
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        p:  Local coordinates, (N, L, ..., 3).
    Returns:
        q:  Global coordinates, (N, L, ..., 3).
    """
    assert p.size(-1) == 3
    p_size = p.size()
    N, L = p_size[0], p_size[1]

    p = p.view(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)
    q = torch.matmul(R, p) + t.unsqueeze(-1)  # (N, L, 3, *)
    q = q.transpose(-1, -2).reshape(p_size)  # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return q


def global_to_local(R, t, q):
    """
    Description:
        Convert global (external) coordinates q to local (internal) coordinates p.
        p <- R^{T}(q - t)
    Args:
        R:  (N, L, 3, 3).
        t:  (N, L, 3).
        q:  Global coordinates, (N, L, ..., 3).
    Returns:
        p:  Local coordinates, (N, L, ..., 3).
    """
    assert q.size(-1) == 3
    q_size = q.size()
    N, L = q_size[0], q_size[1]

    q = q.reshape(N, L, -1, 3).transpose(-1, -2)  # (N, L, *, 3) -> (N, L, 3, *)
    if t is None:
        p = torch.matmul(R.transpose(-1, -2), q)  # (N, L, 3, *)
    else:
        p = torch.matmul(R.transpose(-1, -2), (q - t.unsqueeze(-1)))  # (N, L, 3, *)
    p = p.transpose(-1, -2).reshape(q_size)  # (N, L, 3, *) -> (N, L, *, 3) -> (N, L, ..., 3)
    return p


class PerResidueEncoder(nn.Module):

    def __init__(self, feat_dim):
        super().__init__()
        self.aatype_embed = nn.Embedding(21, feat_dim)
        self.torsion_embed = PositionalEncoding()
        self.mlp = nn.Sequential(
            nn.Linear(21*14*3 + feat_dim, feat_dim * 2), nn.ReLU(),
            nn.Linear(feat_dim * 2, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim), nn.ReLU(),
            nn.Linear(feat_dim, feat_dim)
        )

    def forward(self, aa, pos14, atom_mask):
        """
        Args:
            aa:           (N, L).
            pos14:        (N, L, 14, 3).
            atom_mask:    (N, L, 14).
        """
        N, L = aa.size()

        R = construct_3d_basis(pos14[:, :, 1], pos14[:, :, 2], pos14[:, :, 0])  # (N, L, 3, 3)
        t = pos14[:, :, 1]  # (N, L, 3) C_alpha pos
        crd14 = global_to_local(R, t, pos14)    # (N, L, 14, 3) get local pos for network input
        crd14_mask = atom_mask[:, :, :, None].expand_as(crd14)
        crd14 = torch.where(crd14_mask, crd14, torch.zeros_like(crd14))

        aa_expand = aa[:, :, None, None, None].expand(N, L, 21, 14, 3)
        rng_expand = torch.arange(0, 21)[None, None, :, None, None].expand(N, L, 21, 14, 3).to(aa_expand)
        place_mask = (aa_expand == rng_expand)
        crd_expand = crd14[:, :, None, :, :].expand(N, L, 21, 14, 3)
        crd_expand = torch.where(place_mask, crd_expand, torch.zeros_like(crd_expand))
        crd_feat = crd_expand.reshape(N, L, 21 * 14 * 3)

        aa_feat = self.aatype_embed(aa) # (N, L, feat)

        out_feat = self.mlp(torch.cat([crd_feat, aa_feat], dim=-1))
        return out_feat


class GeoIPAEncoder(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.relpos_embedding = nn.Embedding(args.max_relpos * 2 + 2, args.rel_dim)
        self.residue_encoder = PerResidueEncoder(args.feat_dim)
        self.IPA_blocks = nn.ModuleList([
            InvariantPointAttention(
                dim=args.feat_dim,  # single (and pairwise) representation dimension
                heads=8,  # number of attention heads
                scalar_key_dim=16,  # scalar query-key dimension
                scalar_value_dim=16,  # scalar value dimension
                point_key_dim=4,  # point query-key dimension
                point_value_dim=4  # point value dimension
            )
        for _ in range(args.ipa_layer)
        ])
        self.ga_encoder = GAEncoder(
            node_feat_dim=args.feat_dim,
            pair_feat_dim=args.rel_dim,
            num_layers=args.ga_layer,
            spatial_attn_mode='CB',
        )

        self.coefficient = nn.Parameter(torch.ones(3, 2, 21, args.feat_dim), requires_grad=True)

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
    
    def forward(self, complex, mutation_mask):
        pos14, aa, seq, chain = complex['pos14'], complex['aa'], complex['seq'], complex['chain_seq']
        mask_atom = complex['pos14_mask'].all(dim=-1)

        # feature extract
        same_chain = (chain[:, None, :] == chain[:, :, None])
        relpos = (seq[:, None, :] - seq[:, :, None]).clamp(min=-self.args.max_relpos,
                                                           max=self.args.max_relpos) + self.args.max_relpos  # (N, L, L)
        relpos = torch.where(same_chain, relpos, torch.full_like(relpos, fill_value=self.args.max_relpos * 2 + 1))
        pair_feat = self.relpos_embedding(relpos)  # (N, L, L, pair_ch)
        R = construct_3d_basis(pos14[:, :, ATOM_CA], pos14[:, :, ATOM_C], pos14[:, :, ATOM_N])  # (N, L, 3, 3)

        res_feat = self.residue_encoder(aa, pos14, mask_atom)
        
        # add mutation_mask
        res_feat = self.add_mutation_mask(res_feat, mutation_mask)

        t = pos14[:, :, ATOM_CA]
        mask_residue = mask_atom[:, :, ATOM_CA]

        # IPA forward
        res_feat_IPA = res_feat
        for IPA_block in self.IPA_blocks:
            res_feat_IPA = IPA_block(res_feat_IPA, pair_feat, rotations=R, translations=t,
                                     mask=mask_residue)  # Residual connection within the block
        
        # GA forward and feature integration

        res_feat_ga = res_feat
        res_feat_ga_1 = self.ga_encoder.encoder_layer_1(R, t, get_pos_CB(pos14, mask_atom), res_feat_ga, pair_feat, mask_residue)  # (N, L, feat_dim)

        res_feat_mix_1 = self.coefficient[0, 0, aa, :] * res_feat_ga_1 + \
                            self.coefficient[0, 1, aa, :] * res_feat_IPA

        res_feat_ga_2 = self.ga_encoder.encoder_layer_2(R, t, get_pos_CB(pos14, mask_atom), res_feat_mix_1,
                                                        pair_feat, mask_residue)  # (N, L, feat_dim)

        res_feat_mix_2 = self.coefficient[1, 0, aa, :] * res_feat_ga_2 + \
                        self.coefficient[1, 1, aa, :] * res_feat_IPA

        res_feat_ga_3 = self.ga_encoder.encoder_layer_3(R, t, get_pos_CB(pos14, mask_atom), res_feat_mix_2,
                                                        pair_feat, mask_residue)  # (N, L, feat_dim)

        res_feat_ipa_updated = self.coefficient[2, 0, aa, :] * res_feat_ga_3 + \
                        self.coefficient[2, 1, aa, :] * res_feat_IPA

        local_state = res_feat_ga_3
        global_state = res_feat_ipa_updated

        return local_state, global_state