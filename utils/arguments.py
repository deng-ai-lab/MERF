import argparse

def get_pretrain_args():
    parser = argparse.ArgumentParser()

    # ----------------------- the environment settings ----------------------- #
    # parser.add_argument('--seed', type=int, default=173, help='random seed')
    parser.add_argument('--seed', type=int, default=170, help='random seed')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, help='the initial learning rate')
    parser.add_argument('--is_cuda', type=bool, default=True, help='whether to use cuda')
    parser.add_argument('--gpu_idx', type=int, default=0, help='gpu_idx')
    parser.add_argument('--num_works', type=int, default=16, help='works for loading data')
    parser.add_argument('--n_epoch', type=int, default=200, help='the number of epoch for training')

    # ----------------------- learning rate scheduler settings ----------------------- #
    parser.add_argument('--use_cosine_lr', type=bool, default=True, help='whether to use cosine annealing learning rate scheduler')
    parser.add_argument('--cosine_warmup_epochs', type=int, default=5, help='number of warmup epochs for cosine scheduler')
    parser.add_argument('--cosine_min_lr', type=float, default=1e-6, help='minimum learning rate for cosine scheduler')

    # ----------------------- dataset settings ----------------------- #
    parser.add_argument('--use_plm_embedding', type=bool, default=True, help='whether to use plm embedding as extra feature')
    parser.add_argument('--plm_path', type=str, default="/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/", help='the name of pre-trained language model')
    parser.add_argument('--use_weighted_sampling', type=bool, default=True, help='whether to use weighted sampling for balanced training')
    parser.add_argument('--vm_sampling_weight', type=float, default=15.0, help='relative sampling weight for VM dataset')
    parser.add_argument('--sk_sampling_weight', type=float, default=1.0, help='relative sampling weight for SKEMPIv2 dataset')
    parser.add_argument('--max_repetition_factor', type=float, default=2.0, help='max repetition factor for smaller dataset (0 means no limit)')

    # ----------------------- KL distribution loss settings ----------------------- #
    parser.add_argument('--use_kl_loss', type=bool, default=True, help='whether to use KL divergence loss for site mutation distribution')
    parser.add_argument('--kl_loss_weight', type=float, default=3.5, help='weight for KL divergence loss')

    # ----------------------- Ranking loss settings (v2) ----------------------- #
    parser.add_argument('--use_rank_loss', type=bool, default=True, help='whether to use pairwise RankNet loss for site mutation ranking')
    parser.add_argument('--rank_loss_weight', type=float, default=0.5, help='weight for pairwise RankNet loss')
    parser.add_argument('--rank_tie_threshold', type=float, default=0.0, help='ignore mutation pairs whose ddG difference is below this threshold')

    # ----------------------- network settings ----------------------- #
    # local mutation policy generation module
    parser.add_argument('--feat_dim', type=int, default=128, help='res_feature for encoder input')
    parser.add_argument('--rel_dim', type=int, default=128, help='pair_feature for encoder input')
    # parser.add_argument('--feat_dim', type=int, default=256, help='res_feature for encoder input')
    # parser.add_argument('--rel_dim', type=int, default=256, help='pair_feature for encoder input')
    parser.add_argument('--max_relpos', type=int, default=32, help='restriction for calculating pair feat')
    parser.add_argument('--ipa_layer', type=int, default=3, help='num of ipa layers')
    parser.add_argument('--ga_layer', type=int, default=3, help='num of ga layers')
    # parser.add_argument('--ipa_layer', type=int, default=4, help='num of ipa layers')
    # parser.add_argument('--ga_layer', type=int, default=4, help='num of ga layers')
    parser.add_argument('--knn_neighbors_num', type=int, default=128, help='number of neighbors for feature extraction')
    parser.add_argument('--knn_agents_num', type=int, default=20, help='number of neighbors for policy making')

    # global mutational effects estimation module
    parser.add_argument('--obs_shape', type=int, default=128, help='obs_shape')
    # parser.add_argument('--obs_shape', type=int, default=256, help='obs_shape')
    parser.add_argument('--n_actions', type=int, default=20, help='n_actions')
    parser.add_argument('--n_agents', type=int, default=20, help='residue-type-wise policy network')
    parser.add_argument('--agent_hidden_dim', type=int, default=32, help='rnn_hidden_dim')
    # parser.add_argument('--agent_hidden_dim', type=int, default=128, help='rnn_hidden_dim')

    parser.add_argument('--mixer_rel_dim', type=int, default=512, help='pair_feature for encoder input')
    parser.add_argument('--mixer_ga_layer', type=int, default=3, help='num of ipa layers')
    # parser.add_argument('--mixer_rel_dim', type=int, default=768, help='pair_feature for encoder input')
    # parser.add_argument('--mixer_ga_layer', type=int, default=4, help='num of ipa layers')

    parser.add_argument('--mixing_embed_dim', type=int, default=32, help='对20个qs的输入来说的中间层')
    # parser.add_argument('--mixing_embed_dim', type=int, default=64, help='对20个qs的输入来说的中间层')
    parser.add_argument('--hypernet_embed', type=int, default=512, help='the embedding dim of hypernet, no need for 1 layer')
    # parser.add_argument('--hypernet_embed', type=int, default=1024, help='the embedding dim of hypernet, no need for 1 layer')
    parser.add_argument('--hypernet_layers', type=int, default=2, help='the layer of hypernet')

    args = parser.parse_args()
    return args


def get_evolution_args():
    parser = argparse.ArgumentParser()

    # ----------------------- the environment settings ----------------------- #
    parser.add_argument('--seed', type=int, default=170, help='random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='the initial learning rate')
    parser.add_argument('--is_cuda', type=bool, default=True, help='whether to use cuda')
    parser.add_argument('--gpu_idx', type=int, default=1, help='gpu_idx')
    parser.add_argument('--num_works', type=int, default=8, help='works for loading data')
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epoch for training')

    # ----------------------- dataset settings ----------------------- #
    parser.add_argument('--use_plm_embedding', type=bool, default=True, help='whether to use plm embedding as extra feature')
    parser.add_argument('--plm_path', type=str, default="/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/", help='the name of pre-trained language model')

    # ----------------------- network settings ----------------------- #
    # local mutation policy generation module
    parser.add_argument('--feat_dim', type=int, default=128, help='res_feature for encoder input')
    parser.add_argument('--rel_dim', type=int, default=128, help='pair_feature for encoder input')
    parser.add_argument('--max_relpos', type=int, default=32, help='restriction for calculating pair feat')
    parser.add_argument('--ipa_layer', type=int, default=3, help='num of ipa layers')
    parser.add_argument('--ga_layer', type=int, default=3, help='num of ga layers')
    parser.add_argument('--knn_neighbors_num', type=int, default=128, help='number of neighbors for feature extraction')
    parser.add_argument('--knn_agents_num', type=int, default=20, help='number of neighbors for policy making')

    parser.add_argument('--model_load_path', type=str, default='model/MERF_pretrained.pth', help='pretrained model path')

    # global mutational effects estimation module
    parser.add_argument('--obs_shape', type=int, default=128, help='obs_shape')
    parser.add_argument('--n_actions', type=int, default=20, help='n_actions')
    parser.add_argument('--n_agents', type=int, default=20, help='residue-type-wise policy network')
    parser.add_argument('--agent_hidden_dim', type=int, default=32, help='rnn_hidden_dim')
    
    parser.add_argument('--mixer_rel_dim', type=int, default=512, help='pair_feature for encoder input')
    parser.add_argument('--mixer_ga_layer', type=int, default=3, help='num of ipa layers')

    parser.add_argument('--mixing_embed_dim', type=int, default=32, help='mixing_embed_dim, W2')
    parser.add_argument('--hypernet_embed', type=int, default=512, help='the embedding dim of hypernet, no need for 1 layer')
    parser.add_argument('--hypernet_layers', type=int, default=2, help='the layer of hypernet')

    # ----------------------- evolution settings ----------------------- #
    parser.add_argument('--comb_num', type=int, default=3, help='numbers of mutations')
    parser.add_argument('--training_times', type=int, default=5, help='times for updating using one batch data')

    # ----------------------- GRPO settings ----------------------- #
    parser.add_argument('--inner_epochs', type=int, default=5, help='inner epochs for GRPO')
    parser.add_argument('--inner_batch_size', type=int, default=64, help='inner batch size for GRPO')
    parser.add_argument('--kl_coeff', type=float, default=0.5, help='KL divergence coefficient')

    args = parser.parse_args()
    return args
