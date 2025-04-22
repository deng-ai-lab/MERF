import argparse


def get_pretrain_args():
    parser = argparse.ArgumentParser()

    # ----------------------- the environment settings ----------------------- #
    parser.add_argument('--seed', type=int, default=170, help='random seed')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--lr', type=float, default=3e-4, help='the initial learning rate')
    parser.add_argument('--is_cuda', type=bool, default=True, help='whether to use cuda')
    parser.add_argument('--gpu_idx', type=int, default=1, help='gpu_idx')
    parser.add_argument('--num_works', type=int, default=30, help='works for loading data')
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epoch for training')

    # ----------------------- network settings ----------------------- #
    # local mutation policy generation module
    parser.add_argument('--feat_dim', type=int, default=128, help='res_feature for encoder input')
    parser.add_argument('--rel_dim', type=int, default=128, help='pair_feature for encoder input')
    parser.add_argument('--max_relpos', type=int, default=32, help='restriction for calculating pair feat')
    parser.add_argument('--ipa_layer', type=int, default=3, help='num of ipa layers')
    parser.add_argument('--ga_layer', type=int, default=3, help='num of ga layers')
    parser.add_argument('--knn_neighbors_num', type=int, default=128, help='number of neighbors for feature extraction')
    parser.add_argument('--knn_agents_num', type=int, default=20, help='number of neighbors for policy making')

    # global mutational effects estimation module
    parser.add_argument('--obs_shape', type=int, default=128, help='obs_shape')
    parser.add_argument('--n_actions', type=int, default=20, help='n_actions')
    parser.add_argument('--n_agents', type=int, default=20, help='residue-type-wise policy network')
    parser.add_argument('--agent_hidden_dim', type=int, default=32, help='rnn_hidden_dim')

    parser.add_argument('--mixing_embed_dim', type=int, default=32, help='mixing_embed_dim, W2')
    parser.add_argument('--global_state_dim', type=int, default=128 * 20, help='obs_shape * 20(n_agents), global_state_dim for hypernet')
    parser.add_argument('--hypernet_embed', type=int, default=64, help='the embedding dim of hypernet, no need for 1 layer')
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

    # ----------------------- network settings ----------------------- #
    # local mutation policy generation module
    parser.add_argument('--feat_dim', type=int, default=128, help='res_feature for encoder input')
    parser.add_argument('--rel_dim', type=int, default=128, help='pair_feature for encoder input')
    parser.add_argument('--max_relpos', type=int, default=32, help='restriction for calculating pair feat')
    parser.add_argument('--ipa_layer', type=int, default=3, help='num of ipa layers')
    parser.add_argument('--ga_layer', type=int, default=3, help='num of ga layers')
    parser.add_argument('--knn_neighbors_num', type=int, default=128, help='number of neighbors for feature extraction')
    parser.add_argument('--knn_agents_num', type=int, default=20, help='number of neighbors for policy making')

    # global mutational effects estimation module
    parser.add_argument('--obs_shape', type=int, default=128, help='obs_shape')
    parser.add_argument('--n_actions', type=int, default=20, help='n_actions')
    parser.add_argument('--n_agents', type=int, default=20, help='residue-type-wise policy network')
    parser.add_argument('--agent_hidden_dim', type=int, default=32, help='rnn_hidden_dim')
    
    parser.add_argument('--mixing_embed_dim', type=int, default=32, help='mixing_embed_dim, W2')
    parser.add_argument('--global_state_dim', type=int, default=128 * 20, help='obs_shape * 20(n_agents), global_state_dim for hypernet')
    parser.add_argument('--hypernet_embed', type=int, default=64, help='the embedding dim of hypernet, no need for 1 layer')
    parser.add_argument('--hypernet_layers', type=int, default=2, help='the layer of hypernet')

    # ----------------------- evolution settings ----------------------- #
    parser.add_argument('--evo_pdb_id', type=str, default='6YZ5_omicron', help='the pdb file to be optimized')
    parser.add_argument('--iter_num', type=int, default=0, help='the pdb file to be optimized')  # for 5 max single, start at 0 end at 4
    parser.add_argument('--task', type=str, default='abbind', help='abbind, sars')

    # evolution settings remake
    # parser.add_argument('--iterations_per_epoch', type=int, default=32, help='the pdb file to be optimized')
    parser.add_argument('--iterations_per_epoch', type=int, default=1, help='the pdb file to be optimized')  # todo: test
    parser.add_argument('--server_id', type=int, default=147, help='when 57 and 147 (128 now), using mpi version')
    parser.add_argument('--training_times', type=int, default=5, help='times for updating using one batch data')
    parser.add_argument('--fix_encoder', type=bool, default=False, help='whether to fix the parameters of encoder in evo')

    # evolution settings rebirth
    parser.add_argument('--comb_num', type=int, default=3, help='numbers of mutations')
    # parser.add_argument('--comb_num', type=int, default=1, help='numbers of mutations')  # todo: fast test
    parser.add_argument('--pdb_index', type=int, default=0, help='which pdb to evo')  # todo: fast test
    parser.add_argument('--pdb_start_index', type=int, default=0, help='which pdb to evo')  # todo: fast test
    parser.add_argument('--pdb_end_index', type=int, default=3, help='which pdb to evo')  # todo: fast test

    args = parser.parse_args()
    return args
