import datetime
import argparse
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_cr_evo_v2 import CREvoDataset
from model.MERF_v6 import MERF
from protein.read_pdbs import PaddingCollate
from utils.losshistory import LossHistory
from utils.util import recursive_to, seed_all


cpu_num = 32
torch.set_num_threads(cpu_num)
print(cpu_num)

CHECKPOINT_MODEL_ARG_KEYS = (
    'seed',
    'use_plm_embedding',
    'plm_path',
    'feat_dim',
    'rel_dim',
    'max_relpos',
    'ipa_layer',
    'ga_layer',
    'knn_neighbors_num',
    'knn_agents_num',
    'obs_shape',
    'n_actions',
    'n_agents',
    'agent_hidden_dim',
    'mixer_rel_dim',
    'mixer_ga_layer',
    'mixing_embed_dim',
    'hypernet_embed',
    'hypernet_layers',
    'use_kl_loss',
    'kl_loss_weight',
)


def get_evolution_args_v2():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=172, help='random seed')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for training')
    parser.add_argument('--lr', type=float, default=5e-5, help='the initial learning rate')
    parser.add_argument('--is_cuda', type=bool, default=True, help='whether to use cuda')
    parser.add_argument('--gpu_idx', type=int, default=1, help='gpu_idx')
    parser.add_argument('--num_works', type=int, default=8, help='works for loading data')
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epoch for training')

    parser.add_argument('--use_plm_embedding', type=bool, default=True, help='whether to use plm embedding as extra feature')
    parser.add_argument('--plm_path', type=str, default="/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/", help='the name of pre-trained language model')

    parser.add_argument('--feat_dim', type=int, default=128, help='res_feature for encoder input')
    parser.add_argument('--rel_dim', type=int, default=128, help='pair_feature for encoder input')
    parser.add_argument('--max_relpos', type=int, default=32, help='restriction for calculating pair feat')
    parser.add_argument('--ipa_layer', type=int, default=3, help='num of ipa layers')
    parser.add_argument('--ga_layer', type=int, default=3, help='num of ga layers')
    parser.add_argument('--knn_neighbors_num', type=int, default=128, help='number of neighbors for feature extraction')
    parser.add_argument('--knn_agents_num', type=int, default=20, help='number of neighbors for policy making')

    parser.add_argument('--model_load_path', type=str, default='/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_06_09_16_56_32/Epoch200_MERF.pth', help='pretrained model path')

    parser.add_argument('--obs_shape', type=int, default=128, help='obs_shape')
    parser.add_argument('--n_actions', type=int, default=20, help='n_actions')
    parser.add_argument('--n_agents', type=int, default=20, help='residue-type-wise policy network')
    parser.add_argument('--agent_hidden_dim', type=int, default=32, help='rnn_hidden_dim')

    parser.add_argument('--mixer_rel_dim', type=int, default=512, help='pair_feature for encoder input')
    parser.add_argument('--mixer_ga_layer', type=int, default=3, help='num of ipa layers')

    parser.add_argument('--mixing_embed_dim', type=int, default=32, help='mixing_embed_dim, W2')
    parser.add_argument('--hypernet_embed', type=int, default=512, help='the embedding dim of hypernet, no need for 1 layer')
    parser.add_argument('--hypernet_layers', type=int, default=2, help='the layer of hypernet')

    parser.add_argument('--dataset_idx', type=int, default=None, help='row index in data/sabdab/sabdab_evo.csv for SAbDab evolution; evolve all rows if unset')
    
    # parser.add_argument('--cr_row_idx', type=int, default=0, help='row index in data/CR/cr_evo.csv for CR evolution')
    # parser.add_argument('--cr_row_idx', type=int, default=1, help='row index in data/CR/cr_evo.csv for CR evolution')
    parser.add_argument('--cr_row_idx', type=int, default=2, help='row index in data/CR/cr_evo.csv for CR evolution')
    # parser.add_argument('--cr_row_idx', type=int, default=3, help='row index in data/CR/cr_evo.csv for CR evolution')

    parser.add_argument('--comb_num', type=int, default=3, help='numbers of mutations')
    parser.add_argument('--training_times', type=int, default=5, help='times for updating using one batch data')

    parser.add_argument('--inner_epochs', type=int, default=10, help='inner epochs for PPO')
    parser.add_argument('--inner_batch_size', type=int, default=64, help='kept for compatibility')
    parser.add_argument('--kl_coeff', type=float, default=0.5, help='KL divergence coefficient')

    parser.add_argument('--total_epochs', type=int, default=3000, help='total CR evolution epochs')
    parser.add_argument('--rollout_samples', type=int, default=32, help='raw per-site action samples per outer epoch')
    parser.add_argument('--sample_temperature', type=float, default=1.0, help='temperature for raw binary action sampling')
    parser.add_argument('--ppo_clip_eps', type=float, default=0.2, help='PPO ratio clip epsilon')
    parser.add_argument('--normalize_advantage', type=bool, default=True, help='normalize sampled advantages within each rollout')
    parser.add_argument('--baseline_momentum', type=float, default=0.9, help='EMA baseline momentum; set <0 to use batch mean only')
    parser.add_argument('--eval_interval', type=int, default=10, help='greedy max-eval interval in outer epochs')

    args = parser.parse_args()
    if args.rollout_samples <= 0:
        raise ValueError('rollout_samples must be positive')
    if args.sample_temperature <= 0:
        raise ValueError('sample_temperature must be positive')
    if args.eval_interval <= 0:
        raise ValueError('eval_interval must be positive')
    return args


def load_model_args_from_checkpoint(args):
    checkpoint_dir = os.path.dirname(os.path.abspath(args.model_load_path))
    args_path = os.path.join(checkpoint_dir, 'args.pkl')

    if not os.path.exists(args_path):
        print(f'Checkpoint args not found at {args_path}; using evolution args for model init.')
        return None, []

    with open(args_path, 'rb') as f:
        checkpoint_args = pickle.load(f)

    if not hasattr(checkpoint_args, '__dict__'):
        raise TypeError(f'Unsupported checkpoint args type in {args_path}: {type(checkpoint_args)}')

    checkpoint_args_dict = vars(checkpoint_args)
    loaded_keys = []
    for key in CHECKPOINT_MODEL_ARG_KEYS:
        if key in checkpoint_args_dict:
            setattr(args, key, checkpoint_args_dict[key])
            loaded_keys.append(key)

    missing_keys = [key for key in CHECKPOINT_MODEL_ARG_KEYS if key not in checkpoint_args_dict]
    print(f'Loaded model args from: {args_path}')
    print(f'Overrode model args: {", ".join(loaded_keys)}')
    if missing_keys:
        print(f'Model args missing in checkpoint args.pkl, kept evolution defaults: {", ".join(missing_keys)}')

    return args_path, loaded_keys


def get_binary_log_probs(model, policy_batch, dataset, device):
    qs = model.choose_best_action(policy_batch, device)
    site_indices, stay_actions, forward_actions = dataset.get_site_action_tensors(policy_batch, device)
    site_qs = qs[0, site_indices]
    stay_qs = site_qs.gather(1, stay_actions.unsqueeze(-1)).squeeze(-1)
    forward_qs = site_qs.gather(1, forward_actions.unsqueeze(-1)).squeeze(-1)
    binary_qs = torch.stack([stay_qs, forward_qs], dim=-1)
    return F.log_softmax(-binary_qs, dim=-1)


def apply_sampling_temperature(binary_log_probs, temperature):
    if temperature == 1.0:
        return binary_log_probs
    return F.log_softmax(binary_log_probs / temperature, dim=-1)


def selected_action_log_probs(binary_log_probs, selected_actions, reduce='sum'):
    if selected_actions.dim() == 1:
        selected_actions = selected_actions.unsqueeze(0)
    selected_actions = selected_actions.to(binary_log_probs.device)
    expanded_log_probs = binary_log_probs.unsqueeze(0).expand(selected_actions.size(0), -1, -1)
    selected = expanded_log_probs.gather(2, selected_actions.unsqueeze(-1)).squeeze(-1)
    if reduce == 'sum':
        return selected.sum(dim=1)
    if reduce == 'mean':
        return selected.mean(dim=1)
    raise ValueError(f"Unsupported reduce mode: {reduce}")


def score_projected_actions(train_dataset, collate_fn, reward_model, projected_actions, device):
    score_batches = [
        train_dataset.getitem_by_action_tensor(projected_actions[i].detach().cpu())
        for i in range(projected_actions.size(0))
    ]
    score_batch = collate_fn(score_batches)
    score_batch = recursive_to(score_batch, device)
    with torch.no_grad():
        model_scores, _, _, _ = reward_model(score_batch, device)
    return model_scores.view(-1)


def summarize_projected_candidates(train_dataset, reverse_keys):
    true_scores = []
    ranks = []
    for reverse_key in reverse_keys:
        eval_info = train_dataset.evaluate_reverse_key(reverse_key)
        true_scores.append(eval_info['true_score'])
        ranks.append(eval_info['rank'])

    valid_scores = [score for score in true_scores if score is not None and not pd.isna(score)]
    valid_ranks = [rank for rank in ranks if rank is not None and not pd.isna(rank)]
    return {
        'true_scores': true_scores,
        'ranks': ranks,
        'best_true_score': float(np.max(valid_scores)) if valid_scores else np.nan,
        'best_rank': int(np.min(valid_ranks)) if valid_ranks else np.nan,
        'mean_rank': float(np.mean(valid_ranks)) if valid_ranks else np.nan,
    }


def train_one_epoch_cr(args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
                       train_dataset, collate_fn, loss_history, device, reward_baseline, valid_ranks,
                       writer=None):

    loss_history.write(f'\nEpoch{epoch}: Start CR v2 raw-action evolution\n')

    policy_batch = train_dataset.__getitem__(0)
    policy_batch = collate_fn([policy_batch])
    policy_batch = recursive_to(policy_batch, device)

    was_training = evo_model.training
    evo_model.eval()
    with torch.no_grad():
        binary_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device)
        sampling_log_probs = apply_sampling_temperature(binary_log_probs, args.sample_temperature)
        raw_action_dist = torch.distributions.Categorical(logits=sampling_log_probs)
        raw_actions = raw_action_dist.sample((args.rollout_samples,))
        projected_actions, reverse_keys, projection_distances, _ = train_dataset.project_raw_actions_to_candidates(
            raw_actions, binary_log_probs=binary_log_probs
        )
        policy_entropy = -(sampling_log_probs.exp() * sampling_log_probs).sum(dim=-1).mean()
    if was_training:
        evo_model.train()

    projected_actions = projected_actions.to(device)
    raw_actions = raw_actions.to(device)

    with torch.no_grad():
        model_scores = score_projected_actions(train_dataset, collate_fn, reward_model, projected_actions, device)
    model_score_values = model_scores.detach()
    batch_baseline = float(model_score_values.mean().item())

    if reward_baseline is None:
        reward_baseline = batch_baseline
    advantage_values = reward_baseline - model_score_values
    if args.normalize_advantage and advantage_values.numel() > 1:
        advantage_std = advantage_values.std(unbiased=False)
        if float(advantage_std.item()) > 1e-8:
            advantage_values = (advantage_values - advantage_values.mean()) / (advantage_std + 1e-8)
    advantages = advantage_values.detach().to(device)

    old_model = MERF(args).to(device)
    old_model.load_state_dict(evo_model.state_dict())
    old_model.eval()

    epoch_losses = []
    policy_losses = []
    inner_kl_values = []
    with torch.no_grad():
        old_log_probs = get_binary_log_probs(old_model, policy_batch, train_dataset, device)
        old_sampling_log_probs = apply_sampling_temperature(old_log_probs, args.sample_temperature)
        old_raw_log_probs = selected_action_log_probs(old_sampling_log_probs, raw_actions, reduce='sum')

    for inner_epoch in tqdm(range(args.inner_epochs), desc='CR PPO Training'):
        optimizer.zero_grad()

        with torch.no_grad():
            ref_log_probs = get_binary_log_probs(reference_model, policy_batch, train_dataset, device)

        curr_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device)
        curr_sampling_log_probs = apply_sampling_temperature(curr_log_probs, args.sample_temperature)
        curr_raw_log_probs = selected_action_log_probs(curr_sampling_log_probs, raw_actions, reduce='sum')

        ratio = torch.exp(curr_raw_log_probs - old_raw_log_probs).clamp(max=20.0)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        kl_div = (ref_log_probs.exp() * (ref_log_probs - curr_log_probs)).sum(dim=-1).mean()
        batch_loss = policy_loss + args.kl_coeff * kl_div

        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(float(batch_loss.item()))
        policy_losses.append(float(policy_loss.item()))
        inner_kl_values.append(float(kl_div.item()))
        loss_history.write(
            f'  Inner epoch {inner_epoch}: loss={batch_loss.item():.6f}, '
            f'policy_loss={policy_loss.item():.6f}, '
            f'adv_mean={advantages.mean().item():.6f}, kl={kl_div.item():.6f}\n'
        )

    if args.baseline_momentum >= 0:
        reward_baseline = args.baseline_momentum * reward_baseline + (1.0 - args.baseline_momentum) * batch_baseline
    else:
        reward_baseline = batch_baseline

    sample_summary = summarize_projected_candidates(train_dataset, reverse_keys)
    best_sample_idx = int(torch.argmin(model_score_values).item())
    train_best_reverse_key = reverse_keys[best_sample_idx]
    train_best_model_score = float(model_score_values[best_sample_idx].item())
    train_best_eval_info = train_dataset.evaluate_reverse_key(train_best_reverse_key)
    train_best_true_score = train_best_eval_info['true_score']
    train_best_rank = train_best_eval_info['rank']
    forward_reverse_info, forward_info = train_dataset.reverse_key_to_forward_info(train_best_reverse_key)

    greedy_reverse_key = None
    greedy_forward_info = None
    greedy_model_score = np.nan
    greedy_true_score = np.nan
    greedy_rank = np.nan
    greedy_policy_score = np.nan
    if epoch % args.eval_interval == 0:
        was_training = evo_model.training
        evo_model.eval()
        with torch.no_grad():
            greedy_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device)
            greedy_actions, greedy_reverse_key, greedy_policy_score_tensor = train_dataset.choose_candidate_from_log_probs(greedy_log_probs)
            greedy_scores = score_projected_actions(
                train_dataset, collate_fn, reward_model, greedy_actions.unsqueeze(0).to(device), device
            )
        if was_training:
            evo_model.train()
        greedy_model_score = float(greedy_scores[0].item())
        greedy_eval_info = train_dataset.evaluate_reverse_key(greedy_reverse_key)
        greedy_true_score = greedy_eval_info['true_score']
        if greedy_true_score == None:
            greedy_true_score = 7.0  # lowest value in dataset
        greedy_rank = greedy_eval_info['rank']
        if greedy_rank == None:
            greedy_rank = max(train_dataset.rank_by_reverse_key.values()) + 1 if train_dataset.rank_by_reverse_key else 1
        greedy_policy_score = float(greedy_policy_score_tensor.item())
        _, greedy_forward_info = train_dataset.reverse_key_to_forward_info(greedy_reverse_key)
        if greedy_rank is not None:
            valid_ranks.append(greedy_rank)

    mean_rank = float(np.mean(valid_ranks)) if valid_ranks else np.nan
    best_rank = int(np.min(valid_ranks)) if valid_ranks else np.nan
    train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    train_policy_loss = float(np.mean(policy_losses)) if policy_losses else 0.0
    train_kl = float(np.mean(inner_kl_values)) if inner_kl_values else 0.0
    projection_distances_cpu = projection_distances.detach().cpu().float()
    exact_match_rate = float((projection_distances_cpu == 0).float().mean().item())
    unique_candidates = len(set(reverse_keys))

    loss_history.write(
        f'Epoch{epoch}: train_best_reverse_key={train_best_reverse_key}, forward_info={forward_info}, '
        f'train_best_model_score={train_best_model_score:.6f}, '
        f'train_best_true_score={train_best_true_score}, train_best_rank={train_best_rank}, '
        f'sample_best_true_score={sample_summary["best_true_score"]}, '
        f'sample_best_rank={sample_summary["best_rank"]}, '
        f'greedy_reverse_key={greedy_reverse_key}, greedy_forward_info={greedy_forward_info}, '
        f'greedy_model_score={greedy_model_score}, greedy_true_score={greedy_true_score}, '
        f'greedy_rank={greedy_rank}, mean_eval_rank={mean_rank}, best_eval_rank={best_rank}, '
        f'loss={train_loss:.6f}\n'
    )

    print('Epoch:' + str(epoch + 1) + '/' + str(total_epoch))
    print(
        'Sample Best Model Score: %.6f, Sample Best Rank: %s, Greedy Rank: %s, Train Loss: %.6f'
        % (train_best_model_score, str(train_best_rank), str(greedy_rank), train_loss)
    )

    # if epoch % 50 == 0:
    #     save_path = os.path.join(loss_history.save_path, 'Epoch%d_evo.pth' % (epoch + 1))
    #     torch.save(evo_model.state_dict(), save_path)

    metrics = {
        'epoch': epoch,
        'train_best_reverse_key': train_best_reverse_key,
        'forward_reverse_ids': forward_reverse_info,
        'forward_mutations': forward_info,
        'model_score': greedy_model_score if greedy_reverse_key is not None else train_best_model_score,
        'reward_baseline': reward_baseline,
        'advantage_mean': float(advantages.mean().item()),
        'advantage_std': float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
        'advantage_min': float(advantages.min().item()),
        'advantage_max': float(advantages.max().item()),
        'true_score': greedy_true_score if greedy_reverse_key is not None else train_best_true_score,
        'rank': greedy_rank if greedy_reverse_key is not None else train_best_rank,
        'num_ranked': len(train_dataset.rank_by_reverse_key),
        'mean_eval_rank': mean_rank,
        'best_eval_rank': best_rank,
        'policy_score': greedy_policy_score,
        'train_best_model_score': train_best_model_score,
        'train_best_true_score': train_best_true_score,
        'train_best_rank': train_best_rank,
        'train_sample_model_score_mean': float(model_score_values.mean().item()),
        'train_sample_model_score_std': float(model_score_values.std(unbiased=False).item()) if model_score_values.numel() > 1 else 0.0,
        'train_sample_model_score_min': float(model_score_values.min().item()),
        'train_sample_model_score_max': float(model_score_values.max().item()),
        'train_sample_best_true_score': sample_summary['best_true_score'],
        'train_sample_best_rank': sample_summary['best_rank'],
        'train_sample_mean_rank': sample_summary['mean_rank'],
        'train_sample_unique_candidates': unique_candidates,
        'projection_hamming_mean': float(projection_distances_cpu.mean().item()),
        'projection_hamming_max': float(projection_distances_cpu.max().item()),
        'projection_exact_match_rate': exact_match_rate,
        'raw_policy_entropy': float(policy_entropy.item()),
        'greedy_reverse_key': greedy_reverse_key,
        'greedy_forward_mutations': greedy_forward_info,
        'greedy_model_score': greedy_model_score,
        'greedy_true_score': greedy_true_score,
        'greedy_rank': greedy_rank,
        'train_loss': train_loss,
        'train_policy_loss': train_policy_loss,
        'train_kl': train_kl,
    }

    if writer is not None:
        writer.add_scalar('Sample/best_model_score', train_best_model_score, epoch + 1)
        writer.add_scalar('Sample/best_true_score', sample_summary['best_true_score'], epoch + 1)
        writer.add_scalar('Sample/best_rank', sample_summary['best_rank'], epoch + 1)
        writer.add_scalar('Sample/unique_candidates', unique_candidates, epoch + 1)
        writer.add_scalar('Sample/model_score_mean', float(model_score_values.mean().item()), epoch + 1)
        writer.add_scalar('Sample/model_score_std', metrics['train_sample_model_score_std'], epoch + 1)
        writer.add_scalar('Projection/hamming_mean', metrics['projection_hamming_mean'], epoch + 1)
        writer.add_scalar('Projection/hamming_max', metrics['projection_hamming_max'], epoch + 1)
        writer.add_scalar('Projection/exact_match_rate', exact_match_rate, epoch + 1)
        writer.add_scalar('Policy/raw_entropy', float(policy_entropy.item()), epoch + 1)
        writer.add_scalar('Policy/advantage_mean', metrics['advantage_mean'], epoch + 1)
        writer.add_scalar('Policy/advantage_std', metrics['advantage_std'], epoch + 1)
        writer.add_scalar('Evolution/reward_baseline', reward_baseline, epoch + 1)
        writer.add_scalar('Eval/greedy_rank', greedy_rank, epoch + 1)
        writer.add_scalar('Eval/greedy_true_score', greedy_true_score, epoch + 1)
        writer.add_scalar('Eval/greedy_model_score', greedy_model_score, epoch + 1)
        writer.add_scalar('Eval/mean_rank', mean_rank, epoch + 1)
        writer.add_scalar('Eval/best_rank', best_rank, epoch + 1)
        writer.add_scalar('Eval/greedy_policy_score', greedy_policy_score, epoch + 1)
        writer.add_scalar('Train/loss', train_loss, epoch + 1)
        writer.add_scalar('Train/policy_loss', train_policy_loss, epoch + 1)
        writer.add_scalar('Train/kl', train_kl, epoch + 1)
        writer.flush()

    save_file = os.path.join(loss_history.save_path, train_dataset.pdb_id + '.csv')
    new_row = pd.DataFrame(metrics, index=[0])
    if os.path.exists(save_file):
        save_df = pd.read_csv(save_file)
        save_df = pd.concat([save_df, new_row], ignore_index=True)
    else:
        save_df = new_row
    save_df.to_csv(save_file, index=False)

    return reward_baseline, valid_ranks, metrics


if __name__ == '__main__':
    args = get_evolution_args_v2()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)

    seed = args.seed
    seed_all(seed)
    print(f"setting random seed...{seed}")

    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")

    loss_dir = "logs/evo_cr_v2/"
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    loss_dir = loss_dir + time_str + '/'
    print(f'Loss and model checkpoints will be saved to {loss_dir}')

    cr_dir = '/home/dataset-local/projects_dir/MERF/data/CR'
    train_path = os.path.join(cr_dir, 'cr_evo.csv')
    train_df = pd.read_csv(train_path, dtype={"pdb_id": "string"})
    pdb_dir = os.path.join(cr_dir, 'PDBs')
    fixed_dir = os.path.join(cr_dir, 'PDBs_fixed')
    mutated_dir = os.path.join(cr_dir, 'PDBs_mutated')

    if args.cr_row_idx < 0 or args.cr_row_idx >= len(train_df):
        raise IndexError(f'cr_row_idx {args.cr_row_idx} out of range [0, {len(train_df) - 1}]')

    i = args.cr_row_idx
    pdb_id = train_df['pdb_id'].iloc[i].replace('+', '').replace('.00', '')
    antibody_chain = train_df['antibody_chain'].iloc[i]
    partner = train_df['partner'].iloc[i]
    sequence = train_df['sequence'].iloc[i]
    cdrs = [train_df['cdr1'].iloc[i], train_df['cdr2'].iloc[i], train_df['cdr3'].iloc[i]]
    cdr = cdrs[2]

    print(f'Start evolving row {i}: {pdb_id}...')
    print(f'Antibody chain: {antibody_chain}, Partner: {partner}, CDRH3: {cdr}')
    print(f'Sequence length: {len(sequence)}; sequence: {sequence}')
    print('----------------------------------------')

    plm_embedding_path = os.path.join(cr_dir, f'{pdb_id}_immature_esm2_650_embeddings.pkl')
    train_dataset = CREvoDataset(
        pdb_id, antibody_chain, partner, sequence, cdr,
        cr_dir, pdb_dir, fixed_dir, mutated_dir,
        knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
        plm_embedding_path=plm_embedding_path, device=device,
    )
    collate_fn = PaddingCollate()

    this_loss_dir = loss_dir + pdb_id + '/'
    loss_history = LossHistory(this_loss_dir, is_evolve=True)
    writer = SummaryWriter(log_dir=loss_history.save_path)
    loss_history.write(str(args) + '\n')
    if checkpoint_args_path is not None:
        loss_history.write(f'Loaded model args from {checkpoint_args_path}\n')
        loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')
    loss_history.write(f'CR row index: {i}\n')
    loss_history.write(f'CR immature PLM embedding path: {plm_embedding_path}\n')
    loss_history.write(f'CR candidates: {len(train_dataset.candidate_reverse_keys)}\n')
    loss_history.write(f'CR ranked candidates: {len(train_dataset.rank_by_reverse_key)}\n')
    loss_history.write(
        f'CR v2 raw rollout samples: {args.rollout_samples}, '
        f'sample_temperature: {args.sample_temperature}, eval_interval: {args.eval_interval}\n'
    )

    evo_model = MERF(args).to(device)
    evo_model.load_state_dict(torch.load(args.model_load_path, map_location=device))
    optimizer = optim.Adam(evo_model.parameters(), lr=args.lr, weight_decay=0.1)

    reference_model = MERF(args).to(device)
    reference_model.load_state_dict(evo_model.state_dict())
    reference_model.eval()

    reward_model = MERF(args).to(device)
    reward_model.load_state_dict(torch.load(args.model_load_path, map_location=device))
    reward_model.eval()

    loss_history.write(f'\nLoading model from {args.model_load_path}\n')

    total_epoch = args.total_epochs
    reward_baseline = None
    valid_ranks = []
    for epoch in range(total_epoch):
        reward_baseline, valid_ranks, _ = train_one_epoch_cr(
            args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
            train_dataset, collate_fn, loss_history, device, reward_baseline, valid_ranks,
            writer=writer,
        )

    writer.close()
