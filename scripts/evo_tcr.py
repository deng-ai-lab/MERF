"""
Selective TCR evolution script.

Policy objective:
  - Maximise affinity towards the target complex  (subtract ddG predictions → more negative = better)
  - Minimise affinity towards each offtarget complex (add ddG predictions → want high/positive)

Concretely, the combined action Q-values are:
    Q_combined = Q_target - sum(Q_offtarget_i)

The reward signal mirrors this: lower target ddG is good, higher offtarget ddG is good, so:
    score = target_score - sum(offtarget_scores)
and we optimise to *minimise* this score (lower = better binding differential).
"""

import os
import sys
import datetime
import pickle
from itertools import combinations

import numpy as np
import pandas as pd

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.MERF_v6 import MERF
from dataset.dataset_tcr_evo import EvoTCRDataset
from utils.arguments import get_evolution_args
from utils.util import recursive_to, seed_all
from utils.losshistory import LossHistory
from protein.read_pdbs import PaddingCollate

cpu_num = 32
torch.set_num_threads(cpu_num)
print(cpu_num)

# ---------------------------------------------------------------------------
# Default offtarget peptides (overridable via --offtarget_list argument)
# ---------------------------------------------------------------------------
DEFAULT_OFFTARGET_LIST = ['FLDLGPPGI', 'VMAEAPPGV', 'MTDKAPPGV']

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


# ---------------------------------------------------------------------------
# Argument loading helpers
# ---------------------------------------------------------------------------

def get_tcr_evolution_args():
    """Extend the standard evolution args with TCR-specific flags."""
    import argparse

    parser = argparse.ArgumentParser(description='Selective TCR evolution')

    # ---------- copied from get_evolution_args ----------
    parser.add_argument('--seed', type=int, default=172)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--is_cuda', type=bool, default=True)
    parser.add_argument('--gpu_idx', type=int, default=0)
    parser.add_argument('--num_works', type=int, default=8)
    parser.add_argument('--n_epoch', type=int, default=30)

    parser.add_argument('--use_plm_embedding', type=bool, default=True)
    parser.add_argument(
        '--plm_path', type=str,
        default='/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/',
    )

    parser.add_argument('--feat_dim', type=int, default=128)
    parser.add_argument('--rel_dim', type=int, default=128)
    parser.add_argument('--max_relpos', type=int, default=32)
    parser.add_argument('--ipa_layer', type=int, default=3)
    parser.add_argument('--ga_layer', type=int, default=3)
    parser.add_argument('--knn_neighbors_num', type=int, default=128)
    parser.add_argument('--knn_agents_num', type=int, default=20)

    parser.add_argument(
        '--model_load_path', type=str,
        default='/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_06_09_16_56_32/Epoch200_MERF.pth',
    )

    parser.add_argument('--obs_shape', type=int, default=128)
    parser.add_argument('--n_actions', type=int, default=20)
    parser.add_argument('--n_agents', type=int, default=20)
    parser.add_argument('--agent_hidden_dim', type=int, default=32)

    parser.add_argument('--mixer_rel_dim', type=int, default=512)
    parser.add_argument('--mixer_ga_layer', type=int, default=3)
    parser.add_argument('--mixing_embed_dim', type=int, default=32)
    parser.add_argument('--hypernet_embed', type=int, default=512)
    parser.add_argument('--hypernet_layers', type=int, default=2)

    parser.add_argument('--dataset_idx', type=int, default=None)
    parser.add_argument('--comb_num', type=int, default=3)
    parser.add_argument('--training_times', type=int, default=5)

    parser.add_argument('--inner_epochs', type=int, default=10)
    parser.add_argument('--inner_batch_size', type=int, default=64)
    parser.add_argument('--kl_coeff', type=float, default=0.5)

    # ---------- TCR-specific ----------
    parser.add_argument(
        '--offtarget_list', type=str, nargs='+',
        default=DEFAULT_OFFTARGET_LIST,
        help='List of offtarget peptide names (must match PDB filenames in PDBs_offtarget/, case-insensitive)',
    )

    args = parser.parse_args()
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
        raise TypeError(f'Unsupported checkpoint args type: {type(checkpoint_args)}')

    checkpoint_args_dict = vars(checkpoint_args)
    loaded_keys = []
    for key in CHECKPOINT_MODEL_ARG_KEYS:
        if key in checkpoint_args_dict:
            setattr(args, key, checkpoint_args_dict[key])
            loaded_keys.append(key)

    missing_keys = [k for k in CHECKPOINT_MODEL_ARG_KEYS if k not in checkpoint_args_dict]
    print(f'Loaded model args from: {args_path}')
    print(f'Overrode model args: {", ".join(loaded_keys)}')
    if missing_keys:
        print(f'Missing in checkpoint (kept defaults): {", ".join(missing_keys)}')

    return args_path, loaded_keys


# ---------------------------------------------------------------------------
# Core computation helpers
# ---------------------------------------------------------------------------

def compute_selective_qs(model, train_dataset, collate_fn, positions, device):
    """
    Return combined Q-values for action selection:
        Q_combined = Q_target - sum(Q_offtarget_i)

    Shape: (num_positions, n_actions)
    """
    # --- target ---
    batch = train_dataset.get_policy_item(positions)
    batch = collate_fn([batch])
    batch = recursive_to(batch, device)

    with torch.no_grad():
        qs_target = model.choose_best_action(batch, device)

    combined_qs = qs_target[batch['mutation_mask']]  # (num_pos, n_actions)

    # --- offtargets: subtract ---
    for offtarget_name in train_dataset.offtarget_names:
        ot_batch = train_dataset.get_offtarget_policy_item(offtarget_name, positions)
        ot_batch = collate_fn([ot_batch])
        ot_batch = recursive_to(ot_batch, device)

        with torch.no_grad():
            qs_ot = model.choose_best_action(ot_batch, device)

        combined_qs = combined_qs - qs_ot[ot_batch['mutation_mask']]

    return combined_qs


def aggregate_selective_log_probs(model, train_dataset, collate_fn, mutate_info_list, device):
    """
    Compute per-agent log-softmax over the *selective* combined Q-values.

    Q_combined = Q_target - sum(Q_offtarget_i)
    log_probs  = log_softmax(-Q_combined)   (lower combined Q → higher probability)

    Returns: (batch_size, n_agents, n_actions) log-probabilities.
    """
    batch_size = len(mutate_info_list)

    # target
    target_batches = [train_dataset.getitem_by_mutate_info(mi) for mi in mutate_info_list]
    target_batch = collate_fn(target_batches)
    target_batch = recursive_to(target_batch, device)
    qs_target = model.choose_best_action(target_batch, device)
    combined_qs = qs_target[target_batch['wt']['agent_mask']].view(batch_size, -1, qs_target.size(-1))

    # offtargets
    for offtarget_name in train_dataset.offtarget_names:
        ot_batches = [train_dataset.getitem_offtarget_by_mutate_info(offtarget_name, mi) for mi in mutate_info_list]
        ot_batch = collate_fn(ot_batches)
        ot_batch = recursive_to(ot_batch, device)
        qs_ot = model.choose_best_action(ot_batch, device)
        qs_ot_agents = qs_ot[ot_batch['wt']['agent_mask']].view(batch_size, -1, qs_ot.size(-1))
        combined_qs = combined_qs - qs_ot_agents

    return F.log_softmax(-combined_qs, dim=-1)


def score_mutation_selective(reward_model, train_dataset, collate_fn, mutate_info, device):
    """
    Compute the selective reward score:
        score = target_score - sum(offtarget_scores)

    Lower is better (model predicts ddG; negative ddG = better binding).
    For offtargets, we *subtract* their score so that improved offtarget
    binding raises the score (penalises the mutation).
    """
    scores = {}

    # target
    batch = train_dataset.getitem_by_mutate_info(mutate_info)
    batch = collate_fn([batch])
    batch = recursive_to(batch, device)
    with torch.no_grad():
        q_tot, _, _, _ = reward_model(batch, device)
    target_score = q_tot.item()
    scores['target'] = target_score

    # offtargets
    offtarget_total = 0.0
    for offtarget_name in train_dataset.offtarget_names:
        ot_batch = train_dataset.getitem_offtarget_by_mutate_info(offtarget_name, mutate_info)
        ot_batch = collate_fn([ot_batch])
        ot_batch = recursive_to(ot_batch, device)
        with torch.no_grad():
            ot_q, _, _, _ = reward_model(ot_batch, device)
        ot_score = ot_q.item()
        scores[offtarget_name] = ot_score
        offtarget_total += ot_score

    combined_score = target_score - offtarget_total
    return combined_score, scores


def update_candidate_scores(best_candidate_scores, epoch_candidate_scores):
    for candidate in epoch_candidate_scores:
        mutate_info = candidate['mutate_info']
        prev = best_candidate_scores.get(mutate_info)
        if prev is None or candidate['reward_score'] < prev['reward_score']:
            best_candidate_scores[mutate_info] = candidate


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
        args, evo_model, reward_model, reference_model, optimizer,
        epoch, total_epoch, train_dataset, collate_fn, loss_history, device,
        tb_writer=None,
):
    print('Start Train')
    loss_history.write(f'\nEpoch{epoch}: Start Train\n')

    pdb_id = train_dataset.pdb_id
    antibody_chain = train_dataset.antibody_chain
    cdr_entries = train_dataset.get_cdr_entries()
    mutate_entry_combinations = list(combinations(cdr_entries, args.comb_num))

    loss_history.write(f'\nEpoch{epoch}: 1. Agent interaction\n')
    loss_history.write(f'Offtarget list: {train_dataset.offtarget_names}\n')

    aa_key = 'ACDEFGHIKLMNPQRSTVWY'
    mutate_info_list = []
    seen_mutate_infos = set()

    for mutate_entries in tqdm(mutate_entry_combinations, desc='Generating TCR mutation actions'):
        positions = [entry['position'] for entry in mutate_entries]
        wt_states = [entry['aa'] for entry in mutate_entries]

        combined_qs = compute_selective_qs(evo_model, train_dataset, collate_fn, positions, device)
        action_dist = torch.distributions.Categorical(logits=-combined_qs)
        actions = action_dist.sample()

        this_mutate_info_list = []
        has_mutation = False
        for state, action_tensor, position in zip(wt_states, actions, positions):
            action = action_tensor.item()
            this_mutate_info_list.append(f'{aa_key[state]}{antibody_chain}{position}{aa_key[action]}')
            if state != action:
                has_mutation = True

        mutate_info = ','.join(this_mutate_info_list)
        if has_mutation and mutate_info not in seen_mutate_infos:
            mutate_info_list.append(mutate_info)
            seen_mutate_infos.add(mutate_info)

    loss_history.write(f'\nEpoch{epoch}: 1. Finish interaction, candidates={len(mutate_info_list)}\n')
    if len(mutate_info_list) == 0:
        loss_history.write('No valid mutation candidates; skipping epoch.\n')
        return {
            'best_score': np.nan,
            'avg_score': np.nan,
            'train_loss': 0.0,
            'candidate_scores': [],
        }

    loss_history.write(f'\nEpoch{epoch}: 2. Mutation\n')
    train_dataset.mutate_pdb(mutate_info_list)
    loss_history.write(f'\nEpoch{epoch}: 2. Finish mutation\n')

    loss_history.write(f'\nEpoch{epoch}: 3. Scoring\n')
    score_list = []
    detail_score_list = []

    all_score_keys = ['target'] + train_dataset.offtarget_names

    for mutate_info in mutate_info_list:
        combined_score, scores = score_mutation_selective(reward_model, train_dataset, collate_fn, mutate_info, device)
        score_list.append(combined_score)
        detail_score_list.append(scores)
        loss_history.write(
            mutate_info
            + '\tcombined=' + str(combined_score)
            + '\t' + '\t'.join(f'{k}={scores[k]}' for k in all_score_keys)
            + '\n'
        )

    loss_history.write(f'\nEpoch{epoch}: 3. Finish Scoring\n')
    if tb_writer is not None:
        tb_writer.add_scalar('evo/num_candidates', len(score_list), epoch)

    loss_history.write(f'\nEpoch{epoch}: 4. GRPO Training\n')

    old_model = MERF(args).to(device)
    old_model.load_state_dict(evo_model.state_dict())
    old_model.eval()

    score_array = np.array(score_list)
    baseline = score_array.mean()
    advantages = baseline - score_array

    advantage_std = np.abs(advantages).std()
    if advantage_std > 1e-8:
        advantages = advantages / (advantage_std + 1e-8)

    n_samples = len(mutate_info_list)
    epoch_losses = []

    for inner_epoch in tqdm(range(args.inner_epochs), desc='GRPO Training'):
        sample_indices = np.random.choice(n_samples, size=min(args.inner_batch_size, n_samples), replace=False)
        sample_mutate_infos = [mutate_info_list[idx] for idx in sample_indices]

        optimizer.zero_grad()

        with torch.no_grad():
            ref_logits = aggregate_selective_log_probs(reference_model, train_dataset, collate_fn, sample_mutate_infos, device)
            old_logits = aggregate_selective_log_probs(old_model, train_dataset, collate_fn, sample_mutate_infos, device)

        curr_logits = aggregate_selective_log_probs(evo_model, train_dataset, collate_fn, sample_mutate_infos, device)

        log_ratio = curr_logits - old_logits
        ratio = torch.exp(log_ratio).clamp(max=20.0)

        adv_tensor = torch.tensor(
            [advantages[idx] for idx in sample_indices], dtype=torch.float32, device=device
        )
        adv_tensor = adv_tensor.unsqueeze(-1).unsqueeze(-1).expand_as(ratio)

        policy_loss = -(ratio * adv_tensor).mean()
        kl_div = (ref_logits.exp() * (ref_logits - curr_logits)).sum(dim=-1).mean()
        batch_loss = policy_loss + args.kl_coeff * kl_div

        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(batch_loss.item())
        if tb_writer is not None:
            inner_step = epoch * args.inner_epochs + inner_epoch
            tb_writer.add_scalar('train/policy_loss', policy_loss.item(), inner_step)
            tb_writer.add_scalar('train/kl_div', kl_div.item(), inner_step)
            tb_writer.add_scalar('train/batch_loss', batch_loss.item(), inner_step)
        loss_history.write(f'  Inner epoch {inner_epoch}: batch_loss={batch_loss.item():.6f}\n')

    loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0

    best_score = float(np.min(score_array))
    avg_score = float(np.mean(score_array))
    std_score = float(np.std(score_array))
    max_score = float(np.max(score_array))

    if tb_writer is not None:
        tb_writer.add_scalar('reward/best_combined_score', best_score, epoch)
        tb_writer.add_scalar('reward/avg_combined_score', avg_score, epoch)
        tb_writer.add_scalar('reward/std_combined_score', std_score, epoch)
        tb_writer.add_scalar('train/loss', loss, epoch)
        tb_writer.add_histogram('reward/combined_scores', score_array, epoch)

    loss_history.write(
        f'Epoch{epoch}: Best={best_score:.6f}, Avg={avg_score:.6f}, Loss={loss:.6f}\n'
    )
    print('Finish Train')
    print(f'Epoch: {epoch + 1}/{total_epoch}')
    print(f'Best combined score: {best_score:.6f}, Avg: {avg_score:.6f}, Loss: {loss:.6f}')

    # save per-epoch CSV
    save_file = os.path.join(loss_history.save_path, pdb_id + '.csv')
    new_row = pd.DataFrame({
        'epoch': [epoch],
        'best_score': [best_score],
        'avg_score': [avg_score],
        'std_score': [std_score],
        'worst_score': [max_score],
        'train_loss': [loss],
    })
    if not os.path.exists(save_file):
        new_row.to_csv(save_file, index=False)
    else:
        save_df = pd.read_csv(save_file)
        save_df = pd.concat([save_df, new_row], ignore_index=True)
        save_df.to_csv(save_file, index=False)

    epoch_candidate_scores = []
    for mutate_info, score, detail_scores in zip(mutate_info_list, score_list, detail_score_list):
        candidate = {'epoch': epoch, 'mutate_info': mutate_info, 'reward_score': score}
        for k in all_score_keys:
            candidate[f'{k}_score'] = detail_scores[k]
        epoch_candidate_scores.append(candidate)

    return {
        'best_score': best_score,
        'avg_score': avg_score,
        'train_loss': loss,
        'candidate_scores': epoch_candidate_scores,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    args = get_tcr_evolution_args()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)

    seed_all(args.seed)
    print(f'Setting random seed... {args.seed}')

    device = torch.device(f'cuda:{args.gpu_idx}' if args.is_cuda else 'cpu')

    loss_dir = 'logs/evo_tcr/'
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    loss_dir = loss_dir + time_str + '/'

    tcr_dir = '/home/dataset-local/projects_dir/MERF/data/TCR'
    train_path = os.path.join(tcr_dir, 'f8_evo.csv')
    train_df = pd.read_csv(train_path, dtype={'pdb_id': 'string'})

    wt_dir = os.path.join(tcr_dir, 'PDBs')
    fix_dir = os.path.join(tcr_dir, 'PDBs_fixed')
    mut_dir = os.path.join(tcr_dir, 'PDBs_mutated')

    offtarget_wt_dir = os.path.join(tcr_dir, 'PDBs_offtarget')
    offtarget_fix_dir = os.path.join(tcr_dir, 'PDBs_offtarget_fixed')
    offtarget_mut_dir = os.path.join(tcr_dir, 'PDBs_offtarget_mutated')

    plm_embedding_path = os.path.join(tcr_dir, 'PLM_embeddings_tcr_evo.pkl')

    os.makedirs(fix_dir, exist_ok=True)
    os.makedirs(mut_dir, exist_ok=True)
    os.makedirs(offtarget_fix_dir, exist_ok=True)
    os.makedirs(offtarget_mut_dir, exist_ok=True)

    if args.dataset_idx is not None:
        if args.dataset_idx < 0 or args.dataset_idx >= len(train_df):
            raise IndexError(
                f'dataset_idx {args.dataset_idx} out of range for {train_path} with {len(train_df)} rows'
            )
        dataset_indices = [args.dataset_idx]
        print(f'Only evolving dataset row {args.dataset_idx}.')
    else:
        dataset_indices = range(len(train_df))

    for i in dataset_indices:
        pdb_id = str(train_df['pdb_id'].iloc[i])
        antibody_chain = train_df['antibody_chain'].iloc[i]
        partner = train_df['partner'].iloc[i]
        sequence = train_df['sequence'].iloc[i]
        cdr = train_df['cdr3'].iloc[i]

        print(f'Start evolving {pdb_id}...')
        print(f'Antibody chain: {antibody_chain}, CDR3: {cdr}')
        print(f'Offtarget list: {args.offtarget_list}')
        print('----------------------------------------')

        train_dataset = EvoTCRDataset(
            pdb_id=pdb_id,
            antibody_chain=antibody_chain,
            partner=partner,
            sequence=sequence,
            cdr=cdr,
            wt_dir=wt_dir,
            fix_dir=fix_dir,
            mut_dir=mut_dir,
            offtarget_names=args.offtarget_list,
            offtarget_wt_dir=offtarget_wt_dir,
            offtarget_fix_dir=offtarget_fix_dir,
            offtarget_mut_dir=offtarget_mut_dir,
            knn_num=args.knn_neighbors_num,
            knn_agents_num=args.knn_agents_num,
            use_plm_embedding=args.use_plm_embedding,
            plm_path=args.plm_path,
            plm_embedding_path=plm_embedding_path,
            device=device,
        )

        collate_fn = PaddingCollate()

        this_loss_dir = loss_dir + pdb_id + '/'
        loss_history = LossHistory(this_loss_dir, is_evolve=True)
        tb_writer = SummaryWriter(log_dir=this_loss_dir)
        loss_history.write(str(args) + '\n')
        loss_history.write(f'Offtarget list: {args.offtarget_list}\n')
        if checkpoint_args_path is not None:
            loss_history.write(f'Loaded model args from {checkpoint_args_path}\n')
            loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')

        try:
            tb_writer.add_text('run/args', str(args), 0)
            tb_writer.add_text('run/offtarget_list', str(args.offtarget_list), 0)
            tb_writer.add_text('data/pdb_id', pdb_id, 0)
            tb_writer.add_text('data/cdr3', cdr, 0)
            tb_writer.add_scalar('data/dataset_idx', i, 0)

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

            total_epoch = args.n_epoch
            best_candidate_scores = {}
            for epoch in range(total_epoch):
                epoch_metrics = train_one_epoch(
                    args, evo_model, reward_model, reference_model, optimizer,
                    epoch, total_epoch, train_dataset, collate_fn, loss_history, device,
                    tb_writer=tb_writer,
                )
                update_candidate_scores(best_candidate_scores, epoch_metrics['candidate_scores'])

            candidate_path = os.path.join(this_loss_dir, f'{pdb_id}_best_candidates.csv')
            pd.DataFrame(list(best_candidate_scores.values())).sort_values('reward_score').to_csv(
                candidate_path, index=False
            )
            loss_history.write(f'Best candidates saved to {candidate_path}\n')

        finally:
            tb_writer.flush()
            tb_writer.close()
