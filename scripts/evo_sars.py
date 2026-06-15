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
from dataset.dataset_evo import EvoSarsDataset
from utils.arguments import get_evolution_args
from utils.util import *
from utils.losshistory import LossHistory
from protein.read_pdbs import PaddingCollate

cpu_num = 32
torch.set_num_threads(cpu_num)
print(cpu_num)

VARIANT_WEIGHTS = {
    'wt': 1.0,
    'delta': 1.0,
    'gamma': 1.0,
    'omicron': 2.0,
}

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


def update_candidate_scores(best_candidate_scores, epoch_candidate_scores):
    for candidate in epoch_candidate_scores:
        mutate_info = candidate['mutate_info']
        prev_candidate = best_candidate_scores.get(mutate_info)
        if prev_candidate is None or candidate['reward_score'] < prev_candidate['reward_score']:
            best_candidate_scores[mutate_info] = candidate


def weighted_variant_qs(model, train_dataset, collate_fn, positions, device):
    weighted_qs = None
    for variant, weight in VARIANT_WEIGHTS.items():
        batch = train_dataset.get_policy_item(variant, positions)
        batch = collate_fn([batch])
        batch = recursive_to(batch, device)

        with torch.no_grad():
            qs = model.choose_best_action(batch, device)

        qs_sites = qs[batch['mutation_mask']]
        if weighted_qs is None:
            weighted_qs = weight * qs_sites
        else:
            weighted_qs = weighted_qs + weight * qs_sites

    return weighted_qs


def aggregate_agent_log_probs(model, train_dataset, collate_fn, mutate_info_list, device):
    weighted_qs = None
    batch_size = len(mutate_info_list)

    for variant, weight in VARIANT_WEIGHTS.items():
        sample_batches = [train_dataset.getitem_by_mutate_info(variant, mutate_info) for mutate_info in mutate_info_list]
        batch = collate_fn(sample_batches)
        batch = recursive_to(batch, device)

        qs = model.choose_best_action(batch, device)
        qs_agents = qs[batch['wt']['agent_mask']].view(batch_size, -1, qs.size(-1))

        if weighted_qs is None:
            weighted_qs = weight * qs_agents
        else:
            weighted_qs = weighted_qs + weight * qs_agents

    return F.log_softmax(-weighted_qs, dim=-1)


def score_mutation(reward_model, train_dataset, collate_fn, mutate_info, device):
    weighted_score = 0.0
    variant_scores = {}

    for variant, weight in VARIANT_WEIGHTS.items():
        batch = train_dataset.getitem_by_mutate_info(variant, mutate_info)
        batch = collate_fn([batch])
        batch = recursive_to(batch, device)

        with torch.no_grad():
            q_tot_wt, _, _, _ = reward_model(batch, device)

        score = q_tot_wt.item()
        variant_scores[variant] = score
        weighted_score += weight * score

    return weighted_score, variant_scores


def train_one_epoch(args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
                    train_dataset, collate_fn, loss_history, device, tb_writer=None):

    print("Start Train")
    loss_history.write(f'\nEpoch{epoch}: Start Train\n')

    pdb_id = train_dataset.pdb_id
    antibody_chain = train_dataset.antibody_chain
    cdr_entries = train_dataset.get_cdr_entries()
    mutate_entry_combinations = list(combinations(cdr_entries, args.comb_num))

    loss_history.write(f'\nEpoch{epoch}: 1. Agent interaction\n')
    loss_history.write(f'Variant weights: {VARIANT_WEIGHTS}\n')

    aa_key = "ACDEFGHIKLMNPQRSTVWY"
    mutate_info_list = []
    seen_mutate_infos = set()

    for mutate_entries in tqdm(mutate_entry_combinations, desc='Generating SARS mutation actions'):
        positions = [entry['position'] for entry in mutate_entries]
        wt_states = [entry['aa'] for entry in mutate_entries]

        weighted_qs = weighted_variant_qs(evo_model, train_dataset, collate_fn, positions, device)
        action_dist = torch.distributions.Categorical(logits=-weighted_qs)
        actions = action_dist.sample()

        this_mutate_info_list = []
        has_mutation = False
        for state, action_tensor, position in zip(wt_states, actions, positions):
            action = action_tensor.item()
            this_mutate_info_list.append(f"{aa_key[state]}{antibody_chain}{position}{aa_key[action]}")
            if state != action:
                has_mutation = True

        mutate_info = ','.join(this_mutate_info_list)
        if has_mutation and mutate_info not in seen_mutate_infos:
            mutate_info_list.append(mutate_info)
            seen_mutate_infos.add(mutate_info)

    loss_history.write(f'\nEpoch{epoch}: 1. Finish interaction, candidates={len(mutate_info_list)}\n')
    if len(mutate_info_list) == 0:
        loss_history.write('No valid mutation candidates were sampled; skipping epoch.\n')
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
    variant_score_list = []

    for mutate_info in mutate_info_list:
        weighted_score, variant_scores = score_mutation(reward_model, train_dataset, collate_fn, mutate_info, device)
        score_list.append(weighted_score)
        variant_score_list.append(variant_scores)
        loss_history.write(
            mutate_info
            + '\tweighted=' + str(weighted_score)
            + '\t' + '\t'.join(f'{variant}={variant_scores[variant]}' for variant in train_dataset.variants)
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
            ref_logits = aggregate_agent_log_probs(reference_model, train_dataset, collate_fn, sample_mutate_infos, device)
            old_logits = aggregate_agent_log_probs(old_model, train_dataset, collate_fn, sample_mutate_infos, device)

        curr_logits = aggregate_agent_log_probs(evo_model, train_dataset, collate_fn, sample_mutate_infos, device)

        log_ratio = curr_logits - old_logits
        ratio = torch.exp(log_ratio).clamp(max=20.0)

        adv_tensor = torch.tensor([advantages[idx] for idx in sample_indices], dtype=torch.float32, device=device)
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
        loss_history.write(f'  Inner epoch {inner_epoch}: avg_batch_loss={batch_loss.item():.6f}\n')

    loss = float(np.mean(epoch_losses)) if len(epoch_losses) > 0 else 0.0

    best_score = float(np.min(score_array))
    avg_score = float(np.mean(score_array))
    std_score = float(np.std(score_array))
    max_score = float(np.max(score_array))

    if tb_writer is not None:
        tb_writer.add_scalar('reward_model/best_weighted_score', best_score, epoch)
        tb_writer.add_scalar('reward_model/avg_weighted_score', avg_score, epoch)
        tb_writer.add_scalar('reward_model/std_weighted_score', std_score, epoch)
        tb_writer.add_scalar('reward_model/worst_weighted_score', max_score, epoch)
        tb_writer.add_scalar('train/loss', loss, epoch)
        tb_writer.add_histogram('reward_model/weighted_scores', score_array, epoch)

    loss_history.write(f'Epoch{epoch}: Best Weighted Score={best_score:.6f}, Avg Weighted Score={avg_score:.6f}, Loss={loss:.6f}\n')

    print('Finish Train')
    loss_history.write(f'\nFinish Train\n')

    print('Epoch:' + str(epoch + 1) + '/' + str(total_epoch))
    print('Best Weighted Score: %.6f, Avg Weighted Score: %.6f, Train Loss: %.6f' % (best_score, avg_score, loss))

    print('Saving state, iter:', str(epoch + 1))
    # save_path = loss_history.save_path + 'Epoch%d_evo.pth' % ((epoch + 1))
    # torch.save(evo_model.state_dict(), save_path)

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
    for mutate_info, score, variant_scores in zip(mutate_info_list, score_list, variant_score_list):
        candidate = {'epoch': epoch, 'mutate_info': mutate_info, 'reward_score': score}
        for variant in train_dataset.variants:
            candidate[f'{variant}_score'] = variant_scores[variant]
        epoch_candidate_scores.append(candidate)

    return {
        'best_score': best_score,
        'avg_score': avg_score,
        'train_loss': loss,
        'candidate_scores': epoch_candidate_scores,
    }


if __name__ == '__main__':
    args = get_evolution_args()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)

    seed = args.seed
    seed_all(seed)
    print(f"setting random seed...{seed}")

    device = torch.device(f'cuda:{args.gpu_idx}' if args.is_cuda else 'cpu')

    loss_dir = "logs/evo_sars/"
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    loss_dir = loss_dir + time_str + '/'

    sars_dir = '/home/dataset-local/projects_dir/MERF/data/SARS_COV_2'
    train_path = os.path.join(sars_dir, 'sars_evo_v5_with_resolution_hand_correct_seperate_pdb_correct_start_idx.csv')
    train_df = pd.read_csv(train_path, dtype={"pdb_id": "string"})
    wt_dir = os.path.join(sars_dir, 'PDBs_fixed')
    variant_dir = os.path.join(sars_dir, 'PDBs_mutated')
    evolved_dir = os.path.join(sars_dir, 'PDBs_evolved')
    plm_embedding_path = os.path.join(sars_dir, 'PLM_embeddings_sars_evo.pkl')
    os.makedirs(evolved_dir, exist_ok=True)

    if args.dataset_idx is not None:
        if args.dataset_idx < 0 or args.dataset_idx >= len(train_df):
            raise IndexError(f'dataset_idx {args.dataset_idx} is out of range for {train_path} with {len(train_df)} rows')
        dataset_indices = [args.dataset_idx]
        print(f'Only evolving dataset row {args.dataset_idx}.')
    else:
        dataset_indices = range(len(train_df))

    for i in dataset_indices:
        pdb_id = str(train_df['pdb_id'].iloc[i])
        antibody_name = train_df['Antibody_name'].iloc[i]
        antibody_chain = train_df['H_chain'].iloc[i]
        sequence = train_df['seq'].iloc[i]
        cdr = train_df['cdr3'].iloc[i]

        print(f'Start evolving {pdb_id}...')
        print(f'Antibody: {antibody_name}, antibody chain: {antibody_chain}, CDRH3: {cdr}')
        print(f'Sequence length: {len(sequence)}; CDRH3 length: {len(cdr)}')
        print('----------------------------------------')

        train_dataset = EvoSarsDataset(
            pdb_id, antibody_chain, sequence, cdr,
            wt_dir, variant_dir, evolved_dir,
            knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
            use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=plm_embedding_path,
            device=device
        )

        collate_fn = PaddingCollate()

        this_loss_dir = loss_dir + pdb_id + '/'
        loss_history = LossHistory(this_loss_dir, is_evolve=True)
        tb_writer = SummaryWriter(log_dir=this_loss_dir)
        loss_history.write(str(args) + '\n')
        loss_history.write(f'Variant weights: {VARIANT_WEIGHTS}\n')
        if checkpoint_args_path is not None:
            loss_history.write(f'Loaded model args from {checkpoint_args_path}\n')
            loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')

        try:
            tb_writer.add_text('run/args', str(args), 0)
            tb_writer.add_text('run/variant_weights', str(VARIANT_WEIGHTS), 0)
            tb_writer.add_text('data/pdb_id', pdb_id, 0)
            tb_writer.add_text('data/antibody_name', str(antibody_name), 0)
            tb_writer.add_text('data/cdrh3', cdr, 0)
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
                    args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
                    train_dataset, collate_fn, loss_history, device, tb_writer=tb_writer
                )
                update_candidate_scores(best_candidate_scores, epoch_metrics['candidate_scores'])

            candidate_path = os.path.join(this_loss_dir, f'{pdb_id}_best_candidates.csv')
            pd.DataFrame(list(best_candidate_scores.values())).sort_values('reward_score').to_csv(candidate_path, index=False)
            loss_history.write(f'Best candidates saved to {candidate_path}\n')
        finally:
            tb_writer.flush()
            tb_writer.close()
