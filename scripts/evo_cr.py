import datetime
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

from dataset.dataset_cr_evo import CREvoDataset
from model.MERF_v6 import MERF
from protein.read_pdbs import PaddingCollate
from utils.arguments import get_evolution_args
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


def selected_mean_log_prob(binary_log_probs, selected_actions):
    selected_actions = selected_actions.to(binary_log_probs.device)
    selected = binary_log_probs.gather(1, selected_actions.unsqueeze(-1)).squeeze(-1)
    return selected.mean()


def train_one_epoch_cr(args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
                       train_dataset, collate_fn, loss_history, device, reward_baseline, valid_ranks,
                       writer=None):

    loss_history.write(f'\nEpoch{epoch}: Start CR evolution\n')

    policy_batch = train_dataset.__getitem__(0)
    policy_batch = collate_fn([policy_batch])
    policy_batch = recursive_to(policy_batch, device)

    was_training = evo_model.training
    evo_model.eval()
    with torch.no_grad():
        binary_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device)
        selected_actions, reverse_key, policy_score = train_dataset.choose_candidate_from_log_probs(binary_log_probs)
    if was_training:
        evo_model.train()

    score_batch = train_dataset.getitem_by_action_tensor(selected_actions)
    score_batch = collate_fn([score_batch])
    score_batch = recursive_to(score_batch, device)

    with torch.no_grad():
        model_score, _, _, _ = reward_model(score_batch, device)
    model_score_value = float(model_score.item())

    if reward_baseline is None:
        # reward_baseline = model_score_value
        reward_baseline = 0
    advantage_value = reward_baseline - model_score_value  # score都是越低越好，这里advantage取了负号，因此advantage是越高越好
    advantage = torch.tensor(advantage_value, dtype=torch.float32, device=device)

    old_model = MERF(args).to(device)
    old_model.load_state_dict(evo_model.state_dict())
    old_model.eval()

    selected_actions = selected_actions.to(device)
    epoch_losses = []
    inner_kl_values = []
    for inner_epoch in tqdm(range(args.inner_epochs), desc='CR PPO Training'):
        optimizer.zero_grad()

        with torch.no_grad():
            old_log_probs = get_binary_log_probs(old_model, policy_batch, train_dataset, device)
            old_selected_log_prob = selected_mean_log_prob(old_log_probs, selected_actions)

        with torch.no_grad():
            ref_log_probs = get_binary_log_probs(reference_model, policy_batch, train_dataset, device)

        curr_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device)
        curr_selected_log_prob = selected_mean_log_prob(curr_log_probs, selected_actions)

        ratio = torch.exp(curr_selected_log_prob - old_selected_log_prob).clamp(max=20.0)
        policy_loss = -(ratio * advantage)  # advantage是越高越好，所以loss加个负号，梯度下降时loss下降，也就会带动advantage上升
        kl_div = (ref_log_probs.exp() * (ref_log_probs - curr_log_probs)).sum(dim=-1).mean()
        batch_loss = policy_loss + args.kl_coeff * kl_div

        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(float(batch_loss.item()))
        inner_kl_values.append(float(kl_div.item()))
        loss_history.write(
            f'  Inner epoch {inner_epoch}: loss={batch_loss.item():.6f}, '
            f'advantage={advantage_value:.6f}, kl={kl_div.item():.6f}\n'
        )

    # reward_baseline = 0.9 * reward_baseline + 0.1 * model_score_value
    reward_baseline = reward_baseline  # 0615 test

    eval_info = train_dataset.evaluate_reverse_key(reverse_key)
    true_rank = eval_info['rank']
    true_score = eval_info['true_score']
    if true_rank is not None:
        valid_ranks.append(true_rank)

    mean_rank = float(np.mean(valid_ranks)) if valid_ranks else np.nan
    best_rank = int(np.min(valid_ranks)) if valid_ranks else np.nan
    forward_reverse_info, forward_info = train_dataset.reverse_key_to_forward_info(reverse_key)
    train_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
    train_kl = float(np.mean(inner_kl_values)) if inner_kl_values else 0.0

    loss_history.write(
        f'Epoch{epoch}: reverse_key={reverse_key}, forward_info={forward_info}, '
        f'model_score={model_score_value:.6f}, true_score={true_score}, rank={true_rank}, '
        f'mean_rank={mean_rank}, best_rank={best_rank}, loss={train_loss:.6f}\n'
    )

    print('Epoch:' + str(epoch + 1) + '/' + str(total_epoch))
    print(
        'Model Score: %.6f, Rank: %s, Mean Rank: %s, Best Rank: %s, Train Loss: %.6f'
        % (model_score_value, str(true_rank), str(mean_rank), str(best_rank), train_loss)
    )

    # if epoch % 50 == 0:
    #     save_path = os.path.join(loss_history.save_path, 'Epoch%d_evo.pth' % (epoch + 1))
    #     torch.save(evo_model.state_dict(), save_path)

    metrics = {
        'epoch': epoch,
        'reverse_mutations_in_structure': reverse_key,
        'forward_reverse_ids': forward_reverse_info,
        'forward_mutations': forward_info,
        'model_score': model_score_value,
        'reward_baseline': reward_baseline,
        'advantage': advantage_value,
        'true_score': true_score,
        'rank': true_rank,
        'num_ranked': eval_info['num_ranked'],
        'mean_rank': mean_rank,
        'best_rank': best_rank,
        'policy_score': float(policy_score.item()),
        'train_loss': train_loss,
        'train_kl': train_kl,
    }

    if writer is not None:
        writer.add_scalar('Evolution/rank', true_rank if true_rank is not None else np.nan, epoch + 1)
        writer.add_scalar('Evolution/mean_rank', mean_rank, epoch + 1)
        writer.add_scalar('Evolution/best_rank', best_rank, epoch + 1)
        writer.add_scalar('Evolution/reward', advantage_value, epoch + 1)
        writer.add_scalar('Evolution/model_score', model_score_value, epoch + 1)
        writer.add_scalar('Evolution/reward_baseline', reward_baseline, epoch + 1)
        writer.add_scalar('Evolution/policy_score', float(policy_score.item()), epoch + 1)
        writer.add_scalar('Train/loss', train_loss, epoch + 1)
        writer.add_scalar('Train/kl', train_kl, epoch + 1)
        if true_score is not None:
            writer.add_scalar('Evolution/true_score', true_score, epoch + 1)
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
    args = get_evolution_args()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)

    seed = args.seed
    seed_all(seed)
    print(f"setting random seed...{seed}")

    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")

    loss_dir = "logs/evo_cr/"
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

    total_epoch = 3000
    reward_baseline = None
    valid_ranks = []
    for epoch in range(total_epoch):
        reward_baseline, valid_ranks, _ = train_one_epoch_cr(
            args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
            train_dataset, collate_fn, loss_history, device, reward_baseline, valid_ranks,
            writer=writer,
        )

    writer.close()
