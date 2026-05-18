import os
import numpy as np
import pandas as pd
import datetime
import pickle

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from itertools import combinations

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from model.MERF import MERF

from dataset.dataset_evo import EvoBaseDataset
from utils.arguments import get_evolution_args
from utils.util import *
from utils.losshistory import LossHistory

from protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet, PaddingCollate
from protein.docking_scripts import docking_script_abbind
from protein.mutate_scripts import mut_list_abbind

cpu_num = 32
torch.set_num_threads(cpu_num)
print(cpu_num)

CHECKPOINT_MODEL_ARG_KEYS = (
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


def train_one_epoch(args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
                    train_dataset, collate_fn, loss_history, device, tb_writer=None):

    print("Start Train")
    loss_history.write(f'\nEpoch{epoch}: Start Train\n')
    
    # 0. get dataset info
    batch = train_dataset.__getitem__(0)
    batch = collate_fn([batch])
    batch = recursive_to(batch, device)

    pdb_id = batch['expand_data_info']['pdb_id'][0]
    chain = batch['expand_data_info']['chain'][0]
    antibody_chain = batch['expand_data_info']['antibody_chain'][0]
    cdr_mask = batch['mutation_mask'][0]  # the initial mutation mask is the CDR

    # 1. agent interaction with samples, sample/argmax the best local actions
    loss_history.write(f'\nEpoch{epoch}: 1. Agent interaction\n')

    # create mutation site combinations for all {comb_num} mutations
    true_indices = torch.nonzero(cdr_mask, as_tuple=True)[0]
    mutate_indices_list = list(combinations(true_indices, args.comb_num))

    mutation_mask_list = []
    for mutate_indices in mutate_indices_list:
        mutate_indices_for_mask = torch.stack(mutate_indices)
        
        mutation_mask = torch.zeros_like(cdr_mask, dtype=torch.bool)
    
        mutation_mask[mutate_indices_for_mask] = True
        mutation_mask_list.append(mutation_mask.unsqueeze(0))
    
    # sample the best actions
    aa_key = "ACDEFGHIKLMNPQRSTVWY"
    mutate_info_list = []

    for mutation_mask in tqdm(mutation_mask_list,desc='Generating mutation actions for all combinations of sites'):
        batch['mutation_mask'] = mutation_mask
        qs = evo_model.choose_best_action(batch, device)
        
        qs_CDR = qs[mutation_mask]
        actions = torch.argmin(qs_CDR, dim=1)  # the same as argmax in paper, where in training stage the lower energy the better affinity
        
        this_mutate_info_list = []
        position_CDR = batch['wt']['resseq'][0][mutation_mask[0]]  # 考虑到已经取了128最近，所以应该用筛选后的CDR位置，并非直接拿indices
        seq_CDR = batch['wt']['aa'][0][mutation_mask[0]]

        has_mutation = False

        for ac_idx in range(len(actions)):
            state = seq_CDR[ac_idx].item()
            action = actions[ac_idx].item()
            position = position_CDR[ac_idx].item()
            
            this_mutate_info_list.append(f"{aa_key[state]}{antibody_chain}{position}{aa_key[action]}")

            if state != action:
                has_mutation = True
        
        mutate_info = ','.join(this_mutate_info_list)
        
        if has_mutation:
            mutate_info_list.append(mutate_info)
        
    loss_history.write(f'\nEpoch{epoch}: 1. Finish interaction\n')

    # 2. mutate to generate files for the limited possible combination
    loss_history.write(f'\nEpoch{epoch}: 2. Mutation\n')

    # mutate_info_list = mutate_info_list[0:5]  # TODO: debug only
    train_dataset.mutate_pdb(mutate_info_list)

    loss_history.write(f'\nEpoch{epoch}: 2. Finish mutation\n')
    
    # 3. Scoring all the mutated structures with reward_model
    loss_history.write(f'\nEpoch{epoch}: 3. Scoring\n')

    score_list = []
    
    for mutate_info in mutate_info_list:

        batch = train_dataset.getitem_by_mutate_info(mutate_info)
        batch = collate_fn([batch])
        batch = recursive_to(batch, device)
        
        with torch.no_grad():
            q_tot_wt, _, _, _ = reward_model(batch, device)

        score_list.append(q_tot_wt.item())

        loss_history.write(mutate_info + '\t' + str(q_tot_wt.item()) + '\n')
    
    loss_history.write(f'\nEpoch{epoch}: 3. Finish Scoring\n')
    if tb_writer is not None:
        tb_writer.add_scalar('evo/num_candidates', len(score_list), epoch)

    # 4. using GRPO to train the evo_model

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
    loss = 0.0

    for inner_epoch in tqdm(range(args.inner_epochs), desc='GRPO Training'):
        sample_indices = np.random.choice(n_samples, size=min(args.inner_batch_size, n_samples), replace=False)

        sample_batches = [train_dataset.getitem_by_mutate_info(mutate_info_list[idx]) for idx in sample_indices]
        batch = collate_fn(sample_batches)
        batch = recursive_to(batch, device)

        optimizer.zero_grad()

        with torch.no_grad():
            ref_qs = reference_model.choose_best_action(batch, device)
            ref_logits = torch.nn.functional.log_softmax(-ref_qs, dim=-1)

        with torch.no_grad():
            old_qs = old_model.choose_best_action(batch, device)
            old_logits = torch.nn.functional.log_softmax(-old_qs, dim=-1)

        curr_qs = evo_model.choose_best_action(batch, device)
        curr_logits = torch.nn.functional.log_softmax(-curr_qs, dim=-1)

        # using batch['wt]['agent_mask'] (sample batch size * 128) to index turly contributed agents
        ref_logits = ref_logits[batch['wt']['agent_mask']].view(len(sample_indices), -1, ref_logits.size(-1))
        old_logits = old_logits[batch['wt']['agent_mask']].view(len(sample_indices), -1, ref_logits.size(-1))
        curr_logits = curr_logits[batch['wt']['agent_mask']].view(len(sample_indices), -1, ref_logits.size(-1))

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

    loss = np.mean(epoch_losses)

    best_score = np.min(score_array)
    avg_score = np.mean(score_array)
    std_score = np.std(score_array)
    max_score = np.max(score_array)

    if tb_writer is not None:
        tb_writer.add_scalar('reward_model/best_score', best_score, epoch)
        tb_writer.add_scalar('reward_model/avg_score', avg_score, epoch)
        tb_writer.add_scalar('reward_model/std_score', std_score, epoch)
        tb_writer.add_scalar('reward_model/worst_score', max_score, epoch)
        tb_writer.add_scalar('eval/best_score', best_score, epoch)
        tb_writer.add_scalar('eval/avg_score', avg_score, epoch)
        tb_writer.add_scalar('train/loss', loss, epoch)
        tb_writer.add_histogram('reward_model/scores', score_array, epoch)

    loss_history.write(f'Epoch{epoch}: Best Score={best_score:.6f}, Avg Score={avg_score:.6f}, Loss={loss:.6f}\n')

    print('Finish Train')
    loss_history.write(f'\nFinish Train\n')

    print('Epoch:' + str(epoch + 1) + '/' + str(total_epoch))
    print('Best Score: %.6f, Avg Score: %.6f, Train Loss: %.6f' % (best_score, avg_score, loss))

    print('Saving state, iter:', str(epoch + 1))
    save_path = loss_history.save_path + 'Epoch%d_evo.pth' % ((epoch + 1))
    torch.save(evo_model.state_dict(), save_path)

    save_file = os.path.join(loss_history.save_path, pdb_id + '.csv')
    if not os.path.exists(save_file):
        save_df = pd.DataFrame({
            'epoch': [epoch],
            'best_score': [best_score],
            'avg_score': [avg_score],
            'train_loss': [loss]
        })
        save_df.to_csv(save_file, index=False)
    else:
        save_df = pd.read_csv(save_file)
        new_row = pd.DataFrame({
            'epoch': [epoch],
            'best_score': [best_score],
            'avg_score': [avg_score],
            'train_loss': [loss]
        })
        save_df = pd.concat([save_df, new_row], ignore_index=True)
        save_df.to_csv(save_file, index=False)

    return best_score


if __name__ == '__main__':
    # ----------------------- environment setting ----------------------- #
    args = get_evolution_args()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)
    
    seed = args.seed
    seed_all(seed)
    print(f"setting random seed...{seed}")
    
    GPU_indx = args.gpu_idx
    device = torch.device(GPU_indx if args.is_cuda else "cpu")

    loss_dir = "logs/evo_sabdab/"
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    loss_dir = loss_dir + time_str + '/'
    
    # ----------------------- Iteratively evo antibodies ----------------------- #
    train_path = 'data/sabdab/sabdab_evo.csv'
    train_df = pd.read_csv(train_path, dtype={"pdb_id": "string"})
    wt_dir = '/home/dataset-local/projects_dir/MERF/data/sabdab/PDBs'
    fix_dir = '/home/dataset-local/projects_dir/MERF/data/sabdab/PDBs_fixed'
    mut_dir = '/home/dataset-local/projects_dir/MERF/data/sabdab/PDBs_evo'
    plm_embedding_path = '/home/dataset-local/projects_dir/MERF/data/sabdab/PLM_embeddings_sabdab.pkl'

    if args.dataset_idx is not None:
        if args.dataset_idx < 0 or args.dataset_idx >= len(train_df):
            raise IndexError(f'dataset_idx {args.dataset_idx} is out of range for {train_path} with {len(train_df)} rows')
        dataset_indices = [args.dataset_idx]
        print(f'Only evolving dataset row {args.dataset_idx}.')
    else:
        dataset_indices = range(len(train_df))

    for i in dataset_indices:

        # Data info and dataset
        pdb_id = train_df['pdb_id'].iloc[i].replace('+', '').replace('.00', '')
        antibody_chain = train_df['antibody_chain'].iloc[i]
        partner = train_df['partner'].iloc[i]
        sequence = train_df['sequence'].iloc[i]
        cdrs = [train_df['cdr1'].iloc[i], train_df['cdr2'].iloc[i], train_df['cdr3'].iloc[i]]
        cdr = cdrs[2]  # only evolve CDRH3 in this script

        print(f'Start evolving {pdb_id}...')
        print(f'Antibody chain: {antibody_chain}, Partner: {partner}, CDRH3: {cdr}')
        print(f'Sequence length: {len(sequence)}; sequence: {sequence}')
        print(f'CDRH3 length: {len(cdr)}, CDRH3 sequence: {cdr}')
        print('----------------------------------------')

        train_dataset = EvoBaseDataset(
            pdb_id, antibody_chain, partner, sequence, cdr,
            wt_dir, fix_dir, mut_dir,
            knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
            use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path, plm_embedding_path=plm_embedding_path,
            device=device
        )
        
        collate_fn = PaddingCollate()
        
        # ----------------------- Loss History ----------------------- #
        this_loss_dir = loss_dir + pdb_id + '/'
        loss_history = LossHistory(this_loss_dir, is_evolve=True)
        tb_writer = SummaryWriter(log_dir=this_loss_dir)
        loss_history.write(str(args) + '\n')
        if checkpoint_args_path is not None:
            loss_history.write(f'Loaded model args from {checkpoint_args_path}\n')
            loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')

        try:
            tb_writer.add_text('run/args', str(args), 0)
            if checkpoint_args_path is not None:
                tb_writer.add_text('run/checkpoint_args_path', checkpoint_args_path, 0)
                tb_writer.add_text('run/overrode_model_args', ', '.join(checkpoint_model_arg_keys), 0)
            tb_writer.add_text('data/pdb_id', pdb_id, 0)
            tb_writer.add_text('data/cdrh3', cdr, 0)
            tb_writer.add_scalar('data/dataset_idx', i, 0)

            # ----------------------- Initial Networks ----------------------- #
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

            # ----------------------- fit one epoch ----------------------- #
            total_epoch = 3000
            for epoch in range(total_epoch):
                wt_energy = train_one_epoch(
                    args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
                    train_dataset, collate_fn, loss_history, device, tb_writer=tb_writer
                )
        finally:
            tb_writer.flush()
            tb_writer.close()
