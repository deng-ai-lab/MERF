import os
import numpy as np
import pandas as pd
import datetime

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tqdm import tqdm
from itertools import combinations

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.MERF import MERF

from dataset.dataset_evo import EvoBaseDataset
from utils.arguments import get_evolution_args
from utils.util import *
from utils.losshistory import LossHistory

from protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet, PaddingCollate
from protein.docking_scripts import docking_script_abbind
from protein.mutate_scripts import mut_list_abbind

cpu_num = 100
torch.set_num_threads(cpu_num)
print(cpu_num)

def train_one_epoch(args, evo_model, reward_model, optimizer, epoch, total_epoch, 
                    train_dataset, collate_fn, loss_history, device):

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

    mutate_info_list = mutate_info_list[0:5]  # TODO: debug only
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
            q_tot_wt, _ = reward_model(batch, device)

        score_list.append(q_tot_wt.item())

        loss_history.write(mutate_info + '\t' + str(q_tot_wt.item()) + '\n')
    
    loss_history.write(f'\nEpoch{epoch}: 3. Finish Scoring\n')

    # 4. using GRPO to train the evo_model

    loss_history.write(f'\nEpoch{epoch}: 4. GRPO Training\n')
    
    # TODO: add GRPO training here


    # TODO: end of GRPO training    

    print('Finish Train')
    loss_history.write(f'\nFinish Train\n')
    
    print('Epoch:' + str(epoch + 1) + '/' + str(n_epoch))
    print('Batch Train Loss: %.4f' % (loss))
    
    print('Saving state, iter:', str(epoch + 1))
    save_path = loss_history.save_path + 'Epoch%d_evo.pth' % ((epoch + 1))
    torch.save(MERF_model.state_dict(), save_path)
    
    # save results
    save_file = os.path.join(loss_history.save_path, pdb_id + '.csv')
    if not os.path.exists(save_file):
        save_df = pd.DataFrame({
            'iteration': [0, 1],
            'docking_energy': [wt_energy, mut_energy]
        })
        save_df.to_csv(save_file, index=False)
    else:
        save_df = pd.read_csv(save_file)
        save_df = pd.concat([save_df, pd.DataFrame({'iteration': [len(save_df)], 'docking_energy': [mut_energy]})], ignore_index=True)
        save_df.to_csv(save_file, index=False)
        
    return wt_energy


if __name__ == '__main__':
    # ----------------------- environment setting ----------------------- #
    args = get_evolution_args()
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
    wt_dir = '/home/lfj/projects_dir/MERF/data/sabdab/PDBs'
    fix_dir = '/home/lfj/projects_dir/MERF/data/sabdab/PDBs_fixed'
    mut_dir = '/home/lfj/projects_dir/MERF/data/sabdab/PDBs_evo'
    plm_embedding_path = '/home/lfj/projects_dir/MERF/data/sabdab/PLM_embeddings_sabdab.pkl'
    
    for i in range(len(train_df)):

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
        loss_history.write(str(args) + '\n')

        # ----------------------- Initial Networks ----------------------- #
        evo_model = MERF(args).to(device)
        evo_path = "model/MERF_pretrained.pth"

        # evo_model.load_state_dict(torch.load(evo_path, map_location=device))  # TODO: add this back
        loss_history.write(f'\nLoading model from {evo_path}\n')
    
        optimizer = optim.Adam(evo_model.parameters(), lr=args.lr, weight_decay=0.1)

        reward_model = MERF(args).to(device)
        # reward_model.load_state_dict(torch.load(evo_path, map_location=device))  # TODO: add this back
        reward_model.eval()

        # ----------------------- fit one epoch ----------------------- #                
        total_epoch = 3000
        for epoch in range(total_epoch):
            wt_energy = train_one_epoch(
                args, evo_model, reward_model, optimizer, epoch, total_epoch, 
                train_dataset, collate_fn, loss_history, device
            )
