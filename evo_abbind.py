import os
import numpy as np
import pandas as pd
import datetime

from tqdm import tqdm
from itertools import combinations

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from model.MERF import MERF

from utils.dataloader import ABbindEvoDataset
from utils.arguments import get_evolution_args
from utils.util import *
from utils.losshistory import LossHistory

from protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet, PaddingCollate
from protein.docking_scripts import docking_script_abbind
from protein.mutate_scripts import mut_list_abbind

cpu_num = 100
torch.set_num_threads(cpu_num)
print(cpu_num)

def train_one_epoch(args, MERF_model, optimizer, epoch, n_epoch, train_loader, loss_history, device,
                    docking_folder, transform, agent_select, wt_energy):

    # ----------------------- Train ----------------------- #
    print("Start Train")
    loss_history.write(f'\nEpoch{epoch}: Start Train\n')
    
    # 1. agent interaction with samples from dataloader, sample/get the best local actions
    loss_history.write(f'\nEpoch{epoch}: 1. Agent interaction\n')

    for iteration, batch in enumerate(train_loader):
        batch = recursive_to(batch, device)

        # basic information
        pdb_id = batch['expand_data_info']['PDB_id'][0]
        chain = batch['expand_data_info']['chain'][0]
        antibody_chain = batch['expand_data_info']['antibody_chain'][0]
        
        # mutation information
        mutate_info_list = []
        mutate_path_list = []
        
        # selection information
        score_list = []
        
        # rl training information
        wt_path = batch['index_info']['wt_path'][0]
        wt_energy = wt_energy
        mut_info = None  # the final one
        mut_path = None  # the final one
        mut_docking_energy = None  # for docking reward
        
        # create mutation site combinations
        mutation_mask_list = []
        CDR_mask = batch['mutation_mask'][0]  # the initial mutation mask is the CDR
        true_indices = torch.nonzero(CDR_mask, as_tuple=True)[0]
        mutate_indices_list = list(combinations(true_indices, args.comb_num))
        
        for mutate_indices in mutate_indices_list:
            mutate_indices_for_mask = torch.stack(mutate_indices)
            
            mutation_mask = torch.zeros_like(CDR_mask, dtype=torch.bool)
        
            mutation_mask[mutate_indices_for_mask] = True
            mutation_mask_list.append(mutation_mask.unsqueeze(0))
        
        # sample the best actions
        aa_key = "ACDEFGHIKLMNPQRSTVWY"

        for mutation_mask in mutation_mask_list:
            batch['mutation_mask'] = mutation_mask
            qs = MERF_model.choose_best_action(batch, device)
            
            CDR_mask = batch['mutation_mask'][0]
            qs_CDR = qs[0, CDR_mask]
            actions = torch.argmin(qs_CDR, dim=1)  # the same as argmax in paper, where in training stage the lower energy the better affinity
            
            this_mutate_info_list = []
            position_CDR = batch['wt']['resseq'][0][mutation_mask[0]]
            seq_CDR = batch['wt']['aa'][0][mutation_mask[0]]
            for ac_idx in range(len(actions)):
                state = seq_CDR[ac_idx].item()
                action = actions[ac_idx].item()
                position = position_CDR[ac_idx].item()
                
                this_mutate_info_list.append(str(aa_key[state]) + str(position) + str(aa_key[action]))
            
            mutate_info = '_'.join(this_mutate_info_list)
            
            mutate_path = 'data/ABbind/PDBs_evo/' + pdb_id + '_' + mutate_info + '.pdb'
            mutate_info_list.append(mutate_info)
            mutate_path_list.append(mutate_path)
        
        loss_history.write(f'\nEpoch{epoch} Iteration{iteration}: PDB id is {pdb_id}\n')
        
    loss_history.write(f'\nEpoch{epoch}: 1. Finish interaction\n')
    
    # 2. mutate to generate files for the limited possible combination
    loss_history.write(f'\nEpoch{epoch}: 2. Mutation\n')
    for idx in range(len(mutate_info_list)):
        mutate_info = mutate_info_list[idx]
        mutate_path = mutate_path_list[idx]
        
        loss_history.write(f'\nEpoch{epoch}: Mutating {pdb_id} to {mutate_info}\n')
        mut_list_abbind(pdb_id, wt_path, mutate_path, [mutate_info], antibody_chain)

    loss_history.write(f'\nEpoch{epoch}: 2. Finish mutation\n')
    
    loss_history.write(f'\nEpoch{epoch}: 3. Selection\n')
    
    # 3. select the best actions using mixer network
    for idx in range(len(mutate_info_list)):
        mutate_info = mutate_info_list[idx]
        mutate_path = mutate_path_list[idx]
        
        complex_wt_info = parse_pdb(wt_path)
        complex_mut_info = parse_pdb(mutate_path)
        
        # define mutation mask
        mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])
        
        # if all mutations suggest the same residues as wildtype, continue
        if len(mutation_mask[mutation_mask == True]) == 0:
            score_list.append(0)
            continue
        
        agent_mask = agent_select({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
        complex_wt_info['agent_mask'] = agent_mask
        complex_mut_info['agent_mask'] = agent_mask
        
        batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
        batch['ddG'] = 0
        
        batch = recursive_to(batch, device)
        
        batch = collate_fn([batch])
        
        q_tot_wt, _ = MERF_model(batch, device)
        score_list.append(q_tot_wt.item())
        loss_history.write(mutate_info + '\t' + str(q_tot_wt.item()) + '\n')
    
    # select the best mutant with the lowest estimated global energy
    score_list_w0 = [100 if x == 0 else x for x in score_list]
    score_array = np.array(score_list_w0)
    score_sort_idx = score_array.argsort()
    best_idx = score_sort_idx[0]
    mut_info = mutate_info_list[best_idx]
    mut_path = mutate_path_list[best_idx]
    
    loss_history.write(f'\nEpoch{epoch}: 3. Finish selection\n')
    
    # 4. docking to get the rewards
    loss_history.write(f'\nEpoch{epoch}: 4. Docking\n')
    
    if wt_energy == 'no_docking':
        wt_energy = docking_script_abbind(pdb_id, wt_path, docking_folder, chain, score_key='I_sc')

    pdb_mut_id = pdb_id + '_' + mut_info
    mut_energy = docking_script_abbind(pdb_mut_id, mut_path, docking_folder, chain, score_key='I_sc')
    
    loss_history.write(f'\nEpoch{epoch}: PDB_id_old is {pdb_id}\n')
    loss_history.write(f'Epoch{epoch}: wt docking energy is {wt_energy}\n')
    loss_history.write(f'Epoch{epoch}: PDB_id_new is {pdb_mut_id}\n')
    loss_history.write(f'Epoch{epoch}: mut docking energy is {mut_energy}\n')
    
    loss_history.write(f'\nEpoch{epoch}: 4. Finish docking\n')
    
    # 5. critic network with two structures as input and loss calculation
    loss_history.write(f'\nEpoch{epoch}: 5. Updating\n')
    
    with tqdm(total=args.training_times, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration in range(args.training_times):
            # change the label to -8 or 8
            ddG = float(mut_energy) - float(wt_energy)
            
            if abs(ddG / wt_energy) > 0.05:
                if ddG < 0:
                    label = -8
                else:
                    label = 8
            else:
                label = 0
            
            complex_wt_info = parse_pdb(wt_path)
            complex_mut_info = parse_pdb(mut_path)
            
            mutation_mask = (complex_wt_info['aa'] != complex_mut_info['aa'])
            
            agent_mask = agent_select(
                {'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
            complex_wt_info['agent_mask'] = agent_mask
            complex_mut_info['agent_mask'] = agent_mask
            
            batch = transform({'wt': complex_wt_info, 'mut': complex_mut_info, 'mutation_mask': mutation_mask})
            
            batch = recursive_to(batch, device)
            
            batch['ddG'] = label
            
            batch = collate_fn([batch])
            
            q_tot_wt, loss = MERF_model(batch, device)
            
            optimizer.zero_grad()
            loss.backward()
            
            optimizer.step()
        
            loss_history.write(f'\nEpoch{epoch} Iteration{iteration}: batch train loss is {loss.item()}\n')
            pbar.update(1)
    
    loss_history.write(f'\nEpoch{epoch}: 5. Finish Update\n')
    
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

    loss_dir = "logs/evo_abbind/"
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    loss_dir = loss_dir + time_str + '/'
    
    # add docking files storages
    docking_folder = "docking/Abbind_evo/"
    
    # ----------------------- prepare all antibodies to be evolved ----------------------- #
    train_path = 'data/ABbind/AB-Bind_evo.csv'
    train_df = pd.read_csv(train_path, dtype={"PDB_id": "string"})
    
    for i in range(len(train_df)):
        pdb_id = train_df['PDB_id'].iloc[i]
        
        this_loss_dir = loss_dir + pdb_id + '/'
        loss_history = LossHistory(this_loss_dir, is_evolve=True)
        
        loss_history.write(str(args) + '\n')

        # ----------------------- data read in and create dataset ----------------------- #
        train_df_temp = train_df.iloc[i: i+1]
        train_dataset = ABbindEvoDataset(train_df_temp, knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num)
        
        collate_fn = PaddingCollate()
        train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, pin_memory=False,
                                  drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)
            
        # ----------------------- Initial Networks ----------------------- #
        MERF_model = MERF(args).to(device)
        MERF_path = "model/MERF_pretrained.pth"

        MERF_model.load_state_dict(torch.load(MERF_path, map_location=device))
        loss_history.write(f'\nLoading model from {MERF_path}\n')
    
        optimizer = optim.Adam(MERF_model.parameters(), lr=args.lr, weight_decay=0.1)
    
        transform = KnnResidue(num_neighbors=args.knn_neighbors_num)
        agent_select = KnnAgnet(num_neighbors=args.knn_agents_num)

        # ----------------------- fit one epoch ----------------------- #        
        wt_energy = 'no_docking'
        
        total_epoch = 30
        for epoch in range(total_epoch):
            wt_energy = train_one_epoch(args, MERF_model, optimizer, epoch, total_epoch, train_loader,
                                                     loss_history, device,
                                                     docking_folder, transform, agent_select, wt_energy)
