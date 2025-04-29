import os
import numpy as np
import pandas as pd
import datetime

from tqdm import tqdm
from itertools import combinations

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn import functional as F

from model.MERF import MERF

from utils.dataloader import SARSEvoDataset
from utils.arguments import get_evolution_args
from utils.util import *
from utils.losshistory import LossHistory

from protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet, PaddingCollate
from protein.docking_scripts import docking_script_sars
from protein.mutate_scripts import mut_list_sars

cpu_num = 100
torch.set_num_threads(cpu_num)
print(cpu_num)


def train_one_epoch(args, MERF_model, optimizer, epoch, n_epoch, train_loader, loss_history, device,
                    docking_folder, transform, agent_select,
                    wt_energy_omicron, wt_energy_delta, wt_energy_gamma, wt_energy_target, MIX_REWARD=False):
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
        
        wt_docking_energy_omicron = wt_energy_omicron
        wt_docking_energy_delta = wt_energy_delta
        wt_docking_energy_gamma = wt_energy_gamma
        
        mut_info = None  # the final one
        mut_path_omicron = None  # the final one
        mut_path_gamma = None  # the final one
        mut_path_delta = None  # the final one
        
        mut_docking_energy_omicron = None  # for docking reward
        mut_docking_energy_delta = None  # for docking reward
        mut_docking_energy_gamma = None  # for docking reward
        target_docking_energy = wt_energy_target
        
        # create mutation site combinations
        mutation_mask_list = []
        CDR_mask = batch['mutation_mask'][0]  # the initial mutation mask is the CDR
        true_indices = torch.nonzero(CDR_mask, as_tuple=True)[0]
        mutate_indices_list = list(combinations(true_indices, 3))
                
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
            
            actions = torch.argmin(qs_CDR, dim=1)
            
            this_mutate_info_list = []
            position_CDR = batch['wt']['resseq'][0][mutation_mask[0]]
            seq_CDR = batch['wt']['aa'][0][mutation_mask[0]]
            for ac_idx in range(len(actions)):
                state = seq_CDR[ac_idx].item()
                action = actions[ac_idx].item()
                position = position_CDR[ac_idx].item()
                
                this_mutate_info_list.append(str(aa_key[state]) + str(position) + str(aa_key[action]))
            
            mutate_info = '_'.join(this_mutate_info_list)
            
            mutate_path = 'data/SARS_COV_2/PDBs_evo/' + pdb_id + '_' + mutate_info + '.pdb'
            mutate_info_list.append(mutate_info)
            mutate_path_list.append(mutate_path)
        
        loss_history.write(f'\nEpoch{epoch} Iteration{iteration}: PDB id is {pdb_id}\n')
    
    loss_history.write(f'\nEpoch{epoch}: 1. Finish interaction\n')
    
    # 2. mutate to generate files for all possible combination
    # generate the combined rl
    loss_history.write(f'\nEpoch{epoch}: 2. Mutation\n')
    for idx in range(len(mutate_info_list)):
        mutate_info = mutate_info_list[idx]
        mutate_path = mutate_path_list[idx]
        
        loss_history.write(f'\nEpoch{epoch}: Mutating {pdb_id} to {mutate_info}\n')
        mut_list_sars(pdb_id, wt_path, mutate_path, [mutate_info], antibody_chain)
    
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
        
        q_tot_wt, _, = MERF_model(batch, device)
        score_list.append(q_tot_wt.item())
        loss_history.write(mutate_info + '\t' + str(q_tot_wt.item()) + '\n')
    
    score_list_w0 = [100 if x == 0 else x for x in score_list]
    score_array = np.array(score_list_w0)
    score_sort_idx = score_array.argsort()
    best_idx = score_sort_idx[0]  # min
    
    mut_info = mutate_info_list[best_idx]
    mut_path_omicron = mutate_path_list[best_idx]
    
    loss_history.write(f'\nEpoch{epoch}: 3. Finish selection\n')
    
    # 3.5 mutate the same type to delta and gamma complex!
    loss_history.write(f'\nEpoch{epoch}: 3.5. Mutation to the other two variants\n')
    pdb_id_wt = pdb_id.split('_')[0]
    pdb_id_gamma = pdb_id_wt + '_gamma'
    pdb_id_delta = pdb_id_wt + '_delta'
    
    mut_path_gamma = 'data/SARS_COV_2/PDBs_evo/' + pdb_id_gamma + '_' + mut_info + '.pdb'
    mut_path_delta = 'data/SARS_COV_2/PDBs_evo/' + pdb_id_delta + '_' + mut_info + '.pdb'
    
    wt_path_gamma = os.getcwd() + '/' + 'data/SARS_COV_2/PDBs_mutated/' + pdb_id_gamma + '.pdb'
    wt_path_delta = os.getcwd() + '/' + 'data/SARS_COV_2/PDBs_mutated/' + pdb_id_delta + '.pdb'
    
    mut_list_sars(pdb_id_gamma, wt_path_gamma, mut_path_gamma, [mut_info], antibody_chain)
    mut_list_sars(pdb_id_delta, wt_path_delta, mut_path_delta, [mut_info], antibody_chain)
    
    # 4. docking to get the rewards
    loss_history.write(f'\nEpoch{epoch}: 4. Docking\n')
    
    pdb_mut_id_omicron = pdb_id + '_' + mut_info
    
    mut_docking_energy_omicron = docking_script_sars(pdb_mut_id_omicron, mut_path_omicron, docking_folder, chain, score_key='I_sc')
    
    loss_history.write(f'\nEpoch{epoch}: PDB_id_old is {pdb_id}\n')
    loss_history.write(f'Epoch{epoch}: wt docking energy is {wt_docking_energy_omicron}\n')
    loss_history.write(f'Epoch{epoch}: PDB_id_new is {pdb_mut_id_omicron}\n')
    loss_history.write(f'Epoch{epoch}: mut docking energy is {mut_docking_energy_omicron}\n')
    
    loss_history.write(f'\nEpoch{epoch}: 4. Finish docking\n')
    
    # 4.5 further docking with two variants
    loss_history.write(f'\nEpoch{epoch}: 4.5 Further Docking\n')
    pdb_mut_id_gamma = pdb_id_gamma + '_' + mut_info
    pdb_mut_id_delta = pdb_id_delta + '_' + mut_info
    
    mut_docking_energy_gamma = docking_script_sars(pdb_mut_id_gamma, mut_path_gamma, docking_folder, chain, score_key='I_sc')
    mut_docking_energy_delta = docking_script_sars(pdb_mut_id_delta, mut_path_delta, docking_folder, chain, score_key='I_sc')
    
    # 5. critic network with two structures as input and loss calculation
    loss_history.write(f'\nEpoch{epoch}: 5. Updating\n')
    
    with tqdm(total=args.training_times, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration in range(args.training_times):
            
            # create label
            ddG_omicron = float(mut_docking_energy_omicron) - float(wt_docking_energy_omicron)
            if abs(ddG_omicron / wt_docking_energy_omicron) > 0.05:
                if ddG_omicron > 0:
                    label_omicron = 8
                else:
                    if mut_docking_energy_omicron < target_docking_energy:
                        label_omicron = -8
                    else:
                        label_omicron = -2
            else:
                label_omicron = 0
                
            if MIX_REWARD:
                ddG_gamma = float(mut_docking_energy_gamma) - float(wt_docking_energy_gamma)
                if abs(ddG_gamma / wt_docking_energy_gamma) > 0.05:
                    if ddG_gamma > 0:
                        label_gamma = 8
                    else:
                        if mut_docking_energy_gamma < target_docking_energy:
                            label_gamma = -8
                        else:
                            label_gamma = -2
                else:
                    label_gamma = 0
                    
                ddG_delta = float(mut_docking_energy_delta) - float(wt_docking_energy_delta)
                if abs(ddG_delta / wt_docking_energy_delta) > 0.05:
                    if ddG_delta > 0:
                        label_delta = 8
                    else:
                        if mut_docking_energy_omicron < target_docking_energy:
                            label_delta = -8
                        else:
                            label_delta = -2
                else:
                    label_delta = 0
                
                label = label_omicron + 0.5 * label_gamma + 0.5 * label_delta
            else:
                label = label_omicron
            
            complex_wt_info = parse_pdb(wt_path)
            complex_mut_info = parse_pdb(mut_path_omicron)
            
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
            'docking_energy_omicron': [wt_docking_energy_omicron, mut_docking_energy_omicron],
            'docking_energy_gamma': [wt_docking_energy_gamma, mut_docking_energy_gamma],
            'docking_energy_delta': [wt_docking_energy_delta, mut_docking_energy_delta],
        })
        save_df.to_csv(save_file, index=False)
        
    else:
        save_df = pd.read_csv(save_file)
        save_df = pd.concat([save_df, pd.DataFrame({
            'iteration': [len(save_df)], 'docking_energy_omicron': [mut_docking_energy_omicron],
            'docking_energy_gamma': mut_docking_energy_gamma, 'docking_energy_delta': mut_docking_energy_delta
        })],
                            ignore_index=True)
        save_df.to_csv(save_file, index=False)
    

if __name__ == '__main__':
    # ----------------------- environment setting ----------------------- #
    args = get_evolution_args()
    print(args)
    
    seed = args.seed
    seed_all(seed)
    print(f"setting random seed...{seed}")
    
    GPU_indx = args.gpu_idx
    device = torch.device(GPU_indx if args.is_cuda else "cpu")
    
    MIX_REWARD = True
    
    loss_dir = "logs/evo_sars/"
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time, '%Y_%m_%d_%H_%M_%S')
    loss_dir = loss_dir + time_str + '/'
    
    # add docking files storages
    docking_folder = "docking/sars_evo/"
    
    # ----------------------- prepare all antibodies to be evolved ----------------------- #
    train_path = 'data/SARS_COV_2/antibodies_for_evo_preprocessed.csv'
    train_df = pd.read_csv(train_path, dtype={"PDB_id": "string"})
    
    for i in range(len(train_df)):
        pdb_id = train_df['PDB_mut_id'].iloc[i]
        
        this_loss_dir = loss_dir + pdb_id + '/'
        loss_history = LossHistory(this_loss_dir, is_evolve=True)
        
        loss_history.write(str(args) + '\n')

        # ----------------------- data read in and create dataset ----------------------- #
        train_df_temp = train_df.iloc[i: i + 1]
        train_dataset = SARSEvoDataset(train_df_temp, knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num)
        
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
        
        wt_energy_omicron = float(train_df['omicron'].iloc[i])
        wt_energy_delta = float(train_df['delta'].iloc[i])
        wt_energy_gamma = float(train_df['gamma'].iloc[i])
        wt_energy_target = float(train_df['wt_energy'].iloc[i])
        
        total_epoch = 30
        for epoch in range(total_epoch):
            train_one_epoch(args, MERF_model, optimizer, epoch, total_epoch, train_loader,
                                          loss_history, device,
                                          docking_folder, transform, agent_select,
                                          wt_energy_omicron, wt_energy_delta, wt_energy_gamma, wt_energy_target, MIX_REWARD)
        
