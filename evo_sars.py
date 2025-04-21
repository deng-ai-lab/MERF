import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import datetime
from tqdm import tqdm
import json

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.dataloader import ABbindEvoDataset, SARSEvoDataset
from utils.arguments import get_common_args
from utils.util import *
from protein.read_pdbs import parse_pdb, KnnResidue, KnnAgnet

from protein.read_pdbs import PaddingCollate
from model.MERF import MERF
from utils.losshistory import LossHistory
from utils.fine_manager import FileManager
from itertools import combinations
from scipy.stats import pearsonr, spearmanr

cpu_num = 30
torch.set_num_threads(cpu_num)
print(cpu_num)


def get_lr(optimizer):
	for param_group in optimizer.param_groups:
		return param_group['lr']


def add_aa_id(batch_embed, complex_info):
	aa_onehot = F.one_hot(complex_info['aa'], 21)  # 20 is for padding
	return torch.cat((batch_embed, aa_onehot), dim=2)


# Generate Mutant chain using scap
def Gen_mut(workdir, pdbid, chainid, resid, mutname, idx):
	scapfile = open('./tmp_scap.list', 'w')
	scapfile.write(chainid + ',' + str(resid) + ',' + mutname)
	scapfile.close()
	if idx == 0:
		filename = workdir + '/' + pdbid + '.pdb'
	else:
		filename = workdir + '/' + pdbid + '_mut.pdb'
	os.system('./scap -ini 20 -min 4 ' + filename + ' ./tmp_scap.list')
	
	# change the mutation file to shorter name
	tmp_name = workdir + '/' + 'tmp.pdb'
	cmd = 'mv ' + filename + ' ' + tmp_name
	os.system(cmd)
	
	os.system('./scap -ini 20 -min 4 ' + tmp_name + ' ./tmp_scap.list')
	
	os.system('mv ' + 'tmp_scap.pdb ' + workdir + '/' + pdbid + '_mut.pdb')
	
	# if idx == 0:
	#     os.system('mv ' + pdbid + '_scap.pdb ' + workdir + '/' + pdbid + '_mut.pdb')
	# else:
	#     os.system('mv ' + pdbid + '_mut_scap.pdb ' + workdir + '/' + pdbid + '_mut.pdb')
	
	os.system('rm ./tmp_scap.list')
	return


def modifypdb(name1, name2):
	pdbfile = open(name1)
	outfile = open(name2, "w")
	lines = pdbfile.read().splitlines()
	PRO = []
	for line in lines:
		if line[0:4] == 'ATOM':
			outfile.write(line + '\n')
	outfile.close()
	return


def fix_structure(filepath):
	os.system('./profix -fix 0 ' + '/' + filepath + '/complex.pdb')  # 调用data中software
	os.system('mv complex_fix.pdb complex.pdb')
	os.system('mv complex.pdb ' + filepath + '/complex.pdb')
	return


def mut_list(pdb_id, pdb_path, mut_path, mutate_info_list, chainid):
	# 获取已突变的文件名
	os.chdir('protein')
	
	for idx in range(len(mutate_info_list)):
		mutate_info = mutate_info_list[idx]
		
		PDB_wt_file_path = pdb_path
		PDB_mut_name = pdb_id + '_' + mutate_info + '.pdb'
		
		target_dir = '../data/SARS_COV_2_Antibody/PDBs_evo'
		mutated_files = os.listdir(target_dir)
		
		# the first time to delete repeat
		if PDB_mut_name in mutated_files:
			continue
		
		# mutate, 注意多点突变调用多次突变函数！
		# create the workdir
		workdir = 'pdbfile'
		if os.path.exists(workdir):
			os.system('rm -r pdbfile')
		if not os.path.exists(workdir):
			os.system('mkdir ' + workdir)
		command = 'cp ' + PDB_wt_file_path + ' ' + workdir
		os.system(command)
		
		# fix structure, no need here
		# modifypdb(workdir + '/' + pdb_id + '.pdb', workdir + '/complex.pdb')
		# fix_structure(workdir)
		# os.system("mv " + workdir + '/complex.pdb ' + workdir + '/' + pdb_id_iter + '.pdb ')
		# os.system()
		
		# mutate files
		mutate_list = mutate_info.split('_')
		
		# temp_mutate_info = ''
		for i, single_mutate in enumerate(mutate_list):
			resid = str(int(single_mutate[1:-1]))
			mutname = single_mutate[-1]
			
			Gen_mut(workdir, pdb_id, chainid, str(resid), mutname, i)
		
		os.system('mv ' + workdir + '/' + pdb_id + '_mut.pdb ' + mut_path)
	
	os.chdir('..')


def clear_dir_single(workdir):
	print(f'-- clear the workdir of {workdir} ! --')
	
	output_path = os.path.join(workdir, 'output_files')
	output_relax_input_path = os.path.join(output_path, 'output_relax_input')
	output_docking_path = os.path.join(output_path, 'output_docking')
	output_refine_path = os.path.join(output_path, 'output_refine')
	output_relax_output_path = os.path.join(output_path, 'output_relax_output')
	if os.path.exists(output_path):
		cmd = 'rm -r ' + output_path
		os.system(cmd)
	if not os.path.exists(output_path):
		cmd = 'mkdir ' + output_path
		os.system(cmd)
		
		cmd = 'mkdir ' + output_relax_input_path
		os.system(cmd)
		cmd = 'mkdir ' + output_docking_path
		os.system(cmd)
		cmd = 'mkdir ' + output_refine_path
		os.system(cmd)
		cmd = 'mkdir ' + output_relax_output_path
		os.system(cmd)
	
	print('-- finish clearing ! --')


def docking_simple_single(workdir, pdb_id, chain, is_parallel, cpu_num=50, is_accurate=False, simple_docking_samples=50,
						  score_key='total_score'):
	print(f'-- start docking pdb:{pdb_id} ! --')
	
	output_path = os.path.join(workdir, 'output_files')
	output_relax_input_path = os.path.join(output_path, 'output_relax_input')
	output_docking_path = os.path.join(output_path, 'output_docking')
	output_refine_path = os.path.join(output_path, 'output_refine')
	output_relax_output_path = os.path.join(output_path, 'output_relax_output')
	
	# -------------------------------- 1. input relax -------------------------------- #
	if is_accurate:
		print(f'-- 1. input relax --')
		
		input_file = os.path.join(workdir, pdb_id + '.pdb')
		
		pdb_relaxed_name = pdb_id + '_0001.pdb'
		pdb_relaxed_temp = os.path.join(output_relax_input_path, pdb_relaxed_name)
		pdb_relaxed = os.path.join(workdir, pdb_id + '_relaxed.pdb')
		score_relax = os.path.join(output_relax_input_path, 'score.sc')
		
		if not os.path.exists(pdb_relaxed_temp):
			if is_parallel:
				cmd = 'mpirun -np ' + str(2) + \
					  ' relax.mpi.linuxgccrelease -in:file:s ' + input_file + \
					  ' -out:path:all ' + output_relax_input_path + \
					  ' -scorefile_format json' + \
					  ' -nstruct 2 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -mute all'
			else:
				cmd = 'relax.default.linuxgccrelease -in:file:s ' + input_file + \
					  ' -out:path:all ' + output_relax_input_path + \
					  ' -scorefile_format json' + \
					  ' -nstruct 2 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -mute all'
			os.system(cmd)
			
			# choose pdb with lower energy and change the path of pdb_relaxed_temp
			with open(score_relax, "r") as f:
				data_relax = f.readlines()
			info_list = []
			score_list = []
			for data in data_relax:
				info = json.loads(data)
				score = info[score_key]
				info_list.append(info)
				score_list.append(score)
			min_idx = np.argsort(np.array(score_list))[0]
			
			pdb_relaxed_real_name = info_list[min_idx]['decoy'] + '.pdb'
			pdb_relaxed_real_temp = os.path.join(output_relax_input_path, pdb_relaxed_real_name)
			cmd = 'cp ' + pdb_relaxed_real_temp + ' ' + pdb_relaxed
			os.system(cmd)
	
	# -------------------------------- 2. local dock -------------------------------- #
	print(f'-- 2. local dock --')
	if is_accurate:
		input_file = pdb_relaxed
		pdb_docked_name = pdb_id + '_relaxed_0001.pdb'
		num_struct = 500
	else:
		input_file = os.path.join(workdir, pdb_id + '.pdb')
		pdb_docked_name = pdb_id + '_0001.pdb'
		num_struct = simple_docking_samples
	
	pdb_docked_temp = os.path.join(output_docking_path, pdb_docked_name)
	pdb_docked = os.path.join(workdir, pdb_id + '_docked.pdb')
	score_docking = os.path.join(output_docking_path, 'score.sc')
	
	if not os.path.exists(pdb_docked_temp):
		if is_parallel:
			cmd = 'mpirun -np ' + str(cpu_num) + \
				  ' docking_protocol.mpi.linuxgccrelease -in:file:s ' + input_file + \
				  ' -out:path:all ' + output_docking_path + \
				  ' -scorefile_format json' + \
				  ' -partners ' + chain + \
				  ' -mute all -dock_pert 3 8 -ex1 -ex2aro -nstruct ' + str(num_struct)
		else:
			cmd = 'docking_protocol.linuxgccrelease -in:file:s ' + input_file + \
				  ' -out:path:all ' + output_docking_path + \
				  ' -scorefile_format json' + \
				  ' -partners ' + chain + \
				  ' -mute all -dock_pert 3 8 -ex1 -ex2aro -nstruct ' + str(num_struct)
		os.system(cmd)
		
		with open(score_docking, "r") as f:
			data_relax = f.readlines()
		info_list = []
		score_list = []
		for data in data_relax:
			info = json.loads(data)
			score = info[score_key]
			info_list.append(info)
			score_list.append(score)
		min_idx = np.argsort(np.array(score_list))[0]
		
		pdb_docked_real_name = info_list[min_idx]['decoy'] + '.pdb'
		pdb_docked_real_temp = os.path.join(output_docking_path, pdb_docked_real_name)
		
		cmd = 'cp ' + pdb_docked_real_temp + ' ' + pdb_docked
		os.system(cmd)
		
		if not is_accurate:
			final_score = score_list[min_idx]
			print(f'--finish SIMPLE docking pdb:{pdb_id} !--')
			print(f'final docking score is {final_score}')
	
	if is_accurate:
		# -------------------------------- 3. local refine -------------------------------- #
		print(f'-- 3. local refine --')
		input_file = pdb_docked
		
		pdb_refined_name = pdb_id + '_docked_0001.pdb'
		pdb_refined_temp = os.path.join(output_refine_path, pdb_refined_name)
		pdb_refined = os.path.join(workdir, pdb_id + '_refined.pdb')
		score_refine = os.path.join(output_refine_path, 'score.sc')
		
		if not os.path.exists(pdb_refined_temp):
			if is_parallel:
				cmd = 'mpirun -np ' + str(1) + \
					  ' docking_protocol.mpi.linuxgccrelease -in:file:s ' + input_file + \
					  ' -out:path:all ' + output_refine_path + \
					  ' -scorefile_format json' + \
					  ' -partners ' + chain + \
					  ' -mute all -dock_pert 3 8 -ex1 -ex2aro -nstruct 1  -use_input_sc -docking_local_refine'
			else:
				cmd = 'docking_protocol.linuxgccrelease -in:file:s ' + input_file + \
					  ' -out:path:all ' + output_refine_path + \
					  ' -scorefile_format json' + \
					  ' -partners ' + chain + \
					  ' -mute all -dock_pert 3 8 -ex1 -ex2aro -nstruct 1  -use_input_sc -docking_local_refine'
			os.system(cmd)
			
			cmd = 'cp ' + pdb_refined_temp + ' ' + pdb_refined
			os.system(cmd)
		
		# -------------------------------- 4. output relax -------------------------------- #
		print(f'-- 4. output relax --')
		input_file = pdb_refined
		
		pdb_result_name = pdb_id + '_refined_0001.pdb'
		pdb_result_temp = os.path.join(output_relax_output_path, pdb_result_name)
		pdb_result = os.path.join(workdir, pdb_id + '_result.pdb')
		score_result = os.path.join(output_relax_output_path, 'score.sc')
		
		if not os.path.exists(pdb_result_temp):
			if is_parallel:
				cmd = 'mpirun -np ' + str(2) + \
					  ' relax.mpi.linuxgccrelease -in:file:s ' + input_file + \
					  ' -out:path:all ' + output_relax_output_path + \
					  ' -scorefile_format json' + \
					  ' -nstruct 2 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -mute all'
			else:
				cmd = 'relax.default.linuxgccrelease -in:file:s ' + input_file + \
					  ' -out:path:all ' + output_relax_output_path + \
					  ' -scorefile_format json' + \
					  ' -nstruct 1 -relax:constrain_relax_to_start_coords -relax:ramp_constraints false -ex1 -ex2 -use_input_sc -flip_HNQ -no_optH false -mute all'
			os.system(cmd)
			
			with open(score_result, "r") as f:
				data_relax = f.readlines()
			info_list = []
			score_list = []
			for data in data_relax:
				info = json.loads(data)
				score = info['total_score']
				info_list.append(info)
				score_list.append(score)
			min_idx = np.argsort(np.array(score_list))[0]
			
			pdb_result_real_name = info_list[min_idx]['decoy'] + '.pdb'
			pdb_result_real_temp = os.path.join(output_relax_output_path, pdb_result_real_name)
			
			cmd = 'cp ' + pdb_result_real_temp + ' ' + pdb_result
			os.system(cmd)
			
			final_score = score_list[min_idx]
			
			print(f'--finish docking pdb:{pdb_id} !--')
			print(f'final docking score is {final_score}')
	
	return final_score


def docking_script(pdb_id, pdb_path, is_parallel, docking_folder, chain,
				   is_accurate=False, simple_docking_samples=50, score_key='total_score'):
	"""
		-workdir (docking_folder)
			-PDB_id (workdir)
				-PDB_id.pdb
				-(PDB_id_relaxed.pdb)
				-(PDB_id_docked.pdb)
				-(PDB_id_refined.pdb)
				-(PDB_id_result.pdb)
				-(score.sc) (? or in temp)
				-<temp> output_files
					-output_relax_input: 2
					-output_docking: 500
					-output_refine: 1
					-output_relax_output: 2

		-PDB_id.csv
	"""
	workdir = os.path.join(docking_folder, pdb_id)
	if not os.path.exists(workdir):
		os.system('mkdir ' + workdir)
	
	command = 'cp ' + pdb_path + ' ' + workdir
	os.system(command)
	
	clear_dir_single(workdir)
	
	final_score = docking_simple_single(workdir, pdb_id, chain, is_parallel,
										is_accurate=is_accurate, simple_docking_samples=simple_docking_samples,
										score_key=score_key)
	
	return final_score


def train_one_epoch(args, MERF_model, optimizer, epoch, n_epoch, train_loader, loss_history, device,
					is_parallel, docking_folder, transform, agent_select, global_step,
					wt_energy_omicron, wt_energy_delta, wt_energy_gamma, wt_energy_target, MIX_REWARD=False):
	# ----------------------- Train ----------------------- #
	print("Start Train")
	loss_history.write(f'\nEpoch{epoch}: Start Train\n')
	
	# 1. agent interaction with samples from dataloader, sample/get the best local actions
	# only 1 sample here, so the information is saved in iteration
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
		# mutate_indices_list = list(combinations(true_indices, 2))  # todo: change the mutation number here
		
		# todo: debug
		# mutate_indices_list = mutate_indices_list[0: 2]
		
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
			# probs = F.softmax(qs_CDR, dim=1)
			# actions = torch.multinomial(probs, num_samples=1).squeeze()
			
			this_mutate_info_list = []
			position_CDR = batch['wt']['resseq'][0][mutation_mask[0]]
			seq_CDR = batch['wt']['aa'][0][mutation_mask[0]]
			for ac_idx in range(len(actions)):
				state = seq_CDR[ac_idx].item()
				action = actions[ac_idx].item()
				position = position_CDR[ac_idx].item()
				
				this_mutate_info_list.append(str(aa_key[state]) + str(position) + str(aa_key[action]))
			
			mutate_info = '_'.join(this_mutate_info_list)
			
			mutate_path = '/home/lfj/projects_dir/Antibody_Mutation/data/SARS_COV_2_Antibody/PDBs_evo/' + pdb_id + '_' + mutate_info + '.pdb'
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
		mut_list(pdb_id, wt_path, mutate_path, [mutate_info], antibody_chain)
	
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
		
		q_tot_wt, _, _ = MERF_model(batch, device)
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
	
	mut_path_gamma = '/home/lfj/projects_dir/Antibody_Mutation/data/SARS_COV_2_Antibody/PDBs_evo/' + pdb_id_gamma + '_' + mut_info + '.pdb'
	mut_path_delta = '/home/lfj/projects_dir/Antibody_Mutation/data/SARS_COV_2_Antibody/PDBs_evo/' + pdb_id_delta + '_' + mut_info + '.pdb'
	
	wt_path_gamma = '/home/lfj/projects_dir/Antibody_Mutation/data/SARS_COV_2_Antibody/PDBs_mutated/' + pdb_id_gamma + '.pdb'
	wt_path_delta = '/home/lfj/projects_dir/Antibody_Mutation/data/SARS_COV_2_Antibody/PDBs_mutated/' + pdb_id_delta + '.pdb'
	
	mut_list(pdb_id_gamma, wt_path_gamma, mut_path_gamma, [mut_info], antibody_chain)
	mut_list(pdb_id_delta, wt_path_delta, mut_path_delta, [mut_info], antibody_chain)
		
	# 4. docking to get the rewards
	loss_history.write(f'\nEpoch{epoch}: 4. Docking\n')
	
	pdb_mut_id_omicron = pdb_id + '_' + mut_info
	
	# todo: debug
	mut_docking_energy_omicron = docking_script(pdb_mut_id_omicron, mut_path_omicron, is_parallel, docking_folder, chain,
								is_accurate=False, simple_docking_samples=200, score_key='I_sc')
	# mut_docking_energy_omicron = -10
	
	loss_history.write(f'\nEpoch{epoch}: PDB_id_old is {pdb_id}\n')
	loss_history.write(f'Epoch{epoch}: wt docking energy is {wt_docking_energy_omicron}\n')
	loss_history.write(f'Epoch{epoch}: PDB_id_new is {pdb_mut_id_omicron}\n')
	loss_history.write(f'Epoch{epoch}: mut docking energy is {mut_docking_energy_omicron}\n')
	
	loss_history.write(f'\nEpoch{epoch}: 4. Finish docking\n')
	
	# 4.5 further docking with two variants
	loss_history.write(f'\nEpoch{epoch}: 4.5 Further Docking\n')
	pdb_mut_id_gamma = pdb_id_gamma + '_' + mut_info
	pdb_mut_id_delta = pdb_id_delta + '_' + mut_info
	
	# todo: debug
	mut_docking_energy_gamma = docking_script(pdb_mut_id_gamma, mut_path_gamma, is_parallel, docking_folder, chain,
								is_accurate=False, simple_docking_samples=200, score_key='I_sc')
	mut_docking_energy_delta = docking_script(pdb_mut_id_delta, mut_path_delta, is_parallel, docking_folder, chain,
								is_accurate=False, simple_docking_samples=200, score_key='I_sc')
	# mut_docking_energy_gamma = -10
	# mut_docking_energy_delta = -10
	
	# 5. critic network with two structures as input and loss calculation
	loss_history.write(f'\nEpoch{epoch}: 5. Updating\n')
	
	with tqdm(total=args.training_times, desc=f'Epoch {epoch + 1}/{n_epoch}', postfix=dict, mininterval=0.3) as pbar:
		for iteration in range(args.training_times):
			# change the label to -8 or 8
			
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
			
			q_tot_wt, loss, _ = MERF_model(batch, device)
			
			optimizer.zero_grad()
			loss.backward()
			
			optimizer.step()
			
			writer.add_scalar('Loss/train', loss.item(), global_step)
			
			if global_step % args.training_times == 0:
				writer.add_scalar('Loss/mut_docking_energy_omicron', mut_docking_energy_omicron, global_step)
				writer.add_scalar('Loss/wt_docking_energy_omicron', wt_docking_energy_omicron, global_step)
				writer.add_scalar('Loss/mut_docking_energy_gamma', mut_docking_energy_gamma, global_step)
				writer.add_scalar('Loss/wt_docking_energy_gamma', wt_docking_energy_gamma, global_step)
				writer.add_scalar('Loss/mut_docking_energy_delta', mut_docking_energy_delta, global_step)
				writer.add_scalar('Loss/wt_docking_energy_delta', wt_docking_energy_delta, global_step)
				writer.add_scalar('Loss/target_energy', target_docking_energy, global_step)
			
			global_step += 1
			
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
	
	return global_step


if __name__ == '__main__':
	# ----------------------- environment setting ----------------------- #
	# add common envs
	args = get_common_args()
	print(args)
	
	seed = args.seed
	seed_all(seed)
	print(f"setting random seed...{seed}")
	
	GPU_indx = args.gpu_idx
	device = torch.device(GPU_indx if args.is_cuda else "cpu")
	
	MIX_REWARD = True
	
	# add rosetta envs
	is_parallel = True
	print('is parallel: ', is_parallel)
	
	rosetta_bin = ':/home/lfj/software/rosetta.source.release-340/main/source/bin/'
	mpi_bin = ':/home/lfj/projects_dir/mpi_install/bin'
	mpi_lib = ':/home/lfj/projects_dir/mpi_install/lib'
	mpi_manpath = ':/home/lfj/software_install/share/man'
	
	os.environ['PATH'] = os.environ['PATH'] + rosetta_bin
	os.environ['PATH'] = os.environ['PATH'] + mpi_bin
	cmd = "export LD_LIBRARY_PATH=/home/lfj/projects_dir/mpi_instll/lib/:$LD_LIBRARY_PATH"
	os.system(cmd)
	
	os.system('export PATH=\"/home/lfj/projects_dir/mpi_instll/bin/:$PATH\"')
	
	"""
	if the above not work, then:
	export PATH="/home/lfj/projects_dir/mpi_instll/bin/:$PATH"
	export PATH="/home/lfj/projects_dir/rosetta.source.release-340/main/source/bin/:$PATH"
	export LD_LIBRARY_PATH="/home/lfj/projects_dir/mpi_instll/lib/":$LD_LIBRARY_PATH
	"""
	
	# add logger envs, for the whole dataset
	# loss_dir = "logs/evo_rl_abbind_3/"
	loss_dir = "logs/evo_rl_sars/"
	curr_time0 = datetime.datetime.now()
	time_str0 = datetime.datetime.strftime(curr_time0, '%Y_%m_%d_%H_%M_%S')
	loss_dir = loss_dir + time_str0 + '/'
	
	# add docking files storages
	docking_folder = "/home/lfj/projects_dir/Antibody_Mutation/docking/sars_evo/"
	
	# read in files and start train loop
	
	train_path = 'data/SARS_COV_2_Antibody/antibodies_for_evo_preprocessed.csv'
	train_df = pd.read_csv(train_path, dtype={"PDB_id": "string"})
	
	pdb_start_index = args.pdb_start_index  # change to circle future
	pdb_end_index = args.pdb_end_index  # change to circle future
	
	train_df = train_df.iloc[pdb_start_index:pdb_end_index]
	
	for i in range(len(train_df)):
		pdb_id = train_df['PDB_mut_id'].iloc[i]
		
		this_loss_dir = loss_dir + pdb_id + '/'
		loss_history = LossHistory(this_loss_dir, only_pdb=True)
		
		loss_history.write(str(args) + '\n')
		
		tensor_log_dir = os.path.join('tensor_logs/evo_3', time_str0, pdb_id)
		
		writer = SummaryWriter(tensor_log_dir)
		writer.add_text(tag='args', text_string=str(args), global_step=1)
		
		# ----------------------- data read in and create dataset ----------------------- #
		train_df_temp = train_df.iloc[i: i + 1]
		train_dataset = SARSEvoDataset(train_df_temp, is_train=True, knn_num=args.knn_neighbors_num,
									   knn_agents_num=args.knn_agents_num)
		
		collate_fn = PaddingCollate()
		train_loader = DataLoader(train_dataset, shuffle=True, batch_size=1, pin_memory=False,
								  drop_last=True, num_workers=args.num_works, collate_fn=collate_fn)
		
		# ----------------------- Initial Networks ----------------------- #
		MERF_model = MERF(args).to(device)
		
		load_time_str = "2024_10_12_21_33_10"  # 1012last mask forward + BN + elu + 20 agent
		epoch_num = 15
		
		# the docking score in here is not accurate
		MERF_path = "/home/lfj/projects_dir/Antibody_Mutation/logs/abbind/" + load_time_str + "/Epoch" + str(
			epoch_num) + "_qmix_merf.pth"
		
		MERF_model.load_state_dict(torch.load(MERF_path, map_location=device))
		loss_history.write(f'\nLoading model from {MERF_path}\n')
		
		optimizer = optim.Adam(MERF_model.parameters(), lr=args.lr, weight_decay=0.1)
		
		transform = KnnResidue(num_neighbors=args.knn_neighbors_num)
		agent_select = KnnAgnet(num_neighbors=args.knn_agents_num)
		# ----------------------- fit one epoch ----------------------- #
		
		global_step = 0
		
		wt_energy_omicron = float(train_df['omicron'].iloc[i])
		wt_energy_delta = float(train_df['delta'].iloc[i])
		wt_energy_gamma = float(train_df['gamma'].iloc[i])
		wt_energy_target = float(train_df['wt_energy'].iloc[i])
		
		for epoch in range(15):
			global_step = train_one_epoch(args, MERF_model, optimizer, epoch, 15, train_loader,
										  loss_history, device,
										  is_parallel, docking_folder, transform, agent_select,
										  global_step,
										  wt_energy_omicron, wt_energy_delta, wt_energy_gamma, wt_energy_target, MIX_REWARD)
		
		writer.close()
