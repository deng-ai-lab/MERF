import os
import json
import numpy as np


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


def docking_simple_single_abbind(workdir, pdb_id, chain, cpu_num=100, docking_samples=200, score_key='total_score'):
    print(f'-- start docking pdb:{pdb_id} ! --')
    
    output_path = os.path.join(workdir, 'output_files')
    output_docking_path = os.path.join(output_path, 'output_docking')
    
    # -------------------------------- local dock -------------------------------- #
    print(f'-- start local dock --')

    input_file = os.path.join(workdir, pdb_id + '.pdb')
    num_struct = docking_samples
    
    pdb_docked = os.path.join(workdir, pdb_id + '_docked.pdb')
    score_docking = os.path.join(output_docking_path, 'score.sc')

    cmd = 'mpirun -np ' + str(cpu_num) + \
            ' docking_protocol.mpi.linuxgccrelease -in:file:s ' + input_file + \
            ' -out:path:all ' + output_docking_path + \
            ' -scorefile_format json' + \
            ' -partners ' + chain + \
            ' -mute all -dock_pert 3 8 -ex1 -ex2aro -nstruct ' + str(num_struct)
    os.system(cmd)
    
    with open(score_docking, "r") as f:
        data_score = f.readlines()
    info_list = []
    score_list = []
    for data in data_score:
        info = json.loads(data)
        score = info[score_key]
        info_list.append(info)
        score_list.append(score)
    min_idx = np.argsort(np.array(score_list))[0]
    
    pdb_docked_real_name = info_list[min_idx]['decoy'] + '.pdb'
    pdb_docked_real_temp = os.path.join(output_docking_path, pdb_docked_real_name)
    
    cmd = 'cp ' + pdb_docked_real_temp + ' ' + pdb_docked
    os.system(cmd)
    
    final_score = score_list[min_idx]
    print(f'final docking score is {final_score}')
    
    return final_score


def docking_script_abbind(pdb_id, pdb_path, docking_folder, chain, score_key='total_score'):
    """
        -workdir (docking_folder)
            -PDB_id (workdir)
                -PDB_id.pdb
                -(PDB_id_relaxed.pdb)
                -(PDB_id_docked.pdb)
                -(PDB_id_refined.pdb)
                -(PDB_id_result.pdb)
                -(score.sc)
                -<temp> output_files
        -PDB_id.csv
    """
    workdir = os.path.join(docking_folder, pdb_id)
    if not os.path.exists(workdir):
        os.system('mkdir ' + workdir)
    
    command = 'cp ' + pdb_path + ' ' + workdir
    os.system(command)
    
    clear_dir_single(workdir)
    
    final_score = docking_simple_single_abbind(workdir, pdb_id, chain, score_key=score_key)
    
    return final_score



def docking_simple_single_sars(workdir, pdb_id, chain, cpu_num=100, simple_docking_samples=200, score_key='total_score'):
    print(f'-- start docking pdb:{pdb_id} ! --')
    
    output_path = os.path.join(workdir, 'output_files')
    output_docking_path = os.path.join(output_path, 'output_docking')

    # -------------------------------- local dock -------------------------------- #
    print(f'-- local dock --')

    input_file = os.path.join(workdir, pdb_id + '.pdb')
    num_struct = simple_docking_samples
    
    pdb_docked = os.path.join(workdir, pdb_id + '_docked.pdb')
    score_docking = os.path.join(output_docking_path, 'score.sc')
    
    cmd = 'mpirun -np ' + str(cpu_num) + \
            ' docking_protocol.mpi.linuxgccrelease -in:file:s ' + input_file + \
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
    
    final_score = score_list[min_idx]
    print(f'--finish SIMPLE docking pdb:{pdb_id} !--')
    print(f'final docking score is {final_score}')
    
    return final_score


def docking_script_sars(pdb_id, pdb_path, docking_folder, chain, score_key='total_score'):
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
    
    final_score = docking_simple_single_sars(workdir, pdb_id, chain, score_key=score_key)
    
    return final_score
