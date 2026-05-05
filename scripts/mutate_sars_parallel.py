import pandas as pd
import os
import sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.mutate_scripts_foldx import mut_list, Repair_wt, Gen_mut

def process_pdb(args):
    """Process mutations for a single PDB ID"""
    pdb_id, antigen_chain_id, wt_dir, fix_dir, mut_dir, mutate_type_pos_mapping = args

    wt_path = os.path.join(wt_dir, pdb_id + '.pdb')
    fix_path = os.path.join(fix_dir, pdb_id + '.pdb')

    os.chdir('protein')

    # 根据wt路径是否存在，判断是否需要修复wt结构
    if not os.path.exists(fix_path):
        Repair_wt(pdb_id, wt_path, fix_path)

    # 根据mutate_list逐个判断突变文件是否存在，若不存在则进行突变
    for key, mutate_infos in mutate_type_pos_mapping.items():
        mut_path = os.path.join(mut_dir, pdb_id + '_' + key + '.pdb')

        if os.path.exists(mut_path):
            continue

        mutate_info_new = []
        for mutate_info in mutate_infos.split('_'):
            wt = mutate_info[0]
            pos = int(mutate_info[1:-1])
            mt = mutate_info[-1]
            mutate_info_new.append(f"{wt}{antigen_chain_id}{pos}{mt}")

        mutate_info = ','.join(mutate_info_new)

        Gen_mut(pdb_id, fix_path, mutate_info, mut_path)

    os.chdir('..')
    print(f"Completed: {pdb_id}")

if __name__ == '__main__':

    data_path = "/home/lfj/projects_dir/MERF/data/SARS_COV_2/sars_evo_v5_with_resolution_hand_correct_seperate_pdb_correct_start_idx.csv"
    wt_dir = "/home/lfj/projects_dir/MERF/data/SARS_COV_2/PDBs/"
    fix_dir = "/home/lfj/projects_dir/MERF/data/SARS_COV_2/PDBs_fixed/"
    mut_dir = "/home/lfj/projects_dir/MERF/data/SARS_COV_2/PDBs_mutated/"

    mutate_type_pos_mapping = {
        'gamma': 'K417N_E484K_N501Y',
        'delta': 'L452R_T478K',
        'omicron': 'G339D_S371L_S373P_S375F_K417N_N440K_G446S_S477N_T478K_E484A_Q493R_G496S_Q498R_N501Y_Y505H'
    }

    df = pd.read_csv(data_path)

    # 准备参数列表
    args_list = [
        (df.iloc[idx]['pdb_id'], df.iloc[idx]['Antigen_chain'], wt_dir, fix_dir, mut_dir, mutate_type_pos_mapping)
        for idx in range(len(df))
    ]

    # 并行处理
    with Pool(processes=32) as pool:
        pool.map(process_pdb, args_list)

    print("All tasks completed!")
