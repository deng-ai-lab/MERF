import pandas as pd
import os
import sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.mutate_scripts_foldx import mut_list

def process_pdb(args):
    """Process mutations for a single PDB ID"""
    pdb_id, df_pdb, wt_dir, fix_dir, mut_dir = args

    mutate_infos = df_pdb['mutant'].tolist()
    chain_ids = df_pdb['chain_id'].tolist()

    mutate_info_new_list = []
    for i in range(len(mutate_infos)):
        mutate_info = mutate_infos[i]
        chain_id = chain_ids[i]

        mutate_info_new = []
        for mut in mutate_info.split(','):
            wtname = mut[0]
            resid = str(int(mut[1:-1]))
            mutname = mut[-1]
            mutate_info_reformat = f"{wtname}{chain_id}{resid}{mutname}"
            mutate_info_new.append(mutate_info_reformat)
        mutate_info_new = ','.join(mutate_info_new)
        mutate_info_new_list.append(mutate_info_new)

    mut_list(pdb_id, mutate_info_new_list, wt_dir, fix_dir, mut_dir)
    print(f"Completed: {pdb_id}")

if __name__ == '__main__':
    data_dir = "/home/lfj/projects_dir/MERF/data/VenusMaxwell"
    wt_dir = "/home/lfj/projects_dir/MERF/data/VenusMaxwell/PDBs"
    fix_dir = "/home/lfj/projects_dir/MERF/data/VenusMaxwell/PDBs_fixed"
    mut_dir = "/home/lfj/projects_dir/MERF/data/VenusMaxwell/PDBs_mutated"

    # Combine all splits
    dfs = []
    for split in ['train', 'valid', 'test']:
        data_path = os.path.join(data_dir, f"{split}_data.csv")
        dfs.append(pd.read_csv(data_path))

    df_all = pd.concat(dfs, ignore_index=True)
    unique_pdb_ids = df_all['pdb_id'].unique()

    # Prepare arguments for each PDB ID
    args_list = [
        (pdb_id, df_all[df_all['pdb_id'] == pdb_id], wt_dir, fix_dir, mut_dir)
        for pdb_id in unique_pdb_ids
    ]

    # Parallel processing
    with Pool(processes=32) as pool:
        pool.map(process_pdb, args_list)

    print("All tasks completed!")
