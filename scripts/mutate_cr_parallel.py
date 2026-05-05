import pandas as pd
import os
import sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.mutate_scripts_foldx import mut_list

def process_row(args):
    """Process mutations for a single dataframe row (mutate only, repair already done)"""
    pdb_id, mutant, wt_dir, fix_dir, mut_dir = args
    mutate_info_list = [mutant]
    mut_list(pdb_id, mutate_info_list, wt_dir, fix_dir, mut_dir)
    print(f"Completed: {pdb_id} - {mutant}")

if __name__ == '__main__':
    data_path_list = [
        "/home/lfj/projects_dir/MERF/data/CR/cr6261_h1.csv",
        "/home/lfj/projects_dir/MERF/data/CR/cr6261_h9.csv",
        "/home/lfj/projects_dir/MERF/data/CR/cr9114_h1.csv",
        "/home/lfj/projects_dir/MERF/data/CR/cr9114_h3.csv"
    ]
    wt_dir = "/home/lfj/projects_dir/MERF/data/CR/PDBs/"
    fix_dir = "/home/lfj/projects_dir/MERF/data/CR/PDBs_fixed/"
    mut_dir = "/home/lfj/projects_dir/MERF/data/CR/PDBs_mutated/"

    df_list = []
    for data_path in data_path_list:
        df_list.append(pd.read_csv(data_path))
    df_all = pd.concat(df_list, ignore_index=True)

    # Stage 1: Repair all unique PDB IDs serially to avoid conflicts
    print("Stage 1: Repairing structures...")
    unique_pdb_ids = df_all['pdb_id'].unique()
    for pdb_id in unique_pdb_ids:
        mut_list(pdb_id, [], wt_dir, fix_dir, mut_dir)
        print(f"Repaired: {pdb_id}")

    # Stage 2: Parallel processing for mutations (repair already done, will be skipped)
    print("\nStage 2: Generating mutations in parallel...")
    args_list = [
        (df_all.loc[idx, 'pdb_id'], df_all.loc[idx, 'mutant'], wt_dir, fix_dir, mut_dir)
        for idx in range(len(df_all))
    ]

    with Pool(processes=32) as pool:
        pool.map(process_row, args_list)
    # for args in args_list:  # debugging: run serially
    #     process_row(args)

    print("All tasks completed!")
