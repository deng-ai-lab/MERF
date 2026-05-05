import pandas as pd
import os
import sys
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.mutate_scripts_foldx import mut_list

def process_pdb(args):
    """Process mutations for a single PDB ID"""
    pdb_id, df_pdb, wt_dir, fix_dir, mut_dir = args

    pdb_id = pdb_id.replace('+', '')
    pdb_id = pdb_id.replace('.00', '')

    mutate_info_list = df_pdb['mutant'].tolist()
    mut_list(pdb_id, mutate_info_list, wt_dir, fix_dir, mut_dir)
    print(f"Completed: {pdb_id}")

if __name__ == '__main__':

    data_path = "/home/lfj/projects_dir/MERF/data/SKEMPIv2/SKEMPIv2.csv"
    wt_dir = "/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs/"
    fix_dir = "/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs_fixed/"
    mut_dir = "/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs_mutated/"

    df = pd.read_csv(data_path)
    unique_pdb_ids = df['pdb_id'].unique()

    # Prepare arguments for each PDB ID
    args_list = [
        (pdb_id, df[df['pdb_id'] == pdb_id], wt_dir, fix_dir, mut_dir)
        for pdb_id in unique_pdb_ids
    ]

    # Parallel processing
    with Pool(processes=8) as pool:
        pool.map(process_pdb, args_list)

    print("All tasks completed!")
