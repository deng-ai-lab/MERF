import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.mutate_scripts_foldx import mut_list

if __name__ == '__main__':

    data_path = "/home/lfj/projects_dir/MERF/data/SKEMPIv2/SKEMPIv2.csv"
    wt_dir = "/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs/"
    fix_dir = "/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs_fixed/"
    mut_dir = "/home/lfj/projects_dir/MERF/data/SKEMPIv2/PDBs_mutated/"

    df = pd.read_csv(data_path)

    unique_pdb_ids = df['pdb_id'].unique()

    for pdb_id in unique_pdb_ids[0:1]:  # debug only
        df_pdb = df[df['pdb_id'] == pdb_id]
        mutate_info_list = df_pdb['mutant'].tolist()
        
        mut_list(pdb_id, mutate_info_list, wt_dir, fix_dir, mut_dir)
