import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.mutate_scripts_foldx import mut_list

if __name__ == '__main__':

    data_path = "/home/lfj/projects_dir/MERF/data/7FAE/7FAE.csv"
    wt_dir = "/home/lfj/projects_dir/MERF/data/7FAE/PDBs/"
    fix_dir = "/home/lfj/projects_dir/MERF/data/7FAE/PDBs_fixed/"
    mut_dir = "/home/lfj/projects_dir/MERF/data/7FAE/PDBs_mutated/"

    df = pd.read_csv(data_path)
    for idx in range(len(df)):
        pdb_id = df.iloc[idx]['pdb_id']
        mutate_info_list = [df.iloc[idx]['mutant']]
        
        mut_list(pdb_id, mutate_info_list, wt_dir, fix_dir, mut_dir)
