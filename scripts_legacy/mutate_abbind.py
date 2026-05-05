import pandas as pd
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.mutate_scripts_foldx import mut_list

if __name__ == '__main__':

    data_path = "/home/lfj/projects_dir/MERF/data/ABbind/AB-Bind_645pMulti.csv"
    wt_dir = "/home/lfj/projects_dir/MERF/data/ABbind/PDBs"
    fix_dir = "/home/lfj/projects_dir/MERF/data/ABbind/PDBs_fixed"
    mut_dir = "/home/lfj/projects_dir/MERF/data/ABbind/PDBs_mutated"

    df = pd.read_csv(data_path)
    unique_pdb_ids = df['pdb_id'].unique()

    for pdb_id in unique_pdb_ids[0:1]:  # debug only
    # for pdb_id in unique_pdb_ids:
        df_pdb = df[df['pdb_id'] == pdb_id]
        mutate_infos = df_pdb['mutant'].tolist()

        mutate_info_new_list = []
        for mutate_info in mutate_infos:
            # format like "D:L483T"

            mutate_info_new = []

            for mut in mutate_info.split(','):
                chain_id = mut[0]
                wtname = mut[2]
                resid = str(int(mut[3:-1]))
                mutname = mut[-1]
                mutate_info_reformat = f"{wtname}{chain_id}{resid}{mutname}"
                mutate_info_new.append(mutate_info_reformat)
            mutate_info_new = ','.join(mutate_info_new)

            mutate_info_new_list.append(mutate_info_new)
        
        mut_list(pdb_id, mutate_info_new_list, wt_dir, fix_dir, mut_dir)
