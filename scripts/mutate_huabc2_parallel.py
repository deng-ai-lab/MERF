import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.mutate_scripts_foldx import mut_list


def process_pdb(args):
    pdb_id, mutations, wt_dir, fix_dir, mut_dir = args
    mut_list(pdb_id, mutations, wt_dir, fix_dir, mut_dir)
    print(f"Completed: {pdb_id}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="/home/dataset-local/projects_dir/MERF/data/HuABC2/HuABC2_test.csv",
    )
    parser.add_argument(
        "--wt_dir",
        type=str,
        default="/home/dataset-local/projects_dir/MERF/data/HuABC2/PDBs",
    )
    parser.add_argument(
        "--fix_dir",
        type=str,
        default="/home/dataset-local/projects_dir/MERF/data/HuABC2/PDBs_fixed",
    )
    parser.add_argument(
        "--mut_dir",
        type=str,
        default="/home/dataset-local/projects_dir/MERF/data/HuABC2/PDBs_mutated",
    )
    parser.add_argument("--foldx_dir", type=str, default="/home/dataset-local/software_package")
    parser.add_argument("--processes", type=int, default=1)
    args = parser.parse_args()

    if args.foldx_dir:
        os.environ["PATH"] = args.foldx_dir + os.pathsep + os.environ.get("PATH", "")

    os.makedirs(args.fix_dir, exist_ok=True)
    os.makedirs(args.mut_dir, exist_ok=True)

    df = pd.read_csv(args.data_path, dtype={"pdb_id": "string"})
    df["pdb_id_base"] = df["pdb_id"].map(lambda value: os.path.splitext(os.path.basename(str(value)))[0])

    args_list = []
    for pdb_id, df_pdb in df.groupby("pdb_id_base"):
        mutations = df_pdb["mutation"].drop_duplicates().tolist()
        args_list.append((pdb_id, mutations, args.wt_dir, args.fix_dir, args.mut_dir))

    with Pool(processes=args.processes) as pool:
        pool.map(process_pdb, args_list)

    print("All tasks completed!")


if __name__ == "__main__":
    main()
