import argparse
import os
import sys
from multiprocessing import Pool

import pandas as pd

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from protein.mutate_scripts_foldx import mut_list


def find_single_file(folder, suffix):
    matches = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one *{suffix} in {folder}, found {matches}")
    return matches[0]


def find_landscape_csv(folder, pdb_id):
    preferred = os.path.join(folder, f"{pdb_id}.csv")
    if os.path.exists(preferred):
        return preferred
    matches = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(".csv") and not name.endswith("_evo.csv")
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one landscape csv in {folder}, found {matches}")
    return matches[0]


def process_row(args):
    pdb_id, mutant, wt_dir, fix_dir, mut_dir = args
    os.chdir(PROJECT_DIR)
    mut_list(pdb_id, [mutant], wt_dir, fix_dir, mut_dir)
    print(f"Completed: {pdb_id} - {mutant}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evo_folder", type=str, default="/home/dataset-local/projects_dir/MERF/data/CR_evo/cr6261_h9_PA28T,PA62Q,FA75S,GA77S,VA79A", help="Path to one CR start-point evolution folder under data/CR_evo")
    parser.add_argument("--processes", type=int, default=32)
    parser.add_argument("--dry_run", action="store_true", help="Only print planned FoldX tasks")
    args = parser.parse_args()

    evo_folder = os.path.abspath(args.evo_folder)
    if not os.path.isdir(evo_folder):
        raise NotADirectoryError(evo_folder)

    evo_csv = find_single_file(evo_folder, "_evo.csv")
    evo_df = pd.read_csv(evo_csv, dtype={"pdb_id": "string"})
    if len(evo_df) != 1:
        raise ValueError(f"Expected one row in {evo_csv}, found {len(evo_df)}")

    pdb_id = os.path.basename(evo_folder.rstrip(os.sep))
    wt_dir = os.path.join(evo_folder, "PDBs")
    fix_dir = os.path.join(evo_folder, "PDBs_fixed")
    mut_dir = os.path.join(evo_folder, "PDBs_mutated")
    os.makedirs(fix_dir, exist_ok=True)
    os.makedirs(mut_dir, exist_ok=True)

    landscape_csv = find_landscape_csv(evo_folder, pdb_id)
    landscape_df = pd.read_csv(landscape_csv, dtype={"pdb_id": "string"})
    if "mutant" not in landscape_df.columns:
        raise ValueError(f"'mutant' column not found in {landscape_csv}")

    mutants = [
        str(mutant).strip()
        for mutant in landscape_df["mutant"].fillna("").tolist()
        if str(mutant).strip() not in ("", "nan")
    ]
    mutants = list(dict.fromkeys(mutants))

    print(f"Evo folder: {evo_folder}")
    print(f"Evo csv: {evo_csv}")
    print(f"Landscape csv: {landscape_csv}")
    print(f"PDB ID: {pdb_id}")
    print(f"Mutation tasks: {len(mutants)}")

    if args.dry_run:
        for mutant in mutants[:10]:
            print(f"DRY_RUN {pdb_id}: {mutant}")
        if len(mutants) > 10:
            print(f"... {len(mutants) - 10} more")
        return

    os.chdir(PROJECT_DIR)
    print("Stage 1: Repairing start-point structure...")
    mut_list(pdb_id, [], wt_dir, fix_dir, mut_dir)
    print(f"Repaired: {pdb_id}")

    print("Stage 2: Generating relative mutations in parallel...")
    args_list = [(pdb_id, mutant, wt_dir, fix_dir, mut_dir) for mutant in mutants]
    with Pool(processes=args.processes) as pool:
        pool.map(process_row, args_list)

    print("All tasks completed!")


if __name__ == "__main__":
    main()
