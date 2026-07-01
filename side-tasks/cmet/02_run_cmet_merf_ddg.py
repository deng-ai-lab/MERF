#!/usr/bin/env python
"""Run MERF ddG prediction for nominated CMET peptide mutations."""

from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_DIR = Path(__file__).resolve().parent
PDB_DIR = SCRIPT_DIR / "pdbs"
RESULTS_DIR = SCRIPT_DIR / "results"
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from cmet_common import ensure_standard_pdb, read_structure_records, write_mutant_pdb  # noqa: E402
from dataset.dataset_ddg import DDGBaseDataset  # noqa: E402
from model.MERF_v6 import MERF  # noqa: E402
from protein.read_pdbs import PaddingCollate  # noqa: E402
from utils.util import recursive_to, seed_all  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="all", help="1011, 1012, or all")
    parser.add_argument("--pdb-dir", type=Path, default=PDB_DIR)
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--chain", default="B")
    parser.add_argument("--log-dir", type=Path, default=REPO_ROOT / "logs/pretrain_vm/2026_06_09_16_56_32")
    parser.add_argument("--checkpoint-epoch", type=int, default=200)
    parser.add_argument("--gpu-idx", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--foldx-bin", type=Path, default=Path("/home/dataset-local/software_package/foldx_20270131"))
    parser.add_argument(
        "--plm-path",
        type=Path,
        default=Path("/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D"),
    )
    parser.add_argument("--no-foldx", action="store_true", help="Use direct residue renaming instead of FoldX BuildModel")
    return parser.parse_args()


def selected_structures(value: str) -> list[str]:
    if value == "all":
        return ["1011", "1012"]
    return [item.strip() for item in value.split(",") if item.strip()]


def prepare_structures(structure_id: str, args: argparse.Namespace, candidates: pd.DataFrame) -> tuple[Path, Path, Path]:
    merf_dir = (args.results_dir / structure_id / "merf").resolve()
    wt_dir = merf_dir / "wt_pdbs"
    fix_dir = merf_dir / "fixed_pdbs"
    mut_dir = merf_dir / "mut_pdbs"
    wt_dir.mkdir(parents=True, exist_ok=True)
    fix_dir.mkdir(parents=True, exist_ok=True)
    mut_dir.mkdir(parents=True, exist_ok=True)

    src_pdb = (args.pdb_dir / f"{structure_id}.pdb").resolve()
    wt_pdb = ensure_standard_pdb(src_pdb, wt_dir / f"{structure_id}.pdb")
    fixed_pdb = fix_dir / f"{structure_id}.pdb"

    used_foldx = False
    if not args.no_foldx:
        if not args.foldx_bin.exists():
            raise FileNotFoundError(f"FoldX executable not found: {args.foldx_bin}")
        try:
            old_path = os.environ.get("PATH", "")
            os.environ["PATH"] = f"{args.foldx_bin.parent}:{old_path}"
            from protein.mutate_scripts_foldx import mut_list

            expected_mutants = [
                mut_dir / f"{structure_id}_{mutation}.pdb" for mutation in candidates["pdb_mutation"].tolist()
            ]
            used_foldx = fixed_pdb.exists() and all(path.exists() for path in expected_mutants)
            if not used_foldx:
                fixed_pdb.unlink(missing_ok=True)
                for path in expected_mutants:
                    path.unlink(missing_ok=True)

                mut_list(
                    structure_id,
                    candidates["pdb_mutation"].tolist(),
                    str(wt_dir.resolve()),
                    str(fix_dir.resolve()),
                    str(mut_dir.resolve()),
                )
                expected_mutants = [
                    mut_dir / f"{structure_id}_{mutation}.pdb" for mutation in candidates["pdb_mutation"].tolist()
                ]
                used_foldx = fixed_pdb.exists() and all(path.exists() for path in expected_mutants)
            if not used_foldx:
                missing = [str(path) for path in expected_mutants if not path.exists()]
                if not fixed_pdb.exists():
                    missing.insert(0, str(fixed_pdb))
                raise RuntimeError(f"FoldX did not create all expected files: {missing[:5]}")
        except Exception as exc:
            raise RuntimeError(f"{structure_id}: FoldX failed") from exc

    if not used_foldx:
        shutil.copyfile(wt_pdb, fixed_pdb)
        records = read_structure_records(wt_pdb)
        for row in candidates.itertuples(index=False):
            out_name = f"{structure_id}_{row.pdb_mutation}.pdb"
            write_mutant_pdb(records, args.chain, int(row.pdb_resseq), str(row.icode or " "), row.mt, mut_dir / out_name)

    return merf_dir, fix_dir, mut_dir


def run_model(input_csv: Path, fix_dir: Path, mut_dir: Path, out_csv: Path, args: argparse.Namespace) -> None:
    with (args.log_dir / "args.pkl").open("rb") as handle:
        train_args = pickle.load(handle)
    train_args.gpu_idx = args.gpu_idx
    seed_all(train_args.seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    checkpoint_path = args.log_dir / f"Epoch{args.checkpoint_epoch}_MERF.pth"
    model = MERF(train_args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = DDGBaseDataset(
        str(input_csv),
        str(fix_dir),
        str(mut_dir),
        knn_num=train_args.knn_neighbors_num,
        knn_agents_num=train_args.knn_agents_num,
        use_plm_embedding=True,
        plm_path=str(args.plm_path),
        plm_embedding_path=str(input_csv.parent / "esm2_650_embeddings.pkl"),
        device=device,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=0,
        collate_fn=PaddingCollate(),
    )

    rows = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"MERF {input_csv.parent.parent.name}"):
            batch = recursive_to(batch, device)
            pred, _, _, _ = model(batch, device)
            preds = pred.detach().cpu().view(-1).tolist()
            for pdb_id, mut, score in zip(batch["wt"]["PDB_id"], batch["wt"]["mutate_info"], preds):
                rows.append({"PDB_id": pdb_id, "pdb_mutation": mut, "merf_ddg": score})
    pd.DataFrame(rows).to_csv(out_csv, index=False)


def process_structure(structure_id: str, args: argparse.Namespace) -> None:
    candidates_csv = args.results_dir / structure_id / "intersection_candidates.csv"
    candidates = pd.read_csv(candidates_csv)
    if candidates.empty:
        raise ValueError(f"No candidates in {candidates_csv}")
    merf_dir, fix_dir, mut_dir = prepare_structures(structure_id, args, candidates)
    input_csv = merf_dir / "cmet_ddg_input.csv"
    pd.DataFrame(
        {
            "pdb_id": structure_id,
            "mutant": candidates["pdb_mutation"],
            "ddG": 0.0,
        }
    ).to_csv(input_csv, index=False)
    run_model(input_csv, fix_dir, mut_dir, merf_dir / "merf_ddg_scores.csv", args)


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_idx)
    for structure_id in selected_structures(args.structure):
        process_structure(structure_id, args)


if __name__ == "__main__":
    main()
