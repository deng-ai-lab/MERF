#!/usr/bin/env python
"""Run AF2-Multimer scoring for top plectin1 multipoint combinations by MERF ddG."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
os.environ.setdefault("JAX_PLATFORMS", "cuda")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-colabdesign")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(SCRIPT_DIR))

from calculate_colabdesign import interface_metrics_from_pae  # noqa: E402
from colabdesign import mk_afdesign_model  # noqa: E402
from plectin1_common import (  # noqa: E402
    chain_ids,
    chain_residues,
    ensure_standard_pdb,
    find_structure_file,
    read_structure_records,
    sequence_from_residues,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="all", help="ptp or all")
    parser.add_argument("--pdb-dir", type=Path, default=SCRIPT_DIR / "pdbs")
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--chain", default="B")
    parser.add_argument("--target-chains", default=None)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--af2-params", type=Path, default=Path("/home/dataset-local/software_package/ColabDesign/params"))
    parser.add_argument("--model-index", type=int, default=0)
    parser.add_argument("--recycles", type=int, default=3)
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def selected_structures(value: str) -> list[str]:
    if value == "all":
        return ["ptp"]
    return [item.strip() for item in value.split(",") if item.strip()]


def chain_lengths(records: list[dict], chains: list[str]) -> dict[str, int]:
    return {chain: len(chain_residues(records, chain)) for chain in chains}


def scalar_log(aux_log: dict) -> dict[str, float]:
    return {key: float(np.asarray(value)) for key, value in aux_log.items() if np.asarray(value).ndim == 0}


def score_sequences(
    pdb_path: Path,
    target_chains: list[str],
    binder_chain: str,
    binder_lengths: dict[str, int],
    target_lengths: dict[str, int],
    sequence_rows: list[dict],
    args: argparse.Namespace,
    out_dir: Path,
) -> pd.DataFrame:
    if args.dry_run:
        return pd.DataFrame([{**row, "dry_run": True} for row in sequence_rows])

    model = mk_afdesign_model(
        protocol="binder",
        num_recycles=args.recycles,
        data_dir=str(args.af2_params),
        use_multimer=True,
        use_initial_guess=True,
        use_initial_atom_pos=False,
    )
    model.prep_inputs(
        pdb_filename=str(pdb_path),
        chain=",".join(target_chains),
        binder_chain=binder_chain,
        binder_len=binder_lengths[binder_chain],
        use_binder_template=True,
        rm_target_seq=False,
        rm_target_sc=False,
        rm_template_ic=True,
    )

    rows = []
    for row in sequence_rows:
        started = time.time()
        model.predict(seq=row["binder_sequence"], models=[args.model_index], num_recycles=args.recycles, verbose=False)
        log = scalar_log(model.aux["log"])
        pae = np.asarray(model.aux["pae"], dtype=float)
        metrics = interface_metrics_from_pae(
            pae=pae,
            target_lengths=target_lengths,
            binder_lengths=binder_lengths,
            primary_binder_chain=binder_chain,
        )
        primary_metrics = metrics[f"{binder_chain}_vs_{''.join(target_chains)}"]
        plddt = np.asarray(model.aux["plddt"], dtype=float)
        chain_masks = metrics.pop("chain_masks")
        binder_plddt = float(plddt[chain_masks[binder_chain].astype(bool)].mean())
        out_pdb = out_dir / "af2_pdbs" / f"{row['candidate_id']}_model{args.model_index + 1}.pdb"
        out_pdb.parent.mkdir(parents=True, exist_ok=True)
        model.save_pdb(str(out_pdb))
        rows.append(
            {
                **row,
                "af2_pLDDT": round(float(log["plddt"]), 6),
                "af2_binder_pLDDT": round(binder_plddt, 6),
                "af2_i_pAE": primary_metrics["i_pAE"],
                "af2_i_pAE_A": primary_metrics["i_pAE_A"],
                "af2_i_pTM": round(float(log["i_ptm"]), 6),
                "af2_min_ipAE_A": primary_metrics["min_ipAE_A"],
                "af2_runtime_sec": round(time.time() - started, 2),
                "af2_pdb_path": str(out_pdb),
            }
        )
        print(f"AF2 multipoint {row['candidate_id']} done in {rows[-1]['af2_runtime_sec']} sec", flush=True)
    return pd.DataFrame(rows)


def process_structure(structure_id: str, args: argparse.Namespace) -> pd.DataFrame:
    base = args.results_dir / structure_id
    combo_dir = base / "multipoint"
    af2_dir = base / "multipoint_af2"
    af2_dir.mkdir(parents=True, exist_ok=True)

    input_dir = base / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    std_pdb = ensure_standard_pdb(find_structure_file(args.pdb_dir, structure_id), input_dir / f"{structure_id}.pdb")
    records = read_structure_records(std_pdb)
    all_chains = chain_ids(records)
    target_chains = [c.strip() for c in args.target_chains.split(",") if c.strip()] if args.target_chains else [c for c in all_chains if c != args.chain]
    binder_lengths = chain_lengths(records, [args.chain])
    target_lengths = chain_lengths(records, target_chains)
    wt_sequence = sequence_from_residues(chain_residues(records, args.chain))

    candidates = pd.read_csv(combo_dir / "multipoint_candidates_with_merf.csv", keep_default_na=False)
    top = candidates.sort_values(["merf_ddg", "combo_size", "combo_combined_rank_sum"]).head(args.top_n).copy()
    top.to_csv(combo_dir / "multipoint_top20_by_merf.csv", index=False)

    rows = [
        {
            "structure": structure_id,
            "candidate_id": f"{structure_id}_WT",
            "mutation": "WT",
            "pdb_mutation": "WT",
            "binder_sequence": wt_sequence,
            "is_wt": True,
        }
    ]
    for row in top.itertuples(index=False):
        rows.append(
            {
                "structure": structure_id,
                "candidate_id": row.candidate_id,
                "mutation": row.mutation,
                "pdb_mutation": row.pdb_mutation,
                "binder_sequence": row.mutant_sequence,
                "is_wt": False,
            }
        )

    meta = {
        "structure": structure_id,
        "standard_pdb": str(std_pdb),
        "binder_chain": args.chain,
        "target_chains": target_chains,
        "top_n": args.top_n,
        "model_index": args.model_index,
        "recycles": args.recycles,
    }
    (af2_dir / "af2_input_summary.json").write_text(json.dumps(meta, indent=2) + "\n")
    scores = score_sequences(std_pdb, target_chains, args.chain, binder_lengths, target_lengths, rows, args, af2_dir)
    scores.to_csv(af2_dir / "af2_scores.csv", index=False)
    if not scores.empty and "dry_run" not in scores.columns:
        wt = scores[scores["is_wt"]].iloc[0]
        mut = scores[~scores["is_wt"]].copy()
        for col in ["af2_pLDDT", "af2_binder_pLDDT", "af2_i_pAE", "af2_i_pAE_A", "af2_i_pTM", "af2_min_ipAE_A"]:
            mut[f"delta_{col}"] = mut[col] - wt[col]
        mut.to_csv(af2_dir / "af2_delta_scores.csv", index=False)
        final = top.merge(
            mut.drop(columns=["structure", "mutation"], errors="ignore"),
            on=["candidate_id", "pdb_mutation"],
            how="left",
        )
    else:
        final = top
    final.to_csv(combo_dir / "multipoint_final_top20.csv", index=False)
    print(f"{structure_id}: wrote {combo_dir / 'multipoint_final_top20.csv'} ({len(final)} rows)")
    return final


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    frames = [process_structure(structure_id, args) for structure_id in selected_structures(args.structure)]
    if frames:
        all_df = pd.concat(frames, ignore_index=True)
        all_df.to_csv(args.results_dir / "multipoint_final_top20_all.csv", index=False)
        print(f"wrote {args.results_dir / 'multipoint_final_top20_all.csv'} ({len(all_df)} rows)")


if __name__ == "__main__":
    main()
