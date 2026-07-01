#!/usr/bin/env python
"""Merge CMET nomination, MERF ddG, and AF2 scores into final tables."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="all", help="1011, 1012, or all")
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    return parser.parse_args()


def selected_structures(value: str) -> list[str]:
    if value == "all":
        return ["1011", "1012"]
    return [item.strip() for item in value.split(",") if item.strip()]


def merge_structure(structure_id: str, results_dir: Path) -> pd.DataFrame:
    base = results_dir / structure_id
    nomination = pd.read_csv(base / "nomination_scores_all.csv")
    candidates = nomination[nomination.get("in_relaxed_intersection", False)].copy()
    merf_path = base / "merf" / "merf_ddg_scores.csv"
    af2_path = base / "af2" / "af2_delta_scores.csv"
    if merf_path.exists():
        merf = pd.read_csv(merf_path)
        candidates = candidates.merge(merf[["pdb_mutation", "merf_ddg"]], on="pdb_mutation", how="left")
    if af2_path.exists():
        af2 = pd.read_csv(af2_path)
        keep = [
            "pdb_mutation",
            "af2_pLDDT",
            "af2_binder_pLDDT",
            "af2_i_pAE",
            "af2_i_pAE_A",
            "af2_i_pTM",
            "af2_min_ipAE_A",
            "delta_af2_pLDDT",
            "delta_af2_binder_pLDDT",
            "delta_af2_i_pAE",
            "delta_af2_i_pAE_A",
            "delta_af2_i_pTM",
            "delta_af2_min_ipAE_A",
            "af2_pdb_path",
        ]
        candidates = candidates.merge(af2[[col for col in keep if col in af2.columns]], on="pdb_mutation", how="left")
    candidates = candidates.sort_values(["combined_rank_sum", "esm1v_rank", "if_rank"])
    candidates.to_csv(base / "final_results.csv", index=False)
    return candidates


def main() -> None:
    args = parse_args()
    all_frames = []
    for structure_id in selected_structures(args.structure):
        frame = merge_structure(structure_id, args.results_dir)
        all_frames.append(frame)
        print(f"{structure_id}: wrote {args.results_dir / structure_id / 'final_results.csv'} ({len(frame)} rows)")
    if all_frames:
        all_df = pd.concat(all_frames, ignore_index=True)
        all_df.to_csv(args.results_dir / "final_results_all.csv", index=False)
        print(f"wrote {args.results_dir / 'final_results_all.csv'} ({len(all_df)} rows)")


if __name__ == "__main__":
    main()
