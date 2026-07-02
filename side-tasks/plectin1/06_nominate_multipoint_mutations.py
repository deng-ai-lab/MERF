#!/usr/bin/env python
"""Nominate plectin1 2-point and 3-point mutation combinations from single-mutant model ranks."""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="all", help="ptp or all")
    parser.add_argument("--results-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--combo-sizes", default="2,3")
    parser.add_argument("--extra-pdb-mutations", default="TB6F,KB1M,TB6I,TB6L")
    return parser.parse_args()


def selected_structures(value: str) -> list[str]:
    if value == "all":
        return ["ptp"]
    return [item.strip() for item in value.split(",") if item.strip()]


def load_full_single_scores(base: Path) -> pd.DataFrame:
    esm = pd.read_csv(base / "esm1v_scores.csv", keep_default_na=False)
    inv = pd.read_csv(base / "inverse_folding_scores.csv", keep_default_na=False)
    inv_cols = [
        "mutation",
        "if_log_likelihood",
        "if_log_likelihood_target",
        "if_wt_log_likelihood",
        "if_wt_log_likelihood_target",
        "if_delta_log_likelihood",
        "if_score",
    ]
    scored = esm.merge(inv[inv_cols], on="mutation", how="inner")
    scored["esm1v_rank"] = scored["esm1v_score"].rank(ascending=False, method="min")
    scored["if_rank"] = scored["if_score"].rank(ascending=False, method="min")
    scored["combined_rank_sum"] = scored["esm1v_rank"] + scored["if_rank"]
    return scored.sort_values(["combined_rank_sum", "esm1v_rank", "if_rank", "mutation"])


def apply_mutations(sequence: str, rows: list[pd.Series]) -> str:
    chars = list(sequence)
    for row in rows:
        idx = int(row["seq_pos"]) - 1
        if chars[idx] != row["wt"]:
            raise ValueError(f"WT mismatch at position {row['seq_pos']}: expected {chars[idx]}, row has {row['wt']}")
        chars[idx] = row["mt"]
    return "".join(chars)


def wildtype_sequence_from_row(row: pd.Series) -> str:
    chars = list(str(row["mutant_sequence"]))
    chars[int(row["seq_pos"]) - 1] = str(row["wt"])
    return "".join(chars)


def best_single_per_position(scored: pd.DataFrame) -> pd.DataFrame:
    return (
        scored.sort_values(["seq_pos", "combined_rank_sum", "esm1v_rank", "if_rank", "mutation"])
        .groupby("seq_pos", as_index=False)
        .head(1)
        .copy()
        .sort_values("seq_pos")
    )


def add_extra_singles(singles: pd.DataFrame, scored: pd.DataFrame, extra_pdb_mutations: list[str]) -> pd.DataFrame:
    frames = [singles.assign(single_source="best_per_position")]
    extras = []
    for pdb_mutation in extra_pdb_mutations:
        match = scored[scored["pdb_mutation"] == pdb_mutation]
        if match.empty:
            raise ValueError(f"Extra mutation {pdb_mutation} was not found in score tables")
        row = match.iloc[0].copy()
        row["single_source"] = "user_extra"
        extras.append(row)
    if extras:
        frames.append(pd.DataFrame(extras))
    merged = pd.concat(frames, ignore_index=True)
    merged = merged.drop_duplicates("pdb_mutation", keep="first")
    return merged.sort_values(["seq_pos", "single_source", "combined_rank_sum", "pdb_mutation"])


def make_combo_row(structure_id: str, combo: tuple[pd.Series, ...]) -> dict:
    rows = sorted(combo, key=lambda row: int(row["seq_pos"]))
    pdb_mutation = ",".join(str(row["pdb_mutation"]) for row in rows)
    mutation = ",".join(str(row["mutation"]) for row in rows)
    sequence = apply_mutations(wildtype_sequence_from_row(rows[0]), rows)
    return {
        "structure": structure_id,
        "combo_size": len(rows),
        "combo_positions": ",".join(str(int(row["seq_pos"])) for row in rows),
        "mutation": mutation,
        "pdb_mutation": pdb_mutation,
        "candidate_id": f"{structure_id}_{pdb_mutation.replace(',', '__')}",
        "mutant_sequence": sequence,
        "component_sources": ",".join(str(row["single_source"]) for row in rows),
        "component_chains": ",".join(str(row["chain"]) for row in rows),
        "component_seq_pos": ",".join(str(int(row["seq_pos"])) for row in rows),
        "component_pdb_resseqs": ",".join(str(int(row["pdb_resseq"])) for row in rows),
        "component_icodes": ",".join(str(row["icode"]) for row in rows),
        "component_wts": ",".join(str(row["wt"]) for row in rows),
        "component_mts": ",".join(str(row["mt"]) for row in rows),
        "component_esm1v_ranks": ",".join(f"{float(row['esm1v_rank']):.0f}" for row in rows),
        "component_if_ranks": ",".join(f"{float(row['if_rank']):.0f}" for row in rows),
        "component_combined_rank_sums": ",".join(f"{float(row['combined_rank_sum']):.0f}" for row in rows),
        "combo_combined_rank_sum": sum(float(row["combined_rank_sum"]) for row in rows),
        "combo_esm1v_rank_sum": sum(float(row["esm1v_rank"]) for row in rows),
        "combo_if_rank_sum": sum(float(row["if_rank"]) for row in rows),
    }


def nominate_structure(structure_id: str, args: argparse.Namespace) -> None:
    base = args.results_dir / structure_id
    out_dir = base / "multipoint"
    out_dir.mkdir(parents=True, exist_ok=True)
    scored = load_full_single_scores(base)
    extra_pdb_mutations = [item.strip() for item in args.extra_pdb_mutations.split(",") if item.strip()]
    singles = add_extra_singles(best_single_per_position(scored), scored, extra_pdb_mutations)
    singles.to_csv(out_dir / "multipoint_single_candidates.csv", index=False)

    combo_sizes = [int(item.strip()) for item in args.combo_sizes.split(",") if item.strip()]
    combo_rows = []
    single_rows = [row for _, row in singles.iterrows()]
    for size in combo_sizes:
        for combo in itertools.combinations(single_rows, size):
            positions = [int(row["seq_pos"]) for row in combo]
            if len(set(positions)) != len(positions):
                continue
            combo_rows.append(make_combo_row(structure_id, combo))
    combos = pd.DataFrame(combo_rows).sort_values(
        ["combo_size", "combo_combined_rank_sum", "combo_esm1v_rank_sum", "combo_if_rank_sum", "pdb_mutation"]
    )
    combos.to_csv(out_dir / "multipoint_candidates.csv", index=False)
    print(f"{structure_id}: singles={len(singles)}, combinations={len(combos)}")


def main() -> None:
    args = parse_args()
    for structure_id in selected_structures(args.structure):
        nominate_structure(structure_id, args)


if __name__ == "__main__":
    main()
