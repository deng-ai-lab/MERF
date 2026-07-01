#!/usr/bin/env python
"""Summarize plectin1 mutation evolution results without manual filtering."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path(__file__).resolve().parent / "results")
    return parser.parse_args()


def add_rank_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["rank_merf_ddg_low"] = out.groupby("structure")["merf_ddg"].rank(method="min", ascending=True).astype(int)
    out["rank_delta_plddt_high"] = out.groupby("structure")["delta_af2_pLDDT"].rank(method="min", ascending=False).astype(int)
    out["rank_delta_ipae_low"] = out.groupby("structure")["delta_af2_i_pAE"].rank(method="min", ascending=True).astype(int)
    out["rank_delta_ipae_A_low"] = out.groupby("structure")["delta_af2_i_pAE_A"].rank(method="min", ascending=True).astype(int)
    out["summary_rank_sum"] = (
        out["rank_merf_ddg_low"]
        + out["rank_delta_plddt_high"]
        + out["rank_delta_ipae_low"]
        + out["rank_delta_ipae_A_low"]
    )
    out["summary_rank"] = out.groupby("structure")["summary_rank_sum"].rank(method="min", ascending=True).astype(int)
    return out.sort_values(["structure", "summary_rank", "summary_rank_sum", "mutation"])


def write_markdown(ranked: pd.DataFrame, out_path: Path) -> None:
    lines = [
        "# Plectin1 mutation evolution summary",
        "",
        "This report is generated directly from `final_results_all.csv`.",
        "Lower `merf_ddg`, lower AF2 iPAE deltas, and higher AF2 pLDDT delta are ranked favorably.",
        "No candidates are manually filtered in this report.",
        "",
    ]
    for structure, group in ranked.groupby("structure", sort=True):
        lines.extend(
            [
                f"## {structure}",
                "",
                f"- candidates: {len(group)}",
                f"- MERF ddG min/median/max: {group['merf_ddg'].min():.4f} / {group['merf_ddg'].median():.4f} / {group['merf_ddg'].max():.4f}",
                f"- delta AF2 pLDDT min/median/max: {group['delta_af2_pLDDT'].min():.4f} / {group['delta_af2_pLDDT'].median():.4f} / {group['delta_af2_pLDDT'].max():.4f}",
                f"- delta AF2 iPAE_A min/median/max: {group['delta_af2_i_pAE_A'].min():.4f} / {group['delta_af2_i_pAE_A'].median():.4f} / {group['delta_af2_i_pAE_A'].max():.4f}",
                "",
                "Top rows by aggregate rank:",
                "",
            ]
        )
        cols = [
            "summary_rank",
            "mutation",
            "pdb_mutation",
            "merf_ddg",
            "delta_af2_pLDDT",
            "delta_af2_i_pAE",
            "delta_af2_i_pAE_A",
            "esm1v_rank",
            "if_rank",
        ]
        lines.extend(markdown_table(group[cols].head(15)))
        lines.append("")
    out_path.write_text("\n".join(lines), encoding="utf-8")


def format_cell(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def markdown_table(df: pd.DataFrame) -> list[str]:
    rows = [[format_cell(value) for value in row] for row in df.itertuples(index=False, name=None)]
    headers = list(df.columns)
    widths = [
        max(len(str(header)), *(len(row[idx]) for row in rows)) if rows else len(str(header))
        for idx, header in enumerate(headers)
    ]
    out = [
        "| " + " | ".join(str(header).ljust(widths[idx]) for idx, header in enumerate(headers)) + " |",
        "| " + " | ".join("-" * widths[idx] for idx in range(len(headers))) + " |",
    ]
    for row in rows:
        out.append("| " + " | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))) + " |")
    return out


def main() -> None:
    args = parse_args()
    in_csv = args.results_dir / "final_results_all.csv"
    if not in_csv.exists():
        raise FileNotFoundError(in_csv)
    df = pd.read_csv(in_csv)
    ranked = add_rank_columns(df)
    ranked.to_csv(args.results_dir / "final_results_all_ranked.csv", index=False)
    write_markdown(ranked, args.results_dir / "summary_report.md")
    print(f"wrote {args.results_dir / 'final_results_all_ranked.csv'} ({len(ranked)} rows)")
    print(f"wrote {args.results_dir / 'summary_report.md'}")


if __name__ == "__main__":
    main()
