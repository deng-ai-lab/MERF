#!/usr/bin/env python
"""Nominate plectin1 peptide mutations with ESM-1v and ESM-IF1."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")

import pandas as pd
import torch
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(REPO_ROOT / "baselines" / "esm"))
sys.path.insert(0, str(REPO_ROOT / "baselines" / "structural-evolution" / "bin"))

from plectin1_common import (  # noqa: E402
    AA20,
    all_single_mutations,
    chain_residues,
    ensure_standard_pdb,
    find_structure_file,
    mutate_sequence,
    read_structure_records,
    sequence_from_residues,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--structure", default="all", help="ptp or all")
    parser.add_argument("--pdb-dir", type=Path, default=SCRIPT_DIR / "pdbs")
    parser.add_argument("--chain", default="B")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--gpu", default="2")
    parser.add_argument("--top-n", type=int, default=50)
    parser.add_argument("--min-intersection", type=int, default=30)
    parser.add_argument("--max-relaxed-top-n", type=int, default=400)
    parser.add_argument("--esm-model", default="esm1v_t33_650M_UR90S_1")
    parser.add_argument(
        "--if-model",
        type=Path,
        default=Path("/home/batchcom/.cache/torch/hub/checkpoints/esm_if1_20220410.pt"),
        help="Local ESM-IF1 checkpoint path. Defaults to the cached checkpoint on this machine.",
    )
    parser.add_argument("--skip-esm", action="store_true")
    parser.add_argument("--skip-if", action="store_true")
    return parser.parse_args()


def selected_structures(value: str) -> list[str]:
    if value == "all":
        return ["ptp"]
    return [item.strip() for item in value.split(",") if item.strip()]


def score_with_esm1v(sequence: str, mutations: pd.DataFrame, model_name: str, device: torch.device) -> pd.DataFrame:
    import esm

    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    _, _, tokens = batch_converter([("peptide", sequence)])
    tokens = tokens.to(device)

    rows = []
    with torch.no_grad():
        for seq_pos in tqdm(range(1, len(sequence) + 1), desc="ESM-1v masked positions"):
            masked = tokens.clone()
            masked[0, seq_pos] = alphabet.mask_idx
            logits = model(masked)["logits"]
            log_probs = torch.log_softmax(logits, dim=-1)[0, seq_pos]
            wt = sequence[seq_pos - 1]
            wt_score = log_probs[alphabet.get_idx(wt)].item()
            for mt in AA20:
                if mt == wt:
                    continue
                rows.append(
                    {
                        "seq_pos": seq_pos,
                        "wt": wt,
                        "mt": mt,
                        "esm1v_log_prob_mt": log_probs[alphabet.get_idx(mt)].item(),
                        "esm1v_log_prob_wt": wt_score,
                        "esm1v_score": log_probs[alphabet.get_idx(mt)].item() - wt_score,
                    }
                )
    scored = pd.DataFrame(rows)
    return mutations.merge(scored, on=["seq_pos", "wt", "mt"], how="left")


def score_with_if1(
    pdb_path: Path,
    sequence: str,
    mutations: pd.DataFrame,
    chain: str,
    device: torch.device,
    if_model: Path,
) -> pd.DataFrame:
    import esm
    from score_log_likelihoods import extract_coords_from_complex, load_structure, score_sequence_in_complex

    model, alphabet = esm.pretrained.load_model_and_alphabet(str(if_model))
    model = model.eval().to(device)
    structure = load_structure(str(pdb_path))
    coords, native_seqs = extract_coords_from_complex(structure)
    wt_ll, wt_ll_target = score_sequence_in_complex(model, alphabet, coords, native_seqs, chain, sequence)

    rows = []
    for row in tqdm(mutations.itertuples(index=False), total=len(mutations), desc="ESM-IF1 mutants"):
        mutant_sequence = mutate_sequence(sequence, int(row.seq_pos), row.mt)
        ll, ll_target = score_sequence_in_complex(model, alphabet, coords, native_seqs, chain, mutant_sequence)
        rows.append(
            {
                "mutation": row.mutation,
                "if_log_likelihood": ll,
                "if_log_likelihood_target": ll_target,
                "if_wt_log_likelihood": wt_ll,
                "if_wt_log_likelihood_target": wt_ll_target,
                "if_delta_log_likelihood": ll - wt_ll,
                "if_score": ll_target - wt_ll_target,
            }
        )
    return mutations.merge(pd.DataFrame(rows), on="mutation", how="left")


def rank_and_select(df: pd.DataFrame, top_n: int, min_intersection: int, max_relaxed_top_n: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ranked = df.copy()
    ranked["esm1v_rank"] = ranked["esm1v_score"].rank(ascending=False, method="min")
    ranked["if_rank"] = ranked["if_score"].rank(ascending=False, method="min")
    relaxed_top_n = top_n
    while relaxed_top_n <= max_relaxed_top_n:
        mask = (ranked["esm1v_rank"] <= relaxed_top_n) & (ranked["if_rank"] <= relaxed_top_n)
        if int(mask.sum()) >= min_intersection or relaxed_top_n == max_relaxed_top_n:
            break
        relaxed_top_n += top_n
    ranked["in_esm_top50"] = ranked["esm1v_rank"] <= top_n
    ranked["in_if_top50"] = ranked["if_rank"] <= top_n
    ranked["in_relaxed_intersection"] = (ranked["esm1v_rank"] <= relaxed_top_n) & (ranked["if_rank"] <= relaxed_top_n)
    ranked["combined_rank_sum"] = ranked["esm1v_rank"] + ranked["if_rank"]
    candidates = ranked[ranked["in_relaxed_intersection"]].sort_values(
        ["combined_rank_sum", "esm1v_rank", "if_rank"]
    )
    summary = {
        "initial_top_n": top_n,
        "relaxed_top_n": relaxed_top_n,
        "intersection_count": int(len(candidates)),
        "esm_top_n_count": int((ranked["esm1v_rank"] <= top_n).sum()),
        "if_top_n_count": int((ranked["if_rank"] <= top_n).sum()),
    }
    return ranked.sort_values(["combined_rank_sum", "esm1v_rank", "if_rank"]), candidates, summary


def process_structure(structure_id: str, args: argparse.Namespace) -> None:
    out_dir = args.output_dir / structure_id
    input_dir = out_dir / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    src_pdb = find_structure_file(args.pdb_dir, structure_id)
    std_pdb = ensure_standard_pdb(src_pdb, input_dir / f"{structure_id}.pdb")

    records = read_structure_records(std_pdb)
    residues = chain_residues(records, args.chain)
    sequence = sequence_from_residues(residues)
    mutations = pd.DataFrame(all_single_mutations(structure_id, residues))
    mutations["mutant_sequence"] = mutations.apply(lambda r: mutate_sequence(sequence, int(r.seq_pos), r.mt), axis=1)
    mutations.to_csv(out_dir / "all_single_mutations.csv", index=False)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    scored = mutations
    if not args.skip_esm:
        esm_df = score_with_esm1v(sequence, mutations, args.esm_model, device)
        esm_df.to_csv(out_dir / "esm1v_scores.csv", index=False)
        scored = scored.merge(
            esm_df[["mutation", "esm1v_log_prob_mt", "esm1v_log_prob_wt", "esm1v_score"]],
            on="mutation",
            how="left",
        )
    if not args.skip_if:
        if_df = score_with_if1(std_pdb, sequence, mutations, args.chain, device, args.if_model)
        if_df.to_csv(out_dir / "inverse_folding_scores.csv", index=False)
        scored = scored.merge(
            if_df[
                [
                    "mutation",
                    "if_log_likelihood",
                    "if_log_likelihood_target",
                    "if_wt_log_likelihood",
                    "if_wt_log_likelihood_target",
                    "if_delta_log_likelihood",
                    "if_score",
                ]
            ],
            on="mutation",
            how="left",
        )

    if args.skip_esm or args.skip_if:
        scored.to_csv(out_dir / "nomination_scores_all.csv", index=False)
        return

    ranked, candidates, summary = rank_and_select(scored, args.top_n, args.min_intersection, args.max_relaxed_top_n)
    ranked.to_csv(out_dir / "nomination_scores_all.csv", index=False)
    candidates.to_csv(out_dir / "intersection_candidates.csv", index=False)
    summary.update({"structure": structure_id, "chain": args.chain, "sequence": sequence, "standard_pdb": str(std_pdb)})
    (out_dir / "nomination_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"{structure_id}: sequence={sequence}, candidates={len(candidates)}, relaxed_top_n={summary['relaxed_top_n']}")


def main() -> None:
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    for structure_id in selected_structures(args.structure):
        process_structure(structure_id, args)


if __name__ == "__main__":
    main()
