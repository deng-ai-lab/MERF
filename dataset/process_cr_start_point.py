import argparse
import os
import re
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.read_pdbs import parse_pdb, parse_pdb_chain2sequence


CR_DIR = "/home/dataset-local/projects_dir/MERF/data/CR"
DATASETS = ("cr6261_h1", "cr6261_h9", "cr9114_h1", "cr9114_h3")
MUTATION_RE = re.compile(r"^([A-Z])([A-Za-z])(-?\d+)([A-Z])$")
OUTPUT_COLUMNS = [
    "pdb_id",
    "difficulty",
    "reverse_mutation",
    "heavy_chain_sequence",
    "score",
    "score_column",
    "score_percentile",
    "score_percentile_min",
    "score_percentile_max",
    "mutation_count",
    "min_mutation_distance_to_high_fitness",
    "score_quantile_low",
    "score_quantile_high",
    "score_percentile_low",
    "score_percentile_high",
    "high_fitness_score_threshold_q99",
    "high_fitness_size",
    "source_file",
    "pdb_path",
    "selection_rule",
]


def split_mutations(mutation_key):
    if mutation_key is None:
        return []
    if isinstance(mutation_key, float) and pd.isna(mutation_key):
        return []
    mutation_key = str(mutation_key).strip()
    if mutation_key == "" or mutation_key == "nan":
        return []
    return [m.strip() for m in mutation_key.split(",") if m.strip()]


def mutation_sort_key(mutation):
    mature_aa, chain, position, immature_aa = parse_reverse_mutation(mutation)
    return int(position), chain, mature_aa, immature_aa


def parse_reverse_mutation(mutation):
    match = MUTATION_RE.match(mutation)
    if match is None:
        raise ValueError(f"Invalid CR mutation string: {mutation}")
    mature_aa, chain, position, immature_aa = match.groups()
    return mature_aa, chain, int(position), immature_aa


def canonical_mutation_key(mutation_key):
    return ",".join(sorted(split_mutations(mutation_key), key=mutation_sort_key))


def mutation_bitmask(mutation_key, mutation_to_bit):
    mask = 0
    for mutation in split_mutations(mutation_key):
        mask |= 1 << mutation_to_bit[mutation]
    return mask


def popcount(value):
    return bin(value).count("1")


def pdb_path_for_variant(cr_dir, pdb_id, reverse_mutation):
    if reverse_mutation == "":
        return os.path.join(cr_dir, "PDBs", f"{pdb_id}.pdb")
    return os.path.join(cr_dir, "PDBs_mutated", f"{pdb_id}_{reverse_mutation}.pdb")


def score_column_for_dataset(pdb_id):
    return pdb_id.split("_")[1] + "_score"


def load_antibody_chain(cr_dir, pdb_id):
    evo_path = os.path.join(cr_dir, "cr_evo.csv")
    evo_df = pd.read_csv(evo_path, dtype={"pdb_id": "string"})
    rows = evo_df[evo_df["pdb_id"] == pdb_id]
    if rows.empty:
        raise ValueError(f"{pdb_id} not found in {evo_path}")
    return rows["antibody_chain"].iloc[0]


def load_mature_heavy_chain_context(cr_dir, pdb_id, antibody_chain):
    mature_pdb_path = os.path.join(cr_dir, "PDBs", f"{pdb_id}.pdb")
    if not os.path.exists(mature_pdb_path):
        raise FileNotFoundError(f"Mature PDB does not exist: {mature_pdb_path}")

    chain2sequence = parse_pdb_chain2sequence(mature_pdb_path)
    if antibody_chain not in chain2sequence:
        raise ValueError(f"Chain {antibody_chain} not found in {mature_pdb_path}")

    parsed = parse_pdb(mature_pdb_path)
    if parsed is None:
        raise ValueError(f"Could not parse mature PDB: {mature_pdb_path}")

    mature_sequence = chain2sequence[antibody_chain]
    chain_resseq = [
        int(resseq)
        for chain, resseq in zip(parsed["chain_id"], parsed["resseq"].tolist())
        if chain == antibody_chain
    ]
    if len(chain_resseq) != len(mature_sequence):
        raise ValueError(
            f"Residue numbering length mismatch for {pdb_id} chain {antibody_chain}: "
            f"{len(chain_resseq)} residue numbers vs {len(mature_sequence)} sequence residues"
        )

    resseq_to_index = {}
    for index, resseq in enumerate(chain_resseq):
        if resseq in resseq_to_index:
            raise ValueError(
                f"Duplicate residue number {resseq} in {pdb_id} chain {antibody_chain}; "
                "cannot map CR mutation positions unambiguously"
            )
        resseq_to_index[resseq] = index

    return mature_sequence, resseq_to_index


def apply_reverse_mutations_to_sequence(mature_sequence, resseq_to_index, antibody_chain, reverse_mutation):
    sequence = list(mature_sequence)
    for mutation in split_mutations(reverse_mutation):
        mature_aa, chain, position, immature_aa = parse_reverse_mutation(mutation)
        if chain != antibody_chain:
            raise ValueError(
                f"Mutation {mutation} targets chain {chain}, but dataset antibody chain is {antibody_chain}"
            )
        if position not in resseq_to_index:
            raise ValueError(f"Mutation {mutation} position {position} not found in mature heavy chain")

        sequence_index = resseq_to_index[position]
        observed_aa = sequence[sequence_index]
        if observed_aa != mature_aa:
            raise ValueError(
                f"Mutation {mutation} expects mature residue {mature_aa} at chain {chain}{position}, "
                f"but mature sequence has {observed_aa}"
            )
        sequence[sequence_index] = immature_aa
    return "".join(sequence)


def load_scored_variants(cr_dir, pdb_id):
    score_col = score_column_for_dataset(pdb_id)
    data_path = os.path.join(cr_dir, f"{pdb_id}.csv")
    ref_path = os.path.join(cr_dir, f"{pdb_id}_ref.csv")

    df = pd.read_csv(data_path, dtype={"pdb_id": "string"})
    df = df[pd.notna(df[score_col])].copy()
    df["reverse_mutation"] = df["mutant"].apply(canonical_mutation_key)
    df["score"] = df[score_col].astype(float)
    df["source_file"] = os.path.basename(data_path)

    ref_df = pd.read_csv(ref_path, dtype={"pdb_id": "string"})
    if len(ref_df) != 1 or pd.isna(ref_df[score_col].iloc[0]):
        raise ValueError(f"Expected one scored mature reference in {ref_path}")
    ref_row = pd.DataFrame([{
        "pdb_id": pdb_id,
        "mutant": "",
        score_col: float(ref_df[score_col].iloc[0]),
        "reverse_mutation": "",
        "score": float(ref_df[score_col].iloc[0]),
        "source_file": os.path.basename(ref_path),
    }])

    all_df = pd.concat([df, ref_row], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=["reverse_mutation"], keep="first").reset_index(drop=True)
    return all_df, score_col


def add_distance_and_percentile_columns(df):
    all_mutations = sorted(
        {mutation for key in df["reverse_mutation"] for mutation in split_mutations(key)},
        key=mutation_sort_key,
    )
    mutation_to_bit = {mutation: bit for bit, mutation in enumerate(all_mutations)}

    q20 = df["score"].quantile(0.20)
    q30 = df["score"].quantile(0.30)
    q40 = df["score"].quantile(0.40)
    q50 = df["score"].quantile(0.50)
    q60 = df["score"].quantile(0.60)
    q99 = df["score"].quantile(0.99)

    high_df = df[df["score"] >= q99].copy()
    high_keys = high_df["reverse_mutation"].tolist()
    if not high_keys:
        raise ValueError("No high fitness variants found")

    df = df.copy()
    percentile_denominator = max(len(df) - 1, 1)
    df["score_percentile"] = (df["score"].rank(method="min") - 1) / percentile_denominator
    df["score_percentile_min"] = df["score_percentile"]
    df["score_percentile_max"] = (df["score"].rank(method="max") - 1) / percentile_denominator
    df["mutation_count"] = df["reverse_mutation"].apply(lambda key: len(split_mutations(key)))
    df["mutation_bitmask"] = df["reverse_mutation"].apply(lambda key: mutation_bitmask(key, mutation_to_bit))
    high_masks = [mutation_bitmask(high_key, mutation_to_bit) for high_key in high_keys]
    distance_by_mask = {
        mask: min(popcount(int(mask) ^ high_mask) for high_mask in high_masks)
        for mask in df["mutation_bitmask"].unique()
    }
    df["min_mutation_distance_to_high_fitness"] = df["mutation_bitmask"].map(distance_by_mask)

    quantiles = {
        "q20": q20,
        "q30": q30,
        "q40": q40,
        "q50": q50,
        "q60": q60,
        "q99": q99,
        "high_fitness_size": len(high_df),
    }
    return df, quantiles


def select_starts(df, quantiles, difficulty):
    if difficulty == "medium":
        candidates = df[
            (df["score_percentile"] >= 0.40)
            & (df["score_percentile"] <= 0.60)
            & (df["min_mutation_distance_to_high_fitness"] >= 3)
            & (df["min_mutation_distance_to_high_fitness"] <= 5)
        ].copy()
        return candidates.sort_values([
            "score_percentile",
            "min_mutation_distance_to_high_fitness",
            "mutation_count",
            "reverse_mutation",
        ]).reset_index(drop=True)

    if difficulty == "hard":
        candidates = df[
            (df["score_percentile"] >= 0.20)
            & (df["score_percentile"] <= 0.40)
            & (df["min_mutation_distance_to_high_fitness"] >= 6)
        ].copy()
        return candidates.sort_values([
            "score_percentile",
            "min_mutation_distance_to_high_fitness",
            "mutation_count",
            "reverse_mutation",
        ]).reset_index(drop=True)

    raise ValueError(f"Unsupported difficulty: {difficulty}")


def build_output_row(
    cr_dir,
    pdb_id,
    antibody_chain,
    score_col,
    selected,
    quantiles,
    difficulty,
    mature_sequence,
    resseq_to_index,
):
    reverse_mutation = selected["reverse_mutation"]
    pdb_path = pdb_path_for_variant(cr_dir, pdb_id, reverse_mutation)
    if not os.path.exists(pdb_path):
        raise FileNotFoundError(f"Selected start PDB does not exist: {pdb_path}")
    heavy_chain_sequence = apply_reverse_mutations_to_sequence(
        mature_sequence,
        resseq_to_index,
        antibody_chain,
        reverse_mutation,
    )

    if difficulty == "medium":
        score_quantile_low, score_quantile_high = quantiles["q40"], quantiles["q60"]
        score_percentile_low, score_percentile_high = 0.40, 0.60
        selection_rule = "score percentile in [0.40,0.60], min distance to high fitness in [3,5]"
    else:
        score_quantile_low, score_quantile_high = quantiles["q20"], quantiles["q40"]
        score_percentile_low, score_percentile_high = 0.20, 0.40
        selection_rule = "score percentile in [0.20,0.40], min distance to high fitness >= 6"

    return {
        "pdb_id": pdb_id,
        "difficulty": difficulty,
        "reverse_mutation": reverse_mutation,
        "heavy_chain_sequence": heavy_chain_sequence,
        "score": float(selected["score"]),
        "score_column": score_col,
        "score_percentile": float(selected["score_percentile"]),
        "score_percentile_min": float(selected["score_percentile_min"]),
        "score_percentile_max": float(selected["score_percentile_max"]),
        "mutation_count": int(selected["mutation_count"]),
        "min_mutation_distance_to_high_fitness": int(selected["min_mutation_distance_to_high_fitness"]),
        "score_quantile_low": float(score_quantile_low),
        "score_quantile_high": float(score_quantile_high),
        "score_percentile_low": float(score_percentile_low),
        "score_percentile_high": float(score_percentile_high),
        "high_fitness_score_threshold_q99": float(quantiles["q99"]),
        "high_fitness_size": int(quantiles["high_fitness_size"]),
        "source_file": selected["source_file"],
        "pdb_path": pdb_path,
        "selection_rule": selection_rule,
    }


def process_dataset(cr_dir, pdb_id):
    antibody_chain = load_antibody_chain(cr_dir, pdb_id)
    mature_sequence, resseq_to_index = load_mature_heavy_chain_context(cr_dir, pdb_id, antibody_chain)
    df, score_col = load_scored_variants(cr_dir, pdb_id)
    df, quantiles = add_distance_and_percentile_columns(df)

    output_paths = []
    for difficulty in ("medium", "hard"):
        selected_df = select_starts(df, quantiles, difficulty)
        rows = [
            build_output_row(
                cr_dir,
                pdb_id,
                antibody_chain,
                score_col,
                selected,
                quantiles,
                difficulty,
                mature_sequence,
                resseq_to_index,
            )
            for _, selected in selected_df.iterrows()
        ]
        out_path = os.path.join(cr_dir, f"{pdb_id}_{difficulty}_start_point.csv")
        pd.DataFrame(rows, columns=OUTPUT_COLUMNS).to_csv(out_path, index=False)
        output_paths.append(out_path)
    return output_paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cr_dir", type=str, default=CR_DIR)
    parser.add_argument("--datasets", type=str, default=",".join(DATASETS))
    args = parser.parse_args()

    all_outputs = []
    for pdb_id in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        all_outputs.extend(process_dataset(args.cr_dir, pdb_id))

    print("Generated CR start point files:")
    for path in all_outputs:
        print(path)


if __name__ == "__main__":
    main()
