"""Create balanced CR evolution start points for ESMFold2-backed v4 experiments.

The difficulty definition remains data driven: a start point is characterized by
its score percentile and the minimum Hamming distance to the top 1% measured
landscape variants.  This version only changes how many starts are retained and
how they are sampled: each difficulty receives a deterministic, diverse set of
50 starts instead of retaining every matching row.
"""

import argparse
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protein.read_pdbs import parse_pdb, parse_pdb_chain2sequence


PROJECT_DIR = "/home/dataset-local/projects_dir/MERF"
CR_DIR = os.path.join(PROJECT_DIR, "data", "CR")
EVO_DIR = os.path.join(PROJECT_DIR, "data", "CR_evo_esmfold2")
DATASETS = ("cr6261_h1", "cr6261_h9", "cr9114_h1")
MUTATION_RE = re.compile(r"^([A-Z])([A-Za-z])(-?\d+)([A-Z])$")

BASE_OUTPUT_COLUMNS = [
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
SELECTION_COLUMNS = [
    "selection_score_bin",
    "selection_distance_stratum",
    "selection_order",
    "selection_min_hamming_to_previous",
]
OUTPUT_COLUMNS = BASE_OUTPUT_COLUMNS + SELECTION_COLUMNS


def split_mutations(mutation_key):
    if mutation_key is None or (isinstance(mutation_key, float) and pd.isna(mutation_key)):
        return []
    value = str(mutation_key).strip()
    if value in {"", "nan"}:
        return []
    return [mutation.strip() for mutation in value.split(",") if mutation.strip()]


def parse_reverse_mutation(mutation):
    match = MUTATION_RE.match(mutation)
    if match is None:
        raise ValueError(f"Invalid CR mutation string: {mutation}")
    mature_aa, chain, position, immature_aa = match.groups()
    return mature_aa, chain, int(position), immature_aa


def mutation_sort_key(mutation):
    mature_aa, chain, position, immature_aa = parse_reverse_mutation(mutation)
    return position, chain, mature_aa, immature_aa


def canonical_mutation_key(mutation_key):
    return ",".join(sorted(split_mutations(mutation_key), key=mutation_sort_key))


def score_column_for_dataset(pdb_id):
    return pdb_id.split("_")[1] + "_score"


def folder_id_for_start(pdb_id, reverse_mutation):
    return pdb_id if reverse_mutation == "" else f"{pdb_id}_{reverse_mutation}"


def future_pdb_path(evo_dir, pdb_id, reverse_mutation):
    folder_id = folder_id_for_start(pdb_id, reverse_mutation)
    return os.path.join(evo_dir, folder_id, "PDBs", f"{folder_id}.pdb")


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
    return pd.concat([df, ref_row], ignore_index=True).drop_duplicates(
        subset=["reverse_mutation"], keep="first"
    ).reset_index(drop=True), score_col


def add_difficulty_columns(df):
    """Add score percentile and min distance to the top 1% without O(N*H) Python loops.

    CR landscapes have at most 16 binary mutation sites.  A precomputed bitcount
    table therefore gives every Hamming distance exactly and makes CR9114 fast.
    """
    all_mutations = sorted(
        {mutation for key in df["reverse_mutation"] for mutation in split_mutations(key)},
        key=mutation_sort_key,
    )
    if len(all_mutations) > 20:
        raise ValueError(f"Expected a small binary CR landscape, got {len(all_mutations)} sites")
    mutation_to_bit = {mutation: index for index, mutation in enumerate(all_mutations)}

    def key_to_mask(key):
        mask = 0
        for mutation in split_mutations(key):
            mask |= 1 << mutation_to_bit[mutation]
        return mask

    out = df.copy()
    out["mutation_bitmask"] = out["reverse_mutation"].apply(key_to_mask).astype(np.int64)
    out["mutation_count"] = out["reverse_mutation"].apply(lambda key: len(split_mutations(key)))
    denominator = max(len(out) - 1, 1)
    out["score_percentile"] = (out["score"].rank(method="min") - 1) / denominator
    out["score_percentile_min"] = out["score_percentile"]
    out["score_percentile_max"] = (out["score"].rank(method="max") - 1) / denominator

    q20 = float(out["score"].quantile(0.20))
    q40 = float(out["score"].quantile(0.40))
    q60 = float(out["score"].quantile(0.60))
    q99 = float(out["score"].quantile(0.99))
    high_masks = out.loc[out["score"] >= q99, "mutation_bitmask"].unique()
    if len(high_masks) == 0:
        raise ValueError("No top-1% variants found")

    # 对所有可能的二进制状态预先计算最近 top-1% 状态；避免 CR9114 的逐行嵌套循环。
    state_count = 1 << len(all_mutations)
    state_ids = np.arange(state_count, dtype=np.int64)
    bitcount = np.fromiter((bin(value).count("1") for value in range(state_count)), dtype=np.uint8)
    nearest_distance = np.full(state_count, len(all_mutations) + 1, dtype=np.uint8)
    for high_mask in high_masks:
        nearest_distance = np.minimum(nearest_distance, bitcount[state_ids ^ int(high_mask)])
    out["min_mutation_distance_to_high_fitness"] = nearest_distance[
        out["mutation_bitmask"].to_numpy(dtype=np.int64)
    ].astype(int)
    return out, {
        "q20": q20,
        "q40": q40,
        "q60": q60,
        "q99": q99,
        "high_fitness_size": int(len(high_masks)),
        "mutation_to_bit": mutation_to_bit,
    }


def difficulty_pool(df, difficulty):
    """Retain the original score/distance motivation while avoiding overly harsh hard starts."""
    if difficulty == "medium":
        # 中等任务：分数居中、距 top 1% 三至五步、三至五个逆向突变。
        return df[
            df["score_percentile"].between(0.40, 0.60)
            & df["min_mutation_distance_to_high_fitness"].between(3, 5)
            & df["mutation_count"].between(3, 5)
        ].copy(), (0.40, 0.60), (3, 4, 5)
    if difficulty == "hard":
        # 困难任务不使用比旧规则更远的起点：距离只取 5/6，不取 >6；
        # 保留多点突变，但限制到不超过十个，避免仅因突变数极大而失效。
        return df[
            df["score_percentile"].between(0.20, 0.40)
            & df["min_mutation_distance_to_high_fitness"].between(5, 6)
            & df["mutation_count"].between(5, 10)
        ].copy(), (0.20, 0.40), (5, 6)
    raise ValueError(f"Unsupported difficulty: {difficulty}")


def add_selection_strata(pool, score_range):
    low, high = score_range
    span = high - low
    values = ((pool["score_percentile"] - low) / span * 5).astype(int).clip(0, 4)
    out = pool.copy()
    out["selection_score_bin"] = values.astype(int)
    return out


def hamming_distance(mask_a, mask_b):
    return int(bin(int(mask_a) ^ int(mask_b)).count("1"))


def select_diverse_stratified(pool, target_count):
    """Select a fixed-size, reproducible set with balanced difficulty strata.

    First, score-percentile bins and nearest-top-1% distance groups receive near
    equal quotas.  Within and across those strata, candidates are chosen by a
    greedy max-min Hamming criterion so that the retained starts are not merely
    near-duplicate mutation sets.
    """
    group_keys = sorted({
        (int(row.selection_score_bin), int(row.min_mutation_distance_to_high_fitness))
        for row in pool.itertuples()
    })
    groups = {
        key: pool[
            (pool["selection_score_bin"] == key[0])
            & (pool["min_mutation_distance_to_high_fitness"] == key[1])
        ].copy().sort_values(["reverse_mutation"]).reset_index(drop=True)
        for key in group_keys
    }
    groups = {key: value for key, value in groups.items() if not value.empty}
    if len(pool) < target_count:
        raise ValueError(f"Only {len(pool)} eligible candidates, fewer than requested {target_count}")

    ordered_groups = sorted(groups)
    base, remainder = divmod(target_count, len(ordered_groups))
    quotas = {
        key: min(len(groups[key]), base + (index < remainder))
        for index, key in enumerate(ordered_groups)
    }
    remaining = target_count - sum(quotas.values())
    # 某个稀有分层容量不足时，把名额补到仍有空间且当前配额最少的分层。
    while remaining > 0:
        eligible = [key for key in ordered_groups if quotas[key] < len(groups[key])]
        if not eligible:
            raise ValueError("Unable to allocate the requested number of start points")
        key = min(eligible, key=lambda item: (quotas[item], -len(groups[item]), item))
        quotas[key] += 1
        remaining -= 1

    selected_rows = []
    selected_masks = []
    selection_order = 0
    selected_per_group = {key: 0 for key in ordered_groups}
    while any(selected_per_group[key] < quotas[key] for key in ordered_groups):
        for key in ordered_groups:
            if selected_per_group[key] >= quotas[key]:
                continue
            candidates = groups[key]
            candidates = candidates[~candidates["reverse_mutation"].isin(
                [row["reverse_mutation"] for row in selected_rows]
            )]
            if candidates.empty:
                raise ValueError(f"Stratum {key} exhausted before quota was met")
            score_center = candidates["score_percentile"].median()
            mutation_center = candidates["mutation_count"].median()
            ranked = []
            for _, candidate in candidates.iterrows():
                mask = int(candidate["mutation_bitmask"])
                min_distance = (
                    min(hamming_distance(mask, other_mask) for other_mask in selected_masks)
                    if selected_masks else -1
                )
                center_offset = abs(float(candidate["score_percentile"]) - float(score_center))
                center_offset += 0.02 * abs(float(candidate["mutation_count"]) - float(mutation_center))
                ranked.append((
                    -min_distance,
                    center_offset,
                    str(candidate["reverse_mutation"]),
                    candidate,
                ))
            _, _, _, chosen = min(ranked, key=lambda item: item[:3])
            chosen = chosen.copy()
            chosen["selection_distance_stratum"] = int(chosen["min_mutation_distance_to_high_fitness"])
            chosen["selection_order"] = selection_order
            chosen["selection_min_hamming_to_previous"] = (
                pd.NA if not selected_masks else min(
                    hamming_distance(int(chosen["mutation_bitmask"]), other_mask)
                    for other_mask in selected_masks
                )
            )
            selected_rows.append(chosen)
            selected_masks.append(int(chosen["mutation_bitmask"]))
            selected_per_group[key] += 1
            selection_order += 1
    return pd.DataFrame(selected_rows)


def load_mature_context(cr_dir, pdb_id, antibody_chain):
    pdb_path = os.path.join(cr_dir, "PDBs", f"{pdb_id}.pdb")
    chain2sequence = parse_pdb_chain2sequence(pdb_path)
    mature_sequence = chain2sequence[antibody_chain]
    parsed = parse_pdb(pdb_path)
    chain_resseq = [
        int(resseq)
        for chain, resseq in zip(parsed["chain_id"], parsed["resseq"].tolist())
        if chain == antibody_chain
    ]
    if len(chain_resseq) != len(mature_sequence) or len(set(chain_resseq)) != len(chain_resseq):
        raise ValueError(f"Ambiguous heavy-chain numbering in {pdb_path}")
    return mature_sequence, {resseq: index for index, resseq in enumerate(chain_resseq)}


def apply_reverse_mutations(mature_sequence, resseq_to_index, antibody_chain, reverse_mutation):
    sequence = list(mature_sequence)
    for mutation in split_mutations(reverse_mutation):
        mature_aa, chain, position, immature_aa = parse_reverse_mutation(mutation)
        if chain != antibody_chain or position not in resseq_to_index:
            raise ValueError(f"Cannot map mutation {mutation} to heavy chain {antibody_chain}")
        sequence_index = resseq_to_index[position]
        if sequence[sequence_index] != mature_aa:
            raise ValueError(f"Mature sequence mismatch while applying {mutation}")
        sequence[sequence_index] = immature_aa
    return "".join(sequence)


def build_output_rows(cr_dir, evo_dir, pdb_id, difficulty, selected, quantiles, score_col):
    metadata = pd.read_csv(os.path.join(cr_dir, "cr_evo.csv"), dtype={"pdb_id": "string"})
    row = metadata.loc[metadata["pdb_id"] == pdb_id]
    if row.empty:
        raise ValueError(f"{pdb_id} is absent from {cr_dir}/cr_evo.csv")
    antibody_chain = row.iloc[0]["antibody_chain"]
    mature_sequence, resseq_to_index = load_mature_context(cr_dir, pdb_id, antibody_chain)
    if difficulty == "medium":
        score_low, score_high = 0.40, 0.60
        rule = (
            "score percentile in [0.40,0.60]; min distance to top-1% in [3,5]; "
            "reverse mutation count in [3,5]; 50-way stratified diverse sample"
        )
    else:
        score_low, score_high = 0.20, 0.40
        rule = (
            "score percentile in [0.20,0.40]; min distance to top-1% in [5,6]; "
            "reverse mutation count in [5,10]; 50-way stratified diverse sample"
        )

    rows = []
    for _, item in selected.sort_values("selection_order").iterrows():
        reverse_mutation = canonical_mutation_key(item["reverse_mutation"])
        rows.append({
            "pdb_id": pdb_id,
            "difficulty": difficulty,
            "reverse_mutation": reverse_mutation,
            "heavy_chain_sequence": apply_reverse_mutations(
                mature_sequence, resseq_to_index, antibody_chain, reverse_mutation
            ),
            "score": float(item["score"]),
            "score_column": score_col,
            "score_percentile": float(item["score_percentile"]),
            "score_percentile_min": float(item["score_percentile_min"]),
            "score_percentile_max": float(item["score_percentile_max"]),
            "mutation_count": int(item["mutation_count"]),
            "min_mutation_distance_to_high_fitness": int(item["min_mutation_distance_to_high_fitness"]),
            "score_quantile_low": float(quantiles["q40"] if difficulty == "medium" else quantiles["q20"]),
            "score_quantile_high": float(quantiles["q60"] if difficulty == "medium" else quantiles["q40"]),
            "score_percentile_low": score_low,
            "score_percentile_high": score_high,
            "high_fitness_score_threshold_q99": float(quantiles["q99"]),
            "high_fitness_size": int(quantiles["high_fitness_size"]),
            "source_file": item["source_file"],
            "pdb_path": future_pdb_path(evo_dir, pdb_id, reverse_mutation),
            "selection_rule": rule,
            "selection_score_bin": int(item["selection_score_bin"]),
            "selection_distance_stratum": int(item["selection_distance_stratum"]),
            "selection_order": int(item["selection_order"]),
            "selection_min_hamming_to_previous": item["selection_min_hamming_to_previous"],
        })
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def process_dataset(cr_dir, evo_dir, pdb_id, target_count):
    df, score_col = load_scored_variants(cr_dir, pdb_id)
    df, quantiles = add_difficulty_columns(df)
    output_paths = []
    records = []
    for difficulty in ("medium", "hard"):
        pool, score_range, _ = difficulty_pool(df, difficulty)
        pool = add_selection_strata(pool, score_range)
        selected = select_diverse_stratified(pool, target_count)
        output = build_output_rows(cr_dir, evo_dir, pdb_id, difficulty, selected, quantiles, score_col)
        out_path = os.path.join(cr_dir, f"{pdb_id}_{difficulty}_start_point_v4.csv")
        output.to_csv(out_path, index=False)
        output_paths.append(out_path)
        records.append({
            "pdb_id": pdb_id,
            "difficulty": difficulty,
            "eligible_pool_size": len(pool),
            "selected_count": len(output),
            "score_percentile_min": float(output["score_percentile"].min()),
            "score_percentile_max": float(output["score_percentile"].max()),
            "distance_min": int(output["min_mutation_distance_to_high_fitness"].min()),
            "distance_max": int(output["min_mutation_distance_to_high_fitness"].max()),
            "mutation_count_min": int(output["mutation_count"].min()),
            "mutation_count_max": int(output["mutation_count"].max()),
            "output_path": out_path,
        })
    return output_paths, records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cr_dir", default=CR_DIR)
    parser.add_argument("--evo_dir", default=EVO_DIR)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--target_count", type=int, default=50)
    parser.add_argument(
        "--manifest",
        default=os.path.join(PROJECT_DIR, "analysis", "0724analysis_1", "results", "start_point_v4_manifest.csv"),
    )
    args = parser.parse_args()
    if args.target_count <= 0:
        raise ValueError("--target_count must be positive")
    os.makedirs(os.path.dirname(args.manifest), exist_ok=True)

    records = []
    for pdb_id in [item.strip() for item in args.datasets.split(",") if item.strip()]:
        _, dataset_records = process_dataset(args.cr_dir, args.evo_dir, pdb_id, args.target_count)
        records.extend(dataset_records)
    pd.DataFrame(records).to_csv(args.manifest, index=False)
    print(pd.DataFrame(records).to_string(index=False))
    print(f"Wrote manifest: {args.manifest}")


if __name__ == "__main__":
    main()
