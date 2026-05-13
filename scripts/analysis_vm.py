import argparse
import os
import sys
from collections import Counter, defaultdict

import pandas as pd
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.util import seed_all


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_VM_PATH = os.path.join(ROOT_DIR, "data/VenusMaxwell/train_data.csv")
TRAIN_SK_PATH = os.path.join(ROOT_DIR, "data/SKEMPIv2/SKEMPIv2.csv")


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in ("true", "1", "yes", "y"):
        return True
    if value in ("false", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def normalize_pdb_id(pdb_id):
    return str(pdb_id).replace("+", "").replace(".00", "")


class SampleMetadataDataset(Dataset):
    def __init__(self, csv_path, dataset_name, global_offset=0):
        self.dataset_name = dataset_name
        self.global_offset = global_offset
        self.data_df = pd.read_csv(csv_path, dtype={"pdb_id": "string"})
        self.protein_ids = self.data_df["pdb_id"].map(normalize_pdb_id).tolist()
        self.mutants = self.data_df["mutant"].astype(str).tolist()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        return {
            "dataset": self.dataset_name,
            "protein_id": self.protein_ids[index],
            "mutant": self.mutants[index],
            "row_index": index,
            "global_index": self.global_offset + index,
        }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze VM pretraining sampler by counting sampled protein frequencies."
    )
    parser.add_argument("--seed", type=int, default=171, help="random seed")
    parser.add_argument("--batch_size", type=int, default=64, help="batch size used to build train batches")
    parser.add_argument("--n_epoch", type=int, default=1, help="number of sampled epochs to analyze")
    parser.add_argument("--num_works", type=int, default=0, help="DataLoader workers for metadata-only batches")

    parser.add_argument("--use_weighted_sampling", type=str2bool, default=True)
    parser.add_argument("--vm_sampling_weight", type=float, default=15.0)
    parser.add_argument("--sk_sampling_weight", type=float, default=1.0)
    parser.add_argument("--max_repetition_factor", type=float, default=2.0)

    parser.add_argument("--output_csv", type=str, default=None, help="CSV path for protein frequency output")
    parser.add_argument(
        "--output_dir",
        type=str,
        default=os.path.join(ROOT_DIR, "logs/analysis_vm"),
        help="directory used when --output_csv is not set",
    )

    args, unknown = parser.parse_known_args()
    if unknown:
        print(f"Ignoring unused training args: {' '.join(unknown)}")
    return args


def make_output_path(args):
    if args.output_csv:
        return args.output_csv

    sampling_name = "weighted" if args.use_weighted_sampling else "shuffle"
    file_name = (
        f"vm_sampling_seed{args.seed}_{sampling_name}"
        f"_bs{args.batch_size}_epochs{args.n_epoch}.csv"
    )
    return os.path.join(args.output_dir, file_name)


def build_train_loader(args, train_vm_dataset, train_sk_dataset):
    train_dataset = ConcatDataset([train_vm_dataset, train_sk_dataset])

    if args.use_weighted_sampling:
        len_vm = len(train_vm_dataset)
        len_sk = len(train_sk_dataset)

        weight_per_vm_sample = args.vm_sampling_weight / len_vm
        weight_per_sk_sample = args.sk_sampling_weight / len_sk
        sample_weights = [weight_per_vm_sample] * len_vm + [weight_per_sk_sample] * len_sk

        if args.max_repetition_factor > 0:
            smaller_dataset_size = min(len_vm, len_sk)
            num_samples_per_epoch = min(
                len(train_dataset),
                int(smaller_dataset_size * args.max_repetition_factor * 2),
            )
        else:
            num_samples_per_epoch = len(train_dataset)

        sampler = WeightedRandomSampler(
            weights=sample_weights,
            num_samples=num_samples_per_epoch,
            replacement=True,
        )
        shuffle = False
    else:
        sampler = None
        shuffle = True
        num_samples_per_epoch = len(train_dataset)

    loader = DataLoader(
        train_dataset,
        sampler=sampler,
        shuffle=shuffle,
        batch_size=args.batch_size,
        pin_memory=False,
        drop_last=True,
        num_workers=args.num_works,
    )
    return loader, num_samples_per_epoch


def source_counts(*datasets):
    rows = []
    for dataset in datasets:
        grouped = dataset.data_df.assign(
            dataset=dataset.dataset_name,
            protein_id=dataset.data_df["pdb_id"].map(normalize_pdb_id),
        ).groupby(["dataset", "protein_id"])
        for (dataset_name, protein_id), group in grouped:
            rows.append(
                {
                    "dataset": dataset_name,
                    "protein_id": protein_id,
                    "source_row_count": len(group),
                    "source_unique_mutants": group["mutant"].nunique(),
                }
            )
    return pd.DataFrame(rows)


def analyze_batches(args, loader):
    sample_counter = Counter()
    mutation_counter = defaultdict(set)
    batch_counter = Counter()
    max_count_in_batch = Counter()

    total_batches = 0
    total_samples = 0

    for _ in range(args.n_epoch):
        for batch in loader:
            total_batches += 1

            dataset_names = batch["dataset"]
            protein_ids = batch["protein_id"]
            mutants = batch["mutant"]

            keys_in_batch = []
            for dataset_name, protein_id, mutant in zip(dataset_names, protein_ids, mutants):
                key = (dataset_name, protein_id)
                sample_counter[key] += 1
                mutation_counter[key].add(mutant)
                keys_in_batch.append(key)
                total_samples += 1

            batch_local_counts = Counter(keys_in_batch)
            for key, count in batch_local_counts.items():
                batch_counter[key] += 1
                max_count_in_batch[key] = max(max_count_in_batch[key], count)

    rows = []
    for key, count in sample_counter.items():
        dataset_name, protein_id = key
        rows.append(
            {
                "seed": args.seed,
                "dataset": dataset_name,
                "protein_id": protein_id,
                "sampled_count": count,
                "sampled_fraction": count / total_samples if total_samples else 0.0,
                "avg_count_per_epoch": count / args.n_epoch,
                "batch_occurrences": batch_counter[key],
                "batch_occurrence_fraction": batch_counter[key] / total_batches if total_batches else 0.0,
                "max_count_in_single_batch": max_count_in_batch[key],
                "sampled_unique_mutants": len(mutation_counter[key]),
            }
        )

    result_df = pd.DataFrame(rows)
    if result_df.empty:
        return result_df, total_batches, total_samples

    result_df = result_df.sort_values(
        ["dataset", "sampled_count", "protein_id"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return result_df, total_batches, total_samples


def main():
    args = parse_args()
    seed_all(args.seed)

    train_vm_dataset = SampleMetadataDataset(TRAIN_VM_PATH, "VM", global_offset=0)
    train_sk_dataset = SampleMetadataDataset(
        TRAIN_SK_PATH,
        "SKEMPIv2",
        global_offset=len(train_vm_dataset),
    )
    train_loader, num_samples_per_epoch = build_train_loader(
        args,
        train_vm_dataset,
        train_sk_dataset,
    )

    result_df, total_batches, total_samples = analyze_batches(args, train_loader)
    source_df = source_counts(train_vm_dataset, train_sk_dataset)
    result_df = result_df.merge(source_df, on=["dataset", "protein_id"], how="left")

    result_df["batch_size"] = args.batch_size
    result_df["n_epoch"] = args.n_epoch
    result_df["use_weighted_sampling"] = args.use_weighted_sampling
    result_df["vm_sampling_weight"] = args.vm_sampling_weight
    result_df["sk_sampling_weight"] = args.sk_sampling_weight
    result_df["max_repetition_factor"] = args.max_repetition_factor
    result_df["configured_samples_per_epoch"] = num_samples_per_epoch
    result_df["actual_batches_total"] = total_batches
    result_df["actual_samples_total"] = total_samples

    output_csv = make_output_path(args)
    output_dir = os.path.dirname(output_csv)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    result_df.to_csv(output_csv, index=False)

    print(f"VM rows: {len(train_vm_dataset)}, SKEMPIv2 rows: {len(train_sk_dataset)}")
    print(f"Configured samples/epoch: {num_samples_per_epoch}")
    print(f"Actual batches: {total_batches}, actual samples after drop_last: {total_samples}")
    print(f"Protein frequency CSV saved to: {output_csv}")


if __name__ == "__main__":
    main()
