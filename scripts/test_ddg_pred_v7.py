import argparse
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error
from torch.utils.data import DataLoader
from tqdm import tqdm


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dataset.dataset_ddg import VMDataset
from model.MERF_v6 import MERF
from protein.read_pdbs import PaddingCollate
from utils.util import recursive_to, seed_all


VM_DIR = ROOT_DIR / "data" / "VenusMaxwell"
TEST_DATA_PATH = VM_DIR / "test_data.csv"
WT_DIR = VM_DIR / "PDBs_fixed"
MUT_DIR = VM_DIR / "PDBs_mutated"
DEFAULT_ANALYSIS_DIR = ROOT_DIR / "analysis" / "0730analysis_2"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate one MERF checkpoint on the held-out VenusMaxwell test set."
    )
    parser.add_argument("--log_dir", required=True, help="Training log directory.")
    parser.add_argument("--output_dir", required=True, help="Directory for this checkpoint's results.")
    parser.add_argument("--checkpoint_epoch", type=int, default=200)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument(
        "--embedding_path",
        default=str(DEFAULT_ANALYSIS_DIR / "test_esm2_650_embeddings.pkl"),
        help="Cached test-set ESM2 embeddings. They are generated automatically when absent.",
    )
    return parser.parse_args()


def finite_metrics(ground_truth, prediction):
    ground_truth = np.asarray(ground_truth, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    finite = np.isfinite(ground_truth) & np.isfinite(prediction)
    ground_truth = ground_truth[finite]
    prediction = prediction[finite]

    metrics = {
        "num_samples": int(len(ground_truth)),
        "pearson_r": np.nan,
        "pearson_p": np.nan,
        "spearman_r": np.nan,
        "spearman_p": np.nan,
        "rmse": np.nan,
        "mae": np.nan,
    }
    if len(ground_truth) == 0:
        return metrics

    metrics["rmse"] = float(np.sqrt(mean_squared_error(ground_truth, prediction)))
    metrics["mae"] = float(mean_absolute_error(ground_truth, prediction))

    if len(ground_truth) >= 2:
        if np.ptp(ground_truth) > 0 and np.ptp(prediction) > 0:
            pearson_r, pearson_p = pearsonr(ground_truth, prediction)
            metrics["pearson_r"] = float(pearson_r)
            metrics["pearson_p"] = float(pearson_p)
        if len(np.unique(ground_truth)) > 1 and len(np.unique(prediction)) > 1:
            spearman_r, spearman_p = spearmanr(ground_truth, prediction)
            metrics["spearman_r"] = float(spearman_r)
            metrics["spearman_p"] = float(spearman_p)

    return metrics


def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    ground_truths = []
    pdb_ids = []
    mutation_infos = []

    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Evaluating VM test", ncols=100):
            batch = recursive_to(batch, device)
            prediction, _, _, _ = model(batch, device)

            predictions.extend(prediction.detach().cpu().reshape(-1).tolist())
            ground_truths.extend(batch["ddG"].detach().cpu().reshape(-1).tolist())
            pdb_ids.extend(batch["wt"]["PDB_id"])
            mutation_infos.extend(batch["wt"]["mutate_info"])

    return pd.DataFrame(
        {
            "PDB_id": pdb_ids,
            "mutate_info": mutation_infos,
            "Prediction": predictions,
            "Ground_Truth": ground_truths,
        }
    )


def calculate_metrics(results_df):
    pooled = finite_metrics(results_df["Ground_Truth"], results_df["Prediction"])

    protein_rows = []
    for pdb_id, group in results_df.groupby("PDB_id", sort=True):
        row = {"PDB_id": pdb_id}
        row.update(finite_metrics(group["Ground_Truth"], group["Prediction"]))
        protein_rows.append(row)
    protein_df = pd.DataFrame(protein_rows)

    metrics = {
        "num_samples": int(len(results_df)),
        "num_proteins": int(results_df["PDB_id"].nunique()),
        "pooled_pearson_r": pooled["pearson_r"],
        "pooled_pearson_p": pooled["pearson_p"],
        "pooled_spearman_r": pooled["spearman_r"],
        "pooled_spearman_p": pooled["spearman_p"],
        "pooled_rmse": pooled["rmse"],
        "pooled_mae": pooled["mae"],
        "macro_pearson_r": float(protein_df["pearson_r"].mean()),
        "macro_spearman_r": float(protein_df["spearman_r"].mean()),
        "macro_rmse": float(protein_df["rmse"].mean()),
        "macro_mae": float(protein_df["mae"].mean()),
        "num_proteins_valid_pearson": int(protein_df["pearson_r"].notna().sum()),
        "num_proteins_valid_spearman": int(protein_df["spearman_r"].notna().sum()),
    }
    return protein_df, metrics


def json_safe(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    return value


def main():
    test_args = parse_args()
    log_dir = Path(test_args.log_dir).resolve()
    output_dir = Path(test_args.output_dir).resolve()
    embedding_path = Path(test_args.embedding_path).resolve()
    checkpoint_path = log_dir / f"Epoch{test_args.checkpoint_epoch}_MERF.pth"
    args_path = log_dir / "args.pkl"

    for required_path in (TEST_DATA_PATH, WT_DIR, MUT_DIR, args_path, checkpoint_path):
        if not required_path.exists():
            raise FileNotFoundError(f"Required path does not exist: {required_path}")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this evaluation is required to run on GPU 0.")

    output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path.parent.mkdir(parents=True, exist_ok=True)

    with args_path.open("rb") as handle:
        train_args = pickle.load(handle)
    train_args.gpu_idx = test_args.gpu_idx
    train_args.is_cuda = True

    seed_all(train_args.seed)
    device = torch.device(f"cuda:{test_args.gpu_idx}")
    torch.cuda.set_device(device)

    print("=" * 88)
    print("VenusMaxwell held-out test evaluation")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Test CSV: {TEST_DATA_PATH}")
    print(f"ESM embedding cache: {embedding_path}")
    print(f"Output directory: {output_dir}")
    print(f"Device: {device} ({torch.cuda.get_device_name(device)})")
    print("=" * 88)

    dataset = VMDataset(
        str(TEST_DATA_PATH),
        str(WT_DIR),
        str(MUT_DIR),
        knn_num=train_args.knn_neighbors_num,
        knn_agents_num=train_args.knn_agents_num,
        use_plm_embedding=train_args.use_plm_embedding,
        plm_path=train_args.plm_path,
        plm_embedding_path=str(embedding_path),
        device=device,
    )

    # When the cache is generated in this process, VMDataset retains the ESM2
    # model even though inference only needs the cached CPU embedding tensors.
    for attribute in ("plm_model", "plm_tokenizer"):
        if hasattr(dataset, attribute):
            delattr(dataset, attribute)
    torch.cuda.empty_cache()

    dataloader = DataLoader(
        dataset,
        batch_size=test_args.batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=test_args.num_workers,
        collate_fn=PaddingCollate(),
    )

    model = MERF(train_args).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)

    results_df = evaluate_model(model, dataloader, device)
    protein_df, metrics = calculate_metrics(results_df)
    metrics.update(
        {
            "checkpoint_run": log_dir.name,
            "checkpoint_epoch": test_args.checkpoint_epoch,
            "checkpoint_path": str(checkpoint_path),
            "test_data_path": str(TEST_DATA_PATH),
            "embedding_path": str(embedding_path),
            "gpu_idx": test_args.gpu_idx,
            "batch_size": test_args.batch_size,
            "num_workers": test_args.num_workers,
        }
    )

    epoch = test_args.checkpoint_epoch
    predictions_path = output_dir / f"vm_test_predictions_epoch{epoch}.csv"
    protein_metrics_path = output_dir / f"vm_test_per_protein_metrics_epoch{epoch}.csv"
    metrics_csv_path = output_dir / f"vm_test_metrics_epoch{epoch}.csv"
    metrics_json_path = output_dir / f"vm_test_metrics_epoch{epoch}.json"

    results_df.to_csv(predictions_path, index=False)
    protein_df.to_csv(protein_metrics_path, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_csv_path, index=False)
    with metrics_json_path.open("w") as handle:
        json.dump({key: json_safe(value) for key, value in metrics.items()}, handle, indent=2)

    print("\nEvaluation complete")
    print(f"  Samples / proteins: {metrics['num_samples']} / {metrics['num_proteins']}")
    print(f"  Pooled Pearson / Spearman: {metrics['pooled_pearson_r']:.6f} / {metrics['pooled_spearman_r']:.6f}")
    print(f"  Pooled RMSE / MAE: {metrics['pooled_rmse']:.6f} / {metrics['pooled_mae']:.6f}")
    print(f"  Macro Pearson / Spearman: {metrics['macro_pearson_r']:.6f} / {metrics['macro_spearman_r']:.6f}")
    print(f"  Predictions: {predictions_path}")
    print(f"  Metrics: {metrics_json_path}")


if __name__ == "__main__":
    main()
