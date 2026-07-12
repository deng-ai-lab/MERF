import argparse
import copy
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from model.MERF_v6 import MERF
from protein.read_pdbs import KnnAgnet, KnnResidue, PaddingCollate, parse_pdb, parse_pdb_chain2sequence
from utils.util import recursive_to, seed_all


def find_single_file(folder, suffix):
    matches = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(suffix)
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one *{suffix} in {folder}, found {matches}")
    return matches[0]


def find_landscape_csv(folder, pdb_id):
    preferred = os.path.join(folder, f"{pdb_id}.csv")
    if os.path.exists(preferred):
        return preferred
    matches = [
        os.path.join(folder, name)
        for name in os.listdir(folder)
        if name.endswith(".csv") and not name.endswith("_evo.csv")
    ]
    if len(matches) != 1:
        raise FileNotFoundError(f"Expected exactly one landscape csv in {folder}, found {matches}")
    return matches[0]


def clone_complex_info(complex_info):
    out = {}
    for key, value in complex_info.items():
        if isinstance(value, torch.Tensor):
            out[key] = value.clone()
        elif isinstance(value, (dict, list)):
            out[key] = copy.deepcopy(value)
        else:
            out[key] = value
    return out


def build_plm_embedding(pdb_id, pdb_path, plm_path, embedding_path, device):
    if os.path.exists(embedding_path):
        embeddings = torch.load(embedding_path, map_location="cpu")
        if pdb_id in embeddings:
            return embeddings[pdb_id]

    if plm_path is None:
        raise ValueError("plm_path must be provided to build missing PLM embeddings")

    tokenizer = AutoTokenizer.from_pretrained(plm_path)
    model = AutoModel.from_pretrained(plm_path)
    model.eval()
    model = model.to(device)

    chain2sequence = parse_pdb_chain2sequence(pdb_path)
    complex_info = parse_pdb(pdb_path)
    chain_id_list = complex_info["chain_id"]
    assert (
        list(chain2sequence.keys())[0] == chain_id_list[0]
        and list(chain2sequence.keys())[-1] == chain_id_list[-1]
    ), "chain order mismatch!"

    embedding_list = []
    for _, sequence in tqdm(chain2sequence.items(), desc=f"Processing PLM embedding for {pdb_id}", mininterval=30):
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.squeeze(0)[1:len(sequence) + 1, :].cpu()
        embedding_list.append(embedding)

    embedding = torch.cat(embedding_list, dim=0)
    assert embedding.shape[0] == len(chain_id_list), "PLM embedding length mismatch!"

    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    torch.save({pdb_id: embedding}, embedding_path)
    return embedding


class CRStartPointDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        evo_folder,
        pdb_id,
        landscape_csv,
        target_column,
        knn_num,
        knn_agents_num,
        plm_embedding=None,
    ):
        super().__init__()
        self.evo_folder = evo_folder
        self.pdb_id = pdb_id
        self.fix_dir = os.path.join(evo_folder, "PDBs_fixed")
        self.mut_dir = os.path.join(evo_folder, "PDBs_mutated")
        self.target_column = target_column
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        self.plm_embedding = plm_embedding

        df = pd.read_csv(landscape_csv, dtype={"pdb_id": "string"})
        df = df[pd.notna(df[self.target_column])].copy()
        df["mutant"] = df["mutant"].fillna("").astype(str).str.strip()
        df = df[df["mutant"] != ""].copy()
        self.data_df = df.reset_index(drop=True)

        self.start_pdb_path = os.path.join(self.fix_dir, f"{self.pdb_id}.pdb")
        if not os.path.exists(self.start_pdb_path):
            raise FileNotFoundError(f"Fixed start PDB not found: {self.start_pdb_path}")

        self.start_complex_info = parse_pdb(self.start_pdb_path)
        if self.plm_embedding is not None:
            self.start_complex_info["plm_embedding"] = self.plm_embedding

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        mutate_info = row["mutant"]
        target_pdb_path = os.path.join(self.mut_dir, f"{self.pdb_id}_{mutate_info}.pdb")
        if not os.path.exists(target_pdb_path):
            raise FileNotFoundError(f"Target PDB not found: {target_pdb_path}")

        complex_wt_info = clone_complex_info(self.start_complex_info)
        complex_mut_info = parse_pdb(target_pdb_path)
        if self.plm_embedding is not None:
            complex_mut_info["plm_embedding"] = self.plm_embedding

        mutation_mask = complex_wt_info["aa"] != complex_mut_info["aa"]
        if not mutation_mask.any():
            raise ValueError(f"No sequence difference for {self.pdb_id}: {mutate_info}")

        agent_mask = KnnAgnet(num_neighbors=self.knn_agents_num)({
            "wt": complex_wt_info,
            "mut": complex_mut_info,
            "mutation_mask": mutation_mask,
        })

        complex_wt_info["agent_mask"] = agent_mask
        complex_wt_info["PDB_id"] = self.pdb_id
        complex_wt_info["mutate_info"] = mutate_info

        complex_mut_info["agent_mask"] = agent_mask
        complex_mut_info["PDB_id"] = self.pdb_id
        complex_mut_info["mutate_info"] = mutate_info

        batch = KnnResidue(num_neighbors=self.knn_num)({
            "wt": complex_wt_info,
            "mut": complex_mut_info,
            "mutation_mask": mutation_mask,
        })
        batch["mutation_mask"] = batch["wt"]["aa"] != batch["mut"]["aa"]
        batch["ddG"] = -1.0 * float(row[self.target_column])
        return batch


def load_args_and_model(test_args, device):
    args_path = os.path.join(test_args.log_dir, "args.pkl")
    with open(args_path, "rb") as f:
        args = pickle.load(f)
    args.gpu_idx = test_args.gpu_idx

    checkpoint_path = os.path.join(test_args.log_dir, f"Epoch{test_args.checkpoint_epoch}_MERF.pth")
    print(f"Loaded args from: {args_path}")
    print(f"Loading model from: {checkpoint_path}")

    model = MERF(args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    return args, model, checkpoint_path


def evaluate(model, dataloader, device):
    predictions = []
    ground_truths = []
    mutants = []
    pdb_ids = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", mininterval=30):
            batch = recursive_to(batch, device)
            pred, _, _, _ = model(batch, device)
            if pred.dim() == 0:
                predictions.append(pred.item())
            else:
                predictions.extend(pred.cpu().tolist())

            ddg = batch["ddG"]
            if ddg.dim() == 0:
                ground_truths.append(ddg.item())
            else:
                ground_truths.extend(ddg.cpu().tolist())

            pdb_ids.extend(batch["wt"]["PDB_id"])
            mutants.extend(batch["wt"]["mutate_info"])

    results_df = pd.DataFrame({
        "PDB_id": np.array(pdb_ids),
        "mutant": np.array(mutants),
        "Prediction": np.array(predictions),
        "Ground_Truth": np.array(ground_truths),
    })

    pred_array = np.array(predictions)
    gt_array = np.array(ground_truths)
    valid_mask = ~(np.isnan(gt_array) | np.isnan(pred_array) | np.isinf(gt_array) | np.isinf(pred_array))
    pred_valid = pred_array[valid_mask]
    gt_valid = gt_array[valid_mask]
    metrics = {
        "num_valid_samples": int(valid_mask.sum()),
        "num_total_samples": int(len(gt_array)),
    }
    if len(pred_valid) >= 2:
        pearson_r, pearson_p = pearsonr(pred_valid, gt_valid)
        spearman_r, spearman_p = spearmanr(pred_valid, gt_valid)
        metrics.update({
            "pearson_r": pearson_r,
            "pearson_p": pearson_p,
            "spearman_r": spearman_r,
            "spearman_p": spearman_p,
        })
    else:
        metrics.update({
            "pearson_r": np.nan,
            "pearson_p": np.nan,
            "spearman_r": np.nan,
            "spearman_p": np.nan,
        })
    return results_df, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--evo_folder", type=str, default="/home/dataset-local/projects_dir/MERF/data/CR_evo/cr6261_h1_PA28T,RA30S,TA58A,KA59N,PA62Q,DA74K,FA75S,AA76T,GA77S,VA79A,VA104L")
    parser.add_argument("--log_dir", type=str, default="/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_06_09_16_56_32")
    parser.add_argument("--checkpoint_epoch", type=int, default=200)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="/home/dataset-local/projects_dir/MERF/analysis/0708analysis_1/results")
    test_args = parser.parse_args()

    evo_folder = os.path.abspath(test_args.evo_folder)
    if not os.path.isdir(evo_folder):
        raise NotADirectoryError(evo_folder)

    pdb_id = os.path.basename(evo_folder.rstrip(os.sep))
    evo_csv = find_single_file(evo_folder, "_evo.csv")
    evo_df = pd.read_csv(evo_csv, dtype={"pdb_id": "string"})
    if len(evo_df) != 1:
        raise ValueError(f"Expected one row in {evo_csv}, found {len(evo_df)}")
    original_pdb_id = evo_df["pdb_id"].iloc[0]
    target_column = evo_df["score_column"].iloc[0]
    landscape_csv = find_landscape_csv(evo_folder, pdb_id)

    device = torch.device(f"cuda:{test_args.gpu_idx}" if torch.cuda.is_available() else "cpu")
    args, model, checkpoint_path = load_args_and_model(test_args, device)
    seed_all(args.seed)

    fixed_pdb_path = os.path.join(evo_folder, "PDBs_fixed", f"{pdb_id}.pdb")
    embedding_path = os.path.join(evo_folder, "plm_embeddings", f"{pdb_id}_esm2_650_embeddings.pkl")

    print("=" * 80)
    print("CR start-point ddG evaluation")
    print(f"Evo folder: {evo_folder}")
    print(f"PDB ID: {pdb_id}")
    print(f"Original PDB ID: {original_pdb_id}")
    print(f"Evo csv: {evo_csv}")
    print(f"Landscape csv: {landscape_csv}")
    print(f"Fixed PDB: {fixed_pdb_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("=" * 80)

    plm_embedding = None
    if args.use_plm_embedding:
        plm_embedding = build_plm_embedding(
            pdb_id=pdb_id,
            pdb_path=fixed_pdb_path,
            plm_path=args.plm_path,
            embedding_path=embedding_path,
            device=device,
        )

    dataset = CRStartPointDataset(
        evo_folder=evo_folder,
        pdb_id=pdb_id,
        landscape_csv=landscape_csv,
        target_column=target_column,
        knn_num=args.knn_neighbors_num,
        knn_agents_num=args.knn_agents_num,
        plm_embedding=plm_embedding,
    )
    print(f"Dataset size after excluding start: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=test_args.batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=test_args.num_workers,
        collate_fn=PaddingCollate(),
    )

    results_df, metrics = evaluate(model, dataloader, device)
    metadata_cols = [
        "mutant",
        target_column,
        "absolute_score",
        "start_score",
        "start_reverse_mutation",
        "target_reverse_mutation",
        "target_source_file",
    ]
    metadata = dataset.data_df[[col for col in metadata_cols if col in dataset.data_df.columns]].copy()
    results_df = results_df.merge(metadata, on="mutant", how="left")
    results_df.insert(1, "original_pdb_id", original_pdb_id)
    results_df.insert(2, "evo_folder", evo_folder)

    output_dir = os.path.join(test_args.output_dir, pdb_id)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{pdb_id}_predictions_epoch{test_args.checkpoint_epoch}.csv")
    metrics_file = os.path.join(output_dir, f"{pdb_id}_metrics_epoch{test_args.checkpoint_epoch}.csv")
    results_df.to_csv(results_file, index=False)

    metrics.update({
        "pdb_id": pdb_id,
        "original_pdb_id": original_pdb_id,
        "evo_folder": evo_folder,
        "evo_csv": evo_csv,
        "landscape_csv": landscape_csv,
        "fixed_pdb_path": fixed_pdb_path,
        "plm_embedding_path": embedding_path,
        "checkpoint_epoch": test_args.checkpoint_epoch,
    })
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    print(f"Results saved to: {results_file}")
    print(f"Metrics saved to: {metrics_file}")
    print(f"Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
    print(f"Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")


if __name__ == "__main__":
    main()
