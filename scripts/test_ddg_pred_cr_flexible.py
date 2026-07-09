import argparse
import copy
import os
import pickle
import re
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.MERF_v6 import MERF
from protein.read_pdbs import KnnAgnet, KnnResidue, PaddingCollate, parse_pdb, parse_pdb_chain2sequence
from utils.util import recursive_to, seed_all


CR_DIR = "/home/dataset-local/projects_dir/MERF/data/CR"
PDB_DIR = os.path.join(CR_DIR, "PDBs")
MUTATED_DIR = os.path.join(CR_DIR, "PDBs_mutated")
PLM_EMBEDDING_DIR = os.path.join(CR_DIR, "plm_embeddings")
MUTATION_RE = re.compile(r"^([A-Z])([A-Za-z])(-?\d+)([A-Z])$")


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
    match = MUTATION_RE.match(mutation)
    if match is None:
        raise ValueError(f"Invalid CR mutation string: {mutation}")
    mature_aa, chain, position, immature_aa = match.groups()
    return int(position), chain, mature_aa, immature_aa


def canonical_mutation_key(mutation_key):
    return ",".join(sorted(split_mutations(mutation_key), key=mutation_sort_key))


def variant_id(pdb_id, mutation_key):
    if mutation_key == "":
        return pdb_id
    return f"{pdb_id}_{mutation_key}"


def variant_pdb_path(pdb_id, mutation_key):
    if mutation_key == "":
        return os.path.join(PDB_DIR, f"{pdb_id}.pdb")
    return os.path.join(MUTATED_DIR, f"{pdb_id}_{mutation_key}.pdb")


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


def build_plm_embedding(pdb_id, start_mutant, pdb_path, plm_path, embedding_path, embedding_key, device):
    if os.path.exists(embedding_path):
        embeddings = torch.load(embedding_path, map_location="cpu")
        if embedding_key in embeddings:
            return embeddings[embedding_key]

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
    desc = f"Processing PLM embedding for {pdb_id}:{start_mutant or 'mature'}"
    for _, sequence in tqdm(chain2sequence.items(), desc=desc, mininterval=30):
        inputs = tokenizer(sequence, return_tensors="pt")
        inputs = {key: val.to(device) for key, val in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            embedding = outputs.last_hidden_state.squeeze(0)[1:len(sequence) + 1, :].cpu()
        embedding_list.append(embedding)

    embedding = torch.cat(embedding_list, dim=0)
    assert embedding.shape[0] == len(chain_id_list), "PLM embedding length mismatch!"

    os.makedirs(os.path.dirname(embedding_path), exist_ok=True)
    torch.save({embedding_key: embedding}, embedding_path)
    return embedding


class FlexibleCRDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        data_path,
        pdb_id,
        start_mutant,
        knn_num,
        knn_agents_num,
        plm_embedding=None,
        target_column=None,
    ):
        super().__init__()
        self.data_path = data_path
        self.pdb_id = pdb_id
        self.start_mutant = canonical_mutation_key(start_mutant)
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        self.plm_embedding = plm_embedding

        df = pd.read_csv(data_path, dtype={"pdb_id": "string"})
        if target_column is None:
            target_column = [col for col in df.columns if col.endswith("_score")][0]
        self.target_column = target_column

        df = df[pd.notna(df[self.target_column])].copy()
        df["target_mutant"] = df["mutant"].apply(canonical_mutation_key)
        df = df[df["target_mutant"] != self.start_mutant].copy()
        self.data_df = df.reset_index(drop=True)

        self.start_pdb_path = variant_pdb_path(self.pdb_id, self.start_mutant)
        if not os.path.exists(self.start_pdb_path):
            raise FileNotFoundError(f"Start PDB not found: {self.start_pdb_path}")

        self.start_complex_info = parse_pdb(self.start_pdb_path)
        if self.plm_embedding is not None:
            self.start_complex_info["plm_embedding"] = self.plm_embedding

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        target_mutant = row["target_mutant"]
        target_pdb_path = variant_pdb_path(self.pdb_id, target_mutant)
        if not os.path.exists(target_pdb_path):
            raise FileNotFoundError(f"Target PDB not found: {target_pdb_path}")

        complex_wt_info = clone_complex_info(self.start_complex_info)
        complex_mut_info = parse_pdb(target_pdb_path)
        if self.plm_embedding is not None:
            complex_mut_info["plm_embedding"] = self.plm_embedding

        mutation_mask = complex_wt_info["aa"] != complex_mut_info["aa"]
        if not mutation_mask.any():
            raise ValueError(f"No sequence difference for {self.pdb_id}: {self.start_mutant} -> {target_mutant}")

        agent_mask = KnnAgnet(num_neighbors=self.knn_agents_num)({
            "wt": complex_wt_info,
            "mut": complex_mut_info,
            "mutation_mask": mutation_mask,
        })

        complex_wt_info["agent_mask"] = agent_mask
        complex_wt_info["PDB_id"] = self.pdb_id
        complex_wt_info["mutate_info"] = target_mutant
        complex_wt_info["start_mutate_info"] = self.start_mutant

        complex_mut_info["agent_mask"] = agent_mask
        complex_mut_info["PDB_id"] = self.pdb_id
        complex_mut_info["mutate_info"] = target_mutant
        complex_mut_info["start_mutate_info"] = self.start_mutant

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
    target_mutants = []
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
            target_mutants.extend(batch["wt"]["mutate_info"])

    results_df = pd.DataFrame({
        "PDB_id": np.array(pdb_ids),
        "target_mutant": np.array(target_mutants),
        "Prediction": np.array(predictions),
        "Ground_Truth": np.array(ground_truths),
    })

    pred_array = np.array(predictions)
    gt_array = np.array(ground_truths)
    valid_mask = ~(np.isnan(gt_array) | np.isnan(pred_array) | np.isinf(gt_array) | np.isinf(pred_array))
    pred_valid = pred_array[valid_mask]
    gt_valid = gt_array[valid_mask]

    pearson_r, pearson_p = pearsonr(pred_valid, gt_valid)
    spearman_r, spearman_p = spearmanr(pred_valid, gt_valid)
    metrics = {
        "pearson_r": pearson_r,
        "pearson_p": pearson_p,
        "spearman_r": spearman_r,
        "spearman_p": spearman_p,
        "num_valid_samples": int(valid_mask.sum()),
        "num_total_samples": int(len(gt_array)),
    }
    return results_df, metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_dir", type=str, default="/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_06_09_16_56_32")
    parser.add_argument("--dataset", type=str, default="cr_6261_h1")
    parser.add_argument("--start_mutant", type=str, required=True, help="Comma-separated reverse mutation combination used as the prediction origin")
    parser.add_argument("--checkpoint_epoch", type=int, default=200)
    parser.add_argument("--gpu_idx", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="/home/dataset-local/projects_dir/MERF/analysis/0707analysis_1/results")
    test_args = parser.parse_args()

    parts = test_args.dataset.split("_")
    if len(parts) != 3 or parts[0] != "cr":
        raise ValueError("--dataset must look like cr_6261_h1")

    antibody = parts[1]
    target = parts[2]
    pdb_id = f"cr{antibody}_{target}"
    target_column = f"{target}_score"
    data_path = os.path.join(CR_DIR, f"{pdb_id}.csv")
    start_mutant = canonical_mutation_key(test_args.start_mutant)
    start_pdb_path = variant_pdb_path(pdb_id, start_mutant)
    start_variant_id = variant_id(pdb_id, start_mutant)
    embedding_path = os.path.join(PLM_EMBEDDING_DIR, f"{start_variant_id}_esm2_650_embeddings.pkl")

    seed_all(307)
    device = torch.device(f"cuda:{test_args.gpu_idx}" if torch.cuda.is_available() else "cpu")
    args, model, checkpoint_path = load_args_and_model(test_args, device)
    seed_all(args.seed)

    print("=" * 80)
    print("Flexible CR ddG evaluation")
    print(f"Dataset: {test_args.dataset}")
    print(f"PDB ID: {pdb_id}")
    print(f"Start mutant: {start_mutant}")
    print(f"Start PDB: {start_pdb_path}")
    print(f"PLM embedding: {embedding_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print("=" * 80)

    plm_embedding = None
    if args.use_plm_embedding:
        plm_embedding = build_plm_embedding(
            pdb_id=pdb_id,
            start_mutant=start_mutant,
            pdb_path=start_pdb_path,
            plm_path=args.plm_path,
            embedding_path=embedding_path,
            embedding_key=start_variant_id,
            device=device,
        )

    dataset = FlexibleCRDataset(
        data_path=data_path,
        pdb_id=pdb_id,
        start_mutant=start_mutant,
        knn_num=args.knn_neighbors_num,
        knn_agents_num=args.knn_agents_num,
        plm_embedding=plm_embedding,
        target_column=target_column,
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
    results_df.insert(1, "start_mutant", start_mutant)
    results_df.insert(2, "target_score_negative", results_df["Ground_Truth"])

    safe_start = start_mutant.replace(",", "__")
    output_dir = os.path.join(test_args.output_dir, test_args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f"{test_args.dataset}_from_{safe_start}_predictions_epoch{test_args.checkpoint_epoch}.csv")
    metrics_file = os.path.join(output_dir, f"{test_args.dataset}_from_{safe_start}_metrics_epoch{test_args.checkpoint_epoch}.csv")
    results_df.to_csv(results_file, index=False)

    metrics.update({
        "dataset": test_args.dataset,
        "pdb_id": pdb_id,
        "start_mutant": start_mutant,
        "start_pdb_path": start_pdb_path,
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
