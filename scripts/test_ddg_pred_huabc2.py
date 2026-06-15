import argparse
import glob
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.MERF_v6 import MERF
from protein.read_pdbs import KnnAgnet, KnnResidue, PaddingCollate, parse_pdb, parse_pdb_chain2sequence
from utils.util import recursive_to, seed_all


class HuABC2Dataset(Dataset):
    def __init__(
        self,
        data_path,
        wt_dir,
        mut_dir,
        knn_num,
        knn_agents_num,
        use_plm_embedding=False,
        plm_path=None,
        plm_embedding_path=None,
        device="cpu",
    ):
        super().__init__()
        self.data_df = pd.read_csv(data_path, dtype={"pdb_id": "string"})
        self.wt_dir = wt_dir
        self.mut_dir = mut_dir
        self.knn_num = knn_num
        self.knn_agents_num = knn_agents_num
        self.use_plm_embedding = use_plm_embedding
        self.plm_path = plm_path
        self.plm_embedding_path = plm_embedding_path
        self.device = device

        if self.use_plm_embedding:
            if self.plm_embedding_path is None and self.plm_path is None:
                raise ValueError("PLM embedding path or PLM model path must be provided.")
            self.process_plm_embedding()

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, index):
        row = self.data_df.iloc[index]
        pdb_file_name = os.path.basename(str(row["pdb_id"]))
        pdb_id = os.path.splitext(pdb_file_name)[0]
        mutate_info = str(row["mutation"])
        ddg = float(row["DDG"])

        mut_file_name = os.path.basename(str(row["path_mut"]))
        if not mut_file_name or mut_file_name == "nan":
            mut_file_name = f"{pdb_id}_{mutate_info}.pdb"

        wt_file_path = os.path.join(self.wt_dir, pdb_file_name)
        mut_file_path = os.path.join(self.mut_dir, mut_file_name)

        complex_wt_info = parse_pdb(wt_file_path)
        complex_mut_info = parse_pdb(mut_file_path)

        if self.use_plm_embedding:
            if pdb_id not in self.plm_embeddings_dict:
                raise KeyError(f"PLM embedding for {pdb_id} not found.")
            plm_embedding = self.plm_embeddings_dict[pdb_id]
            complex_wt_info["plm_embedding"] = plm_embedding
            complex_mut_info["plm_embedding"] = plm_embedding

        transform = KnnResidue(num_neighbors=self.knn_num)
        agent_select = KnnAgnet(num_neighbors=self.knn_agents_num)

        mutation_mask = complex_wt_info["aa"] != complex_mut_info["aa"]
        agent_mask = agent_select({"wt": complex_wt_info, "mut": complex_mut_info, "mutation_mask": mutation_mask})

        complex_wt_info["agent_mask"] = agent_mask
        complex_wt_info["PDB_id"] = pdb_id
        complex_wt_info["mutate_info"] = mutate_info

        complex_mut_info["agent_mask"] = agent_mask
        complex_mut_info["PDB_id"] = pdb_id
        complex_mut_info["mutate_info"] = mutate_info

        batch = transform({"wt": complex_wt_info, "mut": complex_mut_info, "mutation_mask": mutation_mask})
        batch["ddG"] = ddg
        return batch

    def process_plm_embedding(self):
        if self.plm_embedding_path is not None and os.path.exists(self.plm_embedding_path):
            self.plm_embeddings_dict = torch.load(self.plm_embedding_path, map_location="cpu")
            return

        self.plm_tokenizer = AutoTokenizer.from_pretrained(self.plm_path)
        self.plm_model = AutoModel.from_pretrained(self.plm_path)
        self.plm_model.eval()
        self.plm_model = self.plm_model.to(self.device)

        self.plm_embeddings_dict = {}
        unique_pdb_ids = sorted({os.path.splitext(os.path.basename(str(pdb_id)))[0] for pdb_id in self.data_df["pdb_id"]})
        for pdb_id in tqdm(unique_pdb_ids, desc="Processing PLM embeddings"):
            wt_file_path = os.path.join(self.wt_dir, f"{pdb_id}.pdb")
            complex_wt_info = parse_pdb(wt_file_path)
            chain_id_list = complex_wt_info["chain_id"]
            chain2sequence = parse_pdb_chain2sequence(wt_file_path)

            embeddings = []
            for _, sequence in chain2sequence.items():
                inputs = self.plm_tokenizer(sequence, return_tensors="pt")
                inputs = {key: val.to(self.device) for key, val in inputs.items()}
                with torch.no_grad():
                    outputs = self.plm_model(**inputs)
                    embedding = outputs.last_hidden_state.squeeze(0)[1 : len(sequence) + 1, :].cpu()
                embeddings.append(embedding)

            embedding = torch.cat(embeddings, dim=0)
            if embedding.shape[0] != len(chain_id_list):
                raise ValueError(f"PLM embedding length mismatch for {pdb_id}.")
            self.plm_embeddings_dict[pdb_id] = embedding

        if self.plm_embedding_path is not None:
            os.makedirs(os.path.dirname(self.plm_embedding_path), exist_ok=True)
            torch.save(self.plm_embeddings_dict, self.plm_embedding_path)


def find_checkpoint(log_dir, checkpoint_epoch):
    if checkpoint_epoch is not None:
        return os.path.join(log_dir, f"Epoch{checkpoint_epoch}_MERF.pth"), checkpoint_epoch

    checkpoints = glob.glob(os.path.join(log_dir, "Epoch*_MERF.pth"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoint files found in {log_dir}")
    checkpoints.sort(key=lambda x: int(os.path.basename(x).split("Epoch")[1].split("_")[0]))
    checkpoint_path = checkpoints[-1]
    epoch = int(os.path.basename(checkpoint_path).split("Epoch")[1].split("_")[0])
    return checkpoint_path, epoch


def evaluate_model(model, dataloader, device):
    model.eval()
    predictions = []
    ground_truths = []
    pdb_ids = []
    mutation_infos = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating HuABC2"):
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
            mutation_infos.extend(batch["wt"]["mutate_info"])

    pred_array = np.asarray(predictions, dtype=float)
    gt_array = np.asarray(ground_truths, dtype=float)
    valid_mask = ~(np.isnan(gt_array) | np.isnan(pred_array) | np.isinf(gt_array) | np.isinf(pred_array))

    if valid_mask.sum() < 2:
        raise ValueError(f"Need at least 2 valid samples for Pearson r, got {valid_mask.sum()}.")

    pearson_r, pearson_p = pearsonr(pred_array[valid_mask], gt_array[valid_mask])
    spearman_r, spearman_p = spearmanr(pred_array[valid_mask], gt_array[valid_mask])

    results_df = pd.DataFrame(
        {
            "PDB_id": pdb_ids,
            "mutation": mutation_infos,
            "Prediction": pred_array,
            "Ground_Truth": gt_array,
            "is_valid": valid_mask,
        }
    )
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
    parser.add_argument("--log_dir", type=str, default="/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_05_30_22_50_05")
    parser.add_argument("--checkpoint_epoch", type=int, default=200)
    parser.add_argument("--gpu_idx", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--data_path", type=str, default="/home/dataset-local/projects_dir/MERF/data/HuABC2/HuABC2_test.csv")
    parser.add_argument("--wt_dir", type=str, default="/home/dataset-local/projects_dir/MERF/data/HuABC2/PDBs")
    parser.add_argument("--mut_dir", type=str, default="/home/dataset-local/projects_dir/MERF/data/HuABC2/PDBs_mutated")
    parser.add_argument("--plm_embedding_path", type=str, default="/home/dataset-local/projects_dir/MERF/data/HuABC2/huabc2_esm2_650_embeddings.pkl")
    parser.add_argument("--output_dir", type=str, default=None)
    test_args = parser.parse_args()

    args_path = os.path.join(test_args.log_dir, "args.pkl")
    with open(args_path, "rb") as f:
        args = pickle.load(f)

    if test_args.gpu_idx is not None:
        args.gpu_idx = test_args.gpu_idx

    checkpoint_path, checkpoint_epoch = find_checkpoint(test_args.log_dir, test_args.checkpoint_epoch)
    output_dir = test_args.output_dir or test_args.log_dir
    os.makedirs(output_dir, exist_ok=True)

    seed_all(args.seed)
    device = torch.device(args.gpu_idx if args.is_cuda else "cpu")

    print("=" * 80)
    print("HuABC2 Testing Configuration:")
    print(f"Log directory: {test_args.log_dir}")
    print(f"Checkpoint: {checkpoint_path} (Epoch {checkpoint_epoch})")
    print(f"Data path: {test_args.data_path}")
    print(f"WT directory: {test_args.wt_dir}")
    print(f"Mutant directory: {test_args.mut_dir}")
    print(f"Batch size: {test_args.batch_size}")
    print(f"Device: {device}")
    print("=" * 80)

    model = MERF(args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    dataset = HuABC2Dataset(
        test_args.data_path,
        test_args.wt_dir,
        test_args.mut_dir,
        knn_num=args.knn_neighbors_num,
        knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding,
        plm_path=args.plm_path,
        plm_embedding_path=test_args.plm_embedding_path,
        device=device,
    )
    print(f"Dataset size: {len(dataset)}")

    dataloader = DataLoader(
        dataset,
        batch_size=test_args.batch_size,
        shuffle=False,
        pin_memory=False,
        drop_last=False,
        num_workers=test_args.num_workers,
        collate_fn=PaddingCollate(),
    )

    results_df, metrics = evaluate_model(model, dataloader, device)

    results_file = os.path.join(output_dir, f"huabc2_predictions_epoch{checkpoint_epoch}.csv")
    metrics_file = os.path.join(output_dir, f"huabc2_metrics_epoch{checkpoint_epoch}.csv")
    results_df.to_csv(results_file, index=False)
    pd.DataFrame([metrics]).to_csv(metrics_file, index=False)

    print("\nHuABC2 Metrics:")
    print(f"  Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
    print(f"  Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")
    print(f"  Valid samples: {metrics['num_valid_samples']}/{metrics['num_total_samples']}")
    print(f"Results saved to: {results_file}")
    print(f"Metrics saved to: {metrics_file}")


if __name__ == "__main__":
    main()
