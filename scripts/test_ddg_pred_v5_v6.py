import os
import numpy as np
import pandas as pd
import pickle
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error

from model.MERF_v6 import MERF
from dataset.dataset_ddg import VMDataset, DDGBaseDataset, CRDataset
from utils.arguments import get_pretrain_args
from utils.util import seed_all, recursive_to
from protein.read_pdbs import PaddingCollate
from tqdm import tqdm


def load_dataset(dataset_name, args, device):
    """Load dataset based on dataset name"""

    if dataset_name == 'vm':
        data_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/valid_data.csv"
        wt_dir = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/PDBs_fixed"
        mut_dir = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/PDBs_mutated"
        embedding_path = "/home/dataset-local/projects_dir/MERF/data/VenusMaxwell/valid_esm2_650_embeddings.pkl"

        dataset = VMDataset(
            data_path, wt_dir, mut_dir,
            knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
            use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
            plm_embedding_path=embedding_path, device=device
        )
        has_ground_truth = True

    elif dataset_name == 'abbind':
        data_path = '/home/dataset-local/projects_dir/MERF/data/ABbind/AB-Bind_645pMulti.csv'
        wt_dir = '/home/dataset-local/projects_dir/MERF/data/ABbind/PDBs_fixed'
        mut_dir = '/home/dataset-local/projects_dir/MERF/data/ABbind/PDBs_mutated'
        embedding_path = '/home/dataset-local/projects_dir/MERF/data/ABbind/abbind_esm2_650_embeddings.pkl'

        dataset = DDGBaseDataset(
            data_path, wt_dir, mut_dir,
            knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
            use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
            plm_embedding_path=embedding_path, device=device
        )
        has_ground_truth = True

    elif dataset_name.startswith('cr_'):
        # CR datasets: cr_6261_h1, cr_6261_h9, cr_9114_h1, cr_9114_h3
        parts = dataset_name.split('_')  # ['cr', '6261', 'h1']
        antibody = parts[1]  # '6261'
        target = parts[2]    # 'h1'

        data_path = f'/home/dataset-local/projects_dir/MERF/data/CR/cr{antibody}_{target}.csv'
        wt_dir = '/home/dataset-local/projects_dir/MERF/data/CR/PDBs_fixed'
        mut_dir = '/home/dataset-local/projects_dir/MERF/data/CR/PDBs_mutated'
        embedding_path = f'/home/dataset-local/projects_dir/MERF/data/CR/cr{antibody}_{target}_esm2_650_embeddings.pkl'

        target_column = f'{target}_score'

        dataset = CRDataset(
            data_path, wt_dir, mut_dir,
            knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
            use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
            plm_embedding_path=embedding_path, device=device,
            target_column=target_column
        )
        has_ground_truth = True

    elif dataset_name == '7fae':
        data_path = '/home/dataset-local/projects_dir/MERF/data/7FAE/7FAE.csv'
        wt_dir = '/home/dataset-local/projects_dir/MERF/data/7FAE/PDBs_fixed'
        mut_dir = '/home/dataset-local/projects_dir/MERF/data/7FAE/PDBs_mutated'
        embedding_path = '/home/dataset-local/projects_dir/MERF/data/7FAE/7fae_esm2_650_embeddings.pkl'

        dataset = DDGBaseDataset(
            data_path, wt_dir, mut_dir,
            knn_num=args.knn_neighbors_num, knn_agents_num=args.knn_agents_num,
            use_plm_embedding=args.use_plm_embedding, plm_path=args.plm_path,
            plm_embedding_path=embedding_path, device=device
        )
        has_ground_truth = False  # No ground truth in the CSV

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    return dataset, has_ground_truth


def evaluate_model(model, dataloader, device, has_ground_truth=True, is_cr_dataset=False, is_7fae_dataset=False):
    """Evaluate model on a dataset

    Args:
        is_cr_dataset: If True, only compute correlation metrics (for CR datasets where values are not ddG)
        is_7fae_dataset: If True, compute recall metrics for key mutations
    """
    model.eval()

    predictions = []
    ground_truths = []
    pdb_ids = []
    mutation_infos = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = recursive_to(batch, device)

            # Get prediction (shape: [batch_size] or scalar for batch_size=1)
            pred, _, _, _ = model(batch, device)

            # Convert to list and extend predictions
            if pred.dim() == 0:  # scalar (batch_size=1)
                predictions.append(pred.item())
            else:  # batch (batch_size>1)
                predictions.extend(pred.cpu().tolist())

            # Get ground truth if available
            if has_ground_truth:
                ddg = batch['ddG']
                if ddg.dim() == 0:  # scalar
                    ground_truths.append(ddg.item())
                else:  # batch
                    ground_truths.extend(ddg.cpu().tolist())

            # Get metadata (handle batch)
            batch_pdb_ids = batch['wt']['PDB_id']
            batch_mutate_infos = batch['wt']['mutate_info']

            # These are lists in the batch
            pdb_ids.extend(batch_pdb_ids)
            mutation_infos.extend(batch_mutate_infos)

    # Create results dataframe
    results = {
        "PDB_id": np.array(pdb_ids),
        "mutate_info": np.array(mutation_infos),
        "Prediction": np.array(predictions)
    }

    if has_ground_truth:
        results["Ground_Truth"] = np.array(ground_truths)

    results_df = pd.DataFrame(results)

    # Calculate metrics
    metrics = {}

    # For 7FAE dataset, compute recall metrics
    if is_7fae_dataset:
        # Key mutations for 7FAE
        key_mutations = ['TH31W', 'AH53F', 'NH57L', 'RH103M', 'LH104F']

        # Sort predictions in ascending order (lower prediction = better)
        results_df['rank'] = results_df['Prediction'].rank(ascending=True, method='min').astype(int)
        # sorted_df = results_df.sort_values('Prediction', ascending=False).reset_index(drop=True)
        sorted_df = results_df.sort_values('Prediction', ascending=True).reset_index(drop=True)

        # Find ranks of key mutations
        key_mutation_ranks = {}
        key_mutation_found = {}
        for key_mut in key_mutations:
            mask = results_df['mutate_info'] == key_mut
            if mask.any():
                rank = results_df.loc[mask, 'rank'].values[0]
                key_mutation_ranks[key_mut] = rank
                key_mutation_found[key_mut] = True
            else:
                key_mutation_ranks[key_mut] = -1  # Not found
                key_mutation_found[key_mut] = False

        # Calculate recall metrics
        found_mutations = [k for k, v in key_mutation_found.items() if v]
        num_found = len(found_mutations)
        total_mutations = len(key_mutations)

        # Top-K recall
        total_samples = len(results_df)
        for k in [5, 10, 20, 50, 100]:
            if k <= total_samples:
                top_k_mutations = sorted_df.head(k)['mutate_info'].tolist()
                recall_at_k = sum([1 for mut in key_mutations if mut in top_k_mutations]) / total_mutations
                metrics[f'recall@{k}'] = recall_at_k

        # Average rank of key mutations (only for found mutations)
        if num_found > 0:
            valid_ranks = [rank for rank in key_mutation_ranks.values() if rank > 0]
            metrics['avg_rank'] = np.mean(valid_ranks)
            metrics['median_rank'] = np.median(valid_ranks)
        else:
            metrics['avg_rank'] = -1
            metrics['median_rank'] = -1

        # Store individual ranks
        for key_mut, rank in key_mutation_ranks.items():
            metrics[f'rank_{key_mut}'] = rank

        metrics['num_found'] = num_found
        metrics['total_key_mutations'] = total_mutations

    elif has_ground_truth:
        pred_array = np.array(predictions)
        gt_array = np.array(ground_truths)

        # Filter out NaN and inf values
        valid_mask = ~(np.isnan(gt_array) | np.isnan(pred_array) |
                       np.isinf(gt_array) | np.isinf(pred_array))

        num_invalid = (~valid_mask).sum()
        num_total = len(gt_array)
        num_valid = valid_mask.sum()

        if num_invalid > 0:
            print(f"  Warning: Filtered out {num_invalid}/{num_total} samples with NaN/inf values ({100*num_invalid/num_total:.2f}%)")

        pred_array_valid = pred_array[valid_mask]
        gt_array_valid = gt_array[valid_mask]

        # Pearson correlation
        pearson_r, pearson_p = pearsonr(pred_array_valid, gt_array_valid)
        metrics['pearson_r'] = pearson_r
        metrics['pearson_p'] = pearson_p

        # Spearman correlation
        spearman_r, spearman_p = spearmanr(pred_array_valid, gt_array_valid)
        metrics['spearman_r'] = spearman_r
        metrics['spearman_p'] = spearman_p

        # For CR datasets, only compute correlation metrics
        if is_cr_dataset:
            metrics['mse'] = 0.0
            metrics['rmse'] = 0.0
            metrics['mae'] = 0.0
        else:
            # MSE and RMSE
            mse = mean_squared_error(gt_array_valid, pred_array_valid)
            rmse = np.sqrt(mse)
            metrics['mse'] = mse
            metrics['rmse'] = rmse

            # MAE
            mae = mean_absolute_error(gt_array_valid, pred_array_valid)
            metrics['mae'] = mae

        metrics['num_valid_samples'] = num_valid
        metrics['num_total_samples'] = num_total

    return results_df, metrics


def main():
    # ----------------------- Parse arguments ----------------------- #
    import argparse

    # Add test-specific arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default="/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_05_28_23_53_09",
                       help='Path to log directory (e.g., logs/pretrain_vm/2025_11_22_00_40_23)')
    parser.add_argument('--test_datasets', type=str, default='cr_6261_h1',
                       help='Comma-separated list of datasets to test (e.g., vm,abbind,cr_6261_h1,7fae)')
    parser.add_argument('--checkpoint_epoch', type=int, default=200,
                       help='Epoch number to test (default: use last epoch based on Epoch*_MERF.pth files)')
    parser.add_argument('--gpu_idx', type=int, default=3,
                       help='GPU index to use (default: use value from training args)')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for inference (default: 128)')

    test_args = parser.parse_args()

    # Load args from log directory
    args_path = os.path.join(test_args.log_dir, 'args.pkl')
    with open(args_path, 'rb') as f:
        args = pickle.load(f)
    print(f"Loaded args from: {args_path}")

    # Override gpu_idx if provided
    if test_args.gpu_idx is not None:
        args.gpu_idx = test_args.gpu_idx
        print(f"Overriding gpu_idx: {args.gpu_idx}")

    # Find checkpoint path and extract epoch number
    if test_args.checkpoint_epoch is not None:
        checkpoint_path = os.path.join(test_args.log_dir, f'Epoch{test_args.checkpoint_epoch}_MERF.pth')
        checkpoint_epoch = test_args.checkpoint_epoch
    else:
        # Find the latest checkpoint
        import glob
        checkpoints = glob.glob(os.path.join(test_args.log_dir, 'Epoch*_MERF.pth'))
        if not checkpoints:
            raise FileNotFoundError(f"No checkpoint files found in {test_args.log_dir}")
        checkpoints.sort(key=lambda x: int(x.split('Epoch')[1].split('_')[0]))
        checkpoint_path = checkpoints[-1]
        checkpoint_epoch = int(checkpoint_path.split('Epoch')[1].split('_')[0])
        print(f"Using latest checkpoint: {checkpoint_path}")

    # Set output directory to log directory
    output_dir = test_args.log_dir

    print("=" * 80)
    print("Testing Configuration:")
    print(f"Log directory: {test_args.log_dir}")
    print(f"Checkpoint: {checkpoint_path} (Epoch {checkpoint_epoch})")
    print(f"Test datasets: {test_args.test_datasets}")
    print(f"Batch size: {test_args.batch_size}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)

    # ----------------------- Setup environment ----------------------- #
    seed = args.seed
    seed_all(seed)
    print(f"Setting random seed...{seed}")

    GPU_indx = args.gpu_idx
    device = torch.device(GPU_indx if args.is_cuda else "cpu")
    print(f"Using device: {device}")

    # ----------------------- Load model ----------------------- #
    print(f"\nLoading model from {checkpoint_path}...")
    model = MERF(args).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()

    # ----------------------- Test on each dataset ----------------------- #
    dataset_names = [name.strip() for name in test_args.test_datasets.split(',')]

    all_metrics = {}

    for dataset_name in dataset_names:
        print("\n" + "=" * 80)
        print(f"Testing on dataset: {dataset_name}")
        print("=" * 80)

        # Load dataset
        print(f"Loading {dataset_name} dataset...")
        dataset, has_ground_truth = load_dataset(dataset_name, args, device)

        print(f"Dataset size: {len(dataset)}")

        # Create dataloader
        collate_fn = PaddingCollate()
        dataloader = DataLoader(
            dataset,
            batch_size=test_args.batch_size,
            shuffle=False,
            pin_memory=False,
            drop_last=False,
            num_workers=64,
            collate_fn=collate_fn
        )

        # Evaluate
        # Check dataset type for specific metrics
        is_cr_dataset = dataset_name.startswith('cr_')
        is_7fae_dataset = dataset_name == '7fae'
        results_df, metrics = evaluate_model(model, dataloader, device, has_ground_truth, is_cr_dataset, is_7fae_dataset)

        # Save results with epoch information
        results_file = os.path.join(output_dir, f"{dataset_name}_predictions_epoch{checkpoint_epoch}.csv")
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")

        # Print and save metrics
        if is_7fae_dataset:
            print(f"\nRecall Metrics for {dataset_name}:")
            print(f"  Found {metrics['num_found']}/{metrics['total_key_mutations']} key mutations")

            # Print recall@k metrics
            for k in [5, 10, 20, 50, 100]:
                if f'recall@{k}' in metrics:
                    print(f"  Recall@{k}: {metrics[f'recall@{k}']:.4f}")

            # Print ranking metrics
            if metrics['avg_rank'] > 0:
                print(f"  Average Rank: {metrics['avg_rank']:.2f}")
                print(f"  Median Rank: {metrics['median_rank']:.2f}")

            # Print individual key mutation ranks
            print(f"\n  Individual Key Mutation Ranks:")
            key_mutations = ['TH31W', 'AH53F', 'NH57L', 'RH103M', 'LH104F']
            for key_mut in key_mutations:
                rank = metrics[f'rank_{key_mut}']
                if rank > 0:
                    print(f"    {key_mut}: Rank {rank}")
                else:
                    print(f"    {key_mut}: Not found")

            all_metrics[dataset_name] = metrics

        elif has_ground_truth:
            print(f"\nMetrics for {dataset_name}:")
            print(f"  Pearson r: {metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4e})")
            print(f"  Spearman r: {metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4e})")

            if not is_cr_dataset:
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                print(f"  MSE: {metrics['mse']:.4f}")

            all_metrics[dataset_name] = metrics
        else:
            print(f"\nNo ground truth available for {dataset_name}, only predictions saved.")

    # ----------------------- Save summary metrics ----------------------- #
    if all_metrics:
        metrics_file = os.path.join(output_dir, f"metrics_summary_epoch{checkpoint_epoch}.csv")
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.to_csv(metrics_file)
        print("\n" + "=" * 80)
        print("Summary metrics:")
        print(metrics_df.to_string())
        print(f"\nMetrics summary saved to: {metrics_file}")
        print("=" * 80)


if __name__ == '__main__':
    main()
