"""Run the v8 CR evolution algorithm from an arbitrary reverse-mutated start."""

import argparse
import datetime
import os
import sys

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

# 允许从项目根目录以外的位置直接执行该脚本。
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_cr_evo_flexible_v1 import (
    FlexibleCREvoDataset,
    canonical_reverse_mutation_key,
)
from protein.read_pdbs import PaddingCollate
from utils.losshistory import LossHistory
from utils.util import seed_all

# 复用 v8 的模型调用、PPO 训练循环、奖励聚合和日志指标，避免改变进化算法。
from evo_cr_v8 import (
    MERF,
    build_optimizer,
    configure_trainable_params,
    get_args as get_v8_args,
    load_model_args_from_checkpoint,
    load_pretrained,
    train_one_epoch,
)


CR_DIR = "/home/dataset-local/projects_dir/MERF/data/CR"


def normalize_dataset_name(dataset):
    """Accept both cr_6261_h9 and the on-disk PDB id cr6261_h9."""
    value = dataset.strip()
    parts = value.split("_")
    if len(parts) == 3 and parts[0] == "cr" and parts[1].isdigit() and parts[2].startswith("h"):
        return f"cr{parts[1]}_{parts[2]}"
    if len(parts) == 2 and parts[0].startswith("cr") and parts[0][2:].isdigit() and parts[1].startswith("h"):
        return value
    raise ValueError("--dataset must look like cr_6261_h9 or cr6261_h9")


def get_args():
    flexible_parser = argparse.ArgumentParser(add_help=False)
    flexible_parser.add_argument("--dataset", default="cr6261_h9", help="CR dataset, e.g. cr_6261_h9")
    flexible_parser.add_argument(
        "--start_reverse_mutations",
        "--start_mutant",
        dest="start_reverse_mutations",
        default="RA30S,FA75S,AA76T,VA79A,VA104L",
        help="Comma-separated mature-to-start reverse mutations; empty string denotes mature antibody",
    )
    flexible_parser.add_argument(
        "--log_root",
        default="logs/evo_cr_flexible_v1",
        help="Directory for evolution logs and metrics",
    )
    flexible_args, remaining_args = flexible_parser.parse_known_args()

    # 让 v8 原始参数解析器继续处理其全部超参数，保持训练框架和默认值一致。
    import sys

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining_args]
        args = get_v8_args()
    finally:
        sys.argv = original_argv

    args.dataset = normalize_dataset_name(flexible_args.dataset)
    args.start_reverse_mutations = flexible_args.start_reverse_mutations
    args.log_root = flexible_args.log_root
    return args


def build_dataset(args, device):
    metadata_df = pd.read_csv(os.path.join(CR_DIR, "cr_evo.csv"), dtype={"pdb_id": "string"})
    metadata_rows = metadata_df[metadata_df["pdb_id"] == args.dataset]
    if metadata_rows.empty:
        raise ValueError(f"{args.dataset} not found in {CR_DIR}/cr_evo.csv")
    row = metadata_rows.iloc[0]

    # Dataset 内部将起点规范化；这里使用其同名的结构/embedding 缓存命名规则。
    start_key = canonical_reverse_mutation_key(args.start_reverse_mutations)
    variant_id = args.dataset if start_key == "" else f"{args.dataset}_{start_key}"
    plm_embedding_path = os.path.join(CR_DIR, "plm_embeddings", f"{variant_id}_esm2_650_embeddings.pkl")

    return FlexibleCREvoDataset(
        pdb_id=args.dataset,
        antibody_chain=row["antibody_chain"],
        partner=row["partner"],
        sequence=row["sequence"],
        cdr=row["cdr3"],
        cr_dir=CR_DIR,
        pdb_dir=os.path.join(CR_DIR, "PDBs"),
        fixed_dir=os.path.join(CR_DIR, "PDBs_fixed"),
        mutated_dir=os.path.join(CR_DIR, "PDBs_mutated"),
        start_reverse_mutations=start_key,
        knn_num=args.knn_neighbors_num,
        knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding,
        plm_path=args.plm_path,
        plm_embedding_path=plm_embedding_path,
        device=device,
    )


def main():
    args = get_args()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")

    train_dataset = build_dataset(args, device)
    collate_fn = PaddingCollate()

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    loss_dir = os.path.join(args.log_root, f"{time_str}_{args.run_name}", train_dataset.pdb_id)
    loss_history = LossHistory(loss_dir, is_evolve=True)
    writer = SummaryWriter(log_dir=loss_history.save_path)
    loss_history.write(str(args) + "\n")
    loss_history.write(f"Flexible CR dataset: {args.dataset}\n")
    loss_history.write(f"Start reverse key: {train_dataset.start_reverse_key}\n")
    loss_history.write(f"Start PDB: {train_dataset.immature_pdb_path}\n")
    loss_history.write(f"Start PLM embedding: {train_dataset.plm_embedding_path}\n")
    if checkpoint_args_path is not None:
        loss_history.write(f"Loaded model args from {checkpoint_args_path}\n")
        loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')
    loss_history.write(f"PDB-backed landscape candidates: {len(train_dataset.candidate_reverse_keys)}\n")
    loss_history.write(f"Global ranked candidates for evaluation: {len(train_dataset.rank_by_reverse_key)}\n")
    loss_history.write(f"Reward paths: {args.reward_model_paths}\n")

    evo_model = MERF(args).to(device)
    load_pretrained(evo_model, args.model_load_path, device)
    configure_trainable_params(evo_model, args)
    optimizer = build_optimizer(evo_model, args)

    reference_model = MERF(args).to(device)
    load_pretrained(reference_model, args.model_load_path, device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    reward_models = []
    for reward_path in [path for path in args.reward_model_paths.split(",") if path]:
        reward_model = MERF(args).to(device)
        load_pretrained(reward_model, reward_path, device)
        reward_model.eval()
        for param in reward_model.parameters():
            param.requires_grad = False
        reward_models.append(reward_model)
    if not reward_models:
        raise ValueError("No reward models loaded")

    reward_baseline = None
    valid_ranks = []
    low_entropy_epochs = 0
    for epoch in range(args.total_epochs):
        reward_baseline, valid_ranks, metrics = train_one_epoch(
            args,
            evo_model,
            reward_models,
            reference_model,
            optimizer,
            epoch,
            args.total_epochs,
            train_dataset,
            collate_fn,
            loss_history,
            device,
            reward_baseline,
            valid_ranks,
            writer=writer,
        )
        if args.entropy_stop and epoch + 1 >= args.entropy_stop_min_epoch:
            if metrics["raw_policy_entropy"] < args.entropy_stop_threshold:
                low_entropy_epochs += 1
            else:
                low_entropy_epochs = 0
            if low_entropy_epochs >= args.entropy_stop_patience:
                save_file = os.path.join(loss_history.save_path, train_dataset.pdb_id + ".csv")
                save_df = pd.read_csv(save_file)
                save_df.loc[save_df.index[-1], "early_stop_reason"] = (
                    f"entropy_below_{args.entropy_stop_threshold}_for_"
                    f"{args.entropy_stop_patience}_epochs"
                )
                save_df.to_csv(save_file, index=False)
                loss_history.write(f"Early stopped: {save_df.loc[save_df.index[-1], 'early_stop_reason']}\n")
                break

    writer.close()


if __name__ == "__main__":
    main()
