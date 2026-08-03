#!/home/dataset-local/anaconda3/envs/merf/bin/python
"""SABDAB v3 进化脚本。

v3 保留 SABDAB 的 GRPO 与能量奖励，但将策略 WT、FoldX 父结构和候选评分
统一到 RepairPDB 输出；最终外部评测只使用最后一次训练更新后的策略提名的
top-8 候选，绝不从整个进化过程的历史候选池中挑选。
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import pickle
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_evo_sabdab_v3 import SAbDabEvoDatasetV3
from model.MERF_v6 import MERF
from protein.mutate_scripts_foldx_sabdab_v3 import (
    DEFAULT_FOLDX_BIN,
    format_mutation_token,
    mutant_pdb_path,
)
from protein.read_pdbs import PaddingCollate
from utils.losshistory import LossHistory
from utils.util import recursive_to, seed_all


PROJECT_DIR = Path("/home/dataset-local/projects_dir/MERF")
DEFAULT_COLABDESIGN_PYTHON = "/home/dataset-local/anaconda3/envs/colabdesign-af2/bin/python"
DEFAULT_COLABDESIGN_SCRIPT = PROJECT_DIR / "scripts" / "calculate_colabdesign_v3.py"
DEFAULT_AF2_PARAMS = Path("/home/dataset-local/software_package/ColabDesign/params")
AA_KEY = "ACDEFGHIKLMNPQRSTVWY"
MONITORED_COLABDESIGN_METRICS = (
    "af2_pLDDT",
    "af2_pLDDT_primary_chain",
    "af2_pTM",
    "af2_i_pTM",
    "af2_pAE",
    "af2_i_pAE",
    "af2_min_ipAE",
    "af2_min_ipSAE",
    "af2_avg_ipSAE",
)
_PYROSETTA_INITIALIZED = False


CHECKPOINT_MODEL_ARG_KEYS = (
    "seed",
    "use_plm_embedding",
    "plm_path",
    "feat_dim",
    "rel_dim",
    "max_relpos",
    "ipa_layer",
    "ga_layer",
    "knn_neighbors_num",
    "knn_agents_num",
    "obs_shape",
    "n_actions",
    "n_agents",
    "agent_hidden_dim",
    "mixer_rel_dim",
    "mixer_ga_layer",
    "mixing_embed_dim",
    "hypernet_embed",
    "hypernet_layers",
    "use_kl_loss",
    "kl_loss_weight",
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def get_args():
    parser = argparse.ArgumentParser(description="SABDAB v3 GRPO evolution")

    # 运行和优化器：默认值与 CR v4 的全模型微调设置对齐。
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--is_cuda", type=str2bool, default=True)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="v3")
    parser.add_argument("--log_root", type=str, default="logs/evo_sabdab_v3")

    # 预训练模型与结构编码器参数。
    parser.add_argument("--use_plm_embedding", type=str2bool, default=True)
    parser.add_argument(
        "--plm_path",
        type=str,
        default="/home/dataset-local/projects_dir/pretrained_model/models--facebook--esm2_t33_650M_UR50D/",
    )
    parser.add_argument("--feat_dim", type=int, default=128)
    parser.add_argument("--rel_dim", type=int, default=128)
    parser.add_argument("--max_relpos", type=int, default=32)
    parser.add_argument("--ipa_layer", type=int, default=3)
    parser.add_argument("--ga_layer", type=int, default=3)
    parser.add_argument("--knn_neighbors_num", type=int, default=128)
    parser.add_argument("--knn_agents_num", type=int, default=20)
    parser.add_argument(
        "--model_load_path",
        type=str,
        default="/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_06_09_16_56_32/Epoch200_MERF.pth",
    )
    parser.add_argument("--obs_shape", type=int, default=128)
    parser.add_argument("--n_actions", type=int, default=20)
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--agent_hidden_dim", type=int, default=32)
    parser.add_argument("--mixer_rel_dim", type=int, default=512)
    parser.add_argument("--mixer_ga_layer", type=int, default=3)
    parser.add_argument("--mixing_embed_dim", type=int, default=32)
    parser.add_argument("--hypernet_embed", type=int, default=512)
    parser.add_argument("--hypernet_layers", type=int, default=2)
    parser.add_argument("--use_kl_loss", type=str2bool, default=False)
    parser.add_argument("--kl_loss_weight", type=float, default=0.0)

    # SABDAB 数据和 v3 结构缓存。v3 独立目录避免复用 v2 的无来源 PDB。
    parser.add_argument("--train_csv", type=str, default=str(PROJECT_DIR / "data/sabdab/sabdab_evo.csv"))

    # parser.add_argument("--dataset_idx", type=int, default=None)
    parser.add_argument("--dataset_idx", type=int, default=0)

    parser.add_argument("--wt_dir", type=str, default=str(PROJECT_DIR / "data/sabdab/PDBs"))
    parser.add_argument("--fixed_dir", type=str, default=str(PROJECT_DIR / "data/sabdab/PDBs_fixed_v3"))
    parser.add_argument("--mut_dir", type=str, default=str(PROJECT_DIR / "data/sabdab/PDBs_evo_v3"))
    parser.add_argument(
        "--plm_embedding_path",
        type=str,
        default=str(PROJECT_DIR / "data/sabdab/PLM_embeddings_sabdab.pkl"),
    )
    parser.add_argument("--foldx_workers", type=int, default=8)
    parser.add_argument("--foldx_bin", type=str, default=DEFAULT_FOLDX_BIN)

    # 保持 GRPO，而非把 CR 的 PPO 机制迁入。
    parser.add_argument("--comb_num", type=int, default=3)
    parser.add_argument("--total_epochs", type=int, default=5)
    parser.add_argument("--inner_epochs", type=int, default=6)
    parser.add_argument("--inner_batch_size", type=int, default=8)
    parser.add_argument("--kl_coeff", type=float, default=0.5)

    # reward model 与候选评分设置。
    parser.add_argument("--reward_model_paths", type=str, default="")
    parser.add_argument("--reward_label", type=str, default="reward")
    parser.add_argument("--uncertainty_weight", type=float, default=0.5)
    parser.add_argument("--score_agg", choices=["mean", "median"], default="mean")
    parser.add_argument("--reward_transform", choices=["raw", "rank", "clip"], default="raw")
    parser.add_argument("--reward_clip_mad", type=float, default=3.0)
    parser.add_argument("--reward_direction", choices=["min", "max"], default="min")
    parser.add_argument("--candidate_score_batch_size", type=int, default=64)

    # 最终只评估最后一轮策略提名的 top-k；默认严格为用户要求的 8 个。
    parser.add_argument("--final_top_k", type=int, choices=[8], default=8)
    parser.add_argument("--run_pyrosetta", type=str2bool, default=True)
    parser.add_argument("--run_colabdesign", type=str2bool, default=True)
    parser.add_argument("--colabdesign_include_wt", type=str2bool, default=True)
    parser.add_argument("--colabdesign_python", type=str, default=DEFAULT_COLABDESIGN_PYTHON)
    parser.add_argument("--colabdesign_script", type=str, default=str(DEFAULT_COLABDESIGN_SCRIPT))
    parser.add_argument("--colabdesign_af2_params", type=str, default=str(DEFAULT_AF2_PARAMS))
    parser.add_argument("--colabdesign_model_index", type=int, default=0)
    parser.add_argument("--colabdesign_recycles", type=int, default=3)
    parser.add_argument("--colabdesign_cuda_visible_devices", type=str, default="2")
    parser.add_argument(
        "--periodic_external_eval_interval",
        type=int,
        default=1,
        help="每隔多少个已完成的 GRPO epoch 监测当前策略 top-8 的 Rosetta/AF2 指标",
    )

    args = parser.parse_args()
    if not args.reward_model_paths:
        args.reward_model_paths = args.model_load_path
    for name in (
        "comb_num",
        "total_epochs",
        "inner_epochs",
        "inner_batch_size",
        "candidate_score_batch_size",
        "final_top_k",
        "foldx_workers",
        "colabdesign_recycles",
        "periodic_external_eval_interval",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} 必须为正数")
    if args.lr <= 0:
        raise ValueError("--lr 必须为正数")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay 不能为负数")
    return args


def load_model_args_from_checkpoint(args):
    args_path = os.path.join(os.path.dirname(os.path.abspath(args.model_load_path)), "args.pkl")
    if not os.path.exists(args_path):
        print(f"Checkpoint args not found at {args_path}; using evolution args for model init.")
        return None, []
    with open(args_path, "rb") as handle:
        checkpoint_args = pickle.load(handle)
    if not hasattr(checkpoint_args, "__dict__"):
        raise TypeError(f"Unsupported checkpoint args type in {args_path}: {type(checkpoint_args)}")
    checkpoint_args_dict = vars(checkpoint_args)
    loaded_keys = []
    for key in CHECKPOINT_MODEL_ARG_KEYS:
        if key in checkpoint_args_dict:
            setattr(args, key, checkpoint_args_dict[key])
            loaded_keys.append(key)
    print(f"Loaded model args from: {args_path}")
    print(f'Overrode model args: {", ".join(loaded_keys)}')
    return args_path, loaded_keys


def load_pretrained(model, path, device):
    state = torch.load(path, map_location=device)
    model.load_state_dict(state)
    print(f"Loaded {path}")


def parse_reward_model_paths(paths):
    return [path.strip() for path in paths.split(",") if path.strip()]


def aggregate_score_stack(score_stack, score_agg):
    if score_agg == "median":
        center_score = score_stack.median(dim=0).values
    elif score_agg == "mean":
        center_score = score_stack.mean(dim=0)
    else:
        raise ValueError(f"Unsupported score_agg: {score_agg}")
    std_score = (
        score_stack.std(dim=0, unbiased=False)
        if score_stack.size(0) > 1
        else torch.zeros_like(center_score)
    )
    return center_score, std_score


def transform_acquisition(acquisition, reward_transform, clip_mad):
    if reward_transform == "raw":
        return acquisition
    if reward_transform == "rank":
        if acquisition.numel() <= 1:
            return torch.zeros_like(acquisition)
        order = torch.argsort(acquisition.detach())
        ranks = torch.empty_like(acquisition)
        ranks[order] = torch.arange(
            acquisition.numel(), device=acquisition.device, dtype=acquisition.dtype
        )
        return ranks / float(acquisition.numel() - 1)
    if reward_transform == "clip":
        median = acquisition.detach().median()
        mad = (acquisition.detach() - median).abs().median()
        scale = torch.clamp(1.4826 * mad, min=1e-6)
        return acquisition.clamp(median - clip_mad * scale, median + clip_mad * scale)
    raise ValueError(f"Unsupported reward_transform: {reward_transform}")


def make_acquisition(center_scores, std_scores, uncertainty_weight, reward_direction):
    if reward_direction == "min":
        return center_scores + uncertainty_weight * std_scores
    if reward_direction == "max":
        return -center_scores + uncertainty_weight * std_scores
    raise ValueError(f"Unsupported reward_direction: {reward_direction}")


def canonical_mutate_info(mutations):
    """对实际发生的突变排序，并保留 PDB insertion code。"""
    ordered = sorted(mutations, key=lambda item: (item[1], item[2], item[0], item[3], item[4]))
    return ",".join(
        format_mutation_token(chain, position, icode, wt_aa, mut_aa)
        for chain, position, icode, wt_aa, mut_aa in ordered
    )


def generate_mutate_info_list(args, evo_model, train_dataset, collate_fn, device):
    """根据当前策略产生唯一的实际突变候选，不保留任何自替换。"""
    policy_batch = recursive_to(collate_fn([train_dataset[0]]), device)
    pdb_id = policy_batch["expand_data_info"]["pdb_id"][0]
    antibody_chain = policy_batch["expand_data_info"]["antibody_chain"][0]
    cdr_mask = policy_batch["mutation_mask"][0]
    true_indices = torch.nonzero(cdr_mask, as_tuple=True)[0]
    if args.comb_num > len(true_indices):
        raise ValueError(
            f"comb_num={args.comb_num} exceeds the {len(true_indices)} mutable CDR residues for {pdb_id}"
        )

    mutate_info_list = []
    evo_model.eval()
    with torch.no_grad():
        for mutate_indices in tqdm(
            combinations(true_indices, args.comb_num),
            desc=f"{pdb_id}: 生成 CDR 候选",
        ):
            mutation_mask = torch.zeros_like(cdr_mask, dtype=torch.bool)
            mutation_mask[torch.stack(mutate_indices)] = True
            policy_batch["mutation_mask"] = mutation_mask.unsqueeze(0)
            qs = evo_model.choose_best_action(policy_batch, device)
            actions = torch.argmin(qs[policy_batch["mutation_mask"]], dim=1)
            positions = policy_batch["wt"]["resseq"][0][mutation_mask]
            states = policy_batch["wt"]["aa"][0][mutation_mask]
            selected_indices = torch.nonzero(mutation_mask, as_tuple=True)[0]
            # PaddingCollate 将单样本的 insertion code 保留为一个字符串；按全局
            # PDB 索引取字符，防止 C100A/C100B 等位点折叠为相同候选字符串。
            icodes = policy_batch["wt"]["icode"][0]

            actual_mutations = []
            for global_index, state, action, position in zip(selected_indices, states, actions, positions):
                state_index = int(state.item())
                action_index = int(action.item())
                if state_index == action_index:
                    continue
                actual_mutations.append(
                    (
                        antibody_chain,
                        int(position.item()),
                        icodes[int(global_index.item())],
                        AA_KEY[state_index],
                        AA_KEY[action_index],
                    )
                )
            if actual_mutations:
                mutate_info_list.append(canonical_mutate_info(actual_mutations))

    # 只有 canonical 的实际替换才能去重；不同位点组合中的自替换不会再污染候选池。
    return list(dict.fromkeys(mutate_info_list))


def score_mutated_candidates(
    train_dataset,
    collate_fn,
    reward_models,
    mutate_info_list,
    device,
    score_agg,
    batch_size,
):
    """以 chunked batch 对所有候选做 reward 前向，复用 CR 的批处理吞吐。"""
    if not mutate_info_list:
        raise ValueError("不能对空候选池评分")
    model_score_chunks = [[] for _ in reward_models]
    with torch.no_grad():
        for start in range(0, len(mutate_info_list), batch_size):
            chunk = mutate_info_list[start : start + batch_size]
            sample_batches = [train_dataset.getitem_by_mutate_info(item) for item in chunk]
            score_batch = recursive_to(collate_fn(sample_batches), device)
            for model_index, reward_model in enumerate(reward_models):
                scores, _, _, _ = reward_model(score_batch, device)
                model_score_chunks[model_index].append(scores.reshape(-1))
    score_stack = torch.stack(
        [torch.cat(chunks, dim=0) for chunks in model_score_chunks],
        dim=0,
    )
    if score_stack.shape[1] != len(mutate_info_list):
        raise RuntimeError(
            f"奖励分数数量 {score_stack.shape[1]} 与候选数量 {len(mutate_info_list)} 不一致"
        )
    center_scores, std_scores = aggregate_score_stack(score_stack, score_agg)
    return center_scores, std_scores, score_stack


def nominate_and_score_candidates(
    args,
    evo_model,
    reward_models,
    train_dataset,
    collate_fn,
    device,
):
    """生成当前策略候选、构建缺失结构，并返回未变换的能量 acquisition。"""
    mutate_info_list = generate_mutate_info_list(
        args, evo_model, train_dataset, collate_fn, device
    )
    if not mutate_info_list:
        return None
    train_dataset.mutate_pdb(mutate_info_list)
    center_scores, std_scores, score_stack = score_mutated_candidates(
        train_dataset,
        collate_fn,
        reward_models,
        mutate_info_list,
        device,
        args.score_agg,
        args.candidate_score_batch_size,
    )
    acquisition = make_acquisition(
        center_scores,
        std_scores,
        args.uncertainty_weight,
        args.reward_direction,
    )
    return {
        "mutate_info_list": mutate_info_list,
        "center_scores": center_scores,
        "std_scores": std_scores,
        "score_stack": score_stack,
        "acquisition": acquisition,
    }


def candidate_rows_from_pool(candidate_pool, epoch, source):
    """将 tensor 形式的候选评分转成可写入 CSV 的独立记录。"""
    if candidate_pool is None:
        return []
    rows = []
    for index, mutate_info in enumerate(candidate_pool["mutate_info_list"]):
        rows.append(
            {
                "epoch": int(epoch),
                "source": source,
                "mutate_info": mutate_info,
                "center_score": float(candidate_pool["center_scores"][index].item()),
                "std_score": float(candidate_pool["std_scores"][index].item()),
                "acquisition": float(candidate_pool["acquisition"][index].item()),
                "per_model_scores": "|".join(
                    f"{value:.6f}"
                    for value in candidate_pool["score_stack"][:, index].detach().cpu().tolist()
                ),
            }
        )
    return rows


def get_topk_energy_metrics(candidate_pool, top_k, prefix):
    """汇总当前候选池的模型能量指标；不涉及 CR landscape rank。"""
    if candidate_pool is None:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_best_acquisition": np.nan,
            f"{prefix}_mean_acquisition": np.nan,
            f"{prefix}_best_reward_energy": np.nan,
            f"{prefix}_mean_reward_energy": np.nan,
            f"{prefix}_mean_reward_uncertainty": np.nan,
            f"{prefix}_selected_mutations": "",
        }
    mutate_info_list = candidate_pool["mutate_info_list"]
    acquisition = candidate_pool["acquisition"]
    selected_indices = torch.argsort(acquisition.detach())[: min(top_k, len(mutate_info_list))]
    center_scores = candidate_pool["center_scores"][selected_indices]
    std_scores = candidate_pool["std_scores"][selected_indices]
    selected_acquisition = acquisition[selected_indices]
    return {
        f"{prefix}_count": len(selected_indices),
        f"{prefix}_best_acquisition": float(selected_acquisition.min().item()),
        f"{prefix}_mean_acquisition": float(selected_acquisition.mean().item()),
        f"{prefix}_best_reward_energy": float(center_scores.min().item()),
        f"{prefix}_mean_reward_energy": float(center_scores.mean().item()),
        f"{prefix}_mean_reward_uncertainty": float(std_scores.mean().item()),
        f"{prefix}_selected_mutations": "|".join(
            mutate_info_list[index] for index in selected_indices.tolist()
        ),
    }


def save_epoch_metrics(loss_history, pdb_id, metrics):
    save_file = os.path.join(loss_history.save_path, pdb_id + ".csv")
    new_row = pd.DataFrame(metrics, index=[0])
    if os.path.exists(save_file):
        save_df = pd.concat([pd.read_csv(save_file), new_row], ignore_index=True)
    else:
        save_df = new_row
    save_df.to_csv(save_file, index=False)


def train_one_epoch(
    args,
    evo_model,
    reward_models,
    reference_model,
    optimizer,
    epoch,
    train_dataset,
    collate_fn,
    loss_history,
    device,
    tb_writer=None,
):
    """执行一轮 SABDAB GRPO；候选池仅服务于当前 epoch。"""
    loss_history.write(f"\nEpoch {epoch + 1}: 开始 GRPO 训练\n")
    candidate_pool = nominate_and_score_candidates(
        args,
        evo_model,
        reward_models,
        train_dataset,
        collate_fn,
        device,
    )
    pdb_id = train_dataset.pdb_id
    epoch_prefix = f"epoch_top{args.final_top_k}"
    if candidate_pool is None:
        metrics = {
            "epoch": epoch,
            "reward_model_count": len(reward_models),
            "candidate_count": 0,
            "grpo_skipped": True,
            "train_loss": np.nan,
            **get_topk_energy_metrics(None, args.final_top_k, epoch_prefix),
        }
        save_epoch_metrics(loss_history, pdb_id, metrics)
        loss_history.write(f"Epoch {epoch + 1}: 没有非 WT 候选，GRPO 跳过。\n")
        return metrics

    mutate_info_list = candidate_pool["mutate_info_list"]
    acquisition = candidate_pool["acquisition"]
    train_scores = transform_acquisition(
        acquisition,
        args.reward_transform,
        args.reward_clip_mad,
    )

    # old_model 是 GRPO ratio 的冻结行为策略；reference_model 是预训练 KL 锚点。
    old_model = MERF(args).to(device)
    old_model.load_state_dict(evo_model.state_dict())
    old_model.eval()
    for parameter in old_model.parameters():
        parameter.requires_grad = False

    score_array = train_scores.detach().cpu().numpy()
    advantages = score_array.mean() - score_array
    advantage_std = advantages.std()
    if advantage_std > 1e-8:
        advantages = advantages / (advantage_std + 1e-8)

    epoch_losses, policy_losses, kl_values = [], [], []
    n_samples = len(mutate_info_list)
    evo_model.train()
    for inner_epoch in tqdm(
        range(args.inner_epochs),
        desc=f"{pdb_id}: GRPO",
    ):
        sample_indices = np.random.choice(
            n_samples,
            size=min(args.inner_batch_size, n_samples),
            replace=False,
        )
        sample_batches = [
            train_dataset.getitem_by_mutate_info(mutate_info_list[index])
            for index in sample_indices
        ]
        train_batch = recursive_to(collate_fn(sample_batches), device)
        optimizer.zero_grad()

        with torch.no_grad():
            ref_logits = torch.nn.functional.log_softmax(
                -reference_model.choose_best_action(train_batch, device),
                dim=-1,
            )
            old_logits = torch.nn.functional.log_softmax(
                -old_model.choose_best_action(train_batch, device),
                dim=-1,
            )
        curr_logits = torch.nn.functional.log_softmax(
            -evo_model.choose_best_action(train_batch, device),
            dim=-1,
        )

        agent_mask = train_batch["wt"]["agent_mask"]
        ref_logits = ref_logits[agent_mask].view(len(sample_indices), -1, ref_logits.size(-1))
        old_logits = old_logits[agent_mask].view(len(sample_indices), -1, old_logits.size(-1))
        curr_logits = curr_logits[agent_mask].view(len(sample_indices), -1, curr_logits.size(-1))
        ratio = torch.exp(curr_logits - old_logits).clamp(max=20.0)
        advantage_tensor = torch.as_tensor(
            advantages[sample_indices],
            dtype=torch.float32,
            device=device,
        )
        policy_loss = -(ratio * advantage_tensor[:, None, None]).mean()
        kl_divergence = (
            ref_logits.exp() * (ref_logits - curr_logits)
        ).sum(dim=-1).mean()
        batch_loss = policy_loss + args.kl_coeff * kl_divergence
        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(float(batch_loss.item()))
        policy_losses.append(float(policy_loss.item()))
        kl_values.append(float(kl_divergence.item()))
        if tb_writer is not None:
            inner_step = epoch * args.inner_epochs + inner_epoch
            tb_writer.add_scalar("train/policy_loss", policy_loss.item(), inner_step)
            tb_writer.add_scalar("train/kl_div", kl_divergence.item(), inner_step)
            tb_writer.add_scalar("train/batch_loss", batch_loss.item(), inner_step)

    topk_metrics = get_topk_energy_metrics(candidate_pool, args.final_top_k, epoch_prefix)
    score_stack = candidate_pool["score_stack"]
    per_model_min = score_stack.min(dim=1).values.detach().cpu().tolist()
    metrics = {
        "epoch": epoch,
        "reward_label": args.reward_label,
        "reward_model_count": len(reward_models),
        "score_agg": args.score_agg,
        "uncertainty_weight": args.uncertainty_weight,
        "reward_transform": args.reward_transform,
        "reward_direction": args.reward_direction,
        "candidate_count": n_samples,
        "grpo_skipped": False,
        "reward_energy_min": float(candidate_pool["center_scores"].min().item()),
        "reward_energy_mean": float(candidate_pool["center_scores"].mean().item()),
        "reward_uncertainty_mean": float(candidate_pool["std_scores"].mean().item()),
        "acquisition_min": float(acquisition.min().item()),
        "acquisition_mean": float(acquisition.mean().item()),
        "per_model_min_energies": "|".join(f"{value:.6f}" for value in per_model_min),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
        "train_loss": float(np.mean(epoch_losses)),
        "train_policy_loss": float(np.mean(policy_losses)),
        "train_kl": float(np.mean(kl_values)),
        **topk_metrics,
    }
    save_epoch_metrics(loss_history, pdb_id, metrics)

    if tb_writer is not None:
        tb_writer.add_scalar("evo/num_candidates", n_samples, epoch)
        tb_writer.add_scalar("reward_model/reward_energy_min", metrics["reward_energy_min"], epoch)
        tb_writer.add_scalar("reward_model/acquisition_min", metrics["acquisition_min"], epoch)
        tb_writer.add_scalar("train/loss", metrics["train_loss"], epoch)
        tb_writer.add_scalar(
            f"epoch/top{args.final_top_k}_mean_acquisition",
            topk_metrics[f"{epoch_prefix}_mean_acquisition"],
            epoch,
        )

    loss_history.write(
        f"Epoch {epoch + 1}: candidates={n_samples}, "
        f"acquisition_min={metrics['acquisition_min']:.6f}, "
        f"top{args.final_top_k}_mean_acquisition="
        f"{topk_metrics[f'{epoch_prefix}_mean_acquisition']:.6f}, "
        f"loss={metrics['train_loss']:.6f}\n"
    )
    print(
        f"Epoch:{epoch + 1}/{args.total_epochs} Candidates:{n_samples} "
        f"Acquisition:{metrics['acquisition_min']:.6f} "
        f"Top{args.final_top_k}Mean:{topk_metrics[f'{epoch_prefix}_mean_acquisition']:.6f} "
        f"Loss:{metrics['train_loss']:.6f}",
        flush=True,
    )
    return metrics


def select_final_top_candidates(candidate_rows, top_k):
    """只从最后更新后的策略候选池中选择 top-k，禁止读取历史 epoch。"""
    return sorted(candidate_rows, key=lambda item: item["acquisition"])[:top_k]


def _safe_name(text):
    return text.replace(",", "_").replace("/", "_").replace("\\", "_")


def write_final_nominations(loss_history, pdb_id, final_candidates, requested_top_k):
    """保存唯一允许进入外部评估的最后一轮候选清单。"""
    rows = []
    for rank, candidate in enumerate(final_candidates, start=1):
        rows.append(
            {
                "rank_by_final_epoch_acquisition": rank,
                "requested_top_k": requested_top_k,
                **candidate,
            }
        )
    output_path = os.path.join(loss_history.save_path, f"{pdb_id}_final_epoch_top{requested_top_k}.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    return pd.DataFrame(rows), output_path


def evaluate_final_candidates_with_pyrosetta(
    pdb_id,
    interface_str,
    wt_pdb,
    mut_dir,
    final_candidates,
    output_dir,
    loss_history,
    tb_writer=None,
    evaluation_label="final_epoch",
    tb_tag_prefix="pyrosetta_final",
    tb_step=0,
):
    """计算一个独立策略快照的 Rosetta ddG；默认保留最终评测的原有命名。"""
    global _PYROSETTA_INITIALIZED

    if not final_candidates:
        return pd.DataFrame(), {
            "pyrosetta_candidate_count": 0,
            "pyrosetta_success_count": 0,
            "rosetta_wt_interface_energy": np.nan,
            "rosetta_min_ddg_relative_to_wt": np.nan,
            "rosetta_mean_ddg_relative_to_wt": np.nan,
            "rosetta_top1_ddg_relative_to_wt": np.nan,
            "affinity_improved_by_rosetta": np.nan,
        }

    from scripts.calculate_ddG_pyrosetta import (
        calculate_interface_energy,
        init_pyrosetta,
        relax_structure,
    )

    if not _PYROSETTA_INITIALIZED:
        init_pyrosetta()
        _PYROSETTA_INITIALIZED = True

    pyrosetta_dir = os.path.join(output_dir, f"pyrosetta_{evaluation_label}_v3")
    os.makedirs(pyrosetta_dir, exist_ok=True)
    wt_relaxed_pdb = os.path.join(pyrosetta_dir, "wt_relaxed.pdb")
    try:
        relax_structure(wt_pdb, wt_relaxed_pdb)
        wt_energy_pdb = wt_relaxed_pdb
        wt_relaxed = True
    except Exception as exc:
        loss_history.write(f"PyRosetta WT relax 失败，退回 fixed WT: {exc}\n")
        wt_energy_pdb = wt_pdb
        wt_relaxed = False
    dG_wt = calculate_interface_energy(wt_energy_pdb, interface_str)

    rows = []
    rank_column = (
        "rank_by_final_epoch_acquisition"
        if evaluation_label == "final_epoch"
        else "rank_by_current_policy_acquisition"
    )
    for rank, candidate in enumerate(final_candidates, start=1):
        mutate_info = candidate["mutate_info"]
        mut_pdb = mutant_pdb_path(mut_dir, pdb_id, mutate_info)
        candidate_dir = os.path.join(
            pyrosetta_dir,
            f"rank_{rank:03d}_{_safe_name(mutate_info)}",
        )
        os.makedirs(candidate_dir, exist_ok=True)
        mut_relaxed_pdb = os.path.join(candidate_dir, "mut_relaxed.pdb")
        row = {
            "mutate_info": mutate_info,
            rank_column: rank,
            "acquisition": candidate["acquisition"],
            "reward_energy": candidate["center_score"],
            "reward_uncertainty": candidate["std_score"],
            "wt_pdb": os.path.abspath(wt_pdb),
            "mut_pdb": os.path.abspath(mut_pdb),
            "interface": interface_str,
            "wt_relaxed": wt_relaxed,
            "mut_relaxed": False,
            "dG_wt": float(dG_wt),
            "dG_mut": np.nan,
            "ddG": np.nan,
            "pyrosetta_result_dir": candidate_dir,
            "pyrosetta_error": "",
        }
        try:
            loss_history.write(
                f"PyRosetta {evaluation_label} rank {rank}: {mutate_info}, "
                f"acquisition={candidate['acquisition']:.6f}\n"
            )
            relax_structure(mut_pdb, mut_relaxed_pdb)
            row["mut_relaxed"] = True
            dG_mut = calculate_interface_energy(mut_relaxed_pdb, interface_str)
            row["dG_mut"] = float(dG_mut)
            row["ddG"] = float(dG_mut - dG_wt)
        except Exception as exc:
            row["pyrosetta_error"] = str(exc)
            loss_history.write(
                f"PyRosetta {evaluation_label} rank {rank} 失败: {mutate_info}; {exc}\n"
            )
        rows.append(row)

    result_df = pd.DataFrame(rows)
    result_path = os.path.join(
        pyrosetta_dir,
        f"{pdb_id}_{evaluation_label}_top{len(final_candidates)}_pyrosetta_ddg.csv",
    )
    result_df.to_csv(result_path, index=False)
    valid_ddg = result_df["ddG"].dropna()
    best_ddg = float(valid_ddg.min()) if len(valid_ddg) else np.nan
    mean_ddg = float(valid_ddg.mean()) if len(valid_ddg) else np.nan
    top1_ddg = result_df.iloc[0]["ddG"] if len(result_df) else np.nan
    top1_ddg = float(top1_ddg) if pd.notna(top1_ddg) else np.nan
    summary = {
        "pyrosetta_candidate_count": len(final_candidates),
        "pyrosetta_success_count": len(valid_ddg),
        "pyrosetta_result_csv": result_path,
        "rosetta_wt_interface_energy": float(dG_wt),
        "rosetta_min_ddg_relative_to_wt": best_ddg,
        "rosetta_mean_ddg_relative_to_wt": mean_ddg,
        "rosetta_top1_ddg_relative_to_wt": top1_ddg,
        "affinity_improved_by_rosetta": bool(best_ddg < 0) if not np.isnan(best_ddg) else np.nan,
    }
    loss_history.write(
        f"PyRosetta {evaluation_label} top-{len(final_candidates)}: min_ddG={best_ddg}, "
        f"affinity_improved={summary['affinity_improved_by_rosetta']}\n"
    )
    if tb_writer is not None:
        tb_writer.add_scalar(f"{tb_tag_prefix}/num_evaluated", len(final_candidates), tb_step)
        tb_writer.add_scalar(f"{tb_tag_prefix}/num_success", len(valid_ddg), tb_step)
        if len(valid_ddg):
            tb_writer.add_scalar(f"{tb_tag_prefix}/wt_interface_energy", dG_wt, tb_step)
            tb_writer.add_scalar(f"{tb_tag_prefix}/min_ddG", best_ddg, tb_step)
            tb_writer.add_scalar(f"{tb_tag_prefix}/mean_ddG", mean_ddg, tb_step)
            if not np.isnan(top1_ddg):
                tb_writer.add_scalar(f"{tb_tag_prefix}/top1_ddG", top1_ddg, tb_step)
    return result_df, summary


def _flatten_colabdesign_metrics(metrics):
    """把 AF2 顶层数值指标写入候选 CSV，同时保留原始 JSON 路径。"""
    flat = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
            flat[f"af2_{key}"] = float(value)
    return flat


def _run_colabdesign_once(args, pdb_path, partner, antibody_chain, output_dir):
    """以用户指定的 colabdesign-af2 解释器运行独立 v3 评估辅助脚本。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_json = output_dir / "metrics.json"
    prediction_pdb = output_dir / "prediction.pdb"
    log_path = output_dir / "colabdesign.log"
    command = [
        args.colabdesign_python,
        args.colabdesign_script,
        "--pdb",
        str(pdb_path),
        "--partner",
        partner,
        "--primary-binder-chain",
        antibody_chain,
        "--af2-params",
        args.colabdesign_af2_params,
        "--output-dir",
        str(output_dir),
        "--metrics-json",
        str(metrics_json),
        "--output-pdb",
        str(prediction_pdb),
        "--model-index",
        str(args.colabdesign_model_index),
        "--recycles",
        str(args.colabdesign_recycles),
    ]
    environment = os.environ.copy()
    if args.colabdesign_cuda_visible_devices:
        environment["CUDA_VISIBLE_DEVICES"] = args.colabdesign_cuda_visible_devices
    try:
        with open(log_path, "w", encoding="utf-8") as log_handle:
            subprocess.run(
                command,
                cwd=str(PROJECT_DIR),
                env=environment,
                text=True,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=True,
            )
        with open(metrics_json, encoding="utf-8") as handle:
            metrics = json.load(handle)
        return {
            "status": "ok",
            "metrics": metrics,
            "metrics_json": str(metrics_json),
            "prediction_pdb": str(prediction_pdb),
            "log_path": str(log_path),
            "error": "",
        }
    except Exception as exc:
        return {
            "status": "error",
            "metrics": {},
            "metrics_json": str(metrics_json),
            "prediction_pdb": str(prediction_pdb),
            "log_path": str(log_path),
            "error": str(exc),
        }


def evaluate_final_candidates_with_colabdesign(
    args,
    pdb_id,
    partner,
    antibody_chain,
    wt_pdb,
    mut_dir,
    final_candidates,
    output_dir,
    loss_history,
    evaluation_label="final_epoch",
):
    """评估一个独立策略快照的 AF2-Multimer 指标；默认保留最终评测命名。"""
    if not final_candidates:
        return pd.DataFrame(), {}

    colabdesign_dir = Path(output_dir) / f"colabdesign_{evaluation_label}_v3"
    wt_metrics = {}
    if args.colabdesign_include_wt:
        wt_result = _run_colabdesign_once(
            args,
            wt_pdb,
            partner,
            antibody_chain,
            colabdesign_dir / "wt",
        )
        if wt_result["status"] == "ok":
            wt_metrics = _flatten_colabdesign_metrics(wt_result["metrics"])
            loss_history.write(f"ColabDesign WT 完成: {wt_result['metrics_json']}\n")
        else:
            loss_history.write(f"ColabDesign WT 失败: {wt_result['error']}\n")

    rows = []
    rank_column = (
        "rank_by_final_epoch_acquisition"
        if evaluation_label == "final_epoch"
        else "rank_by_current_policy_acquisition"
    )
    for rank, candidate in enumerate(final_candidates, start=1):
        mutate_info = candidate["mutate_info"]
        candidate_result = _run_colabdesign_once(
            args,
            mutant_pdb_path(mut_dir, pdb_id, mutate_info),
            partner,
            antibody_chain,
            colabdesign_dir / f"rank_{rank:03d}_{_safe_name(mutate_info)}",
        )
        row = {
            "mutate_info": mutate_info,
            rank_column: rank,
            "colabdesign_status": candidate_result["status"],
            "colabdesign_metrics_json": candidate_result["metrics_json"],
            "colabdesign_prediction_pdb": candidate_result["prediction_pdb"],
            "colabdesign_log": candidate_result["log_path"],
            "colabdesign_error": candidate_result["error"],
        }
        if candidate_result["status"] == "ok":
            flattened = _flatten_colabdesign_metrics(candidate_result["metrics"])
            row.update(flattened)
            for metric_name, wt_value in wt_metrics.items():
                if metric_name in flattened:
                    row[f"delta_{metric_name}_vs_wt"] = flattened[metric_name] - wt_value
            loss_history.write(
                f"ColabDesign {evaluation_label} rank {rank} 完成: {mutate_info}; "
                f"{candidate_result['metrics_json']}\n"
            )
        else:
            loss_history.write(
                f"ColabDesign {evaluation_label} rank {rank} 失败: {mutate_info}; "
                f"{candidate_result['error']}\n"
            )
        rows.append(row)
    return pd.DataFrame(rows), wt_metrics


def _write_tensorboard_scalars(tb_writer, tag_prefix, metrics, step):
    """只写入有限数值，避免字符串路径和 NaN 污染 TensorBoard。"""
    if tb_writer is None:
        return
    for name, value in metrics.items():
        if isinstance(value, (bool, np.bool_)):
            continue
        try:
            scalar = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(scalar):
            tb_writer.add_scalar(f"{tag_prefix}/{name}", scalar, step)


def summarize_colabdesign_snapshot(colabdesign_df, wt_metrics):
    """汇总当前策略 top-8 的 AF2 指标：reward top-1 与成功候选均值。"""
    summary = {
        "colabdesign_candidate_count": int(len(colabdesign_df)),
        "colabdesign_success_count": 0,
        "colabdesign_top1_success": 0.0,
    }
    if colabdesign_df.empty:
        return summary

    successful = colabdesign_df[colabdesign_df["colabdesign_status"] == "ok"]
    summary["colabdesign_success_count"] = int(len(successful))
    top1 = colabdesign_df.iloc[0]
    if top1["colabdesign_status"] == "ok":
        summary["colabdesign_top1_success"] = 1.0

    for metric_name in MONITORED_COLABDESIGN_METRICS:
        if metric_name in wt_metrics:
            summary[f"colabdesign_wt_{metric_name}"] = float(wt_metrics[metric_name])
        for column in (metric_name, f"delta_{metric_name}_vs_wt"):
            if column not in successful.columns:
                continue
            values = pd.to_numeric(successful[column], errors="coerce").dropna()
            if len(values):
                summary[f"colabdesign_mean_{column}"] = float(values.mean())
            if summary["colabdesign_top1_success"] and pd.notna(top1.get(column)):
                summary[f"colabdesign_top1_{column}"] = float(top1[column])
    return summary


def write_periodic_external_metrics(loss_history, pdb_id, metrics):
    """按 epoch 追加监测摘要；它与训练损失 CSV、最终评测 CSV 相互独立。"""
    output_path = os.path.join(loss_history.save_path, f"{pdb_id}_periodic_external_metrics_v3.csv")
    new_row = pd.DataFrame([metrics])
    if os.path.exists(output_path):
        history = pd.read_csv(output_path)
        new_row = pd.concat([history, new_row], ignore_index=True)
    new_row.to_csv(output_path, index=False)
    return output_path


def monitor_current_policy_external_metrics(
    args,
    evo_model,
    reward_models,
    train_dataset,
    collate_fn,
    device,
    epoch,
    row,
    loss_history,
    tb_writer=None,
):
    """监测已更新策略的 top-8，不向最终评测候选池回写任何内容。"""
    pdb_id = train_dataset.pdb_id
    evaluation_label = f"epoch_{epoch + 1:04d}"
    monitor_dir = Path(loss_history.save_path) / "periodic_external_eval_v3" / evaluation_label
    monitor_dir.mkdir(parents=True, exist_ok=True)

    # 这里重新提名的是本 epoch 更新后的策略；结果只作为监测快照，不会被最终评测读取。
    current_pool = nominate_and_score_candidates(
        args,
        evo_model,
        reward_models,
        train_dataset,
        collate_fn,
        device,
    )
    policy_metrics = get_topk_energy_metrics(
        current_pool,
        args.final_top_k,
        prefix=f"periodic_policy_top{args.final_top_k}",
    )
    current_rows = candidate_rows_from_pool(
        current_pool,
        epoch,
        source="periodic_policy_after_epoch_update",
    )
    current_candidates = select_final_top_candidates(current_rows, args.final_top_k)
    snapshot_rows = [
        {
            "rank_by_current_policy_acquisition": rank,
            "requested_top_k": args.final_top_k,
            **candidate,
        }
        for rank, candidate in enumerate(current_candidates, start=1)
    ]
    snapshot_path = monitor_dir / f"{pdb_id}_{evaluation_label}_top{args.final_top_k}_monitor.csv"
    pd.DataFrame(snapshot_rows).to_csv(snapshot_path, index=False)

    summary = {
        "epoch": epoch,
        "epoch_number": epoch + 1,
        "evaluation_label": evaluation_label,
        "requested_top_k": args.final_top_k,
        "current_policy_candidate_count": len(current_candidates),
        "current_policy_snapshot_csv": str(snapshot_path),
        "monitoring_only": True,
        **policy_metrics,
    }
    _write_tensorboard_scalars(tb_writer, "periodic_external/policy", policy_metrics, epoch)

    if not current_candidates:
        summary["monitoring_note"] = "当前策略没有非 WT 候选，未运行外部评估"
        metrics_path = write_periodic_external_metrics(loss_history, pdb_id, summary)
        loss_history.write(
            f"周期外部评估 Epoch {epoch + 1}: 没有非 WT 候选；仅记录空监测快照 {metrics_path}\n"
        )
        return summary

    if args.run_pyrosetta:
        try:
            _, pyrosetta_summary = evaluate_final_candidates_with_pyrosetta(
                pdb_id=pdb_id,
                interface_str=row["partner"],
                wt_pdb=train_dataset.fixed_wt_pdb_path,
                mut_dir=args.mut_dir,
                final_candidates=current_candidates,
                output_dir=monitor_dir,
                loss_history=loss_history,
                tb_writer=tb_writer,
                evaluation_label=evaluation_label,
                tb_tag_prefix="periodic_external/pyrosetta",
                tb_step=epoch,
            )
            summary.update(pyrosetta_summary)
        except Exception as exc:
            summary["pyrosetta_monitor_error"] = str(exc)
            loss_history.write(f"周期 PyRosetta {evaluation_label} 失败，不影响训练: {exc}\n")
            if tb_writer is not None:
                tb_writer.add_scalar("periodic_external/pyrosetta/error", 1.0, epoch)

    if args.run_colabdesign:
        try:
            colabdesign_df, wt_metrics = evaluate_final_candidates_with_colabdesign(
                args=args,
                pdb_id=pdb_id,
                partner=row["partner"],
                antibody_chain=row["antibody_chain"],
                wt_pdb=train_dataset.fixed_wt_pdb_path,
                mut_dir=args.mut_dir,
                final_candidates=current_candidates,
                output_dir=monitor_dir,
                loss_history=loss_history,
                evaluation_label=evaluation_label,
            )
            colabdesign_result_path = (
                monitor_dir / f"{pdb_id}_{evaluation_label}_top{len(current_candidates)}_colabdesign.csv"
            )
            colabdesign_df.to_csv(colabdesign_result_path, index=False)
            colabdesign_summary = summarize_colabdesign_snapshot(colabdesign_df, wt_metrics)
            summary.update(colabdesign_summary)
            summary["colabdesign_result_csv"] = str(colabdesign_result_path)
            _write_tensorboard_scalars(
                tb_writer,
                "periodic_external/colabdesign",
                colabdesign_summary,
                epoch,
            )
        except Exception as exc:
            summary["colabdesign_monitor_error"] = str(exc)
            loss_history.write(f"周期 ColabDesign {evaluation_label} 失败，不影响训练: {exc}\n")
            if tb_writer is not None:
                tb_writer.add_scalar("periodic_external/colabdesign/error", 1.0, epoch)

    metrics_path = write_periodic_external_metrics(loss_history, pdb_id, summary)
    loss_history.write(
        f"周期外部评估 Epoch {epoch + 1}: 当前策略 top-{len(current_candidates)} 已完成；"
        f"仅用于调参监测，不进入最终候选池。摘要: {metrics_path}\n"
    )
    return summary


def build_dataset(args, device, row):
    pdb_id = row["pdb_id"].replace("+", "").replace(".00", "")
    return SAbDabEvoDatasetV3(
        pdb_id=pdb_id,
        antibody_chain=row["antibody_chain"],
        partner=row["partner"],
        sequence=row["sequence"],
        cdr=row["cdr3"],
        wt_dir=args.wt_dir,
        fixed_dir=args.fixed_dir,
        mut_dir=args.mut_dir,
        knn_num=args.knn_neighbors_num,
        knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding,
        plm_path=args.plm_path,
        plm_embedding_path=args.plm_embedding_path,
        device=device,
        foldx_workers=args.foldx_workers,
        foldx_bin=args.foldx_bin,
    )


def merge_final_evaluations(final_df, pyrosetta_df, colabdesign_df):
    """将外部评测结果合并回最后一轮 top-k 表，不引入任何历史候选。"""
    merged = final_df.copy()
    if not pyrosetta_df.empty:
        pyrosetta_columns = [
            "mutate_info",
            "dG_wt",
            "dG_mut",
            "ddG",
            "wt_relaxed",
            "mut_relaxed",
            "pyrosetta_result_dir",
            "pyrosetta_error",
        ]
        merged = merged.merge(
            pyrosetta_df[pyrosetta_columns],
            on="mutate_info",
            how="left",
        )
    if not colabdesign_df.empty:
        colabdesign_columns = [
            column
            for column in colabdesign_df.columns
            if column not in {"rank_by_final_epoch_acquisition"}
        ]
        merged = merged.merge(
            colabdesign_df[colabdesign_columns],
            on="mutate_info",
            how="left",
        )
    return merged


def run_one_dataset(
    args,
    row,
    dataset_index,
    run_dir,
    device,
    checkpoint_args_path,
    checkpoint_model_arg_keys,
    reward_paths,
):
    """完成一个 SABDAB 条目的训练和最终仅 top-8 的外部评测。"""
    train_dataset = build_dataset(args, device, row)
    pdb_id = train_dataset.pdb_id
    loss_history = LossHistory(os.path.join(run_dir, pdb_id), is_evolve=True)
    tb_writer = SummaryWriter(log_dir=loss_history.save_path)
    collate_fn = PaddingCollate()
    loss_history.write(str(args) + "\n")
    loss_history.write(f"dataset_index={dataset_index}; pdb_id={pdb_id}\n")
    loss_history.write(f"Reward paths: {args.reward_model_paths}\n")
    loss_history.write(f"Fixed WT: {train_dataset.fixed_wt_pdb_path}\n")
    loss_history.write(
        "v3 外部评测只使用最后一次训练更新后策略重新提名的 top-k；"
        "不累积任何历史 epoch 候选。\n"
    )
    loss_history.write(
        f"周期外部监测间隔: 每 {args.periodic_external_eval_interval} 个 epoch；"
        "监测快照不参与最终候选筛选。\n"
    )
    if checkpoint_args_path is not None:
        loss_history.write(f"Loaded model args from {checkpoint_args_path}\n")
        loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')

    try:
        evo_model = MERF(args).to(device)
        load_pretrained(evo_model, args.model_load_path, device)
        optimizer = optim.Adam(
            evo_model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        reference_model = MERF(args).to(device)
        reference_model.load_state_dict(evo_model.state_dict())
        reference_model.eval()
        for parameter in reference_model.parameters():
            parameter.requires_grad = False

        reward_models = []
        for reward_path in reward_paths:
            reward_model = MERF(args).to(device)
            load_pretrained(reward_model, reward_path, device)
            reward_model.eval()
            for parameter in reward_model.parameters():
                parameter.requires_grad = False
            reward_models.append(reward_model)
        if not reward_models:
            raise ValueError("No reward models loaded")

        for epoch in range(args.total_epochs):
            train_one_epoch(
                args,
                evo_model,
                reward_models,
                reference_model,
                optimizer,
                epoch,
                train_dataset,
                collate_fn,
                loss_history,
                device,
                tb_writer=tb_writer,
            )
            if (epoch + 1) % args.periodic_external_eval_interval == 0:
                try:
                    monitor_current_policy_external_metrics(
                        args=args,
                        evo_model=evo_model,
                        reward_models=reward_models,
                        train_dataset=train_dataset,
                        collate_fn=collate_fn,
                        device=device,
                        epoch=epoch,
                        row=row,
                        loss_history=loss_history,
                        tb_writer=tb_writer,
                    )
                except Exception as exc:
                    loss_history.write(
                        f"周期外部评估 Epoch {epoch + 1} 启动失败，不影响后续训练或最终评测: {exc}\n"
                    )
                    tb_writer.add_scalar("periodic_external/error", 1.0, epoch)

        # 关键：训练结束后重新用最终策略提名。这里没有读取或合并历史 epoch 候选。
        final_pool = nominate_and_score_candidates(
            args,
            evo_model,
            reward_models,
            train_dataset,
            collate_fn,
            device,
        )
        final_epoch = args.total_epochs - 1
        final_rows = candidate_rows_from_pool(
            final_pool,
            final_epoch,
            source="final_policy_after_last_update",
        )
        final_candidates = select_final_top_candidates(final_rows, args.final_top_k)
        final_df, nomination_path = write_final_nominations(
            loss_history,
            pdb_id,
            final_candidates,
            args.final_top_k,
        )
        final_metrics = get_topk_energy_metrics(
            final_pool,
            args.final_top_k,
            prefix=f"final_epoch_top{args.final_top_k}",
        )
        loss_history.write(
            f"最终策略提名 {len(final_candidates)}/{args.final_top_k} 个候选: {nomination_path}\n"
        )

        pyrosetta_df = pd.DataFrame()
        pyrosetta_summary = {
            "pyrosetta_candidate_count": 0,
            "pyrosetta_success_count": 0,
            "rosetta_wt_interface_energy": np.nan,
            "rosetta_min_ddg_relative_to_wt": np.nan,
            "rosetta_mean_ddg_relative_to_wt": np.nan,
            "rosetta_top1_ddg_relative_to_wt": np.nan,
            "affinity_improved_by_rosetta": np.nan,
        }
        if args.run_pyrosetta:
            pyrosetta_df, pyrosetta_summary = evaluate_final_candidates_with_pyrosetta(
                pdb_id=pdb_id,
                interface_str=row["partner"],
                wt_pdb=train_dataset.fixed_wt_pdb_path,
                mut_dir=args.mut_dir,
                final_candidates=final_candidates,
                output_dir=loss_history.save_path,
                loss_history=loss_history,
                tb_writer=tb_writer,
            )

        colabdesign_df = pd.DataFrame()
        wt_colabdesign_metrics = {}
        if args.run_colabdesign:
            colabdesign_df, wt_colabdesign_metrics = evaluate_final_candidates_with_colabdesign(
                args=args,
                pdb_id=pdb_id,
                partner=row["partner"],
                antibody_chain=row["antibody_chain"],
                wt_pdb=train_dataset.fixed_wt_pdb_path,
                mut_dir=args.mut_dir,
                final_candidates=final_candidates,
                output_dir=loss_history.save_path,
                loss_history=loss_history,
            )

        evaluated_df = merge_final_evaluations(final_df, pyrosetta_df, colabdesign_df)
        evaluated_path = os.path.join(
            loss_history.save_path,
            f"{pdb_id}_final_epoch_top{args.final_top_k}_evaluated.csv",
        )
        evaluated_df.to_csv(evaluated_path, index=False)
        final_summary = {
            "pdb_id": pdb_id,
            "dataset_index": dataset_index,
            "final_epoch": final_epoch,
            "final_top_k_requested": args.final_top_k,
            "final_candidate_count": len(final_candidates),
            "final_nomination_csv": nomination_path,
            "final_evaluated_csv": evaluated_path,
            "fixed_wt_pdb": train_dataset.fixed_wt_pdb_path,
            **final_metrics,
            **pyrosetta_summary,
            **{f"wt_{key}": value for key, value in wt_colabdesign_metrics.items()},
        }
        summary_path = os.path.join(loss_history.save_path, f"{pdb_id}_final_summary.csv")
        pd.DataFrame([final_summary]).to_csv(summary_path, index=False)
        loss_history.write(f"Final summary: {final_summary}\n")
        return final_summary
    finally:
        tb_writer.flush()
        tb_writer.close()


def main():
    args = get_args()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)
    seed_all(args.seed)
    if args.is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")

    train_df = pd.read_csv(args.train_csv, dtype={"pdb_id": "string"})
    if args.dataset_idx is not None:
        if args.dataset_idx < 0 or args.dataset_idx >= len(train_df):
            raise IndexError(
                f"dataset_idx {args.dataset_idx} is out of range for {args.train_csv} "
                f"with {len(train_df)} rows"
            )
        dataset_indices = [args.dataset_idx]
    else:
        dataset_indices = range(len(train_df))

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join(args.log_root, f"{time_str}_{args.run_name}")
    reward_paths = parse_reward_model_paths(args.reward_model_paths)
    for dataset_index in dataset_indices:
        run_one_dataset(
            args=args,
            row=train_df.iloc[dataset_index],
            dataset_index=dataset_index,
            run_dir=run_dir,
            device=device,
            checkpoint_args_path=checkpoint_args_path,
            checkpoint_model_arg_keys=checkpoint_model_arg_keys,
            reward_paths=reward_paths,
        )


if __name__ == "__main__":
    main()
