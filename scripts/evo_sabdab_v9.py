#!/home/dataset-local/anaconda3/envs/merf/bin/python
"""SABDAB v9 固定 20 epoch 进化脚本。

v9 保留 SABDAB 的 GRPO、RepairPDB WT、FoldX 突变结构、批处理 reward 评分及
最终仅评测最终策略 top-8 的规则。在 v3 数据/评测框架内，它进一步实现：

1. 每个三位点组合训练期独立采样 3 个完整非 WT 动作，评测期逐位点 greedy argmax；
2. GRPO ratio 只 gather 三个真实突变 action 的联合 log-probability；
3. coverage-balanced inner update：每个候选每个 pass 精确覆盖一次；
4. deterministic greedy top-8 acquisition 监测、真实历史最佳权重恢复和有效早停；
5. 默认固定训练 20 个 epoch，并在第 20 个 epoch 恢复前 20 轮内的最佳策略。

本文件是独立入口：不动态加载、不 monkey-patch evo_sabdab_v3.py。
"""

from __future__ import annotations

import argparse
import datetime
import json
import math
import os
import pickle
import subprocess
import sys
import time
from copy import deepcopy
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
    parse_mutation_info,
)
from protein.read_pdbs import PaddingCollate
from utils.losshistory import LossHistory
from utils.util import recursive_to, seed_all


PROJECT_DIR = Path(__file__).resolve().parents[1]
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
V9_POLICY_UPDATE_MODE = "v9_fixed20_coverage_balanced_action_gather_restore_best"


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
    parser = argparse.ArgumentParser(description="SABDAB v9 fixed-20-epoch GRPO evolution")

    # 运行和优化器：与 v3 / CR v4 的全模型微调设置对齐。
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--is_cuda", type=str2bool, default=True)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="v9")
    parser.add_argument("--log_root", type=str, default=str(PROJECT_DIR / "logs/evo_sabdab_v9"))

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

    # SABDAB 数据和 v9 独立结构缓存。
    parser.add_argument("--train_csv", type=str, default=str(PROJECT_DIR / "data/sabdab/sabdab_evo.csv"))
    parser.add_argument("--dataset_idx", type=int, default=0)
    parser.add_argument("--wt_dir", type=str, default=str(PROJECT_DIR / "data/sabdab/PDBs"))
    parser.add_argument("--fixed_dir", type=str, default=str(PROJECT_DIR / "data/sabdab/PDBs_fixed_v9"))
    parser.add_argument("--mut_dir", type=str, default=str(PROJECT_DIR / "data/sabdab/PDBs_evo_v9"))
    parser.add_argument(
        "--plm_embedding_path",
        type=str,
        default=str(PROJECT_DIR / "data/sabdab/PLM_embeddings_sabdab_v9.pkl"),
    )
    parser.add_argument("--foldx_workers", type=int, default=8)
    parser.add_argument("--foldx_bin", type=str, default=DEFAULT_FOLDX_BIN)

    # v9 GRPO：固定 20 epoch，使用三位点完整动作和 coverage-balanced inner pass。
    parser.add_argument("--comb_num", type=int, default=3)
    parser.add_argument("--total_epochs", type=int, default=20)
    parser.add_argument(
        "--inner_passes",
        "--inner-passes",
        "--inner_epochs",
        "--inner-epochs",
        dest="inner_passes",
        type=int,
        default=2,
        help="每个 epoch 对完整候选池进行多少个 coverage-balanced pass",
    )
    parser.add_argument("--inner_batch_size", type=int, default=32)
    parser.add_argument("--kl_coeff", type=float, default=0.5)
    parser.add_argument(
        "--actions_per_triple",
        "--actions-per-triple",
        dest="actions_per_triple",
        type=int,
        default=3,
        help="训练期每个三位点组合独立采样的完整非 WT 动作数",
    )
    parser.add_argument(
        "--enable_early_stop",
        "--enable-early-stop",
        dest="enable_early_stop",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--early_stop_min_epochs",
        "--early-stop-min-epochs",
        dest="early_stop_min_epochs",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--early_stop_patience",
        "--early-stop-patience",
        dest="early_stop_patience",
        type=int,
        default=12,
    )
    parser.add_argument(
        "--early_stop_min_delta",
        "--early-stop-min-delta",
        dest="early_stop_min_delta",
        type=float,
        default=0.03,
    )

    # reward model 与候选评分设置。
    parser.add_argument("--reward_model_paths", type=str, default="")
    parser.add_argument("--reward_label", type=str, default="reward")
    parser.add_argument("--uncertainty_weight", type=float, default=0.5)
    parser.add_argument("--score_agg", choices=["mean", "median"], default="mean")
    parser.add_argument("--reward_transform", choices=["raw", "rank", "clip"], default="raw")
    parser.add_argument("--reward_clip_mad", type=float, default=3.0)
    parser.add_argument("--reward_direction", choices=["min", "max"], default="min")
    parser.add_argument("--candidate_score_batch_size", type=int, default=64)

    # 最终仅评估最终策略的 top-8；周期快照仅用于调参监测。
    parser.add_argument("--final_top_k", type=int, choices=[8], default=8)
    parser.add_argument("--run_pyrosetta", type=str2bool, default=True)
    parser.add_argument("--run_colabdesign", type=str2bool, default=True)
    parser.add_argument("--colabdesign_include_wt", type=str2bool, default=True)
    parser.add_argument("--colabdesign_python", type=str, default=DEFAULT_COLABDESIGN_PYTHON)
    parser.add_argument("--colabdesign_script", type=str, default=str(DEFAULT_COLABDESIGN_SCRIPT))
    parser.add_argument("--colabdesign_af2_params", type=str, default=str(DEFAULT_AF2_PARAMS))
    parser.add_argument("--colabdesign_model_index", type=int, default=0)
    parser.add_argument("--colabdesign_recycles", type=int, default=3)
    parser.add_argument("--colabdesign_cuda_visible_devices", type=str, default="0")
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
        "inner_passes",
        "inner_batch_size",
        "actions_per_triple",
        "candidate_score_batch_size",
        "final_top_k",
        "foldx_workers",
        "colabdesign_recycles",
        "periodic_external_eval_interval",
        "early_stop_min_epochs",
        "early_stop_patience",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} 必须为正数")
    if args.comb_num != 3:
        raise ValueError("SABDAB v9 仅支持 --comb_num=3")
    if args.early_stop_min_delta < 0:
        raise ValueError("--early_stop_min_delta 不能为负数")
    if args.lr <= 0:
        raise ValueError("--lr 必须为正数")
    if args.weight_decay < 0:
        raise ValueError("--weight_decay 不能为负数")
    # 保留旧实验启动命令中的字段名兼容性；v9 实际语义是完整覆盖 pass。
    args.inner_epochs = args.inner_passes
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
    """生成 v8 三位点完整动作候选。

    训练阶段每组三位点独立采样 ``actions_per_triple`` 个完整非 WT 动作；
    deterministic greedy 监测和最终提名则对每个位置取排除 WT 后的 argmax。
    """
    policy_batch = recursive_to(collate_fn([train_dataset[0]]), device)
    pdb_id = policy_batch["expand_data_info"]["pdb_id"][0]
    antibody_chain = policy_batch["expand_data_info"]["antibody_chain"][0]
    cdr_mask = policy_batch["mutation_mask"][0]
    true_indices = torch.nonzero(cdr_mask, as_tuple=True)[0]
    if args.comb_num > len(true_indices):
        raise ValueError(
            f"comb_num={args.comb_num} exceeds the {len(true_indices)} mutable CDR residues for {pdb_id}"
        )

    training_sampling = bool(getattr(args, "_v8_sampling_training_phase", False))
    samples_per_triple = args.actions_per_triple if training_sampling else 1
    phase_name = (
        f"train_independent_sample_{args.actions_per_triple}x"
        if training_sampling
        else "deterministic_greedy_argmax"
    )
    print(f"v8_proposal phase={phase_name} pdb={pdb_id}", flush=True)

    mutate_info_list = []
    evo_model.eval()
    with torch.no_grad():
        for mutate_indices in tqdm(
            combinations(true_indices, args.comb_num),
            desc=f"{pdb_id}: 生成 v8 CDR 候选 ({phase_name})",
        ):
            mutation_mask = torch.zeros_like(cdr_mask, dtype=torch.bool)
            mutation_mask[torch.stack(mutate_indices)] = True
            policy_batch["mutation_mask"] = mutation_mask.unsqueeze(0)
            qs = evo_model.choose_best_action(policy_batch, device)
            states = policy_batch["wt"]["aa"][0][mutation_mask]
            site_logits = -qs[policy_batch["mutation_mask"]].clone()
            row_index = torch.arange(site_logits.shape[0], device=site_logits.device)
            # 显式排除 native action，确保每条候选都是三个真实替换。
            site_logits[row_index, states.to(site_logits.device)] = torch.finfo(
                site_logits.dtype
            ).min
            if training_sampling:
                action_rows = sample_distinct_action_rows(
                    torch.softmax(site_logits, dim=1), samples_per_triple
                )
            else:
                action_rows = [torch.argmax(site_logits, dim=1)]

            positions = policy_batch["wt"]["resseq"][0][mutation_mask]
            selected_indices = torch.nonzero(mutation_mask, as_tuple=True)[0]
            icodes = policy_batch["wt"]["icode"][0]
            for actions in action_rows:
                mutations = []
                for global_index, state, action, position in zip(
                    selected_indices, states, actions, positions
                ):
                    state_index = int(state.item())
                    action_index = int(action.item())
                    if state_index == action_index:
                        raise RuntimeError("屏蔽 WT 动作后仍选中了 native action")
                    mutations.append(
                        (
                            antibody_chain,
                            int(position.item()),
                            icodes[int(global_index.item())],
                            AA_KEY[state_index],
                            AA_KEY[action_index],
                        )
                    )
                if len(mutations) != 3:
                    raise RuntimeError("三位点组合未生成恰好三个实际替换")
                mutate_info_list.append(canonical_mutate_info(mutations))

    expected_count = math.comb(len(true_indices), args.comb_num) * samples_per_triple
    if len(mutate_info_list) != expected_count or len(set(mutate_info_list)) != expected_count:
        raise RuntimeError(
            f"v8 候选数异常：期望 {expected_count} 条唯一候选，得到 "
            f"{len(mutate_info_list)} 条、其中唯一 {len(set(mutate_info_list))} 条"
        )
    return mutate_info_list


def sample_distinct_action_rows(
    probabilities: torch.Tensor, count: int
) -> list[torch.Tensor]:
    """逐位点独立采样，返回 ``count`` 个互不重复的完整动作。"""
    sampled_actions = []
    seen = set()
    max_attempts = count * 200
    for _ in range(max_attempts):
        actions = torch.multinomial(probabilities, num_samples=1).squeeze(1)
        action_key = tuple(int(action.item()) for action in actions)
        if action_key in seen:
            continue
        seen.add(action_key)
        sampled_actions.append(actions)
        if len(sampled_actions) == count:
            return sampled_actions
    raise RuntimeError(
        f"在 {max_attempts} 次逐位点独立采样后，仍无法得到 {count} 个不同完整动作"
    )


def make_policy_samples(train_dataset, mutate_info_list):
    """从同一 RepairPDB WT 状态重建三位点 mask 与实际目标 action。"""
    base_sample = train_dataset[0]
    index_by_residue = {
        (str(chain), int(resseq.item()), str(icode).strip()): index
        for index, (chain, resseq, icode) in enumerate(
            zip(
                base_sample["wt"]["chain_id"],
                base_sample["wt"]["resseq"],
                base_sample["wt"]["icode"],
            )
        )
    }
    samples, action_rows = [], []
    for mutate_info in mutate_info_list:
        target_by_index = {}
        for chain, position, icode, wt_aa, mut_aa in parse_mutation_info(mutate_info):
            residue_key = (str(chain), int(position), str(icode).strip())
            if residue_key not in index_by_residue:
                raise ValueError(f"v8 无法在 WT 策略输入中定位突变位点: {mutate_info}")
            residue_index = index_by_residue[residue_key]
            wt_index = AA_KEY.index(wt_aa)
            mut_index = AA_KEY.index(mut_aa)
            if int(base_sample["wt"]["aa"][residue_index].item()) != wt_index:
                raise ValueError(f"v8 WT 残基与突变记录不一致: {mutate_info}")
            if wt_index == mut_index:
                raise ValueError(f"v8 拒绝自替换候选: {mutate_info}")
            target_by_index[residue_index] = mut_index
        if len(target_by_index) != 3:
            raise ValueError(f"v8 只支持恰好三个实际突变: {mutate_info}")
        ordered_indices = sorted(target_by_index)
        sample = deepcopy(base_sample)
        sample_mask = torch.zeros_like(base_sample["mutation_mask"], dtype=torch.bool)
        sample_mask[torch.tensor(ordered_indices, dtype=torch.long)] = True
        sample["mutation_mask"] = sample_mask
        samples.append(sample)
        action_rows.append([target_by_index[index] for index in ordered_indices])
    return samples, action_rows


def non_wt_log_probs(model, policy_batch, mutation_mask, device):
    """排除 native action 后，返回每个决策位点的 categorical log-probability。"""
    logits = -model.choose_best_action(policy_batch, device)
    logits = logits.clone()
    selected_indices = mutation_mask.nonzero(as_tuple=False)
    native_actions = policy_batch["wt"]["aa"][mutation_mask]
    logits[selected_indices[:, 0], selected_indices[:, 1], native_actions] = torch.finfo(
        logits.dtype
    ).min
    return torch.nn.functional.log_softmax(logits, dim=-1)


def selected_joint_log_probs(
    log_probs: torch.Tensor,
    mutation_mask: torch.Tensor,
    target_actions: torch.Tensor,
) -> torch.Tensor:
    """只 gather 三个实际突变动作，并计算其联合 log-probability。"""
    batch_size = log_probs.size(0)
    selected = log_probs[mutation_mask]
    expected = batch_size * target_actions.size(1)
    if selected.size(0) != expected:
        raise RuntimeError(
            f"v8 选择动作数 {selected.size(0)} 与目标数 {expected} 不一致"
        )
    selected = selected.view(batch_size, target_actions.size(1), log_probs.size(-1))
    return selected.gather(2, target_actions.unsqueeze(-1)).squeeze(-1).sum(dim=1)


def balanced_inner_batches(candidate_count: int, batch_size: int, inner_passes: int):
    """每个 pass 使用一个独立随机排列，保证所有候选恰好遍历一次。"""
    if candidate_count <= 0 or batch_size <= 0 or inner_passes <= 0:
        raise ValueError("candidate_count、batch_size 与 inner_passes 必须为正数")
    for pass_index in range(inner_passes):
        permutation = torch.randperm(candidate_count).cpu().numpy()
        for start in range(0, candidate_count, batch_size):
            yield pass_index, permutation[start : start + batch_size]


def initial_early_stop_state() -> dict:
    """早停锚点与可恢复的真实历史最佳策略分别保存。"""
    return {
        "anchor_metric": math.inf,
        "anchor_epoch": 0,
        "bad_epochs": 0,
        "stop_epoch": 0,
        "best_policy_metric": math.inf,
        "best_policy_epoch": 0,
        "best_policy_state_dict": None,
        "best_policy_monitor_metrics": {},
        "last_monitor_metrics": {},
        "policy_restored": False,
        "restored_epoch": 0,
    }


def capture_best_policy(state, evo_model, monitor_value, epoch_one_based, monitor_metrics):
    """只要数值更低就快照，和 early-stop 的 min_delta 判据保持解耦。"""
    if monitor_value >= state["best_policy_metric"]:
        return state, False
    state["best_policy_metric"] = monitor_value
    state["best_policy_epoch"] = epoch_one_based
    state["best_policy_state_dict"] = {
        name: value.detach().cpu().clone()
        for name, value in evo_model.state_dict().items()
    }
    state["best_policy_monitor_metrics"] = dict(monitor_metrics)
    return state, True


def restore_best_policy(state, evo_model):
    """恢复 deterministic-greedy 指标数值最小的完整模型权重。"""
    checkpoint = state["best_policy_state_dict"]
    if checkpoint is None:
        raise RuntimeError("v8 无可恢复的历史最佳策略")
    evo_model.load_state_dict(checkpoint, strict=True)
    state["policy_restored"] = True
    state["restored_epoch"] = state["best_policy_epoch"]
    state["last_monitor_metrics"] = dict(state["best_policy_monitor_metrics"])
    return state


def update_early_stop_state(state, monitor_value, epoch_one_based, args):
    """用显著改善重置 patience；真实最优 checkpoint 由独立函数保存。"""
    if not math.isfinite(monitor_value):
        raise ValueError("deterministic greedy monitor 产生了非有限 acquisition")
    significant_improvement = monitor_value < state["anchor_metric"] - args.early_stop_min_delta
    if significant_improvement:
        state["anchor_metric"] = monitor_value
        state["anchor_epoch"] = epoch_one_based
        state["bad_epochs"] = 0
    elif epoch_one_based >= args.early_stop_min_epochs:
        state["bad_epochs"] += 1

    should_stop = (
        args.enable_early_stop
        and epoch_one_based >= args.early_stop_min_epochs
        and state["bad_epochs"] >= args.early_stop_patience
    )
    if should_stop and state["stop_epoch"] == 0:
        state["stop_epoch"] = epoch_one_based
    return state, should_stop, significant_improvement


def run_deterministic_greedy_monitor(
    args,
    evo_model,
    reward_models,
    train_dataset,
    collate_fn,
    device,
):
    """生成最终同构的 non-WT greedy 候选池；只用于监测和 checkpoint 选择。"""
    previous_phase = bool(getattr(args, "_v8_sampling_training_phase", False))
    args._v8_sampling_training_phase = False
    try:
        monitor_pool = nominate_and_score_candidates(
            args, evo_model, reward_models, train_dataset, collate_fn, device
        )
    finally:
        args._v8_sampling_training_phase = previous_phase
    prefix = f"deterministic_greedy_top{args.final_top_k}"
    return get_topk_energy_metrics(monitor_pool, args.final_top_k, prefix)


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
    """生成当前策略候选、构建缺失结构，并批处理返回未变换的能量 acquisition。"""
    started = time.perf_counter()
    mutate_info_list = generate_mutate_info_list(
        args, evo_model, train_dataset, collate_fn, device
    )
    if not mutate_info_list:
        seconds = time.perf_counter() - started
        print(f"v8_timing candidate_pool_seconds={seconds:.3f} candidates=0", flush=True)
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
    candidate_pool = {
        "mutate_info_list": mutate_info_list,
        "center_scores": center_scores,
        "std_scores": std_scores,
        "score_stack": score_stack,
        "acquisition": acquisition,
    }
    seconds = time.perf_counter() - started
    print(
        f"v8_timing candidate_pool_seconds={seconds:.3f} candidates={len(mutate_info_list)}",
        flush=True,
    )
    return candidate_pool


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
    """执行一轮 v8 GRPO，并在需要时恢复历史最佳 deterministic-greedy 策略。"""
    started_epoch = time.perf_counter()
    state = args._v8_early_stop_state
    pdb_id = train_dataset.pdb_id
    epoch_prefix = f"epoch_top{args.final_top_k}"
    loss_history.write(
        f"\nEpoch {epoch + 1}: v8 coverage-balanced action-gather GRPO；"
        f"inner_passes={args.inner_passes}。\n"
    )

    args._v8_sampling_training_phase = True
    try:
        candidate_pool = nominate_and_score_candidates(
            args, evo_model, reward_models, train_dataset, collate_fn, device
        )
    finally:
        # 后续 deterministic monitor 与最终提名均必须回到 greedy，不可遗留采样标记。
        args._v8_sampling_training_phase = False

    if candidate_pool is None:
        metrics = {
            "epoch": epoch,
            "policy_update_mode": V9_POLICY_UPDATE_MODE,
            "actions_per_triple": args.actions_per_triple,
            "inner_passes": args.inner_passes,
            "candidate_count": 0,
            "inner_update_count": 0,
            "inner_candidate_exposures": 0,
            "grpo_skipped": True,
            "training_update_skipped_by_early_stop": False,
            "early_stop_enabled": args.enable_early_stop,
            "early_stop_triggered": False,
            "train_loss": np.nan,
            **get_topk_energy_metrics(None, args.final_top_k, epoch_prefix),
        }
        save_epoch_metrics(loss_history, pdb_id, metrics)
        seconds = time.perf_counter() - started_epoch
        loss_history.write(f"v8_timing epoch={epoch + 1} total_epoch_seconds={seconds:.3f}\n")
        print(f"v8_timing epoch={epoch + 1} total_epoch_seconds={seconds:.3f}", flush=True)
        return metrics

    mutate_info_list = candidate_pool["mutate_info_list"]
    acquisition = candidate_pool["acquisition"]
    # 保留 v3 的全候选池相对优势，而非按三位点组合分组。
    train_scores = transform_acquisition(
        acquisition, args.reward_transform, args.reward_clip_mad
    )
    advantages = train_scores.detach().cpu().numpy()
    advantages = advantages.mean() - advantages
    advantage_std = advantages.std()
    if advantage_std > 1e-8:
        advantages = advantages / (advantage_std + 1e-8)

    policy_samples, action_rows = make_policy_samples(train_dataset, mutate_info_list)
    old_model = MERF(args).to(device)
    old_model.load_state_dict(evo_model.state_dict())
    old_model.eval()
    for parameter in old_model.parameters():
        parameter.requires_grad = False

    epoch_losses, policy_losses, kl_values = [], [], []
    update_count, candidate_exposures = 0, 0
    evo_model.train()
    for pass_index, sample_indices in tqdm(
        list(balanced_inner_batches(
            len(mutate_info_list), args.inner_batch_size, args.inner_passes
        )),
        desc=f"{pdb_id}: v8 balanced GRPO",
    ):
        policy_batch = recursive_to(
            collate_fn([policy_samples[index] for index in sample_indices]), device
        )
        target_actions = torch.as_tensor(
            [action_rows[index] for index in sample_indices],
            dtype=torch.long,
            device=device,
        )
        mutation_mask = policy_batch["mutation_mask"]
        if not torch.all(mutation_mask.sum(dim=1) == 3):
            raise RuntimeError("v8 策略 batch 未保留每条候选的三个突变位点")
        native_actions = policy_batch["wt"]["aa"][mutation_mask].view_as(target_actions)
        if torch.any(native_actions == target_actions):
            raise RuntimeError("v8 训练候选包含 native action")

        optimizer.zero_grad()
        with torch.no_grad():
            ref_log_probs = non_wt_log_probs(
                reference_model, policy_batch, mutation_mask, device
            )
            old_log_probs = non_wt_log_probs(
                old_model, policy_batch, mutation_mask, device
            )
            old_joint_log_probs = selected_joint_log_probs(
                old_log_probs, mutation_mask, target_actions
            )
        curr_log_probs = non_wt_log_probs(evo_model, policy_batch, mutation_mask, device)
        curr_joint_log_probs = selected_joint_log_probs(
            curr_log_probs, mutation_mask, target_actions
        )
        ratio = torch.exp(curr_joint_log_probs - old_joint_log_probs).clamp(max=20.0)
        advantage_tensor = torch.as_tensor(
            advantages[sample_indices], dtype=torch.float32, device=device
        )
        policy_loss = -(ratio * advantage_tensor).mean()
        ref_selected = ref_log_probs[mutation_mask]
        curr_selected = curr_log_probs[mutation_mask]
        kl_divergence = (
            ref_selected.exp() * (ref_selected - curr_selected)
        ).sum(dim=-1).mean()
        batch_loss = policy_loss + args.kl_coeff * kl_divergence
        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(float(batch_loss.item()))
        policy_losses.append(float(policy_loss.item()))
        kl_values.append(float(kl_divergence.item()))
        candidate_exposures += len(sample_indices)
        update_count += 1
        if tb_writer is not None:
            inner_step = epoch * 1000 + update_count
            tb_writer.add_scalar("train/policy_loss", policy_loss.item(), inner_step)
            tb_writer.add_scalar("train/kl_div", kl_divergence.item(), inner_step)
            tb_writer.add_scalar("train/batch_loss", batch_loss.item(), inner_step)
            tb_writer.add_scalar("train/coverage_pass", pass_index + 1, inner_step)

    topk_metrics = get_topk_energy_metrics(candidate_pool, args.final_top_k, epoch_prefix)
    monitor_metrics = run_deterministic_greedy_monitor(
        args, evo_model, reward_models, train_dataset, collate_fn, device
    )
    monitor_key = f"deterministic_greedy_top{args.final_top_k}_mean_acquisition"
    monitor_value = monitor_metrics[monitor_key]
    state, checkpoint_updated = capture_best_policy(
        state, evo_model, monitor_value, epoch + 1, monitor_metrics
    )
    state, should_stop, significant_improvement = update_early_stop_state(
        state, monitor_value, epoch + 1, args
    )
    is_last_epoch = epoch + 1 == args.total_epochs
    restore_now = args.enable_early_stop and (should_stop or is_last_epoch)
    if restore_now:
        state = restore_best_policy(state, evo_model)
    else:
        state["last_monitor_metrics"] = dict(monitor_metrics)
    args._v8_early_stop_state = state

    score_stack = candidate_pool["score_stack"]
    per_model_min = score_stack.min(dim=1).values.detach().cpu().tolist()
    selected_monitor = state["best_policy_monitor_metrics"]
    metrics = {
        "epoch": epoch,
        "policy_update_mode": V9_POLICY_UPDATE_MODE,
        "actions_per_triple": args.actions_per_triple,
        "inner_passes": args.inner_passes,
        "inner_update_count": update_count,
        "inner_candidate_exposures": candidate_exposures,
        "reward_label": args.reward_label,
        "reward_model_count": len(reward_models),
        "score_agg": args.score_agg,
        "uncertainty_weight": args.uncertainty_weight,
        "reward_transform": args.reward_transform,
        "reward_direction": args.reward_direction,
        "candidate_count": len(mutate_info_list),
        "grpo_skipped": False,
        "training_update_skipped_by_early_stop": False,
        "early_stop_enabled": args.enable_early_stop,
        "early_stop_significant_improvement": significant_improvement,
        "early_stop_checkpoint_updated": checkpoint_updated,
        "early_stop_triggered": should_stop,
        "early_stop_epoch": state["stop_epoch"],
        "early_stop_anchor_epoch": state["anchor_epoch"],
        "early_stop_anchor_metric": state["anchor_metric"],
        "early_stop_best_policy_epoch": state["best_policy_epoch"],
        "early_stop_best_policy_metric": state["best_policy_metric"],
        "early_stop_bad_epochs": state["bad_epochs"],
        "early_stop_policy_restored": state["policy_restored"],
        "early_stop_restored_epoch": state["restored_epoch"],
        "selected_policy_greedy_top8_mean_acquisition": selected_monitor.get(
            monitor_key, np.nan
        ),
        "selected_policy_greedy_top8_best_acquisition": selected_monitor.get(
            f"deterministic_greedy_top{args.final_top_k}_best_acquisition", np.nan
        ),
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
        **monitor_metrics,
    }
    save_epoch_metrics(loss_history, pdb_id, metrics)
    if tb_writer is not None:
        tb_writer.add_scalar("evo/num_candidates", len(mutate_info_list), epoch)
        tb_writer.add_scalar("train/loss", metrics["train_loss"], epoch)
        tb_writer.add_scalar(
            f"epoch/top{args.final_top_k}_mean_acquisition",
            topk_metrics[f"{epoch_prefix}_mean_acquisition"],
            epoch,
        )
        tb_writer.add_scalar(
            f"monitor/deterministic_greedy_top{args.final_top_k}_mean_acquisition",
            monitor_value,
            epoch,
        )
        tb_writer.add_scalar(
            "early_stop/best_policy_mean_acquisition", state["best_policy_metric"], epoch
        )
        tb_writer.add_scalar("early_stop/best_policy_epoch", state["best_policy_epoch"], epoch)
        tb_writer.add_scalar("early_stop/bad_epochs", state["bad_epochs"], epoch)
        tb_writer.add_scalar(
            "early_stop/policy_restored", float(state["policy_restored"]), epoch
        )
    loss_history.write(
        f"Epoch {epoch + 1}: train_candidates={len(mutate_info_list)}, "
        f"updates={update_count}, exposures={candidate_exposures}, "
        f"current_greedy_top{args.final_top_k}_mean={monitor_value:.6f}, "
        f"best_checkpoint_epoch={state['best_policy_epoch']}, "
        f"best_checkpoint_mean={state['best_policy_metric']:.6f}, "
        f"stop={should_stop}, restored={restore_now}\n"
    )
    seconds = time.perf_counter() - started_epoch
    loss_history.write(f"v8_timing epoch={epoch + 1} total_epoch_seconds={seconds:.3f}\n")
    print(
        f"Epoch:{epoch + 1}/{args.total_epochs} Candidates:{len(mutate_info_list)} "
        f"Updates:{update_count} GreedyTop{args.final_top_k}Mean:{monitor_value:.6f} "
        f"BestEpoch:{state['best_policy_epoch']} Stop:{should_stop} "
        f"Restored:{restore_now} Mode:v8_restore_best",
        flush=True,
    )
    print(f"v8_timing epoch={epoch + 1} total_epoch_seconds={seconds:.3f}", flush=True)
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

    pyrosetta_dir = os.path.join(output_dir, f"pyrosetta_{evaluation_label}_v8")
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

    colabdesign_dir = Path(output_dir) / f"colabdesign_{evaluation_label}_v8"
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
    output_path = os.path.join(loss_history.save_path, f"{pdb_id}_periodic_external_metrics_v8.csv")
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
    monitor_dir = Path(loss_history.save_path) / "periodic_external_eval_v8" / evaluation_label
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
    """完成一个 SAbDab v8 条目的训练和最终仅 top-8 的外部评测。"""
    # 每个起点独立维护 early-stop 状态，避免全数据集顺序运行时状态串扰。
    args._v8_early_stop_state = initial_early_stop_state()
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
        "v8 外部评测只使用选定最终策略重新提名的 top-k；"
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

        last_completed_epoch = -1
        training_stopped_early = False
        for epoch in range(args.total_epochs):
            epoch_metrics = train_one_epoch(
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

            last_completed_epoch = epoch
            if epoch_metrics.get("early_stop_triggered", False):
                training_stopped_early = True
                loss_history.write(
                    f"v8 早停于 epoch {epoch + 1} 触发；已恢复 epoch "
                    f"{args._v8_early_stop_state['restored_epoch']} 的历史最佳策略。\n"
                )
                break

        # 关键：训练结束或早停恢复后重新用最终策略提名；不读取历史 epoch 候选。
        final_pool = nominate_and_score_candidates(
            args,
            evo_model,
            reward_models,
            train_dataset,
            collate_fn,
            device,
        )
        early_stop_state = args._v8_early_stop_state
        selected_policy_epoch = (
            early_stop_state["restored_epoch"]
            if early_stop_state["policy_restored"]
            else last_completed_epoch + 1
        )
        final_epoch = selected_policy_epoch - 1
        final_rows = candidate_rows_from_pool(
            final_pool,
            final_epoch,
            source=(
                "final_policy_after_restored_best"
                if early_stop_state["policy_restored"]
                else "final_policy_after_last_update"
            ),
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
            "final_policy_epoch": selected_policy_epoch,
            "training_stopped_early": training_stopped_early,
            "early_stop_epoch": early_stop_state["stop_epoch"],
            "early_stop_best_policy_epoch": early_stop_state["best_policy_epoch"],
            "early_stop_best_policy_metric": early_stop_state["best_policy_metric"],
            "early_stop_policy_restored": early_stop_state["policy_restored"],
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
