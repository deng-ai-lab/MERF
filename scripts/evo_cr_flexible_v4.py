"""ESMFold2-started CR evolution with on-demand FoldX structures.

v4 keeps v3's full-backbone PPO, KL anchor, rollout reward, and optional
current-rollout elite loss.  Its data path is different: every binary action is
valid, and FoldX creates only the structures actually nominated by the policy.

The final nomination is always the exact policy top-k from the last completed
outer epoch.  The reward cache is retained only for per-epoch diagnostics.

中文审查导航（以下说明不参与任何计算）：
1. ``build_dataset`` 读取一个起点 CSV，并把其中对应的 fixed PDB 设为 WT；
2. ``train_one_epoch`` 从全 landscape 的二元绝对动作空间采样，按预测奖励进行 PPO 更新；
3. ``score_action_batch`` 才会为被采样/提名的动作按需生成 FoldX 结构；
4. ``write_final_candidates`` 只写最后一个已完成 epoch 的 policy top-k。
``candidate_cache`` 是跨 epoch 的奖励诊断缓存，绝不是最终提名来源。
"""

import argparse
import datetime
import math
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_cr_evo_flexible_v4 import FlexibleCREvoDatasetV4, canonical_reverse_mutation_key
from model.MERF_evo_flexible_v3 import MERF
from protein.mutate_scripts_foldx_flexible_v4 import DEFAULT_FOLDX_BIN
from protein.read_pdbs import PaddingCollate
from utils.losshistory import LossHistory
from utils.util import recursive_to, seed_all


PROJECT_DIR = "/home/dataset-local/projects_dir/MERF"
CR_DIR = os.path.join(PROJECT_DIR, "data", "CR")
EVO_ROOT = os.path.join(PROJECT_DIR, "data", "CR_evo_esmfold2")
CHECKPOINT_MODEL_ARG_KEYS = (
    "seed", "use_plm_embedding", "plm_path", "feat_dim", "rel_dim", "max_relpos",
    "ipa_layer", "ga_layer", "knn_neighbors_num", "knn_agents_num", "obs_shape",
    "n_actions", "n_agents", "agent_hidden_dim", "mixer_rel_dim", "mixer_ga_layer",
    "mixing_embed_dim", "hypernet_embed", "hypernet_layers", "use_kl_loss", "kl_loss_weight",
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"true", "1", "yes", "y"}:
        return True
    if value.lower() in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid bool value: {value}")


def normalize_dataset_name(dataset):
    value = dataset.strip()
    parts = value.split("_")
    if len(parts) == 3 and parts[0] == "cr" and parts[1].isdigit() and parts[2].startswith("h"):
        return f"cr{parts[1]}_{parts[2]}"
    if len(parts) == 2 and parts[0].startswith("cr") and parts[0][2:].isdigit() and parts[1].startswith("h"):
        return value
    raise ValueError("--dataset must look like cr6261_h9 or cr_6261_h9")


def get_args():
    parser = argparse.ArgumentParser()

    # 起点 CSV 必须恰好有一行；其中的 reverse_mutation 标识“该结构对应的起点”。
    # 它不缩小后续策略可选的 landscape，也不等同于“只允许向成熟方向突变”。
    parser.add_argument("--dataset", default="cr6261_h1")
    parser.add_argument("--start_evo_csv", default="/home/dataset-local/projects_dir/MERF/data/CR_evo_esmfold2/cr6261_h1_KA59N,FA75S,GA77S,VA79A,VA104L/cr6261_h1_KA59N,FA75S,GA77S,VA79A,VA104L_evo.csv")
    # parser.add_argument("--dataset", default="cr6261_h9")
    # parser.add_argument("--start_evo_csv", default="")
    # parser.add_argument("--dataset", default="cr9114_h1")
    # parser.add_argument("--start_evo_csv", default="")

    # 此参数仅用于与 start_evo_csv 交叉校验；真正的起点以 CSV 内容为准。
    parser.add_argument("--start_reverse_mutations", "--start_mutant", default="")
    parser.add_argument("--evo_root", default=EVO_ROOT)
    parser.add_argument("--log_root", default="logs/evo_cr_flexible_v4")
    parser.add_argument("--run_name", default="v4")
    # 每个 epoch 诊断当前策略的精确 top-k；最终输出也固定使用同一个 k。
    parser.add_argument("--policy_eval_top_k", type=int, default=8)
    parser.add_argument("--policy_eval_interval", type=int, default=5)
    parser.add_argument("--foldx_workers", type=int, default=8)
    parser.add_argument("--foldx_bin", default=DEFAULT_FOLDX_BIN)
    # 必须为 true：raw ESMFold2 PDB 会先 RepairPDB，fixed PDB 同时作为 WT 与 BuildModel 父结构。
    parser.add_argument(
        "--repair_start",
        type=str2bool,
        default=True,
        help="Must remain true: v4 always uses FoldX RepairPDB before MERF/FoldX mutation.",
    )

    # Checkpoint/model arguments: these defaults match v3/v8 and are overwritten
    # from args.pkl when the chosen checkpoint supplies architecture settings.
    # 审查时须注意：架构字段可能被 checkpoint 邻近的 args.pkl 覆盖，命令行值不一定生效。
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--adapter_lr", type=float, default=3e-2)
    parser.add_argument("--is_cuda", type=str2bool, default=True)
    parser.add_argument("--gpu_idx", type=int, default=0)
    parser.add_argument("--use_plm_embedding", type=str2bool, default=True)
    parser.add_argument(
        "--plm_path",
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
        default="/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_06_09_16_56_32/Epoch200_MERF.pth",
    )
    # reward model 只产生优化用的预测分数；不读取 landscape 的真实 score/rank。
    parser.add_argument("--reward_model_paths", default="")
    parser.add_argument("--uncertainty_weight", type=float, default=0.5)
    parser.add_argument("--score_agg", choices=["mean", "median"], default="mean")
    parser.add_argument("--reward_transform", choices=["raw", "rank", "clip"], default="raw")
    parser.add_argument("--reward_direction", choices=["min", "max"], default="min")
    parser.add_argument("--reward_clip_mad", type=float, default=3.0)
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

    # v3 PPO settings. ``total_epochs`` 是 outer rollout 次数，``inner_epochs``
    # 是每次固定 rollout 上连续执行的 PPO 梯度更新次数。
    # FoldX structures are generated on demand.  Five rollout batches with six
    # PPO updates each retain 30 clipped policy updates while avoiding the
    # prohibitive 200-batch structural workload of the former default.
    parser.add_argument("--total_epochs", type=int, default=5)
    parser.add_argument("--rollout_samples", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.5)
    parser.add_argument("--inner_epochs", type=int, default=6)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--normalize_advantage", type=str2bool, default=True)
    parser.add_argument("--baseline_momentum", type=float, default=0.8)
    parser.add_argument("--kl_coeff", type=float, default=0.02)
    parser.add_argument("--train_mode", choices=["bias_only", "finetune"], default="finetune")
    parser.add_argument("--elite_mode", choices=["none", "old", "hamming"], default="none")
    parser.add_argument("--elite_loss_weight", type=float, default=0.2)
    # The default rollout has eight candidates.  Keeping all eight as elites
    # would reinforce bad and good samples alike, so retain only the best two.
    # Users running larger rollouts can still set this explicitly.
    parser.add_argument("--elite_top_k", type=int, default=2)
    parser.add_argument("--elite_min_hamming", type=int, default=2)
    parser.add_argument("--entropy_reg_weight", type=float, default=0.0)
    args = parser.parse_args()

    args.dataset = normalize_dataset_name(args.dataset)
    # 统一突变字符串的顺序，确保目录名、结构文件名、评测表能使用同一个 key 对齐。
    args.start_reverse_mutations = canonical_reverse_mutation_key(args.start_reverse_mutations)
    if not args.reward_model_paths:
        args.reward_model_paths = args.model_load_path
    for name in (
        "total_epochs", "rollout_samples", "inner_epochs", "policy_eval_top_k",
        "policy_eval_interval", "foldx_workers", "elite_top_k",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name} must be positive")
    if args.sample_temperature <= 0:
        raise ValueError("--sample_temperature must be positive")
    if not args.repair_start:
        raise ValueError(
            "v4 requires --repair_start true so WT and all candidate structures use the same repaired start."
        )
    return args


def load_model_args_from_checkpoint(args):
    args_path = os.path.join(os.path.dirname(os.path.abspath(args.model_load_path)), "args.pkl")
    if not os.path.exists(args_path):
        print(f"Checkpoint args not found at {args_path}; using command-line architecture settings.")
        return None, []
    with open(args_path, "rb") as handle:
        checkpoint_args = pickle.load(handle)
    loaded = []
    # 这里只覆盖网络结构/输入相关参数，不覆盖本次实验的 PPO 超参数或起点参数。
    for key in CHECKPOINT_MODEL_ARG_KEYS:
        if key in vars(checkpoint_args):
            setattr(args, key, vars(checkpoint_args)[key])
            loaded.append(key)
    print(f"Loaded architecture args from {args_path}: {', '.join(loaded)}")
    return args_path, loaded


def load_pretrained(model, path, device):
    state = torch.load(path, map_location=device)
    # strict=False 不会因为键缺失/多余而中断；审查每次运行日志中的 missing/unexpected
    # 是确认 checkpoint 与当前 MERF 架构兼容的必要步骤。
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {path}; missing={missing}; unexpected={unexpected}")


def configure_trainable_params(model, args):
    """Keep v3's full-backbone finetuning; do not train a site-specific bias."""
    for name, parameter in model.named_parameters():
        # v4 的策略更新调整主干网络；site_action_bias 即使存在也明确冻结，
        # 避免通过单个位点偏置绕过结构/图网络表征。
        if name == "site_action_bias":
            parameter.requires_grad = False
        else:
            parameter.requires_grad = args.train_mode == "finetune"


def build_optimizer(model, args):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        raise ValueError("No trainable model parameters")
    return optim.Adam(parameters, lr=args.lr, weight_decay=1e-4)


def selected_action_log_probs(binary_log_probs, actions, reduce="sum"):
    # 每个位点独立输出二元 logits。一个完整动作的概率是各位点概率的乘积，
    # 因而其 log probability 是各位点被选 action 的 log probability 之和。
    if actions.dim() == 1:
        actions = actions.unsqueeze(0)
    actions = actions.to(binary_log_probs.device)
    expanded = binary_log_probs.unsqueeze(0).expand(actions.size(0), -1, -1)
    selected = expanded.gather(2, actions.unsqueeze(-1)).squeeze(-1)
    if reduce == "sum":
        return selected.sum(dim=1)
    if reduce == "mean":
        return selected.mean(dim=1)
    raise ValueError(f"Unsupported reduction: {reduce}")


def aggregate_score_stack(score_stack, score_agg):
    # 多个 reward checkpoint（若提供）在此汇总；std 仅代表模型间离散度。
    if score_agg == "mean":
        center = score_stack.mean(dim=0)
    elif score_agg == "median":
        center = score_stack.median(dim=0).values
    else:
        raise ValueError(f"Unsupported score aggregation: {score_agg}")
    std = score_stack.std(dim=0, unbiased=False) if score_stack.size(0) > 1 else torch.zeros_like(center)
    return center, std


def make_acquisition(center_scores, std_scores, uncertainty_weight, reward_direction):
    # 后续所有“更好”的定义都统一为 acquisition 更小：
    # reward_direction=max 时先取负号，仍可沿用同一套 PPO/elite 排序。
    if reward_direction == "min":
        return center_scores + uncertainty_weight * std_scores
    if reward_direction == "max":
        return -center_scores + uncertainty_weight * std_scores
    raise ValueError(f"Unsupported reward direction: {reward_direction}")


def transform_acquisition(acquisition, reward_transform, clip_mad):
    # 这是 PPO 更新时使用的奖励尺度变换；不会改变日志中的原始 acquisition。
    if reward_transform == "raw":
        return acquisition
    if reward_transform == "rank":
        if acquisition.numel() <= 1:
            return torch.zeros_like(acquisition)
        order = torch.argsort(acquisition.detach())
        ranks = torch.empty_like(acquisition)
        ranks[order] = torch.arange(acquisition.numel(), dtype=acquisition.dtype, device=acquisition.device)
        return ranks / float(acquisition.numel() - 1)
    if reward_transform == "clip":
        median = acquisition.detach().median()
        mad = (acquisition.detach() - median).abs().median()
        scale = torch.clamp(1.4826 * mad, min=1e-6)
        return acquisition.clamp(median - clip_mad * scale, median + clip_mad * scale)
    raise ValueError(f"Unsupported reward transform: {reward_transform}")


def choose_elites(acquisition, actions, top_k, mode, min_hamming):
    if mode == "none":
        return None
    count = min(top_k, acquisition.numel())
    # acquisition 越小越优。hamming 模式先尽量让精英动作彼此相隔足够远，
    # 再在候选不够时回退到普通 top-k；它不使用真实 landscape rank。
    sorted_indices = torch.argsort(acquisition.detach())
    if mode == "old":
        return actions[sorted_indices[:count]]
    selected = []
    for index in sorted_indices.tolist():
        action = actions[index]
        if all(int((action != actions[other]).sum().item()) >= min_hamming for other in selected):
            selected.append(index)
        if len(selected) == count:
            break
    for index in sorted_indices.tolist():
        if len(selected) == count:
            break
        if index not in selected:
            selected.append(index)
    return actions[torch.tensor(selected, dtype=torch.long, device=actions.device)]


def score_action_batch(dataset, collate_fn, reward_models, actions, device, score_agg):
    """为被提名的动作建模并打分；未被提名的全 landscape 不会在此生成结构。"""
    # Dataset 内部先确认每个 action 对应的 PDB：以 fixed 起点为共同父结构，
    # 将绝对目标状态转换成相对 FoldX 突变，再按需调用 BuildModel。
    dataset.ensure_action_structures(actions)
    # reward model 输入的是同一 fixed WT 与该动作的 FoldX mutant 结构对。
    score_batches = [dataset.getitem_by_action_tensor(action) for action in actions.detach().cpu()]
    score_batch = recursive_to(collate_fn(score_batches), device)
    model_scores = []
    with torch.no_grad():
        for reward_model in reward_models:
            scores, _, _, _ = reward_model(score_batch, device)
            model_scores.append(scores.reshape(-1))
    score_stack = torch.stack(model_scores, dim=0)
    center, std = aggregate_score_stack(score_stack, score_agg)
    return center, std, score_stack


def exact_policy_topk_actions(dataset, binary_log_probs, top_k):
    """在全部二元动作状态中精确找策略概率最大的 top-k，尚不生成任何 PDB。"""
    # all_action_matrix 的每行是一个完整 landscape 状态：0=非成熟残基，1=成熟残基。
    # 该枚举只计算 2^site_num 个动作的概率；结构生成延迟到 score_action_batch。
    actions = dataset.all_action_matrix.to(binary_log_probs.device)
    expanded = binary_log_probs.unsqueeze(0).expand(actions.size(0), -1, -1)
    scores = expanded.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    indices = torch.topk(scores, k=min(top_k, actions.size(0))).indices
    return actions[indices].detach(), scores[indices].detach()


def rank_metrics(dataset, reverse_keys, prefix):
    # 真实 landscape score/rank 只用于离线评测和日志，不参与 reward、advantage 或梯度。
    # rank 仅在当前 pdb_id 自己的 landscape 内定义，不能跨抗体-抗原复合物比较数值。
    ranks = []
    shown_ranks = []
    num_ranked = len(dataset.rank_by_reverse_key)
    top1pct_cutoff = int(math.ceil(num_ranked * 0.01))
    top5pct_cutoff = int(math.ceil(num_ranked * 0.05))
    for key in reverse_keys:
        rank = dataset.evaluate_reverse_key(key)["rank"]
        if rank is None or pd.isna(rank):
            shown_ranks.append("NA")
        else:
            rank = int(rank)
            ranks.append(rank)
            shown_ranks.append(str(rank))
    return {
        f"{prefix}_ranked_count": len(ranks),
        f"{prefix}_min_rank": int(min(ranks)) if ranks else np.nan,
        f"{prefix}_mean_rank": float(np.mean(ranks)) if ranks else np.nan,
        # 命中表示提名候选中至少有一个落入完整实验 landscape 的前 1% / 前 5%。
        f"{prefix}_landscape_size": num_ranked,
        f"{prefix}_top1pct_rank_cutoff": top1pct_cutoff,
        f"{prefix}_top5pct_rank_cutoff": top5pct_cutoff,
        f"{prefix}_recall_top1pct": int(any(rank <= top1pct_cutoff for rank in ranks)),
        f"{prefix}_recall_top5pct": int(any(rank <= top5pct_cutoff for rank in ranks)),
        f"{prefix}_selected_keys": "|".join(reverse_keys),
        f"{prefix}_selected_ranks": "|".join(shown_ranks),
    }


def update_candidate_cache(cache, dataset, actions, center_scores, std_scores, acquisition, epoch, source):
    # 这是跨训练轨迹的“历史最小 acquisition”缓存：同一 reverse_key 只保留其
    # 历史最佳预测分数。它用于 reward_topk 诊断，绝不能当成 final policy top-k。
    for index, action in enumerate(actions.detach().cpu()):
        reverse_key = dataset.action_tensor_to_reverse_key(action)
        candidate = {
            "reverse_key": reverse_key,
            "foldx_mutations": dataset.foldx_mutations_from_target(reverse_key),
            "epoch": int(epoch),
            "source": source,
            "center_score": float(center_scores[index].item()),
            "std_score": float(std_scores[index].item()),
            "acquisition": float(acquisition[index].item()),
        }
        previous = cache.get(reverse_key)
        if previous is None or candidate["acquisition"] < previous["acquisition"]:
            cache[reverse_key] = candidate


def top_cache_metrics(cache, dataset, top_k):
    # 返回的是“至今为止 cache 最优”的 reward_topk 指标；名称刻意保留 reward_ 前缀，
    # 与 final_topk（最后策略）严格区分。
    candidates = sorted(cache.values(), key=lambda item: item["acquisition"])[:top_k]
    reverse_keys = [candidate["reverse_key"] for candidate in candidates]
    metrics = rank_metrics(dataset, reverse_keys, f"reward_top{top_k}")
    metrics[f"reward_top{top_k}_count"] = len(candidates)
    metrics[f"reward_top{top_k}_best_acquisition"] = (
        float(candidates[0]["acquisition"]) if candidates else np.nan
    )
    metrics[f"reward_top{top_k}_mean_acquisition"] = (
        float(np.mean([candidate["acquisition"] for candidate in candidates])) if candidates else np.nan
    )
    return metrics, candidates


def resolve_start_evo_csv(args):
    # 优先使用显式路径；否则按 dataset + 起点 reverse key 推导 CR_evo_esmfold2 目录。
    if args.start_evo_csv:
        path = os.path.abspath(args.start_evo_csv)
    else:
        folder_id = args.dataset if args.start_reverse_mutations == "" else (
            f"{args.dataset}_{args.start_reverse_mutations}"
        )
        path = os.path.join(args.evo_root, folder_id, f"{folder_id}_evo.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Start evolution CSV not found: {path}")
    return path


def build_dataset(args, device):
    start_evo_csv = resolve_start_evo_csv(args)
    # 防止批处理把多行 CSV 或错误起点静默传入一次单起点进化。
    start_df = pd.read_csv(start_evo_csv, dtype={"pdb_id": "string"})
    if len(start_df) != 1:
        raise ValueError(f"Expected exactly one start row in {start_evo_csv}, got {len(start_df)}")
    start_row = start_df.iloc[0]
    start_dataset = normalize_dataset_name(str(start_row["pdb_id"]))
    if start_dataset != args.dataset:
        raise ValueError(f"--dataset={args.dataset} conflicts with start CSV dataset={start_dataset}")
    start_key = canonical_reverse_mutation_key(start_row["reverse_mutation"])
    if args.start_reverse_mutations and start_key != args.start_reverse_mutations:
        raise ValueError("--start_reverse_mutations conflicts with the start_evo_csv content")
    args.start_reverse_mutations = start_key

    # cr_evo.csv 提供链、抗原与 CDR 区域定义；它不是起点突变后的序列来源。
    # 起点 WT 的实际序列/坐标由 start_dir 下的 fixed PDB 决定。
    metadata = pd.read_csv(os.path.join(CR_DIR, "cr_evo.csv"), dtype={"pdb_id": "string"})
    rows = metadata[metadata["pdb_id"] == args.dataset]
    if rows.empty:
        raise ValueError(f"{args.dataset} not found in {CR_DIR}/cr_evo.csv")
    row = rows.iloc[0]
    start_dir = os.path.dirname(start_evo_csv)
    embedding_path = os.path.join(start_dir, "plm_embeddings", f"{os.path.basename(start_dir)}_esm2_650_embeddings.pkl")
    # FlexibleCREvoDatasetV4 负责：RepairPDB、全 landscape 动作枚举、
    # 绝对动作到“相对当前起点”的 FoldX 突变映射，以及真实 rank 的离线查询。
    dataset = FlexibleCREvoDatasetV4(
        pdb_id=args.dataset,
        antibody_chain=row["antibody_chain"],
        partner=row["partner"],
        sequence=row["sequence"],
        cdr=row["cdr3"],
        cr_dir=CR_DIR,
        start_dir=start_dir,
        start_reverse_mutations=start_key,
        knn_num=args.knn_neighbors_num,
        knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding,
        plm_path=args.plm_path,
        plm_embedding_path=embedding_path,
        device=device,
        foldx_workers=args.foldx_workers,
        foldx_bin=args.foldx_bin,
        repair_start=args.repair_start,
    )
    return dataset, start_evo_csv


def train_one_epoch(
    args,
    model,
    reward_models,
    reference_model,
    optimizer,
    epoch,
    dataset,
    collate_fn,
    loss_history,
    device,
    reward_baseline,
    candidate_cache,
    writer=None,
):
    # Dataset 长度固定为 1：这个 batch 始终是“当前起点的 fixed WT 结构”，
    # 而不是 rollout 中每个突变体的结构。突变体只在 reward 打分时单独生成。
    policy_batch = recursive_to(collate_fn([dataset.__getitem__(0)]), device)
    model.eval()
    with torch.no_grad():
        # 先冻结本轮 rollout 的旧策略。采样温度只影响探索时的分布，
        # raw_actions 的每一行仍是全 landscape 的绝对成熟/非成熟状态。
        old_log_probs = model.get_binary_log_probs(
            policy_batch, dataset, device, temperature=args.sample_temperature
        )
        raw_actions = torch.distributions.Categorical(logits=old_log_probs).sample((args.rollout_samples,))
        old_action_log_probs = selected_action_log_probs(old_log_probs, raw_actions, reduce="sum")
        # 注意：v4 的 reference 这里未显式传 sample_temperature，因而使用模型方法默认值；
        # v5 对此处做了温度一致性的单独处理。该 reference 只用于 KL 正则。
        reference_log_probs = reference_model.get_binary_log_probs(policy_batch, dataset, device)

        # 同一动作可能被采样多次。去重后才做 FoldX + reward，随后 inverse 映射回 rollout，
        # 这样重复样本保留在 PPO 估计中，但不会重复建模相同结构。
        unique_actions, inverse = torch.unique(raw_actions, dim=0, return_inverse=True)
        unique_center, unique_std, _ = score_action_batch(
            dataset, collate_fn, reward_models, unique_actions, device, args.score_agg
        )
        center_scores = unique_center[inverse]
        std_scores = unique_std[inverse]
        # 这里的分数完全来自 reward model；真实 h1_score/rank 未被读取。
        acquisition = make_acquisition(
            center_scores, std_scores, args.uncertainty_weight, args.reward_direction
        )
        train_acquisition = transform_acquisition(acquisition, args.reward_transform, args.reward_clip_mad)
        raw_policy_entropy = -(old_log_probs.exp() * old_log_probs).sum(dim=-1).mean()

    # 对最小化 acquisition 的任务，低于 baseline 的候选获得正 advantage，
    # 因而 PPO 会提高该完整动作在下一轮被采样到的概率。
    batch_baseline = float(train_acquisition.mean().item())
    if reward_baseline is None:
        reward_baseline = batch_baseline
    advantages = reward_baseline - train_acquisition.detach()
    if args.normalize_advantage and advantages.numel() > 1:
        standard_deviation = advantages.std(unbiased=False)
        if float(standard_deviation.item()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (standard_deviation + 1e-8)
    elite_actions = choose_elites(
        train_acquisition, raw_actions, args.elite_top_k, args.elite_mode, args.elite_min_hamming
    )

    # 以下 inner_epochs 次更新都复用本轮 raw_actions 和对应 reward，
    # 不会在每个 inner step 重新 FoldX 或重新查询真实 landscape。
    losses, policy_losses, kl_values, entropy_values, elite_values = [], [], [], [], []
    model.train()
    for _ in range(args.inner_epochs):
        optimizer.zero_grad()
        current_log_probs = model.get_binary_log_probs(
            policy_batch, dataset, device, temperature=args.sample_temperature
        )
        current_action_log_probs = selected_action_log_probs(current_log_probs, raw_actions, reduce="sum")
        # PPO clipped surrogate：ratio 与 old policy 比较，限制一次 rollout 内的策略漂移。
        ratio = torch.exp(current_action_log_probs - old_action_log_probs).clamp(max=20.0)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        # 冻结的 reference 防止主干网络在小样本 rollout 上偏离预训练分布过远。
        kl_divergence = (
            reference_log_probs.exp() * (reference_log_probs - current_log_probs)
        ).sum(dim=-1).mean()
        entropy = -(current_log_probs.exp() * current_log_probs).sum(dim=-1).mean()
        # elite loss 是额外地提高本轮低 acquisition 样本的概率；
        # 是否启用及其多样性约束完全由 --elite_* 参数决定。
        if elite_actions is not None and args.elite_loss_weight > 0:
            elite_loss = -selected_action_log_probs(current_log_probs, elite_actions, reduce="mean").mean()
        else:
            elite_loss = torch.zeros((), device=device)
        loss = (
            policy_loss
            + args.kl_coeff * kl_divergence
            + args.elite_loss_weight * elite_loss
            - args.entropy_reg_weight * entropy
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        kl_values.append(float(kl_divergence.item()))
        entropy_values.append(float(entropy.item()))
        elite_values.append(float(elite_loss.item()))

    if args.baseline_momentum >= 0:
        reward_baseline = args.baseline_momentum * reward_baseline + (1.0 - args.baseline_momentum) * batch_baseline
    else:
        reward_baseline = batch_baseline

    # PPO 更新后，重新计算“当前模型”的精确 policy top-k。这里的默认温度为 1.0，
    # 因此它表示部署/汇报时的策略排序，而不是带探索温度的 rollout 排序。
    # 该步骤不依赖预先存在的全 landscape PDB。
    model.eval()
    with torch.no_grad():
        evaluation_log_probs = model.get_binary_log_probs(policy_batch, dataset, device)
        policy_actions, _ = exact_policy_topk_actions(dataset, evaluation_log_probs, args.policy_eval_top_k)
    # 为节省 FoldX 时间，非末轮只按 interval 对这批诊断 top-k 建模；
    # 最后一个 epoch 强制建模一次，保证最终 policy top-k 可被打分并落盘。
    score_policy_structures = (
        epoch % args.policy_eval_interval == 0 or epoch == args.total_epochs - 1
    )
    if score_policy_structures:
        with torch.no_grad():
            policy_center, policy_std, _ = score_action_batch(
                dataset, collate_fn, reward_models, policy_actions, device, args.score_agg
            )
            policy_acquisition = make_acquisition(
                policy_center, policy_std, args.uncertainty_weight, args.reward_direction
            )
    # rollout 与（部分 epoch 的）policy top-k 都可以进入 cache，便于观察历史 reward 最优点。
    # 这一段不会决定 final_top8_candidates.csv 的内容。
    update_candidate_cache(
        candidate_cache, dataset, unique_actions, unique_center, unique_std,
        make_acquisition(unique_center, unique_std, args.uncertainty_weight, args.reward_direction),
        epoch, "rollout",
    )
    if score_policy_structures:
        update_candidate_cache(
            candidate_cache, dataset, policy_actions, policy_center, policy_std, policy_acquisition,
            epoch, "policy_topk",
        )
    # 两类 rank 指标都仅为离线诊断：policy_* 是当前 epoch 策略，reward_* 是历史 cache。
    # 二者都不反向传播，也不参与下一轮的 reward 计算。
    policy_keys = [dataset.action_tensor_to_reverse_key(action) for action in policy_actions]
    policy_metrics = rank_metrics(dataset, policy_keys, f"policy_top{args.policy_eval_top_k}")
    reward_metrics, _ = top_cache_metrics(candidate_cache, dataset, args.policy_eval_top_k)

    sample_best_index = int(torch.argmin(acquisition).item())
    sample_best_key = dataset.action_tensor_to_reverse_key(raw_actions[sample_best_index])
    sample_best_rank = dataset.evaluate_reverse_key(sample_best_key)["rank"]
    metrics = {
        "epoch": epoch,
        "run_name": args.run_name,
        "reward_baseline": reward_baseline,
        "reward_model_count": len(reward_models),
        "train_mode": args.train_mode,
        "elite_mode": args.elite_mode,
        "elite_loss_weight": args.elite_loss_weight,
        "acquisition_min": float(acquisition.min().item()),
        "acquisition_mean": float(acquisition.mean().item()),
        "sample_best_reverse_key": sample_best_key,
        "sample_best_rank": np.nan if sample_best_rank is None else int(sample_best_rank),
        "train_sample_unique_candidates": int(unique_actions.size(0)),
        "policy_structure_scored": bool(score_policy_structures),
        "raw_policy_entropy": float(raw_policy_entropy.item()),
        "train_entropy": float(np.mean(entropy_values)),
        "train_loss": float(np.mean(losses)),
        "train_policy_loss": float(np.mean(policy_losses)),
        "train_kl": float(np.mean(kl_values)),
        "elite_loss": float(np.mean(elite_values)),
        **policy_metrics,
        **reward_metrics,
    }
    print(
        f"Epoch:{epoch + 1} SampleRank:{metrics['sample_best_rank']} "
        f"PolicyTop{args.policy_eval_top_k}Min:{metrics[f'policy_top{args.policy_eval_top_k}_min_rank']} "
        f"RewardTop{args.policy_eval_top_k}Min:{metrics[f'reward_top{args.policy_eval_top_k}_min_rank']} "
        f"Entropy:{metrics['raw_policy_entropy']:.4f}",
        flush=True,
    )
    loss_history.write(
        f"Epoch {epoch + 1}: sample_rank={metrics['sample_best_rank']}, "
        f"policy_top{args.policy_eval_top_k}_min={metrics[f'policy_top{args.policy_eval_top_k}_min_rank']}, "
        f"reward_top{args.policy_eval_top_k}_min={metrics[f'reward_top{args.policy_eval_top_k}_min_rank']}\n"
    )
    if writer is not None:
        for key in (
            "raw_policy_entropy", "train_loss",
            f"policy_top{args.policy_eval_top_k}_min_rank",
            f"reward_top{args.policy_eval_top_k}_min_rank",
        ):
            value = metrics[key]
            if not pd.isna(value):
                writer.add_scalar(key, value, epoch + 1)
        writer.flush()
    save_file = os.path.join(loss_history.save_path, dataset.pdb_id + ".csv")
    history = pd.read_csv(save_file) if os.path.exists(save_file) else pd.DataFrame()
    pd.concat([history, pd.DataFrame([metrics])], ignore_index=True).to_csv(save_file, index=False)
    return reward_baseline, metrics


def write_initial_metrics(args, model, dataset, collate_fn, reward_models, loss_history, device, candidate_cache):
    # epoch=-1 是训练前的 policy 基线。它写入 cache 仅用于观察 reward_topk 轨迹，
    # 不会与训练结束时的 final policy top-k 混合输出。
    policy_batch = recursive_to(collate_fn([dataset.__getitem__(0)]), device)
    model.eval()
    with torch.no_grad():
        log_probs = model.get_binary_log_probs(policy_batch, dataset, device)
        policy_actions, _ = exact_policy_topk_actions(dataset, log_probs, args.policy_eval_top_k)
        center, std, _ = score_action_batch(dataset, collate_fn, reward_models, policy_actions, device, args.score_agg)
        acquisition = make_acquisition(center, std, args.uncertainty_weight, args.reward_direction)
    update_candidate_cache(candidate_cache, dataset, policy_actions, center, std, acquisition, -1, "before_training")
    policy_keys = [dataset.action_tensor_to_reverse_key(action) for action in policy_actions]
    metrics = {
        "epoch": -1,
        "metric_stage": "before_training",
        "run_name": args.run_name,
        "start_reverse_key": dataset.start_reverse_key,
        "start_rank": dataset.evaluate_reverse_key(dataset.start_reverse_key)["rank"],
        **rank_metrics(dataset, policy_keys, f"policy_top{args.policy_eval_top_k}"),
        **top_cache_metrics(candidate_cache, dataset, args.policy_eval_top_k)[0],
    }
    save_file = os.path.join(loss_history.save_path, dataset.pdb_id + ".csv")
    pd.DataFrame([metrics]).to_csv(save_file, index=False)
    loss_history.write(
        f"Before training: start_rank={metrics['start_rank']}, "
        f"policy_top{args.policy_eval_top_k}_min={metrics[f'policy_top{args.policy_eval_top_k}_min_rank']}\n"
    )


def write_final_candidates(
    args,
    model,
    dataset,
    collate_fn,
    reward_models,
    device,
    candidate_cache,
    loss_history,
    final_epoch,
):
    """写入最终模型状态的精确 policy top-k，而非训练轨迹的 cache top-k。

    ``candidate_cache`` 只作为诊断计数写入 summary；它汇总所有 epoch 的 rollout，
    因此不能决定最终提名。此处重新评估最终模型，也使 v5 早停时能够正确使用
    “最后实际完成的 epoch”，而不是预设的 ``args.total_epochs``。
    """
    policy_batch = recursive_to(collate_fn([dataset.__getitem__(0)]), device)
    model.eval()
    with torch.no_grad():
        log_probs = model.get_binary_log_probs(policy_batch, dataset, device)
        # 这是 final_top8 的唯一候选来源。exact_policy_topk_actions 返回顺序为
        # 总 policy log probability 由高到低；不是 reward/acquisition 由低到高。
        policy_actions, policy_log_probabilities = exact_policy_topk_actions(
            dataset, log_probs, args.policy_eval_top_k
        )
        center, std, _ = score_action_batch(
            dataset, collate_fn, reward_models, policy_actions, device, args.score_agg
        )
        acquisition = make_acquisition(
            center, std, args.uncertainty_weight, args.reward_direction
        )

    rows = []
    for index, action in enumerate(policy_actions.detach().cpu()):
        reverse_key = dataset.action_tensor_to_reverse_key(action)
        rank = dataset.evaluate_reverse_key(reverse_key)["rank"]
        rows.append({
            # nomination_rank 的排序依据是最终策略概率，而不是预测奖励或真实 rank。
            "nomination_rank": index + 1,
            "reverse_key": reverse_key,
            "foldx_mutations_from_start": dataset.foldx_mutations_from_target(reverse_key),
            "landscape_rank": rank,
            "policy_log_probability": float(policy_log_probabilities[index].item()),
            "center_score": float(center[index].item()),
            "std_score": float(std[index].item()),
            "acquisition": float(acquisition[index].item()),
            "epoch": int(final_epoch),
            "source": "last_completed_epoch_policy_topk",
        })
    output_path = os.path.join(loss_history.save_path, "final_top8_candidates.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    # 真实 rank 仅附加在最终候选的离线评测中；候选身份在上一段已经确定。
    final_metrics = rank_metrics(dataset, [row["reverse_key"] for row in rows], f"final_top{args.policy_eval_top_k}")
    final_metrics.update({
        "start_reverse_key": dataset.start_reverse_key,
        "start_rank": dataset.evaluate_reverse_key(dataset.start_reverse_key)["rank"],
        "final_selection_source": "last_completed_epoch_policy_topk",
        "final_policy_epoch": int(final_epoch),
        "final_policy_completed_epochs": int(final_epoch) + 1,
        "reward_cache_scored_candidate_count": len(candidate_cache),
        "final_candidate_csv": output_path,
    })
    pd.DataFrame([final_metrics]).to_csv(
        os.path.join(loss_history.save_path, "final_summary.csv"), index=False
    )
    return final_metrics


def main():
    args = get_args()
    checkpoint_args_path, checkpoint_keys = load_model_args_from_checkpoint(args)
    seed_all(args.seed)
    if args.is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")
    # 这里完成 raw 起点 -> fixed 起点的 Dataset 初始化；之后的 WT 输入均来自 fixed PDB。
    dataset, start_evo_csv = build_dataset(args, device)
    collate_fn = PaddingCollate()
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    loss_dir = os.path.join(args.log_root, f"{timestamp}_{args.run_name}", dataset.start_folder_id)
    history = LossHistory(loss_dir, is_evolve=True)
    writer = SummaryWriter(log_dir=history.save_path)
    history.write("v4: v3 PPO with on-demand FoldX structures from an ESMFold2 start.\n")
    history.write(f"Start CSV: {start_evo_csv}\n")
    history.write(f"Start reverse key: {dataset.start_reverse_key}\n")
    history.write("Actions are absolute immature/mature states; no CSV-PDB feasibility projection is used.\n")
    history.write("FoldX RepairPDB before MERF WT input and BuildModel: required\n")
    if checkpoint_args_path is not None:
        history.write(f"Loaded checkpoint args: {checkpoint_args_path}; keys={','.join(checkpoint_keys)}\n")

    # 三种模型角色：model 可训练策略；reference 冻结并提供 KL 锚点；
    # reward_models 冻结并为 rollout 候选提供优化信号（可为单模型或 ensemble）。
    model = MERF(args).to(device)
    load_pretrained(model, args.model_load_path, device)
    configure_trainable_params(model, args)
    optimizer = build_optimizer(model, args)
    reference = MERF(args).to(device)
    load_pretrained(reference, args.model_load_path, device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    # 未显式给出 reward_model_paths 时，get_args 已将其设为 model_load_path；
    # 这里仍会创建独立、冻结的 reward 副本，避免 PPO 过程中 reward 随策略参数一起变化。
    reward_models = []
    for path in [item.strip() for item in args.reward_model_paths.split(",") if item.strip()]:
        reward = MERF(args).to(device)
        load_pretrained(reward, path, device)
        reward.eval()
        for parameter in reward.parameters():
            parameter.requires_grad = False
        reward_models.append(reward)
    if not reward_models:
        raise ValueError("No reward models loaded")

    # cache 生命周期覆盖整个 run，但它仅服务 reward_topk 诊断，不能作为最终结果来源。
    candidate_cache = {}
    try:
        write_initial_metrics(args, model, dataset, collate_fn, reward_models, history, device, candidate_cache)
        reward_baseline = None
        # v4 正常情况下会完成 total_epochs；显式保留最后成功 epoch，
        # 使最终写入逻辑不依赖“预期应完成多少轮”的假设。
        final_epoch = None
        for epoch in range(args.total_epochs):
            reward_baseline, _ = train_one_epoch(
                args, model, reward_models, reference, optimizer, epoch, dataset, collate_fn,
                history, device, reward_baseline, candidate_cache, writer,
            )
            final_epoch = epoch
        if final_epoch is None:
            raise RuntimeError("No outer epoch completed; cannot write final policy top-k")
        final_metrics = write_final_candidates(
            args, model, dataset, collate_fn, reward_models, device,
            candidate_cache, history, final_epoch,
        )
        history.write(f"Final metrics: {final_metrics}\n")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
