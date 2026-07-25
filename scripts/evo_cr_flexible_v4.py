"""ESMFold2-started CR evolution with on-demand FoldX structures.

v4 keeps v3's full-backbone PPO, KL anchor, rollout reward, and optional
current-rollout elite loss.  Its data path is different: every binary action is
valid, and FoldX creates only the structures actually nominated by the policy.
"""

import argparse
import datetime
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
    
    parser.add_argument("--dataset", default="cr6261_h1")
    parser.add_argument("--start_evo_csv", default="/home/dataset-local/projects_dir/MERF/data/CR_evo_esmfold2/cr6261_h1_KA59N,FA75S,GA77S,VA79A,VA104L/cr6261_h1_KA59N,FA75S,GA77S,VA79A,VA104L_evo.csv")
    # parser.add_argument("--dataset", default="cr6261_h9")
    # parser.add_argument("--start_evo_csv", default="")
    # parser.add_argument("--dataset", default="cr9114_h1")
    # parser.add_argument("--start_evo_csv", default="")

    parser.add_argument("--start_reverse_mutations", "--start_mutant", default="")
    parser.add_argument("--evo_root", default=EVO_ROOT)
    parser.add_argument("--log_root", default="logs/evo_cr_flexible_v4")
    parser.add_argument("--run_name", default="v4")
    parser.add_argument("--policy_eval_top_k", type=int, default=8)
    parser.add_argument("--policy_eval_interval", type=int, default=5)
    parser.add_argument("--foldx_workers", type=int, default=8)
    parser.add_argument("--foldx_bin", default=DEFAULT_FOLDX_BIN)
    parser.add_argument(
        "--repair_start",
        type=str2bool,
        default=True,
        help="Must remain true: v4 always uses FoldX RepairPDB before MERF/FoldX mutation.",
    )

    # Checkpoint/model arguments: these defaults match v3/v8 and are overwritten
    # from args.pkl when the chosen checkpoint supplies architecture settings.
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--lr", type=float, default=1e-6)
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

    # v3 PPO settings.
    parser.add_argument("--total_epochs", type=int, default=200)
    parser.add_argument("--rollout_samples", type=int, default=8)
    parser.add_argument("--sample_temperature", type=float, default=1.5)
    parser.add_argument("--inner_epochs", type=int, default=1)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--normalize_advantage", type=str2bool, default=True)
    parser.add_argument("--baseline_momentum", type=float, default=0.8)
    parser.add_argument("--kl_coeff", type=float, default=0.02)
    parser.add_argument("--train_mode", choices=["bias_only", "finetune"], default="finetune")
    parser.add_argument("--elite_mode", choices=["none", "old", "hamming"], default="none")
    parser.add_argument("--elite_loss_weight", type=float, default=0.2)
    parser.add_argument("--elite_top_k", type=int, default=8)
    parser.add_argument("--elite_min_hamming", type=int, default=2)
    parser.add_argument("--entropy_reg_weight", type=float, default=0.0)
    args = parser.parse_args()

    args.dataset = normalize_dataset_name(args.dataset)
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
    for key in CHECKPOINT_MODEL_ARG_KEYS:
        if key in vars(checkpoint_args):
            setattr(args, key, vars(checkpoint_args)[key])
            loaded.append(key)
    print(f"Loaded architecture args from {args_path}: {', '.join(loaded)}")
    return args_path, loaded


def load_pretrained(model, path, device):
    state = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {path}; missing={missing}; unexpected={unexpected}")


def configure_trainable_params(model, args):
    """Keep v3's full-backbone finetuning; do not train a site-specific bias."""
    for name, parameter in model.named_parameters():
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
    if score_agg == "mean":
        center = score_stack.mean(dim=0)
    elif score_agg == "median":
        center = score_stack.median(dim=0).values
    else:
        raise ValueError(f"Unsupported score aggregation: {score_agg}")
    std = score_stack.std(dim=0, unbiased=False) if score_stack.size(0) > 1 else torch.zeros_like(center)
    return center, std


def make_acquisition(center_scores, std_scores, uncertainty_weight, reward_direction):
    if reward_direction == "min":
        return center_scores + uncertainty_weight * std_scores
    if reward_direction == "max":
        return -center_scores + uncertainty_weight * std_scores
    raise ValueError(f"Unsupported reward direction: {reward_direction}")


def transform_acquisition(acquisition, reward_transform, clip_mad):
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
    """Fold missing nominated structures once, then score the resulting batched pairs."""
    dataset.ensure_action_structures(actions)
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
    """Find exact top-k binary action states without pre-generating their structures."""
    actions = dataset.all_action_matrix.to(binary_log_probs.device)
    expanded = binary_log_probs.unsqueeze(0).expand(actions.size(0), -1, -1)
    scores = expanded.gather(2, actions.unsqueeze(-1)).squeeze(-1).sum(dim=1)
    indices = torch.topk(scores, k=min(top_k, actions.size(0))).indices
    return actions[indices].detach(), scores[indices].detach()


def rank_metrics(dataset, reverse_keys, prefix):
    ranks = []
    shown_ranks = []
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
        f"{prefix}_selected_keys": "|".join(reverse_keys),
        f"{prefix}_selected_ranks": "|".join(shown_ranks),
    }


def update_candidate_cache(cache, dataset, actions, center_scores, std_scores, acquisition, epoch, source):
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

    metadata = pd.read_csv(os.path.join(CR_DIR, "cr_evo.csv"), dtype={"pdb_id": "string"})
    rows = metadata[metadata["pdb_id"] == args.dataset]
    if rows.empty:
        raise ValueError(f"{args.dataset} not found in {CR_DIR}/cr_evo.csv")
    row = rows.iloc[0]
    start_dir = os.path.dirname(start_evo_csv)
    embedding_path = os.path.join(start_dir, "plm_embeddings", f"{os.path.basename(start_dir)}_esm2_650_embeddings.pkl")
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
    policy_batch = recursive_to(collate_fn([dataset.__getitem__(0)]), device)
    model.eval()
    with torch.no_grad():
        old_log_probs = model.get_binary_log_probs(
            policy_batch, dataset, device, temperature=args.sample_temperature
        )
        raw_actions = torch.distributions.Categorical(logits=old_log_probs).sample((args.rollout_samples,))
        old_action_log_probs = selected_action_log_probs(old_log_probs, raw_actions, reduce="sum")
        reference_log_probs = reference_model.get_binary_log_probs(policy_batch, dataset, device)

        unique_actions, inverse = torch.unique(raw_actions, dim=0, return_inverse=True)
        unique_center, unique_std, _ = score_action_batch(
            dataset, collate_fn, reward_models, unique_actions, device, args.score_agg
        )
        center_scores = unique_center[inverse]
        std_scores = unique_std[inverse]
        acquisition = make_acquisition(
            center_scores, std_scores, args.uncertainty_weight, args.reward_direction
        )
        train_acquisition = transform_acquisition(acquisition, args.reward_transform, args.reward_clip_mad)
        raw_policy_entropy = -(old_log_probs.exp() * old_log_probs).sum(dim=-1).mean()

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

    losses, policy_losses, kl_values, entropy_values, elite_values = [], [], [], [], []
    model.train()
    for _ in range(args.inner_epochs):
        optimizer.zero_grad()
        current_log_probs = model.get_binary_log_probs(
            policy_batch, dataset, device, temperature=args.sample_temperature
        )
        current_action_log_probs = selected_action_log_probs(current_log_probs, raw_actions, reduce="sum")
        ratio = torch.exp(current_action_log_probs - old_action_log_probs).clamp(max=20.0)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        kl_divergence = (
            reference_log_probs.exp() * (reference_log_probs - current_log_probs)
        ).sum(dim=-1).mean()
        entropy = -(current_log_probs.exp() * current_log_probs).sum(dim=-1).mean()
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

    # 当前策略的精确 top-k 不依赖预先存在的全 landscape PDB。为控制 FoldX
    # 成本，可只在固定间隔把这些诊断候选真正建模并加入全局提名缓存。
    model.eval()
    with torch.no_grad():
        evaluation_log_probs = model.get_binary_log_probs(policy_batch, dataset, device)
        policy_actions, _ = exact_policy_topk_actions(dataset, evaluation_log_probs, args.policy_eval_top_k)
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


def write_final_candidates(args, dataset, candidate_cache, loss_history):
    _, candidates = top_cache_metrics(candidate_cache, dataset, args.policy_eval_top_k)
    rows = []
    for nomination_rank, candidate in enumerate(candidates, start=1):
        rank = dataset.evaluate_reverse_key(candidate["reverse_key"])["rank"]
        rows.append({
            "nomination_rank": nomination_rank,
            "reverse_key": candidate["reverse_key"],
            "foldx_mutations_from_start": candidate["foldx_mutations"],
            "landscape_rank": rank,
            **candidate,
        })
    output_path = os.path.join(loss_history.save_path, "final_top8_candidates.csv")
    pd.DataFrame(rows).to_csv(output_path, index=False)
    final_metrics = rank_metrics(dataset, [row["reverse_key"] for row in rows], f"final_top{args.policy_eval_top_k}")
    final_metrics.update({
        "start_reverse_key": dataset.start_reverse_key,
        "start_rank": dataset.evaluate_reverse_key(dataset.start_reverse_key)["rank"],
        "total_scored_candidates": len(candidate_cache),
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

    model = MERF(args).to(device)
    load_pretrained(model, args.model_load_path, device)
    configure_trainable_params(model, args)
    optimizer = build_optimizer(model, args)
    reference = MERF(args).to(device)
    load_pretrained(reference, args.model_load_path, device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
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

    candidate_cache = {}
    try:
        write_initial_metrics(args, model, dataset, collate_fn, reward_models, history, device, candidate_cache)
        reward_baseline = None
        for epoch in range(args.total_epochs):
            reward_baseline, _ = train_one_epoch(
                args, model, reward_models, reference, optimizer, epoch, dataset, collate_fn,
                history, device, reward_baseline, candidate_cache, writer,
            )
        final_metrics = write_final_candidates(args, dataset, candidate_cache, history)
        history.write(f"Final metrics: {final_metrics}\n")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
