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
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_cr_evo_v2 import CREvoDataset
from model.MERF_evo_v7 import MERF
from protein.read_pdbs import PaddingCollate
from utils.losshistory import LossHistory
from utils.util import recursive_to, seed_all


cpu_num = 32
torch.set_num_threads(cpu_num)
print(cpu_num)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--adapter_lr", type=float, default=3e-2)
    parser.add_argument("--is_cuda", type=str2bool, default=True)
    parser.add_argument("--gpu_idx", type=int, default=1)

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
    parser.add_argument("--reward_model_paths", type=str, default="")
    parser.add_argument("--reward_label", type=str, default="reward")
    parser.add_argument("--uncertainty_weight", type=float, default=0.5)
    parser.add_argument("--score_agg", choices=["mean", "median"], default="mean")
    parser.add_argument("--reward_transform", choices=["raw", "rank", "clip"], default="raw")
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

    parser.add_argument("--cr_row_idx", type=int, default=0)
    parser.add_argument("--run_name", type=str, default="v7")
    parser.add_argument("--total_epochs", type=int, default=120)
    parser.add_argument("--rollout_samples", type=int, default=24)
    parser.add_argument("--sample_temperature", type=float, default=1.5)
    parser.add_argument("--inner_epochs", type=int, default=1)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--normalize_advantage", type=str2bool, default=True)
    parser.add_argument("--baseline_momentum", type=float, default=0.8)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--eval_sample_count", type=int, default=128)
    parser.add_argument("--eval_top_k", type=int, default=8)
    parser.add_argument("--kl_coeff", type=float, default=0.02)

    parser.add_argument("--train_mode", choices=["bias_only", "finetune"], default="finetune")
    parser.add_argument("--use_site_bias", type=str2bool, default=True)
    parser.add_argument("--max_evo_sites", type=int, default=64)
    parser.add_argument("--evo_bias_scale", type=float, default=1.0)

    parser.add_argument("--elite_mode", choices=["none", "old", "hamming"], default="none")
    parser.add_argument("--elite_loss_weight", type=float, default=0.2)
    parser.add_argument("--elite_top_k", type=int, default=8)
    parser.add_argument("--elite_min_hamming", type=int, default=2)

    parser.add_argument("--entropy_reg_weight", type=float, default=0.0)
    parser.add_argument("--entropy_stop", type=str2bool, default=False)
    parser.add_argument("--entropy_stop_threshold", type=float, default=0.12)
    parser.add_argument("--entropy_stop_patience", type=int, default=5)
    parser.add_argument("--entropy_stop_min_epoch", type=int, default=10)

    args = parser.parse_args()
    if not args.reward_model_paths:
        args.reward_model_paths = args.model_load_path
    if args.rollout_samples <= 0:
        raise ValueError("rollout_samples must be positive")
    if args.sample_temperature <= 0:
        raise ValueError("sample_temperature must be positive")
    if args.elite_top_k <= 0:
        raise ValueError("elite_top_k must be positive")
    if args.eval_sample_count <= 0:
        raise ValueError("eval_sample_count must be positive")
    if args.eval_top_k <= 0:
        raise ValueError("eval_top_k must be positive")
    return args


def load_model_args_from_checkpoint(args):
    args_path = os.path.join(os.path.dirname(os.path.abspath(args.model_load_path)), "args.pkl")
    if not os.path.exists(args_path):
        print(f"Checkpoint args not found at {args_path}; using evolution args.")
        return None, []
    with open(args_path, "rb") as f:
        checkpoint_args = pickle.load(f)
    loaded_keys = []
    for key in CHECKPOINT_MODEL_ARG_KEYS:
        if key in vars(checkpoint_args):
            setattr(args, key, vars(checkpoint_args)[key])
            loaded_keys.append(key)
    print(f"Loaded model args from: {args_path}")
    print(f'Overrode model args: {", ".join(loaded_keys)}')
    return args_path, loaded_keys


def load_pretrained(model, path, device):
    state = torch.load(path, map_location=device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {path}; missing={missing}; unexpected={unexpected}")


def configure_trainable_params(model, args):
    for name, param in model.named_parameters():
        if name == "site_action_bias":
            param.requires_grad = args.use_site_bias
        else:
            param.requires_grad = args.train_mode == "finetune"


def build_optimizer(model, args):
    adapter_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == "site_action_bias":
            adapter_params.append(param)
        else:
            base_params.append(param)
    groups = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": args.adapter_lr})
    if base_params:
        groups.append({"params": base_params, "lr": args.lr})
    if not groups:
        raise ValueError("No trainable parameters")
    return optim.Adam(groups, weight_decay=1e-4)


def get_binary_log_probs(model, policy_batch, dataset, device, temperature=1.0):
    return model.get_binary_log_probs(policy_batch, dataset, device, temperature=temperature)


def selected_action_log_probs(binary_log_probs, selected_actions, reduce="sum"):
    if selected_actions.dim() == 1:
        selected_actions = selected_actions.unsqueeze(0)
    selected_actions = selected_actions.to(binary_log_probs.device)
    expanded = binary_log_probs.unsqueeze(0).expand(selected_actions.size(0), -1, -1)
    selected = expanded.gather(2, selected_actions.unsqueeze(-1)).squeeze(-1)
    if reduce == "sum":
        return selected.sum(dim=1)
    if reduce == "mean":
        return selected.mean(dim=1)
    raise ValueError(f"Unsupported reduce mode: {reduce}")


def aggregate_score_stack(score_stack, score_agg):
    if score_agg == "median":
        center_score = score_stack.median(dim=0).values
    elif score_agg == "mean":
        center_score = score_stack.mean(dim=0)
    else:
        raise ValueError(f"Unsupported score_agg: {score_agg}")
    std_score = score_stack.std(dim=0, unbiased=False) if score_stack.size(0) > 1 else torch.zeros_like(center_score)
    return center_score, std_score


def transform_acquisition(acquisition, reward_transform, clip_mad):
    if reward_transform == "raw":
        return acquisition
    if reward_transform == "rank":
        if acquisition.numel() <= 1:
            return torch.zeros_like(acquisition)
        order = torch.argsort(acquisition.detach())
        ranks = torch.empty_like(acquisition)
        ranks[order] = torch.arange(acquisition.numel(), device=acquisition.device, dtype=acquisition.dtype)
        return ranks / float(acquisition.numel() - 1)
    if reward_transform == "clip":
        median = acquisition.detach().median()
        mad = (acquisition.detach() - median).abs().median()
        scale = torch.clamp(1.4826 * mad, min=1e-6)
        return acquisition.clamp(median - clip_mad * scale, median + clip_mad * scale)
    raise ValueError(f"Unsupported reward_transform: {reward_transform}")


def score_projected_actions(train_dataset, collate_fn, reward_models, projected_actions, device, score_agg):
    score_batches = [
        train_dataset.getitem_by_action_tensor(projected_actions[i].detach().cpu())
        for i in range(projected_actions.size(0))
    ]
    score_batch = recursive_to(collate_fn(score_batches), device)
    model_scores = []
    with torch.no_grad():
        for reward_model in reward_models:
            scores, _, _, _ = reward_model(score_batch, device)
            model_scores.append(scores.view(-1))
    score_stack = torch.stack(model_scores, dim=0)
    center_score, std_score = aggregate_score_stack(score_stack, score_agg)
    return center_score, std_score, score_stack


def summarize_candidates(train_dataset, reverse_keys):
    fallback_rank = max(train_dataset.rank_by_reverse_key.values()) + 1
    true_scores = []
    ranks = []
    mean_ranks = []
    for reverse_key in reverse_keys:
        eval_info = train_dataset.evaluate_reverse_key(reverse_key)
        true_scores.append(eval_info["true_score"])
        rank = eval_info["rank"]
        ranks.append(rank)
        mean_ranks.append(fallback_rank if rank is None or pd.isna(rank) else rank)
    valid_scores = [score for score in true_scores if score is not None and not pd.isna(score)]
    valid_ranks = [rank for rank in ranks if rank is not None and not pd.isna(rank)]
    return {
        "best_true_score": float(np.max(valid_scores)) if valid_scores else np.nan,
        "best_rank": int(np.min(valid_ranks)) if valid_ranks else np.nan,
        "mean_rank": float(np.mean(mean_ranks)) if mean_ranks else np.nan,
    }


def choose_elites(acquisition, projected_actions, top_k, mode, min_hamming):
    k = min(top_k, acquisition.numel())
    sorted_indices = torch.argsort(acquisition.detach())
    if mode == "old":
        return projected_actions[sorted_indices[:k]]
    if mode != "hamming":
        return None

    selected = []
    for idx in sorted_indices.tolist():
        action = projected_actions[idx]
        if all(int((action != projected_actions[j]).sum().item()) >= min_hamming for j in selected):
            selected.append(idx)
        if len(selected) == k:
            break
    if len(selected) < k:
        for idx in sorted_indices.tolist():
            if idx not in selected:
                selected.append(idx)
            if len(selected) == k:
                break
    return projected_actions[torch.tensor(selected, dtype=torch.long, device=projected_actions.device)]


def evaluate_reward_topk_samples(args, evo_model, policy_batch, train_dataset, collate_fn, reward_models, device):
    fallback_rank = max(train_dataset.rank_by_reverse_key.values()) + 1
    with torch.no_grad():
        eval_log_probs = get_binary_log_probs(
            evo_model, policy_batch, train_dataset, device, temperature=args.sample_temperature
        )
        eval_dist = torch.distributions.Categorical(logits=eval_log_probs)
        eval_raw_actions = eval_dist.sample((args.eval_sample_count,))
        eval_projected_actions, eval_reverse_keys, _, _ = train_dataset.project_raw_actions_to_candidates(
            eval_raw_actions, binary_log_probs=eval_log_probs
        )
        eval_center, eval_std, _ = score_projected_actions(
            train_dataset,
            collate_fn,
            reward_models,
            eval_projected_actions.to(device),
            device,
            args.score_agg,
        )
        eval_acquisition = eval_center + args.uncertainty_weight * eval_std
        eval_select_score = transform_acquisition(eval_acquisition, args.reward_transform, args.reward_clip_mad)

    best_by_key = {}
    for idx, reverse_key in enumerate(eval_reverse_keys):
        select_score = float(eval_select_score[idx].item())
        raw_acquisition = float(eval_acquisition[idx].item())
        existing = best_by_key.get(reverse_key)
        if existing is None or select_score < existing["select_score"]:
            eval_info = train_dataset.evaluate_reverse_key(reverse_key)
            rank = eval_info["rank"]
            if rank is None or pd.isna(rank):
                rank = fallback_rank
            best_by_key[reverse_key] = {
                "select_score": select_score,
                "raw_acquisition": raw_acquisition,
                "rank": int(rank),
                "true_score": eval_info["true_score"],
            }

    selected = sorted(best_by_key.items(), key=lambda item: item[1]["select_score"])[: args.eval_top_k]
    ranks = [item[1]["rank"] for item in selected]
    true_scores = [
        item[1]["true_score"] for item in selected
        if item[1]["true_score"] is not None and not pd.isna(item[1]["true_score"])
    ]
    return {
        "eval_top8_min_rank": int(np.min(ranks)) if ranks else np.nan,
        "eval_top8_mean_rank": float(np.mean(ranks)) if ranks else np.nan,
        "eval_top8_best_true_score": float(np.max(true_scores)) if true_scores else np.nan,
        "eval_top8_unique_candidates": len(best_by_key),
        "eval_top8_selected_keys": "|".join(item[0] for item in selected),
        "eval_top8_selected_ranks": "|".join(str(rank) for rank in ranks),
    }


def train_one_epoch(args, evo_model, reward_models, reference_model, optimizer, epoch, total_epoch,
                    train_dataset, collate_fn, loss_history, device, reward_baseline, valid_ranks,
                    writer=None):
    policy_batch = recursive_to(collate_fn([train_dataset.__getitem__(0)]), device)

    evo_model.eval()
    with torch.no_grad():
        old_log_probs = get_binary_log_probs(
            evo_model, policy_batch, train_dataset, device, temperature=args.sample_temperature
        )
        raw_dist = torch.distributions.Categorical(logits=old_log_probs)
        raw_actions = raw_dist.sample((args.rollout_samples,))
        projected_actions, reverse_keys, projection_distances, _ = (
            train_dataset.project_raw_actions_to_candidates(raw_actions, binary_log_probs=old_log_probs)
        )
        old_raw_log_probs = selected_action_log_probs(old_log_probs, raw_actions, reduce="sum")
        policy_entropy = -(old_log_probs.exp() * old_log_probs).sum(dim=-1).mean()
        ref_log_probs = get_binary_log_probs(reference_model, policy_batch, train_dataset, device)
        mean_scores, std_scores, score_stack = score_projected_actions(
            train_dataset, collate_fn, reward_models, projected_actions.to(device), device, args.score_agg
        )
        acquisition = mean_scores + args.uncertainty_weight * std_scores
        train_acquisition = transform_acquisition(acquisition, args.reward_transform, args.reward_clip_mad)
    evo_model.train()

    batch_baseline = float(train_acquisition.mean().item())
    if reward_baseline is None:
        reward_baseline = batch_baseline
    advantage_values = reward_baseline - train_acquisition.detach()
    if args.normalize_advantage and advantage_values.numel() > 1:
        advantage_std = advantage_values.std(unbiased=False)
        if float(advantage_std.item()) > 1e-8:
            advantage_values = (advantage_values - advantage_values.mean()) / (advantage_std + 1e-8)
    advantages = advantage_values.detach()

    elite_actions = choose_elites(
        train_acquisition, projected_actions.to(device), args.elite_top_k, args.elite_mode, args.elite_min_hamming
    )

    epoch_losses = []
    policy_losses = []
    kl_values = []
    entropy_values = []
    elite_losses = []
    for inner_epoch in tqdm(range(args.inner_epochs), desc="CR v7 Training"):
        optimizer.zero_grad()
        curr_log_probs = get_binary_log_probs(
            evo_model, policy_batch, train_dataset, device, temperature=args.sample_temperature
        )
        curr_raw_log_probs = selected_action_log_probs(curr_log_probs, raw_actions, reduce="sum")
        ratio = torch.exp(curr_raw_log_probs - old_raw_log_probs).clamp(max=20.0)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        kl_div = (ref_log_probs.exp() * (ref_log_probs - curr_log_probs)).sum(dim=-1).mean()
        curr_entropy = -(curr_log_probs.exp() * curr_log_probs).sum(dim=-1).mean()

        if args.elite_mode != "none" and elite_actions is not None and args.elite_loss_weight > 0:
            elite_loss = -selected_action_log_probs(curr_log_probs, elite_actions, reduce="mean").mean()
        else:
            elite_loss = torch.tensor(0.0, device=device)

        batch_loss = (
            policy_loss
            + args.kl_coeff * kl_div
            + args.elite_loss_weight * elite_loss
            - args.entropy_reg_weight * curr_entropy
        )
        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(float(batch_loss.item()))
        policy_losses.append(float(policy_loss.item()))
        kl_values.append(float(kl_div.item()))
        entropy_values.append(float(curr_entropy.item()))
        elite_losses.append(float(elite_loss.item()))
        loss_history.write(
            f"  Inner epoch {inner_epoch}: loss={batch_loss.item():.6f}, "
            f"policy_loss={policy_loss.item():.6f}, kl={kl_div.item():.6f}, "
            f"entropy={curr_entropy.item():.6f}, elite_loss={elite_loss.item():.6f}\n"
        )

    if args.baseline_momentum >= 0:
        reward_baseline = args.baseline_momentum * reward_baseline + (
            1.0 - args.baseline_momentum
        ) * batch_baseline
    else:
        reward_baseline = batch_baseline

    sample_summary = summarize_candidates(train_dataset, reverse_keys)
    best_sample_idx = int(torch.argmin(train_acquisition.detach()).item())
    pred_best_reverse_key = reverse_keys[best_sample_idx]
    pred_best_eval_info = train_dataset.evaluate_reverse_key(pred_best_reverse_key)
    pred_best_rank = pred_best_eval_info["rank"]
    if pred_best_rank is None:
        pred_best_rank = max(train_dataset.rank_by_reverse_key.values()) + 1

    greedy_reverse_key = None
    greedy_rank = np.nan
    greedy_true_score = np.nan
    greedy_policy_score = np.nan
    greedy_acquisition = np.nan
    greedy_reward_mean = np.nan
    greedy_reward_std = np.nan
    greedy_forward_info = None
    topk_metrics = {
        "eval_top8_min_rank": np.nan,
        "eval_top8_mean_rank": np.nan,
        "eval_top8_best_true_score": np.nan,
        "eval_top8_unique_candidates": np.nan,
        "eval_top8_selected_keys": "",
        "eval_top8_selected_ranks": "",
    }
    if epoch % args.eval_interval == 0:
        evo_model.eval()
        with torch.no_grad():
            greedy_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device)
            candidate_scores = train_dataset.candidate_log_probs(greedy_log_probs, reduce="mean")
            greedy_idx = int(torch.argmax(candidate_scores).item())
            greedy_actions = train_dataset.candidate_action_matrix[greedy_idx].to(device)
            greedy_reverse_key = train_dataset.candidate_reverse_keys[greedy_idx]
            greedy_policy_score = float(candidate_scores[greedy_idx].item())
            greedy_mean, greedy_std, _ = score_projected_actions(
                train_dataset, collate_fn, reward_models, greedy_actions.unsqueeze(0), device, args.score_agg
            )
            greedy_reward_mean = float(greedy_mean[0].item())
            greedy_reward_std = float(greedy_std[0].item())
            greedy_acquisition = greedy_reward_mean + args.uncertainty_weight * greedy_reward_std
            topk_metrics = evaluate_reward_topk_samples(
                args, evo_model, policy_batch, train_dataset, collate_fn, reward_models, device
            )
        evo_model.train()
        greedy_eval_info = train_dataset.evaluate_reverse_key(greedy_reverse_key)
        greedy_true_score = greedy_eval_info["true_score"]
        greedy_rank = greedy_eval_info["rank"]
        if greedy_rank is None:
            greedy_rank = max(train_dataset.rank_by_reverse_key.values()) + 1
        _, greedy_forward_info = train_dataset.reverse_key_to_forward_info(greedy_reverse_key)
        valid_ranks.append(greedy_rank)

    mean_rank = float(np.mean(valid_ranks)) if valid_ranks else np.nan
    best_rank = int(np.min(valid_ranks)) if valid_ranks else np.nan
    projection_distances_cpu = projection_distances.detach().cpu().float()
    per_model_min = score_stack.min(dim=1).values.detach().cpu().tolist()

    metrics = {
        "epoch": epoch,
        "run_name": args.run_name,
        "reward_label": args.reward_label,
        "reward_model_count": len(reward_models),
        "train_mode": args.train_mode,
        "elite_mode": args.elite_mode,
        "entropy_reg_weight": args.entropy_reg_weight,
        "uncertainty_weight": args.uncertainty_weight,
        "score_agg": args.score_agg,
        "reward_transform": args.reward_transform,
        "reward_clip_mad": args.reward_clip_mad,
        "sample_temperature": args.sample_temperature,
        "rollout_samples": args.rollout_samples,
        "inner_epochs": args.inner_epochs,
        "kl_coeff": args.kl_coeff,
        "eval_sample_count": args.eval_sample_count,
        "eval_top_k": args.eval_top_k,
        "supervision": "reward_model_acquisition",
        "reward_baseline": reward_baseline,
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
        "reward_mean_min": float(mean_scores.min().item()),
        "reward_mean_mean": float(mean_scores.mean().item()),
        "reward_std_mean": float(std_scores.mean().item()),
        "acquisition_min": float(acquisition.min().item()),
        "acquisition_mean": float(acquisition.mean().item()),
        "train_acquisition_min": float(train_acquisition.min().item()),
        "train_acquisition_mean": float(train_acquisition.mean().item()),
        "per_model_min_scores": "|".join(f"{value:.6f}" for value in per_model_min),
        "pred_best_reverse_key": pred_best_reverse_key,
        "pred_best_rank": pred_best_rank,
        "pred_best_true_score": pred_best_eval_info["true_score"],
        "train_sample_best_rank": sample_summary["best_rank"],
        "train_sample_best_true_score": sample_summary["best_true_score"],
        "train_sample_mean_rank": sample_summary["mean_rank"],
        "train_sample_unique_candidates": len(set(reverse_keys)),
        "projection_hamming_mean": float(projection_distances_cpu.mean().item()),
        "projection_exact_match_rate": float((projection_distances_cpu == 0).float().mean().item()),
        "raw_policy_entropy": float(policy_entropy.item()),
        "train_entropy": float(np.mean(entropy_values)),
        "greedy_reverse_key": greedy_reverse_key,
        "greedy_forward_mutations": greedy_forward_info,
        "greedy_rank": greedy_rank,
        "greedy_true_score": greedy_true_score,
        "greedy_policy_score": greedy_policy_score,
        "greedy_reward_mean": greedy_reward_mean,
        "greedy_reward_std": greedy_reward_std,
        "greedy_acquisition": greedy_acquisition,
        "mean_eval_rank": mean_rank,
        "best_eval_rank": best_rank,
        **topk_metrics,
        "train_loss": float(np.mean(epoch_losses)),
        "train_policy_loss": float(np.mean(policy_losses)),
        "train_kl": float(np.mean(kl_values)),
        "elite_loss": float(np.mean(elite_losses)),
        "early_stop_reason": "",
    }

    loss_history.write(
        f"Epoch{epoch}: pred_best_rank={pred_best_rank}, sample_best_rank={sample_summary['best_rank']}, "
        f"greedy_rank={greedy_rank}, best_eval_rank={best_rank}, "
        f"eval_top8_min_rank={topk_metrics['eval_top8_min_rank']}, "
        f"eval_top8_mean_rank={topk_metrics['eval_top8_mean_rank']}, "
        f"acq_min={metrics['acquisition_min']:.6f}, entropy={metrics['raw_policy_entropy']:.6f}\n"
    )
    print(
        "Epoch:%d/%d PredRank:%s SampleRank:%s GreedyRank:%s BestRank:%s Top8Min:%s Top8Mean:%s Entropy:%.4f"
        % (
            epoch + 1,
            total_epoch,
            str(pred_best_rank),
            str(sample_summary["best_rank"]),
            str(greedy_rank),
            str(best_rank),
            str(topk_metrics["eval_top8_min_rank"]),
            str(topk_metrics["eval_top8_mean_rank"]),
            metrics["raw_policy_entropy"],
        )
    )

    if writer is not None:
        for key in [
            "pred_best_rank",
            "train_sample_best_rank",
            "train_sample_unique_candidates",
            "raw_policy_entropy",
            "greedy_rank",
            "best_eval_rank",
            "eval_top8_min_rank",
            "eval_top8_mean_rank",
            "acquisition_min",
            "reward_std_mean",
            "train_loss",
            "train_kl",
            "elite_loss",
        ]:
            writer.add_scalar(key, metrics[key], epoch + 1)
        writer.flush()

    save_file = os.path.join(loss_history.save_path, train_dataset.pdb_id + ".csv")
    new_row = pd.DataFrame(metrics, index=[0])
    if os.path.exists(save_file):
        save_df = pd.concat([pd.read_csv(save_file), new_row], ignore_index=True)
    else:
        save_df = new_row
    save_df.to_csv(save_file, index=False)
    return reward_baseline, valid_ranks, metrics


def build_dataset(args, device):
    cr_dir = "/home/dataset-local/projects_dir/MERF/data/CR"
    train_df = pd.read_csv(os.path.join(cr_dir, "cr_evo.csv"), dtype={"pdb_id": "string"})
    if args.cr_row_idx < 0 or args.cr_row_idx >= len(train_df):
        raise IndexError(f"cr_row_idx {args.cr_row_idx} out of range [0, {len(train_df) - 1}]")
    row = train_df.iloc[args.cr_row_idx]
    pdb_id = row["pdb_id"].replace("+", "").replace(".00", "")
    plm_embedding_path = os.path.join(cr_dir, f"{pdb_id}_immature_esm2_650_embeddings.pkl")
    dataset = CREvoDataset(
        pdb_id,
        row["antibody_chain"],
        row["partner"],
        row["sequence"],
        row["cdr3"],
        cr_dir,
        os.path.join(cr_dir, "PDBs"),
        os.path.join(cr_dir, "PDBs_fixed"),
        os.path.join(cr_dir, "PDBs_mutated"),
        knn_num=args.knn_neighbors_num,
        knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding,
        plm_path=args.plm_path,
        plm_embedding_path=plm_embedding_path,
        device=device,
    )
    return dataset


if __name__ == "__main__":
    args = get_args()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")

    train_dataset = build_dataset(args, device)
    collate_fn = PaddingCollate()

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    loss_dir = os.path.join("logs/evo_cr_v7", f"{time_str}_{args.run_name}", train_dataset.pdb_id)
    loss_history = LossHistory(loss_dir, is_evolve=True)
    writer = SummaryWriter(log_dir=loss_history.save_path)
    loss_history.write(str(args) + "\n")
    if checkpoint_args_path is not None:
        loss_history.write(f"Loaded model args from {checkpoint_args_path}\n")
        loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')
    loss_history.write(f"CR row index: {args.cr_row_idx}\n")
    loss_history.write(f"CR candidates: {len(train_dataset.candidate_reverse_keys)}\n")
    loss_history.write(f"CR ranked candidates only used for evaluation: {len(train_dataset.rank_by_reverse_key)}\n")
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
    for reward_path in [p for p in args.reward_model_paths.split(",") if p]:
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
