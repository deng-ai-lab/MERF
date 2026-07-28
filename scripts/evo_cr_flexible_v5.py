"""v5: v4 evolution with a temperature-consistent KL anchor and optional early stop.

This file deliberately reuses v4's dataset, FoldX path, PPO loss, and candidate
cache.  It changes only two behaviours:
1. the frozen KL reference uses the same sampling temperature as the trainable
   policy, so the KL term compares like-for-like distributions;
2. it records stability of the reward-cache top-k and may stop only from model
   available signals (candidate identities and predicted acquisition), never
   from landscape rank or experimental score.
"""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

import evo_cr_flexible_v4 as v4


def get_args():
    """Parse v5-only controls first, then delegate all v4 arguments unchanged."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--early_stop_enabled", type=v4.str2bool, default=False)
    parser.add_argument("--early_stop_min_epochs", type=int, default=5)
    parser.add_argument("--early_stop_patience", type=int, default=2)
    parser.add_argument("--early_stop_min_jaccard", type=float, default=0.75)
    parser.add_argument("--early_stop_max_mean_acq_improvement", type=float, default=1e-4)
    v5_args, remaining = parser.parse_known_args()
    if v5_args.early_stop_min_epochs <= 0 or v5_args.early_stop_patience <= 0:
        raise ValueError("early-stop epoch and patience values must be positive")
    if not 0.0 <= v5_args.early_stop_min_jaccard <= 1.0:
        raise ValueError("--early_stop_min_jaccard must be in [0, 1]")
    if v5_args.early_stop_max_mean_acq_improvement < 0:
        raise ValueError("--early_stop_max_mean_acq_improvement must be non-negative")

    sys.argv = [sys.argv[0], *remaining]
    args = v4.get_args()
    for key, value in vars(v5_args).items():
        setattr(args, key, value)
    return args


def use_temperature_matched_reference(reference_model, sample_temperature):
    """Make v4's KL call use the rollout temperature for the frozen reference.

    v4 calls the reference without an explicit temperature while its trainable
    policy is evaluated at ``sample_temperature``.  The reference is used only
    by the KL term, so binding its method here does not alter reward scoring or
    final policy evaluation.
    """
    original = reference_model.get_binary_log_probs

    def matched_log_probs(policy_batch, dataset, device, temperature=1.0):
        return original(policy_batch, dataset, device, temperature=sample_temperature)

    reference_model.get_binary_log_probs = matched_log_probs


def cache_stability(previous_keys, previous_mean_acquisition, candidate_cache, dataset, top_k):
    """Return reward-cache stability without touching hidden landscape labels."""
    metrics, candidates = v4.top_cache_metrics(candidate_cache, dataset, top_k)
    keys = {candidate["reverse_key"] for candidate in candidates}
    mean_acquisition = metrics[f"reward_top{top_k}_mean_acquisition"]
    if previous_keys is None:
        return metrics, keys, mean_acquisition, np.nan, np.nan
    union = previous_keys | keys
    jaccard = len(previous_keys & keys) / len(union) if union else 1.0
    improvement = previous_mean_acquisition - mean_acquisition
    return metrics, keys, mean_acquisition, jaccard, improvement


def main():
    args = get_args()
    checkpoint_args_path, checkpoint_keys = v4.load_model_args_from_checkpoint(args)
    v4.seed_all(args.seed)
    if args.is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")
    dataset, start_evo_csv = v4.build_dataset(args, device)
    collate_fn = v4.PaddingCollate()
    timestamp = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    loss_dir = os.path.join(args.log_root, f"{timestamp}_{args.run_name}", dataset.start_folder_id)
    history = v4.LossHistory(loss_dir, is_evolve=True)
    writer = SummaryWriter(log_dir=history.save_path)
    history.write("v5: v4 PPO with a temperature-consistent KL reference and optional cache-stability stop.\n")
    history.write(f"Start CSV: {start_evo_csv}\n")
    history.write(
        "Early stop uses only reward-cache top-k identities/acquisition; landscape rank is diagnostic only.\n"
    )
    if checkpoint_args_path is not None:
        history.write(f"Loaded checkpoint args: {checkpoint_args_path}; keys={','.join(checkpoint_keys)}\n")

    model = v4.MERF(args).to(device)
    v4.load_pretrained(model, args.model_load_path, device)
    v4.configure_trainable_params(model, args)
    optimizer = v4.build_optimizer(model, args)
    reference = v4.MERF(args).to(device)
    v4.load_pretrained(reference, args.model_load_path, device)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad = False
    use_temperature_matched_reference(reference, args.sample_temperature)

    reward_models = []
    for path in [item.strip() for item in args.reward_model_paths.split(",") if item.strip()]:
        reward = v4.MERF(args).to(device)
        v4.load_pretrained(reward, path, device)
        reward.eval()
        for parameter in reward.parameters():
            parameter.requires_grad = False
        reward_models.append(reward)
    if not reward_models:
        raise ValueError("No reward models loaded")

    candidate_cache = {}
    stability_rows = []
    previous_keys, previous_mean_acquisition = None, None
    stable_rounds = 0
    try:
        v4.write_initial_metrics(args, model, dataset, collate_fn, reward_models, history, device, candidate_cache)
        _, previous_keys, previous_mean_acquisition, _, _ = cache_stability(
            None, None, candidate_cache, dataset, args.policy_eval_top_k
        )
        reward_baseline = None
        for epoch in range(args.total_epochs):
            reward_baseline, _ = v4.train_one_epoch(
                args, model, reward_models, reference, optimizer, epoch, dataset, collate_fn,
                history, device, reward_baseline, candidate_cache, writer,
            )
            _, keys, mean_acquisition, jaccard, improvement = cache_stability(
                previous_keys, previous_mean_acquisition, candidate_cache, dataset, args.policy_eval_top_k
            )
            eligible = epoch + 1 >= args.early_stop_min_epochs
            stable = bool(
                eligible
                and jaccard >= args.early_stop_min_jaccard
                and improvement <= args.early_stop_max_mean_acq_improvement
            )
            stable_rounds = stable_rounds + 1 if stable else 0
            should_stop = args.early_stop_enabled and stable_rounds >= args.early_stop_patience
            stability_rows.append({
                "epoch": epoch,
                "completed_epochs": epoch + 1,
                "reward_cache_topk_jaccard_vs_previous": jaccard,
                "reward_cache_mean_acquisition_improvement": improvement,
                "reward_cache_mean_acquisition": mean_acquisition,
                "eligible_for_early_stop": eligible,
                "stability_condition": stable,
                "consecutive_stable_rounds": stable_rounds,
                "early_stop_triggered": should_stop,
                "early_stop_uses_landscape_rank": False,
            })
            pd.DataFrame(stability_rows).to_csv(
                os.path.join(history.save_path, "v5_cache_stability.csv"), index=False
            )
            previous_keys, previous_mean_acquisition = keys, mean_acquisition
            if should_stop:
                history.write(f"Early stop after {epoch + 1} outer epochs.\n")
                break
        final_metrics = v4.write_final_candidates(args, dataset, candidate_cache, history)
        history.write(f"Final metrics: {final_metrics}\n")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
