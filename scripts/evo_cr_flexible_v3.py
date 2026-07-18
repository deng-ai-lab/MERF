"""Flexible CR evolution v3: projection-consistent PPO with deterministic policy diagnostics.

This version deliberately has no ``site_action_bias``.  With ``--train_mode finetune``
the only trainable parameters are the original MERF backbone parameters used by v1.
"""

import argparse
import datetime
import os
import sys

import numpy as np
import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_cr_evo_flexible_v1 import FlexibleCREvoDataset, canonical_reverse_mutation_key
from model.MERF_evo_flexible_v3 import MERF
from protein.read_pdbs import PaddingCollate
from utils.losshistory import LossHistory
from utils.util import recursive_to, seed_all

from evo_cr_v8 import (
    build_optimizer,
    choose_elites,
    configure_trainable_params,
    get_args as get_v8_args,
    load_model_args_from_checkpoint,
    load_pretrained,
    make_acquisition,
    score_projected_actions,
    selected_action_log_probs,
    summarize_candidates,
    transform_acquisition,
)


CR_DIR = "/home/dataset-local/projects_dir/MERF/data/CR"


def normalize_dataset_name(dataset):
    value = dataset.strip()
    parts = value.split("_")
    if len(parts) == 3 and parts[0] == "cr" and parts[1].isdigit() and parts[2].startswith("h"):
        return f"cr{parts[1]}_{parts[2]}"
    if len(parts) == 2 and parts[0].startswith("cr") and parts[0][2:].isdigit() and parts[1].startswith("h"):
        return value
    raise ValueError("--dataset must look like cr_6261_h9 or cr6261_h9")


def get_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--dataset", default="cr6261_h9")
    parser.add_argument("--start_reverse_mutations", "--start_mutant", dest="start_reverse_mutations", default="")
    parser.add_argument("--log_root", default="logs/evo_cr_flexible_v3")
    parser.add_argument("--policy_eval_top_k", type=int, default=8)
    flexible_args, remaining_args = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining_args]
        args = get_v8_args()
    finally:
        sys.argv = original_argv
    if flexible_args.policy_eval_top_k <= 0:
        raise ValueError("--policy_eval_top_k must be positive")
    args.dataset = normalize_dataset_name(flexible_args.dataset)
    args.start_reverse_mutations = flexible_args.start_reverse_mutations
    args.log_root = flexible_args.log_root
    args.policy_eval_top_k = flexible_args.policy_eval_top_k
    return args


def build_dataset(args, device):
    metadata = pd.read_csv(os.path.join(CR_DIR, "cr_evo.csv"), dtype={"pdb_id": "string"})
    rows = metadata[metadata["pdb_id"] == args.dataset]
    if rows.empty:
        raise ValueError(f"{args.dataset} not found in {CR_DIR}/cr_evo.csv")
    row = rows.iloc[0]
    start_key = canonical_reverse_mutation_key(args.start_reverse_mutations)
    variant_id = args.dataset if start_key == "" else f"{args.dataset}_{start_key}"
    embedding_path = os.path.join(CR_DIR, "plm_embeddings", f"{variant_id}_esm2_650_embeddings.pkl")
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
        plm_embedding_path=embedding_path,
        device=device,
    )


def rank_or_fallback(dataset, reverse_key):
    fallback = max(dataset.rank_by_reverse_key.values()) + 1
    rank = dataset.evaluate_reverse_key(reverse_key)["rank"]
    return fallback if rank is None or pd.isna(rank) else int(rank)


def policy_candidate_metrics(dataset, binary_log_probs, top_k):
    """Evaluate the policy over every PDB-backed candidate, without random sampling."""
    scores = dataset.candidate_log_probs(binary_log_probs, reduce="mean")
    indices = torch.argsort(scores, descending=True)[:top_k].detach().cpu().tolist()
    keys = [dataset.candidate_reverse_keys[index] for index in indices]
    ranks = [rank_or_fallback(dataset, key) for key in keys]
    return {
        "policy_greedy_reverse_key": keys[0],
        "policy_greedy_rank": ranks[0],
        "policy_top8_min_rank": int(np.min(ranks)),
        "policy_top8_mean_rank": float(np.mean(ranks)),
        "policy_top8_selected_keys": "|".join(keys),
        "policy_top8_selected_ranks": "|".join(str(rank) for rank in ranks),
    }


def train_one_epoch(
    args, model, reward_models, reference_model, optimizer, epoch, dataset, collate_fn,
    loss_history, device, reward_baseline, writer=None,
):
    policy_batch = recursive_to(collate_fn([dataset.__getitem__(0)]), device)
    model.eval()
    with torch.no_grad():
        old_log_probs = model.get_binary_log_probs(
            policy_batch, dataset, device, temperature=args.sample_temperature
        )
        raw_actions = torch.distributions.Categorical(logits=old_log_probs).sample((args.rollout_samples,))
        projected_actions, reverse_keys, projection_distances, _ = dataset.project_raw_actions_to_candidates(
            raw_actions, binary_log_probs=old_log_probs
        )
        # 奖励对应投影后的合法结构；PPO 的旧策略概率也对同一动作计算。
        old_projected_log_probs = selected_action_log_probs(old_log_probs, projected_actions, reduce="sum")
        reference_log_probs = reference_model.get_binary_log_probs(policy_batch, dataset, device)
        mean_scores, std_scores, _ = score_projected_actions(
            dataset, collate_fn, reward_models, projected_actions.to(device), device, args.score_agg
        )
        acquisition = make_acquisition(mean_scores, std_scores, args.uncertainty_weight, args.reward_direction)
        train_acquisition = transform_acquisition(acquisition, args.reward_transform, args.reward_clip_mad)
        raw_policy_entropy = -(old_log_probs.exp() * old_log_probs).sum(dim=-1).mean()

    batch_baseline = float(train_acquisition.mean().item())
    if reward_baseline is None:
        reward_baseline = batch_baseline
    advantages = (reward_baseline - train_acquisition.detach())
    if args.normalize_advantage and advantages.numel() > 1:
        std = advantages.std(unbiased=False)
        if float(std.item()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)

    # 与 v1 设想一致的 elite 学习：仅以当前 rollout 的冻结 reward 选优，真实 rank 不参与。
    elite_actions = choose_elites(
        train_acquisition, projected_actions.to(device), args.elite_top_k,
        args.elite_mode, args.elite_min_hamming,
    )
    epoch_losses, policy_losses, kl_values, entropy_values, elite_values = [], [], [], [], []
    model.train()
    for _ in range(args.inner_epochs):
        optimizer.zero_grad()
        current_log_probs = model.get_binary_log_probs(
            policy_batch, dataset, device, temperature=args.sample_temperature
        )
        current_projected_log_probs = selected_action_log_probs(
            current_log_probs, projected_actions, reduce="sum"
        )
        ratio = torch.exp(current_projected_log_probs - old_projected_log_probs).clamp(max=20.0)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        kl_div = (reference_log_probs.exp() * (reference_log_probs - current_log_probs)).sum(dim=-1).mean()
        entropy = -(current_log_probs.exp() * current_log_probs).sum(dim=-1).mean()
        if elite_actions is not None and args.elite_loss_weight > 0:
            elite_loss = -selected_action_log_probs(current_log_probs, elite_actions, reduce="mean").mean()
        else:
            elite_loss = torch.zeros((), device=device)
        loss = policy_loss + args.kl_coeff * kl_div + args.elite_loss_weight * elite_loss - args.entropy_reg_weight * entropy
        loss.backward()
        optimizer.step()
        epoch_losses.append(float(loss.item()))
        policy_losses.append(float(policy_loss.item()))
        kl_values.append(float(kl_div.item()))
        entropy_values.append(float(entropy.item()))
        elite_values.append(float(elite_loss.item()))

    if args.baseline_momentum >= 0:
        reward_baseline = args.baseline_momentum * reward_baseline + (1.0 - args.baseline_momentum) * batch_baseline
    else:
        reward_baseline = batch_baseline

    model.eval()
    with torch.no_grad():
        evaluation_log_probs = model.get_binary_log_probs(policy_batch, dataset, device)
        policy_metrics = policy_candidate_metrics(dataset, evaluation_log_probs, args.policy_eval_top_k)
    model.train()
    sample_summary = summarize_candidates(dataset, reverse_keys)
    best_sample_idx = int(torch.argmin(train_acquisition).item())
    best_sample_key = reverse_keys[best_sample_idx]
    projection_cpu = projection_distances.detach().cpu().float()
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
        "sample_best_reverse_key": best_sample_key,
        "sample_best_rank": rank_or_fallback(dataset, best_sample_key),
        "train_sample_best_rank": sample_summary["best_rank"],
        "train_sample_unique_candidates": len(set(reverse_keys)),
        "projection_hamming_mean": float(projection_cpu.mean().item()),
        "projection_exact_match_rate": float((projection_cpu == 0).float().mean().item()),
        "raw_policy_entropy": float(raw_policy_entropy.item()),
        "train_entropy": float(np.mean(entropy_values)),
        "train_loss": float(np.mean(epoch_losses)),
        "train_policy_loss": float(np.mean(policy_losses)),
        "train_kl": float(np.mean(kl_values)),
        "elite_loss": float(np.mean(elite_values)),
        **policy_metrics,
    }
    print(
        f"Epoch:{epoch + 1} SampleRank:{metrics['sample_best_rank']} "
        f"PolicyRank:{metrics['policy_greedy_rank']} Top8Min:{metrics['policy_top8_min_rank']} "
        f"Entropy:{metrics['raw_policy_entropy']:.4f}",
        flush=True,
    )
    loss_history.write(
        f"Epoch {epoch}: sample_rank={metrics['sample_best_rank']}, "
        f"policy_rank={metrics['policy_greedy_rank']}, top8_min={metrics['policy_top8_min_rank']}\n"
    )
    if writer is not None:
        for key in ["policy_greedy_rank", "policy_top8_min_rank", "policy_top8_mean_rank", "raw_policy_entropy", "train_loss"]:
            writer.add_scalar(key, metrics[key], epoch + 1)
        writer.flush()
    save_file = os.path.join(loss_history.save_path, dataset.pdb_id + ".csv")
    history = pd.read_csv(save_file) if os.path.exists(save_file) else pd.DataFrame()
    pd.concat([history, pd.DataFrame([metrics])], ignore_index=True).to_csv(save_file, index=False)
    return reward_baseline, metrics


def write_initial_metrics(args, model, dataset, collate_fn, loss_history, device):
    policy_batch = recursive_to(collate_fn([dataset.__getitem__(0)]), device)
    model.eval()
    with torch.no_grad():
        log_probs = model.get_binary_log_probs(policy_batch, dataset, device)
        metrics = policy_candidate_metrics(dataset, log_probs, args.policy_eval_top_k)
        metrics["epoch"] = -1
        metrics["run_name"] = args.run_name
        metrics["metric_stage"] = "before_training"
        metrics["start_reverse_key"] = dataset.start_reverse_key
        metrics["start_rank"] = rank_or_fallback(dataset, dataset.start_reverse_key)
    save_file = os.path.join(loss_history.save_path, dataset.pdb_id + ".csv")
    pd.DataFrame([metrics]).to_csv(save_file, index=False)
    loss_history.write(
        f"Before training: start_rank={metrics['start_rank']}, "
        f"policy_rank={metrics['policy_greedy_rank']}, top8_min={metrics['policy_top8_min_rank']}\n"
    )


def main():
    args = get_args()
    checkpoint_args_path, checkpoint_keys = load_model_args_from_checkpoint(args)
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")
    dataset = build_dataset(args, device)
    collate_fn = PaddingCollate()
    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    loss_dir = os.path.join(args.log_root, f"{time_str}_{args.run_name}", dataset.pdb_id)
    history = LossHistory(loss_dir, is_evolve=True)
    writer = SummaryWriter(log_dir=history.save_path)
    history.write("v3: full-backbone, projected-action PPO, current-rollout reward elites, deterministic full-candidate evaluation.\n")
    history.write(f"Start reverse key: {dataset.start_reverse_key}\n")
    history.write(f"site_action_bias: absent; train_mode={args.train_mode}\n")
    if checkpoint_args_path is not None:
        history.write(f"Loaded model args from {checkpoint_args_path}; keys={','.join(checkpoint_keys)}\n")

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
    for path in [path for path in args.reward_model_paths.split(",") if path]:
        reward = MERF(args).to(device)
        load_pretrained(reward, path, device)
        reward.eval()
        for parameter in reward.parameters():
            parameter.requires_grad = False
        reward_models.append(reward)
    if not reward_models:
        raise ValueError("No reward model loaded")

    write_initial_metrics(args, model, dataset, collate_fn, history, device)
    reward_baseline = None
    for epoch in range(args.total_epochs):
        reward_baseline, _ = train_one_epoch(
            args, model, reward_models, reference, optimizer, epoch, dataset, collate_fn,
            history, device, reward_baseline, writer,
        )
    writer.close()


if __name__ == "__main__":
    main()
