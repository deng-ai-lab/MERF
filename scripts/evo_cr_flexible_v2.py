"""Flexible CR evolution v2 with projection-consistent PPO and reward-only replay."""

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
from model.MERF_evo_flexible_v2 import MERF
from protein.read_pdbs import PaddingCollate
from utils.losshistory import LossHistory
from utils.util import recursive_to, seed_all

from evo_cr_v8 import (
    build_optimizer,
    configure_trainable_params,
    evaluate_reward_topk_samples,
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
    parser.add_argument("--log_root", default="logs/evo_cr_flexible_v2")
    parser.add_argument("--archive_size", type=int, default=64)
    parser.add_argument("--archive_min_hamming", type=int, default=2)
    flexible_args, remaining_args = parser.parse_known_args()

    original_argv = sys.argv
    try:
        sys.argv = [sys.argv[0], *remaining_args]
        args = get_v8_args()
    finally:
        sys.argv = original_argv
    if flexible_args.archive_size <= 0:
        raise ValueError("--archive_size must be positive")
    args.dataset = normalize_dataset_name(flexible_args.dataset)
    args.start_reverse_mutations = flexible_args.start_reverse_mutations
    args.log_root = flexible_args.log_root
    args.archive_size = flexible_args.archive_size
    args.archive_min_hamming = flexible_args.archive_min_hamming
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


def update_reward_archive(old_actions, old_scores, new_actions, new_scores, args):
    """Keep diverse candidates selected solely by the frozen reward-model acquisition."""
    by_action = {}
    if old_actions is not None:
        for action, score in zip(old_actions.tolist(), old_scores):
            by_action[tuple(action)] = float(score)
    for action, score in zip(new_actions.detach().cpu().tolist(), new_scores.detach().cpu().tolist()):
        key = tuple(action)
        by_action[key] = min(by_action.get(key, float("inf")), float(score))

    # 先按 acquisition 选低分候选，再以 Hamming 距离限制 archive 的重复候选。
    ordered = sorted(by_action.items(), key=lambda item: item[1])
    selected = []
    for action, score in ordered:
        if all(sum(a != b for a, b in zip(action, chosen[0])) >= args.archive_min_hamming for chosen in selected):
            selected.append((action, score))
        if len(selected) == args.archive_size:
            break
    for action, score in ordered:
        if len(selected) == args.archive_size:
            break
        if (action, score) not in selected:
            selected.append((action, score))
    actions = torch.tensor([item[0] for item in selected], dtype=torch.long)
    scores = [item[1] for item in selected]
    return actions, scores


def get_binary_log_probs(model, policy_batch, dataset, device, temperature=1.0):
    return model.get_binary_log_probs(policy_batch, dataset, device, temperature=temperature)


def train_one_epoch(
    args, evo_model, reward_models, reference_model, optimizer, epoch, train_dataset,
    collate_fn, loss_history, device, reward_baseline, valid_ranks, archive_actions,
    archive_scores, writer=None,
):
    policy_batch = recursive_to(collate_fn([train_dataset.__getitem__(0)]), device)
    evo_model.eval()
    with torch.no_grad():
        old_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device, args.sample_temperature)
        raw_actions = torch.distributions.Categorical(logits=old_log_probs).sample((args.rollout_samples,))
        projected_actions, reverse_keys, projection_distances, _ = train_dataset.project_raw_actions_to_candidates(
            raw_actions, binary_log_probs=old_log_probs
        )
        # 奖励在投影后的可行 PDB 候选上计算，因此 PPO 比率也对同一个动作计算。
        old_projected_log_probs = selected_action_log_probs(old_log_probs, projected_actions, reduce="sum")
        policy_entropy = -(old_log_probs.exp() * old_log_probs).sum(dim=-1).mean()
        ref_log_probs = get_binary_log_probs(reference_model, policy_batch, train_dataset, device)
        mean_scores, std_scores, score_stack = score_projected_actions(
            train_dataset, collate_fn, reward_models, projected_actions.to(device), device, args.score_agg
        )
        acquisition = make_acquisition(mean_scores, std_scores, args.uncertainty_weight, args.reward_direction)
        train_acquisition = transform_acquisition(acquisition, args.reward_transform, args.reward_clip_mad)
        archive_actions, archive_scores = update_reward_archive(
            archive_actions, archive_scores, projected_actions, train_acquisition, args
        )

    batch_baseline = float(train_acquisition.mean().item())
    if reward_baseline is None:
        reward_baseline = batch_baseline
    advantages = reward_baseline - train_acquisition.detach()
    if args.normalize_advantage and advantages.numel() > 1:
        std = advantages.std(unbiased=False)
        if float(std.item()) > 1e-8:
            advantages = (advantages - advantages.mean()) / (std + 1e-8)

    evo_model.train()
    optimizer.zero_grad()
    curr_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device, args.sample_temperature)
    curr_projected_log_probs = selected_action_log_probs(curr_log_probs, projected_actions, reduce="sum")
    ratio = torch.exp(curr_projected_log_probs - old_projected_log_probs).clamp(max=20.0)
    clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
    policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
    kl_div = (ref_log_probs.exp() * (ref_log_probs - curr_log_probs)).sum(dim=-1).mean()
    curr_entropy = -(curr_log_probs.exp() * curr_log_probs).sum(dim=-1).mean()
    elite_count = min(args.elite_top_k, len(archive_scores))
    elite_actions = archive_actions[:elite_count].to(device)
    elite_loss = -selected_action_log_probs(curr_log_probs, elite_actions, reduce="mean").mean()
    loss = policy_loss + args.kl_coeff * kl_div + args.elite_loss_weight * elite_loss - args.entropy_reg_weight * curr_entropy
    loss.backward()
    optimizer.step()

    if args.baseline_momentum >= 0:
        reward_baseline = args.baseline_momentum * reward_baseline + (1.0 - args.baseline_momentum) * batch_baseline
    else:
        reward_baseline = batch_baseline

    sample_summary = summarize_candidates(train_dataset, reverse_keys)
    predicted_idx = int(torch.argmin(train_acquisition).item())
    predicted_key = reverse_keys[predicted_idx]
    predicted_info = train_dataset.evaluate_reverse_key(predicted_key)
    fallback_rank = max(train_dataset.rank_by_reverse_key.values()) + 1
    predicted_rank = fallback_rank if pd.isna(predicted_info["rank"]) else int(predicted_info["rank"])
    archive_forward_mutations = {
        mutation for mutation, action in zip(train_dataset.reverse_mutations, archive_actions[0].tolist())
        if action == 1
    }
    archive_key = train_dataset.forward_mutations_to_reverse_key(archive_forward_mutations)
    archive_info = train_dataset.evaluate_reverse_key(archive_key)
    archive_rank = fallback_rank if pd.isna(archive_info["rank"]) else int(archive_info["rank"])

    greedy_rank = np.nan
    greedy_key = ""
    topk_metrics = {"eval_top8_min_rank": np.nan, "eval_top8_mean_rank": np.nan, "eval_top8_best_true_score": np.nan,
                    "eval_top8_unique_candidates": np.nan, "eval_top8_selected_keys": "", "eval_top8_selected_ranks": ""}
    if epoch % args.eval_interval == 0:
        evo_model.eval()
        with torch.no_grad():
            greedy_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device)
            candidate_scores = train_dataset.candidate_log_probs(greedy_log_probs, reduce="mean")
            greedy_index = int(torch.argmax(candidate_scores).item())
            greedy_key = train_dataset.candidate_reverse_keys[greedy_index]
            topk_metrics = evaluate_reward_topk_samples(
                args, evo_model, policy_batch, train_dataset, collate_fn, reward_models, device
            )
        greedy_info = train_dataset.evaluate_reverse_key(greedy_key)
        greedy_rank = fallback_rank if pd.isna(greedy_info["rank"]) else int(greedy_info["rank"])
        valid_ranks.append(greedy_rank)
        evo_model.train()

    projection_cpu = projection_distances.detach().cpu().float()
    metrics = {
        "epoch": epoch, "run_name": args.run_name, "reward_label": args.reward_label,
        "reward_model_count": len(reward_models), "supervision": "reward_model_acquisition_only",
        "reward_baseline": reward_baseline, "pred_best_reverse_key": predicted_key,
        "pred_best_rank": predicted_rank, "train_sample_best_rank": sample_summary["best_rank"],
        "train_sample_unique_candidates": len(set(reverse_keys)), "raw_policy_entropy": float(policy_entropy),
        "projection_hamming_mean": float(projection_cpu.mean()),
        "projection_exact_match_rate": float((projection_cpu == 0).float().mean()),
        "archive_size": len(archive_scores), "archive_best_acquisition": archive_scores[0],
        "archive_best_reverse_key": archive_key, "archive_best_rank": archive_rank,
        "greedy_reverse_key": greedy_key, "greedy_rank": greedy_rank,
        "best_eval_rank": int(np.min(valid_ranks)) if valid_ranks else np.nan,
        "mean_eval_rank": float(np.mean(valid_ranks)) if valid_ranks else np.nan,
        "acquisition_min": float(acquisition.min()), "acquisition_mean": float(acquisition.mean()),
        "train_loss": float(loss), "train_policy_loss": float(policy_loss), "train_kl": float(kl_div),
        "elite_loss": float(elite_loss), "train_entropy": float(curr_entropy),
        **topk_metrics,
    }
    loss_history.write(
        f"Epoch{epoch}: pred_rank={predicted_rank}, archive_rank={archive_rank}, greedy_rank={greedy_rank}, "
        f"best_eval_rank={metrics['best_eval_rank']}, archive={len(archive_scores)}\n"
    )
    print(f"Epoch:{epoch + 1} PredRank:{predicted_rank} ArchiveRank:{archive_rank} BestRank:{metrics['best_eval_rank']}")
    if writer is not None:
        for key in ["pred_best_rank", "archive_best_rank", "greedy_rank", "best_eval_rank", "raw_policy_entropy", "train_loss"]:
            writer.add_scalar(key, metrics[key], epoch + 1)
        writer.flush()
    save_file = os.path.join(loss_history.save_path, train_dataset.pdb_id + ".csv")
    history = pd.read_csv(save_file) if os.path.exists(save_file) else pd.DataFrame()
    pd.concat([history, pd.DataFrame([metrics])], ignore_index=True).to_csv(save_file, index=False)
    return reward_baseline, valid_ranks, archive_actions, archive_scores, metrics


def main():
    args = get_args()
    checkpoint_args_path, checkpoint_keys = load_model_args_from_checkpoint(args)
    seed_all(args.seed)
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")
    dataset = build_dataset(args, device)
    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    loss_dir = os.path.join(args.log_root, f"{time_str}_{args.run_name}", dataset.pdb_id)
    history = LossHistory(loss_dir, is_evolve=True)
    writer = SummaryWriter(log_dir=history.save_path)
    history.write(str(args) + "\n")
    history.write("v2: projected-action PPO plus diverse reward-only archive replay.\n")
    history.write(f"Start reverse key: {dataset.start_reverse_key}\n")
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

    baseline, valid_ranks, archive_actions, archive_scores = None, [], None, []
    for epoch in range(args.total_epochs):
        baseline, valid_ranks, archive_actions, archive_scores, _ = train_one_epoch(
            args, model, reward_models, reference, optimizer, epoch, dataset, PaddingCollate(), history,
            device, baseline, valid_ranks, archive_actions, archive_scores, writer,
        )
    writer.close()


if __name__ == "__main__":
    main()
