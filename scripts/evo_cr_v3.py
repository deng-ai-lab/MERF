import argparse
import datetime
import os
import pickle
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_cr_evo_v2 import CREvoDataset
from model.MERF_evo_v3 import MERF
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


def get_evolution_args_v3():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--policy_bias_lr", type=float, default=5e-2)
    parser.add_argument("--is_cuda", type=str2bool, default=True)
    parser.add_argument("--gpu_idx", type=int, default=1)
    parser.add_argument("--num_works", type=int, default=8)

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

    parser.add_argument("--cr_row_idx", type=int, default=1)
    parser.add_argument("--total_epochs", type=int, default=200)
    parser.add_argument("--rollout_samples", type=int, default=64)
    parser.add_argument("--sample_temperature", type=float, default=1.5)
    parser.add_argument("--inner_epochs", type=int, default=3)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--normalize_advantage", type=str2bool, default=True)
    parser.add_argument("--baseline_momentum", type=float, default=0.8)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--kl_coeff", type=float, default=0.02)

    parser.add_argument("--reward_source", choices=["rank", "true_score", "model"], default="rank")
    parser.add_argument("--freeze_base_model", type=str2bool, default=True)
    parser.add_argument("--max_evo_sites", type=int, default=64)
    parser.add_argument("--max_evo_candidates", type=int, default=70000)
    parser.add_argument("--evo_bias_scale", type=float, default=1.0)
    parser.add_argument("--elite_bc_weight", type=float, default=0.2)
    parser.add_argument("--elite_candidate_weight", type=float, default=1.0)
    parser.add_argument("--elite_top_k", type=int, default=128)

    args = parser.parse_args()
    if args.rollout_samples <= 0:
        raise ValueError("rollout_samples must be positive")
    if args.sample_temperature <= 0:
        raise ValueError("sample_temperature must be positive")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    return args


def load_model_args_from_checkpoint(args):
    checkpoint_dir = os.path.dirname(os.path.abspath(args.model_load_path))
    args_path = os.path.join(checkpoint_dir, "args.pkl")
    if not os.path.exists(args_path):
        print(f"Checkpoint args not found at {args_path}; using evolution args for model init.")
        return None, []

    with open(args_path, "rb") as f:
        checkpoint_args = pickle.load(f)
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
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"Loaded {path}; missing={missing}; unexpected={unexpected}")


def set_base_trainable(model, trainable):
    for name, param in model.named_parameters():
        if name not in {"site_action_bias", "candidate_score_bias"}:
            param.requires_grad = trainable


def get_binary_log_probs(model, policy_batch, dataset, device, temperature=1.0):
    if hasattr(model, "get_binary_log_probs"):
        return model.get_binary_log_probs(policy_batch, dataset, device, temperature=temperature)
    raise TypeError("v3 evolution requires a model with get_binary_log_probs")


def selected_action_log_probs(binary_log_probs, selected_actions, reduce="sum"):
    if selected_actions.dim() == 1:
        selected_actions = selected_actions.unsqueeze(0)
    selected_actions = selected_actions.to(binary_log_probs.device)
    expanded_log_probs = binary_log_probs.unsqueeze(0).expand(selected_actions.size(0), -1, -1)
    selected = expanded_log_probs.gather(2, selected_actions.unsqueeze(-1)).squeeze(-1)
    if reduce == "sum":
        return selected.sum(dim=1)
    if reduce == "mean":
        return selected.mean(dim=1)
    raise ValueError(f"Unsupported reduce mode: {reduce}")


def candidate_log_probs(binary_log_probs, candidate_actions, reduce="mean"):
    candidate_actions = candidate_actions.to(binary_log_probs.device)
    expanded = binary_log_probs.unsqueeze(0).expand(candidate_actions.size(0), -1, -1)
    selected = expanded.gather(2, candidate_actions.unsqueeze(-1)).squeeze(-1)
    if reduce == "mean":
        return selected.mean(dim=1)
    if reduce == "sum":
        return selected.sum(dim=1)
    raise ValueError(f"Unsupported reduce mode: {reduce}")


def candidate_objectives(args, train_dataset, reverse_keys, reward_model, collate_fn, device):
    if args.reward_source == "rank":
        fallback = max(train_dataset.rank_by_reverse_key.values()) + 1
        return torch.tensor(
            [
                train_dataset.rank_by_reverse_key.get(reverse_key, fallback)
                for reverse_key in reverse_keys
            ],
            dtype=torch.float32,
            device=device,
        )

    if args.reward_source == "true_score":
        fallback = min(train_dataset.score_by_reverse_key.values()) - 1.0
        return torch.tensor(
            [
                -train_dataset.score_by_reverse_key.get(reverse_key, fallback)
                for reverse_key in reverse_keys
            ],
            dtype=torch.float32,
            device=device,
        )

    score_batches = [
        train_dataset.getitem_by_action_tensor(action.detach().cpu())
        for action in train_dataset.candidate_action_matrix[:0]
    ]
    del score_batches
    batches = [
        train_dataset.getitem_by_action_tensor(
            train_dataset.candidate_action_matrix[train_dataset.candidate_reverse_keys.index(reverse_key)]
        )
        for reverse_key in reverse_keys
    ]
    score_batch = recursive_to(collate_fn(batches), device)
    with torch.no_grad():
        model_scores, _, _, _ = reward_model(score_batch, device)
    return model_scores.view(-1).detach()


def summarize_candidates(train_dataset, reverse_keys):
    fallback_rank = max(train_dataset.rank_by_reverse_key.values()) + 1
    true_scores = []
    ranks = []
    mean_ranks = []
    for reverse_key in reverse_keys:
        info = train_dataset.evaluate_reverse_key(reverse_key)
        true_scores.append(info["true_score"])
        rank = info["rank"]
        ranks.append(rank)
        mean_ranks.append(fallback_rank if rank is None or pd.isna(rank) else rank)
    valid_scores = [score for score in true_scores if score is not None and not pd.isna(score)]
    valid_ranks = [rank for rank in ranks if rank is not None and not pd.isna(rank)]
    return {
        "best_true_score": float(np.max(valid_scores)) if valid_scores else np.nan,
        "best_rank": int(np.min(valid_ranks)) if valid_ranks else np.nan,
        "mean_rank": float(np.mean(mean_ranks)) if mean_ranks else np.nan,
    }


def get_elite_actions(train_dataset, top_k):
    ranked = sorted(train_dataset.rank_by_reverse_key.items(), key=lambda item: item[1])
    elite_keys = [key for key, _ in ranked[:top_k]]
    key_to_idx = {key: idx for idx, key in enumerate(train_dataset.candidate_reverse_keys)}
    elite_indices = [key_to_idx[key] for key in elite_keys if key in key_to_idx]
    if not elite_indices:
        raise ValueError(f"No elite candidates have existing PDB/action rows for {train_dataset.pdb_id}")
    return (
        train_dataset.candidate_action_matrix[elite_indices].long(),
        torch.tensor(elite_indices, dtype=torch.long),
        elite_keys[: len(elite_indices)],
    )


def train_one_epoch_v3(args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
                       train_dataset, collate_fn, loss_history, device, reward_baseline,
                       valid_ranks, elite_actions, elite_indices, writer=None):
    loss_history.write(f"\nEpoch{epoch}: Start CR v3 rank-aware evolution\n")

    policy_batch = recursive_to(collate_fn([train_dataset.__getitem__(0)]), device)

    evo_model.eval()
    with torch.no_grad():
        binary_log_probs = get_binary_log_probs(
            evo_model, policy_batch, train_dataset, device, temperature=args.sample_temperature
        )
        raw_action_dist = torch.distributions.Categorical(logits=binary_log_probs)
        raw_actions = raw_action_dist.sample((args.rollout_samples,))
        projected_actions, reverse_keys, projection_distances, _ = (
            train_dataset.project_raw_actions_to_candidates(raw_actions, binary_log_probs=binary_log_probs)
        )
        old_raw_log_probs = selected_action_log_probs(binary_log_probs, raw_actions, reduce="sum")
        policy_entropy = -(binary_log_probs.exp() * binary_log_probs).sum(dim=-1).mean()
        objectives = candidate_objectives(
            args, train_dataset, reverse_keys, reward_model, collate_fn, device
        )
        ref_log_probs = get_binary_log_probs(reference_model, policy_batch, train_dataset, device)
    evo_model.train()

    batch_baseline = float(objectives.mean().item())
    if reward_baseline is None:
        reward_baseline = batch_baseline
    advantage_values = reward_baseline - objectives
    if args.normalize_advantage and advantage_values.numel() > 1:
        advantage_std = advantage_values.std(unbiased=False)
        if float(advantage_std.item()) > 1e-8:
            advantage_values = (advantage_values - advantage_values.mean()) / (advantage_std + 1e-8)
    advantages = advantage_values.detach()

    epoch_losses = []
    policy_losses = []
    kl_values = []
    elite_losses = []
    elite_candidate_losses = []
    for inner_epoch in tqdm(range(args.inner_epochs), desc="CR v3 PPO Training"):
        optimizer.zero_grad()
        curr_log_probs = get_binary_log_probs(
            evo_model, policy_batch, train_dataset, device, temperature=args.sample_temperature
        )
        curr_raw_log_probs = selected_action_log_probs(curr_log_probs, raw_actions, reduce="sum")

        ratio = torch.exp(curr_raw_log_probs - old_raw_log_probs).clamp(max=20.0)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        kl_div = (ref_log_probs.exp() * (ref_log_probs - curr_log_probs)).sum(dim=-1).mean()
        elite_loss = -candidate_log_probs(curr_log_probs, elite_actions.to(device), reduce="mean").mean()
        candidate_scores = evo_model.get_candidate_log_probs(curr_log_probs, train_dataset)
        elite_idx = elite_indices.to(device)
        elite_candidate_loss = -(
            torch.logsumexp(candidate_scores[elite_idx], dim=0)
            - torch.logsumexp(candidate_scores, dim=0)
        )
        batch_loss = (
            policy_loss
            + args.kl_coeff * kl_div
            + args.elite_bc_weight * elite_loss
            + args.elite_candidate_weight * elite_candidate_loss
        )

        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(float(batch_loss.item()))
        policy_losses.append(float(policy_loss.item()))
        kl_values.append(float(kl_div.item()))
        elite_losses.append(float(elite_loss.item()))
        elite_candidate_losses.append(float(elite_candidate_loss.item()))
        loss_history.write(
            f"  Inner epoch {inner_epoch}: loss={batch_loss.item():.6f}, "
            f"policy_loss={policy_loss.item():.6f}, kl={kl_div.item():.6f}, "
            f"elite_loss={elite_loss.item():.6f}, "
            f"elite_candidate_loss={elite_candidate_loss.item():.6f}\n"
        )

    if args.baseline_momentum >= 0:
        reward_baseline = args.baseline_momentum * reward_baseline + (
            1.0 - args.baseline_momentum
        ) * batch_baseline
    else:
        reward_baseline = batch_baseline

    sample_summary = summarize_candidates(train_dataset, reverse_keys)

    greedy_reverse_key = None
    greedy_true_score = np.nan
    greedy_rank = np.nan
    greedy_policy_score = np.nan
    greedy_forward_info = None
    if epoch % args.eval_interval == 0:
        evo_model.eval()
        with torch.no_grad():
            greedy_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device)
            candidate_scores = evo_model.get_candidate_log_probs(greedy_log_probs, train_dataset)
            greedy_idx = int(torch.argmax(candidate_scores).item())
            greedy_reverse_key = train_dataset.candidate_reverse_keys[greedy_idx]
            greedy_policy_score_tensor = candidate_scores[greedy_idx]
        evo_model.train()
        greedy_eval_info = train_dataset.evaluate_reverse_key(greedy_reverse_key)
        greedy_true_score = greedy_eval_info["true_score"]
        greedy_rank = greedy_eval_info["rank"]
        if greedy_rank is None:
            greedy_rank = max(train_dataset.rank_by_reverse_key.values()) + 1
        greedy_policy_score = float(greedy_policy_score_tensor.item())
        _, greedy_forward_info = train_dataset.reverse_key_to_forward_info(greedy_reverse_key)
        valid_ranks.append(greedy_rank)

    mean_rank = float(np.mean(valid_ranks)) if valid_ranks else np.nan
    best_rank = int(np.min(valid_ranks)) if valid_ranks else np.nan
    projection_distances_cpu = projection_distances.detach().cpu().float()

    metrics = {
        "epoch": epoch,
        "reward_source": args.reward_source,
        "reward_baseline": reward_baseline,
        "objective_mean": float(objectives.mean().item()),
        "objective_min": float(objectives.min().item()),
        "objective_max": float(objectives.max().item()),
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
        "train_sample_best_true_score": sample_summary["best_true_score"],
        "train_sample_best_rank": sample_summary["best_rank"],
        "train_sample_mean_rank": sample_summary["mean_rank"],
        "train_sample_unique_candidates": len(set(reverse_keys)),
        "projection_hamming_mean": float(projection_distances_cpu.mean().item()),
        "projection_hamming_max": float(projection_distances_cpu.max().item()),
        "projection_exact_match_rate": float((projection_distances_cpu == 0).float().mean().item()),
        "raw_policy_entropy": float(policy_entropy.item()),
        "greedy_reverse_key": greedy_reverse_key,
        "greedy_forward_mutations": greedy_forward_info,
        "greedy_true_score": greedy_true_score,
        "greedy_rank": greedy_rank,
        "greedy_policy_score": greedy_policy_score,
        "mean_eval_rank": mean_rank,
        "best_eval_rank": best_rank,
        "train_loss": float(np.mean(epoch_losses)),
        "train_policy_loss": float(np.mean(policy_losses)),
        "train_kl": float(np.mean(kl_values)),
        "elite_loss": float(np.mean(elite_losses)),
        "elite_candidate_loss": float(np.mean(elite_candidate_losses)),
    }

    loss_history.write(
        f"Epoch{epoch}: sample_best_rank={metrics['train_sample_best_rank']}, "
        f"sample_mean_rank={metrics['train_sample_mean_rank']}, "
        f"greedy_rank={greedy_rank}, best_eval_rank={best_rank}, "
        f"entropy={metrics['raw_policy_entropy']:.6f}, loss={metrics['train_loss']:.6f}\n"
    )
    print(
        "Epoch:%d/%d Sample Best Rank:%s Greedy Rank:%s Best Eval Rank:%s Loss:%.6f"
        % (
            epoch + 1,
            total_epoch,
            str(metrics["train_sample_best_rank"]),
            str(greedy_rank),
            str(best_rank),
            metrics["train_loss"],
        )
    )

    if writer is not None:
        for key in [
            "train_sample_best_rank",
            "train_sample_mean_rank",
            "train_sample_unique_candidates",
            "raw_policy_entropy",
            "greedy_rank",
            "mean_eval_rank",
            "best_eval_rank",
            "train_loss",
            "train_kl",
            "elite_loss",
            "elite_candidate_loss",
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


if __name__ == "__main__":
    args = get_evolution_args_v3()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)

    seed_all(args.seed)
    print(f"setting random seed...{args.seed}")
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")

    loss_dir = "logs/evo_cr_v3/"
    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    loss_dir = os.path.join(loss_dir, time_str)
    print(f"Loss and model checkpoints will be saved to {loss_dir}")

    cr_dir = "/home/dataset-local/projects_dir/MERF/data/CR"
    train_df = pd.read_csv(os.path.join(cr_dir, "cr_evo.csv"), dtype={"pdb_id": "string"})
    if args.cr_row_idx < 0 or args.cr_row_idx >= len(train_df):
        raise IndexError(f"cr_row_idx {args.cr_row_idx} out of range [0, {len(train_df) - 1}]")

    i = args.cr_row_idx
    pdb_id = train_df["pdb_id"].iloc[i].replace("+", "").replace(".00", "")
    antibody_chain = train_df["antibody_chain"].iloc[i]
    partner = train_df["partner"].iloc[i]
    sequence = train_df["sequence"].iloc[i]
    cdrs = [train_df["cdr1"].iloc[i], train_df["cdr2"].iloc[i], train_df["cdr3"].iloc[i]]
    cdr = cdrs[2]

    print(f"Start evolving row {i}: {pdb_id}...")
    print(f"Antibody chain: {antibody_chain}, Partner: {partner}, CDRH3: {cdr}")

    pdb_dir = os.path.join(cr_dir, "PDBs")
    fixed_dir = os.path.join(cr_dir, "PDBs_fixed")
    mutated_dir = os.path.join(cr_dir, "PDBs_mutated")
    plm_embedding_path = os.path.join(cr_dir, f"{pdb_id}_immature_esm2_650_embeddings.pkl")
    train_dataset = CREvoDataset(
        pdb_id,
        antibody_chain,
        partner,
        sequence,
        cdr,
        cr_dir,
        pdb_dir,
        fixed_dir,
        mutated_dir,
        knn_num=args.knn_neighbors_num,
        knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding,
        plm_path=args.plm_path,
        plm_embedding_path=plm_embedding_path,
        device=device,
    )
    collate_fn = PaddingCollate()

    this_loss_dir = os.path.join(loss_dir, pdb_id)
    loss_history = LossHistory(this_loss_dir, is_evolve=True)
    writer = SummaryWriter(log_dir=loss_history.save_path)
    loss_history.write(str(args) + "\n")
    if checkpoint_args_path is not None:
        loss_history.write(f"Loaded model args from {checkpoint_args_path}\n")
        loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')
    loss_history.write(f"CR row index: {i}\n")
    loss_history.write(f"CR candidates: {len(train_dataset.candidate_reverse_keys)}\n")
    loss_history.write(f"CR ranked candidates: {len(train_dataset.rank_by_reverse_key)}\n")
    loss_history.write(
        f"CR v3 reward_source={args.reward_source}, elite_top_k={args.elite_top_k}, "
        f"elite_bc_weight={args.elite_bc_weight}, "
        f"elite_candidate_weight={args.elite_candidate_weight}, "
        f"freeze_base_model={args.freeze_base_model}\n"
    )

    evo_model = MERF(args).to(device)
    load_pretrained(evo_model, args.model_load_path, device)
    set_base_trainable(evo_model, not args.freeze_base_model)
    optimizer = optim.Adam(
        [
            {
                "params": [evo_model.site_action_bias, evo_model.candidate_score_bias],
                "lr": args.policy_bias_lr,
            },
            {
                "params": [
                    p
                    for n, p in evo_model.named_parameters()
                    if n not in {"site_action_bias", "candidate_score_bias"} and p.requires_grad
                ],
                "lr": args.lr,
            },
        ],
        weight_decay=1e-4,
    )

    reference_model = MERF(args).to(device)
    load_pretrained(reference_model, args.model_load_path, device)
    reference_model.eval()
    for param in reference_model.parameters():
        param.requires_grad = False

    reward_model = MERF(args).to(device)
    load_pretrained(reward_model, args.model_load_path, device)
    reward_model.eval()
    for param in reward_model.parameters():
        param.requires_grad = False

    elite_actions, elite_indices, elite_keys = get_elite_actions(train_dataset, args.elite_top_k)
    loss_history.write(f"Elite action rows: {elite_actions.size(0)}\n")
    loss_history.write(f"Best elite key: {elite_keys[0] if elite_keys else None}\n")

    reward_baseline = None
    valid_ranks = []
    for epoch in range(args.total_epochs):
        reward_baseline, valid_ranks, _ = train_one_epoch_v3(
            args,
            evo_model,
            reward_model,
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
            elite_actions,
            elite_indices,
            writer=writer,
        )

    writer.close()
