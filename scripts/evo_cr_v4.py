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
from model.MERF_evo_v4 import MERF
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


def get_evolution_args_v4():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--adapter_lr", type=float, default=5e-2)
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
    parser.add_argument("--run_name", type=str, default="v4")
    parser.add_argument("--total_epochs", type=int, default=200)
    parser.add_argument("--rollout_samples", type=int, default=64)
    parser.add_argument("--sample_temperature", type=float, default=1.5)
    parser.add_argument("--inner_epochs", type=int, default=3)
    parser.add_argument("--ppo_clip_eps", type=float, default=0.2)
    parser.add_argument("--normalize_advantage", type=str2bool, default=True)
    parser.add_argument("--baseline_momentum", type=float, default=0.8)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--kl_coeff", type=float, default=0.02)

    parser.add_argument("--freeze_base_model", type=str2bool, default=False)
    parser.add_argument("--use_site_bias", type=str2bool, default=False)
    parser.add_argument("--use_candidate_bias", type=str2bool, default=False)
    parser.add_argument("--max_evo_sites", type=int, default=64)
    parser.add_argument("--max_evo_candidates", type=int, default=70000)
    parser.add_argument("--evo_bias_scale", type=float, default=1.0)

    parser.add_argument("--elite_loss_weight", type=float, default=0.0)
    parser.add_argument("--candidate_elite_loss_weight", type=float, default=0.0)
    parser.add_argument("--elite_top_k", type=int, default=8)

    parser.add_argument("--entropy_stop", type=str2bool, default=False)
    parser.add_argument("--entropy_stop_threshold", type=float, default=0.12)
    parser.add_argument("--entropy_stop_patience", type=int, default=5)
    parser.add_argument("--entropy_stop_min_epoch", type=int, default=10)

    args = parser.parse_args()
    if args.rollout_samples <= 0:
        raise ValueError("rollout_samples must be positive")
    if args.sample_temperature <= 0:
        raise ValueError("sample_temperature must be positive")
    if args.eval_interval <= 0:
        raise ValueError("eval_interval must be positive")
    if args.elite_top_k <= 0:
        raise ValueError("elite_top_k must be positive")
    return args


def load_model_args_from_checkpoint(args):
    checkpoint_dir = os.path.dirname(os.path.abspath(args.model_load_path))
    args_path = os.path.join(checkpoint_dir, "args.pkl")
    if not os.path.exists(args_path):
        print(f"Checkpoint args not found at {args_path}; using evolution args for model init.")
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
        elif name == "candidate_score_bias":
            param.requires_grad = args.use_candidate_bias
        else:
            param.requires_grad = not args.freeze_base_model


def build_optimizer(model, args):
    adapter_params = []
    base_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name in {"site_action_bias", "candidate_score_bias"}:
            adapter_params.append(param)
        else:
            base_params.append(param)
    groups = []
    if adapter_params:
        groups.append({"params": adapter_params, "lr": args.adapter_lr})
    if base_params:
        groups.append({"params": base_params, "lr": args.lr})
    if not groups:
        raise ValueError("No trainable parameters for this v4 configuration")
    return optim.Adam(groups, weight_decay=1e-4)


def get_binary_log_probs(model, policy_batch, dataset, device, temperature=1.0):
    return model.get_binary_log_probs(policy_batch, dataset, device, temperature=temperature)


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


def score_projected_actions(train_dataset, collate_fn, reward_model, projected_actions, device):
    score_batches = [
        train_dataset.getitem_by_action_tensor(projected_actions[i].detach().cpu())
        for i in range(projected_actions.size(0))
    ]
    score_batch = recursive_to(collate_fn(score_batches), device)
    with torch.no_grad():
        model_scores, _, _, _ = reward_model(score_batch, device)
    return model_scores.view(-1)


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


def select_predicted_elites(model_scores, projected_actions, candidate_indices, top_k):
    k = min(top_k, model_scores.numel())
    elite_sample_idx = torch.topk(-model_scores.detach(), k=k).indices
    return projected_actions[elite_sample_idx], candidate_indices[elite_sample_idx]


def train_one_epoch_v4(args, evo_model, reward_model, reference_model, optimizer, epoch, total_epoch,
                       train_dataset, collate_fn, loss_history, device, reward_baseline,
                       valid_ranks, writer=None):
    loss_history.write(f"\nEpoch{epoch}: Start CR v4 predictor-supervised evolution\n")
    policy_batch = recursive_to(collate_fn([train_dataset.__getitem__(0)]), device)

    evo_model.eval()
    with torch.no_grad():
        old_log_probs = get_binary_log_probs(
            evo_model, policy_batch, train_dataset, device, temperature=args.sample_temperature
        )
        raw_action_dist = torch.distributions.Categorical(logits=old_log_probs)
        raw_actions = raw_action_dist.sample((args.rollout_samples,))
        projected_actions, reverse_keys, projection_distances, candidate_indices = (
            train_dataset.project_raw_actions_to_candidates(raw_actions, binary_log_probs=old_log_probs)
        )
        old_raw_log_probs = selected_action_log_probs(old_log_probs, raw_actions, reduce="sum")
        policy_entropy = -(old_log_probs.exp() * old_log_probs).sum(dim=-1).mean()
        ref_log_probs = get_binary_log_probs(reference_model, policy_batch, train_dataset, device)
        model_scores = score_projected_actions(
            train_dataset, collate_fn, reward_model, projected_actions.to(device), device
        )
    evo_model.train()

    batch_baseline = float(model_scores.mean().item())
    if reward_baseline is None:
        reward_baseline = batch_baseline
    advantage_values = reward_baseline - model_scores.detach()
    if args.normalize_advantage and advantage_values.numel() > 1:
        advantage_std = advantage_values.std(unbiased=False)
        if float(advantage_std.item()) > 1e-8:
            advantage_values = (advantage_values - advantage_values.mean()) / (advantage_std + 1e-8)
    advantages = advantage_values.detach()

    elite_actions, elite_candidate_indices = select_predicted_elites(
        model_scores, projected_actions.to(device), candidate_indices.to(device), args.elite_top_k
    )

    epoch_losses = []
    policy_losses = []
    kl_values = []
    elite_losses = []
    candidate_elite_losses = []
    for inner_epoch in tqdm(range(args.inner_epochs), desc="CR v4 Training"):
        optimizer.zero_grad()
        curr_log_probs = get_binary_log_probs(
            evo_model, policy_batch, train_dataset, device, temperature=args.sample_temperature
        )
        curr_raw_log_probs = selected_action_log_probs(curr_log_probs, raw_actions, reduce="sum")
        ratio = torch.exp(curr_raw_log_probs - old_raw_log_probs).clamp(max=20.0)
        clipped_ratio = torch.clamp(ratio, 1.0 - args.ppo_clip_eps, 1.0 + args.ppo_clip_eps)
        policy_loss = -torch.min(ratio * advantages, clipped_ratio * advantages).mean()
        kl_div = (ref_log_probs.exp() * (ref_log_probs - curr_log_probs)).sum(dim=-1).mean()

        if args.elite_loss_weight > 0:
            elite_loss = -selected_action_log_probs(curr_log_probs, elite_actions, reduce="mean").mean()
        else:
            elite_loss = torch.tensor(0.0, device=device)

        if args.candidate_elite_loss_weight > 0:
            candidate_scores = evo_model.get_candidate_scores(curr_log_probs, train_dataset)
            unique_elite_indices = torch.unique(elite_candidate_indices)
            candidate_elite_loss = -(
                torch.logsumexp(candidate_scores[unique_elite_indices], dim=0)
                - torch.logsumexp(candidate_scores, dim=0)
            )
        else:
            candidate_elite_loss = torch.tensor(0.0, device=device)

        batch_loss = (
            policy_loss
            + args.kl_coeff * kl_div
            + args.elite_loss_weight * elite_loss
            + args.candidate_elite_loss_weight * candidate_elite_loss
        )
        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(float(batch_loss.item()))
        policy_losses.append(float(policy_loss.item()))
        kl_values.append(float(kl_div.item()))
        elite_losses.append(float(elite_loss.item()))
        candidate_elite_losses.append(float(candidate_elite_loss.item()))
        loss_history.write(
            f"  Inner epoch {inner_epoch}: loss={batch_loss.item():.6f}, "
            f"policy_loss={policy_loss.item():.6f}, kl={kl_div.item():.6f}, "
            f"elite_loss={elite_loss.item():.6f}, "
            f"candidate_elite_loss={candidate_elite_loss.item():.6f}\n"
        )

    if args.baseline_momentum >= 0:
        reward_baseline = args.baseline_momentum * reward_baseline + (
            1.0 - args.baseline_momentum
        ) * batch_baseline
    else:
        reward_baseline = batch_baseline

    sample_summary = summarize_candidates(train_dataset, reverse_keys)
    best_sample_idx = int(torch.argmin(model_scores.detach()).item())
    pred_best_reverse_key = reverse_keys[best_sample_idx]
    pred_best_eval_info = train_dataset.evaluate_reverse_key(pred_best_reverse_key)
    pred_best_rank = pred_best_eval_info["rank"]
    if pred_best_rank is None:
        pred_best_rank = max(train_dataset.rank_by_reverse_key.values()) + 1

    greedy_reverse_key = None
    greedy_true_score = np.nan
    greedy_rank = np.nan
    greedy_policy_score = np.nan
    greedy_model_score = np.nan
    greedy_forward_info = None
    if epoch % args.eval_interval == 0:
        evo_model.eval()
        with torch.no_grad():
            greedy_log_probs = get_binary_log_probs(evo_model, policy_batch, train_dataset, device)
            candidate_scores = evo_model.get_candidate_scores(greedy_log_probs, train_dataset)
            greedy_idx = int(torch.argmax(candidate_scores).item())
            greedy_actions = train_dataset.candidate_action_matrix[greedy_idx].to(device)
            greedy_reverse_key = train_dataset.candidate_reverse_keys[greedy_idx]
            greedy_policy_score = float(candidate_scores[greedy_idx].item())
            greedy_scores = score_projected_actions(
                train_dataset, collate_fn, reward_model, greedy_actions.unsqueeze(0), device
            )
            greedy_model_score = float(greedy_scores[0].item())
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

    metrics = {
        "epoch": epoch,
        "run_name": args.run_name,
        "supervision": "predictor_model_score",
        "use_site_bias": args.use_site_bias,
        "use_candidate_bias": args.use_candidate_bias,
        "freeze_base_model": args.freeze_base_model,
        "elite_loss_weight": args.elite_loss_weight,
        "candidate_elite_loss_weight": args.candidate_elite_loss_weight,
        "reward_baseline": reward_baseline,
        "advantage_mean": float(advantages.mean().item()),
        "advantage_std": float(advantages.std(unbiased=False).item()) if advantages.numel() > 1 else 0.0,
        "predictor_score_mean": float(model_scores.mean().item()),
        "predictor_score_min": float(model_scores.min().item()),
        "predictor_score_max": float(model_scores.max().item()),
        "pred_best_reverse_key": pred_best_reverse_key,
        "pred_best_true_score": pred_best_eval_info["true_score"],
        "pred_best_rank": pred_best_rank,
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
        "greedy_model_score": greedy_model_score,
        "greedy_true_score": greedy_true_score,
        "greedy_rank": greedy_rank,
        "greedy_policy_score": greedy_policy_score,
        "mean_eval_rank": mean_rank,
        "best_eval_rank": best_rank,
        "train_loss": float(np.mean(epoch_losses)),
        "train_policy_loss": float(np.mean(policy_losses)),
        "train_kl": float(np.mean(kl_values)),
        "elite_loss": float(np.mean(elite_losses)),
        "candidate_elite_loss": float(np.mean(candidate_elite_losses)),
        "early_stop_reason": "",
    }

    loss_history.write(
        f"Epoch{epoch}: pred_best_rank={metrics['pred_best_rank']}, "
        f"sample_best_rank={metrics['train_sample_best_rank']}, "
        f"greedy_rank={greedy_rank}, best_eval_rank={best_rank}, "
        f"entropy={metrics['raw_policy_entropy']:.6f}, loss={metrics['train_loss']:.6f}\n"
    )
    print(
        "Epoch:%d/%d Pred Best Rank:%s Sample Best Rank:%s Greedy Rank:%s Best Eval Rank:%s Entropy:%.4f"
        % (
            epoch + 1,
            total_epoch,
            str(metrics["pred_best_rank"]),
            str(metrics["train_sample_best_rank"]),
            str(greedy_rank),
            str(best_rank),
            metrics["raw_policy_entropy"],
        )
    )

    if writer is not None:
        for key in [
            "predictor_score_min",
            "pred_best_rank",
            "train_sample_best_rank",
            "train_sample_unique_candidates",
            "raw_policy_entropy",
            "greedy_model_score",
            "greedy_rank",
            "best_eval_rank",
            "train_loss",
            "train_kl",
            "elite_loss",
            "candidate_elite_loss",
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
    args = get_evolution_args_v4()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)

    seed_all(args.seed)
    print(f"setting random seed...{args.seed}")
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    loss_dir = os.path.join("logs/evo_cr_v4", f"{time_str}_{args.run_name}")
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
    cdr = train_df["cdr3"].iloc[i]

    print(f"Start evolving row {i}: {pdb_id}...")
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
    loss_history.write(f"CR ranked candidates only used for evaluation: {len(train_dataset.rank_by_reverse_key)}\n")

    evo_model = MERF(args).to(device)
    load_pretrained(evo_model, args.model_load_path, device)
    configure_trainable_params(evo_model, args)
    optimizer = build_optimizer(evo_model, args)

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

    reward_baseline = None
    valid_ranks = []
    low_entropy_epochs = 0
    for epoch in range(args.total_epochs):
        reward_baseline, valid_ranks, metrics = train_one_epoch_v4(
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
