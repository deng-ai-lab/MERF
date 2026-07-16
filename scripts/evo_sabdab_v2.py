import argparse
import datetime
import os
import pickle
import sys
from itertools import combinations

import numpy as np
import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.dataset_evo import EvoBaseDataset
from model.MERF_v6 import MERF
from protein.read_pdbs import PaddingCollate
from utils.losshistory import LossHistory
from utils.util import recursive_to, seed_all


cpu_num = 32
torch.set_num_threads(cpu_num)
print(cpu_num)

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=172)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--is_cuda", type=str2bool, default=True)
    parser.add_argument("--gpu_idx", type=int, default=0)

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

    parser.add_argument("--dataset_idx", type=int, default=None)
    parser.add_argument("--comb_num", type=int, default=3)
    parser.add_argument("--total_epochs", type=int, default=3000)
    parser.add_argument("--inner_epochs", type=int, default=10)
    parser.add_argument("--inner_batch_size", type=int, default=64)
    parser.add_argument("--kl_coeff", type=float, default=0.5)

    parser.add_argument("--reward_model_paths", type=str, default="")
    parser.add_argument("--reward_label", type=str, default="reward")
    parser.add_argument("--uncertainty_weight", type=float, default=0.5)
    parser.add_argument("--score_agg", choices=["mean", "median"], default="mean")
    parser.add_argument("--reward_transform", choices=["raw", "rank", "clip"], default="raw")
    parser.add_argument("--reward_clip_mad", type=float, default=3.0)
    parser.add_argument("--reward_direction", choices=["min", "max"], default="min")
    parser.add_argument("--eval_top_k", type=int, default=8)

    parser.add_argument("--run_name", type=str, default="v2")
    parser.add_argument("--log_root", type=str, default="logs/evo_sabdab_v2")
    parser.add_argument("--run_pyrosetta", type=str2bool, default=True)

    args = parser.parse_args()
    if not args.reward_model_paths:
        args.reward_model_paths = args.model_load_path
    if args.comb_num <= 0:
        raise ValueError("comb_num must be positive")
    if args.total_epochs <= 0:
        raise ValueError("total_epochs must be positive")
    if args.inner_epochs <= 0:
        raise ValueError("inner_epochs must be positive")
    if args.inner_batch_size <= 0:
        raise ValueError("inner_batch_size must be positive")
    if args.eval_top_k <= 0:
        raise ValueError("eval_top_k must be positive")
    return args


def load_model_args_from_checkpoint(args):
    args_path = os.path.join(os.path.dirname(os.path.abspath(args.model_load_path)), "args.pkl")
    if not os.path.exists(args_path):
        print(f"Checkpoint args not found at {args_path}; using evolution args for model init.")
        return None, []

    with open(args_path, "rb") as f:
        checkpoint_args = pickle.load(f)
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
        ranks[order] = torch.arange(acquisition.numel(), device=acquisition.device, dtype=acquisition.dtype)
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


def score_mutated_candidates(train_dataset, collate_fn, reward_models, mutate_info_list, device, score_agg):
    model_scores = [[] for _ in reward_models]
    with torch.no_grad():
        for mutate_info in mutate_info_list:
            score_batch = recursive_to(collate_fn([train_dataset.getitem_by_mutate_info(mutate_info)]), device)
            for model_idx, reward_model in enumerate(reward_models):
                score, _, _, _ = reward_model(score_batch, device)
                model_scores[model_idx].append(score.reshape(-1)[0])

    score_stack = torch.stack([torch.stack(scores) for scores in model_scores], dim=0)
    center_scores, std_scores = aggregate_score_stack(score_stack, score_agg)
    return center_scores, std_scores, score_stack


def get_topk_energy_metrics(mutate_info_list, center_scores, std_scores, acquisition, top_k):
    prefix = f"eval_top{top_k}"
    if len(mutate_info_list) == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_best_ensemble_energy": np.nan,
            f"{prefix}_mean_ensemble_energy": np.nan,
            f"{prefix}_best_reward_energy": np.nan,
            f"{prefix}_mean_reward_energy": np.nan,
            f"{prefix}_mean_reward_uncertainty": np.nan,
            f"{prefix}_selected_mutations": "",
        }

    selected_indices = torch.argsort(acquisition.detach())[: min(top_k, len(mutate_info_list))]
    selected_center = center_scores[selected_indices]
    selected_std = std_scores[selected_indices]
    selected_acquisition = acquisition[selected_indices]
    return {
        f"{prefix}_count": len(selected_indices),
        f"{prefix}_best_ensemble_energy": float(selected_acquisition.min().item()),
        f"{prefix}_mean_ensemble_energy": float(selected_acquisition.mean().item()),
        f"{prefix}_best_reward_energy": float(selected_center.min().item()),
        f"{prefix}_mean_reward_energy": float(selected_center.mean().item()),
        f"{prefix}_mean_reward_uncertainty": float(selected_std.mean().item()),
        f"{prefix}_selected_mutations": "|".join(mutate_info_list[idx] for idx in selected_indices.tolist()),
    }


def update_candidate_scores(best_candidate_scores, epoch_candidate_scores):
    for candidate in epoch_candidate_scores:
        mutate_info = candidate["mutate_info"]
        previous_candidate = best_candidate_scores.get(mutate_info)
        if previous_candidate is None or candidate["acquisition"] < previous_candidate["acquisition"]:
            best_candidate_scores[mutate_info] = candidate


def get_top_acquisition_candidates(best_candidate_scores, top_k):
    candidates = sorted(best_candidate_scores.values(), key=lambda item: item["acquisition"])
    return candidates[:top_k]


def _safe_name(text):
    return text.replace(",", "_").replace("/", "_").replace("\\", "_")


def evaluate_top_candidates_with_pyrosetta(
        pdb_id, interface_str, wt_dir, fix_dir, mut_dir, best_candidate_scores,
        output_dir, loss_history, tb_writer=None, top_k=8):
    global _PYROSETTA_INITIALIZED

    top_candidates = get_top_acquisition_candidates(best_candidate_scores, top_k)
    if not top_candidates:
        loss_history.write("\nPyRosetta ddG skipped: no ensemble-scored candidates were generated.\n")
        return pd.DataFrame()

    from scripts.calculate_ddG_pyrosetta import init_pyrosetta, relax_structure, calculate_interface_energy

    if not _PYROSETTA_INITIALIZED:
        init_pyrosetta()
        _PYROSETTA_INITIALIZED = True
    pyrosetta_dir = os.path.join(output_dir, "pyrosetta_ddg")
    os.makedirs(pyrosetta_dir, exist_ok=True)

    wt_pdb = os.path.join(fix_dir, f"{pdb_id}.pdb")
    if not os.path.exists(wt_pdb):
        wt_pdb = os.path.join(wt_dir, f"{pdb_id}.pdb")
    wt_relaxed_pdb = os.path.join(pyrosetta_dir, "wt_relaxed.pdb")
    try:
        relax_structure(wt_pdb, wt_relaxed_pdb)
        wt_energy_pdb = wt_relaxed_pdb
        wt_relaxed = True
    except Exception as exc:
        loss_history.write(f"PyRosetta ddG: failed to relax WT, using original WT. Error: {exc}\n")
        wt_energy_pdb = wt_pdb
        wt_relaxed = False

    dG_wt = calculate_interface_energy(wt_energy_pdb, interface_str)
    results = []
    for rank, candidate in enumerate(top_candidates, start=1):
        mutate_info = candidate["mutate_info"]
        mut_pdb = os.path.join(mut_dir, f"{pdb_id}_{mutate_info}.pdb")
        candidate_dir = os.path.join(pyrosetta_dir, f"rank_{rank:03d}_{_safe_name(mutate_info)}")
        os.makedirs(candidate_dir, exist_ok=True)
        mut_relaxed_pdb = os.path.join(candidate_dir, "mut_relaxed.pdb")
        row = {
            "rank_by_ensemble_energy": rank,
            "epoch": candidate["epoch"],
            "mutate_info": mutate_info,
            "ensemble_energy": candidate["acquisition"],
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
            "error": "",
        }
        try:
            loss_history.write(
                f"PyRosetta ddG rank {rank}: {mutate_info}, ensemble_energy={candidate['acquisition']:.6f}\n"
            )
            relax_structure(mut_pdb, mut_relaxed_pdb)
            row["mut_relaxed"] = True
            dG_mut = calculate_interface_energy(mut_relaxed_pdb, interface_str)
            row["dG_mut"] = float(dG_mut)
            row["ddG"] = float(dG_mut - dG_wt)
        except Exception as exc:
            row["error"] = str(exc)
            loss_history.write(f"PyRosetta ddG rank {rank} failed: {mutate_info}. Error: {exc}\n")
        results.append(row)

    result_df = pd.DataFrame(results)
    result_path = os.path.join(pyrosetta_dir, f"{pdb_id}_top{len(top_candidates)}_pyrosetta_ddg.csv")
    result_df.to_csv(result_path, index=False)
    loss_history.write(f"PyRosetta ddG results saved to {result_path}\n")
    valid_ddg = result_df["ddG"].dropna()
    if tb_writer is not None:
        tb_writer.add_scalar("pyrosetta_ddg/num_evaluated", len(result_df), 0)
        tb_writer.add_scalar("pyrosetta_ddg/num_success", len(valid_ddg), 0)
        if len(valid_ddg) > 0:
            tb_writer.add_scalar("pyrosetta_ddg/best_ddG", valid_ddg.min(), 0)
            tb_writer.add_scalar("pyrosetta_ddg/avg_ddG", valid_ddg.mean(), 0)
    return result_df


def save_epoch_metrics(loss_history, pdb_id, metrics):
    save_file = os.path.join(loss_history.save_path, pdb_id + ".csv")
    new_row = pd.DataFrame(metrics, index=[0])
    if os.path.exists(save_file):
        save_df = pd.concat([pd.read_csv(save_file), new_row], ignore_index=True)
    else:
        save_df = new_row
    save_df.to_csv(save_file, index=False)


def train_one_epoch(args, evo_model, reward_models, reference_model, optimizer, epoch, total_epoch,
                    train_dataset, collate_fn, loss_history, device, tb_writer=None):
    print("Start Train")
    loss_history.write(f"\nEpoch{epoch}: Start Train\n")

    batch = recursive_to(collate_fn([train_dataset.__getitem__(0)]), device)
    pdb_id = batch["expand_data_info"]["pdb_id"][0]
    antibody_chain = batch["expand_data_info"]["antibody_chain"][0]
    cdr_mask = batch["mutation_mask"][0]
    true_indices = torch.nonzero(cdr_mask, as_tuple=True)[0]
    if args.comb_num > len(true_indices):
        raise ValueError(
            f"comb_num={args.comb_num} exceeds the {len(true_indices)} mutable CDR residues for {pdb_id}"
        )

    mutation_mask_list = []
    for mutate_indices in combinations(true_indices, args.comb_num):
        mutation_mask = torch.zeros_like(cdr_mask, dtype=torch.bool)
        mutation_mask[torch.stack(mutate_indices)] = True
        mutation_mask_list.append(mutation_mask.unsqueeze(0))

    aa_key = "ACDEFGHIKLMNPQRSTVWY"
    mutate_info_list = []
    for mutation_mask in tqdm(mutation_mask_list, desc="Generating mutation actions for all site combinations"):
        batch["mutation_mask"] = mutation_mask
        qs = evo_model.choose_best_action(batch, device)
        actions = torch.argmin(qs[mutation_mask], dim=1)
        positions = batch["wt"]["resseq"][0][mutation_mask[0]]
        sequence = batch["wt"]["aa"][0][mutation_mask[0]]
        mutations = []
        has_mutation = False
        for state, action, position in zip(sequence, actions, positions):
            mutations.append(f"{aa_key[state.item()]}{antibody_chain}{position.item()}{aa_key[action.item()]}")
            has_mutation |= state.item() != action.item()
        if has_mutation:
            mutate_info_list.append(",".join(mutations))
    mutate_info_list = list(dict.fromkeys(mutate_info_list))

    if not mutate_info_list:
        topk_metrics = get_topk_energy_metrics([], None, None, None, args.eval_top_k)
        metrics = {
            "epoch": epoch,
            "reward_model_count": len(reward_models),
            "candidate_count": 0,
            "grpo_skipped": True,
            "train_loss": np.nan,
            **topk_metrics,
        }
        save_epoch_metrics(loss_history, pdb_id, metrics)
        loss_history.write(f"Epoch{epoch}: no non-wild-type candidates; GRPO skipped.\n")
        return {"candidate_scores": [], "metrics": metrics}

    loss_history.write(f"Epoch{epoch}: mutating {len(mutate_info_list)} candidates.\n")
    train_dataset.mutate_pdb(mutate_info_list)
    center_scores, std_scores, score_stack = score_mutated_candidates(
        train_dataset, collate_fn, reward_models, mutate_info_list, device, args.score_agg
    )
    acquisition = make_acquisition(
        center_scores, std_scores, args.uncertainty_weight, args.reward_direction
    )
    train_scores = transform_acquisition(acquisition, args.reward_transform, args.reward_clip_mad)

    old_model = MERF(args).to(device)
    old_model.load_state_dict(evo_model.state_dict())
    old_model.eval()
    for param in old_model.parameters():
        param.requires_grad = False

    score_array = train_scores.detach().cpu().numpy()
    advantages = score_array.mean() - score_array
    advantage_std = advantages.std()
    if advantage_std > 1e-8:
        advantages = advantages / (advantage_std + 1e-8)

    epoch_losses = []
    policy_losses = []
    kl_values = []
    n_samples = len(mutate_info_list)
    evo_model.train()
    for inner_epoch in tqdm(range(args.inner_epochs), desc="GRPO Training"):
        sample_indices = np.random.choice(n_samples, size=min(args.inner_batch_size, n_samples), replace=False)
        sample_batches = [train_dataset.getitem_by_mutate_info(mutate_info_list[idx]) for idx in sample_indices]
        train_batch = recursive_to(collate_fn(sample_batches), device)
        optimizer.zero_grad()

        with torch.no_grad():
            ref_logits = torch.nn.functional.log_softmax(-reference_model.choose_best_action(train_batch, device), dim=-1)
            old_logits = torch.nn.functional.log_softmax(-old_model.choose_best_action(train_batch, device), dim=-1)
        curr_logits = torch.nn.functional.log_softmax(-evo_model.choose_best_action(train_batch, device), dim=-1)

        agent_mask = train_batch["wt"]["agent_mask"]
        ref_logits = ref_logits[agent_mask].view(len(sample_indices), -1, ref_logits.size(-1))
        old_logits = old_logits[agent_mask].view(len(sample_indices), -1, old_logits.size(-1))
        curr_logits = curr_logits[agent_mask].view(len(sample_indices), -1, curr_logits.size(-1))
        ratio = torch.exp(curr_logits - old_logits).clamp(max=20.0)
        adv_tensor = torch.as_tensor(advantages[sample_indices], dtype=torch.float32, device=device)
        policy_loss = -(ratio * adv_tensor[:, None, None]).mean()
        kl_div = (ref_logits.exp() * (ref_logits - curr_logits)).sum(dim=-1).mean()
        batch_loss = policy_loss + args.kl_coeff * kl_div
        batch_loss.backward()
        optimizer.step()

        epoch_losses.append(float(batch_loss.item()))
        policy_losses.append(float(policy_loss.item()))
        kl_values.append(float(kl_div.item()))
        if tb_writer is not None:
            inner_step = epoch * args.inner_epochs + inner_epoch
            tb_writer.add_scalar("train/policy_loss", policy_loss.item(), inner_step)
            tb_writer.add_scalar("train/kl_div", kl_div.item(), inner_step)
            tb_writer.add_scalar("train/batch_loss", batch_loss.item(), inner_step)

    topk_metrics = get_topk_energy_metrics(
        mutate_info_list, center_scores, std_scores, acquisition, args.eval_top_k
    )
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
        "reward_energy_min": float(center_scores.min().item()),
        "reward_energy_mean": float(center_scores.mean().item()),
        "reward_uncertainty_mean": float(std_scores.mean().item()),
        "ensemble_energy_min": float(acquisition.min().item()),
        "ensemble_energy_mean": float(acquisition.mean().item()),
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
        tb_writer.add_scalar("reward_model/ensemble_energy_min", metrics["ensemble_energy_min"], epoch)
        tb_writer.add_scalar("train/loss", metrics["train_loss"], epoch)
        tb_writer.add_scalar(
            f"eval/top{args.eval_top_k}_mean_ensemble_energy",
            topk_metrics[f"eval_top{args.eval_top_k}_mean_ensemble_energy"],
            epoch,
        )

    loss_history.write(
        f"Epoch{epoch}: candidates={n_samples}, ensemble_energy_min={metrics['ensemble_energy_min']:.6f}, "
        f"top{args.eval_top_k}_mean_ensemble_energy="
        f"{topk_metrics[f'eval_top{args.eval_top_k}_mean_ensemble_energy']:.6f}, "
        f"loss={metrics['train_loss']:.6f}\n"
    )
    print(
        f"Epoch:{epoch + 1}/{total_epoch} Candidates:{n_samples} "
        f"EnsembleEnergy:{metrics['ensemble_energy_min']:.6f} "
        f"Top{args.eval_top_k}Mean:{topk_metrics[f'eval_top{args.eval_top_k}_mean_ensemble_energy']:.6f} "
        f"Loss:{metrics['train_loss']:.6f}"
    )

    epoch_candidate_scores = []
    for idx, mutate_info in enumerate(mutate_info_list):
        epoch_candidate_scores.append({
            "epoch": epoch,
            "mutate_info": mutate_info,
            "center_score": float(center_scores[idx].item()),
            "std_score": float(std_scores[idx].item()),
            "acquisition": float(acquisition[idx].item()),
            "per_model_scores": "|".join(
                f"{value:.6f}" for value in score_stack[:, idx].detach().cpu().tolist()
            ),
        })
    return {"candidate_scores": epoch_candidate_scores, "metrics": metrics}


def build_dataset(args, device, row):
    pdb_id = row["pdb_id"].replace("+", "").replace(".00", "")
    return EvoBaseDataset(
        pdb_id,
        row["antibody_chain"],
        row["partner"],
        row["sequence"],
        row["cdr3"],
        "/home/dataset-local/projects_dir/MERF/data/sabdab/PDBs",
        "/home/dataset-local/projects_dir/MERF/data/sabdab/PDBs_fixed",
        "/home/dataset-local/projects_dir/MERF/data/sabdab/PDBs_evo",
        knn_num=args.knn_neighbors_num,
        knn_agents_num=args.knn_agents_num,
        use_plm_embedding=args.use_plm_embedding,
        plm_path=args.plm_path,
        plm_embedding_path="/home/dataset-local/projects_dir/MERF/data/sabdab/PLM_embeddings_sabdab.pkl",
        device=device,
    )


if __name__ == "__main__":
    args = get_args()
    checkpoint_args_path, checkpoint_model_arg_keys = load_model_args_from_checkpoint(args)
    print(args)
    seed_all(args.seed)
    if args.is_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(f"cuda:{args.gpu_idx}" if args.is_cuda else "cpu")

    train_path = "data/sabdab/sabdab_evo.csv"
    train_df = pd.read_csv(train_path, dtype={"pdb_id": "string"})
    if args.dataset_idx is not None:
        if args.dataset_idx < 0 or args.dataset_idx >= len(train_df):
            raise IndexError(f"dataset_idx {args.dataset_idx} is out of range for {train_path} with {len(train_df)} rows")
        dataset_indices = [args.dataset_idx]
    else:
        dataset_indices = range(len(train_df))

    time_str = datetime.datetime.strftime(datetime.datetime.now(), "%Y_%m_%d_%H_%M_%S")
    run_dir = os.path.join(args.log_root, f"{time_str}_{args.run_name}")
    reward_paths = parse_reward_model_paths(args.reward_model_paths)
    for dataset_idx in dataset_indices:
        row = train_df.iloc[dataset_idx]
        pdb_id = row["pdb_id"].replace("+", "").replace(".00", "")
        train_dataset = build_dataset(args, device, row)
        loss_history = LossHistory(os.path.join(run_dir, pdb_id), is_evolve=True)
        tb_writer = SummaryWriter(log_dir=loss_history.save_path)
        loss_history.write(str(args) + "\n")
        loss_history.write(f"Reward paths: {args.reward_model_paths}\n")
        if checkpoint_args_path is not None:
            loss_history.write(f"Loaded model args from {checkpoint_args_path}\n")
            loss_history.write(f'Overrode model args: {", ".join(checkpoint_model_arg_keys)}\n')

        try:
            evo_model = MERF(args).to(device)
            load_pretrained(evo_model, args.model_load_path, device)
            optimizer = optim.Adam(evo_model.parameters(), lr=args.lr, weight_decay=0.1)

            reference_model = MERF(args).to(device)
            reference_model.load_state_dict(evo_model.state_dict())
            reference_model.eval()
            for param in reference_model.parameters():
                param.requires_grad = False

            reward_models = []
            for reward_path in reward_paths:
                reward_model = MERF(args).to(device)
                load_pretrained(reward_model, reward_path, device)
                reward_model.eval()
                for param in reward_model.parameters():
                    param.requires_grad = False
                reward_models.append(reward_model)
            if not reward_models:
                raise ValueError("No reward models loaded")

            best_candidate_scores = {}
            for epoch in range(args.total_epochs):
                epoch_metrics = train_one_epoch(
                    args,
                    evo_model,
                    reward_models,
                    reference_model,
                    optimizer,
                    epoch,
                    args.total_epochs,
                    train_dataset,
                    PaddingCollate(),
                    loss_history,
                    device,
                    tb_writer=tb_writer,
                )
                update_candidate_scores(best_candidate_scores, epoch_metrics["candidate_scores"])

            if args.run_pyrosetta:
                evaluate_top_candidates_with_pyrosetta(
                    pdb_id,
                    row["partner"],
                    "/home/dataset-local/projects_dir/MERF/data/sabdab/PDBs",
                    "/home/dataset-local/projects_dir/MERF/data/sabdab/PDBs_fixed",
                    "/home/dataset-local/projects_dir/MERF/data/sabdab/PDBs_evo",
                    best_candidate_scores,
                    loss_history.save_path,
                    loss_history,
                    tb_writer=tb_writer,
                    top_k=args.eval_top_k,
                )
        finally:
            tb_writer.flush()
            tb_writer.close()
