#!/usr/bin/env python
"""最小 ColabDesign AF2-Multimer 结构评估脚本。

默认评估:
  PDB:    data/CR/PDBs/cr6261_h1.pdb
  binder: A,B 链
  target: C,D 链

注意：原始复合物是 AB 抗体 + CD 抗原。AF2 输入必须包含 A/B 两条抗体链，
否则轻重链 packing 会不真实。脚本会同时报告:
  1. A_vs_CD: 只看主关注链 A 与抗原 C/D 的界面；
  2. AB_vs_CD: 把完整抗体 A/B 作为 binder context 与抗原 C/D 的界面。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Iterable

import numpy as np

# 默认强制使用物理 2 号 GPU。CUDA_VISIBLE_DEVICES=2 后，JAX 内部会把它显示为 cuda:0。
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "2")
# 禁止 JAX 悄悄 fallback 到 CPU；如果 CUDA 不可用会直接报错。
os.environ.setdefault("JAX_PLATFORMS", "cuda")
# ColabDesign 导入 matplotlib 时会尝试写用户家目录；这里提前改到 /tmp。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-colabdesign")
# 避免 JAX 一次性预占全部显存，便于在共享机器上测试。
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from colabdesign import mk_afdesign_model  # noqa: E402


DEFAULT_PDB = Path("/home/dataset-local/projects_dir/MERF/data/CR/PDBs/cr6261_h1.pdb")
DEFAULT_AF2_PARAMS = Path("/home/dataset-local/software_package/ColabDesign/params")
DEFAULT_OUTPUT_DIR = Path("/home/dataset-local/projects_dir/MERF/scripts/colabdesign_af2_output")

AA3_TO_AA1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


def log_step(message: str) -> None:
    """实时打印进度，避免长时间 AF2 编译/推理时没有任何输出。"""
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 ColabDesign AF2-Multimer 评估 A-CD binder-target 结构")
    parser.add_argument("--pdb", type=Path, default=DEFAULT_PDB, help="输入复合物 PDB")
    parser.add_argument("--af2-params", type=Path, default=DEFAULT_AF2_PARAMS, help="AF2 参数目录")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="输出目录")
    parser.add_argument("--binder-chains", default="A,B", help="AF2 输入中的抗体/binder 链，逗号分隔；默认 A,B")
    parser.add_argument("--primary-binder-chain", default="A", help="主关注 binder 链；默认 A，用于 A-CD 指标")
    parser.add_argument("--target-chains", default="C,D", help="target 链，逗号分隔；本任务默认 C,D")
    parser.add_argument(
        "--binder-sequence",
        default=None,
        help="可选：指定拼接后的 binder 序列，顺序必须与 --binder-chains 一致；默认从 A,B 链提取并拼接",
    )
    parser.add_argument("--model-index", type=int, default=0, help="AF2 multimer 模型编号，从 0 开始；0 表示 model_1")
    parser.add_argument("--recycles", type=int, default=3, help="AF2 recycle 次数")
    parser.add_argument("--dry-run", action="store_true", help="只检查输入和序列，不运行 AF2")
    return parser.parse_args()


def residue_sequence_from_pdb(pdb_path: Path, chain_id: str) -> str:
    """从 PDB 的指定链提取一字母氨基酸序列。"""
    residues: list[tuple[tuple[str, str], str]] = []
    seen: set[tuple[str, str]] = set()

    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            if line[21].strip() != chain_id:
                continue
            res_key = (line[22:26].strip(), line[26].strip())
            if res_key in seen:
                continue
            seen.add(res_key)
            aa3 = line[17:20].strip().upper()
            residues.append((res_key, AA3_TO_AA1.get(aa3, "X")))

    sequence = "".join(aa for _, aa in residues)
    if not sequence:
        raise ValueError(f"在 {pdb_path} 中没有找到链 {chain_id}")
    if "X" in sequence:
        raise ValueError(f"链 {chain_id} 含有非标准氨基酸，当前最小脚本不处理 X 序列")
    return sequence


def chain_lengths_from_pdb(pdb_path: Path, chains: Iterable[str]) -> dict[str, int]:
    """统计每条链的残基数，用于构造 PAE 矩阵切片。"""
    chain_list = list(chains)
    chain_set = set(chain_list)
    residues: dict[str, set[tuple[str, str]]] = {chain: set() for chain in chain_list}
    with pdb_path.open() as handle:
        for line in handle:
            if not line.startswith("ATOM"):
                continue
            chain_id = line[21].strip()
            if chain_id in chain_set:
                residues[chain_id].add((line[22:26].strip(), line[26].strip()))
    lengths = {chain: len(residues[chain]) for chain in chain_list}
    missing = [chain for chain, length in lengths.items() if length == 0]
    if missing:
        raise ValueError(f"输入 PDB 缺少链: {','.join(missing)}")
    return lengths


def calc_d0(length: np.ndarray) -> np.ndarray:
    """ipSAE 使用的 d0 项，复刻 Proteina-Complexa vendored ColabDesign 公式。"""
    length = np.clip(length.astype(float), a_min=27.0, a_max=None)
    d0 = 1.24 * np.power(length - 15.0, 1.0 / 3.0) - 1.8
    return np.clip(d0, a_min=1.0, a_max=None)


def ipsae_from_pae(pae: np.ndarray, mask_a: np.ndarray, mask_b: np.ndarray, pae_cutoff: float) -> dict[str, float]:
    """计算 A/B 两组残基之间的 ipSAE，返回双向 min/max/avg。"""

    def one_direction(row_mask: np.ndarray, col_mask: np.ndarray) -> float:
        pair_mask = row_mask[:, None] * col_mask[None, :] * (pae < pae_cutoff)
        d0 = calc_d0(pair_mask.sum(axis=1))
        tm_term = 1.0 / (1.0 + np.square(pae) / np.square(d0)[:, None])
        denom = pair_mask.sum(axis=1) + 1e-8
        mean_tm = (pair_mask * tm_term).sum(axis=1) / denom
        return float(mean_tm.max())

    ipsae_ab = one_direction(mask_a, mask_b)
    ipsae_ba = one_direction(mask_b, mask_a)
    return {
        "min": min(ipsae_ab, ipsae_ba),
        "max": max(ipsae_ab, ipsae_ba),
        "avg": float(np.mean([ipsae_ab, ipsae_ba])),
    }


def interface_metrics_from_pae(
    pae: np.ndarray,
    target_lengths: dict[str, int],
    binder_lengths: dict[str, int],
    primary_binder_chain: str,
) -> dict[str, object]:
    """从 AF2 PAE 矩阵中计算 A-CD 与 AB-CD 的界面指标。

    ColabDesign binder protocol 的残基顺序是:
      target chains (C,D) + binder chains (A,B)
    所以这里显式构造每条链的 mask，避免多链抗体/抗原时误切 PAE。
    """
    target_len = sum(target_lengths.values())
    binder_len = sum(binder_lengths.values())
    total_len = target_len + binder_len
    if pae.shape != (total_len, total_len):
        raise ValueError(f"PAE 矩阵形状 {pae.shape} 与输入长度 {total_len} 不一致")

    chain_masks: dict[str, np.ndarray] = {}
    start = 0
    for chain, length in target_lengths.items():
        mask = np.zeros(total_len, dtype=float)
        mask[start : start + length] = 1.0
        chain_masks[chain] = mask
        start += length
    for chain, length in binder_lengths.items():
        mask = np.zeros(total_len, dtype=float)
        mask[start : start + length] = 1.0
        chain_masks[chain] = mask
        start += length

    if primary_binder_chain not in chain_masks:
        raise ValueError(f"主关注 binder 链 {primary_binder_chain} 不在输入 binder 链中")

    target_mask = sum((chain_masks[chain] for chain in target_lengths), np.zeros(total_len, dtype=float))
    binder_context_mask = sum((chain_masks[chain] for chain in binder_lengths), np.zeros(total_len, dtype=float))
    primary_binder_mask = chain_masks[primary_binder_chain]

    pae_sym = (pae + pae.T) / 2.0

    def summarize(mask_a: np.ndarray, mask_b: np.ndarray) -> dict[str, float]:
        """汇总两组链之间的 PAE/ipSAE；i_pAE 使用对称 PAE，min_ipAE 用方向性 PAE。"""
        values = pae_sym[(mask_a[:, None] * mask_b[None, :]).astype(bool)]
        directed = pae[mask_a.astype(bool)][:, mask_b.astype(bool)]
        ipsae_15 = ipsae_from_pae(pae, mask_a, mask_b, pae_cutoff=15.0)
        ipsae_10 = ipsae_from_pae(pae, mask_a, mask_b, pae_cutoff=10.0)
        return {
            "i_pAE": round(float(values.mean() / 31.0), 4),
            "i_pAE_A": round(float(values.mean()), 4),
            "min_ipAE": round(float(directed.min() / 31.0), 4),
            "min_ipAE_A": round(float(directed.min()), 4),
            "min_ipSAE": round(ipsae_15["min"], 4),
            "max_ipSAE": round(ipsae_15["max"], 4),
            "avg_ipSAE": round(ipsae_15["avg"], 4),
            "min_ipSAE_10": round(ipsae_10["min"], 4),
            "max_ipSAE_10": round(ipsae_10["max"], 4),
            "avg_ipSAE_10": round(ipsae_10["avg"], 4),
        }

    primary_vs_target = summarize(primary_binder_mask, target_mask)
    context_vs_target = summarize(binder_context_mask, target_mask)

    per_binder_chain_vs_target = {
        f"{binder_chain}_vs_{''.join(target_lengths)}": summarize(chain_masks[binder_chain], target_mask)
        for binder_chain in binder_lengths
    }
    per_chain_pair = {
        f"{binder_chain}_vs_{target_chain}": summarize(chain_masks[binder_chain], chain_masks[target_chain])
        for binder_chain in binder_lengths
        for target_chain in target_lengths
    }

    return {
        f"{primary_binder_chain}_vs_{''.join(target_lengths)}": primary_vs_target,
        f"{''.join(binder_lengths)}_vs_{''.join(target_lengths)}": context_vs_target,
        "per_binder_chain_vs_target": per_binder_chain_vs_target,
        "per_chain_pair": per_chain_pair,
        "chain_masks": chain_masks,
    }


def main() -> None:
    args = parse_args()
    target_chains = [chain.strip() for chain in args.target_chains.split(",") if chain.strip()]
    binder_chains = [chain.strip() for chain in args.binder_chains.split(",") if chain.strip()]
    if not target_chains:
        raise ValueError("--target-chains 不能为空")
    if not binder_chains:
        raise ValueError("--binder-chains 不能为空")
    if args.primary_binder_chain not in binder_chains:
        raise ValueError("--primary-binder-chain 必须包含在 --binder-chains 中")

    if args.binder_sequence:
        binder_sequence = args.binder_sequence
    else:
        binder_sequence = "".join(residue_sequence_from_pdb(args.pdb, chain) for chain in binder_chains)
    target_lengths = chain_lengths_from_pdb(args.pdb, target_chains)
    binder_lengths = chain_lengths_from_pdb(args.pdb, binder_chains)
    binder_len = len(binder_sequence)
    expected_binder_len = sum(binder_lengths.values())
    if binder_len != expected_binder_len:
        raise ValueError(
            f"binder 序列长度为 {binder_len}，但 --binder-chains {binder_chains} "
            f"对应 PDB 长度为 {expected_binder_len}"
        )
    target_len = sum(target_lengths[chain] for chain in target_chains)

    log_step(f"输入 PDB: {args.pdb}")
    log_step(f"AF2 binder/context 链: {','.join(binder_chains)}，长度: {binder_len}，各链: {binder_lengths}")
    log_step(f"主关注 binder 链: {args.primary_binder_chain}")
    log_step(f"target 链: {','.join(target_chains)}，长度: {target_len}，各链: {target_lengths}")
    log_step("B 链会进入 AF2 输入；顶层 i_pAE/min_ipAE 等指标默认报告 A-CD，另报告 AB-CD。")

    if args.dry_run:
        log_step("dry-run 完成：未运行 AF2。")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_pdb = args.output_dir / f"{args.pdb.stem}_{''.join(binder_chains)}_vs_{''.join(target_chains)}_model{args.model_index + 1}.pdb"
    output_json = args.output_dir / f"{args.pdb.stem}_{''.join(binder_chains)}_vs_{''.join(target_chains)}_metrics.json"

    start_time = time.time()
    try:
        import jax

        backend = jax.default_backend()
        devices = [str(device) for device in jax.devices()]
        log_step(f"JAX backend: {backend}; devices: {devices}")
        if backend == "cpu" and target_len + binder_len > 800:
            log_step("警告：当前是 CPU backend，且复合物长度较大；完整 AF2-Multimer 可能运行很久。")
    except Exception as exc:
        log_step(f"无法读取 JAX backend 信息：{exc}")

    # 构建 AF2-Multimer binder 模型。protocol=binder 会固定 target 序列，
    # predict(seq=...) 只替换 binder 序列。
    log_step("开始构建 ColabDesign AF2-Multimer binder 模型...")
    model = mk_afdesign_model(
        protocol="binder",
        num_recycles=args.recycles,
        data_dir=str(args.af2_params),
        use_multimer=True,
        use_initial_guess=True,
        use_initial_atom_pos=False,
    )
    log_step(f"模型构建完成，用时 {time.time() - start_time:.1f} 秒。")

    # 关键：target 传 C,D，binder 传 A,B。B 链作为抗体轻链参与 AF2 结构预测，
    # 后续用 PAE 矩阵手动拆分 A-CD 和 AB-CD 指标。
    log_step("开始准备 AF2 输入特征...")
    model.prep_inputs(
        pdb_filename=str(args.pdb),
        chain=",".join(target_chains),
        binder_chain=",".join(binder_chains),
        binder_len=binder_len,
        use_binder_template=True,
        rm_target_seq=False,
        rm_target_sc=False,
        rm_template_ic=True,
    )
    log_step(f"输入特征准备完成，用时 {time.time() - start_time:.1f} 秒。")

    log_step("开始 AF2-Multimer 预测...")
    model.predict(
        seq=binder_sequence,
        models=[args.model_index],
        num_recycles=args.recycles,
        verbose=False,
    )
    log_step(f"AF2-Multimer 预测完成，用时 {time.time() - start_time:.1f} 秒。")
    model.save_pdb(str(output_pdb))

    log = {key: float(np.asarray(value)) for key, value in model.aux["log"].items() if np.asarray(value).ndim == 0}
    pae = np.asarray(model.aux["pae"], dtype=float)
    interface_metrics = interface_metrics_from_pae(
        pae=pae,
        target_lengths={chain: target_lengths[chain] for chain in target_chains},
        binder_lengths={chain: binder_lengths[chain] for chain in binder_chains},
        primary_binder_chain=args.primary_binder_chain,
    )
    primary_key = f"{args.primary_binder_chain}_vs_{''.join(target_chains)}"
    context_key = f"{''.join(binder_chains)}_vs_{''.join(target_chains)}"
    primary_metrics = interface_metrics[primary_key]
    context_metrics = interface_metrics[context_key]

    plddt = np.asarray(model.aux["plddt"], dtype=float)
    chain_masks = interface_metrics.pop("chain_masks")
    primary_plddt = float(plddt[chain_masks[args.primary_binder_chain].astype(bool)].mean())

    # 顶层字段尽量保持 Proteina-Complexa colabdesign_utils.py 的命名习惯。
    # 对多链抗体，本脚本约定顶层界面字段是主关注链 A 对 C/D；完整 AB-CD
    # 指标放在 binder_context_vs_target 中。
    metrics = {
        "pLDDT": round(log["plddt"], 4),
        "pLDDT_100": round(log["plddt"] * 100.0, 2),
        "pLDDT_primary_chain": round(primary_plddt, 4),
        "pLDDT_primary_chain_100": round(primary_plddt * 100.0, 2),
        "pTM": round(log["ptm"], 4),
        "i_pTM": round(log["i_ptm"], 4),
        "pAE": round(log["pae"], 4),
        "i_pAE": primary_metrics["i_pAE"],
        "i_pAE_A": primary_metrics["i_pAE_A"],
        "min_ipAE": primary_metrics["min_ipAE"],
        "min_ipAE_A": primary_metrics["min_ipAE_A"],
        "min_ipSAE": primary_metrics["min_ipSAE"],
        "max_ipSAE": primary_metrics["max_ipSAE"],
        "avg_ipSAE": primary_metrics["avg_ipSAE"],
        "min_ipSAE_10": primary_metrics["min_ipSAE_10"],
        "max_ipSAE_10": primary_metrics["max_ipSAE_10"],
        "avg_ipSAE_10": primary_metrics["avg_ipSAE_10"],
        "primary_vs_target": primary_metrics,
        "binder_context_vs_target": context_metrics,
        "complex_pdb_path": str(output_pdb),
        "interface_scope": f"top-level metrics are {primary_key}; AF2 input binder context is {context_key}",
        "per_binder_chain_vs_target": interface_metrics["per_binder_chain_vs_target"],
        "per_chain_pair": interface_metrics["per_chain_pair"],
    }

    output_json.write_text(json.dumps(metrics, indent=2, ensure_ascii=False) + "\n")
    log_step(json.dumps(metrics, indent=2, ensure_ascii=False))
    log_step(f"预测 PDB: {output_pdb}")
    log_step(f"指标 JSON: {output_json}")


if __name__ == "__main__":
    main()
