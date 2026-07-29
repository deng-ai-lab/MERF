#!/home/dataset-local/anaconda3/envs/colabdesign-af2/bin/python
"""面向 SABDAB 抗体-抗原复合物的 ColabDesign AF2-Multimer 评估。

SABDAB 的 partner 字段采用“抗体链集合_抗原链集合”格式，例如 HL_A、Aa_B、
CD_AB。v3 会保留完整抗体作为 binder context，并以 antibody_chain 指定
需要重点报告界面指标的主抗体链。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

# 必须在导入 ColabDesign 前设置，外部调用传入的 CUDA_VISIBLE_DEVICES 优先。
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-colabdesign")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from calculate_colabdesign import (  # noqa: E402
    DEFAULT_AF2_PARAMS,
    interface_metrics_from_pae,
    log_step,
    mk_afdesign_model,
)
from colabdesign.af.alphafold.common import protein, residue_constants  # noqa: E402
from colabdesign.shared.protein import MODRES, pdb_to_string  # noqa: E402


def parse_partner(partner: str) -> tuple[list[str], list[str]]:
    """将 SABDAB partner 解析为完整抗体链和抗原链。"""
    parts = partner.strip().split("_")
    if len(parts) != 2 or not parts[0] or not parts[1]:
        raise ValueError(
            f"partner={partner!r} 格式错误；SABDAB v3 需要“抗体链集合_抗原链集合”，"
            "例如 HL_A 或 CD_AB"
        )
    binder_chains = list(parts[0])
    target_chains = list(parts[1])
    if len(set(binder_chains)) != len(binder_chains):
        raise ValueError(f"partner={partner!r} 的抗体链存在重复")
    if len(set(target_chains)) != len(target_chains):
        raise ValueError(f"partner={partner!r} 的抗原链存在重复")
    if set(binder_chains) & set(target_chains):
        raise ValueError(f"partner={partner!r} 的抗体链和抗原链不能重叠")
    return binder_chains, target_chains


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="用 ColabDesign AF2-Multimer 评估 SABDAB 复合物")
    parser.add_argument("--pdb", type=Path, required=True, help="待评估的 WT 或 mutant PDB")
    parser.add_argument("--partner", required=True, help="SABDAB partner，例如 HL_A 或 Aa_B")
    parser.add_argument(
        "--primary-binder-chain",
        required=True,
        help="本次发生进化的抗体链；顶层界面指标以它对抗原计算",
    )
    parser.add_argument("--af2-params", type=Path, default=DEFAULT_AF2_PARAMS, help="AF2 参数目录")
    parser.add_argument("--output-dir", type=Path, required=True, help="预测 PDB 与日志输出目录")
    parser.add_argument("--metrics-json", type=Path, default=None, help="可选的固定指标 JSON 输出路径")
    parser.add_argument("--output-pdb", type=Path, default=None, help="可选的固定预测 PDB 输出路径")
    parser.add_argument(
        "--binder-sequence",
        default=None,
        help="可选完整抗体序列校验值；必须与 AF2 规范化 PDB 中的 binder 序列完全一致",
    )
    parser.add_argument("--model-index", type=int, default=0, help="AF2 multimer 模型编号，从 0 开始")
    parser.add_argument("--recycles", type=int, default=3, help="AF2 recycle 次数")
    parser.add_argument("--dry-run", action="store_true", help="只检查 SABDAB 链定义和序列，不运行 AF2")
    return parser.parse_args()


def _default_output_paths(
    output_dir: Path,
    pdb_path: Path,
    binder_chains: list[str],
    target_chains: list[str],
    model_index: int,
) -> tuple[Path, Path]:
    chain_label = f"{''.join(binder_chains)}_vs_{''.join(target_chains)}"
    output_pdb = output_dir / f"{pdb_path.stem}_{chain_label}_model{model_index + 1}.pdb"
    output_json = output_dir / f"{pdb_path.stem}_{chain_label}_metrics.json"
    return output_pdb, output_json


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def normalize_pdb_for_colabdesign_v3(
    input_pdb: Path,
    output_pdb: Path,
    residue_map_path: Path,
    partner_chains: list[str],
) -> dict:
    """给 AF2 临时输入重新编号，保留 SAbDab 的插入码残基。

    ColabDesign 的 pdb_to_string 会删掉 PDB insertion code，例如 100A 会被
    视为 100。抗体 CDR 中这类残基很常见，因此这里仅在 AF2 的副本中把每个
    (chain, resseq, insertion code) 映射到唯一的连续编号；原始 FoldX/Rosetta
    输入不做任何改动。
    """
    chain_set = set(partner_chains)
    next_resseq = {chain: 1 for chain in partner_chains}
    normalized_resseq: dict[tuple[str, str, str], int] = {}
    residue_rows = {chain: [] for chain in partner_chains}
    output_lines = []
    model = 1

    with input_pdb.open() as handle:
        for raw_line in handle:
            line = raw_line.rstrip("\r\n")
            if line[:5] == "MODEL":
                model = int(line[5:].strip())

            record_name = line[:6].strip()
            residue_name = line[17:20].strip()
            # pdb_to_string 会保留 ATOM，也会把已知的修饰氨基酸 HETATM 转成 ATOM。
            is_protein_record = record_name == "ATOM" or (
                record_name == "HETATM" and residue_name in MODRES
            )
            chain = line[21:22]
            if model == 1 and is_protein_record and chain in chain_set:
                source_resseq = line[22:26].strip()
                source_icode = line[26:27].strip()
                if not source_resseq:
                    raise ValueError(f"PDB 中链 {chain} 存在空残基编号: {input_pdb}")
                residue_key = (chain, source_resseq, source_icode)
                if residue_key not in normalized_resseq:
                    current_resseq = next_resseq[chain]
                    if current_resseq > 9999:
                        raise ValueError(f"链 {chain} 的 AF2 规范化编号超过 PDB 上限 9999")
                    normalized_resseq[residue_key] = current_resseq
                    next_resseq[chain] += 1
                    residue_rows[chain].append(
                        {
                            "source_resseq": source_resseq,
                            "source_insertion_code": source_icode,
                            "normalized_resseq": current_resseq,
                            "residue_name": residue_name,
                        }
                    )
                line = line[:22] + f"{normalized_resseq[residue_key]:4d} " + line[27:]
            output_lines.append(line)

    missing_chains = [chain for chain in partner_chains if not residue_rows[chain]]
    if missing_chains:
        raise ValueError(f"输入 PDB 缺少 partner 链: {','.join(missing_chains)}")

    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_pdb.write_text("\n".join(output_lines) + "\n")
    metadata = {
        "schema": "sabdab_colabdesign_normalization_v3",
        "input_pdb": str(input_pdb.resolve()),
        "normalized_pdb": str(output_pdb.resolve()),
        "partner_chains": partner_chains,
        "normalized_chain_lengths": {
            chain: len(residue_rows[chain]) for chain in partner_chains
        },
        "residue_map": residue_rows,
    }
    _write_json(residue_map_path, metadata)
    return metadata


def _aatype_to_sequence(aatype: np.ndarray, chain: str) -> str:
    """按 AlphaFold 的 aatype 定义恢复序列，拒绝无法用于 AF2 的未知残基。"""
    sequence = []
    for residue_type in np.asarray(aatype, dtype=int):
        if residue_type < 0 or residue_type >= residue_constants.restype_num:
            raise ValueError(f"AF2 解析后的链 {chain} 含未知残基类型 index={residue_type}")
        sequence.append(residue_constants.restypes[residue_type])
    return "".join(sequence)


def effective_chain_info_from_colabdesign(
    normalized_pdb: Path,
    chains: list[str],
) -> dict[str, dict]:
    """复用 ColabDesign 实际解析路径，得到模板真正采用的链序列与长度。"""
    chain_info = {}
    n_atom_index = residue_constants.atom_order["N"]
    for chain in chains:
        pdb_string = pdb_to_string(str(normalized_pdb), chains=chain, models=[1])
        protein_object = protein.from_pdb_string(pdb_string, chain_id=chain)
        atom_mask = np.asarray(protein_object.atom_mask)
        keep = atom_mask[:, n_atom_index] == 1
        aatype = np.asarray(protein_object.aatype)[keep]
        if not len(aatype):
            raise ValueError(f"链 {chain} 在 ColabDesign 过滤缺失主链 N 后为空")
        sequence = _aatype_to_sequence(aatype, chain)
        chain_info[chain] = {
            "sequence": sequence,
            "length": int(len(sequence)),
            "dropped_missing_n": int((~keep).sum()),
        }
    return chain_info


def main() -> None:
    args = parse_args()
    if not args.pdb.exists():
        raise FileNotFoundError(f"输入 PDB 不存在: {args.pdb}")
    binder_chains, target_chains = parse_partner(args.partner)
    if args.primary_binder_chain not in binder_chains:
        raise ValueError(
            f"主抗体链 {args.primary_binder_chain} 不在 partner 的抗体链集合 {binder_chains} 中"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    default_pdb, default_json = _default_output_paths(
        args.output_dir, args.pdb, binder_chains, target_chains, args.model_index
    )
    output_pdb = args.output_pdb or default_pdb
    output_json = args.metrics_json or default_json
    output_pdb.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    # 规范化副本是 AF2 的唯一结构来源：模板、序列、长度和 PAE mask 必须一致。
    normalized_pdb = args.output_dir / "af2_input_normalized_v3.pdb"
    residue_map_json = args.output_dir / "af2_residue_map_v3.json"
    normalization = normalize_pdb_for_colabdesign_v3(
        input_pdb=args.pdb,
        output_pdb=normalized_pdb,
        residue_map_path=residue_map_json,
        partner_chains=target_chains + binder_chains,
    )
    effective_chain_info = effective_chain_info_from_colabdesign(
        normalized_pdb,
        target_chains + binder_chains,
    )
    target_lengths = {chain: effective_chain_info[chain]["length"] for chain in target_chains}
    binder_lengths = {chain: effective_chain_info[chain]["length"] for chain in binder_chains}
    binder_sequence = "".join(effective_chain_info[chain]["sequence"] for chain in binder_chains)
    if args.binder_sequence is not None and args.binder_sequence != binder_sequence:
        raise ValueError(
            "--binder-sequence 与 AF2 规范化 PDB 的完整抗体序列不一致；"
            "v3 不允许用外部序列覆盖结构模板序列。"
        )
    expected_binder_len = sum(binder_lengths.values())
    total_length = sum(target_lengths.values()) + expected_binder_len

    log_step(f"输入 PDB: {args.pdb}")
    log_step(f"AF2 规范化 PDB: {normalized_pdb}")
    log_step(f"插入码映射: {residue_map_json}")
    log_step(f"SABDAB partner: {args.partner}")
    log_step(
        f"AF2 binder context: {','.join(binder_chains)}，主抗体链: "
        f"{args.primary_binder_chain}，各链长度: {binder_lengths}"
    )
    log_step(f"AF2 target: {','.join(target_chains)}，各链长度: {target_lengths}")

    if args.dry_run:
        _write_json(
            output_json,
            {
                "dry_run": True,
                "input_pdb": str(args.pdb),
                "partner": args.partner,
                "binder_chains": binder_chains,
                "target_chains": target_chains,
                "primary_binder_chain": args.primary_binder_chain,
                "af2_input_pdb": str(normalized_pdb),
                "af2_residue_map_json": str(residue_map_json),
                "normalized_chain_lengths": normalization["normalized_chain_lengths"],
                "effective_chain_info": effective_chain_info,
                "binder_lengths": binder_lengths,
                "target_lengths": target_lengths,
                "binder_sequence_length": len(binder_sequence),
                "af2_total_length": total_length,
            },
        )
        log_step("dry-run 完成：AF2 规范化、链、序列和长度检查通过，未运行 AF2。")
        return

    start_time = time.time()
    model = mk_afdesign_model(
        protocol="binder",
        num_recycles=args.recycles,
        data_dir=str(args.af2_params),
        use_multimer=True,
        use_initial_guess=True,
        use_initial_atom_pos=False,
    )
    log_step(f"AF2-Multimer 模型构建完成，用时 {time.time() - start_time:.1f} 秒。")
    model.prep_inputs(
        pdb_filename=str(normalized_pdb),
        chain=",".join(target_chains),
        binder_chain=",".join(binder_chains),
        binder_len=len(binder_sequence),
        use_binder_template=True,
        rm_target_seq=False,
        rm_target_sc=False,
        rm_template_ic=True,
    )
    if model._target_len != sum(target_lengths.values()) or model._binder_len != expected_binder_len:
        raise RuntimeError(
            "ColabDesign 实际模板长度与规范化 PDB 的解析长度不一致："
            f"target {model._target_len}/{sum(target_lengths.values())}, "
            f"binder {model._binder_len}/{expected_binder_len}"
        )
    model.predict(
        seq=binder_sequence,
        models=[args.model_index],
        num_recycles=args.recycles,
        verbose=False,
    )
    model.save_pdb(str(output_pdb))

    log_values = {
        key: float(np.asarray(value))
        for key, value in model.aux["log"].items()
        if np.asarray(value).ndim == 0
    }
    pae = np.asarray(model.aux["pae"], dtype=float)
    if pae.shape != (total_length, total_length):
        raise ValueError(f"AF2 PAE 形状 {pae.shape} 与规范化输入长度 {total_length} 不一致")
    plddt = np.asarray(model.aux["plddt"], dtype=float)
    if plddt.ndim != 1 or plddt.shape[0] != total_length:
        raise ValueError(
            f"AF2 pLDDT 形状 {plddt.shape} 与规范化输入长度 {total_length} 不一致"
        )
    interface_metrics = interface_metrics_from_pae(
        pae=pae,
        target_lengths=target_lengths,
        binder_lengths=binder_lengths,
        primary_binder_chain=args.primary_binder_chain,
    )
    primary_key = f"{args.primary_binder_chain}_vs_{''.join(target_chains)}"
    context_key = f"{''.join(binder_chains)}_vs_{''.join(target_chains)}"
    primary_metrics = interface_metrics[primary_key]
    context_metrics = interface_metrics[context_key]
    chain_masks = interface_metrics.pop("chain_masks")
    primary_plddt = float(plddt[chain_masks[args.primary_binder_chain].astype(bool)].mean())

    metrics = {
        "input_pdb": str(args.pdb),
        "af2_input_pdb": str(normalized_pdb),
        "af2_residue_map_json": str(residue_map_json),
        "partner": args.partner,
        "binder_chains": binder_chains,
        "target_chains": target_chains,
        "primary_binder_chain": args.primary_binder_chain,
        "normalized_chain_lengths": normalization["normalized_chain_lengths"],
        "effective_chain_info": effective_chain_info,
        "af2_total_length": total_length,
        "pLDDT": round(log_values["plddt"], 4),
        "pLDDT_100": round(log_values["plddt"] * 100.0, 2),
        "pLDDT_primary_chain": round(primary_plddt, 4),
        "pLDDT_primary_chain_100": round(primary_plddt * 100.0, 2),
        "pTM": round(log_values["ptm"], 4),
        "i_pTM": round(log_values["i_ptm"], 4),
        "pAE": round(log_values["pae"], 4),
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
        "per_binder_chain_vs_target": interface_metrics["per_binder_chain_vs_target"],
        "per_chain_pair": interface_metrics["per_chain_pair"],
        "complex_pdb_path": str(output_pdb),
        "interface_scope": (
            f"顶层界面指标为 {primary_key}；AF2 输入中的完整抗体 context 为 {context_key}"
        ),
    }
    _write_json(output_json, metrics)
    log_step(json.dumps(metrics, indent=2, ensure_ascii=False))
    log_step(f"预测 PDB: {output_pdb}")
    log_step(f"指标 JSON: {output_json}")


if __name__ == "__main__":
    main()
