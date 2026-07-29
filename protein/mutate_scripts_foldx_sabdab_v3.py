"""SABDAB v3 使用的安全 FoldX 结构构建辅助函数。

所有候选结构均从同一个 RepairPDB 后的 WT 构建；缓存会记录原始 WT 与
修复后父结构的 SHA-256，避免误用旧实验或其他起点遗留的 PDB。
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile


DEFAULT_FOLDX_BIN = "/home/dataset-local/software_package/foldx_20270131"
REPAIR_METADATA_SUFFIX = ".repair_source_v3.json"
PARENT_METADATA_SUFFIX = ".repaired_parent_v3.json"


def mutant_pdb_path(mutated_dir: str, pdb_id: str, mutate_info: str) -> str:
    """返回 v3 候选结构路径；空突变不是合法候选。"""
    if not mutate_info:
        raise ValueError("SABDAB v3 不允许为空的 mutate_info")
    return os.path.join(mutated_dir, f"{pdb_id}_{mutate_info}.pdb")


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_copy(source: str, destination: str) -> None:
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".tmp"
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _write_json_atomically(path: str, payload: dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, path)


def _run_foldx(command: list[str], workdir: str) -> None:
    completed = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"FoldX 失败，退出码={completed.returncode}，工作目录={workdir}:\n"
            f"{completed.stdout[-4000:]}"
        )


def _repair_metadata_path(fixed_pdb_path: str) -> str:
    return fixed_pdb_path + REPAIR_METADATA_SUFFIX


def ensure_repaired_structure(
    wt_pdb_path: str,
    fixed_pdb_path: str,
    foldx_bin: str = DEFAULT_FOLDX_BIN,
) -> str:
    """只修复一次 WT，并验证既有 fixed PDB 的原始来源没有变化。"""
    if not os.path.exists(wt_pdb_path):
        raise FileNotFoundError(f"原始 WT PDB 不存在: {wt_pdb_path}")

    expected_source = {
        "raw_wt_pdb": os.path.abspath(wt_pdb_path),
        "raw_wt_sha256": _file_sha256(wt_pdb_path),
        "repair_stage": "FoldX RepairPDB",
    }
    metadata_path = _repair_metadata_path(fixed_pdb_path)
    if os.path.exists(fixed_pdb_path):
        if not os.path.exists(metadata_path):
            raise RuntimeError(
                f"fixed WT 缺少 v3 来源记录: {fixed_pdb_path}。"
                "请使用新的 PDBs_fixed_v3 目录，或删除该 v3 fixed PDB 后重建。"
            )
        with open(metadata_path, encoding="utf-8") as handle:
            recorded = json.load(handle)
        if (
            recorded.get("raw_wt_pdb") != expected_source["raw_wt_pdb"]
            or recorded.get("raw_wt_sha256") != expected_source["raw_wt_sha256"]
            or recorded.get("repair_stage") != expected_source["repair_stage"]
            or recorded.get("fixed_wt_sha256") != _file_sha256(fixed_pdb_path)
        ):
            raise RuntimeError(
                f"fixed WT 与当前原始结构不匹配: {fixed_pdb_path}。"
                "请清理对应 v3 fixed/mutated 缓存后重新开始，不能静默复用。"
            )
        return fixed_pdb_path

    os.makedirs(os.path.dirname(fixed_pdb_path), exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="foldx_sabdab_v3_repair_") as workdir:
        input_name = "input.pdb"
        shutil.copy2(wt_pdb_path, os.path.join(workdir, input_name))
        _run_foldx([foldx_bin, "--command=RepairPDB", f"--pdb={input_name}"], workdir)
        repaired_path = os.path.join(workdir, "input_Repair.pdb")
        if not os.path.exists(repaired_path):
            raise FileNotFoundError(f"FoldX RepairPDB 未生成输出: {repaired_path}")
        _atomic_copy(repaired_path, fixed_pdb_path)

    expected_source["fixed_wt_pdb"] = os.path.abspath(fixed_pdb_path)
    expected_source["fixed_wt_sha256"] = _file_sha256(fixed_pdb_path)
    _write_json_atomically(metadata_path, expected_source)
    return fixed_pdb_path


def _candidate_metadata_path(mutated_dir: str, pdb_id: str) -> str:
    return os.path.join(mutated_dir, f".{pdb_id}{PARENT_METADATA_SUFFIX}")


def _ensure_candidate_parent_provenance(
    wt_pdb_path: str,
    fixed_pdb_path: str,
    mutated_dir: str,
    pdb_id: str,
) -> None:
    """拒绝复用由不同修复父结构生成的候选 PDB。"""
    os.makedirs(mutated_dir, exist_ok=True)
    expected = {
        "raw_wt_pdb": os.path.abspath(wt_pdb_path),
        "raw_wt_sha256": _file_sha256(wt_pdb_path),
        "fixed_parent_pdb": os.path.abspath(fixed_pdb_path),
        "fixed_parent_sha256": _file_sha256(fixed_pdb_path),
        "parent_stage": "FoldX RepairPDB",
    }
    metadata_path = _candidate_metadata_path(mutated_dir, pdb_id)
    candidate_pdbs = [
        name
        for name in os.listdir(mutated_dir)
        if name.startswith(f"{pdb_id}_") and name.endswith(".pdb")
    ]
    recorded = None
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            recorded = json.load(handle)
    if recorded == expected:
        return
    if candidate_pdbs:
        raise RuntimeError(
            f"{mutated_dir} 中 {pdb_id} 的候选结构没有匹配当前 repaired WT 的来源记录。"
            "请清理该 PDB 的 v3 候选缓存后重新构建。"
        )
    _write_json_atomically(metadata_path, expected)


def build_mutant_from_fixed_parent(
    fixed_pdb_path: str,
    mutate_info: str,
    output_pdb_path: str,
    foldx_bin: str = DEFAULT_FOLDX_BIN,
) -> str:
    """在独立临时目录中从 fixed WT 执行一次 FoldX BuildModel。"""
    if os.path.exists(output_pdb_path):
        return output_pdb_path
    if not mutate_info:
        raise ValueError("不能为自替换或空突变调用 FoldX BuildModel")

    with tempfile.TemporaryDirectory(prefix="foldx_sabdab_v3_build_") as workdir:
        input_name = "input.pdb"
        shutil.copy2(fixed_pdb_path, os.path.join(workdir, input_name))
        mutation_file = os.path.join(workdir, "individual_list.txt")
        with open(mutation_file, "w", encoding="utf-8") as handle:
            handle.write(f"{mutate_info};\n")
        _run_foldx(
            [
                foldx_bin,
                "--command=BuildModel",
                f"--pdb={input_name}",
                "--mutant-file=individual_list.txt",
            ],
            workdir,
        )
        generated_path = os.path.join(workdir, "input_1.pdb")
        if not os.path.exists(generated_path):
            raise FileNotFoundError(f"FoldX BuildModel 未生成输出: {generated_path}")
        _atomic_copy(generated_path, output_pdb_path)
    return output_pdb_path


def ensure_mutant_structures(
    pdb_id: str,
    mutate_infos: list[str],
    wt_pdb_path: str,
    fixed_pdb_path: str,
    mutated_dir: str,
    workers: int = 8,
    foldx_bin: str = DEFAULT_FOLDX_BIN,
) -> dict[str, str]:
    """并行构建缺失候选，并返回 mutation string 到 PDB 路径的映射。"""
    if workers <= 0:
        raise ValueError("FoldX workers 必须为正数")
    unique_mutate_infos = list(dict.fromkeys(item.strip() for item in mutate_infos if item.strip()))
    if not unique_mutate_infos:
        return {}

    repaired_path = ensure_repaired_structure(wt_pdb_path, fixed_pdb_path, foldx_bin=foldx_bin)
    _ensure_candidate_parent_provenance(wt_pdb_path, repaired_path, mutated_dir, pdb_id)
    output_paths = {
        mutate_info: mutant_pdb_path(mutated_dir, pdb_id, mutate_info)
        for mutate_info in unique_mutate_infos
    }
    pending = [
        (mutate_info, output_paths[mutate_info])
        for mutate_info in unique_mutate_infos
        if not os.path.exists(output_paths[mutate_info])
    ]
    if not pending:
        return output_paths

    failures = []
    max_workers = min(workers, len(pending))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_mutation = {
            executor.submit(
                build_mutant_from_fixed_parent,
                repaired_path,
                mutate_info,
                output_path,
                foldx_bin,
            ): mutate_info
            for mutate_info, output_path in pending
        }
        for future in concurrent.futures.as_completed(future_to_mutation):
            mutate_info = future_to_mutation[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{mutate_info}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("FoldX 未能构建部分 SABDAB v3 候选：\n" + "\n".join(failures))
    return output_paths
