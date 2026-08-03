"""SABDAB v3 使用的安全 FoldX 结构构建辅助函数。

所有候选结构均从同一个 RepairPDB 后的 WT 构建；缓存会记录原始 WT 与
修复后父结构的 SHA-256，避免误用旧实验或其他起点遗留的 PDB。
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile


DEFAULT_FOLDX_BIN = "/home/dataset-local/software_package/foldx_20270131"
REPAIR_METADATA_SUFFIX = ".repair_source_v3.json"
PARENT_METADATA_SUFFIX = ".repaired_parent_v3.json"
MUTATION_TOKEN_RE = re.compile(
    r"^(?P<wt>[A-Z])(?P<chain>.)(?P<position>-?\d+)(?P<icode>[^,]?)(?P<mut>[A-Z])$"
)


def _normalise_icode(icode: str) -> str:
    """将 PDB 空 insertion code 规范为候选字符串中的空字段。"""
    value = str(icode).strip()
    if not value:
        return ""
    if len(value) != 1 or value in {",", ";"}:
        raise ValueError(f"不支持的 PDB insertion code: {icode!r}")
    return value


def format_mutation_token(chain: str, position: int, icode: str, wt_aa: str, mut_aa: str) -> str:
    """以 ``WT-chain-resseq-icode-mut`` 的 v3 文本格式保留插入码。

    空 insertion code 保持旧格式（如 ``HC100Y``），非空 code 使用
    ``HC100GY``。该字符串仅是内部候选标识，真正调用 FoldX 时会映射到
    临时顺序编号，避免 FoldX 不支持 insertion code 语法的问题。
    """
    if len(chain) != 1 or wt_aa not in "ACDEFGHIKLMNPQRSTVWY" or mut_aa not in "ACDEFGHIKLMNPQRSTVWY":
        raise ValueError(f"非法突变字段: chain={chain!r}, wt={wt_aa!r}, mut={mut_aa!r}")
    if wt_aa == mut_aa:
        raise ValueError("v3 不允许自替换突变")
    return f"{wt_aa}{chain}{int(position)}{_normalise_icode(icode)}{mut_aa}"


def parse_mutation_info(mutate_info: str) -> list[tuple[str, int, str, str, str]]:
    """解析 v3 候选标识，返回 ``(chain, resseq, icode, wt, mut)``。"""
    if not mutate_info:
        raise ValueError("SABDAB v3 不允许空 mutate_info")
    parsed = []
    seen_sites = set()
    for token in mutate_info.split(","):
        match = MUTATION_TOKEN_RE.fullmatch(token.strip())
        if match is None:
            raise ValueError(f"无法解析 SABDAB v3 突变标识: {token!r}")
        fields = match.groupdict()
        chain = fields["chain"]
        position = int(fields["position"])
        icode = _normalise_icode(fields["icode"])
        wt_aa, mut_aa = fields["wt"], fields["mut"]
        if wt_aa == mut_aa:
            raise ValueError(f"发现自替换突变: {token!r}")
        site = (chain, position, icode)
        if site in seen_sites:
            raise ValueError(f"同一物理位点被重复指定: {mutate_info!r}")
        seen_sites.add(site)
        parsed.append((chain, position, icode, wt_aa, mut_aa))
    return parsed


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


def _pdb_residue_key(line: str) -> tuple[str, int, str] | None:
    """读取坐标记录中的链、resSeq 和 insertion code。"""
    if len(line) < 27 or line[:6] not in {"ATOM  ", "HETATM", "ANISOU", "TER   "}:
        return None
    try:
        position = int(line[22:26].strip())
    except ValueError:
        return None
    return line[21], position, _normalise_icode(line[26])


def _replace_pdb_residue_key(line: str, position: int, icode: str) -> str:
    """仅替换 PDB 第 22--26 列的残基标识，保留其余原子记录。"""
    return line[:22] + f"{position:4d}" + (_normalise_icode(icode) or " ") + line[27:]


def _make_residue_renumbering(pdb_path: str) -> dict[tuple[str, int, str], int]:
    """为每条链建立无 insertion code 的临时连续编号。"""
    mapping: dict[tuple[str, int, str], int] = {}
    next_position: dict[str, int] = {}
    with open(pdb_path, encoding="utf-8") as handle:
        for line in handle:
            if line[:6] not in {"ATOM  ", "HETATM"}:
                continue
            key = _pdb_residue_key(line)
            if key is None or key in mapping:
                continue
            chain = key[0]
            position = next_position.get(chain, 0) + 1
            if position > 9999:
                raise ValueError(f"{pdb_path} 的链 {chain!r} 超出 PDB 四位残基编号上限")
            next_position[chain] = position
            mapping[key] = position
    if not mapping:
        raise ValueError(f"无法从 {pdb_path} 建立残基编号映射")
    return mapping


def _rewrite_with_renumbering(source_path: str, destination_path: str, mapping: dict) -> None:
    """写出 FoldX 可读取的无 insertion code 临时父结构。"""
    with open(source_path, encoding="utf-8") as source, open(destination_path, "w", encoding="utf-8") as destination:
        for line in source:
            key = _pdb_residue_key(line)
            if key is not None and key in mapping:
                line = _replace_pdb_residue_key(line, mapping[key], "")
            destination.write(line)


def _restore_original_numbering(source_path: str, destination_path: str, mapping: dict) -> None:
    """将 FoldX 输出的临时连续编号恢复为 RepairPDB 父结构的原始布局。"""
    reverse = {(chain, position, ""): original for original, position in mapping.items() for chain in [original[0]]}
    with open(source_path, encoding="utf-8") as source, open(destination_path, "w", encoding="utf-8") as destination:
        for line in source:
            key = _pdb_residue_key(line)
            if key is not None:
                original = reverse.get(key)
                if original is None and line[:6] in {"ATOM  ", "HETATM", "ANISOU"}:
                    raise ValueError(f"FoldX 输出含无法恢复编号的残基: {key}")
                if original is not None:
                    line = _replace_pdb_residue_key(line, original[1], original[2])
            destination.write(line)


def _validate_requested_mutations(
    fixed_pdb_path: str,
    mutant_pdb_path: str,
    mutations: list[tuple[str, int, str, str, str]],
) -> None:
    """验证输出布局不变，且每个请求位点恰好发生指定替换。"""
    from protein.read_pdbs import parse_pdb

    wt = parse_pdb(fixed_pdb_path)
    mut = parse_pdb(mutant_pdb_path)
    if wt is None or mut is None:
        raise ValueError("无法解析 FoldX WT 或突变输出")
    if (
        wt["chain_id"] != mut["chain_id"]
        or wt["icode"] != mut["icode"]
        or wt["resseq"].tolist() != mut["resseq"].tolist()
        or wt["aa"].shape != mut["aa"].shape
    ):
        raise ValueError("FoldX 输出改变了 WT 的残基布局")

    aa_key = "ACDEFGHIKLMNPQRSTVWY"
    site_to_index = {
        (chain, int(resseq.item()), _normalise_icode(icode)): index
        for index, (chain, resseq, icode) in enumerate(zip(wt["chain_id"], wt["resseq"], wt["icode"]))
    }
    expected = {}
    for chain, position, icode, wt_aa, mut_aa in mutations:
        site = (chain, position, icode)
        if site not in site_to_index:
            raise ValueError(f"请求突变位点不在 RepairPDB 父结构中: {site}")
        index = site_to_index[site]
        actual_wt = aa_key[int(wt["aa"][index].item())]
        if actual_wt != wt_aa:
            raise ValueError(f"请求 WT 氨基酸与父结构不符: {site}, {wt_aa} != {actual_wt}")
        expected[site] = (wt_aa, mut_aa)

    changed = {}
    for index, (wt_state, mut_state) in enumerate(zip(wt["aa"], mut["aa"])):
        if int(wt_state.item()) == int(mut_state.item()):
            continue
        site = (
            wt["chain_id"][index],
            int(wt["resseq"][index].item()),
            _normalise_icode(wt["icode"][index]),
        )
        changed[site] = (aa_key[int(wt_state.item())], aa_key[int(mut_state.item())])
    if changed != expected:
        raise ValueError(
            "FoldX 实际替换与请求不一致: "
            f"expected={expected}, actual={changed}"
        )


def build_mutant_from_fixed_parent(
    fixed_pdb_path: str,
    mutate_info: str,
    output_pdb_path: str,
    foldx_bin: str = DEFAULT_FOLDX_BIN,
) -> str:
    """在独立临时目录中从 fixed WT 执行一次 FoldX BuildModel。

    FoldX 5.1 无法直接解析 PDB insertion code。含 insertion code 的候选先在
    临时副本中按链顺序重编号，BuildModel 完成后再恢复原始 ``resseq/icode``；
    最后逐位点验证，杜绝两个插入位点折叠成同一实际替换。
    """
    mutations = parse_mutation_info(mutate_info)
    if os.path.exists(output_pdb_path):
        _validate_requested_mutations(fixed_pdb_path, output_pdb_path, mutations)
        return output_pdb_path

    with tempfile.TemporaryDirectory(prefix="foldx_sabdab_v3_build_") as workdir:
        input_name = "input.pdb"
        input_path = os.path.join(workdir, input_name)
        requires_renumbering = any(icode for _chain, _position, icode, _wt, _mut in mutations)
        foldx_mutate_info = mutate_info
        renumbering = None
        if requires_renumbering:
            renumbering = _make_residue_renumbering(fixed_pdb_path)
            _rewrite_with_renumbering(fixed_pdb_path, input_path, renumbering)
            foldx_tokens = []
            for chain, position, icode, wt_aa, mut_aa in mutations:
                original_site = (chain, position, icode)
                if original_site not in renumbering:
                    raise ValueError(f"无法在 RepairPDB 中定位 insertion-code 位点: {original_site}")
                foldx_tokens.append(
                    format_mutation_token(chain, renumbering[original_site], "", wt_aa, mut_aa)
                )
            foldx_mutate_info = ",".join(foldx_tokens)
        else:
            shutil.copy2(fixed_pdb_path, input_path)
        mutation_file = os.path.join(workdir, "individual_list.txt")
        with open(mutation_file, "w", encoding="utf-8") as handle:
            handle.write(f"{foldx_mutate_info};\n")
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
        validated_path = generated_path
        if renumbering is not None:
            validated_path = os.path.join(workdir, "restored.pdb")
            _restore_original_numbering(generated_path, validated_path, renumbering)
        _validate_requested_mutations(fixed_pdb_path, validated_path, mutations)
        _atomic_copy(validated_path, output_pdb_path)
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
