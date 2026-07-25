"""Safe, parallel FoldX helpers for ESMFold2-backed flexible CR evolution.

Unlike the legacy helper, this module never changes the caller's working
directory.  Each BuildModel invocation owns a temporary directory, so a batch
of policy-nominated mutations from one start structure can run in parallel.
"""

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile


DEFAULT_FOLDX_BIN = "/home/dataset-local/software_package/foldx_20270131"
REPAIRED_PARENT_METADATA = ".repaired_parent.json"


def target_pdb_path(mutated_dir, target_reverse_key):
    """Return a readable, start-local path for an absolute landscape state."""
    suffix = target_reverse_key if target_reverse_key else "mature"
    return os.path.join(mutated_dir, f"target_{suffix}.pdb")


def _run_foldx(command, workdir):
    completed = subprocess.run(
        command,
        cwd=workdir,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"FoldX exited with code {completed.returncode} in {workdir}:\n{completed.stdout[-4000:]}"
        )


def _atomic_copy(source, destination):
    os.makedirs(os.path.dirname(destination), exist_ok=True)
    temporary = destination + ".tmp"
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def _file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _ensure_repaired_parent_cache(mutated_dir, fixed_pdb_path):
    """Reject cached target PDBs unless they were built from this repaired parent."""
    os.makedirs(mutated_dir, exist_ok=True)
    metadata_path = os.path.join(mutated_dir, REPAIRED_PARENT_METADATA)
    expected = {
        "parent_pdb": os.path.abspath(fixed_pdb_path),
        "parent_sha256": _file_sha256(fixed_pdb_path),
        "parent_stage": "FoldX RepairPDB",
    }
    target_pdbs = [
        name for name in os.listdir(mutated_dir)
        if name.startswith("target_") and name.endswith(".pdb")
    ]
    recorded = None
    if os.path.exists(metadata_path):
        with open(metadata_path, encoding="utf-8") as handle:
            recorded = json.load(handle)
    if recorded == expected:
        return
    if target_pdbs:
        raise RuntimeError(
            f"Cached target PDBs in {mutated_dir} do not have verified repaired-parent provenance. "
            "Delete them before continuing so they can be rebuilt from PDBs_fixed."
        )
    temporary = metadata_path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(expected, handle, ensure_ascii=False, indent=2)
    os.replace(temporary, metadata_path)


def ensure_repaired_structure(wt_pdb_path, fixed_pdb_path, foldx_bin=DEFAULT_FOLDX_BIN):
    """Run RepairPDB once for one ESMFold2 start structure and cache the result."""
    if os.path.exists(fixed_pdb_path):
        return fixed_pdb_path
    if not os.path.exists(wt_pdb_path):
        raise FileNotFoundError(f"Start PDB does not exist: {wt_pdb_path}")
    with tempfile.TemporaryDirectory(prefix="foldx_repair_") as workdir:
        input_name = "input.pdb"
        shutil.copy2(wt_pdb_path, os.path.join(workdir, input_name))
        _run_foldx([foldx_bin, "--command=RepairPDB", f"--pdb={input_name}"], workdir)
        repaired = os.path.join(workdir, "input_Repair.pdb")
        if not os.path.exists(repaired):
            raise FileNotFoundError(f"FoldX RepairPDB did not create {repaired}")
        _atomic_copy(repaired, fixed_pdb_path)
    return fixed_pdb_path


def build_model_from_start(
    fixed_pdb_path,
    foldx_mutation_info,
    output_pdb_path,
    foldx_bin=DEFAULT_FOLDX_BIN,
):
    """Generate one target structure from the repaired start PDB with BuildModel."""
    if os.path.exists(output_pdb_path):
        return output_pdb_path
    if not foldx_mutation_info:
        _atomic_copy(fixed_pdb_path, output_pdb_path)
        return output_pdb_path
    with tempfile.TemporaryDirectory(prefix="foldx_buildmodel_") as workdir:
        input_name = "input.pdb"
        shutil.copy2(fixed_pdb_path, os.path.join(workdir, input_name))
        mutation_file = os.path.join(workdir, "individual_list.txt")
        with open(mutation_file, "w", encoding="utf-8") as handle:
            handle.write(f"{foldx_mutation_info};\n")
        _run_foldx(
            [
                foldx_bin,
                "--command=BuildModel",
                f"--pdb={input_name}",
                "--mutant-file=individual_list.txt",
            ],
            workdir,
        )
        generated = os.path.join(workdir, "input_1.pdb")
        if not os.path.exists(generated):
            raise FileNotFoundError(f"FoldX BuildModel did not create {generated}")
        _atomic_copy(generated, output_pdb_path)
    return output_pdb_path


def ensure_target_structures(
    wt_pdb_path,
    fixed_pdb_path,
    mutated_dir,
    target_to_foldx_mutation,
    workers=8,
    foldx_bin=DEFAULT_FOLDX_BIN,
):
    """Create missing policy-nominated structures in parallel and return their paths.

    ``target_to_foldx_mutation`` maps an absolute reverse-mutation key to the
    mutation string that transforms the current start sequence into that target.
    The key determines cache identity; the FoldX string determines the actual
    structural edit.
    """
    if workers <= 0:
        raise ValueError("FoldX workers must be positive")
    base_pdb_path = ensure_repaired_structure(
        wt_pdb_path,
        fixed_pdb_path,
        foldx_bin=foldx_bin,
    )
    _ensure_repaired_parent_cache(mutated_dir, base_pdb_path)
    output_paths = {
        target_key: target_pdb_path(mutated_dir, target_key)
        for target_key in target_to_foldx_mutation
    }
    pending = [
        (target_key, target_to_foldx_mutation[target_key], output_paths[target_key])
        for target_key in target_to_foldx_mutation
        if not os.path.exists(output_paths[target_key])
    ]
    if not pending:
        return output_paths

    failures = []
    max_workers = min(workers, len(pending))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_target = {
            executor.submit(
                build_model_from_start,
                base_pdb_path,
                mutation_info,
                output_path,
                foldx_bin,
            ): target_key
            for target_key, mutation_info, output_path in pending
        }
        for future in concurrent.futures.as_completed(future_to_target):
            target_key = future_to_target[future]
            try:
                future.result()
            except Exception as exc:
                failures.append(f"{target_key}: {type(exc).__name__}: {exc}")
    if failures:
        raise RuntimeError("FoldX failed for nominated targets:\n" + "\n".join(failures))
    return output_paths
