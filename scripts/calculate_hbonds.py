#!/usr/bin/env python3
import argparse
import re
import shutil
import subprocess
from pathlib import Path

DEFAULT_HBPLUS_BIN = Path("/home/dataset-local/software_package/hbplus/hbplus")
DEFAULT_WORK_DIR = Path("hbplus_work")

def parse_chain_from_hbplus_res_token(tok: str) -> str:
    """
    HBPlus residue token examples:
      A0052-SER
      B0170-TYR
    First character is chain ID in standard HBPlus output.
    """
    tok = tok.strip()
    if not tok:
        return ""
    return tok[0]

def is_hbond_line(line: str) -> bool:
    """
    HBPlus .hb2 data lines usually begin like:
      A0052-SER N   A0497-ASP OD2 ...
    """
    return bool(re.match(r"^\S\d{4}-[A-Za-z0-9]{3}\s+\S+\s+\S\d{4}-[A-Za-z0-9]{3}\s+\S+", line))

def count_interface_hbonds(hb2_file: Path, binder_chains, target_chains, deduplicate=False):
    binder_chains = set(binder_chains)
    target_chains = set(target_chains)

    count = 0
    records = []
    seen = set()

    with open(hb2_file, "r", errors="ignore") as f:
        for line in f:
            if not is_hbond_line(line):
                continue

            fields = line.split()
            if len(fields) < 4:
                continue

            donor_res = fields[0]
            donor_atom = fields[1]
            acceptor_res = fields[2]
            acceptor_atom = fields[3]

            donor_chain = parse_chain_from_hbplus_res_token(donor_res)
            acceptor_chain = parse_chain_from_hbplus_res_token(acceptor_res)

            cross_interface = (
                (donor_chain in binder_chains and acceptor_chain in target_chains) or
                (donor_chain in target_chains and acceptor_chain in binder_chains)
            )

            if not cross_interface:
                continue

            key = (donor_res, donor_atom, acceptor_res, acceptor_atom)
            if deduplicate:
                # 默认不建议打开；论文更可能是按 HBPlus 条目计数。
                unordered = tuple(sorted([
                    (donor_res, donor_atom),
                    (acceptor_res, acceptor_atom)
                ]))
                key = unordered

            if key in seen:
                continue
            seen.add(key)

            count += 1
            records.append((donor_res, donor_atom, acceptor_res, acceptor_atom, line.rstrip()))

    return count, records

def run_hbplus(pdb_file: Path, work_dir: Path, hbplus_bin: Path) -> Path:
    pdb_file = pdb_file.resolve()
    hbplus_bin = hbplus_bin.resolve()
    work_dir = work_dir.resolve()

    if not pdb_file.is_file():
        raise FileNotFoundError(f"PDB file not found: {pdb_file}")
    if not hbplus_bin.is_file():
        raise FileNotFoundError(f"HBPlus binary not found: {hbplus_bin}")

    work_dir.mkdir(parents=True, exist_ok=True)
    local_pdb = work_dir / pdb_file.name
    if local_pdb.resolve() != pdb_file:
        shutil.copy2(pdb_file, local_pdb)

    expected_hb2 = local_pdb.with_suffix(".hb2")
    if expected_hb2.exists():
        expected_hb2.unlink()

    result = subprocess.run(
        [str(hbplus_bin), local_pdb.name],
        cwd=work_dir,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        output = "\n".join(part for part in [result.stdout, result.stderr] if part)
        raise RuntimeError(f"HBPlus failed for {pdb_file}:\n{output}")
    if not expected_hb2.is_file():
        output = "\n".join(part for part in [result.stdout, result.stderr] if part)
        raise FileNotFoundError(f"HBPlus did not create expected file: {expected_hb2}\n{output}")

    return expected_hb2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path, help="Input .hb2 file, or .pdb file to run through HBPlus first")
    parser.add_argument("--binder", required=True, help="Binder chain IDs, e.g. A or AB")
    parser.add_argument("--target", required=True, help="Target chain IDs, e.g. BCD")
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--out", default=None)
    parser.add_argument("--hbplus-bin", type=Path, default=DEFAULT_HBPLUS_BIN)
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    args = parser.parse_args()

    hb2_file = args.input
    if args.input.suffix.lower() != ".hb2":
        hb2_file = run_hbplus(args.input, args.work_dir, args.hbplus_bin)

    count, records = count_interface_hbonds(
        hb2_file,
        binder_chains=list(args.binder),
        target_chains=list(args.target),
        deduplicate=args.deduplicate,
    )

    print(f"interface_hbonds\t{count}")

    if args.out:
        with open(args.out, "w") as w:
            w.write("donor_res\tdonor_atom\tacceptor_res\tacceptor_atom\thbplus_line\n")
            for r in records:
                w.write("\t".join(r) + "\n")

if __name__ == "__main__":
    main()
