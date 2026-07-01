#!/usr/bin/env python
"""Shared helpers for the plectin1 peptide mutation pipeline."""

from __future__ import annotations

import csv
import shlex
from collections import OrderedDict
from pathlib import Path
from typing import Iterable

AA20 = "ACDEFGHIKLMNPQRSTVWY"

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
AA1_TO_AA3 = {value: key for key, value in AA3_TO_AA1.items()}


def _to_float(value: str, default: float = 0.0) -> float:
    if value in {".", "?"}:
        return default
    return float(value)


def _to_int(value: str, default: int = 0) -> int:
    if value in {".", "?"}:
        return default
    return int(float(value))


def _parse_pdb_atom_line(line: str) -> dict | None:
    try:
        x = float(line[30:38])
        y = float(line[38:46])
        z = float(line[46:54])
    except ValueError:
        return None

    return {
        "serial": _to_int(line[6:11].strip(), 0),
        "atom": line[12:16].strip(),
        "alt": line[16].strip() or " ",
        "resname": line[17:20].strip().upper(),
        "chain": line[21].strip(),
        "resseq": _to_int(line[22:26].strip(), 0),
        "icode": line[26].strip() or " ",
        "x": x,
        "y": y,
        "z": z,
        "occ": _to_float(line[54:60].strip(), 1.0),
        "bfac": _to_float(line[60:66].strip(), 0.0),
        "element": (line[76:78].strip() or line[12:16].strip()[0]).upper(),
    }


def _parse_mmcif_atom_rows(path: Path) -> list[dict]:
    records: list[dict] = []
    fields: list[str] = []
    in_atom_loop = False

    with path.open() as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "loop_":
                fields = []
                in_atom_loop = False
                continue
            if line.startswith("_atom_site."):
                fields.append(line)
                in_atom_loop = True
                continue
            if line.startswith("_"):
                in_atom_loop = False
                fields = []
                continue
            if line == "#":
                in_atom_loop = False
                fields = []
                continue
            if not in_atom_loop or not fields:
                continue
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            parts = shlex.split(line)
            if len(parts) < len(fields):
                continue
            row = {field.split(".", 1)[1]: value for field, value in zip(fields, parts)}
            resname = row.get("label_comp_id", "").upper()
            if resname not in AA3_TO_AA1:
                continue
            chain = row.get("auth_asym_id") or row.get("label_asym_id") or ""
            if chain in {".", "?"}:
                chain = row.get("label_asym_id", "")
            resseq = row.get("auth_seq_id") or row.get("label_seq_id") or "0"
            icode = row.get("pdbx_PDB_ins_code", " ")
            records.append(
                {
                    "serial": _to_int(row.get("id", "0")),
                    "atom": row.get("label_atom_id", "").strip(),
                    "alt": " " if row.get("label_alt_id", ".") in {".", "?"} else row.get("label_alt_id", " "),
                    "resname": resname,
                    "chain": chain,
                    "resseq": _to_int(resseq),
                    "icode": " " if icode in {".", "?"} else icode[:1],
                    "x": _to_float(row.get("Cartn_x", "0")),
                    "y": _to_float(row.get("Cartn_y", "0")),
                    "z": _to_float(row.get("Cartn_z", "0")),
                    "occ": _to_float(row.get("occupancy", "1"), 1.0),
                    "bfac": _to_float(row.get("B_iso_or_equiv", "0"), 0.0),
                    "element": row.get("type_symbol", "").upper(),
                }
            )
    return records


def read_structure_records(path: str | Path) -> list[dict]:
    path = Path(path)
    pdb_records: list[dict] = []
    with path.open() as handle:
        for line in handle:
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue
            record = _parse_pdb_atom_line(line)
            if record is not None and record["resname"] in AA3_TO_AA1:
                pdb_records.append(record)
    if pdb_records:
        return pdb_records

    records = _parse_mmcif_atom_rows(path)
    if not records:
        raise ValueError(f"No protein ATOM records parsed from {path}")
    return records


def _format_atom_name(atom: str, element: str) -> str:
    atom = atom[:4]
    if len(atom) < 4 and len(element) == 1:
        return f" {atom:<3}"
    return f"{atom:<4}"


def write_pdb(records: Iterable[dict], out_path: str | Path) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as handle:
        serial = 1
        for rec in records:
            atom_name = _format_atom_name(str(rec["atom"]), str(rec.get("element", "")))
            chain = str(rec["chain"] or "A")[:1]
            icode = str(rec.get("icode", " ") or " ")[:1]
            element = str(rec.get("element", "") or str(rec["atom"])[0]).strip().upper()[:2]
            handle.write(
                f"ATOM  {serial:5d} {atom_name}{str(rec.get('alt', ' ') or ' ')[:1]}"
                f"{str(rec['resname'])[:3]:>3} {chain}{int(rec['resseq']):4d}{icode}   "
                f"{float(rec['x']):8.3f}{float(rec['y']):8.3f}{float(rec['z']):8.3f}"
                f"{float(rec.get('occ', 1.0)):6.2f}{float(rec.get('bfac', 0.0)):6.2f}"
                f"          {element:>2}\n"
            )
            serial += 1
        handle.write("END\n")
    return out_path


def ensure_standard_pdb(input_path: str | Path, output_path: str | Path) -> Path:
    records = read_structure_records(input_path)
    return write_pdb(records, output_path)


def find_structure_file(structure_dir: str | Path, structure_id: str) -> Path:
    structure_dir = Path(structure_dir)
    for suffix in (".pdb", ".cif", ".mmcif"):
        path = structure_dir / f"{structure_id}{suffix}"
        if path.exists():
            return path
    raise FileNotFoundError(f"No structure file found for {structure_id} in {structure_dir}")


def chain_ids(records: Iterable[dict]) -> list[str]:
    ids = []
    seen = set()
    for rec in records:
        chain = rec["chain"]
        if chain not in seen:
            seen.add(chain)
            ids.append(chain)
    return ids


def chain_residues(records: Iterable[dict], chain: str) -> list[dict]:
    residues: OrderedDict[tuple, dict] = OrderedDict()
    for rec in records:
        if rec["chain"] != chain:
            continue
        key = (rec["chain"], int(rec["resseq"]), rec.get("icode", " "))
        if key in residues:
            continue
        aa = AA3_TO_AA1.get(rec["resname"])
        if aa:
            residues[key] = {
                "chain": rec["chain"],
                "resseq": int(rec["resseq"]),
                "icode": rec.get("icode", " "),
                "resname": rec["resname"],
                "aa": aa,
            }
    if not residues:
        raise ValueError(f"No standard residues found for chain {chain}")
    return list(residues.values())


def sequence_from_residues(residues: Iterable[dict]) -> str:
    return "".join(res["aa"] for res in residues)


def all_single_mutations(structure_id: str, residues: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for seq_pos, res in enumerate(residues, start=1):
        wt = res["aa"]
        for mt in AA20:
            if mt == wt:
                continue
            rows.append(
                {
                    "structure": structure_id,
                    "chain": res["chain"],
                    "seq_pos": seq_pos,
                    "pdb_resseq": res["resseq"],
                    "icode": res["icode"].strip(),
                    "wt": wt,
                    "mt": mt,
                    "mutation": f"{wt}{seq_pos}{mt}",
                    "pdb_mutation": f"{wt}{res['chain']}{res['resseq']}{mt}",
                }
            )
    return rows


def mutate_sequence(sequence: str, seq_pos: int, mt: str) -> str:
    idx = seq_pos - 1
    return sequence[:idx] + mt + sequence[idx + 1 :]


def write_mutant_pdb(records: list[dict], chain: str, resseq: int, icode: str, mt: str, out_path: str | Path) -> Path:
    mt_resname = AA1_TO_AA3[mt]
    norm_icode = icode or " "
    mutated: list[dict] = []
    for rec in records:
        new_rec = dict(rec)
        if rec["chain"] == chain and int(rec["resseq"]) == int(resseq) and (rec.get("icode", " ") or " ") == norm_icode:
            new_rec["resname"] = mt_resname
        mutated.append(new_rec)
    return write_pdb(mutated, out_path)


def write_csv(path: str | Path, rows: list[dict], fieldnames: list[str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None and rows:
        fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or [])
        writer.writeheader()
        writer.writerows(rows)
    return path
