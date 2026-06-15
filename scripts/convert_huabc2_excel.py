import argparse
import csv
import math
import os
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile


NAMESPACE = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
MUTATION_RE = re.compile(r"^([A-Z])(\d+)([A-Z])$")


def column_index(column_name):
    index = 0
    for char in column_name:
        index = index * 26 + ord(char) - ord("A") + 1
    return index


def read_xlsx_first_sheet(path):
    with ZipFile(path) as archive:
        shared_strings = []
        shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
        for item in shared_root.findall("a:si", NAMESPACE):
            shared_strings.append("".join(text.text or "" for text in item.findall(".//a:t", NAMESPACE)))

        sheet_root = ET.fromstring(archive.read("xl/worksheets/sheet1.xml"))
        rows = []
        for row in sheet_root.findall(".//a:sheetData/a:row", NAMESPACE):
            row_values = {}
            for cell in row.findall("a:c", NAMESPACE):
                ref = cell.attrib.get("r", "")
                match = re.match(r"([A-Z]+)", ref)
                if match is None:
                    continue
                col = match.group(1)
                value_node = cell.find("a:v", NAMESPACE)
                if value_node is None:
                    value = ""
                elif cell.attrib.get("t") == "s":
                    value = shared_strings[int(value_node.text)]
                else:
                    value = value_node.text
                row_values[col] = value
            rows.append(row_values)

    if not rows:
        return []

    columns = sorted({col for row in rows for col in row}, key=column_index)
    header = [rows[0].get(col, "") for col in columns]
    records = []
    for row in rows[1:]:
        records.append({name: row.get(col, "") for name, col in zip(header, columns)})
    return records


def convert_mutation(raw_mutation):
    converted = []
    chain_parts = raw_mutation.split(":")
    if len(chain_parts) != 2:
        raise ValueError(f"Expected heavy:light mutation format, got {raw_mutation!r}")

    for chain_id, chain_mutations in zip(("A", "B"), chain_parts):
        if chain_mutations == "WT":
            continue
        for mutation in chain_mutations.split("/"):
            if mutation == "WT":
                continue
            match = MUTATION_RE.match(mutation)
            if match is None:
                raise ValueError(f"Unsupported mutation token {mutation!r} in {raw_mutation!r}")
            wt_aa, position, mut_aa = match.groups()
            converted.append(f"{wt_aa}{chain_id}{position}{mut_aa}")

    if not converted:
        return "WT"
    return ",".join(converted)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--excel_path",
        type=str,
        default="/home/dataset-local/projects_dir/MERF/data/HuABC2/aea1820_Suppl. Excel_seq16_v2.xlsx",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="/home/dataset-local/projects_dir/MERF/data/HuABC2/HuABC2_test.csv",
    )
    parser.add_argument("--pdb_id", type=str, default="HuABC2_h1.pdb")
    parser.add_argument("--skip_missing_binding", action="store_true", default=True)
    args = parser.parse_args()

    records = read_xlsx_first_sheet(args.excel_path)
    output_rows = []
    for record in records:
        raw_mutation = record["Mutations"].strip()
        binding_value = record["Fold Improvement (Binding)"].strip()
        if args.skip_missing_binding and binding_value in ("", "n/a"):
            continue
        if raw_mutation == "WT:WT":
            continue

        binding_fold_improvement = float(binding_value)
        mutation = convert_mutation(raw_mutation)
        ddg = -math.log(binding_fold_improvement)
        mut_file = f"{os.path.splitext(args.pdb_id)[0]}_{mutation}.pdb"

        output_rows.append(
            {
                "pdb_id": args.pdb_id,
                "raw_mutation": raw_mutation,
                "mutation": mutation,
                "mutant": mutation,
                "DDG": ddg,
                "binding_fold_improvement": binding_fold_improvement,
                "expression_fold_improvement": record["Fold Improvement (Expression)"],
                "origin": record["Origin"],
                "variant_id": record["Variant ID"],
                "path_mut": mut_file,
            }
        )

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    print(f"Wrote {len(output_rows)} rows to {args.output_path}")


if __name__ == "__main__":
    main()
