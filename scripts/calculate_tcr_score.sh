#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/home/dataset-local/projects_dir/MERF"
PMTNET_DIR="${PROJECT_DIR}/baselines/pMTnet"
CONDA_SH="/home/dataset-local/anaconda3/etc/profile.d/conda.sh"
PMTNET_ENV="/home/dataset-local/anaconda3/envs/pmtnet"
PMTNET_PYTHON="${PMTNET_ENV}/bin/python"

SOURCE_CSV="${PROJECT_DIR}/data/TCR/f8_evo.csv"
PMTNET_INPUT_CSV="${PROJECT_DIR}/data/TCR/f8_evo_pmtnet_input.csv"
OUTPUT_DIR="${PROJECT_DIR}/data/TCR/f8_evo_pmtnet_output"
OUTPUT_CSV="${OUTPUT_DIR}/prediction.csv"
OUTPUT_LOG="${OUTPUT_DIR}/output.log"

PEPTIDE="VLDFAPPGA"
HLA_ALLELE_ORIGINAL="HLA-A* 02:01"
# pMTnet's HLA library stores names like A*02:01:01:01 and matches with startswith().
# The original "HLA-A* 02:01" form is not directly supported, so use the compatible prefix.
HLA_ALLELE_PMTNET="A*02:01"

echo "Switching to pMTnet python environment:"
echo "  conda activate ${PMTNET_ENV}"
source "${CONDA_SH}"
conda activate "${PMTNET_ENV}"
if [[ "$(command -v python)" != "${PMTNET_PYTHON}" ]]; then
  echo "Expected python at ${PMTNET_PYTHON}, got $(command -v python)" >&2
  exit 1
fi
echo "Active python: $(command -v python)"
python --version

mkdir -p "${OUTPUT_DIR}"

echo "Building pMTnet input:"
echo "  source: ${SOURCE_CSV}"
echo "  target: ${PMTNET_INPUT_CSV}"
echo "  peptide: ${PEPTIDE}"
echo "  HLA original: ${HLA_ALLELE_ORIGINAL}"
echo "  HLA for pMTnet: ${HLA_ALLELE_PMTNET}"

SOURCE_CSV="${SOURCE_CSV}" \
PMTNET_INPUT_CSV="${PMTNET_INPUT_CSV}" \
PEPTIDE="${PEPTIDE}" \
HLA_ALLELE_PMTNET="${HLA_ALLELE_PMTNET}" \
"${PMTNET_PYTHON}" - <<'PY'
import os
import pandas as pd

source_csv = os.environ["SOURCE_CSV"]
output_csv = os.environ["PMTNET_INPUT_CSV"]
peptide = os.environ["PEPTIDE"]
hla = os.environ["HLA_ALLELE_PMTNET"]

df = pd.read_csv(source_csv)
if "cdr3" not in df.columns:
    raise KeyError(f"Expected column 'cdr3' in {source_csv}; found {list(df.columns)}")

pmtnet_input = pd.DataFrame({
    "CDR3": df["cdr3"].astype(str).str.strip(),
    "Antigen": peptide,
    "HLA": hla,
})
pmtnet_input = pmtnet_input.dropna()
pmtnet_input.to_csv(output_csv, index=False)
print(f"Wrote {len(pmtnet_input)} rows to {output_csv}")
PY

echo "Running pMTnet..."
cd "${PMTNET_DIR}"
"${PMTNET_PYTHON}" pMTnet.py \
  -input "${PMTNET_INPUT_CSV}" \
  -library "${PMTNET_DIR}/library" \
  -output "${OUTPUT_DIR}" \
  -output_log "${OUTPUT_LOG}"

echo "pMTnet output CSV:"
echo "  ${OUTPUT_CSV}"
echo "pMTnet log:"
echo "  ${OUTPUT_LOG}"
