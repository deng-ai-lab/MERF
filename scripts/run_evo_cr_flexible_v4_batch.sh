#!/usr/bin/env bash
# Batch v4 evolution from every start point in a CSV.
#
# Usage:
#   bash scripts/run_evo_cr_flexible_v4_batch.sh [start_points.csv] [log_root] [gpu_idx] [run_name]
#
# With no arguments it uses the H1 medium-v4 CSV below.  Each row must contain
# pdb_id and reverse_mutation.  If pdb_path is present, its parent start folder
# is used to locate the matching *_evo.csv; otherwise CR_evo_esmfold2 is used.

set -euo pipefail

PROJECT_DIR="/home/dataset-local/projects_dir/MERF"
PYTHON_BIN="/home/dataset-local/anaconda3/envs/merf/bin/python"
EVO_SCRIPT="${PROJECT_DIR}/scripts/evo_cr_flexible_v4.py"
DEFAULT_START_POINTS_CSV="${PROJECT_DIR}/data/CR/cr6261_h1_medium_start_point_v4.csv"
DEFAULT_LOG_ROOT="${PROJECT_DIR}/analysis/0727analysis_1/evolution_logs/v4_5outer6inner_h1_medium50"
DEFAULT_RUN_NAME="v4_5outer6inner_batch"
DEFAULT_EVO_ROOT="${PROJECT_DIR}/data/CR_evo_esmfold2"

START_POINTS_CSV="${1:-${DEFAULT_START_POINTS_CSV}}"
LOG_ROOT="${2:-${DEFAULT_LOG_ROOT}}"
GPU_IDX="${3:-0}"
RUN_NAME="${4:-${DEFAULT_RUN_NAME}}"

if [[ ! -f "${START_POINTS_CSV}" ]]; then
    echo "Start-point CSV not found: ${START_POINTS_CSV}" >&2
    exit 1
fi
if [[ ! -f "${EVO_SCRIPT}" ]]; then
    echo "Evolution script not found: ${EVO_SCRIPT}" >&2
    exit 1
fi

mkdir -p "${LOG_ROOT}/batch_runner_logs"

while IFS=$'\t' read -r DATASET START_KEY START_EVO_CSV START_ID; do
    if [[ -z "${DATASET}" || -z "${START_KEY}" || -z "${START_EVO_CSV}" ]]; then
        echo "Invalid CSV row: dataset=${DATASET}, mutation=${START_KEY}, evo_csv=${START_EVO_CSV}" >&2
        exit 1
    fi
    if [[ ! -f "${START_EVO_CSV}" ]]; then
        echo "Missing start evolution CSV for ${START_ID}: ${START_EVO_CSV}" >&2
        exit 1
    fi

    # A legacy final_summary.csv was produced from the all-epoch reward cache.
    # Only a summary explicitly marked as the last completed policy top-k can
    # suppress a rerun of this start point.
    if "${PYTHON_BIN}" -c '
import csv
import os
import sys

log_root, start_id = sys.argv[1:]
for root, _, files in os.walk(log_root):
    if os.path.basename(root) != start_id or "final_summary.csv" not in files:
        continue
    with open(os.path.join(root, "final_summary.csv"), newline="") as handle:
        row = next(csv.DictReader(handle), None)
    if row and row.get("final_selection_source") == "last_completed_epoch_policy_topk":
        sys.exit(0)
sys.exit(1)
' "${LOG_ROOT}" "${START_ID}"; then
        echo "SKIP completed: ${START_ID}"
        continue
    fi

    RUN_LOG="${LOG_ROOT}/batch_runner_logs/${START_ID}.log"
    echo "START ${START_ID}"
    CUDA_VISIBLE_DEVICES="${GPU_IDX}" "${PYTHON_BIN}" "${EVO_SCRIPT}" \
        --dataset "${DATASET}" \
        --start_evo_csv "${START_EVO_CSV}" \
        --log_root "${LOG_ROOT}" \
        --run_name "${RUN_NAME}" \
        --gpu_idx 0 \
        --total_epochs 5 \
        --rollout_samples 8 \
        --inner_epochs 6 \
        --lr 5e-6 \
        --sample_temperature 1.5 \
        --train_mode finetune \
        --elite_mode hamming \
        --elite_loss_weight 0.2 \
        --elite_top_k 2 \
        --elite_min_hamming 2 \
        --kl_coeff 0.02 \
        --uncertainty_weight 0 \
        --policy_eval_top_k 8 \
        --policy_eval_interval 5 \
        --foldx_workers 8 \
        --repair_start true \
        >"${RUN_LOG}" 2>&1
    echo "DONE ${START_ID}"
done < <(
    "${PYTHON_BIN}" -c '
import csv
import os
import sys

csv_path, evo_root = sys.argv[1:]
with open(csv_path, newline="") as handle:
    reader = csv.DictReader(handle)
    required = {"pdb_id", "reverse_mutation"}
    missing = required - set(reader.fieldnames or [])
    if missing:
        raise ValueError("Missing CSV columns: " + ", ".join(sorted(missing)))
    for row in reader:
        dataset = row["pdb_id"].strip()
        mutation = row["reverse_mutation"].strip()
        start_id = dataset + "_" + mutation
        pdb_path = (row.get("pdb_path") or "").strip()
        if pdb_path:
            start_dir = os.path.dirname(os.path.dirname(pdb_path))
        else:
            start_dir = os.path.join(evo_root, start_id)
        start_evo_csv = os.path.join(start_dir, start_id + "_evo.csv")
        print("\t".join([dataset, mutation, start_evo_csv, start_id]))
' "${START_POINTS_CSV}" "${DEFAULT_EVO_ROOT}"
)
