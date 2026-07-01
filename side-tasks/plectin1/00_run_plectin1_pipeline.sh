#!/usr/bin/env bash
set -euo pipefail

STRUCTURE="${1:-all}"
TASK_DIR="/home/dataset-local/projects_dir/MERF/side-tasks/plectin1"
PDB_DIR="${TASK_DIR}/pdbs"
RESULTS_DIR="${TASK_DIR}/results"
MERF_PY="/home/dataset-local/anaconda3/envs/merf/bin/python"
AF2_PY="/home/dataset-local/anaconda3/envs/colabdesign-af2/bin/python"
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH="/home/dataset-local/projects_dir/MERF:${TASK_DIR}:${PYTHONPATH:-}"

"${MERF_PY}" "${TASK_DIR}/01_nominate_mutations.py" \
  --structure "${STRUCTURE}" \
  --pdb-dir "${PDB_DIR}" \
  --output-dir "${RESULTS_DIR}" \
  --chain B \
  --gpu 2

"${MERF_PY}" "${TASK_DIR}/02_run_merf_ddg.py" \
  --structure "${STRUCTURE}" \
  --pdb-dir "${PDB_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --chain B \
  --gpu-idx 2

"${AF2_PY}" "${TASK_DIR}/03_run_af2_multimer.py" \
  --structure "${STRUCTURE}" \
  --pdb-dir "${PDB_DIR}" \
  --results-dir "${RESULTS_DIR}" \
  --chain B \
  --gpu 2

"${MERF_PY}" "${TASK_DIR}/04_merge_results.py" \
  --structure "${STRUCTURE}" \
  --results-dir "${RESULTS_DIR}"

"${MERF_PY}" "${TASK_DIR}/05_summarize_results.py" \
  --results-dir "${RESULTS_DIR}"
