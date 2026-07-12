#!/bin/bash

LOG_DIR=${LOG_DIR:-/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_06_09_16_56_32}
PYTHON=${PYTHON:-/home/dataset-local/anaconda3/envs/merf/bin/python}
GPU_IDX=${GPU_IDX:-0}
EPOCH=${EPOCH:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-16}
OUTPUT_DIR=${OUTPUT_DIR:-/home/dataset-local/projects_dir/MERF/analysis/0708analysis_1/results}

# cr6261的4个，immatured，medium * 3，hard * 0
EVO_FOLDER_LIST=(
    "/home/dataset-local/projects_dir/MERF/data/CR_evo/cr6261_h1_PA28T,RA30S,TA58A,KA59N,PA62Q,DA74K,FA75S,AA76T,GA77S,VA79A,VA104L"
    "/home/dataset-local/projects_dir/MERF/data/CR_evo/cr6261_h1_PA28T,RA30S,KA59N,PA62Q,VA79A"
    "/home/dataset-local/projects_dir/MERF/data/CR_evo/cr6261_h1_PA28T,RA30S,KA59N"
    "/home/dataset-local/projects_dir/MERF/data/CR_evo/cr6261_h1_RA30S,TA58A,KA59N"
)

# cr6261_h9的4个
# EVO_FOLDER_LIST=(
# 
# )

STATUS=0
for EVO_FOLDER in "${EVO_FOLDER_LIST[@]}"; do
    echo "Evaluating EVO_FOLDER: ${EVO_FOLDER}"
    if ! "${PYTHON}" scripts/test_ddg_pred_cr_start_point.py \
        --evo_folder "${EVO_FOLDER}" \
        --log_dir "${LOG_DIR}" \
        --gpu_idx "${GPU_IDX}" \
        --checkpoint_epoch "${EPOCH}" \
        --batch_size "${BATCH_SIZE}" \
        --num_workers "${NUM_WORKERS}" \
        --output_dir "${OUTPUT_DIR}"; then
        echo "Failed EVO_FOLDER: ${EVO_FOLDER}" >&2
        STATUS=1
    fi
done

exit "${STATUS}"
