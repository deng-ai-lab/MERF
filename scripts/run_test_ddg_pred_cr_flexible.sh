#!/bin/bash

LOG_DIR="/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_06_09_16_56_32"
PYTHON="/home/dataset-local/anaconda3/envs/merf/bin/python"
DATASET=${1:-cr_6261_h1}
START_MUTANT=${2:?Please provide a comma-separated reverse mutation combination}
GPU_IDX=${GPU_IDX:-1}
EPOCH=${EPOCH:-200}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_WORKERS=${NUM_WORKERS:-16}
OUTPUT_DIR=${OUTPUT_DIR:-/home/dataset-local/projects_dir/MERF/analysis/0707analysis_1/results}

${PYTHON} scripts/test_ddg_pred_cr_flexible.py \
    --log_dir ${LOG_DIR} \
    --dataset ${DATASET} \
    --start_mutant "${START_MUTANT}" \
    --gpu_idx ${GPU_IDX} \
    --checkpoint_epoch ${EPOCH} \
    --batch_size ${BATCH_SIZE} \
    --num_workers ${NUM_WORKERS} \
    --output_dir ${OUTPUT_DIR}
