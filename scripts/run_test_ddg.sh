#!/bin/bash

# Simple test script for ddG prediction evaluation
# Usage: bash scripts/run_test_ddg.sh

# Configuration - modify these variables as needed
LOG_DIR="/home/dataset-local/projects_dir/MERF/logs/pretrain_vm/2026_05_07_20_57_23"
# TEST_DATASETS="vm,abbind,cr_6261_h1,cr_6261_h9,cr_9114_h1,cr_9114_h3,7fae"
# TEST_DATASETS="7fae"
TEST_DATASETS="cr_9114_h1,cr_9114_h3"
GPU_IDX=0
EPOCH=150
BATCH_SIZE=128

# Run test
python scripts/test_ddg_pred.py \
    --log_dir ${LOG_DIR} \
    --test_datasets ${TEST_DATASETS} \
    --gpu_idx ${GPU_IDX} \
    --checkpoint_epoch ${EPOCH} \
    --batch_size ${BATCH_SIZE}

echo "Testing completed! Results saved to: ${LOG_DIR}"
