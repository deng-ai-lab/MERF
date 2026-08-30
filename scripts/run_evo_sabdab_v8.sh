#!/bin/sh
# 从任意当前工作目录启动正式的 SAbDab v8 入口；其余参数原样传递。
set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
PROJECT_DIR=$(dirname "$SCRIPT_DIR")
MERF_PYTHON=${MERF_PYTHON:-/home/dataset-local/anaconda3/envs/merf/bin/python}

cd "$PROJECT_DIR"
exec "$MERF_PYTHON" "$SCRIPT_DIR/evo_sabdab_v8.py" "$@"
