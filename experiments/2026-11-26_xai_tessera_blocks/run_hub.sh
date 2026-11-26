#!/usr/bin/env bash
# Every chip of the sample on the hub GPU (a chip already written is skipped), then the comparison.
#
#     setsid nohup bash experiments/2026-11-26_xai_tessera_blocks/run_hub.sh > results/.../logs/run.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXP=experiments/2026-11-26_xai_tessera_blocks
LOGS=results/2026-11-26_xai_tessera_blocks/EE_2021_rev/logs
mkdir -p $LOGS
echo "[hub $(date +%m-%d\ %H:%M:%S)] explain"
poetry run python $EXP/blocks.py > $LOGS/explain.log 2>&1 || { echo "explain failed (log $LOGS/explain.log)"; exit 1; }
echo "[hub $(date +%m-%d\ %H:%M:%S)] compare"
CUDA_VISIBLE_DEVICES="" poetry run python $EXP/compare.py > $LOGS/compare.log 2>&1 || { echo "compare failed"; exit 1; }
echo "[hub $(date +%m-%d\ %H:%M:%S)] all done"
