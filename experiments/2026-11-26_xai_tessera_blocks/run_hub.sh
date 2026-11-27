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
N=$(python3 -c "import json; print(len(json.load(open('$EXP/sample/chips.json'))))")
for attempt in 1 2 3 4 5; do                      # a chip already written is skipped, so a rerun resumes
    poetry run python $EXP/blocks.py >> $LOGS/explain.log 2>&1
    DONE=$(ls results/2026-11-26_xai_tessera_blocks/EE_2021_rev/explain/*.npz 2>/dev/null | grep -vc part)
    echo "[hub $(date +%m-%d\ %H:%M:%S)] attempt $attempt: $DONE of $N chips"
    [ "$DONE" -ge "$N" ] && break
    sleep 120
done
[ "$DONE" -ge "$N" ] || { echo "explain incomplete after 5 attempts (log $LOGS/explain.log)"; exit 1; }
echo "[hub $(date +%m-%d\ %H:%M:%S)] compare"
CUDA_VISIBLE_DEVICES="" poetry run python $EXP/compare.py > $LOGS/compare.log 2>&1 || { echo "compare failed"; exit 1; }
echo "[hub $(date +%m-%d\ %H:%M:%S)] all done"
