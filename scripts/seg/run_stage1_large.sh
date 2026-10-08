#!/usr/bin/env bash
# Stage 1 for the large encoders on the full Estonian chip set: TerraMind v1 large, then
# Prithvi-EO-2.0 600M TL, one after the other on the single GPU. Each run resumes from the
# chips already cached, so the chain can be restarted after an interruption.
#
#   setsid nohup bash scripts/seg/run_stage1_large.sh > data/embeddings/stage1_EE_2021.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="$(poetry env info --path 2>/dev/null)/bin/python"
[ -x "$PY" ] || PY="$HOME/.cache/pypoetry/virtualenvs/gfm4agri-7U2rqC_9-py3.11/bin/python"

run() {  # config, batch size
  echo "=== $(date +%H:%M) $1 (batch $2)"
  "$PY" -u scripts/seg/encode.py -c "$1" --batch-size "$2" --num-workers 8
  echo "=== $(date +%H:%M) $1 exit $?"
}

run configs/seg/terramind_v1_large_ee.yaml 4
run configs/seg/prithvi_eo_v2_600_tl_ee.yaml 2
echo "=== $(date +%H:%M) stage 1 chain finished"
