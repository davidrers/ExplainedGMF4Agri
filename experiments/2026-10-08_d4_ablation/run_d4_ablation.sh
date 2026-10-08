#!/usr/bin/env bash
# How many cached D4 variants does the augmentation need? TerraMind v1 large at K = 5 %, draw 0,
# 15 epochs, three seeds, trained from the identity only (k0), the four quarter turns (rot4) and
# all eight (all8, the workflow default). The all-eight seed 0 fit is the full-Estonia grid's own
# result, results/seg_cached/terramind_v1_large_ee/P5_draw0_seed0. Outcome, 8 October 2026: test
# Macro-F1 0.304 +- 0.030 (k0), 0.335 +- 0.049 (rot4), 0.358 +- 0.003 (all8).
# A cell whose results.json exists is skipped, so the run resumes.
#
#   setsid nohup bash scripts/seg/run_d4_ablation.sh > results/seg_cached/d4_ablation/run.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="$(poetry env info --path 2>/dev/null)/bin/python"
[ -x "$PY" ] || PY="$HOME/.cache/pypoetry/virtualenvs/gfm4agri-7U2rqC_9-py3.11/bin/python"
CFG=terramind_v1_large_ee
PCT=5
OUT=results/seg_cached/d4_ablation

run() {  # arm seed [variants...]
  local arm=$1 seed=$2; shift 2
  local res="$OUT/$arm/$CFG/P${PCT}_draw0_seed${seed}/results.json"
  if [ -f "$res" ]; then echo "=== skip $arm seed $seed (done)"; return; fi
  echo "=== $(date '+%m-%d %H:%M') $arm seed $seed"
  "$PY" -u scripts/seg/fit_cached.py -c "configs/seg/$CFG.yaml" --pct "$PCT" --draw-seed 0 \
    --max-epochs 15 --seed "$seed" --output-root "$OUT/$arm" "$@"
  echo "=== $(date '+%m-%d %H:%M') $arm seed $seed exit $?"
}

run k0 0 --train-variants k0
run k0 1 --train-variants k0
ALL8=(k0 k0f k1 k1f k2 k2f k3 k3f)
run all8 1 --train-variants "${ALL8[@]}"
run k0 2 --train-variants k0
run all8 2 --train-variants "${ALL8[@]}"
run rot4 0 --train-variants k0 k1 k2 k3
run rot4 1 --train-variants k0 k1 k2 k3
run rot4 2 --train-variants k0 k1 k2 k3
echo "=== $(date '+%m-%d %H:%M') ablation finished"
