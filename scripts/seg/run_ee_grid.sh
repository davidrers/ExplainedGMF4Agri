#!/usr/bin/env bash
# Full-country Estonia, one draw and one seed per cell: TerraMind v1 large and Prithvi-EO-2.0
# 600M TL trained in two stages from their feature caches, TESSERA v1 and AlphaEarth V1 trained
# on their embedding rasters, at each label budget. Budgets run outermost, so every model has a result at 100 %
# before any runs at 20 %. A cell whose results.json exists is skipped, so the grid resumes.
#
#   EPOCHS=15 setsid nohup bash scripts/seg/run_ee_grid.sh > results/seg_cached/ee_grid.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")/../.."
PY="$(poetry env info --path 2>/dev/null)/bin/python"
[ -x "$PY" ] || PY="$HOME/.cache/pypoetry/virtualenvs/gfm4agri-7U2rqC_9-py3.11/bin/python"
EPOCHS="${EPOCHS:-15}"
PCTS="${PCTS:-100 20 5}"
DRAW="${DRAW:-0}"

for pct in $PCTS; do
  for cfg in terramind_v1_large_ee prithvi_eo_v2_600_tl_ee; do
    out="results/seg_cached/$cfg/P${pct}_draw${DRAW}_seed0/results.json"
    if [ -f "$out" ]; then echo "=== skip $cfg P$pct (done)"; continue; fi
    echo "=== $(date '+%m-%d %H:%M') $cfg P$pct two-stage, $EPOCHS epochs"
    "$PY" -u scripts/seg/fit_cached.py -c "configs/seg/$cfg.yaml" --pct "$pct" \
      --draw-seed "$DRAW" --max-epochs "$EPOCHS"
    echo "=== $(date '+%m-%d %H:%M') $cfg P$pct exit $?"
  done
  for cfg in tessera_v1_mlp_ee alphaearth_v1_mlp_ee; do
    out="results/seg/$cfg/P${pct}_draw${DRAW}_seed0/results.json"
    if [ -f "$out" ]; then echo "=== skip $cfg P$pct (done)"; continue; fi
    echo "=== $(date '+%m-%d %H:%M') $cfg P$pct, $EPOCHS epochs"
    "$PY" -u scripts/seg/train.py -c "configs/seg/$cfg.yaml" --set label_budget.mode=pct \
      "label_budget.pct=$pct" "label_budget.draw_seed=$DRAW" "trainer.max_epochs=$EPOCHS"
    echo "=== $(date '+%m-%d %H:%M') $cfg P$pct exit $?"
  done
done
echo "=== $(date '+%m-%d %H:%M') grid finished"
