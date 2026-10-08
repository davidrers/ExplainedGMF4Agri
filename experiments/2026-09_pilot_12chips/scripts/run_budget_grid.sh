#!/usr/bin/env bash
# Run the label-budget grid for every backbone, sequentially, on one GPU.
#
# The budget is a percentage of the independent training parcels of each class, so the grid
# entries are passed as label_budget.mode=pct label_budget.pct=<p>. Runs land in
# results/seg/<run_name>/P<p>_draw0_seed0/. A run whose results.json already exists is
# skipped, so the script resumes.
#
#     bash scripts/seg/run_budget_grid.sh
#     PCTS="5 100" CONFIGS="prithvi_eo_v2_600_tl_ee_pilot" bash scripts/seg/run_budget_grid.sh
set -u
cd "$(dirname "$0")/../.."
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PATH="$HOME/.local/bin:$PATH"

CONFIGS="${CONFIGS:-prithvi_eo_v2_600_tl_ee_pilot terramind_v1_large_ee_pilot tessera_v1_mlp_ee_pilot terramind_v1_small_ee_pilot prithvi_eo_v2_300_tl_ee_pilot}"
PCTS="${PCTS:-5 20 100}"
DRAW="${DRAW:-0}"

for cfg in $CONFIGS; do
  run_name=$(grep -m1 '^run_name:' "configs/seg/${cfg}.yaml" | awk '{print $2}')
  for p in $PCTS; do
    out="results/seg/${run_name}/P${p}_draw${DRAW}_seed0"
    if [ -f "${out}/results.json" ]; then
      echo "=== skip ${run_name} P${p}, already done"
      continue
    fi
    echo "=== $(date -u +%H:%M:%S) ${run_name} at ${p} % (draw ${DRAW})"
    poetry run python scripts/seg/train.py -c "configs/seg/${cfg}.yaml" \
      --set label_budget.mode=pct "label_budget.pct=${p}" "label_budget.draw_seed=${DRAW}" \
      2>&1 | grep -viE "^\s*$|it/s\]|DoesNotConformTo|warnings.warn" | tail -30
    echo "=== exit ${PIPESTATUS[0]}"
  done
done
echo "=== grid finished $(date -u +%H:%M:%S)"
