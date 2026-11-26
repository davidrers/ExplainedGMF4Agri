#!/usr/bin/env bash
# Scoring and report once the predictions are in (README.md): every open-model file rescored with the three truths
# as soon as the open models are done (CPU, while the arms use the GPU), then the arms, then the report.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
X=experiments/2026-11-26_delineation_countries
L=results/2026-11-26_delineation_countries/EU18_2021_delin/logs
until grep -q "=== open models done" $L/run_open.log; do grep -q Traceback $L/run_open.log && exit 1; sleep 60; done
poetry run python $X/score.py --only prue_b7 v2_multi v2_single delany utae --force --workers 12 || true   # the
# summary at its end may fail while arm files are missing; the per-chip counts are what matters here
until grep -q "=== arms done" $L/run_arms.log; do grep -q Traceback $L/run_arms.log && exit 1; sleep 60; done
poetry run python $X/score.py --only arm_ --workers 24
poetry run python $X/score.py --workers 24
poetry run python $X/report.py
echo "=== finished"
