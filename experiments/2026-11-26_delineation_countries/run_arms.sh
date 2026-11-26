#!/usr/bin/env bash
# After run_open.sh and worldcover.py: FTW with each country's own windows, then our arms (README.md). The token-grid
# arms first encode the sample into data/embeddings/<backbone>/EU18_2021_delin/ (about 6 s per chip each on the hub).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
X=experiments/2026-11-26_delineation_countries
L=results/2026-11-26_delineation_countries/EU18_2021_delin/logs
F="${F:-/tmp/ftw312/env/bin/python}"
until grep -q "=== open models done" $L/run_open.log && [ -s data/worldcover/EU18_2021_delin/worldcover.npz ]; do
    if grep -q Traceback $L/run_open.log; then echo "run_open failed"; exit 1; fi
    sleep 30
done
env -u PYTHONPATH "$F" $X/predict_ftw_own.py
for arm in unet_b7_s2s1 tessera_v2_medium terramind_v1_large_s2s1 thor_v1_large_s2s1 prithvi_eo_v2_600_tl; do
    poetry run python $X/predict_arms.py --arms $arm
done
echo "=== arms done"
