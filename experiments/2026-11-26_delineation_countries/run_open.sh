#!/usr/bin/env bash
# The open models on the sample, one after another on the GPU (README.md).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
X=experiments/2026-11-26_delineation_countries
F="${F:-/tmp/ftw312/env/bin/python}"
V="$HOME/.venvs/open_delineation/bin/python"
env -u PYTHONPATH "$F" $X/predict_ftw.py
env -u PYTHONPATH "$V" $X/predict_delany.py
poetry run python $X/predict_utae.py --stage final --fold 1 --norm pastis
poetry run python $X/fuse.py
echo "=== open models done"
