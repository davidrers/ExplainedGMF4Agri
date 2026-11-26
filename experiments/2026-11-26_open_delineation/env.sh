#!/usr/bin/env bash
# The separate environment of Delineate Anything and ftw-tools, outside the repository and never the Poetry one:
# ultralytics brings opencv-python, and an OpenCV in the Poetry environment is what once broke `import terratorch`
# on the hub (CLAUDE.md). The image's PYTHONPATH (Spark's Python 3.8 trees) is unset for every call.
#
#     bash experiments/2026-11-26_open_delineation/env.sh          # creates ~/.venvs/open_delineation
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
V="${V:-$HOME/.venvs/open_delineation}"
"$HOME/.pyenv/versions/3.11.15/bin/python" -m venv "$V"
pip() { env -u PYTHONPATH "$V/bin/pip" "$@"; }
pip install -q --upgrade pip
pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128   # the Poetry env's
pip install -q ultralytics==8.3.148 rasterio==1.4.3 huggingface-hub shapely scipy numba==0.62.1 tqdm pyyaml  # Delineate Anything's pins
pip install -q ftw-tools                                                                                 # only for check_wiring.py
pip freeze > "$HERE/reference/open_delineation_freeze.txt"
