#!/usr/bin/env bash
# The environment of FTW's current inference stack: ftw-tools at the commit read for this experiment (eba8740) needs
# Python >= 3.12 and torchgeo >= 0.10, which the hub's pyenv does not have, so uv fetches a standalone Python 3.12.
# FTW's predictions are made here (predict_ftw.py) and the two ftw-tools wiring checks run here (check_wiring.py
# ftw, delany): with segmentation-models-pytorch 0.5.0, the Poetry environment's, the EfficientNet encoders give
# 0.5 to 4 % different class maps from ftw-tools on the same input; with this stack's 0.6.0 they agree to the pixel.
# Built on local disk (fast; it does not survive a restart of the hub), outside the repository.
#
#     bash experiments/2026-11-26_open_delineation/env_ftw312.sh            # D=<dir> to choose where
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
D="${D:-/tmp/ftw312}"
"$HOME/.pyenv/versions/3.11.15/bin/python" -m venv "$D/uvtool"
env -u PYTHONPATH "$D/uvtool/bin/pip" install -q uv
export UV_PYTHON_INSTALL_DIR="$D/python" UV_CACHE_DIR="$D/cache"
uv() { env -u PYTHONPATH VIRTUAL_ENV="$D/env" "$D/uvtool/bin/uv" "$@"; }
env -u PYTHONPATH "$D/uvtool/bin/uv" python install 3.12
env -u PYTHONPATH "$D/uvtool/bin/uv" venv -p 3.12 "$D/env"
uv pip install -q torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
uv pip install -q "git+https://github.com/fieldsoftheworld/ftw-baselines@eba8740056cc105700ae8c72fbdd7ca17ae0d2e2" \
    ultralytics==8.3.148
uv pip freeze > "$HERE/reference/ftw_tools_py312_freeze.txt"
echo "python: $D/env/bin/python (run with env -u PYTHONPATH)"
