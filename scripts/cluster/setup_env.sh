#!/bin/bash -l
# One-time set-up of the Poetry environment on the UT HPC cluster, from the repository root on the
# login node (a login shell, so `module` works):
#
#     bash -l scripts/cluster/setup_env.sh
#
# Python 3.11 comes from a conda-forge environment created with the miniconda3 module (the cluster has
# no python/3.11 module), Poetry 2.2.1 from its installer, the packages from poetry.lock, and THOR outside
# the lock as on the hub. The encoder weights are downloaded here, on the login node, which has direct
# internet; compute nodes reach it only through the UT proxy. Each step skips what is in place.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY_ENV="$HOME/envs/py311"
export PATH="$HOME/.local/bin:$PATH"

module load miniconda3/25.7
[ -x "$PY_ENV/bin/python" ] || conda create -y -p "$PY_ENV" -c conda-forge --override-channels python=3.11
if ! poetry --version 2>/dev/null | grep -q "2\.2\.1"; then
  curl -sSL https://install.python-poetry.org | "$PY_ENV/bin/python" - --version 2.2.1
fi
poetry env use "$PY_ENV/bin/python"
poetry install
bash scripts/env/install_thor.sh
poetry run python scripts/cluster/prefetch_weights.py
poetry run pytest -q
echo "environment ready: $(poetry env info --path)"
