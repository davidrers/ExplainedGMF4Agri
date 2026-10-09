#!/usr/bin/env bash
# Install THOR and its TerraTorch extension into the Poetry environment, outside the lock.
#
#     bash scripts/env/install_thor.sh
#
# THOR (https://github.com/FM4CS/THOR) registers its backbones in TerraTorch through
# thor_terratorch_ext (https://github.com/FM4CS/thor_terratorch_ext). Neither can go through
# Poetry: the extension pins terratorch==1.2.5 and THOR pins timm<=1.0.15, against
# terratorch 1.2.11 and timm 1.0.29 in poetry.lock. Both run on the locked versions, which
# tests/test_thor_backbone.py checks, so they are installed here without their dependency
# pins, at fixed commits. Their remaining runtime dependencies are already in the lock,
# except fire, which THOR imports at package level.
#
# `poetry install` leaves these packages in place; `poetry sync` or recreating the environment
# removes them, and this script must then run again.
set -euo pipefail

THOR_REV=7ded3cea673ac21fe2bcadf9f3d9d4506eb6ab5f    # main, 21 May 2026, release 1.0.3
EXT_REV=00223bb70dac06d60aac38347afb54e32a3aabf7    # main, 17 Jun 2026, release 0.3.0
FIRE_VERSION=0.7.1

export PATH="$HOME/.local/bin:$PATH"
cd "$(dirname "$0")/../.."

poetry run python -m pip install --no-deps \
    "fire==${FIRE_VERSION}" \
    "thor @ git+https://github.com/FM4CS/THOR@${THOR_REV}" \
    "thor-terratorch-ext @ git+https://github.com/FM4CS/thor_terratorch_ext@${EXT_REV}"

poetry run python -c "import thor_terratorch_ext; from terratorch import BACKBONE_REGISTRY; \
print(sorted(b for b in BACKBONE_REGISTRY if 'thor_v1' in b))"
