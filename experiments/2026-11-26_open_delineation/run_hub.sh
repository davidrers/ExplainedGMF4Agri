#!/usr/bin/env bash
# The whole experiment on the hub, in order (README.md). Every step skips nothing on its own, so rerun single steps
# by hand when only one changes. The U-TAE fold and normalisation of the final stage are those report.py prints as
# chosen on the validation chips after the select stage (FOLD, NORM below).
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
X=experiments/2026-11-26_open_delineation
V="$HOME/.venvs/open_delineation/bin/python"
F="${F:-/tmp/ftw312/env/bin/python}"
FOLD="${FOLD:-1}"            # chosen on val, 26 November 2026
NORM="${NORM:-pastis}"

bash $X/fetch_weights.sh
[ -x "$V" ] || bash $X/env.sh
[ -x "$F" ] || D=/tmp/ftw312 bash $X/env_ftw312.sh
poetry run python $X/ftw_overlap.py
poetry run python $X/ftw_labels.py
env -u PYTHONPATH "$F" $X/predict_ftw.py
poetry run python $X/predict_utae.py --stage select
poetry run python $X/score.py --only utae_paps
poetry run python $X/report.py --select-only          # prints the U-TAE fold and normalisation chosen on val
poetry run python $X/predict_utae.py --stage final --fold "$FOLD" --norm "$NORM"
env -u PYTHONPATH "$V" $X/predict_delany.py
poetry run python $X/fuse.py
poetry run python $X/score.py
poetry run python $X/check_wiring.py utae
env -u PYTHONPATH "$F" $X/check_wiring.py ftw
env -u PYTHONPATH "$F" $X/check_wiring.py delany
poetry run python $X/report.py
