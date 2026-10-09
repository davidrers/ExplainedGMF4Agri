#!/usr/bin/env bash
# Bring an experiment's results back from the cluster to the JupyterHub:
#
# The first argument is the experiment's name field, which names its results folder.
#
#     bash scripts/cluster/pull_results.sh kshot                  # every chip set
#     bash scripts/cluster/pull_results.sh kshot EE_2021          # one chip set
#     DEST=results/_cluster bash scripts/cluster/pull_results.sh kshot EE_2021_mini   # elsewhere
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=${HOST:-utwente-hpc}
REMOTE=${REMOTE:-ExplainedGMF4Agri}
EXPERIMENT=${1:?experiment name, e.g. kshot}
SUB="$EXPERIMENT${2:+/$2}"
DEST="${DEST:-results}/$SUB"
mkdir -p "$DEST"
rsync -a --info=progress2 "$HOST:$REMOTE/results/$SUB/" "$DEST/"
