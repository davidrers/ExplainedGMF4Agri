#!/usr/bin/env bash
# Copy data directories from the JupyterHub to the same place in the cluster's clone, resumably:
#
#     bash scripts/cluster/push_data.sh eurocrops_chips/EE_2021 eurocrops_chips/EE_2021_mini eurocrops/parquet eurocrops/vector
#
# Paths are relative to data/. Links are copied as links, so a pilot chip set stays links into its
# parent, which must be copied too. Log and pid files stay behind. Each copy is checked against the
# free space on the cluster first.
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=${HOST:-utwente-hpc}
REMOTE=${REMOTE:-ExplainedGMF4Agri}
for rel in "$@"; do
  src="data/${rel%/}"
  [ -e "$src" ] || { echo "no $src" >&2; exit 1; }
  need=$(du -sk "$src" | cut -f1)
  free=$(ssh "$HOST" "df -k --output=avail ~ | tail -1")
  echo "=== $src: $((need / 1048576)) GiB to copy, $((free / 1048576)) GiB free on $HOST"
  [ "$need" -lt "$free" ] || { echo "not enough space on $HOST for $src" >&2; exit 1; }
  ssh "$HOST" "mkdir -p '$REMOTE/$src'"
  rsync -a --partial --info=progress2 --exclude='*.log' --exclude='*.pid' "$src/" "$HOST:$REMOTE/$src/"
done
