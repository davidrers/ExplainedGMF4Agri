#!/usr/bin/env bash
# Submit one Slurm job per arm of a K-shot experiment on one chip set, from the repository root on the
# cluster's login node:
#
#     bash scripts/cluster/submit.sh kshot EE_2021                        # every arm of the experiment
#     bash scripts/cluster/submit.sh kshot EE_2021_mini tessera_v1        # some arms
#     TIME=0-04:00:00 bash scripts/cluster/submit.sh kshot EE_2021_mini   # a shorter limit for the pilot
#     DRY_RUN=1 bash scripts/cluster/submit.sh kshot EE_2021             # print the sbatch commands only
#
# Logs go to results/<experiment>/<chips>/logs/<arm>_<job id>.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
EXPERIMENT=${1:?experiment name, e.g. kshot}
CHIPS=${2:?chip set name under data/eurocrops_chips, e.g. EE_2021}
shift 2
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  mapfile -t ARMS < <(poetry run python -c 'import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["arms"]))' \
    "configs/experiments/$EXPERIMENT.yaml")
fi
[ -d "data/eurocrops_chips/$CHIPS" ] || { echo "no data/eurocrops_chips/$CHIPS on this machine" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || echo "warning: local changes; results will record a dirty tree" >&2
TIME=${TIME:-2-00:00:00}; CPUS=${CPUS:-32}; MEM=${MEM:-120G}
LOGS="results/$EXPERIMENT/$CHIPS/logs"
mkdir -p "$LOGS"
for arm in "${ARMS[@]}"; do
  cmd=(sbatch -J "$EXPERIMENT-$arm" -c "$CPUS" --mem="$MEM" -t "$TIME" -o "$LOGS/${arm}_%j.log"
       --export="ALL,EXPERIMENT=$EXPERIMENT,CHIPS=$CHIPS,ARM=$arm" scripts/cluster/kshot.sbatch)
  if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
done
