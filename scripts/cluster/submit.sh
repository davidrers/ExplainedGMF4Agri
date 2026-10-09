#!/usr/bin/env bash
# Submit one Slurm job per arm of an experiment on one chip set, from the repository root on the cluster's
# login node. The experiment is a main-workflow spec by name, or an experiment folder's spec by path:
#
#     bash scripts/cluster/submit.sh kshot EE_2021                                    # configs/experiments/kshot.yaml
#     bash scripts/cluster/submit.sh kshot EE_2021_mini tessera_v1                    # some arms
#     bash scripts/cluster/submit.sh experiments/<date>_<name>/experiment.yaml EE_2021_mini
#     TIME=0-04:00:00 bash scripts/cluster/submit.sh kshot EE_2021_mini               # a shorter limit
#     DRY_RUN=1 bash scripts/cluster/submit.sh kshot EE_2021                         # print the sbatch commands only
#
# Jobs run scripts/run_kshot.py, or the experiment folder's own run.py when it has one (an experiment that
# changes code keeps its copy there); RUNNER=<script> overrides either. Logs go to
# results/<name>/<chips>/logs/<arm>_<job id>.log, where <name> is the experiment's name.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
EXPERIMENT=${1:?experiment name under configs/experiments, or the path of an experiment.yaml}
CHIPS=${2:?chip set name under data/eurocrops_chips, e.g. EE_2021}
shift 2
ARMS=("$@")
if [ -f "$EXPERIMENT" ]; then SPEC="$EXPERIMENT"; else SPEC="configs/experiments/$EXPERIMENT.yaml"; fi
[ -f "$SPEC" ] || { echo "no $SPEC" >&2; exit 1; }
if [ -z "${RUNNER:-}" ]; then
  if [ -f "$(dirname "$SPEC")/run.py" ]; then RUNNER="$(dirname "$SPEC")/run.py"; else RUNNER=scripts/run_kshot.py; fi
fi
[ -f "$RUNNER" ] || { echo "no $RUNNER" >&2; exit 1; }
read -r NAME < <(poetry run python -c 'import sys; sys.path.insert(0, "src")
from gfm4agri.pipeline.config import load_experiment; print(load_experiment(sys.argv[1])["name"])' "$SPEC")
if [ ${#ARMS[@]} -eq 0 ]; then
  mapfile -t ARMS < <(poetry run python -c 'import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["arms"]))' "$SPEC")
fi
[ -d "data/eurocrops_chips/$CHIPS" ] || { echo "no data/eurocrops_chips/$CHIPS on this machine" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || echo "warning: local changes; results will record a dirty tree" >&2
TIME=${TIME:-2-00:00:00}; CPUS=${CPUS:-32}; MEM=${MEM:-120G}
LOGS="results/$NAME/$CHIPS/logs"
mkdir -p "$LOGS"
echo "experiment $NAME from $SPEC, runner $RUNNER, chips $CHIPS"
for arm in "${ARMS[@]}"; do
  cmd=(sbatch -J "$NAME-$arm" -c "$CPUS" --mem="$MEM" -t "$TIME" -o "$LOGS/${arm}_%j.log"
       --export="ALL,SPEC=$SPEC,RUNNER=$RUNNER,CHIPS=$CHIPS,ARM=$arm" scripts/cluster/kshot.sbatch)
  if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
done
