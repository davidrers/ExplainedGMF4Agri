#!/bin/bash -l
# One arm of an experiment on one chip set, as a Slurm job on the UT HPC itc-gpu partition. Submitted by
# scripts/cluster/submit.sh, which sets the name, log, CPUs, memory and time, and passes the experiment spec,
# the entry script (scripts/run_kshot.py, or an experiment folder's run.py), the chip set and the arm:
#
#     sbatch --export=ALL,SPEC=configs/experiments/kshot.yaml,RUNNER=scripts/run_kshot.py,CHIPS=EE_2021,ARM=terramind_v1_large scripts/cluster/kshot.sbatch
#
# A feature cache found under data/embeddings/ is used as it is; a missing one is computed into the
# node's local NVMe, /local/$SLURM_JOB_ID, and deleted when the job ends, fails or is cancelled.
#SBATCH -A itc-tech
#SBATCH -q research
#SBATCH -p itc-gpu
#SBATCH --gres=gpu:1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HTTP_PROXY=http://proxy.utwente.nl:3128 HTTPS_PROXY=http://proxy.utwente.nl:3128
export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY
export PATH="$HOME/.local/bin:$PATH"
SCRATCH="/local/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
trap 'exit 143' TERM

RUNNER=${RUNNER:-scripts/run_kshot.py}
echo "=== $(date '+%F %T') job $SLURM_JOB_ID on $(hostname -s): $SPEC $CHIPS $ARM via $RUNNER, commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
poetry run python -u "$RUNNER" -e "$SPEC" \
  --chips "data/eurocrops_chips/$CHIPS" --arms "$ARM" --machine cluster --scratch "$SCRATCH"
status=$?
echo "=== $(date '+%F %T') exit $status"
exit $status
