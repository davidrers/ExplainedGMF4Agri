#!/usr/bin/env bash
# The whole experiment on the hub GPU, one step after the other, when the cluster has no GPU free (24 November 2026).
# Waits for a running step 0 (pid as $1, optional), applies the chain check's pass criterion (README), then runs the
# pilot, the eight shards and the sanity check. Stops at the first failure; a rerun skips every chip already written.
#
#     setsid nohup bash experiments/2026-11-20_xai_tessera/run_hub.sh [step0 pid] > results/.../logs/hub_chain.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # 16 GB: less fragmentation; memory only
EXP=experiments/2026-11-20_xai_tessera
OUT=results/2026-11-20_xai_tessera/EE_2021_rev
LOGS=$OUT/logs
say() { echo "[hub $(date +%m-%d\ %H:%M:%S)] $*"; }

if [ -n "${1:-}" ]; then
    say "waiting for step 0 (pid $1)"
    while kill -0 "$1" 2>/dev/null; do sleep 60; done
fi
if [ ! -f "$OUT/step0/verify.json" ]; then
    say "running step 0"
    poetry run python $EXP/run.py -e $EXP/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --arms step0 \
        --machine hub > $LOGS/step0_hub.log 2>&1 || { say "step 0 failed"; exit 1; }
fi

# the pass criterion, fixed before the check ran: fp32 pixels >= 0.990 and parcels >= 0.995
poetry run python - "$OUT/step0/verify.json" <<'PY' || { say "chain check FAILED: nothing is explained"; exit 1; }
import json, sys
v = json.load(open(sys.argv[1]))["32"]
ok = v["pixel_agreement"] >= 0.990 and v["parcel_agreement"] >= 0.995
print(f"fp32 pixels {v['pixel_agreement']:.5f}, parcels {v['parcel_agreement']:.5f}: {'pass' if ok else 'FAIL'}")
sys.exit(0 if ok else 1)
PY
say "chain check passed"

for step in pilot explain0 explain1 explain2 explain3 explain4 explain5 explain6 explain7 sanity; do
    say "$step"
    poetry run python $EXP/run.py -e $EXP/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --arms $step \
        --machine hub > $LOGS/${step}_hub.log 2>&1 || { say "$step failed (log $LOGS/${step}_hub.log)"; exit 1; }
    say "$step done"
done
say "all steps done"
