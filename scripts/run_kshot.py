"""The K-shot workflow: every arm x budget x draw x seed of an experiment, on one chip set.

    poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_mini
    poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021 --dry-run
    poetry run python scripts/run_kshot.py -e ... --chips ... --arms thor_v1_large --machine cluster --scratch /local/$SLURM_JOB_ID
    poetry run python scripts/run_kshot.py -e ... --chips ... --summarise

Results land in results/<experiment>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/, and a finished cell
is skipped, so a run resumes. On the hub, run it detached with its log beside the results:

    mkdir -p results/kshot/EE_2021_mini/logs
    setsid nohup poetry run python -u scripts/run_kshot.py -e configs/experiments/kshot.yaml \
        --chips data/eurocrops_chips/EE_2021_mini > results/kshot/EE_2021_mini/logs/hub_$(date +%m%d_%H%M).log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-e", "--experiment", type=Path, required=True)
    ap.add_argument("--chips", type=Path, required=True, help="the chip set directory")
    ap.add_argument("--split", default=None, help="the chip set's split, if it holds several")
    ap.add_argument("--arms", nargs="+", default=None, help="restrict to these arms")
    ap.add_argument("--machine", default="hub", choices=["hub", "cluster"])
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="where caches are looked up; default from the machine profile")
    ap.add_argument("--scratch", type=Path, default=None,
                    help="where missing caches are computed; default the cache root")
    ap.add_argument("--dry-run", action="store_true", help="plan and check, compute nothing")
    ap.add_argument("--summarise", action="store_true", help="only collect finished cells")
    args = ap.parse_args()

    from gfm4agri.pipeline.config import load_experiment
    from gfm4agri.pipeline.runner import run, summarise

    if args.summarise:
        out = REPO / "results" / load_experiment(args.experiment)["name"] / args.chips.name
        print(summarise(out).to_string(index=False))
        return
    rows = run(args.experiment, args.chips, split=args.split, arms=args.arms,
               machine=args.machine, cache_root=args.cache_root, scratch=args.scratch,
               dry_run=args.dry_run)
    for r in rows:
        print(f"  {r['arm']:<22} {r['cell']:<18} {r['state']:<8} {r['cache'] or ''}")


if __name__ == "__main__":
    main()
