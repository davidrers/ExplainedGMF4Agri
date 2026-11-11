"""The K-shot workflow: every arm x budget x draw x seed of an experiment, on one chip set.

    poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_mini
    poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021 --dry-run
    poetry run python scripts/run_kshot.py -e ... --chips ... --arms thor_v1_large --machine cluster --scratch /local/$SLURM_JOB_ID
    poetry run python scripts/run_kshot.py -e ... --chips ... --summarise

A spec with ``workflow: borders`` (configs/experiments/kshot_borders.yaml) runs the two-head workflow
instead, every arm with a border head beside its crop head (gfm4agri.borders.runner); ``--device`` and
``--rescore`` apply to it only.

    poetry run python scripts/run_kshot.py -e configs/experiments/kshot_borders.yaml --chips data/eurocrops_chips/EE_2021_rev_mini
    poetry run python scripts/run_kshot.py -e configs/experiments/kshot_borders.yaml --chips ... --summarise [--rescore]

Results land in results/<experiment>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/, and a finished cell
is skipped, so a run resumes. Every finished cell is also scored per parcel; ``--summarise`` scores
the cells that have no parcel scores yet, such as those pulled back from the cluster, and collects
them all. On the hub, run it detached with its log beside the results:

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
    ap.add_argument("--summarise", action="store_true",
                    help="only score per parcel the finished cells not yet scored, and collect them")
    ap.add_argument("--device", default=None,
                    help="two-head workflow: cuda or cpu; default cuda when available")
    ap.add_argument("--rescore", action="store_true",
                    help="two-head workflow, with --summarise: score every finished cell again")
    args = ap.parse_args()

    from gfm4agri.pipeline.config import load_experiment
    from gfm4agri.pipeline.runner import add_parcel_metrics, run, summarise

    exp = load_experiment(args.experiment)
    if exp.get("workflow") == "borders":
        from gfm4agri.borders import runner as borders

        if args.summarise:
            out = REPO / "results" / exp["name"] / args.chips.name
            done = borders.rescore(out, args.chips, machine=args.machine, every=args.rescore)
            print(f"border and field scores added to {len(done)} cells")
            print(borders.summarise(out).to_string(index=False))
            return
        rows = borders.run(args.experiment, args.chips, split=args.split, arms=args.arms,
                           machine=args.machine, cache_root=args.cache_root, scratch=args.scratch,
                           device=args.device, dry_run=args.dry_run)
        for r in rows:
            print(f"  {r['arm']:<24} {r['cell']:<18} {r['state']:<8} {r['cache'] or ''}")
        return
    if args.device or args.rescore:
        ap.error("--device and --rescore apply to the two-head workflow only")

    if args.summarise:
        out = REPO / "results" / exp["name"] / args.chips.name
        scored = add_parcel_metrics(out, args.chips, arms=args.arms)
        print(f"parcel scores added to {len(scored)} cells")
        print(summarise(out).to_string(index=False))
        return
    rows = run(args.experiment, args.chips, split=args.split, arms=args.arms,
               machine=args.machine, cache_root=args.cache_root, scratch=args.scratch,
               dry_run=args.dry_run)
    for r in rows:
        print(f"  {r['arm']:<22} {r['cell']:<18} {r['state']:<8} {r['cache'] or ''}")


if __name__ == "__main__":
    main()
