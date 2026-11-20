"""Entry script: one step of the XAI experiment per call, with the K-shot runner's command line so that
``scripts/cluster/submit.sh`` runs it as a Slurm job, the step in place of an arm:

    bash scripts/cluster/submit.sh experiments/2026-11-20_xai_tessera/experiment.yaml EE_2021_rev step0     # cluster
    poetry run python experiments/2026-11-20_xai_tessera/run.py -e experiments/2026-11-20_xai_tessera/experiment.yaml \
        --chips data/eurocrops_chips/EE_2021_rev --arms step0 --limit 2                                     # hub test

Steps (``experiment.yaml`` lists them):

* ``step0``: training-chip statistics, mean field, chain check against ``predictions_test.npz`` (``step0.py``);
* ``sample``: the test fields explained (``sample.py``);
* ``explain<i>``: shard ``i`` of the sampled chips through the three methods (``explain.py``);
* ``sanity``: the randomised-model check (``explain.py``).

Outputs: ``results/<name>/<model chip set>/<step>/``; the statistics always come from the model's chip set,
``EE_2021_rev``, whatever ``--chips`` holds (a pilot set's chips are links to the same files).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-e", "--experiment", required=True)
    ap.add_argument("--chips", required=True)
    ap.add_argument("--arms", required=True, help="the step(s), comma separated")
    ap.add_argument("--machine", default="hub")
    ap.add_argument("--scratch", default=None)
    ap.add_argument("--split", default=None)
    ap.add_argument("--limit", type=int, default=None, help="chips per step, for a test")
    args = ap.parse_args()

    import yaml

    from gfm4agri.pipeline.config import load_machine, resolve_split

    import chain

    spec = yaml.safe_load(Path(args.experiment).read_text())
    chips = Path(args.chips) if Path(args.chips).is_absolute() else Path.cwd() / args.chips
    split_dir = resolve_split(chips, args.split)
    batch = int(load_machine(args.machine)["encode_batch_size"]["tessera_v2_medium"])
    root = chain.results_dir(chips.name)
    for step in args.arms.split(","):
        if step == "step0":
            import step0

            out = chain.results_dir() / "step0"
            if args.limit:
                out = root / "step0_test"
            if not (out / "embedding_stats.json").exists():
                step0.prep(chips, split_dir, out, batch, args.limit)
            step0.verify(chips, split_dir, out, batch, args.limit)
        elif step == "sample":
            import sample

            sample.run(chips, split_dir, root / "sample", spec["sample"])
        elif step.startswith("explain") or step == "sanity":
            import explain

            explain.run(step, chips, root, spec, batch, args.limit)
        else:
            raise SystemExit(f"unknown step {step!r}")


if __name__ == "__main__":
    main()
