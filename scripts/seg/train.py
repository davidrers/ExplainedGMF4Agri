"""Train and evaluate one frozen-encoder segmentation fit on the EuroCrops chips.

    poetry run python scripts/seg/train.py -c configs/seg/terramind_v1_small_ee_pilot.yaml
    poetry run python scripts/seg/train.py -c ... --fast-dev-run
    poetry run python scripts/seg/train.py -c ... --set label_budget.mode=pct label_budget.pct=5

Writes ``<output_root>/<run_name>/<budget>_seed<seed>/``: the Lightning logs and checkpoint,
``config.yaml`` as resolved, and ``results.json`` carrying the configuration, the label
budget, the country, the split protocol, the parameter counts and the metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def _set(cfg: dict, dotted: str) -> None:
    key, value = dotted.split("=", 1)
    *parents, leaf = key.split(".")
    node = cfg
    for p in parents:
        node = node.setdefault(p, {})
    node[leaf] = yaml.safe_load(value)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, dotted keys")
    ap.add_argument("--fast-dev-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for s in args.set:
        _set(cfg, s)

    from gfm4agri.benchmark.fit import fit_end_to_end

    fit_end_to_end(cfg, fast_dev_run=args.fast_dev_run)


if __name__ == "__main__":
    main()
