"""Cut a pilot chip set from a country chip set: whole spatial blocks of every partition.

    poetry run python scripts/hub/build_pilot_chipset.py --parent data/eurocrops_chips/EE_2021 --name EE_2021_mini

The pilot is written beside its parent as links into it, so it costs no space; copy the parent
to the cluster before the pilot. An existing pilot directory is never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default=None, help="the parent's split, if it holds several")
    ap.add_argument("--train-blocks", type=int, default=3)
    ap.add_argument("--val-blocks", type=int, default=1)
    ap.add_argument("--test-blocks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from gfm4agri.data.chip_subset import choose_blocks, write_subset
    from gfm4agri.pipeline.config import resolve_split

    parent = args.parent if args.parent.is_absolute() else REPO / args.parent
    split_dir = resolve_split(parent, args.split)
    chips = pd.read_csv(split_dir / "chips.csv", dtype={"block_id": str})
    blocks = choose_blocks(chips, {"pool": args.train_blocks, "val": args.val_blocks,
                                   "test": args.test_blocks}, args.seed)
    dest = write_subset(parent, parent.parent / args.name, split_dir, blocks, args.seed)
    print(f"{dest.relative_to(REPO)}: blocks {blocks}")
    print((dest / "subset.json").read_text())


if __name__ == "__main__":
    main()
