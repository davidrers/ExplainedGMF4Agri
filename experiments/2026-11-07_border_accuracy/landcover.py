"""What the chips hold: declared parcels, undeclared WorldCover farmland and non-farm land, per chip and split.

Every pixel of every chip of the split falls in one category (``CATEGORIES``), from the declared parcel layer
(``<chip>.parcels.tif``, ``<chip>.mask.tif``) and ESA WorldCover 2021 (grassland 30 and cropland 40 are farmland):

Also per chip: the WorldCover farmland pixels (``farm``) and those inside a declared parcel (``farm_declared``).

* declared, in scheme: a parcel of one of the 20 classes; what PRUE, PaPs and the boundary head learn as parcel;
* declared, other crop, on WorldCover farmland / on non-farm land: a parcel of a crop outside the 20 classes. PRUE
  and PaPs ignore it on farmland but take it as background on non-farm land, since their background is WorldCover's;
* undeclared, WorldCover grassland / cropland: ignored by every method in training, not scored at test;
* undeclared, non-farm land: the background of PRUE and PaPs; ignored by the boundary head;
* undeclared, no WorldCover data: ignored.

Writes ``results/2026-11-07_border_accuracy/<set>/landcover.npz``: ``ids``, ``split``, ``counts`` (chips x
categories) and ``in_scheme_on_nonfarm`` (declared in-scheme pixels WorldCover calls non-farm).

    poetry run python experiments/2026-11-07_border_accuracy/landcover.py --chips EE_2021_rev --workers 12
"""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402

NAME = "2026-11-07_border_accuracy"
SPLIT = "blocks4_buf1600_seed0__6b0eb4cb"
SPLITS = {"train": "training_data.txt", "validation": "validation_data.txt", "test": "test_data.txt"}
CATEGORIES = ("declared, in scheme", "declared, other crop, on WorldCover farmland",
              "declared, other crop, on non-farm land", "undeclared, WorldCover grassland",
              "undeclared, WorldCover cropland", "undeclared, non-farm land", "undeclared, no WorldCover data")
AGRI = (30, 40)


def chip(args):
    cid, chips_dir, wc = args
    import rasterio

    from gfm4agri.benchmark.parcel_metrics import read_parcels

    parcels = read_parcels(chips_dir, cid)
    with rasterio.open(chips_dir / f"{cid}.mask.tif") as s:
        mask = s.read(1).astype(np.int64)
    declared = parcels > 0
    lab = declared & (mask >= 0)
    other = declared & ~lab
    farm = np.isin(wc, AGRI)
    cats = [lab, other & farm, other & ~farm, ~declared & (wc == 30), ~declared & (wc == 40),
            ~declared & ~farm & (wc != 0), ~declared & (wc == 0)]
    counts = np.array([c.sum() for c in cats], np.int64)
    return counts, int((lab & ~farm & (wc != 0)).sum()), int((declared & farm).sum()), int(farm.sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chips", default="EE_2021_rev")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    root = REPO / "data/eurocrops_chips" / a.chips
    chips_dir = root / "chips"
    wc = np.load(REPO / "data/worldcover" / a.chips.replace("_mini", "") / "worldcover.npz")
    ids, split = [], []
    for s, f in SPLITS.items():
        these = (root / "splits" / SPLIT / f).read_text().split()
        ids += these
        split += [s] * len(these)
    with ProcessPoolExecutor(a.workers) as ex:
        out = list(ex.map(chip, [(c, chips_dir, wc[c]) for c in ids], chunksize=32))
    counts = np.stack([o[0] for o in out])
    on_nonfarm = np.array([o[1] for o in out], np.int64)
    farm_declared = np.array([o[2] for o in out], np.int64)
    farm = np.array([o[3] for o in out], np.int64)
    dest = REPO / "results" / NAME / a.chips
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / "landcover.npz", ids=np.array(ids), split=np.array(split), counts=counts,
                        categories=np.array(CATEGORIES), in_scheme_on_nonfarm=on_nonfarm,
                        farm_declared=farm_declared, farm=farm)
    split = np.array(split)
    for s in SPLITS:
        c = counts[split == s].sum(0)
        print(f"{s:10s} {int((split == s).sum()):5d} chips: " +
              ", ".join(f"{k} {v / c.sum():.1%}" for k, v in zip(CATEGORIES, c)))
        sel = split == s
        print(f"{'':10s} WorldCover farmland inside a declared parcel: {farm_declared[sel].sum() / farm[sel].sum():.1%};"
              f" in-scheme pixels WorldCover calls non-farm: {on_nonfarm[sel].sum() / c[0]:.1%}")
    print(f"wrote {(dest / 'landcover.npz').relative_to(REPO)}")


if __name__ == "__main__":
    main()
