"""How WorldCover 2021 covers the parcels of the test chips, before its class set is fixed as the extent.

Writes ``results/<name>/<set>/worldcover_stats.json`` and prints two tables:

* for each of our 20 classes, the share of its labelled test pixels in each WorldCover class (does grassland, 30,
  or herbaceous wetland, 90, hold a notable share of a class?);
* for the whole test set, where each WorldCover class lies: on in-scheme parcels, on void (declared parcels of a
  class outside the scheme), or on background (no declared parcel). WorldCover cropland or grassland on background
  is the undeclared-farmland leak: a correct parcel there scores as an FP.

    poetry run python experiments/2026-11-05_parcel_extent/worldcover_stats.py --chips EE_2021_rev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402

WC = [10, 20, 30, 40, 50, 60, 80, 90]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    ap.add_argument("--name", default="2026-11-05_parcel_extent")
    a = ap.parse_args()

    from gfm4agri.benchmark.parcel_metrics import read_parcels
    from gfm4agri.benchmark.segmentation_data import load_manifest
    from gfm4agri.pipeline.config import resolve_split

    root = REPO / "data/eurocrops_chips" / a.chips
    man = load_manifest(root)
    names = [c["name"] for c in man["classes"]]
    split = resolve_split(root)
    test = (split / "test_data.txt").read_text().split()
    wc_all = np.load(REPO / "data/worldcover" / a.chips.replace("_mini", "") / "worldcover.npz")
    import rasterio

    by_class = np.zeros((len(names), 256), np.int64)
    land = np.zeros((3, 256), np.int64)               # parcels, void, background
    for cid in test:
        with rasterio.open(root / "chips" / f"{cid}.mask.tif") as m:
            mask = m.read(1).astype(np.int64)
        parcels = read_parcels(root / "chips", cid)
        wc = wc_all[cid].astype(np.int64)
        lab = (mask >= 0) & (parcels > 0)
        np.add.at(by_class, (mask[lab], wc[lab]), 1)
        for k, sel in enumerate((lab, (mask < 0) & (parcels > 0), parcels == 0)):
            land[k] += np.bincount(wc[sel], minlength=256)

    share = by_class[:, WC] / by_class.sum(1, keepdims=True)
    out = {"test_chips": len(test), "worldcover_classes": WC,
           "class_share": {n: dict(zip(map(str, WC), map(float, s))) for n, s in zip(names, share)},
           "class_pixels": {n: int(v) for n, v in zip(names, by_class.sum(1))},
           "all_parcels_share": dict(zip(map(str, WC), map(float, by_class.sum(0)[WC] / by_class.sum()))),
           "worldcover_class_on": {str(c): {"parcels": int(land[0, c]), "void": int(land[1, c]),
                                            "background": int(land[2, c])} for c in WC}}
    for c in (30, 40):
        tot = land[:, c].sum()
        out[f"wc{c}_on_background_share"] = float(land[2, c] / tot)
    agri = land[:, [30, 40]].sum()
    out["wc30_40_on_background_share"] = float(land[2, [30, 40]].sum() / agri)
    out["parcels_inside_wc30_40"] = float(land[0, [30, 40]].sum() / land[0].sum())
    out["parcels_inside_wc30_40_90"] = float(land[0, [30, 40, 90]].sum() / land[0].sum())
    dest = REPO / "results" / a.name / a.chips
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "worldcover_stats.json").write_text(json.dumps(out, indent=2))

    print("class".ljust(36) + "".join(f"{c:>7}" for c in WC) + "   pixels")
    for n, s, t in zip(names, share, by_class.sum(1)):
        print(n[:35].ljust(36) + "".join(f"{v:7.3f}" for v in s) + f"  {t:8d}")
    print("all parcels".ljust(36) + "".join(f"{v:7.3f}" for v in by_class.sum(0)[WC] / by_class.sum()))
    print("\nWorldCover class on    parcels       void  background")
    for c in WC:
        print(f"{c:>18}  {land[0, c]:9d}  {land[1, c]:9d}  {land[2, c]:10d}")
    print(json.dumps({k: v for k, v in out.items() if k.startswith(("wc", "parcels_"))}, indent=1))


if __name__ == "__main__":
    main()
