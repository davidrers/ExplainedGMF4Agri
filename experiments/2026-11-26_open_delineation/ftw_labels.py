"""What Fields of The World calls a field in Estonia, against our parcels: ``reference/ftw_label_check.json``.

Run in the Poetry environment, after ``ftw_overlap.py``::

    poetry run python experiments/2026-11-26_open_delineation/ftw_labels.py [--chips 40]

FTW's Estonian data config lists two crop groups of the declarations, "Põllukultuurid" (arable crops) and
"Püsikultuurid" (permanent crops); permanent grassland is not among them. For a random sample of our chips that FTW
covers, its three-class label masks (``label_masks/semantic_3class``, 0 background, 1 field, 2 boundary) are warped
onto our chip grid (nearest) and tallied under our grassland parcels, our other in-scheme parcels, our out-of-scheme
parcels and land outside every parcel. Masks are downloaded once into ``data/external_models/ftw/estonia_masks/``.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.warp import Resampling, reproject
from shapely.geometry import box

import common as C

URL = "https://data.source.coop/kerner-lab/fields-of-the-world/estonia/label_masks/semantic_3class/{}.tif"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", type=int, default=40)
    args = ap.parse_args()
    cache = C.WEIGHTS / "ftw" / "estonia_masks"
    cache.mkdir(parents=True, exist_ok=True)
    m = json.loads((C.CHIPS / "manifest.json").read_text())
    info = {c["chip_id"]: c for c in m["chips"]}
    ov = pd.read_csv(C.HERE / "reference" / "ftw_overlap.csv")
    pool = ov[ov.ftw != "none"].chip_id.values
    pick = np.random.default_rng(0).choice(pool, min(args.chips, len(pool)), replace=False)
    ftw = gpd.read_parquet(C.WEIGHTS / "ftw" / "chips_estonia.parquet").to_crs(3035)
    groups = ("grassland parcels", "other in-scheme parcels", "out-of-scheme parcels", "no parcel")
    tot = {g: np.zeros(3, np.int64) for g in groups}
    for cid in pick:
        hits = ftw[ftw.intersects(box(*info[cid]["bounds_3035"]))]
        with rasterio.open(C.CHIP_DIR / f"{cid}.mask.tif") as s:
            mask, prof = s.read(1), s.profile
        with rasterio.open(C.CHIP_DIR / f"{cid}.parcels.tif") as s:
            par = s.read(1)
        lab = np.full(mask.shape, 255, np.uint8)
        for aoi in hits.aoi_id:
            p = cache / f"{aoi}.tif"
            if not p.exists():
                urllib.request.urlretrieve(URL.format(aoi), p)
            with rasterio.open(p) as s:
                tmp = np.full(mask.shape, 255, np.uint8)
                reproject(s.read(1), tmp, src_transform=s.transform, src_crs=s.crs, dst_transform=prof["transform"],
                          dst_crs=prof["crs"], resampling=Resampling.nearest, dst_nodata=255)
            lab = np.where(lab == 255, tmp, lab)
        cov = lab != 255
        sel = {"grassland parcels": (mask == 0) & cov, "other in-scheme parcels": (mask > 0) & cov,
               "out-of-scheme parcels": (mask < 0) & (par > 0) & cov, "no parcel": (par == 0) & cov}
        for g in groups:
            tot[g] += np.bincount(lab[sel[g]], minlength=256)[:3]
    out = {"chips": [str(c) for c in pick], "classes": ["background", "field", "boundary"],
           "share": {g: (v / max(v.sum(), 1)).round(4).tolist() for g, v in tot.items()},
           "pixels": {g: int(v.sum()) for g, v in tot.items()}}
    (C.HERE / "reference" / "ftw_label_check.json").write_text(json.dumps(out, indent=1))
    for g in groups:
        s = out["share"][g]
        print(f"{g:24s} {out['pixels'][g]:9d} px  background {s[0]:.3f}  field {s[1]:.3f}  boundary {s[2]:.3f}")


if __name__ == "__main__":
    main()
