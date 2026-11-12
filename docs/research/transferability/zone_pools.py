"""Whole blocks of chips per candidate country and GAEZ climatic zone: the pools the C, Z and M training sets draw from.

A block is the 4 x 4 chips (8.96 km) of the main workflow's split, aligned to the EPSG:3035 grid. Chips are the grid
cells `download_estimate.py` counts (cropland plus grassland above its fitted threshold; the real chips for Estonia
and Latvia). Each chip gets the climatic AEZ class of GAEZ v5 at 1 km (AEZ33-1KM), by majority over the chip, after
the special classes (25 to 33: soil, terrain, irrigation, wetness, water, built-up) are filled with the nearest
climatic class, since they override the climate they sit in. A block is

* **pure** when all its chips are in one zone, and
* **core** when it is pure and none of its eight neighbours holds a chip of another zone: the 9 km buffer at zone
  borders.

    poetry run python docs/research/transferability/zone_pools.py

Writes `zone_pools.csv` (country x zone: chips, blocks, pure and core blocks, chips in core blocks), `aez33_chips.csv`
(chips per AEZ33 class as GAEZ maps it, special classes kept, per country) and
`zone_blocks.parquet` (one row per block of the candidate countries: country, zone, chips, purity, core) next to
this file. Reuses the helpers of `gaez_domains.py` and `download_estimate.py`.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from scipy.ndimage import distance_transform_edt

from download_estimate import CHIPS, HAVE, agri_share, grid
from gaez_domains import BOUNDS, DOMAINS, GAEZ, HERE, aez33_labels

AEZ_1KM = GAEZ / "GAEZ-V5.AEZ33-1KM.tif"
CELL = 2240.0
THRESHOLD = 4.5                                     # % cropland + grassland, fitted in download_estimate.py
# AEZ33 climatic classes -> zone. Moisture is kept apart only where it splits European cropland (temperate cool).
ZONES = {
    21: "cool wet", 20: "cool moist", 19: "cool dry",
    16: "moderate", 17: "moderate", 18: "moderate",
    10: "subtropics mod. cool", 11: "subtropics mod. cool", 12: "subtropics mod. cool",
    13: "subtropics cool", 14: "subtropics cool", 15: "subtropics cool",
    22: "cold", 23: "cold", 24: "cold",
}


def aez_on_grid(transform, shape, fill: bool) -> np.ndarray:
    """AEZ33 class of every chip cell, by majority; with `fill`, special classes first take the nearest climatic class."""
    with rasterio.open(AEZ_1KM) as src:
        win = from_bounds(*BOUNDS, transform=src.transform)
        aez = src.read(1, window=win).astype("int16")
        src_transform, crs = src.window_transform(win), src.crs
    if fill:
        climatic = (aez >= 1) & (aez <= 24)
        _, (iy, ix) = distance_transform_edt(~climatic, return_indices=True)
        aez = aez[iy, ix]
    out = np.zeros(shape, dtype="int16")
    reproject(aez, out, src_transform=src_transform, src_crs=crs, dst_transform=transform, dst_crs="EPSG:3035",
              resampling=Resampling.mode)
    return out


def main() -> None:
    transform, shape, ids, (col0, row_top) = grid()
    share = agri_share(transform, shape)
    aez = aez_on_grid(transform, shape, fill=True)
    raw = aez_on_grid(transform, shape, fill=False)
    names = list(DOMAINS)

    rows = []
    for k, d in enumerate(names, start=1):
        if d in HAVE:   # the real chips
            b = np.array([c["bounds_3035"] for c in json.loads((CHIPS / HAVE[d] / "manifest.json").read_text())["chips"]])
            gx = np.floor(b[:, 0] / CELL + 0.5).astype(int)
            gy = np.floor(b[:, 1] / CELL + 0.5).astype(int)
            r, c = row_top - 1 - gy, gx - col0
        else:
            r, c = np.nonzero((ids == k) & (share > THRESHOLD))
            gx, gy = col0 + c, row_top - 1 - r
        rows.append(pd.DataFrame({"country": d, "gx": gx, "gy": gy, "aez": aez[r, c], "aez_raw": raw[r, c]}))
    chips = pd.concat(rows, ignore_index=True)
    chips["zone"] = chips.aez.map(ZONES).fillna("other")
    chips["bx"], chips["by"] = chips.gx // 4, chips.gy // 4

    # Chips per AEZ33 class as GAEZ maps it (special classes kept), per country
    labels = aez33_labels()
    per_class = chips.pivot_table(index="aez_raw", columns="country", values="gx", aggfunc="size", fill_value=0)
    per_class.insert(0, "class", per_class.index.map(labels))
    countries = per_class.columns[1:]
    per_class["chips"] = per_class[countries].sum(axis=1)
    per_class["countries_1000"] = (per_class[countries] >= 1000).sum(axis=1)
    per_class.rename_axis("code").sort_values("chips", ascending=False).to_csv(HERE / "aez33_chips.csv")

    blocks = (chips.groupby(["country", "bx", "by"])
              .agg(chips=("zone", "size"), zone=("zone", lambda s: s.mode().iloc[0]), zones=("zone", "nunique"))
              .reset_index())
    blocks["pure"] = blocks.zones == 1
    # Core: pure, and no chip of another zone (any country) in the 8 neighbouring blocks
    zones_at = chips.groupby(["bx", "by"]).zone.agg(set).to_dict()
    def core(row) -> bool:
        if not row.pure:
            return False
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if zones_at.get((row.bx + dx, row.by + dy), {row.zone}) - {row.zone}:
                    return False
        return True
    blocks["core"] = blocks.apply(core, axis=1)
    blocks.drop(columns="zones").to_parquet(HERE / "zone_blocks.parquet", index=False)

    pools = (blocks.groupby(["country", "zone"])
             .agg(chips=("chips", "sum"), blocks=("chips", "size"), pure_blocks=("pure", "sum"),
                  core_blocks=("core", "sum"),
                  core_chips=("chips", lambda s: int(s[blocks.loc[s.index, "core"]].sum())))
             .reset_index())
    pools.to_csv(HERE / "zone_pools.csv", index=False)
    pd.set_option("display.width", 200)
    print(pools[pools.chips >= 500].sort_values(["zone", "core_chips"], ascending=[True, False]).to_string(index=False))
    print("\nchips per block, EE (real chips):", round(blocks[blocks.country == "EE"].chips.mean(), 1))


if __name__ == "__main__":
    main()
