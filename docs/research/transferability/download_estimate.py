"""Estimated chips and download volume of every candidate domain of the transfer matrix.

A chip set holds one 2.24 km chip for every cell of the EPSG:3035 chip grid that contains a labelled parcel. Estonia
and Latvia have their chips; for the other domains the count is estimated from GAEZ v5 land cover: a cell counts
when cropland plus grassland (LR-LCC LC02 + LC03, the land EuroCrops labels) cover more than a threshold share of it.
The threshold is fitted so that the estimate reproduces the chips of Estonia and Latvia, and it is checked against
the cells that Lithuania's EuroCrops v11 parcels touch.

Volume per chip is measured on the chip sets on disk: the Sentinel-2 composites (`_merged.tif`, revised screening)
and the Sentinel-1 RTC chips. AlphaEarth is downloaded as whole source tiles, chosen with the same rule as
`scripts/hub/fetch_alphaearth_tiles.py` (`select_tiles`), at the mean size of the 22 tiles Estonia needed; tiles already on disk are not counted
again. TESSERA v2 is computed from the composites and downloads nothing. Time is the Sentinel-2 export at the rate
Latvia's export logged on the hub (12 workers), the slower of the two exports that run side by side.

    poetry run python docs/research/transferability/download_estimate.py

Writes `download_estimate.csv` next to this file. Reuses the domains and helpers of `gaez_domains.py`.
"""
from __future__ import annotations

import json
import re

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject

from gaez_domains import BOUNDS, DOMAINS, HERE, NUTS, ROOT
from gfm4agri.embeddings.alphaearth import select_tiles

CELL = 2240.0
CHIPS = ROOT / "data/eurocrops_chips"
LCC = "/vsicurl/https://storage.googleapis.com/fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET/LR-LCC/GAEZ-V5.LR-LCC.{}.tif"
AE_INDEX = ROOT / "data/alphaearth/aef_v1_annual/aef_index.parquet"
AE_TILES = ROOT / "data/alphaearth/aef_v1_annual/2021/tiles.json"
HAVE = {"EE": "EE_2021_rev", "LV": "LV_2021"}          # domains whose chips are on disk
PARCELS = {"LT": ROOT / "data/eurocrops/parquet/LT_2021.parquet"}   # check of the estimate


def mean_size(pattern: str, sets: list[str]) -> float:
    sizes = [p.stat().st_size for s in sets for p in (CHIPS / s / "chips").glob(pattern) if p.is_file()]
    return float(np.mean(sizes))


def grid():
    """The chip grid over all domains: transform, shape, and the domain id of every cell (cell centre inside)."""
    nuts = gpd.read_file(NUTS)
    geoms = {d: nuts[nuts.NUTS_ID.isin(c)].to_crs(3035).geometry.union_all() for d, c in DOMAINS.items()}
    xmin, ymin, xmax, ymax = gpd.GeoSeries(list(geoms.values()), crs=3035).total_bounds
    col0, row0 = int(np.floor(xmin / CELL)), int(np.floor(ymin / CELL))
    col1, row1 = int(np.ceil(xmax / CELL)), int(np.ceil(ymax / CELL))
    shape = (row1 - row0, col1 - col0)
    transform = from_origin(col0 * CELL, row1 * CELL, CELL, CELL)
    ids = rasterize([(g, k) for k, g in enumerate(geoms.values(), start=1)], out_shape=shape, transform=transform,
                    fill=0, dtype="int16")
    return transform, shape, ids, (col0, row1)


def agri_share(transform, shape) -> np.ndarray:
    """Mean cropland + grassland share (%) of every chip cell, from the 30 arc-second GAEZ layers."""
    total = np.zeros(shape, dtype="float32")
    for code in ("LC02", "LC03"):
        with rasterio.open(LCC.format(code)) as src:
            win = rasterio.windows.from_bounds(*BOUNDS, transform=src.transform)
            arr = np.nan_to_num(src.read(1, window=win).astype("float32"), nan=0.0)
            out = np.zeros(shape, dtype="float32")
            reproject(arr, out, src_transform=src.window_transform(win), src_crs=src.crs, dst_transform=transform,
                      dst_crs="EPSG:3035", resampling=Resampling.average)
        total += np.clip(out, 0, 100)
    return total


def cells_of(rows: np.ndarray, cols: np.ndarray, origin) -> np.ndarray:
    """Chip bounds (EPSG:3035) of grid cells given as array rows and columns."""
    col0, row_top = origin
    x0 = (col0 + cols) * CELL
    y1 = (row_top - rows) * CELL
    return np.stack([x0, y1 - CELL, x0 + CELL, y1], axis=1)


def parcel_cells(path) -> int:
    """Cells that any parcel's bounding box touches: a slight over-count of the cells holding labelled pixels."""
    b = gpd.read_parquet(path).to_crs(3035).bounds
    c0, c1 = np.floor(b.minx / CELL).astype(int), np.floor(b.maxx / CELL).astype(int)
    r0, r1 = np.floor(b.miny / CELL).astype(int), np.floor(b.maxy / CELL).astype(int)
    cells = set()
    for a, z, q, w in zip(c0, c1, r0, r1):
        cells.update((c, r) for c in range(a, z + 1) for r in range(q, w + 1))
    return len(cells)


def main() -> None:
    s2 = mean_size("*_merged.tif", list(HAVE.values()))
    s1 = mean_size("*_s1rtc.tif", ["EE_2021", "LV_2021"])
    ae_chip = mean_size("*_alphaearth.tif", ["EE_2021"])
    fetched = json.loads(AE_TILES.read_text())["fetched"]
    ae_tile = float(np.mean([v["size"] for v in fetched.values()]))
    actual = {d: len(json.loads((CHIPS / s / "manifest.json").read_text())["chips"]) for d, s in HAVE.items()}
    n, hours = re.search(r"exported ([\d,]+) chips, \d+ failed, in ([\d.]+) h",
                         (CHIPS / "LV_2021/export.log").read_text()).groups()
    s2_rate = int(n.replace(",", "")) / float(hours)    # chips per hour

    transform, shape, ids, origin = grid()
    share = agri_share(transform, shape)

    # Fit the threshold on the two domains with chips
    names = list(DOMAINS)
    def counts(t):
        return {d: int(((ids == k) & (share > t)).sum()) for k, d in enumerate(names, start=1)}
    grid_t = np.arange(0.5, 40.5, 0.5)
    err = [sum(abs(np.log(counts(t)[d] / actual[d])) for d in HAVE) for t in grid_t]
    t_best = float(grid_t[int(np.argmin(err))])
    est = counts(t_best)
    check = {d: parcel_cells(p) for d, p in PARCELS.items()}

    index = pd.read_parquet(AE_INDEX, columns=["path", "year", "utm_zone", "crs", "utm_west", "utm_south", "utm_east",
                                               "utm_north"])
    index["key"] = index["path"].str.split("/").str[-3:].str.join("/")
    rows = []
    for k, d in enumerate(names, start=1):
        r, c = np.nonzero((ids == k) & (share > t_best))
        bounds = cells_of(r, c, origin)
        if d in HAVE:
            bounds = np.array([ch["bounds_3035"] for ch in json.loads(
                (CHIPS / HAVE[d] / "manifest.json").read_text())["chips"]], dtype=float)
        tiles = select_tiles(index, bounds, "EPSG:3035", 2021)
        chips = actual.get(d, est[d])
        new = set(tiles["key"]) - set(fetched)
        on_disk = d in HAVE
        rows.append({"domain": d, "chips": chips, "chips_estimated": d not in HAVE, "grid_cells": int((ids == k).sum()),
                     "s2_gb": chips * s2 / 1e9, "s1_gb": chips * s1 / 1e9, "alphaearth_tiles": len(tiles),
                     "alphaearth_tiles_gb": len(tiles) * ae_tile / 1e9, "alphaearth_chips_gb": chips * ae_chip / 1e9,
                     "alphaearth_tiles_new": len(new), "keys": set(tiles["key"]),
                     "to_download_gb": (0 if on_disk else chips * (s2 + s1)) / 1e9 + len(new) * ae_tile / 1e9,
                     "s2_export_days": 0.0 if on_disk else chips / s2_rate / 24})
    df = pd.DataFrame(rows).set_index("domain")
    df["download_gb"] = df.s2_gb + df.s1_gb + df.alphaearth_tiles_gb
    df.drop(columns="keys").round(1).to_csv(HERE / "download_estimate.csv")

    print(f"Sentinel-2 export rate {s2_rate:.0f} chips per hour")
    print(f"per chip: S2 {s2 / 1e6:.2f} MB, S1 {s1 / 1e6:.2f} MB, AlphaEarth chip {ae_chip / 1e6:.2f} MB; "
          f"AlphaEarth tile {ae_tile / 1e9:.2f} GB")
    print(f"threshold {t_best} % cropland + grassland; fit: " + ", ".join(
        f"{d} {est[d]} estimated vs {actual[d]} chips" for d in HAVE))
    print("check: " + ", ".join(f"{d} {est[d]} estimated vs {n} cells under its v11 parcels" for d, n in check.items()))
    all_keys = set().union(*df["keys"])
    print(f"AlphaEarth tiles over all domains: {len(all_keys)} unique ({len(all_keys) * ae_tile / 1e9:.0f} GB), "
          f"{int(df.alphaearth_tiles.sum())} counted per domain")
    pd.set_option("display.width", 200)
    print(df.drop(columns="keys").round(1).to_string())


if __name__ == "__main__":
    main()
