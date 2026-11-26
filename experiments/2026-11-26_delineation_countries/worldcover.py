"""ESA WorldCover 2021 on every chip of the sample, read in windows from the remote tiles: ``data/worldcover/EU18...``.

Run in the Poetry environment::

    poetry run python experiments/2026-11-26_delineation_countries/worldcover.py [--workers 16]

The same product, collection, version and resampling as the core's ``scripts/hub/build_worldcover_chips.py`` (ESA
WorldCover 10 m 2021 v200 from the Planetary Computer, nearest neighbour onto each chip's grid in EPSG:3035, 0 where
no tile covers a pixel). What differs is only how the tiles are read: the core downloads every 3 x 3 degree tile over
the bounding box of all chips, which for a sample spread over Europe would be most of the continent. Here each chip
asks the catalogue for the tiles over its own footprint and reads only the window it needs from each (cloud-optimised
GeoTIFFs over HTTPS). The catalogue is searched once per block, since it rate-limits a search per chip. Writes ``worldcover.npz`` (one uint8 array per chip id) and ``worldcover.json`` beside it.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import common as C

COLLECTION, ASSET, VERSION = "esa-worldcover", "map", "2.0.0"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
PX, SIZE = 10.0, 224
os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
os.environ.setdefault("CPL_VSIL_CURL_ALLOWED_EXTENSIONS", ".tif")
os.environ.setdefault("GDAL_HTTP_MAX_RETRY", "5")
os.environ.setdefault("GDAL_HTTP_RETRY_DELAY", "2")


def footprint(bounds):
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    x0, y0, x1, y1 = bounds
    lon, lat = tr.transform([x0, x1, x0, x1], [y0, y0, y1, y1])
    return min(lon), min(lat), max(lon), max(lat)


def search(bbox, tries: int = 6):
    """The v200 items over ``bbox`` (lon/lat), signed, retried with a growing pause when rate limited."""
    import time

    import planetary_computer
    import pystac_client

    for k in range(tries):
        try:
            cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
            return [it for it in cat.search(collections=[COLLECTION], bbox=list(bbox)).items()
                    if it.properties.get("esa_worldcover:product_version") == VERSION]
        except Exception:                                   # noqa: BLE001 (rate limit, network)
            if k == tries - 1:
                raise
            time.sleep(5 * 2 ** k)


def chip(args):
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    cid, bounds, hrefs = args
    dst = np.zeros((SIZE, SIZE), np.uint8)
    transform = from_origin(bounds[0], bounds[3], PX, PX)
    for href in hrefs:
        part = np.zeros_like(dst)
        with rasterio.open(href) as src:
            reproject(rasterio.band(src, 1), part, dst_transform=transform, dst_crs="EPSG:3035",
                      resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        dst = np.where(dst == 0, part, dst)
    return cid, dst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    man = json.loads((C.CHIPS / "manifest.json").read_text())
    chips = [(c["chip_id"], c["bounds_3035"]) for c in man["chips"]]
    out = C.WORLDCOVER.parent
    out.mkdir(parents=True, exist_ok=True)
    import time

    import pandas as pd

    # one catalogue search per block (the block's footprint), not per chip: the catalogue rate-limits
    blocks = pd.read_csv(C.HERE / "reference" / "sample_chips.csv").groupby(["country", "bx", "by"]).chip_id.apply(list)
    bounds = dict(chips)
    todo, tiles = [], set()
    for (cc, bx, by), ids in blocks.items():
        b = np.array([bounds[c] for c in ids])
        lo = footprint((b[:, 0].min(), b[:, 1].min(), b[:, 2].max(), b[:, 3].max()))
        items = search(lo)
        boxes = [(it.id, it.bbox, it.assets[ASSET].href) for it in sorted(items, key=lambda i: i.id)]
        tiles.update(i for i, _, _ in boxes)
        for c in ids:
            x0, y0, x1, y1 = footprint(bounds[c])
            todo.append((c, bounds[c], [h for _, (a0, b0, a1, b1), h in boxes
                                        if x0 < a1 and x1 > a0 and y0 < b1 and y1 > b0]))
        time.sleep(0.5)
    print(f"{len(blocks)} blocks searched, {len(tiles)} tiles", flush=True)
    have = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (cid, arr) in enumerate(ex.map(chip, todo, chunksize=4), start=1):
            have[cid] = arr
            if i % 200 == 0 or i == len(todo):
                print(f"  {i} / {len(todo)}", flush=True)
    tmp = out / "worldcover.part.npz"
    np.savez_compressed(tmp, **{cid: have[cid] for cid, _ in chips})
    tmp.replace(C.WORLDCOVER)
    allpx = np.stack([have[cid] for cid, _ in chips])
    meta = {"created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
            "product": "ESA WorldCover 10 m 2021 v200", "collection": COLLECTION, "asset": ASSET, "version": VERSION,
            "tiles": sorted(tiles), "method": "per chip: the tiles over its footprint, the needed window read from "
            "each cloud-optimised GeoTIFF, nearest neighbour onto the chip's grid (EPSG:3035, 10 m); 0 = no data",
            "chip_set": C.CHIPSET, "by": f"experiments/{C.NAME}/worldcover.py",
            "no_data_share": round(float((allpx == 0).mean()), 5)}
    (out / "worldcover.json").write_text(json.dumps(meta, indent=1))
    print(f"{len(chips)} chips from {len(tiles)} tiles, no-data share {meta['no_data_share']}")


if __name__ == "__main__":
    main()
