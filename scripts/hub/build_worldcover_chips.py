"""ESA WorldCover 2021 (v200) on the chip grid of every chip of a chip set: the background of the two-head workflow.

The two-head workflow (``gfm4agri.borders``, protocol D30) takes WorldCover's non-agricultural land as the background
class of PRUE's border target, and scores borders only where WorldCover and the declarations together say what the
land is. Moved from ``experiments/2026-11-05_parcel_extent/fetch_worldcover.py`` on 11 November 2026, unchanged; the
file it wrote for ``EE_2021_rev`` on 5 November 2026 is the one the workflow reads.

The three 3 x 3 degree tiles over Estonia are read from the Planetary Computer (collection ``esa-worldcover``, asset
``map``, EPSG:4326, 1/12,000 degree) and downloaded once into ``data/worldcover/tiles/``. Each chip is then resampled
onto its own grid (EPSG:3035, 10 m, 224 x 224, from the manifest's ``bounds_3035``) by nearest neighbour, from every
tile that overlaps it. Classes: 10 tree cover, 20 shrubland, 30 grassland, 40 cropland, 50 built-up, 60 bare or sparse
vegetation, 70 snow and ice, 80 permanent water, 90 herbaceous wetland, 95 mangroves, 100 moss and lichen; 0 where
no tile covers the pixel.

Writes, outside the chip set:

* ``data/worldcover/<set>/worldcover.npz``: one uint8 array per chip id;
* ``data/worldcover/<set>/worldcover.json``: the items, their versions and hrefs, the method, the class table and
  the chip manifest's SHA-256.

All chips of the set are written (train, validation and test): the targets need every chip, the scoring the test
chips. A pilot set (``subset.json``) needs none of its own: the workflow reads its parent's. Run on the hub, then
copy to the cluster with ``bash scripts/cluster/push_data.sh worldcover/<set>``:

    poetry run python scripts/hub/build_worldcover_chips.py --chips EE_2021_rev --workers 16
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402

COLLECTION, ASSET, VERSION = "esa-worldcover", "map", "2.0.0"
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"
CLASSES = {10: "tree cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up",
           60: "bare or sparse vegetation", 70: "snow and ice", 80: "permanent water", 90: "herbaceous wetland",
           95: "mangroves", 100: "moss and lichen"}
TILES = REPO / "data/worldcover/tiles"


def _items(bbox_4326):
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    items = [it for it in cat.search(collections=[COLLECTION], bbox=list(bbox_4326)).items()
             if it.properties.get("esa_worldcover:product_version") == VERSION]
    if not items:
        raise RuntimeError(f"no {COLLECTION} v{VERSION} items over {bbox_4326}")
    return sorted(items, key=lambda it: it.id)


def _download(item) -> Path:
    import urllib.request

    TILES.mkdir(parents=True, exist_ok=True)
    out = TILES / f"{item.id}.tif"
    if out.exists():
        return out
    tmp = out.with_suffix(".tif.part")
    print(f"downloading {item.id}", flush=True)
    urllib.request.urlretrieve(item.assets[ASSET].href, tmp)
    tmp.replace(out)
    return out


def _chip(args) -> tuple[str, np.ndarray]:
    """One chip's WorldCover classes on its grid, nearest neighbour from every overlapping tile."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import from_origin
    from rasterio.warp import reproject

    cid, bounds, tiles, px, size = args
    xmin, ymin, xmax, ymax = bounds
    dst = np.zeros((size, size), np.uint8)
    transform = from_origin(xmin, ymax, px, px)
    for t in tiles:
        part = np.zeros_like(dst)
        with rasterio.open(t) as src:
            reproject(rasterio.band(src, 1), part, dst_transform=transform, dst_crs="EPSG:3035",
                      resampling=Resampling.nearest, src_nodata=0, dst_nodata=0)
        dst = np.where(dst == 0, part, dst)
    return cid, dst


def _tiles_for(bounds, tile_boxes) -> list[str]:
    from pyproj import Transformer

    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    xmin, ymin, xmax, ymax = bounds
    lon, lat = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    lo0, lo1, la0, la1 = min(lon), max(lon), min(lat), max(lat)
    return [p for p, (a, b, c, d) in tile_boxes if lo0 < c and lo1 > a and la0 < d and la1 > b]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chips", required=True, help="the chip set's name, e.g. EE_2021_rev")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    import rasterio
    from pyproj import Transformer

    from gfm4agri.benchmark.segmentation_data import load_manifest, manifest_sha256

    root = REPO / "data/eurocrops_chips" / args.chips
    man = load_manifest(root)
    px, size = float(man["grid"]["pixel_m"]), int(man["grid"]["chip_px"])
    chips = [(c["chip_id"], c["bounds_3035"]) for c in man["chips"]]
    b = np.array([bb for _, bb in chips])
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(np.r_[b[:, 0], b[:, 2]], np.r_[b[:, 1], b[:, 3]])
    bbox = (float(lon.min()) - 0.05, float(lat.min()) - 0.05, float(lon.max()) + 0.05, float(lat.max()) + 0.05)
    items = _items(bbox)
    paths = [_download(it) for it in items]
    tile_boxes = [(str(p), tuple(it.bbox)) for p, it in zip(paths, items)]

    # The manifest's bounds give the chip's own grid: check it against a few masks.
    for cid, bounds in chips[:: max(1, len(chips) // 5)]:
        with rasterio.open(root / "chips" / f"{cid}.mask.tif") as m:
            t = m.transform
            if (abs(t.c - bounds[0]) > 1e-6 or abs(t.f - bounds[3]) > 1e-6 or abs(t.a - px) > 1e-9
                    or m.width != size or str(m.crs) != "EPSG:3035"):
                raise SystemExit(f"{cid}: manifest bounds {bounds} do not give the mask's grid {t}")

    out_dir = REPO / "data/worldcover" / args.chips
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "worldcover.npz"
    have = dict(np.load(path)) if path.exists() else {}
    todo = [(cid, bounds, _tiles_for(bounds, tile_boxes), px, size) for cid, bounds in chips if cid not in have]
    print(f"{len(chips)} chips, {len(have)} already done, {len(todo)} to fetch from {len(paths)} tiles", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for i, (cid, arr) in enumerate(ex.map(_chip, todo, chunksize=8), start=1):
            have[cid] = arr
            if i % 500 == 0 or i == len(todo):
                print(f"  {i} / {len(todo)}", flush=True)
    missing = [cid for cid, _ in chips if cid not in have]
    if missing:
        raise SystemExit(f"{len(missing)} chips missing, e.g. {missing[:3]}")
    tmp = out_dir / "worldcover.part.npz"
    np.savez_compressed(tmp, **{cid: have[cid] for cid, _ in chips})
    tmp.replace(path)
    allpx = np.stack([have[cid] for cid, _ in chips])
    shares = {int(k): round(float((allpx == k).mean()), 5) for k in [0, *CLASSES]}
    meta = {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "product": "ESA WorldCover 10 m 2021 v200", "collection": COLLECTION, "asset": ASSET, "version": VERSION,
        "items": [{"id": it.id, "bbox": it.bbox, "href": it.assets[ASSET].href.split("?")[0]} for it in items],
        "method": "nearest neighbour onto each chip's grid (EPSG:3035, 10 m) from every overlapping tile; 0 = no data",
        "classes": CLASSES, "chip_set": args.chips, "chip_manifest_sha256": manifest_sha256(root),
        "chips": len(chips), "pixel_share_per_class": shares,
    }
    (out_dir / "worldcover.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(shares))


if __name__ == "__main__":
    main()
