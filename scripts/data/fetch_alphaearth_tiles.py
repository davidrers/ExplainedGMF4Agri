"""Download the AlphaEarth Foundations tiles a chip set needs, checked against Google's MD5.

The tiles are selected from the provider's index exactly as
:meth:`gfm4agri.embeddings.alphaearth.AlphaEarthStore.read_chip` reads them: every tile of the
chip's own UTM zone that meets the chip's footprint, and of the neighbouring zone for a chip
near the zone edge. Each file is fetched from the Source Cooperative mirror, or with ``--source
official`` from Google's bucket, which has answered anonymous reads although it is labelled
requester pays; a partial download resumes, and a file is kept only if its MD5 matches the one
Google's bucket publishes for the same object. ``<out>/<year>/tiles.json`` lists every tile of each zone touched, downloaded or not,
so that a chip needing a tile that is not on disk fails instead of coming back empty.

Run it once per chip set before ``build_alphaearth_chips.py``; a tile already on disk with a
verified checksum is skipped, so a second chip set only adds what it lacks.

    poetry run python scripts/data/fetch_alphaearth_tiles.py --root data/eurocrops_chips/EE_2021_pilot
    poetry run python scripts/data/fetch_alphaearth_tiles.py --root data/eurocrops_chips/EE_2021 --list-only
    poetry run python scripts/data/fetch_alphaearth_tiles.py --root data/eurocrops_chips/EE_2021 --workers 4
"""

from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import sys
import time
import urllib.parse
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.embeddings.alphaearth import (  # noqa: E402
    INDEX_URL, MIRROR_URL, OFFICIAL_URL, TILE_TABLE, select_tiles)

GCS_API = "https://storage.googleapis.com/storage/v1/b/alphaearth_foundations/o/"
GCS_PREFIX = "satellite_embedding/v1/annual/"
INDEX_COLUMNS = ["path", "year", "utm_zone", "crs", "utm_west", "utm_south", "utm_east",
                 "utm_north"]
CHUNK = 8 << 20
#: Attempts after the first, 60 s apart and growing, before a tile is left for the next run.
RETRIES = 3


def key_of(path: str) -> str:
    """``"<year>/<zone>/<name>.tiff"`` from an index path."""
    return path.split("/v1/annual/", 1)[1]


def official_meta(key: str) -> dict:
    """Size and MD5 of the object in Google's bucket; the mirror's header when that fails."""
    import requests

    try:
        r = requests.get(GCS_API + urllib.parse.quote(GCS_PREFIX + key, safe=""), timeout=60)
        r.raise_for_status()
        m = r.json()
        return {"size": int(m["size"]), "md5": m["md5Hash"], "md5_source": "gs://alphaearth_foundations"}
    except Exception:  # metadata reads have been free so far; fall back if that changes
        h = requests.head(f"{MIRROR_URL}/{key}", timeout=60)
        h.raise_for_status()
        return {"size": int(h.headers["content-length"]), "md5": h.headers["x-amz-meta-md5chksum"],
                "md5_source": "mirror header x-amz-meta-md5chksum"}


def md5_b64(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return base64.b64encode(h.digest()).decode()


SOURCES = {"mirror": MIRROR_URL, "official": OFFICIAL_URL}


def fetch(key: str, dest: Path, meta: dict, base: str) -> None:
    """Stream ``<base>/<key>`` to ``<dest>.part``, resuming, verify, then rename."""
    import requests

    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    have = part.stat().st_size if part.exists() else 0
    if have < meta["size"]:
        headers = {"Range": f"bytes={have}-"} if have else {}
        with requests.get(f"{base}/{key}", headers=headers, stream=True, timeout=120) as r:
            r.raise_for_status()
            if have and r.status_code != 206:
                have = 0   # the server ignored the range; start again
            with open(part, "ab" if have else "wb") as f:
                for block in r.iter_content(CHUNK):
                    f.write(block)
    got = md5_b64(part)
    if got != meta["md5"]:
        part.unlink()
        raise IOError(f"MD5 {got} != {meta['md5']} ({meta['md5_source']}); partial file removed")
    part.replace(dest)


def main() -> None:
    import pandas as pd
    import requests

    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=REPO / "data/eurocrops_chips/EE_2021_pilot")
    ap.add_argument("--out", type=Path, default=REPO / "data/alphaearth/aef_v1_annual")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--source", choices=sorted(SOURCES), default="mirror",
                    help="where the bytes come from; the MD5 is Google's either way")
    ap.add_argument("--list-only", action="store_true", help="print the selection, fetch nothing")
    args = ap.parse_args()

    root = args.root if args.root.is_absolute() else REPO / args.root
    out = args.out if args.out.is_absolute() else REPO / args.out
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    year, crs = manifest["year"], manifest["grid"]["crs"]

    index_path = out / "aef_index.parquet"
    if not index_path.exists():
        out.mkdir(parents=True, exist_ok=True)
        tmp = index_path.with_name(index_path.name + ".part")
        with requests.get(INDEX_URL, stream=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for block in r.iter_content(CHUNK):
                    f.write(block)
        tmp.replace(index_path)
    index = pd.read_parquet(index_path, columns=INDEX_COLUMNS)
    index["key"] = index["path"].map(key_of)

    bounds = np.array([c["bounds_3035"] for c in manifest["chips"]], dtype=float)
    chosen = select_tiles(index, bounds, crs, year)
    print(f"{len(chosen)} tiles for {len(bounds)} chips of {root.name}, "
          f"zones {sorted(chosen['utm_zone'].unique())}", flush=True)

    metas = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for key, meta in zip(chosen["key"], pool.map(official_meta, chosen["key"])):
            metas[key] = meta
    total = sum(m["size"] for m in metas.values())
    todo = [k for k in chosen["key"] if not (out / k).exists()]
    print(f"{total / 1e9:.1f} GB in all, {len(todo)} tiles "
          f"({sum(metas[k]['size'] for k in todo) / 1e9:.1f} GB) still to fetch", flush=True)
    if args.list_only:
        return

    t0 = time.time()
    failures = []

    def job(key: str) -> tuple[str, float]:
        t = time.time()
        for attempt in range(RETRIES + 1):
            try:
                fetch(key, out / key, metas[key], SOURCES[args.source])
                metas[key]["fetched_from"] = SOURCES[args.source]
                return key, time.time() - t
            except Exception:
                if attempt == RETRIES:
                    raise
                time.sleep(60 * (attempt + 1))

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(job, k): k for k in todo}
        for i, fut in enumerate(as_completed(futures), start=1):
            key = futures[fut]
            try:
                _, secs = fut.result()
                mb = metas[key]["size"] / 1e6
                print(f"[{i}/{len(todo)}] {key} {mb:.0f} MB in {secs:.0f} s "
                      f"({mb / max(secs, 1e-3):.0f} MB/s), MD5 ok", flush=True)
            except Exception as exc:
                failures.append({"key": key, "error": f"{type(exc).__name__}: {exc}"})
                print(f"[{i}/{len(todo)}] {key} FAILED {exc}", flush=True)

    # The table lists every tile of each touched zone, so the reader can tell a tile that does
    # not exist from one that was never fetched. It is merged with an earlier chip set's.
    table_path = out / str(year) / TILE_TABLE
    table = (json.loads(table_path.read_text()) if table_path.exists()
             else {"source": {"official": OFFICIAL_URL, "mirror": MIRROR_URL, "index": INDEX_URL},
                   "year": year, "zones": {}, "fetched": {}, "chip_sets": []})
    rows = index[index["year"] == year]
    for zone in sorted(chosen["utm_zone"].unique()):
        z = rows[rows["utm_zone"] == zone]
        table["zones"][zone] = {
            "crs": z["crs"].iloc[0],
            "tiles": [{"key": r.key, "utm": [r.utm_west, r.utm_south, r.utm_east, r.utm_north]}
                      for r in z.itertuples()]}
    for key in chosen["key"]:
        if (out / key).exists():
            table["fetched"][key] = {"fetched_from": None, **table["fetched"].get(key, {}),
                                     **metas[key]}
    sha = hashlib.sha256(manifest_bytes).hexdigest()
    table["chip_sets"] = [s for s in table["chip_sets"] if s["chip_manifest_sha256"] != sha] + [{
        "root": str(root.relative_to(REPO)) if root.is_relative_to(REPO) else str(root),
        "chip_manifest_sha256": sha, "tiles": sorted(chosen["key"]),
        "complete": not failures,
        "updated": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")}]
    table_path.parent.mkdir(parents=True, exist_ok=True)
    table_path.write_text(json.dumps(table, indent=1))
    if failures:
        sys.exit(f"{len(failures)} tiles failed after {RETRIES} retries; re-run to resume:\n"
                 + "\n".join(f"  {f['key']}: {f['error']}" for f in failures))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {table_path}")


if __name__ == "__main__":
    main()
