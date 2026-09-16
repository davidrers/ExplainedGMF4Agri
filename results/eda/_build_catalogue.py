"""Build a parcel catalogue over every EuroCropsML .npz file on disk.

One row per parcel:
    path, country, nuts, parcel_id, hcat, n_timesteps, first_date, last_date,
    lat, lon, m01..m12 (observations falling in each calendar month of 2021)

Output
------
    data/catalogue_parcels_full.parquet
    results/eda/cache/catalogue_config.json   (configuration + wall-clock time)
    results/eda/cache/date_histogram.parquet  (observations per acquisition date per country)

Usage
-----
    python results/eda/_build_catalogue.py [--workers 16] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
DATA_ROOT = Path(os.environ.get("EUROCROPSML_DATA", REPO / "data" / "eurocropsml")).expanduser()
PREPROCESS_DIR = DATA_ROOT / "preprocess"
OUT_PARQUET = REPO / "data" / "catalogue_parcels_full.parquet"
CACHE_DIR = HERE / "cache"
OUT_CONFIG = CACHE_DIR / "catalogue_config.json"
OUT_DATEHIST = CACHE_DIR / "date_histogram.parquet"

NUTS_PREFIX = {"EE": "Estonia", "LV": "Latvia", "PT": "Portugal"}
FNAME_RE = re.compile(r"^(?P<nuts>[A-Z]{2}[A-Z0-9]+)_(?P<pid>\d+)_(?P<cls>\d+)\.npz$")
SEED = 42
CHUNK = 1000


def _scan_chunk(args: tuple[str, list[str]]) -> tuple[list[tuple], dict]:
    """Read dates/center from a chunk of files. Returns rows and a date histogram."""
    directory, names = args
    rows: list[tuple] = []
    hist: Counter = Counter()
    for name in names:
        m = FNAME_RE.match(name)
        if m is None:
            continue
        nuts = m.group("nuts")
        country = NUTS_PREFIX.get(nuts[:2], "Unknown")
        path = os.path.join(directory, name)
        try:
            with np.load(path) as z:
                dates = z["dates"]
                center = np.asarray(z["center"], dtype="float64").ravel()
        except Exception:
            rows.append((name, country, nuts, m.group("pid"), m.group("cls"), 0,
                         None, None, np.nan, np.nan) + (0,) * 12)
            continue
        n_t = int(dates.shape[0])
        if n_t:
            d = np.asarray(dates, dtype="datetime64[D]")
            first, last = d.min(), d.max()
            months = d.astype("datetime64[M]").astype(int) % 12 + 1
            mcounts = np.bincount(months, minlength=13)[1:13]
            for dd, cc in zip(*np.unique(d, return_counts=True)):
                hist[(country, str(dd))] += int(cc)
        else:
            first = last = None
            mcounts = np.zeros(12, dtype=int)
        lon = float(center[0]) if center.size >= 1 else np.nan
        lat = float(center[1]) if center.size >= 2 else np.nan
        rows.append((name, country, nuts, m.group("pid"), m.group("cls"), n_t,
                     None if first is None else str(first),
                     None if last is None else str(last),
                     lat, lon) + tuple(int(v) for v in mcounts))
    return rows, dict(hist)


def main(workers: int, limit: int | None) -> None:
    if not PREPROCESS_DIR.is_dir():
        raise SystemExit(f"preprocess directory not found: {PREPROCESS_DIR}")

    t0 = time.perf_counter()
    names = sorted(e.name for e in os.scandir(PREPROCESS_DIR) if e.name.endswith(".npz"))
    t_list = time.perf_counter() - t0
    print(f"listed {len(names):,} .npz files in {t_list:.1f} s", flush=True)
    if limit:
        rng = np.random.default_rng(SEED)
        names = sorted(rng.choice(np.asarray(names), size=min(limit, len(names)), replace=False).tolist())
        print(f"limited to a random sample of {len(names):,} files (seed {SEED})", flush=True)

    directory = str(PREPROCESS_DIR)
    chunks = [(directory, names[i:i + CHUNK]) for i in range(0, len(names), CHUNK)]

    t1 = time.perf_counter()
    rows: list[tuple] = []
    hist: Counter = Counter()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for r, h in ex.map(_scan_chunk, chunks, chunksize=1):
            rows.extend(r)
            hist.update(h)
            done += 1
            if done % 50 == 0 or done == len(chunks):
                el = time.perf_counter() - t1
                print(f"  {done}/{len(chunks)} chunks | {len(rows):,} parcels | {el:.0f} s", flush=True)
    t_scan = time.perf_counter() - t1

    cols = (["file", "country", "nuts", "parcel_id", "hcat", "n_timesteps",
             "first_date", "last_date", "lat", "lon"] + [f"m{i:02d}" for i in range(1, 13)])
    df = pd.DataFrame.from_records(rows, columns=cols)
    df["path"] = directory + os.sep + df["file"]
    df["hcat"] = df["hcat"].astype("string")
    df["parcel_id"] = df["parcel_id"].astype("string")
    df["first_date"] = pd.to_datetime(df["first_date"])
    df["last_date"] = pd.to_datetime(df["last_date"])
    for c in [f"m{i:02d}" for i in range(1, 13)]:
        df[c] = df[c].astype("int16")
    df["n_timesteps"] = df["n_timesteps"].astype("int32")
    df = df[["path", "country", "nuts", "parcel_id", "hcat", "n_timesteps",
             "first_date", "last_date", "lat", "lon"] + [f"m{i:02d}" for i in range(1, 13)]]

    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET, index=False)
    print(f"wrote {OUT_PARQUET} ({len(df):,} rows)", flush=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    hdf = pd.DataFrame(
        [(c, d, n) for (c, d), n in sorted(hist.items())],
        columns=["country", "date", "n_observations"],
    )
    hdf["date"] = pd.to_datetime(hdf["date"])
    hdf.to_parquet(OUT_DATEHIST, index=False)
    print(f"wrote {OUT_DATEHIST} ({len(hdf):,} rows)", flush=True)

    total = time.perf_counter() - t0
    cfg = {
        "script": "results/eda/_build_catalogue.py",
        "preprocess_dir": str(PREPROCESS_DIR),
        "n_files_listed": len(names),
        "n_parcels_catalogued": int(len(df)),
        "is_full_scan": limit is None,
        "sample_limit": limit,
        "random_seed": SEED,
        "workers": workers,
        "chunk_size": CHUNK,
        "seconds_listing": round(t_list, 1),
        "seconds_scanning": round(t_scan, 1),
        "seconds_total": round(total, 1),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "pandas": pd.__version__,
    }
    OUT_CONFIG.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    print(json.dumps(cfg, indent=2), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 4))
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    main(args.workers, args.limit)
