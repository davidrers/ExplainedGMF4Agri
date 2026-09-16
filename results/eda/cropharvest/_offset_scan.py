"""Exhaustive label-to-pixel offset scan over every CropHarvest feature array.

Reads only the HDF5 attributes, never the (12, 18) array, so it is much cheaper than
the phenology scan. Caches to offset_scan.parquet next to this file.

The offset is the great-circle distance between the label coordinate stored in the .h5
attributes (`label_lat`, `label_lon`, which for a polygon is its centroid) and the
coordinate of the pixel the CropHarvest engineer actually sampled (`instance_lat`,
`instance_lon`). It should never exceed one pixel diagonal, roughly 15 m at 10 m
resolution. Where it does, the instance carries the spectral signature of a different
place while keeping its original label.

    python results/eda/cropharvest/_offset_scan.py
"""
from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

HERE = Path(__file__).parent.resolve()
REPO = HERE.parents[2]
DATA_ROOT = Path(os.environ.get("CROPHARVEST_DATA", REPO / "data" / "cropharvest"))
ARRAYS_DIR = DATA_ROOT / "features" / "arrays"
OUT = HERE / "offset_scan.parquet"
OUT_JSON = HERE / "offset_scan.json"

PIXEL_M = 10.0
TOLERANCE_M = PIXEL_M * np.sqrt(2)  # one pixel diagonal, 14.14 m


def main() -> None:
    files = sorted(ARRAYS_DIR.glob("*.h5"))
    print(f"scanning {len(files):,} arrays for label-to-pixel offset ...")
    idx, dss, offs = [], [], []
    for n, p in enumerate(files):
        i, _, ds = p.stem.partition("_")
        with h5py.File(p, "r") as h:
            a = h.attrs
            la, lo = float(a["label_lat"]), float(a["label_lon"])
            ia, io = float(a["instance_lat"]), float(a["instance_lon"])
        idx.append(int(i))
        dss.append(ds)
        offs.append(float(np.hypot((ia - la) * 111_320.0,
                                   (io - lo) * 111_320.0 * np.cos(np.radians(la)))))
        if (n + 1) % 10_000 == 0:
            print(f"  {n+1:,} / {len(files):,}")

    df = pd.DataFrame({"index": idx, "dataset": dss, "pixel_offset_m": offs})
    df["offset_ok"] = df.pixel_offset_m <= TOLERANCE_M
    df.to_parquet(OUT, index=False)
    print(f"wrote {OUT.name} ({len(df):,} rows)")

    per_ds = (
        df.groupby("dataset")
        .agg(n=("pixel_offset_m", "size"),
             n_bad=("offset_ok", lambda s: int((~s).sum())),
             median_m=("pixel_offset_m", "median"),
             max_km=("pixel_offset_m", lambda s: s.max() / 1000))
        .assign(pct_bad=lambda d: (d.n_bad / d.n * 100).round(2))
        .sort_values("pct_bad", ascending=False)
    )
    summary = {
        "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "arrays_scanned": int(len(df)),
        "tolerance_m": round(float(TOLERANCE_M), 2),
        "n_ok": int(df.offset_ok.sum()),
        "n_bad": int((~df.offset_ok).sum()),
        "pct_bad": round(float((~df.offset_ok).mean() * 100), 2),
        "n_over_1km": int((df.pixel_offset_m > 1000).sum()),
        "pct_over_1km": round(float((df.pixel_offset_m > 1000).mean() * 100), 2),
        "median_m": round(float(df.pixel_offset_m.median()), 2),
        "p95_m": round(float(df.pixel_offset_m.quantile(0.95)), 2),
        "p99_m": round(float(df.pixel_offset_m.quantile(0.99)), 2),
        "max_km": round(float(df.pixel_offset_m.max() / 1000), 1),
        "per_dataset": per_ds.round(2).reset_index().to_dict("records"),
        "fully_affected_datasets": sorted(per_ds[per_ds.pct_bad >= 99.9].index.tolist()),
    }
    OUT_JSON.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {OUT_JSON.name}")
    print(per_ds.head(12).to_string())
    print(f"\ntotal affected: {summary['n_bad']:,} of {summary['arrays_scanned']:,} "
          f"({summary['pct_bad']} %)")


if __name__ == "__main__":
    main()
