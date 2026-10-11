"""Draw the sample and download every 2021 Sentinel-2 L2A acquisition over its chips.

The read is the export's own (``gfm4agri.chips.s2_monthly``): the same search, the same stackstac call, the
same alignment on item ids and the same per-month fallback for an unreadable asset. Only the five bands the
screenings and the measures need are read, with the scene classification, and they are stored as raw digital
numbers, before the BOA offset, so every screening is applied offline to identical data.
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ROOT = REPO / "data" / "eurocrops_chips" / "EE_2021"
OUT = REPO / "results" / HERE.name / "acq"
BANDS = ["B02", "B03", "B04", "B08", "B11"]


def draw_sample(seed: int = 0) -> pd.DataFrame:
    chips = pd.DataFrame([{"chip_id": c["chip_id"], "col": c["col"], "row": c["row"],
                           "labelled_share": c["labelled_share"],
                           "filled_aug": c["imagery"]["filled_share_per_month"][7],
                           "filled_oct": c["imagery"]["filled_share_per_month"][9],
                           "bounds": c["bounds_3035"]}
                          for c in json.loads((ROOT / "manifest.json").read_text())["chips"]])
    rng = np.random.default_rng(seed)
    farm = chips[chips.labelled_share >= 0.15].copy()
    farm["cell"] = (pd.qcut(farm.col, 6, labels=False).astype(str) + "_"
                    + pd.qcut(farm.row, 4, labels=False).astype(str))
    picks = [g.iloc[rng.integers(len(g))] for _, g in farm.groupby("cell", sort=True)]
    out = pd.DataFrame(picks).assign(stratum="grid")
    cloudy = farm[(farm[["filled_aug", "filled_oct"]].max(axis=1) > 0.3)
                  & ~farm.chip_id.isin(out.chip_id)]
    extra = cloudy.iloc[rng.choice(len(cloudy), size=6, replace=False)].assign(stratum="cloudy")
    return pd.concat([out, extra]).drop(columns="cell").reset_index(drop=True)


def fetch(chip_id: str, bounds) -> str:
    import stackstac
    from pyproj import Transformer

    from gfm4agri.chips.s2_monthly import _align_on_ids, _baseline, _resampling, _search

    path = OUT / f"{chip_id}.npz"
    if path.exists():
        return f"{chip_id}: on disk"
    xmin, ymin, xmax, ymax = bounds
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    items = _search((min(lon), min(lat), max(lon), max(lat)), 2021, 80.0)
    kw = dict(epsg=3035, resolution=10, bounds=tuple(bounds), dtype="float32",
              fill_value=np.float32(np.nan), rescale=False, chunksize=2048)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spec = stackstac.stack(items, assets=BANDS, resampling=_resampling("bilinear"),
                               sortby_date=False, **kw)
        scl = stackstac.stack(items, assets=["SCL"], resampling=_resampling("nearest"),
                              sortby_date=False, **kw)
    spec, scl, ids = _align_on_ids(spec, scl)
    by_id = {it.id: it for it in items}
    months = np.array([by_id[i].datetime.month for i in ids])
    s_out, c_out, kept, dropped = [], [], [], []
    for m in range(1, 13):
        idx = np.where(months == m)[0]
        if not len(idx):
            continue
        try:
            s_out.append(spec.isel(time=idx).values)
            c_out.append(scl.isel(time=idx).values[:, 0])
            kept.extend(idx)
        except Exception:
            for j in idx:
                try:
                    sj, cj = spec.isel(time=[j]).values, scl.isel(time=[j]).values[:, 0]
                except Exception:
                    dropped.append(ids[j])
                    continue
                s_out.append(sj)
                c_out.append(cj)
                kept.append(j)
    s = np.concatenate(s_out)
    c = np.concatenate(c_out)
    meta = [{"id": ids[j], "datetime": by_id[ids[j]].datetime.isoformat(),
             "platform": by_id[ids[j]].properties.get("platform", ""),
             "tile": by_id[ids[j]].properties.get("s2:mgrs_tile", ""),
             "baseline": _baseline(by_id[ids[j]]),
             "cloud_cover": by_id[ids[j]].properties.get("eo:cloud_cover")} for j in kept]
    OUT.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, spec=np.nan_to_num(s, nan=0).clip(0, 65535).astype(np.uint16),
                        scl=np.where(np.isnan(c), 255, c).astype(np.uint8),
                        meta=json.dumps(meta), unreadable=json.dumps(dropped))
    tmp.rename(path)
    return f"{chip_id}: {len(kept)} acquisitions, {len(dropped)} unreadable"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    sample_path = HERE / "sample.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path, converters={"bounds": json.loads})
    else:
        sample = draw_sample()
        sample.assign(bounds=sample.bounds.map(json.dumps)).to_csv(sample_path, index=False)
        sample = pd.read_csv(sample_path, converters={"bounds": json.loads})
    print(f"{len(sample)} chips; {sample.stratum.value_counts().to_dict()}", flush=True)
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(fetch, r.chip_id, r.bounds): r.chip_id for r in sample.itertuples()}
        for f in as_completed(futs):
            try:
                print(f.result(), flush=True)
            except Exception as e:  # one failed chip must not stop the others; rerun resumes
                print(f"{futs[f]}: FAILED {type(e).__name__}: {e}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
