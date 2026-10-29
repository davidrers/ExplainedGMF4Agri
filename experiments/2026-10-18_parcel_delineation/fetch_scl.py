"""The July scene classification (SCL) of the example chips, so ``report.py`` can show the predicted crop types
only where Sentinel-2 saw vegetation.

The chips keep no SCL: the compositor uses it to screen clouds and drops it. This reads every Sentinel-2 L2A
acquisition of July 2021 over a chip from the Planetary Computer, as ``gfm4agri.chips.s2_monthly`` does, onto the
chip grid (EPSG:3035, 10 m, nearest neighbour from 20 m). An acquisition sees a pixel clearly where its class is
vegetation (4), not vegetated (5) or water (6), the classes the ``revised`` screening keeps; the pixel's July class is
the most frequent of the three over those acquisitions, ties going to the lower class, and 0 where none sees it.

Writes ``results/<name>/<set>/scl_july.npz``: per chip id, the July class (``<id>``) and the number of clear
acquisitions (``<id>_n``); chips already there are skipped. Without ``--ids`` the chips are those of ``example_<n>.png``:

    poetry run python experiments/2026-10-18_parcel_delineation/fetch_scl.py --chips EE_2021_rev
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402

CLEAR = (4, 5, 6)
YEAR, MONTH = 2021, 7


def july_scl(bounds_3035: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    """The July class and the number of clear acquisitions per pixel of one chip window."""
    import stackstac
    from pyproj import Transformer

    from gfm4agri.chips.s2_monthly import _resampling, _search

    xmin, ymin, xmax, ymax = bounds_3035
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    items = [it for it in _search((min(lon), min(lat), max(lon), max(lat)), YEAR, 80.0)
             if it.datetime.month == MONTH]
    if not items:
        raise RuntimeError(f"no Sentinel-2 L2A items over {bounds_3035} in {YEAR}-{MONTH:02d}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scl = stackstac.stack(items, assets=["SCL"], resampling=_resampling("nearest"), epsg=3035, resolution=10,
                              bounds=bounds_3035, dtype="float32", fill_value=np.float32(np.nan), rescale=False,
                              chunksize=2048).values[:, 0]
    counts = np.stack([(scl == k).sum(0) for k in CLEAR])
    n = counts.sum(0)
    cls = np.where(n > 0, np.asarray(CLEAR)[counts.argmax(0)], 0).astype(np.uint8)
    return cls, n.astype(np.int16)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chips", required=True, help="the chip set's name, e.g. EE_2021_rev")
    ap.add_argument("--ids", nargs="*", help="chip ids; by default the chips of example_<n>.png")
    ap.add_argument("--name", default=None, help="the results' experiment name, by default experiment.yaml's")
    args = ap.parse_args()

    import rasterio

    from gfm4agri.pipeline.config import load_experiment

    name = args.name or load_experiment(HERE / "experiment.yaml")["name"]
    results = REPO / "results" / name / args.chips
    chips_dir = REPO / "data/eurocrops_chips" / args.chips / "chips"
    ids = args.ids
    if not ids:
        import report

        first = next(iter(sorted(results.glob("*/P*_draw0_seed0/predictions_test.npz"))))
        ids = report.pick_chips(chips_dir, [str(c) for c in np.load(first)["ids"]])
    path = results / "scl_july.npz"
    out = dict(np.load(path)) if path.exists() else {}
    for cid in ids:
        if cid in out:
            continue
        with rasterio.open(chips_dir / f"{cid}_merged.tif") as s:
            bounds = tuple(s.bounds)
        cls, n = july_scl(bounds)
        out[cid], out[f"{cid}_n"] = cls, n
        shares = {k: f"{np.mean(cls == k):.0%}" for k in (0, *CLEAR)}
        print(f"{cid}: July class shares {shares}, clear acquisitions {n.min()} to {n.max()}", flush=True)
        np.savez_compressed(path, **out)
    print(f"{len([k for k in out if not k.endswith('_n')])} chips in {path}")


if __name__ == "__main__":
    main()
