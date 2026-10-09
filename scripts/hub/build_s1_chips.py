"""Add Sentinel-1 RTC monthly composites to a chip set, on the same grid as its labels.

For every chip this writes ``<chip>_s1rtc.tif`` beside the chip's Sentinel-2 image: 24 bands
of float32 backscatter in decibels, twelve monthly composites of ``VV`` then ``VH``, in the
``(time channels)`` layout of the Sentinel-2 chips and on the chip's exact transform. The
format is TerraMind's ``S1RTC`` modality; the compositing is described in
:mod:`gfm4agri.chips.s1_monthly`.

Two ways to choose the chips:

* **Country mode**, the default: every cell of the 2,240 m grid holding an in-scheme parcel,
  the same cells :mod:`scripts.data.build_country_chips` exports, written into
  ``<root>/chips/``. It does not depend on the Sentinel-2 export having finished.
* ``--from-manifest``: the chips listed in ``<root>/manifest.json``, written into their split
  directories, which is how the pilot chip set gets its Sentinel-1.

Each chip's provenance goes to ``<root>/s1_reports/<chip>.json``, kept apart from the
Sentinel-2 reports so that neither manifest can pick up the other's. A chip is skipped when
its raster and report both exist, so the command resumes. ``<root>/s1_rtc.json`` records the
source, the format and the normalisation statistics, assembled from the reports at the end.

    poetry run python scripts/hub/build_s1_chips.py --country EE --year 2021 --workers 12
    poetry run python scripts/hub/build_s1_chips.py --from-manifest \
        --root data/eurocrops_chips/EE_2021_pilot
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.chips.s1_monthly import (  # noqa: E402
    COLLECTION,
    S1_BAND_NAMES,
    composite_window_s1,
    s1_window_report,
)
from gfm4agri.chips.s2_monthly import N_MONTHS  # noqa: E402
from gfm4agri.data.chip_grid import (  # noqa: E402
    CHIP_CRS, CHIP_PX, PIXEL_M, Chip, apply_vector_aliases, score_cells, vector_layer_aliases)

SUFFIX = "_s1rtc.tif"
SPLIT_DIRS = {"train": "training", "val": "validation", "test": "test"}
COUNTRY = {"EE": "Estonia"}


def write_tif(path: Path, arr: np.ndarray, chip: Chip) -> None:
    import rasterio

    desc = [f"M{m + 1:02d}_{b}" for m in range(N_MONTHS) for b in S1_BAND_NAMES]
    profile = dict(driver="GTiff", width=CHIP_PX, height=CHIP_PX, count=arr.shape[0],
                   dtype="float32", crs=CHIP_CRS, transform=chip.transform, nodata=np.nan,
                   compress="deflate", predictor=3, tiled=True,
                   blockxsize=CHIP_PX, blockysize=CHIP_PX)
    tmp = path.with_suffix(path.suffix + ".part")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr)
        for i, d in enumerate(desc, start=1):
            dst.set_band_description(i, d)
    os.replace(tmp, path)


def _export(task: tuple[str, int, int, str, str, int, int]) -> dict:
    """Composite and write one chip. Runs in a worker process."""
    cc, col, row, img_dir, rep_dir, year, retries = task
    chip = Chip(cc, col, row)
    t0 = time.time()
    last = None
    for attempt in range(retries + 1):
        try:
            arr, maps, report = composite_window_s1(chip.bounds, year)
            break
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(60 * (attempt + 1))
    else:
        return {"chip_id": chip.chip_id, "error": f"{type(last).__name__}: {last}"}
    rep = s1_window_report(maps, report)
    rep["seconds"] = round(time.time() - t0, 1)
    write_tif(Path(img_dir) / f"{chip.chip_id}{SUFFIX}", arr, chip)
    entry = {"chip_id": chip.chip_id, "col": col, "row": row,
             "bounds_3035": list(chip.bounds), "s1": rep}
    (Path(rep_dir) / f"{chip.chip_id}.json").write_text(json.dumps(entry, indent=1))
    return entry


def band_stats(paths: list[Path]) -> dict:
    """Per-band mean and standard deviation in dB, all months pooled, NaN ignored."""
    import rasterio

    s = np.zeros(2)
    s2 = np.zeros(2)
    n = np.zeros(2)
    for p in paths:
        with rasterio.open(p) as src:
            a = src.read().astype(np.float64).reshape(N_MONTHS, 2, -1)
        ok = np.isfinite(a)
        s += np.where(ok, a, 0).sum((0, 2))
        s2 += np.where(ok, a * a, 0).sum((0, 2))
        n += ok.sum((0, 2))
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 1e-12))
    return {"computed_on": f"{len(paths)} chips, all months pooled",
            "band_names": S1_BAND_NAMES, "means": [round(float(x), 4) for x in mean],
            "stds": [round(float(x), 4) for x in std]}


def assemble(root: Path, rep_dir: Path, img_path_of, year: int, config: dict,
             stats_sample: int, eligible: set[str] | None = None) -> None:
    """Write the sidecar. Normalisation statistics come from ``eligible`` chips only, which
    for a chip set with splits is its training chips, so validation never informs them."""
    entries = []
    for f in sorted(rep_dir.glob("*.json")):
        try:
            entries.append(json.loads(f.read_text()))
        except ValueError:
            print(f"  skipping unreadable {f.name}", flush=True)
    rng = np.random.default_rng(0)
    ids = [e["chip_id"] for e in entries if eligible is None or e["chip_id"] in eligible]
    sample = (sorted(rng.choice(ids, size=min(stats_sample, len(ids)), replace=False).tolist())
              if ids else [])
    sidecar = {
        "representation": "s1rtc",
        "target_modality": "TerraMind S1RTC: VV then VH, backscatter in dB",
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": config,
        "source": {"collection": f"Planetary Computer {COLLECTION}", "year": year,
                   "product": "radiometrically terrain-corrected gamma nought, linear power"},
        "file_suffix": SUFFIX, "band_names": S1_BAND_NAMES, "n_months": N_MONTHS,
        "layout": "(time channels): band index = month_index * 2 + band_index",
        "units": "decibels, 10 * log10 of the linear-power median",
        "dtype": "float32", "nodata": "NaN where a pixel was never observed",
        "composite": "per-pixel monthly median over every valid acquisition, both orbit "
                     "directions and all relative orbits, taken in linear power, then "
                     "converted to dB; missing months filled by linear interpolation",
        "resampling": "bilinear, in linear power",
        "grid": {"crs": CHIP_CRS, "pixel_m": PIXEL_M, "chip_px": CHIP_PX},
        "n_chips": len(entries),
        "normalisation": band_stats([img_path_of(i) for i in sample]) if sample else None,
    }
    (root / "s1_rtc.json").write_text(json.dumps(sidecar, indent=2))
    print(f"s1_rtc.json: {len(entries):,} chips -> {root / 's1_rtc.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--country", default="EE", choices=sorted(COUNTRY))
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--root", type=Path, default=None)
    ap.add_argument("--from-manifest", action="store_true",
                    help="take the chips from <root>/manifest.json instead of the country grid")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--stats-sample", type=int, default=200)
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()

    cc = args.country
    root = args.root or REPO / "data" / "eurocrops_chips" / f"{cc}_{args.year}"
    root = root if root.is_absolute() else REPO / root
    rep_dir = root / "s1_reports"
    rep_dir.mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    eligible = None
    if args.from_manifest:
        manifest = json.loads((root / "manifest.json").read_text())
        eligible = {c["chip_id"] for c in manifest["chips"] if c["split"] == "train"}
        dirs = {c["chip_id"]: root / f"{SPLIT_DIRS[c['split']]}_chips"
                for c in manifest["chips"]}
        cells = [(c["chip_id"], int(c["chip_id"].split("_")[1]), int(c["chip_id"].split("_")[2]))
                 for c in manifest["chips"]]
    else:
        import geopandas as gpd
        import yaml

        scheme = yaml.safe_load((REPO / "configs" / "class_scheme_eurocropsml.yaml").read_text())
        codes = [str(c) for c in scheme["in_country_class_lists"][COUNTRY[cc]]]
        src = REPO / "data" / "eurocrops" / "parquet" / f"{cc}_{args.year}.parquet"
        parcels = gpd.read_parquet(src, columns=["EC_hcat_c", "geometry"]).to_crs(CHIP_CRS)
        parcels["EC_hcat_c"] = apply_vector_aliases(parcels["EC_hcat_c"],
                                                    vector_layer_aliases(scheme))
        grid = score_cells(parcels, "EC_hcat_c", codes).sort_values(["col", "row"])
        (root / "chips").mkdir(parents=True, exist_ok=True)
        cells = [(f"{cc}_{int(c):05d}_{int(r):05d}", int(c), int(r))
                 for c, r in zip(grid["col"], grid["row"], strict=True)]
        dirs = {cid: root / "chips" for cid, _, _ in cells}

    def img_path_of(cid: str) -> Path:
        return dirs[cid] / f"{cid}{SUFFIX}"

    if args.assemble_only:
        assemble(root, rep_dir, img_path_of, args.year, config, args.stats_sample, eligible)
        return

    todo = [(cid, c, r) for cid, c, r in cells
            if not (img_path_of(cid).exists() and (rep_dir / f"{cid}.json").exists())]
    n_have = len(cells) - len(todo)
    if args.limit:
        todo = todo[: args.limit]
    print(f"{len(cells):,} chips, {n_have:,} already have Sentinel-1, "
          f"{len(todo):,} to export", flush=True)

    tasks = [(cc, c, r, str(dirs[cid]), str(rep_dir), args.year, args.retries)
             for cid, c, r in todo]
    done, failures, t1 = 0, [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_export, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            try:
                e = fut.result()
            except Exception as exc:
                t = futures[fut]
                e = {"chip_id": f"{t[0]}_{t[1]:05d}_{t[2]:05d}",
                     "error": f"{type(exc).__name__}: {exc}"}
            rate = done / max(time.time() - t1, 1e-9)
            eta_h = (len(tasks) - done) / rate / 3600
            if "error" in e:
                failures.append(e)
                (root / "s1_failures.json").write_text(json.dumps(failures, indent=1))
                print(f"[{done}/{len(tasks)}] FAILED {e['chip_id']}: {e['error']}", flush=True)
            else:
                print(f"[{done}/{len(tasks)}] {e['chip_id']} {e['s1']['seconds']:.0f}s "
                      f"eta {eta_h:.1f} h", flush=True)
    print(f"exported {done - len(failures):,}, failed {len(failures):,}, "
          f"in {(time.time() - t1) / 3600:.2f} h", flush=True)
    assemble(root, rep_dir, img_path_of, args.year, config, args.stats_sample, eligible)


if __name__ == "__main__":
    main()
