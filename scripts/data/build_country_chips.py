"""Export every EuroCrops chip of one country, in parallel and resumably.

This is the full-country counterpart of ``build_pilot_chips.py``. That script picks a
handful of well-labelled, widely separated cells to wire up the pipeline; this one takes
**every** cell of the 2,240 m grid that holds an in-scheme parcel, which for Estonia 2021 is
7,398 chips.

Three differences from the pilot follow from the scale:

* **No split is assigned at export time.** Chips are written to a single ``chips/``
  directory. The train, validation and test partition is the spatial block assignment of
  :mod:`gfm4agri.data.blocks`, which is a Phase 1 decision, and materialising it here would
  freeze it into the export.
* **Each chip writes its own report sidecar** next to its rasters, so an interrupted run
  loses at most the chip in flight. ``manifest.json`` is assembled from the sidecars at the
  end, and ``--assemble-only`` rebuilds it from whatever is on disk.
* **Chips are composited in a process pool.** Compositing is dominated by reading COGs from
  the Planetary Computer, so the work is network bound and parallelises well. Each worker
  loads the parcel layer once.

A chip is skipped when its three rasters and its report are all present, so re-running the
command resumes. Transient network failures are retried, and a chip that still fails is
recorded in ``failures.json`` and does not stop the run.

    poetry run python scripts/data/build_country_chips.py --country EE --year 2021 --workers 12
    poetry run python scripts/data/build_country_chips.py --assemble-only
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

from gfm4agri.chips.s2_monthly import (  # noqa: E402
    BAND_NAMES,
    BANDS,
    N_MONTHS,
    composite_window,
    window_report,
)
from gfm4agri.data.chip_grid import (  # noqa: E402
    CHIP_CRS,
    CHIP_PX,
    IGNORE_INDEX,
    PIXEL_M,
    Chip,
    apply_vector_aliases,
    rasterise_labels,
    score_cells,
    vector_layer_aliases,
)

COUNTRY = {"EE": ("Estonia", "pollu_id")}

#: Set once per worker process by :func:`_init_worker`.
_PARCELS = None
_CODES: list[str] = []
_NAMES: list[str] = []
_ID_COL = ""
_CC = ""


def load_classes(country: str) -> tuple[list[str], list[str]]:
    import yaml

    scheme = yaml.safe_load((REPO / "configs" / "class_scheme_eurocropsml.yaml").read_text())
    codes = [str(c) for c in scheme["in_country_class_lists"][country]]
    names = {str(c["hcat"]): c["thesis_class"] for c in scheme["classes"]}
    return codes, [names[c] for c in codes]


def load_parcels(cc: str, year: int, id_col: str):
    """The vector layer in EPSG:3035, with its HCAT codes mapped onto the scheme's."""
    import geopandas as gpd
    import yaml

    p = gpd.read_parquet(REPO / "data" / "eurocrops" / "parquet" / f"{cc}_{year}.parquet",
                         columns=[id_col, "EC_hcat_c", "geometry"]).to_crs(CHIP_CRS)
    p["geometry"] = p.geometry.make_valid()
    scheme = yaml.safe_load((REPO / "configs" / "class_scheme_eurocropsml.yaml").read_text())
    p["EC_hcat_c"] = apply_vector_aliases(p["EC_hcat_c"], vector_layer_aliases(scheme))
    return p


def write_tif(path: Path, arr: np.ndarray, chip: Chip,
              descriptions: list[str] | None = None) -> None:
    import rasterio

    arr = arr if arr.ndim == 3 else arr[None]
    profile = dict(driver="GTiff", width=CHIP_PX, height=CHIP_PX, count=arr.shape[0],
                   dtype=arr.dtype, crs=CHIP_CRS, transform=chip.transform,
                   compress="deflate", predictor=2, tiled=True, blockxsize=CHIP_PX,
                   blockysize=CHIP_PX)
    tmp = path.with_suffix(path.suffix + ".part")
    with rasterio.open(tmp, "w", **profile) as dst:
        dst.write(arr)
        for i, d in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, d)
    os.replace(tmp, path)


def paths_of(out: Path, chip_id: str) -> dict[str, Path]:
    d = out / "chips"
    return {"img": d / f"{chip_id}_merged.tif", "mask": d / f"{chip_id}.mask.tif",
            "ids": d / f"{chip_id}.parcels.tif", "report": d / f"{chip_id}.report.json"}


def is_done(out: Path, chip_id: str) -> bool:
    return all(p.exists() for p in paths_of(out, chip_id).values())


def _init_worker(cc: str, year: int, id_col: str, codes: list[str], names: list[str]) -> None:
    global _PARCELS, _CODES, _NAMES, _ID_COL, _CC
    _CODES, _NAMES, _ID_COL, _CC = codes, names, id_col, cc
    _PARCELS = load_parcels(cc, year, id_col)
    _ = _PARCELS.sindex  # build the spatial index once, not per chip


def _export_group(task: tuple[list[tuple[int, int]], str, int, float, int]) -> list[dict]:
    """Composite a group of adjacent chips in one read, then cut and write each chip.

    Runs in a worker process. A group of one is the chip-by-chip export.
    """
    members, out_s, year, cloud_cover_max, retries = task
    out = Path(out_s)
    chips = [Chip(_CC, c, r) for c, r in members]
    bounds = (min(ch.bounds[0] for ch in chips), min(ch.bounds[1] for ch in chips),
              max(ch.bounds[2] for ch in chips), max(ch.bounds[3] for ch in chips))
    t0 = time.time()

    last = None
    for attempt in range(retries + 1):
        try:
            arr, maps, report = composite_window(bounds, year, cloud_cover_max=cloud_cover_max)
            break
        except Exception as exc:  # network, STAC, or a scene that fails to read
            last = exc
            if attempt < retries:
                # The archive returns read errors in bursts lasting minutes, so the backoff
                # has to outlast the burst rather than merely pause between attempts; a chip
                # that still fails is picked up by the next pass.
                time.sleep(60 * (attempt + 1))
    else:
        return [{"chip_id": ch.chip_id, "error": f"{type(last).__name__}: {last}"} for ch in chips]
    window_s = round(time.time() - t0, 1)

    desc = [f"M{m + 1:02d}_{b}" for m in range(N_MONTHS) for b in BAND_NAMES]
    entries = []
    for chip in chips:
        p = paths_of(out, chip.chip_id)
        r0 = int(round((bounds[3] - chip.bounds[3]) / PIXEL_M))
        c0 = int(round((chip.bounds[0] - bounds[0]) / PIXEL_M))
        rows, cols = slice(r0, r0 + CHIP_PX), slice(c0, c0 + CHIP_PX)
        report_c = window_report(maps, report, rows, cols)
        report_c["seconds"] = round(window_s / len(chips), 1)
        if len(chips) > 1:
            report_c["window_bounds_3035"] = list(bounds)
            report_c["window_chips"] = len(chips)
            report_c["window_seconds"] = window_s

        mask, ids = rasterise_labels(chip, _PARCELS, "EC_hcat_c", _ID_COL, _CODES)
        write_tif(p["img"], np.ascontiguousarray(arr[:, rows, cols]), chip, desc)
        write_tif(p["mask"], mask, chip)
        write_tif(p["ids"], ids, chip)

        counts = np.bincount(mask[mask >= 0], minlength=len(_CODES))
        entry = {
            "chip_id": chip.chip_id, "col": chip.col, "row": chip.row,
            "bounds_3035": list(chip.bounds),
            "labelled_share": round(float((mask >= 0).mean()), 4),
            "n_parcels": int(len(np.unique(ids[ids > 0]))),
            "pixels_per_class": {_NAMES[k]: int(v) for k, v in enumerate(counts) if v},
            "imagery": report_c,
        }
        p["report"].write_text(json.dumps(entry, indent=1))
        entries.append(entry)
    return entries


def _relabel(task: tuple[str, str, list[int]]) -> dict:
    """Re-rasterise one chip's dense mask from the vector layer, imagery untouched.

    Runs in a worker process. The parcel raster must come out identical, since the aliases
    change codes and never geometry, and a changed pixel must go from ignore to one of
    ``targets``, the class indices the aliases map onto; anything else is an error.
    """
    import rasterio

    chip_id, out_s, targets = task
    p = paths_of(Path(out_s), chip_id)
    entry = json.loads(p["report"].read_text())
    chip = Chip(_CC, int(entry["col"]), int(entry["row"]))
    mask, ids = rasterise_labels(chip, _PARCELS, "EC_hcat_c", _ID_COL, _CODES)
    with rasterio.open(p["mask"]) as src:
        before = src.read(1)
    with rasterio.open(p["ids"]) as src:
        if not np.array_equal(src.read(1), ids):
            raise RuntimeError(f"{chip_id}: parcel raster differs from the one on disk")
    diff = mask != before
    if diff.any() and not ((before[diff] == IGNORE_INDEX) & np.isin(mask[diff], targets)).all():
        raise RuntimeError(f"{chip_id}: a relabelled pixel is not ignore -> aliased class")
    if diff.any():
        write_tif(p["mask"], mask, chip)
    counts = np.bincount(mask[mask >= 0], minlength=len(_CODES))
    entry.update({
        "labelled_share": round(float((mask >= 0).mean()), 4),
        "n_parcels": int(len(np.unique(ids[ids > 0]))),
        "pixels_per_class": {_NAMES[k]: int(v) for k, v in enumerate(counts) if v},
    })
    p["report"].write_text(json.dumps(entry, indent=1))
    return {"chip_id": chip_id, "pixels_changed": int(diff.sum())}


def relabel(out: Path, cc: str, year: int, id_col: str, workers: int, stats_sample: int) -> None:
    """Re-rasterise every chip's mask under the current class scheme and reassemble."""
    import yaml

    country, _ = COUNTRY[cc]
    codes, names = load_classes(country)
    scheme = yaml.safe_load((REPO / "configs" / "class_scheme_eurocropsml.yaml").read_text())
    aliases = vector_layer_aliases(scheme)
    targets = sorted({codes.index(t) for t in aliases.values() if t in codes})
    previous = json.loads((out / "manifest.json").read_text())
    ids = sorted(f.name[: -len(".report.json")] for f in (out / "chips").glob("*.report.json"))
    t0 = time.time()
    results = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker,
                             initargs=(cc, year, id_col, codes, names)) as pool:
        for i, r in enumerate(pool.map(_relabel, [(c, str(out), targets) for c in ids],
                                       chunksize=16), start=1):
            results.append(r)
            if i % 500 == 0 or i == len(ids):
                print(f"[{i}/{len(ids)}] relabelled, {time.time() - t0:.0f} s", flush=True)
    changed = [r for r in results if r["pixels_changed"]]
    record = {
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "reason": "vector_layer_aliases of the class scheme applied to the vector layer",
        "aliases": aliases,
        "chips_changed": len(changed),
        "pixels_changed": int(sum(r["pixels_changed"] for r in changed)),
        "previous_manifest_created": previous.get("created"),
    }
    # A cell holding no in-scheme parcel before the aliases but one after would be a new chip
    # with no imagery yet; it is counted here and not exported.
    parcels = load_parcels(cc, year, id_col)
    cells = score_cells(parcels, "EC_hcat_c", codes)
    have = set(ids)
    missing = [f"{cc}_{int(c):05d}_{int(r):05d}" for c, r in zip(cells["col"], cells["row"])
               if f"{cc}_{int(c):05d}_{int(r):05d}" not in have]
    record["cells_without_chip"] = missing
    assemble(out, cc, year, previous.get("config", {}), stats_sample,
             extra={"relabelled": previous.get("relabelled", []) + [record]})
    print(json.dumps({k: v for k, v in record.items() if k != "cells_without_chip"}
                     | {"cells_without_chip": len(missing)}, indent=1), flush=True)


def band_stats(out: Path, chip_ids: list[str]) -> dict:
    """Per-band mean and standard deviation over all months of ``chip_ids``."""
    import rasterio

    s = np.zeros(len(BANDS))
    s2 = np.zeros(len(BANDS))
    n = 0
    for cid in chip_ids:
        with rasterio.open(out / "chips" / f"{cid}_merged.tif") as src:
            a = src.read().astype(np.float64).reshape(N_MONTHS, len(BANDS), -1)
        s += a.sum((0, 2))
        s2 += (a**2).sum((0, 2))
        n += a.shape[0] * a.shape[2]
    mean = s / n
    std = np.sqrt(s2 / n - mean**2)
    return {"computed_on": f"{len(chip_ids)} chips sampled across the country, all months pooled",
            "band_names": BAND_NAMES, "means": [round(x, 2) for x in mean],
            "stds": [round(x, 2) for x in std]}


def assemble(out: Path, cc: str, year: int, args_config: dict, stats_sample: int,
             extra: dict | None = None) -> dict:
    """Build manifest.json from the per-chip report sidecars on disk.

    The relabelling record of an existing manifest is carried over, so a later export into
    the same chip set does not lose the history of its masks.
    """
    country, _ = COUNTRY[cc]
    if extra is None and (out / "manifest.json").exists():
        prior = json.loads((out / "manifest.json").read_text())
        extra = {"relabelled": prior["relabelled"]} if "relabelled" in prior else None
    codes, names = load_classes(country)
    entries = []
    for f in sorted((out / "chips").glob("*.report.json")):
        try:
            entries.append(json.loads(f.read_text()))
        except ValueError:
            print(f"  skipping unreadable {f.name}", flush=True)
    entries.sort(key=lambda e: e["chip_id"])

    manifest = {
        "dataset": f"eurocrops_s2_seg_{cc}_{year}",
        "purpose": "full-country export; the train, validation and test partition is assigned "
                   "later from the spatial block split and is deliberately not baked in here",
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": args_config,
        "country": cc, "year": year,
        "label_source": f"data/eurocrops/parquet/{cc}_{year}.parquet (EuroCrops v11)",
        "class_scheme": "configs/class_scheme_eurocropsml.yaml, in_country_class_lists." + country,
        "classes": [{"index": i, "hcat_code": c, "name": nm}
                    for i, (c, nm) in enumerate(zip(codes, names, strict=True))],
        "ignore_index": IGNORE_INDEX,
        "label_rule": "pixel centre inside an in-scheme parcel; all other pixels "
                      "ignore_index; dense",
        "grid": {"crs": CHIP_CRS, "pixel_m": PIXEL_M, "chip_px": CHIP_PX,
                 "chip_id": "<CC>_<col>_<row>, cell of a 2240 m grid anchored on "
                            "the EPSG:3035 origin"},
        "imagery": {"source": "Planetary Computer sentinel-2-l2a", "bands": BANDS,
                    "band_names": BAND_NAMES, "n_months": N_MONTHS,
                    "layout": "(time channels): band index = month_index * 12 + band_index",
                    "units": "surface reflectance x 10000, BOA offset removed "
                             "(DN / 10000 convention)",
                    "composite": "per-pixel monthly median of SCL-clear acquisitions, gaps filled "
                                 "by linear interpolation across months",
                    "resampling": "bilinear for spectral bands, nearest for SCL"},
        **(extra or {}),
        "chips": entries,
    }
    ids = [e["chip_id"] for e in entries]
    if ids:
        rng = np.random.default_rng(0)
        sample = sorted(rng.choice(ids, size=min(stats_sample, len(ids)), replace=False).tolist())
        manifest["normalisation"] = band_stats(out, sample)
        totals = np.zeros(len(codes), dtype=np.int64)
        for e in entries:
            for k, v in e["pixels_per_class"].items():
                totals[names.index(k)] += v
        manifest["pixels_per_class_total"] = {
            nm: int(t) for nm, t in zip(names, totals, strict=True)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"manifest: {len(entries):,} chips -> {out / 'manifest.json'}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--country", default="EE", choices=sorted(COUNTRY))
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--min-label-share", type=float, default=0.0,
                    help="keep cells at or above this in-scheme labelled area share; "
                         "0 keeps every cell holding an in-scheme parcel")
    ap.add_argument("--cloud-cover-max", type=float, default=80.0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=1,
                    help="composite G x G blocks of adjacent chips in one read, so each COG "
                         "block is fetched once per group rather than once per chip; 1 is "
                         "the chip-by-chip export")
    ap.add_argument("--cells", type=str, default=None, metavar="C0,R0,C1,R1",
                    help="restrict to grid cells with C0 <= col <= C1 and R0 <= row <= R1")
    ap.add_argument("--limit", type=int, default=None,
                    help="export at most this many groups (chips, at group size 1)")
    ap.add_argument("--stats-sample", type=int, default=200,
                    help="chips sampled for the band statistics")
    ap.add_argument("--assemble-only", action="store_true",
                    help="rebuild manifest.json from the sidecars already on disk")
    ap.add_argument("--relabel", action="store_true",
                    help="re-rasterise every chip's mask under the current class scheme, "
                         "imagery untouched, and rebuild manifest.json")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cc = args.country
    country, id_col = COUNTRY[cc]
    out = args.out or REPO / "data" / "eurocrops_chips" / f"{cc}_{args.year}"
    (out / "chips").mkdir(parents=True, exist_ok=True)
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}

    if args.assemble_only:
        assemble(out, cc, args.year, config, args.stats_sample)
        return
    if args.relabel:
        relabel(out, cc, args.year, id_col, args.workers, args.stats_sample)
        return

    codes, names = load_classes(country)
    t0 = time.time()
    parcels = load_parcels(cc, args.year, id_col)
    print(f"{len(parcels):,} parcels loaded in {time.time() - t0:.0f} s", flush=True)

    cells = score_cells(parcels, "EC_hcat_c", codes)
    cells = cells[cells["label_share"] >= args.min_label_share]
    if args.cells:
        c0, r0, c1, r1 = (int(v) for v in args.cells.split(","))
        cells = cells[cells["col"].between(c0, c1) & cells["row"].between(r0, r1)]
    cells = cells.sort_values(["col", "row"])
    todo = [(int(c), int(r)) for c, r in zip(cells["col"], cells["row"], strict=True)
            if not is_done(out, f"{cc}_{int(c):05d}_{int(r):05d}")]

    # Groups are cells of a coarser grid aligned on the chip grid, so a group's window is
    # itself on the chip grid and every chip cut from it keeps its own transform.
    g = max(args.group_size, 1)
    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for c, r in todo:
        groups.setdefault((c // g, r // g), []).append((c, r))
    group_list = [sorted(m) for _, m in sorted(groups.items())]
    if args.limit:
        group_list = group_list[: args.limit]
    n_chips = sum(len(m) for m in group_list)
    print(f"{len(cells):,} cells at or above label share {args.min_label_share}, "
          f"{len(cells) - len(todo):,} already on disk, {n_chips:,} to export in "
          f"{len(group_list):,} groups of up to {g} x {g}", flush=True)
    if not group_list:
        assemble(out, cc, args.year, config, args.stats_sample)
        return

    tasks = [(m, str(out), args.year, args.cloud_cover_max, args.retries) for m in group_list]
    done = 0
    failures = []
    t1 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers, initializer=_init_worker,
                             initargs=(cc, args.year, id_col, codes, names)) as pool:
        futures = {pool.submit(_export_group, t): t for t in tasks}
        for fut in as_completed(futures):
            members = futures[fut][0]
            try:
                entries = fut.result()
            except Exception as exc:
                entries = [{"chip_id": f"{cc}_{c:05d}_{r:05d}",
                            "error": f"{type(exc).__name__}: {exc}"} for c, r in members]
            done += len(entries)
            bad = [e for e in entries if "error" in e]
            failures += bad
            rate = done / max(time.time() - t1, 1e-9)
            eta_h = (n_chips - done) / rate / 3600 if rate else float("nan")
            if bad:
                print(f"[{done}/{n_chips}] FAILED {len(bad)} chips from {bad[0]['chip_id']}: "
                      f"{bad[0]['error']}", flush=True)
            else:
                e = entries[0]["imagery"]
                print(f"[{done}/{n_chips}] {len(entries)} chips from {entries[0]['chip_id']} "
                      f"{e.get('window_seconds', e.get('seconds', 0)):.0f}s  "
                      f"eta {eta_h:.1f} h", flush=True)
            if failures:
                (out / "failures.json").write_text(json.dumps(failures, indent=1))

    print(f"exported {done - len(failures):,} chips, {len(failures):,} failed, "
          f"in {(time.time() - t1) / 3600:.2f} h", flush=True)
    assemble(out, cc, args.year, config, args.stats_sample)


if __name__ == "__main__":
    main()
