"""Build ``EE_2021_sample_24``: the chips of ``EE_2021_sample`` with 24 half-month Sentinel-2 composites.

Each chip gets ``<chip>_merged.tif`` with 24 x 12 = 288 bands, period-major (band index = period * 12 + band),
composited by ``s2_periods.composite_window`` under the ``REVISED`` screening, and ``<chip>.report.json``. Every
other per-chip file (labels, Sentinel-1, TESSERA, AlphaEarth) and the chip-set records are links into
``EE_2021``, and the split is the sample's. Chips are composited in groups of G x G adjacent chips in one read,
in a process pool; a chip whose composite and report are on disk is skipped, so the build resumes.

    poetry run python experiments/2026-10-12_s2_semimonthly/build.py --workers 4 --group-size 2
    poetry run python experiments/2026-10-12_s2_semimonthly/build.py --assemble-only
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import datetime as dt
import importlib.util
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

from s2_periods import (DESCRIPTIONS, N_PERIODS, PERIOD_NAMES, composite_window,  # noqa: E402
                        period_centres, window_report)

from gfm4agri.chips.s2_monthly import BAND_NAMES, BANDS, REVISED  # noqa: E402
from gfm4agri.data.chip_grid import CHIP_PX, PIXEL_M, Chip  # noqa: E402

PARENT = REPO / "data" / "eurocrops_chips" / "EE_2021"
SAMPLE = REPO / "data" / "eurocrops_chips" / "EE_2021_sample"
OUT = REPO / "data" / "eurocrops_chips" / "EE_2021_sample_24"
LINKED_CHIP_SUFFIXES = [".mask.tif", ".parcels.tif", "_s1rtc.tif", "_tessera.tif", ".tessera.json",
                        "_alphaearth.tif", ".alphaearth.json"]
LINKED_ROOT = ["chip_parcels.parquet", "chip_parcels.parquet.sha256", "s1_rtc.json", "s1_reports",
               "tessera_v1.json", "alphaearth_v1.json"]


def _load_export_script():
    # write_tif of the core export, so the 24-period rasters are written exactly like the 12-month ones.
    spec = importlib.util.spec_from_file_location("build_country_chips",
                                                  REPO / "scripts" / "hub" / "build_country_chips.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_country_chips"] = mod
    spec.loader.exec_module(mod)
    return mod


EXPORT = _load_export_script()


def _link(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        return
    os.symlink(os.path.relpath(src, dst.parent), dst)


def sample_ids() -> list[str]:
    return sorted(p.name[: -len("_merged.tif")] for p in (SAMPLE / "chips").glob("*_merged.tif"))


def link_unchanged(ids: list[str]) -> None:
    (OUT / "chips").mkdir(parents=True, exist_ok=True)
    for name in LINKED_ROOT:
        _link(PARENT / name, OUT / name)
    _link(SAMPLE / "splits", OUT / "splits")
    _link(SAMPLE / "subset.json", OUT / "subset.json")
    for cid in ids:
        for suffix in LINKED_CHIP_SUFFIXES:
            _link(PARENT / "chips" / f"{cid}{suffix}", OUT / "chips" / f"{cid}{suffix}")


def done(cid: str) -> bool:
    return (OUT / "chips" / f"{cid}_merged.tif").exists() and (OUT / "chips" / f"{cid}.report.json").exists()


def export_group(task: tuple[list[str], int]) -> list[dict]:
    members, retries = task
    chips = [Chip("EE", int(c.split("_")[1]), int(c.split("_")[2])) for c in members]
    bounds = (min(ch.bounds[0] for ch in chips), min(ch.bounds[1] for ch in chips),
              max(ch.bounds[2] for ch in chips), max(ch.bounds[3] for ch in chips))
    t0 = time.time()
    last = None
    for attempt in range(retries + 1):
        try:
            arr, maps, report = composite_window(bounds, 2021, screening=REVISED)
            break
        except Exception as exc:
            last = exc
            if attempt < retries:
                time.sleep(60 * (attempt + 1))
    else:
        return [{"chip_id": ch.chip_id, "error": f"{type(last).__name__}: {last}"} for ch in chips]
    window_s = round(time.time() - t0, 1)
    out = []
    for chip in chips:
        r0 = int(round((bounds[3] - chip.bounds[3]) / PIXEL_M))
        c0 = int(round((chip.bounds[0] - bounds[0]) / PIXEL_M))
        rows, cols = slice(r0, r0 + CHIP_PX), slice(c0, c0 + CHIP_PX)
        rep = window_report(maps, report, rows, cols)
        rep |= {"seconds": round(window_s / len(chips), 1), "window_bounds_3035": list(bounds),
                "window_chips": len(chips), "window_seconds": window_s}
        EXPORT.write_tif(OUT / "chips" / f"{chip.chip_id}_merged.tif",
                         np.ascontiguousarray(arr[:, rows, cols]), chip, DESCRIPTIONS)
        (OUT / "chips" / f"{chip.chip_id}.report.json").write_text(
            json.dumps({"chip_id": chip.chip_id, "imagery": rep}, indent=1))
        out.append({"chip_id": chip.chip_id, "imagery": rep})
    return out


def band_stats(ids: list[str]) -> dict:
    import rasterio

    s, s2, n = np.zeros(len(BANDS)), np.zeros(len(BANDS)), 0
    for cid in ids:
        with rasterio.open(OUT / "chips" / f"{cid}_merged.tif") as src:
            a = src.read().astype(np.float64).reshape(N_PERIODS, len(BANDS), -1)
        s += a.sum((0, 2))
        s2 += (a ** 2).sum((0, 2))
        n += a.shape[0] * a.shape[2]
    mean = s / n
    return {"computed_on": f"{len(ids)} chips of the set, all periods pooled", "band_names": BAND_NAMES,
            "means": [round(x, 2) for x in mean], "stds": [round(x, 2) for x in np.sqrt(s2 / n - mean ** 2)]}


def assemble(ids: list[str]) -> None:
    parent = json.loads((PARENT / "manifest.json").read_text())
    entries = {e["chip_id"]: e for e in parent["chips"]}
    chips = []
    for cid in ids:
        f = OUT / "chips" / f"{cid}.report.json"
        if f.exists():
            chips.append({**entries[cid], "imagery": json.loads(f.read_text())["imagery"]})
    m = {k: v for k, v in parent.items() if k not in ("chips", "normalisation", "pixels_per_class_total",
                                                      "config", "relabelled")}
    m |= {"dataset": "eurocrops_s2_seg_EE_2021_sample_24",
          "purpose": "the chips of EE_2021_sample with 24 half-month Sentinel-2 composites, for the comparison "
                     "of 12 and 24 frames; labels and every other product linked from EE_2021",
          "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
          "derived_from": {"chip_set": "data/eurocrops_chips/EE_2021", "split": "data/eurocrops_chips/EE_2021_sample",
                           "experiment": "experiments/2026-10-12_s2_semimonthly"}}
    m["imagery"] = {**parent["imagery"], "n_months": None, "n_periods": N_PERIODS, "periods": PERIOD_NAMES,
                    "period_rule": "days 1 to 15 and day 16 to the month's end, by acquisition datetime (UTC)",
                    "period_centres_doy": [d for _, d in period_centres(2021)],
                    "layout": "(time channels): band index = period_index * 12 + band_index",
                    "screening": asdict(REVISED),
                    "composite": "per-pixel half-month median of clear observations under REVISED, one "
                                 "observation per overpass; gaps filled by linear interpolation across periods"}
    m["chips"] = chips
    if chips:
        rng = np.random.default_rng(0)
        sample = sorted(rng.choice([c["chip_id"] for c in chips], size=min(200, len(chips)), replace=False))
        m["normalisation"] = band_stats(sample)
    (OUT / "manifest.json").write_text(json.dumps(m, indent=2))
    print(f"manifest: {len(chips)} chips -> {OUT / 'manifest.json'}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--group-size", type=int, default=2)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--limit", type=int, default=None, help="export at most this many groups")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    ids = sample_ids()
    link_unchanged(ids)
    if args.assemble_only:
        assemble(ids)
        return
    g = max(args.group_size, 1)
    groups: dict[tuple[int, int], list[str]] = {}
    for cid in ids:
        if not done(cid):
            c, r = int(cid.split("_")[1]), int(cid.split("_")[2])
            groups.setdefault((c // g, r // g), []).append(cid)
    tasks = [(sorted(m), args.retries) for _, m in sorted(groups.items())][: args.limit]
    n = sum(len(m) for m, _ in tasks)
    print(f"{len(ids)} chips, {len(ids) - sum(len(m) for m in groups.values())} on disk, {n} to build in "
          f"{len(tasks)} groups of up to {g} x {g}", flush=True)
    t1, k, failures = time.time(), 0, []
    with ProcessPoolExecutor(args.workers) as pool:
        futs = {pool.submit(export_group, t): t[0] for t in tasks}
        for fut in as_completed(futs):
            try:
                res = fut.result()
            except Exception as exc:
                res = [{"chip_id": c, "error": f"{type(exc).__name__}: {exc}"} for c in futs[fut]]
            k += len(res)
            bad = [r for r in res if "error" in r]
            failures += bad
            rate = k / max(time.time() - t1, 1e-9)
            msg = (f"FAILED {len(bad)} from {bad[0]['chip_id']}: {bad[0]['error']}" if bad else
                   f"{len(res)} chips from {res[0]['chip_id']} {res[0]['imagery']['window_seconds']:.0f}s")
            print(f"[{k}/{n}] {msg}  eta {(n - k) / rate / 3600:.1f} h", flush=True)
            if failures:
                (OUT / "failures.json").write_text(json.dumps(failures, indent=1))
    print(f"built {k - len(failures)} chips, {len(failures)} failed, in {(time.time() - t1) / 3600:.2f} h",
          flush=True)
    assemble(ids)


if __name__ == "__main__":
    main()
