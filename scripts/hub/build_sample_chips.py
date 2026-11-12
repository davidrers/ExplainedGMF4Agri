"""Export the Sentinel-2 composites of the transfer samples: only the drawn chips, for any candidate country.

The spatial transfer experiment (``docs/research/transferability/transfer_plan.md``) trains one model per country and
per GAEZ zone on a balanced, spread sample of whole blocks (4 x 4 chips), drawn by
``docs/research/transferability/draw_samples.py``, which lists every chip to export in ``export_cells.csv``. This
script composites those chips with the same code and screening as ``build_country_chips.py``, in windows of 2 x 2 chips,
into one chip set per country, ``data/eurocrops_chips/<CC>_2021_transfer/``.

``--product s2`` (the default) writes the Sentinel-2 composite (``<chip>_merged.tif`` and a ``.report.json`` with the
imagery report); ``--product s1`` writes the Sentinel-1 RTC composite (``<chip>_s1rtc.tif``, report in ``s1_reports/``)
with ``build_s1_chips.py``'s own per-chip worker, so both are what the main workflow's chip sets hold. The masks wait for
the crop list of the transfer experiment, and are rasterised from the 2021 labels once it is fixed, without touching
the imagery. For Estonia and Latvia the chips already on disk (``EE_2021_rev`` and ``EE_2021``, ``LV_2021``) are
linked into the sample set, never copied or changed, and only the missing ones are exported.

A chip is skipped when its composite and report exist, so re-running resumes. A block that still fails after the
retries is recorded in ``failures.json`` of its chip set.

    poetry run python scripts/hub/build_sample_chips.py --dry-run
    poetry run python scripts/hub/build_sample_chips.py --country-samples --workers 12
    poetry run python scripts/hub/build_sample_chips.py --country-samples --product s1 --workers 8
    poetry run python scripts/hub/build_sample_chips.py --countries SI --limit 2      # a test
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_country_chips import write_tif  # noqa: E402
from build_s1_chips import SUFFIX as S1_SUFFIX  # noqa: E402
from build_s1_chips import _export as export_s1_chip  # noqa: E402
from gfm4agri.chips.s2_monthly import (  # noqa: E402
    BAND_NAMES,
    N_MONTHS,
    SCREENINGS,
    composite_window,
    screening_record,
    window_report,
)
from gfm4agri.data.chip_grid import CHIP_PX, PIXEL_M, Chip  # noqa: E402

CELLS = REPO / "docs" / "research" / "transferability" / "export_cells.csv"
# country -> chip sets already on disk: (Sentinel-2 composites, Sentinel-1 RTC)
ON_DISK = {"EE": ("EE_2021_rev", "EE_2021"), "LV": ("LV_2021", "LV_2021")}


def paths_of(out: Path, chip_id: str) -> dict[str, Path]:
    d = out / "chips"
    return {"img": d / f"{chip_id}_merged.tif", "report": d / f"{chip_id}.report.json"}


def is_done(out: Path, chip_id: str) -> bool:
    return all(p.exists() for p in paths_of(out, chip_id).values())


def _export_block(task) -> list[dict]:
    """Composite one block's window (the bounding box of its drawn chips), then cut and write each chip."""
    cc, members, out_s, year, cloud_cover_max, retries, screening = task
    out = Path(out_s)
    chips = [Chip(cc, c, r) for c, r in members]
    bounds = (min(ch.bounds[0] for ch in chips), min(ch.bounds[1] for ch in chips),
              max(ch.bounds[2] for ch in chips), max(ch.bounds[3] for ch in chips))
    t0 = time.time()
    last = None
    for attempt in range(retries + 1):
        try:
            arr, maps, report = composite_window(bounds, year, cloud_cover_max=cloud_cover_max,
                                                 screening=SCREENINGS[screening])
            break
        except Exception as exc:  # network, STAC, or a scene that fails to read
            last = exc
            if attempt < retries:
                time.sleep(60 * (attempt + 1))   # read errors come in bursts of minutes
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
        report_c.update(seconds=round(window_s / len(chips), 1), window_bounds_3035=list(bounds),
                        window_chips=len(chips), window_seconds=window_s)
        write_tif(p["img"], np.ascontiguousarray(arr[:, rows, cols]), chip, desc)
        entry = {"chip_id": chip.chip_id, "col": chip.col, "row": chip.row, "bounds_3035": list(chip.bounds),
                 "imagery": report_c}
        p["report"].write_text(json.dumps(entry, indent=1))
        entries.append(entry)
    return entries


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", type=Path, default=CELLS)
    ap.add_argument("--countries", nargs="+", default=None, help="only these countries")
    ap.add_argument("--country-samples", action="store_true",
                    help="only the chips a country sample needs (not those drawn for zone samples alone)")
    ap.add_argument("--product", choices=["s2", "s1"], default="s2")
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--cloud-cover-max", type=float, default=80.0)
    ap.add_argument("--screening", choices=sorted(SCREENINGS), default="revised")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--group-size", type=int, default=2,
                    help="chips per side of a compositing window inside a block (4 = the whole block); 2 x 2 windows "
                         "keep the archive's read failures down, as in build_country_chips.py")
    ap.add_argument("--limit", type=int, default=None, help="at most this many blocks (a test)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out-root", type=Path, default=REPO / "data" / "eurocrops_chips")
    args = ap.parse_args()

    cells = pd.read_csv(args.cells)
    if args.country_samples:
        cells = cells[cells.country_sample]
    if args.countries:
        cells = cells[cells.country.isin(args.countries)]
    if not args.dry_run:
        link_on_disk(cells, args)

    guard(cells, args)
    if args.product == "s1":
        if not args.dry_run:
            link_on_disk(cells, args)
        export_s1(cells, args)
        return

    tasks, total, have = [], 0, 0
    for (cc, bx, by), block in cells.groupby(["country", "bx", "by"], sort=True):
        out = args.out_root / f"{cc}_{args.year}_transfer"
        members = sorted((int(c), int(r)) for c, r in zip(block.col, block.row))
        total += len(members)
        todo = [m for m in members if not is_done(out, f"{cc}_{m[0]:05d}_{m[1]:05d}")]
        have += len(members) - len(todo)
        windows: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for c, r in todo:
            windows.setdefault((c // args.group_size, r // args.group_size), []).append((c, r))
        for _, window in sorted(windows.items()):
            tasks.append((cc, window, str(out), args.year, args.cloud_cover_max, args.retries, args.screening))
    if args.limit:
        tasks = tasks[: args.limit]
    n_chips = sum(len(t[1]) for t in tasks)
    per_country = pd.Series([t[0] for t in tasks for _ in t[1]]).value_counts().sort_index()
    print(f"{total:,} chips in {cells.groupby(['country', 'bx', 'by']).ngroups:,} blocks; {have:,} already on disk; "
          f"{n_chips:,} to export in {len(tasks):,} windows of up to {args.group_size} x {args.group_size} chips", flush=True)
    print("to export per country: " + ", ".join(f"{c} {n:,}" for c, n in per_country.items()), flush=True)
    if args.dry_run or not tasks:
        return

    for cc in sorted({t[0] for t in tasks}):
        out = args.out_root / f"{cc}_{args.year}_transfer"
        (out / "chips").mkdir(parents=True, exist_ok=True)
        (out / "export.json").write_text(json.dumps({
            "purpose": "spatial transfer samples (docs/research/transferability/transfer_plan.md); imagery only",
            "cells": str(args.cells.relative_to(REPO)), "year": args.year, "screening": screening_record(
                SCREENINGS[args.screening]) | {"name": args.screening},
            "cloud_cover_max": args.cloud_cover_max, "started": time.strftime("%Y-%m-%d %H:%M:%S")}, indent=1))

    done, failures, t1 = 0, {}, time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(_export_block, t): t for t in tasks}
        for fut in as_completed(futures):
            cc, members, out_s = futures[fut][:3]
            try:
                entries = fut.result()
            except Exception as exc:
                entries = [{"chip_id": f"{cc}_{c:05d}_{r:05d}", "error": f"{type(exc).__name__}: {exc}"}
                           for c, r in members]
            done += len(entries)
            bad = [e for e in entries if "error" in e]
            if bad:
                failures.setdefault(cc, []).extend(bad)
                Path(out_s, "failures.json").write_text(json.dumps(failures[cc], indent=1))
            rate = done / max(time.time() - t1, 1e-9)
            eta_h = (n_chips - done) / rate / 3600 if rate else float("nan")
            status = f"FAILED {len(bad)}: {bad[0]['error']}" if bad else \
                f"{entries[0]['imagery']['window_seconds']:.0f}s"
            print(f"[{done}/{n_chips}] {cc} window of {len(entries)} from {entries[0]['chip_id']} {status}  "
                  f"{rate * 3600:.0f} chips/h, eta {eta_h:.1f} h", flush=True)

    n_bad = sum(len(v) for v in failures.values())
    print(f"exported {done - n_bad:,} chips, {n_bad:,} failed, in {(time.time() - t1) / 3600:.2f} h", flush=True)


def guard(cells: pd.DataFrame, args) -> None:
    """Refuse to write into a chip set this script did not make, such as a main-workflow set with a manifest."""
    for cc in sorted(cells.country.unique()):
        out = args.out_root / f"{cc}_{args.year}_transfer"
        if (out / "manifest.json").exists() and not (out / "export.json").exists():
            raise SystemExit(f"{out} holds a manifest and was not made by this script; refusing to write into it")


def link_on_disk(cells: pd.DataFrame, args) -> None:
    """Link the Estonian and Latvian chips that exist into the sample sets (symbolic links, sources untouched)."""
    for cc, (s2_set, s1_set) in ON_DISK.items():
        d = cells[cells.country == cc]
        if d.empty:
            continue
        out = args.out_root / f"{cc}_{args.year}_transfer"
        s2_dir, s1_dir = args.out_root / s2_set / "chips", args.out_root / s1_set
        pairs = []
        for col, row in zip(d.col, d.row):
            cid = f"{cc}_{int(col):05d}_{int(row):05d}"
            pairs += [(s2_dir / f"{cid}_merged.tif", out / "chips" / f"{cid}_merged.tif"),
                      (s2_dir / f"{cid}.report.json", out / "chips" / f"{cid}.report.json"),
                      (s1_dir / "chips" / f"{cid}{S1_SUFFIX}", out / "chips" / f"{cid}{S1_SUFFIX}"),
                      (s1_dir / "s1_reports" / f"{cid}.json", out / "s1_reports" / f"{cid}.json")]
        n = 0
        for src, dst in pairs:
            if src.exists() and not dst.exists() and not dst.is_symlink():
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src.resolve())
                n += 1
        print(f"{cc}: {n:,} files linked from {s2_set} and {s1_set}", flush=True)


def export_s1(cells: pd.DataFrame, args) -> None:
    """Sentinel-1 RTC, chip by chip, with the worker of ``build_s1_chips.py``."""
    tasks = []
    for cc, col, row in zip(cells.country, cells.col, cells.row):
        out = args.out_root / f"{cc}_{args.year}_transfer"
        cid = f"{cc}_{int(col):05d}_{int(row):05d}"
        if not ((out / "chips" / f"{cid}{S1_SUFFIX}").exists() and (out / "s1_reports" / f"{cid}.json").exists()):
            tasks.append((cc, int(col), int(row), str(out / "chips"), str(out / "s1_reports"), args.year, args.retries))
    n_have = len(cells) - len(tasks)
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"{len(cells):,} chips; {n_have:,} already have Sentinel-1; {len(tasks):,} to export", flush=True)
    if args.dry_run or not tasks:
        return
    for cc in sorted({t[0] for t in tasks}):
        for d in ("chips", "s1_reports"):
            (args.out_root / f"{cc}_{args.year}_transfer" / d).mkdir(parents=True, exist_ok=True)
    done, failures, t1 = 0, [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(export_s1_chip, t): t for t in tasks}
        for fut in as_completed(futures):
            done += 1
            t = futures[fut]
            try:
                e = fut.result()
            except Exception as exc:
                e = {"chip_id": f"{t[0]}_{t[1]:05d}_{t[2]:05d}", "error": f"{type(exc).__name__}: {exc}"}
            rate = done / max(time.time() - t1, 1e-9)
            if "error" in e:
                failures.append(e)
                (args.out_root / "s1_sample_failures.json").write_text(json.dumps(failures, indent=1))
                print(f"[{done}/{len(tasks)}] FAILED {e['chip_id']}: {e['error']}", flush=True)
            elif done % 25 == 0:
                print(f"[{done}/{len(tasks)}] {e['chip_id']} {rate * 3600:.0f} chips/h, "
                      f"eta {(len(tasks) - done) / rate / 3600:.1f} h", flush=True)
    print(f"exported {done - len(failures):,}, failed {len(failures):,}, in {(time.time() - t1) / 3600:.2f} h", flush=True)


if __name__ == "__main__":
    main()
