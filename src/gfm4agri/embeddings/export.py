"""Write a precomputed embedding onto an existing chip set, on the same grid as its labels.

Shared by the TESSERA and AlphaEarth builders, ``scripts/data/build_<name>_chips.py``. For
every chip in ``<root>/manifest.json`` the export writes ``<chip><suffix>`` next to the chip's
image, mask and parcel raster: float32, EPSG:3035, the chip's exact transform, NaN where the
provider has no embedding. The masks, the parcel rasters and the split files are shared with
every other representation, so a label budget draws the same parcels whatever the model.

Provenance and the per-dimension normalisation statistics (training chips only) go to
``<root>/<name>.json``. ``manifest.json`` is left untouched, because results already trained
on the chips are stamped with its hash.

A pilot chip set carries its split in the manifest and in ``training_chips/`` and
``validation_chips/``. A full-country chip set keeps every chip in ``chips/`` and carries no
split of its own, so the training chips are those of a spatial split directory, passed as
``split_dir``. Without one the statistics are left out, and :func:`write_stats` adds them
afterwards without reading the provider again. A chip already on disk is skipped, so an
interrupted run resumes.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

__all__ = ["EmbeddingExport", "band_stats", "chip_folder", "export_chips", "train_chips",
           "write_stats", "write_tif"]

SPLIT_DIRS = {"train": "training", "val": "validation", "test": "test"}


@dataclass(frozen=True)
class EmbeddingExport:
    """What differs between two precomputed embeddings once they are on the chip grid."""

    #: Representation name: the sidecar is ``<name>.json``, the registry's ``representation``.
    name: str
    #: Raster suffix after the chip id, e.g. ``"_tessera.tif"``.
    suffix: str
    band_names: list[str]
    #: Where the embedding comes from, recorded in the sidecar as is.
    source: dict
    #: How a provider pixel became a chip pixel, recorded in the sidecar.
    resampling: str
    #: Attempts after the first, 60 s apart and growing, before a chip is left for the next run.
    retries: int = 3

    @property
    def tag(self) -> str:
        """``"tessera"`` for ``"_tessera.tif"``: names the per-chip report and failure list."""
        return self.suffix.removeprefix("_").removesuffix(".tif")


def write_tif(path: Path, arr: np.ndarray, bounds, crs: str, pixel_m: float,
              band_names: list[str]) -> None:
    import rasterio
    from affine import Affine

    xmin, _, _, ymax = bounds
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype="float32", crs=crs,
                       transform=Affine(pixel_m, 0, xmin, 0, -pixel_m, ymax), nodata=np.nan,
                       compress="deflate", predictor=3, tiled=True,
                       blockxsize=arr.shape[2], blockysize=arr.shape[1]) as dst:
        dst.write(arr)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def chip_folder(root: Path, chip: dict) -> Path:
    """``chips/`` for a full-country set, the split's own folder for a pilot set."""
    if "split" not in chip:
        return root / "chips"
    return root / f"{SPLIT_DIRS[chip['split']]}_chips"


def train_chips(root: Path, manifest: dict, split_dir: Path | None,
                suffix: str) -> tuple[list[Path], str] | None:
    """The training chips' embedding rasters, and where the list came from."""
    if split_dir is not None:
        ids = [s for s in (split_dir / "training_data.txt").read_text().split() if s]
        return ([root / "chips" / f"{cid}{suffix}" for cid in ids],
                f"training chips of split {split_dir.name}")
    chips = manifest["chips"]
    if chips and "split" in chips[0]:
        return ([chip_folder(root, c) / f"{c['chip_id']}{suffix}" for c in chips
                 if c["split"] == "train"], "training chips")
    return None


def band_stats(paths: list[Path], source: str) -> dict:
    """Per-dimension mean and standard deviation over the training chips, NaN ignored."""
    import rasterio

    s = s2 = n = None
    for path in paths:
        with rasterio.open(path) as src:
            a = src.read().astype(np.float64).reshape(src.count, -1)
        if s is None:
            s, s2, n = (np.zeros(len(a)) for _ in range(3))
        ok = np.isfinite(a)
        s += np.where(ok, a, 0).sum(1); s2 += np.where(ok, a * a, 0).sum(1); n += ok.sum(1)
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean**2, 1e-12))
    return {"computed_on": source, "n_chips": len(paths),
            "means": [round(float(x), 5) for x in mean],
            "stds": [round(float(x), 5) for x in std]}


def write_stats(spec: EmbeddingExport, root: Path, split_dir: Path | None) -> None:
    """Recompute the statistics of an existing sidecar from the rasters on disk."""
    manifest = json.loads((root / "manifest.json").read_text())
    sidecar_path = root / f"{spec.name}.json"
    sidecar = json.loads(sidecar_path.read_text())
    paths, source = train_chips(root, manifest, split_dir, spec.suffix)
    sidecar["normalisation"] = band_stats(paths, source)
    sidecar_path.write_text(json.dumps(sidecar, indent=2))
    print(f"statistics over {len(paths)} {source} -> {sidecar_path}")


def export_chips(spec: EmbeddingExport, root: Path,
                 make_reader: Callable[[], Callable[[dict], tuple[np.ndarray, dict]]], *,
                 workers: int, overwrite: bool = False, split_dir: Path | None = None) -> None:
    """Write every chip's embedding raster, then the sidecar.

    ``make_reader`` is called once per worker thread and returns ``chip -> (array, info)``,
    with ``array`` of shape ``(bands, chip_px, chip_px)`` and ``info`` a JSON-able record
    carrying at least ``nodata_share``.
    """
    manifest = json.loads((root / "manifest.json").read_text())
    crs, pixel_m = manifest["grid"]["crs"], manifest["grid"]["pixel_m"]
    local = threading.local()

    def reader() -> Callable[[dict], tuple[np.ndarray, dict]]:
        if not hasattr(local, "reader"):
            local.reader = make_reader()
        return local.reader

    def job(c: dict) -> dict:
        path = chip_folder(root, c) / f"{c['chip_id']}{spec.suffix}"
        record = {"chip_id": c["chip_id"], "split": c.get("split")}
        report = path.with_name(f"{c['chip_id']}.{spec.tag}.json")
        if path.exists() and not overwrite:
            prior = json.loads(report.read_text()) if report.exists() else {}
            return {**record, **prior, "reused": True}
        t = time.time()
        for attempt in range(spec.retries + 1):
            try:
                emb, info = reader()(c)
                break
            except Exception as exc:  # a store may answer bursts of 5xx; outlast the burst
                if attempt == spec.retries:
                    return {**record, "error": f"{type(exc).__name__}: {exc}"}
                time.sleep(60 * (attempt + 1))
        # Written under a temporary name and renamed, so an interrupted write never leaves a
        # truncated raster that a resumed run would take for a finished one.
        tmp = path.with_name(path.name + ".part")
        write_tif(tmp, emb, c["bounds_3035"], crs, pixel_m, spec.band_names)
        tmp.replace(path)
        info = {**info, "seconds": round(time.time() - t, 1)}
        report.write_text(json.dumps(info))
        return {**record, **info}

    t0 = time.time()
    records, failures = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(job, c) for c in manifest["chips"]]
        for i, fut in enumerate(as_completed(futures), start=1):
            r = fut.result()
            if "error" in r:
                failures.append(r)
                print(f"[{i}/{len(futures)}] {r['chip_id']} FAILED {r['error']}", flush=True)
                continue
            records.append(r)
            print(f"[{i}/{len(futures)}] {r['chip_id']} ({r['split'] or 'unsplit'}) "
                  + ("reused" if r.get("reused") else f"{r['seconds']} s, nodata {r['nodata_share']}"),
                  flush=True)
    failures_path = root / f"{spec.tag}_failures.json"
    if failures:
        # No sidecar is written over an incomplete set; a re-run picks the failed chips up.
        failures_path.write_text(json.dumps(failures, indent=1))
        sys.exit(f"{len(failures)} chips failed after {spec.retries} retries; re-run to resume")
    failures_path.unlink(missing_ok=True)

    found = train_chips(root, manifest, split_dir, spec.suffix)
    stats = (band_stats(*found) if found else
             {"computed_on": None, "note": "no split given; add the statistics with "
                                           "--stats-only --split-dir <split>"})
    sidecar = {
        "representation": spec.name,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source": spec.source,
        "file_suffix": spec.suffix, "band_names": spec.band_names, "dtype": "float32",
        "grid": manifest["grid"],
        "resampling": spec.resampling,
        "normalisation": stats,
        "chip_manifest_sha256": hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest(),
        "chips": sorted(({k: v for k, v in r.items() if k != "reused"} for r in records),
                        key=lambda r: r["chip_id"]),
    }
    (root / f"{spec.name}.json").write_text(json.dumps(sidecar, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {root / f'{spec.name}.json'}")
