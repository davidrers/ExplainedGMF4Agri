"""The cache of a computed embedding raster: a frozen per-pixel encoder, run once per chip set by the runner.

A 10 m backbone whose spec carries ``computed`` (TESSERA v2) is not a product in the chip set but a model the K-shot
runner runs, as it runs the token-grid encoders whose feature caches :mod:`gfm4agri.pipeline.cache` manages. Its output
for a chip never changes, so it is computed once per chip set into ``<root>/<backbone>/<chip set>/``:

* ``chips/<chip>_<representation>.tif``: float32 on the chip's grid, NaN where the encoder had no input;
* ``<representation>.json``: the sidecar the datamodule reads, in the layout of the precomputed embeddings' sidecars,
  with per-band statistics of the training chips;
* ``cache.json``: the record the cache was made under, written last; ``building.json`` holds it meanwhile.

Unlike a ViT cache it holds one copy per chip: a per-pixel encoder commutes with rotation and flipping, so the D4
augmentation of the cached raster during training is exact.

The decisions are those of the feature caches. A complete cache under the cache root whose record matches the run is
reused; otherwise the cache is computed into the scratch root, resuming chips already there; a cache made under
another record is never written into.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from gfm4agri.benchmark.backbones import BackboneSpec, get_backbone
from gfm4agri.benchmark.segmentation_data import load_manifest, manifest_sha256, read_split, split_list
from gfm4agri.pipeline.cache import CacheDecision, cache_dir_for

__all__ = ["ENCODERS", "build_raster_cache", "expected_record", "resolve_raster_cache"]

RECORD, BUILDING = "cache.json", "building.json"
#: Chips read ahead of the encoder, so reading overlaps encoding.
READ_AHEAD = 8


def _write(path: Path, arr: np.ndarray, crs, transform, band_names: list[str]) -> None:
    """An uncompressed float32 raster: a cache is written and read again by one job, so speed beats size."""
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype="float32", crs=crs, transform=transform,
                       nodata=np.nan) as dst:
        dst.write(arr)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def _tessera_v2(spec: BackboneSpec, manifest: dict, batch: int):
    from gfm4agri.embeddings.tessera_v2 import WEIGHTS, ChipEncoder

    c = spec.computed
    return ChipEncoder(c["student"], c["checkpoint_sha256"], WEIGHTS,
                       band_names=manifest["imagery"]["band_names"],
                       n_months=manifest["imagery"]["n_months"], year=manifest["year"],
                       precision=c["precision"], batch=batch)


#: ``computed["encoder"]`` -> ``(spec, chip manifest, batch) -> encoder``, the encoder offering
#: ``read(chip_dir, chip_id) -> inputs`` and ``encode(inputs) -> (dims, H, W)``.
ENCODERS: dict[str, Callable] = {"tessera_v2": _tessera_v2}


def expected_record(chips, split_dir, backbone: str) -> dict:
    """What a cache of ``backbone`` on ``chips`` must have been made under to be used for this run."""
    c = get_backbone(backbone).computed
    return {"backbone": backbone, "encoder": c["encoder"], "student": c.get("student"),
            "checkpoint_sha256": c["checkpoint_sha256"], "precision": c["precision"],
            "manifest_sha256": manifest_sha256(chips),
            "split_config_hash": read_split(split_dir)["config_hash"]}


def _mismatches(record: dict, expected: dict) -> list[str]:
    return [f"{k}: cache {record.get(k)!r}, run {v!r}" for k, v in expected.items() if record.get(k) != v]


def _chip_ids(chips, split_dir) -> list[str]:
    ids: set[str] = set()
    for s in ("train", "val", "test"):
        f = split_list(chips, s, split_dir)
        if f.exists():
            ids.update(f.read_text().split())
    return sorted(ids)


def _suffix(spec: BackboneSpec) -> str:
    return f"_{spec.representation}.tif"


def _missing(cache_dir: Path, chips, split_dir, spec: BackboneSpec) -> int:
    folder = cache_dir / "chips"
    have = set(os.listdir(folder)) if folder.is_dir() else set()
    return sum(c + _suffix(spec) not in have for c in _chip_ids(chips, split_dir))


def resolve_raster_cache(cache_root, scratch, backbone: str, chips, split_dir) -> CacheDecision:
    spec = get_backbone(backbone)
    found = cache_dir_for(cache_root, backbone, chips)
    target = cache_dir_for(scratch, backbone, chips)
    same = Path(cache_root).resolve() == Path(scratch).resolve()
    want = expected_record(chips, split_dir, backbone)
    for name in (RECORD, BUILDING):
        path = found / name
        if path.exists() and (bad := _mismatches(json.loads(path.read_text()), want)):
            if same:
                return CacheDecision("stop", found, "the cache was made under another record", bad)
            return CacheDecision("compute", target, f"{found} was made under another record", bad)
    if (found / RECORD).exists() and (found / f"{spec.representation}.json").exists():
        n = _missing(found, chips, split_dir, spec)
        if n == 0:
            return CacheDecision("reuse", found, "complete and matching")
        if same:
            return CacheDecision("compute", found, f"{n} chips missing, resumed in place")
        return CacheDecision("compute", target, f"{n} chips missing in {found}")
    if found.exists() and same:
        return CacheDecision("compute", found, "partial cache, resumed in place")
    return CacheDecision("compute", target, "no cache found")


def build_raster_cache(backbone: str, chips, split_dir, cache_dir, *, batch: int, encoder=None) -> dict:
    """Encode every chip of the split that ``cache_dir`` lacks, then write the sidecar and the record."""
    import rasterio

    from gfm4agri.embeddings.export import band_stats

    spec = get_backbone(backbone)
    chips, split_dir, cache_dir = Path(chips), Path(split_dir), Path(cache_dir)
    want = expected_record(chips, split_dir, backbone)
    for name in (RECORD, BUILDING):
        path = cache_dir / name
        if path.exists() and (bad := _mismatches(json.loads(path.read_text()), want)):
            raise SystemExit(f"{cache_dir} was made under another record, not written into:\n  "
                             + "\n  ".join(bad))
    out = cache_dir / "chips"
    out.mkdir(parents=True, exist_ok=True)
    (cache_dir / BUILDING).write_text(json.dumps(want, indent=1))

    manifest = load_manifest(chips)
    suffix, dims = _suffix(spec), int(spec.computed["dims"])
    band_names = [f"{spec.representation.upper()}_{k:03d}" for k in range(dims)]
    ids = _chip_ids(chips, split_dir)
    have = set(os.listdir(out))
    todo = [c for c in ids if c + suffix not in have]
    t0 = time.time()
    if todo:
        encoder = encoder or ENCODERS[spec.computed["encoder"]](spec, manifest, batch)
        src = chips / "chips"
        with ThreadPoolExecutor(4) as pool:
            ahead = [pool.submit(encoder.read, src, c) for c in todo[:READ_AHEAD]]
            for i, c in enumerate(todo):
                inputs = ahead[i].result()
                ahead[i] = None
                if i + READ_AHEAD < len(todo):
                    ahead.append(pool.submit(encoder.read, src, todo[i + READ_AHEAD]))
                emb = np.asarray(encoder.encode(inputs), dtype=np.float32)
                if emb.shape[0] != dims:
                    raise ValueError(f"{c}: the encoder gave {emb.shape[0]} dimensions, the registry {dims}")
                # On the grid of the chip's own mask, so embeddings and labels coincide pixel for pixel.
                with rasterio.open(src / f"{c}.mask.tif") as m:
                    crs, transform = m.crs, m.transform
                # Written under a temporary name and renamed, so a killed run never leaves a truncated
                # raster that a resumed one would take for a finished chip.
                tmp = out / f"{c}{suffix}.part"
                _write(tmp, emb, crs, transform, band_names)
                tmp.replace(out / f"{c}{suffix}")
                if (i + 1) % 100 == 0 or i + 1 == len(todo):
                    el = time.time() - t0
                    print(f"  {i + 1} / {len(todo)} chips encoded, {el / 60:.1f} min, "
                          f"{el / (i + 1):.2f} s per chip", flush=True)
        del encoder
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    encode_s = time.time() - t0

    nodata = {}
    for c in ids:
        with rasterio.open(out / f"{c}{suffix}") as d:
            nodata[c] = round(float(np.isnan(d.read(1)).mean()), 6)
    train = split_list(chips, "train", split_dir).read_text().split()
    stats = band_stats([out / f"{c}{suffix}" for c in train], f"training chips of split {split_dir.name}")
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    sidecar = {
        "representation": spec.representation, "created": created,
        "source": {"backbone": backbone, **spec.computed, "notes": spec.notes},
        "file_suffix": suffix, "band_names": band_names, "dtype": "float32", "grid": manifest.get("grid"),
        "resampling": "none: encoded on the chip's own grid from its composites",
        "normalisation": stats, "chip_manifest_sha256": want["manifest_sha256"],
        "chips": [{"chip_id": c, "nodata_share": nodata[c]} for c in ids],
    }
    (cache_dir / f"{spec.representation}.json").write_text(json.dumps(sidecar, indent=2))
    record = {**want, "created": created, "chips": len(ids), "encoded_now": len(todo),
              "encode_seconds": round(encode_s, 1)}
    (cache_dir / RECORD).write_text(json.dumps(record, indent=1))
    (cache_dir / BUILDING).unlink(missing_ok=True)
    return record
