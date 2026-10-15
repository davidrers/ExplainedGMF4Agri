"""Frozen TESSERA v2 Medium embedding rasters of a chip set, encoded from its monthly composites into a linked copy.

The core's ``raster`` route reads ``<chip set>/chips/<chip><suffix>`` and the sidecar
``<chip set>/<representation>.json``. Nothing is written into the chip set itself: :func:`ensure` builds a copy of
it made of links, ``<root>/tessera_v2m_rasters/<chip set>/``, with every top-level entry and every chip file linked
to the original, and adds to it

* ``chips/<chip>_tessera_v2m.tif``: float32 (128, 224, 224) on the chip's grid, NaN where a pixel has neither a
  valid S2 nor a valid S1 month (the datamodule reads NaN as 0, as for the v1 store);
* ``tessera_v2m.json``: the sidecar, with per-band statistics of the training chips for the datamodule's
  normalisation, as the v1 store's sidecar holds;
* ``rasters.json``: the key the copy was made under.

The copy is named as the chip set, so the runner pointed at it writes its results where it would for the chip set.

**Encoding** is that of experiments/2026-10-14_tessera_composites (stage 2, frozen arm), per pixel and exact, since
the encoder sees nothing but the pixel's own series: the 12 monthly S2 composites (TESSERA's ten bands, in its
order) and the 12 monthly S1 composites (dB, NaN where unobserved) through ``tessera_v2.encode_batch`` under bf16
autocast, which prepares the input as v2's ``encode_pixels`` does (checked by that experiment's ``check.py``).

**Reuse.** A copy under the cache root whose ``rasters.json`` matches the chip set's manifest, split, checkpoint and
precision, with every chip present, is used as it is. Otherwise the copy is made under the scratch root, resuming any
chips already there (a raster is written under a temporary name and renamed). A copy that exists with another key
is never written into.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

REPRESENTATION = "tessera_v2m"
SUFFIX = f"_{REPRESENTATION}.tif"
SIDECAR = f"{REPRESENTATION}.json"
RECORD = "rasters.json"
N_DIMS = 128


def _say(msg: str) -> None:
    print(f"[v2m rasters {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def _lists(chips: Path, split_dir: Path) -> dict[str, list[str]]:
    from gfm4agri.benchmark.segmentation_data import split_list

    return {s: split_list(chips, s, split_dir).read_text().split() for s in ("train", "val", "test")}


def expected_key(chips: Path, split_dir: Path, encoder: dict) -> dict:
    from gfm4agri.benchmark.segmentation_data import manifest_sha256

    return {"chip_set": chips.name, "manifest_sha256": manifest_sha256(chips), "split": split_dir.name,
            "student": encoder["student"], "checkpoint_sha256": encoder["checkpoint_sha256"],
            "precision": encoder["precision"]}


def missing_inputs(chips: Path, ids: list[str]) -> list[str]:
    """What encoding and fitting would read that the chip set does not hold."""
    present = set(os.listdir(chips / "chips"))
    out = []
    for sfx in ("_merged.tif", "_s1rtc.tif", ".mask.tif", ".parcels.tif"):
        miss = [c for c in ids if c + sfx not in present]
        if miss:
            out.append(f"{len(miss)} chips lack {sfx}, e.g. {miss[:3]}")
    return out


def _complete(copy: Path, key: dict, ids: list[str]) -> bool:
    rec = copy / RECORD
    if not (rec.exists() and (copy / SIDECAR).exists()):
        return False
    if json.loads(rec.read_text()).get("key") != key:
        return False
    present = set(os.listdir(copy / "chips"))
    return all(c + SUFFIX in present for c in ids)


def locate(chips: Path, split_dir: Path, encoder: dict, cache_root: Path, scratch: Path) -> tuple[str, Path]:
    """``("reuse", copy)``, ``("compute", copy)`` or ``("stop", copy)``, as the core decides for a feature cache."""
    key = expected_key(chips, split_dir, encoder)
    ids = sorted({c for v in _lists(chips, split_dir).values() for c in v})
    found = cache_root / "tessera_v2m_rasters" / chips.name
    if _complete(found, key, ids):
        return "reuse", found
    target = scratch / "tessera_v2m_rasters" / chips.name
    rec = target / RECORD
    if rec.exists() and json.loads(rec.read_text()).get("key") != key:
        return "stop", target
    return "compute", target


def _link_copy(chips: Path, copy: Path) -> None:
    """Every top-level entry of ``chips`` and every file of ``chips/chips`` as an absolute link under ``copy``."""
    chips = chips.resolve()
    (copy / "chips").mkdir(parents=True, exist_ok=True)
    for entry in os.listdir(chips):
        if entry in ("chips", SIDECAR, RECORD):
            continue
        dst = copy / entry
        if not dst.is_symlink():
            dst.symlink_to(chips / entry)
    have = set(os.listdir(copy / "chips"))
    for f in os.listdir(chips / "chips"):
        if f not in have:
            (copy / "chips" / f).symlink_to(chips / "chips" / f)


def _read(chip_dir: Path, cid: str):
    from tessera_v2 import S2_IDX

    with rasterio.open(chip_dir / f"{cid}_merged.tif") as s:
        profile = s.profile
        s2 = s.read().reshape(12, 12, -1)[:, S2_IDX]                       # (T, 10, P), DN
    with rasterio.open(chip_dir / f"{cid}_s1rtc.tif") as s:
        s1 = s.read().astype(np.float32).reshape(12, 2, -1)                # (T, 2, P), dB, NaN unobserved
    return (np.ascontiguousarray(s2.transpose(2, 0, 1)).astype(np.float32),
            np.ascontiguousarray(s1.transpose(2, 0, 1)), profile)


def _encode(model, s2: np.ndarray, s1: np.ndarray, device, precision: str, batch: int) -> np.ndarray:
    import torch

    from tessera_v2 import encode_batch

    with torch.no_grad():
        emb = np.concatenate([encode_batch(model, s2[i:i + batch], s1[i:i + batch], device, precision)
                              .cpu().numpy() for i in range(0, len(s2), batch)])
    empty = ~(s2 != 0).any(-1).any(-1) & ~np.isfinite(s1).all(-1).any(-1)
    emb[empty] = np.nan
    return emb


def _write(path: Path, emb: np.ndarray, profile: dict) -> None:
    h, w = profile["height"], profile["width"]
    prof = {"driver": "GTiff", "height": h, "width": w, "count": N_DIMS, "dtype": "float32",
            "crs": profile["crs"], "transform": profile["transform"], "nodata": float("nan"),
            "interleave": "pixel"}
    tmp = path.with_name(path.name + ".tmp")
    with rasterio.open(tmp, "w", **prof) as d:
        d.write(emb.T.reshape(N_DIMS, h, w).astype(np.float32))
    tmp.replace(path)


def _sums(emb: np.ndarray) -> np.ndarray:
    """(3, 128): count, sum and sum of squares per band over the finite pixels."""
    ok = np.isfinite(emb).all(1)
    e = emb[ok].astype(np.float64)
    return np.stack([np.full(N_DIMS, ok.sum(), np.float64), e.sum(0), (e ** 2).sum(0)])


def ensure(chips: Path, split_dir: Path, encoder: dict, *, cache_root: Path, scratch: Path, weights: Path,
           batch: int = 16384) -> Path:
    """The linked copy of ``chips`` holding the v2 rasters of every chip of the split, built if need be."""
    import torch

    from tessera_v2 import load_student

    key = expected_key(chips, split_dir, encoder)
    lists = _lists(chips, split_dir)
    ids = sorted({c for v in lists.values() for c in v})
    action, copy = locate(chips, split_dir, encoder, cache_root, scratch)
    _say(f"{action} {copy}")
    if action == "reuse":
        return copy
    if action == "stop":
        raise SystemExit(f"{copy} was made under another key; pass --scratch to build elsewhere")
    problems = missing_inputs(chips, ids)
    if problems:
        raise SystemExit("chip set check failed:\n  " + "\n  ".join(problems))

    _link_copy(chips, copy)
    (copy / RECORD).write_text(json.dumps({"key": key, "state": "building"}, indent=1))
    chip_dir = copy / "chips"
    present = set(os.listdir(chip_dir))
    todo = [c for c in ids if c + SUFFIX not in present]
    train = set(lists["train"])
    sums = np.zeros((3, N_DIMS), np.float64)
    nodata: dict[str, float] = {}

    for c in [c for c in ids if c + SUFFIX in present]:   # a resumed copy: read back what is there
        with rasterio.open(chip_dir / f"{c}{SUFFIX}") as d:
            emb = d.read().reshape(N_DIMS, -1).T
        nodata[c] = float((~np.isfinite(emb).all(1)).mean())
        if c in train:
            sums += _sums(emb)

    device = torch.device("cuda")
    model = load_student(encoder["student"], weights, encoder["checkpoint_sha256"]).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    _say(f"encoding {len(todo)} of {len(ids)} chips with v2 {encoder['student']}, {encoder['precision']}")
    t0 = time.time()
    with ThreadPoolExecutor(4) as ex:
        ahead = 8
        futures = [ex.submit(_read, chip_dir, c) for c in todo[:ahead]]
        for i, c in enumerate(todo):
            s2, s1, profile = futures[i].result()
            futures[i] = None
            if i + ahead < len(todo):
                futures.append(ex.submit(_read, chip_dir, todo[i + ahead]))
            emb = _encode(model, s2, s1, device, encoder["precision"], batch)
            _write(chip_dir / f"{c}{SUFFIX}", emb, profile)
            nodata[c] = float((~np.isfinite(emb).all(1)).mean())
            if c in train:
                sums += _sums(emb)
            if (i + 1) % 100 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                _say(f"  {i + 1} / {len(todo)} chips, {el / 60:.1f} min, {el / (i + 1):.2f} s per chip")
    del model
    torch.cuda.empty_cache()

    n, s, ss = sums
    means = s / n
    stds = np.sqrt(np.maximum(ss / n - means ** 2, 0.0))
    sidecar = {
        "representation": REPRESENTATION,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source": {"model": f"TESSERA v2 pixel student, {encoder['student']}, frozen",
                   "checkpoint": f"geotessera/TESSERA-V-2.0-2B-M ckpt/student_{encoder['student']}.pt",
                   "checkpoint_sha256": encoder["checkpoint_sha256"], "precision": encoder["precision"],
                   "input": "the chip set's 12 monthly S2 composites (B04 B02 B03 B08 B8A B05 B06 B07 B11 B12) and "
                            "12 monthly S1 composites (VV, VH dB, as TESSERA's ascending stream), day of year the "
                            "15th of each month; experiments/2026-10-14_tessera_composites"},
        "file_suffix": SUFFIX,
        "band_names": [f"TESSERA_V2M_{i:03d}" for i in range(N_DIMS)],
        "dtype": "float32",
        "grid": "the chip's own grid (that of _merged.tif), no resampling",
        "normalisation": {"computed_on": f"training chips of split {split_dir.name}", "n_chips": len(train),
                          "means": [round(float(v), 6) for v in means], "stds": [round(float(v), 6) for v in stds]},
        "chip_manifest_sha256": key["manifest_sha256"],
        "chips": [{"chip_id": c, "nodata_share": round(nodata[c], 6)} for c in ids],
    }
    (copy / SIDECAR).write_text(json.dumps(sidecar, indent=1))
    (copy / RECORD).write_text(json.dumps({"key": key, "state": "complete", "chips": len(ids),
                                           "encoded_now": len(todo), "encode_seconds": round(time.time() - t0),
                                           "created": sidecar["created"]}, indent=1))
    _say(f"{len(todo)} chips encoded in {(time.time() - t0) / 60:.1f} min; sidecar {copy / SIDECAR}")
    return copy
