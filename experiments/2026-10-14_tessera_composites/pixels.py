"""The labelled pixels of a chip set, as TESSERA reads them: a store built once and reused by every cell.

TESSERA encodes each pixel's own time series, so a fit needs only the labelled pixels, not whole chips. Per split
(``train``, ``val``, ``test``, from the split's chip lists) the store holds, one row per labelled pixel:

    s2.npy      int16 (n, 12, 10)   the 12 monthly S2 composites, DN, in TESSERA's band order
    s1.npy      float16 (n, 12, 2)  the 12 monthly S1 composites, VV and VH in dB, NaN where never observed
    y.npy       int16 (n,)          class index (the chip mask)
    parcel.npy  int64 (n,)          parcel id (0 where the pixel is labelled but belongs to no parcel raster id)
    chip.npy    int32 (n,)          index into ``store.json``'s chip list of the split
    pos.npy     int32 (n,)          flat position in the 224 x 224 chip

``store.json`` records the chip manifest's SHA-256, the split and the counts; a store whose record does not match
the chip set is never reused or overwritten. A frozen encoder's embeddings of the whole store are cached beside it
(:func:`embedding_cache`).
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio

from tessera_v2 import S2_IDX

SPLITS = ("train", "val", "test")
FIELDS = {"s2": (np.int16, (12, 10)), "s1": (np.float16, (12, 2)), "y": (np.int16, ()),
          "parcel": (np.int64, ()), "chip": (np.int32, ()), "pos": (np.int32, ())}


def _manifest_sha(chips: Path) -> str:
    return hashlib.sha256((chips / "manifest.json").read_bytes()).hexdigest()


def _lists(chips: Path, split_dir: Path) -> dict[str, list[str]]:
    from gfm4agri.benchmark.segmentation_data import split_list

    return {s: split_list(chips, s, split_dir).read_text().split() for s in SPLITS}


def expected_record(chips: Path, split_dir: Path) -> dict:
    return {"chip_set": chips.name, "manifest_sha256": _manifest_sha(chips), "split": split_dir.name}


def _read_chip(chips: Path, cid: str) -> dict[str, np.ndarray]:
    base = chips / "chips" / cid
    with rasterio.open(f"{base}.mask.tif") as s:
        m = s.read(1).ravel()
    pos = np.flatnonzero(m >= 0).astype(np.int32)
    out = {"y": m[pos].astype(np.int16), "pos": pos}
    if not len(pos):
        out.update(s2=np.zeros((0, 12, 10), np.int16), s1=np.zeros((0, 12, 2), np.float16),
                   parcel=np.zeros(0, np.int64))
        return out
    with rasterio.open(f"{base}.parcels.tif") as s:
        out["parcel"] = s.read(1).ravel()[pos].astype(np.int64)
    with rasterio.open(f"{base}_merged.tif") as s:
        s2 = s.read().reshape(12, 12, -1)[:, S2_IDX][:, :, pos]            # (T, 10, n)
    with rasterio.open(f"{base}_s1rtc.tif") as s:
        s1 = s.read().reshape(12, 2, -1)[:, :, pos]                        # (T, 2, n)
    out["s2"] = np.ascontiguousarray(s2.transpose(2, 0, 1)).astype(np.int16)
    out["s1"] = np.ascontiguousarray(s1.transpose(2, 0, 1)).astype(np.float16)
    return out


def build_store(chips: Path, split_dir: Path, store: Path, workers: int = 8) -> dict:
    """Build the store of ``chips`` under ``split_dir`` into ``store`` (written to a temporary folder first)."""
    tmp = store.with_name(store.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    lists = _lists(chips, split_dir)
    rec = {**expected_record(chips, split_dir), "created": dt.datetime.now(dt.timezone.utc).isoformat(),
           "chips": lists, "pixels": {}}
    for split in SPLITS:
        ids = lists[split]
        with ThreadPoolExecutor(workers) as ex:
            parts = list(ex.map(lambda c: _read_chip(chips, c), ids))
        n = sum(len(p["y"]) for p in parts)
        d = tmp / split
        d.mkdir()
        arrays = {k: np.lib.format.open_memmap(d / f"{k}.npy", mode="w+", dtype=dt_, shape=(n, *sh))
                  for k, (dt_, sh) in FIELDS.items()}
        i = 0
        for k, p in enumerate(parts):
            m = len(p["y"])
            for f in ("s2", "s1", "y", "parcel", "pos"):
                arrays[f][i:i + m] = p[f]
            arrays["chip"][i:i + m] = k
            i += m
        for a in arrays.values():
            a.flush()
        rec["pixels"][split] = int(n)
        print(f"  pixel store {split}: {len(ids)} chips, {n:,} labelled pixels", flush=True)
    (tmp / "store.json").write_text(json.dumps(rec, indent=1))
    tmp.replace(store)
    return rec


def open_store(chips: Path, split_dir: Path, store: Path, build: bool = True) -> dict:
    """The store's record, building the store if absent; refuses a store made from another chip set or split."""
    rec_path = store / "store.json"
    if not rec_path.exists():
        if not build:
            raise FileNotFoundError(rec_path)
        return build_store(chips, split_dir, store)
    rec = json.loads(rec_path.read_text())
    want = expected_record(chips, split_dir)
    bad = [k for k, v in want.items() if rec.get(k) != v]
    if bad:
        raise SystemExit(f"{store} was built for another chip set or split ({bad}); pass --scratch to build "
                         "elsewhere")
    return rec


def load_split(store: Path, split: str, fields=tuple(FIELDS), mmap: bool = False) -> dict[str, np.ndarray]:
    return {f: np.load(store / split / f"{f}.npy", mmap_mode="r" if mmap else None) for f in fields}


def embedding_cache(store: Path, model, size: str, sha256: str, precision: str, device, batch: int = 16384) -> Path:
    """The frozen ``model``'s embeddings of every pixel of the store, float16 (n, 128) per split, computed once.

    Keyed by student size, checkpoint SHA-256 and precision. Exact for a frozen encoder: nothing augments the input,
    so a pixel's embedding never changes between epochs or cells.
    """
    import torch

    from tessera_v2 import encode_batch

    key = {"student": size, "checkpoint_sha256": sha256, "precision": precision}
    d = store / f"emb_v2_{size}_{sha256[:8]}_{precision}"
    if (d / "cache.json").exists() and json.loads((d / "cache.json").read_text())["key"] == key:
        return d
    tmp = d.with_name(d.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    model.eval()
    for split in SPLITS:
        a = load_split(store, split, ("s2", "s1"), mmap=True)
        n = len(a["s2"])
        out = np.lib.format.open_memmap(tmp / f"{split}.npy", mode="w+", dtype=np.float16, shape=(n, 128))
        with torch.no_grad():
            for i in range(0, n, batch):
                out[i:i + batch] = encode_batch(model, a["s2"][i:i + batch], a["s1"][i:i + batch], device,
                                                precision).cpu().numpy().astype(np.float16)
        out.flush()
        print(f"  embedding cache {split}: {n:,} pixels", flush=True)
    (tmp / "cache.json").write_text(json.dumps({"key": key, "created": dt.datetime.now(dt.timezone.utc).isoformat()}))
    tmp.replace(d)
    return d
