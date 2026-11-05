"""The 10 m TESSERA rasters U-Net + PaPs is trained on: frozen, or from an encoder fine-tuned on crop type.

* **Frozen** (``encoder: frozen``): the core's raster cache of ``tessera_v2_medium``
  (:mod:`gfm4agri.pipeline.raster_cache`), reused when one matches the run, else computed into the scratch root,
  as the K-shot runner does for the main workflow's TESSERA arm.
* **Fine-tuned** (``encoder: finetuned``): the student of the ``kshot_finetune`` cell at the same budget, draw and
  seed (``results/2026-10-15_kshot_finetune/<chip set>/tessera_v2_medium_ft/P<k>_draw<d>_seed<s>/``), fine-tuned on
  crop type on the labelled pixels of the training chips under that budget, loaded from the checkpoint its
  ``results.json`` names. Every chip of the split is encoded with it once, frozen from then on, as the frozen cache
  is made (same reader, same bf16 autocast, same layout), into ``<scratch>/tessera_v2_medium_ft/<chip set>/<cell>/``;
  its own training-chip statistics normalise it. The record says which checkpoint, with its SHA-256.

Both return the cache directory, the raster suffix and the per-band means and standard deviations.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

BACKBONE = "tessera_v2_medium"
FT_EXPERIMENT = "2026-10-15_kshot_finetune"
FT_ARM = "tessera_v2_medium_ft"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def _sidecar(cache_dir: Path) -> dict:
    from gfm4agri.benchmark.backbones import get_backbone

    rep = get_backbone(BACKBONE).representation
    return json.loads((cache_dir / f"{rep}.json").read_text())


def frozen(chips: Path, split_dir: Path, cache_root: Path, scratch: Path, batch: int, dry_run: bool = False):
    """The core's frozen raster cache: ``(dir, suffix, means, stds, record, action)``."""
    from gfm4agri.pipeline.raster_cache import build_raster_cache, resolve_raster_cache

    d = resolve_raster_cache(cache_root, scratch, BACKBONE, chips, split_dir)
    if dry_run:
        return d.cache_dir, None, None, None, None, f"{d.action} ({d.reason})"
    if d.action == "stop":
        raise SystemExit(f"{d.cache_dir} does not match the run; pass --scratch")
    if d.action == "compute":
        print(f"frozen TESSERA raster: encoding into {d.cache_dir} ({d.reason})", flush=True)
        build_raster_cache(BACKBONE, chips, split_dir, d.cache_dir, batch=batch)
    side = _sidecar(d.cache_dir)
    record = json.loads((d.cache_dir / "cache.json").read_text())
    return (d.cache_dir, side["file_suffix"], side["normalisation"]["means"], side["normalisation"]["stds"],
            record, d.action)


def ft_cell(repo: Path, chips: Path, pct, draw: int, seed: int) -> Path:
    return repo / "results" / FT_EXPERIMENT / chips.name / FT_ARM / f"P{pct:g}_draw{draw}_seed{seed}"


def finetuned(repo: Path, chips: Path, split_dir: Path, scratch: Path, pct, draw: int, seed: int, batch: int,
              dry_run: bool = False):
    """Every chip of the split encoded with the fine-tuned student of the matching ``kshot_finetune`` cell."""
    import rasterio
    import torch

    from gfm4agri.benchmark.backbones import get_backbone
    from gfm4agri.benchmark.evaluation import fit_checkpoint
    from gfm4agri.benchmark.segmentation_data import load_manifest, manifest_sha256, read_split, split_list
    from gfm4agri.embeddings.export import band_stats
    from gfm4agri.embeddings.tessera_v2 import WEIGHTS, ChipEncoder
    from gfm4agri.pipeline.raster_cache import READ_AHEAD, _chip_ids, _write

    cell = ft_cell(repo, chips, pct, draw, seed)
    if not (cell / "results.json").exists():
        raise SystemExit(f"no fine-tuned TESSERA cell at {cell}")
    r = json.loads((cell / "results.json").read_text())
    ckpt = fit_checkpoint(cell, r)
    if not ckpt.exists():
        raise SystemExit(f"{cell}: checkpoint {ckpt.name} missing")
    spec = get_backbone(BACKBONE)
    suffix = f"_{spec.representation}.tif"
    cache_dir = Path(scratch) / f"{BACKBONE}_ft" / chips.name / cell.name
    want = {"backbone": BACKBONE, "encoder": "tessera_v2 student fine-tuned on crop type, then frozen",
            "source_cell": str(cell.relative_to(repo)), "checkpoint": ckpt.name,
            "precision": spec.computed["precision"], "manifest_sha256": manifest_sha256(chips),
            "split_config_hash": read_split(split_dir)["config_hash"],
            "source_label_budget": {k: r["label_budget"][k] for k in ("pct", "draw_seed")}, "source_seed": r["seed"]}
    if dry_run:
        return cache_dir, None, None, None, None, f"fine-tuned student of {cell.name} ({ckpt.name})"
    want["checkpoint_sha256"] = _sha256(ckpt)
    rec_path = cache_dir / "cache.json"
    if rec_path.exists():
        rec = json.loads(rec_path.read_text())
        bad = [k for k, v in want.items() if rec.get(k) != v]
        if bad:
            raise SystemExit(f"{cache_dir} was made under another record ({bad}); not written into")
        side = _sidecar(cache_dir)
        return (cache_dir, suffix, side["normalisation"]["means"], side["normalisation"]["stds"], rec, "reuse")

    man = load_manifest(chips)
    enc = ChipEncoder(spec.computed["student"], spec.computed["checkpoint_sha256"], WEIGHTS,
                      band_names=man["imagery"]["band_names"], n_months=man["imagery"]["n_months"],
                      year=man["year"], precision=spec.computed["precision"], batch=batch)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    student = {k[len("student."):]: v for k, v in state.items() if k.startswith("student.")}
    if not student:
        raise SystemExit(f"{ckpt}: no student weights; was the encoder trained?")
    enc.model.load_state_dict(student, strict=True)
    enc.model.eval()

    out = cache_dir / "chips"
    out.mkdir(parents=True, exist_ok=True)
    ids = _chip_ids(chips, split_dir)
    have = set(os.listdir(out))
    todo = [c for c in ids if c + suffix not in have]
    dims = int(spec.computed["dims"])
    band_names = [f"{spec.representation.upper()}_{k:03d}" for k in range(dims)]
    print(f"fine-tuned TESSERA raster of {cell.name}: {len(todo)} of {len(ids)} chips into {cache_dir}", flush=True)
    t0 = time.time()
    src = chips / "chips"
    with ThreadPoolExecutor(4) as pool:
        ahead = [pool.submit(enc.read, src, c) for c in todo[:READ_AHEAD]]
        for i, c in enumerate(todo):
            inputs = ahead[i].result()
            ahead[i] = None
            if i + READ_AHEAD < len(todo):
                ahead.append(pool.submit(enc.read, src, todo[i + READ_AHEAD]))
            emb = np.asarray(enc.encode(inputs), dtype=np.float32)
            with rasterio.open(src / f"{c}.mask.tif") as m:
                crs, transform = m.crs, m.transform
            tmp = out / f"{c}{suffix}.part"
            _write(tmp, emb, crs, transform, band_names)
            tmp.replace(out / f"{c}{suffix}")
            if (i + 1) % 500 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                print(f"  {i + 1} / {len(todo)} chips encoded, {el / 60:.1f} min", flush=True)
    del enc
    torch.cuda.empty_cache()
    train = split_list(chips, "train", split_dir).read_text().split()
    stats = band_stats([out / f"{c}{suffix}" for c in train], f"training chips of split {split_dir.name}")
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    side = {"representation": spec.representation, "created": created, "source": want, "file_suffix": suffix,
            "band_names": band_names, "dtype": "float32", "normalisation": stats}
    (cache_dir / f"{spec.representation}.json").write_text(json.dumps(side, indent=2))
    record = {**want, "created": created, "chips": len(ids), "encode_seconds": round(time.time() - t0, 1)}
    rec_path.write_text(json.dumps(record, indent=1))
    return cache_dir, suffix, stats["means"], stats["stds"], record, "compute"
