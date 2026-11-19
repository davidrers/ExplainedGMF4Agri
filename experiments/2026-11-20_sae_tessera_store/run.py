"""Sparse autoencoders on the precomputed TESSERA embeddings of Estonia 2021: every stage of the study.

Stages, each resumable (a stage whose output exists is skipped unless ``--force``):

    poetry run python experiments/2026-11-20_sae_tessera_store/run.py extract     # pixel samples of every chip
    poetry run python experiments/2026-11-20_sae_tessera_store/run.py head        # the crop head, as plain torch
    poetry run python experiments/2026-11-20_sae_tessera_store/run.py train [--run NAME ...]
    poetry run python experiments/2026-11-20_sae_tessera_store/run.py evaluate

Outputs go to ``results/<name>/EE_2021_rev/``: ``samples/``, ``head/``, ``runs/<run>/``, ``eval/``.
Nothing in the main workflow is changed; the crop head is read from the fit named in ``sae.yaml``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import yaml

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

CFG = yaml.safe_load((HERE / "sae.yaml").read_text())
CHIPS = REPO / CFG["chips"]
OUT = REPO / "results" / CFG["name"] / CHIPS.name
SPLIT_FILES = {"train": "training_data.txt", "val": "validation_data.txt", "test": "test_data.txt"}
EMB_SUFFIX = json.loads((CHIPS / f"{CFG['embedding']}.json").read_text())["file_suffix"]  # "_tessera.tif"
RED, NIR, SWIR1 = 3, 7, 10  # band positions within a month of the Sentinel-2 composite
KIND = {"uniform": 0, "class": 1, "nonfield": 2}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------------------------- chips and split
def split_lists() -> dict[str, list[str]]:
    d = CHIPS / "splits" / CFG["split"]
    return {s: [l.strip() for l in (d / f).read_text().splitlines() if l.strip()] for s, f in SPLIT_FILES.items()}


def coverage() -> dict[str, float]:
    """The share of pixels without an embedding per chip, as the export recorded it."""
    side = json.loads((CHIPS / f"{CFG['embedding']}.json").read_text())
    return {c["chip_id"]: c["nodata_share"] for c in side["chips"]}


def class_names() -> list[str]:
    return [c["name"] for c in json.loads((CHIPS / "manifest.json").read_text())["classes"]]


def read_band_stack(path: Path) -> np.ndarray:
    import rasterio

    with rasterio.open(path) as ds:
        return ds.read()


def interior_mask(parcels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Labelled pixels whose eight neighbours lie in the same parcel: the field shrunk by one pixel.

    The store raster sits about 0.7 px east and 0.4 px south of the chip grid (protocol D25), so a pixel on a
    field's edge can carry its neighbour's embedding; one pixel in from the edge it cannot."""
    h, w = parcels.shape
    p = np.pad(parcels, 1, constant_values=-1)
    same = np.ones((h, w), bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            if dy or dx:
                same &= p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w] == parcels
    return same & (parcels > 0) & (mask >= 0)


def read_chip(cid: str, profiles: bool) -> dict:
    emb = read_band_stack(CHIPS / "chips" / f"{cid}{EMB_SUFFIX}")
    mask = read_band_stack(CHIPS / "chips" / f"{cid}.mask.tif")[0].astype(np.int16)
    parcels = read_band_stack(CHIPS / "chips" / f"{cid}.parcels.tif")[0]
    out = {"emb": emb, "mask": mask, "parcels": parcels, "valid": np.isfinite(emb).all(0),
           "interior": interior_mask(parcels, mask)}
    if profiles:
        s2 = read_band_stack(CHIPS / "chips" / f"{cid}_merged.tif").astype(np.float32)
        s1 = read_band_stack(CHIPS / "chips" / f"{cid}_s1rtc.tif")
        red, nir, swir = s2[RED::12], s2[NIR::12], s2[SWIR1::12]
        with np.errstate(invalid="ignore", divide="ignore"):
            out["ndvi"] = np.where(nir + red > 0, (nir - red) / (nir + red), np.nan)
            out["ndmi"] = np.where(nir + swir > 0, (nir - swir) / (nir + swir), np.nan)
        out["vv"], out["vh"] = s1[0::2], s1[1::2]
    return out


# ------------------------------------------------------------------------------------------------------ extract
def sample_chip(cid: str, num: int, split: str) -> dict | None:
    s = CFG["sample"]
    rng = np.random.default_rng([s["seed"], num])
    profiles = split == "test"
    c = read_chip(cid, profiles)
    valid, mask, interior = c["valid"].reshape(-1), c["mask"].reshape(-1), c["interior"].reshape(-1)
    if not valid.any():
        return None
    rows: list[tuple[np.ndarray, int, float]] = []  # (flat indices, kind, weight)

    def pick(idx: np.ndarray, n: int, kind: str) -> None:
        if len(idx) == 0:
            return
        take = idx if len(idx) <= n else rng.choice(idx, n, replace=False)
        rows.append((np.sort(take), KIND[kind], len(idx) / len(take)))

    vidx = np.flatnonzero(valid)
    if split == "train":
        pick(vidx, s["train_pixels_per_chip"], "uniform")
    else:
        pick(vidx, s["uniform_pixels_per_chip"], "uniform")
        inner = valid & interior
        for cls in np.unique(mask[inner]):
            pick(np.flatnonzero(inner & (mask == cls)), s["class_pixels_per_chip"], "class")
        pick(np.flatnonzero(valid & (mask < 0)), s["nonfield_pixels_per_chip"], "nonfield")
    idx = np.concatenate([r[0] for r in rows])
    n = len(idx)
    emb = c["emb"].reshape(c["emb"].shape[0], -1)
    out = {"emb": emb[:, idx].T.astype(np.float16), "label": mask[idx].astype(np.int8),
           "interior": interior[idx], "chip": np.full(n, num, np.int32), "pos": idx.astype(np.int32),
           "parcel": c["parcels"].reshape(-1)[idx].astype(np.int64),
           "kind": np.concatenate([np.full(len(r[0]), r[1], np.int8) for r in rows]),
           "weight": np.concatenate([np.full(len(r[0]), r[2], np.float32) for r in rows])}
    if profiles:
        for key in ("ndvi", "ndmi", "vv", "vh"):
            a = c[key].reshape(12, -1)
            out[key] = a[:, idx].T.astype(np.float16)
    return out


def stage_extract(force: bool) -> None:
    (OUT / "samples").mkdir(parents=True, exist_ok=True)
    lists, cover = split_lists(), coverage()
    meta = {"chips": {}, "skipped_no_embedding": {}}
    for split, chips in lists.items():
        target = OUT / "samples" / f"{split}.npz"
        if target.exists() and not force:
            log(f"extract {split}: {target} exists, skipped")
            continue
        number = {c: i for i, c in enumerate(chips)}
        todo = [c for c in chips if cover.get(c, 1.0) < 0.999]
        meta["skipped_no_embedding"][split] = sorted(set(chips) - set(todo))
        log(f"extract {split}: {len(todo)} of {len(chips)} chips have an embedding")
        parts, t0 = {}, time.time()
        with ThreadPoolExecutor(CFG["sample"]["workers"]) as ex:
            futs = {ex.submit(sample_chip, cid, number[cid], split): cid for cid in todo}
            for i, f in enumerate(as_completed(futs), 1):
                r = f.result()
                if r is not None:
                    parts[futs[f]] = r
                if i % 250 == 0 or i == len(todo):
                    log(f"  {split}: {i}/{len(todo)} chips, {time.time() - t0:.0f} s")
        order = [c for c in chips if c in parts]
        arrays = {k: np.concatenate([parts[c][k] for c in order]) for k in parts[order[0]]}
        np.savez(target, chip_ids=np.array(chips), **arrays)
        meta["chips"][split] = {"listed": len(chips), "sampled": len(order), "pixels": int(len(arrays["label"]))}
        log(f"extract {split}: {len(arrays['label'])} pixels -> {target}")
    old = OUT / "samples" / "meta.json"
    if old.exists():
        prev = json.loads(old.read_text())
        for key in meta:
            prev.setdefault(key, {}).update(meta[key])
        meta = prev
    old.write_text(json.dumps(meta, indent=1))


def load_sample(split: str) -> dict:
    z = np.load(OUT / "samples" / f"{split}.npz")
    return {k: z[k] for k in z.files}


# --------------------------------------------------------------------------------------------------------- head
def head_module():
    """The per-pixel MLP of the crop arm as linear layers on ``(n, 128)`` normalised embeddings."""
    from torch import nn

    return nn.Sequential(nn.Linear(128, 512), nn.ReLU(), nn.BatchNorm1d(512),
                         nn.Linear(512, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Linear(256, 20))


def load_head():
    import torch

    meta = json.loads((OUT / "head" / "head.json").read_text())
    head = head_module()
    head.load_state_dict(torch.load(OUT / "head" / "head.pt"))
    return head.eval(), np.array(meta["means"], np.float32), np.array(meta["stds"], np.float32), meta


def stage_head(force: bool) -> None:
    import hashlib

    import torch

    from gfm4agri.benchmark.evaluation import FitPredictor, fit_checkpoint

    d = OUT / "head"
    if (d / "head.pt").exists() and not force:
        log("head: exists, skipped")
        return
    d.mkdir(parents=True, exist_ok=True)
    fit = REPO / CFG["head_fit"]
    fp = FitPredictor(fit, device="cpu")
    mlp, conv = fp.model.decoder.mlp, fp.model.head.head[2]
    head = head_module()
    pairs = [(head[0], mlp[0]), (head[3], mlp[4]), (head[6], conv)]
    with torch.no_grad():
        for lin, cv in pairs:
            lin.weight.copy_(cv.weight[:, :, 0, 0])
            lin.bias.copy_(cv.bias)
        for bn, src in [(head[2], mlp[2]), (head[5], mlp[6])]:
            bn.load_state_dict(src.state_dict())
    head.eval()
    x = torch.randn(4096, 128)
    with torch.no_grad():
        diff = float((head(x) - fp.model(x[:, :, None, None]).output[:, :, 0, 0]).abs().max())
    log(f"head: max difference to the fit's own model on random inputs {diff:.2e}")
    if diff > 1e-4:
        raise RuntimeError("the exported head does not reproduce the fit's model")
    ckpt = fit_checkpoint(fit, fp.results)
    torch.save(head.state_dict(), d / "head.pt")
    meta = {"fit": CFG["head_fit"], "checkpoint": str(Path(ckpt).relative_to(REPO)),
            "checkpoint_sha256": hashlib.sha256(Path(ckpt).read_bytes()).hexdigest(),
            "class_names": fp.class_names, "means": fp.dm.norm_means, "stds": fp.dm.norm_stds,
            "test_macro_f1_logged": fp.results["test_metrics"]["test/F1_Score"],
            "max_abs_difference_random_inputs": diff,
            "input": "(nan_to_num(embedding, 0) - means) / stds, as the datamodule feeds it"}
    # Reproduce the stored test predictions on a few chips through this path.
    stored = np.load(fit / "predictions_test.npz")
    ids = list(stored["ids"])
    agree, total = 0, 0
    mean, std = np.array(meta["means"], np.float32), np.array(meta["stds"], np.float32)
    for cid in ids[:: max(1, len(ids) // 25)][:25]:
        e = read_band_stack(CHIPS / "chips" / f"{cid}{EMB_SUFFIX}").reshape(128, -1).T
        xin = (np.nan_to_num(e, nan=0.0) - mean) / std
        with torch.no_grad():
            pred = head(torch.from_numpy(xin)).argmax(1).numpy().reshape(224, 224)
        m = stored["mask"][ids.index(cid)] >= 0
        agree += int((pred[m] == stored["pred"][ids.index(cid)][m]).sum())
        total += int(m.sum())
    meta["stored_prediction_agreement"] = {"chips": 25, "labelled_pixels": total, "share": agree / total}
    log(f"head: agreement with the stored test predictions on 25 chips {agree / total:.5f}")
    (d / "head.json").write_text(json.dumps(meta, indent=1))


# ------------------------------------------------------------------------------------------------------- driver
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=["extract", "head", "train", "evaluate"])
    ap.add_argument("--run", nargs="*", help="train: these runs only (default: all of sae.yaml)")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    if a.stage == "extract":
        stage_extract(a.force)
    elif a.stage == "head":
        stage_head(a.force)
    elif a.stage == "train":
        from stages import stage_train

        stage_train(a.run, a.force)
    else:
        from stages import stage_evaluate

        stage_evaluate(a.force)


if __name__ == "__main__":
    main()
