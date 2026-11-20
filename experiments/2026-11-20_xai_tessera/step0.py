"""Step 0: the statistics the model was trained under, the mean field, and the check that the chain reproduces it.

**Why the statistics are recomputed.** The fit standardised TESSERA's embedding with per-dimension statistics of the
training chips, held in the raster cache's sidecar on the cluster node; the job deleted the cache, and neither
``results.json`` nor ``best.pt`` keeps them. They are recomputed as :func:`gfm4agri.pipeline.raster_cache.
build_raster_cache` computed them: the core's encoder (bf16, the cluster's batch of pixels) over every training chip
in the split's order, mean and standard deviation per dimension, NaN ignored, rounded to five decimals
(:func:`gfm4agri.embeddings.export.band_stats`).

**The mean field**, read from the same composites in the same pass: per class, the sum and count of every month and
feature over the labelled pixels of the training chips; a species' mean is its classes' pooled mean, and the
baseline is the mean of the 13 species' means, so that no species is the reference.

**The check.** Every test chip is encoded again, standardised with these statistics and predicted as the fit
predicted (bf16, :func:`gfm4agri.borders.fit.predict`), and the crop class of every labelled pixel compared with
``predictions_test.npz``; then the same in fp32, the precision the XAI methods use. Each parcel's majority class is
compared too.

Writes ``results/<name>/EE_2021_rev/step0/``: ``embedding_stats.json``, ``mean_field.npz``, ``verify.json``.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

import chain

READ_AHEAD = 8


def _say(msg: str) -> None:
    import datetime as dt

    print(f"[step0 {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def _ordered(pool, fn, items):
    """``fn(item)`` for every item, read ahead on ``pool``, yielded in order."""
    ahead = [pool.submit(fn, c) for c in items[:READ_AHEAD]]
    for i in range(len(items)):
        r = ahead[i].result()
        ahead[i] = None
        if i + READ_AHEAD < len(items):
            ahead.append(pool.submit(fn, items[i + READ_AHEAD]))
        yield r


def prep(chips: Path, split_dir: Path, out: Path, batch: int, limit: int | None = None) -> None:
    from gfm4agri.benchmark.segmentation_data import split_list
    from gfm4agri.embeddings import tessera_v2 as tv

    out.mkdir(parents=True, exist_ok=True)
    man = json.loads((chips / "manifest.json").read_text())
    names = [c["name"] for c in man["classes"]]
    train = split_list(chips, "train", split_dir).read_text().split()
    if limit:
        train = train[:limit]
    enc = tv.ChipEncoder("medium", chain.STUDENT_SHA, tv.WEIGHTS, band_names=man["imagery"]["band_names"],
                         n_months=12, year=chain.YEAR, precision="bf16-mixed", batch=batch)
    s = s2 = n = None
    k = len(names)
    c_sum, c_n = np.zeros((k, 12, 12)), np.zeros((k, 12, 12))
    t0 = time.time()
    with ThreadPoolExecutor(4) as pool:
        for i, chip in enumerate(_ordered(pool, lambda c: chain.read_chip(chips / "chips", c), train)):
            X = chip["X"]
            emb = np.asarray(enc.encode((X[..., chain.S2], X[..., chain.S1], chip["hw"])), np.float32)
            a = emb.astype(np.float64).reshape(emb.shape[0], -1)
            if s is None:
                s, s2, n = (np.zeros(len(a)) for _ in range(3))
            ok = np.isfinite(a)
            s += np.where(ok, a, 0).sum(1)
            s2 += np.where(ok, a * a, 0).sum(1)
            n += ok.sum(1)
            lab = ((chip["mask"] >= 0) & (chip["parcels"] > 0)).ravel()
            cls = chip["mask"].ravel()[lab]
            Xl = X[lab].astype(np.float64)
            fin = np.isfinite(Xl)
            Xl = np.where(fin, Xl, 0.0)
            for c in np.unique(cls):
                r = cls == c
                c_sum[c] += Xl[r].sum(0)
                c_n[c] += fin[r].sum(0)
            if (i + 1) % 200 == 0 or i + 1 == len(train):
                el = time.time() - t0
                _say(f"{i + 1} / {len(train)} training chips, {el / 60:.1f} min, {el / (i + 1):.2f} s per chip")
    mean = s / n
    std = np.sqrt(np.maximum(s2 / n - mean ** 2, 1e-12))
    stats = {"computed_on": f"training chips of split {split_dir.name}", "n_chips": len(train),
             "means": [round(float(x), 5) for x in mean], "stds": [round(float(x), 5) for x in std],
             "encoder": "tessera_v2 medium, bf16-mixed", "batch": batch}
    (out / "embedding_stats.json").write_text(json.dumps(stats, indent=1))
    species, sidx = chain.species_of_classes(names)
    sp_n = np.stack([c_n[sidx == j].sum(0) for j in range(len(species))])
    with np.errstate(invalid="ignore", divide="ignore"):
        sp_mean = np.stack([c_sum[sidx == j].sum(0) for j in range(len(species))]) / sp_n   # NaN: no pixel
    present = (sp_n > 0).all((1, 2))
    if not present.all():
        _say(f"species without pixels, left out of the baseline: {[s for s, p in zip(species, present) if not p]}")
    np.savez(out / "mean_field.npz", baseline=np.nanmean(sp_mean[present], 0), species_mean=sp_mean,
             species_present=present, species_pixels=sp_n[:, 0, 0], species=np.array(species),
             class_sum=c_sum, class_n=c_n, class_names=np.array(names), features=np.array(chain.FEATURES))
    _say(f"statistics and mean field of {len(train)} chips -> {out}")


def verify(chips: Path, split_dir: Path, out: Path, batch: int, limit: int | None = None,
           precisions=("bf16-mixed", "32")) -> dict:
    """Pixel and parcel agreement of the rebuilt chain with the stored test predictions, per precision."""
    import torch

    from gfm4agri.benchmark.segmentation_data import split_list

    model = chain.load_model(stats=out / "embedding_stats.json")
    stored = np.load(chain.CELL / "predictions_test.npz")
    pos = {c: i for i, c in enumerate(stored["ids"])}
    pred = stored["pred"]
    test = split_list(chips, "test", split_dir).read_text().split()
    test = [c for c in test if c in pos][: limit or None]
    agree = {p: [0, 0, 0, 0] for p in precisions}                 # pixels same, pixels, parcels same, parcels
    secs = {p: 0.0 for p in precisions}
    with ThreadPoolExecutor(4) as pool:
        for i, chip in enumerate(_ordered(pool, lambda c: chain.read_chip(chips / "chips", c), test)):
            lab = (chip["mask"] >= 0) & (chip["parcels"] > 0)
            ref = pred[pos[chip["chip"]]].astype(np.int64)
            for p in precisions:
                t = time.time()
                emb = chain.encode(model, chip["X"], p, batch)
                with torch.no_grad():
                    lp = chain.crop_logp(model, torch.from_numpy(emb.T.reshape(1, 128, *chip["hw"])).to(model.device),
                                         p)[0]
                got = lp.argmax(0).cpu().numpy()
                if model.device.type == "cuda":
                    torch.cuda.synchronize()
                secs[p] += time.time() - t
                a = agree[p]
                a[0] += int((got[lab] == ref[lab]).sum())
                a[1] += int(lab.sum())
                ids = chip["parcels"][lab]
                for fid in np.unique(ids):
                    sel = ids == fid
                    a[2] += int(np.bincount(got[lab][sel], minlength=20).argmax()
                                == np.bincount(ref[lab][sel], minlength=20).argmax())
                    a[3] += 1
            if (i + 1) % 100 == 0 or i + 1 == len(test):
                _say(f"{i + 1} / {len(test)} test chips: " + ", ".join(
                    f"{p} pixels {agree[p][0] / agree[p][1]:.5f} parcels {agree[p][2] / agree[p][3]:.5f}"
                    for p in precisions))
    res = {p: {"pixel_agreement": agree[p][0] / agree[p][1], "parcel_agreement": agree[p][2] / agree[p][3],
               "pixels": agree[p][1], "parcels": agree[p][3], "seconds_per_chip": secs[p] / len(test)}
           for p in precisions}
    res["chips"] = len(test)
    res["device"] = str(model.device)
    (out / "verify.json").write_text(json.dumps(res, indent=1))
    _say(json.dumps(res))
    return res
