"""PRUE's fields scored by the void rule of ``panoptic_x``, as the PaPs and boundary-head parcels were.

From the stored class probabilities of every test chip (``predictions_test.npz``: ``prob`` ``(N, 3, H, W)`` uint8,
probability x 255; ``mask``; ``ids``), each variant of :func:`prue.parcels` builds the fields over the whole chip,
with no truth. The variant ``<extent>/<variant>`` keeps of each field its pixels inside the extent X (``chip``: as
it is, the headline; ``worldcover``, ``worldcover_30_40_90``: an independent first stage; ``oracle``: the declared
in-scheme parcels, an upper bound that uses the test labels), and is matched to the true parcel pieces by
``panoptic_x.chip_counts``: IoU > 0.5, void pixels out, a field mostly on void dropped, every other unmatched field
an FP wherever it lies.

The task is the geometry, so the headline is **class-agnostic** PQ (with its RQ, the object F1 at IoU > 0.5, FTW's
object-level metric). A field's class, for the class-aware PQ, is the vote (protocol D28) of the crop-type map of
the same encoder when one is given (``cropmap``: a ``predictions_test.npz`` with ``ids`` and ``pred``), as PaPs'
parcels took theirs in ``../2026-11-05_paps_tessera/analyse.py``; without it the class-aware fields are left out.

Beside the fields, per pixel: the boundary pixel precision, recall and F1 inside the declared extent E against the
two-pixel target of the delineation experiment, for the argmax and for ``p(boundary) >= 0.5`` (the boundary head's
rule); and the three-class confusion on the pixels the target knows (``prue.target``), with each class's IoU, the
pixel IoU of FTW.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import panoptic_x as px
import prue

EXTENTS = ("chip", "worldcover", "worldcover_30_40_90", "oracle")


def chip_parts(prob, mask, parcels, worldcover, crop_pred, n_classes, extents=EXTENTS,
               variants=prue.VARIANTS) -> dict:
    E, true, true_cls, void = px.truth(mask, parcels)
    wc = worldcover if worldcover is not None else np.zeros(mask.shape, np.uint8)
    out: dict = {}
    cls = prob.argmax(0)
    # Pixels: three-class confusion where the target knows the class, and the boundary inside E.
    y = prue.target(mask, parcels, wc)
    k = y != prue.IGNORE
    conf = np.bincount(y[k].astype(np.int64) * 3 + cls[k], minlength=9).astype(float)
    out["_confusion"] = (conf, None, None)
    tgt = E & prue.boundary_map(parcels)
    for name, hit in (("argmax", cls == prue.BOUNDARY), ("p50", prob[prue.BOUNDARY] >= 128)):
        hit = hit & E
        out[f"_bpx_{name}"] = (np.array([(hit & tgt).sum(), (hit & ~tgt).sum(), (tgt & ~hit).sum()], float),
                               None, None)
    if not true.any():
        return out
    for v in variants:
        inst = prue.parcels(prob, v)
        seg_cls = (px.vote(inst, crop_pred, n_classes) if crop_pred is not None
                   else np.zeros(int(inst.max()) + 1, np.int64))
        for x in extents:
            X = px.extent_of(x, mask, parcels, worldcover)
            seg = np.where(X, inst, 0)
            out[f"{x}/{v}"] = px.chip_counts(true, true_cls, seg, seg_cls, void, n_classes, (E, wc))
    return out


def _chunk(args) -> dict:
    items, chips_dir, n_classes, wc_path, extents, variants = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    wc = np.load(wc_path) if wc_path else None
    acc: dict = {}
    for cid, prob, mask, crop in items:
        w = None if wc is None else wc[cid]
        px._add(acc, chip_parts(prob, mask, read_parcels(chips_dir, cid), w, crop, n_classes, extents, variants))
    return acc


def _pixels(total: dict) -> dict:
    out = {}
    if "_confusion" in total:
        cm = total["_confusion"][0].reshape(3, 3)
        tp = np.diag(cm)
        iou = tp / np.maximum(cm.sum(0) + cm.sum(1) - tp, 1)
        f1 = 2 * tp / np.maximum(cm.sum(0) + cm.sum(1), 1)
        out["classes"] = {c: {"iou": float(iou[i]), "f1": float(f1[i]), "pixels": int(cm[i].sum())}
                          for i, c in enumerate(prue.CLASSES)}
        out["field_iou"] = float(_field_iou(cm))
        out["confusion"] = cm.astype(int).tolist()
    for name in ("argmax", "p50"):
        if f"_bpx_{name}" in total:
            tp, fp, fn = total[f"_bpx_{name}"][0]
            out[f"boundary_{name}"] = {"f1": float(2 * tp / max(2 * tp + fp + fn, 1)),
                                       "precision": float(tp / max(tp + fp, 1)), "recall": float(tp / max(tp + fn, 1)),
                                       "inside": "oracle extent E"}
    return out


def _field_iou(cm: np.ndarray) -> float:
    """IoU of field (interior or boundary) against background, the two merged."""
    f = np.array([[cm[0, 0], cm[0, 1:].sum()], [cm[1:, 0].sum(), cm[1:, 1:].sum()]])
    return f[1, 1] / max(f[1, 1] + f[0, 1] + f[1, 0], 1)


def score(probs, masks, ids, chips_dir: Path, class_names: list[str], worldcover: Path | None,
          crop: dict | None = None, extents=EXTENTS, variants=prue.VARIANTS, workers: int = 8) -> dict:
    """``probs`` ``(N, 3, H, W)`` uint8, ``masks`` ``(N, H, W)``, ``ids``; ``crop``: chip -> crop-type map or None."""
    if worldcover is None:
        extents = tuple(e for e in extents if not e.startswith("worldcover"))
    n = len(class_names)
    items = [(str(c), probs[i], masks[i], None if crop is None else crop[str(c)]) for i, c in enumerate(ids)]
    wc = None if worldcover is None else str(worldcover)
    total: dict = {}
    if workers <= 1:
        total = _chunk((items, Path(chips_dir), n, wc, extents, variants))
    else:
        parts = np.array_split(np.arange(len(items)), max(1, min(workers * 4, len(items))))
        jobs = [([items[i] for i in p], Path(chips_dir), n, wc, extents, variants) for p in parts if len(p)]
        with ProcessPoolExecutor(workers) as ex:
            for acc in ex.map(_chunk, jobs):
                px._add(total, acc)
    out = px.summarise({k: v for k, v in total.items() if not k.startswith("_")}, class_names)
    if crop is None:
        for v in out["variants"].values():
            for key in ("pq", "sq", "rq", "per_class"):
                v.pop(key, None)
    out["pixels"] = _pixels(total)
    out["class_source"] = "crop-type map vote" if crop is not None else "none (class-agnostic only)"
    out["rule"] = {**px.RECORD, "prue": {"cc": "4-connected regions of the interior class, < 5 px dropped (PRUE)",
                                         "grow": "cc grown by watershed on p(boundary) into the predicted field",
                                         "field": "4-connected regions of interior or boundary, nothing split"}}
    return out


def load_cropmap(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    with np.load(path) as d:
        ids, pred = d["ids"], d["pred"]
    return {str(c): pred[i] for i, c in enumerate(ids)}


def score_file(predictions: Path, chips_dir: Path, class_names: list[str], worldcover: Path | None,
               cropmap: Path | None = None, workers: int = 8) -> dict:
    with np.load(predictions) as d:        # each array read once (see ../2026-11-05_paps_tessera/score_paps.py)
        ids, prob, mask = d["ids"], d["prob"], d["mask"]
    crop = load_cropmap(cropmap)
    if crop is not None and not all(str(c) in crop for c in ids):
        crop = None
    out = score(prob, mask, ids, chips_dir, class_names, worldcover, crop=crop, workers=workers)
    out["cropmap"] = str(cropmap) if crop is not None else None
    return out
