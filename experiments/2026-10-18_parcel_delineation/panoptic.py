"""Parcels from a fit's test predictions, and their Panoptic Quality against the true parcels.

Everything is NumPy and SciPy on a cell's ``predictions_test.npz``, so a finished fit is scored without a refit.

**Where.** Every test chip is scored inside its **extent**: the labelled pixels of the mask, those of the parcels
of an in-scheme class. Non-parcel land and parcels of a class outside the scheme are left out, as they are from the
pixel metrics. This is the second stage of the thesis' two-stage design inside a perfect first-stage mask; the
first stage, the cropland mask, is not built.

**The true parcels.** Each in-scheme parcel's 4-connected pieces inside the extent of the chip, so a parcel cut by
the chip edge or rasterised in several pieces counts once per piece, as PASTIS scores its patches.

**The predicted parcels**, five ways, each piece then taking the class most of its pixels were predicted as, ties to
the lowest class index (the vote of protocol D28):

* ``oracle``: the true pieces. Only the class is predicted, so its recognition quality is the parcel Macro-F1 with
  the true geometry, on the same units as every other variant.
* ``extent``: the 4-connected regions of the extent, nothing predicted about the geometry. The floor: a parcel
  isolated by non-parcel land is found for free, since the extent is the true one.
* ``semantic``: the 4-connected regions of one predicted class inside the extent. What a crop-type map alone gives.
* ``boundary``: the boundary head alone. The extent's pixels predicted interior (probability below ``tau``) form
  seeds, one per 4-connected region, and a watershed on the boundary probability grows them over the rest of the
  extent; an extent region holding no seed becomes one parcel of its own.
* ``combined``: as ``boundary``, but a seed is also cut where the predicted class changes.

**The score**, after Kirillov et al. (2019) as utae-paps computes it for PASTIS: a predicted and a true parcel
match when their IoU exceeds 0.5, which makes every match unique; per class, ``RQ = TP / (TP + FP/2 + FN/2)``, the
F1 of the parcels, ``SQ`` the mean IoU of the matches, ``PQ = SQ x RQ``; each a mean over the classes that occur.
Class-aware, a match must also agree on the class; class-agnostic (``*_agnostic``), any match counts, which scores
the geometry alone.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

#: The threshold on the boundary probability below which a pixel seeds a parcel; the same for every arm.
TAU = 0.5
#: Thresholds of the sensitivity check.
TAUS = (0.3, 0.5, 0.7)
#: A predicted and a true parcel match above this IoU.
IOU_MATCH = 0.5
FOUR = ndimage.generate_binary_structure(2, 1)

RECORD = {
    "extent": "labelled pixels of the test chip (in-scheme parcels); the true parcel extent",
    "true_parcels": "4-connected pieces of each parcel inside the extent of each test chip",
    "variants": {
        "oracle": "the true pieces, classed by the vote",
        "extent": "4-connected regions of the extent, no geometry predicted: the floor",
        "semantic": "4-connected regions of one predicted class inside the extent",
        "boundary": "seeds where the boundary probability < tau, grown by watershed on it inside the extent; "
                    "a seedless extent region is one parcel",
        "combined": "as boundary, seeds also cut where the predicted class changes",
    },
    "class_of_a_parcel": "majority vote of its predicted pixels, ties to the lowest class index",
    "match": f"IoU > {IOU_MATCH}", "tau": TAU, "taus": list(TAUS),
}


# ------------------------------------------------------------------------------------------------- segmentation

def pieces(labels: np.ndarray) -> np.ndarray:
    """``labels`` (0 = none) with every label's 4-connected pieces numbered apart, from 1, 0 kept."""
    labels = np.asarray(labels)
    out = np.zeros(labels.shape, dtype=np.int32)
    vals, inv = np.unique(labels, return_inverse=True)
    dense = inv.reshape(labels.shape).astype(np.int32) + (0 if vals[0] == 0 else 1)
    nxt = 1
    for k, sl in enumerate(ndimage.find_objects(dense), start=1):
        if sl is None:
            continue
        sub = dense[sl] == k
        lab, n = ndimage.label(sub, structure=FOUR)
        if n:
            view = out[sl]
            view[sub] = lab[sub] + (nxt - 1)
            nxt += n
    return out


def vote(segments: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Class of every segment id (index 0 unused): the majority of its predicted pixels, ties to the lowest."""
    n = int(segments.max()) + 1
    sel = segments > 0
    votes = np.bincount(segments[sel].astype(np.int64) * n_classes + pred[sel].astype(np.int64),
                        minlength=n * n_classes).reshape(n, n_classes)
    return votes.argmax(1)


def from_boundary(extent: np.ndarray, prob: np.ndarray, tau: float, pred: np.ndarray | None = None) -> np.ndarray:
    """Parcels from the boundary probability: interior seeds grown by watershed inside the extent.

    With ``pred``, a seed is also cut where the predicted class changes.
    """
    from skimage.segmentation import watershed

    core = extent & (prob < tau)
    seeds = pieces(np.where(core, pred.astype(np.int32) + 1, 0) if pred is not None else core.astype(np.int32))
    grown = watershed(prob, markers=seeds, mask=extent, connectivity=1) if seeds.any() else np.zeros_like(seeds)
    rest = extent & (grown == 0)
    if rest.any():
        extra = pieces(rest.astype(np.int32))
        grown = np.where(rest, extra + grown.max(), grown)
    return grown.astype(np.int32)


# ------------------------------------------------------------------------------------------------------ scoring

def match(true: np.ndarray, seg: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pairs ``(true id, segment id, IoU)`` with IoU above ``IOU_MATCH``; both maps labelled on the same extent."""
    sel = true > 0
    t, s = true[sel].astype(np.int64), seg[sel].astype(np.int64)
    if (s == 0).any():
        raise ValueError("a pixel of the extent belongs to no predicted parcel")
    base = int(s.max()) + 1
    pairs, inter = np.unique(t * base + s, return_counts=True)
    ti, si = pairs // base, pairs % base
    at, as_ = np.bincount(t), np.bincount(s)
    iou = inter / (at[ti] + as_[si] - inter)
    keep = iou > IOU_MATCH
    return ti[keep], si[keep], iou[keep]


def chip_counts(true: np.ndarray, true_cls: np.ndarray, seg: np.ndarray, seg_cls: np.ndarray,
                n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """Per class ``[TP, FP, FN, sum IoU]`` (class-aware) and the same four class-agnostic, for one chip."""
    ti, si, iou = match(true, seg)
    t_ids, s_ids = np.unique(true[true > 0]), np.unique(seg[seg > 0])
    n_true = np.bincount(true_cls[t_ids], minlength=n_classes)
    n_seg = np.bincount(seg_cls[s_ids], minlength=n_classes)
    same = true_cls[ti] == seg_cls[si]
    tp = np.bincount(true_cls[ti][same], minlength=n_classes)
    iou_sum = np.bincount(true_cls[ti][same], weights=iou[same], minlength=n_classes)
    aware = np.stack([tp, n_seg - tp, n_true - tp, iou_sum], axis=1).astype(float)
    agnostic = np.array([len(ti), len(s_ids) - len(ti), len(t_ids) - len(ti), iou.sum()], dtype=float)
    return aware, agnostic


def quality(aware: np.ndarray, agnostic: np.ndarray, class_names: list[str]) -> dict:
    """PQ, SQ and RQ, class-aware (mean over the classes that occur) and class-agnostic, from summed counts."""
    def pq(tp, fp, fn, s):
        with np.errstate(invalid="ignore", divide="ignore"):
            sq = np.where(tp > 0, s / np.maximum(tp, 1), 0.0)
            rq = tp / (tp + fp / 2 + fn / 2)
        return sq * rq, sq, rq

    tp, fp, fn, s = aware.T
    present = (tp + fp + fn) > 0
    p, q, r = pq(tp, fp, fn, s)
    ap, aq, ar = pq(*agnostic)
    per_class = {c: (None if not present[i] else {"pq": float(p[i]), "sq": float(q[i]), "rq": float(r[i]),
                                                  "tp": int(tp[i]), "fp": int(fp[i]), "fn": int(fn[i])})
                 for i, c in enumerate(class_names)}
    return {"pq": float(p[present].mean()), "sq": float(q[present].mean()), "rq": float(r[present].mean()),
            "pq_agnostic": float(ap), "sq_agnostic": float(aq), "rq_agnostic": float(ar),
            "true_parcels": int(agnostic[0] + agnostic[2]), "predicted_parcels": int(agnostic[0] + agnostic[1]),
            "per_class": per_class}


def variants(boundary: bool) -> list[str]:
    names = ["oracle", "extent", "semantic"]
    if boundary:
        names += [f"boundary@{t:g}" for t in TAUS] + [f"combined@{t:g}" for t in TAUS]
    return names


def chip_parts(pred: np.ndarray, mask: np.ndarray, parcels: np.ndarray, prob: np.ndarray | None,
               n_classes: int) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Per variant, the chip's class-aware and class-agnostic counts, and the boundary pixel counts."""
    extent = (mask >= 0) & (parcels > 0)
    true = pieces(np.where(extent, parcels, 0))
    out = {}
    if not true.any():
        return out
    true_cls = np.zeros(int(true.max()) + 1, dtype=np.int64)
    true_cls[true[extent]] = mask[extent]
    segs = {"oracle": true, "extent": pieces(extent.astype(np.int32)),
            "semantic": pieces(np.where(extent, pred.astype(np.int32) + 1, 0))}
    if prob is not None:
        p = prob.astype(np.float32) / 255.0
        for t in TAUS:
            segs[f"boundary@{t:g}"] = from_boundary(extent, p, t)
            segs[f"combined@{t:g}"] = from_boundary(extent, p, t, pred)
        # Boundary pixel counts against the dense target, at TAU: TP, FP, FN.
        target = (mask >= 0) & boundary_target(parcels)
        hit = extent & (p >= TAU)
        out["_boundary_pixels"] = (np.array([(hit & target).sum(), (hit & ~target).sum(),
                                             (extent & target & ~hit).sum()], dtype=float), None)
    for name, seg in segs.items():
        out[name] = chip_counts(true, true_cls, seg, vote(seg, pred, n_classes), n_classes)
    return out


def boundary_target(parcels: np.ndarray) -> np.ndarray:
    from delineation import boundary_map

    return boundary_map(parcels)


def _score_chunk(args) -> dict:
    preds, masks, probs, ids, chips_dir, n_classes = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    acc: dict[str, list] = {}
    for i, cid in enumerate(ids):
        parts = chip_parts(preds[i], masks[i], read_parcels(chips_dir, str(cid)),
                           None if probs is None else probs[i], n_classes)
        for k, (a, b) in parts.items():
            if k not in acc:
                acc[k] = [np.zeros_like(a), None if b is None else np.zeros_like(b)]
            acc[k][0] += a
            if b is not None:
                acc[k][1] += b
    return acc


def score_predictions(predictions: Path, chips_dir: Path, class_names: list[str], workers: int = 8) -> dict:
    """Panoptic scores of a cell's ``predictions_test.npz`` for every variant its predictions allow."""
    d = np.load(predictions)
    probs = d["boundary"] if "boundary" in d.files else None
    n = len(class_names)
    ids = d["ids"]
    chunks = np.array_split(np.arange(len(ids)), max(1, min(workers * 4, len(ids))))
    jobs = [(d["pred"][c], d["mask"][c], None if probs is None else probs[c], ids[c], Path(chips_dir), n)
            for c in chunks if len(c)]
    total: dict[str, list] = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for acc in ex.map(_score_chunk, jobs):
            for k, (a, b) in acc.items():
                if k not in total:
                    total[k] = [np.zeros_like(a), None if b is None else np.zeros_like(b)]
                total[k][0] += a
                if b is not None:
                    total[k][1] += b
    out = {"variants": {k: quality(a, b, class_names) for k, (a, b) in total.items() if not k.startswith("_")},
           "rule": RECORD}
    if "_boundary_pixels" in total:
        tp, fp, fn = total["_boundary_pixels"][0]
        out["boundary_pixels"] = {"f1": float(2 * tp / (2 * tp + fp + fn)), "precision": float(tp / (tp + fp)),
                                  "recall": float(tp / (tp + fn)), "tau": TAU}
    return out
