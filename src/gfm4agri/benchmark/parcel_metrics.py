"""Parcel-level test scores from pixel predictions, by a majority vote of each parcel's pixels.

The labels of EuroCrops are parcel declarations, and the published EuroCrops benchmarks score parcels, so every
fit is scored per parcel beside its pixel metrics. Every parcel of the test chips takes the class most of its
labelled pixels were predicted as, ties going to the lowest class index, and the parcels are then scored as the
pixels are: per-class F1, their unweighted mean over the classes that occur (Macro-F1), overall accuracy, Cohen's
kappa and weighted F1.

A parcel is pooled over every test chip it touches, through the global identifier of ``<chip>.parcels.tif``. Only
its labelled pixels vote: those of a class outside the scheme carry the ignore index in the mask and are left out,
as they are from the pixel metrics. A parcel reaching outside the test chips is scored on the part inside them.
Everything is NumPy on a fit's stored ``predictions_test.npz``, so a finished fit is scored without a refit.
Shown on all of Estonia by ``experiments/2026-10-16_parcel_metric``.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import numpy as np

__all__ = ["RULE", "chip_votes", "pool_votes", "read_parcels", "score_parcels", "score_predictions"]

#: The voting rule, recorded beside every score.
RULE = ("majority vote of the labelled pixels of each parcel in the test chips, pooled across chips; "
        "ties to the lowest class index")


def chip_votes(pred: np.ndarray, mask: np.ndarray, parcels: np.ndarray, n_classes: int,
               where: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The votes of one chip's parcels: ``(ids, votes, reference)``.

    ``votes[i, c]`` counts the pixels of parcel ``ids[i]`` predicted as class ``c``, over its labelled pixels
    (``mask >= 0``, ``parcels > 0``) and, given ``where``, only those where it is true. ``reference[i]`` is the
    parcel's class in the mask, which must be the same under all its pixels.
    """
    sel = (mask >= 0) & (parcels > 0)
    if where is not None:
        sel &= where
    ids, inv = np.unique(parcels[sel], return_inverse=True)
    inv = inv.ravel()
    votes = np.bincount(inv * n_classes + pred[sel].astype(np.int64),
                        minlength=len(ids) * n_classes).reshape(len(ids), n_classes)
    labels = mask[sel].astype(np.int64)
    ref = np.zeros(len(ids), dtype=np.int64)
    ref[inv] = labels
    if not np.array_equal(ref[inv], labels):
        raise ValueError("a parcel carries more than one class in the mask")
    return ids.astype(np.int64), votes.astype(np.int64), ref


def pool_votes(parts: Iterable[tuple[np.ndarray, np.ndarray, np.ndarray]], n_classes: int
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The votes of several chips summed per parcel identifier, as :func:`chip_votes` returns them."""
    parts = list(parts)
    if not parts:
        return np.zeros(0, np.int64), np.zeros((0, n_classes), np.int64), np.zeros(0, np.int64)
    ids = np.concatenate([p[0] for p in parts])
    uniq, inv = np.unique(ids, return_inverse=True)
    votes = np.zeros((len(uniq), n_classes), dtype=np.int64)
    np.add.at(votes, inv, np.concatenate([p[1] for p in parts]))
    refs = np.concatenate([p[2] for p in parts])
    ref = np.zeros(len(uniq), dtype=np.int64)
    ref[inv] = refs
    if not np.array_equal(ref[inv], refs):
        raise ValueError("a parcel carries different classes in different chips")
    return uniq, votes, ref


def _scores(cm: np.ndarray) -> tuple[np.ndarray, dict]:
    """Per-class F1 of a confusion matrix (rows the reference), and its summary scores."""
    cm = cm.astype(float)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    present = (tp + fp + fn) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(present, 2 * tp / (2 * tp + fp + fn), np.nan)
    total, support = cm.sum(), cm.sum(1)
    if not total:
        return f1, {"macro_f1": float("nan"), "accuracy": float("nan"), "kappa": float("nan"),
                    "weighted_f1": float("nan")}
    po = tp.sum() / total
    pe = (cm.sum(0) * support).sum() / total ** 2
    return f1, {"macro_f1": float(np.nanmean(f1)), "accuracy": float(po),
                "kappa": float((po - pe) / (1 - pe)) if pe < 1 else float("nan"),
                "weighted_f1": float(np.nansum(f1 * support) / support.sum())}


def score_parcels(votes: np.ndarray, ref: np.ndarray, class_names: list[str]) -> dict:
    """Macro-F1, accuracy, kappa, weighted F1 and per-class F1 of the parcels' majority votes."""
    n = len(class_names)
    pred = votes.argmax(1)                       # ties to the lowest class index
    cm = np.bincount(ref * n + pred, minlength=n * n).reshape(n, n)
    f1, s = _scores(cm)
    return {**s, "n_parcels": int(cm.sum()),
            "per_class_f1": {c: (None if np.isnan(v) else float(v)) for c, v in zip(class_names, f1, strict=True)},
            "support": {c: int(v) for c, v in zip(class_names, cm.sum(1), strict=True)},
            "rule": RULE}


def read_parcels(chips_dir: Path, chip_id: str) -> np.ndarray:
    """``<chip>.parcels.tif``: the parcel identifier under every pixel, 0 outside any parcel."""
    import rasterio

    with rasterio.open(Path(chips_dir) / f"{chip_id}.parcels.tif") as src:
        return src.read(1)


def score_predictions(predictions: Path, chips_dir: Path, class_names: list[str]) -> dict:
    """The parcel scores of a fit's ``predictions_test.npz`` against the parcel rasters in ``chips_dir``.

    Also returns ``pixel_macro_f1``, the pixel Macro-F1 of the same predictions: it must reproduce the one the
    fit logged, or the predictions are not those of the model that was tested.
    """
    d = np.load(predictions)
    n = len(class_names)
    parts, pix = [], np.zeros((n, n), dtype=np.int64)
    for c, p, m in zip(d["ids"], d["pred"], d["mask"], strict=True):
        parts.append(chip_votes(p, m, read_parcels(chips_dir, str(c)), n))
        sel = m >= 0
        pix += np.bincount(m[sel].astype(np.int64) * n + p[sel].astype(np.int64),
                           minlength=n * n).reshape(n, n)
    _, votes, ref = pool_votes(parts, n)
    return {**score_parcels(votes, ref, class_names), "pixel_macro_f1": _scores(pix)[1]["macro_f1"]}
