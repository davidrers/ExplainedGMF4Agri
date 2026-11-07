"""Every test score of a joint cell, each computed exactly as for the separate model it is compared with.

From the cell's ``predictions_test.npz`` (``prob``: the border classes x 255; ``pred``: the crop class; ``mask``;
``ids``):

* **crop types**, as the main workflow scores its crop arm: the pixel Macro-F1 over the labelled test pixels and the
  parcel scores of protocol D28 (majority vote per parcel), by the core's
  ``gfm4agri.benchmark.parcel_metrics.score_predictions``, with the pixel per-class F1, IoU and accuracy beside them
  (``gfm4agri.benchmark.evaluation.per_class_scores``);
* **parcels**, as the PRUE experiment scores its fields: ``score_prue.score_file`` (void-rule PQ of every variant
  and extent, class-agnostic, and class-aware with each field's class voted by the crop map, here the model's **own**
  crop head);
* **borders**, as the border-accuracy experiment scores the final fields: precision, recall and F1 of the borders of
  PRUE's grown fields against the declared borders at a tolerance of 0, 1 and 2 pixels, recall by what lies across
  the border, the false borders by where they lie (``borders.py``, copied unchanged from
  ``../2026-11-07_border_accuracy``), and its check that the declared parcels score 1.

``check.py --scorer`` runs this on a file assembled from the stored outputs of the two separate frozen models and
must reproduce their published scores.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np


def crop_scores(predictions: Path, chips_dir: Path, class_names: list[str]) -> dict:
    """Pixel and parcel scores of the crop map, the main workflow's ``parcel_metrics`` with the pixel ones."""
    from gfm4agri.benchmark.evaluation import per_class_scores
    from gfm4agri.benchmark.parcel_metrics import score_predictions

    parcel = score_predictions(predictions, chips_dir, class_names)
    n = len(class_names)
    cm = np.zeros((n, n), np.int64)
    with np.load(predictions) as d:
        pred, mask = d["pred"], d["mask"]
    for p, m in zip(pred, mask, strict=True):
        sel = m >= 0
        cm += np.bincount(m[sel].astype(np.int64) * n + p[sel].astype(np.int64), minlength=n * n).reshape(n, n)
    s = per_class_scores(cm)
    pixel = {"macro_f1": parcel["pixel_macro_f1"], "miou": s["miou"], "accuracy": s["accuracy"],
             "per_class": {c: {"f1": float(s["f1"][i]), "iou": float(s["iou"][i]), "pixels": int(cm[i].sum())}
                           for i, c in enumerate(class_names)}, "confusion": cm.tolist()}
    return {"pixel": pixel, "parcel": parcel}


def _border_row(c: dict) -> dict:
    import borders

    row = {}
    for k, d in enumerate(borders.TOLERANCES):
        tp_p, n_p, tp_r, n_r = (int(x) for x in c["tol"][k])
        p, r = tp_p / max(n_p, 1), tp_r / max(n_r, 1)
        row[f"d{d}"] = {"precision": p, "recall": r, "f1": borders._f1(p, r), "pred_border_px": n_p,
                        "true_border_px": n_r}
    row["recall_by_type_d1"] = {t: {"recall": int(f) / max(int(n), 1), "true_border_px": int(n)}
                                for t, (f, n) in zip(borders.TYPES, c["types"])}
    n_p = int(c["tol"][borders.TOLERANCES.index(1)][1])
    row["false_border_share_d1"] = {pl: int(v) / max(n_p, 1) for pl, v in zip(borders.PLACES, c["places"])}
    return row


def border_scores(predictions: Path, chips_dir: Path, worldcover: Path, workers: int = 8, chunk: int = 40,
                  per_chip_out: Path | None = None) -> dict:
    """Border precision, recall and F1 of the grown fields, as ``../2026-11-07_border_accuracy/borders.py``."""
    import borders

    with np.load(predictions) as z:
        ids, prob, mask = [str(c) for c in z["ids"]], z["prob"], z["mask"]
    wc = np.load(worldcover)
    wcs = np.stack([wc[c] for c in ids])
    jobs = [(ids[s:s + chunk], mask[s:s + chunk], wcs[s:s + chunk], {"joint": prob[s:s + chunk]}, {"joint": "prue"},
             Path(chips_dir)) for s in range(0, len(ids), chunk)]
    total: dict = {}
    per_chip: list = []
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for t, pc in ex.map(borders._chunk, jobs):
            for m, c in t.items():
                acc = total.setdefault(m, {k: 0 for k in c})
                for k, v in c.items():
                    acc[k] = acc[k] + v
            per_chip.append(pc["joint"])
    if per_chip_out is not None:
        np.savez_compressed(per_chip_out, ids=np.array(ids), counts=np.concatenate(per_chip))
    out = _border_row(total["joint"])
    check = _border_row(total[borders.CHECK])["d1"]["f1"]
    if abs(check - 1.0) > 1e-9:
        raise RuntimeError(f"the declared parcels score border F1 {check} against themselves, not 1")
    out["check_declared_f1_d1"] = check
    out["fields"] = "PRUE's own fields grown into their rims (prue.parcels 'grow'), whole chip, no test label"
    out["scored_on"] = "declared in-scheme parcels and WorldCover non-agricultural land outside every declared parcel"
    return out


def score_file(predictions: Path, chips_dir: Path, class_names: list[str], worldcover: Path, workers: int = 8,
               per_chip_out: Path | None = None) -> dict:
    """``{"crop", "panoptic", "borders"}`` of one ``predictions_test.npz``."""
    import score_prue

    crop = crop_scores(predictions, chips_dir, class_names)
    pan = score_prue.score_file(predictions, chips_dir, class_names, worldcover, cropmap=predictions,
                                workers=workers)
    pan["class_source"] = "the joint model's own crop head (vote of its pixels in each field)"
    bnd = border_scores(predictions, chips_dir, worldcover, workers=workers, per_chip_out=per_chip_out)
    return {"crop": crop, "panoptic": pan, "borders": bnd}
