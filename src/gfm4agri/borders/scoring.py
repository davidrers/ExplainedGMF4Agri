"""Every test score of a two-head cell: the crop map, the fields, the borders.

From ``experiments/2026-11-09_borders_all_arms/score_joint.py`` and the modules it called (``score_prue.py``,
``panoptic_x.py``, ``borders.py``), unchanged in behaviour, reduced to what the workflow reports: the fields over the
whole chip (the extent ``chip`` of those modules; their WorldCover and oracle extents are left out).

From a cell's ``predictions_test.npz`` (``prob``: the three border classes x 255; ``pred``: the crop class; ``mask``;
``ids``):

* **crop types** (:func:`crop_scores`), as the crop-only workflow scores its arms: the pixel Macro-F1 over the
  labelled test pixels and the parcel scores of protocol D28 (majority vote per parcel), by
  ``gfm4agri.benchmark.parcel_metrics.score_predictions``, with the per-class F1, IoU and accuracy beside them;
* **fields** (:func:`fields_file`): each variant of :func:`gfm4agri.borders.prue.parcels` builds the fields over the
  whole chip, with no truth, and the void rule scores them against the true parcel pieces (:func:`field_counts`):
  class-agnostic PQ, SQ and RQ (RQ is the object F1 at IoU > 0.5, Fields of The World's object metric), and
  class-aware with each field's class voted by the cell's own crop map; beside them, per pixel, the boundary
  precision, recall and F1 inside the declared extent and the three-class confusion where the target knows the class;
* **borders** (:func:`border_scores`): precision, recall and F1 of the borders of the grown fields against the
  declared borders at a tolerance of 0, 1 and 2 pixels, the recall by what lies across the border, the false borders
  by where they lie, and the check that the declared parcels score 1 against themselves.

**The void rule**, after Kirillov et al. (2019). Each test chip splits into the true parcels (``mask >= 0``, their
4-connected pieces the true units), **void** (``mask == -1`` and ``parcels > 0``: declared parcels of crops outside
the scheme) and **background** (``parcels == 0``):

1. void pixels are removed from every predicted field before the IoU;
2. the IoU of a field and a true piece uses the field's whole non-void area, background included;
3. a match is IoU > 0.5, which keeps matches unique;
4. a field matching no true piece (class-agnostic) whose pixels are more than half void is dropped;
5. every other unmatched field is an FP; every unmatched true piece is an FN.

Class-aware, a match must also agree on the class; the drop of rule 4 is decided class-agnostically.

**Borders.** The border of a set of fields, or of the declared parcels, is every field pixel with a 4-neighbour
inside the chip that carries another id or none (:func:`gfm4agri.borders.prue.boundary_map`). Scored where the truth
is known: the declared in-scheme parcels, and the land outside every declared parcel that WorldCover calls
non-agricultural; not scored: declared parcels of crops outside the scheme, and undeclared WorldCover grassland and
cropland. Per tolerance d (any direction, so d = 1 is a 3 x 3 window): precision is the share of the predicted border
pixels with a true border pixel within d, recall the share of the true border pixels with a predicted one within d.
At d = 1 the recall is split by what lies across each true border pixel (another declared parcel of the same crop, of
another crop, undeclared land WorldCover calls grassland or cropland, or other land; the first that applies), and the
false border pixels by where they lie.

The sums of IoU are added chunk by chunk in a fixed order, so a rescore with the same ``workers`` reproduces the
stored scores to every digit.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

from gfm4agri.borders import prue

IOU_MATCH = 0.5
#: WorldCover classes an FP on background is attributed to; any other goes to ``other``.
FP_WC = {10: "trees", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up", 60: "bare",
         80: "water", 90: "wetland"}
#: Border tolerances in pixels.
TOLERANCES = (0, 1, 2)
#: What lies across a true border pixel.
TYPES = ("same crop", "other crop", "undeclared farmland", "non-farm land")
#: Where a false border pixel lies.
PLACES = ("on a declared parcel", "on non-farm land")
#: The declared parcels taken as the prediction: must score 1.
CHECK = "declared parcels (check)"

RECORD = {
    "source": "experiments/2026-11-09_borders_all_arms (score_prue.py, panoptic_x.py), the extent 'chip'",
    "extent": "the whole chip: no test label is used to build the fields",
    "true_parcels": "4-connected pieces of each in-scheme parcel of each test chip",
    "variants": {"cc": "4-connected regions of the interior class, < 5 px dropped (PRUE)",
                 "grow": "cc grown by watershed on p(boundary) into the predicted field",
                 "field": "4-connected regions of interior or boundary, nothing split"},
    "class_of_a_parcel": "majority vote of all its predicted pixels, ties to the lowest class index",
    "void": "mask == -1 and parcels > 0: declared parcels of a class outside the scheme",
    "background": "parcels == 0",
    "rule": ["void pixels leave every predicted parcel before the IoU",
             "the IoU uses the predicted parcel's whole non-void area, background included",
             f"match at IoU > {IOU_MATCH}",
             "an unmatched (class-agnostic) predicted parcel more than half void is dropped",
             "every other unmatched predicted parcel is an FP, every unmatched true piece an FN"],
    "boundary_pixels": "scored inside E only, against the dense target",
}


# ------------------------------------------------------------------------------------------- fields: matching

def vote(segments: np.ndarray, pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Class of every segment id (index 0 unused): the majority of its predicted pixels, ties to the lowest."""
    n = int(segments.max()) + 1
    sel = segments > 0
    votes = np.bincount(segments[sel].astype(np.int64) * n_classes + pred[sel].astype(np.int64),
                        minlength=n * n_classes).reshape(n, n_classes)
    return votes.argmax(1)


def match(true: np.ndarray, seg: np.ndarray, void: np.ndarray):
    """Pairs ``(true id, segment id, IoU)`` above ``IOU_MATCH`` under the void rule, and the segments' areas.

    ``true`` labels the true pieces (0 elsewhere; never on void), ``seg`` the predicted parcels anywhere in the chip
    (0 = none). Returns ``ti, si, iou`` and, per segment id, its pixels and its void pixels.
    """
    seg = seg.astype(np.int64)
    base = int(seg.max()) + 1
    n_all = np.bincount(seg.ravel(), minlength=base)
    n_void = np.bincount(seg[void], minlength=base)
    area = n_all - n_void                                 # rule 1: non-void pixels, anywhere in the chip
    t = true.astype(np.int64)
    at = np.bincount(t.ravel())
    sel = (t > 0) & (seg > 0)
    if not sel.any():
        e = np.zeros(0, np.int64)
        return e, e, np.zeros(0), n_all, n_void
    pairs, inter = np.unique(t[sel] * base + seg[sel], return_counts=True)
    ti, si = pairs // base, pairs % base
    iou = inter / (at[ti] + area[si] - inter)             # rule 2
    keep = iou > IOU_MATCH                                # rule 3
    return ti[keep], si[keep], iou[keep], n_all, n_void


def field_counts(true, true_cls, seg, seg_cls, void, n_classes, where=None):
    """Per class ``[TP, FP, FN, sum IoU]`` and the same four class-agnostic, plus ``[dropped, FP on parcels,
    FP on background by WorldCover class...]``, for one chip under the void rule."""
    ti, si, iou, n_all, n_void = match(true, seg, void)
    t_ids = np.unique(true[true > 0])
    s_ids = np.nonzero(n_all)[0]
    s_ids = s_ids[s_ids > 0]
    matched = np.zeros(len(n_all), bool)
    matched[si] = True
    dropped = (~matched[s_ids]) & (n_void[s_ids] * 2 > n_all[s_ids])     # rule 4
    kept = s_ids[~dropped]
    n_true = np.bincount(true_cls[t_ids], minlength=n_classes)
    n_seg = np.bincount(seg_cls[kept], minlength=n_classes)
    same = true_cls[ti] == seg_cls[si]
    tp = np.bincount(true_cls[ti][same], minlength=n_classes)
    iou_sum = np.bincount(true_cls[ti][same], weights=iou[same], minlength=n_classes)
    aware = np.stack([tp, n_seg - tp, n_true - tp, iou_sum], axis=1).astype(float)
    agnostic = np.array([len(ti), len(kept) - len(ti), len(t_ids) - len(ti), iou.sum()], dtype=float)
    extra = np.zeros(3 + len(FP_WC), dtype=float)
    extra[0] = dropped.sum()
    fp = kept[~matched[kept]]
    if len(fp) and where is not None:
        on_parcel, wc = where
        segl = seg.astype(np.int64)
        nv = ~void
        par = np.bincount(segl[nv & on_parcel], minlength=len(n_all))
        bg = (n_all - n_void) - par
        on_p = par[fp] >= bg[fp]
        extra[1] = on_p.sum()
        rest = fp[~on_p]
        if len(rest):
            bgm = nv & ~on_parcel
            lut = np.full(256, len(FP_WC), np.int64)
            for k, c in enumerate(FP_WC):
                lut[c] = k
            cnt = np.zeros((len(n_all), len(FP_WC) + 1), np.int64)
            np.add.at(cnt, (segl[bgm], lut[wc[bgm]]), 1)
            top = cnt[rest].argmax(1)
            extra[2:] += np.bincount(top, minlength=len(FP_WC) + 1)
    return aware, agnostic, extra


def quality(aware: np.ndarray, agnostic: np.ndarray, class_names: list[str], extra: np.ndarray | None = None) -> dict:
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
    out = {"pq": float(p[present].mean()), "sq": float(q[present].mean()), "rq": float(r[present].mean()),
           "pq_agnostic": float(ap), "sq_agnostic": float(aq), "rq_agnostic": float(ar),
           "tp_agnostic": int(agnostic[0]), "fp_agnostic": int(agnostic[1]), "fn_agnostic": int(agnostic[2]),
           "true_parcels": int(agnostic[0] + agnostic[2]), "predicted_parcels": int(agnostic[0] + agnostic[1]),
           "per_class": per_class}
    if extra is not None:
        out["dropped_void"] = int(extra[0])
        out["fp_on_parcels"] = int(extra[1])
        out["fp_on_background"] = {name: int(v) for name, v in zip([*FP_WC.values(), "other"], extra[2:])}
    return out


def truth(mask: np.ndarray, parcels: np.ndarray):
    """The true pieces, their classes, the void pixels and the in-scheme parcel pixels of one chip."""
    E = (mask >= 0) & (parcels > 0)
    true = prue.pieces(np.where(E, parcels, 0))
    true_cls = np.zeros(int(true.max()) + 1, dtype=np.int64)
    true_cls[true[E]] = mask[E]
    void = (mask < 0) & (parcels > 0)
    return E, true, true_cls, void


def _add(acc: dict, parts: dict) -> None:
    for k, trip in parts.items():
        if k not in acc:
            acc[k] = [None if a is None else np.zeros_like(a) for a in trip]
        for i, a in enumerate(trip):
            if a is not None:
                acc[k][i] += a


# --------------------------------------------------------------------------------------- fields: PRUE's own

def chip_parts(prob, mask, parcels, worldcover, crop_pred, n_classes, variants=prue.VARIANTS) -> dict:
    """The counts of one chip: per ``chip/<variant>`` the fields', and the pixel scores (keys starting ``_``)."""
    E, true, true_cls, void = truth(mask, parcels)
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
        seg_cls = (vote(inst, crop_pred, n_classes) if crop_pred is not None
                   else np.zeros(int(inst.max()) + 1, np.int64))
        out[f"chip/{v}"] = field_counts(true, true_cls, inst, seg_cls, void, n_classes, (E, wc))
    return out


def _fields_chunk(args) -> dict:
    items, chips_dir, n_classes, wc_path, variants = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    wc = np.load(wc_path) if wc_path else None
    acc: dict = {}
    for cid, prob, mask, crop in items:
        w = None if wc is None else wc[cid]
        _add(acc, chip_parts(prob, mask, read_parcels(chips_dir, cid), w, crop, n_classes, variants))
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
          crop: dict | None = None, variants=prue.VARIANTS, workers: int = 8) -> dict:
    """``probs`` ``(N, 3, H, W)`` uint8, ``masks`` ``(N, H, W)``, ``ids``; ``crop``: chip -> crop-type map or None."""
    n = len(class_names)
    items = [(str(c), probs[i], masks[i], None if crop is None else crop[str(c)]) for i, c in enumerate(ids)]
    wc = None if worldcover is None else str(worldcover)
    total: dict = {}
    if workers <= 1:
        total = _fields_chunk((items, Path(chips_dir), n, wc, variants))
    else:
        parts = np.array_split(np.arange(len(items)), max(1, min(workers * 4, len(items))))
        jobs = [([items[i] for i in p], Path(chips_dir), n, wc, variants) for p in parts if len(p)]
        with ProcessPoolExecutor(workers) as ex:
            for acc in ex.map(_fields_chunk, jobs):
                _add(total, acc)
    out = {"variants": {k: quality(a, b, class_names, e) for k, (a, b, e) in total.items()
                        if not k.startswith("_")}}
    if crop is None:
        for v in out["variants"].values():
            for key in ("pq", "sq", "rq", "per_class"):
                v.pop(key, None)
    out["pixels"] = _pixels(total)
    out["class_source"] = "crop-type map vote" if crop is not None else "none (class-agnostic only)"
    out["rule"] = RECORD
    return out


def load_cropmap(path: Path | None) -> dict | None:
    if path is None or not Path(path).exists():
        return None
    with np.load(path) as d:
        ids, pred = d["ids"], d["pred"]
    return {str(c): pred[i] for i, c in enumerate(ids)}


def fields_file(predictions: Path, chips_dir: Path, class_names: list[str], worldcover: Path | None,
                cropmap: Path | None = None, workers: int = 8) -> dict:
    """The fields' scores of one ``predictions_test.npz``, classed by ``cropmap``'s vote when given."""
    with np.load(predictions) as d:
        ids, prob, mask = d["ids"], d["prob"], d["mask"]
    crop = load_cropmap(cropmap)
    if crop is not None and not all(str(c) in crop for c in ids):
        crop = None
    out = score(prob, mask, ids, chips_dir, class_names, worldcover, crop=crop, workers=workers)
    out["cropmap"] = str(cropmap) if crop is not None else None
    return out


# ------------------------------------------------------------------------------------------------------ borders

def regions(mask: np.ndarray, parcels: np.ndarray, wc: np.ndarray):
    """The declared in-scheme parcels E, the scored pixels, and the true border inside E."""
    E = (mask >= 0) & (parcels > 0)
    scored = E | ((parcels == 0) & ~np.isin(wc, prue.AGRI))
    return E, scored, E & prue.boundary_map(parcels)


def border_types(parcels: np.ndarray, mask: np.ndarray, wc: np.ndarray, true_border: np.ndarray) -> np.ndarray:
    """Per pixel, what lies across: 0 same crop, 1 other crop, 2 undeclared farmland, 3 non-farm land; -1 off the
    true border."""
    farm = np.isin(wc, prue.AGRI)
    h, w = parcels.shape
    same = np.zeros((h, w), bool)
    other = np.zeros((h, w), bool)
    undeclared = np.zeros((h, w), bool)
    nonfarm = np.zeros((h, w), bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        p, q = parcels[ys, xs], parcels[yd, xd]
        m, n = mask[ys, xs], mask[yd, xd]
        diff = p != q
        undeclared[ys, xs] |= diff & (q == 0) & farm[yd, xd]
        nonfarm[ys, xs] |= diff & (q == 0) & ~farm[yd, xd]
        same[ys, xs] |= diff & (q > 0) & (n == m) & (n >= 0)
        other[ys, xs] |= diff & (q > 0) & ((n != m) | (n < 0))
    out = np.full((h, w), -1, np.int8)
    out[true_border & nonfarm] = 3
    out[true_border & undeclared] = 2
    out[true_border & other] = 1
    out[true_border & same] = 0
    return out


def _near(border: np.ndarray, d: int) -> np.ndarray:
    return ndimage.binary_dilation(border, np.ones((2 * d + 1, 2 * d + 1), bool)) if d else border


def border_counts(seg, true_border, scored, E, types) -> dict:
    """The counts of one set of fields on one chip: per tolerance, by border type and by place of the false
    borders."""
    pred = prue.boundary_map(seg)
    p = pred & scored
    tol = []
    for d in TOLERANCES:
        near_true, near_pred = _near(true_border, d), _near(pred, d)
        tol.append([(p & near_true).sum(), p.sum(), (true_border & near_pred).sum(), true_border.sum()])
    near_pred = _near(pred, 1)
    by_type = [[(near_pred & (types == k)).sum(), (types == k).sum()] for k in range(len(TYPES))]
    false = p & ~_near(true_border, 1)
    places = [(false & E).sum(), (false & ~E).sum()]
    return {"tol": np.array(tol, np.int64), "types": np.array(by_type, np.int64), "places": np.array(places, np.int64)}


def _borders_chunk(args):
    """The border counts of PRUE's grown fields (``"joint"``) and of the declared parcels (``CHECK``) over a chunk,
    and the per-chip d = 1 counts of the former."""
    ids, masks, wcs, probs, chips_dir = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    total: dict = {}
    per_chip = []
    for i, cid in enumerate(ids):
        parcels, mask = read_parcels(chips_dir, cid), masks[i].astype(np.int64)
        E, scored, true_border = regions(mask, parcels, wcs[i])
        types = border_types(parcels, mask, wcs[i], true_border)
        segs = {"joint": prue.parcels(probs[i], "grow"), CHECK: parcels.astype(np.int64)}
        for m, seg in segs.items():
            c = border_counts(seg, true_border, scored, E, types)
            t = total.setdefault(m, {k: 0 for k in c})
            for k, v in c.items():
                t[k] = t[k] + v
            if m == "joint":
                per_chip.append(c["tol"][TOLERANCES.index(1)])
    return total, np.array(per_chip, np.int64)


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def _border_row(c: dict) -> dict:
    row = {}
    for k, d in enumerate(TOLERANCES):
        tp_p, n_p, tp_r, n_r = (int(x) for x in c["tol"][k])
        p, r = tp_p / max(n_p, 1), tp_r / max(n_r, 1)
        row[f"d{d}"] = {"precision": p, "recall": r, "f1": _f1(p, r), "pred_border_px": n_p,
                        "true_border_px": n_r}
    row["recall_by_type_d1"] = {t: {"recall": int(f) / max(int(n), 1), "true_border_px": int(n)}
                                for t, (f, n) in zip(TYPES, c["types"])}
    n_p = int(c["tol"][TOLERANCES.index(1)][1])
    row["false_border_share_d1"] = {pl: int(v) / max(n_p, 1) for pl, v in zip(PLACES, c["places"])}
    return row


def border_scores(predictions: Path, chips_dir: Path, worldcover: Path, workers: int = 8, chunk: int = 40,
                  per_chip_out: Path | None = None) -> dict:
    """Border precision, recall and F1 of the grown fields; ``per_chip_out``: the d = 1 counts per chip."""
    with np.load(predictions) as z:
        ids, prob, mask = [str(c) for c in z["ids"]], z["prob"], z["mask"]
    wc = np.load(worldcover)
    wcs = np.stack([wc[c] for c in ids])
    jobs = [(ids[s:s + chunk], mask[s:s + chunk], wcs[s:s + chunk], prob[s:s + chunk], Path(chips_dir))
            for s in range(0, len(ids), chunk)]
    total: dict = {}
    per_chip: list = []
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for t, pc in ex.map(_borders_chunk, jobs):
            for m, c in t.items():
                acc = total.setdefault(m, {k: 0 for k in c})
                for k, v in c.items():
                    acc[k] = acc[k] + v
            per_chip.append(pc)
    if per_chip_out is not None:
        np.savez_compressed(per_chip_out, ids=np.array(ids), counts=np.concatenate(per_chip))
    out = _border_row(total["joint"])
    check = _border_row(total[CHECK])["d1"]["f1"]
    if abs(check - 1.0) > 1e-9:
        raise RuntimeError(f"the declared parcels score border F1 {check} against themselves, not 1")
    out["check_declared_f1_d1"] = check
    out["fields"] = "PRUE's own fields grown into their rims (prue.parcels 'grow'), whole chip, no test label"
    out["scored_on"] = "declared in-scheme parcels and WorldCover non-agricultural land outside every declared parcel"
    return out


# --------------------------------------------------------------------------------------------------- crop types

def crop_scores(predictions: Path, chips_dir: Path, class_names: list[str]) -> dict:
    """Pixel and parcel scores of the crop map, the crop-only workflow's ``parcel_metrics`` with the pixel ones."""
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


def score_file(predictions: Path, chips_dir: Path, class_names: list[str], worldcover: Path, workers: int = 8,
               per_chip_out: Path | None = None) -> dict:
    """``{"crop", "panoptic", "borders"}`` of one ``predictions_test.npz``."""
    crop = crop_scores(predictions, chips_dir, class_names)
    pan = fields_file(predictions, chips_dir, class_names, worldcover, cropmap=predictions, workers=workers)
    pan["class_source"] = "the model's own crop head (vote of its pixels in each field)"
    bnd = border_scores(predictions, chips_dir, worldcover, workers=workers, per_chip_out=per_chip_out)
    return {"crop": crop, "panoptic": pan, "borders": bnd}
