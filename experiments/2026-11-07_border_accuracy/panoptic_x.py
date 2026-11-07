"""Copied unchanged from experiments/2026-11-05_parcel_extent/panoptic_x.py, so this folder scores PaPs' parcels by
the same rule and stays deletable on its own.

Parcels built and scored inside an extent that is not the truth: a copy of
``experiments/2026-10-18_parcel_delineation/panoptic.py`` with an extent argument and the void rule.

The source folder's ``panoptic.py`` builds every predicted parcel inside **E**, ``(mask >= 0) & (parcels > 0)``, the
pixels of the declared in-scheme parcels: an oracle first stage. Here the parcels are built inside an extent **X**
taken from something other than the truth, and the truth is used only to score them
(``NEXT_non_oracle_extent.md`` of that folder):

* ``oracle``: E, the source's extent, as the upper bound;
* ``worldcover``: the pixels ESA WorldCover 2021 calls cropland (40) or grassland (30) (``WORLDCOVER``);
* ``chip``: the whole chip, nothing known, as the lower bound.

**Building the parcels, no ground truth.** For every X, as the source does with E: ``extent`` (the 4-connected
regions of X), ``semantic`` (the 4-connected regions of one predicted class inside X), ``boundary@tau`` (seeds where
the boundary probability is below tau, grown by watershed on it inside X; a seedless region of X is one parcel) and
``combined@tau`` (seeds also cut where the class changes). Each parcel's class is the vote of all its predicted
pixels. The ``oracle`` variant (the true pieces, classed by the vote) does not depend on X.

**Scoring: the truth is used only here.** Each test chip splits into the true parcels (``mask >= 0``, their
4-connected pieces the true units), **void** (``mask == -1`` and ``parcels > 0``: declared parcels of crops outside
the 20 classes) and **background** (``parcels == 0``). The void rule, after Kirillov et al. (2019):

1. void pixels are removed from every predicted parcel before the IoU: they count in neither intersection nor union;
2. the IoU of a predicted parcel and a true piece uses the predicted parcel's whole non-void area, background
   included;
3. a match is IoU > 0.5, which keeps matches unique;
4. a predicted parcel matching no true piece (class-agnostic) whose pixels are more than half void is dropped: it is
   neither TP nor FP;
5. every other unmatched predicted parcel is an FP; every unmatched true piece is an FN, including true pixels
   outside X, where no parcel was built.

utae-paps' ``PanopticMeter`` (``src/panoptic/metrics.py``) does the same with PASTIS' void label in two steps: it
deletes every predicted instance whose IoU with a void instance exceeds 0.5, then zeroes the void pixels of the rest
before matching. Rule 4 drops a parcel by its void share instead, which also drops a parcel spread over several
small void parcels. Class-aware, a match must also agree on the class; the drop of rule 4 is decided
class-agnostically, so a parcel matched with the wrong class stays an FP of its class.

With X = E there is no void or background inside X, and these scores equal the source's to every digit
(``check.py``).

Each FP is also counted by where it lies: on the true parcels (most of its non-void pixels on in-scheme parcels:
over- or under-segmentation) or on background, then by the WorldCover class most of its background pixels carry.

The boundary pixel F1 stays scored inside E: outside E there is no border truth.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage

#: The threshold on the boundary probability below which a pixel seeds a parcel; the same for every arm.
TAU = 0.5
TAUS = (0.3, 0.5, 0.7)
IOU_MATCH = 0.5
FOUR = ndimage.generate_binary_structure(2, 1)
#: WorldCover classes of the ``worldcover`` extent: grassland and cropland.
WORLDCOVER = (30, 40)
#: The extents, in the order they are reported.
EXTENTS = ("oracle", "worldcover", "chip")
#: WorldCover classes an FP on background is attributed to; any other goes to ``other``.
FP_WC = {10: "trees", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built-up", 60: "bare",
         80: "water", 90: "wetland"}

RECORD = {
    "source": "experiments/2026-10-18_parcel_delineation/panoptic.py, extended with an extent and the void rule",
    "extents": {"oracle": "(mask >= 0) & (parcels > 0): the declared in-scheme parcels, the upper bound",
                "worldcover": f"ESA WorldCover 2021 v200 classes {list(WORLDCOVER)}",
                "chip": "the whole chip, the lower bound"},
    "true_parcels": "4-connected pieces of each in-scheme parcel of each test chip",
    "variants": {
        "oracle": "the true pieces, classed by the vote (independent of the extent)",
        "extent": "4-connected regions of X, no geometry predicted: the floor of X",
        "semantic": "4-connected regions of one predicted class inside X",
        "boundary": "seeds where the boundary probability < tau, grown by watershed on it inside X; "
                    "a seedless region of X is one parcel",
        "combined": "as boundary, seeds also cut where the predicted class changes",
    },
    "class_of_a_parcel": "majority vote of all its predicted pixels, ties to the lowest class index",
    "void": "mask == -1 and parcels > 0: declared parcels of a class outside the scheme",
    "background": "parcels == 0",
    "rule": ["void pixels leave every predicted parcel before the IoU",
             "the IoU uses the predicted parcel's whole non-void area, background included",
             f"match at IoU > {IOU_MATCH}",
             "an unmatched (class-agnostic) predicted parcel more than half void is dropped",
             "every other unmatched predicted parcel is an FP, every unmatched true piece an FN"],
    "tau": TAU, "taus": list(TAUS),
    "boundary_pixels": "scored inside E only, against the dense target",
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
    """Parcels from the boundary probability: interior seeds grown by watershed inside the extent."""
    from skimage.segmentation import watershed

    core = extent & (prob < tau)
    seeds = pieces(np.where(core, pred.astype(np.int32) + 1, 0) if pred is not None else core.astype(np.int32))
    grown = watershed(prob, markers=seeds, mask=extent, connectivity=1) if seeds.any() else np.zeros_like(seeds)
    rest = extent & (grown == 0)
    if rest.any():
        extra = pieces(rest.astype(np.int32))
        grown = np.where(rest, extra + grown.max(), grown)
    return grown.astype(np.int32)


def extent_of(name: str, mask: np.ndarray, parcels: np.ndarray, worldcover: np.ndarray | None) -> np.ndarray:
    if name == "oracle":
        return (mask >= 0) & (parcels > 0)
    if name == "chip":
        return np.ones(mask.shape, dtype=bool)
    if name.startswith("worldcover"):
        if worldcover is None:
            raise ValueError("the worldcover extent needs the chip's WorldCover map")
        classes = WORLDCOVER if name == "worldcover" else tuple(int(c) for c in name.split("_")[1:])
        return np.isin(worldcover, classes)
    raise ValueError(f"extent {name!r}")


# ------------------------------------------------------------------------------------------------------ scoring

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


def chip_counts(true, true_cls, seg, seg_cls, void, n_classes, where=None):
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


def variants(boundary: bool) -> list[str]:
    names = ["extent", "semantic"]
    if boundary:
        names += [f"boundary@{t:g}" for t in TAUS] + [f"combined@{t:g}" for t in TAUS]
    return names


def segments_in(extent: np.ndarray, pred: np.ndarray, prob: np.ndarray | None) -> dict[str, np.ndarray]:
    """Every variant's predicted parcels inside one extent."""
    segs = {"extent": pieces(extent.astype(np.int32)),
            "semantic": pieces(np.where(extent, pred.astype(np.int32) + 1, 0))}
    if prob is not None:
        p = prob.astype(np.float32) / 255.0
        for t in TAUS:
            segs[f"boundary@{t:g}"] = from_boundary(extent, p, t)
            segs[f"combined@{t:g}"] = from_boundary(extent, p, t, pred)
    return segs


def truth(mask: np.ndarray, parcels: np.ndarray):
    """The true pieces, their classes, the void pixels and the in-scheme parcel pixels of one chip."""
    E = (mask >= 0) & (parcels > 0)
    true = pieces(np.where(E, parcels, 0))
    true_cls = np.zeros(int(true.max()) + 1, dtype=np.int64)
    true_cls[true[E]] = mask[E]
    void = (mask < 0) & (parcels > 0)
    return E, true, true_cls, void


def chip_parts(pred, mask, parcels, prob, worldcover, n_classes, extents=EXTENTS) -> dict:
    """Per ``<extent>/<variant>`` (and ``oracle``), the chip's class-aware, class-agnostic and extra counts."""
    E, true, true_cls, void = truth(mask, parcels)
    out = {}
    if not true.any():
        return out
    wc = worldcover if worldcover is not None else np.zeros(mask.shape, np.uint8)
    where = (E, wc)
    out["oracle"] = chip_counts(true, true_cls, true, vote(true, pred, n_classes), void, n_classes, where)
    for x in extents:
        X = extent_of(x, mask, parcels, worldcover)
        for name, seg in segments_in(X, pred, prob).items():
            out[f"{x}/{name}"] = chip_counts(true, true_cls, seg, vote(seg, pred, n_classes), void, n_classes, where)
        # How much of the truth X holds: true pixels inside X.
        out[f"{x}/_cover"] = (np.array([(E & X).sum(), E.sum(), X.sum(), (X & (parcels == 0)).sum(),
                                        (X & void).sum(), X.size], dtype=float), None, None)
    if prob is not None:
        from delineation import boundary_map

        p = prob.astype(np.float32) / 255.0
        target = (mask >= 0) & boundary_map(parcels)
        hit = E & (p >= TAU)
        out["_boundary_pixels"] = (np.array([(hit & target).sum(), (hit & ~target).sum(),
                                             (E & target & ~hit).sum()], dtype=float), None, None)
    return out


def _add(acc: dict, parts: dict) -> None:
    for k, trip in parts.items():
        if k not in acc:
            acc[k] = [None if a is None else np.zeros_like(a) for a in trip]
        for i, a in enumerate(trip):
            if a is not None:
                acc[k][i] += a


def _score_chunk(args) -> dict:
    preds, masks, probs, ids, chips_dir, n_classes, wc_path, extents = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    wc = np.load(wc_path) if wc_path else None
    acc: dict = {}
    for i, cid in enumerate(ids):
        w = None if wc is None else wc[str(cid)]
        parts = chip_parts(preds[i], masks[i], read_parcels(chips_dir, str(cid)),
                           None if probs is None else probs[i], w, n_classes, extents)
        _add(acc, parts)
    return acc


def summarise(total: dict, class_names: list[str]) -> dict:
    """The scores of summed chip counts: ``variants`` keyed ``<extent>/<variant>`` (and ``oracle``), the extents'
    cover of the truth, and the boundary pixel F1 inside E."""
    out = {"variants": {}, "extents": {}, "rule": RECORD}
    for k, (a, b, e) in total.items():
        if k.endswith("/_cover"):
            inside, true_px, x_px, x_bg, x_void, n = a
            out["extents"][k.split("/")[0]] = {
                "true_pixels_inside": float(inside / true_px), "share_of_chip": float(x_px / n),
                "background_share_of_extent": float(x_bg / max(x_px, 1)),
                "void_share_of_extent": float(x_void / max(x_px, 1))}
        elif not k.startswith("_"):
            out["variants"][k] = quality(a, b, class_names, e)
    if "_boundary_pixels" in total:
        tp, fp, fn = total["_boundary_pixels"][0]
        out["boundary_pixels"] = {"f1": float(2 * tp / (2 * tp + fp + fn)), "precision": float(tp / (tp + fp)),
                                  "recall": float(tp / (tp + fn)), "tau": TAU, "inside": "oracle extent E"}
    return out


def score_predictions(predictions: Path, chips_dir: Path, class_names: list[str], worldcover: Path | None,
                      extents=EXTENTS, workers: int = 8) -> dict:
    """Panoptic scores of a cell's ``predictions_test.npz`` for every extent and variant its predictions allow."""
    d = np.load(predictions)
    probs = d["boundary"] if "boundary" in d.files else None
    n = len(class_names)
    ids = d["ids"]
    chunks = np.array_split(np.arange(len(ids)), max(1, min(workers * 4, len(ids))))
    jobs = [(d["pred"][c], d["mask"][c], None if probs is None else probs[c], ids[c], Path(chips_dir), n,
             None if worldcover is None else str(worldcover), tuple(extents)) for c in chunks if len(c)]
    total: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for acc in ex.map(_score_chunk, jobs):
            _add(total, acc)
    return summarise(total, class_names)
