"""PaPs' parcels scored by the void rule of ``panoptic_x`` in each extent.

PaPs builds its parcels over the whole chip: a parcel is a centre's mask that survived the pseudo-NMS, its class the
class head's at the centre. The variant ``<extent>/paps`` keeps of every parcel only its pixels inside the extent X
(``oracle``, ``worldcover``, ``worldcover_30_40_90``, ``chip``); a parcel with no pixel left disappears. So
``chip/paps`` is PaPs as it is, and the other extents ask what a first-stage mask would add to it. The true parcels,
the void rule and the counts are ``panoptic_x``'s, exactly as for the boundary parcels of
``experiments/2026-11-05_parcel_extent``.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import panoptic_x as px

EXTENTS = ("oracle", "worldcover", "worldcover_30_40_90", "chip")


def chip_parts(inst, cls, mask, parcels, worldcover, n_classes, extents=EXTENTS) -> dict:
    E, true, true_cls, void = px.truth(mask, parcels)
    if not true.any():
        return {}
    inst = inst.astype(np.int64)
    seg_cls = np.zeros(int(inst.max()) + 1, np.int64)
    seg_cls[inst[inst > 0]] = cls[inst > 0]
    wc = worldcover if worldcover is not None else np.zeros(mask.shape, np.uint8)
    out = {}
    for x in extents:
        X = px.extent_of(x, mask, parcels, worldcover)
        seg = np.where(X, inst, 0)
        out[f"{x}/paps"] = px.chip_counts(true, true_cls, seg, seg_cls, void, n_classes, (E, wc))
        out[f"{x}/_cover"] = (np.array([(E & X).sum(), E.sum(), X.sum(), (X & (parcels == 0)).sum(),
                                        (X & void).sum(), X.size], dtype=float), None, None)
    return out


def _chunk(args) -> dict:
    items, chips_dir, n_classes, wc_path, extents = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    wc = np.load(wc_path) if wc_path else None
    acc: dict = {}
    for cid, inst, cls, mask in items:
        w = None if wc is None else wc[cid]
        px._add(acc, chip_parts(inst, cls, mask, read_parcels(chips_dir, cid), w, n_classes, extents))
    return acc


def score(preds: dict, chips_dir: Path, class_names: list[str], worldcover: Path | None, extents=EXTENTS,
          workers: int = 8) -> dict:
    """``preds``: ``chip -> (inst, cls, heatmap, saliency, mask)``, as ``fit_paps.predict`` returns them."""
    items = [(cid, v[0], v[1], v[4]) for cid, v in sorted(preds.items())]
    if worldcover is None:
        extents = tuple(e for e in extents if not e.startswith("worldcover"))
    n = len(class_names)
    wc = None if worldcover is None else str(worldcover)
    total: dict = {}
    if workers <= 1:
        total = _chunk((items, Path(chips_dir), n, wc, extents))
    else:
        parts = np.array_split(np.arange(len(items)), max(1, min(workers * 4, len(items))))
        jobs = [([items[i] for i in p], Path(chips_dir), n, wc, extents) for p in parts if len(p)]
        with ProcessPoolExecutor(workers) as ex:
            for acc in ex.map(_chunk, jobs):
                px._add(total, acc)
    out = px.summarise(total, class_names)
    out["rule"] = {**px.RECORD, "paps": "PaPs' parcels over the whole chip, each cut to the extent; class from "
                                        "PaPs' class head"}
    return out


def score_file(predictions: Path, chips_dir: Path, class_names: list[str], worldcover: Path | None,
               workers: int = 8) -> dict:
    # Each ``d[key]`` of an npz decompresses the whole array again, and a slice of it keeps that copy alive: read
    # each array once (indexing ``d["inst"][i]`` per chip held one full copy per chip and ran out of memory).
    with np.load(predictions) as d:
        ids, inst, cls, mask = d["ids"], d["inst"], d["cls"], d["mask"]
    preds = {str(c): (inst[i], cls[i], None, None, mask[i]) for i, c in enumerate(ids)}
    return score(preds, chips_dir, class_names, worldcover, workers=workers)
