"""Checks of this folder's scorer (``panoptic_x.py``) before it scores anything, from the repository root:

    poetry run python experiments/2026-11-05_parcel_extent/check.py                       # synthetic cases
    poetry run python experiments/2026-11-05_parcel_extent/check.py --regression EE_2021_rev   # and the stored cells

The synthetic cases test the void rule: a predicted parcel on void is dropped, one on background is an FP, a true
piece outside the extent is an FN, and a parcel's IoU counts its background pixels but not its void ones. The
regression check rescores the stored predictions of every cell of ``2026-10-18_parcel_delineation`` with X = E and
compares every variant with the ``panoptic_metrics`` its ``results.json`` holds, to every digit: with X = E there is no
void and no background inside X, so the void rule must change nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402

import panoptic_x as px  # noqa: E402

NAMES = ["c0", "c1", "c2"]


def chip():
    """A (id 7, class 1) and B (id 9, class 1) touch; C (id 4, class 2) lies apart; V (id 5) is a declared parcel
    of a class outside the scheme (void); the rest is background."""
    parcels = np.zeros((12, 12), np.int64)
    parcels[1:7, 1:5] = 7
    parcels[1:7, 5:9] = 9
    parcels[8:11, 2:10] = 4
    parcels[1:5, 9:12] = 5
    mask = np.full(parcels.shape, -1, np.int64)
    mask[parcels == 7] = 1
    mask[parcels == 9] = 1
    mask[parcels == 4] = 2
    return parcels, mask


def qual(trip):
    a, b, e = trip
    return px.quality(a, b, NAMES, e)


def counts(seg, seg_cls=None):
    parcels, mask = chip()
    _, true, true_cls, void = px.truth(mask, parcels)
    seg = seg.astype(np.int32)
    cls = seg_cls if seg_cls is not None else np.ones(int(seg.max()) + 1, np.int64)
    a, b, e = px.chip_counts(true, true_cls, seg, cls, void, 3, ((mask >= 0) & (parcels > 0),
                                                                np.full(mask.shape, 10, np.uint8)))
    return a, b, e


def test_void_parcel_dropped():
    parcels, _ = chip()
    seg = np.where(parcels == 7, 1, 0) + np.where(parcels == 9, 2, 0) + np.where(parcels == 4, 3, 0)
    seg = seg + np.where(parcels == 5, 4, 0)
    cls = np.array([0, 1, 1, 2, 1])
    _, b, e = counts(seg, cls)
    assert b[0] == 3 and b[1] == 0 and b[2] == 0, b          # A, B, C found; V dropped, not an FP
    assert e[0] == 1, e


def test_background_parcel_fp():
    parcels, _ = chip()
    seg = np.where(parcels == 7, 1, 0)
    seg[11, :] = 2                                           # a strip of background
    _, b, e = counts(seg)
    assert b[0] == 1 and b[1] == 1, b                        # A found, the strip an FP
    assert b[2] == 2, b                                      # B and C missed: no parcel was built over them
    assert e[0] == 0 and e[1] == 0 and e[2] == 1, e          # the FP lies on background, WorldCover trees


def test_iou_counts_background_not_void():
    parcels, _ = chip()
    # A grown over the background column 0 (6 pixels): IoU 24 / 30 = 0.8, still a match.
    seg = np.where(parcels == 7, 1, 0)
    seg[1:7, 0] = 1
    ti, si, iou, _, _ = px.match(*_truth_seg(seg))
    assert len(ti) == 1 and abs(iou[0] - 24 / 30) < 1e-12
    # A grown over the background columns 0 and over 24 more background pixels: IoU 24 / 54 < 0.5, no match.
    seg[0, :] = 1
    seg[11, :] = 1
    ti, si, iou, _, _ = px.match(*_truth_seg(seg))
    assert len(ti) == 0, iou
    # B grown over the whole void parcel V (12 pixels): void leaves the union, IoU 1.
    seg = np.where((parcels == 9) | (parcels == 5), 1, 0)
    ti, si, iou, _, _ = px.match(*_truth_seg(seg))
    assert len(ti) == 1 and iou[0] == 1.0


def _truth_seg(seg):
    parcels, mask = chip()
    _, true, _, void = px.truth(mask, parcels)
    return true, seg.astype(np.int32), void


def test_extent_cuts_truth():
    """A true piece outside X is an FN: with X = WorldCover covering A only, B and C are missed."""
    parcels, mask = chip()
    wc = np.where(parcels == 7, 40, 10).astype(np.uint8)
    pred = mask.clip(0)
    prob = np.where(px.pieces(parcels) > 0, 0, 255).astype(np.uint8)
    parts = px.chip_parts(pred, mask, parcels, prob, wc, 3)
    q = qual(parts["worldcover/extent"])
    assert q["tp_agnostic"] == 1 and q["fn_agnostic"] == 2 and q["fp_agnostic"] == 0, q
    # The whole chip as one parcel: no match, one FP (more background than void).
    q = qual(parts["chip/extent"])
    assert q["tp_agnostic"] == 0 and q["fp_agnostic"] == 1 and q["fn_agnostic"] == 3, q


def test_oracle_extent_unchanged():
    """With X = E the void rule changes nothing: the source scorer's synthetic chip gives its numbers."""
    parcels, mask = chip()
    from delineation import boundary_map

    prob = np.where(boundary_map(parcels), 255, 0).astype(np.uint8)
    parts = px.chip_parts(mask.clip(0), mask, parcels, prob, None, 3, extents=("oracle",))
    q = qual(parts["oracle/boundary@0.5"])
    assert q["pq"] == 1.0 and q["pq_agnostic"] == 1.0, q
    s = qual(parts["oracle/semantic"])
    assert abs(s["rq_agnostic"] - 0.4) < 1e-9 and s["per_class"]["c1"]["fn"] == 2


def regression(chips: str, workers: int) -> None:
    src = REPO / "results/2026-10-18_parcel_delineation" / chips
    cells = sorted(src.glob("*/P*_draw*_seed*/results.json"))
    if not cells:
        raise SystemExit(f"no cells under {src}")
    worst = 0.0
    for f in cells:
        r = json.loads(f.read_text())
        old = r["panoptic_metrics"]
        new = px.score_predictions(f.parent / "predictions_test.npz", REPO / "data/eurocrops_chips" / chips / "chips",
                                   r["class_names"], None, extents=("oracle",), workers=workers)
        for name, v in old["variants"].items():
            got = new["variants"]["oracle" if name == "oracle" else f"oracle/{name}"]
            for k in ("pq", "sq", "rq", "pq_agnostic", "sq_agnostic", "rq_agnostic", "true_parcels",
                      "predicted_parcels"):
                d = abs(got[k] - v[k])
                worst = max(worst, d)
                if d > 1e-12:
                    raise SystemExit(f"{f.parent}: {name} {k} stored {v[k]} rescored {got[k]}")
            for c, pc in v["per_class"].items():
                if (pc is None) != (got["per_class"][c] is None) or (
                        pc is not None and any(pc[k] != got["per_class"][c][k] for k in ("tp", "fp", "fn"))):
                    raise SystemExit(f"{f.parent}: {name} {c} stored {pc} rescored {got['per_class'][c]}")
        bp = old.get("boundary_pixels")
        if bp and abs(bp["f1"] - new["boundary_pixels"]["f1"]) > 1e-12:
            raise SystemExit(f"{f.parent}: boundary F1 stored {bp['f1']} rescored {new['boundary_pixels']['f1']}")
        print(f"ok  regression {f.parent.parent.name}/{f.parent.name}", flush=True)
    print(f"regression: {len(cells)} cells reproduced, largest difference {worst:.1e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regression", metavar="CHIPS", default=None)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"ok  {name}")
    if a.regression:
        regression(a.regression, a.workers)
