"""Checks of the experiment's own code on synthetic chips, before any fit: the boundary target, the mask encoding,
the parcels from a boundary map and the Panoptic Quality. Run from the repository root:

    poetry run python experiments/2026-10-18_parcel_delineation/check.py
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import delineation as dl  # noqa: E402
import panoptic as pa  # noqa: E402


def chip():
    """Three parcels on a 12 x 12 chip: A (id 7, class 1) and B (id 9, class 1) touch along a column, C (id 4,
    class 2) is separated from them by non-parcel land; the rest is non-parcel land."""
    parcels = np.zeros((12, 12), np.int64)
    parcels[1:7, 1:5] = 7
    parcels[1:7, 5:9] = 9
    parcels[8:11, 2:10] = 4
    mask = np.full(parcels.shape, -1, np.int64)
    mask[parcels == 7] = 1
    mask[parcels == 9] = 1
    mask[parcels == 4] = 2
    return parcels, mask


def test_boundary_target():
    parcels, mask = chip()
    b = dl.boundary_map(parcels)
    assert not b[parcels == 0].any()
    # A's interior is 4 x 2 pixels, B's too, C's 1 x 6.
    assert (~b & (parcels == 7)).sum() == 8 and (~b & (parcels == 9)).sum() == 8
    assert (~b & (parcels == 4)).sum() == 6
    # The chip edge is no boundary.
    full = np.full((5, 5), 3)
    assert not dl.boundary_map(full).any()
    # D4 equivariance: the target of a rotated chip is the rotated target.
    for k in range(4):
        for flip in (False, True):
            r = np.rot90(parcels, k)
            r = r[:, ::-1] if flip else r
            t = np.rot90(b, k)
            t = t[:, ::-1] if flip else t
            assert (dl.boundary_map(r) == t).all()


def test_encoding():
    parcels, mask = chip()
    enc = dl.encode(mask, parcels)
    cls, bnd = dl.decode(enc)
    assert (cls == mask).all()
    assert (bnd[mask < 0] == -1).all() and (bnd[mask >= 0] == dl.boundary_map(parcels)[mask >= 0]).all()
    tc, tb = dl.decode(torch.from_numpy(enc))
    assert (tc.numpy() == cls).all() and (tb.numpy() == bnd).all()
    # Under a budget only the drawn parcel keeps its labels, its boundary against the undrawn neighbour included.
    sparse = np.where(parcels == 7, mask, -1)
    c2, b2 = dl.decode(dl.encode(sparse, parcels))
    assert (b2[parcels == 7][..., None] >= 0).all() and (b2[parcels != 7] == -1).all()
    assert b2[3, 4] == 1                                  # A's pixel touching B


def test_head():
    from terratorch.models.heads import SegmentationHead

    head = dl.BoundaryHead(SegmentationHead(16, 5, dropout=0.1))
    out = head(torch.randn(2, 16, 8, 8))
    assert out.shape == (2, 6, 8, 8)


def test_pieces():
    lab = np.array([[1, 1, 0, 1],
                    [0, 0, 0, 1],
                    [2, 0, 1, 1]])
    p = pa.pieces(lab)
    assert len(np.unique(p[p > 0])) == 3 and (p[lab == 0] == 0).all()
    assert p[0, 0] == p[0, 1] and p[0, 3] == p[2, 3] and p[0, 0] != p[0, 3]


def test_quality_perfect_and_merged():
    parcels, mask = chip()
    n = 3
    extent = mask >= 0
    # A perfect boundary map recovers the three parcels: PQ 1 class-aware and agnostic.
    prob = np.where(dl.boundary_map(parcels), 255, 0).astype(np.uint8)
    parts = pa.chip_parts(mask.clip(0), mask, parcels, prob, n)
    q = pa.quality(*parts["boundary@0.5"], ["c0", "c1", "c2"])
    assert q["pq"] == 1.0 and q["pq_agnostic"] == 1.0, q
    assert pa.quality(*parts["oracle"], ["c0", "c1", "c2"])["rq"] == 1.0
    # The crop-type map alone merges A and B, which share class 1: one false positive, two misses in class 1,
    # C found. Class-agnostic: TP 1, FP 1, FN 2, RQ = 1 / (1 + 1.5) = 0.4.
    s = pa.quality(*parts["semantic"], ["c0", "c1", "c2"])
    assert s["per_class"]["c1"]["tp"] == 0 and s["per_class"]["c1"]["fn"] == 2
    assert s["per_class"]["c2"]["pq"] == 1.0
    assert abs(s["rq_agnostic"] - 0.4) < 1e-9
    # No boundary predicted: the boundary variant is the connected extent, as semantic here.
    zero = np.zeros_like(prob)
    z = pa.chip_parts(mask.clip(0), mask, parcels, zero, n)
    assert pa.quality(*z["boundary@0.5"], ["c0", "c1", "c2"])["rq_agnostic"] == s["rq_agnostic"]
    # Every pixel of the extent belongs to exactly one predicted parcel.
    seg = pa.from_boundary(extent, prob / 255.0, 0.5)
    assert (seg[extent] > 0).all() and (seg[~extent] == 0).all()


def test_wrong_class():
    parcels, mask = chip()
    pred = np.where(parcels == 4, 1, mask.clip(0))       # C predicted as class 1
    prob = np.where(dl.boundary_map(parcels), 255, 0).astype(np.uint8)
    q = pa.quality(*pa.chip_parts(pred, mask, parcels, prob, 3)["boundary@0.5"], ["c0", "c1", "c2"])
    assert q["pq_agnostic"] == 1.0                       # the geometry is right
    assert q["per_class"]["c2"]["fn"] == 1 and q["per_class"]["c1"]["fp"] == 1


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"ok  {name}")
