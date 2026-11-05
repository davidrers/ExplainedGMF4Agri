"""Checks of this folder's code before any fit, on the CPU, from the repository root:

    poetry run python experiments/2026-11-05_paps_tessera/check.py
    poetry run python experiments/2026-11-05_paps_tessera/check.py --chips EE_2021_rev_mini    # and real chips

* the targets of a synthetic chip: one centre per parcel piece, on the piece; heatmap 1 there; zones, sizes and
  classes; unknown land; a budget that keeps one parcel;
* the zone centre extractor against a brute-force argmax per zone, zone 0 left out;
* U-Net + PaPs: a training step with zones and a backward pass, a validation step, and the pseudo-NMS giving the
  same parcels with and without pruning;
* the scorer: PaPs' parcels equal to the true pieces score PQ 1 in the oracle extent, and the same parcels shifted
  onto background are FPs;
* with ``--chips``: the targets of real chips at K = 100 % and under the 5 % draw, with WorldCover.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import targets  # noqa: E402


def chip(H=32, W=32):
    parcels = np.zeros((H, W), np.int64)
    parcels[2:12, 2:10] = 7
    parcels[2:12, 10:20] = 9
    parcels[16:28, 4:28] = 4
    parcels[2:8, 24:30] = 5            # void: declared, outside the scheme
    mask = np.full((H, W), -1, np.int64)
    mask[parcels == 7], mask[parcels == 9], mask[parcels == 4] = 1, 1, 2
    wc = np.full((H, W), 10, np.uint8)
    wc[parcels > 0] = 40
    wc[12:16, :] = 30                  # an undeclared grass strip: unknown
    return mask, parcels, wc


def test_targets():
    mask, parcels, wc = chip()
    y = targets.build(mask, parcels, wc)
    heat, inst, zone, h, w, ocls, pcls, known = np.moveaxis(y, -1, 0)
    assert inst.max() == 3 and set(np.unique(inst)) == {0, 1, 2, 3}
    centres = np.argwhere(heat == 1)
    assert len(centres) == 3
    for i, j in centres:
        k = inst[i, j]
        assert k > 0 and zone[i, j] == k, (i, j, k, zone[i, j])
    # Sizes and classes of the parcel 7 (10 x 8 box, class 1) on its zone.
    k7 = inst[5, 5]
    assert (h[zone == k7] == 10).all() and (w[zone == k7] == 8).all() and (ocls[zone == k7] == 1).all()
    # The grass strip and the void parcel on grassland would be unknown; the void parcel lies on WorldCover 40.
    assert not known[12:16, :].any() and (zone[12:16, :] == 0).all()
    assert not known[2:8, 24:30].any()
    assert known[0, 0] and known[parcels == 7].all()
    assert (pcls[parcels == 4] == 2).all() and (pcls[parcels == 5] == -1).all()
    # A budget keeping parcel 4 only: the others are unknown land, not parcels and not background.
    yk = targets.build(mask, parcels, wc, keep=np.array([4]))
    assert yk[..., 1].max() == 1 and not yk[..., 7][parcels == 7].any()
    assert (np.argwhere(yk[..., 0] == 1) == np.argwhere((yk[..., 1] == 1) & (yk[..., 0] == 1))).all()
    # D4 commutes with the target, up to the centre pixel chosen.
    for k in range(4):
        for flip in (False, True):
            m, p, c = targets.d4([mask, parcels, wc], k, flip)
            yt = targets.build(m, p, c)
            assert (yt[..., 1] > 0).sum() == (y[..., 1] > 0).sum() and len(np.argwhere(yt[..., 0] == 1)) == 3


def test_centre_extractor():
    from paps_model import CenterExtractor

    torch.manual_seed(0)
    x = torch.rand(2, 1, 16, 16)
    zones = torch.randint(0, 5, (2, 16, 16))
    m, _ = CenterExtractor()(x, zones=zones)
    for b in range(2):
        got = set(map(tuple, torch.nonzero(m[b, 0]).tolist()))
        want = set()
        for z in range(1, 5):
            sel = zones[b] == z
            if sel.any():
                v = torch.where(sel, x[b, 0], torch.tensor(-1.0))
                idx = int(torch.argmax(v.view(-1)))
                want.add((idx // 16, idx % 16))
        assert got == want, (got, want)


def _batch(n=2, H=64, W=64):
    mask, parcels, wc = chip()
    ys, xs = [], []
    for i in range(n):
        m = np.full((H, W), -1, np.int64)
        p = np.zeros((H, W), np.int64)
        c = np.full((H, W), 10, np.uint8)
        m[16 * i:16 * i + 32, 8:40], p[16 * i:16 * i + 32, 8:40], c[16 * i:16 * i + 32, 8:40] = mask, parcels, wc
        ys.append(torch.from_numpy(targets.build(m, p, c)))
        xs.append(torch.randn(8, H, W))
    return torch.stack(xs), torch.stack(ys)


def test_model_steps():
    from paps_loss import PaPsLoss
    from paps_model import build_model

    torch.manual_seed(0)
    model = build_model(8, 3)
    crit = PaPsLoss()
    x, y = _batch()
    model.train()
    p = model(x, zones=y[..., 2], pseudo_nms=False, heatmap_only=True)
    loss = crit(p, y, heatmap_only=True)
    loss.backward()
    model.zero_grad()
    p = model(x, zones=y[..., 2], pseudo_nms=False)
    assert int(p["center_mask"].sum()) == 6, int(p["center_mask"].sum())     # one centre per zone, 3 per chip
    loss = crit(p, y)
    assert torch.isfinite(loss) and all(np.isfinite(crit.value))
    loss.backward()
    assert all(q.grad is not None for n, q in model.named_parameters() if "heatmap" in n or "class_mlp" in n)
    model.eval()
    with torch.no_grad():
        model.min_confidence.fill_(0.0)
        pv = model(x, pseudo_nms=False)
        lv = crit(pv, y)
        assert torch.isfinite(lv)
        model.min_confidence.fill_(float(pv["heatmap"].median()))
        a = model(x, pseudo_nms=True)
        b = model(x, pseudo_nms=True, prune=True)
        # Pruning renumbers nothing that survives: the same parcels, the same classes.
        for i in range(x.shape[0]):
            ia, ib = a["pano_instance"][i], b["pano_instance"][i]
            assert ((ia > 0) == (ib > 0)).all()
            pairs = set(zip(ia[ia > 0].tolist(), ib[ib > 0].tolist()))
            assert len(pairs) == len({u for u, _ in pairs}) == len({v for _, v in pairs})
            assert (a["pano_semantic"][i].argmax(0)[ia > 0] == b["pano_semantic"][i].argmax(0)[ib > 0]).all()
        one = model(x[:1], pseudo_nms=True, prune=True)                       # a batch of one
        assert one["pano_instance"].shape == (1, 64, 64)
        model.min_confidence.fill_(1.1)                                       # no centre survives
        none = model(x, pseudo_nms=True, prune=True)
        assert none["pano_instance"].shape == (2, 64, 64) and not none["pano_instance"].any()
        assert torch.isfinite(crit(model(x, pseudo_nms=False, prune=True), y))


def test_scorer():
    import score_paps

    mask, parcels, wc = chip()
    import panoptic_x as px

    _, true, true_cls, _ = px.truth(mask, parcels)
    cls = np.where(true > 0, mask, 255).astype(np.uint8)
    parts = score_paps.chip_parts(true, cls, mask, parcels, wc, 3)
    a, b, e = parts["oracle/paps"]
    q = px.quality(a, b, ["c0", "c1", "c2"], e)
    assert q["pq"] == 1.0 and q["pq_agnostic"] == 1.0, q
    q = px.quality(*parts["chip/paps"][:2], ["c0", "c1", "c2"], parts["chip/paps"][2])
    assert q["pq"] == 1.0
    # The same parcels moved four rows down onto background and the grass strip: no match.
    moved = np.zeros_like(true)
    moved[4:, :] = true[:-4, :]
    moved[parcels > 0] = 0
    parts = score_paps.chip_parts(moved, np.where(moved > 0, 1, 255).astype(np.uint8), mask, parcels, wc, 3)
    a, b, e = parts["chip/paps"]
    assert b[0] == 0 and b[1] == len(np.unique(moved[moved > 0])) and b[2] == 3, b


def real(chips: str) -> None:
    import rasterio

    from gfm4agri.benchmark.segmentation_data import draw_support, split_list
    from gfm4agri.pipeline.config import resolve_split

    root = REPO / "data/eurocrops_chips" / chips
    split = resolve_split(root)
    wc = np.load(REPO / "data/worldcover/EE_2021_rev/worldcover.npz")
    keep, _ = draw_support(root, 5, 0, split_dir=split)
    ids = split_list(root, "train", split).read_text().split()[:6]
    for cid in ids:
        with rasterio.open(root / "chips" / f"{cid}.mask.tif") as m, \
             rasterio.open(root / "chips" / f"{cid}.parcels.tif") as p:
            mask, parcels = m.read(1).astype(np.int64), p.read(1).astype(np.int64)
        for name, k in (("100 %", None), ("5 %", keep)):
            y = targets.build(mask, parcels, wc[cid], k)
            n = int(y[..., 1].max())
            c = int((y[..., 0] == 1).sum())
            print(f"  {cid} {name:>5}: {n:3d} parcels, {c:3d} centres, known {y[..., 7].mean():.2f}, "
                  f"zone > 0 {(y[..., 2] > 0).mean():.2f}, background {((y[..., 7] > 0) & (y[..., 1] == 0)).mean():.2f}")
            assert c == n


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default=None)
    a = ap.parse_args()
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print(f"ok  {name}")
    if a.chips:
        real(a.chips)
        print("ok  real chips")
