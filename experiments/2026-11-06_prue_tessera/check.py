"""Readiness checks of the PRUE experiment, on the CPU, before anything is submitted.

1. **Targets**: the three classes on the chips of a chip set, their shares, and that the boundary class is exactly
   the delineation experiment's target on the labelled pixels.
2. **Loss**: log-cosh Dice is near 0 for confident right logits, larger for wrong ones, blind to ignored pixels.
3. **Fields from the truth**: class probabilities made from the test chips' own target (interior, boundary,
   background; the unknown pixels called background) are scored as the predictions will be. ``grow`` must recover
   nearly every true piece; ``cc`` loses the rim of each piece and the pieces with no interior; ``field`` is the
   floor of the predicted extent. This checks the post-processing and the scorer together.
4. **Two routes, one input**: on a chip, the composites route (``CompositeChips``, the student on the CPU, float32)
   and the raster route (``RasterChips``, the cached bf16 raster) under the same D4 turn and crop give the same
   embedding (cosine), and the same target.
5. **Forward and backward**: the U-Net (EfficientNet-B7) through both routes on 64 x 64 crops of two chips, with
   gradients of ordinary size reaching the student in the trainable one.

    poetry run python experiments/2026-11-06_prue_tessera/check.py --chips EE_2021_rev_mini
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import prue  # noqa: E402

WC = REPO / "data" / "worldcover" / "EE_2021_rev" / "worldcover.npz"


def _ok(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="EE_2021_rev_mini")
    args = ap.parse_args()

    import rasterio

    from delineation import boundary_map as bmap_delineation
    from gfm4agri.benchmark.parcel_metrics import read_parcels
    from gfm4agri.benchmark.segmentation_data import load_manifest, split_list
    from gfm4agri.pipeline.config import resolve_split

    chips = REPO / "data" / "eurocrops_chips" / args.chips
    split_dir = resolve_split(chips, None)
    man = load_manifest(chips)
    class_names = [c["name"] for c in man["classes"]]
    ids = {s: split_list(chips, s, split_dir).read_text().split() for s in ("train", "val", "test")}
    wc = np.load(WC)

    def labels(cid):
        with rasterio.open(chips / "chips" / f"{cid}.mask.tif") as m:
            mask = m.read(1).astype(np.int64)
        return mask, read_parcels(chips / "chips", cid).astype(np.int64)

    # 1. targets
    counts = np.zeros(4)
    for cid in ids["train"]:
        mask, parcels = labels(cid)
        y = prue.target(mask, parcels, wc[cid])
        counts += np.bincount(y.ravel(), minlength=4)
        lab = (mask >= 0) & (parcels > 0)
        _ok(np.array_equal(y == prue.BOUNDARY, lab & bmap_delineation(parcels)), f"{cid}: boundary = delineation target") \
            if cid == ids["train"][0] else None
    share = counts / counts.sum()
    print("     target shares on the training chips: " + ", ".join(
        f"{n} {s:.3f}" for n, s in zip((*prue.CLASSES, "ignored"), share)))
    _ok(share[prue.BOUNDARY] > 0.02 and share[prue.INTERIOR] > 0.1, "boundary and interior both present")

    # 2. loss
    y = torch.tensor(prue.target(*labels(ids["train"][0]), wc[ids["train"][0]]).astype(np.int64))[None]
    good = torch.nn.functional.one_hot(torch.where(y == 3, 0, y), 3).permute(0, 3, 1, 2).float() * 20
    bad = torch.roll(good, 1, dims=1)
    lg, lb = float(prue.log_cosh_dice(good, y)), float(prue.log_cosh_dice(bad, y))
    noisy = good.clone()
    noisy[:, :, y[0] == 3] = torch.randn(3, int((y == 3).sum()))
    ln = float(prue.log_cosh_dice(noisy, y))
    print(f"     loss: right {lg:.2e}, wrong {lb:.3f}, right with noise on ignored pixels {ln:.2e}")
    _ok(lg < 1e-6 and lb > 0.01 and abs(ln - lg) < 1e-9, "loss: right ~0, wrong > 0, ignored pixels ignored")

    # 3. fields from the truth
    import score_prue

    probs, masks = [], []
    for cid in ids["test"]:
        mask, parcels = labels(cid)
        t = prue.target(mask, parcels, wc[cid])
        t = np.where(t == prue.IGNORE, prue.BACKGROUND, t)
        probs.append((np.eye(3, dtype=np.uint8)[t] * 255).transpose(2, 0, 1))
        masks.append(mask.astype(np.int16))
    s = score_prue.score(np.stack(probs), np.stack(masks), ids["test"], chips / "chips", class_names, WC, workers=1)
    v = s["variants"]
    for k in ("chip/cc", "chip/grow", "chip/field", "oracle/grow"):
        print(f"     truth as prediction, {k}: PQ agnostic {v[k]['pq_agnostic']:.3f}, RQ {v[k]['rq_agnostic']:.3f}, "
              f"{v[k]['tp_agnostic']} TP / {v[k]['fp_agnostic']} FP / {v[k]['fn_agnostic']} FN")
    # The ceiling of the target: pieces with no interior of 5 px (median 2 px, slivers at the chip edge and
    # fragments of multipart parcels) cannot be found; on 150 test chips of EE_2021_rev grow keeps RQ 0.74.
    _ok(v["chip/grow"]["pq_agnostic"] > 0.7, "grow recovers most true pieces from a perfect map")
    _ok(v["chip/cc"]["pq_agnostic"] < v["chip/grow"]["pq_agnostic"], "cc loses the rims")
    _ok(s["pixels"]["boundary_argmax"]["f1"] > 0.999, "boundary pixel F1 of the truth is 1")

    # 4. two routes, one input
    import rasters
    import tessera_ft
    from data import CompositeChips, RasterChips

    cache = rasters.frozen(chips, split_dir, REPO / "data" / "embeddings", REPO / "data" / "embeddings", 16384)
    raster = dict(raster_dir=cache[0], suffix=cache[1], means=cache[2], stds=cache[3])
    cid = ids["train"][1]
    rc = RasterChips([cid], labels_dir=chips / "chips", worldcover=WC, augment=True, crop=32, **raster)
    cc_ = CompositeChips([cid], chips_dir=chips / "chips", band_names=man["imagery"]["band_names"],
                         n_months=man["imagery"]["n_months"], labels_dir=chips / "chips", worldcover=WC,
                         augment=True, crop=32)
    torch.manual_seed(3)
    a = rc[0]
    torch.manual_seed(3)
    b = cc_[0]
    _ok(torch.equal(a["y"], b["y"]), "same target from both routes under the same turn and crop")
    from gfm4agri.embeddings.tessera_v2 import month_doy

    student = tessera_ft.load_student().eval()
    e = tessera_ft.encode(student, tessera_ft.prepare(b["s2"], b["s1"], month_doy(man["year"], 12)), grad=False)
    e = ((e.view(32, 32, -1).permute(2, 0, 1) - torch.tensor(cache[2])[:, None, None])
         / torch.tensor(cache[3])[:, None, None])
    cos = torch.nn.functional.cosine_similarity(e.reshape(128, -1), a["image"].reshape(128, -1), dim=0)
    cos = cos.detach()
    print(f"     cosine composites route vs raster cache: median {cos.median():.5f}, min {cos.min():.5f}")
    _ok(float(cos.median()) > 0.99, "the student on the composites reproduces the cached raster")

    # 5. forward and backward through both routes
    from data import collate_raster, make_collate_composites
    from fit_prue import Net

    # Two different chips: with one chip twice, EfficientNet's 1 x 1 bottleneck has no batch variance and its batch
    # norm sends the gradients to ~1e12 (a property of the test, not of the model).
    two = [ids["train"][1], ids["train"][2]]
    rc2 = RasterChips(two, labels_dir=chips / "chips", worldcover=WC, augment=True, crop=64, **raster)
    cc2 = CompositeChips(two, chips_dir=chips / "chips", band_names=man["imagery"]["band_names"],
                         n_months=man["imagery"]["n_months"], labels_dir=chips / "chips", worldcover=WC,
                         augment=True, crop=64)
    unet = prue.build_unet(128, "efficientnet-b7", None)
    print(f"     U-Net EfficientNet-B7 parameters: {sum(p.numel() for p in unet.parameters()) / 1e6:.2f} M")
    dev = torch.device("cpu")
    for route in ("raster", "composites"):
        torch.manual_seed(0)
        if route == "raster":
            net = Net(unet, cache[2], cache[3])
            batch = collate_raster([rc2[0], rc2[1]])
        else:
            net = Net(unet, cache[2], cache[3], student=student, trainable=True, chunk=4096)
            batch = make_collate_composites(month_doy(man["year"], 12))([cc2[0], cc2[1]])
        net.train()
        t0 = time.time()
        logits = net(batch, dev)
        loss = prue.log_cosh_dice(logits, batch["y"])
        loss.backward()
        g = None if route == "raster" else (sum(float(p.grad.detach().abs().sum()) for p in student.parameters()
                                                if p.grad is not None) / sum(p.numel() for p in student.parameters()))
        print(f"     {route}: logits {tuple(logits.shape)}, loss {float(loss):.4f}, {time.time() - t0:.1f} s"
              + ("" if g is None else f", mean |grad| on the student {g:.2e}"))
        _ok(logits.shape == (2, 3, 64, 64) and torch.isfinite(loss), f"{route}: forward and backward")
        if g is not None:
            _ok(0 < g < 1e-2, "gradients reach the trainable student, of ordinary size")
        unet.zero_grad(set_to_none=True)
    print("all checks passed")


if __name__ == "__main__":
    main()
