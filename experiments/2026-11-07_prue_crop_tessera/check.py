"""Readiness checks of the joint experiment, on the CPU, before anything is submitted.

Default (on a pilot chip set):

1. **Crop target**: on the training chips at K = 100 %, the labelled crop pixels are exactly the in-scheme pixels of
   the drawn parcels, the same pixels PRUE's target calls interior or boundary.
2. **Forward and backward through both routes**: the joint model on 64 x 64 crops of two chips, raster and
   composites; both outputs of the right shape; the crop loss reaches the U-Net and the student, never PRUE's
   border head.

``--balance``: **how hard each loss pulls on the shared U-Net.** The ImageNet-initialised model on training batches
as in training (8 random 128 x 128 windows, D4): the norms of the border loss's and the crop loss's gradients over
the U-Net's parameters, per batch, at the start and after ``--steps`` steps of training with ``--weight``. The crop
loss weight of the arm files is set from the start's median ratio.

``--scorer`` (on ``EE_2021_rev``, needs the stored test outputs of the two frozen models on the hub): **the joint
scorer reproduces the separate models' scores.** A ``predictions_test.npz`` assembled from PRUE frozen's border
probabilities and the main workflow's frozen TESSERA crop map is scored by ``score_joint``; it must give the crop
arm's pixel and parcel Macro-F1, PRUE's class-agnostic and class-aware PQ (PRUE took its class from that same map),
and PRUE's border F1 of the border-accuracy experiment. The two stored test masks must also be the same.

    poetry run python experiments/2026-11-07_prue_crop_tessera/check.py --chips EE_2021_rev_mini
    poetry run python experiments/2026-11-07_prue_crop_tessera/check.py --chips EE_2021_rev_mini --balance --steps 40 --weight 0.1
    poetry run python experiments/2026-11-07_prue_crop_tessera/check.py --chips EE_2021_rev --scorer
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402
import torch  # noqa: E402

import joint  # noqa: E402
import prue  # noqa: E402

WC = REPO / "data" / "worldcover" / "EE_2021_rev" / "worldcover.npz"
CELL = "P100_draw0_seed0"
PRUE_FROZEN = "results/2026-11-06_prue_tessera/{set}/prue_tessera_v2_frozen"
CROP_FROZEN = "results/kshot/{set}/tessera_v2_medium"
BORDERS = "results/2026-11-07_border_accuracy/{set}/borders.json"


def _ok(cond: bool, msg: str) -> None:
    print(("ok   " if cond else "FAIL ") + msg, flush=True)
    if not cond:
        raise SystemExit(1)


def _setup(chips_name: str):
    from gfm4agri.benchmark.segmentation_data import draw_support, load_manifest, split_list
    from gfm4agri.pipeline.config import resolve_split

    chips = REPO / "data" / "eurocrops_chips" / chips_name
    split_dir = resolve_split(chips, None)
    man = load_manifest(chips)
    ids = {s: split_list(chips, s, split_dir).read_text().split() for s in ("train", "val", "test")}
    keep, _ = draw_support(chips, 100, 0, split_dir=split_dir)
    return chips, split_dir, man, ids, keep


def _raster(chips, split_dir):
    import rasters

    cache = rasters.frozen(chips, split_dir, REPO / "data" / "embeddings", REPO / "data" / "embeddings", 16384)
    return cache, dict(raster_dir=cache[0], suffix=cache[1], means=cache[2], stds=cache[3])


def _model(n_in: int, n_classes: int, weights, means, stds, student=None):
    unet = prue.build_unet(n_in, "efficientnet-b7", weights)
    head = joint.CropHead(n_in + joint.decoder_channels(unet), n_classes)
    return joint.JointNet(unet, head, means, stds, student=student, trainable=student is not None, chunk=4096)


def basic(chips_name: str) -> None:
    import rasterio

    from data import CompositeChips, RasterChips, collate_raster, make_collate_composites
    from gfm4agri.benchmark.parcel_metrics import read_parcels
    from gfm4agri.embeddings.tessera_v2 import month_doy

    chips, split_dir, man, ids, keep = _setup(chips_name)
    n_classes = len(man["classes"])
    wc = np.load(WC)

    # 1. crop target
    n_lab = n_drawn = 0
    for cid in ids["train"]:
        with rasterio.open(chips / "chips" / f"{cid}.mask.tif") as m:
            mask = m.read(1).astype(np.int64)
        parcels = read_parcels(chips / "chips", cid).astype(np.int64)
        yc = joint.crop_target(mask, parcels, keep)
        y = prue.target(mask, parcels, wc[cid], keep)
        drawn = (mask >= 0) & np.isin(parcels, keep) & (parcels > 0)
        _ok(np.array_equal(yc >= 0, drawn) and np.array_equal(yc[drawn], mask[drawn]),
            f"{cid}: crop target = class of the drawn parcels' pixels") if cid == ids["train"][0] else None
        if not np.array_equal(yc >= 0, (y == prue.INTERIOR) | (y == prue.BOUNDARY)):
            _ok(False, f"{cid}: crop-labelled pixels differ from PRUE's interior + boundary")
        n_lab += int((yc >= 0).sum())
        n_drawn += int(drawn.sum())
    _ok(n_lab == n_drawn and n_lab > 0, f"crop target on {len(ids['train'])} training chips: {n_lab} labelled "
        "pixels, the same as PRUE's interior + boundary")

    # 2. forward and backward through both routes
    import tessera_ft

    cache, raster = _raster(chips, split_dir)
    two = [ids["train"][1], ids["train"][2]]
    rc2 = RasterChips(two, labels_dir=chips / "chips", worldcover=WC, keep=keep, augment=True, crop=64, **raster)
    cc2 = CompositeChips(two, chips_dir=chips / "chips", band_names=man["imagery"]["band_names"],
                         n_months=man["imagery"]["n_months"], labels_dir=chips / "chips", worldcover=WC, keep=keep,
                         augment=True, crop=64)
    student = tessera_ft.load_student().eval()
    dev = torch.device("cpu")
    for route in ("raster", "composites"):
        torch.manual_seed(0)
        if route == "raster":
            net = _model(128, n_classes, None, cache[2], cache[3])
            batch = collate_raster([rc2[0], rc2[1]])
        else:
            net = _model(128, n_classes, None, cache[2], cache[3], student=student)
            batch = make_collate_composites(month_doy(man["year"], 12))([cc2[0], cc2[1]])
        net.train()
        t0 = time.time()
        seg, crop = net(batch, dev)
        _ok(seg.shape == (2, 3, 64, 64) and crop.shape == (2, n_classes, 64, 64),
            f"{route}: border logits {tuple(seg.shape)}, crop logits {tuple(crop.shape)}")
        lc = joint.crop_ce(crop, batch["yc"])
        lc.backward()
        head_g = [p.grad for p in net.unet.segmentation_head.parameters()]
        unet_g = sum(float(p.grad.abs().sum()) for p in net.unet.decoder.parameters() if p.grad is not None)
        _ok(all(g is None or float(g.abs().sum()) == 0 for g in head_g) and unet_g > 0,
            f"{route}: the crop loss ({float(lc):.3f}) reaches the U-Net decoder, not PRUE's border head")
        if route == "composites":
            g = sum(float(p.grad.abs().sum()) for p in student.parameters() if p.grad is not None)
            _ok(g > 0, "composites: the crop loss reaches the trainable student")
            student.zero_grad(set_to_none=True)
        net.zero_grad(set_to_none=True)
        total, lb, lc = joint.losses(seg, crop, batch["y"], batch["yc"], prue.CLASS_WEIGHTS, 0.1)
        _ok(bool(torch.isfinite(total)) and abs(float(total) - float(lb) - 0.1 * float(lc)) < 1e-5,
            f"{route}: total loss {float(total):.4f} = border {float(lb):.4f} + 0.1 x crop {float(lc):.4f}, "
            f"{time.time() - t0:.1f} s")
    empty = joint.crop_ce(torch.zeros(1, n_classes, 4, 4, requires_grad=True), torch.full((1, 4, 4), -1))
    _ok(float(empty) == 0.0 and empty.requires_grad, "a batch with no labelled crop pixel gives crop loss 0")
    print("all basic checks passed")


def balance(chips_name: str, n_batches: int, steps: int, weight: float) -> None:
    from data import RasterChips, collate_raster

    chips, split_dir, man, ids, keep = _setup(chips_name)
    cache, raster = _raster(chips, split_dir)
    torch.manual_seed(0)
    ds = RasterChips(ids["train"], labels_dir=chips / "chips", worldcover=WC, keep=keep, augment=True, crop=128,
                     **raster)
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True, drop_last=True, collate_fn=collate_raster,
                                         num_workers=4)
    net = _model(128, len(man["classes"]), "imagenet", cache[2], cache[3])
    dev = torch.device("cpu")

    def measure(tag):
        net.train()
        r = []
        while len(r) < n_batches:
            for batch in loader:
                seg, crop = net(batch, dev)
                gb, gc = joint.shared_grad_norms(net, seg, crop, batch["y"], batch["yc"], prue.CLASS_WEIGHTS)
                r.append((gb, gc))
                if len(r) == n_batches:
                    break
        a = np.array(r)
        ratio = a[:, 1] / a[:, 0]
        print(f"     {tag}: |grad border| median {np.median(a[:, 0]):.3e}, |grad crop| median {np.median(a[:, 1]):.3e},"
              f" crop / border median {np.median(ratio):.2f} (range {ratio.min():.2f} to {ratio.max():.2f})",
              flush=True)
        return float(np.median(ratio))

    t0 = time.time()
    r0 = measure("at the start")
    print(f"     a weight of {1 / r0:.3f} makes the crop loss pull on the U-Net as hard as the border loss "
          f"({time.time() - t0:.0f} s)")
    if steps:
        opt = torch.optim.AdamW([{"params": net.unet.parameters(), "lr": 3e-3, "weight_decay": 0.01,
                                  "amsgrad": True},
                                 {"params": net.crop_head.parameters(), "lr": 1e-3, "weight_decay": 0.05}])
        done = 0
        while done < steps:
            for batch in loader:
                net.train()
                opt.zero_grad(set_to_none=True)
                seg, crop = net(batch, dev)
                loss, lb, lc = joint.losses(seg, crop, batch["y"], batch["yc"], prue.CLASS_WEIGHTS, weight)
                loss.backward()
                opt.step()
                done += 1
                if done % 10 == 0:
                    print(f"     step {done}: border {float(lb):.5f}, crop {float(lc):.4f}", flush=True)
                if done == steps:
                    break
        r1 = measure(f"after {steps} steps at weight {weight:g}")
        print(f"     there the weighted pull of the crop loss is {weight * r1:.2f} x the border loss's")


def scorer(chips_name: str, workers: int, tmp_dir: str | None = None) -> None:
    import score_joint

    chips = REPO / "data" / "eurocrops_chips" / chips_name
    pr = REPO / PRUE_FROZEN.format(set=chips_name) / CELL
    cr = REPO / CROP_FROZEN.format(set=chips_name) / CELL
    with np.load(pr / "predictions_test.npz") as a, np.load(cr / "predictions_test.npz") as b:
        ids_a, ids_b = [str(c) for c in a["ids"]], [str(c) for c in b["ids"]]
        order = {c: i for i, c in enumerate(ids_b)}
        _ok(set(ids_a) == set(ids_b), f"the two frozen models predicted the same {len(ids_a)} test chips")
        idx = np.array([order[c] for c in ids_a])
        prob, mask_a = a["prob"], a["mask"]
        pred, mask_b = b["pred"][idx], b["mask"][idx]
    same = np.array_equal(mask_a.astype(np.int64), mask_b.astype(np.int64))
    _ok(same, "the test masks of PRUE and of the main workflow's crop arm are the same")
    import tempfile

    out = Path(tempfile.mkdtemp(prefix="joint_check_", dir=tmp_dir)) / "predictions_test.npz"
    np.savez_compressed(out, prob=prob, pred=pred.astype(np.uint8), mask=mask_a, ids=np.array(ids_a))
    t0 = time.time()
    names = json.loads((pr / "results.json").read_text())["class_names"]
    s = score_joint.score_file(out, chips / "chips", names, WC, workers=workers)
    print(f"     scored in {time.time() - t0:.0f} s ({out})")
    kshot = json.loads((cr / "results.json").read_text())
    prue_r = json.loads((pr / "results.json").read_text())["panoptic_metrics"]["variants"]["chip/grow"]
    bnd = json.loads((REPO / BORDERS.format(set=chips_name)).read_text())["methods"]["PRUE frozen"]["d1"]
    rows = [("crop pixel Macro-F1", s["crop"]["pixel"]["macro_f1"], kshot["test_metrics"]["test/F1_Score"], 1e-5),
            ("crop parcel Macro-F1", s["crop"]["parcel"]["macro_f1"], kshot["parcel_metrics"]["macro_f1"], 1e-9),
            ("PQ agnostic, grow", s["panoptic"]["variants"]["chip/grow"]["pq_agnostic"], prue_r["pq_agnostic"], 1e-9),
            ("PQ class-aware, grow", s["panoptic"]["variants"]["chip/grow"]["pq"], prue_r["pq"], 1e-9),
            ("border F1 at 1 px", s["borders"]["d1"]["f1"], bnd["f1"], 1e-9)]
    for name, got, want, tol in rows:
        _ok(abs(got - want) <= tol, f"{name}: {got:.6f}, published {want:.6f}")
    print("the joint scorer reproduces both separate models")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", default="EE_2021_rev_mini")
    ap.add_argument("--balance", action="store_true")
    ap.add_argument("--batches", type=int, default=16)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--weight", type=float, default=0.1)
    ap.add_argument("--scorer", action="store_true")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--tmp", default=None, help="--scorer: where the assembled file goes (default: the system's)")
    a = ap.parse_args()
    if a.scorer:
        scorer(a.chips, a.workers, a.tmp)
    elif a.balance:
        balance(a.chips, a.batches, a.steps, a.weight)
    else:
        basic(a.chips)


if __name__ == "__main__":
    main()
