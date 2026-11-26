"""Copied from experiments/2026-11-26_open_delineation and run on the 18-country sample (common.py): test
chips only, the variants as chosen there on the Estonian validation chips.

Delineate Anything (Lavreniuk et al.), zero-shot on the validation and test chips: its field instances.

Run in this experiment's own environment (``env.sh``: ``ultralytics`` brings ``opencv-python``, which must not enter
the Poetry environment)::

    env -u PYTHONPATH ~/.venvs/open_delineation/bin/python experiments/2026-11-26_open_delineation/predict_delany.py

The models: **Delineate Anything v2** (``DelineateAnythingv2.pt``, FBIS-73M, ECCV 2026) and v1
(``DelineateAnything.pt``, YOLO11x-seg, FBIS-22M), both ``ultralytics`` YOLO instance segmentation on RGB, trained on
images from 0.25 to 10 m. Each sees **one date**: one monthly composite, April to September.

Two input recipes, both published:

* ``official``: the authors' ``delineate.py`` (``methods/main``, commit 653e261): R, G, B stretched per band between
  the 1st and 99th percentiles of the positive values of each input file, averaged over the files (here: each chip of
  the split, for that month), to uint8; handed to the model in OpenCV's B, G, R order (their ``bands: [3, 2, 1]``);
  super-resolution x 2 for imagery of 4 m or coarser, by cubic resize (their 256 px Sentinel-2 tiles become 512; our
  224 px chips 448, the same scale); confidence 0.15; masks at full resolution (``retina_masks``) smoothed by their
  3 x 3 opening and closing.
* ``ftw``: ftw-tools' wrapper (``ftw_tools/inference/models.py`` ``DelineateAnything``), as Fields of The World runs
  it on Sentinel-2: R, G, B / 3000 clipped to [0, 1], bilinear resize x 2, confidence 0.05, NMS IoU 0.3, at most 100
  fields per patch, masks upsampled from the model's prototypes (no ``retina_masks``). ``check_wiring.py`` shows it
  reproduces the wrapper's fields (the same number per chip, mean IoU 0.985 to 1.000); with full-resolution masks
  they differ (0.84 to 0.92).

Each instance mask is brought back to 10 m (2 x 2 mean >= 0.5) and the instances are painted from the least to the
most confident, so that a pixel belongs to the most confident field covering it (Delineate Anything's own merging of
overlapping and seam-crossing polygons works on whole rasters; a chip is one tile, so only the overlaps remain).
The ids are ranked by confidence and stored with it, so a higher threshold is applied later by removing ids.
"""

from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import common as C

MODELS = {"v2": "delineate_anything/DelineateAnythingv2.pt", "v1": "delineate_anything/DelineateAnything.pt"}
RECIPES = {"official": {"conf": 0.15, "iou": 0.7, "max_det": 300, "retina": True},  # ultralytics' iou, max_det
           "ftw": {"conf": 0.05, "iou": 0.3, "max_det": 100, "retina": False}}
RUNS = ("v2_official", "v2_ftw", "v1_official")
SIZE = 448


def percentile_bounds(rgb: np.ndarray):
    """Per band of an ``(N, 3, H, W)`` R, G, B stack, the 1st and 99th percentiles of the positive values of each
    chip, averaged over the chips (``DataAnalyser.calcNormalizationBounds``)."""
    lo, hi = [], []
    for b in range(3):
        p = [np.percentile(z[z > 0], (1, 99)) for z in rgb[:, b] if (z > 0).any()]
        lo.append(np.mean([q[0] for q in p]))
        hi.append(np.mean([q[1] for q in p]))
    return np.array(lo, np.float32), np.array(hi, np.float32)


def official_images(chunk: np.ndarray, lo, hi) -> list[np.ndarray]:
    """uint8 B, G, R images of 448 x 448 (``DataLoaderCached``: stretch, then cubic resize)."""
    out = []
    for z in chunk:
        rgb = np.moveaxis(z, 0, -1).astype(np.float32)
        u8 = np.clip(255 * (rgb - lo) / (hi - lo), 0, 255).astype(np.uint8)
        out.append(cv2.resize(np.ascontiguousarray(u8[..., ::-1]), (SIZE, SIZE), interpolation=cv2.INTER_CUBIC))
    return out


def ftw_images(chunk: np.ndarray, dev) -> torch.Tensor:
    x = torch.from_numpy(chunk.astype(np.float32)).to(dev) / 3000.0
    return F.interpolate(x.clip(0.0, 1.0), size=(SIZE, SIZE), mode="bilinear", align_corners=False)


def smooth(m: torch.Tensor) -> torch.Tensor:
    """The authors' 3 x 3 opening then closing of the masks (``methods/main/inference.py``)."""
    m = -F.max_pool2d(-m.float(), kernel_size=3, stride=1, padding=1)
    m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    m = F.max_pool2d(m, kernel_size=3, stride=1, padding=1)
    return -F.max_pool2d(-m, kernel_size=3, stride=1, padding=1)


def paint(result, smooth_masks: bool):
    """``(224, 224)`` ids ranked by confidence (1 = most confident) and their confidences (index 0 unused)."""
    inst = np.zeros((224, 224), np.int32)
    if result.masks is None or len(result.masks) == 0:
        return inst, np.zeros(1, np.float32)
    m = result.masks.data.float()
    if smooth_masks:
        m = smooth(m[:, None])[:, 0]
    m = (F.avg_pool2d(m[:, None], 2)[:, 0] >= 0.5).cpu().numpy()              # back to 10 m
    conf = result.boxes.conf.float().cpu().numpy()
    order = np.argsort(-conf, kind="stable")                                  # order[r] = instance of rank r
    for r in range(len(order) - 1, -1, -1):                                    # least confident first
        inst[m[order[r]]] = r + 1
    return inst, np.concatenate([[0.0], conf[order]]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", default=list(RUNS))
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--months", nargs="+", type=int, default=list(C.SEASON))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    from ultralytics import YOLO, settings

    settings.update({"sync": False})
    dev = torch.device("cuda")
    for run in args.runs:
        mname, recipe = run.split("_", 1)
        model = YOLO(str(C.WEIGHTS / MODELS[mname])).to(dev)
        rp = RECIPES[recipe]
        for split in args.splits:
            ids = C.split_ids(split)[: args.limit]
            rgb_all, masks = [], []
            for _, s2, m in C.chips(ids, 32):
                rgb_all.append(s2[:, list(args.months)][:, :, C.RGB])        # (b, months, 3, H, W): R, G, B
                masks.append(m)
            rgb_all, masks = np.concatenate(rgb_all), np.concatenate(masks)
            for j, month in enumerate(args.months):
                t0 = time.time()
                rgb = rgb_all[:, j]
                lo, hi = percentile_bounds(rgb) if recipe == "official" else (None, None)
                inst = np.zeros((len(ids), 224, 224), np.int32)
                confs = []
                for s in range(0, len(ids), args.batch):
                    chunk = rgb[s:s + args.batch]
                    if recipe == "official":
                        res = model.predict(official_images(chunk, lo, hi), conf=rp["conf"], iou=rp["iou"],
                                            max_det=rp["max_det"], imgsz=SIZE, half=True,
                                            retina_masks=rp["retina"], verbose=False)
                    else:
                        res = model.predict(ftw_images(chunk, dev), conf=rp["conf"], iou=rp["iou"],
                                            max_det=rp["max_det"], imgsz=SIZE, retina_masks=rp["retina"],
                                            verbose=False)
                    for i, r in enumerate(res):
                        inst[s + i], c = paint(r, smooth_masks=recipe == "official")
                        confs.append(c)
                meta = {"model": f"delineate_anything_{mname}", "weights": MODELS[mname], "recipe": recipe,
                        "month": C.MONTHS[month], **rp, "imgsz": SIZE,
                        "bounds_p1_p99_rgb": None if lo is None else [lo.tolist(), hi.tolist()]}
                C.save_instances(f"delany_{mname}", f"{recipe}_{C.MONTHS[month]}", split, ids, inst, masks, meta,
                                 conf=confs)
                print(f"{run} {split} {C.MONTHS[month]}: {len(ids)} chips, "
                      f"{np.mean([len(c) - 1 for c in confs]):.1f} fields per chip, {time.time() - t0:.0f} s",
                      flush=True)


if __name__ == "__main__":
    main()
