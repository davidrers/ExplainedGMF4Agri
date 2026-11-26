"""Copied from experiments/2026-11-26_open_delineation and run on the 18-country sample (common.py): test
chips only, the variants as chosen there on the Estonian validation chips.

Fields of The World's released models, zero-shot on the validation and test chips: their three-class maps.

Run in ftw-tools' current stack (``env_ftw312.sh``: Python 3.12, segmentation-models-pytorch 0.6.0, kornia 0.9.2)::

    env -u PYTHONPATH /tmp/ftw312/env/bin/python experiments/2026-11-26_open_delineation/predict_ftw.py

The Poetry environment's segmentation-models-pytorch 0.5.0 builds the same EfficientNet U-Net and loads the same
weights, yet its class maps differ from ftw-tools' on 0.5 to 4 % of the pixels; with 0.6.0 they agree to the pixel
(``check_wiring.py``). The first run, in the Poetry environment, is kept in ``logs/predict_ftw_poetry_smp050.log``.

The models (``ftw-baselines`` ``ftw_tools/inference/model_registry.py``, commit eba8740):

* ``prue_b7``: **FTW v3.1, CC-BY, EfficientNet-B7**, the released PRUE (Muhawenayo et al., CVPR 2026): U-Net,
  three classes (background, field, boundary), two windows of R, G, B, NIR. Trained on the CC-BY part of Fields of
  The World, which includes Estonia 2021 (``data_config_estonia.json``, licence CC-3.0).
* ``v2_multi``: FTW v2, all data, two windows, EfficientNet-B3 (the FTW baseline of v2).
* ``v2_single``: FTW v2, all data, **one window** (four channels): the same family on one date, the temporal ablation.

The inference is ftw-tools' ``run`` (``ftw_tools/inference/inference.py``): channels R, G, B, NIR of window A then of
window B, divided by 3000 (the registry's ``normalization``), upsampled x 2 by ``kornia`` bilinear resize
(``resize_factor`` 2, the CLI default), softmax, downsampled back by nearest resize, probabilities x 255 as uint8.
Our chip is one 224 x 224 patch, so there is no tiling or padding; ``check_wiring.py`` runs ftw-tools itself on
exported chips and checks that the argmax agrees.

Windows. Fields of The World's Estonian windows are April to June (A) and July to September (B), one scene each. A
window here is one monthly composite: every pair of A in {Apr, May, Jun} and B in {Jul, Aug, Sep}, the default being
May and August (the middle of each window), and the mean of the nine pairs' probabilities (``pairs_mean``). The
single-window model takes each month from April to September and their mean (``season_mean``).
"""

from __future__ import annotations

import argparse
import itertools
import time

import numpy as np
import torch
import torch.nn.functional as F

import common as C

MODELS = {
    "prue_b7": ("ftw/prue_efnetb7_ccby_checkpoint.ckpt", 2,
                "FTW v3.1 CC-BY EfficientNet-B7 (PRUE), "
                "https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v3.1/prue_efnetb7_ccby_checkpoint.ckpt"),
    "v2_multi": ("ftw/3_Class_FULL_FTW_Pretrained_v2.ckpt", 2,
                 "FTW v2 full, two windows, https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v2/"
                 "3_Class_FULL_FTW_Pretrained_v2.ckpt"),
    "v2_single": ("ftw/3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt", 1,
                  "FTW v2 full, one window, https://github.com/fieldsoftheworld/ftw-baselines/releases/download/v2/"
                  "3_Class_FULL_FTW_Pretrained_singleWindow_v2.ckpt"),
}
DIVISOR = 3000.0
A_MONTHS, B_MONTHS = (3, 4, 5), (6, 7, 8)          # Apr-Jun, Jul-Sep (0-based)
DEFAULT_PAIR = (4, 7)                              # May, Aug


def load(path):
    """ftw-tools' ``load_model_from_checkpoint``: smp U-Net from the Lightning checkpoint's hyper-parameters."""
    import segmentation_models_pytorch as smp

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    h = ckpt["hyper_parameters"]
    assert h["model"] == "unet", h["model"]
    sd = {k.replace("model.", ""): v for k, v in ckpt["state_dict"].items()}
    for k in ("criterion.weight", "ce_loss.weight"):
        sd.pop(k, None)
    net = smp.Unet(encoder_name=h["backbone"], encoder_weights=None, in_channels=h["in_channels"],
                   classes=h["num_classes"])
    net.load_state_dict(sd)
    hp = {k: h[k] for k in ("model", "backbone", "in_channels", "num_classes") if k in h}
    return net.eval(), hp


def variants(windows: int) -> dict[str, list[tuple[int, ...]]]:
    """Variant name -> the windows whose softmax is averaged (one tuple of months per forward pass)."""
    if windows == 2:
        pairs = list(itertools.product(A_MONTHS, B_MONTHS))
        out = {f"A{C.MONTHS[a]}_B{C.MONTHS[b]}": [(a, b)] for a, b in pairs}
        out["pairs_mean"] = pairs
        return out
    out = {f"W{C.MONTHS[m]}": [(m,)] for m in C.SEASON}
    out["season_mean"] = [(m,) for m in C.SEASON]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=list(MODELS))
    ap.add_argument("--splits", nargs="+", default=["test"])
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N chips only (a smoke test)")
    args = ap.parse_args()
    import kornia.augmentation as K

    dev = torch.device("cuda")
    up = K.Resize((448, 448)).to(dev)                                   # ftw-tools: bilinear, x resize_factor 2
    down = K.Resize((224, 224), resample="nearest").to(dev)
    for name in args.models:
        rel, windows, source = MODELS[name]
        net, hp = load(C.WEIGHTS / rel)
        net = net.to(dev)
        var = variants(windows)
        passes = sorted({w for ws in var.values() for w in ws})
        for split in args.splits:
            ids = C.split_ids(split)[: args.limit]
            out = {v: np.zeros((len(ids), 3, 224, 224), np.uint8) for v in var}
            masks = np.zeros((len(ids), 224, 224), np.int16)
            t0, k = time.time(), 0
            for part, s2, m in C.chips(ids, args.batch):
                x = torch.from_numpy(s2.astype(np.float32)).to(dev) / DIVISOR     # (b, 12, 12, H, W)
                soft = {}
                with torch.inference_mode():
                    for w in passes:
                        inp = torch.cat([x[:, mo][:, C.RGBN] for mo in w], dim=1)
                        p = F.softmax(net(up(torch.nan_to_num(inp, nan=0.0))), dim=1)
                        soft[w] = down(p.float())
                    for v, ws in var.items():
                        p = torch.stack([soft[w] for w in ws]).mean(0)
                        out[v][k:k + len(part)] = (p.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
                masks[k:k + len(part)] = m
                k += len(part)
            print(f"{name} {split}: {len(ids)} chips, {len(passes)} passes, {time.time() - t0:.0f} s", flush=True)
            for v, ws in var.items():
                meta = {"model": name, "source": source, "hyper_parameters": hp, "windows": windows,
                        "variant": v, "months": [[C.MONTHS[mo] for mo in w] for w in ws],
                        "default": v == (f"A{C.MONTHS[DEFAULT_PAIR[0]]}_B{C.MONTHS[DEFAULT_PAIR[1]]}" if windows == 2
                                         else "WMay"),
                        "input": "R, G, B, B08 per window, / 3000, kornia bilinear x2, softmax, nearest back",
                        "output": "prob x 255, classes background, interior (field), boundary"}
                C.save_prob(name, v, split, ids, out[v], masks, meta)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
