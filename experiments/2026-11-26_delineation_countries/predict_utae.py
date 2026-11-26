"""Copied from experiments/2026-11-26_open_delineation and run on the 18-country sample (common.py): test
chips only, the variants as chosen there on the Estonian validation chips.

U-TAE + PaPs (Garnot and Landrieu, ICCV 2021) with the authors' PASTIS weights, zero-shot: its panoptic parcels.

Run in the Poetry environment::

    poetry run python experiments/2026-11-26_open_delineation/predict_utae.py --stage select   # folds x norms, val
    poetry run python experiments/2026-11-26_open_delineation/predict_utae.py --stage final --fold F --norm N

The one released panoptic model that reads a whole Sentinel-2 time series: a U-Net whose skips are pooled over time
by attention (L-TAE), and Parcels-as-Points instances. Trained on PASTIS (2,433 patches of 128 x 128 in France,
September 2018 to November 2019, 38 to 61 acquisitions each, 10 bands, cloudy dates kept), five folds, weights
``data/external_models/utae_paps/UTAE_PAPs/Fold_k/model.pth.tar`` with ``conf.json`` (``vendor/utae_paps``).

Input here: the chips' monthly composites, 10 bands in PASTIS' order (B02, B03, B04, B05, B06, B07, B08, B8A, B11,
B12), standardised per band by either

* ``pastis``: the mean of the five folds' statistics of ``NORM_S2_patch.json`` (read from the PASTIS archive on
  Zenodo, record 5012942; PASTIS' own test code standardises each fold by that fold's statistics), or
* ``estonia``: the chip set's own statistics (``manifest.json`` ``normalisation``: 200 chips, all months pooled),
  the counterpart of PASTIS standardising its test fold by its own statistics.

Each composite is dated the 15th of its month, in days from 2020-09-01, as PASTIS dates its acquisitions from
2018-09-01, so that the calendar is the one the model was trained on. The months fed (``--months``): all twelve,
April to September, or May and August only, the temporal ablation.

Output: PaPs' panoptic instances with its defaults (``conf.json``: confidence 0.2, mask threshold 0.4, minimum
remaining share 0.5). An instance whose class is PASTIS' background (0) or void (19) is dropped, as PASTIS' panoptic
quality ignores those classes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from types import SimpleNamespace

import numpy as np
import torch

import common as C
from vendor.utae_paps import paps, utae

W = C.WEIGHTS / "utae_paps" / "UTAE_PAPs"
NORM_PASTIS = C.HERE / "reference" / "PASTIS_NORM_S2_patch.json"
MONTH_SETS = {"all12": tuple(range(12)), "season6": C.SEASON, "pair2": (4, 7)}
DROP_CLASSES = (0, 19)


def build(fold: int, dev):
    cfg = SimpleNamespace(**json.loads((W / "conf.json").read_text()))
    enc = utae.UTAE(input_dim=10, encoder_widths=cfg.encoder_widths, decoder_widths=cfg.decoder_widths,
                    out_conv=cfg.out_conv, str_conv_k=cfg.str_conv_k, str_conv_s=cfg.str_conv_s,
                    str_conv_p=cfg.str_conv_p, agg_mode=cfg.agg_mode, encoder_norm=cfg.encoder_norm,
                    n_head=cfg.n_head, d_model=cfg.d_model, d_k=cfg.d_k, encoder=True, return_maps=False,
                    pad_value=cfg.pad_value, padding_mode=cfg.padding_mode)
    net = paps.PaPs(encoder=enc, num_classes=cfg.num_classes, shape_size=cfg.shape_size, mask_conv=cfg.mask_conv,
                    min_confidence=cfg.min_confidence, min_remain=cfg.min_remain, mask_threshold=cfg.mask_threshold)
    sd = torch.load(W / f"Fold_{fold}" / "model.pth.tar", map_location="cpu", weights_only=False)["state_dict"]
    net.load_state_dict(sd)
    return net.to(dev).eval(), cfg


def norm_stats(name: str):
    if name == "pastis":
        d = json.loads(NORM_PASTIS.read_text())
        mean = np.mean([d[f"Fold_{f}"]["mean"] for f in range(1, 6)], axis=0)
        std = np.mean([d[f"Fold_{f}"]["std"] for f in range(1, 6)], axis=0)
    else:
        n = json.loads((C.CHIPS / "manifest.json").read_text())["normalisation"]
        mean, std = np.array(n["means"])[C.PASTIS10], np.array(n["stds"])[C.PASTIS10]
    return np.asarray(mean, np.float32), np.asarray(std, np.float32)


def dates(months) -> np.ndarray:
    ref = dt.date(2020, 9, 1)
    return np.array([(dt.date(2021, m + 1, 15) - ref).days for m in months], np.float32)


def predict(net, ids, months, norm, dev, batch=4):
    mean, std = (torch.from_numpy(a).to(dev)[None, None, :, None, None] for a in norm)
    pos = torch.from_numpy(dates(months)).to(dev)
    inst = np.zeros((len(ids), 224, 224), np.int32)
    masks = np.zeros((len(ids), 224, 224), np.int16)
    k = 0
    for part, s2, m in C.chips(ids, batch):
        x = torch.from_numpy(s2[:, list(months)][:, :, C.PASTIS10].astype(np.float32)).to(dev)
        x = ((x - mean) / std).contiguous()
        n = len(part)
        if n == 1:                                   # PaPs squeezes a batch of one: predict it twice
            x = torch.cat([x, x])
        with torch.inference_mode():
            p = net(x, batch_positions=pos[None].expand(len(x), -1), pseudo_nms=True, prune=True)
        pi = p["pano_instance"][:n].long().cpu().numpy()
        cls = p["pano_semantic"][:n].argmax(1).cpu().numpy()
        for i in range(n):
            ids_i = pi[i]
            drop = np.zeros(int(ids_i.max()) + 1, bool)
            for j in np.unique(ids_i[ids_i > 0]):
                drop[j] = np.isin(np.bincount(cls[i][ids_i == j]).argmax(), DROP_CLASSES)
            inst[k + i] = np.where(drop[ids_i], 0, ids_i)
        masks[k:k + n] = m
        k += n
    return inst, masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["select", "final"], required=True)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--norm", choices=["pastis", "estonia"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    dev = torch.device("cuda")
    if args.stage == "select":                       # every fold and norm, all twelve months, validation only
        runs = [(f, nm, "all12", ["val"]) for f in range(1, 6) for nm in ("pastis", "estonia")]
    else:
        runs = [(args.fold, args.norm, ms, ["test"]) for ms in MONTH_SETS]
    for fold, nm, ms, splits in runs:
        net, cfg = build(fold, dev)
        for split in splits:
            ids = C.split_ids(split)[: args.limit]
            t0 = time.time()
            inst, masks = predict(net, ids, MONTH_SETS[ms], norm_stats(nm), dev)
            print(f"fold {fold} {nm} {ms} {split}: {len(ids)} chips, {time.time() - t0:.0f} s", flush=True)
            meta = {"model": "utae_paps", "source": "https://zenodo.org/record/5172301, Fold_%d" % fold,
                    "fold": fold, "norm": nm, "months": [C.MONTHS[m] for m in MONTH_SETS[ms]],
                    "dates_days_from_2020-09-01": dates(MONTH_SETS[ms]).tolist(),
                    "paps": {"min_confidence": cfg.min_confidence, "mask_threshold": cfg.mask_threshold,
                             "min_remain": cfg.min_remain},
                    "dropped_classes": "instances of PASTIS class 0 (background) or 19 (void)"}
            C.save_instances("utae_paps", f"fold{fold}_{nm}_{ms}", split, ids, inst, masks, meta)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
