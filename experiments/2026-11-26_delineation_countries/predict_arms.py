"""Our two-head arms, trained on Estonia 2021 with all labels, applied unchanged to the sample: their border maps.

Run in the Poetry environment, after ``sample.py`` and ``worldcover.py``::

    poetry run python experiments/2026-11-26_delineation_countries/predict_arms.py [--arms ...]

Each arm's checkpoint is the one its Estonian test scores came from (``results/2026-11-09_borders_all_arms/
EE_2021_rev/<arm>/P100_draw0_seed0/best.pt``, the best validation epoch). The network is rebuilt from that cell's
own ``results.json`` (its configuration and input layout) by the core's ``gfm4agri.borders.fit.build_model``, its
weights loaded, and only the border head's softmax is kept (the crop head speaks Estonia's 20 classes). Nothing is
trained: for the arms this is a cross-country transfer, the open models' situation.

The inputs come from the core's own builders on the sample chip set, read-only:

* the PRUE baseline (``unet_b7_s2s1``): the twelve S2 and S1 composites, standardised with the Estonian training
  statistics it was trained under (``EU18_2021_delin``'s manifest and ``s1_rtc.json`` carry them, ``sample.py``);
* TESSERA v2 Medium: the core's raster cache (``build_raster_cache``) computed on the sample, standardised with the
  Estonian statistics (``results/kshot/EE_2021_rev/tessera_v2_medium/P100_draw0_seed0/results.json``), not the
  sample's own;
* TerraMind, THOR, Prithvi: the core's token cache (``generate_embeddings``) computed on the sample under each
  encoder's pretraining statistics, as in training; the cache's layer layout must equal the cell's.

AlphaEarth is left out: its embeddings were never fetched for these countries (the transfer plan leaves it out too).
Saved as ``arm_<arm>/P100/predictions_test.npz`` with the open models' predictions, scored by ``score.py``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import common as C

ARMS = ("unet_b7_s2s1", "tessera_v2_medium", "prithvi_eo_v2_600_tl", "terramind_v1_large_s2s1",
        "thor_v1_large_s2s1")
TESSERA_EE = C.ROOT / "results/kshot/EE_2021_rev/tessera_v2_medium/P100_draw0_seed0/results.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    from gfm4agri.borders import fit, runner
    from gfm4agri.pipeline.config import CONFIGS, load_machine

    _, arms = runner.load(C.ROOT / "configs/experiments/kshot_borders.yaml")
    mach = load_machine("hub", CONFIGS)
    cache_root = Path(mach["cache_root"])
    cache_root = cache_root if cache_root.is_absolute() else C.ROOT / cache_root
    dev = torch.device("cuda")
    ids = C.split_ids("test")
    for name in args.arms:
        cell = C.ARMS_RESULTS / name / "P100_draw0_seed0"
        rec = json.loads((cell / "results.json").read_text())
        t0 = time.time()
        src = runner.source_for(arms[name], C.CHIPS, C.SPLIT, cache_root, cache_root, mach, "hub")
        if name == "tessera_v2_medium":
            ee = json.loads(TESSERA_EE.read_text())["normalisation"]
            src = {**src, "means": ee["means"], "stds": ee["stds"], "stats": str(TESSERA_EE.relative_to(C.ROOT))}
        if src["kind"] == "token":
            for k in ("channel_list", "bands", "n_timesteps"):
                if src[k] != rec["input"][k]:
                    raise SystemExit(f"{name}: the sample cache's {k} {src[k]} differs from the cell's "
                                     f"{rec['input'][k]}")
        t_src = time.time() - t0
        net = fit.build_model(src, len(rec["class_names"]), rec["class_names"], rec["config"])
        net.load_state_dict(torch.load(cell / "best.pt", map_location="cpu", weights_only=False)["net"])
        net = net.to(dev).eval()
        _, _, test = fit.datasets(src, {"train": [], "val": [], "test": ids}, None, C.CHIP_DIR, C.WORLDCOVER,
                                  augment=False)
        loader = fit._loader(test, args.batch, False, args.workers)
        probs, masks, got = [], [], []
        with torch.no_grad():
            for batch in loader:
                x = batch["image"].to(dev, non_blocking=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    seg, _ = net(x)
                probs.append(torch.round(torch.softmax(seg.float(), 1) * 255).to(torch.uint8).cpu().numpy())
                masks.append(batch["mask"].numpy())
                got += batch["chip"]
        assert got == ids
        meta = {"arm": name, "checkpoint": str((cell / "best.pt").relative_to(C.ROOT)),
                "best_epoch": json.loads((cell / "results.json").read_text()).get("best_epoch"),
                "trained_on": "EE_2021_rev, K = 100 %, draw 0, seed 0 (two-head workflow)",
                "input": {k: v for k, v in src.items() if k not in ("means", "stds", "s2_means", "s2_stds",
                                                                     "s1_means", "s1_stds")},
                "seconds_inputs": round(t_src, 1), "seconds_predict": round(time.time() - t0 - t_src, 1)}
        C.save_prob(f"arm_{name}", "P100", "test", ids, np.concatenate(probs), np.concatenate(masks), meta)
        print(f"{name}: {len(ids)} chips, inputs {t_src:.0f} s, predict {time.time() - t0 - t_src:.0f} s",
              flush=True)
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
