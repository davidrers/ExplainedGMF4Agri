"""Does ``predict_arms.py``'s path reproduce each arm's own Estonian test predictions? ``reference/arms_ee_check.json``.

Run in the Poetry environment::

    poetry run python experiments/2026-11-26_delineation_countries/check_arms_ee.py [--arms ...]

The same steps as ``predict_arms.py`` (the core's input builders, the cell's network rebuilt from its
``results.json``, its ``best.pt``, bf16 softmax of the border head), applied to the nine test chips of the Estonian
pilot ``EE_2021_rev_mini``, which are test chips of ``EE_2021_rev``. The token and raster caches are computed afresh
into a scratch directory, as they were for the sample, so the encoding is checked as well as the decoder. Each
arm's border probabilities are compared with those stored by its Estonian fit
(``results/2026-11-09_borders_all_arms/EE_2021_rev/<arm>/P100_draw0_seed0/predictions_test.npz``): mean absolute
difference (0 to 255) and the share of pixels whose most likely class agrees.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import common as C
from predict_arms import ARMS, TESSERA_EE

MINI = C.ROOT / "data" / "eurocrops_chips" / "EE_2021_rev_mini"
MINI_SPLIT = MINI / "splits" / "blocks4_buf1600_seed0__6b0eb4cb"
WC_EE = C.ROOT / "data" / "worldcover" / "EE_2021_rev" / "worldcover.npz"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=list(ARMS))
    ap.add_argument("--scratch", required=True, help="empty directory for the fresh caches")
    args = ap.parse_args()
    from gfm4agri.borders import fit, runner
    from gfm4agri.pipeline.config import CONFIGS, load_machine

    _, arms = runner.load(C.ROOT / "configs/experiments/kshot_borders.yaml")
    mach = load_machine("hub", CONFIGS)
    scratch = Path(args.scratch)
    dev = torch.device("cuda")
    ids = [line.strip() for line in open(MINI_SPLIT / "test_data.txt") if line.strip()]
    out_path = C.HERE / "reference" / "arms_ee_check.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in args.arms:
        cell = C.ARMS_RESULTS / name / "P100_draw0_seed0"
        rec = json.loads((cell / "results.json").read_text())
        # the arm's cache_gb is sized for all of Estonia; the pilot's cache is about 1 GB
        src = runner.source_for({**arms[name], "cache_gb": 0}, MINI, MINI_SPLIT, scratch, scratch, mach, "hub")
        if name == "tessera_v2_medium":
            ee = json.loads(TESSERA_EE.read_text())["normalisation"]
            src = {**src, "means": ee["means"], "stds": ee["stds"]}
        net = fit.build_model(src, len(rec["class_names"]), rec["class_names"], rec["config"])
        net.load_state_dict(torch.load(cell / "best.pt", map_location="cpu", weights_only=False)["net"])
        net = net.to(dev).eval()
        _, _, test = fit.datasets(src, {"train": [], "val": [], "test": ids}, None, MINI / "chips", WC_EE,
                                  augment=False)
        probs, got = [], []
        with torch.no_grad():
            for batch in fit._loader(test, 4, False, 4):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    seg, _ = net(batch["image"].to(dev))
                probs.append(torch.round(torch.softmax(seg.float(), 1) * 255).to(torch.uint8).cpu().numpy())
                got += batch["chip"]
        new = np.concatenate(probs).astype(np.int16)
        z = np.load(cell / "predictions_test.npz")
        pos = {str(c): i for i, c in enumerate(z["ids"])}
        old = z["prob"][[pos[c] for c in got]].astype(np.int16)
        r = {"chips": len(got), "mean_abs_diff_0_255": float(np.abs(new - old).mean()),
             "argmax_agreement": float((new.argmax(1) == old.argmax(1)).mean()),
             "boundary_share_new": float((new.argmax(1) == 2).mean()),
             "boundary_share_stored": float((old.argmax(1) == 2).mean())}
        out[name] = r
        print(name, r, flush=True)
        out_path.write_text(json.dumps(out, indent=1))
        del net
        torch.cuda.empty_cache()


if __name__ == "__main__":
    sys.exit(main())
