"""Proof that the core's two-head workflow (``gfm4agri.borders``) reproduces this experiment, so its results stand.

Moved into the core on 11 November 2026 (``docs/superpowers/specs/2026-11-11-kshot-borders-design.md``, section 5).
Two checks, each exiting 1 on any difference:

* ``--scoring``: the core's ``gfm4agri.borders.scoring.score_file`` rescores every cell of
  ``results/2026-11-09_borders_all_arms/<set>/`` from its stored ``predictions_test.npz`` (in memory and a temporary
  file, nothing written into the results), with the 16 workers the cluster scored with, so the IoU sums are added in
  the same order; every crop, field (extent ``chip``), pixel and border number, and the per-chip border counts, must
  equal the cell's ``results.json`` and ``borders_per_chip.npz`` exactly.
* ``--training``: for one arm per input, this folder's ``run.source_for`` + ``fit_borders.fit`` and the core's
  ``runner.source_for`` + ``fit.fit`` train the same cell (K = 100 %, draw 0, seed 0, two epochs, CPU, no loader
  workers) on a pilot set into a scratch directory; the histories (every key) and the checkpoint weights must be
  identical.

    poetry run python experiments/2026-11-09_borders_all_arms/check_core.py --scoring
    poetry run python experiments/2026-11-09_borders_all_arms/check_core.py --training --chips EE_2021_rev_mini
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402

RESULTS = REPO / "results" / "2026-11-09_borders_all_arms"
SCORING_WORKERS = 16          # configs/machines/cluster.yaml num_workers, as min(num_workers, 16) in run.py
TRAINING_ARMS = ("tessera_v2_medium", "unet_b7_s2s1", "terramind_v1_large_s2s1")


def diff(a, b, path: str = "") -> list[str]:
    """Every place where ``a`` and ``b`` differ, exact for numbers."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = [f"{path}/{k}: only in one" for k in set(a) ^ set(b)]
        for k in set(a) & set(b):
            out += diff(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            return [f"{path}: length {len(a)} != {len(b)}"]
        return [d for i, (x, y) in enumerate(zip(a, b)) for d in diff(x, y, f"{path}[{i}]")]
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return []
    return [] if a == b else [f"{path}: {a!r} != {b!r}"]


def _plain(x):
    return json.loads(json.dumps(x, default=str))


def check_scoring(chips: str) -> list[str]:
    from gfm4agri.borders import scoring

    chips_dir = REPO / "data" / "eurocrops_chips" / chips / "chips"
    worldcover = REPO / "data" / "worldcover" / chips / "worldcover.npz"
    fails = []
    cells = sorted((RESULTS / chips).glob("*/P*_draw*_seed*/results.json"))
    for f in cells:
        cell = f.parent
        name = f"{cell.parent.name}/{cell.name}"
        r = json.loads(f.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            per_chip = Path(tmp) / "per_chip.npz"
            s = _plain(scoring.score_file(cell / "predictions_test.npz", chips_dir, r["class_names"], worldcover,
                                          workers=SCORING_WORKERS, per_chip_out=per_chip))
            with np.load(per_chip) as new, np.load(cell / "borders_per_chip.npz") as old:
                same_chips = (np.array_equal(new["ids"], old["ids"]) and np.array_equal(new["counts"], old["counts"]))
        d = diff(s["crop"]["pixel"], r["crop_metrics"], "crop_metrics")
        d += diff(s["crop"]["parcel"], r["parcel_metrics"], "parcel_metrics")
        for v in ("cc", "grow", "field"):
            d += diff(s["panoptic"]["variants"][f"chip/{v}"], r["panoptic_metrics"]["variants"][f"chip/{v}"],
                      f"panoptic/chip/{v}")
        if sorted(s["panoptic"]["variants"]) != ["chip/cc", "chip/field", "chip/grow"]:
            d.append(f"panoptic variants {sorted(s['panoptic']['variants'])}")
        d += diff(s["panoptic"]["pixels"], r["panoptic_metrics"]["pixels"], "panoptic/pixels")
        d += diff(s["borders"], r["border_metrics"], "border_metrics")
        if not same_chips:
            d.append("borders_per_chip.npz differs")
        print(("  ok    " if not d else "  FAIL  ") + name + ("" if not d else f": {len(d)} differences, e.g. "
                                                            + "; ".join(sorted(d)[:3])), flush=True)
        fails += [f"{name} {x}" for x in d]
    if not cells:
        fails.append(f"no cell under {RESULTS / chips}")
    return fails


def check_training(chips_name: str, arms: list[str], scratch: Path) -> list[str]:
    import torch

    import fit_borders
    import run as exp_run
    from gfm4agri.borders import fit as core_fit
    from gfm4agri.borders import runner
    from gfm4agri.pipeline.config import CONFIGS, load_machine, resolve_split

    chips = REPO / "data" / "eurocrops_chips" / chips_name
    split_dir = resolve_split(chips)
    mach = load_machine("hub")
    root = REPO / mach["cache_root"]
    exp_exp, exp_arms = exp_run.load(HERE / "experiment.yaml")
    core_exp, core_arms = runner.load(CONFIGS / "experiments" / "kshot_borders.yaml")
    worldcover = runner.worldcover_for(chips)
    fails = []
    for arm in arms:
        print(f"== {arm}", flush=True)
        src_e = exp_run.source_for(exp_arms[arm], chips, split_dir, root, root, mach, "hub")
        src_c = runner.source_for(core_arms[arm], chips, split_dir, root, root, mach, "hub")
        d = diff(_plain(src_e), _plain(src_c), "source")
        d = [x for x in d if not x.startswith("source/record_created")]
        cfg_e = exp_run.cell_config(arm, exp_arms[arm], {**exp_exp, "epochs": 2}, chips, split_dir, 100, 0, 0,
                                    scratch / "exp")
        cfg_c = runner.cell_config(arm, core_arms[arm], {**core_exp, "epochs": 2}, chips, split_dir, 100, 0, 0,
                                   scratch / "core")
        for cfg in (cfg_e, cfg_c):
            cfg["data"]["num_workers"] = 0
        ok_cfg = diff({k: v for k, v in cfg_e.items() if k not in ("workflow", "output_root", "worldcover",
                                                                  "kshot_arm_file")},
                      {k: v for k, v in cfg_c.items() if k not in ("workflow", "output_root", "worldcover",
                                                                  "kshot_arm_file")}, "config")
        d += ok_cfg
        quiet = lambda m: None  # noqa: E731
        r_e = fit_borders.fit(cfg_e, scratch / "exp" / arm, src_e, chips, split_dir, worldcover, num_workers=0,
                              device="cpu", log=quiet)
        r_c = core_fit.fit(cfg_c, scratch / "core" / arm, src_c, chips, split_dir, worldcover, num_workers=0,
                           device="cpu", log=quiet)
        h_e = [{k: v for k, v in row.items() if not k.endswith("_s")} for row in r_e["history"]]
        h_c = [{k: v for k, v in row.items() if not k.endswith("_s")} for row in r_c["history"]]
        d += diff(_plain(h_e), _plain(h_c), "history")
        w_e = torch.load(scratch / "exp" / arm / "best.pt", map_location="cpu", weights_only=False)["net"]
        w_c = torch.load(scratch / "core" / arm / "best.pt", map_location="cpu", weights_only=False)["net"]
        bad = [k for k in w_e if k not in w_c or not torch.equal(w_e[k], w_c[k])]
        d += [f"weights differ: {len(bad)} tensors, e.g. {bad[:3]}"] if bad else []
        with np.load(scratch / "exp" / arm / "predictions_test.npz") as a, \
             np.load(scratch / "core" / arm / "predictions_test.npz") as b:
            if not all(np.array_equal(a[k], b[k]) for k in ("prob", "pred", "mask", "ids")):
                d.append("test predictions differ")
        print(("  ok    " if not d else "  FAIL  ") + f"{arm}: history of {len(h_c)} epochs, {len(w_c)} tensors"
              + ("" if not d else f"; {len(d)} differences, e.g. " + "; ".join(d[:3])), flush=True)
        fails += [f"{arm} {x}" for x in d]
    return fails


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--scoring", action="store_true")
    ap.add_argument("--training", action="store_true")
    ap.add_argument("--chips", default=None, help="EE_2021_rev for --scoring, EE_2021_rev_mini for --training")
    ap.add_argument("--arms", nargs="+", default=list(TRAINING_ARMS))
    ap.add_argument("--scratch", type=Path, default=None)
    a = ap.parse_args()
    fails = []
    if a.scoring:
        fails += check_scoring(a.chips or "EE_2021_rev")
    if a.training:
        with tempfile.TemporaryDirectory(dir=a.scratch) as tmp:
            fails += check_training(a.chips or "EE_2021_rev_mini", a.arms, Path(tmp))
    print(f"\n{len(fails)} difference(s)" + "".join(f"\n  {m}" for m in fails[:40]), flush=True)
    sys.exit(1 if fails or not (a.scoring or a.training) else 0)


if __name__ == "__main__":
    main()
