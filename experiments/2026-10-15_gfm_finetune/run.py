"""Entry script: the K-shot runner, with the token-grid encoders frozen (core arms) or fine-tuned (``arms_run/``).

Three of the core's objects are swapped for this folder's before the runner starts (DESIGN.md, "How it runs"):

* arm lookup searches ``arms_run/`` first, and accepts ``route: end_to_end`` with ``model.trainable`` for a
  token-grid backbone, which the core's ``load_arm`` refuses (it allows only ``cache`` there);
* a cell whose model carries ``trainable`` is fitted by ``fit_ft.fit_finetune``; any other by the core;
* test predictions go through ``finetune.FtFitPredictor``, which passes the coordinates and loads strictly for
  this folder's cells and predicts every other cell as the core does.

The command line is that of scripts/run_kshot.py, so ``scripts/cluster/submit.sh`` runs this file unchanged:

    poetry run python experiments/2026-10-15_gfm_finetune/run.py -e experiments/2026-10-15_gfm_finetune/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini [--dry-run]
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]

#: The route of this folder's arms: the encoder inside the training loop.
ROUTE = "end_to_end"


def apply_overrides() -> None:
    import yaml

    from gfm4agri.benchmark import evaluation, fit
    from gfm4agri.pipeline import config

    core_arm_dirs, core_load_arm, core_fit = config.arm_dirs, config.load_arm, fit.fit_end_to_end

    def arm_dirs(experiment, configs=config.CONFIGS):
        return [HERE / "arms_run", *core_arm_dirs(experiment, configs)]

    def load_arm(name, configs=config.CONFIGS, search=None):
        from gfm4agri.benchmark.backbones import get_backbone

        dirs = search or [Path(configs) / "arms"]
        path = next((d / f"{name}.yaml" for d in dirs if (d / f"{name}.yaml").exists()), None)
        arm = yaml.safe_load(path.read_text()) if path is not None else None
        if arm is None or arm.get("route") != ROUTE:
            return core_load_arm(name, configs, search)
        if get_backbone(arm["model"]["backbone"]).representation != "s2_monthly":
            raise ValueError(f"arm {name}: route {ROUTE!r} is for an encoder on the Sentinel-2 stack")
        if "trainable" not in arm["model"]:
            raise ValueError(f"arm {name}: route {ROUTE!r} needs model.trainable")
        arm["_file"] = str(path)
        return arm

    def fit_end_to_end(cfg, **kwargs):
        if "trainable" in cfg["model"]:
            from fit_ft import fit_finetune

            return fit_finetune(cfg, **kwargs)
        return core_fit(cfg, **kwargs)

    from finetune import FtFitPredictor

    config.arm_dirs, config.load_arm = arm_dirs, load_arm
    fit.fit_end_to_end = fit_end_to_end
    evaluation.FitPredictor = FtFitPredictor
    # The runner binds these names when it is imported; rebind them there too, should it be imported already.
    runner = sys.modules.get("gfm4agri.pipeline.runner")
    if runner is not None:
        runner.arm_dirs, runner.load_arm = arm_dirs, load_arm


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
