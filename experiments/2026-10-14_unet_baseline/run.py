"""Entry script of the U-Net baseline experiment: the K-shot runner with a from-scratch 2D U-Net as an arm.

The core knows two routes: a frozen encoder on the Sentinel-2 stack fitted from cached features (``cache``), and a
precomputed embedding raster fitted end to end (``raster``). The baseline is a third case: the Sentinel-2 stack,
fitted end to end, with nothing frozen and nothing cached. This script adds it without editing the core:

* ``unet_r18_scratch`` enters the backbone registry with the ``s2_monthly`` representation and no pretraining
  statistics, so the datamodule reads the 144-band chips under training-chip normalisation;
* its arms declare ``route: end_to_end``, which ``load_arm`` accepts for this backbone alone; the runner sends every
  route other than ``cache`` to ``fit_end_to_end``;
* ``build_task`` builds :class:`scratch_unet.ScratchUNet` for it through its TerraTorch model factory, with nothing
  frozen; every other backbone gets the core's task, so the core arms of the experiment run as in the main workflow;
* an arm may set ``epochs``, which overrides the experiment's for that arm (the longer-training arm);
* ``fit_end_to_end`` writes ``frozen_encoder: true`` into every ``results.json``; for this backbone the record is
  corrected to ``false`` and the model is described.

The arm files are in ``arms_run/``, put first on the arm search path, because the core tests check the arms of an
experiment's ``arms/`` against the core registry, which does not hold this backbone. The command line is that of
scripts/run_kshot.py:

    poetry run python experiments/2026-10-14_unet_baseline/run.py -e experiments/2026-10-14_unet_baseline/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]

#: Backbones of this study trained from scratch, with what ``ScratchUNet`` is built from.
SCRATCH = {"unet_r18_scratch": {"model": "smp.Unet", "encoder": "resnet18", "encoder_weights": None,
                                "temporal_fusion": "early: 12 months x 12 bands as 144 channels"}}


def apply_overrides() -> None:
    import yaml

    from gfm4agri.benchmark import backbones as bb
    from gfm4agri.benchmark import fit
    from gfm4agri.benchmark import segmentation as seg
    from gfm4agri.pipeline import config, runner

    for name in SCRATCH:
        bb.BACKBONES[name] = bb.BackboneSpec(
            name=name, resolution_group="scratch",
            model_args=lambda bands, n_timesteps: {}, stats=None,
            notes="2D U-Net (ResNet-18 encoder) trained from scratch on the 12 monthly Sentinel-2 composites, "
                  "all 12 bands, months stacked on channels")

    core_arm_dirs = config.arm_dirs
    config.arm_dirs = runner.arm_dirs = (
        lambda experiment, configs=config.CONFIGS: [HERE / "arms_run", *core_arm_dirs(experiment, configs)])

    core_load_arm = config.load_arm

    def load_arm(name, configs=config.CONFIGS, search=None):
        dirs = search or [Path(configs) / "arms"]
        path = next((d / f"{name}.yaml" for d in dirs if (d / f"{name}.yaml").exists()), None)
        if path is not None:
            arm = yaml.safe_load(path.read_text())
            if arm["model"]["backbone"] in SCRATCH:
                if arm["route"] != "end_to_end":
                    raise ValueError(f"arm {name}: route {arm['route']!r}, but a scratch model needs 'end_to_end'")
                if arm["data"]["normalisation"] != "chips":
                    raise ValueError(f"arm {name}: a scratch model has no pretraining statistics; use 'chips'")
                arm["_file"] = str(path)
                return arm
        return core_load_arm(name, configs, search)

    config.load_arm = runner.load_arm = load_arm

    core_compose = config.compose

    def compose(arm_name, arm, exp, machine, chips, split_dir, cell, output_root):
        cfg = core_compose(arm_name, arm, exp, machine, chips, split_dir, cell, output_root)
        if "epochs" in arm:
            cfg["trainer"]["max_epochs"] = int(arm["epochs"])
        return cfg

    config.compose = runner.compose = compose

    core_build_task = seg.build_task

    def build_task(backbone, *, num_classes, class_names, bands, n_timesteps, ignore_index=-1, lr=1e-4,
                   weight_decay=0.05, head_dropout=0.1, loss="ce", decoder=None, plot_on_val=False):
        if backbone not in SCRATCH:
            return core_build_task(backbone, num_classes=num_classes, class_names=class_names, bands=bands,
                                   n_timesteps=n_timesteps, ignore_index=ignore_index, lr=lr,
                                   weight_decay=weight_decay, head_dropout=head_dropout, loss=loss,
                                   decoder=decoder, plot_on_val=plot_on_val)
        from terratorch.tasks import SemanticSegmentationTask

        import scratch_unet  # noqa: F401  registers ScratchUNetFactory

        model_args = {"in_channels": len(bands) * n_timesteps, "num_classes": num_classes,
                      "encoder": SCRATCH[backbone]["encoder"], "head_dropout": head_dropout}
        return SemanticSegmentationTask(
            model_factory="ScratchUNetFactory", model_args=model_args, loss=loss, lr=lr, optimizer="AdamW",
            optimizer_hparams={"weight_decay": weight_decay}, ignore_index=ignore_index, freeze_backbone=False,
            freeze_decoder=False, plot_on_val=plot_on_val, class_names=class_names)

    seg.build_task = build_task

    core_fit = fit.fit_end_to_end

    def fit_end_to_end(cfg, **kwargs):
        results = core_fit(cfg, **kwargs)
        backbone = cfg["model"]["backbone"]
        if backbone in SCRATCH:
            path = (REPO / cfg["output_root"] / cfg["run_name"]
                    / f"{fit.budget_tag(cfg['label_budget'])}_seed{cfg['seed']}" / "results.json")
            r = json.loads(path.read_text())
            r["frozen_encoder"] = False
            r["scratch_model"] = SCRATCH[backbone]
            path.write_text(json.dumps(r, indent=2))
            results.update(frozen_encoder=False, scratch_model=SCRATCH[backbone])
        return results

    fit.fit_end_to_end = fit_end_to_end


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
