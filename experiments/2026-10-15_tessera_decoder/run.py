"""Entry script of the TESSERA decoder experiment: the K-shot runner on frozen TESSERA v2 Medium rasters, with the
core's per-pixel MLP or a U-Net as the trainable decoder.

The core's ``raster`` route fits a decoder on a precomputed embedding raster. This script adds, without editing the
core:

* **the representation** ``tessera_v2m``: before the runner starts, :func:`rasters.ensure` encodes every chip of the
  chip set with the frozen v2 Medium student, from the composites on disk, into a linked copy of the chip set (under
  the cache root on the hub, the job's ``--scratch`` on the cluster), and the runner is pointed at that copy;
* **two backbones** of that representation in the registry: ``tessera_v2m``, the core's ``pixel_raster`` case (an
  identity backbone and the group's ``PixelMLPDecoder``), and ``tessera_v2m_unet``, for which ``build_task`` builds
  :class:`scratch_unet.ScratchUNet` on the 128 embedding channels, every weight trainable;
* an arm may set ``epochs``, which overrides the experiment's for that arm;
* ``results.json`` of a U-Net cell records the decoder, and every cell's provenance names the chip set itself and
  the rasters' key, not the linked copy.

TESSERA stays frozen in every arm: it runs once, before any fit, and no gradient reaches it. ``frozen_encoder: true``
in ``results.json`` refers to it. In the U-Net arm the ResNet-18 down path that ``count_params`` lists as
``encoder`` belongs to the trainable decoder.

The arm files are in ``arms_run/``, put first on the arm search path: the core tests check an experiment's
``arms/`` against the core registry, which does not hold these backbones. The command line is that of
scripts/run_kshot.py:

    poetry run python experiments/2026-10-15_tessera_decoder/run.py -e experiments/2026-10-15_tessera_decoder/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]

#: The frozen encoder of the rasters: the v2 Medium student of experiments/2026-10-14_tessera_composites, under the
#: precision of its frozen arm.
ENCODER = {"student": "medium",
           "checkpoint_sha256": "3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e",
           "precision": "bf16-mixed"}
WEIGHTS = REPO / "data/weights/tessera/v2"
#: Backbones whose decoder is a U-Net, with what it is built from.
UNET = {"tessera_v2m_unet": {"model": "smp.Unet", "encoder": "resnet18", "encoder_weights": None,
                             "input": "frozen TESSERA v2 Medium embeddings, 128 channels at 10 m",
                             "trainable": "everything: ResNet-18 down path, decoder and head"}}


def apply_overrides() -> None:
    import yaml

    from gfm4agri.benchmark import backbones as bb
    from gfm4agri.benchmark import fit
    from gfm4agri.benchmark import segmentation as seg
    from gfm4agri.pipeline import config, runner

    import rasters

    bb.BACKBONES["tessera_v2m"] = bb.BackboneSpec(
        name="tessera_v2m", resolution_group="pixel_raster", model_args=bb._precomputed(128), stats=None,
        representation=rasters.REPRESENTATION,
        notes="TESSERA v2 Medium, frozen, 128 dims per 10 m pixel, encoded from the chip set's monthly S2 and S1 "
              "composites; per-pixel MLP decoder")
    bb.BACKBONES["tessera_v2m_unet"] = bb.BackboneSpec(
        name="tessera_v2m_unet", resolution_group="pixel_raster", model_args=lambda bands, n_timesteps: {},
        stats=None, representation=rasters.REPRESENTATION,
        notes="TESSERA v2 Medium, frozen, as tessera_v2m; U-Net decoder (ResNet-18 down path), trained from scratch")

    core_arm_dirs = config.arm_dirs
    config.arm_dirs = runner.arm_dirs = (
        lambda experiment, configs=config.CONFIGS: [HERE / "arms_run", *core_arm_dirs(experiment, configs)])

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
        if backbone not in UNET:
            return core_build_task(backbone, num_classes=num_classes, class_names=class_names, bands=bands,
                                   n_timesteps=n_timesteps, ignore_index=ignore_index, lr=lr,
                                   weight_decay=weight_decay, head_dropout=head_dropout, loss=loss,
                                   decoder=decoder, plot_on_val=plot_on_val)
        from terratorch.tasks import SemanticSegmentationTask

        import scratch_unet  # noqa: F401  registers ScratchUNetFactory

        model_args = {"in_channels": len(bands) * n_timesteps, "num_classes": num_classes,
                      "encoder": UNET[backbone]["encoder"], "head_dropout": head_dropout}
        return SemanticSegmentationTask(
            model_factory="ScratchUNetFactory", model_args=model_args, loss=loss, lr=lr, optimizer="AdamW",
            optimizer_hparams={"weight_decay": weight_decay}, ignore_index=ignore_index, freeze_backbone=False,
            freeze_decoder=False, plot_on_val=plot_on_val, class_names=class_names)

    seg.build_task = build_task

    core_fit = fit.fit_end_to_end

    def fit_end_to_end(cfg, **kwargs):
        results = core_fit(cfg, **kwargs)
        backbone = cfg["model"]["backbone"]
        if backbone in UNET:
            path = (REPO / cfg["output_root"] / cfg["run_name"]
                    / f"{fit.budget_tag(cfg['label_budget'])}_seed{cfg['seed']}" / "results.json")
            r = json.loads(path.read_text())
            r["decoder_model"] = UNET[backbone]
            path.write_text(json.dumps(r, indent=2))
            results.update(decoder_model=UNET[backbone])
        return results

    fit.fit_end_to_end = fit_end_to_end

    core_run = runner.run
    core_check = runner.check_chipset

    def run(experiment, chips, *, split=None, arms=None, machine="hub", cache_root=None, scratch=None,
            dry_run=False, **kwargs):
        chips = Path(chips) if Path(chips).is_absolute() else Path.cwd() / chips
        split_dir = config.resolve_split(chips, split)
        mach = config.load_machine(machine)
        root = Path(cache_root or mach["cache_root"])
        root = root if root.is_absolute() else REPO / root
        scr = Path(scratch) if scratch else root
        if dry_run:
            action, copy = rasters.locate(chips, split_dir, ENCODER, root, scr)
            ids = sorted({c for v in rasters._lists(chips, split_dir).values() for c in v})
            problems = rasters.missing_inputs(chips, ids)
            print(f"  v2m rasters: {action} {copy}, {len(ids)} chips"
                  + "".join(f"\n  problem: {p}" for p in problems), flush=True)
            if problems:
                raise SystemExit("chip set check failed")
            # The rasters do not exist yet: check everything else against the chip set itself.
            runner.check_chipset = lambda c, s, a: [p for p in core_check(c, s, a)
                                                    if rasters.SIDECAR not in p and rasters.SUFFIX not in p]
            return core_run(experiment, chips, split=split, arms=arms, machine=machine, cache_root=cache_root,
                            scratch=scratch, dry_run=True, **kwargs)
        copy = rasters.ensure(chips, split_dir, ENCODER, cache_root=root, scratch=scr, weights=WEIGHTS)
        record = json.loads((copy / rasters.RECORD).read_text())
        core_stamp = runner._stamp

        def stamp(results_path, provenance):
            core_stamp(results_path, {**provenance, "chip_set": config.repo_relative(chips),
                                      "tessera_v2m_rasters": {"dir": str(copy), **record["key"]}})

        runner._stamp = stamp
        return core_run(experiment, copy, split=split, arms=arms, machine=machine, cache_root=cache_root,
                        scratch=scratch, dry_run=False, **kwargs)

    runner.run = run


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
