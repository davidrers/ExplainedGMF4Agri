"""Entry script of the Sentinel-1 and Sentinel-2 study: the K-shot runner with TerraMind on Sentinel-1 alone.

The core registry holds TerraMind v1 large on Sentinel-2 (``terramind_v1_large``) and on Sentinel-2 with
Sentinel-1 RTC (``terramind_v1_large_s2s1``). This script adds ``terramind_v1_large_s1``: the same encoder built
with the S1RTC modality alone (VV and VH in dB, TerraMind's pretrained S1RTC patch embedding and statistics), the
same temporal wrapper, ``ChannelBottleneck`` and decoder.

The datamodule reads the two rasters of a two-modality backbone and splits them into one tensor per modality after
normalisation. For the S1 arm, the Sentinel-2 tensor is then dropped, so the encoder sees Sentinel-1 only. TerraMind
would ignore an input modality it was not built with, but dropping it states the intent and spares the warning.
The S1 arm file is in ``arms_run/``, put first on the arm search path, because the core tests check the arms of an
experiment's ``arms/`` against the core registry, which does not hold this backbone. The command line is that of
scripts/run_kshot.py:

    poetry run python experiments/2026-10-13_terramind_s1_s2/run.py -e experiments/2026-10-13_terramind_s1_s2/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]

#: Backbones of this study whose datamodule reads a modality the encoder must not see.
DROP = {"terramind_v1_large_s1": ("S2L2A",)}


def _s1_only(variant: str, indices: list[int], bottleneck: int):
    from gfm4agri.benchmark import backbones as bb

    core = bb._terramind(variant, indices, bottleneck)

    def build(bands: list[str], n_timesteps: int) -> dict:
        args = core(bands, n_timesteps)
        args["backbone_modalities"] = ["S1RTC"]
        args["backbone_bands"] = {"S1RTC": list(bb.S1RTC["S1RTC"]["bands"])}
        return args
    return build


def apply_overrides() -> None:
    from gfm4agri.benchmark import backbones as bb
    from gfm4agri.benchmark import segmentation_data as sd
    from gfm4agri.pipeline import config

    core_arm_dirs = config.arm_dirs
    config.arm_dirs = lambda experiment, configs=config.CONFIGS: [HERE / "arms_run",
                                                                  *core_arm_dirs(experiment, configs)]

    bb.BACKBONES["terramind_v1_large_s1"] = bb.BackboneSpec(
        name="terramind_v1_large_s1", resolution_group="token_grid",
        model_args=_s1_only("terramind_v1_large", [5, 11, 17, 23], bottleneck=768),
        # The Sentinel-2 stack is still read and normalised before it is dropped, with the S2 statistics.
        stats=bb._terramind_stats, extra_modalities=bb.S1RTC,
        notes="terramind_v1_large on S1RTC alone; the Sentinel-2 input is dropped after the modality split")

    core_init = sd.EuroCropsSegDataModule.__init__
    core_split = sd.EuroCropsSegDataModule.split_modalities

    def __init__(self, root, backbone, *args, **kwargs):
        core_init(self, root, backbone, *args, **kwargs)
        self.drop_modalities = DROP.get(backbone, ())

    def split_modalities(self, x):
        parts = core_split(self, x)
        drop = getattr(self, "drop_modalities", ())
        if drop and isinstance(parts, dict):
            parts = {k: v for k, v in parts.items() if k not in drop}
        return parts

    sd.EuroCropsSegDataModule.__init__ = __init__
    sd.EuroCropsSegDataModule.split_modalities = split_modalities


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
