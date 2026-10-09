"""Entry script of the THOR token-size study: the K-shot runner with THOR v1 large on 40 m tokens registered.

The core registry holds THOR v1 large on 160 m (``thor_v1_large``) and 80 m (``thor_v1_large_80m``) tokens. This
script adds the 40 m entry, built by the core's own THOR builder with ``token_m=40``, and the decoder of its
resolution group, the token-grid UNet of the other two. Nothing else differs. The 40 m arm file is in
``arms_run/``, which this script puts first on the arm search path: the core tests check every arm in an
experiment's ``arms/`` against the core registry, which does not hold this backbone. The command line is that of
scripts/run_kshot.py:

    poetry run python experiments/2026-10-09_thor_token_size/run.py -e experiments/2026-10-09_thor_token_size/experiment.yaml --chips data/eurocrops_chips/EE_2021_sample
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]


def apply_overrides() -> None:
    """Register ``thor_v1_large_40m`` and its decoder in the core registries, which every module reads, look
    arms up in ``arms_run/`` first, and fit with at most four loader workers on the cluster."""
    from gfm4agri.benchmark import backbones as bb
    from gfm4agri.benchmark import segmentation as seg
    from gfm4agri.pipeline import config

    core_arm_dirs = config.arm_dirs
    config.arm_dirs = lambda experiment, configs=config.CONFIGS: [HERE / "arms_run",
                                                                  *core_arm_dirs(experiment, configs)]

    # A 40 m sample is 1.23 GB once the cached features are cast to float32, so the cluster profile's 16 fit
    # workers, each prefetching two batches of 2, held about 79 GB of shared memory and job 617995 failed
    # in its first fit. Four workers hold what 80 m's sixteen did. Workers change throughput and which
    # worker draws a sample's D4 variant, nothing else.
    core_load_machine = config.load_machine

    def load_machine(name: str, configs: Path = config.CONFIGS) -> dict:
        machine = core_load_machine(name, configs)
        if name == "cluster":
            machine["num_workers"] = min(machine["num_workers"], 4)
        return machine

    config.load_machine = load_machine

    bb.BACKBONES["thor_v1_large_40m"] = bb.BackboneSpec(
        name="thor_v1_large_40m", resolution_group="token_grid_40m",
        model_args=bb._thor("thor_v1_large", [5, 11, 17, 23], bottleneck=768, token_m=40),
        stats=bb._thor_stats, input_bands=bb.THOR_BANDS,
        notes="thor_v1_large on 40 m tokens, 4 and 2 px patches, a 56 x 56 grid; 2 px is below the "
              "4 px THOR was pretrained with")
    seg.DECODERS["token_grid_40m"] = dict(seg.DECODERS["token_grid_80m"])


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
