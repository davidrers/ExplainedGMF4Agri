"""Download the pretrained weights of the cache-route arms into the local Hugging Face cache.

Run on the cluster's login node by setup_env.sh, so jobs never download. Building each arm's
end-to-end model once triggers the download; nothing is trained.

    poetry run python scripts/cluster/prefetch_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    from gfm4agri.benchmark.segmentation import build_task
    from gfm4agri.chips.s2_monthly import BAND_NAMES
    from gfm4agri.pipeline.config import CONFIGS, load_arm, load_experiment

    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    for name in exp["arms"]:
        arm = load_arm(name)
        if arm["route"] != "cache":
            continue
        backbone = arm["model"]["backbone"]
        print(f"{backbone}: building once to fetch its weights", flush=True)
        build_task(backbone, num_classes=2, class_names=["a", "b"], bands=list(BAND_NAMES),
                   n_timesteps=12, ignore_index=-1)
    print("weights cached")


if __name__ == "__main__":
    main()
