"""Download the pretrained weights of the arms that run an encoder.

Run on the cluster's login node by setup_env.sh, so jobs never download. Building each arm's
end-to-end model once triggers the download into the Hugging Face cache; TESSERA v2's student
is fetched into data/weights/tessera/v2/ and checked against its SHA-256. Nothing is trained.

    poetry run python scripts/cluster/prefetch_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    from gfm4agri.benchmark.backbones import get_backbone
    from gfm4agri.benchmark.segmentation import build_task
    from gfm4agri.chips.s2_monthly import BAND_NAMES
    from gfm4agri.pipeline.config import CONFIGS, load_arm, load_experiment

    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    for name in exp["arms"]:
        arm = load_arm(name)
        backbone = arm["model"]["backbone"]
        spec = get_backbone(backbone)
        if spec.computed:
            from gfm4agri.embeddings.tessera_v2 import WEIGHTS, load_student

            print(f"{backbone}: fetching and checking its weights", flush=True)
            load_student(spec.computed["student"], WEIGHTS, spec.computed["checkpoint_sha256"])
            continue
        if arm["route"] != "cache":
            continue
        # The bands a fit hands the encoder: its declared subset, else every exported band.
        bands = list(get_backbone(backbone).input_bands or BAND_NAMES)
        print(f"{backbone}: building once to fetch its weights", flush=True)
        build_task(backbone, num_classes=2, class_names=["a", "b"], bands=bands,
                   n_timesteps=12, ignore_index=-1)
    print("weights cached")


if __name__ == "__main__":
    main()
