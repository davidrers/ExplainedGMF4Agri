"""Entry script of the fine-tuned TESSERA boundary-head fits: the K-shot runner with this folder's overrides.

It fits the one arm of this folder, ``tessera_v2_medium_ft_bnd``: TESSERA v2 Medium fine-tuned on the labelled-pixel
route of ``experiments/2026-10-15_kshot_finetune``, with the boundary head and joint loss of
``experiments/2026-10-18_parcel_delineation`` beside the crop-type head. The frozen side needs no fit: it is the
delineation experiment's ``tessera_v2_medium`` cells, already stored. What it swaps in, without editing the core:

* arm lookup searches ``arms_run/`` first and accepts the ``labelled_pixels`` route, which the core's ``load_arm`` refuses (as kshot_finetune's
  ``run.py``); the runner still resolves or builds the frozen raster cache, whose statistics the route reads;
* the cell is fitted by ``fit_pixels.fit_pixels`` with ``pixelbnd``'s boundary task;
* the end-to-end dataset carries the boundary target inside its masks, under the label budget
  (``delineation.sparse_dataset_class``);
* the test predictions keep the boundary probability beside the class (``boundary`` in ``predictions_test.npz``),
  from ``pixelbnd.BndPixelPredictor``;
* every ``results.json`` records the boundary training under ``delineation``.

The parcels are scored afterwards on the hub, for the three extents, by ``score.py``: the WorldCover maps live there.

The command line is that of scripts/run_kshot.py, so ``scripts/cluster/submit.sh`` runs this file:

    poetry run python experiments/2026-11-05_parcel_extent/run.py -e experiments/2026-11-05_parcel_extent/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]


def apply_overrides() -> None:
    import numpy as np
    import yaml

    import delineation
    from gfm4agri.benchmark import fit
    from gfm4agri.benchmark import segmentation_data as sd
    from gfm4agri.pipeline import config

    core_arm_dirs, core_load_arm, core_fit = config.arm_dirs, config.load_arm, fit.fit_end_to_end

    def arm_dirs(experiment, configs=config.CONFIGS):
        # arms_run/, not arms/: the core's arm loader, which the test suite runs on every experiment's arms/,
        # refuses the labelled-pixel route (as in experiments/2026-10-15_kshot_finetune).
        return [HERE / "arms_run", *core_arm_dirs(experiment, configs)]

    def load_arm(name, configs=config.CONFIGS, search=None):
        from gfm4agri.benchmark.backbones import get_backbone

        dirs = search or [Path(configs) / "arms"]
        path = next((d / f"{name}.yaml" for d in dirs if (d / f"{name}.yaml").exists()), None)
        arm = yaml.safe_load(path.read_text()) if path is not None else None
        if arm is None or arm.get("route") != "labelled_pixels":
            return core_load_arm(name, configs, search)
        if not get_backbone(arm["model"]["backbone"]).computed:
            raise ValueError(f"arm {name}: route labelled_pixels is for a computed 10 m raster (TESSERA v2)")
        arm["_file"] = str(path)
        return arm

    def fit_end_to_end(cfg, **kwargs):
        if "trainable" not in cfg["model"]:
            return core_fit(cfg, **kwargs)
        from fit_pixels import fit_pixels

        return fit_pixels(cfg, **kwargs)

    config.arm_dirs, config.load_arm = arm_dirs, load_arm
    fit.fit_end_to_end = fit_end_to_end
    sd.SparseParcelSegmentationDataset = delineation.sparse_dataset_class(sd.SparseParcelSegmentationDataset)

    from gfm4agri.pipeline import runner

    runner.arm_dirs, runner.load_arm = arm_dirs, load_arm

    def predict(cell_dir, test_ids, num_workers):
        import gc

        import torch

        from pixelbnd import BndPixelPredictor

        p = BndPixelPredictor(cell_dir)
        out = p.predict(test_ids, num_workers=num_workers)
        keys = sorted(out)
        tmp = cell_dir / (runner.PREDICTIONS + ".tmp")
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, ids=np.array(keys), pred=np.stack([out[k][0] for k in keys]),
                                mask=np.stack([out[k][1] for k in keys]),
                                boundary=np.stack([out[k][2] for k in keys]))
        tmp.replace(cell_dir / runner.PREDICTIONS)
        del p
        gc.collect()
        torch.cuda.empty_cache()

    runner._predict = predict

    core_stamp = runner._stamp

    def stamp(results_path, provenance):
        core_stamp(results_path, provenance)
        r = json.loads(Path(results_path).read_text())
        r.setdefault("delineation", {})["training"] = delineation.RECORD
        Path(results_path).write_text(json.dumps(r, indent=2))

    runner._stamp = stamp


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
