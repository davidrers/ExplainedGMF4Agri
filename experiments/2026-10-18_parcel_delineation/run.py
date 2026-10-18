"""Entry script of the parcel delineation experiment: the K-shot runner with a boundary head trained beside the
crop-type head, and the test parcels scored by Panoptic Quality.

What it swaps in, without editing the core (``delineation.py`` and ``panoptic.py`` say why):

* TerraTorch's ``SemanticSegmentationTask`` becomes :func:`delineation.task_class`, which adds the boundary logit
  beside every arm's segmentation head and trains the two jointly. ``build_task``, ``build_cached_model``,
  ``fit_cached`` and ``FitPredictor`` all build their task or model through it, so the three routes (cached
  features, embedding rasters, the U-Net trained end to end) and the prediction get the same head;
* the end-to-end and two-stage datasets encode the boundary target into the mask, under the label budget;
* the test predictions keep the boundary probability beside the class (``boundary`` in ``predictions_test.npz``);
* after the parcel scores of protocol D28, every finished cell gets ``panoptic_metrics`` in its ``results.json``
  (:func:`panoptic.score_predictions`), and ``summary.csv`` their headline columns;
* every ``results.json`` records the target, the loss and the scoring rule under ``delineation``.

The command line is that of scripts/run_kshot.py:

    poetry run python experiments/2026-10-18_parcel_delineation/run.py -e experiments/2026-10-18_parcel_delineation/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]

#: summary.csv columns: variant -> metric -> column.
SUMMARY = {"oracle": {"rq": "rq_oracle"},
           "extent": {"pq_agnostic": "pq_agn_extent"},
           "semantic": {"pq": "pq_semantic", "rq": "rq_semantic", "pq_agnostic": "pq_agn_semantic"},
           "boundary@0.5": {"pq": "pq_boundary", "sq": "sq_boundary", "rq": "rq_boundary",
                            "pq_agnostic": "pq_agn_boundary"},
           "combined@0.5": {"pq": "pq_combined", "rq": "rq_combined", "pq_agnostic": "pq_agn_combined"}}


def score_cell(cell: Path, chips: Path, workers: int | None = None) -> dict:
    """Write the panoptic scores of one finished cell into its ``results.json``."""
    import panoptic

    f = cell / "results.json"
    r = json.loads(f.read_text())
    workers = workers or min(16, os.cpu_count() or 1)
    r["panoptic_metrics"] = panoptic.score_predictions(cell / "predictions_test.npz", chips / "chips",
                                                       r["class_names"], workers=workers)
    tmp = f.with_name(f"results.json.{os.getpid()}")
    tmp.write_text(json.dumps(r, indent=2))
    tmp.replace(f)
    return r["panoptic_metrics"]


def apply_overrides() -> None:
    import numpy as np
    import terratorch.tasks as tasks

    import delineation
    import panoptic
    from gfm4agri.benchmark import cached, evaluation
    from gfm4agri.benchmark import segmentation_data as sd
    from gfm4agri.pipeline import runner

    tasks.SemanticSegmentationTask = delineation.task_class(tasks.SemanticSegmentationTask)
    sd.SparseParcelSegmentationDataset = delineation.sparse_dataset_class(sd.SparseParcelSegmentationDataset)
    cached.CachedFeatureDataset = delineation.cached_dataset_class(cached.CachedFeatureDataset)
    Predictor = delineation.predictor_class(evaluation.FitPredictor)

    def predict(cell_dir, test_ids, num_workers):
        import gc

        import torch

        p = Predictor(cell_dir)
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
        r["delineation"] = {"training": delineation.RECORD, "scoring": panoptic.RECORD}
        Path(results_path).write_text(json.dumps(r, indent=2))

    runner._stamp = stamp

    core_add = runner.add_parcel_metrics

    def add_parcel_metrics(out_root, chips=None, arms=None, force=False):
        done = core_add(out_root, chips, arms=arms, force=force)
        for f in sorted(Path(out_root).glob("*/P*_draw*_seed*/results.json")):
            cell = f.parent
            if runner._state(cell) != "done" or (arms is not None and cell.parent.name not in arms):
                continue
            r = json.loads(f.read_text())
            if "panoptic_metrics" in r and not force:
                continue
            root = Path(chips) if chips else REPO / r["config"]["data"]["root"]
            runner._say(f"{cell.parent.name} {cell.name}: panoptic scores")
            score_cell(cell, root)
            if cell not in done:
                done.append(cell)
        return done

    runner.add_parcel_metrics = add_parcel_metrics

    core_row = runner._row

    def row(results_path):
        out = core_row(results_path)
        r = json.loads(Path(results_path).read_text())
        pm = r.get("panoptic_metrics", {})
        v = pm.get("variants", {})
        for variant, cols in SUMMARY.items():
            for metric, col in cols.items():
                out[col] = v.get(variant, {}).get(metric)
        out["boundary_f1"] = pm.get("boundary_pixels", {}).get("f1")
        return out

    runner._row = row


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
