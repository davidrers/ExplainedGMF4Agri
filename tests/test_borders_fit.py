"""One cell of the two-head workflow end to end on the synthetic chip set (gfm4agri.borders.fit)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("rasterio")
pytest.importorskip("skimage")

from conftest import SYN_BLOCKS, SYN_H  # noqa: E402
from gfm4agri.borders import fit, networks, prue  # noqa: E402
from gfm4agri.pipeline.config import resolve_split  # noqa: E402


def _worldcover(tmp_path):
    ids = [c for _, cids in SYN_BLOCKS.values() for c in cids]
    path = tmp_path / "worldcover.npz"
    np.savez_compressed(path, **{c: np.full((SYN_H, SYN_H), 10, np.uint8) for c in ids})
    return path


SOURCE = {"kind": "raster", "backbone": "tessera_v1", "suffix": "_tessera.tif", "means": [0.0] * 4,
          "stds": [1.0] * 4, "channels": 4, "cache": "chip set"}
CFG = {"run_name": "tiny", "workflow": "test", "input": "raster", "seed": 0, "epochs": 1,
       "label_budget": {"mode": "pct", "pct": 100, "draw_seed": 0},
       "data": {"batch_size": 2, "eval_batch_size": 2, "augment": True, "num_workers": 0},
       "unet": {"encoder": "resnet18", "weights": None},
       "crop": {"hidden": [8], "dropout": 0.0, "head_dropout": 0.0},
       "optim": {"unet": {"lr": 1e-3, "weight_decay": 0.01, "amsgrad": True, "schedule": "cosine", "min_lr": 1e-6},
                 "crop_head": {"lr": 1e-3, "weight_decay": 0.05, "schedule": "cosine", "min_lr": 1e-6}},
       "loss": {"class_weights": [0.05, 0.20, 0.75]}, "balance": {"target": 0.3, "every": 1, "smooth": 0.5}}


def test_the_chips_carry_both_targets_on_the_same_pixels(split_chipset, tmp_path):
    split_dir = resolve_split(split_chipset)
    ids = {s: (split_dir / f"{n}_data.txt").read_text().split()
           for s, n in (("train", "training"), ("val", "validation"), ("test", "test"))}
    src = {**SOURCE, "dir": str(split_chipset)}
    _, _, test = fit.datasets(src, ids, None, split_chipset / "chips", _worldcover(tmp_path), augment=False)
    it = test[0]
    assert tuple(it["image"].shape) == (4, SYN_H, SYN_H)
    y, yc = it["y"].numpy(), it["yc"].numpy()
    assert np.array_equal(yc != networks.IGNORE_CROP, (y == prue.INTERIOR) | (y == prue.BOUNDARY))


def test_one_cell_fits_predicts_and_records(split_chipset, tmp_path):
    split_dir = resolve_split(split_chipset)
    cell = tmp_path / "cell"
    r = fit.fit(CFG, cell, {**SOURCE, "dir": str(split_chipset)}, split_chipset, split_dir, _worldcover(tmp_path),
                num_workers=0, device="cpu", log=lambda m: None)
    assert r["best_epoch"] == 1 and np.isfinite(r["final_crop_weight"])
    assert set(r["epoch_by_single_criterion"]) == {"object_f1", "crop_loss", "crop_macro_f1", "pq_agnostic_grow"}
    assert json.loads((cell / "results.json").read_text())["run"] == "tiny"
    with np.load(cell / "predictions_test.npz") as z:
        assert z["prob"].shape == (2, 3, SYN_H, SYN_H) and z["prob"].dtype == np.uint8
        assert z["pred"].shape == (2, SYN_H, SYN_H) and len(z["ids"]) == 2
    assert (cell / "best.pt").exists() and (cell / "metrics.csv").exists()
    assert not (cell / "last.pt").exists()
