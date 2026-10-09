"""The end-to-end fit, moved from scripts/seg/train.py into the library, still writes its result file."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("terratorch")
pytest.importorskip("rasterio")

from conftest import SYN_SPLIT  # noqa: E402


def _cfg(chips, out, pct=100):
    return {
        "run_name": "tiny_tessera", "seed": 0,
        "data": {"root": str(chips), "split": str(chips / "splits" / SYN_SPLIT),
                 "normalisation": "chips", "batch_size": 2, "num_workers": 0, "augment": True},
        "label_budget": {"mode": "pct", "pct": pct, "draw_seed": 0},
        "model": {"backbone": "tessera_v1", "lr": 1.0e-3, "weight_decay": 0.05,
                  "head_dropout": 0.2, "loss": "ce"},
        "trainer": {"max_epochs": 1, "precision": "32-true", "log_every_n_steps": 1},
        "output_root": str(out),
    }


def test_fit_end_to_end_writes_results_where_the_grid_expects_them(split_chipset, tmp_path):
    from gfm4agri.benchmark.fit import fit_end_to_end

    out = tmp_path / "results"
    r = fit_end_to_end(_cfg(split_chipset, out, pct=100), fast_dev_run=True)
    path = out / "tiny_tessera" / "P100_draw0_seed0" / "results.json"
    assert path.exists() and json.loads(path.read_text())["run"] == "tiny_tessera"
    assert r["chips"] == {"train": 2, "val": 2, "test": 2}
    assert r["label_budget"]["support"] == {"a": {"available": 4, "drawn": 4},
                                            "b": {"available": 2, "drawn": 2}}
    assert r["split_protocol"]["split"] == SYN_SPLIT
