"""The runner plans cells, checks the chip set, fits, predicts, stamps and resumes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("terratorch")
pytest.importorskip("rasterio")

from conftest import SYN_SPLIT  # noqa: E402
from gfm4agri.pipeline.runner import PREDICTIONS, check_chipset, run  # noqa: E402

TINY_RASTER = {"route": "raster",
               "data": {"normalisation": "chips", "batch_size": 2, "augment": True},
               "model": {"backbone": "tessera_v1", "lr": 1.0e-3, "weight_decay": 0.05,
                         "head_dropout": 0.2, "loss": "ce"},
               "trainer": {"precision": "32-true", "log_every_n_steps": 1}}
TINY_CACHE = {"route": "cache",
              "data": {"normalisation": "backbone", "batch_size": 2, "augment": True},
              "model": {"backbone": "terramind_v1_large", "lr": 1.0e-4, "weight_decay": 0.05,
                        "head_dropout": 0.1, "loss": "ce"},
              "trainer": {"precision": "16-mixed", "log_every_n_steps": 1,
                          "accumulate_grad_batches": 2}}


@pytest.fixture()
def configs(tmp_path: Path) -> Path:
    c = tmp_path / "configs"
    for sub in ("arms", "experiments", "machines"):
        (c / sub).mkdir(parents=True)
    (c / "arms" / "tiny_tessera.yaml").write_text(yaml.safe_dump(TINY_RASTER))
    (c / "arms" / "tiny_terramind.yaml").write_text(yaml.safe_dump(TINY_CACHE))
    (c / "experiments" / "tiny.yaml").write_text(yaml.safe_dump(
        {"name": "tiny", "arms": ["tiny_tessera", "tiny_terramind"], "budgets_pct": [100, 50],
         "draws": [0], "seeds": [0], "epochs": 1, "checkpoint": "val_loss"}))
    (c / "machines" / "hub.yaml").write_text(yaml.safe_dump(
        {"cache_root": str(tmp_path / "emb"), "num_workers": 0, "encode_num_workers": 0,
         "encode_batch_size": {"terramind_v1_large": 1}}))
    return c


def test_dry_run_lists_cells_and_cache_decisions(split_chipset, configs, tmp_path):
    rows = run(configs / "experiments" / "tiny.yaml", split_chipset, dry_run=True,
               configs=configs, results_root=tmp_path / "results")
    assert [(r["arm"], r["cell"], r["state"]) for r in rows] == [
        ("tiny_tessera", "P100_draw0_seed0", "fit"), ("tiny_tessera", "P50_draw0_seed0", "fit"),
        ("tiny_terramind", "P100_draw0_seed0", "fit"), ("tiny_terramind", "P50_draw0_seed0", "fit")]
    assert rows[2]["cache"] == "compute" and not (tmp_path / "results" / "tiny").exists()


def test_a_chip_set_missing_rasters_is_refused_before_any_compute(split_chipset, configs, tmp_path):
    (split_chipset / "chips" / "EE_00008_00000_tessera.tif").unlink()
    with pytest.raises(SystemExit, match="1 chips lack _tessera.tif"):
        run(configs / "experiments" / "tiny.yaml", split_chipset, dry_run=True,
            configs=configs, results_root=tmp_path / "results")


def test_a_raster_arm_runs_end_to_end_and_resumes(split_chipset, configs, tmp_path):
    exp = configs / "experiments" / "tiny.yaml"
    rows = run(exp, split_chipset, arms=["tiny_tessera"], configs=configs,
               results_root=tmp_path / "results")
    out = tmp_path / "results" / "tiny" / "SYN_2021"
    cell = out / "tiny_tessera" / "P50_draw0_seed0"
    r = json.loads((cell / "results.json").read_text())
    assert r["provenance"]["experiment"] == "tiny" and r["provenance"]["machine"] == "hub"
    assert len(r["provenance"]["git_commit"]) == 40
    z = np.load(cell / PREDICTIONS)
    assert sorted(z["ids"]) == ["EE_00008_00000", "EE_00009_00000"] and z["pred"].shape == (2, 32, 32)
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary["pct"]) == [100, 50] and summary["macro_f1"].notna().all()
    assert [x["state"] for x in rows] == ["done", "done"]
    stamp = (cell / "results.json").stat().st_mtime_ns
    run(exp, split_chipset, arms=["tiny_tessera"], configs=configs, results_root=tmp_path / "results")
    assert (cell / "results.json").stat().st_mtime_ns == stamp          # skipped, not refitted
    (cell / PREDICTIONS).unlink()
    run(exp, split_chipset, arms=["tiny_tessera"], configs=configs, results_root=tmp_path / "results")
    assert (cell / PREDICTIONS).exists() and (cell / "results.json").stat().st_mtime_ns == stamp
    # Finished cells fitted for another number of epochs are not taken for this experiment's.
    exp.write_text(yaml.safe_dump({**yaml.safe_load(exp.read_text()), "epochs": 2}))
    with pytest.raises(SystemExit, match="fitted for 1 epochs, the experiment asks for 2"):
        run(exp, split_chipset, arms=["tiny_tessera"], dry_run=True, configs=configs,
            results_root=tmp_path / "results")


def test_the_chip_set_check_asks_for_every_modality_an_arm_reads(split_chipset):
    split = split_chipset / "splits" / SYN_SPLIT
    problems = check_chipset(split_chipset, split, {"s2s1": {"model": {"backbone": "terramind_v1_large_s2s1"}}})
    assert problems == ["s2s1: 6 chips lack _s1rtc.tif, e.g. "
                        "['EE_00000_00000', 'EE_00001_00000', 'EE_00004_00000']"]


TINY_COMPUTED = {"route": "raster",
                 "data": {"normalisation": "chips", "batch_size": 2, "augment": True},
                 "model": {"backbone": "stub_v2", "lr": 1.0e-3, "weight_decay": 0.05,
                           "head_dropout": 0.2, "loss": "ce"},
                 "trainer": {"precision": "32-true", "log_every_n_steps": 1}}


def test_a_computed_arm_encodes_once_fits_from_its_cache_and_records_it(
        split_chipset, configs, tmp_path, stub_computed):
    (configs / "arms" / "tiny_stub.yaml").write_text(yaml.safe_dump(TINY_COMPUTED))
    exp = configs / "experiments" / "stub.yaml"
    exp.write_text(yaml.safe_dump({"name": "stub", "arms": ["tiny_stub"], "budgets_pct": [100],
                                   "draws": [0], "seeds": [0], "epochs": 1, "checkpoint": "val_loss"}))
    results = tmp_path / "results"
    rows = run(exp, split_chipset, dry_run=True, configs=configs, results_root=results)
    assert rows[0]["cache"] == "compute" and not stub_computed.encoded
    run(exp, split_chipset, configs=configs, results_root=results)
    cache = tmp_path / "emb" / "stub_v2" / "SYN_2021"
    assert (cache / "cache.json").exists() and len(stub_computed.encoded) == 6
    cell = results / "stub" / "SYN_2021" / "tiny_stub" / "P100_draw0_seed0"
    r = json.loads((cell / "results.json").read_text())
    assert Path(r["config"]["data"]["image_root"]).resolve() == cache.resolve()
    assert r["provenance"]["raster_cache"]["backbone"] == "stub_v2"
    assert r["provenance"]["raster_cache"]["encoded_now"] == 6
    z = np.load(cell / PREDICTIONS)
    assert sorted(z["ids"]) == ["EE_00008_00000", "EE_00009_00000"] and z["pred"].shape == (2, 32, 32)
    # A rerun reuses the cache: predictions are redone without encoding again.
    (cell / PREDICTIONS).unlink()
    rows = run(exp, split_chipset, configs=configs, results_root=results)
    assert rows[0]["cache"] == "reuse" and len(stub_computed.encoded) == 6 and (cell / PREDICTIONS).exists()


def test_a_computed_arm_needs_the_rasters_its_encoder_reads(split_chipset):
    problems = check_chipset(split_chipset, split_chipset / "splits" / SYN_SPLIT,
                             {"v2": {"model": {"backbone": "tessera_v2_medium"}}})
    assert problems == ["v2: 6 chips lack _s1rtc.tif, e.g. "
                        "['EE_00000_00000', 'EE_00001_00000', 'EE_00004_00000']"]
