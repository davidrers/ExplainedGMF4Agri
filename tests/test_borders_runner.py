"""The two-head workflow's spec, arms, WorldCover lookup and runs (gfm4agri.borders.runner)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

pytest.importorskip("terratorch")
pytest.importorskip("segmentation_models_pytorch")

from conftest import SYN_BLOCKS, SYN_H  # noqa: E402
from gfm4agri.borders import runner  # noqa: E402
from gfm4agri.pipeline.config import CONFIGS, REPO  # noqa: E402

TINY_CROP_ARM = {"route": "raster",
                 "data": {"normalisation": "chips", "batch_size": 2, "augment": True},
                 "model": {"backbone": "tessera_v1", "lr": 1.0e-3, "weight_decay": 0.05, "head_dropout": 0.2,
                           "loss": "ce"},
                 "trainer": {"precision": "32-true", "log_every_n_steps": 1}}
TINY_ARM = {"kshot_arm": "tiny_tessera", "input": "raster",
            "data": {"batch_size": 2, "eval_batch_size": 2, "augment": True},
            "unet": {"encoder": "resnet18", "weights": None},
            "crop": {"hidden": [8], "dropout": 0.0, "head_dropout": 0.0},
            "optim": {"unet": {"lr": 1e-3, "weight_decay": 0.01, "amsgrad": True, "schedule": "cosine",
                               "min_lr": 1e-6},
                      "crop_head": {"lr": 1e-3, "weight_decay": 0.05, "schedule": "cosine", "min_lr": 1e-6}}}
TINY_SPEC = {"name": "tiny_borders", "workflow": "borders", "arms": ["tiny"], "budgets_pct": [100, 50],
             "draws": [0], "seeds": [0], "epochs": 1, "loss": {"class_weights": [0.05, 0.20, 0.75]},
             "balance": {"target": 0.3, "every": 1, "smooth": 0.5}}


def test_the_core_spec_loads_six_arms_each_on_its_crop_only_arm():
    exp, arms = runner.load(CONFIGS / "experiments" / "kshot_borders.yaml")
    assert exp["name"] == "kshot_borders" and exp["workflow"] == "borders"
    assert exp["budgets_pct"] == [100, 20, 5] and exp["epochs"] == 45
    assert exp["balance"] == {"target": 0.3, "every": 25, "smooth": 0.5}
    assert set(arms) == {"terramind_v1_large_s2s1", "prithvi_eo_v2_600_tl", "thor_v1_large_s2s1",
                         "tessera_v2_medium", "alphaearth_v1", "unet_b7_s2s1"}
    assert {a["input"] for a in arms.values()} == {"raster", "stack", "token"}
    assert arms["unet_b7_s2s1"]["_kshot"]["model"]["backbone"] == "unet_r18_scratch_s2s1"
    assert arms["unet_b7_s2s1"]["unet"] == {"encoder": "efficientnet-b7", "weights": "imagenet"}


def test_the_core_spec_and_arms_equal_the_experiments():
    exp_dir = REPO / "experiments" / "2026-11-09_borders_all_arms"
    if not exp_dir.exists():
        pytest.skip("the experiment folder is gone")
    for f in (CONFIGS / "arms_borders").glob("*.yaml"):
        assert yaml.safe_load(f.read_text()) == yaml.safe_load((exp_dir / "arms_run" / f.name).read_text()), f.name
    core = yaml.safe_load((CONFIGS / "experiments" / "kshot_borders.yaml").read_text())
    old = yaml.safe_load((exp_dir / "experiment.yaml").read_text())
    for k in ("arms", "budgets_pct", "draws", "seeds", "epochs", "loss", "balance"):
        assert core[k] == old[k], k


def test_a_crop_only_spec_is_refused():
    with pytest.raises(ValueError, match="not a two-head spec"):
        runner.load(CONFIGS / "experiments" / "kshot.yaml")


def test_an_experiment_overrides_an_arm_from_its_own_folder(tmp_path):
    spec = tmp_path / "experiment.yaml"
    spec.write_text(yaml.safe_dump({**TINY_SPEC, "name": "x", "arms": ["tessera_v2_medium"]}))
    (tmp_path / "arms_borders").mkdir()
    arm = yaml.safe_load((CONFIGS / "arms_borders" / "tessera_v2_medium.yaml").read_text())
    arm["crop"]["hidden"] = [64]
    (tmp_path / "arms_borders" / "tessera_v2_medium.yaml").write_text(yaml.safe_dump(arm))
    _, arms = runner.load(spec)
    assert arms["tessera_v2_medium"]["crop"]["hidden"] == [64]


def test_worldcover_of_a_pilot_set_is_its_parents(tmp_path, monkeypatch):
    parent, pilot = tmp_path / "chips" / "P_2021", tmp_path / "chips" / "P_2021_mini"
    pilot.mkdir(parents=True)
    (pilot / "subset.json").write_text(json.dumps({"parent": str(parent)}))
    wc = tmp_path / "data" / "worldcover" / "P_2021"
    wc.mkdir(parents=True)
    np.savez(wc / "worldcover.npz", a=np.zeros(1))
    monkeypatch.setattr(runner, "REPO", tmp_path)
    assert runner.worldcover_for(pilot) == wc / "worldcover.npz"
    with pytest.raises(SystemExit, match="build_worldcover_chips"):
        runner.worldcover_for(tmp_path / "chips" / "Q_2021")


@pytest.fixture()
def tiny(split_chipset, tmp_path, monkeypatch):
    configs = tmp_path / "configs"
    for sub in ("arms", "arms_borders", "experiments", "machines"):
        (configs / sub).mkdir(parents=True)
    (configs / "arms" / "tiny_tessera.yaml").write_text(yaml.safe_dump(TINY_CROP_ARM))
    (configs / "arms_borders" / "tiny.yaml").write_text(yaml.safe_dump(TINY_ARM))
    (configs / "experiments" / "tiny_borders.yaml").write_text(yaml.safe_dump(TINY_SPEC))
    (configs / "machines" / "hub.yaml").write_text(yaml.safe_dump(
        {"cache_root": str(tmp_path / "emb"), "num_workers": 0, "encode_num_workers": 0}))
    wc = tmp_path / "data" / "worldcover" / split_chipset.name
    wc.mkdir(parents=True)
    ids = [c for _, cids in SYN_BLOCKS.values() for c in cids]
    np.savez_compressed(wc / "worldcover.npz", **{c: np.full((SYN_H, SYN_H), 10, np.uint8) for c in ids})
    monkeypatch.setattr(runner, "REPO", tmp_path)
    return configs, split_chipset, tmp_path / "results"


def test_dry_run_lists_every_cell(tiny):
    configs, chips, results = tiny
    rows = runner.run(configs / "experiments" / "tiny_borders.yaml", chips, dry_run=True, configs=configs,
                      results_root=results)
    assert [(r["arm"], r["cell"], r["state"]) for r in rows] == [("tiny", "P100_draw0_seed0", "fit"),
                                                                 ("tiny", "P50_draw0_seed0", "fit")]
    assert not (results / "tiny_borders").exists()


def test_a_run_fits_scores_and_summarises_then_skips(tiny):
    configs, chips, results = tiny
    spec = configs / "experiments" / "tiny_borders.yaml"
    rows = runner.run(spec, chips, device="cpu", configs=configs, results_root=results)
    assert [r["state"] for r in rows] == ["done", "done"]
    out = results / "tiny_borders" / chips.name
    r = json.loads((out / "tiny" / "P100_draw0_seed0" / "results.json").read_text())
    assert r["provenance"]["workflow"] == "borders" and r["border_metrics"]["check_declared_f1_d1"] == 1.0
    assert set(r["panoptic_metrics"]["variants"]) == {"chip/cc", "chip/grow", "chip/field"}
    assert (out / "tiny" / "P100_draw0_seed0" / "borders_per_chip.npz").exists()
    summary = (out / "summary.csv").read_text().splitlines()
    assert len(summary) == 3 and "border_f1_d1" in summary[0]
    again = runner.run(spec, chips, device="cpu", configs=configs, results_root=results)
    assert [r["state"] for r in again] == ["done", "done"]
    assert runner.rescore(out, chips, configs=configs) == []
    assert len(runner.rescore(out, chips, every=True, configs=configs)) == 2
