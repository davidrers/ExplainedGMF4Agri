"""The K-shot workflow composes exactly the configurations the full-Estonia grid ran with."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("terratorch")

from gfm4agri.pipeline.config import (  # noqa: E402
    CONFIGS, REPO, Cell, arm_dirs, compose, encode_batch_size, load_arm, load_experiment,
    load_machine, plan_cells, repo_relative, resolve_split)

LEGACY = Path(__file__).parent / "fixtures" / "legacy_configs"
THOR80 = REPO / "experiments" / "2026-10-08_thor_80m"
EE = REPO / "data" / "eurocrops_chips" / "EE_2021"
SPLIT = EE / "splits" / "blocks4_buf1600_seed0__6b0eb4cb"
GRID = {"terramind_v1_large": "terramind_v1_large_ee.yaml",
        "prithvi_eo_v2_600_tl": "prithvi_eo_v2_600_tl_ee.yaml",
        "tessera_v1": "tessera_v1_mlp_ee.yaml",
        "alphaearth_v1": "alphaearth_v1_mlp_ee.yaml"}


def _without_run_specific(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.pop("run_name"), cfg.pop("output_root"), cfg["data"].pop("num_workers")
    return cfg


@pytest.mark.parametrize("pct", [100, 20, 5])
@pytest.mark.parametrize("arm, legacy", GRID.items())
def test_composed_cell_is_the_grid_configuration(arm, legacy, pct):
    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    got = compose(arm, load_arm(arm), exp, load_machine("hub"), EE, SPLIT, Cell(arm, pct, 0, 0),
                  REPO / "results" / "kshot" / "EE_2021")
    # The grid passed the budget, the draw and 15 epochs on the command line.
    want = yaml.safe_load((LEGACY / legacy).read_text())
    want["label_budget"].update(mode="pct", pct=pct, draw_seed=0)
    want["trainer"]["max_epochs"] = 15
    assert _without_run_specific(got) == _without_run_specific(want)
    assert got["run_name"] == arm and got["output_root"] == "results/kshot/EE_2021"
    assert got["data"]["root"] == "data/eurocrops_chips/EE_2021"


def test_thor_arm_keeps_the_160m_settings_of_its_pilot():
    arm, pilot = load_arm("thor_v1_large"), yaml.safe_load(
        (LEGACY / "thor_v1_large_ee_pilot.yaml").read_text())
    assert arm["model"] == pilot["model"]
    assert {k: arm["data"][k] for k in ("normalisation", "batch_size", "augment")} == \
           {k: pilot["data"][k] for k in ("normalisation", "batch_size", "augment")}
    assert arm["trainer"]["accumulate_grad_batches"] == pilot["trainer"]["accumulate_grad_batches"]
    from gfm4agri.benchmark.backbones import get_backbone
    assert get_backbone(arm["model"]["backbone"]).resolution_group == "token_grid"


def test_every_arm_route_matches_its_backbone():
    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    routes = {a: load_arm(a)["route"] for a in exp["arms"]}
    assert routes == {"terramind_v1_large": "cache", "prithvi_eo_v2_600_tl": "cache",
                      "thor_v1_large": "cache", "tessera_v1": "raster", "alphaearth_v1": "raster"}


def test_the_thor_80m_experiment_follows_the_kshot_protocol():
    main = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    add_on = load_experiment(THOR80 / "experiment.yaml")
    assert add_on["arms"] == ["thor_v1_large_80m"] and "thor_v1_large_80m" not in main["arms"]
    for key in ("name", "budgets_pct", "draws", "seeds", "epochs", "checkpoint"):
        assert add_on[key] == main[key], key


def test_thor_80m_arm_keeps_the_settings_of_its_study():
    search = arm_dirs(THOR80 / "experiment.yaml")
    arm, study = load_arm("thor_v1_large_80m", search=search), yaml.safe_load(
        (LEGACY / "thor_v1_large_80m_ee_pilot.yaml").read_text())
    assert arm["model"] == study["model"]
    assert {k: arm["data"][k] for k in ("normalisation", "batch_size", "augment")} == \
           {k: study["data"][k] for k in ("normalisation", "batch_size", "augment")}
    assert arm["trainer"]["accumulate_grad_batches"] == study["trainer"]["accumulate_grad_batches"]
    from gfm4agri.benchmark.backbones import get_backbone
    assert get_backbone(arm["model"]["backbone"]).resolution_group == "token_grid_80m"


def test_every_cache_arm_has_an_encode_batch_on_both_machines():
    # Core arms through the machine profiles; experiment arms may carry their own.
    core = [load_arm(p.stem) for p in sorted((CONFIGS / "arms").glob("*.yaml"))]
    for machine in ("hub", "cluster"):
        profile = load_machine(machine)
        assert {a["model"]["backbone"] for a in core if a["route"] == "cache"} <= \
               set(profile["encode_batch_size"]), machine
        for f in sorted((REPO / "experiments").glob("*/arms/*.yaml")):
            arm = load_arm(f.stem, search=[f.parent])
            if arm["route"] == "cache":
                assert machine in (arm.get("encode_batch_size") or {}) or \
                    arm["model"]["backbone"] in profile["encode_batch_size"], (f, machine)


def test_the_main_workflow_holds_only_its_own_experiment_and_arms():
    assert sorted(p.name for p in (CONFIGS / "experiments").glob("*.yaml")) == ["kshot.yaml"]
    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    assert sorted(p.stem for p in (CONFIGS / "arms").glob("*.yaml")) == sorted(exp["arms"])


def test_an_experiment_folder_arm_shadows_the_core_arm(tmp_path):
    folder = tmp_path / "2026-01-01_try"
    (folder / "arms").mkdir(parents=True)
    arm = yaml.safe_load((CONFIGS / "arms" / "tessera_v1.yaml").read_text())
    arm["model"]["lr"] = 0.5
    (folder / "arms" / "tessera_v1.yaml").write_text(yaml.safe_dump(arm))
    (folder / "experiment.yaml").write_text(yaml.safe_dump({"arms": ["tessera_v1", "alphaearth_v1"]}))
    search = arm_dirs(folder / "experiment.yaml")
    assert load_arm("tessera_v1", search=search)["model"]["lr"] == 0.5        # the experiment's copy
    assert load_arm("alphaearth_v1", search=search)["model"]["lr"] == 1.0e-3  # the core's arm
    assert load_experiment(folder / "experiment.yaml")["name"] == "2026-01-01_try"
    assert arm_dirs(CONFIGS / "experiments" / "kshot.yaml") == [CONFIGS / "arms"]


def test_an_arm_may_carry_its_own_encode_batch():
    arm = {"model": {"backbone": "x"}, "encode_batch_size": {"hub": 3}}
    assert encode_batch_size(arm, {"encode_batch_size": {"x": 7}}, "hub") == 3
    assert encode_batch_size(arm, {"encode_batch_size": {"x": 7}}, "cluster") == 7
    assert encode_batch_size({"model": {"backbone": "y"}}, {"encode_batch_size": {}}, "hub") == 1


def test_a_route_that_contradicts_the_backbone_is_refused(tmp_path):
    (tmp_path / "arms").mkdir()
    bad = yaml.safe_load((CONFIGS / "arms" / "tessera_v1.yaml").read_text()) | {"route": "cache"}
    (tmp_path / "arms" / "bad.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="needs 'raster'"):
        load_arm("bad", configs=tmp_path)


def test_cells_run_budgets_outermost_within_each_arm():
    exp = {"arms": ["a", "b"], "budgets_pct": [100, 5], "draws": [0, 1], "seeds": [0]}
    cells = plan_cells(exp)
    assert [c.tag for c in cells[:4]] == ["P100_draw0_seed0", "P100_draw1_seed0",
                                         "P5_draw0_seed0", "P5_draw1_seed0"]
    assert {c.arm for c in cells[:4]} == {"a"} and len(cells) == 8
    assert [c.arm for c in plan_cells(exp, ["b"])] == ["b"] * 4
    with pytest.raises(ValueError, match="not in the experiment"):
        plan_cells(exp, ["c"])


def test_split_is_the_only_one_or_the_named_one(split_chipset):
    from conftest import SYN_SPLIT

    assert resolve_split(split_chipset).name == SYN_SPLIT
    assert resolve_split(split_chipset, "blocks1_buf0_seed0").name == SYN_SPLIT
    other = split_chipset / "splits" / "other__1"
    other.mkdir()
    (other / "split.json").write_text("{}")
    with pytest.raises(ValueError, match="holds 2 splits"):
        resolve_split(split_chipset)
    assert resolve_split(split_chipset, "other").name == "other__1"


def test_paths_inside_the_repository_are_stored_relative(tmp_path):
    assert repo_relative(REPO / "data" / "x") == "data/x"
    assert repo_relative(tmp_path) == str(tmp_path.resolve())
