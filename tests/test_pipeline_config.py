"""The K-shot workflow composes exactly the configurations the full-Estonia grid ran with."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("terratorch")

from gfm4agri.pipeline.config import (  # noqa: E402
    CONFIGS, REPO, Cell, compose, load_arm, load_experiment, load_machine, plan_cells,
    repo_relative, resolve_split)

LEGACY = Path(__file__).parent / "fixtures" / "legacy_configs"
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
