# The two-head workflow in the core: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or
> superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move `experiments/2026-11-09_borders_all_arms` into the core as `gfm4agri.borders`, a second workflow beside
the crop-only K-shot workflow, reached by `scripts/run_kshot.py` from a spec with `workflow: borders`, with tests,
proof that it reproduces the experiment, the docs, and a results report.

**Architecture:** The experiment's modules are copied into `src/gfm4agri/borders/` with their imports made absolute and
their unused branches (other extents, other experiments' sources) removed; behaviour is otherwise unchanged, so the
experiment's results stay valid. The runner takes the core runner's helpers (config, caches, split, machine) and
keeps the experiment's own fit loop.

**Tech Stack:** Python 3.11, PyTorch, segmentation-models-pytorch 0.5.0 (through TerraTorch's lock), scikit-image,
SciPy, rasterio, pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-kshot-borders-design.md`

## Global Constraints

- The crop-only workflow (`configs/experiments/kshot.yaml`, `gfm4agri.pipeline`, `gfm4agri.benchmark`) keeps its
  behaviour; every existing test passes.
- Stage files by name, never `git add -A`; other sessions have uncommitted edits in this checkout
  (`src/gfm4agri/data/chip_grid.py`, `tests/test_chip_grid.py`, `scripts/hub/build_*`, `docs/phase1/pipeline_page/*`,
  `configs/class_scheme_eurocropsml.yaml`, `docs/research/transferability/*`), which are never staged.
- Commits end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`; push to `origin main`.
- Run everything through `poetry run` with `export PATH=$HOME/.local/bin:$PATH`.
- The hub GPU is taken by another process: checks run on the CPU.
- Never write into `results/2026-11-09_borders_all_arms/` or `data/`.

---

### Task 1: PRUE's targets, loss and fields, `gfm4agri.borders.prue`

**Files:**
- Create: `src/gfm4agri/borders/__init__.py`, `src/gfm4agri/borders/prue.py`
- Test: `tests/test_borders_prue.py`

**Interfaces:**
- Produces: `CLASSES`, `BACKGROUND, INTERIOR, BOUNDARY, IGNORE`, `CLASS_WEIGHTS`, `AGRI`, `MIN_PIXELS`, `VARIANTS`,
  `boundary_map(parcels) -> bool array`, `target(mask, parcels, worldcover, keep=None) -> uint8`,
  `d4(arrays, k, flip) -> list`, `log_cosh_dice(logits, y, weights=CLASS_WEIGHTS, ignore=IGNORE) -> Tensor`,
  `build_unet(in_channels, encoder="efficientnet-b7", weights="imagenet")`, `pieces(labels)`,
  `parcels(prob, variant, min_pixels=MIN_PIXELS) -> int32`.

- [ ] **Step 1: Write the failing tests** (`tests/test_borders_prue.py`)

```python
"""PRUE's border target, loss and field post-processing in the core (gfm4agri.borders.prue)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gfm4agri.borders import prue


def _two_parcels():
    parcels = np.zeros((8, 8), np.int64)
    parcels[:, :4], parcels[:, 4:] = 1, 2
    mask = np.where(parcels > 0, 0, -1)
    return mask, parcels


def test_boundary_map_is_two_pixels_between_parcels_and_none_at_the_chip_edge():
    _, parcels = _two_parcels()
    b = prue.boundary_map(parcels)
    assert b[:, 3].all() and b[:, 4].all()
    assert not b[:, :3].any() and not b[:, 5:].any()


def test_target_classes_and_budget():
    mask, parcels = _two_parcels()
    parcels[:2, :] = 0
    mask[:2, :] = -1
    wc = np.full((8, 8), 10, np.uint8)               # trees: background outside the parcels
    y = prue.target(mask, parcels, wc)
    assert (y[:2] == prue.BACKGROUND).all()
    assert y[5, 1] == prue.INTERIOR and y[5, 3] == prue.BOUNDARY
    y_k = prue.target(mask, parcels, wc, keep=np.array([1]))
    assert y_k[5, 1] == prue.INTERIOR and y_k[5, 6] == prue.BACKGROUND   # undrawn parcel on non-farm land
    wc[:] = 40                                         # cropland: unknown outside the drawn parcels
    assert prue.target(mask, parcels, wc, keep=np.array([1]))[5, 6] == prue.IGNORE


def test_log_cosh_dice_is_zero_for_a_perfect_prediction_and_ignores_the_ignore_class():
    y = torch.tensor([[[0, 1], [2, 3]]])
    logits = torch.full((1, 3, 2, 2), -20.0)
    for c in range(3):
        logits[0, c][y[0] == c] = 20.0
    assert float(prue.log_cosh_dice(logits, y)) == pytest.approx(0.0, abs=1e-6)
    worse = logits.clone()
    worse[0, :, 0, 0] = torch.tensor([-20.0, 20.0, -20.0])
    assert float(prue.log_cosh_dice(worse, y)) > 0


def test_parcels_variants_split_two_fields():
    prob = np.zeros((3, 8, 8), np.uint8)
    prob[prue.INTERIOR] = 255
    prob[:, :, 3:5] = 0
    prob[prue.BOUNDARY, :, 3:5] = 255
    cc = prue.parcels(prob, "cc")
    grow = prue.parcels(prob, "grow")
    field = prue.parcels(prob, "field")
    assert cc.max() == 2 and (cc[:, 3:5] == 0).all()
    assert grow.max() == 2 and (grow > 0).all()
    assert field.max() == 1
```

- [ ] **Step 2: Run, expect `ModuleNotFoundError: gfm4agri.borders`**

Run: `poetry run pytest -q tests/test_borders_prue.py`

- [ ] **Step 3: Move the module**

```bash
mkdir -p src/gfm4agri/borders
cp experiments/2026-11-09_borders_all_arms/prue.py src/gfm4agri/borders/prue.py
```

`__init__.py`: a docstring naming the workflow and its modules. In `prue.py`, rewrite the module docstring's
first paragraph to say it is the core's PRUE module (from the experiment), keep everything else.

- [ ] **Step 4: Run the tests, expect PASS**; **Step 5: commit with Task 2** (one commit for the package's model half).

### Task 2: The two-head networks and the loss balance, `gfm4agri.borders.networks`

**Files:**
- Create: `src/gfm4agri/borders/networks.py` (from `heads.py`)
- Test: `tests/test_borders_networks.py`

**Interfaces:**
- Consumes: `prue.build_unet`, `prue.log_cosh_dice`.
- Produces: `IGNORE_CROP = -1`, `crop_target(mask, parcels, keep=None)`, `CropHead`, `decoder_channels(unet)`,
  `UNetJoint(in_channels, n_classes, *, encoder, weights, hidden, dropout, head_dropout)` with `forward -> (seg, crop)`,
  `shared_parameters()`, `groups() -> {"unet", "crop_head"}`; `TokenJoint(neck, decoder, channel_list, out_size,
  n_classes, *, width=64, head_dropout=0.1)` with `groups() -> {"decoder", "crop_head"}`; `crop_ce(logits, yc)`,
  `losses(...)`, `shared_grad_norms(params, lb, lc) -> (float, float)`, `Balance(target, every, smooth, lo, hi)`.

- [ ] **Step 1: Write the failing tests** (`tests/test_borders_networks.py`)

```python
"""The two-head networks and the crop-loss balance (gfm4agri.borders.networks)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from gfm4agri.borders import networks, prue  # noqa: E402


def test_crop_target_only_on_drawn_in_scheme_parcels():
    mask = np.array([[0, 1], [-1, 1]])
    parcels = np.array([[1, 2], [3, 0]])
    assert networks.crop_target(mask, parcels).tolist() == [[0, 1], [-1, -1]]
    assert networks.crop_target(mask, parcels, keep=np.array([2])).tolist() == [[-1, 1], [-1, -1]]


def test_unet_joint_shapes_and_the_crop_loss_reaches_the_shared_part_not_the_border_head():
    torch.manual_seed(0)
    net = networks.UNetJoint(6, 4, encoder="resnet18", weights=None, hidden=(8,), dropout=0.0, head_dropout=0.0)
    x = torch.randn(2, 6, 32, 32)
    seg, crop = net(x)
    assert tuple(seg.shape) == (2, 3, 32, 32) and tuple(crop.shape) == (2, 4, 32, 32)
    yc = torch.randint(0, 4, (2, 32, 32))
    net.zero_grad()
    networks.crop_ce(crop, yc).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.shared_parameters())
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in net.unet.segmentation_head.parameters())
    assert set(net.groups()) == {"unet", "crop_head"}


def test_crop_ce_is_zero_with_a_graph_when_nothing_is_labelled():
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    loss = networks.crop_ce(logits, torch.full((1, 2, 2), networks.IGNORE_CROP))
    assert float(loss) == 0.0
    loss.backward()


def test_balance_sets_the_weight_to_the_target_pull_and_smooths_it():
    b = networks.Balance(target=0.3, every=25, smooth=0.5)
    assert b.due(7)                                    # the first step always measures
    b.update(gb=2.0, gc=100.0)
    assert b.weight == pytest.approx(0.3 * 2.0 / 100.0)
    assert not b.due(7) and b.due(25)
    b.update(gb=2.0, gc=25.0)                          # new value 0.024, blended in log space with 0.006
    assert b.weight == pytest.approx(np.sqrt(0.006 * 0.024))
    assert b.log[-1][2] == pytest.approx(0.006 * 25.0 / 2.0)   # the pull under the weight in force
    s = b.epoch_summary()
    assert s["balance/n"] == 2 and not b.log
    b.update(gb=float("nan"), gc=1.0)                  # ignored
    assert b.weight == pytest.approx(np.sqrt(0.006 * 0.024))
```

- [ ] **Step 2: Run, expect `ImportError`**: `poetry run pytest -q tests/test_borders_networks.py`
- [ ] **Step 3: Move the module**

```bash
cp experiments/2026-11-09_borders_all_arms/heads.py src/gfm4agri/borders/networks.py
sed -i 's/^import prue$/from gfm4agri.borders import prue/' src/gfm4agri/borders/networks.py
```

Docstring: first line "The networks of the two-head workflow ...", and its reference to
`../2026-11-07_prue_crop_tessera` becomes `experiments/2026-11-07_prue_crop_tessera`.

- [ ] **Step 4: Run both test files, expect PASS**
- [ ] **Step 5: Commit**

```bash
git add src/gfm4agri/borders/__init__.py src/gfm4agri/borders/prue.py src/gfm4agri/borders/networks.py \
        tests/test_borders_prue.py tests/test_borders_networks.py
git commit -m "Two-head workflow, part 1: PRUE's target, loss and fields and the two-head networks in gfm4agri.borders ..."
```

### Task 3: The scores, `gfm4agri.borders.scoring`

**Files:**
- Create: `src/gfm4agri/borders/scoring.py`, from `panoptic_x.py` (void-rule matching and PQ), `score_prue.py` (the
  fields of each variant on the `chip` extent, the pixel scores), `borders.py` (border P / R / F1, types, places),
  `score_joint.py` (crop scores and `score_file`).
- Test: `tests/test_borders_scoring.py`

**What is kept and dropped:**
- Kept from `panoptic_x`: `IOU_MATCH`, `FOUR`, `FP_WC`, `pieces` (use `prue.pieces`), `vote`, `match`,
  `chip_counts` (renamed `field_counts`), `quality`, `truth`, `_add`. The `RECORD` dict is rewritten to name the
  chip extent and the three PRUE variants only.
- Dropped from `panoptic_x`: `TAU`, `TAUS`, `WORLDCOVER`, `EXTENTS`, `from_boundary`, `extent_of`, `variants`,
  `segments_in`, `chip_parts`, `_score_chunk`, `summarise`'s extent branch, `score_predictions`.
- `score_prue.chip_parts(prob, mask, parcels, worldcover, crop_pred, n_classes, variants)` keeps the pixel scores
  and loops the variants over the chip only: `seg = inst` (the `chip` extent is the whole chip, so
  `np.where(X, inst, 0)` is `inst`). Keys stay `chip/<variant>` so `results.json` reads as before.
- `score(probs, masks, ids, chips_dir, class_names, worldcover, crop=None, variants=prue.VARIANTS, workers=8)`
  and `fields_file(predictions, chips_dir, class_names, worldcover, cropmap=None, workers=8)` as `score_prue.score`
  and `score_prue.score_file`, without `extents`.
- From `borders.py`: `TOLERANCES`, `TYPES`, `PLACES`, `CHECK`, `regions`, `border_types`, `_near`,
  `chip_counts` (renamed `border_counts`), `_f1`, and `_chunk` with `fields(kind, a)` reduced to
  `prue.parcels(a, "grow")`. Dropped: `NAME`, `SOURCES`, `CELL`, `load`, `main`.
- From `score_joint.py`: `crop_scores`, `_border_row`, `border_scores`, `score_file` (returns
  `{"crop", "panoptic", "borders"}`, `panoptic` from `fields_file(..., cropmap=predictions)`).

- [ ] **Step 1: Write the failing tests** (`tests/test_borders_scoring.py`)

```python
"""Field and border scores of the two-head workflow (gfm4agri.borders.scoring)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from gfm4agri.borders import prue, scoring  # noqa: E402


def _chip():
    parcels = np.zeros((16, 16), np.int64)
    parcels[2:14, 2:8], parcels[2:14, 8:14] = 1, 2
    mask = np.where(parcels > 0, np.where(parcels == 1, 0, 1), -1)
    wc = np.full((16, 16), 10, np.uint8)
    return mask, parcels, wc


def _perfect_prob(mask, parcels, wc):
    y = prue.target(mask, parcels, wc)
    prob = np.zeros((3, *y.shape), np.uint8)
    for c in range(3):
        prob[c][y == c] = 255
    return prob


def test_the_declared_parcels_score_one_against_themselves():
    mask, parcels, wc = _chip()
    E, scored, true_border = scoring.regions(mask, parcels, wc)
    types = scoring.border_types(parcels, mask, wc, true_border)
    c = scoring.border_counts(parcels, true_border, scored, E, types)
    for tp_p, n_p, tp_r, n_r in c["tol"]:
        assert tp_p == n_p and tp_r == n_r and n_r > 0


def test_a_perfect_prediction_has_pq_one_class_agnostic_and_class_aware():
    mask, parcels, wc = _chip()
    prob = _perfect_prob(mask, parcels, wc)
    parts = scoring.chip_parts(prob, mask, parcels, wc, mask.clip(0), 2)
    acc: dict = {}
    scoring._add(acc, parts)
    q = scoring.quality(*acc["chip/grow"], ["a", "b"])
    assert q["pq_agnostic"] == pytest.approx(1.0) and q["pq"] == pytest.approx(1.0)
    assert q["true_parcels"] == 2 and q["predicted_parcels"] == 2


def test_a_merged_prediction_misses_both_parcels():
    mask, parcels, wc = _chip()
    merged = np.where(parcels > 0, 1, 0)
    prob = _perfect_prob(mask, merged, wc)
    parts = scoring.chip_parts(prob, mask, parcels, wc, mask.clip(0), 2)
    acc: dict = {}
    scoring._add(acc, parts)
    q = scoring.quality(*acc["chip/grow"], ["a", "b"])
    assert q["tp_agnostic"] == 0 and q["fn_agnostic"] == 2
```

- [ ] **Step 2: Run, expect `ImportError`**: `poetry run pytest -q tests/test_borders_scoring.py`
- [ ] **Step 3: Write `scoring.py`** by copying the kept functions verbatim (bodies unchanged except the renames
  and `px.`/`prue.` prefixes becoming module-local or `prue.`), in the order: constants, field matching and
  quality, PRUE fields per chip, borders, crop scores, `score_file`.
- [ ] **Step 4: Run, expect PASS**
- [ ] **Step 5: Commit** `scoring.py` and its test.

### Task 4: Chips with both targets, and one cell's fit, `gfm4agri.borders.data` and `.fit`

**Files:**
- Create: `src/gfm4agri/borders/data.py` (from `chipsets.py`), `src/gfm4agri/borders/fit.py` (from `fit_borders.py`)
- Test: `tests/test_borders_fit.py`

**Edits on copy:** `import heads` / `import prue` become `from gfm4agri.borders import networks, prue` with
`heads.` renamed `networks.`; `from chipsets import ...` becomes `from gfm4agri.borders.data import ...`; in
`fit.validate`, `import score_prue` and `score_prue.score(..., extents=("chip",), ...)` become
`from gfm4agri.borders import scoring` and `scoring.score(...)`. Nothing else changes.

- [ ] **Step 1: Write the failing test** (`tests/test_borders_fit.py`): on the synthetic chip set (`split_chipset`,
  32 x 32 chips, a four-dimension `tessera_v1` raster in the chip set), a WorldCover file written to `tmp_path`,
  `fit.fit` on a `raster` source with `UNetJoint(encoder="resnet18", weights=None)`, one epoch, batch 2, CPU, must
  write `results.json` (with `best_epoch == 1`, `final_crop_weight` finite) and `predictions_test.npz` (`prob`
  `(2, 3, 32, 32)` uint8, `pred`, `mask`, `ids`); a second call with `last.pt` present is not needed (resume is
  the experiment's, unchanged).

```python
"""One cell of the two-head workflow end to end on the synthetic chip set (gfm4agri.borders.fit)."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("segmentation_models_pytorch")
pytest.importorskip("rasterio")

from conftest import SYN_BLOCKS  # noqa: E402
from gfm4agri.borders import fit  # noqa: E402
from gfm4agri.pipeline.config import resolve_split  # noqa: E402


def _worldcover(tmp_path):
    ids = [c for _, cids in SYN_BLOCKS.values() for c in cids]
    path = tmp_path / "worldcover.npz"
    np.savez_compressed(path, **{c: np.full((32, 32), 10, np.uint8) for c in ids})
    return path


def test_one_cell_fits_predicts_and_records(split_chipset, tmp_path):
    split_dir = resolve_split(split_chipset)
    source = {"kind": "raster", "backbone": "tessera_v1", "dir": str(split_chipset), "suffix": "_tessera.tif",
              "means": [0.0] * 4, "stds": [1.0] * 4, "channels": 4, "cache": "chip set"}
    cfg = {"run_name": "tiny", "workflow": "test", "input": "raster", "seed": 0, "epochs": 1,
           "label_budget": {"mode": "pct", "pct": 100, "draw_seed": 0},
           "data": {"batch_size": 2, "eval_batch_size": 2, "augment": True, "num_workers": 0},
           "unet": {"encoder": "resnet18", "weights": None},
           "crop": {"hidden": [8], "dropout": 0.0, "head_dropout": 0.0},
           "optim": {"unet": {"lr": 1e-3, "weight_decay": 0.01, "amsgrad": True, "schedule": "cosine",
                              "min_lr": 1e-6},
                     "crop_head": {"lr": 1e-3, "weight_decay": 0.05, "schedule": "cosine", "min_lr": 1e-6}},
           "loss": {"class_weights": [0.05, 0.20, 0.75]}, "balance": {"target": 0.3, "every": 1, "smooth": 0.5}}
    cell = tmp_path / "cell"
    r = fit.fit(cfg, cell, source, split_chipset, split_dir, _worldcover(tmp_path), num_workers=0, device="cpu",
                log=lambda m: None)
    assert r["best_epoch"] == 1 and np.isfinite(r["final_crop_weight"])
    assert json.loads((cell / "results.json").read_text())["run"] == "tiny"
    with np.load(cell / "predictions_test.npz") as z:
        assert z["prob"].shape == (2, 3, 32, 32) and z["prob"].dtype == np.uint8
        assert z["pred"].shape == (2, 32, 32) and len(z["ids"]) == 2
    assert not (cell / "last.pt").exists()
```

- [ ] **Step 2: Run, expect `ImportError`**: `poetry run pytest -q tests/test_borders_fit.py`
- [ ] **Step 3: Copy and edit both modules as above.**
- [ ] **Step 4: Run, expect PASS.** If the synthetic chips are too small for the scoring's `MIN_PIXELS` or the
  U-Net's depth, change the test's chip set use, never the module.
- [ ] **Step 5: Commit** `data.py`, `fit.py` and the test.

### Task 5: The runner, the configs and the entry point

**Files:**
- Create: `src/gfm4agri/borders/runner.py` (from `run.py`), `configs/experiments/kshot_borders.yaml`,
  `configs/arms_borders/{terramind_v1_large_s2s1,prithvi_eo_v2_600_tl,thor_v1_large_s2s1,tessera_v2_medium,alphaearth_v1,unet_b7_s2s1}.yaml`
- Modify: `scripts/run_kshot.py` (dispatch, `--device`, `--rescore`), `src/gfm4agri/pipeline/config.py` only if a
  helper is missing (none expected)
- Test: `tests/test_borders_runner.py`

**`runner.py`, from `run.py`:**
- Drop the `sys.path` and `REPO` discovery; take `REPO`, `CONFIGS`, `load_arm`, `load_machine`, `resolve_split`,
  `encode_batch_size`, `repo_relative` from `gfm4agri.pipeline.config`.
- `load(spec, configs=CONFIGS) -> (exp, arms)`: arm files from `<spec dir>/arms_borders/<arm>.yaml`, else
  `<configs>/arms_borders/<arm>.yaml`; `kshot_arm` through `load_arm(name, configs)`.
- `worldcover_for(chips) -> Path`: `REPO/data/worldcover/<chips.name>/worldcover.npz`, else the parent's
  (`subset.json`'s `parent`), else `SystemExit` naming `scripts/hub/build_worldcover_chips.py`. The spec no longer
  names the file.
- `rasters.frozen` (TESSERA v2's raster cache) is replaced by the core's
  `gfm4agri.pipeline.raster_cache.resolve_raster_cache` and `build_raster_cache` used exactly as `rasters.frozen`
  uses them (read `rasters.frozen` first and keep its outputs: dir, suffix, means, stds, record).
- `run(spec, chips, *, split=None, arms=None, machine="hub", cache_root=None, scratch=None, device=None,
  dry_run=False, configs=CONFIGS, results_root=REPO/"results") -> list[dict]` with one row per cell
  (`arm`, `cell`, `state`, `cache`), the body of `run.main`'s loop; `rescore(out_root, chips, every=False)`;
  `summarise(out_root)` returning the DataFrame.
- `entry` in provenance: `scripts/run_kshot.py`; `workflow: borders`.

**`kshot_borders.yaml`:** the experiment's `experiment.yaml` with `name: kshot_borders`, `workflow: borders`, no
`worldcover` key, its comments rewritten for the core.

**Arm files:** the experiment's `arms_run/*.yaml`, comments rewritten ("Read by gfm4agri.borders.runner").

**`scripts/run_kshot.py`:** after parsing, `exp = load_experiment(args.experiment)`; when
`exp.get("workflow") == "borders"`, call `gfm4agri.borders.runner` (`rescore` + `summarise` on `--summarise`, else
`run`) and return. Add `--device` and `--rescore` (used by the borders workflow only; the docstring says so).
`load_experiment` must accept the key: check that it does not reject unknown keys.

- [ ] **Step 1: Write the failing tests** (`tests/test_borders_runner.py`)

```python
"""The two-head workflow's spec, arms and dry run (gfm4agri.borders.runner)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml

from gfm4agri.borders import runner
from gfm4agri.pipeline.config import CONFIGS, REPO


def test_the_core_spec_loads_six_arms_each_on_its_crop_only_arm():
    exp, arms = runner.load(CONFIGS / "experiments" / "kshot_borders.yaml")
    assert exp["name"] == "kshot_borders" and exp["workflow"] == "borders"
    assert exp["budgets_pct"] == [100, 20, 5] and exp["epochs"] == 45
    assert exp["balance"] == {"target": 0.3, "every": 25, "smooth": 0.5}
    assert set(arms) == {"terramind_v1_large_s2s1", "prithvi_eo_v2_600_tl", "thor_v1_large_s2s1",
                         "tessera_v2_medium", "alphaearth_v1", "unet_b7_s2s1"}
    assert {a["input"] for a in arms.values()} == {"raster", "stack", "token"}
    assert arms["unet_b7_s2s1"]["_kshot"]["model"]["backbone"] == "unet_r18_scratch_s2s1"


def test_the_core_arms_equal_the_experiments():
    exp_dir = REPO / "experiments" / "2026-11-09_borders_all_arms"
    if not exp_dir.exists():
        return
    for f in (CONFIGS / "arms_borders").glob("*.yaml"):
        assert yaml.safe_load(f.read_text()) == yaml.safe_load((exp_dir / "arms_run" / f.name).read_text())
    core = yaml.safe_load((CONFIGS / "experiments" / "kshot_borders.yaml").read_text())
    old = yaml.safe_load((exp_dir / "experiment.yaml").read_text())
    for k in ("arms", "budgets_pct", "draws", "seeds", "epochs", "loss", "balance"):
        assert core[k] == old[k]


def test_an_experiment_overrides_an_arm_from_its_own_folder(tmp_path):
    spec = tmp_path / "experiment.yaml"
    spec.write_text(yaml.safe_dump({"name": "x", "workflow": "borders", "arms": ["tessera_v2_medium"],
                                    "budgets_pct": [100], "draws": [0], "seeds": [0], "epochs": 1,
                                    "loss": {"class_weights": [0.05, 0.2, 0.75]},
                                    "balance": {"target": 0.3, "every": 25}}))
    (tmp_path / "arms_borders").mkdir()
    arm = yaml.safe_load((CONFIGS / "arms_borders" / "tessera_v2_medium.yaml").read_text())
    arm["token"] = {"width": 32}
    (tmp_path / "arms_borders" / "tessera_v2_medium.yaml").write_text(yaml.safe_dump(arm))
    _, arms = runner.load(spec)
    assert arms["tessera_v2_medium"]["token"] == {"width": 32}


def test_worldcover_of_a_pilot_set_is_its_parents(tmp_path, monkeypatch):
    parent, pilot = tmp_path / "chips" / "P_2021", tmp_path / "chips" / "P_2021_mini"
    pilot.mkdir(parents=True)
    (pilot / "subset.json").write_text(json.dumps({"parent": str(parent)}))
    wc = tmp_path / "data" / "worldcover" / "P_2021"
    wc.mkdir(parents=True)
    np.savez(wc / "worldcover.npz", a=np.zeros(1))
    monkeypatch.setattr(runner, "REPO", tmp_path)
    assert runner.worldcover_for(pilot) == wc / "worldcover.npz"
```

- [ ] **Step 2: Run, expect `ImportError`**: `poetry run pytest -q tests/test_borders_runner.py`
- [ ] **Step 3: Write `runner.py`, the configs and the dispatch.**
- [ ] **Step 4: Run the new tests, then the whole suite** (`poetry run pytest -q`), then the dry runs:

```bash
poetry run python scripts/run_kshot.py -e configs/experiments/kshot_borders.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --dry-run
poetry run python scripts/run_kshot.py -e configs/experiments/kshot_borders.yaml --chips data/eurocrops_chips/EE_2021_rev --dry-run
poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --dry-run
```

Expected: 18 cells for `kshot_borders` (6 arms x 3 budgets), all `fit`; the crop-only dry run unchanged.
- [ ] **Step 5: Commit** the runner, configs, `run_kshot.py` and the test.

### Task 6: WorldCover as a chip-set product

**Files:**
- Create: `scripts/hub/build_worldcover_chips.py` (from `experiments/2026-11-05_parcel_extent/fetch_worldcover.py`)

The body is unchanged; the docstring's run line becomes
`poetry run python scripts/hub/build_worldcover_chips.py --chips EE_2021_rev --workers 16`, its `HERE`/`sys.path`
lines become the `REPO = Path(__file__).resolve().parents[2]` form of the other hub builders, and it says the
two-head workflow reads its output. The experiment's copy stays (its folder is the record of that experiment).

- [ ] **Step 1:** copy, edit, `poetry run python scripts/hub/build_worldcover_chips.py --help` exits 0.
- [ ] **Step 2: Commit** with Task 5 or alone.

### Task 7: Proof that the core reproduces the experiment

**Files:**
- Create: `experiments/2026-11-09_borders_all_arms/check_core.py` (it imports the experiment's modules, so it lives
  with them; the spec's `scripts/hub/check_borders_core.py` is moved here for that reason)

**Check 1, scoring.** For each of the 16 cells under `results/2026-11-09_borders_all_arms/EE_2021_rev/`, score
`predictions_test.npz` with `gfm4agri.borders.scoring.score_file` (into memory, nothing written) and compare with the
cell's `results.json`: `crop_metrics` (`macro_f1`, `miou`, `accuracy`, `confusion`), `parcel_metrics`,
`panoptic_metrics["variants"]["chip/<v>"]` for every variant, `panoptic_metrics["pixels"]`, `border_metrics` (every
tolerance, types, places). Exact equality of every number (`==` on floats from the same arithmetic).

**Check 2, training.** On `EE_2021_rev_mini`, K = 100 %, 2 epochs, seed 0, CPU, `num_workers` 0, for
`tessera_v2_medium`, `unet_b7_s2s1` and `terramind_v1_large_s2s1`: the experiment's `run.source_for` +
`fit_borders.fit` into `<scratch>/exp/<arm>`, the core's `runner` source + `fit.fit` into `<scratch>/core/<arm>`, both
from the same config dict; compare `history` (every key, exact) and the `best.pt` state dicts (`torch.equal` on
every tensor). The token arm's cache must exist on the hub for the mini set (`data/embeddings/<backbone>/EE_2021_rev_mini/`);
skip an arm whose cache is missing and say so.

- [ ] **Step 1:** write `check_core.py` with `--scoring`, `--training`, `--arms`, exiting 1 on any difference.
- [ ] **Step 2:** run `--scoring` (all 16 cells, about 2 to 5 minutes per cell on 16 workers), expect 0 differences.
- [ ] **Step 3:** run `--training`, expect identical histories and weights.
- [ ] **Step 4: Commit** `check_core.py` with its output summarised in the commit message.

### Task 8: Documentation

**Files:** `docs/phase1/protocol.md` (D30), `docs/phase1/pipeline.md` (section 1 table, a new stage section,
WorldCover, section 10, section 11), `docs/phase1/pipeline_overview.md`, `CLAUDE.md`,
`docs/phase1/pipeline_page/build_page.py` and `index.html` (rebuilt), the experiment's README (a line that its setup is
now the core's `kshot_borders`).

- [ ] **Step 1:** read each doc's relevant sections and the page README; write the changes as the spec's section 7.
- [ ] **Step 2:** rebuild the page (`poetry run python docs/phase1/pipeline_page/build_page.py`), check it as its
  README says.
- [ ] **Step 3:** commit the docs; for `build_page.py` and `index.html`, stage only this change's hunks: save the
  other session's diff (`git diff -- <file> > other.patch` before editing), and after editing stage with
  `git diff -- <file> | ...` minus `other.patch` through `git apply --cached` of this change's hunks only. If the
  hunks overlap, stop and tell the user.

### Task 9: The report

**Files:** `docs/phase1/results_estonia_borders/{REPORT.md,report.py,maps.py,tables.md,figures/}`

- [ ] **Step 1:** read `docs/phase1/results_estonia/report.py` and `maps.py`; write `report.py` reading the sources
  of the spec's section 8, writing the figures (borders against K; crop with and without the border head; border
  types; frozen against fine-tuned) and `tables.md`.
- [ ] **Step 2:** write `maps.py` for the tiles and blocks of the crop report's section 3.5, from
  `predictions_test.npz` of the two-head cells.
- [ ] **Step 3:** search the literature for PRUE's and other delineation benchmarks' reported scores; take the
  Estonian crop-type references from the crop report's section 4.
- [ ] **Step 4:** write `REPORT.md` after the crop report's structure; check every number against `tables.md`.
- [ ] **Step 5:** commit and push; give the user the outcome.
