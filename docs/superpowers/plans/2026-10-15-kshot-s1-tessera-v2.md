# The K-shot arms on S1, and frozen TESSERA v2: implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every GFM arm of `kshot` reads every sensor it was pretrained on (TerraMind, THOR and TESSERA on S2 + S1), and frozen TESSERA v2 Medium, run by the runner on our composites and cached like the token-grid encoders, replaces the v1.0 store rasters.

**Architecture:** THOR gets S1 as a third band group, stacked behind its S2 bands, with its groups averaged. TESSERA v2 becomes a 10 m backbone whose raster the runner computes once per chip set into a cache (`gfm4agri.pipeline.raster_cache`), mirroring `gfm4agri.pipeline.cache`; the datamodule reads its embeddings from the cache and its masks from the chip set, and the existing raster route fits the per-pixel MLP on them.

**Tech Stack:** Python 3.11, PyTorch, TerraTorch, Lightning, rasterio, the THOR TerraTorch extension, pytest.

**Spec:** [docs/superpowers/specs/2026-10-15-kshot-s1-tessera-v2-design.md](../specs/2026-10-15-kshot-s1-tessera-v2-design.md)

## Global Constraints

- Work only in the worktree `/data/private/ExplainedGMF4Agri-kshot-s1`, branch `kshot-s1-tessera-v2` (Task 0). Never edit the main checkout `/data/private/THESIS - ExplainedGMF4Agri` before Task 8: other sessions run code from it.
- Interpreter: `PY=/home/jovyan/.cache/pypoetry/virtualenvs/gfm4agri-7U2rqC_9-py3.11/bin/python`, run from the worktree root. Never `poetry run` in the worktree: Poetry would build a second environment for the new path.
- Stage files by name; never `git add -A`; never stage `data` or `results` (symlinks in the worktree).
- Never delete, overwrite or re-export a chip set or a cache under `data/`.
- TESSERA v2 Medium: file `data/weights/tessera/v2/student_medium.pt`, Hugging Face `geotessera/TESSERA-V-2.0-2B-M`, `ckpt/student_medium.pt`, SHA-256 `3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e`; vendored from `ucam-eo/tessera` commit `e368539f7c4711e5f16fdd09801573830e34baf9` under its MIT licence; encoding precision `bf16-mixed`; 128 dimensions.
- The 10 m group's decoder stays the per-pixel MLP (`DECODERS["pixel_raster"]`); nothing in `decoders.py` or `segmentation.py` changes.
- THOR S2 + S1: S1 bands `VV`, `VH` at 10 m, so patch 16 at 160 m tokens; `backbone_merge_method: mean`.
- Every commit message ends with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Code reads like the surrounding code: module docstrings that give reasons, comments where a choice needs one, the existing naming.
- `docs/thesis_design.md`, `docs/phase1/*.md` and `CLAUDE.md` change only in Task 9, after the user agrees to the text.

---

### Task 0: The worktree

**Files:** none in the repository; `.git/info/exclude` of the main checkout.

- [ ] **Step 1: Create the branch and worktree from `main`**

```bash
cd "/data/private/THESIS - ExplainedGMF4Agri"
git worktree add -b kshot-s1-tessera-v2 /data/private/ExplainedGMF4Agri-kshot-s1 main
```

- [ ] **Step 2: Link the data and results of the main checkout into the worktree, and keep the links out of git**

```bash
WT=/data/private/ExplainedGMF4Agri-kshot-s1
ln -s "/data/private/THESIS - ExplainedGMF4Agri/data" "$WT/data"
ln -s "/data/private/THESIS - ExplainedGMF4Agri/results" "$WT/results"
printf '/data\n/results\n' >> "/data/private/THESIS - ExplainedGMF4Agri/.git/info/exclude"
cd "$WT" && git status --short
```

Expected: `git status --short` prints nothing (`.gitignore` has `/data/*` and `/results/*`, which do not cover a symlink; `info/exclude` does).

- [ ] **Step 3: Run the suite from the worktree as the baseline**

```bash
cd /data/private/ExplainedGMF4Agri-kshot-s1
PY=/home/jovyan/.cache/pypoetry/virtualenvs/gfm4agri-7U2rqC_9-py3.11/bin/python
$PY -m pytest -q 2>&1 | tail -3
```

Expected: every test passes (`tests/conftest.py` puts the worktree's `src` first on the path). Record the count.

---

### Task 1: THOR on S2 + S1, and the chip-set check for every modality

**Files:**
- Modify: `src/gfm4agri/benchmark/backbones.py` (the `BackboneSpec` dataclass; `_thor_stats` and `_thor`; the THOR entries of `BACKBONES`)
- Modify: `src/gfm4agri/benchmark/segmentation_data.py` (the extra-modality block of `EuroCropsSegDataModule.__init__`)
- Modify: `src/gfm4agri/pipeline/runner.py` (`check_chipset`)
- Test: `tests/test_thor_backbone.py`, `tests/test_segmentation_data.py`, `tests/test_pipeline_runner.py`

**Interfaces:**
- Produces: `BackboneSpec.stack_modalities: bool = False`; `THOR_S1_BANDS = ("VV", "VH")`; `THOR_S1RTC` (an `extra_modalities` dict); `_thor(variant, indices, bottleneck, token_m=THOR_TOKEN_M, merge="concat", s1=False)`; registry entry `"thor_v1_large_s2s1"`. `check_chipset` requires every extra modality's suffix for an S2 arm.

- [ ] **Step 1: Write the failing THOR tests** (append to `tests/test_thor_backbone.py`; add `THOR_S1_BANDS` to its import from `gfm4agri.benchmark.backbones`)

```python
def test_s1_joins_as_a_third_group_on_the_same_grid_and_the_groups_are_averaged():
    args = get_backbone("thor_v1_large_s2s1").model_args(list(THOR_BANDS), 12)
    assert args["backbone_model_bands"] == list(THOR_BANDS) + list(THOR_S1_BANDS)
    assert {args["backbone_patch_sizes"][b] for b in THOR_S1_BANDS} == {THOR_TOKEN_M // 10}
    assert args["backbone_merge_method"] == "mean"
    s2 = get_backbone("thor_v1_large").model_args(list(THOR_BANDS), 12)
    assert {k for k in s2 if s2[k] != args[k]} == {
        "backbone_model_bands", "backbone_patch_sizes", "backbone_merge_method"}


def test_s1_statistics_are_thors_own_at_10m_in_db():
    spec = get_backbone("thor_v1_large_s2s1")
    means, stds = spec.extra_modalities["S1RTC"]["stats"](list(THOR_S1_BANDS))
    assert means == pytest.approx([-12.9850, -20.6958])
    assert stds == pytest.approx([5.0062, 5.8688])
    assert spec.stack_modalities and spec.input_bands == THOR_BANDS
    assert spec.stats(["RED"]) == get_backbone("thor_v1_large").stats(["RED"])


def test_averaged_groups_hand_the_decoder_one_group_width_per_month():
    from terratorch.models import EncoderDecoderFactory

    import gfm4agri.benchmark.necks  # noqa: F401  registers ChannelBottleneck
    from gfm4agri.benchmark.backbones import _thor

    n_t = 2
    args = _thor("thor_v1_base", [2, 5, 8, 11], bottleneck=768, merge="mean", s1=True)(list(THOR_BANDS), n_t)
    args["backbone_pretrained"] = False
    model = EncoderDecoderFactory().build_model(
        task="segmentation", **args, decoder="UNetDecoder",
        decoder_channels=[512, 256, 128, 64], num_classes=3).eval()
    x = torch.randn(1, len(THOR_BANDS) + len(THOR_S1_BANDS), n_t, 224, 224)
    with torch.no_grad():
        feats = model.encoder(x)
        out = model(x).output
    # One group's 768 channels per month, as TerraMind hands over one modality's width.
    assert tuple(feats[0].shape) == (1, n_t * 768, 14, 14)
    assert tuple(out.shape) == (1, 3, 224, 224)
```

- [ ] **Step 2: Write the failing datamodule and chip-set tests**

Append to `tests/test_segmentation_data.py`:

```python
def test_a_stacked_modality_reaches_the_encoder_as_one_image(root_s1, monkeypatch):
    import dataclasses

    spec = dataclasses.replace(get_backbone("terramind_v1_small_s2s1"), name="stacked_s2s1",
                               stack_modalities=True)
    monkeypatch.setitem(BACKBONES, "stacked_s2s1", spec)
    dm = EuroCropsSegDataModule(root_s1, "stacked_s2s1", normalisation="backbone", batch_size=1,
                                num_workers=0, augment=False)
    dm.setup("fit")
    img = dm.train_dataset[0]["image"]
    assert tuple(img.shape) == (len(BANDS) + 2, T, H, H)
    assert dm.modality_split is None
    x = img[None]
    assert dm.split_modalities(x) is x
```

In the same file, in `test_registry_fragments_are_well_formed`, replace

```python
        if spec.extra_modalities:
            assert args["backbone_modalities"] == ["S2L2A", *spec.extra_modalities]
            assert args["backbone_merge_method"] == "mean"
            for m, v in spec.extra_modalities.items():
                assert args["backbone_bands"][m] == list(v["bands"])
```

with

```python
        if spec.extra_modalities and spec.stack_modalities:
            # Stacked behind the S2 bands (THOR): the modality's bands end the encoder's band list.
            assert args["backbone_merge_method"] == "mean"
            for v in spec.extra_modalities.values():
                assert args["backbone_model_bands"][-len(v["bands"]):] == list(v["bands"])
        elif spec.extra_modalities:
            assert args["backbone_modalities"] == ["S2L2A", *spec.extra_modalities]
            assert args["backbone_merge_method"] == "mean"
            for m, v in spec.extra_modalities.items():
                assert args["backbone_bands"][m] == list(v["bands"])
```

Append to `tests/test_pipeline_runner.py` (add `from conftest import SYN_SPLIT  # noqa: E402` and `check_chipset` to the runner import):

```python
def test_the_chip_set_check_asks_for_every_modality_an_arm_reads(split_chipset):
    split = split_chipset / "splits" / SYN_SPLIT
    problems = check_chipset(split_chipset, split, {"s2s1": {"model": {"backbone": "terramind_v1_large_s2s1"}}})
    assert problems == ["s2s1: 6 chips lack _s1rtc.tif, e.g. "
                        "['EE_00000_00000', 'EE_00001_00000', 'EE_00004_00000']"]
```

- [ ] **Step 3: Run the new tests to verify they fail**

Run: `$PY -m pytest -q tests/test_thor_backbone.py tests/test_segmentation_data.py tests/test_pipeline_runner.py -k "s1 or stacked or registry or modality"`
Expected: FAIL (`KeyError: unknown backbone 'thor_v1_large_s2s1'`, `ImportError: THOR_S1_BANDS`, `stack_modalities` unexpected, and an empty problem list).

- [ ] **Step 4: Add `stack_modalities` to `BackboneSpec`** (after `extra_modalities` in `backbones.py`)

```python
    #: Whether the encoder takes its further modalities as more bands of one image rather than as
    #: a dictionary of modalities. THOR reads Sentinel-1 as a band group of the same image, so its
    #: datamodule hands over the interleaved stack without splitting it; TerraMind takes one
    #: tensor per modality.
    stack_modalities: bool = False
```

- [ ] **Step 5: Add THOR's S1 statistics and modality** (after `_thor_stats` in `backbones.py`)

```python
#: Sentinel-1 as THOR reads it: VV and VH, which THOR maps to its 10 m IW group.
THOR_S1_BANDS = ("VV", "VH")


def _thor_s1_stats(bands: list[str]) -> tuple[list[float], list[float]]:
    """THOR's pretraining statistics of its S1 bands at 10 m, in dB, the unit of ``_s1rtc.tif``."""
    _register_thor()
    from thor_terratorch_ext.models.backbones import thor_vit as tv

    # THOR keys its SAR statistics by ground sampling distance: "S1:IW-VV_10" for 10 m.
    keys = [f"{tv.lookup_band.get(b)}_10" for b in bands]
    missing = [b for b, k in zip(bands, keys) if k not in tv.THOR_NORMALIZATION_PARAMS]
    if missing:
        raise KeyError(f"THOR was not pretrained on {missing}")
    params = [tv.THOR_NORMALIZATION_PARAMS[k] for k in keys]
    return [p["mean"] for p in params], [p["std"] for p in params]


#: Sentinel-1 RTC as THOR's third band group: the ``_s1rtc.tif`` TerraMind reads, normalised with
#: THOR's statistics, and stacked behind the Sentinel-2 bands (``stack_modalities``).
THOR_S1RTC = {"S1RTC": {"suffix": "_s1rtc.tif", "bands": THOR_S1_BANDS,
                        "stats": _thor_s1_stats, "sidecar": "s1_rtc.json"}}
```

- [ ] **Step 6: Replace `_thor` with the version that takes a merge method and S1**

```python
def _thor(variant: str, indices: list[int], bottleneck: int, token_m: int = THOR_TOKEN_M,
          merge: str = "concat", s1: bool = False) -> Callable[[list[str], int], dict]:
    gsd = {**dict.fromkeys(THOR_BANDS_10M, 10), **dict.fromkeys(THOR_BANDS_20M, 20),
           **dict.fromkeys(THOR_S1_BANDS, 10)}
    if THOR_GROUND_COVER_M % token_m or token_m % max(gsd.values()):
        raise ValueError(f"{token_m} m tokens do not tile the chip in every band group")

    def build(bands: list[str], n_timesteps: int) -> dict:
        # Every path that builds a model asks for these arguments first, so THOR is in
        # TerraTorch's registry by the time the factory looks it up.
        _register_thor()
        # ``bands`` are the Sentinel-2 bands; S1, when read, follows them in every month, as the
        # datamodule interleaves it.
        model_bands = list(bands) + (list(THOR_S1_BANDS) if s1 else [])
        return {
            "backbone": variant,
            "backbone_pretrained": True,
            "backbone_model_bands": model_bands,
            "backbone_ground_cover": THOR_GROUND_COVER_M,
            # THOR tokenises each band group on its own, at a patch size chosen at inference.
            # Every group gets tokens of the same ground size: at 160 m, 16 px on the 10 m bands
            # (S2 and S1) and 8 px on the 20 m bands, the 14 x 14 grid TerraMind has on the same
            # chip; at 80 m, 8 and 4 px and a 28 x 28 grid, 4 px being the smallest patch THOR
            # was pretrained with.
            "backbone_patch_sizes": {b: token_m // gsd[b] for b in model_bands},
            # "concat" stacks the groups' maps on the channel axis, as the THOR authors do for
            # dense tasks; "mean" averages them at every token position, as TerraMind averages
            # its modalities, so the decoder receives one group's width.
            "backbone_merge_method": merge,
            # THOR is single-date; the wrapper runs it per month and concatenates the
            # monthly features, as for TerraMind.
            "backbone_use_temporal": True,
            "backbone_temporal_pooling": "concat",
            "backbone_temporal_n_timestamps": n_timesteps,
            "necks": [
                {"name": "SelectIndices", "indices": indices},
                # Concatenated: 12 months x 2 groups x 1024 = 24,576 channels per layer for the
                # large variant. Averaged: 12 x 1024 = 12,288, TerraMind's width.
                {"name": "ChannelBottleneck", "out_channels": bottleneck},
                {"name": "LearnedInterpolateToPyramidal"},
            ],
        }
    return build
```

- [ ] **Step 7: Register `thor_v1_large_s2s1`** (in `BACKBONES`, after `thor_v1_large_80m`)

```python
    # THOR on Sentinel-2 and Sentinel-1: S1 is a third band group on the same 160 m grid, and the
    # three groups are averaged at every token position, as TerraMind averages its two
    # modalities, so the bottleneck and decoder are TerraMind's (decided 15 October 2026).
    "thor_v1_large_s2s1": BackboneSpec(
        name="thor_v1_large_s2s1", resolution_group="token_grid",
        model_args=_thor("thor_v1_large", [5, 11, 17, 23], bottleneck=768, merge="mean", s1=True),
        stats=_thor_stats, input_bands=THOR_BANDS, extra_modalities=THOR_S1RTC,
        stack_modalities=True,
        notes="thor_v1_large on S2 and S1 RTC (VV, VH), the three band groups averaged per token"),
```

- [ ] **Step 8: Let the datamodule keep a stacked modality whole** (`segmentation_data.py`)

Replace

```python
        extra_rasters, split = [], ([("S2L2A", len(bands))] if extra else None)
```

with

```python
        stacked = spec.stack_modalities
        extra_rasters, split = [], ([("S2L2A", len(bands))] if extra and not stacked else None)
```

and replace the last line of the loop over `extra.items()`,

```python
            split.append((name, len(mb)))
```

with

```python
            if split is not None:
                split.append((name, len(mb)))
```

and extend the comment above the block (the one ending "the modality brings its own.") with: `An encoder that reads the modality as more bands of the same image (``stack_modalities``, THOR) gets the stack unsplit.`

- [ ] **Step 9: Make the chip-set check ask for every modality** (`runner.py`, `check_chipset`)

Replace

```python
        if spec.representation == "s2_monthly":
            suffixes = ["_merged.tif"]
```

with

```python
        if spec.representation == "s2_monthly":
            suffixes = ["_merged.tif", *(m["suffix"] for m in spec.extra_modalities.values())]
```

- [ ] **Step 10: Run the tests**

Run: `$PY -m pytest -q tests/test_thor_backbone.py tests/test_segmentation_data.py tests/test_pipeline_runner.py`
Expected: PASS.

- [ ] **Step 11: Commit**

```bash
git add src/gfm4agri/benchmark/backbones.py src/gfm4agri/benchmark/segmentation_data.py \
        src/gfm4agri/pipeline/runner.py tests/test_thor_backbone.py tests/test_segmentation_data.py \
        tests/test_pipeline_runner.py
git commit -m "Give THOR Sentinel-1 as a third band group, its groups averaged

thor_v1_large_s2s1 reads VV and VH stacked behind its S2 bands, normalised with THOR's 10 m
statistics in dB, and averages its three groups per token, so its bottleneck and decoder are
TerraMind's. The chip-set check now asks for every modality an arm reads.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 2: The TESSERA v2 encoder

**Files:**
- Create: `src/gfm4agri/embeddings/tessera_v2_model.py` (vendored, unedited)
- Create: `src/gfm4agri/embeddings/tessera_v2.py`
- Test: `tests/test_tessera_v2.py`

**Interfaces:**
- Produces: `tessera_v2.TESSERA_S2_BANDS`, `WEIGHTS: Path`, `N_DIMS = 128`, `month_doy(year, n_months=12) -> np.ndarray`, `s1_to_tessera(s1_db) -> np.ndarray`, `chip_inputs(chip_dir, chip_id, band_names, n_months) -> (s2 (P,T,10) float32, s1_db (P,T,2) float32, (H, W))`, `load_student(size, weights_dir=WEIGHTS, sha256=None) -> PixelStudent`, `encode_pixels(model, s2, s1_db, doy, device, precision="bf16-mixed", batch=16384) -> (P,128) float32`, `ChipEncoder(student, checkpoint_sha256, weights_dir=WEIGHTS, *, band_names, n_months, year, precision="bf16-mixed", batch=16384, device=None)` with `read(chip_dir, chip_id) -> inputs` and `encode(inputs) -> (128, H, W) float32`.

- [ ] **Step 1: Vendor TESSERA's model code**

```bash
SRC="/data/private/THESIS - ExplainedGMF4Agri/experiments/2026-10-14_tessera_composites/tessera_v2.py"
$PY - "$SRC" src/gfm4agri/embeddings/tessera_v2_model.py <<'PY'
import sys
src = open(sys.argv[1]).read()
licence_at = src.index("    MIT License")
licence = src[licence_at: src.index("SOFTWARE.\n", licence_at) + len("SOFTWARE.\n")]
start = src.index("import math")
end = src.rindex("\n# =", 0, src.index("# This experiment's part.")) + 1
header = (
    '"""TESSERA v2\'s pixel student and its input bucketing, vendored from TESSERA.\n\n'
    "Copied verbatim from https://github.com/ucam-eo/tessera at commit e368539f7c4711e5f16fdd09801573830e34baf9:\n\n"
    "* ``tessera_infer_v2/student/model.py``: everything from its imports to ``PixelStudentV11``;\n"
    "* ``tessera_infer_v2/student/infer.py``: ``BIN_EDGES`` to ``_build_source_indices``, the bucketing and\n"
    "  padding rule of ``encode_pixels``.\n\n"
    "Nothing here is edited; :mod:`gfm4agri.embeddings.tessera_v2` maps the chip set's composites onto it. Under\n"
    "TESSERA's licence:\n\n" + licence + '"""\n\n'
)
open(sys.argv[2], "w").write(header + src[start:end].rstrip() + "\n")
PY
$PY -c "import sys; sys.path.insert(0, 'src'); from gfm4agri.embeddings.tessera_v2_model import PixelStudent, S1A_BAND_MEAN, _build_source_indices; print('ok')"
```

Expected: `ok`. The file holds `S2_BAND_MEAN`, `S2_BAND_STD`, `S1A_BAND_MEAN`, `S1A_BAND_STD`, the model classes, `PixelStudent`, `BIN_EDGES`, `get_bin_size`, `_vec_get_bin_size`, `_pad_pattern`, `_build_source_indices`.

- [ ] **Step 2: Write the failing tests** (`tests/test_tessera_v2.py`)

```python
"""Frozen TESSERA v2 on the chip set's composites: input mapping, per-pixel encoding, validated output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from gfm4agri.embeddings import tessera_v2 as tv  # noqa: E402
from gfm4agri.embeddings.tessera_v2_model import PixelStudent  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MINI = REPO / "data" / "eurocrops_chips" / "EE_2021_rev_mini" / "chips"
SPIKE = REPO / "results" / "2026-10-14_tessera_composites" / "spike" / "emb" / "v2_medium"
CHIP = "EE_02252_01776"
SHA = "3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e"
#: The band names of our chip manifests, in wavelength order.
OURS = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
        "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2"]


def _write(path: Path, arr: np.ndarray) -> None:
    from affine import Affine

    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype=arr.dtype, crs="EPSG:3035",
                       transform=Affine(10, 0, 0, 0, -10, arr.shape[1] * 10)) as dst:
        dst.write(arr)


def test_s2_bands_are_taken_by_name_in_tesseras_order(tmp_path):
    T, H = 2, 4
    s2 = np.zeros((T * len(OURS), H, H), np.int16)
    for t in range(T):
        for b in range(len(OURS)):
            s2[t * len(OURS) + b] = 100 * t + b
    _write(tmp_path / "X_merged.tif", s2)
    _write(tmp_path / "X_s1rtc.tif", np.full((T * 2, H, H), -10.0, np.float32))
    x2, x1, hw = tv.chip_inputs(tmp_path, "X", OURS, T)
    assert hw == (H, H) and x2.shape == (H * H, T, 10) and x1.shape == (H * H, T, 2)
    want = [OURS.index(b) for b in tv.TESSERA_S2_BANDS]
    assert want == [3, 1, 2, 7, 8, 4, 5, 6, 10, 11]          # B04 B02 B03 B08 B8A B05 B06 B07 B11 B12
    for t in range(T):
        assert (x2[:, t] == 100 * t + np.array(want)).all()


def test_s1_db_becomes_tesseras_int16_encoding():
    db = np.array([[[-10.0, -20.0], [np.nan, np.nan], [200.0, -30.0]]], np.float32)
    assert tv.s1_to_tessera(db).tolist() == [[[6000.0, 2000.0], [0.0, 0.0], [32767.0, 0.0]]]


def test_the_composites_are_dated_the_15th():
    assert tv.month_doy(2021).tolist() == [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    assert tv.month_doy(2024)[2] == 75


def _tiny_student() -> PixelStudent:
    torch.manual_seed(0)
    return PixelStudent(latent_dim=16, num_layers=1, nhead=2, dim_feedforward=32).eval()


def test_each_pixel_is_encoded_on_its_own_and_a_pixel_without_input_is_nan():
    rng = np.random.default_rng(0)
    s2 = rng.integers(1, 3000, (6, 12, 10)).astype(np.float32)
    s1 = rng.normal(-12, 3, (6, 12, 2)).astype(np.float32)
    s2[5], s1[5] = 0, np.nan            # no input at all
    s1[2, 4:] = np.nan                  # four S1 months only: another bucket
    model, doy, cpu = _tiny_student(), tv.month_doy(2021), torch.device("cpu")
    emb = tv.encode_pixels(model, s2, s1, doy, cpu, precision="32-true", batch=4)
    assert emb.shape == (6, tv.N_DIMS) and np.isfinite(emb[:5]).all() and np.isnan(emb[5]).all()
    # A pixel's embedding does not depend on the other pixels of its batch.
    part = tv.encode_pixels(model, s2[[3, 2]], s1[[3, 2]], doy, cpu, precision="32-true", batch=1)
    assert np.allclose(part, emb[[3, 2]], atol=1e-5)


@pytest.mark.skipif(not ((tv.WEIGHTS / "student_medium.pt").exists() and (SPIKE / f"{CHIP}.npy").exists()
                         and (MINI / f"{CHIP}_merged.tif").exists()),
                    reason="v2 weights, pilot chips or the spike's embeddings are absent")
def test_the_core_encoder_reproduces_the_composites_experiment():
    model = tv.load_student("medium", tv.WEIGHTS, SHA).eval()
    s2, s1, _ = tv.chip_inputs(MINI, CHIP, OURS, 12)
    n = 2048
    emb = tv.encode_pixels(model, s2[:n], s1[:n], tv.month_doy(2021), torch.device("cpu"),
                           precision="32-true")
    ref = np.load(SPIKE / f"{CHIP}.npy").astype(np.float32).reshape(tv.N_DIMS, -1).T[:n]
    cos = (emb * ref).sum(1) / np.linalg.norm(emb, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.median(cos) > 0.9999 and cos.min() > 0.999
```

- [ ] **Step 3: Run them to verify they fail**

Run: `$PY -m pytest -q tests/test_tessera_v2.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gfm4agri.embeddings.tessera_v2'`.

- [ ] **Step 4: Write `src/gfm4agri/embeddings/tessera_v2.py`**

```python
"""Frozen TESSERA v2 on the chip set's monthly composites: weights, input mapping, per-pixel encoding.

TESSERA encodes each pixel's own Sentinel-2 and Sentinel-1 series, with no spatial context. Here it reads the twelve
monthly composites every other arm reads, as validated in experiments/2026-10-14_tessera_composites (frozen, it
separates crops as well as from its native per-acquisition series) and experiments/2026-10-15_tessera_decoder
(chip rasters equal to the spike's, cosine 0.9999):

* **S2**: TESSERA's ten bands in its order, B04 B02 B03 B08 B8A B05 B06 B07 B11 B12, taken from ``_merged.tif`` by
  band name; DN = reflectance x 10000, TESSERA's convention; every month valid, the composites being gap filled.
* **S1**: our 10 log10 of the linear-power median, in dB, becomes TESSERA's int16 encoding
  (20 log10(power) + 50) x 200 = (2 dB + 50) x 200, clipped to [0, 32767], with 0 where a month was never observed,
  TESSERA's missing value. Our composites merge the two orbit directions, so the frames go in as its ascending stream.
* **Dates**: the 15th of each month, as the raw day of year.

The inputs are standardised with v2's statistics and bucketed and padded as v2's ``encode_pixels`` does
(:mod:`gfm4agri.embeddings.tessera_v2_model`). A pixel with no valid S2 and no valid S1 month comes out NaN; the
student itself would return a vector for it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np

__all__ = ["N_DIMS", "TESSERA_S2_BANDS", "WEIGHTS", "ChipEncoder", "chip_inputs", "encode_pixels",
           "load_student", "month_doy", "s1_to_tessera"]

REPO = Path(__file__).resolve().parents[3]
#: TESSERA's ten S2 bands, B04 B02 B03 B08 B8A B05 B06 B07 B11 B12, by our manifests' band names.
TESSERA_S2_BANDS = ("RED", "BLUE", "GREEN", "NIR_BROAD", "NIR_NARROW", "RED_EDGE_1", "RED_EDGE_2",
                    "RED_EDGE_3", "SWIR_1", "SWIR_2")
#: Hugging Face repository letter of each v2 student, ``geotessera/TESSERA-V-2.0-2B-<letter>``.
STUDENTS = {"nano": "N", "small": "S", "medium": "M", "large": "L"}
WEIGHTS = REPO / "data" / "weights" / "tessera" / "v2"
N_DIMS = 128


def month_doy(year: int, n_months: int = 12) -> np.ndarray:
    """Day of year of the 15th of each month, the raw integers TESSERA takes as dates."""
    return np.array([dt.date(year, m, 15).timetuple().tm_yday for m in range(1, n_months + 1)],
                    dtype=np.float32)


def s1_to_tessera(s1_db) -> np.ndarray:
    """Our S1 in dB, NaN where never observed, in TESSERA's raw units, 0 where never observed."""
    s1_db = np.asarray(s1_db, dtype=np.float32)
    ok = np.isfinite(s1_db).all(-1, keepdims=True)
    return np.where(ok, np.clip((2.0 * s1_db + 50.0) * 200.0, 0, 32767), 0.0).astype(np.float32)


def chip_inputs(chip_dir, chip_id: str, band_names: list[str],
                n_months: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """One chip's pixels as TESSERA reads them: S2 ``(P, T, 10)`` DN, S1 ``(P, T, 2)`` dB, and ``(H, W)``."""
    import rasterio

    idx = [list(band_names).index(b) for b in TESSERA_S2_BANDS]
    with rasterio.open(Path(chip_dir) / f"{chip_id}_merged.tif") as src:
        hw = (src.height, src.width)
        s2 = src.read().reshape(n_months, len(band_names), -1)[:, idx]          # (T, 10, P)
    with rasterio.open(Path(chip_dir) / f"{chip_id}_s1rtc.tif") as src:
        s1 = src.read().astype(np.float32).reshape(n_months, 2, -1)             # (T, 2, P)
    return (np.ascontiguousarray(s2.transpose(2, 0, 1)).astype(np.float32),
            np.ascontiguousarray(s1.transpose(2, 0, 1)), hw)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_student(size: str, weights_dir=WEIGHTS, sha256: str | None = None):
    """A v2 pixel student from ``weights_dir/student_<size>.pt``, fetched from Hugging Face if absent.

    The checkpoint holds tensors and plain containers only, so it loads with ``weights_only=True``; its SHA-256 is
    checked against ``sha256`` when one is given. Built as TESSERA's ``load_model`` builds it, with dropout 0; the
    fusion MLP's fixed dropout of 0.2 is off once the caller puts the model in evaluation mode.
    """
    import torch

    from gfm4agri.embeddings.tessera_v2_model import PixelStudent

    path = Path(weights_dir) / f"student_{size}.pt"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(f"geotessera/TESSERA-V-2.0-2B-{STUDENTS[size]}", f"ckpt/student_{size}.pt")
        tmp = path.with_suffix(".pt.tmp")
        tmp.write_bytes(Path(got).read_bytes())
        tmp.replace(path)
    if sha256 is not None and _sha256(path) != sha256:
        raise ValueError(f"{path}: SHA-256 differs from the registry's {sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    a = payload.get("args", {}) or {}
    model = PixelStudent(
        repr_dim=int(a.get("repr_dim", 128)), latent_dim=int(a.get("latent_dim", 64)),
        num_layers=int(a.get("num_layers", 4)), nhead=int(a.get("nhead", 4)),
        dim_feedforward=int(a.get("dim_feedforward", 1024)), dropout=0.0,
        max_seq_len=int(a.get("max_seq_len", 256)),
        matryoshka_dims=tuple(int(d) for d in str(a.get("matryoshka_dims", "16,32,64,128")).split(",")),
        enable_qk_norm=bool(a.get("enable_qk_norm", False)),
    )
    model.load_state_dict(payload["model"])
    return model


def _encode_batch(model, s2: np.ndarray, s1: np.ndarray, doy: np.ndarray, device, autocast) -> np.ndarray:
    """``(B, 128)`` for ``B`` pixels in TESSERA's raw units, prepared as v2's ``encode_pixels`` prepares them.

    S1 validity comes from the raw values, both sensors are standardised with v2's statistics (S1 with the ascending
    stream's), each pixel's valid frames go to the bucket of 8, 16, ... steps its count falls in, padded by v2's
    pattern, with the raw day of year; pixels sharing a bucket pair go through one forward pass.
    """
    import torch

    from gfm4agri.embeddings import tessera_v2_model as tm

    s1_valid = np.any(s1 != 0, axis=-1)
    s1n = (s1 - tm.S1A_BAND_MEAN) / (tm.S1A_BAND_STD + 1e-9)
    s2_valid = np.any(s2 != 0, axis=-1)
    keys = (tm._vec_get_bin_size(s2_valid.sum(1)).astype(np.int64) * 1000
            + tm._vec_get_bin_size(s1_valid.sum(1)).astype(np.int64))
    out = np.empty((len(s2), N_DIMS), np.float32)
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        s2_b, s1_b = max(int(key // 1000), 1), max(int(key % 1000), 1)
        s2_in = np.zeros((len(idx), s2_b, 11), np.float32)
        s1_in = np.zeros((len(idx), s1_b, 3), np.float32)
        if key // 1000 > 0:
            src = tm._build_source_indices(s2_valid[idx], s2_b)
            g = np.take_along_axis(s2[idx], src[:, :, None].repeat(10, axis=2), axis=1)
            s2_in[:, :, :10] = (g - tm.S2_BAND_MEAN) / (tm.S2_BAND_STD + 1e-9)
            s2_in[:, :, 10] = doy[src]
        if key % 1000 > 0:
            src = tm._build_source_indices(s1_valid[idx], s1_b)
            s1_in[:, :, :2] = np.take_along_axis(s1n[idx], src[:, :, None].repeat(2, axis=2), axis=1)
            s1_in[:, :, 2] = doy[src]
        with autocast:
            emb = model.encode(torch.from_numpy(s2_in).to(device), torch.from_numpy(s1_in).to(device))
        out[idx] = emb.float().cpu().numpy()
    return out


def encode_pixels(model, s2, s1_db, doy, device, precision: str = "bf16-mixed",
                  batch: int = 16384) -> np.ndarray:
    """``(P, 128)`` float32 embeddings of ``P`` pixels, no gradients; NaN where a pixel has no input at all."""
    import torch

    s2 = np.asarray(s2, dtype=np.float32)
    s1 = s1_to_tessera(s1_db)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16,
                              enabled=precision.startswith("bf16") and device.type == "cuda")
    out = np.empty((len(s2), N_DIMS), np.float32)
    with torch.no_grad():
        for i in range(0, len(s2), batch):
            out[i:i + batch] = _encode_batch(model, s2[i:i + batch], s1[i:i + batch], doy, device, autocast)
    empty = ~np.any(s2 != 0, axis=(1, 2)) & ~np.any(s1 != 0, axis=(1, 2))
    out[empty] = np.nan
    return out


class ChipEncoder:
    """A frozen v2 student over whole chips, as :mod:`gfm4agri.pipeline.raster_cache` drives it.

    ``read`` loads a chip's composites, on a reader thread; ``encode`` runs the student and returns
    ``(128, H, W)`` float32 on the chip's own grid.
    """

    def __init__(self, student: str, checkpoint_sha256: str, weights_dir=WEIGHTS, *, band_names,
                 n_months: int, year: int, precision: str = "bf16-mixed", batch: int = 16384,
                 device: str | None = None) -> None:
        import torch

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_student(student, weights_dir, checkpoint_sha256).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.band_names, self.n_months = list(band_names), int(n_months)
        self.doy = month_doy(year, n_months)
        self.precision, self.batch = precision, int(batch)

    def read(self, chip_dir, chip_id: str):
        return chip_inputs(chip_dir, chip_id, self.band_names, self.n_months)

    def encode(self, inputs) -> np.ndarray:
        s2, s1, (h, w) = inputs
        emb = encode_pixels(self.model, s2, s1, self.doy, self.device, self.precision, self.batch)
        return np.ascontiguousarray(emb.T.reshape(N_DIMS, h, w))
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest -q tests/test_tessera_v2.py -v`
Expected: 5 passed (the last one runs on the hub, where weights, pilot chips and spike embeddings exist).

- [ ] **Step 6: Commit**

```bash
git add src/gfm4agri/embeddings/tessera_v2_model.py src/gfm4agri/embeddings/tessera_v2.py tests/test_tessera_v2.py
git commit -m "Add the TESSERA v2 encoder on the chip set's composites

The v2 pixel student vendored from TESSERA (MIT, commit e368539) and the input mapping of the
composites experiments: S2 bands by name in TESSERA's order, S1 dB to its int16 encoding, the
15th of each month. Reproduces the spike's v2 Medium embeddings.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: A computed raster in the registry, the datamodule and the fit

**Files:**
- Modify: `src/gfm4agri/benchmark/backbones.py` (`BackboneSpec.computed`; entry `tessera_v2_medium`; module docstring paragraph on precomputed embeddings)
- Modify: `src/gfm4agri/benchmark/segmentation_data.py` (`EuroCropsSegDataModule.__init__` and `setup`)
- Modify: `src/gfm4agri/pipeline/config.py` (`compose`)
- Modify: `src/gfm4agri/benchmark/fit.py` (`fit_end_to_end`, the datamodule construction)
- Modify: `src/gfm4agri/benchmark/evaluation.py` (`FitPredictor.__init__` and `_build`)
- Test: `tests/test_segmentation_data.py`, `tests/test_pipeline_config.py`

**Interfaces:**
- Consumes: nothing from Tasks 1 and 2.
- Produces: `BackboneSpec.computed: dict` (keys `encoder`, `student`, `checkpoint_sha256`, `precision`, `dims`, `inputs`); entry `"tessera_v2_medium"` with representation `"tessera_v2_medium"`; `EuroCropsSegDataModule(..., image_root=None)`; `compose(..., output_root, image_root=None)` writing `cfg["data"]["image_root"]` only when given; `fit_end_to_end` and `FitPredictor` read `cfg["data"]["image_root"]`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_segmentation_data.py` (add `from conftest import SYN_SPLIT  # noqa: E402` to its imports):

```python
V2 = "tessera_v2_medium"


def _fake_v2_cache(chips: Path, cache: Path, drop: str | None = None) -> None:
    """A computed raster cache for ``chips``: dimension k of a pixel is its parcel id + k."""
    (cache / "chips").mkdir(parents=True)
    dims = get_backbone(V2).computed["dims"]
    for f in sorted((chips / "chips").glob("*.parcels.tif")):
        cid = f.name[: -len(".parcels.tif")]
        if cid == drop:
            continue
        with rasterio.open(f) as src:
            ids = src.read(1)
        _write(cache / "chips" / f"{cid}_{V2}.tif",
               np.stack([ids.astype(np.float32) + k for k in range(dims)]))
    (cache / f"{V2}.json").write_text(json.dumps({
        "file_suffix": f"_{V2}.tif", "band_names": [f"D{k:03d}" for k in range(dims)],
        "normalisation": {"means": [0.0] * dims, "stds": [1.0] * dims}}))


def test_a_computed_raster_is_read_from_its_cache_and_paired_with_the_chip_sets_masks(
        split_chipset, tmp_path):
    cache = tmp_path / "emb" / V2 / split_chipset.name
    _fake_v2_cache(split_chipset, cache)
    dm = EuroCropsSegDataModule(split_chipset, V2, normalisation="chips", batch_size=1, num_workers=0,
                                augment=True, keep_parcel_ids=np.array([101]),
                                split_dir=split_chipset / "splits" / SYN_SPLIT, image_root=cache)
    dm.setup("fit")
    assert {Path(f).parent for f in dm.train_dataset.image_files} == {cache / "chips"}
    assert {Path(f).parent for f in dm.train_dataset.segmentation_mask_files} == {split_chipset / "chips"}
    for _ in range(8):
        s = dm.train_dataset[0]
        lab = s["mask"] >= 0
        # D4 moves image and mask together; only parcel 101 is labelled.
        assert lab.sum() == 256 and (s["image"][0][lab] == 101).all()


def test_a_cache_missing_a_chip_is_refused(split_chipset, tmp_path):
    cache = tmp_path / "emb" / V2 / split_chipset.name
    _fake_v2_cache(split_chipset, cache, drop="EE_00004_00000")
    dm = EuroCropsSegDataModule(split_chipset, V2, normalisation="chips", batch_size=1, num_workers=0,
                                split_dir=split_chipset / "splits" / SYN_SPLIT, image_root=cache)
    with pytest.raises(ValueError, match="input rasters and masks differ"):
        dm.setup("fit")


def test_tessera_v2_is_a_10m_backbone_the_runner_computes():
    spec = get_backbone(V2)
    assert spec.resolution_group == "pixel_raster" and spec.representation == V2
    assert spec.computed == {
        "encoder": "tessera_v2", "student": "medium",
        "checkpoint_sha256": "3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e",
        "precision": "bf16-mixed", "dims": 128, "inputs": ["_merged.tif", "_s1rtc.tif"]}
    assert not get_backbone("tessera_v1").computed
```

Append to `tests/test_pipeline_config.py`:

```python
def test_a_computed_raster_reaches_the_fit_through_image_root(split_chipset, tmp_path):
    arm = {"route": "raster", "data": {"normalisation": "chips", "batch_size": 4, "augment": True},
           "model": {"backbone": "tessera_v2_medium"}, "trainer": {"precision": "16-mixed", "log_every_n_steps": 10}}
    exp = {"epochs": 15}
    split = resolve_split(split_chipset)
    cell = Cell("a", 100, 0, 0)
    plain = compose("a", arm, exp, {"num_workers": 0}, split_chipset, split, cell, tmp_path)
    assert "image_root" not in plain["data"]
    cfg = compose("a", arm, exp, {"num_workers": 0}, split_chipset, split, cell, tmp_path,
                  image_root=tmp_path / "emb")
    assert cfg["data"]["image_root"] == str((tmp_path / "emb").resolve())   # outside the repository
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest -q tests/test_segmentation_data.py tests/test_pipeline_config.py -k "computed or cache_missing or tessera_v2 or image_root"`
Expected: FAIL (`unknown backbone 'tessera_v2_medium'`, `unexpected keyword argument 'image_root'`).

- [ ] **Step 3: Add `computed` to `BackboneSpec`** (after `stack_modalities`)

```python
    #: For a 10 m backbone whose raster the K-shot runner computes rather than reads from the chip
    #: set (TESSERA v2): ``encoder`` (a key of ``raster_cache.ENCODERS``), ``student``,
    #: ``checkpoint_sha256``, ``precision``, ``dims`` and the chip-set rasters it reads, ``inputs``.
    #: Empty for every other backbone.
    computed: dict = field(default_factory=dict)
```

- [ ] **Step 4: Register `tessera_v2_medium`** (in `BACKBONES`, after `tessera_v1`)

```python
    # TESSERA v2 from its open weights, frozen, run by the K-shot runner on the chip set's monthly
    # S2 and S1 composites; its 10 m output is cached as an embedding raster
    # (gfm4agri.pipeline.raster_cache) and read like a precomputed one. It replaces tessera_v1,
    # whose store rasters sit about one pixel off the chip grid (decided 15 October 2026).
    "tessera_v2_medium": BackboneSpec(
        name="tessera_v2_medium", resolution_group="pixel_raster", model_args=_precomputed(128),
        stats=None, representation="tessera_v2_medium",
        computed={"encoder": "tessera_v2", "student": "medium",
                  "checkpoint_sha256": "3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e",
                  "precision": "bf16-mixed", "dims": 128, "inputs": ["_merged.tif", "_s1rtc.tif"]},
        notes="TESSERA v2 Medium pixel student (21.0 M parameters), frozen, 128 dims per 10 m pixel "
              "from the 12 monthly S2 and S1 composites; computed and cached by the runner"),
```

Extend the module docstring's last paragraph with: `TESSERA v2 is a third kind: the same identity backbone and group, but its raster is computed by the K-shot runner from the chip set's composites with the encoder named in ``computed``, and cached outside the chip set.`

- [ ] **Step 5: Read a computed raster from its cache in the datamodule** (`segmentation_data.py`)

Add `image_root: Path | str | None = None,` to `__init__`'s keyword parameters, after `split_dir`. At the top of `__init__`, after `split_dir = ...`:

```python
        # A computed raster (TESSERA v2) lives in its cache, not in the chip set: the inputs come
        # from there, the masks, parcel rasters and split files from the chip set.
        image_root = None if image_root is None else Path(image_root)
```

After `spec = get_backbone(backbone)`:

```python
        if image_root is not None and (spec.representation == "s2_monthly" or split_dir is None):
            raise ValueError("image_root holds a computed embedding raster under a spatial split")
```

Replace

```python
            sidecar = json.loads((root / f"{spec.representation}.json").read_text())
```

with

```python
            sidecar = json.loads(((image_root or root) / f"{spec.representation}.json").read_text())
```

Replace the `roots` block and the data-root arguments of `super().__init__`:

```python
        roots = {}
        for split_name in SPLIT_DIRS:
            listed = split_list(root, split_name, split_dir)
            if listed.exists():
                labels = split_chip_dir(root, split_name, split_dir)
                roots[split_name] = (image_root / "chips" if image_root else labels, labels, listed)
        if "test" not in roots:  # the pilot has no test split; evaluate on validation
            roots["test"] = roots["val"]

        super().__init__(
            batch_size=batch_size, num_workers=num_workers,
            num_classes=len(manifest["classes"]),
            train_data_root=roots["train"][0], train_label_data_root=roots["train"][1],
            train_split=roots["train"][2],
            val_data_root=roots["val"][0], val_label_data_root=roots["val"][1],
            val_split=roots["val"][2],
            test_data_root=roots["test"][0], test_label_data_root=roots["test"][1],
            test_split=roots["test"][2],
```

(the remaining arguments of the call are unchanged). After `self.split_dir = split_dir`, add `self.image_root = image_root`.

In `setup`, directly after `super().setup(stage)`:

```python
        # TerraTorch pairs input rasters and masks by sorted position. With the inputs in a cache
        # apart from the masks, a chip missing on one side would shift every pair after it.
        for name in ("train_dataset", "val_dataset", "test_dataset"):
            ds = getattr(self, name, None)
            if ds is None:
                continue
            imgs = [Path(f).name[: -len(self.img_suffix)] for f in ds.image_files]
            masks = [Path(f).name[: -len(MASK_SUFFIX)] for f in ds.segmentation_mask_files]
            if imgs != masks:
                odd = sorted(set(imgs) ^ set(masks))[:3]
                raise ValueError(f"{name}: input rasters and masks differ ({len(imgs)} against "
                                 f"{len(masks)} chips, e.g. {odd})")
```

- [ ] **Step 6: Carry `image_root` through `compose`, the fit and the predictor**

`config.py`, `compose`: add the parameter `image_root: Path | None = None` after `output_root`; build the data block as a variable and add the key only when given:

```python
    data = {"root": repo_relative(chips), "split": repo_relative(split_dir),
            "normalisation": d["normalisation"], "batch_size": d["batch_size"],
            "num_workers": machine["num_workers"], "augment": d["augment"]}
    if image_root is not None:
        data["image_root"] = repo_relative(image_root)
    return {
        "run_name": arm_name,
        "seed": cell.seed,
        "data": data,
        "label_budget": {"mode": "pct", "pct": _number(cell.pct), "draw_seed": cell.draw},
        "model": copy.deepcopy(arm["model"]),
        "trainer": trainer,
        "output_root": repo_relative(output_root),
    }
```

`fit.py`, `fit_end_to_end`: replace the datamodule construction with

```python
    image_root = REPO / d["image_root"] if d.get("image_root") else None
    dm = EuroCropsSegDataModule(root, cfg["model"]["backbone"], normalisation=d["normalisation"],
                                batch_size=d["batch_size"], num_workers=d["num_workers"],
                                augment=d["augment"], keep_parcel_ids=keep, split_dir=split_dir,
                                image_root=image_root)
```

`evaluation.py`, `FitPredictor.__init__`: after `self.split_dir = ...`, add `self.image_root = REPO / d["image_root"] if d.get("image_root") else None`; in `_build`, pass `image_root=self.image_root` to `EuroCropsSegDataModule`.

(`REPO / "/abs/path"` is the absolute path, so a cache on the cluster's `/local` scratch, stored absolute by `repo_relative`, resolves too.)

- [ ] **Step 7: Run the tests**

Run: `$PY -m pytest -q tests/test_segmentation_data.py tests/test_pipeline_config.py tests/test_pipeline_runner.py tests/test_fit.py`
Expected: PASS.

- [ ] **Step 8: Commit**

```bash
git add src/gfm4agri/benchmark/backbones.py src/gfm4agri/benchmark/segmentation_data.py \
        src/gfm4agri/pipeline/config.py src/gfm4agri/benchmark/fit.py src/gfm4agri/benchmark/evaluation.py \
        tests/test_segmentation_data.py tests/test_pipeline_config.py
git commit -m "Register TESSERA v2 Medium as a 10 m backbone the runner computes

A computed raster is read from its cache (image_root) and paired with the chip set's masks,
parcel rasters and splits; the datamodule now refuses inputs and masks that do not pair chip by
chip. compose, the fit and the predictor carry image_root.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The raster cache

**Files:**
- Create: `src/gfm4agri/pipeline/raster_cache.py`
- Modify: `tests/conftest.py` (a stub encoder and the `stub_computed` fixture)
- Test: `tests/test_raster_cache.py`

**Interfaces:**
- Consumes: `BackboneSpec.computed` (Task 3), `ChipEncoder` (Task 2), `CacheDecision` and `cache_dir_for` from `gfm4agri.pipeline.cache`, `band_stats` from `gfm4agri.embeddings.export`.
- Produces: `ENCODERS: dict[str, factory]` with `factory(spec, manifest, batch) -> encoder` (`read(chip_dir, chip_id)`, `encode(inputs) -> (dims, H, W)`); `expected_record(chips, split_dir, backbone) -> dict`; `resolve_raster_cache(cache_root, scratch, backbone, chips, split_dir) -> CacheDecision`; `build_raster_cache(backbone, chips, split_dir, cache_dir, *, batch, encoder=None) -> dict` (the `cache.json` record). Cache layout: `<dir>/chips/<chip>_<representation>.tif`, `<dir>/<representation>.json`, `<dir>/cache.json`.

- [ ] **Step 1: Add the stub encoder and fixture to `tests/conftest.py`** (at the end)

```python
class StubChipEncoder:
    """Stands in for a per-pixel encoder: dimension k of a pixel is its parcel id + k, and the
    top-left pixel of every chip is NaN, as a pixel without any input would be."""

    def __init__(self, dims: int = 4) -> None:
        self.dims, self.encoded = dims, []

    def read(self, chip_dir: Path, chip_id: str):
        import rasterio

        with rasterio.open(Path(chip_dir) / f"{chip_id}.parcels.tif") as src:
            return chip_id, src.read(1)

    def encode(self, inputs) -> np.ndarray:
        chip_id, ids = inputs
        self.encoded.append(chip_id)
        emb = np.stack([ids.astype(np.float32) + k for k in range(self.dims)])
        emb[:, 0, 0] = np.nan
        return emb


@pytest.fixture()
def stub_computed(monkeypatch) -> StubChipEncoder:
    """A computed 10 m backbone, ``stub_v2``, whose encoder is a :class:`StubChipEncoder`."""
    from gfm4agri.benchmark import backbones as bb
    from gfm4agri.pipeline import raster_cache

    encoder = StubChipEncoder()
    monkeypatch.setitem(bb.BACKBONES, "stub_v2", bb.BackboneSpec(
        name="stub_v2", resolution_group="pixel_raster", model_args=bb._precomputed(4), stats=None,
        representation="stub_v2",
        computed={"encoder": "stub", "student": None, "checkpoint_sha256": "ab" * 32,
                  "precision": "32-true", "dims": 4, "inputs": [".parcels.tif"]}))
    monkeypatch.setitem(raster_cache.ENCODERS, "stub", lambda spec, manifest, batch: encoder)
    return encoder
```

- [ ] **Step 2: Write the failing tests** (`tests/test_raster_cache.py`)

```python
"""A computed raster is encoded once per chip set, reused only when it matches, never overwritten."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("terratorch")
rasterio = pytest.importorskip("rasterio")

from conftest import SYN_SPLIT  # noqa: E402
from gfm4agri.pipeline.raster_cache import build_raster_cache, resolve_raster_cache  # noqa: E402

B = "stub_v2"


def _split(chips):
    return chips / "splits" / SYN_SPLIT


def test_every_chip_is_encoded_with_a_sidecar_of_training_statistics(split_chipset, tmp_path, stub_computed):
    cache = tmp_path / "emb" / B / split_chipset.name
    record = build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
    ids = sorted(p.name[: -len(".parcels.tif")] for p in (split_chipset / "chips").glob("*.parcels.tif"))
    assert sorted(stub_computed.encoded) == ids and record["encoded_now"] == 6
    assert sorted(p.name for p in (cache / "chips").iterdir()) == [f"{c}_{B}.tif" for c in ids]
    side = json.loads((cache / f"{B}.json").read_text())
    assert side["file_suffix"] == f"_{B}.tif" and len(side["band_names"]) == 4
    assert all(c["nodata_share"] == pytest.approx(1 / 32 ** 2, abs=1e-6) for c in side["chips"])
    # Statistics over the two training chips, the NaN pixel left out.
    vals = []
    for c in ("EE_00000_00000", "EE_00001_00000"):
        with rasterio.open(split_chipset / "chips" / f"{c}.parcels.tif") as src:
            vals.append(src.read(1).astype(np.float64).ravel()[1:])
    vals = np.concatenate(vals)
    assert side["normalisation"]["means"] == pytest.approx([vals.mean() + k for k in range(4)], abs=1e-4)
    assert json.loads((cache / "cache.json").read_text())["backbone"] == B
    assert not (cache / "building.json").exists()


def test_no_cache_is_computed_in_scratch_and_a_complete_one_is_reused(split_chipset, tmp_path, stub_computed):
    root, scratch = tmp_path / "emb", tmp_path / "scratch"
    d = resolve_raster_cache(root, scratch, B, split_chipset, _split(split_chipset))
    assert d.action == "compute" and d.cache_dir == scratch / B / split_chipset.name
    build_raster_cache(B, split_chipset, _split(split_chipset), root / B / split_chipset.name, batch=1)
    d = resolve_raster_cache(root, scratch, B, split_chipset, _split(split_chipset))
    assert d.action == "reuse" and d.cache_dir == root / B / split_chipset.name


def test_a_missing_chip_is_encoded_again_and_nothing_else(split_chipset, tmp_path, stub_computed):
    root = tmp_path / "emb"
    cache = root / B / split_chipset.name
    build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
    (cache / "chips" / f"EE_00008_00000_{B}.tif").unlink()
    d = resolve_raster_cache(root, root, B, split_chipset, _split(split_chipset))
    assert d.action == "compute" and d.cache_dir == cache
    stub_computed.encoded.clear()
    build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
    assert stub_computed.encoded == ["EE_00008_00000"]


def test_a_cache_made_under_another_record_is_never_written_into(split_chipset, tmp_path, stub_computed):
    root, scratch = tmp_path / "emb", tmp_path / "scratch"
    cache = root / B / split_chipset.name
    build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
    rec = json.loads((cache / "cache.json").read_text())
    (cache / "cache.json").write_text(json.dumps(rec | {"checkpoint_sha256": "cd" * 32}))
    assert resolve_raster_cache(root, root, B, split_chipset, _split(split_chipset)).action == "stop"
    d = resolve_raster_cache(root, scratch, B, split_chipset, _split(split_chipset))
    assert d.action == "compute" and d.cache_dir == scratch / B / split_chipset.name
    with pytest.raises(SystemExit, match="another record"):
        build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
```

- [ ] **Step 3: Run them to verify they fail**

Run: `$PY -m pytest -q tests/test_raster_cache.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'gfm4agri.pipeline.raster_cache'`.

- [ ] **Step 4: Write `src/gfm4agri/pipeline/raster_cache.py`**

```python
"""The cache of a computed embedding raster: a frozen per-pixel encoder, run once per chip set by the runner.

A 10 m backbone whose spec carries ``computed`` (TESSERA v2) is not a product in the chip set but a model the K-shot
runner runs, as it runs the token-grid encoders whose feature caches :mod:`gfm4agri.pipeline.cache` manages. Its output
for a chip never changes, so it is computed once per chip set into ``<root>/<backbone>/<chip set>/``:

* ``chips/<chip>_<representation>.tif``: float32 on the chip's grid, NaN where the encoder had no input;
* ``<representation>.json``: the sidecar the datamodule reads, in the layout of the precomputed embeddings' sidecars,
  with per-band statistics of the training chips;
* ``cache.json``: the record the cache was made under, written last; ``building.json`` holds it meanwhile.

Unlike a ViT cache it holds one copy per chip: a per-pixel encoder commutes with rotation and flipping, so the D4
augmentation of the cached raster during training is exact.

The decisions are those of the feature caches. A complete cache under the cache root whose record matches the run is
reused; otherwise the cache is computed into the scratch root, resuming chips already there; a cache made under
another record is never written into.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np

from gfm4agri.benchmark.backbones import BackboneSpec, get_backbone
from gfm4agri.benchmark.segmentation_data import load_manifest, manifest_sha256, read_split, split_list
from gfm4agri.pipeline.cache import CacheDecision, cache_dir_for

__all__ = ["ENCODERS", "build_raster_cache", "expected_record", "resolve_raster_cache"]

RECORD, BUILDING = "cache.json", "building.json"
#: Chips read ahead of the encoder, so reading overlaps encoding.
READ_AHEAD = 8


def _write(path: Path, arr: np.ndarray, crs, transform, band_names: list[str]) -> None:
    """An uncompressed float32 raster: a cache is written and read again by one job, so speed beats size."""
    import rasterio

    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype="float32", crs=crs, transform=transform,
                       nodata=np.nan) as dst:
        dst.write(arr)
        for i, name in enumerate(band_names, start=1):
            dst.set_band_description(i, name)


def _tessera_v2(spec: BackboneSpec, manifest: dict, batch: int):
    from gfm4agri.embeddings.tessera_v2 import WEIGHTS, ChipEncoder

    c = spec.computed
    return ChipEncoder(c["student"], c["checkpoint_sha256"], WEIGHTS,
                       band_names=manifest["imagery"]["band_names"],
                       n_months=manifest["imagery"]["n_months"], year=manifest["year"],
                       precision=c["precision"], batch=batch)


#: ``computed["encoder"]`` -> ``(spec, chip manifest, batch) -> encoder``, the encoder offering
#: ``read(chip_dir, chip_id) -> inputs`` and ``encode(inputs) -> (dims, H, W)``.
ENCODERS: dict[str, Callable] = {"tessera_v2": _tessera_v2}


def expected_record(chips, split_dir, backbone: str) -> dict:
    """What a cache of ``backbone`` on ``chips`` must have been made under to be used for this run."""
    c = get_backbone(backbone).computed
    return {"backbone": backbone, "encoder": c["encoder"], "student": c.get("student"),
            "checkpoint_sha256": c["checkpoint_sha256"], "precision": c["precision"],
            "manifest_sha256": manifest_sha256(chips),
            "split_config_hash": read_split(split_dir)["config_hash"]}


def _mismatches(record: dict, expected: dict) -> list[str]:
    return [f"{k}: cache {record.get(k)!r}, run {v!r}" for k, v in expected.items() if record.get(k) != v]


def _chip_ids(chips, split_dir) -> list[str]:
    ids: set[str] = set()
    for s in ("train", "val", "test"):
        f = split_list(chips, s, split_dir)
        if f.exists():
            ids.update(f.read_text().split())
    return sorted(ids)


def _suffix(spec: BackboneSpec) -> str:
    return f"_{spec.representation}.tif"


def _missing(cache_dir: Path, chips, split_dir, spec: BackboneSpec) -> int:
    folder = cache_dir / "chips"
    have = set(os.listdir(folder)) if folder.is_dir() else set()
    return sum(c + _suffix(spec) not in have for c in _chip_ids(chips, split_dir))


def resolve_raster_cache(cache_root, scratch, backbone: str, chips, split_dir) -> CacheDecision:
    spec = get_backbone(backbone)
    found = cache_dir_for(cache_root, backbone, chips)
    target = cache_dir_for(scratch, backbone, chips)
    same = Path(cache_root).resolve() == Path(scratch).resolve()
    want = expected_record(chips, split_dir, backbone)
    for name in (RECORD, BUILDING):
        path = found / name
        if path.exists() and (bad := _mismatches(json.loads(path.read_text()), want)):
            if same:
                return CacheDecision("stop", found, "the cache was made under another record", bad)
            return CacheDecision("compute", target, f"{found} was made under another record", bad)
    if (found / RECORD).exists() and (found / f"{spec.representation}.json").exists():
        n = _missing(found, chips, split_dir, spec)
        if n == 0:
            return CacheDecision("reuse", found, "complete and matching")
        if same:
            return CacheDecision("compute", found, f"{n} chips missing, resumed in place")
        return CacheDecision("compute", target, f"{n} chips missing in {found}")
    if found.exists() and same:
        return CacheDecision("compute", found, "partial cache, resumed in place")
    return CacheDecision("compute", target, "no cache found")


def build_raster_cache(backbone: str, chips, split_dir, cache_dir, *, batch: int, encoder=None) -> dict:
    """Encode every chip of the split that ``cache_dir`` lacks, then write the sidecar and the record."""
    import rasterio

    from gfm4agri.embeddings.export import band_stats

    spec = get_backbone(backbone)
    chips, split_dir, cache_dir = Path(chips), Path(split_dir), Path(cache_dir)
    want = expected_record(chips, split_dir, backbone)
    for name in (RECORD, BUILDING):
        path = cache_dir / name
        if path.exists() and (bad := _mismatches(json.loads(path.read_text()), want)):
            raise SystemExit(f"{cache_dir} was made under another record, not written into:\n  "
                             + "\n  ".join(bad))
    out = cache_dir / "chips"
    out.mkdir(parents=True, exist_ok=True)
    (cache_dir / BUILDING).write_text(json.dumps(want, indent=1))

    manifest = load_manifest(chips)
    suffix, dims = _suffix(spec), int(spec.computed["dims"])
    band_names = [f"{spec.representation.upper()}_{k:03d}" for k in range(dims)]
    ids = _chip_ids(chips, split_dir)
    have = set(os.listdir(out))
    todo = [c for c in ids if c + suffix not in have]
    t0 = time.time()
    if todo:
        encoder = encoder or ENCODERS[spec.computed["encoder"]](spec, manifest, batch)
        src = chips / "chips"
        with ThreadPoolExecutor(4) as pool:
            ahead = [pool.submit(encoder.read, src, c) for c in todo[:READ_AHEAD]]
            for i, c in enumerate(todo):
                inputs = ahead[i].result()
                ahead[i] = None
                if i + READ_AHEAD < len(todo):
                    ahead.append(pool.submit(encoder.read, src, todo[i + READ_AHEAD]))
                emb = np.asarray(encoder.encode(inputs), dtype=np.float32)
                if emb.shape[0] != dims:
                    raise ValueError(f"{c}: the encoder gave {emb.shape[0]} dimensions, the registry {dims}")
                # On the grid of the chip's own mask, so embeddings and labels coincide pixel for pixel.
                with rasterio.open(src / f"{c}.mask.tif") as m:
                    crs, transform = m.crs, m.transform
                # Written under a temporary name and renamed, so a killed run never leaves a truncated
                # raster that a resumed one would take for a finished chip.
                tmp = out / f"{c}{suffix}.part"
                _write(tmp, emb, crs, transform, band_names)
                tmp.replace(out / f"{c}{suffix}")
                if (i + 1) % 100 == 0 or i + 1 == len(todo):
                    el = time.time() - t0
                    print(f"  {i + 1} / {len(todo)} chips encoded, {el / 60:.1f} min, "
                          f"{el / (i + 1):.2f} s per chip", flush=True)
        del encoder
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    encode_s = time.time() - t0

    nodata = {}
    for c in ids:
        with rasterio.open(out / f"{c}{suffix}") as d:
            nodata[c] = round(float(np.isnan(d.read(1)).mean()), 6)
    train = split_list(chips, "train", split_dir).read_text().split()
    stats = band_stats([out / f"{c}{suffix}" for c in train], f"training chips of split {split_dir.name}")
    created = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    sidecar = {
        "representation": spec.representation, "created": created,
        "source": {"backbone": backbone, **spec.computed, "notes": spec.notes},
        "file_suffix": suffix, "band_names": band_names, "dtype": "float32", "grid": manifest.get("grid"),
        "resampling": "none: encoded on the chip's own grid from its composites",
        "normalisation": stats, "chip_manifest_sha256": want["manifest_sha256"],
        "chips": [{"chip_id": c, "nodata_share": nodata[c]} for c in ids],
    }
    (cache_dir / f"{spec.representation}.json").write_text(json.dumps(sidecar, indent=2))
    record = {**want, "created": created, "chips": len(ids), "encoded_now": len(todo),
              "encode_seconds": round(encode_s, 1)}
    (cache_dir / RECORD).write_text(json.dumps(record, indent=1))
    (cache_dir / BUILDING).unlink(missing_ok=True)
    return record
```

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest -q tests/test_raster_cache.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gfm4agri/pipeline/raster_cache.py tests/conftest.py tests/test_raster_cache.py
git commit -m "Add the cache of a computed 10 m raster

A frozen per-pixel encoder run once per chip set into <root>/<backbone>/<chip set>/, one
uncompressed raster per chip, a sidecar of training-chip statistics and a record; reused only
when complete and matching, resumed chip by chip, never written into under another record.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: The runner runs computed arms

**Files:**
- Modify: `src/gfm4agri/pipeline/runner.py` (`check_chipset`; the arm loop of `run`; module docstring)
- Test: `tests/test_pipeline_runner.py`

**Interfaces:**
- Consumes: `resolve_raster_cache`, `build_raster_cache` (Task 4); `compose(..., image_root=)` (Task 3); `stub_computed` (Task 4).
- Produces: for a computed arm, `run` resolves and builds its raster cache, composes cells with `image_root`, and stamps `provenance["raster_cache"] = {"dir": ..., **cache.json}`; the dry run reports the decision; `check_chipset` asks for `computed["inputs"]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_pipeline_runner.py`)

```python
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
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest -q tests/test_pipeline_runner.py -k computed`
Expected: FAIL (`tessera_v2_medium.json missing` in the check; no `raster_cache` in provenance).

- [ ] **Step 3: Teach `check_chipset` the computed inputs** (`runner.py`)

Replace

```python
        spec = get_backbone(arm["model"]["backbone"])
        if spec.representation == "s2_monthly":
```

with

```python
        spec = get_backbone(arm["model"]["backbone"])
        if spec.computed:
            # A raster the runner computes: what its encoder reads must be there, not the raster.
            suffixes = list(spec.computed["inputs"])
        elif spec.representation == "s2_monthly":
```

- [ ] **Step 4: Run computed arms in `run`** (`runner.py`)

At the top of `run`, after `exp = load_experiment(experiment)`, add `from gfm4agri.benchmark.backbones import get_backbone`. Replace the body of `for arm_name, arm in arm_cfgs.items():` with:

```python
        mine = [c for c in cells if c.arm == arm_name]
        todo = [c for c in mine if _state(out_root / arm_name / c.tag) != "done"]
        backbone = arm["model"]["backbone"]
        computed = bool(get_backbone(backbone).computed)
        decision = None
        if arm["route"] == "cache" or computed:
            if computed:
                from gfm4agri.pipeline.raster_cache import resolve_raster_cache

                decision = resolve_raster_cache(root, scratch, backbone, chips, split_dir)
            else:
                from gfm4agri.pipeline.cache import resolve_cache

                decision = resolve_cache(root, scratch, backbone, chips, split_dir,
                                         arm["data"]["normalisation"], arm["trainer"]["precision"])
            _say(f"{arm_name}: cache {decision.action} {decision.cache_dir} ({decision.reason})"
                 + "".join(f"\n    {m}" for m in decision.mismatches))
            if decision.action == "stop" and todo:
                raise SystemExit(f"{arm_name}: {decision.cache_dir} does not match the run and "
                                 "would be written into; pass --scratch to compute elsewhere")
        _say(f"{arm_name}: {len(mine) - len(todo)} of {len(mine)} cells finished")
        if dry_run:
            rows += [{"arm": arm_name, "cell": c.tag, "state": _state(out_root / arm_name / c.tag),
                      "cache": decision.action if decision else None} for c in mine]
            continue
        if todo and decision is not None and decision.action == "compute":
            _say(f"{arm_name}: encoding into {decision.cache_dir}")
            if computed:
                from gfm4agri.pipeline.raster_cache import build_raster_cache

                build_raster_cache(backbone, chips, split_dir, decision.cache_dir,
                                   batch=encode_batch_size(arm, mach, machine))
            else:
                from gfm4agri.benchmark.cached import generate_embeddings

                generate_embeddings(backbone, chips, decision.cache_dir,
                                    normalisation=arm["data"]["normalisation"],
                                    batch_size=encode_batch_size(arm, mach, machine),
                                    num_workers=mach["encode_num_workers"],
                                    precision=arm["trainer"]["precision"], split_dir=split_dir)
            gc.collect()
        # A computed raster is read from its cache, and its record goes into every cell's provenance.
        arm_provenance, extra = provenance, {}
        if computed and todo:
            record = json.loads((decision.cache_dir / "cache.json").read_text())
            arm_provenance = {**provenance,
                              "raster_cache": {"dir": repo_relative(decision.cache_dir), **record}}
            extra = {"image_root": decision.cache_dir}
        for c in todo:
            cell_dir = out_root / arm_name / c.tag
            if _state(cell_dir) == "fit":
                cfg = compose(arm_name, arm, exp, mach, chips, split_dir, c, out_root, **extra)
                _say(f"{arm_name} {c.tag}: fit")
                if arm["route"] == "cache":
                    from gfm4agri.benchmark.cached import fit_cached

                    fit_cached(cfg, decision.cache_dir, pct=c.pct, draw_seed=c.draw,
                               output_root=out_root, num_workers=mach["num_workers"])
                else:
                    from gfm4agri.benchmark.fit import fit_end_to_end

                    fit_end_to_end(cfg)
                _stamp(cell_dir / "results.json", arm_provenance)
                gc.collect()
            _say(f"{arm_name} {c.tag}: test predictions")
            _predict(cell_dir, test_ids, mach["num_workers"])
        rows += [{"arm": arm_name, "cell": c.tag, "state": _state(out_root / arm_name / c.tag),
                  "cache": decision.action if decision else None} for c in mine]
```

(`**extra` is empty for every other arm, so experiment folders that wrap `compose` with the old eight-argument signature keep working.) Add to the module docstring's first paragraph: `a computed 10 m arm (TESSERA v2) has its raster cache reused or computed the same way (:mod:`gfm4agri.pipeline.raster_cache`)`.

- [ ] **Step 5: Run the tests**

Run: `$PY -m pytest -q tests/test_pipeline_runner.py tests/test_pipeline_cache.py tests/test_raster_cache.py`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/gfm4agri/pipeline/runner.py tests/test_pipeline_runner.py
git commit -m "Run computed 10 m arms in the K-shot runner

A computed arm's raster cache is reused or encoded like a feature cache, its cells read the
embeddings from it through image_root, and its record is stamped into their provenance. The
chip-set check asks for what the encoder reads.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: The main workflow's arms, profiles and weights

**Files:**
- Create: `configs/arms/terramind_v1_large_s2s1.yaml`, `configs/arms/thor_v1_large_s2s1.yaml`, `configs/arms/tessera_v2_medium.yaml`
- Modify: `configs/experiments/kshot.yaml`, `configs/machines/hub.yaml`, `configs/machines/cluster.yaml`, `scripts/cluster/prefetch_weights.py`
- Test: `tests/test_pipeline_config.py`

**Interfaces:**
- Consumes: the registry entries of Tasks 1 and 3.
- Produces: `kshot.yaml` arms `[terramind_v1_large_s2s1, prithvi_eo_v2_600_tl, thor_v1_large_s2s1, tessera_v2_medium, alphaearth_v1]`.

- [ ] **Step 1: Update the configuration tests** (`tests/test_pipeline_config.py`)

Replace `test_every_arm_route_matches_its_backbone`'s expected dict with

```python
    assert routes == {"terramind_v1_large_s2s1": "cache", "prithvi_eo_v2_600_tl": "cache",
                      "thor_v1_large_s2s1": "cache", "tessera_v2_medium": "raster",
                      "alphaearth_v1": "raster"}
```

Replace `test_the_main_workflow_holds_only_its_own_experiment_and_arms` with

```python
#: Arms that left kshot.yaml on 15 October 2026 for their S2 + S1 or v2 successors, kept so that
#: earlier results can be refitted and experiments can vary them.
RETIRED = {"tessera_v1", "terramind_v1_large", "thor_v1_large"}


def test_the_main_workflow_holds_only_its_own_experiment_and_arms():
    assert sorted(p.name for p in (CONFIGS / "experiments").glob("*.yaml")) == ["kshot.yaml"]
    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    assert not RETIRED & set(exp["arms"])
    assert sorted(p.stem for p in (CONFIGS / "arms").glob("*.yaml")) == sorted([*exp["arms"], *RETIRED])


def test_every_kshot_encoder_that_reads_s1_is_given_it():
    from gfm4agri.benchmark.backbones import get_backbone

    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    specs = {a: get_backbone(load_arm(a)["model"]["backbone"]) for a in exp["arms"]}
    reads_s1 = {a for a, s in specs.items()
                if "S1RTC" in s.extra_modalities or "_s1rtc.tif" in s.computed.get("inputs", [])}
    # Prithvi-EO-2.0 was pretrained on six HLS optical bands; AlphaEarth carries S1 inside its embedding.
    assert reads_s1 == {"terramind_v1_large_s2s1", "thor_v1_large_s2s1", "tessera_v2_medium"}
```

In `test_every_cache_arm_has_an_encode_batch_on_both_machines`, replace the core-arm assertion with

```python
        from gfm4agri.benchmark.backbones import get_backbone

        encoding = {a["model"]["backbone"] for a in core
                    if a["route"] == "cache" or get_backbone(a["model"]["backbone"]).computed}
        assert encoding <= set(profile["encode_batch_size"]), machine
```

- [ ] **Step 2: Run them to verify they fail**

Run: `$PY -m pytest -q tests/test_pipeline_config.py`
Expected: FAIL (old arm list, missing arm files and profile entries).

- [ ] **Step 3: Write the three arm files**

`configs/arms/terramind_v1_large_s2s1.yaml`:

```yaml
# TerraMind v1 large, frozen, on the twelve monthly Sentinel-2 and Sentinel-1 RTC composites, the tokens of the two
# modalities averaged at every position; ChannelBottleneck and the 12.9 M parameter UNet decoder of the S2 arm,
# fitted from cached features. From experiments/2026-10-13_terramind_s1_s2, where S1 added 0.026, 0.054 and 0.025
# test Macro-F1 at K = 100, 20 and 5 % on all of Estonia; in the main workflow since 15 October 2026.
route: cache                   # cache: fitted from cached encoder features; raster: from embedding rasters
data:
  normalisation: backbone      # TerraMind v1 S2L2A and S1RTC pretraining statistics
  batch_size: 2                # with accumulate_grad_batches 2, an effective batch of 4
  augment: true                # D4; the cached fit draws one of the eight cached variants
model:
  backbone: terramind_v1_large_s2s1
  lr: 1.0e-4
  weight_decay: 0.05
  head_dropout: 0.1
  loss: ce
trainer:
  precision: 16-mixed
  log_every_n_steps: 10
  accumulate_grad_batches: 2
```

`configs/arms/thor_v1_large_s2s1.yaml`:

```yaml
# THOR v1 large, frozen, on 160 m tokens, on the twelve monthly Sentinel-2 and Sentinel-1 RTC composites: S1 is a
# third band group, and the three groups are averaged at every token position, so the ChannelBottleneck and UNet
# decoder are TerraMind's (decided 15 October 2026). Training settings of thor_v1_large.
route: cache                   # cache: fitted from cached encoder features; raster: from embedding rasters
data:
  normalisation: backbone      # THOR pretraining statistics, S2 and S1
  batch_size: 2                # with accumulate_grad_batches 2, an effective batch of 4
  augment: true                # D4; the cached fit draws one of the eight cached variants
model:
  backbone: thor_v1_large_s2s1
  lr: 1.0e-4
  weight_decay: 0.05
  head_dropout: 0.1
  loss: ce
trainer:
  precision: 16-mixed
  log_every_n_steps: 10
  accumulate_grad_batches: 2
```

`configs/arms/tessera_v2_medium.yaml`:

```yaml
# TESSERA v2 Medium, frozen, from its open weights, on the twelve monthly Sentinel-2 and Sentinel-1 composites; its
# 128-dimension 10 m output is computed and cached by the runner (gfm4agri.pipeline.raster_cache), and the per-pixel
# MLP decoder of the pixel_raster group (512, 256) is trained on it, with tessera_v1's settings. The MLP was kept
# over a U-Net on 15 October 2026 (experiments/2026-10-15_tessera_decoder).
route: raster                  # cache: fitted from cached encoder features; raster: from embedding rasters
data:
  normalisation: chips         # an embedding has no pretraining statistics; the cache's training-chip statistics
  batch_size: 4
  augment: true                # D4 is exact for a per-pixel embedding
model:
  backbone: tessera_v2_medium
  lr: 1.0e-3                   # a small head trained from scratch
  weight_decay: 0.05
  head_dropout: 0.2
  loss: ce
trainer:
  precision: 16-mixed
  log_every_n_steps: 10
```

- [ ] **Step 4: Point `kshot.yaml` at the new arms**

```yaml
# The K-shot workflow's arms since 15 October 2026: every GFM on every sensor it was pretrained on (TerraMind, THOR
# and TESSERA v2 on S2 + S1; Prithvi-EO-2.0 on S2, the only input it takes; AlphaEarth's embedding carries S1).
# Run on any chip set:  poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips <dir>
name: kshot
arms: [terramind_v1_large_s2s1, prithvi_eo_v2_600_tl, thor_v1_large_s2s1, tessera_v2_medium, alphaearth_v1]
```

(the remaining lines of the file are unchanged).

- [ ] **Step 5: Add the encoding batches to the machine profiles**

`configs/machines/hub.yaml`, `encode_batch_size`:

```yaml
encode_batch_size: {terramind_v1_large: 4, prithvi_eo_v2_600_tl: 2, thor_v1_large: 2,
                    terramind_v1_large_s2s1: 2, thor_v1_large_s2s1: 1,
                    tessera_v2_medium: 16384}   # chips per batch; pixels for tessera_v2_medium
```

`configs/machines/cluster.yaml`, `encode_batch_size`:

```yaml
encode_batch_size: {terramind_v1_large: 16, prithvi_eo_v2_600_tl: 8, thor_v1_large: 8,
                    terramind_v1_large_s2s1: 8, thor_v1_large_s2s1: 4,
                    tessera_v2_medium: 65536}   # chips per batch; pixels for tessera_v2_medium
```

(Two modalities double TerraMind's tokens per frame, as its experiment's arm halved the batch; a third THOR group adds half again.)

- [ ] **Step 6: Prefetch the TESSERA v2 weights** (`scripts/cluster/prefetch_weights.py`, in `main`'s loop)

Replace

```python
        arm = load_arm(name)
        if arm["route"] != "cache":
            continue
        backbone = arm["model"]["backbone"]
```

with

```python
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
```

and in the docstring replace "the cache-route arms" with "the arms that run an encoder".

- [ ] **Step 7: Run the tests and a dry run**

```bash
$PY -m pytest -q tests/test_pipeline_config.py
$PY scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --dry-run
```

Expected: PASS; the dry run lists 15 cells over the five arms and passes the chip-set check.

- [ ] **Step 8: Commit**

```bash
git add configs/arms/terramind_v1_large_s2s1.yaml configs/arms/thor_v1_large_s2s1.yaml \
        configs/arms/tessera_v2_medium.yaml configs/experiments/kshot.yaml configs/machines/hub.yaml \
        configs/machines/cluster.yaml scripts/cluster/prefetch_weights.py tests/test_pipeline_config.py
git commit -m "Put TerraMind and THOR on S2 + S1 and frozen TESSERA v2 into kshot

kshot runs terramind_v1_large_s2s1, prithvi_eo_v2_600_tl, thor_v1_large_s2s1,
tessera_v2_medium and alphaearth_v1. tessera_v1, terramind_v1_large and thor_v1_large stay as
arm files for refits and experiments. The cluster set-up prefetches the v2 weights.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The full suite and the hub pilot

**Files:** none changed; results under `results/kshot/EE_2021_rev_mini/` and caches under `data/embeddings/`.

- [ ] **Step 1: Run the whole suite**

Run: `$PY -m pytest -q 2>&1 | tail -3`
Expected: every test passes; the count is Task 0's plus the new tests.

- [ ] **Step 2: See what the pilot will reuse**

```bash
ls results/kshot/EE_2021_rev_mini/ 2>/dev/null
$PY scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --dry-run
```

Expected: the plan of 15 cells. Cells of `prithvi_eo_v2_600_tl` and `alphaearth_v1` already in `results/kshot/EE_2021_rev_mini/` are reused, since those arms are unchanged; the three new arms show `fit`.

- [ ] **Step 3: Run the pilot detached**

```bash
mkdir -p results/kshot/EE_2021_rev_mini/logs
L=results/kshot/EE_2021_rev_mini/logs/hub_$(date +%m%d_%H%M).log
setsid nohup $PY -u scripts/run_kshot.py -e configs/experiments/kshot.yaml \
    --chips data/eurocrops_chips/EE_2021_rev_mini > "$L" 2>&1 &
echo "$L"
```

Wait with one background command that loops on `grep -q "summary of" "$L" || grep -q Traceback "$L"` with a sleep of a few minutes; do not poll by hand.

- [ ] **Step 4: Check the pilot**

```bash
grep -v -i warn "$L" | grep "kshot\|v2\|encoded\|Traceback" | tail -40
cat results/kshot/EE_2021_rev_mini/summary.csv
$PY - <<'PY'
import json, glob
for f in sorted(glob.glob("results/kshot/EE_2021_rev_mini/*/P100_draw0_seed0/results.json")):
    r = json.load(open(f))
    print(r["run"], round(r["test_metrics"]["test/F1_Score"], 3), r["params"]["trainable_total"],
          r["provenance"].get("raster_cache", {}).get("encoded_now"), r["provenance"]["git_commit"][:8])
PY
```

Expected: 15 cells with `predictions_test.npz`; `thor_v1_large_s2s1` with about 53 M trainable parameters (TerraMind's); `tessera_v2_medium` with a raster cache of 57 chips and test Macro-F1 near the decoder experiment's pilot MLP (0.209 at K = 100 %, seed 0); every new cell stamped with the branch's commit.

---

### Task 8: Merge, push, and the cluster pilot

**Files:** none changed.

- [ ] **Step 1: Bring the branch onto the current `main`**

```bash
cd /data/private/ExplainedGMF4Agri-kshot-s1
git fetch -q origin && git rebase origin/main
$PY -m pytest -q 2>&1 | tail -2
```

Expected: a clean rebase and a passing suite. If the rebase touched the files of this plan, rerun Task 7 Steps 2 to 4 before going on.

- [ ] **Step 2: Fast-forward `main` and push**

```bash
cd "/data/private/THESIS - ExplainedGMF4Agri"
git pull -q --ff-only origin main
git merge --ff-only kshot-s1-tessera-v2
git push origin main
```

- [ ] **Step 3: Update the cluster and prefetch the weights**

```bash
ssh -n utwente-hpc 'cd ~/ExplainedGMF4Agri && git pull --ff-only && export PATH=$HOME/.local/bin:$PATH && poetry run python scripts/cluster/prefetch_weights.py'
```

Expected: `weights cached`; `poetry.lock` is unchanged, so no `poetry install`.

- [ ] **Step 4: Submit the pilot, one job per arm**

```bash
ssh -n utwente-hpc 'cd ~/ExplainedGMF4Agri && TIME=0-06:00:00 bash scripts/cluster/submit.sh kshot EE_2021_rev_mini'
```

Wait with one background command looping on `squeue -u $USER -h -n kshot-terramind_v1_large_s2s1,kshot-prithvi_eo_v2_600_tl,kshot-thor_v1_large_s2s1,kshot-tessera_v2_medium,kshot-alphaearth_v1` with a sleep of a few minutes, then printing `sacct` and the `=== ... exit` lines of `results/kshot/EE_2021_rev_mini/logs/*.log`. A failure is fixed forward on `main` (in a new worktree), never edited on the cluster.

- [ ] **Step 5: Bring the results back beside the hub's and compare**

```bash
DEST=results/_cluster bash scripts/cluster/pull_results.sh kshot EE_2021_rev_mini
cat results/_cluster/kshot/EE_2021_rev_mini/summary.csv
```

Expected: 15 cells; the new arms' scores close to the hub pilot's (the same draw and seed; GPU nondeterminism moves them by a few thousandths).

- [ ] **Step 6: Remove the worktree and the branch**

```bash
unlink /data/private/ExplainedGMF4Agri-kshot-s1/data
unlink /data/private/ExplainedGMF4Agri-kshot-s1/results
cd "/data/private/THESIS - ExplainedGMF4Agri"
git worktree remove /data/private/ExplainedGMF4Agri-kshot-s1
git branch -d kshot-s1-tessera-v2
```

(`unlink` removes the link only, never what it points to.)

---

### Task 9: The documentation, once the user agrees

**Files:** `docs/phase1/protocol.md`, `docs/thesis_design.md`, `docs/phase1/pipeline.md`, `docs/phase1/pipeline_overview.md`, `CLAUDE.md`; memory files.

- [ ] **Step 1: Draft the edits and show them to the user before writing them**

Draft, as text in the conversation, an edit for each of:
- `protocol.md`: each arm's input (S1 for TerraMind, THOR and TESSERA v2; Prithvi on S2 by nature; AlphaEarth's S1 inside its embedding); TESSERA v2 from open weights, frozen, cached by the runner; THOR's three groups averaged and its caveats (untested against concatenation; RTC against ordinary backscatter); D21's 10 m decoder, the MLP, with the U-Net result behind it.
- `thesis_design.md`: the model table (TESSERA's access: open weights v2, run frozen on our composites); Phase 2, where TESSERA with weights can join the backbone tier; the fine-tuning sentence, same conclusion with the new reason (the protocol compares frozen encoders).
- `pipeline.md`: the stage table of section 1; section 7 (TESSERA v2 and its cache); section 9 (THOR on S2 + S1, the computed raster route); the K-shot workflow; section 10 (the v1.0 store's offset of about 0.7 px east and 0.4 px south; THOR's untested averaging; RTC against backscatter); section 11 (the two pilots).
- `pipeline_overview.md`: the arm list and the TESSERA flow in plain language.
- `CLAUDE.md`: the current state (arms, caches, pilots) and the arm list.

STOP until the user agrees to the text.

- [ ] **Step 2: Write the agreed edits and commit**

```bash
git add docs/phase1/protocol.md docs/thesis_design.md docs/phase1/pipeline.md docs/phase1/pipeline_overview.md CLAUDE.md
git commit -m "Record the K-shot arms on S1 and frozen TESSERA v2 in the docs

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git push origin main
```

- [ ] **Step 3: Update the memories** of the model set (`thesis-model-set.md`) and of the TESSERA store (`tessera-store-rewrite.md`, the measured offset), with their index lines in `MEMORY.md`.
