# The two-head workflow: crop types and field borders from every arm, beside the crop-only K-shot workflow

Design agreed with the user on 11 November 2026, section by section ("basically move the experiment set up as a
core", "go with A", "yes, go ahead").

## 1. Purpose

The main workflow maps crop types (`configs/experiments/kshot.yaml`). The user asked on 11 November 2026 to add, as a
second workflow of the official pipeline, the variant with **two heads**: a crop-type head and a field-border head on
every arm, so that the pipeline answers which GFM draws the field borders best beside its crop map. The crop-only
workflow stays as it is.

The variant is `experiments/2026-11-09_borders_all_arms`, run on all of `EE_2021_rev` on 10 November 2026
(commit `f7046d9`). It moves into the core unchanged in behaviour, and its results are reused, not rerun.

## 2. Decisions

| Decision | Choice | Source |
|---|---|---|
| What the workflow is | The border experiment's setup exactly: six arms, frozen encoders, K = 100, 20, 5 %, draw 0, seed 0, 45 epochs, whole chips in batches of 4, crop weight balanced to 0.3 of the border pull, checkpoint (object F1 + crop Macro-F1) / 2 | User, 11 November 2026 |
| The baseline of this workflow | PRUE's U-Net (EfficientNet-B7 from ImageNet) on S2 + S1, as in the experiment. The crop-only workflow keeps its ResNet-18 from scratch | User, 9 and 11 November 2026 |
| Fine-tuning | Not in the workflow; it stays an experiment (protocol D29) | User, 11 November 2026 |
| Results | Reused from `results/2026-11-09_borders_all_arms/EE_2021_rev/`, once the core is shown to reproduce the experiment (section 5); no rerun | User, 11 November 2026 |
| Code | Approach A: a core package `gfm4agri.borders` beside the crop-only modules, reached from `scripts/run_kshot.py` by the spec's `workflow: borders`; the experiment's plain PyTorch loop kept | User, 11 November 2026 |
| Docs | protocol D30, pipeline.md, pipeline_overview.md, CLAUDE.md, the pipeline page; `thesis_design.md` untouched (no new research question) | User, 11 November 2026 |
| Pipeline page | Only this change's hunks of `build_page.py` and `index.html` are committed; another session's uncommitted edits there stay uncommitted | User, 11 November 2026 |
| Report | `docs/phase1/results_estonia_borders/`, after `docs/phase1/results_estonia/REPORT.md` | User, 11 November 2026 |

**Out of scope:** a rerun of the grid; fine-tuned arms; another crop weight; the crop-only workflow's baseline (an
open decision of its own); a crop head for the token-grid arms that reads more than the fifth block; Latvia.

## 3. The workflow

Each arm keeps its frozen encoder and gets the network that lets it draw 10 m borders the way PRUE does, by what its
input lacks:

| Arm (`configs/arms_borders/`) | Crop-only arm it extends | Input | Network |
|---|---|---|---|
| `tessera_v2_medium` | `tessera_v2_medium` | 10 m embedding, 128 dims, the core's raster cache | PRUE's U-Net (EfficientNet-B7, ImageNet) with the border head on its 16 features, and the crop MLP (512, 256) on the embedding and those features |
| `alphaearth_v1` | `alphaearth_v1` | 10 m embedding, 64 dims, from the chip set | the same |
| `unet_b7_s2s1` (baseline) | `unet_r18_scratch_s2s1` | the 12 monthly S2 + S1 composites, 168 channels | PRUE itself: the same U-Net, everything trained |
| `terramind_v1_large_s2s1` | `terramind_v1_large_s2s1` | cached tokens, 160 m | the crop-only arm's token decoder, then a fifth block to 10 m (bilinear, two 3 x 3 convolutions of 64 channels) carrying both heads |
| `thor_v1_large_s2s1` | `thor_v1_large_s2s1` | cached tokens, 160 m | the same |
| `prithvi_eo_v2_600_tl` | `prithvi_eo_v2_600_tl` | cached tokens | the same |

**Targets.** The border target is PRUE's three classes (background, interior, boundary) from the declared parcels,
with WorldCover's non-agricultural land as background; the crop target is the class of every in-scheme pixel of a
drawn parcel. Under a budget, parcels not drawn are ignored in both, so the borders are as scarce as the crop labels.

**Loss.** PRUE's log-cosh Dice (class weights 0.05, 0.20, 0.75) plus a weight times the crop cross-entropy. The
weight is set during training so that, on the shared part (the U-Net or the token decoder), the crop loss's gradient
norm is 0.3 times the border loss's, measured every 25 steps and smoothed in log space.

**Optimisers.** PRUE's for its U-Net (AdamW, amsgrad, 3e-3, weight decay 0.01, cosine to 1e-6); the crop-only arm's
for the token decoder and for every crop head.

**Checkpoint.** The epoch of best validation `(object F1 of the arm's own fields + crop pixel Macro-F1) / 2`; the
epochs each criterion alone would pick are recorded.

**Scores.** The crop map as the crop-only workflow scores it (pixel Macro-F1, parcel scores of D28); the fields of
PRUE's post-processing (`grow`, whole chip, no test label) by the void-rule PQ, class-agnostic and class-aware with
each field's crop voted by the arm's own crop head; the borders of those fields by precision, recall and F1 at 0, 1
and 2 px against the declared borders, with recall by what lies across the border.

## 4. Code

**Package `src/gfm4agri/borders/`**, from the experiment's modules:

| Module | From | Holds |
|---|---|---|
| `prue.py` | `prue.py` | the border target, the log-cosh Dice, PRUE's U-Net builder, the field post-processing (`cc`, `grow`, `field`) |
| `networks.py` | `heads.py` | `CropHead`, `UNetJoint`, `TokenJoint`, the crop target and loss, `shared_grad_norms`, `Balance` |
| `data.py` | `chipsets.py` | `RasterChips`, `StackChips`, `TokenChips`, `collate`, `train_chips` |
| `fit.py` | `fit_borders.py` | one cell: training, validation, the combined checkpoint, test predictions, resuming from `last.pt` |
| `scoring.py` | `score_joint.py`, `score_prue.py`, `borders.py`, `panoptic_x.py` | crop scores, the void-rule PQ on the `chip` extent, the pixel scores, the border scores; the earlier experiments' other extents (WorldCover, oracle) are dropped |
| `runner.py` | `run.py` | the spec and arm loading, the input of each arm (the core's raster and feature caches), the cells, provenance, scoring, `summary.csv` |

**Configs.** `configs/experiments/kshot_borders.yaml` (the experiment's spec with `name: kshot_borders` and
`workflow: borders`); the six arm files in `configs/arms_borders/`, each naming its crop-only arm (`kshot_arm`) and
its input (`raster`, `stack`, `token`). An experiment overrides one from its own `arms_borders/`, as `arms/` works.

**Entry.** `scripts/run_kshot.py` hands a spec with `workflow: borders` to `gfm4agri.borders.runner`, with the same
command line plus `--device` and `--rescore`. `scripts/cluster/submit.sh` and `pull_results.sh` are unchanged.
Results: `results/kshot_borders/<chip set>/<arm>/P<k>_draw<d>_seed<s>/` (`results.json` with `crop_metrics`,
`parcel_metrics`, `panoptic_metrics`, `border_metrics`; `metrics.csv`; `predictions_test.npz`;
`borders_per_chip.npz`; `best.pt`) and `summary.csv`.

**WorldCover as a chip-set product.** `experiments/2026-11-05_parcel_extent/fetch_worldcover.py` moves to
`scripts/hub/build_worldcover_chips.py`, writing `data/worldcover/<chip set>/worldcover.npz` as now. The runner
reads the chip set's own file, or its parent's for a pilot set (`subset.json`), and stops when there is none.

## 5. Checks that make the reused results valid

1. **Scoring.** The core's scorer rescores the 16 cells of `results/2026-11-09_borders_all_arms/EE_2021_rev/` from
   their stored predictions; every crop, PQ (chip extent) and border number must equal the experiment's.
2. **Training.** On `EE_2021_rev_mini`, two epochs at K = 100 % of one arm per input (`tessera_v2_medium`,
   `unet_b7_s2s1`, `terramind_v1_large_s2s1`), with the core and with the experiment's code, same seed, on the
   CPU: the history (losses, crop weights, validation scores) and the final weights must be identical.

Both are kept as a script (`scripts/hub/check_borders_core.py`) so they can be repeated.

## 6. Tests

Unit tests on toy arrays: the border and crop targets (budget, WorldCover), the log-cosh Dice, `Balance`, the output
shapes of both networks and that the crop loss reaches the shared part and not the border head, the field
post-processing, the void-rule PQ and the border scores (the declared parcels score 1 against themselves); the spec
and arm loading; a dry run of the workflow.

## 7. Documentation

* `docs/phase1/protocol.md`: D30, the two-head workflow (targets, loss balance, checkpoint, metrics, its baseline).
* `docs/phase1/pipeline.md`: the stage in the table of section 1 and its own section with the reasons; WorldCover;
  section 10, known gaps (no fine-tuning; the token-grid crop head reads only the border-trained block; same-crop
  borders mostly missed; one draw and one seed); section 11, the border experiment's grid as the workflow's first
  run and the TESSERA experiments its decisions rest on.
* `docs/phase1/pipeline_overview.md`: the workflow in plain language with a diagram.
* `CLAUDE.md`: the second workflow under Current state, its spec, package and WorldCover builder under Running the
  pipeline and Repository layout.
* The pipeline page: the second workflow and its arms, rebuilt.

## 8. Report

`docs/phase1/results_estonia_borders/`: `REPORT.md`, `report.py`, `maps.py`, `figures/`, `tables.md`. Sections: in
short; what was done; the models; results (borders and fields against K; the crop map with and without the border
head; which borders are found; frozen against fine-tuned; the maps on the tiles and blocks of the crop report's
section 3.5); comparison with published results (the Estonian crop-type results; reference delineation scores of
PRUE and others, not comparable one to one); caveats and next steps. It reads, without writing there:
`results/2026-11-09_borders_all_arms/EE_2021_rev/`, `results/kshot/EE_2021_rev/`,
`results/2026-10-15_kshot_finetune/EE_2021_rev/`, and the TESSERA PRUE and joint experiments' results.

## 9. Order of work

This spec; the implementation plan; the core code and tests with both checks passing; the docs; the report. Each step
committed by name and pushed to `main`.
