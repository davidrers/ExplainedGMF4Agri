# Field borders from every arm: PRUE's border head on each GFM and the baseline, 9 November 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for it. Asked by the user on
9 November 2026, after the joint TESSERA model matched both separate models
([../2026-11-07_prue_crop_tessera](../2026-11-07_prue_crop_tessera/README.md)): "the goal would be to also answer which
GFM better predict borders", with the PRUE architecture matched per group, the baseline's encoder "start from
imagenet", the token-grid decoder's fifth block to 10 m "add it", and "I dont want to change the main pipeline, for
now just want to launch each arm for whole Estonia with the kshot protocol as a experiment. Should be frozen encoder".

**Question.** With PRUE's border head and recipe beside each arm's crop head, encoders frozen, under the K-shot
protocol on all of `EE_2021_rev`: which GFM draws the field borders best, at each label budget, against a baseline
that is PRUE itself on the raw S2 + S1 composites? And what does the second head do to each arm's crop map, against
the main workflow's own results (`results/kshot/EE_2021_rev/`)?

**What it varies.** Each arm keeps its frozen encoder and gets the network that lets it draw 10 m borders the way
PRUE does, by what its input lacks (`heads.py`):

| Arm (`arms_run/`) | Main workflow arm | Input | Network (trainable parameters) |
|---|---|---|---|
| `tessera_v2_medium` | `tessera_v2_medium` | 10 m embedding, 128 dims, the core's frozen raster cache | **PRUE's U-Net** (EfficientNet-B7, ImageNet) with the border head on its 16 features, and the main workflow's crop MLP on the embedding and those features (67.2 M + 0.21 M) |
| `alphaearth_v1` | `alphaearth_v1` | 10 m embedding, 64 dims, from the chip set | the same (67.1 M + 0.18 M) |
| `unet_b7_s2s1` (baseline) | `unet_r18_scratch_s2s1` | the 12 monthly S2 + S1 composites, 168 channels | **PRUE itself**: the same U-Net, EfficientNet-B7 from ImageNet in place of the ResNet-18 from scratch, everything trained (67.2 M + 0.23 M) |
| `terramind_v1_large_s2s1` | `terramind_v1_large_s2s1` | cached tokens, 160 m | **the main workflow's token decoder** (bottleneck, learned pyramid, U-Net decoder 512-256-128-64), then a **fifth block to 10 m** (bilinear, two 3 x 3 convolutions of 64 channels) carrying PRUE's border head and the arm's crop head (53.4 M) |
| `thor_v1_large_s2s1` | `thor_v1_large_s2s1` | cached tokens, 160 m | the same (53.4 M) |
| `prithvi_eo_v2_600_tl` | `prithvi_eo_v2_600_tl` | cached tokens, 140 m | the same (62.8 M) |

The 10 m embeddings lack spatial context (each pixel is encoded on its own), so they get PRUE's U-Net; the
token-grid GFMs carry context through attention but only at the token size, so they keep their own decoder, whose
last learned map is at 20 m and whose output the main workflow stretches bilinearly to 10 m; the fifth block lets
their border head work at 10 m like PRUE's. No raw band enters a token decoder, so their borders come from what the
tokens know.

**Training** (`fit_borders.py`). Loss: PRUE's log-cosh Dice on background, interior and boundary (class weights
0.05, 0.20, 0.75) plus a weight times the crop head's cross-entropy. The weight is set as training goes
(`heads.Balance`): every 25 steps the two losses' gradient norms over the shared part (the U-Net or the token decoder)
are measured, and the weight is moved (smoothed in log space) so that the crop loss pulls 0.3 times as hard as the
border loss. A fixed weight would mean something different on every decoder; 0.3 is the median pull of the joint
TESSERA arm at a fixed 0.01, the setting that matched both separate models. Optimisers: PRUE's for its U-Net
(AdamW with amsgrad, 3e-3, weight decay 0.01, cosine to 1e-6), the main workflow arm's for the token decoder (AdamW,
1e-4, weight decay 0.05, constant) and for every crop head (the arm's learning rate and weight decay). The K-shot
protocol otherwise: budgets 100, 20 and 5 % of the trainable parcels of each class (the core's `draw_support`), draw
0, seed 0, 45 epochs, whole 224 x 224 chips in batches of 4, the eight D4 symmetries (for a token arm, the cache's
eight variants), bf16. The border target is PRUE's with WorldCover's non-farm land as background; under a budget,
parcels not drawn are ignored, as in the crop target, so the borders are as scarce as the crop labels.

**Checkpoint.** One model, one epoch: the best validation `(object F1 of the arm's own fields + crop pixel
Macro-F1) / 2`, as in the joint experiment; the epochs each criterion alone would choose are recorded beside it.

**Scoring** (`score_joint.py`, copied unchanged): the crop map as the main workflow scores it (pixel Macro-F1, parcel
scores of D28); the fields of PRUE's post-processing (`prue.parcels`, variant `grow`) by the void-rule PQ, class-
agnostic and class-aware with each field's crop voted by the arm's own crop head; the borders by precision, recall
and F1 at 0, 1 and 2 px (`borders.py`). References: the main workflow's crop scores per arm and budget
(`results/kshot/EE_2021_rev/`), and PRUE on frozen TESSERA at K = 100 % (border F1 0.862, PQ class-agnostic 0.374).

**Checks** (`check.py`, on `EE_2021_rev_mini`, 9 November 2026, all passed). For three test chips of every arm, this
folder's loader gives exactly the main workflow's standardised input (largest difference 0: the core's datamodule
after its normalisation, flattened month-major for the baseline as its `ScratchUNet` does; the core's
`CachedFeatureDataset` for a token arm). The crop target is labelled exactly on the border target's interior and
boundary, and only on drawn parcels under a budget. Every model runs forward and backward with both heads at the
chip's size; the crop loss reaches the shared part and not the border head; the balance sets a finite weight (at
initialisation 0.0002 to 0.0009, the crop loss's raw gradient being 300 to 1,600 times the border loss's). Pilot:
`run.py` with `experiment_pilot.yaml` (2 epochs, budgets 100 and 5 %) on `EE_2021_rev_mini`, on the hub's CPU, its
GPU being taken by another process.

**Files.** Copied unchanged from `../2026-11-07_prue_crop_tessera`: `prue.py`, `panoptic_x.py`, `score_prue.py`,
`borders.py`, `score_joint.py`, `rasters.py`. New: `heads.py` (the two network shapes, the losses, the balance),
`chipsets.py` (the three inputs with both targets), `fit_borders.py` (one cell), `run.py` (the arms, their inputs, the
cells, the scoring), `check.py`.

**How to run.**

    poetry run python experiments/2026-11-09_borders_all_arms/check.py --chips EE_2021_rev_mini
    poetry run python experiments/2026-11-09_borders_all_arms/run.py -e experiments/2026-11-09_borders_all_arms/experiment_pilot.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --device cpu
    TIME=4-00:00:00 bash scripts/cluster/submit.sh experiments/2026-11-09_borders_all_arms/experiment.yaml EE_2021_rev   # cluster
    bash scripts/cluster/pull_results.sh 2026-11-09_borders_all_arms EE_2021_rev
    poetry run python experiments/2026-11-09_borders_all_arms/run.py -e experiments/2026-11-09_borders_all_arms/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --summarise

**Results.** `results/2026-11-09_borders_all_arms/<set>/<arm>/P<k>_draw0_seed0/`: `results.json` (with
`crop_metrics`, `parcel_metrics`, `panoptic_metrics`, `border_metrics`, the history with the crop weight per epoch),
`metrics.csv`, `predictions_test.npz`, `borders_per_chip.npz`, `best.pt`; `summary.csv` beside the arms.

**Outcome, 10 November 2026** ([REPORT.md](REPORT.md)). Every cell ran on all of `EE_2021_rev`. Among the GFMs,
TESSERA draws the field borders best (F1 at 1 px 0.868, 0.811 and 0.761 at K = 100, 20 and 5 %), with AlphaEarth
within 0.025, but it only ties the baseline, PRUE from ImageNet on S2 + S1 (0.868, 0.812, 0.752). The token-grid GFMs
draw weaker borders and lose more of them as labels get scarce (Prithvi 0.822 to 0.579, TerraMind 0.797 to 0.542,
THOR 0.658 to 0.432). The border head lifts the 10 m arms' crop maps by 0.03 to 0.07 pixel Macro-F1 over the main
workflow's MLP, and lowers the token-grid arms' by 0.02 to 0.16, whose crop head here reads only the border-trained
fifth block. The baseline ties TESSERA on the crop map too and is 0.04 to 0.10 above the main workflow's ResNet-18 from
scratch; one draw and one seed per cell.
