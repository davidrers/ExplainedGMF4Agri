# PRUE field boundary segmentation on TESSERA, fine-tuned end to end, 6 November 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for it. Asked by the user on
6 November 2026: "make a deeper research on frontier methods for only border segmentation, pick the most promising
one and launch an experiment fine tunning it with tessera". The survey and the reasons for the choice are in
[RESEARCH.md](RESEARCH.md).

**Question.** PRUE (CVPR 2026), a U-Net with an EfficientNet-B7 encoder that labels every pixel background, field
interior or field boundary, is the best of 18 models on the Fields of The World benchmark of Sentinel-2 field
boundaries, foundation models among them. Fed TESSERA v2 Medium's 10 m embedding and trained end to end with it, how
well does it delineate the parcels of Estonia over the whole chip, with no extent given; how much does training the
encoder add over the frozen embedding; and how does it compare with PaPs (class-agnostic PQ 0.336 frozen, 0.366
fine-tuned) and the boundary head (0.048) on the same encoder?

**What it varies.**

| Arm (`arms_run/`) | Encoder |
|---|---|
| `prue_tessera_v2_ft` | the pretrained v2 Medium student runs on the composites of every pixel of each training window and is trained with the U-Net, end to end, at 1e-5 after a one-epoch warm-up (the rate of `2026-10-15_kshot_finetune`); its output standardised with fixed statistics from 128 training chips (`tessera_ft.py`) |
| `prue_tessera_v2_frozen` | the core's frozen raster cache (`rasters.py`), the reference: every other setting the same |

**The method** (`prue.py`, after the paper and ftw-prue). Three classes per pixel. A drawn in-scheme parcel pixel is
boundary when a 4-neighbour inside the chip is another parcel or none (the delineation experiment's target, two
pixels between neighbours), interior otherwise; background is ESA WorldCover's non-agricultural land outside the
drawn parcels, exactly as the PaPs targets; WorldCover grassland and cropland outside them is ignored. smp's U-Net
with EfficientNet-B7 (ImageNet), 67.2 M parameters, first convolution taking 128 channels. Log-cosh Dice with class
weights `[0.05, 0.20, 0.75]` (PRUE's omega = 0.75). AdamW with amsgrad at 3e-3, cosine to 1e-6, weight decay 0.01,
bf16, batches of 8 random 128 x 128 windows with D4, 30 epochs (ftw-prue's). Validation on whole chips every epoch;
the checkpoint is the epoch of best validation object F1 (class-agnostic RQ at IoU > 0.5) of PRUE's own fields, as
ftw-prue monitors `val/object_f1`.

**The fields, no truth** (`prue.parcels`): `cc`, PRUE's own (4-connected regions of the interior class, under
500 m2 dropped); `grow`, `cc` grown into the boundary pixels by watershed on the boundary probability, so a field
gets back its rim; `field`, the predicted field (interior or boundary) unsplit, the floor that says what the boundary
class adds.

**Scoring** (`score_prue.py`, with `panoptic_x.py` copied from the extent experiment): the void-rule PQ of the PaPs
and extent experiments over the whole chip (`chip/<variant>`, the headline), and cut to WorldCover 30 + 40, to
30 + 40 + 90 and to the declared parcels (`oracle/<variant>`, an upper bound that uses the test labels).
Class-agnostic first, the task being the geometry; class-aware with each field's class voted by the crop-type map of
the same encoder (`results/kshot` for frozen, `2026-10-15_kshot_finetune` for fine-tuned), as PaPs' `analyse.py`.
Per pixel: the boundary F1 inside the declared extent and the three-class IoU on the pixels the target knows.

**Protocol.** K = 100 % only (a method check, no budget sweep: user, 5 November 2026), draw 0, seed 0, all of
`EE_2021_rev`. One cluster job per arm. Pilot first on `EE_2021_rev_mini` (`experiment_pilot.yaml`, 3 epochs).

**How to run.**

    poetry run python experiments/2026-11-06_prue_tessera/check.py --chips EE_2021_rev_mini
    poetry run python experiments/2026-11-06_prue_tessera/run.py -e experiments/2026-11-06_prue_tessera/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --dry-run
    bash scripts/cluster/submit.sh experiments/2026-11-06_prue_tessera/experiment_pilot.yaml EE_2021_rev_mini   # cluster
    TIME=3-00:00:00 bash scripts/cluster/submit.sh experiments/2026-11-06_prue_tessera/experiment.yaml EE_2021_rev
    bash scripts/cluster/pull_results.sh 2026-11-06_prue_tessera EE_2021_rev
    poetry run python experiments/2026-11-06_prue_tessera/run.py -e experiments/2026-11-06_prue_tessera/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev --summarise

**Results.** `results/2026-11-06_prue_tessera/<set>/<arm>/P100_draw0_seed0/`: `results.json` (with
`panoptic_metrics` and the training history), `metrics.csv`, `predictions_test.npz` (class probabilities x 255),
`best.pt`.

**Outcome.** Pending.
