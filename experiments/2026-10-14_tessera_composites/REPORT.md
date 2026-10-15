# TESSERA driven from our monthly composites: report, 14 and 15 October 2026

This report covers both stages:
- **stage 1**, the feasibility spike on the pilot (14 October);
- **stage 2**, the fit with the encoder frozen or fine-tuned, on all of Estonia (`EE_2021_rev`, 15 October), with
  its pilot.

**Answer, stage 2.** Fine-tuning TESSERA v2 Medium on our composites beats keeping it frozen, at every budget and in
all nine paired fits.
- **Test Macro-F1, fine-tuned against frozen:** 0.667 / 0.630 at K = 100 %, 0.641 / 0.594 at 20 %, 0.589 / 0.553
  at 5 %.
- **Paired gains:** +0.037, +0.047 and +0.037, each 5 to 16 times its spread between seeds.
- **Against the current TESSERA arm** (v1.0 store embeddings on the same chips, draw and seeds):
  - the frozen v2 encoder on composites lands level, within 0.017 at every budget;
  - the fine-tuned one is ahead by 0.042, 0.044 and 0.019.
- **Against every arm run on `EE_2021_rev`:** the fine-tuned arm is the best so far at every budget.
- **Why it can be fine-tuned at all:** TESSERA now runs inside our fit on data already on disk. No new imagery was
  needed.

**Answer, stage 1.** Yes, TESSERA can run on our composites. Frozen and fed the twelve monthly Sentinel-2 and
Sentinel-1 composites every arm already reads, it separates crops on the pilot about as well as from its native
per-acquisition series.
- **Probe Macro-F1:** v2 Medium on composites 0.371 ± 0.003, against 0.370 ± 0.004 for the v1.1 native-series
  store and 0.373 ± 0.002 for the v1.0 store our arm uses today. Every composite variant lands within 0.018 of
  native.
- **The vector itself moves:** for the same v1.1 checkpoint, composite and native embeddings of a pixel have a
  median cosine of 0.76. That is the similarity of two pixels of different classes in the native embedding. So
  the composites give a different but about equally informative representation.
- **What it settles:** fine-tuning TESSERA on our own inputs needs no new imagery.
- **What it does not settle:** this is one small chip set at K = 100 %, so it says nothing about the label-budget
  curve. Stage 2 answers that.

# Stage 1: the spike

## Setup

| | |
|---|---|
| Question | Does the TESSERA encoder still separate crops when fed our 12 monthly composites instead of its native series? |
| Representations | 2 native, read from the stores: `store_v10` (v1.0, `_tessera.tif`, our current arm) and `store_v11` (v1.1 store, AWS v1.1 checkpoint). 6 on composites, encoded here, frozen: v1.1 AWS, v1.1 MPC, v2 Nano, Small, Medium and Large |
| Chip set | `EE_2021_rev_mini`: 57 chips in 5 blocks. Pool blocks of 16, 3 and 13 chips; validation block of 16; test block of 9. 817,086 labelled pixels, 1,692 parcels, 20 classes |
| Measures | (1) per-pixel cosine to `store_v11`; (2) ridge map onto `store_v11`, fitted on the pool blocks and scored as R2 on the other two; (3) per-pixel MLP probe (512, 256; ReLU, batch norm, dropout 0.2; AdamW 1e-3, weight decay 0.05; 15 epochs), leave-one-block-out over the 5 blocks, 3 seeds, pooled Macro-F1 over every labelled pixel |
| Runs | Hub, RTX A4000 shared with another job. v1.1 store read: about 1 h (90 s per chip, 6 threads). Encoding: 6.5 to 7 min per v1.1 checkpoint, 11 to 19 min per v2 student. Probe: about 40 min. No failures |

**From our composites to TESSERA's input.** The spike checked each point against TESSERA's preprocessing and
inference code (commit `e368539`):

| | Ours | Given to TESSERA |
|---|---|---|
| S2 bands | 12, wavelength order | 10, in TESSERA's order B04 B02 B03 B08 B8A B05 B06 B07 B11 B12 (ours at indices 3 1 2 7 8 4 5 6 10 11) |
| S2 scale | DN = reflectance x 10000, BOA offset removed | the same convention; raw values, standardised with the checkpoint's statistics |
| S1 | 10 log10 of the linear-power median, dB, both orbit directions merged | TESSERA stores (20 log10(power) + 50) x 200 as int16, so (2 dB + 50) x 200 clipped to [0, 32767]. The 12 frames go in as the ascending stream with its statistics. TESSERA's own means decode to -11.0 dB for VV and -17.4 dB for VH, which confirms the 20 log10 |
| Dates | 12 months | day of year of the 15th of each month, raw integers |
| Validity | composites are gap filled | every month marked valid |
| Sequence | 12 frames | resampled to the 16-step bucket by each version's own rule (`build_resample_indices` for v1.1, `encode_pixels` for v2) |

## Results

### Headline

![Probe Macro-F1, native series against composites](figures/spike_probe_f1.png)

| Representation | Input | Probe Macro-F1, mean ± sd (3 seeds) | Ridge R2 onto `store_v11` |
|---|---|---|---|
| v1.0 store (current arm) | native series | 0.373 ± 0.002 | 0.577 |
| v1.1 store, AWS | native series | 0.370 ± 0.004 | (target) |
| v1.1 AWS | composites | 0.355 ± 0.009 | 0.670 |
| v1.1 MPC | composites | 0.369 ± 0.014 | 0.678 |
| v2 Nano (1.1 M) | composites | 0.357 ± 0.002 | 0.597 |
| v2 Small (7.1 M) | composites | 0.367 ± 0.010 | 0.622 |
| **v2 Medium (21.0 M)** | composites | **0.371 ± 0.003** | 0.630 |
| v2 Large (43.8 M) | composites | 0.360 ± 0.006 | 0.624 |

The two native representations and the best composite ones (v2 Medium, v1.1 MPC) are level. The worst composite
ones (v1.1 AWS, v2 Nano) are 0.015 to 0.018 below. Seed spread runs from 0.002 to 0.014, so only those last gaps are
clearly larger than the noise.

### How far the embedding moves

![Per-pixel cosine to the native v1.1 embedding](figures/spike_cosine.png)

- **The reference scales**, from the native v1.1 embedding alone: two pixels of the same parcel sit at a median
  0.94, of the same class in different parcels at 0.83, of different classes at 0.74.
- **The same AWS checkpoint** on composites lands at 0.76 from its native output. A composite embedding is
  therefore not a substitute for the store's: a head trained on one would not carry over to the other.
- **A linear relation remains:** a ridge map from composite to native embeddings explains 67 % of the variance on
  held-out blocks, more than v1.0 native explains of v1.1 native (58 %).
- **The MPC checkpoint** lies further from the AWS-native store (0.58), as expected of a different checkpoint, yet
  probes as well as native.

### Per class

![Per-class F1, composites minus native v1.1](figures/spike_per_class.png)

- **Losses on composites:**
  - **Peas** for every encoder, by 0.11 to 0.21.
  - **Harvested green legumes**, by 0.02 to 0.06.
  - **Potatoes** for the three larger v2 students, by 0.11 to 0.15. With 2,675 pixels this is the noisiest row.
  - **Grassland** by 0.02 for every encoder, the largest class.
- **Gains:** **grain maize** gains 0.09 to 0.13 everywhere. **Winter barley** gains for the v2 students from Small
  up and loses for v1.1.
- **No change:** rye, alfalfa, dried pulses, vegetables and orchards score near zero in both, for lack of training
  parcels.

## Analysis

**The information survives the compositing.** Medians over a month, with gaps interpolated, keep what a frozen
TESSERA needs to separate most crops. Composite and native representations probe level to within about 0.015. This
fits how TESSERA was pretrained: Barlow Twins aligns the embeddings of two random sparse subsets of a pixel's series,
so the encoder is built to give a stable answer from incomplete observations. Twelve frames sits inside the range it
was trained on, whose smallest bucket is 8.

**The losses are short events.** Peas and potatoes have short canopies, and harvested green legumes are cut more
than once. A monthly median smears those events, and it removes most of the Sentinel-1 frames: about 200
acquisitions per pixel natively against 12 here. These same crops gained most from Sentinel-1 in
[`../2026-10-13_terramind_s1_s2/`](../2026-10-13_terramind_s1_s2/REPORT.md). Fine-tuning can only partly undo this,
because the frames are gone from the input. That is a reason to test whether the 24 half-month composites of
[`../2026-10-12_s2_semimonthly/`](../2026-10-12_s2_semimonthly/README.md) narrow the gap. The 12 against 24 comparison
for TerraMind gained little ([`../2026-10-12_terramind_12v24/`](../2026-10-12_terramind_12v24/REPORT.md)).

**Which encoder to build on.**
- **v2 Medium** is the best and steadiest composite representation: 0.371 ± 0.003, from 21 M parameters.
- **v2 Large** is lower (0.360); a larger student buys nothing on 12 frames.
- **v1.1 MPC** ties with v2 Medium on the mean but varies about five times more between seeds, and has 57.7 M parameters.

The user chose v2 for stage 2 on 14 October 2026; this spike supports v2 Medium as its default.

**What it means for the current TESSERA arm.** The arm's advantage over the token-grid models at low budget
(0.568 against about 0.36 at 5 % on full Estonia) does not come from TESSERA's native input alone. On this pilot
the same family of encoders is about as informative from our composites. If that holds at scale, most of the
advantage would sit in the encoder and its per-pixel, 10 m design rather than in denser input. Stage 2, which puts
the encoder in our own fit, can test that directly.

## Caveats

- **Pilot scale.** 57 chips, one draw, K = 100 % only. Absolute Macro-F1 sits near 0.37 because 20 classes are
  learned from a few blocks, and the blocks hold unequal numbers of chips (3 to 16). These numbers do not compare
  with the full-Estonia grid.
- **The probe is not the arm's fit.** It trains a per-pixel MLP on labelled pixels, not the chip-level fit of the
  main workflow.
- **The Macro-F1 differs slightly from the main workflow's.** It averages over the classes present in the
  reference (scikit-learn); `per_class_scores` also counts a class that is only predicted.
- **The composite input is an approximation of TESSERA's:**
  - every month is marked valid, though a Baltic December is mostly interpolated;
  - Sentinel-1 orbit directions are merged and handed over as the ascending stream;
  - dates are fixed at the 15th of each month.
- **The v1.0 store on disk is the earlier version.** It was read before the 7 October 2026 rewrite of the store.
- **v2's input is undocumented.** The data source v2 was trained on is not stated, and its normalisation
  statistics match neither the MPC nor the AWS statistics of v1.1. v2's own were used.
- **Timings are indicative.** The GPU was shared with another job.

# Stage 2: frozen against fine-tuned, all of Estonia

## Setup

| | |
|---|---|
| Question | Does fine-tuning TESSERA v2 Medium on our composites help, at each label budget, against the same encoder frozen? |
| Arms | `tessera_v2m_frozen` and `tessera_v2m_full` (`arms_run/`). They differ in `model.trainable` alone; everything else is in [DESIGN.md](DESIGN.md): per-pixel MLP head (512, 256), up to 64 pixels per support parcel per epoch, batches of 4,096 pixels, AdamW (head 1e-3; encoder 1e-5 after a one-epoch warm-up), bf16 |
| Chip set | `EE_2021_rev`: 4,898 training, 769 validation, 1,475 test chips. Pixel store: 62,759,466 training, 9,966,900 validation and 19,379,279 test labelled pixels |
| Protocol | K = 100, 20 and 5 % (86,909, 17,390 and 4,354 parcels), draw 0, seeds 0 to 2, 15 epochs, checkpoint at the lowest validation loss; validation on a fixed 1,015,955-pixel subset (up to 64 per parcel); test on every labelled test pixel |
| Steps | per epoch 1,244 at K = 100 %, 249 at 20 %, 63 at 5 %; the same in both arms |
| Runs | Cluster, `itc-gpu`, commit `fb8654f`. Job 623163 (frozen) 50 min, including 13 min for the pixel store and 16 min for the embedding cache. Job 623164 (fine-tuned) 3 h 46 min: 49, 14.5 and 8.4 min per cell at K = 100, 20 and 5 %, 8.7 GB GPU peak. No failures |

## Results

### Headline and the other arms

![Test Macro-F1 and mIoU against K, beside the other arms on EE_2021_rev](figures/reference_curves.png)

| Test Macro-F1, mean ± sd (3 seeds) | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| **v2 Medium on composites, fine-tuned** | **0.667 ± 0.007** | **0.641 ± 0.004** | **0.589 ± 0.002** |
| v2 Medium on composites, frozen | 0.630 ± 0.006 | 0.594 ± 0.002 | 0.553 ± 0.003 |
| TESSERA v1 store, the current arm ([`../2026-10-14_unet_baseline/`](../2026-10-14_unet_baseline/)) | 0.625 ± 0.004 | 0.597 ± 0.003 | 0.570 ± 0.008 |
| TerraMind v1 large, S2 + S1 ([`../2026-10-13_terramind_s1_s2/`](../2026-10-13_terramind_s1_s2/REPORT.md)) | 0.654 ± 0.006 | 0.565 ± 0.018 | 0.419 ± 0.022 |
| U-Net from scratch, 45 epochs ([`../2026-10-14_unet_baseline/`](../2026-10-14_unet_baseline/)) | 0.544 ± 0.015 | 0.477 ± 0.002 | 0.413 ± 0.032 |

All five rows share the chip set, the budgets, draw 0 and seeds 0 to 2. Test mIoU follows the same order:
- fine-tuned: 0.569, 0.538 and 0.481;
- frozen: 0.522, 0.484 and 0.441.

### Paired differences

![Fine-tuned minus frozen, fit by fit](figures/paired_differences.png)

| Fine-tuned minus frozen, test Macro-F1 | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| mean ± sd (fits won) | +0.037 ± 0.007 (3/3) | +0.047 ± 0.003 (3/3) | +0.037 ± 0.005 (3/3) |

Every one of the nine differences lies between +0.030 and +0.050 (+0.0296 to +0.0496).

### Training dynamics

![Validation loss and Macro-F1 per epoch](figures/validation_curves.png)

- **The fine-tuned arm's validation loss is lowest at epoch 0 or 1.** Its checkpoints come at epoch 0, 0 and 1
  (K = 100 %) and epoch 1 at 20 and 5 %. After that the loss climbs steadily, while validation Macro-F1 holds a
  plateau above its checkpoint value.
  - Validation Macro-F1 at the checkpoint, at the best epoch and at the last epoch: 0.621 / 0.639 / 0.632
    (K = 100 %), 0.593 / 0.603 / 0.597 (20 %), 0.554 / 0.574 / 0.569 (5 %).
  - The loss criterion therefore tests the fine-tuned encoder after one or two epochs, and gives away about 0.01 to
    0.02 of validation Macro-F1.
- **The frozen arm's validation Macro-F1 barely moves after epoch 2**: from checkpoint to best epoch it gains 0.003
  to 0.014.

### Per class

![Per-class test F1, fine-tuned minus frozen](figures/per_class_f1.png)

- **Fine-tuning raises almost every class at every budget.** Only alfalfa at 100 % (-0.01) and two near-zero cells
  go the other way.
- **The largest gains, +0.05 to +0.14:** rye, oats, spring wheat, spring rapeseed, peas, potatoes and beans.
  Peas and potatoes are the classes that compositing hurt most in stage 1, and they recover here.
- **The smallest gains:** grassland, winter wheat and winter rapeseed, the classes the frozen encoder already
  separates well, move by 0.00 to 0.02.

## Analysis

**Fine-tuning pays, and the gain survives label scarcity.**
- **How large and how steady:** the gain is about 0.04 Macro-F1 at every budget, and steady between seeds
  (sd 0.003 to 0.007).
- **How fast:** it arrives within the first one or two epochs. At K = 5 % that is about 126 optimiser steps on
  4,354 parcels.
- **Why the low end matters:** the encoder adapts cheaply to the task, and to our composite input, with few
  labels. A fine-tuned TESSERA keeps the flat label-budget curve that set TESSERA apart from the token-grid
  models: it loses 0.078 from 100 to 5 %, against 0.235 for TerraMind S2 + S1.

**Most of what the frozen v2 encoder lacks is adaptation, not input.**
- **The frozen v2 arm lands level with the current arm.** On composites, with a pixel sampler that sees about 11 %
  of the support pixels per epoch, it matches the current TESSERA arm (native v1.0 embeddings, every pixel, dense
  chip-level fit) to within 0.017.
- **So stage 1 holds at scale:** the composites cost little.
- **The fine-tuning gain is largest where compositing hurt.** Peas and potatoes lost most to the monthly medians
  in stage 1 and gain most here. That is consistent with the encoder learning to read monthly medians rather than
  the per-acquisition series it was pretrained on.

**What it means for RQ1 and the thesis.** RQ1 compares frozen encoders with trainable decoders, so the fine-tuned
arm is a side study, not a fair entry in that comparison:
- the other arms keep their encoders frozen, and the protocol asks for equal tuning across models;
- the frozen v2 arm, though, is a legitimate frozen arm.

Two consequences follow:
- **The TESSERA arm could become one that runs inside the pipeline.** v2 Medium on composites performs like the
  store product while sharing every other arm's input. It needs no external embedding product, closes the
  Saaremaa coverage gap, and does not depend on store rewrites.
- **"How much does adapting the encoder buy under label scarcity" is now measurable.** Here the answer is about
  0.04 Macro-F1, flat across budgets.

## Caveats

- **One draw, three seeds.** The seed spread is small, but the draw-to-draw spread is unknown.
- **The checkpoint criterion favours the frozen arm slightly.** Fine-tuned validation loss rises from epoch 1
  while its validation Macro-F1 holds. A Macro-F1 criterion would raise the fine-tuned arm by up to about 0.02 on
  validation, the frozen arm by less. This is the same open choice `pipeline.md` already records for the
  token-grid models.
- **The encoder learning rate (1e-5) and the 15 epochs were not tuned.** The fine-tuned arm peaks within two
  epochs, so a shorter schedule would cost nothing.
- **The training regime differs from the current arm's.** The v2 arms see up to 64 pixels per parcel per epoch,
  while the TESSERA v1 arm trains densely on whole chips. The frozen v2 result shows the sampler costs little.
- **The arms do not score quite the same test pixels.** The v2 arms read composites, which cover every chip,
  while the v1.0 store has no embedding for 0.66 % of test pixels (western Saaremaa), which the current arm
  scores from zeros. All arms are scored on the same labelled pixels.
- **TESSERA v2's pretraining data and preprocessing source are undocumented** (stage 1 caveats).

## Next steps

- **A Macro-F1 checkpoint, or 3 to 5 epochs.** Re-run with either for both arms. It settles how much the loss
  criterion hides of the fine-tuning gain.
- **A second draw.** It would put an interval on the gain, as the protocol's repeated draws will.
- **Optional, half-month composites.** v2 Medium on the 24 half-month composites, to see whether the classes with
  short events gain further.
- **Should the frozen v2 arm replace the store-based TESSERA arm in the main workflow?** That is the user's
  decision ("add to the main workflow"). Fine-tuning would enter the thesis as a side study beside the frozen
  comparison of RQ1.
- **For `pipeline.md`, if the user agrees:**
  - section 10: the v1.1 store covers western Saaremaa, where v1.0 has no embedding;
  - section 11: this experiment's outcome.

## Stage 2 pilot: frozen against fine-tuned on `EE_2021_rev_mini`, 14 October 2026

This pilot tests the stage 2 pipeline end to end ([DESIGN.md](DESIGN.md)); it is not the answer. It ran all
18 cells on the hub (commit `d91dae9`, 0.2 GPU hours), with tables from `report.py` and figures in
`results/2026-10-14_tessera_composites/EE_2021_rev_mini/figures/` (not committed).

| Test Macro-F1, mean ± sd (3 seeds) | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| v2 Medium, frozen | 0.168 ± 0.014 | 0.166 ± 0.016 | 0.150 ± 0.011 |
| v2 Medium, fine-tuned | 0.191 ± 0.003 | 0.176 ± 0.020 | 0.153 ± 0.012 |
| **Fine-tuned minus frozen, paired** | **+0.023 ± 0.016 (3/3)** | **+0.010 ± 0.006 (3/3)** | **+0.003 ± 0.002 (3/3)** |

- **The pipeline works.** All 18 cells finished and the checks of DESIGN.md passed beforehand. The frozen
  K = 100 % scores (0.15 to 0.18) sit where the main workflow's TESSERA v1 arm scored on its pilot's 9 test
  chips (0.164).
- **Fine-tuning helps in every paired fit, and the gain shrinks with the budget.** On the pilot that shrinking is
  partly mechanical. At K = 5 % the sampler draws 2,387 pixels per epoch: one optimiser step per epoch, 15 steps
  in all, against 10 per epoch at K = 100 %. On all of Estonia, K = 5 % gives about 50 steps per epoch.
- **The encoder moves gently.** The checkpoint epochs are the same in both arms (3, 9 to 10, 14), so at learning
  rate 1e-5 fine-tuning shifts the optimum's level, not its timing. DESIGN.md held a learning-rate trial in
  reserve should full fine-tuning diverge or not move; neither happened, so it was not run.
- **Speed.** A fine-tuned cell took 2.6 to 2.8 min at K = 100 % on the shared A4000 (40,000 pixels per epoch);
  a frozen cell under 10 s. Scaled to all of Estonia, a fine-tuned K = 100 % cell should take a few hours on the
  cluster GPU.

## Reproduce

    git clone https://github.com/ucam-eo/tessera data/external/tessera
    git -C data/external/tessera checkout e368539f7c4711e5f16fdd09801573830e34baf9
    # weights into data/weights/tessera/{v1_1,v2}/, checked against data/weights/tessera/SHA256SUMS (see README.md)
    poetry run python experiments/2026-10-14_tessera_composites/spike/fetch_store.py
    poetry run python experiments/2026-10-14_tessera_composites/spike/embed_composites.py
    poetry run python experiments/2026-10-14_tessera_composites/spike/compare.py
    poetry run python experiments/2026-10-14_tessera_composites/spike/figures.py
    # stage 2
    poetry run python experiments/2026-10-14_tessera_composites/check.py
    poetry run python experiments/2026-10-14_tessera_composites/run.py -e experiments/2026-10-14_tessera_composites/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini
    poetry run python experiments/2026-10-14_tessera_composites/report.py --chips EE_2021_rev_mini --out results/2026-10-14_tessera_composites/EE_2021_rev_mini/figures
    bash scripts/cluster/submit.sh experiments/2026-10-14_tessera_composites/experiment.yaml EE_2021_rev
    bash scripts/cluster/pull_results.sh 2026-10-14_tessera_composites EE_2021_rev
    poetry run python experiments/2026-10-14_tessera_composites/report.py --chips EE_2021_rev
