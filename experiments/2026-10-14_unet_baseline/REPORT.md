# A from-scratch 2D U-Net baseline on the monthly Sentinel-2 chips, on all of Estonia: report, 14 October 2026

**Answer.** Under the protocol's 15 epochs, the U-Net trained from scratch is clearly below both GFM arms at every
label budget. But 15 epochs undertrain it: trained for 45, it gains 0.05 to 0.07 Macro-F1 and wins all nine paired
fits. Trained that long, it trails TerraMind v1 large by 0.084 at K = 100 % and 0.034 at 20 %, and draws level at
5 %. TESSERA stays well ahead at every budget, and furthest ahead when labels are scarce.

| K | U-Net, 15 epochs | U-Net, 45 epochs | TerraMind v1 large, S2 | TESSERA + MLP |
|---|---|---|---|---|
| 100 % | 0.498 | 0.544 | **0.628** | 0.625 |
| 20 % | 0.415 | 0.477 | 0.511 | **0.597** |
| 5 % | 0.342 | 0.413 | 0.394 | **0.570** |

Test Macro-F1, mean of three seeds on one draw; all four arms were fitted on the same parcels. The two GFM arms
trained for 15 epochs.

Two findings matter for RQ1:
- **The baseline's training budget changes the answer.** At 15 epochs the comparison favours the GFMs by a further
  0.05 to 0.07. That part is the baseline stopping early, not the GFMs being better.
- **TerraMind's pretraining does not help it more when labels are scarce.** Against the U-Net trained for 45 epochs,
  its lead shrinks from 0.084 at K = 100 % to nothing at 5 %. TESSERA's lead does the opposite, growing from 0.081
  to 0.157.

## Setup

| | |
|---|---|
| Question | Where does a plain U-Net trained from scratch on the same chips sit against the frozen-GFM arms at K = 100, 20 and 5 %? Does the protocol's 15 epochs undertrain a model that learns its features from scratch? |
| Arms | `unet_r18_scratch`: the `segmentation_models_pytorch` U-Net with a ResNet-18 encoder, randomly initialised, 14.8 M parameters, all trained. Input: the 12 monthly S2 composites x 12 bands as 144 channels (early fusion); 15 epochs. `unet_r18_scratch_45ep`: the same model for 45 epochs. `tessera_v1`: the core arm, TESSERA v1 embeddings (128 dimensions at 10 m) with a per-pixel MLP, 0.2 M trainable parameters, 15 epochs. Reference: `terramind_v1_large` of [`2026-10-13_terramind_s1_s2`](../2026-10-13_terramind_s1_s2/REPORT.md), on S2 only and 160 m tokens, with the `ChannelBottleneck` and the UNet decoder, 53.3 M trainable parameters, 15 epochs |
| U-Net training | AdamW, lr 1e-3, constant (no schedule), weight decay 0.05, batch 4, D4 augmentation, cross-entropy ignoring unlabelled pixels, dropout 0.1 before the final 1 x 1 convolution, 16-bit mixed precision. Per-band normalisation from the chip set's manifest, the same for every month |
| Chip set | `EE_2021_rev`, all of Estonia under the `revised` screening; split `blocks4_buf1600_seed0__6b0eb4cb`: 4,898 training, 769 validation and 1,475 test chips |
| Protocol | K = 100, 20 and 5 % of the parcels of each class: 86,909, 17,390 and 4,354 parcels on 4,898, 4,213 and 2,706 chips. Draw 0; seeds 0, 1 and 2; checkpoint of lowest validation loss. All four arms share the draw and the seeds, so they are compared fit by fit |
| Runs | UT HPC `itc-gpu`, `hpc-node30`, one job per arm. 15 epochs: job 621675, 2 h 56 min, commit `ac3abbc`. TESSERA: job 621676, 3 h 11 min, `ac3abbc`. 45 epochs: job 621677, 9 h 1 min, `53f7783`, which differs from `ac3abbc` only in an unrelated experiment folder and in this folder's README and report, so all three ran the same code. 27 fits, 14.1 GPU hours of fitting, no failure. The TerraMind fits are those of the S1/S2 experiment (job 621197, `bd4cd1f`) |

## Results

### Headline

![Test Macro-F1 and mIoU against K](figures/kshot_curves.png)

**Test Macro-F1**, mean ± standard deviation over 3 seeds

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net, 15 epochs | 0.498 ± 0.018 | 0.415 ± 0.015 | 0.342 ± 0.009 |
| U-Net, 45 epochs | 0.544 ± 0.015 | 0.477 ± 0.002 | 0.413 ± 0.032 |
| TESSERA + MLP | 0.625 ± 0.004 | **0.597** ± 0.003 | **0.570** ± 0.008 |
| TerraMind v1 large | **0.628** ± 0.009 | 0.511 ± 0.019 | 0.394 ± 0.021 |

**Test mIoU**, mean ± standard deviation over 3 seeds

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net, 15 epochs | 0.394 ± 0.018 | 0.319 ± 0.010 | 0.257 ± 0.007 |
| U-Net, 45 epochs | 0.437 ± 0.014 | 0.372 ± 0.002 | 0.313 ± 0.023 |
| TESSERA + MLP | 0.524 ± 0.002 | **0.491** ± 0.003 | **0.458** ± 0.007 |
| TerraMind v1 large | **0.526** ± 0.007 | 0.407 ± 0.021 | 0.297 ± 0.017 |

The arms differ most in how fast they fall as labels get scarce. From K = 100 to 5 %, Macro-F1 falls by:

| Arm | Fall in Macro-F1 |
|---|---|
| TESSERA + MLP | 0.055 |
| U-Net, 45 epochs | 0.131 |
| U-Net, 15 epochs | 0.156 |
| TerraMind v1 large | 0.234 |

So the TerraMind curve crosses the 45-epoch U-Net between K = 20 and 5 %, in both metrics. At full labels, TerraMind
and TESSERA are level.

### Paired differences

![Paired differences in test Macro-F1](figures/paired_differences.png)

**Paired difference in test Macro-F1**, mean ± standard deviation (fits won by the first arm)

| Pair | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net, 45 minus 15 epochs | +0.046 ± 0.033 (3/3) | +0.062 ± 0.015 (3/3) | +0.071 ± 0.040 (3/3) |
| U-Net 15 ep. minus TerraMind | −0.130 ± 0.017 (0/3) | −0.095 ± 0.030 (0/3) | −0.052 ± 0.013 (0/3) |
| U-Net 45 ep. minus TerraMind | −0.084 ± 0.022 (0/3) | −0.034 ± 0.020 (0/3) | +0.019 ± 0.053 (2/3) |
| U-Net 15 ep. minus TESSERA | −0.127 ± 0.014 (0/3) | −0.182 ± 0.012 (0/3) | −0.228 ± 0.017 (0/3) |
| U-Net 45 ep. minus TESSERA | −0.081 ± 0.020 (0/3) | −0.120 ± 0.003 (0/3) | −0.157 ± 0.024 (0/3) |

Within a pair, both arms used the same parcels, seed and test chips.
- **45 against 15 epochs.** Every fit gains, and the mean gain grows as K falls. The number of epochs is fixed, so
  a smaller K also means fewer optimiser steps: at 15 epochs and batch 4, about 18,400, 15,800 and 10,100 steps at
  K = 100, 20 and 5 %. The smallest gains are single fits: +0.009 at K = 100 %, seed 1, and +0.024 at 5 %, seed 1.
- **Against TerraMind.** At 15 epochs the U-Net loses every fit. At 45 it still loses every fit at K = 100 and 20 %,
  by more than the spread. At 5 % it wins two of three fits by +0.04 and +0.06 and loses the third by −0.04; the
  mean, +0.019, is well within the spread.
- **Against TESSERA.** The U-Net loses all 18 fits, and the gap widens as K falls, at either length.

### Training dynamics

![Validation loss and Macro-F1 per epoch](figures/validation_curves.png)

**Checkpoint epochs**, per fit, as the epoch of lowest validation loss / the epoch of highest validation Macro-F1

| Arm | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| U-Net, 15 epochs | 12, 13, 12 / 14, 13, 12 | 12, 14, 14 / 12, 14, 12 | 14, 14, 14 / 14, 14, 13 |
| U-Net, 45 epochs | 43, 26, 36 / 39, 39, 36 | 38, 38, 35 / 38, 44, 35 | 36, 27, 44 / 38, 44, 44 |
| TESSERA + MLP | 12, 9, 14 / 9, 9, 14 | 11, 14, 13 / 11, 13, 13 | 13, 10, 14 / 13, 10, 14 |
| TerraMind v1 large | 11, 10, 7 / 11, 12, 12 | 6, 11, 7 / 13, 14, 14 | 9, 10, 9 / 13, 13, 14 |

- **The 45-epoch fits are the 15-epoch fits continued.** Training is deterministic and the learning rate constant.
  The validation losses of the two arms agree to every digit for epochs 0 to 14 (checked on K = 5 %, seed 1), and
  in the figure the markers of the 15-epoch fits sit on the 45-epoch lines. So the "45 minus 15 epochs" pair
  measures training longer and nothing else.
- **At 15 epochs the U-Net is still learning.** Its checkpoint is at epoch 12, 13 or 14 in all nine fits: the
  validation loss was still falling when the fit stopped.
- **At 45 epochs it is close to its plateau, though not everywhere.**
  - The checkpoints spread over epochs 26 to 44.
  - At K = 100 %, validation Macro-F1 levels off at 0.53 to 0.55 from about epoch 30.
  - At K = 5 % it is still rising at epoch 44.
  - The validation loss jumps from epoch to epoch, as expected with a constant learning rate of 1e-3 at batch 4.
- **The GFM arms settle earlier.**
  - TerraMind reaches its lowest validation loss at epochs 6 to 11. At K = 20 % the loss then rises, a sign of
    overfitting, while its validation Macro-F1 still climbs to epochs 13 and 14. This is the open question of the
    checkpoint criterion.
  - TESSERA flattens by about epoch 10, with small further drops.
- **Validation agrees with test.** Validation Macro-F1 at the checkpoint is within 0.006 of the test value for every
  arm and budget.

### Per class

![Per-class test F1, paired differences](figures/per_class_f1.png)

**What training longer buys: the small classes.** Classes with the most training parcels barely move from 15 to 45
epochs: grassland, winter wheat and winter rapeseed change by at most 0.02. The gains are in the smaller classes:

| Crop, training parcels at K = 100 % | K = 100 % | K = 20 % | K = 5 % |
|---|---|---|---|
| Potatoes, 1,880 | +0.40 | +0.03 | |
| Buckwheat, 607 | +0.12 | +0.44 | +0.25 |
| Rye, 764 | +0.00 | +0.22 | +0.36 |
| Legumes harvested green, 7,159 | +0.05 | −0.03 | +0.19 |
| Oats, 3,817 | +0.03 | +0.04 | +0.14 |
| Beans, 947 | +0.08 | +0.12 | +0.09 |
| Spring rapeseed, 713 | +0.06 | +0.10 | +0.10 |

**U-Net (45 epochs) against TerraMind.**
- At K = 100 %, the U-Net is behind on most crops, by most on rye −0.23, potatoes −0.21, spring rapeseed −0.19,
  oats −0.18 and alfalfa −0.17.
- At K = 5 %, it is level on most crops and ahead on buckwheat, +0.38, and rye, +0.12.
- Grassland and winter rapeseed are level at every budget.

**U-Net (45 epochs) against TESSERA.**
- The U-Net is behind on nearly every crop at every budget.
- Potatoes are the extreme: −0.35, −0.71 and −0.66.
- At K = 5 %, it is also behind by 0.21 to 0.28 on spring wheat, maize, peas, spring rapeseed, beans and buckwheat.
- Only grassland, winter wheat and winter rapeseed are within 0.03.

No arm detects fresh vegetables or orchards and fruits (F1 zero everywhere). Dried pulses reach F1 0.02 only with
TESSERA at K = 100 %.

## Analysis

**The 15-epoch protocol undertrains a model that learns from scratch, and the bias favours the GFMs.**
- At 15 epochs, every U-Net fit keeps one of its last three epochs. Training to 45 epochs adds 0.046 to 0.071 and
  wins every fit.
- The GFM arms train only a decoder or a head on fixed features. Their validation losses bottom out before the epoch
  limit: TerraMind at epochs 6 to 11, TESSERA around 10.
- So equal epochs are not equal convergence. A comparison at 15 epochs measures how fast each arm learns as much as
  how well it learns.
- The gap is largest at small K, where a fixed number of epochs gives the fewest optimiser steps. That is where
  RQ1 asks its question.
- This bears on the protocol's decision of equal training steps for every model and K (2 October 2026), which still
  needs a number: 15 epochs is too few for this baseline.

**Against a baseline trained long enough, TerraMind's lead is small and vanishes when labels are scarce.**
- The lead falls from +0.084 at K = 100 % to +0.034 at 20 % and to nothing at 5 %. This runs against the
  expectation that pretraining matters most with few labels.
- Two properties of the token-grid arm may explain it. Neither was tested here:
  - **Size.** Its trainable part is larger than the whole U-Net, 53.3 M parameters in the bottleneck and decoder
    against 14.8 M. At K = 20 % its validation loss rises after epoch 6 to 11, which suggests it overfits.
  - **Resolution.** It predicts 10 m pixels from 160 m tokens, so the decoder must learn from the labels how to
    upsample parcel boundaries, and few labels make that harder.
- The U-Net, by contrast, has full-resolution skip connections and fewer parameters to fit.

**TESSERA's lead is large and grows as labels get scarcer, but part of it is input.**
- TESSERA loses only 0.055 from K = 100 to 5 %. Its per-pixel embeddings separate the crops well enough for a head
  of 0.2 M parameters.
- It does not see the same data as the U-Net. Its embeddings come from each pixel's 2021 Sentinel-1 and Sentinel-2
  time series, not from twelve monthly Sentinel-2 composites.
- The S1/S2 experiment measured what adding Sentinel-1 to Sentinel-2 is worth on these chips: +0.025 to +0.054
  for TerraMind. Even if the U-Net gained as much, it would stay well behind TESSERA at K = 20 and 5 %.

**For RQ1:**
- The baseline is ready to anchor the K curves, once the training budget is set.
- Trained to near convergence, it places the token-grid GFM's benefit at about 0.03 to 0.08 Macro-F1 with moderate
  to full labels, and at nothing with 5 % of parcels.
- At the protocol's 15 epochs, the same comparison would overstate that benefit by 0.05 to 0.07.

## Caveats

- **Sample size.** One draw and three seeds. The K = 5 % comparison with TerraMind (+0.019 ± 0.053) is a draw, not
  a win.
- **Unequal epochs.** The 45-epoch U-Net is compared with GFM arms trained for 15 epochs. Their validation losses
  suggest they would gain little from more epochs; that is untested, and TerraMind's validation Macro-F1 was still
  rising at its last epochs.
- **Even 45 epochs may stop short at K = 5 %.** Validation Macro-F1 is still rising at epoch 44.
- **The checkpoint criterion adds variance at 45 epochs.** At K = 5 %, seed 1 kept epoch 27 although its validation
  Macro-F1 peaked at epoch 44. It scores 0.377 against 0.434 and 0.428 for the other seeds, most of the spread at
  that budget.
- **No tuning.** The U-Net has one setting (lr 1e-3, constant, batch 4), taken from the end-to-end arms. A
  learning-rate schedule, a larger batch or another rate could move its level. The protocol's equal tuning trials
  are not implemented for any arm.
- **Resolution groups.** The U-Net predicts at 10 m with the spatial context of a 224 x 224 chip. TerraMind works
  on 160 m tokens, and TESSERA per pixel without spatial context. Under D21 of `protocol.md` Macro-F1 is compared
  within a resolution group, and the U-Net belongs to neither, so here it sits beside both as a reference.
- **Input.** TESSERA's embeddings include Sentinel-1; the U-Net and this TerraMind arm see Sentinel-2 only.
- **Normalisation.** One mean and standard deviation per band, from 200 chips across the country with all months
  pooled; no per-month statistics.

## Next steps

1. **Set the shared training budget for the K sweep.** This is your decision, to be written into `protocol.md`. The
   options are 45 epochs for every arm, or a fixed number of optimiser steps at every K, which gives small budgets
   more epochs.
2. **Test whether the GFM arms gain from 45 epochs.** Rerun TerraMind S2 and TESSERA at 45 epochs on the same draw
   and seeds, about 9 to 10 h per arm on the cluster. That settles whether the gaps measured here hold at equal
   training length.
3. **Optional sensitivity check:** a U-Net on S2 + S1, as suggested earlier. It would not replace the S2-only
   baseline.
4. **More draws**, to settle K = 5 %.
5. **Adding the U-Net to the main workflow is your decision.** It would move the `end_to_end` route for an unfrozen
   model and the `ScratchUNet` factory from `run.py` and `scratch_unet.py` into the core, with tests and
   documentation. The change would be described before it is made.

## Reproduce

    bash scripts/cluster/submit.sh experiments/2026-10-14_unet_baseline/experiment.yaml EE_2021_rev
    bash scripts/cluster/pull_results.sh 2026-10-14_unet_baseline EE_2021_rev
    bash scripts/cluster/pull_results.sh 2026-10-13_terramind_s1_s2 EE_2021_rev    # the TerraMind reference
    poetry run python experiments/2026-10-14_unet_baseline/report.py --chips EE_2021_rev
