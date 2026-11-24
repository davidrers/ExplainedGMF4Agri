# XAI on TESSERA: occlusion, SHAP and Integrated Gradients, 20 November 2026

**Question.** Which months, band groups and sensors does the TESSERA model use to recognise each crop of an
Estonian field, and do three feature attribution methods (occlusion, SHAP, Integrated Gradients) agree with each
other and with the ground truth of `docs/research/xai/2026-11-16_crop_signatures`?

**What is explained.** The two-head TESSERA arm of `experiments/2026-11-09_borders_all_arms` at K = 100 %, frozen
TESSERA v2 Medium on the monthly composites with PRUE's U-Net and its crop head
(`results/2026-11-09_borders_all_arms/EE_2021_rev/tessera_v2_medium/P100_draw0_seed0/best.pt`). No model is
retrained (user, 20 November 2026). The core is not touched: `chain.py` imports it.

## Decisions (user, 20 November 2026)

| Choice | Decided |
|---|---|
| Unit and score | the field; the log-probability of its **species**, averaged over the field's interior pixels. The model's 20 classes are summed into the 13 species of `docs/research/transferability/crop_classes.csv` (winter and spring wheat are "wheat"); the classes themselves serve the sowing-form split (Q6). "A rather than B" is log p(A) − log p(B), from the same passes |
| Input groups | 12 months; 10 windows of three months; 6 band groups (Visible B02 B03 B04, Red-edge B05 B06 B07, NIR B08 B8A, SWIR B11 B12, VV, VH); 2 sensors; 72 cells (month × band group). TESSERA reads no atmospheric band |
| State test | the **training mean**: the removed cells take the mean of the 13 species' means over the labelled pixels of the training chips, so that no species (grassland is half the parcels) is the reference |
| Event test | **drop**: the frame is set to TESSERA's "no observation" (S2 all zero, S1 zero). Checked on a pilot chip: identical, to the last digit, to removing the frame from the sequence. Months, windows, sensors and month × sensor only |
| What is changed | the field's own pixels (main reading); the whole chip as a check of the evidence that comes from the surroundings |
| Reference of SHAP and IG | the same training mean: SHAP is baseline Shapley, and the gradient method is **Integrated Gradients from the mean field** |
| Not in this round | stability across seeds and the planted signal (both need refits); zones (Estonia is one zone) |

## Steps

| Step | What | Where |
|---|---|---|
| 0 | the embedding statistics of the fit, recomputed (the job deleted its cache, `best.pt` and `results.json` do not hold them); the mean field; the chain checked against `predictions_test.npz` in bf16 and fp32 | `step0.py`, cluster |
| 0, pass | set on 24 November 2026, before any check ran: in fp32, the precision the explanations use, **pixel agreement ≥ 99.0 % and parcel-majority agreement ≥ 99.5 %** with `predictions_test.npz` on the 300 test chips. Below that, nothing is explained until the cause is found. bf16 is reported, not gated | `verify.json` |
| R | the Estonian reference: the crop-signature analysis on the Estonian fields only, at species level | `reference_ee.py`, hub |
| S | the sample: test chips chosen so that every class has its quota of fields with 20 interior pixels or more; every such field of those chips is explained | `sample.py`, hub |
| 1 | occlusion: windows, months, month × sensor and sensors under drop; windows, months, band groups, sensors and cells under the mean; windows, band groups and sensors on the whole chip | `explain.py`, cluster |
| 2 | SHAP: a game over the 12 months (32 permutations, antithetic) and an exact game over the 6 band groups, both under the mean | `explain.py` |
| 3 | Integrated Gradients from the mean field, 32 steps, for the field's species and its rival; per cell, month and band group, with the sign of the gradient for the direction | `explain.py` |
| 4 | checks: completeness of SHAP and IG; convergence (two halves of the permutations, 32 against 64 steps); deletion curves, months removed in each method's order with the *other* replacement, against random orders; agreement between methods; the crop head's weights randomised | `explain.py`, `report.py` |
| 5 | assessment against the reference (Q1 when, Q2 which property, Q3 against which crop, Q4 direction, Q6 sowing form, Q7 cloud-filled months and null cells) and `REPORT.md` | `report.py` |

## How to run

    bash scripts/cluster/submit.sh experiments/2026-11-20_xai_tessera/experiment.yaml EE_2021_rev step0         # cluster
    poetry run python experiments/2026-11-20_xai_tessera/run.py -e experiments/2026-11-20_xai_tessera/experiment.yaml \
        --chips data/eurocrops_chips/EE_2021_rev --arms sample                                                  # hub

`--limit <n>` runs a step on `n` chips, for a test. **Results:** `results/2026-11-20_xai_tessera/EE_2021_rev/<step>/`.

**Outcome.** Not run yet.
