# XAI on TESSERA, focused: windows, band groups and indices against an AUC map on the same blocks, 26 November 2026

**Question.** For fields the TESSERA model predicts correctly, does the evidence the XAI methods find agree with what
tells the crop apart in the data, when both are read on exactly the same blocks: three-month windows, band groups and
vegetation indices?

**Why a second run.** The first run (`experiments/2026-11-20_xai_tessera`) compared single months and 72 month x band
group cells with the crop-signature map. Its main drivers were noisy at those levels (occlusion's single months and
cells often rested on falls near zero), the reference came from another export of the composites
(`EE_2021_transfer`), and a quarter of the explained fields were predicted wrongly. This run reduces the comparison to
fewer, coarser blocks defined identically on both sides (user, 26 November 2026).

**What is explained.** The same model: the two-head TESSERA arm of `experiments/2026-11-09_borders_all_arms`,
K = 100 % (`best.pt`), with the step-0 statistics and mean field of the first run, copied with their checksums
(`results/2026-11-26_xai_tessera_blocks/EE_2021_rev/step0/`). No retraining; the core is not touched.

## Design

| Choice | Decided |
|---|---|
| Blocks | 10 windows (Jan-Mar ... Oct-Dec, all 12 inputs); 8 properties: 6 band groups (all months), NDVI, NDMI (all months); 80 cells = window x property |
| Indices on the XAI side | an **index swap**: in the hidden months the field's index is set to the mean field's, the sum of its two bands kept (NDVI on B08, B04; NDMI on B8A, B11), so only what the index measures is removed. Hiding the index's bands instead would also remove their brightness and overlap the Visible and NIR groups. EVI shares B04 and B08 with NDVI and cannot be removed apart from it: on the AUC side only |
| AUC map | **Estonian, on the same blocks, from the model's own input**: field means over the interior pixels of the `EE_2021_rev` composites, training chips only (109,469 fields); window means of each band (best band per group) and of each index; species against the rest as in the crop-signature study (`auc_map.py`, `reference/ee_block_auc.csv`). Its monthly cells match the crop-signature map at Spearman 0.83 to 0.97 per species |
| Fields | only test fields predicted correctly (stored predictions; checked again with the explained score), 40 per species for the 11 species, at most 3 per species and chip: 440 fields, 42 chips (`sample.py`) |
| Methods | occlusion (mean replacement and swap; the windows also under drop); SHAP, the 12 months (16 permutations) summed to windows and the 6 band groups exactly (indices share bands, so they cannot be players); IG from the mean field (32 steps), with each index pair split into its index part and its brightness part (`blocks.py`) |
| Comparison | per field: the top block a main AUC cue (within 0.02 of the best, so near ties count), and the Spearman of the block ranking with the AUC, both against the other species' maps as the baseline; the consensus of the methods; per-species 95 % bootstrap intervals (`compare.py`) |
| Dropped from the first run | single months, the whole-chip scope, the deletion curves, the 64-step IG check and the rival's IG, all checked there |

## Steps

| Step | What | Where |
|---|---|---|
| A | the AUC map on the blocks, from the training chips | `auc_map.py`, hub CPU |
| S | the sample | `sample.py`, hub |
| P | the pilot chip, with the block asserts | `blocks.py --pilot`, hub GPU |
| E | the 42 chips | `run_hub.sh`, hub GPU |
| C | the comparison, figures and tables | `compare.py` |

## How to run

    poetry run python experiments/2026-11-26_xai_tessera_blocks/auc_map.py --workers 24
    poetry run python experiments/2026-11-26_xai_tessera_blocks/sample.py
    bash experiments/2026-11-26_xai_tessera_blocks/run_hub.sh          # pilot, then every chip
    poetry run python experiments/2026-11-26_xai_tessera_blocks/compare.py

**Results:** `results/2026-11-26_xai_tessera_blocks/EE_2021_rev/`. **Outcome.** Not run yet.
