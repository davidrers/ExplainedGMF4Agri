# Parcel-level test metrics beside the pixel-level ones, 16 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) was edited for the experiment, and nothing
is refitted: it scores the test predictions the fits already stored. `parcel_metrics.py` is the scorer it ran;
it moved to the main workflow afterwards as `src/gfm4agri/benchmark/parcel_metrics.py` (protocol D28).

**Question.** The task is pixel-level segmentation, but EuroCrops labels are parcel declarations and the
published EuroCrops benchmarks score parcels. Scored per parcel, with each parcel taking the class most of its
pixels were predicted as, what are the test Macro-F1, overall accuracy and kappa of the full-Estonia fits, how
far do they sit from the pixel scores, and does the ranking of the arms change? Is the voting rule robust to
parcel edges, to parcels cut by the test area and to very small parcels?

**What it varies.** Only the scoring. The fits are those of the 15-epoch grid of the current arms
(`results/kshot_15ep/EE_2021_rev/`: TerraMind and THOR on S2 + S1, Prithvi on S2, TESSERA v2 and AlphaEarth with
the per-pixel MLP, one seed) and the Sentinel-2 U-Net baseline at 15 and 45 epochs
(`results/2026-10-14_unet_baseline/EE_2021_rev/`, three seeds). The rule: every parcel of the 1,475 test chips,
pooled across the chips it touches, takes the majority class of its labelled pixels, ties to the lowest class
index. Three variants test it: only interior pixels vote, parcels reaching outside the test chips are left out,
parcels under 16 pixels are left out.

**How to run.**

    poetry run python experiments/2026-10-16_parcel_metric/report.py --chips EE_2021_rev

**Results.** `results/2026-10-16_parcel_metric/EE_2021_rev/` (`cells.csv`, `per_class.csv`, `size_bins.csv`);
figures and tables in `figures/`.

**Outcome.** Per parcel, every arm scores lower than per pixel, by 0.002 to 0.078 Macro-F1, because the parcel
score gives every parcel the same weight and small parcels are classified far worse: a parcel under 0.16 ha is
right 24 to 38 % of the time, one above 0.64 ha 67 to 87 %. The token-grid models lose the most (TerraMind and
THOR 0.04 to 0.08), TESSERA the least (0.002 to 0.016), so TESSERA overtakes TerraMind per parcel at K = 100 and
20 %. The rule is robust: interior-pixel voting and leaving out cut parcels move it by less than 0.01. Reading the
predictions back also found a fault: two of the 33 fits had been predicted with a stale checkpoint (fixed in the
core). See [REPORT.md](REPORT.md).
