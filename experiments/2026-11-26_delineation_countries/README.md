# Field delineation outside Estonia: all models on a distributed sample of 18 countries, 26 November 2026

**Question** (user, 26 November 2026, after [the Estonian comparison](../2026-11-26_open_delineation/REPORT.md)).
Check the delineation metrics on a sample of other countries, not Estonia, spread over them, and compare all the
models.

**What is compared.** The models of the Estonian comparison, unchanged, on a sample none of them was tuned on:

- **Our arms of the two-head workflow, trained on Estonia only** (K = 100 %): the PRUE baseline, TESSERA v2,
  TerraMind, THOR and Prithvi, with their Estonian checkpoints (`predict_arms.py`). AlphaEarth is left out: its
  embeddings were never fetched for these countries. For our arms this is a **cross-country transfer**.
- **The open models, not tuned on this sample**, each with the setting chosen on the Estonian validation chips. None is
  trained on these chips, but FTW was trained in 14 of the 18 countries and U-TAE + PaPs in France, so for them
  most of the sample is not a new country:
  - FTW PRUE B7 and FTW v2, also with each country's own FTW windows, their published setting (`predict_ftw_own.py`);
  - Delineate Anything v2 and v1;
  - U-TAE + PaPs.

**The sample** (`sample.py`). Six test blocks of 4 x 4 chips per country, taken by farthest-point sampling from the
transfer plan's country test sets (`docs/research/transferability/samples.csv`, spread over each country and matched
to its crop mix). 18 countries (AT, BE, BG, CZ, DE, DK, ES, FI, FR, IE, IT, LT, LV, NL, PT, SE, SI, SK), 108 blocks,
1,513 chips. Their composites, Sentinel-1 and masks are those of the transfer export (`<CC>_2021_transfer`, 2021,
the `revised` screening, 21 species and 9 groups). They are linked into `data/eurocrops_chips/EU18_2021_delin/`.

**Fairness.**
- **One scorer.** The Estonian experiment's scorer with the two-head workflow's functions, read-only (`score.py`);
  grassland, species 14 and group 26, is set aside in the second truth.
- **FTW leakage.** Fields of The World touches 29 of the 1,513 chips, in Austria, Latvia and Slovakia
  (`ftw_overlap.py`). It does not cover Bulgaria, Czechia, Ireland or Italy at all, so these four are new countries
  for it, as all 18 are for our arms.
- **No tuning here.** No setting is chosen on this sample.

**How to run** (hub; the environments of the Estonian experiment):

    poetry run python experiments/2026-11-26_delineation_countries/sample.py
    poetry run python experiments/2026-11-26_delineation_countries/worldcover.py
    poetry run python experiments/2026-11-26_delineation_countries/ftw_overlap.py
    bash experiments/2026-11-26_delineation_countries/run_open.sh     # FTW, Delineate Anything, U-TAE, fusion
    bash experiments/2026-11-26_delineation_countries/run_arms.sh     # FTW own windows, then our five arms
    poetry run python experiments/2026-11-26_delineation_countries/score.py
    poetry run python experiments/2026-11-26_delineation_countries/report.py
    poetry run python experiments/2026-11-26_delineation_countries/check_arms_ee.py --scratch <empty dir>   # arms' path reproduces Estonia

`vendor/` links to the Estonian experiment's vendored U-TAE + PaPs. Nothing of the core is changed and nothing is
trained.

**Results.** `results/2026-11-26_delineation_countries/EU18_2021_delin/`. The token caches of the three token-grid
arms are in `data/embeddings/<backbone>/EU18_2021_delin/`, and TESSERA's raster cache beside them.

**Report.** [REPORT.md](REPORT.md).

**Outcome** (27 November 2026; [REPORT.md](REPORT.md)). On the 1,513 chips, TESSERA v2 with PRUE's U-Net, trained
on Estonia only, draws borders best on all parcels (border F1 0.733 [0.686, 0.775]; baseline 0.691, best open
model FTW v2 one-window 0.678); with grassland set aside FTW PRUE B7 leads (0.805 against 0.786). Within Estonia's
agro-climate group our arms transfer (TESSERA 0.788, best open model 0.624); in the Mediterranean they fail
(baseline 0.16, GFM arms 0.40 to 0.48, FTW B7 0.64, trained there), leaving grassland, fallow, orchards, olives and
vineyards undrawn, the frozen encoders holding far better than the baseline. The best of our arms beats the best
open model in 14 of 18 countries. THOR collapses (0.365) although its inference path reproduces its Estonian
predictions (`check_arms_ee.py`).
