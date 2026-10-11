# Sentinel-2 cloud and snow screening on a sample of Estonia, 12 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for this study. It imports
`gfm4agri.chips.s2_monthly` read-only.

**Question.** Before the Sentinel-2 chips are re-exported, which screening should they use? The QA of the composites
(notebook 06, section 5) found that the export screening, `SCL_DROP` alone, lets snow and cloud rims through
scene classification classes 2 (dark area) and 7 (unclassified), and that it counts an overpass once per MGRS tile
and reprocessing. `REVISED` in `s2_monthly.py` closes those leaks, but it was checked on three chips and six
parcels only. The open question is whether it is too strict: whether dropping classes 2 and 7, the 20 m buffer
and the haze test also throw away real observations in the growing season.

**What it varies.** Seven screenings, applied offline to the same downloaded acquisitions:

| Variant | Flagged as not clear | One observation per overpass |
|---|---|---|
| `export` | `SCL_DROP` (0, 1, 3, 8, 9, 10, 11) | no |
| `export_merged` | `SCL_DROP` | yes |
| `revised` | `SCL_DROP`, classes 2 and 7, cloud, shadow and snow grown by 2 px, Fmask snow test, haze test | yes |
| `revised_no_haze` | as `revised`, without the haze test | yes |
| `revised_keep7` | as `revised`, but class 7 kept unless the haze test flags it | yes |
| `revised_27_near` | as `revised`, but classes 2 and 7 dropped only within 5 px of cloud, shadow or snow | yes |
| `revised_keep2` | as `revised`, but class 2 kept; added after the first results | yes |

**Measures.** Per variant and month, over the labelled pixels of the sample: the share gap-filled, the clear dates
per pixel, and the contamination left in the composite (white pixels, and NDVI dips of more than 0.15 below both
neighbouring months). Per screening rule, the observations it removes from `export`, compared with the same
pixel's clean observations of the same month (the median of those `revised` keeps): a rule that removes
contamination removes observations bluer and with lower NDVI than the pixel's clean ones; a rule that removes real
surface removes observations like them.

**Decision rule, fixed before the results.** The least restrictive variant that (1) raises the April to September
gap-filled share over labelled pixels by no more than 2 percentage points over `export`, (2) removes, with every
rule it adds, observations whose blue reflectance stands clearly above the pixel's clean observations, and (3)
leaves no more white pixels and NDVI dips in the growing season than the strictest variant, within noise.

**Sample.** 30 chips of `EE_2021`, drawn with seed 0 by `fetch.py`: one chip with at least 15 % labelled pixels from
each cell of a 6 x 4 grid over the chip columns and rows, plus six from the chips whose August or October was more
than 30 % gap-filled in the export. The list is written to `sample.csv`.

**How to run, on the hub.**

    poetry run python experiments/2026-10-12_s2_screening/fetch.py --workers 6     # downloads, resumable
    poetry run python experiments/2026-10-12_s2_screening/analyse.py                # variants and measures
    poetry run python experiments/2026-10-12_s2_screening/report.py                 # figures/ and tables.md

**Results.** `results/2026-10-12_s2_screening/`: `acq/<chip>.npz` (B02, B03, B04, B08, B11 and SCL of every
acquisition, raw digital numbers), and the tables `analyse.py` writes.

**Outcome.** Re-export with `REVISED` as it stands ([REPORT.md](REPORT.md)).

- **Every rule removes contamination.** Each rule it adds removes observations with a contamination signature.
  For class 2 this is cloud shadow, with near infrared 55 % below the pixel's clean median; for the others it is
  bluer, lower-NDVI haze and cloud.
- **The relaxations stay dirtier.** Each one keeps such observations and leaves 2 to 3 more NDVI dips per 1,000
  observations in the growing season.
- **The cost.** The April to September gap-filled share rises from 4.0 % to 6.4 %, 0.4 points over the limit set
  beforehand, so no variant passes that rule and the choice is a judgement. February, March and November become
  mostly interpolated.
- **What it fixes.** White pixels fall from 342 to 0.2 per 10,000 observations in October to March.
