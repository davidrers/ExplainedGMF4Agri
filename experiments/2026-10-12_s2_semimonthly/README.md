# Half-month Sentinel-2 composites, 24 frames, on a sample of Estonia, 12 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited for this build. The compositor
is a copy of the core's in this folder (`s2_periods.py`), changed only in how acquisitions are assigned to
frames.

**Question.** This folder builds the data for a later experiment: does a frozen encoder segment crops better from 24
half-month frames than from 12 monthly ones? Here only the 24-frame chip set is built, and the 12-frame
counterpart is named. The fits are a separate experiment that will need the datamodule and the encoders to accept
T = 24.

**What it varies.** The temporal grid, and nothing else.

| | 12 frames: `EE_2021_sample_rev` | 24 frames: `EE_2021_sample_24` |
|---|---|---|
| Chips | the 733 chips of `EE_2021_sample`, whole 4 x 4 blocks, of which 413 training, 88 validation and 207 test chips are in the split lists | the same |
| Frame | calendar month | half month: days 1 to 15, day 16 to the month's end |
| Composite | `_merged.tif`, 144 bands (12 x 12) | `_merged.tif`, 288 bands (24 x 12), period-major |
| Screening | `REVISED` | `REVISED` |
| Everything else | per-pixel median, one observation per overpass, linear gap filling along the frames, int16 reflectance x 10000 | the same |
| Labels, S1, TESSERA, AlphaEarth, split | linked from `EE_2021` and `EE_2021_sample` | the same |

The 12-frame set is cut from `EE_2021_rev`, the re-export of
[`../2026-10-12_s2_revised_export/`](../2026-10-12_s2_revised_export/README.md), with the blocks of
`EE_2021_sample`. That export processes these 733 chips first. Both sets therefore use the same screening, and a
difference between them is a difference of temporal grid. They are not compared with the old `EE_2021_sample`,
whose screening differs.

[`PROCESS.md`](PROCESS.md) explains how the 24 frames are built and what changes against 12.

**How to run, on the hub.**

    cd "/data/private/THESIS - ExplainedGMF4Agri"
    export PATH="$HOME/.local/bin:$PATH"
    mkdir -p data/eurocrops_chips/EE_2021_sample_24
    setsid nohup poetry run python experiments/2026-10-12_s2_semimonthly/build.py --workers 4 --group-size 2 > data/eurocrops_chips/EE_2021_sample_24/build.log 2>&1 < /dev/null &
    # once EE_2021_rev holds the sample's chips: the 12-frame counterpart, as links
    poetry run python experiments/2026-10-12_s2_semimonthly/make_sample_rev.py

**Results.**
- `data/eurocrops_chips/EE_2021_sample_24/`: `chips/` (288-band composite and report per chip, links for the
  rest), `manifest.json` (`n_periods` 24, period names and centre days, normalisation statistics over the set),
  `build.log`.
- `data/eurocrops_chips/EE_2021_sample_rev/`: the 12-frame counterpart.

**Report.** `REPORT.md` once both sets exist. It covers the gap-filled share per frame, clear dates per frame,
and how often a half month has no clear overpass where its month has one.

**Outcome.** Pending: the build is running.

**For the 12 against 24 experiment.** Two things will need attention when that experiment is set up:
- **Frame count.** The datamodule takes the frame count from `gfm4agri.chips.s2_monthly.N_MONTHS` (12).
- **Dates.** Prithvi's `temporal_coords` are the 15th of each month. For 24 frames they are the period centres
  in the manifest, the 8th and the 23rd.

TerraMind and THOR run once per frame, so 24 frames double their feature caches and their encoding time.
