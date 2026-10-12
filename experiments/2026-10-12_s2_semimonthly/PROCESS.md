# How the 24 half-month composites are built

The 24-frame chips are made exactly like the 12-month chips of `EE_2021_rev`, except in how acquisitions are
assigned to frames. [`../2026-10-12_s2_revised_export/PROCESS.md`](../2026-10-12_s2_revised_export/PROCESS.md)
describes the shared steps: search, read, offset, screening, merge, median, gap filling and writing. This note
gives only what differs and why.

## The frame

An acquisition belongs to the half month of its sensing date (UTC): days 1 to 15, or day 16 to the month's end.
That gives two frames per month, named `Jan1`, `Jan2`, ..., `Dec2`. The frames are 15 or 16 days long, apart from
the second half of February (13 days). The middle day of each frame, the 8th or the 23rd, is recorded in the
manifest as `period_centres_doy` for encoders that take a date per frame.

A half month holds about two or three Sentinel-2 revisits over Estonia, against five or six in a month. So a
frame is filled more often, and on fewer dates its median is closer to a single acquisition. Short events, such as
the three-week potato peak that a monthly median dilutes (notebook 06), have a better chance of standing out.

## What is the same

- **The read.** The search, the read, the BOA offset, the `REVISED` screening and the merge of tile copies are the
  core's functions, imported unchanged.
- **The loop.** `s2_periods.composite_window` is the core's composite loop with `period_of(date)` in place of
  `date.month`.
- **Gap filling.** It is the core's `_fill_months`. It interpolates along the frame axis of any length, so a half
  month without a clear overpass is filled from its neighbouring half months, which are 15 days away rather than 30.
- **Writing.** The composites are written by the core export's `write_tif`: int16 reflectance x 10000,
  period-major (band index = period index x 12 + band index), with descriptions `P01_COASTAL_AEROSOL` to
  `P24_SWIR_2`.
- **Grouping.** Chips are read in 2 x 2 groups of adjacent chips, which fit inside the 4 x 4 blocks of the sample.

## Linked products

The labels, the parcel rasters, Sentinel-1, TESSERA and AlphaEarth are linked from `EE_2021`, and the split from
`EE_2021_sample`. Sentinel-1 stays monthly: the arms that would read 24 frames use Sentinel-2 only.

## The manifest

`manifest.json` holds:
- the parent's chip entries, with each chip's 24-period report in place of the monthly one;
- `n_periods` 24, the period names, the period rule and the centre days;
- the layout and the screening;
- normalisation statistics over 200 chips of the set, all periods pooled.

`n_months` is set to null so that no loader takes the set for a monthly one by mistake.
