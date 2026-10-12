# How the Sentinel-2 composites are downloaded and processed

This note follows one chip from the archive to the file a model reads, under the `REVISED` screening used for
`EE_2021_rev`. For each step it gives what is done and why. The code is `gfm4agri.chips.s2_monthly`
(`composite_window`, `_screen`, `_merge_overpasses`, `_fill_months`), driven by
`scripts/hub/build_country_chips.py` through `run.py` in this folder. The evidence for the screening is the
study in [`../2026-10-12_s2_screening/REPORT.md`](../2026-10-12_s2_screening/REPORT.md).

```
 chip cells of EE_2021 (2,240 m grid, EPSG:3035)
   │  grouped 2 x 2 into one read window, padded by 2 px
   ▼
 1 search      Planetary Computer, sentinel-2-l2a, 2021, scene cloud cover < 80 %
   ▼
 2 read        12 bands + SCL of every item, warped onto the 10 m chip grid
   ▼
 3 offset      baseline 04.00 items shifted onto reflectance = DN / 10000
   ▼
 4 screen      per acquisition and pixel: clear or not (SCL, 2 px buffer, snow test, haze test)
   ▼
 5 merge       tile copies and reprocessings of one overpass become one observation
   ▼
 6 median      per pixel, per band, per calendar month, over the clear overpasses
   ▼
 7 fill        months without a clear overpass interpolated along the year
   ▼
 8 write       <chip>_merged.tif (144 bands, int16), labels, report; manifest.json
```

## 0. Which chips

The chips are the 7,402 cells of `EE_2021`. These are the cells of a 2,240 m grid anchored on the origin of
EPSG:3035 that hold at least one parcel of the thesis class scheme in EuroCrops v11 Estonia 2021. A chip is
224 x 224 pixels of 10 m.

`run.py` restricts the export to exactly the parent's chip list, so the parent's spatial block split, and every
list derived from it (`EE_2021_mini`, `EE_2021_sample`), apply to the new set unchanged.

Adjacent chips are composited in **groups of 2 x 2**, one read per group. A Cloud-Optimised GeoTIFF is read in
blocks of 512 x 512 pixels, larger than a chip, so a chip-by-chip read fetches the same blocks for each of its
neighbours; a group fetches them once. Every operation after the read is per pixel, apart from the 2 px buffer of
step 4, so a group window is padded by 2 px and cropped afterwards. A chip cut from a group is then the chip
composited alone, up to the resampling noted under *Properties*.

## 1. Search

The STAC catalogue of the Microsoft Planetary Computer is searched for `sentinel-2-l2a` items intersecting the
window in WGS 84, acquired in 2021, with scene-level `eo:cloud_cover` below 80 %. The filter only spares the
download of scenes that are nearly all cloud; the decision about each pixel is taken in step 4.

The search returns every MGRS tile that touches the window (up to four in tile overlaps) and every processing
baseline the archive serves. In 2021 that includes reprocessings at baseline 04.00 beside the originals. Both kinds
of duplicate are kept here and resolved in step 5.

## 2. Read and warp

`stackstac` reads, for every item:
- the twelve L2A spectral bands: B01 to B12, B10 being absent from L2A;
- the scene classification layer (SCL).

Both are warped straight onto the window's 10 m grid in EPSG:3035:
- **Spectral bands:** bilinear resampling.
- **SCL:** nearest-neighbour resampling, because interpolating a class code would invent classes.

Two guards keep every acquisition attached to its own data.

- **Search order.** The stacks are built with `sortby_date=False`, so they keep the search order. A date-sorted
  stack once reversed the season of every chip.
- **Alignment by item id.** The spectral and SCL stacks are aligned on item ids, never on positions, because
  `stackstac` silently drops an item that is empty over the window.

An asset the archive cannot serve is dropped and recorded under `unreadable_items`. One such asset is tolerated,
or up to 5 % of a window's items; more fails the window, which is then retried, rather than gap-filling a
composite that only looks complete.

## 3. BOA offset

From processing baseline 04.00 the L2A digital numbers carry an offset of +1000. Items at that baseline are
shifted by −1000, so every date sits on the convention `reflectance = DN / 10000`.

## 4. Screening: is this pixel of this acquisition clear?

Each acquisition is judged pixel by pixel.

- **Not covered.** A pixel is *not covered* when the item has no data there: SCL 0, or a band with no value.
- **Flagged.** A covered pixel is *flagged* when any of the four tests below fails.
- **Clear.** It is *clear* otherwise.

| Test | Rule | Why |
|---|---|---|
| Scene classification | only class 4 (vegetation), 5 (not vegetated) or 6 (water) is clear | The other classes are cloud (8, 9), cirrus (10), cloud shadow (3), snow (11), saturated (1), and two that the export used to keep: 2, dark area, and 7, unclassified. In the screening study, observations removed by class 2 alone had near infrared 55 % below the pixel's clean median: cloud shadow. Those removed by class 7 alone were bluer (+0.022) and lower in NDVI (−0.085): haze and cloud edges |
| Buffer | not within 2 px (20 m) of a pixel of class 3, 8, 9, 10 or 11 on the same acquisition | Cloud, shadow and snow edges a pixel or two beyond the classified object pass as vegetation or soil; the buffer's removals were bluer (+0.033) with NDVI 0.115 lower |
| Snow test (Fmask) | not NDSI > 0.15 with B08 > 0.11 and B03 > 0.1, where NDSI = (B03 − B11) / (B03 + B11) | Snow the classification calls vegetation or soil; removals in winter were bluer by +0.097 |
| Haze test (haze-optimised transform, Fmask) | not B02 − 0.5 · B04 − 0.08 > 0 | Haze raises blue more than red; removals were bluer by +0.133 with NDVI 0.185 lower |

The thresholds of the two spectral tests are those of Zhu and Woodcock (2012). Applied to surface reflectance,
which is darker than the top-of-atmosphere reflectance Fmask was designed on, they are conservative for clear
land.

## 5. One observation per overpass

The catalogue lists one overpass once per MGRS tile, and again where it was reprocessed. In a tile overlap the
same date used to enter the median up to four times. Sen2Cor also classifies every tile on its own, so one copy
can call a pixel clear that another calls cloud.

The items of a month are therefore grouped by platform and sensing time, which are equal for all copies of one
overpass, and each group becomes one observation:
- **Rejected:** a pixel is rejected if any copy covering it flags it.
- **Value:** otherwise its value is the mean of the covering copies. These differ only by per-tile processing.

## 6. Monthly median

For every pixel, band and calendar month, the composite is the median over that month's clear overpasses. The
median is taken band by band, so in a month with several clear dates the bands of one pixel may come from
different dates. This is the export's rule, unchanged.

## 7. Gap filling

A pixel with no clear overpass in a month is filled along the year: by linear interpolation between its nearest
clear months, or with the nearest clear month at either end of the year. A pixel never clear in the whole year,
which is rare, is written as 0.

The share of filled pixels per month is written into the chip's report, because the encoder cannot tell a filled
month from an observed one and a reader must be able to. A stricter screening leaves more months to the filling
(see *Properties*).

## 8. Writing the chip

**`<chip>_merged.tif`.** 144 bands of int16, reflectance x 10000 clipped to 0 to 32767, time-major: band index =
month index x 12 + band index, which is the `(time channels)` layout TerraTorch's `expand_temporal_dimension`
expects. Each band carries a description, `M01_COASTAL_AEROSOL` to `M12_SWIR_2`, which loaders check. The file is
a tiled, deflate-compressed GeoTIFF on the chip's own transform.

**`<chip>.mask.tif` and `<chip>.parcels.tif`.** The class index and the parcel id under every pixel centre,
re-rasterised from the polygons. They do not depend on the imagery and `check.py` verifies that they equal the
parent's.

**`<chip>.report.json`.** What went into the chip:
- items and distinct overpasses per month;
- processing baselines, offset-corrected dates and unreadable items;
- the screening;
- mean clear overpasses per pixel and the filled share per month;
- never-observed pixels;
- the group window it came from.

**`manifest.json`.** Assembled from the reports when the run ends. It holds:
- the class list, the grid and the band layout;
- the normalisation statistics: per-band mean and standard deviation over 200 chips, all months pooled;
- `run.py`'s stamp: the screening, the parent chip set and the list of linked products.

The new manifest has a new hash, so any feature cache built from `EE_2021` is recomputed for `EE_2021_rev`
rather than reused.

## 9. Products linked from the parent

These are links into `EE_2021`, because the Sentinel-2 screening does not touch them:
- **Per chip:** Sentinel-1 RTC composites, TESSERA and AlphaEarth rasters, and their records.
- **Per chip set:** `chip_parcels.parquet`, the S1, TESSERA and AlphaEarth records, and the split.

The links are relative, so `EE_2021` and `EE_2021_rev` must be copied together, to the cluster for example.

## 10. Resuming and failures

A chip is done when its composite, its two label rasters and its report exist. A rerun skips it, so an
interrupted export continues where it stopped.

A window whose read fails is retried three times, with back-offs of one, two and three minutes, because the
archive fails in bursts. A window that still fails is recorded in `failures.json` and does not stop the run;
running the command again picks it up.

## Properties of the result

- **Fewer clear observations.** On the 30 chips of the screening study, the share of labelled pixels gap-filled
  rises:

  | Period | Old screening | `REVISED` |
  |---|---|---|
  | April to September | 4.0 % | 6.4 % (August 23 → 35 %) |
  | February | 65 % | 87 % |
  | March | 21 % | 55 % |
  | November | 19 % | 43 % |

  April to July is unaffected. The months that lose observations are those whose observations were mostly snow,
  cloud and shadow.
- **Cleaner composites.**
  - White pixels from October to March fall from 342 to 0.2 per 10,000 observations.
  - NDVI dips, a month more than 0.15 below both neighbours, fall by 15 % from April to September and by 58 %
    from October to March.
- **Resampling depends on the window.** GDAL's warp uses an approximate coordinate transform whose sampling
  depends on the output extent. A chip read in a 2 x 2 group therefore differs slightly from the same chip read
  alone: a median of zero, and up to a few hundred digital numbers at single pixels. No export reproduces another
  bit for bit unless it uses the same windows.
- **What remains.**
  - Haze that passes all four tests.
  - The seams of the per-pixel median, where a field takes its value from different dates on either side of a
    cloud edge.

  Notebook 06, section 5, shows both.
