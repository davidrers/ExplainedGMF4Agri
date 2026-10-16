# The Phase 1 pipeline, from download to segmentation

This document traces the implemented pipeline end to end and states the reason for each
design decision at the point where it is taken. It is the missing middle layer between two
documents that already exist:

| Document | Answers |
|---|---|
| [thesis_design.md](../thesis_design.md) | What the research is and why, the task formulation, the model set |
| **this document** | How the data becomes a trained segmenter, and why each step is built that way |
| [protocol.md](protocol.md) | How the evaluation is designed: blocking, partitions, budgets, metrics, transfer |

A plain-language walkthrough with diagrams is in [pipeline_overview.md](pipeline_overview.md).

Everything below describes code that runs today. Where the implementation departs from
[protocol.md](protocol.md), the departure is stated in [section 10](#10-known-gaps) rather
than hidden. The runs made with it so far are listed in [section 11](#11-runs).

**This document changes only by agreement with the author**, never as a side effect of an
experiment or an exploration. An agreed update goes where it belongs: a changed stage, data
product, script or configuration to its section and the stage table below; a fault found and
fixed to where the stage is described; a departure from the protocol to section 10; a finished
run or grid to section 11.

---

## 1. The shape of the chain

```
EuroCrops vectors ──┐
                    ├─→ chip grid ──→ label rasters ──┐
EuroCropsML index ──┘                                 │
                                                      ├─→ chip set ──→ supervision ──→ fit
Sentinel-2 L2A (Planetary Computer) ──→ monthly composites ──┤                          ↑
Sentinel-1 RTC (Planetary Computer) ──→ monthly composites ──┤              frozen encoders, run once
AlphaEarth v1 (annual COG tiles) ──────→ embedding rasters ──┤              per chip set and cached:
TESSERA v1.0 store, retired from K-shot → embedding rasters ─┘                ViT feature maps and
                                                                               TESSERA v2's raster
```

Every stage writes to disk and is resumable, because the network-bound stages, the Sentinel-2
composites and the TESSERA and AlphaEarth reads, dominate the wall clock and must never be
repeated by accident. The frozen encoders are not stages of the chip set: the fit runs them over
the chip set's composites once and caches their output (section 9), TESSERA v2 among them since
15 October 2026 (section 7).

| Stage | Code | Output |
|---|---|---|
| 1. Vector labels | `scripts/hub/fetch_eurocrops.py`, `scripts/hub/fetch_eurocropsml.py` | `data/eurocrops/parquet/<CC>_<year>.parquet` |
| 2. Chip grid and label rasters | `src/gfm4agri/data/chip_grid.py` | `<chip>.mask.tif`, `<chip>.parcels.tif` |
| 3. Sentinel-2 composites | `src/gfm4agri/chips/s2_monthly.py` | `<chip>_merged.tif` |
| 3b. Sentinel-1 composites | `src/gfm4agri/chips/s1_monthly.py`, `scripts/hub/build_s1_chips.py` | `<chip>_s1rtc.tif`, `s1_rtc.json` |
| 4. Chip export | `scripts/hub/build_country_chips.py` (`--screening revised` by default); a pilot cut from it by `scripts/hub/build_pilot_chipset.py` | `manifest.json` and the chip set; for Estonia `EE_2021_rev` and its pilot `EE_2021_rev_mini` |
| 5. TESSERA v1.0 store embeddings, retired from K-shot | `src/gfm4agri/embeddings/tessera.py`, `scripts/hub/build_tessera_chips.py` | `<chip>_tessera.tif`, `tessera_v1.json` |
| 5b. AlphaEarth embeddings | `src/gfm4agri/embeddings/alphaearth.py`, `scripts/hub/fetch_alphaearth_tiles.py`, `scripts/hub/build_alphaearth_chips.py` | tiles in `data/alphaearth/aef_v1_annual/<year>/`; `<chip>_alphaearth.tif`, `alphaearth_v1.json` |
| 6. Spatial split | `src/gfm4agri/data/chip_split.py`, `scripts/hub/build_chip_split.py` | `splits/<name>__<hash8>/`, `chip_parcels.parquet` |
| 7. Supervision | `src/gfm4agri/benchmark/segmentation_data.py` | the training mask at a budget |
| 8. Model assembly | `src/gfm4agri/benchmark/{backbones,necks,decoders,segmentation}.py`; the baseline's U-Net in `benchmark/scratch_unet.py` | a `SemanticSegmentationTask` |
| 9. The fit | `scripts/run_kshot.py`, the K-shot workflow of `src/gfm4agri/pipeline/`: `fit_end_to_end` (`benchmark/fit.py`) for the raster arms and the baseline, after `build_raster_cache` (`pipeline/raster_cache.py`, with `embeddings/tessera_v2.py`) for a computed raster; `generate_embeddings` then `fit_cached` (`benchmark/cached.py`) for the token-grid arms; every finished cell scored per parcel by `benchmark/parcel_metrics.py` | `results/<experiment>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/results.json`, with the pixel test metrics and `parcel_metrics`, and `predictions_test.npz`; `summary.csv` with both; the feature cache, or TESSERA v2's `chips/<chip>_tessera_v2_medium.tif` and `tessera_v2_medium.json`, in `data/embeddings/<backbone>/<chip set>/`, or job-scoped on the cluster |
| 10. Inspection | `src/gfm4agri/benchmark/evaluation.py`, `notebooks/pipeline/eurocrops_ee_results.ipynb` | reloaded fits, test-chip predictions in `results/seg_cached/ee_predictions/` |

---

## 2. Vector labels

Parcel polygons come from the EuroCrops v11 release, one GeoParquet per country and
declaration year, written with `write_covering_bbox=True` so that a chip-sized bounding-box
read is cheap. Nothing is filtered, reprojected or reclassified on the way in, because the
acquisition step must stay separable from the design decisions that follow it.

EuroCropsML is fetched separately and serves as a parcel index, a class source and a sanity
reference. It ships **no polygon geometry**, so it cannot be the training data for
segmentation; the geometry comes from EuroCrops and the two join on `parcel_id`.

The crop classes are not taken from the data. `configs/class_scheme_eurocropsml.yaml` is the
authoritative scheme, loaded through `src/gfm4agri/data/class_scheme.py`, which defines the
contract that file must satisfy. Estonia 2021 carries 20 classes.

**The scheme's codes and the vector layer's codes are not the same release.** The scheme takes
its codes from EuroCropsML, which uses HCAT2; the polygons come from EuroCrops v11, which uses
HCAT3 and renames some codes. `vector_layer_aliases` in the scheme lists each such rename, and
`apply_vector_aliases` in `chip_grid.py` maps the vector layer onto the scheme before any label is
rasterised. One rename affects Estonia: spring rapeseed is `3301060402` in EuroCropsML and
`3301060403 summer_rapeseed_rape` in v11. Joining the two by parcel identifier confirms it, since
all 1,236 Estonian EuroCropsML parcels coded `3301060402` carry `3301060403` in v11, and no v11
parcel carries `3301060402`. Until 29 September 2026 the rename was not applied, so those parcels
were rasterised as `ignore_index` and the class was empty in every chip. The full Estonian chip
set was relabelled with `build_country_chips.py --relabel`, which re-rasterises the masks only,
leaves the imagery untouched, and fails a chip if any pixel changes other than from ignore to the
aliased class. It changed 1.20 M pixels in 814 chips and nothing else: every other class total
and the band statistics came out identical. Four grid cells held an in-scheme parcel only
through the alias, so they had no chip; they were exported with Sentinel-2 and Sentinel-1 like
any other, bringing the set to 7,402 chips and spring rapeseed to 1.21 M labelled pixels. The
relabelling is recorded under `relabelled` in the manifest. The twelve-chip pilot set predates
the fix and was not relabelled.

Only one-to-one renames belong in the alias list. The other differences between the two
releases are reclassifications, for instance 2,855 parcels that EuroCropsML codes as grassland
are `poaceae_grasses` in v11, and v11 is taken as the label source for those as it stands.

---

## 3. The chip grid

A chip is a square cell of a regular grid in ETRS89-LAEA (EPSG:3035), anchored on the
projection origin, of `CHIP_PX * PIXEL_M` = 224 x 10 m = **2,240 m** on a side.

Three properties follow from making the grid fixed rather than data-dependent.

* A chip identifier, `<CC>_<col>_<row>`, names the same ground footprint in every country and
  in every rebuild, so a chip set can be extended without renaming anything.
* A chip nests inside a spatial block whenever the block edge is a multiple of 2,240 m, which
  is what lets the block cross-validation of `src/gfm4agri/data/blocks.py` be applied to chips
  without a second tessellation.
* Chip selection and chip export are separable, because a cell is addressable before anything
  has been downloaded for it.

`score_cells` summarises each cell that holds an in-scheme parcel: the labelled area share,
the number of distinct classes, the parcel count, and the share of labelled area taken by the
dominant class. A parcel is assigned to the cell holding its representative point, which is
approximate at cell edges and is therefore used only to rank candidates, never as a label.

`select_chips` is used only for the pilot. It takes cells above a labelled-share floor, ranks
them by class count and by the share of labelled area not taken by the dominant class with a
seeded jitter to break ties, and walks the ranking greedily subject to a pairwise minimum
separation. The separation is what makes the pilot's train and validation chips spatially
independent in the absence of a block split. The full-country export does not use it: it takes
every cell holding an in-scheme parcel, and the partition is assigned later.

---

## 4. Label rasterisation, and why there are two rasters

Each chip carries two label rasters, and the pair is the mechanism that makes the label budget
work.

* `<chip>.mask.tif`, int16, the class index of every pixel whose centre falls inside an
  in-scheme parcel, and `IGNORE_INDEX` (-1) everywhere else. This is the **dense** label, and
  it is what a validation or test chip is scored against.
* `<chip>.parcels.tif`, int32, the parcel identifier under every pixel, `0` outside any parcel.

A pixel is labelled when its **centre** falls inside a parcel, `all_touched=False`, so a pixel
straddling a parcel edge is left out rather than guessed. Where two declared polygons overlap
the later one in file order wins; the overlap is negligible in declaration layers and is not
worth a rule.

The reason for the second raster: the sparse training mask at any label budget is derived from
it at training time, by keeping the mask only where the parcel identifier is one of the drawn
polygons. **The chips are therefore never re-exported per budget.** A budget is a filter applied
when the mask is loaded, not a property of the data on disk, so one 56 GB export serves every
point on the learning curve and every model.

---

## 5. Sentinel-2 monthly composites

Imagery is read from the Microsoft Planetary Computer `sentinel-2-l2a` collection and warped
straight onto the chip grid, twelve monthly composites of twelve L2A bands (B01 to B12, B10
being absent from L2A), written time-major as 144 bands of int16. The layout is
`band index = month_index * 12 + band_index`, which is the `(time channels)` arrangement
TerraTorch's `expand_temporal_dimension` expects.

Per month the composite is the per-pixel median over the acquisitions whose scene
classification marks that pixel as clear.

**Acquisition order is asserted, not assumed.** The month of every acquisition is read from the
search results, and the per-month arrays are cut from a single stack of all acquisitions by
position. That is only correct if the stack's time axis is in the search order, and by default
it is not: `stackstac.stack` sorts acquisitions oldest first, while the Planetary Computer search
returns them newest first. Until 25 September 2026 the compositor relied on the stack keeping the
search order, so every month label landed on the wrong acquisitions and the season came out
approximately reversed: labelled January was composited from December, labelled March from August
and September, labelled August from March and April. The seasonal peak stayed near June, because
the reversal mirrors around midsummer, which is why the fault was invisible in a quick look and
only showed as cropland NDVI of 0.58 in an Estonian March and 0.33 in August. The stack is now
built with `sortby_date=False`, and `_assert_item_order` fails the chip unless the stack's time
axis matches the search order acquisition for acquisition. Every chip exported before the fix,
and every Sentinel-2 run trained on them, was discarded.

Four corrections apply on top of that, and each is recorded in the chip manifest so a reader can
see what happened rather than trust that it did.

**BOA offset.** Acquisitions at processing baseline 04.00 or later are shifted so every date
sits on the `reflectance = DN / 10000` convention. The Planetary Computer also serves 2021 scenes
reprocessed at baseline 04.00 beside the originals, 10,767 items over 5,068 chips of the first
Estonian export, so the correction does apply to this season.

**Cloud and snow screening.** This is the correction that matters most and the one the vendored
loaders do not perform. It is a named `Screening` of `s2_monthly.py`, chosen with
`build_country_chips.py --screening` and recorded in every chip report and in the manifest.

* `revised`, the main workflow's since 13 October 2026 and the screening of `EE_2021_rev`.
  - **Clear classes.** A pixel of an acquisition is clear only if the scene classification calls
    it vegetation, bare soil or water (classes 4, 5, 6).
  - **Buffer.** It must also lie at least 2 px from cloud, cloud shadow, cirrus and snow.
  - **Spectral tests.** It must pass the Fmask snow test and the haze-optimised transform
    (Zhu and Woodcock 2012).
  - **One observation per overpass.** The tile copies and reprocessings of one overpass are merged
    into one observation, flagged if any copy is flagged.
* `export_2021`, the screening of the first export, `EE_2021`. It excludes nodata, saturated,
  cloud shadow, cloud medium and high probability, thin cirrus and snow (`SCL_DROP`), and counts
  every catalogue item as an observation.

The change is a fault found and fixed, described below under *What the screening missed*.

**Gap filling.** A pixel with no clear acquisition in a month is filled by linear interpolation
along the month axis from its nearest clear months, and by the nearest clear month at the ends
of the season. The share of filled pixels per month is returned per chip, because a Baltic
December is mostly filled rather than observed and a reader must be able to see that rather
than discover it later.

**Unreadable acquisitions.** An isolated asset the archive cannot serve is dropped and its item
id is recorded under `unreadable_items`, rather than failing the chip. The tolerance is narrow on
purpose: more than one dropped acquisition, or more than 5 % of the window's acquisitions, fails
the chip instead, so that a burst of archive errors or an expired download token is retried
rather than gap-filled into a composite that looks complete. A looser version of this fallback
once dropped 98 of 149 acquisitions from a test window and silently interpolated May to December
from April; the threshold is what prevents that.

Resampling is bilinear for the spectral bands and nearest for the scene classification, because
interpolating a categorical layer would invent classes that were never observed.

**What the screening missed, and the fix.** `notebooks/eda/06_monthly_composites_qa.ipynb` opens the
stored composites of six parcels exactly as the datamodule reads them. Under `export_2021`, three
faults show there.
- **Leaks.** Snow and cloud that the scene classification marks clear enter the median. One
  grassland parcel's December composite is snow, at NDVI -0.04.
- **Residual haze.** Cloud and haze survive in months reported as fully observed, so the filled
  share is not a measure of composite quality.
- **Seams.** Because each pixel takes its median over its own clear acquisitions, a cloud edge on a
  single date becomes a straight seam inside a field.

The leaks showed as irregular white rings and streaks in the composites. They have four causes.
1. **Dark area rims.** Sen2Cor rings snow inside cloud with class 2, dark area. In one January,
   2,307 of the 2,317 observed white pixels of a chip were class 2.
2. **Unclassified fringes.** It fringes cloud edges with class 7, unclassified. Neither class 2 nor
   class 7 is in `SCL_DROP`.
3. **Edges.** Cloud, shadow and snow edges a pixel or two beyond the classified object pass as clear.
4. **Duplicate copies.** One overpass is catalogued once per MGRS tile and again where reprocessed,
   so it entered the median up to four times, and one copy could pass what another flagged.

`experiments/2026-10-12_s2_screening` compared seven screenings on every 2021 acquisition of 30
chips. Each rule of `revised` removes observations with a contamination signature against the
same pixel's clean ones:
- **Class 2:** near infrared 55 % lower, so cloud shadow.
- **Class 7:** bluer by +0.022, NDVI 0.085 lower.
- **The buffer:** bluer by +0.033, NDVI 0.115 lower.
- **The haze test:** bluer by +0.133.

Each relaxation tested kept contamination.

The composites of `EE_2021` were superseded and nothing was deleted. `EE_2021_rev` replaces its
Sentinel-2 composites and links its other products, and `EE_2021` stays as their parent. Every
result fitted before 13 October 2026 used `export_2021`.

**What `revised` costs and leaves.** On all 7,402 Estonian chips, the mean share of chip pixels
gap-filled rises.

| Month | `export_2021` | `revised` |
|---|---|---|
| February | 50 % | 89 % |
| March | 22 % | 64 % |
| August | 11 % | 19 % |
| October | 8 % | 19 % |
| November | 21 % | 50 % |

April to July stays at 2 % or less. White pixels in a 300-chip sample fall from 454,256 to 85 in
January and from 624,764 to 1,240 in December.

The per-pixel seams remain, and so does haze that passes all four tests. Separately, and by design,
the monthly median dilutes canopy peaks shorter than a month: a potato parcel above NDVI 0.8 for
three weeks has July and August composites of 0.53 and 0.51.

The per-pixel filled mask is not stored, but it can be recovered exactly from the stack: a filled
month lies on the line between its neighbours in all twelve bands at once, with a second difference
of at most 2 after rounding to int16, and a filled month at either end repeats its neighbour. On the
home chips of those six parcels the recovered shares equal the exported ones to the four decimals
the report keeps.

### Sentinel-1 RTC composites

Sentinel-1 is consumed by TerraMind through its `S1RTC` modality, by THOR as a third band group and
by TESSERA v2 through its Sentinel-1 branch (protocol decision D24). The format is TerraMind's, the
first consumer: two bands, VV then VH, as backscatter in decibels; THOR and TESSERA map it onto their
own inputs. `s1_monthly.py` reads the
Planetary Computer `sentinel-1-rtc` collection, radiometrically terrain-corrected gamma nought in
linear power, and writes `<chip>_s1rtc.tif`, 24 bands of float32 on the chip's exact transform, in
the same `(time channels)` layout as the Sentinel-2 chips: `band index = month_index * 2 + band_index`.

Per month the composite is the per-pixel median over every valid acquisition, taken in **linear
power** and converted to decibels afterwards, because linear power is the domain in which
backscatter aggregates physically and the median also suppresses speckle. Both orbit directions and
every relative orbit are composited together, which is defensible on a terrain-corrected product
over flat Estonia; the orbit mix of every month is recorded in the chip's report. Resampling is
bilinear in linear power, empty months are filled along the month axis as for Sentinel-2, and the
unreadable-acquisition tolerance of Sentinel-2 applies unchanged.

Acquisitions are identified by item id and months are read from the stack's own time axis, never by
position, because `stackstac` reorders acquisitions and silently drops any that are empty over the
window, and positional indexing is what reversed the Sentinel-2 season.

`scripts/hub/build_s1_chips.py` writes the rasters beside the Sentinel-2 chips, either for every cell
of a country or, with `--from-manifest`, for the chips of an existing set such as the pilot. Reports go
to `s1_reports/<chip>.json` and the provenance and normalisation statistics to `s1_rtc.json`, kept apart
from the Sentinel-2 manifest so that neither picks up the other's chips and the manifest hash stamped
in earlier results stays valid. All 7,402 Estonian chips have Sentinel-1.

In the model, the backbone registry declares `S1RTC` as an extra modality of `terramind_v1_small_s2s1`,
`terramind_v1_large_s2s1` and `thor_v1_large_s2s1`. The datamodule interleaves the Sentinel-1 raster with
the Sentinel-2 stack by month, so augmentation keeps the two aligned, and normalises each with its own
statistics: the encoder's Sentinel-1 pretraining statistics under `normalisation: backbone` (TerraMind's
`S1RTC`, THOR's VV and VH at 10 m, both in dB), those of `s1_rtc.json` otherwise. For TerraMind it splits
them back into one tensor per modality after normalisation, and TerraMind averages the tokens of the two
modalities at every position, so the token grid, the neck and the decoder are exactly those of the
Sentinel-2 run and the two compare at matched capacity. THOR takes the two as one stacked image (section
9). TESSERA v2 reads the raster directly when its cache is computed (section 7). TerraMind on Sentinel-2
and Sentinel-1 ran on all of Estonia as an experiment before it joined the main workflow
(`experiments/2026-10-13_terramind_s1_s2`): Sentinel-1 added 0.026, 0.054 and 0.025 test Macro-F1 at
K = 100, 20 and 5 %, in all nine paired fits.

---

## 6. Chip export

Two scripts write chip sets with the same on-disk layout.

`build_pilot_chips.py`, now archived with the 12-chip pilot in
`experiments/2026-09_pilot_12chips/`, selected a handful of well-labelled, widely separated cells
and split them into `training_chips/` and `validation_chips/`. It existed to wire up and debug
the TerraTorch connection, and its manifest says so: *pipeline test set for the TerraTorch
connection; not a Phase 1 split*. Its successor is the pilot chip set at the end of this
section.

`build_country_chips.py` exports **every** cell holding an in-scheme parcel, which for Estonia
2021 is 7,398 chips. Three differences follow from the scale.

* **No split is baked in.** Chips go to a single `chips/` directory, because the partition is
  the spatial block assignment of `src/gfm4agri/data/blocks.py`, which is an evaluation
  decision, and materialising it at export time would freeze it into the data.
* **Each chip writes its own report sidecar**, so an interrupted run loses at most the chip in
  flight. `manifest.json` is assembled from the sidecars, and `--assemble-only` rebuilds it
  from whatever is on disk.
* **Chips composite in a process pool**, because the work is dominated by reading COGs over the
  network. Each worker loads the parcel layer and its spatial index once.

A chip is skipped when its three rasters and its report are all present, so re-running the
command resumes. The same property means a chip set produced by a faulty compositor must be
removed before a corrected export is run into the same directory, or the faulty chips are
treated as done.

Measured cost is roughly 150 s of worker time and 7.6 MB per chip. The first Estonian export
sustained about 185 chips an hour at sixteen workers, which puts the full 7,398 chips near forty
hours and 56 GB on disk; the machine was at about a third of its CPU and 50 MB/s of network
during that run, so the limit is request latency rather than either resource.

`--group-size G` composites a G x G block of adjacent chips in one read, which fetches each COG
block once instead of once per chip. The re-export of 12 and 13 October 2026, `EE_2021_rev`, ran
with `--group-size 2` and 8 workers: 2,245 windows for 7,402 chips, in 36.5 h with no failed chip,
against 53.6 h for the chip-by-chip first export at 4 workers.

One property comes with it. GDAL's warp uses an approximate coordinate transform whose sampling
depends on the output extent, so a chip read in a group differs slightly from the same chip read
alone: a median of zero, and up to a few hundred digital numbers at single pixels. No export
reproduces another bit for bit unless it uses the same windows.

`--screening` names the Sentinel-2 screening of section 5, `revised` by default. A chip set holds
one screening only. An export into a set whose manifest names another is refused, and the
manifest's screening is read from the chip reports, so a manifest rebuilt with `--assemble-only`
states what its chips were made with.

`EE_2021_rev` was derived from `EE_2021` by `experiments/2026-10-12_s2_revised_export/run.py`. It
runs this script with `revised` and with the parent's chip list, re-rasterises the labels (checked
identical to the parent's), and links the Sentinel-1, TESSERA and AlphaEarth products,
`chip_parcels.parquet` and the split from `EE_2021`. The links are relative, so the two sets are
copied together. That folder's `PROCESS.md` follows one chip through every step.

The manifest is the provenance record: the class scheme, the label rule, the grid, the imagery
parameters, the per-chip labelled share, parcel count, per-class pixel counts and imagery
report, the band statistics, and the per-class pixel totals. `results.json` stamps its SHA-256,
so a result can always be tied to the exact chip set that produced it.

### The spatial split of a full-country chip set

`scripts/hub/build_chip_split.py`, with the logic in `src/gfm4agri/data/chip_split.py`, assigns
the partition that the export deliberately leaves out. The unit of partition is the chip. A
block is a square of `block_chips x block_chips` chips, which makes it a cell of the EPSG:3035
tessellation of `blocks.py` with an edge of `block_chips x 2,240 m`, and every chip nests in
exactly one block. Whole blocks go to `test`, `val` or `pool` through `assign_partitions`, at the
protocol's shares of 20, 10 and 70 % of parcels, from a permutation seeded on
`(protocol_seed, country, block edge)` alone.

**The buffer is applied to parcels, not to chips.** A pool parcel is withheld from training when
any of its labelled pixels lies within `buffer_m` of the rectangle of a test or validation block,
or when the same parcel identifier also occurs in a held-out chip. The pool chip stays in the
training set and the withheld parcels carry `ignore_index`, so the buffer costs only the labels
near a held-out block and not the whole chip. The distance is taken from the pixel centres of
the rasterised labels, so the rule is exactly the one the training masks see. A test chip's input
never contains a pixel of a training chip, because chips do not overlap; the buffer closes the
two channels that remain, a field cut by a block edge and the short-range autocorrelation of
crop labels across it.

The split is written to `<chip set>/splits/<name>__<hash8>/`, where the hash covers every knob and
the SHA-256 of the chip manifest, so relabelling the chips produces a new split rather than a
stale one:

| File | Content |
|---|---|
| `split.json` | configuration and hash, chips, blocks, parcels and labelled pixels per partition, the cost of the buffer, parcels per class and partition, the classes below the protocol's 200 test parcels |
| `training_data.txt`, `validation_data.txt`, `test_data.txt` | chip lists; a pool chip left with no trainable parcel is not listed |
| `chips.csv` | every chip with its block, partition, labelled and trainable parcel counts |
| `buffer_parcels.npy` | the withheld parcel identifiers |
| `map.png` | the partition on the chip grid |

It also writes `chip_parcels.parquet` once per chip set, one row per `(chip, parcel, class)` with
the parcel's labelled pixels in that chip. `draw_support` reads that table under a split, so a
draw costs under a second instead of a pass over every parcel raster, and removes the withheld
parcels before drawing. At 100 % the draw is therefore every trainable parcel, and
`fit_end_to_end` treats `dense` under a split as that draw, so a buffered parcel is never
trained on.

Every consumer takes the split directory as `data.split` in the run configuration:
`EuroCropsSegDataModule`, `draw_support`, the stage 1 encoder of `cached.py` and its stage 2
datamodule. The result file records the split's name, hash, protocol and chip counts under
`split_protocol`.

### The pilot chip set

A pilot is a smaller chip set, not a different procedure. `scripts/hub/build_pilot_chipset.py`,
with the logic in `src/gfm4agri/data/chip_subset.py`, draws whole blocks of a country chip set
with a fixed seed, by default three from the pool, one validation and one test block, among
blocks that hold a trainable parcel (pool) or a labelled one (validation and test). It writes a
chip set directory of its own beside the parent: `chips/` holds relative links into the
parent's chips, the manifest, the per-source records and `chip_parcels.parquet` are links to the
parent's, and the split keeps its name and holds the parent's lists restricted to the chosen
chips, with `split.json` recording the parent and the blocks under `subset`. The normalisation
statistics and the manifest hash are therefore the parent's, the pilot costs no disk space, and
the K-shot workflow of section 9 cannot tell it from a country. `EE_2021_mini`, built with the
defaults on 9 October 2026, holds blocks `582_450`, `586_447` and `591_463` (pool), `563_444`
(validation) and `566_444` (test): 32 training, 16 validation and 9 test chips. `EE_2021_rev_mini`,
the main workflow's pilot since 13 October 2026, is cut from `EE_2021_rev` with the same defaults
and holds the same blocks. A pilot is copied to the cluster after its parent, whose chips its links
point to.

---

## 7. TESSERA embeddings

TESSERA encodes each 10 m pixel's Sentinel-2 and Sentinel-1 series into one 128-dimensional
vector, from that pixel alone: no part of the encoder sees a neighbouring pixel. Since 15 October
2026 the main workflow runs **TESSERA v2 Medium from its open weights, frozen, on the chip set's
own monthly composites** (protocol decision D25). The v1.0 store rasters it used before are
described after it; they stay on disk and are no longer read by `kshot`.

### TESSERA v2, computed by the runner

**The encoder.** `src/gfm4agri/embeddings/tessera_v2_model.py` is TESSERA's v2 pixel student and
its input bucketing, copied verbatim from `ucam-eo/tessera` at commit `e368539` under its MIT
licence. The Medium student, 21.0 M parameters, is loaded from
`data/weights/tessera/v2/student_medium.pt`, fetched from Hugging Face
(`geotessera/TESSERA-V-2.0-2B-M`) when absent and checked against the SHA-256 the registry
records; it runs in evaluation mode under bf16 with no gradients.

**The inputs are the composites every other arm reads** (`src/gfm4agri/embeddings/tessera_v2.py`):

* Sentinel-2: TESSERA's ten bands in its own order, B04 B02 B03 B08 B8A B05 B06 B07 B11 B12,
  taken from `<chip>_merged.tif` by band name, as reflectance x 10,000, TESSERA's convention.
* Sentinel-1: our VV and VH in dB become TESSERA's int16 encoding, (2 dB + 50) x 200, clipped to
  0 to 32,767, with 0 for a month never observed, TESSERA's missing value. Our composites merge the
  two orbit directions, so the twelve frames enter as its ascending stream.
* Dates: the fifteenth of each month, as the day of year.

Both sensors are standardised with v2's own statistics, and each pixel's valid frames are bucketed
and padded as v2's `encode_pixels` does. A pixel whose Sentinel-2 is zero in every month, the
never-observed pixels of section 10, is encoded from Sentinel-1 alone; a pixel with neither comes
out NaN, which the datamodule replaces with 0 as it did for the store's gaps.
`experiments/2026-10-14_tessera_composites` established that frozen v2 on these monthly composites
separates crops about as well as from its native per-acquisition series, and scored the frozen
arm within 0.017 test Macro-F1 of the v1.0 store arm on all of Estonia.

**The cache** (`src/gfm4agri/pipeline/raster_cache.py`). The encoder's output for a chip never
changes, so the runner computes it once per chip set, as it does a ViT's feature cache, into
`<cache root>/tessera_v2_medium/<chip set>/`: `chips/<chip>_tessera_v2_medium.tif`, float32 on the
chip's own grid; `tessera_v2_medium.json`, the sidecar the datamodule reads, with per-dimension
statistics of the training chips; and `cache.json`, written last, recording the backbone, the
student, the checkpoint's SHA-256, the precision, the chip manifest's SHA-256 and the split hash.
The lookup decides as for the token-grid caches (section 9): a complete cache whose record matches
is reused, a missing one is computed into the scratch root and resumes the chips already there,
and a cache made under another record is never written into. It holds one copy per chip, not
eight D4 variants, because a per-pixel encoder commutes with rotation and flipping. Chips are read
on a thread eight ahead of the encoder. On the cluster's RTX PRO 6000 a chip takes 0.63 s, so the
7,142 chips of the Estonian split take about 75 minutes and about 175 GB.

**The fit** is the `raster` route of section 9, unchanged except that the datamodule reads the
embeddings from the cache and the masks and parcel rasters from the chip set, paired by chip id.

### The v1.0 store, retired from the K-shot workflow

TESSERA v1 encodes each 10 m pixel's full-year Sentinel-1 and Sentinel-2 series into one
128-dimensional vector. The embeddings are precomputed and served from Source Cooperative as a
Zarr v3 store, one group per UTM zone, quantised to int8 with one float32 scale per pixel.

**Its rasters are offset from the chip grid**, by about 0.7 pixel east and 0.4 pixel south, 5 to
8 m, measured on `EE_2021_rev_mini` on 15 October 2026: ridge maps from the v1.0 raster, shifted
by up to two pixels, onto three grid-aligned references (v2 Medium and v1.1 on our composites, and
the v1.1 store read afresh) all fit best at one pixel east. Every TESSERA v1 result, the
full-Estonia grid of section 11 included, was trained with that offset. Its cause is not
established; section 10 records it.

The store is read directly with `zarr` over HTTPS rather than through the `geotessera` client,
for two reasons: the client that knows the new location requires Python 3.12 and this
environment is 3.11, and the store's 32 x 32 inner chunks make a chip window cost a few hundred
kilobytes rather than a whole 0.1 degree tile.

Each window is read in the zone's native UTM grid with an eight-pixel margin, dequantised, and
warped onto the chip's EPSG:3035 grid by **nearest neighbour**, so every output pixel is a
genuine TESSERA vector rather than an interpolation between two of them. Interpolating an
embedding would produce a vector the encoder never emitted.

**A chip near a zone edge is read from both zones.** The store files each tile under a single
zone, so a chip straddling a zone edge, which in Estonia is 24 E between zones 34 and 35, can
extend onto tiles held by the neighbouring zone. Reading the chip's own zone alone left that
strip empty: 8.4 % of chip `EE_02283_01827`, for instance. Pixels still empty after the chip's own
zone are now filled from the neighbouring zone whenever the chip lies within half a degree of
the edge, and the fill is recorded per chip under `filled_from`. The own-zone pixels are
unchanged by the fill, which was checked on that chip.

**Coverage is not complete.** Western Saaremaa has no TESSERA embedding in the store for 2021
or for any other year, over farmland as well as sea, so chips there come back largely or wholly
NaN. The share per chip is recorded as `nodata_share` in `tessera_v1.json`. Over the Estonian
split, 415 of 7,402 chips have some pixels without an embedding and 64 have none over any of
their labelled pixels; the labelled pixels without an embedding are 0.48 % of the pool, 0.33 % of
validation and 0.66 % of test, concentrated in grassland, 2.0 % of whose test pixels are
affected. How the TESSERA arm treats those pixels, and whether the other models are scored on
the same reduced set so that every model sees the same test pixels, is still to be decided.

**The store holds corrupt scales in one Estonian patch.** In the 2021 layer of zone 35, around
26.1 E, 58.9 N, about 1.7 % of the per-pixel scales are garbage, up to 2.8e32 where every genuine
scale lies below 0.14, and the store serves them unchanged on every read; the other eight years
at the same place are clean. Dequantised, they gave embeddings up to 1e34 in fourteen chips and
put 1.85e31 into the per-dimension statistics. The reader now treats a scale above
`MAX_SCALE = 0.2` as missing, like a pixel outside the coverage, and records the count per chip as
`implausible_scale_pixels`. The bound sits well above the genuine range, whose embeddings reach
at most 17.7 in absolute value across the other 7,388 chips, so it removes nothing genuine.

**The store fails in bursts.** Source Cooperative answered with HTTP 520 for long enough during
the Estonian export that one unretried read stopped the run. Each chip is now retried three
times with a backoff of 60 s, 120 s and 180 s, a chip still failing is listed in
`tessera_failures.json`, and no sidecar is written over an incomplete set, so a re-run completes
it. The full Estonian set of 7,402 chips finished with no failure.

The embeddings are written as `<chip>_tessera.tif` beside the chip's image, mask and parcel
raster, and share the masks, parcel rasters and split files with every other representation. A
label budget therefore draws the same parcels whatever the model consumes. Provenance and the
per-dimension normalisation statistics go to `tessera_v1.json`; `manifest.json` is left
untouched, because results already trained on the chips are stamped with its hash.

### AlphaEarth embeddings

AlphaEarth Foundations V1 summarises each 10 m pixel's calendar year of multi-source observation
into one 64-dimensional unit vector. Google publishes the annual layers, the Earth Engine
collection `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, as Cloud Optimized GeoTIFFs of 8,192 x 8,192
pixels and 64 int8 bands per UTM tile. `src/gfm4agri/embeddings/alphaearth.py` reads them, and the
export shares `src/gfm4agri/embeddings/export.py` with TESSERA, so the two embeddings land on the
chip grid in the same way.

**Tiles are downloaded whole rather than read by window.** The files are band interleaved in
1,024-pixel blocks, so one chip window costs a block per band, about 60 MB and one to three
minutes per chip over HTTP. `scripts/hub/fetch_alphaearth_tiles.py` instead fetches every tile a
chip set touches from the Source Cooperative mirror, which serves Google's requester-pays bucket
file for file at no charge, and keeps a file only if its MD5 matches the one Google publishes.
The Estonian chips need 22 tiles in zones 34 and 35, 42 GB in `data/alphaearth/aef_v1_annual/2021/`;
after that a chip takes a few seconds.

**Two properties of the files set how a chip is read.** The tiles are stored bottom-up, with the
origin at the south-west corner, so a window is read in raw rows and flipped to north-up before
the warp. The values are quantised, and `((q / 127.5) ** 2) * sign(q)` recovers the embedding as
the dataset README specifies; the de-quantised vectors keep norms within about 1 % of one and are
not renormalised. As for TESSERA, the window is warped from the tile's UTM grid onto the chip's
EPSG:3035 grid by nearest neighbour, so every output pixel is a genuine AlphaEarth vector, and a
chip near a zone edge is filled from the neighbouring zone.

`scripts/hub/build_alphaearth_chips.py` wrote `<chip>_alphaearth.tif` for all 7,402 Estonian chips
in 61 minutes on 7 October 2026, with the per-dimension statistics of the 4,898 training chips in
`alphaearth_v1.json`. **No labelled pixel lacks an embedding**: the masked value `-128` occurs in
none of the chips, so the zero filling that affects 0.66 % of TESSERA's test pixels does not
arise for AlphaEarth.

---

## 8. Supervision and the label budget

Supervision is **sparse in training and dense in evaluation**. The budget restricts what the
model trains on, never what it is scored against, so validation and test masks always stay
dense.

`draw_support` implements the budget. It is a **percentage of the independent training parcels
of each class**: a class with `n` parcels contributes `ceil(K / 100 * n)` of them, so a class
present at all keeps at least one parcel. A parcel's class is the majority class of its
labelled pixels across the chips, and a parcel cut by a chip edge contributes only the pixels
inside the chip.

The draw for a class is seeded on `(country, hcat_code, K, draw_seed)` and on nothing else. It
therefore does not depend on the model, on the other classes, or on iteration order, which is
what makes the comparison between foundation models a comparison of encoders rather than of the
labels each happened to receive. At `K = 100` the draw is every parcel, which reproduces the
dense mask exactly and is the consistency check on the machinery.

`SparseParcelSegmentationDataset` applies the budget at **load time**, before any transform:
every pixel whose parcel id is not in the drawn set becomes NaN, which `no_label_replace` then
maps to the ignore index. Doing it before the transform is what keeps the restriction aligned
with the image under any flip or rotation.

Augmentation is the eight D4 symmetries, applied identically to image and mask, and implemented
in NumPy rather than through `albumentations.D4`, because the latter goes through `cv2.flip`,
which rejects arrays with more channels than OpenCV supports, and a flattened twelve-month,
twelve-band stack has 144.

---

## 9. Model assembly and the fit

A fit is a **frozen encoder with a trainable decoder**. Full encoder fine-tuning is outside the
main comparison: AlphaEarth ships only embeddings, so a frozen encoder is the one setting every
model shares, and allowing fine-tuning for the open-weight models alone would make the comparison
incoherent. What fine-tuning adds is studied in experiments (`experiments/2026-10-14_tessera_composites`,
`experiments/2026-10-15_gfm_finetune`).

`backbones.py` is the registry. Each entry declares the model arguments, the pretraining
statistics, the band subset the encoder consumes, the resolution group, and whether the encoder
takes coordinates. The registry is what keeps per-model differences declarative: the datamodule
reads the band subset and the statistics from it rather than having them copied into a config
by hand.

**Inputs differ by model, and the differences are declared rather than incidental.** Each model
consumes every sensor of the chip set it was pretrained on (protocol decision D24): TerraMind, THOR
and TESSERA v2 the Sentinel-2 and Sentinel-1 stacks, Prithvi the Sentinel-2 stack alone. Prithvi
consumes only its six pretrained HLS bands, so on Sentinel-2 `NIR_NARROW` is B8A. TerraMind is
single-date, so a temporal wrapper runs it per month and concatenates the monthly features; its
Sentinel-1 is the `S1RTC` modality of section 5.
Prithvi is natively multi-temporal, so the twelve months enter one ViT as twelve token planes
with a 3D positional embedding. THOR v1 large, like TerraMind, is single-date and runs per month
through the same temporal wrapper. It tokenises its two Sentinel-2 groups apart, the four 10 m
bands and the six 20 m bands (B01 and B09 at 60 m are left out, since 60 m pixels do not tile the
2,240 m chip and THOR's authors drop the two atmospheric bands too), at patch sizes chosen at
inference. The arm uses **160 m tokens**, 16 px on the 10 m bands and 8 px on the 20 m bands,
which gives the 14 x 14 grid TerraMind has on the same chip; the user fixed this on 9 October
2026 for parity with TerraMind. The 80 m variant, 8 and 4 px patches on a 28 x 28 grid, was studied on the
12-chip pilot and run again under the K-shot protocol as an experiment (`experiments/2026-10-08_thor_80m/`),
`thor_v1_large_80m` in its own resolution group `token_grid_80m`. It is not an arm of the main workflow: its
full-Estonia cache, 154 MB per chip encoding, comes to about 6.4 TB. THOR's pretraining statistics, given in
reflectance, are scaled to the chips' reflectance x 10,000.

**THOR on Sentinel-2 and Sentinel-1**, `thor_v1_large_s2s1`, is the main workflow's THOR since 15
October 2026. VV and VH, read from `<chip>_s1rtc.tif`, enter as one stacked image with the
Sentinel-2 bands and form a third band group, THOR's 10 m IW group, at 16 px and so on the same
14 x 14 grid, normalised with THOR's Sentinel-1 statistics at 10 m in dB; a month never observed
enters at the band mean. The three groups are **averaged** at every token position
(`backbone_merge_method: mean`), as TerraMind averages its modalities, so each month contributes
1,024 channels per selected layer and the bottleneck and decoder are exactly TerraMind's: 53,281,492
trainable parameters for both. THOR's authors concatenate the groups on the channel axis for dense
tasks, as the Sentinel-2 arm `thor_v1_large` does, and the two have not been compared (section 10). THOR is not part of TerraTorch: its
backbones register through `thor_terratorch_ext`, which `scripts/env/install_thor.sh` installs
outside the Poetry lock at fixed commits on both machines. Encoders with date and location encodings additionally receive
`temporal_coords`, the year and zero-based day of year of the fifteenth of each month, and
`location_coords`, the chip centre in latitude and longitude.

**Normalisation** is either the statistics the encoder was pretrained under, or training-chip
statistics. A frozen encoder only understands inputs on the scale it was pretrained on, so
`backbone` is the default for the open-weight models. An embedding raster has no pretraining
statistics of its own, so both modes fall back to training-chip statistics for it.

**Decoder capacity is matched within a resolution group**, which is the condition under which a
score difference is attributable to the encoder.

* `token_grid`: `ChannelBottleneck` projecting the `T x embed_dim` channels to 768, then
  `LearnedInterpolateToPyramidal` and a 12.9 M parameter `UNetDecoder`. The bottleneck is not
  cosmetic. TerraMind concatenates twelve monthly feature maps, so without it the large variant
  carries 815 M trainable parameters against Prithvi's 63 M, which is neither comparable nor
  trainable on a 16 GB card. THOR on Sentinel-2 alone concatenates its two groups, which doubles
  that width to 24,576 channels per selected layer, so its bottleneck carries more weights than
  TerraMind's; THOR on Sentinel-2 and Sentinel-1 averages its three groups and matches TerraMind
  exactly. The projection also mixes the months of each token position, so it
  is where the decoder first learns which dates matter.
* `pixel_raster`: `PixelMLPDecoder`, a per-pixel MLP written as 1 x 1 convolutions, two hidden
  layers of 512 and 256. Every output pixel depends only on that pixel's embedding, so the
  score measures the embedding itself. This follows the head the TESSERA authors use for crop
  classification. TESSERA v2 and AlphaEarth use the same head with the same hyperparameters, so
  the two embeddings differ only in the embedding. A spatial U-Net in its place was tested on
  frozen TESSERA v2 and kept out (`experiments/2026-10-15_tessera_decoder`, section 11): the 10 m
  group therefore sees no spatial context beyond its encoder's, which for TESSERA is none.

**The baseline**, `unet_r18_scratch_s2s1`, is the model RQ1 measures the GFMs against: a 2D U-Net
with a ResNet-18 encoder (`segmentation_models_pytorch`), randomly initialised, 14.8 M
parameters, all trained. It reads the Sentinel-2 and Sentinel-1 stacks, Sentinel-1 interleaved
behind Sentinel-2 in every month as for THOR, and flattens the months onto the channels, 12 x 14
= 168, so the first convolution holds a weight for every band in every month and everything after
it is spatial (early fusion). It sees every sensor a GFM arm sees, so a gap between a GFM and the
baseline is not the second sensor. Having no pretraining statistics, it is normalised with the
chip set's: the manifest's for Sentinel-2 and `s1_rtc.json`'s for Sentinel-1, both computed on 200
chips sampled across the country rather than on the training chips alone. It sits in a `scratch`
group of its own: its output is at 10 m, as the 10 m group's, and it has spatial context
throughout, as the token-grid group has. It was tested on Sentinel-2 alone in
`experiments/2026-10-14_unet_baseline` (section 11), where it replaced the two per-pixel
raw-feature baselines the protocol first named, and entered the main workflow on 15 October 2026
on Sentinel-2 and Sentinel-1.

The task is a TerraTorch `SemanticSegmentationTask` built through `EncoderDecoderFactory`, with
`freeze_backbone=True`, AdamW, and cross-entropy that ignores the unlabelled pixels. The
baseline's task builds its U-Net through `ScratchUNetFactory` instead, with nothing frozen, and its
`results.json` records `frozen_encoder: false` and the model it built.

`fit_end_to_end` (`src/gfm4agri/benchmark/fit.py`, formerly `scripts/seg/train.py`) resolves the
configuration, draws the support set, builds the datamodule and the task, fits, tests from the
best validation-loss checkpoint, and writes `results.json` carrying
the configuration, the label budget and its realised per-class draw, the country, the split
protocol, the parameter counts by component, the fit time, the peak GPU memory and every
metric. Gradient accumulation keeps the effective batch identical across backbones when a
larger encoder forces a smaller micro-batch onto the card.

### The two-stage fit

Because the encoder is frozen, its output for a given chip never changes, so an end-to-end fit
re-encodes every chip at every epoch of every fit for nothing. `src/gfm4agri/benchmark/cached.py`
implements TerraTorch's second workflow instead. **Stage 1**, `generate_embeddings`, runs the
encoder once per chip through `EmbeddingGenerationTask` and stores the features at the cache
point, the output of `SelectIndices` and `ReshapeTokensToImage` just before the first trainable
neck, as float16 NumPy arrays, one per chip and encoder layer. **Stage 2**, `fit_cached`, builds the
end-to-end model, keeps exactly its trainable modules, and trains them from the stored
features, so architecture and parameter count are those of the end-to-end fit by construction.

A ViT is not equivariant to rotation, so rotating a cached feature map is not the same as
encoding a rotated chip; on the pilot the two correlate at only 0.81 to 0.92. Training chips are
therefore encoded in all eight D4 variants and stage 2 draws one per sample, as the end-to-end
fit draws a random symmetry; validation and test chips are encoded once, unrotated. Stage 1
passes the date and location coordinates to encoders that take them, which the stock task does
not, writes every file atomically, and skips a chip whose last layer is already on disk, so an
interrupted run resumes. On the pilot the two workflows gave Macro-F1 within the seed spread of
each other at every budget.

The cache is large and the storage it sits on sets its cost. On the full Estonian split,
4,898 training chips in eight variants and 2,244 held-out chips once make 41,428 encodings. At
float16 that is 19.3 MB per encoding for TerraMind v1 large, whose twelve monthly maps of 1,024
channels sit on a 14 x 14 grid, and 31.5 MB for Prithvi-EO-2.0 600M TL, whose 14-pixel patches
give a 16 x 16 grid of 12 x 1,280 channels: 798 GB and about 1.3 TB. Stage 1 for TerraMind large
took 11.5 h, bound by writing to the network file system at 25 to 36 MB/s rather than by the
encoder, with the GPU mostly idle. Reading the cache back reaches about 36 MB/s from a single
reader and about 140 MB/s from sixteen, so one stage 2 epoch over the TerraMind large training
set, 94 GB, takes on the order of ten minutes of reading. Neither cache fits the 335 GB of local
disk.

### The K-shot workflow

`scripts/run_kshot.py`, with the logic in `src/gfm4agri/pipeline/`, is the single entry point of
the fit. It takes an experiment and the path of a chip set, and runs every arm, budget, draw and
seed of the experiment on that chip set, the same way on the JupyterHub and on the cluster:

```
poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021 [--dry-run]
```

**Three configuration files replace the per-run YAML.** `configs/arms/<arm>.yaml` describes what
is trained and nothing about a country or a budget: the backbone, the route, the normalisation,
the decoder and head settings, the optimiser, the loss, the precision and the training batch.
`configs/experiments/<name>.yaml` lists the arms, the budgets, the draws, the seeds, the epochs
and the checkpoint criterion. `configs/machines/{hub,cluster}.yaml` holds what only changes
throughput: the encoding batch, the data loader workers and the cache root. `config.py` composes
the three and the chip set into the configuration dictionary the fit functions already took, so
`fit_cached`, `generate_embeddings`, `fit_end_to_end` and `FitPredictor` are unchanged; a test
checks that every cell of the full-Estonia grid of 2 October composes to the configuration that
grid ran with. The experiment `kshot` holds six arms at 100, 20 and 5 %, draw 0, seed 0 and 45
epochs, every arm alike (protocol D27); until 15 October 2026 it was 15 epochs, and those results
were moved to `results/kshot_15ep/`. Since 15 October 2026 they are `terramind_v1_large_s2s1` and `thor_v1_large_s2s1` (160 m
tokens) on Sentinel-2 and Sentinel-1, `prithvi_eo_v2_600_tl` on Sentinel-2, `tessera_v2_medium`
and `alphaearth_v1`, and the baseline `unet_r18_scratch_s2s1`. Until then they were TerraMind v1 large and THOR v1 large on Sentinel-2,
Prithvi-EO-2.0 600M TL, TESSERA v1 from the store and AlphaEarth v1; those arm files stay in
`configs/arms/` for the earlier results and for experiments.

**Experiments run on the same runner without touching it.** An experiment lives in its own folder,
`experiments/<date>_<name>/`, with an `experiment.yaml` the runner takes by path, arm files in `arms/` that are
found before `configs/arms/`, an arm's own `encode_batch_size` when the machine profiles do not know its backbone,
and, if code must change, its own `run.py` with copies of the changed modules swapped in; `submit.sh` takes the
spec's path and uses that `run.py` when present. Each `results.json` records the spec, the entry script and the
arm files. THOR on 80 m tokens runs this way, with the name `kshot` so its cells land beside the main
workflow's.

**Three routes.** The token-grid arms are always fitted from cached features with `fit_cached`;
training them with the encoder inside the loop is no longer part of the workflow. The raster
arms train end to end on their embedding rasters with `fit_end_to_end`, and so does the baseline
on the composites (`route: end_to_end`), with nothing frozen and so nothing to cache. A raster is either a
product of the chip set, as AlphaEarth's is, or computed by a frozen encoder the registry entry
names under `computed`, as TESSERA v2's is: the runner then resolves its cache (section 7) before
the arm's first fit, and passes the cache as the image root while the masks come from the chip set.

**A cache is looked up before it is computed.** For a token-grid arm the runner looks for
`<cache root>/<backbone>/<chip set>/`. It reuses that cache as it is only when its `cache.json`
matches the run on everything that changes the stored features (backbone, manifest SHA-256,
split hash, normalisation, precision and the eight training variants) and every chip of the
split is on disk; the existing TerraMind and Prithvi caches of `EE_2021` pass this check.
Otherwise the cache is computed into the scratch root, which resumes any chips already there. A
cache that exists but does not match is never written into: when the scratch root is the cache
root, as on the hub, the run stops and names the fields that differ. Before this check the only
test was the backbone's name.

**What a run writes.** Before any compute the runner checks that the chip set holds the split and
every raster the arms read, every modality included: an arm on Sentinel-1 requires `<chip>_s1rtc.tif`,
and TESSERA v2 requires the two stacks its encoder reads. A cell is finished when its directory holds `results.json` and
`predictions_test.npz`, the test-chip predictions in the format of the results notebook, and a
finished cell is skipped, so a run resumes. Because a skipped cell is not refitted, the runner
first compares every finished cell's recorded epochs with the experiment's and stops, naming the
cells, when they differ; such cells are moved aside by hand and never overwritten. `results.json` gains a `provenance` block with the
experiment, the chip set, the split, the git commit, whether the tree was dirty, the machine and
the host. `summary.csv` collects every finished cell of an experiment and chip set. Results land
in `results/<experiment>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/`, logs in
`results/<experiment>/<chip set>/logs/`.

**Every cell is also scored per parcel** (protocol D28, since 16 October 2026). Once a cell holds
its predictions, `add_parcel_metrics` scores them with `benchmark/parcel_metrics.py` and writes a
`parcel_metrics` block into `results.json`: Macro-F1, overall accuracy, kappa, weighted F1, per-class
F1 and support over the test parcels, their number and the rule. Every parcel, pooled over the test
chips it touches through the global identifier of `<chip>.parcels.tif`, takes the class most of its
labelled pixels were predicted as, ties going to the lowest class index; pixels of classes outside
the scheme do not vote. The scoring is NumPy on the stored predictions, a few seconds a cell on all
of Estonia, and a cell is scored once: a cell finished before the parcel scores existed, or pulled
from the cluster without them, gets them on the next run or from `run_kshot.py --summarise`, without
a refit. A cluster job scores only its own arms' cells, because the jobs of the other arms write into
the same folder. The block also holds the pixel Macro-F1 recomputed from the same predictions, and
`summary.csv` gains `parcel_macro_f1`, `parcel_accuracy`, `parcel_kappa`, `test_parcels` and
`predictions_match`, which is false when that recomputed score misses the logged one by more than
0.002, that is when the stored predictions are not those of the model that was tested. The rule and
its robustness were established in `experiments/2026-10-16_parcel_metric`.

**Two cells were predicted with a stale checkpoint** (found and fixed on 16 October 2026). When a
cell is fitted again into a directory that still holds the `best-loss.ckpt` of an interrupted fit,
Lightning names the new best checkpoint `best-loss-v1.ckpt`. The fit is tested with it and
`results.json` records it, but `FitPredictor` loaded `best-loss.ckpt` by name, so the stored
predictions came from the interrupted fit. It showed when the parcel scoring recomputed the pixel
Macro-F1 from the stored predictions: in 2 of the 33 full-Estonia cells scored it missed the logged
value, Prithvi at 20 % in `results/kshot_15ep/EE_2021_rev/` (0.5014 against 0.5054, the cell whose
job failed, section 11) and the 45-epoch U-Net at 100 %, seed 0, of
`experiments/2026-10-14_unet_baseline` (0.5574 against 0.5449). The other 31 reproduce it exactly,
and the logged pixel metrics are unaffected. `FitPredictor` now loads the checkpoint `results.json`
names (`fit_checkpoint`). The two cells' predictions were not redone, Prithvi's because it needs its
feature cache, which is not kept for `EE_2021_rev`; `predictions_match` flags them.

**On the cluster** each arm runs as one Slurm job (`scripts/cluster/submit.sh`,
`scripts/cluster/kshot.sbatch`). The cluster profile looks for caches under `data/embeddings/` in
its clone; a missing cache is computed onto the node's local NVMe, used for that arm's whole sweep
and deleted when the job ends. The test predictions are written inside the job for this reason,
since two-stage prediction reads the cache. The workflow between the two machines is in
[`docs/utwente_hpc.md`](../utwente_hpc.md).

---

## 10. Known gaps

These are places where the implementation and [protocol.md](protocol.md) do not yet agree, or
where the pilot is knowingly not the designed experiment.

**The 12-chip pilot was not a Phase 1 split.** Its train and validation chips were separated by
the minimum chip spacing only, with no block assignment, and it had no test partition, so its
test metrics were computed on the validation chips; every `results.json` from it records this in
`split_protocol`. The pilot chip set of section 6 replaces it and carries the parent's block
split, test partition included.

**Support sets are not nested across budgets.** Protocol decision D8 requires the support set at
one budget to be contained in the support set at any larger one, so that moving along the curve
adds labels rather than exchanging them. The current draw seeds the generator on the budget
itself, so the draws at 5 % and 20 % are independent rather than nested. Nesting would need a
stored per-class support ordering, with the budget taking a prefix of it, as protocol section
4.4 describes. Until then, part of the movement between adjacent budget points is a change of
labels rather than an addition of them.

**Single draws.** Every point currently runs one draw and one seed, so nothing has an interval.
Repeated fits of an identical configuration have shown Macro-F1 moving by several hundredths on
the twelve-chip pilot, which is the same order as the differences between the token-grid
backbones, so those differences are not yet resolvable.

**Percentages are not comparable across countries.** A percentage budget confounds the label
budget with country size, so 5 % of Estonia and 5 % of Portugal are different experiments. This
does not affect the in-country curves, which are the answer to RQ1, but the cross-country sweep
will need either a count axis or an explicit statement that the curves are not on a common axis.

**Stage 1 is not built.** [thesis_design.md](../thesis_design.md) specifies a two-stage segmentation, cropland against
non-cropland and then crop type inside the mask. Only the crop-type stage exists; the current
chips carry `ignore_index` outside the declared parcels rather than a cropland mask from ESA
WorldCover.

**Checkpoint selection consults validation labels at every budget.** `fit_end_to_end` and
`fit_cached` test the checkpoint of lowest validation loss. [protocol.md](protocol.md) allows the validation
partition only for early stopping within the full-budget fits and never at a finite budget, so
the 20 % and 5 % fits of section 11 depart from it. The criterion also moves the scores: on the
full Estonian grid the token-grid models reach their lowest validation loss at epoch 6 to 9 while
validation Macro-F1 keeps rising to epoch 13, so a checkpoint chosen on Macro-F1, or the last
epoch of a fixed budget, would score higher. TESSERA is still improving on both at epoch 15, so
it is undertrained rather than overfitted. The fit budget was 15 epochs for that grid, against
40 in the configurations, and for `kshot` until 15 October 2026; it is 45 epochs since (protocol
D27).

**Pixels never observed in the year enter as zeros.** A pixel with no clear observation in any
month of the year is written as 0 in every band. Under `revised` that holds for 100,367 pixels of
`EE_2021_rev`, against 20 under `export_2021`. Only 520 of them lie on labelled parcels. The rest
are bright, unvegetated surfaces, probably quarries, sand or built-up land, that the haze test
flags on every date. The datamodule does not treat them as nodata.

**Results before 13 October 2026 are on `EE_2021`.** The full-Estonia grid of section 11 and every
experiment before that date used the `export_2021` composites. They are not paired with fits on
`EE_2021_rev`, so a difference between the two includes the screening.

**TESSERA v1 pixels without an embedding enter as zeros.** Section 7 gives their share. The
datamodule replaces the missing vector with 0 when the chip is loaded, before normalisation, and
those pixels are still scored, so TESSERA v1 is tested on pixels it has no input for. The comment beside that replacement in `segmentation_data.py`
says the share is zero over Estonian farmland, which predates the coverage check. TESSERA v2 is
computed from the chip set's own composites, so only a pixel with neither Sentinel-2 nor
Sentinel-1 in any month lacks an embedding.

**The v1.0 store rasters are offset, for a cause not established.** Section 7 gives the
measurement, about 0.7 pixel east and 0.4 pixel south. A nearest-neighbour warp between two 10 m
grids displaces a pixel by at most half a pixel, so the warp alone does not explain it; a
georeferencing convention of the store or of the reader may. AlphaEarth's rasters are cut from
UTM tiles by the same export code, `src/gfm4agri/embeddings/export.py`, and have not been checked
for an offset.

**THOR's band groups are averaged, which its authors do not do.** THOR's authors concatenate the
groups' token maps for dense tasks. `thor_v1_large_s2s1` averages them so that it matches
TerraMind's decoder exactly; averaging and concatenation have not been compared on these chips.

**THOR meets terrain-corrected Sentinel-1 it was not pretrained on.** The chip set's Sentinel-1
is RTC gamma nought; THOR was pretrained on ordinary backscatter. Over flat Estonia the two
differ little, but the mismatch would grow on relief. TerraMind (`S1RTC`) and TESSERA (Planetary
Computer `sentinel-1-rtc`) were pretrained on RTC.

**The 10 m group has no spatial context in its decoder.** Protocol decision D21 records it: the
token-grid arms see neighbouring pixels in the ViT and again in the UNet decoder, while TESSERA
sees none at all and its per-pixel MLP adds none. The U-Net experiment of section 11 bounds what
that costs at the full budget, about 0.016 test Macro-F1 on a quarter of Estonia; at 5 % the
U-Net lost 0.093 against the MLP.

**The baseline sits outside both resolution groups.** Protocol section 4.8 computes the matching
budget `K*` within a group, and D21 compares within groups only. The U-Net outputs at 10 m and has
spatial context throughout, so it belongs to neither group, and any `K*` against it crosses the
boundary the protocol otherwise forbids. How `K*` is read against it is not yet written into the
protocol.

**Earlier `kshot` results are on other arms.** Results of `kshot` before 15 October 2026 used
TerraMind and THOR on Sentinel-2 alone and TESSERA v1 from the store; they are not paired with the
current arms.

**The parcel score is a hard vote.** Protocol section 4.6 asks for predicted probabilities to be
persisted, but the fits store only each test pixel's predicted class. A parcel is therefore scored by
the majority of its predicted classes rather than by its mean probability, and Phase 2 cannot reuse
probabilities yet. Like the pixel metrics, the parcel metrics have no interval (single draws, above).

**Feature caches on the cluster are job-scoped.** The cluster home holds 1 TB, which the Estonian
chip set (284 GB) and one token-grid cache (744 GB to 1.3 TB) do not fit together, and no project
directory has been granted. Each cluster job therefore encodes its arm's chip set again unless a
persistent cache is placed under `data/embeddings/` in the clone. A project directory linked there
would remove the repetition without a change to the code.

---

## 11. Runs

Every run so far is Estonia 2021, with one draw (`draw_seed` 0) and one training seed per cell
unless stated. Each fit directory holds the resolved configuration, the checkpoint, the Lightning
logs and `results.json`. The runs up to 8 October were made before the K-shot workflow, with the
scripts of the snapshot commit `a81bd9e`, and stay in `results/seg/` and `results/seg_cached/`;
runs of the workflow are in `results/<experiment>/<chip set>/`.

### Pilot, 12 chips

`EE_2021_pilot`, manifest `f114f0a4`, 25 to 28 September 2026. Test metrics are computed on the
validation chips, so these runs wire and debug the chain and are not results.

* End to end, at 5, 20 and 100 %, in `results/seg/*_ee_pilot/`: Prithvi-EO-2.0 300M TL and 600M TL,
  TerraMind v1 small and large, TESSERA v1 with the per-pixel MLP, and TerraMind v1 small and
  large on Sentinel-2 plus Sentinel-1.
* Two stages, at the same budgets with seeds 0 to 2, in `results/seg_cached/*_ee_pilot/`:
  TerraMind v1 small and Prithvi-EO-2.0 300M TL. They agree with the end-to-end fits within the
  seed spread (section 9).
* Adding Sentinel-1 moved TerraMind's pilot Macro-F1 by at most 0.05 in either direction, within
  what repeated fits move on the pilot, so no conclusion is drawn from it.

### The K-shot workflow pilot, `EE_2021_mini`

`EE_2021_mini` (section 6), 9 October 2026, the experiment `kshot` on the JupyterHub: the five arms at 100, 20 and
5 %, draw 0, seed 0, 15 epochs, 15 cells in `results/kshot_15ep/EE_2021_mini/`. The three token-grid caches were computed
on the fly, 5.1 GB for TerraMind, 8.3 GB for Prithvi and 11 GB for THOR, and the whole run took 68 minutes on the
A4000. With 32 training and 9 test chips this checks the workflow, not the models. Test Macro-F1:

| Arm | 100 % | 20 % | 5 % |
|---|---|---|---|
| TerraMind v1 large | 0.132 | 0.150 | 0.083 |
| Prithvi-EO-2.0 600M TL | 0.101 | 0.077 | 0.088 |
| THOR v1 large, 160 m | 0.129 | 0.071 | 0.096 |
| THOR v1 large, 80 m (add-on, run after the others) | 0.146 | 0.110 | 0.109 |
| TESSERA v1 + MLP | 0.164 | 0.155 | 0.147 |
| AlphaEarth v1 + MLP | 0.161 | 0.126 | 0.132 |

The same 15 cells then ran on the UT HPC cluster the same day, as five Slurm jobs on `itc-gpu` at commit
`0250e4c`, 2.5 to 6.5 minutes each on an RTX PRO 6000 Blackwell, with every token-grid cache computed on the node's
local NVMe and gone after the job. Every cell drew the same parcels as on the JupyterHub. The raster arms
reproduced the hub's Macro-F1 to within 0.001; the token-grid arms differed by up to 0.034 (TerraMind at 20 %),
which is the run-to-run variation of GPU fits on another GPU type and encoding batch, and on nine test chips is as
large as the gaps between arms. The cluster's results are in `results/_cluster_check/kshot/EE_2021_mini/`.

### Full Estonia

`EE_2021`, manifest `4fb7b380`, split `blocks4_buf1600_seed0__6b0eb4cb`, 1 to 2 October 2026, run
by `scripts/seg/run_ee_grid.sh` (snapshot commit `a81bd9e`) with 15 epochs and the checkpoint of
lowest validation loss. Every model is scored on the same 1,475 test chips. TerraMind v1 large and
Prithvi-EO-2.0 600M TL train in two stages from the caches of section 9; TESSERA trains end to end
on its embedding rasters. The summary is `results/seg_cached/ee_grid_summary.csv`, and
`notebooks/pipeline/eurocrops_ee_results.ipynb` holds the curves, the validation histories, the
per-class scores and maps of segmented test chips.

| Model | K, % | Training chips | Parcels drawn | Test Macro-F1 | Test mIoU | Fit, h |
|---|---|---|---|---|---|---|
| TESSERA v1 + MLP | 100 | 4,898 | 86,909 | 0.631 | 0.523 | 1.17 |
| TESSERA v1 + MLP | 20 | 4,213 | 17,390 | 0.611 | 0.504 | 0.93 |
| TESSERA v1 + MLP | 5 | 2,706 | 4,354 | 0.568 | 0.460 | 0.67 |
| TerraMind v1 large | 100 | 4,898 | 86,909 | 0.586 | 0.484 | 1.75 |
| TerraMind v1 large | 20 | 4,213 | 17,390 | 0.479 | 0.378 | 1.45 |
| TerraMind v1 large | 5 | 2,706 | 4,354 | 0.362 | 0.269 | 0.91 |
| Prithvi-EO-2.0 600M TL | 100 | 4,898 | 86,909 | 0.573 | 0.470 | 2.42 |
| Prithvi-EO-2.0 600M TL | 20 | 4,213 | 17,390 | 0.496 | 0.386 | 1.95 |
| Prithvi-EO-2.0 600M TL | 5 | 2,706 | 4,354 | 0.368 | 0.278 | 1.17 |

The fit hours of the two-stage models exclude stage 1, which took 11.5 h for TerraMind v1 large.

* TESSERA loses about 0.06 Macro-F1 from 100 to 5 %, against about 0.21 for both token-grid
  models. Part of that gap is the decoder input, since a per-pixel head on 10 m embeddings keeps
  the parcel edges that a 14 x 14 or 16 x 16 token grid must recover by upsampling, so it is
  not attributable to the encoder alone.
* TerraMind v1 large and Prithvi-EO-2.0 600M TL differ by at most 0.02 at every budget, inside
  what a second seed would move.
* The hard classes are the same for every model: green legumes, clover and alfalfa go to
  grassland, spring wheat to spring barley and oats, and vegetables and orchards are almost
  never predicted.
* The 100 % fits' test-chip predictions are stored in `results/seg_cached/ee_predictions/`.
* None of this answers RQ1 yet. No baseline has run, every cell is a single draw and seed, and
  the checkpoint criterion of section 10 is unsettled.
* THOR v1 large and AlphaEarth v1 were implemented after this grid, on 7 and 8 October, and did
  not run in it. The arms of the K-shot experiment changed on 15 October; their full-Estonia run
  on `EE_2021_rev` is listed below.

### The U-Net baseline on Sentinel-2, `EE_2021_rev`

`experiments/2026-10-14_unet_baseline`, 14 October 2026, jobs 621675 to 621677 on `itc-gpu` at
commits `ac3abbc` and `53f7783` (same code). The from-scratch U-Net of section 9 on the 144
Sentinel-2 channels alone, at 15 and at 45 epochs, against TESSERA v1 from the store with the MLP
and TerraMind v1 large on Sentinel-2 (from `experiments/2026-10-13_terramind_s1_s2`); K = 100, 20
and 5 %, draw 0, seeds 0 to 2, every arm on the same parcels. Mean test Macro-F1 over the three seeds:

| Arm | 100 % | 20 % | 5 % |
|---|---|---|---|
| U-Net, 15 epochs | 0.498 | 0.415 | 0.342 |
| U-Net, 45 epochs | 0.544 | 0.477 | 0.413 |
| TerraMind v1 large, S2 | 0.628 | 0.511 | 0.394 |
| TESSERA v1 + MLP | 0.625 | 0.597 | 0.570 |

* At 15 epochs the U-Net was still improving: its checkpoint fell at epoch 12 to 14 in all nine
  fits. 45 epochs won all nine paired fits, by 0.05 to 0.07, most in the small classes.
* At 45 epochs it trailed TerraMind by 0.084 at 100 % and 0.034 at 20 % and drew level at 5 %;
  TESSERA stayed ahead at every budget, by most at 5 %.
* The U-Net entered the main workflow on 15 October 2026 as the baseline, on Sentinel-2 and
  Sentinel-1 (section 9). The full report is the experiment's `REPORT.md`.

### The decoder of the 10 m group, `EE_2021_rev_quarter`

`experiments/2026-10-15_tessera_decoder`, 15 October 2026, jobs 623444 to 623446 on `itc-gpu` at
commit `0ab085d`. On frozen TESSERA v2 Medium rasters, the per-pixel MLP at 15 epochs against a
U-Net with a ResNet-18 down path, 14.7 M parameters trained from scratch, at 15 and at 45 epochs; K = 100, 20
and 5 %, draw 0, seeds 0 to 2, on a quarter of `EE_2021_rev`'s blocks (1,190 training, 169
validation and 411 test chips, same split). Mean test Macro-F1 over the three seeds:

| Decoder | 100 % | 20 % | 5 % |
|---|---|---|---|
| Per-pixel MLP, 15 epochs | 0.573 | 0.532 | 0.451 |
| U-Net, 15 epochs | 0.516 | 0.444 | 0.259 |
| U-Net, 45 epochs | 0.589 | 0.530 | 0.358 |

* At the protocol's 15 epochs the U-Net lost all nine paired fits.
* At 45 epochs it gained 0.016 at K = 100 % (3 of 3 fits), mostly at parcel edges, was level at
  20 % and lost 0.093 at 5 % (0 of 3).
* It makes parcels more coherent, but loses whole small parcels, concentrated in the rare
  classes, which Macro-F1 weighs as much as grassland.
* The main workflow kept the MLP (the user, 15 October 2026; protocol decision D21). The full
  report is the experiment's `REPORT.md`.

### The arms of 15 October 2026, pilots on `EE_2021_rev_mini`

The experiment `kshot` with its new arms (section 9) on `EE_2021_rev_mini`, 15 October 2026, at
commit `804dd16`: on the JupyterHub, and as five Slurm jobs on `itc-gpu` (623888 to 623892, 2.4 to
4.6 minutes each), every cache computed on the fly. The cluster's results are in
`results/_cluster/kshot_15ep/EE_2021_rev_mini/`. With 32 training and 9 test chips this checks the
workflow, not the models. Test Macro-F1, JupyterHub / cluster:

| Arm | 100 % | 20 % | 5 % |
|---|---|---|---|
| TerraMind v1 large, S2 + S1 | 0.117 / 0.135 | 0.075 / 0.082 | 0.056 / 0.040 |
| Prithvi-EO-2.0 600M TL | 0.092 / 0.091 | 0.111 / 0.084 | 0.106 / 0.079 |
| THOR v1 large, S2 + S1 | 0.118 / 0.156 | 0.067 / 0.054 | 0.065 / 0.089 |
| TESSERA v2 Medium + MLP | 0.209 / 0.209 | 0.192 / 0.191 | 0.157 / 0.157 |
| AlphaEarth v1 + MLP | 0.161 / 0.161 | 0.126 / 0.126 | 0.132 / 0.133 |

* The 10 m arms agree across the two machines to within 0.001. The token-grid arms differ by up
  to 0.038, Prithvi, whose code did not change, by up to 0.027: the run-to-run variation of GPU
  fits seen on `EE_2021_mini` on 9 October, not an effect of the change.
* THOR on Sentinel-2 and Sentinel-1 has exactly TerraMind's 53,281,492 trainable parameters.
* TESSERA v2's cache took 0.64 s per chip on the cluster. The caches came to 5.1 GB each for
  TerraMind and THOR, 8.3 GB for Prithvi and 1.4 GB for TESSERA v2.

### Full Estonia, the arms of 15 October 2026, 15 epochs

`EE_2021_rev`, the experiment `kshot` at commit `27a51a8` with 15 epochs, five Slurm jobs on
`itc-gpu` (623991 to 623995), 15 October 2026, every cache computed on the node's local NVMe.
Prithvi's job failed during its 20 % fit, when a data-loader worker could not unlink a
shared-memory file, a fault of the node; its two remaining cells ran as job 624290. The baseline
did not run at this budget. The results are in `results/kshot_15ep/EE_2021_rev/`. Test Macro-F1,
draw 0, seed 0:

| Arm | 100 % | 20 % | 5 % |
|---|---|---|---|
| TerraMind v1 large, S2 + S1 | 0.661 | 0.566 | 0.433 |
| THOR v1 large, S2 + S1 | 0.594 | 0.547 | 0.393 |
| Prithvi-EO-2.0 600M TL, S2 | 0.580 | 0.505 | 0.445 |
| TESSERA v2 Medium + MLP | 0.605 | 0.562 | 0.504 |
| AlphaEarth v1 + MLP | 0.521 | 0.465 | 0.427 |

* TerraMind on Sentinel-2 and Sentinel-1 reproduces the seed-0 fits of
  `experiments/2026-10-13_terramind_s1_s2` to every digit.
* Frozen TESSERA v2 scores below TESSERA v1 from the store fitted the same way, on the same
  parcels and seed (`experiments/2026-10-14_unet_baseline`: 0.623, 0.599 and 0.572), by 0.018,
  0.037 and 0.068. Its validation Macro-F1 is lower at every budget, and at 5 % its checkpoint of
  lowest validation loss fell at epoch 4 while validation Macro-F1 peaked at epoch 13. Its cache
  was checked: its normalisation statistics match the rasters, and the rasters match those of
  `experiments/2026-10-14_tessera_composites` at a cosine similarity of 0.9999.
* AlphaEarth is the lowest arm. Its rasters pass through the export code of the TESSERA store and
  have not been checked for the offset found there.
* Checkpoints fell at epochs 4 to 14. The budget became 45 epochs the same day (protocol D27), so
  these cells are not the main workflow's results.

Per parcel (protocol D28), scored on 16 October 2026 from the stored predictions over the 34,610
test parcels, Macro-F1 / overall accuracy:

| Arm | 100 % | 20 % | 5 % |
|---|---|---|---|
| TerraMind v1 large, S2 + S1 | 0.583 / 0.787 | 0.515 / 0.742 | 0.381 / 0.689 |
| THOR v1 large, S2 + S1 | 0.516 / 0.753 | 0.491 / 0.727 | 0.349 / 0.673 |
| Prithvi-EO-2.0 600M TL, S2 | 0.535 / 0.763 | stale predictions (section 9) | 0.409 / 0.677 |
| TESSERA v2 Medium + MLP | 0.589 / 0.793 | 0.547 / 0.779 | 0.502 / 0.755 |
| AlphaEarth v1 + MLP | 0.474 / 0.745 | 0.432 / 0.723 | 0.408 / 0.715 |

* Per parcel every arm is lower than per pixel: the token-grid models by 0.04 to 0.08 Macro-F1,
  AlphaEarth by 0.02 to 0.05, TESSERA by 0.002 to 0.016, so TESSERA leads per parcel at every
  budget. Parcels under 0.16 ha, 5 % of the test parcels, are voted their class 24 to 38 % of the
  time, and the pixel score hardly sees them (`experiments/2026-10-16_parcel_metric`).

### Full Estonia, the arms of 15 October 2026, 45 epochs

`EE_2021_rev`, the experiment `kshot` at commit `94f87ca`, the six arms with the baseline at 45
epochs, six Slurm jobs on `itc-gpu` (624583 to 624588), submitted on 15 October 2026. Results to
follow.
