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
Sentinel-2 L2A (Planetary Computer) ──→ monthly composites ──┤
Sentinel-1 RTC (Planetary Computer) ──→ monthly composites ──┤
TESSERA v1 (Source Cooperative Zarr) ──→ embedding rasters ──┘
```

Every stage writes to disk and is resumable, because the two network-bound stages, the
Sentinel-2 composites and the TESSERA reads, dominate the wall clock and must never be
repeated by accident.

| Stage | Code | Output |
|---|---|---|
| 1. Vector labels | `scripts/data/fetch_eurocrops.py`, `scripts/data/fetch_eurocropsml.py` | `data/eurocrops/parquet/<CC>_<year>.parquet` |
| 2. Chip grid and label rasters | `src/gfm4agri/data/chip_grid.py` | `<chip>.mask.tif`, `<chip>.parcels.tif` |
| 3. Sentinel-2 composites | `src/gfm4agri/chips/s2_monthly.py` | `<chip>_merged.tif` |
| 3b. Sentinel-1 composites | `src/gfm4agri/chips/s1_monthly.py`, `scripts/data/build_s1_chips.py` | `<chip>_s1rtc.tif`, `s1_rtc.json` |
| 4. Chip export | `scripts/data/build_country_chips.py`, `scripts/data/build_pilot_chips.py` | `manifest.json` and the chip set |
| 5. TESSERA embeddings | `src/gfm4agri/embeddings/tessera.py`, `scripts/data/build_tessera_chips.py` | `<chip>_tessera.tif`, `tessera_v1.json` |
| 6. Spatial split | `src/gfm4agri/data/chip_split.py`, `scripts/data/build_chip_split.py` | `splits/<name>__<hash8>/`, `chip_parcels.parquet` |
| 7. Supervision | `src/gfm4agri/benchmark/segmentation_data.py` | the training mask at a budget |
| 8. Model assembly | `src/gfm4agri/benchmark/{backbones,necks,decoders,segmentation}.py` | a `SemanticSegmentationTask` |
| 9. The fit | `scripts/seg/train.py` end to end, or `scripts/seg/encode.py` then `scripts/seg/fit_cached.py` in two stages | `results/seg/<run>/<budget>_seed<s>/results.json` end to end, `results/seg_cached/...` in two stages; the stage 1 cache in `data/embeddings/<backbone>/<chip set>/` |
| 10. Inspection | `src/gfm4agri/benchmark/evaluation.py`, `notebooks/terratorch/eurocrops_ee_results.ipynb` | reloaded fits, test-chip predictions in `results/seg_cached/ee_predictions/` |

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
sits on the `reflectance = DN / 10000` convention. Every 2021 scene predates the change, so for
this season the correction is a guard rather than an adjustment, and it is kept because the
pipeline is meant to extend to later years.

**Cloud and snow screening** from the scene classification layer. This is the correction that
matters most and the one the vendored loaders do not perform: nodata, saturated, cloud shadow,
cloud medium and high probability, thin cirrus and snow are all excluded from the median.

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

**What the screening misses.** `notebooks/06_monthly_composites_qa.ipynb` opens the stored
composites of six parcels exactly as the datamodule reads them, beside their Sentinel-1 composites
and, for two parcels, the individual acquisitions. Three faults show there, and none is corrected
yet. Snow and cloud that the scene classification marks clear enter the median: the few pixels
observed in an Estonian January or December are mostly such leaks, and one grassland parcel's
December composite is snow, at NDVI -0.04. Residual cloud and haze also survive in months
reported as fully observed, so the filled share is not a measure of composite quality. And because
each pixel takes its median over its own clear acquisitions, a cloud edge on a single date becomes
a straight seam inside a field. Separately, and by design, the monthly median dilutes canopy peaks
shorter than a month: a potato parcel above NDVI 0.8 for three weeks has July and August
composites of 0.53 and 0.51, in agreement with the monthly median of its acquisitions.

The per-pixel filled mask is not stored, but it can be recovered exactly from the stack: a filled
month lies on the line between its neighbours in all twelve bands at once, with a second difference
of at most 2 after rounding to int16, and a filled month at either end repeats its neighbour. On the
home chips of those six parcels the recovered shares equal the exported ones to the four decimals
the report keeps.

### Sentinel-1 RTC composites

The only consumer of Sentinel-1 among the models is TerraMind, through its `S1RTC` modality, so the
format is TerraMind's: two bands, VV then VH, as backscatter in decibels. `s1_monthly.py` reads the
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

`scripts/data/build_s1_chips.py` writes the rasters beside the Sentinel-2 chips, either for every cell
of a country or, with `--from-manifest`, for the chips of an existing set such as the pilot. Reports go
to `s1_reports/<chip>.json` and the provenance and normalisation statistics to `s1_rtc.json`, kept apart
from the Sentinel-2 manifest so that neither picks up the other's chips and the manifest hash stamped
in earlier results stays valid. All 7,402 Estonian chips have Sentinel-1.

In the model, the backbone registry declares `S1RTC` as an extra modality of `terramind_v1_small_s2s1`
and `terramind_v1_large_s2s1`. The datamodule interleaves the Sentinel-1 raster with the Sentinel-2 stack
by month, so augmentation keeps the two aligned, normalises each with its own statistics (TerraMind's
`S1RTC` pretraining statistics under `normalisation: backbone`, those of `s1_rtc.json` otherwise), and
splits them back into one tensor per modality after normalisation. TerraMind averages the tokens of the two modalities at every position, so the token
grid, the neck and the decoder are exactly those of the Sentinel-2 run and the two compare at matched
capacity. So far only the pilot has been fitted with Sentinel-1 (section 11).

---

## 6. Chip export

Two scripts write chip sets with the same on-disk layout.

`build_pilot_chips.py` selects a handful of well-labelled, widely separated cells and splits
them into `training_chips/` and `validation_chips/`. It exists to wire up and debug the
TerraTorch connection, and its manifest says so: *pipeline test set for the TerraTorch
connection; not a Phase 1 split*.

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
block once instead of once per chip and cuts the downloaded bytes by about sixteen times on a
densely chipped scene. It is **not validated**. A window takes long enough that the download
tokens issued at search time expire partway through, and its speed gain has not been measured
cleanly. The chip-by-chip export is the one in use.

The manifest is the provenance record: the class scheme, the label rule, the grid, the imagery
parameters, the per-chip labelled share, parcel count, per-class pixel counts and imagery
report, the band statistics, and the per-class pixel totals. `results.json` stamps its SHA-256,
so a result can always be tied to the exact chip set that produced it.

### The spatial split of a full-country chip set

`scripts/data/build_chip_split.py`, with the logic in `src/gfm4agri/data/chip_split.py`, assigns
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
parcels before drawing. At 100 % the draw is therefore every trainable parcel, and `train.py`
treats `dense` under a split as that draw, so a buffered parcel is never trained on.

Every consumer takes the split directory as `data.split` in the run configuration:
`EuroCropsSegDataModule`, `draw_support`, the stage 1 encoder of `cached.py` and its stage 2
datamodule. The result file records the split's name, hash, protocol and chip counts under
`split_protocol`.

---

## 7. TESSERA embeddings

TESSERA v1 encodes each 10 m pixel's full-year Sentinel-1 and Sentinel-2 series into one
128-dimensional vector. The embeddings are precomputed and served from Source Cooperative as a
Zarr v3 store, one group per UTM zone, quantised to int8 with one float32 scale per pixel.

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

A fit is a **frozen encoder with a trainable decoder**. Full encoder fine-tuning is out of
scope because it is unavailable for the precomputed-embedding models, so allowing it for the
open-weight models would make the comparison incoherent.

`backbones.py` is the registry. Each entry declares the model arguments, the pretraining
statistics, the band subset the encoder consumes, the resolution group, and whether the encoder
takes coordinates. The registry is what keeps per-model differences declarative: the datamodule
reads the band subset and the statistics from it rather than having them copied into a config
by hand.

**Inputs differ by model, and the differences are declared rather than incidental.** Prithvi
consumes only its six pretrained HLS bands, so on Sentinel-2 `NIR_NARROW` is B8A. TerraMind is
single-date, so a temporal wrapper runs it per month and concatenates the monthly features.
Prithvi is natively multi-temporal, so the twelve months enter one ViT as twelve token planes
with a 3D positional embedding. Encoders with date and location encodings additionally receive
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
  trainable on a 16 GB card. The projection also mixes the months of each token position, so it
  is where the decoder first learns which dates matter.
* `pixel_raster`: `PixelMLPDecoder`, a per-pixel MLP written as 1 x 1 convolutions, two hidden
  layers of 512 and 256. Every output pixel depends only on that pixel's embedding, so the
  score measures the embedding itself. This follows the head the TESSERA authors use for crop
  classification.

The task is a TerraTorch `SemanticSegmentationTask` built through `EncoderDecoderFactory`, with
`freeze_backbone=True`, AdamW, and cross-entropy that ignores the unlabelled pixels.

`train.py` resolves the configuration, draws the support set, builds the datamodule and the
task, fits, tests from the best validation-loss checkpoint, and writes `results.json` carrying
the configuration, the label budget and its realised per-class draw, the country, the split
protocol, the parameter counts by component, the fit time, the peak GPU memory and every
metric. Gradient accumulation keeps the effective batch identical across backbones when a
larger encoder forces a smaller micro-batch onto the card.

### The two-stage fit

Because the encoder is frozen, its output for a given chip never changes, so `train.py`
re-encodes every chip at every epoch of every fit for nothing. `src/gfm4agri/benchmark/cached.py`
implements TerraTorch's second workflow instead. **Stage 1**, `scripts/seg/encode.py`, runs the
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

---

## 10. Known gaps

These are places where the implementation and [protocol.md](protocol.md) do not yet agree, or
where the pilot is knowingly not the designed experiment.

**The pilot is not a Phase 1 split.** Train and validation chips are separated by the minimum
chip spacing only, with no block assignment, and there is no test partition, so test metrics
are computed on the validation chips. Every `results.json` from the pilot records this in
`split_protocol`. The full-country chip set carries the spatial block split of section 6.

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

**Checkpoint selection consults validation labels at every budget.** `train.py` and `fit_cached`
test the checkpoint of lowest validation loss. [protocol.md](protocol.md) allows the validation
partition only for early stopping within the full-budget fits and never at a finite budget, so
the 20 % and 5 % fits of section 11 depart from it. The criterion also moves the scores: on the
full Estonian grid the token-grid models reach their lowest validation loss at epoch 6 to 9 while
validation Macro-F1 keeps rising to epoch 13, so a checkpoint chosen on Macro-F1, or the last
epoch of a fixed budget, would score higher. TESSERA is still improving on both at epoch 15, so
it is undertrained rather than overfitted. The fit budget, fixed at 15 epochs for that grid
against 40 in the configurations, is open with it.

**TESSERA pixels without an embedding enter as zeros.** Section 7 gives their share. The
datamodule replaces the missing vector with 0 when the chip is loaded, before normalisation, and
those pixels are still scored, so TESSERA is tested on pixels it has no input for. The comment beside that replacement in `segmentation_data.py`
says the share is zero over Estonian farmland, which predates the coverage check.

---

## 11. Runs

Every run so far is Estonia 2021, with one draw (`draw_seed` 0) and one training seed per cell
unless stated. Each fit directory holds the resolved configuration, the checkpoint, the Lightning
logs and `results.json`.

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

### Full Estonia

`EE_2021`, manifest `4fb7b380`, split `blocks4_buf1600_seed0__6b0eb4cb`, 1 to 2 October 2026, run
by `scripts/seg/run_ee_grid.sh` with 15 epochs and the checkpoint of lowest validation loss. Every
model is scored on the same 1,475 test chips. TerraMind v1 large and Prithvi-EO-2.0 600M TL train
in two stages from the caches of section 9; TESSERA trains end to end on its embedding rasters.
The summary is `results/seg_cached/ee_grid_summary.csv`, and
`notebooks/terratorch/eurocrops_ee_results.ipynb` holds the curves, the validation histories,
the per-class scores and maps of segmented test chips.

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
