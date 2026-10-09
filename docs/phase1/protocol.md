# Phase 1 evaluation protocol

**Label-efficient benchmarking of geospatial foundation models for pixel-level crop type segmentation on EuroCrops chips**

Version 2.0, 24 September 2026. Author: David Reyes. Addresses SO1 and RQ1.

This document specifies the evaluation protocol for Phase 1 in sufficient detail for an independent implementation. It supersedes the Phase 1 passages of `docs/proposal/current_proposal.md` and of `docs/thesis_design.md`, formerly `CLAUDE.md`, wherever the two disagree; every such departure is itemised in `docs/phase1/proposal_deltas.md`.

> **Version 2.0 revises this document onto the segmentation task.** Version 1.0 specified per-parcel classification over the EuroCropsML median time series, and was written before Phase 1 was refocused on pixel-level segmentation of EuroCrops polygons rasterised onto Sentinel-2 chips. Sections 2.4, 4.3, 4.4, 6.1 and 6.2 and decisions D1, D2, D8, D17 and D20 to D22 are rewritten. The spatial structure of section 3, the metrics of section 4.6, the matching criterion of section 4.8 and the transfer design of section 5 carry over, because they concern the evaluation rather than the input.
>
> The mechanism that implements this protocol is documented in [pipeline.md](pipeline.md), with a plain-language companion in [pipeline_overview.md](pipeline_overview.md).

---

## 0. Decisions taken

| # | Decision | Justification |
|---|---|---|
| D1 | K is defined as a **percentage of the independent training parcel polygons of each class**, applied per class. | Superseded decision: version 1.0 defined K as a count per class. The annotation unit under segmentation is the polygon, and the number of polygons available differs by more than an order of magnitude between the retained classes, so a fixed count is a different fraction of each class's pool and exhausts the rare classes long before the abundant ones. Applying the percentage per class keeps every class represented at the scarce end. The cost, that percentages are not comparable across countries, is real and is handled in section 5.5. |
| D2 | The grid is **K in {1, 5, 10, 20, 50, 100} %**, with a working subset of {5, 20, 100} % used while the pilot stands up. | The proposal's percentage grid. K = 100 % is the dense mask, which makes the top of every curve a consistency check on the budget machinery rather than a separate configuration. |
| D3 | The official EuroCropsML splits are **extended, not adopted wholesale and not replaced**, through two parallel tracks. | The official splits deliver comparability with a published benchmark but cover only one target country, apply no spatial control and offer no per-budget seed control. Both requirements are met by running both. |
| D4 | Track A, the thesis track, uses **spatially blocked partitions**; Track B reproduces the **official splits verbatim**. The difference between them on the shared cell is reported as the **spatial leakage gap**. | This reconciles "use the official protocol for comparability" with "control spatial autocorrelation" without sacrificing either, and turns the tension into a measured quantity. |
| D5 | Blocks are cells of a **square tessellation anchored on the EPSG:3035 false origin**, with the edge length estimated **per country from the label-agreement decay**. | A fixed global grid makes block identifiers stable across countries and rebuilds. Choosing the edge length from the data satisfies the requirement that the block size not be picked arbitrarily. |
| D6 | A **buffer of one kilometre** removes training-pool parcels adjacent to any held-out block. | Block assignment alone still permits leakage across a block boundary; buffering is the standard remedy (Roberts et al., 2017) and costs few parcels at the chosen block sizes. |
| D7 | The test partition is **fixed once per country, at every budget, seed, model and head**, and is **naturally distributed rather than class balanced**. | A fixed test set is what makes a learning curve a learning curve. Natural priors preserve the precision penalty that rare classes incur from abundant ones, which is the operational failure mode of interest; Macro-F1 already equalises the weight of each class in the metric. |
| D8 | Support sets are **nested across budgets** and drawn **spread across blocks** by default. | Nesting means that moving along the curve adds labels rather than exchanging them. Block spreading makes K mean K independent annotations rather than K polygons from one field cluster. **Not yet implemented:** the current draw seeds on the budget itself, so adjacent budget points draw independently; see section 4.4. |
| D9 | The support set at a given `(country, class, K, draw seed)` is a **pure function of the split configuration**, independent of model, head and iteration order. | Otherwise the comparison between foundation models is confounded by a difference in the labels each was given. |
| D10 | Head hyperparameters are **fixed once at the full budget and frozen across all budgets**, with a declared set of label-free budget-aware rules as the only exception. | Tuning on a five-label support set is not possible honestly. The bias introduced runs against the low-budget end, which is the conservative direction. |
| D11 | The headline metric is **Macro-F1**, with **seed variance and test-set variance reported separately** and the test-set interval computed by a **block cluster bootstrap**. | An independent parcel bootstrap on spatially clustered test data understates the interval. Pooling the two variance components into a single number overstates precision. |
| D12 | "The GFM matches the baseline" is decided by a **one-sided paired non-inferiority criterion at a margin of 0.02 Macro-F1**, evaluated on the same test parcels and required to hold at every larger budget. | Overlap of two independent confidence intervals has no defined error rate. Pairing removes the common test-set variance and is strictly more powerful. |
| D13 | Nine directed transfer settings are run: **six ordered country pairs and three many-to-one settings**. Three regimes are run per setting: zero-shot, target-only and source-plus-K-target. | Zero-shot and few-shot answer different questions and both bear on SO1. Target-only is the indispensable reference line. |
| D14 | In transfer, the head is **retrained on the intersected label space**, not masked at inference. | Masking leaves the decision boundaries shaped by classes that cannot occur, which conflates representation quality with an artefact of the label space. Retraining is cheap because the encoder is frozen. |
| D15 | Transfer settings involving Portugal use a **coarsened HCAT level** for the intersection and are reported as a distinct low-overlap regime. | At full HCAT depth Portugal shares approximately one top-class with each Baltic country, which is not a usable label space. |
| D16 | The class prior shift confound is separated from representation failure by **per-class F1, a prior-corrected variant, and a prior-matched test subsample**. | A drop in transfer score must not be attributed to the representation when it is attributable to the label distribution. |
| D17 | Imagery is exported **once per (country, year)** into a chip set, and no budget, seed, head or experiment may trigger a download. Frozen encoder feature maps are cached per `(model, chip)`. | Compositing is network bound and is the true bottleneck of Phase 1, at roughly 100 s per chip. A budget is a filter applied to the mask at training time, so one export serves every point on every curve. |
| D18 | The training pool is **capped per class at a single protocol-level constant, set by the most restrictive model in the sweep**, and the full-budget point is labelled accordingly. | A per-model cap would make the full-budget point a different experiment for each model. The cap must be stated wherever the full-budget point appears. |
| D19 | The recommended grid is a reduction of the full grid to approximately **fourteen thousand model fits**. | The full grid implies approximately thirty thousand fits. The reduction removes low-value cells from the transfer sweep, not from the in-country sweep, which carries the answer to RQ1. |
| D20 | **Decoder input resolution is a declared experimental factor**, not a per-model implementation detail. Two levels are defined: the per-pixel embedding raster at 10 m, and the ViT token grid at approximately 160 m upsampled back to 10 m. | Every model now consumes the same chip, so the representation confound of version 1.0 is gone. What remains is that a decoder over a 10 m embedding raster starts from 256 times more spatial detail than one over a token grid, and a Macro-F1 difference across that boundary is not attributable to the encoder. |
| D21 | Macro-F1 comparisons are made **within a resolution group**, and **decoder capacity is matched within each group**. | Within a group the encoder is the only factor that varies. Matching capacity is not automatic: TerraMind concatenates twelve monthly feature maps, so without a channel bottleneck its trainable decoder reaches 815 M parameters against Prithvi's 63 M. Every token-grid backbone now carries an identical 12.9 M decoder, and the counts are reported with every result. |
| D22 | A **resolution bridge** is run: the token-grid backbones are additionally scored with their features upsampled to 10 m before the decoder, and the per-pixel backbones with their embeddings pooled to the token grid. | The offset between the two is what permits a reader to judge how much of a cross-group difference is the decoder input resolution rather than the encoder. |
| D23 | The buffer width is set from the **largest patch footprint used by any model in the sweep**, not from the 1 km default, and is identical for every model. | A test parcel's patch must not contain a training parcel. The buffer is a property of the shared test partition, so it must be set once from the most demanding model. |

---

## 1. Relationship to the published EuroCropsML benchmark

### 1.1 What the official protocol actually does

The statements below were established by reading the installed `eurocropsml` package, version 0.4.1, in particular `eurocropsml/dataset/splits.py`, `eurocropsml/dataset/utils.py`, `eurocropsml/dataset/config.py` and the four shipped split configurations, and were cross-checked against Reuss et al. (2025) and the companion benchmarking study (Reuss et al., 2025b).

1. **The protocol is a transfer protocol, not an in-country protocol.** The four shipped configurations are `latvia_vs_estonia`, `latvia_portugal_vs_estonia`, `overlap_latvia_vs_estonia` and `overlap_latvia_portugal_vs_estonia`. In every one, `pretrain_regions` holds the Latvian NUTS-3 regions, optionally with the Portuguese NUTS-2 regions, and `finetune_regions` holds `EE`. Estonia is therefore the only target country the official splits support. There is no shipped configuration in which Latvia or Portugal is the target, and none that produces an in-country label-budget curve for any country other than Estonia.
2. **The fine-tuning split is drawn uniformly at random over parcels.** In `_create_finetune_set`, the sorted list of target-country parcels is passed to `sklearn.model_selection.train_test_split` twice, yielding sixty per cent training pool, twenty per cent validation and twenty per cent test. No `stratify` argument is supplied, so the split is not class stratified, and no spatial information enters at any point. The companion paper states the position plainly: the entire Estonian data is randomly split into train, validation and test. The reported sizes are 105,543 training, 1,000 validation and 35,182 test parcels.
3. **The K-shot support set is a per-class cap, not an exact count.** `_sample_max_samples` calls `_downsample_class` with `n_samples=max_samples`, which reduces any class holding more than `max_samples` parcels and leaves smaller classes untouched. With 127 fine-tuning classes on Estonia, a large part of the label space is therefore below the nominal budget at every K above about twenty.
4. **The budgets are nested, by accident of implementation.** `sklearn.utils.resample` with a fixed `random_state` permutes the class list and takes a prefix, so the support set at one budget is contained in the support set at any larger budget. The protocol here makes that property explicit and required rather than incidental.
5. **The test partition is fixed across budgets.** `finetune_test` is computed once, before the loop over budgets. This requirement of the thesis proposal is therefore already satisfied by the official splits.
6. **There is one seed, and it controls everything.** `EuroCropsSplit.random_seed` is a single integer, fixed at 42 in all four configurations. Changing it changes the train, validation and test partition as well as the support draw, so the official artefacts offer no way to obtain several support draws on a fixed test set. The companion paper obtains its five repetitions by varying the model initialisation, not the support set.
7. **The official grid is {1, 5, 10, 20, 100, 200, 500, all}.** There is no point at fifty.
8. **The reported metrics are overall accuracy and Cohen's kappa.** Macro-F1 is not reported.
9. **One class is downsampled during pre-training only.** `meadow_class: 3302000000`, the pasture, meadow and grassland class, is reduced to the median frequency of the remaining classes in the pre-training split. The fine-tuning split is untouched.
10. **The `overlap` configurations restrict the pre-training class set** to those classes also present in the fine-tuning set, so that no pre-training class is unseen at fine-tuning time. They do not alter the fine-tuning split.

### 1.2 Verdict: extend through two tracks

The official splits are adopted where they can be, and extended where they cannot.

**Track B, the benchmark track.** The official `region` splits `latvia_vs_estonia` and `latvia_portugal_vs_estonia` are reproduced exactly, obtained from Zenodo version 11 with `benchmark: true` so that the published partition files themselves are used rather than regenerated. Each foundation model and each head is evaluated on this split at the official budget grid. Overall accuracy and Cohen's kappa are reported, so that the numbers sit beside the published ones, and Macro-F1 is reported additionally. Five support draws are obtained by subsampling the official training pool with five draw seeds, which leaves the official test partition untouched.

**Track A, the thesis track.** A spatially blocked protocol, specified in sections 2 to 5 below, covering all three countries in-country and nine directed transfer settings. Track A carries the answer to RQ1.

The reason for two tracks rather than one is that four properties the thesis requires are absent from the official protocol and cannot be added to it without changing it:

- in-country label-budget curves for Latvia and Portugal, which the official splits do not provide at all;
- spatial separation between the training pool and the test partition, which the official splits explicitly do not attempt;
- a class set restricted to the fifteen to twenty most frequent crops, without which Macro-F1 over 127 classes is dominated by classes holding a handful of parcels;
- several support draws on a fixed test partition, which a single seed controlling both cannot deliver.

The two tracks share the cell LV to EE at every budget with a common head and feature set. The difference in Macro-F1 between Track B and Track A on that cell is reported as the **spatial leakage gap**, an estimate of how much the published benchmark is inflated by spatial autocorrelation. This is a contribution in its own right and is the reason both tracks are retained rather than one being discarded.

What is reused from the official protocol, and stated as such: the candidate class list of the `finetune_classes` block as an input to the class scheme; the meadow downsampling rule, applied in Track B only; the budget grid, extended by one point; and the fixed-test-set requirement, which Track A inherits.

---

## 2. Data and class scheme

### 2.1 Parcel catalogue

The unit of analysis is the parcel. The catalogue is one row per parcel, at `data/catalogue_parcels_full.parquet`, with the schema documented in `src/gfm4agri/data/catalogue.py`: `path`, `country`, `nuts`, `parcel_id`, `hcat`, `n_timesteps`, `lat`, `lon`, and the optional `first_date`, `last_date` and monthly observation counts `m01` to `m12`. Loading adds `country_code`, the ISO-2 code, and `uid`, the `.npz` filename stem, which is the primary key used throughout Phase 1 and which is also the key of the official split files, so the two tracks share an identifier space.

Parcels with non-finite centroids are dropped at load time and the count is recorded. If the catalogue parquet is absent, `load_catalogue` degrades to a filename-only catalogue, which supports class counting but not spatial blocking; blocking then raises rather than silently producing a non-spatial split.

### 2.2 Class scheme interface

The class scheme is fixed once, before any experiment, in `configs/class_scheme_eurocropsml.yaml`, and is derived from the data separately from this protocol. The interface this protocol requires of that file is:

```yaml
schema_version: 1
scheme_id: eurocropsml_top20_v1          # optional; the filename stem is used if absent
countries:
  EE:
    classes:
      - hcat_code: "3301010101"           # required, a string of digits
        name: common_winter_wheat         # required, snake_case English name
        n_parcels: 579                    # optional, informational
  LV: {...}
  PT: {...}
aliases:                                  # optional, applied before any counting
  "3301010100": ["3301010101", "3301010102"]
excluded:                                 # optional, informational
  - hcat_code: "3302000000"
    reason: "..."
```

Only `countries.<CC>.classes[].hcat_code` and `.name` are consumed. Any further key is ignored, so the file may carry documentation freely. Aliases are applied to the `hcat` column before counting, partitioning or sampling, so a merge is invisible to everything downstream. The scheme's content digest is stamped into every split manifest and every result file, so a change of scheme invalidates results rather than silently altering them.

If the file is absent, `load_class_scheme` falls back to the twenty most frequent classes per country holding at least five hundred parcels, ties broken by ascending HCAT code, and stamps `source: derived_fallback`. The fallback exists so the machinery is testable before the file is written; it is not the thesis scheme and results carrying that stamp are not reportable.

The pasture, meadow and grassland class 3302000000 is retained in Track A rather than excluded or downsampled. It accounts for approximately forty-five per cent of all parcels, so excluding it would produce a task that does not resemble the operational one, and downsampling it would change the test priors. Macro-F1 is the instrument that prevents it from dominating the headline number.

### 2.3 Class eligibility filter

A candidate class is retained for a country only if, after partitioning:

- its training pool holds at least `min_pool_per_class = 500` parcels, which is the largest finite budget, so that every point on the grid is attainable without a cap; and
- the test partition holds at least `min_test_per_class = 200` parcels, so that per-class F1 and the per-class bootstrap interval are meaningful.

Classes failing either test are dropped, and the drop and its reason are recorded in the split manifest. Parcels of dropped classes are excluded from every partition, so they cannot appear as false positives against the retained classes. The filter is applied after partitioning, so the retained class set is a property of the protocol seed and is recorded with it.

Provisional figures from the thirty-thousand-parcel exploratory catalogue give ten retained classes for Estonia, fifteen for Latvia and six for Portugal, at a relaxed threshold. These must be recomputed on the full catalogue before WP2.

### 2.4 Input representations

Under segmentation every model consumes the **same chip**, so the representation confound of version 1.0 largely dissolves: there is no longer a choice between a parcel median, a patch and a centroid sample. What a model consumes is a raster on the chip grid, and the levels are distinguished by what that raster holds and at what spatial resolution the decoder receives it.

| Level | Notation | Raster | Decoder input resolution | Consumed by |
|---|---|---|---|---|
| Monthly Sentinel-2 stack | `R_s2` | `<chip>_merged.tif`, 12 bands x 12 monthly composites, int16 | ViT token grid, approximately 160 m, upsampled to 10 m | TerraMind, THOR, Prithvi |
| Precomputed embedding raster | `R_emb` | `<chip>_<representation>.tif`, D dimensions per pixel, float32 | per-pixel, 10 m | TESSERA, AlphaEarth |

The chip geometry is declared once and held identical for every model: a square of 224 by 224 pixels at 10 m in EPSG:3035, that is 2.24 km on a side, on a grid anchored to the projection origin; the twelve Sentinel-2 L2A bands, reduced per model to the subset that model was pretrained on and documented per model; and the temporal axis on a fixed grid of twelve monthly composites.

Band subsetting is a per-model property held in the backbone registry rather than in a configuration file, so it cannot drift between experiments. Prithvi consumes only its six pretrained HLS bands, which on Sentinel-2 means `NIR_NARROW` is B8A. Normalisation is by default the statistics the encoder was pretrained under, since a frozen encoder only interprets inputs on the scale it saw in pretraining; an embedding raster has no pretraining statistics of its own and uses training-chip statistics.

The masks, the parcel rasters and the split files are **shared by every representation**. A label budget therefore draws the identical polygons whatever the model consumes, which is the condition under which a comparison between models is a comparison of encoders.

### 2.5 The representation confound, comparison groups and bridges

> **Version 2.0 note.** This section was written when models consumed different inputs derived from a parcel. Under segmentation they consume the same chip, so the confound narrows to the decoder input resolution of section 2.4, and the three representation groups below collapse to the two resolution groups. The reasoning about comparison groups and bridges carries over unchanged; read "representation group" as "resolution group" and the bridge experiments as the single resolution bridge of D22.

**The problem.** If TerraMind consumes `R_patch` and AlphaEarth consumes `R_emb_c`, a difference in Macro-F1 between them mixes at least three effects: the encoder itself, the input representation, and the spatial support. The patch model sees field boundaries, parcel shape, texture and the crops of neighbouring parcels; the centroid embedding sees a single pixel. Attributing the resulting difference to the encoder would be a straightforward error, and since the whole of SO1 is a comparison between encoders, it would be a fatal one. This is therefore treated as a first-class design requirement rather than a caveat in the discussion.

**The primary rule: compare within a representation group.** Three groups are defined by spatial support, and every headline comparison is made inside one group.

- **G1, the parcel-support group.** Raw-feature baselines on `R_ts`; AlphaEarth and TESSERA on `R_emb_z`; TerraMind and THOR on `R_patch_m`. Every member sees the parcel interior and nothing else. **G1 is the headline group for RQ1**, because within it the encoder is the only factor that varies, so a difference in Macro-F1 is attributable to the encoder and the matching budget `K*` of section 4.8 has its intended meaning.
- **G2, the context-inclusive group.** Raw-feature baselines on `R_patch_ts`; TerraMind and THOR on `R_patch`; AlphaEarth and TESSERA on a patch-footprint zonal mean where the provider's tiling permits it. Every member sees the parcel plus its neighbourhood. G2 answers a different and practically important question: what is achievable when the surrounding landscape is available?
- **G3, the point-support group.** AlphaEarth and TESSERA on `R_emb_c`. This is the cheapest sampling rule and the one most practitioners apply, so it is reported, but it is not a group in which encoders can be fairly compared against patch models, and no `K*` is computed across its boundary.

Every figure and every table states its group. A figure that places a G1 curve and a G3 curve on one axis without that label is not admissible.

**The bridges: estimating the representation offset at fixed encoder.** Comparison groups prevent the error but do not quantify it, and a reader is entitled to know how large the representation effect is relative to the encoder effect. Three bridge experiments do so, each holding the encoder fixed and varying only the representation.

- **Bridge A, the context bridge.** TerraMind and THOR are each run on `R_patch` and on `R_patch_m`. The difference is the value of spatial context at a fixed encoder, fixed weights and fixed patch footprint. It is the cleanest available estimate of how much of a patch model's advantage comes from seeing the neighbourhood rather than from the encoder. It also bears directly on Phase 2: if context contributes substantially, the backbone-tier attributions must be examined for reliance on neighbouring parcels.
- **Bridge B, the support bridge.** AlphaEarth and TESSERA are each run on `R_emb_c` and on `R_emb_z`. The difference is the value of aggregating over the parcel rather than sampling one pixel, at a fixed encoder. It also sharpens the AlphaEarth caveat: the annual-embedding limitation is a temporal one, and Bridge B separates it from a sampling-rule artefact.
- **Bridge C, the encoder-free bridge.** The raw-feature baseline is computed on `R_ts` and on `R_patch_ts`. Because the feature extractor is identical in both and contains no learned encoder at all, the difference is a pure estimate of the representation offset attributable to spatial support, uncontaminated by any model. It is the reference against which Bridges A and B are read.

**Reporting.** A single **representation offset table** is produced, giving, for each encoder and each bridge, the Macro-F1 difference and its paired block-bootstrap interval at K in {20, 200, POOL}, on the same fixed test set. Two summary statements are then made explicitly in the results.

1. Whether the representation offset estimated by Bridge C, which involves no encoder, accounts for most of the cross-group difference observed between G1 and G2. If it does, cross-group differences are representation effects and not encoder effects, and the thesis says so.
2. Whether the within-group encoder spread in G1 is large relative to the representation offsets. If the offsets dominate, then the practical conclusion of SO1 is that the input representation matters more than the choice of foundation model, which is a substantive and reportable finding rather than a negative result.

No additive decomposition of a cross-group difference into an encoder term and a representation term is attempted, because the design is not fully crossed: AlphaEarth and TESSERA cannot consume `R_patch`, and TerraMind and THOR cannot meaningfully consume `R_ts`, so the relevant cells do not exist. The bridges bound the representation term rather than identifying it, and the text says so rather than implying a decomposition the design cannot support.

**Consequences elsewhere in this protocol.** Three, each carried through below: the buffer width is set from the patch footprint rather than from a default (section 3.5); the training-pool cap is a single protocol-level constant set by the most restrictive model rather than a per-model value (section 4.1); and the feature cache is keyed by representation as well as by model (section 6.2).

---

## 3. Spatial structure

### 3.1 The problem

Agricultural parcels of the same crop are strongly clustered: a farm sows one crop across a contiguous run of parcels, and crop choice follows soil, climate and regional convention. A uniformly random parcel split therefore places near-duplicate parcels in both the training pool and the test set, and the resulting score measures the model's ability to recognise a field it has already seen rather than its ability to classify a new one. This is the standard problem treated by Roberts et al. (2017). It bears directly on RQ1, because leakage is most damaging at small budgets: a single leaked neighbour is worth a large fraction of a five-label support set.

### 3.2 The block grid

Blocks are the cells of a square tessellation of the ETRS89-LAEA projected plane, EPSG:3035, which is the standard equal-area projection for pan-European analysis. The tessellation is anchored on the projection's false origin rather than on the bounding box of the data, so a block identifier means the same location in every country, at every protocol seed and after any rebuild of the catalogue. A parcel at projected position `(x, y)` falls in block `(floor(x / L), floor(y / L))`, written `"<ix>_<iy>"`.

The projection is implemented in pure NumPy in `src/gfm4agri/data/projection.py` rather than delegated to `pyproj`, so that block identifiers do not depend on the PROJ version installed. The implementation reproduces the published EPSG:3035 check point at 50 N, 5 E to within one centimetre and agrees with `pyproj` to within one nanometre.

Regular blocks are preferred to NUTS regions as the blocking unit because NUTS regions vary in area by more than an order of magnitude across the three countries, which would make the effective degree of spatial separation vary with the region rather than being held constant, and because Portugal has twenty-one NUTS-3 regions against five for each Baltic country, so a NUTS-based partition would be much finer in Portugal than in the Baltics. NUTS regions are nonetheless retained in the catalogue and are reported as a secondary stratification in the results.

### 3.3 Choosing the block size from the data

The block edge length is estimated per country from the **label-agreement decay**, the categorical analogue of an empirical variogram, implemented in `src/gfm4agri/data/autocorr.py`. Writing `p0 = sum_c p_c^2` for the agreement expected under independence given the observed marginal class frequencies, the estimator forms

```
E(d) = Pr[ y_i = y_j | ||s_i - s_j|| ~ d ] - p0
```

over logarithmically spaced distance bins from 250 m to 60 km, and fits `E(d) = E0 exp(-d / a)` by weighted least squares on `log E`. The reported quantity is the **practical range** `R = 3a`, the distance at which the excess agreement has decayed to five per cent of its value at the origin. This is the direct counterpart of the practical range of an exponential variogram, and it measures exactly the quantity that causes leakage, namely the tendency of nearby parcels to carry the same crop. The block edge length is `R` rounded up to a whole kilometre and clipped to the interval from 2 km to 50 km.

Two guards prevent the estimator from inventing a range on a spatially random label field. The excess agreement in the shortest well-populated bin must exceed both an absolute floor of 0.02 and three binomial standard errors, and the log-linear fit must reach an `R^2` of 0.5. If either guard fails, no range is reported and the configured fallback of 10 km is used, with the failure recorded in the manifest. Without the first guard, restricting the fit to bins of positive excess retains only the upward half of the sampling noise and produces a spurious decay; this was observed and corrected during implementation.

### 3.4 Provisional block sizes

Run on the thirty-thousand-parcel exploratory catalogue, restricted to the derived class scheme:

| Country | Practical range | `R^2` | Block edge used | Status |
|---|---|---|---|---|
| Estonia | 10.2 km | 0.98 | 18 km | fitted |
| Latvia | not fitted | n/a | 10 km | fallback, short-range excess below the significance threshold at this sample density |
| Portugal | 244 km | 0.58 | 50 km | fitted but clipped at the maximum |

These are provisional and must be recomputed on the full catalogue, where the parcel density is approximately twenty-three times higher and the Latvian fit is expected to succeed. Two points already require comment. The Latvian failure is a density artefact of the exploratory sample, not a property of Latvia. The Portuguese estimate of 244 km exceeds any plausible field-block scale and reflects a regional gradient in class composition, driven by the concentration of the dominant Portuguese classes in particular NUTS-2 regions, rather than field-scale autocorrelation. For Portugal the clipped value of 50 km is used and the diagnostic is reported alongside it, and a NUTS-2 blocking variant is run as a sensitivity check.

### 3.5 Partition assignment and buffering

Whole blocks are assigned to partitions. The unique block identifiers are sorted, then permuted with a generator derived from `(protocol_seed, country_code, block_size_m)` and nothing else. The permutation is walked, filling the test partition until the target parcel count is reached, then the validation partition, leaving the remainder as the training pool. Because the permutation is random rather than spatially ordered, the test blocks are scattered across the country, so the test set retains the environmental and regional coverage of the population while every test parcel remains spatially separated from the training pool. Target shares are twenty per cent test, ten per cent validation and seventy per cent pool.

Block assignment alone still permits leakage across a block boundary, where two adjacent parcels of one field fall either side of a line. A buffer is therefore applied: any training-pool parcel lying within `buffer_m` of the rectangle of any test or validation block is discarded. The buffer removes training parcels only and never test parcels, so the test set remains a fixed, unbiased spatial sample. Because the buffer is smaller than the block edge, only the eight adjacent cells can be in range, so the computation is exact.

**The buffer width is set from the patch footprint, not from a round default.** Once patch-based models are in scope, a second and sharper leakage channel opens: a test parcel's patch can physically contain a training parcel, so the patch model is shown a labelled training location at test time. The buffer closes it, symmetrically in both directions, provided that

```
buffer_m  >=  half the diagonal of the largest patch footprint used by any model in the sweep
```

For the declared footprint of 224 by 224 pixels at 10 m, the half-diagonal is 1,584 m, so

```
buffer_m = 1,600 m
```

This value is a property of the shared test partition and is therefore identical for every model, including those that consume no patch at all. Setting it per model would give each model a different training pool and would destroy the comparison. If the patch footprint is later reduced, the buffer may be reduced with it, but only by changing this document and regenerating every split, since the split configuration hash depends on it.

**The buffer interacts with the block size and the interaction is not negligible.** The share of a block lying within `buffer_m` of its own edge grows roughly as the ratio of the buffer to the block edge, so a small block with a large buffer is expensive. Measured on the exploratory catalogue with the 1,600 m buffer: Estonia, on an 18 km block, loses eleven per cent of its pool; Latvia, on a 10 km fallback block, loses twenty-eight per cent; Portugal, on a 50 km block, loses three per cent. The Latvian figure is a consequence of the fallback block size and is expected to fall once the label range is fitted successfully on the full catalogue. If it does not, the correct response is to increase the Latvian block size rather than to shrink the buffer, since the buffer is fixed by the patch geometry and shrinking it would reopen the patch leakage channel.

The buffer width is varied to 1 km and to 3.2 km as a sensitivity check, and the no-buffer case is reported to quantify what the buffer is worth. The realised loss is recorded per country in the split manifest.

### 3.6 Reconciling the fixed test set with block cross-validation

The proposal asks for both a fixed held-out test set at every budget and spatial block cross-validation. These pull in different directions, and the resolution is explicit.

The **primary protocol** uses a single spatial partition, at `protocol_seed = 0`, whose test set is fixed for every budget, draw seed, model and head. This is what makes the learning curves comparable and it is the requirement the proposal is right to insist on.

The **block cross-validation robustness check** repeats the whole protocol at `protocol_seed` one to four, so that five disjoint spatial partitions each serve as the test set in turn, at the reduced budget grid {1, 20, 200}, with one head per family and all feature sets. Its purpose is to quantify how far the conclusions depend on which blocks happened to be held out, and it is reported as a robustness appendix rather than as the headline. The reported quantity is the spread of `K*`, the matching budget defined in section 4.8, across the five partitions.

---

## 4. In-country protocol

Run independently for Estonia, Latvia and Portugal.

### 4.1 Partitions

Per country, the retained parcels are divided into `test` (twenty per cent of parcels, whole blocks), `val` (ten per cent, whole blocks), `pool` (the remainder, whole blocks) and `buffer` (pool parcels discarded by the buffer rule, used for nothing). The test partition is capped at 30,000 parcels by uniform random subsampling, which preserves the natural class priors; the cap is recorded.

The training pool is capped per class. **The cap is a single protocol-level constant, set by the most restrictive model in the sweep, and is not a per-model value.** A per-model cap would make the full-budget point a different experiment for each model, so that a full-budget comparison would confound the encoder with the amount of training data. The binding constraint is patch extraction for the open-weight models, which is download bound at roughly two orders of magnitude the cost per parcel of sampling a precomputed embedding. With patch models in the sweep the cap is

```
max_pool_per_class = 2,000
```

which gives approximately 40,000 pool parcels per country at twenty classes. If the patch-based models are dropped from a given sweep, the cap may be raised to 20,000, but the resulting full-budget numbers are then not comparable with those of a sweep that included them, and the cap value in the manifest is what makes that visible.

The cap is recorded, and the full-budget point is labelled `POOL` rather than "100 per cent" in every figure and table, together with the cap value, since it is not the whole of the country's labels. Because the cap truncates the upper end of the learning curve, the full-budget point should be read as an attainable plateau and not as an asymptote; where a curve has not visibly saturated by `POOL`, that is stated rather than extrapolated.

### 4.2 The test set: size, priors and what Macro-F1 means

The test set is a **naturally distributed** spatial sample of the population, not a class-balanced one. The argument is as follows.

Macro-F1 is the unweighted mean of per-class F1, so every class already carries equal weight in the metric regardless of the test priors. Balancing the test set as well would change only the precision term: on a naturally distributed test set, a rare class's precision is penalised by false positives arriving from the abundant classes, and on a balanced test set it is not. That penalty is not a nuisance; it is the operational failure mode of crop mapping, in which a grassland-dominated landscape swamps a rare cereal. A balanced test set would produce an optimistic number that does not correspond to any deployment. Natural priors additionally permit overall accuracy and Cohen's kappa to be computed on the same predictions, which is what makes Track A and Track B comparable.

The cost is that a rare class receives few test parcels and its per-class F1 is noisy. This is controlled by the `min_test_per_class = 200` eligibility filter and by reporting per-class bootstrap intervals alongside the per-class point estimates.

### 4.3 The label budget

**K denotes the percentage of the independent training parcel polygons of each class.** The grid is

```
K in {1, 5, 10, 20, 50, 100} %
```

with the working subset `{5, 20, 100}` % used while the pilot stands up, and `K = 100` % being the dense mask.

The annotation unit is the polygon and the inference unit is the pixel, so the budget must be counted in polygons: one polygon is one act of annotation and yields a few thousand labelled pixels, and a budget counted in pixels would measure neither annotation effort nor anything a practitioner can act on.

The percentage is applied **per class**, so a class holding `n` eligible training polygons contributes `ceil(K / 100 * n)` of them and a class present at all retains at least one. The alternative, drawing K per cent of the pooled training polygons, is rejected: the retained classes differ by more than an order of magnitude in polygon count, so a pooled draw at the scarce end removes the rare classes entirely and Macro-F1 then falls for a reason that has nothing to do with the encoder under test.

This reverses decision D1 of version 1.0, which defined K as a count per class. The count definition is coherent for a per-parcel benchmark with an eligibility floor that guarantees a large pool for every retained class. Under segmentation the pool is the polygons falling inside the training chips, which is both smaller and far more uneven, so a fixed count is a different fraction of every class and saturates the rare classes while barely touching the abundant ones.

The realised draw is recorded per class at every budget, as `available` and `drawn`, so any class that could not meet its quota is visible in the result file rather than assumed away.

**The cost of the percentage definition is that budgets are not comparable across countries.** Five per cent of Estonia and five per cent of Portugal are different numbers of polygons, so the three national curves do not share an x-axis. This does not affect the in-country curves, which carry the answer to RQ1. It is handled for the cross-country comparison in section 5.5, where the curves are placed on a polygon-count axis as well as a percentage axis and the reader is told which one is being read.

### 4.4 Support-set sampling

A parcel's class is the majority class of its labelled pixels across the training chips, and a parcel cut by a chip edge contributes only the pixels inside that chip. The eligible pool for a class is the set of its polygons holding at least one labelled pixel in a training chip.

Three properties are required of the draw. Two hold today and one does not.

**Model independence.** *Holds.* The draw for a class is derived from a generator seeded on `(country, hcat_code, K, draw_seed)` and on nothing else. Every foundation model, every baseline and every head therefore sees the identical support set at a given `(country, K, seed)`, the result does not depend on the order in which classes or countries are iterated, and a partially completed sweep may be resumed or parallelised without changing any answer.

**Budget realisation is recorded.** *Holds.* The per-class `available` and `drawn` counts are written into every result file, so the support set is inspectable after the fact.

**Nesting.** *Not yet implemented.* Decision D8 requires the support set at one budget to be contained in the support set at every larger budget, so that moving along the curve adds labels rather than exchanging them. The current implementation seeds the generator on the budget itself, so the draws at 5 % and 20 % are independent; the only nesting that holds today is trivial, since `K = 100` % is the whole pool. Part of the movement between adjacent budget points is therefore a change of labels rather than an addition of them, which inflates the between-budget variance.

The fix is the version 1.0 mechanism, which should be restored: construct a single **support ordering** per `(country, class, draw_seed)`, store it once, and take the support set at budget K as the prefix of rank below `ceil(K / 100 * n)`. The ordering is built by shuffling the polygons within each spatial block, shuffling the blocks, and interleaving the per-block queues round-robin, so that the first entries fall in as many distinct blocks as possible. Without block spreading, a draw can return a set of polygons from a single field cluster, which is closer to one independent label than to many, and the x-axis of the learning curve would not mean what it claims.

The number of draws is `R = 10` at K in {1, 5} %, where the between-draw variance is largest and the fits are cheapest, `R = 5` at K in {10, 20, 50} %, and `R = 3` at `K = 100` %, where the support set is the entire pool and the only remaining variation is the decoder's own initialisation. Repeated fits of an identical configuration on the twelve-chip pilot have moved Macro-F1 by several hundredths, which is the same order as the differences between the token-grid backbones, so the draw count is not optional.

### 4.5 Validation and hyperparameter selection

Tuning hyperparameters on a five-label support set cannot be done honestly, and using the large validation partition to select per budget would inject thousands of labels of supervision into a cell nominally holding five. The scheme is therefore three-tiered, and the compromise is stated rather than hidden.

**Tier 0, used for every headline curve.** Each combination of feature set, head and country receives a single hyperparameter configuration, selected once at the full budget on the spatially separate validation partition, and then frozen across every budget, every draw seed and both tracks. The selected values are written into the run configuration and reported in an appendix. The known bias is that a regularisation strength appropriate to a pool of tens of thousands of parcels is too weak for a support set of five, so Tier 0 understates performance at the low-budget end. That is the conservative direction for a claim of the form "the GFM needs only K labels", so it is accepted, and it is stated.

**Tier 1, the only exception: budget-aware but label-free rules.** Hyperparameters that must scale with the sample size follow rules declared in advance that read only `n` and `K` and never the labels, so they cost no supervision: for penalised linear heads, the penalty scales as `1 / n` about a fixed reference; for nearest-neighbour heads, `k = min(k0, K)`; for the shallow neural head, a fixed epoch budget with a fixed learning-rate schedule and no early stopping below K equal to twenty.

**Tier 2, reported as a sensitivity and never as the headline.** At K of twenty and above, a stratified five-fold cross-validation *inside the support set* selects from a small declared grid. The difference between Tier 0 and Tier 2 is reported at K in {20, 100, POOL} so that the reader can see how much the headline depends on the compromise. Below K equal to twenty no selection is attempted, and this is stated in the figure caption rather than papered over.

The validation partition is used only for the Tier 0 selection and for early stopping within the full-budget fits. It is never consulted at any finite budget below `POOL`.

### 4.6 Metrics

**Primary:** Macro-F1 over the retained class set.

**Secondary, reported for every cell:** weighted F1, overall accuracy, Cohen's kappa (which makes Track A and Track B commensurable and matches the published benchmark), balanced accuracy, per-class precision, recall and F1, and the confusion matrix. Per-parcel predictions and predicted probabilities are persisted so that Phase 2 can reuse them without refitting, so that the bootstrap operates on stored predictions rather than on refits, and so that any metric not anticipated here can be added later without rerunning the sweep.

### 4.7 Separating seed variance from test-set variance

Two distinct sources of variation are present and are reported separately, because conflating them overstates precision.

**Seed variance** is the variation of Macro-F1 across the `R` support draws at a fixed budget. It is reported as the mean across draws, the empirical standard deviation across draws, and the minimum to maximum band. It answers: how much does the answer depend on which K labels the annotator happened to collect?

**Test-set variance** is the variation attributable to the finite test sample, for a single fitted model. It is estimated by a **block cluster bootstrap**: test blocks are resampled with replacement, all parcels of a resampled block are taken together, and Macro-F1 is recomputed from the stored predictions. `B = 1000` replicates per fitted model. The cluster bootstrap is essential rather than decorative: parcels within a block are strongly correlated, so an independent parcel bootstrap treats correlated observations as independent and returns an interval that is too narrow, by a factor that grows with the within-block correlation.

**Reporting.** The headline interval is the total interval, obtained by pooling the bootstrap replicates across draw seeds and taking the 2.5th and 97.5th percentiles of the pooled distribution. The decomposition is reported alongside it in every results table:

```
SD_total^2  ~=  SD_seed^2  +  mean_r( SD_bootstrap,r^2 )
```

with both components given numerically. A figure that shows only the bootstrap band is not acceptable, because at small K the seed component dominates and a bootstrap-only band would suggest a precision the experiment does not have.

### 4.8 The matching criterion for RQ1

RQ1 asks how few labelled samples per class are required for a GFM-based classifier to match a raw-feature baseline. The phrasing is ambiguous between matching the baseline at the same budget and matching the baseline at its full budget. The operationally meaningful reading, and the one adopted as primary, is the second: with only K per cent of each class's polygons the GFM reaches what the baseline needs the whole training pool to reach.

**The criterion is evaluated within a representation group, and the headline `K*` is the G1 one.** Per section 2.5, a matching budget computed against a baseline in a different group would confound the encoder with the input representation and would not mean what its name says. The G1 `K*` is therefore the number reported as the answer to RQ1. A G2 `K*` is reported additionally, comparing patch models against the patch-footprint baseline, which answers the practitioner's question of what is achievable when imagery is available. No `K*` is computed across a group boundary.

Let `B*` be the Macro-F1 of the better raw-feature baseline variant **of the same group** at `POOL`, and let `G(K)` be the Macro-F1 of a given foundation model, representation and head at budget K. Because both are evaluated on the **same fixed test parcels**, they must not be compared as two independent intervals. Instead, for each bootstrap replicate the *same* resampled blocks are used for both models and the paired difference is formed:

```
Delta(K) = MacroF1_GFM(K) - MacroF1_baseline(POOL)
```

The replicates are pooled across draw seeds. The criterion is one-sided non-inferiority at a margin `delta = 0.02` Macro-F1, declared in advance as a difference of no practical consequence for crop mapping:

> **K\*, the matching budget,** is the smallest K on the grid such that the 2.5th percentile of `Delta(K)` is at least `-delta`, **and** the same holds at every larger K on the grid.

The monotone-attainment clause prevents a single lucky point from being reported as the answer. Non-inferiority is preferred to an overlap-of-intervals test because overlap has no defined error rate and is known to be conservative in a way that varies with the relative interval widths, whereas a one-sided bound has a clear interpretation. Pairing removes the common test-set variance and is therefore strictly more powerful than comparing independent intervals.

Two further budgets are reported for every cell:

- **K\*_parity**, the smallest K at which the GFM matches the baseline *at the same budget*, by the same paired criterion. This answers whether the GFM is more label-efficient per label, which is a different and equally interesting question.
- **K\*_ceiling**, the smallest K at which the GFM reaches ninety-five per cent of its own `POOL` score, which locates the saturation point of the curve.

The reverse comparison, the baseline's matching budget against the GFM's full-budget score, is reported for symmetry, so that the claim is not stated only in the direction that favours the foundation models.

If no K on the grid satisfies the criterion, the result is reported as "greater than 500" or "not attained". No interpolation between grid points is performed, and no value of K absent from the grid is quoted.

### 4.9 Robustness checks

Each is run at a reduced grid and reported as a sensitivity, not as a headline:

1. block cross-validation over five spatial partitions (section 3.6);
2. buffer width of zero and of 2 km;
3. unconstrained support sampling against block-spread sampling;
4. NUTS-2 blocking for Portugal, against the 50 km grid;
5. Tier 0 against Tier 2 hyperparameter selection;
6. the spatial leakage gap, Track A against Track B on the shared cell.

---

## 5. Cross-country transfer protocol

The transfer sweep is run **within the G1 parcel-support group only**, per section 6.4. A transfer comparison across representation groups would inherit the confound of section 2.5 on top of the country shift, and nothing interpretable would remain. Every feature set in a transfer setting therefore consumes the same spatial support, and the representation is held fixed for both the source and the target country, so that a difference between them is attributable to the country and not to the input.

### 5.1 Settings

Six ordered pairs, each direction run separately because transfer is not symmetric:

```
EE -> LV,  EE -> PT,  LV -> EE,  LV -> PT,  PT -> EE,  PT -> LV
```

and three many-to-one settings, the first of which is the official EuroCropsML setting:

```
{LV, PT} -> EE,   {EE, LV} -> PT,   {EE, PT} -> LV
```

The many-to-one settings test whether adding a climatically distant source country helps or harms, which is the question the official `latvia_portugal_vs_estonia` configuration was built to ask.

### 5.2 The label space

The label space of a directed setting is the **intersection of the evaluated class sets**, not of the candidate class scheme, so a class that failed the eligibility filter in either country is excluded from the transfer setting as well. For a many-to-one setting the source class sets are intersected first, so that the label space is identical for every source country and the pooled source head is well defined. The intersection is ordered as in the target country's class set, so the label ordering is a property of the target and is stable across sources.

**Source classes absent from the target are dropped before the head is trained**, and the head is retrained on the intersected label space. It is not masked at inference. Masking leaves the softmax normalisation and the decision boundaries shaped by classes that cannot occur in the target, which conflates representation quality with an artefact of the label space, and it is also not what an operator would do. Retraining is cheap because the encoder is frozen and the head is shallow. The masked variant is nonetheless reported for the zero-shot regime alone, as a sensitivity, since it is the only regime in which the retrained head consumes source labels that the masked head does not.

**Within one transfer setting, every reported number is computed on the identical label space and the identical target test subset.** This is what makes the zero-shot score, the transfer curve and the target-only reference comparable, and it is why the reference lines in section 5.4 are indispensable rather than decorative.

### 5.3 The Portugal label-space problem

At full HCAT depth, the top-class sets of Portugal and each Baltic country intersect in approximately one class on the exploratory catalogue, which is not a usable label space for a Macro-F1 experiment. Truncating the HCAT code to six digits, which is the crop-group level, raises the intersection to approximately three classes; truncating further does not help, because the Baltic class sets then collapse as well.

The consequences are specified rather than discovered later:

1. **The EE and LV pairs are the substantive transfer experiments.** Their intersection is approximately nine classes at full depth and is expected to be larger on the full catalogue.
2. **The Portugal pairs are run at the six-digit HCAT level** and are reported as a distinct **low-overlap regime**, with the label space size printed in every caption. A Macro-F1 over three classes is not comparable with a Macro-F1 over nine and must never be placed on the same axis without that annotation.
3. **Every Portugal transfer number is interpreted only against its own target-only reference** on the same three-class space, never against the in-country curve on the full class set.

This is a finding about the dataset, not a defect of the protocol, and it belongs in the results.

### 5.4 What transfers, and the regimes run

With a frozen encoder, nothing in the representation transfers or fails to transfer by training; only the head is fitted. Three regimes are therefore distinguished, and all three are run, in this priority order.

**P1, zero-shot transfer.** The head is trained on the source country at `K_src = POOL` over the intersected label space and applied unchanged to the target test set. This asks whether the frozen representation places the same crop in the same region of embedding space in a different country, which is the sharpest available test of the representation itself. Only `K_src = POOL` is run; the smaller source budgets are dropped in the recommended grid.

**P1, target-only few-shot.** The in-country protocol of section 4, restricted to the intersected label space. This is the reference line without which no transfer number can be interpreted, and it must be refitted rather than reused from the in-country sweep, because the label space differs.

**P2, source plus K target.** The head is trained on the pooled union of the source pool at `POOL` and a K-shot target support set, over the full budget grid. This is the practical setting and it is the setting the official EuroCropsML benchmark evaluates. The source and target contributions are weighted equally by default, and a class-balanced reweighting variant is reported at the headline cells.

**P3, source-initialised adaptation,** in which a parametric head trained on the source is warm-started and then fitted on K target labels rather than the two label sets being pooled. This is run only if the schedule allows and is not part of the recommended grid.

### 5.5 Curves and reference lines

> **Version 2.0 note on the percentage budget.** K is a percentage of each class's training polygons (section 4.3), so five per cent of Estonia and five per cent of Portugal are different numbers of annotations and the national curves do not share an x-axis. Every cross-country curve is therefore drawn twice: once against K in per cent, which is the budget a practitioner controls, and once against the realised median polygon count per class, which is the annotation effort actually spent. A claim comparing two countries must state which axis it is read from. Within a country the two axes are monotone transformations of each other and the distinction does not arise.

The cross-country learning curve plots Macro-F1 on the target's fixed test set, restricted to the intersected label space, against `K_target`, the number of target labels per class, on a symmetric logarithmic axis so that the zero-shot point can be drawn at `K_target = 0`.

Three reference lines accompany every transfer curve:

1. the **zero-shot score**, as a horizontal line and as the point at `K_target = 0`;
2. the **target-only curve at the same K**, which isolates the contribution of the source data;
3. the **target-only score at `POOL`**, as the ceiling the transfer curve is trying to reach with fewer target labels.

The headline quantity derived from the pair of curves is the **label saving**: the horizontal displacement between the source-plus-target curve and the target-only curve at a fixed Macro-F1, expressed as a ratio of budgets, for example "transfer from Latvia is worth a factor of eight in Estonian labels at a Macro-F1 of 0.60". The ratio is read at the Macro-F1 levels 0.4, 0.5 and 0.6 and at the level of the target-only `POOL` score, and is reported as "greater than the grid" where the target-only curve never reaches the level.

### 5.6 Separating class prior shift from representation failure

Portugal's class distribution and crop calendar differ from those of the Baltic countries so strongly that a fall in transfer score may reflect a shift in `p(y)` rather than a failure of the representation `p(x | y)`. Four instruments separate the two, and all four are reported for every transfer setting.

1. **Per-class F1 on the intersected label space**, together with the per-class difference between the transfer setting and the target-only reference at the same budget. A roughly uniform per-class fall indicates representation failure. A fall concentrated in the classes whose prior changes most indicates prior shift. This is the primary diagnostic and it is cheap.
2. **A prior-corrected variant.** The source-trained head's posteriors are corrected for prior shift by the standard expectation-maximisation procedure of Saerens, Latinne and Decaestecker (2002), using the target prior estimated from the K target labels where they exist and a uniform prior in the zero-shot regime. Macro-F1 is reported before and after correction. The gap is the share of the fall attributable to prior shift; the residual is the share attributable to the representation.
3. **A prior-matched target test subsample.** The target test set is subsampled per class, without replacement, so that its class priors match the source priors over the intersected space. Any residual fall on this subsample is not prior shift. The subsample is smaller and noisier, so it is reported with its own bootstrap interval and is a secondary rather than a headline number.
4. **Covariates of the transfer gap.** Two scalar descriptors are computed per directed setting: the Jensen-Shannon divergence between the source and target class priors over the intersected space, and a phenological distance, the mean absolute shift in NDVI peak day-of-year per class between the two countries, taken from the raw-feature baseline phenometrics that Phase 1 already computes. The transfer gap is plotted against each across the nine settings. A gap that tracks the prior divergence and not the phenological distance, or the reverse, is a directly interpretable result.

---

## 6. Compute and data plumbing

### 6.1 Split artefacts

A split is reproducible from a configuration and a seed and is never regenerated ad hoc. The configuration is a frozen dataclass, `SplitConfig`, whose JSON serialisation is hashed with BLAKE2b to a sixteen-character `config_hash`; the artefact directory is `<name>__<hash[:8]>`. Any change to any knob produces a new directory rather than silently overwriting an old one.

The unit of partition is the **chip**, not the parcel. A chip is a cell of a fixed 2,240 m grid anchored on the EPSG:3035 origin, so a chip nests inside a spatial block whenever the block edge is a multiple of 2,240 m, and the block assignment of section 3 applies to chips without a second tessellation.

```
configs/
  seg/<run>.yaml                # one fit: data root, budget, model, trainer
  phase1/incountry.yaml         # SplitConfig for Track A, in-country
  phase1/transfer.yaml          # SplitConfig plus the transfer setting matrix
  class_scheme_eurocropsml.yaml

data/eurocrops_chips/<CC>_<year>/
  chips/                        # <chip>_merged.tif, .mask.tif, .parcels.tif, _tessera.tif
  chips/<chip>.report.json      # per-chip provenance sidecar, written as the chip lands
  manifest.json                 # assembled from the sidecars; classes, grid, imagery, stats
  <representation>.json         # per-representation sidecar and normalisation statistics

results/seg/<run_name>/<budget>_seed<s>/
  config.yaml                   # the configuration as resolved
  results.json                  # budget, realised draw, parameter counts, metrics
  checkpoints/best-loss.ckpt
```

The chip set carries **no split directory of its own** in the full-country export. Partition membership is an artefact of the split configuration, written as a chip-keyed table, so the same 56 GB export serves Track A, Track B and every robustness seed without being copied.

The support artefact stores an **ordering** rather than a support set, once the nesting of section 4.4 is implemented. One file per `(country, draw seed)` then yields every budget by prefix, which reduces the artefact count by the size of the budget grid and makes the nesting property inspectable rather than merely asserted.

### 6.2 The chip set and the feature cache

Two things are persisted, and the distinction matters because one is shared by every model and the other is not.

**The chip set is shared and is exported once per `(country, year)`.** It holds the imagery, the dense mask, the parcel raster and any precomputed embedding raster. It is exported by `scripts/data/build_country_chips.py`, which takes every grid cell holding an in-scheme parcel, composites in a process pool, and writes a per-chip report sidecar so that an interrupted run loses at most the chip in flight. A chip whose rasters and report are all present is skipped, so the command resumes.

Measured cost on the Estonian export is roughly 100 s and 7.6 MB per chip for the Sentinel-2 stack, plus 20.1 MB per chip for the 128-dimensional TESSERA raster in float32. Estonia's 7,398 chips therefore come to of the order of 56 GB of imagery and 149 GB of TESSERA, and roughly a day of wall clock at sixteen workers. **The export is the true bottleneck of Phase 1 and no budget, seed, model or experiment may trigger one.**

The manifest records the class scheme, the label rule, the grid, the imagery parameters, the per-chip labelled share, parcel count, per-class pixel counts and imagery report, the band statistics and the per-class pixel totals. Its SHA-256 is stamped into every result file as `manifest_sha256`, so a result traces to the exact chip set that produced it.

**The feature cache is per model and holds frozen encoder output.** Because the encoder is frozen, its multi-scale feature maps for a given chip are constant, so they are computed once per `(model, chip)` and decoder training reads from cache. This is an implementation optimisation and not a change of method. The eight D4 augmentation variants are cached rather than approximated in feature space, because a ViT is not exactly equivariant and rotating a feature map is not the same as encoding a rotated chip.

```
data/features/<model>/<model_version>/
    <chip>_<d4>.npy         # float16 multi-scale feature maps
    manifest.json           # model, model version, checkpoint SHA-256, band subset,
                            # normalisation, chip manifest SHA-256, D4 variant list
    shards_done.json        # completed chips, so extraction is resumable
```

**Raw imagery is not re-downloaded and composites are not recomputed.** The chip set is the cache for that stage. The levers, in the order they should be pulled if the compute budget binds, are: reduce the temporal grid from twelve monthly composites to six bimonthly ones; restrict the export to cells above a labelled-area floor, which at 5 % retains 96 % of the parcels for 82 % of the cells; and cache fewer D4 variants. The first changes the declared chip geometry and therefore requires this document to be amended and every chip re-exported. None of them may be applied to one model and not another.

### 6.3 The result manifest

Every result file in `results/phase1/runs/` carries, at minimum:

```json
{
  "run_id": "...", "protocol_id": "incountry_v1__3f2a1c08", "config_hash": "...",
  "git_commit": "...", "created_utc": "...",
  "track": "A", "setting": "in_country",
  "country": "EE", "source_country": null,
  "class_scheme_id": "eurocropsml_top20_v1", "class_scheme_digest": "...",
  "n_classes": 17, "classes": ["..."],
  "K": 20, "K_definition": "samples_per_class", "K_realised_per_class": {"...": 20},
  "draw_seed": 3, "protocol_seed": 0,
  "model": "alphaearth", "model_version": "V1_ANNUAL_2021",
  "representation": "emb_zonal", "representation_group": "G1",
  "patch_footprint_px": null, "patch_gsd_m": null,
  "feature_sha256": "...", "sampling": "zonal_mean",
  "head": "logistic_regression", "head_hparams": {"...": "..."},
  "hparam_tier": 0,
  "split_protocol": "spatial_block_L18km_buffer1km",
  "n_train": 340, "n_val": 0, "n_test": 29817,
  "metrics": {"macro_f1": 0.0, "weighted_f1": 0.0, "overall_accuracy": 0.0, "kappa": 0.0,
              "balanced_accuracy": 0.0, "per_class_f1": {"...": 0.0}},
  "bootstrap": {"scheme": "block_cluster", "B": 1000, "macro_f1_ci95": [0.0, 0.0]},
  "environment": {"python": "...", "numpy": "...", "sklearn": "...", "torch": "..."},
  "wall_seconds": 0.0
}
```

The fields `K`, `country`, `split_protocol`, `draw_seed`, `model`, `representation`, `head` and `config_hash` are the repository's reproducibility convention and are mandatory. A result file lacking any of them is not admissible into the results tables. `representation` and `representation_group` are mandatory because, per section 2.5, a Macro-F1 number whose representation group is unknown cannot be placed on any axis.

### 6.4 The fit budget

The unit of the grid is a **feature set**, defined as a `(model, representation)` pair rather than a model. Ten feature sets are in scope:

| Group | Feature sets |
|---|---|
| G1, parcel support | `baseline_phenometrics x ts_parcel`, `baseline_monthly x ts_parcel`, `alphaearth x emb_zonal`, `tessera x emb_zonal`, `terramind x patch_masked`, `thor x patch_masked` |
| G2, context inclusive | `baseline_monthly x ts_patch`, `terramind x patch_full`, `thor x patch_full` |
| G3, point support | `alphaearth x emb_centroid`, `tessera x emb_centroid` |

The bridges of section 2.5 are not additional fits; they are differences between feature sets that the grid already contains. Bridge A is `patch_full` against `patch_masked` for TerraMind and THOR, Bridge B is `emb_centroid` against `emb_zonal` for AlphaEarth and TESSERA, and Bridge C is `ts_patch` against `ts_parcel` for the monthly baseline.

**Full grid.** Eleven feature sets, four heads (nearest class mean, logistic regression, a shallow multi-layer perceptron, and histogram gradient boosting), nine budgets, three countries, and the draw counts of section 4.4.

| Component | Fits |
|---|---|
| Track A, in-country, eleven feature sets | 5,940 |
| Track A, transfer (9 settings, 3 regimes) | 41,580 |
| Track B, official benchmark | 3,520 |
| Tier 0 hyperparameter selection | 1,056 |
| Block cross-validation robustness | 2,475 |
| **Total** | **approximately 54,600** |

The transfer sweep is three quarters of the total and much of it is low-value. The individual fits are not expensive: at K of two hundred with twenty classes the support set holds four thousand rows of at most 1024 dimensions, so a linear or distance-based head fits in under a second and a boosted-tree or neural head in a few seconds. The bootstrap operates on stored predictions rather than on refits and is therefore negligible. The expensive cells are the full-budget fits, of which there are a few hundred, at minutes each.

**Recommended reduced grid.** The reduction preserves the G1 in-country sweep, which carries RQ1, and the bridge cells, which carry the representation-offset result, and cuts elsewhere.

- **Representation.** The full eleven feature sets are run in-country, because the bridges live there and the in-country sweep is the cheap one. The transfer sweep is restricted to **G1 only**, six feature sets, because a transfer comparison across representation groups is not interpretable anyway and G2 and G3 would only add uninterpretable cells.
- **Heads.** Three heads across the full grid, one per family: nearest class mean (distance-based), logistic regression (shallow linear), histogram gradient boosting (tree-based). The multi-layer perceptron is run in-country at every budget and in transfer only at K in {20, 200, POOL}.
- **Transfer feature sets.** Four of the six G1 feature sets: the two foundation models that perform best in-country within G1, AlphaEarth as the annual-embedding caveat case, and the stronger of the two raw-feature baseline variants.
- **Settings.** The six ordered pairs in full; the three many-to-one settings only at K in {20, 200, POOL}. Zero-shot at `K_src = POOL` only.

| Component | Fits |
|---|---|
| Track A, in-country, eleven feature sets, three heads plus MLP extras | approximately 5,900 |
| Track A, transfer, six ordered pairs, four G1 feature sets | 6,840 |
| Track A, transfer, three many-to-one settings | 1,260 |
| Track B | 1,440 |
| Tier 0 selection | approximately 790 |
| Block cross-validation robustness, G1 only | 810 |
| **Total** | **approximately 17,000** |

At a mean of four seconds per fit this is roughly nineteen CPU-hours, embarrassingly parallel, plus approximately eight hours for the few hundred full-budget fits. The sweep is therefore not the constraint; the feature extraction is, and within it the patch representations, which is why section 6.2 fixes the parcel set and the pool cap in advance.

**What the reduction costs.** The head-invariance claim of RQ1 is established in-country across four heads and carried over to transfer on three, so it is evidenced in-country and assumed in transfer; this is stated as a limitation. The representation-offset result is established in-country only, so the thesis does not establish whether the representation offset is itself stable under cross-country transfer, which is a genuine and acknowledged gap and a natural item for further work. Two G1 feature sets are absent from the transfer sweep. The zero-shot regime is characterised at one source budget rather than three. None of these bears on RQ1 directly.

---

## 7. Reference implementation

The splitting logic is implemented in `src/gfm4agri/data/`, is pure and deterministic given a configuration and a seed, and has no dependency on any foundation model, so it is exercisable before any embedding exists.

| Module | Responsibility |
|---|---|
| `projection.py` | Pure-NumPy EPSG:3035 forward projection, so that block identifiers do not depend on the installed PROJ version. |
| `catalogue.py` | Parcel catalogue loading, schema validation, and the filename-only degraded fallback. |
| `class_scheme.py` | The class scheme contract, YAML loading, alias merging, intersection, and the frequency-derived fallback. |
| `representations.py` | The input-representation levels, the comparison groups, the bridge definitions, the feature cache key and path convention, and `required_buffer_m`, which derives the buffer width from the patch footprint. |
| `autocorr.py` | The label-agreement decay estimator and the data-driven block size, with the two significance guards. |
| `blocks.py` | The block tessellation, block-to-partition assignment, and exact buffer computation. |
| `seeding.py` | Context-derived random number generation, which is what makes support draws model-independent and order-independent. |
| `splits.py` | `SplitConfig`, `build_split`, `build_support_order`, `support_at`, `intersect_label_space`, the artefact writer and reader, and a reader for the official EuroCropsML split files. |

The test suite is `tests/`, run with `python -m pytest tests -q`; it currently comprises seventy-nine tests, all passing. It asserts, among other properties, that the test partition is fixed across budgets and draws, that no block spans two partitions, that no retained pool parcel lies within the buffer of a held-out block, that support sets are nested, that the support draw is invariant to the ordering of the pool and of the class list, that block spreading strictly increases block diversity, that the split bundle round-trips through disk unchanged, that the comparison groups partition the representations and that each group carries exactly one spatial support, which is the property that makes a within-group comparison an encoder comparison, that every bridge crosses a group boundary, and that the default buffer width covers the declared patch footprint.

---

## 8. Open items

1. The block sizes of section 3.4 are provisional, computed on a thirty-thousand-parcel exploratory sample. They must be recomputed on `data/catalogue_parcels_full.parquet` and the table updated before WP2 begins.
2. The Portuguese practical range of 244 km reflects a regional composition gradient rather than field-scale autocorrelation. The clipped 50 km grid and the NUTS-2 variant must both be run and the choice justified from the result.
3. The retained class counts and the transfer intersections must be recomputed once `configs/class_scheme_eurocropsml.yaml` exists. If the Portugal intersections remain at three classes at the six-digit HCAT level, the P3 regime should be dropped for the Portugal pairs entirely.
4. The margin `delta = 0.02` in the matching criterion is declared here and must be fixed before any result is seen. If a supervisor prefers a different margin, it must be changed in this document before WP2 and not afterwards.
5. Whether the two raw-feature baseline variants are both carried through the full grid, or whether the stronger is selected in-country and carried alone into transfer, is settled in section 6.4 in favour of the latter. This should be confirmed once the in-country results exist.
6. The availability of parcel polygons from the EuroCrops vector release must be confirmed per country before WP1 ends, since `R_patch_m` and therefore Bridge A depend on them. If they are unavailable for a country, the circular-mask substitute of section 2.4 is used and the substitution is recorded, and the Bridge A result for that country is reported as an approximation.
7. The patch footprint of 224 pixels at 10 m is declared here on the basis of the native input size of the intended backbones. It must be confirmed against the actual TerraMind and THOR configurations before extraction begins, because the buffer width, and therefore every split, depends on it. Changing it after splits are generated requires regenerating them.
8. Whether AlphaEarth and TESSERA can supply a patch-footprint zonal mean, and therefore whether they can join the G2 group, depends on the provider tiling and must be established during WP1. If they cannot, G2 holds only the patch models and the patch-footprint baseline, which is sufficient for Bridge C but narrows the G2 comparison.
9. The total patch download volume implied by the pool cap of 2,000 parcels per class must be estimated before WP1 ends, and the levers of section 6.2 applied in the stated order if it exceeds the available budget.

---

## References

Reuss, J., Macdonald, J., Becker, S., Richter, L., and Körner, M. (2025). The EuroCropsML time series benchmark dataset for few-shot crop type classification in Europe. *Scientific Data*. https://www.nature.com/articles/s41597-025-04952-7

Reuss, J., Macdonald, J., Becker, S., Dunbar, W. A., and Körner, M. (2025b). Benchmarking for practice: few-shot time-series crop-type classification on the EuroCropsML dataset. arXiv:2504.11022.

Roberts, D. R., Bahn, V., Ciuti, S., Boyce, M. S., Elith, J., Guillera-Arroita, G., Hauenstein, S., Lahoz-Monfort, J. J., Schröder, B., Thuiller, W., Warton, D. I., Wintle, B. A., Hartig, F., and Dormann, C. F. (2017). Cross-validation strategies for data with temporal, spatial, hierarchical, or phylogenetic structure. *Ecography*, 40(8), 913 to 929.

Saerens, M., Latinne, P., and Decaestecker, C. (2002). Adjusting the outputs of a classifier to new a priori probabilities: a simple procedure. *Neural Computation*, 14(1), 21 to 41.

Snyder, J. P. (1987). *Map projections: a working manual*. USGS Professional Paper 1395.
