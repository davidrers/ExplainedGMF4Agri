# Towards an Explainable Panoptic Crop Mapping and Reporting System with Geospatial Foundation Models and Large Language Models

MSc thesis by David Reyes (student number 3598535), M-GEO 2026-2027, GEO-AI track, ITC, University of Twente.
Supervisors: Dr. Mahdi Farnaghi (first) and Dr. Mariana Belgiu (second).
Title, objective and research questions revised on 11 November 2026, when the pipeline gained the two-head
workflow that maps fields and their crops together. The former title was *Geospatial Foundation Models for
Transparent Agricultural Monitoring under Label-Scarce Conditions*; the former questions are kept in
[phase1/proposal_deltas.md](phase1/proposal_deltas.md), Delta 25. Label scarcity stays in the thesis as a dimension
of the evaluation (RQ1.2), no longer its focus.

This document states the research design: the target system, the objectives and research questions,
the task formulation, the datasets, the model set and the phases. It was the project `CLAUDE.md` until
7 October 2026, when that file became the operational guide to the repository and its current state.

Precedence: this file states the intent, [phase1/protocol.md](phase1/protocol.md) states the evaluation
mechanism, and [phase1/pipeline.md](phase1/pipeline.md) traces the implemented pipeline and carries the
reasoning for each step. When this file and the protocol disagree, the protocol is corrected to match. A
plain-language walkthrough with diagrams is in [phase1/pipeline_overview.md](phase1/pipeline_overview.md).
The working environments, repository layout and conventions are in [CLAUDE.md](../CLAUDE.md).

## Target system

A user supplies an area of interest or a bounding box. The system returns the fields of that area with
the crop each holds (a panoptic crop map), explanations of why each field was mapped as it was,
calibrated confidence per field, and a report written by a large language model that is traceable to that
evidence. Every design decision below serves that end state.

## Main objective

To design and evaluate a system that maps agricultural fields and their crops from satellite imagery with geospatial foundation models (GFMs), explains and qualifies that map, and turns it into reports grounded in that evidence with large language models (LLMs).

## Sub-objectives and research questions

- **SO1.** Panoptic crop mapping with geospatial foundation models: the fields of an area and their crop types, from frozen GFMs, against a network trained on the raw imagery.
  - **RQ1.1.** How accurately can frozen geospatial foundation models, each with a trained decoder, delineate agricultural fields and assign their crop types, compared with a segmentation network trained on the raw imagery?
  - **RQ1.2.** How does that accuracy hold as fewer fields are labelled, and across countries of the same agro-ecological zone?
- **SO2.** Explaining and qualifying the panoptic map through explainability and uncertainty quantification.
  - **RQ2.1.** Can explainability methods identify the spectral and temporal evidence behind each field's crop type and behind its borders?
  - **RQ2.2.** Can uncertainty quantification give calibrated confidence for each field's crop type and for its delineation?
  - **RQ2.3.** Do the explanations and confidence agree with agronomic knowledge, such as crop calendars, and with the raw-imagery baseline?
- **SO3.** Evidence-grounded reporting with large language models, traceable to the map, its explanations and its confidence.
  - **RQ3.1.** Given the field map, its explanations and its confidence as structured context, can a large language model write agricultural reports that are faithful to that evidence?
  - **RQ3.2.** Do the explanations and confidence make the reports better than the map alone, in accuracy, traceability and usefulness?

## Task formulation

The task is **panoptic crop mapping**: delineating the fields of an area and assigning each its crop type,
built from pixel-level predictions (a border head and a crop head on every model, protocol D30), not per-parcel
classification of parcels given in advance. Crop-type semantic segmentation alone, the crop-only workflow,
remains the reference for crop types.

The distinction that makes the label budget meaningful:

> The **annotation unit** is the parcel polygon. The **inference unit** is the pixel.

One polygon is one human annotation action and yields a few thousand labelled pixels, so K counts
polygons per class. K counted in pixels would not measure annotation effort and would not answer RQ1.2.

The fields are scored as objects, by panoptic quality under a void rule, class-agnostic and class-aware, and their
borders by precision, recall and F1 within one pixel of the declared borders (protocol D30, section 4.6). The crop
map is **scored both per pixel and per parcel**, its pixel Macro-F1 the primary crop-type metric. Beside
it, each test parcel takes the class most of its pixels were predicted as, and Macro-F1, overall accuracy
and kappa are reported over the parcels, because the EuroCrops labels are parcel declarations and the
published EuroCrops benchmarks score parcels (protocol decision D28). The pixel score weighs every
hectare and the parcel score every declaration; in Estonia they differ by up to 0.08, most for the
token-grid models, because small parcels are classified far worse.

Supervision is **sparse**. Chips are exported in full, only the K selected polygons per class are
burned into the training mask, and every other pixel carries `ignore_index`. Loss is computed on
labelled pixels only. Test blocks are labelled **densely** from the full EuroCrops layer, because the
budget restricts what the model trains on and not what it is evaluated against.

Segmentation runs in **two stages**. Stage 1 predicts cropland against non-cropland, with the mask
taken from ESA WorldCover and the declared-parcel extent. Stage 2 predicts crop type inside that mask
only. This keeps crop-type Macro-F1 uncontaminated by easy background classes while still producing a
complete wall-to-wall map for an arbitrary area of interest. Stage 1 errors propagate into stage 2 and
must be reported as such. In the two-head workflow the border head's background class, learnt from WorldCover's
non-agricultural land, already separates field from non-field, the role stage 1 was planned for; whether a separate
stage 1 is still needed is open.

## Datasets

**Primary, self-built: EuroCrops polygons rasterised onto Sentinel-2 chips.**
Parcel polygons come from the EuroCrops vector release, joined to the EuroCropsML parcel index by
`parcel_id`. Imagery is Sentinel-2 at 10 m, composited onto a **fixed monthly grid, T = 12**, over the
2021 growing season, with Sentinel-1 RTC backscatter (VV, VH) composited onto the same grid for the models
pretrained on it. Chips are 224 x 224 pixels, that is 2.24 km on a side. Estonia and Latvia are
built first; Portugal is staged and added if the schedule allows. Estimated export is roughly 100 GB
for the two Baltic countries.

**Reference index: EuroCropsML** (Reuss et al., 2025). The local copy in `data/eurocropsml/preprocess/`
holds 706,683 `.npz` files, one per parcel, filename pattern `<NUTS><id>_<parcelid>_<class>.npz`, each
carrying a `(T, 13)` per-parcel spatial median, its acquisition dates and a centroid. It ships **no
polygon geometry**, so it serves as a parcel index, a class source and a sanity reference, not as the
training data for segmentation.

**Pipeline rig: `ibm-nasa-geospatial/multi-temporal-crop-classification`.**
3,854 chips of 224 x 224 at 30 m, HLS S30, 6 bands across 3 timesteps, 13 classes from USDA CDL, CONUS
2022. TerraTorch ships `MultiTemporalCropClassificationDataModule` and working notebooks exist in
`notebooks/terratorch/`. Used to stand up and debug the segmentation, XAI and reporting chain end to
end against a published Prithvi baseline (60.64 % accuracy, 0.4269 mIoU). It is **not** a scientific
target: three timesteps cannot resolve phenology, CDL labels are themselves classifier output, it is
30 m, and it is single-country.

**Parked: CropHarvest** (Tseng et al., 2021). Its polygon subset numbers 35,169 labels with features,
but the shipped feature arrays are a single 10 m pixel at the label coordinate rather than polygon
aggregates, so it does not support segmentation without a 27 GB re-export. Reconsider only if the
thesis needs a smallholder or tropical extension.

## Selection of GFMs

Four pretrained GFMs, all evaluated with **frozen encoders and trainable decoders**. Full encoder
fine-tuning is outside the main comparison: AlphaEarth ships only embeddings, so a frozen encoder is the one
setting every model shares, and it keeps a difference in Macro-F1 a difference between representations.
What fine-tuning adds is studied separately, as experiments; for TESSERA v2 it added about 0.04 test
Macro-F1 on Estonia at every budget (15 October 2026).

Each model consumes **every sensor of the chip set it was pretrained on**, since withholding a pretrained
modality would evaluate the model below its design.

| Model | Distribution | Access | Input | Decoder input | Backbone-tier XAI |
|---|---|---|---|---|---|
| TerraMind (Jakubik et al., 2025) | open weights | TerraTorch | monthly S2 + S1 | ViT token grid, ~160 m | yes |
| THOR (Forgaard et al., 2026) | open weights | TerraTorch | monthly S2 + S1 | ViT token grid, ~160 m | yes |
| AlphaEarth Foundations (Brown et al., 2025) | precomputed annual embeddings | Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | its own sources, S1 among them | per-pixel raster, 10 m | no |
| TESSERA (Feng et al., 2025) | open weights, v2 Medium | vendored encoder, run frozen on the chip set's composites | monthly S2 + S1 | per-pixel raster, 10 m | yes |

The weights-versus-embeddings split is not merely logistical: it sets the ceiling on explainability,
because attribution back to bands, dates or patches requires gradients or perturbations through the
encoder. This is why Phase 2 is organised into an embedding tier and a backbone tier.

**Decoder input resolution is a declared experimental factor.** A decoder over a 10 m embedding raster
begins with 256 times more spatial detail than one over a 14 x 14 token grid, so a Macro-F1
difference across that boundary is not attributable to the encoder. Headline comparisons are made
within a resolution group, one decoder family is fixed per group, and decoder parameter counts are
reported alongside every result so that a reader can confirm capacity was matched. The token-grid group
carries a UNet decoder; the 10 m group carries a per-pixel MLP, because a spatial UNet over frozen TESSERA
v2 did not beat it under label scarcity (15 October 2026). The 10 m group therefore sees no spatial
context beyond what its encoder supplies, which for TESSERA, a per-pixel encoder, is none.

**AlphaEarth caveat.** AlphaEarth ships annual embeddings while the labels target a single growing
season, so its per-pixel vector carries no intra-season phenology at all. This is expected to look
like a disadvantage in the results and is a finding worth reporting, consistent with the limited
temporal sensitivity noted by Ma et al. (2025), rather than a methodological flaw. State the protocol
explicitly wherever AlphaEarth numbers appear.

## Phases

**Phase 1 is the sole active workstream.** Phases 2 and 3 are described only well enough to keep Phase 1
decisions compatible with them, and are respecified once Phase 1 lands.

### Phase 1 (SO1, RQ1.1 and RQ1.2): panoptic crop mapping

RQ1.1 and RQ1.2 are answered by the two-head workflow (protocol D30, D31): every model frozen, a border head
beside its crop head, the fields built from the border map, against PRUE's U-Net (EfficientNet-B7 from ImageNet)
trained on the raw composites, the strongest published recipe for field boundaries at 10 m. The crop-only
workflow below, with its U-Net from scratch, remains the reference for crop types alone. Both run under the
protocol of this section.

A fit is a segmentation training run: frozen encoder, cached multi-scale feature maps, trainable
decoder, per-pixel logits, cross-entropy with `ignore_index`. Because the encoder is frozen its output
for a given chip is constant, so feature maps are cached once per `(model, chip)` and decoder training
reads from cache. This is an implementation optimisation and not a change of method. `albumentations.D4`
augmentation is not free under caching, because ViTs are not exactly equivariant, so the eight D4
variants are cached rather than approximated in feature space.

Crop-only baseline: a 2D U-Net trained from scratch on the same chips, a ResNet-18 encoder randomly
initialised, about 15 M parameters, all trained, on the twelve monthly Sentinel-2 and Sentinel-1
composites stacked on 168 input channels. It sees every sensor a foundation model sees and learns its
spatial and temporal features from the labels alone, so the gap between it and a foundation model is
what pretraining contributes. It replaces the two per-pixel raw-feature variants first planned, TIMESAT
phenometrics (NDVI and EVI peak value and day-of-year, length of season, sowing and harvest
day-of-year, per-band mean and standard deviation) and the flattened monthly stack; whether the
phenometrics keep a role is open (protocol decision D26).

Protocol: the 15 to 20 most frequent crops per country; a label-budget grid K expressed as a
**percentage of the independent parcel polygons available for training, applied per class**, so a class
holding n parcels contributes ceil(K / 100 x n) of them and a class present at all keeps at least one;
multiple random draws per K; the same fixed held-out test blocks at every K; Macro-F1 with 95 %
bootstrap confidence intervals, per pixel and per parcel; spatial block cross-validation (Roberts et al., 2017). Run in-country
within each country, then repeated as cross-country transfer on the intersection of each pair of
countries' top classes.

The percentage settles the inconsistency the proposal carried between a percentage grid and a count of
samples per class. Counting polygons rather than pixels is what makes the budget a measure of
annotation effort, and applying the percentage per class rather than over the pooled training set keeps
the rare classes represented, without which Macro-F1 would fall at the scarce end for reasons that have
nothing to do with the encoder. At K = 100 the draw is every parcel, which reproduces the dense mask and
is the consistency check on the budget machinery.

**Open, to be settled in the design sections still outstanding:** the values on the K grid beyond the
working set of 5, 20 and 100 % (the fit budget is 45 epochs for every model since 15 October 2026); chip-to-block assignment
and the buffer width; HCAT depth and the class eligibility filter under segmentation; the transfer
protocol; and how pixel-level XAI is aggregated for Phase 3.

**Matched decoder capacity.** Every token-grid backbone carries the same trainable stack: a
`ChannelBottleneck` projecting the T x embed_dim channels to 768, then `LearnedInterpolateToPyramidal`
and a 12.9 M parameter `UNetDecoder`. Without the bottleneck TerraMind, which concatenates twelve
monthly feature maps, reaches 815 M trainable parameters for the large variant against Prithvi's 63 M,
which is neither comparable nor trainable on a 16 GB card.

### Phase 2 (SO2, RQ2.1 to RQ2.3): explainability and uncertainty

Not run yet. **Explainability** has an experimental set-up since 13 November 2026,
[phase2/explainability_setup.md](phase2/explainability_setup.md). It explains the crop type of each field by
feature attribution over months and bands, summarised per field: occlusion and accumulated local effects
(perturbation), SHAP and LIME (distillation), and Expected or Integrated Gradients (backpropagation). The
explanations are assessed against an agronomic ground truth per crop type and zone, assembled from
agro-ecological zones, crop calendar maps and theoretical band values, in a loop that returns to that
assessment after every round. The field borders are not explained in that set-up, and AlphaEarth, of which only
the embedding is available, is an open point. This replaces the earlier plan of Integrated Gradients, Occlusion
and AttnLRP with an embedding tier and a backbone tier.
**Uncertainty** is not designed yet. The earlier plan stands until it is: Monte Carlo Dropout as primary and
Deep Ensembles as the heavier comparison, both at decoder level, reporting per-pixel predictive entropy with
reliability diagrams and Expected Calibration Error, with confidence summarised per field, for its crop type
and for its borders. RQ2.3 is answered by comparing the importances with the agronomic ground truth, and by
placing the GFM importances on the same axes as the baseline importances.

### Phase 3 (SO3, RQ3): XAI-grounded LLM reporting

Not active. RQ3.1 asks whether reports stay faithful to their evidence, RQ3.2 whether explanations and
confidence improve them over the map alone (an ablation of the context document). A structured JSON context
document per target zone with three blocks: predictions,
explanations, and external context (ERA5 weather variables, geographic descriptors including country,
elevation and soil class from SoilGrids, and image-texture descriptors where available). Select an LLM
that follows the required output format reliably and is reproducible, generate at temperature zero,
constrain it to the context document, and score each report for traceability against its source.
Pixel-level attribution cannot be passed to an LLM directly; the fields of the panoptic map give the
aggregation its unit, and how fields are summarised into zones is a Phase 3 design problem.

## Priority under time pressure

Phases 1 and 2 are core deliverables. Phase 3 is highly valuable but is the first to be reduced if the
schedule slips. Phase 3 also degrades gracefully: if explanations fail their faithfulness checks, that
is itself a reportable Phase 2 finding, and the context document can still operate from predictions and
calibrated uncertainty alone. Within Phase 1, Portugal is the first thing cut, then the cross-country
transfer sweep, then the decoder comparison across resolution groups.

## Compute

ITC and UTwente institutional GPU cluster, with project drive storage. This is what makes per-cell
decoder training and backbone-tier attribution feasible at full scope.
