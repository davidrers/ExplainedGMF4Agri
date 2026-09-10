# Exploring the use of geospatial foundation models for distinguishing grazing and mowing in grasslands

**Internship Proposal**

| | |
|---|---|
| Student | David A. Reyes Muñoz |
| Programme | MSc Geo-Information Science and Earth Observation, University of Twente (ITC) |
| Host organisation | Terramind |
| Company supervisors | Prashant Pandit, Henk Janssen |
| University supervisor | Dr. Mariana Belgiu |
| Planned internship period | 1 September 2026 to 31 December 2026 (four months) |

---

## 1. Introduction

European grassland is managed predominantly by mowing, by grazing, or by a combination of the two. Mowing is the mechanical removal of standing biomass, generally repeated several times within a growing season, and it affects the parcel abruptly and almost uniformly. Grazing is the removal of biomass by livestock. It proceeds gradually, is distributed unevenly within the parcel, and seldom reduces the canopy to bare soil.

Determining which practice occurred, and with what frequency, is relevant to biodiversity assessment, to greenhouse gas and soil carbon accounting, and to the area monitoring obligations imposed on member states under the Common Agricultural Policy. Ground records of management events are costly to collect and depend largely on farmer declarations, and they are consequently scarce.

Mowing is detected from satellite observations with reasonable reliability, since a cut produces a pronounced decline in vegetation index that is identifiable within a dense optical or radar time series. Grazing presents considerably greater difficulty. Its signal is weak, distributed over several weeks, and readily confounded with drought stress, senescence, or a partial cut. Methods constructed from hand designed features and fixed thresholds generalise poorly beyond the region and the season for which they were calibrated.

Geospatial foundation models offer a possible means of addressing this limitation. Such models are pretrained without labels on extensive volumes of satellite imagery and yield general purpose representations of spectral, spatial, and temporal structure. Two properties are pertinent here. The representation encodes temporal dynamics without requiring the analyst to specify in advance which features are informative, which is precisely where the difficulty in the grazing case resides. Furthermore, because the representation has already been learned, a classifier trained upon it requires comparatively few labelled parcels. Whether either property confers a measurable benefit for the discrimination of grazing from mowing has, to date, not been systematically assessed.

## 2. Problem evaluation

The detection of mowing is methodologically mature whereas the detection of grazing is not, although the characterisation of management intensity requires both. Existing attempts at grazing detection depend on thresholds calibrated for a single region and on reference data that are unavailable in most others.

Three difficulties follow. First, grazing produces a weak and gradual signal that overlaps with drought, senescence, and partial mowing, and its separation therefore requires the joint exploitation of spectral and temporal structure rather than a single index trajectory. Second, reliable management records exist for few areas and few seasons, which precludes any method that depends on a large training set. Third, threshold based approaches are calibrated to a particular climate, sward composition, and mowing calendar, and consequently transfer poorly.

The internship examines whether the representations produced by pretrained geospatial foundation models retain sufficient information to separate the two practices under realistic label budgets.

**Goal.** To examine the contribution that geospatial foundation models can make to the discrimination of grazing from mowing in grasslands, and to establish the conditions under which they improve upon conventional time series methods.

**Core research objective.** To evaluate whether embeddings derived from pretrained geospatial foundation models permit accurate classification of grazing and mowing with fewer labelled samples than raw spectral and radar time series require, and to identify the components of the signal responsible for that performance.

Supporting questions:

* How do foundation model embeddings compare with raw time series features for this classification task?
* How does accuracy vary with the number of labelled parcels available per class?
* Which periods of the growing season, and which spectral or radar channels, account for the separation between the two practices?

## 3. Project plan

The internship runs from 1 September to 31 December 2026. The plan is formulated at the level of phases rather than individual tasks. The study area, the reference data, and the selection of models will be established with the company supervisors during the opening weeks, and the schedule will be adjusted accordingly.

1. Onboarding and scoping. Familiarisation with the data infrastructure and the ongoing work of the host organisation, a focused literature review, and an inventory of the available reference data, which determines what is feasible.
2. Data preparation. Assembly of the Sentinel-2 and Sentinel-1 time series and of the parcel level reference dataset, together with the definition of training and testing splits, including a spatially disjoint split for the assessment of transferability.
3. Baseline. Implementation of conventional time series approaches, which establish the reference performance against which the foundation models are evaluated.
4. Foundation model embeddings. Extraction of embeddings over the same parcels, training of lightweight classifiers on the frozen representations, and comparison with the baseline under identical splits and metrics.
5. Label efficiency and interpretation. Derivation of learning curves by varying the number of labelled parcels per class, and analysis of the components of the signal on which the classifier relies.
6. Reporting. Consolidation of results, preparation of the internship report, and a final presentation.

The intermediate phases overlap where the outcome of one informs the design of the next. Regular contact with the company supervisors and periodic progress meetings with the university supervisor are foreseen throughout.

**Optional extension.** Embeddings are computed once and subsequently reused, and a classifier trained upon them can therefore be executed at interactive speed. An optional line of exploration is an embeddings based application in which the user labels grassland parcels on a map, a classifier is trained within seconds, and the resulting grazing and mowing predictions are displayed across the wider area for inspection and refinement. Such a prototype would demonstrate the embedding based approach operating as an instrument for inference on the fly and interactive exploration rather than as a batch procedure.

## 4. Expected results

The work is exploratory in character, and its contribution consists of evidence rather than a finished operational product. The following outcomes are anticipated.

* A benchmark comparing foundation model embeddings against conventional time series features for the discrimination of grazing from mowing, established on a single dataset with identical splits and metrics.
* Learning curves quantifying the manner in which accuracy scales with the number of labelled parcels per class, for both families of methods. Should the foundation models behave as the literature suggests, they will retain usable accuracy at label budgets under which the baseline degrades substantially.
* An account of the learned signal, identifying the periods of the growing season and the spectral or radar channels that carry the discriminating information, together with an assessment of whether these correspond to agronomically meaningful cues.
* A documented processing pipeline for embedding extraction and evaluation over parcel geometries, suitable for subsequent reuse within the host organisation.
* An internship report and a final presentation.
* Optionally, a prototype application for interactive labelling, inference on the fly, and visual exploration of the resulting predictions.

A negative result retains value. Should the foundation models fail to surpass the baseline, an account of where and why they fail is itself informative. One plausible failure mode is that the pretrained representation attenuates the abrupt change produced by a single mowing event, which would establish a concrete limit on the applicability of these models to grassland monitoring.
