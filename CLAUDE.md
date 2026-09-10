# Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions

MSc thesis by David Reyes (student number 3598535), M-GEO 2026-2027, GEO-AI track, ITC, University of Twente.
Supervisors: Dr. Mahdi Farnaghi (first) and Dr. Mariana Belgiu (second).

The authoritative specification of this research is [docs/proposal/current_proposal.md](docs/proposal/current_proposal.md).
When this file and the proposal disagree, the proposal wins, and this file should be corrected.

## Main objective

To investigate the potential of geospatial foundation models (GFMs) as a transparent and scalable foundation for agricultural monitoring under label-scarce conditions.

## Sub-objectives and research questions

- **SO1.** Label-budget characterisation of GFM-based classifiers against a raw-feature baseline, in-country and cross-country.
  - **RQ1.** How few labelled samples per class are required for GFM-based crop classifiers to match a raw-feature baseline?
- **SO2.** Transparency and reliability of GFM-based crop classifications through explainability and uncertainty quantification.
  - **RQ2.1.** Can explainability methods identify the spectral and temporal information used by GFM-based crop classifiers?
  - **RQ2.2.** Can uncertainty quantification methods deliver reliable, calibrated confidence estimates?
  - **RQ2.3.** Do the explanations and confidence estimates agree with the raw-feature baseline and with agronomic knowledge?
- **SO3.** An XAI-grounded LLM reporting pipeline producing reports traceable to the underlying evidence.
  - **RQ3.** Can XAI-derived information serve as effective structured context for an LLM generating agricultural monitoring reports?

## Datasets

- **Primary: EuroCropsML** (Reuss et al., 2025). Approximately 706,000 labelled parcels across Estonia, Latvia and Portugal, each with a per-parcel Sentinel-2 reflectance time series for 2021 (spatial median over parcel pixels at every cloud-free acquisition). Built-in K-shot protocol and cross-country transfer protocol. Local copy in `data/eurocropsml/preprocess/`, one `.npz` per parcel, filename pattern `<NUTS><id>_<parcelid>_<class>.npz`.
- **Secondary: CropHarvest** (Tseng et al., 2021). Polygon-labelled subset only. Exercises the same pipeline on smallholder and tropical systems across Sub-Saharan Africa, South America, Central Asia and elsewhere. Candidate dataset, may be revised.

## Selection of GFMs

Four pretrained GFMs, all evaluated with frozen encoders. Full fine-tuning is out of scope, since it is unavailable for the precomputed-embedding models and tends to underperform shallow heads at small label budgets.

| Model | Distribution | Access | Backbone-tier XAI |
|---|---|---|---|
| TerraMind (Jakubik et al., 2025) | open weights | TerraTorch | yes |
| THOR (Forgaard et al., 2026) | open weights | TerraTorch | yes |
| AlphaEarth Foundations (Brown et al., 2025) | precomputed annual embeddings | Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | no |
| TESSERA (Feng et al., 2025) | precomputed pixel-time-series embeddings | Python sampling library | no |

The weights-versus-embeddings split is not merely logistical: it sets the ceiling on explainability, because attribution back to bands, dates or patches requires gradients or perturbations through the encoder. This is why Phase 2 is organised into an embedding tier and a backbone tier.

**AlphaEarth caveat.** AlphaEarth ships annual embeddings while EuroCropsML labels target a single growing season, so sampling at parcel centroids collapses intra-season phenology. This is expected to look like a disadvantage in the results and is a finding worth reporting, consistent with the limited temporal sensitivity noted by Ma et al. (2025), rather than a methodological flaw. State the protocol explicitly wherever AlphaEarth numbers appear.

## Three phases

### Phase 1 (SO1, RQ1): label-efficient benchmarking

Frozen encoders with lightweight to medium-capacity heads (distance-based, shallow neural, tree-based), so that the answer to RQ1 does not depend on a single choice of downstream classifier. Raw-feature baseline in two variants: TIMESAT phenometrics (NDVI and EVI peak value and day-of-year, length of season, sowing and harvest day-of-year, per-band mean and standard deviation) and a fixed monthly-grid resample flattened to a vector.

Protocol: the 15 to 20 most frequent crops per country; a label-budget grid K; at least five random draws per K; the same fixed held-out test set at every K; Macro-F1 with 95 % bootstrap confidence intervals; spatial block cross-validation (Roberts et al., 2017). Run in-region within each country, then repeated as cross-country transfer on the intersection of each pair of countries' top classes.

**Open inconsistency to resolve before WP2.** The proposal's methods section defines K as a percentage grid `{1, 5, 10, 20, 50, 100} %` of the available training labels, while the work-plan table states `K in {1, 5, 10, 20, 50, 100, 200}` samples per class. Settle on one definition and apply it everywhere.

### Phase 2 (SO2, RQ2.1 to RQ2.3): explainability and uncertainty

**Embedding tier**, applicable to all four models: SHAP and permutation importance over the frozen output vector; progressive per-dimension ablation following Benavides-Martinez et al. (2026); linear probes from the leading dimensions onto agronomic targets (NDVI and EVI peak day-of-year, length of season, soil-moisture proxies, per-band statistics), with probe R squared forming a binding table that links GFM dimensions to agronomic concepts.

**Backbone tier**, applicable to open-weight models only: Integrated Gradients for per-band and per-date attribution, Occlusion as a causal check through systematic blanking of bands, timesteps or patches, and AttnLRP for ViT backbones. The method set is not fixed in advance; faithfulness is evaluated per method per backbone and unreliable methods are flagged.

**Uncertainty:** Monte Carlo Dropout as the primary method and Deep Ensembles as the heavier comparison, both applied at head level over the frozen embeddings. Report predictive entropy per parcel and assess calibration with reliability diagrams and Expected Calibration Error.

RQ2.3 is answered by placing the GFM importances, after binding, on the same agronomic axes as the baseline importances.

**Note.** The proposal's backbone-tier paragraph lists "Prithvi-EO-2.0, TerraMind and THOR" although Prithvi is not part of the selected GFM set. Treat the backbone tier as TerraMind and THOR unless Prithvi is deliberately added.

### Phase 3 (SO3, RQ3): XAI-grounded LLM reporting

Assemble a structured JSON context document per target zone with three blocks: predictions, explanations, and external context (ERA5 weather variables, geographic descriptors including country, elevation and soil class from SoilGrids, and image-texture descriptors where available). Select an LLM that follows the required output format reliably and is reproducible, generate at temperature zero, constrain it to the context document, and score each report for traceability against its source. Evaluate on a sample of zones drawn from the Phase 1 held-out test partition.

## Priority under time pressure

Phases 1 and 2 are core deliverables. Phase 3 is highly valuable but is the first to be reduced if the schedule slips. Phase 3 also degrades gracefully: if explanations fail their faithfulness checks, that is itself a reportable Phase 2 finding, and the context document can still operate from predictions and calibrated uncertainty alone.

## Repository layout

```
docs/proposal/      the proposal (authoritative), the submitted .docx, the one-page idea
docs/research/      research notes and deep-research documents from the proposal phase
docs/internship/    the separate grazing-versus-mowing internship at Terramind, Sep to Dec 2026
src/gfm4agri/       the pipeline package, subpackages mapping onto the work packages
  data/             EuroCropsML and CropHarvest loading, splits, spatial blocking
  embeddings/       GFM embedding extraction (TerraTorch backbones, AlphaEarth, TESSERA)
  baselines/        TIMESAT phenometrics and monthly-grid raw-feature baselines
  benchmark/        K-shot protocol, heads, metrics, learning curves, transfer experiments
  xai/              embedding-tier and backbone-tier explainability
  uncertainty/      MC Dropout, Deep Ensembles, calibration
  reporting/        context-document assembly and LLM reporting
configs/            experiment configuration
notebooks/          exploratory notebooks
scripts/legacy/     precursor AlphaEarth and TESSERA scripts from the ML-Embeddings project
data/               datasets, git-ignored
results/            experiment outputs, git-ignored apart from results/eda
figures/            scripts producing thesis and presentation figures
```

## Conventions

- Working language is English. The user is a Spanish and English bilingual MSc student at ITC, comfortable with EO, ML and GIS terminology, so prefer concise technical responses.
- Windows environment. Bash and PowerShell are both available and take their own syntax.
- Python, with PyTorch, scikit-learn, TerraTorch, TorchGeo, rasterio and GDAL, and Earth Engine.
- Every experiment must be reproducible: fixed seeds, the configuration recorded alongside the results, and the label budget, the country and the split protocol stated in every result file.
- In prose drafted for the thesis, avoid dashes and use a formal academic register.
- Do not commit data or large binaries. `data/` and `results/` are git-ignored, apart from `results/eda`.
