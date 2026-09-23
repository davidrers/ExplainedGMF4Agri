# Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions

MSc thesis by David Reyes (student number 3598535), M-GEO 2026-2027, GEO-AI track, ITC, University of Twente.
Supervisors: Dr. Mahdi Farnaghi (first) and Dr. Mariana Belgiu (second).

The authoritative specification of the current work is this file together with [docs/phase1/protocol.md](docs/phase1/protocol.md).
When the two disagree, this file states the intent and the protocol states the mechanism, and the protocol is corrected to match.

## Target system

A user supplies an area of interest or a bounding box. The system returns a pixel-level crop
segmentation of that area, pixel-level explanations of why each region was classified as it was,
calibrated per-pixel confidence, and an LLM-written report traceable to that evidence. Every design
decision below serves that end state.

## Main objective

To investigate the potential of geospatial foundation models (GFMs) as a transparent and scalable foundation for agricultural monitoring under label-scarce conditions.

## Sub-objectives and research questions

- **SO1.** Label-budget characterisation of GFM-based crop segmentation against a raw-feature baseline, in-country and cross-country.
  - **RQ1.** How few labelled parcel polygons per class are required for GFM-based crop segmentation to match a raw-feature baseline?
- **SO2.** Transparency and reliability of GFM-based crop segmentation through explainability and uncertainty quantification.
  - **RQ2.1.** Can explainability methods identify the spectral and temporal information used by GFM-based crop segmenters?
  - **RQ2.2.** Can uncertainty quantification methods deliver reliable, calibrated confidence estimates?
  - **RQ2.3.** Do the explanations and confidence estimates agree with the raw-feature baseline and with agronomic knowledge?
- **SO3.** An XAI-grounded LLM reporting pipeline producing reports traceable to the underlying evidence.
  - **RQ3.** Can XAI-derived information serve as effective structured context for an LLM generating agricultural monitoring reports?

## Task formulation

The task is **pixel-level crop-type semantic segmentation**, not per-parcel classification.

The distinction that makes the label budget meaningful:

> The **annotation unit** is the parcel polygon. The **inference unit** is the pixel.

One polygon is one human annotation action and yields a few thousand labelled pixels, so K counts
polygons per class. K counted in pixels would not measure annotation effort and would not answer RQ1.

Supervision is **sparse**. Chips are exported in full, only the K selected polygons per class are
burned into the training mask, and every other pixel carries `ignore_index`. Loss is computed on
labelled pixels only. Test blocks are labelled **densely** from the full EuroCrops layer, because the
budget restricts what the model trains on and not what it is evaluated against.

Segmentation runs in **two stages**. Stage 1 predicts cropland against non-cropland, with the mask
taken from ESA WorldCover and the declared-parcel extent. Stage 2 predicts crop type inside that mask
only. This keeps crop-type Macro-F1 uncontaminated by easy background classes while still producing a
complete wall-to-wall map for an arbitrary area of interest. Stage 1 errors propagate into stage 2 and
must be reported as such.

## Datasets

**Primary, self-built: EuroCrops polygons rasterised onto Sentinel-2 chips.**
Parcel polygons come from the EuroCrops vector release, joined to the EuroCropsML parcel index by
`parcel_id`. Imagery is Sentinel-2 at 10 m, composited onto a **fixed monthly grid, T = 12**, over the
2021 growing season. Chips are 224 x 224 pixels, that is 2.24 km on a side. Estonia and Latvia are
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
fine-tuning is out of scope, since it is unavailable for the precomputed-embedding models.

| Model | Distribution | Access | Decoder input | Backbone-tier XAI |
|---|---|---|---|---|
| TerraMind (Jakubik et al., 2025) | open weights | TerraTorch | ViT token grid, ~160 m | yes |
| THOR (Forgaard et al., 2026) | open weights | TerraTorch | ViT token grid, ~160 m | yes |
| AlphaEarth Foundations (Brown et al., 2025) | precomputed annual embeddings | Earth Engine `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` | per-pixel raster, 10 m | no |
| TESSERA (Feng et al., 2025) | precomputed pixel-time-series embeddings | Python sampling library | per-pixel raster, 10 m | no |

The weights-versus-embeddings split is not merely logistical: it sets the ceiling on explainability,
because attribution back to bands, dates or patches requires gradients or perturbations through the
encoder. This is why Phase 2 is organised into an embedding tier and a backbone tier.

**Decoder input resolution is a declared experimental factor.** A UNet over a 10 m embedding raster
begins with 256 times more spatial detail than a UperNet over a 14 x 14 token grid, so a Macro-F1
difference across that boundary is not attributable to the encoder. Headline comparisons are made
within a resolution group, one decoder family is fixed per group, and decoder parameter counts are
reported alongside every result so that a reader can confirm capacity was matched.

**AlphaEarth caveat.** AlphaEarth ships annual embeddings while the labels target a single growing
season, so its per-pixel vector carries no intra-season phenology at all. This is expected to look
like a disadvantage in the results and is a finding worth reporting, consistent with the limited
temporal sensitivity noted by Ma et al. (2025), rather than a methodological flaw. State the protocol
explicitly wherever AlphaEarth numbers appear.

## Phases

**Phase 1 is the sole active workstream.** Phases 2 and 3 are described only well enough to keep Phase 1
decisions compatible with them, and are respecified once Phase 1 lands.

### Phase 1 (SO1, RQ1): label-efficient crop segmentation

A fit is a segmentation training run: frozen encoder, cached multi-scale feature maps, trainable
decoder, per-pixel logits, cross-entropy with `ignore_index`. Because the encoder is frozen its output
for a given chip is constant, so feature maps are cached once per `(model, chip)` and decoder training
reads from cache. This is an implementation optimisation and not a change of method. `albumentations.D4`
augmentation is not free under caching, because ViTs are not exactly equivariant, so the eight D4
variants are cached rather than approximated in feature space.

Raw-feature baseline in two per-pixel variants: TIMESAT phenometrics (NDVI and EVI peak value and
day-of-year, length of season, sowing and harvest day-of-year, per-band mean and standard deviation)
and the flattened monthly stack, which is the T = 12 chip itself.

Protocol: the 15 to 20 most frequent crops per country; a label-budget grid K counted in polygons per
class; multiple random draws per K; the same fixed held-out test blocks at every K; Macro-F1 with 95 %
bootstrap confidence intervals; spatial block cross-validation (Roberts et al., 2017). Run in-country
within each country, then repeated as cross-country transfer on the intersection of each pair of
countries' top classes.

**Open, to be settled in the design sections still outstanding:** the revised K grid and fit budget
under per-cell GPU training; chip-to-block assignment and the buffer width; the decoder family fixed
per resolution group; HCAT depth and the class eligibility filter under segmentation; the transfer
protocol; and how pixel-level XAI is aggregated for Phase 3.

### Phase 2 (SO2, RQ2.1 to RQ2.3): explainability and uncertainty

Not active. Pixel-level segmentation makes Integrated Gradients, Occlusion and AttnLRP natively
per-pixel, per-band and per-date, which is a stronger basis for RQ2.1 than attributing a pooled vector.
The **embedding tier** applies to all four models and the **backbone tier** to TerraMind and THOR only.
Uncertainty uses Monte Carlo Dropout as primary and Deep Ensembles as the heavier comparison, both at
decoder level, reporting per-pixel predictive entropy with reliability diagrams and Expected
Calibration Error. RQ2.3 is answered by placing the GFM importances, after binding to agronomic
concepts, on the same axes as the baseline importances.

### Phase 3 (SO3, RQ3): XAI-grounded LLM reporting

Not active. A structured JSON context document per target zone with three blocks: predictions,
explanations, and external context (ERA5 weather variables, geographic descriptors including country,
elevation and soil class from SoilGrids, and image-texture descriptors where available). Select an LLM
that follows the required output format reliably and is reproducible, generate at temperature zero,
constrain it to the context document, and score each report for traceability against its source.
Pixel-level attribution cannot be passed to an LLM directly, so the aggregation from pixels to zone
summaries is a Phase 3 design problem that Phase 1 must not foreclose.

## Priority under time pressure

Phases 1 and 2 are core deliverables. Phase 3 is highly valuable but is the first to be reduced if the
schedule slips. Phase 3 also degrades gracefully: if explanations fail their faithfulness checks, that
is itself a reportable Phase 2 finding, and the context document can still operate from predictions and
calibrated uncertainty alone. Within Phase 1, Portugal is the first thing cut, then the cross-country
transfer sweep, then the decoder comparison across resolution groups.

## Compute

ITC and UTwente institutional GPU cluster, with project drive storage. This is what makes per-cell
decoder training and backbone-tier attribution feasible at full scope.

## Working environments

Two machines carry this repository and they differ enough that a command written for one fails on the
other. Establish which one you are on before running anything.

**Windows laptop.** The machine the README documents. The repository sits inside OneDrive, the GPU is an
NVIDIA RTX PRO 1000 Blackwell at compute capability 12.0, and both PowerShell and bash are available.

**ITC JupyterHub, Linux.** The machine described in the rest of this section. Bash only, no PowerShell.

| Item | Value |
|---|---|
| Repository | `/data/private/THESIS - ExplainedGMF4Agri`. `/home/jovyan/private` is a symlink to `/data/private`, so the two paths are the same directory |
| Interpreter | Python 3.11.15 from pyenv, at `~/.pyenv/versions/3.11.15/bin/python`. The system `python3` is 3.8 and is not usable here |
| Poetry | 2.2.1, reachable through `~/.local/bin`, which a non-interactive shell has to export itself |
| Environment | `~/.cache/pypoetry/virtualenvs/gfm4agri-7U2rqC_9-py3.11`, outside the repository because `poetry.toml` sets `in-project = false`. Locate it with `poetry env info --path` |
| GPU | NVIDIA RTX A4000, 16 GB, compute capability 8.6, driver 550.54.14, CUDA 12.4 |
| Jupyter kernel | `Python 3.11 (gfm4agri)`, registered under the name `gfm4agri` |
| Data | `data/eurocropsml/` and `data/eurocrops/` are present locally. Neither `EUROCROPSML_DATA` nor `CROPHARVEST_DATA` is exported, so the code takes its in repository fallback |

Run everything through `poetry run`, or activate `$(poetry env info --path)/bin/activate`. After pulling a
commit that touches `pyproject.toml` or `poetry.lock`, run `poetry install` again.

### Traps specific to the JupyterHub machine

**Poetry must be version 2.** `pyproject.toml` declares its metadata in the PEP 621 `[project]` table and
`poetry.lock` is lock version 2.1. Poetry 1.8 reads neither, and the 1.8 that ships in this image runs on
Python 3.8, so it cannot update itself. It was replaced using the official installer driven by the pyenv
interpreter rather than the system one:

```bash
curl -sSL https://install.python-poetry.org | ~/.pyenv/versions/3.11.15/bin/python - --version 2.2.1
```

**The image exports a `PYTHONPATH` that shadows the environment.** JupyterHub sets `PYTHONPATH` to Spark's
Python 3.8 trees for every process, and those directories land ahead of the environment's own
`site-packages`. Any package present in both is then imported from the 3.8 build. `cv2`, `osgeo`, `vtk`,
`itk` and `mpi4py` all collide this way, and `cv2` alone breaks `import terratorch` with an OpenCV loader
error that names nothing of the real cause. The fix is a `sitecustomize.py` inside the environment, which
runs at interpreter start, after `site` has finished, and drops those entries. It is scoped to this
environment, so PySpark elsewhere is untouched, and it also covers the Jupyter kernel, which inherits the
same `PYTHONPATH`. It lives in the environment, so **recreating the environment loses it and the failure
returns.** Recreate it with:

```bash
cat > "$(poetry env info --path)/lib/python3.11/site-packages/sitecustomize.py" <<'PY'
import sys

_FOREIGN = ("/opt/spark/python", "python3.8", "/usr/local/lib/python3/dist-packages")
sys.path[:] = [p for p in sys.path if not any(marker in p for marker in _FOREIGN)]
PY
```

`poetry run python -c "import terratorch"` is the check. It fails without that file and succeeds with it.

**The CUDA wheels are newer than the driver.** torch is pinned to the cu128 index for the Blackwell
laptop, while this driver reports CUDA 12.4. The cu128 build nonetheless runs correctly on the A4000
through CUDA minor version compatibility, confirmed with a GPU matmul, so do not substitute a cu124 build
or the CPU build here. Verify with:

```bash
poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**Imports are slow because the environment is on NFS.** The home filesystem is a network mount, so the
first `import terratorch` after an install took 8 minutes 46 seconds against 55 seconds of CPU time,
nearly all of it waiting on I/O while bytecode was written. Warm imports settle at about 30 seconds. A
long first import is not a hung process. Container local disk would be faster but does not survive a
restart, which is why the environment stays in the home cache.

### Rebuilding the environment on the JupyterHub machine

```bash
export PATH="$HOME/.local/bin:$PATH"
poetry env use ~/.pyenv/versions/3.11.15/bin/python
poetry install
# recreate sitecustomize.py at this point, see above
poetry run python -m ipykernel install --user --name gfm4agri --display-name "Python 3.11 (gfm4agri)"
poetry run pytest
```

## Repository layout

```
docs/phase1/        the Phase 1 protocol and the record of decisions taken
docs/research/      research notes and deep-research documents
docs/internship/    the separate grazing-versus-mowing internship at Terramind, Sep to Dec 2026
src/gfm4agri/       the pipeline package, subpackages mapping onto the work packages
  data/             EuroCrops polygon loading, chip tiling, label rasterisation, splits, spatial blocking
  chips/            Sentinel-2 export and monthly compositing onto the fixed T = 12 grid
  embeddings/       frozen encoder feature caching (TerraTorch backbones, AlphaEarth, TESSERA)
  baselines/        per-pixel TIMESAT phenometrics and monthly-stack raw-feature baselines
  benchmark/        K-shot protocol, decoders, metrics, learning curves, transfer experiments
  xai/              embedding-tier and backbone-tier explainability
  uncertainty/      MC Dropout, Deep Ensembles, calibration
  reporting/        context-document assembly and LLM reporting
configs/            experiment configuration
notebooks/          exploratory notebooks, including the TerraTorch rig notebooks
scripts/legacy/     precursor AlphaEarth and TESSERA scripts from the ML-Embeddings project
data/               datasets, git-ignored
results/            experiment outputs, git-ignored apart from results/eda
figures/            scripts producing thesis and presentation figures
```

## Conventions

- Working language is English. The user is a Spanish and English bilingual MSc student at ITC, comfortable with EO, ML and GIS terminology, so prefer concise technical responses.
- Two working environments, the Windows laptop and the ITC JupyterHub Linux machine, described under
  Working environments. Confirm which one you are on before writing a command, since PowerShell exists
  only on the laptop and the JupyterHub machine needs the handling documented there.
- Python, with PyTorch, scikit-learn, TerraTorch, TorchGeo, rasterio and GDAL, and Earth Engine.
- Every experiment must be reproducible: fixed seeds, the configuration recorded alongside the results, and the label budget, the country and the split protocol stated in every result file.
- In prose drafted for the thesis, avoid dashes and use a formal academic register.
- Do not commit data or large binaries. `data/` and `results/` are git-ignored, apart from `results/eda`.
