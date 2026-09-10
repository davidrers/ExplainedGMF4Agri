# ExplainedGFM4Agri

Working repository for the MSc thesis *Geospatial Foundation Models for Transparent Agricultural Monitoring under Label-Scarce Conditions* (David Reyes, ITC, University of Twente, 2026-2027).

The research design, objectives, research questions and work plan are specified in
[docs/proposal/current_proposal.md](docs/proposal/current_proposal.md). A condensed working summary, intended for
day-to-day orientation, is in [CLAUDE.md](CLAUDE.md).

## Layout

| Path | Contents |
|---|---|
| `docs/proposal/` | The proposal, authoritative for the research design |
| `docs/research/` | Research notes and deep-research documents from the proposal phase |
| `docs/internship/` | The separate Terramind internship proposal, September to December 2026 |
| `src/gfm4agri/` | The pipeline package |
| `configs/` | Experiment configuration |
| `notebooks/` | Exploratory notebooks |
| `scripts/legacy/` | Precursor AlphaEarth and TESSERA scripts carried over from the ML-Embeddings project |
| `data/` | Datasets, not tracked in git. See [data/README.md](data/README.md) |
| `results/eda/` | EuroCropsML exploratory analysis carried over from the proposal phase |
| `figures/` | Scripts producing thesis and presentation figures |

## Environment

Not yet pinned. The intended stack is Python with PyTorch, scikit-learn, TerraTorch, TorchGeo, geopandas,
rasterio and the Earth Engine Python API. An environment specification will be added in WP1.

## Provenance

The proposal-phase folder
`Documents\Students - MSc and PhD-26-David - MSc David Reyes - Explainable Foundation Models & LLM-Powered Agricultural Reporting`
is retained unchanged as an archive of the proposal phase. It holds the defence deck, assessment material, templates
and drafts that are not needed for the implementation work.
