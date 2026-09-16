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

Python 3.11 managed with [Poetry](https://python-poetry.org/). The dependency set is declared in
`pyproject.toml` and frozen in `poetry.lock`; both are tracked in git, so any machine resolves to
exactly the same versions. The virtual environment itself is not tracked.

### Install

```powershell
poetry env use 3.11
poetry install
```

This installs the project package `gfm4agri` in editable mode, so `import gfm4agri` works from
anywhere without manipulating `sys.path`.

To also run the upstream TerraTorch tutorials in `notebooks/terratorch/`:

```powershell
poetry install --with tutorials
```

### GPU and CPU builds

`torch` and `torchvision` are pinned to the PyTorch **cu128** index rather than PyPI, because the
development laptop carries an NVIDIA RTX PRO 1000 Blackwell (compute capability 12.0), which requires
CUDA 12.8 or newer, and because the PyPI wheel for Windows is CPU-only.

On a machine **without** an NVIDIA GPU, `poetry install` will fail on the CUDA wheels. Install the
rest of the stack first and then replace torch with the CPU build:

```powershell
poetry install --no-root
poetry run pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
poetry install
```

Verify which build is active:

```powershell
poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

### Virtual environment location

The repository sits inside a OneDrive folder, so the environment is deliberately created outside it
(in Poetry's cache directory) to avoid synchronising several gigabytes of CUDA libraries. This is set
per project in `poetry.toml`. Run `poetry env info --path` to locate it, and select that interpreter
in VS Code.

## Provenance

The proposal-phase folder
`Documents\Students - MSc and PhD-26-David - MSc David Reyes - Explainable Foundation Models & LLM-Powered Agricultural Reporting`
is retained unchanged as an archive of the proposal phase. It holds the defence deck, assessment material, templates
and drafts that are not needed for the implementation work.
