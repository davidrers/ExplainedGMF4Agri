# ExplainedGFM4Agri: working guide

MSc thesis of David Reyes, ITC, University of Twente, 2026-2027: frozen geospatial foundation models (GFMs) with
trainable decoders for pixel-level crop-type segmentation under label scarcity, followed by explainability,
uncertainty and LLM reporting. **Phase 1, the label-budget curves of RQ1, is the only active workstream.**

## Where things are written down

| Document | Holds |
|---|---|
| [docs/thesis_design.md](docs/thesis_design.md) | Research intent: objectives, RQs, task formulation, datasets, model set, phases, priorities. Read before any design decision |
| [docs/phase1/protocol.md](docs/phase1/protocol.md) | Evaluation mechanism: blocking, partitions, budgets, metrics, transfer |
| [docs/phase1/pipeline.md](docs/phase1/pipeline.md) | **The implemented pipeline**, stage by stage, with the reason for each decision, the faults found and fixed, and the known gaps (section 10). Read before touching the code |
| [docs/phase1/pipeline_overview.md](docs/phase1/pipeline_overview.md) | The same pipeline in plain language, with diagrams |
| [docs/phase1/proposal_deltas.md](docs/phase1/proposal_deltas.md) | Where the protocol departs from the proposal and from the research design |

Precedence: `thesis_design.md` states the intent and `protocol.md` the mechanism; when they disagree the protocol
is corrected to match.

### Documentation changes only by agreement

Change the documentation (this file, `thesis_design.md`, `protocol.md`, `pipeline.md`, `pipeline_overview.md`)
**only when the user has agreed to that change**. Do not edit it while running experiments or explorations. When
work produces something the docs should record, say so at the end and propose the edit; make it once agreed.

When an agreed update is made, `pipeline.md` is where it goes:

- a new or changed stage, data product, script or config: its section, and the stage table of section 1;
- a fault found and fixed: where the stage is described, say what went wrong, how it showed and what was discarded;
- a departure from `protocol.md` or a deferred decision: section 10, Known gaps;
- a finished run or grid: section 11, Runs, and the Current state below.

## Current state, 7 October 2026

The chain runs end to end on **all of Estonia 2021**: EuroCrops v11 polygons, a fixed 2,240 m chip grid in
EPSG:3035, dense mask and parcel-id rasters, twelve monthly Sentinel-2 L2A composites, a spatial block split, a
per-class percentage label budget applied at load time, a frozen encoder with a trainable decoder, and a
`results.json` per fit.

| Item | Value |
|---|---|
| Chip set | `data/eurocrops_chips/EE_2021/`: 7,402 chips of 224 x 224 at 10 m, 20 classes. Per chip: `_merged.tif` (S2, 144 bands), `_s1rtc.tif` (S1 RTC, 24 bands), `_tessera.tif` (128-d), `.mask.tif`, `.parcels.tif` |
| Split | `splits/blocks4_buf1600_seed0__6b0eb4cb`: blocks of 4 x 4 chips, 1,600 m parcel buffer; 4,898 train, 769 validation, 1,475 test chips |
| Feature caches | `data/embeddings/<backbone>/EE_2021/`: TerraMind v1 large (about 0.8 TB) and Prithvi-EO-2.0 600M TL (about 1.3 TB) |
| Pilot | `data/eurocrops_chips/EE_2021_pilot/`: 12 chips, no test partition, for wiring and debugging only |

Arms implemented: **Prithvi-EO-2.0 600M TL** and **TerraMind v1 large** (token grid; `ChannelBottleneck` plus a
12.9 M parameter UNet decoder; two-stage fit from cached features) and **TESSERA v1** (10 m pixel raster; per-pixel
MLP; end to end). The small variants and TerraMind on S2 plus S1 have run on the pilot only.

Full-Estonia grid, finished 2 October 2026. Test Macro-F1 over the 1,475 test chips, one draw and one seed per
cell, 15 epochs, checkpoint of lowest validation loss. Details in pipeline.md section 11,
[`results/seg_cached/ee_grid_summary.csv`](results/seg_cached/ee_grid_summary.csv) and
[`notebooks/terratorch/eurocrops_ee_results.ipynb`](notebooks/terratorch/eurocrops_ee_results.ipynb).

| K, % of parcels per class | TESSERA + MLP | TerraMind v1 large | Prithvi-EO-2.0 600M TL |
|---|---|---|---|
| 100 | 0.631 | 0.586 | 0.573 |
| 20 | 0.611 | 0.479 | 0.496 |
| 5 | 0.568 | 0.362 | 0.368 |

This does not answer RQ1 yet: TESSERA sits in another resolution group, no cell has an interval, and no
baseline has run.

**Not built:** the baseline arm; THOR and AlphaEarth; repeated, nested draws with bootstrap intervals; the stage 1
cropland mask; Latvia and Portugal; Phases 2 and 3.

**Decided on 2 October 2026, not yet in `protocol.md` or the code.** These live only in
[`figures/phase1_experimental_setup_prompt.md`](figures/phase1_experimental_setup_prompt.md) and session memory,
so confirm them with the user and write them into the protocol before building on them:
THOR joins the arms, handled like TerraMind; the baseline becomes a plain U-Net trained from scratch on the same
monthly Sentinel-2 chips, replacing the per-pixel TIMESAT and monthly-stack baselines in the K sweep, with the role
of the phenometrics still open; K grid of 1, 5, 10, 20, 50 and 100 %; nested repeated draws, with models compared on
the same draws; equal tuning trials and training steps for every model and K; no validation labels at low K;
cross-country transfer only within one GAEZ v5 agro-ecological zone. AlphaEarth is not confirmed for this set-up.

**Open choices that move the numbers:** the checkpoint criterion (validation loss picks epoch 6 to 9 for the
token-grid models while validation Macro-F1 peaks at 13), the treatment of the 0.66 % of test pixels without a
TESSERA embedding (currently zeros), and the block size.

## Running the pipeline

From the repository root on the JupyterHub machine. Every stage resumes from what is already on disk.

```bash
poetry run python scripts/data/build_country_chips.py --country EE --year 2021 --workers 12   # S2 chips, masks, manifest
poetry run python scripts/data/build_s1_chips.py --country EE --year 2021 --workers 12        # S1 RTC beside them
poetry run python scripts/data/build_tessera_chips.py --root data/eurocrops_chips/EE_2021 --workers 16
poetry run python scripts/data/build_chip_split.py --root data/eurocrops_chips/EE_2021       # block split, chip_parcels.parquet
poetry run python scripts/seg/encode.py -c configs/seg/terramind_v1_large_ee.yaml            # stage 1: feature cache
poetry run python scripts/seg/fit_cached.py -c configs/seg/terramind_v1_large_ee.yaml --pct 5  # stage 2: decoder
poetry run python scripts/seg/train.py -c configs/seg/tessera_v1_mlp_ee.yaml --set label_budget.mode=pct label_budget.pct=5
EPOCHS=15 setsid nohup bash scripts/seg/run_ee_grid.sh > results/seg_cached/ee_grid.log 2>&1 &   # the full grid
```

Configs live in `configs/seg/`, one per run: `*_ee` for full Estonia, `*_ee_pilot` for the pilot. End-to-end fits
write to `results/seg/<run>/P<pct>_draw<d>_seed<s>/`, two-stage fits to `results/seg_cached/...`.

The chip set took days of network time and the caches are terabytes. Never delete, overwrite or re-export them
without asking. An export skips chips already on disk, so a faulty set must be removed before a corrected export
runs into the same directory.

## Working environments

Two machines carry the repository. Establish which one you are on before running anything.

**Windows laptop.** Repository inside OneDrive, RTX PRO 1000 Blackwell (compute capability 12.0), PowerShell and
bash. Set-up in [README.md](README.md).

**ITC JupyterHub, Linux.** Where the Estonian data and runs live. Bash only.

| Item | Value |
|---|---|
| Repository | `/data/private/THESIS - ExplainedGMF4Agri`; `/home/jovyan/private` is a symlink to `/data/private` |
| Interpreter | Python 3.11.15 at `~/.pyenv/versions/3.11.15/bin/python`. The system `python3` is 3.8 and unusable |
| Poetry | 2.2.1 in `~/.local/bin`, which a non-interactive shell must add to `PATH` |
| Environment | `~/.cache/pypoetry/virtualenvs/gfm4agri-7U2rqC_9-py3.11` (`poetry env info --path`); Jupyter kernel `gfm4agri` |
| GPU | RTX A4000, 16 GB, compute capability 8.6, driver CUDA 12.4 |
| Storage | `/data/private` is a 115 TB NFS project drive; local disk is about 335 GB and does not survive a restart |

Run everything through `poetry run`, and run `poetry install` again after a pull touching `pyproject.toml` or
`poetry.lock`. Traps on this machine:

- **Poetry must be 2.x**, because of the PEP 621 metadata and lock version 2.1. The image's 1.8 runs on Python 3.8
  and cannot update itself: `curl -sSL https://install.python-poetry.org | ~/.pyenv/versions/3.11.15/bin/python - --version 2.2.1`.
- **The image's `PYTHONPATH` shadows the environment** with Spark's Python 3.8 trees (`cv2`, `osgeo`, `vtk`, `itk`,
  `mpi4py`). `cv2` alone breaks `import terratorch` with an OpenCV error that names nothing of the cause. A
  `sitecustomize.py` in the environment strips those entries, and **recreating the environment loses it**:

  ```bash
  cat > "$(poetry env info --path)/lib/python3.11/site-packages/sitecustomize.py" <<'PY'
  import sys

  _FOREIGN = ("/opt/spark/python", "python3.8", "/usr/local/lib/python3/dist-packages")
  sys.path[:] = [p for p in sys.path if not any(marker in p for marker in _FOREIGN)]
  PY
  ```

  Check with `poetry run python -c "import terratorch"`.
- **torch is the cu128 build on a CUDA 12.4 driver.** It works through minor version compatibility; do not swap in
  a cu124 or CPU build. Check with
  `poetry run python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.
- **Imports are slow on NFS.** The first `import terratorch` after an install took almost 9 minutes; warm imports
  take about 30 s. That is not a hung process.
- **Long jobs** run detached, `setsid nohup ... > log 2>&1 &`, as the `scripts/seg/run_*.sh` headers show.

Rebuilding the environment: `export PATH="$HOME/.local/bin:$PATH"`, `poetry env use ~/.pyenv/versions/3.11.15/bin/python`,
`poetry install`, recreate `sitecustomize.py`, then
`poetry run python -m ipykernel install --user --name gfm4agri --display-name "Python 3.11 (gfm4agri)"` and `poetry run pytest`.

## Repository layout

```
docs/thesis_design.md      research design, the former CLAUDE.md
docs/phase1/               protocol, pipeline, plain-language overview, proposal deltas
docs/proposal/, docs/research/, docs/internship/   proposal, research notes, the separate Terramind internship
src/gfm4agri/data/         EuroCrops loading, class scheme, chip grid and label rasters, spatial blocks, chip split
src/gfm4agri/chips/        Sentinel-2 and Sentinel-1 monthly compositing
src/gfm4agri/embeddings/   TESSERA Zarr reader
src/gfm4agri/benchmark/    backbone registry, necks, decoders, datamodule and budget draw, two-stage fit, reload and predict
src/gfm4agri/{baselines,xai,uncertainty,reporting}/   empty placeholders for later work
scripts/data/              vector fetch; chip, S1, TESSERA and split builders
scripts/seg/               train.py, encode.py, fit_cached.py and the run_*.sh grids
configs/                   class scheme; configs/seg/ one YAML per run
notebooks/                 EDA notebooks 01 to 06; terratorch/ per-model pilot notebooks and the full-Estonia results
data/, results/            git-ignored, apart from results/eda
figures/                   figure scripts and diagram prompts
tests/                     pytest suite
```

Notebooks with a `_build_*.py` beside them are generated: edit the builder and regenerate, as its docstring shows.

## Conventions

- Working language is English. The user is a bilingual Spanish and English MSc student comfortable with EO, ML and
  GIS terminology, so keep responses concise and technical.
- Every experiment is reproducible: fixed seeds, and the configuration, label budget, country and split protocol
  recorded in every result file. `results.json` stamps the chip manifest's SHA-256.
- Thesis prose avoids dashes and uses a formal academic register.
- Do not commit data or large binaries.
