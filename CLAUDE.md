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
| [docs/phase1/results_estonia/REPORT.md](docs/phase1/results_estonia/REPORT.md) | First results for Estonia 2021: every arm frozen and fine-tuned at 45 epochs, per pixel and per parcel, each model's configuration in plain language, and the published Estonian results beside ours |

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

## Current state, 22 October 2026

The chain runs end to end on **all of Estonia 2021**: EuroCrops v11 polygons, a fixed 2,240 m chip grid in
EPSG:3035, dense mask and parcel-id rasters, twelve monthly Sentinel-2 L2A composites, a spatial block split, a
per-class percentage label budget applied at load time, a frozen encoder with a trainable decoder, and a
`results.json` per fit, scored per pixel and per parcel.

| Item | Value |
|---|---|
| Chip set | **`data/eurocrops_chips/EE_2021_rev/`**, the main workflow's since 13 October 2026: 7,402 chips of 224 x 224 at 10 m, 20 classes, Sentinel-2 composites under the `revised` screening (pipeline.md section 5). Per chip: `_merged.tif` (S2, 144 bands), `.mask.tif`, `.parcels.tif`, and as links into `EE_2021`: `_s1rtc.tif` (S1 RTC, 24 bands), `_tessera.tif` (128-d), `_alphaearth.tif` (64-d). `EE_2021/` is the first export, under the `export_2021` screening; it stays as the parent of those links, and every result before 13 October 2026 was fitted on it |
| Split | `splits/blocks4_buf1600_seed0__6b0eb4cb`: blocks of 4 x 4 chips, 1,600 m parcel buffer; 4,898 train, 769 validation, 1,475 test chips |
| Feature caches | None persistent for `EE_2021_rev`: the cluster computes each job's caches on the node and deletes them after. On the hub, `data/embeddings/<backbone>/EE_2021_rev_mini/` for the four encoders of the current arms, TESSERA v2 included. Of `EE_2021`: `data/embeddings/<backbone>/EE_2021/`, TerraMind v1 large (about 0.8 TB) and Prithvi-EO-2.0 600M TL (about 1.3 TB), and `EE_2021_mini` for the three token-grid arms (about 25 GB) |
| Pilot | **`data/eurocrops_chips/EE_2021_rev_mini/`**: whole blocks of `EE_2021_rev` as links, 32 training, 16 validation, 9 test chips, same split and the same blocks as `EE_2021_mini` (its counterpart on `EE_2021`); `subset.json` records the blocks. The old 12-chip `EE_2021_pilot/` is archived with `experiments/2026-09_pilot_12chips/` |
| Cluster | The UT HPC clone holds `EE_2021`, `EE_2021_rev`, `EE_2021_rev_mini`, `EE_2021_mini`, the sample sets and `data/eurocrops/{parquet,vector}`, no caches |

The main workflow's arms since 15 October 2026 (`configs/experiments/kshot.yaml`, one file each in `configs/arms/`);
each model consumes every sensor of the chip set it was pretrained on (protocol D24), and the baseline every sensor
any of them sees:

| Arm | Input | Group and decoder |
|---|---|---|
| `terramind_v1_large_s2s1` | S2 + S1, modality tokens averaged | token grid, 160 m; `ChannelBottleneck` + 12.9 M UNet, 53.3 M trainable; from cached features |
| `thor_v1_large_s2s1` | S2 + S1, its three band groups averaged | as TerraMind, exactly 53,281,492 trainable |
| `prithvi_eo_v2_600_tl` | S2 only, all it was pretrained on | token grid; same decoder stack |
| `tessera_v2_medium` | S2 + S1, frozen v2 Medium from open weights, run on the composites and cached by the runner (`pipeline/raster_cache.py`) | 10 m raster; per-pixel MLP, 0.2 M |
| `alphaearth_v1` | its precomputed 64-d embedding | 10 m raster; per-pixel MLP |
| `unet_r18_scratch_s2s1` | S2 + S1, months stacked on 168 channels | **the baseline** (protocol D26): a ResNet-18 U-Net trained from scratch, everything trainable, end to end; group `scratch` |

The 10 m group keeps the per-pixel MLP: a U-Net did not beat it under label scarcity
(`experiments/2026-10-15_tessera_decoder`). TESSERA v1 from the store is retired from `kshot`: its rasters sit about
0.7 px east and 0.4 px south of the chip grid (protocol D25). The earlier arms (`terramind_v1_large`, `thor_v1_large`,
`tessera_v1`) keep their files for earlier results and experiments. All five current arms ran on `EE_2021_rev_mini`
on the hub and the cluster on 15 October 2026, and on all of `EE_2021_rev` at 15 epochs the same day
(`results/kshot_15ep/EE_2021_rev/`, pipeline.md section 11). **Since then every arm trains for 45 epochs (protocol
D27); the six arms, the baseline among them, ran on all of `EE_2021_rev` at 45 epochs on 16 October 2026
(`results/kshot/EE_2021_rev/`)**. **Every foundation-model encoder stays frozen in the main workflow; fine-tuning
is an experiment** (protocol D29, user 22 October 2026): the four encoders with weights ran fine-tuned in
`experiments/2026-10-15_kshot_finetune`. **THOR v1 large on 80 m tokens** is an experiment, not an arm of the main
workflow (`experiments/2026-10-08_thor_80m/`, with its own arm file): its full-Estonia cache, about 6.4 TB, fits
no cluster node; it has run on `EE_2021_mini` only. The small variants and TerraMind on S2 plus S1 are
archived studies in `experiments/`.

**Since 16 October 2026 every fit is also scored per parcel** (protocol D28): each test parcel takes the class most
of its pixels were predicted as, and parcel Macro-F1, accuracy and kappa sit beside the pixel metrics in
`results.json` (`parcel_metrics`) and `summary.csv`; the pixel Macro-F1 stays the primary metric. On the 15-epoch
grid the parcel Macro-F1 is 0.002 to 0.078 below the pixel one, most for the token-grid models, and TESSERA leads
per parcel at every budget (`experiments/2026-10-16_parcel_metric`). `FitPredictor` loaded `best-loss.ckpt` by
name, so a cell refitted after an interrupted fit was predicted with the stale checkpoint; it now loads the one
`results.json` names, and `summary.csv`'s `predictions_match` flags stored predictions that miss the logged score.

The main workflow's full-Estonia grid at 45 epochs, finished 16 October 2026. Test Macro-F1 over the 1,475 test
chips, per pixel / per parcel, one draw and one seed per cell, checkpoint of lowest validation loss. Details in
pipeline.md section 11 and [docs/phase1/results_estonia/REPORT.md](docs/phase1/results_estonia/REPORT.md).

| Arm | K = 100 % | 20 % | 5 % |
|---|---|---|---|
| TerraMind v1 large, S2 + S1 | 0.661 / 0.583 | 0.566 / 0.515 | 0.433 / 0.381 |
| THOR v1 large, S2 + S1 | 0.594 / 0.516 | 0.547 / 0.491 | 0.393 / 0.349 |
| Prithvi-EO-2.0 600M TL | 0.580 / 0.535 | 0.505 / 0.472 | 0.445 / 0.409 |
| TESSERA v2 Medium + MLP | 0.618 / 0.602 | 0.582 / 0.574 | 0.504 / 0.502 |
| AlphaEarth v1 + MLP | 0.523 / 0.484 | 0.497 / 0.455 | 0.449 / 0.425 |
| U-Net from scratch (baseline) | 0.616 / 0.553 | 0.564 / 0.520 | 0.467 / 0.446 |

The token-grid arms equal their 15-epoch cells to every digit, their checkpoint falling at epoch 7 to 14. Frozen
TESSERA v2 scores 0.005, 0.017 and 0.068 below the retired v1.0 store arm per pixel; the switch holds for the whole
pipeline (protocol D25, user 22 October 2026). In the fine-tuning experiment TESSERA v2 reaches 0.660 / 0.639,
0.630 / 0.603 and 0.581 / 0.566, 0.04 to 0.08 above frozen, while TerraMind, THOR and Prithvi gain about 0.01 in
validation, their test score swinging either way with the checkpoint. This does not answer RQ1 yet: no cell has an
interval, and the checkpoint criterion decides the token-grid arms. pipeline.md section 11 keeps the runs of the
main workflow, the experiment runs its decisions rest on, and the earlier runs (the grid of 2 October 2026 on
`EE_2021` among them) apart.

**Not built:** repeated, nested draws with bootstrap intervals; the stage 1 cropland mask; Latvia
and Portugal; Phases 2 and 3.

**Decided on 2 October 2026, not yet in `protocol.md` or the code.** These live only in
[`figures/phase1_experimental_setup_prompt.md`](figures/phase1_experimental_setup_prompt.md) and session memory, so
confirm them with the user and write them into the protocol before building on them: THOR joins the arms, handled
like TerraMind (now implemented, on 160 m tokens as confirmed on 9 October); the baseline becomes a plain U-Net
trained from scratch, replacing the per-pixel TIMESAT and monthly-stack baselines in the K sweep (now protocol
D26 and the arm `unet_r18_scratch_s2s1`, on S2 + S1), with the role of the phenometrics still open; K grid of 1, 5, 10, 20, 50 and 100 %
(protocol D2; `kshot` runs the working subset 5, 20 and 100); nested repeated draws (protocol D8, not implemented),
with models compared on the same draws; equal tuning trials and training steps for every model and
K; no validation labels at low K; cross-country transfer only within one GAEZ v5 agro-ecological zone. AlphaEarth
was confirmed on 7 October and is implemented, handled like TESSERA.

**Open choices that move the numbers:** how `K*` is read against a baseline outside both resolution groups
(protocol open item 11); the checkpoint criterion (at 45 epochs validation loss picks epoch 7 to 14
for the frozen token-grid models and 5 to 9 for the fine-tuned ones, while their validation Macro-F1 peaks at
epochs 13 to 44), and the block size. THOR's group averaging is untested
against the concatenation its authors use (pipeline.md section 10).

## Running the pipeline

From the repository root. Every stage resumes from what is already on disk.

**Chip extraction, on the JupyterHub only** (`scripts/hub/`):

```bash
poetry run python scripts/hub/build_country_chips.py --country LV --year 2021 --workers 8 --group-size 2   # S2 chips, masks, manifest; --screening revised by default
poetry run python scripts/hub/build_s1_chips.py --country EE --year 2021 --workers 12        # S1 RTC beside them
poetry run python scripts/hub/build_tessera_chips.py --root data/eurocrops_chips/EE_2021 --workers 16   # the v1.0 store, retired from kshot
poetry run python scripts/hub/build_chip_split.py --root data/eurocrops_chips/EE_2021       # block split, chip_parcels.parquet
poetry run python scripts/hub/fetch_alphaearth_tiles.py --root data/eurocrops_chips/EE_2021 --workers 4
poetry run python scripts/hub/build_alphaearth_chips.py --root data/eurocrops_chips/EE_2021 --workers 16 \
    --split-dir data/eurocrops_chips/EE_2021/splits/blocks4_buf1600_seed0__6b0eb4cb    # statistics on the training chips
poetry run python scripts/hub/build_pilot_chipset.py --parent data/eurocrops_chips/EE_2021_rev --name EE_2021_rev_mini
```

`EE_2021_rev` itself was derived from `EE_2021` by `experiments/2026-10-12_s2_revised_export/run.py`, which re-exports
the Sentinel-2 composites and links everything else. The export script refuses to write one screening into a chip set
made with another, so `EE_2021` cannot be overwritten by accident. The other builders above still name `EE_2021`, where
the Sentinel-1, TESSERA and AlphaEarth products live.

**The K-shot workflow, on either machine** (`scripts/run_kshot.py`): every arm, budget, draw and seed of an
experiment on one chip set. A pilot and a country differ only in `--chips`.

```bash
poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_rev --dry-run
poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_rev_mini
```

Configs: `configs/arms/<arm>.yaml` (what is trained), `configs/experiments/<name>.yaml` (arms, budgets, draws, seeds,
epochs), `configs/machines/{hub,cluster}.yaml` (throughput only). A token-grid arm reuses its cache in
`data/embeddings/<backbone>/<chip set>/` when `cache.json` matches the run, and computes it otherwise. Results go to
`results/<experiment>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/` (`results.json` with the pixel metrics and
`parcel_metrics`, `predictions_test.npz`) and `summary.csv`; runs before 9 October stay in `results/seg/` and
`results/seg_cached/`. `run_kshot.py ... --summarise` scores per parcel any finished cell that has no parcel scores
yet, without a refit, and refreshes `summary.csv`.

### Two instructions: run an experiment, or add to the main workflow

The user gives one of two instructions, and they are kept apart:

* **"Run an experiment of X"** (on the hub or the cluster). **The main workflow is not touched**: `src/`,
  `configs/{arms,experiments,machines}`, `scripts/` and `tests/` stay as they are. Everything X needs lives in
  `experiments/<YYYY-MM-DD>_<name>/`, started by copying `experiments/_template/`: a `README.md`, an
  `experiment.yaml` (arms, budgets, draws, seeds, epochs, optional `name`), arm files in `arms/` (found before
  `configs/arms/`, so a copy there varies a core arm), a `report.py` for the figures, and, only if code must
  change, a `run.py` with copies of the changed modules swapped in (see the template). **Every experiment ends
  with a `REPORT.md`**: clear figures, the results and their analysis (user, 11 October 2026). The whole run
  below, commits and pushes to `main` included, is authorised without asking at each step (user, 9 October 2026).
* **"Add X to the main workflow."** Only on that instruction: move what the experiment proved from its folder into
  the core, with tests and documentation, and describe the change before making it.

If running an experiment seems to need a change to the core, copy the file into the experiment folder instead;
if it cannot be done that way, ask.

**Running an experiment, step by step:**

1. **Create, on the hub.** Copy `experiments/_template/` to `experiments/<YYYY-MM-DD>_<name>/`; fill the README's
   question, `experiment.yaml` and any arm files; keep `run.py` only if code changes, otherwise delete it.
2. **Check, on the hub.** `poetry run pytest -q`, then a dry run on the target chip set:
   `poetry run python scripts/run_kshot.py -e experiments/<folder>/experiment.yaml --chips data/eurocrops_chips/<set> --dry-run`
   (or the folder's `run.py`). Every cell must be listed and the chip-set check must pass. Something new runs on
   `EE_2021_rev_mini` before a full country.
3. **Commit and push.** Stage the experiment folder **by name**, never `git add -A`: other sessions may be
   editing this checkout. `git commit` with the attribution line, `git push origin main`.
4. **Update the cluster.** `ssh -n utwente-hpc 'cd ~/ExplainedGMF4Agri && git pull --ff-only'`, plus
   `poetry install` if `poetry.lock` changed. Never edit or commit on the cluster. A chip set not yet there is
   copied from the hub first: `bash scripts/cluster/push_data.sh eurocrops_chips/<set>` (a pilot needs its
   parent too).
5. **Submit.** `ssh -n utwente-hpc 'cd ~/ExplainedGMF4Agri && bash scripts/cluster/submit.sh experiments/<folder>/experiment.yaml <set> [arms]'`:
   one Slurm job per arm, running the folder's `run.py` if it has one. `TIME=`, `CPUS=`, `MEM=` override 2 days,
   32 CPUs, 120 GB; `DRY_RUN=1` prints the commands. Over ssh, a command that calls `poetry` needs
   `export PATH=$HOME/.local/bin:$PATH` first (the scripts set it themselves).
6. **Wait without polling.** `itc-gpu` has four GPUs and our jobs the lowest priority, so hours of queueing
   happen (`squeue -u $USER --start`), though the 9 October pilot started within two hours. Start one background
   command that loops on `squeue -u $USER -h -n <job names>` with a sleep of a few minutes and, once the jobs are
   gone, prints `sacct -j <ids> --format=JobID,JobName%28,State,Elapsed` and the `=== ... exit` lines of
   `results/<name>/<set>/logs/*.log`. A failed job leaves its traceback in that log.
7. **Bring the results back.** `bash scripts/cluster/pull_results.sh <name> <set>`, with the experiment's `name`
   (by default its folder's name). Results already on the hub under the same `<name>/<set>` are kept apart with
   `DEST=results/_cluster`. Every `results.json` records the spec, the entry script, the arm files, the machine
   and the commit. Cells fitted by a job that started before the parcel scores existed (commit `b2b159c`, 16 October
   2026) have none: add them with
   `poetry run python scripts/run_kshot.py -e <spec> --chips data/eurocrops_chips/<set> --summarise`.
8. **Figures.** Set `ARMS` (short labels, colour slots) and `PAIRS` (arms compared fit by fit on shared draws
   and seeds) in the folder's `report.py`, adapt it if the question needs another figure, and run
   `poetry run python experiments/<folder>/report.py --chips <set>`. It writes `figures/` (K curves per pixel and
   per parcel, paired differences, validation curves with the checkpoint epochs, per-class F1) and
   `figures/tables.md`. For a quick look, `notebooks/pipeline/kshot_results.ipynb` with `EXPERIMENT` and `CHIPS`
   set still works.
9. **Report and record.** Write `REPORT.md` from the template's sections: the answer first, the setup and the
   runs, every figure with what it shows, the tables, the analysis, the caveats and the next steps. Put a
   few-line outcome with a link to it in the README, commit the folder with `figures/`, push, and give the user
   the outcome. An entry in `pipeline.md` section 11 is proposed and written only once agreed.

The main workflow itself runs the same way with its spec by name: `scripts/run_kshot.py -e
configs/experiments/kshot.yaml` and `submit.sh kshot <set>`.

The chip set took days of network time and the caches are terabytes. Never delete, overwrite or re-export them
without asking. An export skips chips already on disk, so a faulty set must be removed before a corrected export
runs into the same directory.

## Working environments

Three machines carry the repository. Establish which one you are on before running anything.

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
- **Long jobs** run detached, `setsid nohup ... > log 2>&1 &`, as the docstring of `scripts/run_kshot.py` shows.

Rebuilding the environment: `export PATH="$HOME/.local/bin:$PATH"`, `poetry env use ~/.pyenv/versions/3.11.15/bin/python`,
`poetry install`, recreate `sitecustomize.py`, then
`poetry run python -m ipykernel install --user --name gfm4agri --display-name "Python 3.11 (gfm4agri)"` and `poetry run pytest`.

**UT HPC cluster, Linux, Slurm.** Reached with `ssh utwente-hpc` from the JupyterHub. Runs the K-shot workflow only;
**code is never edited there**, every change goes hub, git, cluster.

| Item | Value |
|---|---|
| Repository | `~/ExplainedGMF4Agri`, cloned over HTTPS (the repository is public) |
| Environment | `bash -l scripts/cluster/setup_env.sh`: Python 3.11 from conda-forge via `miniconda3/25.7`, Poetry 2.2.1, the same lock, THOR, weights prefetched |
| GPU | `itc-gpu`: RTX PRO 6000 Blackwell, 96 GB, one per job, account `itc-tech` |
| Storage | 1 TB home; feature caches are job-scoped on the node's `/local` NVMe until a project directory exists |

## Repository layout

```
docs/thesis_design.md      research design, the former CLAUDE.md
docs/phase1/               protocol, pipeline, plain-language overview, proposal deltas
docs/utwente_hpc.md        the cluster, and the hub, git and cluster workflow
docs/superpowers/          design specs and implementation plans
docs/proposal/, docs/research/, docs/internship/   proposal, research notes, the separate Terramind internship
src/gfm4agri/data/         EuroCrops loading, class scheme, chip grid and label rasters, spatial blocks, chip split, pilot subset
src/gfm4agri/chips/        Sentinel-2 and Sentinel-1 monthly compositing
src/gfm4agri/embeddings/   TESSERA v2 encoder (vendored, MIT), TESSERA v1.0 Zarr reader, AlphaEarth tile reader, shared raster export
src/gfm4agri/benchmark/    backbone registry, necks, decoders, datamodule and budget draw, end-to-end and two-stage fits, reload and predict
src/gfm4agri/pipeline/     the K-shot workflow: config composition, cache lookup, TESSERA v2 raster cache, runner
src/gfm4agri/{baselines,xai,uncertainty,reporting}/   empty placeholders for later work
scripts/hub/               JupyterHub only: vector fetch; chip, S1, TESSERA, AlphaEarth, split and pilot builders
scripts/run_kshot.py       the one entry point of the fit, on either machine
scripts/cluster/           cluster environment, sbatch job, submit, data push, results pull
scripts/env/               THOR installer (outside the Poetry lock)
configs/                   class schemes; the main workflow's arms/, experiments/ (kshot.yaml) and machines/ (hub, cluster)
experiments/               one dated folder per experiment (README, experiment.yaml, arms/, optional run.py, report.py,
                           REPORT.md with figures/); _template/ to copy; an experiment never edits the core and is
                           deletable once pipeline.md records its outcome
notebooks/eda/             EDA notebooks 01 to 07, their builders, analysis/ (code whose outputs go to results/eda)
notebooks/pipeline/        cache anatomy and full-Estonia results notebooks, with builders
data/, results/            git-ignored, apart from results/eda
figures/                   figure scripts and diagram prompts
tests/                     pytest suite; tests/fixtures/legacy_configs/ holds the configs of the 2 October grid
```

Notebooks with a `_build_*.py` beside them are generated: edit the builder and regenerate, as its docstring shows.

## Conventions

- Working language is English. The user is a bilingual Spanish and English MSc student comfortable with EO, ML and
  GIS terminology, so keep responses concise and technical.
- Every experiment is reproducible: fixed seeds, and the configuration, label budget, country and split protocol
  recorded in every result file. `results.json` stamps the chip manifest's SHA-256.
- Thesis prose avoids dashes and uses a formal academic register.
- Do not commit data or large binaries.
