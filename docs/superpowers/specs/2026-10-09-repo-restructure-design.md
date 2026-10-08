# Repository restructure and the hub, git and cluster workflow

Design agreed with the user on 9 October 2026. Status: written, awaiting the user's review.

## 1. Purpose

Two goals, both serving Phase 1:

1. Separate the **core pipeline** from **experiments**, so that at any point it is clear what is where and what
   can be cleaned.
2. Make one compact workflow, the **K-shot workflow**, that takes an existing chip set and runs encoding,
   decoder fitting and evaluation for every arm, label budget, draw and seed. A pilot, Estonia or a later country
   differ only in the chip set path given to it. It runs the same way on the JupyterHub and on the UT HPC cluster,
   with git carrying the code from the hub to the cluster.

## 2. Decisions

| Decision | Choice |
|---|---|
| Machine roles | The hub develops code, extracts chips and analyses results. The cluster encodes and fits. Code is never edited on the cluster |
| Feature caches on the cluster | Job-scoped: one Slurm job per arm encodes onto the node's local NVMe, runs that arm's whole sweep, and deletes the cache on exit. The cache location is a parameter, so a project directory can replace it later without code changes |
| What a pilot is | A smaller chip set: whole spatial blocks drawn from each partition of a country chip set, written as a chip set directory of its own. The workflow cannot tell it from a country |
| Order of work | Restructure first, reproducing today's behaviour. The 2 October protocol decisions that change the numbers (nested repeated draws, equal training steps, no validation labels at low K) follow as a separate change, after they are written into `protocol.md` |
| Arms | TerraMind v1 large, Prithvi-EO-2.0 600M TL, THOR v1 large at 160 m tokens (patch 16 on 10 m bands, 8 on 20 m, to match TerraMind), TESSERA v1, AlphaEarth v1. All five already exist in the code |
| Experiment content | Only what exists: K of 100, 20 and 5 %, draw 0, seed 0, 15 epochs, the checkpoint of lowest validation loss |
| Runner approach | A thin runner over the existing fit functions (approach A). No workflow engine |

Out of scope: the baseline arm, cross-country transfer, the K grid of 1 to 100 %, nested or repeated draws, equal
training steps, the low-K validation rule, Latvia and Portugal chips, THOR's open choices other than the token
size, and the TESSERA store rewrite.

## 3. What is wrong today

- Git holds 11 commits and almost none of the working pipeline: `src/gfm4agri/benchmark/`, `chips/`, `embeddings/`,
  `scripts/data/` (apart from the two fetchers), `scripts/seg/`, `configs/seg/`, seven test files and the Phase 1
  docs are untracked.
- The pilot and the country take different routes. Pilot configs fit end to end into `results/seg/`; full-Estonia
  token-grid arms fit in two stages into `results/seg_cached/`; the per-arm route is hardcoded in
  `run_ee_grid.sh`. The 12-chip pilot has no test partition.
- One config mixes the arm, the chip set and the experiment, so each arm is duplicated per chip set, and the copies
  drift.
- Entry points share `scripts/seg/` with grid scripts and a finished ablation.
- Logs and pid files are written into `data/`, the chip directory and `results/`.
- Analysis code lives under `results/eda/`; TerraTorch tutorial scripts sit with the pilot notebooks.
- `/experiments/` is git-ignored, which contradicts tracking experiments.

## 4. Target layout

```
src/gfm4agri/
  benchmark/                 unchanged, plus fit.py: the end-to-end fit moved out of scripts/seg/train.py
  pipeline/                  new: experiment spec -> cells -> route per arm; config composition; summary
  data/ chips/ embeddings/   unchanged
  baselines/ xai/ uncertainty/ reporting/   unchanged placeholders
configs/
  arms/                      terramind_v1_large, prithvi_eo_v2_600_tl, thor_v1_large, tessera_v1, alphaearth_v1
  experiments/               kshot.yaml
  machines/                  hub.yaml, cluster.yaml
  class_scheme_*.yaml        unchanged
scripts/
  hub/                       chip extraction and the pilot builder (hub only)
  cluster/                   environment set-up, sbatch template, submit, data push, results pull
  env/install_thor.sh        unchanged, used on both machines
  run_kshot.py               the one entry point of the workflow
experiments/                 tracked; one dated folder per study
notebooks/
  eda/                       notebooks 01 to 07, their builders, and analysis/ (the code now in results/eda)
  pipeline/                  cache anatomy and full-Estonia results notebooks, with their builders
results/                     git-ignored output, apart from results/eda (unchanged)
tests/                       existing suite plus tests of the runner and the pilot builder
```

### 4.1 Moves

| From | To |
|---|---|
| `scripts/data/{fetch_eurocrops,fetch_eurocropsml,build_country_chips,build_s1_chips,build_tessera_chips,build_alphaearth_chips,fetch_alphaearth_tiles,build_chip_split}.py` | `scripts/hub/` |
| Fit logic of `scripts/seg/train.py` (`main` body, `_budget_tag`, `_labelled_fraction`) | `src/gfm4agri/benchmark/fit.py`, as `fit_end_to_end(cfg, out_dir)`, moved without behavioural change |
| `scripts/seg/encode.py` `cache_dir_for` | `src/gfm4agri/pipeline/` |
| `scripts/data/build_pilot_chips.py`, `scripts/seg/run_budget_grid.sh`, every `configs/seg/*_ee_pilot.yaml` except THOR 80 m, the per-model pilot notebooks (`eurocrops_{prithvi,terramind,tessera}_pilot.ipynb`, `eurocrops_{prithvi,terramind,thor}_pilot_embeddings.ipynb`) and `_build_embedding_workflow_notebooks.py` | `experiments/2026-09_pilot_12chips/` |
| `scripts/seg/run_d4_ablation.sh` | `experiments/2026-10-08_d4_ablation/` |
| `configs/seg/thor_v1_large_80m_ee_pilot.yaml` | `experiments/2026-10-08_thor_80m/` |
| `notebooks/0[1-7]_*.ipynb`, `results/eda/**/_build_notebook_*.py` | `notebooks/eda/` |
| `results/eda/**/_*.py` (analysis code, not builders) | `notebooks/eda/analysis/`, keeping the `cropharvest/` and `eurocrops/` subfolders; path constants adjusted so outputs still land in `results/eda/` |
| `notebooks/terratorch/{eurocrops_cache_anatomy,eurocrops_ee_results}.ipynb`, their builders, `architecture-diagram.png` | `notebooks/pipeline/` |

Moves of tracked files use `git mv`. The 12-chip pilot notebooks moved into the experiment folder keep their
outputs as a record and are not regenerated.

### 4.2 Removed

Recoverable from git history (the snapshot commit of section 9 holds the untracked ones):
`untitled.txt`, the root `.ipynb_checkpoints/`, `scripts/legacy/`, the four TerraTorch tutorial scripts in
`notebooks/terratorch/`, `scripts/seg/{encode,fit_cached,train}.py` (replaced by the runner),
`scripts/seg/run_ee_grid.sh` and `scripts/seg/run_stage1_large.sh` (replaced by the runner and the sbatch template),
and `configs/seg/*_ee.yaml` (replaced by `configs/arms/`).

### 4.3 Left in place

`results/seg/` and `results/seg_cached/`, including `d4_ablation/` and `ee_predictions/`, stay as the record of the
runs before the restructure; the docs and the results notebook point to them. `data/` is not touched: its logs and
pid files stay, but nothing new is written there except caches on the hub. `figures/`, `docs/`, `external/` stay.

### 4.4 Experiments and the cleanup rule

Each `experiments/<date>_<name>/` holds a `README.md` (question, what ran, outcome, where its results are, and the
commit to check out to re-run it), its scripts and its config overrides. Archived studies are a record: they ran
against the snapshot commit, not against the new runner. `.gitignore` stops ignoring `/experiments/`.

| Area | Can be removed when |
|---|---|
| Core: `src/`, `configs/{arms,experiments,machines}`, `scripts/{hub,cluster,env}`, `run_kshot.py`, `tests/` | Never; it changes only with the tests passing |
| `experiments/<date>_<name>/` and its results | Once its conclusion is written into `pipeline.md` |
| `results/<experiment>/` | Once superseded; it is regenerable from code and chips |
| `data/` | Only on the user's decision |

## 5. The K-shot workflow

### 5.1 Chip set contract

A chip set is a directory that describes itself: `manifest.json`, `chips/` with the per-chip rasters, the
per-source records (`tessera_v1.json`, `alphaearth_v1.json`, `s1_rtc.json`), `chip_parcels.parquet` and
`splits/<name>/`. `EE_2021/` already has this shape. The workflow's only data input is the path to it. If the
directory holds one split it is used; if it holds several, `--split` names one.

### 5.2 Configs

| File | Holds | Never holds |
|---|---|---|
| `configs/arms/<arm>.yaml` | Backbone, route (`cache` or `raster`), normalisation, augmentation, decoder and head settings, optimiser, loss, precision, training batch | A chip set, a budget, a path |
| `configs/experiments/<name>.yaml` | Arms, K values, draws, seeds, epochs, checkpoint criterion | Paths |
| `configs/machines/{hub,cluster}.yaml` | Encode batch size, data loader workers, default cache root | Anything that changes a result |

The training batch stays in the arm config, so a result does not depend on the machine. The encode batch size only
affects stage 1 throughput. The arm values are those of today's `_ee` configs; THOR's arm takes the 160 m settings
of `thor_v1_large_ee_pilot.yaml`.

The runner composes the three files and the chip set into the configuration dictionary the fit functions take
today (`run_name`, `seed`, `data`, `label_budget`, `model`, `trainer`, `output_root`), so `fit_cached`,
`generate_embeddings`, `fit_end_to_end` and `FitPredictor` keep their interfaces.

### 5.3 Routes

- **Cache route** (TerraMind, Prithvi, THOR): `generate_embeddings` into `<cache root>/<backbone>/<chip set name>/`,
  skipping chips already cached, then `fit_cached` per cell. Token-grid arms always take this route; end-to-end
  training of a token-grid arm is no longer part of the workflow.
- **Raster route** (TESSERA, AlphaEarth): `fit_end_to_end` per cell, reading the embedding rasters.

### 5.4 Interface

```
python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021
    [--split NAME] [--arms ARM ...] [--machine hub|cluster] [--cache-root DIR] [--dry-run] [--summarise]
```

`--machine` defaults to `hub`. `--arms` restricts the run, which is how the cluster runs one arm per job.
`--dry-run` prints the cells and runs the checks without compute. `--summarise` only collects finished cells.

### 5.5 What a run does, per arm

1. **Checks** before any compute: the manifest and the split exist, and every chip has the rasters this arm needs
   (`_merged.tif` for the token-grid arms, `_tessera.tif` or `_alphaearth.tif` for the raster arms). It stops with
   the list of what is missing.
2. **Encode** (cache route only).
3. **Cells**: every K by draw by seed. A cell is finished when it has both `results.json` and
   `predictions_test.npz`; a finished cell is skipped, and a cell with only `results.json` gets its predictions
   written without a refit.
4. **Test predictions**: after a cell is fitted, `FitPredictor` writes the test-chip predictions to
   `predictions_test.npz` in the cell directory, in the format the results notebook caches today. This is needed
   because a job-scoped cache is gone when the job ends, and two-stage prediction reads the cache.
5. **Summary**: `summary.csv` with one row per cell, replacing `ee_grid_summary.csv`.

### 5.6 Results layout and stamping

```
results/<experiment>/<chip set>/
  <arm>/P<k>_draw<d>_seed<s>/   config.yaml, results.json, checkpoints/best-loss.ckpt, logs/, predictions_test.npz
  logs/<arm>_<timestamp or job id>.log
  summary.csv
```

`results.json` keeps today's fields (configuration, budget and support, country, split protocol, manifest SHA-256,
metrics) and adds the experiment name, the chip set path, the git commit and whether the tree was dirty, the
machine profile and the host name. Nothing is written outside `results/` and the cache root.

### 5.7 The pilot chip set

`scripts/hub/build_pilot_chipset.py --parent data/eurocrops_chips/EE_2021 --name EE_2021_mini` draws whole blocks
with a fixed seed, by default 3 training, 1 validation and 1 test block (about 80 chips), among blocks that have
trainable parcels (training) or labelled parcels (validation and test). It writes a chip set directory per 5.1:
`chips/` as relative links into the parent, and the parent's manifest, per-source records,
`chip_parcels.parquet` and split restricted to the chosen chips, each recording the parent it came from. The
normalisation statistics are the parent's. The name is `EE_2021_mini` because the old 12-chip `EE_2021_pilot`
stays on disk; it costs no extra space and is copied to the cluster beside its parent.

## 6. The hub, git and cluster workflow

### 6.1 Layout on the cluster

The cluster mirrors the hub, so relative paths in configs and `results.json` resolve on both:

```
~/ExplainedGMF4Agri/                    clone of origin
  data/eurocrops_chips/EE_2021/         copied from the hub
  data/eurocrops_chips/EE_2021_mini/    copied from the hub; links into EE_2021
  data/eurocrops/{parquet,vector}/      copied from the hub, about 3.6 GB
  results/                              written by jobs
```

The chip set's size is checked against the free home space before the copy.

### 6.2 One-time set-up

1. A read-only GitHub deploy key, generated on the cluster; the user adds the public key to the repository. The
   cluster can pull and cannot push.
2. `scripts/cluster/setup_env.sh`: Python 3.11 from the `miniconda3` module, Poetry 2.2.1, `poetry install` from
   the same `poetry.lock`, `scripts/env/install_thor.sh`, then a prefetch of the TerraMind, Prithvi and THOR
   weights into the Hugging Face cache on the login node, since compute nodes reach the internet only through the
   UT proxy. It ends with `pytest`.

### 6.3 The loop

| Step | Where | Command |
|---|---|---|
| 1. Change code, run `pytest`, commit, push | Hub | git |
| 2. A new chip set: build it, then copy it | Hub | `scripts/cluster/push_data.sh eurocrops_chips/EE_2021 eurocrops/parquet eurocrops/vector` (rsync, resumable, skips logs and pid files) |
| 3. Update | Cluster | `git pull`, and `poetry install` if the lock changed |
| 4. Run | Cluster | `scripts/cluster/submit.sh kshot EE_2021 [arms]`: one job per arm |
| 5. Bring results back | Hub | `scripts/cluster/pull_results.sh kshot` (rsync of `results/kshot/`) |

Steps 3 to 5 can be driven from the hub over `ssh utwente-hpc`.

### 6.4 One job

`scripts/cluster/kshot.sbatch` runs under `#!/bin/bash -l` on `itc-gpu` with account `itc-tech`, QoS `research`
and one GPU; CPU, memory and time limits are set in `submit.sh`. It sets the UT proxy, creates
`/local/$SLURM_JOB_ID`, removes it on exit, failure or termination, and runs
`run_kshot.py --arms <arm> --machine cluster --cache-root /local/$SLURM_JOB_ID`. Results go to the home results
directory as each cell finishes. A resubmitted job skips finished cells and re-encodes only if cells remain.

### 6.5 The hub

The hub profile's cache root is `data/embeddings/`, so the existing TerraMind and Prithvi Estonia caches are reused
there, and the hub can run the workflow on any chip set as before.

## 7. Verification

1. `pytest` on the hub, including two new tests: the composed configuration for TerraMind, Prithvi, TESSERA and
   AlphaEarth on `EE_2021` at K 5 equals today's `_ee` config with the grid's overrides (`--pct 5`,
   15 epochs), except for `run_name`, `output_root` and machine-only fields, with copies of those configs kept
   as test fixtures since `configs/seg/` is removed; and the pilot builder on a synthetic
   chip set produces a valid chip set with whole blocks in every partition.
2. `--dry-run` on `EE_2021` and `EE_2021_mini` lists 15 cells each and passes the checks.
3. The full `kshot` experiment on `EE_2021_mini` on the hub: 15 `results.json`, 15 prediction files, one
   `summary.csv`, nothing written outside `results/` and `data/embeddings/`.
4. On the cluster: set-up, `pytest`, the same run on `EE_2021_mini` through `submit.sh` (5 jobs), results pulled
   back. Metrics are expected to be of the same order as the hub's, not identical, since repeated GPU fits are not
   bit-identical.

The full Estonia run on the cluster is the first use after the merge and is started when the user decides.

## 8. Documentation edits agreed with this spec

- `CLAUDE.md`: Running the pipeline (hub stages, the runner, the cluster loop), Repository layout, a cluster entry
  under Working environments pointing to `docs/utwente_hpc.md`, Current state (the arms implemented include THOR
  and AlphaEarth, THOR at 160 m confirmed on 9 October, `EE_2021_mini`, the new results layout).
- `pipeline.md`: the stage table of section 1 (code paths), section 6 (the pilot chip set), section 9 (the runner
  and the two routes), section 10 (saved test predictions under the job-scoped cache; the pilot is a block
  subset), section 11 (the new results layout; earlier runs stay in `results/seg*`).
- `docs/utwente_hpc.md`: a section on this project's workflow, replacing the note on the `run_*.sh` grids.
- `README.md`: layout and commands where they name moved paths.

`protocol.md`, `thesis_design.md` and `pipeline_overview.md` do not change.

## 9. Git

All work happens on a branch `restructure`:

1. Snapshot: today's untracked and modified files exactly as they are, minus `untitled.txt`, so that history holds
   the pre-restructure pipeline that the archived experiments refer to. Any file over 5 MB is listed for the user's
   decision before it is committed.
2. The moves and removals of section 4.
3. The runner, the configs and the tests.
4. The cluster scripts.
5. The documentation edits of section 8.

The branch is pushed so the cluster can check it out for verification step 4. It is merged into `main` and pushed
after section 7 passes; the cluster then follows `main`.
