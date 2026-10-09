# Repository Restructure and K-shot Workflow Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Separate the core pipeline from experiments and give the repository one K-shot workflow that runs any chip set, identically on the JupyterHub and on the UT HPC cluster, with git carrying the code between them.

**Architecture:** A thin runner (`src/gfm4agri/pipeline/`, `scripts/run_kshot.py`) composes an arm config, an experiment config and a machine profile into the configuration dictionary the existing fit functions already take, resolves each cache-route arm's feature cache (reuse if it matches, compute otherwise), runs every K x draw x seed cell, and stamps provenance. Cluster jobs are one Slurm job per arm with a job-scoped cache on the node's NVMe.

**Tech Stack:** Python 3.11, Poetry 2.2.1, PyTorch cu128, Lightning, TerraTorch 1.2.11, rasterio, pandas, pytest; bash and Slurm 21.08 on the cluster; rsync and git over SSH.

**Spec:** `docs/superpowers/specs/2026-10-09-repo-restructure-design.md`

## Global Constraints

- Branch `restructure`; nothing is merged into `main` before Task 11 passes.
- Never delete, overwrite or re-export anything under `data/eurocrops_chips/EE_2021/`, `data/embeddings/`, or any other existing data; new data directories only (`data/eurocrops_chips/EE_2021_mini/`, `data/embeddings/<backbone>/EE_2021_mini/`, `data/embeddings/thor_v1_large/EE_2021/` only if the user runs THOR on the hub).
- `results/seg/` and `results/seg_cached/` are left untouched.
- Arms: `terramind_v1_large`, `prithvi_eo_v2_600_tl`, `thor_v1_large` (160 m tokens), `tessera_v1`, `alphaearth_v1`. No new arm, no baseline, no protocol change.
- Experiment `kshot`: budgets 100, 20, 5 % per class; draw 0; seed 0; 15 epochs; checkpoint of lowest validation loss.
- Every command on the hub runs through `poetry run`, from the repository root, with `export PATH="$HOME/.local/bin:$PATH"` in non-interactive shells.
- Tests must not need real data: they build synthetic chip sets in `tmp_path`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Prose in docs: formal register, no dashes used as punctuation.

---

### Task 1: Snapshot commit of the pre-restructure pipeline

**Files:** every untracked and modified file reported by `git status`, except `untitled.txt`.

- [ ] **Step 1: List files over 5 MB.** Run:
  `git ls-files -o -m --exclude-standard -z | xargs -0 du -k | sort -rn | awk '$1>5120'`
  Expected on 9 October 2026: `notebooks/06_monthly_composites_qa.ipynb` (8.2 MB), `notebooks/07_class_phenology_three_seasons.ipynb` (6.4 MB), `notebooks/terratorch/eurocrops_cache_anatomy.ipynb` (6.3 MB). Commit them only if the user approved; otherwise strip their outputs first with `poetry run jupyter nbconvert --clear-output --inplace <file>` on a copy and commit that.
- [ ] **Step 2: Delete `untitled.txt`** (plugin install notes, not project content): `rm untitled.txt`.
- [ ] **Step 3: Commit.**

```bash
git add -A
git status --short | grep -v '^A ' | head    # expect nothing but deletions/modifications already staged
git commit -m "Snapshot the working pipeline before the restructure

Tracks the code, configs, tests, notebooks and Phase 1 docs that had been
developed without being committed, exactly as they ran the full-Estonia grid
of 2 October 2026 and the THOR and D4 studies of 8 October.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
git rev-parse --short HEAD     # record as SNAPSHOT for Task 2's READMEs
```

---

### Task 2: Move hub stages, archive experiments, tidy the root

**Files:**
- Move: `scripts/data/*.py` except `build_pilot_chips.py` to `scripts/hub/`
- Move: 12-chip pilot material to `experiments/2026-09_pilot_12chips/`
- Move: `scripts/seg/run_d4_ablation.sh` to `experiments/2026-10-08_d4_ablation/`
- Move: `configs/seg/thor_v1_large_80m_ee_pilot.yaml` to `experiments/2026-10-08_thor_80m/`
- Move: pipeline notebooks to `notebooks/pipeline/`
- Delete: `scripts/legacy/`, the four TerraTorch tutorial scripts, root `.ipynb_checkpoints/`
- Create: `experiments/README.md` and one `README.md` per study
- Modify: `.gitignore` (drop `/experiments/`), `tests/test_segmentation_data.py:17` (pilot path is unchanged, nothing to edit; verify only)

- [ ] **Step 1: Moves and deletions.**

```bash
mkdir -p scripts/hub experiments/2026-09_pilot_12chips/{configs,notebooks,scripts} \
         experiments/2026-10-08_d4_ablation experiments/2026-10-08_thor_80m notebooks/pipeline
for f in fetch_eurocrops fetch_eurocropsml build_country_chips build_s1_chips build_tessera_chips \
         build_alphaearth_chips fetch_alphaearth_tiles build_chip_split; do
  git mv scripts/data/$f.py scripts/hub/$f.py
done
git mv scripts/data/build_pilot_chips.py experiments/2026-09_pilot_12chips/scripts/
git mv scripts/seg/run_budget_grid.sh experiments/2026-09_pilot_12chips/scripts/
for c in prithvi_eo_v2_300_tl prithvi_eo_v2_600_tl terramind_v1_large terramind_v1_large_s2s1 \
         terramind_v1_small terramind_v1_small_s2s1 tessera_v1_mlp thor_v1_large alphaearth_v1_mlp; do
  git mv configs/seg/${c}_ee_pilot.yaml experiments/2026-09_pilot_12chips/configs/
done
for n in eurocrops_prithvi_pilot eurocrops_terramind_pilot eurocrops_tessera_pilot \
         eurocrops_prithvi_pilot_embeddings eurocrops_terramind_pilot_embeddings eurocrops_thor_pilot_embeddings; do
  git mv notebooks/terratorch/$n.ipynb experiments/2026-09_pilot_12chips/notebooks/
done
git mv notebooks/terratorch/_build_embedding_workflow_notebooks.py experiments/2026-09_pilot_12chips/notebooks/
git mv scripts/seg/run_d4_ablation.sh experiments/2026-10-08_d4_ablation/
git mv configs/seg/thor_v1_large_80m_ee_pilot.yaml experiments/2026-10-08_thor_80m/
for n in eurocrops_cache_anatomy.ipynb eurocrops_ee_results.ipynb _build_cache_anatomy_notebook.py \
         _build_ee_results_notebook.py architecture-diagram.png; do
  git mv notebooks/terratorch/$n notebooks/pipeline/$n
done
git rm -q -r scripts/legacy notebooks/terratorch/example_multitemporalcrop.py \
  notebooks/terratorch/terramind_v1_small_multitemporal_crop.py \
  notebooks/terratorch/terratorch_handson_part1.py notebooks/terratorch/terratorch_handson_part2_example.py
rm -rf .ipynb_checkpoints notebooks/terratorch/__pycache__ scripts/data/__pycache__
rmdir notebooks/terratorch scripts/data 2>/dev/null; ls notebooks scripts
```

- [ ] **Step 2: The pilot notebooks' builder writes beside itself, so check it still resolves.** `_build_embedding_workflow_notebooks.py` uses `HERE = Path(__file__).resolve().parent` for output and walks up from the working directory to `pyproject.toml` for the repository. Its generated code reads `configs/seg/<x>_ee_pilot.yaml`; replace that prefix so a regeneration finds the archived configs:

```bash
sed -i 's#"configs/seg/#"experiments/2026-09_pilot_12chips/configs/#g' \
  experiments/2026-09_pilot_12chips/notebooks/_build_embedding_workflow_notebooks.py
grep -n 'configs/' experiments/2026-09_pilot_12chips/notebooks/_build_embedding_workflow_notebooks.py | head
```
Expected: every config path now under `experiments/2026-09_pilot_12chips/configs/`. The notebooks themselves are a record and are not regenerated.

- [ ] **Step 3: `.gitignore`.** Remove the two lines

```
# Experiments scratch space (runs, checkpoints, logs, ad-hoc outputs)
/experiments/
```

- [ ] **Step 4: `experiments/README.md`.**

```markdown
# Experiments

One folder per study, named `<date>_<name>`. A study is anything that is not the K-shot workflow itself: an
ablation, an exploration, a pilot of something not adopted. Each folder holds a `README.md` with the question,
what ran, the outcome, where the results are and the commit to check out to re-run it, plus the study's own
scripts and configs.

**Cleanup rule.** A study and its results may be deleted once its conclusion is written into
`docs/phase1/pipeline.md`. The core (`src/`, `configs/{arms,experiments,machines}`, `scripts/{hub,cluster,env}`,
`scripts/run_kshot.py`, `tests/`) is never deleted.

Archived studies are a record: they ran against the code of their commit, not against the current runner.
```

- [ ] **Step 5: Study READMEs.** Replace `SNAPSHOT` with the hash from Task 1 Step 3.

`experiments/2026-09_pilot_12chips/README.md`:

```markdown
# The 12-chip pilot, September to October 2026

**Question.** Does each encoder wire up to the EuroCrops chips end to end, and do the two-stage (cached) and
end-to-end fits agree?

**What ran.** TerraMind v1 small and large, TerraMind on S2 plus S1 (small and large), Prithvi-EO-2.0 300M TL and
600M TL, TESSERA v1 with a per-pixel MLP, AlphaEarth v1, and THOR v1 large at 160 m, on the 12 chips of
`data/eurocrops_chips/EE_2021_pilot/` built by `scripts/build_pilot_chips.py`. The pilot has no block split and no
test partition, so test metrics are computed on the validation chips. Budget grids by `scripts/run_budget_grid.sh`.

**Outcome.** The wiring of every arm, the D4 caching of the two-stage fit and the budget machinery were checked
here; the numbers are not Phase 1 results. Superseded by the K-shot workflow and the `EE_2021_mini` pilot.

**Results.** `results/seg/*_ee_pilot/` and `results/seg_cached/*_ee_pilot/`. Notebooks in `notebooks/`.

**Re-run.** `git checkout SNAPSHOT`, then the commands in the notebooks or `scripts/run_budget_grid.sh` from the
repository root.
```

`experiments/2026-10-08_d4_ablation/README.md`:

```markdown
# How many cached D4 variants does the two-stage fit need? 8 October 2026

**Question.** The two-stage fit encodes training chips in all eight D4 symmetries. Would the identity alone, or the
four quarter turns, train as well?

**What ran.** TerraMind v1 large on full Estonia at K = 5 %, draw 0, 15 epochs, seeds 0 to 2, trained from the
identity only (k0), the four quarter turns (rot4) and all eight (all8). `run_d4_ablation.sh`.

**Outcome.** Test Macro-F1 0.304 +- 0.030 (k0), 0.335 +- 0.049 (rot4), 0.358 +- 0.003 (all8). The workflow keeps
all eight.

**Results.** `results/seg_cached/d4_ablation/`.

**Re-run.** `git checkout SNAPSHOT`, then
`setsid nohup bash experiments/2026-10-08_d4_ablation/run_d4_ablation.sh > results/seg_cached/d4_ablation/run.log 2>&1 &`
after moving the script back to `scripts/seg/`, where it resolves the repository.
```

`experiments/2026-10-08_thor_80m/README.md`:

```markdown
# THOR on 80 m tokens, 8 October 2026

**Question.** THOR's paper finds smaller patches much better. Does THOR v1 large on 80 m tokens (8 and 4 px
patches, a 28 x 28 grid) beat the same encoder on 160 m tokens?

**What ran.** `thor_v1_large_80m` and `thor_v1_large` on the 12-chip pilot, two-stage, K = 100, 20 and 5 %,
seeds 0 to 2. Config `thor_v1_large_80m_ee_pilot.yaml`; the 160 m config is in `../2026-09_pilot_12chips/configs/`.

**Outcome.** Test Macro-F1 (on the pilot's validation chips), mean of three seeds: 80 m 0.283, 0.176, 0.106 against
160 m 0.230, 0.092, 0.088 at 100, 20 and 5 %. On 9 October 2026 the user kept 160 m for parity with TerraMind;
80 m is not an arm.

**Results.** `results/seg_cached/thor_v1_large_80m_ee_pilot/`, `results/seg_cached/thor_v1_large_ee_pilot/`.

**Re-run.** `git checkout SNAPSHOT`, then `scripts/seg/encode.py` and `scripts/seg/fit_cached.py` with this config.
```

- [ ] **Step 6: Run the test suite** (moves must not break imports):
  `poetry run pytest -q -x`
  Expected: same pass and skip counts as before the moves.

- [ ] **Step 7: Commit.**

```bash
git add -A
git commit -m "Separate hub stages, archived studies and pipeline notebooks

Chip extraction scripts move to scripts/hub, the 12-chip pilot, the D4 ablation
and the THOR 80 m study to experiments/ with a README each, the pipeline
notebooks to notebooks/pipeline. Legacy scripts and TerraTorch tutorials are
removed; experiments/ is tracked.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 3: Move the EDA code out of `results/eda`

**Files:**
- Move: `notebooks/0[1-7]_*.ipynb` to `notebooks/eda/`
- Move: `results/eda/eurocrops/_build_notebook_0{4,5,7}.py` to `notebooks/eda/`
- Move: every other `results/eda/**/_*.py` to `notebooks/eda/analysis/` (keeping `cropharvest/`, `eurocrops/`)
- Create (scratchpad, not committed): `patch_eda.py`, `check_eda.py`

Outputs stay in `results/eda/`; only the code moves.

- [ ] **Step 1: Moves.**

```bash
mkdir -p notebooks/eda/analysis/cropharvest notebooks/eda/analysis/eurocrops
git mv notebooks/0[1-7]_*.ipynb notebooks/eda/
for n in 04 05 07; do git mv results/eda/eurocrops/_build_notebook_$n.py notebooks/eda/; done
for f in results/eda/_*.py; do git mv "$f" notebooks/eda/analysis/; done
for f in results/eda/cropharvest/_*.py; do git mv "$f" notebooks/eda/analysis/cropharvest/; done
for f in results/eda/eurocrops/_*.py; do git mv "$f" notebooks/eda/analysis/eurocrops/; done
rm -rf results/eda/__pycache__ results/eda/*/__pycache__
```

- [ ] **Step 2: Write the patch script** to the scratchpad as `patch_eda.py`:

```python
"""Point the moved EDA code and notebooks at their new places; outputs stay in results/eda."""
import json
import re
import sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
NB = REPO / "notebooks" / "eda"
FIND = 'next(p for p in {start} if (p / "pyproject.toml").exists())'

TEXT_RULES = [  # applied to notebook sources and builder sources
    (r"\]\(\.\./results/eda/((?:cropharvest/|eurocrops/)?_(?!build_notebook)[a-z0-9_]+\.py)\)", r"](analysis/\1)"),
    (r"\]\(\.\./results/", "](../../results/"),
    (r"results/eda/eurocrops/_build_notebook_", "notebooks/eda/_build_notebook_"),
    (r"results/eda/((?:cropharvest/|eurocrops/)?_(?!build_notebook)[a-z0-9_]+\.py)", r"notebooks/eda/analysis/\1"),
    (r"notebooks/(0[1-7]_)", r"notebooks/eda/\1"),
    (r'Path\.cwd\(\)\.parent if Path\.cwd\(\)\.name == "notebooks" else Path\.cwd\(\)',
     FIND.format(start="[Path.cwd(), *Path.cwd().parents]")),
    (r'^(\s*)EDA = REPO / "results" / "eda"(.*)$',
     r'\1EDA = REPO / "results" / "eda"\2\n\1EDA_CODE = REPO / "notebooks" / "eda" / "analysis"\2'),
    (r"str\(EDA / (name|script)\)", r"str(EDA_CODE / \1)"),
    (r"sys\.path\.insert\(0, str\(EDA\)\)", "sys.path.insert(0, str(EDA_CODE))"),
]


def sub_all(text: str, rules) -> tuple[str, int]:
    n = 0
    for pat, rep in rules:
        text, k = re.subn(pat, rep, text, flags=re.M)
        n += k
    return text, n


def patch_notebook(path: Path) -> int:
    raw = path.read_text(encoding="utf-8")
    indent = len(raw.splitlines()[1]) - len(raw.splitlines()[1].lstrip())
    nb = json.loads(raw)
    total = 0
    for cell in nb["cells"]:
        src, n = sub_all("".join(cell["source"]), TEXT_RULES)
        if n:
            cell["source"] = src.splitlines(keepends=True)
            total += n
    path.write_text(json.dumps(nb, indent=indent, ensure_ascii=False) + "\n", encoding="utf-8")
    return total


def patch_builder(path: Path) -> int:
    text = path.read_text(encoding="utf-8")
    text, n = sub_all(text, TEXT_RULES)
    text, k = re.subn(r'^REPO = Path\("/data/private/THESIS - ExplainedGMF4Agri"\)$',
                      "REPO = " + FIND.format(start="Path(__file__).resolve().parents"), text, flags=re.M)
    text, j = re.subn(r'REPO / "notebooks" / "(0[1-7]_)', r'REPO / "notebooks" / "eda" / "\1', text)
    path.write_text(text, encoding="utf-8")
    return n + k + j


def patch_script(path: Path) -> int:
    sub = path.parent.name if path.parent.name in ("cropharvest", "eurocrops") else ""
    out = 'REPO / "results" / "eda"' + (f' / "{sub}"' if sub else "")
    text = path.read_text(encoding="utf-8")
    rules = [
        (r"^HERE = Path\(__file__\)\.(?:resolve\(\)\.parent|parent\.resolve\(\))$",
         "CODE = Path(__file__).resolve().parent\n"
         "REPO = " + FIND.format(start="CODE.parents") + "\n"
         f"HERE = {out}  # outputs stay in results/eda; the code lives in notebooks/eda/analysis"),
        (r"^REPO = HERE\.parents\[\d\]\n", ""),
        (r"^REPO = Path\(__file__\)\.resolve\(\)\.parents\[\d\]\n", ""),
        (r"sys\.path\.insert\(0, str\(HERE\)\)", "sys.path.insert(0, str(CODE))"),
        (r'sys\.path\.insert\(0, str\(REPO / "results" / "eda"\)\)',
         'sys.path.insert(0, str(REPO / "notebooks" / "eda" / "analysis"))'),
        (r'str\(HERE / "_run_analysis\.py"\)', 'str(CODE / "_run_analysis.py")'),
    ]
    text, n = sub_all(text, rules)
    path.write_text(text, encoding="utf-8")
    return n


for p in sorted(NB.glob("0*.ipynb")):
    print(f"{p.relative_to(REPO)}: {patch_notebook(p)} edits")
for p in sorted(NB.glob("_build_notebook_*.py")):
    print(f"{p.relative_to(REPO)}: {patch_builder(p)} edits")
for p in sorted((NB / "analysis").rglob("_*.py")):
    print(f"{p.relative_to(REPO)}: {patch_script(p)} edits")
```

- [ ] **Step 3: Run it.** `poetry run python "$SCRATCH/patch_eda.py" .`
  Expected: every analysis script reports at least 1 edit; `_hcat.py` 1 edit; notebook 05 reports the REPO edit.

- [ ] **Step 4: Static check: no stale references remain.**

```bash
grep -rnE 'results/eda/(cropharvest/|eurocrops/)?_[a-z0-9_]+\.py|str\(EDA / (name|script)\)|HERE\.parents|parents\[2\]' \
  notebooks/eda | grep -v '^Binary' | head
poetry run python -m py_compile notebooks/eda/_build_notebook_0*.py notebooks/eda/analysis/*.py notebooks/eda/analysis/*/*.py
```
Expected: grep prints nothing; py_compile is silent.

- [ ] **Step 5: Runtime check of paths** with scratchpad `check_eda.py`: load every guarded analysis script without running its `main`, and execute each notebook's setup cells from `notebooks/eda/`.

```python
import json, os, runpy, sys
from pathlib import Path

REPO = Path(sys.argv[1]).resolve()
GUARDED = [p for p in (REPO / "notebooks/eda/analysis").rglob("_*.py")
           if '__name__ == "__main__"' in p.read_text()]
for p in sorted(GUARDED):
    g = runpy.run_path(str(p), run_name="eda_check")
    assert g["REPO"] == REPO, (p, g["REPO"])
    assert g["HERE"].is_dir(), (p, g["HERE"])
    print("ok", p.relative_to(REPO), "->", g["HERE"].relative_to(REPO))
os.chdir(REPO / "notebooks/eda")
for nb in sorted(Path(".").glob("0*.ipynb")):
    ns = {}
    setup = next("".join(c["source"]) for c in json.loads(nb.read_text())["cells"]
                 if c["cell_type"] == "code" and "REPO =" in "".join(c["source"]))
    try:
        exec(compile(setup, str(nb), "exec"), ns)   # only the cell that defines REPO
    except ModuleNotFoundError as e:
        print("skip", nb, "(import not installed:", e, ")")
        continue
    assert ns.get("REPO") == REPO, (nb, ns.get("REPO"))
    for key in ("EDA", "EDA_CODE"):
        if key in ns:
            assert Path(ns[key]).is_dir(), (nb, key, ns[key])
    print("ok", nb, {k: str(Path(ns[k]).relative_to(REPO)) for k in ("EDA", "EDA_CODE") if k in ns})
```
Run: `poetry run python "$SCRATCH/check_eda.py" .` Expected: one `ok` line per guarded script and per notebook. `_estonia_phenology.py` and `_hcat.py` have no main guard and are covered by Step 4 only.

- [ ] **Step 6: Commit.**

```bash
git add -A
git commit -m "Move the EDA code from results/eda to notebooks/eda

Notebooks 01 to 07 and their builders move to notebooks/eda, the analysis
scripts to notebooks/eda/analysis. Their outputs stay in results/eda; paths
are resolved from the repository root rather than the script's folder.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 4: The end-to-end fit as a library function, and a synthetic split chip set

**Files:**
- Create: `src/gfm4agri/benchmark/fit.py`
- Modify: `scripts/seg/train.py` (becomes a wrapper; removed in Task 8)
- Modify: `tests/conftest.py` (add `make_split_chipset` and the `split_chipset` fixture)
- Test: `tests/test_fit.py`

**Interfaces:**
- Produces: `gfm4agri.benchmark.fit.fit_end_to_end(cfg: dict, *, fast_dev_run: bool = False) -> dict` (the `results` dict, also written to `<REPO>/<output_root>/<run_name>/<budget_tag>_seed<seed>/results.json`), `budget_tag(b: dict) -> str`, `labelled_fraction(dataset) -> float`.
- Produces: `tests/conftest.py::make_split_chipset(root: Path) -> Path`, fixture `split_chipset(tmp_path) -> Path`, constant `SYN_SPLIT = "blocks1_buf0_seed0__0000abcd"`. Chips `EE_00000_00000`, `EE_00001_00000` (block `0_0`, pool), `EE_00004_00000`, `EE_00005_00000` (block `1_0`, val), `EE_00008_00000`, `EE_00009_00000` (block `2_0`, test). Each chip: 32 x 32, parcels `base+1..base+4` with classes 0, 1, 0, -1; `_merged.tif` (6 bands), `_tessera.tif` (4 bands), `.mask.tif`, `.parcels.tif`.

- [ ] **Step 1: Add the fixture to `tests/conftest.py`** (append after the existing fixtures):

```python
SYN_H = 32
SYN_SPLIT = "blocks1_buf0_seed0__0000abcd"
SYN_BLOCKS = {"0_0": ("pool", ["EE_00000_00000", "EE_00001_00000"]),
              "1_0": ("val", ["EE_00004_00000", "EE_00005_00000"]),
              "2_0": ("test", ["EE_00008_00000", "EE_00009_00000"])}
SYN_LISTS = {"pool": "training", "val": "validation", "test": "test"}


def _write_raster(path: Path, arr: np.ndarray) -> None:
    import rasterio
    from affine import Affine

    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=SYN_H, height=SYN_H, count=arr.shape[0],
                       dtype=arr.dtype, crs="EPSG:3035",
                       transform=Affine(10, 0, 0, 0, -10, SYN_H * 10)) as dst:
        dst.write(arr)


def make_split_chipset(root: Path) -> Path:
    """A chip set in the full-country layout, small enough for CPU fits.

    Six chips in three blocks of two: block ``0_0`` is the pool (training), ``1_0`` validation and
    ``2_0`` test. Every chip has four quadrant parcels of classes 0, 1, 0 and out of scheme, a
    two-band three-month Sentinel-2 stack, a four-dimension TESSERA raster, a mask and a parcel
    raster; ``chip_parcels.parquet`` and one spatial split directory describe them.
    """
    chips_dir = root / "chips"
    chips_dir.mkdir(parents=True)
    q = SYN_H // 2
    rows, table, manifest_chips, lists = [], [], [], {v: [] for v in SYN_LISTS.values()}
    for n, (block, (part, cids)) in enumerate(SYN_BLOCKS.items()):
        for i, cid in enumerate(cids):
            base = 100 * (2 * n + i + 1)
            ids = np.zeros((SYN_H, SYN_H), np.int32)
            ids[:q, :q], ids[:q, q:], ids[q:, :q], ids[q:, q:] = base + 1, base + 2, base + 3, base + 4
            mask = np.select([ids % 100 == 1, ids % 100 == 2, ids % 100 == 3], [0, 1, 0], -1)
            rng = np.random.default_rng(base)
            _write_raster(chips_dir / f"{cid}_merged.tif",
                          rng.integers(0, 3000, (6, SYN_H, SYN_H)).astype(np.int16))
            _write_raster(chips_dir / f"{cid}_tessera.tif",
                          np.stack([ids.astype(np.float32) / 1000 + k for k in range(4)]))
            _write_raster(chips_dir / f"{cid}.mask.tif", mask.astype(np.int16))
            _write_raster(chips_dir / f"{cid}.parcels.tif", ids)
            col = int(cid.split("_")[1])
            rows.append({"chip_id": cid, "col": col, "row": 0, "partition": part, "block_id": block,
                         "labelled_parcels": 3, "trainable_parcels": 3 if part == "pool" else 0})
            table += [{"chip_id": cid, "parcel_id": base + k, "class_index": c, "pixels": q * q}
                      for k, c in ((1, 0), (2, 1), (3, 0))]
            manifest_chips.append({"chip_id": cid, "col": col, "row": 0,
                                   "bounds_3035": [col * 2240.0, 0.0, (col + 1) * 2240.0, 2240.0]})
            lists[SYN_LISTS[part]].append(cid)
    manifest = {
        "dataset": "synthetic", "country": "EE", "year": 2021, "ignore_index": -1,
        "classes": [{"index": 0, "hcat_code": "1", "name": "a"},
                    {"index": 1, "hcat_code": "2", "name": "b"}],
        "imagery": {"band_names": ["RED", "NIR_NARROW"], "n_months": 3},
        "normalisation": {"means": [0.0, 0.0], "stds": [1.0, 1.0]},
        "chips": manifest_chips,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "tessera_v1.json").write_text(json.dumps({
        "representation": "tessera_v1", "file_suffix": "_tessera.tif",
        "band_names": [f"TESSERA_{k:03d}" for k in range(4)],
        "normalisation": {"means": [0.0] * 4, "stds": [1.0] * 4}}))
    pd.DataFrame(table).astype({"class_index": np.int16}).to_parquet(root / "chip_parcels.parquet",
                                                                      index=False)
    split = root / "splits" / SYN_SPLIT
    split.mkdir(parents=True)
    for name, cids in lists.items():
        (split / f"{name}_data.txt").write_text("\n".join(cids) + "\n")
    pd.DataFrame(rows).to_csv(split / "chips.csv", index=False)
    np.save(split / "buffer_parcels.npy", np.array([], dtype=np.int64))
    (split / "split.json").write_text(json.dumps({
        "split": SYN_SPLIT, "config_hash": "0000abcd00000000", "protocol": "synthetic blocks",
        "lists": {k: len(v) for k, v in lists.items()}}))
    return root


@pytest.fixture()
def split_chipset(tmp_path: Path) -> Path:
    """A fresh synthetic full-country chip set under ``tmp_path / "SYN_2021"``."""
    return make_split_chipset(tmp_path / "SYN_2021")
```
Also add `import json` at the top of `tests/conftest.py`.

- [ ] **Step 2: Write the failing test** `tests/test_fit.py`:

```python
"""The end-to-end fit, moved from scripts/seg/train.py into the library, still writes its result file."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("terratorch")
pytest.importorskip("rasterio")

from conftest import SYN_SPLIT  # noqa: E402


def _cfg(chips, out, pct=100):
    return {
        "run_name": "tiny_tessera", "seed": 0,
        "data": {"root": str(chips), "split": str(chips / "splits" / SYN_SPLIT),
                 "normalisation": "chips", "batch_size": 2, "num_workers": 0, "augment": True},
        "label_budget": {"mode": "pct", "pct": pct, "draw_seed": 0},
        "model": {"backbone": "tessera_v1", "lr": 1.0e-3, "weight_decay": 0.05,
                  "head_dropout": 0.2, "loss": "ce"},
        "trainer": {"max_epochs": 1, "precision": "32-true", "log_every_n_steps": 1},
        "output_root": str(out),
    }


def test_fit_end_to_end_writes_results_where_the_grid_expects_them(split_chipset, tmp_path):
    from gfm4agri.benchmark.fit import fit_end_to_end

    out = tmp_path / "results"
    r = fit_end_to_end(_cfg(split_chipset, out, pct=100), fast_dev_run=True)
    path = out / "tiny_tessera" / "P100_draw0_seed0" / "results.json"
    assert path.exists() and json.loads(path.read_text())["run"] == "tiny_tessera"
    assert r["chips"] == {"train": 2, "val": 2, "test": 2}
    assert r["label_budget"]["support"] == {"a": {"available": 4, "drawn": 4},
                                            "b": {"available": 2, "drawn": 2}}
    assert r["split_protocol"]["split"] == SYN_SPLIT
```

- [ ] **Step 3: Run it to see it fail.** `poetry run pytest tests/test_fit.py -v`
  Expected: FAIL, `ModuleNotFoundError: No module named 'gfm4agri.benchmark.fit'`.

- [ ] **Step 4: Create `src/gfm4agri/benchmark/fit.py`** by moving the body of `scripts/seg/train.py::main` after the argument parsing, unchanged except: `_budget_tag` and `_labelled_fraction` lose their underscore; `cfg` is the argument; `args.fast_dev_run` becomes `fast_dev_run`; the function returns `results`.

```python
"""One end-to-end fit: the frozen encoder inside the training loop, the neck and decoder trained.

Moved unchanged from ``scripts/seg/train.py``. The K-shot workflow uses it for the raster arms,
whose features are the embedding rasters themselves; token-grid arms are fitted from cached
features by :func:`gfm4agri.benchmark.cached.fit_cached`.

Writes ``<output_root>/<run_name>/<budget>_seed<seed>/``: the Lightning logs and checkpoint,
``config.yaml`` as resolved, and ``results.json`` carrying the configuration, the label budget,
the country, the split protocol, the parameter counts and the metrics. Relative paths in the
configuration are taken from the repository root.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[3]

__all__ = ["budget_tag", "fit_end_to_end", "labelled_fraction"]


def budget_tag(b: dict) -> str:
    return "dense" if b["mode"] == "dense" else f"P{b['pct']:g}_draw{b['draw_seed']}"


def labelled_fraction(dataset) -> float:
    # body of scripts/seg/train.py::_labelled_fraction, verbatim (docstring included)
    ...


def fit_end_to_end(cfg: dict, *, fast_dev_run: bool = False) -> dict:
    """Train and evaluate one fit described by ``cfg``; return and write its result record."""
    # body of scripts/seg/train.py::main from `import lightning.pytorch as pl` to the final
    # metric prints, verbatim, with args.fast_dev_run -> fast_dev_run, _budget_tag -> budget_tag,
    # _labelled_fraction -> labelled_fraction, and `return results` at the end.
    ...
```
The two `...` bodies are copied verbatim from `scripts/seg/train.py` at the snapshot commit: the whole body of `_labelled_fraction`, and the body of `main` from the line `import lightning.pytorch as pl` to its last statement. No statement is added or removed beyond the renames listed.

- [ ] **Step 5: Make `scripts/seg/train.py` a wrapper** (keeps `run_ee_grid.sh` working until Task 8):

```python
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override config entries, dotted keys")
    ap.add_argument("--fast-dev-run", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    for s in args.set:
        _set(cfg, s)

    from gfm4agri.benchmark.fit import fit_end_to_end

    fit_end_to_end(cfg, fast_dev_run=args.fast_dev_run)
```
Delete `_budget_tag`, `_labelled_fraction`, and the now unused imports (`datetime`, `json`, `time`, `numpy`).

- [ ] **Step 6: Run the tests.** `poetry run pytest tests/test_fit.py tests/test_segmentation_data.py -v`
  Expected: PASS.

- [ ] **Step 7: Commit.**

```bash
git add src/gfm4agri/benchmark/fit.py scripts/seg/train.py tests/conftest.py tests/test_fit.py
git commit -m "Move the end-to-end fit into gfm4agri.benchmark.fit

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 5: Arm, experiment and machine configs, and their composition

**Files:**
- Create: `configs/arms/{terramind_v1_large,prithvi_eo_v2_600_tl,thor_v1_large,tessera_v1,alphaearth_v1}.yaml`
- Create: `configs/experiments/kshot.yaml`, `configs/machines/{hub,cluster}.yaml`
- Create: `src/gfm4agri/pipeline/__init__.py`, `src/gfm4agri/pipeline/config.py`
- Create: `tests/fixtures/legacy_configs/` (copies of five `configs/seg/` files)
- Test: `tests/test_pipeline_config.py`

**Interfaces:**
- Produces in `gfm4agri.pipeline.config`: `REPO: Path`, `CONFIGS: Path`, `Cell(arm: str, pct: float, draw: int, seed: int)` with `.tag -> "P<pct:g>_draw<d>_seed<s>"`, `load_arm(name, configs=CONFIGS) -> dict`, `load_experiment(path) -> dict`, `load_machine(name, configs=CONFIGS) -> dict`, `plan_cells(exp: dict, arms: list[str] | None = None) -> list[Cell]`, `resolve_split(chips: Path, split: str | None = None) -> Path`, `repo_relative(path) -> str`, `compose(arm_name: str, arm: dict, exp: dict, machine: dict, chips: Path, split_dir: Path, cell: Cell, output_root: Path) -> dict`.

- [ ] **Step 1: Legacy fixtures.**

```bash
mkdir -p tests/fixtures/legacy_configs
cp configs/seg/{terramind_v1_large_ee,prithvi_eo_v2_600_tl_ee,tessera_v1_mlp_ee,alphaearth_v1_mlp_ee}.yaml tests/fixtures/legacy_configs/
cp experiments/2026-09_pilot_12chips/configs/thor_v1_large_ee_pilot.yaml tests/fixtures/legacy_configs/
```

- [ ] **Step 2: Arm configs.**

`configs/arms/terramind_v1_large.yaml`:
```yaml
# TerraMind v1 large, frozen, on the twelve monthly Sentinel-2 composites; ChannelBottleneck and a
# 12.9 M parameter UNet decoder fitted from cached features. Settings of the full-Estonia grid of
# 2 October 2026 (configs/seg/terramind_v1_large_ee.yaml at the snapshot commit).
route: cache                   # cache: fitted from cached encoder features; raster: from embedding rasters
data:
  normalisation: backbone      # TerraMind v1 S2L2A pretraining statistics
  batch_size: 2                # with accumulate_grad_batches 2, an effective batch of 4
  augment: true                # D4; the cached fit draws one of the eight cached variants
model:
  backbone: terramind_v1_large
  lr: 1.0e-4
  weight_decay: 0.05
  head_dropout: 0.1
  loss: ce
trainer:
  precision: 16-mixed
  log_every_n_steps: 10
  accumulate_grad_batches: 2
```

`configs/arms/prithvi_eo_v2_600_tl.yaml`: identical except the header comment names Prithvi-EO-2.0 600M TL and `configs/seg/prithvi_eo_v2_600_tl_ee.yaml`, `normalisation: backbone  # Prithvi-EO-2.0 HLS pretraining statistics`, `backbone: prithvi_eo_v2_600_tl`.

`configs/arms/thor_v1_large.yaml`: identical to the TerraMind file except the header:
```yaml
# THOR v1 large, frozen, on 160 m tokens (patch 16 on the 10 m bands, 8 on the 20 m bands) to match
# TerraMind, as decided on 9 October 2026; ChannelBottleneck and the UNet decoder fitted from cached
# features. Settings of configs/seg/thor_v1_large_ee_pilot.yaml at the snapshot commit.
```
with `normalisation: backbone  # THOR pretraining statistics` and `backbone: thor_v1_large`.

`configs/arms/tessera_v1.yaml`:
```yaml
# TESSERA v1 precomputed embeddings (128 dims per 10 m pixel) with a per-pixel MLP head (512, 256),
# trained end to end on the embedding rasters. Pixels without an embedding enter as zero. Settings of
# the full-Estonia grid of 2 October 2026 (configs/seg/tessera_v1_mlp_ee.yaml at the snapshot commit).
route: raster
data:
  normalisation: chips         # embeddings have no pretraining statistics; training-chip statistics
  batch_size: 4
  augment: true                # D4 is exact for a per-pixel embedding
model:
  backbone: tessera_v1
  lr: 1.0e-3                   # a small head trained from scratch
  weight_decay: 0.05
  head_dropout: 0.2
  loss: ce
trainer:
  precision: 16-mixed
  log_every_n_steps: 10
```

`configs/arms/alphaearth_v1.yaml`: identical to TESSERA except the header (AlphaEarth V1, 64 dims, unit vectors, `configs/seg/alphaearth_v1_mlp_ee.yaml`) and `backbone: alphaearth_v1`.

- [ ] **Step 3: Experiment and machine configs.**

`configs/experiments/kshot.yaml`:
```yaml
# The K-shot workflow as run on full Estonia on 2 October 2026, for the five arms of Phase 1.
# Run on any chip set:  poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips <dir>
name: kshot
arms: [terramind_v1_large, prithvi_eo_v2_600_tl, thor_v1_large, tessera_v1, alphaearth_v1]
budgets_pct: [100, 20, 5]      # per cent of the trainable parcels of each class
draws: [0]
seeds: [0]
epochs: 15
checkpoint: val_loss           # the checkpoint of lowest validation loss, the only criterion implemented
```

`configs/machines/hub.yaml`:
```yaml
# The ITC JupyterHub: one RTX A4000, 16 GB. Nothing here changes a result.
cache_root: data/embeddings    # caches are looked up here, and missing ones computed here
num_workers: 16                # data loader workers of a fit and of the test predictions
encode_num_workers: 8
encode_batch_size: {terramind_v1_large: 4, prithvi_eo_v2_600_tl: 2, thor_v1_large: 2}
```

`configs/machines/cluster.yaml`:
```yaml
# A UT HPC itc-gpu node: one RTX PRO 6000 Blackwell, 96 GB, per job. Nothing here changes a result.
cache_root: data/embeddings    # persistent caches are looked up here; a job computes missing ones into its --scratch
num_workers: 16
encode_num_workers: 16
encode_batch_size: {terramind_v1_large: 16, prithvi_eo_v2_600_tl: 8, thor_v1_large: 8}
```

- [ ] **Step 4: Write the failing tests** `tests/test_pipeline_config.py`:

```python
"""The K-shot workflow composes exactly the configurations the full-Estonia grid ran with."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

pytest.importorskip("terratorch")

from gfm4agri.pipeline.config import (  # noqa: E402
    CONFIGS, REPO, Cell, compose, load_arm, load_experiment, load_machine, plan_cells,
    repo_relative, resolve_split)

LEGACY = Path(__file__).parent / "fixtures" / "legacy_configs"
EE = REPO / "data" / "eurocrops_chips" / "EE_2021"
SPLIT = EE / "splits" / "blocks4_buf1600_seed0__6b0eb4cb"
GRID = {"terramind_v1_large": "terramind_v1_large_ee.yaml",
        "prithvi_eo_v2_600_tl": "prithvi_eo_v2_600_tl_ee.yaml",
        "tessera_v1": "tessera_v1_mlp_ee.yaml",
        "alphaearth_v1": "alphaearth_v1_mlp_ee.yaml"}


def _without_run_specific(cfg: dict) -> dict:
    cfg = copy.deepcopy(cfg)
    cfg.pop("run_name"), cfg.pop("output_root"), cfg["data"].pop("num_workers")
    return cfg


@pytest.mark.parametrize("pct", [100, 20, 5])
@pytest.mark.parametrize("arm, legacy", GRID.items())
def test_composed_cell_is_the_grid_configuration(arm, legacy, pct):
    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    got = compose(arm, load_arm(arm), exp, load_machine("hub"), EE, SPLIT, Cell(arm, pct, 0, 0),
                  REPO / "results" / "kshot" / "EE_2021")
    # The grid passed the budget, the draw and 15 epochs on the command line.
    want = yaml.safe_load((LEGACY / legacy).read_text())
    want["label_budget"].update(mode="pct", pct=pct, draw_seed=0)
    want["trainer"]["max_epochs"] = 15
    assert _without_run_specific(got) == _without_run_specific(want)
    assert got["run_name"] == arm and got["output_root"] == "results/kshot/EE_2021"
    assert got["data"]["root"] == "data/eurocrops_chips/EE_2021"


def test_thor_arm_keeps_the_160m_settings_of_its_pilot():
    arm, pilot = load_arm("thor_v1_large"), yaml.safe_load(
        (LEGACY / "thor_v1_large_ee_pilot.yaml").read_text())
    assert arm["model"] == pilot["model"]
    assert {k: arm["data"][k] for k in ("normalisation", "batch_size", "augment")} == \
           {k: pilot["data"][k] for k in ("normalisation", "batch_size", "augment")}
    assert arm["trainer"]["accumulate_grad_batches"] == pilot["trainer"]["accumulate_grad_batches"]
    from gfm4agri.benchmark.backbones import get_backbone
    assert get_backbone(arm["model"]["backbone"]).resolution_group == "token_grid"


def test_every_arm_route_matches_its_backbone():
    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    routes = {a: load_arm(a)["route"] for a in exp["arms"]}
    assert routes == {"terramind_v1_large": "cache", "prithvi_eo_v2_600_tl": "cache",
                      "thor_v1_large": "cache", "tessera_v1": "raster", "alphaearth_v1": "raster"}


def test_a_route_that_contradicts_the_backbone_is_refused(tmp_path):
    (tmp_path / "arms").mkdir()
    bad = yaml.safe_load((CONFIGS / "arms" / "tessera_v1.yaml").read_text()) | {"route": "cache"}
    (tmp_path / "arms" / "bad.yaml").write_text(yaml.safe_dump(bad))
    with pytest.raises(ValueError, match="needs 'raster'"):
        load_arm("bad", configs=tmp_path)


def test_cells_run_budgets_outermost_within_each_arm():
    exp = {"arms": ["a", "b"], "budgets_pct": [100, 5], "draws": [0, 1], "seeds": [0]}
    cells = plan_cells(exp)
    assert [c.tag for c in cells[:4]] == ["P100_draw0_seed0", "P100_draw1_seed0",
                                         "P5_draw0_seed0", "P5_draw1_seed0"]
    assert {c.arm for c in cells[:4]} == {"a"} and len(cells) == 8
    assert [c.arm for c in plan_cells(exp, ["b"])] == ["b"] * 4
    with pytest.raises(ValueError, match="not in the experiment"):
        plan_cells(exp, ["c"])


def test_split_is_the_only_one_or_the_named_one(split_chipset):
    from conftest import SYN_SPLIT

    assert resolve_split(split_chipset).name == SYN_SPLIT
    assert resolve_split(split_chipset, "blocks1_buf0_seed0").name == SYN_SPLIT
    other = split_chipset / "splits" / "other__1"
    other.mkdir()
    (other / "split.json").write_text("{}")
    with pytest.raises(ValueError, match="holds 2 splits"):
        resolve_split(split_chipset)
    assert resolve_split(split_chipset, "other").name == "other__1"


def test_paths_inside_the_repository_are_stored_relative(tmp_path):
    assert repo_relative(REPO / "data" / "x") == "data/x"
    assert repo_relative(tmp_path) == str(tmp_path.resolve())
```

- [ ] **Step 5: Run them to see them fail.** `poetry run pytest tests/test_pipeline_config.py -v`
  Expected: FAIL, `ModuleNotFoundError: No module named 'gfm4agri.pipeline'`.

- [ ] **Step 6: Implement.** `src/gfm4agri/pipeline/__init__.py`:

```python
"""The K-shot workflow: arms x label budgets x draws x seeds on one chip set."""
```

`src/gfm4agri/pipeline/config.py`:

```python
"""Compose the configuration of one K-shot cell from an arm, an experiment, a machine and a chip set.

The fit functions take one dictionary in the layout of the earlier per-run YAML files
(``run_name``, ``seed``, ``data``, ``label_budget``, ``model``, ``trainer``, ``output_root``). The
workflow keeps that layout and builds it from three smaller files, so each is written once:

* ``configs/arms/<arm>.yaml``: what is trained, independent of country and budget;
* ``configs/experiments/<name>.yaml``: which arms, budgets, draws, seeds and epochs;
* ``configs/machines/<machine>.yaml``: throughput settings that never change a result.

The chip set is a path. Paths inside the repository are stored relative to it, so a result
reads the same on the JupyterHub and on the cluster, whose clone mirrors the hub's layout.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[3]
CONFIGS = REPO / "configs"

__all__ = ["CONFIGS", "REPO", "Cell", "compose", "load_arm", "load_experiment", "load_machine",
           "plan_cells", "repo_relative", "resolve_split"]


@dataclass(frozen=True)
class Cell:
    """One fit: an arm at a label budget, a support draw and a training seed."""

    arm: str
    pct: float
    draw: int
    seed: int

    @property
    def tag(self) -> str:
        return f"P{self.pct:g}_draw{self.draw}_seed{self.seed}"


def _yaml(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text())


def load_arm(name: str, configs: Path = CONFIGS) -> dict:
    """``configs/arms/<name>.yaml``, checked against the backbone registry."""
    from gfm4agri.benchmark.backbones import get_backbone

    arm = _yaml(Path(configs) / "arms" / f"{name}.yaml")
    spec = get_backbone(arm["model"]["backbone"])
    want = "cache" if spec.representation == "s2_monthly" else "raster"
    if arm["route"] != want:
        raise ValueError(f"arm {name}: route {arm['route']!r}, but {spec.name} needs {want!r}")
    return arm


def load_experiment(path: Path) -> dict:
    exp = _yaml(path)
    exp.setdefault("name", Path(path).stem)
    if exp.get("checkpoint", "val_loss") != "val_loss":
        raise ValueError(f"checkpoint {exp['checkpoint']!r}: only 'val_loss' is implemented")
    return exp


def load_machine(name: str, configs: Path = CONFIGS) -> dict:
    return _yaml(Path(configs) / "machines" / f"{name}.yaml")


def plan_cells(exp: dict, arms: list[str] | None = None) -> list[Cell]:
    """Every cell of ``exp``, arm by arm, budgets outermost within an arm."""
    unknown = sorted(set(arms or []) - set(exp["arms"]))
    if unknown:
        raise ValueError(f"arms {unknown} are not in the experiment ({exp['arms']})")
    chosen = [a for a in exp["arms"] if arms is None or a in arms]
    return [Cell(a, p, int(d), int(s)) for a in chosen for p in exp["budgets_pct"]
            for d in exp["draws"] for s in exp["seeds"]]


def repo_relative(path) -> str:
    """``path`` relative to the repository when it lies inside it, else absolute."""
    p = Path(path)
    p = (p if p.is_absolute() else Path.cwd() / p).resolve()
    try:
        return p.relative_to(REPO).as_posix()
    except ValueError:
        return str(p)


def resolve_split(chips: Path, split: str | None = None) -> Path:
    """The chip set's split directory: the only one, or the one named (with or without hash)."""
    base = Path(chips) / "splits"
    splits = sorted(p for p in base.iterdir() if (p / "split.json").exists()) if base.is_dir() else []
    names = [p.name for p in splits]
    if split is not None:
        match = [p for p in splits if split in (p.name, p.name.split("__")[0])]
        if len(match) != 1:
            raise ValueError(f"split {split!r} not found in {base}: {names}")
        return match[0]
    if len(splits) != 1:
        raise ValueError(f"{chips} holds {len(splits)} splits {names}; name one with --split")
    return splits[0]


def _number(x):
    return int(x) if float(x).is_integer() else float(x)


def compose(arm_name: str, arm: dict, exp: dict, machine: dict, chips: Path, split_dir: Path,
            cell: Cell, output_root: Path) -> dict:
    """The configuration dictionary of one cell, in the layout the fit functions read."""
    d, t = arm["data"], arm["trainer"]
    trainer = {"max_epochs": exp["epochs"], "precision": t["precision"],
               "log_every_n_steps": t["log_every_n_steps"]}
    if "accumulate_grad_batches" in t:
        trainer["accumulate_grad_batches"] = t["accumulate_grad_batches"]
    return {
        "run_name": arm_name,
        "seed": cell.seed,
        "data": {"root": repo_relative(chips), "split": repo_relative(split_dir),
                 "normalisation": d["normalisation"], "batch_size": d["batch_size"],
                 "num_workers": machine["num_workers"], "augment": d["augment"]},
        "label_budget": {"mode": "pct", "pct": _number(cell.pct), "draw_seed": cell.draw},
        "model": copy.deepcopy(arm["model"]),
        "trainer": trainer,
        "output_root": repo_relative(output_root),
    }
```

- [ ] **Step 7: Run the tests.** `poetry run pytest tests/test_pipeline_config.py -v`
  Expected: PASS (20 parametrised and 6 plain tests).

- [ ] **Step 8: Commit.**

```bash
git add configs/arms configs/experiments configs/machines src/gfm4agri/pipeline tests/fixtures tests/test_pipeline_config.py
git commit -m "Add arm, experiment and machine configs and their composition into a fit config

The composed configuration of every grid cell equals the full-Estonia grid's
configuration with its command-line overrides, checked against copies of the
earlier configs kept as test fixtures.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 6: Find or compute a feature cache

**Files:**
- Create: `src/gfm4agri/pipeline/cache.py`
- Test: `tests/test_pipeline_cache.py`

**Interfaces:**
- Consumes: `gfm4agri.benchmark.cached.{EMB_SUFFIX, IDENTITY, TRAIN_VARIANTS, variant_name}`, `gfm4agri.benchmark.segmentation_data.{manifest_sha256, read_split, split_list}`.
- Produces: `CacheDecision(action: str, cache_dir: Path, reason: str, mismatches: list[str])` with `action` in `{"reuse", "compute", "stop"}`; `cache_dir_for(root, backbone, chips) -> Path`; `expected_record(chips, split_dir, backbone, normalisation, precision) -> dict`; `resolve_cache(cache_root, scratch, backbone, chips, split_dir, normalisation, precision) -> CacheDecision`.

- [ ] **Step 1: Write the failing tests** `tests/test_pipeline_cache.py`:

```python
"""A feature cache is reused only when it matches the run and is complete, and never overwritten."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("terratorch")

from conftest import SYN_SPLIT  # noqa: E402
from gfm4agri.benchmark.cached import EMB_SUFFIX, IDENTITY, TRAIN_VARIANTS, variant_name  # noqa: E402
from gfm4agri.pipeline.cache import expected_record, resolve_cache  # noqa: E402

BACKBONE = "terramind_v1_large"


def _fake_cache(root, chips, *, drop=None, **override):
    """A cache directory as generate_embeddings leaves it, two layers, empty feature files."""
    split = chips / "splits" / SYN_SPLIT
    d = root / BACKBONE / chips.name
    ids = {s: (split / f"{n}_data.txt").read_text().split()
           for s, n in (("train", "training"), ("val", "validation"), ("test", "test"))}
    for k, f in TRAIN_VARIANTS:
        want = ids["train"] + (ids["val"] + ids["test"] if (k, f) == IDENTITY else [])
        folder = d / variant_name(k, f) / "layer_01"
        folder.mkdir(parents=True)
        for c in want:
            if c != drop:
                (folder / f"{c}{EMB_SUFFIX}").touch()
    rec = expected_record(chips, split, BACKBONE, "backbone", "16-mixed")
    record = {"backbone": rec["backbone"], "manifest_sha256": rec["manifest_sha256"],
              "split": {"dir": str(split), "config_hash": rec["split_config_hash"]},
              "normalisation": "backbone", "precision": "16-mixed",
              "train_variants": rec["train_variants"], "encoder_layers": [5, 11]} | override
    (d / "cache.json").write_text(json.dumps(record))
    return d


def _resolve(chips, cache_root, scratch):
    return resolve_cache(cache_root, scratch, BACKBONE, chips, chips / "splits" / SYN_SPLIT,
                         "backbone", "16-mixed")


def test_a_complete_matching_cache_is_reused(split_chipset, tmp_path):
    d = _fake_cache(tmp_path / "emb", split_chipset)
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "scratch")
    assert (r.action, r.cache_dir) == ("reuse", d)


def test_no_cache_is_computed_in_scratch(split_chipset, tmp_path):
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "scratch")
    assert (r.action, r.cache_dir) == ("compute", tmp_path / "scratch" / BACKBONE / "SYN_2021")


def test_a_partial_cache_is_resumed_in_place_when_scratch_is_the_cache_root(split_chipset, tmp_path):
    d = tmp_path / "emb" / BACKBONE / "SYN_2021" / "k0" / "layer_01"
    d.mkdir(parents=True)
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "emb")
    assert (r.action, r.cache_dir) == ("compute", tmp_path / "emb" / BACKBONE / "SYN_2021")


def test_missing_chips_are_computed_in_scratch(split_chipset, tmp_path):
    _fake_cache(tmp_path / "emb", split_chipset, drop="EE_00008_00000")
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "scratch")
    assert r.action == "compute" and r.cache_dir.parent.parent == tmp_path / "scratch"
    assert "1 chip encodings missing" in r.reason


def test_a_mismatching_cache_stops_the_run_rather_than_being_overwritten(split_chipset, tmp_path):
    _fake_cache(tmp_path / "emb", split_chipset, normalisation="chips")
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "emb")
    assert r.action == "stop" and r.mismatches == ["normalisation: cache 'chips', run 'backbone'"]


def test_a_mismatching_cache_is_left_alone_and_scratch_used(split_chipset, tmp_path):
    _fake_cache(tmp_path / "emb", split_chipset, split={"config_hash": "ffff"})
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "scratch")
    assert r.action == "compute" and r.cache_dir.parent.parent == tmp_path / "scratch"
    assert r.mismatches and r.mismatches[0].startswith("split_config_hash")
```

- [ ] **Step 2: Run them to see them fail.** `poetry run pytest tests/test_pipeline_cache.py -v`
  Expected: FAIL, `ModuleNotFoundError: No module named 'gfm4agri.pipeline.cache'`.

- [ ] **Step 3: Implement** `src/gfm4agri/pipeline/cache.py`:

```python
"""Find the feature cache of a cache-route arm, or decide where to compute it.

A cache is reused as it is only when its ``cache.json`` matches the run on every field that
changes the stored features (backbone, chip manifest, split, normalisation, precision, the
training variants) and every chip of the split is on disk. Otherwise it is computed into the
scratch root, which resumes whatever chips are already there. A cache that exists but does
not match is never written into: when the scratch root is the cache root, the run stops.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from gfm4agri.benchmark.cached import EMB_SUFFIX, IDENTITY, TRAIN_VARIANTS, variant_name
from gfm4agri.benchmark.segmentation_data import manifest_sha256, read_split, split_list

__all__ = ["CacheDecision", "cache_dir_for", "expected_record", "resolve_cache"]


@dataclass
class CacheDecision:
    action: str            # "reuse", "compute" or "stop"
    cache_dir: Path
    reason: str
    mismatches: list[str] = field(default_factory=list)


def cache_dir_for(root, backbone: str, chips) -> Path:
    """``<root>/<backbone>/<chip set name>``, the layout of ``data/embeddings``."""
    return Path(root) / backbone / Path(chips).name


def expected_record(chips, split_dir, backbone: str, normalisation: str, precision: str) -> dict:
    return {"backbone": backbone, "manifest_sha256": manifest_sha256(chips),
            "split_config_hash": read_split(split_dir)["config_hash"],
            "normalisation": normalisation, "precision": precision,
            "train_variants": [variant_name(k, f) for k, f in TRAIN_VARIANTS]}


def _mismatches(record: dict, expected: dict) -> list[str]:
    got = {"backbone": record.get("backbone"), "manifest_sha256": record.get("manifest_sha256"),
           "split_config_hash": (record.get("split") or {}).get("config_hash"),
           "normalisation": record.get("normalisation"), "precision": record.get("precision"),
           "train_variants": record.get("train_variants")}
    return [f"{k}: cache {got[k]!r}, run {v!r}" for k, v in expected.items() if got[k] != v]


def _missing(cache_dir: Path, record: dict, chips, split_dir) -> int:
    """Chip encodings the fit would read and the cache does not hold."""
    last = f"layer_{len(record['encoder_layers']) - 1:02d}"
    ids = {s: split_list(chips, s, split_dir).read_text().split() for s in ("train", "val", "test")}
    n = 0
    for k, f in TRAIN_VARIANTS:
        want = ids["train"] + (ids["val"] + ids["test"] if (k, f) == IDENTITY else [])
        folder = cache_dir / variant_name(k, f) / last
        have = ({p[: -len(EMB_SUFFIX)] for p in os.listdir(folder) if p.endswith(EMB_SUFFIX)}
                if folder.is_dir() else set())
        n += sum(c not in have for c in want)
    return n


def resolve_cache(cache_root, scratch, backbone: str, chips, split_dir, normalisation: str,
                  precision: str) -> CacheDecision:
    found = cache_dir_for(cache_root, backbone, chips)
    target = cache_dir_for(scratch, backbone, chips)
    same = Path(cache_root).resolve() == Path(scratch).resolve()
    record_path = found / "cache.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        bad = _mismatches(record, expected_record(chips, split_dir, backbone, normalisation,
                                                  precision))
        if bad:
            if same:
                return CacheDecision("stop", found, "the cache does not match the run", bad)
            return CacheDecision("compute", target, f"{found} does not match the run", bad)
        n = _missing(found, record, chips, split_dir)
        if n == 0:
            return CacheDecision("reuse", found, "complete and matching")
        if same:
            return CacheDecision("compute", found, f"{n} chip encodings missing, resumed in place")
        return CacheDecision("compute", target, f"{n} chip encodings missing in {found}")
    if found.exists() and same:
        return CacheDecision("compute", found, "partial cache without cache.json, resumed in place")
    return CacheDecision("compute", target, "no cache found")
```

- [ ] **Step 4: Run the tests.** `poetry run pytest tests/test_pipeline_cache.py -v` Expected: PASS (6).

- [ ] **Step 5: Check the real hub caches** (read only):

```bash
poetry run python - <<'PY'
import sys; sys.path.insert(0, "src")
from pathlib import Path
from gfm4agri.pipeline.cache import resolve_cache
ee = Path("data/eurocrops_chips/EE_2021"); sp = ee / "splits/blocks4_buf1600_seed0__6b0eb4cb"
for b in ("terramind_v1_large", "prithvi_eo_v2_600_tl", "thor_v1_large"):
    r = resolve_cache("data/embeddings", "data/embeddings", b, ee, sp, "backbone", "16-mixed")
    print(b, r.action, r.cache_dir, r.reason, r.mismatches)
PY
```
Expected: TerraMind and Prithvi `reuse`; THOR `compute ... no cache found`.

- [ ] **Step 6: Commit.**

```bash
git add src/gfm4agri/pipeline/cache.py tests/test_pipeline_cache.py
git commit -m "Reuse a feature cache only when it matches the run and is complete

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 7: The pilot chip set

**Files:**
- Create: `src/gfm4agri/data/chip_subset.py`, `scripts/hub/build_pilot_chipset.py`
- Test: `tests/test_chip_subset.py`
- Data (new directory, hub only): `data/eurocrops_chips/EE_2021_mini/`

**Interfaces:**
- Consumes: `gfm4agri.pipeline.config.resolve_split`.
- Produces: `choose_blocks(chips: pd.DataFrame, n_blocks: dict[str, int], seed: int) -> dict[str, list[str]]` (keys are `chips.csv` partitions `pool`, `val`, `test`); `write_subset(parent: Path, dest: Path, split_dir: Path, blocks: dict[str, list[str]], seed: int) -> Path`.

- [ ] **Step 1: Write the failing tests** `tests/test_chip_subset.py`:

```python
"""A pilot chip set is whole blocks of its parent and reads exactly like a chip set."""

from __future__ import annotations

import json
import os

import pandas as pd
import pytest

from conftest import SYN_SPLIT
from gfm4agri.data.chip_subset import choose_blocks, write_subset


def test_whole_blocks_are_drawn_per_partition_and_seeded(split_chipset):
    chips = pd.read_csv(split_chipset / "splits" / SYN_SPLIT / "chips.csv", dtype={"block_id": str})
    a = choose_blocks(chips, {"pool": 1, "val": 1, "test": 1}, seed=0)
    assert a == {"pool": ["0_0"], "val": ["1_0"], "test": ["2_0"]}
    assert a == choose_blocks(chips, {"pool": 1, "val": 1, "test": 1}, seed=0)
    with pytest.raises(ValueError, match="2 blocks wanted"):
        choose_blocks(chips, {"pool": 2}, seed=0)


def test_the_subset_is_a_chip_set_of_links_with_a_restricted_split(split_chipset):
    split = split_chipset / "splits" / SYN_SPLIT
    dest = split_chipset.parent / "SYN_2021_mini"
    write_subset(split_chipset, dest, split, {"pool": ["0_0"], "val": ["1_0"], "test": ["2_0"]}, 0)
    links = sorted(os.listdir(dest / "chips"))
    assert len(links) == 6 * 4 and all((dest / "chips" / f).is_symlink() for f in links)
    assert os.readlink(dest / "chips" / links[0]).startswith("../../SYN_2021/chips/")
    assert (dest / "manifest.json").read_bytes() == (split_chipset / "manifest.json").read_bytes()
    sub = dest / "splits" / SYN_SPLIT
    assert (sub / "training_data.txt").read_text().split() == ["EE_00000_00000", "EE_00001_00000"]
    rec = json.loads((sub / "split.json").read_text())
    assert rec["lists"] == {"training": 2, "validation": 2, "test": 2}
    assert rec["subset"]["parent"].endswith("SYN_2021")
    assert json.loads((dest / "subset.json").read_text())["blocks"]["val"] == ["1_0"]
    with pytest.raises(FileExistsError):
        write_subset(split_chipset, dest, split, {"pool": ["0_0"]}, 0)


def test_the_datamodule_reads_the_subset(split_chipset):
    pytest.importorskip("terratorch")
    from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule

    split = split_chipset / "splits" / SYN_SPLIT
    dest = split_chipset.parent / "SYN_2021_mini"
    write_subset(split_chipset, dest, split, {"pool": ["0_0"], "val": ["1_0"], "test": ["2_0"]}, 0)
    dm = EuroCropsSegDataModule(dest, "tessera_v1", normalisation="chips", batch_size=1,
                                num_workers=0, split_dir=dest / "splits" / SYN_SPLIT)
    dm.setup("fit")
    dm.setup("test")
    assert (len(dm.train_dataset), len(dm.val_dataset), len(dm.test_dataset)) == (2, 2, 2)
```

- [ ] **Step 2: Run them to see them fail.** `poetry run pytest tests/test_chip_subset.py -v`
  Expected: FAIL, `ModuleNotFoundError: No module named 'gfm4agri.data.chip_subset'`.

- [ ] **Step 3: Implement** `src/gfm4agri/data/chip_subset.py`:

```python
"""A smaller chip set cut from a country chip set: whole spatial blocks of every partition.

The subset is a chip set directory of its own, so the K-shot workflow runs on it exactly as on
a country. ``chips/`` holds relative links into the parent's chips; the manifest, the per-source
records and the chip-parcel table are links to the parent's, so the normalisation statistics
and the manifest hash are the parent's; the split keeps its name and holds the parent's lists
restricted to the chosen chips. ``subset.json`` records the parent, the split and the blocks.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["choose_blocks", "write_subset"]

#: ``chips.csv`` partition -> split list name.
LISTS = {"pool": "training", "val": "validation", "test": "test"}
CHIP_ID = re.compile(r"^[A-Z]{2}_\d+_\d+")


def choose_blocks(chips: pd.DataFrame, n_blocks: dict[str, int], seed: int) -> dict[str, list[str]]:
    """``n_blocks[partition]`` whole blocks per partition, drawn with ``seed``.

    Only blocks that can serve their partition are eligible: pool blocks holding a trainable
    parcel, validation and test blocks holding a labelled one.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for part, n in n_blocks.items():
        col = "trainable_parcels" if part == "pool" else "labelled_parcels"
        per_block = chips[chips["partition"] == part].groupby("block_id")[col].sum()
        eligible = sorted(per_block[per_block > 0].index)
        if len(eligible) < n:
            raise ValueError(f"{n} blocks wanted from {part}, {len(eligible)} eligible")
        out[part] = sorted(rng.choice(eligible, size=n, replace=False).tolist())
    return out


def _link(src: Path, dst: Path) -> None:
    os.symlink(os.path.relpath(src, dst.parent), dst)


def write_subset(parent: Path, dest: Path, split_dir: Path, blocks: dict[str, list[str]],
                 seed: int) -> Path:
    parent, dest, split_dir = Path(parent), Path(dest), Path(split_dir)
    if dest.exists():
        raise FileExistsError(f"{dest} exists; remove it first")
    chips = pd.read_csv(split_dir / "chips.csv", dtype={"block_id": str})
    chosen = {b for bs in blocks.values() for b in bs}
    keep = set(chips.loc[chips["block_id"].isin(chosen), "chip_id"])

    (dest / "chips").mkdir(parents=True)
    for f in sorted(os.listdir(parent / "chips")):
        m = CHIP_ID.match(f)
        if m and m.group(0) in keep:
            _link(parent / "chips" / f, dest / "chips" / f)
    for f in sorted(parent.iterdir()):
        if f.is_file() and f.suffix in (".json", ".parquet", ".sha256"):
            _link(f, dest / f.name)

    out = dest / "splits" / split_dir.name
    out.mkdir(parents=True)
    counts = {}
    for name in LISTS.values():
        src = split_dir / f"{name}_data.txt"
        if src.exists():
            ids = [c for c in src.read_text().split() if c in keep]
            (out / f"{name}_data.txt").write_text("\n".join(ids) + "\n")
            counts[name] = len(ids)
    chips[chips["chip_id"].isin(keep)].to_csv(out / "chips.csv", index=False)
    _link(split_dir / "buffer_parcels.npy", out / "buffer_parcels.npy")
    record = json.loads((split_dir / "split.json").read_text())
    record["lists"] = counts
    record["subset"] = {"parent": str(parent), "parent_split": split_dir.name, "blocks": blocks,
                        "seed": seed}
    (out / "split.json").write_text(json.dumps(record, indent=2))
    (dest / "subset.json").write_text(json.dumps({
        "parent": str(parent), "split": split_dir.name, "blocks": blocks, "seed": seed,
        "chips": counts, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    return dest
```

`scripts/hub/build_pilot_chipset.py`:

```python
"""Cut a pilot chip set from a country chip set: whole spatial blocks of every partition.

    poetry run python scripts/hub/build_pilot_chipset.py --parent data/eurocrops_chips/EE_2021 --name EE_2021_mini

The pilot is written beside its parent as links into it, so it costs no space; copy the parent
to the cluster before the pilot. An existing pilot directory is never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--parent", type=Path, required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--split", default=None, help="the parent's split, if it holds several")
    ap.add_argument("--train-blocks", type=int, default=3)
    ap.add_argument("--val-blocks", type=int, default=1)
    ap.add_argument("--test-blocks", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from gfm4agri.data.chip_subset import choose_blocks, write_subset
    from gfm4agri.pipeline.config import resolve_split

    parent = args.parent if args.parent.is_absolute() else REPO / args.parent
    split_dir = resolve_split(parent, args.split)
    chips = pd.read_csv(split_dir / "chips.csv", dtype={"block_id": str})
    blocks = choose_blocks(chips, {"pool": args.train_blocks, "val": args.val_blocks,
                                   "test": args.test_blocks}, args.seed)
    dest = write_subset(parent, parent.parent / args.name, split_dir, blocks, args.seed)
    print(f"{dest.relative_to(REPO)}: blocks {blocks}")
    print((dest / "subset.json").read_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests.** `poetry run pytest tests/test_chip_subset.py -v` Expected: PASS (3).

- [ ] **Step 5: Build the Estonian pilot** (new directory only):
  `poetry run python scripts/hub/build_pilot_chipset.py --parent data/eurocrops_chips/EE_2021 --name EE_2021_mini`
  Expected: about 80 chips (`chips` counts in `subset.json`), training > 0, validation > 0, test > 0. `ls data/eurocrops_chips/EE_2021_mini/chips | wc -l` is 9 times the chip count (eight files per chip plus the per-chip report).

- [ ] **Step 6: Commit** (code only; data is git-ignored).

```bash
git add src/gfm4agri/data/chip_subset.py scripts/hub/build_pilot_chipset.py tests/test_chip_subset.py
git commit -m "Add the pilot chip set: whole blocks of a country chip set, as links

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 8: The runner and its entry point

**Files:**
- Create: `src/gfm4agri/pipeline/runner.py`, `scripts/run_kshot.py`
- Delete: `scripts/seg/` (all), `configs/seg/` (remaining `_ee` files)
- Test: `tests/test_pipeline_runner.py`

**Interfaces:**
- Consumes: Task 5 (`Cell`, `compose`, `load_*`, `plan_cells`, `repo_relative`, `resolve_split`, `REPO`, `CONFIGS`), Task 6 (`resolve_cache`), Task 4 (`fit_end_to_end`), `gfm4agri.benchmark.cached.{fit_cached, generate_embeddings}`, `gfm4agri.benchmark.evaluation.FitPredictor`.
- Produces: `check_chipset(chips, split_dir, arms: dict[str, dict]) -> list[str]`; `run(experiment: Path, chips: Path, *, split=None, arms=None, machine="hub", cache_root=None, scratch=None, dry_run=False, configs=CONFIGS, results_root=REPO / "results") -> list[dict]`; `summarise(out_root: Path) -> pandas.DataFrame` (also writes `summary.csv`); constant `PREDICTIONS = "predictions_test.npz"`.

- [ ] **Step 1: Write the failing tests** `tests/test_pipeline_runner.py`:

```python
"""The runner plans cells, checks the chip set, fits, predicts, stamps and resumes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

pytest.importorskip("terratorch")
pytest.importorskip("rasterio")

from conftest import SYN_SPLIT  # noqa: E402
from gfm4agri.pipeline.runner import PREDICTIONS, run  # noqa: E402

TINY_RASTER = {"route": "raster",
               "data": {"normalisation": "chips", "batch_size": 2, "augment": True},
               "model": {"backbone": "tessera_v1", "lr": 1.0e-3, "weight_decay": 0.05,
                         "head_dropout": 0.2, "loss": "ce"},
               "trainer": {"precision": "32-true", "log_every_n_steps": 1}}
TINY_CACHE = {"route": "cache",
              "data": {"normalisation": "backbone", "batch_size": 2, "augment": True},
              "model": {"backbone": "terramind_v1_large", "lr": 1.0e-4, "weight_decay": 0.05,
                        "head_dropout": 0.1, "loss": "ce"},
              "trainer": {"precision": "16-mixed", "log_every_n_steps": 1,
                          "accumulate_grad_batches": 2}}


@pytest.fixture()
def configs(tmp_path: Path) -> Path:
    c = tmp_path / "configs"
    for sub in ("arms", "experiments", "machines"):
        (c / sub).mkdir(parents=True)
    (c / "arms" / "tiny_tessera.yaml").write_text(yaml.safe_dump(TINY_RASTER))
    (c / "arms" / "tiny_terramind.yaml").write_text(yaml.safe_dump(TINY_CACHE))
    (c / "experiments" / "tiny.yaml").write_text(yaml.safe_dump(
        {"name": "tiny", "arms": ["tiny_tessera", "tiny_terramind"], "budgets_pct": [100, 50],
         "draws": [0], "seeds": [0], "epochs": 1, "checkpoint": "val_loss"}))
    (c / "machines" / "hub.yaml").write_text(yaml.safe_dump(
        {"cache_root": str(tmp_path / "emb"), "num_workers": 0, "encode_num_workers": 0,
         "encode_batch_size": {"terramind_v1_large": 1}}))
    return c


def test_dry_run_lists_cells_and_cache_decisions(split_chipset, configs, tmp_path):
    rows = run(configs / "experiments" / "tiny.yaml", split_chipset, dry_run=True,
               configs=configs, results_root=tmp_path / "results")
    assert [(r["arm"], r["cell"], r["state"]) for r in rows] == [
        ("tiny_tessera", "P100_draw0_seed0", "fit"), ("tiny_tessera", "P50_draw0_seed0", "fit"),
        ("tiny_terramind", "P100_draw0_seed0", "fit"), ("tiny_terramind", "P50_draw0_seed0", "fit")]
    assert rows[2]["cache"] == "compute" and not (tmp_path / "results" / "tiny").exists()


def test_a_chip_set_missing_rasters_is_refused_before_any_compute(split_chipset, configs, tmp_path):
    (split_chipset / "chips" / "EE_00008_00000_tessera.tif").unlink()
    with pytest.raises(SystemExit, match="1 chips lack _tessera.tif"):
        run(configs / "experiments" / "tiny.yaml", split_chipset, dry_run=True,
            configs=configs, results_root=tmp_path / "results")


def test_a_raster_arm_runs_end_to_end_and_resumes(split_chipset, configs, tmp_path):
    exp = configs / "experiments" / "tiny.yaml"
    rows = run(exp, split_chipset, arms=["tiny_tessera"], configs=configs,
               results_root=tmp_path / "results")
    out = tmp_path / "results" / "tiny" / "SYN_2021"
    cell = out / "tiny_tessera" / "P50_draw0_seed0"
    r = json.loads((cell / "results.json").read_text())
    assert r["provenance"]["experiment"] == "tiny" and r["provenance"]["machine"] == "hub"
    assert len(r["provenance"]["git_commit"]) == 40
    z = np.load(cell / PREDICTIONS)
    assert sorted(z["ids"]) == ["EE_00008_00000", "EE_00009_00000"] and z["pred"].shape == (2, 32, 32)
    summary = pd.read_csv(out / "summary.csv")
    assert list(summary["pct"]) == [100, 50] and summary["macro_f1"].notna().all()
    assert [x["state"] for x in rows] == ["done", "done"]
    stamp = (cell / "results.json").stat().st_mtime_ns
    run(exp, split_chipset, arms=["tiny_tessera"], configs=configs, results_root=tmp_path / "results")
    assert (cell / "results.json").stat().st_mtime_ns == stamp          # skipped, not refitted
    (cell / PREDICTIONS).unlink()
    run(exp, split_chipset, arms=["tiny_tessera"], configs=configs, results_root=tmp_path / "results")
    assert (cell / PREDICTIONS).exists() and (cell / "results.json").stat().st_mtime_ns == stamp
```

- [ ] **Step 2: Run them to see them fail.** `poetry run pytest tests/test_pipeline_runner.py -v`
  Expected: FAIL, `ModuleNotFoundError: No module named 'gfm4agri.pipeline.runner'`.

- [ ] **Step 3: Implement** `src/gfm4agri/pipeline/runner.py`:

```python
"""Run a K-shot experiment on one chip set: every arm x budget x draw x seed.

Per arm: check that the chip set has what the arm reads; for a cache-route arm, reuse its
feature cache or compute it (:mod:`gfm4agri.pipeline.cache`); then fit every cell that is not
finished, write its test predictions and stamp its provenance. A cell is finished when it
holds ``results.json`` and ``predictions_test.npz``; a cell with only the first gets its
predictions without a refit, which a job-scoped cache needs because two-stage prediction reads
the cache. Results land in ``<results_root>/<experiment>/<chip set>/<arm>/<cell>/``.
"""

from __future__ import annotations

import datetime as dt
import gc
import json
import os
import socket
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from gfm4agri.pipeline.config import (CONFIGS, REPO, Cell, compose, load_arm, load_experiment,
                                      load_machine, plan_cells, repo_relative, resolve_split)

__all__ = ["PREDICTIONS", "check_chipset", "run", "summarise"]

PREDICTIONS = "predictions_test.npz"
METRICS = {"test/F1_Score": "macro_f1", "test/mIoU": "miou",
           "test/Pixel_Accuracy": "pixel_accuracy", "test/loss": "test_loss"}


def _say(msg: str) -> None:
    print(f"[kshot {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


def _lists(chips: Path, split_dir: Path) -> dict[str, list[str]]:
    from gfm4agri.benchmark.segmentation_data import split_list

    out = {}
    for s in ("train", "val", "test"):
        f = split_list(chips, s, split_dir)
        out[s] = f.read_text().split() if f.exists() else []
    return out


def check_chipset(chips: Path, split_dir: Path, arms: dict[str, dict]) -> list[str]:
    """What the arms would read and the chip set does not hold; empty when all is there."""
    from gfm4agri.benchmark.backbones import get_backbone

    problems = [f"{chips / f} missing" for f in ("manifest.json", "chip_parcels.parquet")
                if not (chips / f).exists()]
    lists = _lists(chips, split_dir)
    if not lists["test"]:
        problems.append(f"{split_dir} has no test list")
    ids = sorted({c for v in lists.values() for c in v})
    present = set(os.listdir(chips / "chips")) if (chips / "chips").is_dir() else set()
    for name, arm in arms.items():
        spec = get_backbone(arm["model"]["backbone"])
        if spec.representation == "s2_monthly":
            suffixes = ["_merged.tif"]
        else:
            sidecar = chips / f"{spec.representation}.json"
            if not sidecar.exists():
                problems.append(f"{name}: {sidecar.name} missing")
                continue
            suffixes = [json.loads(sidecar.read_text())["file_suffix"]]
        for sfx in suffixes + [".mask.tif", ".parcels.tif"]:
            miss = [c for c in ids if c + sfx not in present]
            if miss:
                problems.append(f"{name}: {len(miss)} chips lack {sfx}, e.g. {miss[:3]}")
    return problems


def _state(cell_dir: Path) -> str:
    if (cell_dir / "results.json").exists():
        return "done" if (cell_dir / PREDICTIONS).exists() else "predict"
    return "fit"


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(REPO), *args], capture_output=True, text=True).stdout.strip()


def _stamp(results_path: Path, provenance: dict) -> None:
    r = json.loads(results_path.read_text())
    r["provenance"] = provenance
    results_path.write_text(json.dumps(r, indent=2))


def _predict(cell_dir: Path, test_ids: list[str], num_workers: int) -> None:
    import torch

    from gfm4agri.benchmark.evaluation import FitPredictor

    p = FitPredictor(cell_dir)
    out = p.predict(test_ids, num_workers=num_workers)
    keys = sorted(out)
    tmp = cell_dir / (PREDICTIONS + ".tmp")
    with open(tmp, "wb") as f:
        np.savez_compressed(f, ids=np.array(keys), pred=np.stack([out[k][0] for k in keys]),
                            mask=np.stack([out[k][1] for k in keys]))
    tmp.replace(cell_dir / PREDICTIONS)
    del p
    gc.collect()
    torch.cuda.empty_cache()


def _epochs(cell_dir: Path) -> tuple[float | None, float | None]:
    """Epoch of the lowest validation loss and of the highest validation Macro-F1."""
    logs = sorted(cell_dir.glob("logs/version_*/metrics.csv"))
    if not logs:
        return None, None
    m = pd.read_csv(logs[-1])
    best = []
    for col, pick in (("val/loss", "idxmin"), ("val/F1_Score", "idxmax")):
        v = m.dropna(subset=[col]) if col in m else None
        best.append(None if v is None or v.empty else int(v.loc[getattr(v[col], pick)(), "epoch"]))
    return best[0], best[1]


def _row(results_path: Path) -> dict:
    r = json.loads(results_path.read_text())
    lb, m, prov = r["label_budget"], r["test_metrics"], r.get("provenance", {})
    ckpt, best_f1 = _epochs(results_path.parent)
    return {"arm": r["run"], "pct": lb["pct"], "draw": lb["draw_seed"], "seed": r["seed"],
            **{v: m.get(k) for k, v in METRICS.items()},
            "train_chips": r["chips"]["train"],
            "parcels": sum(v["drawn"] for v in (lb.get("support") or {}).values()),
            "ckpt_epoch": ckpt, "best_val_f1_epoch": best_f1,
            "fit_h": round(r["fit_seconds"] / 3600, 3),
            "machine": prov.get("machine"), "git_commit": prov.get("git_commit")}


def summarise(out_root: Path) -> pd.DataFrame:
    """One row per finished cell under ``out_root``, written to ``summary.csv``."""
    rows = [_row(p) for p in sorted(Path(out_root).glob("*/P*_draw*_seed*/results.json"))]
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.sort_values(["arm", "pct", "draw", "seed"], ascending=[True, False, True, True])
    tmp = Path(out_root) / f"summary.csv.{os.getpid()}"   # jobs of other arms may write at once
    df.to_csv(tmp, index=False)
    tmp.replace(Path(out_root) / "summary.csv")
    return df


def run(experiment: Path, chips: Path, *, split: str | None = None,
        arms: list[str] | None = None, machine: str = "hub", cache_root: Path | None = None,
        scratch: Path | None = None, dry_run: bool = False, configs: Path = CONFIGS,
        results_root: Path = REPO / "results") -> list[dict]:
    exp = load_experiment(experiment)
    mach = load_machine(machine, configs)
    chips = Path(chips) if Path(chips).is_absolute() else Path.cwd() / chips
    split_dir = resolve_split(chips, split)
    cells = plan_cells(exp, arms)
    arm_cfgs = {a: load_arm(a, configs) for a in dict.fromkeys(c.arm for c in cells)}
    problems = check_chipset(chips, split_dir, arm_cfgs)
    if problems:
        raise SystemExit("chip set check failed:\n  " + "\n  ".join(problems))

    out_root = Path(results_root) / exp["name"] / chips.name
    root = Path(cache_root or mach["cache_root"])
    root = root if root.is_absolute() else REPO / root
    scratch = Path(scratch) if scratch else root
    test_ids = _lists(chips, split_dir)["test"]
    provenance = {"experiment": exp["name"], "chip_set": repo_relative(chips),
                  "split": split_dir.name, "git_commit": _git("rev-parse", "HEAD"),
                  "git_dirty": bool(_git("status", "--porcelain")), "machine": machine,
                  "host": socket.gethostname()}
    _say(f"{exp['name']} on {chips.name} ({split_dir.name}), {len(cells)} cells, "
         f"machine {machine}, commit {provenance['git_commit'][:8]}"
         + (" (dirty tree)" if provenance["git_dirty"] else ""))

    rows = []
    for arm_name, arm in arm_cfgs.items():
        mine = [c for c in cells if c.arm == arm_name]
        todo = [c for c in mine if _state(out_root / arm_name / c.tag) != "done"]
        backbone = arm["model"]["backbone"]
        decision = None
        if arm["route"] == "cache":
            from gfm4agri.pipeline.cache import resolve_cache

            decision = resolve_cache(root, scratch, backbone, chips, split_dir,
                                     arm["data"]["normalisation"], arm["trainer"]["precision"])
            _say(f"{arm_name}: cache {decision.action} {decision.cache_dir} ({decision.reason})"
                 + "".join(f"\n    {m}" for m in decision.mismatches))
            if decision.action == "stop" and todo:
                raise SystemExit(f"{arm_name}: {decision.cache_dir} does not match the run and "
                                 "would be written into; pass --scratch to compute elsewhere")
        _say(f"{arm_name}: {len(mine) - len(todo)} of {len(mine)} cells finished")
        if dry_run:
            rows += [{"arm": arm_name, "cell": c.tag, "state": _state(out_root / arm_name / c.tag),
                      "cache": decision.action if decision else None} for c in mine]
            continue
        if todo and decision is not None and decision.action == "compute":
            from gfm4agri.benchmark.cached import generate_embeddings

            _say(f"{arm_name}: encoding into {decision.cache_dir}")
            generate_embeddings(backbone, chips, decision.cache_dir,
                                normalisation=arm["data"]["normalisation"],
                                batch_size=mach["encode_batch_size"][backbone],
                                num_workers=mach["encode_num_workers"],
                                precision=arm["trainer"]["precision"], split_dir=split_dir)
            gc.collect()
        for c in todo:
            cell_dir = out_root / arm_name / c.tag
            if _state(cell_dir) == "fit":
                cfg = compose(arm_name, arm, exp, mach, chips, split_dir, c, out_root)
                _say(f"{arm_name} {c.tag}: fit")
                if arm["route"] == "cache":
                    from gfm4agri.benchmark.cached import fit_cached

                    fit_cached(cfg, decision.cache_dir, pct=c.pct, draw_seed=c.draw,
                               output_root=out_root, num_workers=mach["num_workers"])
                else:
                    from gfm4agri.benchmark.fit import fit_end_to_end

                    fit_end_to_end(cfg)
                _stamp(cell_dir / "results.json", provenance)
                gc.collect()
            _say(f"{arm_name} {c.tag}: test predictions")
            _predict(cell_dir, test_ids, mach["num_workers"])
        rows += [{"arm": arm_name, "cell": c.tag, "state": _state(out_root / arm_name / c.tag),
                  "cache": decision.action if decision else None} for c in mine]
    if not dry_run:
        df = summarise(out_root)
        _say(f"summary of {len(df)} cells -> {out_root / 'summary.csv'}")
    return rows
```

`Cell` is imported for the type of `plan_cells`' items only; drop it from the import if the linter flags it.

`scripts/run_kshot.py`:

```python
"""The K-shot workflow: every arm x budget x draw x seed of an experiment, on one chip set.

    poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_mini
    poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021 --dry-run
    poetry run python scripts/run_kshot.py -e ... --chips ... --arms thor_v1_large --machine cluster --scratch /local/$SLURM_JOB_ID
    poetry run python scripts/run_kshot.py -e ... --chips ... --summarise

Results land in results/<experiment>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/, and a finished cell
is skipped, so a run resumes. On the hub, run it detached with its log beside the results:

    mkdir -p results/kshot/EE_2021_mini/logs
    setsid nohup poetry run python -u scripts/run_kshot.py -e configs/experiments/kshot.yaml \
        --chips data/eurocrops_chips/EE_2021_mini > results/kshot/EE_2021_mini/logs/hub_$(date +%m%d_%H%M).log 2>&1 &
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-e", "--experiment", type=Path, required=True)
    ap.add_argument("--chips", type=Path, required=True, help="the chip set directory")
    ap.add_argument("--split", default=None, help="the chip set's split, if it holds several")
    ap.add_argument("--arms", nargs="+", default=None, help="restrict to these arms")
    ap.add_argument("--machine", default="hub", choices=["hub", "cluster"])
    ap.add_argument("--cache-root", type=Path, default=None,
                    help="where caches are looked up; default from the machine profile")
    ap.add_argument("--scratch", type=Path, default=None,
                    help="where missing caches are computed; default the cache root")
    ap.add_argument("--dry-run", action="store_true", help="plan and check, compute nothing")
    ap.add_argument("--summarise", action="store_true", help="only collect finished cells")
    args = ap.parse_args()

    from gfm4agri.pipeline.config import load_experiment
    from gfm4agri.pipeline.runner import run, summarise

    if args.summarise:
        out = REPO / "results" / load_experiment(args.experiment)["name"] / args.chips.name
        print(summarise(out).to_string(index=False))
        return
    rows = run(args.experiment, args.chips, split=args.split, arms=args.arms,
               machine=args.machine, cache_root=args.cache_root, scratch=args.scratch,
               dry_run=args.dry_run)
    for r in rows:
        print(f"  {r['arm']:<22} {r['cell']:<18} {r['state']:<8} {r['cache'] or ''}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run the tests.** `poetry run pytest tests/test_pipeline_runner.py -v`
  Expected: PASS (3). The end-to-end test takes about a minute (two one-epoch fits on CPU or GPU).

- [ ] **Step 5: Remove what the runner replaces.**

```bash
git rm -q -r scripts/seg configs/seg
grep -rn "scripts/seg\|configs/seg" --include=*.py --include=*.sh src scripts tests | head
```
Expected: no hits (docs are updated in Task 12).

- [ ] **Step 6: Full suite.** `poetry run pytest -q` Expected: all pass, THOR tests skipped only if THOR is not installed.

- [ ] **Step 7: Commit.**

```bash
git add -A
git commit -m "Add the K-shot runner and scripts/run_kshot.py; remove scripts/seg and configs/seg

One entry point runs every arm, budget, draw and seed of an experiment on any
chip set, reusing a matching feature cache or computing it, writing the test
predictions of every cell and stamping its provenance.

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 9: Verify on the hub

No code changes unless a check fails (then fix, with a test, and commit).

- [ ] **Step 1: Dry runs.**

```bash
poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021 --dry-run
poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_mini --dry-run
```
Expected: 15 cells each, all `fit`; on `EE_2021` TerraMind and Prithvi `reuse`, THOR `compute ... no cache found`; on `EE_2021_mini` all three `compute`.

- [ ] **Step 2: Check the GPU is free** (`nvidia-smi`); another process holding memory (9.8 GB was held on 9 October) must be stopped by the user or waited out.

- [ ] **Step 3: The full experiment on the pilot.**

```bash
mkdir -p results/kshot/EE_2021_mini/logs
setsid nohup poetry run python -u scripts/run_kshot.py -e configs/experiments/kshot.yaml \
  --chips data/eurocrops_chips/EE_2021_mini > results/kshot/EE_2021_mini/logs/hub_$(date +%m%d_%H%M).log 2>&1 &
```
Monitor the log until `summary of 15 cells`. Expected duration under 2 hours.

- [ ] **Step 4: Check the outputs.**

```bash
ls results/kshot/EE_2021_mini/*/P*/results.json | wc -l          # 15
ls results/kshot/EE_2021_mini/*/P*/predictions_test.npz | wc -l   # 15
cat results/kshot/EE_2021_mini/summary.csv
ls data/embeddings/*/EE_2021_mini/cache.json                     # three new caches, nothing else new
git status --short                                               # clean: nothing written outside results/ and data/
```

---

### Task 10: Cluster scripts

**Files:**
- Create: `scripts/cluster/{setup_env.sh, prefetch_weights.py, kshot.sbatch, submit.sh, push_data.sh, pull_results.sh}`

- [ ] **Step 1: `scripts/cluster/setup_env.sh`.**

```bash
#!/bin/bash -l
# One-time set-up of the Poetry environment on the UT HPC cluster, from the repository root on the
# login node (a login shell, so `module` works):
#
#     bash -l scripts/cluster/setup_env.sh
#
# Python 3.11 comes from a conda-forge environment created with the miniconda3 module (the cluster has
# no python/3.11 module), Poetry 2.2.1 from its installer, the packages from poetry.lock, and THOR outside
# the lock as on the hub. The encoder weights are downloaded here, on the login node, which has direct
# internet; compute nodes reach it only through the UT proxy. Each step skips what is in place.
set -euo pipefail
cd "$(dirname "$0")/../.."
PY_ENV="$HOME/envs/py311"
export PATH="$HOME/.local/bin:$PATH"

module load miniconda3/25.7
[ -x "$PY_ENV/bin/python" ] || conda create -y -p "$PY_ENV" -c conda-forge --override-channels python=3.11
if ! poetry --version 2>/dev/null | grep -q "2\.2\.1"; then
  curl -sSL https://install.python-poetry.org | "$PY_ENV/bin/python" - --version 2.2.1
fi
poetry env use "$PY_ENV/bin/python"
poetry install
bash scripts/env/install_thor.sh
poetry run python scripts/cluster/prefetch_weights.py
poetry run pytest -q
echo "environment ready: $(poetry env info --path)"
```

- [ ] **Step 2: `scripts/cluster/prefetch_weights.py`.**

```python
"""Download the pretrained weights of the cache-route arms into the local Hugging Face cache.

Run on the cluster's login node by setup_env.sh, so jobs never download. Building each arm's
end-to-end model once triggers the download; nothing is trained.

    poetry run python scripts/cluster/prefetch_weights.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def main() -> None:
    from gfm4agri.benchmark.segmentation import build_task
    from gfm4agri.chips.s2_monthly import BAND_NAMES
    from gfm4agri.pipeline.config import CONFIGS, load_arm, load_experiment

    exp = load_experiment(CONFIGS / "experiments" / "kshot.yaml")
    for name in exp["arms"]:
        arm = load_arm(name)
        if arm["route"] != "cache":
            continue
        backbone = arm["model"]["backbone"]
        print(f"{backbone}: building once to fetch its weights", flush=True)
        build_task(backbone, num_classes=2, class_names=["a", "b"], bands=list(BAND_NAMES),
                   n_timesteps=12, ignore_index=-1)
    print("weights cached")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: `scripts/cluster/kshot.sbatch`.**

```bash
#!/bin/bash -l
# One arm of a K-shot experiment on one chip set, as a Slurm job on the UT HPC itc-gpu partition.
# Submitted by scripts/cluster/submit.sh, which sets the name, log, CPUs, memory and time:
#
#     sbatch --export=ALL,EXPERIMENT=kshot,CHIPS=EE_2021,ARM=terramind_v1_large scripts/cluster/kshot.sbatch
#
# A feature cache found under data/embeddings/ is used as it is; a missing one is computed into the
# node's local NVMe, /local/$SLURM_JOB_ID, and deleted when the job ends, fails or is cancelled.
#SBATCH -A itc-tech
#SBATCH -q research
#SBATCH -p itc-gpu
#SBATCH --gres=gpu:1
set -uo pipefail
cd "$SLURM_SUBMIT_DIR"
export HTTP_PROXY=http://proxy.utwente.nl:3128 HTTPS_PROXY=http://proxy.utwente.nl:3128
export http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY
export PATH="$HOME/.local/bin:$PATH"
SCRATCH="/local/${SLURM_JOB_ID}"
mkdir -p "$SCRATCH"
trap 'rm -rf "$SCRATCH"' EXIT
trap 'exit 143' TERM

echo "=== $(date '+%F %T') job $SLURM_JOB_ID on $(hostname -s): $EXPERIMENT $CHIPS $ARM, commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
poetry run python -u scripts/run_kshot.py -e "configs/experiments/$EXPERIMENT.yaml" \
  --chips "data/eurocrops_chips/$CHIPS" --arms "$ARM" --machine cluster --scratch "$SCRATCH"
status=$?
echo "=== $(date '+%F %T') exit $status"
exit $status
```

- [ ] **Step 4: `scripts/cluster/submit.sh`.**

```bash
#!/usr/bin/env bash
# Submit one Slurm job per arm of a K-shot experiment on one chip set, from the repository root on the
# cluster's login node:
#
#     bash scripts/cluster/submit.sh kshot EE_2021                        # every arm of the experiment
#     bash scripts/cluster/submit.sh kshot EE_2021_mini tessera_v1        # some arms
#     TIME=0-04:00:00 bash scripts/cluster/submit.sh kshot EE_2021_mini   # a shorter limit for the pilot
#     DRY_RUN=1 bash scripts/cluster/submit.sh kshot EE_2021             # print the sbatch commands only
#
# Logs go to results/<experiment>/<chips>/logs/<arm>_<job id>.log.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PATH="$HOME/.local/bin:$PATH"
EXPERIMENT=${1:?experiment name, e.g. kshot}
CHIPS=${2:?chip set name under data/eurocrops_chips, e.g. EE_2021}
shift 2
ARMS=("$@")
if [ ${#ARMS[@]} -eq 0 ]; then
  mapfile -t ARMS < <(poetry run python -c 'import sys, yaml; print("\n".join(yaml.safe_load(open(sys.argv[1]))["arms"]))' \
    "configs/experiments/$EXPERIMENT.yaml")
fi
[ -d "data/eurocrops_chips/$CHIPS" ] || { echo "no data/eurocrops_chips/$CHIPS on this machine" >&2; exit 1; }
[ -z "$(git status --porcelain)" ] || echo "warning: local changes; results will record a dirty tree" >&2
TIME=${TIME:-2-00:00:00}; CPUS=${CPUS:-32}; MEM=${MEM:-120G}
LOGS="results/$EXPERIMENT/$CHIPS/logs"
mkdir -p "$LOGS"
for arm in "${ARMS[@]}"; do
  cmd=(sbatch -J "$EXPERIMENT-$arm" -c "$CPUS" --mem="$MEM" -t "$TIME" -o "$LOGS/${arm}_%j.log"
       --export="ALL,EXPERIMENT=$EXPERIMENT,CHIPS=$CHIPS,ARM=$arm" scripts/cluster/kshot.sbatch)
  if [ -n "${DRY_RUN:-}" ]; then echo "${cmd[*]}"; else "${cmd[@]}"; fi
done
```

- [ ] **Step 5: `scripts/cluster/push_data.sh`** (run on the hub).

```bash
#!/usr/bin/env bash
# Copy data directories from the JupyterHub to the same place in the cluster's clone, resumably:
#
#     bash scripts/cluster/push_data.sh eurocrops_chips/EE_2021 eurocrops_chips/EE_2021_mini eurocrops/parquet eurocrops/vector
#
# Paths are relative to data/. Links are copied as links, so a pilot chip set stays links into its
# parent, which must be copied too. Log and pid files stay behind. Each copy is checked against the
# free space on the cluster first.
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=${HOST:-utwente-hpc}
REMOTE=${REMOTE:-ExplainedGMF4Agri}
for rel in "$@"; do
  src="data/${rel%/}"
  [ -e "$src" ] || { echo "no $src" >&2; exit 1; }
  need=$(du -sk "$src" | cut -f1)
  free=$(ssh "$HOST" "df -k --output=avail ~ | tail -1")
  echo "=== $src: $((need / 1048576)) GiB to copy, $((free / 1048576)) GiB free on $HOST"
  [ "$need" -lt "$free" ] || { echo "not enough space on $HOST for $src" >&2; exit 1; }
  ssh "$HOST" "mkdir -p '$REMOTE/$src'"
  rsync -a --partial --info=progress2 --exclude='*.log' --exclude='*.pid' "$src/" "$HOST:$REMOTE/$src/"
done
```

- [ ] **Step 6: `scripts/cluster/pull_results.sh`** (run on the hub).

```bash
#!/usr/bin/env bash
# Bring an experiment's results back from the cluster to the JupyterHub:
#
#     bash scripts/cluster/pull_results.sh kshot                  # every chip set
#     bash scripts/cluster/pull_results.sh kshot EE_2021          # one chip set
#     DEST=results/_cluster bash scripts/cluster/pull_results.sh kshot EE_2021_mini   # elsewhere
set -euo pipefail
cd "$(dirname "$0")/../.."
HOST=${HOST:-utwente-hpc}
REMOTE=${REMOTE:-ExplainedGMF4Agri}
EXPERIMENT=${1:?experiment name, e.g. kshot}
SUB="$EXPERIMENT${2:+/$2}"
DEST="${DEST:-results}/$SUB"
mkdir -p "$DEST"
rsync -a --info=progress2 "$HOST:$REMOTE/results/$SUB/" "$DEST/"
```

- [ ] **Step 7: Syntax checks.**

```bash
chmod +x scripts/cluster/*.sh
for f in scripts/cluster/*.sh scripts/cluster/kshot.sbatch; do bash -n "$f" && echo "ok $f"; done
command -v shellcheck && shellcheck scripts/cluster/*.sh scripts/cluster/kshot.sbatch
poetry run python -m py_compile scripts/cluster/prefetch_weights.py
```
Expected: `ok` for every file; shellcheck clean or absent.

- [ ] **Step 8: Commit.**

```bash
git add scripts/cluster
git commit -m "Add the cluster scripts: environment, weights, sbatch job, submit, data push, results pull

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 11: Set up the cluster and verify there

- [ ] **Step 1: Deploy key** (on the cluster, then a user action):

```bash
ssh utwente-hpc 'test -f ~/.ssh/id_ed25519_github || ssh-keygen -q -t ed25519 -N "" -C "utwente-hpc read-only" -f ~/.ssh/id_ed25519_github
grep -q "Host github.com" ~/.ssh/config 2>/dev/null || printf "Host github.com\n  IdentityFile ~/.ssh/id_ed25519_github\n  IdentitiesOnly yes\n" >> ~/.ssh/config
chmod 600 ~/.ssh/config; cat ~/.ssh/id_ed25519_github.pub'
```
The user adds the printed key at GitHub, repository Settings, Deploy keys, without write access. If port 22 to GitHub is blocked, add `Hostname ssh.github.com` and `Port 443` to that block.

- [ ] **Step 2: Push the branch** (from the hub): `git push -u origin restructure`.

- [ ] **Step 3: Clone and set up** (on the cluster login node):

```bash
ssh utwente-hpc 'git clone -b restructure git@github.com:davidrers/ExplainedGMF4Agri.git ~/ExplainedGMF4Agri'
ssh utwente-hpc 'cd ~/ExplainedGMF4Agri && bash -l scripts/cluster/setup_env.sh' 2>&1 | tail -20
```
Expected: `pytest` passes; `environment ready`.

- [ ] **Step 4: Copy the data** (from the hub; the EE chip set takes about 45 to 60 minutes):

```bash
setsid nohup bash scripts/cluster/push_data.sh eurocrops_chips/EE_2021 eurocrops_chips/EE_2021_mini \
  eurocrops/parquet eurocrops/vector > results/push_data_$(date +%m%d_%H%M).log 2>&1 &
```
Expected: the space check passes (about 300 GB of 1 TB), rsync completes; `ssh utwente-hpc 'ls ~/ExplainedGMF4Agri/data/eurocrops_chips/EE_2021_mini/chips | head -2; readlink -f ~/ExplainedGMF4Agri/data/eurocrops_chips/EE_2021_mini/chips/$(ls ~/ExplainedGMF4Agri/data/eurocrops_chips/EE_2021_mini/chips | head -1)'` resolves into `EE_2021/chips`.

- [ ] **Step 5: Dry run and submit the pilot** (on the cluster):

```bash
ssh utwente-hpc 'cd ~/ExplainedGMF4Agri && poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_mini --machine cluster --dry-run'
ssh utwente-hpc 'cd ~/ExplainedGMF4Agri && TIME=0-06:00:00 bash scripts/cluster/submit.sh kshot EE_2021_mini'
```
Expected: five job ids. Follow with `ssh utwente-hpc 'squeue -u $USER'` and the logs in `results/kshot/EE_2021_mini/logs/`.

- [ ] **Step 6: Pull and compare** (on the hub, after the jobs end):

```bash
DEST=results/_cluster_check bash scripts/cluster/pull_results.sh kshot EE_2021_mini
poetry run python - <<'PY'
import pandas as pd
hub = pd.read_csv("results/kshot/EE_2021_mini/summary.csv")
cl = pd.read_csv("results/_cluster_check/kshot/EE_2021_mini/summary.csv")
m = hub.merge(cl, on=["arm", "pct", "draw", "seed"], suffixes=("_hub", "_cluster"))
print(m[["arm", "pct", "macro_f1_hub", "macro_f1_cluster", "machine_hub", "machine_cluster"]])
PY
```
Expected: 15 rows on both sides, every cell has a prediction file, Macro-F1 of the same order on both machines (not identical). Each job's summary only covers what had finished when it ended; run `poetry run python scripts/run_kshot.py -e configs/experiments/kshot.yaml --chips data/eurocrops_chips/EE_2021_mini --summarise` on the cluster first if the pulled summary has fewer than 15 rows.

- [ ] **Step 7: Check `/local` was cleaned**: `ssh utwente-hpc 'sacct -u $USER -S today --format=JobID,NodeList,State'` and, for one finished job, confirm nothing of it remains: `ssh utwente-hpc 'srun -p itc-gpu -w <node> --gres=gpu:0 -t 1 ls /local | head'` (optional; skip if the partition is busy).

---

### Task 12: Documentation

Agreed edits only (spec section 8). Use the humanizer register of the existing docs: formal, no dashes as punctuation.

- [ ] **Step 1: `CLAUDE.md`.**
  - *Current state*: add `EE_2021_mini` (whole blocks of `EE_2021`, links) to the chip set table; arms implemented are the five of `configs/arms/`, THOR at 160 m confirmed on 9 October 2026, AlphaEarth implemented; the 2 October paragraph keeps its open items but no longer lists THOR's token size as open.
  - *Running the pipeline*: hub chip stages under `scripts/hub/`; `scripts/hub/build_pilot_chipset.py`; the runner command with `--dry-run`; the cluster loop (push data, `git pull`, `submit.sh`, `pull_results.sh`) pointing to `docs/utwente_hpc.md`; results under `results/<experiment>/<chip set>/`; earlier runs stay in `results/seg*`.
  - *Working environments*: a third short entry, the UT HPC cluster, with clone path `~/ExplainedGMF4Agri`, environment from `scripts/cluster/setup_env.sh`, and the rule that code is never edited there.
  - *Repository layout*: the tree of spec section 4, with `experiments/` and its cleanup rule in one line.
- [ ] **Step 2: `docs/phase1/pipeline.md`.**
  - Section 1 stage table: code paths `scripts/hub/...`; row 9 "The fit" becomes `scripts/run_kshot.py` with the two routes; output `results/<experiment>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/`.
  - Section 6: a subsection "The pilot chip set" describing `EE_2021_mini` (blocks, links, parent statistics, `subset.json`).
  - Section 9: the runner, the composition of arm, experiment and machine, the cache lookup with its match check, and the job-scoped cache on the cluster.
  - Section 10: the test predictions are written per cell because a job-scoped cache is gone after the job; the pilot is now a block subset with a test partition, so the "pilot is not a Phase 1 split" gap applies only to the archived 12-chip pilot.
  - Section 11: the results layout; the runs of 2 October remain under `results/seg*`; the pilot runs of Task 9 and Task 11.
- [ ] **Step 3: `docs/utwente_hpc.md`.** Replace the paragraph on `scripts/seg/run_*.sh` (around line 260) with a section "Running this project's workflow": the one-time set-up, the loop table of spec section 6.3, the job anatomy of 6.4, and where logs and results go.
- [ ] **Step 4: `README.md`.** Update any layout or command that names moved paths (`scripts/data`, `scripts/seg`, `configs/seg`, `notebooks/terratorch`).
- [ ] **Step 5: Check.** `grep -rn "scripts/seg\|scripts/data\|configs/seg\|notebooks/terratorch" CLAUDE.md README.md docs/phase1/pipeline.md docs/utwente_hpc.md`
  Expected: only historical mentions that say "at the snapshot commit".
- [ ] **Step 6: Commit.**

```bash
git add CLAUDE.md README.md docs/phase1/pipeline.md docs/utwente_hpc.md
git commit -m "Document the restructured repository and the hub, git and cluster workflow

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

---

### Task 13: Merge

- [ ] **Step 1:** `poetry run pytest -q` on the hub; all pass.
- [ ] **Step 2:** `git checkout main && git merge --no-ff restructure -m "Merge the repository restructure and the K-shot workflow" && git push origin main`.
- [ ] **Step 3:** On the cluster: `ssh utwente-hpc 'cd ~/ExplainedGMF4Agri && git fetch && git checkout main && git pull --ff-only && git log --oneline -1'`.
- [ ] **Step 4:** Update session memory (`repo-restructure.md`): done, merged, where things are; the full Estonia cluster run is the user's call.
