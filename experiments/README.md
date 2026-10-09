# Experiments

One folder per experiment, named `<YYYY-MM-DD>_<name>`, started by copying `_template/`. An experiment is anything
run on request that is not the main workflow itself: an ablation, a variant, a new arm on trial. **The main
workflow (`src/`, `configs/{arms,experiments,machines}`, `scripts/`, `tests/`) is never edited for an experiment.**
Whatever an experiment needs beyond it lives in its folder:

| File | Holds |
|---|---|
| `README.md` | question, what it varies, how to run it, results path, outcome |
| `experiment.yaml` | the spec the runner reads: arms, budgets, draws, seeds, epochs, and optionally `name` |
| `arms/<arm>.yaml` | arms of this experiment; found before `configs/arms/`, so a copy here varies a core arm |
| `run.py` and module copies | only when code must change: the runner with this folder's copies swapped in |

`scripts/run_kshot.py -e experiments/<folder>/experiment.yaml` runs it on the hub, and
`scripts/cluster/submit.sh experiments/<folder>/experiment.yaml <set>` on the cluster (using `run.py` when the
folder has one). Results go to `results/<name>/<set>/`.

**Adding to the main workflow** is a separate decision of the user: what an experiment proved is then moved from
its folder into the core, with tests and documentation.

**Cleanup rule.** An experiment and its results may be deleted once its conclusion is written into
`docs/phase1/pipeline.md`. Folders from before 9 October 2026 are records: they ran against the code of their
commit, named in their README.
