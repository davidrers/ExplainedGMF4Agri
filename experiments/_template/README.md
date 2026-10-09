# <Title of the study>, <date>

Copy this folder to `experiments/<YYYY-MM-DD>_<name>/` to start an experiment; nothing in the main workflow
(`src/`, `configs/`, `scripts/`, `tests/`) is edited for it.

**Question.** What the experiment asks.

**What it varies.** Which arms, budgets, chip set; what differs from the main workflow and where that difference
lives in this folder (`arms/`, `run.py` and its module copies).

**How to run.**

    poetry run python scripts/run_kshot.py -e experiments/<folder>/experiment.yaml --chips data/eurocrops_chips/<set>   # hub
    bash scripts/cluster/submit.sh experiments/<folder>/experiment.yaml <set>                                             # cluster

(If this folder has a `run.py`, use it in place of `scripts/run_kshot.py` on the hub; `submit.sh` picks it up
itself.)

**Results.** `results/<name>/<set>/`, where `<name>` is `experiment.yaml`'s `name` (default: this folder's name).

**Outcome.** Filled in when the results are in. Once `docs/phase1/pipeline.md` records it, this folder and its
results may be deleted; adding what it proved to the main workflow is a separate decision of the user.
