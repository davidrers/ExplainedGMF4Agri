# Experiments

One folder per study, named `<date>_<name>`. A study is anything that is not the K-shot workflow itself: an
ablation, an exploration, a pilot of something not adopted. Each folder holds a `README.md` with the question,
what ran, the outcome, where the results are and the commit to check out to re-run it, plus the study's own
scripts and configs.

**Cleanup rule.** A study and its results may be deleted once its conclusion is written into
`docs/phase1/pipeline.md`. The core (`src/`, `configs/{arms,experiments,machines}`, `scripts/{hub,cluster,env}`,
`scripts/run_kshot.py`, `tests/`) is never deleted.

Archived studies are a record: they ran against the code of their commit, not against the current runner.
