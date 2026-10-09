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

**Re-run.** `git checkout a81bd9e`, where the scripts sit in `scripts/data/` and `scripts/seg/` and the configs in
`configs/seg/`, then the commands in the notebooks or `scripts/seg/run_budget_grid.sh` from the repository root.
