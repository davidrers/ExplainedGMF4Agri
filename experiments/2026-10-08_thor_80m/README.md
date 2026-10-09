# THOR on 80 m tokens, 8 October 2026

**Question.** THOR's paper finds smaller patches much better. Does THOR v1 large on 80 m tokens (8 and 4 px
patches, a 28 x 28 grid) beat the same encoder on 160 m tokens?

**What ran.** `thor_v1_large_80m` and `thor_v1_large` on the 12-chip pilot, two-stage, K = 100, 20 and 5 %,
seeds 0 to 2. Config `thor_v1_large_80m_ee_pilot.yaml`; the 160 m config is in `../2026-09_pilot_12chips/configs/`.
The `thor_v1_large_80m` entry stays in the backbone registry so this study can be re-run.

**Outcome.** Test Macro-F1 (on the pilot's validation chips), mean of three seeds: 80 m 0.283, 0.176, 0.106 against
160 m 0.230, 0.092, 0.088 at 100, 20 and 5 %. On 9 October 2026 the user kept 160 m as the THOR arm for parity
with TerraMind. The same day 80 m was run again under the K-shot protocol on `EE_2021_mini`, as an experiment
of this folder: `experiment.yaml` (name `kshot`, so its cells sit beside the main workflow's) and
`arms/thor_v1_large_80m.yaml`, the settings of this study. It is not part of the main workflow: its full-Estonia
cache, about 6.4 TB at 154 MB per chip encoding, fits no cluster node. On `EE_2021_mini` it scored test Macro-F1
0.146, 0.110 and 0.109 at 100, 20 and 5 %, above THOR at 160 m at every budget, on one seed and nine test chips.

    poetry run python scripts/run_kshot.py -e experiments/2026-10-08_thor_80m/experiment.yaml --chips data/eurocrops_chips/EE_2021_mini
    bash scripts/cluster/submit.sh experiments/2026-10-08_thor_80m/experiment.yaml EE_2021_mini

**Results.** 12-chip pilot: `results/seg_cached/thor_v1_large_80m_ee_pilot/`, `results/seg_cached/thor_v1_large_ee_pilot/`.
`EE_2021_mini`: `results/kshot/EE_2021_mini/thor_v1_large_80m/`.

**Re-run.** `git checkout a81bd9e`, then `scripts/seg/encode.py` and `scripts/seg/fit_cached.py` with
`configs/seg/thor_v1_large_80m_ee_pilot.yaml`.
