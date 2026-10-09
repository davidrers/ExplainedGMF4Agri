# THOR on 80 m tokens, 8 October 2026

**Question.** THOR's paper finds smaller patches much better. Does THOR v1 large on 80 m tokens (8 and 4 px
patches, a 28 x 28 grid) beat the same encoder on 160 m tokens?

**What ran.** `thor_v1_large_80m` and `thor_v1_large` on the 12-chip pilot, two-stage, K = 100, 20 and 5 %,
seeds 0 to 2. Config `thor_v1_large_80m_ee_pilot.yaml`; the 160 m config is in `../2026-09_pilot_12chips/configs/`.
The `thor_v1_large_80m` entry stays in the backbone registry so this study can be re-run.

**Outcome.** Test Macro-F1 (on the pilot's validation chips), mean of three seeds: 80 m 0.283, 0.176, 0.106 against
160 m 0.230, 0.092, 0.088 at 100, 20 and 5 %. On 9 October 2026 the user kept 160 m for parity with TerraMind;
80 m is not an arm.

**Results.** `results/seg_cached/thor_v1_large_80m_ee_pilot/`, `results/seg_cached/thor_v1_large_ee_pilot/`.

**Re-run.** `git checkout a81bd9e`, then `scripts/seg/encode.py` and `scripts/seg/fit_cached.py` with
`configs/seg/thor_v1_large_80m_ee_pilot.yaml`.
