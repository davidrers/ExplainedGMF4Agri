# The token-grid encoders frozen or fine-tuned, 15 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited. The design, agreed with the user
on 15 October 2026, is in [DESIGN.md](DESIGN.md).

**Question.** The main workflow keeps TerraMind v1 large, Prithvi-EO-2.0 600M TL and THOR v1 large frozen: each
encodes every chip once into a feature cache, and only the bottleneck and UNet decoder train. What does each model
gain, at each label budget, when its encoder trains with the decoder? The TESSERA study
(`../2026-10-14_tessera_composites/`) found +0.04 test Macro-F1 for fine-tuning there, in all nine paired fits.

**What it varies.** One switch per model, everything else as the core arm:

| Arm | Encoder | Route | Where |
|---|---|---|---|
| `terramind_v1_large`, `prithvi_eo_v2_600_tl`, `thor_v1_large` | frozen | cache, as `kshot` | `configs/arms/` (core) |
| `terramind_v1_large_ft`, `prithvi_eo_v2_600_tl_ft`, `thor_v1_large_ft` | trained, at 1e-5 with a one-epoch warm-up, in eval mode | end to end | `arms_run/` |

The code is in this folder:
- `finetune.py`: the task with the trainable encoder and its two learning rates, and the predictor that passes
  Prithvi's coordinates and loads the fine-tuned checkpoint strictly;
- `fit_ft.py`: one cell, a copy of the core's end-to-end fit with that task and a second checkpoint (highest
  validation Macro-F1) scored as a secondary readout;
- `run.py`: the entry, the K-shot runner with the three swaps;
- `check.py`: the readiness checks of DESIGN.md, and the check of the pilot's predictions;
- `report.py`: figures and tables.

**Protocol.** K = 100, 20 and 5 %, draw 0, seeds 0 to 2, 15 epochs, checkpoint of lowest validation loss
(`experiment.yaml`): 54 cells. Frozen and fine-tuned share draw and seeds, so they are compared fit by fit.

**How to run.**

    poetry run python experiments/2026-10-15_gfm_finetune/check.py -e experiments/2026-10-15_gfm_finetune/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --arms terramind_v1_large_ft
    poetry run python experiments/2026-10-15_gfm_finetune/run.py -e experiments/2026-10-15_gfm_finetune/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini
    RUNNER=experiments/2026-10-15_gfm_finetune/check.py bash scripts/cluster/submit.sh experiments/2026-10-15_gfm_finetune/experiment.yaml EE_2021_rev_mini terramind_v1_large_ft prithvi_eo_v2_600_tl_ft thor_v1_large_ft
    bash scripts/cluster/submit.sh experiments/2026-10-15_gfm_finetune/experiment.yaml EE_2021_rev_mini    # one job per arm
    poetry run python experiments/2026-10-15_gfm_finetune/check.py -e experiments/2026-10-15_gfm_finetune/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --predictions
    poetry run python experiments/2026-10-15_gfm_finetune/report.py --chips EE_2021_rev_mini

**Results.** `results/2026-10-15_gfm_finetune/<chip set>/`, with `checks/` for the readiness checks.

**Status, 15 October 2026: ready, stage 1 reported in [REPORT.md](REPORT.md).**
- **Checks.** All readiness checks pass for the three models on the cluster: identical step-0 features, the switch,
  eval mode and the warm-up.
- **Pilot.** All 54 cells on `EE_2021_rev_mini` reproduce their logged test accuracy from their predictions, with
  no numeric fault.
- **Too small to answer.** The pilot shows no effect of fine-tuning beyond seed noise: paired differences run from
  +0.045 (Prithvi, K = 100 %) to -0.021 (THOR, K = 20 %).
- **Next.** The run on all of Estonia (`EE_2021_rev`) waits for the user's go: five jobs, the longest about 17 h,
  reusing frozen TerraMind from `../2026-10-13_terramind_s1_s2/`.
