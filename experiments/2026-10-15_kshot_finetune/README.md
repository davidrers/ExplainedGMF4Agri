# The K-shot workflow with every encoder fine-tuned, 15 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited. The design is in
[DESIGN.md](DESIGN.md); it continues [`../2026-10-15_gfm_finetune/`](../2026-10-15_gfm_finetune/README.md), whose
S2-only arms the main workflow no longer has.

**Question.** The main workflow (`configs/experiments/kshot.yaml`) keeps every encoder frozen. What does each model
gain, at each label budget, when its encoder trains with the decoder? The arms are kshot's with the encoder
trainable: TerraMind v1 large and THOR v1 large on S2 + S1, Prithvi-EO-2.0 600M TL, and TESSERA v2 Medium. AlphaEarth
is left out, having no encoder to train.

**What it varies.** One switch per model; the frozen side is the main workflow's own results on the same chip set.

| Arm (`arms_run/`) | Encoder | Route | Paired with (`results/kshot/<chips>/`) |
|---|---|---|---|
| `terramind_v1_large_s2s1_ft` | trained | end to end | `terramind_v1_large_s2s1` |
| `prithvi_eo_v2_600_tl_ft` | trained | end to end | `prithvi_eo_v2_600_tl` |
| `thor_v1_large_s2s1_ft` | trained | end to end | `thor_v1_large_s2s1` |
| `tessera_v2_medium_ft` | trained | labelled pixels | `tessera_v2_medium`, and the control below |
| `tessera_v2_medium_lab` | frozen | labelled pixels | `tessera_v2_medium` |

The encoders train at 1e-5 after a one-epoch warm-up, in eval mode; everything else is the core arm's.

The code is in this folder:
- `finetune.py`: the token-grid task with the trainable encoder and its two learning rates, and the predictor of
  this folder's cells;
- `fit_ft.py`: one token-grid cell, a copy of the core's end-to-end fit with that task and a second checkpoint
  (highest validation Macro-F1) scored as a secondary readout;
- `pixelft.py` and `fit_pixels.py`: the TESSERA labelled-pixel route and one TESSERA cell;
- `run.py`: the entry, the K-shot runner with the swaps;
- `check.py` and `check_pixels.py`: the readiness checks of DESIGN.md, and the check of the predictions;
- `report.py`: figures and tables, reading the frozen arms from `results/kshot/<chips>/` without writing there.

**Protocol.** kshot's: K = 100, 20 and 5 %, draw 0, seed 0, 45 epochs (15 until 15 October 2026), checkpoint of lowest
validation loss
(`experiment.yaml`): 15 cells.

**How to run.**

    poetry run python experiments/2026-10-15_kshot_finetune/check.py -e experiments/2026-10-15_kshot_finetune/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --arms tessera_v2_medium_ft
    RUNNER=experiments/2026-10-15_kshot_finetune/check.py bash scripts/cluster/submit.sh experiments/2026-10-15_kshot_finetune/experiment.yaml EE_2021_rev_mini
    bash scripts/cluster/submit.sh experiments/2026-10-15_kshot_finetune/experiment.yaml EE_2021_rev_mini    # one job per arm
    poetry run python experiments/2026-10-15_kshot_finetune/check.py -e experiments/2026-10-15_kshot_finetune/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini --predictions
    poetry run python experiments/2026-10-15_kshot_finetune/report.py --chips EE_2021_rev_mini

**Results.** `results/2026-10-15_kshot_finetune/<chip set>/`, with `checks/` for the readiness checks.

**Status, 15 October 2026: ready; all of Estonia running at 45 epochs. Stage 1 in [REPORT.md](REPORT.md).**
- **Checks.** Every readiness check passes on the cluster for the four fine-tuned arms: step-0 equivalence with the
  core's caches, the switch, eval mode and the warm-up.
- **Pilot.** The 15 cells on `EE_2021_rev_mini` ran without a numeric fault and reproduce their logged test
  accuracy from their predictions; too small to say whether fine-tuning helps.
- **Epochs.** The user moved this experiment and the main workflow from 15 to 45 epochs the same evening. The
  15-epoch run on `EE_2021_rev` (jobs 624389 to 624393) was cancelled after about 2 h, one cell finished; its outputs
  are kept in `results/2026-10-15_kshot_finetune/_cancelled_15_epochs/EE_2021_rev/`. The pilot stays at 15 epochs:
  it tested the workflow, which the epoch count does not change.
- **Running.** `EE_2021_rev` at 45 epochs, about 10, 17, 18 and 34 h for TerraMind, Prithvi, THOR and TESSERA once
  started. Its frozen side is the main workflow's 45-epoch run, which the report pairs by epochs.
