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
    poetry run python experiments/2026-10-15_kshot_finetune/report.py --chips EE_2021_rev    # figures/<chip set>/

**Results.** `results/2026-10-15_kshot_finetune/<chip set>/`, with `checks/` for the readiness checks.

**Outcome, 18 October 2026: all of Estonia done at 45 epochs. Details in [REPORT.md](REPORT.md).**
- **TESSERA v2:** fine-tuning pays off. Test Macro-F1 rises from 0.618, 0.582 and 0.504 to 0.660, 0.630 and 0.581
  at K = 100, 20 and 5 %, and the gain holds per parcel and in validation. The frozen control of the same route
  moves less than 0.005 in validation, so the gain is the encoder's. The fine-tuned arm is the best of the main grid
  at K = 20 and 5 %.
- **TerraMind, Prithvi and THOR:** fine-tuning gains about +0.01 in best validation Macro-F1 (THOR -0.033 at
  K = 5 %), within the seed spread. At the protocol's checkpoint of lowest validation loss, test differences swing
  from -0.082 to +0.034. Fine-tuned validation loss turns at epochs 5 to 9, so the checkpoint lands before the
  rarer classes are learnt.
- **Cost:** 4 to 9 times the frozen fitting time for the token-grid models, 15 times for TESSERA (34.6 h for three
  cells).
- **Run:** one draw and one seed per cell. Node failures on 16 October forced three cells to be refitted, and two of
  them were first predicted with the interrupted fit's checkpoint. Commit 02a10f1 fixed the predictor and the two
  were predicted again. All 15 cells' predictions now reproduce their logged scores.
- **Stage 1** (readiness checks and the 15-epoch pilot on `EE_2021_rev_mini`): every check passed. The pilot was too
  small to answer the question.
