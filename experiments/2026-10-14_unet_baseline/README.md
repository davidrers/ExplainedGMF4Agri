# A from-scratch 2D U-Net baseline on the monthly Sentinel-2 chips, 14 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited. `run.py` registers the U-Net as
a backbone and swaps in the task that builds it; the model is `scratch_unet.py`.

**Question.** Where does a plain U-Net trained from scratch on the same chips sit against the frozen-GFM arms at
K = 100, 20 and 5 %? And does the protocol's 15 epochs undertrain a model that learns its features from scratch?

The baseline was redefined on 2 October 2026: a CNN trained from scratch on the twelve monthly Sentinel-2
composites the GFMs consume, run through the same K-shot experiments, replacing the per-pixel TIMESAT and
monthly-stack baselines in the K sweep. The user chose a plain U-Net, not U-TAE, and on 14 October a 2D one with
the months stacked on channels. If it holds up, it becomes an arm of the main workflow (a separate decision).

**In the main workflow since 15 October 2026** as `unet_r18_scratch_s2s1`, on Sentinel-2 and Sentinel-1 (168
channels), 15 epochs like every arm; `scratch_unet.py` moved to `src/gfm4agri/benchmark/`. This folder's `run.py`
still reproduces the Sentinel-2 runs.

**What it varies.**

| Arm | Input | Model | Trained |
|---|---|---|---|
| `unet_r18_scratch` (`arms_run/`, registered by `run.py`) | 12 monthly S2 composites, all 12 bands, as 144 channels | `segmentation_models_pytorch` U-Net, ResNet-18 encoder, random initialisation | everything, 15 epochs |
| `unet_r18_scratch_45ep` (`arms_run/`) | the same | the same | everything, 45 epochs |
| `tessera_v1` (core arm) | TESSERA v1 embeddings, 128 dims at 10 m | per-pixel MLP (512, 256) | the head, 15 epochs |

- **Time handling.** Early fusion: the 12 months x 12 bands are 144 input channels, months outermost, as in
  `_merged.tif`. The first convolution has its own weight per band and month; nothing after it has a time axis.
  The composites are on a fixed, gap-filled monthly grid, so channel position always means the same month.
- **Normalisation.** The chip set's per-band statistics (`manifest.json`, the same for every month).
- **Training.** Settings as the end-to-end arms where they apply: AdamW, lr 1e-3, weight decay 0.05, batch 4,
  D4 augmentation, cross-entropy that ignores unlabelled pixels, dropout 0.1 before the last 1 x 1 convolution,
  checkpoint of lowest validation loss.
- **References.** TESSERA runs here on the same draws and seeds. The token-grid reference is `terramind_v1_large`
  of [`2026-10-13_terramind_s1_s2`](../2026-10-13_terramind_s1_s2/): same chip set, budgets, draw and seeds, so
  every arm is paired fit by fit.

**Shared by all arms:**
- **Chips:** `EE_2021_rev`, all of Estonia under the `revised` screening: 4,898 training, 769 validation and 1,475
  test chips.
- **Protocol:** K = 100, 20 and 5 %, draw 0, seeds 0 to 2, checkpoint of lowest validation loss.

**How to run.** On the cluster, one job per arm:

    bash scripts/cluster/submit.sh experiments/2026-10-14_unet_baseline/experiment.yaml EE_2021_rev
    bash scripts/cluster/pull_results.sh 2026-10-14_unet_baseline EE_2021_rev
    poetry run python experiments/2026-10-14_unet_baseline/report.py --chips EE_2021_rev

On the hub, `run.py` in place of `scripts/run_kshot.py`, with the same arguments.

**Checked before submission, on the hub.**
- `pytest` passes, and the dry run on `EE_2021_rev` lists 27 cells with the chip-set check passing.
- The composed configurations give the U-Net arms the `end_to_end` route at 15 and 45 epochs, and TESSERA its
  core settings.
- The U-Net arm ran all 9 cells on `EE_2021_rev_mini` (32 training chips): 14.8 M trainable parameters, fit, test
  from the best checkpoint, predictions and `frozen_encoder: false` in `results.json`. Test Macro-F1 0.130, 0.159
  and 0.120 at K = 100, 20 and 5 % (mean of 3 seeds), the same order as the GFM arms on `EE_2021_mini`.
- Two faults found there and fixed in `run.py`: TerraTorch's task sizes its metrics from `model_args` even when
  handed a model, and a model handed over as an object is pickled into the checkpoint, which Lightning's
  weights-only loading then refuses at test time. The model is now built by a registered factory from plain
  arguments.
- The first validation loss of a fit is in the hundreds (BatchNorm statistics not yet settled) and falls to about
  2 by epoch 3 to 5; `report.py` draws the validation loss on a log axis.

Submitted on 14 October 2026 as jobs 621675 (`unet_r18_scratch`), 621676 (`tessera_v1`) and 621677
(`unet_r18_scratch_45ep`, with a 3-day limit) at commit `ac3abbc`. All three finished the same day on `hpc-node30`:
2 h 56 min, 3 h 11 min and 9 h 1 min, no failure.

**Results.** `results/2026-10-14_unet_baseline/EE_2021_rev/`; the pilot in `.../EE_2021_rev_mini/`.

**Report.** `report.py` compares the three arms with TerraMind v1 large from the S1/S2 experiment, with the paired
differences 45 minus 15 epochs and each U-Net arm minus each GFM arm.

**Outcome.** At the protocol's 15 epochs the U-Net is clearly below both GFM arms, but 15 epochs undertrain it
([REPORT.md](REPORT.md)).
- **Longer training.** 45 epochs add +0.046, +0.062 and +0.071 Macro-F1 at K = 100, 20 and 5 %, winning all nine
  paired fits. At 15 epochs, every fit's checkpoint is at epoch 12 to 14.
- **Test Macro-F1, U-Net at 45 epochs against TerraMind v1 large (S2) and TESSERA + MLP:**
  - K = 100 %: 0.544 against 0.628 and 0.625;
  - K = 20 %: 0.477 against 0.511 and 0.597;
  - K = 5 %: 0.413 against 0.394 and 0.570.
- **TerraMind's lead over the longer-trained U-Net shrinks as labels get scarce:** −0.084, −0.034, then +0.019 for the
  U-Net at 5 %, which is within the noise.
- **TESSERA's lead grows as labels get scarce:** −0.081, −0.120 and −0.157 for the U-Net. Its embeddings also carry
  Sentinel-1.
- **The shared training budget must be set before the K sweep.** At 15 epochs the comparison overstates the GFMs'
  benefit by 0.05 to 0.07.
