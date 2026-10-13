# A from-scratch 2D U-Net baseline on the monthly Sentinel-2 chips, 14 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited. `run.py` registers the U-Net as
a backbone and swaps in the task that builds it; the model is `scratch_unet.py`.

**Question.** Where does a plain U-Net trained from scratch on the same chips sit against the frozen-GFM arms at
K = 100, 20 and 5 %? And does the protocol's 15 epochs undertrain a model that learns its features from scratch?

The baseline was redefined on 2 October 2026: a CNN trained from scratch on the twelve monthly Sentinel-2
composites the GFMs consume, run through the same K-shot experiments, replacing the per-pixel TIMESAT and
monthly-stack baselines in the K sweep. The user chose a plain U-Net, not U-TAE, and on 14 October a 2D one with
the months stacked on channels. If it holds up, it becomes an arm of the main workflow (a separate decision).

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

**Results.** `results/2026-10-14_unet_baseline/EE_2021_rev/`; the pilot in `.../EE_2021_rev_mini/`.

**Report.** `report.py` compares the three arms with TerraMind v1 large from the S1/S2 experiment, with the paired
differences U-Net minus each GFM arm and 45 minus 15 epochs.

**Outcome.** Pending: submitted to the cluster.
