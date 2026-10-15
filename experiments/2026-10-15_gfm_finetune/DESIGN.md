# Design: the token-grid encoders frozen or fine-tuned, 15 October 2026

Agreed with the user on 15 October 2026. Everything lives in this folder; the main workflow (`src/`, `configs/`,
`scripts/`, `tests/`) is not edited.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Models | TerraMind v1 large, Prithvi-EO-2.0 600M TL, THOR v1 large: the three token-grid arms of the main workflow | User |
| The switch | Frozen or fully fine-tuned, nothing in between | User: the same switch as the TESSERA study (`../2026-10-14_tessera_composites/`) |
| Frozen side | The core arms, unchanged, on the cache route, exactly as `kshot` runs them | User. The cached and the end-to-end fit train the same model by design (`cached.py`), and the frozen Prithvi and THOR cells on `EE_2021_rev` are main-workflow cells that are still missing |
| Fine-tuned side | `arms_run/<arm>_ft.yaml`: a copy of the core arm plus `trainable: true`, on the end-to-end route with the encoder trained | Everything else stays the core arm's, so the switch is the only variable |
| Rollout | Build, check and pilot first; the run on all of Estonia (`EE_2021_rev`) waits for the user's go | User |

## What stays the core arm's

The data and normalisation, the D4 augmentation (applied live to the input instead of drawn from the eight cached
variants: the same distribution), the batch of 2 with 2 accumulated (effective 4), the `ChannelBottleneck` and
UNet decoder at a constant learning rate of 1e-4 with AdamW and weight decay 0.05, the loss, `16-mixed` precision,
15 epochs, and the checkpoint of lowest validation loss.

## What the switch changes

* **The encoder trains.** TerraTorch's task is built with `freeze_backbone: false`.
* **Its own learning rate.** The encoder is a second AdamW group at 1e-5 (`encoder_lr`), with the same weight
  decay, rising linearly over the first epoch (`encoder_warmup_epochs: 1`) and constant after. The decoder group
  keeps the core's constant 1e-4. This follows the TESSERA study.
* **Eval mode.** The encoder is held in eval mode throughout. None of the three has active dropout, drop-path or
  batch norm (only LayerNorm; checked 15 October 2026), so train and eval mode compute the same; holding it states
  that at step 0 the encoder hands the decoder exactly the features the frozen cache holds.
* **Sizes.** The encoders hold 305 M (TerraMind), 631 M (Prithvi) and 314 M (THOR) parameters, which all train,
  against about 53 to 63 M trainable in the frozen arms.
* **Fallbacks, reported if used.** bf16 instead of `16-mixed` if the pilot shows fp16 overflow; activation
  checkpointing if a model does not fit the cluster's 96 GB at batch 2.

## The second checkpoint

The fine-tuned arms also keep the checkpoint of highest validation Macro-F1 and score it on test. The result goes
to `test_metrics_best_f1` in `results.json`, beside the protocol's `test_metrics` (lowest validation loss), which
stays the primary result. The second checkpoint file is deleted once scored, since each holds the whole encoder
(1.2 to 2.5 GB). This is a secondary readout of how much the early checkpoint hides, which the TESSERA study
suggested (its fine-tuned validation loss was lowest at epoch 0 to 1).

## How it runs

`run.py` is the K-shot runner with three of this folder's objects swapped in; the command line is
`scripts/run_kshot.py`'s, so `submit.sh` runs it unchanged:

1. **Arm lookup.** `arms_run/` is searched first. A `route: end_to_end` arm with `model.trainable` is accepted for a
   token-grid backbone (the core accepts only `cache` there). The arm files sit in `arms_run/`, not `arms/`, because
   the core test loads every `arms/` file through the core's route check.
2. **Fit.** A cell whose model carries `trainable` is fitted by `fit_ft.fit_finetune`, a copy of the core's
   `fit_end_to_end` with the task of `finetune.py`, the second checkpoint, `frozen_encoder: false` and the workflow
   named. Any other cell goes to the core.
3. **Test predictions.** `finetune.FtFitPredictor` replaces the core's `FitPredictor`. For the fine-tuned cells it
   passes the batch's `temporal_coords` and `location_coords` to the model, as the task does in training and test.
   The core predictor calls the model with the image alone, which is right on its two routes but would drop
   Prithvi's date and location encodings here. It also loads the checkpoint strictly, so a fine-tuned encoder
   that failed to load would stop the run instead of silently predicting with the pretrained one. Other cells
   are predicted exactly as by the core.

The frozen arms need nothing new: the runner resolves or computes their cache and fits them from it.

**Frozen TerraMind on `EE_2021_rev`.** Its nine cells (K = 100, 20, 5; draw 0; seeds 0 to 2) exist in
`results/2026-10-13_terramind_s1_s2/EE_2021_rev/terramind_v1_large/`, from the same core arm, chips and split. For
the full run they are copied (not moved) into this experiment's results folder, where the runner finds them
finished; their `results.json` keeps the provenance of the run that made them.

## Outputs

The main workflow's layout, `results/2026-10-15_gfm_finetune/<chip set>/<arm>/P<k>_draw<d>_seed<s>/`:
`results.json` (with `frozen_encoder`, the parameter counts, `encoder_training` and, for the fine-tuned arms,
`test_metrics_best_f1`), `logs/version_0/metrics.csv`, `checkpoints/best-loss.ckpt` and `predictions_test.npz`;
`summary.csv` from the core's `summarise`. `report.py` pairs each model frozen against fine-tuned, fit by fit on
the same draw and seed.

## Checks before the run on all of Estonia

`check.py`, per fine-tuned arm (on the hub where the card holds the model, else as a cluster job through
`RUNNER=... submit.sh`), writing `results/<name>/<chip set>/checks/<arm>.json`:

1. **Equivalence.** At step 0, the fine-tuned model's features at the cache point (the input of
   `ChannelBottleneck`) equal the features the main workflow's `generate_embeddings` caches for the same chips.
2. **The switch.** One optimiser step moves the encoder in the trainable model and leaves it bit-identical in the
   frozen one; the decoder moves in both. The optimiser holds two groups at the designed learning rates, the
   encoder group holds every encoder parameter, and the encoder runs in eval mode inside the training step.
3. **Cost.** Seconds per training chip and peak GPU memory at batch 2, on the machine the check runs on, and the
   estimate they give for the nine cells on `EE_2021_rev`.

Then:

4. `poetry run pytest -q`, and `run.py --dry-run` on `EE_2021_rev` and `EE_2021_rev_mini` listing 54 cells and
   passing the chip-set check.
5. **Pilot.** All six arms, 54 cells, on `EE_2021_rev_mini`; `check.py --predictions` then confirms, cell by
   cell, that the pixel accuracy of `predictions_test.npz` equals the test pixel accuracy the fit logged, so the
   predictions come from the model that was tested.

## Order of work

1. Write the folder: `README.md`, this file, `experiment.yaml`, the three arm files, `finetune.py`, `fit_ft.py`,
   `run.py`, `check.py`, `report.py`.
2. Checks 1 to 4 on the hub for what fits its 16 GB card; commit and push.
3. On the cluster: `check.py` for every fine-tuned arm, then the pilot, then pull, `check.py --predictions` and
   `report.py --chips EE_2021_rev_mini`.
4. Report readiness to the user, with the measured cost of the full run. The full run on `EE_2021_rev` follows
   only on the user's go.
