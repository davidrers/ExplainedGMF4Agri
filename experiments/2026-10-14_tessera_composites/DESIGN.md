# Stage 2 design: TESSERA v2 on our composites, frozen or fine-tuned

Agreed with the user on 14 October 2026, after the stage 1 spike ([REPORT.md](REPORT.md)). Everything lives in
this folder; the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is not edited.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Input | The 12 monthly S2 and S1 composites already on disk, mapped as in the spike | No new imagery (user). The spike shows they keep what a frozen encoder needs |
| Encoder | TESSERA v2 pixel student, Medium by default; the size is a field of the arm | User chose v2 over v1.1. Medium was the best and steadiest on composites (0.371 ± 0.003) |
| The switch | `trainable: false` or `true`, nothing in between | User: frozen or full only |
| Head | Per-pixel MLP 128 → 512 → 256 → classes (ReLU, batch norm, dropout 0.2) | Today's TESSERA arm's head, so the switch is the only new variable |
| Training pixels | Each epoch, up to 64 pixels per support parcel, a fresh seeded subset per epoch; the same sampler in both modes | Pixels within a parcel are near duplicates. Full fine-tuning over every pixel would cost about 45 h per cell on all of Estonia. The same sampler gives both modes the same pixels and steps (user) |
| Budgets | The core's `draw_support`: K % of the trainable parcels per class, buffer withheld | The same parcels as every other arm |
| Protocol | K = 100, 20 and 5 %, draw 0, seeds 0 to 2, 15 epochs, checkpoint at the lowest validation loss | As the main workflow and the recent experiments |

## How it runs

The entry point is `run.py`, with the command line of `scripts/run_kshot.py`, so `submit.sh` runs it unchanged. For
each arm it does the following:

1. **Pixel store.** It builds the store of the chip set once, or reuses it (`pixels.py`). Every labelled pixel of
   every training, validation and test chip is kept:
   - S2 as int16 (n, 12, 10), in TESSERA's band order;
   - S1 as float16 dB (n, 12, 2);
   - class, parcel id, chip and position in the chip.

   It goes under `<cache root>/tessera_pixels/<chip set>/`: `data/embeddings/` on the hub, the job's `--scratch`
   on the cluster. A `store.json` records the chip manifest's SHA-256 and the split, and a store that does not
   match is rebuilt elsewhere, never overwritten.
2. **Frozen arm.** The frozen arm encodes the whole store once into an embedding cache beside it (float16,
   128-d). It is keyed by encoder, checkpoint SHA-256 and precision, and its cells train the head on the cache.
   The cache is exact: nothing augments the input, so the frozen encoder's output per pixel never changes.
3. **Each cell.** It runs `fit_pixel.py`: draw the support, train for 15 epochs with the sampler, keep the
   checkpoint at the lowest validation loss, score every labelled test pixel, and write the cell.

`summary.csv` comes from the core's `summarise`. Provenance is stamped as the runner does it: experiment, entry,
arm files, chip set, split, commit, dirty flag, machine, host.

## Model and the switch

- **Encoder.** `tessera_v2.py` holds a copy of TESSERA's v2 `PixelStudent` and its bucketing helpers (MIT licence,
  commit `e368539`). Weights load from `data/weights/tessera/v2/student_<size>.pt` with `weights_only=True`.
  - A missing file is fetched from Hugging Face, `geotessera/TESSERA-V-2.0-2B-<N|S|M|L>`.
  - The file's SHA-256 must match the one recorded in the arm.
- **Inputs per batch.** They are prepared exactly as v2's `encode_pixels` does: S2 and S1 standardised with v2's
  statistics, every frame's validity, the bucket of 8 or 16 steps, v2's padding pattern, day of year on the 15th of
  each month. Pixels of the same bucket go through one forward pass. The only difference from `encode_pixels` is
  that gradients are allowed.
- **The switch.**
  - `trainable: false`: the encoder takes no gradient and its outputs come from the cache.
  - `trainable: true`: every encoder parameter trains.
  - In both modes the encoder stays in evaluation mode. v2's only dropout, 0.2 in the fusion MLP, is then off in
    both, so the switch toggles gradients and nothing else.
- **Optimiser.** AdamW with weight decay 0.05, two learning rates:
  - head 1e-3, constant;
  - encoder 1e-5 in full mode, rising linearly over the first epoch.
- **Precision and batches.** bf16 autocast for the encoder in both modes; batches of 4,096 pixels; cross-entropy
  loss.

## Outputs

Each cell writes `results/<name>/<chip set>/<arm>/P<k>_draw<d>_seed<s>/`, in the schema the core runner writes:
- **`results.json`:**
  - identity and data: `run`, `workflow`, `config`, `country`, `year`, `dataset`, `manifest_sha256`,
    `split_protocol`;
  - budget and training: `label_budget` with `support`, `seed`, `chips`, `backbone`, `frozen_encoder`, `params`,
    `fit_seconds`, `peak_gpu_gb`;
  - results: `test_metrics` with `test/F1_Score`, `test/mIoU`, `test/Pixel_Accuracy`, `test/loss`,
    `test/IoU_<class>` and `test/F1_<class>`; `class_names`, `provenance`.
- **`logs/version_0/metrics.csv`:** `epoch`, `train/loss`, `val/loss`, `val/F1_Score`, `val/mIoU`.
- **`predictions_test.npz`:** `ids`, `pred` and `mask` per test chip, with -1 wherever no label exists. The model
  predicts labelled pixels only.

So `summary.csv` and the template's `report.py` (curves, paired differences, validation curves, per class) work
unchanged. The metrics come from the core's `confusion` and `per_class_scores`, the same definition as every other
arm. Validation uses a fixed subset of up to 64 pixels per validation parcel, seeded once; test uses every labelled
test pixel.

## Checks before any run at scale (`check.py`)

1. The fit's input path, in float32, reproduces the spike's `v2_medium` embeddings of the same pixels, within
   float16 storage precision.
2. One optimiser step in full mode changes encoder weights. In frozen mode they stay bit-identical and the head
   changes.
3. Bucket handling is right for a pixel with unobserved Sentinel-1 months (bucket 8 instead of 16).
4. `poetry run pytest -q` passes, and `run.py --dry-run` lists 18 cells and passes the chip-set check.

## Order of work

1. Add `tessera_v2.py`, `pixels.py`, `fit_pixel.py`, `run.py`, `check.py`, `experiment.yaml`, two arm files (in `arms_run/`: the core test loads every `arms/` file through the backbone registry, which has no entry for this route; the same convention as `2026-10-13_terramind_s1_s2`) and
   `report.py`.
2. Run the checks on the hub.
3. **Pilot on the hub:** `EE_2021_rev_mini`, all 18 cells, which tests the pipeline end to end and measures speed.
4. Commit and push.
5. **On the cluster:** `EE_2021_rev`, one job per arm, then pull, `report.py` and stage 2 of `REPORT.md`.

## Open points, measured on the pilot

- The encoder learning rate, 1e-5, is a default and not tuned. If full fine-tuning diverges or does not move, try
  3e-5 or 3e-6 on the pilot only, and report it.
- Throughput on the cluster, which decides whether 64 pixels per parcel stays.
