# The K-shot workflow with every encoder fine-tuned: design, 15 October 2026

Asked by the user on 15 October 2026, after the main workflow moved to TerraMind and THOR on S2 + S1 and to frozen
TESSERA v2: "the same workflow but with frozen = false, so all models get fine-tuned (except AlphaEarth)", run on all
of Estonia once ready. It continues `../2026-10-15_gfm_finetune/`, whose S2-only token-grid arms the main workflow no
longer has; its fine-tuning machinery and its checks carry over.

## What is compared

| Model | Frozen (main workflow, `results/kshot/<chips>/`) | Fine-tuned (this folder) |
|---|---|---|
| TerraMind v1 large, S2 + S1 | `terramind_v1_large_s2s1`, cache route | `terramind_v1_large_s2s1_ft`, end to end |
| Prithvi-EO-2.0 600M TL | `prithvi_eo_v2_600_tl`, cache route | `prithvi_eo_v2_600_tl_ft`, end to end |
| THOR v1 large, S2 + S1, 160 m | `thor_v1_large_s2s1`, cache route | `thor_v1_large_s2s1_ft`, end to end |
| TESSERA v2 Medium | `tessera_v2_medium`, MLP on whole chips of the raster cache | `tessera_v2_medium_ft`, labelled-pixel route; and `tessera_v2_medium_lab`, its frozen control |
| AlphaEarth v1 | `alphaearth_v1` | none: an embedding product, no encoder to train |

Every fine-tuned arm is its core arm with `model.trainable: true` and nothing else changed but the route: the encoder
is a second AdamW group at `encoder_lr` 1e-5, warmed up linearly over the first epoch, held in eval mode; the
decoder keeps the core arm's learning rate, batch, precision and epochs. Budgets K = 100, 20 and 5 %, draw 0, seed 0,
45 epochs (15 until the user's change of 15 October 2026), checkpoint of lowest validation loss: `kshot.yaml`'s protocol, so each fine-tuned cell pairs with the main
workflow's frozen cell on the same parcels.

## The TESSERA route

The core arm trains its per-pixel MLP on whole chips of a raster computed once by the frozen encoder. Fine-tuning on
whole chips would run the encoder, forward and backward, on all 50,176 pixels of every training chip in every epoch:
about 90 GPU hours per seed on `EE_2021_rev` at the cache's encoding speed. Pixels without a label give no loss, and
both the encoder and the MLP work pixel by pixel, so `tessera_v2_medium_ft` runs the encoder on the labelled pixels of
each batch only (46.6 M per epoch at K = 100 %). The loss is the core's, cross-entropy over the labelled pixels of the
same four chips; their embeddings are normalised with the raster cache's training-chip statistics, fixed, as the core
does.

One thing changes: the MLP's batch norm takes its training statistics from the labelled pixels instead of from all
pixels of the four chips. `tessera_v2_medium_lab` is the core MLP trained the same way on the frozen cache, so:

- fine-tuned minus control is the effect of fine-tuning alone;
- control minus the core arm is the effect of the route alone.

The encoder runs under bf16 autocast, as the cache was made, in gradient-checkpointed chunks of 4,096 pixels, which
bounds memory and does not change the result. Boundary mIoU is not computed on this route, which has no pixel
geometry; every other metric is the core's. Test predictions cover every pixel of each test chip.

## Readiness checks (`check.py`, `check_pixels.py`)

1. **Equivalence.** At step 0 the fine-tuned model reproduces the frozen arm's input: token-grid features at the
   `ChannelBottleneck` input against `generate_embeddings`; TESSERA embeddings of labelled pixels against the raster
   cache.
2. **The switch.** A short fit moves every encoder tensor on the path from the inputs to the decoder, and the decoder;
   two learning-rate groups, the warm-up, the encoder in eval mode in every training step. For the control, no
   encoder and a moving MLP.
3. **Cost.** Speed and peak memory on the cluster's GPU, and the estimate for the three cells on `EE_2021_rev`.
4. **Predictions.** After the pilot, every cell's `predictions_test.npz` reproduces its logged test pixel accuracy.

Then a pilot of the 15 cells on `EE_2021_rev_mini`, and the run on `EE_2021_rev`, one Slurm job per arm.
