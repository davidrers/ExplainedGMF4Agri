# The K-shot arms on every sensor they read, and frozen TESSERA v2 in the main workflow

Design agreed with the user on 15 October 2026, section by section; reviewed and approved the same day, with the
per-pixel MLP as the 10 m decoder (section 6).

## 1. Purpose

Bring the main workflow's arms as close to each other as their models allow, so that RQ1 compares encoders and not
set-ups:

1. **Every GFM reads every sensor it was pretrained on.** TerraMind, THOR and TESSERA take Sentinel-1 beside
   Sentinel-2; Prithvi-EO-2.0 cannot, AlphaEarth already carries it inside its embedding.
2. **TESSERA runs as a model, not a product.** Frozen TESSERA v2 Medium, from its open weights, encodes our own monthly
   composites, and its output is cached by the runner exactly as the token-grid encoders' is. It replaces the v1.0
   store rasters, which are offset by about one pixel from the chip grid (section 3).
3. **The 10 m group keeps one decoder, chosen by evidence.** A U-Net was tested against the per-pixel MLP on frozen
   TESSERA v2 and trailed it; the group keeps the MLP (section 6).

## 2. Decisions

| Decision | Choice | Source |
|---|---|---|
| S1 | Every GFM that accepts it reads it: TerraMind, THOR, TESSERA v2. Prithvi stays on S2 (pretrained on six HLS optical bands only) | User, 15 October 2026 |
| TESSERA | v2 Medium, frozen, on the 12 monthly S2 + S1 composites, cached by the runner (approach 1: as the token-grid models, not a hub-built data product) | User, 15 October 2026 |
| `tessera_v1` | Leaves `kshot.yaml`; its arm file, builder and rasters stay, so earlier results remain refittable | User, 15 October 2026 |
| The 10 m decoder | The per-pixel MLP, for the whole group (TESSERA v2 and AlphaEarth), as now; the U-Net is not added to the core | User, 15 October 2026, on the first results of `experiments/2026-10-15_tessera_decoder` |
| AlphaEarth | Takes the group's decoder; with the MLP kept, it is unchanged | User, 15 October 2026 |
| THOR's band groups | Averaged at every token position, as TerraMind averages its modalities, so THOR's bottleneck and decoder match TerraMind's | User, 15 October 2026 |
| Order of work | A branch in its own worktree; tests, hub pilot, merge, cluster pilot | This design |

**Out of scope:** a full-Estonia grid (run on the user's word, after this change); comparing THOR's averaging with
concatenation; THOR on S2 alone with averaging; the from-scratch baseline as a core arm; the U-Net as a core
decoder; the 2 October protocol changes (K grid, nested repeated draws, equal training steps, low-K validation);
TESSERA fine-tuning; the v1.1 store; half-month composites.

## 3. Why the v1.0 store rasters go

On `EE_2021_rev_mini`, ridge maps from the on-disk v1.0 raster, shifted by up to two pixels, onto three grid-aligned
references (v2 Medium and v1.1 AWS run on our composites, and the v1.1 store read afresh) all fit best one pixel
east. Interpolated, the offset is about 0.7 px east and 0.4 px south, 5 to 8 m. The sidecar records a nearest-neighbour
warp from the store's UTM grid, which can produce such an offset. Every TESSERA result so far is therefore trained on
embeddings slightly displaced from their labels, which costs at parcel edges and favours any decoder that can look
across a pixel. The v2 rasters are encoded from the composites on the chip grid itself.

## 4. The arms of `kshot.yaml`

| Arm | Input | Resolution group | Trainable part | Change |
|---|---|---|---|---|
| `terramind_v1_large_s2s1` | S2 + S1 RTC, modality tokens averaged | `token_grid`, 160 m | `ChannelBottleneck` (40.4 M) + 12.9 M UNet decoder | Replaces `terramind_v1_large`. In the registry since 13 October and run on all of Estonia (`experiments/2026-10-13_terramind_s1_s2`: +0.026, +0.054, +0.025 over S2); its arm file moves to `configs/arms/` |
| `thor_v1_large_s2s1` | S2 (10 m and 20 m groups) + S1 (VV, VH), the three groups averaged | `token_grid`, 160 m | as TerraMind's, about 53 M | New; replaces `thor_v1_large` |
| `prithvi_eo_v2_600_tl` | S2, six HLS bands | `token_grid`, 160 m | as TerraMind's | Unchanged |
| `tessera_v2_medium` | S2 + S1, frozen, cached by the runner | `pixel_raster`, 10 m | per-pixel MLP, about 0.2 M | New; replaces `tessera_v1` |
| `alphaearth_v1` | the precomputed annual embedding | `pixel_raster`, 10 m | per-pixel MLP, about 0.2 M | Unchanged |

The arm files of `tessera_v1`, `terramind_v1_large` and `thor_v1_large` stay in `configs/arms/` for refits and
experiments. An arm that reads S1 makes the chip-set check require `_s1rtc.tif`; every chip set on the hub and the
cluster holds it. The protocol states Prithvi's S2-only input as a property of the model, as it states AlphaEarth's
annual embedding.

## 5. Frozen TESSERA v2 in the core

### 5.1 Encoder and weights

- **Code.** TESSERA's v2 `PixelStudent` and its input bucketing go into `src/gfm4agri/embeddings/`, copied under its MIT
  licence from `ucam-eo/tessera` at commit `e368539`, with the input mapping both experiments validated:
  - TESSERA's ten S2 bands in its order (B04 B02 B03 B08 B8A B05 B06 B07 B11 B12, our indices 3 1 2 7 8 4 5 6 10 11);
  - S1 from our dB, 10 log10 of linear power, to TESSERA's units, (2 dB + 50) x 200 clipped to [0, 32767], 0 where
    unobserved; the merged orbits handed over as its ascending stream;
  - day of year the 15th of each month; every S2 month valid, since the composites are gap filled.
- **Weights.** `data/weights/tessera/v2/student_medium.pt`, fetched from Hugging Face
  (`geotessera/TESSERA-V-2.0-2B-M`, `ckpt/student_medium.pt`) when missing, loaded with `weights_only=True` and checked
  against SHA-256 `3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e`, recorded in the registry.
  `scripts/cluster/prefetch_weights.py` fetches it with the others.

### 5.2 Registry

`tessera_v2_medium` joins `BACKBONES` in the `pixel_raster` group. A 10 m backbone then comes in two kinds: one that
names a raster in the chip set (`tessera_v1`, `alphaearth_v1`), and one that carries an encoder whose output the
runner computes (`tessera_v2_medium`). Arms of both kinds keep `route: raster`. The spec records the student, the
checkpoint's SHA-256, the encoding precision (bf16) and the S1 modality it reads.

### 5.3 The cache

As for a token-grid arm (`gfm4agri.pipeline.cache`):

- **Lookup.** `<cache root>/tessera_v2_medium/<chip set>/`, reused when its `cache.json` matches the run on the
  backbone, the checkpoint's SHA-256, the chip manifest's SHA-256, the split and the precision, and every chip of the
  split is there. Otherwise computed into the scratch root, resuming chips already there. A cache that exists with
  another record is never written into.
- **Content.** `chips/<chip>_tessera_v2_medium.tif`, float32, 128 bands on the chip's own grid, NaN where a pixel has
  neither a valid S2 nor a valid S1 month; a sidecar `tessera_v2_medium.json` in the layout of the store sidecars, with
  per-band statistics of the training chips for the datamodule's normalisation; `cache.json`.
- **One copy per chip.** A per-pixel encoder commutes with rotation and flipping, so D4 augmentation is applied to the
  cached raster during training, exactly; the eight-variant precomputation of the ViT caches is not needed.
- **Throughput.** Measured on 15 October 2026: 1,770 chips in 19 min on the cluster's RTX PRO 6000, about 5 s per chip
  on the hub's shared A4000. All of `EE_2021_rev` is about 80 min and 190 GB of job scratch per cluster job; the hub
  runs pilots.

### 5.4 The fit

The core's `raster` route, unchanged except that the datamodule reads the embedding rasters from the cache and the
masks and parcel rasters from the chip set. TerraTorch's dataset already takes the two roots apart and pairs files
by chip id after the split filter. `results.json` records the checkpoint and the cache record in its provenance, as
it does a token-grid cache.

## 6. The decoder of the 10 m group

- **The per-pixel MLP stays** (`DECODERS["pixel_raster"]`, `PixelMLPDecoder`, 512 and 256, about 0.2 M parameters, lr
  1e-3, dropout 0.2), for TESSERA v2 and AlphaEarth alike. Nothing in the decoder code changes.
- **Why.** In `experiments/2026-10-15_tessera_decoder` on `EE_2021_rev_quarter`, a U-Net (`segmentation_models_pytorch`
  `Unet`, ResNet-18 down path, 14.7 M parameters, trained from scratch) on the same frozen TESSERA v2 rasters trailed
  the MLP at the protocol's 15 epochs: test Macro-F1 0.516 against 0.574 at K = 100 % (3 of 3 seeds), and 0.425
  against 0.527 at K = 20 % (seed 0), the first results on 15 October 2026. The user chose the MLP on them.
- **Still open.** The experiment's 45-epoch U-Net arm tells whether the U-Net was only undertrained. If it comes out
  ahead, adding it to the core is a new decision, against the protocol's equal training for every model.
- **Consequence for the comparison.** The 10 m arms keep a decoder without spatial context while the token-grid arms
  have a UNet decoder; the protocol states this beside the decoder input resolution (D20, D21).

## 7. THOR on S2 + S1

- **Input.** `_s1rtc.tif`, VV and VH in dB, read beside the S2 stack and interleaved by month by the datamodule's
  existing extra-modality mechanism. THOR takes the two as one stacked image whose band list names both (`VV`, `VH`
  map to its 10 m IW group); TerraMind keeps its modality dictionary.
- **Tokens.** S1 is a third group beside the 10 m and 20 m S2 groups, at 16 px, so 160 m, on the same 14 x 14 grid.
- **Normalisation.** THOR's pretraining statistics for its S1 bands at 10 m, in dB, as for its S2 bands. An S1 month
  never observed enters at the band mean.
- **Averaging.** `backbone_merge_method: mean` over the three groups: 12 x 1024 channels per selected layer, the
  `ChannelBottleneck` and UNet decoder of TerraMind, about 53 M trainable (against about 91 M for `thor_v1_large`).
  The cache is half as wide per month as the concatenated set-up's.
- **Caveats the protocol states.** Averaging is untested for THOR, whose authors concatenate groups for dense tasks;
  the pilot shows the arm runs, not how the two compare. Our S1 is terrain-corrected (RTC), THOR was pretrained on
  ordinary backscatter; in flat Estonia the two differ little, and TerraMind meets the converse mismatch.

## 8. Verification

**Tests** (`poetry run pytest -q`), added to the suite:

1. The core's TESSERA encoder reproduces `v2_medium` of `experiments/2026-10-14_tessera_composites` on pilot chips,
   cosine at least 0.999; skipped where the weights are absent. Band order and the S1 unit conversion tested on
   their own.
2. TESSERA cache decisions (reuse, compute, stop) from its record; a cache missing a chip is not reused.
3. The datamodule pairs cached embeddings with masks and parcels by chip id, including under D4.
4. THOR S2 + S1: its groups, its S1 statistics, the averaging, a bottleneck as wide as TerraMind's.
5. The five arms of `kshot.yaml` load and pass the chip-set check.

**Pilots.** `kshot` on `EE_2021_rev_mini`, all five arms, on the hub from the branch, then on the cluster from `main`;
every new cache computed once there. A failure on the cluster is fixed forward on `main`.

## 9. Documentation edits

Proposed as text with the implementation and written only once the user agrees:

- **`docs/phase1/protocol.md`:** each arm's input (S1 for TerraMind, THOR and TESSERA; Prithvi on S2 by nature);
  TESSERA v2 from open weights, frozen; THOR's averaging; the 10 m group's MLP, and the U-Net result behind it, in D21.
- **`docs/thesis_design.md`:** the model table (TESSERA's access becomes open weights); Phase 2, where TESSERA with
  weights can join the backbone tier of explainability; the sentence on fine-tuning keeps its conclusion with a new
  reason.
- **`docs/phase1/pipeline.md`:** the stage table; section 7 (TESSERA v2); section 9 (decoders, THOR); the K-shot
  workflow; section 10 (the v1.0 offset, THOR's untested averaging, RTC against ordinary backscatter).
- **`CLAUDE.md`:** the current state and the arm list.

## 10. Git

A branch, `kshot-s1-tessera-v2`, in its own worktree, so that other sessions working in the main checkout are not
disturbed. Commits stage files by name. The branch merges into `main` once the tests and the hub pilot pass; the
cluster pulls `main` for its pilot. The implementation plan goes to `docs/superpowers/plans/`.
