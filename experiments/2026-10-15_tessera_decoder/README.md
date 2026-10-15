# A U-Net or a per-pixel MLP on frozen TESSERA embeddings, 15 October 2026

Nothing in the main workflow (`src/`, `configs/`, `scripts/`, `tests/`) is edited.

**Question.** Does a spatial U-Net decoder recognise crops better than the per-pixel MLP, on frozen TESSERA
embeddings, at each label budget? The answer decides the decoder of the TESSERA arm when the main workflow moves
from the v1 store to frozen TESSERA v2 Medium.

**Why it matters.**
- **TESSERA has no spatial context.** It encodes each pixel's own time series and nothing else (its README: "only
  spectral-temporal features at pixel level"; no convolution or spatial attention anywhere in v1, v1.1 or v2).
- **The MLP adds none.** With the per-pixel MLP of the core, no part of the TESSERA arm ever sees a neighbouring
  pixel. The token-grid arms see neighbours twice: in their ViT and in their 12.9 M UNet decoder.
- **The design asks for it.** `thesis_design.md` speaks of "a UNet over a 10 m embedding raster" and one decoder
  family per resolution group. The core's decoder table already names "a light spatial UNet" as the planned second
  head of the `pixel_raster` group.

**What it varies.**

| Arm | Decoder | Trainable parameters | Epochs |
|---|---|---|---|
| `tessera_v2m_mlp` | the core's `PixelMLPDecoder` (512, 256), each pixel on its own | about 0.2 M | 15 |
| `tessera_v2m_unet` | `segmentation_models_pytorch` U-Net, ResNet-18 down path, random initialisation | about 14 M | 15 |
| `tessera_v2m_unet_45ep` | the same U-Net | about 14 M | 45 |

- **Shared by all three arms.** The same frozen TESSERA v2 Medium rasters (128 dims per 10 m pixel), the chip-level
  `raster` route of the core, D4 augmentation, batch 4, AdamW at 1e-3 with weight decay 0.05, training-chip
  normalisation, cross-entropy, the checkpoint at the lowest validation loss.
- **The U-Net** is the network of the from-scratch baseline ([`../2026-10-14_unet_baseline/`](../2026-10-14_unet_baseline/)),
  with the 128 embedding channels in place of the 144 composite bands, and its head dropout of 0.1. The MLP keeps
  the TESSERA arm's 0.2.
- **The 45-epoch arm** tells whether 15 epochs undertrain a U-Net learned from scratch. The baseline experiment
  found they do on raw composites, and on a quarter of the chips each epoch holds a quarter of the steps.

**Why v2 rasters, not the v1 store on disk.**
- **The v1.0 store rasters are offset from our chip grid.** They are offset by about 0.7 px east and 0.4 px south
  (5 to 8 m), measured on `EE_2021_rev_mini` on 15 October 2026. Ridge maps from the shifted v1.0 raster onto
  three grid-aligned references (v2 Medium and v1.1 AWS on our composites, the freshly read v1.1 store) all fit
  best at a one-pixel shift east.
- **That would bias this test.** A U-Net can learn to look one pixel over and an MLP cannot.
- **v2 is the right choice anyway.** The v2 rasters are encoded from the composites on the chip grid itself, and
  they are the representation the main workflow is to move to.

**How the rasters are made** (`rasters.py`, called by `run.py` before the runner starts).
- **Encoding.** Every chip of the chip set is encoded with the frozen v2 Medium student from its 12 monthly S2 and
  S1 composites, under bf16, as the frozen arm of
  [`../2026-10-14_tessera_composites/`](../2026-10-14_tessera_composites/REPORT.md) did.
- **Where they go.** Into a copy of the chip set made of links, with one `<chip>_tessera_v2m.tif` (float32,
  128 bands) per chip and a sidecar `tessera_v2m.json` holding the training-chip statistics. The copy goes under
  `data/embeddings/tessera_v2m_rasters/<set>/` on the hub, and under the job's `/local` scratch on the cluster.
- **How the runner uses them.** It is pointed at the copy, so the core's raster route runs unchanged. Results land
  under the chip set's own name. Every `results.json` records the chip set and the rasters' key in its provenance.

**Chip set.** `EE_2021_rev_quarter`, made for this experiment on 15 October 2026 with
`scripts/hub/build_pilot_chipset.py --parent data/eurocrops_chips/EE_2021_rev --train-blocks 116 --val-blocks 16
--test-blocks 32 --seed 0`.
- **Size.** About a quarter of `EE_2021_rev`'s blocks, as links: 1,190 training, 169 validation and 411 test chips.
- **Split.** The same split, `blocks4_buf1600_seed0__6b0eb4cb`.
- **Why a quarter.** For a quick answer, at about a quarter of the cost of all of Estonia.

**Protocol.** K = 100, 20 and 5 %, draw 0, seeds 0 to 2, 15 epochs (45 for the long arm), checkpoint at the lowest
validation loss. All arms share the draw and seeds, so U-Net and MLP are compared fit by fit.

**How to run.**

    poetry run python experiments/2026-10-15_tessera_decoder/run.py -e experiments/2026-10-15_tessera_decoder/experiment.yaml --chips data/eurocrops_chips/EE_2021_rev_mini   # hub pilot
    bash scripts/cluster/submit.sh experiments/2026-10-15_tessera_decoder/experiment.yaml EE_2021_rev_quarter                                                    # cluster, one job per arm
    poetry run python experiments/2026-10-15_tessera_decoder/report.py --chips EE_2021_rev_quarter

The v2 weights are those of the TESSERA-on-composites experiment, `data/weights/tessera/v2/student_medium.pt`,
checked by SHA-256 (see its README).

**Results.** `results/2026-10-15_tessera_decoder/<set>/`.

**Outcome.** ([REPORT.md](REPORT.md)) No: on frozen TESSERA v2 Medium the U-Net does not beat the per-pixel MLP under
label scarcity.
- **At 15 epochs it loses every paired fit:** test Macro-F1 0.516 / 0.444 / 0.259 against 0.573 / 0.532 / 0.451 at
  K = 100 / 20 / 5 %.
- **At 45 epochs it catches up only with every label:** +0.016 at K = 100 %, level at 20 %, -0.093 at 5 %.
- **It gains spatial coherence and parcel edges, and loses whole small parcels,** concentrated in the rare classes.
- **The main workflow keeps the MLP** for its 10 m group (the user, 15 October 2026).
- **Runs:** jobs 623444 to 623446 on `itc-gpu`, commit `0ab085d`, `EE_2021_rev_quarter`.
