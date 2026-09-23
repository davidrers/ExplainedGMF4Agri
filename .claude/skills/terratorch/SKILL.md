---
name: terratorch
description: >
  Fine-tune geospatial foundation models (Prithvi, TerraMind, Clay, SatMAE, DOFA, etc.) for downstream tasks
  using TerraTorch. Use this skill whenever the user wants to train, fine-tune, or run inference with
  geospatial/Earth observation models, write TerraTorch YAML configs, set up datasets for remote sensing tasks,
  pick backbones or decoders for segmentation/regression/classification, or anything involving TerraTorch,
  TorchGeo + foundation models, or satellite imagery deep learning pipelines. Also use when the user mentions
  Prithvi, TerraMind, HLS bands, or geospatial model fine-tuning — even if they don't say "TerraTorch" explicitly.
---

# TerraTorch — Geospatial Foundation Model Fine-Tuning

TerraTorch is a fine-tuning framework for Geospatial Foundation Models (GFMs) built on PyTorch Lightning and TorchGeo.
It lets you combine pretrained backbones (Prithvi, TerraMind, Clay, SatMAE, DOFA, timm, SMP, etc.) with flexible
decoders to train downstream tasks: semantic segmentation, pixel-wise regression, classification, object detection,
and embedding generation.

## Core Workflow

Every TerraTorch project follows this pipeline:

1. **Pick a task** — SemanticSegmentationTask, PixelwiseRegressionTask, ClassificationTask, etc.
2. **Pick a backbone** — a pretrained encoder (e.g. `prithvi_eo_v2_300`)
3. **Pick a decoder** — UNetDecoder, UperNetDecoder, FCNDecoder, etc.
4. **Configure necks** — bridge encoder output shape to decoder input (SelectIndices, ReshapeTokensToImage, etc.)
5. **Set up data** — use generic datamodules or built-in datasets
6. **Train** — via CLI (`terratorch fit --config config.yaml`) or Python API

## Two Ways to Use TerraTorch

### Option A: YAML Config + CLI (recommended for most users)

```bash
terratorch fit --config my_config.yaml
terratorch test --config my_config.yaml --ckpt_path best.ckpt
terratorch predict -c my_config.yaml --ckpt_path best.ckpt --predict_output_dir output/
```

### Option B: Python API

```python
from lightning.pytorch import Trainer
import terratorch  # registers all components
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask

datamodule = GenericNonGeoSegmentationDataModule(...)
task = SemanticSegmentationTask(model_factory="EncoderDecoderFactory", model_args={...}, ...)
trainer = Trainer(max_epochs=100, precision="16-mixed")
trainer.fit(model=task, datamodule=datamodule)
```

## YAML Config Structure

A config has four sections: `trainer`, `data`, `model`, `optimizer`/`lr_scheduler`.
See `references/config_templates.md` for complete annotated templates for each task type.

```yaml
seed_everything: 0
trainer:
  accelerator: auto
  devices: auto
  max_epochs: 200
  precision: 16-mixed
  callbacks:
    - class_path: LearningRateMonitor
    - class_path: EarlyStopping
      init_args: { monitor: val/loss, patience: 20 }
    - class_path: ModelCheckpoint
      init_args: { monitor: val/loss, mode: min }
  default_root_dir: ./output

data:
  class_path: GenericNonGeoSegmentationDataModule
  init_args:
    batch_size: 16
    num_workers: 8
    train_data_root: data/train/images
    train_label_data_root: data/train/labels
    val_data_root: data/val/images
    val_label_data_root: data/val/labels
    means: [0.033, 0.045, 0.033, 0.274, 0.116, 0.076]
    stds:  [0.023, 0.026, 0.024, 0.029, 0.041, 0.035]
    num_classes: 2
    dataset_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    output_bands:  [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    img_grep: "*_merged.tif"
    label_grep: "*.mask.tif"

model:
  class_path: SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: prithvi_eo_v2_300
      backbone_pretrained: true
      backbone_num_frames: 1
      backbone_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
      necks:
        - name: SelectIndices
          indices: [5, 11, 17, 23]
        - name: ReshapeTokensToImage
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      decoder_channels: [512, 256, 128, 64]
      head_dropout: 0.1
      num_classes: 2
    loss: ce
    freeze_backbone: false

optimizer:
  class_path: torch.optim.AdamW
  init_args: { lr: 1.e-4, weight_decay: 0.05 }

lr_scheduler:
  class_path: ReduceLROnPlateau
  init_args: { monitor: val/loss, patience: 5, factor: 0.5 }
```

## Backbone Selection

The backbone is the pretrained encoder. Different backbones require different neck configurations because
ViT-based models output flat token sequences while CNN-based models output spatial feature maps.

### SelectIndices by Model Size

When using ViT-based backbones (Prithvi, SatMAE, etc.), you must select which transformer layers to extract:

| Model size | SelectIndices |
|---|---|
| ~100M params (tiny/small) | `[2, 5, 8, 11]` |
| ~300M params | `[5, 11, 17, 23]` |
| ~600M params | `[7, 15, 23, 31]` |

### Backbone Quick Reference

| Backbone | Type | Config name | Needs necks? |
|---|---|---|---|
| Prithvi EO V1 100M | ViT | `prithvi_eo_v1_100` | Yes |
| Prithvi EO V2 300M | ViT | `prithvi_eo_v2_300` | Yes |
| Prithvi EO V2 600M | ViT | `prithvi_eo_v2_600` | Yes |
| Prithvi EO V2 300M TL | ViT+Temporal/Location | `prithvi_eo_v2_300_tl` | Yes |
| Prithvi EO V2 600M TL | ViT+Temporal/Location | `prithvi_eo_v2_600_tl` | Yes |
| TerraMind | Multi-modal | See references/backbones.md | Yes |
| SatMAE | ViT | See references/backbones.md | Yes |
| ScaleMAE | ViT | See references/backbones.md | Yes |
| Clay v1 / v1.5 | ViT | See references/backbones.md | Yes |
| DOFA ViT | ViT (TorchGeo) | See references/backbones.md | Yes |
| timm models | CNN/ViT | `timm_<model_name>` | Depends |
| SMP models | CNN | Use SMPModelFactory | No |

For full backbone details including bands, parameters, and registry usage, read `references/backbones.md`.

## Decoder Selection

| Decoder | Best for | Key params |
|---|---|---|
| UNetDecoder | Segmentation with skip connections | `decoder_channels: [512,256,128,64]` |
| UperNetDecoder | Multi-scale feature fusion | `decoder_embed_dim`, `decoder_channels: 256` |
| FCNDecoder | Simple, fast | `decoder_channels: 256`, `decoder_num_convs: 4` |
| IdentityDecoder | Classification (passthrough) | `decoder_embed_dim` |
| LinearDecoder | Lightweight projection | `decoder_embed_dim` |
| MLPDecoder | Moderate complexity | - |

## Neck Configuration

Necks transform backbone outputs for decoder compatibility. ViT backbones always need necks.

**Standard neck chain for ViT backbone -> spatial decoder:**
```yaml
necks:
  - name: SelectIndices
    indices: [5, 11, 17, 23]       # pick transformer layers
  - name: ReshapeTokensToImage      # tokens -> spatial feature maps
  - name: LearnedInterpolateToPyramidal  # create multi-scale pyramid
```

**Available necks:** SelectIndices, ReshapeTokensToImage, LearnedInterpolateToPyramidal,
InterpolateToPyramidal, MaxpoolToPyramidal, PermuteDims, AddBottleneckLayer.

## Tasks Reference

| Task class | Use case | Losses |
|---|---|---|
| SemanticSegmentationTask | Pixel classification | ce, dice, jaccard, focal |
| PixelwiseRegressionTask | Continuous per-pixel values | mse, rmse, mae, huber |
| ClassificationTask | Image-level labels | ce, jaccard, focal |
| MultilabelClassificationTask | Multi-label images | - |
| ObjectDetectionTask | Bounding boxes | - |
| EmbeddingGenerationTask | Extract embeddings | - |

### Common Task Parameters

```yaml
model_factory: EncoderDecoderFactory
loss: ce                        # loss function
freeze_backbone: false          # freeze encoder weights
freeze_decoder: false           # freeze decoder weights
lr: 1.e-4
class_weights: [0.3, 0.7]      # for imbalanced classes (segmentation/classification)
ignore_index: -1                # label value to ignore
tiled_inference_parameters:     # for large images at test time
  h_crop: 224
  h_stride: 112
  w_crop: 224
  w_stride: 112
  average_patches: true
```

## Dataset Setup

TerraTorch's generic datasets require no custom code — just point to your data directories.

### Directory Structure Expected

```
data/
  train/
    images/    # GeoTIFF files
    labels/    # GeoTIFF masks (segmentation) or same-name files
  val/
    images/
    labels/
  test/
    images/
    labels/
```

### Split Files (alternative to separate directories)

Instead of separate train/val/test folders, use split files — text files with one filename prefix per line:
```
train_split: data/splits/train.txt
val_split: data/splits/val.txt
```

### Key DataModule Parameters

- `means`, `stds` — normalization stats (REQUIRED, compute from your data)
- `dataset_bands` — bands present in your files
- `output_bands` — bands to actually use (subset of dataset_bands)
- `img_grep`, `label_grep` — regex/glob to match files
- `constant_scale` — multiply pixel values (e.g., 0.0001 for scaling)
- `no_data_replace`, `no_label_replace` — handle nodata values
- `expand_temporal_dimension: true` — for multi-temporal data

For full dataset and datamodule details, read `references/datasets.md`.

## Advanced Features

### Multi-temporal Data
Set `backbone_num_frames: N` and use `expand_temporal_dimension: true` in the datamodule.
Prithvi `_tl` models support temporal+location encoding natively.

### PEFT (LoRA)
```yaml
peft_config:
  method: LORA
  replace_qkv: qkv
  peft_config_kwargs:
    target_modules: [qkv.q_linear, qkv.v_linear, mlp.fc1, mlp.fc2]
```

### Auxiliary Heads
```yaml
aux_heads:
  - name: aux_head
    decoder: FCNDecoder
    decoder_args: { decoder_channels: 256, decoder_in_index: -1 }
aux_loss:
  aux_head: 1.0
```

### Multi-modal Data
Use `GenericMultiModalDataModule` with modality-specific data roots:
```yaml
data:
  class_path: GenericMultiModalDataModule
  init_args:
    modalities: [S2L1C, S1]
    train_data_root: { S2L1C: data/s2, S1: data/s1 }
```

### Layer-wise Learning Rates
```yaml
lr_overrides:
  backbone: 1.e-5
  decoder: 1.e-4
```

## Spectral Bands Reference

Common band names used in configs (HLSBands enum):
`BLUE`, `GREEN`, `RED`, `NIR_NARROW`, `NIR_BROAD`, `SWIR_1`, `SWIR_2`,
`COASTAL_AEROSOL`, `RED_EDGE_1`, `RED_EDGE_2`, `RED_EDGE_3`, `WATER_VAPOR`, `CIRRUS`

## Installation

```bash
# Requires Python 3.10-3.12, GDAL
conda install -c conda-forge gdal
pip install terratorch
```

## Bundled References

Read these for deeper detail on specific topics:
- `references/config_templates.md` — Full YAML templates for every task type (segmentation, regression, classification, multi-temporal, multi-modal, SMP, PEFT)
- `references/backbones.md` — Complete backbone catalog with registry usage, band configs, and model-specific notes
- `references/datasets.md` — Generic and built-in datasets, datamodule params, transforms
