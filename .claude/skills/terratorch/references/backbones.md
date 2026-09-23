# TerraTorch Backbone Reference

Complete guide to all supported backbones, their configuration, and how to switch between them.

## Table of Contents
1. [Backbone Registry](#backbone-registry)
2. [Prithvi Family](#prithvi-family)
3. [TerraMind](#terramind)
4. [SatMAE](#satmae)
5. [ScaleMAE](#scalemae)
6. [Clay](#clay)
7. [DOFA](#dofa)
8. [timm Models](#timm-models)
9. [SMP Models](#smp-models)
10. [TorchGeo Models](#torchgeo-models)
11. [Other Backbones](#other-backbones)
12. [How to Switch Backbones](#how-to-switch-backbones)

---

## Backbone Registry

TerraTorch uses a central registry to manage all backbones. You can query it programmatically:

```python
from terratorch import BACKBONE_REGISTRY

# List all available backbones
print(list(BACKBONE_REGISTRY))

# Filter by name
prithvi_models = [m for m in BACKBONE_REGISTRY if "prithvi" in m]

# Check if a backbone exists
assert "prithvi_eo_v2_300" in BACKBONE_REGISTRY

# Build a backbone
model = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_300",
    pretrained=True,
    bands=["RED", "GREEN", "BLUE", "NIR_NARROW", "SWIR_1", "SWIR_2"],
    num_frames=1,
)

# Build with custom checkpoint
model = BACKBONE_REGISTRY.build(
    "prithvi_eo_v2_300",
    ckpt_path="path/to/custom_weights.pt",
    bands=["RED", "GREEN", "BLUE"],
)
```

---

## Prithvi Family

IBM/NASA Prithvi Earth Observation foundation models. ViT-based, pretrained on HLS data.

### Available Models

| Config name | Params | Architecture | Bands (pretrained) | Temporal |
|---|---|---|---|---|
| `prithvi_eo_v1_100` | 100M | ViT | 6 HLS bands | Yes |
| `prithvi_eo_v2_300` | 300M | ViT | 6 HLS bands | Yes |
| `prithvi_eo_v2_600` | 600M | ViT | 6 HLS bands | Yes |
| `prithvi_eo_v2_300_tl` | 300M | ViT + Temporal/Location encoding | 6 HLS bands | Yes + location |
| `prithvi_eo_v2_600_tl` | 600M | ViT + Temporal/Location encoding | 6 HLS bands | Yes + location |
| `prithvi_eo_tiny` | ~25M | ViT (small) | 6 HLS bands | Yes |

### Standard HLS Bands
The 6 bands Prithvi was pretrained on: `BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2`

### Neck Configuration by Model

**prithvi_eo_v1_100 / prithvi_eo_tiny:**
```yaml
necks:
  - name: SelectIndices
    indices: [2, 5, 8, 11]          # 12-layer ViT
  - name: ReshapeTokensToImage
  - name: LearnedInterpolateToPyramidal
```

**prithvi_eo_v2_300 / prithvi_eo_v2_300_tl:**
```yaml
necks:
  - name: SelectIndices
    indices: [5, 11, 17, 23]        # 24-layer ViT
  - name: ReshapeTokensToImage
  - name: LearnedInterpolateToPyramidal
```

**prithvi_eo_v2_600 / prithvi_eo_v2_600_tl:**
```yaml
necks:
  - name: SelectIndices
    indices: [7, 15, 23, 31]        # 32-layer ViT
  - name: ReshapeTokensToImage
  - name: LearnedInterpolateToPyramidal
```

### Temporal/Location Models (_tl)

The `_tl` variants accept temporal and location metadata for improved performance:
- Set `backbone_num_frames` to number of time steps
- Location encoding is automatic from GeoTIFF metadata
- Best for multi-temporal tasks where acquisition date matters

### Example Config (Prithvi V2 300M)

```yaml
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
```

---

## TerraMind

Multi-modal geospatial foundation model supporting diverse input modalities.

### Key Features
- Supports multiple input modalities (optical, SAR, DEM, etc.)
- Can be used with GenericMultiModalDataModule
- ViT-based architecture

### Config Pattern
```yaml
model_args:
  backbone: terramind_v1          # check BACKBONE_REGISTRY for exact names
  backbone_pretrained: true
  # TerraMind-specific parameters depend on modalities used
  necks:
    - name: SelectIndices
      indices: [5, 11, 17, 23]    # adjust based on model depth
    - name: ReshapeTokensToImage
    - name: LearnedInterpolateToPyramidal
  decoder: UNetDecoder
  decoder_channels: [512, 256, 128, 64]
```

### Notes
- Check `BACKBONE_REGISTRY` for latest available TerraMind variants
- Multi-modal usage requires `GenericMultiModalDataModule`
- Supports both single-modal and multi-modal operation

---

## SatMAE

Satellite MAE (Masked Autoencoder) pretrained on satellite imagery.

### Config Pattern
```yaml
# Option 1: via EncoderDecoderFactory
model_args:
  backbone: satmae_vit_large      # check registry for exact name
  backbone_pretrained: true
  necks:
    - name: SelectIndices
      indices: [5, 11, 17, 23]
    - name: ReshapeTokensToImage
    - name: LearnedInterpolateToPyramidal
  decoder: UNetDecoder
  decoder_channels: [512, 256, 128, 64]

# Option 2: via dedicated SatMAEModelFactory
model_factory: SatMAEModelFactory
model_args:
  backbone: satmae_vit_large
  backbone_pretrained: true
```

---

## ScaleMAE

Scale-aware Masked Autoencoder for multi-scale geospatial representation.

### Config Pattern
```yaml
model_args:
  backbone: scalemae_large        # check registry for exact name
  backbone_pretrained: true
  necks:
    - name: SelectIndices
      indices: [5, 11, 17, 23]
    - name: ReshapeTokensToImage
    - name: LearnedInterpolateToPyramidal
  decoder: UNetDecoder
  decoder_channels: [512, 256, 128, 64]
```

---

## Clay

Clay foundation model for Earth observation.

### Available Versions
- `clay_v1` — original Clay model
- `clay_v1.5` — improved version

### Config Pattern
```yaml
# Via EncoderDecoderFactory
model_args:
  backbone: clay_v1               # or clay_v15
  backbone_pretrained: true
  necks:
    - name: SelectIndices
      indices: [5, 11, 17, 23]
    - name: ReshapeTokensToImage
    - name: LearnedInterpolateToPyramidal
  decoder: UNetDecoder
  decoder_channels: [512, 256, 128, 64]

# Or via dedicated ClayModelFactory
model_factory: ClayModelFactory
```

---

## DOFA

Dynamic One-For-All ViT model (via TorchGeo). Supports dynamic band adaptation.

### Config Pattern
```yaml
model_args:
  backbone: dofa_vit_base         # check registry
  backbone_pretrained: true
  backbone_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
  necks:
    - name: SelectIndices
      indices: [2, 5, 8, 11]
    - name: ReshapeTokensToImage
    - name: LearnedInterpolateToPyramidal
  decoder: UNetDecoder
  decoder_channels: [512, 256, 128, 64]
```

### Notes
- DOFA dynamically adapts to different band configurations
- Good choice when working with non-standard band combinations

---

## timm Models

Any model from the `timm` library (thousands of pretrained models). Prefix with `timm_`.

### Config Pattern
```yaml
model_args:
  backbone: timm_resnet50         # timm_ + model name
  backbone_pretrained: true       # uses ImageNet weights
  # timm CNN models typically don't need necks
  # timm ViT models need necks similar to Prithvi
  decoder: UNetDecoder
  decoder_channels: [512, 256, 128, 64]
```

### Common timm Backbones
- `timm_resnet50`, `timm_resnet101`
- `timm_efficientnet_b0` through `timm_efficientnet_b7`
- `timm_swin_base_patch4_window7_224`
- `timm_vit_base_patch16_224`
- `timm_convnext_base`

### Notes
- ImageNet-pretrained only (not geospatial-pretrained)
- CNN models output spatial features directly — usually no necks needed
- ViT models need SelectIndices + ReshapeTokensToImage necks
- `in_channels` may need adjustment for multi-band geospatial data

---

## SMP Models

Segmentation Models PyTorch — use `SMPModelFactory` instead of `EncoderDecoderFactory`.
SMP handles the full encoder-decoder pipeline internally, so no necks are needed.

### Config Pattern
```yaml
model:
  class_path: SemanticSegmentationTask
  init_args:
    model_factory: SMPModelFactory
    model_args:
      model: Unet                    # SMP architecture
      backbone: resnet50             # SMP encoder name
      in_channels: 6                 # number of input bands
      num_classes: 2
      encoder_weights: imagenet      # or null for random init
```

### Available SMP Architectures
`Unet`, `UnetPlusPlus`, `DeepLabV3`, `DeepLabV3Plus`, `FPN`, `PSPNet`, `PAN`, `MAnet`, `LinkNet`

### Available SMP Encoders
`resnet18/34/50/101`, `efficientnet-b0` to `b7`, `mobilenet_v2`, `dpn68/92/107/131`,
`vgg11/13/16/19`, `densenet121/161/169/201`, `senet154`, `resnext50_32x4d`, etc.

### Notes
- SMP models are CNN-based and well-tested for segmentation
- Good baseline — faster training, lower memory than ViT models
- ImageNet-pretrained (not geospatial-specific)
- Simpler config since no necks needed

---

## TorchGeo Models

Models from the TorchGeo library, some pretrained on geospatial data.

### Available Models
- **ResNet** (SSL4EO pretrained): `torchgeo_resnet18`, `torchgeo_resnet50`
- **Swin** (Satlas pretrained): `torchgeo_swin_v2_base`
- **ViT** (SSL4EO/SatMAE pretrained): `torchgeo_vit_small`, `torchgeo_vit_base`

### Config Pattern
```yaml
model_args:
  backbone: torchgeo_resnet50
  backbone_pretrained: true
  # ResNet: no necks needed
  # Swin/ViT: needs necks
  decoder: UNetDecoder
  decoder_channels: [512, 256, 128, 64]
```

---

## Other Backbones

### DINOv3
```yaml
model_args:
  backbone: dinov3_vit_base       # check registry
  backbone_pretrained: true
```

### MMEarth ConvNeXtV2
```yaml
model_args:
  backbone: mmearth_convnextv2_base  # check registry
  backbone_pretrained: true
```

### Satlas (via TorchGeo)
```yaml
model_args:
  backbone: torchgeo_swin_v2_base  # Satlas weights
  backbone_pretrained: true
```

---

## How to Switch Backbones

Changing backbone is a common operation. Here's what you need to update:

### Step 1: Change `backbone` name
```yaml
backbone: prithvi_eo_v2_600    # was prithvi_eo_v2_300
```

### Step 2: Update `SelectIndices` for the new model depth
```yaml
# 100M models:  [2, 5, 8, 11]
# 300M models:  [5, 11, 17, 23]
# 600M models:  [7, 15, 23, 31]
```

### Step 3: Update `backbone_bands` if the new model expects different bands
```yaml
backbone_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
```

### Step 4: If switching between ViT and CNN
- **ViT -> CNN**: Remove necks (CNN outputs spatial features directly)
- **CNN -> ViT**: Add necks (SelectIndices + ReshapeTokensToImage + InterpolateToPyramidal)

### Step 5: If switching to SMP
- Change `model_factory` to `SMPModelFactory`
- Replace all `model_args` with SMP-specific params
- Remove all necks

### Example: Prithvi V2 300M -> DOFA ViT

```yaml
# Before (Prithvi V2 300M)
model_args:
  backbone: prithvi_eo_v2_300
  backbone_pretrained: true
  necks:
    - name: SelectIndices
      indices: [5, 11, 17, 23]       # 24-layer model
    - name: ReshapeTokensToImage
    - name: LearnedInterpolateToPyramidal

# After (DOFA ViT Base — 12-layer model)
model_args:
  backbone: dofa_vit_base
  backbone_pretrained: true
  backbone_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
  necks:
    - name: SelectIndices
      indices: [2, 5, 8, 11]         # 12-layer model
    - name: ReshapeTokensToImage
    - name: LearnedInterpolateToPyramidal
```

### Example: Prithvi V2 -> SMP ResNet50

```yaml
# Before (Prithvi V2)
model_factory: EncoderDecoderFactory
model_args:
  backbone: prithvi_eo_v2_300
  backbone_pretrained: true
  necks: [...]
  decoder: UNetDecoder
  decoder_channels: [512, 256, 128, 64]
  num_classes: 2

# After (SMP ResNet50)
model_factory: SMPModelFactory
model_args:
  model: Unet
  backbone: resnet50
  in_channels: 6
  num_classes: 2
  encoder_weights: imagenet
```
