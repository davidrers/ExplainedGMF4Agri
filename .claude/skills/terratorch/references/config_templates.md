# TerraTorch Config Templates

Complete YAML configuration templates for every major task type. Copy the relevant template and adapt to your data.

## Table of Contents
1. [Semantic Segmentation](#semantic-segmentation)
2. [Pixel-wise Regression](#pixel-wise-regression)
3. [Classification](#classification)
4. [Multi-temporal Segmentation](#multi-temporal-segmentation)
5. [Multi-modal Segmentation](#multi-modal-segmentation)
6. [SMP Models](#smp-models)
7. [PEFT / LoRA Fine-tuning](#peft--lora-fine-tuning)
8. [Object Detection](#object-detection)
9. [Embedding Generation](#embedding-generation)

---

## Semantic Segmentation

Binary or multi-class pixel-level classification (e.g., burn scars, flood mapping, land cover).

```yaml
seed_everything: 0

trainer:
  accelerator: auto
  devices: auto
  max_epochs: 200
  precision: 16-mixed
  callbacks:
    - class_path: RichProgressBar
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: EarlyStopping
      init_args:
        monitor: val/loss
        patience: 20
    - class_path: ModelCheckpoint
      init_args:
        dirpath: output/checkpoints
        monitor: val/loss
        mode: min
        filename: "best-{epoch:02d}-{val_loss:.4f}"
  check_val_every_n_epoch: 1
  log_every_n_steps: 50
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
    test_data_root: data/test/images
    test_label_data_root: data/test/labels
    img_grep: "*.tif"
    label_grep: "*.tif"
    means: [0.033, 0.045, 0.033, 0.274, 0.116, 0.076]
    stds:  [0.023, 0.026, 0.024, 0.029, 0.041, 0.035]
    num_classes: 2
    dataset_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    output_bands:  [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    rgb_indices: [2, 1, 0]
    no_data_replace: 0
    no_label_replace: -1
    train_transform:
      - class_path: albumentations.augmentations.geometric.resize.Resize
        init_args:
          height: 224
          width: 224
      - class_path: albumentations.augmentations.geometric.transforms.D4
      - class_path: ToTensorV2
    val_transform:
      - class_path: albumentations.augmentations.geometric.resize.Resize
        init_args:
          height: 224
          width: 224
      - class_path: ToTensorV2
    test_transform:
      - class_path: albumentations.augmentations.geometric.resize.Resize
        init_args:
          height: 224
          width: 224
      - class_path: ToTensorV2

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
    # class_weights: [0.3, 0.7]  # uncomment for imbalanced classes
    # ignore_index: -1            # uncomment to ignore a label value

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1.e-4
    weight_decay: 0.05

lr_scheduler:
  class_path: ReduceLROnPlateau
  init_args:
    monitor: val/loss
    patience: 5
    factor: 0.5
```

---

## Pixel-wise Regression

Continuous per-pixel prediction (e.g., biomass estimation, canopy height, carbon flux).

```yaml
seed_everything: 0

trainer:
  accelerator: auto
  devices: auto
  max_epochs: 200
  precision: 16-mixed
  callbacks:
    - class_path: RichProgressBar
    - class_path: LearningRateMonitor
      init_args:
        logging_interval: epoch
    - class_path: EarlyStopping
      init_args:
        monitor: val/loss
        patience: 20
    - class_path: ModelCheckpoint
      init_args:
        dirpath: output/checkpoints
        monitor: val/loss
        mode: min
  default_root_dir: ./output

data:
  class_path: GenericNonGeoPixelwiseRegressionDataModule
  init_args:
    batch_size: 16
    num_workers: 8
    train_data_root: data/train/images
    train_label_data_root: data/train/labels
    val_data_root: data/val/images
    val_label_data_root: data/val/labels
    img_grep: "*.tif"
    label_grep: "*.tif"
    means: [0.033, 0.045, 0.033, 0.274, 0.116, 0.076]
    stds:  [0.023, 0.026, 0.024, 0.029, 0.041, 0.035]
    dataset_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    output_bands:  [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    no_data_replace: 0
    no_label_replace: -1
    train_transform:
      - class_path: albumentations.augmentations.geometric.resize.Resize
        init_args:
          height: 224
          width: 224
      - class_path: albumentations.augmentations.geometric.transforms.D4
      - class_path: ToTensorV2
    val_transform:
      - class_path: albumentations.augmentations.geometric.resize.Resize
        init_args:
          height: 224
          width: 224
      - class_path: ToTensorV2

model:
  class_path: PixelwiseRegressionTask
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
      num_classes: 1
    loss: mse
    freeze_backbone: false

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1.e-4
    weight_decay: 0.05

lr_scheduler:
  class_path: ReduceLROnPlateau
  init_args:
    monitor: val/loss
    patience: 5
    factor: 0.5
```

---

## Classification

Image-level classification (e.g., EuroSAT land use, crop type).

```yaml
seed_everything: 0

trainer:
  accelerator: auto
  devices: auto
  max_epochs: 200
  precision: 16-mixed
  callbacks:
    - class_path: RichProgressBar
    - class_path: EarlyStopping
      init_args:
        monitor: val/loss
        patience: 20
    - class_path: ModelCheckpoint
      init_args:
        dirpath: output/checkpoints
        monitor: val/loss
        mode: min
  default_root_dir: ./output

data:
  class_path: GenericNonGeoClassificationDataModule
  init_args:
    batch_size: 32
    num_workers: 8
    train_data_root: data/train
    val_data_root: data/val
    test_data_root: data/test
    img_grep: "*.tif"
    means: [0.033, 0.045, 0.033, 0.274, 0.116, 0.076]
    stds:  [0.023, 0.026, 0.024, 0.029, 0.041, 0.035]
    num_classes: 10
    dataset_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    output_bands:  [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    train_transform:
      - class_path: albumentations.augmentations.geometric.resize.Resize
        init_args:
          height: 224
          width: 224
      - class_path: albumentations.augmentations.geometric.transforms.D4
      - class_path: ToTensorV2
    val_transform:
      - class_path: albumentations.augmentations.geometric.resize.Resize
        init_args:
          height: 224
          width: 224
      - class_path: ToTensorV2

model:
  class_path: ClassificationTask
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
      decoder: IdentityDecoder
      # For classification, IdentityDecoder passes features through
      # The head handles pooling and linear projection
      head_dropout: 0.2
      num_classes: 10
    loss: ce
    freeze_backbone: false

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1.e-4
    weight_decay: 0.05

lr_scheduler:
  class_path: CosineAnnealingLR
  init_args:
    T_max: 200
```

---

## Multi-temporal Segmentation

For time-series satellite data (e.g., crop classification across growing season).

```yaml
seed_everything: 0

trainer:
  accelerator: auto
  devices: auto
  max_epochs: 200
  precision: 16-mixed
  callbacks:
    - class_path: EarlyStopping
      init_args:
        monitor: val/loss
        patience: 20
    - class_path: ModelCheckpoint
      init_args:
        dirpath: output/checkpoints
        monitor: val/loss
        mode: min
  default_root_dir: ./output

data:
  class_path: GenericNonGeoSegmentationDataModule
  init_args:
    batch_size: 8
    num_workers: 8
    train_data_root: data/train/images
    train_label_data_root: data/train/labels
    val_data_root: data/val/images
    val_label_data_root: data/val/labels
    img_grep: "*.tif"
    label_grep: "*.tif"
    means: [0.033, 0.045, 0.033, 0.274, 0.116, 0.076,
            0.033, 0.045, 0.033, 0.274, 0.116, 0.076,
            0.033, 0.045, 0.033, 0.274, 0.116, 0.076]
    stds:  [0.023, 0.026, 0.024, 0.029, 0.041, 0.035,
            0.023, 0.026, 0.024, 0.029, 0.041, 0.035,
            0.023, 0.026, 0.024, 0.029, 0.041, 0.035]
    num_classes: 13
    dataset_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    output_bands:  [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    expand_temporal_dimension: true   # KEY: reshapes (T*C, H, W) -> (C, T, H, W)
    train_transform:
      - class_path: FlattenTemporalIntoChannels
      - class_path: albumentations.augmentations.geometric.resize.Resize
        init_args:
          height: 224
          width: 224
      - class_path: ToTensorV2
      - class_path: UnflattenTemporalFromChannels
        init_args:
          n_timesteps: 3
    val_transform:
      - class_path: FlattenTemporalIntoChannels
      - class_path: albumentations.augmentations.geometric.resize.Resize
        init_args:
          height: 224
          width: 224
      - class_path: ToTensorV2
      - class_path: UnflattenTemporalFromChannels
        init_args:
          n_timesteps: 3

model:
  class_path: SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: prithvi_eo_v2_300
      backbone_pretrained: true
      backbone_num_frames: 3           # number of time steps
      backbone_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
      necks:
        - name: SelectIndices
          indices: [5, 11, 17, 23]
        - name: ReshapeTokensToImage
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      decoder_channels: [512, 256, 128, 64]
      head_dropout: 0.1
      num_classes: 13
    loss: ce
    freeze_backbone: false

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 6.e-5
    weight_decay: 0.05

lr_scheduler:
  class_path: ReduceLROnPlateau
  init_args:
    monitor: val/loss
    patience: 5
```

---

## Multi-modal Segmentation

Combining data from different sensors (e.g., Sentinel-2 + Sentinel-1).

```yaml
seed_everything: 0

trainer:
  accelerator: auto
  devices: auto
  max_epochs: 200
  precision: 16-mixed
  default_root_dir: ./output

data:
  class_path: GenericMultiModalDataModule
  init_args:
    batch_size: 16
    num_workers: 8
    task: segmentation
    modalities: [S2, S1]
    train_data_root:
      S2: data/train/s2
      S1: data/train/s1
    train_label_data_root: data/train/labels
    val_data_root:
      S2: data/val/s2
      S1: data/val/s1
    val_label_data_root: data/val/labels
    means:
      S2: [0.033, 0.045, 0.033, 0.274, 0.116, 0.076]
      S1: [0.5, 0.5]
    stds:
      S2: [0.023, 0.026, 0.024, 0.029, 0.041, 0.035]
      S1: [0.25, 0.25]
    num_classes: 2
    concat_bands: true  # concatenate all modalities for single-backbone models

model:
  class_path: SemanticSegmentationTask
  init_args:
    model_factory: EncoderDecoderFactory
    model_args:
      backbone: prithvi_eo_v2_300
      backbone_pretrained: true
      backbone_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2, VV, VH]
      necks:
        - name: SelectIndices
          indices: [5, 11, 17, 23]
        - name: ReshapeTokensToImage
        - name: LearnedInterpolateToPyramidal
      decoder: UNetDecoder
      decoder_channels: [512, 256, 128, 64]
      num_classes: 2
    loss: ce

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1.e-4
    weight_decay: 0.05
```

---

## SMP Models

Using Segmentation Models PyTorch (ResNet, EfficientNet, etc.) via SMPModelFactory.
No necks needed — SMP handles encoder-decoder wiring internally.

```yaml
seed_everything: 0

trainer:
  accelerator: auto
  devices: auto
  max_epochs: 200
  precision: 16-mixed
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
    img_grep: "*.tif"
    label_grep: "*.tif"
    means: [0.033, 0.045, 0.033]
    stds:  [0.023, 0.026, 0.024]
    num_classes: 2

model:
  class_path: SemanticSegmentationTask
  init_args:
    model_factory: SMPModelFactory
    model_args:
      model: Unet                    # or DeepLabV3Plus, FPN, PSPNet, etc.
      backbone: resnet50             # any SMP-supported encoder
      in_channels: 3
      num_classes: 2
      encoder_weights: imagenet
    loss: dice

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1.e-4
```

---

## PEFT / LoRA Fine-tuning

Parameter-efficient fine-tuning — trains only adapter weights, keeping backbone frozen.

```yaml
seed_everything: 0

trainer:
  accelerator: auto
  devices: auto
  max_epochs: 100
  precision: 16-mixed
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
    img_grep: "*.tif"
    label_grep: "*.tif"
    means: [0.033, 0.045, 0.033, 0.274, 0.116, 0.076]
    stds:  [0.023, 0.026, 0.024, 0.029, 0.041, 0.035]
    num_classes: 2
    dataset_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    output_bands:  [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]

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
      num_classes: 2
      peft_config:
        method: LORA
        replace_qkv: qkv
        peft_config_kwargs:
          r: 16
          lora_alpha: 32
          target_modules:
            - qkv.q_linear
            - qkv.v_linear
            - mlp.fc1
            - mlp.fc2
    loss: ce
    freeze_backbone: false  # PEFT handles selective freezing

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 5.e-4
    weight_decay: 0.01
```

---

## Object Detection

```yaml
seed_everything: 0

trainer:
  accelerator: auto
  devices: auto
  max_epochs: 100
  precision: 16-mixed
  default_root_dir: ./output

model:
  class_path: ObjectDetectionTask
  init_args:
    model_factory: ObjectDetectionModelFactory
    model_args:
      backbone: prithvi_eo_v2_300
      backbone_pretrained: true
      backbone_bands: [RED, GREEN, BLUE]
      num_classes: 5
    loss: ce

optimizer:
  class_path: torch.optim.AdamW
  init_args:
    lr: 1.e-4
```

---

## Embedding Generation

Extract embeddings from a pretrained backbone (useful for downstream ML, clustering, retrieval).

```python
# Python API approach (simpler for embeddings)
import terratorch
from terratorch.tasks import EmbeddingGenerationTask
from lightning.pytorch import Trainer

task = EmbeddingGenerationTask(
    model_factory="EncoderDecoderFactory",
    model_args=dict(
        backbone="prithvi_eo_v2_300",
        backbone_pretrained=True,
        backbone_num_frames=1,
        backbone_bands=["BLUE", "GREEN", "RED", "NIR_NARROW", "SWIR_1", "SWIR_2"],
        necks=[
            {"name": "SelectIndices", "indices": [23]},  # last layer only
            {"name": "ReshapeTokensToImage"},
        ],
        decoder="IdentityDecoder",
    ),
)

trainer = Trainer(accelerator="auto")
embeddings = trainer.predict(model=task, datamodule=datamodule)
```
