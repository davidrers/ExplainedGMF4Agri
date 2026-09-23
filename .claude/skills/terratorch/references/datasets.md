# TerraTorch Datasets & DataModules Reference

## Table of Contents
1. [Generic Datasets (No Custom Code)](#generic-datasets)
2. [Generic DataModules](#generic-datamodules)
3. [Built-in Datasets](#built-in-datasets)
4. [Transforms](#transforms)
5. [Computing Band Statistics](#computing-band-statistics)
6. [Data Directory Patterns](#data-directory-patterns)

---

## Generic Datasets

These require no custom classes — configure everything at runtime via YAML or Python kwargs.

### GenericNonGeoSegmentationDataset
For pixel-level classification tasks (land cover, flood, burn scar, etc.).

**Key Parameters:**
- `data_root` — directory containing images
- `label_data_root` — directory containing label masks
- `image_grep` — regex/glob to match image files (e.g., `"*.tif"`, `"*_merged.tif"`)
- `label_grep` — regex/glob to match label files (e.g., `"*.mask.tif"`)
- `split` — path to text file with newline-separated filename prefixes
- `dataset_bands` — list of band names in the files
- `output_bands` — subset of bands to actually load
- `num_classes` — number of classes
- `rgb_indices` — indices for RGB visualization (e.g., `[2, 1, 0]`)
- `constant_scale` — multiply pixel values (e.g., `0.0001`)
- `no_data_replace` — value to replace nodata pixels with
- `no_label_replace` — value to replace nodata labels with
- `reduce_zero_label` — subtract 1 from all labels (when labels are 1-indexed)
- `expand_temporal_dimension` — reshape `(T*C, H, W)` to `(C, T, H, W)`
- `transform` — albumentations pipeline

### GenericNonGeoPixelwiseRegressionDataset
Same parameters as segmentation, but for continuous per-pixel targets.

### GenericNonGeoClassificationDataset
For image-level labels. Expects directory structure:
```
data_root/
  class_0/
    img1.tif
    img2.tif
  class_1/
    img3.tif
```
Or uses a split file with format: `filename label`

### GenericMultiModalDataset
For combining data from multiple sensors/modalities.
- `modalities` — list of modality names (e.g., `[S2, S1]`)
- Data roots, bands, stats are dicts keyed by modality name

---

## Generic DataModules

DataModules wrap datasets with train/val/test splits, batching, and transforms.

### GenericNonGeoSegmentationDataModule

```yaml
data:
  class_path: GenericNonGeoSegmentationDataModule
  init_args:
    # Data paths — can use shared root + splits, or separate directories
    train_data_root: data/train/images
    train_label_data_root: data/train/labels
    val_data_root: data/val/images
    val_label_data_root: data/val/labels
    test_data_root: data/test/images
    test_label_data_root: data/test/labels

    # OR use a single root with split files
    # data_root: data/images
    # label_data_root: data/labels
    # train_split: splits/train.txt
    # val_split: splits/val.txt
    # test_split: splits/test.txt

    # File matching
    img_grep: "*.tif"
    label_grep: "*.tif"

    # Normalization (REQUIRED)
    means: [0.033, 0.045, 0.033, 0.274, 0.116, 0.076]
    stds:  [0.023, 0.026, 0.024, 0.029, 0.041, 0.035]

    # Band configuration
    dataset_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    output_bands:  [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    num_classes: 2

    # Data loading
    batch_size: 16
    num_workers: 8

    # Augmentation
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

    # Optional
    rgb_indices: [2, 1, 0]
    no_data_replace: 0
    no_label_replace: -1
    constant_scale: 0.0001
    reduce_zero_label: false
    expand_temporal_dimension: false
    check_stackability: false    # verify all files have same dimensions

    # Prediction (inference) paths
    predict_data_root: data/predict/images
    predict_dataset_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
    predict_output_bands: [BLUE, GREEN, RED, NIR_NARROW, SWIR_1, SWIR_2]
```

### GenericNonGeoPixelwiseRegressionDataModule
Same interface as segmentation datamodule, but creates regression datasets.

### GenericNonGeoClassificationDataModule
Similar but for image-level classification. No `label_data_root` — labels come from directory structure or split file.

### GenericMultiModalDataModule
For multi-sensor data:
```yaml
data:
  class_path: GenericMultiModalDataModule
  init_args:
    task: segmentation            # or regression, classification
    modalities: [S2, S1]
    train_data_root:
      S2: data/train/sentinel2
      S1: data/train/sentinel1
    train_label_data_root: data/train/labels
    means:
      S2: [0.1, 0.2, 0.15, 0.3, 0.2, 0.1]
      S1: [0.5, 0.5]
    stds:
      S2: [0.05, 0.06, 0.05, 0.08, 0.07, 0.06]
      S1: [0.25, 0.25]
    concat_bands: true            # concatenate into single tensor
    num_classes: 2
    batch_size: 16
```

---

## Built-in Datasets

TerraTorch includes 20+ ready-to-use datasets. These auto-download and set up data.

### Segmentation / Burn / Fire
| Dataset class | Task | Description |
|---|---|---|
| `FireScarsNonGeo` | Segmentation | HLS burn scar detection |
| `BurnIntensityNonGeo` | Regression | Burn severity estimation |
| `Sen1Floods11NonGeo` | Segmentation | Flood mapping (S1+S2) |

### Biomass / Carbon
| Dataset class | Task | Description |
|---|---|---|
| `BioMasstersNonGeo` | Regression | Above-ground biomass |
| `CarbonFluxNonGeo` | Regression | Carbon flux estimation |

### Land Cover / Forest
| Dataset class | Task | Description |
|---|---|---|
| `OpenEarthMapNonGeo` | Segmentation | Global land cover (8 classes) |
| `ForestNetNonGeo` | Classification | Forest type classification |
| `MChesapeakeLandcover` | Segmentation | Chesapeake Bay land cover |
| `Landslide4SenseNonGeo` | Segmentation | Landslide detection |

### Crop / Agriculture
| Dataset class | Task | Description |
|---|---|---|
| `MultiTemporalCropClassification` | Segmentation | Multi-temporal crop type |
| `MSACropTypeNonGeo` | Segmentation | South Africa crop mapping |
| `Sen4AgriNet` | Segmentation | EU crop classification |
| `PASTIS` | Segmentation | Panoptic crop segmentation |

### Classification
| Dataset class | Task | Description |
|---|---|---|
| `MEuroSATNonGeo` | Classification | Land use (10 classes) |
| `MBigEarthNonGeo` | Multi-label classification | Sentinel-2 scene labels |
| `MSo2SatNonGeo` | Classification | Local climate zones |
| `MBrickKilnNonGeo` | Classification | Brick kiln detection |

### Other
| Dataset class | Task | Description |
|---|---|---|
| `MPv4gerNonGeo` | Segmentation | Solar panel detection |
| `OpenSentinelMap` | Segmentation | Open land cover map |

### Using Built-in Datasets
```yaml
data:
  class_path: FireScarsNonGeoDataModule   # or the corresponding DataModule
  init_args:
    batch_size: 16
    num_workers: 8
    data_root: ./data/firescars           # auto-downloads if not present
```

### GEO-Bench Datasets
All GEO-Bench benchmark datasets are supported. Check the registry for available names.

### TorchGeo Datasets
All TorchGeo datasets can be used via `TorchNonGeoDataModule`:
```yaml
data:
  class_path: TorchNonGeoDataModule
  init_args:
    cls: torchgeo.datasets.EuroSAT       # any TorchGeo dataset class
    batch_size: 32
```

---

## Transforms

TerraTorch transforms work with albumentations and add geospatial-specific operations.

### Standard Transform Pipeline
```yaml
train_transform:
  - class_path: albumentations.augmentations.geometric.resize.Resize
    init_args:
      height: 224
      width: 224
  - class_path: albumentations.augmentations.geometric.transforms.D4
    # D4 = random rotation (0/90/180/270) + random flip
  - class_path: ToTensorV2

val_transform:
  - class_path: albumentations.augmentations.geometric.resize.Resize
    init_args:
      height: 224
      width: 224
  - class_path: ToTensorV2
```

### Multi-temporal Transforms
When using temporal data, wrap augmentations with flatten/unflatten:
```yaml
train_transform:
  - class_path: FlattenTemporalIntoChannels
  - class_path: albumentations.augmentations.geometric.resize.Resize
    init_args:
      height: 224
      width: 224
  - class_path: albumentations.augmentations.geometric.transforms.D4
  - class_path: ToTensorV2
  - class_path: UnflattenTemporalFromChannels
    init_args:
      n_timesteps: 3       # must match number of time steps
```

### TerraTorch-specific Transforms
| Transform | Purpose |
|---|---|
| `FlattenTemporalIntoChannels` | Merge time dimension into channels for augmentation |
| `UnflattenTemporalFromChannels(n_timesteps)` | Restore temporal dimension after augmentation |
| `FlattenSamplesIntoChannels(time_dim)` | Merge sample dimension into channels |
| `UnflattenSamplesFromChannels` | Reverse sample flattening |
| `MultimodalTransforms` | Apply transforms across modalities |
| `Padding` | Pad image dimensions |
| `Rearrange` | einops-based tensor reshaping |
| `SelectBands` | Filter to specific bands |

---

## Computing Band Statistics

The `means` and `stds` are required for normalization. Compute them from your training data:

```python
import rasterio
import numpy as np
from pathlib import Path

def compute_band_stats(data_dir, pattern="*.tif"):
    """Compute per-band mean and std from a directory of GeoTIFFs."""
    files = list(Path(data_dir).glob(pattern))

    # First pass: compute means
    pixel_sum = None
    pixel_count = 0
    for f in files:
        with rasterio.open(f) as src:
            data = src.read().astype(np.float64)  # (bands, H, W)
            if pixel_sum is None:
                pixel_sum = np.zeros(data.shape[0])
            mask = ~np.isnan(data)
            pixel_sum += np.nansum(data, axis=(1, 2))
            pixel_count += mask.sum(axis=(1, 2))
    means = pixel_sum / pixel_count

    # Second pass: compute stds
    pixel_sq_diff_sum = np.zeros_like(means)
    for f in files:
        with rasterio.open(f) as src:
            data = src.read().astype(np.float64)
            for b in range(data.shape[0]):
                valid = data[b][~np.isnan(data[b])]
                pixel_sq_diff_sum[b] += np.sum((valid - means[b]) ** 2)
    stds = np.sqrt(pixel_sq_diff_sum / pixel_count)

    return means.tolist(), stds.tolist()

means, stds = compute_band_stats("data/train/images")
print(f"means: {means}")
print(f"stds: {stds}")
```

---

## Data Directory Patterns

### Pattern 1: Separate directories per split
```
data/
  train/
    images/
      scene_001.tif
      scene_002.tif
    labels/
      scene_001.tif
      scene_002.tif
  val/
    images/
    labels/
  test/
    images/
    labels/
```

### Pattern 2: Single directory with split files
```
data/
  images/
    scene_001.tif
    scene_002.tif
    scene_003.tif
  labels/
    scene_001.tif
    scene_002.tif
    scene_003.tif
  splits/
    train.txt      # scene_001\nscene_002
    val.txt        # scene_003
```

### Pattern 3: Classification (directory per class)
```
data/
  train/
    forest/
      img_001.tif
    urban/
      img_002.tif
    water/
      img_003.tif
  val/
    forest/
    urban/
    water/
```

### Pattern 4: Multi-temporal (stacked bands)
Each file contains all time steps stacked as bands: `(T*C, H, W)`
Use `expand_temporal_dimension: true` to reshape to `(C, T, H, W)`

### Pattern 5: Multi-modal (separate directories per modality)
```
data/
  train/
    sentinel2/
      scene_001.tif
    sentinel1/
      scene_001.tif
    labels/
      scene_001.tif
```
