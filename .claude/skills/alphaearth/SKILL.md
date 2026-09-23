---
name: alphaearth
description: >
  Downstream applications with Google AlphaEarth Foundations Satellite Embeddings — classification,
  regression, similarity search, change detection, and clustering using the 64-band embedding dataset
  in Earth Engine or locally with scikit-learn. Use this skill whenever the user wants to classify
  land cover, detect changes, find similar areas, or train ML models using AlphaEarth / Google Satellite
  Embeddings, even if they just say "satellite embeddings" or "AlphaEarth" without specifying a task.
  Also trigger when the user mentions the Earth Engine collection `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`,
  `gs://alphaearth_foundations`, few-shot geospatial classification, or embedding-based land cover mapping.
---

# AlphaEarth Foundations — Downstream Applications

Google's AlphaEarth Foundations compresses petabytes of Earth observation data (Landsat, Sentinel-1,
Sentinel-2, LiDAR, radar) into **64-dimensional unit vectors** at 10 m resolution, annually from 2017.
These embeddings are analysis-ready features that work with simple classifiers — no deep learning needed.

The key insight: because the embeddings already encode rich spectral-temporal semantics, **10s to 100s of
labeled samples** are enough for high-quality classification (few-shot learning). This makes them ideal
for rapid land cover mapping, change detection, and similarity search.

## Quick Reference

| Property | Value |
|---|---|
| Collection ID | `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL` |
| GCS bucket | `gs://alphaearth_foundations` |
| Bands | `A00`–`A63` (64 dimensions) |
| Resolution | 10 m |
| Temporal range | 2017–2025 |
| Value range | -1 to 1 (unit-length vectors) |
| License | CC-BY 4.0 |

## Two Access Paths

### Path A: Earth Engine Python API (recommended for most tasks)

Best for: server-side classification, large-area analysis, export to Drive/Asset.

```python
import ee
import geemap  # for interactive map visualization in Jupyter

ee.Authenticate()  # one-time; opens browser for OAuth
ee.Initialize(project='your-project-id')

embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')

# Always filter by BOTH date and bounds before mosaic
year = 2024
start = ee.Date.fromYMD(year, 1, 1)
end = start.advance(1, 'year')

emb_image = (embeddings
    .filter(ee.Filter.date(start, end))
    .filter(ee.Filter.bounds(geometry))
    .mosaic())

# Create interactive map (Jupyter notebooks)
Map = geemap.Map()
Map.centerObject(geometry, 12)
```

### Path B: Google Cloud Storage + local Python

Best for: custom ML pipelines, scikit-learn/PyTorch workflows, offline analysis.

```bash
# Download a UTM zone (GeoTIFF tiles)
gsutil -m cp -r "gs://alphaearth_foundations/v1.1/2024/UTM_32N/" ./data/
```

```python
import rasterio
import numpy as np

with rasterio.open('data/tile.tif') as src:
    embeddings = src.read()  # shape: (64, H, W)
    transform = src.transform
```

## Core Workflow: Few-Shot Classification

This is the most common use case. You have a small set of labeled points and want to classify
a region. The workflow is the same conceptually whether you use Earth Engine or local Python.

### Earth Engine Workflow

#### 1. Prepare labeled points

Create a FeatureCollection with a class property. Points can come from manual digitization,
a CSV, or an existing asset. Even 20-30 points per class can produce good results.

```python
# From manually defined coordinates
mangroves = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([lon, lat]), {'landcover': 0}),
    # ... more points
])
water = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([lon, lat]), {'landcover': 1}),
])
other = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([lon, lat]), {'landcover': 2}),
])
gcps = mangroves.merge(water).merge(other)

# OR from a CSV with lat, lon, label columns
import pandas as pd
df = pd.read_csv('training_points.csv')
features = [
    ee.Feature(ee.Geometry.Point([r['lon'], r['lat']]), {'landcover': int(r['label'])})
    for _, r in df.iterrows()
]
gcps = ee.FeatureCollection(features)
```

#### 2. Sample embeddings at labeled locations

```python
training = emb_image.sampleRegions(
    collection=gcps,
    properties=['landcover'],
    scale=10
)
```

#### 3. Train a classifier

kNN and Random Forest both work well. kNN is a natural fit because the embedding space is
designed so that similar surfaces cluster together — nearest-neighbor lookup directly exploits this.

```python
# kNN — good default, fast, works well with few samples
classifier = ee.Classifier.smileKNN(k=5).train(
    features=training,
    classProperty='landcover',
    inputProperties=emb_image.bandNames()
)

# Random Forest — better when you have more samples or noisy labels
classifier = ee.Classifier.smileRandomForest(numberOfTrees=50).train(
    features=training,
    classProperty='landcover',
    inputProperties=emb_image.bandNames()
)
```

#### 4. Classify and visualize

```python
classified = emb_image.classify(classifier)

# Visualization
palette = ['green', 'blue', 'gray']  # one color per class
vis = {'min': 0, 'max': 2, 'palette': palette}
Map.addLayer(classified.clip(geometry), vis, 'Classification')

# Extract a single class as binary mask
target_class = classified.eq(0).selfMask()
```

#### 5. Accuracy assessment

```python
# Split data for validation (or use separate validation points)
withRandom = training.randomColumn('random')
train_set = withRandom.filter(ee.Filter.lt('random', 0.7))
test_set = withRandom.filter(ee.Filter.gte('random', 0.7))

classifier = ee.Classifier.smileRandomForest(50).train(
    features=train_set,
    classProperty='landcover',
    inputProperties=emb_image.bandNames()
)

validated = test_set.classify(classifier)
error_matrix = validated.errorMatrix('landcover', 'classification')
print('Overall accuracy:', error_matrix.accuracy().getInfo())
print('Kappa:', error_matrix.kappa().getInfo())
print('Confusion matrix:', error_matrix.array().getInfo())
```

#### 6. Export results

```python
# To Google Drive
task = ee.batch.Export.image.toDrive(
    image=classified.clip(geometry).toInt8(),
    description='landcover_classification',
    folder='EarthEngine',
    scale=10,
    region=geometry,
    crs='EPSG:4326',
    maxPixels=1e10
)
task.start()

# To Earth Engine Asset (use pyramidingPolicy MODE for discrete classes)
task = ee.batch.Export.image.toAsset(
    image=classified.clip(geometry).toInt8(),
    description='landcover_asset',
    assetId='projects/your-project/assets/landcover',
    scale=10,
    region=geometry,
    pyramidingPolicy={'classification': 'MODE'},
    maxPixels=1e10
)
task.start()
```

### Local Python (scikit-learn) Workflow

Use this when you want more control over the ML pipeline, need cross-validation, or want
to use models not available in Earth Engine.

#### 1. Extract embeddings at labeled points

The recommended approach is batch extraction via `sampleRegions` + export, then load
the CSV locally. Avoid calling `getInfo()` in a loop — it makes one HTTP request per
point and is extremely slow for more than ~50 points.

```python
import numpy as np
import pandas as pd
import ee

ee.Initialize(project='your-project-id')

embeddings_col = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
emb_image = (embeddings_col
    .filterDate('2024-01-01', '2025-01-01')
    .filterBounds(ee.Geometry.BBox(xmin, ymin, xmax, ymax))
    .mosaic())

bands = [f'A{i:02d}' for i in range(64)]
labeled = pd.read_csv('labeled_points.csv')  # columns: lon, lat, label

# Option A: Batch extraction (recommended, fast)
points_fc = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([r['lon'], r['lat']]), {'label': int(r['label'])})
    for _, r in labeled.iterrows()
])
sampled = emb_image.sampleRegions(collection=points_fc, properties=['label'], scale=10)
task = ee.batch.Export.table.toDrive(
    collection=sampled, description='training_embeddings', fileFormat='CSV')
task.start()
# ... wait for export, then load:
# df = pd.read_csv('training_embeddings.csv')
# X = df[[f'A{i:02d}' for i in range(64)]].values
# y = df['label'].values

# Option B: Small datasets only (<50 points) — direct extraction
features = []
for _, row in labeled.iterrows():
    point = ee.Geometry.Point([row['lon'], row['lat']])
    vals = emb_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point.buffer(10),
        scale=10
    ).getInfo()
    features.append([vals[b] for b in bands])

X = np.array(features)
y = labeled['label'].values
```

#### 2. Train and evaluate

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import classification_report

# Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
print(f'CV accuracy: {scores.mean():.3f} +/- {scores.std():.3f}')

clf.fit(X, y)
print(classification_report(y, clf.predict(X), target_names=class_names))
```

#### 3. Predict on new areas (from local GeoTIFF)

```python
import rasterio
from rasterio.transform import rowcol

with rasterio.open('embedding_tile.tif') as src:
    data = src.read()  # (64, H, W)
    profile = src.profile

# Reshape to (N_pixels, 64) for prediction
H, W = data.shape[1], data.shape[2]
flat = data.reshape(64, -1).T  # (H*W, 64)

# Remove nodata pixels
valid = ~np.isnan(flat).any(axis=1)
predictions = np.full(H * W, -1, dtype=np.int8)
predictions[valid] = clf.predict(flat[valid])
result = predictions.reshape(H, W)

# Save classified raster
profile.update(count=1, dtype='int8')
with rasterio.open('classified.tif', 'w', **profile) as dst:
    dst.write(result, 1)
```

## Other Downstream Tasks

For detailed workflows on similarity search, change detection, regression, and clustering,
see `references/advanced_workflows.md`.

**Quick pointers:**

- **Similarity search**: dot product between a reference embedding and all pixels gives a
  similarity score (0–1). High values = similar surface conditions.
- **Change detection**: dot product between embeddings from two different years. Values near 1
  mean no change; values below ~0.8 indicate meaningful surface change.
- **Regression**: same workflow as classification but use `ee.Classifier.smileRandomForest`
  in `.setOutputMode('REGRESSION')` or scikit-learn regressors locally.
- **Unsupervised clustering**: sample random pixels, train `ee.Clusterer.wekaKMeans`, apply
  to full image. Good for exploratory analysis when you have no labels.

## Common Pitfalls

1. **Forgetting to filter by bounds** — without `.filterBounds()`, Earth Engine loads tiles
   globally and the mosaic will be slow or fail.
2. **Using individual bands** — bands A00–A63 are not independently interpretable. Always use
   all 64 together as a feature vector.
3. **Over-engineering the model** — simple classifiers (kNN, RF) work better than deep learning
   here because the embeddings already encode the complex patterns. Start simple.
4. **Too few classes** — if your "other" class is too heterogeneous, split it into subclasses
   (e.g., bare soil, urban, vegetation) for better results.
5. **Wrong export pyramiding** — use `pyramidingPolicy: 'MODE'` for classification results
   (discrete values), not the default mean.
