# Advanced Downstream Workflows

## Table of Contents
- [Similarity Search](#similarity-search)
- [Change Detection](#change-detection)
- [Regression](#regression)
- [Unsupervised Clustering](#unsupervised-clustering)
- [PCA Visualization](#pca-visualization)
- [Multi-Year Analysis](#multi-year-analysis)
- [Batch Point Extraction](#batch-point-extraction)

---

## Similarity Search

Find areas worldwide that resemble a set of reference locations. This exploits the fact that
AlphaEarth embeddings are unit vectors — the dot product directly gives cosine similarity.

### Earth Engine

```python
import ee

ee.Initialize(project='your-project-id')

embeddings = ee.ImageCollection('GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL')
emb_image = (embeddings
    .filterDate('2024-01-01', '2025-01-01')
    .filterBounds(region)
    .mosaic())

bands = emb_image.bandNames()

# Extract embedding at a reference point
ref_point = ee.Geometry.Point([lon, lat])
ref_embedding = emb_image.reduceRegion(
    reducer=ee.Reducer.mean(),
    geometry=ref_point.buffer(500),  # 500m buffer for stable estimate
    scale=10
).getInfo()

# Build constant image from reference embedding
ref_values = [ref_embedding[b] for b in bands.getInfo()]
ref_img = ee.Image.constant(ref_values).rename(bands)

# Dot product similarity (pixel-wise)
similarity = emb_image.multiply(ref_img).reduce(ee.Reducer.sum())

# Visualize: white = dissimilar, dark = similar
Map.addLayer(similarity, {'min': 0.5, 'max': 1, 'palette': ['white', 'black']}, 'Similarity')

# Threshold to find highly similar areas
similar_areas = similarity.gte(0.95).selfMask()
```

### Multiple Reference Points

When you have several reference locations (e.g., known farms, forests, wetlands), compute
similarity to each and take the maximum:

```python
import pandas as pd
import numpy as np

refs = pd.read_csv('reference_locations.csv')  # lon, lat columns

similarities = []
for _, row in refs.iterrows():
    point = ee.Geometry.Point([row['lon'], row['lat']])
    ref_emb = emb_image.reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point.buffer(500),
        scale=10
    ).getInfo()
    ref_vals = [ref_emb[b] for b in bands.getInfo()]
    ref_img = ee.Image.constant(ref_vals).rename(bands)
    sim = emb_image.multiply(ref_img).reduce(ee.Reducer.sum())
    similarities.append(sim)

# Max similarity across all references
max_sim = ee.Image.cat(similarities).reduce(ee.Reducer.max())
```

### Local Python Similarity

```python
import numpy as np

# reference_embedding: shape (64,)
# pixel_embeddings: shape (N, 64)
similarities = pixel_embeddings @ reference_embedding  # dot product
# Values range 0-1 for unit vectors; >0.9 is very similar
```

---

## Change Detection

Compare embeddings between two years to detect surface changes. The dot product between
same-location embeddings from different years measures temporal stability.

### Earth Engine

```python
year_a, year_b = 2020, 2024

img_a = (embeddings
    .filterDate(f'{year_a}-01-01', f'{year_a + 1}-01-01')
    .filterBounds(geometry)
    .mosaic())

img_b = (embeddings
    .filterDate(f'{year_b}-01-01', f'{year_b + 1}-01-01')
    .filterBounds(geometry)
    .mosaic())

# Dot product = temporal similarity (1 = no change, 0 = total change)
similarity = img_a.multiply(img_b).reduce(ee.Reducer.sum())

# Visualize
Map.addLayer(similarity, {
    'min': 0.7, 'max': 1.0,
    'palette': ['red', 'yellow', 'green']
}, f'Change {year_a}-{year_b}')

# Binary change mask (threshold depends on your application)
change_mask = similarity.lt(0.85).selfMask()
Map.addLayer(change_mask, {'palette': ['red']}, 'Changed areas')
```

### Interpreting Change Scores

| Similarity | Interpretation |
|---|---|
| > 0.95 | No meaningful change |
| 0.85 – 0.95 | Subtle change (phenological, gradual) |
| 0.70 – 0.85 | Significant change (land use conversion) |
| < 0.70 | Dramatic change (deforestation, urbanization, flooding) |

These thresholds are approximate — calibrate for your specific landscape and classes.

### Angular Distance (alternative metric)

```python
# Angle between vectors (in radians) — more uniform scale than dot product
angle = img_a.multiply(img_b).reduce(ee.Reducer.sum()).acos()
# 0 = identical, pi/2 = orthogonal (very different)
```

---

## Regression

Predict continuous variables (canopy height, biomass, soil moisture) from embeddings.

### Earth Engine

```python
# Training points with continuous target variable
training_points = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([lon, lat]), {'biomass': 45.2}),
    # ...
])

training = emb_image.sampleRegions(
    collection=training_points,
    properties=['biomass'],
    scale=10
)

# Random Forest in regression mode
regressor = (ee.Classifier.smileRandomForest(numberOfTrees=100)
    .setOutputMode('REGRESSION')
    .train(
        features=training,
        classProperty='biomass',
        inputProperties=emb_image.bandNames()
    ))

predicted = emb_image.classify(regressor).rename('predicted_biomass')
```

### Local Python

```python
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score

# X: (N, 64) embeddings, y: (N,) continuous target
reg = RandomForestRegressor(n_estimators=200, random_state=42)
scores = cross_val_score(reg, X, y, cv=5, scoring='r2')
print(f'R2: {scores.mean():.3f} +/- {scores.std():.3f}')

reg.fit(X, y)
```

---

## Unsupervised Clustering

Discover natural groupings in the landscape without any labels. Useful for exploratory
analysis or generating a preliminary classification to refine with labels later.

### Earth Engine

```python
# Sample random pixels
n_samples = 5000
samples = emb_image.sample(
    region=geometry,
    scale=10,
    numPixels=n_samples,
    seed=42
)

# K-Means clustering
n_clusters = 8
clusterer = ee.Clusterer.wekaKMeans(nClusters=n_clusters).train(samples)
clustered = emb_image.cluster(clusterer)

# Visualize with random colors
Map.addLayer(clustered.randomVisualizer(), {}, f'K-Means {n_clusters} clusters')
```

### Choosing Number of Clusters

No single right answer — try a range (5, 10, 15, 20) and visually inspect which level of
detail matches your needs. Higher cluster counts capture finer landscape distinctions.

### Local Python with HDBSCAN

```python
from sklearn.cluster import KMeans
import hdbscan

# K-Means
kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings_array)

# HDBSCAN (density-based, auto-detects number of clusters)
clusterer = hdbscan.HDBSCAN(min_cluster_size=50)
labels = clusterer.fit_predict(embeddings_array)
```

---

## PCA Visualization

Reduce 64 dimensions to 2D or 3D for visual exploration. PCA on AlphaEarth embeddings
typically reveals meaningful spatial structure — the first 3 components often correspond
to broad land cover types.

### As RGB Composite

```python
from sklearn.decomposition import PCA
import numpy as np

# embeddings_array: shape (N, 64)
pca = PCA(n_components=3)
rgb = pca.fit_transform(embeddings_array)

# Normalize to 0-255 for display
rgb_norm = ((rgb - rgb.min(axis=0)) / (rgb.max(axis=0) - rgb.min(axis=0)) * 255).astype(np.uint8)
```

### Earth Engine PCA

```python
# Server-side PCA
band_names = emb_image.bandNames()
mean_dict = emb_image.reduceRegion(
    reducer=ee.Reducer.mean(), geometry=geometry, scale=100, maxPixels=1e8)
means = ee.Image.constant(mean_dict.values(band_names))
centered = emb_image.subtract(means)

arrays = centered.toArray()
covar = arrays.reduceRegion(
    reducer=ee.Reducer.covariance(), geometry=geometry, scale=100, maxPixels=1e8)
covar_array = ee.Array(covar.get('array'))
eigens = covar_array.eigen()
eigenvectors = eigens.slice(1, 1)

pc_image = (ee.Image(eigenvectors)
    .matrixMultiply(arrays.toArray2D().matrixTranspose())
    .arrayProject([0])
    .arrayFlatten([['PC1', 'PC2', 'PC3']])
)

Map.addLayer(pc_image, {
    'bands': ['PC1', 'PC2', 'PC3'],
    'min': -2, 'max': 2
}, 'PCA RGB')
```

---

## Multi-Year Analysis

### Temporal Stack

Combine embeddings from multiple years for temporal analysis:

```python
years = range(2017, 2025)
annual_images = []
for year in years:
    img = (embeddings
        .filterDate(f'{year}-01-01', f'{year + 1}-01-01')
        .filterBounds(geometry)
        .mosaic())
    annual_images.append(img)

# Extract multi-year embedding at a point (64 dims x N years)
point = ee.Geometry.Point([lon, lat])
for i, year in enumerate(years):
    vals = annual_images[i].reduceRegion(
        reducer=ee.Reducer.mean(),
        geometry=point.buffer(100),
        scale=10
    ).getInfo()
    # vals is a dict with A00..A63 keys
```

### Trend Detection

```python
# Compute pairwise similarity across consecutive years
for i in range(len(annual_images) - 1):
    sim = (annual_images[i]
        .multiply(annual_images[i + 1])
        .reduce(ee.Reducer.sum()))
    Map.addLayer(sim, {'min': 0.8, 'max': 1}, f'{years[i]}-{years[i+1]}')
```

---

## Batch Point Extraction

Efficiently extract embeddings for many points (e.g., field survey locations):

### Earth Engine (server-side, fast)

```python
# Best for < 5000 points
points_fc = ee.FeatureCollection([
    ee.Feature(ee.Geometry.Point([lon, lat]), {'id': site_id})
    for lon, lat, site_id in zip(lons, lats, ids)
])

sampled = emb_image.sampleRegions(
    collection=points_fc,
    properties=['id'],
    scale=10
)

# Export to CSV for local analysis
task = ee.batch.Export.table.toDrive(
    collection=sampled,
    description='embedding_samples',
    fileFormat='CSV'
)
task.start()
```

### For Large Point Sets (> 5000)

```python
# Break into batches to avoid Earth Engine memory limits
batch_size = 2000
for i in range(0, len(points), batch_size):
    batch = points[i:i + batch_size]
    batch_fc = ee.FeatureCollection([
        ee.Feature(ee.Geometry.Point([r['lon'], r['lat']]), {'id': r['id']})
        for _, r in batch.iterrows()
    ])
    sampled = emb_image.sampleRegions(
        collection=batch_fc, properties=['id'], scale=10)
    task = ee.batch.Export.table.toDrive(
        collection=sampled,
        description=f'embeddings_batch_{i // batch_size}',
        fileFormat='CSV'
    )
    task.start()
```
