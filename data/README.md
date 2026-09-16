# Data

Nothing in this directory is tracked in git, apart from this file.

## `eurocropsml/`

The EuroCropsML benchmark (Reuss et al., 2025), moved here from `C:\Users\darey\eurocropsml_data`.

```
eurocropsml/
  preprocess/       706,683 .npz files, one per parcel
  preprocess.zip    the source archive from Zenodo, redundant once extracted
```

Each `.npz` filename follows the pattern `<region><id>_<parcelid>_<crop class code>.npz`, for example
`EE001_19990038_3302000000.npz`, where the leading two letters give the country (EE Estonia, LV Latvia, PT Portugal)
and the trailing code is the harmonised HCAT crop class.

Code should locate this directory through the `EUROCROPSML_DATA` environment variable and fall back to
`<repo>/data/eurocropsml`:

```python
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("EUROCROPSML_DATA", Path(__file__).parents[2] / "data" / "eurocropsml"))
```

`preprocess.zip` is roughly 1.4 GB and duplicates the extracted tree. It can be deleted once the extraction is
verified, which also removes 1.4 GB from OneDrive synchronisation.

## `cropharvest/`

The CropHarvest benchmark (Tseng et al., NeurIPS 2021 Datasets and Benchmarks), downloaded from Zenodo
record [10251170](https://zenodo.org/records/10251170) (version v14, published 2023-12-04, concept DOI
`10.5281/zenodo.5021761`, licence CC-BY-SA-4.0). That is the record the current `cropharvest` package pins in
`cropharvest/config.py` as `DATASET_VERSION_ID`. Repository: https://github.com/nasaharvest/cropharvest.

The `cropharvest` PyPI package was **not** installed: version 0.7.0 requires `pandas<2.0.0` and
`geopandas==0.9.0`, which would downgrade the shared environment (pandas 2.3.3, geopandas 1.1.3). The
artefacts were therefore fetched directly from Zenodo with `curl` and are read with plain `h5py`,
`geopandas` and `pandas`.

```
cropharvest/
  labels.geojson              81.7 MB   113,893 labels, EPSG:4326, md5 54a5070f103bc3e635afba27c139ac8d
  features.tar.gz             78.7 MB   source archive, redundant once extracted, md5 d757e6c32cb6d65aa517f003607f6f81
  features/
    arrays/                  656.6 MB   87,464 .h5 files, one per label, each exactly 7,872 bytes
    normalizing_dict.h5        2 kB     per-band mean and standard deviation over the training instances
  aux/
    ne_10m_admin_0_countries.geojson   13.3 MB   Natural Earth 1:10m admin-0, used for the country and
                                                 region join (CropHarvest itself ships the coarser 1:50m)
```

Total 829.7 MB across 87,468 files. Both Zenodo files took about 22 s each at roughly 3.8 MB/s; extraction
of the 87,464 HDF5 files took 48 s.

Two further Zenodo files were **not** downloaded: `eo_data.tar.gz` (26.7 GB, the raw per-label GeoTIFF
exports from which `features/` was derived) and `test_features.tar.gz` (786 MB, the arrays for the
held-out benchmark regions in Kenya, Brazil, Togo and China). Neither is needed for the exploratory
analysis, and the held-out labels are still described in `labels.geojson` through the `is_test` column.

Feature files are named `<index>_<dataset>.h5`, for example `1137_togo.h5`, where `<index>` and
`<dataset>` join back onto the `index` and `dataset` columns of `labels.geojson`. Each file holds a single
dataset `array` of shape `(12, 18)` and dtype `float64`, plus the attributes `dataset`, `label`, `is_crop`,
`label_lat`, `label_lon`, `instance_lat` and `instance_lon`. The 18 channels are, in fixed order:

```
VV, VH,                                             Sentinel-1 GRD backscatter, dB
B2, B3, B4, B5, B6, B7, B8, B8A, B9, B11, B12,      Sentinel-2 L1C TOA reflectance x 1e4 (B1 and B10 dropped)
temperature_2m, total_precipitation,                ERA5, kelvin and metres
elevation, slope,                                   SRTM, metres and degrees, constant across timesteps
NDVI                                                derived from B8 and B4
```

The 12 timesteps are 30-day composites; the window ends on `export_end_date`, which is 1 February for
every label in the file, so the window runs from 1 February of the preceding year. `array` values are a
single 10 m pixel at the label coordinate, not an aggregate over the polygon.

Code should locate this directory through the `CROPHARVEST_DATA` environment variable and fall back to
`<repo>/data/cropharvest`. Re-download with:

```bash
mkdir -p data/cropharvest && cd data/cropharvest
curl -L -o labels.geojson  "https://zenodo.org/api/records/10251170/files/labels.geojson/content"
curl -L -o features.tar.gz "https://zenodo.org/api/records/10251170/files/features.tar.gz/content"
tar -xzf features.tar.gz
mkdir -p aux && curl -L -o aux/ne_10m_admin_0_countries.geojson   "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"
```

The analysis in `notebooks/02_cropharvest_eda.ipynb` and `results/eda/cropharvest/_run_analysis.py` reads
only `labels.geojson` and `features/arrays/`.

## `catalogue_parcels.parquet`

Parcel catalogue built during the proposal-phase exploratory analysis: one row per parcel, with the file path,
country, NUTS region, crop class, number of timesteps and centroid coordinates. Regenerated by
`results/eda/_run_analysis.py`.

## `commercial_crops_ml_ready.gpkg`

Precursor dataset from the earlier ML-Embeddings project, used by the scripts in `scripts/legacy/`. Not part of the
thesis experiments; retained because those scripts are the starting point for the AlphaEarth and TESSERA embedding
extraction.

## OneDrive

This directory sits inside the OneDrive tree, so the parcel files are synchronised to the cloud. If the OneDrive
client struggles with the file count, exclude the folder from synchronisation through
OneDrive settings, Account, Choose folders.
