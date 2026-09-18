# Data

Nothing in this directory is tracked in git, apart from this file.

## `eurocrops/`

The **EuroCrops vector release**, version 11, the source of the parcel polygons and the
declared crop labels. Zenodo record [14094196](https://zenodo.org/records/14094196), concept
DOI `10.5281/zenodo.6866846`, licence CC-BY-SA-4.0, published 2024. Repository and wiki:
https://github.com/maja601/EuroCrops.

EuroCrops is a **single declaration year per country**, not a time series. The four countries
downloaded here are all 2021, the season the thesis targets: Estonia, Latvia, Lithuania and
Portugal. Other countries in the release sit on other years (France 2018, Denmark 2019,
Netherlands and Finland 2020, Czechia, Brandenburg, Spain and Ireland 2023) and would not line
up with 2021 imagery.

```
eurocrops/
  raw/                        the Zenodo files, md5-verified, 1.4 GB
    EE_2021.zip  LV_2021.zip  LT_2021.zip  PT_2021.zip
    ee_2021.csv  lv_2021.csv  lt_2021.csv  pt_2021.csv   national crop code to HCAT mappings
    HCAT3.csv  HCAT2.csv                                 the harmonised taxonomy
  vector/<CC>/                the unpacked shapefiles, 2.3 GB
  parquet/<CC>_2021.parquet   one GeoParquet per country, native CRS preserved, 1.7 GB
  parquet/<CC>_2021.summary.json   schema, CRS, bounds, class counts, provenance
  derived/<CC>_2021_metrics.parquet   per-parcel geometry metrics from the EDA, git-ignored
  zenodo_record_14094196.json the full record metadata as downloaded
```

Re-download and rebuild with:

```bash
python scripts/data/fetch_eurocrops.py --countries EE LV LT PT
```

The script verifies every file against the md5 published by Zenodo, unpacks it, and writes the
GeoParquet with `write_covering_bbox=True` so a chip-sized bounding-box read is cheap. Nothing
is filtered, reprojected or reclassified on the way in.

Every country carries the three harmonisation attributes `EC_trans_n` (national crop name in
English), `EC_hcat_n` (HCAT3 name) and `EC_hcat_c` (the ten-digit HCAT3 code) beside its own
national attribute table.

| | parcels | CRS | declared area attribute | parcel identifier |
|---|---|---|---|---|
| EE | 176,064 | EPSG:4326 | `pindala_ha`, hectares | `pollu_id`, unique |
| LV | 432,188 | EPSG:3059 | `AREA_DECLA`, hectares | `PARCEL_ID`, 1,005 duplicates |
| LT | 1,102,471 | EPSG:4326 | `DKL_PLOTAS`, hectares | none |
| PT | 100,000 | EPSG:4326 | `OSA_AREA`, **square metres** | `OSA_ID`, unique |

Three traps worth remembering:

- **Portugal's `OSA_AREA` is in square metres** while every other country reports hectares. Its
  median parcel is 2,284 m², that is 0.23 ha.
- **Lithuania has no parcel identifier.** `KZS_NR` is a field-block number (278,891 distinct
  values) and `NMA_ID` a holding number (120,901), so an individual Lithuanian polygon cannot be
  addressed through the attribute table. Lithuania also carries only 22 HCAT classes, against
  128 for Estonia, because its national declaration scheme is far coarser.
- **Portugal is exactly 100,000 features**, which is consistent with a sampled release rather
  than the full national declaration.

## `eurocropsml/`

The **EuroCropsML benchmark** (Reuss et al., 2025), the per-parcel Sentinel-2 time series and the
official few-shot splits. Zenodo record [15095445](https://zenodo.org/records/15095445), concept
DOI `10.5281/zenodo.10629609`, licence CC-BY-SA-4.0, published 2025-03-31. Estonia, Latvia and
Portugal only.

```
eurocropsml/
  archives/                   the Zenodo files, md5-verified, 4.6 GB
    preprocess.zip  raw_data.zip  split.zip
  raw_data/
    <Country>.parquet         the unfiltered annual observation series per parcel
    geometries/<Country>.geojson   parcel polygons, CRS84, keyed by parcel_id
    labels/<Country>_labels.parquet  parcel_id, EC_hcat_c, EC_hcat_n
  split/<use case>/           the official pre-training, meta and fine-tuning splits
  preprocess/                 the 706,683 .npz files, NOT unpacked by default
  preprocess_index.parquet    706,683 rows parsed from the .npz filenames: NUTS3, parcel id, class
  zenodo_record_15095445.json the full record metadata as downloaded
```

Re-download with:

```bash
python scripts/data/fetch_eurocropsml.py                      # all three archives
python scripts/data/fetch_eurocropsml.py --stages extract --files preprocess.zip
```

**`preprocess.zip` is left packed.** Unpacking 706,683 small files onto the network filesystem
runs at a few thousand files per minute, and nothing in the exploratory analysis needs the
arrays: the `.npz` filename `<NUTS3>_<parcelID>_<EC_hcat_c>.npz` already carries the region, the
parcel and the class, and the archive's central directory lists all 706,683 of them. Unpack it
before any work that reads the time series themselves.

**EuroCropsML does ship parcel geometries**, in `raw_data/geometries/`, keyed by the same
`parcel_id` that appears in the `.npz` filenames. They are compared against the EuroCrops
polygons in `results/eda/eurocrops/` and in `notebooks/03_eurocrops_vector_eda.ipynb`.

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
