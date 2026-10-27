# EuroCrops coverage: countries, regions and years

Checked on 26 and 27 October 2026 for the planning of the transfer experiments. Two releases exist, and they differ
in which countries they hold, how many years, and the crop taxonomy.

| | EuroCrops v11 (Zenodo) | EuroCrops v2.0 (JRC) |
|---|---|---|
| Used by the pipeline | **Yes**: `data/eurocrops/`, `configs/class_scheme_eurocropsml.yaml` | No |
| Coverage | 19 countries, 23 regional datasets, **one year each**, 2018 to 2023 | 16 countries, 18 regional datasets, **3 to 16 years each**, 2008 to 2023 |
| Files | 23 zipped vector layers plus per-dataset HCAT mapping CSVs | 153 yearly GeoParquet files plus 17 multi-year stack files |
| Crop taxonomy | HCAT2 and HCAT3 (our class scheme uses HCAT codes at that depth) | HCAT4 |
| Inclusion rule | Whatever each member state released | A region is kept only if it has 2021 and at least three consecutive years |
| Geometry | Per-country CRS, reprojected by our loader | EPSG:3035 in every file, the CRS of our chip grid |
| Reference | [Zenodo record 14094196](https://zenodo.org/records/14094196) (2024); local copy of the record: `data/eurocrops/zenodo_record_14094196.json` | [ESSD 18, 4075 (2026)](https://essd.copernicus.org/articles/18/4075/2026/); DOI 10.2905/b9fb9e67-78a9-4327-9d59-39a928d812d3; [download server](https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/EuroCropsV2/); [code and crop-code mapping](https://github.com/Martincccc/EuroCropsV2) |

## Per region

v11 years and archive sizes are from the Zenodo record. v2 years are from the `gpqtv202/` directory of the JRC server
(listed 26 October 2026; the full listing is in [eurocrops_v2_files.csv](eurocrops_v2_files.csv)). The older index
`all_gpqt_files.csv` on the same server (dated 14 November 2025) lacks `pt_2023`, so it gives 152 files.

| Country | Region | v11 year | v11 archive | v2 years (count) | Status in this repository |
|---|---|---|---|---|---|
| Austria | national | 2021 | 845 MB | 2015 to 2023 (9) | |
| Belgium | Flanders (`be2`) | 2021 | 140 MB | 2008 to 2023 (16) | |
| Belgium | Wallonia (`be3`) | 2021 | 122 MB | 2015 to 2023 (9) | |
| Bulgaria | national | not in v11 | | 2014 to 2023 (10), no stack file | |
| Czechia | national | 2023 | 278 MB | 2021 to 2023 (3) | |
| Germany | Brandenburg (`de4`) | 2023 | 103 MB | 2010 to 2023 (14) | |
| Germany | Lower Saxony | 2021 | 242 MB | not in v2 | |
| Germany | North Rhine-Westphalia (`dea`) | 2021 | 277 MB | 2019 to 2023 (5) | |
| Denmark | national | 2019 | 298 MB | 2010 to 2023 (14) | |
| **Estonia** | national | **2021** | 288 MB | 2015 to 2023 (9) | **v11 vectors; chip sets `EE_2021`, `EE_2021_rev`; all results so far** |
| Spain | national | 2023 | 9,792 MB | 2021 to 2023 (3) | |
| Spain | Navarra | 2020 | 468 MB | not in v2 | |
| Finland | national | 2020 | 369 MB | 2020 to 2023 (4) | |
| France | national | 2018 | 2,645 MB | 2015 to 2023 (9) | |
| Croatia | national | 2020 | 351 MB | not in v2 | |
| Ireland | national | 2023 | 1,092 MB | 2017 to 2023 (7) | |
| Italy | Tuscany (`iti1`) | not in v11 | | 2016 to 2023 (8) | |
| **Latvia** | national | **2021** | 548 MB | **not in v2** | **v11 vectors; chip set `LV_2021`, 11,500 chips with S2 and S1, exported 18 to 23 October 2026** |
| **Lithuania** | national | **2021** | 545 MB | **not in v2** | **v11 vectors only; 22 coarse HCAT classes** |
| Netherlands | national | 2020 | 283 MB | 2009 to 2023 (15) | |
| **Portugal** | national | **2021**, a 100,000-parcel sample | 35 MB | 2017 to 2023 (7) | **v11 vectors only** |
| Romania | national | no year given | 686 MB | not in v2 | |
| Sweden | national | 2021 | 427 MB | not in v2 | |
| Slovenia | national | 2021 | 213 MB | 2019 to 2023 (5) | |
| Slovakia | national | 2021 | 325 MB | 2018 to 2023 (6) | |

The v11 HCAT mapping CSVs also include `be_wal_all_years`, `fr_other_years`, `nl_other_years` and `es_ne_2021`,
which have no vector archive in the record. The ESSD paper mentions the Spanish provinces León (ES413) and Jaén
(ES616); the server holds only national `es` files.

## The v2 file layout

From the server's README:

- **Yearly files** `<nuts>_<year>.parquet`: `cropfield` (id within the table), `original_code` (the member state's
  crop code; synthetic codes of 10,000 and above where the source had none), `area_ha`, `geom` (EPSG:3035). The HCAT4
  name comes from joining `eurocrops.csv` of the EuroCropsV2 repository on `nuts` and `original_code`.
- **Stack files** `<nuts>_stack.parquet`: one row per field tracked across years, with `cf<YEAR>` (the field's id in
  that year's table), `c<YEAR>` (its crop code that year), `area_ha` and `geom`. This is the crop sequence of each
  field, which a temporal design can use directly: the same field in different years, with its rotation.
- Four directories exist, `gpqt/`, `gpqtv2/`, `gpqtv201/` and `gpqtv202/` (the last dated 2 July 2026). What
  changed between them was not established: the server timed out on 27 October 2026.

## What this means for transfer

- **The Baltic pair rests on v11 and on 2021 alone.** Latvia and Lithuania are not in v2. In v2, Estonia's nearest
  neighbours in climate are Finland and then Denmark.
- **v11 compares different seasons across countries**, except among its 2021 sets: AT, BE (both), DE (Lower Saxony
  and North Rhine-Westphalia), EE, LT, LV, PT, SE, SI and SK. Every v2 region includes 2021.
- **v2 makes a temporal axis possible.** Most regions have 5 to 16 consecutive years, so a model can be trained on
  one year and tested on another in the same place, where only the weather changes. Estonia has 2015 to 2023.
- **Portugal in v2 may be complete.** If its 2017 to 2023 files cover the whole country, unlike v11's 100,000-parcel
  sample, the choice of a third country changes. Not checked yet.
- **The taxonomy differs.** Moving to v2 means mapping our HCAT classes to HCAT4, Estonia included, and checking
  that v2's Estonia 2021 matches v11's parcel for parcel.
- **Imagery and products by year are a separate check.** Sentinel-2 L2A from our source, Sentinel-1 RTC,
  AlphaEarth's annual layers and the TESSERA store each start in a different year; which v2 years every arm can
  actually be run on has not been checked.
