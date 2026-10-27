# GAEZ v5 agro-ecological zones, and the alternatives

Checked on 26 October 2026 against the GAEZ v5 model documentation (the GitHub wiki of `un-fao/gaezv5`, commit
`c6bf96c`, 24 February 2026). On 2 October 2026 the user decided that cross-country transfer runs only within one
GAEZ v5 zone; that decision is not yet in `protocol.md` (it lives in
`figures/phase1_experimental_setup_prompt.md`).

## Access

| What | Where |
|---|---|
| Map viewer and catalogue | <https://data.apps.fao.org/gaez/?lang=en> (JavaScript only; its layers could not be read by script) |
| Model documentation | <https://github.com/un-fao/gaezv5/wiki>; the classification is [chapter 10](https://github.com/un-fao/gaezv5/wiki/10.-Agro%E2%80%90ecological-Zones-classification), the input data [chapter 2](https://github.com/un-fao/gaezv5/wiki/02.-GAEZ-input-datasets), the downloads [chapter 21](https://github.com/un-fao/gaezv5/wiki/21.-GAEZ-v5-Platform-User-Guide) |
| 57-class AEZ layer | [FAO catalogue record](https://data.apps.fao.org/catalog/iso/22c4b002-ae80-41f7-9f36-463b546378a2); Theme 1, Land and Water Resources |
| Agro-climatic layers (thermal climate and zone, growing period, temperature sums) | Theme 2, Agro-Climatic Resources; file names in [`_readme_RES02.xlsx`](https://github.com/user-attachments/files/21917513/_readme_RES02.xlsx) |
| All variables of all themes | [`GAEZv5_Themes_Sub-themes_Variables.xlsx`](https://github.com/user-attachments/files/20961388/GAEZv5_Themes_Sub-themes_Variables.xlsx) |
| Download | `gsutil -m cp -r gs://fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET/<VARIABLE CODE>/ <dest>`; GeoTIFF |
| Contact | GAEZ@fao.org |
| Older version | [GAEZ v4](https://www.fao.org/gaez/gaezv4/en), if a v5 layer is missing |

Theme 2 file names follow `GAEZ-V5.<VARIABLE>.<PERIOD>.<CLIMATE_MODEL>.<SCENARIO>`, for example `RES01-ET0` for
reference evapotranspiration. Periods: `HP8100` (1981 to 2000), `HP0120` (2001 to 2020), and the futures `FP2140`,
`FP4160`, `FP6180`, `FP8100`; time-series layers carry the year instead. Historical data has scenario `HIST`.

## Inputs and resolution

- **Climate:** AgERA5 (C3S, daily, 0.1°), interpolated by IIASA onto a **5 arc-minute grid, about 9 km**, with an
  altitude correction for temperature. Year-by-year analysis covers **up to 2020**; the 20-year averages are
  1981 to 2000 and 2001 to 2020. For 2021 to 2023 imagery, year-specific indicators must be computed from AgERA5
  directly. Futures use ISIMIP3b bias-corrected CMIP6.
- **Soil and terrain:** 30 arc-seconds, about 1 km. Chapter 2 names HWSD v2.01 and the ALOS AW3D30 DEM; chapter 10
  still names HWSD v1.2.1 and SRTM for the soil classes. Which one the released AEZ layer used is to be confirmed.
- **Land cover:** 30 arc-seconds, 11 classes calibrated to FAOSTAT 2019 to 2021, from six global products
  (WorldCover, ESRI, GLAD and others); irrigation from GMIA v5.
- Our 2.24 km chips are about one sixteenth of a climate cell; a 4 x 4 chip block (8.96 km) is about one cell.

## The class structure: 57 zones

**Thermal climate (TC1 to TC6)**, latitudinal belts from sea-level monthly temperature: tropics lowland, tropics
highland, subtropics, temperate, boreal, arctic.

**Thermal zone (TZ1 to TZ6)**, actual temperatures, tested in order:

| Zone | Name | Rule |
|---|---|---|
| TZ1 | Warm | Every month at least 10 °C, annual mean at least 20 °C |
| TZ2 | Moderately cool | Every month at least 5 °C, at least 10 °C in 8 or more months |
| TZ3 | Moderate (temperate climate only) | At least 10 °C in 5 or more months, at least 75 days with a daily mean above 20 °C, more than 3,000 degree-days above 10 °C |
| TZ4 | Cool | At least 10 °C in 4 or more months, annual mean at least 0 °C |
| TZ5 | Cold | At least 10 °C in 1 to 3 months, annual mean at least 0 °C |
| TZ6 | Very cold | Every month below 10 °C, or annual mean below 0 °C |

**Temperature regime (TRC1 to TRC10)**, thermal climate crossed with thermal zone. This is the natural macro zone:

| Regime | Name | Built from | Crops it allows |
|---|---|---|---|
| TRC1 | Tropics, lowland | TC1 × TZ1 | |
| TRC2 | Tropics, highland | TC2 × TZ2 or TZ4 | |
| TRC3 | Subtropics, warm | TC3 × TZ1 | |
| TRC4 | Subtropics, moderately cool | TC3 × TZ2 | |
| TRC5 | Subtropics, cool | TC3 × TZ4 | |
| TRC6 | Temperate, moderate | TC4 × TZ3 | Thermophilic crops: maize, soybean, sunflower, as well as wheat, barley, potato, rapeseed |
| TRC7 | Temperate, cool | TC4 × TZ4 | Mostly wheat, barley, potato, rapeseed; frost excludes olive and citrus |
| TRC8 | Boreal or cold, no permafrost | TC2 to TC5 × TZ5 | Pasture and very short spring cereals and potato |
| TRC9 | Boreal or cold, with permafrost | as TRC8 | Becomes the special class AEZ-54 |
| TRC10 | Arctic or very cold | TC2 to TC6 × TZ6 | Becomes the special class AEZ-55 |

**Moisture regime (M1 to M4)**, from the length of growing period LGP*, the natural sub zone:

| Class | Name | LGP* (days) |
|---|---|---|
| M1 | Desert, arid | under 60 (becomes AEZ-53) |
| M2 | Semi-arid, dry | 60 to 180 |
| M3 | Sub-humid, moist | 180 to 270 |
| M4 | Humid, wet | over 270 |

Where winter limits the season (LGPt5 under 330 days, as in all of Europe north of the Mediterranean), LGP* is the
larger of the growing-period days and an "equivalent LGP" from the annual P/ET0 ratio: 60 days at P/ET0 of about
0.15, 120 at 0.40, 180 at 0.65 and 270 at 1.15.

**Soil and terrain (S1 to S5):** S1 very steep (at least 75 % of the cell above 30 % slope); S2 hydromorphic (at
least 67 % Gleysols, Histosols, gleyic or stagnic soils); S3, S4 and S5 by a soil suitability index for grain
maize at low input (S3 0.667 to 1, S4 0.25 to 0.667, S5 below 0.25 and all permafrost).

**Special land cover (L1 to L3):** L1 inland water (at least 75 % of a 30 arc-second cell); L2 built-up (at least
75 %); L3 irrigated (at least 20 % irrigated cropland, or at least 40 % cropland of which half is equipped).

**The 57 classes:**

- **AEZ-01 to AEZ-48:** TRC1 to TRC8, each × M2 to M4 × S3 to S4, six codes per regime (TRC6 is AEZ-31 to 36, TRC7
  AEZ-37 to 42, TRC8 AEZ-43 to 48). The order of the six within a regime is in the documentation's appendix, not in
  the markdown; read it from the layer's legend.
- **Special classes:** AEZ-49 very steep; AEZ-50 severe soil limits; AEZ-51 irrigated; AEZ-52 hydromorphic soils;
  AEZ-53 desert; AEZ-54 permafrost; AEZ-55 arctic; AEZ-56 built-up; AEZ-57 water.
- **Priority:** 56 and 57 first, then 49, 51 and 52, then 53 and 54, then 50, and only then 01 to 48. A special
  class therefore overrides the climate wherever its condition holds.

GAEZ also publishes a **multiple-cropping zone** classification (eight classes, rain-fed MCR and irrigated MCI):
whether one, two or three sequential crops fit in a year. Cool temperate and boreal zones allow one cryophilic crop.

## How it bears on the design (notes of 26 October, not decided)

- **Tag with the component layers, not the 57-class map.** Because of the priority, a cell of mostly peat or
  gleyed soil becomes AEZ-52 whatever its climate. Estonia has much of both, so the final map would erase the climate
  exactly where some fields are. Tag each block with TRC (macro) and moisture class (sub); keep soil and the special
  classes as covariates.
- **Expected for Europe (not yet checked on the layers):** the Baltics in TRC7 (too few days above 20 °C for TRC6);
  TRC7 reaching far into central Europe; the Pannonian basin and the Po valley in TRC6; the Mediterranean in TRC3 to
  TRC5. "Same zone" may therefore span a wide gradient.
- **Measure transfer distance with continuous variables** (temperature sums, thermal season length, LGP or P/ET0),
  from the GAEZ normals or from AgERA5 for the imaged year; the named zones then serve design and reporting.
- **Border effects:** a 9 km climate grid gives blocky borders; cross-zone tests want a buffer around them.
- **Normals, not the season:** 2001 to 2020 averages do not show a dry or late 2021. For a temporal design, the
  year-specific AgERA5 indicators are the axis.

## Alternatives compared

| Scheme | Built for | Based on | In Europe | Fit for this work |
|---|---|---|---|---|
| **GAEZ v5 AEZ** (FAO and IIASA, April 2025) | Crop suitability and production | Thermal regime, growing period, soil and terrain | Splits by temperature and season length; the variables are published layers | The crop rationale; the 2 October choice |
| **Köppen-Geiger** ([Beck et al. 2023](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10593765/), 1 km, 1991 to 2020 and future scenarios) | General climate | Monthly temperature and precipitation thresholds | Coarse: the Baltics and much of central and eastern Europe in one class | Widely understood; a robustness check, not the main axis |
| **Environmental Stratification of Europe** ([Metzger et al. 2005](https://research.wur.nl/en/publications/a-climatic-stratification-of-the-environment-of-europe); [dataset](https://www.research.ed.ac.uk/en/datasets/the-environmental-stratification-of-europe/)) | Stratified sampling and modelling in Europe | 20 climate and terrain variables clustered into 84 strata and 13 zones, 1 km | Separates the Baltic from central Europe; an [agri-environmental version](https://test01.ku.dk/:obvius/pureproxy/3974268/en/publications/a-biophysical-typology-in-agri-environmental-modelling) adds soil and farming suitability | Built for stratified sampling, distributed in ETRS89-LAEA (our EPSG:3035); Europe only |
| **EEA biogeographical regions** ([EEA](https://www.eea.europa.eu/data-and-maps/data/biogeographical-regions-europe-3)) | Habitats Directive reporting | Expert delineation | About 11 regions; EE, LV and LT all Boreal | Known in EU policy; coarse, not crop-based |
| **UNESCO map of arid regions** ([1979](https://unesdoc.unesco.org/ark:/48223/pf0000032661)) | Desertification | Aridity index P/PET | Most of northern and central Europe is the unclassified humid remainder | Not usable as the main axis; for aridity a current continuous index (CGIAR Global Aridity Index) fits better |

Suggested use from that comparison: GAEZ for the named zones, a second zoning (Köppen-Geiger or the European
stratification) tagged alongside as a robustness check, and continuous climate variables for the transfer distance.

Sources not reachable by script on 26 October 2026: the GAEZ portal (JavaScript) and the FAO v5 documentation PDF
(HTTP 403); the wiki replaced both.
