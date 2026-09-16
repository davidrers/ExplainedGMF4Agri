"""End-to-end CropHarvest exploratory analysis.

Mirrors results/eda/_run_analysis.py (EuroCropsML) for the thesis's secondary dataset.

Outputs, all written next to this file:
  catalogue_labels.parquet   one row per CropHarvest label + country/region/geometry metadata
  class_inventory.parquet    crop class x region counts (full dataset)
  phenology.parquet          sampled per-instance band statistics read from the .h5 arrays
  findings.json              every number the notebook quotes in a markdown cell
  run_config.json            the configuration this run used (seed, paths, sample sizes)
  figures/NN_*.png           all charts

Plus, at the repository level:
  configs/class_scheme_cropharvest.yaml   the proposed task definition per region

Run after labels.geojson and features.tar.gz have been downloaded and extracted under
<repo>/data/cropharvest (see data/README.md).

    python results/eda/cropharvest/_run_analysis.py
    python results/eda/cropharvest/_run_analysis.py --array-sample 2000   # quick pass
"""

from __future__ import annotations

import argparse
import json
import os
import random
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

import geopandas as gpd
import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.spatial import cKDTree

sns.set_theme(context="notebook", style="whitegrid")

SEED = 42
RNG = np.random.default_rng(SEED)
random.seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------------- paths
HERE = Path(__file__).parent.resolve()
REPO = HERE.parents[2]
DATA_ROOT = Path(os.environ.get("CROPHARVEST_DATA", REPO / "data" / "cropharvest")).expanduser()
LABELS_PATH = DATA_ROOT / "labels.geojson"
ARRAYS_DIR = DATA_ROOT / "features" / "arrays"
NE_PATH = DATA_ROOT / "aux" / "ne_10m_admin_0_countries.geojson"

FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_CATALOGUE = HERE / "catalogue_labels.parquet"
OUT_INVENTORY = HERE / "class_inventory.parquet"
OUT_PHENOLOGY = HERE / "phenology.parquet"
OUT_FINDINGS = HERE / "findings.json"
OUT_RUNCFG = HERE / "run_config.json"
OFFSET_SCAN = HERE / "offset_scan.parquet"
OUT_SCHEME = REPO / "configs" / "class_scheme_cropharvest.yaml"

# ---------------------------------------------------------------- constants
# Verified empirically against the .h5 arrays; matches cropharvest/bands.py.
# BANDS = [S1, S2 minus B1 and B10] + [ERA5] + [SRTM static] + [NDVI]
BANDS = [
    "VV", "VH",
    "B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12",
    "temperature_2m", "total_precipitation",
    "elevation", "slope",
    "NDVI",
]
BAND_GROUP = (
    ["Sentinel-1"] * 2 + ["Sentinel-2"] * 11 + ["ERA5"] * 2 + ["SRTM (static)"] * 2 + ["derived"]
)
NDVI_IDX = BANDS.index("NDVI")
N_TIMESTEPS = 12
DAYS_PER_TIMESTEP = 30

# AlphaEarth GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL, verified from the Earth Engine catalogue:
# availability 2017-01-01T00:00:00Z to 2025-01-01T00:00:00Z, one image per calendar year.
ALPHAEARTH_YEARS = list(range(2017, 2025))
# TESSERA, verified from github.com/ucam-eo/tessera: global for 2024; 2017-2025 for
# "regions such as the United States and Europe"; elsewhere by request.
TESSERA_GLOBAL_YEARS = [2024]
TESSERA_REGIONAL_YEARS = list(range(2017, 2026))
TESSERA_REGIONAL_REGIONS = {"Europe", "North America"}

# UN M49-consistent grouping of Natural Earth SUBREGION into the regions the thesis uses.
SUBREGION_TO_REGION = {
    "Eastern Africa": "Sub-Saharan Africa",
    "Middle Africa": "Sub-Saharan Africa",
    "Western Africa": "Sub-Saharan Africa",
    "Southern Africa": "Sub-Saharan Africa",
    "South America": "South America",
    "Central America": "Central America & Caribbean",
    "Caribbean": "Central America & Caribbean",
    "Central Asia": "Central Asia",
    "Northern Europe": "Europe",
    "Southern Europe": "Europe",
    "Eastern Europe": "Europe",
    "Western Europe": "Europe",
    "Northern America": "North America",
    "Eastern Asia": "East & Southeast Asia",
    "South-Eastern Asia": "East & Southeast Asia",
    "Southern Asia": "South Asia",
    "Northern Africa": "MENA",
    "Western Asia": "MENA",
    "Australia and New Zealand": "Oceania",
    "Melanesia": "Oceania",
    "Micronesia": "Oceania",
    "Polynesia": "Oceania",
}

# Natural Earth admin-0 merges the French overseas departments into metropolitan France.
# UN M49 places Reunion in Eastern Africa and Martinique in the Caribbean, so we override
# by source dataset, which is unambiguous.
DATASET_GEO_OVERRIDE = {
    "reunion-france": ("Reunion (FR)", "Sub-Saharan Africa"),
    "martinique-france": ("Martinique (FR)", "Central America & Caribbean"),
}

# Source datasets that carry no crop-type label at all (binary crop / non-crop only).
BINARY_ONLY_DATASETS = [
    "geowiki-landcover-2017", "china-crop", "rwanda-ceo", "kenya-non-crop",
    "tanzania-ceo", "togo", "ethiopia", "sudan", "togo-eval", "mali-non-crop",
    "brazil-non-crop",
]

# Curated normalisation of the raw free-text `label` field. Covers every label with
# n >= 90 in the full dataset plus the obvious synonyms below that. Two corrections are
# folded in: French RPG land-tenure suffixes ("fermage" = tenanted, "propriete ou faire
# valoir direct" = owner-occupied) describe tenure, not crop, so they are merged; and
# untranslated French crop names are mapped onto their English equivalents.
LABEL_NORM = {
    # synonyms in English sources
    "corn": "maize", "sweet corn": "maize",
    "soybeans": "soybean", "soja": "soybean",
    "potatoes": "potato", "pomme de terre de consommation": "potato",
    "pomme de terre feculiere": "potato", "pomme de terre fculire": "potato",
    "barley (undiff)": "barley", "spring barley": "barley", "winter barley": "barley",
    "rye (undiff)": "rye", "spring wheat": "wheat", "winter wheat": "wheat",
    "wheat (undiff)": "wheat", "oats (undiff)": "oats",
    "sunflowers": "sunflower", "tournesol": "sunflower",
    "beans (undiff)": "bean", "dry bean": "bean", "bush bean": "bean",
    "tomatoes": "tomato", "onions": "onion",
    "sugarbeets": "sugarbeet", "betterave non fourragere / bette": "sugarbeet",
    "betterave non fourragre / bette": "sugarbeet",
    # French RPG (Ile-de-France, Reunion, Martinique)
    "mais": "maize", "mas": "maize", "mais ensilage": "maize", "mas ensilage": "maize",
    "sorgho": "sorghum", "sarrasin": "buckwheat",
    "avoine dhiver": "oats", "avoine de printemps": "oats",
    "triticale dhiver": "triticale", "triticale de printemps": "triticale",
    "canne a sucre - fermage": "sugarcane",
    "canne  sucre - fermage": "sugarcane",
    "canne a sucre - propriete ou faire valoir direct": "sugarcane",
    "canne  sucre - proprit ou faire valoir direct": "sugarcane",
    "canne a sucre - autre": "sugarcane", "canne  sucre - autre": "sugarcane",
    "banane creole (fruit et legume) - fermage": "banana",
    "banane crole (fruit et lgume) - fermage": "banana",
    "banane creole (fruit et legume) - propriete ou faire valoir direct": "banana",
    "banane crole (fruit et lgume) - proprit ou faire valoir direct": "banana",
    "banane creole (fruit et legume) - autre": "banana",
    "banane crole (fruit et lgume) - autre": "banana",
    "banane export - fermage": "banana",
    "banane export - propriete ou faire valoir direct": "banana",
    "banane export - proprit ou faire valoir direct": "banana",
    "banane export - autre": "banana",
    "agrume": "citrus", "verger": "orchard", "verger (dom)": "orchard",
    "ananas": "pineapple", "fraise": "strawberry",
    "cafe / cacao": "coffee/cocoa", "caf / cacao": "coffee/cocoa",
    "vanille sous bois": "vanilla",
    "lentille cultivee (non fourragere)": "lentil",
    "lentille cultive (non fourragre)": "lentil",
    "pois de printemps seme avant le 31/05": "pea",
    "pois de printemps sem avant le 31/05": "pea",
    "orange tree": "citrus", "apple tree": "apple",
}

# Labels that are not crop types even though they may sit inside a crop-type source.
NON_CROP_TYPE_LABELS = {
    "meadows", "pasture", "unimproved pasture", "fallow", "young fallow", "cerrado",
    "uncultivated soil", "shrubland", "urban", "water", "barren", "abandoned (overgrown)",
    "abandoned (shrubs)", "too wet to seed", "weakly vegetated agricultural",
    "annual crop", "fruit crop", "vegetables (undiff)", "grasses and other fodder crop",
    "pine", "eucalyptus", "bois pature", "horticulture ornementale sous abri",
    "horticulture ornementale de plein champ", "culture sous serre hors sol",
    "forage crops", "oil seeds", "sod", "ryegrass", "timothy", "birdsfoot trefoil",
    "orchard", "crop", "non-crop", "noncrop",
}

# FAO ICC group assignments in the released labels.geojson that are demonstrably wrong;
# used only to report the defect rate, not to silently rewrite the data.
FAO_DEFECTS = {
    ("pine", "fruits_nuts"): "not a crop (forestry)",
    ("eucalyptus", "other"): "not a crop (forestry)",
    ("sugarbeet", "vegetables_melons"): "inconsistent with sugarcane -> sugar",
    ("potato", "vegetables_melons"): "inconsistent with potato -> root_tuber",
    ("cotton", "other"): "fibre crop absent from ICC11 core, silently pooled into 'other'",
}

# Class-aggregation levels used for the cross-region analysis.
CROP_TO_AGGREGATE = {
    "maize": "cereals", "rice": "cereals", "wheat": "cereals", "millet": "cereals",
    "sorghum": "cereals", "barley": "cereals", "oats": "cereals", "rye": "cereals",
    "buckwheat": "cereals", "triticale": "cereals",
    "soybean": "oilseeds", "groundnut": "oilseeds", "sunflower": "oilseeds",
    "rapeseed": "oilseeds", "sesame": "oilseeds",
    "cowpea": "legumes", "bean": "legumes", "pea": "legumes", "lentil": "legumes",
    "chickpea": "legumes", "pigeon pea": "legumes",
    "potato": "roots_tubers", "cassava": "roots_tubers", "sweet potato": "roots_tubers",
    "yam": "roots_tubers", "taro": "roots_tubers",
    "sugarcane": "sugar", "sugarbeet": "sugar",
    "cotton": "fibre",
    "banana": "perennial_fruit", "citrus": "perennial_fruit", "apple": "perennial_fruit",
    "pineapple": "perennial_fruit", "strawberry": "perennial_fruit",
    "mango": "perennial_fruit", "avocado": "perennial_fruit", "grape": "perennial_fruit",
    "coffee/cocoa": "beverage_spice", "coffee": "beverage_spice", "cocoa": "beverage_spice",
    "tea": "beverage_spice", "vanilla": "beverage_spice", "tobacco": "beverage_spice",
    "carrot": "vegetables", "eggplant": "vegetables", "cabbage": "vegetables",
    "cucumber": "vegetables", "tomato": "vegetables", "onion": "vegetables",
    "asparagus": "vegetables", "melon": "vegetables", "lettuce": "vegetables",
    "alfalfa": "fodder",
}


# ---------------------------------------------------------------- helpers
def _ascii(s):
    if not isinstance(s, str):
        return s
    return (
        s.replace("è", "e").replace("é", "e").replace("ê", "e")
        .replace("à", "a").replace("â", "a").replace("ç", "c")
        .replace("î", "i").replace("ï", "i").replace("ô", "o")
        .replace("ù", "u").replace("û", "u").replace("æ", "ae")
    )


def normalise_label(s):
    if not isinstance(s, str):
        return None
    key = _ascii(s.strip().lower())
    return LABEL_NORM.get(key, key)


def save(fig, name: str) -> None:
    fig.savefig(FIG_DIR / name, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")


# ---------------------------------------------------------------- 1. catalogue
def build_catalogue() -> gpd.GeoDataFrame:
    print(f"reading {LABELS_PATH} ...")
    g = gpd.read_file(LABELS_PATH)
    print(f"  {len(g):,} labels, CRS {g.crs}")

    g["gtype"] = g.geometry.geom_type.fillna("Missing")
    g["is_polygon"] = g["gtype"].eq("Polygon")

    # polygon area on an equal-area projection
    area_ha = pd.Series(np.nan, index=g.index)
    poly = g[g.is_polygon & g.geometry.notna()]
    if len(poly):
        area_ha.loc[poly.index] = poly.to_crs("EPSG:6933").geometry.area.values / 1e4
    g["area_ha"] = area_ha

    # country / region by point-in-polygon on Natural Earth 10m admin-0
    ne = gpd.read_file(NE_PATH)[["NAME_EN", "CONTINENT", "SUBREGION", "ISO_A3", "geometry"]]
    pts = gpd.GeoDataFrame(geometry=gpd.points_from_xy(g.lon, g.lat), crs="EPSG:4326")
    j = gpd.sjoin(pts, ne, how="left", predicate="within")
    j = j[~j.index.duplicated()].reindex(g.index)
    unmatched = j["NAME_EN"].isna()
    n_unmatched = int(unmatched.sum())
    if n_unmatched:  # coastal / island points: fall back to the nearest country polygon
        near = gpd.sjoin_nearest(
            pts.loc[unmatched].to_crs("EPSG:6933"), ne.to_crs("EPSG:6933"), how="left"
        )
        near = near[~near.index.duplicated()]
        for col in ["NAME_EN", "CONTINENT", "SUBREGION", "ISO_A3"]:
            j.loc[unmatched, col] = near[col]
    g["country"] = j["NAME_EN"].values
    g["iso_a3"] = j["ISO_A3"].values
    g["subregion"] = j["SUBREGION"].values
    g["region"] = g["subregion"].map(SUBREGION_TO_REGION).fillna("Other")

    for ds, (country, region) in DATASET_GEO_OVERRIDE.items():
        m = g["dataset"].eq(ds)
        g.loc[m, "country"] = country
        g.loc[m, "region"] = region

    # feature-array availability: features/arrays/{index}_{dataset}.h5
    have = set()
    for p in ARRAYS_DIR.glob("*.h5"):
        idx, _, ds = p.stem.partition("_")
        have.add((int(idx), ds))
    g["has_array"] = [(int(i), d) in have for i, d in zip(g["index"], g["dataset"])]
    g["array_path"] = [
        str(ARRAYS_DIR / f"{i}_{d}.h5") if h else None
        for i, d, h in zip(g["index"], g["dataset"], g["has_array"])
    ]

    # label-to-pixel offset, from the exhaustive scan in _offset_scan.py. The
    # CropHarvest exporter snaps a label lying outside its shared GeoTIFF to that tif's
    # edge instead of rejecting it, so some instances carry the spectral signature of a
    # different place. Anything beyond one pixel diagonal (14.14 m) is unusable.
    if OFFSET_SCAN.exists():
        off = pd.read_parquet(OFFSET_SCAN)
        m = off.set_index(["index", "dataset"])
        key = pd.MultiIndex.from_arrays([g["index"], g["dataset"]])
        g["pixel_offset_m"] = m["pixel_offset_m"].reindex(key).to_numpy()
        g["offset_ok"] = m["offset_ok"].reindex(key).fillna(False).to_numpy()
    else:
        print("  WARNING: offset_scan.parquet absent; run _offset_scan.py first")
        g["pixel_offset_m"] = np.nan
        g["offset_ok"] = g["has_array"]
    # a label is only usable if its array exists AND samples the right pixel
    g["array_usable"] = g["has_array"] & g["offset_ok"]

    # label normalisation and aggregation levels
    g["fao_group_raw"] = g["classification_label"]
    g["crop"] = g["label"].map(normalise_label)
    # A label is treated as a usable crop type only when it names a specific crop.
    # Three filters, all of them auditable:
    #   1. it must survive the curated non-crop-type list above;
    #   2. CropHarvest's own FAO group must not be non_crop (this removes the French
    #      RPG grassland, meadow and set-aside categories without hand-listing them);
    #   3. it must not be a French "autre ..." catch-all ("other vegetable or annual
    #      fruit" and similar), which names a residual group rather than a crop.
    g["is_crop_type"] = (
        g["crop"].notna()
        & ~g["crop"].isin(NON_CROP_TYPE_LABELS)
        & ~g["fao_group_raw"].eq("non_crop")
        & ~g["crop"].str.startswith("autre ", na=False)
    )
    F_EXCLUDED = {
        "non_crop_fao_group": int((g["crop"].notna() & g["fao_group_raw"].eq("non_crop")).sum()),
        "french_autre_catchall": int(
            g["crop"].str.startswith("autre ", na=False).sum()),
        "curated_non_crop_type_list": int(g["crop"].isin(NON_CROP_TYPE_LABELS).sum()),
    }
    g.attrs["excluded_label_counts"] = F_EXCLUDED
    g.loc[~g["is_crop_type"], "crop"] = None
    g["fao_group"] = g["classification_label"]
    g["agg_group"] = g["crop"].map(CROP_TO_AGGREGATE)

    # the 12 x 30-day observation window ends on export_end_date (always 1 February)
    g["window_end"] = g["export_end_date"]
    g["window_start"] = g["export_end_date"] - timedelta(days=N_TIMESTEPS * DAYS_PER_TIMESTEP)
    g["window_end_year"] = g["window_end"].dt.year
    # the calendar year the window mostly falls in (11 of 12 months)
    g["dominant_year"] = g["window_start"].dt.year

    # AlphaEarth / TESSERA sampling feasibility
    g["alphaearth_ok"] = g["dominant_year"].isin(ALPHAEARTH_YEARS)
    g["tessera_ok"] = g["dominant_year"].isin(TESSERA_GLOBAL_YEARS) | (
        g["region"].isin(TESSERA_REGIONAL_REGIONS)
        & g["dominant_year"].isin(TESSERA_REGIONAL_YEARS)
    )

    cols = [c for c in g.columns if c != "geometry"]
    g[cols].to_parquet(OUT_CATALOGUE, index=False)
    print(f"  wrote {OUT_CATALOGUE.name}  ({len(g):,} rows)")
    return g


# ---------------------------------------------------------------- 2. figures
def fig_regions(g, F):
    rc = g["region"].value_counts()
    F["labels_per_region"] = rc.to_dict()
    F["labels_per_country_top20"] = g["country"].value_counts().head(20).to_dict()
    F["n_countries"] = int(g["country"].nunique())

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    rc.sort_values().plot(kind="barh", color="steelblue", ax=axes[0])
    axes[0].set_xlabel("labels")
    axes[0].set_title(f"CropHarvest labels per region (n={len(g):,})")
    g["country"].value_counts().head(20).sort_values().plot(
        kind="barh", color="seagreen", ax=axes[1]
    )
    axes[1].set_xlabel("labels")
    axes[1].set_title(f"Top-20 countries of {g['country'].nunique()}")
    save(fig, "01_labels_per_region.png")


def fig_sources(g, F):
    t = (
        g.groupby("dataset")
        .agg(
            n=("index", "size"),
            polygons=("is_polygon", "sum"),
            with_croptype=("is_crop_type", "sum"),
            with_array=("has_array", "sum"),
        )
        .sort_values("n", ascending=False)
    )
    t["pct_croptype"] = (t.with_croptype / t.n * 100).round(1)
    t["pct_array"] = (t.with_array / t.n * 100).round(1)
    F["source_datasets"] = t.reset_index().to_dict("records")
    F["n_source_datasets"] = int(len(t))
    F["binary_only_datasets"] = BINARY_ONLY_DATASETS
    F["n_labels_binary_only"] = int(g["dataset"].isin(BINARY_ONLY_DATASETS).sum())

    fig, ax = plt.subplots(figsize=(9, 7))
    d = t.sort_values("n")
    ax.barh(d.index, d.n, color="lightgrey", label="points, no crop type")
    ax.barh(d.index, d.polygons, color="steelblue", label="polygon geometry")
    ax.barh(d.index, d.with_croptype, color="darkorange", height=0.45,
            label="carries a crop-type label")
    ax.set_xscale("log")
    ax.set_xlabel("labels (log scale)")
    ax.set_title("Source datasets: size, polygon share and crop-type coverage")
    ax.legend(loc="lower right")
    save(fig, "02_source_datasets.png")


def fig_geometry(g, F):
    F["n_points"] = int((g.gtype == "Point").sum())
    F["n_polygons"] = int((g.gtype == "Polygon").sum())
    F["n_missing_geom"] = int((g.gtype == "Missing").sum())
    F["polygon_share_pct"] = round(F["n_polygons"] / len(g) * 100, 1)
    F["polygons_with_array"] = int(g[g.is_polygon].has_array.sum())
    F["polygons_with_croptype"] = int(g[g.is_polygon].is_crop_type.sum())
    F["polygons_with_array_and_croptype"] = int(
        (g.is_polygon & g.has_array & g.is_crop_type).sum()
    )
    F["polygons_usable"] = int((g.is_polygon & g.array_usable & g.is_crop_type).sum())
    F["polygons_lost_to_offset"] = int(
        (g.is_polygon & g.has_array & g.is_crop_type & ~g.offset_ok).sum())
    F["polygon_datasets"] = (
        g[g.is_polygon].dataset.value_counts().to_dict()
    )
    F["polygons_per_region"] = g[g.is_polygon].region.value_counts().to_dict()
    F["polygons_per_country_top15"] = (
        g[g.is_polygon].country.value_counts().head(15).to_dict()
    )

    ct = pd.crosstab(g.region, g.gtype)
    for c in ["Point", "Polygon"]:
        if c not in ct:
            ct[c] = 0
    ct = ct[["Point", "Polygon"]].sort_values("Polygon")
    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    ct.plot(kind="barh", stacked=True, color=["lightgrey", "steelblue"], ax=ax)
    ax.set_xlabel("labels")
    ax.set_title(
        f"Geometry type by region: {F['n_polygons']:,} polygons "
        f"({F['polygon_share_pct']} %) vs {F['n_points']:,} points"
    )
    save(fig, "03_geometry_types.png")


def fig_polygon_area(g, F):
    poly = g[g.is_polygon & g.area_ha.notna()]
    q = poly.area_ha.describe(percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    F["polygon_area_ha"] = {k: round(float(v), 4) for k, v in q.items()}
    F["polygon_area_by_dataset"] = (
        poly.groupby("dataset")
        .area_ha.agg(["count", "median", "mean", "max"])
        .round(3)
        .reset_index()
        .to_dict("records")
    )
    F["polygons_below_0p1ha"] = int((poly.area_ha < 0.1).sum())
    F["polygons_below_1_s2_pixel"] = int((poly.area_ha < 0.01).sum())
    F["polygons_below_10_s2_pixels"] = int((poly.area_ha < 0.1).sum())
    F["polygon_median_area_ha"] = round(float(poly.area_ha.median()), 3)
    F["polygons_smallholder_under_2ha_pct"] = round(
        float((poly.area_ha < 2).mean() * 100), 1
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    axes[0].hist(np.log10(poly.area_ha.clip(lower=1e-4)), bins=60, color="steelblue")
    for x, lab in [(-2, "1 S2 pixel\n(0.01 ha)"), (np.log10(2), "2 ha\nsmallholder")]:
        axes[0].axvline(x, color="crimson", ls="--", lw=1)
        axes[0].text(x, axes[0].get_ylim()[1] * 0.85, lab, fontsize=8, color="crimson")
    axes[0].set_xlabel("log10 polygon area (ha)")
    axes[0].set_ylabel("polygons")
    axes[0].set_title(f"Polygon area, all {len(poly):,} polygons")

    order = poly.groupby("dataset").area_ha.median().sort_values().index
    sns.boxplot(data=poly, y="dataset", x="area_ha", order=order, ax=axes[1],
                color="steelblue", fliersize=1)
    axes[1].set_xscale("log")
    axes[1].axvline(2, color="crimson", ls="--", lw=1)
    axes[1].set_xlabel("polygon area (ha, log scale)")
    axes[1].set_ylabel("")
    axes[1].set_title("Field size by source dataset (red line: 2 ha)")
    save(fig, "04_polygon_area.png")


def fig_croptype(g, F):
    ct = g[g.is_crop_type]
    F["n_with_croptype"] = int(len(ct))
    F["n_distinct_crops"] = int(ct.crop.nunique())
    F["n_raw_labels"] = int(g.label.nunique())
    F["n_binary_only_labels"] = int(len(g) - len(ct))
    inv = (
        ct.groupby("crop")
        .agg(n=("index", "size"), n_datasets=("dataset", "nunique"),
             n_countries=("country", "nunique"), n_regions=("region", "nunique"),
             n_polygons=("is_polygon", "sum"))
        .sort_values("n", ascending=False)
    )
    inv["share_pct"] = (inv.n / len(ct) * 100).round(2)
    F["crop_inventory_top30"] = inv.head(30).reset_index().to_dict("records")
    top_countries = ct.country.value_counts().head(12).index.tolist()
    F["crop_by_country_top12"] = (
        pd.crosstab(ct.crop, ct.country)[top_countries]
        .loc[inv.head(20).index].to_dict()
    )
    F["croptype_labels_per_country_top20"] = ct.country.value_counts().head(20).to_dict()
    F["croptype_labels_per_region"] = ct.region.value_counts().to_dict()

    fig, ax = plt.subplots(figsize=(9, 8))
    d = inv.head(35).sort_values("n")
    ax.barh(d.index, d.n, color="lightgrey", label="all labels")
    ax.barh(d.index, d.n_polygons, color="steelblue", label="polygon labels")
    for y, (nd, n) in enumerate(zip(d.n_datasets, d.n)):
        ax.text(n * 1.03, y, f"{nd}", va="center", fontsize=7, color="dimgrey")
    ax.set_xscale("log")
    ax.set_xlabel("labels (log scale); grey number = contributing source datasets")
    ax.set_title(
        f"Top-35 harmonised crop types of {inv.shape[0]} "
        f"({len(ct):,} of {len(g):,} labels carry a crop type)"
    )
    ax.legend(loc="lower right")
    save(fig, "05_croptype_inventory.png")
    return inv


def fig_fao_by_region(g, F):
    ct = g[g.fao_group.notna()]
    F["n_with_fao_group"] = int(len(ct))
    F["fao_group_counts"] = ct.fao_group.value_counts().to_dict()
    piv = pd.crosstab(ct.fao_group, ct.region)
    F["fao_group_by_region"] = piv.to_dict()
    # defects
    defects = []
    raw = g["label"].map(normalise_label)
    for (crop, grp), why in FAO_DEFECTS.items():
        n = int(((raw == crop) & (g.fao_group == grp)).sum())
        if n:
            defects.append({"crop": crop, "assigned_group": grp, "n": n, "problem": why})
    F["fao_group_defects"] = defects
    F["n_labels_in_flagged_fao_assignments"] = int(sum(d["n"] for d in defects))

    fig, ax = plt.subplots(figsize=(11, 5))
    sns.heatmap(piv, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                cbar_kws={"label": "labels"}, linewidths=0.4)
    ax.set_title("FAO indicative crop classification group x region (full dataset)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    save(fig, "06_fao_group_by_region.png")


def fig_years(g, F):
    F["export_end_month_day"] = g.export_end_date.dt.strftime("%m-%d").value_counts().to_dict()
    F["window_end_year_counts"] = g.window_end_year.value_counts().sort_index().to_dict()
    F["dominant_year_counts"] = g.dominant_year.value_counts().sort_index().to_dict()
    F["collection_year_counts"] = (
        g.collection_date.dt.year.value_counts().sort_index().to_dict()
    )
    F["window_span"] = [
        str(g.window_start.min().date()), str(g.window_end.max().date())
    ]
    F["n_alphaearth_ok"] = int(g.alphaearth_ok.sum())
    F["n_alphaearth_out_of_range"] = int((~g.alphaearth_ok).sum())
    F["alphaearth_out_of_range_years"] = (
        g.loc[~g.alphaearth_ok, "dominant_year"].value_counts().sort_index().to_dict()
    )
    F["n_tessera_ok"] = int(g.tessera_ok.sum())
    F["n_tessera_out_of_coverage"] = int((~g.tessera_ok).sum())
    # the same, restricted to the usable crop-type polygon subset
    sub = g[g.is_polygon & g.array_usable & g.is_crop_type]
    F["polygon_croptype_n"] = int(len(sub))
    F["polygon_croptype_alphaearth_ok"] = int(sub.alphaearth_ok.sum())
    F["polygon_croptype_tessera_ok"] = int(sub.tessera_ok.sum())
    F["polygon_croptype_dominant_years"] = (
        sub.dominant_year.value_counts().sort_index().to_dict()
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    yc = g.dominant_year.value_counts().sort_index()
    colours = ["crimson" if y not in ALPHAEARTH_YEARS else "steelblue" for y in yc.index]
    axes[0].bar(yc.index.astype(str), yc.values, color=colours)
    axes[0].set_ylabel("labels")
    axes[0].set_title(
        "Observation-window year (11 of 12 months)\n"
        "red = outside AlphaEarth 2017-2024 coverage"
    )
    sc = sub.dominant_year.value_counts().sort_index()
    colours = ["crimson" if y not in ALPHAEARTH_YEARS else "steelblue" for y in sc.index]
    axes[1].bar(sc.index.astype(str), sc.values, color=colours)
    axes[1].set_ylabel("labels")
    axes[1].set_title(
        f"Same, polygon + crop-type + array subset (n={len(sub):,})"
    )
    save(fig, "07_label_years_coverage.png")


def fig_map(g, F):
    ne = gpd.read_file(NE_PATH)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    ne.boundary.plot(ax=ax, color="lightgrey", linewidth=0.3)
    pts = g[~g.is_polygon]
    poly = g[g.is_polygon]
    ax.scatter(pts.lon, pts.lat, s=0.7, c="lightsteelblue", alpha=0.4,
               label=f"point labels (n={len(pts):,})")
    ax.scatter(poly.lon, poly.lat, s=1.6, c="crimson", alpha=0.7,
               label=f"polygon labels (n={len(poly):,})")
    ax.set_xlim(-180, 180)
    ax.set_ylim(-60, 80)
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.set_title("CropHarvest label locations, full dataset")
    ax.legend(loc="lower left", markerscale=8)
    save(fig, "08_label_map.png")


def fig_class_survival(g, F):
    """How many classes survive a minimum-samples-per-class-per-region threshold."""
    thresholds = [20, 50, 100, 150, 200, 300, 500, 1000]
    sub = g[g.is_crop_type & g.array_usable]
    subp = sub[sub.is_polygon]
    rows = []
    for name, d in [("all geometries", sub), ("polygons only", subp)]:
        cnt = d.groupby(["region", "crop"]).size()
        for t in thresholds:
            surv = cnt[cnt >= t]
            rows.append(dict(subset=name, threshold=t,
                             n_region_class_pairs=int(len(surv)),
                             n_distinct_crops=int(surv.index.get_level_values(1).nunique()),
                             n_regions_with_2plus=int(
                                 (surv.groupby(level=0).size() >= 2).sum()),
                             n_regions_with_4plus=int(
                                 (surv.groupby(level=0).size() >= 4).sum()),
                             n_labels=int(surv.sum())))
    surv_df = pd.DataFrame(rows)
    F["class_survival"] = surv_df.to_dict("records")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    for name, mk in [("all geometries", "o"), ("polygons only", "s")]:
        d = surv_df[surv_df.subset == name]
        axes[0].plot(d.threshold, d.n_region_class_pairs, marker=mk, label=name)
        axes[1].plot(d.threshold, d.n_regions_with_4plus, marker=mk, label=name)
    axes[0].axvline(150, color="crimson", ls="--", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("minimum labels per (region, crop)")
    axes[0].set_ylabel("surviving (region, crop) pairs")
    axes[0].set_title("Class survival (red line: recommended floor of 150)")
    axes[0].legend()
    axes[1].axvline(150, color="crimson", ls="--", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("minimum labels per (region, crop)")
    axes[1].set_ylabel("regions with >= 4 surviving crops")
    axes[1].set_title("Regions able to support a multi-class task")
    axes[1].legend()
    save(fig, "09_class_survival.png")
    return surv_df


def fig_intersection(g, F, floor=150):
    """The cross-region shared-class analysis: the direct analogue of the
    EuroCropsML cross-country intersection."""
    sub = g[g.is_crop_type & g.array_usable]
    cnt = sub.groupby(["region", "crop"]).size().unstack(fill_value=0)
    keep_cols = (cnt >= floor).any(axis=0)
    cnt = cnt.loc[:, keep_cols]
    cnt = cnt.loc[cnt.sum(axis=1).sort_values(ascending=False).index]
    F["region_crop_matrix"] = cnt.to_dict()

    surv = cnt >= floor
    F["crops_meeting_floor_per_region"] = {
        r: sorted(cnt.columns[surv.loc[r]].tolist()) for r in cnt.index
    }
    shared = surv.sum(axis=0)
    F["crop_n_regions_meeting_floor"] = shared[shared > 0].sort_values(
        ascending=False).to_dict()
    F["cross_region_shared_crops"] = sorted(shared[shared >= 2].index.tolist())
    F["class_floor"] = floor

    # the same at aggregate level
    agg = sub[sub.agg_group.notna()]
    acnt = agg.groupby(["region", "agg_group"]).size().unstack(fill_value=0)
    asurv = acnt >= floor
    F["agg_group_matrix"] = acnt.to_dict()
    F["agg_groups_meeting_floor_per_region"] = {
        r: sorted(acnt.columns[asurv.loc[r]].tolist()) for r in acnt.index
    }
    ashared = asurv.sum(axis=0)
    F["cross_region_shared_agg_groups"] = sorted(ashared[ashared >= 2].index.tolist())
    F["agg_group_n_regions"] = ashared[ashared > 0].sort_values(ascending=False).to_dict()

    # pairwise region intersections: the direct analogue of the EuroCropsML
    # cross-country intersection, and the only admissible transfer label spaces
    regions = [r for r in surv.index if surv.loc[r].sum() > 0]
    pair_rows = []
    for i, a in enumerate(regions):
        for b in regions[i + 1:]:
            inter = sorted(cnt.columns[surv.loc[a] & surv.loc[b]].tolist())
            pair_rows.append(dict(
                region_a=a, region_b=b, n_shared=len(inter), shared=inter,
                n_labels_a=int(cnt.loc[a, inter].sum()) if inter else 0,
                n_labels_b=int(cnt.loc[b, inter].sum()) if inter else 0))
    pair_rows.sort(key=lambda d: -d["n_shared"])
    F["region_pair_intersections"] = pair_rows
    F["n_admissible_transfer_pairs"] = int(sum(1 for d in pair_rows if d["n_shared"] >= 2))
    F["best_transfer_pair"] = pair_rows[0] if pair_rows else None

    pm = pd.DataFrame(0, index=regions, columns=regions)
    for d in pair_rows:
        pm.loc[d["region_a"], d["region_b"]] = d["n_shared"]
        pm.loc[d["region_b"], d["region_a"]] = d["n_shared"]
    for r in regions:
        pm.loc[r, r] = int(surv.loc[r].sum())
    fig, ax = plt.subplots(figsize=(7.5, 6))
    sns.heatmap(pm, annot=True, fmt="d", cmap="Purples", ax=ax, linewidths=0.5,
                cbar_kws={"label": "shared crop classes"})
    ax.set_title(
        f"Cross-region transfer: crop classes shared by both regions at >= {floor} "
        "labels\n(diagonal = classes available in that region alone)", fontsize=10)
    ax.set_xlabel("")
    ax.set_ylabel("")
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    plt.setp(ax.get_yticklabels(), rotation=0, fontsize=8)
    save(fig, "15_region_pair_intersection.png")

    fig, axes = plt.subplots(2, 1, figsize=(13, 10))
    for ax, m, title in [
        (axes[0], cnt, f"Crop type x region (counts; classes with >= {floor} somewhere)"),
        (axes[1], acnt, "Aggregated group x region (counts)"),
    ]:
        sns.heatmap(m, annot=True, fmt="d", cmap="YlOrRd", ax=ax, linewidths=0.4,
                    cbar_kws={"label": "labels"},
                    mask=(m == 0), annot_kws={"size": 7})
        ax.set_title(title + f"; bold border = >= {floor}")
        ax.set_xlabel("")
        ax.set_ylabel("")
        plt.setp(ax.get_xticklabels(), rotation=40, ha="right", fontsize=8)
        # outline cells meeting the floor
        for i, r in enumerate(m.index):
            for j, c in enumerate(m.columns):
                if m.loc[r, c] >= floor:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, fill=False,
                                               edgecolor="black", lw=1.6))
    save(fig, "10_region_class_intersection.png")
    return cnt, acnt


def find_duplicates(g, F):
    """Exact and near-duplicate label locations, within and across source datasets."""
    ok = np.isfinite(g.lon.to_numpy()) & np.isfinite(g.lat.to_numpy())
    gg = g[ok]
    pts = gpd.GeoDataFrame(
        geometry=gpd.points_from_xy(gg.lon, gg.lat), crs="EPSG:4326"
    ).to_crs("EPSG:6933")
    xy = np.c_[pts.geometry.x, pts.geometry.y]
    tree = cKDTree(xy)
    F["n_labels_without_coordinates"] = int((~ok).sum())
    F["n_exact_coordinate_duplicates"] = int(
        len(g) - len(g[["lat", "lon"]].drop_duplicates())
    )
    ds = gg["dataset"].to_numpy()
    out = {}
    for radius in [10, 100, 1000]:
        pairs = tree.query_pairs(r=radius, output_type="ndarray")
        cross = int((ds[pairs[:, 0]] != ds[pairs[:, 1]]).sum()) if len(pairs) else 0
        involved = len(np.unique(pairs)) if len(pairs) else 0
        out[radius] = {"n_pairs": int(len(pairs)), "n_cross_dataset_pairs": cross,
                       "n_labels_involved": int(involved),
                       "pct_labels_involved": round(involved / len(gg) * 100, 2)}
    F["near_duplicates"] = out
    print("  duplicates:", json.dumps(out))


# ---------------------------------------------------------------- 3. arrays
def scan_arrays(g, n_sample: int, F):
    """Stratified sample of the .h5 feature arrays. Cached to phenology.parquet."""
    sub = g[g.has_array].copy()
    sub["stratum"] = sub["region"] + "|" + sub["fao_group"].fillna("none")
    cap = max(1, n_sample // max(1, sub["stratum"].nunique()))
    picked = (
        sub.groupby("stratum", group_keys=False)
        .apply(lambda d: d.sample(min(len(d), cap), random_state=SEED), include_groups=False)
    )
    if len(picked) > n_sample:
        picked = picked.sample(n_sample, random_state=SEED)
    print(f"  reading {len(picked):,} .h5 arrays "
          f"({sub['stratum'].nunique()} strata, cap {cap}) ...")

    rows, ndvi = [], []
    bad = 0
    for i, (path, crop, fao, region, country, ds, yr, poly) in enumerate(zip(
        picked.array_path, picked.crop, picked.fao_group, picked.region,
        picked.country, picked.dataset, picked.dominant_year, picked.is_polygon,
    )):
        try:
            with h5py.File(path, "r") as h:
                a = np.asarray(h["array"])
                at = h.attrs
                la, lo = float(at["label_lat"]), float(at["label_lon"])
                ia, io = float(at["instance_lat"]), float(at["instance_lon"])
        except Exception:
            bad += 1
            continue
        # distance between the label coordinate and the pixel actually sampled
        offset_m = float(np.hypot((ia - la) * 111_320.0,
                                  (io - lo) * 111_320.0 * np.cos(np.radians(la))))
        if a.shape != (N_TIMESTEPS, len(BANDS)):
            bad += 1
            continue
        v = a[:, NDVI_IDX]
        rows.append(dict(
            path=path, crop=crop, fao_group=fao, region=region, country=country,
            dataset=ds, dominant_year=yr, is_polygon=bool(poly),
            n_nan=int(np.isnan(a).sum()),
            pixel_offset_m=offset_m,
            ndvi_mean=float(v.mean()), ndvi_min=float(v.min()), ndvi_max=float(v.max()),
            ndvi_amp=float(v.max() - v.min()), ndvi_peak_step=int(np.argmax(v)),
            n_constant_bands=int((a.std(axis=0) == 0).sum()),
            b4_mean=float(a[:, BANDS.index("B4")].mean()),
            b8_mean=float(a[:, BANDS.index("B8")].mean()),
            vv_mean=float(a[:, BANDS.index("VV")].mean()),
            t2m_mean=float(a[:, BANDS.index("temperature_2m")].mean()),
            t2m_amp=float(np.ptp(a[:, BANDS.index("temperature_2m")])),
            precip_sum=float(a[:, BANDS.index("total_precipitation")].sum()),
            elevation=float(a[0, BANDS.index("elevation")]),
        ))
        ndvi.append(v)
        if (i + 1) % 2000 == 0:
            print(f"    {i+1:,} / {len(picked):,}")
    ph = pd.DataFrame(rows)
    ndvi = np.array(ndvi)
    for t in range(N_TIMESTEPS):
        ph[f"ndvi_t{t:02d}"] = ndvi[:, t]
    ph.to_parquet(OUT_PHENOLOGY, index=False)
    print(f"  wrote {OUT_PHENOLOGY.name} ({len(ph):,} rows, {bad} unreadable)")

    F["array_sample_n"] = int(len(ph))
    F["array_sample_unreadable"] = int(bad)
    F["array_shape"] = [N_TIMESTEPS, len(BANDS)]
    F["array_bands"] = BANDS
    F["arrays_total_on_disk"] = int(sum(1 for _ in ARRAYS_DIR.glob("*.h5")))
    F["array_nan_count"] = int(ph.n_nan.sum())
    F["array_constant_band_mean"] = round(float(ph.n_constant_bands.mean()), 3)
    F["ndvi_peak_step_by_region"] = (
        ph.groupby("region").ndvi_peak_step.median().round(1).to_dict()
    )
    # label-to-pixel offset: the exporter batches labels into shared GeoTIFFs and then
    # snaps to the nearest grid coordinate of that tif, so a label outside the tif's
    # extent silently receives the spectral signature of a different place.
    off = ph.pixel_offset_m
    F["pixel_offset_m"] = {
        "median": round(float(off.median()), 2),
        "p95": round(float(off.quantile(0.95)), 2),
        "p99": round(float(off.quantile(0.99)), 2),
        "max": round(float(off.max()), 1),
    }
    F["pixel_offset_over_20m"] = int((off > 20).sum())
    F["pixel_offset_over_20m_pct"] = round(float((off > 20).mean() * 100), 2)
    F["pixel_offset_over_1km"] = int((off > 1000).sum())
    F["pixel_offset_over_1km_pct"] = round(float((off > 1000).mean() * 100), 2)
    F["pixel_offset_over_20m_by_dataset"] = (
        ph.assign(bad=off > 20).groupby("dataset").bad.agg(["size", "sum"])
        .assign(pct=lambda d: (d["sum"] / d["size"] * 100).round(1))
        .sort_values("pct", ascending=False).head(12).reset_index().to_dict("records")
    )
    return ph


def fig_example_ts(g, ph, F):
    """One instance per region for three widely shared crops: all bands plus NDVI."""
    crops = [c for c in ["maize", "rice", "wheat"] if (ph.crop == c).any()]
    for crop in crops:
        d = ph[ph.crop == crop]
        regions = d.region.value_counts().head(3).index.tolist()
        fig, axes = plt.subplots(len(regions), 2, figsize=(13, 3.1 * len(regions)),
                                 squeeze=False)
        for r, region in enumerate(regions):
            pick = d[d.region == region].sample(1, random_state=SEED).iloc[0]
            with h5py.File(pick.path, "r") as h:
                a = np.asarray(h["array"])
            x = np.arange(N_TIMESTEPS)
            ax = axes[r][0]
            for bi, b in enumerate(BANDS[2:13], start=2):
                ax.plot(x, a[:, bi], lw=1, label=b)
            ax.set_title(f"{region} - {pick.country} - Sentinel-2 TOA reflectance")
            ax.set_ylabel("reflectance x 1e4")
            if r == 0:
                ax.legend(ncol=6, fontsize=6, loc="upper left")
            ax2 = axes[r][1]
            ax2.plot(x, a[:, NDVI_IDX], color="darkgreen", marker="o", label="NDVI")
            ax2.set_ylim(-0.1, 1.0)
            ax2.set_ylabel("NDVI")
            axt = ax2.twinx()
            axt.plot(x, a[:, BANDS.index("VV")], color="firebrick", lw=1, ls="--",
                     label="S1 VV (dB)")
            axt.set_ylabel("VV (dB)", color="firebrick")
            ax2.set_title(f"{region} - NDVI and Sentinel-1 VV")
            for ax_ in (ax, ax2):
                ax_.set_xticks(x)
                ax_.set_xticklabels([f"t{ i }" for i in x], fontsize=7)
        axes[-1][0].set_xlabel("timestep (30 days each; t0 starts 1 February)")
        axes[-1][1].set_xlabel("timestep (30 days each; t0 starts 1 February)")
        fig.suptitle(f"CropHarvest feature arrays, crop = {crop}", y=1.001)
        save(fig, f"11_bands_example_ts_{crop}.png")
    F["example_ts_crops"] = crops


def fig_phenology(g, ph, F):
    """Mean NDVI trajectory per aggregated group, faceted by region."""
    d = ph[ph.crop.notna()].copy()
    d["agg_group"] = d.crop.map(CROP_TO_AGGREGATE)
    d = d[d.agg_group.notna()]
    regions = (
        d.groupby("region").agg_group.nunique().sort_values(ascending=False).head(4).index
    )
    ncols = len(regions)
    fig, axes = plt.subplots(1, ncols, figsize=(4.0 * ncols, 4.2), sharey=True,
                             squeeze=False)
    cols = [f"ndvi_t{t:02d}" for t in range(N_TIMESTEPS)]
    summary = {}
    for j, region in enumerate(regions):
        ax = axes[0][j]
        dr = d[d.region == region]
        for grp, sub in dr.groupby("agg_group"):
            if len(sub) < 15:
                continue
            m = sub[cols].mean().to_numpy()
            s = sub[cols].std().to_numpy()
            x = np.arange(N_TIMESTEPS)
            ax.plot(x, m, marker="o", ms=3, label=f"{grp} (n={len(sub)})")
            ax.fill_between(x, m - s, m + s, alpha=0.12)
            summary.setdefault(region, {})[grp] = dict(
                n=int(len(sub)), peak_step=int(np.argmax(m)), peak_ndvi=round(float(m.max()), 3),
                amplitude=round(float(m.max() - m.min()), 3))
        ax.set_title(region, fontsize=10)
        ax.set_xticks(range(N_TIMESTEPS))
        ax.set_xticklabels([f"{(1 + t) % 12 + 1}" for t in range(N_TIMESTEPS)], fontsize=7)
        ax.set_xlabel("approximate calendar month of timestep")
        ax.legend(fontsize=6)
    axes[0][0].set_ylabel("NDVI (mean +/- 1 sd)")
    fig.suptitle(
        "NDVI phenology by aggregated crop group and region "
        f"(sample of {len(d):,} instances; timestep 0 begins 1 February)", y=1.02)
    save(fig, "12_ndvi_phenology_by_region.png")
    F["phenology_summary"] = summary


def fig_array_diagnostics(g, ph, F):
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2))
    axes[0].hist(ph.n_constant_bands, bins=np.arange(0, 8) - 0.5, color="steelblue")
    axes[0].set_xlabel("bands constant across all 12 timesteps")
    axes[0].set_ylabel("instances")
    axes[0].set_title("Constant-band count\n(2 = the SRTM static bands, as expected)")

    sns.boxplot(data=ph, x="region", y="ndvi_amp", ax=axes[1], color="steelblue",
                fliersize=1)
    axes[1].set_ylabel("NDVI amplitude (max - min)")
    axes[1].set_xlabel("")
    axes[1].set_title("Seasonal NDVI amplitude by region")
    plt.setp(axes[1].get_xticklabels(), rotation=40, ha="right", fontsize=7)

    sns.boxplot(data=ph, x="region", y="t2m_amp", ax=axes[2], color="indianred",
                fliersize=1)
    axes[2].set_ylabel("ERA5 2 m temperature range (K)")
    axes[2].set_xlabel("")
    axes[2].set_title("Annual temperature range: tropical vs temperate")
    plt.setp(axes[2].get_xticklabels(), rotation=40, ha="right", fontsize=7)
    off = ph.pixel_offset_m.clip(lower=0.1)
    axes[3].hist(np.log10(off), bins=60, color="darkorange")
    axes[3].axvline(1, color="crimson", ls="--", lw=1)
    axes[3].text(1.05, axes[3].get_ylim()[1] * 0.8, "one 10 m\npixel", fontsize=8,
                 color="crimson")
    axes[3].set_xlabel("log10 distance, label to sampled pixel (m)")
    axes[3].set_ylabel("instances")
    axes[3].set_title(
        "Label-to-pixel offset\n"
        f"{(ph.pixel_offset_m > 20).mean()*100:.1f} % exceed 20 m")
    save(fig, "13_array_diagnostics.png")

    F["ndvi_amp_by_region"] = ph.groupby("region").ndvi_amp.median().round(3).to_dict()
    F["t2m_amp_by_region"] = ph.groupby("region").t2m_amp.median().round(2).to_dict()


def fig_comparability(g, F):
    """Side by side: what a shared pipeline has to reconcile."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))
    ax = axes[0]
    # EuroCropsML: irregular acquisitions, ~46 per year; CropHarvest: 12 fixed steps
    RNG2 = np.random.default_rng(SEED)
    doy = np.sort(RNG2.choice(np.arange(60, 300), size=46, replace=False))
    ax.vlines(doy, 0.55, 0.95, color="steelblue", lw=1)
    ax.text(5, 1.02, "EuroCropsML: irregular cloud-filtered S2 acquisitions "
                     "(median 45 per parcel, calendar year 2021)", fontsize=8)
    ch = np.arange(32, 32 + 12 * 30, 30)
    ax.vlines(ch % 365, 0.05, 0.45, color="crimson", lw=2.5)
    ax.text(5, 0.48, "CropHarvest: 12 fixed 30-day composites, window anchored on "
                     "1 February", fontsize=8)
    ax.set_xlim(0, 366)
    ax.set_ylim(0, 1.15)
    ax.set_yticks([])
    ax.set_xlabel("day of year")
    ax.set_title("Temporal sampling: irregular vs fixed monthly grid")

    ax = axes[1]
    ec = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09",
          "B10", "B11", "B12"]
    ch_b = BANDS
    common = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B9", "B11", "B12"]
    ax.barh(["EuroCropsML\n(T, 13)"], [len(ec)], color="steelblue")
    ax.barh(["CropHarvest\n(12, 18)"], [len(ch_b)], color="crimson")
    ax.barh(["shared S2 bands"], [len(common)], color="seagreen")
    for i, (lab, v) in enumerate([("13 S2 bands, L1C TOA", len(ec)),
                                  ("2 S1 + 11 S2 + 2 ERA5 + 2 SRTM + NDVI", len(ch_b)),
                                  ("B1 and B10 dropped by CropHarvest", len(common))]):
        ax.text(v + 0.2, i, lab, va="center", fontsize=8)
    ax.set_xlim(0, 32)
    ax.set_xlabel("channels")
    ax.set_title("Channel sets: the shared subset is 11 Sentinel-2 bands")
    save(fig, "14_comparability_eurocrops.png")
    F["shared_s2_bands"] = common
    F["n_shared_s2_bands"] = len(common)
    F["eurocropsml_only_bands"] = ["B01", "B10"]
    F["cropharvest_only_channels"] = ["VV", "VH", "temperature_2m",
                                      "total_precipitation", "elevation", "slope", "NDVI"]


# ---------------------------------------------------------------- 4. scheme
def write_scheme(g, F, floor=150):
    sub = g[g.is_crop_type & g.array_usable]
    cnt = sub.groupby(["region", "crop"]).size()
    acnt = sub[sub.agg_group.notna()].groupby(["region", "agg_group"]).size()

    regions = {}
    for region in sorted(r for r in g.region.unique() if r != "Other"):
        crops = sorted(
            [(c, int(n)) for (r, c), n in cnt.items() if r == region and n >= floor],
            key=lambda x: -x[1])
        aggs = sorted(
            [(c, int(n)) for (r, c), n in acnt.items() if r == region and n >= floor],
            key=lambda x: -x[1])
        n_total = int((g.region == region).sum())
        n_poly = int(((g.region == region) & g.is_polygon & g.array_usable).sum())
        if len(crops) >= 6:
            task = "multiclass_croptype"
        elif len(crops) >= 2:
            task = "small_multiclass_croptype"
        elif len(aggs) >= 3:
            task = "multiclass_aggregated"
        elif n_total >= 1000:
            task = "binary_cropland"
        else:
            task = "excluded_insufficient_labels"
        regions[region] = {
            "task": task,
            "n_labels_total": n_total,
            "n_polygon_with_array": n_poly,
            "crop_classes": [c for c, _ in crops],
            "crop_class_counts": {c: n for c, n in crops},
            "aggregated_classes": [c for c, _ in aggs],
            "aggregated_class_counts": {c: n for c, n in aggs},
        }

    shared_crops = F["cross_region_shared_crops"]
    shared_aggs = F["cross_region_shared_agg_groups"]

    scheme = {
        "_meta": {
            "generated_by": "results/eda/cropharvest/_run_analysis.py",
            "generated_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "dataset": "CropHarvest",
            "zenodo_record": 10251170,
            "zenodo_doi": "10.5281/zenodo.10251170",
            "labels_md5": "54a5070f103bc3e635afba27c139ac8d",
            "seed": SEED,
            "min_labels_per_class_per_region": floor,
            "basis": (
                "counts computed on the full labels.geojson (113,893 labels) restricted "
                "to labels that carry a crop type and have a feature array on disk"
            ),
        },
        "exclusions": {
            "binary_only_source_datasets": BINARY_ONLY_DATASETS,
            "drop_labels_without_feature_array": True,
            "drop_label_to_pixel_offset_above_m": 14.14,
            "fully_mismatched_source_datasets": ["tanzania", "uganda"],
            "drop_non_crop_type_labels": sorted(NON_CROP_TYPE_LABELS),
            "drop_fao_groups": ["non_crop", "other"],
            "drop_label_years_before": 2017,
            "merge_french_rpg_tenure_suffixes": True,
            "flagged_low_geolocation_precision_sources": [
                "geowiki-landcover-2017", "croplands",
            ],
        },
        "regions": regions,
        "cross_region_transfer": {
            "shared_crop_classes": shared_crops,
            "shared_aggregated_classes": shared_aggs,
            "admissible_region_pairs": [
                {"region_a": d["region_a"], "region_b": d["region_b"],
                 "shared_classes": d["shared"]}
                for d in F.get("region_pair_intersections", []) if d["n_shared"] >= 2
            ],
            "note": (
                "a transfer pair is only admissible where both regions reach the "
                "minimum count on every class in the intersection"
            ),
        },
        "aggregation_map": CROP_TO_AGGREGATE,
        "label_normalisation_map": LABEL_NORM,
    }

    OUT_SCHEME.parent.mkdir(parents=True, exist_ok=True)
    try:
        import yaml

        with open(OUT_SCHEME, "w", encoding="utf-8") as f:
            yaml.safe_dump(scheme, f, sort_keys=False, allow_unicode=True, width=100)
    except ImportError:  # pragma: no cover
        with open(OUT_SCHEME.with_suffix(".json"), "w", encoding="utf-8") as f:
            json.dump(scheme, f, indent=2)
    print(f"  wrote {OUT_SCHEME}")
    F["scheme_regions"] = {r: v["task"] for r, v in regions.items()}
    F["scheme_multiclass_regions"] = [
        r for r, v in regions.items() if v["task"].startswith("multiclass")]
    return scheme


# ---------------------------------------------------------------- main
def main(array_sample: int = 15000, skip_arrays: bool = False):
    F: dict = {
        "_basis": "all counts are from the full labels.geojson unless stated otherwise",
        "zenodo_record": 10251170,
        "zenodo_doi": "10.5281/zenodo.10251170",
        "licence": "CC-BY-SA-4.0",
    }
    g = build_catalogue()
    F["n_labels_total"] = int(len(g))
    F["crs"] = str(g.crs)
    F["columns"] = [c for c in g.columns if c != "geometry"]
    F["n_with_array"] = int(g.has_array.sum())
    F["n_offset_ok"] = int(g.offset_ok.sum())
    F["n_array_usable"] = int(g.array_usable.sum())
    F["n_lost_to_offset"] = int((g.has_array & ~g.offset_ok).sum())
    F["offset_scan"] = json.loads((HERE / "offset_scan.json").read_text(encoding="utf-8"))         if (HERE / "offset_scan.json").exists() else {}
    F["n_without_array"] = int((~g.has_array).sum())
    F["n_is_test"] = int(g.is_test.sum())
    F["is_crop_counts"] = g.is_crop.value_counts().to_dict()
    F["excluded_label_counts"] = g.attrs.get("excluded_label_counts", {})

    print("figures ...")
    fig_regions(g, F)
    fig_sources(g, F)
    fig_geometry(g, F)
    fig_polygon_area(g, F)
    inv = fig_croptype(g, F)
    inv.reset_index().to_parquet(OUT_INVENTORY, index=False)
    fig_fao_by_region(g, F)
    fig_years(g, F)
    fig_map(g, F)
    fig_class_survival(g, F)
    fig_intersection(g, F)
    print("duplicates ...")
    find_duplicates(g, F)
    fig_comparability(g, F)

    if not skip_arrays:
        print("arrays ...")
        ph = scan_arrays(g, array_sample, F)
        fig_example_ts(g, ph, F)
        fig_phenology(g, ph, F)
        fig_array_diagnostics(g, ph, F)

    print("scheme ...")
    write_scheme(g, F)

    with open(OUT_FINDINGS, "w", encoding="utf-8") as f:
        json.dump(F, f, indent=2, default=str)
    print(f"wrote {OUT_FINDINGS.name} ({len(F)} keys)")

    with open(OUT_RUNCFG, "w", encoding="utf-8") as f:
        json.dump({
            "script": str(HERE / "_run_analysis.py"),
            "run_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "seed": SEED,
            "data_root": str(DATA_ROOT),
            "labels_path": str(LABELS_PATH),
            "arrays_dir": str(ARRAYS_DIR),
            "natural_earth": str(NE_PATH),
            "array_sample_requested": array_sample,
            "skip_arrays": skip_arrays,
            "zenodo_record": 10251170,
            "bands": BANDS,
            "n_timesteps": N_TIMESTEPS,
            "days_per_timestep": DAYS_PER_TIMESTEP,
            "class_floor": F.get("class_floor"),
            "alphaearth_years": ALPHAEARTH_YEARS,
            "tessera_global_years": TESSERA_GLOBAL_YEARS,
        }, f, indent=2)
    print(f"wrote {OUT_RUNCFG.name}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--array-sample", type=int, default=15000)
    ap.add_argument("--skip-arrays", action="store_true")
    a = ap.parse_args()
    main(array_sample=a.array_sample, skip_arrays=a.skip_arrays)
