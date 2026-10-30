"""GAEZ v5 zones of the EuroCrops domains with 2021 labels, weighted by cropland.

For each candidate domain of the transfer matrix (a country, or the EuroCrops regions of it that have 2021 labels)
this computes, over its cropland only:

* the share of each GAEZ thermal climate class (RES01-MCL, the agro-climate);
* the share of each GAEZ agro-ecological zone (AEZ33, thermal regime x moisture, with the special classes);
* a climate profile from the 2001-2020 GAEZ normals: heat sum above 5 C (TS2), days above 5 C (LT2), growing
  period days (LGD, which carries the moisture limit) and annual precipitation (PRC).

Cropland weights are the GAEZ v5 cropland share (LR-LCC LC02, 30 arc-seconds), averaged to the 5 arc-minute grid of
the climate layers and read straight from the public bucket for the European window only.

Outputs, next to this file: `gaez_domains.csv` and `figures/gaez_agroclimate.png`, `figures/gaez_aez.png`.

    poetry run python docs/research/transferability/gaez_domains.py

Inputs: `data/gaez/v5/` (see its SOURCES.txt) and `data/gisco/NUTS_RG_20M_2021_4326.geojson` (Eurostat GISCO, NUTS
2021, 1:20 million).
"""

from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from rasterio.features import rasterize
from rasterio.windows import from_bounds

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
GAEZ = ROOT / "data/gaez/v5"
NUTS = ROOT / "data/gisco/NUTS_RG_20M_2021_4326.geojson"
CROPLAND = (
    "/vsicurl/https://storage.googleapis.com/fao-gismgr-gaez-v5-data/DATA/GAEZ-V5/MAPSET/LR-LCC/GAEZ-V5.LR-LCC.LC02.tif"
)

# Europe window on the 5 arc-minute grid (lon_min, lat_min, lon_max, lat_max).
BOUNDS = (-11.0, 35.0, 33.0, 70.5)
MIN_CROPLAND_PCT = 2.0  # cells below this share of cropland are drawn grey and carry no weight

# Domain -> NUTS 2021 codes. Outermost regions are left out (Canarias ES7, the French overseas FRY, Azores and
# Madeira PT2/PT3). Germany, Belgium and Italy are the EuroCrops regions with 2021 labels only.
DOMAINS = {
    "AT": ["AT"],
    "BE": ["BE2", "BE3"],
    "BG": ["BG"],
    "CZ": ["CZ"],
    "DE": ["DE4", "DE9", "DEA"],
    "DK": ["DK"],
    "EE": ["EE"],
    "ES": ["ES1", "ES2", "ES3", "ES4", "ES5", "ES6"],
    "FI": ["FI"],
    "FR": ["FR1", "FRB", "FRC", "FRD", "FRE", "FRF", "FRG", "FRH", "FRI", "FRJ", "FRK", "FRL", "FRM"],
    "IE": ["IE"],
    "IT": ["ITI1"],
    "LT": ["LT"],
    "LV": ["LV"],
    "NL": ["NL"],
    "PT": ["PT1"],
    "SE": ["SE"],
    "SI": ["SI"],
    "SK": ["SK"],
}


# Display groups, coloured to match the markers of transfer_setup.md (the GAEZ legends use near-identical greens).
# Thermal climate (RES01-MCL): value -> group.
CLIMATE_GROUPS = {
    3: "Subtropics, summer rainfall", 4: "Subtropics, winter rainfall (Mediterranean)",
    6: "Temperate oceanic", 7: "Temperate sub-continental", 8: "Temperate continental",
    9: "Boreal", 10: "Boreal", 11: "Boreal",
}
CLIMATE_COLOURS = {
    "Temperate oceanic": "#3cb44b", "Temperate sub-continental": "#4363d8", "Temperate continental": "#000075",
    "Subtropics, winter rainfall (Mediterranean)": "#ffe119", "Subtropics, summer rainfall": "#f58231",
    "Boreal": "#911eb4",
}
# AEZ33: thermal regime and moisture as present on European cropland; classes 25 to 33 override the climate.
AEZ_GROUPS = {
    21: "Temperate cool, wet", 20: "Temperate cool, moist", 19: "Temperate cool, dry",
    16: "Temperate moderate", 17: "Temperate moderate", 18: "Temperate moderate",
    10: "Subtropics moderately cool", 11: "Subtropics moderately cool", 12: "Subtropics moderately cool",
    13: "Subtropics cool", 14: "Subtropics cool", 15: "Subtropics cool",
    22: "Cold, no permafrost", 23: "Cold, no permafrost", 24: "Cold, no permafrost",
    **{c: "Special class (soil, terrain, irrigation, wetness, water)" for c in range(25, 34)},
}
AEZ_COLOURS = {
    "Temperate cool, wet": "#4363d8", "Temperate cool, moist": "#3cb44b", "Temperate cool, dry": "#9a6324",
    "Temperate moderate": "#ffe119", "Subtropics moderately cool": "#f58231", "Subtropics cool": "#e6194b",
    "Cold, no permafrost": "#911eb4", "Special class (soil, terrain, irrigation, wetness, water)": "#ffffff",
}
SPECIAL = "Special class (soil, terrain, irrigation, wetness, water)"


def grouped(classes: np.ndarray, groups: dict[int, str]) -> tuple[np.ndarray, dict[str, int]]:
    """Recode a class raster to display groups: (recoded raster, group label -> code)."""
    names = list(dict.fromkeys(groups.values()))
    code_of = {n: i + 1 for i, n in enumerate(names)}
    out = np.zeros(classes.shape, dtype="int16")
    for value, name in groups.items():
        out[classes == value] = code_of[name]
    return out, code_of


def aez33_labels() -> dict[int, str]:
    meta = json.loads((GAEZ / "GAEZ-V5.AEZ33.json").read_text())
    return {int(k): v["caption"] for k, v in meta["classes"].items()}


def read_window(path: Path | str):
    with rasterio.open(path) as src:
        win = from_bounds(*BOUNDS, transform=src.transform)
        return src.read(1, window=win), src.window_transform(win), src.nodata


def cropland_share(shape: tuple[int, int]) -> np.ndarray:
    """Mean cropland share (%) per 5 arc-minute cell, from the 30 arc-second layer (10 x 10 blocks)."""
    fine, _, _ = read_window(CROPLAND)
    fine = np.nan_to_num(fine.astype("float32"), nan=0.0)
    h, w = shape
    fine = fine[: h * 10, : w * 10]
    return fine.reshape(h, 10, w, 10).mean(axis=(1, 3))


def domain_raster(transform, shape) -> tuple[np.ndarray, gpd.GeoDataFrame]:
    nuts = gpd.read_file(NUTS)
    shapes, rows = [], []
    for k, (name, codes) in enumerate(DOMAINS.items(), start=1):
        geom = nuts[nuts.NUTS_ID.isin(codes)].geometry.union_all()
        shapes.append((geom, k))
        rows.append({"domain": name, "geometry": geom})
    ids = rasterize(shapes, out_shape=shape, transform=transform, fill=0, dtype="int16")
    return ids, gpd.GeoDataFrame(rows, crs="EPSG:4326")


def shares(classes: np.ndarray, ids: np.ndarray, weight: np.ndarray, labels: dict[int, str]) -> pd.DataFrame:
    out = {}
    for k, name in enumerate(DOMAINS, start=1):
        m = (ids == k) & (weight > 0) & (classes > 0)
        s = pd.Series(weight[m]).groupby(classes[m]).sum()
        out[name] = (100 * s / s.sum()).rename(index=lambda c: labels.get(int(c), str(c)))
    return pd.DataFrame(out).fillna(0).T


def profile(ids: np.ndarray, weight: np.ndarray) -> pd.DataFrame:
    layers = {"heat_sum_5C": "TS2", "days_above_5C": "LT2", "growing_period_days": "LGD", "precip_mm": "PRC"}
    data = {}
    for name, code in layers.items():
        arr, _, nodata = read_window(GAEZ / f"GAEZ-V5.RES01-{code}.HP0120.AGERA5.HIST.tif")
        data[name] = (arr.astype("float64"), nodata)
    rows = {}
    for k, dom in enumerate(DOMAINS, start=1):
        row = {}
        for name, (arr, nodata) in data.items():
            m = (ids == k) & (weight > 0) & (arr != nodata)
            row[name] = float(np.average(arr[m], weights=weight[m]))
        rows[dom] = row
    return pd.DataFrame(rows).T.round(0)


def draw(classes, transform, weight, outlines, legend_items, title, path):
    h, w = classes.shape
    left, top = transform.c, transform.f
    extent = (left, left + w * transform.a, top + h * transform.e, top)
    present = [c for c in sorted(legend_items) if ((classes == c) & (weight >= MIN_CROPLAND_PCT)).sum() >= 20]
    lut = {c: i for i, c in enumerate(present)}
    img = np.full(classes.shape, np.nan)
    for c, i in lut.items():
        img[(classes == c) & (weight >= MIN_CROPLAND_PCT)] = i
    fig, ax = plt.subplots(figsize=(9, 8.5))
    ax.set_facecolor("#cfcfcf")
    ax.imshow(img, extent=extent, cmap=ListedColormap([legend_items[c][1] for c in present]),
              vmin=-0.5, vmax=len(present) - 0.5, interpolation="nearest")
    nuts0 = gpd.read_file(NUTS)
    nuts0[nuts0.LEVL_CODE == 0].boundary.plot(ax=ax, color="#7a7a7a", linewidth=0.3)
    outlines.boundary.plot(ax=ax, color="black", linewidth=0.8)
    for _, r in outlines.iterrows():
        p = r.geometry.representative_point()
        ax.annotate(r.domain, (p.x, p.y), ha="center", va="center", fontsize=8, fontweight="bold",
                    bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none", "alpha": 0.75})
    ax.set_xlim(BOUNDS[0], BOUNDS[2])
    ax.set_ylim(BOUNDS[1], BOUNDS[3])
    ax.set_aspect(1 / np.cos(np.deg2rad(52)))
    ax.set_title(title, fontsize=11)
    ax.legend(handles=[Patch(facecolor=legend_items[c][1], edgecolor="black", linewidth=0.4,
                             label=legend_items[c][0]) for c in present]
              + [Patch(facecolor="#cfcfcf", label=f"cropland below {MIN_CROPLAND_PCT:g} % of the cell")],
              loc="upper center", bbox_to_anchor=(0.5, -0.01), ncol=2, fontsize=8, frameon=False)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def dominant(df: pd.DataFrame, exclude: str | None = None) -> pd.DataFrame:
    """Largest group per domain and its share; with `exclude`, shares are renormalised without that group."""
    d = df.drop(columns=[exclude], errors="ignore") if exclude else df
    d = 100 * d.div(d.sum(axis=1), axis=0)
    return pd.DataFrame({"group": d.idxmax(axis=1), "pct": d.max(axis=1).round(0)})


def main() -> None:
    mcl, transform, _ = read_window(GAEZ / "GAEZ-V5.RES01-MCL.HP0120.AGERA5.HIST.tif")
    aez, _, _ = read_window(GAEZ / "GAEZ-V5.AEZ33.tif")
    weight = cropland_share(mcl.shape)
    weight[weight < MIN_CROPLAND_PCT] = 0.0
    ids, outlines = domain_raster(transform, mcl.shape)

    mcl_g, mcl_codes = grouped(mcl, CLIMATE_GROUPS)
    aez_g, aez_codes = grouped(aez, AEZ_GROUPS)
    mcl_legend = {c: (n, CLIMATE_COLOURS[n]) for n, c in mcl_codes.items()}
    aez_legend = {c: (n, AEZ_COLOURS[n]) for n, c in aez_codes.items()}

    climate = shares(mcl_g, ids, weight, {c: v[0] for c, v in mcl_legend.items()})
    zones = shares(aez_g, ids, weight, {c: v[0] for c, v in aez_legend.items()})
    zones_detail = shares(aez, ids, weight, aez33_labels())
    prof = profile(ids, weight)

    summary = pd.concat({
        "agro_climate": dominant(climate),
        "aez_climatic": dominant(zones, exclude=SPECIAL),
    }, axis=1)
    summary[("aez_special", "pct")] = zones.get(SPECIAL, 0).round(0)
    summary = pd.concat([summary, pd.concat({"profile": prof}, axis=1)], axis=1)
    summary.to_csv(HERE / "gaez_domains.csv")
    zones_detail.round(1).to_csv(HERE / "gaez_domains_aez33_detail.csv")

    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)
    draw(mcl_g, transform, weight, outlines, mcl_legend,
         "Agro-climate: GAEZ v5 thermal climate, 2001-2020, on cropland", figs / "gaez_agroclimate.png")
    draw(aez_g, transform, weight, outlines, aez_legend,
         "Agro-ecological zones: GAEZ v5 AEZ (33 classes, grouped), on cropland", figs / "gaez_aez.png")

    pd.set_option("display.width", 250)
    print(summary.to_string())


if __name__ == "__main__":
    main()
