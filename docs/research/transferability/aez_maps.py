"""Maps of the zonings for the transfer plan: GAEZ v5 AEZ with its classes, and GAEZ against ESA WorldCereal.

* `figures/aez33_classes.png`: the GAEZ v5 AEZ at 1 km (AEZ33-1KM) as GAEZ maps it, special classes included, in
  GAEZ's own colours; the candidate countries in full colour and outlined, the rest of Europe faded.
* `figures/zonings_compared.png`: every candidate block coloured by its GAEZ climatic zone (special classes filled,
  `zone_pools.py`) and by its ESA WorldCereal AEZ (v100, crop-calendar zones).
* `worldcereal_zones.csv`: chips per WorldCereal zone and country, with each zone's crop calendar.
* `figures/zones_matrix.png`: every chip cell of the 19 countries (a cell holding a 2021 parcel) in the colour of its
  zone in the matrices of `transfer_setup.md`; classes outside the zone matrix in beige.

    poetry run python docs/research/transferability/aez_maps.py

Needs `zone_blocks.parquet` (from `zone_pools.py`) and `data/worldcereal/WorldCereal_AEZ.geojson` (see its SOURCES.txt).
"""
from __future__ import annotations

import json

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from matplotlib.collections import PatchCollection
from matplotlib.patches import Patch, Rectangle
from rasterio.enums import Resampling
from rasterio.features import rasterize
from rasterio.transform import from_origin
from rasterio.warp import reproject
from rasterio.windows import from_bounds
from shapely.geometry import Point

from gaez_domains import BOUNDS, DOMAINS, GAEZ, HERE, NUTS, ROOT

WORLDCEREAL = ROOT / "data/worldcereal/WorldCereal_AEZ.geojson"
EXTENT = (2.55e6, 1.45e6, 5.95e6, 4.95e6)      # EPSG:3035, the candidate countries
PIXEL = 2000.0
BLOCK = 8960.0
FIGS = HERE / "figures"
GAEZ_ZONES = {"cool wet": "#4363d8", "cool moist": "#3cb44b", "moderate": "#ffe119", "subtropics mod. cool": "#f58231",
              "subtropics cool": "#e6194b", "cold": "#911eb4", "cool dry": "#9a6324"}
WC_ZONES = {46172: "West", 22190: "Central, east and Baltic", 43153: "Iberia", 22095: "North",
            43170: "Mediterranean France and Italy", 6136: "Bulgaria, zone 6136", 46171: "Bulgaria, zone 46171"}
WC_COLOURS = {"West": "#4363d8", "Central, east and Baltic": "#3cb44b", "Iberia": "#f58231", "North": "#911eb4",
              "Mediterranean France and Italy": "#e6194b", "Bulgaria, zone 6136": "#ffe119", "Bulgaria, zone 46171": "#d4b000"}


def outlines():
    nuts = gpd.read_file(NUTS)
    land = nuts[nuts.LEVL_CODE == 0].to_crs(3035)
    codes = [c for cs in DOMAINS.values() for c in cs]
    return land, nuts[nuts.NUTS_ID.isin(codes)].to_crs(3035)


def aez_classes_map() -> None:
    style = json.loads((GAEZ / "GAEZ-V5.AEZ33.style.json").read_text())["colorMap"]["items"]
    colour = {int(i["quantity"]): i["color"] for i in style}
    label = {int(i["quantity"]): i["label"] for i in style}
    x0, y0, x1, y1 = EXTENT
    shape = (int((y1 - y0) / PIXEL), int((x1 - x0) / PIXEL))
    transform = from_origin(x0, y1, PIXEL, PIXEL)
    with rasterio.open(GAEZ / "GAEZ-V5.AEZ33-1KM.tif") as src:
        win = from_bounds(*BOUNDS, transform=src.transform)
        aez = src.read(1, window=win).astype("int16")
        out = np.zeros(shape, dtype="int16")
        reproject(aez, out, src_transform=src.window_transform(win), src_crs=src.crs, dst_transform=transform,
                  dst_crs="EPSG:3035", resampling=Resampling.nearest)
    land, candidates = outlines()
    inside = rasterize([(g, 1) for g in candidates.geometry], out_shape=shape, transform=transform, fill=0).astype(bool)

    rgb = np.ones(shape + (3,))
    for code, hexc in colour.items():
        if 1 <= code <= 33:
            rgb[out == code] = np.array([int(hexc[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    faded = 1 - 0.3 * (1 - rgb)                 # outside the candidate countries
    rgb = np.where(inside[..., None], rgb, faded)
    rgb[(out < 1) | (out > 33)] = 1.0

    present = pd.Series(out[inside]).value_counts()
    present = present[(present.index >= 1) & (present.index <= 33) & (present > 50)]
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.imshow(rgb, extent=(x0, x1, y0, y1), interpolation="nearest")
    land.boundary.plot(ax=ax, color="#bbbbbb", linewidth=0.3)
    candidates.dissolve().boundary.plot(ax=ax, color="#222222", linewidth=0.6)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    climatic = [Patch(color=colour[c], label=f"{c:>2}  {label[c]}") for c in sorted(present.index) if c <= 24]
    special = [Patch(color=colour[c], label=f"{c:>2}  {label[c]}") for c in sorted(present.index) if c >= 25]
    leg1 = ax.legend(handles=climatic, title="Climate: thermal regime; moisture", loc="upper left",
                     bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=10, title_fontsize=11, alignment="left")
    ax.add_artist(leg1)
    ax.legend(handles=special, title="Special classes (override the climate)", loc="upper left",
              bbox_to_anchor=(1.0, 0.52), frameon=False, fontsize=10, title_fontsize=11, alignment="left")
    ax.set_title("GAEZ v5 agro-ecological zones (AEZ, 33 classes, 1 km): the classes found in the candidate countries",
                 fontsize=12, loc="left")
    fig.savefig(FIGS / "aez33_classes.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


def zonings_compared() -> None:
    blocks = pd.read_parquet(HERE / "zone_blocks.parquet")
    wc = gpd.read_file(WORLDCEREAL).to_crs(3035)
    pts = gpd.GeoDataFrame(blocks, crs=3035, geometry=[Point((x + .5) * BLOCK, (y + .5) * BLOCK)
                                                         for x, y in zip(blocks.bx, blocks.by)])
    cal = ["tc-wintercereals_sos", "tc-wintercereals_eos", "tc-springcereals_sos", "tc-springcereals_eos",
           "tc-maize-main_sos", "tc-maize-main_eos"]
    j = gpd.sjoin(pts, wc[["aez_id", "aez_groupid", *cal, "geometry"]], how="left", predicate="within")
    j["wc_zone"] = j.aez_id.map(lambda v: WC_ZONES.get(int(v)) if pd.notna(v) else None)

    table = j.pivot_table(index="wc_zone", columns="country", values="chips", aggfunc="sum", fill_value=0)
    countries = list(table.columns)
    table["chips"] = table[countries].sum(axis=1)
    table["countries_1000"] = (table[countries] >= 1000).sum(axis=1)
    table = table.join(j.groupby("wc_zone")[["aez_id", *cal]].first())
    gz = j.pivot_table(index="wc_zone", columns="zone", values="chips", aggfunc="sum", fill_value=0)
    table.join(gz.add_prefix("gaez_")).sort_values("chips", ascending=False).to_csv(HERE / "worldcereal_zones.csv")

    land, candidates = outlines()
    fig, axes = plt.subplots(1, 2, figsize=(18, 9.5))
    panels = [(axes[0], "GAEZ v5 AEZ, climatic classes (special classes filled)", j.zone.map(GAEZ_ZONES),
               {z: c for z, c in GAEZ_ZONES.items() if z in set(j.zone)}),
              (axes[1], "ESA WorldCereal AEZ v100 (crop-calendar zones)", j.wc_zone.map(WC_COLOURS), WC_COLOURS)]
    for ax, title, colours, legend in panels:
        land.plot(ax=ax, color="#f2f2f2", edgecolor="#cccccc", linewidth=0.4)
        ax.add_collection(PatchCollection([Rectangle((x * BLOCK, y * BLOCK), BLOCK, BLOCK)
                                           for x, y in zip(j.bx, j.by)], facecolor=colours.fillna("#bbbbbb"),
                                          edgecolor="none"))
        candidates.boundary.plot(ax=ax, color="#333333", linewidth=0.3)
        ax.set_xlim(EXTENT[0], EXTENT[2])
        ax.set_ylim(EXTENT[1], EXTENT[3])
        ax.set_axis_off()
        ax.set_title(title, fontsize=13)
        ax.legend(handles=[Patch(color=c, label=z) for z, c in legend.items()], loc="upper left", frameon=False,
                  fontsize=10)
    fig.savefig(FIGS / "zonings_compared.png", dpi=90, bbox_inches="tight")
    plt.close(fig)


# The 12 zones of the zone matrix, in the colours of the matrices' key (special classes in greys)
MATRIX_ZONES = {
    24: ("24 Cold, wet", "#911eb4"), 21: ("21 Temperate cool, wet", "#4363d8"),
    20: ("20 Temperate cool, moist", "#3cb44b"), 17: ("17 Temperate moderate, moist", "#ffe119"),
    11: ("11 Subtropics mod. cool, sub-humid", "#f58231"), 12: ("12 Subtropics mod. cool, humid", "#f8a96a"),
    13: ("13 Subtropics cool, semi-arid", "#e6194b"), 14: ("14 Subtropics cool, sub-humid", "#ee5a7a"),
    15: ("15 Subtropics cool, humid", "#f497ab"), 26: ("26 Severe soil or terrain limitations", "#8c8c8c"),
    27: ("27 Ample irrigated soils", "#b8b8b8"), 28: ("28 Hydromorphic soils", "#9fb6c7"),
}


def zones_matrix_map() -> None:
    from gaez_domains import ROOT
    cells = pd.read_parquet(ROOT / "data/eurocrops/label_parcels_2021.parquet", columns=["gx", "gy", "aez"])
    cells = cells.drop_duplicates(["gx", "gy"])
    x0, y0, x1, y1 = EXTENT
    cell = 2240.0
    c0, r1 = int(x0 // cell), int(y1 // cell)
    w, h = int((x1 - x0) // cell), int((y1 - y0) // cell)
    rgb = np.ones((h, w, 3))
    col, row = cells.gx.to_numpy() - c0, r1 - 1 - cells.gy.to_numpy()
    ok = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    hexrgb = lambda c: np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    rgb[row[ok], col[ok]] = hexrgb("#e9dfc6")               # classes outside the zone matrix
    for z, (_, colour) in MATRIX_ZONES.items():
        m = ok & (cells.aez.to_numpy() == z)
        rgb[row[m], col[m]] = hexrgb(colour)
    land, candidates = outlines()
    fig, ax = plt.subplots(figsize=(13, 11))
    land.plot(ax=ax, color="#f2f2f2", edgecolor="#cccccc", linewidth=0.4)
    rgba = np.dstack([rgb, np.where(np.all(rgb == 1, axis=2), 0, 1)])
    ax.imshow(rgba, extent=(c0 * cell, (c0 + w) * cell, (r1 - h) * cell, r1 * cell), interpolation="nearest", zorder=2)
    candidates.dissolve().boundary.plot(ax=ax, color="#333333", linewidth=0.5, zorder=3)
    ax.set_xlim(x0, x1)
    ax.set_ylim(y0, y1)
    ax.set_axis_off()
    handles = [Patch(color=c, label=l) for l, c in MATRIX_ZONES.values()]
    handles.append(Patch(color="#e9dfc6", label="other classes, not in the zone matrix"))
    ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.0, 1.0), frameon=False, fontsize=10.5,
              title="The 12 zones of the zone matrix", title_fontsize=11.5, alignment="left")
    fig.savefig(FIGS / "zones_matrix.png", dpi=100, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(exist_ok=True)
    aez_classes_map()
    zonings_compared()
    zones_matrix_map()
