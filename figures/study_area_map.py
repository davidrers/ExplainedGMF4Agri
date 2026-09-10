"""
Study-area map for the proposal deck (slide 7).

One world map (Robinson) highlighting the two benchmark datasets:
  - EuroCropsML (primary): Estonia, Latvia, Portugal      -> gold
  - CropHarvest (secondary): global smallholder/tropical  -> teal

A Europe inset makes the three small EuroCropsML countries legible.
Palette matches the deck (cream background, forest/gold/teal).
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib import font_manager  # noqa: F401
import matplotlib.patheffects as pe

OUT = Path(__file__).resolve().parent
SRC = OUT / "ne_110m_admin_0_countries.geojson"

# Deck palette
CREAM   = "#F4F2EA"
LAND    = "#E3DECC"
LAND_ED = "#F4F2EA"
GOLD    = "#C0860F"
GOLD_ED = "#8A5E0A"
TEAL    = "#1A6B5B"
TEAL_ED = "#0E4438"
INK     = "#143D33"

EURO = {"Estonia", "Latvia", "Portugal"}
CROP = {
    "Rwanda", "Kenya", "Ethiopia", "Sudan", "Mali", "United Republic of Tanzania",
    "Uganda", "Zimbabwe", "Togo", "France", "Germany", "Canada", "Brazil",
    "Uzbekistan", "Tajikistan",
}

ROBINSON = "ESRI:54030"


def categorize(name):
    if name in EURO:
        return "euro"
    if name in CROP:
        return "crop"
    return "other"


def europe_bounds_robinson():
    """Robinson-projected bounding box around Europe for the inset."""
    box = gpd.GeoSeries.from_xy(
        x=[-12, 32, -12, 32], y=[35, 35, 62, 62], crs="EPSG:4326"
    ).to_crs(ROBINSON)
    minx, miny, maxx, maxy = box.total_bounds
    return minx, miny, maxx, maxy


def draw(ax, gdf, edge_w=0.4):
    gdf[gdf.cat == "other"].plot(ax=ax, color=LAND, edgecolor=LAND_ED, linewidth=edge_w)
    gdf[gdf.cat == "crop"].plot(ax=ax, color=TEAL, edgecolor=TEAL_ED, linewidth=edge_w)
    gdf[gdf.cat == "euro"].plot(ax=ax, color=GOLD, edgecolor=GOLD_ED, linewidth=edge_w)


def main():
    g = gpd.read_file(SRC)
    g = g[g["ADMIN"] != "Antarctica"].copy()
    g["cat"] = g["ADMIN"].map(categorize)
    g = g.to_crs(ROBINSON)

    fig, ax = plt.subplots(figsize=(13, 6.8))
    fig.patch.set_facecolor(CREAM)
    ax.set_facecolor(CREAM)

    draw(ax, g, edge_w=0.4)
    ax.set_axis_off()
    ax.margins(0)

    # Legend
    handles = [
        mpatches.Patch(facecolor=GOLD, edgecolor=GOLD_ED, label="EuroCropsML  —  primary (Estonia · Latvia · Portugal)"),
        mpatches.Patch(facecolor=TEAL, edgecolor=TEAL_ED, label="CropHarvest  —  secondary (smallholder & tropical)"),
    ]
    leg = ax.legend(
        handles=handles, loc="lower right", frameon=True, fontsize=12.5,
        bbox_to_anchor=(0.995, 0.03), handlelength=1.3, handleheight=1.1,
        borderpad=0.8, labelspacing=0.7,
    )
    leg.get_frame().set_facecolor("#FBFAF4")
    leg.get_frame().set_edgecolor("#D6D1C0")
    for t in leg.get_texts():
        t.set_color(INK)

    # ---- Europe inset ----
    axin = ax.inset_axes([0.015, 0.06, 0.26, 0.44])
    axin.set_facecolor(CREAM)
    draw(axin, g, edge_w=0.5)
    minx, miny, maxx, maxy = europe_bounds_robinson()
    axin.set_xlim(minx, maxx)
    axin.set_ylim(miny, maxy)
    axin.set_xticks([]); axin.set_yticks([])
    for s in axin.spines.values():
        s.set_edgecolor("#B9B49C"); s.set_linewidth(1.0)
    axin.set_title("Europe (detail)", fontsize=10.5, color=INK, pad=3,
                   fontfamily="serif")

    # Label the three EuroCropsML countries in the inset
    halo = [pe.withStroke(linewidth=2.4, foreground="#FBFAF4")]
    g_proj = g.set_index("ADMIN")
    # per-label offsets (points) so adjacent Baltic labels don't overlap
    offsets = {"Estonia": (16, 9), "Latvia": (16, -9), "Portugal": (0, 0)}
    for nm, (dx, dy) in offsets.items():
        p = g_proj.loc[nm].geometry.representative_point()
        axin.annotate(nm, (p.x, p.y), xytext=(dx, dy), textcoords="offset points",
                      fontsize=10, fontweight="bold", color=GOLD_ED,
                      ha="center", va="center", path_effects=halo)

    fig.tight_layout(pad=0.4)
    png = OUT / "study_area_map.png"
    svg = OUT / "study_area_map.svg"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=CREAM)
    fig.savefig(svg, bbox_inches="tight", facecolor=CREAM)
    plt.close(fig)
    print("wrote", png)
    print("wrote", svg)


if __name__ == "__main__":
    main()
