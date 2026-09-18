"""Exploratory analysis of the EuroCrops vector release, for the segmentation task.

Consumes the four GeoParquet layers written by ``scripts/data/fetch_eurocrops.py``
(Estonia, Latvia, Lithuania and Portugal, declaration year 2021) and characterises them
as what Phase 1 actually needs: an annotation source whose unit is the parcel polygon
and whose inference unit is the 10 m pixel.

The questions this script answers are therefore geometric as much as taxonomic. How many
labelled pixels does one polygon buy? How much of a polygon is boundary, and therefore
mixed at 10 m? How does the class prior shift when it is weighted by area, as a per-pixel
loss weights it, rather than by parcel count? How many 224 x 224 chips does a country's
declared area occupy, and what does that imply for the export?

Outputs
-------
``results/eda/eurocrops/findings.json``      every number quoted by the notebook
``results/eda/eurocrops/cache/*.parquet``    small aggregated tables, tracked in git
``results/eda/eurocrops/figures/*.png``      figures 01 to 12
``data/eurocrops/derived/<CC>_metrics.parquet``  per-parcel metrics, git-ignored

Every statistic is computed over all 1,810,723 parcels unless the text says otherwise;
the only sampled step is the validation of the boundary approximation in section 4,
which is seeded.

Usage
-----
    python results/eda/eurocrops/_vector_analysis.py [--force]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / "results" / "eda"))
import _hcat  # noqa: E402  the HCAT helper shared with the EuroCropsML analysis

PARQUET_DIR = REPO / "data" / "eurocrops" / "parquet"
DERIVED_DIR = REPO / "data" / "eurocrops" / "derived"
FIG_DIR = HERE / "figures"
CACHE_DIR = HERE / "cache"
OUT_FINDINGS = HERE / "findings.json"

SEED = 42
YEAR = 2021
#: Equal-area CRS for every geometric statistic, so countries are comparable.
EQUAL_AREA = "EPSG:3035"  # ETRS89-extended / LAEA Europe
PIXEL_M = 10.0  # Sentinel-2 ground sampling distance
CHIP_PX = 224  # chip side in pixels
CHIP_M = CHIP_PX * PIXEL_M  # 2,240 m
#: Export volume assumption, stated so the estimate can be recomputed: twelve monthly
#: composites, ten 10 m bands, int16.
EXPORT_BANDS, EXPORT_T, EXPORT_BYTES = 10, 12, 2

COUNTRIES = {"EE": "Estonia", "LV": "Latvia", "LT": "Lithuania", "PT": "Portugal"}
ORDER = ["Estonia", "Latvia", "Lithuania", "Portugal"]
PALETTE = {"Estonia": "#1f77b4", "Latvia": "#ff7f0e", "Lithuania": "#d62728", "Portugal": "#2ca02c"}

#: National identifier and declared-area columns, per country. The identifier is not a
#: parcel key everywhere: Lithuania publishes only a field-block number (KZS_NR, 278,891
#: distinct values for 1,102,471 geometries) and a holding number (NMA_ID, 120,901), so a
#: Lithuanian polygon cannot be addressed individually through the attribute table. The
#: declared-area attribute is in hectares everywhere except Portugal, which reports square
#: metres; the unit is stated here and verified in figure 11.
NATIVE = {
    "EE": {"id": "pollu_id", "id_kind": "parcel", "declared_area": "pindala_ha",
           "declared_area_unit": "ha", "nuts3": None},
    "LV": {"id": "PARCEL_ID", "id_kind": "parcel", "declared_area": "AREA_DECLA",
           "declared_area_unit": "ha", "nuts3": "EC_NUTS3"},
    "LT": {"id": "KZS_NR", "id_kind": "field_block", "declared_area": "DKL_PLOTAS",
           "declared_area_unit": "ha", "nuts3": None},
    "PT": {"id": "OSA_ID", "id_kind": "parcel", "declared_area": "OSA_AREA",
           "declared_area_unit": "m2", "nuts3": None},
}
#: Conversion of the declared-area attribute into hectares.
AREA_UNIT_TO_HA = {"ha": 1.0, "m2": 1e-4}

sns.set_theme(context="notebook", style="whitegrid")


def normalise_id(values) -> pd.Series:
    """Identifiers as strings, without the float formatting of the source layer.

    Portugal stores ``OSA_ID`` as a float, so a plain string cast yields ``"228.0"`` where
    EuroCropsML writes ``228``, and the two releases then fail to join although they carry
    the same parcels. Numeric identifiers are therefore cast through a nullable integer
    first, and only genuinely non-numeric identifiers are kept as text.
    """
    s = pd.Series(values).reset_index(drop=True)
    num = pd.to_numeric(s, errors="coerce")
    if num.notna().all() and float(np.nanmax(np.abs(num.to_numpy() % 1))) == 0.0:
        return num.astype("int64").astype(str)
    return s.astype(str).str.strip()


# --------------------------------------------------------------------- per-parcel pass


def parcel_metrics(cc: str, force: bool = False) -> pd.DataFrame:
    """Per-parcel geometric metrics for one country, cached under data/eurocrops/derived."""
    import geopandas as gpd

    out = DERIVED_DIR / f"{cc}_{YEAR}_metrics.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)

    src = PARQUET_DIR / f"{cc}_{YEAR}.parquet"
    cols = ["EC_hcat_c", "EC_hcat_n", "EC_trans_n", "geometry"]
    for key in ("id", "declared_area", "nuts3"):
        col = NATIVE[cc][key]
        if col:
            cols.append(col)
    gdf = gpd.read_parquet(src, columns=cols)
    print(f"  {cc}: {len(gdf):,} parcels read, reprojecting {gdf.crs.name} -> {EQUAL_AREA}", flush=True)
    gdf = gdf.to_crs(EQUAL_AREA)

    geom = gdf.geometry
    area = geom.area.to_numpy()
    perim = geom.length.to_numpy()
    centroid = geom.centroid
    with np.errstate(divide="ignore", invalid="ignore"):
        # Polsby-Popper compactness, 1 for a circle and smaller for a ragged parcel.
        compactness = 4.0 * np.pi * area / np.square(perim)
        # Share of the parcel lying within one pixel of its own boundary, the first-order
        # approximation perimeter x width / area, validated against true erosion below.
        edge_share = np.clip(perim * PIXEL_M / area, 0.0, 1.0)

    df = pd.DataFrame({
        "country": COUNTRIES[cc],
        "cc": cc,
        "hcat": gdf["EC_hcat_c"].astype(str).str.strip(),
        "hcat_name": gdf["EC_hcat_n"].astype(str),
        "trans_name": gdf["EC_trans_n"].astype(str),
        "area_m2": area,
        "perimeter_m": perim,
        "compactness": compactness,
        "edge_share": edge_share,
        "n_pixels": area / (PIXEL_M ** 2),
        "interior_pixels": np.maximum(area - perim * PIXEL_M, 0.0) / (PIXEL_M ** 2),
        "cx": centroid.x.to_numpy(),
        "cy": centroid.y.to_numpy(),
        "is_valid": geom.is_valid.to_numpy(),
        "geom_type": geom.geom_type.to_numpy(),
    })
    id_col, area_col = NATIVE[cc]["id"], NATIVE[cc]["declared_area"]
    df["parcel_id"] = normalise_id(gdf[id_col]).to_numpy() if id_col else ""
    if area_col:
        declared = pd.to_numeric(gdf[area_col], errors="coerce").to_numpy()
        df["declared_area_ha"] = declared * AREA_UNIT_TO_HA[NATIVE[cc]["declared_area_unit"]]
    else:
        df["declared_area_ha"] = np.nan
    nuts_col = NATIVE[cc]["nuts3"]
    df["nuts3"] = gdf[nuts_col].astype(str).to_numpy() if nuts_col else ""

    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, compression="zstd", index=False)
    print(f"  {cc}: metrics written to {out.relative_to(REPO)}", flush=True)
    return df


def validate_edge_approximation(cc: str, n: int = 5000) -> dict:
    """Check the perimeter approximation of the edge share against true erosion."""
    import geopandas as gpd

    gdf = gpd.read_parquet(PARQUET_DIR / f"{cc}_{YEAR}.parquet", columns=["geometry"])
    gdf = gdf.sample(n=min(n, len(gdf)), random_state=SEED).to_crs(EQUAL_AREA)
    geom = gdf.geometry.buffer(0)  # repair the handful of invalid rings before eroding
    area = geom.area.to_numpy()
    eroded = geom.buffer(-PIXEL_M).area.to_numpy()
    true_share = np.clip((area - eroded) / area, 0, 1)
    approx = np.clip(geom.length.to_numpy() * PIXEL_M / area, 0, 1)
    ok = np.isfinite(true_share) & np.isfinite(approx) & (area > 0)
    return {
        "country": COUNTRIES[cc],
        "n_sampled": int(ok.sum()),
        "median_true_edge_share": float(np.median(true_share[ok])),
        "median_approx_edge_share": float(np.median(approx[ok])),
        "median_ratio_approx_over_true": float(np.median(approx[ok] / np.maximum(true_share[ok], 1e-9))),
        "pearson_r": float(np.corrcoef(true_share[ok], approx[ok])[0, 1]),
    }


# -------------------------------------------------------------------------- aggregates


def quantile_table(df: pd.DataFrame, col: str, qs=(0.05, 0.25, 0.5, 0.75, 0.95)) -> pd.DataFrame:
    rows = []
    for country, g in df.groupby("country"):
        row = {"country": country, "n": len(g), "mean": g[col].mean()}
        row.update({f"p{int(q * 100):02d}": g[col].quantile(q) for q in qs})
        rows.append(row)
    return pd.DataFrame(rows).set_index("country").reindex([c for c in ORDER if c in df["country"].values])


def class_inventory(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (country, HCAT class): parcel count, area and pixel share."""
    g = df.groupby(["country", "hcat", "hcat_name"], observed=True).agg(
        parcels=("area_m2", "size"),
        area_ha=("area_m2", lambda s: s.sum() / 1e4),
        median_area_ha=("area_m2", lambda s: s.median() / 1e4),
        median_pixels=("n_pixels", "median"),
        median_edge_share=("edge_share", "median"),
    ).reset_index()
    totals = g.groupby("country")[["parcels", "area_ha"]].transform("sum")
    g["parcel_share_pct"] = 100 * g["parcels"] / totals["parcels"]
    g["area_share_pct"] = 100 * g["area_ha"] / totals["area_ha"]
    g["level"] = g["hcat"].map(_hcat.hcat_level)
    return g.sort_values(["country", "parcels"], ascending=[True, False])


def chip_occupancy(df: pd.DataFrame) -> pd.DataFrame:
    """Parcels and declared area per 224 x 224 chip, from parcel centroids."""
    ix = np.floor(df["cx"].to_numpy() / CHIP_M).astype(np.int64)
    iy = np.floor(df["cy"].to_numpy() / CHIP_M).astype(np.int64)
    tmp = pd.DataFrame({"country": df["country"].to_numpy(), "ix": ix, "iy": iy,
                        "area_m2": df["area_m2"].to_numpy()})
    return tmp.groupby(["country", "ix", "iy"], observed=True).agg(
        parcels=("area_m2", "size"), area_m2=("area_m2", "sum")).reset_index()


def lorenz(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Lorenz curve and Gini coefficient of a class-size distribution."""
    v = np.sort(np.asarray(values, dtype=float))
    cum = np.cumsum(v) / v.sum()
    x = np.arange(1, len(v) + 1) / len(v)
    gini = float(1 - 2 * np.trapezoid(cum, x)) if hasattr(np, "trapezoid") else float(1 - 2 * np.trapz(cum, x))
    return x, cum, gini


# ----------------------------------------------------------------------------- figures


def fig01_inventory(df: pd.DataFrame, inv: pd.DataFrame, findings: dict) -> None:
    per = df.groupby("country").agg(parcels=("area_m2", "size"), area_ha=("area_m2", lambda s: s.sum() / 1e4))
    per = per.reindex(ORDER)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, col, title, fmt in (
        (axes[0], "parcels", "parcels declared", "{:,.0f}"),
        (axes[1], "area_ha", "declared area (ha)", "{:,.0f}"),
    ):
        ax.bar(per.index, per[col], color=[PALETTE[c] for c in per.index])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        for i, v in enumerate(per[col]):
            ax.text(i, v, fmt.format(v), ha="center", va="bottom", fontsize=8)
        ax.set_ylim(0, per[col].max() * 1.18)
    nclass = inv.groupby("country")["hcat"].nunique().reindex(ORDER)
    axes[2].bar(nclass.index, nclass.values, color=[PALETTE[c] for c in nclass.index])
    axes[2].set_title("distinct HCAT classes")
    axes[2].tick_params(axis="x", rotation=20)
    for i, v in enumerate(nclass.values):
        axes[2].text(i, v, f"{v}", ha="center", va="bottom", fontsize=8)
    fig.suptitle("EuroCrops 2021, the four downloaded countries", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "01_inventory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["inventory"] = {
        c: {"parcels": int(per.at[c, "parcels"]), "declared_area_ha": round(float(per.at[c, "area_ha"]), 1),
            "n_hcat_classes": int(nclass[c])} for c in per.index}


def fig02_area_ecdf(df: pd.DataFrame, findings: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for country, g in df.groupby("country"):
        v = np.sort(g["n_pixels"].to_numpy())
        y = np.arange(1, len(v) + 1) / len(v)
        step = max(1, len(v) // 20000)
        axes[0].plot(v[::step], y[::step], label=country, color=PALETTE[country], lw=1.8)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("labelled 10 m pixels in one parcel")
    axes[0].set_ylabel("share of parcels at or below")
    axes[0].set_title("how many pixels does one polygon buy?")
    for thr, style in ((100, ":"), (1000, "--")):
        axes[0].axvline(thr, color="grey", ls=style, lw=1)
    axes[0].legend(title=None, fontsize=8)

    med = df.groupby("country")["n_pixels"].median().reindex(ORDER)
    axes[1].bar(med.index, med.values, color=[PALETTE[c] for c in med.index])
    axes[1].set_title("median pixels per parcel")
    axes[1].tick_params(axis="x", rotation=20)
    for i, v in enumerate(med.values):
        axes[1].text(i, v, f"{v:,.0f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "02_pixels_per_parcel.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    tbl = quantile_table(df, "n_pixels")
    findings["pixels_per_parcel"] = json.loads(tbl.round(1).to_json(orient="index"))
    findings["parcels_below_pixel_thresholds_pct"] = {
        country: {f"lt_{t}px": round(float((g["n_pixels"] < t).mean() * 100), 2)
                  for t in (10, 100, 1000)}
        for country, g in df.groupby("country")}


def fig03_k_budget(df: pd.DataFrame, findings: dict) -> None:
    """What a label budget of K polygons per class delivers in labelled pixels."""
    ks = [1, 5, 10, 20, 50, 100, 200, 500]
    rows = []
    for country, g in df.groupby("country"):
        med = float(g["n_pixels"].median())
        interior = float(g["interior_pixels"].median())
        for k in ks:
            rows.append({"country": country, "K": k, "pixels": k * med, "interior_pixels": k * interior})
    tbl = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for country, g in tbl.groupby("country"):
        ax.plot(g["K"], g["pixels"], marker="o", ms=4, color=PALETTE[country], label=country)
        ax.plot(g["K"], g["interior_pixels"], marker="", ls="--", lw=1, color=PALETTE[country], alpha=0.6)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("K, labelled polygons per class")
    ax.set_ylabel("labelled 10 m pixels per class")
    ax.set_title("label budget in polygons against supervision in pixels\n"
                 "(dashed: pixels at least one pixel inside the boundary)", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "03_k_budget_in_pixels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["k_budget_pixels"] = json.loads(
        tbl.pivot(index="K", columns="country", values="pixels").round(0).to_json(orient="index"))


def fig04_edge(df: pd.DataFrame, findings: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    sample = df.sample(n=min(200_000, len(df)), random_state=SEED)
    for country, g in sample.groupby("country"):
        axes[0].scatter(g["n_pixels"], g["edge_share"], s=1, alpha=0.05,
                        color=PALETTE[country], rasterized=True, label=country)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("parcel size (10 m pixels)")
    axes[0].set_ylabel("share of parcel within one pixel of the boundary")
    axes[0].set_title("mixed-pixel burden against parcel size")
    leg = axes[0].legend(fontsize=8, markerscale=8)
    for h in leg.legend_handles:
        h.set_alpha(1)

    order = [c for c in ORDER if c in df["country"].unique()]
    sns.boxplot(data=df, x="country", y="edge_share", order=order, hue="country",
                palette=PALETTE, legend=False, showfliers=False, ax=axes[1])
    axes[1].set_title("distribution of the boundary share")
    axes[1].set_xlabel("")
    axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "04_edge_share.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["edge_share"] = json.loads(quantile_table(df, "edge_share").round(3).to_json(orient="index"))
    findings["compactness"] = json.loads(quantile_table(df, "compactness").round(3).to_json(orient="index"))


def fig05_class_prior(inv: pd.DataFrame, findings: dict) -> None:
    """The class prior by parcel count against the class prior by area, which is what a
    per-pixel loss sees."""
    countries = [c for c in ORDER if c in inv["country"].unique()]
    fig, axes = plt.subplots(len(countries), 1, figsize=(11, 3.1 * len(countries)), squeeze=False)
    shifts = {}
    for ax, country in zip(axes[:, 0], countries):
        g = inv[inv["country"] == country].nlargest(15, "parcels").copy()
        g["label"] = g["hcat_name"].str.replace("_", " ").str.slice(0, 34)
        x = np.arange(len(g))
        ax.bar(x - 0.2, g["parcel_share_pct"], width=0.4, label="by parcel count", color=PALETTE[country])
        ax.bar(x + 0.2, g["area_share_pct"], width=0.4, label="by area", color=PALETTE[country], alpha=0.45)
        ax.set_xticks(x)
        ax.set_xticklabels(g["label"], rotation=35, ha="right", fontsize=7)
        ax.set_ylabel("share (%)")
        ax.set_title(f"{country}: top 15 classes by parcel count", fontsize=10)
        if ax is axes[0, 0]:
            ax.legend(fontsize=8)
        g["ratio"] = g["area_share_pct"] / g["parcel_share_pct"]
        shifts[country] = {r.hcat_name: round(float(r.ratio), 2) for r in g.itertuples()}
    fig.tight_layout()
    fig.savefig(FIG_DIR / "05_class_prior_count_vs_area.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["area_over_count_share_ratio"] = shifts


def fig06_concentration(inv: pd.DataFrame, findings: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ginis = {}
    for country, g in inv.groupby("country"):
        for ax, col, tag in ((axes[0], "parcels", "count"), (axes[1], "area_ha", "area")):
            x, cum, gini = lorenz(g[col].to_numpy())
            ax.plot(x, cum, color=PALETTE[country], lw=1.8,
                    label=f"{country} (G={gini:.2f})")
            ginis.setdefault(country, {})[f"gini_{tag}"] = round(gini, 3)
    for ax, title in ((axes[0], "by parcel count"), (axes[1], "by declared area")):
        ax.plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
        ax.set_xlabel("cumulative share of classes")
        ax.set_ylabel("cumulative share of labels")
        ax.set_title(f"class concentration, {title}")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "06_class_concentration.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["class_concentration"] = ginis
    findings["class_coverage_pct"] = {
        country: {f"top{n}": round(float(g.nlargest(n, "parcels")["parcel_share_pct"].sum()), 1)
                  for n in (1, 5, 10, 20)}
        for country, g in inv.groupby("country")}


def fig07_overlap(inv: pd.DataFrame, findings: dict) -> None:
    """Cross-country class overlap at the native HCAT level and rolled up to level 4."""
    countries = [c for c in ORDER if c in inv["country"].unique()]
    inv = inv.copy()
    inv["hcat_l4"] = inv["hcat"].map(lambda c: _hcat.roll_up(c, 4))
    mats = {}
    for tag, col in (("native", "hcat"), ("level4", "hcat_l4")):
        sets = {c: set(inv.loc[inv["country"] == c, col]) for c in countries}
        m = pd.DataFrame(0, index=countries, columns=countries, dtype=int)
        for a in countries:
            for b in countries:
                m.loc[a, b] = len(sets[a] & sets[b])
        mats[tag] = m
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    for ax, (tag, m) in zip(axes, mats.items()):
        sns.heatmap(m, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                    annot_kws={"size": 9})
        ax.set_title(f"shared classes, {tag} HCAT")
        ax.tick_params(axis="x", rotation=25)
        ax.tick_params(axis="y", rotation=0)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "07_class_overlap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["class_overlap_native"] = json.loads(mats["native"].to_json(orient="index"))
    findings["class_overlap_level4"] = json.loads(mats["level4"].to_json(orient="index"))
    shared_all = set.intersection(*[set(inv.loc[inv["country"] == c, "hcat"]) for c in countries])
    findings["classes_shared_by_all_four"] = sorted(shared_all)
    findings["n_classes_shared_by_all_four"] = len(shared_all)


def fig08_maps(df: pd.DataFrame, findings: dict) -> None:
    countries = [c for c in ORDER if c in df["country"].unique()]
    fig, axes = plt.subplots(1, len(countries), figsize=(4.0 * len(countries), 4.4))
    for ax, country in zip(np.atleast_1d(axes), countries):
        g = df[df["country"] == country]
        hb = ax.hexbin(g["cx"] / 1000, g["cy"] / 1000, gridsize=90, bins="log", cmap="viridis", mincnt=1)
        ax.set_title(f"{country}\n{len(g):,} parcels", fontsize=10)
        ax.set_aspect("equal")
        ax.set_xlabel("ETRS89-LAEA easting (km)", fontsize=8)
        ax.grid(False)
        fig.colorbar(hb, ax=ax, shrink=0.75, label="parcels per cell (log)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "08_parcel_density.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["bounding_box_laea_km"] = {
        country: {"x": [round(float(g["cx"].min() / 1000), 1), round(float(g["cx"].max() / 1000), 1)],
                  "y": [round(float(g["cy"].min() / 1000), 1), round(float(g["cy"].max() / 1000), 1)]}
        for country, g in df.groupby("country")}


def fig09_chips(chips: pd.DataFrame, findings: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for country, g in chips.groupby("country"):
        v = np.sort(g["parcels"].to_numpy())[::-1]
        axes[0].plot(np.arange(1, len(v) + 1), np.cumsum(v) / v.sum(),
                     color=PALETTE[country], lw=1.8, label=f"{country} ({len(v):,} chips)")
        axes[1].hist(g["parcels"], bins=np.logspace(0, 4, 40), histtype="step",
                     color=PALETTE[country], lw=1.6, label=country)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("chips, ranked by parcel count")
    axes[0].set_ylabel("cumulative share of parcels covered")
    axes[0].set_title(f"how few {CHIP_PX} x {CHIP_PX} chips cover the labels?")
    axes[0].axhline(0.9, color="grey", ls=":", lw=1)
    axes[0].legend(fontsize=8)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("parcels per chip")
    axes[1].set_ylabel("chips")
    axes[1].set_title("parcel density per chip")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "09_chip_occupancy.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    per_chip_gb = CHIP_PX * CHIP_PX * EXPORT_BANDS * EXPORT_T * EXPORT_BYTES / 1e9
    out = {}
    for country, g in chips.groupby("country"):
        v = np.sort(g["parcels"].to_numpy())[::-1]
        cum = np.cumsum(v) / v.sum()
        n90 = int(np.searchsorted(cum, 0.90) + 1)
        out[country] = {
            "chips_occupied": int(len(v)),
            "chips_for_90pct_of_parcels": n90,
            "median_parcels_per_chip": float(np.median(v)),
            "export_gb_all_chips": round(len(v) * per_chip_gb, 1),
            "export_gb_90pct": round(n90 * per_chip_gb, 1),
        }
    findings["chip_occupancy"] = out
    findings["chip_assumptions"] = {
        "chip_px": CHIP_PX, "pixel_m": PIXEL_M, "chip_m": CHIP_M, "crs": EQUAL_AREA,
        "bands": EXPORT_BANDS, "timesteps": EXPORT_T, "bytes_per_value": EXPORT_BYTES,
        "gb_per_chip": round(per_chip_gb, 4),
        "note": "chips are counted from parcel centroids on a fixed grid, so a parcel "
                "straddling a chip edge is counted once",
    }


def fig10_geometry_health(df: pd.DataFrame, findings: dict) -> None:
    rows = []
    for country, g in df.groupby("country"):
        cc = {v: k for k, v in COUNTRIES.items()}[country]
        dup = int(g.duplicated(subset=["parcel_id"]).sum()) if (g["parcel_id"] != "").any() else -1
        rows.append({
            "country": country,
            "id_column": NATIVE[cc]["id"],
            "id_kind": NATIVE[cc]["id_kind"],
            "invalid_geometries": int((~g["is_valid"]).sum()),
            "multipart": int((g["geom_type"] == "MultiPolygon").sum()),
            "duplicate_parcel_ids": dup,
            "area_below_1_pixel": int((g["n_pixels"] < 1).sum()),
            "area_below_10_pixels": int((g["n_pixels"] < 10).sum()),
            "no_interior_pixel": int((g["interior_pixels"] < 1).sum()),
        })
    tbl = pd.DataFrame(rows).set_index("country").reindex([c for c in ORDER if c in df["country"].values])
    fig, ax = plt.subplots(figsize=(9, 3.2))
    plot = tbl.drop(columns=["duplicate_parcel_ids", "id_column", "id_kind"]).clip(lower=0.7)
    plot.plot(kind="bar", ax=ax, width=0.8, logy=True, colormap="tab20c")
    ax.set_ylabel("parcels (log)")
    ax.set_title("geometry health and parcels too small to supervise")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "10_geometry_health.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["geometry_health"] = json.loads(tbl.to_json(orient="index"))


def fig11_declared_area(df: pd.DataFrame, findings: dict) -> None:
    """The national declared-area attribute against the area computed from the geometry."""
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    out = {}
    for country, g in df.groupby("country"):
        m = g["declared_area_ha"].notna() & (g["declared_area_ha"] > 0) & (g["area_m2"] > 0)
        if m.sum() == 0:
            continue
        ratio = (g.loc[m, "area_m2"] / 1e4) / g.loc[m, "declared_area_ha"]
        ax.hist(np.clip(ratio, 0, 2), bins=120, histtype="step", lw=1.6,
                color=PALETTE[country], label=f"{country} (median {ratio.median():.3f})")
        cc = {v: k for k, v in COUNTRIES.items()}[country]
        out[country] = {
            "attribute": NATIVE[cc]["declared_area"],
            "native_unit": NATIVE[cc]["declared_area_unit"],
            "n_with_declared_area": int(m.sum()),
            "median_ratio_computed_ha_over_declared": round(float(ratio.median()), 4),
            "share_within_5pct": round(float(((ratio - 1).abs() < 0.05).mean() * 100), 1),
            "share_below_half": round(float((ratio < 0.5).mean() * 100), 2),
        }
    ax.axvline(1.0, color="grey", ls=":", lw=1)
    ax.set_xlabel("computed area (ha) / declared area attribute")
    ax.set_ylabel("parcels")
    ax.set_title("does the geometry agree with the declared area attribute?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "11_declared_vs_computed_area.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["declared_area_consistency"] = out


def fig12_class_sizes(inv: pd.DataFrame, df: pd.DataFrame, findings: dict) -> None:
    """How many classes survive a minimum parcel count, the constraint on K and on the
    number of classes a country can support."""
    thresholds = [1, 10, 50, 100, 200, 500, 1000, 5000]
    rows = []
    for country, g in inv.groupby("country"):
        for t in thresholds:
            rows.append({"country": country, "min_parcels": t, "classes": int((g["parcels"] >= t).sum())})
    surv = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    for country, g in surv.groupby("country"):
        axes[0].plot(g["min_parcels"], g["classes"], marker="o", ms=4,
                     color=PALETTE[country], label=country)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("minimum parcels per class")
    axes[0].set_ylabel("classes retained")
    axes[0].set_title("class survival under a minimum size")
    axes[0].legend(fontsize=8)

    for country, g in inv.groupby("country"):
        v = np.sort(g["median_pixels"].to_numpy())
        axes[1].plot(v, np.arange(1, len(v) + 1) / len(v), color=PALETTE[country], lw=1.8, label=country)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("median parcel size of a class (10 m pixels)")
    axes[1].set_ylabel("share of classes at or below")
    axes[1].set_title("typical parcel size varies by class")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "12_class_survival.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["class_survival"] = json.loads(
        surv.pivot(index="min_parcels", columns="country", values="classes").to_json(orient="index"))


# -------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", nargs="+", default=list(COUNTRIES), choices=list(COUNTRIES))
    parser.add_argument("--force", action="store_true", help="recompute the per-parcel metrics")
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    print("per-parcel metrics", flush=True)
    frames = [parcel_metrics(cc, force=args.force) for cc in args.countries]
    df = pd.concat(frames, ignore_index=True)
    print(f"total {len(df):,} parcels across {df['country'].nunique()} countries", flush=True)

    findings: dict = {
        "source": "EuroCrops v11, Zenodo 14094196, declaration year 2021",
        "countries": {cc: COUNTRIES[cc] for cc in args.countries},
        "n_parcels_total": int(len(df)),
        "is_full_dataset": True,
        "equal_area_crs": EQUAL_AREA,
        "pixel_m": PIXEL_M,
        "random_seed": SEED,
    }

    inv = class_inventory(df)
    inv.to_parquet(CACHE_DIR / "class_inventory.parquet", index=False)
    chips = chip_occupancy(df)
    chips.to_parquet(CACHE_DIR / "chip_occupancy.parquet", index=False)
    quantile_table(df, "n_pixels").to_parquet(CACHE_DIR / "pixels_per_parcel.parquet")

    print("figures", flush=True)
    fig01_inventory(df, inv, findings)
    fig02_area_ecdf(df, findings)
    fig03_k_budget(df, findings)
    fig04_edge(df, findings)
    fig05_class_prior(inv, findings)
    fig06_concentration(inv, findings)
    fig07_overlap(inv, findings)
    fig08_maps(df, findings)
    fig09_chips(chips, findings)
    fig10_geometry_health(df, findings)
    fig11_declared_area(df, findings)
    fig12_class_sizes(inv, df, findings)

    print("validating the boundary approximation", flush=True)
    findings["edge_approximation_check"] = [validate_edge_approximation(cc) for cc in args.countries]

    OUT_FINDINGS.write_text(json.dumps(findings, indent=2))
    print(f"wrote {OUT_FINDINGS.relative_to(REPO)} and {len(list(FIG_DIR.glob('*.png')))} figures", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
