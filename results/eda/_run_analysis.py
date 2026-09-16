"""General EuroCropsML exploratory analysis, over the full parcel catalogue.

Consumes ``data/catalogue_parcels_full.parquet`` (built by ``_build_catalogue.py``,
one row per parcel for all 706,683 parcels) and regenerates figures 01 to 10 with
human readable HCAT crop names, plus ``findings.json``, the numerical summary the
notebook quotes.

Every number written here is computed over the full catalogue. The only sampled
step is the drawing of individual example parcels for the time-series panels,
which is seeded.

Usage
-----
    python results/eda/_run_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
import _hcat  # noqa: E402

DATA_ROOT = Path(os.environ.get("EUROCROPSML_DATA", REPO / "data" / "eurocropsml")).expanduser()
PREPROCESS_DIR = DATA_ROOT / "preprocess"
CATALOGUE = REPO / "data" / "catalogue_parcels_full.parquet"
FIG_DIR = HERE / "figures"
OUT_FINDINGS = HERE / "findings.json"

SEED = 42
COUNTRIES = ["Estonia", "Latvia", "Portugal"]
PALETTE = {"Estonia": "#1f77b4", "Latvia": "#ff7f0e", "Portugal": "#2ca02c"}
S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A",
            "B09", "B10", "B11", "B12"]
RED_IDX, NIR_IDX = S2_BANDS.index("B04"), S2_BANDS.index("B08")

sns.set_theme(context="notebook", style="whitegrid")


def label(code: str, width: int = 34) -> str:
    """Crop name for a figure axis, truncated and with the HCAT code appended."""
    name = _hcat.name_of(code).replace("_", " ")
    if len(name) > width:
        name = name[: width - 1] + "…"
    return f"{name}\n{code}"


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CATALOGUE.exists():
        raise SystemExit(f"catalogue not found: {CATALOGUE}\n"
                         "run results/eda/_build_catalogue.py first")
    df = pd.read_parquet(CATALOGUE)
    df["hcat"] = df["hcat"].astype(str)
    n_total = len(df)
    print(f"catalogue: {n_total:,} parcels", flush=True)

    findings: dict = {"n_parcels_total": int(n_total),
                      "source": "full catalogue, data/catalogue_parcels_full.parquet",
                      "is_full_dataset": True,
                      "random_seed": SEED}

    # 01 parcels per country
    cc = df["country"].value_counts().reindex(COUNTRIES)
    findings["parcels_per_country"] = {k: int(v) for k, v in cc.items()}
    findings["parcels_per_country_pct"] = {k: round(100 * v / n_total, 2) for k, v in cc.items()}
    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(cc.index, cc.to_numpy(), color=[PALETTE[c] for c in cc.index])
    for i, v in enumerate(cc.to_numpy()):
        ax.annotate(f"{v:,}\n{100 * v / n_total:.1f}%", (i, v), ha="center",
                    va="bottom", fontsize=8)
    ax.set_ylabel("parcels")
    ax.set_title(f"Parcels per country (all {n_total:,})")
    ax.set_ylim(0, cc.max() * 1.2)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "01_parcels_per_country.png", dpi=130)
    plt.close(fig)

    # 02 top-30 classes
    cls = df["hcat"].value_counts()
    findings["n_distinct_classes"] = int(cls.size)
    findings["top10_classes"] = [
        {"hcat": h, "crop_name": _hcat.name_of(h).replace("_", " "), "n": int(v),
         "share_pct": round(100 * v / n_total, 2)} for h, v in cls.head(10).items()]
    top30 = cls.head(30)
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(np.arange(len(top30)), top30.to_numpy()[::-1], color="darkgreen")
    ax.set_yticks(np.arange(len(top30)))
    ax.set_yticklabels([_hcat.name_of(h).replace("_", " ") for h in top30.index][::-1], fontsize=7)
    ax.set_xscale("log")
    ax.set_xlabel("parcels (log scale)")
    ax.set_title(f"Thirty most frequent crop classes, all {n_total:,} parcels")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "02_top30_classes.png", dpi=130)
    plt.close(fig)

    # 03 top-10 per country
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    per_country_top = {}
    for ax, c in zip(axes, COUNTRIES):
        s = df.loc[df.country == c, "hcat"].value_counts().head(10)
        per_country_top[c] = [{"hcat": h, "crop_name": _hcat.name_of(h).replace("_", " "),
                               "n": int(v)} for h, v in s.items()]
        ax.barh(np.arange(len(s)), s.to_numpy()[::-1], color=PALETTE[c])
        ax.set_yticks(np.arange(len(s)))
        ax.set_yticklabels([_hcat.name_of(h).replace("_", " ") for h in s.index][::-1], fontsize=7)
        ax.set_xscale("log")
        ax.set_title(f"{c} ({int(cc[c]):,} parcels)")
        ax.set_xlabel("parcels (log scale)")
    fig.suptitle("Ten most frequent crop classes per country", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "03_top10_per_country.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    findings["per_country_top10"] = per_country_top

    # 04 cumulative class coverage
    sorted_counts = cls.sort_values(ascending=False).to_numpy()
    cum = np.cumsum(sorted_counts) / sorted_counts.sum()
    frac = np.arange(1, len(cum) + 1) / len(cum)
    coverage = {f"top_k_for_{int(q * 100)}pct": int(np.searchsorted(cum, q) + 1)
                for q in (0.5, 0.8, 0.95, 0.99)}
    findings["class_coverage"] = coverage
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frac, cum, lw=2)
    ax.plot([0, 1], [0, 1], "--", color="grey")
    for q in (0.5, 0.8, 0.95):
        k = int(np.searchsorted(cum, q)) + 1
        ax.axhline(q, color="red", alpha=0.25, lw=0.8)
        ax.annotate(f"{int(q * 100)}% of parcels in the top {k} classes",
                    xy=(k / len(cum), q), xytext=(0.3, q - 0.06), fontsize=8)
    ax.set_xlabel("fraction of classes, ordered by frequency")
    ax.set_ylabel("cumulative fraction of parcels")
    ax.set_title(f"Concentration of the label distribution ({cls.size} classes)")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "04_lorenz_classes.png", dpi=130)
    plt.close(fig)

    # 05 class by country heatmap
    top_classes = cls.head(20).index
    pivot = (df[df.hcat.isin(top_classes)].groupby(["hcat", "country"], observed=True).size()
             .unstack("country", fill_value=0).reindex(index=top_classes, columns=COUNTRIES,
                                                       fill_value=0))
    fig, ax = plt.subplots(figsize=(7, 9))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                yticklabels=[_hcat.name_of(h).replace("_", " ") for h in pivot.index],
                cbar_kws={"label": "parcels"})
    ax.set_ylabel("")
    ax.set_title("Twenty most frequent classes by country")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "05_class_country_heatmap.png", dpi=130)
    plt.close(fig)
    findings["top20_pivot"] = {
        h: {"crop_name": _hcat.name_of(h).replace("_", " "),
            **{c: int(pivot.at[h, c]) for c in COUNTRIES}} for h in pivot.index}
    findings["top20_classes_present_in_all_three"] = int((pivot > 0).all(axis=1).sum())

    # 06 NUTS coverage
    nuts_summary = {}
    for c in COUNTRIES:
        s = df[df.country == c]["nuts"].value_counts()
        nuts_summary[c] = {"n_nuts_regions": int(s.size),
                           "top3": {k: int(v) for k, v in s.head(3).items()}}
        fig, ax = plt.subplots(figsize=(8, 3.2))
        ax.bar(s.index, s.to_numpy(), color=PALETTE[c])
        ax.set_title(f"{c}: parcels per NUTS region ({s.size} regions)")
        ax.set_ylabel("parcels")
        plt.xticks(rotation=75, fontsize=7)
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"06_nuts_{c.lower()}.png", dpi=130)
        plt.close(fig)
    findings["nuts_summary"] = nuts_summary

    # 07 time-series length
    fig, ax = plt.subplots(figsize=(8, 4))
    ts_stats = {}
    for c in COUNTRIES:
        sub = df.loc[df.country == c, "n_timesteps"]
        ax.hist(sub.to_numpy(), bins=np.arange(0, 160, 2), alpha=0.55,
                label=c, color=PALETTE[c])
        ts_stats[c] = {"mean": round(float(sub.mean()), 2),
                       "median": int(sub.median()),
                       "min": int(sub.min()), "max": int(sub.max()),
                       "std": round(float(sub.std()), 2)}
    ax.set_yscale("log")
    ax.set_xlabel("cloud-free observations per parcel")
    ax.set_ylabel("parcels (log scale)")
    ax.legend()
    ax.set_title(f"Time-series length per country, all {n_total:,} parcels")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "07_timestep_hist.png", dpi=130)
    plt.close(fig)
    findings["timestep_stats"] = ts_stats

    # 08 centroids
    rng = np.random.default_rng(SEED)
    geo = df.dropna(subset=["lat", "lon"])
    idx = rng.choice(len(geo), size=min(60_000, len(geo)), replace=False)
    geo = geo.iloc[idx]
    fig, ax = plt.subplots(figsize=(8, 6))
    for c in COUNTRIES:
        sub = geo[geo.country == c]
        ax.scatter(sub.lon, sub.lat, s=1.5, alpha=0.35, label=c, color=PALETTE[c])
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
    ax.legend(markerscale=6)
    ax.set_title(f"Parcel centroids, WGS-84 (random sample of {len(geo):,} of {n_total:,})")
    plt.tight_layout()
    fig.savefig(FIG_DIR / "08_centroids.png", dpi=130)
    plt.close(fig)
    findings["centroid_bounds"] = {
        c: {"lon_min": round(float(df.loc[df.country == c, "lon"].min()), 3),
            "lon_max": round(float(df.loc[df.country == c, "lon"].max()), 3),
            "lat_min": round(float(df.loc[df.country == c, "lat"].min()), 3),
            "lat_max": round(float(df.loc[df.country == c, "lat"].max()), 3)}
        for c in COUNTRIES}

    # 09 example parcels for the three most frequent classes
    top3 = cls.head(3).index.tolist()
    findings["plotted_classes"] = [
        {"hcat": h, "crop_name": _hcat.name_of(h).replace("_", " ")} for h in top3]
    for code in top3:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8))
        fig.suptitle(f"{_hcat.name_of(code).replace('_', ' ')} (HCAT {code}): "
                     "one example parcel per country", fontsize=12, y=1.01)
        plotted = False
        for row, c in enumerate(COUNTRIES):
            cand = df[(df.country == c) & (df.hcat == code) & (df.n_timesteps >= 10)]
            if cand.empty:
                axes[row, 0].set_visible(False)
                axes[row, 1].set_visible(False)
                continue
            pick = cand.sample(1, random_state=SEED).iloc[0]
            with np.load(pick["path"]) as z:
                data, dates = z["data"], pd.to_datetime(z["dates"])
            for i, name in enumerate(S2_BANDS[: data.shape[1]]):
                axes[row, 0].plot(dates, data[:, i], lw=0.8, label=name)
            axes[row, 0].set_ylabel(f"{c}\nreflectance")
            axes[row, 0].set_title(Path(pick["path"]).name, fontsize=8)
            axes[row, 0].tick_params(axis="x", labelrotation=45)
            red, nir = data[:, RED_IDX].astype(float), data[:, NIR_IDX].astype(float)
            axes[row, 1].plot(dates, (nir - red) / (nir + red + 1e-9), color="green", lw=1.5)
            axes[row, 1].set_ylim(-0.2, 1.0)
            axes[row, 1].set_ylabel("NDVI")
            axes[row, 1].tick_params(axis="x", labelrotation=45)
            plotted = True
        if plotted:
            axes[0, 0].legend(ncol=4, fontsize=6, loc="upper right")
        plt.tight_layout()
        fig.savefig(FIG_DIR / f"09_ts_class_{code}.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    # 10 intra-class variance
    ee_top = df[df.country == "Estonia"]["hcat"].value_counts().index[0]
    ee = df[(df.country == "Estonia") & (df.hcat == ee_top) & (df.n_timesteps >= 20)]
    ee = ee.sample(min(3, len(ee)), random_state=SEED + 1)
    fig, axes = plt.subplots(len(ee), 2, figsize=(12, 8))
    axes = np.atleast_2d(axes)
    fig.suptitle(f"Estonia, {_hcat.name_of(ee_top).replace('_', ' ')} (HCAT {ee_top}): "
                 "three random parcels", y=1.01)
    for i, (_, r) in enumerate(ee.iterrows()):
        with np.load(r["path"]) as z:
            data, dates = z["data"], pd.to_datetime(z["dates"])
        for j, name in enumerate(S2_BANDS[: data.shape[1]]):
            axes[i, 0].plot(dates, data[:, j], lw=0.8, label=name)
        axes[i, 0].set_title(Path(r["path"]).name, fontsize=8)
        axes[i, 0].tick_params(axis="x", labelrotation=45)
        red, nir = data[:, RED_IDX].astype(float), data[:, NIR_IDX].astype(float)
        axes[i, 1].plot(dates, (nir - red) / (nir + red + 1e-9), color="green", lw=1.5)
        axes[i, 1].set_ylim(-0.2, 1.0)
        axes[i, 1].tick_params(axis="x", labelrotation=45)
    plt.tight_layout()
    fig.savefig(FIG_DIR / "10_intraclass_variance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    OUT_FINDINGS.write_text(json.dumps(findings, indent=2, default=str), encoding="utf-8")
    print(f"wrote {OUT_FINDINGS}; figures in {FIG_DIR}", flush=True)


if __name__ == "__main__":
    main()
