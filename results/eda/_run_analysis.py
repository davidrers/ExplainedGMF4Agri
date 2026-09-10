"""End-to-end EuroCropsML analysis. Outputs:
- catalogue_parcels.parquet (parcel-level metadata)
- figures/*.png (all charts)
- findings.json (numerical summary the notebook will quote in markdown)

Run after preprocess.zip has been downloaded + extracted under DATA_ROOT/preprocess.
"""
from __future__ import annotations

import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sns.set_theme(context="notebook", style="whitegrid")
RNG = np.random.default_rng(42)
random.seed(42)

# ---------- paths ----------
DATA_ROOT = Path(os.environ.get("EUROCROPSML_DATA", Path.home() / "eurocropsml_data")).expanduser()
HERE = Path(__file__).parent.resolve()
FIG_DIR = HERE / "figures"
FIG_DIR.mkdir(exist_ok=True)
OUT_PARQUET = HERE / "catalogue_parcels.parquet"
OUT_FINDINGS = HERE / "findings.json"

S2_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]
RED_IDX, NIR_IDX = S2_BANDS.index("B04"), S2_BANDS.index("B08")

# NUTS3 prefix -> country (Eurostat standard)
NUTS_PREFIX = {"EE": "Estonia", "LV": "Latvia", "PT": "Portugal"}

# Filename format: {NUTS3}_{parcel_id}_{label}.npz
FNAME_RE = re.compile(r"^(?P<nuts>[A-Z]{2}[A-Z0-9]+)_(?P<pid>\d+)_(?P<cls>\d+)\.npz$")


def find_npz_dir(root: Path) -> Path:
    cand = list(root.rglob("S2/2021"))
    if cand:
        return cand[0]
    npzs = list(root.rglob("*.npz"))
    if npzs:
        return npzs[0].parent
    raise FileNotFoundError(f"No .npz files under {root}")


def main(sample: int | None = None):
    npz_dir = find_npz_dir(DATA_ROOT)
    print(f"npz dir: {npz_dir}")

    all_npz = sorted(npz_dir.glob("*.npz"))
    print(f"found {len(all_npz):,} .npz files")
    if not all_npz:
        raise SystemExit("no files")

    # ---------- inspect one file ----------
    npz = np.load(all_npz[0])
    print("npz keys:", list(npz.files))
    for k in npz.files:
        a = npz[k]
        print(f"  {k:8s} shape={a.shape} dtype={a.dtype}")

    # ---------- catalogue: parse filenames + cheap metadata ----------
    files = all_npz if sample is None or len(all_npz) <= sample else random.sample(all_npz, sample)
    rows = []
    skipped = 0
    for p in tqdm(files, desc="catalogue"):
        m = FNAME_RE.match(p.name)
        if not m:
            skipped += 1
            continue
        nuts = m.group("nuts")
        country = NUTS_PREFIX.get(nuts[:2], "Unknown")
        try:
            with np.load(p) as z:
                n_t = int(z["dates"].shape[0])
                center = z["center"]
                arr = np.atleast_1d(center).ravel()
                lon = float(arr[0]) if arr.size >= 1 else np.nan
                lat = float(arr[1]) if arr.size >= 2 else np.nan
                # File size in KB as a proxy for total data volume
        except Exception as e:
            n_t, lat, lon = 0, np.nan, np.nan
        rows.append(dict(path=str(p), country=country, nuts=nuts, parcel_id=m.group("pid"),
                          hcat=m.group("cls"), n_timesteps=n_t, lat=lat, lon=lon))
    df = pd.DataFrame(rows)
    print(f"catalogued {len(df):,} parcels (skipped {skipped})")
    df.to_parquet(OUT_PARQUET, index=False)

    # If centers came as (lon, lat) instead of (lat, lon), swap.
    if df["lat"].dropna().between(-90, 90).mean() < 0.9 and df["lon"].dropna().between(-180, 180).mean() > 0.9:
        df = df.rename(columns={"lat": "lon", "lon": "lat"})
        print("swapped lat/lon (raw order was lon,lat)")
    # If centers are projected meters (not degrees), flag and don't plot geographically
    centers_look_geographic = (
        df["lat"].dropna().between(-90, 90).mean() > 0.95
        and df["lon"].dropna().between(-180, 180).mean() > 0.95
    )
    print(f"centers look like degrees: {centers_look_geographic}")

    findings: dict = {"n_parcels_total": len(df), "centers_geographic": centers_look_geographic}

    # ---------- 4.1 parcel counts per country ----------
    cc = df["country"].value_counts()
    print("\nparcels per country:\n", cc)
    findings["parcels_per_country"] = cc.to_dict()
    fig, ax = plt.subplots(figsize=(5, 3))
    cc.plot(kind="bar", color="steelblue", ax=ax)
    ax.set_ylabel("parcels"); ax.set_title("Parcels per country")
    plt.xticks(rotation=0); plt.tight_layout()
    fig.savefig(FIG_DIR / "01_parcels_per_country.png", dpi=130); plt.close(fig)

    # ---------- 4.2 class distribution ----------
    cls_counts = df["hcat"].value_counts()
    findings["n_distinct_classes"] = int(cls_counts.size)
    findings["top10_classes"] = cls_counts.head(10).to_dict()
    print(f"\ndistinct classes: {cls_counts.size}")
    print("top-10:\n", cls_counts.head(10))

    fig, ax = plt.subplots(figsize=(11, 5))
    cls_counts.head(30).plot(kind="bar", ax=ax, color="darkgreen")
    ax.set_title("Top-30 HCAT classes (global)"); ax.set_ylabel("parcels"); ax.set_xlabel("HCAT class id")
    plt.tight_layout(); fig.savefig(FIG_DIR / "02_top30_classes.png", dpi=130); plt.close(fig)

    # top-10 per country
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    per_country_top = {}
    for ax, c in zip(axes, ["Estonia", "Latvia", "Portugal"]):
        s = df.loc[df.country == c, "hcat"].value_counts().head(10)
        per_country_top[c] = s.to_dict()
        s.plot(kind="bar", ax=ax, color="steelblue")
        ax.set_title(c); ax.set_xlabel("HCAT id"); ax.set_ylabel("parcels")
    plt.tight_layout(); fig.savefig(FIG_DIR / "03_top10_per_country.png", dpi=130); plt.close(fig)
    findings["per_country_top10"] = per_country_top

    # Lorenz / cumulative coverage
    sorted_counts = cls_counts.sort_values(ascending=False).values
    cum = np.cumsum(sorted_counts) / sorted_counts.sum()
    frac = np.arange(1, len(cum) + 1) / len(cum)
    coverage = {}
    for q in (0.5, 0.8, 0.95):
        k = int(np.searchsorted(cum, q)) + 1
        coverage[f"{int(q*100)}pct_covered_by_top_k"] = k
    findings["class_coverage"] = coverage
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(frac, cum, lw=2); ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.set_xlabel("fraction of classes (sorted desc)"); ax.set_ylabel("cum. fraction of parcels")
    ax.set_title("How concentrated is the label distribution?")
    for q in (0.5, 0.8, 0.95):
        k = int(np.searchsorted(cum, q)) + 1
        ax.axhline(q, color="red", alpha=0.2)
        ax.annotate(f"{int(q*100)}% by top-{k}", xy=(k/len(cum), q), xytext=(0.4, q-0.04), fontsize=9)
    plt.tight_layout(); fig.savefig(FIG_DIR / "04_lorenz_classes.png", dpi=130); plt.close(fig)

    # ---------- 4.3 country × top-class overlap heatmap ----------
    top_classes = cls_counts.head(20).index
    pivot = (df[df.hcat.isin(top_classes)]
             .groupby(["hcat", "country"]).size().unstack(fill_value=0).reindex(top_classes))
    fig, ax = plt.subplots(figsize=(6, 8))
    sns.heatmap(pivot, annot=True, fmt="d", cmap="YlGnBu", ax=ax)
    ax.set_title("Top-20 classes × country (parcel count)")
    plt.tight_layout(); fig.savefig(FIG_DIR / "05_class_country_heatmap.png", dpi=130); plt.close(fig)
    # Shared classes (any presence) and dominant
    findings["top20_pivot"] = {str(k): {c: int(v) for c, v in row.items()} for k, row in pivot.to_dict("index").items()}
    findings["classes_present_in_all_three"] = int((pivot > 0).all(axis=1).sum())

    # ---------- 4.4 NUTS-3 coverage ----------
    nuts_summary = {}
    for c in ["Estonia", "Latvia", "Portugal"]:
        s = df[df.country == c]["nuts"].value_counts()
        nuts_summary[c] = {"n_nuts3_regions": int(s.size), "top3": s.head(3).to_dict()}
        if s.empty: continue
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.bar(s.index, s.values, color="slategray")
        ax.set_title(f"{c} — parcels per NUTS-3 region")
        plt.xticks(rotation=75); plt.tight_layout()
        fig.savefig(FIG_DIR / f"06_nuts_{c.lower()}.png", dpi=130); plt.close(fig)
    findings["nuts_summary"] = nuts_summary

    # ---------- 4.5 timestep distribution ----------
    fig, ax = plt.subplots(figsize=(8, 4))
    ts_stats = {}
    for c, color in zip(["Estonia", "Latvia", "Portugal"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
        sub = df.loc[df.country == c, "n_timesteps"].dropna()
        if sub.empty: continue
        ax.hist(sub, bins=40, alpha=0.5, label=c, color=color)
        ts_stats[c] = {"mean": float(sub.mean()), "median": float(sub.median()),
                       "min": int(sub.min()), "max": int(sub.max()), "std": float(sub.std())}
    ax.set_xlabel("# observations / parcel"); ax.set_ylabel("parcels"); ax.legend()
    ax.set_title("Time-series length per country (after cloud filter)")
    plt.tight_layout(); fig.savefig(FIG_DIR / "07_timestep_hist.png", dpi=130); plt.close(fig)
    findings["timestep_stats"] = ts_stats

    # ---------- 4.6 spatial scatter ----------
    if centers_look_geographic:
        geo = df.dropna(subset=["lat", "lon"]).sample(min(20_000, len(df)), random_state=0)
        fig, ax = plt.subplots(figsize=(8, 6))
        for c, color in zip(["Estonia", "Latvia", "Portugal"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            sub = geo[geo.country == c]
            if sub.empty: continue
            ax.scatter(sub.lon, sub.lat, s=2, alpha=0.4, label=c, color=color)
        ax.set_xlabel("lon"); ax.set_ylabel("lat"); ax.legend()
        ax.set_title("Parcel centroids (sampled)")
        plt.tight_layout(); fig.savefig(FIG_DIR / "08_centroids.png", dpi=130); plt.close(fig)
    else:
        # Centers are projected meters per country — plot per-country scatter in native units
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        for ax, c, color in zip(axes, ["Estonia", "Latvia", "Portugal"], ["#1f77b4", "#ff7f0e", "#2ca02c"]):
            sub = df[df.country == c].dropna(subset=["lat", "lon"]).sample(min(8000, len(df)), random_state=0)
            if sub.empty: continue
            ax.scatter(sub["lon"], sub["lat"], s=2, alpha=0.4, color=color)
            ax.set_title(f"{c} centroids (projected)")
        plt.tight_layout(); fig.savefig(FIG_DIR / "08_centroids.png", dpi=130); plt.close(fig)

    # ---------- 5. Per-parcel time-series plots (Part 2) ----------
    top3 = cls_counts.head(3).index.tolist()
    findings["plotted_classes"] = top3
    for cls in top3:
        fig, axes = plt.subplots(3, 2, figsize=(12, 8), sharex=False)
        fig.suptitle(f"HCAT class {cls} — one example parcel per country", fontsize=12, y=1.02)
        any_plotted = False
        for row, c in enumerate(["Estonia", "Latvia", "Portugal"]):
            cand = df[(df.country == c) & (df.hcat == cls)]
            if cand.empty:
                axes[row, 0].set_visible(False); axes[row, 1].set_visible(False); continue
            pick = cand.sample(1, random_state=0).iloc[0]
            with np.load(pick["path"]) as z:
                data = z["data"]; dates = z["dates"]
            dates = pd.to_datetime(dates)
            for i, name in enumerate(S2_BANDS[: data.shape[1]]):
                axes[row, 0].plot(dates, data[:, i], lw=0.8, label=name)
            axes[row, 0].set_ylabel(f"{c}\nreflectance"); axes[row, 0].set_title(Path(pick["path"]).name, fontsize=8)
            axes[row, 0].tick_params(axis="x", labelrotation=45)
            red, nir = data[:, RED_IDX].astype(float), data[:, NIR_IDX].astype(float)
            ndvi = (nir - red) / (nir + red + 1e-9)
            axes[row, 1].plot(dates, ndvi, color="green", lw=1.5)
            axes[row, 1].set_ylim(-0.2, 1.0); axes[row, 1].set_ylabel("NDVI")
            axes[row, 1].tick_params(axis="x", labelrotation=45)
            any_plotted = True
        if any_plotted:
            axes[0, 0].legend(ncol=4, fontsize=7, loc="upper right")
        plt.tight_layout(); fig.savefig(FIG_DIR / f"09_ts_class_{cls}.png", dpi=130); plt.close(fig)

    # Intra-class variance: 3 random parcels of the most common Estonia class
    ee_classes = df[df.country == "Estonia"]["hcat"].value_counts()
    if not ee_classes.empty:
        ee_top = ee_classes.index[0]
        ee_samples = df[(df.country == "Estonia") & (df.hcat == ee_top)].sample(min(3, ee_classes.iloc[0]), random_state=1)
        fig, axes = plt.subplots(len(ee_samples), 2, figsize=(12, 8))
        if len(ee_samples) == 1:
            axes = np.array([axes])
        fig.suptitle(f"Estonia · HCAT {ee_top} — random parcels (intra-class variance)", y=1.02)
        for i, (_, r) in enumerate(ee_samples.iterrows()):
            with np.load(r["path"]) as z:
                data = z["data"]; dates = z["dates"]
            dates = pd.to_datetime(dates)
            for j, name in enumerate(S2_BANDS[: data.shape[1]]):
                axes[i, 0].plot(dates, data[:, j], lw=0.8, label=name)
            axes[i, 0].set_title(Path(r["path"]).name, fontsize=8)
            axes[i, 0].tick_params(axis="x", labelrotation=45)
            red, nir = data[:, RED_IDX].astype(float), data[:, NIR_IDX].astype(float)
            ndvi = (nir - red) / (nir + red + 1e-9)
            axes[i, 1].plot(dates, ndvi, color="green", lw=1.5)
            axes[i, 1].set_ylim(-0.2, 1.0); axes[i, 1].tick_params(axis="x", labelrotation=45)
        plt.tight_layout(); fig.savefig(FIG_DIR / "10_intraclass_variance.png", dpi=130); plt.close(fig)

    OUT_FINDINGS.write_text(json.dumps(findings, indent=2, default=str))
    print(f"\nwrote {OUT_FINDINGS}")
    print(f"figures in {FIG_DIR}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--sample", type=int, default=None)
    args = p.parse_args()
    main(sample=args.sample)
