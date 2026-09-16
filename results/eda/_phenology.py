"""NDVI phenology of the retained classes, per country.

Reads the band matrices of a stratified sample of parcels drawn from the retained
class list produced by ``_class_design.py``, computes NDVI, bins it onto a
ten-day grid and caches the per-class mean and standard deviation. A coarse
separability statistic is also computed, namely the ratio of between-class to
within-class variance of the binned NDVI profile (a one-way analysis of variance
F-like statistic), which indicates whether the retained classes are distinguishable
from the vegetation index alone before any foundation model is involved.

Outputs
-------
    results/eda/cache/phenology_profiles.parquet
    results/eda/cache/phenology_config.json
    results/eda/figures/20_phenology_retained_classes.png
    results/eda/figures/21_class_separability.png

Usage
-----
    python results/eda/_phenology.py [--per-class 250]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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

CATALOGUE = REPO / "data" / "catalogue_parcels_full.parquet"
CACHE = HERE / "cache"
FIG = HERE / "figures"
SEED = 42
COUNTRIES = ["Estonia", "Latvia", "Portugal"]
RED_IDX, NIR_IDX = 3, 7  # B04 and B08 in the fixed 13-band order
N_BINS = 36  # ten-day bins across 2021
MIN_TS = 10

sns.set_theme(context="notebook", style="whitegrid")


def _profile_chunk(paths: list[str]) -> np.ndarray:
    """Return an (n_parcels, N_BINS) array of binned NDVI, NaN where a bin is empty."""
    outs = np.full((len(paths), N_BINS), np.nan, dtype=np.float32)
    for r, p in enumerate(paths):
        try:
            with np.load(p) as z:
                data = z["data"].astype(np.float32)
                dates = np.asarray(z["dates"], dtype="datetime64[D]")
        except Exception:
            continue
        if data.size == 0:
            continue
        red, nir = data[:, RED_IDX], data[:, NIR_IDX]
        ndvi = (nir - red) / (nir + red + 1e-9)
        doy = (dates - np.datetime64("2021-01-01")).astype(int)
        b = np.clip(doy // 10, 0, N_BINS - 1)
        ok = np.isfinite(ndvi)
        if not ok.any():
            continue
        s = np.bincount(b[ok], weights=ndvi[ok], minlength=N_BINS)
        n = np.bincount(b[ok], minlength=N_BINS)
        with np.errstate(invalid="ignore", divide="ignore"):
            outs[r] = np.where(n > 0, s / np.maximum(n, 1), np.nan)
    return outs


def main(per_class: int, workers: int) -> None:
    t0 = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    ret_path = CACHE / "retained_classes.parquet"
    if not ret_path.exists():
        raise SystemExit("run results/eda/_class_design.py first")
    retained = pd.read_parquet(ret_path)
    cat = pd.read_parquet(CATALOGUE)
    cat["hcat"] = cat["hcat"].astype(str)
    cat = cat[cat["n_timesteps"] >= MIN_TS]

    rng = np.random.default_rng(SEED)
    picks = []
    for _, r in retained.iterrows():
        sub = cat[(cat["country"] == r["country"]) & (cat["hcat"] == r["hcat"])]
        if sub.empty:
            continue
        take = sub.sample(min(per_class, len(sub)), random_state=SEED)
        picks.append(take.assign(country_=r["country"], hcat_=r["hcat"]))
    sel = pd.concat(picks, ignore_index=True)
    print(f"sampling {len(sel):,} parcels across "
          f"{retained.groupby('country').size().to_dict()} retained classes", flush=True)

    paths = sel["path"].tolist()
    chunks = [paths[i:i + 500] for i in range(0, len(paths), 500)]
    mats = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, m in enumerate(ex.map(_profile_chunk, chunks, chunksize=1), start=1):
            mats.append(m)
            if i % 20 == 0 or i == len(chunks):
                print(f"  {i}/{len(chunks)} chunks, {time.perf_counter() - t0:.0f} s", flush=True)
    prof = np.vstack(mats)

    rows = []
    bin_doy = np.arange(N_BINS) * 10 + 5
    for (country, hcat), idx in sel.groupby(["country_", "hcat_"], observed=True).groups.items():
        block = prof[np.asarray(idx)]
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(block, axis=0)
            std = np.nanstd(block, axis=0)
            cnt = np.sum(np.isfinite(block), axis=0)
        for b in range(N_BINS):
            rows.append({"country": country, "hcat": hcat,
                         "crop_name": _hcat.name_of(hcat).replace("_", " "),
                         "bin": b, "doy": int(bin_doy[b]),
                         "ndvi_mean": float(mean[b]) if np.isfinite(mean[b]) else np.nan,
                         "ndvi_std": float(std[b]) if np.isfinite(std[b]) else np.nan,
                         "n_parcels_with_obs": int(cnt[b]),
                         "n_parcels_sampled": int(len(idx))})
    pdf = pd.DataFrame(rows)
    pdf.to_parquet(CACHE / "phenology_profiles.parquet", index=False)

    # separability: between-class over within-class variance of the binned NDVI profile
    sep = {}
    for country in COUNTRIES:
        m = (sel["country_"] == country).to_numpy()
        if m.sum() < 50:
            continue
        X = prof[m]
        lab = sel.loc[m, "hcat_"].to_numpy()
        keep_bins = np.isfinite(X).mean(axis=0) > 0.5
        if keep_bins.sum() < 5:
            continue
        Xk = X[:, keep_bins]
        col_mean = np.nanmean(Xk, axis=0)
        Xk = np.where(np.isfinite(Xk), Xk, col_mean)
        grand = Xk.mean(axis=0)
        ssb = ssw = 0.0
        for cl in np.unique(lab):
            g = Xk[lab == cl]
            ssb += len(g) * float(((g.mean(axis=0) - grand) ** 2).sum())
            ssw += float(((g - g.mean(axis=0)) ** 2).sum())
        k, n = len(np.unique(lab)), len(Xk)
        sep[country] = {"n_classes": int(k), "n_parcels": int(n),
                        "n_bins_used": int(keep_bins.sum()),
                        "between_within_ratio": round(ssb / max(ssw, 1e-9), 4),
                        "F_statistic": round((ssb / max(k - 1, 1)) / (ssw / max(n - k, 1)), 2)}
    (CACHE / "phenology_config.json").write_text(json.dumps({
        "script": "results/eda/_phenology.py", "random_seed": SEED,
        "parcels_per_class": per_class, "n_parcels_sampled": int(len(sel)),
        "n_bins": N_BINS, "bin_length_days": 10, "min_timesteps": MIN_TS,
        "separability": sep, "seconds_total": round(time.perf_counter() - t0, 1)},
        indent=2), encoding="utf-8")

    _figures(pdf, sep, per_class)
    print(json.dumps(sep, indent=2), flush=True)
    print(f"done in {time.perf_counter() - t0:.0f} s", flush=True)


def _figures(pdf: pd.DataFrame, sep: dict, per_class: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.5), sharey=True)
    for ax, country in zip(axes, COUNTRIES):
        sub = pdf[pdf.country == country]
        if sub.empty:
            ax.set_visible(False)
            continue
        order = (sub.groupby("crop_name")["n_parcels_sampled"].max()
                    .sort_values(ascending=False).index.tolist())
        cmap = plt.get_cmap("tab20")
        for i, name in enumerate(order):
            s = sub[sub.crop_name == name].sort_values("doy")
            s = s[s["n_parcels_with_obs"] >= 5]
            ax.plot(s["doy"], s["ndvi_mean"], lw=1.4, color=cmap(i % 20), label=name)
        ax.set_title(f"{country} ({len(order)} classes, "
                     f"up to {per_class} parcels each)", fontsize=10)
        ax.set_xlabel("day of year, 2021")
        ax.set_ylim(0.0, 0.85)
        ax.set_xlim(0, 365)
        ax.legend(fontsize=7, ncol=3, loc="upper center",
                  bbox_to_anchor=(0.5, -0.12), frameon=False)
    axes[0].set_ylabel("mean NDVI")
    fig.suptitle("NDVI phenology of the retained classes", y=1.02)
    plt.tight_layout()
    fig.savefig(FIG / "20_phenology_retained_classes.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    if sep:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ks = list(sep.keys())
        ax.bar(ks, [sep[k]["between_within_ratio"] for k in ks],
               color=["#1f77b4", "#ff7f0e", "#2ca02c"][:len(ks)])
        for i, k in enumerate(ks):
            ax.annotate(f"{sep[k]['between_within_ratio']:.3f}\n({sep[k]['n_classes']} classes)",
                        (i, sep[k]["between_within_ratio"]), ha="center", va="bottom", fontsize=8)
        ax.set_ylabel("between-class / within-class variance")
        ax.set_title("NDVI-only separability of the retained classes")
        plt.tight_layout()
        fig.savefig(FIG / "21_class_separability.png", dpi=130)
        plt.close(fig)
    print("figures 20 and 21 written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-class", type=int, default=250)
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    main(a.per_class, a.workers)
