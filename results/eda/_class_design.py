"""Class-design analysis for the EuroCropsML Phase 1 benchmark.

Consumes ``data/catalogue_parcels_full.parquet`` (built by ``_build_catalogue.py``)
and produces every table, statistic and figure needed to decide the thesis class
scheme: the class inventory, the exclusion rules, the aggregation level, the
spatial-blocking statistics and the temporal-sampling statistics.

Outputs
-------
    results/eda/cache/class_inventory.parquet (and .csv)
    results/eda/cache/class_design.json
    results/eda/cache/retained_classes.parquet
    results/eda/cache/spatial_stats.parquet
    results/eda/cache/block_occupancy.parquet
    results/eda/figures/12_*.png ... 19_*.png
    configs/class_scheme_eurocropsml.yaml

Usage
-----
    python results/eda/_class_design.py
"""

from __future__ import annotations

import json
import os
import sys
import time
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
CONFIG_OUT = REPO / "configs" / "class_scheme_eurocropsml.yaml"

SEED = 42
COUNTRIES = ["Estonia", "Latvia", "Portugal"]
PALETTE = {"Estonia": "#1f77b4", "Latvia": "#ff7f0e", "Portugal": "#2ca02c"}

sns.set_theme(context="notebook", style="whitegrid")

# --------------------------------------------------------------------------- #
# Design parameters. Every downstream threshold is derived from these.
# --------------------------------------------------------------------------- #
K_GRID = [1, 5, 10, 20, 50]  # recommended label budget, absolute samples per class
K_GRID_EXTENDED = [100, 200]  # only Estonia and Latvia can support these
K_GRID_ALL = K_GRID + K_GRID_EXTENDED  # used for the minimum-class-size table
N_TEST_MIN = 100  # held-out test parcels per class
N_VAL_MIN = 50  # validation parcels per class
N_DRAWS = 5  # independent random draws per K
POOL_MULTIPLIER = 2  # train pool must be at least 2 * K_max
MIN_TIMESTEPS_CANDIDATES = [1, 3, 5, 10, 12, 15, 20, 25, 30]
CLASS_SIZE_CANDIDATES = [20, 50, 100, 200, 250, 300, 350, 500, 550, 1000, 2000]
BLOCK_SIZES_KM = [1, 2, 5, 10, 20, 50, 100]
MAX_CLASSES_PER_COUNTRY = 20  # reporting cap, applied after the minimum-size rule

CATCH_ALL_PATTERNS = ["not_known", "other", "unknown", "fallow", "set_aside",
                      "idle", "unspecified", "mixed"]
NON_CROP_CODES = {
    "3399000000": ("not known and other; a level-3 catch-all that also absorbs national "
                   "declarations with no HCAT equivalent, including landscape features"),
    "3301990000": "other arable land crops; a catch-all for mixed and unresolved arable declarations",
    "3301110000": "fallow land, a land status rather than a crop",
    "3301011500": "unspecified cereals; species unresolved",
    "3301019900": "other cereals; species unresolved",
    "3301069900": "other industrial non-food crops; a catch-all",
    "3303990000": "other permanent crops; a catch-all within permanent crops",
    "3303120000": "unspecified permanent crops; declared as mixed permanent cultures",
    "3303080700": ("wire bush; the Portuguese declarations behind it are hedges, windbreaks "
                   "and non-grazable shrub, that is landscape features rather than a crop"),
    "3305000000": "greenhouse, foil and film; a structure rather than a crop",
    "3304000000": "mushrooms, energy and genetically modified crops; a heterogeneous residual",
    "3300000000": "crop type; a taxonomy root, uninformative as a label",
    "3000000000": "characteristics; a taxonomy root, uninformative as a label",
}

#: Whole HCAT branches removed by code prefix.
EXCLUDE_PREFIXES = {
    "3306": "tree, wood and forest; forestry rather than an agricultural crop",
}

#: Classes excluded above that are defensible to reinstate, with the reason.
BORDERLINE_CODES = {
    "3306060000": ("oak; the Portuguese declarations are cork and holm oak plantations, that is "
                   "montado silvo-pastoral systems, which are agriculturally managed even though "
                   "the HCAT branch is forestry"),
    "3301110000": ("fallow land; a legitimate class if the task is framed as land use rather "
                   "than crop type, and it is the third largest Latvian class"),
}


# Generous national bounding boxes (lon_min, lon_max, lat_min, lat_max), WGS-84.
BBOX = {
    "Estonia": (21.5, 28.3, 57.4, 59.8),
    "Latvia": (20.8, 28.3, 55.6, 58.2),
    "Portugal": (-31.5, -6.1, 32.3, 42.3),  # mainland plus Azores and Madeira
}
BBOX_MAINLAND = {"Portugal": (-9.7, -6.1, 36.8, 42.3)}

EARTH_R_KM = 6371.0088


def _excluded_mask(codes: "pd.Series") -> "pd.Series":
    """True where an HCAT code is a catch-all, a taxonomy root or an excluded branch."""
    m = codes.isin(NON_CROP_CODES)
    for pref in EXCLUDE_PREFIXES:
        m = m | codes.str.startswith(pref)
    return m


def _exclusion_reason(code: str) -> str:
    if code in NON_CROP_CODES:
        return NON_CROP_CODES[code]
    for pref, reason in EXCLUDE_PREFIXES.items():
        if str(code).startswith(pref):
            return reason
    return ""


def _min_class_size(k_max: int) -> int:
    return POOL_MULTIPLIER * k_max + N_VAL_MIN + N_TEST_MIN


def _gini(counts) -> float:
    x = np.sort(np.asarray(counts, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return float("nan")
    idx = np.arange(1, n + 1)
    return float((2 * idx - n - 1).dot(x) / (n * x.sum()))


def _norm_entropy(counts) -> float:
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    if p.size <= 1:
        return 0.0
    p = p / p.sum()
    return float(-(p * np.log(p)).sum() / np.log(p.size))


def load_catalogue() -> pd.DataFrame:
    if not CATALOGUE.exists():
        raise SystemExit(f"catalogue not found: {CATALOGUE}\n"
                         "run results/eda/_build_catalogue.py first")
    df = pd.read_parquet(CATALOGUE)
    df["hcat"] = df["hcat"].astype(str)
    df["country"] = pd.Categorical(df["country"], categories=COUNTRIES + ["Unknown"])
    return df


# --------------------------------------------------------------------------- #
def main() -> None:
    t0 = time.perf_counter()
    CACHE.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    df = load_catalogue()
    n_total = len(df)
    print(f"catalogue: {n_total:,} parcels", flush=True)

    lc = [f"n_{c.lower()}" for c in COUNTRIES]
    out: dict = {
        "source_catalogue": str(CATALOGUE.relative_to(REPO)).replace(os.sep, "/"),
        "n_parcels": int(n_total),
        "random_seed": SEED,
        "design_parameters": {
            "K_grid": K_GRID, "K_grid_extended_baltic_only": K_GRID_EXTENDED,
            "n_test_min": N_TEST_MIN, "n_val_min": N_VAL_MIN,
            "n_draws": N_DRAWS, "pool_multiplier": POOL_MULTIPLIER,
        },
    }

    # ------------------------------------------------------------------ 3a --
    inv = (df.groupby(["hcat", "country"], observed=True).size()
             .unstack("country", fill_value=0).reindex(columns=COUNTRIES, fill_value=0))
    inv.columns = lc
    inv["n_total"] = inv.sum(axis=1)
    inv = inv.reset_index()
    inv["hcat_name"] = inv["hcat"].map(_hcat.name_of)
    inv["crop_name"] = inv["hcat_name"].str.replace("_", " ")
    inv["level"] = inv["hcat"].map(_hcat.hcat_level)
    inv["share_pct"] = 100 * inv["n_total"] / n_total
    inv["n_countries"] = (inv[lc] > 0).sum(axis=1)
    inv = inv.sort_values("n_total", ascending=False).reset_index(drop=True)
    inv.to_parquet(CACHE / "class_inventory.parquet", index=False)
    inv.to_csv(CACHE / "class_inventory.csv", index=False, encoding="utf-8")

    out["n_distinct_classes"] = int(inv["hcat"].nunique())
    out["n_classes_unresolved_name"] = int(inv["hcat_name"].str.startswith("unknown_").sum())
    out["unresolved_codes"] = inv.loc[inv["hcat_name"].str.startswith("unknown_"), "hcat"].tolist()
    out["parcels_per_country"] = {c: int((df["country"] == c).sum()) for c in COUNTRIES}
    out["classes_per_country"] = {c: int((inv[f"n_{c.lower()}"] > 0).sum()) for c in COUNTRIES}
    print(f"distinct classes: {out['n_distinct_classes']} "
          f"(unresolved names: {out['n_classes_unresolved_name']})", flush=True)

    cum = np.cumsum(inv["n_total"].to_numpy()) / n_total
    out["class_coverage"] = {f"top_k_for_{int(q * 100)}pct": int(np.searchsorted(cum, q) + 1)
                             for q in (0.5, 0.8, 0.95, 0.99)}
    out["level_distribution"] = {int(k): int(v) for k, v in inv.groupby("level").size().items()}
    out["top20_classes"] = inv.head(20)[
        ["hcat", "crop_name", "level"] + lc + ["n_total", "share_pct", "n_countries"]
    ].to_dict("records")

    # ------------------------------------------------------------ 3b (i) ---
    survival = []
    for thr in CLASS_SIZE_CANDIDATES:
        row = {"threshold": thr,
               "implied_k_max": int(max(0, (thr - N_VAL_MIN - N_TEST_MIN) // POOL_MULTIPLIER))}
        for c in COUNTRIES:
            col = f"n_{c.lower()}"
            keep = inv[inv[col] >= thr]
            row[f"n_classes_{c.lower()}"] = int(len(keep))
            row[f"parcels_kept_{c.lower()}"] = int(keep[col].sum())
            row[f"pct_parcels_kept_{c.lower()}"] = round(
                100 * keep[col].sum() / max(1, inv[col].sum()), 2)
        row["n_classes_all_three"] = int((inv[lc] >= thr).all(axis=1).sum())
        survival.append(row)
    survival_df = pd.DataFrame(survival)
    survival_df.to_parquet(CACHE / "class_size_survival.parquet", index=False)
    out["class_size_survival"] = survival_df.to_dict("records")
    out["min_class_size_for_K"] = {str(k): _min_class_size(k) for k in K_GRID_ALL}

    # ----------------------------------------------------------- 3b (ii) ---
    catchall = inv[_excluded_mask(inv["hcat"])].copy()
    catchall["reason"] = catchall["hcat"].map(_exclusion_reason)
    out["non_crop_classes"] = catchall[
        ["hcat", "crop_name"] + lc + ["n_total", "share_pct", "reason"]].to_dict("records")
    out["borderline_classes"] = [
        {"hcat": k, "crop_name": _hcat.name_of(k).replace("_", " "),
         "n_total": int(inv.loc[inv["hcat"] == k, "n_total"].sum()),
         **{c.lower(): int(inv.loc[inv["hcat"] == k, f"n_{c.lower()}"].sum())
            for c in COUNTRIES}, "note": v}
        for k, v in BORDERLINE_CODES.items()]
    out["non_crop_parcels_total"] = int(catchall["n_total"].sum())
    out["non_crop_parcels_pct"] = round(100 * catchall["n_total"].sum() / n_total, 2)

    susp = inv[inv["hcat_name"].str.contains("|".join(CATCH_ALL_PATTERNS), case=False, na=False)]
    out["suspicious_named_classes"] = susp[
        ["hcat", "crop_name"] + lc + ["n_total", "share_pct"]].head(40).to_dict("records")

    g = inv[inv["hcat"] == "3302000000"]
    if len(g):
        out["grassland"] = g[["hcat", "crop_name"] + lc + ["n_total", "share_pct"]].iloc[0].to_dict()

    # national declarations behind the catch-all and grassland codes
    try:
        cm = _hcat.country_mapping()
        out["catch_all_national_declarations"] = {
            code: {c: sorted(str(x) for x in
                             cm[(cm["hcat"] == code) & (cm["country"] == c)]["translated_name"]
                             .dropna().unique())[:12]
                   for c in COUNTRIES}
            for code in ["3399000000", "3301990000", "3302000000", "3301110000"]}
    except Exception as exc:  # pragma: no cover
        out["catch_all_national_declarations"] = {"error": str(exc)}

    # ---------------------------------------------------------- 3b (iii) ---
    ts_rows = []
    for thr in MIN_TIMESTEPS_CANDIDATES:
        row = {"min_timesteps": thr}
        for c in COUNTRIES:
            sub = df[df["country"] == c]
            lost = int((sub["n_timesteps"] < thr).sum())
            row[f"lost_{c.lower()}"] = lost
            row[f"pct_lost_{c.lower()}"] = round(100 * lost / max(1, len(sub)), 3)
        row["lost_total"] = int((df["n_timesteps"] < thr).sum())
        row["pct_lost_total"] = round(100 * row["lost_total"] / n_total, 3)
        ts_rows.append(row)
    ts_df = pd.DataFrame(ts_rows)
    ts_df.to_parquet(CACHE / "timestep_threshold_loss.parquet", index=False)
    out["timestep_threshold_loss"] = ts_df.to_dict("records")
    out["timestep_stats"] = {}
    for c in COUNTRIES:
        s = df.loc[df.country == c, "n_timesteps"]
        out["timestep_stats"][c] = {
            "mean": round(float(s.mean()), 2), "std": round(float(s.std()), 2),
            "median": int(s.median()), "p01": int(s.quantile(0.01)),
            "p05": int(s.quantile(0.05)), "min": int(s.min()), "max": int(s.max())}

    ts_by_class = (df.groupby(["country", "hcat"], observed=True)["n_timesteps"]
                     .agg(["median", "min", "count"]).reset_index())
    ts_by_class["crop_name"] = ts_by_class["hcat"].map(
        lambda c: _hcat.name_of(c).replace("_", " "))
    ts_by_class.to_parquet(CACHE / "timesteps_by_class.parquet", index=False)

    # ----------------------------------------------------------- 3b (iv) ---
    geom = {}
    for c in COUNTRIES:
        sub = df[df["country"] == c]
        lo, hi, la, ha = BBOX[c]
        outside = ~(sub["lon"].between(lo, hi) & sub["lat"].between(la, ha))
        geom[c] = {"n": int(len(sub)), "n_outside_bbox": int(outside.sum()),
                   "pct_outside_bbox": round(100 * outside.sum() / max(1, len(sub)), 4),
                   "n_missing_coords": int(sub[["lat", "lon"]].isna().any(axis=1).sum()),
                   "lon_min": round(float(sub["lon"].min()), 4),
                   "lon_max": round(float(sub["lon"].max()), 4),
                   "lat_min": round(float(sub["lat"].min()), 4),
                   "lat_max": round(float(sub["lat"].max()), 4)}
        if c in BBOX_MAINLAND:
            lo2, hi2, la2, ha2 = BBOX_MAINLAND[c]
            off = ~(sub["lon"].between(lo2, hi2) & sub["lat"].between(la2, ha2))
            geom[c]["n_outside_mainland"] = int(off.sum())
            geom[c]["pct_outside_mainland"] = round(100 * off.sum() / max(1, len(sub)), 3)
    out["geometry"] = geom

    dup_coord = df.duplicated(subset=["lat", "lon"], keep=False)
    pid_multi_nuts = df.groupby("parcel_id", observed=True)["nuts"].nunique()
    out["integrity"] = {
        "n_rows_sharing_a_coordinate": int(dup_coord.sum()),
        "n_unique_coordinates": int(df[["lat", "lon"]].drop_duplicates().shape[0]),
        "n_rows_duplicate_parcel_id_global": int(df.duplicated(subset=["parcel_id"],
                                                               keep=False).sum()),
        "n_rows_duplicate_parcel_id_within_country": int(
            df.duplicated(subset=["country", "parcel_id"], keep=False).sum()),
        "n_rows_duplicate_nuts_parcel_id": int(
            df.duplicated(subset=["nuts", "parcel_id"], keep=False).sum()),
        "n_parcel_ids_in_multiple_nuts": int((pid_multi_nuts > 1).sum()),
        "n_parcels_zero_timesteps": int((df["n_timesteps"] == 0).sum()),
        "n_rows_duplicate_filename_key": int(
            df.duplicated(subset=["nuts", "parcel_id", "hcat"], keep=False).sum()),
    }
    if dup_coord.any():
        dd = df[dup_coord]
        nclass = dd.groupby(["lat", "lon"], observed=True)["hcat"].nunique()
        ncountry = dd.groupby(["lat", "lon"], observed=True)["country"].nunique()
        out["integrity"]["n_coordinate_groups_with_multiple_classes"] = int((nclass > 1).sum())
        out["integrity"]["n_coordinate_groups_spanning_countries"] = int((ncountry > 1).sum())

    # ------------------------------------------------------------------ 3c --
    agg_rows = []
    for lvl in (3, 4, 5, 6):
        d = df[["country", "hcat"]].copy()
        d["cls"] = d["hcat"].map(lambda c: _hcat.roll_up(c, lvl))
        piv = (d.groupby(["cls", "country"], observed=True).size()
                 .unstack("country", fill_value=0).reindex(columns=COUNTRIES, fill_value=0))
        counts = piv.sum(axis=1).to_numpy()
        row = {"level": lvl, "n_classes": int(piv.shape[0]),
               "gini": round(_gini(counts), 4),
               "normalised_entropy": round(_norm_entropy(counts), 4),
               "imbalance_ratio": round(float(counts.max() / max(1, counts.min())), 1),
               "max_class_share_pct": round(100 * counts.max() / counts.sum(), 2)}
        for thr in (0, 100, 350, 550):
            ok = (piv >= thr) if thr else (piv > 0)
            row[f"shared_all_three_min{thr}"] = int(ok.all(axis=1).sum())
            for c in COUNTRIES:
                row[f"n_classes_{c.lower()}_min{thr}"] = int(ok[c].sum())
        for a, b in [("Estonia", "Latvia"), ("Estonia", "Portugal"), ("Latvia", "Portugal")]:
            for thr in (0, 350):
                ok = (piv >= thr) if thr else (piv > 0)
                row[f"shared_{a[:2].upper()}_{b[:2].upper()}_min{thr}"] = int((ok[a] & ok[b]).sum())
        agg_rows.append(row)
    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_parquet(CACHE / "aggregation_levels.parquet", index=False)
    out["aggregation_levels"] = agg_df.to_dict("records")

    seasonal = inv[inv["level"] == 6].copy()
    seasonal["parent"] = seasonal["hcat"].map(lambda c: _hcat.roll_up(c, 5))
    seasonal["season"] = np.select(
        [seasonal["hcat_name"].str.startswith("winter"),
         seasonal["hcat_name"].str.startswith("spring"),
         seasonal["hcat_name"].str.startswith("summer"),
         seasonal["hcat_name"].str.startswith("unspecified")],
        ["winter", "spring", "summer", "unspecified"], default="other")
    pairs = []
    for parent, grp in seasonal[seasonal["season"].isin(["winter", "spring"])].groupby("parent"):
        if grp["season"].nunique() < 2:
            continue
        rec = {"parent": parent, "parent_name": _hcat.name_of(parent).replace("_", " ")}
        for _, r in grp.iterrows():
            rec[f"{r['season']}_code"] = r["hcat"]
            rec[f"{r['season']}_n_total"] = int(r["n_total"])
            for c in COUNTRIES:
                rec[f"{r['season']}_n_{c.lower()}"] = int(r[f"n_{c.lower()}"])
        rec["merged_n_total"] = int(grp["n_total"].sum())
        pairs.append(rec)
    seasonal_df = (pd.DataFrame(pairs).sort_values("merged_n_total", ascending=False)
                   if pairs else pd.DataFrame(columns=["parent", "parent_name", "merged_n_total"]))
    seasonal_df.to_parquet(CACHE / "seasonal_pairs.parquet", index=False)
    out["seasonal_pairs"] = seasonal_df.to_dict("records")

    # -------------------------------------------------------------- scheme --
    K_MAX = 50
    MIN_CLASS = _min_class_size(K_MAX)
    MIN_TS = 10
    out["recommended"] = {
        "min_class_size_per_country": MIN_CLASS, "K_max": K_MAX,
        "max_classes_per_country": MAX_CLASSES_PER_COUNTRY, "min_timesteps": MIN_TS,
        "aggregation": "native HCAT level, no roll-up, for the in-country task",
        "rationale_min_class_size":
            f"{POOL_MULTIPLIER} * K_max({K_MAX}) + validation({N_VAL_MIN}) + test({N_TEST_MIN})"}

    excluded = _excluded_mask(df["hcat"])
    clean = df[(df["n_timesteps"] >= MIN_TS) & (~excluded)].copy()
    keep_mask = pd.Series(True, index=clean.index)
    for c in COUNTRIES:
        lo, hi, la, ha = BBOX[c]
        m = (clean["country"] == c).to_numpy()
        inside = (clean["lon"].between(lo, hi) & clean["lat"].between(la, ha)).to_numpy()
        keep_mask &= ~m | inside
    clean = clean[keep_mask]
    out["n_parcels_after_filters"] = int(len(clean))
    out["pct_parcels_after_filters"] = round(100 * len(clean) / n_total, 2)
    out["parcels_after_filters_per_country"] = {
        c: int((clean["country"] == c).sum()) for c in COUNTRIES}

    piv_clean = (clean.groupby(["hcat", "country"], observed=True).size()
                   .unstack("country", fill_value=0).reindex(columns=COUNTRIES, fill_value=0))
    retained = {}
    retained_untrimmed = {}
    for c in COUNTRIES:
        keep = piv_clean[piv_clean[c] >= MIN_CLASS].sort_values(c, ascending=False)
        rows = [{"hcat": h, "crop_name": _hcat.name_of(h).replace("_", " "),
                 "n": int(keep.at[h, c])} for h in keep.index]
        retained_untrimmed[c] = rows
        retained[c] = rows[:MAX_CLASSES_PER_COUNTRY]
    out["retained_classes_per_country_before_cap"] = {
        c: len(v) for c, v in retained_untrimmed.items()}
    out["max_classes_per_country"] = MAX_CLASSES_PER_COUNTRY
    out["retained_classes_per_country"] = {c: len(v) for c, v in retained.items()}
    out["retained_classes_detail"] = retained
    out["retained_coverage_pct"] = {
        c: round(100 * sum(r["n"] for r in retained[c])
                 / max(1, int((clean["country"] == c).sum())), 2) for c in COUNTRIES}

    sel_codes = {c: {r['hcat'] for r in retained[c]} for c in COUNTRIES}
    ok = pd.DataFrame({c: piv_clean.index.isin(list(sel_codes[c])) for c in COUNTRIES},
                      index=piv_clean.index)
    shared3 = piv_clean[ok.all(axis=1)]
    out["shared_three_way"] = [
        {"hcat": h, "crop_name": _hcat.name_of(h).replace("_", " "),
         **{c.lower(): int(shared3.at[h, c]) for c in COUNTRIES}}
        for h in shared3.sum(axis=1).sort_values(ascending=False).index]
    pairwise = {}
    for a, b in [("Estonia", "Latvia"), ("Estonia", "Portugal"), ("Latvia", "Portugal")]:
        sel = piv_clean[ok[a] & ok[b]]
        pairwise[f"{a}|{b}"] = [
            {"hcat": h, "crop_name": _hcat.name_of(h).replace("_", " "),
             a.lower(): int(sel.at[h, a]), b.lower(): int(sel.at[h, b])}
            for h in sel[a].sort_values(ascending=False).index]
    out["shared_pairwise"] = pairwise
    out["shared_pairwise_counts"] = {k: len(v) for k, v in pairwise.items()}

    # the same at level 4, to show what aggregation buys the transfer experiment
    clean4 = clean[["country", "hcat"]].copy()
    clean4["cls"] = clean4["hcat"].map(lambda c: _hcat.roll_up(c, 4))
    piv4 = (clean4.groupby(["cls", "country"], observed=True).size()
              .unstack("country", fill_value=0).reindex(columns=COUNTRIES, fill_value=0))
    ok4 = piv4 >= MIN_CLASS
    out["level4_retained_per_country"] = {c: int(ok4[c].sum()) for c in COUNTRIES}
    out["level4_shared_three_way"] = [
        {"hcat": h, "crop_name": _hcat.name_of(h).replace("_", " "),
         **{c.lower(): int(piv4.at[h, c]) for c in COUNTRIES}}
        for h in piv4[ok4.all(axis=1)].sum(axis=1).sort_values(ascending=False).index]
    out["level4_shared_pairwise_counts"] = {
        f"{a}|{b}": int((ok4[a] & ok4[b]).sum())
        for a, b in [("Estonia", "Latvia"), ("Estonia", "Portugal"), ("Latvia", "Portugal")]}

    # sensitivity of the retained class count to the minimum-size threshold,
    # computed after the catch-all, branch, length and bounding-box filters
    clean_surv = []
    for thr in CLASS_SIZE_CANDIDATES:
        row = {"threshold": thr,
               "implied_k_max": int(max(0, (thr - N_VAL_MIN - N_TEST_MIN) // POOL_MULTIPLIER))}
        okt = piv_clean >= thr
        okt4 = piv4 >= thr
        for c in COUNTRIES:
            row[f"n_classes_{c.lower()}"] = int(okt[c].sum())
            row[f"n_classes_level4_{c.lower()}"] = int(okt4[c].sum())
            row[f"pct_parcels_kept_{c.lower()}"] = round(
                100 * piv_clean.loc[okt[c], c].sum()
                / max(1, int(piv_clean[c].sum())), 2)
        row["n_classes_all_three"] = int(okt.all(axis=1).sum())
        row["n_classes_all_three_level4"] = int(okt4.all(axis=1).sum())
        for a, b in [("Estonia", "Latvia"), ("Estonia", "Portugal"), ("Latvia", "Portugal")]:
            row[f"shared_{a[:2].upper()}_{b[:2].upper()}"] = int((okt[a] & okt[b]).sum())
            row[f"shared_level4_{a[:2].upper()}_{b[:2].upper()}"] = int((okt4[a] & okt4[b]).sum())
        clean_surv.append(row)
    clean_surv_df = pd.DataFrame(clean_surv)
    clean_surv_df.to_parquet(CACHE / "retained_class_survival.parquet", index=False)
    out["retained_class_survival"] = clean_surv_df.to_dict("records")

    ret_rows = [{"country": c, **r} for c in COUNTRIES for r in retained[c]]
    pd.DataFrame(ret_rows).to_parquet(CACHE / "retained_classes.parquet", index=False)

    # ------------------------------------------------------------- spatial --
    spatial_df, block_df = _spatial(clean, retained)
    spatial_df.to_parquet(CACHE / "spatial_stats.parquet", index=False)
    block_df.to_parquet(CACHE / "block_occupancy.parquet", index=False)
    out["spatial_label_autocorrelation"] = (
        spatial_df.dropna(subset=["radius_km"]).to_dict("records"))
    out["median_nn_distance_km"] = {
        r["country"]: round(float(r["median_nn_distance_km"]), 4)
        for r in spatial_df[spatial_df["radius_km"].isna()].to_dict("records")}
    out["block_occupancy"] = block_df.to_dict("records")

    # ------------------------------------------------------------ temporal --
    mcols = [f"m{i:02d}" for i in range(1, 13)]
    out["monthly_coverage_pct_parcels_with_obs"] = {
        c: {m: round(100 * float((clean.loc[clean.country == c, m] > 0).mean()), 2) for m in mcols}
        for c in COUNTRIES}
    out["monthly_mean_observations"] = {
        c: {m: round(float(clean.loc[clean.country == c, m].mean()), 2) for m in mcols}
        for c in COUNTRIES}
    out["monthly_grid_completeness_pct"] = {
        c: round(100 * float((clean.loc[clean.country == c, mcols] > 0).to_numpy().mean()), 2)
        for c in COUNTRIES}
    growing = [f"m{i:02d}" for i in range(3, 11)]
    out["growing_season_grid_completeness_mar_oct_pct"] = {
        c: round(100 * float((clean.loc[clean.country == c, growing] > 0).to_numpy().mean()), 2)
        for c in COUNTRIES}

    # ------------------------------------------------------------- figures --
    _figures(inv, survival_df, ts_df, df, agg_df, seasonal_df, retained,
             spatial_df, block_df, clean, MIN_CLASS, MIN_TS)

    _write_scheme(out, retained, MIN_CLASS, MIN_TS, K_MAX)

    out["seconds_total"] = round(time.perf_counter() - t0, 1)
    (CACHE / "class_design.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"wrote {CACHE / 'class_design.json'} ({out['seconds_total']} s)", flush=True)


# --------------------------------------------------------------------------- #
def _spatial(clean: pd.DataFrame, retained: dict):
    """Label autocorrelation by distance ring and spatial-block occupancy."""
    from sklearn.neighbors import BallTree

    dist_edges_km = np.array([0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0])
    rows = []
    for c in COUNTRIES:
        sub = clean[(clean["country"] == c)].dropna(subset=["lat", "lon"])
        if len(sub) < 1000:
            continue
        ref = sub.sample(min(120_000, len(sub)), random_state=SEED)
        codes, uniq = pd.factorize(ref["hcat"].to_numpy())
        p = np.bincount(codes) / len(codes)
        prior = float((p ** 2).sum())  # probability two random parcels share a class
        tree = BallTree(np.radians(ref[["lat", "lon"]].to_numpy()), metric="haversine")
        q = ref.sample(min(2500, len(ref)), random_state=SEED + 1)
        qcodes = pd.Categorical(q["hcat"], categories=list(uniq)).codes
        qrad = np.radians(q[["lat", "lon"]].to_numpy())
        prev_same = np.zeros(len(q))
        prev_tot = np.zeros(len(q))
        for r_km in dist_edges_km:
            idx = tree.query_radius(qrad, r=float(r_km) / EARTH_R_KM)
            tot = np.array([len(i) - 1 for i in idx], dtype=float)
            same = np.array([int((codes[i] == qc).sum()) - 1 for i, qc in zip(idx, qcodes)],
                            dtype=float)
            ring_tot, ring_same = tot - prev_tot, same - prev_same
            prev_tot, prev_same = tot, same
            with np.errstate(invalid="ignore", divide="ignore"):
                frac = np.where(ring_tot > 0, ring_same / ring_tot, np.nan)
            rows.append({"country": c, "radius_km": float(r_km),
                         "mean_same_class_fraction_in_ring": float(np.nanmean(frac)),
                         "background_prior": prior,
                         "median_neighbours_cumulative": float(np.median(tot)),
                         "median_nn_distance_km": np.nan,
                         "n_query_points": int(len(q)), "n_reference_parcels": int(len(ref))})
        d_any, _ = tree.query(qrad, k=2)
        rows.append({"country": c, "radius_km": np.nan,
                     "mean_same_class_fraction_in_ring": np.nan, "background_prior": prior,
                     "median_neighbours_cumulative": np.nan,
                     "median_nn_distance_km": float(np.median(d_any[:, 1]) * EARTH_R_KM),
                     "n_query_points": int(len(q)), "n_reference_parcels": int(len(ref))})
    spatial_df = pd.DataFrame(rows)

    block_rows = []
    for c in COUNTRIES:
        keep_codes = {r["hcat"] for r in retained[c]}
        sub = clean[(clean["country"] == c) & (clean["hcat"].isin(keep_codes))]
        sub = sub.dropna(subset=["lat", "lon"])
        if sub.empty:
            continue
        lat0 = float(sub["lat"].mean())
        for bkm in BLOCK_SIZES_KM:
            dlat = bkm / 110.574
            dlon = bkm / (111.320 * np.cos(np.radians(lat0)))
            bi = np.floor(sub["lat"].to_numpy() / dlat).astype(np.int64)
            bj = np.floor(sub["lon"].to_numpy() / dlon).astype(np.int64)
            tmp = pd.DataFrame({"hcat": sub["hcat"].to_numpy(), "block": bi * 1_000_003 + bj})
            per_class = tmp.groupby("hcat")["block"].nunique()
            per_block = tmp.groupby("block").size()
            block_rows.append({
                "country": c, "block_km": bkm, "n_blocks": int(tmp["block"].nunique()),
                "median_parcels_per_block": float(per_block.median()),
                "p90_parcels_per_block": float(per_block.quantile(0.9)),
                "min_blocks_per_class": int(per_class.min()),
                "median_blocks_per_class": float(per_class.median()),
                "n_classes_with_ge_5_blocks": int((per_class >= 5).sum()),
                "n_classes_with_ge_10_blocks": int((per_class >= 10).sum()),
                "n_retained_classes": int(len(per_class))})
    return spatial_df, pd.DataFrame(block_rows)


# --------------------------------------------------------------------------- #
def _figures(inv, survival_df, ts_df, df, agg_df, seasonal_df, retained,
             spatial_df, block_df, clean, MIN_CLASS, MIN_TS) -> None:
    # 12 class survival versus minimum class size
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for c in COUNTRIES:
        axes[0].plot(survival_df["threshold"], survival_df[f"n_classes_{c.lower()}"],
                     marker="o", color=PALETTE[c], label=c)
    axes[0].plot(survival_df["threshold"], survival_df["n_classes_all_three"],
                 marker="s", ls="--", color="black", label="shared by all three")
    axes[0].axvline(MIN_CLASS, color="red", ls=":", lw=1.5)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("minimum parcels per class per country")
    axes[0].set_ylabel("classes surviving")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Classes surviving a minimum-size threshold\n(red line: recommended {MIN_CLASS})")
    for c in COUNTRIES:
        axes[1].plot(survival_df["threshold"], survival_df[f"pct_parcels_kept_{c.lower()}"],
                     marker="o", color=PALETTE[c], label=c)
    axes[1].axvline(MIN_CLASS, color="red", ls=":", lw=1.5)
    axes[1].set_xscale("log")
    axes[1].set_xlabel("minimum parcels per class per country")
    axes[1].set_ylabel("% of national parcels retained")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Parcels retained by the same threshold")
    plt.tight_layout()
    fig.savefig(FIG / "12_class_size_survival.png", dpi=130)
    plt.close(fig)

    # 13 time-series length
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for c in COUNTRIES:
        axes[0].hist(df.loc[df.country == c, "n_timesteps"].to_numpy(),
                     bins=np.arange(0, 160, 2), alpha=0.55, label=c, color=PALETTE[c])
    axes[0].axvline(MIN_TS, color="red", ls=":", lw=1.5)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("cloud-free observations per parcel")
    axes[0].set_ylabel("parcels (log scale)")
    axes[0].legend(fontsize=8)
    axes[0].set_title(f"Time-series length, all {len(df):,} parcels")
    for c in COUNTRIES:
        axes[1].plot(ts_df["min_timesteps"], ts_df[f"pct_lost_{c.lower()}"],
                     marker="o", color=PALETTE[c], label=c)
    axes[1].axvline(MIN_TS, color="red", ls=":", lw=1.5)
    axes[1].set_xlabel("minimum observations required")
    axes[1].set_ylabel("% of national parcels discarded")
    axes[1].legend(fontsize=8)
    axes[1].set_title("Cost of a minimum-length rule")
    plt.tight_layout()
    fig.savefig(FIG / "13_timeseries_length_rule.png", dpi=130)
    plt.close(fig)

    # 14 aggregation levels
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].bar(agg_df["level"].astype(str), agg_df["n_classes"], color="steelblue")
    for x, v in zip(range(len(agg_df)), agg_df["n_classes"]):
        axes[0].annotate(str(v), (x, v), ha="center", va="bottom", fontsize=8)
    axes[0].set_xlabel("HCAT level")
    axes[0].set_ylabel("distinct classes")
    axes[0].set_title("Classes per aggregation level")
    axes[1].plot(agg_df["level"], agg_df["normalised_entropy"], marker="o",
                 label="normalised entropy")
    axes[1].plot(agg_df["level"], agg_df["gini"], marker="s", label="Gini coefficient")
    axes[1].set_xlabel("HCAT level")
    axes[1].set_xticks(agg_df["level"])
    axes[1].legend(fontsize=8)
    axes[1].set_title("Balance of the class distribution")
    axes[2].plot(agg_df["level"], agg_df["shared_all_three_min0"], marker="o",
                 label="present in all three")
    axes[2].plot(agg_df["level"], agg_df["shared_all_three_min350"], marker="s",
                 label="at least 350 parcels in all three")
    axes[2].set_xlabel("HCAT level")
    axes[2].set_xticks(agg_df["level"])
    axes[2].set_ylabel("shared classes")
    axes[2].legend(fontsize=8)
    axes[2].set_title("Cross-country intersection")
    plt.tight_layout()
    fig.savefig(FIG / "14_aggregation_levels.png", dpi=130)
    plt.close(fig)

    # 15 seasonal variants
    if len(seasonal_df) and "winter_n_total" in seasonal_df.columns:
        sp = seasonal_df.head(10).copy()
        fig, ax = plt.subplots(figsize=(9, 4.5))
        y = np.arange(len(sp))
        ax.barh(y - 0.2, sp["winter_n_total"].fillna(0), height=0.4,
                color="#3b6ea5", label="winter variant")
        ax.barh(y + 0.2, sp["spring_n_total"].fillna(0), height=0.4,
                color="#8fbf6a", label="spring variant")
        ax.set_yticks(y)
        ax.set_yticklabels(sp["parent_name"], fontsize=8)
        ax.invert_yaxis()
        ax.set_xscale("symlog")
        ax.set_xlabel("parcels (symmetric log scale)")
        ax.legend(fontsize=8)
        ax.set_title("Winter and spring variants of the same crop, all parcels")
        plt.tight_layout()
        fig.savefig(FIG / "15_seasonal_variants.png", dpi=130)
        plt.close(fig)

    # 16 retained classes per country
    fig, axes = plt.subplots(1, 3, figsize=(17, 6))
    for ax, c in zip(axes, COUNTRIES):
        items = retained[c]
        if not items:
            ax.set_visible(False)
            continue
        names = [i["crop_name"] for i in items][::-1]
        vals = [i["n"] for i in items][::-1]
        ax.barh(np.arange(len(vals)), vals, color=PALETTE[c])
        ax.set_yticks(np.arange(len(vals)))
        ax.set_yticklabels(names, fontsize=7)
        ax.set_xscale("log")
        ax.set_xlabel("parcels (log scale)")
        ax.set_title(f"{c}: {len(items)} retained classes")
    fig.suptitle(f"Retained classes after the recommended cut-offs (at least {MIN_CLASS} parcels "
                 f"per class per country, at least {MIN_TS} observations, catch-alls removed)",
                 y=1.02, fontsize=11)
    plt.tight_layout()
    fig.savefig(FIG / "16_retained_classes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # 17 spatial label autocorrelation
    sdf = spatial_df.dropna(subset=["radius_km"])
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    for c in COUNTRIES:
        s = sdf[sdf.country == c]
        if s.empty:
            continue
        ax.plot(s["radius_km"], s["mean_same_class_fraction_in_ring"],
                marker="o", color=PALETTE[c], label=c)
        ax.axhline(float(s["background_prior"].iloc[0]), color=PALETTE[c], ls=":", lw=1)
    ax.set_xscale("log")
    ax.set_xlabel("distance ring (km)")
    ax.set_ylabel("fraction of neighbours sharing the class")
    ax.set_title("Spatial label autocorrelation\n(dotted lines: national random-pair baseline)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(FIG / "17_spatial_autocorrelation.png", dpi=130)
    plt.close(fig)

    # 18 block occupancy
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for c in COUNTRIES:
        s = block_df[block_df.country == c]
        if s.empty:
            continue
        axes[0].plot(s["block_km"], s["n_blocks"], marker="o", color=PALETTE[c], label=c)
        axes[1].plot(s["block_km"], s["median_blocks_per_class"], marker="o",
                     color=PALETTE[c], label=c)
    for ax, ttl, ylb in ((axes[0], "Spatial blocks available", "blocks (log scale)"),
                         (axes[1], "Median blocks occupied per retained class",
                          "blocks (log scale)")):
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("block side (km)")
        ax.set_ylabel(ylb)
        ax.set_title(ttl)
        ax.legend(fontsize=8)
    axes[1].axhline(5, color="red", ls=":", lw=1.2)
    plt.tight_layout()
    fig.savefig(FIG / "18_block_occupancy.png", dpi=130)
    plt.close(fig)

    # 19 temporal sampling
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    hist_path = CACHE / "date_histogram.parquet"
    if hist_path.exists():
        h = pd.read_parquet(hist_path)
        for c in COUNTRIES:
            s = h[h.country == c].sort_values("date")
            if s.empty:
                continue
            axes[0].plot(s["date"], s["n_observations"], lw=0.9, color=PALETTE[c], label=c)
        axes[0].set_yscale("log")
        axes[0].set_ylabel("observations (log scale)")
        axes[0].set_xlabel("acquisition date, 2021")
        axes[0].legend(fontsize=8)
        axes[0].set_title("Cloud-free acquisitions per date")
        axes[0].tick_params(axis="x", labelrotation=30)
    mcols = [f"m{i:02d}" for i in range(1, 13)]
    mat = np.vstack([[100 * float((clean.loc[clean.country == c, m] > 0).mean()) for m in mcols]
                     for c in COUNTRIES])
    sns.heatmap(mat, annot=True, fmt=".0f", cmap="YlGnBu", vmin=0, vmax=100,
                xticklabels=["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug",
                             "Sep", "Oct", "Nov", "Dec"],
                yticklabels=COUNTRIES, ax=axes[1], cbar_kws={"label": "% of parcels"})
    axes[1].set_title("Parcels with at least one observation in each month")
    plt.tight_layout()
    fig.savefig(FIG / "19_temporal_sampling.png", dpi=130)
    plt.close(fig)
    print("figures 12 to 19 written", flush=True)


# --------------------------------------------------------------------------- #
def _write_scheme(out: dict, retained: dict, MIN_CLASS: int, MIN_TS: int, K_MAX: int) -> None:
    import yaml

    CONFIG_OUT.parent.mkdir(parents=True, exist_ok=True)
    classes: dict = {}
    for c in COUNTRIES:
        for r in retained[c]:
            entry = classes.setdefault(r["hcat"], {
                "hcat": r["hcat"],
                "crop_name": r["crop_name"],
                "hcat_level": _hcat.hcat_level(r["hcat"]),
                "thesis_class": r["crop_name"].replace(" ", "_"),
                "parent_level4": _hcat.roll_up(r["hcat"], 4),
                "parent_level4_name": _hcat.name_of(_hcat.roll_up(r["hcat"], 4)),
                "countries": {}})
            entry["countries"][c] = r["n"]

    doc = {
        "name": "eurocropsml_thesis_class_scheme",
        "version": 1,
        "generated_by": "results/eda/_class_design.py",
        "source": {
            "catalogue": out["source_catalogue"],
            "n_parcels_scanned": out["n_parcels"],
            "is_full_dataset": True,
            "hcat_version": "HCAT2, EuroCrops (maja601/EuroCrops, hcat_core/HCAT2.csv)"},
        "random_seed": SEED,
        "protocol": {
            "K_grid_samples_per_class": K_GRID,
            "K_grid_extended_estonia_latvia_only": K_GRID_EXTENDED,
            "K_definition": ("absolute number of labelled parcels per class, never a "
                             "percentage of the available labels"),
            "n_draws_per_K": N_DRAWS,
            "n_test_min_per_class": N_TEST_MIN,
            "n_val_min_per_class": N_VAL_MIN,
            "train_pool_multiplier": POOL_MULTIPLIER,
            "K_max_supported": K_MAX},
        "aggregation": {
            "in_country": "native HCAT level as published, no roll-up",
            "seasonal_variants": ("kept separate; winter and spring variants of the same species "
                                  "differ in phenology rather than in species, which is the "
                                  "discriminative signal the benchmark is meant to test"),
            "cross_country_fallback_level": 4},
        "exclusions": {
            "min_parcels_per_class_per_country": MIN_CLASS,
            "max_classes_per_country": MAX_CLASSES_PER_COUNTRY,
            "min_timesteps": MIN_TS,
            "drop_outside_national_bbox": True,
            "national_bbox_lon_min_lon_max_lat_min_lat_max": {k: list(v) for k, v in BBOX.items()},
            "non_crop_and_catch_all_codes": [
                {"hcat": k, "crop_name": _hcat.name_of(k).replace("_", " "), "reason": v}
                for k, v in NON_CROP_CODES.items()],
            "excluded_branches_by_prefix": [
                {"prefix": k, "reason": v} for k, v in EXCLUDE_PREFIXES.items()],
            "borderline_may_be_reinstated": [
                {"hcat": k, "crop_name": _hcat.name_of(k).replace("_", " "), "note": v}
                for k, v in BORDERLINE_CODES.items()]},
        "classes": list(classes.values()),
        "in_country_class_lists": {c: [r["hcat"] for r in retained[c]] for c in COUNTRIES},
        "in_country_class_counts": {c: len(retained[c]) for c in COUNTRIES},
        "cross_country_shared": {
            "Estonia|Latvia": [r["hcat"] for r in out["shared_pairwise"]["Estonia|Latvia"]],
            "Estonia|Portugal": [r["hcat"] for r in out["shared_pairwise"]["Estonia|Portugal"]],
            "Latvia|Portugal": [r["hcat"] for r in out["shared_pairwise"]["Latvia|Portugal"]],
            "Estonia|Latvia|Portugal": [r["hcat"] for r in out["shared_three_way"]]},
        "cross_country_shared_counts": {
            **out["shared_pairwise_counts"],
            "Estonia|Latvia|Portugal": len(out["shared_three_way"])},
        "cross_country_shared_level4": {
            "Estonia|Latvia|Portugal": [r["hcat"] for r in out["level4_shared_three_way"]],
            "counts": {**out["level4_shared_pairwise_counts"],
                       "Estonia|Latvia|Portugal": len(out["level4_shared_three_way"])}},
    }
    CONFIG_OUT.write_text(yaml.safe_dump(doc, sort_keys=False, allow_unicode=True),
                          encoding="utf-8")
    print(f"wrote {CONFIG_OUT}", flush=True)


if __name__ == "__main__":
    main()
