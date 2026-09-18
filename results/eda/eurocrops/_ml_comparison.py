"""EuroCrops vector release against the EuroCropsML benchmark.

Two artefacts describe the same 2021 declarations and the thesis uses both, so what
matters is exactly how they differ: which parcels survive into EuroCropsML, whether the
identifiers join, whether the class labels agree, and whether the geometries that
EuroCropsML ships are the EuroCrops geometries.

Sources
-------
EuroCrops v11, Zenodo 14094196, via ``data/eurocrops/parquet`` and the per-parcel metrics
in ``data/eurocrops/derived`` written by ``_vector_analysis.py``.
EuroCropsML, Zenodo 15095445, via ``data/eurocropsml``:
    ``raw_data/labels/<Country>_labels.parquet``     parcel_id, EC_hcat_c, EC_hcat_n
    ``raw_data/geometries/<Country>.geojson``        parcel_id and polygon, CRS84
    ``preprocess/<NUTS3>_<parcelID>_<EC_hcat_c>.npz``  the ready-to-use time series
    ``split/<use case>/...``                         the official benchmark splits

Outputs
-------
``results/eda/eurocrops/comparison.json``      every number quoted by the notebook
``results/eda/eurocrops/cache/ml_*.parquet``   small aggregated tables
``results/eda/eurocrops/figures/1[3-8]_*.png`` figures 13 to 18

The geometry agreement is measured on a seeded sample per country, stated in the output;
everything else is computed over all parcels.

Usage
-----
    python results/eda/eurocrops/_ml_comparison.py [--iou-sample 5000]
"""

from __future__ import annotations

import argparse
import json
import os
import re
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
sys.path.insert(0, str(HERE))
from _vector_analysis import (  # noqa: E402
    COUNTRIES, DERIVED_DIR, EQUAL_AREA, ORDER, PALETTE, PARQUET_DIR, PIXEL_M, SEED, YEAR,
    normalise_id,
)

ML_ROOT = Path(os.environ.get("EUROCROPSML_DATA", REPO / "data" / "eurocropsml")).expanduser()
PREPROCESS_DIR = ML_ROOT / "preprocess"
RAW_DIR = ML_ROOT / "raw_data"
SPLIT_DIR = ML_ROOT / "split"

FIG_DIR = HERE / "figures"
CACHE_DIR = HERE / "cache"
OUT = HERE / "comparison.json"

#: EuroCropsML covers three of the four countries downloaded from EuroCrops.
ML_COUNTRIES = {"EE": "Estonia", "LV": "Latvia", "PT": "Portugal"}
#: Candidate join keys in the EuroCrops attribute table, tried in order.
ID_CANDIDATES = {
    "EE": ["pollu_id"],
    "LV": ["PARCEL_ID", "OBJECTID"],
    "PT": ["OSA_ID", "PAR_ID", "OSA_NUM"],
}

sns.set_theme(context="notebook", style="whitegrid")


# ------------------------------------------------------------------------ ml inventory


PREPROCESS_ZIP = ML_ROOT / "archives" / "preprocess.zip"


def _preprocess_names() -> tuple[list[str], str]:
    """The .npz filenames, taken from the archive index when it is available.

    The 706,683 files are small, so unpacking them onto network storage is slow and is not
    required to characterise the benchmark: the archive's central directory carries every
    filename, and the filename is what encodes the region, the parcel and the class.
    """
    import zipfile

    if PREPROCESS_ZIP.exists():
        with zipfile.ZipFile(PREPROCESS_ZIP) as zf:
            return [Path(n).name for n in zf.namelist() if n.endswith(".npz")], "preprocess.zip"
    if PREPROCESS_DIR.is_dir():
        return [e.name for e in os.scandir(PREPROCESS_DIR) if e.name.endswith(".npz")], "preprocess/"
    raise FileNotFoundError(f"neither {PREPROCESS_ZIP} nor {PREPROCESS_DIR} is present")


def scan_preprocess() -> pd.DataFrame:
    """Parse the .npz filenames into NUTS3 region, parcel id and HCAT code."""
    # 706,683 rows, so it lives with the data rather than in the tracked results tree
    cache = ML_ROOT / "preprocess_index.parquet"
    if cache.exists():
        return pd.read_parquet(cache)
    names, source = _preprocess_names()
    print(f"  reading {len(names):,} .npz names from {source}", flush=True)
    pat = re.compile(r"^([A-Z]{2}[0-9A-Z]*)_(\d+)_(\d+)\.npz$")
    nuts, pid, hcat = [], [], []
    for name in names:
        m = pat.match(name)
        if m:
            nuts.append(m.group(1))
            pid.append(m.group(2))
            hcat.append(m.group(3))
    df = pd.DataFrame({"nuts3": nuts, "parcel_id": pid, "hcat": hcat})
    df["cc"] = df["nuts3"].str[:2]
    df["country"] = df["cc"].map(COUNTRIES)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    return df


def ml_labels(cc: str) -> pd.DataFrame:
    df = pd.read_parquet(RAW_DIR / "labels" / f"{ML_COUNTRIES[cc]}_labels.parquet")
    df["parcel_id"] = normalise_id(df["parcel_id"])
    df["EC_hcat_c"] = normalise_id(df["EC_hcat_c"])
    return df


def ec_metrics(cc: str) -> pd.DataFrame:
    return pd.read_parquet(DERIVED_DIR / f"{cc}_{YEAR}_metrics.parquet")


def best_join_key(cc: str, ml_ids: set[str]) -> tuple[str, float, dict]:
    """The EuroCrops attribute column whose values overlap the EuroCropsML parcel ids."""
    import geopandas as gpd

    tried = {}
    best, best_rate = None, -1.0
    for col in ID_CANDIDATES[cc]:
        try:
            vals = gpd.read_parquet(PARQUET_DIR / f"{cc}_{YEAR}.parquet", columns=[col, "geometry"])[col]
        except Exception as exc:  # column absent in this country's schema
            tried[col] = f"unavailable ({type(exc).__name__})"
            continue
        s = set(normalise_id(vals))
        rate = len(s & ml_ids) / max(len(ml_ids), 1)
        tried[col] = round(100 * rate, 2)
        if rate > best_rate:
            best, best_rate = col, rate
    return best, best_rate, tried


# ---------------------------------------------------------------------------- figures


def fig13_counts(findings: dict) -> pd.DataFrame:
    idx = scan_preprocess()
    rows = []
    for cc, country in ML_COUNTRIES.items():
        ec = ec_metrics(cc)
        lab = ml_labels(cc)
        npz = idx[idx["cc"] == cc]
        rows.append({
            "country": country,
            "eurocrops_parcels": len(ec),
            "eurocropsml_raw_labels": len(lab),
            "eurocropsml_npz": len(npz),
            "eurocrops_classes": ec["hcat"].nunique(),
            "eurocropsml_classes": lab["EC_hcat_c"].nunique(),
            "npz_classes": npz["hcat"].nunique(),
            "nuts3_regions": npz["nuts3"].nunique(),
        })
    tbl = pd.DataFrame(rows).set_index("country")
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.0))
    x = np.arange(len(tbl))
    for i, (col, lbl) in enumerate((("eurocrops_parcels", "EuroCrops vector"),
                                    ("eurocropsml_raw_labels", "EuroCropsML raw labels"),
                                    ("eurocropsml_npz", "EuroCropsML .npz"))):
        axes[0].bar(x + (i - 1) * 0.27, tbl[col], width=0.27, label=lbl)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(tbl.index)
    axes[0].set_ylabel("parcels")
    axes[0].set_title("parcels per country, by source")
    axes[0].legend(fontsize=8)
    for i, (col, lbl) in enumerate((("eurocrops_classes", "EuroCrops vector"),
                                    ("eurocropsml_classes", "EuroCropsML raw labels"),
                                    ("npz_classes", "EuroCropsML .npz"))):
        axes[1].bar(x + (i - 1) * 0.27, tbl[col], width=0.27, label=lbl)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(tbl.index)
    axes[1].set_ylabel("distinct HCAT classes")
    axes[1].set_title("class count per country, by source")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "13_source_inventory.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    tbl.to_parquet(CACHE_DIR / "ml_source_inventory.parquet")
    findings["source_inventory"] = json.loads(tbl.to_json(orient="index"))
    findings["npz_total"] = int(len(idx))
    return tbl


def fig14_join(findings: dict) -> dict:
    """Do the identifiers join, and what does each source hold that the other does not?"""
    idx = scan_preprocess()
    rows, keys = [], {}
    for cc, country in ML_COUNTRIES.items():
        lab = ml_labels(cc)
        ml_ids = set(lab["parcel_id"])
        key, rate, tried = best_join_key(cc, ml_ids)
        keys[country] = {"join_key": key, "match_rate_pct": round(100 * rate, 2), "candidates": tried}
        ec = ec_metrics(cc)
        ec_ids = set(ec["parcel_id"])
        npz_ids = set(idx.loc[idx["cc"] == cc, "parcel_id"])
        rows.append({
            "country": country,
            "in_both": len(ec_ids & ml_ids),
            "eurocrops_only": len(ec_ids - ml_ids),
            "eurocropsml_only": len(ml_ids - ec_ids),
            "npz_not_in_eurocrops": len(npz_ids - ec_ids),
            "eurocrops_with_npz": len(ec_ids & npz_ids),
        })
    tbl = pd.DataFrame(rows).set_index("country")
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    tbl[["in_both", "eurocrops_only", "eurocropsml_only"]].plot(
        kind="barh", stacked=True, ax=ax, color=["#4c72b0", "#dd8452", "#55a868"])
    ax.set_xlabel("parcels")
    ax.set_title("identifier join between the two releases")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "14_identifier_join.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    tbl.to_parquet(CACHE_DIR / "ml_identifier_join.parquet")
    findings["identifier_join"] = json.loads(tbl.to_json(orient="index"))
    findings["join_keys"] = keys
    return keys


def fig15_classes(findings: dict) -> None:
    """Per-class parcel counts in the two releases, and the label agreement on the join."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    agreement, exclusives, mismatch_frames = {}, {}, []
    for cc, country in ML_COUNTRIES.items():
        ec = ec_metrics(cc)
        lab = ml_labels(cc)
        a = ec.groupby("hcat").size().rename("eurocrops")
        b = lab.groupby("EC_hcat_c").size().rename("eurocropsml")
        m = pd.concat([a, b], axis=1).fillna(0)
        axes[0].scatter(m["eurocrops"] + 0.5, m["eurocropsml"] + 0.5, s=14, alpha=0.6,
                        color=PALETTE[country], label=country)
        exclusives[country] = {
            "classes_only_in_eurocrops": int(((m["eurocrops"] > 0) & (m["eurocropsml"] == 0)).sum()),
            "classes_only_in_eurocropsml": int(((m["eurocrops"] == 0) & (m["eurocropsml"] > 0)).sum()),
            "parcels_in_eurocrops_only_classes": int(m.loc[m["eurocropsml"] == 0, "eurocrops"].sum()),
        }
        # label agreement on the joined parcels
        j = ec[["parcel_id", "hcat"]].merge(
            lab[["parcel_id", "EC_hcat_c"]], on="parcel_id", how="inner")
        if len(j):
            same = (j["hcat"] == j["EC_hcat_c"]).mean()
            agreement[country] = {"joined_parcels": int(len(j)),
                                  "identical_hcat_pct": round(float(100 * same), 3)}
            mis = j[j["hcat"] != j["EC_hcat_c"]]
            if len(mis):
                pairs = (mis.merge(ec[["parcel_id", "hcat_name"]], on="parcel_id", how="left")
                         .merge(lab[["parcel_id", "EC_hcat_n"]], on="parcel_id", how="left")
                         .groupby(["hcat", "hcat_name", "EC_hcat_c", "EC_hcat_n"], observed=True)
                         .size().rename("parcels").reset_index()
                         .sort_values("parcels", ascending=False))
                pairs.insert(0, "country", country)
                mismatch_frames.append(pairs)
                agreement[country]["top_mismatches"] = [
                    {"eurocrops": f"{r.hcat_name} ({r.hcat})",
                     "eurocropsml": f"{r.EC_hcat_n} ({r.EC_hcat_c})",
                     "parcels": int(r.parcels)}
                    for r in pairs.head(8).itertuples()]
    lims = [0.5, 10 ** 6]
    axes[0].plot(lims, lims, color="grey", ls=":", lw=1)
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("parcels per class, EuroCrops vector")
    axes[0].set_ylabel("parcels per class, EuroCropsML")
    axes[0].set_title("per-class counts agree along the diagonal")
    axes[0].legend(fontsize=8)

    names = list(agreement)
    if names:
        axes[1].bar(names, [agreement[n]["identical_hcat_pct"] for n in names],
                    color=[PALETTE[n] for n in names])
        axes[1].set_ylim(0, 105)
        axes[1].set_ylabel("% of joined parcels with the same HCAT code")
        axes[1].set_title("label agreement on the joined parcels")
        for i, n in enumerate(names):
            axes[1].text(i, agreement[n]["identical_hcat_pct"],
                         f"{agreement[n]['identical_hcat_pct']:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "15_class_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["class_exclusives"] = exclusives
    findings["label_agreement"] = agreement
    if mismatch_frames:
        pd.concat(mismatch_frames, ignore_index=True).to_parquet(
            CACHE_DIR / "ml_label_mismatches.parquet", index=False)


def fig16_dropped(findings: dict) -> None:
    """Which EuroCrops parcels EuroCropsML does not carry, by size and by class."""
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.2))
    out = {}
    for cc, country in ML_COUNTRIES.items():
        ec = ec_metrics(cc)
        ml_ids = set(ml_labels(cc)["parcel_id"])
        kept = ec["parcel_id"].isin(ml_ids)
        for mask, ls, tag in ((kept, "-", "kept"), (~kept, "--", "dropped")):
            v = np.sort(ec.loc[mask, "n_pixels"].to_numpy())
            if len(v) < 10:
                continue
            step = max(1, len(v) // 20000)
            axes[0].plot(v[::step], (np.arange(1, len(v) + 1) / len(v))[::step],
                         ls=ls, lw=1.6, color=PALETTE[country],
                         label=f"{country}, {tag} ({len(v):,})")
        d = ec.loc[~kept]
        out[country] = {
            "dropped_parcels": int((~kept).sum()),
            "dropped_pct": round(float(100 * (~kept).mean()), 2),
            "median_pixels_kept": round(float(ec.loc[kept, "n_pixels"].median()), 1) if kept.any() else None,
            "median_pixels_dropped": round(float(d["n_pixels"].median()), 1) if len(d) else None,
            "top_dropped_classes": {str(k): int(v) for k, v in
                                    d["hcat_name"].value_counts().head(8).items()},
        }
    axes[0].set_xscale("log")
    axes[0].set_xlabel("parcel size (10 m pixels)")
    axes[0].set_ylabel("share at or below")
    axes[0].set_title("size of the parcels EuroCropsML keeps against those it drops")
    axes[0].legend(fontsize=7)

    names = list(out)
    axes[1].bar(names, [out[n]["dropped_pct"] for n in names], color=[PALETTE[n] for n in names])
    axes[1].set_ylabel("% of EuroCrops parcels absent from EuroCropsML")
    axes[1].set_title("attrition into the benchmark")
    for i, n in enumerate(names):
        axes[1].text(i, out[n]["dropped_pct"], f"{out[n]['dropped_pct']:.1f}%",
                     ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "16_dropped_parcels.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["attrition"] = out


def fig17_geometry(findings: dict, n_sample: int) -> None:
    """Are the EuroCropsML geometries the EuroCrops geometries? Measured by IoU."""
    import geopandas as gpd

    fig, ax = plt.subplots(figsize=(7.6, 4.2))
    out = {}
    for cc, country in ML_COUNTRIES.items():
        gj = RAW_DIR / "geometries" / f"{country}.geojson"
        if not gj.exists():
            out[country] = {"status": f"missing {gj.name}"}
            continue
        ml = gpd.read_file(gj, engine="pyogrio")
        ml["parcel_id"] = normalise_id(ml["parcel_id"]).to_numpy()
        ml = ml.sample(n=min(n_sample, len(ml)), random_state=SEED)

        ec = gpd.read_parquet(PARQUET_DIR / f"{cc}_{YEAR}.parquet",
                              columns=[ID_CANDIDATES[cc][0], "geometry"])
        ec["parcel_id"] = normalise_id(ec[ID_CANDIDATES[cc][0]]).to_numpy()
        ec = ec[ec["parcel_id"].isin(set(ml["parcel_id"]))]
        j = ml[["parcel_id", "geometry"]].merge(
            ec[["parcel_id", "geometry"]], on="parcel_id", suffixes=("_ml", "_ec"))
        if j.empty:
            out[country] = {"status": "no joined parcels in the sample"}
            continue
        a = gpd.GeoSeries(j["geometry_ml"], crs=ml.crs).to_crs(EQUAL_AREA).buffer(0)
        b = gpd.GeoSeries(j["geometry_ec"], crs=ec.crs).to_crs(EQUAL_AREA).buffer(0)
        inter = a.intersection(b, align=False).area.to_numpy()
        union = a.union(b, align=False).area.to_numpy()
        iou = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        ax.hist(iou, bins=np.linspace(0, 1, 51), histtype="step", lw=1.7,
                color=PALETTE[country],
                label=f"{country}: n={len(iou):,}, median {np.median(iou):.4f}, "
                      f"{100 * (iou > 0.99).mean():.1f} % above 0.99")
        out[country] = {
            "n_compared": int(len(iou)),
            "median_iou": round(float(np.median(iou)), 5),
            "share_iou_above_0.99_pct": round(float(100 * (iou > 0.99).mean()), 2),
            "share_iou_below_0.5_pct": round(float(100 * (iou < 0.5).mean()), 2),
            "median_area_ratio_ml_over_ec": round(float(np.median(a.area.to_numpy() /
                                                                 np.maximum(b.area.to_numpy(), 1e-9))), 5),
        }
        print(f"  {country}: IoU median {out[country]['median_iou']}", flush=True)
    # The distribution is degenerate at 1.0, so a linear count axis would hide the tail.
    ax.set_yscale("log")
    ax.set_xlabel("intersection over union, EuroCropsML polygon against EuroCrops polygon")
    ax.set_ylabel("parcels (log)")
    ax.set_title("do the two releases ship the same geometry?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "17_geometry_agreement.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    findings["geometry_agreement"] = out
    findings["geometry_agreement_sample"] = n_sample


def fig18_splits(findings: dict) -> None:
    """The official EuroCropsML splits: use cases, budget grid and fine-tuning sizes."""
    if not SPLIT_DIR.exists():
        findings["official_splits"] = {"status": "split/ not extracted"}
        return
    summary = {}
    for case in sorted(p for p in SPLIT_DIR.iterdir() if p.is_dir()):
        ks, sizes = [], {}
        for f in sorted((case / "finetune").glob("region_split_*.json")):
            tag = f.stem.replace("region_split_", "")
            ks.append(tag)
            try:
                d = json.loads(f.read_text())
                sizes[tag] = {k: len(v) for k, v in d.items()} if isinstance(d, dict) else len(d)
            except Exception as exc:
                sizes[tag] = f"unreadable ({type(exc).__name__})"
        summary[case.name] = {"budget_grid": ks, "finetune_sizes": sizes,
                              "has_pretrain": (case / "pretrain").exists(),
                              "has_meta": (case / "meta").exists()}
    findings["official_splits"] = summary

    rows = []
    for case, d in summary.items():
        for k, s in d["finetune_sizes"].items():
            if isinstance(s, dict):
                for part, n in s.items():
                    rows.append({"use_case": case, "K": k, "partition": part, "n": n})
    if not rows:
        return
    tbl = pd.DataFrame(rows)
    tbl.to_parquet(CACHE_DIR / "ml_official_splits.parquet", index=False)
    train = tbl[tbl["partition"].str.contains("train", case=False)]
    if train.empty:
        return
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    order = [k for k in ["1", "5", "10", "20", "50", "100", "200", "500", "all"]
             if k in set(train["K"])]
    for case, g in train.groupby("use_case"):
        g = g.set_index("K").reindex(order).reset_index()
        ax.plot(range(len(order)), g["n"], marker="o", ms=4, label=case)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(order)
    ax.set_yscale("log")
    ax.set_xlabel("official budget grid K (samples per class)")
    ax.set_ylabel("fine-tuning training parcels")
    ax.set_title("EuroCropsML official splits, fine-tuning pool against K")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "18_official_splits.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--iou-sample", type=int, default=5000)
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    findings: dict = {
        "eurocrops": "v11, Zenodo 14094196, declaration year 2021",
        "eurocropsml": "Zenodo 15095445, published 2025-03-31",
        "countries": ML_COUNTRIES,
        "random_seed": SEED,
        "pixel_m": PIXEL_M,
    }
    print("inventory", flush=True)
    fig13_counts(findings)
    print("identifier join", flush=True)
    fig14_join(findings)
    print("class agreement", flush=True)
    fig15_classes(findings)
    print("attrition", flush=True)
    fig16_dropped(findings)
    print("geometry agreement", flush=True)
    fig17_geometry(findings, args.iou_sample)
    print("official splits", flush=True)
    fig18_splits(findings)
    OUT.write_text(json.dumps(findings, indent=2))
    print(f"wrote {OUT.relative_to(REPO)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
