"""Sentinel-2 and Sentinel-1 phenology for single parcels of the main Estonian crops.

Pulls imagery on the fly with the vendored ``space_time_deepsearch`` loaders, reduces it
over one parcel polygon, and places the result beside the per-parcel time series that
EuroCropsML ships for the same parcel. Three questions are answered per crop:

1. What does the parcel look like through the season, date by date?
2. What does its own vegetation-index trajectory look like, computed here from L2A surface
   reflectance clipped to the polygon?
3. Does that trajectory agree with the EuroCropsML series, which is a spatial median over
   the same polygon but taken from **L1C top-of-atmosphere** reflectance?

The third question matters because Phase 1 treats EuroCropsML as the reference index while
the chips are exported independently; if the two disagree systematically, every comparison
against published EuroCropsML numbers inherits that offset.

Outputs
-------
``results/eda/eurocrops/gallery.json``                   every number quoted by the notebook
``results/eda/eurocrops/cache/gallery_*.parquet``        the selected parcels and their series
``results/eda/eurocrops/figures/gallery/*.png``          contact sheets and series figures
``data/eurocrops/gallery/*.gif``                         the timelapses, git-ignored

Usage
-----
    python results/eda/eurocrops/_parcel_gallery.py                 # all crops
    python results/eda/eurocrops/_parcel_gallery.py --crops potatoes --no-gif
"""

from __future__ import annotations

import argparse
import io as _io
import json
import sys
import time
import warnings
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import xarray as xr  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

# The corrections that must be identical everywhere - the processing-baseline reflectance
# offset, the cloud screening and the parcel mask - live in the package, so this script and
# gfm4agri.data.parcel_explorer cannot drift apart.
from gfm4agri.data.parcel_explorer import compare_series, shrink_gif  # noqa: E402
from gfm4agri.data.sentinel import (  # noqa: E402
    MIN_VALID_FRACTION, SCL_DROP, baseline_lookup, clear_fraction, ensure_vendored,
    eurocropsml_series, harmonise_boa, parcel_mask,
)

ensure_vendored()

FIG_DIR = HERE / "figures" / "gallery"
CACHE_DIR = HERE / "cache"
OUT = HERE / "gallery.json"
GIF_DIR = REPO / "data" / "eurocrops" / "gallery"

PARQUET_DIR = REPO / "data" / "eurocrops" / "parquet"
DERIVED_DIR = REPO / "data" / "eurocrops" / "derived"
ML_ROOT = REPO / "data" / "eurocropsml"

SEED = 42
COUNTRY, YEAR = "EE", 2021
ID_COL = "pollu_id"
SEASON = ("2021-03-01", "2021-11-15")
#: B05 and B11 are 20 m bands, resampled to 10 m by stackstac; they carry the red edge and
#: the SWIR that NDRE and NDMI need. SCL drives the per-pixel cloud screening.
BANDS = ["B02", "B03", "B04", "B05", "B08", "B11", "SCL"]
BUFFER_M = 250  # context ring around the parcel, for the timelapse only
CLOUD_MAX, MIN_COVERAGE = 70, 80
#: A frame is shown in the timelapse only if this much of the parcel is clear.
FRAME_MIN_CLEAR = 0.60
#: The contact sheet is stricter than the animation: it has twelve slots and should spend
#: them on dates that actually show the field.
SHEET_MIN_CLEAR = 0.85
#: True-colour display stretch, in DN on the DN / 10000 convention.
TRUE_COLOUR_MAX, TRUE_COLOUR_GAMMA = 3000.0, 0.75

#: The thirteen L1C bands EuroCropsML stores, in the order of its ``data`` array.
ML_BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10", "B11", "B12"]

CROPS = {
    "winter_common_soft_wheat": "winter wheat",
    "spring_barley": "spring barley",
    "winter_rapeseed_rape": "winter rapeseed",
    "potatoes": "potatoes",
    "pasture_meadow_grassland_grass": "grassland",
    "oats": "oats",
}
#: The crop that also gets a Sentinel-1 panel and an NDVI animation.
SHOWCASE = "winter_common_soft_wheat"

CROP_COLOR = {
    "winter_common_soft_wheat": "#1f77b4",
    "spring_barley": "#ff7f0e",
    "winter_rapeseed_rape": "#d62728",
    "potatoes": "#9467bd",
    "pasture_meadow_grassland_grass": "#2ca02c",
    "oats": "#8c564b",
}
NDVI_RAMP = ["#8c510a", "#d8b365", "#f6e8c3", "#c7eae5", "#5ab4ac", "#01665e"]

sns.set_theme(context="notebook", style="whitegrid")


# ------------------------------------------------------------------- parcel selection


def select_parcels(crops: list[str], min_ha: float = 8.0, max_ha: float = 25.0,
                   min_compactness: float = 0.65) -> pd.DataFrame:
    """One parcel per crop: compact, mid-sized, and labelled identically in both releases.

    Compactness and size are selection criteria rather than findings: a ragged two hectare
    parcel would show mostly mixed pixels and would say more about the boundary than about
    the crop.
    """
    cache = CACHE_DIR / "gallery_parcels.parquet"
    if cache.exists():
        got = pd.read_parquet(cache)
        if set(crops) <= set(got["crop"]):
            return got[got["crop"].isin(crops)].reset_index(drop=True)

    ec = pd.read_parquet(DERIVED_DIR / f"{COUNTRY}_{YEAR}_metrics.parquet")
    ml = pd.read_parquet(ML_ROOT / "raw_data" / "labels" / "Estonia_labels.parquet")
    ml["parcel_id"] = ml["parcel_id"].astype("int64").astype(str)
    j = ec.merge(ml.rename(columns={"EC_hcat_c": "ml_hcat"})[["parcel_id", "ml_hcat"]],
                 on="parcel_id", how="inner")
    j = j[j["hcat"] == j["ml_hcat"]]  # only parcels the two releases agree on

    rows = []
    for crop in crops:
        g = j[(j["hcat_name"] == crop)
              & j["area_m2"].between(min_ha * 1e4, max_ha * 1e4)
              & (j["compactness"] > min_compactness)]
        if g.empty:
            print(f"  no candidate parcel for {crop}", flush=True)
            continue
        r = g.sort_values(["compactness", "area_m2"], ascending=False).iloc[0]
        rows.append({"crop": crop, "label": CROPS.get(crop, crop), "parcel_id": r["parcel_id"],
                     "hcat": r["hcat"], "area_ha": round(r["area_m2"] / 1e4, 2),
                     "n_pixels": int(r["n_pixels"]), "compactness": round(r["compactness"], 3),
                     "edge_share": round(r["edge_share"], 3), "candidates": int(len(g))})
    out = pd.DataFrame(rows)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(cache, index=False)
    return out


def parcel_geometry(parcel_id: str):
    import geopandas as gpd

    gdf = gpd.read_parquet(PARQUET_DIR / f"{COUNTRY}_{YEAR}.parquet",
                           columns=[ID_COL, "EC_hcat_n", "geometry"])
    hit = gdf[gdf[ID_COL].astype("int64").astype(str) == str(parcel_id)]
    if hit.empty:
        raise KeyError(f"parcel {parcel_id} not found in the {COUNTRY} layer")
    return hit.iloc[[0]]


# ------------------------------------------------------------------------ imagery pull


def fetch_cube(geom_4326, start: str, end: str):
    """The Sentinel-2 L2A cube for one area of interest, pulled on the fly from STAC."""
    from space_time_deepsearch.io import get_sentinel2_imagery

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return get_sentinel2_imagery(
            custom_geometry=geom_4326, start_date=start, end_date=end, bands=BANDS,
            resolution=10, cloud_cover_max=CLOUD_MAX, min_coverage=MIN_COVERAGE)


# ------------------------------------------------------------------ EuroCropsML series


# ---------------------------------------------------------------------------- figures


def contact_sheet(cube, mask, geom_utm, crop: str, label: str, clear: np.ndarray,
                  n_cols: int = 6, n_rows: int = 2) -> Path:
    """A grid of true-colour thumbnails through the season, each titled with its date."""
    keep = np.where(clear >= SHEET_MIN_CLEAR)[0]
    if len(keep) < n_cols * n_rows:
        keep = np.where(clear >= FRAME_MIN_CLEAR)[0]
    if len(keep) == 0:
        keep = np.arange(len(cube.time))
    pick = keep[np.linspace(0, len(keep) - 1, min(n_cols * n_rows, len(keep))).astype(int)]

    rgb = cube.sel(band=["B04", "B03", "B02"]).isel(time=pick)
    # A fixed stretch, 0 to 0.30 reflectance with a gamma lift, rather than a percentile
    # stretch: a percentile taken over the whole cube is dominated by the cloudy frames and
    # renders every field almost black, and a per-frame percentile would make the dates
    # incomparable, which is the one thing a phenology contact sheet must not do.
    img = (rgb / TRUE_COLOUR_MAX).clip(0, 1) ** TRUE_COLOUR_GAMMA
    img = img.transpose("time", "y", "x", "band").values

    xs, ys = geom_utm.exterior.xy
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.1 * n_cols, 2.55 * n_rows),
                             layout="constrained")
    for ax, k, frame in zip(np.ravel(axes), pick, img):
        ax.imshow(frame, extent=[float(cube.x.min()), float(cube.x.max()),
                                 float(cube.y.min()), float(cube.y.max())])
        ax.plot(xs, ys, color="#ffd166", lw=1.1)
        ax.set_title(f"{str(cube.time.values[k])[:10]}\n{100 * clear[k]:.0f} % clear", fontsize=7.5)
        ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    for ax in np.ravel(axes)[len(pick):]:
        ax.axis("off")
    fig.suptitle(f"{label} ({crop}), Estonia 2021, Sentinel-2 L2A true colour, "
                 f"fixed stretch 0 to {TRUE_COLOUR_MAX / 1e4:.2f} reflectance", fontsize=11)
    # A contact sheet is twelve photographs, so JPEG at high quality is a fifth of the
    # size of the equivalent PNG and the difference is invisible. The plots stay PNG.
    path = FIG_DIR / f"{crop}_contact.jpg"
    fig.savefig(path, dpi=140, bbox_inches="tight", pil_kwargs={"quality": 88, "optimize": True})
    plt.close(fig)
    return path


def series_figure(series: dict, ml: pd.DataFrame, crop: str, label: str, stats: dict) -> Path:
    """Our L2A indices, the EuroCropsML L1C NDVI, and the date-matched difference."""
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.0),
                             gridspec_kw={"width_ratios": [1.25, 1.25, 1]})
    for name, s in series.items():
        axes[0].plot(s["date"], s["median"], marker="o", ms=2.6, lw=1.3, label=name)
    axes[0].set_title(f"{label}: vegetation indices, this pull (L2A)", fontsize=10)
    axes[0].set_ylabel("index, median over the parcel")
    axes[0].legend(fontsize=7, ncol=3)
    axes[0].tick_params(axis="x", rotation=30)

    ndvi = series["NDVI"]
    axes[1].plot(ndvi["date"], ndvi["median"], marker="o", ms=3.4, lw=1.5,
                 color=CROP_COLOR.get(crop, "#1f77b4"), label=f"here, L2A ({len(ndvi)} dates)")
    axes[1].plot(ml["date"], ml["NDVI"], marker="s", ms=3.0, lw=1.2, ls="--", color="#555555",
                 label=f"EuroCropsML, L1C ({len(ml)} dates)")
    axes[1].set_title("NDVI: this pull against EuroCropsML", fontsize=10)
    axes[1].set_ylabel("NDVI")
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)

    a = ndvi[["date", "median"]].rename(columns={"median": "l2a"})
    b = ml[["date", "NDVI"]].rename(columns={"NDVI": "l1c"})
    a["date"] = pd.to_datetime(a["date"]).dt.tz_localize(None).dt.as_unit("ns").dt.normalize()
    b["date"] = pd.to_datetime(b["date"]).dt.tz_localize(None).dt.as_unit("ns").dt.normalize()
    m = pd.merge_asof(a.sort_values("date"), b.sort_values("date"), on="date",
                      tolerance=pd.Timedelta(days=1), direction="nearest").dropna()
    axes[2].scatter(m["l1c"], m["l2a"], s=18, color=CROP_COLOR.get(crop, "#1f77b4"), alpha=0.8)
    axes[2].plot([0, 1], [0, 1], color="grey", ls=":", lw=1)
    axes[2].set_xlim(0, 1); axes[2].set_ylim(0, 1)
    axes[2].set_xlabel("EuroCropsML NDVI (L1C)")
    axes[2].set_ylabel("NDVI here (L2A)")
    txt = (f"n={stats.get('matched_dates', 0)}  r={stats.get('pearson_r', float('nan')):.3f}\n"
           f"mean diff {stats.get('mean_difference', float('nan')):+.3f}  "
           f"RMSE {stats.get('rmse', float('nan')):.3f}")
    axes[2].set_title(txt, fontsize=9)
    fig.tight_layout()
    path = FIG_DIR / f"{crop}_series.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def overview_figure(all_series: dict, all_ml: dict) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.4), sharey=True)
    for crop, s in all_series.items():
        axes[0].plot(s["NDVI"]["date"], s["NDVI"]["median"], lw=1.6,
                     color=CROP_COLOR.get(crop), label=CROPS.get(crop, crop))
    axes[0].set_title("NDVI pulled here from Sentinel-2 L2A, clipped to the parcel", fontsize=10)
    axes[0].set_ylabel("NDVI, median over the parcel")
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)
    for crop, ml in all_ml.items():
        axes[1].plot(ml["date"], ml["NDVI"], lw=1.6, ls="--", color=CROP_COLOR.get(crop),
                     label=CROPS.get(crop, crop))
    axes[1].set_title("NDVI as EuroCropsML ships it, from L1C top of atmosphere", fontsize=10)
    axes[1].legend(fontsize=8)
    axes[1].tick_params(axis="x", rotation=30)
    fig.suptitle("Six Estonian crops, 2021, one parcel each", y=1.02)
    fig.tight_layout()
    path = FIG_DIR / "00_overview_ndvi.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def timelapse(cube, clear: np.ndarray, crop: str, kind: str = "rgb") -> Path | None:
    """A GIF with the acquisition date burned into each frame, via the vendored viz module."""
    from space_time_deepsearch.vis import create_timelapse

    keep = np.where(clear >= FRAME_MIN_CLEAR)[0]
    if len(keep) < 4:
        return None
    GIF_DIR.mkdir(parents=True, exist_ok=True)
    path = GIF_DIR / f"{crop}_{kind}.gif"
    sub = cube.isel(time=keep)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if kind == "rgb":
            create_timelapse(sub, output_path=str(path), fps=3, bands=["B04", "B03", "B02"],
                             vmin=0.0, vmax=TRUE_COLOUR_MAX, add_text=True,
                             figsize=(5, 5), dpi=90)
        else:
            red = sub.sel(band="B04").astype("float32")
            nir = sub.sel(band="B08").astype("float32")
            with np.errstate(divide="ignore", invalid="ignore"):
                ndvi = ((nir - red) / (nir + red)).where(lambda a: np.isfinite(a))
            cmap = matplotlib.colors.LinearSegmentedColormap.from_list("ndvi", NDVI_RAMP)
            create_timelapse(ndvi, output_path=str(path), fps=3, vmin=-0.1, vmax=0.95,
                             cmap=cmap, add_text=True, figsize=(5, 5), dpi=90)
    return path


def sentinel1_panel(geom_4326, geom_utm, ndvi: pd.DataFrame, crop: str, label: str) -> tuple:
    """Sentinel-1 RTC backscatter over the same parcel, beside the optical NDVI."""
    from space_time_deepsearch.io import (compute_features, get_sentinel1_rtc_imagery,
                                          normalize_orbit)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # min_coverage is measured over the bounding box of the area of interest, and a
        # parcel polygon fills only part of its own bounding box, so any non-zero threshold
        # rejects every date. With a polygon AOI the test belongs at zero.
        cube = get_sentinel1_rtc_imagery(custom_geometry=geom_4326, start_date=SEASON[0],
                                         end_date=SEASON[1], resolution=10, min_coverage=0.0,
                                         verbose=False)
        feat = compute_features(cube)
        # Four tracks see this parcel from four geometries, each with its own constant
        # offset; without removing it the series is a sawtooth that says nothing about the
        # crop. normalize_orbit adds the corrected <col>_adj columns.
        feat = normalize_orbit(feat, verbose=False)

    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    for col, colour, name in (("vv_median_db", "#1f77b4", "VV"),
                              ("vh_median_db", "#d62728", "VH")):
        if col in feat:
            ax.scatter(feat["date"], feat[col], s=9, color=colour, alpha=0.35,
                       label=f"{name}, as acquired")
        adj = f"{col}_adj"
        if adj in feat:
            ax.plot(feat["date"], feat[adj], lw=1.4, color=colour,
                    label=f"{name}, track offset removed")
    ax.set_ylabel("gamma0 (dB), median over the parcel")
    ax.tick_params(axis="x", rotation=30)
    ax2 = ax.twinx()
    ax2.plot(ndvi["date"], ndvi["median"], lw=2.0, color="#2ca02c", alpha=0.75, label="NDVI (S2 L2A)")
    ax2.set_ylabel("NDVI")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    ax.set_title(f"{label}: Sentinel-1 RTC backscatter and Sentinel-2 NDVI, same parcel, 2021",
                 fontsize=10)
    fig.tight_layout()
    path = FIG_DIR / f"{crop}_sentinel1.png"
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path, feat


# ------------------------------------------------------------------------------- main


def run_crop(row: pd.Series, make_gif: bool) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    import geopandas as gpd

    from space_time_deepsearch.vis.parcel_timeseries import compute_indices

    crop, label, pid = row["crop"], row["label"], row["parcel_id"]
    print(f"\n=== {label} ({crop}), parcel {pid}, {row['area_ha']} ha ===", flush=True)
    hit = parcel_geometry(pid)
    geom_4326 = hit.geometry.iloc[0]
    # A buffered bounding box, not a buffered polygon: the frames then come out
    # rectangular and the parcel sits in its landscape context.
    from shapely.geometry import box as _box

    aoi = gpd.GeoSeries([_box(*hit.to_crs(3035).buffer(BUFFER_M).total_bounds)],
                        crs=3035).to_crs(4326).iloc[0]

    t0 = time.time()
    cube = fetch_cube(aoi, *SEASON)
    geom_utm = hit.to_crs(int(cube.epsg)).geometry.iloc[0]
    mask = parcel_mask(cube, geom_utm)
    clear = clear_fraction(cube, mask)
    print(f"  cube {dict(cube.sizes)} in {time.time() - t0:.0f}s, "
          f"{int((clear >= FRAME_MIN_CLEAR).sum())} dates at least "
          f"{100 * FRAME_MIN_CLEAR:.0f} % clear over the parcel", flush=True)

    cube_h, boa = harmonise_boa(cube, fallback=baseline_lookup(aoi, *SEASON))
    print(f"  processing baselines {boa['baselines']}, "
          f"{boa.get('dates_offset_corrected', 0)} dates offset-corrected", flush=True)
    raw_series = compute_indices(cube_h.where(mask), offset=0.0, scl_drop=SCL_DROP)
    inside = int(mask.sum())
    min_px = MIN_VALID_FRACTION * inside
    series = {k: v[v["n_pixels"] >= min_px].reset_index(drop=True) for k, v in raw_series.items()}
    dropped = len(raw_series["NDVI"]) - len(series["NDVI"])
    print(f"  screening: {dropped} of {len(raw_series['NDVI'])} dates dropped for having less "
          f"than {100 * MIN_VALID_FRACTION:.0f} % of the parcel clear", flush=True)
    ml = eurocropsml_series(pid)
    stats = compare_series(series["NDVI"], ml)
    # What does EuroCropsML carry on the dates this screening rejects? Those dates are in its
    # series too, since it ships one row per acquisition; comparing the two populations says
    # how much of its own noise is residual cloud rather than crop.
    kept_days = set(pd.to_datetime(series["NDVI"]["date"]).dt.normalize())
    all_days = set(pd.to_datetime(raw_series["NDVI"]["date"]).dt.normalize())
    ml_days = pd.to_datetime(ml["date"]).dt.normalize()
    on_kept = ml.loc[ml_days.isin(kept_days), "NDVI"]
    on_rejected = ml.loc[ml_days.isin(all_days - kept_days), "NDVI"]
    stats["eurocropsml_on_dates_we_keep"] = {
        "n": int(len(on_kept)), "median_ndvi": round(float(on_kept.median()), 4) if len(on_kept) else None}
    stats["eurocropsml_on_dates_we_reject"] = {
        "n": int(len(on_rejected)),
        "median_ndvi": round(float(on_rejected.median()), 4) if len(on_rejected) else None}
    print(f"  NDVI: {stats.get('matched_dates')} matched dates, r={stats.get('pearson_r')}, "
          f"mean difference {stats.get('mean_difference')}", flush=True)

    contact_sheet(cube_h, mask, geom_utm, crop, label, clear)
    series_figure(series, ml, crop, label, stats)
    gif = timelapse(cube_h, clear, crop, "rgb") if make_gif else None
    if make_gif and crop == SHOWCASE:
        ndvi_gif = timelapse(cube_h, clear, crop, "ndvi")
        if ndvi_gif:
            shrink_gif(ndvi_gif)

    record = {
        "crop": crop, "label": label, "parcel_id": pid, "hcat": row["hcat"],
        "area_ha": row["area_ha"], "n_pixels": int(row["n_pixels"]),
        "compactness": row["compactness"], "edge_share": row["edge_share"],
        "candidates_considered": int(row["candidates"]),
        "nuts3": str(ml["nuts3"].iloc[0]),
        "boa_harmonisation": boa,
        "scenes_retained": int(len(cube.time)),
        "scenes_clear_over_parcel": int((clear >= FRAME_MIN_CLEAR).sum()),
        "dates_before_screening": int(len(raw_series["NDVI"])),
        "dates_after_screening": int(len(series["NDVI"])),
        "scl_drop": list(SCL_DROP),
        "min_valid_fraction": MIN_VALID_FRACTION,
        "median_clear_fraction": round(float(np.median(clear)), 3),
        "ndvi_peak_here": round(float(series["NDVI"]["median"].max()), 3),
        "ndvi_peak_date_here": str(series["NDVI"].loc[series["NDVI"]["median"].idxmax(), "date"])[:10],
        "ndvi_peak_eurocropsml": round(float(ml["NDVI"].max()), 3),
        "ndvi_peak_date_eurocropsml": str(ml.loc[ml["NDVI"].idxmax(), "date"])[:10],
        "gif": str(gif.relative_to(REPO)) if gif else None,
        "comparison": stats,
    }
    ours = series["NDVI"].assign(crop=crop, source="s2_l2a_here")
    theirs = ml.assign(crop=crop, source="eurocropsml_l1c").rename(columns={"NDVI": "median"})
    return record, ours, theirs[["date", "median", "crop", "source"]], series, ml


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--crops", nargs="+", default=list(CROPS), choices=list(CROPS))
    parser.add_argument("--no-gif", action="store_true")
    parser.add_argument("--no-s1", action="store_true")
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    parcels = select_parcels(args.crops)
    print(parcels.to_string(index=False), flush=True)

    findings = {
        "source_imagery": "Sentinel-2 L2A, Microsoft Planetary Computer, pulled with "
                          "space_time_deepsearch.io.get_sentinel2_imagery",
        "reference_series": "EuroCropsML .npz, Sentinel-2 L1C top of atmosphere, "
                            "spatial median over the same parcel",
        "country": COUNTRY, "year": YEAR, "season": list(SEASON), "bands": BANDS,
        "cloud_cover_max": CLOUD_MAX, "min_coverage": MIN_COVERAGE,
        "frame_min_clear": FRAME_MIN_CLEAR, "buffer_m": BUFFER_M, "random_seed": SEED,
        "parcels": [],
    }
    frames, all_series, all_ml = [], {}, {}
    for _, row in parcels.iterrows():
        try:
            rec, ours, theirs, series, ml = run_crop(row, make_gif=not args.no_gif)
        except Exception as exc:  # one crop failing must not lose the others
            print(f"  FAILED {row['crop']}: {type(exc).__name__}: {exc}", flush=True)
            findings["parcels"].append({"crop": row["crop"], "error": f"{type(exc).__name__}: {exc}"})
            continue
        findings["parcels"].append(rec)
        frames += [ours[["date", "median", "crop", "source"]], theirs]
        all_series[row["crop"]], all_ml[row["crop"]] = series, ml

    if all_series:
        overview_figure(all_series, all_ml)
        pd.concat(frames, ignore_index=True).to_parquet(
            CACHE_DIR / "gallery_ndvi_series.parquet", index=False)

    if not args.no_s1 and SHOWCASE in all_series:
        print("\n=== Sentinel-1 panel ===", flush=True)
        try:
            hit = parcel_geometry(parcels.loc[parcels["crop"] == SHOWCASE, "parcel_id"].iloc[0])
            _, feat = sentinel1_panel(hit.geometry.iloc[0], None,
                                      all_series[SHOWCASE]["NDVI"], SHOWCASE, CROPS[SHOWCASE])
            feat.to_parquet(CACHE_DIR / "gallery_sentinel1.parquet", index=False)
            findings["sentinel1"] = {
                "crop": SHOWCASE, "acquisitions": int(len(feat)),
                "orbits": sorted(set(int(o) for o in feat["relative_orbit"])) if "relative_orbit" in feat else [],
                "orbit_states": sorted(set(feat["orbit_state"].astype(str))) if "orbit_state" in feat else [],
                "vv_db_range": [round(float(feat["vv_median_db"].min()), 2),
                                round(float(feat["vv_median_db"].max()), 2)],
                "vh_db_range": [round(float(feat["vh_median_db"].min()), 2),
                                round(float(feat["vh_median_db"].max()), 2)],
            }
        except Exception as exc:
            print(f"  Sentinel-1 failed: {type(exc).__name__}: {exc}", flush=True)
            findings["sentinel1"] = {"error": f"{type(exc).__name__}: {exc}"}

    ok = [p for p in findings["parcels"] if "comparison" in p]
    if ok:
        findings["comparison_summary"] = {
            "median_pearson_r": round(float(np.median([p["comparison"]["pearson_r"] for p in ok])), 4),
            "median_mean_difference": round(
                float(np.median([p["comparison"]["mean_difference"] for p in ok])), 4),
            "median_rmse": round(float(np.median([p["comparison"]["rmse"] for p in ok])), 4),
            "dates_here_vs_eurocropsml": {
                p["crop"]: [p["comparison"]["dates_l2a_here"], p["comparison"]["dates_l1c_eurocropsml"]]
                for p in ok},
        }
    OUT.write_text(json.dumps(findings, indent=2))
    print(f"\nwrote {OUT.relative_to(REPO)} and {len(list(FIG_DIR.glob('*.png')))} figures", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
