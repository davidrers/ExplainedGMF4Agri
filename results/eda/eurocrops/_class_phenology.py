"""Class-level Sentinel-2 and Sentinel-1 phenology for the main Estonian crops, three seasons.

Notebooks 04 and 05 follow single parcels. This follows **classes**: for each of the main
Estonian crops it reduces many parcels to one curve per season, so what is plotted is the
behaviour of the crop rather than the behaviour of one field.

Three seasons are built, 2020, 2021 and 2022, and only 2021 carries labels. **That is the
point of the figure rather than a flaw in it.** EuroCrops declares one year, so a parcel
declared winter wheat in 2021 was something else in 2020 and will be something else again in
2022 wherever rotation applies. The off-year curves therefore measure how much of a 2021
label survives into its neighbouring seasons, which is exactly what a reader needs before
imagery from another year, or an annual embedding, is used against these labels.

Method, which follows ``gfm4agri.data.parcel_explorer`` rather than the chip export:

* parcels are drawn from windows of 2 x 2 chips, so one STAC read serves many parcels of
  many classes, instead of one read per parcel;
* every acquisition is reduced over each parcel polygon, Sentinel-2 after the processing
  baseline offset and the scene-classification screening of :mod:`gfm4agri.data.sentinel`,
  Sentinel-1 in linear power and with the per-track offset removed per parcel;
* per-parcel series are placed on a ten-day grid, then each class takes the median across its
  parcels, with the interquartile range across parcels as the band.

Outputs
-------
``results/eda/eurocrops/phenology.json``              every number the notebook quotes
``results/eda/eurocrops/cache/phenology_*.parquet``   the class curves and the parcel sample
``results/eda/eurocrops/figures/phenology/*.png``     the figures
``data/eurocrops/phenology/<sensor>_<window>_<year>.parquet``  per-parcel reductions, ignored

Usage
-----
    python results/eda/eurocrops/_class_phenology.py                  # everything
    python results/eda/eurocrops/_class_phenology.py --stages fetch   # only the pulls
    python results/eda/eurocrops/_class_phenology.py --windows 4 --years 2021
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
import yaml  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

from gfm4agri.data.chip_grid import CHIP_M, apply_vector_aliases, chip_of, vector_layer_aliases  # noqa: E402
from gfm4agri.data.sentinel import (  # noqa: E402
    SCL_DROP, baseline_lookup, ensure_vendored, harmonise_boa, parcel_mask,
)

ensure_vendored()

FIG_DIR = HERE / "figures" / "phenology"
CACHE_DIR = HERE / "cache"
OUT = HERE / "phenology.json"
PARQUET_DIR = REPO / "data" / "eurocrops" / "parquet"
PULL_DIR = REPO / "data" / "eurocrops" / "phenology"
SCHEME = REPO / "configs" / "class_scheme_eurocropsml.yaml"

SEED = 42
COUNTRY, ID_COL, LAYER_YEAR = "EE", "pollu_id", 2021
#: One season before the declaration, the declaration itself, one after.
YEARS = (2020, 2021, 2022)
LABEL_YEAR = 2021
#: March to mid-November. Outside it a Baltic parcel is snow or darkness, and Sentinel-2
#: returns almost nothing usable.
SEASON = ("{year}-03-01", "{year}-11-15")
#: How many of the scheme's Estonian classes to follow, in frequency order.
N_CLASSES = 10
#: A window is a square block of chips: one STAC read then serves every parcel inside it.
WINDOW_CHIPS = 2            # 2 x 2 chips, 4.48 km on a side
N_WINDOWS = 16
MIN_SEP_KM = 25.0
#: 20 m rather than 10 m. A class median over many parcels does not need the finer grid, the
#: read is four times smaller, and for Sentinel-1 the coarser pixel also halves the speckle.
RESOLUTION = 20
MIN_AREA_HA = 1.0
MIN_COMPACTNESS = 0.45
MAX_PER_CLASS_PER_WINDOW = 8
#: A parcel's date is kept only when this share of its pixels survives the screening.
MIN_VALID_FRACTION = 0.6
#: Ten-day compositing grid, so the three seasons are comparable date by date.
DEKAD_DAYS = 10
CLOUD_COVER_MAX = 80

YEAR_COLOR = {2020: "#9fb8d4", 2021: "#1b4f8a", 2022: "#d4845f"}
YEAR_STYLE = {2020: (0, (4, 2)), 2021: "solid", 2022: (0, (1, 1.4))}

sns.set_theme(context="notebook", style="whitegrid")


# ------------------------------------------------------------------ parcels and windows


def scheme_classes() -> pd.DataFrame:
    """The leading Estonian classes of the thesis class scheme, with their vector aliases."""
    scheme = yaml.safe_load(SCHEME.read_text())
    order = scheme["in_country_class_lists"]["Estonia"][:N_CLASSES]
    names = {c["hcat"]: c["crop_name"] for c in scheme["classes"]}
    return pd.DataFrame({"hcat": order, "crop": [names[h] for h in order],
                         "rank": range(1, len(order) + 1)}), vector_layer_aliases(scheme)


def load_parcels():
    """Estonian parcels of the scheme's leading classes, with geometry metrics attached."""
    import geopandas as gpd

    classes, aliases = scheme_classes()
    gdf = gpd.read_parquet(PARQUET_DIR / f"{COUNTRY}_{LAYER_YEAR}.parquet",
                           columns=[ID_COL, "EC_hcat_c", "EC_hcat_n", "geometry"]).to_crs(3035)
    gdf["hcat"] = apply_vector_aliases(gdf["EC_hcat_c"].astype(str), aliases)
    gdf = gdf[gdf["hcat"].isin(set(classes["hcat"]))].copy()
    gdf["parcel_id"] = gdf[ID_COL].astype("int64").astype(str)
    gdf["area_ha"] = gdf.geometry.area / 1e4
    gdf["compactness"] = 4 * np.pi * gdf.geometry.area / gdf.geometry.length ** 2
    gdf = gdf[(gdf["area_ha"] >= MIN_AREA_HA) & (gdf["compactness"] >= MIN_COMPACTNESS)]
    pts = gdf.geometry.representative_point()
    col, row = chip_of(pts.x.to_numpy(), pts.y.to_numpy())
    gdf["wcol"], gdf["wrow"] = col // WINDOW_CHIPS, row // WINDOW_CHIPS
    return gdf.merge(classes, on="hcat", how="left")


def select_windows(parcels, n_windows: int) -> pd.DataFrame:
    """Windows that carry as many of the classes as possible, kept far apart.

    Ranked by the number of distinct classes present and then by the count of parcels outside
    the dominant class, because a window that is all grassland costs a full read and returns
    one curve. The separation keeps the sample from describing one landscape.
    """
    g = parcels.groupby(["wcol", "wrow"])
    cells = pd.DataFrame({
        "n_classes": g["hcat"].nunique(),
        "n_parcels": g.size(),
        "n_non_dominant": g["hcat"].apply(lambda s: len(s) - s.value_counts().iloc[0]),
    }).reset_index()
    rng = np.random.default_rng(SEED)
    cells["rank"] = (cells["n_classes"] + 0.01 * cells["n_non_dominant"]
                     + rng.uniform(0, 0.4, len(cells)))
    cells = cells.sort_values("rank", ascending=False)
    win_m = CHIP_M * WINDOW_CHIPS
    min_sep = MIN_SEP_KM * 1000 / win_m
    taken: list[tuple[int, int]] = []
    rows = []
    for r in cells.itertuples():
        if any(np.hypot(r.wcol - a, r.wrow - b) < min_sep for a, b in taken):
            continue
        taken.append((r.wcol, r.wrow))
        rows.append({"window": f"{COUNTRY}_{r.wcol:05d}_{r.wrow:05d}", "wcol": int(r.wcol),
                     "wrow": int(r.wrow), "n_classes": int(r.n_classes),
                     "n_parcels": int(r.n_parcels),
                     "xmin": r.wcol * win_m, "ymin": r.wrow * win_m,
                     "xmax": (r.wcol + 1) * win_m, "ymax": (r.wrow + 1) * win_m})
        if len(rows) == n_windows:
            break
    return pd.DataFrame(rows)


def sample_parcels(parcels, windows: pd.DataFrame):
    """Up to ``MAX_PER_CLASS_PER_WINDOW`` parcels of each class in each window, largest first."""
    keep = parcels.merge(windows[["window", "wcol", "wrow"]], on=["wcol", "wrow"], how="inner")
    keep = (keep.sort_values("area_ha", ascending=False)
            .groupby(["window", "hcat"], observed=True).head(MAX_PER_CLASS_PER_WINDOW))
    return keep.reset_index(drop=True)


# -------------------------------------------------------------------------- the reductions


def _labels(window_geom_crs, sample, cube):
    """Per-pixel parcel index for one window, and the parcel order it indexes."""
    from affine import Affine
    from rasterio.features import rasterize

    x, y = cube.x.values, cube.y.values
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    transform = Affine.translation(x[0] - dx / 2, y[0] - dy / 2) * Affine.scale(dx, dy)
    geoms = [(geom, i + 1) for i, geom in enumerate(window_geom_crs)]
    arr = rasterize(geoms, out_shape=(len(y), len(x)), transform=transform, fill=0,
                    dtype="int32")
    return arr, sample["parcel_id"].to_numpy()


def _reduce(values: np.ndarray, labels_flat: np.ndarray, n_parcels: int):
    """Median and valid-pixel count of one date's band, per parcel."""
    v = values.ravel()
    ok = np.isfinite(v)
    df = pd.DataFrame({"label": labels_flat[ok], "v": v[ok]})
    g = df.groupby("label")["v"]
    return g.median(), g.size()


def pull_s2(window: pd.Series, year: int, sample, force: bool = False) -> pd.DataFrame:
    """Per-parcel, per-acquisition NDVI and NDMI for one window and season."""
    import geopandas as gpd
    from shapely.geometry import box

    out = PULL_DIR / f"s2_{window['window']}_{year}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    from gfm4agri.data.sentinel import fetch_s2

    aoi = gpd.GeoSeries([box(window["xmin"], window["ymin"], window["xmax"], window["ymax"])],
                        crs=3035).to_crs(4326).iloc[0]
    start, end = SEASON[0].format(year=year), SEASON[1].format(year=year)
    cube, baseline = fetch_s2(aoi, start, end, bands=["B04", "B08", "B11", "SCL"],
                              resolution=RESOLUTION, cloud_cover_max=CLOUD_COVER_MAX,
                              min_coverage=60, verbose=False)
    geoms = sample.to_crs(int(cube.epsg)).geometry.to_list()
    lab, parcel_ids = _labels(geoms, sample, cube)
    inside = lab > 0
    labels_flat = lab[inside]
    counts = np.bincount(labels_flat, minlength=len(parcel_ids) + 1)[1:]

    scl = cube.sel(band="SCL").values
    red = cube.sel(band="B04").values.astype("float32")
    nir = cube.sel(band="B08").values.astype("float32")
    swir = cube.sel(band="B11").values.astype("float32")
    bad = np.isin(scl, SCL_DROP) | ~np.isfinite(red) | ~np.isfinite(nir)
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = np.where(bad, np.nan, (nir - red) / (nir + red))
        ndmi = np.where(bad, np.nan, (nir - swir) / (nir + swir))

    dates = pd.DatetimeIndex(cube.time.values).tz_localize(None).as_unit("ns")
    rows = []
    for t, date in enumerate(dates):
        med_v, n_v = _reduce(ndvi[t][inside], labels_flat, len(parcel_ids))
        med_m, _ = _reduce(ndmi[t][inside], labels_flat, len(parcel_ids))
        for label, value in med_v.items():
            n = int(n_v.get(label, 0))
            if n < MIN_VALID_FRACTION * counts[label - 1]:
                continue
            rows.append({"parcel_id": parcel_ids[label - 1], "date": date, "NDVI": float(value),
                         "NDMI": float(med_m.get(label, np.nan)), "n_pixels": n})
    df = pd.DataFrame(rows)
    df["window"], df["year"], df["sensor"] = window["window"], year, "s2"
    PULL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"    s2 {window['window']} {year}: {len(dates)} acquisitions, "
          f"{len(df):,} parcel-dates, baselines {baseline.get('baselines')}", flush=True)
    return df


def pull_s1(window: pd.Series, year: int, sample, force: bool = False) -> pd.DataFrame:
    """Per-parcel, per-acquisition VV and VH for one window and season, in decibels."""
    import geopandas as gpd
    from shapely.geometry import box

    out = PULL_DIR / f"s1_{window['window']}_{year}.parquet"
    if out.exists() and not force:
        return pd.read_parquet(out)
    from space_time_deepsearch.io import get_sentinel1_rtc_imagery, normalize_orbit

    aoi = gpd.GeoSeries([box(window["xmin"], window["ymin"], window["xmax"], window["ymax"])],
                        crs=3035).to_crs(4326).iloc[0]
    start, end = SEASON[0].format(year=year), SEASON[1].format(year=year)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cube = get_sentinel1_rtc_imagery(custom_geometry=aoi, start_date=start, end_date=end,
                                         resolution=RESOLUTION, min_coverage=0.0,
                                         add_indices=False, verbose=False)
    geoms = sample.to_crs(int(cube.epsg)).geometry.to_list()
    lab, parcel_ids = _labels(geoms, sample, cube)
    inside = lab > 0
    labels_flat = lab[inside]
    counts = np.bincount(labels_flat, minlength=len(parcel_ids) + 1)[1:]

    vv = cube.sel(band="vv").values.astype("float64")
    vh = cube.sel(band="vh").values.astype("float64")
    bad = ~np.isfinite(vv) | ~np.isfinite(vh) | (vv <= 0) | (vh <= 0)
    vv, vh = np.where(bad, np.nan, vv), np.where(bad, np.nan, vh)
    dates = pd.DatetimeIndex(cube.time.values).tz_localize(None).as_unit("ns")
    # The loader renames the STAC properties, so these are the coordinate names the cube
    # actually carries; reading sat:relative_orbit instead silently collapses every track into
    # one group and the offset correction then removes nothing.
    def _coord(name, fill):
        if name not in cube.coords:
            return np.full(len(dates), fill)
        v = np.atleast_1d(cube[name].values)
        # A season flown by one track leaves the coordinate scalar, as above.
        return np.repeat(v, len(dates)) if v.size == 1 and len(dates) > 1 else v

    orbit = _coord("relative_orbit", -1)
    state = _coord("orbit_state", "")
    platform = _coord("platform", "")

    rows = []
    for t, date in enumerate(dates):
        med_vv, n_vv = _reduce(vv[t][inside], labels_flat, len(parcel_ids))
        med_vh, _ = _reduce(vh[t][inside], labels_flat, len(parcel_ids))
        for label, value in med_vv.items():
            n = int(n_vv.get(label, 0))
            if n < MIN_VALID_FRACTION * counts[label - 1]:
                continue
            rows.append({"parcel_id": parcel_ids[label - 1], "date": date,
                         "vv_median": float(value), "vh_median": float(med_vh.get(label, np.nan)),
                         "relative_orbit": int(orbit[t]), "orbit_state": str(state[t]),
                         "platform": str(platform[t]), "n_pixels": n})
    df = pd.DataFrame(rows)
    if df.empty:
        print(f"    s1 {window['window']} {year}: nothing reduced", flush=True)
        return df
    # Reductions are in linear power; decibels come after, and the per-track offset is removed
    # per parcel, because a track's offset is a property of the parcel's viewing geometry.
    df["vv_median_db"] = 10 * np.log10(df["vv_median"])
    df["vh_median_db"] = 10 * np.log10(df["vh_median"])
    out_frames = []
    for pid, part in df.groupby("parcel_id"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                part = normalize_orbit(part.copy(), cols=("vv_median_db", "vh_median_db"),
                                       verbose=False)
            except Exception:
                part = part.assign(vv_median_db_adj=part["vv_median_db"],
                                   vh_median_db_adj=part["vh_median_db"])
        out_frames.append(part)
    df = pd.concat(out_frames, ignore_index=True)
    df["window"], df["year"], df["sensor"] = window["window"], year, "s1"
    PULL_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"    s1 {window['window']} {year}: {len(dates)} acquisitions, "
          f"{len(df):,} parcel-dates", flush=True)
    return df


# ------------------------------------------------------------------------- aggregation


def to_dekads(df: pd.DataFrame, value_cols: list[str]) -> pd.DataFrame:
    """One row per parcel, year and ten-day period, so the seasons line up date by date."""
    out = df.copy()
    doy = pd.to_datetime(out["date"]).dt.dayofyear
    out["dekad"] = ((doy - 1) // DEKAD_DAYS).astype(int)
    out["dekad_doy"] = out["dekad"] * DEKAD_DAYS + DEKAD_DAYS / 2
    keys = ["parcel_id", "year", "dekad", "dekad_doy"]
    return out.groupby(keys, observed=True)[value_cols].median().reset_index()


def class_curves(parcel_dekads: pd.DataFrame, sample: pd.DataFrame,
                 value_cols: list[str]) -> pd.DataFrame:
    """Median across the parcels of a class, per ten-day period, with the interquartile band."""
    merged = parcel_dekads.merge(sample[["parcel_id", "hcat", "crop", "rank"]],
                                 on="parcel_id", how="inner")
    keys = ["crop", "hcat", "rank", "year", "dekad", "dekad_doy"]
    rows = merged.groupby(keys, observed=True)[value_cols].median()
    rows.columns = [f"{c}_median" for c in value_cols]
    for c in value_cols:
        q = merged.groupby(keys, observed=True)[c].quantile([0.25, 0.75]).unstack()
        rows[f"{c}_q25"], rows[f"{c}_q75"] = q[0.25], q[0.75]
    rows["n_parcels"] = merged.groupby(keys, observed=True)["parcel_id"].nunique()
    return rows.reset_index()


# ---------------------------------------------------------------------------- figures


def windows_figure(parcels, windows: pd.DataFrame, sample: pd.DataFrame, dest: Path) -> Path:
    """Where the sample sits: the declared parcels of Estonia and the windows drawn from them.

    A class curve is only as general as the landscape it was measured in, so the geography of
    the sample belongs beside the curves rather than in a caption.
    """
    import matplotlib.patches as mpatches

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4), width_ratios=[1.35, 1])
    pts = parcels.geometry.representative_point()
    axes[0].hexbin(pts.x / 1000, pts.y / 1000, gridsize=120, bins="log", cmap="Greys", mincnt=1)
    for w in windows.itertuples():
        axes[0].add_patch(mpatches.Rectangle((w.xmin / 1000, w.ymin / 1000),
                                             (w.xmax - w.xmin) / 1000, (w.ymax - w.ymin) / 1000,
                                             fill=False, ec="#d4845f", lw=1.6))
        axes[0].plot(w.xmin / 1000, w.ymin / 1000, marker="", ls="")
    axes[0].set_aspect("equal")
    axes[0].set_xlabel("ETRS89-LAEA easting (km)")
    axes[0].set_ylabel("northing (km)")
    axes[0].set_title(f"{len(windows)} windows of "
                      f"{CHIP_M * WINDOW_CHIPS / 1000:.2f} km over the declared parcels",
                      fontsize=10)
    axes[0].grid(False)

    counts = sample.groupby("crop").size().sort_values()
    axes[1].barh([c.replace("_", " ") for c in counts.index], counts.values, color="#1b4f8a")
    axes[1].set_xlabel("parcels in the sample")
    axes[1].set_title("parcels per class", fontsize=10)
    for i, v in enumerate(counts.values):
        axes[1].text(v, i, f" {v}", va="center", fontsize=8)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return dest


def panel_figure(curves: pd.DataFrame, value: str, label: str, title: str, dest: Path,
                 ylim: tuple[float, float] | None = None) -> Path:
    crops = (curves.sort_values("rank")["crop"].drop_duplicates().to_list())
    n = len(crops)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 2.9 * nrow), sharex=True,
                             sharey=True, layout="constrained")
    for ax, crop in zip(np.ravel(axes), crops):
        g = curves[curves["crop"] == crop]
        for year, gy in g.groupby("year"):
            gy = gy.sort_values("dekad_doy")
            ax.fill_between(gy["dekad_doy"], gy[f"{value}_q25"], gy[f"{value}_q75"],
                            color=YEAR_COLOR[year], alpha=0.13, lw=0)
            ax.plot(gy["dekad_doy"], gy[f"{value}_median"], color=YEAR_COLOR[year],
                    ls=YEAR_STYLE[year], lw=1.9, label=f"{year}  (n={int(gy['n_parcels'].max())})")
        ax.set_title(crop.replace("_", " "), fontsize=10)
        ax.legend(fontsize=7, loc="best")
        if ylim:
            ax.set_ylim(*ylim)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    # A shared x axis hides the tick labels of every panel that has one below it. The last
    # row is rarely full, so the panels sitting above an empty slot would lose their labels
    # although nothing covers them.
    for i, ax in enumerate(np.ravel(axes)[:n]):
        if i + ncol >= n:
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("day of year")
    for r in range(nrow):
        np.ravel(axes)[r * ncol].set_ylabel(label)
    fig.suptitle(title, fontsize=13)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return dest


def overview_figure(curves: pd.DataFrame, value: str, label: str, year: int,
                    dest: Path) -> Path | None:
    if not len(curves[curves["year"] == year]):
        return None
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    palette = sns.color_palette("tab10", curves["crop"].nunique())
    for colour, (crop, g) in zip(palette, curves[curves["year"] == year].groupby("crop")):
        g = g.sort_values("dekad_doy")
        ax.plot(g["dekad_doy"], g[f"{value}_median"], lw=1.9, color=colour,
                label=f"{crop.replace('_', ' ')} (n={int(g['n_parcels'].max())})")
    ax.set_xlabel("day of year")
    ax.set_ylabel(label)
    ax.set_title(f"{label}, class medians, Estonia {year}")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return dest


def stability_figure(curves: pd.DataFrame, value: str, dest: Path) -> tuple[Path, dict]:
    """How much of a 2021 class curve survives into the seasons on either side."""
    out = {}
    for crop, g in curves.groupby("crop"):
        ref = g[g["year"] == LABEL_YEAR].set_index("dekad")[f"{value}_median"]
        for year in YEARS:
            if year == LABEL_YEAR:
                continue
            other = g[g["year"] == year].set_index("dekad")[f"{value}_median"]
            common = ref.index.intersection(other.index)
            if len(common) < 8:
                continue
            r = float(np.corrcoef(ref.loc[common], other.loc[common])[0, 1])
            out.setdefault(crop, {})[str(year)] = round(r, 3)
    tbl = pd.DataFrame(out).T.reindex(curves.sort_values("rank")["crop"].drop_duplicates())
    fig, ax = plt.subplots(figsize=(7.6, 0.45 * len(tbl) + 1.8))
    sns.heatmap(tbl.astype(float), annot=True, fmt=".2f", cmap="RdYlGn", vmin=-1, vmax=1,
                cbar_kws={"label": f"correlation with the {LABEL_YEAR} curve"}, ax=ax)
    ax.set_title(f"How far a {LABEL_YEAR} label describes the neighbouring seasons ({value})",
                 fontsize=11)
    ax.set_xlabel("season")
    fig.tight_layout()
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return dest, out


# -------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--windows", type=int, default=N_WINDOWS)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS))
    parser.add_argument("--stages", nargs="+", default=["fetch", "aggregate", "figures"],
                        choices=["fetch", "aggregate", "figures"])
    parser.add_argument("--sensors", nargs="+", default=["s2", "s1"], choices=["s2", "s1"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    parcels = load_parcels()
    windows = select_windows(parcels, args.windows)
    sample = sample_parcels(parcels, windows)
    sample_out = sample[["parcel_id", "hcat", "crop", "rank", "window", "area_ha",
                         "compactness"]].copy()
    sample_out.to_parquet(CACHE_DIR / "phenology_sample.parquet", index=False)
    print(f"{len(windows)} windows, {len(sample):,} parcels, "
          f"{sample['crop'].nunique()} classes", flush=True)
    print(sample.groupby("crop").size().sort_values(ascending=False).to_string(), flush=True)

    if "fetch" in args.stages:
        t0 = time.time()
        for _, w in windows.iterrows():
            inside = sample[sample["window"] == w["window"]]
            for year in args.years:
                for sensor in args.sensors:
                    try:
                        (pull_s2 if sensor == "s2" else pull_s1)(w, year, inside, args.force)
                    except Exception as exc:
                        print(f"    {sensor} {w['window']} {year} FAILED: "
                              f"{type(exc).__name__}: {exc}", flush=True)
        print(f"  fetch stage took {(time.time() - t0) / 60:.1f} min", flush=True)

    findings: dict = {
        "country": "Estonia", "label_year": LABEL_YEAR, "years": list(args.years),
        "season": list(SEASON), "resolution_m": RESOLUTION, "dekad_days": DEKAD_DAYS,
        "n_windows": int(len(windows)), "window_km": CHIP_M * WINDOW_CHIPS / 1000,
        "n_parcels_sampled": int(len(sample)), "min_area_ha": MIN_AREA_HA,
        "min_valid_fraction": MIN_VALID_FRACTION, "scl_drop": list(SCL_DROP),
        "random_seed": SEED,
        "parcels_per_class": {k: int(v) for k, v in sample.groupby("crop").size().items()},
    }

    if "aggregate" in args.stages:
        s2 = [pd.read_parquet(p) for p in sorted(PULL_DIR.glob("s2_*.parquet"))]
        s1 = [pd.read_parquet(p) for p in sorted(PULL_DIR.glob("s1_*.parquet"))]
        if s2:
            d = to_dekads(pd.concat(s2, ignore_index=True), ["NDVI", "NDMI"])
            c = class_curves(d, sample_out, ["NDVI", "NDMI"])
            c.to_parquet(CACHE_DIR / "phenology_s2_curves.parquet", index=False)
            findings["s2_parcel_dates"] = int(sum(len(x) for x in s2))
        if s1:
            cols = ["vv_median_db_adj", "vh_median_db_adj"]
            frame = pd.concat(s1, ignore_index=True)
            for c_ in cols:
                if c_ not in frame:
                    frame[c_] = frame[c_.replace("_adj", "")]
            d = to_dekads(frame, cols)
            c = class_curves(d, sample_out, cols)
            c.to_parquet(CACHE_DIR / "phenology_s1_curves.parquet", index=False)
            findings["s1_parcel_dates"] = int(len(frame))

    if "figures" in args.stages:
        windows_figure(parcels, windows, sample, FIG_DIR / "00_sample_windows.png")
        s2p = CACHE_DIR / "phenology_s2_curves.parquet"
        s1p = CACHE_DIR / "phenology_s1_curves.parquet"
        if s2p.exists():
            c = pd.read_parquet(s2p)
            panel_figure(c, "NDVI", "NDVI", "Sentinel-2 NDVI by crop, class medians over "
                         f"{findings['n_parcels_sampled']} Estonian parcels, three seasons",
                         FIG_DIR / "01_s2_ndvi_by_crop.png", ylim=(0, 1))
            panel_figure(c, "NDMI", "NDMI", "Sentinel-2 NDMI by crop, class medians, "
                         "three seasons", FIG_DIR / "02_s2_ndmi_by_crop.png")
            overview_figure(c, "NDVI", "NDVI", LABEL_YEAR, FIG_DIR / "03_s2_overview_2021.png")
            _, stab = stability_figure(c, "NDVI", FIG_DIR / "05_label_stability.png")
            findings["label_stability_ndvi"] = stab
        if s1p.exists():
            c = pd.read_parquet(s1p)
            panel_figure(c, "vh_median_db_adj", "VH gamma0 (dB)",
                         "Sentinel-1 VH by crop, class medians, three seasons",
                         FIG_DIR / "04_s1_vh_by_crop.png")
            panel_figure(c, "vv_median_db_adj", "VV gamma0 (dB)",
                         "Sentinel-1 VV by crop, class medians, three seasons",
                         FIG_DIR / "06_s1_vv_by_crop.png")
            overview_figure(c, "vh_median_db_adj", "VH gamma0 (dB)", LABEL_YEAR,
                            FIG_DIR / "07_s1_overview_2021.png")

    OUT.write_text(json.dumps(findings, indent=2))
    print(f"wrote {OUT.relative_to(REPO)} and {len(list(FIG_DIR.glob('*.png')))} figures",
          flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
