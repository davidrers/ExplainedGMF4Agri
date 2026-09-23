"""Explore one EuroCrops parcel: charts and animations, Sentinel-2 and Sentinel-1.

The vendored package ships ``space_time_deepsearch.vis.parcel_timeseries.explore``, which
does exactly what is wanted here, but it is wired to the Austrian GSA and ISV GeoPackages
and addresses a parcel by its ``KENNUNG``. There is no Sentinel-1 counterpart; the radar
equivalent of the workflow is ``space_time_deepsearch.io.sentinel1.parcel_series``.

This module supplies both for the data of this thesis, keyed by the **national parcel
identifier** of the EuroCrops vector release and a **year**:

    from gfm4agri.data.parcel_explorer import explore, explore_s1, explore_both

    s2 = explore(21121746)                 # Estonia, 2021, inferred from the identifier
    s1 = explore_s1(21121746)
    s2, s1 = explore_both(21121746, year=2021)

    s2.series["NDVI"]     # date, median, n_pixels, moran, moran_z
    s2.png, s2.gif        # the five-index chart and the NDVI animation
    s1.features           # one row per acquisition, with the orbit-corrected columns

The charts and the Sentinel-2 animation are the vendored ones, reached through a small
adapter: ``plot`` and ``animate`` need five attributes from the parcel row and an
inspection frame that is allowed to be empty, so a EuroCrops parcel is presented to them
under the names they expect. The Sentinel-1 animation has no upstream counterpart and is
written here in the same visual language.

Three corrections are applied that the vendored calls do not make on their own, all of
them in :mod:`gfm4agri.data.sentinel`: the processing-baseline reflectance offset, cloud
screening at reduction time, and the parcel mask.
"""

from __future__ import annotations

import contextlib
import io
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from gfm4agri.data.sentinel import (
    MIN_VALID_FRACTION, PIXEL_M, SCL_DROP, S2_BANDS, REPO, clear_fraction, ensure_vendored,
    eurocropsml_series, fetch_s2, parcel_mask, quiet_lines, screen,
)

PARQUET_DIR = Path(os.environ.get("EUROCROPS_DATA", REPO / "data" / "eurocrops")) / "parquet"
OUT_ROOT = Path(os.environ.get("GFM4AGRI_EXPLORE_OUT", REPO / "results" / "explore"))

#: The national identifier column per country. Lithuania publishes no parcel key at all,
#: only a field-block number, so an identifier there addresses a block and not a parcel.
# The vendored charts ask for font weight 600, which DejaVu does not ship, and matplotlib
# warns once per figure. The substitution it makes is correct, so the warning is noise.
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

COUNTRY_ID = {"EE": "pollu_id", "LV": "PARCEL_ID", "LT": "KZS_NR", "PT": "OSA_ID"}
COUNTRY_NAME = {"EE": "Estonia", "LV": "Latvia", "LT": "Lithuania", "PT": "Portugal"}
#: EuroCrops covers one declaration year per country, and these four are all 2021.
COUNTRY_YEAR = {"EE": 2021, "LV": 2021, "LT": 2021, "PT": 2021}
#: Default season. Wide enough to carry autumn sowing and spring green-up for a winter crop.
SEASON = ("{year}-03-01", "{year}-11-15")
#: HCAT's first four digits carry the broad group, which is what the chart header shows.
HCAT_GROUP = {"3301": "arable", "3302": "grassland", "3303": "permanent", "3304": "fallow",
              "3305": "greenhouse", "3306": "wood", "3399": "other"}


@dataclass
class S2Exploration:
    """Everything the Sentinel-2 pass produced for one parcel."""

    parcel_id: str
    country: str
    year: int
    crop: str
    hcat: str
    area_ha: float
    parcel: object  # GeoDataFrame, one row, in the layer's native CRS
    cube: xr.DataArray
    series: dict[str, pd.DataFrame]
    table: pd.DataFrame
    clear: np.ndarray
    baseline: dict
    dates_dropped: int
    png: Path | None = None
    gif: Path | None = None
    eurocropsml: pd.DataFrame | None = None
    comparison: dict = field(default_factory=dict)
    comparison_png: Path | None = None


@dataclass
class S1Exploration:
    """Everything the Sentinel-1 pass produced for one parcel."""

    parcel_id: str
    country: str
    year: int
    crop: str
    parcel: object
    cube: xr.DataArray
    features: pd.DataFrame
    png: Path | None = None
    gif: Path | None = None


# --------------------------------------------------------------------------- parcel lookup


def _read_by_id(cc: str, parcel_id: str | int):
    """One parcel from a country layer, tolerating the identifier's stored dtype.

    Portugal stores ``OSA_ID`` as a float and Lithuania stores ``KZS_NR`` as text, so a
    single comparison cannot serve all four countries; the candidates are tried in turn and
    a full scan is the last resort.
    """
    import geopandas as gpd

    path = PARQUET_DIR / f"{cc}_{COUNTRY_YEAR[cc]}.parquet"
    if not path.exists():
        return None
    col = COUNTRY_ID[cc]
    cols = [col, "EC_hcat_n", "EC_hcat_c", "EC_trans_n", "geometry"]
    candidates: list = [str(parcel_id)]
    try:
        candidates += [int(parcel_id), float(parcel_id)]
    except (TypeError, ValueError):
        pass
    for value in candidates:
        try:
            hit = gpd.read_parquet(path, columns=cols, filters=[(col, "==", value)])
        except Exception:
            continue
        if len(hit):
            return hit.iloc[[0]]
    # Last resort: normalise the whole column and match on that.
    full = gpd.read_parquet(path, columns=cols)
    ids = pd.to_numeric(full[col], errors="coerce")
    if ids.notna().all():
        ids = ids.astype("int64").astype(str)
    else:
        ids = full[col].astype(str).str.strip()
    hit = full[ids == str(parcel_id).strip()]
    return hit.iloc[[0]] if len(hit) else None


def find_parcel(parcel_id: str | int, country: str | None = None):
    """The parcel row and its country, searching every downloaded layer when needed."""
    order = [country.upper()] if country else list(COUNTRY_ID)
    for cc in order:
        hit = _read_by_id(cc, parcel_id)
        if hit is not None:
            return hit, cc
    raise KeyError(f"parcel {parcel_id} not found in {', '.join(order)}. "
                   f"Layers are read from {PARQUET_DIR}")


def _shim(parcel, cc: str, parcel_id: str):
    """Present a EuroCrops parcel under the attribute names the vendored charts expect.

    ``plot`` and ``animate`` read five fields from the row: the declaration name and code,
    a land-use class, the area in square metres and a free-text code list. Supplying them
    is cheaper and far less brittle than forking four hundred lines of plotting.
    """
    row = parcel.iloc[0]
    hcat = str(row["EC_hcat_c"])
    out = parcel.copy()
    out["SNAR_BEZEICHNUNG"] = str(row["EC_hcat_n"]).replace("_", " ")
    out["SNAR_CODE"] = hcat
    out["FS_NART_CODE"] = HCAT_GROUP.get(hcat[:4], "other")
    out["FLAECHE_BRUTTO"] = float(parcel.to_crs(3035).geometry.area.iloc[0])
    out["CODES"] = (f"EuroCrops {COUNTRY_NAME[cc]} {COUNTRY_YEAR[cc]}, "
                    f"{COUNTRY_ID[cc]} {parcel_id}, declared "
                    f"\"{row['EC_trans_n']}\"")
    return out


def _empty_isv() -> pd.DataFrame:
    """The inspection frame the vendored charts take; EuroCrops has no inspections."""
    return pd.DataFrame({"cls": pd.Series(dtype="object"),
                         "inspection_date": pd.Series(dtype="datetime64[ns]"),
                         "event_date": pd.Series(dtype="datetime64[ns]")})


def _season(year: int, season: tuple[str, str] | None) -> tuple[str, str]:
    if season:
        return season
    return SEASON[0].format(year=year), SEASON[1].format(year=year)


def _out_dir(cc: str, parcel_id: str, year: int, out_dir: Path | None) -> Path:
    d = Path(out_dir) if out_dir else OUT_ROOT / f"{cc}_{parcel_id}_{year}"
    d.mkdir(parents=True, exist_ok=True)
    return d


# ------------------------------------------------------------------------------ Sentinel-2


def explore(parcel_id: str | int, year: int | None = None, country: str | None = None, *,
            season: tuple[str, str] | None = None, animation: bool = True,
            compare: bool = True, anim_band: str = "NDVI", fps: int = 2,
            resolution: int = 10, cloud_cover_max: int = 70, min_coverage: int = 80,
            min_pixels: int = 5, min_valid_fraction: float = MIN_VALID_FRACTION,
            out_dir: Path | None = None, verbose: bool = True,
            show: bool = True) -> S2Exploration:
    """Fetch Sentinel-2 for one parcel, chart the indices, animate, and compare.

    The chart is the vendored five-index panel with Moran's I; the animation is the
    vendored NDVI map with a cursor sweeping the series. When ``compare`` is set and the
    parcel is in EuroCropsML, its shipped L1C series is drawn against the series computed
    here, which is the check that matters for Phase 1.
    """
    ensure_vendored()
    from space_time_deepsearch.vis.parcel_timeseries import animate, compute_indices, plot

    parcel, cc = find_parcel(parcel_id, country)
    year = year or COUNTRY_YEAR[cc]
    row = parcel.iloc[0]
    pid = str(parcel_id)
    crop = str(row["EC_hcat_n"]).replace("_", " ")
    shim = _shim(parcel, cc, pid)
    area_ha = float(shim.iloc[0]["FLAECHE_BRUTTO"]) / 1e4
    start, end = _season(year, season)
    dest = _out_dir(cc, pid, year, out_dir)

    if verbose:
        print(f"Parcel {pid} ({COUNTRY_NAME[cc]}, {year})")
        print(f"  crop  : {crop} (HCAT {row['EC_hcat_c']}), declared \"{row['EC_trans_n']}\"")
        print(f"  area  : {area_ha:.2f} ha, {area_ha * 1e4 / PIXEL_M ** 2:,.0f} pixels at 10 m")
        print(f"  S2    : {start} -> {end} @{resolution} m ...")

    geom_4326 = parcel.to_crs(4326).geometry.iloc[0]
    cube, baseline = fetch_s2(geom_4326, start, end, resolution=resolution,
                              cloud_cover_max=cloud_cover_max, min_coverage=min_coverage,
                              verbose=verbose)
    mask = parcel_mask(cube, parcel.to_crs(int(cube.epsg)).geometry.iloc[0])
    clear = clear_fraction(cube, mask)
    masked = cube.where(mask)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = compute_indices(masked, offset=0.0, scl_drop=SCL_DROP)
    if not len(raw.get("NDVI", [])):
        raise SystemExit("no usable observations: raise cloud_cover_max or widen the season")
    series, dropped = screen(raw, int(mask.sum()), min_valid_fraction)
    if verbose:
        print(f"  dates : {len(series['NDVI'])} usable, {dropped} dropped below "
              f"{100 * min_valid_fraction:.0f} % clear, median "
              f"{series['NDVI'].n_pixels.median():.0f} valid px/date")

    stem = f"{cc}_{pid}_{year}"
    png = dest / f"{stem}_s2_indices.png"
    fig = plot(series, shim, _empty_isv(), year, pid, png, min_pixels)
    _relabel(fig, png, cc, year)
    if show and _in_notebook():
        from IPython.display import display as _display
        import matplotlib.pyplot as plt
        _display(fig)
        plt.close(fig)

    gif = None
    if animation:
        gif = animate(masked, series, _empty_isv(), shim, year, pid,
                      dest / f"{stem}_s2_animation.gif", fps=fps, offset=0.0,
                      scl_drop=SCL_DROP, ts_band=anim_band)
        if show:
            _show_gif(gif)

    table = None
    for name, df in series.items():
        col = df[["date", "median", "n_pixels"]].rename(
            columns={"median": name, "n_pixels": f"{name}_n_px"})
        table = col if table is None else table.merge(col, on="date", how="outer")
    table = table.sort_values("date").reset_index(drop=True)

    ml, stats, cmp_png = None, {}, None
    if compare:
        try:
            ml = eurocropsml_series(pid)
        except (KeyError, FileNotFoundError) as exc:
            if verbose:
                print(f"  EuroCropsML: not available ({type(exc).__name__})")
        if ml is not None:
            stats = compare_series(series["NDVI"], ml)
            cmp_png = _comparison_figure(series["NDVI"], ml, crop, pid, stats,
                                         dest / f"{stem}_s2_vs_eurocropsml.png", show=show)
            if verbose:
                print(f"  vs ML : {stats['matched_dates']} matched dates, "
                      f"r={stats['pearson_r']}, mean difference "
                      f"{stats['mean_difference']:+.3f} (L2A minus L1C)")

    (dest / f"{stem}_s2.json").write_text(json.dumps({
        "parcel_id": pid, "country": cc, "year": year, "crop": crop,
        "hcat": str(row["EC_hcat_c"]), "area_ha": round(area_ha, 3),
        "season": [start, end], "scenes_retained": int(len(cube.time)),
        "dates_after_screening": int(len(series["NDVI"])), "dates_dropped": dropped,
        "baseline": baseline, "scl_drop": list(SCL_DROP),
        "min_valid_fraction": min_valid_fraction, "comparison": stats,
    }, indent=2))

    return S2Exploration(pid, cc, year, crop, str(row["EC_hcat_c"]), area_ha, parcel, cube,
                         series, table, clear, baseline, dropped, png, gif, ml, stats, cmp_png)


# ------------------------------------------------------------------------------ Sentinel-1


def explore_s1(parcel_id: str | int, year: int | None = None, country: str | None = None, *,
               season: tuple[str, str] | None = None, animation: bool = True,
               band: str = "vh", fps: int = 2, resolution: int = 10, min_pixels: int = 5,
               ndvi: pd.DataFrame | None = None, out_dir: Path | None = None,
               verbose: bool = True, show: bool = True) -> S1Exploration:
    """Fetch Sentinel-1 RTC for one parcel, chart the backscatter, and animate it.

    ``parcel_series`` does the fetching, the per-date reduction in linear power and the
    removal of the per-track offset; what is added here is the chart and the animation, so
    that radar and optical can be read side by side for the same field.

    ``ndvi`` optionally overlays an optical series, normally ``explore(...).series["NDVI"]``.
    """
    ensure_vendored()
    from space_time_deepsearch.io import parcel_series

    parcel, cc = find_parcel(parcel_id, country)
    year = year or COUNTRY_YEAR[cc]
    row = parcel.iloc[0]
    pid = str(parcel_id)
    crop = str(row["EC_hcat_n"]).replace("_", " ")
    start, end = _season(year, season)
    dest = _out_dir(cc, pid, year, out_dir)

    if verbose:
        print(f"Parcel {pid} ({COUNTRY_NAME[cc]}, {year})")
        print(f"  crop  : {crop} (HCAT {row['EC_hcat_c']})")
        print(f"  S1    : {start} -> {end} @{resolution} m ...")

    geom_4326 = parcel.to_crs(4326).geometry.iloc[0]
    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        # min_coverage stays at the loader default of 0: it is measured over the bounding
        # box of the area of interest, and a parcel polygon fills only part of its own
        # bounding box, so any positive threshold rejects every date.
        feat, cube = parcel_series(geom_4326, start, end, resolution=resolution,
                                   min_pixels=min_pixels, orbit_normalize=True,
                                   return_cube=True, verbose=True)
    if verbose:
        for line in quiet_lines(buf.getvalue()):
            print(f"  {line}")
    if feat.empty:
        raise SystemExit("no Sentinel-1 acquisitions survived the reduction")
    if verbose:
        tracks = ", ".join(str(t) for t in sorted(set(feat["relative_orbit"])))
        print(f"  dates : {len(feat)} acquisitions on tracks {tracks}")

    stem = f"{cc}_{pid}_{year}"
    png = _s1_figure(feat, crop, pid, cc, year, ndvi, dest / f"{stem}_s1_series.png", show=show)
    gif = None
    if animation:
        gif = _animate_s1(cube, feat, crop, pid, cc, year, band,
                          dest / f"{stem}_s1_animation.gif", fps=fps)
        if show:
            _show_gif(gif)
    feat.to_csv(dest / f"{stem}_s1_features.csv", index=False)

    return S1Exploration(pid, cc, year, crop, parcel, cube, feat, png, gif)


def explore_both(parcel_id: str | int, year: int | None = None, country: str | None = None,
                 **kwargs) -> tuple[S2Exploration, S1Exploration]:
    """Both sensors for one parcel, with the optical series overlaid on the radar chart."""
    s2 = explore(parcel_id, year, country, **kwargs)
    s1 = explore_s1(parcel_id, s2.year, s2.country, ndvi=s2.series["NDVI"],
                    **{k: v for k, v in kwargs.items()
                       if k in {"season", "animation", "fps", "resolution", "out_dir",
                                "verbose", "show"}})
    return s2, s1


# ---------------------------------------------------------------------------------- charts


def compare_series(ours: pd.DataFrame, theirs: pd.DataFrame, tol_days: int = 1) -> dict:
    """Match this NDVI series against the EuroCropsML one by date and quantify the gap."""
    a = ours[["date", "median"]].rename(columns={"median": "ndvi_l2a"})
    b = theirs[["date", "NDVI"]].rename(columns={"NDVI": "ndvi_l1c"})
    for f in (a, b):
        f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None).dt.as_unit("ns").dt.normalize()
    m = pd.merge_asof(a.sort_values("date"), b.sort_values("date"), on="date",
                      tolerance=pd.Timedelta(days=tol_days), direction="nearest").dropna()
    if len(m) < 3:
        return {"matched_dates": int(len(m))}
    d = m["ndvi_l2a"] - m["ndvi_l1c"]
    rough = lambda v: float(np.median(np.abs(np.diff(np.asarray(v, dtype=float)))))
    return {
        "dates_l2a_here": int(len(a)), "dates_l1c_eurocropsml": int(len(b)),
        "matched_dates": int(len(m)),
        "pearson_r": round(float(np.corrcoef(m["ndvi_l2a"], m["ndvi_l1c"])[0, 1]), 4),
        "mean_difference": round(float(d.mean()), 4),
        "rmse": round(float(np.sqrt((d ** 2).mean())), 4),
        "max_abs_difference": round(float(d.abs().max()), 4),
        "roughness_here": round(rough(a["ndvi_l2a"]), 4),
        "roughness_eurocropsml": round(rough(b["ndvi_l1c"]), 4),
    }


def _palette():
    ensure_vendored()
    from space_time_deepsearch.vis import parcel_timeseries as pt
    return pt


def _comparison_figure(ours, ml, crop, pid, stats, dest: Path, show: bool = True):
    import matplotlib.pyplot as plt

    pt = _palette()
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.0), gridspec_kw={"width_ratios": [1.7, 1]})
    axes[0].plot(ours["date"], ours["median"], marker="o", ms=3.4, lw=1.6, color="#2a78d6",
                 label=f"here, Sentinel-2 L2A, screened ({len(ours)} dates)")
    axes[0].plot(ml["date"], ml["NDVI"], marker="s", ms=3.0, lw=1.2, ls="--", color=pt.MUTED,
                 label=f"EuroCropsML, L1C, as shipped ({len(ml)} dates)")
    axes[0].set_ylabel("NDVI")
    axes[0].set_title(f"parcel {pid}, {crop}: NDVI against the EuroCropsML series", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].tick_params(axis="x", rotation=30)

    a = ours[["date", "median"]].rename(columns={"median": "l2a"})
    b = ml[["date", "NDVI"]].rename(columns={"NDVI": "l1c"})
    for f in (a, b):
        f["date"] = pd.to_datetime(f["date"]).dt.tz_localize(None).dt.as_unit("ns").dt.normalize()
    m = pd.merge_asof(a.sort_values("date"), b.sort_values("date"), on="date",
                      tolerance=pd.Timedelta(days=1), direction="nearest").dropna()
    axes[1].scatter(m["l1c"], m["l2a"], s=20, color="#2a78d6", alpha=0.85)
    axes[1].plot([0, 1], [0, 1], color=pt.AXIS, ls=":", lw=1)
    axes[1].set_xlim(0, 1), axes[1].set_ylim(0, 1)
    axes[1].set_xlabel("EuroCropsML NDVI (L1C)")
    axes[1].set_ylabel("NDVI here (L2A)")
    axes[1].set_title(f"n={stats.get('matched_dates', 0)}  r={stats.get('pearson_r', float('nan')):.3f}  "
                      f"mean diff {stats.get('mean_difference', float('nan')):+.3f}", fontsize=10)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    print(f"wrote {dest}")
    if show and _in_notebook():
        from IPython.display import display as _display
        _display(fig)
    plt.close(fig)
    return dest


def _s1_figure(feat: pd.DataFrame, crop: str, pid: str, cc: str, year: int,
               ndvi: pd.DataFrame | None, dest: Path, show: bool = True):
    """Backscatter as acquired and with the per-track offset removed, over one season."""
    import matplotlib.pyplot as plt

    pt = _palette()
    fig, ax = plt.subplots(figsize=(11.5, 4.6))
    for col, colour, name in (("vv_median_db", "#2a78d6", "VV"),
                              ("vh_median_db", "#eb6834", "VH")):
        if col in feat:
            ax.scatter(feat["date"], feat[col], s=10, color=colour, alpha=0.35,
                       label=f"{name}, as acquired")
        if f"{col}_adj" in feat:
            ax.plot(feat["date"], feat[f"{col}_adj"], lw=1.5, color=colour,
                    label=f"{name}, per-track offset removed")
    ax.set_ylabel("gamma0 (dB), parcel median")
    ax.tick_params(axis="x", rotation=30)
    pt.style(ax, sub="Sentinel-1 RTC; reductions in linear power, shown in dB. Backscatter "
                     "responds to canopy structure and water, not to greenness")

    handles, labels = ax.get_legend_handles_labels()
    if ndvi is not None and len(ndvi):
        ax2 = ax.twinx()
        ax2.plot(ndvi["date"], ndvi["median"], lw=2.0, color="#1baf7a", alpha=0.8,
                 label="NDVI (Sentinel-2 L2A)")
        ax2.set_ylabel("NDVI")
        ax2.grid(False)
        h2, l2 = ax2.get_legend_handles_labels()
        handles, labels = handles + h2, labels + l2
    ax.legend(handles, labels, fontsize=8, loc="upper left", ncol=2)
    tracks = ", ".join(str(t) for t in sorted(set(feat["relative_orbit"])))
    fig.suptitle(f"Parcel {pid}  |  {crop.title()}  |  {COUNTRY_NAME[cc]} {year}  |  "
                 f"{len(feat)} acquisitions on tracks {tracks}",
                 x=0.006, ha="left", fontsize=13, fontweight="600", color=pt.INK)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140, bbox_inches="tight")
    print(f"wrote {dest}")
    if show and _in_notebook():
        from IPython.display import display as _display
        _display(fig)
    plt.close(fig)
    return dest


def _animate_s1(cube, feat: pd.DataFrame, crop: str, pid: str, cc: str, year: int,
                band: str, dest: Path, fps: int = 2):
    """The radar counterpart of the vendored NDVI animation: map plus sweeping cursor.

    The map is one polarisation in decibels on a fixed scale, so brightness means the same
    thing in every frame, and the cursor below marks which acquisition is on screen.
    """
    import matplotlib as mpl
    import matplotlib.animation as manim
    import matplotlib.pyplot as plt

    pt = _palette()
    arr = cube.sel(band=band).astype("float64")
    arr = arr.where(np.isfinite(arr) & (arr > 0))
    db = 10.0 * np.log10(arr)
    dates = pd.to_datetime(cube.time.values)
    keep = [i for i in range(len(dates)) if np.isfinite(db[i].values).any()]
    if not keep:
        print("  no usable frames, animation skipped")
        return None
    frames, fdates = db.values[keep], dates[keep]

    col = f"{band}_median_db_adj" if f"{band}_median_db_adj" in feat else f"{band}_median_db"
    ts_date = pd.to_datetime(feat["date"]).to_numpy()
    ts_val = feat[col].to_numpy(dtype=float)

    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.7], width_ratios=[1.0, 2.4, 1.0],
                          hspace=0.34, left=0.07, right=0.95, top=0.88, bottom=0.06)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_im = fig.add_subplot(gs[1, 1])

    ax_ts.plot(ts_date, ts_val, color="#eb6834", lw=1.7, marker="o", ms=3.5, zorder=3,
               label=f"{band.upper()} median, per-track offset removed")
    cursor = ax_ts.axvline(fdates[0], color=pt.INK, lw=2.2, alpha=0.9, zorder=6)
    dot, = ax_ts.plot([], [], marker="o", ms=10, mfc="none", mec=pt.INK, mew=2.0,
                      zorder=7, ls="none")
    ax_ts.set_ylabel(f"{band.upper()} (dB)")
    ax_ts.legend(loc="best", frameon=False, fontsize=8)
    pt.style(ax_ts, sub=f"{band.upper()} gamma0 - parcel median in dB; "
                        "black cursor = frame below")

    footprint = np.isfinite(frames).any(axis=0)
    ax_im.imshow(np.where(footprint, 1.0, np.nan),
                 cmap=mpl.colors.ListedColormap([pt.AXIS]), vmin=0, vmax=1,
                 interpolation="nearest", zorder=1)
    vmin = float(np.nanpercentile(frames, 2))
    vmax = float(np.nanpercentile(frames, 98))
    im = ax_im.imshow(frames[0], cmap="magma", vmin=vmin, vmax=vmax,
                      interpolation="nearest", zorder=2)
    ax_im.set_xticks([]), ax_im.set_yticks([])
    for sp in ax_im.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax_im, fraction=0.046, pad=0.02)
    cb.set_label(f"{band.upper()} gamma0 (dB)", color=pt.INK_2)
    cb.outline.set_visible(False)
    stamp = ax_im.set_title("", loc="center", color=pt.INK, fontsize=12,
                            fontweight="600", pad=10)
    fig.suptitle(f"Parcel {pid}  |  {crop.title()}  |  {COUNTRY_NAME[cc]} {year}  |  "
                 f"Sentinel-1 RTC {band.upper()}",
                 x=0.02, ha="left", fontsize=13, fontweight="600", color=pt.INK)

    orbit = dict(zip(pd.to_datetime(feat["date"]), feat.get("relative_orbit", pd.Series(dtype=int))))
    state = dict(zip(pd.to_datetime(feat["date"]), feat.get("orbit_state", pd.Series(dtype=str))))
    lookup = dict(zip(ts_date, ts_val))

    def draw(i):
        im.set_data(frames[i])
        d = fdates[i]
        cursor.set_xdata([d, d])
        v = lookup.get(np.datetime64(d))
        dot.set_data([d], [v]) if v is not None else dot.set_data([], [])
        px = int(np.isfinite(frames[i]).sum())
        track = orbit.get(pd.Timestamp(d))
        direction = str(state.get(pd.Timestamp(d), ""))[:3]
        stamp.set_text(f"{pd.Timestamp(d):%d %b %Y}    {px} px"
                       + (f"    track {int(track)} {direction}" if track is not None else "")
                       + (f"    {band.upper()} {v:.2f} dB" if v is not None else ""))
        return im, cursor, dot, stamp

    ani = manim.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ani.save(dest, writer=manim.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {dest}  ({len(frames)} frames @ {fps} fps)")
    return dest


# --------------------------------------------------------------------------------- display


def _relabel(fig, dest: Path, cc: str, year: int) -> None:
    """Replace the Austrian vocabulary the vendored chart writes into its header.

    ``plot`` composes its title around the GSA scheme and the ISV inspection register,
    neither of which exists here. The figure is finished by then, so the two text objects
    are edited in place and the file is written again rather than forking the function.
    """
    for text in fig.texts:
        value = text.get_text()
        if f"GSA {year}" in value:
            text.set_text(value.replace(f"GSA {year}", f"EuroCrops {COUNTRY_NAME[cc]} {year}"))
        if "not present in ISV" in value:
            text.set_text(value.replace("   |   not present in ISV",
                                        "   |   no field inspections in this dataset"))
    fig.savefig(dest, bbox_inches="tight")


def _in_notebook() -> bool:
    try:
        from IPython import get_ipython
        return "IPKernelApp" in get_ipython().config
    except Exception:
        return False


#: Above this size an animation is displayed through a downscaled copy, because a notebook
#: stores every frame again as base64 and a seven megabyte GIF makes the file unopenable.
GIF_INLINE_LIMIT = 1.5e6


def shrink_gif(path: Path, scale: float = 0.5, stride: int = 1,
               colors: int = 128) -> Path | None:
    """A smaller copy of an animation, for embedding; the original is left alone."""
    from PIL import Image, ImageSequence

    path = Path(path)
    if not path.exists():
        return None
    src = Image.open(path)
    frames = [f.copy().resize((max(1, int(f.width * scale)), max(1, int(f.height * scale))),
                              Image.LANCZOS).convert("P", palette=Image.ADAPTIVE, colors=colors)
              for i, f in enumerate(ImageSequence.Iterator(src)) if i % stride == 0]
    out = path.with_name(path.stem + "_small.gif")
    frames[0].save(out, save_all=True, append_images=frames[1:], loop=0,
                   duration=src.info.get("duration", 400) * stride, optimize=True)
    return out


def _show_gif(path) -> bool:
    """Display an animation in a notebook, downscaling it first when it is large."""
    if path is None or not _in_notebook():
        return False
    from IPython.display import Image, display

    path = Path(path)
    shown, size = path, path.stat().st_size
    if size > GIF_INLINE_LIMIT:
        # Halving the frame count as well as the size keeps a season readable while cutting
        # what the notebook stores by roughly four.
        stride = 2 if size > 2.5e6 else 1
        shown = shrink_gif(path, scale=0.45, stride=stride, colors=96) or path
    display(Image(filename=str(shown)))
    return True
