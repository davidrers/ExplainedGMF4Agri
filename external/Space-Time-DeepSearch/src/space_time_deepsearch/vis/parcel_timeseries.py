"""Sentinel-2 vegetation-index time series for a single parcel.

Pulls the S2 L2A cube for one GSA parcel, clips it to the parcel polygon,
reduces each date to the MEDIAN over the parcel's pixels, and plots five
vegetation indices plus a spatial-heterogeneity panel. The GSA declaration is
shown in the header; if the parcel also appears in ISV, each inspection is
drawn as a vertical line coloured by the observed BODENBEDECKUNG class.

No temporal segmentation, no LandTrendr - just the raw per-date medians.

Indices (all from Sentinel-2 L2A surface reflectance):
  NDVI  (B08-B04)/(B08+B04)                   greenness / standing biomass
  NDRE  (B08-B05)/(B08+B05)                   red edge; saturates later
  EVI   2.5*(B08-B04)/(B08+6*B04-7.5*B02+1)   structure, resists saturation
  EVI2  2.5*(B08-B04)/(B08+2.4*B04+1)         EVI without the noisy blue band
  NDMI  (B08-B11)/(B08+B11)                   canopy water content

Spatial heterogeneity: alongside the median, each date also gets a global
MORAN'S I over the parcel's pixels (rook contiguity) with its randomisation
z-score - how spatially organised the field is, which the median cannot see.
A partial cut splits the parcel into two contiguous zones and drives I up;
uniform grass, or uniform stubble, leaves I near 0.

As a library - one call does everything
--------------------------------------
    from space_time_deepsearch.vis import parcel_timeseries as pt
    r = pt.explore(233103922, 2024)          # chart + animation + numbers
    r.series["NDVI"], r.table, r.png, r.gif

As a CLI
--------
    python -m space_time_deepsearch.vis.parcel_timeseries --kennung 123456789
    python -m space_time_deepsearch.vis.parcel_timeseries --sl-id 987654321 --year 2025
    python -m space_time_deepsearch.vis.parcel_timeseries --isv-only --label Beweidung
    python -m space_time_deepsearch.vis.parcel_timeseries --kennung 1 --animate --csv

Data and output locations
-------------------------
The GSA and ISV GeoPackages are read from `./data` and results are written to
`./output`, both relative to the working directory. Set STDS_PARCEL_DATA and
STDS_PARCEL_OUT to point elsewhere; they are read once, at import.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import os
import sqlite3
import warnings
from dataclasses import dataclass
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Inside the package there is no project root to anchor on, so data and output
# default to the working directory, which reproduces the old script's layout
# when run from its project root.
DATA = Path(os.environ.get("STDS_PARCEL_DATA", Path.cwd() / "data"))
OUT = Path(os.environ.get("STDS_PARCEL_OUT", Path.cwd() / "output"))

# The Sentinel-2 STAC loader is a sibling module in this package.
S2_LOADER = Path(__file__).resolve().parent.parent / "io" / "sentinel2.py"

YEARS = {
    2024: {
        "at": DATA / "AT_MFA2024_20241201_Neo.gpkg",
        "isv": DATA / "ISV_MFA2024_20241220_Neo.gpkg",
        "isv_layer": "ISV_MFA2024_20241220_Neo",
        "season": ("2024-03-15", "2024-12-15"),
    },
    2025: {
        "at": DATA / "AT_MFA2025_20251201_Neo.gpkg",
        "isv": DATA / "ISV_MFA2025_20250128_Neo.gpkg",
        # NOTE: this layer name really does end in a newline.
        "isv_layer": "ISV_MFA2025_20250128_Neo\n",
        "season": ("2025-03-15", "2025-12-15"),
    },
}

# ---------------------------------------------------------------- palette
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
CLASS_COLOR = {
    "Gemaeht": "#2a78d6",       # slot 1 - mown
    "Beweidung": "#eb6834",     # slot 2 - grazed
    "Gehaeckselt": "#1baf7a",   # slot 3 - mulched
}
# English labels for every BODENBEDECKUNG value, condensed from the vendor's
# own definitions in the `ISV` sheet of MFA_ISV_Info_Neo_updated_Max.xlsx.
# The five AMA highlights green as mowing-relevant are marked below.
CLASS_LABEL = {
    # -- the five classes AMA flags as mowing-relevant -------------------
    "Gemaeht": "Mown",
    "Beweidung": "Grazed",
    "Gehaeckselt": "Mulched (chopped, residues left)",
    "Kultur": "Crop growing (no mowing, not grazed)",
    "Abgereifte Kultur": "Senesced (dead/ripe vegetation)",
    # -- quality / non-agricultural flags --------------------------------
    "Falsche Kultur": "Wrongly declared crop type",
    "NLN nicht versiegelt": "Non-agricultural, unsealed",
    "NLN versiegelt": "Non-agricultural, sealed",
    "Unkraut": "Weeds",
    # -- arable crop stages ----------------------------------------------
    "BareSoil": "Bare soil",
    "Begruent": "Vegetated (winter catch crop)",
    "Jungpflanzen": "Young plants",
    "Folgekultur": "Second crop of the season",
    "Geerntet": "Harvested",
    "Stoppelsturz": "Harvested + shallow residue incorporation",
    # -- present in the data but NOT documented in the vendor sheet -------
    "Anbau": "Sown / planted (undocumented class)",
    "Wendende Bodenbearbeitung": "Inversion tillage (2025 only)",
    "Nicht wendende Bodenbearbeitung/Bare Soil":
        "Non-inversion tillage / bare soil (2025 only)",
}
OTHER_COLOR = MUTED

# Line style per class, FIXED so a class always looks the same in every plot
# (style follows the entity, never its position in this parcel's list).
#
# Only the three management actions carry a hue: a 4th categorical colour
# cannot clear the all-pairs CVD and normal-vision floors beside blue/orange/
# aqua - magenta, green, red and yellow were each run through the palette
# validator and all four fail. Every other class is therefore separated by
# COMPOSITE encoding: three ink levels x six dash patterns, ordered so that
# classes likely to co-occur differ in both lightness and pattern.
DASH = {
    "solid": "-",
    "dot": (0, (1, 2)),
    "dash": (0, (5, 2)),
    "dashdot": (0, (5, 1, 1, 1)),
    "longdash": (0, (9, 3)),
    "dashdotdot": (0, (6, 1, 1, 1, 1, 1)),
}
CLASS_STYLE = {
    # -- management actions: hue-coded, solid, heavier ---------------------
    "Gemaeht":     (CLASS_COLOR["Gemaeht"],     DASH["solid"], 1.8),
    "Beweidung":   (CLASS_COLOR["Beweidung"],   DASH["solid"], 1.8),
    "Gehaeckselt": (CLASS_COLOR["Gehaeckselt"], DASH["solid"], 1.8),
    # -- the other two AMA-highlighted classes ----------------------------
    "Kultur":            (MUTED, DASH["solid"],   1.2),
    "Abgereifte Kultur": (INK_2, DASH["dashdot"], 1.5),
    # -- quality / non-agricultural flags ---------------------------------
    "Falsche Kultur":       (INK_2, DASH["dash"],       1.5),
    "NLN nicht versiegelt": (MUTED, DASH["dot"],        1.3),
    "NLN versiegelt":       (INK_2, DASH["dot"],        1.3),
    "Unkraut":              (MUTED, DASH["dashdotdot"], 1.3),
    # -- arable crop stages -----------------------------------------------
    "BareSoil":     (AXIS,  DASH["longdash"],   1.6),
    "Begruent":     (AXIS,  DASH["dash"],       1.5),
    "Jungpflanzen": (AXIS,  DASH["dot"],        1.4),
    "Folgekultur":  (AXIS,  DASH["dashdot"],    1.5),
    "Geerntet":     (INK_2, DASH["dashdotdot"], 1.4),
    "Stoppelsturz": (MUTED, DASH["dash"],       1.3),
    "Anbau":        (MUTED, DASH["longdash"],   1.4),
    # -- 2025-only tillage classes ----------------------------------------
    "Wendende Bodenbearbeitung":                 (INK_2, DASH["longdash"],   1.5),
    "Nicht wendende Bodenbearbeitung/Bare Soil": (AXIS,  DASH["dashdotdot"], 1.5),
}
FALLBACK_STYLE = (MUTED, DASH["dashdot"], 1.2)


def class_style(cls: str):
    """Colour, dash pattern and width for a BODENBEDECKUNG value."""
    return CLASS_STYLE.get(cls, FALLBACK_STYLE)


def class_label(cls: str, short: bool = False) -> str:
    """English label for a BODENBEDECKUNG value.

    `short=True` drops the parenthetical gloss, for compact contexts such as
    the header summary. Unknown values fall back to the raw German so a new
    class in a future delivery is visible rather than silently mislabelled.
    """
    label = CLASS_LABEL.get(cls)
    if label is None:
        return f"{cls} (untranslated)"
    return label.split(" (")[0] if short else label


INDEX_COLOR = {"NDVI": "#2a78d6", "NDRE": "#b0468c", "EVI": "#4a3aa7",
               "EVI2": "#7a63d9", "NDMI": "#1baf7a"}
INDEX_DESC = {
    "NDVI": "greenness / standing biomass",
    "NDRE": "red edge - saturates later than NDVI",
    "EVI": "structure (saturation-resistant)",
    "EVI2": "EVI without the blue band",
    "NDMI": "canopy water content",
}
INDEX_ORDER = ["NDVI", "NDRE", "EVI", "EVI2", "NDMI"]

# The Moran panel is not a category alongside the indices - it is a different
# quantity - so it gets graphite rather than a sixth hue.
MORAN_COLOR = "#3f3d3a"
MORAN_INDEX = "NDVI"        # which index the Moran panel is computed on

# The animation's series panel draws NDMI, not NDVI: NDMI is the mowing
# feature (AUC 0.851 event / 0.904 parcel vs NDVI's 0.78-0.85 - mowing removes
# *wet* biomass, so canopy water is the right physics). The map underneath
# stays NDVI, which is what the green ramp is semantic for.
ANIM_INDEX = "NDMI"

# ---- persistence filter --------------------------------------------------
# A drop is only believable if it is still there at the next clear date.
PERSIST_MIN_DROP = 0.05     # below this a dip is noise, not worth judging
PERSIST_MAX_GAP = 16        # days - beyond this the next look is too far off
PERSIST_RETAIN = 0.5        # fraction of the drop that must survive

# ---- seasonal baseline ---------------------------------------------------
# Phenology, fitted so a cut cannot hide inside it. See seasonal_baseline().
BASELINE_BANDWIDTH = 35.0   # days, tricube half-width - a cut is 5-10 days
BASELINE_ITERATIONS = 4     # robustness passes
BASELINE_ENVELOPE = 0.4     # <1 downweights drops harder -> upper envelope
BASELINE_MIN_OBS = 8        # below this the fit interpolates noise

# Plot geometry. The figure grows downward to fit however many panels and
# legend rows this parcel needs, so nothing is ever clipped.
FIGWIDTH = 15.0             # inches
PANEL_IN = 2.4              # inches per stacked panel
LEGEND_NCOL = 3
HEADER_IN = 1.00            # title + subtitle band above the plots
XLABEL_IN = 0.55            # x tick labels + "date" axis label


# ============================================================ environment
def fix_proj_env() -> None:
    """Drop PROJ_LIB / PROJ_DATA when they point somewhere without a proj.db.

    This machine has PROJ_LIB set system-wide to another user's Anaconda env
    (C:\\Users\\maria.perez\\...), which this account cannot read. rasterio then
    fails with "Cannot find proj.db" on any CRS lookup. pyproj, rasterio and
    pyogrio each ship their own PROJ data, so the fix is simply to stop
    pointing at the broken path and let them use the bundled copies.
    """
    for var in ("PROJ_LIB", "PROJ_DATA"):
        path = os.environ.get(var)
        if path and not Path(path, "proj.db").is_file():
            print(f"note: {var}={path} has no proj.db - unsetting it "
                  "so the bundled PROJ data is used")
            del os.environ[var]


# ============================================================ S2 loader
def load_s2_loader():
    """Import `get_sentinel2_imagery` from the sibling `io/sentinel2.py`.

    Loaded straight from its file rather than as `space_time_deepsearch.io.
    sentinel2`, because that package's __init__ chain pulls in osmnx, torch,
    holoviews and panel - none of which this module needs.
    """
    import importlib.util

    src = S2_LOADER
    if not src.exists():
        raise SystemExit(
            f"Sentinel-2 loader not found at {src}\n"
            "The space_time_deepsearch installation is incomplete.")

    fix_proj_env()
    spec = importlib.util.spec_from_file_location("_std_sentinel2", src)
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except ImportError as exc:
        raise SystemExit(
            f"The Sentinel-2 loader needs a package this project lacks: {exc}\n"
            "  poetry add pystac-client stackstac planetary-computer "
            '"rioxarray@<0.16.0" dask') from exc
    return mod.get_sentinel2_imagery


# ============================================================ data access
def load_parcel(year: int, kennung: int | None, sl_id: int | None) -> gpd.GeoDataFrame:
    """Read one parcel from the GSA layer using an attribute filter."""
    cfg = YEARS[year]
    where = f"KENNUNG = {int(kennung)}" if kennung is not None else f"SL_ID = {int(sl_id)}"
    gdf = gpd.read_file(cfg["at"], layer="parcels", where=where)
    if gdf.empty:
        raise SystemExit(f"No parcel matching {where} in {cfg['at'].name}")
    return gdf


def load_isv(year: int, kennung: int) -> pd.DataFrame:
    """Inspections for this parcel, chronological. Empty frame if none."""
    cfg = YEARS[year]
    if not cfg["isv"].exists():
        return pd.DataFrame()
    con = sqlite3.connect(cfg["isv"])
    df = pd.read_sql_query(
        f'''SELECT date(BESICHTIGUNGSDATUM) AS inspection_date,
                   BODENBEDECKUNG          AS cls,
                   date(EVENTDATUM)        AS event_date,
                   KZ_HL_VORHANDEN         AS subarea,
                   BESICHTIGUNGSFLAECHE    AS inspected_m2,
                   KOMMENTAR               AS comment
            FROM "{cfg["isv_layer"]}"
            WHERE SL_KENNUNG = ?
            ORDER BY BESICHTIGUNGSDATUM''',
        con, params=(int(kennung),))
    con.close()
    if not df.empty:
        df["inspection_date"] = pd.to_datetime(df.inspection_date)
        df["event_date"] = pd.to_datetime(df.event_date)
    return df


def pick_parcel(year: int, label: str | None, min_visits: int) -> int:
    """Pick a well-observed KENNUNG from ISV - handy for exploration.

    `min_visits` counts ALL inspections of the parcel; `label` only requires
    that at least one of them recorded that class. Conflating the two asks for
    e.g. five separate 'Gemaeht' observations on one parcel, which barely exists.
    """
    cfg = YEARS[year]
    con = sqlite3.connect(cfg["isv"])
    lbl = "AND SUM(CASE WHEN BODENBEDECKUNG = ? THEN 1 ELSE 0 END) > 0" if label else ""
    df = pd.read_sql_query(
        f'''SELECT SL_KENNUNG,
                   COUNT(*) AS n_visits,
                   SUM(CASE WHEN BODENBEDECKUNG = ? THEN 1 ELSE 0 END) AS n_label
            FROM "{cfg["isv_layer"]}"
            GROUP BY SL_KENNUNG
            HAVING COUNT(*) >= ? {lbl}
            ORDER BY n_visits DESC, SL_KENNUNG
            LIMIT 25''',
        con, params=[label or "", min_visits] + ([label] if label else []))
    con.close()
    if df.empty:
        raise SystemExit(
            f"No parcel with >={min_visits} inspections"
            + (f" including a '{label}' observation." if label else "."))
    top = df.iloc[0]
    note = f", {top.n_label}x {label}" if label else ""
    print(f"Auto-picked KENNUNG={top.SL_KENNUNG} ({top.n_visits} inspections{note}). "
          f"Other candidates: {list(df.SL_KENNUNG[1:6])}")
    return int(top.SL_KENNUNG)


# ============================================================ indices
# Sentinel-2 L2A is delivered as integer DN, not reflectance:
#     reflectance = (DN + BOA_ADD_OFFSET) / QUANTIFICATION_VALUE
# Since Processing Baseline 04.00 (2022-01-25) BOA_ADD_OFFSET is -1000, and
# stackstac hands back the raw DN. The offset is ADDITIVE, so it does NOT
# cancel in a normalised difference: skipping it shifts NDVI by ~0.2 and makes
# EVI meaningless (its +1 constant assumes 0-1 reflectance).
BOA_ADD_OFFSET = -1000.0
QUANTIFICATION_VALUE = 10000.0

# Extra Scene Classification classes to drop, ON TOP of the loader's cloud mask
# ({3,8,9,10} = cloud shadow, medium/high cloud, thin cirrus).
#
# This is a BLACKLIST on purpose. An allowlist of {4 vegetation, 5 bare} was
# measured against 8 meadow parcels with dated cuts and would discard 5.76% of
# pixels in the 12 days after a cut versus 2.20% over the rest of the season -
# SCL falls back to 7 UNCLASSIFIED on freshly mown ground (1.55% -> 5.68%), so
# an allowlist deletes the positive class 2.6x more often than the background.
#
#   0  NO_DATA              not an observation
#   1  SATURATED/DEFECTIVE  sensor defect
#  11  SNOW / ICE           low NIR + high visible -> NDVI collapse that mimics
#                           a cut; 25% of pixel-observations and SIX entirely
#                           snow-covered dates on an alpine BERGMAEHDER parcel
#
# Deliberately NOT masked: 2 CAST_SHADOW and 7 UNCLASSIFIED - both carry real
# ground, and 7 in particular is where freshly mown grass ends up.
SCL_DROP_DEFAULT = (0, 1, 11)
SCL_NAMES = {
    0: "NO_DATA", 1: "SATURATED/DEFECTIVE", 2: "CAST_SHADOW", 3: "CLOUD_SHADOW",
    4: "VEGETATION", 5: "NOT_VEGETATED", 6: "WATER", 7: "UNCLASSIFIED",
    8: "CLOUD medium", 9: "CLOUD high", 10: "THIN_CIRRUS", 11: "SNOW/ICE",
}


# ============================================================ heterogeneity
# Minimum valid pixels before a Moran's I is reported. Below ~20 the
# randomisation variance is dominated by the small-sample correction and the
# statistic swings wildly between dates for no physical reason.
MIN_MORAN_PIXELS = 20


def moran_i(arr) -> tuple[float, float]:
    """Global Moran's I over one date's pixel grid, and its z-score.

    Spatial autocorrelation, i.e. how ORGANISED the parcel is - the parcel
    median cannot distinguish "half the field cut" from "the whole field
    grazed down a little", but I can: a partial cut leaves two large
    contiguous zones (I high), uniform growth or uniform stubble leaves
    neighbouring pixels no more alike than distant ones (I near 0).

    Weights are ROOK contiguity on the pixel grid (the four edge-sharing
    neighbours), binary, restricted to valid pixels - so cloud gaps and the
    parcel boundary simply remove links rather than fabricating them.

        I = (n / W) * SUM_ij w_ij z_i z_j / SUM_i z_i^2 ,  z = x - mean(x)

    The z-score is analytic under the RANDOMISATION assumption (Cliff & Ord
    1981): it conditions on the observed values and asks only whether their
    arrangement is special, which is what we want - a mown field's values are
    not remotely normal. E[I] = -1/(n-1), not 0, and the difference matters at
    the n of a small parcel (~100 px at 10 m on 1 ha).

    Returns (nan, nan) when there are too few pixels, no adjacent pairs, or
    zero variance.

    CAVEAT: only NDVI is computed from natively 10 m bands. NDRE and NDMI
    inherit B05 / B11, which are 20 m upsampled to the 10 m grid, so each
    value is duplicated across 2x2 blocks and I is inflated toward 1 by
    construction. Compare Moran across DATES of one index, never across
    indices. MORAN_INDEX is NDVI for this reason.
    """
    a = np.asarray(arr, dtype="float64")
    m = np.isfinite(a)
    n = int(m.sum())
    if n < MIN_MORAN_PIXELS:
        return float("nan"), float("nan")

    z = np.where(m, a - a[m].mean(), 0.0)

    # Rook pairs, counted once each: horizontal then vertical.
    hp, vp = m[:, :-1] & m[:, 1:], m[:-1, :] & m[1:, :]
    n_pairs = int(hp.sum() + vp.sum())
    s_zz = float((z[:, :-1] * z[:, 1:])[hp].sum() + (z[:-1, :] * z[1:, :])[vp].sum())
    s2 = float((z[m] ** 2).sum())
    if n_pairs == 0 or s2 <= 0:
        return float("nan"), float("nan")

    # W counts ORDERED pairs, so W = 2 * n_pairs and the doubled cross-product
    # cancels: I = (n / W) * 2 * s_zz / s2.
    i_val = n * s_zz / (n_pairs * s2)
    if n < 5:
        return i_val, float("nan")

    # Neighbour count per pixel, for S2 below.
    k = np.zeros(a.shape, dtype="float64")
    k[:, :-1] += hp; k[:, 1:] += hp
    k[:-1, :] += vp; k[1:, :] += vp

    w = 2.0 * n_pairs
    s1 = 2.0 * w                                  # 0.5 * sum (w_ij + w_ji)^2
    s2_w = 4.0 * float((k[m] ** 2).sum())         # sum_i (w_i. + w_.i)^2
    b2 = n * float((z[m] ** 4).sum()) / (s2 ** 2)  # kurtosis of the values

    var = ((n * ((n * n - 3 * n + 3) * s1 - n * s2_w + 3 * w * w)
            - b2 * ((n * n - n) * s1 - 2 * n * s2_w + 6 * w * w))
           / ((n - 1) * (n - 2) * (n - 3) * w * w)
           - 1.0 / ((n - 1) ** 2))
    if not np.isfinite(var) or var <= 0:
        return i_val, float("nan")
    return i_val, (i_val + 1.0 / (n - 1)) / math.sqrt(var)


def compute_indices(cube, offset: float = BOA_ADD_OFFSET,
                    scl_drop=SCL_DROP_DEFAULT) -> dict:
    """Per-date MEDIAN over the parcel's pixels, for each index.

    Also computes a global Moran's I per date (see moran_i) so spatial
    structure is available next to the median it cannot be read from.
    Applies the extra SCL blacklist (see SCL_DROP_DEFAULT) before reducing.
    """
    valid = None
    if scl_drop and "SCL" in list(cube.band.values):
        scl = cube.sel(band="SCL")
        valid = ~scl.isin(list(scl_drop))
        dropped = {}
        for code in scl_drop:
            n = int((scl == code).sum())
            if n:
                dropped[SCL_NAMES.get(code, code)] = n
        if dropped:
            print("  SCL blacklist removed: "
                  + ", ".join(f"{k} {v:,}px" for k, v in dropped.items()))

    have = list(cube.band.values)
    b = {}
    for name in ("B02", "B04", "B05", "B08", "B11"):
        if name not in have:
            continue
        arr = (cube.sel(band=name).astype("float32") + offset) / QUANTIFICATION_VALUE
        b[name] = arr if valid is None else arr.where(valid)

    with np.errstate(divide="ignore", invalid="ignore"):
        raw = {
            "NDVI": (b["B08"] - b["B04"]) / (b["B08"] + b["B04"]),
            "EVI": 2.5 * (b["B08"] - b["B04"]) / (
                b["B08"] + 6.0 * b["B04"] - 7.5 * b["B02"] + 1.0),
            # EVI2 drops the blue term. Blue is the noisiest L2A band
            # (aerosol residual), and on a small parcel that noise is not
            # averaged away, so EVI2 is often the steadier of the two.
            "EVI2": 2.5 * (b["B08"] - b["B04"]) / (
                b["B08"] + 2.4 * b["B04"] + 1.0),
            "NDMI": (b["B08"] - b["B11"]) / (b["B08"] + b["B11"]),
        }
        # B05 is a 20 m band; it is only present when it was requested.
        if "B05" in b:
            raw["NDRE"] = (b["B08"] - b["B05"]) / (b["B08"] + b["B05"])

    out = {}
    for name in INDEX_ORDER:
        if name not in raw:
            continue
        arr = raw[name].where(np.isfinite(raw[name]))
        med = arr.median(dim=("y", "x"), skipna=True)
        cnt = arr.notnull().sum(dim=("y", "x"))
        # Moran's I is per DATE, on the pixel grid, so it cannot come out of
        # an xarray reduction - one call per time slice.
        vals = arr.values
        mor = np.array([moran_i(vals[i]) for i in range(vals.shape[0])],
                       dtype="float64").reshape(-1, 2)
        out[name] = pd.DataFrame({
            "date": pd.to_datetime(med.time.values),
            "median": med.values,
            "n_pixels": cnt.values,
            "moran": mor[:, 0],
            "moran_z": mor[:, 1],
        }).dropna(subset=["median"]).sort_values("date").reset_index(drop=True)
    return out


# ==================================================== persistence filter
def persistence_mask(dates, values,
                     min_drop: float = PERSIST_MIN_DROP,
                     max_gap: int = PERSIST_MAX_GAP,
                     retain: float = PERSIST_RETAIN) -> np.ndarray:
    """Which observations survive the persistence test. True = keep.

    A cut removes biomass and regrowth takes weeks, so a real drop is still
    there at the next clear date. Haze, thin cloud and cast shadow hit one
    acquisition and are gone by the next pass. So for each interior point,

        retention = (before - after) / (before - dip)

    and the dip is rejected when that falls below `retain` - most of the drop
    evaporated, so nothing actually left the field.

    Measured on parcel 233103922-style series in the 584-parcel 2024 sample:
    at retain=0.5 this keeps 64% of NDVI drops at dated cuts against 43% of
    drops elsewhere. It removes **transient** artefacts only - NDMI's seasonal
    decline is itself persistent and sails through it (75% vs 74%), which is
    why removing the seasonal trend is a separate job, not this one.

    Worked example, parcel 229720359 (NDVI): DOY 197 = 0.467, DOY 210 = 0.301
    (a drop of 0.166), DOY 212 = 0.428. retention = 0.039/0.166 = 0.24, so
    DOY 210 is rejected - a field does not regrow three quarters of its canopy
    in 48 hours. It was aerosol.

    Two deliberate simplifications: the first and last observations are always
    kept (there is no "next" to judge them against), and a rejected point still
    serves as the `before` for its successor, so a run of consecutive artefacts
    is judged against raw neighbours rather than cleaned ones.
    """
    v = np.asarray(values, dtype=float)
    keep = np.ones(len(v), dtype=bool)
    if len(v) < 3:
        return keep
    t = pd.to_datetime(pd.Series(dates)).to_numpy()
    days = (t - t[0]) / np.timedelta64(1, "D")
    for i in range(1, len(v) - 1):
        if not np.isfinite(v[i - 1:i + 2]).all():
            continue
        drop = v[i - 1] - v[i]
        if drop < min_drop:
            continue                      # too small to be worth judging
        if days[i + 1] - days[i] > max_gap:
            continue                      # next look too far off to judge it
        if (v[i - 1] - v[i + 1]) / drop < retain:
            keep[i] = False
    return keep


# ==================================================== seasonal baseline
def seasonal_baseline(dates, values, bandwidth: float = BASELINE_BANDWIDTH,
                      iterations: int = BASELINE_ITERATIONS,
                      envelope: float = BASELINE_ENVELOPE,
                      min_obs: int = BASELINE_MIN_OBS) -> np.ndarray:
    """The parcel's own phenology, fitted so a cut cannot hide inside it.

    Robust local **linear** regression (LOWESS) on the raw irregular dates, with
    an asymmetric twist. Three choices, each forced by something measured:

    * **LOWESS, not Savitzky-Golay.** SG assumes evenly spaced samples; our
      gaps run 2-15 days, so it would distort worst exactly where the clouds
      were. Using it at all would mean interpolating to a regular grid first,
      and interpolating before cleaning smears a bad date into its neighbours.
    * **Bandwidth in DAYS, not in neighbours.** A window of "9 nearest points"
      is 20 days in a clear spell and 90 in a cloudy one, so the amount of
      smoothing would drift with the weather. `bandwidth` is a half-width in
      days and the tricube weight goes to zero beyond it.
    * **Asymmetric robustness - the upper envelope.** Ordinary LOWESS with a
      35-day window still bends toward a deep cut, and then the residual that
      was supposed to reveal the cut has been fitted away. So residuals BELOW
      the curve are downweighted `envelope`x harder than those above, and the
      fit converges on the parcel's *unmown potential* instead of its mean.
      Cuts, grazing and mulchings all fall out underneath it.

    Local *linear* rather than local constant because the season's ends are
    where early-May and October cuts sit, and a constant fit flattens the trend
    there and manufactures residuals.

    Returns NaN everywhere if fewer than `min_obs` finite values - below that
    the fit is interpolating noise. Feed it the series AFTER `persistence_mask`:
    an unrejected haze point drags the curve down and hides the very drop you
    are looking for.
    """
    v = np.asarray(values, dtype=float)
    t = pd.to_datetime(pd.Series(dates)).to_numpy()
    x = (t - t[0]) / np.timedelta64(1, "D")
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    if ok.sum() < min_obs:
        return out

    xf, vf = x[ok], v[ok]
    w_rob = np.ones(len(vf))
    fit = np.empty(len(vf))
    for it in range(max(1, iterations)):
        for i, x0 in enumerate(xf):
            d = np.abs(xf - x0) / bandwidth
            w = np.where(d < 1, (1 - d ** 3) ** 3, 0.0) * w_rob
            if (w > 0).sum() < 3:                 # too thin for a slope
                fit[i] = np.average(vf, weights=np.maximum(w, 1e-12))
                continue
            # weighted least squares on [1, x - x0]; the intercept IS the fit
            dx = xf - x0
            sw = w.sum()
            mx = (w * dx).sum() / sw
            my = (w * vf).sum() / sw
            sxx = (w * (dx - mx) ** 2).sum()
            sxy = (w * (dx - mx) * (vf - my)).sum()
            slope = sxy / sxx if sxx > 1e-12 else 0.0
            fit[i] = my + slope * (0.0 - mx)
        if it == iterations - 1:
            break
        r = vf - fit
        s = np.median(np.abs(r))
        if s <= 1e-9:
            break
        # negative residuals are divided by a SMALLER scale, so they reach the
        # biweight's zero sooner - that is the upper envelope.
        u = np.where(r >= 0, r / (6 * s), r / (6 * s * envelope))
        w_rob = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)

    out[ok] = fit
    return out


# ============================================================ animation
# Sequential single-hue green ramp for the NDVI map: light -> dark, semantic
# for vegetation, always shipped with a colourbar so colour never carries the
# value alone.
NDVI_RAMP = ["#f2f8f2", "#d7ecd7", "#b3dbb3", "#86c586", "#57ab5d",
             "#2f8f43", "#1c7034", "#125327", "#0b3a1c"]


def animate(cube, series, isv, parcel, year, kennung, dest,
            fps=2, offset=BOA_ADD_OFFSET, scl_drop=SCL_DROP_DEFAULT,
            ts_band: str = ANIM_INDEX):
    """GIF: NDVI map per date, with a cursor sweeping the `ts_band` series.

    The cloned repo's `create_timelapse` builds its own figure and writes the
    GIF itself, so it cannot drive a cursor on a second axes. This is the same
    idea in one shared FuncAnimation so the map and the cursor stay in step.

    The series panel draws `ts_band` (NDMI by default) twice - as delivered and
    after `persistence_mask` - so the frames that fail the filter can be read
    against the imagery that produced them. The map stays NDVI.
    """
    import matplotlib.animation as manim

    # NDVI per pixel per date, same corrections as compute_indices
    valid = None
    if scl_drop and "SCL" in list(cube.band.values):
        valid = ~cube.sel(band="SCL").isin(list(scl_drop))
    ref = {n: (cube.sel(band=n).astype("float32") + offset) / QUANTIFICATION_VALUE
           for n in ("B04", "B08")}
    if valid is not None:
        ref = {n: a.where(valid) for n, a in ref.items()}
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (ref["B08"] - ref["B04"]) / (ref["B08"] + ref["B04"])
    ndvi = ndvi.where(np.isfinite(ndvi))

    dates = pd.to_datetime(cube.time.values)
    keep = [i for i, _ in enumerate(dates) if np.isfinite(ndvi[i].values).any()]
    if not keep:
        print("  no usable frames - animation skipped")
        return None
    frames = ndvi.values[keep]
    fdates = dates[keep]

    ts = series[ts_band]
    ts_date = ts.date.to_numpy()
    ts_val = ts["median"].to_numpy(dtype=float)
    keep = persistence_mask(ts_date, ts_val)
    cmap = mpl.colors.LinearSegmentedColormap.from_list("ndvi", NDVI_RAMP)

    # The series spans the full width; the map sits in a centred middle column
    # so the colourbar cannot shove it off-centre.
    fig = plt.figure(figsize=(11.5, 8.2))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.7],
                          width_ratios=[1.0, 2.4, 1.0], hspace=0.34,
                          left=0.07, right=0.95, top=0.88, bottom=0.06)
    ax_ts = fig.add_subplot(gs[0, :])
    ax_im = fig.add_subplot(gs[1, 1])

    # --- the series, with the ISV inspections for context ------------------
    for _, r in isv.iterrows():
        c, ls, lw = class_style(r.cls)
        ax_ts.axvline(r.inspection_date, color=c, lw=lw, ls=ls, alpha=0.8, zorder=1)
    # Before and after are the SAME quantity, not two categories, so they share
    # the index's hue and separate by ink level + dash - the project's rule for
    # anything past the three validated class hues.
    ax_ts.plot(ts_date, ts_val, color=MUTED, lw=1.2, ls=DASH["dash"],
               marker="o", ms=3, zorder=2, label="as delivered")
    ax_ts.plot(ts_date[keep], ts_val[keep], color=INDEX_COLOR[ts_band], lw=1.8,
               marker="o", ms=4, zorder=3, label="after persistence filter")
    base = seasonal_baseline(ts_date, np.where(keep, ts_val, np.nan))
    if np.isfinite(base).any():
        ax_ts.plot(ts_date, base, color=INDEX_COLOR[ts_band], lw=1.4, alpha=0.45,
                   zorder=1, label="seasonal baseline (upper envelope)")
    n_cut = int((~keep).sum())
    if n_cut:
        ax_ts.plot(ts_date[~keep], ts_val[~keep], ls="none", marker="x", ms=8,
                   mew=1.8, color=INK_2, zorder=4,
                   label=f"rejected - drop did not persist ({n_cut})")
    # One column and loc="best": the panel is short, so a wide legend collides
    # with the series, and which corner is free depends on the parcel's own
    # season - let matplotlib place it rather than hardcoding a guess. Outside
    # the axes is not an option either: below it lands on the map's caption.
    ax_ts.legend(loc="best", frameon=False, fontsize=8, ncol=1,
                 handlelength=2.2, borderaxespad=0.4)
    cursor = ax_ts.axvline(fdates[0], color=INK, lw=2.2, alpha=0.9, zorder=6)
    dot, = ax_ts.plot([], [], marker="o", ms=10, mfc="none", mec=INK,
                      mew=2.0, zorder=7, ls="none")
    ax_ts.set_ylabel(ts_band)
    style(ax_ts, sub=f"{ts_band} - parcel median, {INDEX_DESC[ts_band]}; "
                     "black cursor = frame below (map is NDVI)")

    # --- the map ------------------------------------------------------------
    # A masked pixel is NaN, and NaN renders TRANSPARENT - which would make it
    # indistinguishable from the area outside the parcel. So draw the parcel
    # footprint underneath in a flat neutral: masked pixels then read as
    # "inside the field, no usable observation" rather than vanishing.
    footprint = np.isfinite(frames).any(axis=0)
    ax_im.imshow(np.where(footprint, 1.0, np.nan),
                 cmap=mpl.colors.ListedColormap([AXIS]), vmin=0, vmax=1,
                 interpolation="nearest", zorder=1)

    vmin = float(np.nanpercentile(frames, 2))
    vmax = float(np.nanpercentile(frames, 98))
    im = ax_im.imshow(frames[0], cmap=cmap, vmin=vmin, vmax=vmax,
                      interpolation="nearest", zorder=2)
    ax_im.set_xticks([]); ax_im.set_yticks([])
    for sp in ax_im.spines.values():
        sp.set_visible(False)
    cb = fig.colorbar(im, ax=ax_im, fraction=0.046, pad=0.02)
    cb.set_label("NDVI", color=INK_2)
    cb.outline.set_visible(False)
    ax_im.legend(handles=[mpl.patches.Patch(facecolor=AXIS, edgecolor="none",
                                            label="masked (cloud / snow / no data)")],
                 loc="upper center", bbox_to_anchor=(0.5, -0.02),
                 frameon=False, fontsize=9)
    stamp = ax_im.set_title("", loc="center", color=INK, fontsize=12,
                            fontweight="600", pad=10)

    p = parcel.iloc[0]
    fig.suptitle(f"Parcel {kennung}  |  {str(p.SNAR_BEZEICHNUNG).title()} "
                 f"({p.SNAR_CODE})  |  {p.FLAECHE_BRUTTO/10_000:.2f} ha",
                 x=0.02, ha="left", fontsize=13, fontweight="600", color=INK)

    lookup = dict(zip(ts_date, ts_val))
    rejected = set(pd.to_datetime(ts_date[~keep]))
    # Moran stays on MORAN_INDEX even when the series panel moves off it: NDMI
    # comes from B11, a 20 m band upsampled to the 10 m grid, so its I is
    # inflated by construction and not comparable to the NDVI figure the rest
    # of the project quotes.
    mts = series.get(MORAN_INDEX)
    moran = (dict(zip(mts.date, mts.moran))
             if mts is not None and "moran" in mts else {})

    def draw(i):
        im.set_data(frames[i])
        d = fdates[i]
        cursor.set_xdata([d, d])
        v = lookup.get(d)
        dot.set_data([d], [v]) if v is not None else dot.set_data([], [])
        px = int(np.isfinite(frames[i]).sum())
        masked = int(footprint.sum()) - px
        mi = moran.get(d)
        stamp.set_text(f"{d:%d %b %Y}    {px} valid"
                       + (f" / {masked} masked" if masked else "")
                       + f" of {int(footprint.sum())} px"
                       + (f"    {ts_band} {v:.2f}" if v is not None else "")
                       + ("  (rejected)" if d in rejected else "")
                       + (f"    Moran's I {mi:.2f}"
                          if mi is not None and np.isfinite(mi) else ""))
        return im, cursor, dot, stamp

    ani = manim.FuncAnimation(fig, draw, frames=len(frames), blit=False)
    dest.parent.mkdir(parents=True, exist_ok=True)
    ani.save(dest, writer=manim.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"wrote {dest}  ({len(frames)} frames @ {fps} fps)")
    return dest


# ============================================================ plot
def style(ax, sub=None):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.yaxis.grid(True)
    ax.set_axisbelow(True)
    if sub:
        ax.annotate(sub, xy=(0, 1), xycoords="axes fraction", xytext=(0, 6),
                    textcoords="offset points", fontsize=9, color=MUTED,
                    va="bottom", ha="left")


def plot(series, parcel, isv, year, kennung, dest, min_pixels,
         figsize=None, moran_index: str | None = MORAN_INDEX):
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "system-ui", "DejaVu Sans"],
        "font.size": 10,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8,
        "axes.labelcolor": INK_2,
        "xtick.color": MUTED, "ytick.color": MUTED,
        "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
        "xtick.major.size": 0, "ytick.major.size": 0,
        "grid.color": GRID, "grid.linewidth": 0.8, "grid.linestyle": "-",
        "legend.frameon": False, "legend.fontsize": 9,
        "figure.dpi": 110,
    })

    p = parcel.iloc[0]
    names = [n for n in INDEX_ORDER if n in series]
    show_moran = bool(moran_index) and moran_index in series \
        and series[moran_index].moran.notna().any()
    n_panels = len(names) + (1 if show_moran else 0)

    # EVERY class present gets its own legend row, so size the figure to fit
    # them: 3 columns of legend, ~0.30 in per row, on top of the plot stack.
    # Vertical budget in inches, so nothing is ever clipped or overlapped:
    #   header | plot stack | x-axis labels | legend
    present = list(dict.fromkeys(isv.cls)) if not isv.empty else []
    n_entries = len(present) + (1 if not isv.empty
                                and isv.event_date.notna().any() else 0)
    n_rows = max(1, -(-n_entries // LEGEND_NCOL))            # ceil
    legend_in = 0.30 * n_rows + 0.40 if n_entries else 0.0
    width, stack_in = figsize or (FIGWIDTH, PANEL_IN * n_panels)
    fig_h = stack_in + HEADER_IN + XLABEL_IN + legend_in
    fig, axes = plt.subplots(n_panels, 1, figsize=(width, fig_h), sharex=True)

    # --- ISV inspections as vertical lines --------------------------------
    # Style per class comes from the module-level CLASS_STYLE table, so a class
    # looks identical in every plot and no two rows of one legend collide.
    # EVENTDATUM is drawn as a triangle on the bottom axis rather than a second
    # vertical line: several classes are themselves dotted or dashed, so a
    # dashed "event" line would be indistinguishable from a class line.
    from matplotlib.transforms import blended_transform_factory

    # Several classes can share an inspection date (typical on sub-area
    # parcels). Draw the management actions LAST so they end up on top, and do
    # it deterministically rather than in whatever order the rows arrived.
    draw_order = isv.assign(
        _z=[2 if c in CLASS_COLOR else (1 if c == "Kultur" else 0)
            for c in isv.cls]).sort_values("_z") if not isv.empty else isv

    seen = {}
    for ax in axes:
        xaxis_t = blended_transform_factory(ax.transData, ax.transAxes)
        for _, r in draw_order.iterrows():
            c, ls, lw = class_style(r.cls)
            ax.axvline(r.inspection_date, color=c, lw=lw, ls=ls,
                       alpha=0.9, zorder=1 + r._z * 0.1)
            seen[r.cls] = (c, ls, lw)
            if pd.notna(r.event_date):
                ax.plot([r.event_date], [0], marker="^", ms=8, color=c,
                        transform=xaxis_t, clip_on=False, zorder=5,
                        markeredgecolor=SURFACE, markeredgewidth=0.8)

    # --- the index series --------------------------------------------------
    for ax, name in zip(axes, names):
        df = series[name]
        thin = df[df.n_pixels < min_pixels]
        keep = df[df.n_pixels >= min_pixels]
        ax.plot(keep.date, keep["median"], color=INDEX_COLOR[name], lw=1.8,
                marker="o", ms=4, zorder=3, label=f"{name} (parcel median)")
        if not thin.empty:
            ax.scatter(thin.date, thin["median"], s=18, facecolors="none",
                       edgecolors=MUTED, linewidths=0.9, zorder=2,
                       label=f"< {min_pixels} valid px")
        ax.set_ylabel(name)
        style(ax, sub=f"{name} - {INDEX_DESC[name]}")
        ax.legend(loc="upper right", ncol=2)

    # --- spatial heterogeneity: Moran's I ----------------------------------
    # Significance is encoded by MARKER FILL, not by colour: a hollow marker is
    # a date whose arrangement is indistinguishable from random at p<0.05, and
    # reading it as structure would be over-reading.
    if show_moran:
        ax = axes[len(names)]
        df = series[moran_index].dropna(subset=["moran"])
        sig = df[df.moran_z.abs() >= 1.96]
        ns = df[~(df.moran_z.abs() >= 1.96)]
        lo, hi = float(df.moran.min()), float(df.moran.max())
        # Satellite imagery is smoothly varying, so I sits well above 0 on
        # every date and it is the MOVEMENT that carries information. Only
        # anchor the panel on 0 when the series actually approaches it -
        # otherwise the reference line would eat most of the panel height and
        # flatten the signal it is there to reveal.
        if lo < 0.15:
            ax.axhline(0.0, color=AXIS, lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.plot(df.date, df.moran, color=MORAN_COLOR, lw=1.6, zorder=3,
                label=f"Moran's I of {moran_index} (rook, per date)")
        ax.scatter(sig.date, sig.moran, s=26, color=MORAN_COLOR, zorder=4,
                   label="p < 0.05 (structured)")
        if not ns.empty:
            ax.scatter(ns.date, ns.moran, s=26, facecolors=SURFACE,
                       edgecolors=MORAN_COLOR, linewidths=1.2, zorder=4,
                       label="not significant (random)")
        # Headroom so the legend cannot sit on top of the line.
        span = max(hi - lo, 0.05)
        ax.set_ylim(min(lo - 0.15 * span, 0.0 if lo < 0.15 else lo - 0.15 * span),
                    hi + 0.55 * span)
        ax.set_ylabel("Moran's I")
        style(ax, sub=f"Moran's I of {moran_index} - spatial autocorrelation "
                      "within the parcel; 0 = random arrangement, "
                      "higher = organised patches")
        ax.legend(loc="upper right", ncol=3)

    # --- header: the GSA declaration ---------------------------------------
    area_ha = p.FLAECHE_BRUTTO / 10_000
    nart = {"A": "arable", "G": "grassland", "D": "communal pasture"}.get(
        p.FS_NART_CODE, str(p.FS_NART_CODE))
    head = (f"Parcel {kennung}  |  GSA {year}: {str(p.SNAR_BEZEICHNUNG).title()} "
            f"({p.SNAR_CODE})  |  {nart}  |  {area_ha:.2f} ha")
    sub = f"declared codes: {p.CODES or '-'}"
    if isv.empty:
        sub += "   |   not present in ISV"
    else:
        counts = isv.cls.value_counts()
        sub += ("   |   ISV: "
                + ", ".join(f"{n}x {class_label(c, short=True)}"
                            for c, n in counts.items())
                + f"  ({len(isv)} inspections)")
        shared = int((isv.groupby("inspection_date").cls.nunique() > 1).sum())
        if shared:
            sub += (f"   |   {shared} date(s) carry >1 class - "
                    "overlapping lines, see table")
    fig.suptitle(head, x=0.006, ha="left", fontsize=14, fontweight="600",
                 color=INK, y=1 - 0.22 / fig_h)
    fig.text(0.006, 1 - 0.50 / fig_h, sub, ha="left", fontsize=10, color=INK_2)

    # --- ISV legend ---------------------------------------------------------
    if seen:
        # One row per class actually present - nothing folded into "Other".
        order = ([c for c in CLASS_STYLE if c in seen]
                 + [c for c in seen if c not in CLASS_STYLE])
        handles = [mpl.lines.Line2D([], [], color=seen[k][0], lw=seen[k][2],
                                    ls=seen[k][1], label=class_label(k))
                   for k in order]
        if isv.event_date.notna().any():
            handles.append(mpl.lines.Line2D(
                [], [], color=MUTED, lw=0, marker="^", ms=8,
                markeredgecolor=SURFACE, markeredgewidth=0.8,
                label="EVENTDATUM (when the event happened)"))
        leg = fig.legend(handles=handles, loc="lower center",
                         ncol=LEGEND_NCOL, frameon=False,
                         title="ISV inspection - observed class (BODENBEDECKUNG)",
                         bbox_to_anchor=(0.5, 0.006))
        leg.get_title().set_color(INK)
        leg.get_title().set_fontweight("600")

    axes[-1].set_xlabel("date")
    fig.tight_layout()
    fig.subplots_adjust(top=1 - HEADER_IN / fig_h,
                        bottom=(legend_in + XLABEL_IN) / fig_h)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, bbox_inches="tight")
    print(f"\nwrote {dest}")
    return fig


# ============================================================ display
def in_notebook() -> bool:
    """True when running under a Jupyter kernel (not a plain REPL or script)."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        return ip is not None and ip.__class__.__name__ == "ZMQInteractiveShell"
    except Exception:                                        # noqa: BLE001
        return False


def show_gif(path) -> bool:
    """Render a GIF inline in Jupyter. No-op (returns False) elsewhere.

    matplotlib figures show themselves at the end of a cell, but an animation
    is written straight to disk and its figure is closed, so it has to be
    displayed explicitly or the caller just gets a file path.
    """
    if path is None or not in_notebook():
        return False
    from IPython.display import Image, display
    display(Image(filename=str(path)))
    return True


# ============================================================ one-call API
@dataclass
class ParcelResult:
    """Everything one parcel run produced."""
    kennung: int
    year: int
    parcel: "gpd.GeoDataFrame"
    isv: "pd.DataFrame"
    cube: object
    series: dict
    table: "pd.DataFrame"
    png: Path | None = None
    gif: Path | None = None
    csv: Path | None = None

    def __repr__(self) -> str:                       # keep notebook output tidy
        p = self.parcel.iloc[0]
        return (f"<ParcelResult {self.kennung} {self.year} "
                f"{p.SNAR_CODE} · {len(self.series['NDVI'])} dates · "
                f"{len(self.isv)} inspections>")


def explore(kennung: int, year: int = 2024, *,
            animation: bool = True, csv: bool = False,
            start: str | None = None, end: str | None = None,
            resolution: int = 10, cloud_cover_max: int = 60,
            min_pixels: int = 5, fps: int = 2,
            scl_drop=SCL_DROP_DEFAULT, moran_index: str | None = MORAN_INDEX,
            anim_band: str = ANIM_INDEX, out_dir: Path = OUT,
            verbose: bool = True, show: bool = True) -> ParcelResult:
    """Fetch, plot the five indices and Moran's I, and animate - in one call.

        r = explore(233103922, 2024)
        r.series["NDVI"]        # date, median, n_pixels, moran, moran_z
        r.png, r.gif            # what it wrote

    `moran_index` chooses which index the heterogeneity panel is drawn from
    (None drops the panel); every index carries its own Moran columns either
    way.

    `anim_band` chooses the index the animation's series panel draws, as
    delivered and after `persistence_mask`, so a frame that fails the filter
    can be read against the imagery that produced it. NDMI by default - it is
    the mowing feature. The map underneath is always NDVI.

    Every step is also available separately (load_parcel, load_isv,
    compute_indices, plot, animate) - this just chains them.
    """
    cfg = YEARS[year]
    if not cfg["at"].exists():
        raise SystemExit(f"missing {cfg['at']}")

    parcel = load_parcel(year, kennung, None)
    p = parcel.iloc[0]
    kennung = int(p.KENNUNG)
    isv = load_isv(year, kennung)

    if verbose:
        nart = {"A": "arable", "G": "grassland",
                "D": "communal pasture"}.get(p.FS_NART_CODE, str(p.FS_NART_CODE))
        print(f"Parcel {kennung} ({year})")
        print(f"  GSA   : {p.SNAR_BEZEICHNUNG} ({p.SNAR_CODE}) · {nart} · "
              f"{p.FLAECHE_BRUTTO/10_000:.2f} ha")
        if isv.empty:
            print("  ISV   : not inspected")
        else:
            tally = ", ".join(f"{n}x {class_label(c, short=True)}"
                              for c, n in isv.cls.value_counts().items())
            print(f"  ISV   : {len(isv)} inspections - {tally}")

    get_s2 = load_s2_loader()
    geom = parcel.to_crs("EPSG:4326").geometry.iloc[0]
    s_date = start or cfg["season"][0]
    e_date = end or cfg["season"][1]
    if verbose:
        print(f"  S2    : {s_date} -> {e_date} @{resolution} m ...")

    buf = io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        cube = get_s2(custom_geometry=geom, start_date=s_date, end_date=e_date,
                      bands=["B02", "B04", "B05", "B08", "B11", "SCL"],
                      resolution=resolution, cloud_cover_max=cloud_cover_max,
                      mask_clouds=True).compute()
    for line in buf.getvalue().splitlines():
        if verbose and line.strip() and "Completed" not in line                 and not line.lstrip().startswith("["):
            print(f"          {line.strip()}")

    series = compute_indices(cube, scl_drop=scl_drop)
    if not len(series["NDVI"]):
        raise SystemExit("No usable observations - raise cloud_cover_max.")
    if verbose:
        print(f"  dates : {len(series['NDVI'])} usable · median "
              f"{series['NDVI'].n_pixels.median():.0f} valid px/date")

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"parcel_{kennung}_{year}"
    png = out_dir / f"{stem}_indices.png"
    fig = plot(series, parcel, isv, year, kennung, png, min_pixels,
               moran_index=moran_index)

    # Push the chart out now rather than letting matplotlib flush it at the end
    # of the cell - otherwise the GIF, which is displayed during this call,
    # would appear ABOVE the chart it belongs under.
    if show and in_notebook():
        from IPython.display import display as _display
        _display(fig)
        plt.close(fig)

    gif = None
    if animation:
        gif = animate(cube, series, isv, parcel, year, kennung,
                      out_dir / f"{stem}_animation.gif", fps=fps,
                      scl_drop=scl_drop, ts_band=anim_band)
        if show:
            show_gif(gif)

    table = None
    for name in INDEX_ORDER:
        if name not in series:
            continue
        col = series[name][["date", "median", "n_pixels"]].rename(
            columns={"median": name, "n_pixels": f"{name}_n_px"})
        table = col if table is None else table.merge(col, on="date", how="outer")
    # Moran for the panel's index only - every index carries its own in
    # r.series[name].moran, and 10 more columns here would bury the medians.
    if moran_index in series:
        mor = series[moran_index][["date", "moran", "moran_z"]].rename(
            columns={"moran": f"MoranI_{moran_index}",
                     "moran_z": f"MoranI_{moran_index}_z"})
        table = table.merge(mor, on="date", how="outer")
    table = table.sort_values("date").reset_index(drop=True)

    csv_path = None
    if csv:
        csv_path = out_dir / f"{stem}_indices.csv"
        table.to_csv(csv_path, index=False)
        print(f"wrote {csv_path}")

    return ParcelResult(kennung, year, parcel, isv, cube, series, table,
                        png, gif, csv_path)


# ============================================================ main
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--kennung", type=int, help="parcel KENNUNG (stable ID)")
    sel.add_argument("--sl-id", type=int, help="parcel SL_ID (may change mid-season)")
    sel.add_argument("--isv-only", action="store_true",
                     help="auto-pick a parcel that appears in ISV")
    ap.add_argument("--year", type=int, default=2024, choices=sorted(YEARS))
    ap.add_argument("--label", help="with --isv-only: require this BODENBEDECKUNG")
    ap.add_argument("--min-visits", type=int, default=3,
                    help="with --isv-only: minimum inspections (default 3)")
    ap.add_argument("--start", help="override season start (YYYY-MM-DD)")
    ap.add_argument("--end", help="override season end (YYYY-MM-DD)")
    ap.add_argument("--resolution", type=int, default=10, help="metres (default 10)")
    ap.add_argument("--cloud-cover-max", type=int, default=60,
                    help="max scene cloud percent (default 60)")
    ap.add_argument("--scl-drop", default=",".join(map(str, SCL_DROP_DEFAULT)),
                    help="extra SCL classes to mask, comma-separated "
                         f"(default {','.join(map(str, SCL_DROP_DEFAULT))} = "
                         "no-data, defective, snow/ice). '' disables.")
    ap.add_argument("--min-pixels", type=int, default=5,
                    help="flag dates with fewer valid pixels (default 5)")
    ap.add_argument("--moran-index", default=MORAN_INDEX,
                    choices=[*INDEX_ORDER, "none"],
                    help="index the Moran's I panel is computed on "
                         f"(default {MORAN_INDEX}; 'none' drops the panel). "
                         "NDVI is the only one from natively 10 m bands.")
    ap.add_argument("--csv", action="store_true", help="also write the series to CSV")
    ap.add_argument("--animate", action="store_true",
                    help="also write an animated GIF: NDVI map per date with a "
                         "cursor sweeping the series, drawn before and after "
                         "the persistence filter")
    ap.add_argument("--anim-band", default=ANIM_INDEX, choices=INDEX_ORDER,
                    help="index the animation's series panel draws "
                         f"(default {ANIM_INDEX}, the mowing feature)")
    ap.add_argument("--fps", type=int, default=2, help="animation fps (default 2)")
    ap.add_argument("--out", type=Path, help="output PNG path")
    args = ap.parse_args()

    kennung = args.kennung
    if args.isv_only:
        kennung = pick_parcel(args.year, args.label, args.min_visits)
    elif kennung is None:                       # --sl-id: resolve to KENNUNG
        kennung = int(load_parcel(args.year, None, args.sl_id).iloc[0].KENNUNG)

    scl_drop = tuple(int(x) for x in args.scl_drop.split(",") if x.strip())
    res = explore(kennung, args.year,
                  animation=args.animate, csv=args.csv,
                  start=args.start, end=args.end,
                  resolution=args.resolution,
                  cloud_cover_max=args.cloud_cover_max,
                  min_pixels=args.min_pixels, fps=args.fps,
                  scl_drop=scl_drop, anim_band=args.anim_band,
                  moran_index=(None if args.moran_index == "none"
                               else args.moran_index),
                  out_dir=(args.out.parent if args.out else OUT))

    # full inspection listing is CLI-only detail
    if not res.isv.empty:
        print("\nISV inspections")
        for _, r in res.isv.iterrows():
            ev = f"  event={r.event_date.date()}" if pd.notna(r.event_date) else ""
            sa = "  [sub-area]" if r.subarea == "J" else ""
            print(f"  {r.inspection_date.date()}  {r.cls}{ev}{sa}")

    if args.out and res.png and res.png != args.out:
        res.png.replace(args.out)
        print(f"moved plot to {args.out}")


if __name__ == "__main__":
    main()
