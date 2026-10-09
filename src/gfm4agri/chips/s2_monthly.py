"""Sentinel-2 L2A monthly composites on the fixed T = 12 grid, for one chip footprint.

Imagery is read from the Microsoft Planetary Computer ``sentinel-2-l2a`` collection and
warped straight onto the chip grid (EPSG:3035, 10 m, 224 x 224). Per month, the composite is
the per-pixel median over the acquisitions whose scene classification marks that pixel as
clear. Three corrections are applied, and each is recorded in the chip manifest:

* **BOA offset.** Acquisitions at processing baseline 04.00 or later are shifted by
  ``BOA_ADD_OFFSET`` so every date sits on ``reflectance = DN / 10000``, the same convention
  :func:`gfm4agri.data.sentinel.harmonise_boa` enforces. The Planetary Computer also serves
  2021 scenes reprocessed at baseline 04.00 beside the originals, so for the 2021 season the
  correction does apply.
* **Cloud and snow screening** as a :class:`Screening`. ``EXPORT_2021``, the default, is
  ``SCL_DROP`` alone, as in :mod:`gfm4agri.data.sentinel`, and is what the Estonian chip set
  was exported with. ``REVISED`` closes the leaks found in its QA; see :class:`Screening`.
* **Unreadable acquisitions.** An asset the archive cannot serve is dropped and its item id
  is recorded under ``unreadable_items``, rather than failing the chip. A handful of 2021
  scenes are permanently broken on the Planetary Computer, and one of them would otherwise
  cost every chip whose footprint it touches.
* **Gap filling.** A pixel with no clear acquisition in a month is filled by linear
  interpolation along the month axis from its nearest clear months, and by the nearest clear
  month at the ends of the season. The share of filled pixels per month is returned, because
  a Baltic December is mostly filled rather than observed and a reader must be able to see
  that.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass

import numpy as np

from gfm4agri.data.sentinel import BOA_ADD_OFFSET, OFFSET_BASELINE, SCL_DROP

__all__ = ["BANDS", "BAND_NAMES", "EXPORT_2021", "N_MONTHS", "REVISED", "Screening",
           "composite_window", "monthly_composite", "window_report"]

#: The twelve L2A spectral bands, B10 being absent from L2A, in wavelength order.
BANDS = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12"]
#: The same bands under TerraTorch's band names, which is how a config selects a subset.
BAND_NAMES = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2",
              "RED_EDGE_3", "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2"]
N_MONTHS = 12
#: Largest share of a window's acquisitions that may be dropped as unreadable before the
#: window is failed rather than composited; one dropped acquisition is always tolerated.
MAX_UNREADABLE_SHARE = 0.05
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


@dataclass(frozen=True)
class Screening:
    """What the compositor accepts as a clear observation of a pixel.

    The default reproduces the export of the Estonian chip set: scene classification classes
    in ``drop`` are excluded and every item is its own observation. Its QA found four leaks,
    which the other fields close:

    * SCL class 2 (dark area) rings every snow patch inside a cloud, and class 7
      (unclassified) fringes cloud edges; both pass ``SCL_DROP`` and show as white rims.
    * Cloud, shadow and snow edges a pixel or two beyond the classified object pass as clear;
      ``dilate_px`` grows classes ``dilate_classes`` by that many pixels.
    * Snow and haze the classification calls vegetation or bare soil; ``snow_test`` and
      ``haze_test`` apply the Fmask snow test and haze-optimised transform (Zhu and Woodcock
      2012) to the reflectance of each acquisition.
    * One overpass is catalogued once per MGRS tile, and again where it was reprocessed, so a
      pixel in a tile overlap entered the median up to four times. ``one_per_overpass`` merges
      the copies of an overpass into one observation, flagged if any copy flags it, because
      Sen2Cor classifies every tile on its own and a copy can call clear what another calls
      cloud.
    """

    drop: tuple[int, ...] = SCL_DROP
    dilate_px: int = 0
    dilate_classes: tuple[int, ...] = (3, 8, 9, 10, 11)
    snow_test: bool = False
    haze_test: bool = False
    one_per_overpass: bool = False


#: The screening the Estonian 2021 chip set was exported with.
EXPORT_2021 = Screening()
#: Only vegetation, bare soil and water are clear; cloud, shadow and snow grown by 20 m.
REVISED = Screening(drop=(0, 1, 2, 3, 7, 8, 9, 10, 11), dilate_px=2, snow_test=True,
                    haze_test=True, one_per_overpass=True)


def _screen(s: np.ndarray, c: np.ndarray, screening: Screening) -> tuple[np.ndarray, np.ndarray]:
    """Per-acquisition ``(nodata, flagged)`` masks, each ``(n, H, W)``.

    ``s`` is ``(n, 12, H, W)`` reflectance times 10000 after the BOA offset, ``c`` the scene
    classification ``(n, H, W)``. A pixel is nodata where the acquisition does not cover it,
    and flagged where it covers it but is not clear.
    """
    nodata = np.isnan(c) | np.isnan(s).any(1) | (c == 0)
    flagged = np.isin(c, screening.drop)
    if screening.dilate_px:
        from scipy.ndimage import binary_dilation

        grow = np.isin(c, screening.dilate_classes)
        flagged |= np.stack([binary_dilation(g, iterations=screening.dilate_px) for g in grow])
    r = {b: s[:, BANDS.index(b)] / 10000.0 for b in ("B02", "B03", "B04", "B08", "B11")}
    with np.errstate(invalid="ignore", divide="ignore"):
        if screening.snow_test:
            ndsi = (r["B03"] - r["B11"]) / (r["B03"] + r["B11"])
            flagged |= (ndsi > 0.15) & (r["B08"] > 0.11) & (r["B03"] > 0.1)
        if screening.haze_test:
            flagged |= r["B02"] - 0.5 * r["B04"] - 0.08 > 0
    return nodata, flagged & ~nodata


def _overpass_key(item) -> tuple[str, str]:
    """Platform and sensing time: equal for every tile and reprocessing of one overpass."""
    return item.properties.get("platform", ""), item.datetime.isoformat()


def _merge_overpasses(s: np.ndarray, nodata: np.ndarray, flagged: np.ndarray,
                      keys: list) -> tuple[np.ndarray, np.ndarray]:
    """One observation per overpass: ``(g, 12, H, W)`` values and ``(g, H, W)`` bad mask.

    A pixel is bad when no copy covers it or when any copy covering it is flagged; otherwise
    its value is the mean of the copies, which differ only by the per-tile processing.
    """
    groups = [np.flatnonzero([k == key for k in keys]) for key in dict.fromkeys(keys)]
    vals, bads = [], []
    for g in groups:
        data = ~nodata[g]
        bad = (flagged[g] & data).any(0) | ~data.any(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            v = np.nanmean(np.where(data[:, None], s[g], np.nan), axis=0)
        vals.append(v)
        bads.append(bad)
    return np.stack(vals), np.stack(bads)


def _search(bbox_4326, year: int, cloud_cover_max: float):
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    return cat.search(collections=["sentinel-2-l2a"], bbox=list(bbox_4326),
                      datetime=f"{year}-01-01/{year}-12-31",
                      query={"eo:cloud_cover": {"lt": cloud_cover_max}}).item_collection()


def _align_on_ids(spec, scl):
    """Both stacks restricted to the acquisitions present in both, in the same order.

    Every per-acquisition quantity, the month, the processing baseline and the scene
    classification, is attached by item id rather than by position. ``stackstac.stack``
    reorders acquisitions by date unless told not to, and silently leaves out any
    acquisition it finds empty over the window, so an acquisition's position in the search
    results says nothing reliable about its position in a stack. Indexing by position once
    reversed every season; a single left-out acquisition would shift every later label by
    one.

    Returns the two aligned stacks and the item ids along their time axis.
    """
    sid = [str(i) for i in np.atleast_1d(spec["id"].values)]
    cid = [str(i) for i in np.atleast_1d(scl["id"].values)]
    if len(set(sid)) != len(sid) or len(set(cid)) != len(cid):
        raise RuntimeError("a stack holds the same acquisition twice")
    pos_c = {i: k for k, i in enumerate(cid)}
    pairs = [(k, pos_c[i]) for k, i in enumerate(sid) if i in pos_c]
    spec = spec.isel(time=[a for a, _ in pairs])
    scl = scl.isel(time=[b for _, b in pairs])
    ids = [sid[a] for a, _ in pairs]
    if [str(i) for i in np.atleast_1d(scl["id"].values)] != ids:
        raise RuntimeError("spectral and scene-classification stacks could not be aligned")
    return spec, scl, ids


def _fill_months(stack: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill NaN months per pixel along axis 0; return the stack and the filled mask."""
    t = np.arange(stack.shape[0], dtype=np.float32)
    missing = np.isnan(stack)
    flat = stack.reshape(stack.shape[0], -1)
    todo = np.where(np.isnan(flat).any(0) & ~np.isnan(flat).all(0))[0]
    for j in todo:
        v = flat[:, j]
        ok = ~np.isnan(v)
        flat[:, j] = np.interp(t, t[ok], v[ok])
    return flat.reshape(stack.shape), missing


def composite_window(bounds_3035, year: int = 2021, *, cloud_cover_max: float = 80.0,
                     screening: Screening = EXPORT_2021,
                     verbose: bool = False) -> tuple[np.ndarray, dict, dict]:
    """Monthly composite over any window on the chip grid, with per-pixel diagnostics.

    ``bounds_3035`` must lie on the 10 m grid. A group of adjacent chips is composited in one
    read, so each 512 x 512 block of each COG is fetched once for the group rather than once
    per chip; for a densely chipped scene the chip-by-chip read fetches about sixteen times
    the data the chips need. Every operation after the read is per pixel, namely the
    screening, the offset, the median and the gap filling, and the output grid is the chip
    grid, so a chip cut from a window is the chip composited alone.

    Returns the ``(144, H, W)`` int16 array, the per-pixel diagnostic maps that
    :func:`window_report` reduces to a chip's report, and the report of what was read for
    the window as a whole.

    Growing the masks is the one spatial operation, so a screening with ``dilate_px`` reads
    the window with that margin and crops it afterwards, and a chip cut from a window is
    still the chip composited alone.
    """
    import stackstac
    from pyproj import Transformer

    pad = screening.dilate_px
    xmin, ymin, xmax, ymax = (bounds_3035[0] - pad * 10, bounds_3035[1] - pad * 10,
                              bounds_3035[2] + pad * 10, bounds_3035[3] + pad * 10)
    read_bounds = (xmin, ymin, xmax, ymax)
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    bbox = (min(lon), min(lat), max(lon), max(lat))

    items = _search(bbox, year, cloud_cover_max)
    if not items:
        raise RuntimeError(f"no Sentinel-2 L2A items for {bbox} in {year}")
    kw = dict(epsg=3035, resolution=10, bounds=read_bounds, dtype="float32",
              fill_value=np.float32(np.nan), rescale=False, chunksize=2048)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # sortby_date=False keeps the search order, which is the order the months and the
        # processing baselines are read in below. stackstac sorts by date by default and the
        # search returns the newest acquisitions first, so without it every month label
        # lands on the wrong acquisitions and the season comes out approximately reversed.
        spec = stackstac.stack(items, assets=BANDS, resampling=_resampling("bilinear"),
                               sortby_date=False, **kw)
        scl = stackstac.stack(items, assets=["SCL"], resampling=_resampling("nearest"),
                              sortby_date=False, **kw)
    spec, scl, stacked = _align_on_ids(spec, scl)
    by_id = {it.id: it for it in items}
    stacked_items = [by_id[i] for i in stacked]
    baseline = np.array([_baseline(it) for it in stacked_items])
    months = np.array([it.datetime.month for it in stacked_items])
    h, w = spec.shape[-2:]
    expected = (round((ymax - ymin) / 10), round((xmax - xmin) / 10))
    assert (h, w) == expected, ((h, w), expected)

    out = np.full((N_MONTHS, len(BANDS), h, w), np.nan, dtype=np.float32)
    clear = np.zeros((N_MONTHS, h, w), dtype=np.int16)
    report = {"n_items": len(items), "items_per_month": [],
              "baselines": sorted({f"{b:.2f}" for b in baseline}),
              "dates_offset_corrected": int((baseline >= OFFSET_BASELINE).sum())}
    if screening != EXPORT_2021:
        # Recorded only when it departs from the export, so the reports of the existing chip
        # set and of a default export stay identical.
        report["screening"] = asdict(screening)
        report["overpasses_per_month"] = [0] * N_MONTHS
    not_stacked = sorted(set(by_id) - set(stacked))
    if not_stacked:
        # stackstac leaves out an acquisition it finds empty over the window; recorded so
        # the difference between n_items and the acquisitions composited is explained.
        report["not_stacked_items"] = not_stacked
    for m in range(1, N_MONTHS + 1):
        idx = np.where(months == m)[0]
        report["items_per_month"].append(int(len(idx)))
        if not len(idx):
            continue
        try:
            s = spec.isel(time=idx).values
            c = scl.isel(time=idx).values[:, 0]
            kept = idx
        except Exception:
            # A single asset that the archive cannot serve must not cost the whole window.
            # Fall back to reading the month one acquisition at a time and drop what fails;
            # the dropped items are recorded so the composite stays auditable.
            keep_s, keep_c, kept, dropped = [], [], [], []
            for j in idx:
                # Both reads must succeed before either is kept, or the spectral and
                # scene-classification arrays fall out of step with each other.
                try:
                    sj = spec.isel(time=[j]).values[0]
                    cj = scl.isel(time=[j]).values[0, 0]
                except Exception:
                    dropped.append(stacked[j])
                    continue
                keep_s.append(sj)
                keep_c.append(cj)
                kept.append(j)
            report.setdefault("unreadable_items", []).extend(dropped)
            if not keep_s:
                continue
            s, c, kept = np.stack(keep_s), np.stack(keep_c), np.array(kept)
        # The offset follows the acquisitions actually read, which after a dropped asset is
        # fewer than the month's items.
        shift = np.where(baseline[kept] >= OFFSET_BASELINE, BOA_ADD_OFFSET, 0.0)
        s = s + shift[:, None, None, None].astype(np.float32)
        nodata, flagged = _screen(s, c, screening)
        if screening.one_per_overpass:
            keys = [_overpass_key(stacked_items[j]) for j in kept]
            s, bad = _merge_overpasses(s, nodata, flagged, keys)
        else:
            bad = nodata | flagged
        if "overpasses_per_month" in report:
            report["overpasses_per_month"][m - 1] = int(len(s))
        s[np.broadcast_to(bad[:, None], s.shape)] = np.nan
        clear[m - 1] = (~bad).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[m - 1] = np.nanmedian(s, axis=0)
        if verbose:
            print(f"    month {m:02d}: {len(idx):3d} items, "
                  f"{clear[m - 1].mean():.1f} clear obs per pixel")

    n_dropped = len(report.get("unreadable_items", []))
    if n_dropped > max(1, MAX_UNREADABLE_SHARE * len(items)):
        # Dropping acquisitions is only honest when it is an isolated asset failure. A burst
        # of failures, or an expired token, would otherwise be gap-filled into a composite
        # that looks complete, so the window fails loudly and is retried instead.
        raise RuntimeError(f"systematic read failure: {n_dropped} of {len(items)} acquisitions "
                           f"unreadable; refusing to gap-fill them")

    if pad:
        out, clear = out[..., pad:-pad, pad:-pad], clear[..., pad:-pad, pad:-pad]
    filled, missing = _fill_months(np.ascontiguousarray(out))
    maps = {"clear_obs": clear, "filled": missing.any(1),
            "never_observed": np.isnan(filled).any((0, 1))}
    filled = np.nan_to_num(filled, nan=0.0)
    arr = np.clip(np.round(filled), 0, 32767).astype(np.int16)
    return arr.reshape(N_MONTHS * len(BANDS), *arr.shape[-2:]), maps, report


def window_report(maps: dict, report: dict, rows: slice = slice(None),
                  cols: slice = slice(None)) -> dict:
    """The provenance report of one chip cut from a composited window.

    The pixel diagnostics are reduced over the chip's own pixels. The item counts describe
    what was read for the window, which for a group is a superset of the items over any one
    chip; an item that does not cover a chip contributes nothing to its pixels.
    """
    clear = maps["clear_obs"][:, rows, cols].reshape(N_MONTHS, -1)
    filled = maps["filled"][:, rows, cols].reshape(N_MONTHS, -1)
    out = dict(report)
    out["clear_obs_mean_per_month"] = [float(x) for x in clear.mean(1)]
    out["filled_share_per_month"] = [round(float(x), 4) for x in filled.mean(1)]
    out["never_observed_pixels"] = int(maps["never_observed"][rows, cols].sum())
    return out


def monthly_composite(bounds_3035, year: int = 2021, *, cloud_cover_max: float = 80.0,
                      screening: Screening = EXPORT_2021,
                      verbose: bool = False) -> tuple[np.ndarray, dict]:
    """``(12 * 12, 224, 224)`` int16 composite of one chip, time-major, and its report.

    Band order is ``month 1 band 1 .. month 1 band 12, month 2 band 1, ...``, which is the
    ``(time channels)`` layout TerraTorch's ``expand_temporal_dimension`` expects.
    """
    arr, maps, report = composite_window(bounds_3035, year, cloud_cover_max=cloud_cover_max,
                                         screening=screening, verbose=verbose)
    assert arr.shape[-2:] == (224, 224), arr.shape
    return arr, window_report(maps, report)


def _resampling(name: str):
    from rasterio.enums import Resampling

    return getattr(Resampling, name)


def _baseline(item) -> float:
    try:
        return float(str(item.properties.get("s2:processing_baseline", "0")))
    except ValueError:
        return 0.0
