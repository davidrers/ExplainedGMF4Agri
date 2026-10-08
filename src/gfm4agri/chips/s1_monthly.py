"""Sentinel-1 RTC monthly composites on the fixed T = 12 grid, for one chip window.

The target format is TerraMind's ``S1RTC`` modality, the only Sentinel-1 input among the
models of this thesis: two bands, ``VV`` then ``VH``, as backscatter in decibels. Its
pretraining statistics put VV at a mean of -10.9 dB and VH at -17.3 dB.

The source is the Planetary Computer ``sentinel-1-rtc`` collection: radiometrically
terrain-corrected gamma nought, stored as linear power in float32 on the scene's UTM grid at
10 m, with ``-32768`` as nodata. Converting it with ``10 * log10`` puts it on TerraMind's
scale; over Estonian farmland the VV median comes to about -10 dB.

Per month the composite is the per-pixel median over every valid acquisition, taken in
**linear power** and only then converted to decibels, because linear power is the domain in
which backscatter aggregates physically, and the median also suppresses speckle. Both orbit
directions and every relative orbit are composited together. That is defensible because the
product is terrain-corrected, which removes most of the dependence on look geometry, and
Estonia is flat; the orbit mix of every month is recorded so a reader can see it.

Months are read from the stack's own time axis and acquisitions are identified by their item
id, never by their position in the search results: ``stackstac`` reorders acquisitions and
silently leaves out any it finds empty over the window, and positional indexing once
reversed every season of the Sentinel-2 composites.
"""

from __future__ import annotations

import warnings

import numpy as np

from gfm4agri.chips.s2_monthly import MAX_UNREADABLE_SHARE, N_MONTHS, _fill_months, _resampling

__all__ = ["COLLECTION", "S1_ASSETS", "S1_BAND_NAMES", "composite_window_s1",
           "s1_window_report"]

COLLECTION = "sentinel-1-rtc"
#: Asset keys on the Planetary Computer, in TerraMind's band order.
S1_ASSETS = ["vv", "vh"]
#: TerraMind's ``S1RTC`` band names.
S1_BAND_NAMES = ["VV", "VH"]
#: Linear power below this is treated as the floor, -50 dB, rather than taken to minus infinity.
LINEAR_FLOOR = 1e-5
NODATA = -32768.0
STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def _search(bbox_4326, year: int):
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    return cat.search(collections=[COLLECTION], bbox=list(bbox_4326),
                      datetime=f"{year}-01-01/{year}-12-31").item_collection()


def composite_window_s1(bounds_3035, year: int = 2021, *,
                        verbose: bool = False) -> tuple[np.ndarray, dict, dict]:
    """``(12 * 2, H, W)`` float32 composite in dB, the per-pixel maps, and the window report.

    Band order is ``month 1 VV, month 1 VH, month 2 VV, ...``, the ``(time channels)`` layout
    of the Sentinel-2 chips.
    """
    import pandas as pd
    import stackstac
    from pyproj import Transformer

    xmin, ymin, xmax, ymax = bounds_3035
    tr = Transformer.from_crs("EPSG:3035", "EPSG:4326", always_xy=True)
    lon, lat = tr.transform([xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
    items = _search((min(lon), min(lat), max(lon), max(lat)), year)
    if not items:
        raise RuntimeError(f"no Sentinel-1 RTC items for {bounds_3035} in {year}")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        stack = stackstac.stack(items, assets=S1_ASSETS, resampling=_resampling("bilinear"),
                                epsg=3035, resolution=10, bounds=bounds_3035, dtype="float32",
                                fill_value=np.float32(np.nan), rescale=False, chunksize=2048,
                                sortby_date=False)
    h, w = stack.shape[-2:]
    expected = (round((ymax - ymin) / 10), round((xmax - xmin) / 10))
    assert (h, w) == expected, ((h, w), expected)

    # Everything per acquisition comes from the stack itself: its time axis for the month,
    # its id coordinate for the item, so no position in the search results is ever used.
    ids = [str(i) for i in np.atleast_1d(stack["id"].values)]
    if len(set(ids)) != len(ids):
        raise RuntimeError("the stack holds the same acquisition twice")
    months = pd.DatetimeIndex(np.atleast_1d(stack.time.values)).month.values
    by_id = {it.id: it for it in items}
    props = [by_id[i].properties for i in ids]

    out = np.full((N_MONTHS, len(S1_ASSETS), h, w), np.nan, dtype=np.float32)
    valid = np.zeros((N_MONTHS, h, w), dtype=np.int16)
    report = {"n_items": len(items), "items_per_month": [], "orbits_per_month": [],
              "relative_orbits": sorted({int(p.get("sat:relative_orbit", -1)) for p in props}),
              "platforms": sorted({str(p.get("platform", "")) for p in props})}
    not_stacked = sorted(set(by_id) - set(ids))
    if not_stacked:
        report["not_stacked_items"] = not_stacked

    for m in range(1, N_MONTHS + 1):
        idx = np.where(months == m)[0]
        report["items_per_month"].append(int(len(idx)))
        states = [str(props[j].get("sat:orbit_state", "")) for j in idx]
        report["orbits_per_month"].append({"ascending": states.count("ascending"),
                                           "descending": states.count("descending")})
        if not len(idx):
            continue
        try:
            s = stack.isel(time=idx).values
        except Exception:
            # An isolated asset the archive cannot serve is dropped and recorded; the guard
            # below fails the window if the failures are more than isolated.
            keep, dropped = [], []
            for j in idx:
                try:
                    keep.append(stack.isel(time=[j]).values[0])
                except Exception:
                    dropped.append(ids[j])
            report.setdefault("unreadable_items", []).extend(dropped)
            if not keep:
                continue
            s = np.stack(keep)
        # Nodata, non-finite and non-positive power are all missing; an acquisition counts at
        # a pixel only when both polarisations are valid there.
        bad = ~np.isfinite(s) | (s <= 0) | (s == NODATA)
        bad = bad.any(1)
        s[np.broadcast_to(bad[:, None], s.shape)] = np.nan
        valid[m - 1] = (~bad).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[m - 1] = np.nanmedian(s, axis=0)
        if verbose:
            print(f"    month {m:02d}: {len(idx):3d} items, {valid[m - 1].mean():.1f} valid obs")

    n_dropped = len(report.get("unreadable_items", []))
    if n_dropped > max(1, MAX_UNREADABLE_SHARE * len(items)):
        raise RuntimeError(f"systematic read failure: {n_dropped} of {len(items)} acquisitions "
                           f"unreadable; refusing to gap-fill them")

    filled, missing = _fill_months(out)
    maps = {"valid_obs": valid, "filled": missing.any(1),
            "never_observed": np.isnan(filled).any((0, 1))}
    # A pixel never observed stays NaN rather than taking the floor, which would read as
    # genuinely dark water; the raster is float32, so the gap can be carried honestly.
    with np.errstate(invalid="ignore"):
        db = 10.0 * np.log10(np.maximum(filled, LINEAR_FLOOR))
    return db.astype(np.float32).reshape(N_MONTHS * len(S1_ASSETS), h, w), maps, report


def s1_window_report(maps: dict, report: dict, rows: slice = slice(None),
                     cols: slice = slice(None)) -> dict:
    """The provenance report of one chip cut from a composited window."""
    obs = maps["valid_obs"][:, rows, cols].reshape(N_MONTHS, -1)
    filled = maps["filled"][:, rows, cols].reshape(N_MONTHS, -1)
    out = dict(report)
    out["valid_obs_mean_per_month"] = [round(float(x), 3) for x in obs.mean(1)]
    out["filled_share_per_month"] = [round(float(x), 4) for x in filled.mean(1)]
    out["never_observed_pixels"] = int(maps["never_observed"][rows, cols].sum())
    return out
