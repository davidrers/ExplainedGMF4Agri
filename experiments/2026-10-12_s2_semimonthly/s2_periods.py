"""Sentinel-2 L2A composites on a grid of half-month periods, T = 24.

A copy of :func:`gfm4agri.chips.s2_monthly.composite_window` that differs in one place: an acquisition is
assigned to the half month it falls in, days 1 to 15 or day 16 to the month's end, rather than to its calendar
month. The read, the alignment on item ids, the unreadable-asset fallback, the BOA offset, the screening
(``REVISED`` by default), the merge of tile copies, the per-pixel median and the gap filling are the core's own,
imported or copied unchanged, so a 24-period chip differs from its 12-month counterpart only in the periods.
"""

from __future__ import annotations

import calendar
import warnings

import numpy as np

from gfm4agri.chips.s2_monthly import (BAND_NAMES, BANDS, MAX_UNREADABLE_SHARE, REVISED, Screening,
                                       _align_on_ids, _baseline, _fill_months, _merge_overpasses,
                                       _overpass_key, _resampling, _screen, _search)
from gfm4agri.data.sentinel import BOA_ADD_OFFSET, OFFSET_BASELINE

N_PERIODS = 24
#: ``Jan1`` is 1 to 15 January, ``Jan2`` 16 to 31 January, and so on.
PERIOD_NAMES = [f"{calendar.month_abbr[m]}{h}" for m in range(1, 13) for h in (1, 2)]


def period_of(when) -> int:
    """Zero-based half-month index of a datetime: 2 per month, the second from day 16."""
    return (when.month - 1) * 2 + (when.day > 15)


def period_centres(year: int) -> list[tuple[int, int]]:
    """(year, zero-based day of year) of each period's middle day, the 8th and the 23rd of every month,
    for encoders that take the date of each frame."""
    import datetime as dt

    return [(year, dt.date(year, m, d).timetuple().tm_yday - 1) for m in range(1, 13) for d in (8, 23)]


def composite_window(bounds_3035, year: int = 2021, *, cloud_cover_max: float = 80.0,
                     screening: Screening = REVISED) -> tuple[np.ndarray, dict, dict]:
    """Half-month composite over a window on the chip grid: ``(24 * 12, H, W)`` int16, maps, report."""
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
        spec = stackstac.stack(items, assets=BANDS, resampling=_resampling("bilinear"),
                               sortby_date=False, **kw)
        scl = stackstac.stack(items, assets=["SCL"], resampling=_resampling("nearest"),
                              sortby_date=False, **kw)
    spec, scl, stacked = _align_on_ids(spec, scl)
    by_id = {it.id: it for it in items}
    stacked_items = [by_id[i] for i in stacked]
    baseline = np.array([_baseline(it) for it in stacked_items])
    periods = np.array([period_of(it.datetime) for it in stacked_items])
    h, w = spec.shape[-2:]
    assert (h, w) == (round((ymax - ymin) / 10), round((xmax - xmin) / 10))

    out = np.full((N_PERIODS, len(BANDS), h, w), np.nan, dtype=np.float32)
    clear = np.zeros((N_PERIODS, h, w), dtype=np.int16)
    report = {"n_items": len(items), "items_per_period": [], "overpasses_per_period": [0] * N_PERIODS,
              "baselines": sorted({f"{b:.2f}" for b in baseline}),
              "dates_offset_corrected": int((baseline >= OFFSET_BASELINE).sum())}
    from dataclasses import asdict

    report["screening"] = asdict(screening)
    not_stacked = sorted(set(by_id) - set(stacked))
    if not_stacked:
        report["not_stacked_items"] = not_stacked
    for p in range(N_PERIODS):
        idx = np.where(periods == p)[0]
        report["items_per_period"].append(int(len(idx)))
        if not len(idx):
            continue
        try:
            s = spec.isel(time=idx).values
            c = scl.isel(time=idx).values[:, 0]
            kept = idx
        except Exception:
            keep_s, keep_c, kept, dropped = [], [], [], []
            for j in idx:
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
        shift = np.where(baseline[kept] >= OFFSET_BASELINE, BOA_ADD_OFFSET, 0.0)
        s = s + shift[:, None, None, None].astype(np.float32)
        nodata, flagged = _screen(s, c, screening)
        if screening.one_per_overpass:
            keys = [_overpass_key(stacked_items[j]) for j in kept]
            s, bad = _merge_overpasses(s, nodata, flagged, keys)
        else:
            bad = nodata | flagged
        report["overpasses_per_period"][p] = int(len(s))
        s[np.broadcast_to(bad[:, None], s.shape)] = np.nan
        clear[p] = (~bad).sum(0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[p] = np.nanmedian(s, axis=0)

    n_dropped = len(report.get("unreadable_items", []))
    if n_dropped > max(1, MAX_UNREADABLE_SHARE * len(items)):
        raise RuntimeError(f"systematic read failure: {n_dropped} of {len(items)} acquisitions "
                           f"unreadable; refusing to gap-fill them")
    if pad:
        out, clear = out[..., pad:-pad, pad:-pad], clear[..., pad:-pad, pad:-pad]
    filled, missing = _fill_months(np.ascontiguousarray(out))   # interpolates along axis 0, any length
    maps = {"clear_obs": clear, "filled": missing.any(1), "never_observed": np.isnan(filled).any((0, 1))}
    arr = np.clip(np.round(np.nan_to_num(filled, nan=0.0)), 0, 32767).astype(np.int16)
    return arr.reshape(N_PERIODS * len(BANDS), *arr.shape[-2:]), maps, report


def window_report(maps: dict, report: dict, rows: slice = slice(None), cols: slice = slice(None)) -> dict:
    clear = maps["clear_obs"][:, rows, cols].reshape(N_PERIODS, -1)
    filled = maps["filled"][:, rows, cols].reshape(N_PERIODS, -1)
    out = dict(report)
    out["clear_obs_mean_per_period"] = [float(x) for x in clear.mean(1)]
    out["filled_share_per_period"] = [round(float(x), 4) for x in filled.mean(1)]
    out["never_observed_pixels"] = int(maps["never_observed"][rows, cols].sum())
    return out


DESCRIPTIONS = [f"P{p + 1:02d}_{b}" for p in range(N_PERIODS) for b in BAND_NAMES]
