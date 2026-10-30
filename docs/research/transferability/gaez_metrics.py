"""GAEZ v5 climate metrics of the EuroCrops domains with 2021 labels, weighted by cropland, and their maps.

The transfer study correlates the loss of every pair of countries with the difference of each metric between them.
The metrics follow the four GAEZ v5 agro-climatic sub-themes: climate classification (the thermal climate, whose
map and shares `gaez_domains.py` draws), thermal regime, moisture regime and growing period. Each is the 2001-2020
GAEZ average (AgERA5), read on the 5 arc-minute grid and averaged over each domain's cropland with the same weights
and domains as `gaez_domains.py`. The start of the growing period is a day of the year and is averaged on the circle.

Outputs, next to this file: `gaez_metrics.csv` (one row per domain), `gaez_metrics_correlation.csv` (Spearman
correlation between the pairwise differences of every two metrics, over the 171 pairs of domains) and
`figures/gaez_{thermal_regime,moisture_regime,growing_period}.png`.

    poetry run python docs/research/transferability/gaez_metrics.py
"""

from __future__ import annotations

import itertools

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from gaez_domains import BOUNDS, DOMAINS, GAEZ, HERE, MIN_CROPLAND_PCT, NUTS, cropland_share, domain_raster, read_window

# Days with Tmax > 35 C (X35) are left out: about 0 on the cropland of every domain but Spain (2).
# code -> (sub-theme, label, unit, colour map). One hue per meaning: warmth orange, cold and water supply blue, water
# demand and deficit brown, growing period green, its start purple. The start is drawn counted from 1 July, so that
# autumn starts (the Mediterranean) and spring starts (the rest) fall at the two ends of one scale.
METRICS = {
    "TMP": ("Thermal regime", "Mean annual temperature", "°C", "Oranges"),
    "TS2": ("Thermal regime", "Heat sum, days > 5 °C", "°C days", "Oranges"),
    "LT2": ("Thermal regime", "Days with Ta > 5 °C", "days", "Oranges"),
    "LT3": ("Thermal regime", "Days with Ta > 10 °C", "days", "Oranges"),
    "LT4": ("Thermal regime", "Days with Ta > 20 °C", "days", "Oranges"),
    "N00": ("Thermal regime", "Frost days, Tmin < 0 °C", "days", "Blues"),
    "PRC": ("Moisture regime", "Annual precipitation P", "mm", "Blues"),
    "ET0": ("Moisture regime", "Reference potential ET (PET)", "mm", "YlOrBr"),
    "ETA": ("Moisture regime", "Reference actual ET", "mm", "Blues"),
    "RFM": ("Moisture regime", "Modified Fournier index", "mm", "Blues"),
    "NDR": ("Moisture regime", "Rain days, P > 1 mm", "days", "Blues"),
    "RID": ("Moisture regime", "P/PET, annual", "%", "Blues"),
    "RIS": ("Moisture regime", "P/PET, April to September", "%", "Blues"),
    "RIW": ("Moisture regime", "P/PET, October to March", "%", "Blues"),
    "WDE": ("Moisture regime", "Reference ET deficit", "mm", "YlOrBr"),
    "LGD": ("Growing period", "Total growing-period days", "days", "Greens"),
    "LD1": ("Growing period", "Longest growing period", "days", "Greens"),
    "LGB": ("Growing period", "Start of the longest growing period", "day of year", "Purples"),
}
CIRCULAR = {"LGB"}
JULY_1 = 182  # day of year
MONTH_STARTS = {"Jul": 182, "Aug": 213, "Sep": 244, "Oct": 274, "Nov": 305, "Dec": 335, "Jan": 1, "Feb": 32,
                "Mar": 60, "Apr": 91, "May": 121, "Jun": 152}


def layer(code: str):
    arr, transform, nodata = read_window(GAEZ / f"GAEZ-V5.RES01-{code}.HP0120.AGERA5.HIST.tif")
    arr = arr.astype("float64")
    arr[arr == nodata] = np.nan
    return arr, transform


def weighted_mean(values: np.ndarray, weights: np.ndarray, circular: bool) -> float:
    if not circular:
        return float(np.average(values, weights=weights))
    angle = 2 * np.pi * (values - 1) / 365
    mean = np.arctan2(np.average(np.sin(angle), weights=weights), np.average(np.cos(angle), weights=weights))
    return float((mean % (2 * np.pi)) * 365 / (2 * np.pi) + 1)


def domain_values(arrays: dict[str, np.ndarray], ids: np.ndarray, weight: np.ndarray) -> pd.DataFrame:
    rows = {}
    for k, dom in enumerate(DOMAINS, start=1):
        row = {}
        for code, arr in arrays.items():
            m = (ids == k) & (weight > 0) & np.isfinite(arr)
            row[code] = weighted_mean(arr[m], weight[m], code in CIRCULAR)
        rows[dom] = row
    return pd.DataFrame(rows).T


def pair_differences(values: pd.DataFrame) -> pd.DataFrame:
    """|difference| of every metric over the unordered pairs of domains; the start date on the circle."""
    rows = {}
    for a, b in itertools.combinations(values.index, 2):
        d = (values.loc[a] - values.loc[b]).abs()
        for code in CIRCULAR:
            d[code] = min(d[code], 365 - d[code])
        rows[f"{a}-{b}"] = d
    return pd.DataFrame(rows).T


def panel(arrays, transform, weight, outlines, codes, ncols, title, path):
    h, w = weight.shape
    left, top = transform.c, transform.f
    extent = (left, left + w * transform.a, top + h * transform.e, top)
    nrows = int(np.ceil(len(codes) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 4.6 * nrows), squeeze=False)
    nuts0 = gpd.read_file(NUTS)
    nuts0 = nuts0[nuts0.LEVL_CODE == 0]
    crop = weight >= MIN_CROPLAND_PCT
    for ax, code in zip(axes.flat, codes):
        _, label, unit, cmap = METRICS[code]
        img = np.where(crop, arrays[code], np.nan)
        if code in CIRCULAR:
            img = (img - JULY_1) % 365
        lo, hi = np.nanpercentile(img, [2, 98])
        ax.set_facecolor("#cfcfcf")
        im = ax.imshow(img, extent=extent, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
        nuts0.boundary.plot(ax=ax, color="#7a7a7a", linewidth=0.25)
        outlines.boundary.plot(ax=ax, color="black", linewidth=0.5)
        ax.set_xlim(BOUNDS[0], BOUNDS[2])
        ax.set_ylim(BOUNDS[1], BOUNDS[3])
        ax.set_aspect(1 / np.cos(np.deg2rad(52)))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{label} ({code})", fontsize=9)
        cb = fig.colorbar(im, ax=ax, orientation="horizontal", fraction=0.045, pad=0.02, extend="both")
        if code in CIRCULAR:
            ticks = {m: (d - JULY_1) % 365 for m, d in MONTH_STARTS.items()}
            ticks = {m: t for m, t in ticks.items() if lo <= t <= hi}
            cb.set_ticks(list(ticks.values()), labels=list(ticks))
            unit = "month the growing period starts"
        cb.set_label(unit, fontsize=8)
        cb.ax.tick_params(labelsize=7)
    for ax in axes.flat[len(codes):]:
        ax.axis("off")
    fig.suptitle(f"{title}: GAEZ v5, 2001-2020, on cropland (grey: cropland below {MIN_CROPLAND_PCT:g} % of the cell)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> None:
    arrays, transform = {}, None
    for code in METRICS:
        arrays[code], transform = layer(code)
    weight = cropland_share(next(iter(arrays.values())).shape)
    weight[weight < MIN_CROPLAND_PCT] = 0.0
    ids, outlines = domain_raster(transform, weight.shape)

    values = domain_values(arrays, ids, weight)
    values.round(1).to_csv(HERE / "gaez_metrics.csv")
    diffs = pair_differences(values)
    diffs.corr(method="spearman").round(2).to_csv(HERE / "gaez_metrics_correlation.csv")

    figs = HERE / "figures"
    for theme, ncols in (("Thermal regime", 3), ("Moisture regime", 3), ("Growing period", 3)):
        codes = [c for c, m in METRICS.items() if m[0] == theme]
        panel(arrays, transform, weight, outlines, codes, ncols, theme,
              figs / f"gaez_{theme.lower().replace(' ', '_')}.png")

    pd.set_option("display.width", 250)
    print(values.round(0).astype(int).to_string())


if __name__ == "__main__":
    main()
