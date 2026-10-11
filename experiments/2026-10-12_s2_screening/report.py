"""Figures and tables of the screening study, from the tables ``analyse.py`` wrote.

Writes ``figures/*.png`` and ``figures/tables.md`` in this folder.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
RES = REPO / "results" / HERE.name
TAB = RES / "tables"
FIG = HERE / "figures"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
GROWING = [4, 5, 6, 7, 8, 9]
VARIANTS = ["export", "export_merged", "revised", "revised_no_haze", "revised_keep7", "revised_27_near",
            "revised_keep2"]
RULES = ["class_2", "class_7", "buffer", "snow", "haze", "merge"]
RULE_LABEL = {"class_2": "class 2, dark area", "class_7": "class 7, unclassified", "buffer": "2 px buffer",
              "snow": "snow test", "haze": "haze test", "merge": "another tile copy flags it",
              "class_2_near": "class 2, near cloud or snow", "class_2_far": "class 2, away from them",
              "class_7_near": "class 7, near cloud or snow", "class_7_far": "class 7, away from them",
              "kept": "kept by revised (reference)"}
SPLITS = ["class_2_near", "class_2_far", "class_7_near", "class_7_far"]

# Reference categorical order, slots 1 to 7, one per variant in a fixed order.
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]
COLOR = dict(zip(VARIANTS, SLOT))
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False, "axes.spines.right": False,
    "text.color": INK, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "xtick.color": MUTED,
    "ytick.color": MUTED, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2, "font.size": 9,
    "axes.titlesize": 10, "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    "figure.dpi": 110,
})


def month_table(mm: pd.DataFrame) -> pd.DataFrame:
    # Pooled over chips, weighted by labelled pixels.
    g = mm.assign(filled_px=mm.filled_share * mm.labelled_px, dates_px=mm.clear_dates * mm.labelled_px)
    t = g.groupby(["variant", "month"]).agg(labelled=("labelled_px", "sum"), filled=("filled_px", "sum"),
                                            dates=("dates_px", "sum"), white=("white", "sum"),
                                            observed=("observed", "sum"), dip=("dip", "sum"),
                                            dip_den=("dip_denominator", "sum"))
    t["filled_share"] = t.filled / t.labelled
    t["clear_dates"] = t.dates / t.labelled
    return t.reset_index()


def season_summary(t: pd.DataFrame) -> pd.DataFrame:
    out = []
    for v in VARIANTS:
        r = {"variant": v}
        for name, ms in (("Apr-Sep", GROWING), ("Oct-Mar", [10, 11, 12, 1, 2, 3])):
            s = t[(t.variant == v) & t.month.isin(ms)]
            r[f"filled {name}"] = s.filled.sum() / s.labelled.sum()
            r[f"white per 10k obs {name}"] = 1e4 * s.white.sum() / max(s.observed.sum(), 1)
            d = s[s.dip_den > 0]
            r[f"dips per 1k obs {name}"] = 1e3 * d.dip.sum() / max(d.dip_den.sum(), 1)
        out.append(r)
    return pd.DataFrame(out).set_index("variant")


def paired_by_chip(mm: pd.DataFrame, ref: str = "revised") -> pd.DataFrame:
    """Per chip, April to September: each variant minus ``ref``, as mean, spread and chips above."""
    g = mm[mm.month.isin(GROWING)].assign(filled_px=lambda d: d.filled_share * d.labelled_px)
    c = g.groupby(["chip_id", "variant"]).agg(filled=("filled_px", "sum"), lab=("labelled_px", "sum"),
                                              dip=("dip", "sum"), den=("dip_denominator", "sum"))
    c["filled_pp"] = 100 * c.filled / c.lab
    c["dips_per_1k"] = 1e3 * c.dip / c.den.clip(lower=1)
    rows = []
    for v in VARIANTS:
        if v == ref:
            continue
        d = c.xs(v, level="variant")[["filled_pp", "dips_per_1k"]] - c.xs(ref, level="variant")[["filled_pp", "dips_per_1k"]]
        rows.append({"variant": v,
                     "filled, points": f"{d.filled_pp.mean():+.2f} ± {d.filled_pp.std():.2f}",
                     "dips per 1k": f"{d.dips_per_1k.mean():+.1f} ± {d.dips_per_1k.std():.1f}",
                     "chips with more dips": f"{int((d.dips_per_1k > 0).sum())}/{len(d)}"})
    return pd.DataFrame(rows).set_index("variant")


def hist_quantiles(h: np.ndarray, edges: np.ndarray, qs=(0.25, 0.5, 0.75)) -> list[float]:
    c = np.cumsum(h)
    if c[-1] == 0:
        return [np.nan] * len(qs)
    centres = (edges[:-1] + edges[1:]) / 2
    return [float(centres[np.searchsorted(c, q * c[-1])]) for q in qs]


def removed_table() -> pd.DataFrame:
    z = np.load(TAB / "removed_hist.npz")
    rows = []
    for k, hb, hn, hr, nr in zip(z["keys"], z["b02"], z["ndvi"], z["nir"], z["no_ref"]):
        kind, rule, season = str(k).split("|")
        qb, qn = hist_quantiles(hb, z["b02_edges"]), hist_quantiles(hn, z["ndvi_edges"])
        qr = hist_quantiles(hr, z["nir_edges"])
        nir_c = (z["nir_edges"][:-1] + z["nir_edges"][1:]) / 2
        n = int(hb.sum())
        rows.append({"kind": kind, "rule": rule, "season": season, "n": n, "no_reference": int(nr),
                     "dB02 q25": qb[0], "dB02 median": qb[1], "dB02 q75": qb[2],
                     "dNDVI q25": qn[0], "dNDVI median": qn[1], "dNDVI q75": qn[2],
                     "dNIR q25": qr[0], "dNIR median": qr[1], "dNIR q75": qr[2],
                     "share dNIR < -0.25": float(hr[nir_c < -0.25].sum() / max(hr.sum(), 1)),
                     "share dB02 > 0.02": float(hb[(z["b02_edges"][:-1] + z["b02_edges"][1:]) / 2 > 0.02].sum() / max(n, 1))})
    return pd.DataFrame(rows)


def fig_filled(t: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    x = np.arange(1, 13)
    for v in VARIANTS:
        s = t[t.variant == v].set_index("month").reindex(x)
        axes[0].plot(x, s.filled_share, color=COLOR[v], label=v, marker="o", ms=4, mec=SURFACE)
        axes[1].plot(x, s.clear_dates, color=COLOR[v], label=v, marker="o", ms=4, mec=SURFACE)
    axes[0].set_title("Share of labelled pixels gap-filled")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_title("Clear dates per labelled pixel")
    for a in axes:
        a.set_xticks(x, MONTHS)
        a.axvspan(3.5, 9.5, color=GRID, alpha=0.35, lw=0, zorder=0)
    axes[0].legend(loc="upper center", ncol=2, fontsize=8)
    fig.suptitle("Monthly composites of 30 sample chips under each screening; shaded: April to September",
                 x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "filled_and_dates.png")
    plt.close(fig)


def fig_removed(rt: pd.DataFrame) -> None:
    order = RULES + SPLITS + ["kept"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8), layout="constrained", sharey=True)
    seasons = [("Apr-Sep", SLOT[0], -0.13), ("Oct-Mar", SLOT[1], 0.13)]
    y = np.arange(len(order))[::-1]
    for ax, q, title in ((axes[0], "dB02", "Blue reflectance minus the clean median"),
                         (axes[1], "dNIR", "Near infrared relative to the clean median"),
                         (axes[2], "dNDVI", "NDVI minus the clean median")):
        for season, color, off in seasons:
            s = rt[(rt.kind == "only") & (rt.season == season)].set_index("rule").reindex(order)
            ax.hlines(y + off, s[f"{q} q25"], s[f"{q} q75"], color=color, lw=2)
            ax.plot(s[f"{q} median"], y + off, "o", color=color, ms=7, mec=SURFACE, mew=1.5, label=season)
        ax.axvline(0, color=MUTED, lw=0.8)
        ax.set_title(title)
    axes[0].set_yticks(y, [RULE_LABEL[r] for r in order])
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Observations removed by one rule only, against the same pixel's clean observations that "
                 "month; median and interquartile range", x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "removed_observations.png")
    plt.close(fig)


# Label offsets in points for the trade-off plot, per panel, so neighbouring variants do not collide.
OFFSETS = [{"export": (6, 4), "export_merged": (6, -10), "revised": (6, -12), "revised_no_haze": (-30, 8),
            "revised_keep7": (-20, -14), "revised_27_near": (6, 4), "revised_keep2": (6, 4)},
           {"export": (6, 4), "export_merged": (6, 4), "revised": (6, -12), "revised_no_haze": (-20, 12),
            "revised_keep7": (6, 10), "revised_27_near": (-30, -14), "revised_keep2": (6, 10)}]


def fig_tradeoff(ss: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), layout="constrained")
    base = ss.loc["export"]
    for v in VARIANTS:
        r = ss.loc[v]
        dx = 100 * (r["filled Apr-Sep"] - base["filled Apr-Sep"])
        axes[0].plot(dx, r["dips per 1k obs Apr-Sep"], "o", color=COLOR[v], ms=9, mec=SURFACE, mew=1.5)
        axes[0].annotate(v, (dx, r["dips per 1k obs Apr-Sep"]), xytext=OFFSETS[0][v], textcoords="offset points",
                         fontsize=8, color=INK_2)
        dx2 = 100 * (r["filled Oct-Mar"] - base["filled Oct-Mar"])
        axes[1].plot(dx2, r["white per 10k obs Oct-Mar"], "o", color=COLOR[v], ms=9, mec=SURFACE, mew=1.5)
        axes[1].annotate(v, (dx2, r["white per 10k obs Oct-Mar"]), xytext=OFFSETS[1][v], textcoords="offset points",
                         fontsize=8, color=INK_2)
    axes[0].axvline(2, color=MUTED, lw=0.8, ls="--")
    axes[0].set_title("April to September")
    axes[0].set_xlabel("gap-filled share of labelled pixels, points above export")
    axes[0].set_ylabel("NDVI dips per 1,000 observed pixel-months")
    axes[1].set_title("October to March")
    axes[1].set_xlabel("gap-filled share of labelled pixels, points above export")
    axes[1].set_ylabel("white pixels per 10,000 observed pixel-months")
    fig.suptitle("What each screening costs in observations and what it removes; dashed: the 2 point limit of "
                 "the decision rule", x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "tradeoff.png")
    plt.close(fig)


def fig_gallery(picks: list[tuple[str, int]]) -> None:
    show = ["export", "revised", "revised_27_near", "revised_keep2"]
    fig, axes = plt.subplots(len(picks), len(show), figsize=(3.1 * len(show), 3.2 * len(picks)),
                             layout="constrained", squeeze=False)
    for i, (chip, m) in enumerate(picks):
        z = np.load(RES / "composites" / f"{chip}.npz")
        for j, v in enumerate(show):
            a = z[v][m - 1].astype(np.float32) / 10000.0
            rgb = np.clip(np.stack([a[2], a[1], a[0]], -1) / 0.25, 0, 1)
            empty = (z[v][m - 1] == 0).all(0)
            rgb[empty] = 0.85
            ax = axes[i, j]
            ax.imshow(rgb, interpolation="nearest")
            ax.contour(z["lab"].astype(float), levels=[0.5], colors="#fab219", linewidths=0.5)
            ax.set_title(f"{chip} {MONTHS[m - 1]}, {v}\nlabelled pixels without a clear observation: "
                         f"{empty[z['lab']].mean():.0%}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
    fig.suptitle("Monthly composites before gap filling, true colour 0 to 0.25; light grey: no clear "
                 "observation; yellow: labelled parcels", x=0.01, ha="left", fontsize=11)
    fig.savefig(FIG / "gallery.png")
    plt.close(fig)


def md_table(df: pd.DataFrame, fmt) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join([df.index.name or ""] + cols) + " |", "|" + "---|" * (len(cols) + 1)]
    for idx, r in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + [fmt(c, r[c]) for c in cols]) + " |")
    return "\n".join(lines)


def main() -> None:
    FIG.mkdir(exist_ok=True)
    mm = pd.read_parquet(TAB / "month_metrics.parquet")
    t = month_table(mm)
    ss = season_summary(t)
    rt = removed_table()
    rc = pd.read_parquet(TAB / "removed_counts.parquet")
    fig_filled(t)
    fig_removed(rt)
    fig_tradeoff(ss)
    picks = [tuple(p) for p in pd.read_csv(HERE / "gallery.csv").itertuples(index=False)] \
        if (HERE / "gallery.csv").exists() else []
    if picks:
        fig_gallery(picks)

    pct = lambda c, v: f"{v:.1%}"
    num = lambda c, v: f"{v:.2f}"
    filled = t.pivot(index="variant", columns="month", values="filled_share").reindex(VARIANTS)
    filled.columns = MONTHS
    dates = t.pivot(index="variant", columns="month", values="clear_dates").reindex(VARIANTS)
    dates.columns = MONTHS
    ssf = ss.copy()
    share = (rc.groupby(["rule"]).agg(removed=("removed", "sum"), only=("removed_only", "sum")))
    total = rc.drop_duplicates(["chip_id", "month"]).export_clear.sum()
    share = (share / total).reindex(RULES + SPLITS)
    share.index.name = "rule"
    grow = rc[rc.month.isin(GROWING)]
    gtotal = grow.drop_duplicates(["chip_id", "month"]).export_clear.sum()
    gshare = (grow.groupby("rule")[["removed", "removed_only"]].sum() / gtotal).reindex(RULES + SPLITS)
    share["removed Apr-Sep"] = gshare.removed
    share["only Apr-Sep"] = gshare.removed_only
    rto = rt[rt.kind == "only"].set_index(["rule", "season"]).sort_index()
    rto = rto[["n", "dB02 median", "share dB02 > 0.02", "dNIR median", "share dNIR < -0.25",
               "dNDVI median", "dNDVI q25"]]
    chips = pd.read_csv(TAB / "chips.csv")
    paired = paired_by_chip(mm)
    text = [
        "# Tables", "",
        f"{len(chips)} chips, {chips['items'].sum():,} acquisitions, {chips.overpasses.sum():,} distinct overpasses, "
        f"{chips.reprocessed_items.sum():,} reprocessed at baseline 04.00, {chips.unreadable.sum()} unreadable.", "",
        "## Share of labelled pixels gap-filled", "", md_table(filled, pct), "",
        "## Clear dates per labelled pixel", "", md_table(dates, num), "",
        "## Seasons", "",
        md_table(ssf, lambda c, v: f"{v:.1%}" if c.startswith("filled") else f"{v:.1f}"), "",
        "## Per chip, April to September, minus revised: mean ± standard deviation over chips", "",
        md_table(paired, lambda c, v: v), "",
        "## Share of the observations export keeps that each rule removes, labelled pixels", "",
        "`removed`: all the rule removes; `only`: what no other revised rule removes.", "",
        md_table(share, pct), "",
        "## Observations removed by one rule only, against the pixel's clean median", "",
        md_table(rto.reset_index().set_index("rule"),
                 lambda c, v: f"{v:,}" if c == "n" else (v if c == "season" else
                                                         (f"{v:.0%}" if c.startswith("share") else
                                                          (f"{v:+.0%}" if c.startswith("dNIR") else f"{v:+.3f}")))),
    ]
    (FIG / "tables.md").write_text("\n".join(text) + "\n")
    print("\n".join(text))


if __name__ == "__main__":
    main()
