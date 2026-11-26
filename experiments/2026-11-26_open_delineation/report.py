"""Figures and tables of the comparison with the open delineation models: ``figures/`` and ``figures/tables.md``.

Run in the Poetry environment, after ``score.py``::

    poetry run python experiments/2026-11-26_open_delineation/report.py [--select-only]

**Choice on validation.** For each external model the variant (window or month, input recipe, confidence, fold and
normalisation) and the field rule are those of the best border F1 within one pixel on the 769 validation chips,
truth ``all``; nothing is chosen on the test chips. ``--select-only`` prints the choices (the U-TAE fold and
normalisation feed ``predict_utae.py --stage final``).

**Intervals.** 95 % intervals from a block bootstrap over the test chips' spatial blocks (the split's 4 x 4 chip
blocks, 1,000 resamples, seed 0): the same resamples for every model, so the paired differences are paired.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402

import common as C  # noqa: E402
import score as SC  # noqa: E402

OUT = C.HERE / "figures"
SUBSET = "test_clean"

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "x", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7,
})

#: Our arms of the two-head workflow: label and colour slot (those of docs/phase1/results_estonia_borders).
ARMS = {"unet_b7_s2s1": ("Ours: PRUE from ImageNet, S2 + S1 (baseline)", 6),
        "tessera_v2_medium": ("Ours: TESSERA v2 M + PRUE's U-Net", 4),
        "alphaearth_v1": ("Ours: AlphaEarth v1 + PRUE's U-Net", 5),
        "prithvi_eo_v2_600_tl": ("Ours: Prithvi-EO-2.0 600M", 2),
        "terramind_v1_large_s2s1": ("Ours: TerraMind v1 L, S2 + S1", 1),
        "thor_v1_large_s2s1": ("Ours: THOR v1 L, S2 + S1", 3)}
#: External families: label, label prefix, variant filter, colour, dates read by the variant.
FAMILIES = {
    "prue_b7": ("FTW PRUE B7 (v3.1), 2 dates", "prue_b7/", lambda v: v != "pairs_mean", SLOTS[6]),
    "v2_multi": ("FTW v2, 2 dates", "v2_multi/", lambda v: v != "pairs_mean", SLOTS[6]),
    "v2_single": ("FTW v2, 1 date", "v2_single/", lambda v: v != "season_mean", SLOTS[6]),
    "v2_single_mean": ("FTW v2 one-window, 6 dates averaged", "v2_single/", lambda v: v == "season_mean", SLOTS[6]),
    "delany_v2": ("Delineate Anything v2, 1 date", "delany_v2/", lambda v: "fused" not in v and "3class" not in v,
                  SLOTS[7]),
    "delany_v2_fused": ("Delineate Anything v2, 6 dates fused", "delany_v2/", lambda v: "fused" in v, SLOTS[7]),
    "delany_v1": ("Delineate Anything v1, 1 date", "delany_v1/", lambda v: "fused" not in v and "3class" not in v,
                  SLOTS[7]),
    "utae_paps": ("U-TAE + PaPs (PASTIS), 12 dates", "utae_paps/", lambda v: v.endswith("all12"), INK_2),
}
#: Mid-window settings, scored beside the validation choice: (variant, fields, label). FTW's Estonian windows are
#: April to June and July to September (one scene each); the middle months stand in for an untuned choice.
DEFAULTS = {"prue_b7": ("AMay_BAug", "cc", "mid-window pair (May + Aug)"),
            "v2_multi": ("AMay_BAug", "cc", "mid-window pair (May + Aug)"),
            "v2_single": ("WMay", "cc", "mid-window (May)")}


def load() -> pd.DataFrame:
    df = pd.read_csv(SC.SCORES / "summary.csv", dtype={"fields": str})
    df["variant"] = df.label.str.split("/").str[1]
    return df


def choose(df: pd.DataFrame, fam: str):
    """(label, fields) of the family's best validation border F1, truth all; None if not scored yet."""
    _, prefix, keep, _ = FAMILIES[fam]
    v = df[(df.split == "val") & (df.truth == "all") & df.label.str.startswith(prefix)]
    v = v[v.variant.map(keep)]
    if v.empty:
        return None
    r = v.loc[v.border_f1_d1.idxmax()]
    return r.label, r.fields


def entries(df: pd.DataFrame) -> list[dict]:
    """The models of the headline, in display order: our arms at K = 100 %, then the external choices."""
    out = []
    for arm, (lab, slot) in ARMS.items():
        out.append({"key": arm, "name": lab, "label": f"arm/{arm}/P100", "fields": "grow", "group": "ours",
                    "colour": SLOTS[slot - 1]})
    for fam, (lab, prefix, _, col) in FAMILIES.items():
        ch = choose(df, fam)
        if ch is None or row(df, ch[0], ch[1], SUBSET, "all") is None:
            continue
        out.append({"key": fam, "name": lab, "label": ch[0], "fields": ch[1], "group": "open", "colour": col})
        if fam in DEFAULTS:
            d = f"{prefix}{DEFAULTS[fam][0]}"
            if (d, DEFAULTS[fam][1]) != ch:
                out.append({"key": fam + "_default", "name": lab + ", " + DEFAULTS[fam][2], "label": d,
                            "fields": DEFAULTS[fam][1], "group": "open", "colour": col})
    return out


def row(df, label, fields, subset, truth, split="test") -> pd.Series:
    r = df[(df.label == label) & (df.fields == fields) & (df.subset == subset) & (df.truth == truth)
           & (df.split == split)]
    return r.iloc[0] if len(r) else None


# ------------------------------------------------------------------------------------------- block bootstrap

def _counts(label: str, split: str, fields: str):
    z = np.load(SC.SCORES / f"{label.replace('/', '__')}__{split}.npz")
    return [str(c) for c in z["ids"]], z[f"v_{fields}"]


def bootstrap(models: list[dict], subset: str = SUBSET, n: int = 1000, seed: int = 0) -> dict:
    """Per model and truth: border F1 (d = 1) and class-agnostic PQ, point and 2.5 / 97.5 % over block resamples;
    and per model the resampled values, for paired differences."""
    from gfm4agri.borders import scoring as S

    nfp = len(S.FP_WC)
    d1 = 4 + 3 + nfp + 4                                       # tol rows d0, d1, d2 of 4 counts each
    blocks = pd.read_csv(C.SPLIT / "chips.csv").set_index("chip_id")["block_id"]
    ov = pd.read_csv(C.HERE / "reference" / "ftw_overlap.csv").set_index("chip_id")["ftw"]
    ids = [c for c in C.split_ids("test") if subset == "test" or ov.get(c, "none") == "none"]
    bl = pd.Index(sorted(set(blocks[ids])))
    bidx = bl.get_indexer(blocks[ids])
    rng = np.random.default_rng(seed)
    M = np.stack([np.bincount(rng.integers(0, len(bl), len(bl)), minlength=len(bl)) for _ in range(n)])
    out = {}
    for m in models:
        cid, c = _counts(m["label"], "test", m["fields"])
        pos = {k: i for i, k in enumerate(cid)}
        c = c[[pos[k] for k in ids]]                            # (chips, truths, K)
        res = {}
        for t, name in enumerate(SC.TRUTHS):
            per = np.zeros((len(bl), 8))
            np.add.at(per, bidx, np.concatenate([c[:, t, 0:4], c[:, t, d1:d1 + 4]], axis=1))
            full = per.sum(0)[None]
            vals = []
            for s in (full, M @ per):
                tp, fp, fn, iou, tpp, npred, tpr, ntrue = s.T
                p, r = tpp / np.maximum(npred, 1), tpr / np.maximum(ntrue, 1)
                f1 = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0)
                pq = iou / np.maximum(tp + fp / 2 + fn / 2, 1e-12)
                vals.append((f1, pq))
            res[name] = {"border_f1": float(vals[0][0][0]), "pq": float(vals[0][1][0]),
                         "border_f1_ci": np.percentile(vals[1][0], [2.5, 97.5]).tolist(),
                         "pq_ci": np.percentile(vals[1][1], [2.5, 97.5]).tolist(),
                         "_f1": vals[1][0], "_pq": vals[1][1]}
        out[m["key"]] = res
    out["_blocks"] = len(bl)
    out["_chips"] = len(ids)
    return out


def paired(bs: dict, a: str, b: str, truth: str) -> dict:
    d_f1 = bs[a][truth]["_f1"] - bs[b][truth]["_f1"]
    d_pq = bs[a][truth]["_pq"] - bs[b][truth]["_pq"]
    return {"border_f1": bs[a][truth]["border_f1"] - bs[b][truth]["border_f1"],
            "border_f1_ci": np.percentile(d_f1, [2.5, 97.5]).tolist(),
            "pq": bs[a][truth]["pq"] - bs[b][truth]["pq"], "pq_ci": np.percentile(d_pq, [2.5, 97.5]).tolist()}


# ------------------------------------------------------------------------------------------------------ figures

def fig_headline(models, bs):
    models = [m for m in models if m["key"] not in ("v2_multi_default", "v2_single_default")]   # in tables.md
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2), sharey=True)
    names = [m["name"] for m in models][::-1]
    for j, truth in enumerate(SC.TRUTHS):
        for i, (metric, title) in enumerate((("border_f1", "Border F1 within 1 px"),
                                             ("pq", "Fields: panoptic quality, class-agnostic"))):
            ax = axes[i, j]
            for y, m in enumerate(models[::-1]):
                r = bs[m["key"]][truth]
                v, (lo, hi) = r[metric], r[f"{metric}_ci"]
                ax.barh(y, v, height=0.62, color=m["colour"], alpha=0.9 if m["group"] == "ours" else 0.75,
                        hatch=None if m["group"] == "ours" else "///", edgecolor=SURFACE, linewidth=0.6)
                ax.plot([lo, hi], [y, y], color=INK, lw=1)
                ax.text(hi + 0.01, y, f"{v:.3f}", va="center", fontsize=8, color=INK_2)
            ax.set_yticks(range(len(names)), names, fontsize=8.5)
            ax.set_xlim(0, 1.0 if metric == "border_f1" else 0.6)
            ax.set_title(f"{title}\n{'all parcels' if truth == 'all' else 'grassland parcels set aside'}")
    fig.suptitle(f"Test chips no FTW chip touches ({bs['_chips']} chips, {bs['_blocks']} blocks); "
                 "bars: point, whiskers: 95 % block bootstrap.\nSolid: our arms, trained on our labels "
                 "(K = 100 %); hatched: open models, zero-shot, variant chosen on validation",
                 x=0.01, ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "headline.png", dpi=150)
    plt.close(fig)


def fig_budget(df, models):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    ext = [m for m in models if m["key"] in ("prue_b7", "v2_single", "delany_v2", "delany_v2_fused", "utae_paps")]
    for j, truth in enumerate(SC.TRUTHS):
        ax = axes[j]
        ax.grid(axis="y")
        for arm, (lab, slot) in ARMS.items():
            ys = [row(df, f"arm/{arm}/P{k}", "grow", SUBSET, truth).border_f1_d1 for k in SC.BUDGETS]
            ax.plot(range(3), ys, color=SLOTS[slot - 1], marker="o", label=lab.replace("Ours: ", ""))
        styles = ["--", ":", "-.", (0, (5, 1)), (0, (1, 3))]
        vals = sorted((row(df, m["label"], m["fields"], SUBSET, truth).border_f1_d1, i, m) for i, m in enumerate(ext))
        ylab, last = [], -1.0
        for v, _, _ in vals:                              # labels at least 0.022 apart
            last = max(v, last + 0.022)
            ylab.append(last)
        for (v, i, m), yl in zip(vals, ylab):
            ax.plot([-0.2, 2.25], [v, v], color=m["colour"], ls=styles[i % len(styles)], lw=1.3)
            ax.plot([2.25, 2.4], [v, yl], color=MUTED, lw=0.6)
            ax.text(2.42, yl, f"{m['name']} ({v:.3f})", fontsize=7, color=INK_2, va="center")
        ax.set_xticks(range(3), [f"K = {k} %" for k in SC.BUDGETS])
        ax.set_xlim(-0.2, 4.6)
        ax.set_ylabel("border F1 within 1 px")
        ax.set_title("all parcels" if truth == "all" else "grassland parcels set aside")
    axes[0].legend(fontsize=7.5, loc="lower left")
    fig.suptitle("Our arms against the label budget; lines: open models, zero-shot (test chips no FTW chip touches)",
                 x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "budget.png", dpi=150)
    plt.close(fig)


def _best(df, prefix, keep, split="val"):
    v = df[(df.split == split) & (df.truth == "all") & df.label.str.startswith(prefix) & df.variant.map(keep)]
    return v.loc[v.border_f1_d1.idxmax()] if len(v) else None


def temporal_rows(df) -> list[dict]:
    """Each family's border F1 by the number of monthly composites it reads; choices on val, values on test."""
    pts = []

    def add(fam, n, label, fields, how):
        for truth in SC.TRUTHS:
            r = row(df, label, fields, SUBSET, truth)
            if r is not None:
                pts.append({"family": fam, "dates": n, "how": how, "truth": truth, "label": label,
                            "fields": fields, "border_f1": r.border_f1_d1, "pq": r.pq_agnostic})

    for fam, pre in (("delany_v2", "delany_v2/"), ("delany_v1", "delany_v1/")):
        b = _best(df, pre, lambda v: "fused" not in v and "3class" not in v)
        if b is None:
            continue
        recipe = b.variant.split("_")[0]
        add(fam + "_inst", 1, b.label, b.fields, f"best month on val ({b.variant}), its own instances")
        b3 = _best(df, pre, lambda v, r=recipe: v.startswith(r + "_") and v.endswith("_3class"))
        if b3 is not None:                         # the same field rule as the fusion: one date against six
            add(fam, 1, b3.label, "grow", f"best month on val ({b3.variant}), fields as the fusion builds them")
        add(fam, 6, f"{pre}{recipe}_fused6", "grow", "six months fused")
    b = _best(df, "v2_single/", lambda v: v != "season_mean")
    if b is not None:
        add("v2s", 1, b.label, b.fields, f"best month on val ({b.variant})")
        add("v2s", 6, "v2_single/season_mean", b.fields, "six single-date maps averaged")
    for fam, pre in (("v2m", "v2_multi/"), ("prue_b7", "prue_b7/")):
        b = _best(df, pre, lambda v: v != "pairs_mean")
        if b is not None:
            add(fam, 2, b.label, b.fields, f"best pair on val ({b.variant})")
            add(fam, 6, f"{pre}pairs_mean", b.fields, "nine pairs averaged (Apr to Sep)")
    u = _best(df, "utae_paps/", lambda v: v.endswith("all12"))
    if u is not None:
        stem = u.variant.rsplit("_", 1)[0]
        for ms, n in (("pair2", 2), ("season6", 6), ("all12", 12)):
            add("utae", n, f"utae_paps/{stem}_{ms}", u.fields, {2: "May, Aug", 6: "Apr to Sep", 12: "all"}[n])
    add("ours", 12, "arm/unet_b7_s2s1/P100", "grow", "trained on our labels, S2 + S1")
    add("ours_tessera", 12, "arm/tessera_v2_medium/P100", "grow", "trained on our labels")
    return pts


TFAM = {"delany_v2": ("Delineate Anything v2", SLOTS[7], "-"), "delany_v1": ("Delineate Anything v1", SLOTS[7], "--"),
        "delany_v2_inst": ("Delineate Anything v2", SLOTS[7], None),
        "delany_v1_inst": ("Delineate Anything v1", SLOTS[7], None),
        "v2s": ("FTW v2, one window", SLOTS[6], ":"), "v2m": ("FTW v2, two windows", SLOTS[6], "--"),
        "prue_b7": ("FTW PRUE B7", SLOTS[6], "-"),
        "utae": ("U-TAE + PaPs", INK_2, "-"), "ours": ("Ours: PRUE from ImageNet", SLOTS[5], ""),
        "ours_tessera": ("Ours: TESSERA v2 M", SLOTS[3], "")}


def fig_temporal(df, pts):
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.3))
    t = pd.DataFrame(pts)
    for j, truth in enumerate(SC.TRUTHS):
        ax = axes[j]
        ax.grid(axis="y")
        for fam, (lab, col, ls) in TFAM.items():
            s = t[(t.family == fam) & (t.truth == truth)].sort_values("dates")
            if s.empty or ls is None:                  # own-instance single dates: in the table only
                continue
            ax.plot(s.dates, s.border_f1, color=col, ls=ls or "none", marker="o" if ls else "*",
                    markersize=7 if ls else 13, label=lab)
        ax.set_xscale("log", base=2)
        ax.set_xticks([1, 2, 6, 12], ["1", "2", "6", "12"])
        ax.set_xlabel("monthly composites the model reads")
        ax.set_ylabel("border F1 within 1 px")
        ax.set_title("all parcels" if truth == "all" else "grassland parcels set aside")
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, bbox_to_anchor=(0.36, 0.0))
    ax = axes[2]
    ax.grid(axis="y")
    for fam, pre, keep, lab, col, ls in (("delany_v2", "delany_v2/", "official_", "Delineate Anything v2", SLOTS[7], "-"),
                                         ("delany_v1", "delany_v1/", "official_", "Delineate Anything v1", SLOTS[7], "--"),
                                         ("v2", "v2_single/", "W", "FTW v2, 1 window", SLOTS[6], "--")):
        ys = []
        for m in C.SEASON:
            sub = df[(df.split == "test") & (df.subset == SUBSET) & (df.truth == "no_grass")
                     & (df.label == f"{pre}{keep}{C.MONTHS[m]}")]
            b = _best(df, pre, lambda v, k=keep: v.startswith(k) and "fused" not in v and "3class" not in v
                      and v != "season_mean")
            ys.append(sub[sub.fields == b.fields].border_f1_d1.iloc[0] if b is not None and len(sub) else np.nan)
        ax.plot(range(6), ys, color=col, ls=ls, marker="o", label=lab)
    ax.set_xticks(range(6), [C.MONTHS[m] for m in C.SEASON])
    ax.set_ylabel("border F1 within 1 px")
    ax.set_title("one date: which month (grassland set aside; each model's own fields)")
    ax.legend(fontsize=7.5)
    fig.suptitle("Does seeing more of the season help? Choices on validation, values on the test chips no FTW chip "
                 "touches", x=0.01, ha="left", fontsize=10.5, color=INK)
    fig.tight_layout(rect=(0, 0.12, 1, 1))
    fig.savefig(OUT / "temporal.png", dpi=150)
    plt.close(fig)


def fig_pairs(df):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, pre, title in zip(axes, ("prue_b7/", "v2_multi/"), ("FTW PRUE B7", "FTW v2, two windows")):
        b = _best(df, pre, lambda v: v != "pairs_mean")
        if b is None:
            continue
        grid = np.zeros((3, 3))
        for i, a in enumerate((3, 4, 5)):
            for k, bb in enumerate((6, 7, 8)):
                grid[i, k] = row(df, f"{pre}A{C.MONTHS[a]}_B{C.MONTHS[bb]}", b.fields, SUBSET,
                                 "no_grass").border_f1_d1
        im = ax.imshow(grid, cmap="Blues", vmin=grid.min() - 0.02, vmax=grid.max())
        for i in range(3):
            for k in range(3):
                ax.text(k, i, f"{grid[i, k]:.3f}", ha="center", va="center", fontsize=8.5,
                        color="white" if grid[i, k] > grid.mean() else INK)
        pm = row(df, f"{pre}pairs_mean", b.fields, SUBSET, "no_grass").border_f1_d1
        ax.set_xticks(range(3), ["Jul", "Aug", "Sep"])
        ax.set_yticks(range(3), ["Apr", "May", "Jun"])
        ax.set_xlabel("window B")
        ax.set_ylabel("window A")
        ax.grid(False)
        ax.set_title(f"{title} ({b.fields} fields)\nmean of the nine pairs: {pm:.3f}")
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("Border F1 by window pair, grassland set aside, test chips no FTW chip touches", x=0.01,
                 ha="left", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "pairs.png", dpi=150)
    plt.close(fig)


def fig_types(df, models):
    from gfm4agri.borders.scoring import TYPES

    keep = [m for m in models if not m["key"].endswith("_default")
            and m["key"] in ("unet_b7_s2s1", "tessera_v2_medium", "prithvi_eo_v2_600_tl", "prue_b7", "delany_v2",
                             "delany_v2_fused", "utae_paps")]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.grid(axis="y")
    w = 0.8 / len(keep)
    for i, m in enumerate(keep):
        r = row(df, m["label"], m["fields"], SUBSET, "all")
        ys = [r[f"recall_{t.replace(' ', '_')}"] for t in TYPES]  # noqa: F841
        xs = np.arange(len(TYPES)) + (i - len(keep) / 2 + 0.5) * w
        ax.bar(xs, ys, width=w * 0.92, color=m["colour"], hatch=None if m["group"] == "ours" else "///",
               edgecolor=SURFACE, label=m["name"])
        for x, y in zip(xs, ys):
            ax.text(x, y + 0.01, f"{y:.2f}", ha="center", fontsize=6.5, color=INK_2)
    ax.set_xticks(range(len(TYPES)), [f"across: {t}" for t in TYPES])
    ax.set_ylabel("recall of the declared borders within 1 px")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.set_title("Which declared borders each model draws (all parcels, test chips no FTW chip touches)")
    fig.tight_layout()
    fig.savefig(OUT / "border_types.png", dpi=150)
    plt.close(fig)


def fig_examples(models, n: int = 3):
    """A few test chips no FTW chip touches: the July composite, the declared parcels, each model's fields."""
    from gfm4agri.benchmark.parcel_metrics import read_parcels
    from gfm4agri.borders import prue

    keep = [m for m in models if m["key"] in ("unet_b7_s2s1", "tessera_v2_medium", "prue_b7", "delany_v2",
                                              "utae_paps")]
    ov = pd.read_csv(C.HERE / "reference" / "ftw_overlap.csv").set_index("chip_id")["ftw"]
    man = {c["chip_id"]: c for c in json.loads((C.CHIPS / "manifest.json").read_text())["chips"]}
    cand = [c for c in C.split_ids("test") if ov.get(c, "none") == "none" and man[c]["labelled_share"] > 0.55
            and 25 <= man[c]["n_parcels"] <= 80]
    pick = [cand[i] for i in np.linspace(0, len(cand) - 1, n).astype(int)]
    segs = {}
    for m in keep:
        path = (C.ARMS_RESULTS / m["label"].split("/")[1] / f"P100_draw0_seed0" / "predictions_test.npz"
                if m["group"] == "ours" else C.RESULTS / m["label"] / "predictions_test.npz")
        items = {it["id"]: it for it in SC.load_items(path) if it["id"] in pick}
        segs[m["key"]] = {c: SC.fields_of(items[c], m["fields"]) for c in pick}
    cols = 2 + len(keep)
    fig, axes = plt.subplots(n, cols, figsize=(2.6 * cols, 2.75 * n))
    rng = np.random.default_rng(0)
    for i, cid in enumerate(pick):
        s2 = C.read_s2(cid)[6][C.RGB].astype(np.float32)
        rgb = np.clip(np.moveaxis(s2, 0, -1) / np.percentile(s2, 99, axis=(1, 2)), 0, 1)
        parcels = read_parcels(C.CHIP_DIR, cid)
        mask = C.read_mask(cid)
        kind = np.where(mask == 0, 1, np.where(mask > 0, 2, np.where(parcels > 0, 3, 0)))
        panels = [("July composite", None), ("declared parcels: grassland green,\nother crops orange", parcels)]
        panels += [(m["name"].replace("Ours: ", "ours: ").split(",")[0], segs[m["key"]][cid]) for m in keep]
        for k, (title, seg) in enumerate(panels):
            ax = axes[i, k]
            ax.imshow(rgb if seg is None else 0.45 + 0.55 * rgb.mean(-1, keepdims=True).repeat(3, -1))
            if seg is not None:
                if k == 1:                                   # declared: grassland, other crops, out of scheme
                    lut = np.array([[1, 1, 1], to_rgb(SLOTS[2]), to_rgb(SLOTS[1]), to_rgb(MUTED)])
                    fill = lut[kind]
                else:
                    lut = rng.uniform(0.2, 0.95, (int(seg.max()) + 1, 3))
                    lut[0] = 1
                    fill = lut[seg]
                ax.imshow(fill, alpha=np.where(seg > 0, 0.5, 0.0))
                b = prue.boundary_map(seg)
                ax.imshow(np.where(b, 1.0, np.nan), cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            if i == 0:
                ax.set_title(title, fontsize=8)
        axes[i, 0].set_ylabel(cid, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "examples.png", dpi=140)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------- tables

def tables(df, models, bs, pts) -> str:
    L = ["# Tables of the open-model comparison", "",
         "Generated by `report.py` from `results/2026-11-26_open_delineation/EE_2021_rev/scores/summary.csv`.", ""]
    L += ["## Choices on the validation chips (best border F1 within 1 px, all parcels)", "",
          "| Model | Variant chosen | Fields | Val border F1 |", "|---|---|---|---|"]
    for m in models:
        if m["group"] == "open":
            r = row(df, m["label"], m["fields"], "val", "all", split="val")
            L.append(f"| {m['name']} | `{m['label']}` | {m['fields']} | {r.border_f1_d1:.3f} |")
    for subset in (SUBSET, "test"):
        for truth in SC.TRUTHS:
            L += ["", f"## Test, {subset} ({'all parcels' if truth == 'all' else 'grassland set aside'})", "",
                  "| Model | Border P / R / F1 (1 px) | F1 at 0 / 2 px | PQ / SQ / RQ (agnostic) | "
                  "Fields predicted / true |", "|---|---|---|---|---|"]
            for m in models:
                r = row(df, m["label"], m["fields"], subset, truth)
                if r is None:
                    continue
                L.append(f"| {m['name']} | {r.border_precision_d1:.3f} / {r.border_recall_d1:.3f} / "
                         f"**{r.border_f1_d1:.3f}** | {r.border_f1_d0:.3f} / {r.border_f1_d2:.3f} | "
                         f"{r.pq_agnostic:.3f} / {r.sq_agnostic:.3f} / {r.rq_agnostic:.3f} | "
                         f"{int(r.predicted_parcels):,} / {int(r.true_parcels):,} |")
    L += ["", f"## 95 % block-bootstrap intervals, {SUBSET} ({bs['_chips']} chips, {bs['_blocks']} blocks)", "",
          "| Model | Border F1, all | PQ, all | Border F1, grassland aside | PQ, grassland aside |",
          "|---|---|---|---|---|"]
    for m in models:
        r = bs[m["key"]]
        f = lambda t, k: f"{r[t][k]:.3f} [{r[t][k + '_ci'][0]:.3f}, {r[t][k + '_ci'][1]:.3f}]"  # noqa: E731
        L.append(f"| {m['name']} | {f('all', 'border_f1')} | {f('all', 'pq')} | {f('no_grass', 'border_f1')} | "
                 f"{f('no_grass', 'pq')} |")
    L += ["", "## Paired differences against our baseline (PRUE from ImageNet, K = 100 %)", "",
          "| Model minus baseline | Border F1, all | PQ, all | Border F1, grassland aside | PQ, grassland aside |",
          "|---|---|---|---|---|"]
    for m in models:
        if m["key"] == "unet_b7_s2s1":
            continue
        cells = []
        for t in SC.TRUTHS:
            p = paired(bs, m["key"], "unet_b7_s2s1", t)
            cells += [f"{p['border_f1']:+.3f} [{p['border_f1_ci'][0]:+.3f}, {p['border_f1_ci'][1]:+.3f}]",
                      f"{p['pq']:+.3f} [{p['pq_ci'][0]:+.3f}, {p['pq_ci'][1]:+.3f}]"]
        L.append(f"| {m['name']} | " + " | ".join(cells) + " |")
    L += ["", f"## Dates read, {SUBSET}", "", "| Model | Dates | How | Border F1 all / grassland aside | "
          "PQ all / grassland aside |", "|---|---|---|---|---|"]
    t = pd.DataFrame(pts)
    for (fam, n, how), g in t.groupby(["family", "dates", "how"], sort=False):
        a, ng = g[g.truth == "all"].iloc[0], g[g.truth == "no_grass"].iloc[0]
        L.append(f"| {TFAM[fam][0]} | {n} | {how} | {a.border_f1:.3f} / {ng.border_f1:.3f} | "
                 f"{a.pq:.3f} / {ng.pq:.3f} |")
    L += ["", "## Recall by what lies across the border (1 px, all parcels), " + SUBSET, "",
          "| Model | same crop | other crop | undeclared farmland | non-farm land |", "|---|---|---|---|---|"]
    for m in models:
        r = row(df, m["label"], m["fields"], SUBSET, "all")
        L.append(f"| {m['name']} | {r.recall_same_crop:.3f} | {r.recall_other_crop:.3f} | "
                 f"{r.recall_undeclared_farmland:.3f} | {r['recall_non-farm_land']:.3f} |")
    L += ["", "## Our arms by label budget, border F1 within 1 px, " + SUBSET, "",
          "| Arm | K = 100 % all / aside | K = 20 % | K = 5 % |", "|---|---|---|---|"]
    for arm, (lab, _) in ARMS.items():
        cells = []
        for k in SC.BUDGETS:
            a = row(df, f"arm/{arm}/P{k}", "grow", SUBSET, "all").border_f1_d1
            ng = row(df, f"arm/{arm}/P{k}", "grow", SUBSET, "no_grass").border_f1_d1
            cells.append(f"{a:.3f} / {ng:.3f}")
        L.append(f"| {lab} | " + " | ".join(cells) + " |")
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--select-only", action="store_true")
    args = ap.parse_args()
    df = load()
    if args.select_only:
        for fam in FAMILIES:
            print(fam, choose(df, fam))
        return
    OUT.mkdir(exist_ok=True)
    models = entries(df)
    bs = bootstrap(models)
    pts = temporal_rows(df)
    fig_headline(models, bs)
    fig_budget(df, models)
    fig_temporal(df, pts)
    fig_pairs(df)
    fig_types(df, models)
    fig_examples(models)
    (OUT / "tables.md").write_text(tables(df, models, bs, pts))
    keep = {k: {t: {kk: vv for kk, vv in v.items() if not kk.startswith("_")} for t, v in r.items()}
            for k, r in bs.items() if not k.startswith("_")}
    (OUT / "bootstrap.json").write_text(json.dumps({"chips": bs["_chips"], "blocks": bs["_blocks"],
                                                    "models": keep}, indent=1))
    print("figures and tables in", OUT)


if __name__ == "__main__":
    main()
