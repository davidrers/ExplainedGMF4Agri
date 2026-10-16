"""Parcel-level test metrics of the full-Estonia fits, beside the pixel-level ones.

Scores the stored test predictions (``predictions_test.npz``) of the fits listed in ``SOURCES`` per parcel, with
``parcel_metrics.py`` beside this file: every parcel takes the class most of its labelled pixels were predicted
as. Nothing is refitted. Writes into ``results/2026-10-16_parcel_metric/<chip set>/``:

* ``cells.csv``: one row per fit, pixel and parcel metrics, the sensitivity variants and the error split;
* ``per_class.csv``: per-class F1 per pixel and per parcel;
* ``size_bins.csv``: parcel and pixel accuracy by parcel size;

and into ``figures/`` beside this file:

* ``kshot_curves.png``: test Macro-F1 per pixel and per parcel against K, per arm;
* ``parcel_minus_pixel.png``: parcel minus pixel Macro-F1, per arm and K;
* ``error_split.png``: the misclassified pixels split by whether the vote gets their parcel right;
* ``accuracy_by_size.png``: parcel accuracy by parcel size;
* ``per_class_gap.png``: per-class F1 per parcel minus per pixel at K = 100 %;
* ``tables.md``: the tables of REPORT.md, also printed.

    poetry run python experiments/2026-10-16_parcel_metric/report.py --chips EE_2021_rev
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from parcel_metrics import chip_votes, pool_votes, read_parcels, score_parcels  # noqa: E402

NAME = HERE.name
#: (results folder, arm) -> (short label, categorical slot 1 to 8). Legend order and colour follow this.
SOURCES: dict[tuple[str, str], tuple[str, int]] = {
    ("kshot_15ep", "terramind_v1_large_s2s1"): ("TerraMind L, S2+S1", 1),
    ("kshot_15ep", "thor_v1_large_s2s1"): ("THOR L, S2+S1", 2),
    ("kshot_15ep", "prithvi_eo_v2_600_tl"): ("Prithvi 600M, S2", 3),
    ("kshot_15ep", "tessera_v2_medium"): ("TESSERA v2 + MLP", 4),
    ("kshot_15ep", "alphaearth_v1"): ("AlphaEarth + MLP", 5),
    ("2026-10-14_unet_baseline", "unet_r18_scratch"): ("U-Net S2, 15 ep", 6),
    ("2026-10-14_unet_baseline", "unet_r18_scratch_45ep"): ("U-Net S2, 45 ep", 7),
}
#: Parcel size bins, in labelled pixels inside the test chips (one pixel is 0.01 ha).
SIZE_EDGES = [1, 16, 64, 256, 1024, np.inf]
SIZE_LABELS = ["< 0.16 ha", "0.16 to 0.64", "0.64 to 2.6", "2.6 to 10", "> 10 ha"]
#: Smallest parcel kept by the size sensitivity, in labelled pixels.
SMALL = 16
#: A fit whose stored predictions miss its logged pixel Macro-F1 by more than this was predicted with another
#: checkpoint than the one tested (pipeline.md section 10), and is left out.
TOLERANCE = 0.002

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "h", "P", "X"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("gain", ["#e34948", "#f0efec", "#2a78d6"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "font.size": 9.5, "legend.frameon": False, "legend.labelcolor": INK,
})


# ---------------------------------------------------------------------------------------------------------- data

def interior(parcels: np.ndarray) -> np.ndarray:
    """Pixels whose four neighbours lie in the same parcel; the chip's own edge does not count as a boundary."""
    p = np.pad(parcels, 1, mode="edge")
    c = p[1:-1, 1:-1]
    return (c == p[:-2, 1:-1]) & (c == p[2:, 1:-1]) & (c == p[1:-1, :-2]) & (c == p[1:-1, 2:])


def cm_scores(cm: np.ndarray) -> dict:
    """Macro-F1, per-class F1, accuracy and kappa of a confusion matrix, rows the reference."""
    cm = cm.astype(float)
    tp = np.diag(cm)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    present = (tp + fp + fn) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(present, 2 * tp / (2 * tp + fp + fn), np.nan)
    total = cm.sum()
    po, pe = tp.sum() / total, (cm.sum(0) * cm.sum(1)).sum() / total ** 2
    return {"macro_f1": float(np.nanmean(f1)), "f1": f1, "accuracy": float(po), "kappa": float((po - pe) / (1 - pe))}


def cells(results: Path, chips: str) -> list[tuple[str, Path]]:
    out = []
    for (folder, arm) in SOURCES:
        out += [(arm, p.parent) for p in sorted((results / folder / chips / arm).glob("P*_draw*_seed*/results.json"))
                if (p.parent / "predictions_test.npz").exists()]
    return out


def score_cell(cell: Path, rasters: dict, cut: np.ndarray, names: list[str]) -> tuple[dict, list, list]:
    """One fit: its row, its per-class rows and its size-bin rows."""
    r = json.loads((cell / "results.json").read_text())
    d = np.load(cell / "predictions_test.npz")
    n = len(names)
    every, inner, pix = [], [], np.zeros((n, n), dtype=np.int64)
    for c, p, m in zip(d["ids"], d["pred"], d["mask"], strict=True):
        parcels, core = rasters[str(c)]
        every.append(chip_votes(p, m, parcels, n))
        inner.append(chip_votes(p, m, parcels, n, where=core))
        sel = m >= 0
        pix += np.bincount(m[sel].astype(np.int64) * n + p[sel], minlength=n * n).reshape(n, n)
    ids, votes, ref = pool_votes(every, n)
    iids, ivotes, _ = pool_votes(inner, n)
    # The interior vote: a parcel's interior pixels when it has any, all its pixels otherwise.
    vin = votes.copy()
    has = ivotes.sum(1) > 0
    vin[np.searchsorted(ids, iids[has])] = ivotes[has]

    size = votes.sum(1)
    is_cut = np.isin(ids, cut)
    big = size >= SMALL
    main = score_parcels(votes, ref, names)
    px = cm_scores(pix)
    right = votes[np.arange(len(ref)), ref]           # pixels predicted as the parcel's class
    ok = votes.argmax(1) == ref
    wrong_px = size - right
    lb = r["label_budget"]
    key = {"arm": r["run"], "pct": lb["pct"], "draw": lb["draw_seed"], "seed": r["seed"]}
    row = {**key,
           "pixel_macro_f1": px["macro_f1"], "pixel_macro_f1_logged": r["test_metrics"].get("test/F1_Score"),
           "pixel_accuracy": px["accuracy"], "pixel_kappa": px["kappa"],
           "parcel_macro_f1": main["macro_f1"], "parcel_accuracy": main["accuracy"], "parcel_kappa": main["kappa"],
           "parcel_weighted_f1": main["weighted_f1"], "n_parcels": main["n_parcels"],
           "interior_macro_f1": score_parcels(vin, ref, names)["macro_f1"],
           "no_cut_macro_f1": score_parcels(votes[~is_cut], ref[~is_cut], names)["macro_f1"],
           "n_cut": int(is_cut.sum()),
           "min_size_macro_f1": score_parcels(votes[big], ref[big], names)["macro_f1"],
           "n_small": int((~big).sum()),
           "pixels_wrong": int(wrong_px.sum()), "pixels": int(size.sum()),
           "wrong_in_right_parcels": int(wrong_px[ok].sum()),
           "wrong_in_wrong_parcels": int(wrong_px[~ok].sum()),
           "right_in_wrong_parcels": int(right[~ok].sum())}
    per_class = [{**key, "cls": c, "pixel_f1": px["f1"][i], "parcel_f1": main["per_class_f1"][c],
                  "test_parcels": main["support"][c]} for i, c in enumerate(names)]
    bins = np.digitize(size, SIZE_EDGES) - 1
    sizes = [{**key, "bin": SIZE_LABELS[b], "parcels": int((bins == b).sum()),
              "parcel_accuracy": float(ok[bins == b].mean()),
              "pixel_accuracy": float(right[bins == b].sum() / size[bins == b].sum())}
             for b in range(len(SIZE_LABELS)) if (bins == b).any()]
    return row, per_class, sizes


def load(chips: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    results = REPO / "results"
    todo = cells(results, chips)
    if not todo:
        raise SystemExit(f"no fits with test predictions for {chips} in {list(SOURCES)}")
    first = json.loads((todo[0][1] / "results.json").read_text())
    names = first["class_names"]
    chip_dir = REPO / first["config"]["data"]["root"]
    split_dir = REPO / first["config"]["data"]["split"]
    test_ids = (split_dir / "test_data.txt").read_text().split()
    ids0 = None
    for _, cell in todo:
        ids = tuple(np.load(cell / "predictions_test.npz")["ids"])
        if ids0 is None:
            ids0 = ids
        elif ids != ids0:
            raise SystemExit(f"{cell} was tested on other chips than {todo[0][1]}")
        if json.loads((cell / "results.json").read_text())["class_names"] != names:
            raise SystemExit(f"{cell} has another class list")
    if set(ids0) != set(test_ids):
        raise SystemExit(f"the predictions do not cover the test list of {split_dir.name}")
    print(f"reading the parcel rasters of {len(ids0)} test chips", flush=True)
    rasters = {}
    for c in ids0:
        p = read_parcels(chip_dir / "chips", c)
        rasters[c] = (p, interior(p))
    cp = pd.read_parquet(chip_dir / "chip_parcels.parquet")
    cut = np.unique(cp.loc[~cp["chip_id"].isin(set(ids0)), "parcel_id"].to_numpy(np.int64))
    rows, classes, sizes = [], [], []
    for arm, cell in todo:
        print(f"  {arm} {cell.name}", flush=True)
        row, pc, sz = score_cell(cell, rasters, cut, names)
        rows.append(row)
        classes += pc
        sizes += sz
    df, classes, sizes = pd.DataFrame(rows), pd.DataFrame(classes), pd.DataFrame(sizes)
    off = (df["pixel_macro_f1"] - df["pixel_macro_f1_logged"]).abs() > TOLERANCE
    stale = df.loc[off, ["arm", "pct", "draw", "seed", "pixel_macro_f1", "pixel_macro_f1_logged"]]
    keys = ["arm", "pct", "draw", "seed"]
    keep = df.loc[~off, keys]
    classes, sizes = classes.merge(keep, on=keys), sizes.merge(keep, on=keys)
    facts = {"chips": chip_dir.name, "split": split_dir.name, "test_chips": len(ids0),
             "stale": stale.to_dict("records")}
    return df[~off].reset_index(drop=True), classes, sizes, facts


def arm_styles(arms: list[str]) -> dict[str, tuple[str, str, str]]:
    """Arm -> (label, colour, marker), in the order of SOURCES."""
    return {a: (label, SLOTS[slot - 1], MARKERS[slot - 1]) for (_, a), (label, slot) in SOURCES.items() if a in arms}


# ------------------------------------------------------------------------------------------------------- figures

def _budget_axis(ax, budgets: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(budgets, [f"{b:g}" for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("K, % of trainable parcels per class")


def _end_labels(ax, items: list[tuple[float, float, str, str]], gap: float) -> None:
    """Direct labels right of the line ends, pushed apart by at least ``gap`` in data units."""
    items = sorted(items, key=lambda t: t[1])
    ys = []
    for _, y, _, _ in items:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    shift = max(0.0, ys[-1] - ax.get_ylim()[1] + gap / 2) if ys else 0.0
    for (x, _, text, colour), y in zip(items, ys, strict=True):
        ax.plot([x * 1.03, x * 1.12], [y - shift, y - shift], color=colour, lw=2, clip_on=False)
        ax.annotate(text, (x * 1.14, y - shift), va="center", fontsize=8.5, color=INK, annotation_clip=False)


def fig_kshot(df: pd.DataFrame, styles: dict, path: Path) -> None:
    budgets = sorted(df["pct"].unique())
    metrics = [("pixel_macro_f1", "Test Macro-F1, per pixel"), ("parcel_macro_f1", "Test Macro-F1, per parcel")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
    n = len(styles)
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        ends = []
        for i, (arm, (label, colour, marker)) in enumerate(styles.items()):
            g = df[df["arm"] == arm]
            mean = g.groupby("pct")[metric].mean()
            jitter = np.exp((i - (n - 1) / 2) * 0.03)
            ax.scatter(g["pct"] * jitter, g[metric], s=16, color=colour, alpha=0.45, lw=0, zorder=2)
            ax.plot(mean.index, mean.values, color=colour, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5,
                    zorder=3, label=label)
            ends.append((mean.index[-1], mean.values[-1], label, colour))
        _budget_axis(ax, budgets)
        ax.set_xlim(min(budgets) / 1.3, max(budgets) * 1.3)
        ax.set_title(title)
        ax.set_ylim(0.25, 0.80)
        _end_labels(ax, [(max(budgets) * 1.3, y, t, c) for _, y, t, c in ends], gap=0.55 * 0.06)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.text(0.01, -0.02, "Both panels score the same stored predictions. Per parcel: the class most of the parcel's "
             "pixels were predicted as. Lines: mean over seeds; dots: single fits.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 0.88, 0.92), w_pad=9)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_gap(df: pd.DataFrame, styles: dict, path: Path) -> None:
    budgets = sorted(df["pct"].unique(), reverse=True)
    d = df.assign(gap=df["parcel_macro_f1"] - df["pixel_macro_f1"])
    lim = (min(float(d["gap"].min()) * 1.3, -0.01), 0.004)
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.3 * len(budgets) + 1.6, 0.42 * len(styles) + 1.4),
                             sharey=True)
    arms = list(styles)
    for ax, k in zip(axes, budgets, strict=True):
        for i, arm in enumerate(arms):
            label, colour, marker = styles[arm]
            v = d[(d["arm"] == arm) & (d["pct"] == k)]["gap"]
            if v.empty:
                continue
            ax.hlines(i, 0, v.mean(), color=GRID, lw=2, zorder=1)
            ax.scatter(v, [i] * len(v), s=46, color=colour, marker=marker, edgecolor=SURFACE, lw=1.5, zorder=3)
            ax.annotate(f"{v.mean():+.3f}", (v.min() + lim[0] * 0.04, i), ha="right", va="center", fontsize=8.5,
                        color=INK)
        ax.set_xlim(*lim)
        ax.set_yticks(range(len(arms)), [styles[a][0] for a in arms])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        ax.axvline(0, color=BASELINE, lw=1.2)
        ax.set_title(f"K = {k:g} %")
        ax.set_xlabel("Parcel minus pixel Macro-F1")
    fig.text(0.01, -0.03, "One marker per fit; the value is the mean. Left of zero: the parcel score is the lower. "
             "The U-Net arms have up to three seeds.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_error_split(df: pd.DataFrame, styles: dict, path: Path) -> None:
    budgets = sorted(df["pct"].unique(), reverse=True)
    g = df.groupby(["arm", "pct"])[["wrong_in_right_parcels", "wrong_in_wrong_parcels", "pixels_wrong",
                                     "pixels"]].sum()
    arms = list(styles)
    fig, axes = plt.subplots(1, len(budgets), figsize=(3.4 * len(budgets) + 1.6, 0.42 * len(arms) + 1.6),
                             sharey=True)
    removed_c, kept_c = SLOTS[0], BASELINE
    for ax, k in zip(axes, budgets, strict=True):
        for i, arm in enumerate(arms):
            if (arm, k) not in g.index:
                continue
            s = g.loc[(arm, k)]
            a = s["wrong_in_right_parcels"] / s["pixels_wrong"]
            ax.barh(i, a, color=removed_c, height=0.62, edgecolor=SURFACE, lw=2)
            ax.barh(i, 1 - a, left=a, color=kept_c, height=0.62, edgecolor=SURFACE, lw=2)
            ax.annotate(f"{a:.0%}", (a / 2, i), ha="center", va="center", fontsize=8.5, color=SURFACE)
            ax.annotate(f"{s['pixels_wrong'] / s['pixels']:.0%} wrong", (1.02, i), va="center", fontsize=8,
                        color=INK_2, annotation_clip=False)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1], ["0", "50 %", "100 %"])
        ax.set_yticks(range(len(arms)), [styles[a][0] for a in arms])
        ax.invert_yaxis()
        ax.grid(False)
        ax.set_title(f"K = {k:g} %", pad=14)
    handles = [plt.Rectangle((0, 0), 1, 1, color=removed_c), plt.Rectangle((0, 0), 1, 1, color=kept_c)]
    fig.legend(handles, ["in parcels the vote gets right: removed by voting",
                         "in parcels the vote gets wrong: the whole parcel is wrong"],
               loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.06))
    fig.text(0.01, -0.04, "Share of the misclassified labelled pixels. Right of each bar: the share of labelled "
             "pixels misclassified.", color=MUTED, fontsize=8.5)
    fig.tight_layout(w_pad=4.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_size(sizes: pd.DataFrame, styles: dict, facts: dict, path: Path) -> None:
    budgets = [k for k in (100, 5) if k in set(sizes["pct"])]
    share = sizes[sizes["arm"] == sizes["arm"].iloc[0]].groupby("bin")["parcels"].first()
    share = (share / share.sum()).reindex(SIZE_LABELS).dropna()
    fig, axes = plt.subplots(1, len(budgets), figsize=(5.6 * len(budgets) + 1.6, 4.2), sharey=True, squeeze=False)
    x = np.arange(len(share))
    for ax, k in zip(axes[0], budgets, strict=True):
        ends = []
        for arm, (label, colour, marker) in styles.items():
            m = (sizes[(sizes["arm"] == arm) & (sizes["pct"] == k)].groupby("bin")["parcel_accuracy"].mean()
                 .reindex(share.index))
            if m.isna().all():
                continue
            ax.plot(x, m.values, color=colour, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5, label=label)
            ends.append((len(x) - 1, m.values[-1], label, colour))
        ax.set_xticks(x, [f"{b}\n{s:.0%} of parcels" for b, s in share.items()], fontsize=8.5)
        ax.set_xlim(-0.3, len(x) - 0.4)
        ax.set_ylim(0, 1)
        ax.set_title(f"Parcel accuracy by parcel size, K = {k:g} %")
        ax.set_xlabel("Parcel area inside the test chips")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, -0.06, f"{facts['parcels']:,} test parcels. Accuracy is the share of parcels whose majority "
             "vote is their declared class; mean over seeds.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heat(ax, data: pd.DataFrame, lim: float) -> object:
    cmap = DIVERGING.copy()
    cmap.set_bad(SURFACE)
    im = ax.imshow(np.ma.masked_invalid(data.values), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for (i, j), v in np.ndenumerate(data.values):
        if not np.isfinite(v):
            ax.text(j, i, "-", ha="center", va="center", fontsize=7.5, color=MUTED)
            continue
        dark = abs(v) / lim > 0.6
        v = v if abs(v) >= 0.005 else 0.0
        ax.text(j, i, format(v, "+.2f"), ha="center", va="center", fontsize=7.5, color=SURFACE if dark else INK)
    ax.set_xticks(range(data.shape[1]), data.columns, rotation=30, ha="right")
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def pretty(cls: str) -> str:
    return cls.replace("_", " ")


def fig_per_class(classes: pd.DataFrame, styles: dict, path: Path, k: float = 100) -> None:
    c = classes[classes["pct"] == k].assign(gap=lambda t: (t["parcel_f1"] - t["pixel_f1"]).where(
        (t["parcel_f1"] >= 0.005) | (t["pixel_f1"] >= 0.005)))
    support = c.groupby("cls")["test_parcels"].first().sort_values(ascending=False)
    data = c.groupby(["cls", "arm"])["gap"].mean().unstack("arm").reindex(index=support.index,
                                                                          columns=list(styles))
    data.columns = [styles[a][0] for a in data.columns]
    lim = float(np.nanmax(np.abs(data.values)))
    fig, ax = plt.subplots(figsize=(0.95 * data.shape[1] + 3.6, 0.3 * len(data) + 1.8))
    im = _heat(ax, data, lim)
    ax.set_yticks(range(len(data)), [f"{pretty(cl)} ({support[cl]:,})" for cl in data.index])
    ax.set_title(f"Per-class F1, per parcel minus per pixel, K = {k:g} %")
    fig.colorbar(im, ax=ax, shrink=0.6, label="Difference in test F1")
    fig.text(0.01, 0.0, "Classes by test parcels, in brackets. Mean over seeds. Dash: F1 zero both ways.",
             color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------- tables

def _cell(g: pd.Series, fmt: str = ".3f") -> str:
    if g.empty:
        return ""
    return f"{g.mean():{fmt}} +- {g.std():{fmt}}" if len(g) > 1 else f"{g.mean():{fmt}}"


def tables(df: pd.DataFrame, sizes: pd.DataFrame, styles: dict, facts: dict) -> str:
    budgets = sorted(df["pct"].unique(), reverse=True)
    head = "| Arm | " + " | ".join(f"K = {k:g} %" for k in budgets) + " |\n|---|" + "---|" * len(budgets) + "\n"
    sel = (lambda arm, k: df[(df["arm"] == arm) & (df["pct"] == k)])
    out = []

    def grid(title: str, fn) -> None:
        rows = [f"| {label} | " + " | ".join(fn(sel(arm, k)) for k in budgets) + " |"
                for arm, (label, _, _) in styles.items()]
        out.append(f"**{title}**\n\n{head}" + "\n".join(rows))

    grid("Test Macro-F1 per pixel / per parcel", lambda g: "" if g.empty else
         f"{g['pixel_macro_f1'].mean():.3f} / **{g['parcel_macro_f1'].mean():.3f}**")
    grid("Parcel minus pixel Macro-F1", lambda g: _cell(g["parcel_macro_f1"] - g["pixel_macro_f1"], "+.3f"))
    grid("Parcel overall accuracy / kappa", lambda g: "" if g.empty else
         f"{g['parcel_accuracy'].mean():.3f} / {g['parcel_kappa'].mean():.3f}")
    grid("Pixel overall accuracy / kappa", lambda g: "" if g.empty else
         f"{g['pixel_accuracy'].mean():.3f} / {g['pixel_kappa'].mean():.3f}")
    grid("Misclassified pixels removed by the vote / correct pixels lost to it", lambda g: "" if g.empty else
         f"{g['wrong_in_right_parcels'].sum() / g['pixels_wrong'].sum():.0%} / "
         f"{g['right_in_wrong_parcels'].sum() / (g['pixels'].sum() - g['pixels_wrong'].sum()):.0%}")

    rows = []
    for arm, (label, _, _) in styles.items():
        for k in budgets:
            g = sel(arm, k)
            if g.empty:
                continue
            base = g["parcel_macro_f1"].mean()
            rows.append(f"| {label} | {k:g} % | {base:.3f} | "
                        + " | ".join(f"{g[c].mean() - base:+.3f}" for c in
                                     ("interior_macro_f1", "no_cut_macro_f1", "min_size_macro_f1")) + " |")
    out.append("**Sensitivity of the parcel Macro-F1**: the main rule, and the change when only interior pixels vote, "
               f"when the {int(df['n_cut'].iloc[0]):,} parcels reaching outside the test chips are left out, and when "
               f"the {int(df['n_small'].iloc[0]):,} parcels under {SMALL} pixels are left out\n\n"
               "| Arm | K | All parcels, all pixels | Interior pixels vote | Without cut parcels | "
               f"Parcels of {SMALL} px or more |\n|---|---|---|---|---|---|\n" + "\n".join(rows))

    rows = []
    for k in budgets:
        m = df[df["pct"] == k].groupby("arm")[["pixel_macro_f1", "parcel_macro_f1"]].mean()
        order = (lambda col: " > ".join(styles[a][0] for a in m[col].sort_values(ascending=False).index))
        rows.append(f"| {k:g} % | {order('pixel_macro_f1')} | {order('parcel_macro_f1')} |")
    out.append("**Ranking of the arms**\n\n| K | Per pixel | Per parcel |\n|---|---|---|\n" + "\n".join(rows))

    s = sizes[(sizes["arm"] == sizes["arm"].iloc[0]) & (sizes["pct"] == sizes["pct"].iloc[0])
              & (sizes["seed"] == sizes["seed"].iloc[0])].set_index("bin")["parcels"].reindex(SIZE_LABELS)
    out.append("**Run**\n\n"
               f"* Chip set {facts['chips']}, split {facts['split']}: {facts['test_chips']:,} test chips, "
               f"{facts['parcels']:,} test parcels with labelled pixels, of which {int(df['n_cut'].iloc[0]):,} reach "
               f"outside the test chips and {int(df['n_small'].iloc[0]):,} have fewer than {SMALL} pixels.\n"
               "* Parcels by area inside the test chips: "
               + ", ".join(f"{b} {int(v):,}" for b, v in s.items()) + ".\n"
               f"* {len(df)} fits scored. Largest difference between the pixel Macro-F1 recomputed from the stored "
               "predictions and the one the fit logged: "
               f"{(df['pixel_macro_f1'] - df['pixel_macro_f1_logged']).abs().max():.4f}.\n"
               + "".join(_stale_line(r, styles) for r in facts["stale"]))
    return "\n\n".join(out) + "\n"


def _stale_line(r: dict, styles: dict) -> str:
    label = styles[r["arm"]][0] if r["arm"] in styles else r["arm"]
    return (f"* Left out, predicted with another checkpoint than the one tested: {label}, K = {r['pct']:g} %, "
            f"seed {r['seed']} (pixel Macro-F1 {r['pixel_macro_f1']:.4f} from the stored predictions, "
            f"{r['pixel_macro_f1_logged']:.4f} logged).\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chips", required=True, help="the chip set's name, e.g. EE_2021_rev")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    df, classes, sizes, facts = load(args.chips)
    facts["parcels"] = int(df["n_parcels"].iloc[0])
    store = REPO / "results" / NAME / args.chips
    store.mkdir(parents=True, exist_ok=True)
    df.to_csv(store / "cells.csv", index=False)
    classes.to_csv(store / "per_class.csv", index=False)
    sizes.to_csv(store / "size_bins.csv", index=False)

    styles = arm_styles(sorted(df["arm"].unique()))
    args.out.mkdir(parents=True, exist_ok=True)
    fig_kshot(df, styles, args.out / "kshot_curves.png")
    fig_gap(df, styles, args.out / "parcel_minus_pixel.png")
    fig_error_split(df, styles, args.out / "error_split.png")
    fig_size(sizes, styles, facts, args.out / "accuracy_by_size.png")
    fig_per_class(classes, styles, args.out / "per_class_gap.png")
    text = tables(df, sizes, styles, facts)
    (args.out / "tables.md").write_text(text)
    print(text)
    print(f"figures and tables in {args.out.relative_to(REPO)}, scores in {store.relative_to(REPO)}")


if __name__ == "__main__":
    main()
