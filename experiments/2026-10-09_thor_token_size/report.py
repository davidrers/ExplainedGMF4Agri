"""Figures and tables of an experiment's results, for its REPORT.md.

Reads ``results/<name>/<chip set>/`` as the K-shot runner writes it, on the hub or pulled back from the cluster,
refreshes its ``summary.csv``, and writes into ``figures/`` beside this file:

* ``kshot_curves.png``: test Macro-F1 and mIoU against K per arm, the mean over draws and seeds and every fit;
* ``paired_differences.png``: the difference in test Macro-F1 between two arms fit by fit, on the same draw and
  seed, for every pair of ``PAIRS``;
* ``validation_curves.png``: validation loss and Macro-F1 per epoch, per arm and K, with the epochs the checkpoint
  criterion kept;
* ``per_class_f1.png``: per-class test F1, as the differences of ``PAIRS`` or, without pairs, per arm;
* ``cost.png``, added for this study: encoding time, feature cache and fit time per arm, and the cache of all
  Estonia against a node's local disk;
* ``tables.md``: the tables of REPORT.md in Markdown, also printed.

Copy it with the template, set ``ARMS`` and ``PAIRS`` below, and run it from the repository root:

    poetry run python experiments/<folder>/report.py --chips EE_2021_mini
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(REPO / "src")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

#: Arm -> (short label, categorical slot 1 to 8). The legend follows this order and the colour follows the slot,
#: so an arm keeps its colour in every figure. Arms in the results but not here take the next free slots.
ARMS: dict[str, tuple[str, int]] = {
    "thor_v1_large": ("160 m tokens", 1),            # the core arm
    "thor_v1_large_80m": ("80 m tokens", 2),
    "thor_v1_large_40m": ("40 m tokens", 3),
}
#: Differences shown, as (label, arm a, arm b) for a minus b. The two arms must share draws and seeds.
PAIRS: list[tuple[str, str, str]] = [
    ("80 m minus 160 m", "thor_v1_large_80m", "thor_v1_large"),
    ("40 m minus 160 m", "thor_v1_large_40m", "thor_v1_large"),
    ("40 m minus 80 m", "thor_v1_large_40m", "thor_v1_large_80m"),
]

#: Chip encodings of the full Estonian split, 4,898 training chips in 8 D4 variants plus 769 validation and 1,475
#: test chips, and a cluster node's local NVMe in TB: the scale the cost figure projects to.
FULL_EE_ENCODINGS, NODE_LOCAL_TB = 4_898 * 8 + 769 + 1_475, 6.4

# The reference palette of the dataviz method: categorical slots in their validated order (the first three pass
# every check all-pairs on the light surface), chart chrome, and a blue to red diverging ramp with a grey midpoint.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "h", "P", "X"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("gain", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("level", ["#cde2fb", "#2a78d6", "#0d366b"])
METRICS = [("macro_f1", "Test Macro-F1"), ("miou", "Test mIoU")]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "font.size": 9.5, "legend.frameon": False, "legend.labelcolor": INK,
})


# ---------------------------------------------------------------------------------------------------------- data

def load(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """The summary (one row per fit), per-class test F1 (long), validation curves (long) and run facts."""
    from gfm4agri.benchmark.evaluation import f1_from_iou
    from gfm4agri.pipeline.runner import summarise

    df = summarise(out)
    classes, curves, support = [], [], {}
    facts = {"split": None, "chips": None, "commits": set(), "machines": set(), "cache": {}}
    for path in sorted(out.glob("*/P*_draw*_seed*/results.json")):
        r = json.loads(path.read_text())
        key = {"arm": r["run"], "pct": r["label_budget"]["pct"], "draw": r["label_budget"]["draw_seed"],
               "seed": r["seed"]}
        for c in r["class_names"]:
            iou = r["test_metrics"].get(f"test/IoU_{c}")
            if iou is not None:
                classes.append({**key, "cls": c, "f1": f1_from_iou(iou)})
        if key["pct"] == df["pct"].max():
            support = {c: v["drawn"] for c, v in (r["label_budget"].get("support") or {}).items()}
        logs = sorted(path.parent.glob("logs/version_*/metrics.csv"))
        if logs:
            m = pd.read_csv(logs[-1])
            for col, name in (("val/loss", "val_loss"), ("val/F1_Score", "val_f1")):
                if col in m:
                    v = m.dropna(subset=[col]).groupby("epoch")[col].last()
                    curves += [{**key, "metric": name, "epoch": int(e), "value": float(x)} for e, x in v.items()]
        sp = r.get("split_protocol", {})
        facts["split"], facts["chips"] = sp.get("split"), sp.get("chips")
        fc = r.get("feature_cache") or {}
        if fc.get("bytes"):
            facts["cache"][key["arm"]] = {"bytes": fc["bytes"], "encode_s": fc["encode_seconds"],
                                          "encodings": fc["chip_encodings"], "grid": fc["feature_size"][0],
                                          "peak_gpu_gb": max(r.get("peak_gpu_gb") or 0, facts["cache"].get(
                                              key["arm"], {}).get("peak_gpu_gb", 0))}
        prov = r.get("provenance", {})
        facts["commits"].add((prov.get("git_commit") or "")[:8])
        facts["machines"].add(prov.get("machine"))
    facts["support"] = support
    return df, pd.DataFrame(classes), pd.DataFrame(curves), facts


def arm_styles(arms: list[str]) -> dict[str, tuple[str, str, str]]:
    """Arm -> (label, colour, marker), in the legend order: ARMS first, then the rest."""
    order = [a for a in ARMS if a in arms] + sorted(set(arms) - set(ARMS))
    used = {slot for a, (_, slot) in ARMS.items() if a in arms}
    free = iter(s for s in range(1, len(SLOTS) + 1) if s not in used)
    out = {}
    for a in order:
        label, slot = ARMS.get(a, (a, None))
        slot = slot or next(free)
        out[a] = (label, SLOTS[slot - 1], MARKERS[slot - 1])
    return out


def paired(df: pd.DataFrame, a: str, b: str, metric: str) -> pd.Series:
    """``metric`` of ``a`` minus ``b``, fit by fit on the same budget, draw and seed."""
    piv = df.pivot_table(index=["pct", "draw", "seed"], columns="arm", values=metric)
    return (piv[a] - piv[b]).dropna() if a in piv and b in piv else pd.Series(dtype=float)


def pretty(cls: str) -> str:
    return cls.replace("_", " ")


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
    for (x, _, text, colour), y in zip(items, ys):
        ax.plot([x * 1.03, x * 1.12], [y - shift, y - shift], color=colour, lw=2, clip_on=False)
        ax.annotate(text, (x * 1.14, y - shift), va="center", fontsize=9, color=INK, annotation_clip=False)


def fig_kshot(df: pd.DataFrame, styles: dict, path: Path) -> None:
    budgets = sorted(df["pct"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    n = len(styles)
    for ax, (metric, title) in zip(axes, METRICS):
        ends = []
        for i, (arm, (label, colour, marker)) in enumerate(styles.items()):
            g = df[df["arm"] == arm]
            mean = g.groupby("pct")[metric].mean()
            jitter = np.exp((i - (n - 1) / 2) * 0.035)
            ax.scatter(g["pct"] * jitter, g[metric], s=16, color=colour, alpha=0.45, lw=0, zorder=2)
            ax.plot(mean.index, mean.values, color=colour, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5,
                    zorder=3, label=label)
            ends.append((mean.index[-1], mean.values[-1], label, colour))
        _budget_axis(ax, budgets)
        ax.set_xlim(min(budgets) / 1.3, max(budgets) * 1.3)
        ax.set_title(title)
        lo, hi = ax.get_ylim()
        _end_labels(ax, [(max(budgets) * 1.3, y, t, c) for _, y, t, c in ends], gap=(hi - lo) * 0.07)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=min(n, 4), bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.02, "Lines: mean over draws and seeds. Dots: single fits.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 0.9, 0.93))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_paired(df: pd.DataFrame, path: Path, metric: str = "macro_f1") -> None:
    budgets = sorted(df["pct"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(PAIRS), figsize=(3.4 * len(PAIRS) + 0.8, 3.9), sharey=True, squeeze=False)
    rng = np.random.default_rng(0)
    for ax, (label, a, b) in zip(axes[0], PAIRS):
        d = paired(df, a, b, metric)
        for i, k in enumerate(budgets):
            v = d.xs(k, level="pct") if k in d.index.get_level_values("pct") else pd.Series(dtype=float)
            if v.empty:
                continue
            x = i + rng.uniform(-0.12, 0.12, len(v))
            ax.scatter(x, v, s=40, color=INK_2, alpha=0.75, lw=1.5, edgecolor=SURFACE, zorder=3)
            ax.hlines(v.mean(), i - 0.28, i + 0.28, color=INK, lw=2, zorder=4)
            ax.annotate(f"{v.mean():+.3f}", (i + 0.3, v.mean()), va="center", fontsize=9, color=INK)
        ax.axhline(0, color=BASELINE, lw=1.2, zorder=1)
        ax.set_xticks(range(len(budgets)), [f"K = {k:g} %" for k in budgets])
        ax.set_xlim(-0.6, len(budgets) - 0.2)
        ax.grid(axis="x", visible=False)
        ax.set_title(label)
    axes[0][0].set_ylabel(f"Difference in {dict(METRICS)[metric]}")
    fig.text(0.01, -0.03, "Dots: one draw and seed, both arms fit on the same parcels. Bar and value: the mean.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_validation(df: pd.DataFrame, curves: pd.DataFrame, styles: dict, path: Path) -> None:
    budgets = sorted(df["pct"].unique(), reverse=True)
    fig, axes = plt.subplots(len(budgets), 2, figsize=(10, 2.6 * len(budgets) + 0.6), sharex=True, squeeze=False)
    for row, k in zip(axes, budgets):
        for ax, (metric, title) in zip(row, (("val_loss", "Validation loss"), ("val_f1", "Validation Macro-F1"))):
            for arm, (label, colour, marker) in styles.items():
                c = curves[(curves["arm"] == arm) & (curves["pct"] == k) & (curves["metric"] == metric)]
                if c.empty:
                    continue
                mean = c.groupby("epoch")["value"].mean()
                ax.plot(mean.index, mean.values, color=colour, lw=2, label=label, zorder=2)
                kept = df[(df["arm"] == arm) & (df["pct"] == k)]["ckpt_epoch"].dropna().astype(int)
                kept = kept[kept.isin(mean.index)]
                ax.scatter(kept, mean.loc[kept], s=46, marker=marker, color=colour, edgecolor=SURFACE, lw=1.5,
                           zorder=3)
            ax.set_title(f"{title}, K = {k:g} %")
    for ax in axes[-1]:
        ax.set_xlabel("Epoch")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=min(len(styles), 4),
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, -0.01, "Lines: mean over draws and seeds. Markers: the epoch of lowest validation loss of each "
             "fit, the checkpoint tested.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heat(ax, data: pd.DataFrame, cmap, vmin: float, vmax: float, fmt: str):
    im = ax.imshow(data.values, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(data.values):
        if np.isfinite(v):
            dark = abs((v - (vmin + vmax) / 2) / ((vmax - vmin) / 2)) > 0.6
            ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=7.5, color=SURFACE if dark else INK)
    ax.set_xticks(range(data.shape[1]), data.columns)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_per_class(classes: pd.DataFrame, styles: dict, support: dict, path: Path) -> None:
    order = sorted(classes["cls"].unique(), key=lambda c: -support.get(c, 0))
    budgets = sorted(classes["pct"].unique(), reverse=True)
    mean = classes.groupby(["arm", "cls", "pct"])["f1"].mean()
    if PAIRS:
        panels = [(label, (mean[a] - mean[b]).unstack("pct")) for label, a, b in PAIRS]
        lim = max(np.nanmax(np.abs(p.values)) for _, p in panels)
        cmap, vmin, vmax, fmt, cbar = DIVERGING, -lim, lim, "+.2f", "Difference in test F1"
    else:
        panels = [(styles[a][0], mean[a].unstack("pct")) for a in styles if a in mean.index.get_level_values(0)]
        cmap, vmin, vmax, fmt, cbar = SEQUENTIAL, 0.0, 1.0, ".2f", "Test F1"
    fig, axes = plt.subplots(1, len(panels), figsize=(2.3 * len(panels) + 3.4, 0.3 * len(order) + 1.6),
                             sharey=True, squeeze=False)
    for ax, (label, p) in zip(axes[0], panels):
        p = p.reindex(index=order, columns=budgets)
        p.columns = [f"{k:g} %" for k in budgets]
        im = _heat(ax, p, cmap, vmin, vmax, fmt)
        ax.set_title(label)
    axes[0][0].set_yticks(range(len(order)),
                          [f"{pretty(c)} ({support[c]:,})" if c in support else pretty(c) for c in order])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.6, label=cbar)
    fig.text(0.01, 0.0, "Mean over draws and seeds. Classes by training parcels at the largest K, in brackets; "
             "columns are K.", color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_cost(df: pd.DataFrame, styles: dict, cache: dict, path: Path) -> None:
    arms = [a for a in styles if a in cache]
    top = df["pct"].max()
    fit_min = df[df["pct"] == top].groupby("arm")["fit_h"].mean() * 60
    panels = [("Encoding the chip set, hours", [cache[a]["encode_s"] / 3600 for a in arms], "{:.1f} h"),
              ("Feature cache, TB", [cache[a]["bytes"] / 1e12 for a in arms], "{:.2f} TB"),
              (f"Fit of one cell at K = {top:g} %, minutes", [fit_min.get(a, np.nan) for a in arms], "{:.0f} min")]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6))
    x = np.arange(len(arms))
    for ax, (title, vals, fmt) in zip(axes, panels):
        ax.bar(x, vals, width=0.6, color=[styles[a][1] for a in arms], zorder=2)
        for i, v in enumerate(vals):
            ax.annotate(fmt.format(v), (i, v), xytext=(0, 3), textcoords="offset points", ha="center",
                        fontsize=9, color=INK)
        ax.set_xticks(x, [styles[a][0] for a in arms])
        ax.set_title(title)
        ax.grid(axis="x", visible=False)
        ax.margins(y=0.15)
    ax = axes[1]
    full = [cache[a]["bytes"] / cache[a]["encodings"] * FULL_EE_ENCODINGS / 1e12 for a in arms]
    ax.bar(x, full, width=0.6, fill=False, edgecolor=[styles[a][1] for a in arms], lw=1.5, ls="--", zorder=1)
    for i, v in enumerate(full):
        ax.annotate(f"{v:.1f} TB", (i, v), xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9,
                    color=INK_2)
    ax.axhline(NODE_LOCAL_TB, color=INK_2, lw=1, ls=":", zorder=3)
    ax.annotate(f"node disk, {NODE_LOCAL_TB:g} TB", (-0.42, NODE_LOCAL_TB), xytext=(0, -11),
                textcoords="offset points", fontsize=8, color=INK_2)
    ax.set_title("Feature cache, TB: this sample and all Estonia")
    ax.set_yscale("log")
    ax.set_ylim(0.05, max(full) * 2.5)
    ax.grid(axis="y", which="minor", visible=False)
    fig.text(0.01, -0.03, f"Filled: this chip set ({cache[arms[0]]['encodings']:,} chip encodings), measured on "
             f"the cluster. Dashed: all Estonia ({FULL_EE_ENCODINGS:,} chip encodings), scaled. Fit: mean of "
             "the seeds.", color=MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------- tables

def tables(df: pd.DataFrame, styles: dict, facts: dict) -> str:
    budgets = sorted(df["pct"].unique(), reverse=True)
    head = "| Arm | " + " | ".join(f"K = {k:g} %" for k in budgets) + " |\n|---|" + "---|" * len(budgets) + "\n"
    out = []
    for metric, title in METRICS:
        rows = []
        for arm, (label, _, _) in styles.items():
            g = df[df["arm"] == arm].groupby("pct")[metric]
            m, s, n = g.mean(), g.std(), g.count()
            rows.append(f"| {label} | " + " | ".join(
                f"{m[k]:.3f} +- {s[k]:.3f} ({n[k]})" if n.get(k, 0) > 1 else
                (f"{m[k]:.3f}" if k in m else "") for k in budgets) + " |")
        out.append(f"**{title}**, mean +- standard deviation (fits)\n\n{head}" + "\n".join(rows))
    if PAIRS:
        rows = []
        for label, a, b in PAIRS:
            d = paired(df, a, b, "macro_f1")
            cells = []
            for k in budgets:
                v = d.xs(k, level="pct") if k in d.index.get_level_values("pct") else pd.Series(dtype=float)
                cells.append(f"{v.mean():+.3f} +- {v.std():.3f} ({(v > 0).sum()}/{len(v)})" if len(v) else "")
            rows.append(f"| {label} | " + " | ".join(cells) + " |")
        out.append("**Paired difference in test Macro-F1**, mean +- standard deviation (fits won by the first)\n\n"
                   + head.replace("| Arm |", "| Pair |") + "\n".join(rows))
    rows = []
    for arm, (label, _, _) in styles.items():
        cells = []
        for k in budgets:
            g = df[(df["arm"] == arm) & (df["pct"] == k)].sort_values(["draw", "seed"])
            cells.append(", ".join(str(int(e)) for e in g["ckpt_epoch"].dropna()) + " / "
                         + ", ".join(str(int(e)) for e in g["best_val_f1_epoch"].dropna()))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    out.append("**Epochs**, per fit: the checkpoint (lowest validation loss) / the highest validation Macro-F1\n\n"
               + head + "\n".join(rows))
    if facts.get("cache"):
        top = df["pct"].max()
        rows = []
        for arm, (label, _, _) in styles.items():
            c = facts["cache"].get(arm)
            if not c:
                continue
            fit = df[(df["arm"] == arm) & (df["pct"] == top)]["fit_h"].mean() * 60
            rows.append(f"| {label} | {c['grid']} x {c['grid']} | {c['bytes'] / c['encodings'] / 1e6:,.0f} MB | "
                        f"{c['encode_s'] / 60:,.0f} min | {c['bytes'] / 1e12:.2f} TB | "
                        f"{c['bytes'] / c['encodings'] * FULL_EE_ENCODINGS / 1e12:.1f} TB | {fit:.0f} min | "
                        f"{c['peak_gpu_gb']:.1f} GB |")
        out.append("**Cost**, on the cluster\n\n| Arm | Grid | Per chip encoding | Encoding | Cache | Cache, all "
                   f"Estonia | Fit at K = {top:g} % | Peak GPU, fit |\n|---|---|---|---|---|---|---|---|\n"
                   + "\n".join(rows))
    budget = df.groupby("pct")[["train_chips", "parcels"]].first().sort_index(ascending=False)
    chips = facts.get("chips") or {}
    out.append("**Run**\n\n"
               f"* Split {facts.get('split')}: {chips.get('training')} training, {chips.get('validation')} "
               f"validation, {chips.get('test')} test chips.\n"
               + "".join(f"* K = {k:g} %: {int(r.train_chips)} training chips, {int(r.parcels):,} parcels "
                         "(draw 0).\n" for k, r in budget.iterrows())
               + f"* {len(df)} fits, {df['fit_h'].sum():.1f} GPU hours of fitting; machines "
               f"{', '.join(sorted(m for m in facts['machines'] if m))}; commits "
               f"{', '.join(sorted(c for c in facts['commits'] if c))}.")
    return "\n\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chips", required=True, help="the chip set's name, e.g. EE_2021_mini")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    args = ap.parse_args()

    from gfm4agri.pipeline.config import load_experiment

    name = load_experiment(HERE / "experiment.yaml")["name"]
    results = REPO / "results" / name / args.chips
    df, classes, curves, facts = load(results)
    if df.empty:
        raise SystemExit(f"no finished fits in {results}")
    styles = arm_styles(sorted(df["arm"].unique()))
    args.out.mkdir(parents=True, exist_ok=True)
    fig_kshot(df, styles, args.out / "kshot_curves.png")
    if PAIRS:
        fig_paired(df, args.out / "paired_differences.png")
    if not curves.empty:
        fig_validation(df, curves, styles, args.out / "validation_curves.png")
    if not classes.empty:
        fig_per_class(classes, styles, facts["support"], args.out / "per_class_f1.png")
    if facts["cache"]:
        fig_cost(df, styles, facts["cache"], args.out / "cost.png")
    text = tables(df, styles, facts)
    (args.out / "tables.md").write_text(text)
    print(text)
    print(f"figures and tables in {args.out.relative_to(REPO)}")


if __name__ == "__main__":
    main()
