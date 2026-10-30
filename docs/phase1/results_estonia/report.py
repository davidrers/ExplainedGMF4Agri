"""Figures and tables of the first Estonian results report (REPORT.md beside this file).

Reads, without writing there, the main workflow's frozen arms (``results/kshot/EE_2021_rev/``) and the fine-tuned
arms of ``experiments/2026-10-15_kshot_finetune`` (``results/2026-10-15_kshot_finetune/EE_2021_rev/``), both at
45 epochs, and writes into ``figures/`` beside this file:

* ``label_curves.png``: test Macro-F1 against the label budget K, per pixel and per parcel, frozen and fine-tuned;
* ``finetune_gain.png``: fine-tuned minus frozen, per model and K, under three readouts;
* ``per_class.png``: per-class test F1 per pixel at K = 100 and 5 %;
* ``literature.png``: parcel Macro-F1 against labelled parcels per class, ours beside the published Estonian
  numbers;
* ``tables.md``: every number the report quotes.

    poetry run python docs/phase1/results_estonia/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
FROZEN_DIR = REPO / "results/kshot/EE_2021_rev"
FT_DIR = REPO / "results/2026-10-15_kshot_finetune/EE_2021_rev"
OUT = HERE / "figures"

#: Model label, frozen arm, fine-tuned arm (None: no encoder to train), colour slot, marker.
MODELS = [
    ("TerraMind v1 L", "terramind_v1_large_s2s1", "terramind_v1_large_s2s1_ft", 0, "o"),
    ("THOR v1 L", "thor_v1_large_s2s1", "thor_v1_large_s2s1_ft", 1, "s"),
    ("Prithvi-EO-2.0 600M", "prithvi_eo_v2_600_tl", "prithvi_eo_v2_600_tl_ft", 2, "D"),
    ("TESSERA v2 M", "tessera_v2_medium", "tessera_v2_medium_ft", 3, "^"),
    ("AlphaEarth v1", "alphaearth_v1", None, 4, "v"),
]
BASELINE = ("U-Net from scratch", "unet_r18_scratch_s2s1")
KS = [100, 20, 5]
PARCELS = {100: 86909, 20: 17390, 5: 4354}

# The reference palette of the dataviz method (validated: five slots pass CVD and normal-vision separation on the
# light surface; contrast below 3:1 for three slots is relieved by a marker per model and the tables).
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE_RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SEQUENTIAL = LinearSegmentedColormap.from_list("level", ["#f0efec", "#cde2fb", "#2a78d6", "#0d366b"])
K_SHADES = {100: "#0d366b", 20: "#2a78d6", 5: "#9ec5f4"}

#: The only published Estonian parcel Macro-F1 (verified against the paper, 30 October 2026): DirPA, Reuss, Gikalo
#: and Körner 2026, arXiv 2603.12905v2, Table 6, EuroCropsML 2.0, rows trained with plain cross-entropy (CE and
#: CE PRESTO) as ours are, mean of five runs, at k labelled parcels per class at most.
SHOTS = [1, 5, 10, 20, 100]
LITERATURE = [
    ("Transformer from scratch", "127 crop types", [0.049, 0.094, 0.115, 0.117, 0.174]),
    ("Presto, released weights", "127 crop types", [0.011, 0.020, 0.026, 0.038, 0.064]),
    ("Transformer from scratch", "coarse groups", [0.209, 0.233, 0.261, 0.272, 0.307]),
    ("Presto, released weights", "coarse groups", [0.152, 0.195, 0.203, 0.213, 0.255]),
]


def f1_from_iou(iou: float) -> float:
    return 2 * iou / (1 + iou)


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per fit with every summary metric and the best validation Macro-F1, and per-class pixel F1."""
    rows, classes = [], []
    for root, tuned in ((FROZEN_DIR, False), (FT_DIR, True)):
        summary = pd.read_csv(root / "summary.csv")
        for _, s in summary.iterrows():
            cell = root / s["arm"] / f"P{int(s['pct'])}_draw{int(s['draw'])}_seed{int(s['seed'])}"
            logs = sorted(cell.glob("logs/version_*/metrics.csv"))
            best_val = np.nan
            if logs:
                m = pd.read_csv(logs[-1])
                if "val/F1_Score" in m:
                    best_val = m.dropna(subset=["val/F1_Score"]).groupby("epoch")["val/F1_Score"].last().max()
            rows.append({**s.to_dict(), "fine_tuned": tuned, "best_val_f1": best_val})
            r = json.loads((cell / "results.json").read_text())
            for c in r["class_names"]:
                iou = r["test_metrics"].get(f"test/IoU_{c}")
                if iou is not None:
                    classes.append({"arm": s["arm"], "pct": int(s["pct"]), "cls": c, "f1": f1_from_iou(iou),
                                    "parcel_f1": r["parcel_metrics"]["per_class_f1"].get(c),
                                    "train_parcels": r["label_budget"]["support"].get(c, {}).get("available"),
                                    "labelled_parcels": r["label_budget"]["support"].get(c, {}).get("drawn")})
    return pd.DataFrame(rows), pd.DataFrame(classes)


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def value(df: pd.DataFrame, arm: str, pct: int, col: str) -> float:
    v = df[(df["arm"] == arm) & (df["pct"] == pct)][col]
    return float(v.iloc[0]) if len(v) else np.nan


def label_curves(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), sharey=True, facecolor=SURFACE)
    for ax, (col, title) in zip(axes, (("macro_f1", "Per pixel"), ("parcel_macro_f1", "Per parcel"))):
        style(ax)
        for label, frozen, tuned, slot, marker in MODELS:
            ax.plot(KS, [value(df, frozen, k, col) for k in KS], color=SLOTS[slot], marker=marker, lw=2, ms=7,
                    markeredgecolor=SURFACE, markeredgewidth=1)
            if tuned:
                ax.plot(KS, [value(df, tuned, k, col) for k in KS], color=SLOTS[slot], marker=marker, lw=2, ms=7,
                        ls=(0, (4, 2)), markerfacecolor=SURFACE, markeredgewidth=1.5)
        ax.plot(KS, [value(df, BASELINE[1], k, col) for k in KS], color=INK, marker="X", lw=2, ms=8,
                markeredgecolor=SURFACE)
        ax.set_xscale("log")
        ax.set_xticks(KS, [f"{k} %\n{PARCELS[k]:,} parcels" for k in KS])
        ax.minorticks_off()
        ax.set_xlim(4, 125)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
        ax.set_xlabel("Label budget K (share of the training parcels of each class)", fontsize=9, color=INK_2)
    axes[0].set_ylabel("Test Macro-F1, 20 classes", fontsize=9, color=INK_2)
    handles = [Line2D([], [], color=SLOTS[s], marker=m, lw=2, label=l) for l, _, _, s, m in MODELS]
    handles += [Line2D([], [], color=INK, marker="X", lw=2, label=BASELINE[0]),
                Line2D([], [], color=MUTED, lw=2, marker="o", label="frozen encoder"),
                Line2D([], [], color=MUTED, lw=2, ls=(0, (4, 2)), marker="o", markerfacecolor=SURFACE,
                       label="fine-tuned encoder")]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, fontsize=9, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout(rect=(0, 0, 1, 0.88))
    fig.savefig(OUT / "label_curves.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def gains(df: pd.DataFrame) -> pd.DataFrame:
    out = []
    for label, frozen, tuned, _, _ in MODELS:
        if not tuned:
            continue
        for k in KS:
            out.append({"model": label, "pct": k, **{c: value(df, tuned, k, c) - value(df, frozen, k, c)
                                                    for c in ("macro_f1", "parcel_macro_f1", "best_val_f1")}})
    return pd.DataFrame(out)


def finetune_gain(g: pd.DataFrame) -> None:
    readouts = [("macro_f1", "Test, per pixel"), ("parcel_macro_f1", "Test, per parcel"),
                ("best_val_f1", "Best validation, per pixel")]
    models = list(dict.fromkeys(g["model"]))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2), sharey=True, facecolor=SURFACE)
    x = np.arange(len(models))
    w = 0.26
    for ax, (col, title) in zip(axes, readouts):
        style(ax)
        ax.grid(False, axis="x")
        for i, k in enumerate(KS):
            vals = [g[(g["model"] == m) & (g["pct"] == k)][col].iloc[0] for m in models]
            ax.bar(x + (i - 1) * w, vals, w - 0.03, color=K_SHADES[k], label=f"K = {k} %")
        ax.axhline(0, color=INK_2, lw=1)
        ax.set_xticks(x, [m.replace(" ", "\n", 1) for m in models], fontsize=8.5)
        ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold")
    axes[0].set_ylabel("Fine-tuned minus frozen, Macro-F1", fontsize=9, color=INK_2)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left")
    fig.tight_layout()
    fig.savefig(OUT / "finetune_gain.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def per_class(classes: pd.DataFrame) -> None:
    cols = [("TerraMind", "terramind_v1_large_s2s1"), ("THOR", "thor_v1_large_s2s1"),
            ("Prithvi", "prithvi_eo_v2_600_tl"), ("TESSERA", "tessera_v2_medium"), ("AlphaEarth", "alphaearth_v1"),
            ("U-Net", BASELINE[1]), ("TESSERA\nfine-tuned", "tessera_v2_medium_ft")]
    order = (classes[(classes["arm"] == "tessera_v2_medium") & (classes["pct"] == 100)]
             .sort_values("train_parcels", ascending=False))
    names = list(order["cls"])
    counts = dict(zip(order["cls"], order["train_parcels"]))
    fig, axes = plt.subplots(1, 2, figsize=(12, 8.2), sharey=True, facecolor=SURFACE)
    for ax, k in zip(axes, (100, 5)):
        m = np.array([[classes[(classes["arm"] == a) & (classes["pct"] == k) & (classes["cls"] == c)]["f1"].iloc[0]
                       for _, a in cols] for c in names])
        ax.imshow(m, cmap=SEQUENTIAL, vmin=0, vmax=1, aspect="auto")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7.5,
                        color=SURFACE if m[i, j] > 0.6 else INK)
        ax.set_xticks(range(len(cols)), [l for l, _ in cols], fontsize=8.5)
        ax.xaxis.tick_top()
        ax.set_title(f"Per-pixel F1, K = {k} %", loc="left", fontsize=11, color=INK, fontweight="bold", pad=36)
        for side in ax.spines.values():
            side.set_visible(False)
        ax.tick_params(length=0, colors=INK_2)
    axes[0].set_yticks(range(len(names)), [f"{c.replace('_', ' ')} ({counts[c]:,})" for c in names], fontsize=8.5)
    fig.tight_layout()
    fig.savefig(OUT / "per_class.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def labelled_per_class(classes: pd.DataFrame) -> pd.DataFrame:
    """Labelled training parcels per class at each K: the draw is the same for every arm, so the baseline's is read."""
    c = classes[classes["arm"] == BASELINE[1]]
    return c.groupby("pct")["labelled_parcels"].agg(["min", "median", "max", "sum"])


def literature(df: pd.DataFrame, classes: pd.DataFrame) -> None:
    per_class = labelled_per_class(classes)["median"]
    x = [per_class[k] for k in KS]
    fig, ax = plt.subplots(figsize=(10, 5.8), facecolor=SURFACE)
    style(ax)
    published = []
    for model, scheme, vals in LITERATURE:
        coarse, presto = scheme.startswith("coarse"), model.startswith("Presto")
        colour, marker = (MUTED, "s") if coarse else (INK_2, "o")
        ls = (0, (4, 2)) if presto else "-"
        ax.plot(SHOTS, vals, color=colour, marker=marker, ls=ls, lw=1.8, ms=6, markeredgecolor=SURFACE)
        published.append(Line2D([], [], color=colour, marker=marker, ls=ls, lw=1.8, label=f"{model}, {scheme}"))
    ours = []
    for label, frozen, _, slot, marker in MODELS:
        ax.plot(x, [value(df, frozen, k, "parcel_macro_f1") for k in KS], color=SLOTS[slot], marker=marker, lw=2,
                ms=7, markeredgecolor=SURFACE, markeredgewidth=1)
        ours.append(Line2D([], [], color=SLOTS[slot], marker=marker, lw=2, label=f"{label}, frozen"))
    ax.plot(x, [value(df, "tessera_v2_medium_ft", k, "parcel_macro_f1") for k in KS], color=SLOTS[3], marker="^",
            lw=2, ms=7, ls=(0, (4, 2)), markerfacecolor=SURFACE, markeredgewidth=1.5)
    ours.append(Line2D([], [], color=SLOTS[3], marker="^", lw=2, ls=(0, (4, 2)), markerfacecolor=SURFACE,
                       label="TESSERA v2 M, fine-tuned"))
    ax.plot(x, [value(df, BASELINE[1], k, "parcel_macro_f1") for k in KS], color=INK, marker="X", lw=2, ms=8,
            markeredgecolor=SURFACE)
    ours.append(Line2D([], [], color=INK, marker="X", lw=2, label=BASELINE[0]))
    ax.set_xscale("log")
    ticks = [1, 5, 10, 20, 100, 1000]
    ax.set_xticks(ticks, [f"{t:,}" for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.7, 2500)
    ax.set_ylim(0, 0.7)
    for k, xi in zip(KS, x):
        ax.annotate(f"K = {k} %", (xi, 0.7), xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                    fontsize=8.5, color=INK_2)
    ax.set_xlabel("Labelled training parcels per class (published: at most k; ours: median of the 20 classes)",
                  fontsize=9, color=INK_2)
    ax.set_ylabel("Test Macro-F1 per parcel", fontsize=9, color=INK_2)
    ax.set_title("Parcel Macro-F1, ours against the published Estonian results", loc="left", fontsize=11,
                 color=INK, fontweight="bold", pad=22)
    first = ax.legend(handles=ours, title="Ours: 20 classes, spatial blocks, one run", loc="upper left",
                      frameon=False, fontsize=8.5, title_fontsize=8.5, alignment="left", handlelength=3)
    ax.add_artist(first)
    ax.legend(handles=published, title="Published: DirPA, Reuss et al. 2026,\nrandom split, mean of five runs",
              loc="lower right", frameon=False, fontsize=8.5, title_fontsize=8.5, alignment="left", handlelength=3)
    fig.tight_layout()
    fig.savefig(OUT / "literature.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def tables(df: pd.DataFrame, g: pd.DataFrame, classes: pd.DataFrame) -> str:
    out = ["# Tables of the Estonian results report", ""]
    names = {f: l for l, f, _, _, _ in MODELS} | {t: f"{l}, fine-tuned" for l, _, t, _, _ in MODELS if t}
    names[BASELINE[1]] = BASELINE[0]
    names["tessera_v2_medium_lab"] = "TESSERA v2 M, frozen control of the fine-tuning route"
    for col, title in (("macro_f1", "Test Macro-F1 per pixel"), ("parcel_macro_f1", "Test Macro-F1 per parcel"),
                       ("miou", "Test mIoU"), ("pixel_accuracy", "Test pixel accuracy"),
                       ("parcel_accuracy", "Test parcel overall accuracy"), ("parcel_kappa", "Test parcel kappa"),
                       ("best_val_f1", "Best validation Macro-F1 over the 45 epochs"),
                       ("ckpt_epoch", "Epoch of the kept checkpoint (lowest validation loss)"),
                       ("fit_h", "Fit time, hours")):
        out += [f"## {title}", "", "| Arm | " + " | ".join(f"K = {k} %" for k in KS) + " |",
                "|---|" + "---|" * len(KS)]
        for arm, label in names.items():
            vals = [value(df, arm, k, col) for k in KS]
            fmt = "{:.0f}" if col == "ckpt_epoch" else "{:.1f}" if col == "fit_h" else "{:.3f}"
            out.append(f"| {label} | " + " | ".join(fmt.format(v) for v in vals) + " |")
        out.append("")
    out += ["## Fine-tuned minus frozen", "", "| Model | Readout | " + " | ".join(f"K = {k} %" for k in KS) + " |",
            "|---|---|" + "---|" * len(KS)]
    for m in dict.fromkeys(g["model"]):
        for col, lab in (("macro_f1", "test, pixel"), ("parcel_macro_f1", "test, parcel"),
                         ("best_val_f1", "best validation")):
            out.append(f"| {m} | {lab} | " + " | ".join(
                f"{g[(g['model'] == m) & (g['pct'] == k)][col].iloc[0]:+.3f}" for k in KS) + " |")
    out.append("")
    n = labelled_per_class(classes)
    out += ["## Labelled training parcels per class", "", "| | " + " | ".join(f"K = {k} %" for k in KS) + " |",
            "|---|" + "---|" * len(KS)]
    for stat, lab in (("min", "fewest"), ("median", "median"), ("max", "most"), ("sum", "total")):
        out.append(f"| {lab} | " + " | ".join(f"{n.loc[k, stat]:,.0f}" for k in KS) + " |")
    out += ["", "## Published Estonian parcel Macro-F1 (DirPA, Table 6, plain cross-entropy, mean of five runs)", "",
            "| Model | Classes | " + " | ".join(f"k = {k}" for k in SHOTS) + " |", "|---|---|" + "---|" * len(SHOTS)]
    for model, scheme, vals in LITERATURE:
        out.append(f"| {model} | {scheme} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
    out.append("")
    return "\n".join(out)


def main() -> None:
    OUT.mkdir(exist_ok=True)
    df, classes = load()
    g = gains(df)
    label_curves(df)
    finetune_gain(g)
    per_class(classes)
    literature(df, classes)
    (HERE / "tables.md").write_text(tables(df, g, classes))
    print((HERE / "tables.md").read_text())


if __name__ == "__main__":
    main()
