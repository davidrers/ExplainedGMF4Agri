"""Figures and tables of the PRUE experiment, for its REPORT.md, in the style of the PaPs report
(``../2026-11-05_paps_tessera/REPORT.md``).

Reads ``results/<name>/<chip set>/`` as ``run.py`` and ``analyse.py`` write it and, as the references on the same
encoders, PaPs (``results/2026-11-05_paps_tessera``: its parcels with the crop-map class from its ``analysis.json``,
with its own class from its ``results.json``) and the boundary head (``results/2026-11-05_parcel_extent``,
``extent_scores.json``, tau 0.5). Only K = 100 %, the experiment's one budget. Writes into ``figures/``:

* ``pq_bars.png``: PQ, class-agnostic and class-aware, of PRUE's fields (grown and its own), PaPs and the boundary head
  over the whole chip, cut to WorldCover and, shaded apart, cut to the declared parcels (the upper bound, which uses
  the test labels); the ceiling of the grown fields (a perfect three-class map) marked;
* ``where_found.png``: the share of the true parcels found, by what they touch; ``found_by_size.png``: by area;
* ``errors.png``: what happened to the true parcels: matched, merged, split, missed, by area;
* ``fp_where.png``: where the false fields lie;
* ``validation_curves.png``: training loss, validation object F1 and PQ, boundary F1 per epoch, the epoch kept;
* ``example_<n>.png``: PaPs' three example test chips: the class map, the boundary probability, the fields;
* ``tables.md``: the tables of REPORT.md, also printed.

    poetry run python experiments/2026-11-06_prue_tessera/analyse.py --chips EE_2021_rev
    poetry run python experiments/2026-11-06_prue_tessera/report.py --chips EE_2021_rev
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
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import analyse  # noqa: E402
import panoptic_x as px  # noqa: E402
import prue  # noqa: E402

NAME = "2026-11-06_prue_tessera"
PAPS = "2026-11-05_paps_tessera"
EXTENT = "2026-11-05_parcel_extent"
PCT = 100
#: Arm -> (label, colour slot, the PaPs arm and the boundary-head arm on the same encoder).
ARMS = {"prue_tessera_v2_frozen": ("TESSERA frozen", 0, "paps_tessera_v2_frozen", "tessera_v2_medium"),
        "prue_tessera_v2_ft": ("TESSERA fine-tuned", 1, "paps_tessera_v2_ft", "tessera_v2_medium_ft_bnd")}
EXTENTS = {"chip": "whole chip", "worldcover": "WorldCover 30 + 40", "oracle": "declared parcels"}
SYSTEMS = {"grow": "PRUE, fields grown into their rims", "cc": "PRUE, its own fields",
           "field": "PRUE, predicted field unsplit", "paps_cropmap": "PaPs, crop-map class",
           "paps": "PaPs, its own class", "boundary": "boundary head (tau 0.5)", "ceiling": "perfect 3-class map, grown"}
#: Systems of the bar figure: key, hatch, short label.
BARS = [("grow", None, "PRUE grown"), ("cc", "...", "PRUE own"), ("paps_cropmap", "///", "PaPs"),
        ("boundary", "xx", "boundary head")]
FP_PLACES = ["on parcels", "trees", "grassland", "cropland", "water", "built-up", "wetland", "shrubland", "bare",
             "other"]

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
SHADE = "#f0efec"
PROBABILITY = LinearSegmentedColormap.from_list("probability", ["#ffffff", "#2a78d6", "#0d366b"])
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7,
})
METRICS = ("pq", "sq", "rq", "pq_agnostic", "sq_agnostic", "rq_agnostic", "tp_agnostic", "fp_agnostic",
           "fn_agnostic", "predicted_parcels", "true_parcels", "dropped_void", "fp_on_parcels")


# ---------------------------------------------------------------------------------------------------------- data

def _rows(variants: dict, base: dict, systems: dict) -> list[dict]:
    rows = []
    for k, q in variants.items():
        if "/" not in k:
            continue
        ext, var = k.split("/", 1)
        if var not in systems:
            continue
        row = {**base, "extent": ext, "system": systems[var], **{m: q.get(m) for m in METRICS}}
        for place, n in (q.get("fp_on_background") or {}).items():
            row[f"fp_{place}"] = n
        rows.append(row)
    return rows


def _cell(name: str, chips: str, arm: str) -> Path:
    return REPO / "results" / name / chips / arm / f"P{PCT}_draw0_seed0"


def load(chips: str) -> tuple[pd.DataFrame, dict, dict, dict]:
    """Every score as one row (arm, extent, system); the PRUE fits; the analyses; the PaPs fits."""
    rows, cells, analyses, paps_cells = [], {}, {}, {}
    for arm, (_, _, paps_arm, bnd_arm) in ARMS.items():
        cell = _cell(NAME, chips, arm)
        if (cell / "results.json").exists():
            r = json.loads((cell / "results.json").read_text())
            if "panoptic_metrics" in r:
                cells[arm] = r
                rows += _rows(r["panoptic_metrics"]["variants"], {"arm": arm}, {v: v for v in prue.VARIANTS})
        if (cell / "analysis.json").exists():
            analyses[arm] = json.loads((cell / "analysis.json").read_text())
        pc = _cell(PAPS, chips, paps_arm)
        if (pc / "results.json").exists():
            r = json.loads((pc / "results.json").read_text())
            paps_cells[arm] = r
            rows += _rows(r["panoptic_metrics"]["variants"], {"arm": arm}, {"paps": "paps"})
        if (pc / "analysis.json").exists():
            a = json.loads((pc / "analysis.json").read_text())
            rows += _rows(a["variants"], {"arm": arm}, {"paps_cropmap": "paps_cropmap", "cropmap": "truth_cropmap"})
        f = _cell(EXTENT, chips, bnd_arm) / "extent_scores.json"
        if f.exists():
            rows += _rows(json.loads(f.read_text())["variants"], {"arm": arm}, {"boundary@0.5": "boundary"})
    f = REPO / "results" / NAME / chips / "ceiling.json"
    if f.exists():
        rows += _rows(json.loads(f.read_text())["variants"], {"arm": "ceiling"},
                      {"grow": "ceiling", "cc": "ceiling_cc", "field": "ceiling_field"})
    return pd.DataFrame(rows), cells, analyses, paps_cells


def _get(df, arm, system, extent, metric):
    q = df[(df.arm == arm) & (df.system == system) & (df.extent == extent)]
    return float(q[metric].iloc[0]) if len(q) and q[metric].iloc[0] is not None else np.nan


# ------------------------------------------------------------------------------------------------------- figures

def fig_pq(df: pd.DataFrame, path: Path) -> None:
    """Grouped bars per extent, colour the encoder, hatch the system; the ceiling of the grown fields marked."""
    arms = [a for a in ARMS if not df[(df.arm == a) & (df.system == "grow")].empty]
    exts = list(EXTENTS)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6))
    for ax, metric, ylab, title in ((axes[0], "pq_agnostic", "PQ, class-agnostic", "Class-agnostic: the geometry alone"),
                                    (axes[1], "pq", "PQ, class-aware", "Class-aware: field and crop (crop-map class)")):
        bars = [(arm, s, h, f"{lab}, {ARMS[arm][0]}") for s, h, lab in BARS for arm in arms
                if not df[(df.arm == arm) & (df.system == s)].empty]
        w = 0.86 / max(len(bars), 1)
        ax.axvspan(1.5, 2.5, color=SHADE, zorder=0, lw=0)
        top = 0.0
        for b, (arm, s, hatch, lab) in enumerate(bars):
            slot = SLOTS[ARMS[arm][1]]
            for j, ext in enumerate(exts):
                v = _get(df, arm, s, ext, metric)
                if np.isnan(v):
                    continue
                x = j + (b - (len(bars) - 1) / 2) * w
                ax.bar(x, v, w * 0.94, color=SURFACE if hatch else slot, edgecolor=slot, hatch=hatch, lw=1.2,
                       label=lab if j == 0 else None)
                ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=6.5, color=INK_2, rotation=90)
                top = max(top, v)
        for j, ext in enumerate(exts):
            c = _get(df, "ceiling", "ceiling", ext, metric)
            if not np.isnan(c):
                ax.hlines(c, j - 0.45, j + 0.45, color=INK, lw=1.4, ls="--")
                top = max(top, c)
        ax.set_ylim(0, top * 1.15 + 0.02)
        ax.text(2, top * 1.13 + 0.02, "upper bound:\nuses the test labels", ha="center", va="top", fontsize=8,
                color=INK_2)
        ax.set_xticks(range(len(exts)), ["whole chip\n(no extent)", "cut to\nWorldCover 30 + 40",
                                         "cut to the\ndeclared parcels"], fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_title(title)
    h, lab = axes[0].get_legend_handles_labels()
    h.append(Line2D([], [], color=INK, lw=1.4, ls="--"))
    lab.append("ceiling: the true 3-class map, grown")
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8)
    fig.suptitle(f"PRUE's fields against PaPs and the boundary head, same encoders, K = {PCT} %", x=0.01, ha="left",
                 fontsize=11.5)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


FOUND = [("grow", "o", True, "PRUE grown"), ("cc", "o", False, "PRUE own"), ("paps", "s", True, "PaPs")]


def fig_where(analyses: dict, path: Path) -> None:
    if not analyses:
        return
    arms = list(analyses)
    f0 = next(iter(analyses.values()))["found"]
    tot = f0["true_parcels"]["by_kind"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 1.0 * len(arms) + 2.0), sharey=True)
    for k, ax in enumerate(axes):
        for i, arm in enumerate(arms):
            sh = analyses[arm]["found"]["share"]
            slot = SLOTS[ARMS[arm][1]]
            vals = [sh[s]["by_kind"][k] for s, *_ in FOUND]
            ax.hlines(i, min(vals), max(vals), color=GRID, lw=3, zorder=1)
            for s, m, filled, lab in FOUND:
                v = sh[s]["by_kind"][k]
                ax.scatter([v], [i], s=70, marker=m, color=slot if filled else SURFACE,
                           edgecolor=INK if s == "paps" else slot, lw=1.6, zorder=4,
                           label=lab if i == 0 and k == 0 else None)
        ax.set_title(f"{f0['kinds'][k]}: {tot[k]:,} parcels, {tot[k] / sum(tot):.0%}", fontsize=10)
        ax.set_xlim(-0.02, 0.82)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("share found, class-agnostic IoU > 0.5")
    axes[0].set_yticks(range(len(arms)), [ARMS[a][0] for a in arms])
    axes[0].set_ylim(-0.6, len(arms) - 0.4)
    axes[0].invert_yaxis()
    fig.legend(loc="upper center", ncol=3, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, f"K = {PCT} %, whole chip, no extent. True parcel pieces of the test chips by what they touch "
             "inside the declared extent. Marks in the encoder's colour; PaPs squares outlined in black.",
             color=MUTED, fontsize=8.5, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_size(analyses: dict, path: Path) -> None:
    if not analyses:
        return
    f0 = next(iter(analyses.values()))["found"]
    labels = [f"{s}\n{n:,}" for s, n in zip(f0["sizes"], f0["true_parcels"]["by_size"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    x = np.arange(len(labels))
    for arm, a in analyses.items():
        sh = a["found"]["share"]
        slot = SLOTS[ARMS[arm][1]]
        for s, m, filled, lab in FOUND:
            ls = {"grow": "-", "cc": ":", "paps": "--"}[s]
            ax.plot(x, sh[s]["by_size"], color=slot, ls=ls, lw=2 if s == "grow" else 1.5, marker=m,
                    mfc=slot if filled else SURFACE, mec=INK if s == "paps" else slot, label=f"{lab}, {ARMS[arm][0]}")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_xlabel("area of the true parcel piece, and how many there are in the test chips")
    ax.set_ylabel("share found, class-agnostic IoU > 0.5")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=7.5, loc="upper left", ncol=2)
    ax.set_title(f"The parcels found by area, whole chip, K = {PCT} %")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


ERROR_COLOURS = {"matched": "#1baf7a", "merged": "#eda100", "split": "#e87ba4", "missed": "#c3c2b7"}


def fig_errors(analyses: dict, path: Path) -> None:
    """Per encoder, per area bin (and all): stacked shares of matched, merged, split and missed, three systems."""
    if not analyses:
        return
    arms = list(analyses)
    f0 = next(iter(analyses.values()))
    sizes = ["all"] + f0["errors"]["sizes"]
    systems = [("cc", "PRUE own"), ("grow", "PRUE grown"), ("paps", "PaPs")]
    fig, axes = plt.subplots(len(arms), 1, figsize=(12, 3.3 * len(arms)), sharex=True)
    axes = np.atleast_1d(axes)
    w = 0.26
    for ax, arm in zip(axes, arms):
        for j, (s, lab) in enumerate(systems):
            e = np.array(analyses[arm]["errors"]["counts"][s], float)          # (4 types, 5 sizes)
            e = np.concatenate([e.sum(1, keepdims=True), e], axis=1)
            share = e / e.sum(0, keepdims=True)
            x = np.arange(len(sizes)) + (j - 1) * (w + 0.02)
            bottom = np.zeros(len(sizes))
            for t, typ in enumerate(f0["errors"]["types"]):
                ax.bar(x, share[t], w, bottom=bottom, color=ERROR_COLOURS[typ], edgecolor=SURFACE, lw=0.8,
                       label=typ if (j == 0 and ax is axes[0]) else None)
                bottom += share[t]
            for xi in x:
                ax.text(xi, 1.01, {"cc": "own", "grow": "grown", "paps": "PaPs"}[s], ha="center", va="bottom",
                        fontsize=6.5, color=INK_2, rotation=90)
        ax.set_ylim(0, 1.13)
        ax.set_ylabel("share of the true parcels")
        ax.set_title(ARMS[arm][0], pad=14)
    tot = [sum(f0["found"]["true_parcels"]["by_size"])] + f0["found"]["true_parcels"]["by_size"]
    axes[-1].set_xticks(range(len(sizes)), [f"{s}\n{n:,}" for s, n in zip(sizes, tot)], fontsize=8.5)
    fig.legend(loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.0, "matched: IoU > 0.5. merged: one predicted field holds more than half of the parcel but is too "
             "large (joined with a neighbour or spilling into background). split: mostly inside fields, no field holds "
             "half. missed: at most half inside any field. Whole chip, K = 100 %.", color=MUTED, fontsize=8.5,
             va="bottom", wrap=True)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_fp(df: pd.DataFrame, path: Path) -> None:
    rows = [(arm, s) for arm in ARMS for s in ("grow", "cc", "paps")
            if not df[(df.arm == arm) & (df.system == s) & (df.extent == "chip")].empty]
    if not rows:
        return
    colours = [INK_2, "#008300", "#eda100", "#e87ba4", "#2a78d6", "#4a3aa7", "#1baf7a", "#898781", "#c3c2b7",
               "#e1e0d9"]
    fig, ax = plt.subplots(figsize=(10, 0.5 * len(rows) + 1.8))
    for r, (arm, s) in enumerate(rows):
        q = df[(df.arm == arm) & (df.system == s) & (df.extent == "chip")].iloc[0]
        vals = [q["fp_on_parcels"]] + [q.get(f"fp_{p}", 0) or 0 for p in FP_PLACES[1:]]
        vals = [0 if (isinstance(v, float) and np.isnan(v)) else v for v in vals]
        left = 0
        for v, c, p in zip(vals, colours, FP_PLACES):
            ax.barh(r, v, left=left, color=c, height=0.62, edgecolor=SURFACE, linewidth=1.5,
                    label=p if r == 0 else None)
            left += v
        ax.text(left * 1.01 + 50, r, f"{int(left):,} FP of {int(q['predicted_parcels']):,} fields", va="center",
                fontsize=8, color=INK_2)
    lab = {"grow": "PRUE grown", "cc": "PRUE own", "paps": "PaPs"}
    ax.set_yticks(range(len(rows)), [f"{lab[s]}, {ARMS[a][0]}" for a, s in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.set_xlim(0, ax.get_xlim()[1] * 1.25)
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"false fields over the whole chip, K = {PCT} %")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.22 - 0.015 * len(rows)))
    fig.suptitle("Where the false fields lie: on the true parcels (a wrong split or poor fit), or on background by "
                 "WorldCover class", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_validation(cells: dict, paps_cells: dict, path: Path) -> None:
    if not cells:
        return
    fig, axes = plt.subplots(1, 3, figsize=(13, 5.0))
    for arm, r in cells.items():
        label, slot = ARMS[arm][0], SLOTS[ARMS[arm][1]]
        h = pd.DataFrame(r["history"])
        axes[0].plot(h["epoch"], h["train/loss"], color=slot, lw=1.5, label=f"{label}, training")
        axes[0].plot(h["epoch"], h["val/loss"], color=slot, lw=1.5, ls="--", label=f"{label}, validation")
        axes[1].plot(h["epoch"], h["val/grow/PQ_agnostic"], color=slot, lw=1.8, marker="o", ms=3,
                     label=f"{label}, PQ grown")
        axes[1].plot(h["epoch"], h["val/cc/PQ_agnostic"], color=slot, lw=1.3, ls=":", marker="o", ms=3, mfc=SURFACE,
                     label=f"{label}, PQ own")
        axes[1].plot(h["epoch"], h["val/cc/RQ_agnostic"], color=slot, lw=1.3, ls="--",
                     label=f"{label}, object F1 own (kept on)")
        axes[2].plot(h["epoch"], h["val/boundary_f1"], color=slot, lw=1.5, label=f"{label}, boundary F1")
        axes[2].plot(h["epoch"], h["val/iou_boundary"], color=slot, lw=1.3, ls="--", label=f"{label}, boundary IoU")
        for ax in axes[1:]:
            ax.axvline(r["best_epoch"], color=slot, ls=":", lw=1)
    for arm, r in paps_cells.items():
        hp = pd.DataFrame(r["history"]).dropna(subset=["val/PQ_agnostic"])
        if len(hp):
            axes[1].axhline(hp["val/PQ_agnostic"].max(), color=SLOTS[ARMS[arm][1]], lw=1, ls="-.", alpha=0.7)
    axes[1].plot([], [], color=MUTED, lw=1, ls="-.", label="PaPs' best validation PQ (of 100 epochs)")
    axes[0].set_title("loss (log-cosh Dice)")
    axes[0].set_yscale("log")
    axes[1].set_title("validation fields, whole chip, class-agnostic")
    axes[2].set_title("validation boundary pixels, inside the parcels")
    for ax in axes:
        ax.set_xlabel("epoch")
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
    fig.suptitle(f"PRUE training, K = {PCT} %; dotted vertical: the epoch kept (best validation object F1 of PRUE's own "
                 "fields)", x=0.01, ha="left", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------------ examples

SHORT = {"pasture_meadow_grassland_grass": "grassland", "winter_common_soft_wheat": "winter wheat",
         "spring_common_soft_wheat": "spring wheat", "legumes_harvested_green": "legumes, green harvest",
         "winter_rapeseed_rape": "winter rapeseed", "spring_rapeseed_rape": "spring rapeseed",
         "grain_maize_corn_popcorn": "grain maize", "legumes_dried_pulses_protein_crops": "dried pulses",
         "alfalfa_lucerne": "alfalfa", "orchards_fruits": "orchards", "fresh_vegetables": "vegetables"}
OTHER_CLASS = "#a3a29b"
CLASS3 = {prue.BACKGROUND: "#e9e2cf", prue.INTERIOR: "#a9c8f0", prue.BOUNDARY: "#0d366b", prue.IGNORE: "#d6d5d0"}


def _outline(seg: np.ndarray) -> np.ndarray:
    e = np.zeros(seg.shape, bool)
    v = seg[:-1] != seg[1:]
    e[:-1] |= v & (seg[:-1] > 0)
    e[1:] |= v & (seg[1:] > 0)
    h = seg[:, :-1] != seg[:, 1:]
    e[:, :-1] |= h & (seg[:, :-1] > 0)
    e[:, 1:] |= h & (seg[:, 1:] > 0)
    return e


def pick_chips(chips_dir: Path, ids: list[str], n: int = 3) -> list[str]:
    """The PaPs and delineation reports' choice: the test chips with the most contacts between same-class parcels."""
    import rasterio

    from gfm4agri.benchmark.parcel_metrics import read_parcels

    scores = []
    for cid in ids:
        p = read_parcels(chips_dir, cid)
        with rasterio.open(chips_dir / f"{cid}.mask.tif") as s:
            m = s.read(1)
        same = 0
        for a, b, ma, mb in ((p[:-1], p[1:], m[:-1], m[1:]), (p[:, :-1], p[:, 1:], m[:, :-1], m[:, 1:])):
            same += int(((a != b) & (a > 0) & (b > 0) & (ma == mb) & (ma >= 0)).sum())
        scores.append((same, cid))
    return [c for _, c in sorted(scores, reverse=True)[:n]]


def _show(ax, img, title):
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=8.5)
    ax.axis("off")


def _classes_img(cls, where, colours, outline=None):
    img = np.ones(cls.shape + (3,))
    for k in np.unique(cls[where]):
        img[where & (cls == k)] = matplotlib.colors.to_rgb(colours.get(int(k), OTHER_CLASS))
    if outline is not None:
        img[outline] = matplotlib.colors.to_rgb(INK)
    return img


def _rows_of(path: Path, keys: tuple[str, ...], chosen: list[str]) -> dict | None:
    if not path.exists():
        return None
    with np.load(path) as d:
        order = {str(c): i for i, c in enumerate(d["ids"])}
        return {k: d[k][[order[c] for c in chosen]] for k in keys if k in d.files}


def fig_examples(chips: str, folder: Path) -> list[Path]:
    """One figure per chip. Row 1: the July image, the three-class target, and per encoder the class map and the
    boundary probability. Row 2: the true parcels, and per encoder PRUE's own fields, the grown ones and PaPs'."""
    import rasterio

    from gfm4agri.benchmark.parcel_metrics import read_parcels

    chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
    first = _cell(NAME, chips, next(iter(ARMS)))
    if not (first / "predictions_test.npz").exists():
        return []
    names = json.loads((first / "results.json").read_text())["class_names"]
    n_cls = len(names)
    with np.load(first / "predictions_test.npz") as d:
        ids = [str(c) for c in d["ids"]]
    chosen = pick_chips(chips_dir, ids)
    data = {}
    for arm, (_, _, paps_arm, _) in ARMS.items():
        pr = _rows_of(_cell(NAME, chips, arm) / "predictions_test.npz", ("prob", "mask"), chosen)
        if pr is None:
            continue
        src = {k: REPO / v.format(set=chips) for k, v in analyse.SOURCES[arm].items()}
        data[arm] = {"prue": pr, "crop": _rows_of(src["cropmap"] / "predictions_test.npz", ("pred",), chosen),
                     "paps": _rows_of(src["paps"] / "predictions_test.npz", ("inst",), chosen)}
    manifest = json.loads((REPO / "data/eurocrops_chips" / chips / "manifest.json").read_text())
    bands = manifest["imagery"]["band_names"]
    wc_all = np.load(REPO / "data/worldcover" / chips.replace("_mini", "") / "worldcover.npz")

    def chip_score(true, true_cls, seg, crop, void):
        cls = px.vote(seg, crop, n_cls)
        aware, agn, _ = px.chip_counts(true, true_cls, seg, cls, void, n_cls)
        q = px.quality(aware, agn, names)
        return cls, int(agn[0]), q["pq_agnostic"]

    shown, area = [], np.zeros(n_cls, np.int64)
    for i, cid in enumerate(chosen):
        parcels = read_parcels(chips_dir, cid)
        mask = next(iter(data.values()))["prue"]["mask"][i].astype(np.int64)
        E, true, true_cls, void = px.truth(mask, parcels)
        area += np.bincount(mask[E], minlength=n_cls)
        y = prue.target(mask, parcels.astype(np.int64), wc_all[cid])
        target_b = y == prue.BOUNDARY
        panels = {}
        for arm, d in data.items():
            prob = d["prue"]["prob"][i]
            crop = d["crop"]["pred"][i]
            hit = E & (prob.argmax(0) == prue.BOUNDARY)
            tp, fp, fn = (hit & target_b).sum(), (hit & ~target_b).sum(), (E & target_b & ~hit).sum()
            p = {"prob": prob, "bf1": 2 * tp / max(2 * tp + fp + fn, 1)}
            for v in ("cc", "grow"):
                seg = prue.parcels(prob, v)
                p[v] = (seg, *chip_score(true, true_cls, seg, crop, void))
            inst = d["paps"]["inst"][i].astype(np.int64)
            p["paps"] = (inst, *chip_score(true, true_cls, inst, crop, void))
            panels[arm] = p
        for p in panels.values():
            for v in ("cc", "grow", "paps"):
                seg, cls = p[v][0], p[v][1]
                area += np.bincount(cls[seg][seg > 0], minlength=n_cls)
        shown.append((cid, E, true, true_cls, mask, y, panels))
    ranked = [int(k) for k in np.argsort(-area, kind="stable") if area[k] > 0]
    colours = dict(zip(ranked, SLOTS))

    paths = []
    for r, (cid, E, true, true_cls, mask, y, panels) in enumerate(shown, start=1):
        ncol = 2 + 3 * len(panels)
        fig, axes = plt.subplots(2, ncol, figsize=(2.3 * ncol, 5.6))
        with rasterio.open(chips_dir / f"{cid}_merged.tif") as s:
            july = 6 * len(bands)
            rgb = np.stack([s.read(july + bands.index(b) + 1) for b in ("RED", "GREEN", "BLUE")], -1)
        rgb = np.clip(rgb.astype(np.float32) / np.nanpercentile(rgb, 98), 0, 1)
        _show(axes[0, 0], np.nan_to_num(rgb), f"{cid}\nSentinel-2, July")
        t_img = np.ones(y.shape + (3,))
        for k, c in CLASS3.items():
            t_img[y == k] = matplotlib.colors.to_rgb(c)
        _show(axes[0, 1], t_img, "target: background (sand),\ninterior, boundary;\ngrey: no loss")
        _show(axes[1, 0], np.ones(E.shape + (3,)), "")
        _show(axes[1, 1], _classes_img(true_cls[true], E, colours, _outline(true) & E),
              f"true parcels: {int(np.unique(true[E]).size)}\n(declared, in scheme)")
        for a, (arm, p) in enumerate(panels.items()):
            lab = ARMS[arm][0]
            c0 = 2 + 3 * a
            cm = np.ones(E.shape + (3,))
            for k in (prue.BACKGROUND, prue.INTERIOR, prue.BOUNDARY):
                cm[p["prob"].argmax(0) == k] = matplotlib.colors.to_rgb(CLASS3[k])
            _show(axes[0, c0], cm, f"PRUE, {lab}\nclass map")
            axes[0, c0 + 1].imshow(p["prob"][prue.BOUNDARY] / 255.0, cmap=PROBABILITY, vmin=0, vmax=1,
                                   interpolation="nearest")
            axes[0, c0 + 1].set_title(f"PRUE, {lab}\np(boundary); pixel F1\n{p['bf1']:.2f} in the parcels",
                                      fontsize=8.5)
            axes[0, c0 + 1].axis("off")
            axes[0, c0 + 2].axis("off")
            for j, (v, t) in enumerate((("cc", "PRUE own fields"), ("grow", "PRUE grown"), ("paps", "PaPs"))):
                seg, cls, found, q = p[v]
                n_pred = int(np.unique(seg[seg > 0]).size)
                _show(axes[1, c0 + j], _classes_img(cls[seg], seg > 0, colours, _outline(seg)),
                      f"{t}, {lab.split()[1]}\n{n_pred} fields, {found} found\nPQ agn. {q:.2f}")
        present = {int(k) for k in np.unique(mask[E])}
        for p in panels.values():
            for v in ("cc", "grow", "paps"):
                seg, cls = p[v][0], p[v][1]
                present |= {int(k) for k in np.unique(cls[seg][seg > 0])}
        handles = [matplotlib.patches.Patch(color=colours[k], label=SHORT.get(names[k], names[k].replace("_", " ")))
                   for k in ranked if k in present and k in colours]
        if present - set(colours):
            handles.append(matplotlib.patches.Patch(color=OTHER_CLASS, label="other classes"))
        axes[1, 0].legend(handles=handles, loc="center left", fontsize=7.5, handlelength=1.2, borderaxespad=0,
                          title="crop type", title_fontsize=8, alignment="left")
        for a in range(len(panels)):
            axes[0, 2 + 3 * a + 2].text(0.0, 0.5, "fields coloured by the\ncrop-type map's vote\n(same encoder);\n"
                                        "found: IoU > 0.5\nwith a true parcel", fontsize=7.5, color=INK_2,
                                        va="center", transform=axes[0, 2 + 3 * a + 2].transAxes)
        fig.tight_layout()
        path = folder / f"example_{r}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


# -------------------------------------------------------------------------------------------------------- tables

def _md(df: pd.DataFrame, fmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if isinstance(v, float) and np.isnan(v) else f"{v:{fmt}}"
                                        if isinstance(v, float) else f"{v:,}" if isinstance(v, (int, np.integer))
                                        else str(v) for v in r) + " |")
    return "\n".join(lines)


def _system_table(df: pd.DataFrame, ext: str) -> pd.DataFrame:
    out = []
    for arm in ARMS:
        for s in ("grow", "cc", "field", "paps_cropmap", "paps", "boundary"):
            q = df[(df.arm == arm) & (df.system == s) & (df.extent == ext)]
            if q.empty:
                continue
            q = q.iloc[0]
            out.append({"Encoder": ARMS[arm][0], "Fields from": SYSTEMS[s], "PQ": q.pq, "SQ": q.sq, "RQ": q.rq,
                        "PQ agn.": q.pq_agnostic, "SQ agn.": q.sq_agnostic, "RQ agn.": q.rq_agnostic,
                        "TP": int(q.tp_agnostic), "FP": int(q.fp_agnostic), "FN": int(q.fn_agnostic),
                        "fields": int(q.predicted_parcels)})
    for s, lab in (("ceiling", "perfect 3-class map, grown"), ("ceiling_cc", "perfect 3-class map, own"),
                   ("ceiling_field", "perfect 3-class map, unsplit")):
        q = df[(df.arm == "ceiling") & (df.system == s) & (df.extent == ext)]
        if len(q):
            q = q.iloc[0]
            out.append({"Encoder": "ceiling (test labels)", "Fields from": lab, "PQ": q.pq, "SQ": q.sq, "RQ": q.rq,
                        "PQ agn.": q.pq_agnostic, "SQ agn.": q.sq_agnostic, "RQ agn.": q.rq_agnostic,
                        "TP": int(q.tp_agnostic), "FP": int(q.fp_agnostic), "FN": int(q.fn_agnostic),
                        "fields": int(q.predicted_parcels)})
    return pd.DataFrame(out)


def tables(df: pd.DataFrame, cells: dict, analyses: dict, paps_cells: dict) -> str:
    out = []
    for ext, title in (("chip", "Whole chip, no extent"), ("worldcover", "Cut to WorldCover 30 + 40"),
                       ("worldcover_30_40_90", "Cut to WorldCover 30 + 40 + 90 (sensitivity)"),
                       ("oracle", "Cut to the declared parcels: upper bound, uses the test labels")):
        t = _system_table(df, ext)
        if not t.empty:
            out.append(f"**{title}**\n\n" + _md(t))
    if cells:
        rows = []
        for a, r in cells.items():
            pm = r["panoptic_metrics"]["pixels"]
            rows.append({"Encoder": ARMS[a][0], "IoU background": pm["classes"]["background"]["iou"],
                         "IoU interior": pm["classes"]["interior"]["iou"],
                         "IoU boundary": pm["classes"]["boundary"]["iou"], "field IoU": pm["field_iou"],
                         "boundary F1 (in parcels)": pm["boundary_argmax"]["f1"],
                         "boundary precision": pm["boundary_argmax"]["precision"],
                         "boundary recall": pm["boundary_argmax"]["recall"]})
        out.append("**Per pixel, test chips** (three-class IoU on the pixels the target knows; boundary pixels inside "
                   "the declared parcels)\n\n" + _md(pd.DataFrame(rows)))
    if analyses:
        rows = []
        for a, an in analyses.items():
            f = an["found"]
            for s, *_, lab in FOUND:
                rows.append({"Encoder": ARMS[a][0], "Fields from": lab, "all": f["share"][s]["all"],
                             **dict(zip(f["kinds"], f["share"][s]["by_kind"])),
                             **dict(zip(f["sizes"], f["share"][s]["by_size"]))})
        f = next(iter(analyses.values()))["found"]
        counts = ", ".join(f"{k} {n:,}" for k, n in zip(f["kinds"] + f["sizes"],
                                                        f["true_parcels"]["by_kind"] + f["true_parcels"]["by_size"]))
        out.append(f"**Share of the true parcels found, whole chip** (true parcels: {counts})\n\n"
                   + _md(pd.DataFrame(rows)))
        rows = []
        for a, an in analyses.items():
            for s, *_, lab in FOUND:
                e = np.array(an["errors"]["counts"][s])
                rows.append({"Encoder": ARMS[a][0], "Fields from": lab,
                             **{t: int(n) for t, n in zip(an["errors"]["types"], e.sum(1))},
                             **{f"merged, {z}": float(e[1, k] / max(e[:, k].sum(), 1))
                                for k, z in enumerate(an["errors"]["sizes"])}})
        out.append("**What happened to the true parcels** (counts; merged as a share of each area bin)\n\n"
                   + _md(pd.DataFrame(rows)))
        rows = [{"Encoder": ARMS[a][0], "Fields from": lab, "parcels found": an["class_on_matched"][s]["parcels"],
                 "crop-map vote accuracy": an["class_on_matched"][s]["accuracy"],
                 "crop-map vote macro F1": an["class_on_matched"][s]["macro_f1"]}
                for a, an in analyses.items() for s, *_, lab in FOUND]
        out.append("**The class of the parcels found** (the crop-type map's vote against the true class)\n\n"
                   + _md(pd.DataFrame(rows)))
    if cells:
        rows = []
        for a, r in cells.items():
            h = pd.DataFrame(r["history"])
            b = h[h.epoch == r["best_epoch"]].iloc[0]
            row = {"Encoder": ARMS[a][0], "train chips": r["n_chips"]["train"], "best epoch": r["best_epoch"],
                   "val object F1 (own)": b["val/cc/RQ_agnostic"], "val PQ agn. own": b["val/cc/PQ_agnostic"],
                   "val PQ agn. grown": b["val/grow/PQ_agnostic"],
                   "best val PQ agn. grown (epoch)": f"{h['val/grow/PQ_agnostic'].max():.3f} "
                                                     f"({int(h.loc[h['val/grow/PQ_agnostic'].idxmax(), 'epoch'])})",
                   "fit h": r["fit_seconds"] / 3600, "peak GB": r["peak_gpu_gb"],
                   "trainable M": (r["params"]["unet"] + r["params"]["encoder_trainable"]) / 1e6,
                   "commit": r.get("provenance", {}).get("git_commit", "")[:8]}
            pc = paps_cells.get(a)
            if pc is not None:
                hp = pd.DataFrame(pc["history"]).dropna(subset=["val/PQ_agnostic"])
                row["PaPs best val PQ agn."] = float(hp["val/PQ_agnostic"].max())
            rows.append(row)
        out.append("**Fits**\n\n" + _md(pd.DataFrame(rows)))
    return "\n\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args()
    df, cells, analyses, paps_cells = load(a.chips)
    if df.empty or (df.system == "grow").sum() == 0:
        raise SystemExit("no scored PRUE cells yet")
    folder = HERE / "figures" if a.chips == "EE_2021_rev" else HERE / "figures" / a.chips
    folder.mkdir(parents=True, exist_ok=True)
    fig_pq(df, folder / "pq_bars.png")
    fig_where(analyses, folder / "where_found.png")
    fig_size(analyses, folder / "found_by_size.png")
    fig_errors(analyses, folder / "errors.png")
    fig_fp(df, folder / "fp_where.png")
    fig_validation(cells, paps_cells, folder / "validation_curves.png")
    if not a.no_examples:
        fig_examples(a.chips, folder)
    t = tables(df, cells, analyses, paps_cells)
    (folder / "tables.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
