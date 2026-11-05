"""Figures and tables of the PaPs experiment, for its REPORT.md.

Reads ``results/<name>/<chip set>/`` as ``run.py`` writes it, and, as the reference, the boundary-head parcels of the
same TESSERA encoders in the same extents (``results/2026-11-05_parcel_extent/<chip set>/*/*/extent_scores.json``,
tau 0.5) when they are there. Writes into ``figures/`` beside this file:

* ``pq_curves.png``: PQ of PaPs' parcels against K, class-agnostic and class-aware, one column per extent, frozen
  and fine-tuned, with the boundary head's parcels and each extent's floor;
* ``validation_curves.png``: training losses and validation PQ per epoch, with the epoch kept;
* ``fp_where.png``: where PaPs' false parcels lie, per extent and arm at K = 100 %;
* ``example_<n>.png``: test chips at K = 100 %: the heatmap, then PaPs' parcels over the whole chip, coloured by how
  they score, for each arm;
* ``tables.md``: the tables of REPORT.md, also printed.

    poetry run python experiments/2026-11-05_paps_tessera/report.py --chips EE_2021_rev
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

import panoptic_x as px  # noqa: E402

NAME = "2026-11-05_paps_tessera"
REFERENCE = "2026-11-05_parcel_extent"
#: Arm -> (label, slot, the boundary-head arm on the same encoder).
ARMS = {"paps_tessera_v2_frozen": ("PaPs, TESSERA frozen", 0, "tessera_v2_medium"),
        "paps_tessera_v2_ft": ("PaPs, TESSERA fine-tuned", 1, "tessera_v2_medium_ft_bnd")}
EXTENTS = {"chip": "whole chip (PaPs as it is)", "worldcover": "cut to WorldCover 30 + 40",
           "oracle": "cut to the declared parcels (oracle)"}
FP_PLACES = ["on parcels", "trees", "grassland", "cropland", "water", "built-up", "wetland", "shrubland", "bare",
             "other"]

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
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


def _rows(variants: dict, base: dict) -> list[dict]:
    rows = []
    for k, q in variants.items():
        ext, var = (None, "oracle") if k == "oracle" else k.split("/")
        row = {**base, "extent": ext, "variant": var, **{m: q.get(m) for m in METRICS}}
        for place, n in (q.get("fp_on_background") or {}).items():
            row[f"fp_{place}"] = n
        rows.append(row)
    return rows


def load(chips: str) -> tuple[pd.DataFrame, dict]:
    rows, cells = [], {}
    for arm in ARMS:
        for f in sorted((REPO / "results" / NAME / chips / arm).glob("P*_draw*_seed*/results.json")):
            r = json.loads(f.read_text())
            pm = r.get("panoptic_metrics")
            if pm is None:
                continue
            pct = float(r["label_budget"]["pct"])
            cells[(arm, pct)] = r
            rows += _rows(pm["variants"], {"arm": arm, "pct": pct, "source": "paps"})
    for arm, (_, _, ref) in ARMS.items():
        for f in sorted((REPO / "results" / REFERENCE / chips / ref).glob("P*_draw*_seed*/extent_scores.json")):
            s = json.loads(f.read_text())
            pct = float(f.parent.name.split("_")[0][1:])
            rows += _rows(s["variants"], {"arm": arm, "pct": pct, "source": "boundary"})
    return pd.DataFrame(rows), cells


def _k_axis(ax, ks):
    ax.set_xticks(range(len(ks)), [f"{k:g} %" for k in ks])
    ax.set_xlim(-0.3, len(ks) - 0.7)
    ax.set_xlabel("label budget K")


def fig_pq(df: pd.DataFrame, path: Path) -> None:
    ks = sorted(df[df.source == "paps"]["pct"].unique(), reverse=True)
    fig, axes = plt.subplots(2, len(EXTENTS), figsize=(3.4 * len(EXTENTS), 6.4), sharey="row")
    for j, (ext, title) in enumerate(EXTENTS.items()):
        for i, (metric, ylab) in enumerate((("pq_agnostic", "PQ, class-agnostic"), ("pq", "PQ, class-aware"))):
            ax = axes[i, j]
            for arm, (label, slot, _) in ARMS.items():
                for src, ls, mk, lab in (("paps", "-", "o", label),
                                         ("boundary", ":", "s", label.replace("PaPs", "boundary head"))):
                    d = df[(df.arm == arm) & (df.source == src) & (df.extent == ext) &
                           (df.variant == ("paps" if src == "paps" else "boundary@0.5"))].set_index("pct")
                    d = d[d.index.isin(ks)]
                    if not d.empty:
                        ax.plot([ks.index(k) for k in d.index], d[metric], ls=ls, marker=mk, color=SLOTS[slot],
                                label=lab, lw=2 if src == "paps" else 1.5)
            fl = df[(df.source == "boundary") & (df.extent == ext) & (df.variant == "extent")]
            fl = fl[fl.arm == next(iter(ARMS))].set_index("pct")
            fl = fl[fl.index.isin(ks)]
            if not fl.empty and metric == "pq_agnostic":
                ax.plot([ks.index(k) for k in fl.index], fl[metric], ls="--", lw=1.3, color=MUTED,
                        label="floor: the extent's regions")
            _k_axis(ax, ks)
            if j == 0:
                ax.set_ylabel(ylab)
            if i == 0:
                ax.set_title(title, fontsize=9.5)
    axes[0, 0].legend(loc="lower left", fontsize=7.5)
    fig.suptitle("PaPs' parcels (solid) against the boundary head's (dotted), same encoders, same extents",
                 x=0.01, ha="left", fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_validation(cells: dict, path: Path) -> None:
    ks = sorted({k for _, k in cells}, reverse=True)
    if not ks:
        return
    fig, axes = plt.subplots(2, len(ks), figsize=(3.6 * len(ks), 5.8), squeeze=False)
    for j, k in enumerate(ks):
        for arm, (label, slot, _) in ARMS.items():
            r = cells.get((arm, k))
            if r is None:
                continue
            h = pd.DataFrame(r["history"])
            axes[0, j].plot(h["epoch"], h["train/loss"], color=SLOTS[slot], lw=1.5, label=label)
            v = h.dropna(subset=["val/PQ"]) if "val/PQ" in h else h.iloc[:0]
            axes[1, j].plot(v["epoch"], v["val/PQ"], color=SLOTS[slot], marker="o", ms=4, lw=1.5, label=label)
            axes[1, j].axvline(r["best_epoch"], color=SLOTS[slot], ls=":", lw=1)
        axes[0, j].set_title(f"K = {k:g} %: training loss")
        axes[0, j].set_yscale("log")
        axes[1, j].set_title("validation PQ (void rule, whole chip)")
        axes[1, j].set_xlabel("epoch")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_fp(df: pd.DataFrame, path: Path, pct: float = 100) -> None:
    d = df[(df.source == "paps") & (df.variant == "paps") & (df.pct == pct) & df.extent.isin(list(EXTENTS))]
    if d.empty:
        return
    rows = [(arm, ext) for ext in EXTENTS for arm in ARMS if not d[(d.arm == arm) & (d.extent == ext)].empty]
    colours = [INK_2, "#008300", "#eda100", "#e87ba4", "#2a78d6", "#4a3aa7", "#1baf7a", "#898781", "#c3c2b7",
               "#e1e0d9"]
    fig, ax = plt.subplots(figsize=(9.5, 0.55 * len(rows) + 1.6))
    for r, (arm, ext) in enumerate(rows):
        q = d[(d.arm == arm) & (d.extent == ext)].iloc[0]
        vals = [q["fp_on_parcels"]] + [q.get(f"fp_{p}", 0) or 0 for p in FP_PLACES[1:]]
        left = 0
        for v, c, p in zip(vals, colours, FP_PLACES):
            ax.barh(r, v, left=left, color=c, height=0.62, edgecolor=SURFACE, linewidth=1.5,
                    label=p if r == 0 else None)
            left += v
        ax.text(left * 1.01 + 50, r, f"{int(left):,} FP; {int(q['dropped_void']):,} dropped as void", va="center",
                fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)), [f"{EXTENTS[e].split(' (')[0]}, {ARMS[a][0]}" for a, e in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"false parcels of PaPs, K = {pct:g} %")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.18 - 0.02 * len(rows)))
    ax.set_title("Where PaPs' false parcels lie: on the true parcels, or on background by WorldCover class")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _outline(seg):
    o = np.zeros(seg.shape, bool)
    o[:-1] |= seg[:-1] != seg[1:]
    o[1:] |= seg[:-1] != seg[1:]
    o[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    o[:, 1:] |= seg[:, :-1] != seg[:, 1:]
    return o


def fig_examples(chips: str, folder: Path, n: int = 2) -> list[Path]:
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    cells = {a: REPO / "results" / NAME / chips / a / "P100_draw0_seed0" / "predictions_test.npz" for a in ARMS}
    data = {a: np.load(f) for a, f in cells.items() if f.exists()}
    if not data:
        return []
    first = next(iter(data.values()))
    ids = [str(c) for c in first["ids"]]
    wc_all = np.load(REPO / "data/worldcover" / chips.replace("_mini", "") / "worldcover.npz")
    chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
    scored = []
    for i in range(0, len(ids), 7 if len(ids) > 50 else 1):
        m = first["mask"][i]
        agri = np.isin(wc_all[ids[i]], px.WORLDCOVER).mean()
        if 0.35 < agri < 0.75 and (m >= 0).mean() > 0.25:
            scored.append((len(np.unique(read_parcels(chips_dir, ids[i])[m >= 0])), i))
    picks = [i for _, i in sorted(scored, reverse=True)[:n]] or [0]
    colours = {1: "#2a78d6", 2: "#eb6834", 3: "#c3c2b7"}
    out = []
    for p_i, i in enumerate(picks, start=1):
        cid = ids[i]
        parcels = read_parcels(chips_dir, cid)
        mask = first["mask"][i].astype(np.int64)
        E, true, _, void = px.truth(mask, parcels)
        panels = []
        for arm, d in data.items():
            j = list(map(str, d["ids"])).index(cid)
            panels.append((f"{ARMS[arm][0]}: heatmap", d["heatmap"][j], None))
            inst = d["inst"][j].astype(np.int64)
            ti, si, _, n_all, n_void = px.match(true, inst, void)
            st = np.zeros(len(n_all), np.int64)
            ids_ = np.nonzero(n_all)[0]
            ids_ = ids_[ids_ > 0]
            st[ids_] = np.where(n_void[ids_] * 2 > n_all[ids_], 3, 2)
            st[si] = 1
            panels.append((f"{ARMS[arm][0]}: parcels", inst, st))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.5))
        for ax, (title, a, st) in zip(np.atleast_1d(axes), panels):
            if st is None:
                ax.imshow(a, cmap="Greys", vmin=0, vmax=255, interpolation="nearest")
                ov = np.zeros(mask.shape + (4,))
                ov[_outline(np.where(E, true, 0)) & E] = (*matplotlib.colors.to_rgb("#2a78d6"), 0.8)
                ax.imshow(ov, interpolation="nearest")
            else:
                img = np.ones(mask.shape + (3,)) * np.array(matplotlib.colors.to_rgb("#f0efec"))
                for s, c in colours.items():
                    img[np.isin(a, np.nonzero(st == s)[0]) & (a > 0)] = matplotlib.colors.to_rgb(c)
                img[_outline(a) & (a > 0)] = matplotlib.colors.to_rgb(SURFACE)
                img[_outline(np.where(E, true, 0)) & E] = matplotlib.colors.to_rgb(INK)
                ax.imshow(img, interpolation="nearest")
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
        handles = [matplotlib.patches.Patch(color=c, label=lab) for c, lab in
                   ((colours[1], "matched a true parcel (IoU > 0.5)"), (colours[2], "false parcel"),
                    (colours[3], "dropped: mostly void"))]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8)
        fig.suptitle(f"Test chip {cid}, K = 100 %: PaPs over the whole chip; true parcels outlined", x=0.01,
                     ha="left", fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 1))
        f = folder / f"example_{p_i}.png"
        fig.savefig(f, dpi=150)
        plt.close(fig)
        out.append(f)
    return out


def _md(df: pd.DataFrame, fmt: str = ".3f") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(f"{v:{fmt}}" if isinstance(v, float) else f"{v:,}" if isinstance(v, (int, np.integer))
                                        else str(v) for v in r) + " |")
    return "\n".join(lines)


def tables(df: pd.DataFrame, cells: dict) -> str:
    out = []
    lab = {a: v[0] for a, v in ARMS.items()}
    p = df[(df.source == "paps") & (df.variant == "paps")]
    ref = df[(df.source == "boundary") & (df.variant == "boundary@0.5")].set_index(["arm", "pct", "extent"])
    for ext, title in {**EXTENTS, "worldcover_30_40_90": "cut to WorldCover 30 + 40 + 90"}.items():
        d = p[p.extent == ext].sort_values(["arm", "pct"], ascending=[True, False])
        if d.empty:
            continue
        refv = [ref.loc[(a, k, ext), "pq_agnostic"] if (a, k, ext) in ref.index else np.nan
                for a, k in zip(d.arm, d.pct)]
        t = pd.DataFrame({"Arm": d.arm.map(lab).values, "K": d.pct.map(lambda k: f"{k:g} %").values,
                          "PQ": d.pq.values, "SQ": d.sq.values, "RQ": d.rq.values,
                          "PQ agn.": d.pq_agnostic.values, "boundary head PQ agn.": refv,
                          "TP": d.tp_agnostic.astype(int).values, "FP": d.fp_agnostic.astype(int).values,
                          "FN": d.fn_agnostic.astype(int).values, "parcels": d.predicted_parcels.astype(int).values,
                          "dropped": d.dropped_void.astype(int).values})
        out.append(f"**PaPs, {title}**\n\n" + _md(t))
    if cells:
        t = pd.DataFrame([{"Arm": lab[a], "K": f"{k:g} %", "train chips": r["chips"]["train"],
                           "best epoch": r["best_epoch"], "val PQ": r["best_val_pq"],
                           "confidence": r["min_confidence"], "fit h": r["fit_seconds"] / 3600,
                           "peak GB": r["peak_gpu_gb"]} for (a, k), r in sorted(cells.items(), reverse=True)])
        out.append("**Fits**\n\n" + _md(t, ".2f"))
    return "\n\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    a = ap.parse_args()
    df, cells = load(a.chips)
    if df.empty or (df.source == "paps").sum() == 0:
        raise SystemExit("no scored PaPs cells yet")
    folder = HERE / "figures" if a.chips == "EE_2021_rev" else HERE / "figures" / a.chips
    folder.mkdir(parents=True, exist_ok=True)
    fig_pq(df, folder / "pq_curves.png")
    fig_validation(cells, folder / "validation_curves.png")
    fig_fp(df, folder / "fp_where.png")
    fig_examples(a.chips, folder)
    t = tables(df, cells)
    (folder / "tables.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
