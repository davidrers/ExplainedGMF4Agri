"""Figures and tables of the extent experiment, for its REPORT.md, from ``score.py``'s ``extent_scores.json``.

Writes into ``figures/`` beside this file:

* ``extents.png``: PQ of the boundary parcels (tau 0.5) at K = 100 %, class-agnostic and class-aware, grouped by
  extent, frozen and fine-tuned TESSERA, each bar with its extent's floor (its own connected regions, nothing
  predicted);
* ``variants.png``: class-agnostic PQ of every variant (floor, crop-type map, boundary head at three thresholds,
  boundary head and class changes) per extent;
* ``fp_where.png``: where the false parcels of the boundary head lie (tau 0.5), per extent and arm at K = 100 %: on
  the true parcels (wrong split or merge) or on background, by WorldCover class; and the parcels dropped as void;
* ``example_<n>.png``: test chips at K = 100 %: WorldCover's extent and the true parcels, then the boundary parcels
  of each arm built inside WorldCover and inside the whole chip, coloured by how they score;
* ``tables.md``: the tables of REPORT.md, also printed.

    poetry run python experiments/2026-11-05_parcel_extent/report.py --chips EE_2021_rev
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

import panoptic_x as px  # noqa: E402

NAME = "2026-11-05_parcel_extent"
#: Arm -> (label, slot); the frozen cells are the delineation experiment's.
ARMS = {"tessera_v2_medium": ("TESSERA v2, frozen", 0), "tessera_v2_medium_ft_bnd": ("TESSERA v2, fine-tuned", 1)}
SOURCES = {"tessera_v2_medium": "2026-10-18_parcel_delineation", "tessera_v2_medium_ft_bnd": NAME}
EXTENTS = {"oracle": "declared parcels (oracle)", "worldcover": "WorldCover 30 + 40", "chip": "whole chip"}
VARIANTS = {"extent": "extent alone (floor)", "semantic": "crop-type map", "boundary@0.3": "boundary, tau 0.3",
            "boundary@0.5": "boundary, tau 0.5", "boundary@0.7": "boundary, tau 0.7",
            "combined@0.5": "boundary + class, tau 0.5"}
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


def load(chips: str) -> pd.DataFrame:
    rows = []
    for arm in ARMS:
        for f in sorted((REPO / "results" / NAME / chips / arm).glob("P*_draw*_seed*/extent_scores.json")):
            s = json.loads(f.read_text())
            pct = float(f.parent.name.split("_")[0][1:])
            for k, q in s["variants"].items():
                ext, var = (None, "oracle") if k == "oracle" else k.split("/")
                row = {"arm": arm, "pct": pct, "extent": ext, "variant": var,
                       **{m: q[m] for m in ("pq", "sq", "rq", "pq_agnostic", "sq_agnostic", "rq_agnostic",
                                            "tp_agnostic", "fp_agnostic", "fn_agnostic", "predicted_parcels",
                                            "true_parcels", "dropped_void", "fp_on_parcels")}}
                for place, n in (q.get("fp_on_background") or {}).items():
                    row[f"fp_{place}"] = n
                rows.append(row)
            for ext, c in s["extents"].items():
                rows.append({"arm": arm, "pct": pct, "extent": ext, "variant": "_cover", **c})
            bp = s.get("boundary_pixels", {})
            rows.append({"arm": arm, "pct": pct, "extent": None, "variant": "_boundary", **bp,
                         "pixel_macro_f1": s["source"]["pixel_macro_f1"]})
    return pd.DataFrame(rows)


#: The extents of the bar figure, the sensitivity variant of WorldCover among them.
BAR_EXTENTS = {"oracle": "declared parcels\n(oracle)", "worldcover": "WorldCover\n30 + 40",
               "worldcover_30_40_90": "WorldCover\n30 + 40 + 90", "chip": "whole chip"}


def fig_extents(df: pd.DataFrame, path: Path, pct: float = 100) -> None:
    """Grouped bars: PQ of the boundary parcels (tau 0.5) per extent and arm, each bar's floor as a black tick."""
    from matplotlib.lines import Line2D

    d = df[(df.pct == pct) & (df.variant == "boundary@0.5")]
    fl = df[(df.pct == pct) & (df.variant == "extent")]
    exts = [e for e in BAR_EXTENTS if not d[d.extent == e].empty]
    arms = [a for a in ARMS if not d[d.arm == a].empty]
    w = 0.8 / len(arms)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0))
    for ax, (metric, ylab) in zip(axes, (("pq_agnostic", "PQ, class-agnostic"), ("pq", "PQ, class-aware"))):
        for a, arm in enumerate(arms):
            label, slot = ARMS[arm]
            for j, ext in enumerate(exts):
                q = d[(d.arm == arm) & (d.extent == ext)]
                if q.empty:
                    continue
                x = j + (a - (len(arms) - 1) / 2) * w
                v = float(q[metric].iloc[0])
                ax.bar(x, v, w, color=SLOTS[slot], edgecolor=SURFACE, linewidth=1.5,
                       label=f"{label}, boundary head" if j == 0 else None)
                f = fl[(fl.arm == arm) & (fl.extent == ext)]
                top = v
                if not f.empty:
                    fv = float(f[metric].iloc[0])
                    ax.hlines(fv, x - w / 2 + 0.02, x + w / 2 - 0.02, color=INK, lw=2)
                    top = max(v, fv)
                ax.text(x, top + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
        ax.set_xticks(range(len(exts)), [BAR_EXTENTS[e] for e in exts], fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_ylim(0, ax.get_ylim()[1] * 1.08)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.append(Line2D([], [], color=INK, lw=2))
    labels.append("floor: the extent's own regions, no boundary")
    axes[0].legend(handles, labels, loc="upper right", fontsize=8)
    axes[0].set_title("Class-agnostic: is the parcel there?")
    axes[1].set_title("Class-aware: macro mean over the classes")
    fig.suptitle(f"Boundary parcels (tau 0.5) in each extent, K = {pct:g} %", x=0.01, ha="left", color=INK,
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_variants(df: pd.DataFrame, path: Path) -> None:
    ks = sorted(df["pct"].unique(), reverse=True)
    fig, axes = plt.subplots(len(ks), len(EXTENTS), figsize=(3.4 * len(EXTENTS), 2.6 * len(ks)), sharey=True,
                             squeeze=False)
    names = list(VARIANTS)
    for i, k in enumerate(ks):
        for j, (ext, title) in enumerate(EXTENTS.items()):
            ax = axes[i, j]
            w = 0.8 / len(ARMS)
            for a, (arm, (label, slot)) in enumerate(ARMS.items()):
                d = df[(df.arm == arm) & (df.extent == ext) & (df.pct == k)].set_index("variant")
                if d.empty:
                    continue
                vals = [d.loc[v, "pq_agnostic"] if v in d.index else np.nan for v in names]
                ax.bar(np.arange(len(names)) + (a - (len(ARMS) - 1) / 2) * w, vals, w - 0.03, color=SLOTS[slot],
                       label=label)
            ax.set_xticks(range(len(names)), [VARIANTS[v] for v in names], rotation=35, ha="right", fontsize=7.5)
            ax.set_title(title if len(ks) == 1 else f"{title}, K = {k:g} %", fontsize=9.5)
            if j == 0:
                ax.set_ylabel("PQ, class-agnostic")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_fp(df: pd.DataFrame, path: Path, pct: float = 100) -> None:
    d = df[(df.variant == "boundary@0.5") & (df.pct == pct) & df.extent.isin(list(EXTENTS))]
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
        ax.text(left + 150, r, f"{int(left):,} FP; {int(q['dropped_void']):,} dropped as void", va="center",
                fontsize=8, color=INK_2)
    ax.set_yticks(range(len(rows)), [f"{EXTENTS[e]}, {ARMS[a][0]}" for a, e in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"false parcels of the boundary head (tau 0.5), K = {pct:g} %")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.18 - 0.02 * len(rows)))
    ax.set_title("Where the false parcels lie: on the true parcels, or on background by WorldCover class")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _outline(seg: np.ndarray) -> np.ndarray:
    o = np.zeros(seg.shape, bool)
    o[:-1] |= seg[:-1] != seg[1:]
    o[1:] |= seg[:-1] != seg[1:]
    o[:, :-1] |= seg[:, :-1] != seg[:, 1:]
    o[:, 1:] |= seg[:, :-1] != seg[:, 1:]
    return o


def _status(seg, true, void):
    """Per segment id: 1 TP (class-agnostic), 2 FP, 3 dropped as void."""
    ti, si, _, n_all, n_void = px.match(true, seg, void)
    st = np.zeros(len(n_all), np.int64)
    ids = np.nonzero(n_all)[0]
    ids = ids[ids > 0]
    st[ids] = np.where(n_void[ids] * 2 > n_all[ids], 3, 2)
    st[si] = 1
    return st


def fig_examples(chips: str, folder: Path, n: int = 2) -> list[Path]:
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    cells = {arm: REPO / "results" / SOURCES[arm] / chips / arm / "P100_draw0_seed0" for arm in ARMS}
    cells = {a: c for a, c in cells.items() if (c / "predictions_test.npz").exists()}
    if not cells:
        return []
    data = {}
    for a, c in cells.items():
        with np.load(c / "predictions_test.npz") as z:   # each array once: z[key] decompresses it again
            data[a] = {k: z[k] for k in ("ids", "pred", "boundary", "mask")}
    first = next(iter(data.values()))
    ids = [str(c) for c in first["ids"]]
    wc_all = np.load(REPO / "data/worldcover" / chips.replace("_mini", "") / "worldcover.npz")
    chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
    # Chips with many parcels and a fair share of non-agricultural land, so both extents matter.
    scored = []
    for i in range(0, len(ids), 7):
        m = first["mask"][i]
        wc = wc_all[ids[i]]
        n_par = len(np.unique(read_parcels(chips_dir, ids[i])[m >= 0]))
        agri = np.isin(wc, px.WORLDCOVER).mean()
        if 0.35 < agri < 0.75 and (m >= 0).mean() > 0.25:
            scored.append((n_par, i))
    picks = [i for _, i in sorted(scored, reverse=True)[:n]]
    out = []
    colours = {1: "#2a78d6", 2: "#eb6834", 3: "#c3c2b7"}
    for p_i, i in enumerate(picks, start=1):
        cid = ids[i]
        parcels = read_parcels(chips_dir, cid)
        mask = first["mask"][i].astype(np.int64)
        E, true, _, void = px.truth(mask, parcels)
        wc = wc_all[cid]
        panels = [("WorldCover 30 + 40 (green); true parcels", None, None)]
        for arm in cells:
            j = list(map(str, data[arm]["ids"])).index(cid)
            pred, prob = data[arm]["pred"][j], data[arm]["boundary"][j]
            for ext in ("worldcover", "chip"):
                X = px.extent_of(ext, mask, parcels, wc)
                seg = px.from_boundary(X, prob.astype(np.float32) / 255.0, 0.5)
                panels.append((f"{ARMS[arm][0]}, {EXTENTS[ext]}", seg, _status(seg, true, void)))
        fig, axes = plt.subplots(1, len(panels), figsize=(3.1 * len(panels), 3.5))
        for ax, (title, seg, st) in zip(axes, panels):
            img = np.ones(mask.shape + (3,)) * np.array(matplotlib.colors.to_rgb("#f0efec"))
            if seg is None:
                img[np.isin(wc, px.WORLDCOVER)] = matplotlib.colors.to_rgb("#bfe3c9")
                img[void] = matplotlib.colors.to_rgb("#c3c2b7")
            else:
                for s, c in colours.items():
                    img[np.isin(seg, np.nonzero(st == s)[0]) & (seg > 0)] = matplotlib.colors.to_rgb(c)
                img[_outline(seg) & (seg > 0)] = matplotlib.colors.to_rgb(SURFACE)
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
        fig.suptitle(f"Test chip {cid}, K = 100 %: boundary parcels (tau 0.5), true parcels outlined in black",
                     x=0.01, ha="left", fontsize=10)
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


def tables(df: pd.DataFrame, chips: str) -> str:
    out = []
    lab = {a: v[0] for a, v in ARMS.items()}
    b = df[df.variant == "boundary@0.5"]
    for ext, title in {**EXTENTS, "worldcover_30_40_90": "WorldCover 30 + 40 + 90"}.items():
        d = b[b.extent == ext]
        if d.empty:
            continue
        fl = df[(df.extent == ext) & (df.variant == "extent")].set_index(["arm", "pct"])
        sem = df[(df.extent == ext) & (df.variant == "semantic")].set_index(["arm", "pct"])
        t = pd.DataFrame({
            "Arm": d.arm.map(lab),
            "floor PQ agn.": [fl.loc[(a, k), "pq_agnostic"] for a, k in zip(d.arm, d.pct)],
            "crop map PQ agn.": [sem.loc[(a, k), "pq_agnostic"] for a, k in zip(d.arm, d.pct)],
            "PQ agn.": d.pq_agnostic.values, "SQ agn.": d.sq_agnostic.values, "RQ agn.": d.rq_agnostic.values,
            "PQ": d.pq.values, "TP": d.tp_agnostic.astype(int).values, "FP": d.fp_agnostic.astype(int).values,
            "FN": d.fn_agnostic.astype(int).values, "parcels": d.predicted_parcels.astype(int).values,
            "dropped": d.dropped_void.astype(int).values})
        out.append(f"**{title}**: boundary parcels at tau 0.5\n\n" + _md(t.sort_values("Arm")))
    tau = df[df.variant.isin(["boundary@0.3", "boundary@0.5", "boundary@0.7"]) & df.extent.isin(list(EXTENTS))]
    if not tau.empty:
        t = tau.pivot_table(index=["arm", "extent"], columns="variant", values="pq_agnostic").reset_index()
        t["arm"] = t.arm.map(lab)
        out.append("**Class-agnostic PQ against the threshold**\n\n" + _md(t))
    c = df[df.variant == "_cover"].drop_duplicates(["extent"])
    if not c.empty:
        t = c[["extent", "true_pixels_inside", "share_of_chip", "background_share_of_extent",
               "void_share_of_extent"]]
        out.append("**The extents on the test chips**\n\n" + _md(t))
    bp = df[df.variant == "_boundary"]
    if not bp.empty:
        t = pd.DataFrame({"Arm": bp.arm.map(lab),
                          "pixel Macro-F1": bp.pixel_macro_f1.values, "boundary F1 in E": bp.f1.values,
                          "precision": bp.precision.values, "recall": bp.recall.values})
        out.append("**The two heads** (boundary pixels at 0.5 inside E; crop-type pixel Macro-F1)\n\n" +
                   _md(t.sort_values("Arm")))
    stats = REPO / "results" / NAME / chips / "worldcover_stats.json"
    if stats.exists():
        s = json.loads(stats.read_text())
        out.append("**WorldCover on the test chips**: parcel pixels inside 30 + 40: "
                   f"{s['parcels_inside_wc30_40']:.3f}, with 90: {s['parcels_inside_wc30_40_90']:.3f}; share of "
                   f"WorldCover grassland on background {s['wc30_on_background_share']:.3f}, of cropland "
                   f"{s['wc40_on_background_share']:.3f}.")
    return "\n\n".join(out) + "\n"


def main() -> None:
    import matplotlib.patches  # noqa: F401

    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    a = ap.parse_args()
    df = load(a.chips)
    df = df[df.pct == 100]                     # the experiment's one budget
    if df.empty:
        raise SystemExit("no extent_scores.json yet: run score.py")
    folder = HERE / "figures" if a.chips == "EE_2021_rev" else HERE / "figures" / a.chips
    folder.mkdir(parents=True, exist_ok=True)
    fig_extents(df, folder / "extents.png")
    fig_variants(df, folder / "variants.png")
    fig_fp(df, folder / "fp_where.png")
    fig_examples(a.chips, folder)
    t = tables(df, a.chips)
    (folder / "tables.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
