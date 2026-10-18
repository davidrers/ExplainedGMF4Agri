"""Figures and tables of the parcel delineation experiment, for its REPORT.md.

Reads ``results/<name>/<chip set>/`` as ``run.py`` writes it, and, for the comparison with the crop-type-only fits,
the main workflow's ``results/kshot/<chip set>/``, whose ``oracle`` and ``semantic`` parcels it scores once into
``results/<name>/<chip set>/kshot_panoptic.json``. Writes into ``figures/`` beside this file:

* ``pq_curves.png``: PQ of the boundary parcels, class-aware and class-agnostic, and the boundary pixel F1,
  against K per arm;
* ``geometry_cost.png``: RQ (the parcel Macro-F1 at IoU > 0.5) with the true parcels against the boundary parcels,
  per arm and K: what is lost when the parcels come from the model;
* ``variants.png``: class-agnostic PQ of the parcels from the crop-type map alone, the boundary head and both,
  per arm against K;
* ``joint_vs_kshot.png``: joint training minus the crop-type-only fit of the main workflow, same draw and seed,
  in pixel Macro-F1, in RQ with the true parcels and in class-agnostic PQ of the crop-type map's parcels;
* ``tau.png``: class-agnostic PQ of the boundary parcels against the threshold on the boundary probability;
* ``validation_curves.png``: validation crop-type and boundary loss per epoch, with the checkpoint epochs;
* ``examples.png``: test chips: a July composite, the true parcels, and the boundary parcels of every arm at the
  largest K;
* ``tables.md``: the tables of REPORT.md in Markdown, also printed.

    poetry run python experiments/2026-10-18_parcel_delineation/report.py --chips EE_2021_rev
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

import panoptic  # noqa: E402

#: Arm -> (short label, categorical slot). The legend follows this order and the colour the slot.
ARMS: dict[str, tuple[str, int]] = {
    "terramind_v1_large_s2s1": ("TerraMind S2+S1", 1),
    "prithvi_eo_v2_600_tl": ("Prithvi 600M", 2),
    "thor_v1_large_s2s1": ("THOR S2+S1", 3),
    "tessera_v2_medium": ("TESSERA v2", 4),
    "alphaearth_v1": ("AlphaEarth", 5),
    "unet_r18_scratch_s2s1": ("U-Net baseline", 6),
}
#: The crop-type-only fits of the main workflow, on the same draws and seeds.
REFERENCE = "kshot"
#: Parcel variants and their labels; the slots of the variants in variants.png. ``extent`` (no geometry predicted,
#: the floor) is drawn as a grey reference line.
VARIANTS = {"semantic": ("crop-type map alone", 7), "boundary@0.5": ("boundary head", 1),
            "combined@0.5": ("boundary head and class changes", 2)}

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "h", "P", "X"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
DIVERGING = LinearSegmentedColormap.from_list("gain", ["#e34948", "#f0efec", "#2a78d6"])
SEQUENTIAL = LinearSegmentedColormap.from_list("level", ["#cde2fb", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 11,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "xtick.labelcolor": INK_2, "ytick.labelcolor": INK_2,
    "font.size": 9.5, "legend.frameon": False, "legend.labelcolor": INK,
})


# ---------------------------------------------------------------------------------------------------------- data

def _key(r: dict) -> dict:
    return {"arm": r["run"], "pct": r["label_budget"]["pct"], "draw": r["label_budget"]["draw_seed"],
            "seed": r["seed"]}


def _flat(pm: dict) -> dict:
    out = {}
    for variant, q in pm.get("variants", {}).items():
        for m in ("pq", "sq", "rq", "pq_agnostic", "sq_agnostic", "rq_agnostic"):
            out[f"{variant}/{m}"] = q.get(m)
        out[f"{variant}/n_pred"] = q.get("predicted_parcels")
        out["true_parcels"] = q.get("true_parcels")
    out["boundary_f1"] = pm.get("boundary_pixels", {}).get("f1")
    return out


def _ckpt_epoch(cell: Path) -> int | None:
    logs = sorted(cell.glob("logs/version_*/metrics.csv"))
    if not logs:
        return None
    m = pd.read_csv(logs[-1])
    v = m.dropna(subset=["val/loss"]) if "val/loss" in m else None
    return None if v is None or v.empty else int(v.loc[v["val/loss"].idxmin(), "epoch"])


def load(out: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """One row per fit of the experiment, the validation curves (long) and the run facts."""
    rows, curves = [], []
    facts = {"split": None, "chips": None, "commits": set(), "machines": set()}
    for path in sorted(out.glob("*/P*_draw*_seed*/results.json")):
        r = json.loads(path.read_text())
        if "panoptic_metrics" not in r:
            continue
        key = _key(r)
        m = r["test_metrics"]
        rows.append({**key, "macro_f1": m.get("test/F1_Score"),
                     "parcel_macro_f1": r.get("parcel_metrics", {}).get("macro_f1"),
                     "test_boundary_f1": m.get("test/boundary_F1"),
                     "ckpt_epoch": _ckpt_epoch(path.parent), "fit_h": r["fit_seconds"] / 3600,
                     "parcels": sum(v["drawn"] for v in (r["label_budget"].get("support") or {}).values()),
                     "train_chips": r["chips"]["train"], **_flat(r["panoptic_metrics"])})
        logs = sorted(path.parent.glob("logs/version_*/metrics.csv"))
        if logs:
            lm = pd.read_csv(logs[-1])
            for col in ("val/semantic_loss", "val/boundary_loss", "val/loss"):
                if col in lm:
                    v = lm.dropna(subset=[col]).groupby("epoch")[col].last()
                    curves += [{**key, "metric": col, "epoch": int(e), "value": float(x)} for e, x in v.items()]
        sp = r.get("split_protocol", {})
        facts["split"], facts["chips"] = sp.get("split"), sp.get("chips")
        prov = r.get("provenance", {})
        facts["commits"].add((prov.get("git_commit") or "")[:8])
        facts["machines"].add(prov.get("machine"))
    return pd.DataFrame(rows), pd.DataFrame(curves), facts


def load_reference(out: Path, chips: str) -> pd.DataFrame:
    """The crop-type-only fits on the same chip set, with their oracle and semantic parcels scored (cached)."""
    ref = REPO / "results" / REFERENCE / chips
    cache_file = out / "kshot_panoptic.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    rows = []
    for path in sorted(ref.glob("*/P*_draw*_seed*/results.json")):
        cell = path.parent
        if not (cell / "predictions_test.npz").exists():
            continue
        r = json.loads(path.read_text())
        tag = f"{cell.parent.name}/{cell.name}"
        if tag not in cache:
            print(f"scoring {REFERENCE}/{chips}/{tag}")
            cache[tag] = panoptic.score_predictions(cell / "predictions_test.npz",
                                                    REPO / "data/eurocrops_chips" / chips / "chips",
                                                    r["class_names"], workers=16)
            cache_file.write_text(json.dumps(cache, indent=1))
        rows.append({**_key(r), "macro_f1": r["test_metrics"].get("test/F1_Score"),
                     "parcel_macro_f1": r.get("parcel_metrics", {}).get("macro_f1"), **_flat(cache[tag])})
    return pd.DataFrame(rows)


def arm_styles(arms: list[str]) -> dict[str, tuple[str, str, str]]:
    order = [a for a in ARMS if a in arms] + sorted(set(arms) - set(ARMS))
    used = {slot for a, (_, slot) in ARMS.items() if a in arms}
    free = iter(s for s in range(1, len(SLOTS) + 1) if s not in used)
    out = {}
    for a in order:
        label, slot = ARMS.get(a, (a, None))
        slot = slot or next(free)
        out[a] = (label, SLOTS[slot - 1], MARKERS[slot - 1])
    return out


def paired(df: pd.DataFrame, ref: pd.DataFrame, metric: str) -> pd.DataFrame:
    """``metric`` of the joint fit minus the crop-type-only fit, per arm and K, on the same draw and seed."""
    a = df.set_index(["arm", "pct", "draw", "seed"])[metric]
    b = ref.set_index(["arm", "pct", "draw", "seed"])[metric]
    return (a - b).dropna().groupby(["arm", "pct"]).mean().unstack("pct")


# ------------------------------------------------------------------------------------------------------- figures

def _budget_axis(ax, budgets: list[float]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(budgets, [f"{b:g}" for b in budgets])
    ax.minorticks_off()
    ax.set_xlabel("K, % of trainable parcels per class")


def _end_labels(ax, items: list[tuple[float, float, str, str]], gap: float) -> None:
    items = sorted(items, key=lambda t: t[1])
    ys = []
    for _, y, _, _ in items:
        ys.append(max(y, ys[-1] + gap) if ys else y)
    shift = max(0.0, ys[-1] - ax.get_ylim()[1] + gap / 2) if ys else 0.0
    for (x, _, text, colour), y in zip(items, ys, strict=True):
        ax.plot([x * 1.03, x * 1.12], [y - shift, y - shift], color=colour, lw=2, clip_on=False)
        ax.annotate(text, (x * 1.14, y - shift), va="center", fontsize=8.5, color=INK, annotation_clip=False)


def fig_curves(df: pd.DataFrame, styles: dict, metrics: list[tuple[str, str]], path: Path, note: str) -> None:
    budgets = sorted(df["pct"].unique())
    fig, axes = plt.subplots(1, len(metrics), figsize=(5.2 * len(metrics), 4.3), squeeze=False)
    for ax, (metric, title) in zip(axes[0], metrics, strict=True):
        ends = []
        for arm, (label, colour, marker) in styles.items():
            g = df[df["arm"] == arm].groupby("pct")[metric].mean().dropna()
            if g.empty:
                continue
            ax.plot(g.index, g.values, color=colour, lw=2, marker=marker, ms=7, mec=SURFACE, mew=1.5, label=label,
                    zorder=3)
            ends.append((max(budgets) * 1.3, g.values[-1], label, colour))
        _budget_axis(ax, budgets)
        ax.set_xlim(min(budgets) / 1.3, max(budgets) * 1.3)
        ax.set_title(title)
        lo, hi = ax.get_ylim()
        _end_labels(ax, ends, gap=(hi - lo) * 0.07)
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=min(len(styles), 6),
               bbox_to_anchor=(0.5, 1.03))
    fig.text(0.01, -0.03, note, color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 0.92, 0.94))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_geometry_cost(df: pd.DataFrame, styles: dict, path: Path) -> None:
    """Dumbbells: RQ with the true parcels and with the boundary parcels, per arm, one panel per K."""
    budgets = sorted(df["pct"].unique(), reverse=True)
    arms = list(styles)
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.2 * len(budgets) + 1.6, 0.42 * len(arms) + 1.6),
                             sharey=True, squeeze=False)
    for ax, k in zip(axes[0], budgets, strict=True):
        g = df[df["pct"] == k].groupby("arm")[["oracle/rq", "boundary@0.5/rq"]].mean()
        for i, arm in enumerate(arms):
            if arm not in g.index:
                continue
            o, b = g.loc[arm, "oracle/rq"], g.loc[arm, "boundary@0.5/rq"]
            ax.plot([b, o], [i, i], color=BASELINE, lw=2.5, zorder=1, solid_capstyle="round")
            ax.scatter([o], [i], s=56, color=INK_2, edgecolor=SURFACE, lw=1.5, zorder=3,
                       label="true parcels" if i == 0 else None)
            ax.scatter([b], [i], s=56, color=styles[arm][1], marker=styles[arm][2], edgecolor=SURFACE, lw=1.5,
                       zorder=3)
            ax.annotate(f"{b - o:+.3f}", (max(b, o) + 0.012, i), ha="left", va="center", fontsize=8.5, color=INK)
        ax.set_yticks(range(len(arms)), [styles[a][0] for a in arms])
        ax.invert_yaxis()
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", visible=True)
        lo = np.nanmin(g.values) if not g.empty else 0
        ax.set_xlim(max(0, lo - 0.03), min(1.08, np.nanmax(g.values) + 0.09) if not g.empty else 1)
        ax.set_title(f"K = {k:g} %")
        ax.set_xlabel("RQ, parcel Macro-F1 at IoU > 0.5")
    fig.text(0.01, -0.04, "Grey dot: true parcels, only the class predicted (oracle). Coloured mark: parcels from the "
             "boundary head, threshold 0.5. Value: the difference, the parcel F1 lost to the predicted geometry.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_variants(df: pd.DataFrame, styles: dict, path: Path) -> None:
    budgets = sorted(df["pct"].unique())
    arms = list(styles)
    ncol = 3
    nrow = int(np.ceil(len(arms) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.1 * ncol, 3.2 * nrow), sharey=True, squeeze=False)
    for ax, arm in zip(axes.flat, arms):
        g = df[df["arm"] == arm].groupby("pct").mean(numeric_only=True)
        if "extent/pq_agnostic" in g:
            ax.plot(g.index, g["extent/pq_agnostic"], color=MUTED, lw=1.5, ls="--", marker="_", ms=12, mew=2,
                    label="extent alone, no geometry predicted")
        for v, (label, slot) in VARIANTS.items():
            col = f"{v}/pq_agnostic"
            if col in g:
                ax.plot(g.index, g[col], color=SLOTS[slot - 1], marker=MARKERS[slot - 1], lw=2, ms=6, mec=SURFACE,
                        mew=1.2, label=label)
        _budget_axis(ax, budgets)
        ax.set_title(styles[arm][0])
    for ax in axes.flat[len(arms):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("PQ, class-agnostic")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.text(0.01, -0.02, "Parcels inside the true parcel extent; class-agnostic PQ scores the geometry alone. Dashed: "
             "the extent's connected regions, which find the parcels isolated by non-parcel land for free.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _heat(ax, data: pd.DataFrame, cmap, vmin: float, vmax: float, fmt: str):
    cmap = cmap.copy()
    cmap.set_bad(SURFACE)
    im = ax.imshow(np.ma.masked_invalid(data.values.astype(float)), cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    for (i, j), v in np.ndenumerate(data.values.astype(float)):
        if not np.isfinite(v):
            ax.text(j, i, "-", ha="center", va="center", fontsize=8, color=MUTED)
            continue
        t = (v - vmin) / (vmax - vmin)
        dark = abs(2 * t - 1) > 0.6 if vmin < 0 < vmax else t > 0.55
        v = v if abs(v) >= 0.0005 else 0.0
        ax.text(j, i, format(v, fmt), ha="center", va="center", fontsize=8, color=SURFACE if dark else INK)
    ax.set_xticks(range(data.shape[1]), data.columns)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    return im


def fig_joint(df: pd.DataFrame, ref: pd.DataFrame, styles: dict, path: Path) -> None:
    panels = [("macro_f1", "Pixel Macro-F1"), ("oracle/rq", "RQ, true parcels"),
              ("semantic/pq_agnostic", "PQ agnostic, crop-type map")]
    diffs = [(t, paired(df, ref, m)) for m, t in panels]
    lim = max(np.nanmax(np.abs(d.values)) for _, d in diffs if not d.empty) or 0.01
    budgets = sorted(df["pct"].unique(), reverse=True)
    arms = list(styles)
    fig, axes = plt.subplots(1, len(panels), figsize=(2.6 * len(panels) + 3.0, 0.4 * len(arms) + 1.7), sharey=True,
                             squeeze=False)
    for ax, (title, d) in zip(axes[0], diffs, strict=True):
        d = d.reindex(index=arms, columns=budgets)
        d.columns = [f"{k:g} %" for k in budgets]
        im = _heat(ax, d, DIVERGING, -lim, lim, "+.3f")
        ax.set_title(title)
    axes[0][0].set_yticks(range(len(arms)), [styles[a][0] for a in arms])
    fig.colorbar(im, ax=axes[0].tolist(), shrink=0.7, label="Joint minus crop-type only")
    fig.text(0.01, -0.02, "Each cell: the fit with the boundary head minus the main workflow's fit on the same draw "
             "and seed; columns are K.", color=MUTED, fontsize=8.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_tau(df: pd.DataFrame, styles: dict, path: Path) -> None:
    budgets = sorted(df["pct"].unique(), reverse=True)
    fig, axes = plt.subplots(1, len(budgets), figsize=(4.0 * len(budgets), 3.6), sharey=True, squeeze=False)
    for ax, k in zip(axes[0], budgets, strict=True):
        g = df[df["pct"] == k].groupby("arm").mean(numeric_only=True)
        for arm, (label, colour, marker) in styles.items():
            if arm not in g.index:
                continue
            ys = [g.loc[arm, f"boundary@{t:g}/pq_agnostic"] for t in panoptic.TAUS]
            ax.plot(panoptic.TAUS, ys, color=colour, marker=marker, lw=2, ms=6, mec=SURFACE, mew=1.2, label=label)
        ax.set_xticks(panoptic.TAUS)
        ax.set_xlabel("Threshold on the boundary probability")
        ax.set_title(f"K = {k:g} %")
    axes[0][0].set_ylabel("PQ, class-agnostic")
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=min(len(styles), 6),
               bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def fig_validation(df: pd.DataFrame, curves: pd.DataFrame, styles: dict, path: Path) -> None:
    budgets = sorted(df["pct"].unique(), reverse=True)
    cols = (("val/semantic_loss", "Validation crop-type loss"), ("val/boundary_loss", "Validation boundary loss"))
    fig, axes = plt.subplots(len(budgets), 2, figsize=(10, 2.6 * len(budgets) + 0.6), sharex=True, squeeze=False)
    for row, k in zip(axes, budgets, strict=True):
        for ax, (metric, title) in zip(row, cols, strict=True):
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
    fig.legend(*axes[0][0].get_legend_handles_labels(), loc="upper center", ncol=min(len(styles), 6),
               bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, -0.01, "Markers: the epoch of lowest validation loss (crop-type plus boundary), the checkpoint "
             "tested.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _outline(seg: np.ndarray) -> np.ndarray:
    """The pixels of a labelled map on the edge of their label (4-neighbourhood), 0 not counted as a label."""
    e = np.zeros(seg.shape, bool)
    v = seg[:-1] != seg[1:]
    e[:-1] |= v & (seg[:-1] > 0)
    e[1:] |= v & (seg[1:] > 0)
    h = seg[:, :-1] != seg[:, 1:]
    e[:, :-1] |= h & (seg[:, :-1] > 0)
    e[:, 1:] |= h & (seg[:, 1:] > 0)
    return e


def _show_parcels(ax, seg: np.ndarray, extent: np.ndarray, title: str) -> None:
    img = np.ones(seg.shape + (3,))
    img[extent] = matplotlib.colors.to_rgb("#e3eefb")
    img[_outline(seg) & extent] = matplotlib.colors.to_rgb("#0d366b")
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def pick_chips(chips_dir: Path, ids: list[str], n: int = 3) -> list[str]:
    """Test chips with the most contacts between same-class parcels: where delineation matters most."""
    from gfm4agri.benchmark.parcel_metrics import read_parcels
    import rasterio

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


def fig_examples(out: Path, chips: str, styles: dict, path: Path) -> None:
    import rasterio

    from gfm4agri.benchmark.parcel_metrics import read_parcels

    chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
    cells = {}
    for arm in styles:
        found = sorted(out.glob(f"{arm}/P*_draw0_seed0/predictions_test.npz"),
                       key=lambda p: -float(p.parent.name[1:].split("_")[0]))
        if found:
            cells[arm] = np.load(found[0])
    if not cells:
        return
    ids = [str(c) for c in next(iter(cells.values()))["ids"]]
    chosen = pick_chips(chips_dir, ids)
    manifest = json.loads((REPO / "data/eurocrops_chips" / chips / "manifest.json").read_text())
    bands = manifest["imagery"]["band_names"]
    fig, axes = plt.subplots(len(chosen), 2 + len(cells), figsize=(2.3 * (2 + len(cells)), 2.5 * len(chosen)),
                             squeeze=False)
    for row, cid in zip(axes, chosen, strict=True):
        parcels = read_parcels(chips_dir, cid)
        with rasterio.open(chips_dir / f"{cid}_merged.tif") as s:
            # Band index = month index x bands per month + band index (the manifest's layout).
            july = 6 * len(bands)
            rgb = np.stack([s.read(july + bands.index(b) + 1) for b in ("RED", "GREEN", "BLUE")], -1)
            rgb = rgb.astype(np.float32)
        rgb = np.clip(rgb / np.nanpercentile(rgb, 98), 0, 1)
        row[0].imshow(np.nan_to_num(rgb), interpolation="nearest")
        row[0].set_title(f"{cid}, July", fontsize=9)
        row[0].axis("off")
        first = next(iter(cells.values()))
        i = ids.index(cid)
        mask = first["mask"][i]
        extent = (mask >= 0) & (parcels > 0)
        true = panoptic.pieces(np.where(extent, parcels, 0))
        _show_parcels(row[1], true, extent, "true parcels")
        for ax, (arm, d) in zip(row[2:], cells.items()):
            j = [str(c) for c in d["ids"]].index(cid)
            seg = panoptic.from_boundary(extent, d["boundary"][j].astype(np.float32) / 255.0, panoptic.TAU)
            _show_parcels(ax, seg, extent, styles[arm][0])
    fig.text(0.01, 0.0, "Inside the true parcel extent (light); dark lines are parcel outlines. Boundary parcels at "
             "threshold 0.5, the largest K fitted. Chips with the most contacts between parcels of one class.",
             color=MUTED, fontsize=8.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# -------------------------------------------------------------------------------------------------------- tables

def _table(df: pd.DataFrame, styles: dict, cols: list[tuple[str, str]], fmt: str = ".3f") -> str:
    budgets = sorted(df["pct"].unique(), reverse=True)
    head = "| Arm | " + " | ".join(f"{t}, K = {k:g} %" for k in budgets for _, t in cols) + " |\n|---|" \
        + "---|" * (len(budgets) * len(cols)) + "\n"
    g = df.groupby(["arm", "pct"]).mean(numeric_only=True)
    rows = []
    for arm, (label, _, _) in styles.items():
        cells = []
        for k in budgets:
            for c, _ in cols:
                v = g[c].get((arm, k), np.nan) if c in g else np.nan
                cells.append("" if pd.isna(v) else format(v, fmt))
        rows.append(f"| {label} | " + " | ".join(cells) + " |")
    return head + "\n".join(rows)


def tables(df: pd.DataFrame, ref: pd.DataFrame, styles: dict, facts: dict) -> str:
    out = []
    out.append("**Panoptic quality of the boundary parcels** (threshold 0.5), class-aware\n\n"
               + _table(df, styles, [("boundary@0.5/pq", "PQ"), ("boundary@0.5/sq", "SQ"),
                                     ("boundary@0.5/rq", "RQ")]))
    out.append("**Class-agnostic PQ**, the geometry alone: the extent alone (floor), crop-type map, boundary head, "
               "both\n\n"
               + _table(df, styles, [("extent/pq_agnostic", "floor"), ("semantic/pq_agnostic", "map"),
                                     ("boundary@0.5/pq_agnostic", "head"), ("combined@0.5/pq_agnostic", "both")]))
    d = df.copy()
    d["gap"] = d["boundary@0.5/rq"] - d["oracle/rq"]
    out.append("**Parcel Macro-F1 at IoU > 0.5 (RQ)**: true parcels, boundary parcels, the difference\n\n"
               + _table(d, styles, [("oracle/rq", "true"), ("boundary@0.5/rq", "pred."), ("gap", "diff.")]))
    out.append("**Boundary pixels**: test F1 of the boundary head at 0.5, against the dense target\n\n"
               + _table(df, styles, [("boundary_f1", "F1")]))
    out.append("**Threshold**: class-agnostic PQ of the boundary parcels at 0.3, 0.5, 0.7\n\n"
               + _table(df, styles, [(f"boundary@{t:g}/pq_agnostic", f"{t:g}") for t in panoptic.TAUS]))
    if not ref.empty:
        j = df.set_index(["arm", "pct", "draw", "seed"])
        r = ref.set_index(["arm", "pct", "draw", "seed"])
        both = j.join(r, rsuffix="_ref", how="inner").reset_index()
        for m in ("macro_f1", "parcel_macro_f1", "oracle/rq", "semantic/pq_agnostic"):
            both[f"{m}_diff"] = both[m] - both[f"{m}_ref"]
        out.append("**Joint against crop-type-only training** (main workflow, same draw and seed): pixel Macro-F1 "
                   "crop-type only, joint, difference; parcel Macro-F1 of D28, difference\n\n"
                   + _table(both, styles, [("macro_f1_ref", "pixel, ref."), ("macro_f1", "pixel, joint"),
                                           ("macro_f1_diff", "diff."), ("parcel_macro_f1_diff", "D28 diff.")]))
        out.append("**The crop-type-only fits' own parcels**: class-agnostic PQ of their crop-type map, and RQ with "
                   "the true parcels\n\n"
                   + _table(ref, styles, [("semantic/pq_agnostic", "PQ agn., map"), ("oracle/rq", "RQ, true")]))
    chips = facts.get("chips") or {}
    budget = df.groupby("pct")[["train_chips", "parcels"]].first().sort_index(ascending=False)
    n_true = df["true_parcels"].dropna()
    out.append("**Run**\n\n"
               f"* Split {facts.get('split')}: {chips.get('training')} training, {chips.get('validation')} "
               f"validation, {chips.get('test')} test chips; {int(n_true.iloc[0]) if len(n_true) else '?':,} true "
               "parcel pieces in the test chips.\n"
               + "".join(f"* K = {k:g} %: {int(r.train_chips)} training chips, {int(r.parcels):,} parcels "
                         "(draw 0).\n" for k, r in budget.iterrows())
               + f"* {len(df)} fits, {df['fit_h'].sum():.1f} GPU hours of fitting; machines "
               f"{', '.join(sorted(m for m in facts['machines'] if m))}; commits "
               f"{', '.join(sorted(c for c in facts['commits'] if c))}.")
    return "\n\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chips", required=True, help="the chip set's name, e.g. EE_2021_rev")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    ap.add_argument("--name", default=None, help="the results' experiment name, by default experiment.yaml's")
    args = ap.parse_args()

    from gfm4agri.pipeline.config import load_experiment

    name = args.name or load_experiment(HERE / "experiment.yaml")["name"]
    results = REPO / "results" / name / args.chips
    df, curves, facts = load(results)
    if df.empty:
        raise SystemExit(f"no scored fits in {results}")
    ref = load_reference(results, args.chips)
    styles = arm_styles(sorted(df["arm"].unique()))
    args.out.mkdir(parents=True, exist_ok=True)
    fig_curves(df, styles, [("boundary@0.5/pq", "PQ, boundary parcels"),
                            ("boundary@0.5/pq_agnostic", "PQ class-agnostic, boundary parcels"),
                            ("boundary_f1", "Boundary pixel F1")], args.out / "pq_curves.png",
               "Parcels inside the true parcel extent of the test chips, boundary threshold 0.5; one fit per point.")
    fig_geometry_cost(df, styles, args.out / "geometry_cost.png")
    fig_variants(df, styles, args.out / "variants.png")
    if not ref.empty:
        fig_joint(df, ref, styles, args.out / "joint_vs_kshot.png")
    fig_tau(df, styles, args.out / "tau.png")
    if not curves.empty:
        fig_validation(df, curves, styles, args.out / "validation_curves.png")
    fig_examples(results, args.chips, styles, args.out / "examples.png")
    text = tables(df, ref, styles, facts)
    (args.out / "tables.md").write_text(text)
    print(text)
    print(f"figures and tables in {args.out}")


if __name__ == "__main__":
    main()
