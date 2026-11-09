"""Figures and tables of the border experiment: which arm draws the field borders best, at each budget, and what the
border head does to its crop map against the main workflow's.

Reads ``results/2026-11-09_borders_all_arms/<set>/<arm>/P<k>_draw0_seed0/results.json`` (scored by ``run.py``) and,
for the crop reference, the main workflow's cells ``results/kshot/<set>/<kshot arm>/P<k>_draw0_seed0/results.json``.
Writes into ``figures/``:

* ``borders_by_k.png``: border F1 at 1 px, PQ class-agnostic and PQ class-aware of the grown fields against K;
* ``crop_by_k.png``: crop pixel and parcel Macro-F1 against K, with the main workflow's crop-only fit dashed;
* ``border_types.png``: border recall at 1 px by what lies across the border, K = 100 %;
* ``training.png``: the validation object F1, the validation crop Macro-F1 and the crop loss weight per epoch, K = 100 %;
* ``tables.md``.

    poetry run python experiments/2026-11-09_borders_all_arms/report.py --chips EE_2021_rev
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

NAME = "2026-11-09_borders_all_arms"
#: Arm -> (label, categorical slot, the main workflow's arm); the slots of ``2026-10-18_parcel_delineation``.
ARMS = {"terramind_v1_large_s2s1": ("TerraMind S2+S1", 1, "terramind_v1_large_s2s1"),
        "prithvi_eo_v2_600_tl": ("Prithvi 600M", 2, "prithvi_eo_v2_600_tl"),
        "thor_v1_large_s2s1": ("THOR S2+S1", 3, "thor_v1_large_s2s1"),
        "tessera_v2_medium": ("TESSERA v2", 4, "tessera_v2_medium"),
        "alphaearth_v1": ("AlphaEarth", 5, "alphaearth_v1"),
        "unet_b7_s2s1": ("Baseline: PRUE on S2+S1", 6, "unet_r18_scratch_s2s1")}
TYPES = ("same crop", "other crop", "undeclared farmland", "non-farm land")
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "h", "P", "X"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5, "legend.frameon": False,
    "lines.linewidth": 2, "lines.markersize": 7,
})


def _json(p: Path) -> dict:
    return json.loads(p.read_text())


def colour(arm: str) -> str:
    return SLOTS[ARMS[arm][1] - 1]


def marker(arm: str) -> str:
    return MARKERS[ARMS[arm][1] - 1]


def load(chips: str) -> dict:
    """``arm -> pct -> row`` of the scored cells, with the main workflow's crop scores beside them."""
    out: dict = {}
    for arm, (_, _, kshot) in ARMS.items():
        for f in sorted((REPO / "results" / NAME / chips / arm).glob("P*_draw0_seed0/results.json")):
            r = _json(f)
            if "border_metrics" not in r:
                continue
            pct = float(r["label_budget"]["pct"])
            g = r["panoptic_metrics"]["variants"]["chip/grow"]
            b = r["border_metrics"]
            row = {"crop_pixel": r["crop_metrics"]["macro_f1"], "crop_parcel": r["parcel_metrics"]["macro_f1"],
                   "border_p": b["d1"]["precision"], "border_r": b["d1"]["recall"], "border_f1": b["d1"]["f1"],
                   "pq_agn": g["pq_agnostic"], "rq_agn": g["rq_agnostic"], "pq": g["pq"],
                   "fields": g["predicted_parcels"], "true": g["true_parcels"],
                   "pq_agn_cc": r["panoptic_metrics"]["variants"]["chip/cc"]["pq_agnostic"],
                   "types": {t: b["recall_by_type_d1"][t]["recall"] for t in TYPES},
                   "best_epoch": r["best_epoch"], "alone": r["epoch_by_single_criterion"],
                   "weight": r.get("final_crop_weight"), "fit_h": r["fit_seconds"] / 3600,
                   "history": r["history"], "train_chips": r["n_chips"]["train"]}
            ref = REPO / "results" / "kshot" / chips / kshot / f.parent.name / "results.json"
            if ref.exists():
                pm = _json(ref).get("parcel_metrics", {})
                row["ref_pixel"], row["ref_parcel"] = pm.get("pixel_macro_f1"), pm.get("macro_f1")
            out.setdefault(arm, {})[pct] = row
    return out


def _k_axis(ax, ks):
    ax.set_xscale("log")
    ax.set_xticks(ks)
    ax.set_xticklabels([f"{k:g} %" for k in ks])
    ax.minorticks_off()
    ax.set_xlabel("label budget K")


def _legend(fig, d, extra=()):
    from matplotlib.lines import Line2D

    h = [Line2D([], [], color=colour(a), marker=marker(a), label=ARMS[a][0]) for a in d] + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), 4), bbox_to_anchor=(0.5, -0.02))


def fig_borders(d: dict, dest: Path) -> None:
    ks = sorted({k for a in d for k in d[a]})
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax, key, title in zip(axes, ("border_f1", "pq_agn", "pq"),
                              ("borders: F1 at 1 px", "fields: PQ, class-agnostic", "fields: PQ, class-aware")):
        for a in d:
            xs = sorted(d[a])
            ax.plot(xs, [d[a][k][key] for k in xs], color=colour(a), marker=marker(a))
            if xs:
                ax.annotate(f"{d[a][xs[-1]][key]:.3f}", (xs[-1], d[a][xs[-1]][key]), xytext=(6, 0),
                            textcoords="offset points", fontsize=8, color=INK_2, va="center")
        ax.set_title(title)
        _k_axis(ax, ks)
    fig.suptitle("Each arm with PRUE's border head, frozen encoders, test chips of Estonia", x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    _legend(fig, d)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(dest / "borders_by_k.png", dpi=150)
    plt.close(fig)


def fig_crop(d: dict, dest: Path) -> None:
    from matplotlib.lines import Line2D

    ks = sorted({k for a in d for k in d[a]})
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for ax, key, ref, title in zip(axes, ("crop_pixel", "crop_parcel"), ("ref_pixel", "ref_parcel"),
                                   ("crop types: pixel Macro-F1", "crop types: parcel Macro-F1")):
        for a in d:
            xs = sorted(d[a])
            ax.plot(xs, [d[a][k][key] for k in xs], color=colour(a), marker=marker(a))
            rx = [k for k in xs if d[a][k].get(ref) is not None]
            if rx:
                ax.plot(rx, [d[a][k][ref] for k in rx], color=colour(a), ls="--", lw=1.4, alpha=0.8)
        ax.set_title(title)
        _k_axis(ax, ks)
    extra = [Line2D([], [], color=MUTED, label="with the border head"),
             Line2D([], [], color=MUTED, ls="--", lw=1.4, label="main workflow, crop types only")]
    _legend(fig, d, extra)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(dest / "crop_by_k.png", dpi=150)
    plt.close(fig)


def fig_types(d: dict, dest: Path, k: float = 100.0) -> None:
    arms = [a for a in d if k in d[a]]
    if not arms:
        return
    fig, ax = plt.subplots(figsize=(10, 4))
    w = 0.8 / len(arms)
    x = np.arange(len(TYPES))
    for i, a in enumerate(arms):
        ax.bar(x + (i - (len(arms) - 1) / 2) * w, [d[a][k]["types"][t] for t in TYPES], w * 0.92,
               color=colour(a), label=ARMS[a][0])
    ax.set_xticks(x)
    ax.set_xticklabels([f"border with {t}" for t in TYPES])
    ax.set_ylabel("recall at 1 px")
    ax.set_ylim(0, 1)
    ax.set_title(f"Which declared borders each arm draws, K = {k:g} %")
    ax.legend(ncol=3, loc="upper left", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(dest / "border_types.png", dpi=150)
    plt.close(fig)


def fig_training(d: dict, dest: Path, k: float = 100.0) -> None:
    arms = [a for a in d if k in d[a]]
    if not arms:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for a in arms:
        h = d[a][k]["history"]
        ep = [r["epoch"] for r in h]
        axes[0].plot(ep, [r["val/cc/RQ_agnostic"] for r in h], color=colour(a), lw=1.6)
        axes[1].plot(ep, [r["val/crop_macro_f1"] for r in h], color=colour(a), lw=1.6)
        axes[2].plot(ep, [r.get("balance/crop_weight") or np.nan for r in h], color=colour(a), lw=1.6)
        be = d[a][k]["best_epoch"]
        axes[0].plot([be], [h[be - 1]["val/cc/RQ_agnostic"]], marker=marker(a), color=colour(a))
        axes[1].plot([be], [h[be - 1]["val/crop_macro_f1"]], marker=marker(a), color=colour(a))
    axes[0].set_title("validation: object F1 of the arm's own fields")
    axes[1].set_title("validation: crop pixel Macro-F1")
    axes[2].set_title("crop loss weight (pull 0.3 x the border loss)")
    axes[2].set_yscale("log")
    for ax in axes:
        ax.set_xlabel("epoch")
    _legend(fig, {a: d[a] for a in arms})
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(dest / "training.png", dpi=150)
    plt.close(fig)


def tables(d: dict, dest: Path) -> None:
    L = ["## Borders and fields", "",
         "| Arm | K | border P / R / F1 at 1 px | PQ agn. (grow) | RQ agn. | PQ class-aware | fields / declared | "
         "PQ agn. (PRUE's own fields) |", "|---|---|---|---|---|---|---|---|"]
    for a in d:
        for k in sorted(d[a], reverse=True):
            r = d[a][k]
            L.append(f"| {ARMS[a][0]} | {k:g} % | {r['border_p']:.3f} / {r['border_r']:.3f} / {r['border_f1']:.3f} "
                     f"| {r['pq_agn']:.3f} | {r['rq_agn']:.3f} | {r['pq']:.3f} | {r['fields']:,} / {r['true']:,} "
                     f"| {r['pq_agn_cc']:.3f} |")
    L += ["", "## Crop types, against the main workflow's crop-only fit", "",
          "| Arm | K | pixel Macro-F1 | main workflow | difference | parcel Macro-F1 | main workflow | difference |",
          "|---|---|---|---|---|---|---|---|"]

    def diff(x, y):
        return f"{x - y:+.3f}" if y is not None else ""

    for a in d:
        for k in sorted(d[a], reverse=True):
            r = d[a][k]
            rp, rq = r.get("ref_pixel"), r.get("ref_parcel")
            L.append(f"| {ARMS[a][0]} | {k:g} % | {r['crop_pixel']:.3f} | {'' if rp is None else f'{rp:.3f}'} | "
                     f"{diff(r['crop_pixel'], rp)} | {r['crop_parcel']:.3f} | {'' if rq is None else f'{rq:.3f}'} | "
                     f"{diff(r['crop_parcel'], rq)} |")
    L += ["", "## Border recall at 1 px by what lies across the border", "",
          "| Arm | K | " + " | ".join(TYPES) + " |", "|---|---|" + "---|" * len(TYPES)]
    for a in d:
        for k in sorted(d[a], reverse=True):
            L.append(f"| {ARMS[a][0]} | {k:g} % | " + " | ".join(f"{d[a][k]['types'][t]:.3f}" for t in TYPES) + " |")
    L += ["", "## Training", "", "| Arm | K | train chips | epoch kept | by object F1 alone | by crop loss alone | "
          "final crop weight | fit h |", "|---|---|---|---|---|---|---|---|"]
    for a in d:
        for k in sorted(d[a], reverse=True):
            r = d[a][k]
            w = "" if r["weight"] is None else f"{r['weight']:.4g}"
            L.append(f"| {ARMS[a][0]} | {k:g} % | {r['train_chips']:,} | {r['best_epoch']} | "
                     f"{r['alone']['object_f1']} | {r['alone']['crop_loss']} | {w} | {r['fit_h']:.2f} |")
    (dest / "tables.md").write_text("\n".join(L) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chips", default="EE_2021_rev")
    ap.add_argument("--out", type=Path, default=HERE / "figures")
    a = ap.parse_args()
    d = load(a.chips)
    if not d:
        raise SystemExit(f"no scored cell under results/{NAME}/{a.chips}")
    a.out.mkdir(exist_ok=True)
    fig_borders(d, a.out)
    fig_crop(d, a.out)
    fig_types(d, a.out)
    fig_training(d, a.out)
    tables(d, a.out)
    print((a.out / "tables.md").read_text())
    print(f"figures of {list(d)} in {a.out}")


if __name__ == "__main__":
    main()
