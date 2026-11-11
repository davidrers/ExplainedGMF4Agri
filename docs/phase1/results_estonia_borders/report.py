"""Figures and tables of the two-head workflow's report on Estonia (REPORT.md beside this file).

Reads, without writing there:

* the two-head workflow's grid on ``EE_2021_rev``, its first run (protocol D30):
  ``results/2026-11-09_borders_all_arms/EE_2021_rev/<arm>/P<k>_draw0_seed0/results.json``;
* the crop-only workflow, frozen: ``results/kshot/EE_2021_rev/``, and fine-tuned:
  ``results/2026-10-15_kshot_finetune/EE_2021_rev/``;
* TESSERA at K = 100 %, frozen and fine-tuned: its separate border model, PRUE
  (``results/2026-11-06_prue_tessera/EE_2021_rev/``, fields; ``results/2026-11-07_border_accuracy/EE_2021_rev/``,
  borders), and one network with both heads at a fixed crop weight of 0.1 or 0.01
  (``results/2026-11-07_prue_crop_tessera/EE_2021_rev/``);

and writes into ``figures/``:

* ``borders_by_k.png``: border F1 within 1 px and the PQ of the grown fields, class-agnostic and class-aware,
  against K;
* ``crop_by_k.png``: crop pixel and parcel Macro-F1 against K, with the crop-only workflow dashed;
* ``border_types.png``: border recall within 1 px by what lies across the border, K = 100 and 5 %;
* ``finetune.png``: TESSERA at K = 100 %, frozen and fine-tuned, separate models and one network;
* ``tables.md`` beside this file: every number the report quotes.

    poetry run python docs/phase1/results_estonia_borders/report.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
CHIPS = "EE_2021_rev"
RES = REPO / "results"
TWO = RES / "2026-11-09_borders_all_arms" / CHIPS
CROP = RES / "kshot" / CHIPS
CROP_FT = RES / "2026-10-15_kshot_finetune" / CHIPS
PRUE = RES / "2026-11-06_prue_tessera" / CHIPS
BORDERS = RES / "2026-11-07_border_accuracy" / CHIPS / "borders.json"
JOINT = RES / "2026-11-07_prue_crop_tessera" / CHIPS
OUT = HERE / "figures"
CELL = "P{k:g}_draw0_seed0"

#: Arm -> (label, colour slot, crop-only arm, crop-only fine-tuned arm or None).
ARMS = {"terramind_v1_large_s2s1": ("TerraMind v1 L, S2 + S1", 1, "terramind_v1_large_s2s1",
                                    "terramind_v1_large_s2s1_ft"),
        "prithvi_eo_v2_600_tl": ("Prithvi-EO-2.0 600M", 2, "prithvi_eo_v2_600_tl", "prithvi_eo_v2_600_tl_ft"),
        "thor_v1_large_s2s1": ("THOR v1 L, S2 + S1", 3, "thor_v1_large_s2s1", "thor_v1_large_s2s1_ft"),
        "tessera_v2_medium": ("TESSERA v2 M", 4, "tessera_v2_medium", "tessera_v2_medium_ft"),
        "alphaearth_v1": ("AlphaEarth v1", 5, "alphaearth_v1", None),
        "unet_b7_s2s1": ("Baseline: PRUE, S2 + S1", 6, "unet_r18_scratch_s2s1", None)}
TYPES = ("same crop", "other crop", "undeclared farmland", "non-farm land")
KS = (100.0, 20.0, 5.0)
#: TESSERA at K = 100 %: setting -> (label, encoder, kind). Kinds: separate (the crop MLP and PRUE, two models),
#: w001 / w01 (one network, fixed crop weight), balanced (one network, the workflow's balance).
TESSERA = [("frozen_separate", "two separate models", "frozen", "separate"),
           ("frozen_w01", "one network, crop weight 0.1", "frozen", "w01"),
           ("frozen_w001", "one network, crop weight 0.01", "frozen", "w001"),
           ("frozen_balanced", "one network, balanced (this workflow)", "frozen", "balanced"),
           ("ft_separate", "two separate models", "fine-tuned", "separate"),
           ("ft_w01", "one network, crop weight 0.1", "fine-tuned", "w01"),
           ("ft_w001", "one network, crop weight 0.01", "fine-tuned", "w001")]
METRICS = {"crop_pixel": "crop types:\npixel Macro-F1", "crop_parcel": "crop types:\nparcel Macro-F1",
           "border_f1": "borders:\nF1 within 1 px", "pq_agn": "fields:\nPQ, class-agnostic",
           "pq": "fields:\nPQ, class-aware"}
#: Published references for field delineation on Fields of The World (PRUE, Table 1): pixel IoU, object F1.
FTW = [("PRUE: U-Net + EfficientNet-B7, log-cosh Dice", "0.76", "0.47"),
       ("FTW baseline: U-Net + EfficientNet-B3", "0.70", "0.38"),
       ("Mask2Former (Swin-S)", "0.68", "0.39"),
       ("SAM ViT-H, fine-tuned / zero-shot", "0.45 / 0.32", "0.37 / 0.17"),
       ("DECODE (FracTAL ResUNet, multi-task)", "0.71", "0.21"),
       ("Delineate Anything, zero-shot", "0.37", "0.09"),
       ("Clay ViT-L, frozen; fully fine-tuned", "0.67; 0.73", "0.36; 0.42"),
       ("Galileo ViT-B, DINOv3 ViT-L, frozen", "0.66, 0.60", "0.32, 0.31"),
       ("TerraMind ViT-B, Prithvi 2.0 ViT-L, frozen", "0.57, 0.56", "0.26, 0.25")]

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
MARKERS = ["o", "s", "D", "^", "v", "h", "P", "X"]
KIND_COL = {"separate": SLOTS[0], "w01": "#f2a57a", "w001": SLOTS[1], "balanced": SLOTS[2]}
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
    return json.loads(Path(p).read_text())


def colour(arm: str) -> str:
    return SLOTS[ARMS[arm][1] - 1]


def marker(arm: str) -> str:
    return MARKERS[ARMS[arm][1] - 1]


def _crop_ref(arm_dir: Path, k: float) -> dict:
    f = arm_dir / CELL.format(k=k) / "results.json"
    if not f.exists():
        return {}
    pm = _json(f)["parcel_metrics"]
    return {"pixel": pm["pixel_macro_f1"], "parcel": pm["macro_f1"]}


def load() -> dict:
    """``arm -> pct -> row`` of the two-head grid, with the crop-only workflow's crop scores beside each cell."""
    out: dict = {}
    for arm, (_, _, crop_arm, ft_arm) in ARMS.items():
        for k in KS:
            f = TWO / arm / CELL.format(k=k) / "results.json"
            if not f.exists():
                continue
            r = _json(f)
            v = r["panoptic_metrics"]["variants"]
            g, cc = v["chip/grow"], v["chip/cc"]
            b = r["border_metrics"]
            ref, ref_ft = _crop_ref(CROP / crop_arm, k), _crop_ref(CROP_FT / ft_arm, k) if ft_arm else {}
            out.setdefault(arm, {})[k] = {
                "crop_pixel": r["crop_metrics"]["macro_f1"], "crop_parcel": r["parcel_metrics"]["macro_f1"],
                "border_p": b["d1"]["precision"], "border_r": b["d1"]["recall"], "border_f1": b["d1"]["f1"],
                "border_f1_d0": b["d0"]["f1"], "border_f1_d2": b["d2"]["f1"],
                "pq_agn": g["pq_agnostic"], "sq_agn": g["sq_agnostic"], "rq_agn": g["rq_agnostic"], "pq": g["pq"],
                "rq_agn_cc": cc["rq_agnostic"], "pq_agn_cc": cc["pq_agnostic"],
                "fields": g["predicted_parcels"], "true": g["true_parcels"],
                "types": {t: b["recall_by_type_d1"][t]["recall"] for t in TYPES},
                "false_on_parcels": b["false_border_share_d1"]["on a declared parcel"],
                "best_epoch": r["best_epoch"], "alone": r["epoch_by_single_criterion"],
                "weight": r.get("final_crop_weight"), "fit_h": r["fit_seconds"] / 3600,
                "train_chips": r["n_chips"]["train"], "support": r["support"],
                "ref_pixel": ref.get("pixel"), "ref_parcel": ref.get("parcel"),
                "ft_pixel": ref_ft.get("pixel"), "ft_parcel": ref_ft.get("parcel")}
    return out


def load_tessera(d: dict) -> dict:
    """TESSERA at K = 100 %: the seven settings of ``TESSERA``, each with the five headline scores and its fit time."""
    bnd = _json(BORDERS)["methods"]
    out = {}
    for enc, crop_dir, prue_cell, bname, jprefix in (
            ("frozen", CROP / "tessera_v2_medium", PRUE / "prue_tessera_v2_frozen", "PRUE frozen",
             "joint_tessera_v2_frozen"),
            ("fine-tuned", CROP_FT / "tessera_v2_medium_ft", PRUE / "prue_tessera_v2_ft", "PRUE fine-tuned",
             "joint_tessera_v2_ft")):
        key = "frozen" if enc == "frozen" else "ft"
        kc = _json(crop_dir / CELL.format(k=100) / "results.json")
        pr = _json(prue_cell / CELL.format(k=100) / "results.json")
        g = pr["panoptic_metrics"]["variants"]["chip/grow"]
        out[f"{key}_separate"] = {"crop_pixel": kc["parcel_metrics"]["pixel_macro_f1"],
                                  "crop_parcel": kc["parcel_metrics"]["macro_f1"],
                                  "border_f1": bnd[bname]["d1"]["f1"], "pq_agn": g["pq_agnostic"], "pq": g["pq"],
                                  "rq_agn": g["rq_agnostic"],
                                  "fit_h": (kc["fit_seconds"] + pr["fit_seconds"]) / 3600}
        for suffix, kind in (("", "w01"), ("_cw001", "w001")):
            r = _json(JOINT / f"{jprefix}{suffix}" / CELL.format(k=100) / "results.json")
            jg = r["panoptic_metrics"]["variants"]["chip/grow"]
            out[f"{key}_{kind}"] = {"crop_pixel": r["crop_metrics"]["macro_f1"],
                                    "crop_parcel": r["parcel_metrics"]["macro_f1"],
                                    "border_f1": r["border_metrics"]["d1"]["f1"], "pq_agn": jg["pq_agnostic"],
                                    "pq": jg["pq"], "rq_agn": jg["rq_agnostic"], "fit_h": r["fit_seconds"] / 3600}
    t = d["tessera_v2_medium"][100.0]
    out["frozen_balanced"] = {k: t[k] for k in ("crop_pixel", "crop_parcel", "border_f1", "pq_agn", "pq", "rq_agn",
                                                "fit_h")}
    return out


def _k_axis(ax):
    ax.set_xscale("log")
    ax.set_xticks(sorted(KS))
    ax.set_xticklabels([f"{k:g} %" for k in sorted(KS)])
    ax.minorticks_off()
    ax.set_xlabel("label budget K")
    ax.set_xlim(min(KS) / 1.25, max(KS) * 1.25)


def _legend(fig, arms, extra=(), ncol=4):
    h = [Line2D([], [], color=colour(a), marker=marker(a), label=ARMS[a][0]) for a in arms] + list(extra)
    fig.legend(handles=h, loc="lower center", ncol=min(len(h), ncol), bbox_to_anchor=(0.5, -0.01))


def _end_labels(ax, points: list[tuple[float, float]], gap: float = 0.055) -> None:
    """The value at the end of each line, pushed apart so that none overlaps, with a leader when moved."""
    if not points:
        return
    lo, hi = ax.get_ylim()
    step = gap * (hi - lo)
    order = sorted(range(len(points)), key=lambda i: points[i][1])
    ys = [points[i][1] for i in order]
    for j in range(1, len(ys)):
        ys[j] = max(ys[j], ys[j - 1] + step)
    over = ys[-1] - (hi + 0.5 * step)
    if over > 0:
        ys = [y - over for y in ys]
        for j in range(len(ys) - 2, -1, -1):
            ys[j] = min(ys[j], ys[j + 1] - step)
    for j, i in enumerate(order):
        x, y = points[i]
        ax.annotate(f"{y:.3f}", (x, y), xytext=(x * 1.12, ys[j]), textcoords="data", fontsize=8, color=INK_2,
                    va="center", annotation_clip=False,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6) if abs(ys[j] - y) > 0.3 * step else None)


def fig_borders(d: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.5))
    for ax, key, title in zip(axes, ("border_f1", "pq_agn", "pq"),
                              ("borders: F1 within 1 px", "fields: PQ, class-agnostic", "fields: PQ, class-aware")):
        ends = []
        for a in d:
            xs = sorted(d[a])
            ax.plot(xs, [d[a][k][key] for k in xs], color=colour(a), marker=marker(a))
            ends.append((xs[-1], d[a][xs[-1]][key]))
        ax.set_title(title)
        _k_axis(ax)
        _end_labels(ax, ends)
    fig.suptitle("Field borders and fields of every arm, encoders frozen, the 1,475 test tiles of Estonia",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    _legend(fig, d)
    fig.tight_layout(rect=(0, 0.1, 1, 0.95))
    fig.savefig(OUT / "borders_by_k.png", dpi=150)
    plt.close(fig)


def fig_crop(d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5))
    for ax, key, ref, title in zip(axes, ("crop_pixel", "crop_parcel"), ("ref_pixel", "ref_parcel"),
                                   ("crop types: pixel Macro-F1", "crop types: parcel Macro-F1")):
        for a in d:
            xs = sorted(d[a])
            ax.plot(xs, [d[a][k][key] for k in xs], color=colour(a), marker=marker(a))
            rx = [k for k in xs if d[a][k].get(ref) is not None]
            ax.plot(rx, [d[a][k][ref] for k in rx], color=colour(a), ls="--", lw=1.4, alpha=0.8)
        ax.set_title(title)
        _k_axis(ax)
    extra = [Line2D([], [], color=MUTED, label="two-head workflow"),
             Line2D([], [], color=MUTED, ls="--", lw=1.4, label="crop-only workflow (its baseline: U-Net from scratch)")]
    _legend(fig, d, extra, ncol=4)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(OUT / "crop_by_k.png", dpi=150)
    plt.close(fig)


def fig_types(d: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.3), sharey=True)
    for ax, k in zip(axes, (100.0, 5.0)):
        arms = [a for a in d if k in d[a]]
        w = 0.8 / len(arms)
        x = np.arange(len(TYPES))
        for i, a in enumerate(arms):
            ax.bar(x + (i - (len(arms) - 1) / 2) * w, [d[a][k]["types"][t] for t in TYPES], w * 0.9,
                   color=colour(a), label=ARMS[a][0])
        ax.set_xticks(x)
        ax.set_xticklabels([f"against\n{t}" for t in TYPES])
        ax.set_ylim(0, 1)
        ax.set_title(f"K = {k:g} %")
    axes[0].set_ylabel("share of the declared border found within 1 px")
    fig.suptitle("Which declared borders each arm draws, by what lies across the border", x=0.01, ha="left",
                 fontsize=11.5, color=INK)
    _legend(fig, list(d))
    fig.tight_layout(rect=(0, 0.1, 1, 0.94))
    fig.savefig(OUT / "border_types.png", dpi=150)
    plt.close(fig)


def _tint(hex_: str, t: float = 0.5) -> tuple:
    c = np.array(to_rgb(hex_))
    return tuple(c + (1 - c) * t)


def fig_finetune(t: dict) -> None:
    fig, ax = plt.subplots(figsize=(12.5, 4.6))
    order = [s for s, *_ in TESSERA if s in t]
    x = np.arange(len(METRICS))
    w = 0.8 / len(order)
    for i, s in enumerate(order):
        _, label, enc, kind = next(r for r in TESSERA if r[0] == s)
        col = KIND_COL[kind]
        frozen = enc == "frozen"
        ax.bar(x + (i - (len(order) - 1) / 2) * w, [t[s][m] for m in METRICS], w * 0.9,
               color=_tint(col, 0.55) if frozen else col, edgecolor=col, linewidth=0.8,
               hatch="///" if frozen else None)
    ax.set_xticks(x)
    ax.set_xticklabels(list(METRICS.values()))
    lo = min(t[s][m] for s in order for m in METRICS)
    ax.set_ylim(max(0.0, lo - 0.08), 0.92)
    ax.set_title("TESSERA v2 at K = 100 %: frozen (hatched) and fine-tuned encoder; two separate models or one "
                 "network with both heads")
    handles = [Patch(facecolor=_tint(KIND_COL[k], 0.55), edgecolor=KIND_COL[k], hatch="///", label=lab)
               for k, lab in (("separate", "frozen: two separate models"), ("w01", "frozen: one network, weight 0.1"),
                              ("w001", "frozen: one network, weight 0.01"),
                              ("balanced", "frozen: one network, balanced (this workflow)"))]
    handles += [Patch(facecolor=KIND_COL[k], edgecolor=KIND_COL[k], label=lab)
                for k, lab in (("separate", "fine-tuned: two separate models"),
                               ("w01", "fine-tuned: one network, weight 0.1"),
                               ("w001", "fine-tuned: one network, weight 0.01"))]
    fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.01), fontsize=8.5)
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    fig.savefig(OUT / "finetune.png", dpi=150)
    plt.close(fig)


def tables(d: dict, t: dict) -> str:
    def f3(x):
        return "" if x is None else f"{x:.3f}"

    def df(x, y):
        return "" if x is None or y is None else f"{x - y:+.3f}"

    L = ["# Tables of REPORT.md", "", "Written by [report.py](report.py). Test tiles of `EE_2021_rev` (1,475), draw 0, "
         "seed 0 per cell.", "", "## Borders and fields, two-head workflow", "",
         "| Arm | K | border P / R / F1 within 1 px | border F1 at 0 / 2 px | PQ agn. | SQ / RQ agn. | PQ "
         "class-aware | fields / declared | RQ agn. of PRUE's own fields (cc) |",
         "|---|---|---|---|---|---|---|---|---|"]
    for a in d:
        for k in sorted(d[a], reverse=True):
            r = d[a][k]
            L.append(f"| {ARMS[a][0]} | {k:g} % | {r['border_p']:.3f} / {r['border_r']:.3f} / {r['border_f1']:.3f} | "
                     f"{r['border_f1_d0']:.3f} / {r['border_f1_d2']:.3f} | {r['pq_agn']:.3f} | {r['sq_agn']:.3f} / "
                     f"{r['rq_agn']:.3f} | {r['pq']:.3f} | {r['fields']:,} / {r['true']:,} | {r['rq_agn_cc']:.3f} |")
    L += ["", "## Crop types: two-head against crop-only workflow", "",
          "| Arm | K | pixel Macro-F1 | crop-only | difference | parcel Macro-F1 | crop-only | difference | "
          "crop-only fine-tuned, pixel / parcel |", "|---|---|---|---|---|---|---|---|---|"]
    for a in d:
        for k in sorted(d[a], reverse=True):
            r = d[a][k]
            ft = "" if r["ft_pixel"] is None else f"{r['ft_pixel']:.3f} / {r['ft_parcel']:.3f}"
            L.append(f"| {ARMS[a][0]} | {k:g} % | {r['crop_pixel']:.3f} | {f3(r['ref_pixel'])} | "
                     f"{df(r['crop_pixel'], r['ref_pixel'])} | {r['crop_parcel']:.3f} | {f3(r['ref_parcel'])} | "
                     f"{df(r['crop_parcel'], r['ref_parcel'])} | {ft} |")
    L += ["", "The crop-only baseline is the ResNet-18 U-Net from scratch (`unet_r18_scratch_s2s1`); the two-head "
          "baseline is PRUE from ImageNet (`unet_b7_s2s1`).", "",
          "## Border recall within 1 px by what lies across the border", "",
          "| Arm | K | " + " | ".join(TYPES) + " | false borders on a declared parcel |",
          "|---|---|" + "---|" * (len(TYPES) + 1)]
    for a in d:
        for k in sorted(d[a], reverse=True):
            r = d[a][k]
            L.append(f"| {ARMS[a][0]} | {k:g} % | " + " | ".join(f"{r['types'][x]:.3f}" for x in TYPES)
                     + f" | {r['false_on_parcels']:.3f} |")
    L += ["", "## Training", "", "| Arm | K | training tiles | labelled parcels | epoch kept | by object F1 alone | "
          "by crop loss alone | final crop weight | fit h |", "|---|---|---|---|---|---|---|---|---|"]
    for a in d:
        for k in sorted(d[a], reverse=True):
            r = d[a][k]
            n = sum(int(v["drawn"]) for v in r["support"].values())
            L.append(f"| {ARMS[a][0]} | {k:g} % | {r['train_chips']:,} | {n:,} | "
                     f"{r['best_epoch']} | {r['alone']['object_f1']} | {r['alone']['crop_loss']} | "
                     f"{r['weight']:.4f} | {r['fit_h']:.2f} |")
    L += ["", "## TESSERA v2 at K = 100 %: frozen and fine-tuned, separate models and one network", "",
          "| Encoder | Setting | crop pixel / parcel Macro-F1 | border F1 within 1 px | PQ agn. | RQ agn. | PQ "
          "class-aware | fit h |", "|---|---|---|---|---|---|---|---|"]
    for s, label, enc, _ in TESSERA:
        if s in t:
            r = t[s]
            L.append(f"| {enc} | {label} | {r['crop_pixel']:.3f} / {r['crop_parcel']:.3f} | {r['border_f1']:.3f} | "
                     f"{r['pq_agn']:.3f} | {r['rq_agn']:.3f} | {r['pq']:.3f} | {r['fit_h']:.1f} |")
    L += ["", "Fit hours: the two separate models' sum; the frozen balanced network trained on whole chips for 45 "
          "epochs, the fixed-weight networks on 128 x 128 windows for 30 epochs.", "",
          "## Crop types, fine-tuned against frozen, crop-only workflow (pixel / parcel Macro-F1)", "",
          "| Arm | K | frozen | fine-tuned | difference |", "|---|---|---|---|---|"]
    for a in d:
        for k in sorted(d[a], reverse=True):
            r = d[a][k]
            if r["ft_pixel"] is None:
                continue
            L.append(f"| {ARMS[a][0]} | {k:g} % | {r['ref_pixel']:.3f} / {r['ref_parcel']:.3f} | {r['ft_pixel']:.3f} / "
                     f"{r['ft_parcel']:.3f} | {r['ft_pixel'] - r['ref_pixel']:+.3f} / "
                     f"{r['ft_parcel'] - r['ref_parcel']:+.3f} |")
    L += ["", "## Field delineation: published on Fields of The World (PRUE, Table 1) and ours", "",
          "| Model | Data | Pixel IoU | Object F1 |", "|---|---|---|---|"]
    for name, iou, f1 in FTW:
        L.append(f"| {name} | FTW test sets, 24 countries | {iou} | {f1} |")
    for a in d:
        r = d[a][100.0]
        L.append(f"| {ARMS[a][0]}, this workflow, K = 100 % | `EE_2021_rev` test tiles | not comparable | "
                 f"{r['rq_agn']:.2f} grown, {r['rq_agn_cc']:.2f} interior only |")
    return "\n".join(L) + "\n"


def main() -> None:
    d = load()
    if not d:
        raise SystemExit(f"no scored cell under {TWO}")
    t = load_tessera(d)
    OUT.mkdir(exist_ok=True)
    fig_borders(d)
    fig_crop(d)
    fig_types(d)
    fig_finetune(t)
    (HERE / "tables.md").write_text(tables(d, t))
    print((HERE / "tables.md").read_text())


if __name__ == "__main__":
    main()
