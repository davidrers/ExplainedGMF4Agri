"""Figures and tables of the joint experiment, for its REPORT.md: the one model against the two separate ones.

Reads ``results/<name>/<chip set>/`` as ``run.py`` writes it and, as the separate models on the same encoder, the
main workflow's crop arms (``results/kshot`` frozen, ``results/2026-10-15_kshot_finetune`` fine-tuned: their
``results.json`` and, for the per-class F1, their ``predictions_test.npz``), PRUE (``results/2026-11-06_prue_tessera``:
PQ of the grown fields) and the border-accuracy experiment (``results/2026-11-07_border_accuracy``: border P / R / F1
of PRUE's grown fields, overall, by border type and per chip). K = 100 %, draw 0, seed 0. An arm without results is
left out, so the report can be drawn before both jobs are back. Writes into ``figures/``:

* ``headline.png``: each metric of the comparison, separate against joint, per encoder;
* ``validation_curves.png``: per arm, the validation crop Macro-F1, object F1 and their mean per epoch with the epoch
  kept; below, how hard the weighted crop loss pulled on the U-Net against the border loss;
* ``crop_per_class.png``: per crop class, the joint model's pixel F1 minus the separate crop arm's;
* ``border_types.png``: border recall at 1 px by what lies across the border, separate against joint;
* ``per_chip.png``: per test chip, the joint model's border F1 minus PRUE's;
* ``tables.md``: the tables of REPORT.md, also printed.

    poetry run python experiments/2026-11-07_prue_crop_tessera/report.py --chips EE_2021_rev
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

NAME = "2026-11-07_prue_crop_tessera"
CELL = "P100_draw0_seed0"
#: Arm -> label, the separate crop arm's cell, PRUE's cell, PRUE's name in the border-accuracy experiment.
ARMS = {"joint_tessera_v2_frozen": ("TESSERA frozen", "kshot/{set}/tessera_v2_medium",
                                    "2026-11-06_prue_tessera/{set}/prue_tessera_v2_frozen", "PRUE frozen"),
        "joint_tessera_v2_ft": ("TESSERA fine-tuned", "2026-10-15_kshot_finetune/{set}/tessera_v2_medium_ft",
                                "2026-11-06_prue_tessera/{set}/prue_tessera_v2_ft", "PRUE fine-tuned")}
BORDERS = "2026-11-07_border_accuracy/{set}"
#: Metric key -> label; all higher-is-better scores in [0, 1].
METRICS = {"crop_pixel_macro_f1": "crop types: pixel Macro-F1", "crop_parcel_macro_f1": "crop types: parcel Macro-F1",
           "border_f1_d1": "borders: F1 at 1 px", "pq_agnostic_grow": "parcels: PQ, class-agnostic",
           "pq_grow": "parcels: PQ, class-aware"}
TYPES = ("same crop", "other crop", "undeclared farmland", "non-farm land")

SEPARATE, JOINT = "#2a78d6", "#eb6834"         # categorical slots 1 and 2 of the validated default palette
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


def _f1_rows(counts: np.ndarray) -> np.ndarray:
    """Per chip border F1 at 1 px from ``(n, 4)`` counts ``[tp_p, n_p, tp_r, n_r]``; NaN without a true border."""
    tp_p, n_p, tp_r, n_r = (counts[:, k].astype(float) for k in range(4))
    with np.errstate(invalid="ignore", divide="ignore"):
        p, r = tp_p / n_p, tp_r / n_r
        f = 2 * p * r / (p + r)
    f[(n_r == 0) | (n_p == 0)] = np.nan
    return f


def load(chips: str) -> dict:
    """Per arm with results: the joint cell's and the separate models' scores, history and per-class F1."""
    import score_joint

    out = {}
    bnd = _json(REPO / "results" / BORDERS.format(set=chips) / "borders.json")["methods"]
    pc = np.load(REPO / "results" / BORDERS.format(set=chips) / "per_chip.npz")
    pc_ids = [str(c) for c in pc["ids"]]
    for arm, (label, crop_rel, prue_rel, bname) in ARMS.items():
        cell = REPO / "results" / NAME / chips / arm / CELL
        if not (cell / "results.json").exists() or "border_metrics" not in _json(cell / "results.json"):
            continue
        r = _json(cell / "results.json")
        crop_cell = REPO / "results" / crop_rel.format(set=chips) / CELL
        kc = _json(crop_cell / "results.json")
        pr = _json(REPO / "results" / prue_rel.format(set=chips) / CELL / "results.json")["panoptic_metrics"]
        g, jg = pr["variants"]["chip/grow"], r["panoptic_metrics"]["variants"]["chip/grow"]
        sep_crop = score_joint.crop_scores(crop_cell / "predictions_test.npz", REPO / "data/eurocrops_chips" / chips
                                           / "chips", r["class_names"])
        jb = np.load(cell / "borders_per_chip.npz")
        j_ids = [str(c) for c in jb["ids"]]
        order = {c: i for i, c in enumerate(pc_ids)}
        sep_counts = pc["counts"][list(pc["methods"]).index(bname)][[order[c] for c in j_ids]]
        out[arm] = {
            "label": label, "results": r,
            "separate": {"crop_pixel_macro_f1": kc["parcel_metrics"]["pixel_macro_f1"],
                         "crop_parcel_macro_f1": kc["parcel_metrics"]["macro_f1"],
                         "border_precision_d1": bnd[bname]["d1"]["precision"],
                         "border_recall_d1": bnd[bname]["d1"]["recall"], "border_f1_d1": bnd[bname]["d1"]["f1"],
                         "pq_agnostic_grow": g["pq_agnostic"], "pq_grow": g["pq"],
                         "fields_grow": g["predicted_parcels"],
                         "recall_by_type": {t: bnd[bname]["recall_by_type_d1"][t]["recall"] for t in TYPES},
                         "per_class_f1": {c: v["f1"] for c, v in sep_crop["pixel"]["per_class"].items()},
                         "fit_h": (kc["fit_seconds"] + _json(REPO / "results" / prue_rel.format(set=chips) / CELL
                                                             / "results.json")["fit_seconds"]) / 3600},
            "joint": {"crop_pixel_macro_f1": r["crop_metrics"]["macro_f1"],
                      "crop_parcel_macro_f1": r["parcel_metrics"]["macro_f1"],
                      "border_precision_d1": r["border_metrics"]["d1"]["precision"],
                      "border_recall_d1": r["border_metrics"]["d1"]["recall"],
                      "border_f1_d1": r["border_metrics"]["d1"]["f1"],
                      "pq_agnostic_grow": jg["pq_agnostic"], "pq_grow": jg["pq"], "fields_grow": jg["predicted_parcels"],
                      "recall_by_type": {t: r["border_metrics"]["recall_by_type_d1"][t]["recall"] for t in TYPES},
                      "per_class_f1": {c: v["f1"] for c, v in r["crop_metrics"]["per_class"].items()},
                      "fit_h": r["fit_seconds"] / 3600},
            "per_chip": {"ids": j_ids, "joint": _f1_rows(jb["counts"]), "separate": _f1_rows(sep_counts),
                         "true_px": jb["counts"][:, 3]},
            "class_px": {c: v["pixels"] for c, v in r["crop_metrics"]["per_class"].items()},
        }
    return out


def fig_headline(d: dict, dest: Path) -> None:
    arms = list(d)
    fig, axes = plt.subplots(1, len(METRICS), figsize=(2.6 * len(METRICS), 3.4), sharey=False)
    w = 0.36
    for ax, (k, label) in zip(axes, METRICS.items()):
        x = np.arange(len(arms))
        for off, sys_, col in ((-w / 2, "separate", SEPARATE), (w / 2, "joint", JOINT)):
            vals = [d[a][sys_][k] for a in arms]
            ax.bar(x + off, vals, w - 0.03, color=col, label="two separate models" if sys_ == "separate"
                   else "one joint model")
            for xi, v in zip(x + off, vals):
                ax.text(xi, v + 0.008, f"{v:.3f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        lo = min(min(d[a]["separate"][k], d[a]["joint"][k]) for a in arms)
        hi = max(max(d[a]["separate"][k], d[a]["joint"][k]) for a in arms)
        ax.set_ylim(max(0, lo - 0.12), min(1.0, hi + 0.06))
        ax.set_xticks(x, [d[a]["label"].replace("TESSERA ", "") for a in arms])
        ax.set_title(label, fontsize=9.5)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.12), ncol=2)
    fig.suptitle("Each head against its separate model, test chips of Estonia, K = 100 %", x=0.01, ha="left",
                 color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(dest / "headline.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_validation(d: dict, dest: Path) -> None:
    arms = list(d)
    fig, axes = plt.subplots(2, len(arms), figsize=(5.6 * len(arms), 6.2), squeeze=False,
                             gridspec_kw={"height_ratios": [2, 1]})
    for j, a in enumerate(arms):
        h = [row for row in d[a]["results"]["history"] if "val/joint" in row]
        ep = [row["epoch"] for row in h]
        best = d[a]["results"]["best_epoch"]
        ax = axes[0, j]
        ax.plot(ep, [row["val/crop_macro_f1"] for row in h], color=JOINT, label="crop pixel Macro-F1")
        ax.plot(ep, [row["val/cc/RQ_agnostic"] for row in h], color=SEPARATE, label="object F1 of PRUE's own fields")
        ax.plot(ep, [row["val/joint"] for row in h], color=INK_2, lw=1.2, ls="--", label="their mean (checkpoint)")
        ax.axvline(best, color=MUTED, lw=1)
        ax.text(best, ax.get_ylim()[1], f" epoch {best}", color=INK_2, va="top", fontsize=8.5)
        ax.set_title(f"{d[a]['label']}: validation per epoch")
        ax.legend(loc="lower right", fontsize=8.5)
        ax = axes[1, j]
        hh = d[a]["results"]["history"]
        ax.plot([row["epoch"] for row in hh], [row["grad_unet/crop_weighted_over_border"] for row in hh],
                color=INK_2, lw=1.5, marker="o", markersize=4)
        ax.axhline(1, color=MUTED, lw=1, ls=":")
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title("pull on the U-Net: weighted crop loss / border loss", fontsize=9.5)
    fig.tight_layout()
    fig.savefig(dest / "validation_curves.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_per_class(d: dict, dest: Path) -> None:
    arms = list(d)
    a0 = arms[0]
    classes = sorted(d[a0]["class_px"], key=lambda c: -d[a0]["class_px"][c])
    y = np.arange(len(classes))
    h = 0.8 / len(arms)
    fig, ax = plt.subplots(figsize=(7.6, 0.32 * len(classes) + 1.4))
    shades = {arms[0]: JOINT, **({arms[1]: "#b04a1d"} if len(arms) > 1 else {})}
    for i, a in enumerate(arms):
        diff = [d[a]["joint"]["per_class_f1"][c] - d[a]["separate"]["per_class_f1"][c] for c in classes]
        ax.barh(y + (i - (len(arms) - 1) / 2) * h, diff, h - 0.04, color=shades[a], label=d[a]["label"])
    ax.axvline(0, color=BASELINE, lw=1)
    ax.set_yticks(y, [c.replace("_", " ") for c in classes], fontsize=8)
    ax.invert_yaxis()
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("pixel F1, joint model minus the separate crop arm")
    ax.set_title("Per crop class, most labelled pixels first")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(dest / "crop_per_class.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_border_types(d: dict, dest: Path) -> None:
    arms = list(d)
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 3.4), squeeze=False, sharey=True)
    w = 0.38
    for ax, a in zip(axes[0], arms):
        x = np.arange(len(TYPES))
        for off, sys_, col, lab in ((-w / 2, "separate", SEPARATE, "PRUE alone"), (w / 2, "joint", JOINT, "joint")):
            v = [d[a][sys_]["recall_by_type"][t] for t in TYPES]
            ax.bar(x + off, v, w - 0.03, color=col, label=lab)
            for xi, vi in zip(x + off, v):
                ax.text(xi, vi + 0.01, f"{vi:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(x, [t.replace(" ", "\n", 1) for t in TYPES], fontsize=8.5)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{d[a]['label']}: border recall at 1 px, by what lies across")
        ax.legend(loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(dest / "border_types.png", dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_per_chip(d: dict, dest: Path) -> dict:
    arms = list(d)
    stats = {}
    fig, axes = plt.subplots(1, len(arms), figsize=(5.2 * len(arms), 3.2), squeeze=False, sharey=True)
    for ax, a in zip(axes[0], arms):
        p = d[a]["per_chip"]
        ok = np.isfinite(p["joint"]) & np.isfinite(p["separate"]) & (p["true_px"] >= 100)
        diff = p["joint"][ok] - p["separate"][ok]
        stats[a] = {"chips": int(ok.sum()), "better": float((diff > 0).mean()), "median": float(np.median(diff))}
        ax.hist(diff, bins=np.linspace(-0.2, 0.2, 41), color=JOINT, edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(0, color=INK_2, lw=1)
        ax.set_xlabel("border F1 at 1 px, joint minus PRUE alone")
        ax.set_title(f"{d[a]['label']}: {stats[a]['better']:.0%} of {stats[a]['chips']} chips better")
    axes[0, 0].set_ylabel("test chips")
    fig.tight_layout()
    fig.savefig(dest / "per_chip.png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return stats


def tables(d: dict, chip_stats: dict) -> str:
    rows = ["| Encoder | Model | crop pixel Macro-F1 | crop parcel Macro-F1 | border P / R / F1 at 1 px | PQ agn. | "
            "PQ class-aware | fields | fit h |", "|---|---|---|---|---|---|---|---|---|"]
    for a, v in d.items():
        for sys_, name in (("separate", "two separate models"), ("joint", "**one joint model**")):
            s = v[sys_]
            rows.append(f"| {v['label']} | {name} | {s['crop_pixel_macro_f1']:.3f} | {s['crop_parcel_macro_f1']:.3f} | "
                        f"{s['border_precision_d1']:.3f} / {s['border_recall_d1']:.3f} / {s['border_f1_d1']:.3f} | "
                        f"{s['pq_agnostic_grow']:.3f} | {s['pq_grow']:.3f} | {s['fields_grow']:,} | {s['fit_h']:.2f} |")
        dd = {k: v["joint"][k] - v["separate"][k] for k in METRICS}
        rows.append(f"| {v['label']} | difference | {dd['crop_pixel_macro_f1']:+.3f} | {dd['crop_parcel_macro_f1']:+.3f}"
                    f" | F1 {dd['border_f1_d1']:+.3f} | {dd['pq_agnostic_grow']:+.3f} | {dd['pq_grow']:+.3f} | | |")
    out = ["## Headline", "", *rows, ""]
    out += ["## Checkpoint", "", "| Encoder | epoch kept | val joint | val object F1 | val crop Macro-F1 | epoch by "
            "object F1 alone | by crop loss alone | by crop Macro-F1 alone | crop loss weight |",
            "|---|---|---|---|---|---|---|---|---|"]
    for a, v in d.items():
        r = v["results"]
        e = r["epoch_by_single_criterion"]
        out.append(f"| {v['label']} | {r['best_epoch']} | {r['best_val_joint']:.4f} | {r['best_val_object_f1']:.4f} | "
                   f"{r['best_val_crop_macro_f1']:.4f} | {e['object_f1']} | {e['crop_loss']} | {e['crop_macro_f1']} | "
                   f"{r['config']['crop']['loss_weight']:g} |")
    out += ["", "## Border recall at 1 px by type", "", "| Encoder | Model | " + " | ".join(TYPES) + " |",
            "|---|---|" + "---|" * len(TYPES)]
    for a, v in d.items():
        for sys_, name in (("separate", "PRUE alone"), ("joint", "joint")):
            out.append(f"| {v['label']} | {name} | " + " | ".join(f"{v[sys_]['recall_by_type'][t]:.3f}" for t in TYPES)
                       + " |")
    out += ["", "## Per chip", "", "| Encoder | chips (>= 100 true border px) | joint better | median difference |",
            "|---|---|---|---|"]
    for a, s in chip_stats.items():
        out.append(f"| {d[a]['label']} | {s['chips']} | {s['better']:.1%} | {s['median']:+.4f} |")
    arms = list(d)
    classes = sorted(d[arms[0]]["class_px"], key=lambda c: -d[arms[0]]["class_px"][c])
    out += ["", "## Pixel F1 per crop class", "", "| Class | test pixels | " + " | ".join(
        f"{d[a]['label']} separate | joint" for a in arms) + " |", "|---|---|" + "---|---|" * len(arms)]
    for c in classes:
        out.append(f"| {c.replace('_', ' ')} | {d[arms[0]]['class_px'][c]:,} | " + " | ".join(
            f"{d[a]['separate']['per_class_f1'][c]:.3f} | {d[a]['joint']['per_class_f1'][c]:.3f}" for a in arms) + " |")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chips", default="EE_2021_rev")
    a = ap.parse_args()
    d = load(a.chips)
    if not d:
        raise SystemExit("no scored cell yet")
    dest = HERE / "figures"
    dest.mkdir(exist_ok=True)
    fig_headline(d, dest)
    fig_validation(d, dest)
    fig_per_class(d, dest)
    fig_border_types(d, dest)
    stats = fig_per_chip(d, dest)
    t = tables(d, stats)
    (dest / "tables.md").write_text(t)
    print(t)
    print(f"figures of {list(d)} in {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
