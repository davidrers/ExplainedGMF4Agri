"""Figures and tables of the border accuracy of PRUE, PaPs and the boundary head (``borders.py`` first).

Writes into ``figures/``: ``scores.png`` (precision, recall and F1 at one pixel), ``border_types.png`` (recall by
what lies across the true border), ``per_chip.png`` (the F1 of every test chip), ``example_<n>.png`` (frozen
encoders) and ``example_<n>_ft.png`` (fine-tuned), and ``tables.md``.

**The example chips** are picked by a fixed rule among the test chips with at least 20 declared parcels: those at
the 90th, 50th and 10th percentile of PRUE's (frozen) F1, and the one where PaPs (frozen) beats PRUE (frozen) most.

    poetry run python experiments/2026-11-07_border_accuracy/report.py --chips EE_2021_rev
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from scipy import ndimage  # noqa: E402

import borders  # noqa: E402
import panoptic_x as px  # noqa: E402
import prue  # noqa: E402

METHODS = {"PRUE": 0, "PaPs": 1, "boundary head": 2}
ENCODERS = ("frozen", "fine-tuned")
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SURFACE, INK, INK_2, MUTED, GRID, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
#: The border check: correct (a true border within one pixel), false, missed; the base of the maps.
CORRECT, FALSE, MISSED = "#2a78d6", "#eb6834", "#0b0b0b"
PARCEL_BASE, NOT_SCORED, OTHER_CLASS = "#eeede8", "#cfcec8", "#a3a29b"
PASTEL = ["#cfe0f7", "#fbd3bf", "#c4ecd9", "#fbe3a8", "#f7d1e0", "#d9d3f0", "#e6e4da", "#bfe3e8"]
CLASS3 = {prue.BACKGROUND: "#e9e2cf", prue.INTERIOR: "#a9c8f0", prue.BOUNDARY: "#0d366b"}
PROBABILITY = LinearSegmentedColormap.from_list("probability", ["#ffffff", "#2a78d6", "#0d366b"])
SHORT = {"pasture_meadow_grassland_grass": "grassland", "winter_common_soft_wheat": "winter wheat",
         "spring_common_soft_wheat": "spring wheat", "legumes_harvested_green": "legumes, green harvest",
         "winter_rapeseed_rape": "winter rapeseed", "spring_rapeseed_rape": "spring rapeseed",
         "grain_maize_corn_popcorn": "grain maize", "legumes_dried_pulses_protein_crops": "dried pulses",
         "alfalfa_lucerne": "alfalfa", "orchards_fruits": "orchards", "fresh_vegetables": "vegetables"}
MIN_PARCELS = 20
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK_2, "axes.titlecolor": INK, "axes.titlesize": 10.5,
    "axes.titlelocation": "left", "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "axes.grid.axis": "y", "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "xtick.color": INK_2, "ytick.color": INK_2, "font.size": 9.5, "legend.frameon": False,
})


def _key(method: str, enc: str) -> str:
    return f"{method} {enc}"


def _hatch(enc: str) -> str | None:
    return "///" if enc == "fine-tuned" else None


def _legend_handles(methods=True):
    h = []
    if methods:
        h += [matplotlib.patches.Patch(facecolor=SLOTS[s], label=m) for m, s in METHODS.items()]
    h += [matplotlib.patches.Patch(facecolor="white", edgecolor=INK_2, label="frozen"),
          matplotlib.patches.Patch(facecolor="white", edgecolor=INK_2, hatch="///", label="fine-tuned")]
    return h


# ------------------------------------------------------------------------------------------------- aggregate

def fig_scores(res: dict, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), sharey=True)
    for ax, (metric, title) in zip(axes, (("precision", "Precision: drawn borders that are real"),
                                          ("recall", "Recall: real borders that are drawn"),
                                          ("f1", "F1"))):
        for i, (m, s) in enumerate(METHODS.items()):
            for j, enc in enumerate(ENCODERS):
                v = res[_key(m, enc)]["d1"][metric]
                x = i + (j - 0.5) * 0.38
                ax.bar(x, v, 0.36, color=SLOTS[s], hatch=_hatch(enc), edgecolor=SURFACE, linewidth=0)
                ax.text(x, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8, color=INK_2)
        ax.set_xticks(range(len(METHODS)), list(METHODS))
        ax.set_ylim(0, 1.05)
        ax.set_title(title, fontsize=10)
    axes[0].set_ylabel("share, 1-pixel (10 m) tolerance")
    fig.legend(handles=_legend_handles(methods=False), loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def fig_types(res: dict, path: Path) -> None:
    n = {t: v["true_border_px"] for t, v in res[_key("PRUE", "frozen")]["recall_by_type_d1"].items()}
    total = sum(n.values())
    fig, ax = plt.subplots(figsize=(11, 3.8))
    w = 0.13
    for g, t in enumerate(borders.TYPES):
        k = 0
        for m, s in METHODS.items():
            for enc in ENCODERS:
                v = res[_key(m, enc)]["recall_by_type_d1"][t]["recall"]
                ax.bar(g + (k - 2.5) * (w + 0.01), v, w, color=SLOTS[s], hatch=_hatch(enc), edgecolor=SURFACE,
                       linewidth=0)
                k += 1
    ax.set_xticks(range(len(borders.TYPES)),
                  [f"{t}\n{n[t] / total:.0%} of the real border pixels" for t in borders.TYPES])
    ax.set_xlabel("what lies across the real border")
    ax.set_ylim(0, 1)
    ax.set_ylabel("recall, 1-pixel tolerance")
    ax.set_title("Which real borders are drawn", fontsize=10)
    ax.legend(handles=_legend_handles(), loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_chip_f1(pc: dict) -> dict:
    out = {}
    for m, c in pc.items():
        tp_p, n_p, tp_r, n_r = c.T.astype(float)
        p = np.divide(tp_p, n_p, out=np.zeros_like(tp_p), where=n_p > 0)
        r = np.divide(tp_r, n_r, out=np.full_like(tp_r, np.nan), where=n_r > 0)
        out[m] = np.where(p + r > 0, 2 * p * r / np.maximum(p + r, 1e-12), 0.0)
        out[m][n_r == 0] = np.nan
    return out


def fig_per_chip(f1: dict, keep: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 3.6))
    pos = 0
    ticks, labels = [], []
    for m, s in METHODS.items():
        for enc in ENCODERS:
            v = f1[_key(m, enc)][keep]
            b = ax.boxplot(v[~np.isnan(v)], positions=[pos], widths=0.6, patch_artist=True, showfliers=False,
                           medianprops={"color": INK, "linewidth": 1.5}, whiskerprops={"color": INK_2},
                           capprops={"color": INK_2})
            b["boxes"][0].set(facecolor=SLOTS[s], edgecolor=SURFACE, hatch=_hatch(enc))
            ticks.append(pos)
            labels.append(f"{m}\n{enc}")
            pos += 1
    ax.set_xticks(ticks, labels, fontsize=8.5)
    ax.set_ylim(0, 1)
    ax.set_ylabel("F1 of one chip, 1-pixel tolerance")
    ax.set_title(f"Chip by chip: the {int(keep.sum()):,} test chips with at least {MIN_PARCELS} declared parcels "
                 "(box: middle half; line: median)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# -------------------------------------------------------------------------------------------------- examples

def _fields_img(seg: np.ndarray) -> np.ndarray:
    img = np.ones(seg.shape + (3,))
    lut = np.array([matplotlib.colors.to_rgb(c) for c in PASTEL])
    on = seg > 0
    img[on] = lut[(seg[on].astype(np.int64) * 5) % len(PASTEL)]
    img[prue.boundary_map(seg)] = matplotlib.colors.to_rgb(INK_2)
    return img


def _base(E: np.ndarray, scored: np.ndarray) -> np.ndarray:
    img = np.ones(E.shape + (3,))
    img[E] = matplotlib.colors.to_rgb(PARCEL_BASE)
    img[~scored] = matplotlib.colors.to_rgb(NOT_SCORED)
    return img


def _check(seg: np.ndarray, true_border: np.ndarray, E: np.ndarray, scored: np.ndarray):
    pred = prue.boundary_map(seg) & scored
    near_true, near_pred = borders._near(true_border, 1), borders._near(prue.boundary_map(seg), 1)
    ok, false, missed = pred & near_true, pred & ~near_true, true_border & ~near_pred
    img = _base(E, scored)
    img[missed] = matplotlib.colors.to_rgb(MISSED)
    img[false] = matplotlib.colors.to_rgb(FALSE)
    img[ok] = matplotlib.colors.to_rgb(CORRECT)
    p = ok.sum() / max(pred.sum(), 1)
    r = (true_border & near_pred).sum() / max(true_border.sum(), 1)
    return img, p, r, 2 * p * r / max(p + r, 1e-12)


def _show(ax, img, title, **kw):
    ax.imshow(img, interpolation="nearest", **kw)
    ax.set_title(title, fontsize=8.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)


def _rows(rel: str, chips: str, keys: tuple[str, ...], chosen: list[str]) -> dict:
    with np.load(REPO / "results" / rel.format(set=chips) / borders.CELL / "predictions_test.npz") as d:
        order = {str(c): i for i, c in enumerate(d["ids"])}
        return {k: d[k][[order[c] for c in chosen]] for k in keys}


def pick(f1: dict, keep: np.ndarray, ids: list[str]) -> list[tuple[str, str]]:
    prue_f1, paps_f1 = f1[_key("PRUE", "frozen")], f1[_key("PaPs", "frozen")]
    ok = keep & ~np.isnan(prue_f1)
    idx = np.nonzero(ok)[0]
    order = idx[np.argsort(prue_f1[idx], kind="stable")]
    out = []
    for q, why in ((0.9, "PRUE at its 90th percentile"), (0.5, "PRUE at its median"),
                   (0.1, "PRUE at its 10th percentile")):
        out.append((ids[order[int(round(q * (len(order) - 1)))]], why))
    gap = np.where(ok, paps_f1 - prue_f1, -np.inf)
    out.append((ids[int(np.argmax(gap))], "where PaPs beats PRUE most"))
    return out


def fig_examples(chips: str, chosen: list[tuple[str, str]], folder: Path) -> list[Path]:
    import rasterio

    from gfm4agri.benchmark.parcel_metrics import read_parcels

    chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
    cids = [c for c, _ in chosen]
    src = borders.SOURCES
    names = json.loads((REPO / "results" / src["PRUE frozen"][0].format(set=chips) / borders.CELL /
                        "results.json").read_text())["class_names"]
    data = {}
    for enc in ENCODERS:
        data[enc] = {"PRUE": _rows(src[_key("PRUE", enc)][0], chips, ("prob", "mask"), cids),
                     "PaPs": _rows(src[_key("PaPs", enc)][0], chips, ("inst", "heatmap", "saliency"), cids),
                     "head": _rows(src[_key("boundary head", enc)][0], chips, ("boundary",), cids)}
    manifest = json.loads((REPO / "data/eurocrops_chips" / chips / "manifest.json").read_text())
    bands = manifest["imagery"]["band_names"]
    wc_all = np.load(REPO / "data/worldcover" / chips.replace("_mini", "") / "worldcover.npz")

    masks = data["frozen"]["PRUE"]["mask"].astype(np.int64)
    area = np.zeros(len(names), np.int64)
    for mk in masks:
        area += np.bincount(mk[mk >= 0], minlength=len(names))
    ranked = [int(k) for k in np.argsort(-area, kind="stable") if area[k] > 0]
    colours = dict(zip(ranked, SLOTS))

    paths = []
    for i, (cid, why) in enumerate(chosen):
        parcels, mask, wc = read_parcels(chips_dir, cid), masks[i], wc_all[cid]
        E, scored, true_border = borders.regions(mask, parcels, wc)
        n_parcels = int(np.unique(px.truth(mask, parcels)[1][E]).size)
        with rasterio.open(chips_dir / f"{cid}_merged.tif") as s:
            july = 6 * len(bands)
            rgb = np.stack([s.read(july + bands.index(b) + 1) for b in ("RED", "GREEN", "BLUE")], -1)
        rgb = np.nan_to_num(np.clip(rgb.astype(np.float32) / np.nanpercentile(rgb, 98), 0, 1))
        crop = np.ones(E.shape + (3,))
        for k in np.unique(mask[E]):
            crop[E & (mask == k)] = matplotlib.colors.to_rgb(colours.get(int(k), OTHER_CLASS))
        crop[(mask < 0) & (parcels > 0)] = matplotlib.colors.to_rgb(NOT_SCORED)
        crop[prue.boundary_map(parcels)] = matplotlib.colors.to_rgb(INK)
        truth_img = _base(E, scored)
        truth_img[true_border] = matplotlib.colors.to_rgb(INK)

        for enc in ENCODERS:
            d = data[enc]
            prob = d["PRUE"]["prob"][i]
            pb = d["head"]["boundary"][i].astype(np.float32) / 255.0
            segs = {"PRUE": prue.parcels(prob, "grow"), "PaPs": d["PaPs"]["inst"][i].astype(np.int64),
                    "boundary head": px.from_boundary(np.ones(pb.shape, bool), pb, px.TAU)}
            fig, axes = plt.subplots(4, 4, figsize=(11, 11.9))
            _show(axes[0, 0], rgb, "Sentinel-2, July composite")
            _show(axes[0, 1], crop, f"Declared parcels: {n_parcels},\ncoloured by crop")
            _show(axes[0, 2], truth_img, f"Declared borders (black): {int(true_border.sum()):,} px\n"
                                         "grey: not scored")
            cm = np.ones(E.shape + (3,))
            for k, c in CLASS3.items():
                cm[prob.argmax(0) == k] = matplotlib.colors.to_rgb(c)
            _show(axes[1, 0], cm, "PRUE: class per pixel\nbackground, interior, boundary")
            _show(axes[1, 1], prob[prue.BOUNDARY] / 255.0, "PRUE: probability of\nboundary (0 white, 1 dark)",
                  cmap=PROBABILITY, vmin=0, vmax=1)
            heat = ndimage.maximum_filter(d["PaPs"]["heatmap"][i].astype(np.float32), size=5)
            _show(axes[2, 0], heat / max(heat.max(), 1.0), "PaPs: centre heatmap, peaks = centres\n"
                  "(scaled to its maximum, dots enlarged)", cmap=PROBABILITY, vmin=0, vmax=1)
            _show(axes[2, 1], d["PaPs"]["saliency"][i] / 255.0, "PaPs: saliency\n(part of a parcel?)",
                  cmap=PROBABILITY, vmin=0, vmax=1)
            _show(axes[3, 0], pb, "Boundary head: probability\nof border (0 white, 1 dark)",
                  cmap=PROBABILITY, vmin=0, vmax=1)
            cores = prue.pieces((pb < px.TAU).astype(np.int32))
            _show(axes[3, 1], _fields_img(cores), "Boundary head: cores,\nborder probability below 0.5")
            for row, (m, seg) in enumerate(segs.items(), start=1):
                n_f = int(np.unique(seg[seg > 0]).size)
                _show(axes[row, 2], _fields_img(seg), f"{m}: final fields ({n_f})")
                img, p, r, f = _check(seg, true_border, E, scored)
                _show(axes[row, 3], img, f"{m}: border check\nprecision {p:.2f}, recall {r:.2f}, F1 {f:.2f}")
            ax = axes[0, 3]
            ax.axis("off")
            present = [k for k in ranked if k in set(np.unique(mask[E]).tolist()) and k in colours]
            crops = [matplotlib.patches.Patch(color=colours[k], label=SHORT.get(names[k], names[k].replace("_", " ")))
                     for k in present]
            if set(np.unique(mask[E]).tolist()) - set(colours):
                crops.append(matplotlib.patches.Patch(color=OTHER_CLASS, label="other crops"))
            if ((mask < 0) & (parcels > 0)).any():
                crops.append(matplotlib.patches.Patch(color=NOT_SCORED, label="crop outside the 20 classes"))
            ax.legend(handles=crops, loc="upper left", fontsize=7.5, title="crop", title_fontsize=8,
                      alignment="left", handlelength=1.2, borderaxespad=0)
            check = [matplotlib.patches.Patch(color=CORRECT, label="drawn, real border within 10 m"),
                     matplotlib.patches.Patch(color=FALSE, label="drawn, no real border within 10 m"),
                     matplotlib.patches.Patch(color=MISSED, label="real border, nothing drawn within 10 m"),
                     matplotlib.patches.Patch(facecolor=PARCEL_BASE, edgecolor=BASELINE, label="declared parcel"),
                     matplotlib.patches.Patch(color=NOT_SCORED, label="not scored: undeclared farmland,\n"
                                                                     "crop outside the 20 classes")]
            fig.legend(handles=check, loc="lower center", ncol=3, fontsize=8, title="border check (last column)",
                       title_fontsize=8.5, handlelength=1.2, bbox_to_anchor=(0.5, 0.0))
            fig.suptitle(f"Test chip {cid}, TESSERA {enc}: {why}", x=0.01, ha="left", fontsize=11, color=INK)
            fig.tight_layout(rect=(0, 0.06, 1, 0.985))
            path = folder / (f"example_{i + 1}.png" if enc == "frozen" else f"example_{i + 1}_ft.png")
            fig.savefig(path, dpi=130)
            plt.close(fig)
            paths.append(path)
    return paths


# ---------------------------------------------------------------------------------------------------- tables

def tables(res: dict, f1: dict, keep: np.ndarray, chosen) -> str:
    t = ["**Border precision, recall and F1** over the 1,475 test chips, by tolerance\n",
         "| Method | Encoder | " + " | ".join(f"{h}, {d} px" for d in borders.TOLERANCES
                                             for h in ("precision", "recall", "F1")) + " |",
         "|---|---|" + "---|" * 3 * len(borders.TOLERANCES)]
    for m in METHODS:
        for enc in ENCODERS:
            r = res[_key(m, enc)]
            t.append(f"| {m} | {enc} | " + " | ".join(f"{r[f'd{d}'][h]:.3f}" for d in borders.TOLERANCES
                                                      for h in ("precision", "recall", "f1")) + " |")
    r0 = res[_key("PRUE", "frozen")]["recall_by_type_d1"]
    total = sum(v["true_border_px"] for v in r0.values())
    t += ["", "**Recall at 1 px by what lies across the real border** (share of the real border pixels)\n",
          "| Method | Encoder | " + " | ".join(f"{k} ({r0[k]['true_border_px'] / total:.1%})"
                                             for k in borders.TYPES) + " |",
          "|---|---|" + "---|" * len(borders.TYPES)]
    for m in METHODS:
        for enc in ENCODERS:
            r = res[_key(m, enc)]["recall_by_type_d1"]
            t.append(f"| {m} | {enc} | " + " | ".join(f"{r[k]['recall']:.3f}" for k in borders.TYPES) + " |")
    t += ["", "**Where the drawn borders lie, at 1 px** (shares of all drawn border pixels that are scored)\n",
          "| Method | Encoder | drawn border px | real | false, on a declared parcel | false, on non-farm land |",
          "|---|---|---|---|---|---|"]
    for m in METHODS:
        for enc in ENCODERS:
            r = res[_key(m, enc)]
            fs = r["false_border_share_d1"]
            t.append(f"| {m} | {enc} | {r['d1']['pred_border_px']:,} | {r['d1']['precision']:.3f} | "
                     f"{fs['on a declared parcel']:.3f} | {fs['on non-farm land']:.3f} |")
    t += ["", f"**Chip by chip** ({int(keep.sum()):,} test chips with at least {MIN_PARCELS} declared parcels), "
          "F1 at 1 px\n", "| Method | Encoder | median | 25th pct | 75th pct |", "|---|---|---|---|---|"]
    for m in METHODS:
        for enc in ENCODERS:
            v = f1[_key(m, enc)][keep]
            v = v[~np.isnan(v)]
            t.append(f"| {m} | {enc} | {np.median(v):.3f} | {np.percentile(v, 25):.3f} | "
                     f"{np.percentile(v, 75):.3f} |")
    for enc in ENCODERS:
        a, b = f1[_key("PRUE", enc)][keep], f1[_key("PaPs", enc)][keep]
        ok = ~np.isnan(a) & ~np.isnan(b)
        t.append(f"\nPRUE above PaPs ({enc}) on {(a[ok] > b[ok]).mean():.1%} of these chips.")
    t += ["", "**Example chips**\n", "| Figure | Chip | Why |", "|---|---|---|"]
    t += [f"| example_{i + 1} | {c} | {w} |" for i, (c, w) in enumerate(chosen)]
    return "\n".join(t) + "\n"


def _n_parcels(mask: np.ndarray, parcels: np.ndarray) -> int:
    true = px.truth(mask.astype(np.int64), parcels)[1]
    return int(np.unique(true[true > 0]).size)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chips", default="EE_2021_rev")
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args()
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    res_dir = REPO / "results" / borders.NAME / a.chips
    res = json.loads((res_dir / "borders.json").read_text())["methods"]
    with np.load(res_dir / "per_chip.npz") as z:
        ids, methods, counts = [str(c) for c in z["ids"]], [str(m) for m in z["methods"]], z["counts"]
    f1 = per_chip_f1(dict(zip(methods, counts)))
    chips_dir = REPO / "data/eurocrops_chips" / a.chips / "chips"
    with np.load(REPO / "results" / borders.SOURCES["PRUE frozen"][0].format(set=a.chips) / borders.CELL /
                 "predictions_test.npz") as z:
        masks = z["mask"]
    n_parcels = np.array([_n_parcels(masks[i], read_parcels(chips_dir, c)) for i, c in enumerate(ids)])
    keep = n_parcels >= MIN_PARCELS

    folder = HERE / "figures" if a.chips == "EE_2021_rev" else HERE / "figures" / a.chips
    folder.mkdir(parents=True, exist_ok=True)
    fig_scores(res, folder / "scores.png")
    fig_types(res, folder / "border_types.png")
    fig_per_chip(f1, keep, folder / "per_chip.png")
    chosen = pick(f1, keep, ids)
    if not a.no_examples:
        fig_examples(a.chips, chosen, folder)
    t = tables(res, f1, keep, chosen)
    (folder / "tables.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
