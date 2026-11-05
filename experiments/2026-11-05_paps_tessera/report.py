"""Figures and tables of the PaPs experiment, for its REPORT.md, in the style of the delineation report
(``../2026-10-18_parcel_delineation/REPORT.md``).

Reads ``results/<name>/<chip set>/`` as ``run.py`` and ``analyse.py`` write it, and, as the reference, the boundary
parcels of the same TESSERA encoders (``results/2026-11-05_parcel_extent/<set>/*/P100*/extent_scores.json``, tau 0.5)
and their predictions. Only K = 100 %, the experiment's one budget. Writes into ``figures/`` beside this file:

* ``example_<n>.png``: the delineation report's three test chips, a row per stage: the geometry heads (PaPs' centre
  heatmap and saliency, the boundary head's probability), the class (PaPs' class head on its parcels, the crop-type
  maps), the parcels (PaPs with its own class, PaPs with the crop map's class, the boundary parcels over the whole
  chip);
* ``class_heads.png``: on the true parcels PaPs found, how often PaPs' class head and the crop map's vote give the
  true class;
* ``pq_bars.png``: PQ of PaPs' parcels, with its own class and with the crop map's, against the boundary parcels,
  over the whole chip and inside WorldCover, and, shaded apart, cut to the declared parcels (the upper bound, which
  uses the test labels);
* ``geometry_cost.png``: the parcel Macro-F1 at IoU > 0.5 (class-aware RQ) of the true parcels against the predicted
  ones;
* ``where_found.png``: the share of the true parcels found, by what they touch; ``found_by_size.png``: by area;
* ``fp_where.png``: where PaPs' false parcels lie;
* ``validation_curves.png``: training loss, validation PQ and validation parcels per epoch, with the epoch kept;
* ``tables.md``: the tables of REPORT.md, also printed.

    poetry run python experiments/2026-11-05_paps_tessera/analyse.py --chips EE_2021_rev
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

import analyse  # noqa: E402
import panoptic_x as px  # noqa: E402

NAME = "2026-11-05_paps_tessera"
REFERENCE = "2026-11-05_parcel_extent"
PCT = 100
#: Arm -> (label, slot, the boundary-head arm on the same encoder, its short label).
ARMS = {"paps_tessera_v2_frozen": ("TESSERA frozen", 0, "tessera_v2_medium"),
        "paps_tessera_v2_ft": ("TESSERA fine-tuned", 1, "tessera_v2_medium_ft_bnd")}
#: The extents of the bar figure: the two that use no test label, then the upper bound.
EXTENTS = {"chip": "whole chip", "worldcover": "WorldCover 30 + 40", "oracle": "declared parcels"}
SYSTEMS = {"paps": "PaPs, its own class", "paps_cropmap": "PaPs parcels, crop-map class",
           "boundary": "boundary head (tau 0.5)", "semantic": "crop-map regions", "floor": "extent alone"}
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


def load(chips: str) -> tuple[pd.DataFrame, dict, dict]:
    """Every score as one row (arm, extent, system), the PaPs fits, and the analyses."""
    rows, cells, analyses = [], {}, {}
    for arm, (_, _, ref) in ARMS.items():
        cell = REPO / "results" / NAME / chips / arm / f"P{PCT}_draw0_seed0"
        if (cell / "results.json").exists():
            r = json.loads((cell / "results.json").read_text())
            if "panoptic_metrics" in r:
                cells[arm] = r
                rows += _rows(r["panoptic_metrics"]["variants"], {"arm": arm}, {"paps": "paps"})
        if (cell / "analysis.json").exists():
            a = json.loads((cell / "analysis.json").read_text())
            analyses[arm] = a
            rows += _rows(a["variants"], {"arm": arm}, {"paps_cropmap": "paps_cropmap", "cropmap": "truth_cropmap"})
        f = REPO / "results" / REFERENCE / chips / ref / f"P{PCT}_draw0_seed0" / "extent_scores.json"
        if f.exists():
            s = json.loads(f.read_text())
            rows += _rows(s["variants"], {"arm": arm},
                          {"boundary@0.5": "boundary", "extent": "floor", "semantic": "semantic"})
    return pd.DataFrame(rows), cells, analyses


def _get(df, arm, system, extent, metric):
    q = df[(df.arm == arm) & (df.system == system) & (df.extent == extent)]
    return float(q[metric].iloc[0]) if len(q) else np.nan


# ------------------------------------------------------------------------------------------------------ examples

#: Short labels of the scheme's classes, as in the delineation report.
SHORT = {"pasture_meadow_grassland_grass": "grassland", "winter_common_soft_wheat": "winter wheat",
         "spring_common_soft_wheat": "spring wheat", "legumes_harvested_green": "legumes, green harvest",
         "winter_rapeseed_rape": "winter rapeseed", "spring_rapeseed_rape": "spring rapeseed",
         "grain_maize_corn_popcorn": "grain maize", "legumes_dried_pulses_protein_crops": "dried pulses",
         "alfalfa_lucerne": "alfalfa", "orchards_fruits": "orchards", "fresh_vegetables": "vegetables"}
OTHER_CLASS = "#a3a29b"
SCL_VEGETATION = 4


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


def pick_chips(chips_dir: Path, ids: list[str], n: int = 3) -> list[str]:
    """The delineation report's choice: the test chips with the most contacts between same-class parcels."""
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


def _show_classes(ax, cls, where, colours, title, outline=None):
    img = np.ones(cls.shape + (3,))
    for k in np.unique(cls[where]):
        img[where & (cls == k)] = matplotlib.colors.to_rgb(colours.get(int(k), OTHER_CLASS))
    if outline is not None:
        img[outline] = matplotlib.colors.to_rgb(INK)
    ax.imshow(img, interpolation="nearest")
    ax.set_title(title, fontsize=9)
    ax.axis("off")


def _rows_of(path: Path, keys: tuple[str, ...], chosen: list[str]) -> dict | None:
    if not path.exists():
        return None
    with np.load(path) as d:
        order = {str(c): i for i, c in enumerate(d["ids"])}
        at = [order[c] for c in chosen]
        return {k: d[k][at] for k in keys if k in d.files}


def fig_examples(chips: str, folder: Path) -> list[Path]:
    """One figure per chip, a row per stage and three columns per encoder: PaPs, PaPs with the crop map, the boundary
    head over the whole chip."""
    import rasterio

    import targets
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
    first = REPO / "results" / NAME / chips / next(iter(ARMS)) / f"P{PCT}_draw0_seed0"
    if not (first / "predictions_test.npz").exists():
        return []
    names = json.loads((first / "results.json").read_text())["class_names"]
    n_cls = len(names)
    with np.load(first / "predictions_test.npz") as d:
        ids = [str(c) for c in d["ids"]]
    chosen = pick_chips(chips_dir, ids)
    data = {}
    for arm, (label, _, ref) in ARMS.items():
        paps = _rows_of(REPO / "results" / NAME / chips / arm / f"P{PCT}_draw0_seed0" / "predictions_test.npz",
                        ("inst", "cls", "heatmap", "saliency", "mask"), chosen)
        if paps is None:
            continue
        src = {k: REPO / v.format(set=chips) for k, v in analyse.SOURCES[arm].items()}
        data[arm] = {"paps": paps, "crop": _rows_of(src["cropmap"] / "predictions_test.npz", ("pred",), chosen),
                     "bnd": _rows_of(src["boundary"] / "predictions_test.npz", ("pred", "boundary"), chosen)}
    manifest = json.loads((REPO / "data/eurocrops_chips" / chips / "manifest.json").read_text())
    bands = manifest["imagery"]["band_names"]
    scl_f = REPO / "results/2026-10-18_parcel_delineation" / chips / "scl_july.npz"
    scl = dict(np.load(scl_f)) if scl_f.exists() else {}
    wc_all = np.load(REPO / "data/worldcover" / chips.replace("_mini", "") / "worldcover.npz")

    def chip_pq(true, true_cls, seg, seg_cls, void):
        aware, agn, _ = px.chip_counts(true, true_cls, seg, seg_cls, void, n_cls)
        return int(agn[0]), int(aware[:, 0].sum()), px.quality(aware, agn, names)["pq"]

    shown = []
    for i, cid in enumerate(chosen):
        parcels = read_parcels(chips_dir, cid)
        mask = next(iter(data.values()))["paps"]["mask"][i].astype(np.int64)
        E, true, true_cls, void = px.truth(mask, parcels)
        veg = scl[cid] == SCL_VEGETATION if cid in scl else np.ones(E.shape, bool)
        centres = np.argwhere(targets.build(mask, parcels, wc_all[cid])[..., 0] == 1)
        panels = {}
        for arm, d in data.items():
            inst = d["paps"]["inst"][i].astype(np.int64)
            pcls = np.zeros(int(inst.max()) + 1, np.int64)
            pcls[inst[inst > 0]] = d["paps"]["cls"][i][inst > 0]
            p = {"heat": d["paps"]["heatmap"][i].astype(np.float32), "sal": d["paps"]["saliency"][i] / 255.0,
                 "inst": inst, "pcls": pcls, "n": int(np.unique(inst[inst > 0]).size)}
            p["own"] = chip_pq(true, true_cls, inst, pcls, void)
            cover = E & (inst > 0)
            p["cover"] = float(cover.sum() / max(E.sum(), 1))
            p["cover_ok"] = float((pcls[inst][cover] == mask[cover]).mean()) if cover.any() else 0.0
            if d["crop"] is not None:
                crop = d["crop"]["pred"][i]
                p["crop"] = crop
                p["crop_ok"] = float((crop[E] == mask[E]).mean())
                p["ccls"] = px.vote(inst, crop, n_cls)
                p["cm"] = chip_pq(true, true_cls, inst, p["ccls"], void)
            if d["bnd"] is not None:
                prob = d["bnd"]["boundary"][i].astype(np.float32) / 255.0
                bpred = d["bnd"]["pred"][i]
                seg = px.from_boundary(np.ones(E.shape, bool), prob, px.TAU)
                target = E & analyse_boundary_target(parcels)
                hit = E & (prob >= px.TAU)
                tp, fp, fn = (hit & target).sum(), (hit & ~target).sum(), (target & ~hit).sum()
                p.update(prob=prob, bpred=bpred, bseg=seg, bcls=px.vote(seg, bpred, n_cls),
                         bf1=2 * tp / max(2 * tp + fp + fn, 1), bpred_ok=float((bpred[E] == mask[E]).mean()))
                p["bnd"] = chip_pq(true, true_cls, seg, p["bcls"], void)
            panels[arm] = p
        shown.append((cid, E, veg, true, true_cls, mask, centres, panels))

    # One colour per class across the figures, ranked by the pixels a class covers in all the class maps shown.
    area = np.zeros(n_cls, np.int64)
    for _, E, veg, _, _, mask, _, panels in shown:
        area += np.bincount(mask[E], minlength=n_cls)
        for p in panels.values():
            for key in ("crop", "bpred"):
                if key in p:
                    area += np.bincount(p[key][veg].astype(np.int64), minlength=n_cls)
    ranked = [int(k) for k in np.argsort(-area, kind="stable") if area[k] > 0]
    colours = dict(zip(ranked, SLOTS))

    paths = []
    for r, (cid, E, veg, true, true_cls, mask, centres, panels) in enumerate(shown, start=1):
        cols = [(arm, kind) for arm, p in panels.items() for kind in ("paps", "cm", "bnd")
                if kind == "paps" or (kind == "cm" and "crop" in p) or (kind == "bnd" and "prob" in p)]
        ncol = 2 + len(cols)
        fig, (geo, crop_row, parts) = plt.subplots(3, ncol, figsize=(2.45 * ncol, 3.0 * 3))
        with rasterio.open(chips_dir / f"{cid}_merged.tif") as s:
            july = 6 * len(bands)
            rgb = np.stack([s.read(july + bands.index(b) + 1) for b in ("RED", "GREEN", "BLUE")], -1)
        rgb = np.clip(rgb.astype(np.float32) / np.nanpercentile(rgb, 98), 0, 1)
        geo[0].imshow(np.nan_to_num(rgb), interpolation="nearest")
        geo[0].set_title(f"{cid}\nJuly" + (f", {veg.mean():.0%} SCL\nvegetation" if cid in scl else "\n"), fontsize=9)
        geo[0].axis("off")

        # 1. Geometry: the true parcels and their centres; PaPs' heatmap and saliency; the boundary probability.
        img = np.ones(E.shape + (3,))
        img[E] = matplotlib.colors.to_rgb("#e3eefb")
        img[_outline(true) & E] = matplotlib.colors.to_rgb("#0d366b")
        geo[1].imshow(img, interpolation="nearest")
        geo[1].scatter(centres[:, 1], centres[:, 0], s=5, color=SLOTS[1], lw=0)
        geo[1].set_title(f"1. geometry\ntrue parcels\nand their {len(centres)} centres", fontsize=9)
        geo[1].axis("off")
        for ax, (arm, kind) in zip(geo[2:], cols):
            p, lab = panels[arm], ARMS[arm][0]
            if kind == "paps":
                h = p["heat"]
                ax.imshow(h, cmap=PROBABILITY, vmin=0, vmax=max(1.0, float(np.percentile(h, 99.9))),
                          interpolation="nearest")
                ax.set_title(f"PaPs, {lab}\ncentre heatmap\n{p['n']} parcels kept", fontsize=9)
            elif kind == "cm":
                ax.imshow(p["sal"], cmap=PROBABILITY, vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"PaPs, {lab}\nsaliency\n(parcel or not)", fontsize=9)
            else:
                ax.imshow(p["prob"], cmap=PROBABILITY, vmin=0, vmax=1, interpolation="nearest")
                ax.set_title(f"boundary head, {lab}\nboundary probability\npixel F1 {p['bf1']:.2f}", fontsize=9)
            ax.axis("off")

        # 2. Class: the true classes; PaPs' class head on its parcels; the crop-type maps where SCL sees vegetation.
        _show_classes(crop_row[1], mask, E, colours, "2. class\ntrue crop types\n")
        for ax, (arm, kind) in zip(crop_row[2:], cols):
            p, lab = panels[arm], ARMS[arm][0]
            if kind == "paps":
                _show_classes(ax, p["pcls"][p["inst"]], p["inst"] > 0, colours,
                              f"PaPs class head, {lab}\non {p['cover']:.0%} of the extent\n{p['cover_ok']:.0%} right there")
            elif kind == "cm":
                _show_classes(ax, p["crop"], veg, colours, f"crop-type map, {lab}\n(per-pixel MLP)\n{p['crop_ok']:.0%} correct")
            else:
                _show_classes(ax, p["bpred"], veg, colours, f"its crop-type head, {lab}\n(joint fit)\n{p['bpred_ok']:.0%} correct")

        # 3. Parcels, each coloured by its class, outlines in black, over the whole chip.
        _show_classes(parts[1], true_cls[true], E, colours,
                      f"3. parcels\ntrue: {int(np.unique(true[E]).size)} parcels\n", _outline(true) & E)
        for ax, (arm, kind) in zip(parts[2:], cols):
            p, lab = panels[arm], ARMS[arm][0]
            if kind == "paps":
                seg, cl, (ag, m, q), t = p["inst"], p["pcls"], p["own"], f"PaPs, {lab}\nits own class"
            elif kind == "cm":
                seg, cl, (ag, m, q), t = p["inst"], p["ccls"], p["cm"], f"PaPs, {lab}\ncrop-map class"
            else:
                seg, cl, (ag, m, q), t = p["bseg"], p["bcls"], p["bnd"], f"boundary head, {lab}\nwhole chip"
            n_pred = int(np.unique(seg[seg > 0]).size)
            _show_classes(ax, cl[seg], seg > 0, colours,
                          f"{t}\n{n_pred} parcels, {ag} found\n{m} with class, PQ {q:.2f}", _outline(seg))

        present = {int(k) for k in np.unique(mask[E])}
        for p in panels.values():
            for key in ("crop", "bpred"):
                if key in p:
                    present |= {int(k) for k in np.unique(p[key][veg])}
            present |= {int(k) for k in np.unique(p["pcls"][p["inst"]][p["inst"] > 0])}
        handles = [matplotlib.patches.Patch(color=colours[k], label=SHORT.get(names[k], names[k].replace("_", " ")))
                   for k in ranked if k in present and k in colours]
        if present - set(colours):
            handles.append(matplotlib.patches.Patch(color=OTHER_CLASS, label="other classes"))
        crop_row[0].axis("off")
        crop_row[0].legend(handles=handles, loc="center left", fontsize=8, handlelength=1.2, borderaxespad=0,
                           title="crop type", title_fontsize=8.5, alignment="left")
        parts[0].axis("off")
        parts[0].text(0.02, 0.95, "PaPs: a parcel per centre\nkept on the heatmap, its\nmask from the shape head,\n"
                      "its class from the class\nhead, or the class most\nof its pixels take in the\ncrop-type map.\n\n"
                      "Boundary head: seeds\nbelow 0.5 grown by\nwatershed over the whole\nchip, classed by vote.",
                      transform=parts[0].transAxes, fontsize=8, color=INK_2, va="top")
        fig.text(0.01, 0.0, "Row 1: the geometry each method predicts over the whole chip: PaPs' centre heatmap (scaled "
                 "to the chip's 99.9th centile) and its saliency, and the boundary head's probability with its pixel F1 "
                 "at 0.5 inside the declared extent. Row 2: the class: PaPs' class head on its own parcels (share of the "
                 "declared extent covered, and right there), the crop-type maps where the July scene classification "
                 "calls vegetation, with the share of the declared extent correct. Row 3: the parcels, no extent given; "
                 "how many there are, how many match a true parcel at IoU > 0.5 (found), how many of those also agree on "
                 "the class, and the class-aware PQ of this chip. K = 100 %.",
                 color=MUTED, fontsize=8.5, wrap=True, va="top")
        fig.tight_layout()
        path = folder / f"example_{r}.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths


def analyse_boundary_target(parcels: np.ndarray) -> np.ndarray:
    """A parcel pixel with a 4-neighbour inside the chip of another parcel or of none (the delineation target)."""
    t = np.zeros(parcels.shape, bool)
    for a, b, sa, sb in ((parcels[:-1], parcels[1:], np.s_[:-1], np.s_[1:]),
                         (parcels[:, :-1], parcels[:, 1:], np.s_[:, :-1], np.s_[:, 1:])):
        d = a != b
        t[sa] |= d & (a > 0)
        t[sb] |= d & (b > 0)
    return t


# ------------------------------------------------------------------------------------------------------- figures

def fig_class_heads(analyses: dict, path: Path) -> None:
    if not analyses:
        return
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    groups = [("accuracy", "accuracy"), ("macro_f1", "macro F1 over the classes")]
    arms = list(analyses)
    w = 0.8 / (2 * len(arms))
    for g, (metric, _) in enumerate(groups):
        for a, arm in enumerate(arms):
            com = analyses[arm]["class_on_matched"]
            slot = SLOTS[ARMS[arm][1]]
            for b, (key, hatch, lab) in enumerate((("paps_class_head", None, "PaPs class head"),
                                                    ("cropmap_vote", "...", "crop-type map, vote"))):
                x = g + ((2 * a + b) - (2 * len(arms) - 1) / 2) * w
                v = com[key][metric]
                ax.bar(x, v, w * 0.92, color=SURFACE if hatch else slot, edgecolor=slot, hatch=hatch, lw=1.2,
                       label=f"{lab}, {ARMS[arm][0]}" if g == 0 else None)
                ax.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, color=INK_2)
    ax.set_xticks(range(len(groups)), [g[1] for g in groups])
    ax.set_ylim(0, 1.0)
    n = ", ".join(f"{a['class_on_matched']['parcels']:,} {ARMS[k][0].split()[1]}" for k, a in analyses.items())
    ax.set_title(f"The class of the true parcels PaPs found ({n})")
    ax.legend(fontsize=8, loc="upper right", ncol=1)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_pq(df: pd.DataFrame, path: Path) -> None:
    """Grouped bars per extent; the declared parcels, which use the test labels, shaded apart."""
    arms = [a for a in ARMS if not df[(df.arm == a) & (df.system == "paps")].empty]
    exts = list(EXTENTS)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    panels = ((axes[0], "pq_agnostic", "PQ, class-agnostic", "Class-agnostic: the geometry alone",
               [("paps", None, "PaPs"), ("boundary", "///", "boundary head")]),
              (axes[1], "pq", "PQ, class-aware", "Class-aware: parcel and class",
               [("paps", None, "PaPs, its own class"), ("paps_cropmap", "...", "PaPs, crop-map class"),
                ("boundary", "///", "boundary head")]))
    for ax, metric, ylab, title, systems in panels:
        bars = [(arm, s, h, f"{lab}, {ARMS[arm][0]}") for s, h, lab in systems for arm in arms
                if not df[(df.arm == arm) & (df.system == s)].empty]
        w = 0.84 / max(len(bars), 1)
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
                ax.text(x, v + 0.008, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color=INK_2)
                top = max(top, v)
        if metric == "pq_agnostic":
            for j, ext in enumerate(exts):
                f = _get(df, arms[0], "floor", ext, metric) if arms else np.nan
                if not np.isnan(f):
                    ax.hlines(f, j - 0.44, j + 0.44, color=INK, lw=1.6, ls="--")
        ax.set_ylim(0, top * 1.22 + 0.02)
        ax.text(2, top * 1.17 + 0.02, "upper bound:\nuses the test labels", ha="center", va="top", fontsize=8,
                color=INK_2)
        ax.set_xticks(range(len(exts)), ["whole chip\n(no extent)", "cut to\nWorldCover 30 + 40",
                                         "cut to the\ndeclared parcels"], fontsize=8.5)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        h, lab = ax.get_legend_handles_labels()
        if metric == "pq_agnostic":
            h.append(Line2D([], [], color=INK, lw=1.6, ls="--"))
            lab.append("floor: the extent's own regions")
        ax.legend(h, lab, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2, fontsize=7.5)
    fig.suptitle(f"PaPs' parcels against the boundary head's, same encoders, K = {PCT} %", x=0.01, ha="left",
                 fontsize=11.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_geometry_cost(df: pd.DataFrame, path: Path) -> None:
    arms = [a for a in ARMS if not df[(df.arm == a) & (df.system == "truth_cropmap")].empty]
    if not arms:
        return
    marks = [("paps_cropmap", "o", True, "PaPs parcels, crop-map class"), ("paps", "o", False, "PaPs, its own class"),
             ("boundary", "X", True, "boundary head")]
    fig, ax = plt.subplots(figsize=(8.5, 0.9 * len(arms) + 1.8))
    for i, arm in enumerate(arms):
        slot = SLOTS[ARMS[arm][1]]
        ref = _get(df, arm, "truth_cropmap", "truth", "rq")
        vals = [(_get(df, arm, s, "chip", "rq"), m, filled, lab) for s, m, filled, lab in marks]
        lo = np.nanmin([v for v, *_ in vals])
        ax.hlines(i, lo, ref, color=BASELINE, lw=3, zorder=1)
        ax.scatter([ref], [i], s=70, color=INK_2, edgecolor=SURFACE, lw=1.5, zorder=3,
                   label="true parcels, crop-map class" if i == 0 else None)
        for v, m, filled, lab in vals:
            if np.isnan(v):
                continue
            ax.scatter([v], [i], s=70, marker=m, color=slot if filled else SURFACE, edgecolor=slot, lw=1.6, zorder=4,
                       label=lab if i == 0 else None)
        v0 = vals[0][0]
        ax.text(ref + 0.008, i, f"{v0 - ref:+.3f}", va="center", fontsize=8.5, color=INK)
    ax.set_yticks(range(len(arms)), [ARMS[a][0] for a in arms])
    ax.set_ylim(-0.7, len(arms) - 0.3)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("RQ, parcel Macro-F1 at IoU > 0.5, whole chip")
    ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2)
    fig.suptitle("The parcel F1 lost to predicted geometry, whole chip\nvalue: PaPs with the crop-map class minus "
                 "the true parcels with the crop-map class", x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


FOUND = [("paps", "o", True, "PaPs, whole chip"), ("boundary/chip", "o", False, "boundary head, whole chip"),
         ("boundary/worldcover", "s", False, "boundary head, WorldCover"),
         ("boundary/oracle", "|", None, "boundary head, declared parcels (upper bound)")]


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
            vals = [sh[s]["by_kind"][k] for s, *_ in FOUND if s in sh]
            ax.hlines(i, min(vals), max(vals), color=GRID, lw=3, zorder=1)
            for s, m, filled, lab in FOUND:
                if s not in sh:
                    continue
                v = sh[s]["by_kind"][k]
                if filled is None:
                    ax.scatter([v], [i], s=160, marker="|", color=MUTED, lw=2, zorder=3,
                               label=lab if i == 0 and k == 0 else None)
                else:
                    ax.scatter([v], [i], s=70, marker=m, color=slot if filled else SURFACE, edgecolor=slot, lw=1.6,
                               zorder=4, label=lab if i == 0 and k == 0 else None)
        ax.set_title(f"{f0['kinds'][k]}: {tot[k]:,} parcels, {tot[k] / sum(tot):.0%}", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.grid(axis="x", color=GRID)
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("share found, class-agnostic IoU > 0.5")
    axes[0].set_yticks(range(len(arms)), [ARMS[a][0] for a in arms])
    axes[0].set_ylim(-0.6, len(arms) - 0.4)
    axes[0].invert_yaxis()
    fig.legend(loc="upper center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, 1.0))
    fig.text(0.01, 0.01, f"K = {PCT} %. True parcel pieces of the test chips by what they touch inside the declared "
             "extent. Marks in the encoder's colour; grey bar: the boundary head given the declared parcels, which "
             "uses the test labels.", color=MUTED, fontsize=8.5, va="bottom")
    fig.tight_layout(rect=(0, 0.07, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_size(analyses: dict, path: Path) -> None:
    if not analyses:
        return
    f0 = next(iter(analyses.values()))["found"]
    labels = [f"{s}\n{n:,}" for s, n in zip(f0["sizes"], f0["true_parcels"]["by_size"])]
    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    x = np.arange(len(labels))
    for arm, a in analyses.items():
        sh = a["found"]["share"]
        slot = SLOTS[ARMS[arm][1]]
        for s, m, filled, lab in FOUND:
            if s not in sh or s == "boundary/worldcover":
                continue
            if filled is None:
                ax.plot(x, sh[s]["by_size"], color=MUTED, ls=":", lw=1.6, marker="|", ms=12,
                        label=f"{lab}, {ARMS[arm][0]}")
            else:
                ax.plot(x, sh[s]["by_size"], color=slot, ls="-" if filled else "--", lw=2 if filled else 1.5,
                        marker=m, mfc=slot if filled else SURFACE, mec=slot, label=f"{lab}, {ARMS[arm][0]}")
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_xlabel("area of the true parcel piece, and how many there are in the test chips")
    ax.set_ylabel("share found, class-agnostic IoU > 0.5")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=8, loc="upper left")
    ax.set_title(f"What PaPs misses: the parcels found by area, K = {PCT} %")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_validation(cells: dict, path: Path) -> None:
    """Training loss per epoch (log scale) and validation PQ, class-aware and class-agnostic, with the epoch kept."""
    if not cells:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for arm, r in cells.items():
        label, slot, _ = ARMS[arm]
        h = pd.DataFrame(r["history"])
        axes[0].plot(h["epoch"], h["train/loss"], color=SLOTS[slot], lw=1.5, label=f"PaPs, {label}")
        v = h.dropna(subset=["val/PQ"]) if "val/PQ" in h else h.iloc[:0]
        axes[1].plot(v["epoch"], v["val/PQ"], color=SLOTS[slot], marker="o", ms=4, lw=1.5,
                     label=f"{label}, class-aware")
        axes[1].plot(v["epoch"], v["val/PQ_agnostic"], color=SLOTS[slot], marker="o", ms=4, lw=1.5, ls="--",
                     mfc=SURFACE, label=f"{label}, class-agnostic")
        axes[2].plot(v["epoch"], v["val/predicted_parcels"], color=SLOTS[slot], marker="o", ms=4, lw=1.5,
                     label=label)
        for ax in axes[1:]:
            ax.axvline(r["best_epoch"], color=SLOTS[slot], ls=":", lw=1)
    axes[0].set_title("training loss (the first 5 epochs: heatmap only)")
    axes[0].set_yscale("log")
    axes[1].set_title("validation PQ (void rule, whole chip)")
    axes[2].set_title("validation parcels predicted")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].legend(fontsize=8)
    axes[1].legend(fontsize=7)
    fig.suptitle(f"PaPs training, K = {PCT} %; dotted: the epoch kept (best validation PQ)", x=0.01, ha="left",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_fp(df: pd.DataFrame, path: Path) -> None:
    d = df[(df.system == "paps") & df.extent.isin(list(EXTENTS))]
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
    names = {"chip": "whole chip", "worldcover": "cut to WorldCover", "oracle": "cut to declared (upper bound)"}
    ax.set_yticks(range(len(rows)), [f"{names[e]}, {ARMS[a][0]}" for a, e in rows], fontsize=8.5)
    ax.invert_yaxis()
    ax.grid(axis="x", color=GRID)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel(f"false parcels of PaPs, K = {PCT} %")
    ax.legend(ncol=5, fontsize=8, loc="upper center", bbox_to_anchor=(0.45, -0.18 - 0.02 * len(rows)))
    fig.suptitle("Where PaPs' false parcels lie: on the true parcels, or on background by WorldCover class",
                 x=0.01, ha="left", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


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
        for s in ("paps", "paps_cropmap", "boundary", "semantic", "floor"):
            q = df[(df.arm == arm) & (df.system == s) & (df.extent == ext)]
            if q.empty:
                continue
            q = q.iloc[0]
            out.append({"Encoder": ARMS[arm][0], "Parcels from": SYSTEMS[s], "PQ": q.pq, "SQ": q.sq, "RQ": q.rq,
                        "PQ agn.": q.pq_agnostic, "TP": int(q.tp_agnostic), "FP": int(q.fp_agnostic),
                        "FN": int(q.fn_agnostic), "parcels": int(q.predicted_parcels)})
    return pd.DataFrame(out)


def tables(df: pd.DataFrame, cells: dict, analyses: dict) -> str:
    out = []
    for ext, title in (("chip", "Whole chip, no extent"), ("worldcover", "Cut to WorldCover 30 + 40"),
                       ("worldcover_30_40_90", "Cut to WorldCover 30 + 40 + 90 (sensitivity)"),
                       ("oracle", "Cut to the declared parcels: upper bound, uses the test labels")):
        t = _system_table(df, ext)
        if not t.empty:
            out.append(f"**{title}**\n\n" + _md(t))
    if analyses:
        t = pd.DataFrame([{"Encoder": ARMS[a][0], "parcels PaPs found": an["class_on_matched"]["parcels"],
                           "class head accuracy": an["class_on_matched"]["paps_class_head"]["accuracy"],
                           "class head macro F1": an["class_on_matched"]["paps_class_head"]["macro_f1"],
                           "crop-map vote accuracy": an["class_on_matched"]["cropmap_vote"]["accuracy"],
                           "crop-map vote macro F1": an["class_on_matched"]["cropmap_vote"]["macro_f1"],
                           "crop map pixel Macro-F1": an["sources"]["cropmap_pixel_macro_f1"]}
                          for a, an in analyses.items()])
        out.append("**The class of the parcels PaPs found**\n\n" + _md(t))
        rows = []
        for a, an in analyses.items():
            f = an["found"]
            for s, *_, lab in FOUND:
                if s in f["share"]:
                    rows.append({"Encoder": ARMS[a][0], "Parcels from": lab, "all": f["share"][s]["all"],
                                 **dict(zip(f["kinds"], f["share"][s]["by_kind"])),
                                 **dict(zip(f["sizes"], f["share"][s]["by_size"]))})
        f = next(iter(analyses.values()))["found"]
        counts = ", ".join(f"{k} {n:,}" for k, n in zip(f["kinds"] + f["sizes"],
                                                        f["true_parcels"]["by_kind"] + f["true_parcels"]["by_size"]))
        out.append(f"**Share of the true parcels found** (true parcels: {counts})\n\n" + _md(pd.DataFrame(rows)))
        t = pd.DataFrame([{"Encoder": ARMS[a][0], "true parcels, crop-map class": an["variants"]["truth/cropmap"]["rq"],
                           "PaPs, crop-map class": an["variants"]["chip/paps_cropmap"]["rq"],
                           "PaPs, its own class": an["variants"]["chip/paps"]["rq"],
                           "boundary head": _get(df, a, "boundary", "chip", "rq")} for a, an in analyses.items()])
        out.append("**Class-aware RQ (parcel Macro-F1 at IoU > 0.5), whole chip**\n\n" + _md(t))
    if cells:
        t = pd.DataFrame([{"Encoder": ARMS[a][0], "train chips": r["chips"]["train"], "best epoch": r["best_epoch"],
                           "val PQ": r["best_val_pq"], "confidence": r["min_confidence"],
                           "fit h": r["fit_seconds"] / 3600, "peak GB": r["peak_gpu_gb"],
                           "commit": r.get("provenance", {}).get("git_commit", "")[:8]} for a, r in cells.items()])
        out.append("**Fits**\n\n" + _md(t, ".2f"))
    return "\n\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    ap.add_argument("--no-examples", action="store_true")
    a = ap.parse_args()
    df, cells, analyses = load(a.chips)
    if df.empty or (df.system == "paps").sum() == 0:
        raise SystemExit("no scored PaPs cells yet")
    folder = HERE / "figures" if a.chips == "EE_2021_rev" else HERE / "figures" / a.chips
    folder.mkdir(parents=True, exist_ok=True)
    fig_pq(df, folder / "pq_bars.png")
    fig_class_heads(analyses, folder / "class_heads.png")
    fig_geometry_cost(df, folder / "geometry_cost.png")
    fig_where(analyses, folder / "where_found.png")
    fig_size(analyses, folder / "found_by_size.png")
    fig_fp(df, folder / "fp_where.png")
    fig_validation(cells, folder / "validation_curves.png")
    if not a.no_examples:
        fig_examples(a.chips, folder)
    t = tables(df, cells, analyses)
    (folder / "tables.md").write_text(t)
    print(t)


if __name__ == "__main__":
    main()
