"""Figures for the architecture notes of PaPs, PRUE and the boundary head: WorldCover in training, and one test chip
from the declared parcels to the output.

For each method, written into that method's experiment folder (``figures/``):

* ``worldcover_training.png`` (PaPs and PRUE only, whose training uses WorldCover), on the chip of the architecture
  diagrams: WorldCover 2021 with the declared parcels outlined; what training uses (declared parcel, background, no
  loss); and the method's own training target on that chip.
* ``example_walkthrough.png``, on one test chip (by default the border report's median chip): row 1, what the model
  is told (Sentinel-2, WorldCover, what training uses, the training target); row 2, what it outputs (two
  intermediate outputs, the final fields, the border check against the declared parcels).

Each method is shown with the arm of its architecture note: PRUE fine-tuned, PaPs and the boundary head frozen.
``prue.py`` and ``targets.py`` are unchanged copies of the experiments' own modules, so the targets drawn are those
the models were trained on.

    poetry run python experiments/2026-11-07_border_accuracy/walkthrough.py --chips EE_2021_rev
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
from scipy import ndimage  # noqa: E402

import borders  # noqa: E402
import panoptic_x as px  # noqa: E402
import prue  # noqa: E402
import report as rp  # noqa: E402
import targets as paps_targets  # noqa: E402

EXAMPLE_CHIP = "EE_02302_01815"
ARCH_CHIP = "EE_02265_01779"
#: method -> (its experiment folder, the arm shown, the source key in ``borders.SOURCES``)
METHODS = {"PRUE": ("2026-11-06_prue_tessera", "fine-tuned", "PRUE fine-tuned"),
           "PaPs": ("2026-11-05_paps_tessera", "frozen", "PaPs frozen"),
           "boundary head": ("2026-11-05_parcel_extent", "frozen", "boundary head frozen")}
#: What training uses each pixel for.
LABEL, BACKGROUND, NO_LOSS = "#2a78d6", "#8cc98c", "#bdbcb6"
IGNORE_GREY = NO_LOSS


class Chip:
    """Everything known about one test chip, and one method's stored outputs on it."""

    def __init__(self, chips: str, cid: str):
        import rasterio

        from gfm4agri.benchmark.parcel_metrics import read_parcels

        self.cid = cid
        chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
        self.parcels = read_parcels(chips_dir, cid).astype(np.int64)
        with rasterio.open(chips_dir / f"{cid}.mask.tif") as s:
            self.mask = s.read(1).astype(np.int64)
        self.wc = np.load(REPO / "data/worldcover" / chips.replace("_mini", "") / "worldcover.npz")[cid]
        self.E, self.scored, self.true_border = borders.regions(self.mask, self.parcels, self.wc)
        manifest = json.loads((REPO / "data/eurocrops_chips" / chips / "manifest.json").read_text())
        bands = manifest["imagery"]["band_names"]
        with rasterio.open(chips_dir / f"{cid}_merged.tif") as s:
            july = 6 * len(bands)
            rgb = np.stack([s.read(july + bands.index(b) + 1) for b in ("RED", "GREEN", "BLUE")], -1)
        self.rgb = np.nan_to_num(np.clip(rgb.astype(np.float32) / np.nanpercentile(rgb, 98), 0, 1))
        self.n_parcels = int(np.unique(px.truth(self.mask, self.parcels)[1][self.E]).size)
        self.chips = chips

    def outputs(self, source: str, keys: tuple[str, ...]) -> dict:
        rel = borders.SOURCES[source][0].format(set=self.chips)
        with np.load(REPO / "results" / rel / borders.CELL / "predictions_test.npz") as d:
            i = [str(c) for c in d["ids"]].index(self.cid)
            return {k: d[k][i] for k in keys}


def worldcover_img(c: Chip) -> np.ndarray:
    img = np.ones(c.mask.shape + (3,))
    for k in np.unique(c.wc):
        img[c.wc == k] = matplotlib.colors.to_rgb(rp.WORLDCOVER.get(int(k), ("", "#ffffff"))[1])
    img[prue.boundary_map(c.parcels)] = matplotlib.colors.to_rgb(rp.INK)
    return img


def roles_img(c: Chip, uses_worldcover: bool) -> np.ndarray:
    """What training uses each pixel for: label (declared parcel of the 20 classes), background, or no loss."""
    img = np.full(c.mask.shape + (3,), matplotlib.colors.to_rgb(NO_LOSS))
    if uses_worldcover:
        bg = ~np.isin(c.wc, prue.AGRI) & (c.wc != 0) & ~c.E
        img[bg] = matplotlib.colors.to_rgb(BACKGROUND)
    img[c.E] = matplotlib.colors.to_rgb(LABEL)
    img[prue.boundary_map(c.parcels)] = matplotlib.colors.to_rgb(rp.INK)
    return img


def target_panel(ax, c: Chip, method: str) -> None:
    if method == "PRUE":
        y = prue.target(c.mask, c.parcels, c.wc)
        img = np.full(y.shape + (3,), matplotlib.colors.to_rgb(IGNORE_GREY))
        for k, col in rp.CLASS3.items():
            img[y == k] = matplotlib.colors.to_rgb(col)
        rp._show(ax, img, "PRUE's target: background (sand),\ninterior, boundary; grey: no loss")
    elif method == "PaPs":
        t = paps_targets.build(c.mask, c.parcels, c.wc)
        heat, known = t[..., 0], t[..., 7] > 0
        rgba = rp.PROBABILITY(np.clip(heat, 0, 1))[..., :3]
        rgba[~known] = matplotlib.colors.to_rgb(IGNORE_GREY)
        rp._show(ax, rgba, "PaPs' centre target: a peak per\nparcel, 0 elsewhere; grey: no loss")
    else:
        img = np.full(c.mask.shape + (3,), matplotlib.colors.to_rgb(IGNORE_GREY))
        img[c.E] = matplotlib.colors.to_rgb("#a9c8f0")
        img[c.true_border] = matplotlib.colors.to_rgb("#0d366b")
        rp._show(ax, img, "Boundary head's target: border (dark),\ninside (light); grey: no loss")


def role_handles(uses_worldcover: bool) -> list:
    h = [matplotlib.patches.Patch(color=LABEL, label="declared parcel: learnt from")]
    if uses_worldcover:
        h.append(matplotlib.patches.Patch(color=BACKGROUND, label="WorldCover non-farm land: learnt as no field"))
        h.append(matplotlib.patches.Patch(color=NO_LOSS, label="undeclared WorldCover farmland, other crops:\nno loss"))
    else:
        h.append(matplotlib.patches.Patch(color=NO_LOSS, label="everything else, forest and water included:\nno loss"))
    return h


def wc_handles(c: Chip) -> list:
    return [matplotlib.patches.Patch(facecolor=rp.WORLDCOVER[k][1], edgecolor=rp.BASELINE, label=rp.WORLDCOVER[k][0])
            for k in rp.WORLDCOVER if (c.wc == k).any()]


def fig_worldcover(c: Chip, method: str, path: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.9))
    farm = np.isin(c.wc, prue.AGRI)
    share = (farm & (c.parcels > 0)).sum() / max(farm.sum(), 1)
    rp._show(axes[0], worldcover_img(c), f"WorldCover 2021, declared parcels outlined\n"
                                         f"{share:.0%} of its farmland is declared")
    rp._show(axes[1], roles_img(c, True), "What training uses")
    target_panel(axes[2], c, method)
    fig.legend(handles=wc_handles(c), loc="upper left", ncol=3, fontsize=7.5, title="WorldCover",
               title_fontsize=8, alignment="left", bbox_to_anchor=(0.02, 0.16), handlelength=1.2)
    fig.legend(handles=role_handles(True), loc="upper left", ncol=1, fontsize=7.5, title="what training uses",
               title_fontsize=8, alignment="left", bbox_to_anchor=(0.5, 0.16), handlelength=1.2)
    owner = f"{method}'" if method.endswith("s") else f"{method}'s"
    fig.suptitle(f"Test chip {c.cid}: where WorldCover enters {owner} training", x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.16, 1, 0.97))
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def fig_example(c: Chip, method: str, path: Path) -> dict:
    folder, enc, source = METHODS[method]
    fig, axes = plt.subplots(2, 4, figsize=(11, 8.4))
    farm = np.isin(c.wc, prue.AGRI)
    share = (farm & (c.parcels > 0)).sum() / max(farm.sum(), 1)
    uses_wc = method != "boundary head"
    rp._show(axes[0, 0], c.rgb, f"1  Sentinel-2, July composite\n{c.n_parcels} declared parcels")
    rp._show(axes[0, 1], worldcover_img(c), f"2  WorldCover 2021, parcels outlined;\n{share:.0%} of its farmland "
                                            "declared")
    rp._show(axes[0, 2], roles_img(c, uses_wc), "3  What training uses")
    target_panel(axes[0, 3], c, method)
    axes[0, 3].set_title("4  " + axes[0, 3].get_title(loc="left"), fontsize=8.5, loc="left")
    if method == "PRUE":
        o = c.outputs(source, ("prob",))
        prob = o["prob"]
        cm = np.ones(c.mask.shape + (3,))
        for k, col in rp.CLASS3.items():
            cm[prob.argmax(0) == k] = matplotlib.colors.to_rgb(col)
        rp._show(axes[1, 0], cm, "5  Class per pixel: background,\ninterior, boundary")
        rp._show(axes[1, 1], prob[prue.BOUNDARY] / 255.0, "6  Probability of boundary\n(0 white, 1 dark)",
                 cmap=rp.PROBABILITY, vmin=0, vmax=1)
        seg = prue.parcels(prob, "grow")
    elif method == "PaPs":
        o = c.outputs(source, ("inst", "heatmap", "saliency"))
        heat = ndimage.maximum_filter(o["heatmap"].astype(np.float32), size=5)
        rp._show(axes[1, 0], heat / max(heat.max(), 1.0), "5  Centre heatmap, peaks = centres\n(dots enlarged)",
                 cmap=rp.PROBABILITY, vmin=0, vmax=1)
        rp._show(axes[1, 1], o["saliency"] / 255.0, "6  Saliency (part of a parcel?)",
                 cmap=rp.PROBABILITY, vmin=0, vmax=1)
        seg = o["inst"].astype(np.int64)
    else:
        o = c.outputs(source, ("boundary",))
        pb = o["boundary"].astype(np.float32) / 255.0
        rp._show(axes[1, 0], pb, "5  Probability of border\n(0 white, 1 dark)", cmap=rp.PROBABILITY, vmin=0, vmax=1)
        rp._show(axes[1, 1], rp._fields_img(prue.pieces((pb < px.TAU).astype(np.int32))),
                 "6  Cores: border probability\nbelow 0.5")
        seg = px.from_boundary(np.ones(pb.shape, bool), pb, px.TAU)
    n_f = int(np.unique(seg[seg > 0]).size)
    rp._show(axes[1, 2], rp._fields_img(seg), f"7  Final fields: {n_f}")
    img, p, r, f = rp._check(seg, c.true_border, c.E, c.scored)
    rp._show(axes[1, 3], img, f"8  Border check: precision {p:.2f},\nrecall {r:.2f}, F1 {f:.2f}")
    check = [matplotlib.patches.Patch(color=rp.CORRECT, label="drawn, real border within 10 m"),
             matplotlib.patches.Patch(color=rp.FALSE, label="drawn, no real border within 10 m"),
             matplotlib.patches.Patch(color=rp.MISSED, label="real border, nothing drawn within 10 m"),
             matplotlib.patches.Patch(facecolor=rp.PARCEL_BASE, edgecolor=rp.BASELINE, label="declared parcel"),
             matplotlib.patches.Patch(color=rp.NOT_SCORED, label="not scored: undeclared farmland")]
    fig.legend(handles=wc_handles(c), loc="upper left", ncol=2, fontsize=7.5, title="WorldCover (2)",
               title_fontsize=8, alignment="left", bbox_to_anchor=(0.01, 0.135), handlelength=1.2)
    fig.legend(handles=role_handles(uses_wc), loc="upper left", fontsize=7.5, title="what training uses (3)",
               title_fontsize=8, alignment="left", bbox_to_anchor=(0.33, 0.135), handlelength=1.2)
    fig.legend(handles=check, loc="upper left", fontsize=7.5, title="border check (8)", title_fontsize=8,
               alignment="left", bbox_to_anchor=(0.69, 0.135), handlelength=1.2)
    fig.suptitle(f"{method} ({enc} TESSERA) on test chip {c.cid}: from the declared parcels to the output",
                 x=0.01, ha="left", fontsize=11)
    fig.tight_layout(rect=(0, 0.13, 1, 0.97), h_pad=2.2)
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    true = px.truth(c.mask, c.parcels)[1]
    found = len(px.match(true, seg, (c.mask < 0) & (c.parcels > 0))[0])
    return {"fields": n_f, "found": found, "precision": p, "recall": r, "f1": f}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chips", default="EE_2021_rev")
    ap.add_argument("--chip", default=EXAMPLE_CHIP)
    ap.add_argument("--architecture-chip", default=ARCH_CHIP)
    a = ap.parse_args()
    ex, arch = Chip(a.chips, a.chip), Chip(a.chips, a.architecture_chip)
    for method, (folder, _, _) in METHODS.items():
        out = REPO / "experiments" / folder / "figures"
        if method != "boundary head":
            fig_worldcover(arch, method, out / "worldcover_training.png")
        s = fig_example(ex, method, out / "example_walkthrough.png")
        print(f"{method:14s} {ex.cid}: {ex.n_parcels} declared, {s['fields']} fields, {s['found']} found, "
              f"border precision {s['precision']:.2f}, recall {s['recall']:.2f}, F1 {s['f1']:.2f}")
    farm = np.isin(arch.wc, prue.AGRI)
    print(f"{arch.cid}: {(farm & (arch.parcels > 0)).sum() / farm.sum():.0%} of WorldCover farmland declared")


if __name__ == "__main__":
    main()
