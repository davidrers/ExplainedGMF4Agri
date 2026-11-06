"""The architecture of the boundary head on TESSERA as one diagram, drawn from the real data of one test chip.

The boundary head of ``../2026-10-18_parcel_delineation`` (one edge logit beside the crop-type classifier of the
per-pixel MLP), whose parcels this experiment builds without the declared parcels. Every image is real: the chip's
Sentinel-2 and Sentinel-1 composites, its frozen TESSERA v2 embedding, each head's training target and its stored
test output (the frozen cell's ``predictions_test.npz``), the post-processing steps of ``panoptic_x.from_boundary``
and the final parcels in each area, scored as ``score.py`` scores them.

The embedding is read from the pilot cache (``data/embeddings/tessera_v2_medium/EE_2021_rev_mini``, the same
encoder and chips), normalised with the cell's own training-chip statistics, so the chip must be one of the nine
pilot test chips; the heads' outputs are the stored ones, not recomputed.

    poetry run python experiments/2026-11-05_parcel_extent/architecture.py [--chip EE_02265_01779]

Writes ``figures/architecture.png`` and its three bands apart (``architecture_1.png`` to ``_3.png``).
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_rgb  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

CHIPS = "EE_2021_rev"
CELL = REPO / "results/2026-10-18_parcel_delineation" / CHIPS / "tessera_v2_medium/P100_draw0_seed0"
PILOT_CACHE = REPO / "data/embeddings/tessera_v2_medium/EE_2021_rev_mini"
DPI = 110

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#a3a29b"
SURFACE, INK, INK_2, MUTED, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BAND = "#f3f2ee"
UNKNOWN = "#d4d3cc"
NAVY, LIGHT = "#0d366b", "#e3eefb"
ACCENT = "#eb6834"
PROBABILITY = LinearSegmentedColormap.from_list("probability", ["#ffffff", "#2a78d6", "#0d366b"])
SHORT = {"pasture_meadow_grassland_grass": "grassland", "winter_common_soft_wheat": "winter wheat",
         "spring_common_soft_wheat": "spring wheat", "legumes_harvested_green": "legumes, green harvest",
         "winter_rapeseed_rape": "winter rapeseed", "spring_rapeseed_rape": "spring rapeseed",
         "grain_maize_corn_popcorn": "grain maize", "legumes_dried_pulses_protein_crops": "dried pulses",
         "alfalfa_lucerne": "alfalfa", "orchards_fruits": "orchards", "fresh_vegetables": "vegetables"}
plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


# ------------------------------------------------------------------------------------------------------- canvas

class Canvas:
    """A figure laid out in inches from the top left: bands, boxes, arrows and images."""

    def __init__(self, w: float, h: float):
        self.W, self.H = w, h
        self.fig = plt.figure(figsize=(w, h), dpi=DPI)
        self.ov = self.fig.add_axes([0, 0, 1, 1])
        self.ov.set_xlim(0, w)
        self.ov.set_ylim(h, 0)
        self.ov.axis("off")

    def band(self, y, h, number, title):
        self.ov.add_patch(Rectangle((0.25, y), self.W - 0.5, h, facecolor=BAND, edgecolor="none", zorder=0))
        self.ov.text(0.5, y + 0.3, f"{number}", fontsize=17, weight="bold", color=NAVY, va="center")
        self.ov.text(0.9, y + 0.3, title, fontsize=13.5, weight="bold", color=INK, va="center")

    def text(self, x, y, s, width=None, **kw):
        if width:
            s = "\n".join(textwrap.fill(p, width) for p in s.split("\n"))
        kw.setdefault("va", "top")
        kw.setdefault("color", INK_2)
        kw.setdefault("fontsize", 9)
        self.ov.text(x, y, s, **kw)

    def box(self, x, y, w, h, title, body, badge=None, edge=BASELINE, face="#ffffff", width=40):
        self.ov.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.12",
                                         facecolor=face, edgecolor=edge, lw=1.6, zorder=1))
        self.ov.text(x + w / 2, y + 0.2, title, ha="center", va="top", fontsize=11, weight="bold", color=INK)
        body = "\n".join(textwrap.fill(p, width) for p in body.split("\n"))
        self.ov.text(x + w / 2, y + 0.55, body, ha="center", va="top", fontsize=9, color=INK_2, linespacing=1.35)
        if badge:
            text, colour = badge
            self.ov.text(x + w - 0.12, y - 0.04, f" {text} ", ha="right", va="center", fontsize=8, weight="bold",
                         color="white", bbox=dict(boxstyle="round,pad=0.25", facecolor=colour, edgecolor="none"),
                         zorder=3)

    def arrow(self, x1, y1, x2, y2, style="-|>", dashed=False, colour=INK_2, lw=1.6):
        self.ov.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style, mutation_scale=16, color=colour,
                                          lw=lw, linestyle="--" if dashed else "-", zorder=2,
                                          shrinkA=0, shrinkB=0))

    def line(self, xs, ys, colour=INK_2, lw=1.6):
        self.ov.plot(xs, ys, color=colour, lw=lw, solid_capstyle="butt", zorder=2)

    def image(self, x, y, w, h, arr=None, title=None, caption=None, cap_width=None, **imshow):
        ax = self.fig.add_axes([x / self.W, 1 - (y + h) / self.H, w / self.W, h / self.H])
        if arr is not None:
            imshow.setdefault("interpolation", "nearest")
            ax.imshow(arr, **imshow)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(BASELINE)
        if title:
            self.ov.text(x, y - 0.07, title, va="bottom", fontsize=10, weight="bold", color=INK, linespacing=1.25)
        if caption:
            self.text(x, y + h + 0.07, caption, width=cap_width or int(w * 13.5), fontsize=8.8, linespacing=1.3)
        return ax

    def save(self, path: Path, bands: list[tuple[float, float]]):
        self.fig.savefig(path, dpi=DPI)
        plt.close(self.fig)
        from PIL import Image

        img = Image.open(path)
        for i, (y0, y1) in enumerate(bands, start=1):
            img.crop((0, int(y0 * DPI), img.width, int(min(y1, self.H) * DPI))).save(
                path.with_name(f"{path.stem}_{i}.png"))


# --------------------------------------------------------------------------------------------------- the data

def pca_rgb(x: np.ndarray) -> np.ndarray:
    """``(C, H, W)`` features as an RGB image of their first three principal components, each stretched 2 to 98 %."""
    c, h, w = x.shape
    f = x.reshape(c, -1).T.astype(np.float64)
    f -= f.mean(0)
    _, _, vt = np.linalg.svd(f[:: max(1, len(f) // 20000)], full_matrices=False)
    p = f @ vt[:3].T
    lo, hi = np.percentile(p, 2, axis=0), np.percentile(p, 98, axis=0)
    return np.clip((p - lo) / (hi - lo), 0, 1).reshape(h, w, 3)


def outline(seg: np.ndarray) -> np.ndarray:
    e = np.zeros(seg.shape, bool)
    v = seg[:-1] != seg[1:]
    e[:-1] |= v & (seg[:-1] > 0)
    e[1:] |= v & (seg[1:] > 0)
    h = seg[:, :-1] != seg[:, 1:]
    e[:, :-1] |= h & (seg[:, :-1] > 0)
    e[:, 1:] |= h & (seg[:, 1:] > 0)
    return e


def class_image(cls_map, where, colours, edges=None, faint=None):
    img = np.ones(cls_map.shape + (3,))
    if faint is not None:
        img[faint] = to_rgb("#ecebe6")
    for k in np.unique(cls_map[where]):
        img[where & (cls_map == k)] = to_rgb(colours.get(int(k), OTHER))
    if edges is not None:
        img[edges] = to_rgb(INK)
    return img


def chip_inputs(cid: str):
    import rasterio

    chips_dir = REPO / "data/eurocrops_chips" / CHIPS / "chips"
    bands = json.loads((REPO / "data/eurocrops_chips" / CHIPS / "manifest.json").read_text())["imagery"]["band_names"]
    with rasterio.open(chips_dir / f"{cid}_merged.tif") as s:
        months = {m: np.stack([s.read(m * len(bands) + bands.index(b) + 1) for b in ("RED", "GREEN", "BLUE")], -1)
                  for m in (3, 5, 7)}
    scale = np.nanpercentile(np.stack(list(months.values())), 98)
    months = {m: np.clip(np.nan_to_num(v.astype(np.float32)) / scale, 0, 1) for m, v in months.items()}
    with rasterio.open(chips_dir / f"{cid}_s1rtc.tif") as s:
        vv = s.read(6 * 2 + 1).astype(np.float32)
    lo, hi = np.nanpercentile(vv, [2, 98])
    vv = np.clip((np.nan_to_num(vv, nan=lo) - lo) / (hi - lo), 0, 1)
    return months, vv


def normalisation() -> tuple[np.ndarray, np.ndarray]:
    r = json.loads((CELL / "results.json").read_text())

    def find(d):
        if isinstance(d, dict):
            if isinstance(d.get("normalisation"), dict) and "means" in d["normalisation"]:
                return d["normalisation"]
            for v in d.values():
                if (f := find(v)) is not None:
                    return f
        return None

    n = find(r)
    return np.asarray(n["means"], np.float32)[:, None, None], np.asarray(n["stds"], np.float32)[:, None, None]


def crop_colours(mask: np.ndarray, E: np.ndarray, n: int) -> dict:
    """One colour per crop: the chip's declared crops by area, then the others by index; the rest grey. The same
    rule in both experiments' diagrams, so a chip's crops keep their colours across them."""
    area = np.bincount(mask[E], minlength=n)
    order = [int(k) for k in np.argsort(-area, kind="stable") if area[k] > 0]
    order += [k for k in range(n) if k not in order]
    return {k: SLOTS[j] for j, k in enumerate(order[:len(SLOTS)])}


def load(cid: str) -> dict:
    import rasterio

    import panoptic_x as px
    from delineation import boundary_map
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    res = json.loads((CELL / "results.json").read_text())
    names = res["class_names"]
    n = len(names)
    chips_dir = REPO / "data/eurocrops_chips" / CHIPS / "chips"
    with np.load(CELL / "predictions_test.npz") as d:
        i = [str(c) for c in d["ids"]].index(cid)
        pred, mask = d["pred"][i].astype(np.int64), d["mask"][i].astype(np.int64)
        prob = d["boundary"][i].astype(np.float32) / 255.0
    if mask.max() >= 1000:
        raise ValueError("the stored mask carries the boundary encoding")
    parcels = read_parcels(chips_dir, cid)
    wc = np.load(REPO / "data/worldcover" / CHIPS / "worldcover.npz")[cid]
    with rasterio.open(PILOT_CACHE / "chips" / f"{cid}_tessera_v2_medium.tif") as s:
        raw = np.nan_to_num(s.read().astype(np.float32))
    means, stds = normalisation()
    E, true, true_cls, void = px.truth(mask, parcels)
    target = (mask >= 0) & boundary_map(parcels)
    hit = E & (prob >= px.TAU)
    tp, fp, fn = (hit & target).sum(), (hit & ~target).sum(), (E & target & ~hit).sum()
    finals = {}
    for k in ("chip", "worldcover", "oracle"):
        X = px.extent_of(k, mask, parcels, wc)
        seg = px.from_boundary(X, prob, px.TAU)
        cls = px.vote(seg, pred, n)
        aware, agn, _ = px.chip_counts(true, true_cls, seg, cls, void, n)
        finals[k] = dict(area=X, seg=seg, cls=cls, n=int(len(np.unique(seg[seg > 0]))), found=int(agn[0]),
                         right=int(aware[:, 0].sum()), pq=px.quality(aware, agn, names)["pq"])
    seeds = px.pieces((prob < px.TAU).astype(np.int32))
    return dict(cid=cid, names=names, mask=mask, parcels=parcels, pred=pred, prob=prob, E=E, true=true,
                true_cls=true_cls, target=target, emb=pca_rgb((raw - means) / stds), inputs=chip_inputs(cid),
                bf1=2 * tp / max(2 * tp + fp + fn, 1), acc=float((pred[E] == mask[E]).mean()), seeds=seeds,
                finals=finals, epochs=res["config"]["trainer"]["max_epochs"])


def pastel(seg: np.ndarray, seed: int = 0) -> np.ndarray:
    pal = np.random.default_rng(seed).uniform(0.4, 0.92, size=(int(seg.max()) + 1, 3))
    img = np.ones(seg.shape + (3,))
    img[seg > 0] = pal[seg[seg > 0]]
    img[outline(seg)] = to_rgb(INK)
    return img


def draw(D: dict, path: Path) -> None:
    names, E, true, mask = D["names"], D["E"], D["true"], D["mask"]
    colours = crop_colours(mask, E, len(names))
    unknown = np.array(to_rgb(UNKNOWN))
    labelled = mask >= 0

    C = Canvas(21.0, 24.55)
    S = 2.55
    # ---------------------------------------------------------------- band 1: input to embedding to the MLP
    C.band(0.25, 4.15, "1", "From satellite images to the TESSERA embedding, then the per-pixel MLP")
    months, vv = D["inputs"]
    small, gap = 1.3, 0.12
    for k, (m, lab) in enumerate([(3, "April"), (5, "June"), (7, "August")]):
        r, c = divmod(k, 2)
        C.image(0.6 + c * (small + gap), 1.05 + r * (small + 0.3), small, small, months[m], title=lab)
    C.image(0.6 + small + gap, 1.05 + small + 0.3, small, small, vv, title="July, S1 VV", cmap="gray")
    C.text(0.6, 3.99, "Input: 12 months of Sentinel-2 (10 bands) and Sentinel-1 (VV, VH); one chip, 224 x 224 "
                      "pixels of 10 m", width=50, fontsize=8.6)
    C.arrow(3.55, 2.45, 4.0, 2.45)
    C.box(4.0, 1.25, 3.55, 2.4, "TESSERA v2 encoder",
          "reads the 12-month series of each pixel on its own and turns it into 128 numbers\n"
          "21 M parameters, pre-trained by its authors; not trained here", badge=("FROZEN", MUTED), width=40)
    C.arrow(7.55, 2.45, 8.0, 2.45)
    C.image(8.0, 1.05, S, S, D["emb"], title="TESSERA embedding",
            caption="128 numbers per pixel, shown as colour (their 3 main components): similar colour, similar "
                    "pixel", cap_width=36)
    C.arrow(10.55, 2.45, 11.0, 2.45)
    C.box(11.0, 1.25, 3.9, 2.4, "Per-pixel MLP",
          "takes each pixel's 128 numbers on their own: it never sees the neighbours\n"
          "128 \u2192 512 \u2192 256 numbers; 0.2 M parameters, trained here with the heads", badge=("TRAINED", ACCENT),
          width=42)
    C.text(15.5, 1.1, "How to read", fontsize=10, weight="bold", color=INK, va="top")
    C.text(15.5, 1.45, f"Every image is real: test chip {D['cid']}, never seen in training.\n"
                       "Boxes are steps of the model.\nFROZEN: weights fixed. TRAINED: learned from the "
                       "4,898 training chips.\nFine-tuned arm: same diagram, with TESSERA trained together with "
                       "the MLP and both heads (joint fine-tuning).", width=44, fontsize=8.8, linespacing=1.35)

    # --------------------------------------------------------------------------------------------- band 2: heads
    C.band(4.5, 9.95, "2", "Two heads on the MLP features: what each is trained to match, and what it outputs")
    colw = 4.0
    cols = [0.55, 4.6, 8.65]
    C.line([12.95, 12.95], [3.65, 5.05])
    C.line([cols[0] + colw / 2, 12.95], [5.05, 5.05])
    for x in cols[:2]:
        C.arrow(x + colw / 2, 5.05, x + colw / 2, 5.35)
    C.box(cols[0] + 0.1, 5.35, colw - 0.2, 1.15, "1  Crop-type head", "Which crop is this pixel?\n"
          "reads one pixel at a time", edge=NAVY, face=LIGHT, width=44)
    C.box(cols[1] + 0.1, 5.35, colw - 0.2, 1.15, "2  Boundary head", "Is this pixel on a parcel's edge?\n"
          "reads one pixel at a time", edge=NAVY, face=LIGHT, width=44)
    C.box(cols[2] + 0.1, 5.35, colw - 0.2, 1.15, "What training sees", "the pixels where both heads\n"
          "get feedback", width=44)
    ty, oy = 6.95, 10.8
    ix = [x + (colw - S) / 2 for x in cols]
    for x in cols[:2]:
        C.text(x + 0.12, ty - 0.33, "TRAINED TO MATCH", fontsize=8.5, weight="bold", color=NAVY)
        C.text(x + 0.12, oy - 0.33, "OUTPUT ON THIS TEST CHIP", fontsize=8.5, weight="bold", color=ACCENT)
    C.text(cols[2] + 0.12, ty - 0.33, "PIXELS IN THE LOSS", fontsize=8.5, weight="bold", color=NAVY)
    # 1. crop type
    img = class_image(mask, labelled, colours)
    img[~labelled] = unknown
    C.image(ix[0], ty, S, S, img, caption="The declared crop of every pixel inside a declared parcel. Grey: no "
                                          "label, no loss", cap_width=40)
    C.image(ix[0], oy, S, S, class_image(D["pred"], np.ones_like(labelled), colours),
            caption=f"A crop for every pixel, forest and water included; {D['acc']:.0%} right inside the "
                    "declared parcels", cap_width=40)
    # 2. boundary
    img = np.tile(unknown, mask.shape + (1,))
    img[labelled] = to_rgb("#cfe0f5")
    img[D["target"]] = to_rgb(NAVY)
    C.image(ix[1], ty, S, S, img, caption="1 (dark) on the outer ring of every declared parcel, next to another "
                                          "parcel or to no parcel; 0 (light) inside. Grey: no loss",
            cap_width=40)
    C.image(ix[1], oy, S, S, D["prob"], cmap=PROBABILITY, vmin=0, vmax=1,
            caption=f"Edge probability; pixel F1 {D['bf1']:.2f} inside the declared parcels at 0.5",
            cap_width=40)
    # 3. what training sees
    img = np.tile(unknown, mask.shape + (1,))
    img[labelled] = (1, 1, 1)
    C.image(ix[2], ty, S, S, img, caption="White: the only pixels in the loss, the declared parcels of the 20 "
                                          "crops. Grey: forest, water, villages and undeclared grassland, "
                                          "never seen in training", cap_width=40)
    C.text(cols[2] + 0.2, oy - 0.05, "So nothing tells the heads what 'not a field' looks like. Outside the "
                                     "declared parcels they still answer (the outputs on the left cover the whole "
                                     "chip), from what they learned on fields.", width=42, fontsize=9.5,
           color=INK, linespacing=1.4)
    tx = 13.1
    C.text(tx, 5.4, "How the heads are trained", fontsize=10.5, weight="bold", color=INK)
    C.text(tx, 5.8, "Both heads read the same MLP features.\n"
                    "Loss = crop-type cross-entropy + boundary binary cross-entropy (weight 1), averaged over the "
                    "labelled pixels only.\n"
                    f"{D['epochs']} epochs on the 4,898 training chips; the epoch kept is the one with the lowest "
                    "validation loss.\n"
                    "Frozen arm: fitted in the delineation experiment of 18 October 2026. Fine-tuned arm: fitted "
                    "here, TESSERA trained with the heads.\n"
                    "The boundary probability is the only geometry the model gives: no parcel exists until the "
                    "post-processing below cuts the chip along it.", width=58, fontsize=9.2, linespacing=1.45)

    # ------------------------------------------------------------------------------------- band 3: to parcels
    C.band(14.65, 9.65, "3", "From the head outputs to parcels (post-processing), and the final output")
    F = D["finals"]
    xs = [0.6, 4.02, 7.44]
    ya = 16.2
    C.text(0.6, ya - 0.62, "a.  No area given: the whole chip (the realistic case)", fontsize=10.5,
           weight="bold", color=NAVY, va="bottom")
    C.image(xs[0], ya, S, S, pastel(D["seeds"]), title="Step 1: find the seeds",
            caption=f"Pixels with edge probability below 0.5 count as inside a parcel; each connected patch of "
                    f"them is a seed: {int(D['seeds'].max())} seeds", cap_width=36)
    C.arrow(xs[0] + S + 0.1, ya + S / 2, xs[1] - 0.1, ya + S / 2)
    f = F["chip"]
    C.image(xs[1], ya, S, S, pastel(f["seg"], 1), title="Step 2: grow them (watershed)",
            caption="Each seed grows over the edge probability, lowest first, until every pixel of the area "
                    f"belongs to one: {f['n']} parcels", cap_width=36)
    C.arrow(xs[1] + S + 0.1, ya + S / 2, xs[2] - 0.1, ya + S / 2)
    C.image(xs[2], ya, S, S, class_image(f["cls"][f["seg"]], f["seg"] > 0, colours, outline(f["seg"])),
            title="Step 3: give each a crop  =  FINAL",
            caption=f"The crop most of its pixels get from head 1. {f['n']} parcels; {f['found']} match a "
                    f"declared parcel (IoU > 0.5), {f['right']} with the right crop; chip PQ {f['pq']:.2f}",
            cap_width=36)
    C.line([10.65, 10.65], [ya - 0.4, ya + S + 0.9], colour=BASELINE, lw=1.2)
    n_true = int(len(np.unique(true[true > 0])))
    C.image(11.3, ya, S, S, class_image(D["true_cls"][true], true > 0, colours, outline(true)),
            title="Compare: declared parcels",
            caption=f"{n_true} parcels (pieces inside the chip); the score counts a match at IoU > 0.5",
            cap_width=36)
    C.text(14.4, ya + 0.1, "Why the whole chip fails", fontsize=10.5, weight="bold", color=INK)
    C.text(14.4, ya + 0.5, "Step 2 puts every pixel of the area into some parcel. Over the whole chip that "
                           "includes forest, water and villages: wherever the edge probability is low, a seed "
                           "forms and grows, so they become large false parcels.\n"
                           "Only an area given from outside (row b) keeps the parcels on the fields.",
           width=56, fontsize=9.2, linespacing=1.45)
    yb = ya + S + 1.75
    C.text(0.6, yb - 0.62, "b.  An area given from outside: the same three steps, run only inside it",
           fontsize=10.5, weight="bold", color=NAVY, va="bottom")
    for x0, key, title, cap in [
            (0.6, "worldcover", "Area: WorldCover crop + grassland",
             "ESA WorldCover cropland and grassland, a public map: no test label used"),
            (11.3, "oracle", "Area: the declared parcels",
             "The declared parcels of the 20 crops: an upper bound only, it uses the test labels")]:
        f = F[key]
        img = np.ones(mask.shape + (3,))
        img[f["area"]] = to_rgb("#cfe0f5")
        C.image(x0, yb, S, S, img, title=title, caption=cap, cap_width=36)
        x1 = x0 + (6.84 if key == "worldcover" else 4.1)
        C.arrow(x0 + S + 0.1, yb + S / 2, x1 - 0.1, yb + S / 2)
        C.text((x0 + S + x1) / 2, yb + S / 2 - 0.12, "steps 1 to 3", ha="center", va="bottom", fontsize=8.8)
        C.image(x1, yb, S, S, class_image(f["cls"][f["seg"]], f["seg"] > 0, colours, outline(f["seg"])),
                title="FINAL" + (" (upper bound)" if key == "oracle" else ""),
                caption=f"{f['n']} parcels; {f['found']} match a declared parcel, {f['right']} with the right "
                        f"crop; chip PQ {f['pq']:.2f}", cap_width=36)
    # Crop legend.
    present = set(np.unique(mask[E]).tolist()) | set(np.unique(D["pred"]).tolist())
    for f in F.values():
        present |= set(np.unique(f["cls"][f["seg"][f["seg"] > 0]]).tolist())
    lx, ly = 0.6, 24.0
    C.text(lx, ly, "Crop colours:", fontsize=9, weight="bold", color=INK, va="center")
    lx += 1.15
    for k in [k for k in colours if k in present]:
        C.ov.add_patch(Rectangle((lx, ly - 0.12), 0.24, 0.24, facecolor=colours[k], edgecolor="none"))
        lab = SHORT.get(names[k], names[k].replace("_", " "))
        C.text(lx + 0.32, ly, lab, fontsize=9, va="center")
        lx += 0.5 + 0.075 * len(lab)
    if any(k not in colours for k in present):
        C.ov.add_patch(Rectangle((lx, ly - 0.12), 0.24, 0.24, facecolor=OTHER, edgecolor="none"))
        C.text(lx + 0.32, ly, "other crops", fontsize=9, va="center")
    C.save(path, [(0.2, 4.5), (4.48, 14.5), (14.6, 24.55)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip", default="EE_02265_01779")
    a = ap.parse_args()
    out = HERE / "figures" / "architecture.png"
    draw(load(a.chip), out)
    print(out)


if __name__ == "__main__":
    main()
