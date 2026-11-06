"""The architecture of U-Net + PaPs on TESSERA as one diagram, drawn from the real data of one test chip.

Every image is real: the chip's Sentinel-2 and Sentinel-1 composites, its frozen TESSERA v2 embedding, the
U-Net's features, each head's training target (``targets.py``, built for this chip) and its output, the
post-processing steps and the final parcels, all from the frozen arm's kept checkpoint (``best-pq.pt``).

The model is run again on the hub, on the chip's frozen TESSERA raster from the pilot cache
(``data/embeddings/tessera_v2_medium/EE_2021_rev_mini``, the same encoder and chips), normalised with the full run's
training-chip statistics, which the main workflow's TESSERA cell records (``results/kshot/...``). On the nine pilot
test chips this reproduces the stored test predictions: the same number of parcels and over 99.9 % of the pixels.
So the chip must be one of those nine.

    poetry run python experiments/2026-11-05_paps_tessera/architecture.py [--chip EE_02265_01779]

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
CELL = REPO / "results/2026-11-05_paps_tessera" / CHIPS / "paps_tessera_v2_frozen/P100_draw0_seed0"
CROPMAP = REPO / "results/kshot" / CHIPS / "tessera_v2_medium/P100_draw0_seed0"
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
    r = json.loads((CROPMAP / "results.json").read_text())

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


def run_model(x: np.ndarray, threshold: float) -> dict:
    """The kept checkpoint on one chip: every head's output, the candidates, their masks and the parcels."""
    import torch

    from paps_model import build_model

    model = build_model(x.shape[0], 20)
    model.load_state_dict(torch.load(CELL / "best-pq.pt", map_location="cpu")["model"])
    model.eval()
    model.min_confidence.fill_(threshold)
    caught = {}
    hooks = [model.shape_mlp.register_forward_hook(lambda m, i, o: caught.__setitem__("shape", o.detach())),
             model.encoder.register_forward_hook(lambda m, i, o: caught.__setitem__("features", o[0].detach()))]
    with torch.no_grad():
        t = torch.from_numpy(x)[None]
        p = model(t, pseudo_nms=True, prune=True)
        # The shape stretched to each box, before the saliency is added, as the mask assembly does it.
        stretched = []
        for i, s in enumerate(caught["shape"].view(-1, 1, 16, 16).split(1, 0)):
            h, w = p["size"][i].round().long().clamp_min(1).tolist()
            stretched.append(torch.sigmoid(torch.nn.functional.interpolate(s, size=(h, w), mode="bilinear"))[0, 0])
    for hk in hooks:
        hk.remove()
    centres = torch.stack(torch.where(p["center_mask"][0]), 1).numpy()
    conf = p["confidence"].view(-1).numpy()
    order = np.argsort(-conf, kind="stable")
    inst = p["pano_instance"][0].long().numpy()
    of_id = {n + 1: int(order[n]) for n in range(len(order)) if (inst == n + 1).any()}
    return {"heat": p["heatmap"][0, 0].numpy(), "sal": torch.sigmoid(p["saliency"][0, 0]).numpy(),
            "features": caught["features"][0].numpy(), "centres": centres, "conf": conf,
            "size": p["size"].numpy(), "cls": p["semantic"].argmax(1).numpy(),
            "shape": torch.sigmoid(caught["shape"]).view(-1, 16, 16).numpy(),
            "stretched": [s.numpy() for s in stretched],
            "boxes": p["instance_boxes"].numpy(), "masks": [m[0].numpy() for m in p["instance_masks"]],
            "inst": inst, "of_id": of_id}


def load(cid: str) -> dict:
    import rasterio

    import panoptic_x as px
    import targets
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    res = json.loads((CELL / "results.json").read_text())
    names = res["class_names"]
    chips_dir = REPO / "data/eurocrops_chips" / CHIPS / "chips"
    with np.load(CELL / "predictions_test.npz") as d:
        ids = [str(c) for c in d["ids"]]
        i = ids.index(cid)
        mask, stored = d["mask"][i].astype(np.int64), d["inst"][i]
    with np.load(CROPMAP / "predictions_test.npz") as d:
        crop = d["pred"][[str(c) for c in d["ids"]].index(cid)]
    parcels = read_parcels(chips_dir, cid)
    wc = np.load(REPO / "data/worldcover" / CHIPS / "worldcover.npz")[cid]
    with rasterio.open(PILOT_CACHE / "chips" / f"{cid}_tessera_v2_medium.tif") as s:
        raw = np.nan_to_num(s.read().astype(np.float32))
    means, stds = normalisation()
    x = (raw - means) / stds
    out = run_model(x, float(res["min_confidence"]))
    agree = float(((out["inst"] > 0) == (stored > 0)).mean())
    print(f"{cid}: {len(out['of_id'])} parcels re-run, {len(np.unique(stored[stored > 0]))} stored, "
          f"{agree:.4f} of the pixels agree")
    T = targets.build(mask, parcels, wc)
    E, true, true_cls, void = px.truth(mask, parcels)
    inst = out["inst"]
    own = np.zeros(int(inst.max()) + 1, np.int64)
    for k, c in out["of_id"].items():
        own[k] = out["cls"][c]
    vote = px.vote(inst, crop, len(names))

    def score(cls):
        aware, agn, _ = px.chip_counts(true, true_cls, inst, cls, void, len(names))
        return int(agn[0]), int(aware[:, 0].sum()), px.quality(aware, agn, names)["pq"]

    ti, si, iou, *_ = px.match(true, inst, void)
    return dict(cid=cid, names=names, mask=mask, parcels=parcels, wc=wc, emb=pca_rgb(x),
                feat=pca_rgb(out["features"]), T=T, E=E, true=true, true_cls=true_cls, out=out, own=own,
                vote=vote, s_own=score(own), s_vote=score(vote), matches=list(zip(ti, si, iou)),
                inputs=chip_inputs(cid), threshold=float(res["min_confidence"]), best_epoch=res["best_epoch"])


# ------------------------------------------------------------------------------------------------------ diagram

def pick_examples(D: dict, n: int = 3) -> list[tuple[int, int, int]]:
    """Up to ``n`` found parcels for the shape tiles: ``(true id, parcel id, candidate)``, larger ones first, boxes
    under 110 px."""
    true, out = D["true"], D["out"]
    rows = []
    for t, s, _ in D["matches"]:
        c = out["of_id"].get(int(s))
        if c is None:
            continue
        ys, xs = np.nonzero(true == t)
        if max(np.ptp(ys), np.ptp(xs)) < 110:
            rows.append((len(ys), int(t), int(s), c))
    rows.sort(reverse=True)
    return [(t, s, c) for _, t, s, c in rows[:n]]


def tile_of(true: np.ndarray, t: int) -> np.ndarray:
    from skimage.transform import resize

    ys, xs = np.nonzero(true == t)
    crop = (true[ys.min():ys.max() + 1, xs.min():xs.max() + 1] == t).astype(float)
    return resize(crop, (16, 16), order=1, anti_aliasing=True)


def centre_of(mask: np.ndarray) -> tuple[float, float]:
    ys, xs = np.nonzero(mask)
    return float(ys.mean()), float(xs.mean())


def crop_colours(mask: np.ndarray, E: np.ndarray, n: int) -> dict:
    """One colour per crop: the chip's declared crops by area, then the others by index; the rest grey. The same
    rule in both experiments' diagrams, so a chip's crops keep their colours across them."""
    area = np.bincount(mask[E], minlength=n)
    order = [int(k) for k in np.argsort(-area, kind="stable") if area[k] > 0]
    order += [k for k in range(n) if k not in order]
    return {k: SLOTS[j] for j, k in enumerate(order[:len(SLOTS)])}


def draw(D: dict, path: Path) -> None:
    out, T, true, E = D["out"], D["T"], D["true"], D["E"]
    names = D["names"]
    inst = out["inst"]
    colours = crop_colours(D["mask"], E, len(names))
    present = set(np.unique(D["mask"][E]).tolist()) | set(np.unique(D["own"][inst[inst > 0]]).tolist()) \
        | set(np.unique(D["vote"][inst[inst > 0]]).tolist())
    examples = pick_examples(D)
    letters = "ABC"

    C = Canvas(21.0, 20.25)
    S = 2.55                                             # image side, inches
    # ---------------------------------------------------------------- band 1: input to embedding to features
    C.band(0.25, 4.15, "1", "From satellite images to the TESSERA embedding, then the U-Net")
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
    C.box(11.0, 1.25, 3.55, 2.4, "U-Net",
          "looks at each pixel together with its neighbours, at 4 scales (10, 20, 40 and 80 m)\n"
          "1.2 M parameters, trained here with the heads", badge=("TRAINED", ACCENT), width=40)
    C.arrow(14.55, 2.45, 15.0, 2.45)
    C.image(15.0, 1.05, S, S, D["feat"], title="U-Net features",
            caption="32 numbers per pixel (3 main components): fields now have sharp, uniform colours",
            cap_width=36)
    C.text(17.85, 1.1, "How to read", fontsize=10, weight="bold", color=INK, va="top")
    C.text(17.85, 1.45, f"Every image is real: test chip {D['cid']}, never seen in training.\n"
                        "Boxes are steps of the model.\nFROZEN: weights fixed. TRAINED: learned from the "
                        "4,898 training chips.\nFine-tuned arm: same diagram, with the encoder first fine-tuned "
                        "on crop type (RQ1 experiment), then frozen.", width=34, fontsize=8.8, linespacing=1.35)

    # --------------------------------------------------------------------------------------------- band 2: heads
    C.band(4.5, 9.95, "2", "Five heads on the U-Net features: what each is trained to match, and what it outputs")
    colw, x0 = 4.0, 0.55
    cols = [x0 + k * (colw + 0.05) for k in range(5)]
    # The bus from the features to the five heads.
    C.line([12.775, 12.775], [3.65, 5.05])
    C.line([cols[0] + colw / 2, cols[4] + colw / 2], [5.05, 5.05])
    for x in cols:
        C.arrow(x + colw / 2, 5.05, x + colw / 2, 5.35)
    heads = [("1  Centre heatmap", "Where is the centre of a parcel?\nreads every pixel"),
             ("2  Saliency", "Is this pixel part of a parcel?\nreads every pixel"),
             ("3  Size", "How tall and wide is the parcel?\nreads the features at each centre"),
             ("4  Shape", "What is its outline, on a 16 x 16 grid?\nreads the features at each centre"),
             ("5  Class", "Which crop is it?\nreads the features at each centre")]
    for x, (t, b) in zip(cols, heads):
        C.box(x + 0.1, 5.35, colw - 0.2, 1.15, t, b, edge=NAVY, face=LIGHT, width=44)
    ty, oy = 6.95, 10.8                                  # top of the target and output images
    ix = [x + (colw - S) / 2 for x in cols]
    for x in cols:
        C.text(x + 0.12, ty - 0.33, "TRAINED TO MATCH", fontsize=8.5, weight="bold", color=NAVY)
        C.text(x + 0.12, oy - 0.33, "OUTPUT ON THIS TEST CHIP", fontsize=8.5, weight="bold", color=ACCENT)

    known = T[..., 7] > 0
    unknown_rgb = np.array(to_rgb(UNKNOWN))
    # 1. heatmap
    tgt = PROBABILITY(np.sqrt(T[..., 0]))[..., :3]
    tgt[~known] = unknown_rgb
    C.image(ix[0], ty, S, S, tgt, caption="A peak at the centre of every declared parcel, wider for a bigger "
                                          "parcel; 0 on WorldCover's non-farm land. Grey: undeclared grassland "
                                          "and cropland, no loss", cap_width=40)
    hv = max(1e-3, float(np.percentile(out["heat"], 99.9)))
    C.image(ix[0], oy, S, S, out["heat"], cmap=PROBABILITY, vmin=0, vmax=hv,
            caption=f"Small peaks where it expects a centre (highest {out['heat'].max():.2f}; colour stretched)",
            cap_width=40)
    # 2. saliency: no target of its own; it learns inside each parcel's box through the mask loss.
    sal_t = np.tile(unknown_rgb, true.shape + (1,))
    inst_t = T[..., 1].astype(int)
    for k in range(1, int(inst_t.max()) + 1):
        ys, xs = np.nonzero(inst_t == k)
        if len(ys) == 0:
            continue
        sl = (slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1))
        sub = sal_t[sl]
        sub[inst_t[sl] == 0] = (1, 1, 1)
    sal_t[inst_t > 0] = to_rgb("#2a78d6")
    C.image(ix[1], ty, S, S, sal_t, caption="No target of its own: it learns through head 4's mask loss, inside "
                                            "each parcel's box (blue: the parcel, white: not). Grey: never "
                                            "corrected", cap_width=40)
    C.image(ix[1], oy, S, S, out["sal"], cmap=PROBABILITY, vmin=0, vmax=1,
            caption="Dark: likely parcel. It marks the fields, and also most open land around them",
            cap_width=40)
    # 3. size: the true boxes; the predicted boxes at the parcels kept.
    ax = C.image(ix[2], ty, S, S, class_image(np.zeros_like(true), true > 0, {0: "#cfe0f5"}))
    for k in np.unique(true[true > 0]):
        ys, xs = np.nonzero(true == k)
        ax.add_patch(Rectangle((xs.min() - 0.5, ys.min() - 0.5), np.ptp(xs) + 1, np.ptp(ys) + 1, fill=False,
                               edgecolor=NAVY, lw=0.9))
    for j, (t, s, c) in enumerate(examples):
        cy, cx = centre_of(true == t)
        ax.text(cx, cy, letters[j], ha="center", va="center", fontsize=9, weight="bold", color=INK)
    C.text(ix[2], ty + S + 0.07, "The height and width of each declared parcel's box, learned at its centre",
           width=40, fontsize=8.8)
    ax = C.image(ix[2], oy, S, S, class_image(np.zeros_like(inst), inst > 0, {0: "#cfe0f5"}))
    for k, c in out["of_id"].items():
        x1, y1, x2, y2 = out["boxes"][c]
        ax.add_patch(Rectangle((x1 - 0.5, y1 - 0.5), x2 - x1, y2 - y1, fill=False, edgecolor=ACCENT, lw=0.9))
    for j, (t, s, c) in enumerate(examples):
        cy, cx = out["centres"][c]
        ax.text(cx, cy, letters[j], ha="center", va="center", fontsize=9, weight="bold", color=INK)
    C.text(ix[2], oy + S + 0.07, "A box at each centre kept (orange); light blue: the parcels finally drawn",
           width=40, fontsize=8.8)
    # 4. shape: 16 x 16 tiles of the example parcels, target over output.
    tw = 1.12
    for j, (t, s, c) in enumerate(examples):
        tx = cols[3] + 0.25 + j * (tw + 0.12)
        C.image(tx, ty + 0.45, tw, tw, tile_of(true, t), title=f"parcel {letters[j]}", cmap=PROBABILITY,
                vmin=0, vmax=1)
        C.image(tx, oy + 0.45, tw, tw, out["shape"][c], title=f"parcel {letters[j]}", cmap=PROBABILITY,
                vmin=0, vmax=1)
    C.text(cols[3] + 0.25, ty + 0.45 + tw + 0.12, "The declared parcel inside its box, on 16 x 16 cells "
                                                  "(parcels A, B, C are marked on head 3)", width=40, fontsize=8.8)
    C.text(cols[3] + 0.25, oy + 0.45 + tw + 0.12, "The 16 x 16 outline it predicts for the same parcels: rounded "
                                                  "blobs; the detail comes from the saliency (step 2)",
           width=40, fontsize=8.8)
    # 5. class: a dot per centre in its crop's colour.
    ax = C.image(ix[4], ty, S, S, class_image(np.zeros_like(true), true > 0, {0: "#e6e5df"}))
    cen = np.argwhere(T[..., 0] == 1)
    ax.scatter(cen[:, 1], cen[:, 0], s=22, c=[colours.get(int(T[y, x, 5]), OTHER) for y, x in cen],
               edgecolors=INK, linewidths=0.5)
    C.text(ix[4], ty + S + 0.07, "The crop of each declared parcel, learned at its centre", width=40, fontsize=8.8)
    ax = C.image(ix[4], oy, S, S, class_image(np.zeros_like(inst), inst > 0, {0: "#e6e5df"}))
    kept = [out["of_id"][k] for k in sorted(out["of_id"])]
    pts = out["centres"][kept]
    ax.scatter(pts[:, 1], pts[:, 0], s=22, c=[colours.get(int(out["cls"][c]), OTHER) for c in kept],
               edgecolors=INK, linewidths=0.5)
    C.text(ix[4], oy + S + 0.07, "The crop it gives each centre kept", width=40, fontsize=8.8)
    C.text(0.6, 14.12, "Training: 100 epochs on the 4,898 training chips; the loss adds heads 1, 3, 4 and 5 with "
                      "weight 1 each (head 2 learns through head 4). The epoch kept is the one with the best "
                      f"validation PQ (epoch {D['best_epoch']}).", fontsize=9, color=INK_2)

    # ------------------------------------------------------------------------------------- band 3: to parcels
    C.band(14.65, 5.4, "3", "From the head outputs to parcels (post-processing), and the final output")
    py, step = 15.85, 3.42
    px_ = [0.6 + k * step for k in range(6)]
    # 1. centres
    ax = C.image(px_[0], py, S, S, out["heat"], cmap=PROBABILITY, vmin=0, vmax=hv,
                 title="Step 1: find the centres",
                 caption=f"Local peaks of head 1 above {D['threshold']:.2f} (threshold tuned on the validation "
                         f"chips): {len(out['centres'])} candidates", cap_width=36)
    ax.scatter(out["centres"][:, 1], out["centres"][:, 0], s=10, c=ACCENT, edgecolors="none")
    C.arrow(px_[0] + S + 0.1, py + S / 2, px_[1] - 0.1, py + S / 2)
    # 2. one parcel drawn: the shape stretched to its box, the saliency in the box, the mask kept, the truth.
    if examples:
        t, s, c = examples[0]
        x1, y1, x2, y2 = out["boxes"][c]
        q = 1.05
        st = out["stretched"][c]
        h0, w0 = int(np.clip(-(out["centres"][c][0] - st.shape[0] // 2), 0, None)), \
            int(np.clip(-(out["centres"][c][1] - st.shape[1] // 2), 0, None))
        st = st[h0:h0 + (y2 - y1), w0:w0 + (x2 - x1)]
        parts = [(st, "a. shape in its box"), (out["sal"][y1:y2, x1:x2], "b. saliency there"),
                 (out["masks"][c] > 0.4, "c. a + b, kept > 0.4"), ((true == t)[y1:y2, x1:x2], "declared parcel A")]
        for j, (a, lab) in enumerate(parts):
            r, k = divmod(j, 2)
            C.image(px_[1] + k * (q + 0.15), py + r * (q + 0.3), q, q, a, cmap=PROBABILITY, vmin=0, vmax=1)
            C.text(px_[1] + k * (q + 0.15), py + r * (q + 0.3) + q + 0.05, lab, fontsize=8)
        C.text(px_[1], py - 0.07, "Step 2: draw each parcel (here A)", va="bottom", fontsize=10, weight="bold",
               color=INK)
        C.text(px_[1], py + S + 0.14, "Shape (head 4) resized to the box (head 3) and mixed with the saliency "
                                      "(head 2); the pixels above 0.4 form the parcel", width=36, fontsize=8.8)
    C.arrow(px_[1] + S + 0.1, py + S / 2, px_[2] - 0.1, py + S / 2)
    # 3. overlaps
    rng = np.random.default_rng(0)
    pal = rng.uniform(0.35, 0.9, size=(int(inst.max()) + 1, 3))
    img = np.ones(inst.shape + (3,))
    img[inst > 0] = pal[inst[inst > 0]]
    img[outline(inst)] = to_rgb(INK)
    C.image(px_[2], py, S, S, img, title="Step 3: keep the strongest",
            caption=f"Highest peak first; a parcel is dropped when less than half of it is new. "
                    f"{len(out['centres'])} candidates, {len(out['of_id'])} parcels", cap_width=36)
    C.arrow(px_[2] + S + 0.1, py + S / 2, px_[3] - 0.1, py + S / 2)
    # 4. final, own class
    f_own, w_own, q_own = D["s_own"]
    n_true = int(len(np.unique(true[true > 0])))
    C.image(px_[3], py, S, S, class_image(D["own"][inst], inst > 0, colours, outline(inst)),
            title="Step 4: give each a crop  =  FINAL",
            caption=f"Crop from head 5. {len(out['of_id'])} parcels; {f_own} match a declared parcel "
                    f"(IoU > 0.5), {w_own} with the right crop; chip PQ {q_own:.2f}", cap_width=36)
    f_v, w_v, q_v = D["s_vote"]
    C.text(px_[4] - 0.62, py + S / 2, "or", fontsize=12, weight="bold", color=INK, va="center")
    C.image(px_[4], py, S, S, class_image(D["vote"][inst], inst > 0, colours, outline(inst)),
            title="Variant: crop from the crop map",
            caption=f"Each parcel takes the crop most of its pixels get in the RQ1 crop-type map. "
                    f"{w_v} with the right crop; chip PQ {q_v:.2f}", cap_width=36)
    C.line([px_[5] - 0.45, px_[5] - 0.45], [py - 0.4, py + S + 0.9], colour=BASELINE, lw=1.2)
    C.image(px_[5], py, S, S, class_image(D["true_cls"][true], true > 0, colours, outline(true)),
            title="Compare: declared parcels",
            caption=f"{n_true} parcels (pieces inside the chip); the score counts a match at IoU > 0.5",
            cap_width=36)
    # Crop legend.
    shown = [k for k in colours if k in present]
    lx = 0.6
    C.text(lx, 19.75, "Crop colours:", fontsize=9, weight="bold", color=INK, va="center")
    lx += 1.15
    for k in shown:
        C.ov.add_patch(Rectangle((lx, 19.63), 0.24, 0.24, facecolor=colours[k], edgecolor="none"))
        lab = SHORT.get(names[k], names[k].replace("_", " "))
        C.text(lx + 0.32, 19.75, lab, fontsize=9, va="center")
        lx += 0.5 + 0.075 * len(lab)
    if any(k not in colours for k in present):
        C.ov.add_patch(Rectangle((lx, 19.63), 0.24, 0.24, facecolor=OTHER, edgecolor="none"))
        C.text(lx + 0.32, 19.75, "other crops", fontsize=9, va="center")
    C.save(path, [(0.2, 4.5), (4.48, 14.5), (14.6, 20.25)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip", default="EE_02265_01779")
    a = ap.parse_args()
    out = HERE / "figures" / "architecture.png"
    draw(load(a.chip), out)
    print(out)


if __name__ == "__main__":
    main()
