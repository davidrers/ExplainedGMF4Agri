"""The architecture of PRUE on TESSERA as one diagram, drawn from the real data of one test chip.

PRUE (CVPR 2026, ftw-prue; ``prue.py``): a U-Net with an EfficientNet-B7 encoder labels every pixel background, field
interior or field boundary, and its fields are the connected patches of interior. Here it reads TESSERA v2 Medium's
embedding and the encoder is trained with it (the fine-tuned arm). Every image is real: the chip's Sentinel-2 and
Sentinel-1 composites, the fine-tuned student's embedding, the U-Net's decoder features, the three-class training
target (``prue.target``) and the class probabilities, the fields of ``prue.parcels`` and their crop from the
fine-tuned crop-type map, scored as ``score_prue.py`` scores them.

The model is run on the hub, on the CPU, from the fine-tuned arm's ``best.pt`` (student, standardisation and U-Net
together) with ``embedding_stats.json`` beside it: end to end from the composites, so any test chip works. The student
runs in float32 here, in bf16 on the cluster.

    poetry run python experiments/2026-11-06_prue_tessera/architecture.py [--chip EE_02265_01779] [--checkpoint PATH]

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
CELL = REPO / "results/2026-11-06_prue_tessera" / CHIPS / "prue_tessera_v2_ft/P100_draw0_seed0"
CROPMAP = REPO / "results/2026-10-15_kshot_finetune" / CHIPS / "tessera_v2_medium_ft/P100_draw0_seed0"
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


def crop_colours(mask: np.ndarray, E: np.ndarray, n: int) -> dict:
    """One colour per crop: the chip's declared crops by area, then the others by index; the rest grey. The same
    rule in both experiments' diagrams, so a chip's crops keep their colours across them."""
    area = np.bincount(mask[E], minlength=n)
    order = [int(k) for k in np.argsort(-area, kind="stable") if area[k] > 0]
    order += [k for k in range(n) if k not in order]
    return {k: SLOTS[j] for j, k in enumerate(order[:len(SLOTS)])}



CLASS_RGB = {0: "#eadfc8", 1: "#9cc3f0", 2: "#0d366b"}      # background, interior, boundary


def run_model(cid: str, checkpoint: Path) -> dict:
    """The fine-tuned arm's checkpoint on one chip, end to end: the trained TESSERA student on the composites, the
    standardisation, the U-Net; its embedding, decoder features and class probabilities."""
    import torch

    import prue
    import tessera_ft
    from fit_prue import Net
    from gfm4agri.embeddings.tessera_v2 import chip_inputs as tessera_inputs
    from gfm4agri.embeddings.tessera_v2 import month_doy

    man = json.loads((REPO / "data/eurocrops_chips" / CHIPS / "manifest.json").read_text())
    stats = json.loads((checkpoint.parent / "embedding_stats.json").read_text())
    net = Net(prue.build_unet(128, "efficientnet-b7", None), stats["means"], stats["stds"],
              student=tessera_ft.load_student().eval(), trainable=False)
    st = torch.load(checkpoint, map_location="cpu", weights_only=False)
    net.load_state_dict(st["net"])
    net.eval()
    s2, s1, (h, w) = tessera_inputs(REPO / "data/eurocrops_chips" / CHIPS / "chips", cid,
                                    man["imagery"]["band_names"], man["imagery"]["n_months"])
    batch = {"prep": tessera_ft.prepare(s2, s1, month_doy(man["year"], man["imagery"]["n_months"])),
             "hw": (h, w), "n": 1}
    caught = {}
    hook = net.unet.decoder.register_forward_hook(lambda m, i, o: caught.__setitem__("features", o.detach()))
    with torch.no_grad():
        x = net.embed(batch, torch.device("cpu"))
        logits = net.unet(x)
    hook.remove()
    return {"emb": x[0].numpy(), "features": caught["features"][0].numpy(),
            "prob": torch.softmax(logits[0].float(), 0).numpy(), "epoch": int(st["epoch"])}


def load(cid: str, checkpoint: Path) -> dict:
    import panoptic_x as px
    import prue
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    names = json.loads((CROPMAP / "results.json").read_text())["class_names"]
    n = len(names)
    chips_dir = REPO / "data/eurocrops_chips" / CHIPS / "chips"
    with np.load(CROPMAP / "predictions_test.npz") as d:
        i = [str(c) for c in d["ids"]].index(cid)
        crop, mask = d["pred"][i].astype(np.int64), d["mask"][i].astype(np.int64)
    parcels = read_parcels(chips_dir, cid)
    wc = np.load(REPO / "data/worldcover" / CHIPS / "worldcover.npz")[cid]
    out = run_model(cid, checkpoint)
    prob = out["prob"]
    E, true, true_cls, void = px.truth(mask, parcels)
    fields, scores = {}, {}
    for v in prue.VARIANTS:
        seg = prue.parcels(prob, v)
        cls = px.vote(seg, crop, n)
        aware, agn, _ = px.chip_counts(true, true_cls, seg, cls, void, n)
        q = px.quality(aware, agn, names)
        fields[v] = (seg, cls)
        scores[v] = dict(n=int(len(np.unique(seg[seg > 0]))), found=int(agn[0]), right=int(aware[:, 0].sum()),
                         pq=q["pq"], pq_agn=q["pq_agnostic"])
    y = prue.target(mask, parcels, wc)
    target_b = y == prue.BOUNDARY
    hit = E & (prob.argmax(0) == prue.BOUNDARY)
    tp, fp, fn = (hit & target_b).sum(), (hit & ~target_b).sum(), (E & target_b & ~hit).sum()
    return dict(cid=cid, names=names, mask=mask, parcels=parcels, E=E, true=true, true_cls=true_cls, y=y,
                prob=prob, cls=prob.argmax(0), emb=pca_rgb(out["emb"]), feat=pca_rgb(out["features"]),
                fields=fields, scores=scores, crop=crop, epoch=out["epoch"], inputs=chip_inputs(cid),
                bf1=2 * tp / max(2 * tp + fp + fn, 1))


def pastel(seg: np.ndarray, seed: int = 0) -> np.ndarray:
    pal = np.random.default_rng(seed).uniform(0.4, 0.92, size=(int(seg.max()) + 1, 3))
    img = np.ones(seg.shape + (3,))
    img[seg > 0] = pal[seg[seg > 0]]
    img[outline(seg)] = to_rgb(INK)
    return img


def classes_image(c: np.ndarray, ignored: np.ndarray | None = None) -> np.ndarray:
    img = np.ones(c.shape + (3,))
    for k, col in CLASS_RGB.items():
        img[c == k] = to_rgb(col)
    if ignored is not None:
        img[ignored] = to_rgb(UNKNOWN)
    return img


def draw(D: dict, path: Path) -> None:
    names, E, true, mask, y = D["names"], D["E"], D["true"], D["mask"], D["y"]
    colours = crop_colours(mask, E, len(names))
    ignored = y == 3
    unknown = np.array(to_rgb(UNKNOWN))

    C = Canvas(21.0, 20.25)
    S = 2.55
    # ---------------------------------------------------------------- band 1: input to embedding to features
    C.band(0.25, 4.15, "1", "From satellite images to the TESSERA embedding, then the U-Net (all trained together)")
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
          "21 M parameters, pre-trained by its authors, then fine-tuned here with the U-Net (slowly: "
          "learning rate 0.00001)", badge=("TRAINED", ACCENT), width=40)
    C.arrow(7.55, 2.45, 8.0, 2.45)
    C.image(8.0, 1.05, S, S, D["emb"], title="TESSERA embedding (fine-tuned)",
            caption="128 numbers per pixel, shown as colour (their 3 main components): similar colour, similar "
                    "pixel", cap_width=36)
    C.arrow(10.55, 2.45, 11.0, 2.45)
    C.box(11.0, 1.25, 3.55, 2.4, "U-Net (EfficientNet-B7)",
          "looks at each pixel together with its neighbours, at 6 scales from 10 to 320 m\n"
          "67.2 M parameters, started from ImageNet, trained here", badge=("TRAINED", ACCENT), width=40)
    C.arrow(14.55, 2.45, 15.0, 2.45)
    C.image(15.0, 1.05, S, S, D["feat"], title="U-Net features",
            caption="16 numbers per pixel (3 main components) before the last layer", cap_width=36)
    C.text(17.85, 1.1, "How to read", fontsize=10, weight="bold", color=INK, va="top")
    C.text(17.85, 1.45, f"Every image is real: test chip {D['cid']}, never seen in training.\n"
                        "Boxes are steps of the model. TRAINED: learned from the 4,898 training chips.\n"
                        f"Checkpoint: epoch {D['epoch']} of 30, the one kept (best validation object F1).\n"
                        "Frozen arm: same diagram with TESSERA frozen.",
           width=34, fontsize=8.8, linespacing=1.35)

    # ----------------------------------------------------------------------------------- band 2: one head
    C.band(4.5, 9.95, "2", "One head, three classes per pixel: what each is trained to match, and what it outputs")
    colw, x0 = 4.0, 0.55
    cols = [x0 + k * (colw + 0.05) for k in range(4)]
    C.line([12.775, 12.775], [3.65, 5.05])
    C.line([cols[0] + colw / 2, cols[3] + colw / 2], [5.05, 5.05])
    for x in cols:
        C.arrow(x + colw / 2, 5.05, x + colw / 2, 5.35)
    heads = [("All three classes", "Which class is each pixel?\nthe most likely of the three"),
             ("Background", "Is this pixel off the fields?\nprobability"),
             ("Field interior", "Is it inside a field?\nprobability"),
             ("Field boundary", "Is it on a field's edge?\nprobability")]
    for x, (t, b) in zip(cols, heads):
        C.box(x + 0.1, 5.35, colw - 0.2, 1.15, t, b, edge=NAVY, face=LIGHT, width=44)
    ty, oy = 6.95, 10.8
    ix = [x + (colw - S) / 2 for x in cols]
    for x in cols:
        C.text(x + 0.12, ty - 0.33, "TRAINED TO MATCH", fontsize=8.5, weight="bold", color=NAVY)
        C.text(x + 0.12, oy - 0.33, "OUTPUT ON THIS TEST CHIP", fontsize=8.5, weight="bold", color=ACCENT)
    C.image(ix[0], ty, S, S, classes_image(y, ignored),
            caption="Sand: WorldCover's non-farm land. Light blue: inside a declared parcel. Dark: its outer ring. "
                    "Grey: undeclared grassland and cropland, no loss", cap_width=40)
    C.image(ix[0], oy, S, S, classes_image(D["cls"]),
            caption="The most likely class of every pixel, over the whole chip", cap_width=40)
    caps = {0: ("1 on WorldCover's non-farm land (forest, water, built-up), 0 on the declared parcels. "
                "Grey: no loss", "Dark: off the fields. It learned where fields are not"),
            1: ("1 inside every declared parcel, 0 on its ring and off the fields. Grey: no loss",
                "Dark: inside a field"),
            2: ("1 on the outer ring of every declared parcel (2 pixels between neighbours). Grey: no loss",
                f"Dark: on an edge. Pixel F1 of the boundary class inside the declared parcels: {D['bf1']:.2f}")}
    for j, k in enumerate((0, 1, 2), start=1):
        img = PROBABILITY((y == k).astype(float))[..., :3]
        img[ignored] = unknown
        C.image(ix[j], ty, S, S, img, caption=caps[k][0], cap_width=40)
        C.image(ix[j], oy, S, S, D["prob"][k], cmap=PROBABILITY, vmin=0, vmax=1, caption=caps[k][1],
                cap_width=40)
    tx = 17.0
    C.text(tx, 5.4, "How it is trained", fontsize=10.5, weight="bold", color=INK)
    C.text(tx, 5.8, "One softmax: each pixel's three probabilities add up to 1.\n"
                    "Loss: log-cosh Dice, with the boundary weighted most (0.05 background, 0.20 interior, 0.75 "
                    "boundary), on the known pixels only.\n"
                    "Background comes from WorldCover, as for PaPs: the model learns where fields are not.\n"
                    "30 epochs on the 4,898 training chips, 8 random 128 x 128 windows per step, flips and "
                    "rotations. U-Net at a learning rate of 0.003 (cosine), TESSERA at 0.00001.\n"
                    "The epoch kept is the one whose fields best match the validation parcels (object F1).",
           width=36, fontsize=9.0, linespacing=1.45)

    # ------------------------------------------------------------------------------------- band 3: to parcels
    C.band(14.65, 5.4, "3", "From the class map to parcels (post-processing), and the final output")
    py, step = 15.85, 3.42
    px_ = [0.6 + k * step for k in range(6)]
    sc = D["scores"]
    C.image(px_[0], py, S, S, classes_image(D["cls"]), title="Step 1: the class map",
            caption="Interior (light blue) is where fields are; the boundary class (dark) runs between neighbouring "
                    "fields and keeps them apart", cap_width=36)
    C.arrow(px_[0] + S + 0.1, py + S / 2, px_[1] - 0.1, py + S / 2)
    seg = D["fields"]["cc"][0]
    C.image(px_[1], py, S, S, pastel(seg), title="Step 2: cut into fields (PRUE's own)",
            caption=f"Each connected patch of interior pixels is a field; patches under 5 pixels (500 m\u00b2) are "
                    f"dropped: {sc['cc']['n']} fields", cap_width=36)
    C.arrow(px_[1] + S + 0.1, py + S / 2, px_[2] - 0.1, py + S / 2)
    seg = D["fields"]["grow"][0]
    C.image(px_[2], py, S, S, pastel(seg, 1), title="Step 3: give back the rim",
            caption="Each field grows into the boundary pixels around it (watershed on the boundary probability), "
                    "never into background: the same fields, now full size", cap_width=36)
    C.arrow(px_[2] + S + 0.1, py + S / 2, px_[3] - 0.1, py + S / 2)
    seg, cls = D["fields"]["grow"]
    g = sc["grow"]
    C.image(px_[3], py, S, S, class_image(cls[seg], seg > 0, colours, outline(seg)),
            title="Step 4: give each a crop  =  FINAL",
            caption=f"The crop most of its pixels get in the RQ1 crop-type map (fine-tuned TESSERA). {g['n']} "
                    f"fields; {g['found']} match a declared parcel (IoU > 0.5), {g['right']} with the right crop; "
                    f"chip PQ {g['pq']:.2f} ({g['pq_agn']:.2f} crop ignored)", cap_width=36)
    c = sc["cc"]
    C.text(px_[4], py + 0.1, "Without step 3", fontsize=10, weight="bold", color=INK)
    C.text(px_[4], py + 0.45, f"PRUE's own fields (step 2) with the same crops: {c['found']} match a declared "
                              f"parcel, chip PQ {c['pq']:.2f} ({c['pq_agn']:.2f} crop ignored). Their interiors "
                              "are a ring smaller than the parcels, so fewer reach IoU > 0.5.",
           width=34, fontsize=9, linespacing=1.4)
    C.line([px_[5] - 0.45, px_[5] - 0.45], [py - 0.4, py + S + 0.9], colour=BASELINE, lw=1.2)
    n_true = int(len(np.unique(true[true > 0])))
    C.image(px_[5], py, S, S, class_image(D["true_cls"][true], true > 0, colours, outline(true)),
            title="Compare: declared parcels",
            caption=f"{n_true} parcels (pieces inside the chip); the score counts a match at IoU > 0.5",
            cap_width=36)
    present = set(np.unique(mask[E]).tolist()) | set(np.unique(cls[seg[seg > 0]]).tolist())
    lx, ly = 0.6, 19.75
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
    C.save(path, [(0.2, 4.5), (4.48, 14.5), (14.6, 20.25)])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--chip", default="EE_02265_01779")
    ap.add_argument("--checkpoint", type=Path, default=CELL / "best.pt",
                    help="the fine-tuned arm's best.pt, with embedding_stats.json beside it")
    a = ap.parse_args()
    out = HERE / "figures" / "architecture.png"
    out.parent.mkdir(exist_ok=True)
    D = load(a.chip, a.checkpoint)
    draw(D, out)
    print(out)
    print(f"chip {D['cid']}, epoch {D['epoch']}, {len(np.unique(D['true'][D['true'] > 0]))} true pieces, boundary F1 "
          f"{D['bf1']:.3f}; " + "; ".join(f"{v}: {s}" for v, s in D["scores"].items()))


if __name__ == "__main__":
    main()
