"""The joint model as one diagram drawn from a real test chip, and real test chips to check its results.

The joint model (``joint.py``) is PRUE's U-Net with two heads on its last 16 decoder features: PRUE's border head
(background, field interior, field boundary) and the main workflow's crop head, which also reads TESSERA's 128
dimensions. Its fields come from the border head as in PRUE (``prue.parcels``, variant ``grow``) and each field takes
the crop most of its pixels get from the crop head (``panoptic_x.vote``), scored as ``score_joint.py`` scores them.

Every scored output drawn (both heads' maps, the fields, their crops and the check against the declared parcels) is
read from the arm's stored test predictions, ``predictions_test.npz``, so the figures show exactly what the test
scores were computed from. What is not stored (the embedding, the U-Net's features and the crop head's
probabilities) comes from a rerun of the kept checkpoint on the chip, on the CPU in float32 (bf16 on the cluster);
the script prints how many pixels of the rerun agree with the stored maps.

Outputs, in ``figures/``:

* ``architecture.png`` and its three bands apart (``architecture_1.png`` to ``_3.png``): from the satellite images to
  the features, the two heads with what each is trained to match and what it outputs, and the four steps from the
  heads' outputs to the fields with their crops, checked against the declared parcels;
* ``examples.png``: four test chips chosen from the whole test set by the share of their declared parcels found with
  the right crop (the 90th, 50th and 10th percentiles among chips with at least 20 parcels, and the chip with most
  fields matched with a wrong crop), each from the declared parcels to the check;
* ``architecture_tables.md``: the test-set totals recomputed field by field (they must equal ``results.json``), the
  crop accuracy of the matched fields per class, the most frequent wrong crops, and the chips drawn.

    poetry run python experiments/2026-11-07_prue_crop_tessera/architecture.py \\
        [--arm joint_tessera_v2_frozen_cw001] [--chip EE_02265_01779] [--workers 8]
"""

from __future__ import annotations

import argparse
import json
import sys
import textwrap
from concurrent.futures import ProcessPoolExecutor
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

import joint  # noqa: E402
import panoptic_x as px  # noqa: E402
import prue  # noqa: E402

NAME = "2026-11-07_prue_crop_tessera"
CHIPS = "EE_2021_rev"
CELL = "P100_draw0_seed0"
DPI = 110
#: WorldCover's grassland and cropland: farmland, declared or not.
FARMLAND = (30, 40)

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
OTHER = "#a3a29b"
SURFACE, INK, INK_2, MUTED, BASELINE = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#c3c2b7"
BAND = "#f3f2ee"
UNKNOWN = "#d4d3cc"
FAINT = "#ecebe6"
NAVY, LIGHT = "#0d366b", "#e3eefb"
ACCENT = "#eb6834"
PROBABILITY = LinearSegmentedColormap.from_list("probability", ["#ffffff", "#2a78d6", "#0d366b"])
CLASS_RGB = {0: "#eadfc8", 1: "#9cc3f0", 2: "#0d366b"}      # background, interior, boundary
#: The check of a field against the declared parcels, and its colour.
RIGHT, WRONG, FALSE, VOID = 1, 2, 3, 4
CHECK_RGB = {RIGHT: "#1baf7a", WRONG: "#eda100", FALSE: "#e34948", VOID: "#c3c2b7"}
CHECK_LABEL = {RIGHT: "matches a parcel, right crop", WRONG: "matches a parcel, wrong crop",
               FALSE: "matches no parcel", VOID: "mostly on a parcel outside the 20 crops: not scored"}
SHORT = {"pasture_meadow_grassland_grass": "grassland", "winter_common_soft_wheat": "winter wheat",
         "spring_common_soft_wheat": "spring wheat", "legumes_harvested_green": "legumes, green harvest",
         "winter_rapeseed_rape": "winter rapeseed", "spring_rapeseed_rape": "spring rapeseed",
         "grain_maize_corn_popcorn": "grain maize", "legumes_dried_pulses_protein_crops": "dried pulses",
         "alfalfa_lucerne": "alfalfa", "orchards_fruits": "orchards", "fresh_vegetables": "vegetables",
         "spring_barley": "spring barley", "winter_barley": "winter barley"}
plt.rcParams.update({"font.size": 10, "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE})


def short(name: str) -> str:
    return SHORT.get(name, name.replace("_", " "))


# ------------------------------------------------------------------------------------------------------- canvas

class Canvas:
    """A figure laid out in inches from the top left: bands, boxes, arrows and images (as PRUE's diagram)."""

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

    def arrow(self, x1, y1, x2, y2, dashed=False, colour=INK_2, lw=1.6):
        self.ov.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=16, color=colour,
                                          lw=lw, linestyle="--" if dashed else "-", zorder=2, shrinkA=0, shrinkB=0))

    def line(self, xs, ys, colour=INK_2, lw=1.6, dashed=False):
        self.ov.plot(xs, ys, color=colour, lw=lw, solid_capstyle="butt", linestyle="--" if dashed else "-",
                     zorder=2)

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

    def swatch(self, x, y, colour, label, fontsize=9):
        self.ov.add_patch(Rectangle((x, y - 0.12), 0.24, 0.24, facecolor=colour, edgecolor="none"))
        self.text(x + 0.32, y, label, fontsize=fontsize, va="center")
        return x + 0.55 + 0.072 * len(label) * fontsize / 9

    def save(self, path: Path, bands: list[tuple[float, float]] = ()):
        self.fig.savefig(path, dpi=DPI)
        plt.close(self.fig)
        from PIL import Image

        img = Image.open(path)
        for i, (y0, y1) in enumerate(bands, start=1):
            img.crop((0, int(y0 * DPI), img.width, int(min(y1, self.H) * DPI))).save(
                path.with_name(f"{path.stem}_{i}.png"))


# ----------------------------------------------------------------------------------------------- the pictures

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
        img[faint] = to_rgb(FAINT)
    for k in np.unique(cls_map[where]):
        img[where & (cls_map == k)] = to_rgb(colours.get(int(k), OTHER))
    if edges is not None:
        img[edges] = to_rgb(INK)
    return img


def classes_image(c: np.ndarray, ignored: np.ndarray | None = None) -> np.ndarray:
    img = np.ones(c.shape + (3,))
    for k, col in CLASS_RGB.items():
        img[c == k] = to_rgb(col)
    if ignored is not None:
        img[ignored] = to_rgb(UNKNOWN)
    return img


def pastel(seg: np.ndarray, seed: int = 0) -> np.ndarray:
    pal = np.random.default_rng(seed).uniform(0.4, 0.92, size=(int(seg.max()) + 1, 3))
    img = np.ones(seg.shape + (3,))
    img[seg > 0] = pal[seg[seg > 0]]
    img[outline(seg)] = to_rgb(INK)
    return img


def crop_colours(mask: np.ndarray, E: np.ndarray, n: int, also=(), extra: np.ndarray | None = None) -> dict:
    """One colour per crop: the chip's declared crops by area, then the crops in ``also``, then the others by index;
    the rest grey. The rule of the PRUE and PaPs diagrams, so a chip's crops keep their colours across them. With
    ``extra`` (per-class pixel counts, such as the crops voted inside the fields), ranked by the two areas together."""
    area = np.bincount(mask[E], minlength=n)
    if extra is not None:
        area = area + np.asarray(extra)[:n]
    order = [int(k) for k in np.argsort(-area, kind="stable") if area[k] > 0]
    order += [int(k) for k in also if int(k) not in order]
    order += [k for k in range(n) if k not in order]
    return {k: SLOTS[j] for j, k in enumerate(order[:len(SLOTS)])}


def check_image(outcome: np.ndarray, seg: np.ndarray, true: np.ndarray, missed: np.ndarray,
                farmland: np.ndarray) -> np.ndarray:
    """Fields coloured by their check; missed declared parcels hatched dark; undeclared farmland faint."""
    img = np.ones(outcome.shape + (3,))
    img[farmland] = to_rgb(FAINT)
    for k, col in CHECK_RGB.items():
        img[outcome == k] = to_rgb(col)
    miss = np.isin(true, missed) & (outcome == 0)
    yy, xx = np.mgrid[: outcome.shape[0], : outcome.shape[1]]
    img[miss & ((yy + xx) % 4 < 2)] = to_rgb("#52514e")
    img[miss & ((yy + xx) % 4 >= 2)] = to_rgb("#d4d3cc")
    img[outline(true)] = (1.0, 1.0, 1.0)
    img[outline(seg) & (outcome > 0)] = to_rgb(INK)
    return img


def s2_rgb(cid: str, month: int = 6) -> np.ndarray:
    """The chip's Sentinel-2 composite of ``month`` (0 = January) as true colour, stretched to its 98th percentile."""
    import rasterio

    man = json.loads((REPO / "data/eurocrops_chips" / CHIPS / "manifest.json").read_text())
    bands = man["imagery"]["band_names"]
    with rasterio.open(REPO / "data/eurocrops_chips" / CHIPS / "chips" / f"{cid}_merged.tif") as s:
        rgb = np.stack([s.read(month * len(bands) + bands.index(b) + 1) for b in ("RED", "GREEN", "BLUE")], -1)
    rgb = np.nan_to_num(rgb.astype(np.float32))
    return np.clip(rgb / max(np.percentile(rgb, 98), 1e-6), 0, 1)


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


# ------------------------------------------------------------------------------------------------- the model

def cell_dir(arm: str) -> Path:
    return REPO / "results" / NAME / CHIPS / arm / CELL


def rerun(cid: str, cell: Path, names: list[str]) -> dict:
    """The kept checkpoint on one chip, on the CPU: the standardised embedding, the U-Net's 16 decoder features, and
    both heads' probabilities. Frozen arms embed with the core's TESSERA v2 encoder (as the raster cache), fine-tuned
    arms with the trained student stored in the checkpoint."""
    import torch

    st = torch.load(cell / "best.pt", map_location="cpu", weights_only=False)
    state = st["net"]
    unet = prue.build_unet(128, "efficientnet-b7", None)
    head = joint.CropHead(128 + joint.decoder_channels(unet), len(names))
    man = json.loads((REPO / "data/eurocrops_chips" / CHIPS / "manifest.json").read_text())
    chips_dir = REPO / "data/eurocrops_chips" / CHIPS / "chips"
    if any(k.startswith("student.") for k in state):
        import tessera_ft
        from gfm4agri.embeddings.tessera_v2 import chip_inputs as tessera_inputs
        from gfm4agri.embeddings.tessera_v2 import month_doy

        net = joint.JointNet(unet, head, np.zeros(128), np.ones(128), student=tessera_ft.load_student().eval())
        net.load_state_dict(state)
        s2, s1, (h, w) = tessera_inputs(chips_dir, cid, man["imagery"]["band_names"], man["imagery"]["n_months"])
        batch = {"prep": tessera_ft.prepare(s2, s1, month_doy(man["year"], man["imagery"]["n_months"])),
                 "hw": (h, w), "n": 1}
    else:
        from gfm4agri.benchmark.backbones import get_backbone
        from gfm4agri.embeddings.tessera_v2 import WEIGHTS, ChipEncoder

        net = joint.JointNet(unet, head, np.zeros(128), np.ones(128))
        net.load_state_dict(state)
        c = get_backbone("tessera_v2_medium").computed
        enc = ChipEncoder(c["student"], c["checkpoint_sha256"], WEIGHTS, band_names=man["imagery"]["band_names"],
                          n_months=man["imagery"]["n_months"], year=man["year"], precision="32", device="cpu")
        raw = torch.from_numpy(np.asarray(enc.encode(enc.read(chips_dir, cid)), np.float32))[None]
        batch = {"image": (raw - net.emb_mean) / net.emb_std}
    net.eval()
    with torch.no_grad():
        x = net.embed(batch, torch.device("cpu"))
        dec = net.unet.decoder(net.unet.encoder(x))
        seg = net.unet.segmentation_head(dec)
        crop = net.crop_head(torch.cat([x, dec], dim=1))
    return {"emb": x[0].numpy(), "features": dec[0].numpy(), "border": torch.softmax(seg[0], 0).numpy(),
            "crop": torch.softmax(crop[0], 0).numpy(), "epoch": int(st["epoch"])}


# --------------------------------------------------------------------------------------- fields and the check

def chip_check(prob: np.ndarray, pred: np.ndarray, mask: np.ndarray, parcels: np.ndarray, n: int) -> dict:
    """The chip's fields (``grow``), their crops by majority vote, and the check of each against the declared parcels
    under the void rule of ``panoptic_x.chip_counts``: matched with the right crop, with a wrong crop, no match, or
    dropped as lying mostly on a parcel outside the scheme."""
    E, true, true_cls, void = px.truth(mask, parcels)
    seeds = prue.parcels(prob, "cc")
    seg = prue.parcels(prob, "grow")
    cls = px.vote(seg, pred, n)
    ti, si, iou, n_all, n_void = px.match(true, seg, void)
    s_ids = np.nonzero(n_all)[0]
    s_ids = s_ids[s_ids > 0]
    matched = np.zeros(len(n_all), bool)
    matched[si] = True
    dropped = s_ids[(~matched[s_ids]) & (n_void[s_ids] * 2 > n_all[s_ids])]
    right = true_cls[ti] == cls[si]
    lab = np.zeros(len(n_all), np.uint8)
    lab[s_ids] = FALSE
    lab[dropped] = VOID
    lab[si[right]] = RIGHT
    lab[si[~right]] = WRONG
    t_ids = np.unique(true[true > 0])
    return dict(E=E, true=true, true_cls=true_cls, void=void, seeds=seeds, seg=seg, cls=cls, outcome=lab[seg],
                missed=np.setdiff1d(t_ids, ti), n_true=len(t_ids), n_fields=len(s_ids) - len(dropped),
                n_seeds=int(len(np.unique(seeds[seeds > 0]))), matched=len(ti), right=int(right.sum()),
                wrong=int((~right).sum()), false=int(len(s_ids) - len(dropped) - len(ti)), dropped=len(dropped),
                pairs=np.stack([true_cls[ti], cls[si]], 1))


def _chunk(args):
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    ids, probs, preds, masks, n, chips_dir = args
    rows, pairs = [], []
    for i, cid in enumerate(ids):
        c = chip_check(probs[i], preds[i].astype(np.int64), masks[i].astype(np.int64),
                       read_parcels(chips_dir, cid), n)
        rows.append((cid, c["n_true"], c["n_fields"], c["matched"], c["right"], c["wrong"], c["false"],
                     c["dropped"]))
        pairs.append(c["pairs"])
    return rows, np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int64)


def all_chips(store: dict, n: int, workers: int, chunk: int = 40):
    """Every test chip's check: one row per chip and the (true, voted) crop of every matched field."""
    ids = [str(c) for c in store["ids"]]
    chips_dir = REPO / "data/eurocrops_chips" / CHIPS / "chips"
    jobs = [(ids[a:a + chunk], store["prob"][a:a + chunk], store["pred"][a:a + chunk], store["mask"][a:a + chunk],
             n, chips_dir) for a in range(0, len(ids), chunk)]
    rows, pairs = [], []
    with ProcessPoolExecutor(workers) as ex:
        for r, p in ex.map(_chunk, jobs):
            rows += r
            pairs.append(p)
    return rows, np.concatenate(pairs)


def pick(rows: list[tuple], min_parcels: int = 20) -> list[tuple[str, str]]:
    """Chips at the 90th, 50th and 10th percentile of the share of declared parcels found with the right crop, among
    chips with at least ``min_parcels``, and the chip with most fields matched with a wrong crop."""
    big = [r for r in rows if r[1] >= min_parcels]
    share = np.array([r[4] / r[1] for r in big])
    out = []
    for q, why in ((90, "good: 90th percentile"), (50, "typical: median"), (10, "poor: 10th percentile")):
        v = np.percentile(share, q)
        k = min(range(len(big)), key=lambda j: (abs(share[j] - v), big[j][0]))
        out.append((big[k][0], why))
    taken = {c for c, _ in out}
    k = max((j for j in range(len(big)) if big[j][0] not in taken), key=lambda j: (big[j][5], big[j][0]))
    out.append((big[k][0], "most fields with a wrong crop"))
    return out


# ------------------------------------------------------------------------------------------------- the diagram

def load_chip(cid: str, cell: Path, store: dict, names: list[str], with_model: bool) -> dict:
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    i = [str(c) for c in store["ids"]].index(cid)
    prob, pred, mask = store["prob"][i], store["pred"][i].astype(np.int64), store["mask"][i].astype(np.int64)
    parcels = read_parcels(REPO / "data/eurocrops_chips" / CHIPS / "chips", cid)
    wc = np.load(REPO / "data/worldcover" / CHIPS / "worldcover.npz")[cid]
    D = dict(cid=cid, names=names, prob=prob.astype(np.float32) / 255, pred=pred, mask=mask, parcels=parcels, wc=wc,
             y=prue.target(mask, parcels, wc), yc=joint.crop_target(mask, parcels),
             farmland=np.isin(wc, FARMLAND) & (parcels == 0), **chip_check(prob, pred, mask, parcels, len(names)))
    if with_model:
        m = rerun(cid, cell, names)
        D.update(emb=pca_rgb(m["emb"]), feat=pca_rgb(m["features"]), conf=m["crop"].max(0), epoch=m["epoch"],
                 agree_border=float((m["border"].argmax(0) == prob.argmax(0)).mean()),
                 agree_crop=float((m["crop"].argmax(0) == pred).mean()), inputs=chip_inputs(cid))
    return D


def draw(D: dict, arm: dict, path: Path) -> None:
    names, E, true, mask = D["names"], D["E"], D["true"], D["mask"]
    cls_map = D["prob"].argmax(0)
    colours = crop_colours(mask, E, len(names), also=np.bincount(D["pred"][D["seg"] > 0]).argsort()[::-1][:3])
    ignored = D["y"] == prue.IGNORE
    unknown = np.array(to_rgb(UNKNOWN))
    frozen = arm["frozen"]

    C = Canvas(21.0, 21.45)
    S = 2.55
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
    if frozen:
        C.box(4.0, 1.25, 3.55, 2.4, "TESSERA v2 encoder",
              "reads the 12-month series of each pixel on its own and turns it into 128 numbers\n"
              "21 M parameters, pre-trained by its authors, frozen here: its output is computed once and cached",
              badge=("FROZEN", MUTED), width=40)
    else:
        C.box(4.0, 1.25, 3.55, 2.4, "TESSERA v2 encoder",
              "reads the 12-month series of each pixel on its own and turns it into 128 numbers\n"
              "21 M parameters, pre-trained by its authors, then fine-tuned here with both heads (learning rate "
              "0.00001)", badge=("TRAINED", ACCENT), width=40)
    C.arrow(7.55, 2.45, 8.0, 2.45)
    C.image(8.0, 1.05, S, S, D["emb"], title="TESSERA embedding",
            caption="128 numbers per pixel, shown as colour (3 main components)", cap_width=30)
    C.arrow(10.55, 2.45, 11.0, 2.45)
    C.box(11.0, 1.25, 3.55, 2.4, "U-Net (EfficientNet-B7)",
          "looks at each pixel together with its neighbours, at 6 scales from 10 to 320 m\n"
          "67.2 M parameters, started from ImageNet, trained here by both heads", badge=("TRAINED", ACCENT),
          width=40)
    C.arrow(14.55, 2.45, 15.0, 2.45)
    C.image(15.0, 1.05, S, S, D["feat"], title="U-Net features",
            caption="16 numbers per pixel (3 main components), shared by both heads", cap_width=36)
    C.text(17.85, 1.1, "How to read", fontsize=10, weight="bold", color=INK, va="top")
    C.text(17.85, 1.45, f"Every image is real: test chip {D['cid']}, never seen in training.\n"
                        "Boxes are steps of the model. TRAINED: learned from the 4,898 training chips.\n"
                        f"{arm['label']}, crop loss weight {arm['weight']:g}. Checkpoint: epoch {D['epoch']} of 30, "
                        "the one kept (best validation mean of object F1 and crop Macro-F1).",
           width=34, fontsize=8.8, linespacing=1.35)

    # ------------------------------------------------------------------------------------- band 2: two heads
    C.band(4.5, 10.7, "2", "Two heads on the same features: what each is trained to match, and what it outputs")
    colw = 4.0
    cols = [0.55, 4.6, 8.75, 12.8]
    bx, cx = (cols[0] + cols[1] + colw) / 2, (cols[2] + cols[3] + colw) / 2
    C.line([12.775, 12.775], [3.65, 5.25])
    C.line([bx, cx], [5.25, 5.25])
    C.arrow(bx, 5.25, bx, 5.55)
    C.arrow(cx, 5.25, cx, 5.55)
    C.line([10.78, 10.78], [2.45, 5.1], dashed=True)
    C.line([10.78, cx - 0.6], [5.1, 5.1], dashed=True)
    C.arrow(cx - 0.6, 5.1, cx - 0.6, 5.55, dashed=True)
    C.text(10.86, 4.98, "the embedding too", fontsize=8.3, va="bottom", color=INK_2)
    C.box(cols[0] + 0.1, 5.55, 2 * colw - 0.15, 1.2, "Border head (PRUE's): 435 parameters",
          "Is this pixel off the fields, inside a field or on its edge?\n"
          "16 features in, 3 probabilities out (they add up to 1)", edge=NAVY, face=LIGHT, width=70)
    C.box(cols[2] + 0.1, 5.55, 2 * colw - 0.15, 1.2, "Crop head (new): 0.21 M parameters",
          "Which of the 20 crops grows on this pixel?\n"
          "128 + 16 = 144 numbers in, 20 probabilities out (they add up to 1)", edge=ACCENT, face="#fdf0e8",
          width=70)
    ty, oy = 7.6, 11.5
    ix = [x + (colw - S) / 2 for x in cols]
    for x in cols:
        C.text(x + 0.12, ty - 0.62, "TRAINED TO MATCH", fontsize=8.5, weight="bold", color=NAVY)
        C.text(x + 0.12, oy - 0.33, "OUTPUT ON THIS TEST CHIP", fontsize=8.5, weight="bold", color=ACCENT)
    C.image(ix[0], ty, S, S, classes_image(D["y"], ignored), title="Class: the most likely of 3",
            caption="Sand: WorldCover's non-farm land. Light blue: inside a declared parcel. Dark: its outer ring. "
                    "Grey: undeclared farmland, no loss", cap_width=40)
    C.image(ix[0], oy, S, S, classes_image(cls_map), caption="The most likely class of every pixel", cap_width=40)
    tb = PROBABILITY((D["y"] == prue.BOUNDARY).astype(float))[..., :3]
    tb[ignored] = unknown
    C.image(ix[1], ty, S, S, tb, title="Probability of the boundary",
            caption="1 on the outer ring of every declared parcel (2 pixels between neighbours). Grey: no loss",
            cap_width=40)
    C.image(ix[1], oy, S, S, D["prob"][prue.BOUNDARY], cmap=PROBABILITY, vmin=0, vmax=1,
            caption="Dark: on an edge. With the other two (not drawn) it adds up to 1", cap_width=40)
    yc = D["yc"]
    C.image(ix[2], ty, S, S, class_image(yc, yc >= 0, colours, faint=yc < 0), title="Crop: the most likely of 20",
            caption="The declared crop on every pixel of a declared parcel. Grey: no loss, everywhere else", cap_width=40)
    pct = 100 * (yc >= 0).mean()
    C.image(ix[2], oy, S, S, class_image(D["pred"], np.ones_like(yc, bool), colours),
            caption="The most likely crop of every pixel, forest and water included: it answers everywhere; only "
                    "the pixels inside a field are used (band 3)", cap_width=40)
    loss = np.where(yc >= 0, 1.0, 0.0)
    C.image(ix[3], ty, S, S, PROBABILITY(loss)[..., :3], title="How sure: the top crop's probability",
            caption=f"No target of its own. Dark: the pixels the crop loss sees ({pct:.0f} % of this chip; the "
                    "border loss also sees the background)", cap_width=40)
    C.image(ix[3], oy, S, S, D["conf"], cmap=PROBABILITY, vmin=0, vmax=1,
            caption="Dark: sure of its crop, on forest and water too: it has no 'no crop' answer. From a rerun "
                    "on the CPU, whose top crop is the stored test output's on "
                    f"{100 * D['agree_crop']:.0f} % of the pixels", cap_width=40)
    tx = 17.05
    C.text(tx, 5.55, "How they are trained", fontsize=10.5, weight="bold", color=INK)
    C.text(tx, 5.95, "Each head has its own target and loss:\n"
                     "border head: log-cosh Dice, boundary weighted most (0.05, 0.20, 0.75), as PRUE;\n"
                     "crop head: cross-entropy, on the pixels of declared parcels only.\n"
                     f"Total loss = Dice + {arm['weight']:g} x cross-entropy.\n"
                     "The heads are separate, so the weight hardly changes how fast each learns (the optimiser, "
                     "Adam, scales each step to its own gradient). It decides how each step of the shared U-Net is "
                     "split between the two tasks. At 0.1 the crop task took most of it and the borders fell "
                     "below PRUE's; at 0.01 the borders lead and match PRUE's.\n"
                     "30 epochs on the 4,898 training chips, 8 random 128 x 128 windows per step. One checkpoint for "
                     "both heads: the best validation mean of PRUE's object F1 and the crop Macro-F1.",
           width=36, fontsize=8.9, linespacing=1.42)

    # ------------------------------------------------------------------------------------- band 3: to parcels
    C.band(15.35, 6.05, "3", "From the two heads' outputs to fields with a crop (post-processing), and the check")
    py, step = 16.55, 3.42
    px_ = [0.6 + k * step for k in range(6)]
    C.image(px_[0], py, S, S, classes_image(cls_map), title="Step 1: the class map",
            caption="Border head: each pixel's most likely class. Interior (light blue) is where fields are; "
                    "the boundary (dark) runs between them", cap_width=36)
    C.arrow(px_[0] + S + 0.1, py + S / 2, px_[1] - 0.1, py + S / 2)
    C.image(px_[1], py, S, S, pastel(D["seeds"]), title="Step 2: cut into seeds",
            caption=f"Each patch of interior pixels touching by a side is a seed; under 5 pixels (500 m²) "
                    f"dropped: {D['n_seeds']} seeds. PRUE's own fields", cap_width=36)
    C.arrow(px_[1] + S + 0.1, py + S / 2, px_[2] - 0.1, py + S / 2)
    C.image(px_[2], py, S, S, pastel(D["seg"], 1), title="Step 3: give back the rim",
            caption="Each seed grows into the boundary pixels around it (watershed on the boundary probability), "
                    "never into background: the same fields, full size", cap_width=36)
    C.arrow(px_[2] + S + 0.1, py + S / 2, px_[3] - 0.1, py + S / 2)
    seg, cls = D["seg"], D["cls"]
    C.image(px_[3], py, S, S, class_image(cls[seg], seg > 0, colours, outline(seg)),
            title="Step 4: a crop by majority  =  FINAL",
            caption="Crop head: each field takes the crop most of its pixels get (its rim included). "
                    f"{D['n_fields'] + D['dropped']} fields", cap_width=36)
    C.line([px_[4] - 0.45, px_[4] - 0.45], [py - 0.4, py + S + 1.2], colour=BASELINE, lw=1.2)
    C.image(px_[4], py, S, S, class_image(D["true_cls"][true], true > 0, colours, outline(true),
                                          faint=D["farmland"]),
            title="Compare: declared parcels",
            caption=f"{D['n_true']} parcels (pieces inside the chip) with their crops. Faint: farmland nobody "
                    "declared", cap_width=36)
    C.image(px_[5], py, S, S, check_image(D["outcome"], seg, true, D["missed"], D["farmland"]),
            title="Check: each field against them",
            caption=f"{D['matched']} fields match a parcel (IoU > 0.5): {D['right']} with the right crop, "
                    f"{D['wrong']} with a wrong one; {D['false']} match none. {len(D['missed'])} parcels missed "
                    "(hatched)", cap_width=36)
    present = set(np.unique(mask[E]).tolist()) | set(np.unique(cls[seg[seg > 0]]).tolist()) | \
        set(np.unique(D["pred"]).tolist())
    lx, ly = 0.6, 20.65
    C.text(lx, ly, "Crop colours:", fontsize=9, weight="bold", color=INK, va="center")
    lx += 1.15
    for k in [k for k in colours if k in present]:
        lx = C.swatch(lx, ly, colours[k], short(names[k]))
    if any(k not in colours for k in present):
        C.swatch(lx, ly, OTHER, "other crops")
    lx, ly = 0.6, 21.05
    C.text(lx, ly, "Check:", fontsize=9, weight="bold", color=INK, va="center")
    lx += 1.15
    for k in (RIGHT, WRONG, FALSE, VOID):
        lx = C.swatch(lx, ly, CHECK_RGB[k], CHECK_LABEL[k])
    C.swatch(lx, ly, "#52514e", "declared parcel with no field on it (hatched)")
    C.save(path, [(0.2, 4.5), (4.48, 15.3), (15.3, 21.45)])


def draw_examples(chips: list[tuple[dict, str]], arm: dict, path: Path) -> None:
    """One row per chip: Sentinel-2, the declared parcels, the crop head inside the fields, the fields after the
    vote, and the check."""
    rows = len(chips)
    S, gx, top, rh = 3.0, 0.32, 0.95, 3.75
    W = 0.5 + 5 * (S + gx) + 3.7
    C = Canvas(W, top + rows * rh + 1.0)
    titles = ["Sentinel-2, July", "Declared parcels, their crops", "Crop head per pixel, in the fields",
              "Step 4: one crop per field (final)", "Check: fields against parcels"]
    for j, t in enumerate(titles):
        C.text(0.5 + j * (S + gx), 0.42, t, fontsize=10.5, weight="bold", color=INK, va="center")
    for r, (D, why) in enumerate(chips):
        y = top + r * rh
        names, E, true, seg, cls = D["names"], D["E"], D["true"], D["seg"], D["cls"]
        colours = crop_colours(D["mask"], E, len(names), extra=np.bincount(cls[seg[seg > 0]], minlength=len(names))
                               + np.bincount(D["pred"][seg > 0], minlength=len(names)))
        xs = [0.5 + j * (S + gx) for j in range(5)]
        C.image(xs[0], y, S, S, s2_rgb(D["cid"]))
        C.image(xs[1], y, S, S, class_image(D["true_cls"][true], true > 0, colours, outline(true),
                                            faint=D["farmland"]))
        inside = seg > 0
        C.image(xs[2], y, S, S, class_image(D["pred"], inside, colours, outline(seg), faint=D["farmland"] & ~inside))
        C.image(xs[3], y, S, S, class_image(cls[seg], inside, colours, outline(seg), faint=D["farmland"] & ~inside))
        C.image(xs[4], y, S, S, check_image(D["outcome"], seg, true, D["missed"], D["farmland"]))
        tx = xs[4] + S + 0.3
        C.text(tx, y, f"{D['cid']}", fontsize=10.5, weight="bold", color=INK)
        C.text(tx, y + 0.3, why, fontsize=9, color=ACCENT, weight="bold")
        C.text(tx, y + 0.62, f"{D['n_true']} declared parcels\n{D['n_fields'] + D['dropped']} fields drawn\n"
                             f"{D['matched']} match a parcel: {D['right']} right crop, {D['wrong']} wrong\n"
                             f"{D['false']} match none; {len(D['missed'])} parcels missed\n"
                             f"found with the right crop: {D['right'] / max(D['n_true'], 1):.0%} of the parcels",
               fontsize=8.8, linespacing=1.4)
        present = set(np.unique(D["mask"][E]).tolist()) | set(np.unique(cls[seg[inside]]).tolist()) | \
            set(np.unique(D["pred"][inside]).tolist())
        shown = [k for k in colours if k in present]
        if any(k not in colours for k in present):
            shown = shown[:7] + [None]
        for j, k in enumerate(shown[:8]):
            r_, c_ = divmod(j, 2)
            C.swatch(tx + c_ * 1.85, y + 2.08 + r_ * 0.27, colours[k] if k is not None else OTHER,
                     short(names[k]) if k is not None else "other crops", fontsize=8.0)
    ly = top + rows * rh + 0.35
    lx = 0.5
    C.text(lx, ly, "Check:", fontsize=9, weight="bold", color=INK, va="center")
    lx += 0.75
    for k in (RIGHT, WRONG, FALSE, VOID):
        lx = C.swatch(lx, ly, CHECK_RGB[k], CHECK_LABEL[k])
    lx = C.swatch(lx, ly, "#52514e", "declared parcel with no field on it (hatched)")
    C.swatch(lx, ly, FAINT, "farmland nobody declared")
    C.text(0.5, ly + 0.35, f"{arm['label']}, crop loss weight {arm['weight']:g}: the stored test output. Chips "
                           "chosen from all 1,475 test chips with at least 20 declared parcels, by the share of "
                           "their parcels found with the right crop. Check panel: black lines are the fields' edges, "
                           "white lines the declared parcels' (a white line inside one field: parcels merged).",
           fontsize=8.8)
    C.save(path)


def tables(rows, pairs, names, results: dict, picked, path: Path, arm: dict) -> None:
    n = len(names)
    tot = np.array([r[1:] for r in rows]).sum(0)
    n_true, n_fields, matched, right, wrong, false, dropped = (int(v) for v in tot)
    g = results["panoptic_metrics"]["variants"]["chip/grow"]
    aware_tp = sum(v["tp"] for v in g["per_class"].values())
    L = [f"# Architecture note tables: {arm['label']}, crop loss weight {arm['weight']:g}", "",
         "## Test-set totals, recomputed field by field", "",
         "| | recomputed | results.json |", "|---|---|---|",
         f"| declared parcels (pieces) | {n_true:,} | {g['true_parcels']:,} |",
         f"| fields drawn and scored | {n_fields:,} | {g['predicted_parcels']:,} |",
         f"| fields matching a parcel (IoU > 0.5) | {matched:,} | {g['tp_agnostic']:,} |",
         f"| of which with the right crop | {right:,} ({right / matched:.1%}) | {aware_tp:,} |",
         f"| of which with a wrong crop | {wrong:,} ({wrong / matched:.1%}) | |",
         f"| fields matching no parcel | {false:,} | {g['fp_agnostic']:,} |",
         f"| fields mostly on a parcel outside the scheme, not scored | {dropped:,} | {g['dropped_void']:,} |",
         f"| declared parcels found with the right crop | {right / n_true:.1%} | |", "",
         "## Crop of the matched fields, per declared crop", "",
         "| Declared crop | matched fields | right crop | most frequent wrong crop |", "|---|---|---|---|"]
    cm = np.zeros((n, n), np.int64)
    np.add.at(cm, (pairs[:, 0], pairs[:, 1]), 1)
    for k in np.argsort(-cm.sum(1)):
        m = cm[k].sum()
        if m == 0:
            continue
        off = cm[k].copy()
        off[k] = 0
        w = int(off.argmax())
        worst = f"{short(names[w])} ({off[w] / m:.0%})" if off[w] else ""
        L.append(f"| {short(names[k])} | {m:,} | {cm[k, k] / m:.0%} | {worst} |")
    L += ["", "## The most frequent wrong crops", "", "| Declared | Voted | fields | share of all wrong |",
          "|---|---|---|---|"]
    off = cm.copy()
    np.fill_diagonal(off, 0)
    for f in np.argsort(-off.ravel())[:10]:
        a, b = divmod(int(f), n)
        if off[a, b]:
            L.append(f"| {short(names[a])} | {short(names[b])} | {off[a, b]:,} | {off[a, b] / off.sum():.0%} |")
    L += ["", "## The chips of the examples", "", "| Chip | why | parcels | fields | matched | right crop | wrong crop |"
          " no match | missed |", "|---|---|---|---|---|---|---|---|---|"]
    by = {r[0]: r for r in rows}
    for cid, why in picked:
        r = by[cid]
        L.append(f"| {cid} | {why} | {r[1]} | {r[2] + r[7]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | {r[1] - r[3]} |")
    path.write_text("\n".join(L) + "\n")


ARMS = {"joint_tessera_v2_frozen_cw001": dict(label="TESSERA frozen", weight=0.01, frozen=True),
        "joint_tessera_v2_frozen": dict(label="TESSERA frozen", weight=0.1, frozen=True),
        "joint_tessera_v2_ft_cw001": dict(label="TESSERA fine-tuned", weight=0.01, frozen=False),
        "joint_tessera_v2_ft": dict(label="TESSERA fine-tuned", weight=0.1, frozen=False)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", default="joint_tessera_v2_frozen_cw001", choices=list(ARMS))
    ap.add_argument("--chip", default="EE_02265_01779", help="the chip of the PRUE and PaPs diagrams")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    arm, cell = ARMS[a.arm], cell_dir(a.arm)
    results = json.loads((cell / "results.json").read_text())
    names = results["class_names"]
    with np.load(cell / "predictions_test.npz") as z:
        store = {k: z[k] for k in ("prob", "pred", "mask", "ids")}
    figs = HERE / "figures"
    figs.mkdir(exist_ok=True)

    D = load_chip(a.chip, cell, store, names, with_model=True)
    draw(D, arm, figs / "architecture.png")
    print(f"architecture: chip {a.chip}, epoch {D['epoch']}; rerun agrees with the stored output on "
          f"{D['agree_border']:.2%} (border class) and {D['agree_crop']:.2%} (crop) of the pixels; "
          f"{D['n_true']} parcels, {D['n_fields']} fields, {D['matched']} matched, {D['right']} right crop")

    rows, pairs = all_chips(store, len(names), a.workers)
    picked = pick(rows)
    draw_examples([(load_chip(c, cell, store, names, with_model=False), why) for c, why in picked], arm,
                  figs / "examples.png")
    tables(rows, pairs, names, results, picked, figs / "architecture_tables.md", arm)
    print(f"examples: {picked}")
    print((figs / "architecture_tables.md").read_text())


if __name__ == "__main__":
    main()
