"""Crop maps of held-out test blocks, as the models would map an area they have never seen (REPORT.md section 3.5).

The test blocks of the split are areas the models never saw, 1.6 km away from any training parcel, and every fit
predicted every 10 m pixel of every test tile, forest, roads and towns included. This script reads those stored
predictions (``predictions_test.npz``), refits nothing, and mosaics whole blocks of 4 x 4 tiles (8.96 km):

* ``map_block_<block>.png``: per block, the July true colour and the declared crops, then for fine-tuned TESSERA v2,
  frozen TerraMind and the U-Net baseline at K = 100 and 5 %: the raw map (every pixel as predicted, since no
  cropland mask exists yet) and the per-parcel map (each declared parcel takes the crop most of its pixels were
  predicted as; scored parcels mapped wrong are outlined);
* ``map_zoom.png``: about 1.1 km windows of two blocks at full resolution, the same models at K = 100 %.

The parcel vote is the pipeline's (:mod:`gfm4agri.benchmark.parcel_metrics`). Before drawing, the script reproduces
each fit's parcel accuracy over all test tiles from the same votes and stops if it differs from ``results.json``.

    poetry run python docs/phase1/results_estonia/maps.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import rasterio  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from pyproj import Transformer  # noqa: E402

from gfm4agri.benchmark.parcel_metrics import chip_votes, pool_votes, read_parcels  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
CHIPS = REPO / "data/eurocrops_chips/EE_2021_rev"
SPLIT = REPO / "data/eurocrops_chips/EE_2021/splits/blocks4_buf1600_seed0__6b0eb4cb/chips.csv"
OUT = HERE / "figures"
TILE, GRID_M = 224, 2240

#: Model label and the fit folder holding P<k>_draw0_seed0/.
MODELS = [
    ("TESSERA v2 M, fine-tuned", REPO / "results/2026-10-15_kshot_finetune/EE_2021_rev/tessera_v2_medium_ft"),
    ("TerraMind v1 L, frozen", REPO / "results/kshot/EE_2021_rev/terramind_v1_large_s2s1"),
    ("U-Net from scratch", REPO / "results/kshot/EE_2021_rev/unet_r18_scratch_s2s1"),
]
KS = [100, 5]

#: Complete test blocks, picked from the 40 of the split for their landscape (profile of 6 November 2026).
BLOCKS = [
    ("583_457", "large arable fields: median parcel 6.2 ha, 52 % of the block declared"),
    ("593_445", "small plots: 32 potato parcels, 8 % of parcels under 0.16 ha"),
    ("594_443", "mostly forest and settlements: 11 % of the block declared, median parcel 1 ha"),
]
#: Zoom windows: block, top-left pixel (row, col) in the block mosaic, size in pixels.
#: Picked for the most crops besides grassland (583_457) and the most potato parcels (593_445) in 1.12 km.
ZOOMS = [("583_457", (280, 140), 112), ("593_445", (784, 168), 112)]

# Crop groups: one hue each (the reference palette of the dataviz method), crops within a group by lightness.
# Grassland, half of all parcels, is a recessive neutral so the arable crops stand out.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
NO_PARCEL, OTHER_CROP = "#f0efec", "#c3c2b7"
GROUPS = [
    ("Grassland", "#cfd8c4", ["pasture_meadow_grassland_grass"]),
    ("Winter cereals", "#eb6834", ["winter_common_soft_wheat", "winter_barley", "rye"]),
    ("Spring cereals", "#eda100", ["spring_barley", "oats", "spring_common_soft_wheat"]),
    ("Forage legumes", "#008300", ["legumes_harvested_green", "clover", "alfalfa_lucerne"]),
    ("Oilseeds", "#6250d6", ["winter_rapeseed_rape", "spring_rapeseed_rape"]),
    ("Pulses", "#2a78d6", ["peas", "beans", "legumes_dried_pulses_protein_crops"]),
    ("Potatoes, vegetables", "#e87ba4", ["potatoes", "fresh_vegetables"]),
    ("Other crops", "#1baf7a", ["grain_maize_corn_popcorn", "buckwheat", "orchards_fruits"]),
]
SHORT = {
    "pasture_meadow_grassland_grass": "grassland", "winter_common_soft_wheat": "winter wheat",
    "winter_barley": "winter barley", "rye": "rye", "spring_barley": "spring barley", "oats": "oats",
    "spring_common_soft_wheat": "spring wheat", "legumes_harvested_green": "legumes, green",
    "clover": "clover", "alfalfa_lucerne": "alfalfa", "winter_rapeseed_rape": "winter rapeseed",
    "spring_rapeseed_rape": "spring rapeseed", "peas": "peas", "beans": "beans",
    "legumes_dried_pulses_protein_crops": "dried pulses", "potatoes": "potatoes",
    "fresh_vegetables": "vegetables", "grain_maize_corn_popcorn": "maize", "buckwheat": "buckwheat",
    "orchards_fruits": "orchards",
}


#: Lightness steps within a group: the base hue for its most common crop, then darker, then lighter.
STEPS = [0.0, -0.3, 0.5]


def _shade(hex_: str, i: int) -> np.ndarray:
    base, t = np.array(to_rgb(hex_)), STEPS[i]
    return base * (1 + t) if t < 0 else base + (1 - base) * t


def class_colours(names: list[str]) -> np.ndarray:
    """RGB per class index, in the order of the chip manifest."""
    rgb = np.zeros((len(names), 3))
    for _, hex_, members in GROUPS:
        for i, m in enumerate(members):
            rgb[names.index(m)] = _shade(hex_, i)
    return rgb


def load_fit(fit: Path, k: int) -> dict:
    """Stored predictions of one fit, by chip id, with the parcel scores it logged."""
    cell = fit / f"P{k}_draw0_seed0"
    d = np.load(cell / "predictions_test.npz")
    r = json.loads((cell / "results.json").read_text())
    return {"pred": dict(zip(d["ids"].tolist(), d["pred"], strict=True)),
            "mask": dict(zip(d["ids"].tolist(), d["mask"], strict=True)),
            "parcel": r["parcel_metrics"], "classes": r["class_names"]}


def check_votes(fit: dict, parcels: dict) -> float:
    """Parcel accuracy over all test tiles from the pipeline's vote; must equal the logged one."""
    n = len(fit["classes"])
    parts = [chip_votes(fit["pred"][c], fit["mask"][c], parcels[c], n) for c in fit["pred"]]
    _, votes, ref = pool_votes(parts, n)
    acc = float((votes.argmax(1) == ref).mean())
    logged = fit["parcel"]["accuracy"]
    if abs(acc - logged) > 1e-9 or len(ref) != fit["parcel"]["n_parcels"]:
        raise SystemExit(f"parcel vote gives accuracy {acc:.6f} over {len(ref)} parcels, "
                         f"results.json {logged:.6f} over {fit['parcel']['n_parcels']}")
    return acc


def block_chips(split: pd.DataFrame, block: str) -> pd.DataFrame:
    b = split[(split.block_id == block) & (split.partition == "test")]
    if len(b) != 16:
        raise SystemExit(f"block {block} has {len(b)} test tiles, not 16")
    return b


def mosaic(b: pd.DataFrame, tiles: dict, fill=0) -> np.ndarray:
    """Tiles placed on the grid: column east, row north (the chip id's row counts northings)."""
    c0, r1 = b.col.min(), b.row.max()
    first = tiles[b.chip_id.iloc[0]]
    out = np.full((4 * TILE, 4 * TILE) + first.shape[2:], fill, dtype=first.dtype)
    for cid, col, row in zip(b.chip_id, b.col, b.row, strict=True):
        y, x = (r1 - row) * TILE, (col - c0) * TILE
        out[y:y + TILE, x:x + TILE] = tiles[cid]
    return out


def true_colour(cid: str, bands: list[str], month: int = 6) -> np.ndarray:
    idx = [month * len(bands) + bands.index(b) + 1 for b in ("RED", "GREEN", "BLUE")]
    with rasterio.open(CHIPS / "chips" / f"{cid}_merged.tif") as src:
        return np.moveaxis(src.read(idx).astype(np.float32) / 10000, 0, -1)


def stretch(a: np.ndarray) -> np.ndarray:
    valid = np.isfinite(a).all(-1) & (a.sum(-1) > 0)
    lo, hi = np.percentile(a[valid], [2, 98])
    return np.clip((a - lo) / (hi - lo), 0, 1)


def parcel_vote_map(pred: np.ndarray, pid: np.ndarray, n: int) -> np.ndarray:
    """Every declared parcel of the mosaic coloured by the crop most of its pixels were predicted as.

    Scored parcels vote with all their pixels, which are all labelled, exactly as the pipeline's vote; parcels of
    crops outside the 20 classes vote too, as any parcel of a new area would. -1 outside parcels.
    """
    sel = pid > 0
    ids, inv = np.unique(pid[sel], return_inverse=True)
    votes = np.bincount(inv * n + pred[sel].astype(np.int64), minlength=len(ids) * n).reshape(len(ids), n)
    out = np.full(pid.shape, -1, dtype=np.int64)
    out[sel] = votes.argmax(1)[inv]
    return out


def edges(seg: np.ndarray) -> np.ndarray:
    """Pixels on the border of a region of ``seg``."""
    e = np.zeros(seg.shape, bool)
    e[1:, :] |= seg[1:, :] != seg[:-1, :]
    e[:-1, :] |= seg[1:, :] != seg[:-1, :]
    e[:, 1:] |= seg[:, 1:] != seg[:, :-1]
    e[:, :-1] |= seg[:, 1:] != seg[:, :-1]
    return e


def paint(classes: np.ndarray, rgb: np.ndarray, pid: np.ndarray | None = None, wrong: np.ndarray | None = None,
          width: int = 1) -> np.ndarray:
    """RGB image of a class map: -1 as no parcel, -2 as another crop; parcel borders, and wrong parcels outlined
    ``width`` pixels deep inside them."""
    img = np.empty(classes.shape + (3,))
    img[classes == -1] = to_rgb(NO_PARCEL)
    img[classes == -2] = to_rgb(OTHER_CROP)
    ok = classes >= 0
    img[ok] = rgb[classes[ok]]
    if pid is not None:
        img[edges(pid) & (pid > 0)] = to_rgb(SURFACE)
    if wrong is not None:
        line = edges(np.where(wrong, pid, 0)) & wrong
        for _ in range(width - 1):
            grown = line.copy()
            grown[1:] |= line[:-1]
            grown[:-1] |= line[1:]
            grown[:, 1:] |= line[:, :-1]
            grown[:, :-1] |= line[:, 1:]
            line = grown & wrong
        img[line] = to_rgb(INK)
    return img


def reference(mask: np.ndarray, pid: np.ndarray) -> np.ndarray:
    """The declared crop under each pixel: -1 outside parcels, -2 inside a parcel of a crop not scored."""
    return np.where(mask >= 0, mask, np.where(pid > 0, -2, -1))


def legend(ax, rgb: np.ndarray, names: list[str], present: set[int], ncol: int) -> None:
    """The crops on show, grouped, in an axis of their own; a group never breaks across columns."""
    blocks = []
    for g, _, members in GROUPS:
        here = [m for m in members if names.index(m) in present]
        if here:
            blocks.append([Patch(color="none", label=g)]
                          + [Patch(color=rgb[names.index(m)], label=SHORT[m]) for m in here])
    blocks.append([Patch(color="none", label=" "), Patch(color=OTHER_CROP, label="declared, crop not scored"),
                   Patch(color=NO_PARCEL, label="no declared parcel"),
                   Patch(facecolor="white", edgecolor=INK, linewidth=1.5, label="parcel mapped wrong")])
    target = -(-sum(map(len, blocks)) // ncol)
    columns: list[list[Patch]] = [[]]
    for blk in blocks:
        if columns[-1] and len(columns[-1]) + len(blk) > target and len(columns) < ncol:
            columns.append([])
        columns[-1] += blk
    height = max(map(len, columns))
    # The legend fills column by column, so each column is padded to the same height.
    handles = [h for col in columns for h in col + [Patch(color="none", label="")] * (height - len(col))]
    ax.axis("off")
    leg = ax.legend(handles=handles, loc="center left", ncol=ncol, fontsize=9, frameon=False, handlelength=1.3,
                    columnspacing=1.6, labelspacing=0.45)
    for t, h in zip(leg.get_texts(), handles, strict=True):
        if h.get_facecolor()[3] == 0 and t.get_text().strip():
            t.set_fontweight("bold")
            t.set_color(INK_2)


def style(ax, title: str | None = None) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_color(MUTED)
        s.set_linewidth(0.5)
    if title:
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")


def block_scores(vote: np.ndarray, mask: np.ndarray, pid: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Share right of the block's labelled pixels, and of its scored parcels (vote and class are per parcel)."""
    lab = mask >= 0
    px = float((pred[lab] == mask[lab]).mean())
    sel = lab & (pid > 0)
    _, first = np.unique(pid[sel], return_index=True)
    return px, float((vote[sel][first] == mask[sel][first]).mean())


def lonlat(b: pd.DataFrame) -> tuple[float, float]:
    x = (b.col.min() + 2) * GRID_M
    y = (b.row.min() + 2) * GRID_M
    lon, lat = Transformer.from_crs(3035, 4326, always_xy=True).transform(x, y)
    return lon, lat


def fig_block(block: str, about: str, b: pd.DataFrame, fits: dict, parcels: dict, bands: list[str],
              rgb: np.ndarray, names: list[str]) -> Path:
    tc = stretch(mosaic(b, {c: true_colour(c, bands) for c in b.chip_id}))
    pid = mosaic(b, parcels)
    mask = mosaic(b, fits[(MODELS[0][0], KS[0])]["mask"], fill=-1)
    ref = reference(mask, pid)
    n = len(names)
    present = set(np.unique(mask[mask >= 0]).tolist())

    fig = plt.figure(figsize=(14, 15.2), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 4, left=0.04, right=0.99, top=0.915, bottom=0.01, hspace=0.12, wspace=0.03)
    lon, lat = lonlat(b)
    fig.text(0.04, 0.978, f"Test block {block}, 8.96 x 8.96 km, {lat:.2f} N {lon:.2f} E: {about}",
             fontsize=12.5, color=INK, fontweight="bold")
    fig.text(0.04, 0.948, "Never seen in training: 1.6 km from any training parcel. Raw: every 10 m pixel as "
             "predicted; there is no cropland mask yet, so forest and towns get a crop too.\nPer parcel: each "
             "declared parcel takes the crop most of its pixels were predicted as. Titles: share right of the "
             "labelled pixels, and of the scored parcels.", fontsize=9.5, color=INK_2, linespacing=1.5)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tc, interpolation="nearest")
    style(ax, "July 2021, true colour")
    ax = fig.add_subplot(gs[0, 1])
    ax.imshow(paint(ref, rgb, pid), interpolation="nearest")
    style(ax, "Declared crops (reference)")
    for i, (label, _) in enumerate(MODELS, start=1):
        for j, k in enumerate(KS):
            f = fits[(label, k)]
            pred = mosaic(b, f["pred"])
            vote = parcel_vote_map(pred, pid, n)
            scored = (mask >= 0) & (pid > 0)
            wrong = scored & (vote != mask)
            px, pa = block_scores(vote, mask, pid, pred)
            ax = fig.add_subplot(gs[i, 2 * j])
            ax.imshow(paint(pred, rgb), interpolation="nearest")
            style(ax, f"K = {k} %, raw   pixels {px:.2f}")
            if j == 0:
                ax.set_ylabel(label, fontsize=10.5, color=INK, fontweight="bold")
            ax = fig.add_subplot(gs[i, 2 * j + 1])
            ax.imshow(paint(np.where(pid > 0, vote, -1), rgb, pid, wrong=wrong, width=3),
                      interpolation="nearest")
            style(ax, f"K = {k} %, per parcel   parcels {pa:.2f}")
            present |= set(np.unique(pred).tolist())
    legend(fig.add_subplot(gs[0, 2:]), rgb, names, present, ncol=3)
    path = OUT / f"map_block_{block}.png"
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return path


def fig_zoom(split: pd.DataFrame, fits: dict, parcels: dict, bands: list[str], rgb: np.ndarray,
             names: list[str]) -> Path:
    n = len(names)
    k = KS[0]
    fig = plt.figure(figsize=(15.5, 3.45 * 2 * len(ZOOMS) + 0.6), facecolor=SURFACE)
    gs = fig.add_gridspec(2 * len(ZOOMS), 5, width_ratios=[1, 1, 1, 1, 0.78], left=0.01, right=0.995, top=0.95,
                          bottom=0.01, hspace=0.14, wspace=0.03)
    axes = [[fig.add_subplot(gs[r, c]) for c in range(4)] for r in range(2 * len(ZOOMS))]
    present: set[int] = set()
    for z, (block, (r0, c0), size) in enumerate(ZOOMS):
        b = block_chips(split, block)
        win = (slice(r0, r0 + size), slice(c0, c0 + size))
        tc = stretch(mosaic(b, {c: true_colour(c, bands) for c in b.chip_id}))[win]
        pid = mosaic(b, parcels)[win]
        mask = mosaic(b, fits[(MODELS[0][0], k)]["mask"], fill=-1)[win]
        top, bot = axes[2 * z], axes[2 * z + 1]
        top[0].imshow(tc, interpolation="nearest")
        style(top[0], f"Block {block}, {size * 10 / 1000:.2f} km: July")
        bot[0].imshow(paint(reference(mask, pid), rgb, pid), interpolation="nearest")
        style(bot[0], "Declared crops")
        present |= set(np.unique(mask[mask >= 0]).tolist())
        for j, (label, _) in enumerate(MODELS, start=1):
            pred = mosaic(b, fits[(label, k)]["pred"])[win]
            vote = parcel_vote_map(pred, pid, n)
            wrong = (mask >= 0) & (pid > 0) & (vote != mask)
            top[j].imshow(paint(pred, rgb), interpolation="nearest")
            style(top[j], f"{label}: raw")
            bot[j].imshow(paint(np.where(pid > 0, vote, -1), rgb, pid, wrong=wrong, width=1),
                          interpolation="nearest")
            style(bot[j], f"{label}: per parcel")
            present |= set(np.unique(pred).tolist())
    fig.text(0.01, 0.982, f"Two zooms of 1.12 x 1.12 km at full 10 m resolution, K = {k} %", fontsize=12.5,
             color=INK, fontweight="bold")
    legend(fig.add_subplot(gs[:, 4]), rgb, names, present, ncol=1)
    path = OUT / "map_zoom.png"
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((CHIPS / "manifest.json").read_text())
    names = [c["name"] for c in manifest["classes"]]
    bands = manifest["imagery"]["band_names"]
    rgb = class_colours(names)
    split = pd.read_csv(SPLIT)

    fits = {(label, k): load_fit(path, k) for label, path in MODELS for k in KS}
    ids = list(next(iter(fits.values()))["pred"])
    for f in fits.values():
        if f["classes"] != names or list(f["pred"]) != ids:
            raise SystemExit("fits disagree on the classes or the test tiles")
    parcels = {c: read_parcels(CHIPS / "chips", c) for c in ids}
    for (label, k), f in fits.items():
        print(f"{label}, K = {k} %: parcel accuracy {check_votes(f, parcels):.4f} reproduced")

    for block, about in BLOCKS:
        print(fig_block(block, about, block_chips(split, block), fits, parcels, bands, rgb, names))
    print(fig_zoom(split, fits, parcels, bands, rgb, names))


if __name__ == "__main__":
    main()
