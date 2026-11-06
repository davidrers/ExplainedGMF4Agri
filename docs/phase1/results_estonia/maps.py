"""Crop maps of held-out test areas, as the models would map an area they have never seen (REPORT.md section 3.5).

The test blocks of the split are areas the models never saw, 1.6 km away from any training parcel, and every fit
predicted every 10 m pixel of every test tile, forest, roads and towns included. This script reads those stored
predictions (``predictions_test.npz``) of all ten fits of the report (the six frozen arms and the baseline of
``results/kshot/EE_2021_rev/``, the four fine-tuned arms of ``results/2026-10-15_kshot_finetune/EE_2021_rev/``),
refits nothing, and writes into ``figures/``:

* ``map_<place>_K<k>.png``, for three whole test blocks of 4 x 4 tiles (8.96 km) and two single tiles (2.24 km, at
  full resolution) at K = 100, 20 and 5 %: the July true colour and the declared crops, then every model in a
  column. Rows: the raw map (every pixel as the model output it; nothing is masked, and there is no cropland mask
  yet) and the per-parcel map (each declared parcel takes the crop most of its pixels were predicted as; scored
  parcels mapped wrong are outlined), for the frozen encoder, then the fine-tuned one;

and ``maps_tables.md`` beside this file: the share of each place's scored parcels mapped right, and what each fit,
and the frozen TESSERA control of the fine-tuning experiment, predicts outside the declared parcels over all test
tiles.

The parcel vote is the pipeline's (:mod:`gfm4agri.benchmark.parcel_metrics`). Before drawing, the script reproduces
each fit's parcel accuracy over all test tiles from the same votes and stops if it differs from ``results.json``.

    poetry run python docs/phase1/results_estonia/maps.py
"""

from __future__ import annotations

import gc
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
from PIL import Image  # noqa: E402
from pyproj import Transformer  # noqa: E402

from gfm4agri.benchmark.parcel_metrics import chip_votes, pool_votes, read_parcels  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
CHIPS = REPO / "data/eurocrops_chips/EE_2021_rev"
SPLIT = REPO / "data/eurocrops_chips/EE_2021/splits/blocks4_buf1600_seed0__6b0eb4cb/chips.csv"
FROZEN_DIR = REPO / "results/kshot/EE_2021_rev"
FT_DIR = REPO / "results/2026-10-15_kshot_finetune/EE_2021_rev"
OUT = HERE / "figures"
TILE, GRID_M = 224, 2240
KS = [100, 20, 5]

#: One column per model: label, frozen fit, fine-tuned fit (None: not run). The U-Net has no pretrained encoder:
#: it is trained end to end and sits in the first two rows.
MODELS = [
    ("TerraMind v1 L", FROZEN_DIR / "terramind_v1_large_s2s1", FT_DIR / "terramind_v1_large_s2s1_ft"),
    ("THOR v1 L", FROZEN_DIR / "thor_v1_large_s2s1", FT_DIR / "thor_v1_large_s2s1_ft"),
    ("Prithvi-EO-2.0 600M", FROZEN_DIR / "prithvi_eo_v2_600_tl", FT_DIR / "prithvi_eo_v2_600_tl_ft"),
    ("TESSERA v2 M", FROZEN_DIR / "tessera_v2_medium", FT_DIR / "tessera_v2_medium_ft"),
    ("AlphaEarth v1", FROZEN_DIR / "alphaearth_v1", None),
    ("U-Net from scratch", FROZEN_DIR / "unet_r18_scratch_s2s1", None),
]
#: Not drawn, only in the table of off-parcel predictions: the fine-tuning experiment's frozen control, TESSERA's
#: MLP trained on the labelled pixels alone, the route the fine-tuned TESSERA takes.
CONTROL = ("TESSERA v2 M, frozen, labelled-pixel route (control)", FT_DIR / "tessera_v2_medium_lab")
NO_FT = {"AlphaEarth v1": "AlphaEarth cannot be fine-tuned:\nonly its embeddings are published",
         "U-Net from scratch": "No pretrained encoder: the U-Net\nis trained end to end (rows above)"}
ROWS = ["Frozen encoder: raw", "Frozen encoder: per parcel", "Fine-tuned: raw", "Fine-tuned: per parcel"]

#: Complete test blocks, picked from the 40 of the split for their landscape (profile of 6 November 2026).
BLOCKS = [
    ("583_457", "large arable fields: median parcel 6.2 ha, 52 % of the block declared"),
    ("593_445", "small plots: 32 potato parcels, 8 % of parcels under 0.16 ha"),
    ("594_443", "mostly forest and settlements: 11 % of the block declared, median parcel 1 ha"),
]
#: Single tiles: block, tile row from the top and tile column from the west. They hold the windows with the most
#: crops besides grassland (583_457) and the most potato parcels (593_445) of a 1.12 km scan.
TILES = [("583_457", 1, 0, "large fields beside a forest"), ("593_445", 3, 0, "narrow strips of potatoes and barley")]

# Crop groups: one hue each (the reference palette of the dataviz method), crops within a group by lightness.
# Grassland, half of all parcels, is a mid grey: recessive beside the arable crops, yet clearly a prediction and
# never the near-white of land without a declared parcel. Parcels of crops outside the 20 are hatched.
SURFACE, INK, INK_2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
NO_PARCEL, HATCH = "#f0efec", "#a9a79e"
GROUPS = [
    ("Grassland", "#a9a79e", ["pasture_meadow_grassland_grass"]),
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


def off_parcel(fit: dict, parcels: dict) -> float:
    """Share of the test pixels outside any declared parcel that the fit gives a crop other than grassland."""
    crop = total = 0
    for c, p in fit["pred"].items():
        off = parcels[c] == 0
        crop += int((p[off] != 0).sum())
        total += int(off.sum())
    return crop / total


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
    """Every declared parcel coloured by the crop most of its pixels were predicted as; -1 outside parcels.

    Scored parcels vote with all their pixels, which are all labelled, exactly as the pipeline's vote; parcels of
    crops outside the 20 classes vote too, as any parcel of a new area would.
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
    """RGB image of a class map: -1 as no parcel, -2 hatched (a crop not scored); parcel borders, and wrong parcels
    outlined ``width`` pixels deep inside them."""
    img = np.empty(classes.shape + (3,))
    img[classes == -1] = to_rgb(NO_PARCEL)
    yy, xx = np.indices(classes.shape)
    other = classes == -2
    img[other] = to_rgb(NO_PARCEL)
    img[other & ((yy + xx) % 6 < 2)] = to_rgb(HATCH)
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


def legend(ax, rgb: np.ndarray, names: list[str], ncol: int) -> None:
    """Every crop, grouped, in an axis of their own; a group never breaks across columns."""
    blocks = [[Patch(color="none", label=g)] + [Patch(color=rgb[names.index(m)], label=SHORT[m]) for m in members]
              for g, _, members in GROUPS]
    blocks.append([Patch(color="none", label=" "),
                   Patch(facecolor=NO_PARCEL, edgecolor=HATCH, hatch="////", label="declared, crop not scored"),
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
    leg = ax.legend(handles=handles, loc="center", ncol=ncol, fontsize=9.5, frameon=False, handlelength=1.3,
                    columnspacing=1.2, labelspacing=0.32)
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
        ax.set_title(title, fontsize=10, color=INK, loc="left", pad=3)


def scores(vote: np.ndarray, mask: np.ndarray, pid: np.ndarray, pred: np.ndarray) -> tuple[float, float]:
    """Share right of the labelled pixels, and of the scored parcels (vote and class are per parcel)."""
    lab = mask >= 0
    px = float((pred[lab] == mask[lab]).mean())
    sel = lab & (pid > 0)
    _, first = np.unique(pid[sel], return_index=True)
    return px, float((vote[sel][first] == mask[sel][first]).mean())


def lonlat(x: float, y: float) -> tuple[float, float]:
    return Transformer.from_crs(3035, 4326, always_xy=True).transform(x, y)


def places(split: pd.DataFrame) -> list[dict]:
    """The blocks and single tiles to map: their tiles, the window of the block mosaic, and a heading."""
    out = []
    for block, about in BLOCKS:
        b = block_chips(split, block)
        lon, lat = lonlat((b.col.min() + 2) * GRID_M, (b.row.min() + 2) * GRID_M)
        out.append({"name": f"block_{block}", "tiles": b, "win": (slice(None), slice(None)), "width": 4,
                    "head": f"Test block {block}, 8.96 x 8.96 km, {lat:.2f} N {lon:.2f} E: {about}"})
    for block, r, c, about in TILES:
        b = block_chips(split, block)
        row, col = b.row.max() - r, b.col.min() + c
        cid = b[(b.row == row) & (b.col == col)].chip_id.item()
        lon, lat = lonlat((col + 0.5) * GRID_M, (row + 0.5) * GRID_M)
        out.append({"name": f"tile_{cid}", "tiles": b, "width": 1,
                    "win": (slice(r * TILE, (r + 1) * TILE), slice(c * TILE, (c + 1) * TILE)),
                    "head": f"Test tile {cid} (block {block}), 2.24 x 2.24 km at full 10 m resolution, "
                            f"{lat:.2f} N {lon:.2f} E: {about}"})
    return out


def fig_place(pl: dict, k: int, fits: dict, parcels: dict, bands: list[str], rgb: np.ndarray,
              names: list[str]) -> tuple[Path, dict]:
    """One place at one budget: reference column, then a column per model; rows as in ``ROWS``."""
    b, win, n = pl["tiles"], pl["win"], len(names)
    tc = stretch(mosaic(b, {c: true_colour(c, bands) for c in b.chip_id})[win])
    pid = mosaic(b, parcels)[win]
    mask = mosaic(b, next(iter(fits.values()))["mask"], fill=-1)[win]
    width = 4 if pl["width"] == 4 else 1

    fig = plt.figure(figsize=(23, 14.6), facecolor=SURFACE)
    gs = fig.add_gridspec(4, 1 + len(MODELS), left=0.035, right=0.997, top=0.875, bottom=0.005, hspace=0.1,
                          wspace=0.03)
    fig.text(0.035, 0.972, pl["head"] + f"; K = {k} % of the training parcels labelled", fontsize=14,
             color=INK, fontweight="bold")
    fig.text(0.035, 0.925, "Never seen in training: 1.6 km from any training parcel. Raw: every 10 m pixel exactly as "
             "the model output it; nothing is masked, and there is no cropland mask yet, so forest, roads and towns get "
             "a crop too.\nPer parcel: each declared parcel takes the crop most of its pixels were predicted as. "
             "Titles: share right of the labelled pixels, and of the scored parcels.", fontsize=11, color=INK_2,
             linespacing=1.5)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(tc, interpolation="nearest")
    style(ax, "July 2021, true colour")
    ax = fig.add_subplot(gs[1, 0])
    ax.imshow(paint(reference(mask, pid), rgb, pid), interpolation="nearest")
    style(ax, "Declared crops (reference)")
    legend(fig.add_subplot(gs[2:, 0]), rgb, names, ncol=1)

    shares = {}
    for j, (label, frozen, ft) in enumerate(MODELS, start=1):
        for i, (variant, path) in enumerate([("frozen", frozen), ("fine-tuned", ft)]):
            if path is None:
                ax = fig.add_subplot(gs[2 * i:2 * i + 2, j])
                ax.axis("off")
                ax.text(0.5, 0.5, NO_FT[label], ha="center", va="center", fontsize=10, color=MUTED)
                continue
            pred = mosaic(b, fits[path]["pred"])[win]
            vote = parcel_vote_map(pred, pid, n)
            wrong = (mask >= 0) & (pid > 0) & (vote != mask)
            px, pa = scores(vote, mask, pid, pred)
            shares[(label, variant)] = pa
            ax = fig.add_subplot(gs[2 * i, j])
            ax.imshow(paint(pred, rgb), interpolation="nearest")
            style(ax, f"raw   pixels {px:.2f}")
            if i == 0:
                ax.text(0, 1.13, label, transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=INK)
            ax = fig.add_subplot(gs[2 * i + 1, j])
            ax.imshow(paint(np.where(pid > 0, vote, -1), rgb, pid, wrong=wrong, width=width),
                      interpolation="nearest")
            style(ax, f"per parcel   parcels {pa:.2f}")
    for i, row in enumerate(ROWS):
        y = gs[i, 0].get_position(fig)
        fig.text(0.012, (y.y0 + y.y1) / 2, row, rotation=90, ha="center", va="center", fontsize=11.5,
                 fontweight="bold", color=INK)
    path = OUT / f"map_{pl['name']}_K{k}.png"
    save(fig, path, 150 if pl["width"] == 4 else 110, rgb, tc)
    return path, shares


def save(fig, path: Path, dpi: int, rgb: np.ndarray, tc: np.ndarray) -> None:
    """A PNG on a palette that holds every crop colour and the chrome exactly, a grey ramp for the lettering, and
    the true colour in the remaining slots: a third of the size of full colour, with no crop colour changed."""
    fig.savefig(path, dpi=dpi, facecolor=SURFACE)
    plt.close(fig)
    fixed = [tuple(int(round(v * 255)) for v in c) for c in rgb]
    fixed += [tuple(int(round(v * 255)) for v in to_rgb(c)) for c in (NO_PARCEL, HATCH, SURFACE, INK, INK_2, MUTED)]
    fixed += [(g, g, g) for g in np.linspace(0, 255, 18).astype(int)]
    free = 256 - len(fixed)
    tc_pal = Image.fromarray((tc * 255).astype(np.uint8)).quantize(free, method=Image.Quantize.MEDIANCUT)
    pal = Image.new("P", (1, 1))
    pal.putpalette([v for c in fixed for v in c] + tc_pal.getpalette()[:3 * free])
    Image.open(path).convert("RGB").quantize(palette=pal, dither=Image.Dither.NONE).save(path, optimize=True)


def tables(shares: dict, off: dict, pls: list[dict]) -> str:
    lines = ["# Tables of the maps of REPORT.md section 3.5", "", "Written by [maps.py](maps.py). One draw, one seed "
             "per fit; the shares of a place come from its parcels alone and illustrate the maps, they are not "
             "test scores.", "", "## Share of the scored parcels mapped right, K = 100 / 20 / 5 %", ""]
    head = "| Model | " + " | ".join(p["name"].replace("_", " ") for p in pls) + " |"
    lines += [head, "|---" * (1 + len(pls)) + "|"]
    for label, frozen, ft in MODELS:
        for variant, path in [("frozen", frozen), ("fine-tuned", ft)]:
            if path is None:
                continue
            name = label if label == "U-Net from scratch" else f"{label}, {variant}"
            cells = [" / ".join(f"{shares[(p['name'], k)][(label, variant)]:.2f}" for k in KS) for p in pls]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
    lines += ["", "## Test pixels outside any declared parcel given a crop other than grassland, all test tiles", "",
              "| Model | K = 100 % | K = 20 % | K = 5 % |", "|---|---|---|---|"]
    for label, frozen, ft in MODELS:
        for variant, path in [("frozen", frozen), ("fine-tuned", ft)]:
            if path is None:
                continue
            name = label if label == "U-Net from scratch" else f"{label}, {variant}"
            lines.append(f"| {name} | " + " | ".join(f"{off[(path, k)]:.3f}" for k in KS) + " |")
    lines.append(f"| {CONTROL[0]} | " + " | ".join(f"{off[(CONTROL[1], k)]:.3f}" for k in KS) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    OUT.mkdir(exist_ok=True)
    manifest = json.loads((CHIPS / "manifest.json").read_text())
    names = [c["name"] for c in manifest["classes"]]
    bands = manifest["imagery"]["band_names"]
    rgb = class_colours(names)
    split = pd.read_csv(SPLIT)
    pls = places(split)
    paths = [p for _, frozen, ft in MODELS for p in (frozen, ft) if p is not None]

    parcels, ids, shares, off = None, None, {}, {}
    for k in KS:
        fits = {p: load_fit(p, k) for p in paths}
        if ids is None:
            ids = list(next(iter(fits.values()))["pred"])
            parcels = {c: read_parcels(CHIPS / "chips", c) for c in ids}
        for p, f in fits.items():
            if f["classes"] != names or list(f["pred"]) != ids:
                raise SystemExit(f"{p}: the classes or the test tiles differ from the other fits")
            acc = check_votes(f, parcels)
            off[(p, k)] = off_parcel(f, parcels)
            print(f"{p.name}, K = {k} %: parcel accuracy {acc:.4f} reproduced; "
                  f"off-parcel crop share {off[(p, k)]:.3f}")
        for pl in pls:
            path, shares[(pl["name"], k)] = fig_place(pl, k, fits, parcels, bands, rgb, names)
            print(path)
        off[(CONTROL[1], k)] = off_parcel(load_fit(CONTROL[1], k), parcels)
        del fits
        gc.collect()
    (HERE / "maps_tables.md").write_text(tables(shares, off, pls))
    print(HERE / "maps_tables.md")


if __name__ == "__main__":
    main()
