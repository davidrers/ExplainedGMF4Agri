"""Make the two images of the site's landing: a Sentinel-2 true-colour mosaic of Estonian fields and its crop labels.

Picks the 6 x 3 window of whole chips of `EE_2021_rev` with the largest labelled share, then writes, at the chips' own
10 m resolution:

* `docs/assets/hero_s2.jpg`: the June 2021 composite in true colour (B04, B03, B02), one stretch for the whole window;
* `docs/assets/hero_labels.png`: the EuroCrops 2021 class of every labelled pixel, coloured by crop group, transparent
  where a pixel carries no label.

Run on the hub, where the chips are (needs rasterio):

    poetry run python docs/phase1/pipeline_page/make_hero.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
CHIPS = ROOT / "data/eurocrops_chips/EE_2021_rev"
OUT = ROOT / "docs/assets"
COLS, ROWS = 6, 3
MONTH = 5            # June, 0-based
PX = 224

# Crop groups of the 20 Estonian classes and their colours, muted to sit on the landing's green.
GROUPS = {
    "grassland": ("#9cb87a", ["pasture_meadow_grassland_grass"]),
    "winter cereals": ("#d9a441", ["winter_common_soft_wheat", "winter_barley", "rye"]),
    "spring cereals": ("#ecd38a", ["spring_barley", "oats", "spring_common_soft_wheat", "buckwheat"]),
    "rapeseed": ("#f2d42c", ["winter_rapeseed_rape", "spring_rapeseed_rape"]),
    "legumes and fodder": ("#5fa8a0", ["legumes_harvested_green", "clover", "peas", "beans", "alfalfa_lucerne",
                                       "legumes_dried_pulses_protein_crops"]),
    "maize": ("#e07b39", ["grain_maize_corn_popcorn"]),
    "potatoes and vegetables": ("#b5654a", ["potatoes", "fresh_vegetables"]),
    "orchards": ("#3f6b45", ["orchards_fruits"]),
}


def best_window(chips: list[dict]) -> tuple[int, int]:
    share = {(c["col"], c["row"]): c["labelled_share"] for c in chips}
    best, at = -1.0, None
    for col, row in share:
        cells = [(col + i, row + j) for i in range(COLS) for j in range(ROWS)]
        if all(c in share for c in cells):
            s = sum(share[c] for c in cells)
            if s > best:
                best, at = s, (col, row)
    return at


def main() -> None:
    manifest = json.loads((CHIPS / "manifest.json").read_text())
    names = [c["name"] for c in sorted(manifest["classes"], key=lambda c: c["index"])]
    colour = np.zeros((len(names), 4), dtype=np.uint8)
    for hexc, members in GROUPS.values():
        for m in members:
            colour[names.index(m)] = [int(hexc[i:i + 2], 16) for i in (1, 3, 5)] + [255]
    col0, row0 = best_window(manifest["chips"])
    rgb = np.zeros((3, ROWS * PX, COLS * PX), dtype=np.float32)
    lab = np.full((ROWS * PX, COLS * PX), -1, dtype=np.int16)
    for i in range(COLS):
        for j in range(ROWS):
            # EPSG:3035 rows grow northwards, so the northernmost chip row is drawn first.
            chip = f"EE_{col0 + i:05d}_{row0 + ROWS - 1 - j:05d}"
            ys, xs = slice(j * PX, (j + 1) * PX), slice(i * PX, (i + 1) * PX)
            with rasterio.open(CHIPS / "chips" / f"{chip}_merged.tif") as src:
                rgb[:, ys, xs] = src.read([MONTH * 12 + b + 1 for b in (3, 2, 1)]).astype(np.float32)
            with rasterio.open(CHIPS / "chips" / f"{chip}.mask.tif") as src:
                lab[ys, xs] = src.read(1)
    lo, hi = np.percentile(rgb, [1, 99.5])
    img = np.clip((rgb - lo) / (hi - lo), 0, 1) ** 0.8
    OUT.mkdir(exist_ok=True)
    Image.fromarray((img.transpose(1, 2, 0) * 255).astype(np.uint8)).save(OUT / "hero_s2.jpg", quality=86,
                                                                           optimize=True, progressive=True)
    rgba = np.zeros((*lab.shape, 4), dtype=np.uint8)
    rgba[lab >= 0] = colour[lab[lab >= 0]]
    Image.fromarray(rgba).save(OUT / "hero_labels.png", optimize=True)
    print(f"window EE_{col0:05d}_{row0:05d} + {COLS} x {ROWS} chips; labelled {np.mean(lab >= 0):.0%}")


if __name__ == "__main__":
    main()
