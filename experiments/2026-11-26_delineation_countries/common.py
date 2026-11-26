"""What every script of this experiment shares: the sample chip set, the bands and months, the output layout.

Only numpy and rasterio, so that the separate environments (Delineate Anything, ftw-tools) import it too. The
scripts copied from ``experiments/2026-11-26_open_delineation`` read the same names from here, so they run unchanged
on the sample; nothing of the core is changed.

The chip set is ``EU18_2021_delin`` (``sample.py``): six test blocks of each of the 18 countries of the transfer
samples other than Estonia (``data/eurocrops_chips/<CC>_2021_transfer``), linked, not copied. Its chips are those of
the transfer export: twelve monthly Sentinel-2 L2A composites under the ``revised`` screening, Sentinel-1 RTC beside
them, the transfer class list (21 species, 9 groups) in the masks, 2021, the same 2,240 m grid in EPSG:3035 as
Estonia. Every chip is a test chip; no model is trained here.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAME = HERE.name                                           # 2026-11-26_delineation_countries
CHIPSET = "EU18_2021_delin"
CHIPS = ROOT / "data" / "eurocrops_chips" / CHIPSET
CHIP_DIR = CHIPS / "chips"
SPLIT = CHIPS / "splits" / "sample"
WORLDCOVER = ROOT / "data" / "worldcover" / CHIPSET / "worldcover.npz"
WEIGHTS = ROOT / "data" / "external_models"
RESULTS = ROOT / "results" / NAME / CHIPSET
#: Our arms of the two-head workflow, trained on Estonia 2021 (K = 100 %), whose checkpoints are applied here.
ARMS_RESULTS = ROOT / "results" / "2026-11-09_borders_all_arms" / "EE_2021_rev"
#: The Estonian comparison this experiment repeats elsewhere.
ESTONIA = ROOT / "experiments" / "2026-11-26_open_delineation"

COUNTRIES = ("AT", "BE", "BG", "CZ", "DE", "DK", "ES", "FI", "FR", "IE", "IT", "LT", "LV", "NL", "PT", "SE", "SI",
             "SK")
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
#: The twelve L2A bands of a month, in the chips' order (as in EE_2021_rev).
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
B = {b: i for i, b in enumerate(BANDS)}
RGB = [B["B04"], B["B03"], B["B02"]]
RGBN = [B["B04"], B["B03"], B["B02"], B["B08"]]
PASTIS10 = [B[b] for b in ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")]
SEASON = (3, 4, 5, 6, 7, 8)


def country(cid: str) -> str:
    return cid[:2]


def split_ids(split: str) -> list[str]:
    """Every chip of the sample is a test chip; ``val`` is accepted and gives the same list (no choice is made on
    this sample: the variants of every model are those chosen on the Estonian validation chips)."""
    return [line.strip() for line in open(SPLIT / "test_data.txt") if line.strip()]


def read_s2(cid: str) -> np.ndarray:
    """``(12 months, 12 bands, 224, 224)`` int16 of one chip."""
    import rasterio

    with rasterio.open(CHIP_DIR / f"{cid}_merged.tif") as src:
        a = src.read()
    return a.reshape(12, 12, *a.shape[1:])


def read_mask(cid: str) -> np.ndarray:
    import rasterio

    with rasterio.open(CHIP_DIR / f"{cid}.mask.tif") as src:
        return src.read(1)


def _load(part: list[str]):
    return np.stack([read_s2(c) for c in part]), np.stack([read_mask(c) for c in part])


def chips(ids: list[str], batch: int, workers: int = 8):
    """Batches ``(ids, s2 (b, 12, 12, H, W) int16, masks (b, H, W) int16)`` in order, read ahead by a thread pool."""
    parts = [ids[s:s + batch] for s in range(0, len(ids), batch)]
    with ThreadPoolExecutor(workers) as ex:
        futs = {i: ex.submit(_load, parts[i]) for i in range(min(workers, len(parts)))}
        for i, part in enumerate(parts):
            s2, masks = futs.pop(i).result()
            if i + workers < len(parts):
                futs[i + workers] = ex.submit(_load, parts[i + workers])
            yield part, s2, masks


def out_dir(model: str, variant: str) -> Path:
    d = RESULTS / model / variant
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_prob(model: str, variant: str, split: str, ids, prob, masks, meta: dict) -> Path:
    """Three-class probabilities x 255 (background, interior, boundary), as the two-head workflow stores them."""
    d = out_dir(model, variant)
    p = d / f"predictions_{split}.npz"
    np.savez_compressed(p, ids=np.asarray(ids), prob=np.asarray(prob, np.uint8), mask=np.asarray(masks, np.int16))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return p


def save_instances(model: str, variant: str, split: str, ids, inst, masks, meta: dict, conf=None) -> Path:
    """Field ids per pixel (0 = no field) and, per chip, the confidence of each id (flat, with offsets)."""
    d = out_dir(model, variant)
    p = d / f"predictions_{split}.npz"
    extra = {}
    if conf is not None:
        extra = {"conf": np.concatenate([np.asarray(c, np.float32) for c in conf]),
                 "conf_offsets": np.cumsum([0] + [len(c) for c in conf]).astype(np.int64)}
    np.savez_compressed(p, ids=np.asarray(ids), inst=np.asarray(inst, np.int32), mask=np.asarray(masks, np.int16),
                        **extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return p
