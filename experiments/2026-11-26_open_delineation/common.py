"""What every script of this experiment shares: the chips, their splits, the bands and months, the output layout.

Only numpy and rasterio, so that the separate environment of Delineate Anything (``env.sh``) imports it too. Nothing
of the core is changed; the scorer (``score.py``) imports the core's functions read-only.

The chips are those of the main workflow, ``EE_2021_rev``: twelve monthly Sentinel-2 L2A composites of twelve bands,
time-major (``band index = month * 12 + band``), surface reflectance x 10,000 with the BOA offset removed
(pipeline.md section 5). The split is the main workflow's (``blocks4_buf1600_seed0__6b0eb4cb``); no model of this
experiment is trained, so the training chips are never read.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
NAME = HERE.name                                           # 2026-11-26_open_delineation
CHIPSET = "EE_2021_rev"
CHIPS = ROOT / "data" / "eurocrops_chips" / CHIPSET
CHIP_DIR = CHIPS / "chips"
SPLIT = CHIPS / "splits" / "blocks4_buf1600_seed0__6b0eb4cb"
WORLDCOVER = ROOT / "data" / "worldcover" / CHIPSET / "worldcover.npz"
WEIGHTS = ROOT / "data" / "external_models"
RESULTS = ROOT / "results" / NAME / CHIPSET
#: Our arms of the two-head workflow, whose stored test predictions are rescored here on the same chips.
ARMS_RESULTS = ROOT / "results" / "2026-11-09_borders_all_arms" / CHIPSET

MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
#: The twelve L2A bands of a month, in the chips' order (manifest ``imagery.band_names``).
BANDS = ("B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B11", "B12")
B = {b: i for i, b in enumerate(BANDS)}
RGB = [B["B04"], B["B03"], B["B02"]]
RGBN = [B["B04"], B["B03"], B["B02"], B["B08"]]          # FTW's four channels per window
PASTIS10 = [B[b] for b in ("B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B11", "B12")]
#: The growing season scanned for single dates and pairs: April to September.
SEASON = (3, 4, 5, 6, 7, 8)


def split_ids(split: str) -> list[str]:
    name = {"val": "validation_data.txt", "test": "test_data.txt"}[split]
    return [line.strip() for line in open(SPLIT / name) if line.strip()]


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
    """Batches ``(ids, s2 (b, 12, 12, H, W) int16, masks (b, H, W) int16)`` in order, ``workers`` batches read
    ahead by a thread pool."""
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
    """Three-class probabilities x 255 (background, interior, boundary: ``gfm4agri.borders.prue.CLASSES``), as the
    two-head workflow's ``predictions_test.npz`` stores them, so the same fields and borders are built from them."""
    d = out_dir(model, variant)
    p = d / f"predictions_{split}.npz"
    np.savez_compressed(p, ids=np.asarray(ids), prob=np.asarray(prob, np.uint8), mask=np.asarray(masks, np.int16))
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return p


def save_instances(model: str, variant: str, split: str, ids, inst, masks, meta: dict, conf=None) -> Path:
    """Field ids per pixel (0 = no field), the model's own instances; ``conf``: per chip, the confidence of each id
    (index 0 unused), so that a higher threshold is applied later by removing ids."""
    d = out_dir(model, variant)
    p = d / f"predictions_{split}.npz"
    extra = {}
    if conf is not None:                       # flat, with offsets: chip i's ids 0.. are conf[off[i]:off[i + 1]]
        extra = {"conf": np.concatenate([np.asarray(c, np.float32) for c in conf]),
                 "conf_offsets": np.cumsum([0] + [len(c) for c in conf]).astype(np.int64)}
    np.savez_compressed(p, ids=np.asarray(ids), inst=np.asarray(inst, np.int32), mask=np.asarray(masks, np.int16),
                        **extra)
    (d / "meta.json").write_text(json.dumps(meta, indent=1))
    return p
