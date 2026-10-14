"""Shared paths and the mapping of our monthly composites onto TESSERA's input conventions.

TESSERA's own code is read from a clone of https://github.com/ucam-eo/tessera pinned at ``TESSERA_COMMIT`` in
``data/external/tessera``; its weights are in ``data/weights/tessera/`` (``SHA256SUMS`` beside them). Both are
git-ignored; README.md says how to recreate them.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
TESSERA = REPO / "data/external/tessera"
TESSERA_COMMIT = "e368539f7c4711e5f16fdd09801573830e34baf9"
WEIGHTS = REPO / "data/weights/tessera"
CHIPSET = REPO / "data/eurocrops_chips/EE_2021_rev_mini"
OUT = REPO / "results/2026-10-14_tessera_composites/spike"
FIGURES = HERE.parent / "figures"

#: Our twelve S2 bands are in wavelength order (B01 ... B12); TESSERA's ten are B04 B02 B03 B08 B8A B05 B06 B07
#: B11 B12. These are their positions in ours.
S2_IDX = [3, 1, 2, 7, 8, 4, 5, 6, 10, 11]
#: Day of year of each monthly composite: the 15th of the month, 2021.
DOY = np.array([15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349], dtype=np.float32)


def chip_ids() -> list[str]:
    return sorted(p.name[:-len("_merged.tif")] for p in (CHIPSET / "chips").glob("*_merged.tif"))


def composite_inputs(chip_id: str) -> dict[str, np.ndarray]:
    """One chip's composites in TESSERA's raw units, per pixel: (P, 12, C) arrays and (P, 12) validity.

    S2: DN = reflectance x 10000 with the BOA offset removed, which is TESSERA's convention too; every month is
    valid, since our composites are gap filled. S1: our 10 log10 of the linear-power median, in dB, becomes
    TESSERA's int16 encoding (20 log10(power) + 50) x 200 = (2 dB + 50) x 200, clipped to [0, 32767], with 0 where
    the month was never observed (TESSERA's missing value). Our composites merge the two orbit directions, so
    the 12 frames are handed over as the ascending stream.
    """
    base = CHIPSET / "chips" / chip_id
    with rasterio.open(f"{base}_merged.tif") as s:
        s2 = s.read().astype(np.float32).reshape(12, 12, 224, 224)[:, S2_IDX]
    with rasterio.open(f"{base}_s1rtc.tif") as s:
        db = s.read().astype(np.float32).reshape(12, 2, 224, 224)
    s1_valid = np.isfinite(db).all(1)
    s1 = np.where(s1_valid[:, None], np.clip((2.0 * db + 50.0) * 200.0, 0, 32767), 0.0).astype(np.float32)
    s2_valid = (s2 != 0).any(1)
    return {
        "s2": s2.transpose(2, 3, 0, 1).reshape(-1, 12, 10),
        "s2_valid": s2_valid.reshape(12, -1).T,
        "s1": s1.transpose(2, 3, 0, 1).reshape(-1, 12, 2),
        "s1_valid": s1_valid.reshape(12, -1).T,
    }


def tessera_path(*parts: str) -> None:
    """Put a folder of the TESSERA clone first on sys.path."""
    p = str(TESSERA.joinpath(*parts))
    if p not in sys.path:
        sys.path.insert(0, p)
