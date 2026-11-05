"""Chips for U-Net + PaPs: the TESSERA raster, normalised, and the PaPs target of ``targets.py``.

One item per chip: ``image`` ``(128, H, W)`` float32 (raw NaN -> 0, then the raster's training-chip mean and standard
deviation per band, as the core's datamodule does for an embedding raster), ``target`` ``(H, W, 8)``, ``chip``. The
training chips carry the drawn parcels only and get a random D4 transform (rotation and flip, exact for a per-pixel
embedding), applied to the raster and the label rasters before the target is built, so boxes and kernels follow
the turned parcels. The training chips are those holding at least one drawn parcel, as in the core's datamodule
under a budget; validation and test chips keep every parcel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import targets


class PapsChips(Dataset):
    def __init__(self, chip_ids, raster_dir: Path, suffix: str, means, stds, labels_dir: Path, worldcover: Path,
                 keep: np.ndarray | None = None, augment: bool = False) -> None:
        self.ids = list(chip_ids)
        self.raster_dir, self.suffix, self.labels_dir = Path(raster_dir), suffix, Path(labels_dir)
        self.means = np.asarray(means, np.float32)[:, None, None]
        self.stds = np.asarray(stds, np.float32)[:, None, None]
        self.wc_path = Path(worldcover)
        self._wc = None
        self.keep = None if keep is None else np.asarray(keep, np.int64)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.ids)

    def worldcover(self, cid: str) -> np.ndarray:
        if self._wc is None:                 # opened lazily, once per worker
            self._wc = np.load(self.wc_path)
        return self._wc[cid]

    def labels(self, cid: str) -> tuple[np.ndarray, np.ndarray]:
        import rasterio

        with rasterio.open(self.labels_dir / f"{cid}.mask.tif") as m, \
             rasterio.open(self.labels_dir / f"{cid}.parcels.tif") as p:
            return m.read(1).astype(np.int64), p.read(1).astype(np.int64)

    def image(self, cid: str) -> np.ndarray:
        import rasterio

        with rasterio.open(self.raster_dir / "chips" / f"{cid}{self.suffix}") as s:
            x = s.read().astype(np.float32)
        x = np.nan_to_num(x, nan=0.0)
        return (x - self.means) / self.stds

    def __getitem__(self, i: int) -> dict:
        cid = self.ids[i]
        x = self.image(cid)
        mask, parcels = self.labels(cid)
        wc = self.worldcover(cid)
        if self.augment:
            # torch's generator, seeded per worker and per epoch by the loader from the fit's seed.
            k, flip = int(torch.randint(4, ()).item()), bool(torch.randint(2, ()).item())
            x, mask, parcels, wc = targets.d4([x, mask, parcels, wc], k, flip)
        y = targets.build(mask, parcels, wc, self.keep)
        return {"image": torch.from_numpy(x), "target": torch.from_numpy(y), "chip": cid}


def collate(items: list[dict]) -> dict:
    return {"image": torch.stack([it["image"] for it in items]), "target": torch.stack([it["target"] for it in items]),
            "chip": [it["chip"] for it in items]}


def train_chips(root: Path, ids: list[str], keep: np.ndarray) -> list[str]:
    """The training chips holding a drawn parcel, as the core's datamodule selects them under a budget."""
    from gfm4agri.benchmark.segmentation_data import chips_with_parcels

    have = set(chips_with_parcels(root, ids, keep))
    return [c for c in ids if c in have]
