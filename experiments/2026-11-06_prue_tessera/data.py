"""Chips for PRUE on TESSERA: the frozen raster, or the composites the trainable encoder reads, and the target.

Two datasets, one item per chip, the same geometry for both:

* :class:`RasterChips` (``encoder: frozen``): the cached TESSERA v2 Medium raster ``(128, H, W)``, NaN to 0 and
  standardised with the cache's training-chip statistics, as the core's datamodule and the PaPs experiment read it;
* :class:`CompositeChips` (``encoder: finetuned``): the chip's twelve monthly S2 and S1 composites in TESSERA's
  layout (``gfm4agri.embeddings.tessera_v2.chip_inputs``); :func:`collate_composites` prepares TESSERA's bucketed
  inputs in the loader's workers (``tessera_ft.prepare``), and the encoder runs in the training step.

Training chips are those holding a drawn parcel, as the core's datamodule selects them under a budget. Each gets a
random D4 transform (exact for a per-pixel encoder) and, when ``crop`` is set, a random ``crop x crop`` window, the
same for the input, the labels and WorldCover, before :func:`prue.target` builds the three classes. Validation and
test chips are whole, untransformed, with every parcel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import prue


class _Chips(Dataset):
    def __init__(self, chip_ids, labels_dir: Path, worldcover: Path, keep: np.ndarray | None = None,
                 augment: bool = False, crop: int | None = None) -> None:
        self.ids = list(chip_ids)
        self.labels_dir = Path(labels_dir)
        self.wc_path = Path(worldcover)
        self._wc = None
        self.keep = None if keep is None else np.asarray(keep, np.int64)
        self.augment, self.crop = bool(augment), (int(crop) if crop else None)

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

    def geometry(self, h: int, w: int):
        """``(k, flip, r0, c0)`` of this item, from torch's generator (seeded per worker and epoch by the loader)."""
        k, flip = 0, False
        if self.augment:
            k, flip = int(torch.randint(4, ()).item()), bool(torch.randint(2, ()).item())
        r0 = c0 = 0
        if self.crop and self.crop < min(h, w):
            r0 = int(torch.randint(h - self.crop + 1, ()).item())
            c0 = int(torch.randint(w - self.crop + 1, ()).item())
        return k, flip, r0, c0

    def window(self, arrays, k, flip, r0, c0):
        """Every ``(..., H, W)`` array turned by D4, then cut to the crop window."""
        out = prue.d4(arrays, k, flip)
        if self.crop and self.crop < out[0].shape[-1]:
            s = self.crop
            out = [np.ascontiguousarray(a[..., r0:r0 + s, c0:c0 + s]) for a in out]
        return out


class RasterChips(_Chips):
    def __init__(self, chip_ids, raster_dir: Path, suffix: str, means, stds, labels_dir: Path, worldcover: Path,
                 **kw) -> None:
        super().__init__(chip_ids, labels_dir, worldcover, **kw)
        self.raster_dir, self.suffix = Path(raster_dir), suffix
        self.means = np.asarray(means, np.float32)[:, None, None]
        self.stds = np.asarray(stds, np.float32)[:, None, None]

    def image(self, cid: str) -> np.ndarray:
        import rasterio

        with rasterio.open(self.raster_dir / "chips" / f"{cid}{self.suffix}") as s:
            x = s.read().astype(np.float32)
        return (np.nan_to_num(x, nan=0.0) - self.means) / self.stds

    def __getitem__(self, i: int) -> dict:
        cid = self.ids[i]
        x = self.image(cid)
        mask, parcels = self.labels(cid)
        wc = self.worldcover(cid)
        g = self.geometry(*mask.shape)
        x, mask, parcels, wc = self.window([x, mask, parcels, wc], *g)
        y = prue.target(mask, parcels, wc, self.keep)
        return {"image": torch.from_numpy(x), "y": torch.from_numpy(y.astype(np.int64)),
                "mask": torch.from_numpy(mask.astype(np.int16)), "chip": cid}


class CompositeChips(_Chips):
    def __init__(self, chip_ids, chips_dir: Path, band_names, n_months: int, labels_dir: Path, worldcover: Path,
                 **kw) -> None:
        super().__init__(chip_ids, labels_dir, worldcover, **kw)
        self.chips_dir, self.band_names, self.n_months = Path(chips_dir), list(band_names), int(n_months)

    def __getitem__(self, i: int) -> dict:
        from gfm4agri.embeddings.tessera_v2 import chip_inputs

        cid = self.ids[i]
        s2, s1, (h, w) = chip_inputs(self.chips_dir, cid, self.band_names, self.n_months)   # (P, T, 10), (P, T, 2)
        t = s2.shape[1]
        s2 = s2.reshape(h, w, t, -1).transpose(2, 3, 0, 1)            # (T, 10, H, W)
        s1 = s1.reshape(h, w, t, -1).transpose(2, 3, 0, 1)            # (T, 2, H, W)
        mask, parcels = self.labels(cid)
        wc = self.worldcover(cid)
        g = self.geometry(h, w)
        s2, s1, mask, parcels, wc = self.window([s2, s1, mask, parcels, wc], *g)
        hh, ww = mask.shape
        s2 = np.ascontiguousarray(s2.transpose(2, 3, 0, 1).reshape(hh * ww, t, -1))    # back to (P, T, 10)
        s1 = np.ascontiguousarray(s1.transpose(2, 3, 0, 1).reshape(hh * ww, t, -1))
        y = prue.target(mask, parcels, wc, self.keep)
        return {"s2": s2, "s1": s1, "hw": (hh, ww), "y": torch.from_numpy(y.astype(np.int64)),
                "mask": torch.from_numpy(mask.astype(np.int16)), "chip": cid}


def collate_raster(items: list[dict]) -> dict:
    return {"image": torch.stack([it["image"] for it in items]), "y": torch.stack([it["y"] for it in items]),
            "mask": torch.stack([it["mask"] for it in items]), "chip": [it["chip"] for it in items]}


def make_collate_composites(doy: np.ndarray):
    """Concatenate the chips' pixels and prepare TESSERA's inputs here, in the loader's workers."""
    import tessera_ft

    def collate(items: list[dict]) -> dict:
        hw = items[0]["hw"]
        if any(it["hw"] != hw for it in items):
            raise ValueError("chips of one batch differ in size")
        return {"prep": tessera_ft.prepare(np.concatenate([it["s2"] for it in items]),
                                           np.concatenate([it["s1"] for it in items]), doy),
                "hw": hw, "n": len(items), "y": torch.stack([it["y"] for it in items]),
                "mask": torch.stack([it["mask"] for it in items]), "chip": [it["chip"] for it in items]}

    return collate


def train_chips(root: Path, ids: list[str], keep: np.ndarray) -> list[str]:
    """The training chips holding a drawn parcel, as the core's datamodule selects them under a budget."""
    from gfm4agri.benchmark.segmentation_data import chips_with_parcels

    have = set(chips_with_parcels(root, ids, keep))
    return [c for c in ids if c in have]
