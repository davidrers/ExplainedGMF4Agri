"""Chips for every arm, with both targets: PRUE's three border classes ``y`` and the crop types ``yc``.

One item per chip, whole (224 x 224), as the main workflow trains; validation and test chips untransformed with every
parcel, training chips under the budget (only those holding a drawn parcel, as the core's datamodule selects them)
with one of the eight D4 symmetries drawn from PyTorch's generator (seeded per worker and epoch by the loader). The
labels, the parcel ids and WorldCover go through the same symmetry as the input before :func:`prue.target` and
:func:`heads.crop_target` build the targets. Three inputs:

* :class:`RasterChips`, a 10 m embedding raster ``(C, H, W)``: TESSERA v2 from the core's raster cache, or
  AlphaEarth from the chip set; NaN to 0, then standardised with the training-chip statistics of its sidecar, as the
  core's datamodule reads them.
* :class:`StackChips`, the baseline's input: the 12 monthly S2 composites (12 bands) and S1 composites (VV, VH)
  interleaved month by month into 168 channels, month ``m`` and band ``c`` at channel ``14 m + c``, the layout the
  main workflow's ``ScratchUNet`` flattens its ``(C, T, H, W)`` batch into; standardised per band with the chip
  set's statistics (S2 from the manifest, S1 from ``s1_rtc.json``), as its ``normalisation: chips``. A pixel never
  observed (NaN) enters at the band mean, as the core does for S1; S2 has none in ``EE_2021_rev``, the guard is kept.
* :class:`TokenChips`, the cached token features of a token-grid arm (the core's ``generate_embeddings``): the encoder
  layers of the identity variant for validation and test, and for training one of the eight variants the cache holds,
  drawn per sample, with the labels turned by ``gfm4agri.benchmark.cached.apply_d4`` under the same variant, exactly as
  the core's ``CachedFeatureDataset``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import heads
import prue


class _Chips(Dataset):
    def __init__(self, chip_ids, labels_dir: Path, worldcover: Path, keep: np.ndarray | None = None,
                 augment: bool = False) -> None:
        self.ids = list(chip_ids)
        self.labels_dir = Path(labels_dir)
        self.wc_path = Path(worldcover)
        self._wc = None
        self.keep = None if keep is None else np.asarray(keep, np.int64)
        self.augment = bool(augment)

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

    def symmetry(self) -> tuple[int, bool]:
        if not self.augment:
            return 0, False
        return int(torch.randint(4, ()).item()), bool(torch.randint(2, ()).item())

    def item(self, cid: str, x: np.ndarray, mask, parcels, wc) -> dict:
        y = prue.target(mask, parcels, wc, self.keep)
        return {"image": torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)),
                "y": torch.from_numpy(y.astype(np.int64)),
                "yc": torch.from_numpy(heads.crop_target(mask, parcels, self.keep)),
                "mask": torch.from_numpy(mask.astype(np.int16)), "chip": cid}


class RasterChips(_Chips):
    """A 10 m embedding raster ``<raster_dir>/chips/<chip><suffix>``, standardised with ``means`` and ``stds``."""

    def __init__(self, chip_ids, raster_dir: Path, suffix: str, means, stds, labels_dir: Path, worldcover: Path,
                 **kw) -> None:
        super().__init__(chip_ids, labels_dir, worldcover, **kw)
        self.raster_dir, self.suffix = Path(raster_dir), suffix
        self.means = np.asarray(means, np.float32)[:, None, None]
        self.stds = np.asarray(stds, np.float32)[:, None, None]

    def __getitem__(self, i: int) -> dict:
        import rasterio

        cid = self.ids[i]
        with rasterio.open(self.raster_dir / "chips" / f"{cid}{self.suffix}") as s:
            x = (np.nan_to_num(s.read().astype(np.float32), nan=0.0) - self.means) / self.stds
        mask, parcels = self.labels(cid)
        x, mask, parcels, wc = prue.d4([x, mask, parcels, self.worldcover(cid)], *self.symmetry())
        return self.item(cid, x, mask, parcels, wc)


class StackChips(_Chips):
    """The 12 monthly S2 and S1 composites as 168 standardised channels, month-major."""

    def __init__(self, chip_ids, chips_dir: Path, s2_bands: int, s2_means, s2_stds, s1_means, s1_stds,
                 n_months: int, labels_dir: Path, worldcover: Path, **kw) -> None:
        super().__init__(chip_ids, labels_dir, worldcover, **kw)
        self.chips_dir, self.n_months, self.s2_bands = Path(chips_dir), int(n_months), int(s2_bands)
        m = np.concatenate([np.asarray(s2_means, np.float32), np.asarray(s1_means, np.float32)])
        s = np.concatenate([np.asarray(s2_stds, np.float32), np.asarray(s1_stds, np.float32)])
        self.means, self.stds = m[None, :, None, None], s[None, :, None, None]     # (1, 14, 1, 1)

    def __getitem__(self, i: int) -> dict:
        import rasterio

        cid = self.ids[i]
        with rasterio.open(self.chips_dir / f"{cid}_merged.tif") as s:
            s2 = s.read().astype(np.float32)
        with rasterio.open(self.chips_dir / f"{cid}_s1rtc.tif") as s:
            s1 = s.read().astype(np.float32)
        t, (h, w) = self.n_months, s2.shape[-2:]
        x = np.concatenate([s2.reshape(t, self.s2_bands, h, w), s1.reshape(t, -1, h, w)], axis=1)   # (T, 14, H, W)
        x = np.where(np.isfinite(x), x, self.means)
        x = ((x - self.means) / self.stds).reshape(-1, h, w)                                         # (168, H, W)
        mask, parcels = self.labels(cid)
        x, mask, parcels, wc = prue.d4([x, mask, parcels, self.worldcover(cid)], *self.symmetry())
        return self.item(cid, x, mask, parcels, wc)


class TokenChips(_Chips):
    """Cached token features ``<cache>/<variant>/layer_XX/<chip>_merged_embedding.npy`` of ``n_layers`` layers."""

    def __init__(self, chip_ids, cache_dir: Path, n_layers: int, labels_dir: Path, worldcover: Path,
                 variants: list[str] | None = None, **kw) -> None:
        super().__init__(chip_ids, labels_dir, worldcover, **kw)
        from gfm4agri.benchmark.cached import IDENTITY, TRAIN_VARIANTS

        self.cache_dir, self.n_layers = Path(cache_dir), int(n_layers)
        self.variants = list(TRAIN_VARIANTS) if self.augment else [IDENTITY]
        if variants is not None and self.augment:
            self.variants = [(int(v[1]), v.endswith("f")) for v in variants]

    def __getitem__(self, i: int) -> dict:
        from gfm4agri.benchmark.cached import EMB_SUFFIX, apply_d4, variant_name

        cid = self.ids[i]
        k, flip = (self.variants[int(torch.randint(len(self.variants), ()).item())]
                   if len(self.variants) > 1 else self.variants[0])
        vdir = self.cache_dir / variant_name(k, flip)
        x = np.concatenate([np.load(vdir / f"layer_{layer:02d}" / f"{cid}{EMB_SUFFIX}")
                            for layer in range(self.n_layers)], axis=0).astype(np.float32)
        mask, parcels = self.labels(cid)
        mask, parcels, wc = (apply_d4(a, k, flip) for a in (mask, parcels, self.worldcover(cid)))
        return self.item(cid, x, mask, parcels, wc)


def collate(items: list[dict]) -> dict:
    return {"image": torch.stack([it["image"] for it in items]), "y": torch.stack([it["y"] for it in items]),
            "yc": torch.stack([it["yc"] for it in items]), "mask": torch.stack([it["mask"] for it in items]),
            "chip": [it["chip"] for it in items]}


def train_chips(root: Path, ids: list[str], keep: np.ndarray) -> list[str]:
    """The training chips holding a drawn parcel, as the core's datamodule selects them under a budget."""
    from gfm4agri.benchmark.segmentation_data import chips_with_parcels

    have = set(chips_with_parcels(root, ids, keep))
    return [c for c in ids if c in have]
