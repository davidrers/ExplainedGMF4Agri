"""TerraTorch data plumbing for the EuroCrops segmentation chips.

The chips written by ``scripts/hub/build_country_chips.py`` (and earlier by the 12-chip pilot's
builder, now in ``experiments/2026-09_pilot_12chips/``) already have the layout TerraTorch's
generic segmentation datamodule reads, so the only
things added here are the ones the thesis protocol needs and TerraTorch does not provide:

* **Sparse supervision.** The annotation unit is the parcel polygon, so a label budget of a
  percentage of the independent training parcels of each class is imposed by keeping the
  mask only on the pixels of the drawn parcels.
  :class:`SparseParcelSegmentationDataset` does this when the mask is loaded, before any
  augmentation, from the ``<chip>.parcels.tif`` raster next to each mask. The chips are
  never re-exported per budget, and validation and test masks stay dense.
* **Configuration from the chip manifest**, so class names, band order and statistics are
  never copied by hand into a config.
* **Two chip-set layouts.** A pilot set carries its split in ``training_chips/`` and
  ``validation_chips/``. A full-country set keeps every chip in ``chips/`` and takes its
  partition from a spatial split directory written by ``scripts/hub/build_chip_split.py``,
  passed as ``split_dir``; that directory also names the pool parcels the buffer withholds
  from training.
* **Per-model inputs from the backbone registry**: the band subset the encoder consumes, and
  for encoders with date and location encodings, ``temporal_coords`` (year and zero-based
  day of year of each monthly composite, taken at mid-month) and ``location_coords`` (the
  chip centre, latitude and longitude). TerraTorch forwards any extra batch key to the
  encoder, so nothing else has to change.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.datasets import GenericNonGeoSegmentationDataset

from gfm4agri.benchmark.backbones import get_backbone
from gfm4agri.data.seeding import derived_rng

__all__ = [
    "AUGMENT_SYMMETRIES",
    "EuroCropsSegDataModule",
    "SparseParcelSegmentationDataset",
    "chip_latlon",
    "draw_support",
    "load_manifest",
    "manifest_sha256",
    "monthly_temporal_coords",
    "raster_transforms",
    "read_split",
    "chips_with_parcels",
    "split_chip_dir",
    "split_list",
    "split_protocol",
    "temporal_transforms",
]

IMG_GREP = "*_merged.tif"
IMG_SUFFIX = "_merged.tif"
MASK_GREP = "*.mask.tif"
MASK_SUFFIX = ".mask.tif"
PARCELS_SUFFIX = ".parcels.tif"
SPLIT_DIRS = {"train": "training", "val": "validation", "test": "test"}


def load_manifest(root: Path | str) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def read_split(split_dir: Path | str) -> dict:
    """``split.json`` of a spatial split directory."""
    return json.loads((Path(split_dir) / "split.json").read_text())


def split_chip_dir(root: Path | str, split: str, split_dir: Path | str | None = None) -> Path:
    """Folder holding the chips of ``split``: ``chips/`` under a split directory."""
    root = Path(root)
    return root / "chips" if split_dir is not None else root / f"{SPLIT_DIRS[split]}_chips"


def split_list(root: Path | str, split: str, split_dir: Path | str | None = None) -> Path:
    """The chip-id list of ``split``."""
    base = Path(split_dir) if split_dir is not None else Path(root)
    return base / f"{SPLIT_DIRS[split]}_data.txt"


def chips_with_parcels(root: Path | str, chip_ids: list[str], parcel_ids) -> list[str]:
    """The chips of ``chip_ids`` holding at least one of ``parcel_ids``, in their order.

    Under a label budget most training chips may hold no drawn parcel. Such a chip carries no
    labelled pixel, contributes nothing to the loss, and is dropped from the training set
    rather than read for nothing every epoch.
    """
    import pandas as pd

    table = pd.read_parquet(Path(root) / "chip_parcels.parquet", columns=["chip_id", "parcel_id"])
    hit = set(table.loc[table["parcel_id"].isin(np.asarray(parcel_ids)), "chip_id"])
    return [c for c in chip_ids if c in hit]


def split_protocol(split_dir, test_is_val: bool) -> str | dict:
    """The split a result was produced under, for its result file."""
    if split_dir is None:
        return ("pilot: chips separated by minimum spacing only, no block split; "
                + ("test metrics computed on the validation chips" if test_is_val
                   else "held-out test chips"))
    rec = read_split(split_dir)
    return {"split": rec["split"], "config_hash": rec["config_hash"], "protocol": rec["protocol"],
            "chips": rec["lists"]}


def manifest_sha256(root: Path | str) -> str:
    return hashlib.sha256((Path(root) / "manifest.json").read_bytes()).hexdigest()


class SparseParcelSegmentationDataset(GenericNonGeoSegmentationDataset):
    """Generic segmentation dataset whose mask can be restricted to a set of parcels.

    ``keep_parcel_ids`` is ``None`` for a dense mask. Otherwise every pixel whose parcel id is
    not in the set is turned into NaN at load time, which ``no_label_replace`` then maps to
    the ignore index, so the restriction is in place before the transform runs and stays
    aligned with the image under any flip or rotation.
    """

    keep_parcel_ids: np.ndarray | None = None
    #: ``[(suffix, n_bands, fill)]``: rasters of further modalities, each interleaved month
    #: by month after the Sentinel-2 bands, with NaN replaced by ``fill`` per band.
    extra_rasters: list | None = None
    n_months: int | None = None
    #: ``chip_id -> (lat, lon)``, set when the encoder takes coordinates.
    chip_latlon: dict[str, tuple[float, float]] | None = None
    #: ``(T, 2)`` year and zero-based day of year of each time step.
    temporal_coords: np.ndarray | None = None

    def __getitem__(self, index: int) -> dict:
        item = super().__getitem__(index)
        if self.chip_latlon is not None:
            import torch

            chip_id = Path(item["filename"]).name[: -len("_merged.tif")]
            item["location_coords"] = torch.tensor(self.chip_latlon[chip_id], dtype=torch.float32)
            item["temporal_coords"] = torch.tensor(self.temporal_coords, dtype=torch.float32)
        return item

    def _load_file(self, path, nan_replace=None):
        path = str(path)
        if self.extra_rasters and path.endswith(IMG_SUFFIX):
            return self._load_image_with_extra(path, nan_replace)
        if self.keep_parcel_ids is None or not path.endswith(MASK_SUFFIX):
            return super()._load_file(path, nan_replace=nan_replace)
        import rioxarray

        data, georef = super()._load_file(path, nan_replace=None)
        with rioxarray.open_rasterio(path[: -len(MASK_SUFFIX)] + PARCELS_SUFFIX) as ids:
            keep = np.isin(ids.values, self.keep_parcel_ids)
        data = data.where(keep)
        if nan_replace is not None:
            data = data.fillna(nan_replace)
        return data, georef

    def _load_image_with_extra(self, path: str, nan_replace):
        """The Sentinel-2 stack with each further modality interleaved into it by month."""
        import rioxarray
        import xarray as xr

        base, georef = super()._load_file(path, nan_replace=nan_replace)
        stem = path[: -len(IMG_SUFFIX)]
        extras = []
        for suffix, n_bands, fill in self.extra_rasters:
            with rioxarray.open_rasterio(stem + suffix, masked=True) as r:
                e = r.to_numpy().astype(np.float32)
            e = e.reshape(self.n_months, n_bands, *e.shape[-2:])
            # A pixel never observed carries NaN; it enters at the band mean, which the
            # normalisation maps to zero.
            e = np.where(np.isfinite(e), e, np.asarray(fill, np.float32)[None, :, None, None])
            extras.append(e.reshape(-1, *e.shape[-2:]))
        joined = _interleave(base.to_numpy().astype(np.float32), extras, self.n_months)
        return xr.DataArray(joined, dims=("band", "y", "x")), georef


def _interleave(stack: np.ndarray, extras: list[np.ndarray], n_months: int) -> np.ndarray:
    """``(T * C, H, W)`` time-major stacks joined month by month into one time-major stack.

    Month ``m`` of the result holds the ``C`` bands of ``stack`` for that month followed by the
    bands of each extra stack for the same month, which is the layout the temporal expansion
    downstream expects of a single image.
    """
    hw = stack.shape[-2:]
    parts = [stack.reshape(n_months, -1, *hw)]
    parts += [e.reshape(n_months, -1, *hw) for e in extras]
    return np.concatenate(parts, axis=1).reshape(-1, *hw)


#: The symmetries the training augmentation draws from, as ``(quarter turns, horizontal flip)``,
#: identity first: all eight of D4. At K = 5 % on TerraMind v1 large over three seeds
#: (results/seg_cached/d4_ablation, 8 October 2026), test Macro-F1 was 0.358 +- 0.003 with all
#: eight, 0.335 +- 0.049 with the four quarter turns and 0.304 +- 0.030 without augmentation,
#: so the four that would halve the two-stage feature cache were not adopted.
AUGMENT_SYMMETRIES: list[tuple[int, bool]] = [(k, f) for k in range(4) for f in (False, True)]


def _numpy_d4():
    """The symmetries of ``AUGMENT_SYMMETRIES`` in NumPy, applied identically to image and mask.

    ``albumentations.D4`` goes through ``cv2.flip``, which rejects arrays with more channels
    than OpenCV supports, and a flattened 12-month, 12-band stack has 144.

    The symmetry is drawn from PyTorch's generator, not from the transform's own
    ``py_random``. DataLoader workers are fresh copies of the main process at every epoch, and
    albumentations' private generator is never reseeded in them, so every epoch replayed the
    same short sequence of symmetries: measured over three epochs of a fit, three distinct
    sequences among nine workers and only a few of the eight symmetries in training. PyTorch
    seeds its generator in every worker from a base seed drawn afresh each epoch, so the draw
    is random across workers and epochs, and reproducible under a fixed seed.
    """
    import albumentations as A

    class NumpyD4(A.DualTransform):
        def __init__(self, p: float = 1.0):
            super().__init__(p=p)

        def get_params_dependent_on_data(self, params, data):
            import torch

            k, flip = AUGMENT_SYMMETRIES[int(torch.randint(len(AUGMENT_SYMMETRIES), ()).item())]
            return {"k": k, "flip": flip}

        def _apply(self, arr, k, flip):
            arr = np.rot90(arr, k, axes=(0, 1))
            return np.ascontiguousarray(arr[:, ::-1] if flip else arr)

        def apply(self, img, k=0, flip=False, **params):
            return self._apply(img, k, flip)

        def apply_to_mask(self, mask, k=0, flip=False, **params):
            return self._apply(mask, k, flip)

        def get_transform_init_args_names(self):
            return ("p",)

    return NumpyD4()


def raster_transforms(augment: bool) -> list:
    """Albumentations pipeline for ``(H, W, C)`` embedding rasters: augment, to tensor."""
    from albumentations.pytorch import ToTensorV2

    return [*([_numpy_d4()] if augment else []), ToTensorV2()]


def temporal_transforms(n_timesteps: int, augment: bool) -> list:
    """Albumentations pipeline for ``(T, H, W, C)`` samples: flatten, augment, unflatten."""
    from albumentations.pytorch import ToTensorV2
    from terratorch.datasets.transforms import (
        FlattenTemporalIntoChannels, UnflattenTemporalFromChannels)

    return [FlattenTemporalIntoChannels(), *([_numpy_d4()] if augment else []), ToTensorV2(),
            UnflattenTemporalFromChannels(n_timesteps=n_timesteps)]


def monthly_temporal_coords(year: int, n_months: int = 12) -> np.ndarray:
    """``(T, 2)``: year and zero-based day of year of the 15th of each month."""
    import datetime as dt

    return np.array([[year, dt.date(year, m, 15).timetuple().tm_yday - 1]
                     for m in range(1, n_months + 1)], dtype=np.float32)


def chip_latlon(manifest: dict) -> dict[str, tuple[float, float]]:
    """Chip centre, latitude and longitude, from the EPSG:3035 bounds in the manifest."""
    from pyproj import Transformer

    tr = Transformer.from_crs(manifest["grid"]["crs"], "EPSG:4326", always_xy=True)
    out = {}
    for c in manifest["chips"]:
        xmin, ymin, xmax, ymax = c["bounds_3035"]
        lon, lat = tr.transform((xmin + xmax) / 2, (ymin + ymax) / 2)
        out[c["chip_id"]] = (float(lat), float(lon))
    return out


def draw_support(root: Path | str, pct: float, draw_seed: int, split: str = "train",
                 split_dir: Path | str | None = None) -> tuple[np.ndarray, dict]:
    """A percentage of the parcels of each class in ``split``, and a per-class report.

    The label budget is a **percentage of the independent parcels available for training**,
    applied per class: a class with ``n`` parcels contributes ``ceil(pct / 100 * n)`` of them,
    so a class present at all keeps at least one parcel and Macro-F1 stays interpretable at
    the scarce end of the grid. At ``pct = 100`` the draw is every parcel, which reproduces
    the dense mask and is the consistency check on the budget machinery.

    A parcel's class is the majority class of its labelled pixels across the chips; a parcel
    cut by a chip edge contributes only the pixels inside the chip. The draw for a class is
    seeded by ``(country, hcat_code, pct, draw_seed)`` alone, so it does not depend on the
    model, on the other classes or on iteration order.

    With a ``split_dir`` the parcels are those of the split's training chips, read from the
    chip set's ``chip_parcels.parquet``, less the parcels its buffer withholds. At
    ``pct = 100`` the draw is then every trainable parcel, which is the dense mask of the pool
    with the buffered parcels removed.
    """
    root = Path(root)
    manifest = load_manifest(root)
    if split_dir is not None:
        pids, clss, npx = _split_parcels(root, split, Path(split_dir))
    else:
        pids, clss, npx = _chip_parcels(root, manifest, split)

    n_classes = len(manifest["classes"])
    best: dict[int, tuple[int, int]] = {}
    for pid, cls, n in zip(pids.tolist(), clss.tolist(), npx.tolist()):
        if n > best.get(pid, (-1, 0))[1]:
            best[pid] = (cls, n)

    keep, report = [], {}
    for k, meta in enumerate(manifest["classes"][:n_classes]):
        pool = np.array(sorted(pid for pid, (cls, _) in best.items() if cls == k), dtype=np.int64)
        # The budget enters the seed as ``5``, never ``5.0``, so a draw does not depend on
        # whether the caller passed the percentage as an integer or a float.
        rng = derived_rng("seg_support", manifest["country"], meta["hcat_code"], f"{pct:g}",
                          draw_seed)
        n_draw = int(np.ceil(pct / 100.0 * len(pool))) if len(pool) else 0
        drawn = rng.choice(pool, size=min(n_draw, len(pool)), replace=False) if n_draw else pool[:0]
        keep.extend(drawn.tolist())
        report[meta["name"]] = {"available": int(len(pool)), "drawn": int(len(drawn))}
    return np.array(sorted(keep), dtype=np.int64), report


def _chip_parcels(root: Path, manifest: dict, split: str) -> tuple[np.ndarray, ...]:
    """``(parcel, class, pixels)`` of the chips of ``split`` in a pilot set, from the rasters."""
    import rasterio

    folder = root / f"{SPLIT_DIRS[split]}_chips"
    ids_all, cls_all = [], []
    for c in manifest["chips"]:
        if c["split"] != split:
            continue
        with rasterio.open(folder / f"{c['chip_id']}{MASK_SUFFIX}") as m, \
             rasterio.open(folder / f"{c['chip_id']}{PARCELS_SUFFIX}") as p:
            mask, ids = m.read(1), p.read(1)
        sel = (mask >= 0) & (ids > 0)
        ids_all.append(ids[sel]); cls_all.append(mask[sel])
    ids_all, cls_all = np.concatenate(ids_all), np.concatenate(cls_all)
    pairs, counts = np.unique(np.stack([ids_all, cls_all]), axis=1, return_counts=True)
    return pairs[0].astype(np.int64), pairs[1].astype(np.int64), counts.astype(np.int64)


def _split_parcels(root: Path, split: str, split_dir: Path) -> tuple[np.ndarray, ...]:
    """``(parcel, class, pixels)`` of the chips of ``split`` in a split directory.

    For the training split the parcels withheld by the buffer are removed.
    """
    import pandas as pd

    ids = set(split_list(root, split, split_dir).read_text().split())
    table = pd.read_parquet(root / "chip_parcels.parquet")
    table = table[table["chip_id"].isin(ids)]
    if split == "train":
        withheld = np.load(split_dir / "buffer_parcels.npy")
        table = table[~table["parcel_id"].isin(withheld)]
    g = table.groupby(["parcel_id", "class_index"], as_index=False)["pixels"].sum()
    return (g["parcel_id"].to_numpy(np.int64), g["class_index"].to_numpy(np.int64),
            g["pixels"].to_numpy(np.int64))


class EuroCropsSegDataModule(GenericNonGeoSegmentationDataModule):
    """Generic segmentation datamodule configured from a chip manifest.

    The backbone's ``representation`` picks the input rasters: ``s2_monthly`` is the
    144-band monthly Sentinel-2 stack (bands x months), anything else is a precomputed
    embedding raster described by ``<root>/<representation>.json``, for example
    ``tessera_v1`` (128 dimensions, no time axis) or ``alphaearth_v1`` (64). Masks, parcel
    rasters and split files are shared by every representation.

    ``normalisation`` is ``"backbone"`` for the statistics the encoder was pretrained under,
    or ``"chips"`` for training-chip statistics. An embedding raster has no pretraining
    statistics of its own, so for it both modes use the training-chip statistics.
    """

    def __init__(self, root: Path | str, backbone: str, *, normalisation: str = "backbone",
                 batch_size: int = 4, num_workers: int = 4, augment: bool = True,
                 keep_parcel_ids: np.ndarray | None = None,
                 split_dir: Path | str | None = None, **kwargs) -> None:
        root = Path(root)
        split_dir = None if split_dir is None else Path(split_dir)
        manifest = load_manifest(root)
        spec = get_backbone(backbone)
        if normalisation not in ("backbone", "chips"):
            raise ValueError(f"normalisation must be 'backbone' or 'chips', not {normalisation!r}")

        if spec.representation == "s2_monthly":
            dataset_bands = manifest["imagery"]["band_names"]
            bands = list(spec.input_bands or dataset_bands)
            n_t = manifest["imagery"]["n_months"]
            if normalisation == "backbone":
                means, stds = spec.stats(bands)
            else:
                idx = [dataset_bands.index(b) for b in bands]
                means = [manifest["normalisation"]["means"][i] for i in idx]
                stds = [manifest["normalisation"]["stds"][i] for i in idx]
            img_grep, img_suffix, temporal = IMG_GREP, IMG_SUFFIX, True
            transforms = {k: temporal_transforms(n_t, augment and k == "train")
                          for k in ("train", "val", "test")}
            no_data = None
        else:
            sidecar = json.loads((root / f"{spec.representation}.json").read_text())
            dataset_bands = sidecar["band_names"]
            bands = list(spec.input_bands or dataset_bands)
            idx = [dataset_bands.index(b) for b in bands]
            means = [sidecar["normalisation"]["means"][i] for i in idx]
            stds = [sidecar["normalisation"]["stds"][i] for i in idx]
            n_t = 1
            img_suffix, temporal = sidecar["file_suffix"], False
            img_grep = f"*{img_suffix}"
            transforms = {k: raster_transforms(augment and k == "train") for k in ("train", "val", "test")}
            # A pixel without an embedding (NaN) enters as 0. The export records the share of
            # such pixels per chip in the sidecar.
            no_data = 0.0
            self.representation_meta = sidecar

        # A further modality is read beside the Sentinel-2 stack and interleaved with it month
        # by month, so the temporal expansion, the band selection and the D4 augmentation
        # treat the two as one image and keep them aligned with each other and with the mask.
        # It is split back into one tensor per modality after normalisation, in
        # ``on_after_batch_transfer``. ``bands`` stays the Sentinel-2 list the encoder's S2L2A
        # input is built from; the modality brings its own.
        extra = spec.extra_modalities if spec.representation == "s2_monthly" else {}
        ds_bands, out_bands = list(dataset_bands), list(bands)
        means_all, stds_all = list(means), list(stds)
        extra_rasters, split = [], ([("S2L2A", len(bands))] if extra else None)
        for name, m in extra.items():
            mb = list(m["bands"])
            if normalisation == "backbone":
                mm, ss = m["stats"](mb)
            else:
                side = json.loads((root / m["sidecar"]).read_text())["normalisation"]
                mm = [side["means"][side["band_names"].index(x)] for x in mb]
                ss = [side["stds"][side["band_names"].index(x)] for x in mb]
            ds_bands += mb
            out_bands += mb
            means_all += list(mm)
            stds_all += list(ss)
            extra_rasters.append((m["suffix"], len(mb), list(mm)))
            split.append((name, len(mb)))

        roots = {}
        for split_name in SPLIT_DIRS:
            listed = split_list(root, split_name, split_dir)
            if listed.exists():
                roots[split_name] = (split_chip_dir(root, split_name, split_dir), listed)
        if "test" not in roots:  # the pilot has no test split; evaluate on validation
            roots["test"] = roots["val"]

        super().__init__(
            batch_size=batch_size, num_workers=num_workers,
            num_classes=len(manifest["classes"]),
            train_data_root=roots["train"][0], train_label_data_root=roots["train"][0],
            train_split=roots["train"][1],
            val_data_root=roots["val"][0], val_label_data_root=roots["val"][0],
            val_split=roots["val"][1],
            test_data_root=roots["test"][0], test_label_data_root=roots["test"][0],
            test_split=roots["test"][1],
            img_grep=img_grep, label_grep=MASK_GREP,
            dataset_bands=ds_bands, output_bands=out_bands, means=means_all, stds=stds_all,
            expand_temporal_dimension=temporal, reduce_zero_label=False,
            no_data_replace=no_data, no_label_replace=manifest["ignore_index"],
            train_transform=transforms["train"], val_transform=transforms["val"],
            test_transform=transforms["test"],
            **kwargs)
        self.dataset_class = SparseParcelSegmentationDataset
        self.root = root
        self.split_dir = split_dir
        self.manifest = manifest
        self.representation = spec.representation
        #: What follows the chip id in an input raster's name.
        self.img_suffix = img_suffix
        self.class_names = [c["name"] for c in manifest["classes"]]
        self.bands = bands
        self.n_timesteps = n_t
        self.normalisation = normalisation
        self.norm_means, self.norm_stds = list(means_all), list(stds_all)
        self.extra_rasters = extra_rasters
        self.modality_split = split
        self.keep_parcel_ids = keep_parcel_ids
        self.test_is_val = roots["test"] == roots["val"]
        self.uses_coords = spec.uses_coords
        self._latlon = chip_latlon(manifest) if spec.uses_coords else None
        self._tcoords = monthly_temporal_coords(manifest["year"], n_t) if spec.uses_coords else None

    def setup(self, stage: str) -> None:
        super().setup(stage)
        if self.extra_rasters:
            for name in ("train_dataset", "val_dataset", "test_dataset", "predict_dataset"):
                ds = getattr(self, name, None)
                if ds is not None:
                    ds.extra_rasters, ds.n_months = self.extra_rasters, self.n_timesteps
        if stage == "fit" and self.keep_parcel_ids is not None:
            self.train_dataset.keep_parcel_ids = np.asarray(self.keep_parcel_ids, dtype=np.int64)
            if self.split_dir is not None:
                ds = self.train_dataset
                stems = [Path(f).name[: -len(self.img_suffix)] for f in ds.image_files]
                keep = set(chips_with_parcels(self.root, stems, self.keep_parcel_ids))
                idx = [i for i, c in enumerate(stems) if c in keep]
                ds.image_files = [ds.image_files[i] for i in idx]
                ds.segmentation_mask_files = [ds.segmentation_mask_files[i] for i in idx]
        if self.uses_coords:
            for name in ("train_dataset", "val_dataset", "test_dataset", "predict_dataset"):
                ds = getattr(self, name, None)
                if ds is not None:
                    ds.chip_latlon, ds.temporal_coords = self._latlon, self._tcoords

    def split_modalities(self, x):
        """``(B, C, T, H, W)`` normalised image to ``{modality: (B, C_m, T, H, W)}``."""
        if not self.modality_split:
            return x
        parts, start = {}, 0
        for name, n in self.modality_split:
            parts[name] = x[:, start:start + n]
            start += n
        if start != x.shape[1]:
            raise ValueError(f"modality split covers {start} of {x.shape[1]} channels")
        return parts

    def on_after_batch_transfer(self, batch, dataloader_idx):
        batch = super().on_after_batch_transfer(batch, dataloader_idx)
        if self.modality_split:
            batch["image"] = self.split_modalities(batch["image"])
        return batch
