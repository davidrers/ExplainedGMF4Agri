"""Two-stage training on a frozen encoder: embed every chip once, then train the decoder.

This is the second of TerraTorch's two workflows. The first, used by
:func:`gfm4agri.benchmark.fit.fit_end_to_end`, keeps the frozen encoder inside the training
loop, so every epoch of every fit re-encodes every chip although the encoder never changes.
Here the encoder runs once per chip, through
TerraTorch's ``EmbeddingGenerationTask``, and only the trainable part of the model is trained
from the stored features.

**The cache point** is the output of the last module before the first trainable one. In the
neck stacks of :mod:`gfm4agri.benchmark.backbones` that is after ``SelectIndices`` and
``ReshapeTokensToImage`` and before ``ChannelBottleneck``, so what is stored is, for every
selected encoder layer, the ``(T x embed_dim, 14, 14)`` token map with the months stacked on
the channel axis.

**The two workflows train the same model.** The encoder is frozen, so its output for a given
input is fixed and caching it removes only the repetition. Stage 2 builds the end-to-end model
and keeps exactly its trainable modules, so the architecture and the parameter count are those
of the end-to-end fit.

**Augmentation has to be precomputed.** A ViT is not equivariant to rotation: encoding a
rotated chip is not the same as rotating the encoding of the chip. Training chips are therefore
encoded in every symmetry the augmentation draws, ``TRAIN_VARIANTS`` (all eight of D4), and
stage 2 draws one uniformly for every sample, as the end-to-end fit draws a random symmetry.
Validation and test chips are encoded once, unrotated.

Two departures from TerraTorch's stock classes keep the equivalence exact:

* ``EmbeddingGenerationTask.predict_step`` calls the model with the image alone, so an encoder
  that takes ``temporal_coords`` and ``location_coords``, Prithvi-EO-2.0 TL, would be encoded
  without its date and location embeddings. :class:`CoordsEmbeddingGenerationTask` passes them.
* TerraTorch's segmentation model resizes its prediction to the size of its input, which for
  14 x 14 cached features is not the 224 x 224 mask. :class:`CachedDecoderModel` resizes to
  the chip size instead.
"""

from __future__ import annotations

import datetime as dt
import json
import time
from pathlib import Path

import lightning.pytorch as pl
import numpy as np
import torch
from torch import nn

from gfm4agri.benchmark.backbones import get_backbone
from gfm4agri.benchmark.segmentation_data import (
    AUGMENT_SYMMETRIES,
    EuroCropsSegDataModule,
    load_manifest,
    chips_with_parcels,
    read_split,
    split_chip_dir,
    split_list,
    split_protocol,
)

__all__ = [
    "IDENTITY",
    "TRAIN_VARIANTS",
    "VARIANTS",
    "CachedDecoderModel",
    "CachedFeatureDataModule",
    "CachedFeatureDataset",
    "apply_d4",
    "build_cached_model",
    "embedding_model_args",
    "fit_cached",
    "generate_embeddings",
    "split_at_cache_point",
    "variant_name",
]

#: The eight D4 symmetries as ``(quarter turns, horizontal flip)``, identity first.
VARIANTS: list[tuple[int, bool]] = [(k, f) for k in range(4) for f in (False, True)]
IDENTITY = (0, False)
#: The variants training chips are encoded in and stage 2 draws from: the symmetries the
#: end-to-end augmentation draws, so the two workflows train on the same inputs.
TRAIN_VARIANTS: list[tuple[int, bool]] = list(AUGMENT_SYMMETRIES)
#: Necks carrying trainable parameters; the cache point sits before the first of them.
TRAINABLE_NECKS = {"ChannelBottleneck", "LearnedInterpolateToPyramidal", "AddBottleneckLayer"}
#: File name of the stored features of ``<chip>_merged.tif``, TerraTorch's naming with a NumPy
#: extension (see :class:`CoordsEmbeddingGenerationTask`).
EMB_SUFFIX = "_merged_embedding.npy"
REPO = Path(__file__).resolve().parents[3]


def variant_name(k: int, flip: bool) -> str:
    return f"k{k}" + ("f" if flip else "")


def apply_d4(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    """``(H, W, ...)`` array under ``k`` quarter turns, then a horizontal flip.

    These are the operations of the random D4 augmentation of
    :mod:`gfm4agri.benchmark.segmentation_data`, in the same order, so a cached variant is
    exactly one of the inputs the end-to-end fit can draw.
    """
    arr = np.rot90(arr, k, axes=(0, 1))
    return np.ascontiguousarray(arr[:, ::-1] if flip else arr)


def _fixed_d4(k: int, flip: bool):
    import albumentations as A

    class FixedD4(A.DualTransform):
        def __init__(self) -> None:
            super().__init__(p=1.0)

        def apply(self, img, **params):
            return apply_d4(img, k, flip)

        def apply_to_mask(self, mask, **params):
            return apply_d4(mask, k, flip)

        def get_transform_init_args_names(self):
            return ()

    return FixedD4()


def _fixed_temporal_transforms(n_timesteps: int, k: int, flip: bool) -> list:
    """The temporal pipeline of the end-to-end fit with the D4 draw fixed to one variant."""
    from albumentations.pytorch import ToTensorV2
    from terratorch.datasets.transforms import (
        FlattenTemporalIntoChannels,
        UnflattenTemporalFromChannels,
    )

    return [FlattenTemporalIntoChannels(), _fixed_d4(k, flip), ToTensorV2(),
            UnflattenTemporalFromChannels(n_timesteps=n_timesteps)]


def split_at_cache_point(model_args: dict) -> tuple[list[dict], list[dict]]:
    """The neck stack split into its frozen head and its trainable tail."""
    necks = list(model_args.get("necks", []))
    cut = next((i for i, n in enumerate(necks) if n["name"] in TRAINABLE_NECKS), len(necks))
    return necks[:cut], necks[cut:]


def embedding_model_args(backbone: str, bands: list[str], n_timesteps: int) -> dict:
    """``model_args`` for the encoder and the frozen necks only, for stage 1."""
    spec = get_backbone(backbone)
    if spec.representation != "s2_monthly":
        raise ValueError(f"{backbone} is a precomputed embedding already; there is no encoder "
                         "to cache")
    args = spec.model_args(bands, n_timesteps)
    frozen, _ = split_at_cache_point(args)
    out = {k: v for k, v in args.items() if k.startswith("backbone")}
    out["necks"] = frozen
    return out


def _coords_task_class():
    from terratorch.tasks.embedding_generation import EmbeddingGenerationTask

    class CoordsEmbeddingGenerationTask(EmbeddingGenerationTask):
        """``EmbeddingGenerationTask`` that also hands the encoder the batch's coordinates.

        The stock ``predict_step`` calls the model with the image alone. An encoder with date
        and location embeddings would then be encoded without them and would not reproduce the
        features it computes during training. Features are returned as float32, whatever the
        autocast precision of the forward pass, so they can be written as GeoTIFF.
        """

        _KEYS = ("temporal_coords", "location_coords")
        #: dtype the features are stored in; float16 when the encoder ran under 16-bit autocast.
        store_dtype = np.float32

        def write_batch(self, embedding, file_ids, metadata, dir_path) -> None:
            """One NumPy array per sample and layer, in TerraTorch's folder layout and naming.

            TerraTorch writes a GeoTIFF with one band per channel. At 4,608 to 15,360 channels
            and 14 x 14 pixels, GDAL takes over a second to read one such file back, which
            would make reading the cache slower than recomputing it; the same bytes as an
            array load in milliseconds. The features are not geographic rasters in any case:
            each is the token grid of one chip, whose georeference is the chip's own.
            """
            dir_path.mkdir(parents=True, exist_ok=True)
            arr = embedding.detach().cpu().numpy().astype(self.store_dtype)
            for b, fid in enumerate(file_ids):
                # Written under a temporary name and renamed, so an interrupted run never
                # leaves a truncated array that a resumed run would take for a finished one.
                final = dir_path / f"{Path(fid).stem}_embedding.npy"
                tmp = final.with_name(final.name + ".part")
                with open(tmp, "wb") as fh:
                    np.save(fh, arr[b])
                tmp.replace(final)

        def predict_step(self, batch, *args, **kwargs):
            self._coords = {k: batch.pop(k) for k in self._KEYS if k in batch}
            try:
                return super().predict_step(batch)
            finally:
                self._coords = {}

        def forward(self, x, **kwargs):
            out = self.model(x, **{**getattr(self, "_coords", {}), **kwargs})
            if isinstance(out, list):
                return [f.float() if torch.is_tensor(f) else f for f in out]
            return out

        def save_configuration_summary(self, x) -> None:
            coords = getattr(self, "_coords", {})
            if self._config_saved or not coords:
                return super().save_configuration_summary(x)
            # TerraTorch may already have replaced ``forward`` on the instance (for Prithvi it is
            # ``forward_features``, the class ``forward`` being the MAE pretraining pass), so
            # whatever the instance held is restored rather than the attribute deleted.
            enc = self.model.encoder
            had, previous = "forward" in enc.__dict__, enc.__dict__.get("forward")
            plain = enc.forward
            enc.forward = lambda inp, **kw: plain(inp, **{**coords, **kw})
            try:
                super().save_configuration_summary(x)
            finally:
                if had:
                    enc.forward = previous
                else:
                    del enc.forward

    return CoordsEmbeddingGenerationTask


class _VariantDataModule(EuroCropsSegDataModule):
    """The chips of some splits under one fixed D4 variant, as a predict set for stage 1.

    Normalisation, the second modality and the coordinates are exactly those of the end-to-end
    datamodule, which applies them in ``on_after_batch_transfer`` for prediction as for
    training.
    """

    def __init__(self, root, backbone: str, *, splits: list[str], variant: tuple[int, bool],
                 normalisation: str, batch_size: int, num_workers: int,
                 split_dir=None, done_dir: Path | None = None) -> None:
        super().__init__(root, backbone, normalisation=normalisation, batch_size=batch_size,
                         num_workers=num_workers, augment=False, split_dir=split_dir)
        self._splits, self._variant = list(splits), variant
        #: Folder of the last encoder layer of this variant; a chip already there is skipped.
        self._done_dir = done_dir
        self.skipped = 0

    def setup(self, stage: str) -> None:
        if stage != "predict":
            return super().setup(stage)
        from terratorch.datamodules.utils import wrap_in_compose_is_list

        self.skipped = 0
        super().setup("fit")
        if "test" in self._splits and not self.test_is_val:
            super().setup("test")
        tf = wrap_in_compose_is_list(_fixed_temporal_transforms(self.n_timesteps, *self._variant))
        attr = {"train": "train_dataset", "val": "val_dataset", "test": "test_dataset"}
        self._predict_sets = []
        for s in self._splits:
            ds = getattr(self, attr[s])
            ds.transform = tf
            if self._done_dir is not None and self._done_dir.exists():
                keep = [i for i, f in enumerate(ds.image_files)
                        if not (self._done_dir / f"{Path(f).stem}_embedding.npy").exists()]
                self.skipped += len(ds.image_files) - len(keep)
                ds.image_files = [ds.image_files[i] for i in keep]
                ds.segmentation_mask_files = [ds.segmentation_mask_files[i] for i in keep]
            self._predict_sets.append(ds)

    def predict_dataloader(self):
        from torch.utils.data import ConcatDataset, DataLoader

        return DataLoader(ConcatDataset(self._predict_sets), batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers)

    def n_pending(self) -> int:
        return sum(len(ds) for ds in self._predict_sets)


def generate_embeddings(backbone: str, root, cache_dir, *, normalisation: str = "backbone",
                        variants: list[tuple[int, bool]] | None = None, batch_size: int = 4,
                        num_workers: int = 4, precision: str = "16-mixed",
                        split_dir=None, resume: bool = True) -> dict:
    """Stage 1: encode every chip once at the cache point, with ``EmbeddingGenerationTask``.

    Training chips are encoded in every variant of ``variants`` (``TRAIN_VARIANTS``, all eight
    D4 symmetries, by default); validation and test chips in the identity only. The features land in
    ``<cache_dir>/<variant>/layer_XX/<chip>_merged_embedding.npy``, and ``cache.json`` records
    the cache point, the channel counts per layer, the variants, the time taken and the bytes
    written. The precision should be that of the end-to-end fit, whose encoder runs under the
    same autocast.

    With a ``split_dir`` the chips and their partition come from that spatial split. With
    ``resume`` a chip whose last layer is already on disk is not encoded again, so an
    interrupted run continues where it stopped; every file is written atomically.
    """
    root, cache_dir = Path(root), Path(cache_dir)
    variants = list(variants or TRAIN_VARIANTS)
    if IDENTITY not in variants:
        raise ValueError("the identity variant is needed for validation and test")
    probe = EuroCropsSegDataModule(root, backbone, normalisation=normalisation, batch_size=1,
                                   num_workers=0, augment=False, split_dir=split_dir)
    args = embedding_model_args(backbone, probe.bands, probe.n_timesteps)
    first = args["necks"][0] if args["necks"] else {}
    layers = list(first.get("indices", [-1])) if first.get("name") == "SelectIndices" else [-1]
    splits_eval = ["val"] + (["test"] if not probe.test_is_val else [])

    task = _coords_task_class()(model_args=dict(args), output_dir=str(cache_dir / "_tmp"),
                                layers=layers, output_format="tiff", has_cls=False)
    store_dtype = np.float16 if "16" in precision else np.float32
    task.store_dtype = store_dtype
    trainer = pl.Trainer(accelerator="auto", devices=1, precision=precision, logger=False,
                         enable_checkpointing=False, enable_progress_bar=False)
    runs = []
    for k, flip in variants:
        name = variant_name(k, flip)
        splits = ["train"] + (splits_eval if (k, flip) == IDENTITY else [])
        task.output_path = cache_dir / name
        task.output_path.mkdir(parents=True, exist_ok=True)
        done_dir = task.output_path / f"layer_{len(layers) - 1:02d}" if resume else None
        dm = _VariantDataModule(root, backbone, splits=splits, variant=(k, flip),
                                normalisation=normalisation, batch_size=batch_size,
                                num_workers=num_workers, split_dir=split_dir, done_dir=done_dir)
        dm.setup("predict")
        n = dm.n_pending()
        t0 = time.time()
        if n:
            print(f"[stage 1] {backbone} {name}: encoding {n} chips"
                  + (f", {dm.skipped} already cached" if dm.skipped else ""), flush=True)
            # The datamodule is already set up; Lightning calls setup again, which rebuilds
            # the same filtered predict sets.
            trainer.predict(task, datamodule=dm, return_predictions=False)
        runs.append({"variant": name, "splits": splits, "chips": n, "reused": dm.skipped,
                     "seconds": round(time.time() - t0, 1)})
    if (cache_dir / "_tmp").exists():
        (cache_dir / "_tmp").rmdir()

    channel_list = []
    ident = cache_dir / variant_name(*IDENTITY)
    for layer in range(len(layers)):
        f = next((ident / f"layer_{layer:02d}").glob(f"*{EMB_SUFFIX}"))
        shape = np.load(f, mmap_mode="r").shape
        channel_list.append(int(shape[0]))
        spatial = [int(shape[1]), int(shape[2])]
    record = {
        "backbone": backbone,
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "chip_set": str(root), "manifest_sha256": _sha256(root / "manifest.json"),
        "split": (None if split_dir is None else
                  {"dir": str(split_dir), "config_hash": read_split(split_dir)["config_hash"]}),
        "cache_point": [n["name"] for n in args["necks"]],
        "encoder_layers": layers, "channel_list": channel_list, "feature_size": spatial,
        "normalisation": normalisation, "precision": precision,
        "dtype": np.dtype(store_dtype).name,
        "train_variants": [variant_name(k, f) for k, f in variants],
        "eval_variant": variant_name(*IDENTITY),
        "runs": runs,
        "encode_seconds": round(sum(r["seconds"] for r in runs), 1),
        "chip_encodings": sum(r["chips"] + r["reused"] for r in runs),
        "bytes": sum(f.stat().st_size for f in cache_dir.rglob(f"*{EMB_SUFFIX}")),
    }
    (cache_dir / "cache.json").write_text(json.dumps(record, indent=2))
    return record


def _sha256(path: Path) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CachedFeatureDataset(torch.utils.data.Dataset):
    """Cached encoder features of some chips, with the mask under the same D4 variant.

    ``variants`` are the variants to draw from, uniformly per sample; a single variant is used
    as it stands. A label budget is applied exactly as in the end-to-end fit, by keeping the
    mask only on the pixels of the drawn parcels.
    """

    def __init__(self, cache_dir, label_dir, chip_ids: list[str], n_layers: int, *,
                 variants: list[tuple[int, bool]], keep_parcel_ids: np.ndarray | None = None,
                 ignore_index: int = -1) -> None:
        self.cache_dir, self.label_dir = Path(cache_dir), Path(label_dir)
        self.chip_ids, self.n_layers = list(chip_ids), n_layers
        self.variants = list(variants)
        self.keep = None if keep_parcel_ids is None else np.asarray(keep_parcel_ids, np.int64)
        self.ignore_index = ignore_index

    def __len__(self) -> int:
        return len(self.chip_ids)

    def __getitem__(self, i: int) -> dict:
        import rasterio

        cid = self.chip_ids[i]
        # PyTorch's generator is reseeded in every DataLoader worker at every epoch, so the draw
        # varies across workers and epochs without relying on any other library's seeding.
        k, flip = (self.variants[int(torch.randint(len(self.variants), ()).item())]
                   if len(self.variants) > 1 else self.variants[0])
        vdir = self.cache_dir / variant_name(k, flip)
        feats = [np.load(vdir / f"layer_{layer:02d}" / f"{cid}{EMB_SUFFIX}")
                 for layer in range(self.n_layers)]
        with rasterio.open(self.label_dir / f"{cid}.mask.tif") as src:
            mask = src.read(1).astype(np.int64)
        if self.keep is not None:
            with rasterio.open(self.label_dir / f"{cid}.parcels.tif") as src:
                ids = src.read(1)
            mask = np.where(np.isin(ids, self.keep), mask, self.ignore_index)
        mask = apply_d4(mask, k, flip)
        x = np.concatenate(feats, axis=0).astype(np.float32)
        return {"image": torch.from_numpy(x), "mask": torch.from_numpy(mask), "filename": cid}


class CachedFeatureDataModule(pl.LightningDataModule):
    """Train, validation and test loaders over a feature cache, with the chip set's splits."""

    def __init__(self, root, cache_dir, *, batch_size: int = 4, num_workers: int = 4,
                 keep_parcel_ids: np.ndarray | None = None, split_dir=None,
                 train_variants: list[str] | None = None) -> None:
        super().__init__()
        self.root, self.cache_dir = Path(root), Path(cache_dir)
        self.split_dir = None if split_dir is None else Path(split_dir)
        self.cache = json.loads((self.cache_dir / "cache.json").read_text())
        # The training draw uses TRAIN_VARIANTS unless told otherwise, as in an ablation of the
        # augmentation; validation and test always read the identity.
        self.train_variants = list(train_variants
                                   or [variant_name(k, f) for k, f in TRAIN_VARIANTS])
        missing = sorted(set(self.train_variants) - set(self.cache["train_variants"]))
        if missing:
            raise ValueError(f"variants {missing} are not in the cache {self.cache_dir}")
        self.manifest = load_manifest(self.root)
        self.batch_size, self.num_workers = batch_size, num_workers
        self.keep_parcel_ids = keep_parcel_ids
        self.n_layers = len(self.cache["channel_list"])
        self.class_names = [c["name"] for c in self.manifest["classes"]]
        self.ignore_index = self.manifest["ignore_index"]
        self.test_is_val = not split_list(self.root, "test", self.split_dir).exists()

    def _ids(self, split: str) -> list[str]:
        f = split_list(self.root, split, self.split_dir)
        ids = [s for s in f.read_text().split() if s]
        if split == "train" and self.split_dir is not None and self.keep_parcel_ids is not None:
            ids = chips_with_parcels(self.root, ids, self.keep_parcel_ids)
        return ids

    def _dataset(self, split: str, train: bool) -> CachedFeatureDataset:
        variants = ([(int(v[1]), v.endswith("f")) for v in self.train_variants]
                    if train else [IDENTITY])
        return CachedFeatureDataset(
            self.cache_dir, split_chip_dir(self.root, split, self.split_dir), self._ids(split),
            self.n_layers, variants=variants,
            keep_parcel_ids=self.keep_parcel_ids if train else None,
            ignore_index=self.ignore_index)

    def setup(self, stage: str) -> None:
        self.train_dataset = self._dataset("train", train=True)
        self.val_dataset = self._dataset("val", train=False)
        self.test_dataset = (self.val_dataset if self.test_is_val
                             else self._dataset("test", train=False))

    def _loader(self, ds, shuffle: bool):
        from torch.utils.data import DataLoader

        return DataLoader(ds, batch_size=self.batch_size, shuffle=shuffle,
                          num_workers=self.num_workers)

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)


class CachedDecoderModel(nn.Module):
    """The trainable part of an end-to-end model, fed cached features.

    ``neck``, ``decoder`` and ``head`` are taken from the end-to-end model itself. The forward
    pass is TerraTorch's pixel-wise forward from the cache point on, except that the prediction
    is resized to the chip size rather than to the size of the input, which here is the
    14 x 14 feature map.
    """

    def __init__(self, neck: nn.Module, decoder: nn.Module, head: nn.Module,
                 channel_list: list[int], out_size: tuple[int, int]) -> None:
        super().__init__()
        self.neck, self.decoder, self.head = neck, decoder, head
        self.channel_list = list(channel_list)
        self.out_size = tuple(out_size)

    def forward(self, x: torch.Tensor, **kwargs):
        from terratorch.models.model import ModelOutput

        feats = list(torch.split(x, self.channel_list, dim=1))
        feats = self.neck(feats, image_size=self.out_size)
        mask = self.head(self.decoder([f.clone() for f in feats]))
        if tuple(mask.shape[-2:]) != self.out_size:
            mask = nn.functional.interpolate(mask, size=self.out_size, mode="bilinear")
        return ModelOutput(output=mask)


def build_cached_model(backbone: str, *, num_classes: int, class_names: list[str],
                       bands: list[str], n_timesteps: int, channel_list: list[int],
                       out_size: tuple[int, int] = (224, 224), ignore_index: int = -1,
                       head_dropout: float = 0.1) -> CachedDecoderModel:
    """The end-to-end model with its encoder and frozen necks removed.

    The model is built exactly as for the end-to-end fit and its trainable modules are kept,
    so the two fits share the architecture and the parameter count by construction.
    """
    from gfm4agri.benchmark.segmentation import build_task

    full = build_task(backbone, num_classes=num_classes, class_names=class_names, bands=bands,
                      n_timesteps=n_timesteps, ignore_index=ignore_index,
                      head_dropout=head_dropout).model
    frozen, _ = split_at_cache_point(get_backbone(backbone).model_args(bands, n_timesteps))
    model = CachedDecoderModel(full.neck[len(frozen):], full.decoder, full.head, channel_list,
                               out_size)
    del full
    return model


def _labelled_fraction(ds: CachedFeatureDataset) -> float:
    """Mean labelled share of the training masks under the budget, read from the masks alone."""
    import rasterio

    fr = []
    for cid in ds.chip_ids:
        with rasterio.open(ds.label_dir / f"{cid}.mask.tif") as src:
            mask = src.read(1)
        if ds.keep is not None:
            with rasterio.open(ds.label_dir / f"{cid}.parcels.tif") as src:
                mask = np.where(np.isin(src.read(1), ds.keep), mask, ds.ignore_index)
        fr.append(float((mask >= 0).mean()))
    return float(np.mean(fr))


def fit_cached(config: str | Path | dict, cache_dir, *, pct: float, draw_seed: int = 0,
               output_root: str | Path = "results/seg_cached", num_workers: int = 4,
               max_epochs: int | None = None, train_variants: list[str] | None = None) -> dict:
    """Stage 2: one fit of the decoder on cached features, recorded like an end-to-end fit.

    ``config`` is the end-to-end run configuration, so the label budget draw, the optimiser,
    the learning rate, the weight decay, the dropout, the loss, the precision and the number of
    epochs are those of the end-to-end fit. The effective batch is kept; since the encoder is
    gone, the whole of it fits in one step and no gradient accumulation is needed. The result
    file carries the same fields as :func:`gfm4agri.benchmark.fit.fit_end_to_end` writes, plus
    ``workflow`` and the summary of the feature cache the fit read.
    """
    import yaml
    from terratorch.tasks import SemanticSegmentationTask

    from gfm4agri.benchmark.segmentation import count_params
    from gfm4agri.benchmark.segmentation_data import draw_support, manifest_sha256

    cfg = config if isinstance(config, dict) else yaml.safe_load(Path(config).read_text())
    pl.seed_everything(cfg["seed"], workers=True)
    root, cache_dir = REPO / cfg["data"]["root"], Path(cache_dir)
    m, t, d = cfg["model"], cfg["trainer"], cfg["data"]
    effective_batch = d["batch_size"] * t.get("accumulate_grad_batches", 1)

    split_dir = REPO / d["split"] if d.get("split") else None
    keep, support = draw_support(root, pct, draw_seed, split_dir=split_dir)
    dm = CachedFeatureDataModule(root, cache_dir, batch_size=effective_batch,
                                 num_workers=num_workers, keep_parcel_ids=keep,
                                 split_dir=split_dir, train_variants=train_variants)
    dm.setup("fit")
    if dm.cache["backbone"] != m["backbone"]:
        raise ValueError(f"cache {cache_dir} holds {dm.cache['backbone']}, config wants "
                         f"{m['backbone']}")
    probe = EuroCropsSegDataModule(root, m["backbone"], normalisation=d["normalisation"],
                                   batch_size=1, num_workers=0, augment=False,
                                   split_dir=split_dir)
    model = build_cached_model(m["backbone"], num_classes=len(dm.class_names),
                               class_names=dm.class_names, bands=probe.bands,
                               n_timesteps=probe.n_timesteps,
                               channel_list=dm.cache["channel_list"],
                               ignore_index=dm.ignore_index, head_dropout=m["head_dropout"])
    # With a custom model TerraTorch builds nothing from ``model_args``, but reads the class
    # count from it for its metrics.
    task = SemanticSegmentationTask(model=model, model_args={"num_classes": len(dm.class_names)},
                                    loss=m["loss"], lr=m["lr"], optimizer="AdamW",
                                    optimizer_hparams={"weight_decay": m["weight_decay"]},
                                    ignore_index=dm.ignore_index, class_names=dm.class_names,
                                    plot_on_val=False)
    # TerraTorch records every constructor argument as a hyperparameter, the model object
    # included. Lightning's logger then tries to serialise that model to YAML before skipping
    # it, which took minutes per fit and per test, and the checkpoint would pickle it. The
    # model is described by the configuration and the result file, so it is dropped here.
    for store in (task._hparams, task._hparams_initial):
        store.pop("model", None)
    out = Path(output_root)
    out = (out if out.is_absolute() else REPO / out) / cfg["run_name"] / (
        f"P{pct:g}_draw{draw_seed}_seed{cfg['seed']}")
    out.mkdir(parents=True, exist_ok=True)
    ckpt = pl.callbacks.ModelCheckpoint(dirpath=out / "checkpoints", monitor="val/loss",
                                        mode="min", filename="best-loss",
                                        save_weights_only=True)
    trainer = pl.Trainer(accelerator="auto", devices=1, precision=t["precision"],
                         max_epochs=max_epochs or t["max_epochs"],
                         log_every_n_steps=t["log_every_n_steps"], callbacks=[ckpt],
                         logger=pl.loggers.CSVLogger(out, name="logs"), default_root_dir=out,
                         deterministic="warn", enable_progress_bar=False)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    trainer.fit(task, datamodule=dm)
    fit_s = time.time() - t0
    test = trainer.test(task, datamodule=dm, ckpt_path=ckpt.best_model_path or None)[0]
    peak_gb = torch.cuda.max_memory_allocated() / 1e9 if torch.cuda.is_available() else 0.0

    manifest = dm.manifest
    cache = dm.cache
    results = {
        "run": cfg["run_name"],
        "workflow": "two-stage: frozen encoder features cached once, decoder trained from cache",
        "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
        "config": cfg,
        "country": manifest["country"], "year": manifest["year"],
        "dataset": manifest["dataset"], "manifest_sha256": manifest_sha256(root),
        "split_protocol": split_protocol(split_dir, dm.test_is_val),
        "label_budget": {"mode": "pct", "pct": pct, "draw_seed": draw_seed,
                         "labelled_pixel_fraction_train": round(
                             _labelled_fraction(dm.train_dataset), 4),
                         "support": support},
        "seed": cfg["seed"], "effective_batch": effective_batch,
        "chips": {"train": len(dm.train_dataset), "val": len(dm.val_dataset),
                  "test": len(dm.test_dataset)},
        "backbone": m["backbone"], "frozen_encoder": True,
        "feature_cache": {k: cache[k] for k in (
            "cache_point", "encoder_layers", "channel_list", "feature_size", "normalisation",
            "precision", "train_variants", "eval_variant", "encode_seconds", "chip_encodings",
            "bytes")} | {"dir": str(cache_dir), "train_variants_drawn": dm.train_variants},
        "params": count_params(task),
        "best_checkpoint": ckpt.best_model_path,
        "fit_seconds": round(fit_s, 1), "peak_gpu_gb": round(peak_gb, 2),
        "test_metrics": {k: float(v) for k, v in test.items()},
        "class_names": dm.class_names,
    }
    (out / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    (out / "results.json").write_text(json.dumps(results, indent=2))
    return results
