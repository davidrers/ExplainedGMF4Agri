"""The training side of the parcel delineation experiment: the boundary target, the head and the joint loss.

**The target.** A pixel of a parcel is a *boundary* pixel when one of its four neighbours inside the chip carries
another identifier in ``<chip>.parcels.tif``: another parcel, or none. Every other pixel of the parcel is an
*interior* pixel. The chip edge is not a boundary. The target is defined on the labelled pixels of the mask and
only there, so it follows the label budget exactly as the crop-type label does: under a budget of K % only the
drawn parcels carry it, and every other pixel is ignored by both losses. A drawn polygon fixes its own outline
whatever its neighbours are, so the boundary of a drawn parcel against an undrawn one is known.

**How it travels.** The target rides in the mask itself: on a labelled pixel the mask carries
``class + OFFSET * boundary``, and the ignore index stays as it is. Every loader of the K-shot workflow (the
end-to-end dataset, the two-stage cached dataset, the predictor) then hands it through its own flips and rotations
with the crop-type label, so the two stay aligned without a second mask path. The task and the predictor split it
again with :func:`decode`.

**The head.** :class:`BoundaryHead` keeps the arm's segmentation head untouched and adds, beside it and on the same
decoder features, one boundary logit made like the head's classifier: the same dropout, the same kernel. The
decoder, the neck and the encoder are those of the arm, so the decoder capacity matched within a resolution group
stays matched; the head adds a few hundred parameters at most.

**The loss.** Cross-entropy on the crop types, as in the main workflow, plus ``WEIGHT`` times the binary
cross-entropy of the boundary logit, each the mean over its labelled pixels. ``val/loss``, the checkpoint
criterion, is that sum. Both terms are logged apart, with the boundary F1 at a threshold of 0.5.

Importing this module changes nothing; ``run.py`` swaps the classes in.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

#: The mask carries ``class + OFFSET * boundary`` on a labelled pixel; larger than any class count.
OFFSET = 1000
#: Weight of the boundary loss beside the crop-type loss, the same for every arm and budget.
WEIGHT = 1.0

RECORD = {
    "target": "boundary pixel: a pixel of a labelled parcel with a 4-neighbour inside the chip carrying another "
              "parcel identifier or none; the chip edge is not a boundary; defined on the labelled pixels only, "
              "so it follows the label budget",
    "head": "one logit beside the arm's segmentation head, on the same decoder features, built like the head's "
            "classifier (same dropout and kernel)",
    "loss": "cross-entropy on crop types + weight x binary cross-entropy on the boundary, each the mean over its "
            "labelled pixels; val/loss, the checkpoint criterion, is the sum",
    "weight": WEIGHT,
    "mask_encoding": f"class + {OFFSET} x boundary on labelled pixels",
}


# ---------------------------------------------------------------------------------------------------- the target

def boundary_map(parcels: np.ndarray) -> np.ndarray:
    """``True`` on every parcel pixel with a 4-neighbour inside the chip carrying another identifier."""
    p = np.asarray(parcels)
    b = np.zeros(p.shape, dtype=bool)
    v = p[:-1, :] != p[1:, :]
    b[:-1, :] |= v
    b[1:, :] |= v
    h = p[:, :-1] != p[:, 1:]
    b[:, :-1] |= h
    b[:, 1:] |= h
    return b & (p > 0)


def encode(mask: np.ndarray, parcels: np.ndarray) -> np.ndarray:
    """The mask with ``OFFSET`` added on its labelled boundary pixels; ignored pixels unchanged."""
    out = np.array(mask, copy=True)
    add = (out >= 0) & boundary_map(parcels)
    out[add] = out[add] + OFFSET
    return out


def decode(y):
    """``(class, boundary)`` of an encoded mask, ``-1`` where ignored; NumPy or torch."""
    if isinstance(y, torch.Tensor):
        lab = y >= 0
        cls = torch.where(lab, torch.remainder(y, OFFSET), y)
        bnd = torch.where(lab, torch.div(y, OFFSET, rounding_mode="floor"), torch.full_like(y, -1))
        return cls, bnd
    y = np.asarray(y)
    lab = y >= 0
    return np.where(lab, y % OFFSET, y), np.where(lab, y // OFFSET, -1)


# ------------------------------------------------------------------------------------------------------ the data

def sparse_dataset_class(base):
    """The end-to-end dataset (``SparseParcelSegmentationDataset``) with the boundary encoded into its masks."""

    class DelineationSparseDataset(base):
        def _load_file(self, path, nan_replace=None):
            from gfm4agri.benchmark.segmentation_data import MASK_SUFFIX, PARCELS_SUFFIX

            data, georef = super()._load_file(path, nan_replace=nan_replace)
            path = str(path)
            if not path.endswith(MASK_SUFFIX):
                return data, georef
            import rasterio

            with rasterio.open(path[: -len(MASK_SUFFIX)] + PARCELS_SUFFIX) as src:
                bnd = boundary_map(src.read(1))[None]
            v = data.values
            lab = np.isfinite(v) & (v >= 0)
            return data.copy(data=np.where(lab & bnd, v + OFFSET, v)), georef

    DelineationSparseDataset.__name__ = DelineationSparseDataset.__qualname__ = "DelineationSparseDataset"
    return DelineationSparseDataset


def cached_dataset_class(base):
    """The two-stage dataset (``CachedFeatureDataset``) with the boundary encoded before the D4 variant."""

    class DelineationCachedDataset(base):
        def __getitem__(self, i: int) -> dict:
            import rasterio

            from gfm4agri.benchmark.cached import EMB_SUFFIX, apply_d4, variant_name

            cid = self.chip_ids[i]
            k, flip = (self.variants[int(torch.randint(len(self.variants), ()).item())]
                       if len(self.variants) > 1 else self.variants[0])
            vdir = self.cache_dir / variant_name(k, flip)
            feats = [np.load(vdir / f"layer_{layer:02d}" / f"{cid}{EMB_SUFFIX}")
                     for layer in range(self.n_layers)]
            with rasterio.open(self.label_dir / f"{cid}.mask.tif") as src:
                mask = src.read(1).astype(np.int64)
            with rasterio.open(self.label_dir / f"{cid}.parcels.tif") as src:
                ids = src.read(1)
            if self.keep is not None:
                mask = np.where(np.isin(ids, self.keep), mask, self.ignore_index)
            mask = apply_d4(encode(mask, ids), k, flip)
            x = np.concatenate(feats, axis=0).astype(np.float32)
            return {"image": torch.from_numpy(x), "mask": torch.from_numpy(mask), "filename": cid}

    DelineationCachedDataset.__name__ = DelineationCachedDataset.__qualname__ = "DelineationCachedDataset"
    return DelineationCachedDataset


# ------------------------------------------------------------------------------------------------ head and task

class BoundaryHead(nn.Module):
    """The arm's segmentation head, untouched, and one boundary logit beside it from the same features.

    The output is the head's ``C`` class logits followed by the boundary logit, ``C + 1`` channels, which every
    model of the workflow resizes to the chip as it resizes the class logits.
    """

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        convs = [m for m in head.modules() if isinstance(m, nn.Conv2d)]
        if len(convs) != 1:
            raise ValueError(f"expected a head with a single convolution, found {len(convs)}")
        clf = convs[0]
        drops = [m for m in head.modules() if isinstance(m, nn.Dropout)]
        self.semantic = head
        self.boundary = nn.Sequential(nn.Dropout(drops[0].p) if drops else nn.Identity(),
                                      nn.Conv2d(clf.in_channels, 1, clf.kernel_size, padding=clf.padding))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.semantic(x), self.boundary(x)], dim=1)


def task_class(base):
    """``SemanticSegmentationTask`` with the boundary head and the joint loss."""
    from torchmetrics.classification import BinaryF1Score

    class BoundarySegmentationTask(base):
        def configure_models(self) -> None:
            super().configure_models()
            model = getattr(self, "model", None)
            if model is not None and not isinstance(model.head, BoundaryHead):
                model.head = BoundaryHead(model.head)

        def configure_metrics(self) -> None:
            super().configure_metrics()
            # ``train`` is a method of every module, so the keys carry a suffix.
            self.boundary_f1 = nn.ModuleDict({f"{s}_f1": BinaryF1Score() for s in ("train", "val", "test")})

        def _joint(self, batch, stage: str):
            if not isinstance(self.model.head, BoundaryHead):
                raise RuntimeError("the model has no boundary head")
            n_cls = self.hparams["model_args"]["num_classes"]
            y = self.squeeze_ground_truth(batch["mask"]).long()
            rest = {k: batch[k] for k in batch.keys() - {"image", "mask", "filename"}}
            out = self(batch["image"], **rest).output
            sem, logit = out[:, :n_cls], out[:, n_cls]
            cls, bnd = decode(y)
            sem_loss = self.criterion(sem, cls)
            sel = bnd >= 0
            if sel.any():
                b_loss = F.binary_cross_entropy_with_logits(logit[sel].float(), bnd[sel].float())
                self.boundary_f1[f"{stage}_f1"].update(torch.sigmoid(logit[sel].float()), bnd[sel])
            else:
                b_loss = logit.sum() * 0.0
            loss = sem_loss + WEIGHT * b_loss
            n = y.shape[0]
            on_epoch = stage != "train"
            self.log(f"{stage}/loss", loss, batch_size=n, sync_dist=True)
            self.log(f"{stage}/semantic_loss", sem_loss, batch_size=n, sync_dist=True, on_epoch=on_epoch)
            self.log(f"{stage}/boundary_loss", b_loss, batch_size=n, sync_dist=True, on_epoch=on_epoch)
            self.log(f"{stage}/boundary_F1", self.boundary_f1[f"{stage}_f1"], batch_size=n, on_step=False, on_epoch=True)
            return loss, sem.argmax(1), cls

        def training_step(self, batch, batch_idx, dataloader_idx=0):
            loss, pred, cls = self._joint(batch, "train")
            self.train_metrics.update(pred, cls)
            return loss

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            _, pred, cls = self._joint(batch, "val")
            self.val_metrics.update(pred, cls)

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            _, pred, cls = self._joint(batch, "test")
            self.test_metrics[dataloader_idx].update(pred, cls)

    BoundarySegmentationTask.__name__ = BoundarySegmentationTask.__qualname__ = "BoundarySegmentationTask"
    return BoundarySegmentationTask


# ------------------------------------------------------------------------------------------------- prediction

def predictor_class(base):
    """``FitPredictor`` returning the boundary probability beside the class and the decoded mask."""

    class DelineationPredictor(base):
        @torch.no_grad()
        def predict(self, chip_ids, split: str = "test", num_workers: int = 8) -> dict:
            """``chip_id -> (class uint8, mask int16, boundary probability x 255 uint8)``."""
            out = {}
            n_cls = len(self.class_names)
            for batch in self._loader(list(chip_ids), split, num_workers):
                names = [Path(str(f)).name for f in batch["filename"]]
                if not self.two_stage:
                    batch = self.dm.aug(batch)
                    if self.dm.modality_split:
                        batch["image"] = self.dm.split_modalities(batch["image"])
                x = batch["image"]
                x = ({k: v.to(self.device) for k, v in x.items()} if isinstance(x, dict)
                     else x.to(self.device))
                with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                    logits = self.model(x).output
                if logits.shape[1] != n_cls + 1:
                    raise RuntimeError(f"expected {n_cls} class logits and a boundary logit, got {logits.shape[1]}")
                pred = logits[:, :n_cls].argmax(1).cpu().numpy().astype(np.uint8)
                prob = torch.sigmoid(logits[:, n_cls].float()).cpu().numpy()
                prob = np.round(prob * 255).astype(np.uint8)
                cls, _ = decode(batch["mask"].cpu().numpy().astype(np.int64))
                if cls.ndim == 4:
                    cls = cls[:, 0]
                for nm, p, mk, b in zip(names, pred, cls.astype(np.int16), prob, strict=True):
                    sfx = self.dm.img_suffix
                    cid = nm[: -len(sfx)] if nm.endswith(sfx) else nm
                    out[cid] = (p, mk, b)
            return out

    DelineationPredictor.__name__ = DelineationPredictor.__qualname__ = "DelineationPredictor"
    return DelineationPredictor
