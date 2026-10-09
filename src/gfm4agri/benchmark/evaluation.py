"""Reload a finished fit and predict chips with it, for inspection after training.

A fit directory, ``<output_root>/<run>/<budget>_seed<s>/``, holds the resolved configuration,
``results.json`` and the best checkpoint. :class:`FitPredictor` rebuilds the model the fit
trained, from the configuration rather than from pickled objects, loads the checkpoint, and
predicts chips through the same input path the test metrics were computed on: the cached
features at the identity variant for a two-stage fit, and the datamodule's own loading and
normalisation for an end-to-end fit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import yaml

__all__ = ["FitPredictor", "confusion", "f1_from_iou", "per_class_scores"]

REPO = Path(__file__).resolve().parents[3]


def f1_from_iou(iou: float) -> float:
    """Per-class F1 from per-class IoU: both are functions of the same TP, FP and FN."""
    return 2.0 * iou / (1.0 + iou) if iou == iou else float("nan")


def confusion(pred: np.ndarray, mask: np.ndarray, n_classes: int) -> np.ndarray:
    """``(n_classes, n_classes)`` counts, rows the reference class, over labelled pixels."""
    sel = mask >= 0
    return np.bincount(mask[sel].astype(np.int64) * n_classes + pred[sel].astype(np.int64),
                       minlength=n_classes * n_classes).reshape(n_classes, n_classes)


def per_class_scores(cm: np.ndarray) -> dict:
    """Per-class F1 and IoU, and their macro means over the classes that occur."""
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    present = (tp + fp + fn) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = np.where(present, 2 * tp / (2 * tp + fp + fn), np.nan)
        iou = np.where(present, tp / (tp + fp + fn), np.nan)
    return {"f1": f1, "iou": iou, "macro_f1": float(np.nanmean(f1)),
            "miou": float(np.nanmean(iou)), "accuracy": float(tp.sum() / cm.sum())}


class FitPredictor:
    """The trained model of one fit directory, ready to predict chips."""

    def __init__(self, fit_dir, device: str | None = None) -> None:
        self.fit_dir = Path(fit_dir)
        self.cfg = yaml.safe_load((self.fit_dir / "config.yaml").read_text())
        self.results = json.loads((self.fit_dir / "results.json").read_text())
        self.two_stage = str(self.results.get("workflow", "")).startswith("two-stage")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        d = self.cfg["data"]
        self.root = REPO / d["root"]
        self.split_dir = REPO / d["split"] if d.get("split") else None
        self.class_names = self.results["class_names"]
        self._build()

    def _build(self) -> None:
        from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule

        m, d = self.cfg["model"], self.cfg["data"]
        self.dm = EuroCropsSegDataModule(self.root, m["backbone"], normalisation=d["normalisation"],
                                         batch_size=1, num_workers=0, augment=False,
                                         split_dir=self.split_dir)
        ignore = self.dm.manifest["ignore_index"]
        if self.two_stage:
            from gfm4agri.benchmark.cached import build_cached_model

            self.cache = self.results["feature_cache"]
            model = build_cached_model(m["backbone"], num_classes=len(self.class_names),
                                       class_names=self.class_names, bands=self.dm.bands,
                                       n_timesteps=self.dm.n_timesteps,
                                       channel_list=self.cache["channel_list"],
                                       ignore_index=ignore, head_dropout=m["head_dropout"])
        else:
            from gfm4agri.benchmark.segmentation import build_task

            model = build_task(m["backbone"], num_classes=len(self.class_names),
                               class_names=self.class_names, bands=self.dm.bands,
                               n_timesteps=self.dm.n_timesteps, ignore_index=ignore,
                               head_dropout=m["head_dropout"]).model
        ckpt = self.fit_dir / "checkpoints" / "best-loss.ckpt"
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        missing, unexpected = model.load_state_dict(state, strict=False)
        # An end-to-end checkpoint stores only the trainable weights; the frozen encoder comes
        # from its pretrained file. Anything else missing would be a real mismatch.
        bad = [k for k in missing if not k.startswith("encoder.")]
        if bad or unexpected:
            raise RuntimeError(f"checkpoint does not fit the model: missing {bad[:5]}, "
                               f"unexpected {unexpected[:5]}")
        self.model = model.to(self.device).eval()

    def _loader(self, chip_ids: list[str], split: str, num_workers: int):
        from torch.utils.data import DataLoader

        if self.two_stage:
            from gfm4agri.benchmark.cached import IDENTITY, CachedFeatureDataset
            from gfm4agri.benchmark.segmentation_data import split_chip_dir

            ds = CachedFeatureDataset(self.cache["dir"], split_chip_dir(self.root, split, self.split_dir),
                                      chip_ids, len(self.cache["channel_list"]),
                                      variants=[IDENTITY],
                                      ignore_index=self.dm.manifest["ignore_index"])
        else:
            self.dm.setup("test")
            ds = self.dm.test_dataset if split == "test" else None
            if ds is None:
                raise ValueError("end-to-end prediction is wired for the test split only")
            stem = (lambda f: Path(f).name[: -len(self.dm.img_suffix)])
            want = set(chip_ids)
            idx = [i for i, f in enumerate(ds.image_files) if stem(f) in want]
            ds.image_files = [ds.image_files[i] for i in idx]
            ds.segmentation_mask_files = [ds.segmentation_mask_files[i] for i in idx]
        return DataLoader(ds, batch_size=4, shuffle=False, num_workers=num_workers)

    @torch.no_grad()
    def predict(self, chip_ids: list[str], split: str = "test", num_workers: int = 8) -> dict:
        """``chip_id -> (prediction uint8, reference mask int16)``, both ``(224, 224)``."""
        out = {}
        for batch in self._loader(list(chip_ids), split, num_workers):
            names = [Path(str(f)).name for f in batch["filename"]]
            if not self.two_stage:
                # The datamodule normalises in ``on_after_batch_transfer`` only when a Lightning
                # trainer is attached, so outside one its augmentation is applied directly; the
                # cached features were stored after normalisation and need nothing.
                batch = self.dm.aug(batch)
                if self.dm.modality_split:
                    batch["image"] = self.dm.split_modalities(batch["image"])
            x = batch["image"]
            x = ({k: v.to(self.device) for k, v in x.items()} if isinstance(x, dict)
                 else x.to(self.device))
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(x).output
            pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
            mask = batch["mask"].cpu().numpy().astype(np.int16)
            for n, p, mk in zip(names, pred, mask):
                # Two-stage batches carry the chip id, end-to-end ones the raster's name.
                sfx = self.dm.img_suffix
                cid = n[: -len(sfx)] if n.endswith(sfx) else n
                out[cid] = (p, mk)
        return out
