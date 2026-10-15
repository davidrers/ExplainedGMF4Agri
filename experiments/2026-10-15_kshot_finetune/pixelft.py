"""TESSERA v2 on the labelled-pixel route: the core's 10 m arm with the encoder inside the training loop.

The core's ``tessera_v2_medium`` arm trains a per-pixel MLP on whole chips of a raster the runner computed once with
the frozen encoder. Fine-tuning that encoder on whole chips would run it on every pixel of every training chip in
every epoch, about 90 GPU hours per seed on ``EE_2021_rev``. A pixel without a label gives no loss and, the encoder
and the MLP being per-pixel, no gradient through itself, so this route runs the encoder on the labelled pixels of each
batch only. The loss is the core's: cross-entropy averaged over the labelled pixels of the same four chips. One thing
changes: the MLP's batch norm takes its training statistics from those labelled pixels instead of from every pixel of
the four chips. The frozen control (``trainable: false``, reading the raster cache) carries the same change, so
fine-tuned minus control is the fine-tuning alone (DESIGN.md).

* :func:`prepare` turns pixels' composites into TESSERA's bucketed inputs, on the CPU, exactly as the core's
  ``tessera_v2._encode_batch`` does; :func:`encode` runs the student on them, with gradients when asked, in
  gradient-checkpointed chunks so memory does not grow with the number of labelled pixels.
* :class:`LabelledPixels` wraps the core datamodule's dataset: its masks (with the label budget applied) and its
  cached raster, plus the chip's composites when the encoder runs.
* :func:`build_pixel_task` is the core's task for ``tessera_v2_medium`` (identity backbone, ``PixelMLPDecoder``,
  head) as ``finetune.FineTuneTask``, with the student and the cache's normalisation statistics added; the batch is
  laid out as one ``(1, 128, N, 1)`` image of its ``N`` labelled pixels.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["PIXEL_WORKFLOW", "LabelledPixels", "build_pixel_task", "encode", "load_student", "make_collate", "prepare"]

#: ``results.json`` ``workflow`` of a cell fitted by ``fit_pixels.fit_pixels``, by the switch.
PIXEL_WORKFLOW = {True: "labelled pixels, TESSERA v2 encoder trained",
                  False: "labelled pixels, frozen TESSERA v2 raster cache"}
#: Boundary mIoU needs the pixels in their place on the chip; the labelled-pixel layout has no geometry.
DROPPED_METRICS = ("Boundary_mIoU",)


# ----------------------------------------------------------------------------------------------- encoder inputs

def prepare(s2: np.ndarray, s1_db: np.ndarray, doy: np.ndarray) -> dict:
    """TESSERA's inputs for ``N`` pixels, grouped by bucket pair, as ``tessera_v2._encode_batch`` builds them.

    ``s2``: ``(N, T, 10)`` DN in TESSERA's band order; ``s1_db``: ``(N, T, 2)`` our dB. Returns ``groups``, a list of
    ``(positions, s2_in, s1_in)`` tensors, and ``empty``, the pixels with no S2 and no S1 input at all, which the
    core's cache holds as NaN and its datamodule turns into 0.
    """
    from gfm4agri.embeddings import tessera_v2_model as tm
    from gfm4agri.embeddings.tessera_v2 import s1_to_tessera

    s2 = np.asarray(s2, dtype=np.float32)
    s1 = s1_to_tessera(s1_db)
    s1_valid = np.any(s1 != 0, axis=-1)
    s1n = (s1 - tm.S1A_BAND_MEAN) / (tm.S1A_BAND_STD + 1e-9)
    s2_valid = np.any(s2 != 0, axis=-1)
    keys = (tm._vec_get_bin_size(s2_valid.sum(1)).astype(np.int64) * 1000
            + tm._vec_get_bin_size(s1_valid.sum(1)).astype(np.int64))
    groups = []
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        s2_b, s1_b = max(int(key // 1000), 1), max(int(key % 1000), 1)
        s2_in = np.zeros((len(idx), s2_b, 11), np.float32)
        s1_in = np.zeros((len(idx), s1_b, 3), np.float32)
        if key // 1000 > 0:
            src = tm._build_source_indices(s2_valid[idx], s2_b)
            g = np.take_along_axis(s2[idx], src[:, :, None].repeat(10, axis=2), axis=1)
            s2_in[:, :, :10] = (g - tm.S2_BAND_MEAN) / (tm.S2_BAND_STD + 1e-9)
            s2_in[:, :, 10] = doy[src]
        if key % 1000 > 0:
            src = tm._build_source_indices(s1_valid[idx], s1_b)
            s1_in[:, :, :2] = np.take_along_axis(s1n[idx], src[:, :, None].repeat(2, axis=2), axis=1)
            s1_in[:, :, 2] = doy[src]
        groups.append((torch.from_numpy(idx), torch.from_numpy(s2_in), torch.from_numpy(s1_in)))
    empty = ~np.any(s2 != 0, axis=(1, 2)) & ~np.any(s1 != 0, axis=(1, 2))
    return {"groups": groups, "empty": torch.from_numpy(empty), "n": len(s2)}


def encode(student, prep: dict, *, grad: bool, chunk: int = 16384) -> torch.Tensor:
    """``(N, 128)`` float32 embeddings of the prepared pixels, 0 where a pixel has no input (the core's NaN -> 0).

    Under bf16 autocast, as the core's raster cache. With ``grad``, each chunk of up to ``chunk`` pixels goes through
    ``torch.utils.checkpoint``: its activations are recomputed in the backward pass instead of kept, so memory holds
    one chunk's activations whatever the number of labelled pixels. Chunking does not change the result; each pixel
    is encoded on its own.
    """
    from torch.utils.checkpoint import checkpoint

    device = next(student.parameters()).device
    out = torch.zeros(prep["n"], 128, device=device, dtype=torch.float32)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for pos, s2_in, s1_in in prep["groups"]:
            pos = pos.to(device)
            for i in range(0, len(pos), chunk):
                a, b = s2_in[i:i + chunk].to(device), s1_in[i:i + chunk].to(device)
                e = (checkpoint(student.encode, a, b, use_reentrant=False) if grad
                     else student.encode(a, b))
                out = out.index_copy(0, pos[i:i + chunk], e.float())
    empty = prep["empty"].to(device)
    return torch.where(empty[:, None], torch.zeros((), device=device), out)


# ------------------------------------------------------------------------------------------------------- data

class LabelledPixels(Dataset):
    """One item per chip of a core dataset: its labelled pixels (or all of them), with their labels.

    ``source`` is ``"cache"`` (the cached raster's 128 values per pixel, NaN already 0, as the core reads them) or
    ``"composites"`` (the chip's S2 and S1 composites in TESSERA's layout, for the encoder).
    """

    def __init__(self, ds, chips_dir: Path, *, source: str, band_names: list[str], n_months: int,
                 ignore_index: int, img_suffix: str, all_pixels: bool = False) -> None:
        if source not in ("cache", "composites"):
            raise ValueError(f"source {source!r}")
        self.ds, self.chips_dir, self.source = ds, Path(chips_dir), source
        self.band_names, self.n_months = list(band_names), int(n_months)
        self.ignore, self.sfx, self.all_pixels = int(ignore_index), img_suffix, all_pixels

    def __len__(self) -> int:
        return len(self.ds)

    def chip_id(self, i: int) -> str:
        return Path(self.ds.image_files[i]).name[: -len(self.sfx)]

    def __getitem__(self, i: int) -> dict:
        item = self.ds[i]
        y = item["mask"].reshape(-1).long()
        h, w = item["mask"].shape[-2:]
        idx = torch.arange(len(y)) if self.all_pixels else torch.nonzero(y != self.ignore).squeeze(1)
        out = {"y": y[idx], "idx": idx, "chip": self.chip_id(i), "hw": (int(h), int(w))}
        if self.source == "cache":
            img = item["image"]
            out["emb"] = img.reshape(img.shape[0], -1)[:, idx].T.float().contiguous()
        else:
            from gfm4agri.embeddings.tessera_v2 import chip_inputs

            s2, s1, _ = chip_inputs(self.chips_dir, out["chip"], self.band_names, self.n_months)
            sel = idx.numpy()
            out["s2"], out["s1"] = s2[sel], s1[sel]
        return out


def make_collate(doy: np.ndarray):
    """Concatenate the chips' pixels; for the composites, prepare TESSERA's inputs here, in the loader's workers."""

    def collate(items: list[dict]) -> dict:
        batch = {"y": torch.cat([it["y"] for it in items]), "n_chips": len(items),
                 "chips": [it["chip"] for it in items], "counts": [len(it["y"]) for it in items],
                 "idx": [it["idx"] for it in items], "hw": [it["hw"] for it in items]}
        if "emb" in items[0]:
            batch["emb"] = torch.cat([it["emb"] for it in items])
        else:
            batch["prep"] = prepare(np.concatenate([it["s2"] for it in items]),
                                    np.concatenate([it["s1"] for it in items]), doy)
        return batch

    return collate





# ------------------------------------------------------------------------------------------------------- task

def _pixel_task_class():
    from terratorch.tasks.segmentation_tasks import to_segmentation_prediction
    from torchmetrics import MetricCollection

    from finetune import _task_class

    base = _task_class()

    class PixelFineTuneTask(base):
        """``finetune.FineTuneTask`` on the labelled-pixel layout, the TESSERA student as its encoder."""

        def __init__(self, *args, trainable: bool = True, encoder_chunk: int = 16384, **kwargs):
            super().__init__(*args, **kwargs)
            self.trainable = bool(trainable)
            self.encoder_chunk = int(encoder_chunk)
            self.student = None

        def attach(self, student, means, stds) -> None:
            """The encoder (``None`` for the frozen control, which reads the cache) and the cache's statistics."""
            if student is not None:
                for p in student.parameters():
                    p.requires_grad_(self.trainable)
                self.student = student.eval()
            self.register_buffer("emb_mean", torch.as_tensor(means, dtype=torch.float32))
            self.register_buffer("emb_std", torch.as_tensor(stds, dtype=torch.float32))

        def configure_metrics(self) -> None:
            super().configure_metrics()

            def drop(coll):
                # Rebuilt rather than deleted from, so no compute group keeps the dropped metric.
                keep = {k: m for k, m in coll.items(keep_base=True) if k not in DROPPED_METRICS}
                return MetricCollection(keep, prefix=coll.prefix)

            self.train_metrics, self.val_metrics = drop(self.train_metrics), drop(self.val_metrics)
            self.test_metrics = torch.nn.ModuleList([drop(c) for c in self.test_metrics])

        def encoder_parameters(self) -> list[torch.nn.Parameter]:
            return [] if self.student is None else [p for p in self.student.parameters() if p.requires_grad]

        def on_train_batch_start(self, batch, batch_idx):
            if self.student is not None:
                self.student.eval()
            return super().on_train_batch_start(batch, batch_idx)

        def embed(self, batch) -> torch.Tensor:
            """``(N, 128)`` raw embeddings: from the cache, or from the student on the composites."""
            if "emb" in batch:
                return batch["emb"]
            return encode(self.student, batch["prep"], chunk=self.encoder_chunk,
                          grad=self.trainable and torch.is_grad_enabled())

        def pixel_forward(self, batch):
            e = (self.embed(batch) - self.emb_mean) / self.emb_std
            return self(e.T.reshape(1, e.shape[1], e.shape[0], 1))

        def _step(self, batch, handler, metrics):
            y = batch["y"].view(1, -1, 1)
            out = self.pixel_forward(batch)
            loss = handler.compute_loss(out, y, self.criterion, self.aux_loss)
            # Logged per chip, as the core's whole-chip batches, so epoch means weigh batches alike.
            handler.log_loss(self.log, loss_dict=loss, batch_size=batch["n_chips"])
            metrics.update(to_segmentation_prediction(out), y)
            return loss["loss"]

        def training_step(self, batch, batch_idx, dataloader_idx=0):
            return self._step(batch, self.train_loss_handler, self.train_metrics)

        def validation_step(self, batch, batch_idx, dataloader_idx=0):
            self._step(batch, self.val_loss_handler, self.val_metrics)

        def test_step(self, batch, batch_idx, dataloader_idx=0):
            from functools import partial

            y = batch["y"].view(1, -1, 1)
            out = self.pixel_forward(batch)
            h = self.test_loss_handler[dataloader_idx]
            loss = h.compute_loss(out, y, self.criterion, self.aux_loss)
            h.log_loss(partial(self.log, add_dataloader_idx=False), loss_dict=loss, batch_size=batch["n_chips"])
            self.test_metrics[dataloader_idx].update(to_segmentation_prediction(out), y)

    return PixelFineTuneTask


def build_pixel_task(*, num_classes: int, class_names: list[str], bands: list[str], ignore_index: int,
                     lr: float, weight_decay: float, head_dropout: float, loss: str, trainable: bool,
                     encoder_lr: float, encoder_warmup_epochs: float, encoder_chunk: int,
                     means, stds, student=None, backbone: str = "tessera_v2_medium"):
    """The core's ``tessera_v2_medium`` task (``segmentation.build_task``'s model arguments), as a pixel task."""
    import terratorch  # noqa: F401  registers backbones, necks and decoders

    import gfm4agri.benchmark.decoders  # noqa: F401  registers the thesis decoders
    import gfm4agri.benchmark.necks  # noqa: F401  registers the thesis necks
    from gfm4agri.benchmark.backbones import get_backbone
    from gfm4agri.benchmark.segmentation import DECODERS

    spec = get_backbone(backbone)
    model_args = {**spec.model_args(bands, 1), **DECODERS[spec.resolution_group],
                  "head_dropout": head_dropout, "num_classes": num_classes}
    task = _pixel_task_class()(
        model_factory="EncoderDecoderFactory", model_args=model_args, loss=loss, lr=lr, optimizer="AdamW",
        optimizer_hparams={"weight_decay": weight_decay}, ignore_index=ignore_index, freeze_backbone=True,
        freeze_decoder=False, plot_on_val=False, class_names=class_names, encoder_lr=encoder_lr,
        encoder_warmup_epochs=encoder_warmup_epochs, trainable=trainable, encoder_chunk=encoder_chunk)
    task.attach(student, means, stds)
    return task


def load_student(cfg_model: dict):
    """The core registry's TESSERA v2 student, with its checkpoint's SHA-256 checked."""
    from gfm4agri.benchmark.backbones import get_backbone
    from gfm4agri.embeddings.tessera_v2 import WEIGHTS
    from gfm4agri.embeddings.tessera_v2 import load_student as _load

    c = get_backbone(cfg_model["backbone"]).computed
    return _load(c["student"], WEIGHTS, c["checkpoint_sha256"])
