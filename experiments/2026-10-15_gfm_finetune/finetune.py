"""The fine-tuned task and the predictor that matches it: copies of core pieces, changed for this experiment.

* :func:`build_ft_task` is ``gfm4agri.benchmark.segmentation.build_task`` with the encoder's freeze set by the arm
  (``model.trainable``) and the optimiser of :class:`FineTuneTask`. The model it builds is the core's, argument for
  argument, so a checkpoint loads into either.
* ``FineTuneTask`` is TerraTorch's ``SemanticSegmentationTask`` with two AdamW groups: the bottleneck, decoder and
  head at ``lr``, constant, exactly as the core task's single group, and a trainable encoder at ``encoder_lr``,
  rising linearly over the first ``encoder_warmup_epochs`` and constant after. The encoder is held in eval mode.
* :class:`FtFitPredictor` is the core's ``FitPredictor``. For a cell of this experiment's workflow it passes the
  batch's ``temporal_coords`` and ``location_coords`` to the model, as the task does in training and test, and it
  loads the checkpoint strictly. Any other cell it predicts exactly as the core does.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from gfm4agri.benchmark.evaluation import FitPredictor

__all__ = ["COORDS", "WORKFLOW", "FtFitPredictor", "build_ft_task"]

#: ``results.json`` ``workflow`` of a cell fitted by ``fit_ft.fit_finetune``, by the switch.
WORKFLOW = {True: "end-to-end, encoder trained", False: "end-to-end, encoder frozen"}
#: Batch entries other than the image that the model takes, as TerraTorch's task passes them.
COORDS = ("temporal_coords", "location_coords")


def _task_class():
    from terratorch.tasks import SemanticSegmentationTask

    class FineTuneTask(SemanticSegmentationTask):
        """``SemanticSegmentationTask`` with a separate, warmed-up learning rate for a trainable encoder."""

        def __init__(self, *args, encoder_lr: float = 1e-5, encoder_warmup_epochs: float = 1.0, **kwargs):
            super().__init__(*args, **kwargs)
            self.encoder_lr = float(encoder_lr)
            self.encoder_warmup_epochs = float(encoder_warmup_epochs)

        def on_train_batch_start(self, batch, batch_idx):
            # Lightning restores module modes after validation by setting each module's flag, not through
            # ``train()``, so the encoder is put back in eval mode before every training step. None of the three
            # encoders has active dropout, drop-path or batch norm, so this changes no number; it holds the encoder
            # to the computation the frozen cache was made with.
            self.model.encoder.eval()
            return super().on_train_batch_start(batch, batch_idx)

        def encoder_parameters(self) -> list[torch.nn.Parameter]:
            return [p for p in self.model.encoder.parameters() if p.requires_grad]

        def configure_optimizers(self):
            enc = self.encoder_parameters()
            ids = {id(p) for p in enc}
            rest = [p for p in self.parameters() if id(p) not in ids]
            wd = (self.hparams.get("optimizer_hparams") or {}).get("weight_decay", 0.0)
            # The core task hands AdamW every parameter at ``lr``; a frozen one never receives a gradient and is
            # never stepped, so the first group is the core's optimiser.
            groups = [{"params": rest, "lr": self.hparams["lr"], "name": "decoder"}]
            if not enc:
                return {"optimizer": torch.optim.AdamW(groups, weight_decay=wd)}
            groups.append({"params": enc, "lr": self.encoder_lr, "name": "encoder"})
            opt = torch.optim.AdamW(groups, weight_decay=wd)
            per_epoch = self.trainer.estimated_stepping_batches / max(1, self.trainer.max_epochs)
            warm = max(1, round(self.encoder_warmup_epochs * per_epoch))
            sched = torch.optim.lr_scheduler.LambdaLR(opt, [lambda s: 1.0, lambda s: min(1.0, (s + 1) / warm)])
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}

    return FineTuneTask


def build_ft_task(backbone: str, *, num_classes: int, class_names: list[str], bands: list[str],
                  n_timesteps: int, ignore_index: int = -1, lr: float = 1e-4, weight_decay: float = 0.05,
                  head_dropout: float = 0.1, loss: str = "ce", trainable: bool = True,
                  encoder_lr: float = 1e-5, encoder_warmup_epochs: float = 1.0):
    """The core's segmentation task with the encoder trainable (``trainable``) or frozen."""
    import terratorch  # noqa: F401  registers backbones, necks and decoders

    import gfm4agri.benchmark.decoders  # noqa: F401  registers the thesis decoders
    import gfm4agri.benchmark.necks  # noqa: F401  registers the thesis necks
    from gfm4agri.benchmark.backbones import get_backbone
    from gfm4agri.benchmark.segmentation import DECODERS

    spec = get_backbone(backbone)
    model_args = {
        **spec.model_args(bands, n_timesteps),
        **DECODERS[spec.resolution_group],
        "head_dropout": head_dropout,
        "num_classes": num_classes,
    }
    return _task_class()(
        model_factory="EncoderDecoderFactory",
        model_args=model_args,
        loss=loss,
        lr=lr,
        optimizer="AdamW",
        optimizer_hparams={"weight_decay": weight_decay},
        ignore_index=ignore_index,
        freeze_backbone=not trainable,
        freeze_decoder=False,
        plot_on_val=False,
        class_names=class_names,
        encoder_lr=encoder_lr,
        encoder_warmup_epochs=encoder_warmup_epochs,
    )


class FtFitPredictor(FitPredictor):
    """The core's predictor, which for a cell of this experiment passes the coordinates and loads strictly."""

    def __init__(self, fit_dir, device: str | None = None, checkpoint: str = "best-loss.ckpt") -> None:
        r = json.loads((Path(fit_dir) / "results.json").read_text())
        self.ours = r.get("workflow") in WORKFLOW.values()
        self.checkpoint = checkpoint
        super().__init__(fit_dir, device)

    def _build(self) -> None:
        if not self.ours:
            return super()._build()
        from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule

        m, d = self.cfg["model"], self.cfg["data"]
        self.dm = EuroCropsSegDataModule(self.root, m["backbone"], normalisation=d["normalisation"],
                                         batch_size=1, num_workers=0, augment=False,
                                         split_dir=self.split_dir)
        model = build_ft_task(m["backbone"], num_classes=len(self.class_names), class_names=self.class_names,
                              bands=self.dm.bands, n_timesteps=self.dm.n_timesteps,
                              ignore_index=self.dm.manifest["ignore_index"], head_dropout=m["head_dropout"],
                              trainable=bool(m["trainable"])).model
        state = torch.load(self.fit_dir / "checkpoints" / self.checkpoint, map_location="cpu",
                           weights_only=False)["state_dict"]
        state = {k[len("model."):]: v for k, v in state.items() if k.startswith("model.")}
        # Every weight, the encoder's included, comes from the checkpoint: a fine-tuned encoder that did not load
        # would otherwise be replaced, without a word, by the pretrained one.
        model.load_state_dict(state, strict=True)
        self.model = model.to(self.device).eval()

    @torch.no_grad()
    def predict(self, chip_ids: list[str], split: str = "test", num_workers: int = 8) -> dict:
        if not self.ours:
            return super().predict(chip_ids, split, num_workers)
        out = {}
        for batch in self._loader(list(chip_ids), split, num_workers):
            names = [Path(str(f)).name for f in batch["filename"]]
            batch = self.dm.aug(batch)
            if self.dm.modality_split:
                batch["image"] = self.dm.split_modalities(batch["image"])
            x = batch["image"]
            x = ({k: v.to(self.device) for k, v in x.items()} if isinstance(x, dict) else x.to(self.device))
            rest = {k: batch[k].to(self.device) for k in COORDS if k in batch}
            with torch.autocast(self.device, dtype=torch.float16, enabled=self.device == "cuda"):
                logits = self.model(x, **rest).output
            pred = logits.argmax(1).cpu().numpy().astype(np.uint8)
            mask = batch["mask"].cpu().numpy().astype(np.int16)
            sfx = self.dm.img_suffix
            for n, p, mk in zip(names, pred, mask):
                out[n[: -len(sfx)] if n.endswith(sfx) else n] = (p, mk)
        return out
