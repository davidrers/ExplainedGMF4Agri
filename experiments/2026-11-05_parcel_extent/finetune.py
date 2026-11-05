"""The fine-tuning task of ``experiments/2026-10-15_kshot_finetune/finetune.py``, copied unchanged: TerraTorch's
``SemanticSegmentationTask`` with two AdamW groups, the decoder and head at ``lr``, constant, and a trainable encoder
at ``encoder_lr``, rising linearly over the first ``encoder_warmup_epochs`` and constant after; the encoder is held in
eval mode. ``pixelbnd.py`` builds its boundary task on it. Only ``_task_class`` is copied; the source's predictor and
token-grid task are not used here.
"""

from __future__ import annotations

import torch


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
