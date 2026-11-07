"""One model for the crop types and the field borders: PRUE's U-Net with a crop-type head beside its border head.

The two models it replaces, both on TESSERA v2 Medium's 128 dimensions at 10 m:

* the main workflow's crop-type arm: a per-pixel MLP (two hidden layers of 512 and 256, each ReLU, batch norm and
  dropout 0.2, then dropout 0.2 and a 1 x 1 convolution to the 20 classes: ``PixelMLPDecoder`` and TerraTorch's
  ``SegmentationHead``), cross-entropy on the labelled pixels of the drawn parcels;
* PRUE (``prue.py``): smp's U-Net with an EfficientNet-B7 encoder, three classes (background, interior, boundary),
  log-cosh Dice.

**The model** (:class:`JointNet`). The embedding feeds the U-Net as in PRUE, and its decoder's last features, 16
channels at full resolution, feed PRUE's own head to the three classes. The crop-type head is the main workflow's
MLP, unchanged but for its input: the 128 dimensions of the embedding **and** the U-Net's 16 decoder channels,
concatenated per pixel (144). So the crop head keeps the evidence the MLP map rests on and adds the U-Net's view of
the neighbourhood; its loss reaches the U-Net through those 16 channels, so the two tasks share the U-Net (and, in
the fine-tuned arm, the encoder). One forward pass gives both maps.

**The loss** (:func:`losses`): PRUE's log-cosh Dice on the three classes plus ``crop_weight`` times the
cross-entropy of the crop head on the labelled pixels (ignore index -1, the main workflow's ``ce``). Both heads are
trained by Adam, whose step does not depend on the scale of a parameter's gradient, so the weight decides nothing
for the crop head's own parameters: it sets only how hard the crop loss pulls on what the two share, against the
border loss. ``check.py`` measures the ratio of the two pulls on the U-Net at the start of training; the weight of
the arm files is set from it, so that the crop loss does not take over the U-Net PRUE's borders come from.

**The crop target** (:func:`crop_target`): the class of every in-scheme pixel of a drawn parcel, -1 elsewhere, the
restriction the core's ``SparseParcelSegmentationDataset`` applies under a budget (every trainable parcel at
K = 100 %); validation and test chips keep every parcel.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

#: Ignore index of the crop target, the chip set's ``ignore_index``.
IGNORE_CROP = -1


def crop_target(mask: np.ndarray, parcels: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """``(H, W)`` int64: the crop class of every in-scheme pixel of a drawn parcel, -1 elsewhere.

    The same pixels as :func:`prue.target`'s interior and boundary: ``mask >= 0`` on a declared parcel whose id is in
    ``keep`` (``None``: every parcel).
    """
    lab = (mask >= 0) & (parcels > 0)
    if keep is not None:
        lab &= np.isin(parcels, keep)
    return np.where(lab, mask, IGNORE_CROP).astype(np.int64)


class CropHead(nn.Module):
    """The main workflow's crop head: ``PixelMLPDecoder`` (512, 256) and TerraTorch's ``SegmentationHead``.

    Layer for layer as ``gfm4agri.benchmark.decoders.PixelMLPDecoder`` (1 x 1 convolution, ReLU, batch norm, dropout,
    per hidden layer) followed by dropout and a 1 x 1 convolution to the classes, written out here so this folder
    does not import TerraTorch.
    """

    def __init__(self, in_channels: int, n_classes: int, hidden=(512, 256), dropout: float = 0.2,
                 head_dropout: float = 0.2) -> None:
        super().__init__()
        layers, c = [], int(in_channels)
        for h in hidden:
            layers += [nn.Conv2d(c, int(h), kernel_size=1), nn.ReLU(), nn.BatchNorm2d(int(h)), nn.Dropout(dropout)]
            c = int(h)
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(head_dropout) if head_dropout else nn.Identity(),
                                  nn.Conv2d(c, int(n_classes), kernel_size=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.mlp(x))


def decoder_channels(unet) -> int:
    """Channels of the U-Net decoder's last output, the input of its segmentation head (16 for smp's default)."""
    conv = next(m for m in unet.segmentation_head.modules() if isinstance(m, nn.Conv2d))
    return int(conv.in_channels)


class JointNet(nn.Module):
    """The U-Net on the TESSERA embedding with two heads: borders ``(B, 3, H, W)`` and crop types ``(B, C, H, W)``.

    The embedding comes from the raster's batch (frozen arm) or from the student on the composites (fine-tuned arm),
    standardised either way, as in ``../2026-11-06_prue_tessera/fit_prue.py``'s ``Net``.
    """

    def __init__(self, unet, crop_head: CropHead, means, stds, student=None, trainable: bool = False,
                 chunk: int = 16384):
        super().__init__()
        self.unet, self.crop_head = unet, crop_head
        self.student, self.trainable, self.chunk = student, bool(trainable), int(chunk)
        self.register_buffer("emb_mean", torch.as_tensor(means, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("emb_std", torch.as_tensor(stds, dtype=torch.float32).view(1, -1, 1, 1))
        if student is not None:
            for p in student.parameters():
                p.requires_grad_(self.trainable)

    def embed(self, batch, device) -> torch.Tensor:
        if "image" in batch:
            return batch["image"].to(device, non_blocking=True)             # already standardised
        import tessera_ft

        e = tessera_ft.encode(self.student, batch["prep"], grad=self.trainable and torch.is_grad_enabled(),
                              chunk=self.chunk)
        h, w = batch["hw"]
        e = e.view(batch["n"], h, w, -1).permute(0, 3, 1, 2)
        return (e - self.emb_mean) / self.emb_std

    def forward(self, batch, device) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.embed(batch, device)
        with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            dec = self.unet.decoder(self.unet.encoder(x))
            seg = self.unet.segmentation_head(dec)
            crop = self.crop_head(torch.cat([x.to(dec.dtype), dec], dim=1))
        return seg, crop

    def train(self, mode: bool = True):
        super().train(mode)
        if self.student is not None:
            self.student.eval()
        return self


def crop_ce(logits: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the labelled pixels, in float32; 0 (with a graph) when a batch has none."""
    if not bool((yc != IGNORE_CROP).any()):
        return logits.float().sum() * 0.0
    return nn.functional.cross_entropy(logits.float(), yc, ignore_index=IGNORE_CROP)


def losses(seg, crop, y, yc, weights, crop_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(total, border, crop)``: PRUE's log-cosh Dice + ``crop_weight`` x the crop head's cross-entropy."""
    import prue

    lb = prue.log_cosh_dice(seg, y, weights)
    lc = crop_ce(crop, yc)
    return lb + float(crop_weight) * lc, lb, lc


def shared_grad_norms(net: JointNet, seg, crop, y, yc, weights, retain: bool = False) -> tuple[float, float]:
    """The norms of the border loss's and the crop loss's gradients over the U-Net's parameters, one batch.

    Two backward passes through the graph of one forward pass, by ``torch.autograd.grad``, so no ``.grad`` is
    touched; with ``retain`` the graph is kept for the training step's own backward pass. The crop loss's gradient
    is unweighted: multiply by the loss weight for its pull in training.
    """
    import prue

    params = [p for p in net.unet.parameters() if p.requires_grad]
    lb = prue.log_cosh_dice(seg, y, weights)
    lc = crop_ce(crop, yc)
    gb = torch.autograd.grad(lb, params, retain_graph=True, allow_unused=True)
    gc = torch.autograd.grad(lc, params, retain_graph=retain, allow_unused=True)

    def norm(gs):
        return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))

    return norm(gb), norm(gc)
