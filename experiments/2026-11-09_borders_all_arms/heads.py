"""Every arm of the main workflow with PRUE's border head beside its crop head, and what trains the two together.

Each arm keeps its frozen encoder and gets the network that lets it draw 10 m borders the way PRUE does, by what its
input lacks:

* **10 m embeddings** (TESSERA v2, AlphaEarth) lack spatial context, since every pixel is encoded on its own.
  :class:`UNetJoint` puts PRUE's U-Net on them (smp, EfficientNet-B7, ImageNet, its first convolution widened to the
  embedding's dimensions): the border head is the U-Net's own segmentation head (16 features to 3 classes, a 3 x 3
  convolution), and the crop head is the main workflow's per-pixel MLP reading the embedding **and** the U-Net's 16
  features, as in ``../2026-11-07_prue_crop_tessera`` (option 2 there, which matched both separate models).
* **The baseline** reads the raw monthly S2 + S1 composites (168 channels), so it simply becomes PRUE: the same
  :class:`UNetJoint` on those channels, from ImageNet (user, 9 November 2026), in place of the main workflow's
  ResNet-18 U-Net trained from scratch.
* **Token-grid GFMs** (TerraMind, THOR, Prithvi) already carry context through attention but only at the token size
  (140 to 160 m). :class:`TokenJoint` keeps the main workflow's trainable stack on the cached tokens exactly
  (``ChannelBottleneck``, ``LearnedInterpolateToPyramidal``, TerraTorch's ``UNetDecoder`` of 512, 256, 128, 64),
  whose last learned map is at 20 m and whose output the main workflow stretches bilinearly to 10 m. Here a fifth
  block brings it to the chip's 10 m grid (bilinear, then two 3 x 3 convolutions with batch norm and ReLU, user,
  9 November 2026), so both heads work at 10 m: PRUE's border head (a 3 x 3 convolution to 3 classes) and the main
  workflow's crop head (dropout and a 1 x 1 convolution to the 20 classes, TerraTorch's segmentation head). No raw
  band enters this decoder: the borders must come from what the tokens know.

**The loss** (:func:`losses`) is PRUE's log-cosh Dice on the three classes plus a weight times the crop head's
cross-entropy on the labelled pixels. The heads are separate, so the weight hardly changes how fast each learns
(Adam scales every step to its parameter's own gradient); it decides how each step of the **shared** part (the U-Net
or the token decoder) is split between borders and crop types. A fixed weight would mean something different on
every decoder, so :class:`Balance` sets it as training goes: every ``every`` steps it measures the two losses'
gradient norms over the shared parameters (:func:`shared_grad_norms`) and sets the weight so that the crop loss pulls
``target`` times as hard as the border loss, smoothed in log space. ``target`` 0.3 is the median pull of the TESSERA
arm of the joint experiment at a fixed 0.01, the setting that matched both separate models.

**The crop target** (:func:`crop_target`) is the class of every in-scheme pixel of a drawn parcel, -1 elsewhere, the
restriction the core's ``SparseParcelSegmentationDataset`` applies under a budget.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn

import prue

#: Ignore index of the crop target, the chip set's ``ignore_index``.
IGNORE_CROP = -1


def crop_target(mask: np.ndarray, parcels: np.ndarray, keep: np.ndarray | None = None) -> np.ndarray:
    """``(H, W)`` int64: the crop class of every in-scheme pixel of a drawn parcel (``keep``; ``None``: all), -1
    elsewhere. The same pixels as :func:`prue.target`'s interior and boundary."""
    lab = (mask >= 0) & (parcels > 0)
    if keep is not None:
        lab &= np.isin(parcels, keep)
    return np.where(lab, mask, IGNORE_CROP).astype(np.int64)


class CropHead(nn.Module):
    """The main workflow's per-pixel crop head: ``PixelMLPDecoder`` (512, 256) and TerraTorch's ``SegmentationHead``,
    layer for layer (1 x 1 convolution, ReLU, batch norm, dropout per hidden layer; dropout and a 1 x 1 convolution
    to the classes)."""

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


class UNetJoint(nn.Module):
    """PRUE's U-Net on a 10 m input ``(B, C, H, W)`` with two heads: borders ``(B, 3, H, W)``, crops ``(B, K, H, W)``.

    The input is already standardised (the loaders do it). The crop head reads the input and the U-Net's last
    features, concatenated per pixel.
    """

    def __init__(self, in_channels: int, n_classes: int, *, encoder: str = "efficientnet-b7",
                 weights: str | None = "imagenet", hidden=(512, 256), dropout: float = 0.2,
                 head_dropout: float = 0.2) -> None:
        super().__init__()
        self.unet = prue.build_unet(int(in_channels), encoder, weights)
        self.crop_head = CropHead(int(in_channels) + decoder_channels(self.unet), n_classes, hidden=hidden,
                                  dropout=dropout, head_dropout=head_dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        dec = self.unet.decoder(self.unet.encoder(x))
        return self.unet.segmentation_head(dec), self.crop_head(torch.cat([x.to(dec.dtype), dec], dim=1))

    def shared_parameters(self) -> list[nn.Parameter]:
        """What both losses train: the U-Net's encoder and decoder (not the two heads)."""
        return [p for m in (self.unet.encoder, self.unet.decoder) for p in m.parameters() if p.requires_grad]

    def groups(self) -> dict[str, list[nn.Parameter]]:
        return {"unet": list(self.unet.parameters()), "crop_head": list(self.crop_head.parameters())}


def _conv_bn_relu(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True))


class TokenJoint(nn.Module):
    """The main workflow's trainable stack on cached token features, a fifth block to 10 m, and the two heads.

    ``neck`` and ``decoder`` are those of the core's ``build_cached_model`` for the arm (the end-to-end model's
    trainable modules, so architecture and initialisation are the main workflow's); ``channel_list`` splits the
    cached tensor into its encoder layers, as ``CachedDecoderModel`` does.
    """

    def __init__(self, neck: nn.Module, decoder: nn.Module, channel_list: list[int], out_size: tuple[int, int],
                 n_classes: int, *, width: int = 64, head_dropout: float = 0.1) -> None:
        super().__init__()
        self.neck, self.decoder = neck, decoder
        self.channel_list, self.out_size = list(channel_list), tuple(out_size)
        c = int(decoder.out_channels)
        self.up = nn.Sequential(_conv_bn_relu(c, width), _conv_bn_relu(width, width))
        self.border_head = nn.Conv2d(width, 3, kernel_size=3, padding=1)          # as smp's SegmentationHead
        self.crop_head = nn.Sequential(nn.Dropout(head_dropout) if head_dropout else nn.Identity(),
                                       nn.Conv2d(width, int(n_classes), kernel_size=1))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        feats = list(torch.split(x, self.channel_list, dim=1))
        feats = self.neck(feats, image_size=self.out_size)
        d = self.decoder([f.clone() for f in feats])
        d = nn.functional.interpolate(d, size=self.out_size, mode="bilinear", align_corners=False)
        f = self.up(d)
        return self.border_head(f), self.crop_head(f)

    def shared_parameters(self) -> list[nn.Parameter]:
        return [p for m in (self.neck, self.decoder, self.up) for p in m.parameters() if p.requires_grad]

    def groups(self) -> dict[str, list[nn.Parameter]]:
        return {"decoder": [p for m in (self.neck, self.decoder, self.up, self.border_head) for p in m.parameters()],
                "crop_head": list(self.crop_head.parameters())}


# ---------------------------------------------------------------------------------------------------------- loss

def crop_ce(logits: torch.Tensor, yc: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the labelled pixels, in float32; 0 (with a graph) when a batch has none."""
    if not bool((yc != IGNORE_CROP).any()):
        return logits.float().sum() * 0.0
    return nn.functional.cross_entropy(logits.float(), yc, ignore_index=IGNORE_CROP)


def losses(seg, crop, y, yc, weights, crop_weight: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """``(total, border, crop)``: PRUE's log-cosh Dice + ``crop_weight`` x the crop head's cross-entropy."""
    lb = prue.log_cosh_dice(seg, y, weights)
    lc = crop_ce(crop, yc)
    return lb + float(crop_weight) * lc, lb, lc


def shared_grad_norms(params, lb: torch.Tensor, lc: torch.Tensor) -> tuple[float, float]:
    """Norms of the border loss's and the (unweighted) crop loss's gradients over ``params``, the graph kept for the
    training step's own backward pass; no ``.grad`` is touched."""
    gb = torch.autograd.grad(lb, params, retain_graph=True, allow_unused=True)
    gc = torch.autograd.grad(lc, params, retain_graph=True, allow_unused=True)

    def norm(gs):
        return float(torch.sqrt(sum((g.float() ** 2).sum() for g in gs if g is not None)))

    return norm(gb), norm(gc)


class Balance:
    """The crop loss's weight, set so that its pull on the shared parameters is ``target`` times the border loss's.

    Measured every ``every`` steps (and at the first); each new value ``target x |g_border| / |g_crop|`` is blended
    into the running weight in log space with ``smooth`` on the old value, and held in ``[lo, hi]``. ``pull`` is the
    ratio measured with the weight in force before the update: what the crop loss was actually doing.
    """

    def __init__(self, target: float = 0.3, every: int = 25, smooth: float = 0.5, lo: float = 1e-4,
                 hi: float = 10.0) -> None:
        self.target, self.every, self.smooth = float(target), int(every), float(smooth)
        self.lo, self.hi = float(lo), float(hi)
        self.weight: float | None = None
        self.log: list[tuple[float, float, float, float]] = []     # (|g_border|, |g_crop|, pull, weight)

    def due(self, step: int) -> bool:
        return self.weight is None or step % self.every == 0

    def update(self, gb: float, gc: float) -> None:
        if not (gb > 0 and gc > 0 and math.isfinite(gb) and math.isfinite(gc)):
            return
        pull = (self.weight or 0.0) * gc / gb
        new = min(max(self.target * gb / gc, self.lo), self.hi)
        self.weight = new if self.weight is None else math.exp(
            self.smooth * math.log(self.weight) + (1 - self.smooth) * math.log(new))
        self.log.append((gb, gc, pull, self.weight))

    def epoch_summary(self) -> dict:
        """Median of the measurements since the last call, then cleared."""
        if not self.log:
            return {"balance/crop_weight": self.weight}
        a = np.asarray(self.log)
        self.log = []
        return {"balance/grad_border": float(np.median(a[:, 0])), "balance/grad_crop": float(np.median(a[:, 1])),
                "balance/pull": float(np.median(a[a[:, 2] > 0, 2])) if (a[:, 2] > 0).any() else None,
                "balance/crop_weight": float(np.median(a[:, 3])), "balance/n": int(len(a))}

    def state_dict(self) -> dict:
        return {"weight": self.weight}

    def load_state_dict(self, st: dict) -> None:
        self.weight = st.get("weight")
