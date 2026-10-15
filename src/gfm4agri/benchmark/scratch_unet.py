"""The baseline of RQ1: a plain 2D U-Net trained from scratch on the monthly Sentinel-2 and Sentinel-1 composites.

Time enters by early fusion. The datamodule hands over the stack as ``(B, C, T, H, W)``, with the S1 bands
interleaved behind the S2 bands in every month (14 bands by 12 months for ``unet_r18_scratch_s2s1``); the model
flattens it, months outermost, into the ``T x C`` channels of one 2D image (168). Every filter of the first
convolution then has a weight of its own for each band in each month, which is where the network learns the
phenology; everything after it is spatial. The monthly composites are on a fixed, gap-filled time grid, so channel
position always means the same month.

The network is ``segmentation_models_pytorch``'s ``Unet`` with a ResNet-18 encoder, randomly initialised, about
15 M parameters, close to the 12.9 M UNet decoder of the token-grid arms. Encoder, decoder and head are kept as
attributes of their own, so ``count_params`` reports them like any other arm, and the head applies the arm's
``head_dropout`` before the final 1 x 1 convolution, as TerraTorch's segmentation head does. Everything is trained;
nothing is frozen. Tested on S2 alone in ``experiments/2026-10-14_unet_baseline``, from where it was moved.

Importing this module registers :class:`ScratchUNetFactory` with TerraTorch, so a ``SemanticSegmentationTask`` builds
the model from plain ``model_args``. Handing the task a built model instead stores the module object in the
checkpoint's hyperparameters, which Lightning's weights-only loading then refuses when it tests the best checkpoint.
"""

from __future__ import annotations

import torch
from torch import nn

from terratorch.registry import MODEL_FACTORY_REGISTRY

__all__ = ["ScratchUNet", "ScratchUNetFactory"]


class ScratchUNet(nn.Module):
    """``(B, C, T, H, W)`` monthly stack, or ``(B, C, H, W)``, to class logits at the input resolution."""

    def __init__(self, in_channels: int, num_classes: int, *, encoder: str = "resnet18",
                 head_dropout: float = 0.0) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        net = smp.Unet(encoder_name=encoder, encoder_weights=None, in_channels=in_channels,
                       classes=num_classes)
        self.encoder = net.encoder
        self.decoder = net.decoder
        self.head = nn.Sequential(nn.Dropout(head_dropout), net.segmentation_head)
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor, **_):
        from terratorch.models.model import ModelOutput

        if x.dim() == 5:
            # (B, C, T, H, W) -> (B, T, C, H, W) -> (B, T * C, H, W): month m, band c at channel m * C + c.
            x = x.transpose(1, 2).flatten(1, 2)
        if x.shape[1] != self.in_channels:
            raise ValueError(f"expected {self.in_channels} input channels, got {x.shape[1]}")
        return ModelOutput(self.head(self.decoder(self.encoder(x))))


class ScratchUNetFactory:
    """TerraTorch model factory: ``model_args`` to a :class:`ScratchUNet`, for segmentation only."""

    def build_model(self, task: str, aux_decoders=None, **model_args) -> ScratchUNet:
        if task != "segmentation" or aux_decoders:
            raise ValueError(f"ScratchUNet builds a segmentation model without auxiliary heads, not {task!r}")
        return ScratchUNet(**model_args)


if "ScratchUNetFactory" not in MODEL_FACTORY_REGISTRY:
    MODEL_FACTORY_REGISTRY.register(ScratchUNetFactory)
