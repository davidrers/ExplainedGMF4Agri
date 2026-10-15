"""A 2D U-Net trained from scratch, here as the decoder on frozen TESSERA v2 Medium embeddings.

Copied unchanged in its code from experiments/2026-10-14_unet_baseline, where it is the from-scratch baseline on the
144 composite bands. Here its input is the 128-channel embedding raster, ``(B, C, H, W)``, which the forward pass
takes as it is.

The network is ``segmentation_models_pytorch``'s ``Unet`` with a ResNet-18 down path, randomly initialised, about
14 M parameters with 128 input channels, close to the 12.9 M UNet decoder of the token-grid arms. Its down path,
decoder and head are kept as attributes of their own, so ``count_params`` reports them like any other arm (the down
path under ``encoder``), and the head applies the arm's ``head_dropout`` before the final 1 x 1 convolution, as
TerraTorch's segmentation head does. Everything in it is trained; the TESSERA encoder upstream is frozen.

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
