"""Decoders the thesis adds to TerraTorch's registry.

Importing this module registers them, so an ``EncoderDecoderFactory`` config can name them
like any built-in decoder.
"""

from __future__ import annotations

import torch
from torch import nn

from terratorch.registry import TERRATORCH_DECODER_REGISTRY

__all__ = ["PixelMLPDecoder"]


class PixelMLPDecoder(nn.Module):
    """A per-pixel MLP, written as 1 x 1 convolutions, for 10 m embedding rasters.

    Every output pixel depends only on that pixel's embedding: no spatial context is added,
    so the score measures the embedding itself. This follows the head the TESSERA authors
    use for crop classification (``tessera-downstream-task``, ``austrian_crop``): two hidden
    layers of 512 and 256 units, each followed by ReLU, batch normalisation and dropout. The
    final projection to classes is TerraTorch's segmentation head, a 1 x 1 convolution, so
    the three together are that three-layer MLP.
    """

    includes_head = False

    def __init__(self, embed_dim: list[int], hidden: list[int] | None = None,
                 dropout: float = 0.2, in_index: int = -1) -> None:
        super().__init__()
        hidden = list(hidden or [512, 256])
        self.in_index = in_index
        layers, c = [], embed_dim[in_index]
        for h in hidden:
            layers += [nn.Conv2d(c, h, kernel_size=1), nn.ReLU(), nn.BatchNorm2d(h), nn.Dropout(dropout)]
            c = h
        self.mlp = nn.Sequential(*layers)
        self.out_channels = c

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        return self.mlp(features[self.in_index])


if "PixelMLPDecoder" not in list(TERRATORCH_DECODER_REGISTRY):
    TERRATORCH_DECODER_REGISTRY.register(PixelMLPDecoder)
