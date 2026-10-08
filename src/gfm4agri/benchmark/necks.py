"""Necks the thesis adds to TerraTorch's registry.

Importing this module registers them, so an ``EncoderDecoderFactory`` config can name them
like any built-in neck.
"""

from __future__ import annotations

import torch
from torch import nn

from terratorch.models.necks import Neck
from terratorch.registry import TERRATORCH_NECK_REGISTRY

__all__ = ["ChannelBottleneck"]


class ChannelBottleneck(Neck):
    """Project every feature map to ``out_channels`` with a 1 x 1 convolution.

    A multi-temporal ViT whose months are stacked on the channel axis hands the neck
    ``T x embed_dim`` channels per layer, 12 x 1024 = 12,288 for Prithvi-EO-2.0-300M on the
    monthly grid. Everything downstream scales with that width, so without a projection the
    trainable neck and decoder dwarf the frozen encoder and cannot be matched across models.
    The projection mixes the months of each token position, so it is also where the decoder
    first learns which dates matter.
    """

    def __init__(self, channel_list: list[int], out_channels: int = 768) -> None:
        super().__init__(channel_list)
        self.out_channels = out_channels
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, out_channels, kernel_size=1, bias=False),
                          nn.BatchNorm2d(out_channels), nn.GELU())
            for c in channel_list)

    def forward(self, features: list[torch.Tensor], **kwargs) -> list[torch.Tensor]:
        return [p(f) for p, f in zip(self.proj, features)]

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        return [self.out_channels] * len(channel_list)


if "ChannelBottleneck" not in TERRATORCH_NECK_REGISTRY:
    TERRATORCH_NECK_REGISTRY.register(ChannelBottleneck)
