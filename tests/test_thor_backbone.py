"""THOR enters through its own TerraTorch extension, installed outside the Poetry lock."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("terratorch")
torch = pytest.importorskip("torch")
if importlib.util.find_spec("thor_terratorch_ext") is None:
    pytest.skip("THOR is not installed; see scripts/env/install_thor.sh", allow_module_level=True)

from gfm4agri.benchmark.backbones import (  # noqa: E402
    THOR_BANDS,
    THOR_BANDS_10M,
    THOR_GROUND_COVER_M,
    THOR_TOKEN_M,
    get_backbone,
)


_PATCH_CHECK = """
import logging, sys, torch
from timm.models.vision_transformer import Block
from gfm4agri.benchmark.backbones import THOR_BANDS, get_backbone

torch.manual_seed(0)
block = Block(64, 4, qkv_bias=True).eval()
x = torch.randn(2, 50, 64)
level = logging.getLogger().level
with torch.no_grad():
    before = block(x)
assert "thor" not in sys.modules
get_backbone("thor_v1_large").model_args(list(THOR_BANDS), 12)
assert "thor" in sys.modules
with torch.no_grad():
    after = block(x)
assert torch.equal(before, after), (before - after).abs().max()
assert logging.getLogger().level == level
"""


def test_importing_thor_leaves_a_stock_timm_block_and_the_log_level_unchanged():
    # THOR replaces the forward of timm's Attention and Block for the whole process. Without an
    # attention mask the replacement must compute what the stock forward does, or Prithvi,
    # which is built on timm's Block, would change whenever it shares a process with THOR.
    # A fresh process, because once THOR is imported the stock forward is gone.
    src = str(Path(__file__).resolve().parents[1] / "src")
    done = subprocess.run([sys.executable, "-c", _PATCH_CHECK], capture_output=True, text=True,
                          env={**os.environ, "PYTHONPATH": src}, timeout=1200)
    assert done.returncode == 0, done.stderr[-2000:]


@pytest.mark.parametrize("name, token_m", [("thor_v1_base", THOR_TOKEN_M),
                                           ("thor_v1_large", THOR_TOKEN_M),
                                           ("thor_v1_large_80m", 80)])
def test_both_band_groups_are_tokenised_on_the_same_grid(name, token_m):
    args = get_backbone(name).model_args(list(THOR_BANDS), 12)
    gsd = {b: 10 if b in THOR_BANDS_10M else 20 for b in THOR_BANDS}
    assert {args["backbone_patch_sizes"][b] * gsd[b] for b in THOR_BANDS} == {token_m}
    assert THOR_GROUND_COVER_M % token_m == 0
    # 4 px is the smallest patch THOR was pretrained with.
    assert min(args["backbone_patch_sizes"].values()) >= 4


def test_the_token_size_ablation_changes_the_patch_sizes_and_nothing_else():
    coarse = get_backbone("thor_v1_large").model_args(list(THOR_BANDS), 12)
    fine = get_backbone("thor_v1_large_80m").model_args(list(THOR_BANDS), 12)
    assert {k for k in coarse if coarse[k] != fine[k]} == {"backbone_patch_sizes"}
    assert {b: 2 * p for b, p in fine["backbone_patch_sizes"].items()} == coarse["backbone_patch_sizes"]
    assert get_backbone("thor_v1_large").stats(list(THOR_BANDS)) == \
        get_backbone("thor_v1_large_80m").stats(list(THOR_BANDS))


def test_pretraining_statistics_are_on_the_scale_of_the_chips():
    # THOR states its statistics in reflectance; the chips hold reflectance x 10000.
    means, stds = get_backbone("thor_v1_large").stats(["RED", "NIR_NARROW"])
    assert means == pytest.approx([2139.48, 3209.93])
    assert stds == pytest.approx([2591.80, 2232.74])
    with pytest.raises(KeyError, match="not pretrained"):
        get_backbone("thor_v1_large").stats(["CIRRUS"])


@pytest.mark.parametrize("token_m", [THOR_TOKEN_M, 80])
def test_the_decoder_receives_the_monthly_maps_of_both_groups(token_m):
    from terratorch.models import EncoderDecoderFactory

    import gfm4agri.benchmark.necks  # noqa: F401  registers ChannelBottleneck
    from gfm4agri.benchmark.backbones import _thor

    n_t = 2
    args = _thor("thor_v1_base", [2, 5, 8, 11], bottleneck=768, token_m=token_m)(list(THOR_BANDS), n_t)
    args["backbone_pretrained"] = False
    model = EncoderDecoderFactory().build_model(
        task="segmentation", **args, decoder="UNetDecoder",
        decoder_channels=[512, 256, 128, 64], num_classes=3).eval()
    x = torch.randn(1, len(THOR_BANDS), n_t, 224, 224)
    with torch.no_grad():
        feats = model.encoder(x)
        out = model(x).output
    grid = THOR_GROUND_COVER_M // token_m
    assert len(feats) == 12
    assert tuple(feats[0].shape) == (1, n_t * 2 * 768, grid, grid)
    assert tuple(out.shape) == (1, 3, 224, 224)
