"""The two-head networks and the crop-loss balance (gfm4agri.borders.networks)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("segmentation_models_pytorch")

from gfm4agri.borders import networks  # noqa: E402


def test_crop_target_only_on_drawn_in_scheme_parcels():
    mask = np.array([[0, 1], [-1, 1]])
    parcels = np.array([[1, 2], [3, 0]])
    assert networks.crop_target(mask, parcels).tolist() == [[0, 1], [-1, -1]]
    assert networks.crop_target(mask, parcels, keep=np.array([2])).tolist() == [[-1, 1], [-1, -1]]


def test_unet_joint_shapes_and_the_crop_loss_reaches_the_shared_part_not_the_border_head():
    torch.manual_seed(0)
    net = networks.UNetJoint(6, 4, encoder="resnet18", weights=None, hidden=(8,), dropout=0.0, head_dropout=0.0)
    x = torch.randn(2, 6, 32, 32)
    seg, crop = net(x)
    assert tuple(seg.shape) == (2, 3, 32, 32) and tuple(crop.shape) == (2, 4, 32, 32)
    yc = torch.randint(0, 4, (2, 32, 32))
    net.zero_grad()
    networks.crop_ce(crop, yc).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in net.shared_parameters())
    assert all(p.grad is None or float(p.grad.abs().sum()) == 0 for p in net.unet.segmentation_head.parameters())
    assert set(net.groups()) == {"unet", "crop_head"}


def test_token_joint_brings_the_decoder_to_the_chip_grid():
    class Neck(torch.nn.Module):
        def forward(self, feats, image_size):
            return feats

    class Decoder(torch.nn.Module):
        out_channels = 5

        def __init__(self):
            super().__init__()
            self.conv = torch.nn.Conv2d(3, 5, 1)

        def forward(self, feats):
            return self.conv(feats[-1])

    net = networks.TokenJoint(Neck(), Decoder(), [2, 3], (32, 32), 4, width=8, head_dropout=0.0)
    seg, crop = net(torch.randn(2, 5, 4, 4))
    assert tuple(seg.shape) == (2, 3, 32, 32) and tuple(crop.shape) == (2, 4, 32, 32)
    assert set(net.groups()) == {"decoder", "crop_head"}
    shared = {id(p) for p in net.shared_parameters()}
    assert not any(id(p) in shared for p in net.border_head.parameters())


def test_crop_ce_is_zero_with_a_graph_when_nothing_is_labelled():
    logits = torch.randn(1, 3, 2, 2, requires_grad=True)
    loss = networks.crop_ce(logits, torch.full((1, 2, 2), networks.IGNORE_CROP))
    assert float(loss.detach()) == 0.0
    loss.backward()


def test_shared_grad_norms_leave_grad_untouched():
    w = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    gb, gc = networks.shared_grad_norms([w], (w ** 2).sum(), 3 * w.sum())
    assert gb == pytest.approx(float(np.hypot(2.0, 4.0))) and gc == pytest.approx(float(np.hypot(3.0, 3.0)))
    assert w.grad is None


def test_balance_sets_the_weight_to_the_target_pull_and_smooths_it():
    b = networks.Balance(target=0.3, every=25, smooth=0.5)
    assert b.due(7)                                    # the first step always measures
    b.update(gb=2.0, gc=100.0)
    assert b.weight == pytest.approx(0.3 * 2.0 / 100.0)
    assert not b.due(7) and b.due(25)
    b.update(gb=2.0, gc=25.0)                          # new value 0.024, blended in log space with 0.006
    assert b.weight == pytest.approx(np.sqrt(0.006 * 0.024))
    assert b.log[-1][2] == pytest.approx(0.006 * 25.0 / 2.0)   # the pull under the weight in force
    s = b.epoch_summary()
    assert s["balance/n"] == 2 and not b.log
    b.update(gb=float("nan"), gc=1.0)                  # ignored
    assert b.weight == pytest.approx(np.sqrt(0.006 * 0.024))
    st = b.state_dict()
    c = networks.Balance()
    c.load_state_dict(st)
    assert c.weight == b.weight
