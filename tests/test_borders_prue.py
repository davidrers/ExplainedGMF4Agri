"""PRUE's border target, loss and field post-processing in the core (gfm4agri.borders.prue)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("skimage")

from gfm4agri.borders import prue  # noqa: E402


def _two_parcels():
    parcels = np.zeros((8, 8), np.int64)
    parcels[:, :4], parcels[:, 4:] = 1, 2
    mask = np.where(parcels > 0, 0, -1)
    return mask, parcels


def test_boundary_map_is_two_pixels_between_parcels_and_none_at_the_chip_edge():
    _, parcels = _two_parcels()
    b = prue.boundary_map(parcels)
    assert b[:, 3].all() and b[:, 4].all()
    assert not b[:, :3].any() and not b[:, 5:].any()


def test_target_classes_and_budget():
    mask, parcels = _two_parcels()
    parcels[:2, :] = 0
    mask[:2, :] = -1
    wc = np.full((8, 8), 10, np.uint8)               # trees: background outside the parcels
    y = prue.target(mask, parcels, wc)
    assert (y[:2] == prue.BACKGROUND).all()
    assert y[5, 1] == prue.INTERIOR and y[5, 3] == prue.BOUNDARY
    assert y[2, 1] == prue.BOUNDARY                    # next to the undeclared rows
    y_k = prue.target(mask, parcels, wc, keep=np.array([1]))
    assert y_k[5, 1] == prue.INTERIOR and y_k[5, 6] == prue.BACKGROUND   # undrawn parcel on non-farm land
    wc[:] = 40                                         # cropland: unknown outside the drawn parcels
    assert prue.target(mask, parcels, wc, keep=np.array([1]))[5, 6] == prue.IGNORE


def test_d4_turns_every_array_alike():
    a = np.arange(16).reshape(4, 4)
    b, c = prue.d4([a, a[None]], 1, True)
    assert np.array_equal(b, c[0]) and np.array_equal(b, np.rot90(a, 1)[:, ::-1])


def test_log_cosh_dice_is_zero_for_a_perfect_prediction_and_ignores_the_ignore_class():
    y = torch.tensor([[[0, 1], [2, 3]]])
    logits = torch.full((1, 3, 2, 2), -20.0)
    for c in range(3):
        logits[0, c][y[0] == c] = 20.0
    assert float(prue.log_cosh_dice(logits, y)) == pytest.approx(0.0, abs=1e-6)
    ignored = logits.clone()
    ignored[0, :, 1, 1] = torch.tensor([20.0, -20.0, -20.0])          # the ignored pixel, any prediction
    assert float(prue.log_cosh_dice(ignored, y)) == pytest.approx(0.0, abs=1e-6)
    worse = logits.clone()
    worse[0, :, 0, 0] = torch.tensor([-20.0, 20.0, -20.0])
    assert float(prue.log_cosh_dice(worse, y)) > 0


def test_parcels_variants_split_two_fields():
    prob = np.zeros((3, 8, 8), np.uint8)
    prob[prue.INTERIOR] = 255
    prob[:, :, 3:5] = 0
    prob[prue.BOUNDARY, :, 3:5] = 255
    cc = prue.parcels(prob, "cc")
    grow = prue.parcels(prob, "grow")
    field = prue.parcels(prob, "field")
    assert cc.max() == 2 and (cc[:, 3:5] == 0).all()
    assert grow.max() == 2 and (grow > 0).all()
    assert field.max() == 1
    with pytest.raises(ValueError):
        prue.parcels(prob, "nope")


def test_parcels_drop_fields_under_five_pixels():
    prob = np.zeros((3, 8, 8), np.uint8)
    prob[prue.BACKGROUND] = 255
    prob[:, 0, :4] = 0
    prob[prue.INTERIOR, 0, :4] = 255                   # 4 pixels: under PRUE's 500 m2
    assert prue.parcels(prob, "cc").max() == 0
