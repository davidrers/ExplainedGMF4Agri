"""Field and border scores of the two-head workflow (gfm4agri.borders.scoring)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("skimage")

from gfm4agri.borders import prue, scoring  # noqa: E402


def _chip():
    parcels = np.zeros((16, 16), np.int64)
    parcels[2:14, 2:8], parcels[2:14, 8:14] = 1, 2
    mask = np.where(parcels > 0, np.where(parcels == 1, 0, 1), -1)
    wc = np.full((16, 16), 10, np.uint8)
    return mask, parcels, wc


def _perfect_prob(mask, parcels, wc):
    y = prue.target(mask, parcels, wc)
    prob = np.zeros((3, *y.shape), np.uint8)
    for c in range(3):
        prob[c][y == c] = 255
    return prob


def _grow_quality(prob, mask, parcels, wc):
    acc: dict = {}
    scoring._add(acc, scoring.chip_parts(prob, mask, parcels, wc, mask.clip(0), 2))
    aware, agnostic, extra = acc["chip/grow"]
    return scoring.quality(aware, agnostic, ["a", "b"], extra), acc


def test_the_declared_parcels_score_one_against_themselves():
    mask, parcels, wc = _chip()
    E, scored, true_border = scoring.regions(mask, parcels, wc)
    types = scoring.border_types(parcels, mask, wc, true_border)
    c = scoring.border_counts(parcels, true_border, scored, E, types)
    for tp_p, n_p, tp_r, n_r in c["tol"]:
        assert tp_p == n_p and tp_r == n_r and n_r > 0
    assert c["places"].sum() == 0


def test_border_types_tell_other_crop_from_non_farm_land():
    mask, parcels, wc = _chip()
    _, _, true_border = scoring.regions(mask, parcels, wc)
    types = scoring.border_types(parcels, mask, wc, true_border)
    assert types[5, 7] == 1 and types[5, 8] == 1        # parcel 1 (class 0) against parcel 2 (class 1)
    assert types[5, 2] == 3                              # against trees
    assert (types[~true_border] == -1).all()


def test_a_perfect_prediction_has_pq_one_class_agnostic_and_class_aware():
    mask, parcels, wc = _chip()
    q, acc = _grow_quality(_perfect_prob(mask, parcels, wc), mask, parcels, wc)
    assert q["pq_agnostic"] == pytest.approx(1.0) and q["pq"] == pytest.approx(1.0)
    assert q["true_parcels"] == 2 and q["predicted_parcels"] == 2
    px = scoring._pixels(acc)
    assert px["boundary_argmax"]["f1"] == pytest.approx(1.0) and px["field_iou"] == pytest.approx(1.0)


def test_a_merged_prediction_misses_both_parcels():
    mask, parcels, wc = _chip()
    merged = np.where(parcels > 0, 1, 0)
    q, _ = _grow_quality(_perfect_prob(mask, merged, wc), mask, parcels, wc)
    assert q["tp_agnostic"] == 0 and q["fn_agnostic"] == 2 and q["fp_agnostic"] == 1


def test_a_field_mostly_on_void_is_dropped_not_a_false_positive():
    mask, parcels, wc = _chip()
    mask[parcels == 2] = -1                             # parcel 2 out of the scheme: void
    q, _ = _grow_quality(_perfect_prob(np.where(parcels > 0, 0, -1), parcels, wc), mask, parcels, wc)
    assert q["tp_agnostic"] == 1 and q["fp_agnostic"] == 0 and q["dropped_void"] == 1
