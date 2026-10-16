"""Parcel-level scores: every parcel takes the class most of its labelled pixels were predicted as."""

from __future__ import annotations

import numpy as np
import pytest

from gfm4agri.benchmark.parcel_metrics import chip_votes, pool_votes, score_parcels, score_predictions

NAMES = ["a", "b", "c"]


def _onehot(pred: list[int], n: int = 3) -> np.ndarray:
    return np.eye(n, dtype=np.int64)[pred] * 5


def test_a_parcel_takes_the_class_most_of_its_pixels_were_predicted_as():
    parcels = np.array([[7, 7, 7, 8], [7, 0, 8, 8]])
    mask = np.array([[1, 1, 1, 2], [1, -1, 2, 2]])
    pred = np.array([[1, 0, 1, 2], [1, 2, 0, 0]])
    ids, votes, ref = chip_votes(pred, mask, parcels, 3)
    assert ids.tolist() == [7, 8] and ref.tolist() == [1, 2]
    assert votes.tolist() == [[1, 3, 0], [2, 0, 1]]
    assert score_parcels(votes, ref, NAMES)["accuracy"] == 0.5          # 7 is right, 8 is voted a


def test_unlabelled_pixels_and_pixels_outside_parcels_do_not_vote():
    parcels = np.array([[5, 5, 5, 0]])
    mask = np.array([[0, -1, -1, 0]])          # two pixels out of scheme, one outside any parcel
    pred = np.array([[0, 1, 1, 1]])
    ids, votes, _ = chip_votes(pred, mask, parcels, 3)
    assert ids.tolist() == [5] and votes.tolist() == [[1, 0, 0]]


def test_a_parcel_is_pooled_across_the_chips_it_touches():
    a = chip_votes(np.array([[1, 1, 1, 0]]), np.array([[0, 0, 0, 0]]), np.array([[9, 9, 9, 9]]), 3)
    b = chip_votes(np.array([[0, 0, 0, 0]]), np.array([[0, 0, 0, 0]]), np.array([[9, 9, 9, 9]]), 3)
    ids, votes, ref = pool_votes([a, b], 3)
    assert ids.tolist() == [9] and votes.tolist() == [[5, 3, 0]] and ref.tolist() == [0]
    assert score_parcels(votes, ref, NAMES)["accuracy"] == 1.0          # chip a alone would vote b


def test_ties_go_to_the_lowest_class_index():
    s = score_parcels(np.array([[0, 2, 2]]), np.array([2]), NAMES)
    assert s["accuracy"] == 0.0 and s["per_class_f1"]["b"] == 0.0


def test_a_parcel_under_two_classes_is_refused():
    with pytest.raises(ValueError, match="more than one class"):
        chip_votes(np.zeros((1, 2), int), np.array([[0, 1]]), np.array([[4, 4]]), 3)
    a = chip_votes(np.zeros((1, 1), int), np.array([[0]]), np.array([[4]]), 3)
    b = chip_votes(np.zeros((1, 1), int), np.array([[1]]), np.array([[4]]), 3)
    with pytest.raises(ValueError, match="different classes"):
        pool_votes([a, b], 3)


def test_the_scores_of_a_hand_worked_example():
    # Reference a, a, b, b, c voted a, b, b, b, a: F1 0.5, 0.8 and 0 for a, b and c.
    s = score_parcels(_onehot([0, 1, 1, 1, 0]), np.array([0, 0, 1, 1, 2]), NAMES)
    assert s["macro_f1"] == pytest.approx(1.3 / 3)
    assert s["accuracy"] == pytest.approx(0.6)
    assert s["kappa"] == pytest.approx((0.6 - 0.4) / 0.6)
    assert s["weighted_f1"] == pytest.approx(0.52)
    assert s["n_parcels"] == 5 and s["support"] == {"a": 2, "b": 2, "c": 1}
    assert s["per_class_f1"] == pytest.approx({"a": 0.5, "b": 0.8, "c": 0.0})


def test_a_class_neither_present_nor_predicted_is_left_out_of_the_mean():
    s = score_parcels(_onehot([0, 1]), np.array([0, 1]), NAMES)
    assert s["macro_f1"] == 1.0 and s["per_class_f1"]["c"] is None


def test_a_fits_stored_predictions_are_scored_against_the_parcel_rasters(split_chipset, tmp_path):
    """Two test chips, each with parcels of classes a, b, a and one out of scheme, 16 x 16 pixels each.
    Ten rows of chip 8's class-b parcel are predicted a, so that parcel is voted a."""
    ids = ["EE_00008_00000", "EE_00009_00000"]
    import rasterio

    masks = []
    for c in ids:
        with rasterio.open(split_chipset / "chips" / f"{c}.mask.tif") as src:
            masks.append(src.read(1).astype(np.int16))
    mask = np.stack(masks)
    pred = np.clip(mask, 0, None).astype(np.uint8)
    pred[0, :10, 16:] = 0
    f = tmp_path / "predictions_test.npz"
    np.savez_compressed(f, ids=np.array(ids), pred=pred, mask=mask)
    s = score_predictions(f, split_chipset / "chips", ["a", "b"])
    assert s["n_parcels"] == 6 and s["support"] == {"a": 4, "b": 2}
    assert s["accuracy"] == pytest.approx(5 / 6)
    assert s["macro_f1"] == pytest.approx((8 / 9 + 2 / 3) / 2)
    # The pixel Macro-F1 of the same predictions, to check them against the one the fit logged.
    assert s["pixel_macro_f1"] == pytest.approx((2048 / 2208 + 704 / 864) / 2)
    assert "majority vote" in s["rule"]
