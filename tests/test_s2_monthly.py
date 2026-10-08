"""The monthly compositor must attach every acquisition's data to that acquisition."""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from gfm4agri.chips.s2_monthly import _align_on_ids


def _stack(ids, n_bands=2, offset=0.0):
    """A tiny stack whose every value encodes its acquisition, so misalignment is visible."""
    code = np.array([float(i.split("_")[1]) for i in ids]) + offset
    data = np.broadcast_to(code[:, None, None, None], (len(ids), n_bands, 2, 2)).copy()
    time = np.array([np.datetime64(f"2021-01-{10 + k:02d}", "ns") for k in range(len(ids))])
    return xr.DataArray(data, dims=("time", "band", "y", "x"),
                        coords={"time": time, "id": ("time", list(ids))})


def test_reordered_scene_classification_is_realigned():
    # Planetary Computer returns the newest acquisitions first; a stack sorted by date
    # reverses that. The pairing must follow the ids, not the positions.
    spec = _stack(["a_3", "a_2", "a_1"])
    scl = _stack(["a_1", "a_2", "a_3"], n_bands=1, offset=100.0)
    spec2, scl2, ids = _align_on_ids(spec, scl)
    assert ids == ["a_3", "a_2", "a_1"]
    assert np.array_equal(spec2.values[:, 0, 0, 0], [3, 2, 1])
    assert np.array_equal(scl2.values[:, 0, 0, 0], [103, 102, 101])


def test_acquisition_missing_from_one_stack_is_dropped_from_both():
    # stackstac silently leaves out acquisitions it finds empty over the window; one missing
    # acquisition must not shift the labels of the ones after it.
    spec = _stack(["a_4", "a_3", "a_2", "a_1"])
    scl = _stack(["a_4", "a_2", "a_1"], n_bands=1, offset=100.0)
    spec2, scl2, ids = _align_on_ids(spec, scl)
    assert ids == ["a_4", "a_2", "a_1"]
    assert np.array_equal(spec2.values[:, 0, 0, 0], [4, 2, 1])
    assert np.array_equal(scl2.values[:, 0, 0, 0], [104, 102, 101])


def test_duplicate_acquisition_is_rejected():
    with pytest.raises(RuntimeError, match="twice"):
        _align_on_ids(_stack(["a_1", "a_1"]), _stack(["a_1"], n_bands=1))
