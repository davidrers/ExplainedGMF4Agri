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


# --- Screening ---------------------------------------------------------------------------

from gfm4agri.chips.s2_monthly import (  # noqa: E402
    BANDS, EXPORT_2021, REVISED, Screening, _merge_overpasses, _screen)
from gfm4agri.data.sentinel import SCL_DROP  # noqa: E402


def _scene(scl, n=1, refl=None):
    """``n`` acquisitions of one SCL map, with vegetation-like reflectance unless given."""
    c = np.broadcast_to(np.asarray(scl, np.float32), (n, *np.shape(scl))).copy()
    veg = {"B02": 0.03, "B03": 0.06, "B04": 0.03, "B08": 0.35, "B11": 0.15}
    veg.update(refl or {})
    s = np.full((n, len(BANDS), *c.shape[1:]), 0.05, np.float32)
    for b, v in veg.items():
        s[:, BANDS.index(b)] = v
    return s * 10000.0, c


def test_export_screening_is_scl_drop_alone():
    # The default must reproduce the screening the Estonian chip set was exported with.
    scl = np.arange(12, dtype=np.float32).reshape(3, 4)
    s, c = _scene(scl)
    s[0, 0, 2, 3] = np.nan
    nodata, flagged = _screen(s, c, EXPORT_2021)
    expected = np.isin(c, SCL_DROP) | np.isnan(s).any(1)
    assert np.array_equal(nodata | flagged, expected)


def test_revised_screening_drops_dark_area_and_unclassified():
    s, c = _scene([[2, 7, 4, 5, 6]])
    _, flagged = _screen(s, c, Screening(drop=REVISED.drop))
    assert flagged[0, 0].tolist() == [True, True, False, False, False]


def test_cloud_is_grown_by_dilate_px():
    scl = np.full((1, 9), 4.0)
    scl[0, 4] = 9
    s, c = _scene(scl)
    _, flagged = _screen(s, c, Screening(dilate_px=2))
    assert flagged[0, 0].tolist() == [False, False, True, True, True, True, True, False, False]


def test_snow_and_haze_tests_flag_what_scl_calls_clear():
    s, c = _scene([[4, 4, 4]])
    snow = {"B02": 0.6, "B03": 0.6, "B04": 0.6, "B08": 0.55, "B11": 0.1}
    haze = {"B02": 0.18, "B03": 0.15, "B04": 0.12}
    for k, refl in ((1, snow), (2, haze)):
        for b, v in refl.items():
            s[0, BANDS.index(b), 0, k] = v * 10000.0
    _, f_snow = _screen(s, c, Screening(snow_test=True))
    _, f_haze = _screen(s, c, Screening(haze_test=True))
    assert f_snow[0, 0].tolist() == [False, True, False]
    assert f_haze[0, 0].tolist()[0] is False and f_haze[0, 0, 2]


def test_tile_copies_of_one_overpass_become_one_observation():
    # Two tiles of one overpass and a second overpass. Pixel 0 is clear everywhere; pixel 1
    # is flagged in one tile copy only, which flags the overpass; pixel 2 lies outside the
    # first tile, so the overpass takes the second tile's value alone.
    s = np.zeros((3, 1, 1, 3), np.float32)
    s[:, 0, 0] = [[10, 10, 10], [20, 20, 20], [50, 50, 50]]
    nodata = np.zeros((3, 1, 3), bool)
    nodata[0, 0, 2] = True
    flagged = np.zeros((3, 1, 3), bool)
    flagged[1, 0, 1] = True
    keys = [("S2A", "t1"), ("S2A", "t1"), ("S2B", "t2")]
    vals, bad = _merge_overpasses(s, nodata, flagged, keys)
    assert vals.shape == (2, 1, 1, 3)
    assert bad[:, 0].tolist() == [[False, True, False], [False, False, False]]
    assert vals[0, 0, 0, 0] == 15 and vals[0, 0, 0, 2] == 20 and vals[1, 0, 0, 0] == 50
