"""Tests for the pure-NumPy EPSG:3035 projection."""

from __future__ import annotations

import numpy as np
import pytest

from gfm4agri.data.projection import (
    EPSG3035_LAT0,
    EPSG3035_LON0,
    EPSG3035_X0,
    EPSG3035_Y0,
    to_epsg3035,
)


def test_origin_maps_to_false_origin() -> None:
    x, y = to_epsg3035(np.array([EPSG3035_LON0]), np.array([EPSG3035_LAT0]))
    assert x[0] == pytest.approx(EPSG3035_X0, abs=1e-6)
    assert y[0] == pytest.approx(EPSG3035_Y0, abs=1e-6)


def test_published_check_point() -> None:
    """The EPSG registry check point for ETRS89-LAEA, 50 N 5 E."""
    x, y = to_epsg3035(np.array([5.0]), np.array([50.0]))
    assert x[0] == pytest.approx(3962799.45, abs=0.01)
    assert y[0] == pytest.approx(2999718.85, abs=0.01)


def test_matches_pyproj() -> None:
    pyproj = pytest.importorskip("pyproj")
    lon = np.array([26.6, 24.0, -8.6, 10.0, 21.9])
    lat = np.array([58.4, 56.9, 39.5, 52.0, 57.1])
    x, y = to_epsg3035(lon, lat)
    transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:3035", always_xy=True)
    xr, yr = transformer.transform(lon, lat)
    assert np.max(np.abs(x - np.asarray(xr))) < 1e-3
    assert np.max(np.abs(y - np.asarray(yr))) < 1e-3


def test_equal_area_property() -> None:
    """A degree cell near the study area maps to a plausible metric area."""
    lon = np.array([24.0, 24.1, 24.1, 24.0])
    lat = np.array([57.0, 57.0, 57.05, 57.05])
    x, y = to_epsg3035(lon, lat)
    area = 0.5 * abs(
        sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4))
    )
    # 0.1 degrees of longitude at 57 N is about 6.06 km; 0.05 of latitude about 5.56 km.
    assert 3.0e7 < area < 3.9e7
