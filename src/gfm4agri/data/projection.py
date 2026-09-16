"""Pure-NumPy Lambert Azimuthal Equal Area projection (ETRS89-LAEA, EPSG:3035).

The spatial blocking of Phase 1 must be byte-identical on every machine that
reproduces a split.  Delegating the projection to ``pyproj`` would make the
block identifiers depend on the PROJ version installed, so the forward
projection is implemented here directly from Snyder (1987, pp. 187-190) using
only NumPy.  ``tests/test_projection.py`` cross-checks the implementation
against ``pyproj`` when that package is importable.

References
----------
Snyder, J. P. (1987). *Map projections: A working manual* (USGS Professional
Paper 1395). United States Government Printing Office.
"""

from __future__ import annotations

import numpy as np

# GRS80 ellipsoid, the datum of ETRS89.
GRS80_A = 6378137.0
GRS80_INV_F = 298.257222101
GRS80_F = 1.0 / GRS80_INV_F
GRS80_E2 = 2.0 * GRS80_F - GRS80_F * GRS80_F

# EPSG:3035 (ETRS89-extended / LAEA Europe).
EPSG3035_LAT0 = 52.0
EPSG3035_LON0 = 10.0
EPSG3035_X0 = 4321000.0
EPSG3035_Y0 = 3210000.0

__all__ = ["laea_forward", "to_epsg3035"]


def _authalic_q(sin_phi: np.ndarray, e2: float) -> np.ndarray:
    """Snyder eq. (3-12): the authalic-area function ``q``."""
    e = np.sqrt(e2)
    if e < 1e-12:  # sphere
        return 2.0 * sin_phi
    t = e * sin_phi
    return (1.0 - e2) * (
        sin_phi / (1.0 - t * t) - (1.0 / (2.0 * e)) * np.log((1.0 - t) / (1.0 + t))
    )


def laea_forward(
    lon_deg: np.ndarray,
    lat_deg: np.ndarray,
    lat0_deg: float,
    lon0_deg: float,
    false_easting: float,
    false_northing: float,
    a: float = GRS80_A,
    e2: float = GRS80_E2,
) -> tuple[np.ndarray, np.ndarray]:
    """Forward ellipsoidal Lambert Azimuthal Equal Area projection.

    Args:
        lon_deg: Longitudes in decimal degrees.
        lat_deg: Latitudes in decimal degrees.
        lat0_deg: Latitude of the projection origin, in decimal degrees.
        lon0_deg: Longitude of the projection origin, in decimal degrees.
        false_easting: False easting of the projection origin, in metres.
        false_northing: False northing of the projection origin, in metres.
        a: Semi-major axis of the ellipsoid, in metres.
        e2: Squared first eccentricity of the ellipsoid.

    Returns:
        A pair of arrays holding the eastings and northings, in metres.
    """
    lon = np.deg2rad(np.asarray(lon_deg, dtype=np.float64))
    lat = np.deg2rad(np.asarray(lat_deg, dtype=np.float64))
    lat0 = np.deg2rad(float(lat0_deg))
    lon0 = np.deg2rad(float(lon0_deg))

    sin_phi = np.sin(lat)
    sin_phi0 = np.sin(lat0)

    q = _authalic_q(sin_phi, e2)
    q0 = _authalic_q(np.asarray(sin_phi0), e2)
    qp = _authalic_q(np.asarray(1.0), e2)

    r_q = a * np.sqrt(qp / 2.0)
    beta = np.arcsin(np.clip(q / qp, -1.0, 1.0))
    beta0 = np.arcsin(np.clip(q0 / qp, -1.0, 1.0))

    m0 = np.cos(lat0) / np.sqrt(1.0 - e2 * sin_phi0 * sin_phi0)
    d = (a * m0) / (r_q * np.cos(beta0))

    dlon = lon - lon0
    cos_dlon = np.cos(dlon)
    denom = 1.0 + np.sin(beta0) * np.sin(beta) + np.cos(beta0) * np.cos(beta) * cos_dlon
    b = r_q * np.sqrt(2.0 / np.maximum(denom, 1e-15))

    x = false_easting + b * d * np.cos(beta) * np.sin(dlon)
    y = false_northing + (b / d) * (
        np.cos(beta0) * np.sin(beta) - np.sin(beta0) * np.cos(beta) * cos_dlon
    )
    return x, y


def to_epsg3035(lon_deg: np.ndarray, lat_deg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Project WGS-84 / ETRS89 geographic coordinates to EPSG:3035 metres."""
    return laea_forward(
        lon_deg,
        lat_deg,
        lat0_deg=EPSG3035_LAT0,
        lon0_deg=EPSG3035_LON0,
        false_easting=EPSG3035_X0,
        false_northing=EPSG3035_Y0,
    )
