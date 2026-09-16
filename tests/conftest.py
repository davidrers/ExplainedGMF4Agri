"""Shared fixtures for the gfm4agri test suite.

The project is not installed as a distribution, so ``src`` is placed on the
import path here rather than through packaging metadata.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _make_clustered_catalogue(
    *,
    country: str = "Estonia",
    country_prefix: str = "EE",
    n_clusters: int = 220,
    parcels_per_cluster: int = 60,
    cluster_radius_deg: float = 0.012,
    lon_range: tuple[float, float] = (22.0, 28.0),
    lat_range: tuple[float, float] = (57.6, 59.5),
    classes: tuple[str, ...] = (
        "3302000000",
        "3301010101",
        "3301010102",
        "3301010402",
        "3301010500",
        "3301060401",
    ),
    class_probs: tuple[float, ...] | None = None,
    seed: int = 0,
) -> pd.DataFrame:
    """Build a synthetic catalogue with spatially clustered crop labels.

    Each cluster stands for a field block: it draws a single dominant class
    and fills ninety per cent of its parcels with it, which reproduces the
    short-range label autocorrelation that the blocking is designed to
    control.
    """
    rng = np.random.default_rng(seed)
    if class_probs is None:
        weights = np.array([0.45, 0.18, 0.12, 0.10, 0.09, 0.06])
        class_probs = tuple(weights[: len(classes)] / weights[: len(classes)].sum())
    probs = np.asarray(class_probs, dtype=float)
    probs = probs / probs.sum()

    rows = []
    for c in range(n_clusters):
        c_lon = rng.uniform(*lon_range)
        c_lat = rng.uniform(*lat_range)
        dominant = rng.choice(len(classes), p=probs)
        nuts = f"{country_prefix}{c % 5 + 1:03d}"
        for p in range(parcels_per_cluster):
            if rng.random() < 0.90:
                cls = classes[dominant]
            else:
                cls = classes[rng.choice(len(classes), p=probs)]
            rows.append(
                (
                    country,
                    nuts,
                    f"{c:05d}{p:04d}",
                    cls,
                    int(rng.integers(20, 60)),
                    c_lat + rng.normal(0.0, cluster_radius_deg),
                    c_lon + rng.normal(0.0, cluster_radius_deg),
                )
            )
    df = pd.DataFrame.from_records(
        rows,
        columns=["country", "nuts", "parcel_id", "hcat", "n_timesteps", "lat", "lon"],
    )
    df["path"] = df["nuts"] + "_" + df["parcel_id"] + "_" + df["hcat"] + ".npz"
    return df


@pytest.fixture(scope="session")
def synthetic_catalogue_raw() -> pd.DataFrame:
    """A two-country synthetic catalogue, before validation."""
    ee = _make_clustered_catalogue(seed=1)
    lv = _make_clustered_catalogue(
        country="Latvia",
        country_prefix="LV",
        n_clusters=200,
        lon_range=(21.5, 27.5),
        lat_range=(56.0, 57.9),
        classes=(
            "3302000000",
            "3301010101",
            "3301010102",
            "3301010500",
            "3301110000",
            "3301990000",
        ),
        seed=2,
    )
    return pd.concat([ee, lv], ignore_index=True)


@pytest.fixture(scope="session")
def synthetic_catalogue(synthetic_catalogue_raw: pd.DataFrame) -> pd.DataFrame:
    """A validated two-country synthetic catalogue."""
    from gfm4agri.data.catalogue import validate_catalogue

    df = validate_catalogue(synthetic_catalogue_raw)
    df.attrs["catalogue_source"] = "synthetic"
    df.attrs["catalogue_degraded"] = False
    return df


@pytest.fixture(scope="session")
def synthetic_scheme(synthetic_catalogue: pd.DataFrame):
    """A frequency-derived class scheme over the synthetic catalogue."""
    from gfm4agri.data.class_scheme import derive_class_scheme

    return derive_class_scheme(synthetic_catalogue, n_classes=20, min_parcels=200)


@pytest.fixture()
def base_config():
    """A split configuration small enough to build quickly in tests."""
    from gfm4agri.data.splits import ALL_BUDGET, SplitConfig

    return SplitConfig(
        name="test_protocol",
        countries=("EE", "LV"),
        block_size_m=10_000.0,
        block_size_source="fixed",
        buffer_m=1_000.0,
        min_pool_per_class=100,
        min_test_per_class=50,
        max_test_parcels=100_000,
        max_pool_per_class=None,
        k_grid=(1, 5, 10, 20, 50, ALL_BUDGET),
        n_draws=3,
    )


@pytest.fixture()
def built_bundle(base_config, synthetic_catalogue, synthetic_scheme):
    """A split bundle over the synthetic catalogue."""
    from gfm4agri.data.splits import build_split

    return build_split(base_config, catalogue=synthetic_catalogue, scheme=synthetic_scheme)
