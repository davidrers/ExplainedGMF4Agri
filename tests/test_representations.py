"""Tests for the input-representation factor and the feature cache contract."""

from __future__ import annotations

from pathlib import Path

import pytest

from gfm4agri.data.representations import (
    BRIDGES,
    GROUP_OF,
    REPRESENTATIONS,
    REPRESENTATION_GROUPS,
    SPATIAL_SUPPORT,
    FeatureCacheKey,
    feature_cache_dir,
    group_of,
    required_buffer_m,
)
from gfm4agri.data.splits import SplitConfig


def test_every_representation_has_a_group_and_a_support() -> None:
    assert set(GROUP_OF) == set(REPRESENTATIONS)
    assert set(SPATIAL_SUPPORT) == set(REPRESENTATIONS)


def test_groups_partition_the_representations() -> None:
    """Groups must be disjoint and exhaustive, or a comparison could span two."""
    seen: list[str] = []
    for reps in REPRESENTATION_GROUPS.values():
        seen.extend(reps)
    assert sorted(seen) == sorted(REPRESENTATIONS)
    assert len(seen) == len(set(seen))


def test_a_group_holds_exactly_one_spatial_support() -> None:
    """This is what makes a within-group comparison an encoder comparison."""
    for group, reps in REPRESENTATION_GROUPS.items():
        supports = {SPATIAL_SUPPORT[r] for r in reps}
        assert len(supports) == 1, f"group {group} mixes spatial supports: {supports}"


def test_g1_is_the_parcel_support_group() -> None:
    assert {SPATIAL_SUPPORT[r] for r in REPRESENTATION_GROUPS["G1"]} == {"parcel_interior"}
    assert "ts_parcel" in REPRESENTATION_GROUPS["G1"]
    assert "emb_zonal" in REPRESENTATION_GROUPS["G1"]
    assert "patch_masked" in REPRESENTATION_GROUPS["G1"]


def test_every_bridge_crosses_a_group_boundary() -> None:
    """A bridge that stayed inside one group would measure nothing."""
    for name, (a, b) in BRIDGES.items():
        assert a in REPRESENTATIONS and b in REPRESENTATIONS
        assert group_of(a) != group_of(b), f"bridge {name} does not cross a group boundary"


def test_group_of_rejects_unknown_representation() -> None:
    with pytest.raises(KeyError, match="unknown input representation"):
        group_of("patch_enormous")


def test_required_buffer_is_half_the_patch_diagonal() -> None:
    assert required_buffer_m(224, 10.0) == pytest.approx(1583.92, abs=0.01)
    assert required_buffer_m(128, 10.0) == pytest.approx(905.10, abs=0.01)
    # Monotone in both arguments.
    assert required_buffer_m(256, 10.0) > required_buffer_m(224, 10.0)
    assert required_buffer_m(224, 20.0) > required_buffer_m(224, 10.0)


def test_required_buffer_rejects_bad_geometry() -> None:
    with pytest.raises(ValueError, match="must both be positive"):
        required_buffer_m(0, 10.0)
    with pytest.raises(ValueError, match="must both be positive"):
        required_buffer_m(224, 0.0)


def test_default_buffer_covers_the_declared_patch_footprint() -> None:
    """The protocol declares 224 px at 10 m; the default buffer must cover it."""
    assert SplitConfig().buffer_m >= required_buffer_m(224, 10.0)


def test_cache_key_carries_group_and_support() -> None:
    key = FeatureCacheKey(
        model="terramind",
        representation="patch_masked",
        model_version="v1",
        country_code="EE",
    )
    assert key.group == "G1"
    assert key.spatial_support == "parcel_interior"
    d = key.to_dict()
    assert d["representation_group"] == "G1"
    assert d["model"] == "terramind"


def test_cache_key_rejects_unknown_representation() -> None:
    with pytest.raises(KeyError, match="unknown input representation"):
        FeatureCacheKey(
            model="terramind", representation="nonsense", model_version="v1", country_code="EE"
        )


def test_one_model_yields_distinct_cache_entries_per_representation() -> None:
    """There is not one vector per parcel per model, but one per representation."""
    a = FeatureCacheKey("terramind", "patch_full", "v1", "EE")
    b = FeatureCacheKey("terramind", "patch_masked", "v1", "EE")
    assert a != b
    assert feature_cache_dir("data", a) != feature_cache_dir("data", b)


def test_cache_dir_layout_groups_by_representation_first() -> None:
    key = FeatureCacheKey("alphaearth", "emb_zonal", "V1_ANNUAL_2021", "LV")
    path = feature_cache_dir("data", key)
    assert path == Path("data") / "features" / "emb_zonal" / "alphaearth" / "V1_ANNUAL_2021"
