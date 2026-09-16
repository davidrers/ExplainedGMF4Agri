"""Tests for spatial blocking, buffering and the label-range estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gfm4agri.data.autocorr import estimate_label_range_m, label_agreement_decay
from gfm4agri.data.blocks import assign_blocks, assign_partitions, buffer_mask


def test_blocks_are_anchored_to_the_projection_origin() -> None:
    """The tessellation is a fixed global grid, not a data-dependent one."""
    lon = np.array([24.0, 24.0001, 26.5])
    lat = np.array([57.0, 57.0001, 58.9])
    a = assign_blocks(lon, lat, 10_000.0)
    b = assign_blocks(lon[:1], lat[:1], 10_000.0)
    assert a["block_id"].iloc[0] == b["block_id"].iloc[0]
    assert a["block_id"].iloc[0] == a["block_id"].iloc[1]
    assert a["block_id"].iloc[0] != a["block_id"].iloc[2]


def test_block_size_changes_the_tessellation() -> None:
    lon = np.array([24.0, 24.05])
    lat = np.array([57.0, 57.0])
    coarse = assign_blocks(lon, lat, 20_000.0)
    fine = assign_blocks(lon, lat, 1_000.0)
    assert coarse["block_id"].nunique() == 1
    assert fine["block_id"].nunique() == 2


def test_blocking_rejects_degraded_coordinates() -> None:
    with pytest.raises(ValueError, match="finite centroid coordinates"):
        assign_blocks(np.array([np.nan]), np.array([57.0]), 10_000.0)


def test_blocking_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        assign_blocks(np.array([24.0]), np.array([57.0]), 0.0)


def _toy_blocks(n: int = 4000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(22.0, 27.0, n)
    lat = rng.uniform(57.0, 59.0, n)
    return assign_blocks(lon, lat, 10_000.0)


def test_partitions_respect_block_boundaries() -> None:
    blocks = _toy_blocks()
    part = assign_partitions(
        blocks,
        test_fraction=0.2,
        val_fraction=0.1,
        protocol_seed=0,
        country_code="EE",
        block_size_m=10_000.0,
    )
    per_block = pd.DataFrame({"block_id": blocks["block_id"], "partition": part})
    assert (per_block.groupby("block_id")["partition"].nunique() == 1).all()


def test_partition_sizes_are_close_to_the_targets() -> None:
    blocks = _toy_blocks(n=20_000, seed=3)
    part = assign_partitions(
        blocks,
        test_fraction=0.2,
        val_fraction=0.1,
        protocol_seed=0,
        country_code="EE",
        block_size_m=10_000.0,
    )
    frac = part.value_counts(normalize=True)
    assert frac["test"] == pytest.approx(0.2, abs=0.03)
    assert frac["val"] == pytest.approx(0.1, abs=0.03)


def test_partition_depends_only_on_declared_context() -> None:
    blocks = _toy_blocks()
    kw = dict(test_fraction=0.2, val_fraction=0.1, block_size_m=10_000.0)
    a = assign_partitions(blocks, protocol_seed=0, country_code="EE", **kw)
    b = assign_partitions(blocks, protocol_seed=0, country_code="EE", **kw)
    c = assign_partitions(blocks, protocol_seed=1, country_code="EE", **kw)
    d = assign_partitions(blocks, protocol_seed=0, country_code="LV", **kw)
    assert a.equals(b)
    assert not a.equals(c)
    assert not a.equals(d)


def test_partition_rejects_impossible_fractions() -> None:
    blocks = _toy_blocks(n=500)
    with pytest.raises(ValueError, match="non-empty training pool"):
        assign_partitions(
            blocks,
            test_fraction=0.7,
            val_fraction=0.4,
            protocol_seed=0,
            country_code="EE",
            block_size_m=10_000.0,
        )


def test_buffer_flags_only_pool_parcels_near_held_out_blocks() -> None:
    blocks = _toy_blocks(n=8000, seed=5)
    part = assign_partitions(
        blocks,
        test_fraction=0.2,
        val_fraction=0.1,
        protocol_seed=0,
        country_code="EE",
        block_size_m=10_000.0,
    )
    flagged = buffer_mask(blocks, part, block_size_m=10_000.0, buffer_m=1_000.0)
    assert flagged.any()
    assert (part.to_numpy()[flagged] == "pool").all()

    held = set(
        zip(
            blocks["block_ix"][np.isin(part, ("test", "val"))].tolist(),
            blocks["block_iy"][np.isin(part, ("test", "val"))].tolist(),
        )
    )
    for idx in np.flatnonzero(flagged)[:200]:
        x, y = blocks["x_m"].iloc[idx], blocks["y_m"].iloc[idx]
        best = min(
            np.hypot(
                max(bx * 10_000.0 - x, x - (bx + 1) * 10_000.0, 0.0),
                max(by * 10_000.0 - y, y - (by + 1) * 10_000.0, 0.0),
            )
            for bx, by in held
        )
        assert best < 1_000.0


def test_buffer_zero_flags_nothing() -> None:
    blocks = _toy_blocks()
    part = assign_partitions(
        blocks,
        test_fraction=0.2,
        val_fraction=0.1,
        protocol_seed=0,
        country_code="EE",
        block_size_m=10_000.0,
    )
    assert not buffer_mask(blocks, part, block_size_m=10_000.0, buffer_m=0.0).any()


def test_buffer_rejects_width_at_or_above_block_size() -> None:
    blocks = _toy_blocks(n=500)
    part = assign_partitions(
        blocks,
        test_fraction=0.2,
        val_fraction=0.1,
        protocol_seed=0,
        country_code="EE",
        block_size_m=10_000.0,
    )
    with pytest.raises(ValueError, match="0 <= buffer_m < block_size_m"):
        buffer_mask(blocks, part, block_size_m=10_000.0, buffer_m=10_000.0)


def _clustered_field(radius_m: float, n_clusters: int = 400, per_cluster: int = 40, seed: int = 0):
    """Points clustered at a known scale, each cluster carrying one label."""
    rng = np.random.default_rng(seed)
    xs, ys, labels = [], [], []
    for c in range(n_clusters):
        cx = rng.uniform(0.0, 300_000.0)
        cy = rng.uniform(0.0, 300_000.0)
        lab = rng.integers(0, 5)
        xs.append(cx + rng.normal(0.0, radius_m, per_cluster))
        ys.append(cy + rng.normal(0.0, radius_m, per_cluster))
        labels.append(np.full(per_cluster, lab))
    return np.concatenate(xs), np.concatenate(ys), np.concatenate(labels)


def test_agreement_decay_detects_autocorrelation() -> None:
    x, y, lab = _clustered_field(radius_m=3_000.0, seed=11)
    decay = label_agreement_decay(x, y, lab, seed=0, d_min_m=500.0, d_max_m=60_000.0)
    assert decay.note == "ok"
    assert decay.practical_range_m is not None
    assert decay.excess_agreement[0] > decay.excess_agreement[-1]
    assert 0.15 < decay.chance_agreement < 0.35


def test_estimated_range_grows_with_the_cluster_scale() -> None:
    x_s, y_s, lab_s = _clustered_field(radius_m=2_000.0, seed=21)
    x_l, y_l, lab_l = _clustered_field(radius_m=10_000.0, seed=21)
    r_small, _ = estimate_label_range_m(x_s, y_s, lab_s, seed=0, min_block_m=1_000.0)
    r_large, _ = estimate_label_range_m(x_l, y_l, lab_l, seed=0, min_block_m=1_000.0)
    assert r_large > r_small


def test_range_falls_back_when_labels_are_spatially_random() -> None:
    rng = np.random.default_rng(7)
    x = rng.uniform(0.0, 300_000.0, 20_000)
    y = rng.uniform(0.0, 300_000.0, 20_000)
    lab = rng.integers(0, 5, 20_000)
    block, decay = estimate_label_range_m(x, y, lab, seed=0, fallback_m=12_000.0)
    assert decay.practical_range_m is None
    assert block == pytest.approx(12_000.0)


def test_range_estimate_is_reproducible() -> None:
    x, y, lab = _clustered_field(radius_m=4_000.0, seed=31)
    a, _ = estimate_label_range_m(x, y, lab, seed=0)
    b, _ = estimate_label_range_m(x, y, lab, seed=0)
    c, _ = estimate_label_range_m(x, y, lab, seed=1)
    assert a == b
    assert isinstance(c, float)


def test_range_is_rounded_and_clipped() -> None:
    x, y, lab = _clustered_field(radius_m=40_000.0, seed=41)
    block, _ = estimate_label_range_m(
        x, y, lab, seed=0, min_block_m=2_000.0, max_block_m=15_000.0, round_to_m=1_000.0
    )
    assert block <= 15_000.0
    assert block % 1_000.0 == 0.0
