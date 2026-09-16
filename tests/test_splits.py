"""Tests for the Phase 1 splitting protocol.

The four properties the protocol depends on are asserted explicitly: a single
fixed test partition, spatial separation between partitions, nested support
sets, and independence of the support draw from anything to do with models or
heads.
"""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pandas as pd
import pytest

from gfm4agri.data.seeding import derived_rng, derived_seed
from gfm4agri.data.splits import (
    ALL_BUDGET,
    SplitConfig,
    build_split,
    build_support_order,
    intersect_label_space,
    load_official_eurocropsml_split,
    read_split_bundle,
    support_at,
    write_split_bundle,
)


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------


def test_derived_seed_is_stable_and_context_sensitive() -> None:
    assert derived_seed("support", 0, "EE", "3301010101", 3) == derived_seed(
        "support", 0, "EE", "3301010101", 3
    )
    assert derived_seed("support", 0, "EE") != derived_seed("support", 0, "LV")
    # Distinct tuples cannot collide by concatenation of their parts.
    assert derived_seed("ab", "c") != derived_seed("a", "bc")


def test_derived_rng_streams_match() -> None:
    a = derived_rng("x", 1).random(8)
    b = derived_rng("x", 1).random(8)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def test_config_hash_is_stable_and_sensitive(base_config: SplitConfig) -> None:
    assert base_config.config_hash() == dataclasses.replace(base_config).config_hash()
    changed = dataclasses.replace(base_config, buffer_m=2_000.0)
    assert changed.config_hash() != base_config.config_hash()
    assert base_config.protocol_id().startswith("test_protocol__")


def test_config_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="unknown support_sampling"):
        SplitConfig(support_sampling="nonsense")
    with pytest.raises(ValueError, match="must be the last entry"):
        SplitConfig(k_grid=(ALL_BUDGET, 1, 5))
    with pytest.raises(ValueError, match="n_draws"):
        SplitConfig(n_draws=0)


# ---------------------------------------------------------------------------
# Partitioning
# ---------------------------------------------------------------------------


def test_bundle_has_expected_partitions(built_bundle) -> None:
    for cc, cs in built_bundle.countries.items():
        present = set(cs.partition["partition"])
        assert {"pool", "val", "test"} <= present
        assert cs.classes
        assert len(cs.classes) >= 4
        assert cs.block_size_m == 10_000.0


def test_no_block_spans_two_partitions(built_bundle) -> None:
    for cs in built_bundle.countries.values():
        held = cs.partition.loc[cs.partition["partition"].isin(["test", "val", "pool"])]
        assert (held.groupby("block_id")["partition"].nunique() == 1).all()


def test_test_and_pool_blocks_are_disjoint(built_bundle) -> None:
    for cs in built_bundle.countries.values():
        test_blocks = set(cs.part("test")["block_id"])
        pool_blocks = set(cs.part("pool")["block_id"])
        val_blocks = set(cs.part("val")["block_id"])
        assert not (test_blocks & pool_blocks)
        assert not (test_blocks & val_blocks)
        assert not (val_blocks & pool_blocks)


def test_pool_parcels_are_beyond_the_buffer_from_test_blocks(built_bundle, base_config) -> None:
    """No retained pool parcel lies within the buffer of a held-out block."""
    size = base_config.block_size_m
    buf = base_config.buffer_m
    for cs in built_bundle.countries.values():
        held = cs.partition.loc[cs.partition["partition"].isin(["test", "val"])]
        held_cells = set(zip(held["block_ix"].tolist(), held["block_iy"].tolist()))
        pool = cs.part("pool")
        # Only the eight adjacent cells can be within the buffer.
        for row in pool.itertuples(index=False):
            neighbours = [
                (row.block_ix + dx, row.block_iy + dy)
                for dx in (-1, 0, 1)
                for dy in (-1, 0, 1)
                if (dx, dy) != (0, 0) and (row.block_ix + dx, row.block_iy + dy) in held_cells
            ]
            for bx, by in neighbours:
                dx_out = max(bx * size - row.x_m, row.x_m - (bx + 1) * size, 0.0)
                dy_out = max(by * size - row.y_m, row.y_m - (by + 1) * size, 0.0)
                assert np.hypot(dx_out, dy_out) >= buf


def test_buffer_actually_removes_parcels(built_bundle) -> None:
    assert any(cs.notes["n_buffered"] > 0 for cs in built_bundle.countries.values())


def test_test_partition_is_fixed_across_budgets_and_draws(built_bundle) -> None:
    """The test set is a property of the protocol, not of the budget or draw."""
    for cs in built_bundle.countries.values():
        reference = set(cs.part("test")["uid"])
        pool = cs.part("pool")
        for seed, order in cs.support_orders.items():
            for k in (1, 5, 20, ALL_BUDGET):
                support = support_at(order, k, pool=pool)
                assert not (set(support["uid"]) & reference)
        assert set(cs.part("test")["uid"]) == reference


def test_same_config_gives_identical_split(base_config, synthetic_catalogue, synthetic_scheme) -> None:
    a = build_split(base_config, catalogue=synthetic_catalogue, scheme=synthetic_scheme)
    b = build_split(base_config, catalogue=synthetic_catalogue, scheme=synthetic_scheme)
    for cc in a.countries:
        pd.testing.assert_frame_equal(a.countries[cc].partition, b.countries[cc].partition)
        for seed in a.countries[cc].support_orders:
            pd.testing.assert_frame_equal(
                a.countries[cc].support_orders[seed], b.countries[cc].support_orders[seed]
            )


def test_protocol_seed_changes_the_test_partition(
    base_config, synthetic_catalogue, synthetic_scheme
) -> None:
    other = dataclasses.replace(base_config, protocol_seed=1)
    a = build_split(base_config, catalogue=synthetic_catalogue, scheme=synthetic_scheme)
    b = build_split(other, catalogue=synthetic_catalogue, scheme=synthetic_scheme)
    assert set(a.countries["EE"].part("test")["uid"]) != set(b.countries["EE"].part("test")["uid"])


def test_test_set_keeps_natural_class_priors(built_bundle) -> None:
    """The test set is a spatial sample of the population, not class balanced."""
    cs = built_bundle.countries["EE"]
    test_counts = cs.part("test")["hcat"].value_counts(normalize=True)
    assert test_counts.max() > 2.0 * test_counts.min()


def test_classes_below_the_minimum_are_dropped_with_a_reason(
    synthetic_catalogue, synthetic_scheme
) -> None:
    strict = SplitConfig(
        name="strict",
        countries=("EE",),
        block_size_m=10_000.0,
        min_pool_per_class=100_000,
        min_test_per_class=1,
        max_pool_per_class=None,
        n_draws=1,
    )
    bundle = build_split(strict, catalogue=synthetic_catalogue, scheme=synthetic_scheme)
    cs = bundle.countries["EE"]
    assert cs.classes == ()
    assert cs.dropped_classes
    assert all("training pool holds" in r for r in cs.dropped_classes.values())


def test_test_partition_cap_is_applied(synthetic_catalogue, synthetic_scheme) -> None:
    capped = SplitConfig(
        name="capped",
        countries=("EE",),
        block_size_m=10_000.0,
        min_pool_per_class=50,
        min_test_per_class=10,
        max_test_parcels=500,
        max_pool_per_class=None,
        n_draws=1,
    )
    bundle = build_split(capped, catalogue=synthetic_catalogue, scheme=synthetic_scheme)
    cs = bundle.countries["EE"]
    assert len(cs.part("test")) <= 500
    assert cs.notes["test_subsampled_to"] == 500


def test_pool_cap_is_applied_per_class(synthetic_catalogue, synthetic_scheme) -> None:
    capped = SplitConfig(
        name="poolcap",
        countries=("EE",),
        block_size_m=10_000.0,
        min_pool_per_class=50,
        min_test_per_class=10,
        max_pool_per_class=300,
        n_draws=1,
    )
    bundle = build_split(capped, catalogue=synthetic_catalogue, scheme=synthetic_scheme)
    counts = bundle.countries["EE"].part("pool")["hcat"].value_counts()
    assert counts.max() <= 300


# ---------------------------------------------------------------------------
# Support sets
# ---------------------------------------------------------------------------


def test_support_sets_are_nested_across_budgets(built_bundle) -> None:
    for cs in built_bundle.countries.values():
        pool = cs.part("pool")
        for order in cs.support_orders.values():
            previous: set[str] = set()
            for k in (1, 5, 10, 20, 50):
                current = set(support_at(order, k)["uid"])
                assert previous <= current
                previous = current
            assert previous <= set(support_at(order, ALL_BUDGET, pool=pool)["uid"])


def test_support_is_exactly_k_per_class_when_the_pool_allows(built_bundle) -> None:
    for cs in built_bundle.countries.values():
        pool_counts = cs.part("pool")["hcat"].value_counts()
        for order in cs.support_orders.values():
            for k in (1, 5, 20, 50):
                counts = support_at(order, k)["hcat"].value_counts()
                for cls in cs.classes:
                    expected = min(k, int(pool_counts.get(cls, 0)))
                    assert int(counts.get(cls, 0)) == expected


def test_support_draws_differ_between_seeds(built_bundle) -> None:
    cs = built_bundle.countries["EE"]
    sets = [frozenset(support_at(o, 20)["uid"]) for o in cs.support_orders.values()]
    assert len(set(sets)) == len(sets)


def test_support_depends_only_on_the_declared_context(built_bundle, base_config) -> None:
    """No model, head or iteration order can influence which labels are drawn."""
    cs = built_bundle.countries["EE"]
    pool = cs.part("pool")
    shuffled = pool.sample(frac=1.0, random_state=99).reset_index(drop=True)
    a = build_support_order(pool, cs.classes, base_config, "EE", 0)
    b = build_support_order(shuffled, cs.classes, base_config, "EE", 0)
    c = build_support_order(shuffled, tuple(reversed(cs.classes)), base_config, "EE", 0)
    key = ["uid", "hcat", "rank"]
    pd.testing.assert_frame_equal(
        a[key].sort_values(key).reset_index(drop=True),
        b[key].sort_values(key).reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(
        a[key].sort_values(key).reset_index(drop=True),
        c[key].sort_values(key).reset_index(drop=True),
    )


def test_support_comes_only_from_the_pool(built_bundle) -> None:
    for cs in built_bundle.countries.values():
        pool_uids = set(cs.part("pool")["uid"])
        for order in cs.support_orders.values():
            assert set(order["uid"]) <= pool_uids


def test_block_spread_beats_random_on_block_diversity(
    built_bundle, base_config, synthetic_catalogue, synthetic_scheme
) -> None:
    """Block spreading must make K labels come from K distinct field blocks."""
    random_cfg = dataclasses.replace(base_config, support_sampling="random")
    random_bundle = build_split(
        random_cfg, catalogue=synthetic_catalogue, scheme=synthetic_scheme
    )
    spread = built_bundle.countries["EE"].support_orders[0]
    plain = random_bundle.countries["EE"].support_orders[0]

    def mean_blocks(order: pd.DataFrame, k: int) -> float:
        sub = support_at(order, k)
        return float(sub.groupby("hcat")["block_id"].nunique().mean())

    assert mean_blocks(spread, 20) > mean_blocks(plain, 20)


def test_block_spread_uses_every_available_block_first(built_bundle) -> None:
    cs = built_bundle.countries["EE"]
    pool = cs.part("pool")
    order = cs.support_orders[0]
    for cls in cs.classes:
        n_blocks = pool.loc[pool["hcat"] == cls, "block_id"].nunique()
        k = min(5, n_blocks)
        sub = support_at(order, k)
        got = sub.loc[sub["hcat"] == cls, "block_id"].nunique()
        assert got == k


def test_support_at_validates_its_arguments(built_bundle) -> None:
    order = built_bundle.countries["EE"].support_orders[0]
    with pytest.raises(ValueError, match="must be positive"):
        support_at(order, 0)
    with pytest.raises(ValueError, match="requires the training pool"):
        support_at(order, ALL_BUDGET)


def test_full_budget_support_is_the_whole_pool(built_bundle) -> None:
    cs = built_bundle.countries["EE"]
    pool = cs.part("pool")
    assert len(support_at(cs.support_orders[0], ALL_BUDGET, pool=pool)) == len(pool)


# ---------------------------------------------------------------------------
# Transfer
# ---------------------------------------------------------------------------


def test_intersection_is_ordered_by_the_target(built_bundle) -> None:
    inter = intersect_label_space(built_bundle, "LV", "EE")
    tgt = built_bundle.countries["EE"].classes
    assert set(inter) <= set(tgt)
    assert set(inter) <= set(built_bundle.countries["LV"].classes)
    assert list(inter) == [c for c in tgt if c in set(inter)]


def test_intersection_is_symmetric_in_membership(built_bundle) -> None:
    a = set(intersect_label_space(built_bundle, "LV", "EE"))
    b = set(intersect_label_space(built_bundle, "EE", "LV"))
    assert a == b


def test_many_to_one_intersection_is_no_larger(built_bundle) -> None:
    one = set(intersect_label_space(built_bundle, "LV", "EE"))
    many = set(intersect_label_space(built_bundle, ("LV", "EE"), "EE"))
    assert many <= set(built_bundle.countries["EE"].classes)
    assert len(many) >= len(one & set(built_bundle.countries["EE"].classes)) - len(one)


def test_min_source_pool_shrinks_the_label_space(built_bundle) -> None:
    loose = intersect_label_space(built_bundle, "LV", "EE")
    tight = intersect_label_space(built_bundle, "LV", "EE", min_source_pool=10**9)
    assert set(tight) <= set(loose)
    assert tight == ()


# ---------------------------------------------------------------------------
# Artefacts
# ---------------------------------------------------------------------------


def test_bundle_round_trips_through_disk(built_bundle, tmp_path) -> None:
    out = write_split_bundle(built_bundle, tmp_path)
    assert (out / "manifest.json").is_file()
    back = read_split_bundle(out)

    assert back.config == built_bundle.config
    assert back.config.config_hash() == built_bundle.config.config_hash()
    for cc, cs in built_bundle.countries.items():
        pd.testing.assert_frame_equal(back.countries[cc].partition, cs.partition)
        assert back.countries[cc].classes == cs.classes
        assert back.countries[cc].block_size_m == cs.block_size_m
        for seed, order in cs.support_orders.items():
            pd.testing.assert_frame_equal(back.countries[cc].support_orders[seed], order)


def test_manifest_carries_the_required_provenance(built_bundle) -> None:
    manifest = built_bundle.manifest()
    assert manifest["k_definition"] == "samples_per_class"
    for key in ("protocol_id", "config_hash", "config", "catalogue", "class_scheme", "countries"):
        assert key in manifest
    ee = manifest["countries"]["EE"]
    for key in ("block_size_m", "classes", "dropped_classes", "partition_sizes", "n_blocks_test"):
        assert key in ee
    json.dumps(manifest)  # must be serialisable without a custom encoder


def test_official_split_reader(tmp_path) -> None:
    path = tmp_path / "region_split_20.json"
    path.write_text(
        json.dumps(
            {
                "train": ["EE001_1_3302000000.npz"],
                "val": ["EE002_2_3301010101.npz"],
                "test": ["EE003_3_3301010102.npz"],
            }
        ),
        encoding="utf-8",
    )
    got = load_official_eurocropsml_split(path)
    assert got["train"] == ["EE001_1_3302000000"]
    assert set(got) == {"train", "val", "test"}


def test_official_split_reader_rejects_bad_structure(tmp_path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"train": 5}), encoding="utf-8")
    with pytest.raises(ValueError, match="is not a list"):
        load_official_eurocropsml_split(path)


# ---------------------------------------------------------------------------
# Estimated block size
# ---------------------------------------------------------------------------


def test_estimated_block_size_is_recorded(synthetic_catalogue, synthetic_scheme) -> None:
    cfg = SplitConfig(
        name="estimated",
        countries=("EE",),
        block_size_m=None,
        min_pool_per_class=50,
        min_test_per_class=10,
        max_pool_per_class=None,
        n_draws=1,
    )
    bundle = build_split(cfg, catalogue=synthetic_catalogue, scheme=synthetic_scheme)
    cs = bundle.countries["EE"]
    assert cs.notes["block_size_source"] == "estimated"
    assert cs.block_size_m >= 2_000.0
    assert cs.autocorr is not None
    assert cs.summary()["autocorr"]["note"]
