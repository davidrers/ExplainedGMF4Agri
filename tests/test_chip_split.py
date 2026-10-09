"""The chip-level spatial block split: block nesting, whole-block assignment and the buffer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gfm4agri.data.chip_grid import CHIP_M, CHIP_PX
from gfm4agri.data.chip_split import (
    ChipSplitConfig,
    assign_chip_partitions,
    block_of_chip,
    buffered_parcels,
    chip_parcel_table,
)


def _synthetic(n_side: int = 16, seed: int = 0):
    """``n_side`` x ``n_side`` chips; each holds four quadrant parcels, plus one parcel cut by
    every vertical chip edge, so some parcels span two chips."""
    rng = np.random.default_rng(seed)
    chips, arrays, rows = [], {}, []
    next_id = 1
    edge_ids = {}
    for col in range(1000, 1000 + n_side):
        for row in range(2000, 2000 + n_side):
            cid = f"XX_{col:05d}_{row:05d}"
            ids = np.zeros((CHIP_PX, CHIP_PX), dtype=np.int32)
            mask = np.full((CHIP_PX, CHIP_PX), -1, dtype=np.int16)
            h = CHIP_PX // 2
            for qi, (r0, c0) in enumerate([(8, 8), (8, h), (h, 8), (h, h)]):
                ids[r0:r0 + 90, c0:c0 + 90] = next_id
                mask[r0:r0 + 90, c0:c0 + 90] = rng.integers(0, 3)
                next_id += 1
            # A parcel straddling this chip's east edge and the next chip's west edge.
            key = (col, row)
            if key not in edge_ids:
                edge_ids[key] = next_id
                next_id += 1
            ids[100:110, CHIP_PX - 4:] = edge_ids[key]
            mask[100:110, CHIP_PX - 4:] = 1
            if (col - 1, row) in edge_ids:
                ids[100:110, :4] = edge_ids[(col - 1, row)]
                mask[100:110, :4] = 1
            chips.append({"chip_id": cid, "col": col, "row": row})
            arrays[cid] = (mask, ids)
            rows.append(chip_parcel_table(cid, mask, ids))
    return pd.DataFrame(chips), arrays, pd.concat(rows, ignore_index=True)


@pytest.fixture(scope="module")
def synthetic():
    return _synthetic()


def _cfg(**kw) -> ChipSplitConfig:
    base = dict(name="t", country="XX", year=2021, block_chips=4, buffer_m=1600.0)
    return ChipSplitConfig(**{**base, **kw})


def test_block_edge_is_a_whole_number_of_chips():
    cfg = _cfg(block_chips=4)
    assert cfg.block_size_m == 4 * CHIP_M == 8960.0
    bx, by = block_of_chip(np.array([0, 3, 4, -1]), np.array([7, 8, -4, -5]), 4)
    assert bx.tolist() == [0, 0, 1, -1] and by.tolist() == [1, 2, -1, -2]


def test_config_hash_changes_with_every_knob():
    base = _cfg()
    for kw in ({"block_chips": 8}, {"buffer_m": 1000.0}, {"protocol_seed": 1},
               {"test_fraction": 0.25}, {"manifest_sha256": "x"}):
        assert _cfg(**kw).config_hash() != base.config_hash()
    assert _cfg().dir_name == base.dir_name


def test_whole_blocks_and_target_shares(synthetic):
    chips, _, parcels = synthetic
    part = assign_chip_partitions(chips, parcels, _cfg())
    blocks = pd.Series([f"{a}_{b}" for a, b in zip(*block_of_chip(chips["col"], chips["row"], 4))])
    assert (pd.DataFrame({"b": blocks, "p": part.to_numpy()}).groupby("b")["p"].nunique() == 1).all()
    share = part.value_counts(normalize=True)
    assert set(part) == {"test", "val", "pool"}
    # 16 blocks of equal weight: shares land on whole blocks, near 20 / 10 / 70.
    assert 0.12 <= share["test"] <= 0.32 and 0.05 <= share["val"] <= 0.2


def test_assignment_is_deterministic_and_seeded(synthetic):
    chips, _, parcels = synthetic
    a = assign_chip_partitions(chips, parcels, _cfg())
    shuffled = chips.sample(frac=1, random_state=3)
    b = assign_chip_partitions(shuffled, parcels.sample(frac=1, random_state=4), _cfg())
    assert dict(zip(chips["chip_id"], a)) == dict(zip(shuffled["chip_id"], b))
    c = assign_chip_partitions(chips, parcels, _cfg(protocol_seed=1))
    assert not a.equals(c)


def test_buffer_withholds_every_pool_parcel_near_a_held_out_block(synthetic):
    chips, arrays, parcels = synthetic
    cfg = _cfg()
    part = assign_chip_partitions(chips, parcels, cfg)
    withheld, report = buffered_parcels(chips, part, parcels, arrays, cfg)
    L = cfg.block_size_m
    bx, by = block_of_chip(chips["col"].to_numpy(), chips["row"].to_numpy(), 4)
    held = [(a, b) for a, b, p in zip(bx, by, part) if p in ("test", "val")]
    for cid, col, row, p in zip(chips["chip_id"], chips["col"], chips["row"], part):
        if p != "pool":
            continue
        mask, ids = arrays[cid]
        x0, y1 = col * CHIP_M, (row + 1) * CHIP_M
        offs = (np.arange(CHIP_PX) + 0.5) * 10.0
        x, y = np.meshgrid(x0 + offs, y1 - offs)
        for a, b in held:
            dx = np.maximum(np.maximum(a * L - x, x - (a + 1) * L), 0)
            dy = np.maximum(np.maximum(b * L - y, y - (b + 1) * L), 0)
            close = (np.hypot(dx, dy) < cfg.buffer_m) & (mask >= 0) & (ids > 0)
            assert set(np.unique(ids[close]).tolist()) <= withheld
    # No trainable parcel appears in any held-out chip.
    where = parcels["chip_id"].map(dict(zip(chips["chip_id"], part)))
    held_ids = set(parcels.loc[where.isin(["test", "val"]), "parcel_id"])
    pool_ids = set(parcels.loc[where == "pool", "parcel_id"])
    assert not ((pool_ids - withheld) & held_ids)
    assert withheld <= pool_ids
    assert report["parcels_within_buffer"] > 0


def test_zero_buffer_withholds_only_shared_parcels(synthetic):
    chips, arrays, parcels = synthetic
    cfg = _cfg(buffer_m=0.0)
    part = assign_chip_partitions(chips, parcels, cfg)
    withheld, report = buffered_parcels(chips, part, parcels, arrays, cfg)
    assert report["parcels_within_buffer"] == 0
    where = parcels["chip_id"].map(dict(zip(chips["chip_id"], part)))
    shared = (set(parcels.loc[where == "pool", "parcel_id"])
              & set(parcels.loc[where.isin(["test", "val"]), "parcel_id"]))
    assert withheld == shared and shared


def test_buffer_must_be_shorter_than_a_block(synthetic):
    chips, arrays, parcels = synthetic
    cfg = _cfg(block_chips=1, buffer_m=3000.0)
    part = assign_chip_partitions(chips, parcels, cfg)
    with pytest.raises(ValueError):
        buffered_parcels(chips, part, parcels, arrays, cfg)


def test_chip_parcel_table_counts_labelled_pixels_only():
    ids = np.zeros((CHIP_PX, CHIP_PX), np.int32)
    mask = np.full((CHIP_PX, CHIP_PX), -1, np.int16)
    ids[:10, :10] = 7
    mask[:10, :5] = 2          # half the parcel labelled
    ids[20:30, :10] = 9        # an out-of-scheme parcel, never labelled
    t = chip_parcel_table("c", mask, ids)
    assert t[["parcel_id", "class_index", "pixels"]].values.tolist() == [[7, 2, 50]]
