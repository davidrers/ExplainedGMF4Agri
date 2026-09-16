"""Spatial blocking and the block-to-partition assignment.

Blocks are the cells of a regular square tessellation in ETRS89-LAEA
(EPSG:3035), anchored on the projection's false origin.  The tessellation is
therefore a fixed global grid rather than a data-dependent one: two runs on
different parcel subsets, or on different countries, agree on which block a
given location falls in, which makes block identifiers meaningful across
experiments and stable under any rebuild of the catalogue.

The partition assignment permutes whole blocks and walks the permutation,
filling the test partition, then the validation partition, and leaving the
remainder as the training pool.  Because the permutation is random rather than
spatially ordered, the test blocks are scattered across the country, which
preserves the environmental coverage of the test set while keeping every test
parcel spatially separated from the training pool (Roberts et al., 2017).  An
optional buffer removes training parcels lying within a fixed distance of any
held-out block, so that parcels straddling a block boundary cannot leak.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gfm4agri.data.projection import to_epsg3035
from gfm4agri.data.seeding import derived_rng

__all__ = [
    "PARTITIONS",
    "assign_blocks",
    "assign_partitions",
    "block_bounds",
    "buffer_mask",
]

#: Partition labels.  ``buffer`` parcels are excluded from every use.
PARTITIONS: tuple[str, ...] = ("test", "val", "pool", "buffer")


def assign_blocks(
    lon: np.ndarray,
    lat: np.ndarray,
    block_size_m: float,
) -> pd.DataFrame:
    """Assign each parcel to a cell of the EPSG:3035 square tessellation.

    Args:
        lon: Longitudes in decimal degrees.
        lat: Latitudes in decimal degrees.
        block_size_m: Edge length of the square blocks, in metres.

    Returns:
        A frame with columns ``x_m``, ``y_m``, ``block_ix``, ``block_iy`` and
        ``block_id``, in the order of the inputs.

    Raises:
        ValueError: If ``block_size_m`` is not positive or coordinates are not
            finite.
    """
    if not np.isfinite(block_size_m) or block_size_m <= 0:
        raise ValueError(f"block_size_m must be positive and finite, got {block_size_m!r}")
    lon = np.asarray(lon, dtype=np.float64).ravel()
    lat = np.asarray(lat, dtype=np.float64).ravel()
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError(
            "spatial blocking requires finite centroid coordinates; the parcel catalogue "
            "appears to be the degraded filename-only variant"
        )
    x_m, y_m = to_epsg3035(lon, lat)
    ix = np.floor(x_m / block_size_m).astype(np.int64)
    iy = np.floor(y_m / block_size_m).astype(np.int64)
    block_id = pd.Series(ix).astype(str) + "_" + pd.Series(iy).astype(str)
    return pd.DataFrame(
        {
            "x_m": x_m,
            "y_m": y_m,
            "block_ix": ix,
            "block_iy": iy,
            "block_id": block_id.to_numpy(),
        }
    )


def block_bounds(block_ix: int, block_iy: int, block_size_m: float) -> tuple[float, float, float, float]:
    """Return ``(x_min, y_min, x_max, y_max)`` of one block, in metres."""
    return (
        block_ix * block_size_m,
        block_iy * block_size_m,
        (block_ix + 1) * block_size_m,
        (block_iy + 1) * block_size_m,
    )


def assign_partitions(
    blocks: pd.DataFrame,
    *,
    test_fraction: float,
    val_fraction: float,
    protocol_seed: int,
    country_code: str,
    block_size_m: float,
) -> pd.Series:
    """Assign whole blocks to the test, validation and pool partitions.

    Blocks are permuted with a generator derived from ``(protocol_seed,
    country_code, block_size_m)``, so the assignment depends on nothing else
    and in particular not on which model, head or label budget will later
    consume the split.

    Args:
        blocks: Frame carrying at least a ``block_id`` column, one row per
            parcel.
        test_fraction: Target share of parcels in the test partition.
        val_fraction: Target share of parcels in the validation partition.
        protocol_seed: Seed identifying the spatial partition.  Held at zero
            for the primary protocol; other values generate the folds of the
            block cross-validation robustness check.
        country_code: ISO-2 country code, mixed into the seed so that the
            partitions of different countries are independent.
        block_size_m: Block edge length, mixed into the seed so that a change
            of block size yields a genuinely different partition.

    Returns:
        A string Series of partition labels aligned to ``blocks``, taking
        values ``test``, ``val`` or ``pool``.

    Raises:
        ValueError: If the fractions are not in ``(0, 1)`` or sum to at least
            one.
    """
    if not (0.0 < test_fraction < 1.0) or not (0.0 <= val_fraction < 1.0):
        raise ValueError("test_fraction must lie in (0, 1) and val_fraction in [0, 1)")
    if test_fraction + val_fraction >= 1.0:
        raise ValueError("test_fraction + val_fraction must leave a non-empty training pool")

    counts = blocks["block_id"].value_counts()
    block_ids = np.array(sorted(counts.index.tolist()), dtype=object)
    sizes = counts.loc[list(block_ids)].to_numpy(dtype=np.int64)

    rng = derived_rng("partition", int(protocol_seed), country_code, f"{block_size_m:.0f}")
    order = rng.permutation(block_ids.size)

    n_total = int(sizes.sum())
    n_test_target = int(round(test_fraction * n_total))
    n_val_target = int(round(val_fraction * n_total))

    label_of_block: dict[object, str] = {}
    n_test = 0
    n_val = 0
    for pos in order:
        bid = block_ids[pos]
        size = int(sizes[pos])
        if n_test < n_test_target:
            label_of_block[bid] = "test"
            n_test += size
        elif n_val < n_val_target:
            label_of_block[bid] = "val"
            n_val += size
        else:
            label_of_block[bid] = "pool"

    return blocks["block_id"].map(label_of_block).astype(str)


def buffer_mask(
    blocks: pd.DataFrame,
    partition: pd.Series,
    *,
    block_size_m: float,
    buffer_m: float,
) -> np.ndarray:
    """Flag pool parcels lying within ``buffer_m`` of a held-out block.

    A parcel is flagged when its projected position lies within ``buffer_m``
    of the rectangle of any block assigned to ``test`` or ``val``.  Since
    ``buffer_m`` is required to be smaller than ``block_size_m``, only the
    eight blocks adjacent to the parcel's own block can be within range, so the
    computation is exact and costs eight vectorised passes.

    Args:
        blocks: Frame with ``x_m``, ``y_m``, ``block_ix``, ``block_iy``.
        partition: Partition labels aligned to ``blocks``.
        block_size_m: Block edge length, in metres.
        buffer_m: Buffer width, in metres.  Zero disables buffering.

    Returns:
        A boolean array, ``True`` where a pool parcel must be discarded.

    Raises:
        ValueError: If ``buffer_m`` is negative or not smaller than the block
            size.
    """
    n = len(blocks)
    if buffer_m == 0:
        return np.zeros(n, dtype=bool)
    if buffer_m < 0 or buffer_m >= block_size_m:
        raise ValueError("buffer_m must satisfy 0 <= buffer_m < block_size_m")

    ix = blocks["block_ix"].to_numpy(dtype=np.int64)
    iy = blocks["block_iy"].to_numpy(dtype=np.int64)
    x = blocks["x_m"].to_numpy(dtype=np.float64)
    y = blocks["y_m"].to_numpy(dtype=np.float64)
    part = partition.to_numpy()

    held_out: set[tuple[int, int]] = set(
        zip(ix[np.isin(part, ("test", "val"))].tolist(), iy[np.isin(part, ("test", "val"))].tolist())
    )
    if not held_out:
        return np.zeros(n, dtype=bool)

    is_pool = part == "pool"
    flagged = np.zeros(n, dtype=bool)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            nb_ix = ix + dx
            nb_iy = iy + dy
            neighbour_held = np.fromiter(
                (
                    (int(a), int(b)) in held_out
                    for a, b in zip(nb_ix.tolist(), nb_iy.tolist())
                ),
                dtype=bool,
                count=n,
            )
            candidate = is_pool & neighbour_held & ~flagged
            if not candidate.any():
                continue
            x_min = nb_ix[candidate] * block_size_m
            y_min = nb_iy[candidate] * block_size_m
            dx_out = np.maximum(np.maximum(x_min - x[candidate], x[candidate] - (x_min + block_size_m)), 0.0)
            dy_out = np.maximum(np.maximum(y_min - y[candidate], y[candidate] - (y_min + block_size_m)), 0.0)
            dist = np.sqrt(dx_out * dx_out + dy_out * dy_out)
            idx = np.flatnonzero(candidate)
            flagged[idx[dist < buffer_m]] = True
    return flagged
