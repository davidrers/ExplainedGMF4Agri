"""The spatial block split of a full-country chip set.

The unit of partition is the **chip**. Chips are cells of the fixed 2,240 m grid of
:mod:`gfm4agri.data.chip_grid`, and a block is a square of ``block_chips x block_chips`` chips,
so a block is a cell of the EPSG:3035 tessellation of :mod:`gfm4agri.data.blocks` with edge
``block_chips * 2,240 m`` and every chip nests in exactly one block. Whole blocks are assigned
to the ``test``, ``val`` and ``pool`` partitions with :func:`gfm4agri.data.blocks.assign_partitions`,
so the assignment is a function of ``(protocol_seed, country, block edge)`` alone and the target
shares are shares of parcels, as for the parcel-level protocol.

**The buffer is applied to parcels, not to chips.** A pool parcel is withheld from training when
any of its labelled pixels lies within ``buffer_m`` of the rectangle of a test or validation block,
or when the same parcel identifier also occurs in a held-out chip. The pool chip itself stays in
the training set, with the withheld parcels carrying ``ignore_index``, so the buffer costs only the
labels near a held-out block and not the whole chip. The distance is measured from the pixel
centres of the rasterised labels, so the rule is exactly the one the training masks see.

A chip is the whole input of the segmenter, and chips do not overlap, so a test chip's input
never contains a pixel of a training chip. The buffer closes the remaining channels: a field cut
by a block edge, which would otherwise be labelled on both sides, and the short-range spatial
autocorrelation of crop labels across the edge.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from gfm4agri.data.blocks import assign_partitions
from gfm4agri.data.chip_grid import CHIP_M, CHIP_PX, PIXEL_M

__all__ = [
    "ChipSplitConfig",
    "assign_chip_partitions",
    "block_of_chip",
    "buffered_parcels",
    "chip_parcel_table",
]


@dataclass(frozen=True)
class ChipSplitConfig:
    """Everything that determines a chip split; its hash names the split directory."""

    name: str
    country: str
    year: int
    #: Block edge in chips; the block edge in metres is ``block_chips * 2,240``.
    block_chips: int = 4
    test_fraction: float = 0.2
    val_fraction: float = 0.1
    #: Protocol section 3.5: half the diagonal of the 224 x 224 px, 10 m patch, rounded up.
    buffer_m: float = 1600.0
    protocol_seed: int = 0
    #: SHA-256 of the chip manifest, so a re-exported chip set yields a new split.
    manifest_sha256: str = ""

    @property
    def block_size_m(self) -> float:
        return self.block_chips * CHIP_M

    def config_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()

    @property
    def dir_name(self) -> str:
        return f"{self.name}__{self.config_hash()[:8]}"


def block_of_chip(col: np.ndarray, row: np.ndarray, block_chips: int) -> tuple[np.ndarray, np.ndarray]:
    """Block column and row of chip columns and rows; floor division, so negative cells work."""
    return (np.floor_divide(np.asarray(col), block_chips),
            np.floor_divide(np.asarray(row), block_chips))


def chip_parcel_table(chip_id: str, mask: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    """One row per ``(chip, parcel, class)``: the labelled pixels of that parcel in the chip."""
    sel = (mask >= 0) & (ids > 0)
    if not sel.any():
        return pd.DataFrame({"chip_id": pd.Series(dtype=str), "parcel_id": pd.Series(dtype=np.int64),
                             "class_index": pd.Series(dtype=np.int16),
                             "pixels": pd.Series(dtype=np.int64)})
    pairs, counts = np.unique(np.stack([ids[sel].astype(np.int64), mask[sel].astype(np.int64)]),
                              axis=1, return_counts=True)
    return pd.DataFrame({"chip_id": chip_id, "parcel_id": pairs[0],
                         "class_index": pairs[1].astype(np.int16), "pixels": counts})


def assign_chip_partitions(chips: pd.DataFrame, parcels: pd.DataFrame,
                           config: ChipSplitConfig) -> pd.Series:
    """``test``, ``val``, ``pool`` or ``empty`` for every chip, by whole blocks.

    ``chips`` carries ``chip_id``, ``col`` and ``row``; ``parcels`` is the table of
    :func:`chip_parcel_table` over all chips. A block's weight is the number of distinct
    labelled parcels in its chips. A chip in a block holding no labelled parcel at all is
    ``empty`` and belongs to no partition.
    """
    bx, by = block_of_chip(chips["col"].to_numpy(), chips["row"].to_numpy(), config.block_chips)
    block_id = pd.Series([f"{a}_{b}" for a, b in zip(bx, by)], index=chips.index)
    chip_block = dict(zip(chips["chip_id"], block_id))
    per_parcel = parcels[["chip_id", "parcel_id"]].drop_duplicates()
    per_parcel = per_parcel.assign(block_id=per_parcel["chip_id"].map(chip_block))
    # A parcel is counted once per block it reaches, so the weight is a parcel count rather
    # than a count of parcel fragments cut by chip edges.
    rows = per_parcel[["block_id", "parcel_id"]].drop_duplicates()[["block_id"]].reset_index(drop=True)
    labels = assign_partitions(rows, test_fraction=config.test_fraction,
                               val_fraction=config.val_fraction,
                               protocol_seed=config.protocol_seed,
                               country_code=config.country, block_size_m=config.block_size_m)
    of_block = dict(zip(rows["block_id"], labels))
    return block_id.map(of_block).fillna("empty").astype(str)


def _pixel_centres(col: int, row: int) -> tuple[np.ndarray, np.ndarray]:
    """``(CHIP_PX, CHIP_PX)`` EPSG:3035 x and y of the pixel centres of a chip."""
    x0, y1 = col * CHIP_M, (row + 1) * CHIP_M
    offs = (np.arange(CHIP_PX) + 0.5) * PIXEL_M
    return np.meshgrid(x0 + offs, y1 - offs)


def buffered_parcels(chips: pd.DataFrame, partition: pd.Series, parcels: pd.DataFrame,
                     arrays, config: ChipSplitConfig) -> tuple[set[int], dict]:
    """Pool parcels withheld from training, and a report of why.

    ``parcels`` is the table of :func:`chip_parcel_table` over all chips. ``arrays`` maps a
    chip id to its ``(mask, ids)`` rasters, or is a callable returning them, and is consulted
    only for pool chips within ``buffer_m`` of a held-out block. Because the buffer is shorter
    than a block edge, only the eight blocks around a chip's own block can be in range.
    """
    read = arrays if callable(arrays) else arrays.__getitem__
    if not 0 <= config.buffer_m < config.block_size_m:
        raise ValueError("buffer_m must satisfy 0 <= buffer_m < block edge")
    L = config.block_size_m
    bx, by = block_of_chip(chips["col"].to_numpy(), chips["row"].to_numpy(), config.block_chips)
    part = partition.to_numpy()
    held = {(int(a), int(b)) for a, b, p in zip(bx, by, part) if p in ("test", "val")}

    near: set[int] = set()
    n_chips = 0
    if config.buffer_m > 0:
        for cid, col, row, a, b, p in zip(chips["chip_id"], chips["col"], chips["row"], bx, by, part):
            if p != "pool":
                continue
            rects = [((a + dx) * L, (b + dy) * L, (a + dx + 1) * L, (b + dy + 1) * L)
                     for dx in (-1, 0, 1) for dy in (-1, 0, 1)
                     if (dx or dy) and (int(a + dx), int(b + dy)) in held]
            x0, y0 = col * CHIP_M, row * CHIP_M
            rects = [r for r in rects
                     if np.hypot(max(r[0] - (x0 + CHIP_M), x0 - r[2], 0.0),
                                 max(r[1] - (y0 + CHIP_M), y0 - r[3], 0.0)) < config.buffer_m]
            if not rects:
                continue
            mask, ids = read(cid)
            x, y = _pixel_centres(int(col), int(row))
            close = np.zeros(x.shape, dtype=bool)
            for xmin, ymin, xmax, ymax in rects:
                dx = np.maximum(np.maximum(xmin - x, x - xmax), 0.0)
                dy = np.maximum(np.maximum(ymin - y, y - ymax), 0.0)
                close |= np.hypot(dx, dy) < config.buffer_m
            hit = close & (mask >= 0) & (ids > 0)
            if hit.any():
                n_chips += 1
                near.update(int(v) for v in np.unique(ids[hit]))

    # A multi-part parcel can reach a held-out chip without touching the buffer zone.
    of_chip = dict(zip(chips["chip_id"], part))
    where = parcels["chip_id"].map(of_chip)
    in_held = set(parcels.loc[where.isin(["test", "val"]), "parcel_id"].astype(np.int64).tolist())
    in_pool = set(parcels.loc[where == "pool", "parcel_id"].astype(np.int64).tolist())
    shared = (in_pool & in_held) - near
    report = {"buffer_m": config.buffer_m, "pool_chips_touched": n_chips,
              "parcels_within_buffer": len(near & in_pool),
              "parcels_shared_with_held_out_beyond_buffer": len(shared)}
    return (near & in_pool) | shared, report
