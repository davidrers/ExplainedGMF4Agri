"""Chip grid geometry and label rasterisation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gfm4agri.data.chip_grid import (
    CHIP_M, CHIP_PX, IGNORE_INDEX, Chip, apply_vector_aliases, chip_of, rasterise_labels,
    select_chips, vector_layer_aliases)

gpd = pytest.importorskip("geopandas")
from shapely.geometry import box  # noqa: E402


def test_chip_bounds_and_transform_agree():
    chip = Chip("EE", 2300, 1600)
    xmin, ymin, xmax, ymax = chip.bounds
    assert xmax - xmin == ymax - ymin == CHIP_M == 2240.0
    assert chip.transform @ (0, 0) == (xmin, ymax)
    assert chip.transform @ (CHIP_PX, CHIP_PX) == (xmax, ymin)
    col, row = chip_of(np.array([xmin + 1.0]), np.array([ymin + 1.0]))
    assert (col[0], row[0]) == (2300, 1600)


def test_rasterise_dense_mask_and_parcel_ids():
    chip = Chip("EE", 2300, 1600)
    xmin, ymin, _, ymax = chip.bounds
    parcels = gpd.GeoDataFrame({
        "pid": [11, 22, 33],
        "hcat": ["A", "B", "OUT"],
        "geometry": [box(xmin, ymax - 100, xmin + 100, ymax),          # 10 x 10 px, top left
                     box(xmin + 500, ymin, xmin + 700, ymin + 200),    # 20 x 20 px, bottom
                     box(xmin + 1000, ymin + 1000, xmin + 1100, ymin + 1100)],
    }, crs="EPSG:3035")
    mask, ids = rasterise_labels(chip, parcels, "hcat", "pid", ["A", "B"])
    assert mask.dtype == np.int16 and ids.dtype == np.int32
    assert (mask == 0).sum() == 100 and (mask == 1).sum() == 400
    assert mask[0, 0] == 0 and mask[-1, 55] == 1
    # An out-of-scheme parcel is ignored in the mask but kept in the id raster.
    assert (ids == 33).sum() == 100 and (mask[ids == 33] == IGNORE_INDEX).all()
    assert (mask == IGNORE_INDEX).sum() == CHIP_PX**2 - 500


def test_select_chips_respects_separation():
    cells = pd.DataFrame({"col": [0, 1, 10, 20], "row": [0, 0, 0, 0],
                          "label_share": [0.9, 0.9, 0.9, 0.9], "n_classes": [9, 8, 7, 6],
                          "grass_share": [0.1] * 4})
    chips = select_chips(cells, "EE", 3, min_label_share=0.5, min_sep_m=5 * CHIP_M, seed=0)
    cols = sorted(c.col for c in chips)
    assert cols == [0, 10, 20] or cols == [1, 10, 20]
    with pytest.raises(ValueError):
        select_chips(cells, "EE", 4, min_label_share=0.5, min_sep_m=5 * CHIP_M, seed=0)


def test_vector_aliases_label_a_renamed_code_as_its_scheme_class():
    scheme = {"vector_layer_aliases": {"3301060402": ["3301060403"]}}
    lookup = vector_layer_aliases(scheme)
    assert lookup == {"3301060403": "3301060402"}
    codes = pd.Series(["3301060403", "3301060401", "3302000000"])
    assert apply_vector_aliases(codes, lookup).tolist() == ["3301060402", "3301060401", "3302000000"]
    assert apply_vector_aliases(codes, {}).equals(codes)

    chip = Chip("EE", 2300, 1600)
    xmin, ymin, _, ymax = chip.bounds
    parcels = gpd.GeoDataFrame(
        {"pid": [1], "EC_hcat_c": apply_vector_aliases(pd.Series(["3301060403"]), lookup)},
        geometry=[box(xmin, ymax - 500, xmin + 500, ymax)], crs="EPSG:3035")
    mask, _ = rasterise_labels(chip, parcels, "EC_hcat_c", "pid", ["3301060401", "3301060402"])
    assert (mask == 1).sum() == 50 * 50 and set(np.unique(mask)) == {IGNORE_INDEX, 1}
