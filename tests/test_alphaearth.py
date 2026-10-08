"""AlphaEarth tile reader and the embedding export shared with TESSERA, on synthetic tiles."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from affine import Affine
from pyproj import Transformer

from gfm4agri.embeddings.alphaearth import (
    N_DIMS, AlphaEarthStore, chip_zones, dequantise, select_tiles)
from gfm4agri.embeddings.export import EmbeddingExport, export_chips

UTM = "EPSG:32635"
SOUTH, WEST, N_PX = 6_500_000.0, 500_000.0, 512


def _tile(path: Path, west: float, south: float, n: int = N_PX) -> tuple[float, ...]:
    """A bottom-up tile as published: origin at the south-west corner, positive y resolution.

    Band 0 holds the raw row (south to north) and band 1 the column, both modulo 100, so the
    source pixel behind any chip pixel can be recovered. The south-west 5 x 5 pixels are masked.
    """
    rows, cols = np.mgrid[0:n, 0:n]
    q = np.ones((N_DIMS, n, n), np.int8)
    q[0], q[1] = rows % 100, cols % 100
    q[:, :5, :5] = -128
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", driver="GTiff", width=n, height=n, count=N_DIMS, dtype="int8",
                       crs=UTM, transform=Affine(10, 0, west, 0, 10, south), nodata=-128) as dst:
        dst.write(q)
    return (west, south, west + 10 * n, south + 10 * n)


def _table(tile_dir: Path, tiles: dict[str, tuple[float, ...]]) -> None:
    (tile_dir / "2021").mkdir(parents=True, exist_ok=True)
    (tile_dir / "2021" / "tiles.json").write_text(json.dumps({"zones": {"35N": {
        "crs": UTM, "tiles": [{"key": k, "utm": list(u)} for k, u in tiles.items()]}}}))


def _chip_around(x_utm: float, y_utm: float, px: int = 64) -> tuple[float, ...]:
    """Chip bounds in EPSG:3035 on the 10 m grid, centred near a UTM point."""
    x, y = Transformer.from_crs(UTM, "EPSG:3035", always_xy=True).transform(x_utm, y_utm)
    x0, y0 = np.floor(x / 10) * 10 - px * 5, np.floor(y / 10) * 10 - px * 5
    return (x0, y0, x0 + 10 * px, y0 + 10 * px)


def _expected(bounds, px: int, west: float, south: float) -> tuple[np.ndarray, np.ndarray]:
    """Raw row and column of the tile pixel containing each chip pixel's centre."""
    i, j = np.mgrid[0:px, 0:px]
    x = bounds[0] + 10 * (j + 0.5)
    y = bounds[3] - 10 * (i + 0.5)
    xu, yu = Transformer.from_crs("EPSG:3035", UTM, always_xy=True).transform(x, y)
    return np.floor((np.asarray(yu) - south) / 10), np.floor((np.asarray(xu) - west) / 10)


def test_dequantise_follows_the_readme():
    q = np.array([[[127, -127, 0, 64, -128]]], np.int8).repeat(2, axis=0)
    out = dequantise(q)
    want = np.array([(127 / 127.5) ** 2, -(127 / 127.5) ** 2, 0.0, (64 / 127.5) ** 2])
    assert np.allclose(out[0, 0, :4], want) and np.isnan(out[:, 0, 4]).all()
    # Bit-exact with the README's numpy expression, cast to float32, at every level.
    levels = np.arange(-127, 128, dtype=np.int8)[None, None, :]
    readme = ((levels / 127.5) ** 2 * np.sign(levels)).astype(np.float32)
    assert np.array_equal(dequantise(levels), readme)


def test_chip_zones_add_the_neighbour_only_near_the_edge():
    assert chip_zones(26.6, 58.8) == [["35N"]]
    assert chip_zones([24.2, 23.9, 21.0], [58.0, 58.0, -10.0]) == [
        ["35N", "34N"], ["34N", "35N"], ["34S"]]


def test_bottom_up_tile_lands_north_up_on_the_chip_grid(tmp_path):
    utm = _tile(tmp_path / "2021/35N/a.tiff", WEST, SOUTH)
    _table(tmp_path, {"2021/35N/a.tiff": utm})
    px = 64
    bounds = _chip_around(WEST + 2560, SOUTH + 2560, px)
    emb, info = AlphaEarthStore(tmp_path, 2021).read_chip(bounds, "EPSG:3035", 2021, px, 10.0)
    assert emb.shape == (N_DIMS, px, px) and info["nodata_share"] == 0.0
    rows, cols = _expected(bounds, px, WEST, SOUTH)
    # Row index grows northwards in the file, so on a north-up chip it falls down the rows.
    got_r = np.rint(np.sqrt(emb[0]) * 127.5)
    got_c = np.rint(np.sqrt(emb[1]) * 127.5)
    assert (got_r == rows % 100).mean() > 0.995 and (got_c == cols % 100).mean() > 0.995
    assert got_r[0].mean() > got_r[-1].mean()


def test_masked_pixels_are_nan(tmp_path):
    utm = _tile(tmp_path / "2021/35N/a.tiff", WEST, SOUTH)
    _table(tmp_path, {"2021/35N/a.tiff": utm})
    bounds = _chip_around(WEST + 30, SOUTH + 30, 16)
    emb, info = AlphaEarthStore(tmp_path, 2021).read_chip(bounds, "EPSG:3035", 2021, 16, 10.0)
    rows, cols = _expected(bounds, 16, WEST, SOUTH)
    masked = (rows >= 0) & (rows < 5) & (cols >= 0) & (cols < 5)
    outside = (rows < 0) | (cols < 0)
    assert np.isnan(emb[0][masked | outside]).all() and np.isfinite(emb[0][~masked & ~outside]).all()
    assert 0 < info["nodata_share"] < 1


def test_a_chip_across_two_tiles_is_filled_from_both(tmp_path):
    east_west = WEST + 10 * N_PX
    tiles = {"2021/35N/a.tiff": _tile(tmp_path / "2021/35N/a.tiff", WEST, SOUTH),
             "2021/35N/b.tiff": _tile(tmp_path / "2021/35N/b.tiff", east_west, SOUTH)}
    _table(tmp_path, tiles)
    px = 64
    bounds = _chip_around(east_west, SOUTH + 2560, px)
    emb, info = AlphaEarthStore(tmp_path, 2021).read_chip(bounds, "EPSG:3035", 2021, px, 10.0)
    assert info["nodata_share"] == 0.0 and {t["key"] for t in info["tiles"]} == set(tiles)
    rows, cols = _expected(bounds, px, WEST, SOUTH)
    got_c = np.rint(np.sqrt(emb[1]) * 127.5)
    assert (got_c == cols % N_PX % 100).mean() > 0.995


def test_a_listed_tile_missing_from_disk_raises(tmp_path):
    utm = _tile(tmp_path / "2021/35N/a.tiff", WEST, SOUTH)
    _table(tmp_path, {"2021/35N/a.tiff": utm,
                      "2021/35N/b.tiff": (utm[2], SOUTH, utm[2] + 10 * N_PX, utm[3])})
    bounds = _chip_around(utm[2], SOUTH + 2560, 64)
    with pytest.raises(FileNotFoundError):
        AlphaEarthStore(tmp_path, 2021).read_chip(bounds, "EPSG:3035", 2021, 64, 10.0)


def test_select_tiles_takes_what_the_reader_reads():
    side = 10 * N_PX
    index = pd.DataFrame({
        "year": [2021, 2021, 2021, 2020], "utm_zone": ["35N"] * 4, "crs": [UTM] * 4,
        "utm_west": [WEST, WEST + side, WEST + 5 * side, WEST],
        "utm_south": [SOUTH] * 4, "utm_east": [WEST + side, WEST + 2 * side, WEST + 6 * side, WEST + side],
        "utm_north": [SOUTH + side] * 4})
    chips = np.array([_chip_around(WEST + side, SOUTH + 2560, 64)])
    assert list(select_tiles(index, chips, "EPSG:3035", 2021).index) == [0, 1]


def test_export_writes_rasters_sidecar_and_training_stats(tmp_path):
    chips = [{"chip_id": f"C{i}", "split": s, "bounds_3035": [0, 0, 160, 160]}
             for i, s in enumerate(["train", "train", "val"])]
    (tmp_path / "manifest.json").write_text(json.dumps({
        "year": 2021, "chips": chips,
        "grid": {"crs": "EPSG:3035", "pixel_m": 10.0, "chip_px": 16}}))
    for d in ("training_chips", "validation_chips"):
        (tmp_path / d).mkdir()
    spec = EmbeddingExport(name="fake_v1", suffix="_fake.tif", band_names=["F0", "F1"],
                           source={"store": "test"}, resampling="none", retries=0)
    calls = []

    def make_reader():
        def read(c):
            calls.append(c["chip_id"])
            v = float(c["chip_id"][1:])
            return np.full((2, 16, 16), v, np.float32), {"nodata_share": 0.0}
        return read

    export_chips(spec, tmp_path, make_reader, workers=2)
    side = json.loads((tmp_path / "fake_v1.json").read_text())
    assert (tmp_path / "training_chips/C0_fake.tif").exists()
    assert (tmp_path / "validation_chips/C2_fake.tif").exists()
    assert (tmp_path / "training_chips/C1.fake.json").exists()
    assert side["file_suffix"] == "_fake.tif" and side["band_names"] == ["F0", "F1"]
    assert side["normalisation"]["means"] == [0.5, 0.5]            # chips 0 and 1 only
    assert [c["chip_id"] for c in side["chips"]] == ["C0", "C1", "C2"]

    export_chips(spec, tmp_path, make_reader, workers=2)            # resumes: nothing re-read
    assert sorted(calls) == ["C0", "C1", "C2"]
