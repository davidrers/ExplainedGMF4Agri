"""A computed raster is encoded once per chip set, reused only when it matches, never overwritten."""

from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("terratorch")
rasterio = pytest.importorskip("rasterio")

from conftest import SYN_SPLIT  # noqa: E402
from gfm4agri.pipeline.raster_cache import build_raster_cache, resolve_raster_cache  # noqa: E402

B = "stub_v2"


def _split(chips):
    return chips / "splits" / SYN_SPLIT


def test_every_chip_is_encoded_with_a_sidecar_of_training_statistics(split_chipset, tmp_path, stub_computed):
    cache = tmp_path / "emb" / B / split_chipset.name
    record = build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
    ids = sorted(p.name[: -len(".parcels.tif")] for p in (split_chipset / "chips").glob("*.parcels.tif"))
    assert sorted(stub_computed.encoded) == ids and record["encoded_now"] == 6
    assert sorted(p.name for p in (cache / "chips").iterdir()) == [f"{c}_{B}.tif" for c in ids]
    side = json.loads((cache / f"{B}.json").read_text())
    assert side["file_suffix"] == f"_{B}.tif" and len(side["band_names"]) == 4
    assert all(c["nodata_share"] == pytest.approx(1 / 32 ** 2, abs=1e-6) for c in side["chips"])
    # Statistics over the two training chips, the NaN pixel left out.
    vals = []
    for c in ("EE_00000_00000", "EE_00001_00000"):
        with rasterio.open(split_chipset / "chips" / f"{c}.parcels.tif") as src:
            vals.append(src.read(1).astype(np.float64).ravel()[1:])
    vals = np.concatenate(vals)
    assert side["normalisation"]["means"] == pytest.approx([vals.mean() + k for k in range(4)], abs=1e-4)
    assert json.loads((cache / "cache.json").read_text())["backbone"] == B
    assert not (cache / "building.json").exists()
    # On the grid of the chip's own mask.
    with rasterio.open(cache / "chips" / f"EE_00000_00000_{B}.tif") as e, \
            rasterio.open(split_chipset / "chips" / "EE_00000_00000.mask.tif") as m:
        assert e.transform == m.transform and e.crs == m.crs


def test_no_cache_is_computed_in_scratch_and_a_complete_one_is_reused(split_chipset, tmp_path, stub_computed):
    root, scratch = tmp_path / "emb", tmp_path / "scratch"
    d = resolve_raster_cache(root, scratch, B, split_chipset, _split(split_chipset))
    assert d.action == "compute" and d.cache_dir == scratch / B / split_chipset.name
    build_raster_cache(B, split_chipset, _split(split_chipset), root / B / split_chipset.name, batch=1)
    d = resolve_raster_cache(root, scratch, B, split_chipset, _split(split_chipset))
    assert d.action == "reuse" and d.cache_dir == root / B / split_chipset.name


def test_a_missing_chip_is_encoded_again_and_nothing_else(split_chipset, tmp_path, stub_computed):
    root = tmp_path / "emb"
    cache = root / B / split_chipset.name
    build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
    (cache / "chips" / f"EE_00008_00000_{B}.tif").unlink()
    d = resolve_raster_cache(root, root, B, split_chipset, _split(split_chipset))
    assert d.action == "compute" and d.cache_dir == cache
    stub_computed.encoded.clear()
    build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
    assert stub_computed.encoded == ["EE_00008_00000"]


def test_a_cache_made_under_another_record_is_never_written_into(split_chipset, tmp_path, stub_computed):
    root, scratch = tmp_path / "emb", tmp_path / "scratch"
    cache = root / B / split_chipset.name
    build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
    rec = json.loads((cache / "cache.json").read_text())
    (cache / "cache.json").write_text(json.dumps(rec | {"checkpoint_sha256": "cd" * 32}))
    assert resolve_raster_cache(root, root, B, split_chipset, _split(split_chipset)).action == "stop"
    d = resolve_raster_cache(root, scratch, B, split_chipset, _split(split_chipset))
    assert d.action == "compute" and d.cache_dir == scratch / B / split_chipset.name
    with pytest.raises(SystemExit, match="another record"):
        build_raster_cache(B, split_chipset, _split(split_chipset), cache, batch=1)
