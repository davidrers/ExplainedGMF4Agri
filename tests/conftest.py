"""Shared fixtures for the gfm4agri test suite.

The project is not installed as a distribution, so ``src`` is placed on the
import path here rather than through packaging metadata.
"""

from __future__ import annotations

import json
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


SYN_H = 32
SYN_SPLIT = "blocks1_buf0_seed0__0000abcd"
SYN_BLOCKS = {"0_0": ("pool", ["EE_00000_00000", "EE_00001_00000"]),
              "1_0": ("val", ["EE_00004_00000", "EE_00005_00000"]),
              "2_0": ("test", ["EE_00008_00000", "EE_00009_00000"])}
SYN_LISTS = {"pool": "training", "val": "validation", "test": "test"}


def _write_raster(path: Path, arr: np.ndarray) -> None:
    import rasterio
    from affine import Affine

    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=SYN_H, height=SYN_H, count=arr.shape[0],
                       dtype=arr.dtype, crs="EPSG:3035",
                       transform=Affine(10, 0, 0, 0, -10, SYN_H * 10)) as dst:
        dst.write(arr)


def make_split_chipset(root: Path) -> Path:
    """A chip set in the full-country layout, small enough for CPU fits.

    Six chips in three blocks of two: block ``0_0`` is the pool (training), ``1_0`` validation and
    ``2_0`` test. Every chip has four quadrant parcels of classes 0, 1, 0 and out of scheme, a
    two-band three-month Sentinel-2 stack, a four-dimension TESSERA raster, a mask and a parcel
    raster; ``chip_parcels.parquet`` and one spatial split directory describe them.
    """
    chips_dir = root / "chips"
    chips_dir.mkdir(parents=True)
    q = SYN_H // 2
    rows, table, manifest_chips, lists = [], [], [], {v: [] for v in SYN_LISTS.values()}
    for n, (block, (part, cids)) in enumerate(SYN_BLOCKS.items()):
        for i, cid in enumerate(cids):
            base = 100 * (2 * n + i + 1)
            ids = np.zeros((SYN_H, SYN_H), np.int32)
            ids[:q, :q], ids[:q, q:], ids[q:, :q], ids[q:, q:] = base + 1, base + 2, base + 3, base + 4
            mask = np.select([ids % 100 == 1, ids % 100 == 2, ids % 100 == 3], [0, 1, 0], -1)
            rng = np.random.default_rng(base)
            _write_raster(chips_dir / f"{cid}_merged.tif",
                          rng.integers(0, 3000, (6, SYN_H, SYN_H)).astype(np.int16))
            _write_raster(chips_dir / f"{cid}_tessera.tif",
                          np.stack([ids.astype(np.float32) / 1000 + k for k in range(4)]))
            _write_raster(chips_dir / f"{cid}.mask.tif", mask.astype(np.int16))
            _write_raster(chips_dir / f"{cid}.parcels.tif", ids)
            col = int(cid.split("_")[1])
            rows.append({"chip_id": cid, "col": col, "row": 0, "partition": part, "block_id": block,
                         "labelled_parcels": 3, "trainable_parcels": 3 if part == "pool" else 0})
            table += [{"chip_id": cid, "parcel_id": base + k, "class_index": c, "pixels": q * q}
                      for k, c in ((1, 0), (2, 1), (3, 0))]
            manifest_chips.append({"chip_id": cid, "col": col, "row": 0,
                                   "bounds_3035": [col * 2240.0, 0.0, (col + 1) * 2240.0, 2240.0]})
            lists[SYN_LISTS[part]].append(cid)
    manifest = {
        "dataset": "synthetic", "country": "EE", "year": 2021, "ignore_index": -1,
        "classes": [{"index": 0, "hcat_code": "1", "name": "a"},
                    {"index": 1, "hcat_code": "2", "name": "b"}],
        "imagery": {"band_names": ["RED", "NIR_NARROW"], "n_months": 3},
        "normalisation": {"means": [0.0, 0.0], "stds": [1.0, 1.0]},
        "chips": manifest_chips,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "tessera_v1.json").write_text(json.dumps({
        "representation": "tessera_v1", "file_suffix": "_tessera.tif",
        "band_names": [f"TESSERA_{k:03d}" for k in range(4)],
        "normalisation": {"means": [0.0] * 4, "stds": [1.0] * 4}}))
    pd.DataFrame(table).astype({"class_index": np.int16}).to_parquet(root / "chip_parcels.parquet",
                                                                      index=False)
    split = root / "splits" / SYN_SPLIT
    split.mkdir(parents=True)
    for name, cids in lists.items():
        (split / f"{name}_data.txt").write_text("\n".join(cids) + "\n")
    pd.DataFrame(rows).to_csv(split / "chips.csv", index=False)
    np.save(split / "buffer_parcels.npy", np.array([], dtype=np.int64))
    (split / "split.json").write_text(json.dumps({
        "split": SYN_SPLIT, "config_hash": "0000abcd00000000", "protocol": "synthetic blocks",
        "lists": {k: len(v) for k, v in lists.items()}}))
    return root


@pytest.fixture()
def split_chipset(tmp_path: Path) -> Path:
    """A fresh synthetic full-country chip set under ``tmp_path / "SYN_2021"``."""
    return make_split_chipset(tmp_path / "SYN_2021")
