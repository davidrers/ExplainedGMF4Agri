"""The two-stage workflow must train the same model on the same inputs as the end-to-end one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("terratorch")
rasterio = pytest.importorskip("rasterio")

from gfm4agri.benchmark.backbones import BACKBONES  # noqa: E402
from gfm4agri.benchmark.cached import (  # noqa: E402
    EMB_SUFFIX,
    IDENTITY,
    VARIANTS,
    CachedFeatureDataset,
    apply_d4,
    split_at_cache_point,
    variant_name,
)
from gfm4agri.benchmark.segmentation_data import _numpy_d4  # noqa: E402

H = 16


def test_the_eight_variants_are_the_random_d4_of_the_end_to_end_fit():
    # Each cached variant must be one of the inputs the end-to-end augmentation can draw,
    # produced by the same operations in the same order.
    d4 = _numpy_d4()
    img = np.arange(H * H * 3, dtype=np.float32).reshape(H, H, 3)
    ours = {apply_d4(img, k, f).tobytes() for k, f in VARIANTS}
    theirs = {d4._apply(img, k, f).tobytes() for k in range(4) for f in (False, True)}
    assert len(VARIANTS) == 8 and ours == theirs
    assert np.array_equal(apply_d4(img, *IDENTITY), img)


def test_training_variants_are_exactly_what_the_end_to_end_augmentation_draws():
    # The cached fit trains on TRAIN_VARIANTS; the end-to-end fit draws from the same set.
    from gfm4agri.benchmark.cached import TRAIN_VARIANTS

    d4 = _numpy_d4()
    img = np.arange(H * H, dtype=np.float32).reshape(H, H)
    drawn = {d4(image=img[..., None], mask=img)["mask"].tobytes() for _ in range(200)}
    assert drawn == {apply_d4(img, k, f).tobytes() for k, f in TRAIN_VARIANTS}
    assert TRAIN_VARIANTS[0] == IDENTITY and len(TRAIN_VARIANTS) == 8


@pytest.mark.parametrize("name", [n for n, s in BACKBONES.items()
                                  if s.representation == "s2_monthly"])
def test_cache_point_is_before_the_first_trainable_neck(name):
    if name.startswith("thor"):
        pytest.importorskip("thor_terratorch_ext")
    frozen, tail = split_at_cache_point(BACKBONES[name].model_args(["RED", "BLUE"], 12))
    # A ViT handing over tokens is cached after they are reshaped to maps; THOR's wrapper
    # hands over maps already.
    assert [n["name"] for n in frozen] in (["SelectIndices", "ReshapeTokensToImage"],
                                           ["SelectIndices"])
    assert tail[0]["name"] == "ChannelBottleneck"


def _write(path: Path, arr: np.ndarray) -> None:
    from affine import Affine

    path.parent.mkdir(parents=True, exist_ok=True)
    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype=arr.dtype, crs="EPSG:3035",
                       transform=Affine(10, 0, 0, 0, -10, arr.shape[1] * 10)) as dst:
        dst.write(arr)


@pytest.fixture()
def cache(tmp_path: Path) -> tuple[Path, Path, np.ndarray]:
    """A chip whose labels are four parcels, and a cache holding, for every variant, the
    parcel ids under that variant as its feature map, in two layers."""
    ids = np.zeros((H, H), np.int32)
    ids[:8, :8], ids[:8, 8:], ids[8:, :8], ids[8:, 8:] = 1, 2, 3, 4
    mask = np.select([ids == 1, ids == 2, ids == 3], [0, 1, 0], -1).astype(np.int16)
    labels = tmp_path / "chips"
    _write(labels / "A.mask.tif", mask)
    _write(labels / "A.parcels.tif", ids)
    for k, f in VARIANTS:
        feat = apply_d4(ids.astype(np.float32), k, f)
        for layer in range(2):
            out = tmp_path / "cache" / variant_name(k, f) / f"layer_{layer:02d}"
            out.mkdir(parents=True, exist_ok=True)
            np.save(out / f"A{EMB_SUFFIX}", np.stack([feat, feat + 100 * layer]))
    return tmp_path / "cache", labels, ids


def test_mask_follows_the_drawn_variant_of_the_features(cache):
    cache_dir, labels, _ = cache
    ds = CachedFeatureDataset(cache_dir, labels, ["A"], n_layers=2, variants=VARIANTS)
    seen = set()
    for _ in range(64):
        s = ds[0]
        assert tuple(s["image"].shape) == (4, H, H)  # two layers of two channels
        feat_ids = s["image"][0].round().long()
        labelled = s["mask"] >= 0
        # parcels 1 and 3 are class 0, parcel 2 is class 1, parcel 4 is out of scheme
        assert (s["mask"][feat_ids == 2] == 1).all() and (s["mask"][feat_ids == 4] == -1).all()
        assert set(feat_ids[labelled].tolist()) <= {1, 2, 3}
        seen.add(s["image"][0].numpy().tobytes())
    assert len(seen) == 8  # every variant is drawn


def test_label_budget_keeps_only_drawn_parcels(cache):
    cache_dir, labels, _ = cache
    ds = CachedFeatureDataset(cache_dir, labels, ["A"], n_layers=2, variants=[IDENTITY],
                              keep_parcel_ids=np.array([2]))
    s = ds[0]
    assert set(s["image"][0][s["mask"] >= 0].round().long().tolist()) == {2}


def test_variant_draws_differ_across_dataloader_epochs(cache):
    import torch
    from torch.utils.data import DataLoader

    cache_dir, labels, _ = cache
    ds = CachedFeatureDataset(cache_dir, labels, ["A"] * 8, n_layers=2, variants=VARIANTS)
    loader = DataLoader(ds, batch_size=4, num_workers=2)
    epochs = [torch.cat([b["image"] for b in loader]).numpy().tobytes() for _ in range(3)]
    assert len(set(epochs)) == 3


def test_training_can_be_restricted_to_some_cached_variants(cache, tmp_path):
    import json

    from gfm4agri.benchmark.cached import CachedFeatureDataModule

    cache_dir, _, _ = cache
    every = [variant_name(k, f) for k, f in VARIANTS]
    (cache_dir / "cache.json").write_text(json.dumps({"channel_list": [2, 2],
                                                      "train_variants": every}))
    root = tmp_path / "root"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({"classes": [{"name": "a"}],
                                                    "ignore_index": -1}))
    assert CachedFeatureDataModule(root, cache_dir).train_variants == every
    assert CachedFeatureDataModule(root, cache_dir,
                                   train_variants=["k0", "k1"]).train_variants == ["k0", "k1"]
    assert CachedFeatureDataModule(root, cache_dir, train_variants=["k0"]).train_variants == ["k0"]
    with pytest.raises(ValueError, match="k5"):
        CachedFeatureDataModule(root, cache_dir, train_variants=["k0", "k5"])
