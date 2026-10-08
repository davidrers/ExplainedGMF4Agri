"""TerraTorch data plumbing for the segmentation chips: sparse masks, shapes, support draws."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("terratorch")
rasterio = pytest.importorskip("rasterio")

from gfm4agri.benchmark.backbones import BACKBONES, get_backbone  # noqa: E402
from gfm4agri.benchmark.segmentation_data import EuroCropsSegDataModule, draw_support  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
PILOT = REPO / "data" / "eurocrops_chips" / "EE_2021_pilot"
BANDS = ["RED", "NIR_NARROW"]
T, H = 3, 32


def _write(path: Path, arr: np.ndarray) -> None:
    from affine import Affine

    arr = arr if arr.ndim == 3 else arr[None]
    with rasterio.open(path, "w", driver="GTiff", width=H, height=H, count=arr.shape[0],
                       dtype=arr.dtype, crs="EPSG:3035",
                       transform=Affine(10, 0, 0, 0, -10, H * 10)) as dst:
        dst.write(arr)


def _chip(folder: Path, cid: str, seed: int) -> None:
    """Four 16 x 16 quadrant parcels: ids 1..4 with classes 0, 1, 0, -1 (out of scheme)."""
    rng = np.random.default_rng(seed)
    base = 100 * seed
    img = rng.integers(0, 3000, size=(T * len(BANDS), H, H)).astype(np.int16)
    ids = np.zeros((H, H), np.int32)
    ids[:16, :16], ids[:16, 16:], ids[16:, :16], ids[16:, 16:] = base + 1, base + 2, base + 3, base + 4
    mask = np.select([ids % 100 == 1, ids % 100 == 2, ids % 100 == 3], [0, 1, 0], -1).astype(np.int16)
    # The image encodes the parcel id in band 0 of month 0, so alignment can be checked.
    img[0] = ids.astype(np.int16)
    _write(folder / f"{cid}_merged.tif", img)
    _write(folder / f"{cid}.mask.tif", mask)
    _write(folder / f"{cid}.parcels.tif", ids)


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    chips = []
    for split, d, cids in (("train", "training", ["A", "B"]), ("val", "validation", ["C"])):
        (tmp_path / f"{d}_chips").mkdir()
        for i, cid in enumerate(cids):
            _chip(tmp_path / f"{d}_chips", cid, seed=len(chips) + 1)
            chips.append({"chip_id": cid, "split": split})
        (tmp_path / f"{d}_data.txt").write_text("\n".join(cids) + "\n")
    manifest = {
        "dataset": "synthetic", "country": "EE", "year": 2021, "ignore_index": -1,
        "classes": [{"index": 0, "hcat_code": "1", "name": "a"},
                    {"index": 1, "hcat_code": "2", "name": "b"}],
        "imagery": {"band_names": BANDS, "n_months": T},
        "normalisation": {"means": [0.0, 0.0], "stds": [1.0, 1.0]},
        "chips": chips,
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return tmp_path


def _dm(root: Path, **kw) -> EuroCropsSegDataModule:
    return EuroCropsSegDataModule(root, "terramind_v1_small", normalisation="chips",
                                  batch_size=1, num_workers=0, **kw)


def test_dense_sample_shapes_and_labels(root):
    dm = _dm(root, augment=False)
    dm.setup("fit")
    s = dm.train_dataset[0]
    assert tuple(s["image"].shape) == (len(BANDS), T, H, H)
    assert tuple(s["mask"].shape) == (H, H) and str(s["mask"].dtype) == "torch.int64"
    assert sorted(s["mask"].unique().tolist()) == [-1, 0, 1]
    assert dm.test_is_val and len(dm.val_dataset) == 1


def test_sparse_mask_keeps_only_selected_parcels(root):
    dm = _dm(root, augment=True, keep_parcel_ids=np.array([101]))
    dm.setup("fit")
    for _ in range(8):  # D4 draws a random symmetry each time
        s = dm.train_dataset[0]  # chip A, parcel ids 101..104
        ids = s["image"][0, 0].round().long()  # band RED, month 0 carries the parcel id
        labelled = s["mask"] >= 0
        assert labelled.sum() == 256
        assert (ids[labelled] == 101).all() and (s["mask"][labelled] == 0).all()
    # Validation stays dense whatever the budget.
    assert (dm.val_dataset[0]["mask"] >= 0).sum() == 3 * 256


def test_augmentation_draws_all_of_d4_and_keeps_image_and_mask_aligned():
    from gfm4agri.benchmark.segmentation_data import AUGMENT_SYMMETRIES, _numpy_d4

    assert AUGMENT_SYMMETRIES[0] == (0, False) and len(set(AUGMENT_SYMMETRIES)) == 8
    t = _numpy_d4()
    img = np.arange(4 * 4 * 150, dtype=np.float32).reshape(4, 4, 150)
    seen = set()
    for _ in range(200):
        out = t(image=img, mask=img[..., 0].astype(np.int64))
        assert np.array_equal(out["image"][..., 0], out["mask"])
        seen.add(out["mask"].tobytes())
    m = img[..., 0].astype(np.int64)
    group = {np.ascontiguousarray(np.rot90(m, k)[:, ::-1] if f else np.rot90(m, k)).tobytes()
             for k in range(4) for f in (False, True)}
    assert seen == group  # all eight symmetries


def test_draw_support_is_per_class_and_deterministic(root):
    # The budget is a percentage of each class's parcels, ceil(pct / 100 * n), so a class
    # present at all keeps at least one parcel: 10 % of 4 and of 2 are both one parcel.
    keep, report = draw_support(root, pct=10, draw_seed=0)
    again, _ = draw_support(root, pct=10, draw_seed=0)
    assert np.array_equal(keep, again)
    assert report == {"a": {"available": 4, "drawn": 1}, "b": {"available": 2, "drawn": 1}}
    # The proportion is applied per class, not to the pooled parcels.
    _, half = draw_support(root, pct=50, draw_seed=0)
    assert half == {"a": {"available": 4, "drawn": 2}, "b": {"available": 2, "drawn": 1}}
    # 100 % is every parcel, which is the dense mask.
    all_, _ = draw_support(root, pct=100, draw_seed=0)
    assert all_.tolist() == [101, 102, 103, 201, 202, 203]


def test_registry_fragments_are_well_formed():
    for name, spec in BACKBONES.items():
        args = spec.model_args(BANDS, 12)
        if spec.representation != "s2_monthly":  # precomputed embedding: identity backbone
            assert args["backbone"] == "IdentityBackbone" and spec.stats is None
            assert spec.resolution_group == "pixel_raster"
            continue
        # A multimodal entry is its base encoder given a further modality.
        assert args["backbone"] == name or (spec.extra_modalities
                                            and name.startswith(args["backbone"]))
        if spec.extra_modalities:
            assert args["backbone_modalities"] == ["S2L2A", *spec.extra_modalities]
            assert args["backbone_merge_method"] == "mean"
            for m, v in spec.extra_modalities.items():
                assert args["backbone_bands"][m] == list(v["bands"])
        # Month count reaches the model through the temporal wrapper or natively as frames.
        assert 12 in (args.get("backbone_temporal_n_timestamps"), args.get("backbone_num_frames"))
        assert args["necks"][0]["name"] == "SelectIndices"
        means, stds = spec.stats(list(spec.input_bands or ["RED", "BLUE"]))
        assert all(v > 0 for v in stds)
    means, stds = get_backbone("terramind_v1_small").stats(["RED", "BLUE"])
    assert len(means) == len(stds) == 2 and all(s > 0 for s in stds)
    with pytest.raises(KeyError):
        get_backbone("terramind_v1_small").stats(["CIRRUS"])


@pytest.mark.skipif(not (PILOT / "manifest.json").exists(), reason="pilot chips not built")
def test_pilot_chips_load():
    dm = EuroCropsSegDataModule(PILOT, "terramind_v1_small", batch_size=2, num_workers=0)
    dm.setup("fit")
    for s in (dm.val_dataset[0], dm.train_dataset[0]):  # train applies D4 to 144 channels
        assert tuple(s["image"].shape) == (12, 12, 224, 224)
        assert s["mask"].min() >= -1 and s["mask"].max() < 20


def test_band_subset_and_coords_for_prithvi(root):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["year"] = 2021
    manifest["grid"] = {"crs": "EPSG:3035"}
    for c in manifest["chips"]:
        c["bounds_3035"] = [5.2e6, 3.9e6, 5.20224e6, 3.90224e6]
    (root / "manifest.json").write_text(json.dumps(manifest))
    # The synthetic chips carry RED and NIR_NARROW; Prithvi wants six bands, so restrict.
    from dataclasses import replace

    import gfm4agri.benchmark.backbones as bb
    bb.BACKBONES["_test_prithvi"] = replace(bb.BACKBONES["prithvi_eo_v2_300_tl"],
                                           input_bands=("NIR_NARROW",))
    try:
        dm = EuroCropsSegDataModule(root, "_test_prithvi", normalisation="chips", batch_size=2,
                                    num_workers=0, augment=False)
        dm.setup("fit")
        s = dm.train_dataset[0]
        assert tuple(s["image"].shape) == (1, T, H, H)
        assert tuple(s["temporal_coords"].shape) == (T, 2)
        assert s["temporal_coords"][:, 0].eq(2021).all() and s["temporal_coords"][0, 1] == 14
        lat, lon = s["location_coords"].tolist()
        assert 55 < lat < 62 and 20 < lon < 30          # a point in the Baltic
        b = next(iter(dm.train_dataloader()))
        assert tuple(b["location_coords"].shape) == (2, 2)
    finally:
        del bb.BACKBONES["_test_prithvi"]


def test_channel_bottleneck_neck():
    import torch
    from terratorch.registry import TERRATORCH_NECK_REGISTRY

    from gfm4agri.benchmark.necks import ChannelBottleneck

    assert "ChannelBottleneck" in list(TERRATORCH_NECK_REGISTRY)
    neck = ChannelBottleneck([48, 48], out_channels=8)
    out = neck([torch.randn(2, 48, 7, 7), torch.randn(2, 48, 14, 14)])
    assert [tuple(o.shape) for o in out] == [(2, 8, 7, 7), (2, 8, 14, 14)]
    assert neck.process_channel_list([48, 48]) == [8, 8]


@pytest.mark.parametrize("representation, suffix, band", [
    ("tessera_v1", "_tessera.tif", "TESSERA_{:03d}"), ("alphaearth_v1", "_alphaearth.tif", "A{:02d}")])
def test_embedding_raster_representation(root, representation, suffix, band):
    """A precomputed embedding raster shares masks, parcels and splits with the S2 chips."""
    dims = 5
    names = [band.format(i) for i in range(dims)]
    for d in ("training", "validation"):
        for f in (root / f"{d}_chips").glob("*_merged.tif"):
            with rasterio.open(f) as src:
                ids = rasterio.open(str(f).replace("_merged.tif", ".parcels.tif")).read(1)
            emb = np.stack([ids.astype(np.float32) + k for k in range(dims)])  # dim k = id + k
            _write(Path(str(f).replace("_merged.tif", suffix)), emb)
    (root / f"{representation}.json").write_text(json.dumps({
        "file_suffix": suffix, "band_names": names,
        "normalisation": {"means": [0.0] * dims, "stds": [1.0] * dims}}))
    dm = EuroCropsSegDataModule(root, representation, batch_size=1, num_workers=0, augment=True,
                                keep_parcel_ids=np.array([101]))
    dm.setup("fit")
    assert dm.representation == representation and dm.bands == names
    assert dm.img_suffix == suffix
    for _ in range(8):
        s = dm.train_dataset[0]
        assert tuple(s["image"].shape) == (dims, H, H)
        lab = s["mask"] >= 0
        # D4 moves image and mask together; only parcel 101 is labelled.
        assert lab.sum() == 256 and (s["image"][0][lab] == 101).all()
        assert (s["image"][3] - s["image"][0]).eq(3).all()


def test_pixel_mlp_decoder_is_per_pixel():
    import torch

    from gfm4agri.benchmark.decoders import PixelMLPDecoder

    dec = PixelMLPDecoder([16], hidden=[8, 4]).eval()
    x = torch.randn(1, 16, 6, 6)
    y = dec([x])
    assert tuple(y.shape) == (1, 4, 6, 6) and dec.out_channels == 4
    x2 = x.clone(); x2[..., 0, 0] += 10.0          # perturb one pixel
    changed = (dec([x2]) - y).abs().sum(1)[0] > 0
    assert changed[0, 0] and changed.sum() == 1      # only that pixel's output moves


def _code_s2(t: int, c: int) -> int:
    return 1000 * (t + 1) + 10 * (c + 1)


def _code_s1(t: int, c: int) -> float:
    return -(100.0 * (t + 1) + (c + 1))


@pytest.fixture()
def root_s1(root: Path) -> Path:
    """The synthetic chip set with every value encoding its month and band, plus Sentinel-1.

    Month 0, band 0 of both sensors carries the parcel id, positive in Sentinel-2 and negated
    in Sentinel-1, so their spatial alignment can be checked under augmentation. One
    Sentinel-1 pixel is NaN, as a never-observed pixel would be.
    """
    for d in ("training_chips", "validation_chips"):
        for f in (root / d).glob("*_merged.tif"):
            cid = f.name[: -len("_merged.tif")]
            with rasterio.open(root / d / f"{cid}.parcels.tif") as src:
                ids = src.read(1)
            s2 = np.zeros((T * len(BANDS), H, H), np.int16)
            s1 = np.zeros((T * 2, H, H), np.float32)
            for t in range(T):
                for c in range(2):
                    s2[t * len(BANDS) + c] = _code_s2(t, c)
                    s1[t * 2 + c] = _code_s1(t, c)
            s2[0], s1[0] = ids.astype(np.int16), -ids.astype(np.float32)
            s1[2 * 2 + 1, 5, 5] = np.nan
            _write(f, s2)
            _write(root / d / f"{cid}_s1rtc.tif", s1)
    return root


def _dm_s1(root: Path, **kw) -> EuroCropsSegDataModule:
    return EuroCropsSegDataModule(root, "terramind_v1_small_s2s1", normalisation="backbone",
                                  batch_size=1, num_workers=0, **kw)


def test_s1_is_interleaved_month_by_month_after_the_s2_bands(root_s1):
    dm = _dm_s1(root_s1, augment=False)
    dm.setup("fit")
    img = dm.train_dataset[0]["image"]
    assert tuple(img.shape) == (len(BANDS) + 2, T, H, H)
    s1_stats = get_backbone("terramind_v1_small_s2s1").extra_modalities["S1RTC"]["stats"]
    fill_vh = s1_stats(["VH"])[0][0]
    for t in range(T):
        for c in range(2):
            if (t, c) == (0, 0):
                continue
            assert (img[c, t] == _code_s2(t, c)).all(), ("S2", t, c)
            s1 = img[len(BANDS) + c, t]
            if (t, c) == (2, 1):
                assert s1[5, 5] == pytest.approx(fill_vh)  # never observed: the band mean
                s1 = s1.clone()
                s1[5, 5] = _code_s1(t, c)
            assert (s1 == _code_s1(t, c)).all(), ("S1", t, c)
    # The encoder's S2L2A input is built from the Sentinel-2 bands alone.
    assert dm.bands == BANDS
    assert dm.modality_split == [("S2L2A", len(BANDS)), ("S1RTC", 2)]


def test_s1_stays_aligned_with_s2_and_mask_under_d4(root_s1):
    dm = _dm_s1(root_s1, augment=True)
    dm.setup("fit")
    for _ in range(16):
        s = dm.train_dataset[0]
        s2_ids, s1_ids = s["image"][0, 0], s["image"][len(BANDS), 0]
        assert (s1_ids == -s2_ids).all()
        labelled = s["mask"] >= 0
        assert set(s2_ids[labelled].round().long().tolist()) <= {101, 102, 103}


def test_split_modalities_returns_one_tensor_per_modality(root_s1):
    dm = _dm_s1(root_s1, augment=False)
    dm.setup("fit")
    x = dm.train_dataset[0]["image"][None]
    parts = dm.split_modalities(x)
    assert list(parts) == ["S2L2A", "S1RTC"]
    assert tuple(parts["S2L2A"].shape) == (1, len(BANDS), T, H, H)
    assert tuple(parts["S1RTC"].shape) == (1, 2, T, H, H)
    assert (parts["S1RTC"] == x[:, len(BANDS):]).all()


def test_d4_draws_differ_across_dataloader_epochs():
    # DataLoader workers are fresh copies of the main process every epoch. A generator the
    # workers do not reseed replays the same draws each epoch; the D4 draw must not.
    import torch
    from torch.utils.data import DataLoader, Dataset

    from gfm4agri.benchmark.segmentation_data import _numpy_d4

    class Draws(Dataset):
        def __init__(self):
            self.t = _numpy_d4()
            self.img = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)

        def __len__(self):
            return 16

        def __getitem__(self, i):
            return torch.from_numpy(self.t(image=self.img, mask=self.img[..., 0])["image"].copy())

    loader = DataLoader(Draws(), batch_size=4, num_workers=2)
    epochs = [torch.cat(list(loader)).numpy().tobytes() for _ in range(3)]
    assert len(set(epochs)) == 3


def test_support_draw_ignores_int_or_float_budget(tmp_path):
    """A budget of 5 and of 5.0 must draw the same parcels, whichever script passes it."""
    from gfm4agri.data.seeding import derived_rng

    assert f"{5:g}" == f"{5.0:g}" == "5" and f"{100.0:g}" == "100"
    a = derived_rng("seg_support", "EE", "3302000000", f"{5:g}", 0).permutation(50)
    b = derived_rng("seg_support", "EE", "3302000000", f"{5.0:g}", 0).permutation(50)
    assert (a == b).all()
