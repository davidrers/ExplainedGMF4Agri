"""Frozen TESSERA v2 on the chip set's composites: input mapping, per-pixel encoding, validated output."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")
rasterio = pytest.importorskip("rasterio")

from gfm4agri.embeddings import tessera_v2 as tv  # noqa: E402
from gfm4agri.embeddings.tessera_v2_model import PixelStudent  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
MINI = REPO / "data" / "eurocrops_chips" / "EE_2021_rev_mini" / "chips"
SPIKE = REPO / "results" / "2026-10-14_tessera_composites" / "spike" / "emb" / "v2_medium"
CHIP = "EE_02252_01776"
SHA = "3823be7db9d9cfc93f3c2a47c7699be82821ab4e1117d4d2befdb746941ee96e"
#: The band names of our chip manifests, in wavelength order.
OURS = ["COASTAL_AEROSOL", "BLUE", "GREEN", "RED", "RED_EDGE_1", "RED_EDGE_2", "RED_EDGE_3",
        "NIR_BROAD", "NIR_NARROW", "WATER_VAPOR", "SWIR_1", "SWIR_2"]


def _write(path: Path, arr: np.ndarray) -> None:
    from affine import Affine

    with rasterio.open(path, "w", driver="GTiff", width=arr.shape[2], height=arr.shape[1],
                       count=arr.shape[0], dtype=arr.dtype, crs="EPSG:3035",
                       transform=Affine(10, 0, 0, 0, -10, arr.shape[1] * 10)) as dst:
        dst.write(arr)


def test_s2_bands_are_taken_by_name_in_tesseras_order(tmp_path):
    T, H = 2, 4
    s2 = np.zeros((T * len(OURS), H, H), np.int16)
    for t in range(T):
        for b in range(len(OURS)):
            s2[t * len(OURS) + b] = 100 * t + b
    _write(tmp_path / "X_merged.tif", s2)
    _write(tmp_path / "X_s1rtc.tif", np.full((T * 2, H, H), -10.0, np.float32))
    x2, x1, hw = tv.chip_inputs(tmp_path, "X", OURS, T)
    assert hw == (H, H) and x2.shape == (H * H, T, 10) and x1.shape == (H * H, T, 2)
    want = [OURS.index(b) for b in tv.TESSERA_S2_BANDS]
    assert want == [3, 1, 2, 7, 8, 4, 5, 6, 10, 11]          # B04 B02 B03 B08 B8A B05 B06 B07 B11 B12
    for t in range(T):
        assert (x2[:, t] == 100 * t + np.array(want)).all()


def test_s1_db_becomes_tesseras_int16_encoding():
    db = np.array([[[-10.0, -20.0], [np.nan, np.nan], [200.0, -30.0]]], np.float32)
    assert tv.s1_to_tessera(db).tolist() == [[[6000.0, 2000.0], [0.0, 0.0], [32767.0, 0.0]]]


def test_the_composites_are_dated_the_15th():
    assert tv.month_doy(2021).tolist() == [15, 46, 74, 105, 135, 166, 196, 227, 258, 288, 319, 349]
    assert tv.month_doy(2024)[2] == 75


def _tiny_student() -> PixelStudent:
    torch.manual_seed(0)
    return PixelStudent(latent_dim=16, num_layers=1, nhead=2, dim_feedforward=32).eval()


def test_each_pixel_is_encoded_on_its_own_and_a_pixel_without_input_is_nan():
    rng = np.random.default_rng(0)
    s2 = rng.integers(1, 3000, (6, 12, 10)).astype(np.float32)
    s1 = rng.normal(-12, 3, (6, 12, 2)).astype(np.float32)
    s2[5], s1[5] = 0, np.nan            # no input at all
    s1[2, 4:] = np.nan                  # four S1 months only: another bucket
    model, doy, cpu = _tiny_student(), tv.month_doy(2021), torch.device("cpu")
    emb = tv.encode_pixels(model, s2, s1, doy, cpu, precision="32-true", batch=4)
    assert emb.shape == (6, tv.N_DIMS) and np.isfinite(emb[:5]).all() and np.isnan(emb[5]).all()
    # A pixel's embedding does not depend on the other pixels of its batch.
    part = tv.encode_pixels(model, s2[[3, 2]], s1[[3, 2]], doy, cpu, precision="32-true", batch=1)
    assert np.allclose(part, emb[[3, 2]], atol=1e-5)


@pytest.mark.skipif(not ((tv.WEIGHTS / "student_medium.pt").exists() and (SPIKE / f"{CHIP}.npy").exists()
                         and (MINI / f"{CHIP}_merged.tif").exists()),
                    reason="v2 weights, pilot chips or the spike's embeddings are absent")
def test_the_core_encoder_reproduces_the_composites_experiment():
    model = tv.load_student("medium", tv.WEIGHTS, SHA).eval()
    s2, s1, _ = tv.chip_inputs(MINI, CHIP, OURS, 12)
    n = 2048
    emb = tv.encode_pixels(model, s2[:n], s1[:n], tv.month_doy(2021), torch.device("cpu"),
                           precision="32-true")
    ref = np.load(SPIKE / f"{CHIP}.npy").astype(np.float32).reshape(tv.N_DIMS, -1).T[:n]
    cos = (emb * ref).sum(1) / np.linalg.norm(emb, axis=1) / np.linalg.norm(ref, axis=1)
    assert np.median(cos) > 0.9999 and cos.min() > 0.999
