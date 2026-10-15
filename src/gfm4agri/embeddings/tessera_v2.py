"""Frozen TESSERA v2 on the chip set's monthly composites: weights, input mapping, per-pixel encoding.

TESSERA encodes each pixel's own Sentinel-2 and Sentinel-1 series, with no spatial context. Here it reads the twelve
monthly composites every other arm reads, as validated in experiments/2026-10-14_tessera_composites (frozen, it
separates crops as well as from its native per-acquisition series) and experiments/2026-10-15_tessera_decoder
(chip rasters equal to the spike's, cosine 0.9999):

* **S2**: TESSERA's ten bands in its order, B04 B02 B03 B08 B8A B05 B06 B07 B11 B12, taken from ``_merged.tif`` by
  band name; DN = reflectance x 10000, TESSERA's convention; every month valid, the composites being gap filled.
* **S1**: our 10 log10 of the linear-power median, in dB, becomes TESSERA's int16 encoding
  (20 log10(power) + 50) x 200 = (2 dB + 50) x 200, clipped to [0, 32767], with 0 where a month was never observed,
  TESSERA's missing value. Our composites merge the two orbit directions, so the frames go in as its ascending stream.
* **Dates**: the 15th of each month, as the raw day of year.

The inputs are standardised with v2's statistics and bucketed and padded as v2's ``encode_pixels`` does
(:mod:`gfm4agri.embeddings.tessera_v2_model`). A pixel with no valid S2 and no valid S1 month comes out NaN; the
student itself would return a vector for it.
"""

from __future__ import annotations

import datetime as dt
import hashlib
from pathlib import Path

import numpy as np

__all__ = ["N_DIMS", "TESSERA_S2_BANDS", "WEIGHTS", "ChipEncoder", "chip_inputs", "encode_pixels",
           "load_student", "month_doy", "s1_to_tessera"]

REPO = Path(__file__).resolve().parents[3]
#: TESSERA's ten S2 bands, B04 B02 B03 B08 B8A B05 B06 B07 B11 B12, by our manifests' band names.
TESSERA_S2_BANDS = ("RED", "BLUE", "GREEN", "NIR_BROAD", "NIR_NARROW", "RED_EDGE_1", "RED_EDGE_2",
                    "RED_EDGE_3", "SWIR_1", "SWIR_2")
#: Hugging Face repository letter of each v2 student, ``geotessera/TESSERA-V-2.0-2B-<letter>``.
STUDENTS = {"nano": "N", "small": "S", "medium": "M", "large": "L"}
WEIGHTS = REPO / "data" / "weights" / "tessera" / "v2"
N_DIMS = 128


def month_doy(year: int, n_months: int = 12) -> np.ndarray:
    """Day of year of the 15th of each month, the raw integers TESSERA takes as dates."""
    return np.array([dt.date(year, m, 15).timetuple().tm_yday for m in range(1, n_months + 1)],
                    dtype=np.float32)


def s1_to_tessera(s1_db) -> np.ndarray:
    """Our S1 in dB, NaN where never observed, in TESSERA's raw units, 0 where never observed."""
    s1_db = np.asarray(s1_db, dtype=np.float32)
    ok = np.isfinite(s1_db).all(-1, keepdims=True)
    return np.where(ok, np.clip((2.0 * s1_db + 50.0) * 200.0, 0, 32767), 0.0).astype(np.float32)


def chip_inputs(chip_dir, chip_id: str, band_names: list[str],
                n_months: int) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    """One chip's pixels as TESSERA reads them: S2 ``(P, T, 10)`` DN, S1 ``(P, T, 2)`` dB, and ``(H, W)``."""
    import rasterio

    idx = [list(band_names).index(b) for b in TESSERA_S2_BANDS]
    with rasterio.open(Path(chip_dir) / f"{chip_id}_merged.tif") as src:
        hw = (src.height, src.width)
        s2 = src.read().reshape(n_months, len(band_names), -1)[:, idx]          # (T, 10, P)
    with rasterio.open(Path(chip_dir) / f"{chip_id}_s1rtc.tif") as src:
        s1 = src.read().astype(np.float32).reshape(n_months, 2, -1)             # (T, 2, P)
    return (np.ascontiguousarray(s2.transpose(2, 0, 1)).astype(np.float32),
            np.ascontiguousarray(s1.transpose(2, 0, 1)), hw)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def load_student(size: str, weights_dir=WEIGHTS, sha256: str | None = None):
    """A v2 pixel student from ``weights_dir/student_<size>.pt``, fetched from Hugging Face if absent.

    The checkpoint holds tensors and plain containers only, so it loads with ``weights_only=True``; its SHA-256 is
    checked against ``sha256`` when one is given. Built as TESSERA's ``load_model`` builds it, with dropout 0; the
    fusion MLP's fixed dropout of 0.2 is off once the caller puts the model in evaluation mode.
    """
    import torch

    from gfm4agri.embeddings.tessera_v2_model import PixelStudent

    path = Path(weights_dir) / f"student_{size}.pt"
    if not path.exists():
        from huggingface_hub import hf_hub_download

        path.parent.mkdir(parents=True, exist_ok=True)
        got = hf_hub_download(f"geotessera/TESSERA-V-2.0-2B-{STUDENTS[size]}", f"ckpt/student_{size}.pt")
        tmp = path.with_suffix(".pt.tmp")
        tmp.write_bytes(Path(got).read_bytes())
        tmp.replace(path)
    if sha256 is not None and _sha256(path) != sha256:
        raise ValueError(f"{path}: SHA-256 differs from the registry's {sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    a = payload.get("args", {}) or {}
    model = PixelStudent(
        repr_dim=int(a.get("repr_dim", 128)), latent_dim=int(a.get("latent_dim", 64)),
        num_layers=int(a.get("num_layers", 4)), nhead=int(a.get("nhead", 4)),
        dim_feedforward=int(a.get("dim_feedforward", 1024)), dropout=0.0,
        max_seq_len=int(a.get("max_seq_len", 256)),
        matryoshka_dims=tuple(int(d) for d in str(a.get("matryoshka_dims", "16,32,64,128")).split(",")),
        enable_qk_norm=bool(a.get("enable_qk_norm", False)),
    )
    model.load_state_dict(payload["model"])
    return model


def _encode_batch(model, s2: np.ndarray, s1: np.ndarray, doy: np.ndarray, device, autocast) -> np.ndarray:
    """``(B, 128)`` for ``B`` pixels in TESSERA's raw units, prepared as v2's ``encode_pixels`` prepares them.

    S1 validity comes from the raw values, both sensors are standardised with v2's statistics (S1 with the ascending
    stream's), each pixel's valid frames go to the bucket of 8, 16, ... steps its count falls in, padded by v2's
    pattern, with the raw day of year; pixels sharing a bucket pair go through one forward pass.
    """
    import torch

    from gfm4agri.embeddings import tessera_v2_model as tm

    s1_valid = np.any(s1 != 0, axis=-1)
    s1n = (s1 - tm.S1A_BAND_MEAN) / (tm.S1A_BAND_STD + 1e-9)
    s2_valid = np.any(s2 != 0, axis=-1)
    keys = (tm._vec_get_bin_size(s2_valid.sum(1)).astype(np.int64) * 1000
            + tm._vec_get_bin_size(s1_valid.sum(1)).astype(np.int64))
    out = np.empty((len(s2), N_DIMS), np.float32)
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        s2_b, s1_b = max(int(key // 1000), 1), max(int(key % 1000), 1)
        s2_in = np.zeros((len(idx), s2_b, 11), np.float32)
        s1_in = np.zeros((len(idx), s1_b, 3), np.float32)
        if key // 1000 > 0:
            src = tm._build_source_indices(s2_valid[idx], s2_b)
            g = np.take_along_axis(s2[idx], src[:, :, None].repeat(10, axis=2), axis=1)
            s2_in[:, :, :10] = (g - tm.S2_BAND_MEAN) / (tm.S2_BAND_STD + 1e-9)
            s2_in[:, :, 10] = doy[src]
        if key % 1000 > 0:
            src = tm._build_source_indices(s1_valid[idx], s1_b)
            s1_in[:, :, :2] = np.take_along_axis(s1n[idx], src[:, :, None].repeat(2, axis=2), axis=1)
            s1_in[:, :, 2] = doy[src]
        with autocast:
            emb = model.encode(torch.from_numpy(s2_in).to(device), torch.from_numpy(s1_in).to(device))
        out[idx] = emb.float().cpu().numpy()
    return out


def encode_pixels(model, s2, s1_db, doy, device, precision: str = "bf16-mixed",
                  batch: int = 16384) -> np.ndarray:
    """``(P, 128)`` float32 embeddings of ``P`` pixels, no gradients; NaN where a pixel has no input at all."""
    import torch

    s2 = np.asarray(s2, dtype=np.float32)
    s1 = s1_to_tessera(s1_db)
    autocast = torch.autocast(device.type, dtype=torch.bfloat16,
                              enabled=precision.startswith("bf16") and device.type == "cuda")
    out = np.empty((len(s2), N_DIMS), np.float32)
    with torch.no_grad():
        for i in range(0, len(s2), batch):
            out[i:i + batch] = _encode_batch(model, s2[i:i + batch], s1[i:i + batch], doy, device, autocast)
    empty = ~np.any(s2 != 0, axis=(1, 2)) & ~np.any(s1 != 0, axis=(1, 2))
    out[empty] = np.nan
    return out


class ChipEncoder:
    """A frozen v2 student over whole chips, as :mod:`gfm4agri.pipeline.raster_cache` drives it.

    ``read`` loads a chip's composites, on a reader thread; ``encode`` runs the student and returns
    ``(128, H, W)`` float32 on the chip's own grid.
    """

    def __init__(self, student: str, checkpoint_sha256: str, weights_dir=WEIGHTS, *, band_names,
                 n_months: int, year: int, precision: str = "bf16-mixed", batch: int = 16384,
                 device: str | None = None) -> None:
        import torch

        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = load_student(student, weights_dir, checkpoint_sha256).to(self.device).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.band_names, self.n_months = list(band_names), int(n_months)
        self.doy = month_doy(year, n_months)
        self.precision, self.batch = precision, int(batch)

    def read(self, chip_dir, chip_id: str):
        return chip_inputs(chip_dir, chip_id, self.band_names, self.n_months)

    def encode(self, inputs) -> np.ndarray:
        s2, s1, (h, w) = inputs
        emb = encode_pixels(self.model, s2, s1, self.doy, self.device, self.precision, self.batch)
        return np.ascontiguousarray(emb.T.reshape(N_DIMS, h, w))
