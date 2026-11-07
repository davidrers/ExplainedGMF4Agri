"""TESSERA v2 Medium inside the training loop, on whole chips or crops of them.

:func:`prepare` and :func:`encode` are copied from ``../2026-10-15_kshot_finetune/pixelft.py``, their code unchanged, where the
fine-tuned TESSERA arm used them (its raster reproduced the frozen cache's): ``prepare`` turns pixels' composites into
TESSERA's bucketed inputs on the CPU exactly as the core's ``tessera_v2._encode_batch`` does, and ``encode`` runs the
student under bf16 autocast, with gradients when asked, in gradient-checkpointed chunks so memory holds one chunk's
activations whatever the number of pixels. Here every pixel of the chip (or of its crop) goes through the encoder,
since the U-Net's loss reaches every pixel's embedding through its convolutions.

:func:`load_student` is the core registry's v2 Medium student with its SHA-256 checked, and :func:`sample_stats` the
per-dimension mean and standard deviation of the pretrained student's output over a fixed sample of training chips,
the fixed standardisation of the trainable arm (the frozen arm reads its cache's statistics over every training
chip).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

BACKBONE = "tessera_v2_medium"


def prepare(s2: np.ndarray, s1_db: np.ndarray, doy: np.ndarray) -> dict:
    """TESSERA's inputs for ``N`` pixels, grouped by bucket pair, as ``tessera_v2._encode_batch`` builds them.

    ``s2``: ``(N, T, 10)`` DN in TESSERA's band order; ``s1_db``: ``(N, T, 2)`` our dB. Returns ``groups``, a list of
    ``(positions, s2_in, s1_in)`` tensors, and ``empty``, the pixels with no S2 and no S1 input at all, which the
    core's cache holds as NaN and its datamodule turns into 0.
    """
    from gfm4agri.embeddings import tessera_v2_model as tm
    from gfm4agri.embeddings.tessera_v2 import s1_to_tessera

    s2 = np.asarray(s2, dtype=np.float32)
    s1 = s1_to_tessera(s1_db)
    s1_valid = np.any(s1 != 0, axis=-1)
    s1n = (s1 - tm.S1A_BAND_MEAN) / (tm.S1A_BAND_STD + 1e-9)
    s2_valid = np.any(s2 != 0, axis=-1)
    keys = (tm._vec_get_bin_size(s2_valid.sum(1)).astype(np.int64) * 1000
            + tm._vec_get_bin_size(s1_valid.sum(1)).astype(np.int64))
    groups = []
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
        groups.append((torch.from_numpy(idx), torch.from_numpy(s2_in), torch.from_numpy(s1_in)))
    empty = ~np.any(s2 != 0, axis=(1, 2)) & ~np.any(s1 != 0, axis=(1, 2))
    return {"groups": groups, "empty": torch.from_numpy(empty), "n": len(s2)}


def encode(student, prep: dict, *, grad: bool, chunk: int = 16384) -> torch.Tensor:
    """``(N, 128)`` float32 embeddings of the prepared pixels, 0 where a pixel has no input (the core's NaN -> 0).

    Under bf16 autocast, as the core's raster cache. With ``grad``, each chunk of up to ``chunk`` pixels goes through
    ``torch.utils.checkpoint``: its activations are recomputed in the backward pass instead of kept, so memory holds
    one chunk's activations whatever the number of pixels. Chunking does not change the result; each pixel is
    encoded on its own.
    """
    from torch.utils.checkpoint import checkpoint

    device = next(student.parameters()).device
    out = torch.zeros(prep["n"], 128, device=device, dtype=torch.float32)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
        for pos, s2_in, s1_in in prep["groups"]:
            pos = pos.to(device)
            for i in range(0, len(pos), chunk):
                a, b = s2_in[i:i + chunk].to(device), s1_in[i:i + chunk].to(device)
                e = (checkpoint(student.encode, a, b, use_reentrant=False) if grad
                     else student.encode(a, b))
                out = out.index_copy(0, pos[i:i + chunk], e.float())
    empty = prep["empty"].to(device)
    return torch.where(empty[:, None], torch.zeros((), device=device), out)


def load_student():
    """The core registry's TESSERA v2 Medium student, with its checkpoint's SHA-256 checked."""
    from gfm4agri.benchmark.backbones import get_backbone
    from gfm4agri.embeddings.tessera_v2 import WEIGHTS
    from gfm4agri.embeddings.tessera_v2 import load_student as _load

    c = get_backbone(BACKBONE).computed
    return _load(c["student"], WEIGHTS, c["checkpoint_sha256"])


def sample_stats(student, chips_dir: Path, chip_ids: list[str], band_names, n_months: int, year: int,
                 n_chips: int = 128) -> dict:
    """Mean and standard deviation per dimension of the pretrained student's output over every ``k``-th training
    chip (``n_chips`` of them, in the split's order), every pixel; float64 sums, as ``export.band_stats``."""
    from gfm4agri.embeddings.tessera_v2 import chip_inputs, month_doy

    doy = month_doy(year, n_months)
    step = max(1, len(chip_ids) // n_chips)
    pick = chip_ids[::step][:n_chips]
    s = s2 = None
    n = 0
    with torch.no_grad():
        for cid in pick:
            a, b, _ = chip_inputs(chips_dir, cid, band_names, n_months)
            e = encode(student, prepare(a, b, doy), grad=False).double()
            s = e.sum(0) if s is None else s + e.sum(0)
            s2 = (e * e).sum(0) if s2 is None else s2 + (e * e).sum(0)
            n += len(e)
    mean = (s / n).cpu().numpy()
    std = np.sqrt(np.maximum((s2 / n).cpu().numpy() - mean ** 2, 1e-12))
    return {"computed_on": f"{len(pick)} training chips (every {step}th), pretrained student", "n_chips": len(pick),
            "means": [round(float(x), 5) for x in mean], "stds": [round(float(x), 5) for x in std]}
