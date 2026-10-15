"""TESSERA v2's pixel student and its input bucketing, vendored from TESSERA.

Copied verbatim from https://github.com/ucam-eo/tessera at commit e368539f7c4711e5f16fdd09801573830e34baf9:

* ``tessera_infer_v2/student/model.py``: everything from its imports to ``PixelStudentV11``;
* ``tessera_infer_v2/student/infer.py``: ``BIN_EDGES`` to ``_build_source_indices``, the bucketing and
  padding rule of ``encode_pixels``.

Nothing here is edited; :mod:`gfm4agri.embeddings.tessera_v2` maps the chip set's composites onto it. Under
TESSERA's licence:

    MIT License

    Copyright (c) 2025 Frank Feng

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
"""

import math
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----- Standardization stats (must match training preprocessing) -----
# S1 ascending and descending use independent stats — applied per-source
# BEFORE merging into the single S1 stream the model consumes.
S2_BAND_MEAN = np.array(
    [1633.0042, 1341.1090, 1539.5536, 3054.8269, 3117.4658,
     2004.1648, 2694.7275, 2945.1504, 2266.6079, 1657.3094],
    dtype=np.float32,
)
S2_BAND_STD = np.array(
    [1999.4603, 2014.7549, 1929.2201, 1754.2493, 1649.9807,
     1936.8988, 1748.6041, 1708.6991, 1207.5250, 1108.6046],
    dtype=np.float32,
)
S1A_BAND_MEAN = np.array([5909.3921, 3405.0322], dtype=np.float32)
S1A_BAND_STD  = np.array([1507.1750, 1531.2615], dtype=np.float32)
S1D_BAND_MEAN = np.array([5816.1382, 3277.7576], dtype=np.float32)
S1D_BAND_STD  = np.array([1554.6475, 1546.4733], dtype=np.float32)

# S2 input channel order:
S2_BAND_ORDER = ["B04", "B02", "B03", "B08", "B8A", "B05", "B06", "B07", "B11", "B12"]
# S1 input channel order:
S1_BAND_ORDER = ["VV", "VH"]


# ===== Vendored building blocks (no qk_norm, ReLU, attention pooling) =====


class TemporalPositionalEncoder(nn.Module):
    """Sinusoidal positional encoding using the (raw integer) DOY value."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model

    def forward(self, doy: torch.Tensor) -> torch.Tensor:
        position = doy.unsqueeze(-1).float()
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float, device=doy.device)
            * -(math.log(10000.0) / self.d_model)
        )
        pe = torch.zeros(doy.shape[0], doy.shape[1], self.d_model, device=doy.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return pe


class AttentionPooling(nn.Module):
    """Plain single-head softmax attention pool over T.

    A single `Linear(D, 1) -> softmax -> weighted sum`. No GRU, no
    LayerNorm.
    """

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.query = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, D)
        B, T, D = x.shape
        if T == 0:
            return torch.zeros(B, D, device=x.device, dtype=x.dtype)
        if T == 1:
            return x.squeeze(1)
        w = torch.softmax(self.query(x), dim=1)
        return (w * x).sum(dim=1)


class QKNormEncoderLayer(nn.Module):
    """Transformer encoder layer identical to `nn.TransformerEncoderLayer`
    (post-LN, ReLU FFN, batch_first) EXCEPT it applies per-head RMSNorm to Q and
    K before scaled-dot-product attention (QK-norm). QK-norm is the ONLY
    difference, so a model built from these is directly comparable to one built
    from the standard layer (≈same param count; q/k/v as 3 separate Linears
    matches the standard layer's combined in_proj, plus two tiny RMSNorm gains)."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int,
                 dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"d_model {d_model} % nhead {nhead} != 0"
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.q_norm = nn.RMSNorm(self.head_dim)
        self.k_norm = nn.RMSNorm(self.head_dim)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self._attn_p = float(dropout)

    def _sa(self, x: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        H, hd = self.nhead, self.head_dim
        q = self.q_proj(x).view(B, T, H, hd)
        k = self.k_proj(x).view(B, T, H, hd)
        v = self.v_proj(x).view(B, T, H, hd)
        q = self.q_norm(q); k = self.k_norm(k)                    # per-head RMSNorm (QK-norm)
        q = q.transpose(1, 2); k = k.transpose(1, 2); v = v.transpose(1, 2)  # (B,H,T,hd)
        o = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self._attn_p if self.training else 0.0)
        o = o.transpose(1, 2).reshape(B, T, D)
        return self.out_proj(o)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm1(x + self.dropout1(self._sa(x)))            # post-LN
        x = self.norm2(x + self.dropout2(self.linear2(self.dropout(F.relu(self.linear1(x))))))
        return x


class TransformerEncoder(nn.Module):
    """Per-pixel band embedding + DOY positional + Transformer + attention pooling.

    `enable_qk_norm=False` (default): standard `nn.TransformerEncoderLayer`
    (ReLU, no qk_norm). `True`: `QKNormEncoderLayer` stack (same structure +
    per-head RMSNorm on Q/K).
    """

    def __init__(self, band_num: int, latent_dim: int, nhead: int = 4,
                 num_encoder_layers: int = 3, dim_feedforward: int = 1024,
                 dropout: float = 0.1, max_seq_len: int = 256,
                 enable_qk_norm: bool = False) -> None:
        super().__init__()
        input_dim = band_num
        self.embedding = nn.Sequential(
            nn.Linear(input_dim, latent_dim * 4),
            nn.ReLU(),
            nn.Linear(latent_dim * 4, latent_dim * 4),
        )
        self.temporal_encoder = TemporalPositionalEncoder(d_model=latent_dim * 4)

        self.enable_qk_norm = bool(enable_qk_norm)
        if self.enable_qk_norm:
            self.transformer_encoder = nn.ModuleList([
                QKNormEncoderLayer(latent_dim * 4, nhead, dim_feedforward, dropout)
                for _ in range(num_encoder_layers)
            ])
        else:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=latent_dim * 4, nhead=nhead,
                dim_feedforward=dim_feedforward, dropout=dropout,
                activation="relu", batch_first=True,
            )
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)
        self.attn_pool = AttentionPooling(latent_dim * 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, band_num + 1) — last channel is the (raw integer) DOY value
        bands = x[:, :, :-1]
        doy = x[:, :, -1]
        bands_embedded = self.embedding(bands)
        temporal_encoding = self.temporal_encoder(doy)
        x = bands_embedded + temporal_encoding
        if self.enable_qk_norm:
            for layer in self.transformer_encoder:
                x = layer(x)
        else:
            x = self.transformer_encoder(x)
        return self.attn_pool(x)


class PixelStudent(nn.Module):
    """2-backbone pixel encoder. emb_dim = 128 (Matryoshka-ordered).

    Inputs (per pixel):
        s2_x : (B, T_s2, 11)   10 bands + 1 DOY (raw integer 1..365)
        s1_x : (B, T_s1,  3)    2 bands + 1 DOY (raw integer; s1a + s1d MERGED)

    encode() returns (B, 128). The first K dims are usable on their own for
    K in {16, 32, 64, 128}.
    """

    def __init__(
        self,
        repr_dim: int = 128,
        latent_dim: int = 64,
        num_layers: int = 4,
        nhead: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        max_seq_len: int = 256,
        matryoshka_dims: Tuple[int, ...] = (16, 32, 64, 128),
        enable_qk_norm: bool = False,
    ) -> None:
        super().__init__()
        self.repr_dim = int(repr_dim)
        self.enable_qk_norm = bool(enable_qk_norm)
        self.matryoshka_dims = tuple(int(d) for d in matryoshka_dims)
        for d in self.matryoshka_dims:
            if d > repr_dim:
                raise ValueError(f"matryoshka cut {d} > repr_dim {repr_dim}")

        def make_enc(band_num: int) -> TransformerEncoder:
            return TransformerEncoder(
                band_num=band_num, latent_dim=latent_dim,
                nhead=nhead, num_encoder_layers=num_layers,
                dim_feedforward=dim_feedforward, dropout=dropout,
                max_seq_len=max_seq_len, enable_qk_norm=enable_qk_norm,
            )

        self.s2_backbone = make_enc(10)
        self.s1_backbone = make_enc(2)

        backbone_out = latent_dim * 4
        fused_in = 2 * backbone_out
        self.dim_reducer = nn.Sequential(
            nn.Linear(fused_in, fused_in * 2),
            nn.LayerNorm(fused_in * 2),
            nn.ReLU(inplace=False),
            nn.Dropout(0.2),
            nn.Linear(fused_in * 2, repr_dim),
            # Final non-affine LayerNorm: normalizes the 128-d output to
            # per-pixel mean 0 / std 1 (well-conditioned for downstream use).
            # No learnable params -> state_dict + param count unchanged.
            nn.LayerNorm(repr_dim, elementwise_affine=False),
        )

    def encode(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        s2 = self.s2_backbone(s2_x)
        s1 = self.s1_backbone(s1_x)
        fused = torch.cat([s2, s1], dim=-1)
        return self.dim_reducer(fused)

    def forward(self, s2_x: torch.Tensor, s1_x: torch.Tensor) -> torch.Tensor:
        return self.encode(s2_x, s1_x)


# Backwards-compatible alias.
PixelStudentV11 = PixelStudent


# ----- from tessera_infer_v2/student/infer.py -----

BIN_EDGES = list(range(8, 257, 8))   # [8, 16, 24, ..., 256]


def get_bin_size(n_obs: int) -> int:
    if n_obs <= 0:
        return 0
    for b in BIN_EDGES:
        if n_obs <= b:
            return b
    return BIN_EDGES[-1]


def _vec_get_bin_size(n_obs: np.ndarray) -> np.ndarray:
    out = np.full_like(n_obs, BIN_EDGES[-1])
    out[n_obs <= 0] = 0
    for b in reversed(BIN_EDGES):
        out = np.where((n_obs > 0) & (n_obs <= b), b, out)
    return out


def _pad_pattern(n: int, B: int) -> np.ndarray:
    """(B,) int64 indices into [0, n) reproducing the training pad_to_bin."""
    if n == 0:
        return np.zeros(B, dtype=np.int64)
    if n >= B:
        return np.linspace(0, n - 1, B, dtype=np.int64)
    remain = B - n
    if remain <= n:
        groups = np.array_split(np.arange(n), remain)
        fill = np.array([gp[len(gp) // 2] for gp in groups], dtype=np.int64)
    else:
        fill = (np.arange(remain) % n).astype(np.int64)
    return np.concatenate([np.arange(n, dtype=np.int64), fill])


def _build_source_indices(valid_per_pix: np.ndarray, B: int) -> np.ndarray:
    """Vectorized: O(unique_n) loop instead of O(G).

    Within a bin group all pixels have n_obs in (prev_bin, B], so at most
    `bin_step` distinct n values exist (typically <=8). We:
      1) stable-sort `~valid` so valid time-indices come first per row,
      2) group pixels by n_obs and compute the pad pattern once per n-subgroup,
      3) gather sorted_pos[pix, pattern] in one shot per n-subgroup.

    valid_per_pix: (G, T) bool. Returns (G, B) int64.
    """
    G, T = valid_per_pix.shape
    src = np.zeros((G, B), dtype=np.int64)
    if G == 0 or B == 0:
        return src
    n_per = valid_per_pix.sum(axis=1).astype(np.int64)
    sorted_pos = np.argsort(~valid_per_pix, axis=1, kind="stable").astype(np.int64)
    unique_n, inverse = np.unique(n_per, return_inverse=True)
    for ki, n_val in enumerate(unique_n):
        n = int(n_val)
        if n == 0:
            continue
        pix = np.where(inverse == ki)[0]
        pattern = _pad_pattern(n, B)
        src[pix] = sorted_pos[pix][:, pattern]
    return src
