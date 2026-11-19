"""BatchTopK sparse autoencoder and its training loop, adapted from geosae.

Source: https://github.com/EarthyScience/geosae, commit 13b3e9ca2c1e0e491399d67cba6e6672fdb019cc (10 November 2026),
MIT licence, copyright (c) 2026 Vitus Benson; the licence text is at the end of this file. Taken from
``geosae/models/topk_sae.py`` (``TopKSAE`` with ``batch_topk=True``, ``pre_bias=True``, decoder renormalisation,
``W_enc = W_dec^T`` initialisation, the frozen random decoder control), ``geosae/train/loss.py`` (reconstruction
plus AuxK), ``geosae/train/trainer.py`` (Adam with beta1 = 0, linear warm-up and cosine decay, gradient clipping,
best validation checkpoint) and ``geosae/eval/metrics.py`` (the inference threshold at the ``1 - k/M`` quantile of
the dense activations).

What differs, and why:

* **Two-stage batch top-k.** The batch-wide selection of the ``k * B`` largest pre-activations is done on the top
  ``row_cap`` of each pixel. It equals geosae's selection unless one pixel holds more than ``row_cap`` of the batch
  winners; such pixels are counted (``capped``) and logged. On the hub's CPU a full ``topk`` over ``B * M`` values
  took 750 ms of an 850 ms step.
* **Batches are uniform random pixels** from the sample in memory, not geosae's anchor-patch batches.
* **CPU, float32.** The hub's shared GPU was full; no bf16 autocast.
* No Matryoshka, ReLU + l1 or coordinate-aware variants.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class BatchTopKSAE(nn.Module):
    """``x -> z = BatchTopK(W_enc (x - b) + b_enc) -> x_hat = W_dec z + b``, with unit-norm decoder atoms."""

    def __init__(self, input_dim: int, latent_dim: int, k: int, *, freeze_decoder: bool = False,
                 seed: int = 0, row_cap: int = 128, n_batches_to_dead: int = 25,
                 min_activation_beta: float = 0.999) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.input_dim, self.latent_dim, self.k = int(input_dim), int(latent_dim), int(k)
        self.freeze_decoder, self.row_cap = bool(freeze_decoder), int(row_cap)
        self.n_batches_to_dead, self.beta = int(n_batches_to_dead), float(min_activation_beta)
        self.encoder = nn.Linear(self.input_dim, self.latent_dim, bias=True)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim, bias=False)
        self.b_dec = nn.Parameter(torch.zeros(self.input_dim))
        self.register_buffer("threshold", torch.full((), -1.0))
        self.register_buffer("num_batches_not_active", torch.zeros(self.latent_dim))
        self.capped = 0
        with torch.no_grad():
            if self.freeze_decoder:
                # geosae's frozen-random-decoder control: i.i.d. Gaussian atoms on the unit sphere, frozen with the
                # bias, so that only the encoder learns.
                w = torch.randn_like(self.decoder.weight)
                self.decoder.weight.copy_(w / w.norm(dim=0, keepdim=True).clamp_min(1e-8))
                self.decoder.weight.requires_grad_(False)
                self.b_dec.requires_grad_(False)
            else:
                self._normalize_decoder()
            self.encoder.weight.copy_(self.decoder.weight.t())
            self.encoder.bias.zero_()

    @torch.no_grad()
    def _normalize_decoder(self) -> None:
        w = self.decoder.weight
        w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Dense pre-activations ``(n, M)``."""
        return self.encoder(x - self.b_dec)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z) + self.b_dec

    def sparsify(self, zd: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return zd * (zd > self.threshold)
        n = zd.shape[0]
        cap = min(self.row_cap, self.latent_dim)
        v, i = torch.topk(zd, cap, dim=-1)
        flat = v.reshape(-1)
        _, top = torch.topk(flat, self.k * n)
        keep = torch.zeros_like(flat, dtype=torch.bool)
        keep[top] = True
        keep = keep.view(n, cap)
        self.capped = int(keep.all(dim=1).sum())
        z = torch.zeros_like(zd).scatter(1, i, v * keep)
        self._update_threshold(z)
        return z

    @torch.no_grad()
    def _update_threshold(self, z: torch.Tensor) -> None:
        m = z.masked_fill(z <= 0, float("inf")).min()
        if torch.isinf(m):
            return
        if self.threshold < 0:
            self.threshold.copy_(m)
        else:
            self.threshold.mul_(self.beta).add_((1 - self.beta) * m)

    @torch.no_grad()
    def _update_dead(self, z: torch.Tensor) -> None:
        fired = (z > 0).any(dim=0)
        self.num_batches_not_active += (~fired).float()
        self.num_batches_not_active[fired] = 0

    def forward(self, x: torch.Tensor):
        if self.training and not self.freeze_decoder:
            self._normalize_decoder()
        zd = self.encode(x)
        z = self.sparsify(zd)
        recon = self.decode(z)
        if self.training:
            self._update_dead(z)
        return recon, z, zd

    @torch.no_grad()
    def codes(self, x: torch.Tensor, batch: int = 65536) -> torch.Tensor:
        """Sparse codes in eval mode, in batches."""
        self.eval()
        return torch.cat([self.sparsify(self.encode(x[s:s + batch])) for s in range(0, len(x), batch)])

    def atoms(self) -> np.ndarray:
        """Decoder atoms as ``(M, D)`` rows."""
        return self.decoder.weight.detach().cpu().numpy().T.copy()


def sae_loss(model: BatchTopKSAE, x, recon, zd, *, aux_penalty: float, top_k_aux: int):
    """MSE reconstruction plus AuxK: the largest pre-activations of dead features explain the residual."""
    rec = F.mse_loss(recon, x)
    dead = model.num_batches_not_active >= model.n_batches_to_dead
    n_dead = int(dead.sum())
    aux = torch.zeros((), device=x.device)
    if aux_penalty > 0 and n_dead > 0:
        kk = min(top_k_aux, n_dead)
        zdd = zd[:, dead]
        v, i = torch.topk(zdd, kk, dim=-1)
        z_aux = torch.zeros_like(zd)
        z_aux[:, dead] = torch.zeros_like(zdd).scatter(-1, i, v)
        aux = F.mse_loss(model.decoder(z_aux), (x - recon).detach())
    return rec + aux_penalty * aux, {"recon_mse": float(rec), "aux": float(aux), "n_dead": n_dead}


@torch.no_grad()
def calibrate_threshold(model: BatchTopKSAE, x: torch.Tensor, max_sample: int = 4_000_000, seed: int = 0) -> float:
    """geosae's inference threshold: the ``1 - k/M`` quantile of the dense activations, so that ``k`` of ``M``
    features are active per pixel on average."""
    model.eval()
    acts = torch.cat([model.encode(x[s:s + 65536]).reshape(-1) for s in range(0, len(x), 65536)])
    if acts.numel() > max_sample:
        g = torch.Generator().manual_seed(seed)
        acts = acts[torch.randint(acts.numel(), (max_sample,), generator=g)]
    q = float(torch.quantile(acts.double(), 1.0 - model.k / model.latent_dim))
    model.threshold.fill_(q)
    return q


@torch.no_grad()
def reconstruction_metrics(recon: torch.Tensor, x: torch.Tensor, z: torch.Tensor | None = None,
                           weights: torch.Tensor | None = None) -> dict:
    """Fraction of variance explained (pooled over dimensions), cosine, L0 and the share of dead features."""
    w = torch.ones(len(x), dtype=torch.float64) if weights is None else weights.double()
    w = w / w.sum()
    xd, rd = x.double(), recon.double()
    mean = (w[:, None] * xd).sum(0)
    sse = (w * ((xd - rd) ** 2).sum(1)).sum()
    sst = (w * ((xd - mean) ** 2).sum(1)).sum()
    cos = F.cosine_similarity(xd, rd, dim=1)
    out = {"fve": float(1 - sse / sst), "cosine": float((w * cos).sum())}
    if z is not None:
        out["l0"] = float((w * (z > 0).sum(1).double()).sum())
        out["dead_share"] = float(((z > 0).sum(0) == 0).double().mean())
    return out


def _lr(step: int, warmup: int, total: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))


def train(model: BatchTopKSAE, x_train: torch.Tensor, x_val: torch.Tensor, cfg: dict, out_dir: Path,
          seed: int) -> dict:
    """Train on uniform random pixels of ``x_train``; keep the state of best validation FVE as ``best.pt``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    g = torch.Generator().manual_seed(seed)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(params, lr=cfg["lr"], betas=(cfg["beta1"], cfg["beta2"]), eps=1e-8)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: _lr(s, cfg["warmup"], cfg["steps"]))
    best, best_state, log = -float("inf"), None, []
    t0, run = time.time(), {"recon_mse": 0.0, "aux": 0.0, "n_dead": 0.0, "capped": 0.0}
    for step in range(cfg["steps"]):
        model.train()
        xb = x_train[torch.randint(len(x_train), (cfg["batch"],), generator=g)]
        recon, z, zd = model(xb)
        loss, m = sae_loss(model, xb, recon, zd, aux_penalty=cfg["aux_penalty"], top_k_aux=cfg["top_k_aux"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg["clip_grad_norm"]:
            torch.nn.utils.clip_grad_norm_(params, cfg["clip_grad_norm"])
        opt.step()
        sched.step()
        for key in ("recon_mse", "aux", "n_dead"):
            run[key] += m[key]
        run["capped"] += model.capped
        if (step + 1) % cfg["val_every"] == 0 or step + 1 == cfg["steps"]:
            thr = calibrate_threshold(model, x_val)
            zv = model.codes(x_val)
            val = reconstruction_metrics(model.decode(zv), x_val, zv)
            n = cfg["val_every"]
            rec = {"step": step + 1, "seconds": round(time.time() - t0, 1), "threshold": thr,
                   "train": {k: v / n for k, v in run.items()}, "val": val}
            log.append(rec)
            run = dict.fromkeys(run, 0.0)
            print(f"  step {step + 1} val_fve={val['fve']:.4f} cos={val['cosine']:.4f} l0={val['l0']:.1f} "
                  f"dead={val['dead_share']:.3f} capped/step={rec['train']['capped']:.2f} "
                  f"({rec['seconds']} s)", flush=True)
            if val["fve"] > best:
                best = val["fve"]
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    torch.save(best_state, out_dir / "best.pt")
    (out_dir / "log.json").write_text(json.dumps(log, indent=1))
    return {"best_val_fve": best, "seconds": round(time.time() - t0, 1),
            "best_step": max(log, key=lambda r: r["val"]["fve"])["step"]}


GEOSAE_LICENCE = """
MIT License

Copyright (c) 2026 Vitus Benson

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
