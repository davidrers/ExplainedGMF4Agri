"""Encode the pilot's monthly composites with TESSERA v1.1 and v2, frozen: one float16 (128, 224, 224) .npy per chip.

v1.1 (``v11_aws``, ``v11_mpc``): TESSERA's v1.1 model and checkpoints. Each pixel's valid frames are resampled to
the 16-step bucket with TESSERA's ``build_resample_indices``, standardised with the checkpoint's own statistics
(S1 with the ascending ones), and encoded under bf16 autocast as ``infer_v1_1.py`` does; the first 128 of the 192
output dimensions are kept.

v2 (``v2_nano`` ... ``v2_large``): the v2 pixel students through their own ``encode_pixels``, which standardises,
buckets and pads exactly as at training time and returns the 128-d Matryoshka embedding, in float32.

    poetry run python experiments/2026-10-14_tessera_composites/spike/embed_composites.py [--models ...]
"""

from __future__ import annotations

import argparse
import importlib.util
import time

import numpy as np
import torch

from common import DOY, OUT, WEIGHTS, chip_ids, composite_inputs, tessera_path

V11 = {"v11_aws": "aws", "v11_mpc": "mpc"}
V2 = {"v2_nano": "nano", "v2_small": "small", "v2_medium": "medium", "v2_large": "large"}
BIN = 16


def load_v11(source: str, device):
    tessera_path("tessera_infer_QAT", "src")
    from datasets.v1_1_norm_stats import get_stats
    from models.ssl_model_v1_1 import build_v1_1_inference_model

    from common import TESSERA
    spec = importlib.util.spec_from_file_location("v11_cfg", TESSERA / "tessera_infer_QAT/configs/v1_1_infer_config.py")
    cfg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cfg)
    model = build_v1_1_inference_model(dict(cfg.config, data_source=source), device)
    ck = torch.load(WEIGHTS / f"v1_1/tessera_v1_1_{source}_encoder.pt", map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in ck["model_state"].items()
          if not k.startswith(("projector.", "segmented_matryoshka_projector."))}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)
    return model.eval(), get_stats(source)


def encode_v11(model, st, x: dict, device, batch: int = 8192) -> np.ndarray:
    from datasets.ssl_dataset_v1_1 import build_resample_indices

    def bucket(v, valid, mean, std):
        v = (v - mean) / (std + 1e-9)
        out = np.zeros((v.shape[0], BIN, v.shape[2] + 1), dtype=np.float32)
        n = valid.sum(1)
        for k in np.unique(n):
            if k == 0:
                continue
            pix = np.where(n == k)[0]
            order = np.argsort(~valid[pix], axis=1, kind="stable")[:, :k]
            idx = order[:, build_resample_indices(int(k), BIN)]
            out[pix, :, :-1] = np.take_along_axis(v[pix], idx[..., None], axis=1)
            out[pix, :, -1] = DOY[idx]
        return out, n

    s2x, n2 = bucket(x["s2"], x["s2_valid"], st["s2_mean"], st["s2_std"])
    s1x, n1 = bucket(x["s1"], x["s1_valid"], st["s1a_mean"], st["s1a_std"])
    outs = []
    with torch.no_grad():
        for i in range(0, s2x.shape[0], batch):
            a = torch.from_numpy(s2x[i:i + batch]).to(device)
            b = torch.from_numpy(s1x[i:i + batch]).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                e = model(a, b)
            outs.append(e[:, :128].float().cpu().numpy())
    emb = np.concatenate(outs)
    emb[(n2 == 0) & (n1 == 0)] = np.nan
    return emb


def load_v2(size: str, device):
    tessera_path("tessera_infer_v2", "student")
    from model import PixelStudent

    payload = torch.load(WEIGHTS / f"v2/student_{size}.pt", map_location="cpu", weights_only=True)
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
    return model.to(device).eval(), None


def encode_v2(model, _st, x: dict, device) -> np.ndarray:
    from infer import encode_pixels

    emb = encode_pixels(model, x["s2"], DOY, s1_asc_bands=x["s1"], s1_asc_doys=DOY,
                        s2_masks=x["s2_valid"].astype(np.uint8), batch_pixels=8192, device=device)
    emb[~x["s2_valid"].any(1) & ~x["s1_valid"].any(1)] = np.nan
    return emb


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=[*V11, *V2])
    args = ap.parse_args()
    device = torch.device("cuda")
    ids = chip_ids()
    for name in args.models:
        if name in V11:
            model, st = load_v11(V11[name], device)
            enc = encode_v11
        else:
            model, st = load_v2(V2[name], device)
            enc = encode_v2
        n_params = sum(p.numel() for p in model.parameters())
        out = OUT / "emb" / name
        out.mkdir(parents=True, exist_ok=True)
        t0, done = time.time(), 0
        for cid in ids:
            p = out / f"{cid}.npy"
            if p.exists():
                continue
            emb = enc(model, st, composite_inputs(cid), device)
            np.save(p, emb.T.reshape(128, 224, 224).astype(np.float16))
            done += 1
        print(f"{name}: {n_params / 1e6:.2f} M parameters, {done} chips encoded in {time.time() - t0:.0f} s",
              flush=True)
        del model
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
