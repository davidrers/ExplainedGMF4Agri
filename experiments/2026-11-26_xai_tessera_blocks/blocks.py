"""The focused run: occlusion, SHAP and Integrated Gradients on blocks of three-month windows, band groups and
vegetation indices, for correctly predicted test fields (``sample/``).

    poetry run python experiments/2026-11-26_xai_tessera_blocks/blocks.py [--limit 1] [--pilot]

The chain (``chain.py``), the score (the field-interior mean of log p(species)), the replacements and the reference
(the species-balanced training mean, ``results/<name>/EE_2021_rev/step0``, copied with its checksum from
``experiments/2026-11-20_xai_tessera``) are those of the first run. Everything is on the field's own pixels.

**Blocks** (same definitions as the AUC map, ``auc_map.py``):

* 10 windows: three consecutive months, Jan-Mar to Oct-Dec, all 12 inputs;
* 8 properties: the 6 band groups (all months) and NDVI, NDMI (all months);
* 80 cells: window x property.

A band group or window is hidden by the mean replacement. An **index is hidden by a swap**: in the hidden months, the
field's index is set to the mean field's index while the sum of its two bands (the pair's brightness) is kept, so only
what the index measures is removed (NDVI: B08' = S (1 + v) / 2, B04' = S (1 - v) / 2 with S = B08 + B04 and v the
reference NDVI; NDMI the same on B8A, B11). EVI shares B04 and B08 with NDVI and cannot be removed apart from it; it
is on the AUC side only.

**Methods.** Occlusion: the fall of the score when one block is hidden (the 98 blocks above under the mean or the
swap, and the 10 windows under drop). SHAP: a game over the 12 months (``shap_permutations`` in antithetic pairs),
summed to windows in the analysis, and an exact game over the 6 band groups; indices cannot be players because they
share bands. Integrated Gradients from the mean field (``ig_steps``), the field's species only; each index pair's
attribution is also split, along the same path and by the chain rule, into the part through the index and the part
through the pair's sum (exact for that complement: the two parts add up to the pair's attribution).

Writes ``results/<name>/EE_2021_rev/explain/<chip>.npz``; a chip already written is skipped.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import chain  # noqa: E402
import explain_base as eb  # noqa: E402

MONTHS, FEAT, WINDOWS = chain.MONTHS, chain.FEATURES, chain.WINDOWS
WNAMES = [chain.window_name(w) for w in WINDOWS]
GROUPS = dict(chain.BAND_GROUPS)
INDEX_PAIRS = {"NDVI": (FEAT.index("B08"), FEAT.index("B04")), "NDMI": (FEAT.index("B8A"), FEAT.index("B11"))}
PROPS = list(GROUPS) + list(INDEX_PAIRS)
K_CLS = 20


def blocks() -> list[tuple[str, str, tuple]]:
    """``(kind, name, spec)``; spec ``("cells", mask, op)`` or ``("index", months, index)``."""
    out = []
    for (a, b), w in zip(WINDOWS, WNAMES):
        out.append(("window", w, ("cells", chain.cells_of(range(a, b)), "mean")))
    for g, f in GROUPS.items():
        out.append(("property", g, ("cells", chain.cells_of(None, f), "mean")))
    for ix in INDEX_PAIRS:
        out.append(("property", ix, ("index", list(range(12)), ix)))
    for (a, b), w in zip(WINDOWS, WNAMES):
        for g, f in GROUPS.items():
            out.append(("cell", f"{w} {g}", ("cells", chain.cells_of(range(a, b), f), "mean")))
        for ix in INDEX_PAIRS:
            out.append(("cell", f"{w} {ix}", ("index", list(range(a, b)), ix)))
    for (a, b), w in zip(WINDOWS, WNAMES):
        out.append(("window_drop", w, ("cells", chain.cells_of(range(a, b)), "drop")))
    return out


def index_swap(X: np.ndarray, months: list[int], ix: str, mean: np.ndarray, check: bool = False) -> np.ndarray:
    """``X`` ``(n, 12, 12)`` with the index ``ix`` of ``months`` set to the mean field's, the pair's sum kept."""
    iN, iR = INDEX_PAIRS[ix]
    X = X.copy()
    m = np.asarray(months)
    v = (mean[m, iN] - mean[m, iR]) / (mean[m, iN] + mean[m, iR])               # (k,)
    S = X[:, m, iN] + X[:, m, iR]                                                 # (n, k)
    X[:, m, iN] = S * (1 + v) / 2
    X[:, m, iR] = S * (1 - v) / 2
    if check:
        got = (X[:, m, iN] - X[:, m, iR]) / (X[:, m, iN] + X[:, m, iR])
        assert np.allclose(got, np.broadcast_to(v, got.shape), atol=1e-5), "swapped index differs from the target"
        assert np.allclose(X[:, m, iN] + X[:, m, iR], S, rtol=1e-6), "pair sum changed"
        assert (X[:, m, iN] >= 0).all() and (X[:, m, iR] >= 0).all(), "negative reflectance"
    return X


def apply_block(X: np.ndarray, spec: tuple, mean: np.ndarray, check: bool = False) -> np.ndarray:
    if spec[0] == "cells":
        return chain.apply(X, spec[1], spec[2], mean)
    return index_swap(X, spec[1], spec[2], mean, check)


def occlusion(ev: chain.Evaluator, mean: np.ndarray, check: bool = False) -> dict:
    B = blocks()
    X = ev.field_input()
    scores = np.stack([ev.field_scores(apply_block(X, spec, mean, check)) for _, _, spec in B])
    return {"occl": scores, "occl_kind": np.array([k for k, _, _ in B]), "occl_name": np.array([n for _, n, _ in B])}


def integrated_gradients(ev: chain.Evaluator, mean: np.ndarray, targets: np.ndarray, steps: int,
                         batch: int | None = None) -> dict:
    """IG of the scores ``targets`` ``(N, T)`` from the mean field, as ``explain_base.integrated_gradients``, with
    each index pair's attribution split into its index part (per field, month, index)."""
    import torch

    m, dev = ev.m, ev.m.device
    x = ev.field_input()
    plan, empty = eb.plan_seq(x)
    xb0 = np.broadcast_to(mean, x.shape).copy()
    x0 = np.nan_to_num(x, nan=0.0)
    delta = x0 - xb0
    nt = targets.shape[1]
    G = np.zeros((nt,) + x.shape, np.float32)
    IDX = np.zeros((nt, len(x), 12, len(INDEX_PAIRS)), np.float64)
    base_maps = ev._maps(ev.base)
    tg = torch.as_tensor(targets, device=dev)

    def scores_and_grads(Xp: np.ndarray, want_grad: bool):
        Xt = torch.from_numpy(Xp.astype(np.float32)).to(dev)
        with torch.no_grad():
            emb = eb.encode_t(m, Xt, plan, empty)
        emb = emb.detach().requires_grad_(want_grad)
        g_emb = torch.zeros((nt,) + emb.shape, device=dev)
        sc_all = []
        nb = batch or ev.net_batch                    # fields per pass; the backward needs more memory
        for i0 in range(0, len(ev.fields), nb):
            idx = list(range(i0, min(i0 + nb, len(ev.fields))))
            maps = base_maps.repeat(len(idx), 1, 1, 1).reshape(len(idx), 128, -1)
            with torch.set_grad_enabled(want_grad):
                for k, i in enumerate(idx):
                    pix = torch.as_tensor(ev.fields[i]["pixels"], device=dev)
                    maps[k][:, pix] = emb[ev.span[i]:ev.span[i + 1]].T
                lp = chain.crop_logp(m, maps.reshape(len(idx), 128, ev.h, ev.w))
                lp = lp.reshape(len(idx), lp.shape[1], -1)
                sc = torch.stack([chain.field_scores(m, lp[k], ev.fields[i]["interior"]) for k, i in enumerate(idx)])
                sc_all.append(sc.detach())
                if want_grad:
                    for t in range(nt):
                        obj = sc[torch.arange(len(idx), device=dev), tg[idx, t]].sum()
                        g_emb[t] += torch.autograd.grad(obj, emb, retain_graph=t < nt - 1)[0]
        sc_all = torch.cat(sc_all).cpu().numpy()
        if not want_grad:
            return sc_all, None
        gx = np.zeros((nt,) + Xp.shape, np.float32)
        for idx_b, b2, b1, src2, src1 in plan:
            for a in range(0, len(idx_b), 4096):
                sel = idx_b[a:a + 4096]
                Xc = torch.from_numpy(Xp[sel].astype(np.float32)).to(dev).requires_grad_(True)
                planc = [(np.arange(len(sel)), b2, b1, None if src2 is None else src2[a:a + 4096],
                          None if src1 is None else src1[a:a + 4096])]
                e = eb.encode_t(m, Xc, planc, empty[sel])
                for t in range(nt):
                    g = torch.autograd.grad(e, Xc, grad_outputs=g_emb[t][torch.as_tensor(sel, device=dev)],
                                            retain_graph=t < nt - 1)[0]
                    gx[t, sel] = g.cpu().numpy()
        return sc_all, gx

    f0, _ = scores_and_grads(xb0, False)
    f1, _ = scores_and_grads(x0, False)
    for k in range(steps):
        a = (k + 0.5) / steps
        xk = xb0 + a * delta
        _, gx = scores_and_grads(xk, True)
        G += gx / steps
        for j, (iN, iR) in enumerate(INDEX_PAIRS.values()):
            Nk, Rk = xk[..., iN], xk[..., iR]
            with np.errstate(divide="ignore", invalid="ignore"):
                part = (gx[..., iN] - gx[..., iR]) * (Rk * delta[..., iN] - Nk * delta[..., iR]) / (Nk + Rk)
            IDX[..., j] += np.nan_to_num(part) / steps
    att = delta[None] * G
    att[:, ~np.isfinite(x)] = 0.0
    n = len(ev.fields)
    per = np.zeros((n, nt, 12, 12), np.float32)
    per_ix = np.zeros((n, nt, 12, len(INDEX_PAIRS)), np.float32)
    for i in range(n):
        a, b = ev.span[i], ev.span[i + 1]
        per[i] = att[:, a:b].sum(1)
        per_ix[i] = IDX[:, a:b].sum(1)
    take = lambda f: np.take_along_axis(f, targets, 1)                       # noqa: E731
    return {"ig": per, "ig_index": per_ix, "ig_f0": take(f0), "ig_f1": take(f1)}


def explain_chip(model, data, fields, mean, cfg, rng, check: bool = False) -> dict:
    t = {}
    t0 = time.time()
    ev = chain.Evaluator(model, data, fields, "32", int(cfg["net_batch"]))
    base = ev.field_scores(None, emb=ev.base[ev.rows])
    empty = ev.field_scores(ev.same_for_all(np.ones((12, 12), bool), "mean", mean))
    a_idx = np.array([K_CLS + model.species_idx[f["cls"]] for f in fields])
    sp = base[:, K_CLS:].copy()
    sp[np.arange(len(fields)), a_idx - K_CLS] = -np.inf
    res = {"field": np.array([f["field"] for f in fields]), "cls": np.array([f["cls"] for f in fields]),
           "a_idx": a_idx, "b_idx": K_CLS + sp.argmax(1), "base": base, "empty": empty,
           "n_pixels": np.array([len(f["pixels"]) for f in fields]),
           "n_interior": np.array([len(f["interior"]) for f in fields])}
    t["setup"] = time.time() - t0
    t0 = time.time()
    res.update(occlusion(ev, mean, check))
    t["occlusion"] = time.time() - t0
    t0 = time.time()
    res["shap_months"] = eb.shap_months(ev, mean, base, empty, int(cfg["shap_permutations"]), rng)[0]
    res["shap_groups"] = eb.shap_groups(ev, mean, base, empty)
    t["shap"] = time.time() - t0
    t0 = time.time()
    res.update(integrated_gradients(ev, mean, a_idx[:, None], int(cfg["ig_steps"]), int(cfg.get("ig_batch", 8))))
    t["ig"] = time.time() - t0
    res["seconds"] = np.array([t[k] for k in t])
    res["seconds_names"] = np.array(list(t))
    return res


def checks(res: dict) -> dict:
    """Per chip: SHAP efficiency, IG completeness, the index split against its pair."""
    n = len(res["field"])
    r = np.arange(n)
    a = res["a_idx"]
    tot = res["base"][r, a] - res["empty"][r, a]
    eff_m = np.abs(res["shap_months"][r, :, a].sum(1) - tot).max()
    eff_g = np.abs(res["shap_groups"][r, :, a].sum(1) - tot).max()
    span = res["ig_f1"][:, 0] - res["ig_f0"][:, 0]
    comp = np.median(np.abs(res["ig"][:, 0].sum((1, 2)) - span) / np.maximum(np.abs(span), 1e-6))
    share = []
    for j, (iN, iR) in enumerate(INDEX_PAIRS.values()):
        pair = res["ig"][:, 0, :, iN] + res["ig"][:, 0, :, iR]
        share.append(float(np.median(np.abs(res["ig_index"][:, 0, :, j]).sum(1) / np.maximum(np.abs(pair).sum(1), 1e-9))))
    return {"shap_months_eff": float(eff_m), "shap_groups_eff": float(eff_g), "ig_completeness_median": float(comp),
            "index_part_share_of_pair": share}


def run(limit: int | None, pilot: bool) -> None:
    import torch

    cfg = yaml.safe_load((HERE / "experiment.yaml").read_text())
    chain.ENCODE_BATCH = 16384 if torch.cuda.is_available() and torch.cuda.get_device_properties(0).total_memory < 40e9 else 65536
    sample = pd.read_parquet(HERE / "sample" / "fields.parquet")
    order = json.loads((HERE / "sample" / "chips.json").read_text())[: limit or None]
    model = chain.load_model()
    mean = chain.load_mean_field()
    out = chain.results_dir() / ("pilot" if pilot else "explain")
    out.mkdir(parents=True, exist_ok=True)
    root = chain.REPO / "data/eurocrops_chips/EE_2021_rev"
    print(f"{len(order)} chips on {model.device}, encode batch {chain.ENCODE_BATCH}", flush=True)
    for j, chip in enumerate(order):
        f = out / f"{chip}.npz"
        if f.exists():
            continue
        t0 = time.time()
        data = chain.read_chip(root / "chips", chip)
        want = set(sample.loc[sample.chip == chip, "field"])
        fields = [x for x in chain.chip_fields(data["mask"], data["parcels"]) if x["field"] in want]
        meth = dict(cfg["methods"])
        for attempt in range(4):                       # the hub GPU is shared: on running out of memory, wait and
            rng = np.random.default_rng(int(chip.split("_")[1]) * 100000 + int(chip.split("_")[2]))
            try:                                       # retry with half the fields per pass (memory only)
                res = explain_chip(model, data, fields, mean, meth, rng, check=pilot)
                break
            except torch.OutOfMemoryError:
                torch.cuda.empty_cache()
                meth["net_batch"] = max(1, int(meth["net_batch"]) // 2)
                meth["ig_batch"] = max(1, int(meth.get("ig_batch", 8)) // 2)
                print(f"{chip}: out of GPU memory, retry {attempt + 1} with net_batch {meth['net_batch']}, "
                      f"ig_batch {meth['ig_batch']}", flush=True)
                time.sleep(60)
        else:
            print(f"{chip}: skipped after 4 attempts", flush=True)
            continue
        tmp = f.with_name(f"{chip}.part.npz")
        np.savez_compressed(tmp, **res)
        tmp.replace(f)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        c = checks(res)
        secs = ", ".join(f"{k} {v / 60:.1f}" for k, v in zip(res["seconds_names"], res["seconds"]))
        print(f"{chip} ({j + 1}/{len(order)}): {len(fields)} fields, {(time.time() - t0) / 60:.1f} min "
              f"({(time.time() - t0) / max(len(fields), 1):.0f} s per field) [{secs}] checks {json.dumps(c)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--pilot", action="store_true", help="one chip to pilot/, with the block asserts")
    args = ap.parse_args()
    run(1 if args.pilot else args.limit, args.pilot)
