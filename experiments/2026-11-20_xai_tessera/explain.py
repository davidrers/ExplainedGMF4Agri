"""Steps 1 to 4 on one shard of the sampled chips: occlusion, SHAP, Integrated Gradients, deletion curves; and the
randomised-model check (``sanity``).

Every score is the ``(20 classes + 13 species,)`` vector of :func:`chain.field_scores`: the mean over the field's
interior of each class's and each species' log-probability. All in fp32. For each field, ``A`` is its species and
``B`` its rival, the other species the observed input scores highest.

**Occlusion** (field scope unless named): under *drop*, each month (both sensors), each three-month window, each
month of one sensor, each sensor; under the *mean*, each month, window, band group (all months), sensor and cell
(month x band group); on the whole chip, every window under both, every band group under the mean, each sensor under
drop.

**SHAP**, under the mean (absent players take the mean field): a game over the 12 months, estimated by permutation
sampling (``shap_permutations``, in antithetic pairs, the same permutations for every field of a chip; each
permutation's contributions add up to v(all) − v(none) exactly), kept as the mean of all permutations and of each
half (for the convergence check); and a game over the 6 band groups, exact (64 coalitions).

**Integrated Gradients** from the mean field x': the field's own pixels move from x' to x in ``ig_steps`` midpoint
steps; the gradient of the score of A and of B passes through the network to the embedding and through TESSERA to
its inputs (:func:`encode_t`, which reproduces :func:`gfm4agri.embeddings.tessera_v2.encode_pixels` in torch with
the frame validity of x held fixed). Attribution = (x − x') x mean gradient, summed over the field's pixels per month
and feature; the summed gradient itself gives the direction. Completeness: f(x) − f(x') against the attributions'
sum, both from the same torch path.

**Deletion curves**: the months of each field removed one after another in the order of each method (occlusion
under the mean, occlusion under drop, SHAP, IG; A's score) and in random orders, under each replacement, up to
``deletion_steps`` months.

**Sanity** (on the first ``sanity_chips`` chips): the crop head re-initialised, and the crop head with the U-Net's
decoder; windows under both replacements and IG for A.

Writes ``explain/<chip>.npz`` (``sanity/<chip>.npz``); a chip already written is skipped.
"""

from __future__ import annotations

import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

import chain
from chain import BAND_GROUPS, MONTHS, WINDOWS, cells_of, window_name

K_CLS = 20


def _say(msg: str) -> None:
    import datetime as dt

    print(f"[explain {dt.datetime.now():%m-%d %H:%M:%S}] {msg}", flush=True)


# ------------------------------------------------------------------------------------------------------ units

def units() -> dict[str, list[tuple[str, str, np.ndarray]]]:
    s2, s1 = list(range(10)), [10, 11]
    month = [("month", MONTHS[m], cells_of([m])) for m in range(12)]
    window = [("window", window_name(w), cells_of(range(*w))) for w in WINDOWS]
    sensor = [("sensor", "Sentinel-2", cells_of(None, s2)), ("sensor", "Sentinel-1", cells_of(None, s1))]
    groups = [("band group", g, cells_of(None, f)) for g, f in BAND_GROUPS.items()]
    msens = [("month x sensor", f"{MONTHS[m]} {s}", cells_of([m], f)) for m in range(12)
             for s, f in (("Sentinel-2", s2), ("Sentinel-1", s1))]
    cell = [("cell", f"{MONTHS[m]} {g}", cells_of([m], f)) for m in range(12) for g, f in BAND_GROUPS.items()]
    return {"drop": month + window + msens + sensor,
            "mean": month + window + groups + sensor + cell,
            "chip": [(k, n, c, "mean") for k, n, c in window] + [(k, n, c, "drop") for k, n, c in window]
                    + [(k, n, c, "mean") for k, n, c in groups] + [(k, n, c, "drop") for k, n, c in sensor]}


# ------------------------------------------------------------------------------------------------------ TESSERA in torch

def plan_seq(X: np.ndarray):
    """The bucketing of :func:`gfm4agri.embeddings.tessera_v2._encode_batch` for ``X`` ``(U, 12, 12)``: per bucket
    the pixel indices and the frame source indices, from the frames ``X`` observes."""
    from gfm4agri.embeddings import tessera_v2 as tv
    from gfm4agri.embeddings import tessera_v2_model as tm

    s2_valid = np.any(X[..., :10] != 0, -1)
    s1_valid = np.any(tv.s1_to_tessera(X[..., 10:]) != 0, -1)
    keys = tm._vec_get_bin_size(s2_valid.sum(1)).astype(np.int64) * 1000 + tm._vec_get_bin_size(s1_valid.sum(1))
    plan = []
    for key in np.unique(keys):
        idx = np.where(keys == key)[0]
        b2, b1 = max(int(key // 1000), 1), max(int(key % 1000), 1)
        src2 = tm._build_source_indices(s2_valid[idx], b2) if key // 1000 else None
        src1 = tm._build_source_indices(s1_valid[idx], b1) if key % 1000 else None
        plan.append((idx, b2, b1, src2, src1))
    empty = ~s2_valid.any(1) & ~s1_valid.any(1)
    return plan, empty


def encode_t(model: chain.Model, Xt, plan, empty, chunk: int = 4096):
    """Standardised embeddings ``(U, 128)`` of ``Xt`` ``(U, 12, 12)`` (torch, may require grad), as
    :func:`chain.encode` computes them in fp32, with the frames of ``plan``."""
    import torch

    from gfm4agri.embeddings import tessera_v2_model as tm

    dev = Xt.device
    doy = torch.as_tensor(model.doy, device=dev)
    m2, s2 = (torch.as_tensor(np.asarray(v, np.float32), device=dev) for v in (tm.S2_BAND_MEAN, tm.S2_BAND_STD))
    m1, s1 = (torch.as_tensor(np.asarray(v, np.float32), device=dev) for v in (tm.S1A_BAND_MEAN, tm.S1A_BAND_STD))
    mean, std = (torch.as_tensor(v, device=dev) for v in (model.means, model.stds))
    out = torch.zeros(len(Xt), 128, device=dev, dtype=torch.float32)
    parts = []
    for idx, b2, b1, src2, src1 in plan:
        for a in range(0, len(idx), chunk):
            sel = idx[a:a + chunk]
            x = Xt[torch.as_tensor(sel, device=dev)]
            n = len(sel)
            x2 = torch.zeros(n, b2, 11, device=dev)
            x1 = torch.zeros(n, b1, 3, device=dev)
            if src2 is not None:
                s = torch.as_tensor(src2[a:a + chunk], device=dev)
                g = torch.gather(x[..., :10], 1, s[:, :, None].expand(-1, -1, 10))
                x2 = torch.cat([(g - m2) / (s2 + 1e-9), doy[s][..., None]], -1)
            if src1 is not None:
                s = torch.as_tensor(src1[a:a + chunk], device=dev)
                raw = torch.clamp((2.0 * x[..., 10:] + 50.0) * 200.0, 0, 32767)
                g = torch.gather((raw - m1) / (s1 + 1e-9), 1, s[:, :, None].expand(-1, -1, 2))
                x1 = torch.cat([g, doy[s][..., None]], -1)
            parts.append((sel, model.student.encode(x2, x1).float()))
    for sel, e in parts:
        out = out.index_copy(0, torch.as_tensor(sel, device=dev), e)
    out = torch.where(torch.as_tensor(empty, device=dev)[:, None], torch.zeros_like(out), out)
    return (out - mean) / std


# ------------------------------------------------------------------------------------------------------ the methods

def occlusion(ev: chain.Evaluator, mean: np.ndarray) -> dict:
    u = units()
    out = {}
    for op in ("drop", "mean"):
        out[f"occl_{op}"] = np.stack([ev.field_scores(ev.same_for_all(c, op, mean)) for _, _, c in u[op]])
        out[f"occl_{op}_units"] = np.array([f"{k}|{n}" for k, n, _ in u[op]])
    X = ev.chip["X"]
    out["occl_chip"] = np.stack([ev.chip_scores(chain.apply(X, c, op, mean)) for _, _, c, op in u["chip"]])
    out["occl_chip_units"] = np.array([f"{k}|{n}|{op}" for k, n, _, op in u["chip"]])
    return out


def shap_months(ev, mean, base, empty, n_perm: int, rng) -> np.ndarray:
    """``(3, N, 12, K + S)``: the Shapley values of the months, from all permutations and from each half."""
    perms = []
    for _ in range(n_perm // 2):
        p = rng.permutation(12)
        perms += [p, p[::-1]]
    contrib = np.zeros((len(perms),) + base.shape[:1] + (12, base.shape[1]), np.float32)
    for k, p in enumerate(perms):
        present = np.zeros(12, bool)
        prev = empty
        for j, m in enumerate(p):
            present[m] = True
            if j == 11:
                cur = base
            else:
                cells = np.zeros((12, 12), bool)
                cells[~present] = True
                cur = ev.field_scores(ev.same_for_all(cells, "mean", mean))
            contrib[k, :, m] = cur - prev
            prev = cur
    h = len(perms) // 2 // 2 * 2           # whole antithetic pairs in each half
    return np.stack([contrib.mean(0), contrib[:h].mean(0), contrib[h:].mean(0)])


def shap_groups(ev, mean, base, empty) -> np.ndarray:
    """``(N, 6, K + S)``: the exact Shapley values of the six band groups (every month)."""
    names = list(BAND_GROUPS)
    n = len(names)
    v = {}
    for s in range(2 ** n):
        if s == 2 ** n - 1:
            v[s] = base
        elif s == 0:
            v[s] = empty
        else:
            absent = [f for i, g in enumerate(names) if not s >> i & 1 for f in BAND_GROUPS[g]]
            v[s] = ev.field_scores(ev.same_for_all(cells_of(None, absent), "mean", mean))
    phi = np.zeros(base.shape[:1] + (n, base.shape[1]), np.float32)
    for i in range(n):
        for s in range(2 ** n):
            if s >> i & 1:
                continue
            k = bin(s).count("1")
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            phi[:, i] += w * (v[s | 1 << i] - v[s])
    return phi


def integrated_gradients(ev: chain.Evaluator, net, mean, targets: np.ndarray, steps: int) -> dict:
    """IG of the scores ``targets`` ``(N, T)`` (indices into the score vector) from the mean field."""
    import torch

    m, dev = ev.m, ev.m.device
    m_net = copy.copy(m)
    m_net.net = net
    x = ev.field_input()                                                     # (U, 12, 12), the fields' pixels
    plan, empty = plan_seq(x)
    xb = np.broadcast_to(mean, x.shape).copy()
    x0, xb0 = np.nan_to_num(x, nan=0.0), xb
    delta = x0 - xb0
    nt = targets.shape[1]
    G = np.zeros((nt,) + x.shape, np.float32)
    base_maps = ev._maps(ev.base)
    tg = torch.as_tensor(targets, device=dev)

    def scores_and_grads(Xp: np.ndarray, want_grad: bool):
        Xt = torch.from_numpy(Xp.astype(np.float32)).to(dev)
        with torch.no_grad():
            emb = encode_t(m, Xt, plan, empty)
        emb = emb.detach().requires_grad_(want_grad)
        g_emb = torch.zeros((nt,) + emb.shape, device=dev)
        sc_all = []
        for i0 in range(0, len(ev.fields), ev.net_batch):
            idx = list(range(i0, min(i0 + ev.net_batch, len(ev.fields))))
            maps = base_maps.repeat(len(idx), 1, 1, 1).reshape(len(idx), 128, -1)
            with torch.set_grad_enabled(want_grad):
                for k, i in enumerate(idx):
                    pix = torch.as_tensor(ev.fields[i]["pixels"], device=dev)
                    maps[k][:, pix] = emb[ev.span[i]:ev.span[i + 1]].T
                lp = chain.crop_logp(m_net, maps.reshape(len(idx), 128, ev.h, ev.w))
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
                sub = [(sel, b2, b1, None if src2 is None else src2[a:a + 4096],
                        None if src1 is None else src1[a:a + 4096])]
                Xc = torch.from_numpy(Xp[sel].astype(np.float32)).to(dev).requires_grad_(True)
                planc = [(np.arange(len(sel)), b2, b1, sub[0][3], sub[0][4])]
                e = encode_t(m, Xc, planc, empty[sel])
                for t in range(nt):
                    g = torch.autograd.grad(e, Xc, grad_outputs=g_emb[t][torch.as_tensor(sel, device=dev)],
                                            retain_graph=t < nt - 1)[0]
                    gx[t, sel] = g.cpu().numpy()
        return sc_all, gx

    f0, _ = scores_and_grads(xb0, False)
    f1, _ = scores_and_grads(x0, False)
    for k in range(steps):
        a = (k + 0.5) / steps
        _, gx = scores_and_grads(xb0 + a * delta, True)
        G += gx / steps
    att = delta[None] * G                                                     # (T, U, 12, 12)
    att[:, ~np.isfinite(x)] = 0.0
    n = len(ev.fields)
    per = np.zeros((n, nt, 12, 12), np.float32)
    grad = np.zeros((n, nt, 12, 12), np.float32)
    for i in range(n):
        a, b = ev.span[i], ev.span[i + 1]
        per[i] = att[:, a:b].sum(1)
        grad[i] = G[:, a:b].sum(1)
    take = lambda f: np.take_along_axis(f, targets, 1)                       # noqa: E731
    return {"ig": per, "ig_grad": grad, "ig_f0": take(f0), "ig_f1": take(f1)}


def deletion(ev, mean, rankings: dict[str, np.ndarray], k_max: int, n_random: int, rng) -> dict:
    """Curves ``(R, N, k_max, K + S)``: months removed in each ranking's order, under each replacement."""
    n = len(ev.fields)
    orders = dict(rankings)
    for r in range(n_random):
        orders[f"random{r}"] = np.stack([rng.permutation(12) for _ in range(n)])
    curves, labels = [], []
    for name, order in orders.items():
        for op in ("drop", "mean"):
            c = []
            for k in range(1, k_max + 1):
                cells = [cells_of(order[i, :k]) for i in range(n)]
                c.append(ev.field_scores(ev.per_field(cells, op, mean)))
            curves.append(np.stack(c, 1))
            labels.append(f"{name}|{op}")
    return {"del_curves": np.stack(curves), "del_labels": np.array(labels),
            "del_orders": np.stack([orders[k] for k in orders]), "del_order_names": np.array(list(orders))}


def rankings_of(res: dict, a_idx: np.ndarray, base: np.ndarray) -> dict[str, np.ndarray]:
    """Each field's months, most important first, for its species A, by each method."""
    n = len(a_idx)
    r = np.arange(n)
    du = list(res["occl_drop_units"]).index
    mu = list(res["occl_mean_units"]).index
    occ_d = np.stack([base[r, a_idx] - res["occl_drop"][du(f"month|{m}")][r, a_idx] for m in MONTHS], 1)
    occ_m = np.stack([base[r, a_idx] - res["occl_mean"][mu(f"month|{m}")][r, a_idx] for m in MONTHS], 1)
    shap = res["shap_months"][0][r, :, a_idx]
    ig = res["ig"][:, 0].sum(-1)
    return {k: np.argsort(-v, axis=1, kind="stable") for k, v in
            (("occlusion_mean", occ_m), ("occlusion_drop", occ_d), ("shap", shap), ("ig", ig))}


# ------------------------------------------------------------------------------------------------------ the shard

def chip_fields(chips: Path, chip: str, sample: pd.DataFrame):
    data = chain.read_chip(chips / "chips", chip)
    want = set(sample.loc[sample.chip == chip, "field"])
    fields = [f for f in chain.chip_fields(data["mask"], data["parcels"]) if f["field"] in want]
    return data, fields


def explain_chip(model, data, fields, mean, cfg, rng, ig_check: bool) -> dict:
    t = {}
    t0 = time.time()
    ev = chain.Evaluator(model, data, fields, "32", int(cfg["net_batch"]))
    base = ev.field_scores(None, emb=ev.base[ev.rows])
    empty = ev.field_scores(ev.same_for_all(np.ones((12, 12), bool), "mean", mean))
    a_idx = np.array([K_CLS + model.species_idx[f["cls"]] for f in fields])
    sp = base[:, K_CLS:].copy()
    sp[np.arange(len(fields)), a_idx - K_CLS] = -np.inf
    b_idx = K_CLS + sp.argmax(1)
    targets = np.stack([a_idx, b_idx], 1)
    res = {"field": np.array([f["field"] for f in fields]), "cls": np.array([f["cls"] for f in fields]),
           "a_idx": a_idx, "b_idx": b_idx, "base": base, "empty": empty,
           "chip_base": ev.chip_scores(data["X"]),
           "chip_empty": ev.chip_scores(chain.apply(data["X"], np.ones((12, 12), bool), "mean", mean)),
           "n_pixels": np.array([len(f["pixels"]) for f in fields]),
           "n_interior": np.array([len(f["interior"]) for f in fields])}
    t["setup"] = time.time() - t0
    t0 = time.time()
    res.update(occlusion(ev, mean))
    t["occlusion"] = time.time() - t0
    t0 = time.time()
    res["shap_months"] = shap_months(ev, mean, base, empty, int(cfg["shap_permutations"]), rng)
    res["shap_groups"] = shap_groups(ev, mean, base, empty)
    t["shap"] = time.time() - t0
    t0 = time.time()
    res.update(integrated_gradients(ev, model.net, mean, targets, int(cfg["ig_steps"])))
    t["ig"] = time.time() - t0
    if ig_check:
        t0 = time.time()
        chk = integrated_gradients(ev, model.net, mean, targets, int(cfg["ig_steps_check"]))
        res.update({f"{k}_check": v for k, v in chk.items()})
        t["ig_check"] = time.time() - t0
    t0 = time.time()
    res.update(deletion(ev, mean, rankings_of(res, a_idx, base), int(cfg["deletion_steps"]),
                        int(cfg["deletion_random"]), rng))
    t["deletion"] = time.time() - t0
    res["seconds"] = np.array([t[k] for k in t])
    res["seconds_names"] = np.array(list(t))
    return res


def randomised(net, part: str, seed: int = 0):
    import torch

    r = copy.deepcopy(net)
    torch.manual_seed(seed)
    mods = [r.crop_head] + ([r.unet.decoder, r.unet.segmentation_head] if part == "head+decoder" else [])
    for mod in mods:
        for sub in mod.modules():
            if hasattr(sub, "reset_parameters"):
                sub.reset_parameters()
    return r.eval()


def sanity_chip(model, data, fields, mean, cfg) -> dict:
    u = units()
    ev = chain.Evaluator(model, data, fields, "32", int(cfg["net_batch"]))
    res = {"field": np.array([f["field"] for f in fields])}
    a_idx = np.array([K_CLS + model.species_idx[f["cls"]] for f in fields])
    windows = [(n, c) for k, n, c in u["mean"] if k == "window"]
    for part in ("head", "head+decoder"):
        net = randomised(model.net, part)
        m2 = copy.copy(model)
        m2.net = net
        ev2 = chain.Evaluator.__new__(chain.Evaluator)
        ev2.__dict__.update(ev.__dict__)
        ev2.m = m2
        base = ev2.field_scores(None, emb=ev.base[ev.rows])
        res[f"{part}|base"] = base
        for op in ("mean", "drop"):
            res[f"{part}|windows_{op}"] = np.stack([ev2.field_scores(ev2.same_for_all(c, op, mean)) for _, c in windows])
        ig = integrated_gradients(ev2, net, mean, a_idx[:, None], int(cfg["ig_steps"]))
        res[f"{part}|ig"] = ig["ig"][:, 0]
    return res


def run(step: str, chips: Path, root: Path, spec: dict, batch: int, limit: int | None = None) -> None:
    import torch

    cfg = spec["methods"]
    sample = pd.read_parquet(chain.HERE / "sample" / "fields.parquet")
    order = json.loads((chain.HERE / "sample" / "chips.json").read_text())
    have = set((chips / "chips").glob("*.mask.tif"))
    order = [c for c in order if chips / "chips" / f"{c}.mask.tif" in have]
    model = chain.load_model()
    mean = chain.load_mean_field()
    if step == "sanity":
        todo, out = order[: int(cfg["sanity_chips"])], root / "sanity"
    else:
        shard = int(step[len("explain"):])
        todo, out = order[shard::int(spec["shards"])], root / "explain"
    todo = todo[: limit or None]
    out.mkdir(parents=True, exist_ok=True)
    _say(f"{step}: {len(todo)} chips on {model.device}")
    for j, chip in enumerate(todo):
        f = out / f"{chip}.npz"
        if f.exists():
            continue
        t0 = time.time()
        data, fields = chip_fields(chips, chip, sample)
        rng = np.random.default_rng(int(chip.split("_")[1]) * 100000 + int(chip.split("_")[2]))
        if step == "sanity":
            res = sanity_chip(model, data, fields, mean, cfg)
        else:
            res = explain_chip(model, data, fields, mean, cfg, rng, ig_check=(step == "explain0" and j < 2))
        tmp = f.with_name(f"{chip}.part.npz")
        np.savez_compressed(tmp, **res)
        tmp.replace(f)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        msg = ""
        if "seconds" in res:
            msg = ", ".join(f"{k} {v / 60:.1f}" for k, v in zip(res["seconds_names"], res["seconds"]))
        _say(f"{chip} ({j + 1}/{len(todo)}): {len(fields)} fields, {(time.time() - t0) / 60:.1f} min [{msg}]")
