"""The training and evaluation stages of run.py.

Every score is computed on the test chips; every choice that could fit the scores (the best single feature or
dimension per class, its threshold, the feature families) is made on the validation chips first. Pixel samples
carry a weight that restores the population they were drawn from (``samples/*.npz``, ``weight``), so precision,
F1 and class means are population estimates over field interiors, not over the stratified sample.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from run import CFG, CHIPS, EMB_SUFFIX, OUT, REPO, class_names, interior_mask, load_head, load_sample, log, \
    read_band_stack, split_lists
from sae import BatchTopKSAE, calibrate_threshold, reconstruction_metrics, train

EV = CFG["evaluate"]


# ----------------------------------------------------------------------------------------------- normalisation
def norm_stats() -> dict:
    """The z-score of the crop head (training-chip statistics of the store sidecar) and the scalar that brings
    the z-scored embeddings to unit mean norm, as geosae feeds Tessera to its BatchTopK variants."""
    path = OUT / "samples" / "norm.json"
    if path.exists():
        return json.loads(path.read_text())
    side = json.loads((CHIPS / f"{CFG['embedding']}.json").read_text())["normalisation"]
    mean, std = np.array(side["means"], np.float32), np.array(side["stds"], np.float32)
    x = load_sample("train")["emb"].astype(np.float32)
    zs = (x - mean) / std
    scale = float(1.0 / np.linalg.norm(zs, axis=1).mean())
    out = {"means": mean.tolist(), "stds": std.tolist(), "scale": scale,
           "source": f"{CFG['embedding']}.json normalisation (training chips); scale = 1 / mean norm of the "
                     "z-scored training sample"}
    path.write_text(json.dumps(out, indent=1))
    return out


def to_sae(emb: np.ndarray, ns: dict) -> torch.Tensor:
    x = (emb.astype(np.float32) - np.array(ns["means"], np.float32)) / np.array(ns["stds"], np.float32)
    return torch.from_numpy(x * ns["scale"])


def build_model(name: str) -> BatchTopKSAE:
    r, s = CFG["runs"][name], CFG["sae"]
    return BatchTopKSAE(128, s["latent_dim"], r["k"], freeze_decoder=r["variant"] == "frozen", seed=r["seed"],
                        row_cap=s["row_cap"], n_batches_to_dead=s["n_batches_to_dead"])


def load_model(name: str) -> BatchTopKSAE:
    m = build_model(name)
    m.load_state_dict(torch.load(OUT / "runs" / name / "best.pt"))
    return m.eval()


# ------------------------------------------------------------------------------------------------------- train
def stage_train(names: list[str] | None, force: bool) -> None:
    torch.set_num_threads(CFG["sae"]["threads"])
    ns = norm_stats()
    tr = load_sample("train")
    x_train = to_sae(tr["emb"], ns)
    va = load_sample("val")
    x_val = to_sae(va["emb"][va["kind"] == 0], ns)
    log(f"train: {len(x_train)} training pixels, {len(x_val)} validation pixels, scale {ns['scale']:.4f}")
    for name in names or list(CFG["runs"]):
        d = OUT / "runs" / name
        if (d / "summary.json").exists() and not force:
            log(f"train {name}: exists, skipped")
            continue
        r = CFG["runs"][name]
        log(f"train {name}: {r}")
        model = build_model(name)
        res = train(model, x_train, x_val, CFG["sae"], d, seed=r["seed"])
        res.update({"run": name, **r, "sae": CFG["sae"], "threshold": float(model.threshold),
                    "scale": ns["scale"], "train_pixels": len(x_train), "val_pixels": len(x_val)})
        (d / "summary.json").write_text(json.dumps(res, indent=1))
        log(f"train {name}: best validation FVE {res['best_val_fve']:.4f} at step {res['best_step']}, "
            f"{res['seconds']:.0f} s")


# ----------------------------------------------------------------------------------------------- scoring sets
def scoring_set(split: str, rng: np.random.Generator) -> dict:
    """Up to ``eval_pixels_per_class`` field-interior pixels per class and as many outside fields, drawn with
    probability proportional to their weight, each carrying the share of the population it stands for."""
    s = load_sample(split)
    n = CFG["sample"]["eval_pixels_per_class"]
    keep, wts = [], []
    strata = [(s["kind"] == 1) & (s["label"] == c) for c in range(20)] + [s["kind"] == 2]
    for m in strata:
        idx = np.flatnonzero(m)
        if len(idx) == 0:
            continue
        w = s["weight"][idx].astype(np.float64)
        if len(idx) > n:
            take = rng.choice(idx, n, replace=False, p=w / w.sum())
            keep.append(take)
            wts.append(np.full(n, w.sum() / n))
        else:
            keep.append(idx)
            wts.append(w)
    idx = np.concatenate(keep)
    out = {k: v[idx] for k, v in s.items() if k != "chip_ids"}
    out["w"] = np.concatenate(wts)
    out["field"] = out["kind"] == 1
    out["chip_ids"] = s["chip_ids"]
    return out


def onehot(y: np.ndarray, c: int = 20) -> np.ndarray:
    o = np.zeros((len(y), c))
    ok = y >= 0
    o[np.flatnonzero(ok), y[ok]] = 1
    return o


def f1_from(tp, pred_pos, pos):
    prec = np.divide(tp, pred_pos, out=np.zeros_like(tp, dtype=float), where=pred_pos > 0)
    rec = np.divide(tp, pos, out=np.zeros_like(tp, dtype=float), where=pos > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(tp, dtype=float), where=(prec + rec) > 0)
    return prec, rec, f1


# --------------------------------------------------------------------------------------------- representations
class Bases:
    """PCA and the raw dimensions, fitted on the training sample in the SAE's input space."""

    def __init__(self, ns: dict) -> None:
        x = to_sae(load_sample("train")["emb"], ns).double()
        self.mu = x.mean(0)
        cov = ((x - self.mu).T @ (x - self.mu)) / len(x)
        val, vec = torch.linalg.eigh(cov)
        order = torch.argsort(val, descending=True)
        self.var, self.V = val[order].float(), vec[:, order].float()
        self.raw_order = torch.argsort(torch.diag(cov), descending=True)
        self.mu = self.mu.float()

    def pca_recon(self, x: torch.Tensor, n: int) -> torch.Tensor:
        v = self.V[:, :n]
        return self.mu + ((x - self.mu) @ v) @ v.T

    def raw_recon(self, x: torch.Tensor, n: int) -> torch.Tensor:
        r = self.mu.expand_as(x).clone()
        keep = self.raw_order[:n]
        r[:, keep] = x[:, keep]
        return r

    def pca_scores(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mu) @ self.V


def folded_auroc(units: np.ndarray, pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """AUROC of every column of ``units`` as a score for ``pos`` (ties averaged), and whether it is above 0.5."""
    from scipy.stats import rankdata

    r = rankdata(units, axis=0)
    n1, n0 = pos.sum(), (~pos).sum()
    a = (r[pos].sum(0) - n1 * (n1 + 1) / 2) / (n1 * n0)
    return np.maximum(a, 1 - a), a >= 0.5


def balanced_concept_sets(es: dict, supported: list[int], rng) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Per concept: pixel indices and the positive flag, 5,000 against 5,000 drawn by weight. Crop classes are
    scored against other field pixels; 'outside fields' against field pixels."""
    out = []
    field = np.flatnonzero(es["field"])
    names = class_names()

    def wdraw(idx, n):
        w = es["w"][idx]
        return rng.choice(idx, min(n, len(idx)), replace=False, p=w / w.sum())

    for c in supported:
        pos = wdraw(np.flatnonzero(es["field"] & (es["label"] == c)), 5000)
        neg = wdraw(field[es["label"][field] != c], 5000)
        idx = np.concatenate([pos, neg])
        out.append((names[c], idx, np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]))
    pos = wdraw(np.flatnonzero(es["kind"] == 2), 5000)
    neg = wdraw(field, 5000)
    out.append(("outside_fields", np.concatenate([pos, neg]), np.r_[np.ones(len(pos), bool), np.zeros(len(neg), bool)]))
    return out


# ---------------------------------------------------------------------------------------------- single units
def best_feature_per_class(zv, yv, wv, zt, yt, wt, classes):
    """A feature read as on or off: on means the class. Chosen on validation by F1, scored on test."""
    def stats(z, y, w):
        on = (z > 0).astype(np.float64)
        tp = on.T @ (w[:, None] * onehot(y))
        return tp, on.T @ w, (w[:, None] * onehot(y)).sum(0)

    tp, pp, pos = stats(zv, yv, wv)
    _, _, f1v = f1_from(tp, pp[:, None], pos[None, :])
    tpt, ppt, post = stats(zt, yt, wt)
    pt, rt, f1t = f1_from(tpt, ppt[:, None], post[None, :])
    rows = {}
    for c in classes:
        j = int(np.argmax(f1v[:, c]))
        rows[c] = {"unit": j, "val_f1": float(f1v[j, c]), "precision": float(pt[j, c]),
                   "recall": float(rt[j, c]), "f1": float(f1t[j, c])}
    return rows


def best_dense_unit_per_class(uv, yv, wv, ut, yt, wt, classes, n_q: int = 49):
    """A dense unit with a sign and a threshold: above it means the class. All chosen on validation."""
    qs = np.linspace(0.02, 0.98, n_q)
    ohv, oht = wv[:, None] * onehot(yv), wt[:, None] * onehot(yt)
    posv, post = ohv.sum(0), oht.sum(0)
    best = {c: (-1.0, None) for c in classes}
    for j in range(uv.shape[1]):
        for sgn in (1.0, -1.0):
            a = sgn * uv[:, j]
            th = np.quantile(a, qs)
            pred = (a[:, None] > th[None, :]).astype(np.float64)
            _, _, f1 = f1_from(pred.T @ ohv, (pred.T @ wv)[:, None], posv[None, :])
            for c in classes:
                i = int(np.argmax(f1[:, c]))
                if f1[i, c] > best[c][0]:
                    best[c] = (float(f1[i, c]), (j, sgn, float(th[i])))
    rows = {}
    for c in classes:
        f1v, (j, sgn, th) = best[c]
        pred = (sgn * ut[:, j] > th).astype(np.float64)
        p, r, f = f1_from(np.array([pred @ oht[:, c]]), np.array([pred @ wt]), np.array([post[c]]))
        rows[c] = {"unit": j, "sign": sgn, "threshold": th, "val_f1": f1v, "precision": float(p[0]),
                   "recall": float(r[0]), "f1": float(f[0])}
    return rows


# ------------------------------------------------------------------------------------------------- families
def class_means(z, y, w, classes):
    oh = w[:, None] * onehot(y)
    a = (oh.T @ z) / oh.sum(0)[:, None]            # (20, M) class means
    field_mean = (w @ z) / w.sum()                  # (M,)
    active = ((z > 0).astype(np.float64).T @ w) / w.sum()
    return a, field_mean, active


def families(zv, yv, wv, classes):
    a, fm, active = class_means(zv, yv, wv, classes)
    live = active >= 0.001
    cls = np.array(classes)
    top = cls[np.argmax(a[cls], axis=0)]
    fam = {c: [] for c in classes}
    for j in np.flatnonzero(live):
        c = int(top[j])
        if a[c, j] >= EV["family_selectivity"] * fm[j] and fm[j] > 0:
            fam[c].append(int(j))
    for c in classes:  # strongest first: by how much the class mean exceeds the field mean
        fam[c].sort(key=lambda j: -(a[c, j] - fm[j]))
    return fam, np.flatnonzero(live)


def family_classify(z, fam, classes):
    cls = [c for c in classes if fam[c]]
    s = np.stack([z[:, fam[c]].mean(1) for c in cls], 1)
    return np.array(cls)[np.argmax(s, 1)]


def weighted_macro_f1(y, p, w, classes):
    cm = np.zeros((20, 20))
    np.add.at(cm, (y, p), w)
    tp = np.diag(cm)
    prec, rec, f1 = f1_from(tp, cm.sum(0), cm.sum(1))
    return float(np.mean([f1[c] for c in classes])), {int(c): float(f1[c]) for c in classes}, cm


# ------------------------------------------------------------------------------------------------- full test
def full_test_pass(head, ns, models: dict, bases: Bases) -> dict:
    """The crop head on every labelled test pixel: on the embedding, and on each reconstruction of it."""
    import rasterio  # noqa: F401  (read_band_stack)
    from concurrent.futures import ThreadPoolExecutor

    stored = np.load(REPO / CFG["head_fit"] / "predictions_test.npz")
    ids = list(stored["ids"])
    mean, std, scale = (np.array(ns["means"], np.float32), np.array(ns["stds"], np.float32), ns["scale"])
    variants = ["embedding"] + list(models) + ["pca16"]
    cms = {v: {"all": np.zeros((20, 20)), "interior": np.zeros((20, 20))} for v in variants}
    agree = [0, 0]

    def read(cid):
        e = read_band_stack(CHIPS / "chips" / f"{cid}{EMB_SUFFIX}").reshape(128, -1).T
        mask = read_band_stack(CHIPS / "chips" / f"{cid}.mask.tif")[0].astype(np.int16)
        parcels = read_band_stack(CHIPS / "chips" / f"{cid}.parcels.tif")[0]
        return cid, e, mask, interior_mask(parcels, mask)

    t0 = time.time()
    with ThreadPoolExecutor(16) as ex:
        for i, (cid, e, mask, inner) in enumerate(ex.map(read, ids), 1):
            valid = np.isfinite(e).all(1)
            xh = (np.nan_to_num(e, nan=0.0) - mean) / std
            lab = mask.reshape(-1)
            sel = lab >= 0
            inn = inner.reshape(-1)
            with torch.no_grad():
                xt = torch.from_numpy(xh)
                preds = {"embedding": head(xt).argmax(1).numpy()}
                vt = torch.from_numpy(valid)
                xs = xt[vt] * scale
                recons = {name: (lambda m=m: m.decode(m.codes(xs))) for name, m in models.items()}
                recons["pca16"] = lambda: bases.pca_recon(xs, 16)
                for name, fn in recons.items():
                    xr = xt.clone()
                    if valid.any():  # a chip without an embedding keeps its zero-filled input
                        xr[vt] = fn() / scale
                    preds[name] = head(xr).argmax(1).numpy()
            sp = stored["pred"][ids.index(cid)].reshape(-1)
            agree[0] += int((preds["embedding"][sel] == sp[sel]).sum())
            agree[1] += int(sel.sum())
            for v, p in preds.items():
                np.add.at(cms[v]["all"], (lab[sel], p[sel]), 1)
                np.add.at(cms[v]["interior"], (lab[inn], p[inn]), 1)
            if i % 250 == 0:
                log(f"  full test pass: {i}/{len(ids)} chips, {time.time() - t0:.0f} s")
    out = {"chips": len(ids), "stored_prediction_agreement": agree[0] / agree[1], "variants": {}}
    for v in variants:
        out["variants"][v] = {}
        for scope in ("all", "interior"):
            cm = cms[v][scope]
            prec, rec, f1 = f1_from(np.diag(cm), cm.sum(0), cm.sum(1))
            present = cm.sum(1) > 0
            out["variants"][v][scope] = {
                "macro_f1_present": float(f1[present].mean()), "macro_f1_all20": float(f1.mean()),
                "accuracy": float(np.diag(cm).sum() / cm.sum()), "per_class_f1": f1.tolist(),
                "confusion": cm.tolist()}
    return out


# --------------------------------------------------------------------------------------------------- evaluate
def stage_evaluate(force: bool) -> None:
    torch.set_num_threads(EV["threads"])
    d = OUT / "eval"
    d.mkdir(parents=True, exist_ok=True)
    if (d / "summary.json").exists() and not force:
        log("evaluate: exists, skipped")
        return
    rng = np.random.default_rng(0)
    names = class_names()
    ns = norm_stats()
    head, _, _, head_meta = load_head()
    main, control, seeds = CFG["main"], CFG["control"], CFG["seeds"]
    models = {n: load_model(n) for n in CFG["runs"]}
    bases = Bases(ns)
    summary: dict = {"classes": names, "scale": ns["scale"], "head": head_meta}
    arrays: dict = {}

    # 1. Reconstruction on the uniform test sample, against PCA and the raw dimensions at the same L0.
    te = load_sample("test")
    u = te["kind"] == 0
    xu, wu = to_sae(te["emb"][u], ns), torch.from_numpy(te["weight"][u].astype(np.float64))
    rec = []
    for n, m in models.items():
        z = m.codes(xu)
        rec.append({"method": CFG["runs"][n]["variant"], "run": n, "k": CFG["runs"][n]["k"],
                    **reconstruction_metrics(m.decode(z), xu, z, wu)})
    for L in (8, 16, 32):
        rec.append({"method": "pca", "k": L, "l0": L, **reconstruction_metrics(bases.pca_recon(xu, L), xu, None, wu)})
        rec.append({"method": "raw", "k": L, "l0": L, **reconstruction_metrics(bases.raw_recon(xu, L), xu, None, wu)})
    summary["reconstruction"] = rec
    log("evaluate: reconstruction done")

    # Scoring sets: validation for choices, test for scores.
    ev, et = scoring_set("val", rng), scoring_set("test", rng)
    xv, xt = to_sae(ev["emb"], ns), to_sae(et["emb"], ns)
    counts = {c: int(((et["label"] == c) & et["field"]).sum()) for c in range(20)}
    vcounts = {c: int(((ev["label"] == c) & ev["field"]).sum()) for c in range(20)}
    supported = [c for c in range(20) if counts[c] >= EV["min_class_pixels"] and vcounts[c] >= EV["min_class_pixels"]]
    summary["scoring_sets"] = {"test_pixels_per_class": {names[c]: counts[c] for c in range(20)},
                               "val_pixels_per_class": {names[c]: vcounts[c] for c in range(20)},
                               "supported": [names[c] for c in supported],
                               "test_outside_fields": int((et["kind"] == 2).sum())}
    codes_v = {n: models[n].codes(xv).numpy() for n in [main, control, *seeds]}
    codes_t = {n: models[n].codes(xt).numpy() for n in [main, control, *seeds]}
    reps_t = {"sae": codes_t[main], "frozen": codes_t[control], "raw": xt.numpy(),
              "pca": bases.pca_scores(xt).numpy()}

    # 2. Concept alignment: folded AUROC of every unit for every concept, on balanced test sets.
    concepts = balanced_concept_sets(et, supported, rng)
    summary["concepts"] = [c[0] for c in concepts]
    for rep, units in reps_t.items():
        fa = np.zeros((units.shape[1], len(concepts)))
        up = np.zeros_like(fa, dtype=bool)
        for i, (_, idx, pos) in enumerate(concepts):
            fa[:, i], up[:, i] = folded_auroc(units[idx], pos)
        arrays[f"auroc_{rep}"], arrays[f"auroc_up_{rep}"] = fa, up
        live = fa.max(1) > EV["live_auroc"]
        n_al = (fa >= EV["align_auroc"]).sum(1)
        summary.setdefault("alignment", {})[rep] = {
            "units": int(units.shape[1]), "live": int(live.sum()),
            "share_live_at_most_one": float((n_al[live] <= 1).mean()),
            "share_live_exactly_one": float((n_al[live] == 1).mean()),
            "share_live_none": float((n_al[live] == 0).mean()),
            "median_aligned_live": float(np.median(n_al[live])),
            "aligned_units": int((n_al >= 1).sum()),
            "concepts_with_an_aligned_unit": int(((fa >= EV["align_auroc"]).any(0)).sum())}
    log("evaluate: alignment done")

    # 3. One unit per class, on field interiors.
    fv, ft = ev["field"], et["field"]
    yv, yt, wv, wt = ev["label"][fv], et["label"][ft], ev["w"][fv], et["w"][ft]
    single = {}
    for rep, n in (("sae", main), ("frozen", control), (seeds[0], seeds[0]), (seeds[1], seeds[1])):
        single[rep] = best_feature_per_class(codes_v[n][fv], yv, wv, codes_t[n][ft], yt, wt, supported)
    raw_v, raw_t = xv.numpy()[fv], xt.numpy()[ft]
    single["raw"] = best_dense_unit_per_class(raw_v, yv, wv, raw_t, yt, wt, supported)
    single["pca"] = best_dense_unit_per_class(bases.pca_scores(xv).numpy()[fv], yv, wv,
                                              bases.pca_scores(xt).numpy()[ft], yt, wt, supported)
    summary["single_unit"] = {rep: {names[c]: r for c, r in rows.items()} for rep, rows in single.items()}
    log("evaluate: single units done")

    # 4. Families, and the crop map from families alone, against random families of the same sizes.
    fam_out = {}
    for rep, n in (("sae", main), ("frozen", control)):
        fam, live = families(codes_v[n][fv], yv, wv, supported)
        zt_f = codes_t[n][ft]
        p = family_classify(zt_f, fam, supported)
        mf1, per, _ = weighted_macro_f1(yt, p, wt, supported)
        null = []
        sizes = {c: len(fam[c]) for c in supported}
        for _ in range(EV["n_null"]):
            rf = {c: (rng.choice(live, sizes[c], replace=False).tolist() if sizes[c] else []) for c in supported}
            null.append(weighted_macro_f1(yt, family_classify(zt_f, rf, supported), wt, supported)[0])
        null = np.array(null)
        fam_out[rep] = {"families": {names[c]: fam[c] for c in supported}, "sizes": {names[c]: sizes[c] for c in supported},
                        "live_features": int(len(live)), "macro_f1": mf1,
                        "per_class_f1": {names[c]: v for c, v in per.items()},
                        "null_median": float(np.median(null)), "null_q975": float(np.quantile(null, 0.975)),
                        "null_p": float((null >= mf1).mean())}
        arrays[f"family_null_{rep}"] = null
    with torch.no_grad():
        ph = head(xt[torch.from_numpy(ft)] / ns["scale"]).argmax(1).numpy()
    hm, hper, _ = weighted_macro_f1(yt, ph, wt, supported)
    fam_out["head"] = {"macro_f1": hm, "per_class_f1": {names[c]: v for c, v in hper.items()}}
    summary["families"] = fam_out
    log("evaluate: families done")

    # 5a. The crop head on the reconstructions, every labelled test pixel.
    summary["full_test"] = full_test_pass(head, ns, {main: models[main], control: models[control]}, bases)
    log("evaluate: full test pass done")

    # 5b. Ablation: remove a class's family from the embedding (the rest, SAE error included, untouched).
    m = models[main]
    atoms = torch.from_numpy(m.atoms()).float()                       # (M, D)
    xf = xt[torch.from_numpy(ft)]
    zf = torch.from_numpy(codes_t[main][ft])
    fam = {names.index(k): v for k, v in fam_out["sae"]["families"].items()}
    live = np.flatnonzero(((codes_v[main][fv] > 0).astype(np.float64).T @ wv) / wv.sum() >= 0.001)

    def recalls(x):
        with torch.no_grad():
            p = head(x / ns["scale"]).argmax(1).numpy()
        r = {}
        for c in supported:
            sel = yt == c
            r[c] = float((wt[sel] * (p[sel] == c)).sum() / wt[sel].sum())
        return r, p

    base_r, base_p = recalls(xf)
    abl = {}
    for c in supported:
        if not fam[c]:
            continue
        f = fam[c]
        xa = xf - zf[:, f] @ atoms[f]
        r, p = recalls(xa)
        own = base_r[c] - r[c]
        others = float(np.mean([base_r[o] - r[o] for o in supported if o != c]))
        sel = yt == c
        dest = np.bincount(p[sel], weights=wt[sel], minlength=20) / wt[sel].sum()
        x1 = xf - zf[:, f[:1]] @ atoms[f[:1]]
        own1 = base_r[c] - recalls(x1)[0][c]
        pool = np.setdiff1d(live, f)
        null = []
        for _ in range(EV["n_null_ablation"]):
            g = rng.choice(pool, len(f), replace=False)
            null.append(base_r[c] - recalls(xf - zf[:, g] @ atoms[g])[0][c])
        null = np.array(null)
        abl[names[c]] = {"family_size": len(f), "base_recall": base_r[c], "own_drop": own,
                         "own_drop_strongest_only": own1, "others_mean_drop": others,
                         "null_mean": float(null.mean()), "null_q975": float(np.quantile(null, 0.975)),
                         "null_p": float((null >= own).mean()),
                         "goes_to": {names[i]: float(dest[i]) for i in np.argsort(-dest)[:3]}}
        log(f"  ablation {names[c]}: own drop {own:.3f}, null {null.mean():.3f}, others {others:.3f}")
    summary["ablation"] = abl
    log("evaluate: ablation done")

    # 6. What the features respond to: class composition, class means and phenology of their active pixels.
    zt_main = codes_t[main][ft]
    on = (zt_main > 0).astype(np.float64)
    a_t, fm_t, act_t = class_means(zt_main, yt, wt, supported)
    arrays["class_mean_test"], arrays["field_mean_test"], arrays["active_share_test"] = a_t, fm_t, act_t
    arrays["composition_test"] = (on.T @ (wt[:, None] * onehot(yt))) / np.maximum(on.T @ wt, 1e-12)[:, None]
    for key in ("ndvi", "ndmi", "vv", "vh"):
        prof = et[key][ft].astype(np.float64)
        okm = np.isfinite(prof)
        pz = np.where(okm, prof, 0.0)
        arrays[f"profile_feature_{key}"] = (on.T @ (wt[:, None] * pz)) / np.maximum(on.T @ (wt[:, None] * okm), 1e-12)
        oh = wt[:, None] * onehot(yt)
        arrays[f"profile_class_{key}"] = (oh.T @ pz) / np.maximum(oh.T @ okm, 1e-12)
    log("evaluate: profiles done")

    # 7. Stability across dictionary seeds: for each feature of the main run, its closest atom in the others.
    a0 = m.atoms()
    stab = {}
    for n in seeds:
        cos = a0 @ models[n].atoms().T
        arrays[f"match_cos_{n}"] = cos.max(1)
        stab[n] = {"share_ge_0.9": float((cos.max(1) >= 0.9).mean()), "share_ge_0.7": float((cos.max(1) >= 0.7).mean())}
    best = summary["single_unit"]["sae"]
    stab["class_best_features"] = {k: {n: float((a0[v["unit"]] @ models[n].atoms().T).max()) for n in seeds}
                                   for k, v in best.items()}
    summary["stability"] = stab
    for n in (main, control):
        arrays[f"atoms_{n}"] = models[n].atoms()
    np.savez(d / "arrays.npz", **arrays)
    np.savez(d / "scoring_test.npz", **{k: v for k, v in et.items() if k != "chip_ids"}, chip_ids=et["chip_ids"])
    (d / "summary.json").write_text(json.dumps(summary, indent=1, default=float))
    log(f"evaluate: written to {d}")
