"""How far are composite-fed TESSERA embeddings from the native ones, and does it matter for crops?

Representations, on the labelled pixels of the 57 pilot chips:

    store_v10   our current arm's input: the v1.0 store, native series (``_tessera.tif`` on disk)
    store_v11   the v1.1 store: AWS v1.1 checkpoint on native series
    v11_aws     the same AWS v1.1 checkpoint on our 12 monthly S2 + S1 composites
    v11_mpc     the MPC v1.1 checkpoint on the composites
    v2_*        the four v2 students on the composites

(1) Per-pixel cosine to ``store_v11`` (meaningful for the v1.1 checkpoints only), with reference scales taken
within ``store_v11`` itself. (2) A ridge map from each representation onto ``store_v11``, fitted on the three pool
blocks, R2 on the validation and test blocks. (3) A per-pixel MLP probe (512, 256; ReLU, batch norm, dropout
0.2; the TESSERA arm's head) in leave-one-block-out cross-validation over the five blocks, 3 seeds, 15 epochs:
pooled Macro-F1 over every labelled pixel, and per-class F1 averaged over the seeds.

Writes ``metrics.json`` beside the embeddings and prints a summary.

    poetry run python experiments/2026-10-14_tessera_composites/spike/compare.py
"""

from __future__ import annotations

import csv
import json

import numpy as np
import rasterio
import torch
from sklearn.metrics import f1_score

from common import CHIPSET, OUT

REPS = ["store_v10", "store_v11", "v11_aws", "v11_mpc", "v2_nano", "v2_small", "v2_medium", "v2_large"]
POOL = ["582_450", "586_447", "591_463"]
SEEDS = [0, 1, 2]
EPOCHS = 15


def load():
    rows = list(csv.DictReader(open(next((CHIPSET / "splits").glob("*/chips.csv")))))
    block = {r["chip_id"]: r["block_id"] for r in rows}
    X = {r: [] for r in REPS}
    y, b, parcel = [], [], []
    for cid in sorted(block):
        base = CHIPSET / "chips" / cid
        with rasterio.open(f"{base}.mask.tif") as s:
            m = s.read(1).ravel()
        with rasterio.open(f"{base}.parcels.tif") as s:
            pid = s.read(1).ravel()
        sel = m >= 0
        with rasterio.open(f"{base}_tessera.tif") as s:
            X["store_v10"].append(s.read().reshape(128, -1)[:, sel].T)
        for r in REPS[1:]:
            X[r].append(np.load(OUT / "emb" / r / f"{cid}.npy").astype(np.float32).reshape(128, -1)[:, sel].T)
        y.append(m[sel])
        b.append(np.full(sel.sum(), block[cid]))
        parcel.append(pid[sel])
    X = {r: np.concatenate(v) for r, v in X.items()}
    missing = {}
    for r in REPS:
        bad = ~np.isfinite(X[r]).all(1)
        missing[r] = float(bad.mean())
        X[r][bad] = 0.0          # as the TESSERA arm does
    return X, np.concatenate(y), np.concatenate(b), np.concatenate(parcel), missing


def cos(a, c):
    return (a * c).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(c, axis=1) + 1e-9)


def reference_pairs(y, parcel, rng, n=50000):
    """Pixel pairs within store_v11: same parcel, same class in another parcel, another class."""
    idx = rng.choice(len(y), n, replace=False)
    by_parcel, by_class = {}, {c: np.where(y == c)[0] for c in np.unique(y)}
    for i, p in enumerate(parcel):
        by_parcel.setdefault(p, []).append(i)
    out = {"same parcel": [], "same class, other parcel": [], "other class": []}
    for i in idx:
        pp = by_parcel[parcel[i]]
        j = pp[rng.integers(len(pp))]
        if j != i:
            out["same parcel"].append((i, j))
        cc = by_class[y[i]]
        j = cc[rng.integers(len(cc))]
        if parcel[j] != parcel[i]:
            out["same class, other parcel"].append((i, j))
        j = rng.integers(len(y))
        if y[j] != y[i]:
            out["other class"].append((i, j))
    return {k: np.array(v) for k, v in out.items()}


def mlp(d, c):
    layers, h0 = [], d
    for h in (512, 256):
        layers += [torch.nn.Linear(h0, h), torch.nn.ReLU(), torch.nn.BatchNorm1d(h), torch.nn.Dropout(0.2)]
        h0 = h
    return torch.nn.Sequential(*layers, torch.nn.Linear(h0, c))


def probe(x, y, b, seed, dev, n_classes):
    pred = np.full(len(y), -1)
    for hold in sorted(set(b)):
        te, tr = b == hold, b != hold
        mu, sd = x[tr].mean(0), x[tr].std(0) + 1e-6
        xt = torch.from_numpy((x[tr] - mu) / sd).float().to(dev)
        yt = torch.from_numpy(y[tr]).long().to(dev)
        torch.manual_seed(seed)
        net = mlp(x.shape[1], n_classes).to(dev)
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.05)
        g = torch.Generator().manual_seed(seed)
        for _ in range(EPOCHS):
            net.train()
            perm = torch.randperm(len(yt), generator=g).to(dev)
            for i in range(0, len(perm), 4096):
                k = perm[i:i + 4096]
                loss = torch.nn.functional.cross_entropy(net(xt[k]), yt[k])
                opt.zero_grad()
                loss.backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            pred[te] = net(torch.from_numpy((x[te] - mu) / sd).float().to(dev)).argmax(1).cpu().numpy()
    return pred


def main() -> None:
    rng = np.random.default_rng(0)
    X, y, b, parcel, missing = load()
    names = [c["name"] for c in json.load(open(CHIPSET / "manifest.json"))["classes"]]
    labels = np.unique(y)
    print(f"labelled pixels {len(y):,}, parcels {len(np.unique(parcel)):,}, classes present {len(labels)}")
    res = {"pixels": int(len(y)), "parcels": int(len(np.unique(parcel))), "missing_share": missing,
           "classes": {int(c): names[c] for c in labels},
           "class_pixels": {int(c): int((y == c).sum()) for c in labels}}

    # (1) cosine to store_v11
    ref = X["store_v11"]
    edges = np.linspace(-0.3, 1.0, 66)
    res["cosine"] = {}
    for r in ["v11_aws", "v11_mpc", "store_v10"]:
        c = cos(X[r], ref)
        res["cosine"][f"{r} vs store_v11"] = {"pct": np.percentile(c, [5, 25, 50, 75, 95]).tolist(),
                                             "hist": np.histogram(c, edges)[0].tolist()}
    for k, pairs in reference_pairs(y, parcel, rng).items():
        c = cos(ref[pairs[:, 0]], ref[pairs[:, 1]])
        res["cosine"][f"store_v11, {k}"] = {"pct": np.percentile(c, [5, 25, 50, 75, 95]).tolist(),
                                           "hist": np.histogram(c, edges)[0].tolist()}
    res["cosine_edges"] = edges.tolist()
    print("\n(1) cosine to store_v11, p5 / p25 / median / p75 / p95")
    for k, v in res["cosine"].items():
        print(f"  {k:40s} " + " / ".join(f"{q:.3f}" for q in v["pct"]))

    # (2) ridge map onto store_v11
    tr = np.isin(b, POOL)
    res["ridge_r2"] = {}
    print("\n(2) ridge map onto store_v11: R2 on the validation and test blocks")
    for r in REPS:
        if r == "store_v11":
            continue
        A = np.c_[X[r][tr], np.ones(tr.sum())]
        W = np.linalg.solve(A.T @ A + np.eye(A.shape[1]), A.T @ ref[tr])
        P = np.c_[X[r][~tr], np.ones((~tr).sum())] @ W
        r2 = 1 - ((P - ref[~tr]) ** 2).sum() / ((ref[~tr] - ref[tr].mean(0)) ** 2).sum()
        res["ridge_r2"][r] = float(r2)
        print(f"  {r:10s} {r2:.3f}")

    # (3) probe
    dev = torch.device("cuda")
    res["probe"] = {}
    print("\n(3) MLP probe, leave-one-block-out, pooled Macro-F1")
    for r in REPS:
        f1s, per = [], []
        for seed in SEEDS:
            pred = probe(X[r], y, b, seed, dev, int(y.max()) + 1)
            f1s.append(float(f1_score(y, pred, labels=labels, average="macro")))
            per.append(f1_score(y, pred, labels=labels, average=None))
        res["probe"][r] = {"macro_f1": f1s, "per_class_f1": np.mean(per, 0).tolist()}
        print(f"  {r:10s} {np.mean(f1s):.3f} +- {np.std(f1s):.3f}  ({', '.join(f'{v:.3f}' for v in f1s)})",
              flush=True)

    (OUT / "metrics.json").write_text(json.dumps(res, indent=1))
    print("\nwrote", OUT / "metrics.json")


if __name__ == "__main__":
    main()
