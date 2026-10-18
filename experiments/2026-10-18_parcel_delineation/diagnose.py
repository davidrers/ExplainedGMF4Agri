"""Where the boundary head's parcels gain and lose against the extent alone, per fit.

The class-agnostic PQ of the boundary parcels sits close to that of the extent's connected regions for most arms, so
this splits both sides of the score by where a parcel and a boundary pixel lie:

* **True parcels** (the 4-connected pieces ``panoptic.py`` scores): ``isolated`` when no other piece touches it inside
  the extent, which the extent alone already finds; ``touching other class`` when every piece it touches is of
  another class, which the crop-type map can split; ``touching same class`` when at least one is of its own class,
  which only a predicted boundary can split. For each, the share found (class-agnostic match at IoU > 0.5) by the
  extent, the crop-type map and the boundary head at thresholds 0.3 and 0.5.
* **Boundary pixels of the target**, by what lies across them: ``outer`` (non-parcel land or a parcel outside the
  scheme, which the extent gives for free), ``between classes`` and ``same class`` (another parcel of the same
  class, the boundaries the head exists for); and the parcel ``interior``. For each, the share the head marks as
  boundary at 0.3 and 0.5.
* **Predicted parcels** left unmatched, and how many of them are slivers of fewer than ``SLIVER`` pixels.

Writes ``results/<name>/<set>/diagnostics.json``, which ``report.py`` draws when present:

    poetry run python experiments/2026-10-18_parcel_delineation/diagnose.py --chips EE_2021_rev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402

import panoptic  # noqa: E402

PIECES = ("isolated", "touching other class", "touching same class")
PIXELS = ("interior", "outer", "between classes", "same class")
TAUS = (0.3, 0.5)
SLIVER = 20


def _segs(extent, pred, p):
    segs = {"extent": panoptic.pieces(extent.astype(np.int32)),
            "semantic": panoptic.pieces(np.where(extent, pred.astype(np.int32) + 1, 0))}
    for t in TAUS:
        segs[f"boundary@{t:g}"] = panoptic.from_boundary(extent, p, t)
    return segs


def _neighbours(a: np.ndarray) -> list[np.ndarray]:
    """The four 4-neighbours of every pixel; across the chip edge a pixel is its own neighbour."""
    q = np.pad(a, 1, mode="edge")
    return [q[:-2, 1:-1], q[2:, 1:-1], q[1:-1, :-2], q[1:-1, 2:]]


def chip(pred, mask, parcels, prob) -> dict:
    extent = (mask >= 0) & (parcels > 0)
    true = panoptic.pieces(np.where(extent, parcels, 0))
    out = {"pieces": {}, "pixels": {}, "unmatched": {}}
    if not true.any():
        return out
    n = int(true.max()) + 1
    cls = np.full(n, -1, dtype=np.int64)
    cls[true[extent]] = mask[extent]
    p = prob.astype(np.float32) / 255.0

    # Contacts between true pieces, and the class across every boundary pixel.
    contact_same = np.zeros(n, bool)
    contact_any = np.zeros(n, bool)
    px_same = np.zeros(true.shape, bool)
    px_diff = np.zeros(true.shape, bool)
    px_outer = np.zeros(true.shape, bool)
    for nb_true, nb_ext, nb_par in zip(_neighbours(true), _neighbours(extent), _neighbours(parcels)):
        other = extent & nb_ext & (nb_true != true)
        same = other & (cls[nb_true] == cls[true])
        px_same |= same
        px_diff |= other & ~same
        px_outer |= extent & ~nb_ext & (nb_par != parcels)
        contact_any[true[other]] = True
        contact_same[true[same]] = True
    kind = np.where(contact_same, 2, np.where(contact_any, 1, 0))
    ids = np.unique(true[extent])
    target = extent & panoptic.boundary_target(parcels)
    cat = {"interior": extent & ~target, "same class": px_same,
           "between classes": px_diff & ~px_same, "outer": px_outer & ~px_diff & ~px_same}
    for name, sel in cat.items():
        out["pixels"][name] = [int(sel.sum())] + [int((sel & (p >= t)).sum()) for t in TAUS]

    for vname, seg in _segs(extent, pred, p).items():
        ti, si, _ = panoptic.match(true, seg)
        found = np.zeros(n, bool)
        found[ti] = True
        out["pieces"][vname] = [[int((kind[ids] == k).sum()), int((found[ids] & (kind[ids] == k)).sum())]
                                for k in range(3)]
        s_ids, area = np.unique(seg[extent], return_counts=True)
        um = ~np.isin(s_ids, si)
        out["unmatched"][vname] = [int(len(s_ids)), int(um.sum()), int((um & (area < SLIVER)).sum())]
    return out


def _chunk(args) -> dict:
    preds, masks, probs, ids, chips_dir = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    acc: dict = {}
    for i, cid in enumerate(ids):
        c = chip(preds[i], masks[i], read_parcels(chips_dir, str(cid)), probs[i])
        for group, d in c.items():
            for k, v in d.items():
                a = np.asarray(v, dtype=np.int64)
                g = acc.setdefault(group, {})
                g[k] = g[k] + a if k in g else a
    return acc


def cell(predictions: Path, chips_dir: Path, workers: int) -> dict:
    d = np.load(predictions)
    ids = d["ids"]
    chunks = [c for c in np.array_split(np.arange(len(ids)), max(1, min(workers * 4, len(ids)))) if len(c)]
    jobs = [(d["pred"][c], d["mask"][c], d["boundary"][c], ids[c], chips_dir) for c in chunks]
    total: dict = {}
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for acc in ex.map(_chunk, jobs):
            for group, g in acc.items():
                t = total.setdefault(group, {})
                for k, a in g.items():
                    t[k] = t[k] + a if k in t else a
    pieces = {v: {PIECES[k]: {"n": int(a[k, 0]), "found": float(a[k, 1] / max(a[k, 0], 1))} for k in range(3)}
              for v, a in total["pieces"].items()}
    pixels = {k: {"n": int(a[0]), **{f"marked@{t:g}": float(a[i + 1] / max(a[0], 1)) for i, t in enumerate(TAUS)}}
              for k, a in total["pixels"].items()}
    unmatched = {v: {"predicted": int(a[0]), "unmatched": int(a[1]), f"slivers_lt{SLIVER}px": int(a[2])}
                 for v, a in total["unmatched"].items()}
    return {"pieces": pieces, "pixels": pixels, "unmatched": unmatched}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--chips", required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    args = ap.parse_args()

    from gfm4agri.pipeline.config import load_experiment

    name = args.name or load_experiment(HERE / "experiment.yaml")["name"]
    results = REPO / "results" / name / args.chips
    chips_dir = REPO / "data" / "eurocrops_chips" / args.chips / "chips"
    path = results / "diagnostics.json"
    out = json.loads(path.read_text()) if path.exists() else {}
    for f in sorted(results.glob("*/P*_draw*_seed*/predictions_test.npz")):
        key = f"{f.parent.parent.name}/{f.parent.name}"
        if key in out:
            continue
        print(key, flush=True)
        out[key] = cell(f, chips_dir, args.workers)
        path.write_text(json.dumps(out, indent=1))
    print(f"{len(out)} cells in {path}")


if __name__ == "__main__":
    main()
