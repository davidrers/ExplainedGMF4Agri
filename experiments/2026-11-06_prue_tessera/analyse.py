"""Analyses of PRUE's test fields for REPORT.md, on the hub, beside ``score_prue``'s scores.

For every PRUE cell at K = 100 % it writes ``results/<name>/<set>/<arm>/<cell>/analysis.json``, comparing three
systems over the whole chip, none given an extent: PRUE's own fields (``cc``), the same grown into their rims
(``grow``), and PaPs on the same encoder (``results/2026-11-05_paps_tessera``, its parcels as stored):

* ``found``: the share of the true parcel pieces found (class-agnostic IoU > 0.5, void rule of ``panoptic_x``) by
  what they touch inside the declared extent (isolated; touching only other classes; touching their own class) and by
  area, with the bins of ``../2026-11-05_paps_tessera/analyse.py``, so the two reports read alike;
* ``errors``: what happened to every true piece. ``matched``, IoU > 0.5; ``merged``, one predicted field holds more
  than half of the piece but is too large for IoU > 0.5 (the piece is joined with a neighbour or spills into
  background); ``split``, the piece is mostly inside fields but no one field holds half of it; ``missed``, at most
  half of the piece lies inside any field (it was called background or boundary). Overall and by area;
* ``class_on_matched``: on the true pieces each system found, the crop-type map's vote against the true class.

Once per chip set it also writes ``results/<name>/<set>/ceiling.json``: the test chips' own three-class target taken as
the prediction (ignored pixels called background, as ``check.py``'s third check) and scored by ``score_prue`` exactly
as the predictions are, with the true parcels' class from the frozen crop-type map. It is the ceiling of each field
variant: what a perfect pixel map of the three classes would score.

    poetry run python experiments/2026-11-06_prue_tessera/analyse.py --chips EE_2021_rev
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

import panoptic_x as px  # noqa: E402
import prue  # noqa: E402

NAME = "2026-11-06_prue_tessera"
#: PRUE arm -> the crop-type map and the PaPs cell of the same encoder, K = 100 %, draw 0, seed 0.
SOURCES = {
    "prue_tessera_v2_frozen": {
        "cropmap": "results/kshot/{set}/tessera_v2_medium/P100_draw0_seed0",
        "paps": "results/2026-11-05_paps_tessera/{set}/paps_tessera_v2_frozen/P100_draw0_seed0"},
    "prue_tessera_v2_ft": {
        "cropmap": "results/2026-10-15_kshot_finetune/{set}/tessera_v2_medium_ft/P100_draw0_seed0",
        "paps": "results/2026-11-05_paps_tessera/{set}/paps_tessera_v2_ft/P100_draw0_seed0"},
}
SYSTEMS = ("cc", "grow", "paps")
KINDS = ("isolated", "touching other class", "touching same class")
#: Area bins of the true pieces in pixels (1 pixel = 0.01 ha), as the PaPs report.
SIZE_EDGES = (0, 50, 200, 500, 1000)
SIZE_LABELS = ("< 0.5 ha", "0.5 to 2 ha", "2 to 5 ha", "5 to 10 ha", "> 10 ha")
ERRORS = ("matched", "merged", "split", "missed")


def _neighbours(a: np.ndarray) -> list[np.ndarray]:
    q = np.pad(a, 1, mode="edge")
    return [q[:-2, 1:-1], q[2:, 1:-1], q[1:-1, :-2], q[1:-1, 2:]]


def kinds(E: np.ndarray, true: np.ndarray, true_cls: np.ndarray) -> np.ndarray:
    """Per true piece id: 0 isolated, 1 touching only other classes, 2 touching its own class (as PaPs' analyse)."""
    n = len(true_cls)
    same_c, any_c = np.zeros(n, bool), np.zeros(n, bool)
    for nb_true, nb_ext in zip(_neighbours(true), _neighbours(E)):
        other = E & nb_ext & (nb_true != true)
        any_c[true[other]] = True
        same_c[true[other & (true_cls[nb_true] == true_cls[true])]] = True
    return np.where(same_c, 2, np.where(any_c, 1, 0))


def error_types(true: np.ndarray, seg: np.ndarray, void: np.ndarray, ids: np.ndarray) -> np.ndarray:
    """Per true piece in ``ids``: 0 matched, 1 merged, 2 split, 3 missed (see the module docstring)."""
    ti, _, _, _, _ = px.match(true, seg, void)
    t, s = true.astype(np.int64), seg.astype(np.int64)
    at = np.bincount(t.ravel())
    sel = (t > 0) & (s > 0)
    covered = np.bincount(t[sel], minlength=len(at))
    best = np.zeros(len(at))
    if sel.any():
        base = int(s.max()) + 1
        pairs, inter = np.unique(t[sel] * base + s[sel], return_counts=True)
        np.maximum.at(best, pairs // base, inter)
    share_best, share_cov = best[ids] / at[ids], covered[ids] / at[ids]
    out = np.where(share_best > 0.5, 1, np.where(share_cov > 0.5, 2, 3))
    out[np.isin(ids, ti)] = 0
    return out


def chip(prob, inst, crop, mask, parcels, n) -> dict:
    E, true, true_cls, void = px.truth(mask, parcels)
    if not true.any():
        return {}
    ids = np.unique(true[true > 0])
    kind = kinds(E, true, true_cls)[ids]
    size = np.digitize(np.bincount(true.ravel())[ids], SIZE_EDGES[1:])
    segs = {"cc": prue.parcels(prob, "cc"), "grow": prue.parcels(prob, "grow"), "paps": inst.astype(np.int64)}
    out = {"_total": (np.concatenate([np.bincount(kind, minlength=3), np.bincount(size, minlength=5)])
                      .astype(float), None, None)}
    for name, seg in segs.items():
        err = error_types(true, seg, void, ids)
        hit = err == 0
        out[f"_found/{name}"] = (np.concatenate([np.bincount(kind[hit], minlength=3),
                                                 np.bincount(size[hit], minlength=5)]).astype(float), None, None)
        e = np.zeros((4, 5))
        np.add.at(e, (err, size), 1)
        out[f"_errors/{name}"] = (e.ravel(), None, None)
        ti, si, _, _, _ = px.match(true, seg, void)
        conf = np.zeros((n, n))
        if len(ti):
            np.add.at(conf, (true_cls[ti], px.vote(seg, crop, n)[si]), 1)
        out[f"_class/{name}"] = (conf.ravel(), None, None)
    return out


def _chunk(args) -> dict:
    items, chips_dir, n = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    acc: dict = {}
    for cid, prob, inst, crop, mask in items:
        px._add(acc, chip(prob, inst, crop, mask, read_parcels(chips_dir, cid), n))
    return acc


def _macro_f1(c: np.ndarray) -> tuple[float, float]:
    tp = np.diag(c)
    present = c.sum(1) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * tp / (c.sum(0) + c.sum(1))
    return float(tp.sum() / max(c.sum(), 1)), float(np.nan_to_num(f1[present]).mean())


def _aligned(path: Path, key: str, ids: list[str]) -> np.ndarray:
    with np.load(path) as d:
        order = {str(c): i for i, c in enumerate(d["ids"])}
        return d[key][[order[c] for c in ids]]


def analyse(cell: Path, chips: str, workers: int) -> dict:
    arm = cell.parent.name
    src = {k: REPO / v.format(set=chips) for k, v in SOURCES[arm].items()}
    r = json.loads((cell / "results.json").read_text())
    names = r["class_names"]
    n = len(names)
    with np.load(cell / "predictions_test.npz") as d:
        ids, prob, mask = [str(c) for c in d["ids"]], d["prob"], d["mask"]
    crop = _aligned(src["cropmap"] / "predictions_test.npz", "pred", ids)
    inst = _aligned(src["paps"] / "predictions_test.npz", "inst", ids)
    chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
    items = [(c, prob[i], inst[i], crop[i], mask[i]) for i, c in enumerate(ids)]
    parts = np.array_split(np.arange(len(items)), max(1, min(workers * 4, len(items))))
    total: dict = {}
    with ProcessPoolExecutor(workers) as ex:
        for acc in ex.map(_chunk, [([items[i] for i in p], chips_dir, n) for p in parts if len(p)]):
            px._add(total, acc)

    tot = total["_total"][0]
    out = {"systems": {"cc": "PRUE's fields, whole chip", "grow": "PRUE's fields grown into their rims, whole chip",
                       "paps": f"PaPs, whole chip ({src['paps'].relative_to(REPO)})"},
           "found": {"kinds": list(KINDS), "sizes": list(SIZE_LABELS), "size_edges_px": list(SIZE_EDGES),
                     "true_parcels": {"by_kind": tot[:3].astype(int).tolist(), "by_size": tot[3:].astype(int).tolist()},
                     "share": {}},
           "errors": {"types": list(ERRORS), "sizes": list(SIZE_LABELS), "counts": {}},
           "class_on_matched": {}}
    for s in SYSTEMS:
        f = total[f"_found/{s}"][0]
        out["found"]["share"][s] = {"by_kind": (f[:3] / np.maximum(tot[:3], 1)).tolist(),
                                    "by_size": (f[3:] / np.maximum(tot[3:], 1)).tolist(),
                                    "all": float(f[:3].sum() / tot[:3].sum())}
        out["errors"]["counts"][s] = total[f"_errors/{s}"][0].reshape(4, 5).astype(int).tolist()
        c = total[f"_class/{s}"][0].reshape(n, n)
        acc, f1 = _macro_f1(c)
        out["class_on_matched"][s] = {"parcels": int(c.sum()), "accuracy": acc, "macro_f1": f1}
    out["sources"] = {k: str(v.relative_to(REPO)) for k, v in src.items()}
    # Checks: the found share equals the stored class-agnostic TP over the true pieces.
    stored = r["panoptic_metrics"]["variants"]
    out["check_against_score_prue"] = {
        s: abs(out["found"]["share"][s]["all"] - stored[f"chip/{s}"]["tp_agnostic"]
               / stored[f"chip/{s}"]["true_parcels"]) < 1e-9 for s in ("cc", "grow")}
    return out


def ceiling(chips: str, workers: int) -> dict:
    """The test chips' own target as the prediction, scored as the predictions are (see the module docstring)."""
    import rasterio

    import score_prue
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    cell = REPO / "results" / NAME / chips / "prue_tessera_v2_frozen" / "P100_draw0_seed0"
    r = json.loads((cell / "results.json").read_text())
    with np.load(cell / "predictions_test.npz") as d:
        ids, masks = [str(c) for c in d["ids"]], d["mask"]
    chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
    wc_path = REPO / "data/worldcover" / chips.replace("_mini", "") / "worldcover.npz"
    wc = np.load(wc_path)
    probs = np.zeros((len(ids), 3, *masks.shape[1:]), np.uint8)
    for i, cid in enumerate(ids):
        with rasterio.open(chips_dir / f"{cid}.mask.tif") as m:
            mask = m.read(1).astype(np.int64)
        t = prue.target(mask, read_parcels(chips_dir, cid).astype(np.int64), wc[cid])
        t = np.where(t == prue.IGNORE, prue.BACKGROUND, t)
        probs[i] = (np.eye(3, dtype=np.uint8)[t] * 255).transpose(2, 0, 1)
    crop = _aligned(REPO / SOURCES["prue_tessera_v2_frozen"]["cropmap"].format(set=chips) / "predictions_test.npz",
                    "pred", ids)
    out = score_prue.score(probs, masks, ids, chips_dir, r["class_names"], wc_path,
                           crop={c: crop[i] for i, c in enumerate(ids)}, workers=workers)
    out["what"] = ("the test chips' own three-class target as the prediction (ignored pixels called background); "
                   "class of a field from the frozen crop-type map")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    dest = REPO / "results" / NAME / a.chips / "ceiling.json"
    if not dest.exists() or a.force:
        print("scoring the perfect map (ceiling)", flush=True)
        out = ceiling(a.chips, a.workers)
        dest.write_text(json.dumps(out, indent=1))
        for k in ("chip/cc", "chip/grow", "chip/field"):
            v = out["variants"][k]
            print(f"  {k}: PQ agnostic {v['pq_agnostic']:.3f} (RQ {v['rq_agnostic']:.3f}), PQ {v['pq']:.3f}")
    for arm in SOURCES:
        for cell in sorted((REPO / "results" / NAME / a.chips / arm).glob("P100_draw*_seed*")):
            dest = cell / "analysis.json"
            if not (cell / "predictions_test.npz").exists() or (dest.exists() and not a.force):
                continue
            print(f"analysing {arm}/{cell.name}", flush=True)
            out = analyse(cell, a.chips, a.workers)
            tmp = dest.with_name(f"analysis.json.{os.getpid()}")
            tmp.write_text(json.dumps(out, indent=1))
            tmp.replace(dest)
            print(f"  check against score_prue: {out['check_against_score_prue']}")
            for s in SYSTEMS:
                e = np.array(out["errors"]["counts"][s]).sum(1)
                print(f"  {s:>5}: found {out['found']['share'][s]['all']:.3f}; "
                      + ", ".join(f"{t} {k}" for t, k in zip(ERRORS, e))
                      + f"; class on matched {out['class_on_matched'][s]['accuracy']:.3f}")


if __name__ == "__main__":
    main()
