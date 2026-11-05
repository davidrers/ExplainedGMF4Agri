"""Analyses of PaPs' test parcels for REPORT.md, on the hub, beside ``score_paps``' scores.

For every PaPs cell at K = 100 % it writes ``results/<name>/<set>/<arm>/<cell>/analysis.json``:

* ``variants``: PaPs' parcels in every extent with two classes each, by the void rule of ``panoptic_x``:
  ``<extent>/paps``, the class of PaPs' class head (as ``score_paps``; a check that both agree), and
  ``<extent>/paps_cropmap``, the class most of the parcel's pixels take in the crop-type map of the same encoder (the
  main workflow's per-pixel MLP on frozen TESSERA, ``results/kshot``, or the fine-tuned student of
  ``2026-10-15_kshot_finetune``, the one PaPs' fine-tuned raster was encoded with; the vote of protocol D28). So PaPs
  gives the geometry and the crop-type model the class. ``truth/cropmap``: the true parcels with the crop map's
  vote, whose class-aware RQ is the parcel Macro-F1 at IoU > 0.5 with the true geometry, the reference of the cost of
  predicted geometry;
* ``found``: the share of the true parcel pieces found (class-agnostic IoU > 0.5, void rule) by what they touch inside
  the declared extent (isolated; touching only other classes; touching their own class, as
  ``2026-10-18_parcel_delineation/diagnose.py`` defines them) and by area, for PaPs over the whole chip and for the
  boundary parcels (tau 0.5) of the same encoder built over the whole chip, inside WorldCover 30 + 40 and inside the
  declared extent (the oracle, an upper bound that uses the test labels);
* ``class_on_matched``: on the true parcels PaPs found, the class of PaPs' class head and the crop map's vote against
  the true class: accuracy and macro F1 over the classes present.

    poetry run python experiments/2026-11-05_paps_tessera/analyse.py --chips EE_2021_rev
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

NAME = "2026-11-05_paps_tessera"
#: PaPs arm -> the crop-type map and the boundary-head cell of the same encoder, K = 100 %, draw 0, seed 0.
SOURCES = {
    "paps_tessera_v2_frozen": {
        "cropmap": "results/kshot/{set}/tessera_v2_medium/P100_draw0_seed0",
        "boundary": "results/2026-10-18_parcel_delineation/{set}/tessera_v2_medium/P100_draw0_seed0"},
    "paps_tessera_v2_ft": {
        "cropmap": "results/2026-10-15_kshot_finetune/{set}/tessera_v2_medium_ft/P100_draw0_seed0",
        "boundary": "results/2026-11-05_parcel_extent/{set}/tessera_v2_medium_ft_bnd/P100_draw0_seed0"},
}
EXTENTS = ("chip", "worldcover", "worldcover_30_40_90", "oracle")
KINDS = ("isolated", "touching other class", "touching same class")
#: Area bins of the true pieces in pixels (1 pixel = 0.01 ha).
SIZE_EDGES = (0, 50, 200, 500, 1000)
SIZE_LABELS = ("< 0.5 ha", "0.5 to 2 ha", "2 to 5 ha", "5 to 10 ha", "> 10 ha")
#: The systems whose found parcels are counted: PaPs over the whole chip, and the boundary parcels in three extents.
SYSTEMS = ("paps", "boundary/chip", "boundary/worldcover", "boundary/oracle")


def _neighbours(a: np.ndarray) -> list[np.ndarray]:
    q = np.pad(a, 1, mode="edge")
    return [q[:-2, 1:-1], q[2:, 1:-1], q[1:-1, :-2], q[1:-1, 2:]]


def kinds(E: np.ndarray, true: np.ndarray, true_cls: np.ndarray) -> np.ndarray:
    """Per true piece id: 0 isolated, 1 touching only other classes, 2 touching its own class (``diagnose.py``)."""
    n = len(true_cls)
    same_c, any_c = np.zeros(n, bool), np.zeros(n, bool)
    for nb_true, nb_ext in zip(_neighbours(true), _neighbours(E)):
        other = E & nb_ext & (nb_true != true)
        any_c[true[other]] = True
        same_c[true[other & (true_cls[nb_true] == true_cls[true])]] = True
    return np.where(same_c, 2, np.where(any_c, 1, 0))


def chip(inst, cls, crop, bprob, mask, parcels, wc, n) -> dict:
    E, true, true_cls, void = px.truth(mask, parcels)
    if not true.any():
        return {}
    inst = inst.astype(np.int64)
    paps_cls = np.zeros(int(inst.max()) + 1, np.int64)
    paps_cls[inst[inst > 0]] = cls[inst > 0]
    out = {}
    for x in EXTENTS:
        X = px.extent_of(x, mask, parcels, wc)
        seg = np.where(X, inst, 0)
        out[f"{x}/paps"] = px.chip_counts(true, true_cls, seg, paps_cls, void, n, (E, wc))
        out[f"{x}/paps_cropmap"] = px.chip_counts(true, true_cls, seg, px.vote(seg, crop, n), void, n, (E, wc))
    out["truth/cropmap"] = px.chip_counts(true, true_cls, true, px.vote(true, crop, n), void, n, (E, wc))

    # Found true pieces by kind and by area, per system.
    ids = np.unique(true[true > 0])
    kind = kinds(E, true, true_cls)[ids]
    size = np.digitize(np.bincount(true.ravel())[ids], SIZE_EDGES[1:])
    segs = {"paps": inst}
    if bprob is not None:
        p = bprob.astype(np.float32) / 255.0
        for x in ("chip", "worldcover", "oracle"):
            segs[f"boundary/{x}"] = px.from_boundary(px.extent_of(x, mask, parcels, wc), p, px.TAU)
    out["_found_total"] = (np.concatenate([np.bincount(kind, minlength=3), np.bincount(size, minlength=5)])
                           .astype(float), None, None)
    for name, seg in segs.items():
        ti, si, _, _, _ = px.match(true, seg, void)
        hit = np.isin(ids, ti)
        out[f"_found/{name}"] = (np.concatenate([np.bincount(kind[hit], minlength=3),
                                                 np.bincount(size[hit], minlength=5)]).astype(float), None, None)

    # The class of the parcels PaPs found: its class head and the crop map's vote, against the truth.
    ti, si, _, _, _ = px.match(true, inst, void)
    conf = np.zeros((2, n, n))
    if len(ti):
        np.add.at(conf[0], (true_cls[ti], paps_cls[si]), 1)
        np.add.at(conf[1], (true_cls[ti], px.vote(inst, crop, n)[si]), 1)
    out["_class_on_matched"] = (conf.ravel(), None, None)
    return out


def _chunk(args) -> dict:
    items, chips_dir, n, wc_path = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    wc = np.load(wc_path)
    acc: dict = {}
    for cid, inst, cls, crop, bprob, mask in items:
        px._add(acc, chip(inst, cls, crop, bprob, mask, read_parcels(chips_dir, cid), wc[cid], n))
    return acc


def _macro_f1(c: np.ndarray) -> tuple[float, float]:
    tp = np.diag(c)
    present = c.sum(1) > 0
    with np.errstate(invalid="ignore", divide="ignore"):
        f1 = 2 * tp / (c.sum(0) + c.sum(1))
    return float(tp.sum() / c.sum()), float(np.nan_to_num(f1[present]).mean())


def analyse(cell: Path, chips: str, workers: int) -> dict:
    arm = cell.parent.name
    src = {k: REPO / v.format(set=chips) for k, v in SOURCES[arm].items()}
    r = json.loads((cell / "results.json").read_text())
    names = r["class_names"]
    n = len(names)
    with np.load(cell / "predictions_test.npz") as d:
        ids, inst, cls, mask = [str(c) for c in d["ids"]], d["inst"], d["cls"], d["mask"]
    cm_r = json.loads((src["cropmap"] / "results.json").read_text())
    assert cm_r["class_names"] == names, "crop map and PaPs name the classes differently"
    with np.load(src["cropmap"] / "predictions_test.npz") as d:
        assert [str(c) for c in d["ids"]] == ids
        crop = d["pred"]
    bprob, b_note = None, "missing"
    if (src["boundary"] / "predictions_test.npz").exists():
        with np.load(src["boundary"] / "predictions_test.npz") as d:
            order = {str(c): i for i, c in enumerate(d["ids"])}
            bprob = d["boundary"][[order[c] for c in ids]]
        b_note = str(src["boundary"].relative_to(REPO))
    chips_dir = REPO / "data/eurocrops_chips" / chips / "chips"
    wc = str(REPO / "data/worldcover" / chips.replace("_mini", "") / "worldcover.npz")
    items = [(c, inst[i], cls[i], crop[i], None if bprob is None else bprob[i], mask[i]) for i, c in enumerate(ids)]
    parts = np.array_split(np.arange(len(items)), max(1, min(workers * 4, len(items))))
    total: dict = {}
    with ProcessPoolExecutor(workers) as ex:
        for acc in ex.map(_chunk, [([items[i] for i in p], chips_dir, n, wc) for p in parts if len(p)]):
            px._add(total, acc)

    out = px.summarise({k: v for k, v in total.items() if not k.startswith("_")}, names)
    out.pop("extents", None)
    tot = total["_found_total"][0]
    out["found"] = {"kinds": list(KINDS), "sizes": list(SIZE_LABELS), "size_edges_px": list(SIZE_EDGES),
                    "true_parcels": {"by_kind": tot[:3].astype(int).tolist(), "by_size": tot[3:].astype(int).tolist()},
                    "share": {}}
    for s in SYSTEMS:
        if f"_found/{s}" in total:
            f = total[f"_found/{s}"][0]
            out["found"]["share"][s] = {"by_kind": (f[:3] / np.maximum(tot[:3], 1)).tolist(),
                                        "by_size": (f[3:] / np.maximum(tot[3:], 1)).tolist(),
                                        "all": float(f[:3].sum() / tot[:3].sum())}
    conf = total["_class_on_matched"][0].reshape(2, n, n)
    out["class_on_matched"] = {"parcels": int(conf[0].sum())}
    for k, c in zip(("paps_class_head", "cropmap_vote"), conf):
        acc, f1 = _macro_f1(c)
        out["class_on_matched"][k] = {"accuracy": acc, "macro_f1": f1}
    out["sources"] = {"cropmap": str(src["cropmap"].relative_to(REPO)),
                      "cropmap_pixel_macro_f1": cm_r["test_metrics"].get("test/F1_Score"),
                      "cropmap_parcel_macro_f1": cm_r.get("parcel_metrics", {}).get("macro_f1"),
                      "boundary": b_note, "tau": px.TAU}
    # Check: PaPs with its own class equals score_paps' stored scores.
    stored = r.get("panoptic_metrics", {}).get("variants", {})
    out["check_against_score_paps"] = {k: abs(out["variants"][k]["pq"] - stored[k]["pq"]) < 1e-12
                                       for k in out["variants"] if k.endswith("/paps") and k in stored}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
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
            v = out["variants"]
            print(f"  check against score_paps: {out['check_against_score_paps']}")
            for x in EXTENTS:
                print(f"  {x:>20}: PaPs class PQ {v[f'{x}/paps']['pq']:.3f}, crop-map class PQ "
                      f"{v[f'{x}/paps_cropmap']['pq']:.3f}, agnostic {v[f'{x}/paps']['pq_agnostic']:.3f}")
            print(f"  true parcels, crop-map class: RQ {v['truth/cropmap']['rq']:.3f}")
            print(f"  found: {json.dumps({k: round(s['all'], 3) for k, s in out['found']['share'].items()})}")
            print(f"  class on matched: {out['class_on_matched']}")


if __name__ == "__main__":
    main()
