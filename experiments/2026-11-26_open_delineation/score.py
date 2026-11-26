"""Every model's fields and borders scored exactly as the two-head workflow scores its arms, on the same chips.

Run in the Poetry environment, after the predictions are on disk::

    poetry run python experiments/2026-11-26_open_delineation/score.py [--only <label prefix>] [--workers 32]

**One scoring path for every model.** Each prediction becomes a set of fields per chip, and those fields are scored
by the core's functions, read-only (``gfm4agri.borders.scoring``: ``truth``, ``field_counts``, ``quality``,
``regions``, ``border_types``, ``border_counts``):

* a three-class map (FTW's models, and our arms' stored ``predictions_test.npz``): ``prue.parcels(prob, "grow")``,
  the fields the two-head workflow builds (each region of interior, grown by watershed into the predicted field),
  and for FTW also ``"cc"``, the interior regions alone, which is what FTW's own polygonisation keeps;
* instances (Delineate Anything, U-TAE + PaPs): the model's own fields, below a confidence threshold removed, split
  into their 4-connected pieces (as the true parcels are) and pieces under 5 pixels dropped (``prue.MIN_PIXELS``,
  FTW's 500 m2).

The rules are the workflow's: fields over the whole chip with no truth, the void rule, a match at IoU > 0.5
(class-agnostic PQ, SQ, RQ; RQ is FTW's object F1), chips without a true parcel not counted for the fields; borders
within 0, 1 and 2 pixels, scored on the declared in-scheme parcels and on WorldCover's non-farm land. Rescoring our
arms this way reproduces the border F1 and PQ stored in their ``results.json`` (checked at the start of every run).

**Two truths.** ``all``: our parcels as they are. ``no_grass``: the parcels of grassland (class 0,
``pasture_meadow_grassland_grass``) made void, so that neither their fields nor their borders are scored, because
Fields of The World's Estonian labels call 82 % of our grassland pixels background (``ftw_labels.py``).

**Subsets.** ``test``: the 1,475 test chips; ``test_clean``: the 1,137 test chips that touch no chip of Fields of The
World's Estonia (``ftw_overlap.py``), on which no FTW model can have seen the labels or the place; ``val``: the 769
validation chips, on which the variants of each external model are chosen.

Per chip counts are kept (``scores/<label>__<split>.npz``), so that any subset and the block bootstrap of
``report.py`` are sums over chips. The summary is ``scores/summary.csv`` and ``scores/summary.json``.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import common as C

SCORES = C.RESULTS / "scores"
TRUTHS = ("all", "no_grass")
GRASS = 0
N_CLASSES = 20
#: Confidence thresholds applied to the instance models (their own default first).
THRESHOLDS = {"delany": (0.15, 0.05, 0.3, 0.5), "utae_paps": (0.2,)}
ARMS = ("unet_b7_s2s1", "tessera_v2_medium", "alphaearth_v1", "prithvi_eo_v2_600_tl", "terramind_v1_large_s2s1",
        "thor_v1_large_s2s1")
BUDGETS = (100, 20, 5)


# ------------------------------------------------------------------------------------------------ per chip

def fields_of(item: dict, variant: str) -> np.ndarray:
    from gfm4agri.borders import prue

    if "prob" in item:
        return prue.parcels(item["prob"], variant)
    seg = item["inst"].copy()
    conf = item.get("conf")
    if conf is not None:
        keep = conf >= float(variant)
        keep[0] = False
        seg = np.where(keep[seg], seg, 0)
    return prue._drop_small(prue.pieces(seg), prue.MIN_PIXELS)


def chip_counts(seg, mask, parcels, wc) -> np.ndarray:
    """One flat vector per truth: agnostic [TP, FP, FN, sum IoU], extra [dropped, FP on parcels, FP by WC x 9],
    border tol [3 x 4], types [4 x 2], places [2]."""
    from gfm4agri.borders import scoring as S

    rows = []
    for name in TRUTHS:
        m = mask if name == "all" else np.where(mask == GRASS, -1, mask).astype(mask.dtype)
        E, true, true_cls, void = S.truth(m, parcels)
        if true.any():
            _, agn, extra = S.field_counts(true, true_cls, seg, np.zeros(int(seg.max()) + 1, np.int64), void,
                                           N_CLASSES, (E, wc))
        else:
            agn, extra = np.zeros(4), np.zeros(3 + len(S.FP_WC))
        Er, scored, true_border = S.regions(m, parcels, wc)
        types = S.border_types(parcels, m, wc, true_border)
        bc = S.border_counts(seg, true_border, scored, Er, types)
        rows.append(np.concatenate([agn, extra, bc["tol"].ravel(), bc["types"].ravel(), bc["places"].ravel()]))
    return np.stack(rows)


def _chunk(args):
    items, variants = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    wc = np.load(C.WORLDCOVER)
    out = {v: [] for v in variants}
    for it in items:
        parcels = read_parcels(C.CHIP_DIR, it["id"])
        for v in variants:
            out[v].append(chip_counts(fields_of(it, v), it["mask"].astype(np.int64), parcels, wc[it["id"]]))
    return {v: np.stack(a) for v, a in out.items()}


def load_items(path: Path) -> list[dict]:
    with np.load(path) as z:
        ids, mask = [str(c) for c in z["ids"]], z["mask"]
        prob = z["prob"] if "prob" in z.files else None
        inst = z["inst"] if "inst" in z.files else None
        conf = z["conf"] if "conf" in z.files else None
        off = z["conf_offsets"] if "conf_offsets" in z.files else None
    items = []
    for i, c in enumerate(ids):
        it = {"id": c, "mask": mask[i]}
        if prob is not None:
            it["prob"] = prob[i]
        else:
            it["inst"] = inst[i]
            if conf is not None:
                it["conf"] = conf[off[i]:off[i + 1]]
        items.append(it)
    return items


def count_file(path: Path, variants, workers: int):
    items = load_items(path)
    chunks = [items[s:s + 20] for s in range(0, len(items), 20)]
    parts = {v: [] for v in variants}
    with ProcessPoolExecutor(workers) as ex:
        for res in ex.map(_chunk, [(ch, variants) for ch in chunks]):
            for v in variants:
                parts[v].append(res[v])
    return [it["id"] for it in items], {v: np.concatenate(p) for v, p in parts.items()}


# ------------------------------------------------------------------------------------------------- summaries

def summarise(counts: np.ndarray) -> dict:
    """Scores of summed per chip counts ``(n_chips, len(TRUTHS), K)`` for one truth slice ``(n_chips, K)``."""
    from gfm4agri.borders import scoring as S

    c = counts.sum(0)
    nfp = len(S.FP_WC)
    agn, extra = c[:4], c[4:4 + 3 + nfp]
    o = 4 + 3 + nfp
    tol = c[o:o + 12].reshape(3, 4)
    types = c[o + 12:o + 20].reshape(4, 2)
    places = c[o + 20:o + 22]
    q = S.quality(np.zeros((N_CLASSES, 4)), agn, [str(i) for i in range(N_CLASSES)], extra)
    row = {k: q[k] for k in ("pq_agnostic", "sq_agnostic", "rq_agnostic", "tp_agnostic", "fp_agnostic",
                             "fn_agnostic", "true_parcels", "predicted_parcels", "fp_on_parcels")}
    row["fp_on_background"] = q["fp_on_background"]
    b = S._border_row({"tol": tol.astype(np.int64), "types": types.astype(np.int64),
                       "places": places.astype(np.int64)})
    for d in (0, 1, 2):
        for k in ("precision", "recall", "f1"):
            row[f"border_{k}_d{d}"] = b[f"d{d}"][k]
    row["true_border_px"] = b["d1"]["true_border_px"]
    row["pred_border_px"] = b["d1"]["pred_border_px"]
    for t, v in b["recall_by_type_d1"].items():
        row[f"recall_{t.replace(' ', '_')}"] = v["recall"]
    return row


def subsets(ids: list[str], split: str) -> dict[str, np.ndarray]:
    if split == "val":
        return {"val": np.ones(len(ids), bool)}
    ov = pd.read_csv(C.HERE / "reference" / "ftw_overlap.csv").set_index("chip_id")["ftw"]
    clean = np.array([ov.get(c, "none") == "none" for c in ids])
    return {"test": np.ones(len(ids), bool), "test_clean": clean}


# --------------------------------------------------------------------------------------------------- sources

def sources() -> list[dict]:
    """Every prediction file to score: label, path, split, the field variants."""
    out = []
    for split in ("val", "test"):
        for p in sorted(C.RESULTS.glob(f"*/*/predictions_{split}.npz")):
            model, variant = p.parent.parent.name, p.parent.name
            if "fused" in variant:                               # a three-class map built from instances
                fv = ["grow"]
            elif model.startswith("delany"):
                fv = [str(t) for t in THRESHOLDS["delany"]]
            elif model == "utae_paps":
                fv = [str(t) for t in THRESHOLDS["utae_paps"]]
            else:                                                # FTW: our fields, and FTW's own
                fv = ["grow", "cc"]
            out.append({"label": f"{model}/{variant}", "path": p, "split": split, "variants": fv})
    for arm in ARMS:
        for k in BUDGETS:
            p = C.ARMS_RESULTS / arm / f"P{k}_draw0_seed0" / "predictions_test.npz"
            if p.exists():
                out.append({"label": f"arm/{arm}/P{k}", "path": p, "split": "test", "variants": ["grow"],
                            "results_json": p.parent / "results.json"})
    return out


def check_arm(rows: pd.DataFrame, src: dict) -> None:
    """The rescored arm must reproduce the border F1 and PQ its two-head fit stored."""
    r = json.loads(Path(src["results_json"]).read_text())
    got = rows[(rows.truth == "all") & (rows.subset == "test") & (rows.fields == "grow")].iloc[0]
    want_b = r["border_metrics"]["d1"]["f1"]
    want_pq = r["panoptic_metrics"]["variants"]["chip/grow"]["pq_agnostic"]
    if abs(got.border_f1_d1 - want_b) > 1e-9 or abs(got.pq_agnostic - want_pq) > 1e-6:
        raise RuntimeError(f"{src['label']}: rescored border F1 {got.border_f1_d1} / PQ {got.pq_agnostic}, "
                           f"stored {want_b} / {want_pq}")
    print(f"  check {src['label']}: border F1 {want_b:.6f} and PQ {want_pq:.6f} reproduced", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", default=None, help="label prefixes to (re)score")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--force", action="store_true", help="recount files already counted")
    args = ap.parse_args()
    SCORES.mkdir(parents=True, exist_ok=True)
    for src in sources():
        if args.only and not any(src["label"].startswith(o) for o in args.only):
            continue
        cfile = SCORES / f"{src['label'].replace('/', '__')}__{src['split']}.npz"
        if cfile.exists() and not args.force and cfile.stat().st_mtime > Path(src["path"]).stat().st_mtime:
            continue
        t0 = time.time()
        ids, counts = count_file(src["path"], src["variants"], args.workers)
        np.savez_compressed(cfile, ids=np.array(ids), variants=np.array(src["variants"]),
                            **{f"v_{v}": counts[v] for v in src["variants"]})
        print(f"{src['label']} {src['split']}: {len(ids)} chips, {len(src['variants'])} field rules, "
              f"{time.time() - t0:.0f} s", flush=True)
        if "results_json" in src:
            check_arm(table_of(cfile, src), src)
    rows = [table_of(SCORES / f"{s['label'].replace('/', '__')}__{s['split']}.npz", s) for s in sources()
            if (SCORES / f"{s['label'].replace('/', '__')}__{s['split']}.npz").exists()]
    df = pd.concat(rows, ignore_index=True)
    df.drop(columns=["fp_on_background"]).to_csv(SCORES / "summary.csv", index=False)
    df.to_json(SCORES / "summary.json", orient="records", indent=1)
    print(f"summary: {len(df)} rows -> {SCORES / 'summary.csv'}")


def table_of(cfile: Path, src: dict) -> pd.DataFrame:
    z = np.load(cfile)
    ids = [str(c) for c in z["ids"]]
    rows = []
    for v in [str(x) for x in z["variants"]]:
        counts = z[f"v_{v}"]
        for sub, sel in subsets(ids, src["split"]).items():
            for t, name in enumerate(TRUTHS):
                row = {"label": src["label"], "split": src["split"], "subset": sub, "truth": name, "fields": v,
                       "n_chips": int(sel.sum())}
                row.update(summarise(counts[sel, t]))
                rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    main()
