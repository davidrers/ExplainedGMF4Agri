"""Border precision, recall and F1 of each method's final fields against the borders of the declared parcels.

**Borders.** The border of a set of fields, or of the declared parcels, is every field pixel with a 4-neighbour
inside the chip that carries another id or none (``prue.boundary_map``): two pixels wide between neighbours, one
at an outer edge; the chip edge is no border. The fields are each method's final output over the whole chip, with
no test label used: PRUE's fields grown into their rims, PaPs' parcels, and the boundary head's watershed at 0.5
over the chip.

**Where.** Scored where the truth is known: the declared in-scheme parcels, and the land outside every declared
parcel that WorldCover calls non-agricultural. Not scored: declared parcels of crops outside the 20 classes, and
undeclared WorldCover grassland and cropland, where nobody says whether a field lies.

**Scores**, per tolerance d in pixels (any direction, so d = 1 is a 3 x 3 window):

* precision: the share of the predicted border pixels with a true border pixel within d;
* recall: the share of the true border pixels with a predicted border pixel within d;
* F1: their harmonic mean.

At d = 1 the recall is also split by what lies across each true border pixel (another declared parcel of the same
crop, of another crop, undeclared land WorldCover calls grassland or cropland, or other land; the first that
applies in that order), and the false border pixels by where
they lie (on a declared parcel, a false cut; or on non-farm land). The declared parcels themselves, taken as the
prediction, must score 1 (the check). Per chip, the d = 1 counts go to ``per_chip.npz`` for the report.

    poetry run python experiments/2026-11-07_border_accuracy/borders.py --chips EE_2021_rev --workers 12
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

import panoptic_x as px  # noqa: E402
import prue  # noqa: E402

NAME = "2026-11-07_border_accuracy"
#: method -> (cell under results/, the stored key, how the fields are built). K = 100 %, draw 0, seed 0.
SOURCES = {
    "PRUE frozen": ("2026-11-06_prue_tessera/{set}/prue_tessera_v2_frozen", "prob", "prue"),
    "PRUE fine-tuned": ("2026-11-06_prue_tessera/{set}/prue_tessera_v2_ft", "prob", "prue"),
    "PaPs frozen": ("2026-11-05_paps_tessera/{set}/paps_tessera_v2_frozen", "inst", "inst"),
    "PaPs fine-tuned": ("2026-11-05_paps_tessera/{set}/paps_tessera_v2_ft", "inst", "inst"),
    "boundary head frozen": ("2026-10-18_parcel_delineation/{set}/tessera_v2_medium", "boundary", "head"),
    "boundary head fine-tuned": ("2026-11-05_parcel_extent/{set}/tessera_v2_medium_ft_bnd", "boundary", "head"),
}
CELL = "P100_draw0_seed0"
TOLERANCES = (0, 1, 2)
TYPES = ("same crop", "other crop", "undeclared farmland", "non-farm land")
PLACES = ("on a declared parcel", "on non-farm land")
CHECK = "declared parcels (check)"


def fields(kind: str, a: np.ndarray) -> np.ndarray:
    """One method's final fields of one chip (0 = no field) from its stored output."""
    if kind == "prue":
        return prue.parcels(a, "grow")
    if kind == "inst":
        return a.astype(np.int32)
    return px.from_boundary(np.ones(a.shape, bool), a.astype(np.float32) / 255.0, px.TAU)


def regions(mask: np.ndarray, parcels: np.ndarray, wc: np.ndarray):
    """The declared in-scheme parcels E, the scored pixels, and the true border inside E."""
    E = (mask >= 0) & (parcels > 0)
    scored = E | ((parcels == 0) & ~np.isin(wc, prue.AGRI))
    return E, scored, E & prue.boundary_map(parcels)


def border_types(parcels: np.ndarray, mask: np.ndarray, wc: np.ndarray, true_border: np.ndarray) -> np.ndarray:
    """Per pixel, what lies across: 0 same crop, 1 other crop, 2 undeclared farmland, 3 non-farm land; -1 off the
    true border."""
    farm = np.isin(wc, prue.AGRI)
    h, w = parcels.shape
    same = np.zeros((h, w), bool)
    other = np.zeros((h, w), bool)
    undeclared = np.zeros((h, w), bool)
    nonfarm = np.zeros((h, w), bool)
    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        ys, yd = (slice(0, h - dy), slice(dy, h)) if dy >= 0 else (slice(-dy, h), slice(0, h + dy))
        xs, xd = (slice(0, w - dx), slice(dx, w)) if dx >= 0 else (slice(-dx, w), slice(0, w + dx))
        p, q = parcels[ys, xs], parcels[yd, xd]
        m, n = mask[ys, xs], mask[yd, xd]
        diff = p != q
        undeclared[ys, xs] |= diff & (q == 0) & farm[yd, xd]
        nonfarm[ys, xs] |= diff & (q == 0) & ~farm[yd, xd]
        same[ys, xs] |= diff & (q > 0) & (n == m) & (n >= 0)
        other[ys, xs] |= diff & (q > 0) & ((n != m) | (n < 0))
    out = np.full((h, w), -1, np.int8)
    out[true_border & nonfarm] = 3
    out[true_border & undeclared] = 2
    out[true_border & other] = 1
    out[true_border & same] = 0
    return out


def _near(border: np.ndarray, d: int) -> np.ndarray:
    return ndimage.binary_dilation(border, np.ones((2 * d + 1, 2 * d + 1), bool)) if d else border


def chip_counts(seg, true_border, scored, E, types) -> dict:
    """The counts of one method on one chip: per tolerance, by border type and by place of the false borders."""
    pred = prue.boundary_map(seg)
    p = pred & scored
    tol = []
    for d in TOLERANCES:
        near_true, near_pred = _near(true_border, d), _near(pred, d)
        tol.append([(p & near_true).sum(), p.sum(), (true_border & near_pred).sum(), true_border.sum()])
    near_pred = _near(pred, 1)
    by_type = [[(near_pred & (types == k)).sum(), (types == k).sum()] for k in range(len(TYPES))]
    false = p & ~_near(true_border, 1)
    places = [(false & E).sum(), (false & ~E).sum()]
    return {"tol": np.array(tol, np.int64), "types": np.array(by_type, np.int64), "places": np.array(places, np.int64)}


def _chunk(args):
    ids, masks, wcs, arrays, kinds, chips_dir = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels

    total: dict = {}
    per_chip = {m: [] for m in [*arrays, CHECK]}
    for i, cid in enumerate(ids):
        parcels, mask = read_parcels(chips_dir, cid), masks[i].astype(np.int64)
        E, scored, true_border = regions(mask, parcels, wcs[i])
        types = border_types(parcels, mask, wcs[i], true_border)
        segs = {m: fields(kinds[m], arrays[m][i]) for m in arrays}
        segs[CHECK] = parcels.astype(np.int64)
        for m, seg in segs.items():
            c = chip_counts(seg, true_border, scored, E, types)
            t = total.setdefault(m, {k: 0 for k in c})
            for k, v in c.items():
                t[k] = t[k] + v
            per_chip[m].append(c["tol"][TOLERANCES.index(1)])
    return total, {m: np.array(v, np.int64) for m, v in per_chip.items()}


def load(chips: str):
    """Test chip ids, masks, WorldCover, and every method's stored output, all in the same chip order."""
    arrays, kinds, ids, masks = {}, {}, None, None
    for m, (rel, key, kind) in SOURCES.items():
        z = np.load(REPO / "results" / rel.format(set=chips) / CELL / "predictions_test.npz")
        mid = [str(c) for c in z["ids"]]
        if ids is None:
            ids, masks = mid, z["mask"]
        if mid != ids:
            raise SystemExit(f"{m}: its test chips differ from those of {next(iter(SOURCES))}")
        arrays[m], kinds[m] = z[key], kind
    wc = np.load(REPO / "data/worldcover" / chips.replace("_mini", "") / "worldcover.npz")
    return ids, masks, np.stack([wc[c] for c in ids]), arrays, kinds


def _f1(p: float, r: float) -> float:
    return 2 * p * r / (p + r) if p + r else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chips", default="EE_2021_rev")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--chunk", type=int, default=40)
    a = ap.parse_args()

    chips_dir = REPO / "data/eurocrops_chips" / a.chips / "chips"
    ids, masks, wcs, arrays, kinds = load(a.chips)
    jobs = [(ids[s:s + a.chunk], masks[s:s + a.chunk], wcs[s:s + a.chunk],
             {m: v[s:s + a.chunk] for m, v in arrays.items()}, kinds, chips_dir) for s in range(0, len(ids), a.chunk)]
    total: dict = {}
    per_chip: dict = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for t, pc in ex.map(_chunk, jobs):
            for m, c in t.items():
                acc = total.setdefault(m, {k: 0 for k in c})
                for k, v in c.items():
                    acc[k] = acc[k] + v
            for m, v in pc.items():
                per_chip.setdefault(m, []).append(v)

    out = {"chips": len(ids), "tolerances_px": list(TOLERANCES), "sources": SOURCES, "cell": CELL, "methods": {}}
    for m, c in total.items():
        row = {}
        for k, d in enumerate(TOLERANCES):
            tp_p, n_p, tp_r, n_r = (int(x) for x in c["tol"][k])
            p, r = tp_p / max(n_p, 1), tp_r / max(n_r, 1)
            row[f"d{d}"] = {"precision": p, "recall": r, "f1": _f1(p, r), "pred_border_px": n_p,
                            "true_border_px": n_r}
        row["recall_by_type_d1"] = {t: {"recall": int(f) / max(int(n), 1), "true_border_px": int(n)}
                                    for t, (f, n) in zip(TYPES, c["types"])}
        n_p = int(c["tol"][TOLERANCES.index(1)][1])
        row["false_border_share_d1"] = {pl: int(v) / max(n_p, 1) for pl, v in zip(PLACES, c["places"])}
        out["methods"][m] = row

    dest = REPO / "results" / NAME / a.chips
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "borders.json").write_text(json.dumps(out, indent=1))
    np.savez_compressed(dest / "per_chip.npz", ids=np.array(ids),
                        methods=np.array(list(per_chip)), counts=np.stack([np.concatenate(per_chip[m])
                                                                            for m in per_chip]))
    print(f"{len(ids)} test chips")
    print(f"{'method':28s} " + "  ".join(f"d={d}: prec   rec    F1 " for d in TOLERANCES))
    for m, row in out["methods"].items():
        print(f"{m:28s} " + "  ".join(f"{row[f'd{d}']['precision']:.3f}  {row[f'd{d}']['recall']:.3f}  "
                                      f"{row[f'd{d}']['f1']:.3f}" for d in TOLERANCES))
    print("recall at d=1 by type:", {m: {t: round(v["recall"], 3) for t, v in r["recall_by_type_d1"].items()}
                                     for m, r in out["methods"].items()})
    print("false borders at d=1:", {m: {k: round(v, 3) for k, v in r["false_border_share_d1"].items()}
                                    for m, r in out["methods"].items()})
    print(f"wrote {(dest / 'borders.json').relative_to(REPO)} and per_chip.npz")


if __name__ == "__main__":
    main()
