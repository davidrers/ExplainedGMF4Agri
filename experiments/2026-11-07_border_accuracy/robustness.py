"""How much the border scores of ``borders.py`` can be trusted: three checks on the same test chips.

* **Strict precision.** ``borders.py`` leaves undeclared WorldCover farmland unscored. Here every pixel but the
  declared parcels of crops outside the 20 classes is scored, so a border drawn on undeclared farmland counts as
  false: a lower bound, since some of that land is farmed.
* **Extent only.** Fields that split nothing: PRUE's predicted field left whole (interior and boundary as one patch
  per connected region), and WorldCover's farmland (grassland and cropland) as one patch per connected region. What
  a method scores from knowing where farmland ends, without separating any parcel. PRUE's own fields (its interior
  patches, before they are grown into their rims) are added for reference.
* **Borders between parcels.** The recall on the real border pixels that separate two declared parcels (same crop or
  another crop), leaving out the outer edges against undeclared or non-farm land.

Writes ``results/2026-11-07_border_accuracy/<set>/robustness.json``.

    poetry run python experiments/2026-11-07_border_accuracy/robustness.py --chips EE_2021_rev --workers 12
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

import borders  # noqa: E402
import prue  # noqa: E402


def chunk(args):
    """Per method, summed over the chips: scored precision counts, strict ones, recall, recall between parcels."""
    ids, masks, wcs, arrays, kinds, chips_dir = args
    from gfm4agri.benchmark.parcel_metrics import read_parcels
    acc = {}
    for i, cid in enumerate(ids):
        parcels, mask, wc = read_parcels(chips_dir, cid), masks[i].astype(np.int64), wcs[i]
        E, scored, tb = borders.regions(mask, parcels, wc)
        strict = ~((mask < 0) & (parcels > 0))          # every pixel but out-of-scheme parcels
        types = borders.border_types(parcels, mask, wc, tb)
        between = (types == 0) | (types == 1)           # parcel-to-parcel borders
        segs = {m: borders.fields(kinds[m], arrays[m][i]) for m in arrays}
        for enc in ("frozen", "fine-tuned"):
            prob = arrays[f"PRUE {enc}"][i]
            segs[f"PRUE {enc}, field unsplit"] = prue.parcels(prob, "field")
            segs[f"PRUE {enc}, own fields"] = prue.parcels(prob, "cc")
        segs["WorldCover farmland, unsplit"] = prue.pieces(np.isin(wc, prue.AGRI).astype(np.int32))
        for m, seg in segs.items():
            pred = prue.boundary_map(seg)
            nt, npd = borders._near(tb, 1), borders._near(pred, 1)
            row = np.array([(pred & scored & nt).sum(), (pred & scored).sum(),
                            (pred & strict & nt).sum(), (pred & strict).sum(),
                            (tb & npd).sum(), tb.sum(),
                            (between & npd).sum(), between.sum()], np.int64)
            acc[m] = acc.get(m, 0) + row
    return acc

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--chips", default="EE_2021_rev")
    ap.add_argument("--workers", type=int, default=12)
    a = ap.parse_args()
    ids, masks, wcs, arrays, kinds = borders.load(a.chips)
    chips_dir = REPO / "data/eurocrops_chips" / a.chips / "chips"
    jobs = [(ids[s:s + 40], masks[s:s + 40], wcs[s:s + 40], {m: v[s:s + 40] for m, v in arrays.items()}, kinds,
             chips_dir) for s in range(0, len(ids), 40)]
    tot = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for part in ex.map(chunk, jobs):
            for m, v in part.items():
                tot[m] = tot.get(m, 0) + v

    def f1(p, r):
        return 2 * p * r / (p + r) if p + r else 0.0

    out = {}
    print(f"{'fields':38s} {'P':>6s} {'R':>6s} {'F1':>6s} | {'P strict':>8s} {'F1 strict':>9s} | R between parcels")
    for m, (tp, n, tps, ns, tr, nr, tb, nb) in tot.items():
        p, r, ps, rb = tp / max(n, 1), tr / max(nr, 1), tps / max(ns, 1), tb / max(nb, 1)
        out[m] = {"precision": p, "recall": r, "f1": f1(p, r), "precision_strict": ps, "f1_strict": f1(ps, r),
                  "recall_between_parcels": rb}
        print(f"{m:38s} {p:6.3f} {r:6.3f} {f1(p, r):6.3f} | {ps:8.3f} {f1(ps, r):9.3f} | {rb:17.3f}")
    dest = REPO / "results" / borders.NAME / a.chips / "robustness.json"
    dest.write_text(json.dumps({"chips": len(ids), "tolerance_px": 1, "methods": out}, indent=1))
    print(f"wrote {dest.relative_to(REPO)}")


if __name__ == "__main__":
    main()
