"""Check ``EE_2021_rev`` against its parent once the export has finished, and write ``check.json`` here.

* every chip of the parent has a re-exported composite (a file, not a link) and a report;
* the label and parcel rasters, re-rasterised by the export, are identical to the parent's;
* every report records the ``REVISED`` screening;
* per month, the gap-filled share of every chip, for the parent and the new set;
* per month, white pixels (B02, B03 and B04 all above 0.25 reflectance) in the composites of a sample of chips.

    poetry run python experiments/2026-10-12_s2_revised_export/check.py --workers 16
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import rasterio

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PARENT = REPO / "data" / "eurocrops_chips" / "EE_2021"
NEW = REPO / "data" / "eurocrops_chips" / "EE_2021_rev"
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def labels_equal(cid: str) -> tuple[str, bool, bool]:
    out = []
    for suffix in (".mask.tif", ".parcels.tif"):
        with rasterio.open(PARENT / "chips" / f"{cid}{suffix}") as a, rasterio.open(NEW / "chips" / f"{cid}{suffix}") as b:
            out.append(bool(np.array_equal(a.read(), b.read())))
    return cid, *out


def white_per_month(cid: str) -> tuple[str, list[int], list[int]]:
    res = []
    for root in (PARENT, NEW):
        with rasterio.open(root / "chips" / f"{cid}_merged.tif") as src:
            a = src.read().reshape(12, 12, -1)[:, 1:4] / 10000.0     # B02, B03, B04
        res.append([int(x) for x in (a.min(1) > 0.25).sum(1)])
    return cid, res[0], res[1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--white-sample", type=int, default=300)
    args = ap.parse_args()
    from gfm4agri.chips.s2_monthly import REVISED

    parent = {c["chip_id"]: c for c in json.loads((PARENT / "manifest.json").read_text())["chips"]}
    new = {c["chip_id"]: c for c in json.loads((NEW / "manifest.json").read_text())["chips"]}
    ids = sorted(parent)
    missing = [c for c in ids if c not in new]
    linked = [c for c in ids if (NEW / "chips" / f"{c}_merged.tif").is_symlink()]
    # Compared as JSON, since the reports store the screening's tuples as lists.
    revised = json.loads(json.dumps(asdict(REVISED)))
    wrong_screening = [c for c in new if new[c]["imagery"].get("screening") != revised]
    present = [c for c in ids if c in new]
    with ProcessPoolExecutor(args.workers) as pool:
        eq = list(pool.map(labels_equal, present, chunksize=32))
        rng = np.random.default_rng(0)
        sample = sorted(rng.choice(present, size=min(args.white_sample, len(present)), replace=False))
        white = list(pool.map(white_per_month, sample, chunksize=4))
    fp = np.array([parent[c]["imagery"]["filled_share_per_month"] for c in present])
    fn = np.array([new[c]["imagery"]["filled_share_per_month"] for c in present])
    wp = np.array([w[1] for w in white]).sum(0)
    wn = np.array([w[2] for w in white]).sum(0)
    report = {
        "chips_parent": len(ids), "chips_new": len(new), "missing": missing, "composite_is_link": linked,
        "screening_not_revised": wrong_screening,
        "mask_differs": [c for c, m, _ in eq if not m], "parcels_differ": [c for c, _, p in eq if not p],
        "filled_share_mean": {m: [round(float(a), 4), round(float(b), 4)]
                              for m, a, b in zip(MONTHS, fp.mean(0), fn.mean(0))},
        "chips_over_half_filled": {m: [int(a), int(b)] for m, a, b in zip(MONTHS, (fp > 0.5).sum(0), (fn > 0.5).sum(0))},
        "white_pixels_sample": {"chips": len(sample), **{m: [int(a), int(b)] for m, a, b in zip(MONTHS, wp, wn)}},
        "order": "[parent, new]",
    }
    (HERE / "check.json").write_text(json.dumps(report, indent=1))
    print(json.dumps({k: (len(v) if isinstance(v, list) else v) for k, v in report.items()}, indent=1))


if __name__ == "__main__":
    main()
