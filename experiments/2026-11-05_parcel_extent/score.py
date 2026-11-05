"""Score every TESSERA cell of this experiment in every extent, on the hub, where the WorldCover maps live.

* ``tessera_v2_medium`` (frozen): the delineation experiment's stored cells,
  ``results/2026-10-18_parcel_delineation/<set>/tessera_v2_medium/P<k>_draw0_seed0/predictions_test.npz``, not
  refitted, not written to;
* ``tessera_v2_medium_ft_bnd`` (fine-tuned): this experiment's cells, pulled back from the cluster.

For each cell, ``results/2026-11-05_parcel_extent/<set>/<arm>/<cell>/extent_scores.json`` (``panoptic_x``'s scores
for the extents ``oracle``, ``worldcover``, ``worldcover_30_40_90`` and ``chip``, the source of the predictions, and
the cell's pixel and parcel Macro-F1), then ``extent_summary.csv`` with the headline numbers. A cell already scored
is skipped unless ``--force``.

    poetry run python experiments/2026-11-05_parcel_extent/score.py --chips EE_2021_rev
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src")]

NAME = "2026-11-05_parcel_extent"
EXTENTS = ("oracle", "worldcover", "worldcover_30_40_90", "chip")
#: arm -> where its cells are.
SOURCES = {"tessera_v2_medium": "2026-10-18_parcel_delineation", "tessera_v2_medium_ft_bnd": NAME}
HEAD = ("extent", "semantic", "boundary@0.3", "boundary@0.5", "boundary@0.7", "combined@0.5")


def cells(chips: str):
    for arm, exp in SOURCES.items():
        for f in sorted((REPO / "results" / exp / chips / arm).glob("P*_draw*_seed*/predictions_test.npz")):
            if (f.parent / "results.json").exists():
                yield arm, f.parent


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chips", required=True)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    import pandas as pd

    import panoptic_x as px

    chips_dir = REPO / "data/eurocrops_chips" / a.chips / "chips"
    wc = REPO / "data/worldcover" / a.chips.replace("_mini", "") / "worldcover.npz"
    out_root = REPO / "results" / NAME / a.chips
    rows = []
    for arm, cell in cells(a.chips):
        dest = out_root / arm / cell.name / "extent_scores.json"
        r = json.loads((cell / "results.json").read_text())
        if dest.exists() and not a.force:
            s = json.loads(dest.read_text())
        else:
            print(f"scoring {arm}/{cell.name} from {cell.relative_to(REPO)}", flush=True)
            s = px.score_predictions(cell / "predictions_test.npz", chips_dir, r["class_names"], wc,
                                     extents=EXTENTS, workers=a.workers)
            s["source"] = {"cell": str(cell.relative_to(REPO)), "git_commit": r.get("provenance", {}).get("git_commit"),
                           "best_checkpoint": r.get("best_checkpoint"),
                           "pixel_macro_f1": r.get("test_metrics", {}).get("test/F1_Score"),
                           "parcel_macro_f1": r.get("parcel_metrics", {}).get("macro_f1")}
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f"extent_scores.json.{os.getpid()}")
            tmp.write_text(json.dumps(s, indent=1))
            tmp.replace(dest)
        v = s["variants"]
        base = {"arm": arm, "pct": r["label_budget"]["pct"],
                "pixel_macro_f1": s["source"]["pixel_macro_f1"], "boundary_f1_E": s.get("boundary_pixels", {}).get("f1"),
                "oracle_rq": v["oracle"]["rq"]}
        for x in EXTENTS:
            row = {**base, "extent": x, "true_px_inside": s["extents"][x]["true_pixels_inside"],
                   "background_share": s["extents"][x]["background_share_of_extent"]}
            for name in HEAD:
                q = v.get(f"{x}/{name}")
                if q is None:
                    continue
                row[f"pq_{name}"], row[f"pq_agn_{name}"] = q["pq"], q["pq_agnostic"]
                row[f"parcels_{name}"] = q["predicted_parcels"]
                row[f"fp_{name}"], row[f"dropped_{name}"] = q["fp_agnostic"], q["dropped_void"]
            rows.append(row)
    df = pd.DataFrame(rows).sort_values(["arm", "pct", "extent"], ascending=[True, False, True])
    df.to_csv(out_root / "extent_summary.csv", index=False)
    cols = ["arm", "pct", "extent", "pq_agn_extent", "pq_agn_semantic", "pq_agn_boundary@0.5", "pq_boundary@0.5",
            "pq_agn_boundary@0.3", "parcels_boundary@0.5", "fp_boundary@0.5"]
    with pd.option_context("display.width", 200, "display.max_columns", 20, "display.precision", 3):
        print(df[[c for c in cols if c in df]].to_string(index=False))


if __name__ == "__main__":
    main()
