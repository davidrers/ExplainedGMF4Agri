"""The fields explained: only test fields the model predicts correctly, at most ``per_chip`` per species and chip.

    poetry run python experiments/2026-11-26_xai_tessera_blocks/sample.py

Every test field with 20 interior pixels or more is listed with the species the stored predictions give it
(``predictions_test.npz``, majority of the parcel's labelled pixels, the core's parcel rule), as
``experiments/2026-11-20_xai_tessera/sample.py`` lists them. Only fields whose predicted species is their species are
kept. Chips are taken greedily, the one that fills most of the species' remaining quotas first (each species weighted
by the inverse of its correct fields, so rare species lead), each chip giving at most ``per_chip`` fields per species
(drawn at random, seed 0), until every species has ``quota`` fields or no chip adds any.

Writes ``sample/fields.parquet`` and ``sample/chips.json``.
"""

from __future__ import annotations

import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parents[1] / "src")]
import chain  # noqa: E402

CHIPS = chain.REPO / "data/eurocrops_chips/EE_2021_rev"


def chip_rows(chip: str, pred: np.ndarray) -> list[dict]:
    import rasterio

    with rasterio.open(CHIPS / "chips" / f"{chip}.mask.tif") as m, rasterio.open(CHIPS / "chips" / f"{chip}.parcels.tif") as p:
        mask, parcels = m.read(1).astype(np.int64), p.read(1).astype(np.int64)
    rows = []
    for f in chain.chip_fields(mask, parcels):
        lab = f["pixels"][mask.ravel()[f["pixels"]] >= 0]
        rows.append({"chip": chip, "field": f["field"], "cls": f["cls"], "pixels": len(f["pixels"]),
                     "interior": len(f["interior"]),
                     "pred_cls": int(np.bincount(pred.ravel()[lab].astype(np.int64), minlength=20).argmax())})
    return rows


def main() -> None:
    from gfm4agri.benchmark.segmentation_data import split_list
    from gfm4agri.pipeline.config import resolve_split

    cfg = yaml.safe_load((HERE / "experiment.yaml").read_text())["sample"]
    quota, per_chip = int(cfg["quota"]), int(cfg["per_chip"])
    stored = np.load(chain.CELL / "predictions_test.npz")
    pos = {c: i for i, c in enumerate(stored["ids"])}
    test = [c for c in split_list(CHIPS, "test", resolve_split(CHIPS, None)).read_text().split() if c in pos]
    with ThreadPoolExecutor(16) as pool:
        parts = list(pool.map(lambda c: chip_rows(c, stored["pred"][pos[c]]), test))
    df = pd.DataFrame([r for p in parts for r in p])
    names = json.loads((chain.CELL / "results.json").read_text())["class_names"]
    species, sidx = chain.species_of_classes(names)
    df["class_name"] = [names[c] for c in df.cls]
    df["species"] = [species[sidx[c]] for c in df.cls]
    df["pred_species"] = [species[sidx[c]] for c in df.pred_cls]
    ok = df[df.species == df.pred_species].copy()
    rng = np.random.default_rng(0)
    ok["draw"] = rng.random(len(ok))
    avail = ok.species.value_counts()
    need = {s: min(quota, int(avail[s])) for s in avail.index}
    weight = {s: 1.0 / avail[s] for s in avail.index}
    by_chip = {c: g for c, g in ok.groupby("chip")}
    taken, chosen = [], []
    while sum(need.values()) > 0:
        def gain(g):
            vc = g.species.value_counts()
            return sum(min(need[s], per_chip, int(k)) * weight[s] for s, k in vc.items())
        cand = {c: gain(g) for c, g in by_chip.items() if c not in taken}
        if not cand:
            break
        best = max(cand, key=cand.get)
        if cand[best] <= 0:
            break
        g = by_chip[best]
        for s, gs in g.groupby("species"):
            k = min(need[s], per_chip, len(gs))
            if k > 0:
                chosen.append(gs.nsmallest(k, "draw"))
                need[s] -= k
        taken.append(best)
    sel = pd.concat(chosen, ignore_index=True).drop(columns="draw")
    out = HERE / "sample"
    out.mkdir(exist_ok=True)
    sel.to_parquet(out / "fields.parquet", index=False)
    (out / "chips.json").write_text(json.dumps(taken, indent=0))
    print(sel.species.value_counts().to_string())
    print(f"{len(taken)} chips, {len(sel)} fields, at most {sel.chip.value_counts().max()} per chip; "
          f"short of the quota: {[s for s, k in need.items() if k > 0]}")


if __name__ == "__main__":
    main()
