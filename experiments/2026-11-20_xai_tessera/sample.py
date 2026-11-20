"""Step S: the test fields explained.

Every test chip's fields with 20 interior pixels or more are listed with their class, species, sowing form and the
class the model gave them (the stored ``predictions_test.npz``, majority of the parcel's labelled pixels, the core's
parcel rule). Chips are then taken greedily, the one that fills most of the classes' remaining quotas first (each
class weighted by the inverse of its test fields, so rare classes lead), until every class has ``quota`` fields or
``max_chips`` chips are taken. Of a chip, the fields of classes still short of their quota are explained.

Writes this folder's ``sample/fields.parquet`` (one row per explained field) and ``sample/chips.json`` (the
chips, in order), and every test field to ``results/<name>/EE_2021_rev/sample/all_test_fields.parquet``.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import chain


def _chip_rows(chips_dir: Path, chip: str, pred: np.ndarray) -> list[dict]:
    import rasterio

    with rasterio.open(chips_dir / f"{chip}.mask.tif") as m, rasterio.open(chips_dir / f"{chip}.parcels.tif") as p:
        mask, parcels = m.read(1).astype(np.int64), p.read(1).astype(np.int64)
    rows = []
    for f in chain.chip_fields(mask, parcels):
        lab = f["pixels"][mask.ravel()[f["pixels"]] >= 0]
        rows.append({"chip": chip, "field": f["field"], "cls": f["cls"], "pixels": len(f["pixels"]),
                     "interior": len(f["interior"]),
                     "pred_cls": int(np.bincount(pred.ravel()[lab].astype(np.int64), minlength=20).argmax())})
    return rows


def run(chips: Path, split_dir: Path, out: Path, cfg: dict, full_out: Path) -> pd.DataFrame:
    from gfm4agri.benchmark.segmentation_data import split_list

    out.mkdir(parents=True, exist_ok=True)
    full_out.mkdir(parents=True, exist_ok=True)
    stored = np.load(chain.CELL / "predictions_test.npz")
    pos = {c: i for i, c in enumerate(stored["ids"])}
    pred = stored["pred"]
    test = [c for c in split_list(chips, "test", split_dir).read_text().split() if c in pos]
    with ThreadPoolExecutor(16) as pool:
        parts = list(pool.map(lambda c: _chip_rows(chips / "chips", c, pred[pos[c]]), test))
    df = pd.DataFrame([r for p in parts for r in p])
    names = json.loads((chain.CELL / "results.json").read_text())["class_names"]
    species, sidx = chain.species_of_classes(names)
    df["class_name"] = [names[c] for c in df.cls]
    df["species"] = [species[sidx[c]] for c in df.cls]
    df["pred_species"] = [species[sidx[c]] for c in df.pred_cls]
    df["sowing"] = [n.split("_")[0] if n.split("_")[0] in ("winter", "spring") else "unspecified"
                    for n in df.class_name]
    df["correct_class"] = df.cls == df.pred_cls
    df["correct_species"] = df.species == df.pred_species
    df.to_parquet(full_out / "all_test_fields.parquet", index=False)

    quota, max_chips = int(cfg["quota"]), int(cfg["max_chips"])
    avail = df.cls.value_counts().reindex(range(len(names)), fill_value=0)
    weight = 1.0 / np.maximum(avail.to_numpy(), 1)
    need = np.minimum(quota, avail.to_numpy()).astype(float)
    per_chip = {c: g.cls.value_counts().reindex(range(len(names)), fill_value=0).to_numpy()
                for c, g in df.groupby("chip")}
    taken, chosen = [], []
    while need.sum() > 0 and len(taken) < max_chips:
        gain = {c: float((np.minimum(need, n) * weight).sum()) for c, n in per_chip.items() if c not in taken}
        best = max(gain, key=gain.get)
        if gain[best] <= 0:
            break
        g = df[(df.chip == best) & (need[df.cls] > 0)]
        chosen.append(g)
        need -= np.minimum(need, per_chip[best])
        taken.append(best)
    sel = pd.concat(chosen, ignore_index=True)
    sel.to_parquet(out / "fields.parquet", index=False)
    (out / "chips.json").write_text(json.dumps(taken, indent=0))
    counts = sel.groupby("class_name").agg(fields=("field", "size"), correct=("correct_species", "mean"))
    print(counts.to_string())
    print(f"{len(taken)} chips, {len(sel)} fields, {sel.chip.value_counts().max()} at most per chip; "
          f"classes short of the quota: {[names[k] for k in np.flatnonzero(need > 0)]}")
    return sel
