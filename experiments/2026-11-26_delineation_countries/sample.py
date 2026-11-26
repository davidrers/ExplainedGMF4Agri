"""The distributed sample: six test blocks per country, linked into the chip set ``EU18_2021_delin``.

Run in the Poetry environment (once)::

    poetry run python experiments/2026-11-26_delineation_countries/sample.py

**Which blocks.** The transfer plan's country test sets (``docs/research/transferability/samples.csv``, ``kind``
country, ``sample`` test) are whole blocks of 4 x 4 chips, drawn so that each set is spread over its country and
matches its crop mix, and never trained on by any transfer model. Of each country's test blocks with at least 8
chips, six are taken by farthest-point sampling on the block grid: the block nearest the country's centroid first,
then each time the block farthest from those already taken. 18 countries x 6 blocks; Estonia is left out.

**The chip set.** ``data/eurocrops_chips/EU18_2021_delin/`` holds only links (``chips/<id>_merged.tif``,
``_s1rtc.tif``, ``.mask.tif``, ``.parcels.tif`` into ``<CC>_2021_transfer``) and the files the core's loaders read:

* ``manifest.json``: ``EE_2021_rev``'s, with the chips (id, column, row, ``bounds_3035``) and the classes of the
  sample (the transfer list, 30 codes) in place of Estonia's. Its ``normalisation`` is kept: the Estonian training
  statistics, those our arms were trained under (the PRUE baseline reads them; the token-grid arms use their
  encoders' own pretraining statistics; TESSERA's are set in ``predict_arms.py``).
* ``s1_rtc.json``: ``EE_2021_rev``'s Sentinel-1 sidecar, for its Estonian statistics.
* ``splits/sample/``: every chip in ``test_data.txt``; the first chip also in ``training_data.txt`` and
  ``validation_data.txt``, because the core's cache builder encodes a training and a validation split (that chip
  is encoded in its eight symmetries for nothing; no model is trained).

Writes ``reference/sample_blocks.csv`` and ``reference/sample_chips.csv`` and the map ``figures/sample_map.png``.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

import common as C

PER_COUNTRY = 6
MIN_CHIPS = 8
PX, SIZE = 10.0, 224
TR = C.ROOT / "docs" / "research" / "transferability"


def farthest(xy: np.ndarray, k: int) -> list[int]:
    start = int(np.argmin(((xy - xy.mean(0)) ** 2).sum(1)))
    take = [start]
    d = ((xy - xy[start]) ** 2).sum(1)
    while len(take) < min(k, len(xy)):
        i = int(np.argmax(d))
        take.append(i)
        d = np.minimum(d, ((xy - xy[i]) ** 2).sum(1))
    return take


def main():
    s = pd.read_csv(TR / "samples.csv")
    s = s[(s.kind == "country") & (s["sample"] == "test") & s.domain.isin(C.COUNTRIES)]
    cells = pd.read_csv(TR / "export_cells.csv")
    blocks, rows = [], []
    for cc in C.COUNTRIES:
        b = s[(s.domain == cc) & (s.chips >= MIN_CHIPS)].reset_index(drop=True)
        pick = b.iloc[farthest(b[["bx", "by"]].to_numpy(float), PER_COUNTRY)]
        src = C.ROOT / "data" / "eurocrops_chips" / f"{cc}_2021_transfer" / "chips"
        for _, r in pick.iterrows():
            cc_cells = cells[(cells.country == cc) & (cells.bx == r.bx) & (cells.by == r.by)]
            n = 0
            for _, c in cc_cells.iterrows():
                cid = f"{cc}_{c.col:05d}_{c.row:05d}"
                if all((src / f"{cid}{suf}").exists() for suf in ("_merged.tif", "_s1rtc.tif", ".mask.tif",
                                                                  ".parcels.tif")):
                    rows.append({"chip_id": cid, "country": cc, "col": int(c.col), "row": int(c.row),
                                 "bx": int(r.bx), "by": int(r.by), "aez": int(r.aez)})
                    n += 1
            blocks.append({"country": cc, "bx": int(r.bx), "by": int(r.by), "aez": int(r.aez),
                           "chips_listed": int(r.chips), "chips_linked": n})
    blocks, chips = pd.DataFrame(blocks), pd.DataFrame(rows).sort_values("chip_id").reset_index(drop=True)
    (C.HERE / "reference").mkdir(exist_ok=True)
    blocks.to_csv(C.HERE / "reference" / "sample_blocks.csv", index=False)
    chips.to_csv(C.HERE / "reference" / "sample_chips.csv", index=False)

    # the chip set: links and the loaders' files
    if C.CHIPS.exists() and (C.CHIPS / "manifest.json").exists():
        old = json.loads((C.CHIPS / "manifest.json").read_text())
        if old.get("dataset") != f"{C.CHIPSET} (experiments/{C.NAME})":
            raise SystemExit(f"{C.CHIPS} exists and was not made by this script")
    C.CHIP_DIR.mkdir(parents=True, exist_ok=True)
    for cid, cc in zip(chips.chip_id, chips.country):
        src = C.ROOT / "data" / "eurocrops_chips" / f"{cc}_2021_transfer" / "chips"
        for suf in ("_merged.tif", "_s1rtc.tif", ".mask.tif", ".parcels.tif"):
            dst = C.CHIP_DIR / f"{cid}{suf}"
            if not dst.exists():
                os.symlink(os.path.relpath(src / f"{cid}{suf}", C.CHIP_DIR), dst)
    ee = C.ROOT / "data" / "eurocrops_chips" / "EE_2021_rev"
    man = json.loads((ee / "manifest.json").read_text())
    masks = json.loads((C.ROOT / "data" / "eurocrops_chips" / "AT_2021_transfer" / "masks.json").read_text())
    man.update({
        "dataset": f"{C.CHIPSET} (experiments/{C.NAME})",
        "purpose": "links to six test blocks per country of the transfer samples (18 countries, not Estonia), "
                   "for zero-shot delineation; normalisation kept from EE_2021_rev (the Estonian training "
                   "statistics our arms were trained under)",
        "country": "EU18", "year": 2021,
        "label_source": "the transfer masks of <CC>_2021_transfer (masks.json)",
        "class_scheme": "docs/research/transferability/crop_classes.csv (21 species, 9 groups)",
        "classes": [{"index": i, "hcat_code": "", "name": n} for i, n in enumerate(masks["class_list"])],
        "chips": [{"chip_id": r.chip_id, "col": int(r.col), "row": int(r.row),
                   "bounds_3035": [r.col * 2240.0, r.row * 2240.0, (r.col + 1) * 2240.0, (r.row + 1) * 2240.0]}
                  for r in chips.itertuples()],
        "derived_from": {"chip_sets": [f"{cc}_2021_transfer" for cc in C.COUNTRIES],
                         "normalisation_and_imagery_record": "EE_2021_rev", "experiment": f"experiments/{C.NAME}"},
    })
    for k in ("pixels_per_class_total",):
        man.pop(k, None)
    (C.CHIPS / "manifest.json").write_text(json.dumps(man, indent=1))
    (C.CHIPS / "s1_rtc.json").write_text((ee / "s1_rtc.json").read_text())
    C.SPLIT.mkdir(parents=True, exist_ok=True)
    ids = list(chips.chip_id)
    (C.SPLIT / "test_data.txt").write_text("\n".join(ids) + "\n")
    (C.SPLIT / "training_data.txt").write_text(ids[0] + "\n")
    (C.SPLIT / "validation_data.txt").write_text(ids[0] + "\n")
    (C.SPLIT / "split.json").write_text(json.dumps({
        "split": "sample", "config_hash": "none", "protocol": "every chip a test chip (zero-shot); the first chip "
        "also listed as training and validation for the cache builder only", "n_test": len(ids)}, indent=1))

    # the bounds must give each mask's grid
    import rasterio
    for r in chips.iloc[:: max(1, len(chips) // 12)].itertuples():
        with rasterio.open(C.CHIP_DIR / f"{r.chip_id}.mask.tif") as m:
            t = m.transform
            if abs(t.c - r.col * 2240.0) > 1e-6 or abs(t.f - (r.row + 1) * 2240.0) > 1e-6 or m.width != SIZE:
                raise SystemExit(f"{r.chip_id}: bounds do not give the mask's grid {t}")
    print(blocks.groupby("country").chips_linked.sum().to_string())
    print(f"{len(blocks)} blocks, {len(chips)} chips -> {C.CHIPS}")


if __name__ == "__main__":
    main()
