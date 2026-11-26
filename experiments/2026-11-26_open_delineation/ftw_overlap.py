"""Which of our chips Fields of The World's Estonia covers, and in which of its splits: ``reference/ftw_overlap.csv``.

Run in the Poetry environment::

    poetry run python experiments/2026-11-26_open_delineation/ftw_overlap.py

Fields of The World includes Estonia 2021 (``data_config_estonia.json``: windows April to June and July to September
2021, two grids, ``g10`` in central and ``g15`` in south-eastern Estonia), from the same declarations as EuroCrops,
so its models may have trained on our test parcels. Its chip footprints and splits are read from
``chips_estonia.parquet`` (source.coop, ``kerner-lab/fields-of-the-world/estonia``), downloaded once into
``data/external_models/ftw/``. Every one of our chips gets the FTW splits of the FTW chips it intersects (``none``:
no FTW chip touches it). The scores' ``test_clean`` subset is the test chips with ``none``.
"""

from __future__ import annotations

import json
import urllib.request

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import common as C

BASE = "https://data.source.coop/kerner-lab/fields-of-the-world/estonia/"
FILES = ("chips_estonia.parquet", "data_config_estonia.json")


def main():
    d = C.WEIGHTS / "ftw"
    d.mkdir(parents=True, exist_ok=True)
    for f in FILES:
        if not (d / f).exists():
            urllib.request.urlretrieve(BASE + f, d / f)
    m = json.loads((C.CHIPS / "manifest.json").read_text())
    ours = gpd.GeoDataFrame({"chip_id": [c["chip_id"] for c in m["chips"]]},
                            geometry=[box(*c["bounds_3035"]) for c in m["chips"]], crs=3035)
    split = {}
    for s, f in (("train", "training_data.txt"), ("val", "validation_data.txt"), ("test", "test_data.txt")):
        for line in open(C.SPLIT / f):
            split[line.strip()] = s
    ours["split"] = ours.chip_id.map(split).fillna("none")
    ftw = gpd.read_parquet(d / "chips_estonia.parquet").to_crs(3035)
    j = gpd.sjoin(ours, ftw[["split", "geometry"]].rename(columns={"split": "ftw"}), how="left",
                  predicate="intersects")
    agg = j.groupby("chip_id").ftw.agg(lambda s: ",".join(sorted(set(s.dropna()))))
    ours["ftw"] = ours.chip_id.map(agg).replace("", "none")
    out = C.HERE / "reference" / "ftw_overlap.csv"
    ours.drop(columns="geometry").to_csv(out, index=False)
    tab = pd.crosstab(ours.split, ours.ftw)
    print(tab)
    lines = ["| our split | " + " | ".join(tab.columns) + " |", "|---" * (len(tab.columns) + 1) + "|"]
    lines += [f"| {i} | " + " | ".join(str(v) for v in r) + " |" for i, r in tab.iterrows()]
    (C.HERE / "reference" / "ftw_overlap_table.md").write_text(
        "Our chips (rows: our split) by the Fields of The World Estonia splits of the FTW chips they touch\n\n"
        + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
