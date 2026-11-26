"""Which sample chips Fields of The World covers, and in which of its splits: ``reference/ftw_overlap.csv``.

Run in the Poetry environment::

    poetry run python experiments/2026-11-26_delineation_countries/ftw_overlap.py

Fields of The World covers 14 of the 18 sampled countries (all but Bulgaria, Czechia, Ireland and Italy). Its chip
footprints and splits (``chips_<country>.parquet``) and window configurations (``data_config_<country>.json``) are
read from source.coop (``kerner-lab/fields-of-the-world/<country>/``) into ``data/external_models/ftw/countries/``.
Every sample chip gets the FTW splits of the FTW chips it intersects (``none``: no FTW chip touches it; always so in
the four countries FTW does not cover). The scores' ``ftw_clean`` subset is the chips with ``none``.
"""

from __future__ import annotations

import json
import urllib.request

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

import common as C
from predict_ftw_own import NAMES

BASE = "https://data.source.coop/kerner-lab/fields-of-the-world/{c}/{f}"


def main():
    d = C.WEIGHTS / "ftw" / "countries"
    d.mkdir(parents=True, exist_ok=True)
    for c in NAMES.values():
        for f in (f"chips_{c}.parquet", f"data_config_{c}.json"):
            if not (d / f).exists():
                urllib.request.urlretrieve(BASE.format(c=c, f=f), d / f)
    man = json.loads((C.CHIPS / "manifest.json").read_text())
    ours = gpd.GeoDataFrame({"chip_id": [c["chip_id"] for c in man["chips"]]},
                            geometry=[box(*c["bounds_3035"]) for c in man["chips"]], crs=3035)
    ours["country"] = ours.chip_id.str[:2]
    ftw = pd.concat([gpd.read_parquet(d / f"chips_{c}.parquet").to_crs(3035).assign(ftw_country=cc)
                     for cc, c in NAMES.items()], ignore_index=True)
    j = gpd.sjoin(ours, gpd.GeoDataFrame(ftw[["split", "geometry"]].rename(columns={"split": "ftw"}), crs=3035),
                  how="left", predicate="intersects")
    agg = j.groupby("chip_id").ftw.agg(lambda s: ",".join(sorted(set(s.dropna()))))
    ours["ftw"] = ours.chip_id.map(agg).replace("", "none").fillna("none")
    ours.drop(columns="geometry").to_csv(C.HERE / "reference" / "ftw_overlap.csv", index=False)
    tab = pd.crosstab(ours.country, ours.ftw != "none").rename(columns={False: "untouched", True: "touched"})
    print(tab.to_string())
    lines = ["| country | chips no FTW chip touches | chips touching an FTW chip |", "|---|---|---|"]
    lines += [f"| {cc} | {int(r.get('untouched', 0))} | {int(r.get('touched', 0))} |" for cc, r in tab.iterrows()]
    (C.HERE / "reference" / "ftw_overlap_table.md").write_text(
        "Sample chips by whether a Fields of The World chip touches them\n\n" + "\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
