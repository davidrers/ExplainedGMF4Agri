"""Share of each crop group of the draft EU crop list in every candidate label file of 2021.

EuroCrops v2.0 (HCAT4): the area per original crop code of each 2021 file, read from the JRC server two columns at
a time (`v2_2021_code_areas.csv`, made with `--fetch`), joined to HCAT4 with the release's own mapping table
(`data/eurocrops/v2/cropcodemapping/eurocrops.csv`, from github.com/Martincccc/EuroCropsV2). EuroCrops v11 (HCAT3
names) for Latvia and Estonia, from the parquet files the pipeline uses, with areas computed in EPSG:3035.

    poetry run python docs/research/transferability/crop_groups.py [--fetch]

Writes `crop_groups.csv` (area % per file and group) and `crop_groups_parcels.csv` (parcels per file and group).
"""
from __future__ import annotations

import argparse
import re
from concurrent.futures import ThreadPoolExecutor

import geopandas as gpd
import pandas as pd

from gaez_domains import HERE, ROOT

V2 = "https://jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DRLL/EuroCropsV2/gpqtv202/{}_2021.parquet"
V2_FILES = ["at", "be2", "be3", "bg", "cz", "de4", "dea", "dk", "ee", "es", "fi", "fr", "ie", "iti1", "nl", "pt",
            "si", "sk"]
V11 = {"ee_v11": ROOT / "data/eurocrops/parquet/EE_2021.parquet", "lv_v11": ROOT / "data/eurocrops/parquet/LV_2021.parquet"}
MAPPING = ROOT / "data/eurocrops/v2/cropcodemapping/eurocrops.csv"
CODE_AREAS = HERE / "v2_2021_code_areas.csv"

# Draft EU list: HCAT names (v2 HCAT4 and v11 HCAT3 alike) -> group, first match wins. Cereals by species, since
# Spain and Portugal declare no sowing season; every grassland in one group, since the files split it differently.
GROUPS = [
    ("durum wheat", r"durum"),
    ("wheat", r"common_soft_wheat|(^|_)spelt"),
    ("barley", r"barley"),
    ("rye", r"(^|_)rye($|_)"),
    ("oats", r"(^|_)oats"),
    ("triticale", r"triticale"),
    ("maize", r"maize"),
    ("rice", r"(^|_)rice"),
    ("rapeseed", r"rapeseed"),
    ("sunflower", r"sunflower"),
    ("soy", r"soy"),
    ("potatoes", r"^potatoes$"),
    ("sugar beet", r"sugar_beet"),
    ("dry pulses", r"^(beans|peas|lupins|lentils|chickpeas|other_dry_pulses|legumes_dried_pulses_protein_crops)$"),
    ("forage legumes", r"alfalfa|clover|vetches|sainfoin|serradella|galega|melilot|birdsfoot|legumes_harvested_green"),
    ("grassland", r"grassland|pasture|meadow|poaceae|lolium|festuca|festulolium|timothy|cocksfoot|elymus"),
    ("fallow", r"fallow"),
    ("vegetables", r"vegetables|strawberr|cucurbit|brassica|cabbage|onion|carrot|lettuce|spinach|tomato|alliums"),
    ("orchards", r"orchard|apple|pear|plum|cherr|peach|apricot|berr|(^|_)nuts|almond|hazel|walnut|chestnut|citrus|orange"),
    ("olive", r"olive"),
    ("vineyards", r"vineyard|vine|grape"),
]


def group_of(name: str) -> str:
    for g, pattern in GROUPS:
        if isinstance(name, str) and re.search(pattern, name):
            return g
    return "other"


def fetch() -> None:
    import fsspec
    import pyarrow.parquet as pq

    def one(nuts: str) -> pd.DataFrame:
        with fsspec.filesystem("http").open(V2.format(nuts), block_size=2**22) as fh:
            df = pq.read_table(fh, columns=["original_code", "area_ha"]).to_pandas()
        g = df.groupby("original_code").agg(area_ha=("area_ha", "sum"), parcels=("area_ha", "size")).reset_index()
        g.insert(0, "nuts", nuts)
        return g
    with ThreadPoolExecutor(6) as ex:
        pd.concat(ex.map(one, V2_FILES)).to_csv(CODE_AREAS, index=False)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fetch", action="store_true", help="re-read the code areas from the JRC server")
    if ap.parse_args().fetch:
        fetch()
    codes = pd.read_csv(CODE_AREAS, dtype={"original_code": str})
    mapping = pd.read_csv(MAPPING, dtype=str)[["nuts", "original_code", "hcat4_name"]]
    v2 = codes.merge(mapping, on=["nuts", "original_code"], how="left").rename(columns={"nuts": "file",
                                                                                      "hcat4_name": "hcat"})
    parts = [v2[["file", "hcat", "area_ha", "parcels"]]]
    for name, path in V11.items():
        g = gpd.read_parquet(path, columns=["EC_hcat_n", "geometry"]).to_crs(3035)
        g["area_ha"] = g.geometry.area / 1e4
        t = g.groupby("EC_hcat_n").agg(area_ha=("area_ha", "sum"), parcels=("area_ha", "size")).reset_index()
        parts.append(t.rename(columns={"EC_hcat_n": "hcat"}).assign(file=name))
    allc = pd.concat(parts, ignore_index=True)
    allc["group"] = allc.hcat.map(group_of)
    area = allc.pivot_table(index="file", columns="group", values="area_ha", aggfunc="sum", fill_value=0)
    pct = area.div(area.sum(axis=1), axis=0) * 100
    order = [g for g, _ in GROUPS] + ["other"]
    pct[order].round(1).to_csv(HERE / "crop_groups.csv")
    allc.pivot_table(index="file", columns="group", values="parcels", aggfunc="sum",
                     fill_value=0)[order].to_csv(HERE / "crop_groups_parcels.csv")
    other = allc[allc.group == "other"].groupby("hcat").area_ha.sum().sort_values(ascending=False)
    pd.set_option("display.width", 250)
    print(pct[order].round(1).to_string())
    print("\nlargest names left in 'other' (ha):", other.head(25).round(0).to_dict())


if __name__ == "__main__":
    main()
