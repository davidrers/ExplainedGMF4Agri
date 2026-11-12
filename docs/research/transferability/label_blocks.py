"""The 2021 labels of every candidate country, summed per block and per block and zone: what the draw works from.

Every parcel is placed by its centroid in a chip cell of the EPSG:3035 grid (2,240 m) and in a block (4 x 4 cells,
8.96 km). A chip exists wherever a labelled parcel lies, so the cells holding a parcel centroid stand for the chips;
the count is checked against Estonia's and Latvia's real chip sets. Each cell gets its GAEZ v5 AEZ33 class as GAEZ
maps it, and the nearest climatic class as its climate tag (`zone_pools.py`).

With the 33 classes as GAEZ maps them, a block usually holds two or three classes (the special classes come in 1 km
patches), so a zone is made of cells, not of whole blocks: the unit of a zone sample is one zone's cells inside one
block.

Label sources, all 2021: EuroCrops v2.0 (HCAT4, `data/eurocrops/v2/gpqtv202/`) wherever a region has it; v11
(HCAT3) for Estonia (the masks in use), Latvia, Lithuania, Sweden and Lower Saxony.

    poetry run python docs/research/transferability/label_blocks.py [--countries ES ...]

With `--countries`, only those countries' label files are read again; the others come from the saved parcel table.

Writes next to this file `label_blocks.parquet` (one row per country and block: chips, the class most of its cells
have, how many classes it holds, climate tag, area and parcels per crop group of `crop_groups.py`),
`label_block_zones.parquet` (the same per country, block and AEZ class) and `label_blocks_hcat.parquet` (area and
parcels per country, block, AEZ class and HCAT class, for the choice of crop classes); and the parcel table itself,
`data/eurocrops/label_parcels_2021.parquet`, so that later choices need no second pass over the label files.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor

import geopandas as gpd
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import shapely

from crop_groups import GROUPS, group_of
from download_estimate import grid
from gaez_domains import DOMAINS, HERE, ROOT
from zone_pools import aez_on_grid

CELL = 2240.0
V2 = ROOT / "data/eurocrops/v2/gpqtv202"
MAPPING = ROOT / "data/eurocrops/v2/cropcodemapping/eurocrops.csv"
V11 = ROOT / "data/eurocrops/parquet"
V11_EXTRA = ROOT / "data/eurocrops/v11_extra"
PARCELS = ROOT / "data/eurocrops/label_parcels_2021.parquet"
# country -> label files: ("v2", nuts) or ("v11", path)
SOURCES = {
    "AT": [("v2", "at")], "BE": [("v2", "be2"), ("v2", "be3")], "BG": [("v2", "bg")], "CZ": [("v2", "cz")],
    "DE": [("v2", "de4"), ("v2", "dea"), ("v11", V11_EXTRA / "DE_LS_2021")], "DK": [("v2", "dk")],
    "EE": [("v11", V11 / "EE_2021.parquet")], "ES": [("v2", "es")], "FI": [("v2", "fi")], "FR": [("v2", "fr")],
    "IE": [("v2", "ie")], "IT": [("v2", "iti1")], "LT": [("v11", V11 / "LT_2021.parquet")],
    "LV": [("v11", V11 / "LV_2021.parquet")], "NL": [("v2", "nl")], "PT": [("v2", "pt")],
    "SE": [("v11", V11_EXTRA / "SE_2021")], "SI": [("v2", "si")], "SK": [("v2", "sk")],
}


def parcels_v2(country: str, nuts: str, row_group: int) -> pd.DataFrame:
    """Centroid (EPSG:3035), area and HCAT4 name of the parcels of one row group of one v2 file."""
    mapping = pd.read_csv(MAPPING, dtype=str)
    mapping = mapping[mapping.nuts == nuts].set_index("original_code").hcat4_name
    b = pq.ParquetFile(V2 / f"{nuts}_2021.parquet").read_row_group(
        row_group, columns=["original_code", "area_ha", "geometry"]).to_pandas()
    centroid = shapely.centroid(shapely.from_wkb(b.geometry.to_numpy(), on_invalid="ignore"))
    centroid[shapely.is_empty(centroid)] = None
    df = pd.DataFrame({"x": shapely.get_x(centroid), "y": shapely.get_y(centroid),
                       "area_ha": b.area_ha.to_numpy(),
                       "hcat": b.original_code.astype(str).map(mapping).to_numpy()})   # Spain stores numbers
    return df.dropna(subset=["x", "y"]).assign(country=country)    # empty or missing geometries


def parcels_v11(country: str, path) -> pd.DataFrame:
    """The same for a v11 layer (a GeoParquet, or the extracted archive's vector file), areas computed in 3035."""
    if path.suffix == ".parquet":
        g = gpd.read_parquet(path, columns=["EC_hcat_n", "geometry"])
    else:
        layer = next(p for ext in ("*.gpkg", "*.shp") for p in sorted(path.rglob(ext)))
        g = gpd.read_file(layer, columns=["EC_hcat_n"])
    g = g[g.geometry.notna() & ~g.geometry.is_empty].to_crs(3035)
    c = g.geometry.centroid
    return pd.DataFrame({"x": c.x.to_numpy(), "y": c.y.to_numpy(), "area_ha": (g.geometry.area / 1e4).to_numpy(),
                         "hcat": g.EC_hcat_n.to_numpy(), "country": country})


def task(args) -> pd.DataFrame:
    kind, *rest = args
    return parcels_v2(*rest) if kind == "v2" else parcels_v11(*rest)


def tasks(countries) -> list[tuple]:
    out = []
    for country, sources in SOURCES.items():
        if country not in countries:
            continue
        for kind, src in sources:
            if kind == "v2":
                n = pq.ParquetFile(V2 / f"{src}_2021.parquet").metadata.num_row_groups
                out += [("v2", country, src, i) for i in range(n)]
            else:
                out.append(("v11", country, src))
    return out


def summed(parcels: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    order = [g for g, _ in GROUPS] + ["other"]
    area = parcels.pivot_table(index=keys, columns="group", values="area_ha", aggfunc="sum",
                               fill_value=0).reindex(columns=order, fill_value=0)
    count = parcels.pivot_table(index=keys, columns="group", values="area_ha", aggfunc="size",
                                fill_value=0).reindex(columns=order, fill_value=0)
    return area.add_prefix("ha_").join(count.add_prefix("n_"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--countries", nargs="+", choices=sorted(SOURCES), help="read only these again")
    again = ap.parse_args().countries
    transform, shape, ids, (col0, row_top) = grid()
    raw = aez_on_grid(transform, shape, fill=False)
    filled = aez_on_grid(transform, shape, fill=True)
    names = list(DOMAINS)

    with ProcessPoolExecutor(24) as ex:
        parcels = pd.concat(ex.map(task, tasks(again or names)), ignore_index=True)
    parcels["gx"] = np.floor(parcels.x / CELL).astype(int)
    parcels["gy"] = np.floor(parcels.y / CELL).astype(int)
    r, c = row_top - 1 - parcels.gy.to_numpy(), parcels.gx.to_numpy() - col0
    inside = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    dom = np.zeros(len(parcels), dtype=int)
    dom[inside] = ids[r[inside], c[inside]]
    own = dom == parcels.country.map({d: k for k, d in enumerate(names, start=1)}).to_numpy()
    print(f"parcels {len(parcels):,}; outside their country's domain (overseas, islands, border slivers): "
          f"{(~own).sum():,}")
    parcels = parcels[own].copy()
    r, c = row_top - 1 - parcels.gy.to_numpy(), parcels.gx.to_numpy() - col0
    parcels["aez"], parcels["climate"] = raw[r, c], filled[r, c]
    if again:
        kept = pd.read_parquet(PARCELS)
        parcels = pd.concat([kept[~kept.country.isin(again)], parcels.drop(columns=["x", "y"])], ignore_index=True)
    parcels["group"] = parcels.hcat.map(group_of)
    parcels["bx"], parcels["by"] = parcels.gx // 4, parcels.gy // 4
    parcels[["country", "gx", "gy", "aez", "climate", "hcat", "area_ha"]].astype(
        {"area_ha": "float32", "aez": "int16", "climate": "int16"}).to_parquet(PARCELS, index=False)

    cells = parcels.drop_duplicates(["country", "gx", "gy"])
    blocks = (cells.groupby(["country", "bx", "by"])
              .agg(chips=("aez", "size"), aez=("aez", lambda s: s.mode().iloc[0]), classes=("aez", "nunique"),
                   climate=("climate", lambda s: s.mode().iloc[0]))
              .reset_index())
    blocks = blocks.set_index(["country", "bx", "by"]).join(summed(parcels, ["country", "bx", "by"])).reset_index()
    blocks.to_parquet(HERE / "label_blocks.parquet", index=False)

    zones = (cells.groupby(["country", "bx", "by", "aez"])
             .agg(chips=("gx", "size"), climate=("climate", lambda s: s.mode().iloc[0])).reset_index())
    keys = ["country", "bx", "by", "aez"]
    zones = zones.set_index(keys).join(summed(parcels, keys)).reset_index()
    zones.to_parquet(HERE / "label_block_zones.parquet", index=False)
    (parcels.groupby(keys + ["hcat"]).agg(area_ha=("area_ha", "sum"), parcels=("area_ha", "size"))
     .reset_index().to_parquet(HERE / "label_blocks_hcat.parquet", index=False))

    summary = blocks.groupby("country").agg(chips=("chips", "sum"), blocks=("chips", "size"),
                                            single_class=("classes", lambda s: (s == 1).mean()))
    print(summary.round(2).to_string())
    print("check against the real chip sets: EE 7,402, LV 11,500")


if __name__ == "__main__":
    main()
