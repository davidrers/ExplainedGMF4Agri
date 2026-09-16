"""HCAT taxonomy helpers for EuroCropsML.

EuroCropsML labels are the EuroCrops ``EC_hcat_c`` codes (HCAT version 2). The
``eurocropsml`` distribution ships only the numeric codes, so the human readable
names are taken from the EuroCrops repository (maja601/EuroCrops, ``hcat_core/``),
cached under ``results/eda/cache/hcat/``.

The HCAT code is a ten digit hierarchical path. A node's depth is given by the
length of the informative prefix; the boundaries are 1, 2, 4, 6, 8 and 10 digits,
so truncating a code to one of those lengths and re-padding with zeros climbs the
tree. For example 3301010101 (winter common soft wheat) rolls up to 33010101
(common soft wheat), 330101 (cereal), 3301 (arable crops), 33 (crop type).
"""

from __future__ import annotations

import urllib.request
from functools import lru_cache
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache" / "hcat"
BASE_URL = "https://raw.githubusercontent.com/maja601/EuroCrops/main"
TAXONOMY_FILES = {
    "HCAT2.csv": f"{BASE_URL}/hcat_core/HCAT2.csv",
    "HCAT3.csv": f"{BASE_URL}/hcat_core/HCAT3.csv",
    "ee_2021.csv": f"{BASE_URL}/csvs/country_mappings/ee_2021.csv",
    "lv_2021.csv": f"{BASE_URL}/csvs/country_mappings/lv_2021.csv",
    "pt_2021.csv": f"{BASE_URL}/csvs/country_mappings/pt_2021.csv",
}

#: Informative prefix length of each hierarchical level.
LEVEL_PREFIX = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8, 6: 10}


def _ensure(name: str) -> Path:
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / name
    if not p.exists():
        urllib.request.urlretrieve(TAXONOMY_FILES[name], p)
    return p


@lru_cache(maxsize=None)
def taxonomy() -> pd.DataFrame:
    """HCAT2 taxonomy with derived hierarchical level, indexed by code (string)."""
    df = pd.read_csv(_ensure("HCAT2.csv"), dtype=str)
    df = df.rename(columns={"HCAT2_name": "hcat_name", "HCAT2_code": "hcat"})
    df["hcat"] = df["hcat"].str.strip()
    df["level"] = df["hcat"].map(hcat_level)
    return df.drop_duplicates("hcat").set_index("hcat")


@lru_cache(maxsize=None)
def country_mapping() -> pd.DataFrame:
    """National crop declarations mapped onto HCAT2, for Estonia, Latvia and Portugal."""
    frames = []
    for code, name in (("Estonia", "ee_2021.csv"), ("Latvia", "lv_2021.csv"), ("Portugal", "pt_2021.csv")):
        d = pd.read_csv(_ensure(name), dtype=str)
        d["country"] = code
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    return out.rename(columns={"HCAT2_code": "hcat", "HCAT2_name": "hcat_name"})


def hcat_level(code: str) -> int:
    """Hierarchical depth of an HCAT code (1 coarsest, 6 finest)."""
    informative = len(str(code).rstrip("0")) or 1
    for lvl, width in LEVEL_PREFIX.items():
        if informative <= width:
            return lvl
    return 6


def roll_up(code: str, level: int) -> str:
    """Truncate an HCAT code to the given hierarchical level, re-padded to ten digits."""
    width = LEVEL_PREFIX[level]
    s = str(code)
    return (s[:width]).ljust(10, "0")


def name_of(code: str, fallback: str | None = None) -> str:
    """Human readable crop name for an HCAT code."""
    tax = taxonomy()
    s = str(code)
    if s in tax.index:
        return str(tax.at[s, "hcat_name"])
    return fallback if fallback is not None else f"unknown_{s}"


def pretty(code: str) -> str:
    """Title-cased crop name suitable for figure labels, with the code appended."""
    return f"{name_of(code).replace('_', ' ')} ({code})"


def add_names(df: pd.DataFrame, col: str = "hcat") -> pd.DataFrame:
    """Attach ``hcat_name``, ``crop_name`` and ``level`` columns to a frame of HCAT codes."""
    out = df.copy()
    out["hcat_name"] = out[col].map(lambda c: name_of(c))
    out["crop_name"] = out["hcat_name"].str.replace("_", " ")
    out["level"] = out[col].map(hcat_level)
    return out
