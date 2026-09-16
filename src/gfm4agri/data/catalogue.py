"""Loading of the EuroCropsML parcel catalogue.

The catalogue is one row per parcel and is produced by
``results/eda/_build_catalogue.py`` into ``data/catalogue_parcels_full.parquet``.
This module isolates every assumption Phase 1 makes about that file, so that a
change of schema has a single point of repair.

Canonical schema (columns consumed by Phase 1)
----------------------------------------------
======================  =========  ==================================================
column                  dtype      meaning
======================  =========  ==================================================
``path``                str        absolute path to the parcel ``.npz``
``country``             str        ``Estonia`` | ``Latvia`` | ``Portugal``
``nuts``                str        NUTS code prefix of the filename, e.g. ``EE008``
``parcel_id``           str        parcel identifier, unique within a country
``hcat``                str        HCAT crop code, as a string of digits
``n_timesteps``         int        number of cloud-free Sentinel-2 acquisitions
``lat``, ``lon``        float      parcel centroid, WGS-84 decimal degrees
======================  =========  ==================================================

Optional columns (``first_date``, ``last_date``, ``m01`` .. ``m12``) are carried
through untouched when present.  A ``uid`` column is added by
:func:`load_catalogue`: it is the ``.npz`` stem, which is globally unique and
stable across rebuilds.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "CatalogueError",
    "CatalogueSummary",
    "COUNTRY_CODE",
    "CODE_COUNTRY",
    "REQUIRED_COLUMNS",
    "default_catalogue_path",
    "load_catalogue",
    "catalogue_from_filenames",
    "validate_catalogue",
]


class CatalogueError(RuntimeError):
    """Raised when the parcel catalogue is missing or does not match the schema."""


COUNTRY_CODE: dict[str, str] = {"Estonia": "EE", "Latvia": "LV", "Portugal": "PT"}
CODE_COUNTRY: dict[str, str] = {v: k for k, v in COUNTRY_CODE.items()}

REQUIRED_COLUMNS: tuple[str, ...] = (
    "country",
    "nuts",
    "parcel_id",
    "hcat",
    "n_timesteps",
    "lat",
    "lon",
)

_FNAME_RE = re.compile(r"^(?P<nuts>[A-Z]{2}[A-Z0-9]*)_(?P<pid>\d+)_(?P<cls>\d+)\.npz$")


def default_catalogue_path(repo_root: Path | str | None = None) -> Path:
    """Return the conventional location of the full parcel catalogue."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    return Path(repo_root) / "data" / "catalogue_parcels_full.parquet"


def validate_catalogue(df: pd.DataFrame) -> pd.DataFrame:
    """Normalise dtypes and assert that the required columns are present.

    Args:
        df: A candidate parcel catalogue.

    Returns:
        A normalised copy carrying additional ``uid`` and ``country_code``
        columns, sorted by ``uid``.

    Raises:
        CatalogueError: If a required column is missing, the table is empty, a
            country label is unknown, or parcel identifiers are not unique.
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise CatalogueError(f"parcel catalogue is missing columns: {missing}")
    if len(df) == 0:
        raise CatalogueError("parcel catalogue is empty")

    out = df.copy()
    for col in ("country", "nuts", "parcel_id", "hcat"):
        out[col] = out[col].astype("string").astype(object).astype(str)
    out["lat"] = out["lat"].astype("float64")
    out["lon"] = out["lon"].astype("float64")
    out["n_timesteps"] = out["n_timesteps"].astype("int32")

    unknown = sorted(set(out["country"]) - set(COUNTRY_CODE))
    if unknown:
        raise CatalogueError(f"unknown country labels in catalogue: {unknown}")
    out["country_code"] = out["country"].map(COUNTRY_CODE).astype(str)

    out["uid"] = out["nuts"] + "_" + out["parcel_id"] + "_" + out["hcat"]
    if out["uid"].duplicated().any():
        n_dup = int(out["uid"].duplicated().sum())
        raise CatalogueError(f"parcel catalogue contains {n_dup} duplicate uid values")

    bad = ~np.isfinite(out["lat"].to_numpy()) | ~np.isfinite(out["lon"].to_numpy())
    if bad.any():
        out = out.loc[~bad].reset_index(drop=True)

    return out.sort_values("uid", kind="stable").reset_index(drop=True)


def catalogue_from_filenames(preprocess_dir: Path | str) -> pd.DataFrame:
    """Build a degraded catalogue by parsing ``.npz`` filenames only.

    The centroid is unavailable without opening every file, so ``lat`` and
    ``lon`` are ``NaN`` and spatial blocking cannot be performed.  This exists
    so that the class-frequency and K-shot machinery remains usable before the
    full catalogue has been built.

    Args:
        preprocess_dir: Directory holding the per-parcel ``.npz`` files.

    Returns:
        A catalogue with the required columns, ``lat`` and ``lon`` set to
        ``NaN`` and ``n_timesteps`` set to ``-1``.

    Raises:
        CatalogueError: If no parsable ``.npz`` files are found.
    """
    preprocess_dir = Path(preprocess_dir)
    rows: list[tuple[str, str, str, str, str, int, float, float]] = []
    for entry in os.scandir(preprocess_dir):
        m = _FNAME_RE.match(entry.name)
        if m is None:
            continue
        nuts = m.group("nuts")
        country = CODE_COUNTRY.get(nuts[:2], "")
        if not country:
            continue
        rows.append(
            (
                entry.path,
                country,
                nuts,
                m.group("pid"),
                m.group("cls"),
                -1,
                float("nan"),
                float("nan"),
            )
        )
    df = pd.DataFrame.from_records(
        rows,
        columns=["path", "country", "nuts", "parcel_id", "hcat", "n_timesteps", "lat", "lon"],
    )
    if len(df) == 0:
        raise CatalogueError(f"no EuroCropsML .npz files found under {preprocess_dir}")
    # validate_catalogue drops rows with non-finite coordinates, which would
    # empty the degraded catalogue, so the normalisation is done by hand here.
    df["country_code"] = df["country"].map(COUNTRY_CODE).astype(str)
    df["uid"] = df["nuts"] + "_" + df["parcel_id"] + "_" + df["hcat"]
    return df.sort_values("uid", kind="stable").reset_index(drop=True)


def load_catalogue(
    path: Path | str | None = None,
    *,
    countries: list[str] | None = None,
    fallback_preprocess_dir: Path | str | None = None,
) -> pd.DataFrame:
    """Load the parcel catalogue, degrading gracefully if it is absent.

    Args:
        path: Path to the catalogue parquet.  Defaults to
            ``data/catalogue_parcels_full.parquet`` relative to the repository
            root.
        countries: Optional list of ISO-2 country codes to retain, for example
            ``["EE"]``.
        fallback_preprocess_dir: If the parquet is absent and this directory
            exists, a degraded filename-only catalogue is built from it.

    Returns:
        A validated catalogue.  ``df.attrs`` carries ``catalogue_source`` and
        ``catalogue_degraded``.

    Raises:
        CatalogueError: If neither the parquet nor the fallback is available.
    """
    path = default_catalogue_path() if path is None else Path(path)
    if path.is_file():
        df = validate_catalogue(pd.read_parquet(path))
        df.attrs["catalogue_source"] = str(path)
        df.attrs["catalogue_degraded"] = False
    elif fallback_preprocess_dir is not None and Path(fallback_preprocess_dir).is_dir():
        df = catalogue_from_filenames(fallback_preprocess_dir)
        df.attrs["catalogue_source"] = str(fallback_preprocess_dir)
        df.attrs["catalogue_degraded"] = True
    else:
        raise CatalogueError(
            f"parcel catalogue not found at {path} and no usable fallback directory was given. "
            "Build it with `python results/eda/_build_catalogue.py`."
        )

    if countries is not None:
        keep = set(countries)
        source = df.attrs.get("catalogue_source", "")
        degraded = df.attrs.get("catalogue_degraded", False)
        df = df.loc[df["country_code"].isin(keep)].reset_index(drop=True)
        df.attrs["catalogue_source"] = source
        df.attrs["catalogue_degraded"] = degraded
        if len(df) == 0:
            raise CatalogueError(f"no parcels left after filtering to countries {sorted(keep)}")
    return df


@dataclass(frozen=True)
class CatalogueSummary:
    """Compact description of a catalogue, stamped into split manifests."""

    n_parcels: int
    countries: tuple[str, ...]
    n_classes: int
    degraded: bool
    source: str

    @classmethod
    def of(cls, df: pd.DataFrame) -> "CatalogueSummary":
        """Summarise a loaded catalogue."""
        return cls(
            n_parcels=int(len(df)),
            countries=tuple(sorted(df["country_code"].unique().tolist())),
            n_classes=int(df["hcat"].nunique()),
            degraded=bool(df.attrs.get("catalogue_degraded", False)),
            source=str(df.attrs.get("catalogue_source", "")),
        )

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "n_parcels": self.n_parcels,
            "countries": list(self.countries),
            "n_classes": self.n_classes,
            "degraded": self.degraded,
            "source": self.source,
        }
