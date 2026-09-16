"""Tests for catalogue loading and the class scheme contract."""

from __future__ import annotations

import textwrap

import pandas as pd
import pytest

from gfm4agri.data.catalogue import (
    CatalogueError,
    CatalogueSummary,
    catalogue_from_filenames,
    load_catalogue,
    validate_catalogue,
)
from gfm4agri.data.class_scheme import (
    ClassSchemeError,
    derive_class_scheme,
    load_class_scheme,
)


def test_validate_adds_uid_and_country_code(synthetic_catalogue: pd.DataFrame) -> None:
    assert {"uid", "country_code"} <= set(synthetic_catalogue.columns)
    assert synthetic_catalogue["uid"].is_unique
    assert set(synthetic_catalogue["country_code"]) == {"EE", "LV"}
    assert synthetic_catalogue["uid"].is_monotonic_increasing


def test_validate_rejects_missing_columns() -> None:
    with pytest.raises(CatalogueError, match="missing columns"):
        validate_catalogue(pd.DataFrame({"country": ["Estonia"]}))


def test_validate_rejects_unknown_country(synthetic_catalogue_raw: pd.DataFrame) -> None:
    bad = synthetic_catalogue_raw.copy()
    bad.loc[0, "country"] = "Finland"
    with pytest.raises(CatalogueError, match="unknown country"):
        validate_catalogue(bad)


def test_validate_rejects_duplicate_uid(synthetic_catalogue_raw: pd.DataFrame) -> None:
    bad = pd.concat([synthetic_catalogue_raw, synthetic_catalogue_raw.head(1)], ignore_index=True)
    with pytest.raises(CatalogueError, match="duplicate uid"):
        validate_catalogue(bad)


def test_load_catalogue_missing_file_raises(tmp_path) -> None:
    with pytest.raises(CatalogueError, match="not found"):
        load_catalogue(tmp_path / "absent.parquet")


def test_load_catalogue_degrades_to_filenames(tmp_path) -> None:
    """With no parquet, a filename-only catalogue is built and flagged."""
    preprocess = tmp_path / "preprocess"
    preprocess.mkdir()
    for name in (
        "EE001_19990038_3302000000.npz",
        "EE001_19990039_3301010101.npz",
        "LV005_12345678_3301010102.npz",
        "not_a_parcel.txt",
    ):
        (preprocess / name).write_bytes(b"")

    df = load_catalogue(tmp_path / "absent.parquet", fallback_preprocess_dir=preprocess)
    assert len(df) == 3
    assert df.attrs["catalogue_degraded"] is True
    assert df["lat"].isna().all()
    assert CatalogueSummary.of(df).degraded is True


def test_filename_catalogue_rejects_empty_directory(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(CatalogueError, match="no EuroCropsML"):
        catalogue_from_filenames(empty)


def test_derived_scheme_is_deterministic_and_ordered(synthetic_catalogue: pd.DataFrame) -> None:
    a = derive_class_scheme(synthetic_catalogue, n_classes=4, min_parcels=100)
    b = derive_class_scheme(synthetic_catalogue, n_classes=4, min_parcels=100)
    assert a.classes == b.classes
    assert a.digest() == b.digest()
    assert a.source == "derived_fallback"
    counts = synthetic_catalogue.loc[
        synthetic_catalogue["country_code"] == "EE", "hcat"
    ].value_counts()
    got = list(a.for_country("EE"))
    assert got == sorted(got, key=lambda c: (-int(counts[c]), c))


def test_scheme_intersection_follows_first_country(synthetic_catalogue: pd.DataFrame) -> None:
    scheme = derive_class_scheme(synthetic_catalogue, n_classes=20, min_parcels=100)
    inter = scheme.intersect("EE", "LV")
    assert set(inter) <= set(scheme.for_country("EE"))
    assert set(inter) <= set(scheme.for_country("LV"))
    order = [c for c in scheme.for_country("EE") if c in set(inter)]
    assert list(inter) == order


def test_load_class_scheme_from_yaml(tmp_path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "class_scheme_eurocropsml.yaml"
    path.write_text(
        textwrap.dedent(
            """
            schema_version: 1
            scheme_id: unit_test_scheme
            countries:
              EE:
                classes:
                  - hcat_code: "3301010101"
                    name: common_winter_wheat
                  - hcat_code: "3302000000"
                    name: pasture_meadow_grassland
              LV:
                classes:
                  - hcat_code: "3302000000"
                    name: pasture_meadow_grassland
            aliases:
              "3301010100": ["3301010101", "3301010102"]
            """
        ),
        encoding="utf-8",
    )
    scheme = load_class_scheme(path)
    assert scheme.scheme_id == "unit_test_scheme"
    assert scheme.source == "config"
    assert scheme.for_country("EE") == ("3301010101", "3302000000")
    assert scheme.names["3301010101"] == "common_winter_wheat"
    assert scheme.intersect("EE", "LV") == ("3302000000",)

    merged = scheme.apply_aliases(pd.Series(["3301010101", "3301010102", "3302000000"]))
    assert merged.tolist() == ["3301010100", "3301010100", "3302000000"]


def test_malformed_scheme_is_rejected(tmp_path) -> None:
    pytest.importorskip("yaml")
    path = tmp_path / "bad.yaml"
    path.write_text("countries:\n  EE:\n    classes: []\n", encoding="utf-8")
    with pytest.raises(ClassSchemeError, match="no 'classes' list"):
        load_class_scheme(path)


def test_unknown_country_in_scheme_is_reported(synthetic_catalogue: pd.DataFrame) -> None:
    scheme = derive_class_scheme(synthetic_catalogue, n_classes=5, min_parcels=100)
    with pytest.raises(ClassSchemeError, match="does not describe country"):
        scheme.for_country("PT")
