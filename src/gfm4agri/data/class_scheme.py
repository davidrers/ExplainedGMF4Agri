"""The Phase 1 crop class scheme and its interface contract.

The authoritative class scheme is ``configs/class_scheme_eurocropsml.yaml``,
produced separately from the data.  This module defines the contract that file
must satisfy, loads it, and provides a frequency-derived fallback so that the
splitting logic remains usable and testable before the file exists.

Expected YAML structure
-----------------------
.. code-block:: yaml

    schema_version: 1
    scheme_id: eurocropsml_top20_v1      # optional; derived from the file if absent
    countries:
      EE:
        classes:
          - hcat_code: "3301010101"      # required, string of digits
            name: common_winter_wheat    # required, snake_case English name
            n_parcels: 579               # optional, informational only
      LV: {...}
      PT: {...}
    aliases:                             # optional; merges applied before counting
      "3301010100": ["3301010101", "3301010102"]
    excluded:                            # optional; informational only
      - hcat_code: "3302000000"
        reason: "..."

Only ``countries.<CC>.classes[].hcat_code`` and ``.name`` are consumed.  Any
further key is ignored, so the file may carry documentation freely.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

__all__ = [
    "ClassScheme",
    "ClassSchemeError",
    "default_class_scheme_path",
    "load_class_scheme",
    "derive_class_scheme",
]


class ClassSchemeError(RuntimeError):
    """Raised when the class scheme file is present but malformed."""


def default_class_scheme_path(repo_root: Path | str | None = None) -> Path:
    """Return the conventional location of the class scheme configuration."""
    if repo_root is None:
        repo_root = Path(__file__).resolve().parents[3]
    return Path(repo_root) / "configs" / "class_scheme_eurocropsml.yaml"


@dataclass(frozen=True)
class ClassScheme:
    """An immutable, per-country crop class scheme.

    Attributes:
        scheme_id: Stable identifier stamped into every split manifest.
        source: ``"config"`` when read from YAML, ``"derived_fallback"`` when
            derived from parcel frequencies.
        classes: Mapping from ISO-2 country code to the ordered tuple of HCAT
            codes retained for that country.
        names: Mapping from HCAT code to a human-readable crop name.
        aliases: Mapping from a target HCAT code to the source codes merged
            into it.  Applied before any counting or sampling.
    """

    scheme_id: str
    source: str
    classes: dict[str, tuple[str, ...]]
    names: dict[str, str] = field(default_factory=dict)
    aliases: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def for_country(self, country_code: str) -> tuple[str, ...]:
        """Return the retained HCAT codes for one country.

        Raises:
            ClassSchemeError: If the country is not described by the scheme.
        """
        try:
            return self.classes[country_code]
        except KeyError as exc:
            raise ClassSchemeError(
                f"class scheme {self.scheme_id!r} does not describe country {country_code!r}; "
                f"it describes {sorted(self.classes)}"
            ) from exc

    def intersect(self, *country_codes: str) -> tuple[str, ...]:
        """Return the class codes common to every named country.

        The order follows the first country's ordering, so the intersected
        label space is deterministic regardless of argument order beyond the
        first.
        """
        if not country_codes:
            raise ClassSchemeError("intersect() requires at least one country code")
        first = self.for_country(country_codes[0])
        common = set(first)
        for cc in country_codes[1:]:
            common &= set(self.for_country(cc))
        return tuple(c for c in first if c in common)

    def apply_aliases(self, hcat: pd.Series) -> pd.Series:
        """Map merged HCAT codes onto their target code."""
        if not self.aliases:
            return hcat
        lookup = {src: tgt for tgt, srcs in self.aliases.items() for src in srcs}
        return hcat.map(lambda c: lookup.get(c, c))

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "scheme_id": self.scheme_id,
            "source": self.source,
            "classes": {k: list(v) for k, v in sorted(self.classes.items())},
            "names": dict(sorted(self.names.items())),
            "aliases": {k: list(v) for k, v in sorted(self.aliases.items())},
        }

    def digest(self) -> str:
        """Return a short content hash, stamped into split manifests."""
        payload = json.dumps(
            {
                "classes": {k: list(v) for k, v in sorted(self.classes.items())},
                "aliases": {k: list(v) for k, v in sorted(self.aliases.items())},
            },
            sort_keys=True,
        )
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()


def _parse_scheme_mapping(raw: dict, scheme_id: str) -> ClassScheme:
    countries = raw.get("countries")
    if not isinstance(countries, dict) or not countries:
        raise ClassSchemeError("class scheme must define a non-empty 'countries' mapping")

    classes: dict[str, tuple[str, ...]] = {}
    names: dict[str, str] = {}
    for cc, block in countries.items():
        entries = (block or {}).get("classes")
        if not isinstance(entries, list) or not entries:
            raise ClassSchemeError(f"class scheme country {cc!r} has no 'classes' list")
        codes: list[str] = []
        for item in entries:
            if not isinstance(item, dict) or "hcat_code" not in item:
                raise ClassSchemeError(
                    f"class scheme country {cc!r} has an entry without 'hcat_code': {item!r}"
                )
            code = str(item["hcat_code"]).strip()
            if code in codes:
                raise ClassSchemeError(f"class scheme country {cc!r} repeats class {code!r}")
            codes.append(code)
            if "name" in item:
                names[code] = str(item["name"])
        classes[str(cc)] = tuple(codes)

    aliases_raw = raw.get("aliases") or {}
    aliases = {str(k): tuple(str(v) for v in vs) for k, vs in aliases_raw.items()}

    return ClassScheme(
        scheme_id=str(raw.get("scheme_id") or scheme_id),
        source="config",
        classes=classes,
        names=names,
        aliases=aliases,
    )


def load_class_scheme(
    path: Path | str | None = None,
    *,
    catalogue: pd.DataFrame | None = None,
    n_classes: int = 20,
    min_parcels: int = 500,
) -> ClassScheme:
    """Load the class scheme, falling back to a frequency-derived scheme.

    Args:
        path: Path to the YAML class scheme.  Defaults to
            ``configs/class_scheme_eurocropsml.yaml``.
        catalogue: Parcel catalogue, required only for the fallback.
        n_classes: Number of classes per country in the fallback.
        min_parcels: Minimum parcel count per class in the fallback.

    Returns:
        The loaded or derived scheme.  ``ClassScheme.source`` records which.

    Raises:
        ClassSchemeError: If the file is malformed, or absent with no
            catalogue supplied for the fallback.
    """
    path = default_class_scheme_path() if path is None else Path(path)
    if path.is_file():
        import yaml  # local import so the module imports without PyYAML present

        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ClassSchemeError(f"class scheme at {path} is not a mapping")
        return _parse_scheme_mapping(raw, scheme_id=path.stem)

    if catalogue is None:
        raise ClassSchemeError(
            f"class scheme not found at {path} and no catalogue was supplied for the "
            "frequency-derived fallback"
        )
    return derive_class_scheme(catalogue, n_classes=n_classes, min_parcels=min_parcels)


def derive_class_scheme(
    catalogue: pd.DataFrame,
    *,
    n_classes: int = 20,
    min_parcels: int = 500,
) -> ClassScheme:
    """Derive a per-country class scheme from parcel frequencies.

    This is the documented fallback, not the thesis scheme.  It selects, per
    country, the ``n_classes`` most frequent HCAT codes with at least
    ``min_parcels`` parcels.  Ties are broken by ascending HCAT code so the
    result is deterministic.

    Args:
        catalogue: Validated parcel catalogue.
        n_classes: Maximum number of classes to retain per country.
        min_parcels: Minimum parcel count for a class to be eligible.

    Returns:
        A :class:`ClassScheme` with ``source="derived_fallback"``.
    """
    classes: dict[str, tuple[str, ...]] = {}
    for cc, grp in catalogue.groupby("country_code", sort=True):
        counts = grp["hcat"].value_counts()
        eligible = counts[counts >= min_parcels]
        ordered = sorted(eligible.items(), key=lambda kv: (-int(kv[1]), str(kv[0])))
        classes[str(cc)] = tuple(str(code) for code, _ in ordered[:n_classes])
    scheme_id = f"derived_top{n_classes}_min{min_parcels}"
    return ClassScheme(scheme_id=scheme_id, source="derived_fallback", classes=classes)
