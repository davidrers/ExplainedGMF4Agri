"""Input representations, comparison groups and the feature cache contract.

Phase 1 compares encoders, but the four foundation models do not consume the
same input: TerraMind and THOR take multi-date Sentinel-2 patches, AlphaEarth
and TESSERA take a sampled embedding vector, and the raw-feature baselines take
the per-parcel median time series.  These inputs differ in *spatial support*,
that is in how much of the landscape they carry, so a difference in Macro-F1
between models consuming different inputs is not attributable to the encoder.

The input representation is therefore a declared experimental factor rather
than a per-model implementation detail.  This module fixes its levels, the
comparison groups defined over them, and the cache layout, so that the
convention is enforceable in code rather than only described in prose.  See
``docs/phase1/protocol.md``, sections 2.4, 2.5 and 6.2.

A cached feature vector is keyed by ``(parcel uid, model, representation,
model version)``.  There is deliberately **not** one vector per parcel per
model: a single encoder contributes one cache entry per representation it
consumes, and the differences between those entries are the bridge
experiments that estimate the representation offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "BRIDGES",
    "GROUP_OF",
    "REPRESENTATIONS",
    "REPRESENTATION_GROUPS",
    "SPATIAL_SUPPORT",
    "FeatureCacheKey",
    "feature_cache_dir",
    "group_of",
    "required_buffer_m",
]

#: The declared input representations.
REPRESENTATIONS: tuple[str, ...] = (
    "ts_parcel",      # the shipped (T, 13) per-parcel median time series
    "ts_patch",       # the same reduction taken over the patch footprint
    "patch_full",     # a fixed-footprint multi-date Sentinel-2 patch stack
    "patch_masked",   # patch_full with every pixel outside the parcel blanked
    "emb_centroid",   # provider embedding sampled at the parcel centroid
    "emb_zonal",      # provider embedding averaged over the parcel's pixels
)

#: Spatial support carried by each representation.
SPATIAL_SUPPORT: dict[str, str] = {
    "ts_parcel": "parcel_interior",
    "ts_patch": "parcel_plus_neighbourhood",
    "patch_full": "parcel_plus_neighbourhood",
    "patch_masked": "parcel_interior",
    "emb_centroid": "single_pixel",
    "emb_zonal": "parcel_interior",
}

#: Comparison groups.  Every headline comparison is made inside one group;
#: ``G1`` is the headline group for RQ1, because within it the encoder is the
#: only factor that varies.
REPRESENTATION_GROUPS: dict[str, tuple[str, ...]] = {
    "G1": ("ts_parcel", "emb_zonal", "patch_masked"),
    "G2": ("ts_patch", "patch_full"),
    "G3": ("emb_centroid",),
}

GROUP_OF: dict[str, str] = {
    rep: group for group, reps in REPRESENTATION_GROUPS.items() for rep in reps
}

#: The bridge experiments.  Each holds the encoder fixed and varies only the
#: representation, so the difference estimates the representation offset.
#: Mapping from bridge name to ``(representation_a, representation_b)``.
BRIDGES: dict[str, tuple[str, str]] = {
    "A_context": ("patch_full", "patch_masked"),
    "B_support": ("emb_centroid", "emb_zonal"),
    "C_encoder_free": ("ts_patch", "ts_parcel"),
}


def group_of(representation: str) -> str:
    """Return the comparison group of a representation.

    Args:
        representation: One of :data:`REPRESENTATIONS`.

    Returns:
        ``"G1"``, ``"G2"`` or ``"G3"``.

    Raises:
        KeyError: If the representation is not declared.
    """
    try:
        return GROUP_OF[representation]
    except KeyError as exc:
        raise KeyError(
            f"unknown input representation {representation!r}; declared levels are "
            f"{list(REPRESENTATIONS)}"
        ) from exc


def required_buffer_m(patch_px: int, gsd_m: float) -> float:
    """Return the minimum exclusion buffer implied by a patch footprint.

    A test parcel's patch must not contain a training parcel.  Since a patch is
    a square footprint centred on the parcel centroid, the condition is met
    exactly when the buffer is at least half the patch diagonal.  The buffer is
    a property of the shared test partition, so it must be computed once from
    the *largest* footprint used by any model in the sweep and applied to every
    model, including those that consume no patch at all.

    Args:
        patch_px: Patch edge length in pixels.
        gsd_m: Ground sampling distance in metres.

    Returns:
        Half the patch diagonal, in metres.

    Raises:
        ValueError: If either argument is not positive.
    """
    if patch_px <= 0 or gsd_m <= 0:
        raise ValueError("patch_px and gsd_m must both be positive")
    return 0.5 * math.sqrt(2.0) * float(patch_px) * float(gsd_m)


@dataclass(frozen=True)
class FeatureCacheKey:
    """Identity of one feature cache entry.

    Attributes:
        model: Encoder identifier, for example ``terramind`` or
            ``baseline_monthly``.  The raw-feature baselines are treated as
            encoders of a degenerate kind so that they pass through the same
            cache and manifest discipline as the foundation models.
        representation: One of :data:`REPRESENTATIONS`.
        model_version: Version or checkpoint identifier of the encoder.
        country_code: ISO-2 country code of the shard.
    """

    model: str
    representation: str
    model_version: str
    country_code: str

    def __post_init__(self) -> None:
        group_of(self.representation)  # raises on an undeclared representation

    @property
    def group(self) -> str:
        """Comparison group of this cache entry."""
        return group_of(self.representation)

    @property
    def spatial_support(self) -> str:
        """Spatial support carried by this cache entry."""
        return SPATIAL_SUPPORT[self.representation]

    def to_dict(self) -> dict:
        """Return the fields stamped into a result manifest."""
        return {
            "model": self.model,
            "model_version": self.model_version,
            "representation": self.representation,
            "representation_group": self.group,
            "spatial_support": self.spatial_support,
            "country": self.country_code,
        }


def feature_cache_dir(root: Path | str, key: FeatureCacheKey) -> Path:
    """Return the cache directory of one ``(model, representation)`` pair.

    The layout is ``<root>/features/<representation>/<model>/<model_version>/``,
    holding ``<CC>.npy``, ``<CC>_index.parquet``, ``manifest.json`` and
    ``shards_done.json``.  Representation leads the path so that every entry of
    one comparison group is contiguous on disk and a group can be archived or
    recomputed as a unit.

    Args:
        root: The ``data`` directory.
        key: The cache key.

    Returns:
        The directory for this key, not created.
    """
    return (
        Path(root)
        / "features"
        / key.representation
        / key.model
        / key.model_version
    )
