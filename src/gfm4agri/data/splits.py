"""Reference implementation of the Phase 1 splitting logic.

This module is the single source of truth for how parcels are divided into a
held-out test partition, a validation partition and a training pool, and for
how K-shot support sets are drawn from that pool.  It is pure and
deterministic: given a :class:`SplitConfig`, a parcel catalogue and a class
scheme, it produces byte-identical artefacts on any machine, and it has no
dependency on any foundation model, so it can be exercised before a single
embedding exists.

The design follows ``docs/phase1/protocol.md``.  The four properties the
implementation is required to guarantee, and which the test suite asserts, are:

#. **A single fixed test partition.**  The test set depends only on the
   protocol configuration and the protocol seed, never on the label budget,
   the draw seed, the model or the head.
#. **Spatial separation.**  No block contributes parcels to more than one
   partition, and pool parcels within the buffer distance of a held-out block
   are discarded.
#. **Nested support sets.**  For a fixed ``(country, class, draw_seed)`` the
   support set at budget :math:`K` is a prefix of the support set at any
   larger budget, so moving along the learning curve adds labels and never
   exchanges them.
#. **Model independence.**  The support set is a function of the split
   configuration, the country, the class and the draw seed alone.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd

from gfm4agri.data.autocorr import AgreementDecay, estimate_label_range_m
from gfm4agri.data.blocks import assign_blocks, assign_partitions, buffer_mask
from gfm4agri.data.catalogue import CatalogueSummary, load_catalogue
from gfm4agri.data.class_scheme import ClassScheme, load_class_scheme
from gfm4agri.data.seeding import derived_rng

__all__ = [
    "ALL_BUDGET",
    "K_GRID",
    "CountrySplit",
    "SplitBundle",
    "SplitConfig",
    "build_split",
    "build_support_order",
    "intersect_label_space",
    "load_official_eurocropsml_split",
    "read_split_bundle",
    "support_at",
    "write_split_bundle",
]

#: Sentinel for the full-budget point of the learning curve.
ALL_BUDGET: int = -1

#: The Phase 1 label-budget grid, in labelled samples per class.  The finite
#: values are the union of the EuroCropsML benchmark grid and the grid stated
#: in the thesis work plan; ``ALL_BUDGET`` denotes the whole training pool.
K_GRID: tuple[int, ...] = (1, 5, 10, 20, 50, 100, 200, 500, ALL_BUDGET)


def _json_default(obj: object) -> object:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj)!r}")


@dataclass(frozen=True)
class SplitConfig:
    """Every knob of the Phase 1 splitting protocol.

    The configuration alone, together with the parcel catalogue and the class
    scheme, determines the split.  Its hash is stamped into every artefact.

    Attributes:
        name: Short protocol name, used as the artefact directory prefix.
        countries: ISO-2 country codes to build splits for.
        test_fraction: Share of parcels held out for testing.
        val_fraction: Share of parcels held out for validation.
        protocol_seed: Seed of the block-to-partition assignment.  Zero is the
            primary protocol; values one to four are the folds of the block
            cross-validation robustness check.
        block_size_m: Block edge length in metres.  ``None`` estimates it from
            the label-agreement decay of each country independently.
        block_size_source: ``"estimated"`` or ``"fixed"``; set automatically.
        buffer_m: Width of the exclusion buffer applied to pool parcels
            adjacent to a held-out block, in metres.  It must be at least half
            the diagonal of the largest patch footprint used by any model in
            the sweep, so that a test parcel's patch cannot contain a training
            parcel; see
            :func:`gfm4agri.data.representations.required_buffer_m`.  The
            default corresponds to a footprint of 224 pixels at 10 m.
        min_pool_per_class: A class is evaluated only if its training pool
            holds at least this many parcels.
        min_test_per_class: A class is evaluated only if the test partition
            holds at least this many parcels of it.
        max_test_parcels: Upper bound on the size of a country's test set.
            Larger test partitions are subsampled uniformly at random, which
            preserves the natural class priors.
        max_pool_per_class: Upper bound on the number of pool parcels per
            class retained for the full-budget point, so that feature
            extraction and the full-budget fits stay tractable.  It is a single
            protocol-level constant set by the most restrictive model in the
            sweep, never a per-model value, so that the full-budget point is
            the same experiment for every model.  ``None`` disables the cap.
        support_sampling: ``"block_spread"`` draws support parcels from as
            many distinct blocks as possible; ``"random"`` draws uniformly
            from the class pool and is reported only as a sensitivity.
        k_grid: The label-budget grid.
        n_draws: Number of independent support draws at each budget.
        class_scheme_path: Optional explicit path to the class scheme.
        catalogue_path: Optional explicit path to the parcel catalogue.
        fallback_n_classes: Number of classes per country used by the
            frequency-derived fallback scheme.
        fallback_min_parcels: Minimum class size used by the fallback scheme.
        autocorr_seed: Seed of the label-agreement decay subsampling.
        autocorr_fallback_block_m: Block size used when no range can be fitted.
    """

    name: str = "incountry_v1"
    countries: tuple[str, ...] = ("EE", "LV", "PT")
    test_fraction: float = 0.20
    val_fraction: float = 0.10
    protocol_seed: int = 0
    block_size_m: float | None = None
    block_size_source: str = "estimated"
    buffer_m: float = 1_600.0
    min_pool_per_class: int = 500
    min_test_per_class: int = 200
    max_test_parcels: int = 30_000
    max_pool_per_class: int | None = 2_000
    support_sampling: str = "block_spread"
    k_grid: tuple[int, ...] = K_GRID
    n_draws: int = 5
    class_scheme_path: str | None = None
    catalogue_path: str | None = None
    fallback_n_classes: int = 20
    fallback_min_parcels: int = 500
    autocorr_seed: int = 0
    autocorr_fallback_block_m: float = 10_000.0

    def __post_init__(self) -> None:
        if self.support_sampling not in ("block_spread", "random"):
            raise ValueError(f"unknown support_sampling {self.support_sampling!r}")
        if self.block_size_m is not None and self.block_size_m <= 0:
            raise ValueError("block_size_m must be positive when fixed")
        if ALL_BUDGET in self.k_grid and self.k_grid[-1] != ALL_BUDGET:
            raise ValueError("ALL_BUDGET must be the last entry of k_grid")
        if self.n_draws < 1:
            raise ValueError("n_draws must be at least one")

    @property
    def max_finite_k(self) -> int:
        """Largest finite budget on the grid."""
        finite = [k for k in self.k_grid if k != ALL_BUDGET]
        return max(finite) if finite else 0

    def to_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        d = asdict(self)
        d["countries"] = list(self.countries)
        d["k_grid"] = list(self.k_grid)
        return d

    def config_hash(self) -> str:
        """Return a stable 16-character content hash of the configuration."""
        payload = json.dumps(self.to_dict(), sort_keys=True, default=_json_default)
        return hashlib.blake2b(payload.encode("utf-8"), digest_size=8).hexdigest()

    def protocol_id(self) -> str:
        """Return the artefact directory name, ``<name>__<hash8>``."""
        return f"{self.name}__{self.config_hash()[:8]}"


@dataclass
class CountrySplit:
    """The split of one country.

    Attributes:
        country_code: ISO-2 country code.
        partition: One row per retained parcel, with columns ``uid``,
            ``hcat``, ``block_id``, ``block_ix``, ``block_iy``, ``x_m``,
            ``y_m``, ``lat``, ``lon`` and ``partition``.
        classes: The evaluated class set, in a deterministic order.
        dropped_classes: Candidate classes excluded, mapped to the reason.
        block_size_m: The block edge length actually used.
        autocorr: Diagnostics of the label-range estimation, if performed.
        counts: Per-class parcel counts per partition.
        support_orders: Mapping from draw seed to the support ordering frame.
        notes: Free-form provenance notes.
    """

    country_code: str
    partition: pd.DataFrame
    classes: tuple[str, ...]
    dropped_classes: dict[str, str] = field(default_factory=dict)
    block_size_m: float = 0.0
    autocorr: AgreementDecay | None = None
    counts: pd.DataFrame = field(default_factory=pd.DataFrame)
    support_orders: dict[int, pd.DataFrame] = field(default_factory=dict)
    notes: dict = field(default_factory=dict)

    def part(self, name: str) -> pd.DataFrame:
        """Return the rows of one partition."""
        return self.partition.loc[self.partition["partition"] == name].reset_index(drop=True)

    def summary(self) -> dict:
        """Return a JSON-serialisable summary for the manifest."""
        sizes = self.partition["partition"].value_counts().to_dict()
        return {
            "country_code": self.country_code,
            "block_size_m": float(self.block_size_m),
            "n_classes": len(self.classes),
            "classes": list(self.classes),
            "dropped_classes": dict(self.dropped_classes),
            "partition_sizes": {str(k): int(v) for k, v in sorted(sizes.items())},
            "n_blocks": int(self.partition["block_id"].nunique()),
            "n_blocks_test": int(self.part("test")["block_id"].nunique()),
            "autocorr": None if self.autocorr is None else self.autocorr.to_dict(),
            "notes": self.notes,
        }


@dataclass
class SplitBundle:
    """All country splits produced by one configuration."""

    config: SplitConfig
    countries: dict[str, CountrySplit]
    catalogue: CatalogueSummary
    scheme: ClassScheme

    def manifest(self) -> dict:
        """Return the manifest written alongside the split artefacts."""
        return {
            "protocol_id": self.config.protocol_id(),
            "config_hash": self.config.config_hash(),
            "config": self.config.to_dict(),
            "k_definition": "samples_per_class",
            "catalogue": self.catalogue.to_dict(),
            "class_scheme": {
                "scheme_id": self.scheme.scheme_id,
                "source": self.scheme.source,
                "digest": self.scheme.digest(),
            },
            "countries": {cc: cs.summary() for cc, cs in sorted(self.countries.items())},
            "environment": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "platform": platform.platform(),
            },
        }


# ---------------------------------------------------------------------------
# Support-set construction
# ---------------------------------------------------------------------------


def _block_spread_order(uids: np.ndarray, block_ids: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Order parcels so that every prefix touches as many blocks as possible.

    Members of each block are shuffled, the blocks themselves are shuffled,
    and the per-block queues are then interleaved round-robin.  The first
    :math:`K` entries therefore fall in :math:`\\min(K, B)` distinct blocks,
    where :math:`B` is the number of blocks holding the class.

    Args:
        uids: Parcel identifiers.
        block_ids: Block identifier of each parcel.
        rng: Generator supplying the shuffles.

    Returns:
        The parcel identifiers in interleaved order.
    """
    order = np.argsort(uids, kind="stable")
    uids = uids[order]
    block_ids = block_ids[order]

    queues: dict[object, list] = {}
    for uid, bid in zip(uids.tolist(), block_ids.tolist()):
        queues.setdefault(bid, []).append(uid)

    keys = sorted(queues)
    for k in keys:
        members = np.array(queues[k], dtype=object)
        queues[k] = list(members[rng.permutation(members.size)])
    key_order = [keys[i] for i in rng.permutation(len(keys))]

    out: list = []
    depth = 0
    remaining = sum(len(v) for v in queues.values())
    while remaining:
        for k in key_order:
            q = queues[k]
            if depth < len(q):
                out.append(q[depth])
                remaining -= 1
        depth += 1
    return np.array(out, dtype=object)


def build_support_order(
    pool: pd.DataFrame,
    classes: Iterable[str],
    config: SplitConfig,
    country_code: str,
    draw_seed: int,
) -> pd.DataFrame:
    """Build the per-class support ordering for one draw.

    Rather than materialising a separate support set for every budget, the
    protocol stores a single ordering per ``(country, draw_seed)``.  The
    support set at budget :math:`K` is the prefix of rank below :math:`K`,
    which makes the nesting property structural rather than incidental and
    reduces the artefact count by a factor equal to the size of the budget
    grid.

    Args:
        pool: The training pool of one country, with ``uid``, ``hcat`` and
            ``block_id`` columns.
        classes: The evaluated class set.
        config: The split configuration.
        country_code: ISO-2 country code, mixed into the seed.
        draw_seed: Index of the support draw, from zero to ``n_draws - 1``.

    Returns:
        A frame with columns ``uid``, ``hcat``, ``block_id``, ``rank`` and
        ``draw_seed``, ordered by class then rank.  Only the first
        ``max_finite_k`` ranks per class are materialised; the full-budget
        point uses the pool directly.
    """
    keep = int(config.max_finite_k)
    frames: list[pd.DataFrame] = []
    by_class = {c: g for c, g in pool.groupby("hcat", sort=False)}
    for cls in classes:
        grp = by_class.get(cls)
        if grp is None or len(grp) == 0:
            continue
        rng = derived_rng(
            "support_order",
            config.config_hash(),
            country_code,
            cls,
            int(draw_seed),
            config.support_sampling,
        )
        uids = grp["uid"].to_numpy(dtype=object)
        if config.support_sampling == "block_spread":
            ordered = _block_spread_order(uids, grp["block_id"].to_numpy(dtype=object), rng)
        else:
            sorted_uids = uids[np.argsort(uids, kind="stable")]
            ordered = sorted_uids[rng.permutation(sorted_uids.size)]
        ordered = ordered[:keep]
        block_lookup = dict(zip(grp["uid"].tolist(), grp["block_id"].tolist()))
        frames.append(
            pd.DataFrame(
                {
                    "uid": ordered,
                    "hcat": cls,
                    "block_id": [block_lookup[u] for u in ordered.tolist()],
                    "rank": np.arange(ordered.size, dtype=np.int32),
                    "draw_seed": np.int32(draw_seed),
                }
            )
        )
    if not frames:
        return pd.DataFrame(columns=["uid", "hcat", "block_id", "rank", "draw_seed"])
    return pd.concat(frames, ignore_index=True)


def support_at(order: pd.DataFrame, k: int, pool: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the support set at budget ``k``.

    Args:
        order: A support ordering from :func:`build_support_order`.
        k: The label budget in samples per class, or :data:`ALL_BUDGET`.
        pool: The training pool, required only when ``k`` is
            :data:`ALL_BUDGET`.

    Returns:
        The selected parcels, with ``uid``, ``hcat`` and ``block_id``.

    Raises:
        ValueError: If ``k`` is :data:`ALL_BUDGET` and no pool is supplied, or
            if ``k`` is not positive.
    """
    if k == ALL_BUDGET:
        if pool is None:
            raise ValueError("the full-budget support set requires the training pool")
        return pool[["uid", "hcat", "block_id"]].reset_index(drop=True)
    if k <= 0:
        raise ValueError(f"budget must be positive or ALL_BUDGET, got {k!r}")
    sel = order.loc[order["rank"] < int(k), ["uid", "hcat", "block_id"]]
    return sel.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Split construction
# ---------------------------------------------------------------------------


def _build_country_split(
    catalogue: pd.DataFrame,
    scheme: ClassScheme,
    config: SplitConfig,
    country_code: str,
) -> CountrySplit:
    df = catalogue.loc[catalogue["country_code"] == country_code].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(f"catalogue holds no parcels for country {country_code!r}")
    df = df.copy()
    df["hcat"] = scheme.apply_aliases(df["hcat"])

    candidates = tuple(scheme.for_country(country_code))
    df = df.loc[df["hcat"].isin(candidates)].reset_index(drop=True)
    if len(df) == 0:
        raise ValueError(
            f"no parcels of country {country_code!r} belong to the class scheme "
            f"{scheme.scheme_id!r}"
        )

    notes: dict = {}
    decay: AgreementDecay | None = None
    if config.block_size_m is None:
        x_m, y_m = None, None
        probe = assign_blocks(df["lon"].to_numpy(), df["lat"].to_numpy(), 1.0)
        x_m, y_m = probe["x_m"].to_numpy(), probe["y_m"].to_numpy()
        block_size_m, decay = estimate_label_range_m(
            x_m,
            y_m,
            df["hcat"].to_numpy(),
            seed=config.autocorr_seed,
            fallback_m=config.autocorr_fallback_block_m,
        )
        notes["block_size_source"] = "estimated"
    else:
        block_size_m = float(config.block_size_m)
        notes["block_size_source"] = "fixed"

    blk = assign_blocks(df["lon"].to_numpy(), df["lat"].to_numpy(), block_size_m)
    df = pd.concat([df.reset_index(drop=True), blk], axis=1)

    partition = assign_partitions(
        blk,
        test_fraction=config.test_fraction,
        val_fraction=config.val_fraction,
        protocol_seed=config.protocol_seed,
        country_code=country_code,
        block_size_m=block_size_m,
    )
    flagged = buffer_mask(
        blk, partition, block_size_m=block_size_m, buffer_m=float(config.buffer_m)
    )
    partition = partition.where(~pd.Series(flagged, index=partition.index), "buffer")
    df["partition"] = partition.to_numpy()
    notes["n_buffered"] = int(flagged.sum())

    # Cap the test partition, preserving the natural class priors.
    test_idx = np.flatnonzero(df["partition"].to_numpy() == "test")
    if test_idx.size > config.max_test_parcels:
        rng = derived_rng("test_cap", config.config_hash(), country_code)
        drop = rng.choice(test_idx, size=test_idx.size - config.max_test_parcels, replace=False)
        df.loc[drop, "partition"] = "unused"
        notes["test_subsampled_to"] = int(config.max_test_parcels)

    # Cap the training pool per class, so that the full-budget point and the
    # embedding extraction stay tractable.
    if config.max_pool_per_class is not None:
        rng = derived_rng("pool_cap", config.config_hash(), country_code)
        for cls, grp in df.loc[df["partition"] == "pool"].groupby("hcat", sort=True):
            if len(grp) > config.max_pool_per_class:
                drop = rng.choice(
                    grp.index.to_numpy(),
                    size=len(grp) - config.max_pool_per_class,
                    replace=False,
                )
                df.loc[drop, "partition"] = "unused"
        notes["pool_capped_per_class_at"] = int(config.max_pool_per_class)

    counts = (
        df.groupby(["hcat", "partition"], sort=True)
        .size()
        .unstack(fill_value=0)
        .reindex(columns=["pool", "val", "test", "buffer", "unused"], fill_value=0)
        .reset_index()
    )

    dropped: dict[str, str] = {}
    kept: list[str] = []
    lookup = counts.set_index("hcat")
    for cls in candidates:
        if cls not in lookup.index:
            dropped[cls] = "absent from this country after alias merging"
            continue
        n_pool = int(lookup.loc[cls, "pool"])
        n_test = int(lookup.loc[cls, "test"])
        if n_pool < config.min_pool_per_class:
            dropped[cls] = f"training pool holds {n_pool} < {config.min_pool_per_class} parcels"
        elif n_test < config.min_test_per_class:
            dropped[cls] = f"test partition holds {n_test} < {config.min_test_per_class} parcels"
        else:
            kept.append(cls)

    keep_mask = df["hcat"].isin(kept)
    df.loc[~keep_mask, "partition"] = "unused"

    cols = [
        "uid",
        "hcat",
        "partition",
        "block_id",
        "block_ix",
        "block_iy",
        "x_m",
        "y_m",
        "lat",
        "lon",
        "nuts",
    ]
    cols = [c for c in cols if c in df.columns]
    out = df.loc[df["partition"] != "unused", cols].reset_index(drop=True)

    return CountrySplit(
        country_code=country_code,
        partition=out,
        classes=tuple(kept),
        dropped_classes=dropped,
        block_size_m=float(block_size_m),
        autocorr=decay,
        counts=counts,
        notes=notes,
    )


def build_split(
    config: SplitConfig,
    *,
    catalogue: pd.DataFrame | None = None,
    scheme: ClassScheme | None = None,
    fallback_preprocess_dir: Path | str | None = None,
) -> SplitBundle:
    """Build every country split described by ``config``.

    Args:
        config: The split configuration.
        catalogue: A pre-loaded parcel catalogue.  Loaded from disk when
            omitted.
        scheme: A pre-loaded class scheme.  Loaded from
            ``configs/class_scheme_eurocropsml.yaml`` when omitted, falling
            back to a frequency-derived scheme if that file is absent.
        fallback_preprocess_dir: Passed to
            :func:`~gfm4agri.data.catalogue.load_catalogue` when the catalogue
            parquet is missing.

    Returns:
        A :class:`SplitBundle` holding one :class:`CountrySplit` per country,
        each with its support orderings already drawn.
    """
    if catalogue is None:
        catalogue = load_catalogue(
            config.catalogue_path,
            countries=list(config.countries),
            fallback_preprocess_dir=fallback_preprocess_dir,
        )
    if scheme is None:
        scheme = load_class_scheme(
            config.class_scheme_path,
            catalogue=catalogue,
            n_classes=config.fallback_n_classes,
            min_parcels=config.fallback_min_parcels,
        )

    splits: dict[str, CountrySplit] = {}
    for cc in config.countries:
        cs = _build_country_split(catalogue, scheme, config, cc)
        pool = cs.part("pool")
        for draw_seed in range(config.n_draws):
            cs.support_orders[draw_seed] = build_support_order(
                pool, cs.classes, config, cc, draw_seed
            )
        splits[cc] = cs

    return SplitBundle(
        config=config,
        countries=splits,
        catalogue=CatalogueSummary.of(catalogue),
        scheme=scheme,
    )


# ---------------------------------------------------------------------------
# Cross-country transfer
# ---------------------------------------------------------------------------


def intersect_label_space(
    bundle: SplitBundle,
    source: str | tuple[str, ...],
    target: str,
    *,
    min_source_pool: int | None = None,
) -> tuple[str, ...]:
    """Compute the label space of a directed transfer setting.

    The intersection is taken over the *evaluated* class sets of the split,
    not over the candidate class scheme, so a class that failed the minimum
    count test in either country is excluded from the transfer setting as
    well.  For a many-to-one setting the source class sets are intersected
    first, which keeps the label space identical for every source country and
    therefore keeps the pooled source head well defined.

    Args:
        bundle: A built split bundle.
        source: One source country code, or a tuple of them.
        target: The target country code.
        min_source_pool: Optional additional requirement on the number of
            source pool parcels per class.

    Returns:
        The intersected class codes, ordered as in the target country's class
        set, so that the label ordering is a property of the target.

    Raises:
        KeyError: If a named country was not built in this bundle.
    """
    sources = (source,) if isinstance(source, str) else tuple(source)
    tgt_classes = bundle.countries[target].classes
    common = set(tgt_classes)
    for cc in sources:
        common &= set(bundle.countries[cc].classes)
    if min_source_pool is not None:
        for cc in sources:
            pool = bundle.countries[cc].part("pool")
            counts = pool["hcat"].value_counts()
            common &= {c for c in common if int(counts.get(c, 0)) >= min_source_pool}
    return tuple(c for c in tgt_classes if c in common)


# ---------------------------------------------------------------------------
# Artefact input and output
# ---------------------------------------------------------------------------


def split_dir(results_root: Path | str, config: SplitConfig) -> Path:
    """Return the artefact directory of a protocol."""
    return Path(results_root) / "phase1" / "splits" / config.protocol_id()


def write_split_bundle(bundle: SplitBundle, results_root: Path | str) -> Path:
    """Write a split bundle to the canonical artefact layout.

    The layout under ``results/phase1/splits/<protocol_id>/`` is::

        manifest.json                   configuration, hashes, per-country summary
        partition_<CC>.parquet          one row per parcel, with its partition
        counts_<CC>.parquet             per-class parcel counts per partition
        support/<CC>__seed<s>.parquet   per-class support ordering for one draw

    Args:
        bundle: The bundle to write.
        results_root: The repository's ``results`` directory.

    Returns:
        The protocol directory that was written.
    """
    out = split_dir(results_root, bundle.config)
    (out / "support").mkdir(parents=True, exist_ok=True)
    for cc, cs in bundle.countries.items():
        cs.partition.to_parquet(out / f"partition_{cc}.parquet", index=False)
        cs.counts.to_parquet(out / f"counts_{cc}.parquet", index=False)
        for seed, order in cs.support_orders.items():
            order.to_parquet(out / "support" / f"{cc}__seed{seed}.parquet", index=False)
    (out / "manifest.json").write_text(
        json.dumps(bundle.manifest(), indent=2, sort_keys=True, default=_json_default),
        encoding="utf-8",
    )
    return out


def read_split_bundle(path: Path | str) -> SplitBundle:
    """Read a split bundle written by :func:`write_split_bundle`.

    Args:
        path: The protocol directory.

    Returns:
        The reconstructed bundle.  The autocorrelation diagnostics are read
        back from the manifest as plain dictionaries under
        ``CountrySplit.notes["autocorr"]`` rather than as
        :class:`~gfm4agri.data.autocorr.AgreementDecay` instances.

    Raises:
        FileNotFoundError: If the manifest is absent.
    """
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    cfg_raw = dict(manifest["config"])
    cfg_raw["countries"] = tuple(cfg_raw["countries"])
    cfg_raw["k_grid"] = tuple(cfg_raw["k_grid"])
    config = SplitConfig(**cfg_raw)

    countries: dict[str, CountrySplit] = {}
    for cc, summary in manifest["countries"].items():
        partition = pd.read_parquet(path / f"partition_{cc}.parquet")
        counts = pd.read_parquet(path / f"counts_{cc}.parquet")
        orders: dict[int, pd.DataFrame] = {}
        for f in sorted((path / "support").glob(f"{cc}__seed*.parquet")):
            seed = int(f.stem.split("seed")[-1])
            orders[seed] = pd.read_parquet(f)
        notes = dict(summary.get("notes", {}))
        notes["autocorr"] = summary.get("autocorr")
        countries[cc] = CountrySplit(
            country_code=cc,
            partition=partition,
            classes=tuple(summary["classes"]),
            dropped_classes=dict(summary.get("dropped_classes", {})),
            block_size_m=float(summary["block_size_m"]),
            autocorr=None,
            counts=counts,
            support_orders=orders,
            notes=notes,
        )

    cat = manifest["catalogue"]
    scheme_meta = manifest["class_scheme"]
    return SplitBundle(
        config=config,
        countries=countries,
        catalogue=CatalogueSummary(
            n_parcels=int(cat["n_parcels"]),
            countries=tuple(cat["countries"]),
            n_classes=int(cat["n_classes"]),
            degraded=bool(cat["degraded"]),
            source=str(cat["source"]),
        ),
        scheme=ClassScheme(
            scheme_id=str(scheme_meta["scheme_id"]),
            source=str(scheme_meta["source"]),
            classes={cc: cs.classes for cc, cs in countries.items()},
        ),
    )


def load_official_eurocropsml_split(json_path: Path | str) -> dict[str, list[str]]:
    """Read one of the official EuroCropsML split files.

    The official artefacts are JSON objects mapping ``train``, ``val`` and
    ``test`` to lists of ``.npz`` filenames.  They are consumed unchanged by
    the benchmark track of Phase 1, so that its numbers sit next to the
    published ones.

    Args:
        json_path: Path to a file such as
            ``split/latvia_vs_estonia/finetune/region_split_20.json``.

    Returns:
        A mapping from partition name to a list of parcel ``uid`` values, that
        is filenames stripped of the ``.npz`` suffix.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file does not hold the expected structure.
    """
    raw = json.loads(Path(json_path).read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError(f"{json_path} does not hold a mapping of partitions to file lists")
    out: dict[str, list[str]] = {}
    for key, value in raw.items():
        if not isinstance(value, list):
            raise ValueError(f"partition {key!r} in {json_path} is not a list")
        out[str(key)] = [str(v).removesuffix(".npz") for v in value]
    return out
