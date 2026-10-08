"""Find the feature cache of a cache-route arm, or decide where to compute it.

A cache is reused as it is only when its ``cache.json`` matches the run on every field that
changes the stored features (backbone, chip manifest, split, normalisation, precision, the
training variants) and every chip of the split is on disk. Otherwise it is computed into the
scratch root, which resumes whatever chips are already there. A cache that exists but does
not match is never written into: when the scratch root is the cache root, the run stops.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

from gfm4agri.benchmark.cached import EMB_SUFFIX, IDENTITY, TRAIN_VARIANTS, variant_name
from gfm4agri.benchmark.segmentation_data import manifest_sha256, read_split, split_list

__all__ = ["CacheDecision", "cache_dir_for", "expected_record", "resolve_cache"]


@dataclass
class CacheDecision:
    action: str            # "reuse", "compute" or "stop"
    cache_dir: Path
    reason: str
    mismatches: list[str] = field(default_factory=list)


def cache_dir_for(root, backbone: str, chips) -> Path:
    """``<root>/<backbone>/<chip set name>``, the layout of ``data/embeddings``."""
    return Path(root) / backbone / Path(chips).name


def expected_record(chips, split_dir, backbone: str, normalisation: str, precision: str) -> dict:
    return {"backbone": backbone, "manifest_sha256": manifest_sha256(chips),
            "split_config_hash": read_split(split_dir)["config_hash"],
            "normalisation": normalisation, "precision": precision,
            "train_variants": [variant_name(k, f) for k, f in TRAIN_VARIANTS]}


def _mismatches(record: dict, expected: dict) -> list[str]:
    got = {"backbone": record.get("backbone"), "manifest_sha256": record.get("manifest_sha256"),
           "split_config_hash": (record.get("split") or {}).get("config_hash"),
           "normalisation": record.get("normalisation"), "precision": record.get("precision"),
           "train_variants": record.get("train_variants")}
    return [f"{k}: cache {got[k]!r}, run {v!r}" for k, v in expected.items() if got[k] != v]


def _missing(cache_dir: Path, record: dict, chips, split_dir) -> int:
    """Chip encodings the fit would read and the cache does not hold."""
    last = f"layer_{len(record['encoder_layers']) - 1:02d}"
    ids = {s: split_list(chips, s, split_dir).read_text().split() for s in ("train", "val", "test")}
    n = 0
    for k, f in TRAIN_VARIANTS:
        want = ids["train"] + (ids["val"] + ids["test"] if (k, f) == IDENTITY else [])
        folder = cache_dir / variant_name(k, f) / last
        have = ({p[: -len(EMB_SUFFIX)] for p in os.listdir(folder) if p.endswith(EMB_SUFFIX)}
                if folder.is_dir() else set())
        n += sum(c not in have for c in want)
    return n


def resolve_cache(cache_root, scratch, backbone: str, chips, split_dir, normalisation: str,
                  precision: str) -> CacheDecision:
    found = cache_dir_for(cache_root, backbone, chips)
    target = cache_dir_for(scratch, backbone, chips)
    same = Path(cache_root).resolve() == Path(scratch).resolve()
    record_path = found / "cache.json"
    if record_path.exists():
        record = json.loads(record_path.read_text())
        bad = _mismatches(record, expected_record(chips, split_dir, backbone, normalisation,
                                                  precision))
        if bad:
            if same:
                return CacheDecision("stop", found, "the cache does not match the run", bad)
            return CacheDecision("compute", target, f"{found} does not match the run", bad)
        n = _missing(found, record, chips, split_dir)
        if n == 0:
            return CacheDecision("reuse", found, "complete and matching")
        if same:
            return CacheDecision("compute", found, f"{n} chip encodings missing, resumed in place")
        return CacheDecision("compute", target, f"{n} chip encodings missing in {found}")
    if found.exists() and same:
        return CacheDecision("compute", found, "partial cache without cache.json, resumed in place")
    return CacheDecision("compute", target, "no cache found")
