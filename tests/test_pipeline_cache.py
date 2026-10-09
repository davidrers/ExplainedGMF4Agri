"""A feature cache is reused only when it matches the run and is complete, and never overwritten."""

from __future__ import annotations

import json

import pytest

pytest.importorskip("terratorch")

from conftest import SYN_SPLIT  # noqa: E402
from gfm4agri.benchmark.cached import EMB_SUFFIX, IDENTITY, TRAIN_VARIANTS, variant_name  # noqa: E402
from gfm4agri.pipeline.cache import expected_record, resolve_cache  # noqa: E402

BACKBONE = "terramind_v1_large"


def _fake_cache(root, chips, *, drop=None, **override):
    """A cache directory as generate_embeddings leaves it, two layers, empty feature files."""
    split = chips / "splits" / SYN_SPLIT
    d = root / BACKBONE / chips.name
    ids = {s: (split / f"{n}_data.txt").read_text().split()
           for s, n in (("train", "training"), ("val", "validation"), ("test", "test"))}
    for k, f in TRAIN_VARIANTS:
        want = ids["train"] + (ids["val"] + ids["test"] if (k, f) == IDENTITY else [])
        folder = d / variant_name(k, f) / "layer_01"
        folder.mkdir(parents=True)
        for c in want:
            if c != drop:
                (folder / f"{c}{EMB_SUFFIX}").touch()
    rec = expected_record(chips, split, BACKBONE, "backbone", "16-mixed")
    record = {"backbone": rec["backbone"], "manifest_sha256": rec["manifest_sha256"],
              "split": {"dir": str(split), "config_hash": rec["split_config_hash"]},
              "normalisation": "backbone", "precision": "16-mixed",
              "train_variants": rec["train_variants"], "encoder_layers": [5, 11]} | override
    (d / "cache.json").write_text(json.dumps(record))
    return d


def _resolve(chips, cache_root, scratch):
    return resolve_cache(cache_root, scratch, BACKBONE, chips, chips / "splits" / SYN_SPLIT,
                         "backbone", "16-mixed")


def test_a_complete_matching_cache_is_reused(split_chipset, tmp_path):
    d = _fake_cache(tmp_path / "emb", split_chipset)
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "scratch")
    assert (r.action, r.cache_dir) == ("reuse", d)


def test_no_cache_is_computed_in_scratch(split_chipset, tmp_path):
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "scratch")
    assert (r.action, r.cache_dir) == ("compute", tmp_path / "scratch" / BACKBONE / "SYN_2021")


def test_a_partial_cache_is_resumed_in_place_when_scratch_is_the_cache_root(split_chipset, tmp_path):
    d = tmp_path / "emb" / BACKBONE / "SYN_2021" / "k0" / "layer_01"
    d.mkdir(parents=True)
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "emb")
    assert (r.action, r.cache_dir) == ("compute", tmp_path / "emb" / BACKBONE / "SYN_2021")


def test_missing_chips_are_computed_in_scratch(split_chipset, tmp_path):
    _fake_cache(tmp_path / "emb", split_chipset, drop="EE_00008_00000")
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "scratch")
    assert r.action == "compute" and r.cache_dir.parent.parent == tmp_path / "scratch"
    assert "1 chip encodings missing" in r.reason


def test_a_mismatching_cache_stops_the_run_rather_than_being_overwritten(split_chipset, tmp_path):
    _fake_cache(tmp_path / "emb", split_chipset, normalisation="chips")
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "emb")
    assert r.action == "stop" and r.mismatches == ["normalisation: cache 'chips', run 'backbone'"]


def test_a_mismatching_cache_is_left_alone_and_scratch_used(split_chipset, tmp_path):
    _fake_cache(tmp_path / "emb", split_chipset, split={"config_hash": "ffff"})
    r = _resolve(split_chipset, tmp_path / "emb", tmp_path / "scratch")
    assert r.action == "compute" and r.cache_dir.parent.parent == tmp_path / "scratch"
    assert r.mismatches and r.mismatches[0].startswith("split_config_hash")
