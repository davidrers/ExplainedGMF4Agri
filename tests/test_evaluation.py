"""Reloading a finished fit: the checkpoint it was tested with is the one predicted with."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from gfm4agri.benchmark.evaluation import fit_checkpoint  # noqa: E402


def test_the_checkpoint_the_fit_was_tested_with_is_reloaded(tmp_path):
    # A cell fitted again after an interrupted fit holds the old best-loss.ckpt beside the new -v1.
    ck = tmp_path / "checkpoints"
    ck.mkdir()
    (ck / "best-loss.ckpt").write_bytes(b"interrupted fit")
    (ck / "best-loss-v1.ckpt").write_bytes(b"tested fit")
    results = {"best_checkpoint": "/home/someone/ExplainedGMF4Agri/results/x/checkpoints/best-loss-v1.ckpt"}
    assert fit_checkpoint(tmp_path, results) == ck / "best-loss-v1.ckpt"


def test_a_fit_recording_no_checkpoint_reloads_best_loss(tmp_path):
    assert fit_checkpoint(tmp_path, {}) == tmp_path / "checkpoints" / "best-loss.ckpt"
