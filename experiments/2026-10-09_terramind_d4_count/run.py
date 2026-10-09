"""Entry script of the D4 count study: the K-shot runner with the arm's ``data.train_variants`` passed to the fit.

The core cached fit takes ``train_variants`` but the runner never passes it, so every cache-route arm trains on all
eight D4 variants. This script copies the arm's ``data.train_variants`` into the cell's configuration, where it is
recorded in ``config.yaml`` and ``results.json``, and hands it to :func:`gfm4agri.benchmark.cached.fit_cached`. An
arm without the key (the core ``terramind_v1_large``) trains on all eight, as in the main workflow. Nothing else
differs. The command line is that of scripts/run_kshot.py:

    poetry run python experiments/2026-10-09_terramind_d4_count/run.py -e experiments/2026-10-09_terramind_d4_count/experiment.yaml --chips data/eurocrops_chips/EE_2021_half
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]


def apply_overrides() -> None:
    """Wrap the runner's ``compose`` to carry ``train_variants`` into the cell's configuration, and the cached
    fit to read it from there. The runner imports ``fit_cached`` inside :func:`run`, so the wrapped one is used."""
    from gfm4agri.benchmark import cached
    from gfm4agri.pipeline import runner

    core_compose, core_fit = runner.compose, cached.fit_cached

    def compose(arm_name, arm, *args, **kwargs):
        cfg = core_compose(arm_name, arm, *args, **kwargs)
        if "train_variants" in arm["data"]:
            cfg["data"]["train_variants"] = [str(v) for v in arm["data"]["train_variants"]]
        return cfg

    def fit_cached(config, cache_dir, **kwargs):
        return core_fit(config, cache_dir, train_variants=config["data"].get("train_variants"), **kwargs)

    runner.compose = compose
    cached.fit_cached = fit_cached


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
