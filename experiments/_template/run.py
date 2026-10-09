"""Entry script of an experiment that changes code: the K-shot runner with this folder's overrides.

Delete this file if the experiment only changes configs; then scripts/run_kshot.py runs it. Otherwise copy each
core module to change into this folder under a new name (for example ``decoders_x.py`` from
``src/gfm4agri/benchmark/decoders.py``), edit the copy, and swap it in within :func:`apply_overrides`. The core
stays untouched. The command line is that of scripts/run_kshot.py:

    poetry run python experiments/<folder>/run.py -e experiments/<folder>/experiment.yaml --chips data/eurocrops_chips/<set>
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]


def apply_overrides() -> None:
    """Replace core objects by this folder's copies before the runner starts, for example:

        import gfm4agri.benchmark.decoders as core
        import decoders_x
        core.PixelMLPDecoder = decoders_x.PixelMLPDecoder

    The runner imports what it needs inside its functions, so a replacement made here is the one it uses.
    """


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
