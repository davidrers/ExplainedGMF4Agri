"""Entry script of the 12 against 24 frames study: the K-shot runner, with the frame count of a half-month set.

The core datamodule takes the number of frames from ``manifest["imagery"]["n_months"]``. ``EE_2021_sample_24``
declares ``n_periods: 24`` and leaves ``n_months`` null, so that no loader takes it for a monthly set by mistake.
This script gives the datamodule ``n_periods`` as its frame count, in memory only. Nothing else in the chain depends
on the frame count being 12 for TerraMind: the temporal wrapper runs the encoder once per frame and the
``ChannelBottleneck`` takes ``T x 1024`` channels, both built from it.

Encoders that take a date per frame are refused on a half-month set. The core gives them the 15th of month 1 to T,
which is wrong for T = 24; Prithvi on this set would need the period centres from the manifest. The command line is
that of scripts/run_kshot.py:

    poetry run python experiments/2026-10-12_terramind_12v24/run.py -e experiments/2026-10-12_terramind_12v24/experiment.yaml --chips data/eurocrops_chips/EE_2021_sample_24
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = next(p for p in HERE.parents if (p / "pyproject.toml").exists())
sys.path[:0] = [str(HERE), str(REPO / "src"), str(REPO / "scripts")]


def apply_overrides() -> None:
    from gfm4agri.benchmark import segmentation_data as sd

    core_load_manifest = sd.load_manifest
    core_coords = sd.monthly_temporal_coords

    def load_manifest(root):
        m = core_load_manifest(root)
        img = m.get("imagery", {})
        if img.get("n_periods") and not img.get("n_months"):
            img["n_months"] = img["n_periods"]
        return m

    def monthly_temporal_coords(year, n_months=12):
        if n_months != 12:
            raise ValueError(f"{n_months} frames are not calendar months; an encoder that takes dates needs the "
                             "period centres of the manifest, which this experiment does not wire")
        return core_coords(year, n_months)

    sd.load_manifest = load_manifest
    sd.monthly_temporal_coords = monthly_temporal_coords


if __name__ == "__main__":
    apply_overrides()
    import run_kshot

    run_kshot.main()
