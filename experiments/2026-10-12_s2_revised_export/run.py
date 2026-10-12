"""Re-export the Sentinel-2 composites of ``EE_2021`` under the ``REVISED`` screening, into ``EE_2021_rev``.

The core export script, ``scripts/hub/build_country_chips.py``, is run unchanged with two of its module-level
names swapped in this process, so the main workflow is not edited:

* ``composite_window`` becomes ``composite_window(..., screening=REVISED)``;
* ``score_cells`` keeps only the cells of ``EE_2021``'s manifest, so the new set holds exactly the parent's chips
  and the parent's split applies to it unchanged.

Before the export, every product that does not depend on the Sentinel-2 screening is linked from the parent:
the Sentinel-1, TESSERA and AlphaEarth rasters and their records, ``chip_parcels.parquet`` and the split. The
export itself writes the composite, the two label rasters and the report of every chip, and assembles
``manifest.json``; afterwards the manifest is stamped with the screening and its parent. Every step resumes.

    poetry run python experiments/2026-10-12_s2_revised_export/run.py --workers 8 --group-size 2 \
        --only-chips-of data/eurocrops_chips/EE_2021_sample        # first the sample, then
    poetry run python experiments/2026-10-12_s2_revised_export/run.py --workers 8 --group-size 2
    poetry run python experiments/2026-10-12_s2_revised_export/run.py --assemble-only

Every other option is passed to the export script.
"""

from __future__ import annotations

import argparse
import functools
import importlib.util
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.chips.s2_monthly import REVISED, composite_window  # noqa: E402

PARENT = REPO / "data" / "eurocrops_chips" / "EE_2021"
OUT = REPO / "data" / "eurocrops_chips" / "EE_2021_rev"
#: Per-chip files that do not depend on the Sentinel-2 screening.
LINKED_CHIP_SUFFIXES = ["_s1rtc.tif", "_tessera.tif", ".tessera.json", "_alphaearth.tif", ".alphaearth.json"]
#: Chip-set files and directories that do not depend on it either.
LINKED_ROOT = ["chip_parcels.parquet", "chip_parcels.parquet.sha256", "s1_rtc.json", "s1_reports",
               "tessera_v1.json", "alphaearth_v1.json", "splits"]


def _link(src: Path, dst: Path) -> None:
    """A relative link, so the pair of chip sets can be copied together."""
    if dst.is_symlink() or dst.exists():
        return
    os.symlink(os.path.relpath(src, dst.parent), dst)


def link_unchanged(parent_ids: list[str]) -> None:
    (OUT / "chips").mkdir(parents=True, exist_ok=True)
    for name in LINKED_ROOT:
        _link(PARENT / name, OUT / name)
    for cid in parent_ids:
        for suffix in LINKED_CHIP_SUFFIXES:
            _link(PARENT / "chips" / f"{cid}{suffix}", OUT / "chips" / f"{cid}{suffix}")


def load_export_script(parent_cells: set[tuple[int, int]]):
    """The core export script as a module, with the screening and the chip list swapped in.

    The module is registered under its own name before the pool forks, so the workers find the swapped
    names when they unpickle ``_export_group`` by reference.
    """
    spec = importlib.util.spec_from_file_location("build_country_chips",
                                                  REPO / "scripts" / "hub" / "build_country_chips.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["build_country_chips"] = mod
    spec.loader.exec_module(mod)
    mod.composite_window = functools.partial(composite_window, screening=REVISED)
    score_cells = mod.score_cells

    def parent_cells_only(parcels, code_col, codes):
        cells = score_cells(parcels, code_col, codes)
        keep = [(int(c), int(r)) in parent_cells for c, r in zip(cells["col"], cells["row"])]
        cells = cells[keep]
        got = {(int(c), int(r)) for c, r in zip(cells["col"], cells["row"])}
        if got != parent_cells:
            raise RuntimeError(f"{len(parent_cells - got)} of the chips asked for are not cells of the current "
                               "grid scoring; the parent's chip list cannot be reproduced")
        return cells

    mod.score_cells = parent_cells_only
    return mod


def stamp_manifest() -> None:
    path = OUT / "manifest.json"
    if not path.exists():
        return
    m = json.loads(path.read_text())
    m["purpose"] = ("EE_2021 with the Sentinel-2 composites re-exported under the REVISED screening; every "
                    "other product linked from the parent")
    m["derived_from"] = {"chip_set": "data/eurocrops_chips/EE_2021",
                         "experiment": "experiments/2026-10-12_s2_revised_export",
                         "screening_study": "experiments/2026-10-12_s2_screening",
                         "linked": LINKED_ROOT + [f"chips/<chip>{s}" for s in LINKED_CHIP_SUFFIXES]}
    m["imagery"]["screening"] = asdict(REVISED)
    m["imagery"]["composite"] = (
        "per-pixel monthly median of clear observations, one observation per overpass (tile copies and "
        "reprocessings merged, flagged if any copy is flagged); clear: SCL class 4, 5 or 6, at least 2 px from "
        "cloud, shadow, cirrus and snow, passing the Fmask snow test and the haze-optimised transform; gaps "
        "filled by linear interpolation across months")
    path.write_text(json.dumps(m, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(add_help=False)
    ap.add_argument("--only-chips-of", type=Path, default=None,
                    help="export only the chips of this chip set (its chips/*_merged.tif), e.g. a sample")
    own, rest = ap.parse_known_args()
    parent = json.loads((PARENT / "manifest.json").read_text())
    parent_ids = [c["chip_id"] for c in parent["chips"]]
    parent_cells = {(int(c["col"]), int(c["row"])) for c in parent["chips"]}
    link_unchanged(parent_ids)
    if own.only_chips_of is not None:
        sub_dir = own.only_chips_of if own.only_chips_of.is_absolute() else REPO / own.only_chips_of
        wanted = {p.name[: -len("_merged.tif")] for p in (sub_dir / "chips").glob("*_merged.tif")}
        parent_cells = {(int(i.split("_")[1]), int(i.split("_")[2])) for i in wanted if i in set(parent_ids)}
        print(f"restricted to the {len(parent_cells)} chips of {sub_dir.name}", flush=True)
    mod = load_export_script(parent_cells)
    sys.argv = [sys.argv[0], "--country", "EE", "--year", "2021", "--out", str(OUT), *rest]
    mod.main()
    stamp_manifest()


if __name__ == "__main__":
    main()
