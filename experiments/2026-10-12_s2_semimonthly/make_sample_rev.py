"""Cut ``EE_2021_sample_rev`` from ``EE_2021_rev``: the blocks of ``EE_2021_sample``, as links.

This is the 12-frame counterpart of ``EE_2021_sample_24``: the same chips and split, monthly composites under the
same ``REVISED`` screening. It refuses to run until every chip of the sample has its re-exported composite and
report in ``EE_2021_rev``, which the first pass of the re-export provides.

    poetry run python experiments/2026-10-12_s2_semimonthly/make_sample_rev.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.data.chip_subset import write_subset  # noqa: E402
from gfm4agri.pipeline.config import resolve_split  # noqa: E402

CHIPS = REPO / "data" / "eurocrops_chips"


def main() -> None:
    sample = json.loads((CHIPS / "EE_2021_sample" / "subset.json").read_text())
    rev = CHIPS / "EE_2021_rev"
    ids = sorted(p.name[: -len("_merged.tif")] for p in (CHIPS / "EE_2021_sample" / "chips").glob("*_merged.tif"))
    missing = [c for c in ids if (rev / "chips" / f"{c}_merged.tif").is_symlink()
               or not (rev / "chips" / f"{c}_merged.tif").exists()
               or not (rev / "chips" / f"{c}.report.json").exists()]
    if missing:
        sys.exit(f"{len(missing)} of {len(ids)} sample chips are not re-exported yet, e.g. {missing[:3]}")
    dest = write_subset(rev, CHIPS / "EE_2021_sample_rev", resolve_split(rev, sample["split"]),
                        sample["blocks"], sample["seed"])
    print(f"{dest.relative_to(REPO)}:", (dest / "subset.json").read_text())


if __name__ == "__main__":
    main()
