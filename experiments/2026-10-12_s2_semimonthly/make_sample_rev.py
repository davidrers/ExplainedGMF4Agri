"""Cut ``EE_2021_sample_rev`` from ``EE_2021_rev``: the blocks of ``EE_2021_sample``, as links.

This is the 12-frame counterpart of ``EE_2021_sample_24``: the same chips and split, monthly composites under the
same ``REVISED`` screening. It refuses to run until every chip of the sample has its re-exported composite and
report in ``EE_2021_rev``, which the first pass of the re-export provides.

The manifest is the one file not linked. ``EE_2021_rev``'s manifest is rewritten when its export ends, so the set
gets its own: the parent's entries for its chips and normalisation statistics over 200 of its chips, all months
pooled, as ``EE_2021_sample_24`` has. A set made earlier with a linked manifest gets it replaced.

    poetry run python experiments/2026-10-12_s2_semimonthly/make_sample_rev.py
"""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.data.chip_subset import write_subset  # noqa: E402
from gfm4agri.pipeline.config import resolve_split  # noqa: E402

CHIPS = REPO / "data" / "eurocrops_chips"


def freeze_manifest(dest: Path, rev: Path) -> None:
    import rasterio

    ids = sorted(p.name[: -len("_merged.tif")] for p in (dest / "chips").glob("*_merged.tif"))
    parent = json.loads((rev / "manifest.json").read_text())
    entries = {e["chip_id"]: e for e in parent["chips"]}
    m = {k: v for k, v in parent.items() if k not in ("chips", "normalisation", "pixels_per_class_total")}
    m |= {"dataset": "eurocrops_s2_seg_EE_2021_sample_rev",
          "purpose": "the chips of EE_2021_sample with the monthly composites of EE_2021_rev (REVISED screening), "
                     "the 12-frame counterpart of EE_2021_sample_24",
          "created": dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
          "derived_from": {"chip_set": "data/eurocrops_chips/EE_2021_rev", "blocks_of": "EE_2021_sample",
                           "experiment": "experiments/2026-10-12_s2_semimonthly"},
          "chips": [entries[c] for c in ids]}
    rng = np.random.default_rng(0)
    sample = sorted(rng.choice(ids, size=min(200, len(ids)), replace=False))
    s, s2, n = np.zeros(12), np.zeros(12), 0
    for cid in sample:
        with rasterio.open(dest / "chips" / f"{cid}_merged.tif") as src:
            a = src.read().astype(np.float64).reshape(12, 12, -1)
        s += a.sum((0, 2))
        s2 += (a ** 2).sum((0, 2))
        n += a.shape[0] * a.shape[2]
    mean = s / n
    m["normalisation"] = {"computed_on": f"{len(sample)} chips of the set, all months pooled",
                          "band_names": parent["imagery"]["band_names"], "means": [round(x, 2) for x in mean],
                          "stds": [round(x, 2) for x in np.sqrt(s2 / n - mean ** 2)]}
    path = dest / "manifest.json"
    if path.is_symlink():
        path.unlink()
    path.write_text(json.dumps(m, indent=2))
    print(f"{path.relative_to(REPO)}: {len(ids)} chips, own manifest")


def main() -> None:
    sample = json.loads((CHIPS / "EE_2021_sample" / "subset.json").read_text())
    rev = CHIPS / "EE_2021_rev"
    if (CHIPS / "EE_2021_sample_rev").exists():
        freeze_manifest(CHIPS / "EE_2021_sample_rev", rev)
        return
    ids = sorted(p.name[: -len("_merged.tif")] for p in (CHIPS / "EE_2021_sample" / "chips").glob("*_merged.tif"))
    missing = [c for c in ids if (rev / "chips" / f"{c}_merged.tif").is_symlink()
               or not (rev / "chips" / f"{c}_merged.tif").exists()
               or not (rev / "chips" / f"{c}.report.json").exists()]
    if missing:
        sys.exit(f"{len(missing)} of {len(ids)} sample chips are not re-exported yet, e.g. {missing[:3]}")
    dest = write_subset(rev, CHIPS / "EE_2021_sample_rev", resolve_split(rev, sample["split"]),
                        sample["blocks"], sample["seed"])
    freeze_manifest(dest, rev)
    print(f"{dest.relative_to(REPO)}:", (dest / "subset.json").read_text())


if __name__ == "__main__":
    main()
