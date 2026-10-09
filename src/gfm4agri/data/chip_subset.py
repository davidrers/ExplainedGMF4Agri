"""A smaller chip set cut from a country chip set: whole spatial blocks of every partition.

The subset is a chip set directory of its own, so the K-shot workflow runs on it exactly as on
a country. ``chips/`` holds relative links into the parent's chips; the manifest, the per-source
records and the chip-parcel table are links to the parent's, so the normalisation statistics
and the manifest hash are the parent's; the split keeps its name and holds the parent's lists
restricted to the chosen chips. ``subset.json`` records the parent, the split and the blocks.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["choose_blocks", "write_subset"]

#: ``chips.csv`` partition -> split list name.
LISTS = {"pool": "training", "val": "validation", "test": "test"}
CHIP_ID = re.compile(r"^[A-Z]{2}_\d+_\d+")


def choose_blocks(chips: pd.DataFrame, n_blocks: dict[str, int], seed: int) -> dict[str, list[str]]:
    """``n_blocks[partition]`` whole blocks per partition, drawn with ``seed``.

    Only blocks that can serve their partition are eligible: pool blocks holding a trainable
    parcel, validation and test blocks holding a labelled one.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for part, n in n_blocks.items():
        col = "trainable_parcels" if part == "pool" else "labelled_parcels"
        per_block = chips[chips["partition"] == part].groupby("block_id")[col].sum()
        eligible = sorted(per_block[per_block > 0].index)
        if len(eligible) < n:
            raise ValueError(f"{n} blocks wanted from {part}, {len(eligible)} eligible")
        out[part] = sorted(rng.choice(eligible, size=n, replace=False).tolist())
    return out


def _link(src: Path, dst: Path) -> None:
    os.symlink(os.path.relpath(src, dst.parent), dst)


def write_subset(parent: Path, dest: Path, split_dir: Path, blocks: dict[str, list[str]],
                 seed: int) -> Path:
    parent, dest, split_dir = Path(parent), Path(dest), Path(split_dir)
    if dest.exists():
        raise FileExistsError(f"{dest} exists; remove it first")
    chips = pd.read_csv(split_dir / "chips.csv", dtype={"block_id": str})
    chosen = {b for bs in blocks.values() for b in bs}
    keep = set(chips.loc[chips["block_id"].isin(chosen), "chip_id"])

    (dest / "chips").mkdir(parents=True)
    for f in sorted(os.listdir(parent / "chips")):
        m = CHIP_ID.match(f)
        if m and m.group(0) in keep:
            _link(parent / "chips" / f, dest / "chips" / f)
    for f in sorted(parent.iterdir()):
        if f.is_file() and f.suffix in (".json", ".parquet", ".sha256"):
            _link(f, dest / f.name)

    out = dest / "splits" / split_dir.name
    out.mkdir(parents=True)
    counts = {}
    for name in LISTS.values():
        src = split_dir / f"{name}_data.txt"
        if src.exists():
            ids = [c for c in src.read_text().split() if c in keep]
            (out / f"{name}_data.txt").write_text("\n".join(ids) + "\n")
            counts[name] = len(ids)
    chips[chips["chip_id"].isin(keep)].to_csv(out / "chips.csv", index=False)
    _link(split_dir / "buffer_parcels.npy", out / "buffer_parcels.npy")
    record = json.loads((split_dir / "split.json").read_text())
    record["lists"] = counts
    record["subset"] = {"parent": str(parent), "parent_split": split_dir.name, "blocks": blocks,
                        "seed": seed}
    (out / "split.json").write_text(json.dumps(record, indent=2))
    (dest / "subset.json").write_text(json.dumps({
        "parent": str(parent), "split": split_dir.name, "blocks": blocks, "seed": seed,
        "chips": counts, "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
    }, indent=2))
    return dest
