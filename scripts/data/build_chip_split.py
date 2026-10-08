"""Build the spatial block split of a full-country chip set.

Writes ``<root>/splits/<name>__<hash8>/``:

    split.json            configuration, its hash, counts per partition and per class, buffer cost
    training_data.txt     pool chips holding at least one trainable parcel
    validation_data.txt   chips of the validation blocks
    test_data.txt         chips of the test blocks
    chips.csv             every chip: block, partition, labelled parcels, trainable parcels
    buffer_parcels.npy    pool parcels withheld from training by the buffer, sorted int64
    map.png               the partition on the chip grid

and, once per chip set, ``<root>/chip_parcels.parquet``: one row per ``(chip, parcel, class)``
with the parcel's labelled pixels in that chip, read from the masks and parcel rasters. It is
shared by every split and by every label budget draw.

    poetry run python scripts/data/build_chip_split.py --root data/eurocrops_chips/EE_2021
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.data.chip_split import (  # noqa: E402
    ChipSplitConfig, assign_chip_partitions, block_of_chip, buffered_parcels, chip_parcel_table)

PARCEL_TABLE = "chip_parcels.parquet"
#: The protocol's ``min_test_per_class``: test parcels a class needs for a meaningful per-class F1.
MIN_TEST_PER_CLASS = 200


def read_labels(folder: Path, chip_id: str) -> tuple[np.ndarray, np.ndarray]:
    import rasterio

    with rasterio.open(folder / f"{chip_id}.mask.tif") as m, \
         rasterio.open(folder / f"{chip_id}.parcels.tif") as p:
        return m.read(1), p.read(1)


def parcel_table(root: Path, chip_ids: list[str], manifest_sha: str, workers: int) -> pd.DataFrame:
    """The ``(chip, parcel, class)`` table, built once per chip set and reused."""
    path = root / PARCEL_TABLE
    if path.exists() and _sha_of(path) == manifest_sha:
        return pd.read_parquet(path)
    t0 = time.time()
    folder = root / "chips"

    def one(cid: str) -> pd.DataFrame:
        return chip_parcel_table(cid, *read_labels(folder, cid))

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(one, chip_ids))
    table = pd.concat(parts, ignore_index=True)
    table.to_parquet(path, index=False)
    (root / f"{PARCEL_TABLE}.sha256").write_text(manifest_sha + "\n")
    print(f"parcel table: {len(table):,} rows from {len(chip_ids):,} chips in {time.time() - t0:.0f} s")
    return table


def _sha_of(path: Path) -> str:
    side = path.with_name(path.name + ".sha256")
    return side.read_text().strip() if side.exists() else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=REPO / "data/eurocrops_chips/EE_2021")
    ap.add_argument("--name", default=None, help="split name; default blocks<N>_buf<m>_seed<s>")
    ap.add_argument("--block-chips", type=int, default=4)
    ap.add_argument("--test-fraction", type=float, default=0.2)
    ap.add_argument("--val-fraction", type=float, default=0.1)
    ap.add_argument("--buffer-m", type=float, default=1600.0)
    ap.add_argument("--protocol-seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()

    root = args.root if args.root.is_absolute() else REPO / args.root
    manifest = json.loads((root / "manifest.json").read_text())
    manifest_sha = hashlib.sha256((root / "manifest.json").read_bytes()).hexdigest()
    name = args.name or f"blocks{args.block_chips}_buf{args.buffer_m:g}_seed{args.protocol_seed}"
    cfg = ChipSplitConfig(name=name, country=manifest["country"], year=manifest["year"],
                          block_chips=args.block_chips, test_fraction=args.test_fraction,
                          val_fraction=args.val_fraction, buffer_m=args.buffer_m,
                          protocol_seed=args.protocol_seed, manifest_sha256=manifest_sha)
    out = root / "splits" / cfg.dir_name
    out.mkdir(parents=True, exist_ok=True)

    chips = pd.DataFrame([{"chip_id": c["chip_id"], "col": c["col"], "row": c["row"]}
                          for c in manifest["chips"]])
    parcels = parcel_table(root, chips["chip_id"].tolist(), manifest_sha, args.workers)
    partition = assign_chip_partitions(chips, parcels, cfg)
    chips["partition"] = partition.to_numpy()
    bx, by = block_of_chip(chips["col"].to_numpy(), chips["row"].to_numpy(), cfg.block_chips)
    chips["block_id"] = [f"{a}_{b}" for a, b in zip(bx, by)]

    folder = root / "chips"
    t0 = time.time()
    withheld, buffer_report = buffered_parcels(chips, partition, parcels,
                                               lambda cid: read_labels(folder, cid), cfg)
    print(f"buffer: {len(withheld):,} pool parcels withheld in {time.time() - t0:.0f} s")

    part_of = dict(zip(chips["chip_id"], chips["partition"]))
    parcels = parcels.assign(partition=parcels["chip_id"].map(part_of),
                             withheld=parcels["parcel_id"].isin(withheld))
    pool = parcels[parcels["partition"] == "pool"]
    trainable = pool[~pool["withheld"]]
    n_label = parcels.groupby("chip_id")["parcel_id"].nunique()
    n_train = trainable.groupby("chip_id")["parcel_id"].nunique()
    chips["labelled_parcels"] = chips["chip_id"].map(n_label).fillna(0).astype(int)
    chips["trainable_parcels"] = chips["chip_id"].map(n_train).fillna(0).astype(int)
    chips.loc[chips["partition"] != "pool", "trainable_parcels"] = 0

    lists = {
        "training": chips[(chips["partition"] == "pool") & (chips["trainable_parcels"] > 0)],
        "validation": chips[chips["partition"] == "val"],
        "test": chips[chips["partition"] == "test"],
    }
    for split, frame in lists.items():
        (out / f"{split}_data.txt").write_text("\n".join(sorted(frame["chip_id"])) + "\n")
    chips.to_csv(out / "chips.csv", index=False)
    np.save(out / "buffer_parcels.npy", np.array(sorted(withheld), dtype=np.int64))

    # A parcel's class is the majority class of its labelled pixels across the chips.
    def majority(frame: pd.DataFrame) -> pd.Series:
        g = frame.groupby(["parcel_id", "class_index"])["pixels"].sum().reset_index()
        g = g.sort_values(["parcel_id", "pixels"], ascending=[True, False])
        return g.drop_duplicates("parcel_id").set_index("parcel_id")["class_index"]

    classes = [c["name"] for c in manifest["classes"]]
    per_class = {}
    counts = {
        "pool_trainable": majority(trainable).value_counts(),
        "pool_withheld": majority(pool[pool["withheld"]]).value_counts(),
        "val": majority(parcels[parcels["partition"] == "val"]).value_counts(),
        "test": majority(parcels[parcels["partition"] == "test"]).value_counts(),
    }
    for k, cname in enumerate(classes):
        per_class[cname] = {key: int(v.get(k, 0)) for key, v in counts.items()}
    thin = [c for c, v in per_class.items() if v["test"] < MIN_TEST_PER_CLASS]

    def summary(mask: pd.Series) -> dict:
        sub = parcels[mask]
        return {"parcels": int(sub["parcel_id"].nunique()), "labelled_pixels": int(sub["pixels"].sum())}

    part_summary = {}
    for p in ("test", "val", "pool", "empty"):
        c = chips[chips["partition"] == p]
        part_summary[p] = {"chips": int(len(c)), "blocks": int(c["block_id"].nunique()),
                           **summary(parcels["partition"] == p)}
    pool_all = summary(parcels["partition"] == "pool")
    pool_kept = summary((parcels["partition"] == "pool") & ~parcels["withheld"])
    total = summary(parcels["partition"].isin(["test", "val", "pool"]))

    record = {
        "split": cfg.dir_name,
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": asdict(cfg), "config_hash": cfg.config_hash(),
        "block_size_m": cfg.block_size_m,
        "protocol": (f"spatial blocks of {cfg.block_chips} x {cfg.block_chips} chips "
                     f"({cfg.block_size_m / 1000:g} km) on the EPSG:3035 grid, whole blocks assigned "
                     f"to test/val/pool at {cfg.test_fraction:g}/{cfg.val_fraction:g}/"
                     f"{1 - cfg.test_fraction - cfg.val_fraction:g} of parcels, protocol seed "
                     f"{cfg.protocol_seed}; pool parcels within {cfg.buffer_m:g} m of a held-out block "
                     "withheld from training (ignore_index), pool chips kept"),
        "chip_set": str(root.relative_to(REPO)), "manifest_sha256": manifest_sha,
        "partitions": part_summary,
        "parcel_shares": {p: round(part_summary[p]["parcels"] / total["parcels"], 4)
                          for p in ("test", "val", "pool")},
        "buffer": {**buffer_report,
                   "withheld_parcels": len(withheld),
                   "pool_parcels_lost_share": round(1 - pool_kept["parcels"] / pool_all["parcels"], 4),
                   "pool_pixels_lost_share": round(
                       1 - pool_kept["labelled_pixels"] / pool_all["labelled_pixels"], 4)},
        "pool_chips_without_trainable_parcel": int(((chips["partition"] == "pool")
                                                    & (chips["trainable_parcels"] == 0)).sum()),
        "lists": {k: int(len(v)) for k, v in lists.items()},
        "per_class_parcels": per_class,
        "classes_below_min_test": {"threshold": MIN_TEST_PER_CLASS, "classes": thin,
                                   "action": "recorded only; the eligibility filter under "
                                             "segmentation is an open protocol decision"},
    }
    (out / "split.json").write_text(json.dumps(record, indent=2))
    draw_map(chips, out / "map.png", cfg)
    print(json.dumps({k: record[k] for k in ("split", "partitions", "parcel_shares", "buffer",
                                              "lists", "classes_below_min_test")}, indent=2))
    print(f"-> {out}")


def draw_map(chips: pd.DataFrame, path: Path, cfg: ChipSplitConfig) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    colours = {"pool": "#9bbf85", "val": "#e7b04b", "test": "#c8553d", "empty": "#cccccc"}
    fig, ax = plt.subplots(figsize=(11, 8))
    for p, colour in colours.items():
        c = chips[chips["partition"] == p]
        if p == "pool":
            thin = c[c["trainable_parcels"] < c["labelled_parcels"]]
            ax.scatter(c["col"], c["row"], s=4, marker="s", c=colour, linewidths=0)
            ax.scatter(thin["col"], thin["row"], s=4, marker="s", c="#5d7f4c", linewidths=0)
        else:
            ax.scatter(c["col"], c["row"], s=4, marker="s", c=colour, linewidths=0)
    ax.set_aspect("equal")
    ax.set_xlabel("chip column (2,240 m, EPSG:3035)")
    ax.set_ylabel("chip row")
    ax.set_title(f"{cfg.dir_name}: {cfg.block_chips} x {cfg.block_chips} chip blocks, "
                 f"buffer {cfg.buffer_m:g} m")
    ax.legend(handles=[Patch(color=colours["pool"], label="pool (training)"),
                       Patch(color="#5d7f4c", label="pool chip with buffered parcels"),
                       Patch(color=colours["val"], label="validation"),
                       Patch(color=colours["test"], label="test")],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()
