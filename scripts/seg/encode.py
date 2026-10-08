"""Stage 1 of the two-stage workflow: encode every chip once with the frozen encoder.

Reads a fit configuration, so the backbone, the chip set, the spatial split, the
normalisation and the autocast precision are exactly those the decoder fits will use, and
writes the features at the cache point to ``<cache-root>/<backbone>/<chip set>/``. Training
chips are encoded in all eight D4 variants, validation and test chips once. A chip already
cached is skipped, so an interrupted run resumes.

    poetry run python scripts/seg/encode.py -c configs/seg/terramind_v1_large_ee.yaml
    poetry run python scripts/seg/encode.py -c configs/seg/prithvi_eo_v2_600_tl_ee.yaml --batch-size 2
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))


def cache_dir_for(cfg: dict, cache_root: Path) -> Path:
    """``<cache-root>/<backbone>/<chip set>``; the split is recorded in ``cache.json``."""
    return cache_root / cfg["model"]["backbone"] / Path(cfg["data"]["root"]).name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("-c", "--config", type=Path, required=True)
    ap.add_argument("--cache-root", type=Path, default=REPO / "data" / "embeddings")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=8)
    args = ap.parse_args()

    from gfm4agri.benchmark.cached import generate_embeddings

    cfg = yaml.safe_load(args.config.read_text())
    d = cfg["data"]
    root = REPO / d["root"]
    split_dir = REPO / d["split"] if d.get("split") else None
    cache_root = args.cache_root if args.cache_root.is_absolute() else REPO / args.cache_root
    cache_dir = cache_dir_for(cfg, cache_root)
    print(f"[stage 1] {cfg['model']['backbone']} on {root.name}"
          + (f", split {split_dir.name}" if split_dir else "") + f" -> {cache_dir}", flush=True)
    t0 = time.time()
    record = generate_embeddings(cfg["model"]["backbone"], root, cache_dir,
                                 normalisation=d["normalisation"],
                                 batch_size=args.batch_size, num_workers=args.num_workers,
                                 precision=cfg["trainer"]["precision"], split_dir=split_dir)
    print(json.dumps({k: record[k] for k in ("channel_list", "encode_seconds", "chip_encodings",
                                              "bytes")}, indent=1))
    print(f"[stage 1] done in {(time.time() - t0) / 3600:.2f} h, "
          f"{record['bytes'] / 1e9:.0f} GB -> {cache_dir / 'cache.json'}", flush=True)


if __name__ == "__main__":
    main()
