"""Add AlphaEarth Foundations annual embeddings to an existing chip set, on its label grid.

For every chip in ``<root>/manifest.json`` this writes ``<chip>_alphaearth.tif`` next to the
chip's image, mask and parcel raster: the 64 de-quantised bands ``A00`` to ``A63`` as float32,
EPSG:3035, the chip's exact transform. The chips are cut from the tiles that
``fetch_alphaearth_tiles.py`` downloaded, so that script runs first; nothing here touches the
network. Provenance and the per-dimension normalisation statistics (training chips only) go
to ``<root>/alphaearth_v1.json``, through the export shared with TESSERA,
:func:`gfm4agri.embeddings.export.export_chips`, which documents the layout and the resume.

    poetry run python scripts/hub/fetch_alphaearth_tiles.py --root data/eurocrops_chips/EE_2021_pilot
    poetry run python scripts/hub/build_alphaearth_chips.py --root data/eurocrops_chips/EE_2021_pilot
    poetry run python scripts/hub/build_alphaearth_chips.py --root data/eurocrops_chips/EE_2021 \
        --workers 16 --split-dir data/eurocrops_chips/EE_2021/splits/<split>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.embeddings.alphaearth import (  # noqa: E402
    BAND_NAMES, MIRROR_URL, MODEL_VERSION, OFFICIAL_URL, AlphaEarthStore)
from gfm4agri.embeddings.export import EmbeddingExport, export_chips, write_stats  # noqa: E402


def spec(year: int, tiles: Path) -> EmbeddingExport:
    return EmbeddingExport(
        name="alphaearth_v1", suffix="_alphaearth.tif", band_names=BAND_NAMES,
        source={"dataset": f"GOOGLE/SATELLITE_EMBEDDING/{MODEL_VERSION}/ANNUAL",
                "files": OFFICIAL_URL,
                "fetched_from": f"{MIRROR_URL} or {OFFICIAL_URL}, per tile in the tile table, "
                                "each MD5-checked against Google's",
                "tile_table": str(tiles.relative_to(REPO) if tiles.is_relative_to(REPO) else tiles)
                              + f"/{year}/tiles.json",
                "model": "AlphaEarth Foundations (Brown et al., 2025)", "year": year,
                "product": "annual embedding: one 64-d unit vector per 10 m pixel summarising "
                           "the calendar year of multi-source Earth observation",
                "licence": "CC-BY 4.0; The AlphaEarth Foundations Satellite Embedding dataset "
                           "is produced by Google and Google DeepMind."},
        resampling="nearest neighbour from the tiles' native UTM grid; int8 de-quantised as "
                   "(q / 127.5)^2 sign(q) before warping, not renormalised; the bottom-up "
                   "files flipped to north-up on read; masked (-128) and uncovered pixels are NaN",
        retries=0)   # local files: a failure is not transient


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=REPO / "data/eurocrops_chips/EE_2021_pilot")
    ap.add_argument("--tiles", type=Path, default=REPO / "data/alphaearth/aef_v1_annual",
                    help="directory fetch_alphaearth_tiles.py wrote to")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--split-dir", type=Path, default=None,
                    help="spatial split whose training chips give the normalisation statistics")
    ap.add_argument("--stats-only", action="store_true",
                    help="recompute the statistics of an existing sidecar, read no tile")
    args = ap.parse_args()

    root = args.root if args.root.is_absolute() else REPO / args.root
    tiles = args.tiles if args.tiles.is_absolute() else REPO / args.tiles
    split_dir = None
    if args.split_dir is not None:
        split_dir = args.split_dir if args.split_dir.is_absolute() else REPO / args.split_dir
    manifest = json.loads((root / "manifest.json").read_text())
    year, crs = manifest["year"], manifest["grid"]["crs"]
    px, pixel_m = manifest["grid"]["chip_px"], manifest["grid"]["pixel_m"]

    if args.stats_only:
        write_stats(spec(year, tiles), root, split_dir)
        return

    def make_reader():
        store = AlphaEarthStore(tiles, year)
        return lambda c: store.read_chip(tuple(c["bounds_3035"]), crs, year, size_px=px,
                                         pixel_m=pixel_m)

    export_chips(spec(year, tiles), root, make_reader, workers=args.workers,
                 overwrite=args.overwrite, split_dir=split_dir)


if __name__ == "__main__":
    main()
