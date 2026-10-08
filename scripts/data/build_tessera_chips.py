"""Add TESSERA v1 embeddings to an existing chip set, on the same grid as its labels.

For every chip in ``<root>/manifest.json`` this writes ``<chip>_tessera.tif`` next to the
chip's image, mask and parcel raster: 128 float32 bands, EPSG:3035, the chip's exact
transform. The masks, the parcel rasters and the split files are shared with every other
representation, so a label budget draws the same parcels whatever the model.

Provenance and the per-dimension normalisation statistics (training chips only) go to
``<root>/tessera_v1.json``. ``manifest.json`` is left untouched, because results already
trained on the chips are stamped with its hash. The export itself, shared with AlphaEarth, is
:func:`gfm4agri.embeddings.export.export_chips`.

A pilot chip set carries its split in the manifest and in ``training_chips/`` and
``validation_chips/``. A full-country chip set keeps every chip in ``chips/`` and carries no
split of its own, so the training chips are those of a spatial split directory, passed with
``--split-dir``. Without one the statistics are left out, and ``--stats-only --split-dir``
adds them afterwards without downloading anything. A chip already on disk is skipped, so an
interrupted run resumes.

    poetry run python scripts/data/build_tessera_chips.py
    poetry run python scripts/data/build_tessera_chips.py --root data/eurocrops_chips/EE_2021_pilot --workers 6
    poetry run python scripts/data/build_tessera_chips.py --root data/eurocrops_chips/EE_2021 --workers 16
    poetry run python scripts/data/build_tessera_chips.py --root data/eurocrops_chips/EE_2021 \
        --stats-only --split-dir data/eurocrops_chips/EE_2021/splits/<split>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.embeddings.export import EmbeddingExport, export_chips, write_stats  # noqa: E402
from gfm4agri.embeddings.tessera import MODEL_VERSION, N_DIMS, STORE_URL, TesseraStore  # noqa: E402


def spec(year: int) -> EmbeddingExport:
    return EmbeddingExport(
        name="tessera_v1", suffix="_tessera.tif",
        band_names=[f"TESSERA_{i:03d}" for i in range(N_DIMS)],
        source={"store": STORE_URL, "model_version": MODEL_VERSION,
                "model": "https://geotessera.org/model/1.0", "year": year,
                "product": "annual embedding: one 128-d vector per 10 m pixel from the "
                           "calendar-year Sentinel-1 and Sentinel-2 time series"},
        resampling="nearest neighbour from the store's native UTM grid; int8 x per-pixel "
                   "scale dequantised before warping; no-coverage pixels are NaN")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=REPO / "data/eurocrops_chips/EE_2021_pilot")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--split-dir", type=Path, default=None,
                    help="spatial split whose training chips give the normalisation statistics")
    ap.add_argument("--stats-only", action="store_true",
                    help="recompute the statistics of an existing sidecar, download nothing")
    args = ap.parse_args()

    root = args.root if args.root.is_absolute() else REPO / args.root
    split_dir = None
    if args.split_dir is not None:
        split_dir = args.split_dir if args.split_dir.is_absolute() else REPO / args.split_dir
    manifest = json.loads((root / "manifest.json").read_text())
    year, crs = manifest["year"], manifest["grid"]["crs"]
    px, pixel_m = manifest["grid"]["chip_px"], manifest["grid"]["pixel_m"]

    if args.stats_only:
        write_stats(spec(year), root, split_dir)
        return

    def make_reader():
        store = TesseraStore()
        return lambda c: store.read_chip(tuple(c["bounds_3035"]), crs, year, size_px=px,
                                         pixel_m=pixel_m)

    export_chips(spec(year), root, make_reader, workers=args.workers, overwrite=args.overwrite,
                 split_dir=split_dir)


if __name__ == "__main__":
    main()
