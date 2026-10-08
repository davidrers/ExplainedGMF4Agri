"""Build a small pilot set of EuroCrops segmentation chips for wiring up TerraTorch.

The pilot has the on-disk layout of the ``multi-temporal-crop-classification`` rig, so the
same loading code serves both, but the content of the thesis dataset: Estonian 2021
EuroCrops parcels rasterised onto 224 x 224 Sentinel-2 L2A chips at 10 m, twelve monthly
composites of twelve bands.

    <out>/
      training_chips/<chip>_merged.tif    int16, 144 bands, time-major (month, band)
      training_chips/<chip>.mask.tif      int16, class 0 .. n-1, -1 = ignore
      training_chips/<chip>.parcels.tif   int32, EuroCrops parcel id, 0 = no parcel
      validation_chips/...                the same
      training_data.txt, validation_data.txt   chip ids, as the rig's split files
      manifest.json                       classes, bands, grid, stats, provenance per chip

It is a pipeline test set, not a Phase 1 split: labels are dense in every chip, and the
train and validation chips are separated only by the minimum chip spacing, not by the block
assignment of :mod:`gfm4agri.data.blocks`.

    poetry run python scripts/data/build_pilot_chips.py --n-train 8 --n-val 4
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from gfm4agri.chips.s2_monthly import BAND_NAMES, BANDS, N_MONTHS, monthly_composite  # noqa: E402
from gfm4agri.data.chip_grid import (  # noqa: E402
    CHIP_CRS, CHIP_PX, IGNORE_INDEX, PIXEL_M, apply_vector_aliases, rasterise_labels,
    score_cells, select_chips, vector_layer_aliases)

COUNTRY = {"EE": ("Estonia", "pollu_id")}
SPLIT_DIRS = {"train": "training", "val": "validation"}


def load_classes(country: str) -> tuple[list[str], list[str]]:
    import yaml

    scheme = yaml.safe_load((REPO / "configs" / "class_scheme_eurocropsml.yaml").read_text())
    codes = [str(c) for c in scheme["in_country_class_lists"][country]]
    names = {str(c["hcat"]): c["thesis_class"] for c in scheme["classes"]}
    return codes, [names[c] for c in codes]


def write_tif(path: Path, arr: np.ndarray, chip, descriptions: list[str] | None = None) -> None:
    import rasterio

    arr = arr if arr.ndim == 3 else arr[None]
    profile = dict(driver="GTiff", width=CHIP_PX, height=CHIP_PX, count=arr.shape[0],
                   dtype=arr.dtype, crs=CHIP_CRS, transform=chip.transform,
                   compress="deflate", predictor=2, tiled=True, blockxsize=CHIP_PX, blockysize=CHIP_PX)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr)
        for i, d in enumerate(descriptions or [], start=1):
            dst.set_band_description(i, d)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--country", default="EE", choices=sorted(COUNTRY))
    ap.add_argument("--year", type=int, default=2021)
    ap.add_argument("--n-train", type=int, default=8)
    ap.add_argument("--n-val", type=int, default=4)
    ap.add_argument("--min-label-share", type=float, default=0.45)
    ap.add_argument("--min-sep-km", type=float, default=15.0)
    ap.add_argument("--cloud-cover-max", type=float, default=80.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import geopandas as gpd

    cc = args.country
    country, id_col = COUNTRY[cc]
    out = args.out or REPO / "data" / "eurocrops_chips" / f"{cc}_{args.year}_pilot"
    for d in SPLIT_DIRS.values():
        (out / f"{d}_chips").mkdir(parents=True, exist_ok=True)

    codes, names = load_classes(country)
    t0 = time.time()
    parcels = gpd.read_parquet(REPO / "data" / "eurocrops" / "parquet" / f"{cc}_{args.year}.parquet",
                               columns=[id_col, "EC_hcat_c", "geometry"]).to_crs(CHIP_CRS)
    parcels["geometry"] = parcels.geometry.make_valid()
    import yaml

    scheme = yaml.safe_load((REPO / "configs" / "class_scheme_eurocropsml.yaml").read_text())
    parcels["EC_hcat_c"] = apply_vector_aliases(parcels["EC_hcat_c"], vector_layer_aliases(scheme))
    print(f"{len(parcels):,} parcels loaded and projected in {time.time() - t0:.0f} s")

    cells = score_cells(parcels, "EC_hcat_c", codes)
    n = args.n_train + args.n_val
    chips = select_chips(cells, cc, n, min_label_share=args.min_label_share,
                         min_sep_m=args.min_sep_km * 1000, seed=args.seed)
    # Validation chips are every third pick, so both splits span the ranking.
    order = list(range(n))
    val_idx = set(order[2::3][: args.n_val])
    split_of = {c.chip_id: ("val" if i in val_idx else "train") for i, c in enumerate(chips)}

    manifest = {
        "dataset": f"eurocrops_s2_seg_{cc}_{args.year}_pilot",
        "purpose": "pipeline test set for the TerraTorch connection; not a Phase 1 split",
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "config": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "country": cc, "year": args.year,
        "label_source": f"data/eurocrops/parquet/{cc}_{args.year}.parquet (EuroCrops v11)",
        "class_scheme": "configs/class_scheme_eurocropsml.yaml, in_country_class_lists." + country,
        "classes": [{"index": i, "hcat_code": c, "name": nm} for i, (c, nm) in enumerate(zip(codes, names))],
        "ignore_index": IGNORE_INDEX,
        "label_rule": "pixel centre inside an in-scheme parcel; all other pixels ignore_index; dense",
        "grid": {"crs": CHIP_CRS, "pixel_m": PIXEL_M, "chip_px": CHIP_PX,
                 "chip_id": "<CC>_<col>_<row>, cell of a 2240 m grid anchored on the EPSG:3035 origin"},
        "imagery": {"source": "Planetary Computer sentinel-2-l2a", "bands": BANDS,
                    "band_names": BAND_NAMES, "n_months": N_MONTHS,
                    "layout": "(time channels): band index = month_index * 12 + band_index",
                    "units": "surface reflectance x 10000, BOA offset removed (DN / 10000 convention)",
                    "composite": "per-pixel monthly median of SCL-clear acquisitions, gaps filled "
                                 "by linear interpolation across months",
                    "resampling": "bilinear for spectral bands, nearest for SCL"},
        "chips": [],
    }

    previous = {c["chip_id"]: c.get("imagery", {}) for c in _previous(out)}
    for i, chip in enumerate(chips, start=1):
        split = split_of[chip.chip_id]
        folder = out / f"{SPLIT_DIRS[split]}_chips"
        img_path = folder / f"{chip.chip_id}_merged.tif"
        print(f"[{i}/{n}] {chip.chip_id} ({split})", flush=True)
        mask, ids = rasterise_labels(chip, parcels, "EC_hcat_c", id_col, codes)
        if img_path.exists():
            import rasterio
            with rasterio.open(img_path) as src:
                img = src.read()
            report = previous.get(chip.chip_id, {})
            print("    image exists, reused")
        else:
            t1 = time.time()
            img, report = monthly_composite(chip.bounds, args.year,
                                            cloud_cover_max=args.cloud_cover_max, verbose=True)
            report["seconds"] = round(time.time() - t1, 1)
            desc = [f"M{m + 1:02d}_{b}" for m in range(N_MONTHS) for b in BAND_NAMES]
            write_tif(img_path, img, chip, desc)
        write_tif(folder / f"{chip.chip_id}.mask.tif", mask, chip)
        write_tif(folder / f"{chip.chip_id}.parcels.tif", ids, chip)

        counts = np.bincount(mask[mask >= 0], minlength=len(codes))
        manifest["chips"].append({
            "chip_id": chip.chip_id, "split": split, "bounds_3035": list(chip.bounds),
            "labelled_share": round(float((mask >= 0).mean()), 4),
            "n_parcels": int(len(np.unique(ids[ids > 0]))),
            "pixels_per_class": {names[k]: int(v) for k, v in enumerate(counts) if v},
            "imagery": report,
        })
        (out / "manifest.json").write_text(json.dumps(manifest, indent=2))

    for split, d in SPLIT_DIRS.items():
        ids_ = [c["chip_id"] for c in manifest["chips"] if c["split"] == split]
        (out / f"{d}_data.txt").write_text("\n".join(ids_) + "\n")

    manifest["normalisation"] = band_stats(out, [c["chip_id"] for c in manifest["chips"]
                                                 if c["split"] == "train"])
    totals = np.zeros(len(codes), dtype=np.int64)
    for c in manifest["chips"]:
        for k, v in c["pixels_per_class"].items():
            totals[names.index(k)] += v
    manifest["pixels_per_class_total"] = {nm: int(t) for nm, t in zip(names, totals)}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"done in {(time.time() - t0) / 60:.1f} min -> {out}")


def _previous(out: Path) -> list[dict]:
    try:
        return json.loads((out / "manifest.json").read_text())["chips"]
    except (OSError, ValueError, KeyError):
        return []


def band_stats(out: Path, train_ids: list[str]) -> dict:
    """Per-band mean and standard deviation over all months of the training chips."""
    import rasterio

    s = np.zeros(len(BANDS)); s2 = np.zeros(len(BANDS)); n = 0
    for cid in train_ids:
        with rasterio.open(out / "training_chips" / f"{cid}_merged.tif") as src:
            a = src.read().astype(np.float64).reshape(N_MONTHS, len(BANDS), -1)
        s += a.sum((0, 2)); s2 += (a**2).sum((0, 2)); n += a.shape[0] * a.shape[2]
    mean = s / n
    std = np.sqrt(s2 / n - mean**2)
    return {"computed_on": "training chips, all months pooled",
            "band_names": BAND_NAMES, "means": [round(x, 2) for x in mean],
            "stds": [round(x, 2) for x in std]}


if __name__ == "__main__":
    main()
