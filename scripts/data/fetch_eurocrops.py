"""Fetch the EuroCrops vector release: declared parcel polygons and HCAT crop labels.

EuroCrops version 11 (Zenodo record 14094196, concept DOI 10.5281/zenodo.6866846,
CC-BY-SA-4.0) ships one archive per country, each holding the national self-declared
parcel geometries with three harmonisation attributes appended:

    EC_trans_n  the original national crop name translated into English
    EC_hcat_n   the machine-readable HCAT3 crop name
    EC_hcat_c   the ten-digit HCAT3 code, whose digits encode the taxonomy hierarchy

The release is a single declaration year per country, not a time series. Estonia,
Latvia, Lithuania and Portugal are all 2021, the season the Sentinel-2 composites of
this thesis target.

Three stages, each idempotent and individually selectable through --stages:

    download   pull the archives, the national mapping tables and the HCAT tables,
               verifying every file against the md5 checksum published by Zenodo
    extract    unpack each archive under vector/<CC>/
    parquet    write one GeoParquet per country under parquet/<CC>_<year>.parquet,
               keeping every native attribute and the native CRS untouched, and a
               sidecar JSON summary recording what arrived

Nothing here filters, reprojects or reclassifies. Interpretation belongs downstream.

Usage:
    python scripts/data/fetch_eurocrops.py --countries EE LV LT PT
    python scripts/data/fetch_eurocrops.py --countries EE --stages parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

import requests

ZENODO_RECORD = "14094196"  # EuroCrops version 11
RECORD_DOI = "10.5281/zenodo.14094196"
CONCEPT_DOI = "10.5281/zenodo.6866846"
LICENCE = "CC-BY-SA-4.0"
API_RECORD = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

# country code -> (vector archive, national mapping table, declaration year)
COUNTRIES: dict[str, tuple[str, str, int]] = {
    "EE": ("EE_2021.zip", "ee_2021.csv", 2021),
    "LV": ("LV_2021.zip", "lv_2021.csv", 2021),
    "LT": ("LT_2021.zip", "lt_2021.csv", 2021),
    "PT": ("PT_2021.zip", "pt_2021.csv", 2021),
}
TAXONOMY_FILES = ("HCAT3.csv", "HCAT2.csv")

DEFAULT_DEST = Path(__file__).resolve().parents[2] / "data" / "eurocrops"
CHUNK = 8 * 1024 * 1024


# --------------------------------------------------------------------------- utils


def md5sum(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GB"


def record_metadata(dest: Path, refresh: bool = False) -> dict:
    """Zenodo record metadata, cached on disk as the provenance record."""
    cached = dest / f"zenodo_record_{ZENODO_RECORD}.json"
    if cached.exists() and not refresh:
        return json.loads(cached.read_text())
    resp = requests.get(API_RECORD, timeout=60)
    resp.raise_for_status()
    meta = resp.json()
    cached.write_text(json.dumps(meta, indent=2))
    return meta


# ------------------------------------------------------------------------ download


def download_file(key: str, expected_md5: str, out: Path) -> str:
    """Download one file of the record unless a verified copy is already present."""
    if out.exists():
        if md5sum(out) == expected_md5:
            print(f"  [have] {key:<16} {human(out.stat().st_size)}")
            return "cached"
        print(f"  [bad ] {key:<16} checksum mismatch, refetching")
        out.unlink()

    url = f"{API_RECORD}/files/{key}/content"
    part = out.with_suffix(out.suffix + ".part")
    started = time.time()
    with requests.get(url, stream=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with part.open("wb") as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    pct = 100 * done / total
                    print(f"\r  [get ] {key:<16} {pct:5.1f}% of {human(total)}", end="", flush=True)
    print()
    got = md5sum(part)
    if got != expected_md5:
        part.unlink()
        raise RuntimeError(f"{key}: md5 {got} does not match published {expected_md5}")
    part.rename(out)
    elapsed = time.time() - started
    rate = out.stat().st_size / max(elapsed, 1e-6)
    print(f"  [ok  ] {key:<16} {human(out.stat().st_size)} in {elapsed:.0f} s ({human(rate)}/s)")
    return "downloaded"


def stage_download(countries: list[str], dest: Path) -> None:
    raw = dest / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    meta = record_metadata(dest)
    checksums = {f["key"]: f["checksum"].split(":", 1)[1] for f in meta["files"]}

    wanted: list[str] = []
    for cc in countries:
        archive, mapping, _ = COUNTRIES[cc]
        wanted += [archive, mapping]
    wanted += list(TAXONOMY_FILES)

    print(f"download -> {raw}")
    for key in wanted:
        if key not in checksums:
            raise KeyError(f"{key} is not part of Zenodo record {ZENODO_RECORD}")
        download_file(key, checksums[key], raw / key)


# ------------------------------------------------------------------------- extract


def stage_extract(countries: list[str], dest: Path) -> None:
    raw = dest / "raw"
    print(f"extract  -> {dest / 'vector'}")
    for cc in countries:
        archive, _, _ = COUNTRIES[cc]
        src = raw / archive
        out = dest / "vector" / cc
        if not src.exists():
            raise FileNotFoundError(f"{src} missing, run the download stage first")
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(src) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
            already = all((out / m).exists() for m in members)
            if already:
                print(f"  [have] {cc:<3} {len(members)} members")
                continue
            zf.extractall(out, members=members)
        size = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
        print(f"  [ok  ] {cc:<3} {len(members)} members, {human(size)}")


# ------------------------------------------------------------------------- parquet


def find_layer(country_dir: Path) -> Path:
    """The vector file inside an extracted country archive."""
    candidates = sorted(
        [p for p in country_dir.rglob("*") if p.suffix.lower() in {".shp", ".gpkg", ".geojson"}],
        key=lambda p: (p.suffix.lower() != ".gpkg", -p.stat().st_size),
    )
    if not candidates:
        raise FileNotFoundError(f"no .shp, .gpkg or .geojson under {country_dir}")
    return candidates[0]


def summarise(gdf, source: Path, cc: str, year: int) -> dict:
    counts = (
        gdf["EC_hcat_n"].value_counts().head(25).to_dict() if "EC_hcat_n" in gdf.columns else {}
    )
    id_like = [
        c
        for c in gdf.columns
        if c != gdf.geometry.name and gdf[c].is_unique and gdf[c].notna().all()
    ]
    return {
        "country": cc,
        "declaration_year": year,
        "source_layer": str(source),
        "zenodo_record": ZENODO_RECORD,
        "record_doi": RECORD_DOI,
        "concept_doi": CONCEPT_DOI,
        "licence": LICENCE,
        "n_parcels": int(len(gdf)),
        "crs": str(gdf.crs),
        "crs_epsg": gdf.crs.to_epsg() if gdf.crs else None,
        "total_bounds": [float(v) for v in gdf.total_bounds],
        "geometry_types": {k: int(v) for k, v in gdf.geom_type.value_counts().items()},
        "invalid_geometries": int((~gdf.geometry.is_valid).sum()),
        "empty_geometries": int(gdf.geometry.is_empty.sum()),
        "columns": {c: str(gdf[c].dtype) for c in gdf.columns},
        "unique_id_candidates": id_like,
        "n_hcat_classes": int(gdf["EC_hcat_c"].nunique()) if "EC_hcat_c" in gdf.columns else None,
        "top_hcat_classes": {str(k): int(v) for k, v in counts.items()},
    }


def stage_parquet(countries: list[str], dest: Path) -> None:
    import geopandas as gpd

    out_dir = dest / "parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"parquet  -> {out_dir}")
    for cc in countries:
        _, _, year = COUNTRIES[cc]
        target = out_dir / f"{cc}_{year}.parquet"
        summary_path = out_dir / f"{cc}_{year}.summary.json"
        if target.exists() and summary_path.exists():
            print(f"  [have] {cc:<3} {human(target.stat().st_size)}")
            continue
        layer = find_layer(dest / "vector" / cc)
        started = time.time()
        gdf = gpd.read_file(layer, engine="pyogrio")
        gdf.to_parquet(
            target,
            compression="zstd",
            geometry_encoding="WKB",
            write_covering_bbox=True,
            schema_version="1.1.0",
        )
        summary = summarise(gdf, layer.relative_to(dest), cc, year)
        summary["parquet_bytes"] = target.stat().st_size
        summary_path.write_text(json.dumps(summary, indent=2))
        print(
            f"  [ok  ] {cc:<3} {summary['n_parcels']:>9,} parcels, {summary['crs']}, "
            f"{human(summary['parquet_bytes'])} in {time.time() - started:.0f} s"
        )


# ---------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--countries", nargs="+", default=["EE", "LV", "LT", "PT"], choices=sorted(COUNTRIES))
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--stages", nargs="+", default=["download", "extract", "parquet"],
                        choices=["download", "extract", "parquet"])
    args = parser.parse_args(argv)

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"EuroCrops v11, Zenodo {ZENODO_RECORD} ({RECORD_DOI}), {LICENCE}")
    print(f"countries: {', '.join(args.countries)}")

    if "download" in args.stages:
        stage_download(args.countries, dest)
    if "extract" in args.stages:
        stage_extract(args.countries, dest)
    if "parquet" in args.stages:
        stage_parquet(args.countries, dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
