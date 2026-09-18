"""Fetch the EuroCropsML benchmark (Reuss et al., 2025).

Zenodo record 15095445 (concept DOI 10.5281/zenodo.10629609, CC-BY-SA-4.0), the
version published 2025-03-31. Three archives:

    preprocess.zip   1.47 GB   706,683 .npz files, one per parcel, named
                               <NUTS3>_<parcelID>_<EC_hcat_c>.npz, each carrying the
                               cloud-filtered Sentinel-2 series, its dates and the centroid
    raw_data.zip     3.28 GB   one dataframe per country with the unfiltered annual series,
                               plus separate parcel geometry and class files
    split.zip        20.7 MB   the official pre-training and fine-tuning splits, among them
                               latvia_vs_estonia and latvia_portugal_vs_estonia

EuroCropsML is the reference index of this thesis: it supplies the per-parcel time
series and the official benchmark splits, and it joins to the EuroCrops vector release
through the parcel identifier carried in the .npz filename.

Two stages, both idempotent:

    download   pull the archives, verifying every file against the md5 published by Zenodo
    extract    unpack into data/eurocropsml/, the 706,683 small files in parallel

Usage:
    python scripts/data/fetch_eurocropsml.py
    python scripts/data/fetch_eurocropsml.py --files preprocess.zip split.zip
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import requests

ZENODO_RECORD = "15095445"
RECORD_DOI = "10.5281/zenodo.15095445"
CONCEPT_DOI = "10.5281/zenodo.10629609"
LICENCE = "CC-BY-SA-4.0"
API_RECORD = f"https://zenodo.org/api/records/{ZENODO_RECORD}"

ARCHIVES = ("preprocess.zip", "raw_data.zip", "split.zip")
DEFAULT_DEST = Path(__file__).resolve().parents[2] / "data" / "eurocropsml"
CHUNK = 8 * 1024 * 1024


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
    cached = dest / f"zenodo_record_{ZENODO_RECORD}.json"
    if cached.exists() and not refresh:
        return json.loads(cached.read_text())
    resp = requests.get(API_RECORD, timeout=60)
    resp.raise_for_status()
    meta = resp.json()
    cached.write_text(json.dumps(meta, indent=2))
    return meta


def download_file(key: str, expected_md5: str, out: Path) -> None:
    if out.exists() and md5sum(out) == expected_md5:
        print(f"  [have] {key:<16} {human(out.stat().st_size)}", flush=True)
        return
    if out.exists():
        print(f"  [bad ] {key:<16} checksum mismatch, refetching", flush=True)
        out.unlink()

    part = out.with_suffix(out.suffix + ".part")
    started = time.time()
    with requests.get(f"{API_RECORD}/files/{key}/content", stream=True, timeout=(30, 300)) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        with part.open("wb") as fh:
            for chunk in resp.iter_content(CHUNK):
                fh.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  [get ] {key:<16} {100 * done / total:5.1f}% of {human(total)}",
                          end="", flush=True)
    print()
    got = md5sum(part)
    if got != expected_md5:
        part.unlink()
        raise RuntimeError(f"{key}: md5 {got} does not match published {expected_md5}")
    part.rename(out)
    elapsed = time.time() - started
    print(f"  [ok  ] {key:<16} {human(out.stat().st_size)} in {elapsed:.0f} s "
          f"({human(out.stat().st_size / max(elapsed, 1e-6))}/s)", flush=True)


def stage_download(files: list[str], dest: Path) -> None:
    raw = dest / "archives"
    raw.mkdir(parents=True, exist_ok=True)
    checksums = {f["key"]: f["checksum"].split(":", 1)[1] for f in record_metadata(dest)["files"]}
    print(f"download -> {raw}", flush=True)
    for key in files:
        if key not in checksums:
            raise KeyError(f"{key} is not part of Zenodo record {ZENODO_RECORD}")
        download_file(key, checksums[key], raw / key)


def _extract_chunk(args: tuple[str, str, list[str]]) -> int:
    archive, out_dir, members = args
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(out_dir, members=members)
    return len(members)


def stage_extract(files: list[str], dest: Path, workers: int) -> None:
    raw = dest / "archives"
    print(f"extract  -> {dest}", flush=True)
    for key in files:
        src = raw / key
        if not src.exists():
            raise FileNotFoundError(f"{src} missing, run the download stage first")
        with zipfile.ZipFile(src) as zf:
            members = [m for m in zf.namelist() if not m.startswith("__MACOSX")]
        # The archives carry their own top-level directory, so they all extract into dest.
        target_root = dest / Path(members[0]).parts[0]
        n_on_disk = sum(1 for _ in target_root.rglob("*")) if target_root.exists() else 0
        if n_on_disk >= len(members):
            print(f"  [have] {key:<16} {n_on_disk:,} entries under {target_root.name}/", flush=True)
            continue
        started = time.time()
        if len(members) > 10_000:
            size = (len(members) + workers - 1) // workers
            chunks = [(str(src), str(dest), members[i:i + size]) for i in range(0, len(members), size)]
            with ProcessPoolExecutor(max_workers=workers) as pool:
                done = sum(pool.map(_extract_chunk, chunks))
        else:
            with zipfile.ZipFile(src) as zf:
                zf.extractall(dest, members=members)
            done = len(members)
        print(f"  [ok  ] {key:<16} {done:,} entries in {time.time() - started:.0f} s", flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--files", nargs="+", default=list(ARCHIVES), choices=list(ARCHIVES))
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--stages", nargs="+", default=["download", "extract"],
                        choices=["download", "extract"])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args(argv)

    dest = args.dest.expanduser().resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"EuroCropsML, Zenodo {ZENODO_RECORD} ({RECORD_DOI}), {LICENCE}", flush=True)
    print(f"files: {', '.join(args.files)}", flush=True)

    if "download" in args.stages:
        stage_download(args.files, dest)
    if "extract" in args.stages:
        stage_extract(args.files, dest, args.workers)
    return 0


if __name__ == "__main__":
    sys.exit(main())
