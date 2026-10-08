"""AlphaEarth Foundations annual embeddings on the chip grid, cut from the published COG tiles.

AlphaEarth Foundations (Brown et al., 2025) summarises each 10 m pixel's calendar year of
multi-source Earth observation into one 64-dimensional unit vector. Google publishes the
annual layers, the Earth Engine collection ``GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL``, as Cloud
Optimized GeoTIFFs in ``gs://alphaearth_foundations``: one directory per year and UTM zone,
8192 x 8192 pixels and 64 int8 bands per file, ``-128`` for a masked pixel. That bucket is
requester pays. Source Cooperative mirrors it file for file (``tge-labs/aef``) at no charge
and carries Google's MD5 for each file, so tiles are fetched from the mirror and checked
against the checksum Google publishes (``scripts/data/fetch_alphaearth_tiles.py``, which can
also read Google's bucket directly).

Three properties of the files set how a chip is read:

* **Tiles are downloaded whole, not read remotely by window.** The files are band
  interleaved in 1024 x 1024 blocks, so one chip window costs a block per band: about 60 MB
  and one to three minutes per chip over HTTP, against 42 GB once for the 22 tiles under the
  Estonian chips, after which a chip takes a few seconds.
* **The files are stored bottom-up**: origin at the south-west corner, positive y
  resolution, a layout most tools take for north-up. A window is read in raw array rows and
  flipped to north-up before the warp. GDAL's own handling of the positive y resolution gave
  identical chips on a pilot chip; the flip makes the orientation explicit.
* **Values are quantised.** ``((q / 127.5) ** 2) * sign(q)`` recovers the embedding, as the
  dataset README specifies. The de-quantised vectors are left as they are, with norms within
  about 1 % of one, rather than renormalised.

As for TESSERA, each window is read in the tile's native UTM grid with a margin and warped
onto the chip's EPSG:3035 grid by **nearest neighbour**, so every output pixel is a genuine
AlphaEarth vector, and a chip within :data:`ZONE_EDGE_DEG` of a zone edge has the pixels its
own zone leaves empty filled from the neighbouring zone.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

__all__ = ["BAND_NAMES", "INDEX_URL", "MIRROR_URL", "MODEL_VERSION", "N_DIMS", "OFFICIAL_URL",
           "AlphaEarthStore", "chip_zones", "dequantise", "select_tiles"]

#: Google's bucket, over HTTPS; requester pays.
OFFICIAL_URL = "https://storage.googleapis.com/alphaearth_foundations/satellite_embedding/v1/annual"
#: The Source Cooperative mirror, the fetch script's default source.
MIRROR_URL = "https://data.source.coop/tge-labs/aef/v1/annual"
#: One row per file: its UTM zone, CRS and the UTM bounds of its pixel array.
INDEX_URL = f"{MIRROR_URL}/aef_index.parquet"
MODEL_VERSION = "V1"
N_DIMS = 64
BAND_NAMES = [f"A{i:02d}" for i in range(N_DIMS)]
NODATA = -128
#: The tiles' native pixel size.
TILE_PIXEL_M = 10.0
#: Pixels read beyond the chip's projected footprint, so the warp has no empty edge.
MARGIN_PX = 8
#: Longitude distance from a zone edge within which the neighbouring zone is also read.
ZONE_EDGE_DEG = 0.5
#: Per year, the tiles of every zone a chip set touches: ``<tile dir>/<year>/tiles.json``.
TILE_TABLE = "tiles.json"


#: The README's ``((q / 127.5) ** 2) * sign(q)`` for every int8 level, in float64 then cast,
#: indexed by ``q + 128``. Evaluated in float32 instead, 154 of the 255 levels round differently.
_LEVELS = np.arange(-128, 128)
_DEQUANT = ((_LEVELS / 127.5) ** 2 * np.sign(_LEVELS)).astype(np.float32)
_DEQUANT[0] = np.nan


def dequantise(q: np.ndarray) -> np.ndarray:
    """``(bands, h, w)`` int8 to float32 in [-1, 1], NaN where the pixel is masked."""
    out = _DEQUANT[q.astype(np.int16) + 128]
    out[:, (q == NODATA).any(axis=0)] = np.nan
    return out


def chip_zones(lon: np.ndarray | float, lat: np.ndarray | float) -> list[list[str]]:
    """Per chip centre, its own UTM zone and, near an edge, the neighbouring one: ``"35N"``."""
    lon, lat = np.atleast_1d(lon).astype(float), np.atleast_1d(lat).astype(float)
    own = np.floor((lon + 180.0) / 6.0).astype(int) + 1
    offset = (lon + 180.0) % 6.0
    out = []
    for z, off, la in zip(own, offset, lat):
        hemi = "N" if la >= 0 else "S"
        zones = [int(z)]
        if off < ZONE_EDGE_DEG:
            zones.append(60 if z == 1 else int(z) - 1)
        elif off > 6.0 - ZONE_EDGE_DEG:
            zones.append(1 if z == 60 else int(z) + 1)
        out.append([f"{n}{hemi}" for n in zones])
    return out


def _meets(utm: tuple[float, float, float, float], xs, ys) -> np.ndarray:
    """Whether a tile's pixel array meets each chip footprint widened by the read margin.

    ``xs`` and ``ys`` are a footprint's corners in the tile's CRS, ``(4,)`` for one chip or
    ``(n, 4)`` for many.
    """
    west, south, east, north = utm
    pad = MARGIN_PX * TILE_PIXEL_M
    xs, ys = np.asarray(xs), np.asarray(ys)
    return ((xs.min(-1) - pad < east) & (xs.max(-1) + pad > west)
            & (ys.min(-1) - pad < north) & (ys.max(-1) + pad > south))


def select_tiles(index, bounds: np.ndarray, crs: str, year: int):
    """The rows of ``index`` that :meth:`AlphaEarthStore.read_chip` reads for any chip.

    ``index`` is the provider's tile index (``aef_index.parquet``) as a DataFrame, ``bounds``
    the chips' ``(n, 4)`` bounds in ``crs``.
    """
    from pyproj import Transformer

    b = np.asarray(bounds, dtype=float)
    lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
        (b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2)
    zones = chip_zones(lon, lat)
    rows = index[index["year"] == year]
    keep: set = set()
    for zone in sorted({z for zs in zones for z in zs}):
        tiles = rows[rows["utm_zone"] == zone]
        if tiles.empty:
            continue
        which = np.array([zone in zs for zs in zones])
        cb = b[which]
        t = Transformer.from_crs(crs, tiles["crs"].iloc[0], always_xy=True)
        xs, ys = (np.asarray(v) for v in t.transform(cb[:, [0, 2, 0, 2]], cb[:, [1, 1, 3, 3]]))
        for i, r in tiles.iterrows():
            if _meets((r["utm_west"], r["utm_south"], r["utm_east"], r["utm_north"]), xs, ys).any():
                keep.add(i)
    return rows.loc[sorted(keep)]


@dataclass(frozen=True)
class _Tile:
    key: str                                  # "<year>/<zone>/<name>.tiff", as in the bucket
    utm: tuple[float, float, float, float]    # west, south, east, north of the pixel array


class AlphaEarthStore:
    """Chip-window reader over the AlphaEarth tiles downloaded for one year.

    ``tile_dir`` holds ``<year>/<zone>/<file>.tiff`` and ``<year>/tiles.json``, which lists
    every tile of each zone a chip set touches, downloaded or not. A chip that needs a tile
    not on disk raises rather than coming back empty.
    """

    def __init__(self, tile_dir: Path | str, year: int) -> None:
        self.dir = Path(tile_dir)
        self.year = year
        table = json.loads((self.dir / str(year) / TILE_TABLE).read_text())
        self.crs = {z: rec["crs"] for z, rec in table["zones"].items()}
        self.tiles = {z: [_Tile(t["key"], tuple(t["utm"])) for t in rec["tiles"]]
                      for z, rec in table["zones"].items()}

    def read_chip(self, bounds: tuple[float, float, float, float], crs: str, year: int,
                  size_px: int = 224, pixel_m: float = 10.0) -> tuple[np.ndarray, dict]:
        """``(64, size_px, size_px)`` float32 embeddings on the chip grid, and provenance.

        Masked pixels and pixels outside every tile come back as NaN.
        """
        from pyproj import Transformer

        if year != self.year:
            raise ValueError(f"store holds {self.year}, chip asks for {year}")
        xmin, ymin, xmax, ymax = bounds
        lon, lat = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
            (xmin + xmax) / 2, (ymin + ymax) / 2)
        zones = chip_zones(lon, lat)[0]
        if zones[0] not in self.tiles:
            raise KeyError(f"zone {zones[0]} not in {self.dir / str(year) / TILE_TABLE}; "
                           "fetch the tiles for this chip set first")

        out = np.full((N_DIMS, size_px, size_px), np.nan, dtype=np.float32)
        used = []
        for zone in zones:
            if zone not in self.tiles:
                continue
            xs, ys = Transformer.from_crs(crs, self.crs[zone], always_xy=True).transform(
                [xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
            for tile in self.tiles[zone]:
                empty = np.isnan(out[0])
                if not empty.any():
                    break
                if not _meets(tile.utm, xs, ys):
                    continue
                part = self._read_tile(tile, bounds, crs, xs, ys, size_px, pixel_m)
                fill = empty & ~np.isnan(part[0])
                out[:, fill] = part[:, fill]
                if fill.any():
                    used.append({"key": tile.key, "pixels": int(fill.sum())})
        return out, {"tiles": used, "nodata_share": round(float(np.isnan(out[0]).mean()), 6)}

    def _read_tile(self, tile: _Tile, bounds, crs: str, xs, ys, size_px: int,
                   pixel_m: float) -> np.ndarray:
        """The chip as seen through one tile; NaN where the tile holds nothing."""
        import rasterio
        from affine import Affine
        from rasterio.warp import Resampling, reproject
        from rasterio.windows import Window

        path = self.dir / tile.key
        if not path.exists():
            raise FileNotFoundError(f"{path} is listed but not on disk; run "
                                    "scripts/data/fetch_alphaearth_tiles.py for this chip set")
        xmin, _, _, ymax = bounds
        out = np.full((N_DIMS, size_px, size_px), np.nan, dtype=np.float32)
        with rasterio.open(path) as src:
            a, _, c, _, e, f = tuple(src.transform)[:6]
            # Rows run north to south in a conventional file and south to north in these;
            # sorting the two edges makes the window right for either.
            c0, c1 = sorted(((min(xs) - c) / a, (max(xs) - c) / a))
            r0, r1 = sorted(((min(ys) - f) / e, (max(ys) - f) / e))
            col0, col1 = max(math.floor(c0) - MARGIN_PX, 0), min(math.ceil(c1) + MARGIN_PX, src.width)
            row0, row1 = max(math.floor(r0) - MARGIN_PX, 0), min(math.ceil(r1) + MARGIN_PX, src.height)
            if row0 >= row1 or col0 >= col1:
                return out
            q = src.read(window=Window(col0, row0, col1 - col0, row1 - row0))
            tile_crs = src.crs

        emb = dequantise(q)
        if e > 0:   # bottom-up: flip to north-up, the top row being the window's last
            emb = np.ascontiguousarray(emb[:, ::-1])
            src_transform = Affine(a, 0, c + col0 * a, 0, -e, f + row1 * e)
        else:
            src_transform = Affine(a, 0, c + col0 * a, 0, e, f + row0 * e)
        reproject(emb, out, src_transform=src_transform, src_crs=tile_crs,
                  dst_transform=Affine(pixel_m, 0, xmin, 0, -pixel_m, ymax), dst_crs=crs,
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
        return out
