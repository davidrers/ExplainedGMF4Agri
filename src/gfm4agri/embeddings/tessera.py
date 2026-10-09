"""TESSERA v1 annual embeddings on the chip grid, read straight from the public Zarr store.

TESSERA (Feng et al., 2025) encodes each 10 m pixel's full-year Sentinel-1 and Sentinel-2
time series into one 128-dimensional vector. The embeddings are precomputed and served from
Source Cooperative as a Zarr v3 store with one group per UTM zone, on the grid they were
produced on, quantised to int8 with one float32 scale per pixel.

The ``geotessera`` client that knows the new location requires Python 3.12, and the
environment here is 3.11, so the store is read directly with ``zarr`` over HTTPS. That is
also the cheaper route for chips: the store is sharded in 32 x 32 pixel inner chunks, so a
chip window costs a few hundred kilobytes rather than whole 0.1 degree tiles.

Each chip window is read in the zone's native UTM grid with a margin, dequantised, and
warped onto the chip's EPSG:3035 grid by **nearest neighbour**, so every output pixel is a
genuine TESSERA vector rather than an interpolation between two of them.

The store files each tile under a single zone, so a chip close to a zone edge can extend onto
tiles held by the neighbouring zone, and reading its own zone alone would leave that strip
empty. Pixels still empty after the chip's own zone are therefore filled from the
neighbouring zone whenever the chip lies within :data:`ZONE_EDGE_DEG` of the edge.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

__all__ = ["STORE_URL", "MODEL_VERSION", "N_DIMS", "TesseraStore", "utm_zone"]

STORE_URL = "https://data.source.coop/tessera/tessera/zarr/v1"
MODEL_VERSION = "1.0"
N_DIMS = 128
#: Pixels read beyond the chip's projected footprint, so the warp has no empty edge.
MARGIN_PX = 8
#: Longitude distance from a zone edge within which the neighbouring zone is also read.
ZONE_EDGE_DEG = 0.5
#: Largest per-pixel scale taken as genuine. Over Estonia the embeddings reach at most 17.7 in
#: absolute value, a scale of about 0.14 at the int8 limit, and at the same place every other
#: year stays below 0.1. The 2021 layer of zone 35 carries a patch near 26.1 E, 58.9 N whose
#: scales run up to 2.8e32, which the store serves unchanged on every read; a pixel above this
#: bound is treated as missing, like a pixel outside the coverage.
MAX_SCALE = 0.2


def utm_zone(lon: float) -> int:
    return int(math.floor((lon + 180.0) / 6.0)) + 1


@dataclass
class _Zone:
    group: object
    crs: str
    transform: tuple[float, ...]   # GDAL-style affine: a, b, c, d, e, f
    years: np.ndarray


class TesseraStore:
    """Chip-window reader for the TESSERA v1 Zarr store."""

    def __init__(self, url: str = STORE_URL, concurrency: int = 64) -> None:
        import zarr
        from zarr.storage import FsspecStore

        zarr.config.set({"async.concurrency": concurrency})
        self.url = url
        self.root = zarr.open_group(FsspecStore.from_url(url, read_only=True), mode="r",
                                    zarr_format=3)
        attrs = dict(self.root.attrs)
        self.model = attrs.get("geoemb:model", "")
        self._zones: dict[int, _Zone] = {}

    def zone(self, number: int) -> _Zone:
        if number not in self._zones:
            import json

            g = self.root[f"utm{number:02d}"]
            a = dict(g.attrs)
            tr = a["spatial:transform"]
            tr = json.loads(tr) if isinstance(tr, str) else tr
            self._zones[number] = _Zone(g, a["proj:code"], tuple(float(v) for v in tr),
                                        np.asarray(g["time"][:]))
        return self._zones[number]

    def read_chip(self, bounds: tuple[float, float, float, float], crs: str, year: int,
                  size_px: int = 224, pixel_m: float = 10.0) -> tuple[np.ndarray, dict]:
        """``(128, size_px, size_px)`` float32 embeddings on the chip grid, and provenance.

        Pixels outside the store's coverage come back as NaN.
        """
        from pyproj import Transformer

        xmin, ymin, xmax, ymax = bounds
        lon, _ = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform(
            (xmin + xmax) / 2, (ymin + ymax) / 2)
        own = utm_zone(lon)
        zones = [own]
        offset = (lon + 180.0) % 6.0
        if offset < ZONE_EDGE_DEG:
            zones.append(own - 1)
        elif offset > 6.0 - ZONE_EDGE_DEG:
            zones.append(own + 1)

        out = np.full((N_DIMS, size_px, size_px), np.nan, dtype=np.float32)
        info: dict = {}
        for number in zones:
            empty = np.isnan(out[0])
            if not empty.any():
                break
            part, zinfo = self._read_zone(number, bounds, crs, year, size_px, pixel_m)
            fill = empty & ~np.isnan(part[0])
            out[:, fill] = part[:, fill]
            if number == own:
                info = zinfo
            elif fill.any():
                info.setdefault("filled_from", []).append(
                    {**zinfo, "pixels": int(fill.sum())})
        info["nodata_share"] = round(float(np.isnan(out[0]).mean()), 6)
        return out, info

    def _read_zone(self, number: int, bounds: tuple[float, float, float, float], crs: str,
                   year: int, size_px: int, pixel_m: float) -> tuple[np.ndarray, dict]:
        """The chip as seen through one zone's arrays; NaN where that zone holds nothing."""
        from affine import Affine
        from pyproj import Transformer
        from rasterio.warp import Resampling, reproject

        xmin, ymin, xmax, ymax = bounds
        z = self.zone(number)
        hits = np.where(z.years == year)[0]
        if not len(hits):
            raise KeyError(f"year {year} not in the store; available {z.years.tolist()}")
        ti = int(hits[0])

        a, _, c, _, e, f = z.transform
        xs, ys = Transformer.from_crs(crs, z.crs, always_xy=True).transform(
            [xmin, xmax, xmin, xmax], [ymin, ymin, ymax, ymax])
        col0 = int(math.floor((min(xs) - c) / a)) - MARGIN_PX
        col1 = int(math.ceil((max(xs) - c) / a)) + MARGIN_PX
        row0 = int(math.floor((max(ys) - f) / e)) - MARGIN_PX
        row1 = int(math.ceil((min(ys) - f) / e)) + MARGIN_PX

        # Clipped to the zone's array: a neighbouring zone may not reach the chip at all, and a
        # negative start would otherwise wrap round to the far end of the array.
        n_rows, n_cols = z.group["scales"].shape[1:]
        row0, row1 = max(row0, 0), min(row1, n_rows)
        col0, col1 = max(col0, 0), min(col1, n_cols)
        out = np.full((N_DIMS, size_px, size_px), np.nan, dtype=np.float32)
        if row0 >= row1 or col0 >= col1:
            return out, {"zone_crs": z.crs, "year_index": ti, "window_rows_cols": None,
                         "nodata_share": 1.0}

        q = z.group["embeddings"][ti, :, row0:row1, col0:col1]
        s = z.group["scales"][ti, row0:row1, col0:col1]
        valid = np.isfinite(s) & (s != 0)
        implausible = valid & (s > MAX_SCALE)
        valid &= ~implausible
        emb = q.astype(np.float32) * np.where(valid, s, 0).astype(np.float32)[None]
        emb[:, ~valid] = np.nan

        reproject(emb, out,
                  src_transform=Affine(a, 0, c + col0 * a, 0, e, f + row0 * e), src_crs=z.crs,
                  dst_transform=Affine(pixel_m, 0, xmin, 0, -pixel_m, ymax), dst_crs=crs,
                  resampling=Resampling.nearest, src_nodata=np.nan, dst_nodata=np.nan)
        info = {"zone_crs": z.crs, "year_index": ti,
                "window_rows_cols": [row0, row1, col0, col1],
                "implausible_scale_pixels": int(implausible.sum()),
                "nodata_share": round(float(np.isnan(out[0]).mean()), 6)}
        return out, info
