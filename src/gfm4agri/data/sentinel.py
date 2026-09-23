"""Shared Sentinel-1 and Sentinel-2 access for parcel-level work.

The imagery itself comes from the vendored ``space_time_deepsearch`` loaders under
``external/``. What lives here is the handful of corrections that sit between those
loaders and anything that reduces imagery over a parcel, and that every caller in this
repository must apply identically:

* the **BOA offset**, which changed with Sentinel-2 processing baseline 04.00 and which the
  vendored index code hard-codes for the post-change convention;
* **cloud screening**, which the vendored default does not perform;
* the **parcel mask**, so a median is taken over the declared polygon and nothing else.

Getting any of these wrong is silent: the series still looks like a series.
"""

from __future__ import annotations

import contextlib
import io as _io
import os
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

REPO = Path(__file__).resolve().parents[3]
VENDORED = REPO / "external" / "Space-Time-DeepSearch" / "src"

#: Sentinel-2 ground sampling distance of the bands used here.
PIXEL_M = 10.0
#: The bands a parcel reduction needs: blue for EVI, red and near infrared for NDVI, the red
#: edge for NDRE, the SWIR for NDMI, and the scene classification layer for screening.
S2_BANDS = ["B02", "B03", "B04", "B05", "B08", "B11", "SCL"]
#: Scene classification classes that must not enter a parcel median: nodata, saturated,
#: cloud shadow, cloud medium and high probability, thin cirrus, snow. The vendored default
#: blacklists only nodata, saturated and snow, because its own pipeline masks clouds in the
#: loader; here the cube is kept intact so an animation can still show the weather, and the
#: screening happens at reduction time.
SCL_DROP = (0, 1, 3, 8, 9, 10, 11)
#: A date enters a series only when this share of the parcel survives that screening.
MIN_VALID_FRACTION = 0.60
#: Sentinel-2 quantification: reflectance = (DN + offset) / 10000.
QUANTIFICATION = 10000.0
#: Processing baseline 04.00, operational 25 January 2022, introduced BOA_ADD_OFFSET = -1000.
OFFSET_BASELINE = 4.0
BOA_ADD_OFFSET = -1000.0


def ensure_vendored() -> None:
    """Put the vendored ``space_time_deepsearch`` package on the import path."""
    if str(VENDORED) not in sys.path:
        sys.path.insert(0, str(VENDORED))


#: Dask draws a progress bar with carriage returns and the loaders narrate every step. In a
#: notebook that arrives as hundreds of separate output chunks, so the noise is filtered out
#: and only the lines that say what was fetched are kept.
_NOISE = re.compile(r"^\[?#*\s*\]?\s*\|?\s*\d*%?\s*Completed|^\[\s*#*\s*\]|^$")


def quiet_lines(captured: str) -> list[str]:
    """The informative lines of a loader's captured stdout, progress bars removed."""
    out = []
    for raw in captured.replace("\r", "\n").splitlines():
        line = raw.strip()
        if not line or _NOISE.match(line) or "Completed" in line:
            continue
        out.append(line)
    return out


def baseline_lookup(geom_4326, start: str, end: str) -> dict[str, float]:
    """Acquisition date to Sentinel-2 processing baseline, from the STAC items.

    ``stackstac`` carries the baseline through as a coordinate, but that coordinate is lost
    whenever overlapping tiles of one date are mosaicked, which happens for any parcel on a
    tile overlap. The catalogue is then the only place left to ask.
    """
    import planetary_computer
    import pystac_client

    cat = pystac_client.Client.open("https://planetarycomputer.microsoft.com/api/stac/v1",
                                    modifier=planetary_computer.sign_inplace)
    items = cat.search(collections=["sentinel-2-l2a"], bbox=list(geom_4326.bounds),
                       datetime=f"{start}/{end}").item_collection()
    out: dict[str, float] = {}
    for it in items:
        try:
            value = float(str(it.properties.get("s2:processing_baseline", "0")))
        except ValueError:
            value = 0.0
        day = str(it.datetime.date())
        out[day] = max(out.get(day, 0.0), value)
    return out


def harmonise_boa(cube: xr.DataArray, fallback: dict[str, float] | None = None
                  ) -> tuple[xr.DataArray, dict]:
    """Put every date on the ``reflectance = DN / 10000`` convention.

    From processing baseline 04.00 Sentinel-2 L2A stores reflectance as ``(DN - 1000) /
    10000``; before it, as ``DN / 10000``. Subtracting an offset a scene never carried drops
    both the red and the near-infrared band by 0.1 reflectance, which inflates NDVI by
    roughly 0.45 on a green field. Every 2021 scene predates the change, so a 2021 season
    computed with the offset applied is wrong by that much.

    After this, indices must be computed with ``offset=0``.
    """
    coord = "s2:processing_baseline"
    days = [str(d)[:10] for d in np.atleast_1d(cube.time.values)]
    if coord in cube.coords:
        baselines = np.atleast_1d(cube[coord].values).astype(str)
        numeric = np.array([float(b) if b.replace(".", "").isdigit() else 0.0 for b in baselines])
        source = "cube coordinate"
    elif fallback:
        numeric = np.array([float(fallback.get(d, 0.0)) for d in days])
        baselines = np.array([f"{v:.2f}" for v in numeric])
        source = "STAC item properties"
    else:
        return cube, {"baselines": [], "baseline_source": "unavailable",
                      "offset_applied": "none, baseline metadata absent"}

    shift = xr.DataArray(np.where(numeric >= OFFSET_BASELINE, BOA_ADD_OFFSET, 0.0),
                         dims="time", coords={"time": cube.time})
    spectral = xr.DataArray([str(b) != "SCL" for b in cube.band.values], dims="band",
                            coords={"band": cube.band})
    report = {
        "baseline_source": source,
        "baselines": sorted(set(baselines.tolist())),
        "dates_offset_corrected": int((numeric >= OFFSET_BASELINE).sum()),
        "dates_left_as_is": int((numeric < OFFSET_BASELINE).sum()),
        "convention": "reflectance = DN / 10000 for every date after harmonisation",
    }
    return cube + shift * spectral, report


def parcel_mask(cube: xr.DataArray, geom_projected) -> xr.DataArray:
    """Boolean (y, x) mask of the pixels whose centre falls inside the polygon."""
    from affine import Affine
    from rasterio.features import rasterize

    x, y = cube.x.values, cube.y.values
    dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
    transform = Affine.translation(x[0] - dx / 2, y[0] - dy / 2) * Affine.scale(dx, dy)
    arr = rasterize([geom_projected], out_shape=(len(y), len(x)), transform=transform,
                    fill=0, default_value=1, dtype="uint8").astype(bool)
    return xr.DataArray(arr, dims=("y", "x"), coords={"y": cube.y, "x": cube.x})


def clear_fraction(cube: xr.DataArray, mask: xr.DataArray) -> np.ndarray:
    """Share of the parcel's pixels that are neither cloud, shadow, cirrus, snow nor nodata."""
    scl = cube.sel(band="SCL").where(mask)
    bad = scl.isin(list(SCL_DROP))
    inside = int(mask.sum())
    return (1 - bad.sum(dim=("y", "x")).values / max(inside, 1)).astype(float)


def fetch_s2(geom_4326, start: str, end: str, *, bands: list[str] | None = None,
             resolution: int = 10, cloud_cover_max: int = 70, min_coverage: int = 80,
             verbose: bool = True) -> tuple[xr.DataArray, dict]:
    """A Sentinel-2 L2A cube for one area of interest, already offset-harmonised."""
    ensure_vendored()
    from space_time_deepsearch.io import get_sentinel2_imagery

    buf = _io.StringIO()
    with warnings.catch_warnings(), contextlib.redirect_stdout(buf):
        warnings.simplefilter("ignore")
        cube = get_sentinel2_imagery(
            custom_geometry=geom_4326, start_date=start, end_date=end,
            bands=bands or S2_BANDS, resolution=resolution,
            cloud_cover_max=cloud_cover_max, min_coverage=min_coverage)
    if verbose:
        for line in quiet_lines(buf.getvalue()):
            print(f"  {line}")
    cube, report = harmonise_boa(cube, fallback=baseline_lookup(geom_4326, start, end))
    if verbose:
        print(f"  baselines {report.get('baselines')} via {report.get('baseline_source')}, "
              f"{report.get('dates_offset_corrected', 0)} dates offset-corrected")
    return cube, report


def screen(series: dict[str, pd.DataFrame], n_parcel_pixels: int,
           min_valid_fraction: float = MIN_VALID_FRACTION) -> tuple[dict, int]:
    """Drop the dates whose parcel was mostly cloud, and say how many went."""
    floor = min_valid_fraction * n_parcel_pixels
    out = {k: v[v["n_pixels"] >= floor].reset_index(drop=True) for k, v in series.items()}
    dropped = len(series["NDVI"]) - len(out["NDVI"]) if "NDVI" in series else 0
    return out, int(dropped)


def eurocropsml_series(parcel_id: str | int, ml_root: Path | None = None) -> pd.DataFrame:
    """The per-parcel L1C series EuroCropsML ships, with NDVI attached.

    Read from the extracted tree when it is there and from the archive otherwise, so the
    function works whether or not the 706,683 files have been unpacked.
    """
    import io as _io
    import zipfile

    root = Path(ml_root or os.environ.get("EUROCROPSML_DATA", REPO / "data" / "eurocropsml"))
    idx = pd.read_parquet(root / "preprocess_index.parquet")
    row = idx[idx["parcel_id"] == str(parcel_id)]
    if row.empty:
        raise KeyError(f"parcel {parcel_id} is not in the EuroCropsML index")
    r = row.iloc[0]
    name = f"{r['nuts3']}_{r['parcel_id']}_{r['hcat']}.npz"

    tree = root / "preprocess" / name
    if tree.exists():
        z = np.load(tree, allow_pickle=True)
    else:
        with zipfile.ZipFile(root / "archives" / "preprocess.zip") as zf:
            z = np.load(_io.BytesIO(zf.read(f"preprocess/{name}")), allow_pickle=True)

    bands = ["B01", "B02", "B03", "B04", "B05", "B06", "B07", "B08", "B8A", "B09", "B10",
             "B11", "B12"]
    data = np.asarray(z["data"], dtype="float64")
    dates = pd.DatetimeIndex(pd.to_datetime(np.asarray(z["dates"]))).as_unit("ns")
    red, nir = data[:, bands.index("B04")], data[:, bands.index("B08")]
    with np.errstate(divide="ignore", invalid="ignore"):
        ndvi = (nir - red) / (nir + red)
    out = pd.DataFrame({"date": dates, "NDVI": ndvi, "nuts3": r["nuts3"],
                        **{b: data[:, i] / QUANTIFICATION for i, b in enumerate(bands)}})
    return out.sort_values("date").reset_index(drop=True)
