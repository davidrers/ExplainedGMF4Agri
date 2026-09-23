"""Data loaders for the sources this package can pull on the fly.

Every loader is resolved on first use rather than at import time. Importing the
package therefore costs nothing and, more importantly, does not require the
dependencies of loaders that are not being used: ``osm`` needs ``osmnx``, which
an environment that only wants Sentinel-1 or Sentinel-2 has no reason to carry.
This mirrors what ``space_time_deepsearch.vis`` already does.
"""

from importlib import import_module

_EXPORTS = {
    "download_osm_data": "osm",
    "get_population_data": "population",
    "get_population_raster": "population",
    "get_sentinel2_imagery": "sentinel2",
    "get_modis_temperature": "modis",
    "get_landsat_imagery": "landsat",
    "get_sentinel1_rtc_imagery": "sentinel1",
    "parcel_series": "sentinel1",
    "compute_features": "sentinel1",
    "normalize_orbit": "sentinel1",
    "select_orbit": "sentinel1",
    "to_db": "sentinel1",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name):
    module = _EXPORTS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    return getattr(import_module(f".{module}", __name__), name)


def __dir__():
    return sorted(set(__all__) | set(globals()))
