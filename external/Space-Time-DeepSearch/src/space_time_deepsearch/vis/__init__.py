from .animation import create_timelapse

__all__ = ["create_timelapse", "explore", "ParcelResult"]


def __getattr__(name):
    # parcel_timeseries loads on first use, so importing the package stays light
    # and `python -m space_time_deepsearch.vis.parcel_timeseries` does not find
    # the module already imported.
    if name in ("explore", "ParcelResult"):
        from . import parcel_timeseries
        return getattr(parcel_timeseries, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
