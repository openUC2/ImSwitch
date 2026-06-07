from .Options import Options
from .SetupInfo import DeviceInfo, DetectorInfo, LaserInfo, PositionerInfo, ScanInfo, SetupInfo
from .errors import *
import sys


def __getattr__(name: str):
    """PEP 562: lazily forward manager/io names to the managers sub-package."""
    import importlib
    managers = importlib.import_module(".managers", package=__package__)
    if name in getattr(managers, "__all__", ()) or hasattr(managers, name):
        obj = getattr(managers, name)
        globals()[name] = obj  # cache so this function is not called again
        return obj
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


