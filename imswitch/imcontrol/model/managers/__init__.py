"""
Manager package with lazy loading (PEP 562).

Manager modules — and their heavy third-party dependencies — are only imported
the first time the name is actually accessed.  Subsequent accesses hit the
module-level cache (globals()) and are therefore free.

To add a new manager:
  1. Add an entry to _MANAGER_MAP.
  2. If the manager requires optional dependencies (arkitekt, unitelabs, …),
     add its name to _OPTIONAL_MANAGERS so that an ImportError returns None
     rather than crashing at startup.
"""

import importlib

# ---------------------------------------------------------------------------
# Registry:  public_name -> (relative_module, attribute_in_that_module)
# ---------------------------------------------------------------------------
_MANAGER_MAP: dict = {
    "AutofocusManager":          (".AutofocusManager",          "AutofocusManager"),
    "FOVLockManager":            (".FOVLockManager",            "FOVLockManager"),
    "DetectorsManager":          (".DetectorsManager",          "DetectorsManager"),
    "NoDetectorsError":          (".DetectorsManager",          "NoDetectorsError"),
    "LasersManager":             (".LasersManager",             "LasersManager"),
    "LEDsManager":               (".LEDsManager",               "LEDsManager"),
    "LEDMatrixsManager":         (".LEDMatrixsManager",         "LEDMatrixsManager"),
    "MultiManager":              (".MultiManager",              "MultiManager"),
    "NidaqManager":              (".NidaqManager",              "NidaqManager"),
    "PositionersManager":        (".PositionersManager",        "PositionersManager"),
    "GalvoScannersManager":      (".GalvoScannersManager",      "GalvoScannersManager"),
    "RS232sManager":             (".RS232sManager",             "RS232sManager"),
    "SLMManager":                (".SLMManager",                "SLMManager"),
    "RotatorsManager":           (".RotatorsManager",           "RotatorsManager"),
    "UC2ConfigManager":          (".UC2ConfigManager",          "UC2ConfigManager"),
    "SIMManager":                (".SIMManager",                "SIMManager"),
    "DPCManager":                (".DPCManager",                "DPCManager"),
    "TimelapseManager":          (".TimelapseManager",          "TimelapseManager"),
    "ExperimentManager":         (".ExperimentManager",         "ExperimentManager"),
    "ROIScanManager":            (".ROIScanManager",            "ROIScanManager"),
    "LightsheetManager":         (".LightsheetManager",         "LightsheetManager"),
    "WebRTCManager":             (".WebRTCManager",             "WebRTCManager"),
    "HyphaManager":              (".HyphaManager",              "HyphaManager"),
    "HistoScanManager":          (".HistoScanManager",          "HistoScanManager"),
    "StresstestManager":         (".StresstestManager",         "StresstestManager"),
    "ObjectiveManager":          (".ObjectiveManager",          "ObjectiveManager"),
    "WorkflowManager":           (".WorkflowManager",           "WorkflowManager"),
    "FlowStopManager":           (".FlowStopManager",           "FlowStopManager"),
    "LepmonManager":             (".LepmonManager",             "LepmonManager"),
    "PixelCalibrationManager":   (".PixelCalibrationManager",   "PixelCalibrationManager"),
    "ArkitektManager":           (".ArkitektManager",           "ArkitektManager"),
    "SiLA2Manager":              (".SiLA2Manager",              "SiLA2Manager"),
    "InstrumentMetadataManager": (".InstrumentMetadataManager", "InstrumentMetadataManager"),
}

# Managers whose dependencies may not be installed; return None instead of raising.
_OPTIONAL_MANAGERS: frozenset = frozenset({"ArkitektManager", "SiLA2Manager"})

# Names forwarded from the io sub-module for backwards compatibility.
_IO_NAMES: frozenset = frozenset({"RecordingService", "RecMode", "SaveMode", "SaveFormat"})

# __all__ lets `from .managers import *` work without eagerly importing anything.
__all__ = list(_MANAGER_MAP) + list(_IO_NAMES) + ["MMCoreManager"]


def __getattr__(name: str):
    """PEP 562 lazy loader — called only when *name* is not yet in globals()."""
    if name in _MANAGER_MAP:
        module_path, attr = _MANAGER_MAP[name]
        try:
            mod = importlib.import_module(module_path, package=__package__)
            obj = getattr(mod, attr)
        except ImportError:
            if name in _OPTIONAL_MANAGERS:
                globals()[name] = None  # cache the None so __getattr__ isn't called again
                return None
            raise
        globals()[name] = obj  # cache for subsequent accesses
        return obj

    if name in _IO_NAMES:
        # Bulk-import and cache all four io names together.
        from imswitch.imcontrol.model.io import RecordingService, RecMode, SaveMode, SaveFormat
        globals().update(
            RecordingService=RecordingService,
            RecMode=RecMode,
            SaveMode=SaveMode,
            SaveFormat=SaveFormat,
        )
        return globals()[name]

    if name == "MMCoreManager":
        from . import MMCoreManager as _mod  # shared pymmcore-plus singleton (optional)
        globals()["MMCoreManager"] = _mod
        return _mod

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
