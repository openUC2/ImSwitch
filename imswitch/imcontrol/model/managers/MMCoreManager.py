"""
Process-wide singleton for the Micro-Manager :class:`CMMCorePlus` core.

This is **not** an ImSwitch device manager. It is an internal helper module
shared by the MMCore* device managers (camera, stage, laser) so that a single
Micro-Manager system configuration drives all of them without USB/serial
conflicts.

``pymmcore-plus`` is an optional dependency. Importing this module never
instantiates the core or touches hardware. The first call to
:func:`get_core` lazily creates the singleton, and :func:`ensure_loaded`
applies a configuration only once per ``cfg_path``.
"""

from __future__ import annotations

import os
import sys
import threading
from typing import List, Optional

from imswitch.imcommon.model import initLogger

try:  # pymmcore-plus is optional
    from pymmcore_plus import CMMCorePlus  # type: ignore
except ImportError:  # pragma: no cover - exercised on minimal installs
    CMMCorePlus = None  # type: ignore[assignment]


__all__ = [
    "get_core",
    "ensure_loaded",
    "reload",
    "get_available_adapters",
    "get_available_devices_for_adapter",
    "is_available",
]

if sys.platform.startswith("win"):
    _DEFAULT_ADAPTER_PATH = os.environ.get(
        "MICROMANAGER_PATH", r"C:\Program Files\Micro-Manager-2.0"
    )
elif sys.platform.startswith("darwin"):
    _DEFAULT_ADAPTER_PATH = os.environ.get(
        "MICROMANAGER_PATH", "/Applications/Micro-Manager.app/Contents/DeviceAdapters"
    )
else:
    _DEFAULT_ADAPTER_PATH = os.environ.get(
        "MICROMANAGER_PATH", "/opt/micro-manager/lib/micro-manager"
    )

_lock = threading.Lock()
_loaded_cfg: Optional[str] = None
_core = None  # type: ignore[assignment]
_logger = initLogger("MMCoreManager", tryInheritParent=False)


def _require_pymmcore() -> None:
    if CMMCorePlus is None:
        raise ImportError(
            "pymmcore-plus is not installed. Install it with "
            "'pip install pymmcore-plus' (or 'pip install ImSwitchUC2[pymmcore]') "
            "to use MMCore* device managers."
        )


def is_available() -> bool:
    """Return True when ``pymmcore-plus`` is importable in this environment."""
    return CMMCorePlus is not None


def get_core():
    """Return the process-wide :class:`CMMCorePlus` singleton.

    The instance is created lazily on first call so that simply importing
    this module never touches hardware.
    """
    global _core
    if _core is None:
        _require_pymmcore()
        with _lock:
            if _core is None:
                _core = CMMCorePlus.instance()
    return _core


def _resolve_adapter_paths(adapter_paths: Optional[List[str]]) -> List[str]:
    if adapter_paths:
        return [p for p in adapter_paths if p]
    return [_DEFAULT_ADAPTER_PATH]


def ensure_loaded(cfg_path: str, adapter_paths: Optional[List[str]] = None):
    """Load a Micro-Manager system configuration exactly once.

    Subsequent calls with the same ``cfg_path`` are no-ops and return the
    already-loaded core. A call with a different ``cfg_path`` causes the
    previously loaded devices to be unloaded before the new config is applied.

    Args:
        cfg_path: Absolute path to a Micro-Manager ``.cfg`` file.
        adapter_paths: Optional list of directories containing the compiled
            device adapter libraries. If ``None``, the ``MICROMANAGER_PATH``
            environment variable (or its default) is used.

    Returns:
        The shared :class:`CMMCorePlus` instance with devices loaded.
    """
    global _loaded_cfg
    _require_pymmcore()
    if not cfg_path:
        raise ValueError("ensure_loaded requires a non-empty cfg_path")

    with _lock:
        core = get_core()
        if _loaded_cfg == cfg_path:
            return core

        core.setDeviceAdapterSearchPaths(_resolve_adapter_paths(adapter_paths))

        if _loaded_cfg is not None:
            try:
                core.unloadAllDevices()
            except Exception:  # pragma: no cover - hardware-dependent
                _logger.warning("Failed to cleanly unload previous devices", exc_info=True)

        core.loadSystemConfiguration(cfg_path)
        _loaded_cfg = cfg_path

        try:
            devices = list(core.getLoadedDevices())
            _logger.info(
                f"MMCore loaded {len(devices)} devices from '{cfg_path}': "
                f"{', '.join(devices)}"
            )
        except Exception:  # pragma: no cover
            _logger.info(f"MMCore loaded configuration '{cfg_path}'")

        return core


def reload(cfg_path: str, adapter_paths: Optional[List[str]] = None):
    """Force a reload of the given configuration even if already loaded."""
    global _loaded_cfg
    with _lock:
        _loaded_cfg = None
    return ensure_loaded(cfg_path, adapter_paths)


def ensure_core(adapter_paths: Optional[List[str]] = None):
    """Return the singleton core with adapter paths configured.

    This is used by managers operating in *manual* mode (loading individual
    devices via ``loadDevice``) where no full ``.cfg`` file is supplied.
    """
    _require_pymmcore()
    core = get_core()
    with _lock:
        core.setDeviceAdapterSearchPaths(_resolve_adapter_paths(adapter_paths))
    return core


def get_available_adapters(adapter_path: Optional[str] = None) -> List[str]:
    """List the human-readable adapter names available on disk.

    Scans ``adapter_path`` (or the default Micro-Manager install location)
    for ``libmmgr_dal_*.so`` / ``mmgr_dal_*.dll`` / ``libmmgr_dal_*.dylib``
    files and returns the bare adapter names.
    """
    path = adapter_path or _DEFAULT_ADAPTER_PATH
    if not os.path.isdir(path):
        return []

    adapters = set()
    for entry in os.listdir(path):
        name = entry
        for prefix in ("libmmgr_dal_", "mmgr_dal_"):
            if name.startswith(prefix):
                name = name[len(prefix):]
                break
        else:
            continue
        for suffix in (".so", ".dylib", ".dll"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                adapters.add(name)
                break
    return sorted(adapters)


def get_available_devices_for_adapter(adapter_name: str) -> List[str]:
    """Return the list of device names exposed by a given adapter."""
    core = get_core()
    try:
        return list(core.getAvailableDevices(adapter_name))
    except Exception as exc:
        _logger.warning(f"Could not enumerate devices for adapter '{adapter_name}': {exc}")
        return []
