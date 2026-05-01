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

import glob
import os
import sys
import threading
from typing import List, Optional

from imswitch.imcommon.model import initLogger

try:  # pymmcore-plus is optional
    from pymmcore_plus import CMMCorePlus  # type: ignore
except ImportError:  # pragma: no cover - exercised on minimal installs
    CMMCorePlus = None  # type: ignore[assignment]

try:  # pragma: no cover - depends on pymmcore-plus version
    from pymmcore_plus import find_micromanager  # type: ignore
except ImportError:  # pragma: no cover
    find_micromanager = None  # type: ignore[assignment]


__all__ = [
    "get_core",
    "ensure_loaded",
    "reload",
    "ensure_core",
    "get_available_adapters",
    "get_available_devices_for_adapter",
    "discover_adapter_paths",
    "is_available",
]

# ---------------------------------------------------------------------------
# Micro-Manager 2.0 is required.
#
# pymmcore-plus is built against the MM 2.0 device interface (currently v71+).
# MM 1.4 ``mmgr_dal_*`` libraries will fail to load with an "interface version
# mismatch" error. We only advertise MM 2.0 install locations below.
# ---------------------------------------------------------------------------
_MM2_WIN_GLOBS = [
    r"C:\Program Files\Micro-Manager-2.0*",
    r"C:\Program Files (x86)\Micro-Manager-2.0*",
]
_MM2_MAC_GLOBS = [
    "/Applications/Micro-Manager-2.0*",
    "/Applications/Micro-Manager-2.0*.app/Contents/Resources",
    "/Applications/Micro-Manager.app/Contents/Resources",
]
_MM2_LINUX_GLOBS = [
    "/opt/micro-manager/lib/micro-manager",
    "/opt/Micro-Manager-2.0*",
    "/usr/local/lib/micro-manager",
]
# pymmcore-plus' own managed install location (`mmcore install`) – same on
# all platforms.
_PYMMCORE_PLUS_GLOBS = [
    os.path.expanduser("~/.local/share/pymmcore-plus/mm/Micro-Manager-*"),
    os.path.expanduser("~/Library/Application Support/pymmcore-plus/mm/Micro-Manager-*"),
    os.path.expandvars(r"%LOCALAPPDATA%\pymmcore-plus\mm\Micro-Manager-*"),
]

_lock = threading.Lock()
_loaded_cfg: Optional[str] = None
_core = None  # type: ignore[assignment]
_dll_dirs_added: set = set()
_logger = initLogger("MMCoreManager", tryInheritParent=False)


def _add_windows_dll_dirs(paths: List[str]) -> None:
    """On Windows, add adapter directories to the DLL search path.

    MM 2.0 adapters routinely depend on vendor SDK DLLs that live alongside
    them in ``C:\\Program Files\\Micro-Manager-2.0``. Python 3.8+ no longer
    looks at ``PATH`` for DLL resolution, so we explicitly add each adapter
    directory via :func:`os.add_dll_directory`.
    """
    if not sys.platform.startswith("win"):
        return
    for path in paths:
        norm = os.path.normpath(path)
        if norm in _dll_dirs_added or not os.path.isdir(norm):
            continue
        try:
            os.add_dll_directory(norm)  # type: ignore[attr-defined]
            _dll_dirs_added.add(norm)
        except (OSError, AttributeError):  # pragma: no cover
            pass


def _platform_globs() -> List[str]:
    if sys.platform.startswith("win"):
        return _MM2_WIN_GLOBS + _PYMMCORE_PLUS_GLOBS
    if sys.platform.startswith("darwin"):
        return _MM2_MAC_GLOBS + _PYMMCORE_PLUS_GLOBS
    return _MM2_LINUX_GLOBS + _PYMMCORE_PLUS_GLOBS


def _looks_like_adapter_dir(path: str) -> bool:
    """Return True if *path* contains at least one Micro-Manager adapter library."""
    if not path or not os.path.isdir(path):
        return False
    try:
        for entry in os.listdir(path):
            lower = entry.lower()
            if lower.startswith(("libmmgr_dal_", "mmgr_dal_")) and lower.endswith(
                (".so", ".dylib", ".dll")
            ):
                return True
    except OSError:
        return False
    return False


def discover_adapter_paths() -> List[str]:
    """Return a de-duplicated, ordered list of plausible MM 2.0 adapter dirs.

    Resolution order (first match wins, but every match is returned so that
    ``setDeviceAdapterSearchPaths`` can fall through):

    1. ``MICROMANAGER_PATH`` environment variable (single path).
    2. ``pymmcore_plus.find_micromanager()`` – pymmcore-plus' own discovery,
       which knows about ``mmcore install`` managed installs.
    3. Platform-specific glob patterns for the standard MM 2.0 install
       locations on Windows / macOS / Linux.
    """
    seen: set = set()
    paths: List[str] = []

    def _add(p: Optional[str]) -> None:
        if not p:
            return
        norm = os.path.normpath(p)
        if norm in seen:
            return
        if _looks_like_adapter_dir(norm):
            seen.add(norm)
            paths.append(norm)

    _add(os.environ.get("MICROMANAGER_PATH"))

    if find_micromanager is not None:
        try:
            found = find_micromanager()
        except Exception:  # pragma: no cover
            found = None
        if isinstance(found, (list, tuple)):
            for p in found:
                _add(p)
        else:
            _add(found)

    for pattern in _platform_globs():
        for match in sorted(glob.glob(pattern), reverse=True):
            _add(match)

    return paths


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
    """Pick the adapter directories to feed ``setDeviceAdapterSearchPaths``.

    Explicit ``adapter_paths`` (typically from the setup JSON) always win.
    Otherwise we run :func:`discover_adapter_paths` and, as a last resort,
    fall back to a hard-coded MM 2.0 default per platform so that error
    messages from MMCore at least mention a real-looking directory.
    """
    if adapter_paths:
        explicit = [os.path.normpath(p) for p in adapter_paths if p]
        if explicit:
            return explicit

    discovered = discover_adapter_paths()
    if discovered:
        return discovered

    # Hard-coded last-resort defaults so MMCore produces a useful error.
    if sys.platform.startswith("win"):
        return [r"C:\Program Files\Micro-Manager-2.0"]
    if sys.platform.startswith("darwin"):
        return ["/Applications/Micro-Manager-2.0"]
    return ["/opt/micro-manager/lib/micro-manager"]


def _verify_mm2(core) -> None:
    """Log a clear warning if the loaded core does not look like MM 2.0."""
    try:
        api = str(core.getAPIVersionInfo())
    except Exception:
        return
    # MM 2.0 API version strings look like "Device API version 71, Module API version 10".
    # MM 1.4 reports much lower numbers. We only warn – don't refuse to run –
    # because MM may bump these in future and our pin would become stale.
    import re

    m = re.search(r"Device API version\s+(\d+)", api)
    if m and int(m.group(1)) < 70:
        _logger.warning(
            f"Loaded MMCore reports '{api}'. ImSwitch requires Micro-Manager "
            "2.0 (Device API >= 70). Adapters from MM 1.4 will fail to load."
        )


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

        resolved = _resolve_adapter_paths(adapter_paths)
        _logger.info(f"Setting MMCore adapter search paths: {resolved}")
        _add_windows_dll_dirs(resolved)
        core.setDeviceAdapterSearchPaths(resolved)
        _verify_mm2(core)

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
        resolved = _resolve_adapter_paths(adapter_paths)
        _logger.info(f"Setting MMCore adapter search paths: {resolved}")
        _add_windows_dll_dirs(resolved)
        core.setDeviceAdapterSearchPaths(resolved)
        _verify_mm2(core)
    return core


def get_available_adapters(adapter_path: Optional[str] = None) -> List[str]:
    """List the human-readable adapter names available on disk.

    Scans ``adapter_path`` (or every directory returned by
    :func:`discover_adapter_paths`) for ``libmmgr_dal_*.so`` /
    ``mmgr_dal_*.dll`` / ``libmmgr_dal_*.dylib`` files and returns the
    bare adapter names.
    """
    if adapter_path:
        search_paths = [adapter_path]
    else:
        search_paths = discover_adapter_paths()

    adapters = set()
    for path in search_paths:
        if not os.path.isdir(path):
            continue
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
