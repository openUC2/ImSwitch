"""Locate the native Toupcam library (libtoupcam.so / .dylib / toupcam.dll)
for the current platform and CPU architecture.

Search order (first hit wins):

1. ``IMSWITCH_TOUPCAM_LIB`` environment variable — full path to the library file.
2. ``TOUPCAM_SDK_DIR`` (or ``IMSWITCH_TOUPCAM_SDK``) environment variable —
   path to an unpacked vendor SDK folder (the one containing ``linux/``,
   ``mac/``, ``win/``); the right sub-path is picked automatically.
3. Libraries bundled inside this package under ``lib/`` (populated by
   ``install_toupcam_libs.py`` from a downloaded SDK; mirrors the SDK layout).
4. Common system install locations (``/usr/local/lib`` etc.). On Windows the
   plain DLL name is left to the system search path.

The vendored ``toupcam.py`` consults :func:`find_toupcam_library` before its
own default logic, so dropping the correct library into any of the above
locations is all that is needed — including ``linux/arm64`` for Raspberry Pi.
"""

import os
import platform
import sys

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))

_ENV_LIB = "IMSWITCH_TOUPCAM_LIB"
_ENV_SDK = ("TOUPCAM_SDK_DIR", "IMSWITCH_TOUPCAM_SDK")


def _lib_filename():
    if sys.platform == "win32":
        return "toupcam.dll"
    if sys.platform == "darwin":
        return "libtoupcam.dylib"
    return "libtoupcam.so"


def _machine_arch():
    """Normalized CPU architecture tag matching the vendor SDK layout."""
    m = platform.machine().lower()
    if m in ("x86_64", "amd64"):
        return "x64"
    if m in ("aarch64", "arm64"):
        return "arm64"
    if m.startswith("armv7") or m.startswith("armv6"):
        # Raspberry Pi OS 32-bit is hard-float on both v6 and v7
        return "armhf"
    if m.startswith("arm"):
        return "armel"
    if m in ("i386", "i486", "i586", "i686", "x86"):
        return "x86"
    return m


def _is_musl():
    try:
        libc, _ = platform.libc_ver()
        return libc != "glibc"
    except Exception:
        return False


def _linux_subdirs(arch):
    """Candidate sub-paths below an SDK-style root for Linux, best first."""
    if arch == "arm64":
        flavors = ["musl", "glibc"] if _is_musl() else ["glibc", "musl"]
        subdirs = [os.path.join("linux", "arm64", f) for f in flavors]
        subdirs.append(os.path.join("linux", "arm64"))  # flat fallback
        return subdirs
    return [os.path.join("linux", arch)]


def _sdk_layout_candidates(root):
    """Paths of the native lib inside an SDK-style directory tree."""
    fname = _lib_filename()
    candidates = []
    if sys.platform == "win32":
        candidates.append(os.path.join(root, "win", _machine_arch(), fname))
    elif sys.platform == "darwin":
        candidates.append(os.path.join(root, "mac", fname))
    else:
        for sub in _linux_subdirs(_machine_arch()):
            candidates.append(os.path.join(root, sub, fname))
    # also accept a flat folder containing the lib directly
    candidates.append(os.path.join(root, fname))
    return candidates


def _system_candidates():
    fname = _lib_filename()
    if sys.platform == "win32":
        return []  # leave DLL resolution to the system search path
    if sys.platform == "darwin":
        return [
            os.path.join(p, fname)
            for p in ("/usr/local/lib", "/opt/homebrew/lib", "/Library/Frameworks")
        ]
    triplet = {
        "x64": "x86_64-linux-gnu",
        "arm64": "aarch64-linux-gnu",
        "armhf": "arm-linux-gnueabihf",
        "armel": "arm-linux-gnueabi",
        "x86": "i386-linux-gnu",
    }.get(_machine_arch())
    paths = ["/usr/local/lib", "/usr/lib"]
    if triplet:
        paths.insert(1, os.path.join("/usr/lib", triplet))
    return [os.path.join(p, fname) for p in paths]


def find_toupcam_library():
    """Return the absolute path of the native Toupcam library, or None.

    None means: fall back to the wrapper's default lookup (file directory,
    then system loader search path).
    """
    # 1. explicit file path via env var
    env_lib = os.environ.get(_ENV_LIB)
    if env_lib and os.path.isfile(env_lib):
        return env_lib

    candidates = []

    # 2. unpacked vendor SDK folder via env var
    for env in _ENV_SDK:
        root = os.environ.get(env)
        if root and os.path.isdir(root):
            candidates.extend(_sdk_layout_candidates(root))

    # 3. libs bundled inside this package (SDK layout below lib/)
    candidates.extend(_sdk_layout_candidates(os.path.join(_PKG_DIR, "lib")))

    # 4. common system locations
    candidates.extend(_system_candidates())

    for path in candidates:
        if os.path.isfile(path):
            return path
    return None
