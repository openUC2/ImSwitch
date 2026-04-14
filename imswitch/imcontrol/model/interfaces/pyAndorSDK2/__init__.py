"""
Vendored copy of pyAndorSDK2, bundled inside ImSwitch.

The Andor atmcd DLL (atmcd64d.dll on 64-bit Windows) must already be
installed by the Andor SDK2 installer.  This module adds the default
SDK2 install directories to PATH so ctypes can find the DLL, while also
respecting a user-supplied ANDOR_SDK_PATH environment variable.

No DLL files are shipped here; only the pure-Python wrapper is bundled.
"""

import os
import platform

# -----------------------------------------------------------------------
# Locate the Andor SDK2 DLL at runtime
# -----------------------------------------------------------------------
_candidate_dirs: list[str] = []

# 1. Allow explicit override via env variable
_env = os.environ.get("ANDOR_SDK_PATH", "")
if _env:
    _candidate_dirs.append(_env)

# 2. Common Windows SDK2 install locations
if platform.system() == "Windows":
    _candidate_dirs += [
        r"C:\Program Files\Andor SDK",
        r"C:\Program Files (x86)\Andor SDK",
        r"C:\Program Files\Andor SDK2",
        r"C:\Program Files (x86)\Andor SDK2",
    ]

for _d in _candidate_dirs:
    if os.path.isdir(_d) and _d not in os.environ.get("PATH", ""):
        os.environ["PATH"] = _d + os.pathsep + os.environ.get("PATH", "")

# -----------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------
from imswitch.imcontrol.model.interfaces.pyAndorSDK2._version import (
    __version__,
    __version_info__,
)
from imswitch.imcontrol.model.interfaces.pyAndorSDK2.atmcd import atmcd

__title__     = "pyAndorSDK2"
__authors__   = "Andor SDK2 team"
__email__     = "row_productsupport@andor.com"
__license__   = "Andor internal"
__copyright__ = "Copyright 2017 Andor"

__all__ = [
    "atmcd",
    "__title__",
    "__authors__",
    "__email__",
    "__license__",
    "__copyright__",
    "__version__",
    "__version_info__",
]
