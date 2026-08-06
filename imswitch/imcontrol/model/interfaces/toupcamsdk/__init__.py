"""Vendored ToupTek (Toupcam) camera SDK for ImSwitch.

- ``toupcam.py``  — official ctypes wrapper from the vendor SDK
  (version 60.31631.20260606), with a small marked patch in
  ``Toupcam.__initlib`` that resolves the native library through
  :mod:`.libloader` first.
- ``libloader.py`` — locates libtoupcam for the current OS/architecture
  (linux x64 / arm64 glibc+musl / armhf, macOS, Windows).
- ``install_toupcam_libs.py`` — copies the native libraries from a downloaded
  vendor SDK folder into ``lib/`` (run once per checkout / deployment).
- ``99-toupcam.rules`` — udev rules granting USB access on Linux.

Usage::

    from imswitch.imcontrol.model.interfaces.toupcamsdk import toupcam
    devices = toupcam.Toupcam.EnumV2()
"""

from .libloader import find_toupcam_library

__all__ = ["find_toupcam_library", "toupcam"]
