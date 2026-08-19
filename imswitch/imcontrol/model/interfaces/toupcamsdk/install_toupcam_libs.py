#!/usr/bin/env python3
"""Copy the native Toupcam libraries from a downloaded vendor SDK into this
package's ``lib/`` directory so ImSwitch can load them automatically.

The vendor SDK (toupcamsdk.zip from touptek.com) unpacks to a folder with
``linux/``, ``mac/``, ``win/``, ``python/`` subdirectories. Point this script
at that folder:

    python install_toupcam_libs.py /path/to/toupcamsdk           # current platform only
    python install_toupcam_libs.py /path/to/toupcamsdk --platforms linux/arm64 linux/x64 mac
    python install_toupcam_libs.py /path/to/toupcamsdk --all     # every platform in the SDK

On Linux, also install the udev rules once (USB permission for non-root):

    sudo cp 99-toupcam.rules /etc/udev/rules.d/
    # then unplug/replug the camera

Committing the copied libraries to git is optional: they are large (~20-65 MB
each) but committing e.g. ``lib/linux/arm64`` makes Raspberry Pi deployments
work straight after ``git pull``. Alternatively set ``TOUPCAM_SDK_DIR`` or
``IMSWITCH_TOUPCAM_LIB`` on the target machine (see libloader.py).
"""

import argparse
import os
import shutil
import sys

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
LIB_DIR = os.path.join(PKG_DIR, "lib")

# (sdk-relative source file, lib/-relative destination) per platform tag
PLATFORM_LIBS = {
    "mac": [("mac/libtoupcam.dylib", "mac/libtoupcam.dylib")],
    "win/x64": [("win/x64/toupcam.dll", "win/x64/toupcam.dll")],
    "win/x86": [("win/x86/toupcam.dll", "win/x86/toupcam.dll")],
    "win/arm64": [("win/arm64/toupcam.dll", "win/arm64/toupcam.dll")],
    "linux/x64": [("linux/x64/libtoupcam.so", "linux/x64/libtoupcam.so")],
    "linux/x86": [("linux/x86/libtoupcam.so", "linux/x86/libtoupcam.so")],
    "linux/arm64": [
        ("linux/arm64/glibc/libtoupcam.so", "linux/arm64/glibc/libtoupcam.so"),
        ("linux/arm64/musl/libtoupcam.so", "linux/arm64/musl/libtoupcam.so"),
    ],
    "linux/armhf": [("linux/armhf/libtoupcam.so", "linux/armhf/libtoupcam.so")],
    "linux/armel": [("linux/armel/libtoupcam.so", "linux/armel/libtoupcam.so")],
}


def _current_platform_tag():
    import platform as _pf

    m = _pf.machine().lower()
    if sys.platform == "darwin":
        return "mac"
    if sys.platform == "win32":
        if m in ("amd64", "x86_64"):
            return "win/x64"
        if m in ("arm64", "aarch64"):
            return "win/arm64"
        return "win/x86"
    if m in ("x86_64", "amd64"):
        return "linux/x64"
    if m in ("aarch64", "arm64"):
        return "linux/arm64"
    if m.startswith("armv"):
        return "linux/armhf"
    return "linux/x64"


def install(sdk_root, platforms):
    copied, missing = [], []
    for tag in platforms:
        for src_rel, dst_rel in PLATFORM_LIBS[tag]:
            src = os.path.join(sdk_root, src_rel)
            if not os.path.isfile(src):
                missing.append(src)
                continue
            dst = os.path.join(LIB_DIR, dst_rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            if sys.platform == "darwin":
                # downloaded SDKs carry com.apple.quarantine, which makes
                # dlopen hang/fail under Gatekeeper — strip it
                import subprocess
                subprocess.run(["xattr", "-d", "com.apple.quarantine", dst],
                               capture_output=True)
            copied.append((src, dst, os.path.getsize(dst)))
    return copied, missing


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("sdk_root", help="path to the unpacked vendor SDK folder")
    parser.add_argument(
        "--platforms",
        nargs="+",
        choices=sorted(PLATFORM_LIBS),
        help="platform tags to install (default: current platform)",
    )
    parser.add_argument("--all", action="store_true", help="install every platform")
    args = parser.parse_args()

    if not os.path.isdir(args.sdk_root):
        parser.error(f"SDK folder not found: {args.sdk_root}")

    if args.all:
        platforms = sorted(PLATFORM_LIBS)
    elif args.platforms:
        platforms = args.platforms
    else:
        platforms = [_current_platform_tag()]

    copied, missing = install(args.sdk_root, platforms)
    for src, dst, size in copied:
        print(f"copied {src} -> {dst} ({size / 1e6:.1f} MB)")
    for src in missing:
        print(f"NOT FOUND in SDK: {src}")
    if sys.platform.startswith("linux"):
        print(
            "\nReminder (Linux): install udev rules once for USB access:\n"
            f"  sudo cp {os.path.join(PKG_DIR, '99-toupcam.rules')} /etc/udev/rules.d/\n"
            "  then unplug/replug the camera."
        )
    if not copied:
        sys.exit(1)


if __name__ == "__main__":
    main()
