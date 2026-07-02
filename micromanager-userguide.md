# Micro-Manager + pymmcore-plus on Raspberry Pi: Usage Guide

## Table of Contents

1. [Using pymmcore-plus from Python (standalone, no ImSwitch)](#1-using-pymmcore-plus-from-python)
2. [Plugging into ImSwitch as a camera/stage/laser](#2-plugging-into-imswitch)
3. [Prebuilt binaries via GitHub Actions CI](#3-prebuilt-binaries)
4. [There is no "start Micro-Manager" step](#4-no-server-needed)

---

## 1. Using pymmcore-plus from Python

There is **no daemon or server to start**. Unlike Pycro-Manager (which talks
to a running Java process), pymmcore-plus loads the C++ MMCore library
directly into your Python process. You just `import` it and go.

### 1.1 Basic camera usage

```python
#!/usr/bin/env python3
"""Snap images from a Micro-Manager camera via pymmcore-plus."""

import os
import numpy as np
from pymmcore_plus import CMMCorePlus

# ── Instantiate the core (singleton — safe to call multiple times) ──
core = CMMCorePlus.instance()

# ── Tell it where the compiled .so adapter files live ──
core.setDeviceAdapterSearchPaths([
    os.environ.get("MICROMANAGER_PATH", "/opt/micro-manager/lib/micro-manager")
])

# ── Option A: load a full .cfg file ──
core.loadSystemConfiguration("/home/pi/micro-manager-configs/MMConfig_demo.cfg")

# ── Option B: load devices manually (more explicit) ──
# core.loadDevice("Camera", "DemoCamera", "DCam")
# core.initializeAllDevices()
# core.setCameraDevice("Camera")

# ── Snap a single frame ──
img = core.snap()  # returns numpy ndarray, shape (H, W) or (H, W, C)
print(f"Image shape: {img.shape}, dtype: {img.dtype}")

# ── Change exposure ──
core.setExposure(50.0)  # milliseconds
print(f"Exposure: {core.getExposure()} ms")

# ── Snap with new exposure ──
img2 = core.snap()
print(f"Mean intensity: {img2.mean():.1f}")
```

### 1.2 Continuous (live) acquisition

```python
import time
from pymmcore_plus import CMMCorePlus

core = CMMCorePlus.instance()
# ... (load config as above) ...

# Start free-running acquisition (0 = no interval, as fast as possible)
core.startContinuousSequenceAcquisition(0)

frames = []
t0 = time.perf_counter()
while len(frames) < 100:
    if core.getRemainingImageCount() > 0:
        frames.append(core.getLastImage())

elapsed = time.perf_counter() - t0
core.stopSequenceAcquisition()

print(f"Captured {len(frames)} frames in {elapsed:.2f}s = {len(frames)/elapsed:.1f} FPS")
```

### 1.3 Stage control

```python
core = CMMCorePlus.instance()
# ... (load config) ...

# ── XY Stage ──
x, y = core.getXPosition(), core.getYPosition()
print(f"Current XY: ({x:.1f}, {y:.1f}) µm")

core.setXYPosition(x + 100.0, y + 50.0)
core.waitForDevice(core.getXYStageDevice())  # block until move completes

print(f"New XY: ({core.getXPosition():.1f}, {core.getYPosition():.1f}) µm")

# ── Z / Focus ──
z = core.getPosition()  # uses the default focus device
core.setPosition(z + 10.0)
core.waitForDevice(core.getFocusDevice())
print(f"Z: {z:.1f} -> {core.getPosition():.1f} µm")

# ── Relative move (convenience) ──
core.setRelativeXYPosition(50.0, 0.0)
core.waitForDevice(core.getXYStageDevice())
```

### 1.4 Laser / shutter control

```python
core = CMMCorePlus.instance()
# ... (load config) ...

# ── Shutter (binary on/off) ──
core.setShutterOpen(True)
print(f"Shutter open: {core.getShutterOpen()}")
core.setShutterOpen(False)

# ── Analog property (e.g. laser power via a DA device) ──
# This depends on your .cfg — example for a device called "LaserDA":
# core.setProperty("LaserDA", "Volts", 2.5)

# ── Enumerate all properties of a device ──
for prop in core.getDevicePropertyNames("Camera"):
    val = core.getProperty("Camera", prop)
    print(f"  {prop} = {val}")
```

### 1.5 Listing available device adapters

```python
import os
mm_path = "/opt/micro-manager/lib/micro-manager"
adapters = [f.replace("libmmgr_dal_", "").replace(".so", "")
            for f in os.listdir(mm_path)
            if f.startswith("libmmgr_dal_") and f.endswith(".so")]
print(f"Available adapters ({len(adapters)}):")
for a in sorted(adapters):
    print(f"  {a}")
```

---

## 2. Plugging into ImSwitch

ImSwitch discovers device managers by the `"managerName"` string in the setup
JSON — it maps directly to a Python class. You create new manager classes that
inherit from ImSwitch's base classes and wrap pymmcore-plus calls.

### 2.1 Architecture overview

```
ImSwitch setup JSON
    │
    ├── "MMCoreDetectorManager"  ──►  MMCoreDetectorManager.py
    ├── "MMCorePositionerManager" ──► MMCorePositionerManager.py
    └── "MMCoreLaserManager"     ──►  MMCoreLaserManager.py
                                          │
                                          ▼
                                   MMCoreManager.py  (shared singleton)
                                          │
                                          ▼
                                   CMMCorePlus.instance()
                                          │
                                          ▼
                                   libmmgr_dal_*.so  (device adapters)
```

The pattern is the same as the existing ESP32 managers: the ESP32 managers go
through an `RS232Manager` (serial), while the MM managers go through a shared
`CMMCorePlus` singleton. You do NOT need an RS232/serial layer — pymmcore-plus
talks directly to USB/serial devices via the compiled C++ adapters.

### 2.2 The shared core singleton

Place this in `imswitch/imcontrol/model/managers/MMCoreManager.py`:

```python
"""
Process-wide singleton for the Micro-Manager CMMCorePlus core.

All MMCore*Manager classes share this instance so that a single .cfg
file drives camera + stage + laser without conflicts.
"""

import os
import threading
import logging

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_loaded_cfg = None
_core = None


def get_core():
    """Return the CMMCorePlus singleton, creating it on first call."""
    global _core
    if _core is None:
        from pymmcore_plus import CMMCorePlus
        _core = CMMCorePlus.instance()
    return _core


def ensure_loaded(cfg_path: str, adapter_paths: list = None):
    """
    Load a .cfg file exactly once. Subsequent calls with the same path
    are no-ops. Call with a different path to reload.
    """
    global _loaded_cfg
    with _lock:
        if _loaded_cfg == cfg_path:
            return get_core()

        core = get_core()

        # Set adapter search paths
        if adapter_paths:
            core.setDeviceAdapterSearchPaths(adapter_paths)
        else:
            default = os.environ.get(
                "MICROMANAGER_PATH", "/opt/micro-manager/lib/micro-manager"
            )
            core.setDeviceAdapterSearchPaths([default])

        # Unload any previously loaded config
        if _loaded_cfg is not None:
            core.unloadAllDevices()

        core.loadSystemConfiguration(cfg_path)
        _loaded_cfg = cfg_path

        devices = core.getLoadedDevices()
        logger.info(f"MMCore loaded {len(devices)} devices from {cfg_path}: "
                     f"{', '.join(devices)}")
        return core


def reload(cfg_path: str, adapter_paths: list = None):
    """Force-reload a config (e.g. when user switches setup in ImSwitch)."""
    global _loaded_cfg
    with _lock:
        _loaded_cfg = None  # force re-load
    return ensure_loaded(cfg_path, adapter_paths)
```

### 2.3 The Detector Manager (camera)

Place in `imswitch/imcontrol/model/managers/detectors/MMCoreDetectorManager.py`:

```python
"""
ImSwitch DetectorManager that wraps a Micro-Manager camera device
via pymmcore-plus.

Setup JSON example:
{
  "detectors": {
    "MMCamera": {
      "managerName": "MMCoreDetectorManager",
      "managerProperties": {
        "cfgPath": "/home/pi/mm_configs/my_scope.cfg",
        "deviceLabel": "Camera",
        "adapterPath": "/opt/micro-manager/lib/micro-manager"
      },
      "forAcquisition": true
    }
  }
}
"""

import numpy as np
from imswitch.imcontrol.model.managers.detectors.DetectorManager import (
    DetectorManager, DetectorNumberParameter
)
from imswitch.imcontrol.model.managers import MMCoreManager


class MMCoreDetectorManager(DetectorManager):

    def __init__(self, detectorInfo, name, **lowLevelManagers):
        self._props = detectorInfo.managerProperties

        # Initialize the shared core
        self._core = MMCoreManager.ensure_loaded(
            cfg_path=self._props["cfgPath"],
            adapter_paths=[self._props.get("adapterPath")] if self._props.get("adapterPath") else None
        )

        # Wire up the camera device
        self._label = self._props.get("deviceLabel", self._core.getCameraDevice())
        if self._label:
            self._core.setCameraDevice(self._label)

        # Read sensor dimensions
        w = self._core.getImageWidth()
        h = self._core.getImageHeight()
        fullShape = (w, h)

        # Build parameter dict — expose exposure time as a tunable parameter
        parameters = {
            "Exposure": DetectorNumberParameter(
                group="Acquisition",
                value=self._core.getExposure(),
                valueUnits="ms",
                editable=True
            ),
        }

        model = self._label or "MMCore Camera"

        super().__init__(
            detectorInfo, name,
            fullShape=fullShape,
            supportedBinnings=[1, 2, 4],
            model=model,
            parameters=parameters,
            croppable=True,
            **lowLevelManagers
        )

    def getLatestFrame(self):
        """Return the most recent frame as a numpy array."""
        return self._core.getLastImage()

    def setParameter(self, name, value):
        if name == "Exposure":
            self._core.setExposure(float(value))
        super().setParameter(name, value)
        return self.parameters

    def startAcquisition(self):
        self._core.startContinuousSequenceAcquisition(0)

    def stopAcquisition(self):
        if self._core.isSequenceRunning():
            self._core.stopSequenceAcquisition()

    def crop(self, hpos, vpos, hsize, vsize):
        """Set ROI on the camera."""
        self._core.setROI(self._label, hpos, vpos, hsize, vsize)
        return True

    @property
    def pixelSizeUm(self):
        ps = self._core.getPixelSizeUm()
        return [1, ps, ps] if ps > 0 else [1, 1, 1]

    def finalize(self):
        self.stopAcquisition()
```

### 2.4 The Positioner Manager (stage)

Place in `imswitch/imcontrol/model/managers/positioners/MMCorePositionerManager.py`:

```python
"""
ImSwitch PositionerManager wrapping Micro-Manager XY and Z stages.

Setup JSON example:
{
  "positioners": {
    "MMStage": {
      "managerName": "MMCorePositionerManager",
      "managerProperties": {
        "cfgPath": "/home/pi/mm_configs/my_scope.cfg",
        "xyDeviceLabel": "XYStage",
        "zDeviceLabel": "ZStage"
      },
      "axes": ["X", "Y", "Z"],
      "forScanning": true,
      "forPositioning": true
    }
  }
}
"""

from imswitch.imcontrol.model.managers.positioners.PositionerManager import PositionerManager
from imswitch.imcontrol.model.managers import MMCoreManager


class MMCorePositionerManager(PositionerManager):

    def __init__(self, positionerInfo, name, **lowLevelManagers):
        self._props = positionerInfo.managerProperties

        self._core = MMCoreManager.ensure_loaded(
            cfg_path=self._props["cfgPath"],
            adapter_paths=[self._props.get("adapterPath")] if self._props.get("adapterPath") else None
        )

        self._xy_label = self._props.get("xyDeviceLabel")
        self._z_label = self._props.get("zDeviceLabel")

        if self._xy_label:
            self._core.setXYStageDevice(self._xy_label)
        if self._z_label:
            self._core.setFocusDevice(self._z_label)

        # Units are micrometers (MM's native unit)
        super().__init__(positionerInfo, name, initialPosition={
            "X": self._core.getXPosition() if self._xy_label else 0,
            "Y": self._core.getYPosition() if self._xy_label else 0,
            "Z": self._core.getPosition() if self._z_label else 0,
        }, **lowLevelManagers)

    def move(self, dist, axis):
        """Relative move in micrometers."""
        if axis in ("X", "Y") and self._xy_label:
            dx = dist if axis == "X" else 0
            dy = dist if axis == "Y" else 0
            self._core.setRelativeXYPosition(dx, dy)
            self._core.waitForDevice(self._xy_label)
        elif axis == "Z" and self._z_label:
            self._core.setRelativePosition(dist)
            self._core.waitForDevice(self._z_label)
        self._position[axis] = self.getPosition(axis)

    def setPosition(self, position, axis):
        """Absolute move in micrometers."""
        if axis == "X" and self._xy_label:
            self._core.setXYPosition(position, self._core.getYPosition())
            self._core.waitForDevice(self._xy_label)
        elif axis == "Y" and self._xy_label:
            self._core.setXYPosition(self._core.getXPosition(), position)
            self._core.waitForDevice(self._xy_label)
        elif axis == "Z" and self._z_label:
            self._core.setPosition(position)
            self._core.waitForDevice(self._z_label)
        self._position[axis] = position

    def getPosition(self, axis):
        if axis == "X" and self._xy_label:
            return self._core.getXPosition()
        elif axis == "Y" and self._xy_label:
            return self._core.getYPosition()
        elif axis == "Z" and self._z_label:
            return self._core.getPosition()
        return 0.0

    def finalize(self):
        pass
```

### 2.5 The Laser Manager

Place in `imswitch/imcontrol/model/managers/lasers/MMCoreLaserManager.py`:

```python
"""
ImSwitch LaserManager wrapping a Micro-Manager shutter or analog device.

Setup JSON example (shutter mode):
{
  "lasers": {
    "MMLaser": {
      "managerName": "MMCoreLaserManager",
      "managerProperties": {
        "cfgPath": "/home/pi/mm_configs/my_scope.cfg",
        "mode": "shutter",
        "deviceLabel": "Shutter"
      },
      "wavelength": 488,
      "valueRangeMin": 0,
      "valueRangeMax": 1
    }
  }
}

Setup JSON example (analog/property mode):
{
  "lasers": {
    "Laser488": {
      "managerName": "MMCoreLaserManager",
      "managerProperties": {
        "cfgPath": "/home/pi/mm_configs/my_scope.cfg",
        "mode": "property",
        "deviceLabel": "LaserDA",
        "propertyName": "Volts"
      },
      "wavelength": 488,
      "valueRangeMin": 0,
      "valueRangeMax": 5
    }
  }
}
"""

from imswitch.imcontrol.model.managers.lasers.LaserManager import LaserManager
from imswitch.imcontrol.model.managers import MMCoreManager


class MMCoreLaserManager(LaserManager):

    def __init__(self, laserInfo, name, **lowLevelManagers):
        self._props = laserInfo.managerProperties

        self._core = MMCoreManager.ensure_loaded(
            cfg_path=self._props["cfgPath"],
            adapter_paths=[self._props.get("adapterPath")] if self._props.get("adapterPath") else None
        )

        self._mode = self._props.get("mode", "shutter")  # "shutter" or "property"
        self._label = self._props["deviceLabel"]
        self._property = self._props.get("propertyName", "Volts")

        is_binary = (self._mode == "shutter")
        value_units = "" if is_binary else "V"

        super().__init__(
            laserInfo, name,
            isBinary=is_binary,
            valueUnits=value_units,
            valueDecimals=2,
            **lowLevelManagers
        )

    def setEnabled(self, enabled):
        if self._mode == "shutter":
            self._core.setShutterDevice(self._label)
            self._core.setShutterOpen(enabled)
        elif self._mode == "property":
            if not enabled:
                self._core.setProperty(self._label, self._property, 0)

    def setValue(self, value):
        if self._mode == "property":
            self._core.setProperty(self._label, self._property, value)

    def finalize(self):
        self.setEnabled(False)
```

### 2.6 Example setup JSON for ImSwitch

Save as `~/ImSwitchConfig/imcontrol_setups/example_pymmcore_demo.json`:

```json
{
  "rs232devices": {},
  "lasers": {
    "MMShutter": {
      "managerName": "MMCoreLaserManager",
      "managerProperties": {
        "cfgPath": "/home/pi/micro-manager-configs/MMConfig_demo.cfg",
        "mode": "shutter",
        "deviceLabel": "Shutter"
      },
      "wavelength": 488,
      "valueRangeMin": 0,
      "valueRangeMax": 1
    }
  },
  "positioners": {
    "MMStage": {
      "managerName": "MMCorePositionerManager",
      "managerProperties": {
        "cfgPath": "/home/pi/micro-manager-configs/MMConfig_demo.cfg",
        "xyDeviceLabel": "XY",
        "zDeviceLabel": "Z"
      },
      "axes": ["X", "Y", "Z"],
      "forScanning": true,
      "forPositioning": true
    }
  },
  "detectors": {
    "MMCamera": {
      "managerName": "MMCoreDetectorManager",
      "managerProperties": {
        "cfgPath": "/home/pi/micro-manager-configs/MMConfig_demo.cfg",
        "deviceLabel": "Camera"
      },
      "forAcquisition": true
    }
  },
  "nipiezzos": {},
  "nidaqmanager": null,
  "rois": {},
  "designerId": null
}
```

Then in `~/ImSwitchConfig/config/imcontrol_options.json`:

```json
{
  "setupFileName": "example_pymmcore_demo.json"
}
```

### 2.7 Registering the new managers

ImSwitch discovers managers by importing them from the relevant package.
You need to add imports in the `__init__.py` files:

```python
# In imswitch/imcontrol/model/managers/detectors/__init__.py, add:
from .MMCoreDetectorManager import MMCoreDetectorManager

# In imswitch/imcontrol/model/managers/positioners/__init__.py, add:
from .MMCorePositionerManager import MMCorePositionerManager

# In imswitch/imcontrol/model/managers/lasers/__init__.py, add:
from .MMCoreLaserManager import MMCoreLaserManager
```

Guard these imports so ImSwitch doesn't crash when pymmcore-plus is not installed:

```python
try:
    from .MMCoreDetectorManager import MMCoreDetectorManager
except ImportError:
    pass  # pymmcore-plus not installed, MMCore managers unavailable
```

---

## 3. Prebuilt binaries via GitHub Actions CI

You're right that compiling from source every time is painful. The solution is
to build once in CI and publish a tarball as a GitHub Release that you can
just download and extract on any Pi.

### 3.1 What goes in the tarball

```
micro-manager-arm64-<version>.tar.gz
└── micro-manager/
    ├── lib/
    │   └── micro-manager/
    │       ├── libmmgr_dal_DemoCamera.so
    │       ├── libmmgr_dal_SerialManager.so
    │       ├── libmmgr_dal_... .so
    │       └── libMMCore.so
    ├── share/
    │   └── micro-manager/
    │       └── MMConfig_demo.cfg
    └── install.sh   # simple script: extract, pip install pymmcore-plus
```

### 3.2 GitHub Actions workflow

Save as `.github/workflows/build-mm-arm64.yml`:

```yaml
name: Build Micro-Manager for arm64

on:
  push:
    tags: ["v*"]       # trigger on version tags
  workflow_dispatch:    # manual trigger

jobs:
  build:
    runs-on: ubuntu-24.04
    steps:

    - name: Checkout
      uses: actions/checkout@v4

    - name: Set up QEMU (for arm64 emulation)
      uses: docker/setup-qemu-action@v3
      with:
        platforms: arm64

    - name: Build in arm64 container
      uses: uraimo/run-on-arch-action@v3
      id: build
      with:
        arch: aarch64
        distro: bookworm      # matches Pi OS Bookworm
        githubToken: ${{ github.token }}

        # These packages get cached in the container image layer
        install: |
          apt-get update -y
          apt-get install -y \
            build-essential autoconf automake libtool autoconf-archive \
            pkg-config git swig libboost-all-dev \
            python3-dev python3-pip python3-numpy \
            libusb-1.0-0-dev libudev-dev

        run: |
          set -euxo pipefail

          # Clone repos
          git clone --depth 1 https://github.com/micro-manager/micro-manager.git /build/mm
          cd /build/mm
          git submodule update --init --recursive

          # Build
          cd /build/mm
          ./autogen.sh
          ./configure \
            --prefix=/opt/micro-manager \
            --without-java \
            --disable-java-app
          make -j$(nproc)
          make install DESTDIR=/build/staging

          # Package
          cd /build/staging
          tar czf /artifacts/micro-manager-arm64.tar.gz opt/micro-manager/

        dockerRunArgs: |
          --volume ${{ github.workspace }}/artifacts:/artifacts

    - name: Create install script
      run: |
        mkdir -p artifacts
        cat > artifacts/install.sh << 'INSTALLER'
        #!/usr/bin/env bash
        set -euo pipefail
        echo "Installing Micro-Manager for arm64..."
        sudo tar xzf micro-manager-arm64.tar.gz -C /
        echo "export MICROMANAGER_PATH=/opt/micro-manager/lib/micro-manager" >> ~/.bashrc
        pip install "pymmcore-plus[cli]" --break-system-packages 2>/dev/null || \
          pip install "pymmcore-plus[cli]"
        echo "Done! Run: source ~/.bashrc"
        INSTALLER
        chmod +x artifacts/install.sh

    - name: Upload artifact
      uses: actions/upload-artifact@v4
      with:
        name: micro-manager-arm64
        path: artifacts/

    - name: Create Release
      if: startsWith(github.ref, 'refs/tags/')
      uses: softprops/action-gh-release@v2
      with:
        files: |
          artifacts/micro-manager-arm64.tar.gz
          artifacts/install.sh
```

### 3.3 Using the prebuilt binary on a Pi

After the CI builds and publishes a release:

```bash
# Download from GitHub releases (replace with your repo/tag)
wget https://github.com/YOUR_ORG/YOUR_REPO/releases/download/v1.0/micro-manager-arm64.tar.gz
wget https://github.com/YOUR_ORG/YOUR_REPO/releases/download/v1.0/install.sh

chmod +x install.sh
bash install.sh
source ~/.bashrc

# Verify
python3 -c "
from pymmcore_plus import CMMCorePlus
import os
core = CMMCorePlus.instance()
core.setDeviceAdapterSearchPaths([os.environ['MICROMANAGER_PATH']])
core.loadDevice('Cam', 'DemoCamera', 'DCam')
core.initializeAllDevices()
core.setCameraDevice('Cam')
print(f'snap shape: {core.snap().shape}')
print('SUCCESS')
"
```

### 3.4 Important: pymmcore device interface version pinning

The compiled `.so` adapters and the `pymmcore` Python wheel MUST share the
same device interface version. If they don't match, you'll see errors like
`Failed to load device adapter: interface version mismatch`.

Pin the version in the CI workflow by checking out a specific tag of
`mmCoreAndDevices` and installing the matching `pymmcore` version. You can find
the device interface version in `MMDevice/MMDeviceConstants.h`:

```bash
grep "MODULE_INTERFACE_VERSION" mmCoreAndDevices/MMDevice/MMDeviceConstants.h
```

Then in the install script:
```bash
pip install "pymmcore==XX.Y.Z"  # matching version
```

---

## 4. There is no "start Micro-Manager" step

This is the most common point of confusion. With pymmcore-plus, **there is no
separate Micro-Manager process to launch.** Here's the comparison:

| Approach | What runs | How Python talks to it |
|----------|-----------|----------------------|
| **MMStudio (Java GUI)** | Java process with a GUI window | Not applicable (it IS the GUI) |
| **Pycro-Manager** | Java MMStudio process in background | ZMQ bridge over TCP |
| **pymmcore-plus** | Nothing separate — it's a Python library | Direct C++ calls via SWIG bindings |

So after running the install script:

```bash
# Activate your venv (if you used one)
source ~/mm-venv/bin/activate

# Just run Python
python3 my_script.py
# or
python3 -c "from pymmcore_plus import CMMCorePlus; print('ready')"
# or start ImSwitch
python3 -m imswitch
```

The `CMMCorePlus.instance()` call loads the C++ library into your Python
process. `loadSystemConfiguration()` opens serial ports, USB connections, etc.
to the real hardware. That's it — no daemon, no server, no Java.

### When to set environment variables

The only setup needed before running Python:

```bash
# Tell pymmcore-plus where the compiled .so adapter files are
export MICROMANAGER_PATH="/opt/micro-manager/lib/micro-manager"

# Or set it in your script:
# core.setDeviceAdapterSearchPaths(["/opt/micro-manager/lib/micro-manager"])
```

### Quick-start one-liner after install

```bash
source ~/mm-venv/bin/activate && \
MICROMANAGER_PATH=/opt/micro-manager/lib/micro-manager \
python3 ~/micro-manager-configs/smoke_test.py
```

---

## Summary of the workflow

```
1. Install once (build script OR download prebuilt tarball)
      │
      ▼
2. pip install pymmcore-plus (matching version)
      │
      ▼
3. Write Python script / ImSwitch manager
      │
      ├── from pymmcore_plus import CMMCorePlus
      ├── core.setDeviceAdapterSearchPaths([...])
      ├── core.loadSystemConfiguration("my_scope.cfg")
      ├── core.snap() / core.setXYPosition() / etc.
      │
      ▼
4. Run it: python3 my_script.py
   (or: python3 -m imswitch)
```

No daemon. No server. No Java. Just Python + compiled C++ adapters.