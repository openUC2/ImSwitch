# CLAUDE.md — pymmcore-plus Integration into ImSwitch

You are implementing Micro-Manager hardware support in the openUC2/ImSwitch
microscopy platform via `pymmcore-plus`. This adds support for any camera,
stage, or laser that has a Micro-Manager device adapter (Andor, Hamamatsu,
Basler, ASI, Prior, Thorlabs, Coherent, etc.) alongside the existing
ESP32/UC2-REST managers.


## STEP 0 — Reconnaissance (do this FIRST, before writing any code)

Run these commands and summarize what you find. Paste the summary as a
comment in the first commit.

```bash
# Check for any existing pymmcore/MMCore work
rg -n -S "pymmcore|MMCore|micromanager|micro.manager" --type py

# Map the manager directory structure
find imswitch/imcontrol/model/managers -name "*.py" | head -60

# Read the base classes — you MUST match these signatures exactly
cat imswitch/imcontrol/model/managers/detectors/DetectorManager.py
cat imswitch/imcontrol/model/managers/positioners/PositionerManager.py
cat imswitch/imcontrol/model/managers/lasers/LaserManager.py

# Read one concrete example of each to understand the pattern
cat imswitch/imcontrol/model/managers/detectors/HIKCamManager.py 2>/dev/null || \
  ls imswitch/imcontrol/model/managers/detectors/
cat imswitch/imcontrol/model/managers/positioners/ESP32StageManager.py 2>/dev/null || \
  ls imswitch/imcontrol/model/managers/positioners/
cat imswitch/imcontrol/model/managers/lasers/ESP32LEDLaserManager.py 2>/dev/null || \
  ls imswitch/imcontrol/model/managers/lasers/

# Check how managers are registered / discovered
cat imswitch/imcontrol/model/managers/detectors/__init__.py
cat imswitch/imcontrol/model/managers/positioners/__init__.py
cat imswitch/imcontrol/model/managers/lasers/__init__.py

# Check the setup JSON schema / info classes
rg -n "class DetectorInfo" --type py
rg -n "class LaserInfo" --type py
rg -n "class PositionerInfo" --type py

# Check how lowLevelManagers work
rg -n "lowLevelManagers" imswitch/imcontrol/model/managers/ --type py | head -20

# Look at an existing setup JSON for the structure
find . -name "*.json" -path "*/imcontrol_setups/*" | head -10
cat $(find . -name "example_virtual_microscope.json" -path "*/imcontrol_setups/*" | head -1) 2>/dev/null
```

**Do not proceed until you have read and understood the base class
signatures.** 

---

## STEP 1 — Shared MMCore singleton

Create: `imswitch/imcontrol/model/managers/MMCoreManager.py`

This is NOT an ImSwitch device manager. It is an internal helper module that
the three device managers below will share.

### Requirements

- Provide `get_core() -> CMMCorePlus` that returns a process-wide singleton
  via `CMMCorePlus.instance()`.
- Provide `ensure_loaded(cfg_path: str, adapter_paths: list[str] | None) -> CMMCorePlus`:
  - On first call: sets adapter search paths, calls `loadSystemConfiguration(cfg_path)`.
  - On subsequent calls with the same `cfg_path`: returns the core immediately (no-op).
  - On call with a different `cfg_path`: calls `unloadAllDevices()` first, then reloads.
  - Thread-safe via `threading.Lock`.
  - Falls back to env var `MICROMANAGER_PATH` (default `/opt/micro-manager/lib/micro-manager`)
    if `adapter_paths` is None.
  - Logs all loaded devices at INFO level after loading.
- Provide `reload(cfg_path, adapter_paths)` that forces a reload.
- Provide `get_available_adapters(adapter_path: str) -> list[str]` that
  lists the `.so`/`.dylib` files in the adapter directory and returns
  human-readable adapter names (strip `libmmgr_dal_` prefix and `.so` suffix).
- Provide `get_available_devices_for_adapter(adapter_name: str) -> list[str]`
  that calls `core.getAvailableDevices(adapter_name)` and returns the list.

### Key design decisions

- `pymmcore-plus` is an OPTIONAL dependency. Guard the import:
  ```python
  try:
      from pymmcore_plus import CMMCorePlus
  except ImportError:
      CMMCorePlus = None
  ```
  All three managers must check `if CMMCorePlus is None` and raise a clear
  error message telling the user to `pip install pymmcore-plus`.
- The singleton is important because multiple ImSwitch managers (camera +
  stage + laser) will share the SAME core with the SAME loaded .cfg, and
  MMCore is stateful — you must not create two cores fighting over USB.

---

## STEP 2 — MMCoreDetectorManager (camera)

Create: `imswitch/imcontrol/model/managers/detectors/MMCoreDetectorManager.py`

### Constructor: `__init__(self, detectorInfo, name, **lowLevelManagers)`

Read these from `detectorInfo.managerProperties`:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `cfgPath` | str | YES | Path to MM `.cfg` file |
| `adapterPath` | str | no | Override adapter search path |
| `adapterName` | str | no | e.g. `"DemoCamera"`, `"HamamatsuHam"`, `"AndorSDK3"` |
| `deviceName` | str | no | e.g. `"DCam"`, `"Andor sCMOS Camera"` |
| `deviceLabel` | str | no | Label to assign (default: `"Camera"`) |

**If `cfgPath` is provided**, load via `MMCoreManager.ensure_loaded(cfgPath)`.
The camera device is already in the config.

**If `cfgPath` is NOT provided but `adapterName` + `deviceName` ARE**, do
manual device loading:
```python
core.loadDevice(label, adapterName, deviceName)
core.initializeDevice(label)
core.setCameraDevice(label)
```
This is the mode where the user specifies "I want an Andor camera" directly
in the JSON without writing a separate .cfg file.

**If NEITHER is provided**, raise a clear error.

After loading, read from the core:
- `fullShape = (core.getImageWidth(), core.getImageHeight())`
- Exposure via `core.getExposure()`
- Pixel type via `core.getBytesPerPixel()`
- Binning via `core.getProperty(label, "Binning")` if the property exists

Build the `parameters` dict dynamically by iterating
`core.getDevicePropertyNames(label)` — for each property:
- If it has allowed values (`core.getAllowedPropertyValues`), create a
  `DetectorListParameter`.
- If it is numeric, create a `DetectorNumberParameter`.
- Skip read-only internal properties (property names starting with `"On"`
  or containing `"TransposeCorrection"`).
- Group all as `"MMCore"`.
- Always include `"Exposure"` as an explicit `DetectorNumberParameter`
  (group `"Acquisition"`, units `"ms"`, editable).

Call `super().__init__()` with the signature matching what you found in Step 0.

### Abstract methods to implement

- `getLatestFrame()` → `return self._core.getLastImage()`
  If no image is available yet, call `self._core.snap()` first.
- `getChunk()` → collect all buffered images via `popNextImage()` in a loop,
  stack into a 3D ndarray `(N, H, W)`.
- `flushBuffers()` → `self._core.clearCircularBuffer()`
- `startAcquisition()` → `self._core.startContinuousSequenceAcquisition(0)`
- `stopAcquisition()` → check `isSequenceRunning()` first, then
  `stopSequenceAcquisition()`.
- `crop(hpos, vpos, hsize, vsize)` → `self._core.setROI(label, hpos, vpos, hsize, vsize)`
- `pixelSizeUm` property → read from `core.getPixelSizeUm()`, return `[1, ps, ps]`.
- `setParameter(name, value)` → if `name == "Exposure"`, call
  `core.setExposure(float(value))`. For all other parameters, call
  `core.setProperty(label, name, value)`. Return updated parameters dict.
- `setBinning(binning)` → `core.setProperty(label, "Binning", str(binning))`
- `finalize()` → `stopAcquisition()`

---

## STEP 3 — MMCorePositionerManager (stage)

Create: `imswitch/imcontrol/model/managers/positioners/MMCorePositionerManager.py`

### Constructor: `__init__(self, positionerInfo, name, **lowLevelManagers)`

Read from `positionerInfo.managerProperties`:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `cfgPath` | str | yes* | Path to MM `.cfg` file |
| `adapterPath` | str | no | Override adapter search path |
| `xyAdapterName` | str | no* | e.g. `"DemoCamera"` for manual loading |
| `xyDeviceName` | str | no* | e.g. `"DXYStage"` |
| `xyDeviceLabel` | str | no | Label (default: `"XY"`) |
| `zAdapterName` | str | no* | e.g. `"DemoCamera"` |
| `zDeviceName` | str | no* | e.g. `"DStage"` |
| `zDeviceLabel` | str | no | Label (default: `"Z"`) |

*Either `cfgPath` OR the adapter/device name pairs are required.

Support two modes:
1. **Config mode**: `cfgPath` provided → `MMCoreManager.ensure_loaded(cfgPath)`
   and locate the XY/Z devices by label.
2. **Manual mode**: adapter + device names provided → `loadDevice` + `initializeDevice`
   for XY and/or Z independently.

The axes come from `positionerInfo.axes` (list of `"X"`, `"Y"`, `"Z"`).

### Methods to implement

Follow the pattern you found in Step 0 for the existing positioner managers.
The key methods are:

- `move(dist, axis)` — RELATIVE move in micrometers.
  - X/Y: `core.setRelativeXYPosition(dx, dy)` then `core.waitForDevice(xyLabel)`
  - Z: `core.setRelativePosition(dist)` then `core.waitForDevice(zLabel)`
- `setPosition(position, axis)` — ABSOLUTE move in micrometers.
- `getPosition(axis)` → returns current position for the given axis.
- `finalize()` — no-op (stages don't need cleanup).

Update `self._position[axis]` after every move.

---

## STEP 4 — MMCoreLaserManager (laser / shutter)

Create: `imswitch/imcontrol/model/managers/lasers/MMCoreLaserManager.py`

### Constructor: `__init__(self, laserInfo, name, **lowLevelManagers)`

Read from `laserInfo.managerProperties`:

| Property | Type | Required | Description |
|----------|------|----------|-------------|
| `cfgPath` | str | yes* | Path to MM `.cfg` file |
| `adapterPath` | str | no | Override adapter search path |
| `mode` | str | no | `"shutter"` (default) or `"property"` |
| `adapterName` | str | no* | For manual loading |
| `deviceName` | str | no* | For manual loading |
| `deviceLabel` | str | YES | Label of the shutter or DA device |
| `propertyName` | str | no | For property mode (default: `"Volts"`) |

*Either `cfgPath` OR `adapterName` + `deviceName` required.

### Two modes

1. **Shutter mode** (`mode: "shutter"`):
   - `setEnabled(True/False)` → `core.setShutterDevice(label)` + `core.setShutterOpen(enabled)`
   - `setValue(value)` → no-op (it's binary)
   - `isBinary = True`, `valueUnits = ""`

2. **Property mode** (`mode: "property"`):
   - `setEnabled(False)` → `core.setProperty(label, propertyName, 0)`
   - `setValue(value)` → `core.setProperty(label, propertyName, value)`
   - `isBinary = False`, `valueUnits = laserInfo.managerProperties.get("valueUnits", "mW")`

Call `super().__init__(laserInfo, name, isBinary=..., valueUnits=..., valueDecimals=2)`
matching the base class signature you found in Step 0.

- `finalize()` → `setEnabled(False)`

---

## STEP 5 — Register the managers

Edit the `__init__.py` files to add imports. Guard with try/except so that
ImSwitch doesn't crash when pymmcore-plus is not installed:

```python
# In imswitch/imcontrol/model/managers/detectors/__init__.py
try:
    from .MMCoreDetectorManager import MMCoreDetectorManager  # noqa: F401
except ImportError:
    pass

# Same pattern for positioners/ and lasers/
```

If the openUC2 fork uses a different registration mechanism (e.g. a registry
dict or an explicit list), follow THAT pattern instead. Trust the source.

---

## STEP 6 — Default mock/demo setup JSON

Create: `imswitch/imcontrol/model/SetupInfo/imcontrol_setups/example_mmcore_demo.json`

(Adjust path if the openUC2 fork stores setups elsewhere — check Step 0 output.)

```json
{
  "rs232devices": {},
  "lasers": {
    "MMShutter": {
      "analogChannel": null,
      "digitalLine": null,
      "managerName": "MMCoreLaserManager",
      "managerProperties": {
        "adapterName": "DemoCamera",
        "deviceName": "DShutter",
        "deviceLabel": "Shutter",
        "mode": "shutter"
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
        "xyAdapterName": "DemoCamera",
        "xyDeviceName": "DXYStage",
        "xyDeviceLabel": "XY",
        "zAdapterName": "DemoCamera",
        "zDeviceName": "DStage",
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
        "adapterName": "DemoCamera",
        "deviceName": "DCam",
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

**Important**: this demo setup uses `adapterName`/`deviceName` (manual mode)
instead of `cfgPath`, so it works out of the box without a separate `.cfg`
file. The DemoCamera adapter ships with every Micro-Manager install.

Also create a second example for a real camera — this one uses `cfgPath`:

Create: `imswitch/imcontrol/model/SetupInfo/imcontrol_setups/example_mmcore_andor.json`

```json
{
  "rs232devices": {},
  "lasers": {},
  "positioners": {
    "ASIStage": {
      "managerName": "MMCorePositionerManager",
      "managerProperties": {
        "cfgPath": "/home/pi/mm_configs/andor_asi.cfg",
        "xyDeviceLabel": "XYStage",
        "zDeviceLabel": "ZDrive"
      },
      "axes": ["X", "Y", "Z"],
      "forScanning": true,
      "forPositioning": true
    }
  },
  "detectors": {
    "AndorCamera": {
      "managerName": "MMCoreDetectorManager",
      "managerProperties": {
        "cfgPath": "/home/pi/mm_configs/andor_asi.cfg",
        "deviceLabel": "Andor sCMOS Camera"
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

---

## STEP 7 — Tests

Create: `tests/test_mmcore_managers.py`

```python
"""
Tests for MMCore*Manager classes using the DemoCamera adapter.

These tests require pymmcore-plus and the DemoCamera adapter library.
On x86_64, `mmcore install` (from pymmcore-plus[cli]) downloads adapters.
On arm64, they must be built from source (see install_micromanager_rpi.sh).

Skip gracefully if not available.
"""
import pytest
import os
import numpy as np

# Skip entire module if pymmcore-plus is not installed
pymmcore_plus = pytest.importorskip("pymmcore_plus")

# Skip if no adapter library is available
def _adapters_available():
    paths = [
        os.environ.get("MICROMANAGER_PATH", ""),
        "/opt/micro-manager/lib/micro-manager",
        os.path.expanduser("~/mm-venv/lib/python3.*/site-packages/pymmcore_plus/"),
    ]
    for p in paths:
        import glob
        for expanded in glob.glob(p):
            if os.path.isdir(expanded):
                files = os.listdir(expanded)
                if any(f.startswith("libmmgr_dal_") or f.startswith("mmgr_dal_") for f in files):
                    return True
    return False

pytestmark = pytest.mark.skipif(
    not _adapters_available(),
    reason="No Micro-Manager device adapters found"
)


class TestMMCoreManager:
    def test_get_core(self):
        from imswitch.imcontrol.model.managers.MMCoreManager import get_core
        core = get_core()
        assert core is not None
        assert core.getVersionInfo() != ""

    def test_get_available_adapters(self):
        from imswitch.imcontrol.model.managers.MMCoreManager import get_available_adapters
        mm_path = os.environ.get("MICROMANAGER_PATH", "/opt/micro-manager/lib/micro-manager")
        if os.path.isdir(mm_path):
            adapters = get_available_adapters(mm_path)
            assert "DemoCamera" in adapters


class TestMMCoreDetectorManager:
    """Test the detector manager with DemoCamera."""

    def _make_manager(self):
        """Helper to create a detector manager with DemoCamera."""
        from imswitch.imcontrol.model.managers.detectors.MMCoreDetectorManager import MMCoreDetectorManager
        from unittest.mock import MagicMock

        info = MagicMock()
        info.managerProperties = {
            "adapterName": "DemoCamera",
            "deviceName": "DCam",
            "deviceLabel": "Camera",
        }
        info.forAcquisition = True
        info.forFocusLock = False

        return MMCoreDetectorManager(info, "TestCamera")

    def test_snap(self):
        mgr = self._make_manager()
        frame = mgr.getLatestFrame()
        assert isinstance(frame, np.ndarray)
        assert frame.ndim == 2
        assert frame.shape[0] > 0 and frame.shape[1] > 0
        mgr.finalize()

    def test_continuous_acquisition(self):
        import time
        mgr = self._make_manager()
        mgr.startAcquisition()
        time.sleep(0.3)
        mgr.stopAcquisition()
        mgr.finalize()

    def test_exposure(self):
        mgr = self._make_manager()
        mgr.setParameter("Exposure", 50.0)
        # Verify it was set on the core
        assert abs(mgr._core.getExposure() - 50.0) < 0.1
        mgr.finalize()


class TestMMCorePositionerManager:
    """Test the positioner manager with DemoCamera XY/Z stages."""

    def _make_manager(self):
        from imswitch.imcontrol.model.managers.positioners.MMCorePositionerManager import MMCorePositionerManager
        from unittest.mock import MagicMock

        info = MagicMock()
        info.managerProperties = {
            "xyAdapterName": "DemoCamera",
            "xyDeviceName": "DXYStage",
            "xyDeviceLabel": "XY",
            "zAdapterName": "DemoCamera",
            "zDeviceName": "DStage",
            "zDeviceLabel": "Z",
        }
        info.axes = ["X", "Y", "Z"]
        info.forScanning = True
        info.forPositioning = True

        return MMCorePositionerManager(info, "TestStage")

    def test_move_xy(self):
        mgr = self._make_manager()
        x0 = mgr.getPosition("X")
        mgr.move(100.0, "X")
        x1 = mgr.getPosition("X")
        assert abs(x1 - x0 - 100.0) < 1.0
        mgr.finalize()

    def test_move_z(self):
        mgr = self._make_manager()
        z0 = mgr.getPosition("Z")
        mgr.move(10.0, "Z")
        z1 = mgr.getPosition("Z")
        assert abs(z1 - z0 - 10.0) < 1.0
        mgr.finalize()

    def test_set_position(self):
        mgr = self._make_manager()
        mgr.setPosition(500.0, "X")
        assert abs(mgr.getPosition("X") - 500.0) < 1.0
        mgr.finalize()


class TestMMCoreLaserManager:
    """Test the laser manager with DemoCamera shutter."""

    def _make_manager(self):
        from imswitch.imcontrol.model.managers.lasers.MMCoreLaserManager import MMCoreLaserManager
        from unittest.mock import MagicMock

        info = MagicMock()
        info.managerProperties = {
            "adapterName": "DemoCamera",
            "deviceName": "DShutter",
            "deviceLabel": "Shutter",
            "mode": "shutter",
        }
        info.wavelength = 488
        info.valueRangeMin = 0
        info.valueRangeMax = 1

        return MMCoreLaserManager(info, "TestLaser")

    def test_shutter_toggle(self):
        mgr = self._make_manager()
        mgr.setEnabled(True)
        assert mgr._core.getShutterOpen() == True
        mgr.setEnabled(False)
        assert mgr._core.getShutterOpen() == False
        mgr.finalize()
```

Run tests with: `pytest tests/test_mmcore_managers.py -v`

Adjust the test helpers if the actual base class constructors require
additional arguments you discovered in Step 0.

---

## STEP 8 — pyproject.toml / setup changes

Add `pymmcore-plus` as an optional dependency:

```toml
[project.optional-dependencies]
pymmcore = ["pymmcore-plus>=0.10"]
```

(Adjust the extras group name and location to match the existing project
structure — it may use `setup.py`, `setup.cfg`, or `pyproject.toml`.)

---

## STEP 9 — GitHub Actions: build Micro-Manager arm64 adapters

Create: `.github/workflows/build-mm-arm64.yml`

This workflow compiles mmCoreAndDevices for arm64 on every push to this
branch (or on tag), and uploads the compiled adapter `.so` files as a
release artifact. Users can then download and extract instead of compiling.

```yaml
name: Build Micro-Manager adapters (arm64)

on:
  push:
    branches: [feat/pymmcore-integration]
    tags: ["mm-v*"]
  workflow_dispatch:

jobs:
  build-arm64:
    runs-on: ubuntu-24.04
    timeout-minutes: 120

    steps:
    - name: Checkout
      uses: actions/checkout@v4

    - name: Set up QEMU
      uses: docker/setup-qemu-action@v3
      with:
        platforms: arm64

    - name: Build adapters in arm64 container
      uses: uraimo/run-on-arch-action@v3
      id: build
      with:
        arch: aarch64
        distro: bookworm
        githubToken: ${{ github.token }}

        install: |
          apt-get update -y
          apt-get install -y \
            build-essential autoconf automake libtool autoconf-archive \
            pkg-config git swig libboost-all-dev \
            python3-dev python3-numpy \
            libusb-1.0-0-dev libudev-dev

        run: |
          set -euxo pipefail

          # Clone with submodules
          git clone --depth 1 https://github.com/micro-manager/micro-manager.git /build/mm
          cd /build/mm
          git submodule update --init --recursive

          # Build MMCore + device adapters only (no Java)
          ./autogen.sh
          ./configure \
            --prefix=/opt/micro-manager \
            --without-java \
            --disable-java-app
          make -j$(nproc)
          make install DESTDIR=/build/staging

          # Record device interface version for pinning
          grep "MODULE_INTERFACE_VERSION" \
            mmCoreAndDevices/MMDevice/MMDeviceConstants.h \
            > /build/staging/DEVICE_INTERFACE_VERSION.txt

          # Package
          cd /build/staging
          tar czf /artifacts/micro-manager-arm64.tar.gz \
            opt/micro-manager/ \
            DEVICE_INTERFACE_VERSION.txt

          # List what was built
          find opt/micro-manager/lib/micro-manager -name "*.so" | \
            sed 's|.*/libmmgr_dal_||; s|\.so.*||' | sort \
            > /artifacts/ADAPTERS_BUILT.txt
          cat /artifacts/ADAPTERS_BUILT.txt

        dockerRunArgs: |
          --volume ${{ github.workspace }}/artifacts:/artifacts

    - name: Create install helper
      run: |
        mkdir -p artifacts
        cat > artifacts/install_mm_arm64.sh << 'EOF'
        #!/usr/bin/env bash
        set -euo pipefail
        SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
        echo "=== Installing Micro-Manager arm64 adapters ==="
        sudo tar xzf "$SCRIPT_DIR/micro-manager-arm64.tar.gz" -C /
        echo 'export MICROMANAGER_PATH="/opt/micro-manager/lib/micro-manager"' >> ~/.bashrc
        echo "=== Installing pymmcore-plus ==="
        pip install "pymmcore-plus" --break-system-packages 2>/dev/null || \
          pip install "pymmcore-plus"
        echo "=== Done! Run: source ~/.bashrc ==="
        echo "=== Then test with: python3 -c 'from pymmcore_plus import CMMCorePlus; print(CMMCorePlus.instance().getVersionInfo())' ==="
        EOF
        chmod +x artifacts/install_mm_arm64.sh

    - name: Upload build artifacts
      uses: actions/upload-artifact@v4
      with:
        name: micro-manager-arm64
        path: artifacts/

    - name: Create GitHub Release
      if: startsWith(github.ref, 'refs/tags/mm-v')
      uses: softprops/action-gh-release@v2
      with:
        files: |
          artifacts/micro-manager-arm64.tar.gz
          artifacts/install_mm_arm64.sh
          artifacts/ADAPTERS_BUILT.txt

  # Also run the Python tests on x86_64 with DemoCamera
  test-x86:
    runs-on: ubuntu-24.04
    timeout-minutes: 15

    steps:
    - uses: actions/checkout@v4

    - name: Set up Python
      uses: actions/setup-python@v5
      with:
        python-version: "3.12"

    - name: Install dependencies
      run: |
        pip install -e ".[pymmcore,dev]" 2>/dev/null || pip install -e ".[dev]"
        pip install pymmcore-plus[cli] pytest

    - name: Install MM demo adapters
      run: mmcore install

    - name: Run pymmcore tests
      run: pytest tests/test_mmcore_managers.py -v
      env:
        MICROMANAGER_PATH: ${{ github.workspace }}/.pymmcore_plus
```

---

## STEP 10 — Documentation

Create: `docs/pymmcore-integration.md`

Write a short guide (300-500 words) covering:

1. **What this does**: lets ImSwitch control any Micro-Manager-supported
   camera/stage/laser via pymmcore-plus, without Java, without a server.

2. **Quick start with DemoCamera** (copy the demo JSON, run `mmcore install`,
   start ImSwitch).

3. **Using a real camera** (two modes: write a .cfg file OR specify
   adapterName/deviceName in JSON directly).

4. **Finding your adapter name**: explain how to use
   `MMCoreManager.get_available_adapters()` and
   `MMCoreManager.get_available_devices_for_adapter("HamamatsuHam")` to
   discover what's available.

5. **Raspberry Pi setup**: point to `install_micromanager_rpi.sh` or the
   prebuilt GitHub Release tarball.

6. **Known limitations**: no Java, no MMStudio, adapters requiring
   Windows-only vendor DLLs won't work, Pi 5 8GB recommended.

---

## Constraints and quality checks

Before you consider any step done:

1. **Match the base class exactly.** If `DetectorManager.__init__` in the
   openUC2 fork has different kwargs than documented here (which is likely —
   the fork has diverged from upstream), match what you find in the source.
   The source code is the ground truth.

2. **Guard all pymmcore imports.** ImSwitch must still launch and work for
   users who don't have pymmcore-plus installed. The error should only
   appear if they try to use an `MMCore*Manager` in their setup JSON.

3. **No global state on import.** `MMCoreManager.get_core()` must be lazy.
   Importing the module must not instantiate the core or try to load adapters.

4. **Thread safety.** ImSwitch inits managers from potentially different
   threads. The Lock in MMCoreManager is critical.

5. **Don't duplicate functionality.** If the openUC2 fork already has some
   form of generic camera manager that could be extended, extend it. Check
   Step 0 output carefully.

6. **Test with DemoCamera.** Every code path you write must be testable with
   the DemoCamera adapter, which is a mock that ships with every MM install.

7. **Commit messages.** Use conventional commits:
   `feat(mmcore): add MMCoreDetectorManager with dynamic property loading`

---

## Definition of Done

- [ ] `MMCoreManager.py` exists and provides singleton + lazy loading + adapter listing
- [ ] `MMCoreDetectorManager.py` snaps frames, streams, crops, exposes all device properties dynamically
- [ ] `MMCorePositionerManager.py` moves XY and Z, reads positions back
- [ ] `MMCoreLaserManager.py` toggles shutters and sets analog values
- [ ] `example_mmcore_demo.json` works out of the box with DemoCamera (no .cfg needed)
- [ ] `example_mmcore_andor.json` documents the .cfg-based mode for real hardware
- [ ] All managers import-guarded; ImSwitch works normally without pymmcore-plus
- [ ] `pytest tests/test_mmcore_managers.py` passes on x86_64 with DemoCamera
- [ ] `.github/workflows/build-mm-arm64.yml` builds arm64 adapters and publishes release
- [ ] `docs/pymmcore-integration.md` exists
- [ ] All changes on branch `feat/pymmcore-integration`, no existing files broken