# pymmcore-plus integration

ImSwitch can drive any camera, stage, or laser that has a Micro-Manager
device adapter — Andor, Hamamatsu, Basler, ASI, Prior, Thorlabs, Coherent
and ~250 others — through the [`pymmcore-plus`][pymmcore-plus] Python
bindings to the Micro-Manager `MMCore` C++ library.

There is **no Java**, **no MMStudio**, and **no separate server process**:
`pymmcore-plus` loads the same C++ adapters that MMStudio uses directly
into the ImSwitch Python process.

## What you get

Three new device managers, picked up by the standard `managerName`
mechanism in your setup JSON:

| Manager class               | Replaces                | Wraps                              |
|----------------------------|--------------------------|------------------------------------|
| `MMCoreDetectorManager`     | Camera-specific manager  | `core.snap()`, sequence acquisition |
| `MMCorePositionerManager`   | Stage-specific manager   | XY + Z stage devices                |
| `MMCoreLaserManager`        | Laser-specific manager   | Shutter or DA property device       |

All three share a process-wide `CMMCorePlus` singleton (see
[`MMCoreManager.py`](../imswitch/imcontrol/model/managers/MMCoreManager.py))
so a single `.cfg` file drives every device without USB conflicts.

## Installation

```bash
pip install "ImSwitchUC2[pymmcore]"
# or, in a dev checkout:
pip install -e ".[pymmcore]"
```

To make device adapters available, install the Micro-Manager binaries:

```bash
# x86_64 / arm64 macOS / x86_64 Linux: download via pymmcore-plus CLI
pip install "pymmcore-plus[cli]"
mmcore install      # downloads adapters under ~/.local/share/pymmcore-plus
```

On a **Raspberry Pi (arm64)** there is no published binary build, so use
either:

* The provided helper script
  [`install_micromanager_raspi.sh`](../install_micromanager_raspi.sh), or
* The prebuilt tarball published by the
  [`build-mm-arm64`](../.github/workflows/build-mm-arm64.yml) GitHub
  Actions workflow on every tagged release.

Set `MICROMANAGER_PATH` to the directory containing
`libmmgr_dal_*.so` if it is not auto-discovered.

## Quick start: the DemoCamera setup

The [`example_mmcore_demo.json`](../imswitch/_data/user_defaults/imcontrol_setups/example_mmcore_demo.json)
setup uses the `DemoCamera` adapter that ships with every Micro-Manager
install — no `.cfg` file required.

```bash
imswitch --setup example_mmcore_demo.json
```

You should see a `MMCamera` detector, a `MMStage` XYZ positioner, and
the `MMShutter` laser show up in the UI immediately.

## Using a real camera

Two configuration modes are supported.

### Mode A — write a Micro-Manager `.cfg` file

Configure your hardware once with `MMConfig.exe` (or by hand) and point
ImSwitch at the resulting file. Multiple managers can share the same
`.cfg`; it is loaded only once per process:

```json
{
  "detectors": {
    "AndorCamera": {
      "managerName": "MMCoreDetectorManager",
      "managerProperties": {
        "cfgPath": "/home/pi/configs/Andor_ASI.cfg",
        "deviceLabel": "Andor sCMOS Camera"
      },
      "forAcquisition": true
    }
  }
}
```

A complete example lives in
[`example_mmcore_andor.json`](../imswitch/_data/user_defaults/imcontrol_setups/example_mmcore_andor.json).

### Mode B — declare the device inline

Skip `.cfg` files entirely by listing the adapter and device name in the
manager properties — handy for quick demos:

```json
{
  "detectors": {
    "Cam": {
      "managerName": "MMCoreDetectorManager",
      "managerProperties": {
        "adapterName": "HamamatsuHam",
        "deviceName":  "HamamatsuHam_DCAM",
        "deviceLabel": "Hamamatsu"
      },
      "forAcquisition": true
    }
  }
}
```

## Discovering adapters and devices

Use the helpers on `MMCoreManager` to introspect what is installed:

```python
from imswitch.imcontrol.model.managers import MMCoreManager

# All adapters present on disk
print(MMCoreManager.get_available_adapters("/opt/micro-manager/lib/micro-manager"))

# Devices a specific adapter exposes (requires the singleton to be alive)
core = MMCoreManager.get_core()
print(MMCoreManager.get_available_devices_for_adapter("HamamatsuHam"))
```

## Raspberry Pi notes

* Use a Pi 5 with **8 GB RAM** for anything beyond the demo adapter.
* Building `mmCoreAndDevices` from source on the Pi takes ~30 minutes;
  prefer the prebuilt tarball from CI.
* Adapters that depend on Windows-only vendor DLLs (e.g. some
  AndorSDK3 builds) cannot be used on Linux — check the vendor's docs.

## Known limitations

* No Java, no MMStudio integration.
* `MMCorePositionerManager.moveForever` is a no-op — Micro-Manager has no
  generic jog primitive.
* Laser power calibration is the user's responsibility: `mode: "property"`
  writes a raw value (typically volts).
* Some adapter properties expose values that don't fit our number/list
  parameter widgets; those properties are silently skipped in the UI but
  remain settable via `setProperty` calls.

[pymmcore-plus]: https://pymmcore-plus.github.io/pymmcore-plus/
