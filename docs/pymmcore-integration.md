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

> **Micro-Manager 2.0 is required** (Device API ≥ 70). The `pymmcore` /
> `pymmcore-plus` Python wheels are built against the MM 2.0 device
> interface; MM 1.4 adapters will fail with an *“interface version
> mismatch”* error on load.

```bash
pip install "ImSwitchUC2[pymmcore]"
# or, in a dev checkout:
pip install -e ".[pymmcore]"
```

To install the Micro-Manager 2.0 device adapters themselves, choose the
option that matches your platform:

| Platform | Recommended install                                                                 |
|----------|--------------------------------------------------------------------------------------|
| Windows  | Download MM 2.0 from <https://micro-manager.org/Micro-Manager_Nightly_Builds> – the installer drops adapters in `C:\Program Files\Micro-Manager-2.0` and ImSwitch picks them up automatically. |
| macOS    | Download the MM 2.0 nightly `.dmg`; drag to `/Applications/Micro-Manager-2.0`. |
| Linux x86_64 | `pip install "pymmcore-plus[cli]"` then `mmcore install` – downloads the official MM 2.0 adapters into pymmcore-plus' managed directory. |
| Raspberry Pi (arm64) | Build from source via [`install_micromanager_raspi.sh`](../install_micromanager_raspi.sh) or use the prebuilt tarball from the [`build-mm-arm64`](../.github/workflows/build-mm-arm64.yml) workflow. |

### Adapter path discovery

`MMCoreManager.discover_adapter_paths()` resolves adapter directories in
the following order:

1. `MICROMANAGER_PATH` environment variable (override).
2. `pymmcore_plus.find_micromanager()` – knows about `mmcore install`
   managed installs and any system installs it can find.
3. Platform-specific MM 2.0 install locations:
   * **Windows:** `C:\Program Files\Micro-Manager-2.0*`,
     `C:\Program Files (x86)\Micro-Manager-2.0*`
   * **macOS:** `/Applications/Micro-Manager-2.0*`,
     `/Applications/Micro-Manager.app/Contents/Resources`
   * **Linux:** `/opt/micro-manager/lib/micro-manager`,
     `/opt/Micro-Manager-2.0*`, `/usr/local/lib/micro-manager`
4. pymmcore-plus' managed install dir on every platform
   (`~/.local/share/pymmcore-plus/mm/Micro-Manager-*` and OS-specific
   equivalents).

On **Windows**, every resolved directory is also added to the Python
DLL search path via `os.add_dll_directory`, so vendor SDK DLLs co-located
with the adapter (e.g. Andor, Hamamatsu) are found automatically.

You can override the search at any time by setting `MICROMANAGER_PATH`,
or per-device via the `adapterPath` key in the setup JSON.

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
