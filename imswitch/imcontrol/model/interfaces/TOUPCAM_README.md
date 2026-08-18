# ToupTek (Toupcam) camera integration

Support for ToupTek/Toupcam USB cameras (including astro/microscopy models
sold as ToupCam, RisingCam, Meade, Omegon etc. that use `libtoupcam`),
integrated the same way as the HIK camera: SDK pull-mode callback into a small
ring buffer, hardware ROI/crop, software/external trigger, and automatic
reconnect on USB drop.

## Components

| File | Purpose |
|---|---|
| `toupcamcamera.py` | `CameraToupcam` interface (mirrors `CameraHIK` API) |
| `toupcamsdk/toupcam.py` | official vendor ctypes wrapper (v60.31631.20260606, marked ImSwitch patch in `__initlib`) |
| `toupcamsdk/libloader.py` | finds the native lib per OS/arch (linux x64/arm64/armhf, macOS, Windows) |
| `toupcamsdk/install_toupcam_libs.py` | copies native libs from a downloaded vendor SDK into `toupcamsdk/lib/` |
| `toupcamsdk/99-toupcam.rules` | udev rules for USB access on Linux |
| `../managers/detectors/ToupCamManager.py` | detector manager (mock fallback: `MockCameraTIS`) |

## Native library resolution order

1. `IMSWITCH_TOUPCAM_LIB` — full path to `libtoupcam.so`/`.dylib`/`toupcam.dll`
2. `TOUPCAM_SDK_DIR` (or `IMSWITCH_TOUPCAM_SDK`) — unpacked vendor SDK folder
3. bundled libs in `toupcamsdk/lib/<platform>/` (populated by the install script)
4. system paths (`/usr/local/lib`, `/usr/lib`, multiarch dirs, homebrew)
5. wrapper default (next to `toupcam.py`, then the system loader search path)

On linux/arm64 the loader picks the `glibc` build (Raspberry Pi OS) and falls
back to `musl` (Alpine) automatically.

## Setup

1. Download the SDK from touptek.com (toupcamsdk.zip) and unpack it.
2. Copy the native libraries into the package (once per checkout):

   ```bash
   python imswitch/imcontrol/model/interfaces/toupcamsdk/install_toupcam_libs.py /path/to/toupcamsdk
   # or for a checkout that will be deployed to a Raspberry Pi as well:
   python .../install_toupcam_libs.py /path/to/toupcamsdk --platforms mac linux/x64 linux/arm64
   ```

   Alternatively skip the copy and set `TOUPCAM_SDK_DIR=/path/to/toupcamsdk`.

3. Linux only — install the udev rules once and replug the camera:

   ```bash
   sudo cp imswitch/imcontrol/model/interfaces/toupcamsdk/99-toupcam.rules /etc/udev/rules.d/
   ```

4. Setup configuration (see `example_toupcam.json`):

   ```json
   "detectors": {
     "WidefieldCamera": {
       "managerName": "ToupCamManager",
       "managerProperties": {
         "cameraListIndex": 0,
         "isRGB": false,
         "binning": 1,
         "supportedBinnings": [1, 2, 3, 4],
         "toupcam": {
           "exposure": 100,
           "gain": 100,
           "blacklevel": 0,
           "frame_rate": -1
         }
       },
       "forAcquisition": true
     }
   }
   ```

## Notes

- **Bit depth**: mono cameras run in RAW mode at the highest supported bit
  depth (16-bit container) by default; `pixel_format` = `mono8`/`mono16`
  switches it. RGB models (`isRGB: true`) deliver RGB24.
- **Gain** is in Toupcam native percent: 100 = 1x, 200 = 2x, etc. The manager
  reports the hardware min/max, which the UI uses to bound the input.
- **Binning** is averaged digital binning (`supportedBinnings`, default
  `[1, 2, 3, 4]`). It changes the delivered frame size; the manager reads the
  new size back from the SDK and the viewers are told to re-fit.
- **Trigger**: `Continous` (free run), `Internal trigger` (software trigger,
  used by `snapSync`/deterministic grabs), `External trigger` (hardware input).
- **TEC models** additionally expose `temperature` (read-only),
  `target_temperature` and `fan_speed` parameters.
- **Reconnect**: on `TOUPCAM_EVENT_DISCONNECTED` (USB drop) a background
  thread reopens the camera, re-applies the cached settings and resumes
  streaming.
- The INDI / INDIGO astronomy stacks ship their own copies of this same
  vendor library; they are NOT needed here — the official SDK libraries are
  sufficient (and preferred, since they match the vendored wrapper version).
