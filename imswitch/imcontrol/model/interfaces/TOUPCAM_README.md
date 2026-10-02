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
         "readSaveTemperature": true,
         "toupcam": {
           "exposure": 100,
           "gain": 100,
           "blacklevel": 0,
           "frame_rate": -1,
           "target_temperature": -30
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
  `target_temperature` and `fan_speed` parameters. The target is re-applied
  after a reconnect, so the sensor stays at the requested temperature.
- **Temperature monitor**: a background thread polls the sensor temperature
  every 5 s for as long as the camera is open (the cooler runs then too, so the
  check cannot wait for an acquisition).
  - **TEC keep-alive**: the target is re-written to the camera every 60 s.
    One write is not reliable on this family - the camera reports the target
    back correctly, with the cooler on, and can still sit at ambient for hours
    (seen in a real log at a -40 C target). Re-asserting it costs one register
    write a minute and gets the regulation going again by itself.
  - **Target range**: `set_temperature()` clamps to the model's own range
    (`TOUPCAM_OPTION_TECTARGET_RANGE`, logged at startup) and warns when it
    has to. An out-of-range target is accepted by the SDK and read back
    unchanged, but the cooler may then not regulate at all - which is exactly
    what a sensor stuck at ambient with `tec_on=1` looks like.
  - **Stall warning**: if the sensor stays more than 5 C above the target for
    5 minutes with the cooler on, one warning says so (and `tec_stalled`
    appears in the camera status), instead of the problem only showing up as
    noisy images days later.
  - **Over-temperature cutoff**: two consecutive readings at or above
    `TEMPERATURE_SHUTDOWN_C` (30 C by default) switch the cooler *and* the
    window heater off - a sensor that hot means the
    heat is not getting out (stalled fan, blocked vents, failing TEC), and the
    cooler's own dissipation then makes it worse. The cooler is re-armed at the
    cached target once the camera falls back 5 C below that cutoff, at most
    3 times;
    after that it stays off until a target temperature is set again, which is
    also the manual override. `tec_over_temperature` in the camera status says
    whether the cutoff is currently tripped.
  - **Log** (`"readSaveTemperature": true`): while the camera is armed (live
    view or a snap in progress) the sensor temperature, TEC target and state,
    fan, heater, exposure and the cutoff flag are appended every 5 s to
    `recordings/<YYYY-MM-DD>/toupcam_temperature_log.csv` in the data folder -
    the same folder the day's snaps go to. Off by default.
- **Window heater** is **off** unless the setup asks for it (`"heat": 5`, or
  `true` for the model's maximum level). It fights the cooler, and fogging is
  only a risk at low target temperatures.
- **Reconnect**: on `TOUPCAM_EVENT_DISCONNECTED` (USB drop) a background
  thread reopens the camera, re-applies the cached settings and resumes
  streaming.
- The INDI / INDIGO astronomy stacks ship their own copies of this same
  vendor library; they are NOT needed here — the official SDK libraries are
  sufficient (and preferred, since they match the vendored wrapper version).
