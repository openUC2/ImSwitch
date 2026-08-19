# Camera Settings Dialog & Binning API

The live view exposes exposure, mode, gain and binning directly; everything else
a camera offers (black level, trigger source, cooling, temperature, pixel
format, …) lives behind the **Camera settings** button, which renders whatever
the detector manager declares — no per-camera frontend code.

## Backend API (`SettingsController`)

| Endpoint | Type | Description |
| --- | --- | --- |
| `getDetectorParameterTree` | GET | Full tree for one detector: everything `getCameraStatus()` reports plus the parameter dict grouped by the manager's own groups. Each entry has `name, value, type, editable` and, where applicable, `units`, `options`, `min`, `max`. |
| `setDetectorParameterValue` | POST | `{detectorName, name, value}` — casts the JSON value to the parameter's declared type, applies it, returns the refreshed tree. Failures are reported as `error` inside the tree, never as a 500. |
| `getDetectorSupportedBinnings` | GET | Binning factors the detector accepts. |
| `setDetectorBinning` | GET | `?binning=N[&detectorName=]` — applies binning, broadcasts the new frame shape, returns `{status, binning, shape}`. |
| `getDetectorParameters` | GET | Compact payload for the live-view row; also carries `supportedBinnings`, `exposureMin/Max`, `gainMin/Max`. |

## Manager-side hooks

Two hooks in `DetectorManager` make a camera participate:

- **`refreshParameters()`** — re-read what the hardware can report before the
  tree is serialized (that is what keeps a sensor temperature live). The base
  class returns the cached values; managers whose camera object implements
  `getPropertyValue()` can simply delegate:

  ```python
  def refreshParameters(self):
      return self._refreshParametersFromCamera(_HARDWARE_READABLE_PARAMS)
  ```

  `_HARDWARE_READABLE_PARAMS` is a per-manager whitelist — asking a wrapper for
  a property it does not know only produces log noise. A parameter keeps its
  cached value when the read raises or returns `None`/`False`, so the helper
  must not be used for boolean parameters.

- **`DetectorNumberParameter(valueMin=…, valueMax=…)`** — surfaces as `min`/`max`
  in the tree and bounds the input in the UI. Populate it from the SDK's range
  query (`get_exposuretime()` / `get_gain()` return `(current, min, max)` by
  convention).

A manager that supports binning overrides `setBinning()`, applies it inside
`_performSafeCameraAction`, and follows the resulting frame size with
`self._shape = …` and `self._setFullShape(…)` — binning shrinks the largest
frame the sensor can deliver.

## Binning support per camera

| Manager | Default `supportedBinnings` | Mechanism |
| --- | --- | --- |
| `ToupCamManager` | `[1, 2, 3, 4]` | Averaged digital binning (`TOUPCAM_OPTION_BINNING`, `0x80\|n`); frame size read back via `get_Size()` |
| `HikCamManager` | `[1, 2, 4]` | `BinningX`/`BinningY` nodes; `WidthMax`/`HeightMax` re-read after the change |
| `TucsenCamManager` | `[1, 2]` | `TUIDC_RESOLUTION`: RESOLUTION (full) vs SENSITIVE (2×2 combined) — nothing above 2 exists |
| `GXPIPYManager` (Daheng) | `[1]` | `BinningHorizontal`/`BinningVertical` are **not honoured by every model**, so extra factors must be opted into per setup |
| `OpenCVCamManager` | `[1]` | The OpenCV/V4L backend has no binning control |

All of them accept a `supportedBinnings` list in `managerProperties`, plus a
`binning` startup value:

```json
"managerProperties": {
  "cameraListIndex": 0,
  "binning": 1,
  "supportedBinnings": [1, 2, 4]
}
```

Setting `supportedBinnings` to a single value hides the binning control (the UI
disables a selector with fewer than two options).

## Units

`getParameter("exposure")` returns **milliseconds** for every camera. The
wrappers convert from whatever their SDK uses (Hik/Toupcam/Daheng report µs
internally). Anything consuming exposure should divide by `1e3` to get seconds.
