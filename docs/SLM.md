# SLM / DMD display

A spatial light modulator or digital micromirror device (DMD) driven as a plain
**display** (HDMI) sitting in a plane conjugate to the sample. ImSwitch puts
patterns on that display, maps the camera image onto it, and runs pattern
sequences in lock-step with the camera.

Backend: `SLMManager` (`imswitch/imcontrol/model/managers/SLMManager.py`) owns the
output, the camera→display affine and the sequences; `SLMController` is the HTTP
API; the React app is "SLM / DMD" (`frontend/src/components/SLMController.jsx`).
The old Hamamatsu phase-mask code that lived under the same name is gone.

## Setup

```json
"availableWidgets": [..., "SLM"],
"slm": {
    "display": "framebuffer",
    "displayIndex": 0,
    "width": 1920,
    "height": 1080,
    "settleTimeMs": 50,
    "calibrationDetector": null
}
```

| Key | Meaning |
|---|---|
| `display` | `virtual` (nothing shown; tests, the virtual microscope), `framebuffer` (Linux `/dev/fb<displayIndex>`: the Raspberry Pi HDMI output, no X/Wayland needed), `window` (fullscreen window on monitor `displayIndex`, `pip install pygame-ce` / `imswitch[slm]`) |
| `displayIndex` | framebuffer number or monitor index (0 = primary, 1 = first external) |
| `width`, `height` | pattern size; once the output is open its real size wins |
| `settleTimeMs` | wait after a new frame before the camera is triggered: display latency + DMD refresh. Too short and the previous pattern is imaged. Tune per device |
| `calibrationDetector` | camera for the automatic calibration; `null` = the current one |

The `window` backend runs the window in a **subprocess**
(`slm_display.py --window`), so the server never hosts SDL and macOS' main-thread
rule does not apply. Every `show` blocks until the subprocess flipped the frame
(vsync) and then waits the settle time. Backends that fail to open fall back to a
virtual display of the configured size and `getSLMStatus().error` says why.

### Raspberry Pi (openUC2 OS)

The container reaches `/dev/fb0` through the `a 29:* rwm` device rule in the
`imswitch.pkg` deployment. For a dedicated DMD output enable the pallet feature
`slm-display` (`imswitch.deploy.yml`): its systemd unit adds
`consoleblank=0 vt.global_cursor_default=0 fbcon=map:1` to
`/boot/firmware/cmdline.txt` (no blanking, no cursor, the text console stays off
`/dev/fb0` so it never overwrites a pattern; serial console and SSH remain). A
DMD without usable EDID gets its mode from `/etc/imswitch-slm-display.conf`:
`SLM_VIDEO_MODE=HDMI-A-1:1920x1080@60D`. Reboot once after enabling.

Pi framebuffers are usually 16 bpp (RGB565): grey levels are quantised to 5/6
bits, binary patterns are unaffected. A Wayland/X compositor on the same GPU
(the DSI kiosk) and the framebuffer path cannot share the device.

## Coordinates

Display pixels: x right, y down, origin top-left. Camera pixels: full-sensor
pixels (the live stream may be subsampled; the app multiplies by the factor).
The affine maps **camera → display**:

```
x' = a11·x + a12·y + tx
y' = a21·x + a22·y + ty
```

It reuses `AffineTransform` from the galvo scanner and is saved in
`~/ImSwitchConfig/slm/slm_affine_<name>.json`.

## Calibration

* **Automatic** (`runSLMAutoCalibration(nPoints, spotRadius, detectorName)`): a
  dark frame with the display blank, then one spot per point; the spot is found
  by background subtraction, blur and a half-maximum threshold around the peak
  (`spot_centroid`, needs SNR > 5). Three found spots are enough; the result
  carries the RMS residual in display pixels. The display is blanked afterwards.
* **Manual**: `getSLMCalibrationPoints` → `showSLMPattern(kind=disk, cx, cy)` per
  point, click the spot on the camera, then `computeSLMAffine` with the pairs.
  The app's Calibrate tab does this.

## Patterns, images, drawing

* `showSLMPattern(kind, …)`: `blank`, `checkerboard`, `grating` (period, angle,
  duty, phase, sinusoidal), `disk`, `grid`.
* `showSLMImage` (POST, base64 file): `fit=center` keeps 1:1 pixels, `stretch`
  resizes with nearest neighbour.
* `showSLMCameraMask` (POST): a mask drawn over the camera image at any
  resolution plus the camera size it spans; it is warped with the affine
  (`cv2.warpAffine`, nearest) so the light lands where the mask is. The Draw tab
  rasterises brush, rectangle, ellipse and eraser strokes into that mask and can
  project on every change.
* `getSLMPreview` returns the frame currently on the display as PNG.

## Sequences and synchronised acquisition

`uploadSLMSequence` (POST, named list of base64 images) stores
`~/ImSwitchConfig/slm/sequences/<name>/NNNN_<frame>.png`; `listSLMSequences`,
`getSLMSequenceFrame`, `showSLMSequenceFrame`, `deleteSLMSequence`.
`playSLMSequence(name, intervalMs, loop)` shows it without the camera.

`startSLMSequenceAcquisition` (POST `{sequence, detectorName, settleMs, repeats,
save, fileName}`) does, per frame: show → settle → **one** camera frame. When the
camera supports it the frame is taken with a software trigger
(`setTriggerSource("software")` + `snapSync`, the ExperimentController's
deterministic grab: no buffered pre-pattern frame can slip through); otherwise
the buffer is flushed and the second new free-run frame is used. The stack is
saved as OME-TIFF under `<data>/SLM/` with one channel per pattern (and T =
repeats) through the shared `storage.save_image`. Progress: `getSLMAcquisitionStatus`
(`done/total`, `fps`, `triggered`, `path`) and `sigSLMSequenceProgress`.
Refused while an experiment, job, recording, stage map or focus calibration runs,
and those refuse while it runs (`busy.py`).

## Status and safety

`getSLMStatus` = display, affine, calibration, acquisition/calibration/play
states in one call. On a real instrument the display *is* illumination: treat
`showSLM*` and the sequence calls like switching a light source on.
