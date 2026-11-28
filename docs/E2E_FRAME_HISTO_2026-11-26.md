# FRAME end-to-end test: histopathology slide scanning (2026-11-26)

Hardware-in-the-loop test on the FRAME with `FRAME-berry-muscle.json`: firmware update, FRAME
calibration with the pinhole slides (slots 1, 4), tile scans of the slides in slots 2 and 3 through
`ExperimentController`, benchmarks, and fixes. ImSwitch branch `feat/histo-e2e-frame`, UC2-REST
branch `feat/histo-e2e-frame`, firmware repo branch `fix/hardlimit-debounce` (all uncommitted).

## Setup

| Part | What |
|---|---|
| Host | MacBook M1 Pro, ImSwitch under Rosetta (`arch -x86_64 .venv_x86_311/bin/python -m imswitch …`): the HIK MVS SDK is x86_64 only |
| Camera | HIK MV-CS060-10UC-PRO (IMX178 colour, 3072×2048, BayerBG8) |
| Stage | CAN HAT (CP2102, `/dev/cu.usbserial-0001`, 921600 baud), motors 10 (A), 11 (X), 12 (Y), 13 (Z), LED 30, GPIO 60 |
| Objectives | 4x (slot 0), 20x Nikon (slot 1) on the slider (A) |
| Overview camera | UVC "HD Camera": not usable from this host (see open issues) |

## What ran on the instrument

1. **Firmware**: all 7 boards from a PR build (`v2026.0.0-beta.4-13-ge86ebf3…-pr133`, 2 Nov) to
   **v2026.0.0-beta.5** from the GitHub release, with ImSwitch's verified updater (CAN nodes, then the
   USB master; each re-scanned for the new version). 6.5 min in total.
2. **FRAME calibration** (target `slides`): first run failed at the 4x focus (root cause: turret,
   below). After the fix: 4x 0.5984 × 0.5941 µm/px (fit 1.23 px), focus Z 6960 µm, stage offset
   (−725, 1156) µm, repeatability 0.95 µm, verify 0.49 µm, backlash X 0.14 / Y 1.36 µm, insert
   rotation +0.09°, stage scale 0.99582 over 90 mm. Failed (non-critical): X/Y travel range
   (firmware debounce, below) and the 20x (its focus is beyond the Z end stop, below).
3. **Slide 2**: prescan of the slot → tissue mask → 395 tiles (of 574 in the bounding box) at 4x →
   one OME-Zarr canvas (75 775 × 39 015 px RGB, 4 levels) + RGB pyramidal OME-TIFF.
4. **Slide 3**: four serial sections → four scan areas, 2×2 focus map per area, 57 tiles → four
   canvases + OME-TIFFs.

Data: `~/ImSwitchConfig/data/ExperimentController/20261126_202356` (slide 2),
`…/20261126_203818` (slide 3, four sections), `…/20261126_204133` (re-run of section 1).

## Benchmarks

| What | Result |
|---|---|
| Tile scan, 4x, speed 20000 | 0.82 s/tile (XY move 0.43, Z settle 0.10, triggered grab 0.033 s); 395 tiles in 324 s |
| Tile scan, 4x, speed 30000 | 0.67–0.85 s/tile (XY move 0.30–0.37 s) |
| Canvas write (background) | 160 ms/frame, overlaps the acquisition |
| Finalize, 8.9 GB RGB canvas | 99 s (pyramid ≈ 45 s, RGB OME-TIFF ≈ 53 s); pyramid now 1.6× faster (below) |
| Autofocus, ±300 µm, 20 µm steps | 31 planes in 8.8 s (0.28 s/plane), r² 0.997 |
| Focus map, 16 points (4 regions) | 136 s (≈ 6.3 s per point) |
| Prescan, one slide (27 × 74 mm, 62 lines) | 632 s at "10000" (really 3.1 mm/s), 236 s at 30000 |
| Firmware update, 7 boards | 6.5 min (≈ 60 s per CAN node, 30 s master) |

`step_timings.json` (new) in every run folder has the per-step and per-category numbers.

## Bugs found and fixed

### Scientific correctness
- **Pixel size 1.0 µm/px in every snap and every experiment OME-Zarr** (calibrated 0.595).
  ExperimentController copied the detector pixel size at startup, before PixelCalibration (created
  last) set it; PixelCalibration never published it to the shared attributes / metadata hub; the
  hub's detector context was never updated after registration. Now live property, published on
  every calibration apply, hub context follows shared attributes (incl. field of view).
- **Stitched canvas placed tiles by grid index, side by side**: every overlap appeared twice at the
  seams and the canvas was not in stage coordinates. New `CanvasLayout`: tiles at their stage
  position ÷ pixel size, each keeps the half of every overlap nearer its centre (exact abutment),
  translation = stage position of the top-left pixel. Seam error on slide 2 ≈ 1–5 µm.
- **Colour drift tile to tile**: the HIK colour camera ran continuous auto white balance; the
  "AWB off" code was dead (`elif` after `isRGB=False`) and white balance was not exposed at all.
  Added `awb_mode` / `red_gain` / `blue_gain`; "once" now measures and locks; every scan locks
  white balance and auto exposure for its duration and restores them; the camera settings go into
  the run's protocol JSON.
- **Experiment autofocus scanned half the configured window** (`af_range` ±µm passed as the
  full-width `rangez`).
- **Focus-map searches chained**: each point started where the previous one left the stage (after a
  rejected point: the window edge), so a wrong plane (dust/ink on another glass face) walked the next
  windows off the tissue (slide 3: 7517 / 7298 µm alternating). Now centred on the median of what
  the region (else the run) found. Re-run: points 61 µm apart instead of 219 µm.

### Output format and robustness
- Stitched OME-TIFF of a colour camera was three separate channels → **interleaved RGB pyramid**
  (photometric RGB, SubIFDs, 512 px tiles) that QuPath opens as brightfield.
- OME-TIFF export read whole planes (GBs for a slide) → row bands; tiles compressed with 4 workers.
- Pyramid levels were each decimated from level 0 (three passes over the canvas) → from the previous
  level, planes in parallel: 1.6× faster, pixel-identical.
- **Large-scan policy dropped the stitched canvas above 2 GiB**, i.e. every tissue section at 4x →
  16 GiB default, `IMSWITCH_MAX_CANVAS_GIB` overrides (e.g. 2 on a Pi with an SD card).
- **Autofocus never started the camera**: with no live view open every triggered grab timed out
  ("0/11 valid focus values"); hill climbing read one stale frame. Now starts/stops acquisition.
- **Experiment left the camera streaming** (6 MP at 60 fps over USB) after every run → stopped if
  the run started it.
- `getStageMapOverview` returned a 383 px JPEG while `widthPx`/`umPerPixel` described the full
  canvas (coordinates off ≈ 30×) → encoded at the canvas size.
- Prescan lag calibrated only on the first line (holder end / blank glass: no contrast) → zig-zag
  strips → retried on the next lines (up to 6).
- `snapNumpyToFastAPI` always converted to grey (`convert("L")`), and clipped 12/16-bit at 255.
- HIK black level: set as float / read as enum on an integer node → both failed silently.
- Experiment used the first detector, not the `forAcquisition` one.
- Live-view scale bar used µm per canvas pixel for a CSS-pixel bar (2–4× wrong) → round 1-2-5 length.
- Turret: `calibrateObjective(homeDirection, homePolarity)` ignored its arguments; startup warning
  when `objective.homeDirection` and the positioner's `homeDirectionA` disagree; the calibration's
  "no focus" error names the turret.

### UC2-REST
- macOS auto-discovery skipped every `/dev/cu.usbserial-*`: a CP2102/CH340 board on Apple's driver
  was never found (this HAT, with the Linux port `/dev/ttyUSB0` in the setup). Now skipped only when
  a WCH duplicate port exists.
- `/laser_act` answers `{"qid":n,"state":"done"}` then `{"return":1}`; the second was logged as
  ERROR on every LED switch → debug.

### Firmware (not flashed, branch `fix/hardlimit-debounce` in `uc2-ESP`)
- Hard-limit trip on a rising edge of the raw end-switch input, no debounce: after homing (axis on
  its switch, `homeEndposRelease` 0) the release chatter trips the limit 0.16–0.27 mm off home, so
  the calibration's travel probe never reaches the far switch (also on beta.5). Patch: 5 ms debounce
  before the edge logic; `UC2_canopen_slave_motor_release` builds.

## Setup changes on the instrument

- `objectivePositions` `[0, -29000]` → **`[29000, 0]`** (approved): homing parks the turret with the
  20x on the axis at A = 0. Backup of the previous file in the session scratchpad.
- FRAME calibration results (4x affine, offset, placement, backlash, z0 6960) written by the run.
- `~/ImSwitchConfig/config/imcontrol_options.json` restored to its previous setup.
- Camera white balance locked at R 1.419 / B 2.388 (one-shot on glass), LED off.

## Open issues

1. **20x unusable**: its focus (≈ 17.3 mm) lies beyond the Z end stop (≈ 16.85 mm) with these
   slides. Mechanical (objective height / spacer) or a holder change; parfocality was never measured.
2. **Overview camera**: macOS refuses camera access to processes started from the Claude app
   ("not authorized, status 0"); and `cameraListIndex: 0` is the MacBook's FaceTime camera on this
   host. Selecting the camera by name would make the setup portable.
3. **Flat field**: vignetting and a slight colour shift per tile show as a grid on glass in every
   mosaic. A blank-glass reference (or a per-run estimate) applied to the canvas is the next
   image-quality step for histology.
4. **Tissue detection** (done here by a script on the prescan): grayscale cannot tell marker-pen ink
   from stain (slide 3); the prescan keeps only 70 µm/px grey previews. A colour prescan and a
   `detectTissue` → scan areas endpoint would make slide scanning one click.
5. **Focus points on tissue**: a 2×2 grid on an irregular section puts corners on glass (rejected).
6. **Speed units**: ESP32 stage speeds are firmware steps/s (×0.3125 µm on the FRAME), while
   StageMap documents µm/s; experiment "20000" = 6.25 mm/s.
7. File names repeat the run name and glue the index to it (`…_slide2_4x0_0_slide2_4x_0_…`).
8. UC2-REST sleeps 20 ms per serial write inside the write lock (≈ 3 % of a 4x tile).
9. Firmware debounce patch: review, build in CI, flash.

## Verification

- Unit tests: 862 passed; the 8 failures also fail on master (7 Arkitekt, 1 Aufgaben schema).
  New tests: stage-coordinate canvas, RGB OME-TIFF export, metadata-hub sync, focus-map reference Z,
  prescan lag retry.
- On hardware: firmware, calibration, both slide scans, autofocus with the camera stopped, focus-map
  re-run, RGB snap, scale bar, camera stopped after a run, white-balance "once" ending locked.
- Not verified on hardware: prescan lag retry (unit-tested), the frontend outside the scale bar.
