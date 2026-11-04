# FRAME one-button calibration — plan and status (2026-10-29)

Goal: the user drops the FRAME calibration plate (KiCad `calibration-plate` Rev E, SBS
384-well footprint) into the holder, clicks **Calibrate**, and ~5 min later the setup JSON
holds a reproducible, verified mapping between all coordinate frames — with a report that
says what passed, what failed and why. No AI, no clicking on images.

Status: implemented (FrameCalibrationController + frame_calibration_tools + FrameCalibrationApp), simulation test passes; on hardware verified up to homing + overview motion check (main camera SDK missing on the dev Mac). Sections 4-10 are the original plan; the code is the reference where they differ.

---

## 1. What gets calibrated (outputs)

| # | Result | Where it is stored (existing unless marked NEW) |
|---|--------|--------------------------------------------------|
| 1 | Stage offset: device → user frame, **user frame = plate frame** (origin = A1-notch corner, X along 127.76 mm, Y along 85.48 mm, µm) | `positioners.ESP32Stage.stageOffsets` via `PositionerController.setStageOffsetAxis` |
| 2 | Camera↔stage affine per objective (scale x/y, rotation, shear, flip) for `WidefieldCamera` | `PixelCalibration.affineCalibrations["WidefieldCamera::<slot>"]` via `PixelCalibrationController.setCalibrationData` |
| 3 | Parfocality: plate-surface Z per objective | `objective.zPositions` via `ObjectiveController.saveCurrentZPosition` |
| 4 | Parcentricity: XY offset of objective 1 relative to objective 0 | NEW `objective.xyOffsets = [[0,0],[dx,dy]]` (µm); optionally applied in `moveToObjective` |
| 5 | Backlash X/Y (optional step) | positioner backlash via `PixelCalibrationController.applyBacklash` |
| 6 | Overview camera → stage homography `H` with the stage-translation model (see §2) | NEW entry in `overviewRegistration` (`ObservationCamera__FRAME calibration plate`) |
| 7 | Long-baseline checks: plate-vs-stage rotation θ, stage scale error (stepsize) | NEW `frameCalibration` report block (measure + warn only in v1) |
| 8 | Report (per step: status, numbers, thresholds, images) | `~/imswitch_calibrations/frame_calibration/<timestamp>/` + summary in setup JSON |

Assumptions (from the brief): axis assignment and signs are already correct (moving X moves
X, positive is positive); the FRAME is inverted (objective below, LED + overview camera in
the illumination arm above); the sunburst side of the plate faces the objective, the
well-label side faces the overview camera (matches the LiveView screenshot).

## 2. Coordinate frames and the two formulas everything hangs on

- **Device** `d` (µm): firmware position after homing (stepsize applied). **User** `u = d − offset`
  (existing contract). We choose `u ≡ plate` coordinates, so "move to (63850, 42025)" puts the
  plate's center hole under the optical axis, and well A1 of a 96-plate is at (14380, 11240)
  — same convention the existing `_KNOWN_CALIBRATION_POINTS` already use.
- **Main camera** (per objective k): `Δu = A_k · Δpx` (2×2, existing `affine_matrix`
  convention; translation column unused). The optical axis is the image center.
- **Overview camera** (fixed to the frame, plate moves with the stage): a plate point `p`
  appears at pixel `o = H · [p − u, 1]ᵀ` (homogeneous), where `u` is the current stage user
  position. One fixed `H`, valid at every stage position. Inverse: the plate point under
  overview pixel `o` is `p = H⁻¹(o) + u`; to bring it under the objective, move to `u' = p`.
  This is what the existing overview registration lacks (its H is only used to warp an overlay
  and is never combined with stage motion).

## 3. The calibration plate (nominal landmarks, from the KiCad file)

Stage/user frame = plate top view as inserted: well-label side up, A1 notch at the
stage origin (0,0), +x along 127.76 mm, +y towards row P. The sunburst (KiCad front)
side faces the objective, so KiCad x is mirrored: x_user = 127.76 mm - x_kicad.
Source of truth in code: `PLATE` in `frame_calibration_tools.py`.

| Landmark | Position (stage µm) | Geometry | Used for |
|----------|---------------------|----------|----------|
| Centre via | (63 910, 42 025) | drill ⌀300, copper annulus ⌀600, sunburst rays converge on it | main camera: bright disc in transmission -> search, focus, affine, stage offset |
| Round hole | (21 260, 60 000) | ⌀6000 routed | overview camera landmark (dark, also under the LED glare) |
| Rect hole | centre (21 260, 51 500) | 6000 x 7000 routed | main camera: orientation probe + 4 edges -> plate rotation |
| Via grid | x 16 885..22 885, y 16 225..22 225 | 7 x 7, pitch 1000 | (unused, scale check later) |
| Corner tags | front side only | ArUco ORIGINAL | not visible to the overview camera (it sees the back side) |

Measured on the rig (2026-10-29): overview camera 1920x1080, ~80 µm/px, oblique
(the ⌀6 hole appears 69x47 px); +2 mm X -> (+22.5, +0.1) px, +2 mm Y -> (+5.9, +17.4) px.
Phase correlation reported 0 px for the Y move because the fixed LED glare dominates:
motion direction is therefore always measured on a landmark (round hole, via).

Measured on the rig (2026-10-30, 4x, Hik 3072x2048, 0.617 µm/px):
- The plate focuses at Z ≈ -4450 µm, 3.6 mm below the stored 4x `zPositions` (-871,
  set on another sample). The Laplacian focus metric peaks sharply (FWHM ~300 µm); the
  frame contrast (std/mean) rises monotonically and is useless. Hence: a coarse sweep from
  the homed Z towards the sample that stops after the peak, then a fine sweep; the
  AutofocusController (±range around a stale Z) is no longer used.
- At the live exposure the via glows over half the frame and the sunburst rays dominate;
  at ~1/10 exposure the via is a crisp disc (finished hole ~250 µm) and the plate is
  black. Via work runs at `viaExposureFactor` of an auto exposure (rays/plate p95 at 40 %).
- The LED is nonlinear: below ~100 it is practically off. Overview frames at LED 60 show
  both holes by room light; at 150+ the glare washes them out. Overview falls back to
  full light if the holes are not found.
- `snapNumpyToFastAPI` clips 12-bit frames to 8 bit (PIL "L"), do not judge exposure from it.

Measured on the Pi rig (2026-11-02, enclosed FRAME, runs 5/7/8 agree):
- 4x: 0.598 µm/px, camera rotation -179.6°, plate focus Z -4380..-4420 µm. Slot 1 ("20x"):
  0.240 µm/px, i.e. 2.5x the 4x, not 5x: `objective.magnifications` [4, 20] is wrong for this
  turret (both behave as 1.44x nominal on this tube lens if slot 1 is a 10x). The routine now
  measures the pixel size from +/- stage moves (phase correlation) before the via search.
- Parcentricity of slot 1: (-1696, -289) µm (+/-2 µm over three runs), more than the old 800 µm
  search radius; switch-back residual ~9-11 µm; via re-found 8-9 µm from the image centre.
- Overview webcam: 640x480, auto exposure ("aperture priority"). The LED is either off (<=100)
  or nearly full (>=120); in the dark enclosure LED 60 gives a black frame and auto exposure
  saturates the LED cone over both holes. Hole frames: LED on, webcam manual exposure 10 ms
  (new `exposure_mode` parameter of OpenCVCamManager, V4L2 only). Through the rect hole the
  camera sees the objective turret (reflections that change with the stage position), so only
  the round hole is tracked after the pair is identified once; frames are taken with Z at the
  homed top (at focus the objective under the holes shifts the round hole by ~6 px).
- UC2 firmware sometimes reports the A position under X ("X reads 30377 after a move to 64496");
  offsets and positions are taken from verified move targets, not from fresh reads.
- Open: plate placement (rect-hole edges) gave -0.07° / 5928x6825 µm in three runs and
  -0.20° / 6467x7534 µm in one (run 7, focus 40 µm lower); edge frames are now in the report.

Changes after the Pi runs of 2026-11-03:
- Main camera: maximum gain during the run (`maxGain`), exposure set on the plate (`autoExposure`,
  95th percentile towards 30 %, cap 100 ms); the user's exposure and gain are restored after the run.
  The via search runs short (4x: 0.05 ms at 24 dB; the second objective scaled by the magnification
  ratio squared) so the rays stay dark; centring and the 3x3 grid run with the via ~5x saturated,
  because the unsaturated via core is not flat (10x: 799 px / 25 px fit error instead of 1000 / 6).
- A via candidate that cannot be centred (a ray tip cut by the frame edge) is reported and the
  search continues (up to five).
- Overview frames: LED + short manual exposure first, room light last (in daylight it found
  silkscreen wells as a hole pair).
- Stage positions in the routine come from verified move targets; one fresh read returned X = -26
  instead of 63910 and drove the stage to the end stop.
- Report: every step adds figures (raw frame, yellow = expected, green = measured) shown in the
  wizard; `getFrameCalibrationReportPdf` (summary + figures) and `getFrameCalibrationReportZip`
  (report folder with the raw 16-bit frames) for download.

Decisions (2026-10-29): holder repeats within ±1 mm; Z home = farthest from the sample,
negative Z = towards the sample; manual-focus pause when autofocus fails; parcentricity
stored as `objective.xyOffsets` and applied with the existing Z-follow toggle; stepsize
is trusted (3200 steps/mm), rotation is measured and reported, not applied; the old
FRAME-Settings calibration tabs and StageCenterCalibration were removed.

### 3b. Alternative target: pinhole slides in the 4-slide insert (2026-11-04)

Selected with the setting `target` = `"slides"` (wizard: "Calibration target"). Geometry is
`SLIDES` in `frame_calibration_tools.py`.

- **Holder:** Heidstar HDS-SH-80120-IS4, a 170 x 130 mm stage insert with four slide slots
  on a 30 mm pitch. The slides lie along y. Source: `~/Downloads/00_Heidstar-HDS-SH-80120-IS4.step`;
  its four slide bodies are 76 x 26 x 1.4 mm.
- **Slides:** KiCad `mAIkroscope-electronics-hardware/kicad/pinhole-slide` Rev A, in
  slots 1 (leftmost) and 4 (rightmost).
  - Black PCB, 76 x 26 mm, 1 mm chamfers.
  - Three vias on the centre line, 19 mm apart: drills 0.2 / 0.3 / 0.4 mm, with the
    0.3 mm one at the slide centre.
  - The positions are symmetric, so it doesn't matter which end goes in first.
- **Nominal positions:** the slot centres of ImSwitch's "Heidstar 4x Histosample" layout
  (`frontend/src/axon/WellSelectorUtils.js`): x = 18400 + 30000·k, y = 40600, in the same
  plate frame as the SBS plate. Not measured from the CAD; the search radius covers
  ±4 mm.
- **Via:** the 0.3 mm pinhole of slot 1 at (18400, 40600). It is the same drill as the
  plate's via, so search, centring, camera matrix, stage offset, backlash and the second
  objective run unchanged.
- **Focus:** no texture, since the slide is opaque. `_findLight` spirals at the start Z
  for the defocused pinhole's light (`light_spot`, size-agnostic) and brings it near the
  image centre. The focus sweep then runs on the pinhole itself. Exposures are set on the
  pinhole's light, not on a plate.
- **Placement** (`_placementByPinholes`):
  - slot 4's centre pinhole, 90 mm away: insert rotation and the stage scale over 90 mm;
  - slot 1's outer pinholes (±19 mm): the slide's rotation in its slot;
  - the pinhole diameters: which end has the 0.2 mm one.
- **Second objective:** uses the smallest pinhole the placement check found (the 0.2 mm
  one), not the 0.3 mm one. At a real 20x (0.123 µm/px) a 0.3 mm hole is ~2000 px, the
  full frame height, so it can't be centred or gridded. Parcentricity is measured against
  that pinhole's plate position from the placement step (through the first objective, it
  is centred there).
- **Search centre, both targets:** the second objective's search starts where the stored
  `objective.xyOffsets` put the landmark, not at the nominal via. With a previous
  calibration it is found in the first tiles.
- **Overview:** the motion check and the overview mapping are skipped (no landmark the
  overview camera sees yet). Follow-up: track the slides' QR codes.
- **Stage offset:** the offset then makes slot 1's pinhole sit at its layout position,
  which makes the slide layout exact. The well-plate holder and the slide insert are
  different references, so switching holders needs the matching calibration.

## 4. The routine (v1), step by step

Every step: timeout, abort check, pass/fail with a one-line hint, numbers into the report.
Critical steps stop the run; optional ones (overview, backlash, long-baseline) warn and continue.
Estimated total 4–7 min.

| # | Step | How (reuse first) | Pass | Failure → hint |
|---|------|-------------------|------|----------------|
| S0 | Preflight (5 s) | fresh-frame check on both cameras (frame counter advances); LED off/on frame-mean delta; two consecutive `/motor_get` reads that are not all-zero; arm collision protection if the GPIO slave exists (`UC2ConfigController.armCollisionProtection`); turret: verified home of A + move to slot 0 | all checks | "camera stale", "no illumination (LED/driver)", "stage not responding", "turret endstop not detected (A homing timed out)" |
| S1 | Safe homing (≤60 s) | `home_z` → lift Z 3 mm away from the sample → `home_x` → `home_y`, each via a **verified** home (uc2rest returns `done`/`timeout`; see §6). Overview camera watches: motion during, stillness after (archived `_compute_frame_motion`) | each axis `done` < `homeTimeout` | "X homing timeout — endstop/polarity/wiring", "Y reported done but no motion seen" |
| S2 | Plate check, overview (3 s) | detect L2+L3 as two dark blobs with the right size ratio and 8.5 mm spacing; if `H` is stored: predicted vs detected < 2 mm | plate found; pose plausible | "plate not detected / wrong orientation (notch must be at the origin corner)"; no camera → warn, continue |
| S3 | Coarse find L1, main camera (10–60 s) | Z → `zPositions[0]` (first run: S4 coarse sweep at nominal XY first); spiral tiles (step 0.8·FOV) around nominal L1 (or the S2 prediction), radius R (default 4 mm); per tile: bright-disc detector sized ⌀0.3 mm/scale | disc found with confidence | "center hole not found within R mm — plate seated? LED on? increase R"; attach tile mosaic |
| S4 | Focus (10–30 s) | `AutofocusController.doAutofocusBackground` called directly (blocking, returns z or None): coarse (range 1000 µm, step 50, binned) then fine (range 100, step 5); validity from `_assess_focus_curve`; Z clamped to `zPositions[k] ± 1500`, approach from the far side; retry once with doubled range | valid peak, not at range edge | manual-focus checkpoint (pause, live view, Continue) or fail — configurable |
| S5 | Center L1 (5 s) | circle fit on the disc edge → px error → `A_k` (or nominal scale on first run) → move; ≤5 iterations to < 1 px; first iteration also checks the sign consistency (error must shrink) | residual < 1 px | "affine/sign inconsistent — stage moved the wrong way" |
| S6 | Affine, objective 0 (30–60 s) | existing `PixelCalibrationClass.calibrate_affine(pattern="grid", n_steps=3, step=0.25·FOV)` with software trigger + FFT tracking on the sunburst; cross-check: L1 circle-center tracked over the same grid → second affine; validate: existing `validate_calibration` + both methods agree (scale 1 %, rotation 0.5°) + nominal scale ±15 % | valid, consistent | numbers + "texture/exposure" or "scale far from nominal (wrong objective selected?)" |
| S7 | Stage offset (1 s) | L1 centered → `setStageOffsetAxis(X, knownPosition=63850, currentDevicePosition=d_x)`, same Y | — | — |
| S8 | Long baseline (20 s, optional) | drive to the 4 edge midpoints of L3 (±X, ±Y, 3.0/3.5 mm from its center); 1-D intensity profile across each straight edge → sub-pixel edge point in user µm (via `A_0`); → L3 center, width, height; AF-refine at L3 if contrast is low | center within 0.3 mm of nominal; width/height within 1 % | warn: plate rotation θ > 0.3° (holder), stepsize error > 0.5 % (measure only in v1) |
| S9 | Backlash (40 s, optional, default on) | existing `measure_backlash` X and Y at L1; apply with `applyBacklash` when > 2 µm and the fit improves the residual | — | — |
| S10 | Objective 1 (60–120 s) | retreat Z 1 mm → `moveToObjective(1, skipZ=True)` → poll `getstatus()` until `lastCompletedObjectiveRequestId` matches; Z: `zPositions[1]` if known else coarse+fine AF from far; spiral for L1 within 1.5 mm (step 0.8·FOV₁); center (circle fit); **parcentricity** Δ = d_now − d_S7 → `objective.xyOffsets[1]`; `saveCurrentZPosition(1)`; affine as S6 with step 0.25·FOV₁; validate scale₁/scale₀ ≈ mag₀/mag₁ ±10 %; switch back to slot 0 and re-center L1 → round-trip residual | residual < 5 µm | "L1 not found after switching (parcentricity > 1.5 mm?)", "objective move timeout" |
| S11 | Overview homography (5 s) | from frames collected at S1/S3/S7/S8/S10 (stage at known `u_i`, L2/L3 centers + ArUco corners detected) fit `H` with `cv2.findHomography(RANSAC)` on `(p_j − u_i) → o_ij`; store with `rmse_px`; add `overview_px_to_user` / `user_to_overview_px` helpers | ≥ 4 inlier correspondences, rmse < 1.5 px | warn only; later runs use `H` for S1/S2 verification |
| S12 | Verify + report (10 s) | move to L1 by plate coordinates with slot 0 → detect → residual; write report JSON + PNGs (tiles, focus curves, affine residuals, overview detections); summary into setup JSON | residual < 3 µm | headline fail |

Not in v1 (deliberately): applying θ (needs a rotated user frame in the positioner), applying
the stepsize correction, sunburst-direction search cue, crash-sensor integration (hooks exist:
`sigCollisionStatusUpdate` → abort), Aufgaben wrapper.

## 5. Error-handling model

- One daemon thread runs a list of `Step(name, fn, timeout_s, critical)`; state machine
  `idle → running → (paused) → finished | failed | aborted`; abort = `threading.Event` checked
  between steps and inside every grab/move loop, then `stopAll()` (same pattern as
  `PixelCalibrationController`).
- **Verified move** helper (the core of reliability — `ESM.move` returns `None` and never raises,
  a lost `/motor_get` reply reads as zeros, sub-µm absolute moves are skipped by uc2rest):
  blocking move → read position (re-read when all axes read exactly 0) → `|pos − target| ≤ tol`
  else retry once → `StageError("axis X did not reach target: commanded a, read b")`. Where the
  main camera has texture, the FFT tracker gives a second opinion ("image did not move").
- **Verified home**: `_homeAxis` must return the uc2rest result; `timeout` → fail with the
  endstop hint; Python wait = firmware `homeTimeout` + margin (today it is a fixed 20 s).
- Manual checkpoint (`paused`): only for focus in v1; `continueFrameCalibration()` resumes.
- Mutual exclusion: entry in `busy.py CHECKS` + `busy_reasons()` so experiments/Aufgaben and
  the calibration cannot run together.
- Progress: poll-only in v1 (`getFrameCalibrationStatus` at 1 Hz, as the existing tabs do);
  a `Signal(dict)` broadcast can be added later with one WebSocketHandler branch.

## 6. Prerequisite fixes in existing primitives (small, needed for reliability)

| Where | Fix | Size |
|-------|-----|------|
| ImSwitch `ESP32StageManager._homeAxis/home_*` | return the uc2rest result; pass Python wait = `homeTimeout{ax}` + 5 s (`Home.home(timeout=…)`) | ~10 lines |
| UC2-REST `home.py` callback | parse the firmware's real frame `{"home":{"stepperid","status":"done"|"timeout","pos"}}` so `isHomed` updates (verify the frame on hardware first) | ~10 lines |
| ImSwitch `ESP32StageManager.frameHomingProcedure` | restores Z to 0 instead of the stored height (`_zPositionPriorHoming` reset by `home_z`); routine does its own Z handling anyway | 2 lines |
| config key | `safePositionZHoming` is never read (code reads `safeDistanceZHoming`) | 1 line |
| UC2-REST `motor.get_position` | lost reply → zeros; routine re-reads; proper fix = return `None`/raise | later |
| `ObjectiveController.moveToObjective` | non-blocking + unawaited relative Z: routine uses `skipZ=True` + polling; add `xyOffsets` compensation if agreed (Q4) | ~15 lines |

## 7. Architecture (minimal)

Backend (ImSwitch):
- `imswitch/imcontrol/controller/controllers/FrameCalibrationController.py` (~500 lines):
  `startFrameCalibration(options)`, `stopFrameCalibration()`, `continueFrameCalibration()`,
  `getFrameCalibrationStatus()`, `getFrameCalibrationReport()`, `getFrameCalibrationPlate()`;
  step runner + steps that call the existing controllers through `self._master.getController`.
- `.../controllers/frame_calibration/` pure functions, numpy/cv2 only, no hardware (~400 lines):
  `plate.py` (landmark table above, as a layout entry), `detect.py` (bright-disc circle fit,
  1-D edge, overview hole blobs, ArUco), `fit.py` (circle from edge points, similarity fit,
  homography with the stage-translation model, focus schedule). Each with a synthetic-image
  self-check.
- `SetupInfo`: optional `frameCalibration: dict`, `objective.xyOffsets`; everything else reuses
  existing fields. `"FrameCalibration"` added to `availableWidgets` of the FRAME setup.

Frontend:
- `frontend/src/components/FrameCalibrationApp.jsx` (~400 lines): Start/Abort/Continue, a few
  options (objectives, backlash, manual-focus fallback, search radius), step list with status
  chips and live numbers, two live views (main + overview via `startLiveView(detectorName)`),
  report table + download. `backendapi/apiFrameCalibrationController.js` (one wrapper file),
  one `appRegistry.js` entry (category CALIBRATION, `requiredControllers`), one render line
  in `App.jsx`. The FRAME Settings tabs stay as expert tools.

## 8. Tests

1. Unit (no hardware): `_test/unit/test_frame_calibration_fit.py` — synthetic disc/edge/plate
   images: circle center < 0.1 px, edge < 0.1 px, affine round trip, homography round trip,
   step-runner abort/timeout/failure paths.
2. Virtual microscope: new `imagePath: "frame_calib_plate"` sample rendered from the KiCad
   SVG (center 10 × 10 mm at 2 µm/px; sunburst, via, crosshair) — the existing
   `VirtualMicroscopeManager` already has a defocus-aware stage and an objective toggle, so
   S3–S7 and S10 run headless in CI; S8/S11 are skipped in sim.
3. HIL/e2e on the hil-runner infra: one test that runs the routine on the FRAME and asserts
   pass + repeatability over 3 runs (L1 position std < 5 µm, scale std < 0.5 %, round-trip
   objective residual < 5 µm).

## 9. Interactive hardware session (next), in order

1. Capture raw `/home_act` frames → fix the uc2rest callback; measure homing repeatability
   (home 3× → L1 centroid std); check the Z lift/`safeDistanceZHoming` behavior and the A-axis
   endstop readout.
2. Overview camera: frames at home and at 2–3 stage positions; tune the hole-blob thresholds;
   decode all 8 ArUco ids; fit a first `H`; re-home and check predicted vs detected holes.
3. Main camera at L1: LED level/exposure in transmission; confirm the via is a bright disc
   and the annulus/rays are dark; record the px diameter at both objectives.
4. AF curves on the sunburst at 4x and 20x (coarse/fine ranges, metric choice).
5. Run the existing `calibrate_affine` at both objectives → ground-truth scales; resolve the
   "both 0.6 µm/px" inconsistency.
6. Build S1–S7 first, compare with the manual tabs' results, then S8–S12.

## 10. Open questions

See the chat summary (holder repeatability, Z semantics, manual-focus fallback, parcentricity
compensation, rotation/stepsize apply, controller-vs-Aufgaben, who fixes the primitives).
