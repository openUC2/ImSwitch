# FRAME calibration: hardware test handover for the Raspberry Pi session (2026-11-04)

For a Claude Code session on the FRAME's Raspberry Pi, with the microscope attached.

The goal is to test on hardware what was built and checked only in simulation on the Mac:
- the pinhole-slide calibration target;
- the faster second-objective search;
- the slide orientation shown in the app.

Read `docs/FRAME_AUTOCALIBRATION_PLAN.md` first, in particular section 3b (the slide target)
and the rig measurements in section 3.

## 1. Code state

- ImSwitch branch `feat/AufgabenController`. Commit `10a9a66f1` "adding the 3-pinhole slide
  as a calibration sample" has the slide target and the second-objective changes.
- On top of it, uncommitted on the Mac (2026-11-04): the orientation of both slides and
  its display in the app. Files:
  - `FrameCalibrationController.py`
  - `frame_calibration_tools.py`
  - `FrameCalibrationApp.jsx`
  - `test_frame_calibration.py`
  - this document

  The user commits and pushes them; check with `git log --oneline -3` and `git status`
  on the Pi.
- Run first, on the Pi:
  - `python -m imswitch.imcontrol.controller.controllers.frame_calibration_tools`
    (self-check, a few seconds);
  - `python -m pytest -q imswitch/imcontrol/_test/unit/test_frame_calibration.py`
    (4 tests, about 2 min on a Mac; slower on a Pi).

  The slide test runs both objectives: a simulated turret with a 1.5x second objective,
  a 300/−200 µm parcentric offset and a −300 µm parfocal offset.

## 2. What changed (all in `FrameCalibrationController.py` / `frame_calibration_tools.py`)

1. **Calibration target setting `target`:** `"plate"` (FRAME calibration plate, as before)
   or `"slides"`. The app has a "Calibration target" selector next to Start. The plate map
   draws the selected target.
2. **Slide target (`SLIDES`):**
   - Hardware: the Heidstar HDS-SH-80120-IS4 4-slide insert, with a KiCad `pinhole-slide`
     Rev A in slots 1 (leftmost) and 4 (rightmost).
   - The slide: 76 x 26 mm. Three vias on its centre line, 19 mm apart: drills 0.2 / 0.3 /
     0.4 mm, the 0.3 mm one in the middle. Which end goes in first does not matter.
   - Nominal positions: slot centres x = 18400 + 30000·k, y = 40600 µm, taken from the
     "Heidstar 4x Histosample" layout (`frontend/src/axon/WellSelectorUtils.js`). Section 4
     says how good that is.
   - The "via" is slot 1's middle (0.3 mm) pinhole: same drill as the plate's via.
3. **Light first, then focus (slides only, `_findLight`):**
   - The black slide has no focus texture.
   - At the focus sweep's start height, the routine spirals for the defocused pinhole's
     light (`fct.light_spot`, size-agnostic) and moves it near the image centre. It sets
     the exposure on that light (99.9th percentile to ~40 %), then focuses on the pinhole.
   - After focus, the via exposure is set on the focused pinhole (~50 %), not on a plate.
4. **Placement (slides, `_placementByPinholes`):** five check-point pinholes, each found in
   single frames and placed in plate coordinates with the first objective's camera matrix:
   - slot 4's centre, 90 mm from slot 1's: the insert rotation and the stage scale over
     90 mm (`baselineScale`);
   - each slide's ±19 mm pinholes: its rotation in the slot;
   - their measured diameters: `smallEnd`, i.e. whether the 0.2 mm pinhole is at −y or +y.

   Stored in `result.placement.slides["1"/"4"]`. The app draws "0.2"/"0.4" at the measured
   ends on the map and lists them in "Stored calibration".
5. **Second objective, both targets:**
   - The search starts where the stored `objective.xyOffsets` put the landmark (an
     objective switch moves the stage by offs[k] − offs[k0]), not at the nominal via.
   - On the slides, the landmark is the smallest pinhole found in the placement step (the
     0.2 mm one): at a real 20x a 0.3 mm hole is ~2000 px, the full frame height.
     Parcentricity is measured against that pinhole's measured plate position.
6. **Overview steps on the slides:** the motion check and the overview mapping are skipped
   (no overview landmark on the slides).
7. **Preflight on the slides:** a fully dark start frame is a warning, not an error (the
   stage may sit over the opaque slide).

## 3. Hardware test plan, in order

Note per run: the step results, the report folder (`~/imswitch_calibrations/frame_calibration/<ts>/`,
or "Download raw data (ZIP)" in the app) and the PDF. Look at the figures of every failed
or warned step.

1. **Plate run (regression and reference).**
   - Target "FRAME calibration plate", both objectives.
   - Expect the same results as the last good plate run.
   - New: the second objective should find the via in the first few tiles, because it now
     searches at the stored parcentricity. Compare the number of "tile" trace entries in
     objective1 with an older report.
2. **Slide run right after it, without touching anything but the holder.**
   - Swap in the 4-slide insert with pinhole slides in slots 1 and 4, pushed against the
     stops. Target "4-slide insert with pinhole slides".
   - **Layout check:** the offset step's detail says "the via was (dx, dy) µm from where
     the previous offset expected it". Because the previous offset came from the plate,
     that is how far slot 1's real middle pinhole is from the Heidstar layout's slot 1
     centre, in the well-plate frame (plus the slide's play in its slot). Report it.
     - If it is more than ~0.5 mm, correct `_SLOT_X`/`_SLOT_Y` in
       `frame_calibration_tools.py` and `wellLayoutDevelopment` in `WellSelectorUtils.js`
       together.
     - Ask the user first: this moves every slide scan.
   - Watch, in this order:
     - "Pinhole light found, objective 0": was the light found, at which Z and exposure?
     - The focus sweep: one clear peak?
     - Via search and centring.
     - "Placement: pinholes of the slides": 5 tiles, yellow cross vs green circle; insert
       rotation; stage scale (should be 1.000 ± 0.001 if 3200 steps/mm is right); per-slide
       orientation.
     - Second objective on the 0.2 mm pinhole.
     - Switch-back residual.
   - Check that the app shows "0.2"/"0.4" at the correct ends: look at the slides with the
     eye (the 0.2 mm via is the smallest).
3. **Second slide run:** the second objective should now start right at the pinhole (stored
   parcentricity), and the first objective's focus at the stored Z (window ±2.5 mm).
4. **Turn a slide end for end and re-run placement:** the orientation of that slot must
   flip in the app; positions and rotation must stay.

What is expected to be weak, and what to do:
- **The real defocused pinhole blur is much larger and dimmer than simulated.** If
  `_findLight` finds nothing: look at the brightest frame of the search, and check
  `fct.light_spot`'s `min_ratio` (4x the 5th percentile) against the real black level at
  maximum gain. A hint Z from a previous plate run (`result.objectives["0"].z`) starts the
  search near focus, so a first plate run helps.
- **Switch-back residual of about the backlash** (warned above 10 µm). The objective
  switch's XY compensation reverses an axis against the approach direction (simulation:
  exactly the Y backlash). The alternatives are 'Apply backlash', or making
  `ObjectiveController` finish its XY offset moves from the negative side. The second
  changes every objective switch: ask the user first.

## 4. Findings from the Mac session that need a decision (not changed yet)

1. **Heidstar layout (`wellLayoutDevelopment`, also `SLIDES`).**
   - The holder STEP gives the slot pitch (30 mm) and centring: the slides lie ±15 / ±45
     mm from the insert centre, and 2.5 mm off it along their length.
   - With the insert centred on the SBS frame, the layout agrees with the STEP to
     ~0.1–0.5 mm in x and ~0.4 mm in y (spring side at +y).
   - The CAD does not say where the insert sits relative to the well-plate origin. Test
     3.2 measures that; the "−9000" in the layout's y is an earlier empirical correction.
   - Note: labware `slide_4x_histosample_heidstar` is something else, a single 75 x 25 mm
     slide with 4 sample regions.
2. **Well plates.**
   - Calibration frame: origin = the A1-notch corner, +y towards row P/H, so A1 of an SBS
     96-well plate sits at (14.38, 11.24) mm.
   - The backend labware JSONs (`imswitch/imcontrol/model/labware/definitions/openuc2/`)
     use the Opentrons convention, y from the front edge: A1 at (14.38, 74.24), H1 at
     (14.38, 11.24); 384: A1 (12.13, 76.49).
   - Neither the loader, `labware_api._labware_to_layout_dict` nor `LabwareSelectionPanel`
     flips y. So a labware well position sent to the stage is mirrored top/bottom against
     the calibrated frame: row A lands on row H (96) or P (384).
   - **Verify on the rig first:** a 96-well plate in the holder (A1 chamfer in the origin
     corner), select labware A1 in the experiment UI, move there, see which well is under
     the objective.
   - If mirrored, the fix is one flip, `y = yDimension − y`. Put it where labware becomes
     stage coordinates (`_labware_to_layout_dict`, `selectWellsByPattern`,
     `applyWellSelectionToExperiment`, or once in the loader), and adapt the labware and
     Aufgaben tests. Ask the user which.
   - The legacy frontend "Wellplate 384" (`generateWellLayout384`) is top-referenced (A1 at
     (12.0, 9.0) mm): right direction, 0.13 mm off SBS in x.
3. **Fast search (question from the user): not implemented.**
   - The search is stop-and-go. Each tile: move (each axis verified by a read-back), settle
     0.2 s, wait for 2 new frames, so ~0.5–0.7 s per tile.
   - Worst case: ~80 tiles at 4x over 4 mm (~1 min); hundreds at 20x over 2 mm.
   - The stored parcentricity and the stage offset make later runs start on the landmark.
   - The StageMap "prescan" (`StageMapController.startPrescan` / `_sweepLine`, used by the
     experiment UI) is the fast mode: continuous X sweeps with the camera free-running.
     Frames are placed by time, pinned to the measured sweep duration; the camera lag (up
     to 0.5 s) is calibrated from a forward/backward pair.
   - A fast pre-search on that pattern is worth building on the Pi, with hardware:
     - serpentine sweeps over the search area at the via exposure;
     - `find_bright_disc` / `light_spot` on every frame;
     - on the first hit, the time-interpolated position, then the existing stop-and-go
       search with a small radius around it;
     - fall back to the full spiral.
   - Copy the pattern (`_NonBlockingMove`, time-pinned placement); don't import the other
     controller. Measure the lag and the hit-position error at 2–5 mm/s.

## 5. Gotchas (Mac and Pi sessions)

- **Snapshot endpoint:** `RecordingController/snapNumpyToFastAPI` clips 12-bit frames to
  8 bit. Judge exposure from the report's raw 16-bit PNGs.
- **LED:** nonlinear, practically off below ~100.
- **Gain:** an ImSwitch restart resets the Hik gain to the config value. The calibration
  runs at maximum gain (`maxGain`) and restores it afterwards.
- **Hik stream:** it hung for good after ~185 ms exposures and needed a USB replug. The
  exposure is therefore capped at 100 ms.
- **Stage position reads:** the UC2 firmware sometimes reports another axis's position.
  The controller uses its verified move targets (`_at`), not fresh reads.
- **Manual-focus fallback:** it blocks a test forever, so simulated runs set
  `manualFocusFallback: False`.

## 6. Ways of working the user expects

- Smallest diffs; reuse what exists.
- Keep the simulation tests passing, and add a check for every detector or geometry fix.
- Commit only when asked.
- Report with numbers, the failing step's message and the report folder.
- Say clearly what was verified on hardware and what was not.
