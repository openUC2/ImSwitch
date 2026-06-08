# ImSwitch Issues - Session Plan

Organized from the frame user story review with Franzi/Ethan/Bene.

---

## Session 1: LiveViewController & Camera Stream Stability

Focus: Fix multi-camera handling and stream reliability.

- [ ] Per-detector settings don't work correctly with two different physical cameras (works with virtual microscope but not webcam + HIK camera)
- [ ] Ensure no race condition when two camera streams run simultaneously
- [ ] Define expected behavior when two streams run at the same time (document or fix)

---

## Session 2: Autofocus Frontend Cleanup

Focus: Simplify the autofocus UI by removing unused controls and exposing missing ones.

- [ ] Expose the sub-sample divisor in the Two Stage autofocus frontend
- [ ] Remove "Defocus Z" from the autofocus frontend
- [ ] Remove "Illumination Channel" from the autofocus frontend

---

## Session 3: Manual Pixel Calibration Fixes

Focus: Fix calibration UI bugs and logic issues.

- [ ] Objective selector: always display objective name (not just 0/1); if "current" is selected and undefined, prompt user to move to a defined position
- [ ] Fix unit label: "100mm" should be "100um"
- [ ] Add backlash compensation in Y
- [ ] Show computed values first, then let user accept/save (match automatic calibration flow)
- [ ] Fix subsampling not being taken into account (regression - was working before)

---

## Session 4: Automatic Pixel Calibration & Stage Offset Calibration

Focus: Fix calibration pipeline and stage offset logic.

**Automatic Pixel Calibration:**
- [ ] Fix deshole layer issue
- [ ] Add backlash compensation

**Stage Offset Calibration:**
- [ ] Ensure dx/dy for field of view follow camera dimensions
- [ ] Verify correct dx/dy values for samples and test they work
- [ ] Fix: clicking hotpixel (hole position) in heatmap doesn't move stage to correct position after recalibration (likely stale offset pointing to old position)

---

## Session 5: Objective Controller & Live View Integration

Focus: Wire objective controller with live view settings.

- [ ] Add exposure time / gain settings from live view into the objective controller UI

---

## Session 6: WellPlate App / ExperimentController - Core Fixes (Bene)

Focus: Functional bugs and missing features flagged by Bene.

- [ ] Add camera toggle (overview vs. widefield) to the WellPlate app
- [ ] Fix: automated overview scan may use stale/old values

---

## Session 7: WellPlate App / ExperimentController - UX Improvements (Franzi)

Focus: UI/UX feedback from Franzi's walkthrough.

- [ ] Add option to keep illumination settings as-is when light is already on
- [ ] After scanning, show/link to the file location of scan results
- [ ] Improve area selection UI: separate area selection tools from task actions (the 8 buttons are unintuitive)
- [ ] Rename "Parameter" tab to "Scanning"
- [ ] Fix duplicate tabs: Parameter, Points, and Live View appear in both left and right panels - should only be in one
- [ ] Add legend for well selector markers (green rectangle, blue dot, red dot)
- [ ] Scale selection cursor when zooming into well selector for more precise selection
- [ ] Rename "stitching" to "merging/tiling" to avoid confusion with actual image stitching
- [ ] Hovering over a position in the positions list should highlight the corresponding area in the well selector
- [ ] Fix or clarify "Reset History" button (currently does nothing or is unclear)

---

## Session 8: General UI & Settings Cleanup

Focus: Small cross-cutting UI issues.

- [ ] Remove duplicated apps from the side menu
- [ ] Add tooltips/information to reconnection/pairing settings UI

---

## Session 9: Performance & Infrastructure (Investigation)

Focus: Investigate systemic issues - these may require profiling and deeper analysis.

- [ ] Investigate ~3GB RAM consumption - find memory leak or inefficient process handling
- [ ] Investigate network interruptions between Raspberry Pi and laptop
- [ ] (Ethan) Evaluate why forklift version updates are so large - can updates be made more efficient?

---

## Notes

- Sessions 1-5 are backend/calibration-heavy and have some interdependencies (calibration pipeline)
- Sessions 6-7 are WellPlate/ExperimentController focused
- Session 8 is quick cleanup that could be done anytime
- Session 9 requires profiling tools and hardware access - best done on the actual microscope setup
- Consider doing Session 3 before Session 4 since manual calibration fixes may inform automatic calibration work
