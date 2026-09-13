# Frame Scan Stability Backlog

Franzi's field complaints (frame user story review, Sep 2026) traced against
`feat/flimlabsintegration` @ `b3444d3e`. Companion to the earlier
[`ISSUES_SESSION_PLAN.md`](ISSUES_SESSION_PLAN.md) (camera stream / calibration /
WellPlate UX from the same review series) — this one covers layout state, jog
controls, focus maps, autofocus, multi-site scanning, and external storage.

**17 complaints reduce to 5 structural causes.** Fix the cause, not the symptom —
most WPs below are net deletions. Each WP is scoped to be one Claude Code session:
open it, read only that section (plus the Squid reference where it says to), go.

**Ground rule for every WP:** reduce complexity, don't add helpers. If a fix seems
to need a new abstraction, the fix is probably wrong — look for the duplicate path
that already does this and delete one of the two instead.

**Reference implementation:** Squid (`C:\Users\benir\Documents\Squid\software`) has
solved the position-model and autofocus-reporting problems more simply than we
have. See [§ Squid reference](#squid-reference) at the bottom — WP-05, WP-06 and
WP-11 all depend on reading it first.

## Root causes

| # | Cause | Pattern |
|---|-------|---------|
| C1 | Shared state, no owner | Any mount effect can overwrite the global `experimentState`/`wellSelectorState` |
| C2 | Two paths, one job | Duplicate layout sources, tile-key schemes, folder creators, stitched-TIFF writers drift apart |
| C3 | Failure reported as success | Autofocus returns stage position instead of its own result; truncated files get written, not skipped |
| C4 | Minutes of work inside a click | Focus-map phase runs inside the Start request; jog fires on mouse-up; AF blocks 120s on a hang |
| C5 | State inferred, not observed | Disk usage computed 3 ways from 1 syscall; USB monitor disabled; saved path applied once at boot |

## Phases (sequential — WPs within a phase are independent)

0. **Make failure visible** — WP-13, WP-08, WP-09. Nothing after this is verifiable without a day-long hardware run. Also disable Ashlar here (WP-16, first half) — one flag, removes a whole failure path immediately.
1. **One position model** — WP-05, WP-06, WP-07, WP-03. The revamp. Adopt Squid's region model; kill the four-way group-id guess and the global AF offset.
2. **Stop losing operator work** — WP-01, WP-02, WP-04. The complaint that opens her list.
3. **Make state legible** — WP-11, WP-15. Focus-map provenance and honesty; size vs. free space.
4. **Downstream + new capability** — WP-16, WP-17, WP-12. Hand tiles to the napari processor; fast prescan overlay; boundary-from-points.
5. **Cleanup and deferred** — WP-18, WP-14, WP-10. Dead-path deletion, and storage last (likely Docker/OS-related — see the WP).

## Index

| WP | Title | Sev | Area | Size | Phase |
|----|-------|-----|------|------|-------|
| 13 | Multi-site regression test on virtual microscope | Critical | Backend | M | 0 |
| 08 | Stitcher thread that never stops | Critical | Backend | S | 0 |
| 09 | Never leave a truncated protocol file | High | Backend | XS | 0 |
| 05 | One region model (adopt Squid's) | Critical | Backend | M | 1 |
| 06 | Autofocus must report failure, not stage position | Critical | Backend | M | 1 |
| 07 | One experiment folder, one place that makes it | High | Both | S | 1 |
| 03 | Delete cross-tab state sync | High | Frontend | XS | 1 |
| 01 | No component may reset the shared experiment | Critical | Frontend | S | 2 |
| 02 | Keep the plate map mounted | Critical | Frontend | XS | 2 |
| 04 | Jog pad — finish what Florian started | Medium | Frontend | S | 2 |
| 11 | Focus map review — honesty, provenance, UX | High | Both | L | 3 |
| 15 | Check the size estimate against the drive | Medium | Frontend | S | 3 |
| 16 | Disable Ashlar, hand tiles to napari-openuc2-processor | High | Both | M | 0+4 |
| 17 | Fast prescan overlay for tissue selection | Medium | Both | L | 4 |
| 12 | Boundary from dropped points | Medium | Frontend | M | 4 |
| 18 | Split the focus-map panel into two jobs | Medium | Frontend | L | 5 |
| 14 | Delete the dead paths | Medium | Both | S | 5 |
| 10 | Observe the drive, don't infer it | High | Both | M | 5 |

---

## Phase 0 — Make failure visible

### WP-13 — Multi-site regression test on the virtual microscope
`Critical · Backend · M`

No test runs more than one scan area — every WP below is currently unverifiable
except on real hardware overnight. `imswitch/_data/user_defaults/imcontrol_setups/example_virtual_microscope.json`
and `imswitch/imcontrol/_test/` already exist and are unused for this.

**Add:** one test — 4 sites × 3×3 tiles × 2 channels × 3 Z planes, focus map
enabled. Assert: exactly one experiment folder, one store per site, non-empty
protocol JSON, a stitched TIFF that opens, a wall-clock ceiling. Second variant:
drop a frame mid-run to cover the abort path.

**Done when:** suite runs <5 min locally and fails on current `master`. Write
this first even though it fixes nothing on its own — it's the acceptance gate
for WP-05, WP-06, WP-07, WP-08, WP-09.

### WP-08 — The stitcher thread that never stops
`Critical · Backend · S · ~-450 lines`

> "Scanning multiple sites in one run reliably fails: after the first site
> finishes, an error occurs... Failed output is just an empty JSON protocol
> file and a blank stitched image."

- `imswitch/imcontrol/model/io/ome_writers/ome_tiff_stitcher.py:71` — `stop()`
  has `is_running = False` **commented out**. The loop's only exit
  (`:122`) is having written exactly `nx*ny` images.
- One dropped frame, one aborted run, or a site with fewer tiles than the grid,
  and `stop()` blocks forever — the "Finalize OME writer" step hangs and the
  next site never starts.
- The thread is `daemon=True`, so process exit kills it inside the
  `TiffWriter` context — file never closed. That's the blank stitched image.
- More frames than `nx*ny` (any Z-stack or 2nd channel) breaks the loop early,
  discarding the rest of the queue.

**Fix:**
- Uncomment/restore `is_running = False` in `stop()`. Delete the `nx*ny` term
  from the loop condition entirely — just drain the queue and exit. Drop the
  now-unused `nx`/`ny` constructor args.
- Make the thread non-daemon so it can't be killed mid-write.
- Replace the 3 `print()` calls with the logger (diagnostics currently go to
  stdout, not the log file Franzi can send us).
- Delete `imswitch/imcontrol/model/io/stitched_tiff_writer.py` and its exports
  in `io/__init__.py` — no callers (folds into WP-14), and its `open()`
  allocates the entire mosaic canvas in RAM.

**Done when:** WP-13's dropped-frame variant completes; the stitched TIFF
opens in Fiji.

### WP-09 — Never leave a truncated protocol file
`High · Backend · XS`

> "Failed/error output is just an empty JSON protocol file."

`imswitch/imcontrol/controller/controllers/experiment_controller/experiment_mode_base.py:287`
opens the protocol file with `'w'` (truncating) and only then serializes.
Anything non-serializable in `snake_tiles`/`workflow_steps` raises mid-write;
the exception is swallowed by the caller, leaving a 0-byte file.

**This file is load-bearing downstream.** `napari-openuc2-processor`'s
`load_protocol_grid()` reads `data["snake_tiles"]` to map `iterator → (iX, iY)`;
without it the processor silently falls back to clustering raw stage
coordinates. A 0-byte protocol doesn't just lose provenance, it degrades every
downstream stitch. See WP-16.

**Fix:** `json.dumps()` first, then open-and-write. Let the exception
propagate instead of returning `None` — an unwritable protocol means the run
isn't reproducible, and that should be visible before the run starts.

**Done when:** a deliberately unserializable field aborts the start with a
clear error and leaves no file.

---

## Phase 1 — One position model

> **Read [§ Squid reference](#squid-reference) before starting any WP in this phase.**
> The revamp Bene asked for is essentially "adopt Squid's `ScanCoordinates`
> shape." Squid is not more clever than us here — it is *less* clever, and that
> is the point.

### WP-05 — One region model (adopt Squid's)
`Critical · Backend · M`

> "When multiple focus maps exist there's no clear indication of which one is
> actually active — we suspect this causes out-of-focus scans when an old map
> isn't cleared first."

**The bug, concretely.** `generate_snake_tiles`
(`imswitch/imcontrol/controller/controllers/ExperimentController.py`) writes
`centerIndex` three different ways: the area's string id (`:689`), an integer
index (`:719`, `:739`), and a literal `0` (`:770`). Acquisition resolves the
group with `centerIndex or areaName or wellId or "default"`
(`experiment_controller/experiment_normal_mode.py:677`). `0` is falsy in Python,
so the first group always falls through — and the integer form never matches
anyway, because `_run_focus_map_phase` stores maps under `areaId`
(`ExperimentController.py:3348`). The lookup misses and falls back to the
server-global `"manual"` map singleton, which outlives the experiment and the
sample. Franzi's suspicion is exactly right. `_find_reusable_manual_map`
(`:3547`) makes it worse — its last resort is literally "any fitted map."

**The structural problem behind it.** We carry per-region metadata on every
tile (`wellId`, `wellRow`, `wellColumn`, `labwareLoadName`, `conditionLabel`,
`areaName`, `areaType`, `centerIndex` — 8 keys duplicated across every FOV of
every region), and we compute traversal order in at least three places: the
frontend `calculateScanCoordinates`, `generate_snake_tiles`, and the
`isSnakescan` flag. Squid carries region metadata **once per region** and bakes
traversal order into the stored coordinate list at generation time.

**Fix — mirror Squid's three-dict model:**
- Replace `snake_tiles: List[List[Dict]]` with the Squid shape: `region_centers[region_id] = (x, y, z)`,
  `region_fov_coordinates[region_id] = [(x, y) | (x, y, z), ...]`, plus one
  `region_meta[region_id]` dict for well/labware/condition/areaType. One
  identifier, one type: `region_id` is a **string**, always.
- FOV order is baked in at generation (serpentine applied per row as the list
  is built, Squid `scan_coordinates.py:230`). Delete the downstream "convert to
  snake" step and the `isSnakescan` round-trip. Order is a property of the
  stored list, not something recomputed.
- Region order is applied once by an explicit sort (Squid `sort_coordinates()`,
  `:596`), not inferred at acquisition time.
- Replace the 4-term `or` chain at `experiment_normal_mode.py:677` with
  `m_point["region_id"]`. A missing key should raise, not guess.
- Adopt Squid's homogeneity rule for Z: a region either has Z on every FOV or
  on none (`add_region_from_fovs`, `:354`). Delete the
  `z == 0.0 → treat as None` heuristic at `experiment_normal_mode.py:654`.
- Validate before mutating: Squid's `add_region_from_fovs` raises `ValueError`
  naming the offending coordinate before touching state. We currently accept
  and fail later.
- Delete `_find_reusable_manual_map`. Reuse is a deliberate UI choice
  (`use_manual_map`), not backend inference.

**Done when:** three areas with deliberately different Z surfaces each scan at
their own surface; the log names the region id and map used per region; grep
finds exactly one place that decides traversal order.

### WP-06 — Autofocus must report failure, not the stage position
`Critical · Backend · M`

> "Autofocus is inconsistent — sometimes fast, sometimes takes ~3 minutes per
> point... After running autofocus at one point and moving to another, the Z
> position sometimes doesn't update/reset correctly."

**The wrong-Z mechanism:**
- `ExperimentController.py:1836` reads the result from
  `self.mStage.getPosition()["Z"]` instead of autofocus's own return value.
  On failure or a timed-out join, that's wherever the stage happens to sit —
  often the scan start, `center - range/2`.
- That value feeds `update_af_offset` (`:2246`) as `_experiment_af_offset`, a
  **single global scalar** added to every later capture move via
  `move_stage_z(apply_af_offset=True)`. One bad point silently defocuses the
  rest of the run.

Squid solved both halves of this and the reasoning is worth reading verbatim
(`multi_point_worker.py:1186`): AF reports soft failures **via its return
value, not by raising**, and `af_succeeded` then *gates* whether per-channel Z
offsets are applied at all (`_apply_channel_z_offset(config, af_succeeded)`).
The offset is also **per region** (`region_laser_af_offsets.get(region_id, 0.0)`),
not one global scalar. Success/failure counts are tallied and reported with
the acquisition result.

**The 3-minute mechanism:**
- `ExperimentController.py:1833` — `af_thread.join(timeout=120)`. A hung AF
  costs a flat two minutes per point.
- `AutofocusController.py:1014` — `timeoutFrameRequest = 1` second (has a
  standing `# TODO: Make dependent on exposure time`). Above 1s exposure,
  every frame times out and every step still pays the full second.
- `AutofocusController.py:1212` traverses back to the scan-range start before
  fitting, then out to the best Z — a full extra traverse per point.
- Two-stage AF divides range *and* resolution by the same factor, so the fine
  pass has the same step count as the coarse one: exactly double the frames,
  no extra precision from the pass itself.

**Fix:**
- Return a success flag alongside the Z, Squid-style. Store
  `doAutofocusBackground`'s own return value; delete the `getPosition()` line.
  Return `None`/`False` when `_lastFocusQuality["is_valid"]` is false.
- Make `_experiment_af_offset` **per region**, keyed by the `region_id` from
  WP-05. Gate per-channel Z offsets on AF success.
- Tally AF successes/failures per run and surface them in the experiment result.
- Delete the pre-fit traverse back to range start (2 lines).
- Scale the frame timeout with exposure: `max(2 * exposure_s, 1.0)`.
- Adopt Squid's `NUMBER_OF_FOVS_PER_AF` idea explicitly — AF every Nth FOV as
  one named config, replacing our `autofocus_scope`/`autofocus_period_rounds`
  pair.
- Fine pass: keep divided range, keep original resolution. If that doesn't help
  in practice, delete two-stage AF outright.
- Log per-point AF duration.

**Ask Franzi first:** is the 3-minute case correlated with a specific channel
or exposure time? If always fluorescence, the frame-timeout fix alone may be
the whole story and this WP shrinks a lot.

**Done when:** disconnecting the camera mid-scan leaves the region's Z offset
unchanged, logs a failure, counts it, and the run continues.

### WP-07 — One experiment folder, one place that makes it
`High · Both · S · ~-30 lines`

> "The system creates two folders per experiment even for a single run —
> cause unclear."

- Two creators race on the same run: `ExperimentController.py:1059-1061` and
  `create_experiment_directory` (`experiment_controller/experiment_mode_base.py:194-210`),
  the latter called from `experiment_controller/experiment_performance_mode.py:923`.
  Each stamps its own timestamp, so they never collide — they just both exist.
- Separately: Start stays clickable through the entire focus-map phase. The
  frontend only marks the run started inside the promise's `.then()`
  (`frontend/src/axon/experiment-designer/ExperimentDesigner.js:432`), and
  that promise doesn't resolve until autofocus has visited every grid point.
  The backend guard (`ExperimentController.py:982`) only checks the *workflow*,
  which hasn't started yet. Two clicks → two folders → two focus-map phases.

**Fix:**
- Delete `create_experiment_directory`; pass `dir_path` into the
  performance-mode writer setup exactly as normal mode already does.
- Set the busy flag at the top of `startWellplateExperiment`, before the
  focus-map phase; clear it in a `finally`.
- Frontend: disable Start on click, not on response.

**Done when:** WP-13 asserts exactly one folder; hammering Start during focus
mapping produces one run and one 400.

### WP-03 — Delete cross-tab state sync
`High · Frontend · XS · ~-45 lines`

> "...currently has to switch tabs or use two laptops, losing her point
> selections each time."

`frontend/src/state/store.js:236-275` listens for `persist:root` writes from
other browser tabs and replaces whole slices (`experimentState`,
`wellSelectorState`, `position`) with the other tab's copy. redux-persist
writes on nearly every state change, so two open tabs continuously overwrite
each other's point lists. Nothing in the product needs this.

**Fix:** delete `SYNC_STATE`, `syncStateAction`, `rootReducerWithSync`, and
`syncStateAcrossTabs()`. Configure the store with `persistedReducer` directly.

**Done when:** points added in one tab survive any activity in a second tab.

---

## Phase 2 — Stop losing operator work

### WP-01 — No component may reset the shared experiment
`Critical · Frontend · S · ~-120 lines`

> "Switching from the map view to live view (e.g. to adjust detector
> settings) and back resets the layout to default (or the downscaled version
> of it which I would have expected in the 'shitscope' layout — but why?!),
> wiping out placed focus map points and selected regions... Actually the
> confirmation is the enemy here."

- `frontend/src/components/ShitScopeComponent.js:121` — a mount effect that
  unconditionally overwrites `experimentState.wellLayout`, empties
  `pointList`, and inserts its own scan rectangle. Layout and point list are
  global + persisted, so the damage outlives the visit and survives a reload.
  That's the "downscaled shitscope layout," and it answers her "but why?!".
- Two "switch and clear" dialogs (`frontend/src/axon/WellSelectorComponent.js:625`,
  `frontend/src/components/LabwareSelectionPanel.jsx:686`) delete the point
  list on layout change — but points are absolute stage micrometers, so a new
  plate definition doesn't invalidate their coordinates at all. The dialog
  asks permission for a deletion that was never necessary.

**Fix:**
- Delete the ShitScope mount effect. It builds its scan request from local
  constants at Start time; it has no business writing the shared slice.
- Delete both dialogs and their pending-state machinery. A layout change
  redraws the plate and keeps the points; clear only the per-point
  `wellId`/`wellRow` annotations that genuinely belong to the old plate.
- Convention going forward: a mount effect may *read* shared experiment
  state, never *write* it. This one rule prevents the whole class of bug.

**Done when:** visiting ShitScope and returning leaves the plate map exactly
as it was, with no dialog in between.

### WP-02 — Keep the plate map mounted
`Critical · Frontend · XS`

Second half of the same complaint: pan/zoom resets and a drawn-but-not-yet-
converted region vanishes whenever she switches to adjust exposure.

- `frontend/src/axon/wellplate2/WellPlateWorkspace.jsx` renders only the
  active viewport, so switching to Camera feed unmounts
  `WellSelectorComponent`.
- The canvas keeps `scale`/`offset` (`frontend/src/axon/WellSelectorCanvas.js:56-57`)
  and the freehand polygon (`:87`) in component state. Unmounting throws all
  of it away; Redux state survives, these don't.

**Fix:** render the Plate Map viewport once and toggle visibility with the
`hidden` attribute instead of unmounting. Leave heavy viewports (Camera feed,
Overview, 3D twin) conditional so streams still start/stop on demand.

**Note:** her own aside — "I guess we did that already" re: a floating
position list — is right. The linked workspace already shows map + positions
together and PiP already floats live view. Confirm she's on that build before
building a detachable window; it wouldn't have fixed what she actually lost.

**Done when:** draw a region, switch to Camera feed, come back — polygon and
zoom level are unchanged.

### WP-04 — Jog pad: finish what Florian started
`Medium · Frontend · S`

> "The buttons for adjusting Z/focus position behave erratically — clicks
> register with a delay, appear to 'phantom-repeat,' or move the stage
> without any new input. Step size isn't adjustable either."

**Already fixed** by Florian in `aaf9b5096` (#316, hover trigger + step-size
selection) and `155249c0d` (#317, step-size persistence), both in this branch:

- Step size **is** now adjustable and persisted — `validXYStepSizes`,
  `validZStepSizes`, `localStorage` under `imswitch-stage-control-step-sizes`
  (`PositionControllerComponent.js:15-44`).
- `onMouseLeave={handleButtonUp}` plus touch handlers now cover the
  drag-off-the-button runaway.

**Still open** in the current tree — this is the whole remaining scope:

- **The delay.** `:137` still sends the step only on mouse-*up*, after the 1s
  long-press timer decides it wasn't a hold. Nothing happens while the button
  is down. Fire on `pointerdown` instead; start hold-to-travel from the same
  handler after the timer.
- **Shared press state.** All six buttons still share one `buttonPressRef`
  (`:76`). Press one while another is held and the axis is overwritten, so
  the stop targets the wrong axis. Track per axis, or refuse a second press
  while one is active.
- **Keyboard ignores the new selectors.** `keyMoveDistance = 100` (`:46`) and
  `zCoarseDistance = 500` (`:47`) are still hardcoded and used at `:241-266`.
  Wire them to `xyStepSize`/`zStepSize` — Florian's work only reached the
  on-screen buttons.
- **Double mount.** `LiveViewControlWrapper.js:113` shows the jog pad when
  toggled on *or* hovered, and that wrapper is mounted by both the PiP window
  and the Camera feed viewport. Two pads, each binding `window` keydown/keyup
  (`PositionControllerComponent.js:298`) — every arrow key moves the stage
  twice. Bind keys to the element's own focus, not `window`.

Migrating the mouse/touch handler pairs to pointer events +
`setPointerCapture` folds the runaway fix and the delay fix into one smaller
handler set. `PositionControllerComponent.test.js` already exists — extend it
rather than starting fresh.

**Done when:** one click = one step with no perceptible delay; with PiP open,
one arrow press moves once; keyboard steps honour the selectors.

---

## Phase 3 — Make state legible

### WP-11 — Focus map review: honesty, provenance, UX
`High · Both · L`

> "Unclear what 'fallback' means in the focus map UI (possibly tied to a low
> point count). Also the override current z offset may induce problems."

**Bene is right: a plane needs 3 points, and the code accepts 2.**
`imswitch/imcontrol/model/focus_map.py:306` is `if n < 4 and method != "constant": _fit_plane(...)`,
so **n=2 and n=3 both produce a "plane."** Two points do not define a plane —
there are infinitely many through them, and `lstsq` silently returns the
minimum-norm one. The code comment admits this ("2 points define a ramp...
lstsq returns the minimum-norm plane in degenerate (collinear / 2-point)
cases"). Franzi's screenshot is that exact case: `plane · MAE 0.000 · n=2`.

Three compounding dishonesties:
1. **n=2 is accepted at all.** Minimum is 3, and 3 collinear points is also
   degenerate. Squid checks both: `set_focus_map_use` refuses below 3 points
   and computes `detT` to reject collinear triples, *disabling the map* rather
   than fitting garbage (`auto_focus_controller.py:132-153`).
2. **MAE is meaningless below n=4.** `_compute_fit_stats` (`:430`) measures
   residuals against the same points it just fitted, so an exact fit is
   trivially 0. The badge reads a confident green `MAE 0.000` precisely when
   the surface is least trustworthy. Either leave-one-out or suppress it.
3. **The reason is hidden in a hover tooltip**
   (`frontend/src/axon/experiment-designer/FocusMapDimension.js:1822`).

Plus provenance and Z-offset arithmetic:
- `load_all` (`focus_map.py:789`) merges into whatever's already in memory and
  never clears; maps save to one shared folder with no sample identity.
- Four independent additive Z corrections exist: focus-map `z_offset`,
  per-channel offsets, the runtime AF offset (WP-06), and the frontend's
  override-with-current-Z. No single place shows their sum.

**Fix:**
- Require n ≥ 3 **and** a non-collinearity check before fitting a plane
  (Squid's `detT` test is 3 lines — `auto_focus_controller.py:143`). Below
  that, fall back to constant-Z and say so plainly.
- Suppress MAE/R² when `n_points < 4`; show *"not enough points to validate"*
  instead of a number that can't be wrong.
- Replace the chip with the reason inline: *"plane fit — 3 points, needs 4 for
  spline."*
- Clear existing maps before `load_all`; name saved map sets by sample.
- Fold `z_offset` into the fitted surface at fit time; delete the runtime
  addition. One number, applied once, visible in the preview.
- Show the map that will actually be used (region id, point count, fit method)
  in the Start summary.
- **UX pass on the whole panel** — split out as [WP-18](#wp-18--split-the-focus-map-panel-into-two-jobs) (Bene: "review the overall UX in the
  frontend"). `FocusMapDimension.js` is 113 KB / ~2900 lines in one file with
  four nested accordions, three point lists (manual / planned / measured) and
  a heatmap. Worth a design pass in its own right — consider whether Squid's
  "3 points, one plane, on/off" covers 90% of real use and whether spline/RBF
  should be behind an advanced toggle rather than the default.

**Done when:** a 2-point map refuses to enable and says why; Franzi can answer
"which map is this run using, and how good is it?" from the summary alone.

### WP-15 — Check the size estimate against the drive
`Medium · Frontend · S`

> "Resulting scan datasets are ~100GB, which she suspects is too large to
> stitch (hasn't tested stitching a full slide)."

The estimate is already shown (her screenshot: *Est. Size 33.3GB*) but never
compared to free space on the selected drive.

**Fix:** show the size estimate against free space on the active drive; warn
before Start when it won't fit. (Free-space numbers get trustworthy in WP-10,
but the comparison is still worth having before then.)

**Still open (measurement, not a fix):** whether a full slide can be stitched
at all. Run one through the WP-13 harness at realistic size and record peak
memory before designing anything around it.

**Done when:** starting a run that can't fit on the selected drive warns
first, naming both numbers.

---

## Phase 4 — Downstream and new capability

### WP-16 — Disable Ashlar, hand tiles to napari-openuc2-processor
`High · Both · M`

Ashlar runs in-process as a subprocess against the experiment dir
(`ExperimentController.py:2459-2607`, `_ashlar_proc`), only when
`ome_write_individual_tiffs` is on (`:1249`) — which is also the most expensive
output we write. It is a second stitching path competing with the OME writer's
own stitcher (WP-08), and it fires automatically on experiment finish
(`ExperimentDesigner.js:131-148`).

**Half A — do this in Phase 0, it's one flag.** Default
`ome_write_ashlar_stitch` to false and stop the auto-start on experiment
finish. Keep the endpoint for manual use. This removes a failure path from
every run immediately.

**Half B — make the tiles actually consumable by the plugin.** The plugin
(`C:\Users\benir\Documents\napari-openuc2-processor`, `openUC2/napari-openuc2-processor`)
vendors its engine from our `scripts/convert_experiment_tiffs.py` and expects:

- Filenames matching
  `t{YYYYmmdd_HHMMSS}_x{x}_y{y}_z{z}_c{idx}_{channel}_i{iter}_p{power}.tif`
  (`processing/engine.py:76`), under `tiles/timepoint_{n}/`.
- A `*_protocol.json` sibling with `snake_tiles[][].{iterator,iX,iY}` —
  `load_protocol_grid()` (`:144`). **This is why WP-09 matters downstream.**

Two real gaps to close:

1. **No region dimension anywhere in the plugin.** `TileInfo` (`:89`) has no
   region field, and `_is_multi_experiment_dir` (`:255`) treats experiment
   subdirectories as *timepoints*, not regions. Multi-site data collapses into
   one grid via `_cluster_to_indices`, producing a sparse or wrong mosaic —
   exactly the multi-site case Franzi runs. Add `region` to the filename (or a
   per-region `tiles/` subdir), to `TileInfo`, and to `ExperimentGrid`; group
   before assigning grid indices. Use the `region_id` from WP-05 so there is
   one identifier end to end.
2. **Grid assignment is rank-matched and silently degrades.**
   `assign_grid_indices` (`:166`) pairs JSON entries to XY positions by sorted
   iterator rank and, on any count mismatch, prints a WARNING and falls back to
   coordinate clustering. One dropped tile silently changes the mosaic. Match
   on `iterator` directly and fail loudly.

The engine is "kept in sync manually" with `scripts/convert_experiment_tiffs.py` —
whichever side changes, change both in the same PR, or make one vendor the
other properly.

**Done when:** a 4-region WP-13 run opens in napari as 4 correctly-placed
mosaics with Ashlar off.

### WP-17 — Fast prescan overlay for tissue selection
`Medium · Both · L`

Replaces the "tissue detection" item Franzi asked for, with something buildable
now: no segmentation, just get a real picture of the slide under the map so she
can *see* the tissue and select it herself.

**What it does.** A new "Prescan area" mode in the WellPlate app: pick an area,
set `dy` (line spacing) and `speed_x`. The stage runs continuously in X at high
speed (e.g. 10000 µm/s) while the camera free-runs; at the end of each line,
frames are re-aligned along X assuming equidistant spacing, heavily downsampled,
and pushed as an overlay on the plate map. Coarse in Y, motion-blurred in X —
fine. It is a locator, not data.

**Reuse, don't build:**
- The scan-and-acquire loop already exists in
  `imswitch/imcontrol/controller/controllers/LightsheetController.py:1066-1105`:
  start a non-blocking `MovementController` move, poll `getLatestFrame()` in a
  loop, and derive position from `speed × elapsed_time` rather than polling the
  stage. The comment there explains why position must be time-derived —
  `getPosition()` blocks the shared serial bus and starves live view. Do the
  same, per line, in X.
- The overlay already exists. `StageMapController` produces downscaled base64
  JPEG preview tiles in stage µm (`_makePreview`, `:247`; `_captureTile`,
  `:301`) and the plate map already draws them —
  `drawStageMapTilesOverlay` in `WellSelectorCanvas.js`, toggled by
  `stageMapState.showOnWellplate`, fetched via `apiStageMapGetTiles`. **Emit
  prescan strips into the existing StageMap tile store** rather than inventing
  a second overlay layer. One new API endpoint, no new frontend rendering path.
- Illumination/exposure stay exactly as the user left them — no channel
  switching, no auto-exposure. State that explicitly in the endpoint docstring
  so nobody "helpfully" adds it later.
- Important: It has to be in physical coordinates, so that in case we acquire with a 10x
  the overlay should work equally well as when we acquire with a 20x
  e.g. we want to aquire with low mang preview and high magnification later
  => for this we should expose a dedicated endpoint to retreive this map outside the 
  react app too

**Shape of the work:** one `startPrescan(area, dy, speed_x)` API on a
controller (StageMap is the natural home given the overlay reuse), a background
thread with the lightsheet loop per Y line, per-line re-alignment + downsample,
tiles pushed through the existing StageMap signal. Frontend: one button and two
number fields in the WellPlate toolbar.

**Open question:** whether to store prescan strips as one tile per line or
subdivide into square-ish tiles. Per-line is simpler and the existing tile
store is centre-based rectangles, so a long thin rect should already draw
correctly — verify before assuming. => I would even be fine to have this as a
a large canvas in the end, so line based is fine 

**Done when:** a prescan over one slide produces a recognisable tissue overlay
on the plate map in under a minute, and freehand/point selection can be drawn
straight on top of it.

### WP-12 — Boundary from dropped points
`Medium · Frontend · M`

> "No tissue detection, so we need to manually find tissue edges and would
> like a way to drop points and then auto-generate a boundary around them...
> We want an 'add current position' button to speed up marking points."

Freehand is drag-only — every mouse-down starts a fresh polygon
(`frontend/src/axon/WellSelectorCanvas.js:1385-1389`), discarding the last
one. There's no way to build a boundary a vertex at a time, hence her
workaround of placing reference points and tracing them by hand.

**Fix:**
- In freehand mode, a click *appends* a vertex instead of restarting the
  polygon. Drag still works exactly as now.
- "Add current" appends the current stage XY as a vertex.
- A "wrap points" action that builds the polygon from the existing point
  list (monotone-chain hull, ~15 lines inline — no new helper module).
- Nothing else needed: `generateFreehandScanPositions` (`:157`) already
  fills a closed polygon with FOV-spaced positions.

Pairs naturally with WP-17: prescan gives her something to trace, this gives
her the tracing tool. Squid's equivalent is `get_points_for_manual_region`
(`scan_coordinates.py:415`) — worth a look for the polygon→FOV fill, which
handles the corner cases we don't.

**Done when:** four stage positions, four taps, one Convert — region is tiled.

---

## Phase 5 — Cleanup and deferred

### WP-18 — Split the focus-map panel into two jobs
`Medium · Frontend · L · net deletion expected`

The honesty fixes landed in WP-11, but the panel they live in is
`frontend/src/axon/experiment-designer/FocusMapDimension.js`: **2470 lines,
112 KB, 18 `<Accordion>` blocks, 13 API clients and ~25 Redux setters in one
component.** Nothing here is broken; it is unreadable, which is why WP-11 had to
stop at the parts that could be verified in isolation.

**The structural finding: there are two products in one panel.**

| | Automatic | Manual |
|---|---|---|
| Input | `rows` x `cols` grid over each region's bounds | points the user drops on the plate map |
| Preview list | "Planned Grid Points" + `plannedRemovedKeys` pruning | "Manual Focus Points" |
| Fit trigger | `computeFocusMap` | `measureFocusMapFromPoints` / `computeFocusMapFromPoints` |
| Gate | — | `use_manual_map` |

Both then produce the same third list ("Measured Points"). The two modes share
no controls but are interleaved in one scroll, so at any moment roughly half the
panel is inert and the operator cannot tell which half.

**The duplication: autofocus is configured twice.** `focusMapConfig.af_*` has 12
fields (`af_range`, `af_resolution`, `af_algorithm`, `af_two_stage`, ...) and
`parameterValue.autoFocus*` has 19 setters for the same physical settings, each
with its own UI — this panel's "Autofocus Settings (shared with Z/Focus tab)"
accordion, and `ZFocusDimension.js`. They are not shared; they are two copies
that happen to be adjacent. This is C2 in its purest form. Which one reaches the
hardware depends on whether the run goes through the focus-map phase or the
per-point autofocus step.

**Fix, in order — each step is independently shippable:**

1. **Delete one of the two autofocus configs.** The focus-map phase should use
   the same `parameterValue.autoFocus*` settings as everything else; a focus map
   is autofocus at N positions, not a different instrument. Drop the `af_*`
   block from `FocusMapConfig` and its accordion. Biggest single deletion, and
   it removes a real "which setting won?" ambiguity.
2. **Pick the mode explicitly.** One `ToggleButtonGroup` — *Grid* / *Points* —
   at the top, rendering only that mode's controls. `use_manual_map` becomes the
   toggle's value rather than a switch buried in Advanced.
3. **One point list, not three.** Planned / manual / measured are the same
   table with a `state` column (`planned` / `placed` / `measured` / `rejected`).
   The Go-To, delete and refit actions are already written three times.
4. **Extract the pieces that are already components in all but name:** the fit
   result rows (`:1783-1960`), the measured-point table (`:1961-2393`) and the
   heatmap (`:2394+`). Three files, no new abstraction — this is moving code,
   not designing it.
5. Only then decide whether spline/RBF belong in front of the operator at all.
   With WP-11's rules in place the honest options are: <3 points = flat,
   3 = tilted plane, 4+ = curved. That is a sentence, not a dropdown, and the
   method selector can stay in Advanced where WP-11 already put it.

**Explicitly not in scope:** changing what the backend computes. WP-11 settled
the maths; this is about which of it the operator is asked to think about.

**Done when:** the panel fits on one screen in each mode, no setting appears
twice in the app, and `FocusMapDimension.js` is under ~600 lines with the rest
in named sibling components.

### WP-14 — Delete the dead paths
`Medium · Both · S · ~-500 lines`

Each of these is a second implementation of something that already works
elsewhere. Not causing outages on its own — but it's what makes the outages
above hard to find, and each is a standing invitation to fix the wrong copy.

**Remove:**
- `setup_shared_ome_writers` (`experiment_controller/experiment_normal_mode.py:330`) — no callers.
- `stitched_tiff_writer.py` in full + its `io/__init__.py` exports (folds
  into WP-08).
- The hardcoded layout ladder in
  `WellSelectorComponent.applyLayoutChange` (`:232-287`), including the
  branch that logs `TODO impl me` at the operator, and the two test-fixture
  entries at `:48`. The labware panel already loads real definitions from
  the backend — keep that one path.
- The double finalization: per-tile writers close at
  `experiment_normal_mode.py:1072`, then all close again at `:1117`.
  `finalize_current_ome_writer` (`ExperimentController.py:3021`) also
  indexes the writer list by timepoint although it's indexed by tile — its
  own docstring calls the name misleading. Keep one path.

**Done when:** WP-13 still passes and the diff is entirely deletions.

### WP-10 — Observe the drive, don't infer it
`High · Both · M` — **deliberately last**

> "Disk usage for connected external drives (e.g. the 4TB SSD) displays
> incorrectly... The drive sometimes doesn't appear as a save target after
> being disconnected and reconnected — Franzi has to restart the application,
> which risks accidentally saving to internal storage instead."

**Deferred because the environment is probably the bigger half.** We run
ImSwitch in Docker, so before changing application code, establish:

- Is the mount propagated into the container as `shared`/`rslave`? A bind
  mount created on the host *after* container start is invisible inside a
  `private`-propagation container — which alone explains "needs a restart to
  see the drive," with no application bug at all.
- Does the container see the device re-appear at the same path after replug,
  or does udev give it a new mountpoint?
- Does `shutil.disk_usage` inside the container report the host filesystem or
  the overlay?

Answer those three first with `findmnt -o TARGET,PROPAGATION` and a manual
replug; the code below may turn out to be a small part of the fix.

**Application-side findings (still true regardless):**
- One syscall, three formulas:
  `imswitch/imcommon/model/storage_scanner.py:110` returns `free`/`total`;
  `imswitch/imcontrol/controller/controllers/StorageController.py:200`
  derives `used = total - free`; `:267` computes a percentage a third way.
- `shutil.disk_usage`'s `free` is blocks available to the user, so
  `total - free` overstates usage by the ext4 root reserve (~5% ≈ 200GB
  phantom data on a 4TB drive).
- An *unmounted* mount point is just a directory on the root filesystem, so
  the scan reports the underlying disk's capacity under the SSD's name.
- `StorageController.py:73` — `if False:` disables USB monitoring outright.
- `_load_persisted_path` (`:76`) runs once at startup. If the SSD isn't
  mounted at boot, the saved path is dropped and never re-applied.

**Fix (after the Docker questions are answered):** one `shutil.disk_usage()`
call, one place, delete both derived formulas; skip candidates where
`os.path.ismount()` is false; re-apply the persisted data path on the existing
5s status tick when it becomes valid again; frontend banner whenever the active
save path isn't the configured one.

**Done when:** the Docker mount propagation question is answered in writing,
and unplug/replug returns the SSD as save target within one tick with figures
matching `df -h` **on the host**.

---

<a name="squid-reference"></a>
## Squid reference

Read before WP-05, WP-06, WP-11. Source: `C:\Users\benir\Documents\Squid\software`.

### Position model — `control/core/scan_coordinates.py`

Three parallel dicts, one string key:

```python
self.region_centers = {}         # {region_id: [x, y, z]}
self.region_shapes = {}          # {region_id: "Square"}
self.region_fov_coordinates = {} # {region_id: [(x,y[,z]), ...]}
```

What we should copy:

- **One identifier, one type.** `region_id` is always a string. Well IDs
  ("A1", "AF48") are parsed with an explicit regex that returns `None` for
  non-well names (`_parse_well_key`, `:110`) — no falsy-or chains, and `0` is
  never a valid id, so our `centerIndex or ...` bug cannot occur.
- **Order is baked in, not recomputed.** Serpentine within a region is applied
  as the row is built (`if fov_pattern == "S-Pattern" and i % 2 == 1: row.reverse()`,
  `:230`). Order across regions is applied once by `sort_coordinates()`
  (`:596`), which reorders the dicts themselves. The worker just iterates.
  Compare: we compute order in the frontend, again in `generate_snake_tiles`,
  and carry an `isSnakescan` flag.
- **Region metadata lives once per region**, not copied onto every FOV.
- **Validate before mutating.** `add_region_from_fovs` (`:354`) checks every
  coordinate against software limits and raises `ValueError` naming the bad
  one *before* touching state.
- **Homogeneous Z per region.** `has_z = all(len(fov) > 2 for fov in fovs)` —
  "a region must be homogeneous." No per-tile `z == 0.0 → maybe None` guessing.
- **Explicit update events.** `AddScanCoordinateRegion` /
  `RemovedScanCoordinateRegion` / `ClearedScanCoordinates` dataclasses fire a
  callback so the viewer and focus map stay in sync — one channel, typed.
- **Manual regions keep drawing order** (`_is_manual_region`, `:579`) and sort
  before wells; serpentine is applied only to wells. Deliberate, documented.

Also worth reading: `get_points_for_manual_region` (`:415`) for polygon→FOV
fill (relevant to WP-12), and `get_scan_bounds` (`:679`).

### Autofocus reporting — `control/core/multi_point_worker.py`

- `af_succeeded = self.perform_autofocus(region_id, fov)` (`:1079`) — AF
  returns **success**, not a position.
- The comment at `:1186` is the invariant we're missing, verbatim: AF "reports
  soft failures ... via its return value, NOT by raising — both paths must mark
  the FOV's AF as failed or the per-channel z-offset gate would apply offsets
  from an unanchored z."
- `_apply_channel_z_offset(config, af_succeeded)` (`:1111`) — offsets are
  **gated on AF success**.
- `region_laser_af_offsets.get(region_id, 0.0)` (`:1192`) — AF offset is **per
  region**, not one global scalar.
- `_laser_af_successes` / `_laser_af_failures` are tallied and reported.
- `Acquisition.NUMBER_OF_FOVS_PER_AF` — AF cadence as one named config.

### Focus map — `control/core/auto_focus_controller.py`

Radically simpler than ours, and stricter:

- Exactly **3 points**, one plane, barycentric interpolation
  (`control/utils.py:167`, `interpolate_plane`).
- `set_focus_map_use` (`:132`) **refuses to enable** below 3 points, and
  computes `detT = (y2-y3)(x1-x3) + (x3-x2)(y1-y3)`; if `detT == 0` the points
  are collinear and the map is disabled with an error. It never fits garbage.
- `gen_focus_map` (`:158`) drives to each of 3 coordinates, autofocuses, stores
  the measured `(x, y, z)`.
- `add_current_coords_to_focus_map` (`:190`) re-checks collinearity before
  accepting a third point and raises with an actionable message.

The question for WP-11 is not "how do we make our spline honest" but "does
3-points-and-a-plane cover 90% of real slides, with spline behind an advanced
toggle."