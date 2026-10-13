# Aufgaben: a JOBS-style workflow builder for ImSwitch — analysis, decisions, roadmap

*Status 2026-10-13, branch `feat/AufgabenController`. Compares openUC2/ImSwitch with Nikon
NIS-Elements **JOBS** (NIS-Elements AR online help) and plans a modular, user-composable
acquisition-workflow system that reuses the ExperimentController primitives and also drives pumps,
valves and general-purpose I/O. Phases 0–2 are done (§5). Phases 3–6 are proposals.*

Names used throughout:

| Name | Meaning |
|---|---|
| **Aufgaben** | The feature: the engine, `AufgabenController`, and the frontend app |
| **Job** | One document: a tree of nodes, plus parameters, variables, resources and storage settings |
| **Node** | One entry in the tree. Its `task` is either a control node (loop, condition, capture, flow step) or a hardware task |
| **Task** | A plain Python function registered with `@task`, written against the `Microscope` protocol |

---

## 1. Summary

**What JOBS is.** A tree of tasks that runs top to bottom (§2). Loops and conditions nest other
tasks. Definition tasks (capture definition, z-stack, plate) are declared once and referenced by
execution tasks. Named data objects flow between tasks: point sets, well selections, regions, focus
surfaces and variables. Expressions and analysis results drive conditions. Operators get a wizard;
specialists get the editor.

**What ImSwitch had.** Four overlapping engines and no composable tree (§3). The flat step engine
in `WorkflowManager.py` has a good per-step contract but no loops, variables or conditions, and its
pause did nothing. ExperimentController builds one fixed nested loop. The same tile, Z-stack and
timelapse loops exist 7, 5 and 9 times across controllers. A Blockly prototype in the frontend did
not match the backend. Pumps were a stepper axis, valves did not exist, and the GPIO endpoints are
untyped.

**What Phase 0 delivered.**

- **Fixes** to ExperimentController, the step engine, MDA, writers and the virtual microscope (§7):
  working pause and resume, interruptible waits, the per-point autofocus range, performance-mode
  Z/channel storage, multi-region output paths, single-position routing, and more. ExperimentController
  output stays readable by the napari openUC2 processor (verified on the virtual microscope).
- **Removed** Hypha (backend, setup files, packaging extra) and the Blockly app (frontend, three npm
  packages, its Redux slice and five dead API wrappers).
- **Decisions** D1–D10 (§4.1): the name, the document format, running beside ExperimentController,
  OME output, host independence, and the task convention that Arkitekt accepts unchanged.
- **Scaffolding**: the host-independent package `imswitch/imcontrol/model/aufgaben/` with the job
  document, 23 control nodes, 19 hardware tasks, the safe expression language, the validator, the
  useq bridge, three example jobs and the JSON Schema `docs/aufgaben/job.schema.json`. 62 unit tests.
- **Virtual microscope**: a simulated ESP32 (`VirtualESP32Manager`) with heater, digital I/O, I2C
  sensors, fan and bus power; virtual homing over HTTP works; `TemperatureController` gained a read
  endpoint.

**What Phase 1 delivered.**

- **Engine** (`engine.py`): runs a job tree on any `Microscope` with all 23 control nodes,
  useq-indexed frames, pause, resume, finish and abort, error policies, `every`, `enabled`, overrun
  policies, the idle branch and operator questions. A simulated host drives its tests without
  ImSwitch.
- **One OME writer** (`storage.py`) for Aufgaben, ExperimentController and the recording app.
  Every output validates with `ome-zarr-models` (NGFF 0.4 image and HCS plate) and `ome-types`
  (OME-TIFF). ExperimentController keeps its file names, per-tile TIFFs and protocol file, so the
  napari openUC2 processor still reads its runs.
- **ImSwitch side**: a host adapter over the ExperimentController verbs, `AufgabenController` with
  17 endpoints and one Socket.IO signal, and a shared busy check so Aufgaben, ExperimentController,
  Arkitekt and the firmware updater never drive the instrument at the same time.
- **Removed**: the Timelapse and HistoScan apps (backend, frontend, the `ol` npm package), about 25
  dead Qt-era controllers, all `self._widget` code in the remaining controllers, and 23 unused
  signals. Phase 1 removes about 8,900 lines and adds about 2,600.
- **Virtual microscope**: exposure scales the image, `getParameter` works, blocking stage moves
  take distance divided by speed.

**What Phase 2 delivered.**

- **Focus maps and regions**: `create_focus_map` autofocuses at points, region grids or wells and
  fits one surface per region or well, with the same `FocusMap` ExperimentController uses (now in
  the host-independent package). Position, well, region and tile loops follow a surface
  (`focus_map`); `move_to_focus_surface` moves Z onto it. `loop_regions` visits regions, and
  `loop_grid` without a plan covers the current region with tiles.
- **Capture depth**: ring and DPC channels drive the LED matrix through a typed pattern;
  `keep_light_on_between_channels` switches each light once; every frame records its exposure and
  the camera bit depth.
- **Well plates**: ImSwitch labware registers as useq plates. Both use the same convention (rows
  towards −Y, columns towards +X), checked well by well for every regular plate.
  `getAufgabenWellPlate` turns a labware and its stage offset into a plan.
- **Scans and stitching**: `fast_stage_scan` wraps ExperimentController's firmware-timed scan;
  `stage_prescan` keeps the stage map's prescan overview with the run; `stitched_preview` writes one
  stitched image per position, at stage positions or registered with Ashlar.
- **napari openUC2 processor**: reads OME-Zarr runs through the same processors as TIFF tiles.
- **useq equivalence**: an imported sequence acquires exactly the events useq lists, in four axis
  orders, expanded into loops or kept whole.
- **Requested clean-ups**: ome-zarr-models and ome-types are runtime dependencies; every NGFF
  attribute block is checked against ome-zarr-models when it is written; all OME-XML comes from one
  ome-types builder; the other zarr writers were replaced or removed; the recording service's
  streaming method works; the stage map and focus-lock calibration take part in the busy check;
  `imswitch/_data/openapi.json` is regenerated.
- **Bugs found on the way** (§7.8), among them an OME-TIFF export that failed for any frame wider
  than 512 px and a virtual microscope whose camera was mirrored against its stage.

**Next.** Phase 3 (the frontend app) and Phase 4 (peripherals) can start; Phase 5 adds the analysis
tasks such as `regions_from_overview`. About 11–15 developer-weeks remain to a complete v1.

---

## 2. The reference: what Nikon JOBS actually is

Sources: NIS-Elements AR help pages `jobs.overview`, `jobs.creating`, `jobs.basic.rules`,
`jobs_introduction_to_expressions`, `jobs_introduction_to_conditions`, `hca.viewresults` and the
per-category task pages (links in Appendix B), plus the UCSD NIC service page.

### 2.1 Concepts

| Concept | How JOBS does it | Why it matters for ImSwitch |
|---|---|---|
| **Task** | One icon with a parameter dialog; double-click to edit. Tasks run top to bottom. | The unit of the palette. Needs a schema per task. |
| **Container tasks** | Loops and conditions have an inner block that runs repeatedly or conditionally. Loops nest. Swapping `Time Lapse` and `Loop over Points` changes the experiment. | The engine must walk a tree, not a list. |
| **Definition vs execution** | `Capture Definition` ↑ `Capture`; `Define Z-Stack` ↑ `Z-Stack Loop`; `Define Plate` → `Select Wells` → `Loop over Wells`; `Autofocus Settings` ↑ `Autofocus`. A definition must sit above its executor; the editor marks violations with a red "!" and refuses to run. | Resources are declared once and referenced by name. Static validation before run. |
| **Loops do not move the stage** | `Loop over Wells` only iterates; `Move to Well Center` inside the loop moves. Same for points and regions. | Keeps every hardware action explicit and visible. |
| **Resources** | Point sets (`Generate Points`, `Predefined Points`, `New Point Set`, add/remove/append/reorder/offset), well selections (empty/full/add/remove/copy), region lists (`Draw Regions`, `Region List` filled by analysis), focus surfaces, saved positions (`Get Current XY Position` → `Move to XY Position`), labels and classes. | These are the "variables" most users actually need. ImSwitch already has `Point`, `ScanArea`, labware and `FocusMap` models for them. |
| **Time Lapse** | Duration by time / count / unknown; `Run Loop Every` (period) vs `Wait Between Loops` (gap); `Run Tasks while Waiting` (a second block that executes in the idle time); `Finish when condition is TRUE`; `Redefine Time Lapse` from inside a condition; split storage per time point. | Period scheduling with an explicit overrun policy and an idle-time block. |
| **Control flow** | `Repeat N times`, `Python Loop` (`loop_init/condition/step`), `Condition (If)`, `Condition (If-Else)`, `Every nth`, `Break` (choose which parent loop), `Continue`, `Finish` (keep data), `Abort` (discard), `Exit with Error`, `Block` (collapse/disable), `Phase` (reorderable job parts). | The minimal control-flow set a v1 needs. |
| **Variables and expressions** | `System > Variables` declares; `System > Expression` assigns (`TotalCells = TotalCells + Job.CellCount.Count`); `Logical Expression`; analysis results addressed as `Job.<TaskName>.<Field>`. Expressions can also set device parameters. | A small safe expression language over a namespaced context. |
| **Job parameters and wizard** | Parameters (On/Off, list, numeric) shown in the Wizard and/or at Run Time; tasks gated with `Requires parameter` expressions (`Capture==Z-Stack AND Analysis==Cell Count`); `Put in Wizard` exposes selected tasks to operators; mini-jobs on wizard events. | Separates the specialist who designs from the operator who runs. |
| **Human in the loop** | `Question` (custom buttons, pauses), `Debug` (show values, pause), `Draw Regions` / `Edit Regions` / `Add/Edit Points Manually` / `Adjust Camera Settings Manually in OC` (pause, user edits, continue), `Custom Metadata`. | Needs a blocking "wait for answer" node and a UI dialog. |
| **Feedback** | Analysis tasks (`Intensity Analysis`, `Cell Count Analysis`, `General Analysis`, `Out Proc` = run an external program and read back values/points/masks, `Find Well by Analysis`), `Use ROI from Analysis`, `Region List` filled by GA3. Conditions test the results. The brochure example: multipoint → AI detects target → stimulate or move on. | The "smart microscopy" story. Comes last; needs stage ↔ pixel affine. |
| **Device control** | `Temperature and Gas Control`, `Analog Out` (NIDAQ), `Wait for I/O Event` (NIDAQ input), `Replenish Water`, `Open/Close Shutter`, `Select Objective`, `Set XY Speed/Accuracy`, `Escape Z`, `Move to Loading Position`. Everything else is an *Optical Configuration* (one preset sets many devices). | Thin. ImSwitch can do better here (ESP32 GPIO, pumps, CAN). |
| **Escape hatches** | `Macro`, `Python Script` (`run()` with images, parameters, devices), `Execute Python After Run`, `Send E-mail/SMS Notification`. | `DebugController.runCode` is the existing analogue. |
| **Storage** | ND2; images from matching loops merged into one file; `Split Storage per well/time point/point/region`; `Save Image to Database: Always/Never/Later` (capture for analysis without saving); `Alternative Storage Location`; metadata and results in a job database with job → run → well/point → image links; full history. | OME-Zarr/OME-TIFF via `OMEWriter`, plus a run record. A real job DB (SQLite) is a later phase. |
| **Runtime** | Progress window (wells go green, current yellow; analysis results appear live), `Pause/Continue`, `Finish`, `Abort`; runtime parameters asked before start; results window with heatmaps, grid, graphs; export to TIFF. | Hierarchical progress and two kinds of stop. |

### 2.2 Task catalogue (condensed)

17 categories, roughly 150 tasks. Counts are from the help pages; the rightmost column names what in
ImSwitch already covers the category.

| Category (tasks) | Representative tasks | ImSwitch today |
|---|---|---|
| Well Plates (29) | Define Plate, Select Wells, Loop over Wells, Move to Well / Well Center / Previous-Next, Label Well Plate, Align Well Plate Holder, Detect Plate Automatically, Plate Loader, Loop over Plates, Find Well by Analysis | `labware_api.py` (`getLabwareList`, `selectWellsByPattern`, `applyWellSelectionToExperiment`), `WellSelectorCanvas`; no loader, no labels |
| Sample Holder (9) | Define Slide, Define Stage Area, Align Slide Holder, Slide Loader, Loop over Slides | `overview_api.py` slide slots + `registerOverviewSlide`; `ScanArea` |
| Time Series (6) | Time Lapse, Redefine Time Lapse, Time Sequence, Repeat N times, Keep Object In View, Python Loop | `wait_for_next_timepoint`, `FOVLockController` (drift) |
| Stage XY Points (22) | Generate Points, Predefined Points, Loop over Points, Move to Point, Offset Point Set, Add/Remove Point, Reorder Point Set, Get/Move to XY Position, Set XY Speed/Accuracy, Move to Loading/Acquisition Position | `Point`, `pointList`, `CoordinateCalculator.js`, `movePositioner`, `moveToSampleLoadingPosition`, `setPositionerSpeed` |
| Large Images (16) | Scan Large Image (+ in Slide/Well/Area / in Region / by Continuous Movement), Draw/Edit Regions, Region List, Loop over Regions, Move to Region Center, Store Region Overview | `scanAreas`, `StageMapController` prescan + strobe sweep, Ashlar stitching, `snapOverviewImage` |
| Z-Stack (10) | Define Z-Stack (top/bottom, symmetric, asymmetric, by expression), Z-Stack Loop, Assign Z-Stack to Point/Well, Move to Z-Stack home, Set Z Speed, Capture Continuous Z Stack | `zStack*`, `zStackMode`, `zStackOffsets`, `scan_plan.z_offsets` |
| ROIs (3) | Define ROI, Use ROI from Analysis, Assign ROI to Point/Well | — (galvo stimulation ROIs exist in `GalvoScannerController`) |
| Autofocus + Focus Surface (19) | Autofocus Settings, Autofocus (short/long/SmartRange/AI), Create / Offset / Move to Focus Surface, Generate Focus Points, Assign Current Z to Point, Get/Move to Z Position, Escape Z, Refocus Z | `autofocus_api.py`, `focus_map_api.py`, `FocusMapConfig`, `AutofocusController` |
| PFS (hardware focus lock) | Enable/Disable PFS, PFS offset | `FocusLockController` |
| Acquisition (24) | Capture Definition, Capture, Capture (current), Fast Timelapse, Triggered Experiment, Illumination Sequence, Live Window, Merge and Store Captured Images, Save Captured Image, Detect Background, Loop over Components, Adjust Camera ROI, Assign/Load Capture Definition | `acquire_frame`, `set_exposure_time_gain`, `save_frame_ome`, performance mode, `LiveViewController`, `setDetectorROI` |
| Stimulation (2) | Sequential Stimulation, Simultaneous Stimulation | `GalvoScannerController`, `LEDMatrix` patterns |
| Optical Configurations (14) | Select Channels, Auto Brightness, Autosignal.ai, Set Exposure to OC, Open/Close Shutter, Select Objective, Assign Camera Settings to Point/Well | `set_laser_power`, `set_led_matrix_pattern`, `setDetectorMode(isAuto)`, `moveToObjective`; no preset object |
| Conditions (6) | Condition (If), Condition (If-Else), Every nth, Break, Continue, Logical Expression | — |
| System (33) | Variables, Expression, Wait, Question, Debug, Macro, Python Script, Custom Metadata, Send E-mail/SMS, Create/Assign Label, Set Class, Phase, Block, Finish, Abort, Exit with Error, Storage, Alternative Storage Location, Parameters (Integer/Double/String/Text/Path), Backup After Run | `wait_time`, `DebugController.runCode`, `MetadataController` |
| Analysis (8) | Define Analysis, General Analysis, Cell Count Analysis, Intensity Analysis, General Analysis RGB, Out Proc, Wound Healing, Live/Dead | `AutofocusController` focus metrics; `OpticalFlowController`; microEye / STORM recon |
| GA3 Processing | GA3 recipes during/after acquisition | — |
| Device Control (4) | Temperature and Gas Control, Analog Out, Replenish Water, Wait for I/O Event | `TemperatureController.set_temperature`, `UC2ConfigController` GPIO/trigger/fan |

### 2.3 Design rules worth copying verbatim

1. Top-to-bottom execution; containers nest; order is the program.
2. A definition task above its execution task; static validation with a visible error marker;
   the job cannot run until the tree is valid.
3. Loops iterate, explicit tasks move. No hidden stage motion.
4. Data objects (points, wells, regions, surfaces, saved positions) are named resources, created by
   tasks and consumed by tasks.
5. `Time Lapse` has both a period and a gap mode, an "unknown duration" mode with a stop
   condition, and an idle-time block.
6. `Break` names the loop it leaves. `Finish` keeps data, `Abort` discards it.
7. Capture for analysis without saving (`Save Image to Database: Never/Later`).
8. Split storage per loop level is a storage option on the loop, not a separate task.
9. Parameters have a "Wizard" and a "Run Time" visibility, and tasks can be gated on them.
10. Human-in-the-loop tasks pause the job; the operator confirms with "Next".
11. A `Python Script` escape hatch exists but is a task like any other, with a declared entry point.

---

## 3. What ImSwitch has today

*As audited on 2026-10-12, before Phase 0. Statements marked **[Phase 0]** have changed since;
§7 has the status of every bug. Phase 1 removed several controllers cited here, among them
Timelapse, ROIScan, SquidStageScan and HistoScan (§7.7).*

Paths are relative to the repository root. `EC/` = `imswitch/imcontrol/controller/controllers/experiment_controller/`.

### 3.1 The step engine (`imswitch/imcontrol/model/managers/WorkflowManager.py`)

```python
class WorkflowStep:                       # lines 73-183
    def __init__(self, name, main_func, main_params, step_id,
                 pre_funcs=None, pre_params=None, post_funcs=None, post_params=None)
    # main_func(**main_params); pre/post: f(**params, context=ctx, metadata=md)
    # metadata = {"step_id", **main_params, "result", "pre_result", "post_result", timings}
    # max_retries popped from main_params; on exhaustion: context.should_stop = True

class Workflow:                           # lines 185-220
    def run(self, context):               # for i in range(current_step_index, len(steps)) ...
    def run_in_background(self, context)  # plain threading.Thread, non-daemon

class WorkflowsManager:                   # lines 228-292
    start_workflow / pause_workflow / resume_workflow / stop_workflow / get_status
```

What it gives: a stable *step contract* (`main_func` + pre/post functions sharing a `metadata`
dict, a `WorkflowContext` with `objects`, `data`, `on("progress")`), cooperative stop between
functions, per-step retries, and a progress event stream (`started` / `retrying` / `failed` /
`completed` with timings).

What it lacks or gets wrong:

| Issue | Where |
|---|---|
| No loops, conditions, variables or branching at runtime; the list is fully unrolled before start (normal mode reports `total_step_number` in the thousands) | `Workflow.run` 190-208 |
| `should_pause` is set (71) and read only by `get_status` (290); the run loop never checks it. **Pause is a no-op.** **[Phase 0: fixed]** | 190-197, 290 |
| `resume_workflow` calls `context.request_resume()` which does not exist. **Resume raises `AttributeError`.** **[Phase 0: fixed]** | 263 |
| `store_step_result` drops `result`/`pre_result`/`post_result`; a later step cannot read an earlier step's output | 44-50 |
| Pre/post exceptions are not caught in the step; they abort the whole workflow via the broad `except` in `Workflow.run` | 201-208 |
| `force_stop_*` in every controller does `del self.workflow_manager; self.workflow_manager = WorkflowsManager()`; the old thread keeps running until its next stop check | `WorkflowController.py:426`, `ExperimentController.py:562`, `TimelapseController.py:368` |
| `WorkflowsManager.instance()` exists but every controller instantiates its own (`ExperimentController.py:48`, `WorkflowController.py:184`, `TimelapseController.py:41`); native MDA bypasses it entirely. **No global run lock.** | 228-240 |
| `step_id` is a string on `started` and an int index on `completed` | 110, 174-182 |

### 3.2 The JSON workflow API (`controllers/WorkflowController.py`)

```python
class WorkflowStepDefinition(BaseModel):   # 109
    id: str; stepName: str; mainFuncName: str
    mainParams: Dict[str, Any]; preFuncs: List[str]; preParams: Dict[str, Any]
    postFuncs: List[str]; postParams: Dict[str, Any]
```

`uploadWorkflow(definition)` (444) stores into an in-memory class dict; `start_workflow_api`
(450-496) resolves `mainFuncName` through `function_registry` (194-211, 16 names) and starts the
workflow with a `tifffile.TiffWriter("timelapse.tif")` relative to the CWD. Unknown pre/post names
are silently dropped (464-465). Of the 16 registered names only `move_stage_xy`, `set_laser_power`,
`acquire_frame` (stale `getLatestFrame`) and `dummy_main_func` are real main functions; `autofocus`,
`process_data`, `save_frame`, `save_data`, `addFrametoFile`, `set_exposure_time_gain` are stubs;
`wait_time`, `append_data`, `close_zarr` need `context`/`metadata` and so fail as main functions;
`close_tiff_writer` references an attribute that is never set. `getWorkflowStatus` (436) reads
`self.status`, which is undefined. `startWorkflowTileBasedByParameters` (373) computes and does
nothing. Both histo endpoints (503, 659) crash (`open_ome_zarr` returns `None`; `Workflow(...,
workflow=...)` is an invalid kwarg). No list/delete/persist for stored definitions.

### 3.3 ExperimentController and its package

#### 3.3.1 Models (`EC/models.py`)

- `Point` (84): `id, name, x, y, z?, iX, iY, neighborPointList, wellId, areaType, wellRow,
  wellColumn, labwareLoadName, conditionLabel` — already carries well and label metadata.
- `ScanArea` (129): `areaId, areaName, areaType, centerPosition, bounds, scanPattern, positions`.
- `SyntheticChannel` (162): LED-matrix ring / DPC channels.
- `ParameterValue` (193-386): the one-struct-for-everything (illumination, z-stack, timelapse,
  autofocus incl. `autoFocusEveryNFovs`, `autoFocusPeriodRounds`, camera, speeds, performance
  mode, OME writer flags, illumination timing). `autofocus_kwargs()` (357) is the only translation
  to the `autofocus()` signature.
- `FocusMapConfig` (389-451), `Experiment` (454-476: `name, parameterValue, pointList, scanAreas,
  scanMetadata, focusMap, timepoints`; `timepoints` is unused — the count comes from
  `numberOfImages`), `MDASequenceRequest` (489), `ExperimentWorkflowParams` (510).

#### 3.3.2 Pure planning (`EC/scan_plan.py`)

`resolve_channels` (31), `z_offsets` (85), `writer_flags` (94), `build_region_meta` (109),
`region_fovs` (123), `regions_to_areas` (158), `build_scan_regions` (178),
`stage_scan_frame_table` (299), `stage_scan_frame_count` (335). Hardware-free, unit-tested
(`_test/unit/test_scan_plan.py`, `test_scan_regions.py`). Tile grids and snake order are **not**
here; they are computed in the frontend (`frontend/src/axon/CoordinateCalculator.js:361/439/483`).

#### 3.3.3 The reusable primitives ("verbs")

These are the functions an Aufgaben engine should wrap unchanged. All are methods on the controller
(mixins), act through `self._master.*Manager`, and already honour the pre/main/post contract.

| Verb | Location | Signature (abbrev.) | Notes |
|---|---|---|---|
| `move_stage_xy` | `EC/hardware_api.py:516` | `(posX, posY, relative=False)` | blocking; combined "XY" move when supported |
| `move_stage_z` | `hardware_api.py:537` | `(posZ, relative=False, maxSpeedZ=5000, af_region_id=None)` | adds per-region AF offset from controller state |
| `home_axis` / `homeAllAxes` | `hardware_api.py:499/506` | | **ask-first** class |
| `return_to_initial_position` | `hardware_api.py:563` | `(include_z_position=False)` | |
| `set_laser_power` | `hardware_api.py:421` | `(power, channel)` | enables source; 40 ms sleep |
| `set_led_matrix_pattern` | `hardware_api.py:434` | `(kind, direction, radius, intensity_r/g/b, settle_s)` | ring / halves / off |
| `_switch_off_all_illumination` | `hardware_api.py:580` | | |
| `set_exposure_time_gain` | `hardware_api.py:391` | `(exposure_time, gain, context, metadata)` | pre-function shape |
| `set_detector_parameter` | `hardware_api.py:551` | `(parameter, value)` | |
| `_beginTriggeredAcquisition` / `_end…` | `hardware_api.py:224/252` | | software-trigger bracket |
| `grabCameraFrame` | `hardware_api.py:268` | `(frameSync=2, returnFrameNumber=False)` | `snapSync` fast path, frame-number polling fallback |
| `acquire_frame` | `hardware_api.py:336` | `(channel, frameSync=0)` | optional bus-power darkness |
| `_setBusPower` | `hardware_api.py:370` | via `getController('UC2Config')` | |
| `autofocus` | `EC/autofocus_api.py:75` | `(minZ, maxZ, stepSize, illuminationChannel, mode, …, af_range, af_resolution, …) -> Optional[float]` | software (`AutofocusController`) or hardware (`FocusLockController.performOneStepAutofocus`) |
| `update_af_offset` | `autofocus_api.py:289` | post-function | stores `measured - expected` per region on the controller |
| `apply_focus_map_z` | `EC/focus_map_api.py:736` | `(x, y, group_id, channel)` | |
| `_compute_focus_map_for_group` | `focus_map_api.py:341` | move → autofocus → fit loop | |
| `save_frame_ome` | `EC/frame_saving.py:96` | post-function; reads `metadata["result"]`, writes via `FrameSaveWorker` to `OMEWriter.write_frame(frame, {x, y, z, time_index, z_index, channel_index, …})` | well/region info is per writer, not per frame |
| `finalize_tile_ome_writer` / `finalize_current_ome_writer` | `frame_saving.py:458/473` | | |
| `wait_time`, `wait_for_next_timepoint`, `dummy_main_func` | `ExperimentController.py:422/425/418` | | `wait_for_next_timepoint` is a blocking `time.sleep` that ignores stop |
| `set_led_status` | `hardware_api.py:186` | | status LED |
| `_read_i2c_snapshot` / `_start_i2c_logging` | `hardware_api.py:607/671` | | sensor sidecar |
| labware | `EC/labware_api.py:72-146` | `getLabwareList`, `getLabwareDefinition`, `selectWellsByPattern`, `applyWellSelectionToExperiment` | returns ready `Point` dicts |
| overview | `EC/overview_api.py` | `snapOverviewImage`, `registerOverviewSlide`, `runAutonomousOverviewScan` | own loop (538-599) |
| stitching | `EC/stitching_api.py` | `runAshlarStitching`, `stopAshlarStitching` | |

Normal mode (`EC/experiment_normal_mode.py`) chains these in a fixed order inside
`_create_tile_workflow_steps` (371-885): XY move (470) → focus-map Z (484-526) → autofocus with
`update_af_offset` (531-574) → per Z plane `move_stage_z` + `wait_time` (582-602) → per channel
light on / acquire (pre `set_exposure_time_gain`, post `save_frame_ome`) / light off (650-704), or
ring (706-773) or DPC ×4 (775-843) → return to base Z (854) → finalize writer (873). Finalisation
(887-1003) adds lights-off, `wait_for_next_timepoint` and optional return to origin. All timepoints
are unrolled before the run (`ExperimentController._run_normal_mode`, 262-319). Every step is
serialised into `<file>_protocol.json` (`experiment_mode_base.py:323`), which is already close to a
job export but holds function *names* only and cannot be re-imported.

#### 3.3.4 Performance mode and fast stage scan

`EC/experiment_performance_mode.py` reduces each region to a bounding box and sends one firmware
`stagescan` command (`ESP32StageManager.start_stage_scanning` → uc2rest `motor.py:1477`) with up to
five laser channels, `tPre`/`tPost`, hardware or software camera trigger; frames are drained by
`_writer_loop_ome` (`frame_saving.py:331`) and matched to `stage_scan_frame_table`. It has no
progress signal, no pause, timelapse nested *inside* regions with `time.sleep(t_period)`, uniform Z
only, one exposure for all channels, and the storage bug in §7 (fixed in Phase 0). The standalone copy
`EC/fast_stage_scan.py:startFastStageScanAcquisition` always sends `zstart=0` (117) and has an
inverted timelapse wait (122-123). In Aufgaben this is one opaque task: `fast_stage_scan(region,
channels, z, trigger)` producing a frame table, not something the tree walks into.

#### 3.3.5 useq MDA (`EC/mda_api.py`, `model/managers/MDASequenceManager.py`)

Two paths. `start_mda_experiment` converts a c/z/t sequence into WorkflowSteps
(`convert_sequence_to_workflow_steps`, 573-706; ignores timing, never saves).
`run_native_mda_sequence` runs `MDASequenceManager.run_mda` (165-271): a synchronous `for event in
sequence` loop with psygnal `frameReady` etc., Python-only before/after hooks (552-566), per-event
TIFFs, `getLatestFrame` without trigger sync, channels mapped to lasers by exact name, `CustomAction`
and `HardwareAutofocus` ignored or broken (`mda_api.py:211` looks up `autofocusManager`, but the
master exposes `AutoFocusManager`). It runs in its own thread, invisible to `getExperimentStatus`
and `stopExperiment`, and has no stop endpoint. `docs/NATIVE_USEQ_MDA.md:14-23` over-claims
"supports ALL useq-schema features". useq is the right *import format* for an Aufgaben engine (a
`MDASequence` is a declarative t/p/g/z/c product that compiles cleanly into nested loops), not a
second engine.

### 3.4 Duplicated loop implementations

An Aufgaben engine only pays off if these collapse onto it.

| Logic | Copies | Locations |
|---|---|---|
| Tile grid + snake order | 7 (+3 firmware-order copies) | `CoordinateCalculator.js:361/439`, `WorkflowController.py:217/224/374/568`, `ArkitektController.py:747`, `LightsheetController.py:525+`, `SquidStageScanController.py:157`, `StageMapController.py:974`, `ShitScopeController.py:83`; `scan_plan.stage_scan_frame_table`, uc2rest `zicZac`, `ExperimentModeBase.calculate_grid_parameters` |
| Z-stack loop | 5 | `experiment_normal_mode.py:582`, firmware `nZ`, `LightsheetController._stepAcquireThread`, `SIMController.performSIMZstackThread` (moves axis "X"!), `MDASequenceManager.run_mda` |
| Timelapse loop | 9 | `ExperimentController._run_normal_mode` + `wait_for_next_timepoint`, performance mode 516-564, `fast_stage_scan.py:79-123`, `TimelapseController:211-276`, `ROIScanController.roiscanThread`, `MDASequenceManager.run_mda`, `LightsheetController`, `SIMController.performSIMTimelapseThread`, `WorkflowController` |
| Firmware stage-scan entry | 3 (+1 empty stub) | performance mode, `fast_stage_scan.py`, `PositionerController.startStageScan:607`; `StageScanAcquisitionController.py` is 19 lines of docstring |
| "Grab a fresh triggered frame" | 6 | `hardware_api.py:224/268`, `ArkitektController.py:658/682`, `LightsheetController.py:440/473`, `DPCController.py:196/233`, `AutofocusController.py:965/999`, `StageCenterCalibrationController.py:113` |
| move/laser/acquire/wait wrappers | 5 | ExperimentController, TimelapseController, WorkflowController, `MDASequenceManager` (`_move_xy`, `_setup_channel`, `_acquire_image`), ArkitektController (`_move_xy`) |
| Autofocus loops | 3 | normal mode 531-574, `focus_map_api.py:393/470`, `ArkitektController._tile_scan:820` |
| Writers | 7 | `OMEWriter` via `save_frame_ome`, `_writer_loop_ome`, unused `_setup_ome_writers`, `tif.TiffWriter` in Timelapse/Workflow controllers, per-event TIFFs in MDA, `tif.imwrite` stacks in ROIScan, `StageMapController` stitched writer |
| Step engines | 4 | `WorkflowManager`, `WorkflowController` JSON+registry, `MDASequenceManager.run_mda`, normal-mode step generator |

### 3.5 Peripherals and general-purpose I/O

What exists (all under `UC2ConfigController.py` unless noted) and what a task needs on top.

| Peripheral | Reachable today | Missing for a task |
|---|---|---|
| Digital out 1–3 (ESP32) | `setDigitalOut(digitaloutid, digitaloutval)` (1260; 0/1/-1 = pulse), `getDigitalOut` (1240), `setupDigitalOutPin` (1282) | type hints (today `{}` schema in OpenAPI), named outputs in the setup JSON |
| TTL trigger table | `setTrigger(trigger1, delayOn1, delayOff1, …)` (1313), `sendTrigger(triggerId)` (1341), `resetTriggerTable` (1300) | typing; `ESP32LEDLaserManager.py:138` calls `_esp32.digital.sendTrigger` on a non-existent attribute (§7) |
| Digital in / "wait for signal" | `getDigitalIn(digitalinid)` (1202), `actDigitalIn` (1222): **polling only**. uc2rest `digitalin.register_callback` is never registered, so input changes reach nothing | `waitForDigitalIn(id, edge, timeout)` backed by a callback → `threading.Event` + socket signal |
| Camera HW/SW trigger | `TriggerAcquisitionController.setHardwareTrigger` (ignores its argument), `snap`, `fetchBuffer`; firmware `{"cam":1}` callback in performance mode | "wait for N external frames" |
| Pump (stepper / peristaltic / syringe) | `FlowStopController.movePump(value, speed)` (271) = relative move of `axisFlow` in **steps**; `PositionerController.movePositionerForever` | a `PumpManager` with steps-per-µL calibration and blocking `dispense(volume_ul, rate)`; a `pumps` block in the setup JSON |
| Flow-cell stop-flow | `FlowStopController.startFlowStopExperiment` (own pump → settle → snap loop, 392-436) | becomes one composite task or is rebuilt from `pump` + `wait` + `capture` |
| Valve | nothing in managers, controllers or setups | `ValveManager` over DO / servo / RS232; task `set_valve(name, state)` |
| Servo / gripper / analog in / PID | uc2rest `laser.set_servo`, `gripper.*`, `analog.read_sensor`, `pid.*` — no endpoints (`ESP32StageManager.measure` reaches analog for HoliSheet) | endpoints + tasks |
| Fan, board temperature, bus power | `setFanMode`, `getFanState`, `getBoardTemperature`, `setBusPower`, `getBusStatus` — typed | fine as is |
| Heater PID | `TemperatureController.set_temperature(active, Kp, Ki, Kd, target, sensorOffset)` (105, untyped); no read endpoint | typing, `getTemperature`, `wait_until(sensor cmp value)` |
| I2C sensors | `I2CSensorController.getLatestI2CSensorValues` (232), polling + CSV (SHT45, TSL2591 only) | generic `i2cTransaction`; `wait_until` |
| Generic RS232 device | `rs232sManager[name].query/write` internally; `writeSerial(payload)` (1086) is ESP32-only and untyped | `sendRS232(deviceName, command, expectReply)` with a device enum |
| External robot / OT-2 | Arkitekt apps `fairinogale` (`pick_up`, `release_at`, …) and `OT2windy_service.run_protocol`; ImSwitch is a *provider*, not a *caller* | an Arkitekt client task, or plain HTTP |
| HTTP call | nothing generic | `http_request(url, method, json, timeout)` |
| Scheduler / time window | `lepmon/lepmon_capturing.py` state machine + sunset/sunrise (`lepmon_times.py`); `FlowStopController` uses `stopEvent.wait(t)` (the right wait primitive) | `wait_until_time`, interval loop with interruptible wait |
| Event → action precedent | PTZ keyboard mapping `_buildPtzActions` / `_dispatchPtzEvent` (301-413), `registerCaptureCallback` (85) | the pattern for "on input event, run task" |

The schema side: FastAPI already publishes every `@APIExport` with parameters as JSON schema at
`/imswitch/openapi.json` (`ImSwitchServer.createAPI`, 730-783, uses `@wraps` so the original
signature survives), `HyphaController.get_schema()` (removed in Phase 0) returned `{tool: PydanticModel.schema()}`,
and `ArkitektController._offerer` (243-256) + `_choices` (49-60) build typed actions with
setup-specific enums. `@APIExport` itself records no category, units, "moves hardware" or
"long-running" flags (`imcommon/model/api.py:6-20`).

### 3.6 Frontend

- **[Phase 0: removed]** **Blockly prototype** (`frontend/src/components/BlocklyController.js`, `blockly/customblocks.js`,
  `blockly/jsonGenerator.js`; `appRegistry.js:374`, `enabled: false`). Broken contract: uploads to
  `/WorkflowController/create_workflow_definition_api` (does not exist; backend is `uploadWorkflow`);
  emits `move_stage`, `set_exposure_time`, `set_gain`, `save_image`, `for_loop` which are not in the
  registry; `move_stage_block` uses `field_variable` so X/Y/Z parse to `NaN`; `controls_repeat_ext`
  is unrolled client-side and nested bodies are dropped on `JSON.parse`; `controls_if`/`while` have
  no generator; `sigUpdateWorkflowState` is never consumed; `WorkflowSlice` progress fields are never
  written. One useful asset: `src/__tests__/jsonGenerator.test.js`.
- **ExperimentDesigner** (`frontend/src/axon/experiment-designer/`): `DimensionBar`,
  `PositionsDimension`, `ChannelsDimension`, `ZFocusDimension`/`ZStackPanel`, `FocusMapDimension`,
  `TimeDimension`, `TilingDimension`, `OutputDimension`, `ObjectiveDimension`, `ExperimentSummary`.
  Builds the `Experiment` payload in `handleStart` (~222-468) → `apiExperimentControllerStartWellplateExperiment`.
  These panels are the natural **resource editors** for the Aufgaben app (well selection, point set,
  z-stack definition, capture definition, focus map). Its Pause button hits
  `/WorkflowController/pause_workflow` instead of `/ExperimentController/pauseWorkflow` (§7; fixed in Phase 0).
- **Infrastructure to reuse as-is**: `appRegistry.js` (`requiredControllers` hides an app when the
  controller is absent), `App.jsx` render switch, `NavigationDrawer`, `?app=` deep links,
  one-wrapper-per-endpoint `backendapi/`, the single `WebSocketHandler.js` (`processSignalData` →
  Redux slice; add a branch per new signal), Redux Toolkit + redux-persist, `react-sortablejs`
  (already used for the point list), `react-ace` + `json-edit-react` (JSON editing), MUI Stepper
  wizards (`AcceptanceTestComponent.jsx`, `wizard-steps/`), `StageMapController.jsx` (a "pick a
  position" surface), `JupyterExecutor.js`. Mobile kiosk shell `src/mobile/` has no experiment page.
- **Not installed**: reactflow/xyflow, rete, dnd-kit, react-dnd, monaco, codemirror, rjsf.

### 3.7 Scripting escape hatches

`DebugController.runCode(code, entrypoint="run", background, timeout)` and `runScript` (384-401),
hot-loaded from `<Root>/debug_scripts`, each job in its own thread, with a `ScriptContext`
(`master`, `commChannel`, `setupInfo`, `getController`, `stop_requested()`, `sleep()`); explicitly
unsandboxed. `--with-kernel` embeds an IPython kernel in the main thread (`imcommon/applaunch.py:29`);
`imnotebook` serves JupyterLab. `HyphaController.script_executor` evaluated code (removed in Phase 0). The legacy
`imscripting` module is removed (`__main__.py:175`) and `docs/scripting.rst` describes something that
no longer exists. `ArkitektController` offers nine typed actions to a remote orchestrator;
`SiLa2Controller` offers three features. The `DebugController` pattern is the right basis for a
`Python Script` task.

### 3.8 JOBS task → ImSwitch mapping

Legend: ✅ exists as a callable primitive, 🟡 exists but needs wrapping/typing/fixing, ❌ missing.

| JOBS task | ImSwitch primitive | State |
|---|---|---|
| Define Plate / Select Wells | `getLabwareList`, `getLabwareDefinition`, `selectWellsByPattern`, `WellSelectorCanvas` | ✅ |
| Loop over Wells / Points / Regions | — (unrolled in Python) | ❌ container node |
| Move to Well Center / Move to Point | `move_stage_xy` | ✅ |
| Generate Points / Predefined Points / Reorder / Offset | `CoordinateCalculator.js`, `Point`, `scanAreas` | 🟡 move grid/snake to backend (`scan_plan`) |
| Get Current XY / Z Position, Move to XY/Z Position | `getPositionerPositions`, `move_stage_*` | 🟡 saved-position resource |
| Set XY / Z Speed | `setPositionerSpeed`, `SPEED_X/Y`, `maxSpeedZ` | ✅ |
| Move to Loading / Acquisition Position, Escape Z | `moveToSampleLoadingPosition`, `moveToTransportPosition` | ✅ (ask-first) |
| Define Z-Stack / Z-Stack Loop / Move to Z-Stack home | `scan_plan.z_offsets`, `zStackMode`, `move_stage_z` | 🟡 definition resource + container |
| Assign Z-Stack / Current Z to Point | `Point.z`, `zStackOffsets` | 🟡 |
| Time Lapse / Repeat N times / Every nth | `wait_for_next_timepoint`, `autoFocusEveryNFovs` | 🟡 container nodes with proper scheduling |
| Autofocus Settings / Autofocus | `ParameterValue.autofocus_kwargs()`, `autofocus()` | ✅ (fix §7.2) |
| Create / Move to / Offset Focus Surface | `focus_map_api.py`, `apply_focus_map_z` | ✅ |
| PFS | `FocusLockController.enableFocusLock`, `performOneStepAutofocus` | ✅ |
| Capture Definition | `illumination`, `illuIntensities`, `exposureTimes`, `gains`, `syntheticChannels`, `laserPresets`/`ledPresets` in the setup | 🟡 needs a named resource |
| Capture / Capture (current) / Loop over Components | `set_laser_power` → `set_exposure_time_gain` → `acquire_frame` → `save_frame_ome` | ✅ |
| Save Image Never/Later | `metadata["result"]` without `save_frame_ome` | 🟡 explicit `save: bool` |
| Scan Large Image (+ stitch) | normal-mode tiles + `OMEWriter` mosaic, Ashlar; `StageMapController` prescan | 🟡 wrap as task |
| Scan Large Image by Continuous Movement | `StageMapController.startPrescan` / strobe sweep | 🟡 |
| Fast Timelapse / Triggered Experiment | performance mode, `TriggerAcquisitionController` | 🟡 one opaque task |
| Live Window | `LiveViewController.startLiveView/stopLiveView` | ✅ |
| Auto Brightness | `setDetectorMode(isAuto)`, `setDetectorExposureOnce` | ✅ |
| Select Objective | `moveToObjective(slot)` | ✅ (ask-first) |
| Open/Close Shutter | `setLaserActive` | ✅ |
| Sequential / Simultaneous Stimulation | `GalvoScannerController`, LED matrix | 🟡 |
| Condition (If/If-Else), Break, Continue, Logical Expression | — | ❌ |
| Variables, Expression | — | ❌ |
| Wait | `wait_time` (uninterruptible `sleep`) | 🟡 interruptible |
| Question / Debug | — | ❌ blocking node + socket signal + UI dialog |
| Python Script / Execute Python After Run | `DebugController.runCode` | 🟡 |
| Send E-mail / SMS | — | ❌ (`http_request` to a webhook covers most needs) |
| Create/Assign Label, Set Class | `Point.conditionLabel`, `wellId` | 🟡 |
| Parameters (wizard/run time) | — | ❌ |
| Phase, Block, Finish, Abort, Exit with Error | `stopExperiment` (graceful) vs `forceStopExperiment` | 🟡 |
| Storage / Alternative Storage Location / Split storage | `writer_flags`, `OMEWriter` per region per timepoint, `StorageController.set_active_path` | 🟡 writer-per-scope policy |
| Custom Metadata | `MetadataController` hub, `write_plate_sidecar` | 🟡 |
| Intensity / Cell Count Analysis | focus metrics in `AutofocusController`; scikit-image is a dependency | ❌ as tasks |
| Out Proc (external program) | Arkitekt/HTTP | ❌ as task |
| Use ROI from Analysis / Region List from analysis | — (needs `PixelCalibration` affine) | ❌ |
| Temperature and Gas Control | `TemperatureController.set_temperature` | 🟡 typing + readback |
| Analog Out | laser/LED value, galvo DAC (no endpoint) | 🟡 |
| Wait for I/O Event | `getDigitalIn` polling | ❌ event-based wait |
| Plate / Slide Loader | OT-2 + Fairino via Arkitekt (external) | ❌ as task |
| Results viewer / Job database | `listActiveSessions`, `getSessionMetadata`, `_protocol.json` | ❌ |

---


---

## 4. Architecture

### 4.1 Decisions (Phase 0)

| # | Question | Decision | Why |
|---|---|---|---|
| D1 | Name | **Aufgaben** for the feature (`AufgabenController`, frontend app "Aufgaben"). A document is a **job**. | "JOBS" is Nikon's product name. |
| D2 | Job format: useq-schema or our own? | **Our own small tree format that embeds useq models unchanged.** Positions, well-plate plans, grid plans, z plans, time plans and channels in a job *are* useq objects. Loop nodes index frames on useq axes (t, p, g, z, c). A job without control flow exports to an `MDASequence`; any `MDASequence` imports. | useq 0.8–0.9 is a Cartesian product of axes. It has no conditions, variables, loops on results, peripheral steps or operator prompts. Its unreleased "v2" (`MultiAxisSequence`, nested axes, `should_skip` on already chosen values) adds nesting but no feedback channel. Embedding useq keeps the standard where it applies and stays compatible with pymmcore-plus and useq consumers. |
| D3 | ExperimentController | **Keep it as it is.** Build `AufgabenController` beside it. No bridge that reroutes `startWellplateExperiment`. Phase 1 moved only its OME-Zarr writer onto the shared NGFF writer; names and layout are unchanged. | The WellPlate UI and the napari openUC2 processor depend on its behaviour and file layout. |
| D4 | Output format | **Standard OME.** OME-Zarr as **NGFF 0.4 on zarr v2** by default, NGFF 0.5 on zarr v3 as an option; one image per position, an NGFF plate for well-plate runs; OME-TIFF per position with full OME-XML. Validated in CI. | The OME audit (§7.3) found that no ImSwitch zarr output was valid NGFF. NGFF 0.4/zarr v2 is what napari-ome-zarr, ome-zarr-py, vizarr and the Bio-Formats stack (Fiji, QuPath, OMERO) all read. |
| D5 | napari openUC2 processor | It reads only per-tile `.tif` files and `*_protocol.json` (§7.4). Aufgaben output is OME; the plugin gets an **OME-NGFF reader path** (positions from NGFF translations) in Phase 2. An optional "legacy tiles" export in the Aufgaben writer covers the gap until then. | Keeps the plugin working without tying the new writer to the old naming scheme. |
| D6 | Host independence | The engine core lives in `imswitch/imcontrol/model/aufgaben/` and imports only pydantic, useq-schema, numpy and the standard library (enforced by a test). ImSwitch provides a host adapter implementing the `Microscope` protocol. | The core moves to newswitch unchanged, like `model/canbus`. |
| D7 | Task formalism | **Plain functions.** Typed parameters; units in the name suffix (`_um`, `_ms`, `_s`, `_ul`, `_c`); descriptions only in a Google `Args:` docstring; `Literal`/`Enum` choices; `annotated_types` limits; device choices declared in the decorator; keyword-only injected `hw: Microscope` and `ctx: TaskContext`. `@task` returns the function unchanged. | Arkitekt 5.0.1 and 6.2.1 read exactly this. Verified: Arkitekt's own `prepare_definition` accepts all 19 tasks (a test runs it). Arkitekt reserves the port name `value`; the registry rejects it. |
| D8 | Arkitekt | **Not in scope.** A separate project can register every task with `app.action(locks=…, effects=…, widgets=…)(fn)` without changes. | The user's instruction. Note: the local newswitch checkout still uses rekuest-next on every branch (last fetch 2026-10-09); Arkitekt 6.2.1 exists in `~/Downloads/arkitekt` and the uv cache. |
| D9 | Editor shape | **Outline tree**, not a node graph; schema-driven forms. Blockly removed. | JOBS is sequential with nesting; the tree matches it and needs no new dependency. |
| D10 | Loops and motion | Position, well, tile and Z loops **move to each item by default** (`move: true`). `move: false` plus a `move_to_item` node gives JOBS-style explicit moves (e.g. inside a condition). | The common case needs no extra node; the explicit form stays available and visible. |

### 4.2 The package (`imswitch/imcontrol/model/aufgaben/`, Phase 0)

| Module | Content |
|---|---|
| `document.py` | `Job`, `Node`, `Parameter`, `Storage`, the resource kinds (`positions`, `well_plate`, `grid`, `z_plan`, `time_plan`, `capture`, `autofocus`, `regions`, `focus_map`), `CaptureChannel` (a useq `Channel` plus intensity, gain, LED pattern) |
| `control.py` | 26 control nodes with parameter models, allowed branches, useq axis, accepted resource kinds and expression fields (Phase 2 added `loop_regions`, `create_focus_map`, `move_to_focus_surface`) |
| `hardware.py` | `Microscope`: the protocol a host implements (stage, light, camera, autofocus, objective; optional fast stage scan, prescan, digital I/O, pumps, valves, temperature, sensors); `DerivedImage` for images a task computes |
| `registry.py` | `@task`, `TaskContext` (carries Arkitekt's task marker), `TaskFlags` (moves, homes, light, external, code, confirm, needed capabilities), `catalogue()`, `MEMORY_STRUCTURES` (types an Arkitekt app declares once) |
| `primitives.py` | 21 tasks: move to/by, home, loading position, illumination, camera, autofocus, objective, fast stage scan, prescan, digital out/pulse/wait for input, pump volume/run/stop, valve, set/read temperature, HTTP request |
| `engine.py`, `simulated.py` | The executor (§4.3) and a simulated host for tests (Phase 1) |
| `storage.py`, `stitching.py` | The shared OME writer (§4.5) and tile stitching (stage positions or Ashlar) |
| `focus_map.py` | Surface fitting with leave-one-out error, shared with ExperimentController (moved here in Phase 2) |
| `expressions.py` | Safe expression language: whitelisted AST, no `eval`, no imports, no underscore names, bounded exponents |
| `validate.py` | Structure, references, parameter types, loop scoping, capabilities, and the ask-first list |
| `useq_bridge.py` | `job_from_sequence`, `sequence_from_job`; round trip gives identical events |
| `examples/*.json` | Live-cell plate time lapse; stop-flow with heater, valve and pump; tile scan with focus check and operator question; prescan, focus map and region tiles |

A job, abridged from `examples/stop_flow_fluidics.json`:

```json
{"schema": "aufgaben.job/1", "name": "Stop-flow imaging with temperature control",
 "parameters": {"cycles": {"type": "integer", "default": 200}, "target_c": {"type": "number", "default": 37}},
 "variables": {"images": 0},
 "resources": {"bf": {"kind": "capture", "channels": [{"config": "LED", "exposure": 10, "intensity": 300}]}},
 "tasks": [
   {"id": "heat", "task": "set_temperature", "params": {"target_c": "${param.target_c}"}},
   {"id": "warm", "task": "wait_until", "params": {"expr": "device.temperature_c >= param.target_c - 0.5"}},
   {"id": "to_sample", "task": "set_valve", "params": {"valve": "inlet", "position": "sample"}},
   {"id": "cycle", "task": "repeat", "params": {"count": 200}, "body": [
     {"id": "pump", "task": "pump_volume", "params": {"pump": "peristaltic", "volume_ul": 50}},
     {"id": "settle", "task": "wait", "params": {"seconds": 2}},
     {"id": "img", "task": "capture", "params": {"channels": "@bf"}},
     {"id": "count", "task": "set_variable", "params": {"name": "images", "value": "${images + 1}"}},
     {"id": "enough", "task": "condition", "params": {"expr": "images >= param.cycles"},
      "body": [{"id": "stop", "task": "break", "params": {"loop": "cycle"}}]}]}]}
```

Conventions: `"@name"` references a resource; `"${...}"` is an expression evaluated at run time;
`"$from": "@name"` fills a task's parameters from a resource. Expression namespaces are `param`,
`var` (a bare name is a variable), `loop.<id>` (`index`, `count`, `item`), `result.<node id>`,
`device` and `job`. Node options: `enabled` (bool or expression), `every` (every n-th iteration of a
loop), `on_error` (`abort`, `skip`, `continue_loop`, or retry), `label`.

The JSON Schema describes the canonical saved form (`Job.to_json()`). It is generated in
serialization mode and lets duration fields be seconds, because useq writes intervals as seconds
while its own schema declares ISO duration strings. Loading also accepts useq's shorthands (a plate
name such as `"384-well"`, a scalar grid overlap).

### 4.3 Executor (`engine.py`, built in Phase 1)

- **Walk.** `Run` walks the tree on its own thread. Each loop pushes a frame (`index`, `count`,
  `item`) that expressions read as `loop.<id>`, and adds its useq axis (t, p, g, z, c) to the index
  of every frame captured inside it. A `repeat` counts as t when no time lapse encloses it.
- **State.** Variables, parameters, resources, results and the run id live in the run. It does not
  reuse ExperimentController's `WorkflowContext` (D3); the two share only the busy check.
- **Checkpoints** before every node: stop first, then pause.
- **Stop semantics.** `finish` stops at the next node and keeps the data. `abort` does the same and
  marks the run aborted. `abort(force)` also stops the stage and switches the light off at once.
  Light off and writer close always run in `finally`.
- **Time lapse.** Period scheduling, `on_overrun: run_now | skip`, an `until` expression, and the
  `idle` branch while it waits.
- **Errors** per node through `on_error`: `abort`, `skip`, `continue_loop`, or retry n times and
  then one of those.
- **MDA.** The `mda` node runs the events of an embedded useq sequence directly.
- **Events** are plain dicts: run state, node start and end, questions, log lines.
- **Run record.** `job.json` (document plus resolved parameters) and `run.jsonl` (one line per
  node) next to the data.
- **Tests** run it on `SimulatedMicroscope` (`simulated.py`), a host that records every call and
  returns frames whose brightness encodes the light and Z.

### 4.4 Host adapter (`controllers/aufgaben/host.py`, built in Phase 1)

`ImSwitchMicroscope` implements `Microscope` over ExperimentController's verbs: `move_stage_xy/z`,
`set_led_matrix_pattern`, `set_exposure_time_gain`, `grabCameraFrame`, `autofocus` and the
triggered-acquisition bracket. Lasers go through `lasersManager`, digital I/O through the UC2Config
board (`digitalout`, `digitalin`), temperature through `TemperatureController` without its setpoint
offset (else the board sensor). Pumps and valves raise `NotImplementedError` until Phase 4.
`capabilities()` and `devices(kind)` come from the loaded setup, so the catalogue hides what the
instrument lacks and offers its real device names as choices.

### 4.5 Storage (`storage.py`, built in Phase 1, shared with ExperimentController and recording)

- One small writer on zarr-python, numcodecs and tifffile. `NgffImage` writes one 5-D TCZYX image.
  `NgffPlate` lays out an NGFF plate with row, column and field groups. `OmeStore` is the Aufgaben
  frame sink: a writer thread, one image per capture node, position and tile, and a plate when
  positions carry a well.
- ExperimentController's `OMEWriter` and the recording `OMEZarrWriter` write through `NgffImage`
  too, so chunking, metadata and pyramids exist once (§7.3 lists what that fixed). Their file
  names, per-tile TIFFs and `_protocol.json` are unchanged.
- `save_image(path, data, axes, ...)` writes one array in one call: `.ome.tif` → OME-TIFF,
  `.ome.zarr` → OME-Zarr, with optional pixel size, stage origin, channels and `annotations` (an OME
  `MapAnnotation` and `attrs["acquisition"]`). Snaps, experiment tiles, STORM, Lightsheet, DPC, SIM,
  FlowStop, StageMap, MMCore, read-noise calibration, the workflow engine and the MDA manager all
  use it. After Phase 2 this is the only image writer in ImSwitch; the deleted ones are listed in
  §7.9. Only three flag-guarded debug dumps still call tifffile directly.
- ExperimentController's `stitched.ome.tif` is the region canvas exported once at the end (one page
  per channel and Z plane, with its pyramid). `write_tiff`, `write_stitched_tiff` and
  `write_tiff_single` all request it; without `write_zarr` the canvas lives in a hidden temporary
  store that is removed after the export.
- Format: NGFF 0.4 on zarr v2 (`.zattrs`, "/" chunk keys, Zstd level 1) by default, NGFF 0.5 on
  zarr v3 as an option. Zstd rather than Blosc: Fiji's N5 reader cannot decode Blosc on macOS
  arm64. RGB becomes three uint8 channels coloured red, green and blue;
  thumbnails compose them back.
- Metadata: axes with units; `scale [dt, 1, dz, py, px]` and `translation` at the top-left pixel
  corner in µm (stage position minus half the field of view); no multiscale-level transform; omero
  channels with names, colours and a display window over the dtype range.
- Pyramids are built at `finalize` by stride-2 decimation in row bands, so memory stays bounded for
  large mosaics. T is trimmed to the timepoints actually written, for a time lapse ended by `until`.
- OME-TIFF is exported from zarr level 0, one file per image: BigTIFF, tiles up to 512×512, zlib with
  a predictor, channel names, physical sizes, per-plane positions and `DeltaT`. Planes are streamed,
  not held in memory.
- Validation: ome-zarr-models and ome-types are runtime dependencies. Every NGFF attribute block
  (image, plate, well) is validated against the ome-zarr-models schema of its version before it is
  written, and unit tests open every output with both libraries.
- OME-XML: one builder, `ome_xml`, on ome-types: physical sizes, channels with colours, per-plane
  positions (top-left corner, µm), `DeltaT` and `ExposureTime`, and `SignificantBits` from the
  camera. The OME-TIFF export, ExperimentController's mosaic hyperstack and every single-plane TIFF
  (experiment tiles, snaps) use it.
- OME-TIFF export (Phase 2): the pyramid goes into SubIFDs; planes are streamed tile by tile, so a
  large image never sits in memory. The OME-XML is stored as UTF-8: with `&#181;` escapes
  Bio-Formats does not recognise the file as OME-TIFF.
- Readers checked: Fiji (Bio-Formats `ImageInfo` on the TIFF, the N5 OME-NGFF reader on the zarr;
  `test_ome_fiji_compatibility.py` runs both when Fiji and Java are installed), napari through the
  openUC2 plugin (`ome_layers`), ome-zarr-models and ome-types.
- Orientation: frames reach the writer in stage orientation, because the pixel calibration's flips
  are applied in the camera (`setFlipImage`). Rotation and shear are not applied anywhere in
  ImSwitch; a rotated camera needs that first.
- Display window: from the camera bit depth when the detector manager exposes `bitDepth`, else the
  dtype range. No manager exposes it yet; the plumbing is in place.
- Each image carries `attrs["imswitch"]` (`node`, `position`, `tile`, or `derived`) so readers can
  rebuild the scan without parsing file names.
- Stitching (`stitching.py`): with `stitched_preview`, one stitched image per capture node and
  position from its tiles, at stage positions or after Ashlar registration (ashlarUC2's
  EdgeAligner). The raw tiles stay untouched.
- The legacy per-tile TIFF export (D5) is not needed: the napari plugin reads OME-Zarr (§7.4).

### 4.6 HTTP and Socket.IO (`AufgabenController`, built in Phase 1)

Method names carry "Aufgaben" to stay globally unique (`DebugController` already owns `listJobs`,
`getJobStatus` and `stopJob`).

| Endpoint | Purpose |
|---|---|
| `getAufgabenCatalog()` | Palette: control nodes and tasks with JSON schemas, flags, units, the setup's device choices |
| `getAufgabenJobSchema()` | The job JSON Schema |
| `listAufgabenJobs()`, `getAufgabenJob(name)`, `saveAufgabenJob(job)` (POST), `deleteAufgabenJob(name)` | Documents in `<config folder>/aufgaben/` |
| `validateAufgabenJob(job)` (POST) | Errors, warnings, ask-first list, motion and light summary |
| `startAufgabenJob(job, parameters, confirmed)` (POST) | 409 while anything else runs, 403 for unconfirmed ask-first nodes, 400 for an invalid job |
| `pauseAufgabenJob`, `resumeAufgabenJob`, `finishAufgabenJob`, `abortAufgabenJob(force)` | Run control |
| `getAufgabenStatus()`, `answerAufgabenQuestion(answer)` | Status, and the operator's answer to the open question |
| `importAufgabenUseq(sequence)` (POST), `exportAufgabenUseq(job)` (POST) | The useq bridge |
| `getAufgabenWellPlate(load_name, wells, offset_x_um, offset_y_um)` | A useq well-plate plan from ImSwitch labware and its stage offset (Phase 2) |
| `listAufgabenRuns(limit)` | Run folders, newest first |

Runs write to `<data folder>/Aufgaben/<YYYYmmdd_HHMMSS>_<job>/`: `job.json`, `run.jsonl` and the OME
files. One signal, `sigAufgabenEvent`, carries the engine events to Socket.IO; node events are
throttled to 10 Hz.

The run lock is the busy check in `controller/busy.py`. ExperimentController, Aufgaben, the stage
map and its prescan, focus-lock calibration, Arkitekt and the firmware updater refuse to start while
another of them, a recording or a workflow runs. A job's own prescan calls the stage map past that
check (`runPrescan`), because the job already holds the instrument.

### 4.7 Frontend app (Phase 3)

- Registered like other apps: `appRegistry.js` entry with `requiredControllers: ["AufgabenController"]`,
  the `App.jsx` switch, one wrapper per endpoint in `backendapi/`, an `AufgabenSlice`, and branches
  for the progress and question signals in `WebSocketHandler.js`.
- Palette from the catalogue; outline editor with nested drag and drop (`react-sortablejs`, already
  installed); validation markers on nodes; ask-first badges.
- Parameter forms generated from the task schemas. Resource fields open the existing editors:
  `WellSelectorCanvas`, `ZStackPanel`, `ChannelsDimension`, `FocusMapDimension`, and `StageMapController`
  for picking positions.
- Run monitor: the live path through the tree, counters, ETA, the plate view filling in, question
  dialogs, pause, resume, finish and abort.
- Later: an operator wizard (parameters marked `ui: wizard`) and a kiosk page under `#/mobile`.

### 4.8 Peripherals (Phase 4)

- `DigitalIOManager`: named outputs and inputs in the setup JSON; input callbacks set a
  `threading.Event` and emit `sigDigitalInput`; typed `UC2ConfigController` GPIO endpoints (today
  untyped: IDs arrive as query strings).
- `PumpManager` (`StepperPumpManager` first: an axis plus `steps_per_ul` and a maximum rate; then
  serial syringe pumps) and a `pumps` setup block. FlowStop becomes a template job.
- `ValveManager` (`DOValveManager`, `ServoValveManager`, `SerialValveManager`) and a `valves` block.
- `send_rs232(device, command)`, typed temperature endpoints, `wait_until` over sensors.
- Outside the microscope: `http_request` exists; an Arkitekt client task belongs to the separate
  Arkitekt project (D8).

### 4.9 Virtual microscope

Phase 0 added `VirtualESP32Manager` (`managers/rs232/VirtualESP32Manager.py`) and registered it in
the bundled virtual setup as `rs232devices.ESP32`. Every consumer that reaches
`rs232sManager["ESP32"]._esp32` now works without hardware: UC2Config GPIO, fan, bus power and board
temperature; `TemperatureController`; `I2CSensorController`.

| Simulated | How |
|---|---|
| Heater | Ideal controller: the reading approaches the target (or ambient when off) with a first-order lag `heaterTauS`, limited to `heaterMaxRiseC` above ambient, plus noise |
| Digital outputs 1–3 | Latched; `-1` pulses (counted, ends low); trigger table stored |
| Digital inputs 1–3 | Loopback of an output (`mirrorOut`), a time schedule, or a simulated firmware message through `UC2ConfigController/writeSerial`: `{"task": "/digitalin_sim", "digitalinid": 1, "digitalinval": 1}` |
| E-stop | `{"task": "/estop_sim", "active": 1}` cuts bus power and calls the emergency callbacks |
| SHT45, TSL2591 | Temperature from the heater model, humidity and lux around configured values |
| Not modelled | CAN, collision detector, PTZ, galvo: reads are empty, setters raise `NotImplementedError` |

Phase 1 added: the virtual camera scales intensity with exposure (reference 1 ms) and clips to
16 bit; `VirtualCameraManager.getParameter` works; a blocking virtual stage move takes distance
divided by speed (default 15,000 µm/s, at most 10 s per move), so waits and timeouts behave as on
hardware.

Phase 2: the virtual setup enables the stage map, so `stage_prescan` runs virtually; its pixel
calibration now applies (the camera is mirrored in X and Y against the stage, like an uncalibrated
real camera; before, its overlapping tiles did not match); the virtual stage declares that its
stage scan sends no camera triggers, so `fast_stage_scan` is hidden there.

Still missing (all in device managers, none in shared code):

- `VirtualStageManager`: `moveForever`, and a stage-scan simulation with camera triggers (performance
  mode and `fast_stage_scan` cannot run virtually).
- A virtual LED matrix: ring and DPC channels switch nothing in the virtual image.
- The virtual sample is too sparse for Ashlar: it finds too few edges and keeps the stage positions.
- Virtual pump and valve managers once Phase 4 defines the real ones.
- The Virtual* managers should honour `managerProperties.rs232device` instead of the hard-coded
  "VirtualMicroscope", so a second instance can serve as a focus-lock camera.
- Objective changes do not change the image magnification.
- Autofocus needs NanoImagingPack for the defocus blur; without it there is no focus peak.

---

## 5. Roadmap

Estimates are developer-weeks for someone who knows the codebase, without hardware QA. Each phase
ends with its demo job running on the virtual microscope, pytest coverage, and a clean ruff run on
new files. Nothing is committed without being asked.

### Phase 0 — decisions, fixes, scaffolding — **done 2026-10-12**

| WP | Result |
|---|---|
| 0.1 Fixes | 19 fixes, §7.1 and §7.2. Verified with unit tests (503 backend unit tests pass, 71 of them new; the only errors are 4 Arkitekt tests that fail to import `arkitekt.app.sessions` in this venv) and an end-to-end run on the virtual microscope: pause stops acquisition, resume continues, each region writes its own files, the napari plugin parses the result. |
| 0.2 Removals | Hypha: controller, `hypha/`, manager, `HyphaInfo`, the `imjoy` packaging extra (lock: 10 packages fewer), setup-JSON entries, docs. Blockly: app, blocks, generator, test, `WorkflowSlice`, five dead API wrappers, three npm packages (lock also drops the `jsdom` 26 subtree). The Jest suite fails in the same 7 suites and 27 tests as on unmodified HEAD; those failures predate this work. |
| 0.3 Decisions | D1–D10, §4.1. |
| 0.4 Scaffolding | `model/aufgaben/` (§4.2), `docs/aufgaben/job.schema.json`, 62 unit tests: documents, validator, expressions (including what it refuses), registry, Arkitekt acceptance, useq round trips over four axis orders, host independence, schema drift. |
| 0.5 Virtual microscope | `VirtualESP32Manager` and its setup entry; Temperature and I2CSensor widgets enabled in the virtual setup; `TemperatureController.getTemperature`; virtual homing fixed. 9 unit tests plus checks over HTTP. |

### Phase 1 — engine, host adapter, writer, controller — **done 2026-10-13**

| WP | Result |
|---|---|
| 1.1 Executor | `engine.py` per §4.3. 16 tests on the simulated host: the three example jobs, nesting, `every`, `break` to a named loop and `else`, `repeat … until`, error policies, `exit_with_error`, `wait_until` timeout, pause, resume, abort, finish while paused, ask-first refusal, parameter checks, both overrun policies with the idle branch, Z-stack channel offsets, capture inside a channel loop, the `mda` node. |
| 1.2 Storage | `storage.py` per §4.5. 10 tests validate images (NGFF 0.4 and 0.5), plates, OME-TIFFs, RGB, and the ExperimentController and recording writers. `ome-zarr-models` and `ome-types` are in the `dev` extra. |
| 1.3 Host adapter | `ImSwitchMicroscope` per §4.4. |
| 1.4 Controller | `AufgabenController` per §4.6, the shared busy check, the job library and run folders. |
| 1.5 Virtual microscope | Exposure, `getParameter`, motion time (§4.9). |
| 1.6 API tests | 7 tests over HTTP on the virtual microscope: endpoints, catalogue devices, job library, validation, a run whose output validates as NGFF, ask-first confirmation, useq import and export. |
| 1.7 Common writer (added) | ExperimentController and recording write through `NgffImage` (§4.5, §7.3). |
| 1.8 Removals (added) | §7.7. |

Verified on 2026-10-13:

- Backend unit tests: 524 pass. The 4 Arkitekt tests that fail to import `arkitekt.app.sessions`
  failed before this work.
- API tests for Aufgaben, ExperimentController and multi-site scans: 24 pass. The ExperimentController
  file had failed with 404 since its paths lacked `/imswitch/api`; that is fixed.
- Jest: the same 7 suites and 27 tests fail as on HEAD.
- By hand over HTTP on the virtual microscope: the tile-scan example paused, resumed and finished,
  and its autofocus followed the virtual sample's tilt. The I/O job read the heater model and the
  digital loopback. An ExperimentController run wrote valid NGFF 0.4 with pyramid levels, and
  Aufgaben refused to start while it ran. The napari openUC2 processor parsed all 24 tiles of a
  2-timepoint, 2-region, 3-plane run.
- Not done: opening the output in a napari or Fiji window, and any run on real hardware.

### Phase 2 — acquisition depth and the napari path — **done 2026-10-13**

| WP | Result |
|---|---|
| 2.1 Capture | Ring and DPC channels with a typed `LedPattern` (the host maps DPC to the matrix halves); `keep_light_on_between_channels`, one light on at a time; exposure and bit depth recorded per frame. Channel Z offsets and the trigger bracket were already in Phase 1. |
| 2.2 Focus | `create_focus_map` (points, region grids, wells, or the current loop item; one surface per group; fails below `min_points`), `focus_map` on position, well, region and tile loops, `move_to_focus_surface`, `z_offset_um` per map. |
| 2.3 Well plates | Labware → useq plates (`register_labware`), `getAufgabenWellPlate`. NGFF plates were already in Phase 1. |
| 2.4 Scans, stitching | `loop_regions`, region-covering `loop_grid`, `fast_stage_scan`, `stage_prescan` with derived images, `stitched_preview` with stage or Ashlar placement. |
| 2.5 napari plugin | `processing/ngff.py`: OME-Zarr images become the plugin's tiles; all pixel reads go through one loader; Ashlar runs on its library API (`EdgeAligner`, `Mosaic`) for TIFF and OME-Zarr tiles; the pixel size comes from the data. `ome_layers` shows OME-Zarr and OME-TIFF in napari with scale, stage position, channels and pyramid, without napari-ome-zarr. Committed in `~/Downloads/napari-openuc2-processor`. |
| 2.6 Equivalence | Imported sequences acquire the useq events in order: four axis orders, expanded and kept whole. |

Verified on 2026-10-13:

- Backend unit tests: 559 pass; the 4 Arkitekt import errors predate this work. New: 21 Phase 2
  engine tests (including the region example end to end and Ashlar recovering stage errors), plate
  registration, the shared busy check, recording streaming, OME-TIFF export of large images, the
  experiment mosaic hyperstack and single-plane TIFF metadata.
- API tests on the virtual microscope: 26 pass (Aufgaben, ExperimentController, multi-site scan),
  including a region job with prescan, focus map, region tiling and stitching.
- By hand over HTTP: the region job wrote valid NGFF images and OME-TIFFs; overlapping virtual tiles
  now correlate at 0.997 (X) and 0.57 (Y), up from about zero; the napari plugin read the run as one
  region of 2 × 2 tiles and stitched it.
- napari plugin: 47 tests pass. Its own Ashlar processor fails in its venv on a scikit-image import
  (`skimage.restoration.uft`), unrelated to this change. Its `uv.lock` was already behind its
  `pyproject.toml` and is not regenerated here.
- Not done: any run on real hardware (fast stage scan, LED-matrix channels and prescan on a real
  stage are untested).

After Phase 2 (single writer, §7.9): Fiji's Bio-Formats 8.1.1 reads our OME-TIFFs as OME-TIFF with
channels, Z and three resolutions, and its N5 reader recognises the OME-Zarr stores; napari opens
both through the plugin with the right scale and position (headless `ViewerModel`); the plugin's
Ashlar mosaic matches a reference to 99.4 % of pixels. Not checked: a napari window, Fiji's GUI.

### Phase 3 — Aufgaben app (4–5 weeks; can start once 1.4 fixes the catalogue contract)

The app of §4.7: registry entry, slice, wrappers, signal handling, palette, outline editor with
validation markers, schema forms with the existing resource editors, run monitor, job library with
import/export and useq import. Tests: slice reducers, outline move/nest logic, form rendering per
field type, one RTL test per screen. Ask before adding a form library such as `@rjsf/mui`.

### Phase 4 — peripherals (2–3 weeks, parallel to Phases 2–3)

§4.8: typed GPIO endpoints, `DigitalIOManager`, pump and valve managers with setup blocks and
virtual counterparts, RS232 command task, typed temperature endpoints, FlowStop as a template job.

### Phase 5 — feedback and analysis (2–3 weeks)

Tasks that read captured frames: intensity, object count, focus metric, background detection;
`points_from_mask` through the pixel calibration affine; `regions_from_overview`; a `python_script`
task on the `DebugController` script context (unsandboxed, flagged ask-first, off unless the setup
enables it). Templates: skip empty wells, stop after N cells, overview → regions → high-mag z-stack.

### Phase 6 — operator mode, results, consolidation (3–4 weeks)

Operator wizard and kiosk page; run index (SQLite) with a per-well heatmap of a chosen result; OMERO
upload as a task. Timelapse, HistoScan, ROIScan and SquidStageScan were removed in Phase 1;
`WorkflowController` and the empty `StageScanAcquisitionController` (§3.4) can follow once their
users move to Aufgaben. ExperimentController stays (D3).

### Effort and order

| Phase | Weeks |
|---|---|
| 0 | done |
| 1 | done |
| 2 | done |
| 3 | 4–5 (parallel to 2) |
| 4 | 2–3 (parallel to 2–3) |
| 5 | 2–3 |
| 6 | 3–4 |

```
Phase 1 ──► Phase 2 ──► Phase 5 ──► Phase 6
   ├──────► Phase 3 (after the catalogue contract in 1.4)
   └──────► Phase 4 (after the host adapter in 1.3)
```

---

## 6. Open questions

| Topic | Question | Proposal |
|---|---|---|
| Form library | In-house schema form renderer or `@rjsf/mui` | Start in-house; decide in Phase 3 with real schemas |
| Python script task | Allow unsandboxed code in jobs? | Allowed only when the setup enables it, always ask-first |
| Run lock scope | Which other controllers must respect it? | Done: ExperimentController, Aufgaben, stage map and prescan, focus-lock calibration, Arkitekt, firmware updater; recordings and workflows count as busy |
| NGFF 0.5 | When to make it the default | When Bio-Formats/Fiji/QuPath read it; check release notes before Phase 6 |
| Temperature setpoint | `TemperatureController.set_temperature` adds `sensorOffset=3` °C to the target by default | Keep for the existing UI; the Aufgaben adapter passes 0 |
| newswitch | The core ports by design; the adapter and controller do not | Write the newswitch host adapter when newswitch's Arkitekt 6 migration lands |

---

## 7. Bugs and audit findings

### 7.1 Fixed in Phase 0 — ExperimentController, engine, MDA

| # | Problem | Fix | File |
|---|---|---|---|
| 1 | Pause was a flag nothing read; resume called a method that did not exist | Event-backed stop/pause in `WorkflowContext`, `request_resume`, `wait_if_paused`, interruptible `sleep`; checkpoint between steps | `model/managers/WorkflowManager.py` |
| 2 | Waits ignored stop (`time.sleep`) | `wait_time` and `wait_for_next_timepoint` use `context.sleep` in Experiment, Timelapse and Workflow controllers | `ExperimentController.py`, `TimelapseController.py`, `WorkflowController.py` |
| 3 | An extra full period was waited after the last timepoint | No wait after the last round | `experiment_normal_mode.py` |
| 4 | Force stop replaced the manager while the old thread kept running | Stop, then join the old thread for up to 2 s, then replace | `ExperimentController.py` |
| 5 | Per-point autofocus ignored the UI range and step (±100 µm, 10 µm always won) | Pass `af_range`/`af_resolution` like `autofocus_kwargs()` | `experiment_normal_mode.py` |
| 6 | The focus map for merged single-position areas used default autofocus settings | Pass the run's settings | `focus_map_api.py` |
| 7 | Normal mode allocated every per-timepoint writer for all timepoints (n× disk and RAM; tripped the large-scan policy) | One timepoint per writer; files and names unchanged | `experiment_normal_mode.py` |
| 8 | Single-position runs routed every frame to writer index `t`; with several regions frames went to the wrong writer or were lost | Index by timepoint and region always | `experiment_normal_mode.py` |
| 9 | Performance mode passed writer arguments positionally into the wrong slots: every Z-plane and channel landed in z=0, c=0, and the TIFF mosaic and stitched output were always on | Keyword arguments; the run's output flags | `experiment_performance_mode.py`, `fast_stage_scan.py` |
| 10 | Performance-mode regions shared one output path; each region reopened the same zarr store | `_r<NNN>` in the name when there are several regions (single-region names unchanged) | `experiment_performance_mode.py` |
| 11 | Stage-scan callback unregistration always raised (swallowed), so callbacks piled up | Optional callback argument in both stage managers | `ESP32StageManager.py`, `UC2CANOpenStageManager.py` |
| 12 | MDA autofocus never ran (wrong manager attribute, no `runAutofocus`); autofocus and custom-action events also acquired an image; no stop endpoint; time-lapse waits ignored stop | Adapter to the controller's autofocus; no image for those actions; `stop_native_mda_sequence`; sliced waits | `mda_api.py`, `MDASequenceManager.py` |
| 13 | ExperimentDesigner's Pause called `WorkflowController` | Calls `ExperimentController/pauseWorkflow` and reports a refused pause as a failure | `frontend/src/backendapi/apiExperimentControllerPauseWorkflow.js` |
| 14 | Tiles taken with no light were named `_p-1`, which the stitchers' `_p(\d+)` skips | Power field clamped to 0 | `ome_writer.py` |

### 7.2 Fixed in Phase 0 — other

| # | Problem | Fix |
|---|---|---|
| 15 | `SingleTiffWriter` declared `ZYX` for 2-D frames: every write after the first failed, the file held one image and no OME-XML | Axes match the frame; positions carry units; regression test |
| 16 | `ESP32LEDLaserManager.sendTrigger` used `_esp32.digital` (does not exist) | `_esp32.digitalout` |
| 17 | `TriggerAcquisitionController.setHardwareTrigger` ignored its argument | Passes it on |
| 18 | Virtual stage: `homeAxis` over HTTP raised `TypeError`; no A homing; `home_xyz` read undefined attributes | Accepts homing options; A homing; `home_xyz` homes X, Y, Z |
| 19 | `TemperatureController` had no read endpoint and called Qt widgets in its polling loop | `getTemperature()`; widget calls skipped headless |

### 7.3 OME compliance of the ExperimentController and recording writers (fixed in Phase 1)

Phase 0 left these writers alone (D3). In Phase 1 both moved onto the shared `NgffImage`, with their
file layout unchanged. What the audit found, and where it stands:

| Finding | Now |
|---|---|
| Every zarr output was zarr v3 with NGFF 0.4 metadata, valid in neither version | Zarr v2 with NGFF 0.4; a unit test validates it |
| The multiscale-level transform repeated the dataset scale, so readers squared the pixel size and the time step | Removed |
| Z spacing and time interval were never set (always 1.0) | From the run's Z step and period |
| The translation was the tile centre | The top-left pixel corner |
| RGB added a sixth `s` axis typed as a channel | Three uint8 channels; thumbnails compose them back |
| The chunk key sent to the frontend matched neither layout | The real key of the 5-D chunk grid; a test checks the files exist |
| Each writer had its own pyramid code | One banded implementation |
| No test checked format compliance | `ome-zarr-models` and `ome-types` tests for all three writers |
| Tiles are placed edge to edge, discarding overlap | Open: the canvas is a preview; stitching stays Ashlar's job |
| Plate metadata is only a `plate_metadata.json` sidecar | Open for ExperimentController; Aufgaben writes real NGFF plates |
| The "stitched" OME-TIFF is a multi-series tile collection | Open, unchanged |
| Individual TIFFs got OME-XML only when the optional `ome-types` was installed, and that builder had plane-index and unit errors | Phase 2: ome-types is a dependency; one builder for every OME-XML |
| `RecordingService.start_streaming` passed a path where the writers expect a `SessionContext` | Phase 2: rebuilt on the streaming adapter; the recording app uses it |

### 7.4 The napari openUC2 processor's read contract

Phase 2 added an OME-Zarr path (`processing/ngff.py` in the plugin): any NGFF image becomes the
plugin's tiles, one per plane, placed by its translation; ImSwitch images are grouped into regions
by their `imswitch` attribute and derived images are skipped. The TIFF path below is unchanged.

The TIFF path reads only per-tile `.tif` files matched by
`t(\d{8}_\d{6})(_r<region>)?_x<x>_y<y>_z<z>_c<idx>_<channel>_i<iter>_p<power>\.tif` and the first
`*_protocol.json` (`snake_tiles`, `iterator`, `iX`, `iY`, `region_id`). The timepoint comes from the
folder (`timepoint_NNNN`, or `_experiment<i>_<t>_` when the experiment is named "experiment"). It
never reads zarr, OME-XML, mosaics or the plate sidecar, and it does not depend on array shapes.
Phase 0 and Phase 1 keep this contract: after each phase the plugin parsed a run from the virtual
microscope.

Still open on the plugin side: region tags longer than 24 characters fall back to the merged map;
multi-experiment detection needs the default experiment name.

### 7.5 Other findings, not fixed

- `docs/NATIVE_USEQ_MDA.md` claims support for "ALL useq-schema features"; `docs/scripting.rst`
  describes a removed module; `docs/controller_communication_pattern.md` and
  `docs/INTER_CONTROLLER_COMMUNICATION.md` disagree on `getController` versus signals.
- `WorkflowController`: most registry functions are stubs and `getWorkflowStatus` reads an undefined
  attribute. Its crashing histo workflows were removed in Phase 2; retire the rest in Phase 6.
- The UC2Config GPIO endpoints are untyped (OpenAPI shows empty schemas; IDs arrive as strings).
  Phase 4.
- `pyproject.toml` pins `numpy==2.1.2`; the venv has 2.5.1. (Phase 2 moved the pydantic pin to
  2.12.5, which ome-zarr-models needs.)
- The Arkitekt controller tests fail to import `arkitekt.app.sessions` in the current venv.
- Seven Jest suites (27 tests) fail on HEAD: App, FocusLockController, configValidation,
  fetchWithTimeout, useDeveloperMode, LiveViewComponent, WiFiController.

### 7.6 Fixed in Phase 1

| # | Problem | Fix |
|---|---|---|
| 20 | The ExperimentController writer's OME-Zarr was not valid NGFF (§7.3) | Shared `NgffImage` |
| 21 | ExperimentDesigner's Vizarr link put an unencoded URL inside a query string | `encodeURIComponent` |
| 22 | `getOmeZarrUrl` raised `AttributeError` before the first scan | Returns null until a scan sets it |
| 23 | The Ashlar script path pointed one folder too shallow | Correct relative path |
| 24 | `LightsheetController` passed the tile grid as (rows, columns) where the writer wants (nx, ny) | Swapped |
| 25 | `setPositionerStepSize` was an empty stub; `setDetectorROI` went through the Qt-era parameter tree | Step sizes are stored per positioner; the ROI crops the detector directly, rounded up to multiples of 4 |
| 26 | `SharedAttributes` did not support `in` | `__contains__` |
| 27 | The virtual ESP32 rejected IDs that arrived as strings over HTTP | Coerced to int |
| 28 | ExperimentController started even while a recording or workflow ran | Shared busy check, 409 |
| 29 | `test_experimentcontroller.py` called paths without `/imswitch/api`, so every test hit 404 | Prefixed; discovery matches by suffix |

### 7.7 Removed in Phase 1

- **Apps**: Timelapse (controller, manager, frontend) and HistoScan (manager, setup info, three
  signals, Redux slice, the OpenLayers "Infinity Scanning" view and its `ol` package).
- **Dead controllers** with no headless use: AlignAverage, AlignXY, AlignmentLine, BeadRec, FFT,
  FOVLock, ISM, Image, ROIScan, RotationScan, Rotator, SLM, SquidStageScan, ULenses, Watcher,
  Tiling, WellPlate, HoliSheet, WebRTC, Console, two `_old` backups, PickSetup, PickUC2BoardConfig,
  the imcommon pickers and update checker, and the imnotebook Qt controllers.
- **Widget code**: every `self._widget` branch in the remaining controllers. TemperatureController,
  LEDController and SettingsController are now a fraction of their size; LaserController lost its
  Qt preset slots. 23 signals that nobody emitted or received are gone.

### 7.8 Fixed in Phase 2

| # | Problem | Fix |
|---|---|---|
| 30 | The OME-TIFF export streamed whole planes into tiled pages; tifffile wants tiles, so every frame wider or taller than 512 px failed (Phase 1 tests used small frames) | Streams padded tiles; a test exports a 700 × 1100 image with its pyramid |
| 31 | `PositionerController` did not start on the virtual microscope since Phase 1: the virtual stage replaced its per-axis speed dict with a number | Travel speed kept separately; a test checks the dict |
| 32 | The virtual pixel calibration never applied: old keys (`"0"`) and scale metrics that contradicted the matrix. Its camera is mirrored against the stage, so tiles could not be stitched | Keys `WidefieldCamera::<slot>` and signed metrics in the bundled virtual setup |
| 33 | `MetadataHub`'s OME builders imported a class ome-types does not have, so they never ran; nothing called them | Removed |
| 34 | The single-plane OME-XML builder used plane indices beyond SizeC/SizeZ, guessed mm from µm, referenced light sources that did not exist, and could not describe RGB | Rebuilt on the shared `ome_xml` |
| 35 | `RecordingService.start_streaming` could not work (wrong constructor argument, a missing write method, no finalize) | On the streaming adapter; `RecordingController` delegates to it |
| 36 | STORM wrote OME-Zarr through its own `MinimalZarrDataSource` | Shared `NgffImage`; T grows in doubling steps, trimmed at the end |
| 37 | A focus-lock calibration set its running flag inside its thread, so a busy check right after the start missed it | Set before the thread starts |
| 38 | WorkflowController's histo workflows wrote through a zarr stub that returned `None`; `startWorkflowTileBasedByParameters` did nothing | Removed |

### 7.9 One writer (after Phase 2)

| # | Problem | Fix |
|---|---|---|
| 39 | The OME-XML was ASCII with `&#181;` for µm; Bio-Formats rejected it and opened the file as a plain TIFF without channels or sizes | Written as UTF-8; Fiji test |
| 40 | Blosc chunks could not be decoded by Fiji's N5 reader on macOS arm64 | Zstd level 1 |
| 41 | Seven TIFF paths with their own metadata: `OmeTiffStitcher`, `SingleTiffWriter`, `OMETiffWriter`, `SnapService`, `ome_tiff_metadata` and direct `tifffile` calls in about fifteen controllers | Deleted; everything goes through `save_image` / `NgffImage`. `stitched.ome.tif` changed from a collection of tiles to the canvas |
| 42 | The plugin's Ashlar mode handed its reader to `process_images`, which raised `NameError` (`BioformatsReader`); ashlarUC2 1.18.4 also broke on scikit-image 0.26 | `EdgeAligner` and `Mosaic` called directly; ashlarUC2 ≥ 1.18.5 |
| 43 | The virtual strobe sweep made blocking moves, so it fell behind its own clock and the strobe test failed now and then | Non-blocking sweep steps |
| 44 | Local OME-Zarr needed napari-ome-zarr, which the plugin cannot depend on (s3fs resolution on Python 3.13+) | The plugin's own OME reader |

---

## Appendix A — example jobs

Runnable with the v1 catalogue and validated in CI (`imswitch/imcontrol/model/aufgaben/examples/`):

| File | JOBS use case | Shows |
|---|---|---|
| `hcs_live_timelapse.json` | UCSD NIC: live-cell plate, 4 channels, z-stack every 30 min for 16 h | well-plate plan with labels, grid of sites, `every` (autofocus every 2nd round), job parameters, `enabled` expressions, `idle` branch, plate storage layout |
| `stop_flow_fluidics.json` | FlowStop plus a heater and a valve | temperature, `wait_until` on a device reading, valve, pump, variables, `break` |
| `tile_scan_focus_check.json` | Tile scan with quality control | operator question, `$from` resource, `on_error`, condition on `result.<id>`, `exit_with_error`, `continue` |
| `region_overview_focus_map.json` | Slide overview, then tissue regions at high magnification | `stage_prescan`, `create_focus_map` per region, `loop_regions`, region-covering `loop_grid` on the focus surface, a dark-field ring channel, Ashlar-stitched output |

Needs later phases:

- **Histology overview** (UCSD use case 2): everything but `regions_from_overview` (Phase 5) exists;
  `region_overview_focus_map.json` uses fixed regions until then.
- **Single-molecule tracking** (UCSD use case 3): a 5-minute 30 ms series needs a streaming capture
  node, a `fast_timelapse` in the spirit of JOBS (Phase 2, on the camera's continuous mode).

## Appendix B — sources

Nikon NIS-Elements AR help (nisoftware.net, "Docs-AR/eng_ar"): `jobs.overview.html`,
`jobs.creating.html`, `jobs.basic.rules.html`, `jobs_introduction_to_expressions.html`,
`jobs_introduction_to_conditions.html`, `hca.viewresults.html`, `jobs.tasks.html`,
`task.wellplates_section.html`, `task.sampleholder_section.html`, `task.timeseries_section.html`,
`task.stagexy_section.html`, `tasks.largeimages.html`, `task.zstack.group_section.html`,
`task.rois_section.html`, `task.autofocus.group_section.html`, `task.acquisition_section.html`,
`task.stimulation_section.html`, `task.opticalconfigurations_section.html`,
`task.conditions_section.html`, `task.system_section.html`, `tasks.analysis.html`,
`task.devicecontrol_section.html`. UCSD Nikon Imaging Center, "JOBS: fully automated acquisition
pipeline" (nic.ucsd.edu/services/jobs.html). Nikon NIS-Elements brochure 2CE-MPCJ-82604T
(feedback/AI example).

ImSwitch: files cited inline; sibling projects `ARKITEKT_STUFF/protocolapp`, `OT2windy_service`,
`jena_flow_code_new`, `fairinogale` for the multi-device orchestration precedent.
