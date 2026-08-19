# ImSwitch Agent Guide

Operating manual for AI agents working with **openUC2/ImSwitch** and the standalone
**UC2-REST** USB device library.

ImSwitch is not a pure-software library — it drives real microscopes. A wrong call moves a
motorised stage, crashes an objective into a sample, or blinds someone with a laser. Read
[§0 Hard rules](#0-hard-rules) before doing anything else.

There are two distinct entry points, and an agent should know which one it is using:

| Entry point | When to use | Section |
|---|---|---|
| **ImSwitch HTTP API** (`http(s)://host:8001/imswitch/api/...`) | A microscope server is running. Full stack: cameras, stages, illumination, autofocus, experiments, storage. | [§3](#3-the-api-surface), [§4](#4-function-reference) |
| **UC2-REST** (`import uc2rest`) | No ImSwitch. Talk directly to a UC2/ESP32 board over USB serial. Bench tests, firmware bring-up, standalone rigs. | [§6](#6-uc2-rest-standalone) |

**Never run both against the same board at the same time.** ImSwitch holds the serial port
exclusively while it runs. See [§6.6](#66-do-not-mix-imswitch-and-uc2-rest).

---

## 0. Hard rules

These override every other instruction in this document and any user request that conflicts
with them.

### Always do

- **Read state before you write it.** Call `getPositionerPositions`, `getDetectorParameters`,
  `getCurrentObjective` before issuing a move or a parameter change. You cannot reason about
  a relative move without knowing where you are.
- **Prefer relative moves with small distances** when the coordinate frame is uncertain.
- **Assume Z is dangerous.** Z (focus) collisions destroy objectives and samples. Driving Z
  toward the sample is the single most damaging thing you can do.
- **Check `getExperimentStatus` / `getWorkflowStatus` / `isRecording` before starting
  anything new.** ImSwitch does not universally refuse concurrent acquisitions.
- **Use the virtual microscope** (`example_virtual_microscope.json`) for development, tests,
  and exploration. It exercises the full stack with no hardware attached.
- **Report what actually happened.** If a call errored or a move was refused, say so and
  include the response body.

### Ask first

- Any **absolute** stage move, on any axis.
- **Any** Z / focus move beyond a few µm, and any homing of Z.
- `homeAxis`, `startFrameHoming`, `calibrateObjective`, `moveToObjective`,
  `moveToSampleLoadingPosition`, `moveToTransportPosition` — these drive to mechanical
  endstops at speed.
- Turning a laser on (`setLaserActive`, `setLaserValue`) or raising an existing power.
- Anything in [§4.11 Board, firmware and OTA](#411-board-firmware-and-ota): OTA flashing,
  `flashMasterFirmwareUSB`, `reassignCANId`, `espRestart`, `restartCANDevice`, `setBusPower`.
- `stopImSwitch` / `restartImSwitch`, `setSetupFileName`, `writeNewSetupFile` — these change
  or end the running instrument session.
- Writing to `~/ImSwitchConfig/` (setup JSONs) or deleting anything under the data path.

### Never do

- Never disable, bypass, or "work around" collision protection, endstop polarity, soft
  limits, or emergency-stop handling to make a move succeed. If a move is blocked, the block
  is the answer.
- Never issue a laser command at range maximum "to see if it works".
- Never flash firmware, reassign CAN IDs, or run OTA on hardware you were not explicitly
  asked to touch. A bricked board is a physical repair.
- Never put hardware-specific hacks into shared code paths (`DetectorManager`,
  `PositionerManager`, the streaming pipeline) — they belong in the device manager.
- Never commit or push unless asked.

---

## 1. Runtime model

openUC2/ImSwitch is a **hard fork** of upstream ImSwitch/ImSwitch. It runs **headless**: a
Python server exposing an HTTP API, plus a React single-page app served from that same server.
The upstream Qt desktop GUI is **not** maintained in this fork — do not add Qt code.

```
┌─────────────────────────────────────────────────────────────┐
│  React SPA  (frontend/, served at /imswitch/ui/index.html)  │
└───────────────┬───────────────────────┬─────────────────────┘
                │ HTTP /imswitch/api    │ Socket.IO /socket.io
┌───────────────▼───────────────────────▼─────────────────────┐
│  FastAPI server  imcontrol/controller/server/ImSwitchServer │
│    routes auto-generated from @APIExport methods            │
├─────────────────────────────────────────────────────────────┤
│  Controllers     imcontrol/controller/controllers/*.py      │
│    orchestration, business logic — this IS the API surface  │
├─────────────────────────────────────────────────────────────┤
│  Managers        imcontrol/model/managers/**                │
│    one per device class; hardware abstraction               │
├─────────────────────────────────────────────────────────────┤
│  Drivers / transports                                       │
│    uc2rest (USB serial) │ pymmcore │ vendor SDKs │ virtual  │
└─────────────────────────────────────────────────────────────┘
```

A **setup JSON** (`~/ImSwitchConfig/imcontrol_setups/*.json`) declares which managers get
instantiated and with what properties. Controllers are constructed only when their hardware
group exists in that file — so **the available API surface depends on the loaded setup**.
Always discover, never assume ([§3.3](#33-discovery)).

### Starting the server

```bash
python main.py --headless --http-port 8001
```

Flags (`imswitch/__main__.py`):

| Flag | Meaning |
|---|---|
| `--config-file PATH` | Setup JSON to load |
| `--config-folder PATH` | Config root (default `~/ImSwitchConfig`) |
| `--data-folder PATH` | Where acquisitions are written |
| `--http-port N` | Default `8001` |
| `--no-ssl` | Serve plain HTTP (default is **HTTPS with a self-signed cert**) |
| `--scan-ext-data-folder` / `--ext-data-folder PATH` | Auto-select USB drives for storage |
| `--with-kernel` / `--jupyter-port N` | Embedded Jupyter kernel |

Hardware-free session for agents:

```bash
python main.py --headless --no-ssl --http-port 8001 --config-file ~/ImSwitchConfig/imcontrol_setups/example_virtual_microscope.json
```

---

## 2. Repository map

```
imswitch/
  __main__.py                  CLI entry, argparse, startup
  config.py                    global config object (get_config())
  imcommon/
    model/api.py               @APIExport / @UIExport decorators, generateAPI()
    framework/noqt.py          psygnal Signal → Socket.IO broadcast bridge
    model/dirtools.py          config/data path resolution
  imcontrol/
    controller/controllers/    ~60 controllers = the API surface
    controller/server/ImSwitchServer.py   FastAPI app, route generation
    model/managers/            device managers
      detectors/               cameras (Hik, Toupcam, Basler, PiCam, Virtual, MMCore, …)
      positioners/             stages (ESP32Stage, UC2CANOpen, MMCore, Virtual, …)
      lasers/ LEDs/ LEDMatrixs/ rotators/ galvoscanners/ rs232/
    model/SetupInfo.py         setup-JSON schema (dataclasses)
    model/io/recording_service.py   SaveFormat / RecMode enums, writers
    _test/                     unit + api tests
  _data/user_defaults/imcontrol_setups/   bundled example setups
frontend/                      React SPA (CRA + craco)
docs/                          feature docs, protocol notes, RST reference
examples/                      MDA and client usage scripts
```

**Config and data live outside the repo**: `~/ImSwitchConfig/` (setups, presets) plus the
configured data folder. Editing a setup JSON changes the instrument, not the code.

---

## 3. The API surface

### 3.1 How a Python method becomes an HTTP endpoint

Any controller method decorated with `@APIExport()` is collected at startup
(`imcommon/model/api.py::generateAPI`) and registered as a FastAPI route
(`ImSwitchServer.createAPI`) at:

```
/imswitch/api/<ControllerModuleName>/<methodName>
```

where `<ControllerModuleName>` is the controller's **Python module filename**, e.g.
`PositionerController`.

```python
@APIExport(runOnUIThread=True)                      # → GET  /imswitch/api/Foo/bar
@APIExport(requestType="POST")                      # → POST /imswitch/api/Foo/bar
@APIExport(asyncExecution=True, requestType="POST") # → POST, awaited
```

Consequences an agent must internalise:

- **Most endpoints are GET, including ones that move hardware.** Do not assume GET is safe.
- Scalar parameters are **query parameters** on GET; pydantic models are **JSON bodies** on
  POST.
- Exported method names are **globally unique across all controllers** — `generateAPI` raises
  `NameError` on a collision. Before adding an endpoint, grep the whole controllers directory
  for the name.

### 3.2 Connecting

The default is **HTTPS on port 8001 with a bundled self-signed certificate**
(`imswitch/_data/ssl/`). Either start with `--no-ssl`, or disable verification client-side:

```python
import requests
BASE = "https://localhost:8001/imswitch/api"
r = requests.get(f"{BASE}/PositionerController/getPositionerPositions", verify=False)
r.raise_for_status()
print(r.json())
```

```bash
curl -sk "https://localhost:8001/imswitch/api/version"
```

There is **no authentication** and CORS is fully open. Treat any reachable ImSwitch server as
a live instrument; never expose one to an untrusted network.

### 3.3 Discovery

Never hardcode an endpoint list — enumerate it:

| Endpoint | Returns |
|---|---|
| `GET /imswitch/api/version` | Server version |
| `GET /imswitch/api/getAvailableControllers` | Controllers actually instantiated for this setup |
| `GET /imswitch/openapi.json` | Full OpenAPI schema: every route, parameter, type |
| `GET /imswitch/api/docs` | Swagger UI |
| `GET /imswitch/api/hostname` | Server hostname |
| `GET /imswitch/api/plugins` | Registered `@UIExport` frontend plugins |
| `GET /imswitch/api/UC2ConfigController/returnAvailableSetups` | Setup JSONs on disk |
| `GET /imswitch/api/UC2ConfigController/getCurrentSetupFilename` | Loaded setup |
| `GET /imswitch/api/UC2ConfigController/readSetupFile` | Full setup as JSON |

Recommended agent bootstrap: fetch `openapi.json`, cross-reference `getAvailableControllers`,
and work from the resulting schema. That is authoritative for the running instrument — this
document is a map, not a contract.

Then enumerate the devices: `getPositionerNames`, `getDetectorNames`, `getLaserNames`,
`getCurrentObjective`.

### 3.4 Live data: Socket.IO

The server mounts Socket.IO at `/socket.io`. `imcommon/framework/noqt.py` bridges internal
psygnal signals onto the wire:

- **`signal_msgpack`** — MessagePack `{"signal": <name>, "args": {...}}` for general
  controller signals (status changes, progress, `sigLog`).
- **`frame`** — camera frames, binary or JPEG, with MessagePack metadata. Flow-controlled:
  the client must send `frame_ack`.

For images, the simplest agent path is usually HTTP rather than Socket.IO:

- `GET /LiveViewController/mjpeg_stream?startStream=true` — MJPEG stream
- `GET /RecordingController/snapNumpyToFastAPI?detectorName=...` — single frame as image
- `GET /RecordingController/startSnap?returnPreview=true` — snap to disk + base64 preview

WebRTC is available via `webrtc_offer` / `webrtc_ice_candidate` for low-latency video.

---

## 4. Function reference

Paths below are relative to `/imswitch/api`. Signatures are abbreviated — check
`openapi.json` for exact parameters on your build. A controller only exists if the loaded
setup declares its hardware.

### 4.1 Stage and positioning — `PositionerController`

| Function | Notes |
|---|---|
| `getPositionerNames()` | List of configured positioners |
| `getPositionerPositions()` | `{name: {X: µm, Y: …, Z: …, A: …}}` — **call this first** |
| `movePositioner(positionerName, axis="X", dist, isAbsolute=False, isBlocking=False, speed=None)` | Distances in **µm** |
| `movePositionerXYZ(positionerName, x, y, z, a, isAbsolute, isBlocking, speed)` | Coordinated multi-axis move |
| `setPositioner(positionerName, axis, position)` | Sets the *reported* position; does not move |
| `setPositionerSpeed(positionerName, axis, speed)` | |
| `setPositionerStepSize(positionerName, stepSize)` | |
| `stepPositionerUp/Down(positionerName, axis)` | One configured step |
| `movePositionerForever(positionerName, axis, speed, is_stop)` | Constant-velocity; **you must stop it** |
| `movePositionerForeverXYZA(...)` | Same, all axes |
| `stopAxis(positionerName, axis)` / `stopAllAxes(positionerName)` | Emergency stop for a move |
| `homeAxis(positionerName, axis, isBlocking, homeDirection, homeSpeed, …)` | **Ask first.** Drives to endstop |
| `getHomingStatus()` / `dismissHomingRecommendation()` | |
| `startFrameHoming` / `cancelFrameHoming` / `getFrameHomingState` | Full-frame homing routine |
| `moveToSampleLoadingPosition(positionerName, speed, is_blocking)` | **Ask first** |
| `moveToTransportPosition` / `getTransportPosition` / `setTransportPosition` | **Ask first** |
| `setMotorsEnabled(positionerName, is_enabled)` / `enalbeMotors(enable, enableauto)` | Note the upstream typo in the second name |
| `getStageOffsetAxis` / `setStageOffsetAxis` / `resetStageOffsetAxis` | User-frame ↔ device-frame offset |
| `getDevicePositionAxis` / `getTruePositionerPositionWithoutOffset` | Raw device frame |
| `startStageScan(xstart, xstep, nx, ystart, ystep, ny, tsettle)` (POST) / `stopStageScan` | Firmware-timed raster |
| `startZStageSync` / `cancelZStageSync` / `getZStageSyncState` | |

**Coordinate frames.** ImSwitch reports positions in a *user frame* = device frame + stage
offset. `getPositionerPositions` gives the user frame; `getDevicePositionAxis` gives the raw
device frame. Absolute moves are interpreted in the user frame. If the two disagree,
investigate the offset before moving.

### 4.2 Cameras and detectors — `SettingsController`

| Function | Notes |
|---|---|
| `getDetectorNames()` | |
| `getDetectorParameters()` / `getDetectorParameter(detectorName, parameterName)` | |
| `getDetectorParameterTree(detectorName)` | Structured, typed parameter tree — best for agents |
| `setDetectorParameterValue(body)` (POST) | Typed setter |
| `setDetectorParameter(detectorName, parameterName, value)` | Generic setter |
| `setDetectorExposureTime(detectorName, exposureTime)` | |
| `setDetectorGain` / `setDetectorBlackLevel` | |
| `setDetectorBinning(detectorName, binning)` / `getDetectorSupportedBinnings` | |
| `setDetectorROI(detectorName, frameStart, shape)` | |
| `setDetectorMode(detectorName, isAuto)` / `setDetectorExposureOnce` | Auto-exposure control |
| `getDetectorTriggerTypes` / `getDetectorCurrentTriggerType` / `setDetectorTriggerType` | `Software`, hardware modes |
| `sendSoftwareTrigger()` | |
| `getCameraStatus(detectorName)` | Health / connection |
| `setWhiteBalance(mode, detectorName)` / `getWhiteBalance` / `setColourGains(redGain, blueGain)` | Colour cameras |
| `setDetectorPreviewMinValue` / `MaxValue` / `MinMaxValue` | Display range only, not acquisition |
| `getStreamParams` / `setStreamParams(compression, subsampling, throttle_ms)` (POST) | Preview stream tuning |

Backends live in `model/managers/detectors/`: `HikCamManager`, `ToupCamManager`,
`BaslerManager`, `TucsenCamManager`, `GXPIPYManager` (Daheng), `PCOManager`,
`PhotometricsManager`, `AndorCamManager`, `Picamera2Manager`, `OpenCVCamManager`,
`MMCoreDetectorManager` (Micro-Manager), `VirtualCameraManager`.

### 4.3 Illumination — `LaserController`, `LEDController`, `LEDMatrixController`

`LaserController` covers lasers **and** simple LEDs — anything with a scalar intensity:

- `getLaserNames()`, `getLaserValueRanges(laserName)` — **always read the range first**
- `getLaserValue` / `setLaserValue(laserName, value)`
- `getLaserActive` / `setLaserActive(laserName, active)`
- `setLaserChannelIndex` / `getLaserChannelIndex`

`LEDMatrixController` drives the UC2 8×8 RGB matrix:

- `setAllLEDOn` / `setAllLEDOff` / `setAllLED(state, intensity, intensity_r/g/b)`
- `setLED(LEDid, state)`, `setIntensity(intensity)`
- `setRing(ringRadius, intensity, r, g, b)`, `setCircle(...)` — darkfield / oblique
- `setHalves(intensity, direction, r, g, b)` — DPC half-field illumination
- `setStatus(status)` — status indicator, not sample illumination

### 4.4 Live view and streaming — `LiveViewController`

- `startLiveView(detectorName, protocol="jpeg", params, force)` (POST) — protocols: `jpeg`,
  binary, WebRTC
- `stopLiveView(detectorName, stopCamera=True)`
- `getLiveViewActive()`, `getActiveStreams()`, `getStreamStatus()`,
  `getCurrentStreamProtocol()`
- `setStreamParameters(protocol, params)` (POST) / `getStreamParameters`
- `setDetectorStreamParameters(detectorName, params)` (POST)
- `getStreamDiagnostics(detectorName)` — FPS, drops, backpressure
- `mjpeg_stream(startStream=True, detectorName)` — plain MJPEG over HTTP
- `getLongExposureInfo` / `stopLiveViewForLongExposure` — live view must yield for exposures
  longer than the stream period
- `webrtc_offer(request)` (POST) / `webrtc_ice_candidate`

### 4.5 Snap and record — `RecordingController`

`SaveFormat` (`model/io/recording_service.py`): `TIFF=1`, `ZARR=3`, `MP4=4`, `PNG=5`,
`JPG=6`, `OME_TIFF=7`, `OME_ZARR=8`, `STITCHED_TIFF=9`.

- `startSnap(fileName, saveFormat=1, returnPreview=True, previewMaxSize=1024)` — async job;
  poll `getSnapStatus(jobId)`, abort with `cancelSnap(jobId)`
- `snapImageToPath(fileName, saveFormat, …)` — snap straight to disk
- `snapImage(output=False, toList=True)` — return pixels in the response
- `snapNumpyToFastAPI(detectorName, resizeFactor)` — single frame as an image response
- `startRecording(mSaveFormat=1, fileName)` / `stopRecording()`
- `isRecording()`, `getRecordingStatus()`, `getRecordingDuration()`,
  `getRecordingFrameCount()`

### 4.6 Focus — `AutofocusController`, `FocusLockController`

Software autofocus (`AutofocusController`), all Z ranges in µm:

- `autoFocus(rangez=100, resolutionz=10, defocusz=0, tSettle=0.1, nGauss=0, nCropsize=2048,
  focusAlgorithm="LAPE", twoStage=False, …)` — sweep + metric
- `autoFocusFast(sweep_range=150.0, speed, axis, focusAlgorithm, …)` — continuous sweep
- `autoFocusHillClimbing(initial_step=20.0, min_step=1.0, step_reduction=0.5,
  max_iterations=50, …)`
- `stopAutofocus()`, `getAutofocusStatus()`
- `startLiveMonitoring(period, method, nCropsize)` / `stopLiveMonitoring` /
  `setLiveMonitoringParameters` / `getLiveMonitoringStatus` — continuous focus-metric readout

Hardware focus lock (`FocusLockController`, astigmatism/reflection based):

- `startFocusMeasurement` / `stopFocusMeasurement`, `enableFocusLock(enable)`,
  `isFocusLocked()`, `unlockFocus()`, `toggleFocus(toLock)`
- `getFocusLockState()`, `getCurrentFocusValue()`
- `getFocusLockParams` / `setFocusLockParams`, `getPIParameters` / `setPIParameters(kp, ki)`
- `runFocusCalibrationDynamic(scan_range_um, num_steps, settle_time, …)` /
  `stopFocusCalibration` / `getCalibrationStatus` / `getCalibrationResults`
- `setZStepLimit(limit_nm)` / `getZStepLimit` — **the safety clamp on lock-driven Z moves.
  Do not raise it to make the lock converge.**
- `returnLastImage` / `returnLastCroppedImage` — the spot image, for diagnosing a bad lock

### 4.7 Objectives — `ObjectiveController`

- `getCurrentObjective()`, `getstatus()`, `isSlotConfigured(slot)`
- `moveToObjective(slot, skipZ=False)` — **ask first**; the turret sweeps across the sample
- `calibrateObjective(homeDirection, homePolarity)` — **ask first**
- `setObjectiveParameters(objectiveSlot, pixelsize, objectiveName, NA, magnification, …)`
- `setPositions(x0, x1, z0, z1, isBlocking)`, `setObjectivePosition(slot, position)`,
  `saveCurrentZPosition(slot)`, `setMoveSpeed(speed)`

Objective slots carry the **pixel size**, which every downstream stitching and coordinate
calculation depends on. Changing an objective without updating its pixel size silently
corrupts scan geometry. See `docs/OBJECTIVE_INDEXING_CONVENTION.md`.

### 4.8 Experiments and multi-dimensional acquisition — `ExperimentController`

The main high-level acquisition entry point. Models are in
`controller/controllers/experiment_controller/models.py`.

Capabilities and setup:

- `getHardwareParameters()` — `ExperimentWorkflowParams`: illumination sources, ranges,
  synthetic LED-matrix channels. **Start here**; it tells you what an experiment may request.
- `getDetectorPixelSize()`, `getLabwareList()`, `getLabwareDefinition(load_name, offset_x_um,
  offset_y_um)` — Opentrons-style labware
- `selectWellsByPattern(request)` / `applyWellSelectionToExperiment(request)`

Running:

- `startWellplateExperiment(mExperiment: Experiment)` — the primary call. `Experiment` =
  `{name, parameterValue: {illumination, illuIntensities, timeLapsePeriod, numberOfImages,
  autoFocus, …}, pointList: [{name, x, y, z, iX, iY, wellId, …}], scanAreas?, scanMetadata?,
  focusMap?, timepoints}`
- `getExperimentStatus()`, `pauseWorkflow()`, `resumeExperiment()`, `stopExperiment()`,
  `forceStopExperiment()`
- `homeAllAxes()` — **ask first**
- `startFastStageScanAcquisition(xstart, xstep, nx, ystart, ystep, ny, zstart, …)` /
  `stopFastStageScanAcquisition` / `startFastStageScanAcquisitionFilePath` — firmware-timed,
  fastest path
- `getLastScanAsOMEZARR()`

useq-schema MDA:

- `get_mda_capabilities()`, `get_mda_sequence_info(request)`,
  `start_mda_experiment(request: MDASequenceRequest)`, `run_native_mda_sequence(sequence_dict)`
- Examples in `examples/mda_demo.py`, `examples/native_useq_mda_example.py`; docs in
  `docs/REST_API_MDA.md`, `docs/NATIVE_USEQ_MDA.md`

Focus maps (tilt/curvature compensation over a large scan):

- `computeFocusMap(focusMapConfig, group_id)`, `measureFocusMapFromPoints`,
  `computeFocusMapFromPoints(request)`
- `getFocusMap(group_id)`, `getFocusMapPreview(group_id, resolution)`, `clearFocusMap`,
  `interruptFocusMap`, `saveFocusMaps(path)`, `loadFocusMaps(path)`

Overview / slide registration and stitching:

- `snapOverviewImage(slot_id, camera_name)`, `recaptureSlot`, `refreshOverviewSlideImage`
- `registerOverviewSlide(registration_data)`, `getOverviewRegistrationStatus`,
  `getOverviewOverlayData`, `getOverviewOverlayImage`
- `runAutonomousOverviewScan(camera_name, layout_name, settle_time_s, …)`
- `getKnownCalibrationLayouts`, `getKnownCalibrationPoint`,
  `getOverviewRegistrationConfig` / `…ConfigData` / `updateOverviewRegistrationConfig`
- `runAshlarStitching(pixelSize, maximumShift, alignChannel, experimentDir)` /
  `stopAshlarStitching`

OMERO export: `getOMEROConfig` / `setOMEROConfig` / `isOMEROEnabled` /
`getOMEROConnectionParams` / `getOMEWriterConfig`.

### 4.9 Workflows and tiling — `WorkflowController`, `StageMapController`, `TilingController`

`WorkflowController` — generic step-based execution engine:

- `uploadWorkflow(definition)`, `start_workflow_api(request)`
- `start_xyz_histo_workflow(x_min, x_max, y_min, y_max, …)`,
  `start_xyz_histo_workflow_by_list(req)`
- `startWorkflowTileBasedByParameters(numberTilesX, numberTilesY, stepSizeX, stepSizeY,
  nTimes, tPeriod, …)`
- `pause_workflow` / `resume_workflow` / `stop_workflow` / `force_stop_workflow` /
  `workflow_status`, `getWorkflowStatus` / `setWorkflowStatus`
- `computeOptimalScanStepSize2(overlap=0.75)` — derives step size from FOV and overlap; use
  it instead of guessing

`StageMapController` — live low-magnification map built while you move:

- `startStageMap` / `stopStageMap` / `clearStageMap`, `snapStageMapTile`
- `getStageMapParams` / `setStageMapParams`, `getStageMapStatus`, `setStageMapChannel`
- `getStageMapTiles(fromId, includePreviews)`, `gotoStagePosition(x, y, isAbsolute,
  isBlocking)`, `saveStitchedOmeTiff(filename)`

### 4.10 Metadata, storage and files

`MetadataController` — the metadata hub feeding OME-TIFF/OME-Zarr writers:

- `getMetadataSnapshot(flat, category)`, `getMetadataCategories`, `getMetadataJSON`
- `getDetectorContext(detectorName)`, `getAllDetectorContexts`,
  `getFrameEvents(detectorName, maxEvents)`, `getLatestFrameEvent`
- `getInstrumentInfo`, `getOMEInstrument`, `getInstrumentComponents`, `getInstrumentFilters`
- `setTubeLens(focalLengthMm, magnification)`, `setFirmwareVersion`,
  `loadUC2OptiKitConfig(configPath)`
- `listActiveSessions(basePath, limit)`, `getSessionMetadata(sessionId)`,
  `getZarrStoreUrl(sessionPath)`, `getCurrentSessionInfo`

`StorageController`:

- `get_storage_status()`, `list_external_drives()`, `set_active_path(path, persist)`,
  `get_config_paths()`, `update_config_paths(config_path, data_path, persist)`

File manager (plain FastAPI routes, not `@APIExport`) under `/imswitch/api/FileManager/…`:
list, `folder` (POST), `upload`, `download/{path}`, rename, copy/move, delete.

Logs: `/imswitch/api/LogController/listLogFiles`, `…/downloadLogFile?filename=…`.

### 4.11 Board, firmware and OTA — `UC2ConfigController`

The largest controller (~80 endpoints). **Read-only calls are fine; everything that writes to
the board needs confirmation.**

Safe to read:

- `uc2_board_is_connected(strict)`, `getFirmwareInfo()`, `getBusStatus()`,
  `getMicroscopeStandName()`, `getBoardTemperature()`, `getFanState()`, `getGpioStatus()`
- `listSerialPorts()`, `get_canbus_devices(timeout)`, `scan_canbus(timeout, probe_range)`
- `getMotorSettings()`, `getMotorSettingsForAxis(axis)`, `getTMCSettingsForAxis(axis)`
- `getCollisionState()`, `getJoystickDirection()`, `getPtzStatus()`, `getPtzMapping()`,
  `listPtzActions()`
- `getDataPath()`, `isImSwitchRunning()`, `getOTAStatus`, `getOTADeviceMapping`,
  `listAvailableFirmware`, `listAllFirmwareFiles`, `getUSBFlashStatus`

Confirm before calling:

- Motor tuning: `setMotorSettings`, `setMotorSettingsForAxis(axis, settings)`,
  `setTMCSettingsForAxis`, `setGlobalMotorSettings`, `applyMotorSettingsToDevice`
- Safety: `setCollisionThreshold`, `setCollisionSensitivity`, `setCollisionReference`,
  `calibrateCollisionReference`, `setCollisionMode`, `armCollisionProtection`,
  `resetCollisionAlarm`, `confirmSafeHoming`, `setBusPower`
- Transport: `reconnect(port, baudrate)`, `setSerialConfig(port, baudrate, persist)`,
  `writeSerial(payload)`, `probeDeviceState`
- Triggers/GPIO: `setDigitalOut`, `setupDigitalOutPin`, `setTrigger(...)`, `sendTrigger`,
  `resetTriggerTable`, `getDigitalIn`, `actDigitalIn`
- Lifecycle: `espRestart`, `restartCANDevice`, `stopImSwitch`, `restartImSwitch`,
  `moveToSampleMountingPosition`
- **Firmware — highest risk:** `flashMasterFirmwareUSB(port, match, baud, firmware_filename)`,
  `cancelUSBFlash`, `sendCanAddress`, `reassignCANId(new_id, mac, target)`,
  `startSingleDeviceOTA(can_id, ssid, password, timeout)`, `startMultipleDeviceOTA`,
  `startCANStreamingOTA(can_id, firmware_url, baud)`, `startMultipleCANStreamingOTA`,
  `cancelCANStreamingOTA`, `setOTAWiFiCredentials`, `setOTAFirmwareServer`,
  `clearOTAFirmwareCache`

Setup files (defined directly on the server, not via `@APIExport`):
`returnAvailableSetups`, `getCurrentSetupFilename`, `readSetupFile(setupFileName)`,
`writeNewSetupFile(...)` (POST), `setSetupFileName(setupFileName, restartSoftware)`,
`getDiskUsage`, `is_connected`.

Firmware/OTA background: `docs/CAN_OTA_UPDATE_GUIDE.md`, `docs/OTA_API_QUICKREF.md`,
`docs/CAN_OTA_FIRMWARE_SERVER.md`.

### 4.12 Specialised imaging modes

Present only when the setup enables them. Discover with `getAvailableControllers`.

| Controller | Purpose |
|---|---|
| `SIMController` | Structured illumination microscopy |
| `STORMReconController` | Single-molecule localisation / STORM |
| `DPCController` | Differential phase contrast (LED-matrix halves) |
| `InLineHoloController`, `OffAxisHoloController`, `HoloController`, `HoliSheetController` | Digital holography |
| `LightsheetController` | Light-sheet scanning |
| `GalvoScannerController` | Galvo scanning, incl. FLIM Labs bridge |
| `FLIMLabsController` | FLIM acquisition |
| `ISMController` | Image scanning microscopy |
| `FlowStopController` | Flow-cell / imaging-flow-cytometry |
| `PixelCalibrationController`, `StageCenterCalibrationController` | Pixel size and stage/camera affine calibration |
| `ReadNoiseCalibrationController` | Camera noise characterisation |
| `TimelapseController`, `RotationScanController`, `ROIScanController`, `TriggerAcquisitionController` | Acquisition patterns |
| `MMCoreController` | Micro-Manager device layer |
| `WellPlateController` (`moveToXY(wellID)`), `SquidStageScanController`, `StageScanAcquisitionController` | Plate/stage scanning |
| `HyphaController`, `ArkitektController`, `SiLa2Controller`, `WebRTCController` | External integrations |
| `StresstestController`, `DebugController`, `DemoController`, `AcceptanceTestController` | Diagnostics and self-test |

---

## 5. Agent role definitions

Roles follow the same shape as Optiland's `agents.md`: scope, commands, stack,
responsibilities, boundaries. Adopt exactly one per task.

### `imswitch_operator_agent`

**Drives a running microscope over the HTTP API. Does not write repository code.**

This is the role for "take an image", "run this scan", "what is the stage doing". It is the
role with physical consequences.

- **Reads/writes:** the HTTP API of one ImSwitch server; the data folder (read).
  Never edits repository source.
- **Stack:** `requests` / `httpx`, `python-socketio`, `numpy`, `tifffile`, optionally
  `imswitchclient`.
- **Responsibilities**
  - Bootstrap every session: `version` → `getAvailableControllers` → `openapi.json` →
    `getPositionerNames` / `getDetectorNames` / `getLaserNames` / `getCurrentObjective`.
  - Read state before writing it; re-read after, and report the delta.
  - Use blocking calls (`isBlocking=True`) or poll status endpoints. Never assume a move
    finished because the HTTP call returned.
  - Turn illumination **off** when done. Stop recordings. Stop `movePositionerForever`.
  - Prefer high-level entry points (`startWellplateExperiment`, `start_mda_experiment`) over
    hand-rolled move/snap loops — they handle metadata, focus maps, and storage layout.
- **Always:** confirm before absolute moves, any Z move, homing, objective changes, and laser
  power-up. Verify the loaded setup matches the hardware the user described.
- **Ask first:** anything in [§0 Ask first](#ask-first).
- **Never:** touch firmware/OTA endpoints; disable safety limits; leave a laser on or a
  continuous move running at the end of a task.

### `imswitch_backend_agent`

**Python engineer for controllers, managers, and the FastAPI server.**

- **Reads/writes:** `imswitch/imcontrol/controller/`, `imswitch/imcontrol/model/`,
  `imswitch/imcommon/`. Reads `frontend/` and `_test/`.
- **Stack:** Python ≥ 3.11, FastAPI, uvicorn, pydantic v2, psygnal, numpy, dask, zarr,
  tifffile, `python-socketio`.
- **Responsibilities**
  - Keep the layering intact: controllers orchestrate, managers own hardware. HTTP concerns
    stay in the controller; device I/O stays in the manager.
  - New API method → `@APIExport`, typed signature, docstring. Check the method name is
    unique across all controllers (`generateAPI` raises `NameError` otherwise).
  - Use `requestType="POST"` for anything with a structured payload or non-trivial side
    effects; define a pydantic model for the body.
  - Never block the event loop. Long acquisitions run on worker threads and report via
    signals; expose a status endpoint and a stop endpoint for anything long-running.
  - Signals broadcast to the frontend through `imcommon/framework/noqt.py`. Adding a signal
    means adding traffic — throttle high-rate ones.
- **Always:** verify against `example_virtual_microscope.json` before claiming a change works.
- **Ask first:** changing an existing endpoint's name, path, or response shape (the React SPA
  and external clients depend on it); adding a dependency to `pyproject.toml`.
- **Never:** add Qt/PySide code — this fork is headless-only. Never widen a shared base class
  to accommodate one device.

### `imswitch_hardware_agent`

**Device integration: cameras, stages, lasers, LED matrices, transports.**

- **Reads/writes:** `imswitch/imcontrol/model/managers/**`, the setup JSONs under
  `imswitch/_data/user_defaults/imcontrol_setups/`. Reads controllers.
- **Stack:** `uc2rest`, `pyserial`, `pymmcore-plus`, vendor SDKs, `numpy`.
- **Responsibilities**
  - New device → new manager subclassing the right base (`DetectorManager`,
    `PositionerManager`, `LaserManager`, …), registered by `managerName` in the setup JSON.
    `TEMPLATECamManager.py` is the worked example; see `docs/adding-device-support.rst`.
  - Vendor SDK imports must be **lazy and guarded** — an uninstalled SDK must not break
    startup for every other user.
  - Every manager needs a mock/virtual counterpart or a documented degradation path.
  - Respect units: stage distances are **µm** at the ImSwitch layer, **steps** at the
    firmware layer. Conversion (`stepSize`, direction sign, backlash) belongs in the manager
    or in `uc2rest`, never sprinkled through controllers.
- **Always:** test with the device absent as well as present.
- **Ask first:** editing setup JSONs that describe a user's real instrument; changing
  `stepSize`, `homeDirection`, `homeEndstoppolarity`, or backlash values.
- **Never:** hardcode COM ports, IPs, or serial numbers into managers — those are setup-JSON
  properties.

### `imswitch_frontend_agent`

**React SPA developer.**

- **Reads/writes:** `frontend/`. Reads controllers to learn the API contract.
- **Stack:** React, Create React App via **craco**, axios, socket.io-client, MUI.
- **Commands** (run from `frontend/`):

```bash
npm install
npm start
npm run build
```

- **Responsibilities**
  - API calls go through the shared axios instance
    (`frontend/src/backendapi/createAxiosInstance.js`), base URL
    `${ip}:${apiPort}/imswitch/api`. Do not hand-build URLs in components.
  - One thin wrapper per endpoint in `frontend/src/backendapi/`; components call the wrapper.
  - Handle the "controller not available" case — the API surface varies by setup.
  - The built bundle must be copied/served from the ImSwitch server; the SPA is not deployed
    independently.
- **Note:** the locally installed `babel` is v5 and useless for syntax checking. Parse-check
  with `esbuild`; build with `craco`. Unix-style env-var prefixes in `package.json` scripts
  fail on Windows `cmd`.
- **Ask first:** adding npm dependencies; changing the connection-settings model.
- **Never:** change backend endpoint names to fit the UI — fix the UI, or coordinate a
  deliberate API change.

### `imswitch_test_agent`

**QA engineer. Tests must run with no hardware attached.**

- **Reads/writes:** `imswitch/imcontrol/_test/`, `examples/conftest.py`. Reads application
  code.
- **Commands:**

```bash
pytest imswitch/imcontrol/_test/unit
```

```bash
pytest imswitch/imcontrol/_test/api -m "not hardware"
```

- **Stack:** pytest ≥ 7. Markers declared in `pytest.ini`: `unit`, `integration`, `slow`,
  `hardware`. `addopts` already disables the `arkitekt_next` plugin — keep that.
- **Responsibilities**
  - Every test runs against virtual devices (`VirtualCameraManager`, `VirtualStageManager`,
    `MockPositionerManager`) or mocks. Mark anything else `@pytest.mark.hardware`.
  - API tests assert on the HTTP contract — path, status, response shape — because the React
    SPA and external clients depend on it.
  - Cover unit conversion, coordinate frames, and offset arithmetic explicitly. These are
    where silent physical errors originate.
- **Always:** run the narrowest relevant selection first; the full suite is slow.
- **Never:** write a test that moves real hardware without the `hardware` marker.

### `imswitch_lint_agent`

**Style and static cleanliness. Never changes behaviour.**

- **Commands:**

```bash
ruff check imswitch
```

```bash
ruff format imswitch
```

- **Config** (`pyproject.toml`): line length 100; rules `E`, `F`, `W`, `C90`; max McCabe
  complexity 10; `F401` ignored in `__init__.py`. Excluded: `imswitch/_data`, `build`,
  `ImTools`, `imswitch/imcontrol/model/interfaces/pyicic`.
- **Never:** "fix" a lint error by altering logic; never reformat files unrelated to the
  current change; never widen the exclude list to silence findings.

### `imswitch_docs_agent`

**Technical writer for `docs/` and docstrings.**

- **Reads:** all source. **Writes:** `docs/`, `README.md`, this file, and docstrings.
- **Stack:** Sphinx (`docs/conf.py`), Markdown, reStructuredText.
- **Responsibilities**
  - Every `@APIExport` method gets a docstring stating units, coordinate frame, blocking
    behaviour, and side effects. For this codebase, **units and frames are the documentation**
    — a docstring that omits them is incomplete.
  - New feature → a `docs/*.md` page, and a line in `docs/ImSwitch_Functionality_Overview.md`.
  - Keep [§4](#4-function-reference) of this file in sync when endpoints are added or renamed.
- **Never:** document intended behaviour as if it were implemented; verify against source.

### `uc2rest_agent`

**Direct USB/serial control of a UC2 board, with ImSwitch not running.** See [§6](#6-uc2-rest-standalone).

---

## 6. UC2-REST standalone

`uc2rest` is the Python client for UC2/ESP32 microcontroller firmware. ImSwitch depends on it
(`uc2-rest` in `pyproject.toml`), but it is usable entirely on its own — for bench tests,
firmware bring-up, or a rig with no microscope stack.

Repo: <https://github.com/openUC2/UC2-REST> · Firmware: <https://github.com/youseetoo/uc2-esp32>

### 6.1 Connect

```bash
pip install uc2rest
```

```python
import uc2rest

esp = uc2rest.UC2Client(serialport="COM3")          # Windows
# esp = uc2rest.UC2Client(serialport="/dev/ttyUSB0")  # Linux / macOS

print(esp.is_connected)
print(esp.state.get_firmware_info())
```

Constructor arguments worth knowing:

| Argument | Meaning |
|---|---|
| `serialport` | Port name. **Required** — this is the only transport |
| `baudrate` | Default `115200` |
| `identity` | Expected device identity string, default `"UC2_Feather"` |
| `NLeds` | LED-matrix pixel count, default `64` |
| `DEBUG` | Log every serial exchange |
| `skipFirmwareCheck` | Skip the version handshake |
| `device_id`, `requireMaster` | Select a specific board / require the CAN master |
| `SerialManager` | Browser transport (PyScript / Web Serial) |
| `isPyScript` | Running under PyScript |

**Wi-Fi/HTTP has been removed.** The `host` / `port` arguments are deprecated and ignored;
passing `host=` without `serialport=` only logs a warning and leaves you unconnected. Serial
is the only supported transport.

On Linux the user must be in the `dialout` group:

```bash
sudo usermod -a -G dialout $USER
```

### 6.2 Device model

Every hardware block is an attribute on the client
(`uc2rest/UC2Client.py`, one module per block):

| Attribute | Module | Representative calls |
|---|---|---|
| `esp.motor` | `motor.py` | `move_x/y/z/t`, `move_xy`, `move_xyz`, `move_xyzt`, `move_axis_by_name`, `move_forever`, `stop`, `get_position`, `set_position`, `setup_motor`, `set_backlash`, `set_motor_acceleration`, `set_motor_enable`, `startStageScanning`, `startFocusScanning`, `setTrigger`, `setAxisMode`, `getAxisFeedback`, `calibrateAxis` |
| `esp.home` | `home.py` | `home_x/y/z/a`, `home_xy`, `home(axis, …)`, `stop_home` |
| `esp.led` | `ledmatrix.py` | `setAll`, `setSingle`, `setIntensity`, `setPattern`, `send_LEDMatrix_rings`, `send_LEDMatrix_circles`, `send_LEDMatrix_halves`, `send_LEDMatrix_off`, `send_LEDMatrix_status` |
| `esp.laser` | `laser.py` | `set_laser(channel, value)`, `set_laserpin`, `set_servo` |
| `esp.galvo` | `galvo.py` | `set_dac`, `set_galvo_scan`, `stop_galvo_scan`, `set_position`, `set_arbitrary_points`, `generate_circle_points`, `generate_grid_points`, `set_trigger_mode` |
| `esp.objective` | `objective.py` | `home`, `calibrate`, `move(slot)`, `toggle`, `setPositions`, `getstatus` |
| `esp.rotator` | `rotator.py` | filter wheel / rotation stage |
| `esp.gripper` | `gripper.py` | open / close |
| `esp.state` | `state.py` | `get_state`, `get_firmware_info`, `is_master`, `isBusy`, `set_power`, `get_estop`, `set_estop_polarity`, `register_emergency_callback`, `espRestart`, `pairBT`, `getHeap` |
| `esp.can` / `esp.canota` | `can.py`, `canota.py` | CAN bus scan/addressing; OTA firmware updates |
| `esp.digitalout` / `esp.digitalin` / `esp.gpio` | | TTL I/O |
| `esp.analog` | `analog.py` | analogue outputs |
| `esp.temperature` / `esp.fan` | | NTC sensor, fan control |
| `esp.i2c` / `esp.lcd` / `esp.ptz` | | I²C bus, LCD, pan-tilt-zoom |
| `esp.camera` / `esp.camera_trigger` | | ESP32-camera and trigger lines |
| `esp.wifi` | `wifi.py` | ESP32 network helpers |
| `esp.message` | `message.py` | generic callbacks / triggers |
| `esp.cmdRecorder` | `cmdrecorder.py` | record and replay command sequences |

### 6.3 Protocol

Three JSON endpoints per device block, sent as newline-delimited JSON over serial:

```
/<device>_act   perform an action   (move, flash, trigger)
/<device>_set   configure           (speed, pin mapping, limits)
/<device>_get   query state         (position, temperature, status)
```

A motor move on the wire:

```json
{"task": "/motor_act",
 "motor": {"steppers": [{"stepperid": 1, "position": 1000, "speed": 15000,
                         "isabs": 0, "isaccel": 1, "isen": 1, "accel": 40000}]}}
```

Axis indexing at the firmware layer is `XYZT → 1,2,3,0` (`xyztTo1230`). Responses are JSON;
blocking calls wait for `nResponses` messages or a QID-completion marker.

Raw access when you need it:

```python
esp.serial.post_json("/motor_get", {"task": "/motor_get"}, getReturn=True, timeout=1)
```

### 6.4 Units, direction, backlash

`uc2rest` operates in **steps**, not µm. `Motor` applies per-axis `stepSize`, direction sign,
and backlash compensation on top (`move_stepper`), so wiring polarity stays hidden from
callers. µm ↔ step conversion for the microscope lives in ImSwitch's `ESP32StageManager`, not
here. When comparing an ImSwitch position with a `uc2rest` position, expect them to differ by
the step size and the stage offset — that is not a bug.

### 6.5 Working without hardware

`uc2rest/MockSerial.py` provides a fake serial device. `uc2rest/TEST/` holds ~20 runnable
scripts (`TEST_ESP32_LEDarray.py`, `TEST_HOME_XY.py`, `TEST_ESP32_Serial.py`, …) that double
as usage examples. `DOCUMENTATION/` has notebooks (`DOC_UC2Client.ipynb`,
`DOC_UC2Client-PinConfigurator.ipynb`) and protocol notes
(`HARD_LIMITS_DOCUMENTATION.md`, `CAN_OTA_STREAMING_PROTOCOL.md`, `QID_TRACKING_DOCUMENTATION.md`).

### 6.6 Do not mix ImSwitch and UC2-REST

A serial port has one owner. If ImSwitch is running with an `ESP32Manager` / `ESP32StageManager`
bound to `COM3`, a second `UC2Client(serialport="COM3")` will either fail to open or corrupt
the command stream mid-move — a stage that is moving when its command stream breaks does not
stop cleanly.

Before opening a `UC2Client`, either:

- confirm no ImSwitch server is up — probe `GET /imswitch/api/version` (or
  `.../UC2ConfigController/isImSwitchRunning`, which is a bare liveness ping that always
  returns `true` when reachable). A connection refusal means ImSwitch is down; **or**
- go through ImSwitch instead: `UC2ConfigController` already exposes most board-level
  functionality (`writeSerial`, motor settings, CAN, GPIO, triggers, OTA) over HTTP.

**Rule: if ImSwitch is up, use the ImSwitch API. Use `uc2rest` directly only when ImSwitch is
down.**

### 6.7 `uc2rest_agent` role

- **Reads/writes:** `UC2-REST/uc2rest/`. Reads firmware docs.
- **Stack:** Python, `pyserial`, `numpy`.
- **Responsibilities:** wrap firmware endpoints in typed, documented Python; keep step/µm
  conversion and direction handling in one place; never raise on missing firmware features —
  degrade to a documented default (the encoder/axis helpers are the pattern to follow).
- **Always:** verify against `MockSerial` or a bench board, never a mounted sample.
- **Ask first:** any `esp.canota.*`, `esp.can.*` addressing change, `espRestart`, or
  `state.set_power`.
- **Never:** open a serial port that ImSwitch may already hold.

---

## 7. Recipes

### Snap one image from a running server

```python
import requests
B = "http://localhost:8001/imswitch/api"

det = requests.get(f"{B}/SettingsController/getDetectorNames").json()[0]
requests.get(f"{B}/SettingsController/setDetectorExposureTime",
             params={"detectorName": det, "exposureTime": 50})
r = requests.get(f"{B}/RecordingController/snapNumpyToFastAPI",
                 params={"detectorName": det})
open("snap.png", "wb").write(r.content)
```

### Move the stage safely

```python
pos = requests.get(f"{B}/PositionerController/getPositionerPositions").json()
name = next(iter(pos))
print("before:", pos[name])

requests.get(f"{B}/PositionerController/movePositioner",
             params={"positionerName": name, "axis": "X",
                     "dist": 100, "isAbsolute": False, "isBlocking": True})

print("after:", requests.get(f"{B}/PositionerController/getPositionerPositions").json()[name])
```

Relative, small, blocking, verified. That is the pattern.

### Start and monitor a wellplate experiment

```python
hw = requests.get(f"{B}/ExperimentController/getHardwareParameters").json()

exp = {
  "name": "demo",
  "timepoints": 1,
  "parameterValue": {"illumination": [hw["illuSources"][0]], "illuIntensities": [10],
                     "timeLapsePeriod": 0, "numberOfImages": 1, "autoFocus": False},
  "pointList": [{"name": "p0", "x": 0.0, "y": 0.0, "iX": 0, "iY": 0}],
}
requests.post(f"{B}/ExperimentController/startWellplateExperiment", json=exp)

while True:
    st = requests.get(f"{B}/ExperimentController/getExperimentStatus").json()
    print(st)
    if st.get("status") in ("idle", None):
        break
```

`getExperimentStatus` returns `{"status": "idle" | "running" | "paused" | "stopping"}`; in
performance mode it also carries `running`, `frames_received`, and `expected_frames`.

Stop with `stopExperiment` (graceful) or `forceStopExperiment` (immediate).

### Subscribe to signals

```python
import socketio, msgpack

sio = socketio.Client(ssl_verify=False)

@sio.on("signal_msgpack")
def on_signal(data):
    print(msgpack.unpackb(data, raw=False))

sio.connect("http://localhost:8001", socketio_path="/socket.io")
sio.wait()
```

### Add an endpoint (backend)

1. Pick a unique method name — grep `imswitch/imcontrol/controller/controllers/` first.
2. Add to the relevant controller with a typed signature and a docstring naming units and
   frame.
3. Decorate: `@APIExport()` for a read, `@APIExport(requestType="POST")` for a structured
   write.
4. Restart the server; confirm it appears in `/imswitch/openapi.json`.
5. Add a test under `imswitch/imcontrol/_test/api/`.
6. Add a wrapper in `frontend/src/backendapi/` if the UI needs it.

---

## 8. Failure modes

| Symptom | Likely cause |
|---|---|
| `SSLError` / `CERTIFICATE_VERIFY_FAILED` | HTTPS with the bundled self-signed cert. Use `verify=False`, `curl -k`, or start with `--no-ssl` |
| 404 on an endpoint you expect | The controller is not instantiated — its hardware group is missing from the loaded setup. Check `getAvailableControllers` |
| `NameError: API method name "..." is already in use` at startup | Two controllers export the same method name. Exported names are global |
| Server starts but no camera | Vendor SDK missing or camera claimed by another process. Check the log via `LogController/listLogFiles` |
| `SerialException` on startup | Board not present, wrong port, or missing `dialout` membership |
| Stage reports a position that does not match reality | Stage offset (user frame vs device frame). Compare `getPositionerPositions` with `getDevicePositionAxis` |
| Move returns immediately, hardware still moving | `isBlocking=False`. Pass `isBlocking=True` or poll |
| Live view frozen during a long exposure | Expected — see `getLongExposureInfo` / `stopLiveViewForLongExposure` |
| Stitching misaligned | Wrong pixel size for the current objective, or a stale affine calibration |
| Frontend shows stale data | Socket.IO disconnected, or the signal is throttled/suppressed in `noqt.py` |

---

## 9. Reference

- Fork: <https://github.com/openUC2/ImSwitch> — upstream: <https://github.com/ImSwitch/ImSwitch> (unmaintained)
- UC2-REST: <https://github.com/openUC2/UC2-REST> · ESP32 firmware: <https://github.com/youseetoo/uc2-esp32>
- Configs: <https://github.com/openUC2/ImSwitchConfig> · OS image: <https://github.com/openUC2/rpi-imswitch-os>
- Upstream docs: <https://imswitch.readthedocs.io> · openUC2 docs: <https://openuc2.github.io>
- In-repo: `docs/ImSwitch_Functionality_Overview.md`, `docs/adding-device-support.rst`,
  `docs/imcontrol-setups.rst`, `docs/REST_API_MDA.md`, `docs/streaming_protocol_api.md`,
  `docs/metadata_hub.md`, `docs/FOCUS_MAP_DOCUMENTATION.md`, `docs/storage_management.md`
- Licence: ImSwitch GPL-3.0-or-later; UC2-REST LGPL-3.0-or-later
