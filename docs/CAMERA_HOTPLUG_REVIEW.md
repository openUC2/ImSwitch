# USB camera hotplug in ImSwitch: design review (read-only, nothing run, no hardware)

Paths relative to repo root. Verdict: feasible per vendor, but only the ToupTek driver has the plumbing today. Hik and Daheng need real work. Nothing in the backend or frontend handles "camera appears later".

## 0. Structural facts that shape the design

- **Detectors are built once.** `MultiManager.__init__` (managers/MultiManager.py:15-40) instantiates each manager from the setup JSON at startup. A manager that raises is dropped from `_subManagers` and `getDetectorNames` never lists it. `DetectorsManager` (managers/DetectorsManager.py:19-29) has no add/replace/rescan method and no status signal.
- **Controllers cache manager references.** At least 17 do (`self.detector = self._master.detectorsManager[...]`): Autofocus:300, StageMap:649, Objective:82, SIM:124, PixelCalibration:90, Lightsheet:129 and others. `StreamWorker._detector` (LiveViewController.py:188) holds one too, and `MetadataHub` registers the detectors at startup (MasterController.py:~290). Recreating a manager would leave all of these pointing at a dead object. **The only workable design swaps the camera handle inside the manager, keeping the manager instance.**
- **The manager copies camera facts at construction.** `DetectorManager.__init__` stores `__model`, `__fullShape`, `__parameters` and `__supportedBinnings` as name-mangled privates (DetectorManager.py:145-200). `HikCamManager.__init__` (HikCamManager.py:28-175) builds the parameter set from `self._camera`: exposure/gain min/max, image_width/height, frame_rate. The mock is 512x512 (tiscamera_mock.py:35-44) and a real Hik is about 1440x1080. A swap therefore needs a "rebuild shape/params from the new camera" step that does not exist. This is the largest piece of refactoring.
- **Every `CommunicationChannel` Signal is broadcast to Socket.IO** (noqt.py:91-135, `_safe_broadcast_message` :281-296), arriving at the frontend as `{name, args:{p0}}`. Precedent: `sigDiskFull` and `sigStageLimit` (CommunicationChannel.py:107-109, WebSocketHandler.js:888-915). A status signal costs about 10 lines.

## 1. Today, per camera

### HIK (HikCamManager.py, interfaces/hikcamera.py)

| Case | Behaviour | Code |
|---|---|---|
| Absent at start | `CameraHIK.__init__` raises `RuntimeError("No suitable Hik cameras found.")` (hikcamera.py:~282). The manager catches it and falls back to `MockCameraTIS` (HikCamManager.py:474-488, "loading TIS mocker"). **The mock is permanent.** `startAcquisition` has a comment only: `if model=="mock": 'We could attempt to reconnect'; pass` (:369-371). | |
| Status when mock | `getCameraStatus` reports `isMock = self._mocktype != "normal"` (:500). `mocktype` defaults to "normal", so **a fallback mock is reported as `isMock: False`**. `isConnected = hasattr(self._camera,'cam')` (:501). Neither `CameraHIK` nor the mock has a `.cam` attribute, so it is always False, even for a working camera. The status endpoint is unreliable in both directions. | |
| Unplug mid-run | Nothing is registered. `MV_CC_RegisterExceptionCallBack` is not used anywhere. `is_connected` is never cleared. The SDK callback just stops firing. `getLast` (hikcamera.py:966-1010) returns `frame_buffer[-1]`, which still holds the last 3 frames, or `lastFrameFromBuffer`. **Stale frame with a constant frame id, no error.** `StreamWorker` (LiveViewController.py:252-340) only stops on `frame is None and same frame number and attempts>3`. That never triggers, because the frame is not None. A recording or job would store duplicates, or time out in `snapSoftwareTrigger` (:1360-1375), which returns the stale `frame_buffer[-1]` after the timeout. | |
| Replug | No trigger exists. `CameraHIK.reconnectCamera()` (:371-395) exists but is only reached from `start_live` when `self.camera is None` (:706) or when `StartGrabbing` fails (:761). It has gaps: (a) it does not reset `is_streaming`, so `start_live` returns at its first line (:701) if the stream was running; (b) it does not re-apply exposure, gain, trigger source or ROI, only the `_open_camera` defaults; (c) it swallows exceptions and leaves `self.camera=None` on failure; (d) the `except Exception: raise RuntimeError(f"...{ret:x}")` at :765 reads a possibly unbound `ret`. | |
| Partial-init leak | If `CameraHIK.__init__` fails after `MV_CC_OpenDevice` (e.g. in `setBinning`), the handle stays open in exclusive mode and `_claimedChecksums` stays set (hikcamera.py:312). Every later open of that camera is then denied. Matters for any retry loop. | |
| Index/serial pinning | `_cameraCandidates` (:86-108): `cameraListIndex` <= 9 is an index, larger is a serial checksum. A checksum mismatch falls back to the first unclaimed camera, with a warning. Reconnect reuses `self.cameraNo` (:371-395), so the serial pin survives. | |

### Daheng / gxipy (GXPIPYManager.py, interfaces/gxipycamera.py)

| Case | Behaviour | Code |
|---|---|---|
| Absent at start | `update_device_list()` returns 0 and `CameraGXIPY` raises "No camera GXIPY connected" (gxipycamera.py:65-74). The manager falls back to `MockCameraTIS` (GXPIPYManager.py:386-403). Its `os.name == 'darwin'` guard is dead code, since `os.name` is "posix". | |
| Status | `isMock` is **hard-coded False** (:420) even on the mock. `isConnected = hasattr(self._camera,'camera')` (:421), also meaningless. | |
| Unplug | No offline callback is registered, although gxipy has one (see §2). `getLast` (gxipycamera.py:372-396) explicitly falls back to the last frame after the timeout (:381-388). Same stale-frame behaviour. | |
| Replug | **No reconnect code at all.** There is no `reconnectCamera`. `close()` (:187-203) only swallows exceptions. | |

### ToupTek (ToupCamManager.py, interfaces/toupcamcamera.py)

| Case | Behaviour | Code |
|---|---|---|
| Absent at start | `_open_camera` raises "No Toupcam camera found" (:272-274). The manager falls back to the mock (ToupCamManager.py:537-553). **Permanent**, same as Hik. | |
| Unplug | **Handled in the driver.** `TOUPCAM_EVENT_DISCONNECTED` goes to `_handleDisconnect()` (:553-555 -> :507-540). That sets `is_connected=False` and `is_streaming=False`, then starts a daemon thread `ToupcamReconnect`. It retries `reconnectCamera()` up to 30 times at 2 s intervals (~60 s), and resumes the stream (`start_live`) if it was streaming, otherwise `_startParked()`. | |
| Settings after replug | `reconnectCamera` (:456-473) closes the old handle, reopens it, then `_reapply_settings()` (:475-504) restores exposure, gain, blacklevel, binning, frame rate, trigger source, TEC, fan, conversion gain, low noise, heat and black-level auto-adjust. | |
| Gaps | (a) The thread gives up after 60 s and never recovers, and nothing is reported. (b) `frame_buffer` is not cleared on disconnect, so `getLast` (:763-800) still serves the last frame (the `is_streaming` early-out only applies to an empty buffer). (c) The manager is not informed. `getCameraStatus.isConnected` does read the driver flag (ToupCamManager.py:568), the one honest status of the three. (d) A reopen whose sensor mode or size changes (different binning, different model) is not propagated to manager shape/params. | |

### Cross-cutting

- `DetectorsManager.startAcquisition` (:79-105) calls `execOnAll(lambda c: c.startAcquisition())`. One camera raising (e.g. a dead handle) aborts the loop for every later detector, and the acquisition-handle bookkeeping is already committed.
- **A mock that silently takes over is a data-integrity risk.** Experiments, Aufgaben jobs and recordings do not check `isMock`. A run with an absent camera writes the mock's clock-stamp frames into OME-Zarr.

### Existing status/rescan surface

- `SettingsController.getCameraStatus(detectorName)` (:847-870) returns `detector.getCameraStatus()`, built from `DetectorManager.getCameraStatus` (DetectorManager.py:457-497) plus the per-vendor overrides above. Documented in docs/CAMERA_STATUS_API_IMPLEMENTATION.md and docs/CAMERA_STATUS_QUICKSTART.md.
- `getDetectorNames` (:595-600). No reconnect, rescan or reset endpoint exists.
- There is no signal for camera state anywhere in `CommunicationChannel`.

### Frontend

- `LiveView.js:237-256` fetches `getDetectorNames` **once** per host/port change and builds the detector tabs from it. It never re-polls, and it has no status chip.
- Status is shown only in `CameraSettingsDialog.jsx:420-424`, from the one-shot parameter tree (`tree.isConnected` -> "Connected" / "Not connected" chip, `tree.isMock` -> "Mock camera" chip). It is fetched on dialog open, with no live update, and the underlying values are wrong for Hik/Daheng (above).
- The WebSocket handler has no camera-state branch. After an unplug the user sees a frozen but "running" live view.

## 2. SDK support (vendored wrappers)

| | Disconnect detection | Re-open / enumerate | Notes |
|---|---|---|---|
| **Hik** (Linux/Mac wrapper `hikrobotMac/MvCameraControl_class.py`, which is also what the Pi uses: hikcamera.py:12-14; Win wrapper `hikrobotWin/`) | `MV_CC_RegisterExceptionCallBack` (Mac :281-286, Win :427); const `MV_EXCEPTION_DEV_DISCONNECT=0x8001` (hikrobotMac/CameraParams_const.py:29, Win :59). `MV_CC_IsDeviceConnected` exists **only in the Win wrapper** (:147); the Linux/Mac wrapper lacks it (a ctypes wrapper would have to be added; the symbol should be in libMvCameraControl.so, check with `nm -D`). | `MV_CC_EnumDevices` (Mac :102), already used in `_open_camera` (hikcamera.py:271-280); then `MV_CC_CreateHandle` / `MV_CC_OpenDevice`. | The Mac wrapper's exception-callback prototype sets `argtype` (typo, no effect) and **no CFUNCTYPE is defined**. Need `CFUNCTYPE(None, c_uint, c_void_p)` and a kept reference. Linux lib path: `/opt/MVS/lib/aarch64/` (loader :17-24). |
| **Daheng** (`interfaces/gxipy/gxiapi.py`) | `Device.register_device_offline_callback(call_back)` (:1177-1187), `unregister_device_offline_callback` (:1189-1197), trampoline `__on_device_offline_call_back` (:1199-1204). **Unused in ImSwitch.** | `DeviceManager.update_device_list()` (:89) re-enumerates; `open_device_by_index` (gxipycamera.py:80), `close_device` (gxiapi.py:1160). | Offline callback fires on the SDK thread; do not close the device inside it. Spawn a thread. |
| **ToupTek** (`toupcamsdk/toupcam.py`) | `TOUPCAM_EVENT_DISCONNECTED=0x81` (:108) (used), `TOUPCAM_EVENT_ERROR=0x80` and `NOFRAMETIMEOUT=0x82` (:107, :109), the last two only logged (toupcamcamera.py:558-560). | `Toupcam.EnumV2()` (used, toupcamcamera.py:272); `Toupcam.HotPlug(fun, ctx)` binding present (toupcam.py:~1012-1026, unused, platform behaviour unverified). | Already the model implementation. |

## 3. Proposed design (smallest that works)

**Principle: one in-place "camera slot" in each manager, a generic supervisor that only asks managers two questions, all SDK specifics inside the managers/interfaces.**

### 3.1 Manager contract (base `DetectorManager`, generic, no vendor code)

Add to `DetectorManager`:

- `connectionState` in `{"connected","lost","absent","reconnecting"}`. Mock fallback starts as `absent`; the real camera starts as `connected`.
- Hooks that vendors implement: `_devicePresent() -> bool` (cheap enumeration), `_openCamera()` (build a new camera object), `_adoptCamera(cam)`.
- `_adoptCamera` is the refactor from section 0. It must rebuild `__fullShape`, `_shape`, `__model`, `__parameters` and `__supportedBinnings`, and re-apply the injected pixel size and flip (PixelCalibrationController calls `setPixelSizeUm`/`setFlipImage` once at startup, HikCamManager.py:~404-416). Split the parameter-building section of each `__init__` into `_buildParameters(cam)` so `__init__` and `_adoptCamera` share it. This touches the three managers and the base; it is hardware-agnostic.
- `getLatestFrame` raises (or returns None) when `connectionState != "connected"`, so nobody consumes the stale ring-buffer frame. Also `frame_buffer.clear()` on loss. This is a small fix in each interface's `getLast`, and it independently fixes the "silent frozen frame" bug.

### 3.2 Swap in place, controllers keep their references

```
cam = self._openCamera()        # supervisor thread, old handle untouched until it is dead-closed
with self._camLock:             # RLock, held only for the swap itself
    old, self._camera = self._camera, cam
    self._adoptCamera(cam)
    if self._running: cam.start_live()   # stream resumes where it was
old.close()                     # outside the lock; swallow errors
emit status
```

- Readers (`StreamWorker` -> `getLatestFrame`) read `self._camera` once per call into a local variable, so no lock on the hot path. An in-flight call on the old camera returns None or a stale frame once and is harmless.
- **Lost Hik/Daheng handle: close the old one first** (the device is gone, the handle is a zombie; with `MV_ACCESS_Exclusive` a still-open zombie can refuse the re-open on some firmware). Toupcam already does this inside the driver.
- Pin to the same physical camera: Hik uses `self.cameraNo` (serial checksum, hikcamera.py:371-395, :86-108). Add serial pinning for Daheng/Toupcam only if more than one camera of a vendor is attached.
- Fix the Hik partial-init leak: wrap `CameraHIK.__init__` after `OpenDevice` so a later exception closes the handle and discards the claimed checksum.
- The `mock` camera stays as the placeholder (UI keeps working), but report it honestly (`isMock: True`, `state: "absent"`) and fix the three status bugs: Hik `isMock`/`isConnected` (HikCamManager.py:500-501), Daheng `isMock` (GXPIPYManager.py:420), Toupcam is already ok.

### 3.3 Detection: polling enumeration plus SDK callback as a fast path

- **Do not rely on SDK hotplug events in the container** (section 4). Polling is robust:
  - Hik: `MV_CC_EnumDevices` for the USB layer only.
  - Daheng: `update_device_list()`.
  - Toupcam: `EnumV2()`.
- **Absent or lost: poll every 2-3 s** in a supervisor thread (one thread, `DetectorsManager`, started after startup and joined in `finalize`). It calls `manager._devicePresent()` and, if True, `manager._reconnect()`.
- **Connected: do not enumerate** (enumeration can disturb a streaming USB3 camera, not measured). Use the cheap sources instead:
  - Hik: `MV_CC_RegisterExceptionCallBack`, with `MV_EXCEPTION_DEV_DISCONNECT`.
  - Daheng: `register_device_offline_callback`.
  - Toupcam: the existing event.
  - Plus a watchdog for all three: while `_running`, if `getFrameNumber()` has not advanced for max(5 s, 3x exposure) then set `lost` (covers silent failures, e.g. a USB cable that stalls without an exception). Trigger-mode streams legitimately produce no frames, so the watchdog is off unless the trigger source is continuous.
- On `lost`, never reconnect from inside an SDK callback thread (deadlocks with `Close`). The callback only sets state and wakes the supervisor.
- Exponential backoff is not needed. Fixed 2-3 s while absent is cheap (one enum call). Cap logging at one line per state change.

### 3.4 What happens to what is running

| Running | On loss | On reconnect |
|---|---|---|
| Live view / stream | Stream worker keeps running, gets None, and shows "camera lost" (status event). Do **not** let `StreamWorker` exit on its own (it currently stops after repeated None, LiveViewController.py:286-293, which then needs a manual restart). | `_running` is True, so `start_live()` is called on the new camera and the existing worker picks up frames. The frontend only needs to refresh status and shape; image size changes after mock -> real, so it must re-read `getCameraStatus` (width/height) and redraw. |
| Recording | Fails with `CameraLost`; closes the file cleanly. | Not resumed automatically. |
| Experiment / Aufgaben job | Frame grab fails fast with a clear error and the job aborts through its normal error path (verify per call site, section 5). | **Never auto-resume a job.** Missing frames in a scan are a data-integrity decision for the user. |
| Autofocus / calibration | Same as jobs: abort on `CameraLost`. | Not resumed. |

- Use the existing `controller/busy.py` `busy_reasons()` so that a manual "reconnect" is refused with a reason while a job/recording is running. The automatic swap may still happen (the camera is already dead for that job).
- Add a guard that refuses to start an experiment/job/recording when a used detector is `absent` (mock), so fake data cannot be written.

### 3.5 Status signal and endpoints

- New `CommunicationChannel.sigCameraStatus = Signal(dict)  # {detectorName, state, isMock, model, message, shape}` (next to sigDiskFull, CommunicationChannel.py:107). The supervisor emits it on every state change. It reaches the frontend through noqt.py with no extra wiring.
- Endpoints in `SettingsController` (check global name uniqueness first, per AGENTS.md section 3.1):
  - `getCameraConnectionState()`: all detectors' states. Cheap alternative: just extend `getCameraStatus` with `connectionState`.
  - `rescanCamera(detectorName)`, `POST`: one immediate enumeration plus reconnect attempt. Refused while busy.
  - No endpoint should flash/restart anything. Nothing here is in AGENTS.md's ask-first list; it only enumerates and opens a camera.

### 3.6 Frontend (small)

- New Redux slice (or extend `liveViewSlice`) `cameraStatus[detectorName]`; a `sigCameraStatus` branch in `WebSocketHandler.js` next to :888-915; initial fill from `getCameraStatus`.
- Chip next to each detector tab in `LiveView.js` (Connected / Lost / Absent (mock) / Reconnecting) and a "Reconnect" button calling `rescanCamera`. Reuse the chip vocabulary in `CameraSettingsDialog.jsx:420-424`.
- On a `connected` transition: refetch detector shape/params and restart the stream via the existing live-view start call if the tab was streaming.
- Re-fetch `getDetectorNames` on reconnect. The names never change in this design (the mock holds the name), which is an advantage of the in-place swap.
- Toast on loss/recovery via `notificationSlice.setNotification` (as sigStageLimit does).

## 4. Docker / Pi constraints

- `docker/docker-compose.yml`: `privileged: true` (:9) and `/run/udev:/run/udev:ro` (:21), **but `/dev` is not bind-mounted** (only `/dev/dma_heap`, :23). The deployment compose lives in the openUC2 OS repo (`rpi-imswitch-os`), not here, so the real file must be checked.
- Hotplug risk: a privileged container gets a *copy* of the device nodes present at container start, so a camera plugged in later gets no `/dev/bus/usb/BBB/DDD` node inside the container (from experience; **verify on the Pi** with `ls /dev/bus/usb/*/` inside the container before and after plugging). libusb (used by Hik, Toupcam and Daheng) opens cameras through those nodes. A camera present at container start works; a camera plugged in later does not, and an unplug/replug gets a new device number that is also missing.
- Fix, whichever the OS repo prefers:
  - Add `- /dev/bus/usb:/dev/bus/usb` (live bind of the host directory), or `- /dev:/dev`; both together with `privileged: true`. No `device_cgroup_rules` are needed under privileged.
  - Without privileged: `device_cgroup_rules: ["c 189:* rmw"]` plus the `/dev/bus/usb` bind.
- libusb's own hotplug events (netlink uevents) generally do not reach a container in its own network namespace, which is the argument for polling enumeration (section 3.3) rather than SDK hotplug callbacks. Enumeration reads sysfs/`/dev/bus/usb`, which is visible with the bind mount.
- Host udev rules for the cameras (Toupcam README, `toupcamsdk/99-toupcam.rules`; Hik via MVS install in docker/install-drivers.sh) apply on the host, and permissions are fine under `privileged`.
- MVS libs are installed in the image (`/opt/MVS`, Dockerfile:93-96), so no image change is needed.

## 5. Risks and effort

| Part | Effort | Notes |
|---|---|---|
| Common backend: `connectionState`, supervisor thread, base hooks, `_adoptCamera` refactor of shape/params, signal, endpoint, fail-fast `getLast`, busy/guard checks | 2-3 d | The parameter rebuild is the hard part. Mock-to-real changes sensor size. |
| Hik | 1.5-2 d | ctypes exception callback + (optional) `MV_CC_IsDeviceConnected`; fix `reconnectCamera` (reset streaming, re-apply settings); partial-init leak; status bugs. |
| Daheng | 1-1.5 d | Write a reconnect from scratch around the unused offline callback; new `reconnectCamera` and settings re-apply (mirror toupcam `_reapply_settings`); fix `isMock`. Least tested driver. |
| ToupTek | 0.5 d | Add absent-at-start and give-up-after-60-s handling through the common supervisor; clear buffer on disconnect; propagate size change. |
| Frontend | 1 d | Slice, WS branch, chip, button, stream restart, shape refresh. Jest tests for the slice/WS branch. |
| Docker | 0.25 d + verification | Compose change (in the OS repo) + on-Pi test. |
| Unit tests | 1 d | Fake SDK objects (as done for AV/Basler), no hardware. |

**Only verifiable with a real camera / the Pi:**
- Whether Hik's exception callback actually fires on USB unplug on aarch64 MVS 3.0.1, and its latency.
- Whether `MV_CC_EnumDevices` while another Hik camera streams drops frames (matters only for multi-camera).
- Whether a zombie Hik handle must be destroyed before re-enumeration shows the replugged device, and whether a replug keeps the serial (checksum pin).
- Whether Daheng's offline callback fires on USB3 unplug, and `update_device_list` after replug.
- `/dev/bus/usb` node visibility in the container after plugging.
- That the stale-frame fix does not break trigger-mode snaps (the `lastFrameFromBuffer` fallback exists for trigger mode, e.g. Hik :996, toupcam :782).
- Job abort behaviour on `CameraLost`: each `snapSync`/`getLatestFrame` call site in Experiment/Aufgaben/Recording must be checked to confirm it fails fast rather than hanging or saving None (not audited here).

Main risks: reconnect from an SDK callback thread (deadlock), swap race with a grab in flight (mitigated by the local-variable read), the mock-to-real shape change reaching ROI/crop/stitching state, and silently-written mock data (needs the guard in 3.4).

## 6. newswitch carry-over

Carries over (keep it free of ImSwitch imports, like `model/canbus`):
- Per-vendor camera helpers: enumerate / is-present, open by serial, `reapply_settings`, disconnect-callback wiring (Hik ctypes exception callback, gx offline callback, Toupcam event).
- The `connectionState` machine (connected/lost/absent/reconnecting), the poll-when-absent plus callback-when-connected policy, and the frame-age watchdog.
- The fail-fast frame contract (`CameraLost`, no stale frames).
- The Docker `/dev/bus/usb` finding, which is deployment, not code.

ImSwitch-only (throw away): the `DetectorManager._adoptCamera` refactor of name-mangled privates, the supervisor inside `DetectorsManager`, `sigCameraStatus`/`rescanCamera` on `SettingsController`, and all React work. In newswitch the same state would be a rekuest-next state/action pair (status and reconnect action) rather than a Socket.IO signal plus an `@APIExport`.

Suggested order if time is short: (1) fix the stale-frame and status-lie bugs (they are bugs today regardless), (2) Toupcam absent-at-start via the supervisor as the cheap first vendor, (3) Hik, (4) Daheng, (5) frontend chip, (6) Docker.
