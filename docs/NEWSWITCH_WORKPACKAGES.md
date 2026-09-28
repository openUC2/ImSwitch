# newswitch Migration — Work Packages (Waves 1–3 + §6.4)

*Companion to [NEWSWITCH_MIGRATION.md](NEWSWITCH_MIGRATION.md). Status: 2026-07-15.
Each WP has a goal, repos, dependencies, acceptance criteria, and a ready-to-paste
**Claude Code instruction** block. Rule for all WPs: verify with tests/ruff, but
**never commit or push unless Bene explicitly says so.***

Repos (all local working copies):

| Alias | Path |
|---|---|
| `newswitch` | `~/Dropbox/Dokumente/Promotion/PROJECTS/MicronController/newswitch` |
| `uc2rest` | `~/Dropbox/Dokumente/Promotion/PROJECTS/UC2-REST` |
| `uc2canopen` | `~/Dropbox/Dokumente/Promotion/PROJECTS/UC2-REST-CANOPEN` |
| `imswitch` | `~/Dropbox/Dokumente/Promotion/PROJECTS/MicronController/ImSwitch` (reference only) |

Verification commands (newswitch backend): `uv run --python 3.13 pytest tests/ -k "not integration"`,
`./.venv/bin/ruff check newswitch/`, `just dev` for the full stack, `just drift-check` after
adding `@register`/`@state` surface. Note: the backend does **not** import on Python 3.14
(rekuest state decorator bug) — stick to 3.13.

Debugging (VS Code): use the `newswitch backend: debug (...)` configurations in
`.vscode/launch.json` (virtual / UC2 serial / UC2 CANopen — the hardware ones set
`NEWSWITCH_CONFIG` to the matching `backend/configs/*.json`). They run uvicorn **without**
`--reload` on purpose: the reloader moves the server into a child process and breakpoints
never bind. `backend/main.py` is also directly runnable (F5) — it serves the app object
in-process on :8099. For hot-reload (no debugging) keep using `just dev-backend`.

---

## Status after the 2026-07-15 session (already implemented, uncommitted)

**WP1 (UC2Bus foundation) and WP2 (async facades) are largely DONE:**

- `uc2canopen` (v0.2.0, uncommitted): `TpdoListener` gained `on_motor_update` /
  `on_homing_changed` / `on_heartbeat`; callbacks now fire *outside* the lock (deadlock fix).
  New `src/uc2canopen/aio.py`: `AsyncUC2Client` (`create()`, `.sync`, `call()`, typed motor/
  laser/led/galvo/state wrappers, `move_and_wait` with event-driven done + stop-on-cancel,
  `events()` async stream). Exported from `__init__`. 14/14 tests pass.
- `uc2rest`: new `uc2rest/aio.py` (`AsyncUC2Client`: same shape; serial pattern callbacks
  `steppers/home/emergency/message/...` bridged to typed async events; `serialport=None` →
  auto-discovery; move/home blocking waits off-loop with stop-on-cancel). **Not** imported by
  `uc2rest/__init__.py` → zero impact on ImSwitch; smoke-tested against MockSerial.
- `newswitch`: new `protocols/uc2.py` (`UC2BusManager` @context protocol, `UC2State` @state,
  typed events `PositionUpdate/MotionDone/HomingChanged/EStopChanged/ButtonPressed/NodeSeen/
  BusError`); `managers/uc2/`: `event_broker.py`, `virtual_bus.py`, `canopen_bus.py` (node map,
  µm↔steps scaling), `rest_bus.py`, `dispatch.py` (event→state mirror, holds `stage_position`
  lock), rewritten `stage_manager.py` (bus-based, Z-first homing), new
  `illumination_manager.py`, rewritten `galvo_scanner.py`. `app.py`: transport selection via
  `ImswitchConfig.uc2_transport` ("canopen" default | "rest"), new `@background run_uc2_bus` +
  `run_uc2_event_dispatch`, tuple/annotation extended with `UC2BusManager` + `UC2State`.
  `pyproject.toml`: `uc2` extra. Tests: `tests/test_uc2_bus.py` (8 new) — 38 passed total.
- The legacy mock `managers/uc2/serial_manager.py` is out of the hardware path (kept for
  reference only); `VirtualSerialManager` fills the deprecated `SerialManager` slot.

Open hardening items from this session are folded into WP0/WP3 below.

**Update 2026-07-21 (also uncommitted):**

- **Placeholder purge done:** the mock JSON transport is deleted —
  `managers/uc2/serial_manager.py`, `managers/virtual/virtual_serial_manager.py`,
  `protocols/serial_manager.py` are gone; `SerialManager`/`SerialState` removed from
  `provide_managers` and `protocols/__init__`. Only real transports + `VirtualUC2Bus` remain.
  Frontend codegen regenerated (SerialState.ts dropped, UC2State.ts added).
- **Config files work:** `main.py` loads `NEWSWITCH_CONFIG=<file>.json` into `ImswitchConfig`;
  nested per-component overrides (`uc2_can`, `uc2_rest`, `uc2_stage`, `uc2_illumination`)
  flow into the typed bus/manager configs. Two complete setups ship in `backend/configs/`:
  `uc2_serial.json` (one ESP32 master, UC2-REST, port autodetect, 500000 baud) and
  `uc2_canopen.json` (Waveshare USB-CAN-A on `/dev/cu.SLAB_USBtoUART`, full node map;
  socketcan/can0 variant documented inline). Covered by tests.
- **First hardware validation (REST transport, corrected):** the bench device on
  `/dev/cu.SLAB_USBtoUART` is a **UC2 CAN-HAT master** running firmware
  `UC2_canopen_master` V2.0 (Jul 2026) — its serial link runs at **921600 baud** (115200/
  500000 yield undecodable bytes and uc2rest falls back to MockSerial gracefully). With the
  right baud, the full path works end-to-end on hardware: `UC2RestBus` → `uc2rest.aio` →
  board; firmware identity + isMaster land in `UC2State`, live positions read through the
  bus (steps→µm), clean shutdown; and the whole backend boots from
  `configs/uc2_serial.json` with the bus connecting during startup. Topology note: a
  USB-connected CAN-HAT master is driven via the **REST** transport (it bridges serial
  JSON→CAN itself); the **canopen** transport is for direct host CAN interfaces
  (SocketCAN HAT / Waveshare USB-CAN-A). An earlier claim that the CAN path was validated
  against this device was wrong — port-open + empty scan proves nothing; real CAN
  validation still needs a CAN interface + nodes (WP3).
- **Dev env:** local UC2-REST + UC2-REST-CANOPEN installed editable in the backend venv
  (part of WP0); backend venv re-synced to Python 3.13. Backend boots in virtual mode with
  the new wiring; backend 39 pytest + frontend 13 vitest green.
- **lefthook fixed:** `rc: .lefthookrc` added — hooks now find `uv`/`yarn` when committing
  from GUI/IDE shells.

**Update 2026-07-22 — WP1–WP7 software complete (uncommitted):**

- **WP1 DONE:** reconnect-with-backoff in both buses (transport errors → `BusError` +
  `UC2State.last_error`, capped exponential retry, never kills the agent; REST treats the
  MockSerial fallback as a failure so replug-recovery is real); `_require_client` fails fast
  with a readable error; REST `PositionUpdate` throttled per axis (50 ms default); CAN
  shared-node sub-axis mapping (`sub_x..sub_a`, `axis_for_node(node, sub)`). Covered by a
  fake-transport reconnect test.
- **WP2 DONE:** facade tests added in both library repos — uc2canopen `tests/test_aio.py`
  (synthetic TPDO/heartbeat frames → typed events; move_and_wait done-edge + stop-on-cancel;
  lock-deadlock regression test) 4/4 + 14/14 features; uc2rest `uc2rest/TEST/TEST_aio_mock.py`
  (MockSerial + pattern-frame injection) passing.
- **WP4 DONE:** `newswitch/registers/uc2.py` — home_stage (Z-first), stop_stage (lock-free),
  led_matrix_fill/off, uc2_scan_nodes, home_objective, galvo_set_position/raster_scan/stop/
  get_status (typed `GalvoStatus`; plain dict returns hit the FastApiAgent shelving gap).
  Agent-level tests green; frontend hooks regenerated.
- **WP5 DONE (converter):** `scripts/convert_imswitch_setup.py` converts ImSwitch setups
  (transport from rs232 manager, stepsize→signed steps_per_um, speeds, laser channels →
  illumination sources) with skipped-item reporting; `tests/fixtures/{example_uc2,canopen}.json`
  round-trip-validated against `ImswitchConfig` + bus configs.
- **WP6 DONE:** bus verbs `aobjective_move/home/status` + `ascan_nodes` on all three
  transports (REST = native firmware objective module via new uc2rest.aio wrappers; CAN =
  configurable motor-node with calibrated slot positions; virtual = simulated).
  `UC2ObjectiveManager` and `UC2FilterBankManager` (wheel on a configurable rotation axis
  with per-slot positions) implement the existing protocols and replace the virtual ones in
  hardware mode; stale placeholder copies (`managers/uc2/detector_manager.py`, old
  filter_bank) removed. Config dicts `uc2_objective`/`uc2_filter_bank` added.
- **WP7 DONE:** `routines/autofocus.py` — `compute_focus_metric` (Laplacian/intensity
  variance), plain `autofocus_sweep` core (shared with the hook), registered `run_autofocus`
  with progress/pausepoints + `AutofocusState` (@state, in the provide_managers tuple);
  `hooks/software_autofocus.py` now does a real sweep instead of the random stub. Verified
  against the virtual detector's defocus model (finds z≈0 from a defocused start).
- Suite: backend 54 passed / 1 skipped, frontend 13 passed, ruff clean, codegen regenerated.
- **WP3 partially blocked:** the UC2 HAT vanished from `/dev` mid-session (was
  `cu.SLAB_USBtoUART`, expected UC2-REST JSON at 921600 baud per Bene) — REST-transport
  hardware validation pending replug. Earlier note: the port answered with binary frames at
  115200/500000; retry at 921600 once reconnected.

---

## WP0 — Dev environment & library releases

**Goal:** make the local dev loop and CI reproducible with the new hardware deps.
**Repos:** newswitch, uc2rest, uc2canopen. **Depends:** –.

Tasks: install local editable checkouts into the newswitch venv
(`uv pip install -e ../../../UC2-REST -e ../../../UC2-REST-CANOPEN` from `backend/`);
once bench-validated (WP3), cut releases `uc2-rest > 0.2.0.39` (includes `aio.py`) and
`uc2canopen 0.2.0` and tighten the `uc2` extra pins; verify the ImSwitch test rigs still work
against the new library versions (backwards compat is a hard requirement); keep the
`od.py`-from-firmware generation step documented/automated.

**Acceptance:** `uv run pytest` green with extras installed; ImSwitch boots against the same
uc2rest/uc2canopen checkouts; `just dev` starts with `use_virtual_microscope=True`.

```text
Claude Code prompt (run in newswitch/backend):
Install the local UC2 libraries editable into this venv (uv pip install -e
../../../UC2-REST -e ../../../UC2-REST-CANOPEN), then run
`uv run --python 3.13 pytest tests/ -k "not integration"` and `just dev` (virtual mode)
and fix anything the extras broke. Also open ~/.../PROJECTS/MicronController/ImSwitch and
run its UC2-related unit tests (see docs/NEWSWITCH_MIGRATION.md repo map) to prove the
uc2rest/uc2canopen changes are backwards compatible. Do not commit anything.
```

---

## WP1 — UC2Bus foundation hardening  *(core done — hardening left)*

**Goal:** production-grade bus layer. **Repos:** newswitch. **Depends:** WP0.

Remaining tasks: reconnect/backoff loop in `abackground()` for both real buses (link loss →
`BusError` event + `UC2State.connected=False` → retry, never crash the agent); make
`_require_client` fail fast with a readable error when the bus never connects; REST bus:
throttle `PositionUpdate` from the `steppers` stream if it floods; CAN bus: axis-on-shared-node
support (sub-index mapping) when one board drives multiple motors; verify async `@background`
functions run correctly under the rekuest agent on real startup (they are exercised in tests,
but watch first `just dev` logs).

**Acceptance:** unplugging/replugging the adapter while `just dev` runs recovers without
restart; pytest suite stays green; ruff clean.

```text
Claude Code prompt (run in newswitch/backend):
Read newswitch/managers/uc2/ (bus layer added 2026-07). Add reconnect-with-backoff to
UC2CanBus.abackground and UC2RestBus.abackground: on transport exception publish BusError,
set state.connected=False, close the client, retry with exponential backoff (cap 30 s)
until task cancellation. Add a clear RuntimeError from _require_client when the background
never connected. Extend tests/test_uc2_bus.py with a fake-transport reconnect test.
Run uv run --python 3.13 pytest and ruff. Do not commit.
```

---

## WP2 — Async library facades  *(done — follow-ups only)*

**Goal:** asyncio access to uc2rest/uc2canopen without breaking ImSwitch. **Done** (see
status). Follow-ups: add facade unit tests inside each library repo (uc2canopen: extend
`tests/test_new_features.py` style with a fake listener; uc2rest: MockSerial-driven event
test); fix the known ImSwitch adapter bug (`UC2CANOpenManager` passes `debug=`, client takes
`log_level=`) — in ImSwitch, not the library.

```text
Claude Code prompt (run in UC2-REST-CANOPEN):
Add tests for src/uc2canopen/aio.py: drive TpdoListener.on_message_received with synthetic
can.Message frames (TPDO1 + heartbeat) and assert AsyncUC2Client.events() yields
MotorUpdateEvent/MotorDoneEvent/HomingChangedEvent/HeartbeatEvent, and that move_and_wait
resolves on the done edge and stops the motor on cancellation. Follow the RecordingSdo
pattern in tests/test_new_features.py. Keep the sync API untouched. Do not commit.
```

---

## WP3 — Hardware bench validation (CAN first, REST second)

**Goal:** prove both transports on real openUC2 hardware and that they are plug-in
replacements. **Repos:** newswitch (+ firmware bench). **Depends:** WP0–1. **Human-in-loop:**
needs Bene at the bench.

Tasks: bring up `just dev` with `use_virtual_microscope=False, uc2_transport="canopen"`
(SocketCAN on the Pi / Waveshare on the Mac); smoke: scan_nodes, per-axis move/stop/home from
the web UI, live position in `StageState`, laser/LED, galvo goto+raster+stop (respecting the
doorbell invariants: idle=0, stop=4, qid echo on REST); repeat identical script with
`uc2_transport="rest"` — same UI, zero code changes = plug-in proof; measure move-command
latency + position-update rate on a Raspberry Pi; calibrate and record real
`steps_per_um_*` values per axis.

**Acceptance:** a written smoke log (both transports, same checklist); no event-loop stalls;
Pi CPU acceptable during live telemetry.

```text
Claude Code prompt (run in newswitch/backend, hardware attached):
Help me bench-validate the UC2 bus. Start the backend with use_virtual_microscope=False and
uc2_transport="canopen" (ask me for the CAN interface). Walk through: node scan, X/Y/Z
move+stop+home, laser PWM, LED fill, galvo goto/raster/stop, and confirm StageState updates
live in the frontend. Log every step and any error verbatim into
docs/bench-logs/<date>-canopen.md (create it). Then repeat with uc2_transport="rest".
Do not change hardware settings without asking; do not commit.
```

---

## WP4 — Wave-1 backend surface: UC2 actions & states

**Goal:** expose the bus to the UI: registered functions + states beyond `move_stage`.
**Repos:** newswitch. **Depends:** WP1.

Tasks: add `@register` functions in `app.py` (or a new `newswitch/registers/uc2.py`):
`home_stage(axes)`, `stop_stage(axis|all)` (lock `stage_position`), `set_laser_power`,
`led_fill/led_off`, `uc2_scan_nodes`, `uc2_status` (reads `UC2State`), `galvo_goto/
galvo_raster/galvo_stop` (lock `galvo`); extend `StageState` with per-axis `homed` flags and
speeds if useful; add optimistics where the UI should react instantly; run codegen +
`just drift-check`; extend pytest via `AsyncAgentTestClient` against the virtual bus.

**Acceptance:** generated hooks appear in `frontend/src/apps/default/hooks/actions/`;
agent-level tests for home/stop/laser pass on the virtual bus.

```text
Claude Code prompt (run in newswitch):
Add Wave-1 UC2 registered functions to the backend: home_stage(axes: list[str]) and
stop_stage(axis: str | None) using StageManager/UC2BusManager (lock "stage_position"),
set_laser_power(channel:int, power: float) via IlluminationManager (lock "illumination"),
led_fill(r,g,b)/led_off, uc2_scan_nodes() and galvo_goto/galvo_raster/galvo_stop via
UC2BusManager. Follow the style of existing @register functions in newswitch/app.py
(docstrings, ANN annotations, locks, optimistics where sensible). Add
AsyncAgentTestClient tests patterned on tests/test_api.py using the virtual bus. Then run
`just dev-backend`, regenerate the frontend (vite plugin), and `just drift-check`.
Do not commit.
```

---

## WP5 — Setup/config loader (per-microscope JSON)

**Goal:** replace hard-coded `provide_managers` wiring with declarative per-microscope
setup files, convertible from ImSwitch's `imcontrol_setups/*.json`. **Repos:** newswitch
(+ imswitch as schema reference). **Depends:** WP1.

Tasks: pydantic `SetupInfo` model (`newswitch/setup.py`): transport choice + bus config
(node map, steps_per_um, serial port), axes present, illumination sources (slots, kinds,
wavelengths, pwm_max), objective slots, feature flags; loader
(`NEWSWITCH_SETUP=/path/to/setup.json` env or `ImswitchConfig.setup_file`);
`provide_managers` consumes it; converter script
`scripts/convert_imswitch_setup.py` mapping `positioners.*ESP32StageManager*` /
`UC2CANOpen*` / lasers / LEDMatrix blocks onto the new schema; convert `example_uc2.json`
and `canopen.json` as fixtures + round-trip tests.

**Acceptance:** the same backend binary boots as two different microscopes purely by
setup file; converted ImSwitch setups load without hand-editing.

```text
Claude Code prompt (run in newswitch/backend):
Build a declarative setup system. 1) newswitch/setup.py: pydantic models (SetupInfo:
uc2 transport + UC2CanBusConfig/UC2RestBusConfig fields, stage axes + steps_per_um + limits,
illumination sources, objective slots, feature flags). 2) Load it in create_app/
provide_managers from ImswitchConfig.setup_file (env NEWSWITCH_SETUP overrides); fall back
to current defaults. 3) scripts/convert_imswitch_setup.py: convert ImSwitch setup JSONs
(see ~/.../ImSwitch/imswitch/_data/user_defaults/imcontrol_setups/example_uc2.json and
canopen.json; managerName/managerProperties semantics documented in
~/.../ImSwitch/docs/NEWSWITCH_MIGRATION.md §3.5) into SetupInfo JSON. Add tests: load both
converted fixtures and assert manager wiring. pytest + ruff green. Do not commit.
```

---

## WP6 — Wave-1 devices: objective changer & filter wheel over the bus

**Goal:** complete Wave-1 device parity. **Repos:** newswitch, uc2rest, uc2canopen.
**Depends:** WP1, WP4.

Tasks: extend `UC2BusManager` with `aobjective_move(slot)/aobjective_home()/
aobjective_status()`; REST backend maps to `uc2rest .objective` (home/calibrate/move/
getstatus — wrap in `uc2rest/aio.py`); CAN backend maps to a motor-node move to calibrated
slot positions (or a dedicated OD group if firmware adds one — check
`uc2_canopen_registry.yaml` first); implement `UC2ObjectiveManager` implementing the
existing `ObjectiveManager` protocol (replace the virtual one in hardware mode); same
pattern for `FilterBankManager` (servo/motor-based filter wheel via laser servo or motor
node).

**Acceptance:** objective switch works from the existing ObjectiveControl UI on hardware;
virtual objective still works in sim; tests on the virtual bus.

```text
Claude Code prompt (run in newswitch/backend):
Add objective-changer support to the UC2 bus. Extend newswitch/protocols/uc2.py
UC2BusManager with aobjective_move(slot:int), aobjective_home(), aobjective_status();
implement in rest_bus.py via uc2rest .objective (add thin async wrappers to
../../../UC2-REST/uc2rest/aio.py: objective home/calibrate/move/getstatus), in
canopen_bus.py via a configurable motor-node slot-position map, and in virtual_bus.py as
a simple slot store. Then write UC2ObjectiveManager implementing the ObjectiveManager
protocol (see managers/virtual/virtual_objective.py for the contract) and wire it in
provide_managers for hardware mode. Tests on the virtual bus. pytest + ruff.
Do not commit.
```

---

## WP7 — Autofocus routine + hook (Wave 1→2 bridge)

**Goal:** port ImSwitch's autofocus as a routine and back the already-scaffolded
`SoftwareAutofocusHook`. **Repos:** newswitch (imswitch algorithm reference:
`imswitch/imcontrol/controller/controllers/AutofocusController.py`). **Depends:** WP4.

Tasks: `newswitch/routines/autofocus.py`: `run_autofocus(stage, detector,
acquisition_manager, z_range, steps, metric)` — z-sweep, focus metric (variance of
Laplacian + the Gaussian-fit variant from ImSwitch), `progress()`/`pausepoint()`, lock
`stage_position`, returns best z and moves there; `AutofocusState` (curve, best_z,
running); register it and back `software_autofocus_hook` with the real routine.

**Acceptance:** autofocus converges on the virtual microscope's synthetic PSF stack
(assert best_z ≈ simulated focus in a test); UI shows progress and can cancel.

```text
Claude Code prompt (run in newswitch/backend):
Port ImSwitch autofocus as a routine. Reference algorithm:
~/.../ImSwitch/imswitch/imcontrol/controller/controllers/AutofocusController.py
(hill-climbing + Gaussian fit variants). Create newswitch/routines/autofocus.py with
run_autofocus(stage: StageManager, acquisition: AcquistionManager, state: AutofocusState,
z_range: float, steps: int, metric: str) using progress()/pausepoint() and lock
"stage_position"; define AutofocusState in protocols/. Wire it into
newswitch/hooks/software_autofocus.py and register it in app.py. Test against the
virtual detector's defocus simulation (see managers/virtual/virtual_detector.py PSF
handling) asserting the found z is near the simulated focus. pytest + ruff +
drift-check. Do not commit.
```

---

## WP8 — Recording & storage services (Wave-2 prerequisite)

**Goal:** port ImSwitch's `io/` layer (snap/record, OME-Zarr/OME-TIFF streaming writers,
thumbnails, file listing). **Repos:** newswitch (reference:
`imswitch/imcontrol/model/io/`). **Depends:** none strictly; parallelizable.

Tasks: `RecordingManager` @context + `RecordingState`; writers on zarrs/tifffile (already
deps); actions `snap_to_file`, `start/stop_recording`; extend `routes/http/files.py`
toward list/preview/thumbnail/delete with path-traversal guards (copy ImSwitch's checks);
adopt `useq-schema` `MDASequence` as the acquisition-plan format.

**Acceptance:** snap + timed recording produce valid OME files from the virtual detector;
file endpoints browse them; tests read written files back.

```text
Claude Code prompt (run in newswitch/backend):
Port the ImSwitch recording layer. Reference: ~/.../ImSwitch/imswitch/imcontrol/model/io/
(recording_service.py, writers.py, ome_writers/, thumbnails.py — read them first).
Create protocols/recording.py (RecordingManager protocol + RecordingState) and
managers/recording/ with an OME-TIFF writer and a streaming OME-Zarr writer (tifffile +
zarrs are already dependencies). Add @register snap_to_file and
start_recording/stop_recording; store under the LocalFileIOManager base path. Extend
routes/http/files.py with list/thumbnail/delete incl. path-traversal guards mirroring
ImSwitch's FileManager checks in ImSwitchServer.py. Tests: record N virtual frames, read
the file back, assert shape/metadata. pytest + ruff + drift-check. Do not commit.
```

---

## WP9 — Wave 2: acquisition apps

**Goal:** the product core — wellplate/region experiments, HistoScan-class scanning, stage
map, affine calibration. **Repos:** newswitch (references: `experiment_controller/`,
`HistoScanManager`, `StageMapController`, `PixelCalibrationController` in ImSwitch).
**Depends:** WP4, WP5, WP8 (+ detector work by the camera team).

Sub-packages (each is a separate session-sized task):
1. **Experiment routines**: generalize `routines/region_scan.py` into wellplate scan
   (labware model → positions), z-stack and channel loops composed as routines; use
   `MDASequence`.
2. **Firmware-assisted stage scan**: expose `startStageScanning`/coordinate-list scanning
   through the bus (REST has it; CAN needs firmware support — flag if missing) for
   HistoScan-speed scanning with camera triggering.
3. **Stage map**: push scan tiles into the Expanse (3D canvas) via
   `acquisition_manager` + `expanse_manager`, mirroring StageMapController's stitching
   semantics.
4. **Affine calibration**: port PixelCalibration routines to *write `Kube.affine_matrix` /
   `CalibrationState`* instead of a pixel-size store.

**Acceptance:** a converted ImSwitch wellplate experiment runs end-to-end on the virtual
microscope, produces OME-Zarr + stitched preview, and is pausable/cancellable from the UI.

```text
Claude Code prompt (run in newswitch/backend) — sub-task 1 example:
Generalize routines/region_scan.py into experiment routines. Add
routines/wellplate_scan.py: take a labware definition (build a small pydantic model of
plate: rows/cols/pitch/a1-offset), per-well FOV grid via the same metadata-driven FOV
logic as scan_region, optional z-stack per position, channels via IlluminationManager,
frames via AcquistionManager, progress()/pausepoint(), results streamed through the
RecordingManager (WP8). Register it; agent-test on the virtual microscope asserting the
expected number of images and stage visits. Reference for semantics:
~/.../ImSwitch/imswitch/imcontrol/controller/controllers/experiment_controller/.
pytest + ruff + drift-check. Do not commit.
```

---

## WP10 — Wave 3: UC2 platform apps

**Goal:** the UC2Config-class functionality: fleet management, motion tuning, safety.
**Repos:** newswitch, uc2canopen, uc2rest. **Depends:** WP4; hardware for acceptance.

Sub-packages:
1. **CAN fleet management**: actions `uc2_scan_nodes` (done in WP4), `uc2_reassign_node`,
   `uc2_reboot_node`, `uc2_node_info` (uptime/heap/temp/CAN errors via uc2canopen `State`);
   OTA later (uc2rest `.canota` / CAN OTA OD group).
2. **Motion tuning**: TMC parameters (`configure_tmc`), soft/hard limits, backlash — bus
   verbs + actions + a settings panel; persist per-setup (WP5 files).
3. **Safety/GPIO**: collision detector (uc2canopen `Collision`, uc2rest `.gpio`) →
   `EStopChanged`/dedicated events + `UC2State`; joystick/PTZ mapping.
4. **Sensors & thermal**: I2C sensors (uc2rest `.i2c`), temperature/fan/PID
   (uc2canopen `Pid`, uc2rest `.temperature`/`.fan`) as states + background pollers.
5. **Focus lock**: PI loop as `@background` over detector metric + stage Z (or hardware
   PID node), with `FocusLockState`.

```text
Claude Code prompt (run in newswitch/backend) — sub-task 1 example:
Add CAN fleet-management actions. Extend UC2BusManager with anode_info(node_id) ->
dict (uptime, free_heap, cpu_temp, can_errors; uc2canopen State wrappers — add async
variants to ../../../UC2-REST-CANOPEN/src/uc2canopen/aio.py), areassign_node(old,new),
areboot_node(node). REST backend: map to uc2rest .can/.state equivalents where they
exist, else raise NotImplementedError with a clear message. Register uc2_node_info,
uc2_reassign_node, uc2_reboot_node actions; reflect discovered nodes in
UC2State.nodes_online (already wired). Virtual bus: fake three nodes. Tests + ruff +
drift-check. Do not commit.
```

---

## WP11 — §6.4 end-to-end: spontaneous events → states → hooks → UI

**Goal:** hardware events that belong to no task reach the user instantly and drive
configurable policy. **Repos:** newswitch (+ frontend). **Depends:** WP1 (dispatch exists),
WP4.

Already in place: typed events, `dispatch_uc2_events` background mirroring into
`StageState`/`UC2State` (estop, nodes, last_error). Remaining tasks: `InputState` @state
(last button, joystick axes) fed from `ButtonPressed`; hook policy — extend `HookManager`
with a `HardwareButtonHook` (e.g. button A → snap via IOManager, mirroring ImSwitch's
`message.register_callback` snap-on-button) and route `ButtonPressed` through it in the
dispatcher; e-stop UX: frontend banner bound to `UC2State.estop_active` + disable motion
actions client-side while active (locks help server-side); joystick→stage jog policy
(guarded by the stage lock).

**Acceptance:** with the virtual bus, an injected `ButtonPressed` triggers a snap and an
injected `EStopChanged(True)` shows the banner and blocks `move_stage` until cleared;
same demo on hardware with the physical button/e-stop.

```text
Claude Code prompt (run in newswitch):
Finish the spontaneous-event path (§6.4 of docs/NEWSWITCH_MIGRATION.md in the ImSwitch
repo). 1) Add InputState (@state: last_button_key, last_button_time, joystick axes) in
protocols/uc2.py; update managers/uc2/dispatch.py to fill it from ButtonPressed and to
invoke a new HardwareButtonHook via the HookManager (see protocols/hook_manager.py and
hooks/software_autofocus.py for the pattern) — default policy: snap an image through
AcquistionManager+IOManager. 2) Add a test injecting ButtonPressed/EStopChanged through
VirtualUC2Bus's broker asserting state + hook effects. 3) Frontend: add an EStop banner
component subscribing to UC2State (generated hook) that overlays the UI while
estop_active. pytest, vitest, ruff, drift-check. Do not commit.
```

---

## WP12 — Frontend shell & Wave-1 UC2 panels

**Goal:** user-facing surface for the new backend state. **Repos:** newswitch/frontend.
**Depends:** WP4 (generated hooks).

Tasks: `UC2StatusPanel` (connection/transport/firmware/nodes/estop from `useUC2State`);
extend `StageControl` with home buttons + homing status; `IlluminationControl` against the
UC2 actions; galvo panel (goto pad + raster form + status); a lightweight app registry/
nav (derive "available apps" from which actions exist in the generated schema — strictly
better than ImSwitch's `appRegistry.js` + capability cross-check).

**Acceptance:** `just dev` (virtual) shows the UC2 status panel updating live; all vitest
+ `tsc` + eslint gates green; `just drift-check` clean.

```text
Claude Code prompt (run in newswitch/frontend, backend running in virtual mode):
Build UC2 UI panels using generated hooks in src/apps/default/. 1) Create
src/components/microscope/UC2StatusPanel.tsx modeled on StageControl.tsx: subscribe with
useUC2State({subscribe:true}); show connected/transport/firmware, nodes_online as chips,
and a red banner when estop_active. 2) Add Home buttons (per-axis + all) to
StageControl.tsx using the home_stage action definition with ProgressDisplay. 3) Galvo
panel: XY goto (click-pad), raster form, stop button, status badges. Follow the
theme/Tailwind conventions of the existing components. Run yarn test, tsc, eslint, and
just drift-check. Do not commit.
```

---

## WP13 — Firmware update pipeline (port from ImSwitch)

**Goal:** bring ImSwitch's complete firmware/OTA tooling into newswitch so a microscope can
be flashed and fleet-updated from the web UI. **Repos:** newswitch (reference: ImSwitch
`imswitch/imcontrol/controller/controllers/UC2ConfigController.py` — ~83 endpoints, the
firmware subset only — plus its vendored `espota.py`), uc2rest, uc2canopen.
**Depends:** WP4 (actions pattern), WP10.1 (node discovery); hardware for acceptance.

What ImSwitch has today (port all four mechanisms):
1. **USB master flash via esptool**: `import esptool`, `_run_esptool` subprocess with a
   cooperative cancel flag; flashes `esp32_UC2_canopen_master_release.bin` onto the
   USB-connected master (HAT v2).
2. **Firmware server + cache**: `_firmware_server_url` (default
   `http://host.docker.internal/firmware`), download into a temp cache dir
   (`_download_firmware_for_device`, `clearOTAFirmwareCache`), and the **CAN-ID→binary
   mapping** (`_get_can_id_firmware_mapping`: 1=master, motor/LED/laser slaves…) used by
   `listAvailableFirmware`/`getOTADeviceMapping`.
3. **WiFi (Arduino) OTA for slaves**: `startSingleDeviceOTA`/`startMultipleDeviceOTA` —
   device joins WiFi (`setOTAWiFiCredentials`, default password "youseetoo"), then
   `espota.upload_ota(esp_ip, firmware_path, esp_port=3232, ...)` with progress callbacks
   (`sigOTAStatusUpdate` → in newswitch: an `OTAState` @state + task `progress()`),
   `getOTAStatus`/`clearOTAStatus`.
4. **CAN streaming OTA**: `startCANStreamingOTA(can_id, firmware_url, baud=921600)` /
   `startMultipleCANStreamingOTA`/`cancelCANStreamingOTA` — delegates to uc2rest `.canota`
   (`start_can_streaming_ota[_blocking]`); on the CANopen transport, the OD OTA group
   (`0x2F00-0x2F05`, DOMAIN firmware data) exists in uc2canopen's OD but has **no client
   implementation yet** — decide: reuse the REST-master streaming path or implement
   SDO-segmented OTA in uc2canopen.

newswitch shape: an `OTAManager` @context + `OTAState` @state (per-node status/progress),
registered actions mirroring the four mechanisms, firmware files served/cached through the
IO layer, esptool/espota vendored or added as optional deps. Long flashes are rekuest tasks
(progress + cancel), one node at a time, guarded by a `firmware_update` lock so motion apps
can't run mid-flash.

**Acceptance:** flash the USB master from the UI; OTA one CAN slave via WiFi OTA and (if
implemented) via CAN streaming; status/progress visible live; cancel works; a failed
download leaves a readable error in `OTAState`.

```text
Claude Code prompt (run in newswitch/backend):
Port ImSwitch's firmware-update pipeline. Read
~/.../ImSwitch/imswitch/imcontrol/controller/controllers/UC2ConfigController.py (only the
firmware/OTA members: _get_can_id_firmware_mapping, _download_firmware_for_device,
_upload_firmware_to_device, startSingleDeviceOTA/startMultipleDeviceOTA,
startCANStreamingOTA, setOTAWiFiCredentials/setOTAFirmwareServer, getOTAStatus,
_run_esptool) and its espota.py. Create protocols/ota.py (OTAManager protocol + OTAState
with per-node {phase, progress, error}) and managers/ota/ with: firmware cache +
server-download helper, esptool USB flash (subprocess, cancellable), espota WiFi OTA
(vendor espota.py), and CAN streaming OTA via uc2rest .canota (REST transport only for
now; raise a clear NotImplementedError on the CANopen bus and note the OD 0x2F00 group as
the future path). Register actions flash_master, start_device_ota, start_can_streaming_ota,
ota_status, set_ota_credentials, set_firmware_server, clear_firmware_cache — all as
long-running tasks with progress() and a "firmware_update" lock. Unit-test the mapping,
cache, and status bookkeeping without hardware (mock the uploaders). pytest + ruff +
drift-check. Do not commit.
```

---

## Sequencing

```
WP0 ──► WP1 ──► WP3 (bench, human-in-loop)
         │
         ├──► WP4 ──► WP6, WP7, WP12
         │      └──► WP10 (Wave 3) ──► WP13 (firmware/OTA)
         │             └──► WP11 (§6.4 completion)
         ├──► WP5 (setup files)
         └──  WP8 (parallel) ──► WP9 (Wave 2, also needs WP4+WP5)
```

Wave 1 = WP4 + WP6 + WP7 + WP12. Wave 2 = WP8 + WP9. Wave 3 = WP10 + WP13 (+ WP11 spans
all). Cameras/detector work is explicitly out of scope (other team) — coordinate only on
the `CameraState`/`AcquistionManager` contract before WP9.
