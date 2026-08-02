# ImSwitch → newswitch: Architecture Comparison & Porting Strategy

*Status: 2026-07 analysis of `ImSwitch` (openUC2 fork, `master`) vs `newswitch` (`main`, v1.0.2).
Scope: how to port ImSwitch "apps" (controller / manager / React-widget triples) to newswitch,
with deep UC2-REST + UC2-CANopen hardware support. Cameras are explicitly out of scope
(handled by another team).*

---

## 1. Executive summary

The two systems solve the same problem with opposite philosophies:

- **ImSwitch** is a *name-convention framework*: a JSON setup file selects widgets by string,
  which dynamically imports `FooController`, which instantiates `FooManager`s, whose
  `@APIExport` methods become REST endpoints, whose signals fan out over Socket.IO, and a
  hand-written React app (Redux slice + axios wrapper per feature) mirrors all of that by hand.
- **newswitch** is a *typed dependency-injection framework* (built on `rekuest-next`): backend
  functionality is plain Python functions (`@register`) whose manager/state arguments are
  injected by type; reactive `@state` classes are broadcast to the frontend as JSON-patches;
  and the **entire frontend API layer (typed hooks, zod schemas, lock/optimistic-update
  wiring) is code-generated** from the running backend. You only hand-write the visual
  component.

Consequence: **porting an app is not a file-by-file translation.** An ImSwitch app collapses
from 4 hand-maintained artifacts (controller, manager, Redux slice + axios wrappers, React
widget) into 2–3: a manager implementing a Protocol, a handful of `@register` functions +
`@state` classes, and a React component consuming generated hooks. The business logic (the
manager + the algorithmic core of the controller) ports nearly verbatim; the plumbing
(APIExport routes, CommChannel signals, Redux slices, axios wrappers) is *deleted*, not ported.

The critical-path work is **not** the apps — it is four framework gaps in newswitch:

1. **A real UC2 hardware layer.** newswitch's `UC2SerialManager` is a mock today; there is no
   CANopen support at all. This is where UC2-REST / UC2-REST-CANOPEN integration lands (§6).
2. **A declarative setup/config system** (newswitch wires devices imperatively in
   `provide_managers`; ImSwitch's JSON setup files have no equivalent yet).
3. **Recording/storage services** (OME-Zarr/OME-TIFF writers, snap/record, file manager).
4. **A porting template + conventions** so each app port is mechanical (§5).

Recommended sequencing: build the UC2 adapter layer first (it unblocks everything and is
where you have unique domain knowledge), port the core device apps (stage, illumination,
objective) as proof, then the acquisition stack, then feature apps in value order (§7–8).

---

## 2. The two codebases at a glance

| | **ImSwitch (openUC2 fork)** | **newswitch** |
|---|---|---|
| Repo shape | Single Python package + colocated CRA React app (`frontend/`), prebuilt bundle served at `/ui` | Monorepo (`backend/` + `frontend/`), `justfile` + Docker Compose, semantic-release lockstep versioning |
| Backend | Python ≥3.9-ish, FastAPI + uvicorn on a side thread, Qt-free `noqt` framework (psygnal), setuptools/pip | Python 3.11–3.13, FastAPI + **rekuest-next** agent, hatchling + **uv**, asyncio-first with `koil` sync-bridge |
| Frontend | React (CRA/CRACO), **Redux Toolkit** + redux-persist, axios, socket.io-client, ~670 JS/JSX files, everything hand-written | **React 19 + TS + Vite 7**, **zustand** + zod, Tailwind 4 + Radix/shadcn, **generated typed hooks** (`plugins/generate-app.ts`), vitest |
| API surface | `@APIExport` methods → `GET/POST /imswitch/api/{Controller}/{method}`; OpenAPI + Swagger | `@register` functions → rekuest "actions" assigned over one `/ws` socket (+ generated HTTP surface); schemas at `/schemas/*` drive codegen |
| Realtime state | ~80 `Signal`s on a global `CommunicationChannel`; every emit fans out over Socket.IO as MessagePack `signal_msgpack`; frontend dispatches to Redux by signal name | `@state` dataclasses; mutations broadcast as **JSON-patch envelopes** with revisions over `/ws`; frontend zustand stores apply patches; **optimistic updates** declared on the action |
| Long tasks | Ad-hoc `threading.Thread` per feature; no unified progress/cancel; stop via bespoke signals | First-class task lifecycle (STARTED/PROGRESS/YIELD/PAUSED/CANCELLED/…), `progress()`/`pausepoint()` in-function, **locks** (`stage_position` etc.) enforced server- and client-side, task history DB + `/replay` |
| Video streaming | Unified MessagePack `frame` event with `frame_ack` backpressure; binary(LZ4/Zstd)/JPEG/MJPEG/WebRTC | Dedicated binary WS routes `/stream/zstd/{slot}`, `/stream/h264/{slot}`; shared `FrameBroadcaster` + encoder pool; `fzstd`/`jmuxer` decode |
| Hardware abstraction | `MultiManager` + per-type ABCs (`PositionerManager`, `LaserManager`, …); `managerName` string in JSON → dynamic import | `Protocol` interfaces in `protocols/` + `@state` per device type; concrete managers in `managers/{virtual,uc2}/`; instantiated in `@startup provide_managers` |
| Setup config | JSON setup files (`imcontrol_setups/*.json`): devices (`managerName`/`managerProperties`), feature blocks, `availableWidgets` selector | **None yet** — imperative composition in `app.py:provide_managers`; `ImswitchConfig` pydantic model (`use_virtual_microscope`, `available_cubes`) |
| Plugin system | `imswitch.implugins` entry points (controllers, managers, widgets) | `plugins/` dir exists but is a placeholder; extension = add `@register`/`@context`/`@state` + `register_blok` (server-shipped JSX) |
| Optics model | Flat device lists; pixel size via PixelCalibration | **Light-path graph of "Kubes"** (Objective/Detector/Illumination/Filter/Stage/Dichroic) with affine matrices; `LightPathManager`, `MetadataManager` compute per-image affine/FOV |
| UC2 hardware | Mature: `ESP32Manager` (owns `uc2rest.UC2Client`) + `UC2CANOpenManager` (owns `uc2canopen.UC2Client`), full device-manager stacks on both transports | **Mocked**: `UC2SerialManager` speaks the right JSON `{task, assign_params, qid}` shape but against `mock_resolver`; no `uc2rest`/`uc2canopen` dependency; no CANopen code |
| Maturity | Production, ~75 controllers, huge feature surface, lots of legacy | Alpha framework, virtual microscope works end-to-end; excellent skeleton, tiny feature surface |

---

## 3. The paradigm shift in detail

### 3.1 What an "app" is

**ImSwitch app contract** (all four hand-maintained):

```
availableWidgets: ["Autofocus"]                      # JSON setup file
  → AutofocusController(ImConWidgetController)       # dynamic import by name
      @APIExport() def autoFocusHillClimbing(...)    # → GET /api/AutofocusController/...
      self._commChannel.sigAutoFocusCompleted.emit() # → socket.io "signal_msgpack"
  → (optional) AutofocusManager built in MasterController if "Autofocus" in availableWidgets
  → frontend/src/backendapi/apiAutofocusController*.js   # hand-written axios wrapper
  → frontend/src/state/slices/AutofocusSlice.js          # hand-written Redux slice
  → frontend/src/components/AutofocusController.js       # React widget
  → constants/appRegistry.js entry                       # nav registration
```

**newswitch app contract** (only the component is hand-written frontend code):

```python
# backend/newswitch/protocols/autofocus.py
@state
@dataclass
class AutofocusState: running: bool; best_z: float | None; metric_curve: list[float]

@context
@runtime_checkable
class AutofocusManager(Manager, Protocol):
    def run(self, ...) -> float: ...

# backend/newswitch/routines/autofocus.py   (or managers/…)
@register(locks=["stage_position"])
def run_autofocus(af: AutofocusManager, stage: StageManager,
                  detector: DetectorManager, state: AutofocusState,
                  z_range: float = 100.0, steps: int = 20) -> float:
    for i, z in enumerate(...):
        progress(int(100*i/steps), f"z={z:.1f}")
        pausepoint()
        ...
    return best_z
```

Then: instantiate in `provide_managers`, run `just dev` — the Vite plugin regenerates
`src/apps/default/hooks/actions/runAutofocus.ts` + `states/AutofocusState.ts`, and you write
one component using `useAutofocusState({subscribe: true})` + `<ActionButton action={RunAutofocusDefinition}/>`.

**What disappears in the port** (do not translate these, delete them):

| ImSwitch artifact | newswitch replacement |
|---|---|
| `@APIExport` method + route naming | `@register` function (name = endpoint identity) |
| `CommunicationChannel` signal declarations + `.connect()` wiring | mutate a `@state` object (auto-broadcast) |
| Redux slice + socket-name→slice dispatch in `WebSocketHandler.js` | generated zustand state hook |
| axios wrapper file per endpoint | generated action hook (zod-validated) |
| `availableWidgets` / `getAvailableControllers` gating | app liveness + presence of the registered actions (schema-driven UI) |
| ad-hoc `threading.Thread` + stop flags + `sigXxxStop` signals | task lifecycle: `progress()`, `pausepoint()`, cancel/pause from UI, locks |
| inter-controller `master.getController("Autofocus")` calls | inject the other manager/protocol directly, or call the routine function |

### 3.2 Signals vs. state — the biggest mental shift

ImSwitch is *event-flavored*: "something happened" (`sigUpdateMotorPosition(dict)`), and every
consumer (Python and JS) interprets the payload. newswitch is *state-flavored*: the manager
mutates `StageState.x`, the framework diffs and patches every subscriber. Porting rule of thumb:

- ImSwitch signal that carries **current values** (`sigUpdateMotorPosition`, `sigUpdateLaserPower`,
  `sigHomingState`, temperature/sensor telemetry) → becomes a **field on a `@state` class**.
- ImSwitch signal that means **"task finished/failed"** (`sigAutoFocusCompleted`,
  `sigExperimentStop`, scan-done) → becomes the **task result / task lifecycle event** of the
  `@register` function; no signal needed at all.
- ImSwitch signal that carries **bulk data** (`sigUpdateImage`, reconstruction frames) →
  goes through the **FrameBroadcaster slot** mechanism or is returned/yielded as `Image`
  objects from a routine, not through state.
- Genuinely spontaneous hardware events with no owning task (**e-stop, collision, joystick,
  hardware button press**) → the adapter mutates a dedicated state (e.g. `SafetyState.estop_active`)
  and/or triggers a registered hook (newswitch has a `HookManager`, cf.
  `SoftwareAutofocusHook`). This is the one category with no perfect 1:1 mapping — see §6.4.

### 3.3 Concurrency

ImSwitch: threads everywhere, one shared asyncio loop only for socket emission; blocking
serial I/O is fine because callers are threads. newswitch: asyncio-first; sync `@register`
functions run in worker contexts and use `koil`/`unkoil` to call async managers. Practical
rules for ported code:

- Long-running scan/acquisition loops port well as **sync `@register` functions** with
  `progress()`/`pausepoint()` (see `routines/region_scan.py`) — that model is very close to
  ImSwitch's worker-thread loops, so most algorithmic code moves with minimal change.
- Hardware I/O must not block the event loop. uc2rest is a blocking, thread-based library —
  it must be wrapped (§6.2), not called from async code directly.
- Replace `noqt.Thread/Timer/Worker` with either `@background` loops (for daemons like the
  detector loop, sensor pollers) or plain task functions.

### 3.4 Frontend

The React work per app shrinks dramatically, but it is a **rewrite, not a port**: Redux
Toolkit → zustand-backed generated hooks, MUI-era components → Tailwind/Radix, CRA → Vite,
JS → TS. Plan to re-create each widget's *UI* from scratch using `StageControl.tsx` as the
canonical pattern, keeping only the ImSwitch component as a visual/behavioral spec. Complex
visual assets that are framework-agnostic can be lifted with moderate effort: the wellplate
designer geometry, OpenLayers/deck.gl stage-map logic, Blockly workflow definitions — but each
needs its data flow rebuilt on generated hooks.

One newswitch capability with no ImSwitch equivalent worth exploiting: **`register_blok`**
lets the backend ship a JSX fragment. For long-tail apps (debug panels, one-off instrument
UIs like Lepmon), a blok may be enough — no frontend PR at all.

### 3.5 Configuration

ImSwitch's JSON setup files are load-bearing for the openUC2 product (per-microscope device
lists, axis signs/limits/step sizes, feature blocks, `availableWidgets`). newswitch has
nothing comparable — `provide_managers` is hard-coded and `use_virtual_microscope` is a
boolean. **This must be built** (§8.1): a pydantic `SetupInfo`-like model loaded from
JSON/YAML that `provide_managers` consumes to decide which managers to construct and with
what parameters. Design it against the existing ImSwitch schema so existing
`imcontrol_setups/*.json` files can be converted mechanically (a converter script is cheap
and preserves the installed base).

---

## 4. Concept-mapping cheat sheet

| ImSwitch | newswitch | Notes |
|---|---|---|
| `ImConWidgetController` subclass | set of `@register` functions | logic-heavy controllers → also a `@context` manager |
| `*Manager` (device) + type ABC | `@context` Protocol in `protocols/` + concrete class in `managers/` | port method bodies nearly verbatim |
| `MultiManager` groups (`PositionersManager`…) | multiple manager instances in `provide_managers` (today: one per type) | multi-device-per-type needs a keyed-manager pattern — design decision, §8.1 |
| `MasterController` | `@startup provide_managers` | the composition root |
| `CommunicationChannel` + `noqt.Signal` | `@state` classes + JSON-patch broadcast | see §3.2 mapping rules |
| `@APIExport(runOnUIThread/async)` | `@register(locks=…, optimistics=…)` | locks are new & valuable — use them |
| Socket.IO `signal_msgpack` | `/ws` StatePatchEvent + task events | one socket, typed |
| `frame` event + `frame_ack` | `/stream/zstd/{slot}`, `/stream/h264/{slot}` + `FrameBroadcaster` | camera team's domain, but scan apps push tiles through slots |
| `WorkflowManager` (`WorkflowContext`/`WorkflowStep`) | routines with `progress()`/`pausepoint()` + task engine; compose by calling routines | Blockly authoring layer would need a thin routine-graph runner if kept |
| setup JSON (`managerName`/`managerProperties`) | **missing** — build config loader | §8.1 |
| `availableWidgets` | schema-driven UI (actions exist ⇒ UI works) + per-setup manager instantiation | selective UI needs a small "enabled apps" state |
| PixelCalibration (per-detector px size/flip) | `Kube.affine_matrix` + `MetadataManager` | newswitch model is *richer*; port calibration routines to write affines |
| `imswitchclient` (external REST client) | rekuest client / generated HTTP surface | scripting story changes; keep OpenAPI shim if 3rd parties depend on it (§9 risks) |
| `imswitch.implugins` entry points | not yet; nearest: separate registry + `register_blok` | defer; single-repo apps first |
| MetadataHub / `sharedAttrs` | `MetadataManager` + per-image `Metadata` | |
| `RecordingService`, `io/` writers | **missing** (zarrs/tifffile deps exist; `acquistion_manager` is minimal) | §8.2 |
| File-manager REST routes | `routes/http/files.py` (basic serving only) | §8.2 |

---

## 5. The porting recipe (per app)

Use the stage vertical and `routines/region_scan.py` as templates. For each app:

1. **Classify it** (determines effort):
   - *Pure device app* (Positioner, Laser, LEDMatrix, Objective, Galvo): mostly a manager port.
   - *Algorithm/routine app* (Autofocus, HistoScan, Timelapse, scan controllers): mostly a
     routine port — managers already exist as dependencies.
   - *Hub/config app* (UC2Config, Settings): decompose into many small actions + states.
   - *Legacy/inert* (Qt-era alignment tools, 0-endpoint controllers): **do not port**.
2. **Extract the contract** from the ImSwitch controller: list its `@APIExport` methods, the
   CommChannel signals it emits/consumes, its setup-JSON block, and which managers it touches.
3. **Define Protocol + State** in `newswitch/protocols/<app>.py` (skip the Protocol if the
   app is a pure routine over existing managers). Apply §3.2 to convert each signal into
   either a state field or a task result. Declare `required_locks`.
4. **Port the manager** to `newswitch/managers/…`, implementing the Protocol. Keep the
   algorithmic body; replace signal emissions with state mutation under `acquired_locks(...)`;
   replace thread spawning with the task model. Write a **virtual** implementation too — it's
   what makes tests and frontend dev possible without hardware (ImSwitch's Virtual* managers
   are a good source).
5. **Write `@register` functions** — one per meaningful ImSwitch endpoint, but *consolidate*:
   ImSwitch endpoint counts are inflated (PositionerController: 34; UC2Config: 83). Target
   the operations the frontend actually uses; add `locks`, `optimistics`, `description`.
6. **Wire in `provide_managers`** (behind the config loader once it exists).
7. **Regenerate + build UI**: `just dev`, then one component in `src/components/microscope/`
   modeled on `StageControl.tsx`; `just drift-check` before commit.
8. **Tests**: pytest via `AsyncAgentTestClient` against the virtual manager (see
   `backend/tests/test_api.py`), vitest for any nontrivial component logic.

Rule of thumb for effort: the manager port is ~the same size as the ImSwitch manager; the
controller shrinks ~5–10×; the frontend component is a rewrite but with zero API plumbing.

---

## 6. UC2-REST + UC2-CANopen: the hardware strategy

This is the highest-value, most decision-heavy work package. Current facts:

- newswitch `managers/uc2/serial_manager.py` already defines the right *internal* contract:
  `JSONCommand{task, assign_params, qid}` → `JSONResponse`, plus a `StateUpdate` path
  (`aprocess_state`) that pushes `stage_position_x/y/z` into `StageState`, plus per-command
  cancel/pause/unpause hooks. But `astart()` runs `mock_resolver` — nothing touches hardware.
- ImSwitch proves the useful abstraction boundary: **both transports already converge on the
  same verb surface** (`move_*`, `home_*`, `set_laser/set_value`, `led.fill/pattern`,
  position + done callbacks). Its ESP32* and UC2CANOpen* manager stacks are drop-in-parallel
  precisely because they meet at that surface.
- uc2rest is a mature but **blocking + background-read-thread** library (qid mod-255
  correlation, `++`/`--` framing, pattern-keyed callbacks, ~35 submodules). uc2canopen is a
  young asyncio-agnostic python-can library (expedited SDO, OD index map auto-derived from
  firmware, TPDO listener with `on_motor_done`, command-word doorbell for motor `0x2003` and
  galvo `0x2602`).

### 6.1 Architecture: one protocol surface, two transport backends

```
protocols/stage.py, illumination.py, galvo.py, io.py, safety.py     (transport-agnostic)
        ▲ implemented by
managers/uc2/…            e.g. UC2StageManager, UC2IlluminationManager, UC2GalvoManager
        │ uses
managers/uc2/bus.py       UC2Bus protocol:  async run(command) -> result
        │                                   async subscribe(event_kind) -> AsyncIterator[UC2Event]
        ├── UC2RestBus     wraps uc2rest.UC2Client   (serial JSON, single master board)
        └── UC2CanBus      wraps uc2canopen.UC2Client (SDO/PDO, node-per-board)
```

Key design points:

- **Device managers are transport-agnostic.** One `UC2StageManager` serves both buses; the
  bus adapter translates `move_stage(axis, target, speed, accel, absolute)` into either the
  `/motor_act` JSON task or the OD-write sequence + doorbell. This halves the manager count
  vs. ImSwitch's parallel `ESP32StageManager` / `UC2CANOpenStageManager` stacks and fixes
  their drift problem.
- **Addressing**: the bus adapter owns the address map. REST: everything on one board,
  addressed by JSON fields (axis id 0–3, laser channel 1–3). CAN: axis→node-id
  (X=11, Y=12, Z=13, A=14, LED=20, LASER=21, GALVO=30…). Expose this as bus config, exactly
  like ImSwitch's `nodeIds` overrides.
- **Unified event model** (replaces both uc2rest's pattern-callbacks and uc2canopen's
  `TpdoListener`): the bus emits typed events — `PositionUpdate(axis, pos)`,
  `MotionDone(axis, pos)`, `HomingState(axis, state)`, `EStop(active)`, `Collision(...)`,
  `ButtonEvent(key, data)`, `CanNodeSeen(node, state)`. Device managers subscribe and mutate
  their `@state` objects; that's the whole GUI-update path.

### 6.2 Reuse the libraries or reimplement? → **Wrap first, natively async later (serial only)**

- **UC2-REST (serial)**: *wrap* `uc2rest.UC2Client` initially. Its read thread + qid logic +
  firmware quirks (`nResponses` vs `use_qid_done`, `++`/`--` framing, reconnect, mock
  fallback, master detection) encode years of firmware coupling — do not rewrite that under
  schedule pressure. Wrapping recipe: instantiate the client in the bus adapter; call
  blocking verbs via `asyncio.to_thread(...)`; convert pattern callbacks into events by
  pushing onto a `loop.call_soon_threadsafe`-fed `asyncio.Queue`. This fulfills the existing
  `JSONCommand → JSONResponse / StateUpdate` contract in `serial_manager.py` almost 1:1 —
  the JSON task names are the same protocol. A later phase can replace the wrapper with a
  native `pyserial-asyncio` implementation inside newswitch (the dependency is already
  declared) — worthwhile only once the firmware's `use_qid_done` mode is universal.
- **UC2-CANOPEN**: *wrap and extend the library itself* — it is young enough that improving
  it upstream is cheaper than working around it. python-can's `Notifier` already runs its own
  thread; bridge its listeners into asyncio the same way. Known gaps to fix **in the
  UC2-REST-CANOPEN repo** as part of this effort:
  - **No `Galvo` class** — only OD entries exist (`GALVO_COMMAND_WORD 0x2602` doorbell,
    `GALVO_STATUS_WORD 0x2603`, targets `0x2600`, scan params `0x2604–0x260F`). Write the
    client class mirroring the motor doorbell pattern, honoring the established invariants
    (command word 0 = idle not stop; stop = 4; qid-echo semantics on the REST side).
  - `on_motor_done` exists but there is no generic event subscription (homing, e-stop,
    heartbeat-loss) — add one, since the unified event model needs it.
  - Fix the known adapter mismatch: ImSwitch's `UC2CANOpenManager` passes `debug=` but the
    client takes `log_level=` — don't carry that bug across.
- **Firmware coupling discipline**: uc2canopen's `od.py` is generated from firmware
  `UC2_OD_Indices.h` / `uc2_canopen_registry.yaml`. Keep that generation step in CI and pin
  `uc2rest` / `uc2canopen` versions in newswitch's `pyproject.toml` (ImSwitch pins
  `uc2-rest==0.2.0.39`, `uc2canopen>=0.1.4` — note both local checkouts are ahead of the
  pinned releases; cut releases before integrating).

### 6.3 Functional coverage plan (what "deep support" means)

Wave H1 — core motion & light (unblocks all device apps):
motor move/stop/position/limits/backlash, homing, laser PWM, LED matrix fill/pattern,
position + motion-done + homing events, e-stop.

Wave H2 — platform services:
TMC parameter get/set, soft/hard limits, joystick mapping, objective slider
(`.objective`: home/calibrate/move slot), galvo (DAC set, scan start/stop, arbitrary points,
trigger mode — both transports, incl. the new CANopen Galvo class), digital/analog IO +
trigger tables, stage-scanning (`startStageScanning`, coordinate lists) for HistoScan-class apps.

Wave H3 — fleet/config features (the UC2Config app):
firmware info/master detection, CAN scan/discover/node-reassign/OTA, GPIO collision
protection, PTZ/button events, I2C sensor passthrough, temperature/fan/PID, heap/restart.

### 6.4 Spontaneous hardware events

E-stop, collision, buttons, and joystick don't belong to any task. Pattern: a
`SafetyState` / `InputState` `@state` mutated by the bus event loop (UI reacts instantly via
patch subscription), plus `HookManager` hooks for policy ("on button A → snap image", the
existing ImSwitch `message.register_callback` use case) so behavior stays configurable
without hard-coding it into the bus.

---

## 7. App inventory triage

Of ImSwitch's ~75 controllers, roughly a third deserve porting. Endpoint counts (from the
audit) are a decent proxy for real surface area.

**Wave 0 — already exist in newswitch, verify & harden:** Positioner (StageControl), Laser/LED
(IlluminationControl), Objective, FilterBank, LiveView (camera team), region scan,
multidimensional acquisition, light-path calibration.

**Wave 1 — core parity (needs Wave H1 hardware + config loader):**
| App | Port shape |
|---|---|
| PositionerController (34 EP) | extend existing `move_stage` set: per-axis speed, offsets, soft limits, home-all; `StageState` already fits |
| LaserController / LEDController / LEDMatrixController | extend `IlluminationState`; LED matrix patterns as actions |
| ObjectiveController (10) | manager exists virtually; wire to `.objective` submodule |
| AutofocusController (9) | routine + `AutofocusHook` (hook already scaffolded in newswitch) |
| RecordingController / StorageController / MetadataController | mostly framework work → §8.2 |
| SettingsController (detector params) | camera team, but define the `DetectorState` fields together **early** — every scan app reads them |

**Wave 2 — acquisition apps (the product core):**
ExperimentController (52 EP — decompose into routines: wellplate scan, focus map, z-stack,
channel loop; *not* one giant port), HistoScan (maps beautifully onto `scan_region` +
firmware stage-scanning), WorkflowController/Timelapse (routine composition), StageMap
(FrameBroadcaster tiles + Expanse 3D canvas is arguably a better host than OpenLayers),
PixelCalibration → **re-imagined as affine-calibration writing `Kube.affine_matrix`**
(newswitch's model supersedes it; port the calibration *routines*, not the storage).

**Wave 3 — UC2 platform apps:** UC2ConfigController (83 EP — split into: firmware/OTA app,
CAN-bus app, safety/GPIO app, joystick app; each is a small state+actions cluster),
GalvoScannerController (23), I2CSensorController, TemperatureController, FocusLockController
(32 — PI loop as `@background` + state).

**Wave 4 — long tail, port on demand:** holography suite (InLineHolo/OffAxisHolo — the
reconstruction cores are numpy/scipy and port trivially as routines; UI later or as bloks),
STORMRecon, Lightsheet, DPC/SIM, FlowStop, Lepmon, Goniometer, MazeGame/Demo/Debug (bloks),
SiLA2/Hypha/Arkitekt bridges (re-evaluate: newswitch *is* rekuest/Arkitekt-native, so
ArkitektController is obsolete by construction).

**Do not port:** Qt-era alignment tools (AlignXY, AlignmentLine, ULenses, FFT — inert,
0 endpoints), ConsoleController, duplicated scan controllers superseded by routines
(StageScanAcquisition, SquidStageScan, ROIScan, BeadRec), WatcherController, ImSwitchServer
pseudo-widget, MCT if Experiment/Timelapse covers it (decide with users).

---

## 8. Framework gaps to build in newswitch (prerequisites)

### 8.1 Setup/config system
JSON/YAML → pydantic models → drives `provide_managers`. Requirements: multiple devices per
type (ImSwitch supports N positioners/detectors — newswitch currently instantiates one
`StageManager`; decide between keyed manager registries or one-manager-per-axis-group),
per-device `managerProperties` equivalents (axis signs, step sizes, limits, node ids, bus
selection rest|canopen), feature blocks, an `availableWidgets` analog (which manager sets +
UI apps a given microscope exposes), and a converter script from `imcontrol_setups/*.json`.
This is *the* enabler for shipping newswitch on the existing installed base.

### 8.2 Storage & recording services
Port ImSwitch's `io/` layer (RecordingService, OME-Zarr/OME-TIFF writers, snap service,
stitched-TIFF writer, thumbnails) as a `RecordingManager` context + actions. The deps
(zarrs, tifffile) are already in newswitch. Extend `routes/http/files.py` toward the file
manager surface (list/upload/delete/preview/thumbnail with path-traversal guards — ImSwitch's
implementation is a direct reference). The `useq-schema` dependency suggests aligning
acquisition plans with useq `MDASequence` — worth adopting as the experiment-description
format instead of ImSwitch's bespoke models.

### 8.3 Task/workflow authoring
Routines cover coded workflows. If the Blockly visual authoring in ImSwitch matters to users,
add a small interpreter routine that executes a JSON graph of routine calls — but defer until
Wave 2 demands it.

### 8.4 Multi-app frontend shell
newswitch's IndexPage is a single dashboard. ImSwitch has a nav-drawer app registry. Build a
lightweight app-shell (registry of components, gated by which actions/states exist in the
generated schema — i.e., derive "available apps" from the backend automatically, which is
strictly better than `appRegistry.js` + `getAvailableControllers` cross-checking).

---

## 9. Phased plan, risks, open questions

**Phases** (each ends demoable on a virtual microscope + on UC2 hardware):
1. **Hardware foundation**: UC2Bus + REST wrapper + CAN wrapper, Wave H1 verbs, event model;
   replace `mock_resolver`; virtual bus for CI. *(Everything else stacks on this.)*
2. **Config system** (§8.1) + Wave 1 device apps + converter for existing setup JSONs.
3. **Recording/storage** (§8.2) + Wave 2 acquisition apps.
4. **UC2 platform apps** (Wave 3, incl. CANopen Galvo class + OTA/CAN management).
5. **Long tail on demand** (Wave 4), deprecate ImSwitch per-setup once parity confirmed.

**Risks / open questions:**
- **External API compatibility**: `imswitchclient`, OMERO watcher, SiLA2, and any user
  scripts speak ImSwitch REST paths. Decide early whether to ship a thin compatibility
  router (`/imswitch/api/PositionerController/movePositioner` → `move_stage` action) or
  declare a clean break with a migration guide.
- **rekuest-next dependency**: newswitch's core paradigm lives in an external alpha framework
  by a single author. Budget for pinning + contributing fixes upstream; treat
  `rekuest_next.contrib.fastapi` as part of your own stack operationally.
- **Multi-device-per-type DI**: type-based injection is elegant with one stage; the pattern
  for two stages/detectors (common in UC2 setups) needs a deliberate design (keyed contexts
  or composite managers) before Wave 1 — retrofitting it later touches every manager.
- **Performance envelope on Raspberry Pi**: ImSwitch's fork is tuned for Pi deployments;
  validate rekuest agent + JSON-patch overhead + zstd encoding on target hardware early
  (Phase 1 demo on a Pi, not just a laptop).
- **Team paradigm ramp-up**: the DI/state/codegen model is unfamiliar; the porting recipe
  (§5) plus one exemplary reviewed port (suggest: Objective app — small, touches hardware,
  has UI) should be treated as the training artifact before parallelizing across people.
