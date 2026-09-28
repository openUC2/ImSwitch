# Setup-config JSON reference

This document explains the JSON file that defines a customer/FRAME-specific ImSwitch configuration (the files under `imswitch/_data/user_defaults/imcontrol_setups/*.json`, and the one pointed to by `ImSwitchConfig/imcontrol_setups/<name>.json` on a deployed
Pi). It is derived directly from the Python dataclasses that parse this file —  `imswitch/imcontrol/model/SetupInfo.py` and `imswitch/imcontrol/view/guitools/ViewSetupInfo.py` — plus tracing of how those fields are actually consumed by the backend and frontend.

It exists to answer:

1. **What does each key in the setup JSON do?** (field reference, § 3)
2. **What is deliberately *not* in this schema** — i.e. camera/driver-specific
   parameters that are opaque to ImSwitch itself (§ 4)
3. **How does a key's presence/absence actually control what the customer sees in
   the live software?** (§ 2 — this is the mechanism behind "grey out what isn't in
   the customer's configuration")

A companion machine-readable [`docs/setup-config.schema.json`](setup-config.schema.json) is generated from the same dataclasses by [`scripts/generate_setup_schema.py`](../scripts/generate_setup_schema.py) — see § 6. Re-run that script whenever `SetupInfo.py` changes so the two stay in sync.

> **Scope note:** this document describes the system *as it exists today*, including
> its inconsistencies. It is not a proposal for how the config system should look —
> § 7 lays out the open decisions for the team to make.

---

## 1. The big picture

- The setup JSON is loaded once at startup, parsed via `dataclasses_json` (`Undefined.INCLUDE` mode — unknown top-level keys are tolerated, not rejected)   into a `ViewSetupInfo` object, which is a strict superset of `SetupInfo`   (`ViewSetupInfo(SetupInfo)` in `imswitch/imcontrol/view/guitools/ViewSetupInfo.py:34`).   Everywhere in the backend that reads `setupInfo.X`, it is actually reading from
  this combined object.
- **`SetupInfo`** (`imswitch/imcontrol/model/SetupInfo.py`) holds the *hardware*   config: devices (detectors, lasers, LEDs, positioners, …) and per-feature settings   objects (`focusLock`, `objective`, `sim`, `experiment`, …).
- **`ViewSetupInfo`** adds a handful of *UI-only* fields on top: `rois`,   `ledPresets`, `laserPresets`, and — critically — **`availableWidgets`**, the   master switch described in § 2.
- The Configuration Wizard in the frontend (`frontend/src/components/ConfigurationWizard.js`)   lets a user load/edit/save one of these JSON files through the browser, but it only validates the *filename* and that the text is syntactically valid JSON content against the shape described in this document. That is a real gap; see  § 7.

---

## 2. How "is this feature visible in the software" actually works

This is the mechanism behind the observation that *"some options are visible based on the JSON, some are not."* There isn't one mechanism — there are three, layered, and only some widgets opt into all three.

### 2.1 `availableWidgets` — the master switch (backend, controller instantiation)

`availableWidgets` (declared on `ViewSetupInfo`, **not** on `SetupInfo` itself — see the trap in § 2.4) is a JSON array of widget-key strings, e.g.:

```json
"availableWidgets": ["Settings", "View", "Recording", "Image", "Laser",
                      "Positioner", "Autofocus", "Objective", "HistoScan", "..."]
```

It can also be the boolean `true` (enable everything) or `false` (enable nothing) —
see `ViewSetupInfo.hasWidget()` (`ViewSetupInfo.py:140`).

This list is consumed **before any REST controller exists**, in the headless main view constructor:

```
imswitch/imcontrol/view/ImConMainView.py:26  (class ImConMainViewNoQt)
    enabledDockKeys = list(viewSetupInfo.availableWidgets) ...
    widget_keys = {key: ... for key in enabledDockKeys if key not in disabledKeys}
    self._addWidgetNoQt(widget_keys)   # -> self.widgets
```

`self.widgets` is then the *only* set of widget keys that `imswitch/imcontrol/controller/ImConMainController.py:60` will attempt to build a controller for:

```python
for widgetKey, widget in self.__mainView.widgets.items():
    controller_name = f"{widgetKey}Controller"
    ...
    self.controllers[widgetKey] = self.__factory.createController(controller_class, widget)
```

If `widgetKey` was never in `availableWidgets`, its controller is never even attempted. If it *was* attempted but its `__init__` throws (typically an `AttributeError` because the manager it depends on was never created — see § 2.2), the exception is caught and logged, and the controller simply doesn't exist either way (`ImConMainController.py:119-122`).

Every controller that *did* get built is passed to `generateAPI()` (`ImConMainController.py:247`), which records its class name:

```
imswitch/imcommon/model/api.py:82-87
    from imswitch import __available_controllers__
    for _controller in objs:
        controller_name = _controller.__class__.__name__
        if controller_name not in __available_controllers__:
            __available_controllers__.append(controller_name)
```

...which is exactly what the REST endpoint reports:

```
imswitch/imcontrol/controller/server/ImSwitchServer.py:122
    GET /getAvailableControllers  ->  {"availableControllers": [...]}
```

**So: a widget key absent from `availableWidgets` in the JSON → its controller is
never created → its class name never appears in `/getAvailableControllers` →** (see
§ 2.3 for what the frontend does with that fact).

### 2.2 `MasterController`'s own `availableWidgets` gate (backend, manager instantiation)

Separately, `imswitch/imcontrol/controller/MasterController.py:123-190` re-checks the
*same* `availableWidgets` list to decide whether to build the underlying **manager**
object for ~20 subsystems:

`SLM`, `SIM`, `DPC`, `NIDAQ`, `Hypha`, `ROIScan`, `Lightsheet`, `WebRTC`,
`Timelapse`, `Experiment`, `Objective`, `HistoScan`, `Stresstest`, `FlowStop`,
`Lepmon`, `AutoFocus`, `FOV`, `Workflow`, `Arkitekt`, `SiLa2`.

```python
if "Objective" in self.__setupInfo.availableWidgets:
    self.objectiveManager = ObjectiveManager(self.__setupInfo.objective, setupInfo=self.__setupInfo)
```

If `"Objective"` isn't in the list, `self.objectiveManager` is never set as an
attribute at all — so `ObjectiveController.__init__`, which references
`self._master.objectiveManager`, throws `AttributeError` and gets skipped per § 2.1.
This is somewhat redundant with § 2.1 (both ultimately require the widget key to be
listed), but it's the layer that actually determines whether the *data* object
exists, which matters because a manager can be shared by more than one controller.

**Devices are different.** `detectors`, `lasers`, `LEDs`, `LEDMatrixs`,
`positioners`, `galvoScanners`, `rotators`, `rs232devices` are **not**
`availableWidgets`-gated — their managers are *always* constructed
(`MasterController.py:87-110`), just populated from whatever dict of named devices
was given (possibly empty). `PixelCalibration` is a special case: always
constructed unconditionally regardless of config (`MasterController.py:164-170`,
explicit comment: *"required for the correct functioning of other controllers"*),
and its widget key is force-added to `enabledDockKeys` even if missing from
`availableWidgets` (`ImConMainView.py:29-33`).

### 2.3 Three different things happen when a section is missing — read this before assuming "missing = hidden"

`availableWidgets` controls whether a controller is *attempted*, but what happens
when the controller *is* attempted with its config section `null` varies **per
controller**, and falls into one of four buckets. Knowing which bucket a given
feature is in matters for Step 3 (§7) — some are already safe to leave permanently
visible, some actively need the frontend gate, and one is an outright startup-crash
risk.

**(a) Hard-fails to construct → cleanly absent from `/getAvailableControllers`.**
Only two features actually behave this way today:
- **Objective**: gated by the `"Objective"` **`availableWidgets`** entry, not by
  `objective` section content — `ObjectiveManager` is defensive and falls back to a
  hard-coded default objective set if `objective` is `null`
  (`ObjectiveManager.py:20-45`). `ObjectiveController.__init__` does
  `self._master.objectiveManager`, which raises `AttributeError` only if
  `"Objective"` was missing from `availableWidgets` in the first place.
- **LEDMatrix**: `LEDMatrixController.__init__`
  (`imswitch/imcontrol/controller/controllers/LEDMatrixController.py:11-24`) does a
  **hard-coded lookup** `self._master.LEDMatrixsManager._subManagers['ESP32 LEDMatrix']`.
  This raises `KeyError` — and the controller fails to construct — unless a device
  in `LEDMatrixs` is named **exactly** `"ESP32 LEDMatrix"`. ⚠ A real LED matrix
  configured under any other name is silently treated the same as "not configured."

**(b) Constructs successfully, degrades gracefully — still counts as "available."**
`FocusLockController`, `FOVLockController` do a plain `if setupInfo.focusLock is
None: return` inside `__init__` (`FocusLockController.py:168-169`,
`FOVLockController.py:22-23`) — no exception, so the object still exists and its
class name **still lands in `/getAvailableControllers`**, even though calling any
of its methods would then hit an `AttributeError` on an attribute that was never
set. `SIMController`/`ISMController` similarly check for `None` and call a
Qt-only `replaceWithError()` (legacy desktop-GUI mechanism — irrelevant to the
React frontend), but again don't raise. `DPCController` doesn't even check —
it just logs a warning and continues (`DPCController.py:83-84`); in practice the
`dpcController` app in `appRegistry.js` is gated entirely by
`requiredControllers: ["LEDMatrixController", "DPCController"]`, i.e. it rides on
the LEDMatrix gate above, not on `dpc` section content at all. **None of
FocusLock/FOVLock/SIM/ISM have an `appRegistry.js` entry with `requiredControllers`
set**, so their nav buttons are always visible; SIM/ISM additionally have *no*
`appRegistry.js` entry at all (they're legacy-PyQt-only screens the React frontend
never surfaces).

**(c) Manager ignores its config argument's content entirely — only the
`availableWidgets` key matters, nothing about the section body does.**
`WorkflowManager()` is called with **no arguments at all**
(`MasterController.py:175-176`) — `WorkflowInfo`'s fields are never read anywhere
in the codebase. `HyphaManager`, `ROIScanManager`, `HistoScanManager` likewise
ignore their info argument's content. Same applies to `Stresstest`, `FlowStop`,
`Lepmon`, `WebRTC`. For all of these, writing a detailed section body is
cosmetic — the *only* thing that does anything is whether the widget key string is
present in `availableWidgets`.

**(d) Fully dead — no code reads the section at all, in any form.**
`mockxx`, `jetsonnano`, `pulseStreamer` (defined twice as a field, see §5) have no
manager/controller referencing them anywhere in `imswitch/`. `etSTED` is the most
surprising: `EtSTEDInfo` is defined and documented, but **no
`EtSTEDController.py` file exists anywhere in the repository** — if `"EtSTED"` were
ever added to `availableWidgets`, the dynamic controller import in
`ImConMainController.py:99-109` would simply fail (caught, logged, skipped). Any of
these three can be deleted from a config with zero effect, and none are worth
documenting further than "present for a feature that isn't implemented (yet)."

**⚠ Startup-crash risk — not a hide, a crash.** `ImConMainController.py:69-72`
special-cases the `"Scan"` widget key **outside** the try/except that guards every
other controller's construction:
```python
if widgetKey == "Scan":
    controller_name = f"{widgetKey}Controller{self.__setupInfo.scan.scanWidgetType}"
```
If `"Scan"` is listed in `availableWidgets` but `scan` is `null`, this line raises
`AttributeError` **uncaught**, which crashes the entire server at startup — not a
greyed-out button, a boot failure. If you ever add `"Scan"` to `availableWidgets`,
`scan` must be a fully-populated `ScanInfo` object in the same file.

`microscopeStand` is a special non-gating case worth calling out separately: its
content is read only by `SetupInfo.getMicroscopeStandName()`
(`SetupInfo.py:944-955`) to display a name string — it doesn't hide or show
anything, regardless of whether it's `null`.

### 2.4 Frontend gating — only partially wired to § 2.1/2.2, and not fully centralized even where it is

The React app has a real, generic mechanism for "grey out / redirect away from
what's unavailable," but **only six app entries use it**:

- `frontend/src/backendapi/apiGetAvailableControllers.js` calls
  `GET /getAvailableControllers` once.
- `frontend/src/state/slices/BackendCapabilitiesSlice.js` stores the result.
- `frontend/src/constants/appRegistry.js` — the registry of ~35 apps shown in the
  nav drawer / App Manager — lets any entry declare `requiredControllers: [...]`.
- `frontend/src/hooks/useBackendControllerCapabilities.js` cross-checks the
  *currently open* plugin against that list and bounces back to Live View if its
  required controller(s) are missing.
- `frontend/src/components/navigation/NavigationDrawer.jsx:50-62` filters the nav
  list itself the same way via `isAppAvailableForControllers()`.

The apps that actually declare `requiredControllers` today
(`frontend/src/constants/appRegistry.js`):

| App (registry id)       | `requiredControllers`                          |
|--------------------------|-------------------------------------------------|
| `mmCore`                 | `MMCoreController`                              |
| `stageMap`                | `StageMapController`                            |
| `timelapse`               | `TimelapseController`                           |
| `dpcController`            | `LEDMatrixController`, `DPCController`          |
| `objective`               | `ObjectiveController`                           |
| `extendedLEDMatrix`        | `LEDMatrixController`                           |

**Every other entry in `appRegistry.js` (~29 of ~35 — FocusLock, GalvoScanner,
LightSheet, FlowStop, Lepmon, Stresstest, HoloController, GoniometerController,
WellPlate, Blockly, ImJoy, i2cSensor, MazeGame, …) has no `requiredControllers`, so
`isAppAvailableForControllers()` returns `true` unconditionally for them.** They
remain visible and clickable in the nav drawer / App Manager regardless of
`availableWidgets`, even when their backend controller was never instantiated.
Clicking one whose controller doesn't exist doesn't crash the app — the component
just renders and its API calls fail/return empty, so the user sees a broken-looking
or empty widget instead of the button being greyed out.

**This is precisely the "partially implemented" state described in the task: the
mechanism to grey things out exists and is generic, it's just declared on 6 of ~35
apps.** Closing that gap is a matter of adding `requiredControllers: [...]` entries
to the remaining `appRegistry.js` items (see § 7, Step 3).

**One more wrinkle: even where gating exists, it isn't always routed through
`appRegistry.js`.** A few components bypass the declarative `requiredControllers`
mechanism and call the underlying selector directly with a hard-coded controller
name: `frontend/src/components/FRAMESettingsController.js:44-93` conditionally
renders an "Objective" tab via `useSelector(selectHasController("ObjectiveController"))`,
and `frontend/src/components/TimelapseController.js:20-21,176-183` filters an
illumination-source dropdown via `selectHasController("LEDMatrixController")` —
neither goes through `appRegistry.js` at all. So "add the app to `appRegistry.js`
with `requiredControllers`" closes the nav-drawer/App-Manager gap (§7, Step 3), but
a handful of in-widget checks like these are a separate, one-off pattern to be
aware of if auditing gating exhaustively.

### 2.5 `nonAvailableWidgets` — present in files, read by nothing

Several example configs (`FRAME.json`, `FRAME2b.json`, `example_uc2.json`,
`example_virtual_microscope.json`, …) also set a `nonAvailableWidgets` array, e.g.:

```json
"nonAvailableWidgets": ["STORMRecon", "DPC", "Holo", "FFT", "Hypha", "FocusLock", "FOVLock"]
```

**No code anywhere reads this key** (`grep -rn "\.nonAvailableWidgets\b" imswitch`
returns nothing). It rides into the object through `Undefined.INCLUDE`/`_catchAll`
and does nothing. `example_virtual_microscope.json` even lists `"FocusLock"` in
*both* `availableWidgets` and `nonAvailableWidgets` simultaneously — further proof
it has no effect. **Treat `nonAvailableWidgets` as dead/vestigial** — do not rely on
it to hide anything, and consider removing it from example files to stop it
misleading whoever edits them next.

### 2.6 Summary table

| Mechanism | Where | Governs | Coverage |
|---|---|---|---|
| `availableWidgets` list | `ImConMainViewNoQt` (view) | Whether a controller/widget is attempted at all | All widget keys — the actual master switch |
| `availableWidgets` re-check | `MasterController` | Whether ~20 named managers are constructed | SLM/SIM/DPC/NIDAQ/Hypha/ROIScan/Lightsheet/WebRTC/Timelapse/Experiment/Objective/HistoScan/Stresstest/FlowStop/Lepmon/AutoFocus/FOV/Workflow/Arkitekt/SiLa2 |
| Non-empty device dict | `MasterController` | Whether a device manager has anything to control | detectors/lasers/LEDs/LEDMatrixs/positioners/galvoScanners/rotators/rs232devices (managers always exist) |
| Section-content check inside a controller's `__init__` | Individual controllers (§2.3) | Whether that *one* controller hard-fails (bucket a), soft-degrades (bucket b), or doesn't care (bucket c) | Inconsistent per feature — see §2.3 and the per-field table in §3.1 |
| `requiredControllers` in `appRegistry.js` | Frontend (nav drawer + App Manager) | Whether the app is filtered out of the list entirely | Only 6 of ~35 apps today |
| One-off `selectHasController(...)` calls | A handful of individual components | Whether one tab/dropdown-option within an otherwise-visible widget is shown | `FRAMESettingsController.js`, `TimelapseController.js` only, bypassing `appRegistry.js` |
| `nonAvailableWidgets` | (nowhere) | Nothing | Dead key |

---

## 3. Field-by-field reference

Types are as declared in the dataclasses; `?` marks `Optional`/nullable. "Gates"
notes which widget key(s) in `availableWidgets` this section is tied to, and which
bucket from §2.3 it falls in when the section itself is `null` — **read §2.3
before trusting this column**, since "gates" here means "the key must be in
`availableWidgets`," not "the section content matters."

### 3.1 Top-level keys on `SetupInfo`

| Key | Type | Default | `availableWidgets` key | If section is `null` | Purpose |
|---|---|---|---|---|---|
| `detectors` | `Dict[str, DetectorInfo]` | `{}` | — (always on) | n/a | Cameras. Name → device config (§3.2). |
| `lasers` | `Dict[str, LaserInfo]` | `{}` | — | n/a | Laser lines. |
| `LEDs` | `Dict[str, LEDInfo]` | `{}` | — | n/a | Single-channel LEDs (as opposed to LED matrices). |
| `LEDMatrixs` | `Dict[str, LEDMatrixInfo]` | `{}` | `ExtendedLEDMatrix` (frontend, via `requiredControllers`) | Controller **hard-fails** unless a device is named exactly `"ESP32 LEDMatrix"` (§2.3a) — a real matrix under any other name is treated as absent too. | Addressable LED matrices (illumination patterns, DPC). |
| `positioners` | `Dict[str, PositionerInfo]` | `{}` | — | n/a | Motorized stage axes. |
| `galvoScanners` | `Dict[str, GalvoScannerInfo]` | `{}` | `GalvoScanner` (frontend) | n/a | Galvo-mirror laser scanners. |
| `flimLabs` | `dict?` | `null` | — | n/a | Persisted FLIM LABS bridge settings (browser-side panel state; free-form, written by the controller, not hand-authored). |
| `rs232devices` | `Dict[str, RS232Info]` | `{}` | — | n/a | Serial connections other managers reference by name (e.g. `"ESP32"`). |
| `mmcoreSettings` | `MMCoreSettingsInfo?` | `null` | — | n/a | Persisted MMCore per-device property overrides (written by the controller, not hand-authored). |
| `slm` | `SLMInfo?` | `null` | `SLM` |  |  |
| `sim` | `SIMInfo?` | `null` | `SIM` | Bucket (b): controller shows a Qt-only error, still counts as "available." No `appRegistry.js` entry — legacy-only, not reachable from the React frontend at all. |  |
| `dpc` | `DPCInfo?` | `null` | `DPC` | Bucket (b): controller just logs a warning, doesn't even return early. `appRegistry.js`'s `dpcController` gate rides entirely on `LEDMatrixController`+`DPCController` presence, **not** on `dpc` content. | ⚠ see §5 for real-world field-name/type mismatches. |
| `objective` | `ObjectiveInfo?` | `null` | `Objective` | Bucket (a) *by proxy*: `ObjectiveManager` defensively falls back to a built-in default objective set if `null` — the real gate is the `"Objective"` key being in `availableWidgets` at all, not this field. | Motorized objective/lens turret. |
| `nidaq` | `NidaqInfo` | `NidaqInfo()` | `NIDAQ` | n/a (never Optional) | ⚠ declared twice in `SetupInfo.py` (lines 825 and 916, identical) — harmless (second wins) but worth cleaning up. |
| `roiscan` | `ROIScanInfo?` | `null` | `ROIScan` | Bucket (c): manager ignores content entirely. |  |
| `lightsheet` | `LightsheetInfo?` | `null` | `Lightsheet` | Manager checks `is None` and returns gracefully (bucket b). Marker-only class today (no fields) — no `appRegistry.js` `requiredControllers` either, so its nav entry is always shown. |  |
| `webrtc` | `WebRTCInfo?` | `null` | `WebRTC` | Bucket (c): content-agnostic manager. | Marker-only class. |
| `hypha` | `HyphaInfo?` | `null` | `Hypha` | Bucket (c): content-agnostic manager. | Marker-only class. |
| `mockxx` | `MockXXInfo?` | `null` | — | Bucket (d): fully dead, referenced nowhere. | Marker-only class; not in `MasterController`'s gate list. |
| `jetsonnano` | `JetsonNanoInfo?` | `null` | — | Bucket (d): fully dead. | Marker-only class; not gated. |
| `Stresstest` | `StresstestInfo?` | `null` | `Stresstest` | Bucket (c): content-agnostic manager. `appRegistry.js` has a `stresstest` entry but no `requiredControllers`. | Marker-only class. |
| `HistoScan` | `HistoScanInfo?` | `null` | `HistoScan` | Bucket (c): `HistoScanManager` ignores content (just manages a cache file). No `appRegistry.js` entry. | ⚠ `PreviewCamera` typed `str` but used as `null` — see §5. |
| `Workflow` | `WorkflowInfo?` | `null` | `Workflow` | Bucket (c), extreme case: `WorkflowManager()` is called with **no arguments at all** — `WorkflowInfo`'s fields are never read anywhere. | Marker-only in practice. |
| `FlowStop` | `FlowStopInfo?` | `null` | `FlowStop` | Bucket (c): content-agnostic manager. `appRegistry.js` has a `flowStop` entry but no `requiredControllers`. | Marker-only class. |
| `Lepmon` | `LepmonInfo?` | `null` | `Lepmon` | Bucket (c): content-agnostic manager. `appRegistry.js` has a `lepmon` entry but no `requiredControllers`. | Marker-only class. |
| `PixelCalibration` | `PixelCalibrationInfo?` | `null` | *always constructed* | n/a | Per-detector affine calibration; single source of truth for pixel size + flip. |
| `experiment` | `ExperimentInfo?` | `null` | `Experiment` |  | Also holds OMERO upload settings. |
| `uc2Config` | `UC2ConfigInfo?` | `null` | — | n/a | Marker-only class; UC2Manager is always constructed regardless. |
| `ism` | `ISMInfo?` | `null` | — | Same bucket-(b) pattern as `sim`; not in `MasterController`'s gate list, no `appRegistry.js` entry. |  |
| `focusLock` | `FocusLockInfo?` | `null` | — (not in `MasterController`'s availableWidgets gate list) | Bucket (b): controller does a plain `return` on `None`, still counts as "available." No `appRegistry.js` `requiredControllers` — nav button always visible. | ⚠ several required fields commonly omitted — see §5. |
| `arkitekt` | `ArkitektInfo?` | `null` | `Arkitekt` | Also requires `hasattr(masterController, "arkitektManager")` guard in `ImConMainController.py:80-86` (skips gracefully if the `arkitekt_next` package isn't installed). |  |
| `sila2` | `SiLA2Info?` | `null` | `SiLa2` (note case) | Same `hasattr` guard pattern as `arkitekt` (`ImConMainController.py:87-93`). |  |
| `fovLock` | `FOVLockInfo?` | `null` | `FOV` | Bucket (b): plain `return` on `None`, still "available." No `appRegistry.js` `requiredControllers`. | ⚠ crop fields typed `int` but `null` used in practice — see §5. |
| `autofocus` | `AutofocusInfo?` | `null` | `AutoFocus` (note case) | No `appRegistry.js` entry at all — effectively unreachable from the React frontend regardless of config. |  |
| `scan` | `ScanInfo?` | `null` | `Scan` | **⚠ Crashes the server at startup** if `"Scan"` is in `availableWidgets` but `scan` is `null` — see §2.3, this is not a graceful hide. | Also read by `NidaqManager` for `lineClockLine`/`frameClockLine` hardware triggering. |
| `etSTED` | `EtSTEDInfo?` | `null` | `EtSTED` | Bucket (d): **no `EtSTEDController.py` file exists in the repo at all.** Adding `"EtSTED"` to `availableWidgets` would just fail the dynamic import (caught, logged, skipped). | Not implemented today. |
| `rotators` | `Dict[str, DeviceInfo]?` | `null` | `Rotator`/`RotationScan` (frontend) | n/a | Standa motorized rotator mounts. |
| `microscopeStand` | `MicroscopeStandInfo?` | `null` | — | **Not a gate at all** — content is read only by `SetupInfo.getMicroscopeStandName()` to display a name string; presence/absence hides nothing. | Cosmetic display name only. |
| `storage` | `StorageInfo?` | `null` | — | n/a | Runtime-persisted active data path; not normally hand-authored. |
| `instrument` | `InstrumentInfo?` | `null` | — | n/a | OME metadata (name, tube lens, filters, UC2 frame provenance). |
| `overviewRegistration` | `Dict?` | `null` | — | n/a | Persisted overview-camera slide registration; not hand-authored. |
| `pulseStreamer` | `PulseStreamerInfo` | `PulseStreamerInfo()` | — | Bucket (d): fully dead, no manager/controller reads it. | Always present (not Optional) but has zero effect today. |

### 3.2 `ViewSetupInfo`-only keys (view/UI layer, same JSON file)

| Key | Type | Default | Purpose |
|---|---|---|---|
| `availableWidgets` | `List[str] \| bool` | `[]` | **Master switch** — see §2.1. `true` = all widgets, `false` = none. |
| `nonAvailableWidgets` | *(undeclared — falls into `_catchAll`)* | — | **Dead. Read by nothing.** See §2.5. |
| `rois` | `Dict[str, ROIInfo]` | `{}` | Named ROI presets selectable in detector settings. |
| `ledPresets` | `Dict[str, Dict[str, LEDPresetInfo]]` | `{}` | Preset name → LED name → value. |
| `defaultLEDPresetForScan` | `str?` | `null` |  |
| `laserPresets` | `Dict[str, Dict[str, LaserPresetInfo]]` | `{}` | Preset name → laser name → value. |
| `defaultLaserPresetForScan` | `str?` | `null` |  |

### 3.3 Device entries (`detectors` / `lasers` / `LEDs` / `LEDMatrixs` / `positioners` / `galvoScanners` values)

All device entries share the `DeviceInfo` base:

| Field | Type | Purpose |
|---|---|---|
| `managerName` | `str` | Python manager class to instantiate, e.g. `"HikCamManager"`, `"ESP32StageManager"`. Determines which driver loads — **this is the field that decides camera type, not a schema field.** |
| `managerProperties` | `Dict[str, Any]` | Free-form, driver-specific. **Not part of this schema** — see §4. |
| `analogChannel` / `digitalLine` | `str \| int?` | NI-DAQ analog/digital line, if applicable. |

Type-specific additions:

- **`DetectorInfo`**: `forAcquisition`, `forFocusLock` (bool, both default `False`),
  `defaultStreamSettings` (dict — protocol/subsampling/throttle/compression
  defaults baked in at first stream start; frontend can only override the *current*
  session, not this default).
- **`LaserInfo`**: `valueRangeMin/Max` (nullable), `wavelength` (required),
  `freqRangeMin/Max/Init` (default `0`), `valueRangeStep` (default `1.0`).
- **`LEDInfo`**: `valueRangeMin/Max`, `valueRangeStep`.
- **`LEDMatrixInfo`**: no extra fields (base `DeviceInfo` only).
- **`PositionerInfo`**: `axes` (`List[str]`, required — e.g. `["X","Y","Z","A"]`),
  `isPositiveDirection`, `forPositioning`, `forScanning`, `resetOnClose`,
  `stageOffsets` (named-preset offsets).
- **`GalvoScannerInfo`**: `nx/ny` (samples per line/frame, default 256),
  `x_min/x_max/y_min/y_max` (DAC units 0–4095), `sample_period_us`, `frame_count`
  (0 = continuous), `bidirectional`.

### 3.4 Feature-settings objects

Grouped by theme; field lists are exhaustive per the dataclass, required fields
(no default) are marked **bold**.

**Optics / calibration**
- `ObjectiveInfo`: **`pixelsizes`, `NAs`, `magnifications`, `objectiveNames`,
  `objectivePositions`** (all parallel lists, one entry per objective slot),
  `zPositions`, `homeDirection`, `homePolarity`, `homeSpeed`, `homeAcceleration`,
  `moveSpeed`, `calibrateOnStart`, `active`.
- `PixelCalibrationInfo`: `affineCalibrations` (per-detector 2×3 affine matrix +
  metrics, normally written by the calibration wizard, not hand-authored),
  `defaultAffineMatrix`.
- `FocusLockInfo`: **`camera`, `positioner`, `updateFreq`, `cropCenter`,
  `cropSize`, `piKp`, `piKi`, `focusLockMetric`, `laserName`, `laserValue`**,
  `fovWidth`, `fovCenter`, `calibrationData`. See §5 — several "required" fields
  are routinely omitted in real configs.
- `FOVLockInfo` / `AutofocusInfo`: **`camera`, `positioner`, `updateFreq`,
  `frameCropx/y/w/h`** (+ `piKp`/`piKi` for `FOVLockInfo` only).
- `ScanInfo`: **`scanWidgetType`, `scanDesigner`, `scanDesignerParams`,
  `TTLCycleDesigner`, `TTLCycleDesignerParams`, `sampleRate`, `lineClockLine`,
  `frameClockLine`**. See §5 — one example file uses a completely different shape.
- `SIMInfo` / `SLMInfo` / `ISMInfo`: monitor/pattern/angle config for structured
  illumination / SLM-based techniques.
- `DPCInfo`: **`wavelength`, `pixelsize`, `magnefication`** *(sic — misspelled in
  the dataclass itself, must match in JSON)*, **`NA`, `NAi`, `n`, `rotations`**.
  See §5 for real-world deviations.
- `EtSTEDInfo`: **`detectorFast`, `detectorSlow`, `laserFast`** — device names by
  reference.

**Integrations**
- `ExperimentInfo`: OMERO server/credentials/timeouts (all optional, defaults
  point at `localhost`/disabled), `overviewCameraName`.
- `ArkitektInfo`: `enabled`, `appName`, `redeemToken`, `url`, `syncInAsync`,
  `deconvolveActionHash`.
- `SiLA2Info`: `enabled`, `serverName/Description/Host/Port/Version`, `vendorUrl`.
- `HyphaInfo` / `WebRTCInfo` / `MockXXInfo` / `JetsonNanoInfo` / `LightsheetInfo` /
  `ROIScanInfo` / `StresstestInfo` / `WorkflowInfo` / `FlowStopInfo` / `LepmonInfo` /
  `UC2ConfigInfo`: marker-only classes today (no fields) — their entire purpose in
  the JSON is the *key's presence* (or `availableWidgets` membership), not any
  field inside them. An empty `{}` is the normal/expected value.

**Device/system metadata**
- `InstrumentInfo`: display `name`, `microscopeType`, `manufacturer`, `model`,
  `serialNumber`, `tubeLensFocalLengthMm`/`Magnification`, `uc2Frame*` provenance
  fields, `filters` (list of `{name, filterType, wavelengthNm, bandwidthNm}`).
- `MicroscopeStandInfo`: `name`, `managerName`, `rs232device` — only needed for the
  Leica motorized-correction-collar (`MotCorr`) widget.
- `NidaqInfo`: `timerCounterChannel`, `startTrigger`.
- `PulseStreamerInfo`: `ipAddress`.
- `StorageInfo`: `activeDataPath` (runtime-set, not normally hand-authored).
- `HistoScanInfo`: `PreviewCamera` (detector name to use as HistoScan preview —
  ⚠ typed `str`, used as `null`, see §5).
- `MMCoreSettingsInfo`: `savedProperties` (runtime-persisted, not hand-authored).

---

## 4. What's deliberately *not* part of this schema

`managerProperties: Dict[str, Any]` on every device entry is **intentionally
opaque** — ImSwitch does not know or validate its shape; it is handed verbatim to
whichever manager class `managerName` selects, and each camera/stage/laser driver
defines its own keys. This is the single biggest reason a full strict JSON Schema
for this file isn't practical: `managerProperties` legitimately means something
different for every `managerName`.

Concrete examples pulled from the example configs, to show the range:

```json
// ESP32StageManager (positioners.*.managerProperties) — FRAME.json
{
  "rs232device": "ESP32", "isEnable": true, "enableauto": false,
  "stepsizeX": 0.396, "stepsizeY": 0.396, "stepsizeZ": 0.3125,
  "homeSpeedX": 10000, "homeDirectionX": -1, "homeEndstoppolarityX": 1,
  "backlashX": 0, "initialSpeed": {"X": 15000, "Y": 15000, "Z": 15000, "A": 15000}
}

// HikCamManager (detectors.*.managerProperties) — FRAME.json
{
  "isRGB": 0, "cameraListIndex": 0, "cameraEffPixelsize": 0.2257,
  "mocktype": "OffAxisHolo",
  "hikcam": {"exposure": 0, "gain": 0, "blacklevel": 100,
             "image_width": 1000, "image_height": 1000}
}

// MMCore/Andor manager (detectors.*.managerProperties) — example_mmcore_andor.json
// (pymmcore device adapter + property names, entirely defined by Micro-Manager,
//  not by ImSwitch)
```

If you're documenting or reviewing a customer JSON and hit a `managerProperties`
key you don't recognize: **it's not a bug that this document doesn't explain it** —
go to the specific `<X>Manager.py` under `imswitch/imcontrol/model/managers/` (or
`managers/detectors/`, `managers/positioners/`) named by that entry's
`managerName`, and read what it does with `self._<key>` / `managerProperties.get(...)`.

Two more things that ride along in the JSON but aren't modeled by `SetupInfo`
at all, because of `Undefined.INCLUDE`/`_catchAll` (`SetupInfo.py:924`):
- **Any misspelled or legacy top-level key is silently accepted and ignored** —
  there is no "unknown key" warning today. A typo in a section name (e.g. writing
  `"objectve"`) will not error; it will just silently do nothing.
- `nonAvailableWidgets` (§2.5) is the clearest example of this trap in the wild.

---

## 5. Known inconsistencies between the schema and real configs

Generating `docs/setup-config.schema.json` from the dataclasses and validating
every file in `imswitch/_data/user_defaults/imcontrol_setups/` against it
surfaced concrete, reproducible mismatches. These are **not** hypothetical — they
are the actual shipped example/reference configs. Listed so nobody "fixes" a
config to match the dataclass (or vice versa) without knowing why it diverged.

| File(s) | Section | Problem |
|---|---|---|
| `FRAME.json`, `example_uc2.json`, `example_histo_daheng.json` | `focusLock` | `FocusLockInfo` declares `piKp`, `piKi`, `focusLockMetric`, `laserName`, `laserValue` as **required** (no default), but these real configs omit all five. Either the dataclass grew required fields after these configs were written, or `dataclasses_json` is more lenient than a strict dataclass constructor about missing required fields — worth confirming which, since it affects whether *any* config is safe to omit them. |
| `FRAME2b.json`, `FRAME2b_tucsen.json`, `example_toupcam.json`, `example_tucsen.json`, `example_tucsen_lumencor.json` | `HistoScan` | `HistoScanInfo.PreviewCamera` is typed `str` (not `Optional[str]`) but every one of these configs sets it to `null`. The dataclass annotation should almost certainly be `Optional[str] = None`. |
| `example_histo_daheng.json`, `example_uc2.json`, `example_uc2_lightsheethik.json` | `dpc` | `wavelength` is typed `int` in `DPCInfo` but these configs pass `0.53` — clearly not a wavelength in nm, more likely a coherence/NA-style ratio. The field's type and docstring (copy-pasted from `SLMInfo`) both look wrong for how `dpc.wavelength` is actually used. |
| `example_raspberry_pi_camera.json` | `dpc` | Uses `pixelSize` (capital S) where `DPCInfo` declares `pixelsize` (lowercase), and adds an extra `NA_illu` key not in the dataclass at all, while omitting the required `magnefication`. This file's `dpc` section looks like it was written against a different/older shape of `DPCInfo`. |
| `example_raspberry_pi_camera.json` | `scan` | Value is `{"scanDesigner": {"scanModes": []}}` — `scanDesigner` here is a **nested object**, but `ScanInfo.scanDesigner` is a `str` (a class name to look up), and none of `ScanInfo`'s five other required fields are present. This section does not correspond to `ScanInfo` at all; either it predates the current `ScanInfo` shape or something else reads it directly. |
| `example_raspberry_pi_camera.json` | `focusLock` | Adds a `port` key not in `FocusLockInfo`, and sets `positioner: null` where the field is a required non-optional `str`. |
| `example_virtual_microscope.json` | `focusLock` | Sets `laserName: null` against the required (non-optional) `str` field. |
| `example_virtual_microscope.json` | `fovLock` | Sets `frameCropx/y/w/h: null` against `FOVLockInfo`'s required (non-optional) `int` fields. |
| `SetupInfo.py` itself (not a specific config file) | `nidaq` | The `nidaq: NidaqInfo` field is declared **twice** in the dataclass body (`SetupInfo.py:825` and `:916`, identical type/default). Harmless today — the second definition simply wins — but it's a real duplicate worth removing so the field only exists once. |

**Root-cause pattern across most of these:** several dataclasses (`FocusLockInfo`,
`FOVLockInfo`, `HistoScanInfo`) declare fields as required-and-typed when real
usage clearly needs them to be optional/nullable. This is worth a small, low-risk
cleanup pass (loosen the type hints to match reality) independent of anything else
in this document — it would make the generated schema authoritative instead of
merely descriptive.

---

## 6. The generated JSON Schema

[`docs/setup-config.schema.json`](setup-config.schema.json) is a JSON Schema
(draft-07) mechanically derived from `SetupInfo.py` + `ViewSetupInfo.py` by
[`scripts/generate_setup_schema.py`](../scripts/generate_setup_schema.py). It parses
the dataclass source via Python's `ast` module (no import of the package needed) and
emits one `$defs` entry per dataclass plus the merged top-level `SetupInfo` +
`ViewSetupInfo` properties.

Regenerate it after any change to those two files:

```bash
python3 scripts/generate_setup_schema.py
```

Known limitations (see §5 for the concrete cases):
- It reflects the dataclasses' declared types, which in a few places don't match
  real-world usage — it is **descriptive of intent, not a certified contract**
  until the §5 cleanup happens.
- `managerProperties` is schema'd as `additionalProperties` (any shape) —
  deliberately, per §4.
- Top-level `additionalProperties: true` — deliberately, because of
  `Undefined.INCLUDE` (§4).

Once the §5 inconsistencies are resolved, this schema is suitable for wiring into
the Configuration Wizard's editor (`frontend/src/utils/configValidation.js`) for
live validation — it isn't wired in today (see §7, Step 2).

---

## 7. The process questions

Everything above is a factual account of what the code does today. This section
answers the three "steps" and Dirk/Armin's grey-out recommendation as directly as
the code allows, and separates *technical facts* from *decisions that are yours (or
the team's) to make* — this document can't make organizational calls for you.

### Step 1 — who else should be able to author a customer-specific JSON, besides Bene?

This is a people/process decision, not something the codebase answers. What the
codebase *does* tell you is what's required to do it safely today:

- There is **no schema validation** in the save path today (§1, §6) — a malformed
  or inconsistent config is only caught at runtime, potentially deep into booting
  a specific manager, with an error message that assumes familiarity with the
  Python source.
- The free-form `managerProperties` (§4) means authoring a *new* device entry
  correctly requires reading the specific manager's source, not just this
  document — that's an inherent floor on how self-service this can be made for
  fully novel hardware.
- Editing an *existing* FRAME's `availableWidgets` list (to turn a purchased
  add-on on/off) is comparatively low-risk and well-scoped — a good candidate for
  someone other than Bene to own, *once* §6's schema is wired into the
  Configuration Wizard as a guardrail (Step 2 below).

**Recommendation to put to the team:** split the responsibility along that same
line — a second person (Haoran, or whoever handles order fulfillment) could own
turning `availableWidgets`/device-dict entries on/off for known, already-supported
hardware using the Configuration Wizard, while genuinely new hardware integration
(new `managerName`, new `managerProperties` shape) stays with whoever can read the
manager source, i.e. Bene until the schema/docs are trusted enough to hand off.

### Step 2 — document the current principal settings and parameters

Done by this document (§2–§5) plus the generated schema (§6). Keep it current: the
schema-generation script means "re-run it and diff" is cheap, but the prose in §2–§5
is hand-written and will drift from the code if not revisited alongside future
`SetupInfo.py` / `MasterController.py` / `appRegistry.js` changes.

### Step 3 — which currently-permanent features should also become JSON-configurable?

§2.4 gives you the concrete, actionable list: **every `appRegistry.js` entry
without a `requiredControllers` array is a candidate.** For each one, the fix is
mechanical (one line in `frontend/src/constants/appRegistry.js`) — and it *will*
correctly hide the app for FRAME models that don't list the corresponding widget
key in `availableWidgets` at all, which covers the main scenario ("this customer's
FRAME doesn't have this hardware option"), **because every controller, regardless
of which §2.3 bucket it's in, still requires its widget key to be in
`availableWidgets` before it's attempted at all** (§2.1's master switch applies
universally).

**The caveat is narrower than "does the fix work at all":** for the §2.3-bucket-(b)
features specifically (`focusLock`, `fovLock`, `sim`, `ism`, and — riding on
`LEDMatrixController` — `dpc`), the controller *still* constructs successfully and
counts as "available" even when the widget key is present in `availableWidgets`
but the actual `SetupInfo` section (e.g. `focusLock`) was left `null` — i.e. "this
FRAME model supports Focus Lock" vs. "this unit's Focus Lock is actually
calibrated/configured" collapse into the same green light today. Closing *that*
finer-grained gap needs a small code change in each bucket-(b) controller (raise,
or otherwise signal unconfigured-vs-absent), not just an `appRegistry.js` entry —
worth scoping as a follow-up once the coarse gap is closed. Bucket-(c) features
(content-agnostic managers) don't have this problem: for them, "widget key in
`availableWidgets`" is the *only* thing that ever mattered, so the mechanical fix
is exact.

Concretely, prioritize this list for `requiredControllers` treatment (grouped by
how likely they are to be genuinely FRAME-configuration-dependent, i.e. hardware
options vs. always-available software tools):

- **Hardware-dependent, high priority:** `galvoScanner` (GalvoScannerController),
  `lightsheet` (LightSheet), `flowStop` (FlowStop), `lepmon` (Lepmon Controller),
  `i2cSensor` (Environmental Sensors), `goniometerController` (Goniometer) — each
  corresponds to add-on hardware a given FRAME may or may not have, and (being
  bucket-(c) content-agnostic managers, §2.3) the mechanical fix is exact for all
  of these. `focusLock` (FocusLock) belongs here too but is bucket-(b) — the
  `appRegistry.js` fix handles "FRAME doesn't have Focus Lock hardware" correctly,
  just not "has the hardware but it's uncalibrated" (see the caveat above).
- **Config-dependent but software-only:** `stresstest`, `holoController`,
  `offAxisHoloController`, `wellSelector` — depend on whether the relevant
  `SetupInfo` section / detector role is configured, not on extra hardware per se.
- **Probably fine to leave ungated (genuinely universal):** `blockly`, `imjoy`,
  `jupyterNotebook`, `serialDebug`, `socketView`, `acceptanceTest`,
  `detectorTrigger`, `mazeGame` — these are development/debug tools or generic
  utilities that don't depend on FRAME hardware options.

That split is a starting proposal, not a final answer — confirm the "universal"
bucket with Bene, since some of these (e.g. `detectorTrigger`) may still assume a
specific device role exists.

### Dirk & Armin's grey-out recommendation

The mechanism they're asking for already exists end-to-end (§2.4) — it's a matter
of coverage, not new engineering. Two remaining decisions:

1. **"Greyed out and visible" vs. "hidden."** Today, `isAppAvailableForControllers`
   is used to *filter the list* (hide), not to render-disabled — check
   `NavigationDrawer.jsx` / `AppManager.jsx` if the literal grey-out (visible but
   disabled, per Dirk/Armin's ask) is wanted instead of removal from the list;
   that's a small presentational change once the coverage gap in Step 3 is closed.
2. **`nonAvailableWidgets` (§2.5) looks like an earlier, abandoned attempt at
   exactly this feature** (explicitly listing what's unavailable). Decide whether
   to finish wiring it up or delete it from example configs — leaving it as
   dead-but-present JSON is actively misleading to whoever edits those files next.
