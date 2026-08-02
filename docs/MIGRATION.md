# Migration to Newswitch

ImSwitch is being superseded by **newswitch** (React/Vite frontend) and
**newswitch-py** (FastAPI + `rekuest-next` backend). This document tracks the
per-feature port from this repo to the new stack.

Architectural overview and worked example: see the planning document at
`~/.claude/plans/we-are-currently-working-valiant-goose.md` (or the team's
shared copy once published).

---

## Why we are doing this

Per-feature cost in ImSwitch today: controller file + optional manager + JSON
setup entry + hand-written axios client per endpoint + Redux slice + React
component + socket subscription. No type contract between Python and TypeScript.

Per-feature cost in newswitch: one `@register`-decorated Python function (plus
an optional protocol/state class if new hardware), zero handwritten frontend
client code, one React component that imports a generated, Zod-validated hook.

Concurrency safety, optimistic UI, progress reporting, and pause/resume are
built-in via `rekuest-next` rather than reimplemented per controller.

---

## Decision: clean-room rewrite

ImSwitch is reference material only. No compatibility shims, no bridge layer.
Each feature is rewritten from scratch in newswitch following the patterns
documented below. The two repos run in parallel until cutover (Phase 7).

---

## Status legend

- `[ ]` not started
- `[~]` in progress
- `[x]` ported & verified (virtual + hardware where applicable)
- `[-]` dropped (no longer needed)

---

## Migration phases

| Phase | Scope                                                                          | Status |
| ----- | ------------------------------------------------------------------------------ | ------ |
| 0     | Audit ImSwitch controllers, classify (driver / workflow / calibration / meta) | `[ ]`  |
| 1     | Hardware protocols + UC2 + Virtual managers in newswitch-py                    | `[ ]`  |
| 2     | Core action routines (stage, illumination, detector, objective)                | `[~]`  |
| 3     | Workflow routines (one per ImSwitch controller, see table below)               | `[ ]`  |
| 4     | UI parity — React component per routine in newswitch                           | `[ ]`  |
| 5     | Image streaming parity (Zstd / H.264 / vendor codecs)                          | `[ ]`  |
| 6     | Setup-file format (replace `imcontrol_setups/*.json`)                          | `[ ]`  |
| 7     | Cutover — freeze ImSwitch, point users at newswitch                            | `[ ]`  |

---

## Hardware drivers (Phase 1)

For each kind of hardware, newswitch-py needs (a) a `Protocol` + `@state`
class in `newswitch/protocols/`, (b) a `Virtual…` implementation, and (c) a
hardware-talking implementation (UC2 or vendor).

| Hardware                          | Newswitch protocol            | Virtual | UC2 / vendor | Notes                                  |
| --------------------------------- | ----------------------------- | ------- | ------------ | -------------------------------------- |
| Stage / positioners               | `StageManager` / `StageState` | `[x]`   | `[x]`        | Already in newswitch-py                |
| Illumination (single LED)         | `IlluminationManager`         | `[x]`   | `[x]`        |                                        |
| Detector (single camera)          | `DetectorManager`             | `[x]`   | `[ ]`        | Vendor list TBD in Phase 0              |
| Objective turret                  | `ObjectiveManager`            | `[x]`   | `[ ]`        |                                        |
| Filter bank                       | `FilterBankManager`           | `[x]`   | `[ ]`        |                                        |
| Multi-channel lasers              | TBD                           | `[ ]`   | `[ ]`        | New protocol needed                    |
| Galvo scanners                    | TBD                           | `[ ]`   | `[ ]`        |                                        |
| Autofocus actuator                | TBD                           | `[ ]`   | `[ ]`        |                                        |
| Multi-detector configurations     | extend `DetectorManager`      | `[ ]`   | `[ ]`        |                                        |
| OMERO / Zarr writer               | extend `IOManager`            | `[ ]`   | `[ ]`        |                                        |

---

## Controllers → routines (Phase 3)

Port order recommended below: smallest first to prove the pattern, then
calibrations, then the `ExperimentController` monolith last.

| ImSwitch controller                       | Newswitch routine(s)                                  | Status | PR  |
| ----------------------------------------- | ----------------------------------------------------- | ------ | --- |
| `WellPlateController`                     | `move_to_well`, `home_plate`                          | `[ ]`  |     |
| `StageCenterCalibrationController`        | `calibrate_stage_center` (uses `pausepoint()`)        | `[ ]`  |     |
| `StageOffsetCalibrationController`        | `calibrate_stage_offset`                              | `[ ]`  |     |
| `PixelCalibrationController`              | `calibrate_pixel_size`                                | `[ ]`  |     |
| `GalvoAffineCalibrationWizard`            | `calibrate_galvo_affine`                              | `[ ]`  |     |
| `AutofocusController`                     | `run_autofocus`                                       | `[ ]`  |     |
| `MCTController`                           | `run_mct`                                             | `[ ]`  |     |
| `PositionerController`                    | covered by core `move_stage` / `move_home`            | `[~]`  |     |
| `LaserController` / `LEDMatrixController` | `set_illumination_intensity`, `turn_on_illumination`  | `[~]`  |     |
| `ViewController` / `ImageController`      | covered by `FrameBroadcaster` + Expanse on frontend   | `[ ]`  |     |
| `RecordingController`                     | TBD — likely `start_recording` / `stop_recording`     | `[ ]`  |     |
| `ExperimentController` (~160 KB)          | split: `run_wellplate_experiment`, `run_zstack`,      | `[ ]`  |     |
|                                           | `run_timelapse`, `run_multidim_acquisition`           |        |     |
| `SettingsController`                      | covered by `ImswitchConfig` + setup file (Phase 6)    | `[ ]`  |     |
| _(add rows during Phase 0 audit)_         |                                                       |        |     |

---

## Per-feature port checklist

Copy this into each port PR's description.

- [ ] Routine file created in `newswitch/routines/<feature>.py`.
- [ ] All hardware access goes through injected `Protocol` parameters — no `_master`-style global lookups.
- [ ] `locks=[...]` declared for every shared resource the routine touches.
- [ ] Pure helpers extracted into `newswitch/<feature>/` (no `rekuest` dependency) so they unit-test in isolation.
- [ ] `provide_managers()` updated only if a new state/manager class was introduced.
- [ ] Routine registered in `create_app()` (or imported in `app.py` if using `@register` at module scope).
- [ ] Backend restart → `GET /schemas/implementations` includes the new action.
- [ ] Frontend `yarn dev` → generated hook appears in `src/apps/default/hooks/actions/`.
- [ ] React component built and wired into the relevant page in newswitch.
- [ ] Manual end-to-end test in virtual mode (browser).
- [ ] Manual end-to-end test against real hardware (when available).
- [ ] `tests/test_<feature>.py` covers happy path and at least one error path.
- [ ] Row in this file flipped from `[ ]` → `[x]` with the PR link.

---

## ImSwitch → newswitch concept map

| ImSwitch                                       | Newswitch equivalent                                          |
| ---------------------------------------------- | ------------------------------------------------------------- |
| `controllers/XxxController.py` class           | One or more `@register` functions in `routines/xxx.py`        |
| `@APIExport()` method                          | `@register` decorator on the function                         |
| `model/managers/XxxManager.py`                 | `managers/virtual/xxx.py` + `managers/uc2/xxx.py`             |
| `self._master.detectorsManager[...]`           | `detector: DetectorManager` parameter (rekuest DI)            |
| Qt `sigXxx = Signal(...)` then `socket.emit`   | Mutate a `@state` field → automatic `STATE_UPDATE` broadcast  |
| `imcontrol_setups/*.json` `availableWidgets`   | Component imports in `frontend/src/pages/IndexPage.tsx`       |
| `frontend/src/backendapi/apiXxx.js` (axios)    | Generated `src/apps/default/hooks/actions/xxx.ts`             |
| Redux slice + selector                         | Generated state hook + scoped Zustand store for UI-only state |
| `WebSocketContext` subscription                | `useAction()` / `useXxxState()` lifecycle                     |
| Threading inside controller                    | `@background` decorator, or `await` in an async routine       |
| Long-running progress polling                  | `progress(pct, msg)` inside the routine                       |
| Cancel button hack                             | `pausepoint()` + Rekuest's task control                       |

---

## Cutover gate (Phase 7)

Do not proceed until every box below is checked.

- [ ] Every actively used ImSwitch setup has a newswitch equivalent verified on hardware.
- [ ] At least one full week of production lab use on newswitch with no fallback to ImSwitch.
- [ ] User-facing docs migrated.
- [ ] Sign-off from each early-adopter lab lead.
- [ ] This repo set to read-only on GitHub with a `README.md` notice pointing to newswitch.

---

## Open questions

1. Setup-file format (Phase 6): mimic `imcontrol_setups/*.json` shape, or move to a fresh nested YAML?
2. Repo layout: newswitch + newswitch-py in one org, monorepo, or separate orgs?
3. Which labs run Phase 7 cutover first? Their hardware dictates Phase 1 priorities.
4. Mirror this file into newswitch-py as `MIGRATION_FROM_IMSWITCH.md`?
