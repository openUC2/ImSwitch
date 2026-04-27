# ExperimentController Code Review & Claude Code Refactoring Plan

## Executive summary

`ExperimentController.py` is **3,287 lines** (146 KB) and `startWellplateExperiment` alone runs several hundred lines, performing argument coercion, validation, mode selection, focus mapping, Z-stack derivation, directory creation, OME writer configuration, and dispatch to two execution modes — all inline. This is the central readability problem; everything else flows from it.

The refactor below is split into two parts:

1. **A diagnosis** of concrete readability/maintainability issues, ordered by impact.
2. **A Claude Code instruction set** you can paste into a CLAUDE.md or feed to Claude Code as a sequence of tasks. Each task is small enough to land as one PR and is verifiable in isolation.

---

## Part 1 — Diagnosis

### 1.1 The function does too much

`startWellplateExperiment` mixes ~10 distinct concerns:

1. Unpacking `mExperiment.parameterValue` into ~30 local variables.
2. Type coercion for fields that may arrive as scalar or list (`illumination`, `illuIntensities`, `gains`, `exposureTimes`).
3. Passthrough-illumination detection and override logic.
4. `keepIlluminationOn` resolution (auto/on/off → bool).
5. Speed defaulting.
6. Autofocus parameter extraction via `getattr` chains.
7. Concurrency guard (`workflow_manager` status check).
8. LED status, detector start, scan-area cache, focus-map phase.
9. Snake-tile generation and Z-offset list construction.
10. Directory/filename building, OME writer flag setup.
11. Mode dispatch with two long keyword-argument call sites.

Each of these is a candidate for extraction. The biggest single win is collapsing the **30+ local `getattr` extractions** into a typed config object that is built once and passed by reference.

### 1.2 The Pydantic models are already there — they're just not used downstream

`ParameterValue` and `Experiment` are well-defined Pydantic models. By the time the code reaches `startWellplateExperiment`, all type information is thrown away in favor of local primitive variables (`nTimes`, `tPeriod`, `isZStack`, …). The `getattr(p, 'autoFocusMode', 'software')` calls are particularly telling — those defaults already exist on the model. Reading from `p.autoFocusMode` directly removes ~15 lines of `getattr` boilerplate.

### 1.3 List-or-scalar coercion is duplicated and fragile

```python
if type(illuminationIntensities) is not List and type(illuminationIntensities) is not list:
    illuminationIntensities = [p.illuIntensities]
```

This pattern appears four times. Issues:

- `typing.List` is a generic alias, not a runtime type — `type(x) is not List` is always `True`, so the second clause is the only one that matters. The check is doing the right thing by accident.
- Should be `isinstance(x, list)` with a single negation.
- Belongs on the Pydantic model as a `@field_validator` so it runs once at the API boundary, not in the controller.

### 1.4 The dispatch to `normal_mode.execute_experiment` passes 30+ kwargs

The current call site is a wall of keyword arguments mixing snake_case and camelCase. The execution mode classes already exist (`ExperimentNormalMode`, `ExperimentPerformanceMode`) — they should accept a single typed `ExecutionContext` dataclass (or the validated `Experiment` itself plus a small `RuntimeContext`), not an exploded kwargs list.

### 1.5 Magic strings and unit conversions live inline

- `p.performanceTPreMs / 1000.0` (ms→s) is done inline at the call site. Should be a property or a `to_seconds()` method on the model.
- `"auto" / "on" / "off"` for `keepIlluminationOn` and `"software" / "hardware"` for `autoFocusMode` are stringly-typed — these should be `enum.Enum` or `Literal` types.

### 1.6 Side effects are interleaved with computation

The function alternates between (a) building the workflow plan and (b) mutating controller/device state (LED, detector start, `self._last_scan_areas`, `self._initial_experiment_position`, `self._ome_write_*` flags). Pulling all the state mutation into a single `_prepare_for_run(...)` step makes the function read top-to-bottom as: *validate → plan → prepare → dispatch*.

### 1.7 The 3287-line file is itself the problem

Even after fixing the function, the file holds: 9 Pydantic models, the controller class, focus-map glue, OMERO endpoints, LED helpers, snake-tile generation, MDA models, and more. Targets:

- **`experiment_controller/models.py`** — all `BaseModel` classes.
- **`experiment_controller/wellplate_endpoints.py`** — wellplate-layout endpoints.
- **`experiment_controller/omero_endpoints.py`** — OMERO endpoints.
- **`experiment_controller/focus_map_runner.py`** — `_run_focus_map_phase` + helpers.
- **`experiment_controller/runtime.py`** — `ExecutionContext` dataclass + builder.
- **`ExperimentController.py`** — slim orchestrator only.

### 1.8 Smaller issues worth fixing in the same pass

- Trailing-comma typo on Pydantic fields turns scalar defaults into 1-tuples:
  ```python
  brightfield: bool = 0,
  darkfield: bool = 0,
  differentialPhaseContrast: bool = 0,
  ```
  These three are silently `(False,)` (a tuple), not `False`. The model still validates because Pydantic coerces, but it's a real bug landmine.
- `os.makedirs(self.save_dir) if not os.path.exists(self.save_dir) else None` → `os.makedirs(self.save_dir, exist_ok=True)`.
- Bare `except:` around `self.mStage = ...` should be `except (KeyError, IndexError)` or similar.
- Mixing camelCase (`startWellplateExperiment`, `nTimes`) and snake_case (`set_led_status`, `_run_focus_map_phase`) on the same class. Pick one (PEP 8 says snake_case for methods); keep the camelCase only on `@APIExport` decorators if backward compatibility with the React client matters — and document that explicitly.
- Comments like `# TODO: Is this still needed?` and `# TODO: Maybe not needed!` next to live code should be resolved or filed as issues, not left as inline noise.
- `self._isRGB` vs `self.isRGB` vs `self.mDetector._isRGB` — at least three styles for the same concept; pick one accessor.
- `self._writer_thread`, `self._writer_thread_ome`, `self._current_ome_writer`, `self._stop_writer_evt` — these belong on a small `WriterState` dataclass.

### 1.9 The frontend API contract

The React API call (`apiExperimentControllerStartWellplateExperiment.js`) sends a JSON payload that maps directly to the `Experiment` Pydantic model. The leverage point: **once the model is hardened (validators, enums, defaults), the frontend gets clearer 422 errors instead of silent misbehavior**, and the OpenAPI schema generated by FastAPI becomes a usable source of truth for the React side.

---

## Part 2 — Claude Code instruction set

Save the block below as `CLAUDE.md` at the repo root, or feed each task to Claude Code one at a time. Tasks are ordered so each builds on the previous, and **each task is independently shippable** — meaning you can stop after any task and the code will still work.

> **Conventions for every task**
> - Make minimal diffs. Do not reformat unrelated code.
> - Preserve the public REST API surface — every `@APIExport` endpoint must keep its name, HTTP method, and request/response shape unless a task explicitly says otherwise.
> - Run the existing test suite after each task; if no tests cover the changed code path, write one before refactoring (characterization test).
> - Do not remove `# TODO` comments without first checking with the user — file them as GitHub issues if they should leave the codebase.

---

### Task 1 — Establish a safety net

**Goal:** Before touching `startWellplateExperiment`, capture its current behavior so refactors are verifiable.

1. Find existing tests under `imswitch/imcontrol/` that import `ExperimentController`. List them.
2. If none exercise `startWellplateExperiment`, write a characterization test in `tests/imcontrol/test_experiment_controller_start.py` that:
   - Instantiates the controller with mocked `_master`, `_commChannel`, `_setupInfo`.
   - Calls `startWellplateExperiment` with a minimal valid `Experiment` (1 point, 1 channel, no Z-stack, no autofocus, performance mode off).
   - Asserts the return value (`{"status": "running", "mode": "normal"}` or similar) and that `normal_mode.execute_experiment` was called with the expected `snake_tiles` and `z_positions=[0.0]`.
3. Add a second test for the passthrough-illumination branch (all `illuIntensities` are 0/None).
4. Add a third test for `performanceMode=True` when `performance_mode.is_hardware_capable()` returns True.

Do not change production code in this task.

---

### Task 2 — Fix the silent tuple bug in `ParameterValue`

**Goal:** Eliminate accidental tuple defaults.

In `ParameterValue`, change:

```python
brightfield: bool = 0,
darkfield: bool = 0,
differentialPhaseContrast: bool = 0,
```

to:

```python
brightfield: bool = False
darkfield: bool = False
differentialPhaseContrast: bool = False
```

Verify by adding a test that asserts `ParameterValue(...).brightfield is False` (not `(False,)`).

---

### Task 3 — Move list-or-scalar coercion onto the Pydantic model

**Goal:** Remove the `if type(x) is not list` blocks from the controller.

In `ParameterValue`, add field validators (Pydantic v2 syntax — check what version the project uses first):

```python
from pydantic import field_validator

@field_validator("illumination", "illuIntensities", "gains", "exposureTimes", mode="before")
@classmethod
def _coerce_to_list(cls, v):
    if v is None:
        return v
    return v if isinstance(v, list) else [v]
```

Then in `startWellplateExperiment`, delete the four `if type(...) is not List` blocks. The variables are guaranteed to be lists when they reach the controller.

Add a unit test that posts a scalar `illumination` and confirms it round-trips as a one-element list.

---

### Task 4 — Replace stringly-typed fields with `Literal` types

**Goal:** Make invalid values a 422 at the API boundary.

In `ParameterValue`, change:

```python
keepIlluminationOn: str = Field("auto", ...)
autoFocusMode: str = "software"
autoFocusSoftwareMethod: str = "scan"
performanceTriggerMode: str = Field("hardware", ...)
```

to:

```python
from typing import Literal

KeepIlluminationMode = Literal["auto", "on", "off"]
AutoFocusMode = Literal["software", "hardware"]
AutoFocusSoftwareMethod = Literal["scan", "hillClimbing"]
TriggerMode = Literal["hardware", "software"]

keepIlluminationOn: KeepIlluminationMode = "auto"
autoFocusMode: AutoFocusMode = "software"
autoFocusSoftwareMethod: AutoFocusSoftwareMethod = "scan"
performanceTriggerMode: TriggerMode = Field("hardware", ...)
```

Apply the same to `FocusMapConfig.method`, `FocusMapConfig.af_algorithm`, `FocusMapConfig.af_mode`, `FocusMapConfig.af_software_method`. Verify the OpenAPI schema now exposes enums (check `/docs`).

---

### Task 5 — Add convenience properties to `ParameterValue`

**Goal:** Stop computing things in the controller that the model already knows.

Add to `ParameterValue`:

```python
@property
def performance_t_pre_s(self) -> float:
    return self.performanceTPreMs / 1000.0

@property
def performance_t_post_s(self) -> float:
    return self.performanceTPostMs / 1000.0

@property
def passthrough_illumination(self) -> bool:
    """True when no illumination intensities are set."""
    return not any(self.illuIntensities or [])

@property
def n_active_channels(self) -> int:
    return sum(1 for v in (self.illuIntensities or []) if v and v > 0)

def resolve_keep_illumination_on(self) -> bool:
    """Resolve the auto/on/off setting to a concrete bool."""
    if self.keepIlluminationOn == "on":
        return True
    if self.keepIlluminationOn == "off":
        return False
    # "auto"
    return self.n_active_channels == 1
```

Replace the equivalent inline logic in `startWellplateExperiment`. The controller should now read `p.passthrough_illumination`, `p.resolve_keep_illumination_on()`, `p.performance_t_pre_s` instead of computing those itself.

---

### Task 6 — Extract an `ExecutionContext` dataclass

**Goal:** Replace the 30-kwarg call to `normal_mode.execute_experiment` with a single object.

Create `imswitch/imcontrol/controller/controllers/experiment_controller/execution_context.py`:

```python
from dataclasses import dataclass, field
from typing import List, Optional
from .models import Experiment  # after Task 8 moves models out

@dataclass
class ExecutionContext:
    experiment: Experiment            # the validated Pydantic input
    snake_tiles: list                 # output of generate_snake_tiles
    z_positions: List[float]          # relative offsets from base Z
    illumination_sources: List[str]
    illumination_intensities: List[float]
    exposures: List[float]
    gains: List[float]
    initial_z_position: float
    initial_xyz: dict                 # {"X": ..., "Y": ..., "Z": ...}
    dir_path: str
    file_name: str
    keep_illumination_on: bool
    is_rgb: bool
    timepoint_index: int = 0          # set per-iteration in the time-lapse loop
```

Update `ExperimentNormalMode.execute_experiment(self, ctx: ExecutionContext)` and `ExperimentPerformanceMode.execute_experiment(self, ctx: ExecutionContext)` to accept the dataclass. Inside those methods, read from `ctx.experiment.parameterValue.autoFocusMode` etc. — do not re-flatten.

In `startWellplateExperiment`, build the `ExecutionContext` once outside the time-lapse loop, then mutate only `ctx.timepoint_index` per iteration (or pass it as a separate argument to keep the dataclass frozen — your call).

This is the single largest readability win and should be done as one focused PR.

---

### Task 7 — Extract preparation helpers from `startWellplateExperiment`

**Goal:** The function reads top-to-bottom: *guard → prepare → plan → dispatch*.

Add private methods to `ExperimentController`:

```python
def _guard_concurrent_runs(self) -> None:
    """Raise 400 if another workflow is already running."""

def _apply_speed(self, requested: float) -> None:
    """Set self.SPEED_X/Y/Z from requested or defaults."""

def _cache_scan_areas(self, mExperiment: Experiment) -> None:
    """Populate self._last_scan_areas for focus-map API access."""

def _capture_initial_position(self) -> dict:
    """Read stage XYZ once at the start of a run. Returns {'X','Y','Z'}."""

def _build_z_offsets(self, p: ParameterValue) -> list[float]:
    """Return relative Z offsets ([0.0] if z-stack disabled)."""

def _build_output_paths(self, exp_name: str) -> tuple[str, str]:
    """Return (dir_path, file_name) for this run."""

def _apply_ome_writer_flags(self, p: ParameterValue, snake_tiles: list) -> None:
    """Set self._ome_write_* flags based on params and tile structure."""

def _apply_passthrough_overrides(
    self,
    p: ParameterValue,
    illu_sources: list,
    illu_intensities: list,
    gains: list,
    exposures: list,
    keep_on: bool,
) -> tuple[list, list, list, list, bool]:
    """If passthrough mode, swap in sentinel values. Returns updated tuple."""
```

Then `startWellplateExperiment` becomes a roughly 40-line orchestrator:

```python
@APIExport(requestType="POST")
def startWellplateExperiment(self, mExperiment: Experiment):
    self._guard_concurrent_runs()
    p = mExperiment.parameterValue

    self._apply_speed(p.speed)
    self.set_led_status("rainbow")
    if not self.mDetector._running:
        self.mDetector.startAcquisition()

    self._cache_scan_areas(mExperiment)
    self._illuminationIntensities = p.illuIntensities  # for autofocus_software lookup
    self._illuminationSources = p.illumination

    if mExperiment.focusMap and mExperiment.focusMap.enabled:
        self._run_focus_map_phase(mExperiment, mExperiment.focusMap)
        self._switch_off_all_illumination()

    snake_tiles = self.generate_snake_tiles(mExperiment)
    snake_tiles = [[pt for pt in tile if pt is not None] for tile in snake_tiles]

    z_positions = self._build_z_offsets(p)
    dir_path, file_name = self._build_output_paths(mExperiment.name)
    self._apply_ome_writer_flags(p, snake_tiles)
    self._initial_experiment_position = self._capture_initial_position()

    illu_sources, illu_intensities, gains, exposures, keep_on = (
        self._apply_passthrough_overrides(
            p, p.illumination, p.illuIntensities, p.gains, p.exposureTimes,
            p.resolve_keep_illumination_on(),
        )
    )

    ctx = ExecutionContext(
        experiment=mExperiment,
        snake_tiles=snake_tiles,
        z_positions=z_positions,
        illumination_sources=illu_sources,
        illumination_intensities=illu_intensities,
        exposures=exposures,
        gains=gains,
        initial_z_position=self._initial_experiment_position["Z"],
        initial_xyz=self._initial_experiment_position,
        dir_path=dir_path,
        file_name=file_name,
        keep_illumination_on=keep_on,
        is_rgb=self.mDetector._isRGB,
    )

    if p.performanceMode and self.performance_mode.is_hardware_capable():
        self.performance_mode.execute_experiment(ctx)
        return {"status": "running", "mode": "performance"}

    for t in range(p.numberOfImages):
        ctx.timepoint_index = t
        self.normal_mode.execute_experiment(ctx)

    return {"status": "running", "mode": "normal"}
```

Run the characterization tests from Task 1 — they should still pass.

---

### Task 8 — Split the file by concern

**Goal:** Bring `ExperimentController.py` under ~800 lines.

Create the package:

```
imswitch/imcontrol/controller/controllers/experiment_controller/
    __init__.py
    models.py              # Move all BaseModel classes here
    execution_context.py   # From Task 6
    wellplate_endpoints.py # getAvailableWellplateLayouts, getWellplateLayout, generateCustomWellplateLayout
    omero_endpoints.py     # getOMEROConfig / setOMEROConfig / isOMEROEnabled / getOMEROConnectionParams
    led_status.py          # set_led_status helper
    focus_map_runner.py    # _run_focus_map_phase and any helpers it calls
    snake_tiles.py         # generate_snake_tiles, get_num_xy_steps
```

Re-export from `experiment_controller/__init__.py` so existing imports keep working:

```python
from .models import Experiment, ParameterValue, Point, FocusMapConfig, ScanArea  # etc.
from .execution_context import ExecutionContext
```

Keep `ExperimentController.py` as the controller class only. The endpoint methods that delegate to a sub-module should be one-liners:

```python
@APIExport(requestType="GET")
def getAvailableWellplateLayouts(self):
    return wellplate_endpoints.list_layouts()
```

Update imports across the codebase. Run the full test suite.

---

### Task 9 — Tidy the small stuff (low-risk pass)

In a single PR, fix:

- `os.makedirs(self.save_dir) if not os.path.exists(self.save_dir) else None` → `os.makedirs(self.save_dir, exist_ok=True)`.
- Bare `except:` blocks → catch specific exceptions.
- `type(x) is not List` (already removed by Task 3 — confirm none remain).
- Resolve or file as issues every `# TODO:` and `# TODO: Maybe not needed!` in the touched code paths.
- Replace `getattr(p, 'autoFocusMode', 'software')` patterns with direct `p.autoFocusMode` access (the model now guarantees the field exists).
- Standardize on `self.mDetector.isRGB` (the public property) — remove `_isRGB` access.

---

### Task 10 — Add a typed return model

**Goal:** Tighten the API contract for the React client.

Currently `startWellplateExperiment` returns `{"status": "running", "mode": "performance"}` as a raw dict. Define:

```python
class StartExperimentResponse(BaseModel):
    status: Literal["running", "queued", "rejected"]
    mode: Literal["normal", "performance"]
    experiment_id: Optional[str] = None  # if you have one
    started_at: datetime
```

Annotate the endpoint's return type. The React side can now consume a stable schema instead of a free-form object, and FastAPI's OpenAPI generation will reflect it.

---

### Task 11 — Update the frontend API wrapper (separate PR, after backend is stable)

In `frontend/src/backendapi/apiExperimentControllerStartWellplateExperiment.js`:

- Regenerate or hand-update the TypeScript types from the new OpenAPI schema (the project may use `openapi-typescript` or similar — check first).
- If the wrapper currently does any client-side coercion that the backend now handles (e.g. wrapping scalar `illumination` as a list), remove it.
- Surface the new typed `StartExperimentResponse` so callers get autocomplete on `.mode`.

---

## Suggested order if you only have time for three tasks

1. **Task 2** (the silent tuple bug) — five-line fix, ships today.
2. **Task 6** (`ExecutionContext`) — biggest readability impact for the dispatch site.
3. **Task 7** (extract preparation helpers) — turns `startWellplateExperiment` into a readable orchestrator.

Tasks 8 (file split) and 11 (frontend) can wait until the controller logic stabilizes.

---

## What this refactor explicitly does *not* change

- The REST API surface — every `@APIExport` keeps its name, method, and payload shape. The React client should keep working without modification through Task 10.
- The execution semantics of `normal_mode` and `performance_mode` — only how they're called.
- The Pydantic field names exposed to JSON — internal Python names can be changed via `Field(alias=...)` if needed, but the wire format stays the same.

Anything that would break the frontend should be flagged in the PR description and discussed before merging.