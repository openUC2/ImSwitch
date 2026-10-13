"""Run a job: walk the tree, drive a ``Microscope``, stream frames to a sink.

``Run(job, hw, sink=...).start()`` validates the job and runs it on its own
thread; ``run()`` does the same synchronously. Control: ``pause``, ``resume``,
``finish`` (stop at the next node, keep data), ``abort(force=...)`` (also stop
the stage and the pumps and switch the light off), ``answer`` for question nodes.

Every node starts at a checkpoint: stop first, then pause. Loops push a frame
(index, count, item) that expressions read as ``loop.<id>`` and that gives
captured frames their useq index (t, p, g, z, c). Events go to ``on_event`` as
plain dicts, and one JSON line per node goes to ``run.jsonl`` in ``run_dir``.
"""
from __future__ import annotations

import itertools
import json
import threading
import time
import traceback
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol

import numpy as np
import useq

from . import expressions
from .control import CONTROL_SPECS, MOVING_LOOPS, POSITION_LOOPS
from .document import CaptureChannel, Job, Node, Retry, expression, resource_ref
from .focus_map import FocusMap
from .hardware import DerivedImage, Microscope
from .registry import REGISTRY, TaskSpec
from .storage import FrameInfo, ImageSpec, NgffImage
from .validate import FROM_KEY, validate

POLL_S = 0.2


class FrameSink(Protocol):
    def write(self, frame: np.ndarray, info: FrameInfo) -> None: ...
    def close(self) -> Any: ...


class RunFailed(RuntimeError):
    def __init__(self, node_id: str, cause: BaseException):
        super().__init__(f"{node_id}: {cause}")
        self.node_id, self.cause = node_id, cause


class _Stop(Exception):
    """finish() or abort() was called."""


class _End(Exception):
    """A finish / abort / exit_with_error node."""

    def __init__(self, state: str, message: str):
        super().__init__(message)
        self.state, self.message = state, message


class _Break(Exception):
    def __init__(self, loop_id: Optional[str]):
        super().__init__(loop_id)
        self.loop_id = loop_id


class _Continue(Exception):
    pass


@dataclass
class Loop:
    node_id: str
    task: str
    axis: Optional[str]
    count: Optional[int]
    index: int = 0
    item: Any = None
    items: Optional[list] = None      # the items, when known up front
    focus_map: Optional[str] = None   # resource whose surface sets Z at each item

    def view(self) -> dict:
        return {"index": self.index, "count": self.count, "item": self.item}


class _Device(Mapping):
    """``device.*`` in expressions: read lazily from the hardware."""

    def __init__(self, hw: Microscope):
        self._hw = hw
        self._read = {"position": hw.get_position, "temperature_c": hw.read_temperature,
                      "sensors": hw.read_sensors, "inputs": lambda: _Inputs(hw)}

    def __getitem__(self, key):
        return self._read[key]()

    def __contains__(self, key):
        return key in self._read

    def __iter__(self):
        return iter(self._read)

    def __len__(self):
        return len(self._read)


class _Inputs(Mapping):
    def __init__(self, hw: Microscope):
        self._hw = hw
        self._names = list(hw.devices("digital_input"))

    def __getitem__(self, name):
        return self._hw.read_digital_input(name)

    def __contains__(self, name):
        return name in self._names

    def __iter__(self):
        return iter(self._names)

    def __len__(self):
        return len(self._names)


class _TaskContext:
    """What tasks get as ``ctx`` (engine side of registry.TaskContext)."""

    __arkitekt_task__ = True

    def __init__(self, run: "Run"):
        self._run = run

    def progress(self, percentage: int, message: Optional[str] = None) -> None:
        self._run._emit("task_progress", percentage=percentage, message=message)

    def log(self, message: str) -> None:
        self._run._log(message)

    def pausepoint(self) -> None:
        self._run._checkpoint()

    def check_cancelled(self) -> None:
        if self._run._stop.is_set():
            raise _Stop()


def _bool(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in ("1", "true", "yes", "on")
    return bool(value)


def _resource_value(res: Any) -> Any:
    """The part of a resource a parameter means when it says '@name'."""
    return {"positions": lambda r: r.positions, "capture": lambda r: r.channels,
            "regions": lambda r: r.regions}.get(res.kind, lambda r: getattr(r, "plan", r))(res)


class Run:
    def __init__(self, job: Job, hw: Microscope, *, parameters: Optional[dict] = None,
                 sink: Optional[FrameSink] = None,
                 on_event: Optional[Callable[[dict], None]] = None,
                 tasks: Optional[Mapping[str, TaskSpec]] = None, run_dir=None,
                 run_id: Optional[str] = None):
        self.job, self.hw, self.sink = job, hw, sink
        self.tasks = REGISTRY if tasks is None else tasks
        self.on_event = on_event or (lambda event: None)
        self.run_id = run_id or uuid.uuid4().hex[:12]
        self.run_dir = Path(run_dir) if run_dir else None
        self.params = self._parameters(parameters or {})
        self.variables: Dict[str, Any] = dict(job.variables)
        self.results: Dict[str, Any] = {}
        self.last_frames: Dict[str, np.ndarray] = {}   # per capture node, for analysis tasks
        self.surfaces: Dict[str, Dict[str, FocusMap]] = {}   # focus_map resource -> group -> fit
        self.loops: List[Loop] = []
        self.state, self.message, self.current = "idle", "", None
        self.frames = 0
        self.question: Optional[dict] = None
        self._answer: Optional[str] = None
        self._stop, self._resume = threading.Event(), threading.Event()
        self._resume.set()
        self._stop_state = "finished"
        self._t0 = time.monotonic()
        self._thread: Optional[threading.Thread] = None
        self._ctx = _TaskContext(self)
        self._pixel_size: Optional[float] = None
        self._bit_depth: Optional[int] = None
        self._lit: Optional[tuple] = None     # (key, channel) of the light that is on

    # -- public control -------------------------------------------------------
    def prepare(self, confirmed: bool = False) -> None:
        report = validate(self.job, self.tasks, capabilities=self.hw.capabilities())
        if not report.ok:
            raise ValueError("; ".join(f"{i.node}: {i.message}" for i in report.errors))
        if report.ask_first and not confirmed:
            raise PermissionError(f"confirm these steps first: {', '.join(report.ask_first)}")

    def run(self, confirmed: bool = False) -> str:
        self.prepare(confirmed)
        self._execute()
        return self.state

    def start(self, confirmed: bool = False) -> threading.Thread:
        self.prepare(confirmed)
        self.state = "running"
        self._thread = threading.Thread(target=self._execute, name=f"aufgaben-{self.run_id}")
        self._thread.start()
        return self._thread

    def join(self, timeout: Optional[float] = None) -> None:
        if self._thread:
            self._thread.join(timeout)

    def pause(self) -> None:
        self._resume.clear()

    def resume(self) -> None:
        self._resume.set()

    def finish(self, message: str = "finished on request") -> None:
        self._request_stop("finished", message)

    def abort(self, force: bool = False, message: str = "aborted on request") -> None:
        self._request_stop("aborted", message)
        if force:
            for stop in (self.hw.stop_stage, self.hw.stop_pumps, self.hw.illumination_off):
                self._try(stop)

    def answer(self, text: str) -> None:
        self._answer = str(text)

    def status(self) -> dict:
        return {"run_id": self.run_id, "job": self.job.name, "state": self.state,
                "message": self.message, "node": self.current, "frames": self.frames,
                "path": [f"{lp.node_id}[{lp.index + 1}/{lp.count or '?'}]" for lp in self.loops],
                "elapsed_s": round(time.monotonic() - self._t0, 2), "question": self.question,
                "variables": dict(self.variables)}

    # -- lifecycle --------------------------------------------------------------
    def _request_stop(self, state: str, message: str) -> None:
        self._stop_state, self.message = state, message
        self._stop.set()
        self._resume.set()

    def _execute(self) -> None:
        self._t0 = time.monotonic()
        self._set_state("running")
        self._record({"event": "start", "job": json.loads(self.job.to_json()),
                      "parameters": self.params, "time": time.time()})
        final, message = "finished", ""
        try:
            self._run_nodes(self.job.tasks)
        except _Stop:
            final, message = self._stop_state, self.message
        except _End as e:
            final, message = e.state, e.message
        except RunFailed as e:
            final, message = "failed", str(e)
            self._record({"event": "error", "node": e.node_id, "traceback": "".join(
                traceback.format_exception(e.cause))})
        finally:
            self.current = None
            self._lit = None
            self._try(self.hw.illumination_off)
            if self.sink is not None:
                self._try(self.sink.close)
        self._record({"event": "end", "state": final, "message": message,
                      "frames": self.frames, "time": time.time()})
        self._set_state(final, message)

    def _set_state(self, state: str, message: str = "") -> None:
        self.state, self.message = state, message or self.message
        self._emit("state", state=state, message=self.message)

    def _emit(self, kind: str, **payload) -> None:
        try:
            self.on_event({"type": kind, "run_id": self.run_id, **payload})
        except Exception:  # a broken listener never stops the run
            pass

    def _log(self, message: str, level: str = "info") -> None:
        self._emit("log", level=level, message=message, node=self.current)
        self._record({"event": "log", "level": level, "message": message, "node": self.current})

    def _record(self, line: dict) -> None:
        if self.run_dir is None:
            return
        self.run_dir.mkdir(parents=True, exist_ok=True)
        with open(self.run_dir / "run.jsonl", "a") as f:
            f.write(json.dumps(line, default=str) + "\n")

    def _try(self, fn: Callable[[], Any]) -> None:
        try:
            fn()
        except Exception as e:
            self._log(f"{getattr(fn, '__name__', fn)} failed: {e}", "warning")

    def _checkpoint(self) -> None:
        if self._stop.is_set():
            raise _Stop()
        if not self._resume.is_set():
            self._set_state("paused")
            while not self._resume.wait(POLL_S):
                pass
            if self._stop.is_set():
                raise _Stop()
            self._set_state("running")

    def _wait(self, seconds: float) -> None:
        if seconds > 0 and self._stop.wait(seconds):
            raise _Stop()

    # -- parameters and expressions -------------------------------------------
    def _parameters(self, given: dict) -> dict:
        unknown = set(given) - set(self.job.parameters)
        if unknown:
            raise ValueError(f"unknown parameters: {sorted(unknown)}")
        cast = {"integer": int, "number": float, "boolean": _bool, "string": str, "choice": str}
        out = {}
        for name, p in self.job.parameters.items():
            value = given.get(name, p.default)
            if value is None:
                raise ValueError(f"parameter {name!r} needs a value")
            value = value if p.type == "choice" else cast[p.type](value)
            if p.choices and value not in p.choices:
                raise ValueError(f"parameter {name!r}: {value!r} is not one of {p.choices}")
            if (p.minimum is not None and value < p.minimum) or \
                    (p.maximum is not None and value > p.maximum):
                raise ValueError(f"parameter {name!r}: {value} is out of range")
            out[name] = value
        return out

    def _namespace(self) -> dict:
        return {"param": self.params, "var": self.variables,
                "loop": {lp.node_id: lp.view() for lp in self.loops}, "result": self.results,
                "device": _Device(self.hw),
                "job": {"name": self.job.name, "run_id": self.run_id,
                        "elapsed_s": time.monotonic() - self._t0}}

    def _eval(self, text: str) -> Any:
        return expressions.evaluate(text, self._namespace())

    def _resolve(self, node: Node, model, expr_fields=()) -> Any:
        raw = dict(node.params)
        if FROM_KEY in raw:
            res = self.job.resources[resource_ref(raw.pop(FROM_KEY))]
            base = {k: v for k, v in res.model_dump().items() if k in model.model_fields}
            raw = {**base, **raw}
        values = {}
        for key, value in raw.items():
            if key in expr_fields:
                values[key] = value
            elif (ref := resource_ref(value)) is not None:
                values[key] = _resource_value(self.job.resources[ref])
            elif (text := expression(value)) is not None:
                values[key] = self._eval(text)
            else:
                values[key] = value
        return model.model_validate(values)

    # -- walking --------------------------------------------------------------
    def _run_nodes(self, nodes: Iterable[Node]) -> None:
        for node in nodes:
            self._checkpoint()
            if self._active(node):
                self._run_node(node)

    def _active(self, node: Node) -> bool:
        enabled = node.enabled if isinstance(node.enabled, bool) else bool(self._eval(node.enabled))
        if not enabled or node.every is None:
            return enabled
        loops = [lp for lp in self.loops if lp.node_id == node.every.of] if node.every.of \
            else self.loops
        index, nth = loops[-1].index, node.every.nth
        return index % nth == 0 if node.every.include_first else (index + 1) % nth == 0

    def _run_node(self, node: Node) -> None:
        previous, self.current = self.current, node.id
        self._emit("node", node=node.id, task=node.task)
        started = time.monotonic()
        policy = node.on_error
        retries, then = (policy.retry, policy.then) if isinstance(policy, Retry) else (0, policy)
        for attempt in itertools.count():
            try:
                self.results[node.id] = self._dispatch(node)
                break
            except (_Stop, _End, _Break, _Continue, RunFailed):
                raise
            except Exception as e:
                self._log(f"{node.id} failed: {e}", "error")
                if attempt < retries:
                    continue
                self.results[node.id] = None
                if then == "skip":
                    break
                if then == "continue_loop" and self.loops:
                    raise _Continue() from e
                raise RunFailed(node.id, e) from e
        self._record({"event": "node", "node": node.id, "task": node.task,
                      "seconds": round(time.monotonic() - started, 3),
                      "result": _plain(self.results.get(node.id))})
        self.current = previous

    def _dispatch(self, node: Node) -> Any:
        spec = CONTROL_SPECS.get(node.task)
        if spec is not None:
            params = self._resolve(node, spec.params, spec.expr_fields)
            return getattr(self, f"_do_{node.task}")(node, params)
        task = self.tasks[node.task]
        params = self._resolve(node, task.params_model)
        kwargs = {name: getattr(params, name) for name in task.params_model.model_fields}
        for name, kind in task.injected.items():
            kwargs[name] = self.hw if kind == "hw" else self._ctx
        result = task.function(**kwargs)
        return self._save_derived(node, result) if isinstance(result, DerivedImage) else result

    def _save_derived(self, node: Node, image: DerivedImage) -> dict:
        """Keep a task's image with the run (OME, stage coordinates) and describe it."""
        data = np.asarray(image.data)
        self.last_frames[node.id] = data
        h, w = data.shape[:2]
        info = {"shape": [h, w], "pixel_size_um": image.pixel_size_um,
                "origin_um": list(image.origin_um), "path": None}
        if self.run_dir is not None:
            stem = f"{node.id}_{image.name}"
            spec = ImageSpec(shape=(1, 1, 1, h, w), dtype=str(data.dtype),
                             pixel_size_um=image.pixel_size_um,
                             origin_um=(*image.origin_um, 0.0), channel_names=[image.name],
                             name=stem)
            out = NgffImage.create(self.run_dir / f"{stem}.ome.zarr", spec)
            out.group.attrs["imswitch"] = {"node": node.id, "derived": image.name}
            out.write(data, 0, 0, 0)
            out.finalize()
            if "ome-tiff" in self.job.storage.formats:
                out.export_ome_tiff(self.run_dir / f"{stem}.ome.tif")
            info["path"] = str(self.run_dir / f"{stem}.ome.zarr")
        return info

    def _loop(self, node: Node, axis: Optional[str], items: Iterable[Any],
              count: Optional[int], move: Optional[Callable[[Any], None]] = None,
              focus_map: Optional[str] = None) -> None:
        frame = Loop(node.id, node.task, axis, count,
                     items=items if isinstance(items, list) else None, focus_map=focus_map)
        self.loops.append(frame)
        try:
            for index, item in enumerate(items):
                self._checkpoint()
                frame.index, frame.item = index, item
                if move is not None:
                    move(item)
                try:
                    self._run_nodes(node.body)
                except _Continue:
                    continue
                except _Break as b:
                    if b.loop_id in (None, node.id):
                        break
                    raise
        finally:
            self.loops.pop()

    # -- loops ----------------------------------------------------------------
    def _innermost(self, *tasks: str) -> Optional[Loop]:
        return next((lp for lp in reversed(self.loops) if lp.task in tasks), None)

    def _move_item(self, item: dict, focus_map: Optional[str] = None) -> None:
        """XY of a position/well/region/tile item; Z from the focus surface or the item."""
        target = {"X": item["x"], "Y": item["y"]}
        z = self._surface_z(focus_map, item["x"], item["y"]) if focus_map else item.get("z")
        if z is not None:
            target["Z"] = z
        self.hw.move(target)

    def _xy_loop(self, node: Node, axis: str, items: list, p) -> None:
        """Run a position-like loop: move to each item (following a focus map if given)."""
        fm = resource_ref(node.params.get("focus_map")) if getattr(p, "focus_map", None) else None
        move = (lambda item: self._move_item(item, fm)) if p.move else None
        self._loop(node, axis, items, len(items), move, focus_map=fm)

    def _do_loop_positions(self, node, p):
        items = [{"name": pos.name or f"p{i:03d}", "x": pos.x, "y": pos.y, "z": pos.z}
                 for i, pos in enumerate(p.source)]
        self._xy_loop(node, "p", items, p)

    def _do_loop_wells(self, node, p):
        ref = resource_ref(node.params.get("plate"))
        labels = self.job.resources[ref].labels if ref else {}
        items = []
        for pos, (row, col) in zip(p.plate.selected_well_positions, p.plate.selected_well_indices):
            if p.label is None or labels.get(pos.name) == p.label:
                items.append({"name": pos.name, "x": pos.x, "y": pos.y, "z": None,
                              "row": int(row), "col": int(col), "label": labels.get(pos.name)})
        self._xy_loop(node, "p", items, p)

    def _do_loop_regions(self, node, p):
        items = [{"name": r.name, "x": (r.min_x + r.max_x) / 2, "y": (r.min_y + r.max_y) / 2,
                  "z": None, "min_x": r.min_x, "max_x": r.max_x, "min_y": r.min_y,
                  "max_y": r.max_y, "label": r.label}
                 for r in p.regions if p.label is None or r.label == p.label]
        self._xy_loop(node, "p", items, p)

    def _do_loop_grid(self, node, p):
        anchor = self._innermost(*POSITION_LOOPS)
        width, height = self.hw.fov_um()
        plan = p.plan
        if plan is None:   # cover the current region
            if anchor is None or "min_x" not in anchor.item:
                raise ValueError("loop_grid without a plan needs an enclosing loop_regions")
            r = anchor.item
            plan = useq.GridFromEdges(top=r["max_y"], left=r["min_x"], bottom=r["min_y"],
                                      right=r["max_x"], overlap=p.overlap * 100,
                                      fov_width=width, fov_height=height)
        if hasattr(plan, "iter_grid_positions"):
            points = list(plan.iter_grid_positions(plan.fov_width or width,
                                                   plan.fov_height or height))
        else:
            points = list(plan)
        if plan.is_relative:
            ax, ay = (anchor.item["x"], anchor.item["y"]) if anchor else \
                (lambda pos: (pos["X"], pos["Y"]))(self.hw.get_position())
        else:
            ax, ay = 0.0, 0.0
        items = [{"name": pt.name or f"{i:04d}", "x": ax + pt.x, "y": ay + pt.y,
                  "z": None, "row": getattr(pt, "row", None), "col": getattr(pt, "col", None)}
                 for i, pt in enumerate(points)]
        self._xy_loop(node, "g", items, p)

    def _do_zstack(self, node, p):
        anchor_loop = self._innermost(*POSITION_LOOPS, "loop_grid")
        anchor = anchor_loop.item.get("z") if anchor_loop and not anchor_loop.focus_map else None
        if anchor is None:
            anchor = self.hw.get_position()["Z"]
        values = list(p.plan)
        zs = [anchor + v for v in values] if p.plan.is_relative else values
        step = abs(zs[1] - zs[0]) if len(zs) > 1 else None
        items = [{"z": z, "step": step} for z in zs]
        self._loop(node, "z", items, len(items),
                   (lambda item: self.hw.move({"Z": item["z"]})) if p.move else None)
        if p.move:
            self.hw.move({"Z": anchor})   # back to where the stack started

    def _do_loop_channels(self, node, p):
        try:
            self._loop(node, "c", p.channels, len(p.channels))
        finally:
            self._dark()   # a kept light stays on across the loop, not after it

    def _do_time_lapse(self, node, p):
        offsets = [float(o) for o in p.plan]
        interval = offsets[1] - offsets[0] if len(offsets) > 1 else None

        def rounds():
            start = time.monotonic()
            k = 0
            while k < len(offsets):
                if k > 0:
                    if p.until and self._eval(p.until):
                        return
                    while p.on_overrun == "skip" and k + 1 < len(offsets) and \
                            time.monotonic() > start + offsets[k + 1]:
                        k += 1
                    if node.idle and time.monotonic() < start + offsets[k]:
                        self._run_nodes(node.idle)
                    self._wait(start + offsets[k] - time.monotonic())
                yield {"t_s": offsets[k], "interval_s": interval}
                k += 1
        self._loop(node, "t", rounds(), len(offsets))

    def _do_repeat(self, node, p):
        def rounds():
            for i in itertools.count():
                if (p.count is not None and i >= p.count) or \
                        (i > 0 and p.until and self._eval(p.until)):
                    return
                yield i
        self._loop(node, None, rounds(), p.count)

    # -- flow -----------------------------------------------------------------
    def _do_condition(self, node, p):
        ok = bool(self._eval(p.expr))
        self._run_nodes(node.body if ok else node.orelse)
        return ok

    def _do_block(self, node, p):
        self._run_nodes(node.body)

    _do_phase = _do_block

    def _do_break(self, node, p):
        raise _Break(p.loop)

    def _do_continue(self, node, p):
        raise _Continue()

    def _do_finish(self, node, p):
        raise _End("finished", p.message)

    def _do_abort(self, node, p):
        raise _End("aborted", p.message)

    def _do_exit_with_error(self, node, p):
        raise _End("failed", p.message)

    def _do_wait(self, node, p):
        self._wait(p.seconds)

    def _do_wait_until(self, node, p):
        deadline = time.monotonic() + p.timeout_s if p.timeout_s else None
        while not self._eval(p.expr):
            if deadline is not None and time.monotonic() >= deadline:
                if p.on_timeout == "abort":
                    raise _End("failed", f"{node.id}: timed out waiting for {p.expr}")
                return False
            self._checkpoint()
            self._wait(p.poll_s)
        return True

    def _do_question(self, node, p):
        self._answer = None
        self.question = {"node": node.id, "text": p.text, "buttons": p.buttons}
        self._emit("question", **self.question)
        deadline = time.monotonic() + p.timeout_s if p.timeout_s else None
        try:
            while self._answer is None:
                self._checkpoint()
                if deadline is not None and time.monotonic() >= deadline:
                    raise _End("failed", f"{node.id}: no answer within {p.timeout_s} s")
                self._wait(POLL_S)
        finally:
            self.question = None
        if p.store_as:
            self.variables[p.store_as] = self._answer
        return self._answer

    def _do_set_variable(self, node, p):
        self.variables[p.name] = p.value
        return p.value

    def _do_log(self, node, p):
        self._log(p.message, p.level)

    # -- acquisition ----------------------------------------------------------
    def _do_move_to_item(self, node, p):
        loop = next(lp for lp in reversed(self.loops)
                    if lp.task in MOVING_LOOPS and (p.loop is None or lp.node_id == p.loop))
        if loop.task == "zstack":
            self.hw.move({"Z": loop.item["z"]})
        else:
            self._move_item(loop.item, loop.focus_map)

    def _keep_light(self, node: Node) -> bool:
        """Whether the capture resource behind this capture keeps the light on."""
        source = node
        if node.params.get("channels") is None:   # inside loop_channels
            loop = self._innermost("loop_channels")
            source = next(n for n, _ in self.job.walk() if n.id == loop.node_id)
        ref = resource_ref(source.params.get("channels"))
        return bool(ref and getattr(self.job.resources[ref], "keep_light_on_between_channels",
                                    False))

    def _do_capture(self, node, p):
        loop = self._innermost("loop_channels")
        if p.channels is not None:
            channels = list(enumerate(p.channels))
            names = [c.config for c in p.channels]
        else:
            channels = [(loop.index, loop.item)]
            names = [c.config for c in loop.items]
        keep = self._keep_light(node)
        stats = None
        try:
            for c_index, channel in channels:
                frame = self._acquire(channel, keep_light=keep)
                self.last_frames[node.id] = frame
                stats = {"mean": float(frame.mean()), "max": float(frame.max()),
                         "shape": list(frame.shape)}
                if p.save and self.sink is not None:
                    self.sink.write(frame, self._frame_info(node.id, c_index, names, channel))
                    self.frames += 1
        finally:
            if keep and p.channels is not None:
                self._dark()   # kept on between this capture's channels only
        return {"frames": len(channels), **(stats or {})}

    def _light(self, channel: CaptureChannel) -> None:
        """Switch to this channel's light; one light at a time, no repeat commands."""
        key = (channel.kind, channel.config, channel.intensity, channel.led_pattern)
        if self._lit is not None and self._lit[0] == key:
            return
        self._dark()
        if channel.kind == "default":
            self.hw.set_illumination(channel.config, intensity=channel.intensity, on=True)
        else:
            self.hw.set_led_pattern(channel.kind, **channel.led_pattern.model_dump())
        self._lit = (key, channel)

    def _dark(self) -> None:
        if self._lit is None:
            return
        channel, self._lit = self._lit[1], None
        if channel.kind == "default":
            self.hw.set_illumination(channel.config, on=False)
        else:
            self.hw.set_led_pattern("off")

    def _acquire(self, channel: CaptureChannel, keep_light: bool = False) -> np.ndarray:
        """Light on, camera set, Z offset, snap, back; light off unless kept."""
        self._checkpoint()
        self._light(channel)
        try:
            self.hw.set_camera(exposure_ms=channel.exposure, gain=channel.gain,
                               detector=channel.camera)
            if channel.z_offset:
                self.hw.move({"Z": channel.z_offset}, relative=True)
            try:
                return np.asarray(self.hw.snap(channel.camera))
            finally:
                if channel.z_offset:
                    self.hw.move({"Z": -channel.z_offset}, relative=True)
        finally:
            if not keep_light:
                self._dark()

    # -- focus maps -----------------------------------------------------------
    def _surface_z(self, name: str, x: float, y: float, group: Optional[str] = None) -> float:
        surfaces = self.surfaces.get(name)
        if not surfaces:
            raise RuntimeError(f"focus map {name!r} was not measured (run create_focus_map first)")
        if group is None:
            current = [lp.item.get("name") for lp in reversed(self.loops)
                       if lp.task in POSITION_LOOPS]
            group = next((g for g in current + ["default"] if g in surfaces),
                         next(iter(surfaces)) if len(surfaces) == 1 else None)
        if group not in surfaces:
            raise RuntimeError(f"focus map {name!r} has no surface for {group!r}")
        return surfaces[group].interpolate(x, y)

    def _focus_targets(self, p) -> Dict[str, list]:
        """Measurement points per surface group: (x, y, z or None)."""
        def grid(r: dict) -> list:
            bounds = {"minX": r["min_x"], "maxX": r["max_x"], "minY": r["min_y"],
                      "maxY": r["max_y"]}
            return [(x, y, None) for x, y in FocusMap.generate_grid(
                bounds, rows=p.ny, cols=p.nx, add_margin=p.margin > 0, margin_fraction=p.margin)]

        if p.points:
            return {"default": [(pt.x, pt.y, pt.z) for pt in p.points]}
        if p.regions:
            return {r.name: grid(r.model_dump()) for r in p.regions}
        if p.plate:
            return {"default": [(w.x, w.y, None) for w in p.plate.selected_well_positions]}
        loop = self._innermost(*POSITION_LOOPS)
        if loop is None:
            raise ValueError("create_focus_map needs points, regions, a plate or an enclosing "
                             "position/well/region loop")
        item = loop.item
        return {item["name"]: grid(item) if "min_x" in item else [(item["x"], item["y"],
                                                                   item.get("z"))]}

    def _do_create_focus_map(self, node, p):
        name = resource_ref(node.params["focus_map"])
        af = {k: v for k, v in p.autofocus.model_dump().items()
              if k in self.tasks["autofocus"].params_model.model_fields}
        surfaces = self.surfaces.setdefault(name, {})
        result = {"groups": {}, "measured": 0, "failed": 0}
        for group, points in self._focus_targets(p).items():
            fmap = FocusMap(group_id=group, method=p.focus_map.method,
                            z_offset=p.focus_map.z_offset_um)
            for x, y, z in points:
                self._checkpoint()
                self.hw.move({"X": x, "Y": y, **({"Z": z} if z is not None else {})})
                zf = self.tasks["autofocus"].function(**af, hw=self.hw, ctx=self._ctx)
                if zf is None:
                    result["failed"] += 1
                else:
                    fmap.add_point(x, y, zf)
                    result["measured"] += 1
            if fmap.n_points < p.min_points:
                raise RuntimeError(f"{group}: {fmap.n_points} of {len(points)} focus "
                                   f"measurements succeeded, {p.min_points} needed")
            stats = fmap.fit()
            surfaces[group] = fmap
            result["groups"][group] = {
                "points": fmap.n_points, "method": stats.method, "fallback": stats.fallback_reason,
                "error_um": stats.mean_abs_error if stats.error_is_cross_validated else None}
        return result

    def _do_move_to_focus_surface(self, node, p):
        pos = self.hw.get_position()
        z = self._surface_z(resource_ref(node.params["focus_map"]), pos["X"], pos["Y"], p.group)
        self.hw.move({"Z": z})
        return z

    def _loop_coordinates(self):
        """useq index, axis sizes and position/tile/z/t details of the active loops."""
        index, sizes, info = {}, {}, {}
        for lp in self.loops:
            if lp.axis:
                index[lp.axis], sizes[lp.axis] = lp.index, lp.count
            if lp.axis == "p":
                info["position"] = lp.item
            elif lp.axis == "g":
                info["tile"] = lp.item
            elif lp.axis == "z":
                info["z_um"], info["z_step_um"] = lp.item["z"], lp.item["step"]
            elif lp.axis == "t":
                info["t_step_s"] = lp.item["interval_s"]
        repeat = self._innermost("repeat")
        if "t" not in index and repeat is not None:   # repeats count as time points
            index["t"], sizes["t"] = repeat.index, repeat.count
        return index, sizes, info

    def _frame_info(self, node_id: str, c_index: int, channel_names: List[str],
                    channel: Optional[CaptureChannel] = None) -> FrameInfo:
        index, sizes, info = self._loop_coordinates()
        index["c"], sizes["c"] = c_index, len(channel_names)
        if "z_um" not in info or ("position" not in info and "tile" not in info):
            pos = self.hw.get_position()          # where the frame really was taken
            info.setdefault("z_um", pos.get("Z"))
            if "position" not in info and "tile" not in info:
                info["stage_xy"] = (pos.get("X", 0.0), pos.get("Y", 0.0))
        if self._pixel_size is None:
            self._pixel_size = self.hw.pixel_size_um()
            self._bit_depth = self.hw.bit_depth()
        return FrameInfo(node_id=node_id, index=index, sizes=sizes, channel_names=channel_names,
                         pixel_size_um=self._pixel_size, bit_depth=self._bit_depth,
                         exposure_ms=channel.exposure if channel is not None else None, **info)

    def _do_mda(self, node, p):
        """Run a useq MDASequence event by event (the useq import path)."""
        seq, start = p.sequence, time.monotonic()
        sizes = dict(seq.sizes)
        names = [c.config for c in seq.channels] or ["default"]
        for event in seq:
            self._checkpoint()
            if event.min_start_time:
                self._wait(start + event.min_start_time - time.monotonic())
            target = {k: v for k, v in (("X", event.x_pos), ("Y", event.y_pos),
                                        ("Z", event.z_pos)) if v is not None}
            if target:
                self.hw.move(target)
            action = getattr(event.action, "type", "acquire_image")
            if action == "hardware_autofocus":
                self.hw.autofocus(mode="hardware", range_um=0, step_um=0, algorithm="",
                                  settle_s=0, max_attempts=1)
                continue
            if action != "acquire_image":
                self._log(f"{node.id}: useq action {action!r} is not handled")
                continue
            channel = CaptureChannel(config=event.channel.config if event.channel else names[0],
                                     exposure=event.exposure)
            frame = self._acquire(channel)
            if self.sink is not None:
                index = dict(event.index)
                xy = {"x": event.x_pos or 0.0, "y": event.y_pos or 0.0}
                tile = {"name": f"{index['g']:04d}", **xy} if "g" in index else None
                self.sink.write(frame, FrameInfo(
                    node_id=node.id, index=index, sizes=sizes, channel_names=names,
                    pixel_size_um=self.hw.pixel_size_um(), tile=tile, exposure_ms=event.exposure,
                    z_um=event.z_pos, z_step_um=_z_step(seq),
                    position={"name": event.pos_name or f"p{index.get('p', 0):03d}",
                              "z": event.z_pos, **xy}))
                self.frames += 1


def _z_step(seq: useq.MDASequence) -> Optional[float]:
    zs = list(seq.z_plan) if seq.z_plan is not None else []
    return abs(zs[1] - zs[0]) if len(zs) > 1 else None


def _plain(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        return repr(value)[:200]
