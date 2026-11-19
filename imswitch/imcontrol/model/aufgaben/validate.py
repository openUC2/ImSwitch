"""Check a job before it runs: structure, references, parameters, safety.

Errors make a job unrunnable (the editor marks the node). Warnings do not.
``ask_first`` lists nodes a run has to confirm before it starts (homing,
objective change, code), following the hard rules in AGENTS.md.
"""
from __future__ import annotations

import typing
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple

from pydantic import TypeAdapter, ValidationError

from . import expressions
from .control import CONTROL_SPECS, MOVING_LOOPS, POSITION_LOOPS, ControlSpec
from .document import Job, Node, expression, resource_ref
from .registry import FRAME_STEPS, REGISTRY, TaskSpec

FROM_KEY = "$from"  # node param that pulls defaults from a resource


@dataclass
class Issue:
    level: str          # "error" | "warning"
    node: Optional[str]
    message: str


@dataclass
class Report:
    issues: List[Issue] = field(default_factory=list)
    ask_first: List[str] = field(default_factory=list)
    moves_xy: bool = False
    moves_z: bool = False
    emits_light: bool = False

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.level == "error"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def as_dict(self) -> dict:
        return {"ok": self.ok, "issues": [i.__dict__ for i in self.issues],
                "ask_first": self.ask_first, "moves_xy": self.moves_xy,
                "moves_z": self.moves_z, "emits_light": self.emits_light}


def _field_types(model) -> Dict[str, Tuple[Any, bool]]:
    hints = typing.get_type_hints(model, include_extras=True)
    return {name: (hints.get(name, Any), f.is_required()) for name, f in model.model_fields.items()}


def _is_loop(node: Node) -> bool:
    spec = CONTROL_SPECS.get(node.task)
    return spec is not None and spec.loop


class _Validator:
    def __init__(self, job: Job, tasks: Mapping[str, TaskSpec], caps: Optional[Set[str]]):
        self.job, self.tasks, self.caps = job, tasks, caps
        self.report = Report()
        self.ids: Set[str] = set()

    def err(self, node_id: Optional[str], message: str) -> None:
        self.report.issues.append(Issue("error", node_id, message))

    def warn(self, node_id: Optional[str], message: str) -> None:
        self.report.issues.append(Issue("warning", node_id, message))

    # -- whole job ----------------------------------------------------------
    def run(self) -> Report:
        for node, _ in self.job.walk():
            if node.id in self.ids:
                self.err(node.id, f"duplicate node id {node.id!r}")
            self.ids.add(node.id)
        for node, ancestors in self.job.walk():
            self.check_node(node, ancestors)
        return self.report

    def check_node(self, node: Node, ancestors: Tuple[Node, ...]) -> None:
        spec = CONTROL_SPECS.get(node.task)
        tspec = self.tasks.get(node.task)
        if spec is None and tspec is None:
            self.err(node.id, f"unknown task {node.task!r}")
            return
        self.check_branches(node, spec)
        self.check_enabled_and_every(node, ancestors)
        self.check_params(node, spec, tspec, ancestors)
        self.check_rules(node, ancestors)
        self.summarise(node, tspec)

    # -- structure ----------------------------------------------------------
    def check_branches(self, node: Node, spec: Optional[ControlSpec]) -> None:
        allowed = spec.branches if spec else ()
        for branch in ("body", "orelse", "idle"):
            if getattr(node, branch) and branch not in allowed:
                self.err(node.id, f"{node.task} cannot have {branch!r}")
        if spec and "body" in allowed and not node.body:
            self.warn(node.id, f"{node.task} has nothing in its body")

    def check_enabled_and_every(self, node: Node, ancestors: Tuple[Node, ...]) -> None:
        if isinstance(node.enabled, str):
            self.check_expr(node.id, node.enabled, ancestors)
        if node.every is None:
            return
        loops = {a.id for a in ancestors if _is_loop(a)}
        if not loops:
            self.err(node.id, "'every' needs an enclosing loop")
        elif node.every.of and node.every.of not in loops:
            self.err(node.id, f"every.of {node.every.of!r} is not an enclosing loop")

    # -- expressions --------------------------------------------------------
    def check_expr(self, node_id: Optional[str], text: str, scope: Tuple[Node, ...]) -> None:
        try:
            refs = expressions.references(text)
        except expressions.ExpressionError as e:
            self.err(node_id, str(e))
            return
        missing = (
            [f"unknown parameter param.{n}" for n in refs["param"] - set(self.job.parameters)]
            + [f"unknown variable {n!r}" for n in refs["var"] - set(self.job.variables)]
            + [f"result.{n}: no node with that id" for n in refs["result"] - self.ids]
            + [f"loop.{n}: not an enclosing loop" for n in refs["loop"] - {a.id for a in scope}])
        for message in missing:
            self.err(node_id, message)

    # -- parameters ---------------------------------------------------------
    def merged_params(self, node: Node, model) -> Dict[str, Any]:
        params = dict(node.params)
        if FROM_KEY not in params:
            return params
        src = resource_ref(params.pop(FROM_KEY))
        res = self.job.resources.get(src) if src else None
        if res is None:
            self.err(node.id, f"{FROM_KEY} must name a resource")
            return params
        base = {k: v for k, v in res.model_dump().items()
                if k != "kind" and k in model.model_fields}
        return {**base, **params}

    def check_params(self, node: Node, spec: Optional[ControlSpec], tspec: Optional[TaskSpec],
                     ancestors: Tuple[Node, ...]) -> None:
        model = spec.params if spec else tspec.params_model
        # a loop's own parameters (e.g. time_lapse.until) may read loop.<its id>
        scope = ancestors + (node,) if spec is not None and spec.loop else ancestors
        params = self.merged_params(node, model)
        types = _field_types(model)
        for name in params:
            if name not in types:
                self.err(node.id, f"{node.task} has no parameter {name!r}")
        for name, (annotation, required) in types.items():
            if name in params:
                self.check_value(node, spec, name, annotation, params[name], scope)
            elif required:
                self.err(node.id, f"{node.task} needs {name!r}")
        literal = not any(isinstance(v, str) and (resource_ref(v) or expression(v))
                          for v in params.values())
        if spec and literal:
            try:
                model.model_validate(params)
            except ValidationError as e:
                for problem in e.errors():
                    if not problem["loc"]:
                        self.err(node.id, problem["msg"])

    def check_value(self, node: Node, spec: Optional[ControlSpec], name: str, annotation: Any,
                    value: Any, scope: Tuple[Node, ...]) -> None:
        if spec and name in spec.expr_fields:
            if value is not None:
                self.check_expr(node.id, value, scope)
            return
        ref = resource_ref(value)
        if ref is not None:
            self.check_ref(node, spec, name, ref)
            return
        text = expression(value)
        if text is not None:
            self.check_expr(node.id, text, scope)
            return
        try:
            TypeAdapter(annotation).validate_python(value)
        except ValidationError as e:
            self.err(node.id, f"{name}: {e.errors()[0]['msg']}")

    def check_ref(self, node: Node, spec: Optional[ControlSpec], name: str, ref: str) -> None:
        kinds = spec.ref_kinds.get(name) if spec else None
        res = self.job.resources.get(ref)
        if res is None:
            self.err(node.id, f"{name}: no resource {ref!r}")
        elif not kinds:
            self.err(node.id, f"{name} does not take a resource")
        elif res.kind not in kinds:
            self.err(node.id, f"{name}: '@{ref}' is a {res.kind}, expected {' or '.join(kinds)}")

    # -- node-specific rules --------------------------------------------------
    def check_rules(self, node: Node, ancestors: Tuple[Node, ...]) -> None:
        loops = {a.id for a in ancestors if _is_loop(a)}
        p = node.params
        if node.task in ("break", "continue") and not loops:
            self.err(node.id, f"{node.task} outside a loop")
        if node.task == "break" and p.get("loop") and p["loop"] not in loops:
            self.err(node.id, f"break: {p['loop']!r} is not an enclosing loop")
        if node.task == "capture" and p.get("channels") is None and \
                not any(a.task == "loop_channels" for a in ancestors):
            self.err(node.id, "capture needs channels outside a loop_channels")
        if node.task == "move_to_item":
            targets = {a.id for a in ancestors if a.task in MOVING_LOOPS}
            if not targets or (p.get("loop") and p["loop"] not in targets):
                self.err(node.id, "move_to_item needs an enclosing position/well/tile/Z loop")
        if node.task == "loop_grid" and p.get("plan") is None and \
                not any(a.task == "loop_regions" for a in ancestors):
            self.err(node.id, "loop_grid without a plan needs an enclosing loop_regions")
        if node.task == "create_focus_map" and not any(p.get(k) for k in
                                                       ("points", "regions", "plate")) and \
                not any(a.task in POSITION_LOOPS for a in ancestors):
            self.err(node.id, "create_focus_map needs points, regions, a plate or an enclosing "
                              "position/well/region loop")
        if node.task == "set_variable" and p.get("name") not in self.job.variables:
            self.err(node.id, f"set_variable: {p.get('name')!r} is not a declared variable")
        if node.task == "question" and p.get("store_as") and \
                p["store_as"] not in self.job.variables:
            self.err(node.id, f"question.store_as {p['store_as']!r} is not a declared variable")

    # -- safety summary -----------------------------------------------------
    def summarise(self, node: Node, tspec: Optional[TaskSpec]) -> None:
        r = self.report
        if node.task in MOVING_LOOPS and node.params.get("move", True) is not False:
            r.moves_xy |= node.task != "zstack"
            r.moves_z |= node.task in ("zstack", "loop_positions") or \
                node.params.get("focus_map") is not None
        if node.task in ("capture", "loop_channels", "create_focus_map"):
            r.emits_light = True
        if node.task in ("create_focus_map", "move_to_focus_surface"):
            r.moves_z = True
            r.moves_xy |= node.task == "create_focus_map"
        if tspec is None:
            return
        f = tspec.flags
        r.moves_xy |= f.moves_xy
        r.moves_z |= f.moves_z
        r.emits_light |= f.emits_light
        if f.ask_first:
            r.ask_first.append(node.id)
        if self.caps is not None and not set(f.needs) <= self.caps:
            self.err(node.id, f"{node.task} needs {sorted(set(f.needs) - self.caps)}, "
                              "which this microscope does not have")
        source = node.params.get("source")
        if "source" in tspec.params_model.model_fields and isinstance(source, str) \
                and source and not expression(source) and tspec.category in FRAME_STEPS:
            if source not in self.ids:
                self.err(node.id, f"source {source!r} is not a step of this job")
            elif source == node.id:
                self.err(node.id, "source must be an earlier step, not this one")


def validate(job: Job, tasks: Optional[Mapping[str, TaskSpec]] = None,
             capabilities: Optional[Iterable[str]] = None) -> Report:
    """Check ``job`` against the control nodes and the task catalogue."""
    caps = set(capabilities) if capabilities is not None else None
    return _Validator(job, REGISTRY if tasks is None else tasks, caps).run()
