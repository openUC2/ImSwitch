"""The task registry: plain functions become catalogue entries.

A task is an ordinary function. The decorator records metadata and returns the
function unchanged, so the same function can be registered as an Arkitekt
action (``app.action(...)(fn)``) or called directly. The convention, chosen to
be readable by Arkitekt 5 and 6 without changes:

- typed parameters: primitives, ``Optional``, ``Literal``/``Enum`` for fixed
  choices, ``Annotated[x, annotated_types.Gt(..)/Le(..)]`` for limits;
- units in the parameter name suffix (``_um``, ``_ms``, ``_s``, ``_ul`` ...);
- descriptions only in a Google-style ``Args:`` docstring (Arkitekt rejects a
  second description through ``Annotated``);
- device-dependent choices as ``str`` plus ``choices={param: provider}`` here,
  resolved by the host when it builds the catalogue;
- injected arguments are keyword-only and typed ``Microscope`` (the host
  hardware) or ``TaskContext`` (progress, log, pause, cancel). ``TaskContext``
  carries Arkitekt's task marker, so Arkitekt injects its own Task there.

``catalogue()`` returns the JSON the frontend palette and parameter forms are
built from.
"""
from __future__ import annotations

import inspect
import re
import typing
from dataclasses import dataclass
from typing import (Any, Callable, ClassVar, Dict, Iterable, List, Literal, Mapping, Optional,
                    Protocol, Tuple, Type)

from pydantic import BaseModel, ConfigDict, Field, create_model

from .hardware import DerivedImage, Frame, Microscope

Effects = Literal["NONE", "REPEATABLE", "UNKNOWN", "IRREVERSIBLE"]

UNIT_SUFFIXES = (  # longest first
    ("_ul_min", "µL/min"), ("_um_s", "µm/s"), ("_percent", "%"), ("_um2", "µm²"), ("_um", "µm"),
    ("_px", "px"),
    ("_nm", "nm"), ("_ms", "ms"), ("_ul", "µL"), ("_hz", "Hz"), ("_c", "°C"), ("_s", "s"),
)


class TaskContext(Protocol):
    """Progress, logging, pause and cancel for a running task.

    The engine passes its own implementation. Arkitekt recognises the class
    attribute below (arkitekt_spec TASK_MARKER) and passes its Task instead.
    """

    __arkitekt_task__: ClassVar[bool] = True

    def progress(self, percentage: int, message: Optional[str] = None) -> None: ...
    def log(self, message: str) -> None: ...
    def pausepoint(self) -> None: ...
    def check_cancelled(self) -> None: ...

    # Only the Aufgaben engine provides these (analysis and script steps):
    def frame(self, source: Optional[str] = None) -> "Frame":
        """An image the run has seen: the last one of step ``source``, else the newest."""
    variables: Dict[str, Any]          # the job's variables, writable
    results: Dict[str, Any]            # what each step returned so far (a copy)


INJECTED_TYPES: Tuple[type, ...] = (Microscope, TaskContext)

# Steps of these categories read images the run has seen (ctx.frame) and return
# mixed plain data, so only the Aufgaben engine runs them, not Arkitekt.
FRAME_STEPS = ("Analysis", "Scripts")

# Arkitekt uses "value" for a port's own value inside validators and effects.
RESERVED_PARAM_NAMES = frozenset({"value"})

# Types tasks return that an Arkitekt app declares before registering the tasks:
# ``app.register_memory_structure(cls, identifier=...)`` (kept by the agent).
MEMORY_STRUCTURES: Dict[type, str] = {DerivedImage: "aufgaben.derived_image"}


@dataclass(frozen=True)
class TaskFlags:
    """What a task does to the instrument; drives confirmations and locks."""

    moves_xy: bool = False
    moves_z: bool = False
    homes: bool = False
    emits_light: bool = False
    external: bool = False      # talks to something outside the microscope
    runs_code: bool = False
    confirm: bool = False        # ask before a run, e.g. objective change
    needs: Tuple[str, ...] = ()  # Microscope.capabilities() it requires

    @property
    def ask_first(self) -> bool:
        """Tasks a run must confirm before it starts (AGENTS.md hard rules)."""
        return self.homes or self.runs_code or self.confirm


@dataclass(frozen=True)
class TaskSpec:
    id: str
    title: str
    category: str
    description: str
    function: Callable[..., Any]
    params_model: Type[BaseModel]
    param_docs: Dict[str, str]
    units: Dict[str, str]
    injected: Dict[str, str]  # parameter name -> "hw" | "ctx"
    choices: Dict[str, str]
    locks: Tuple[str, ...]
    effects: Effects
    flags: TaskFlags

    def schema(self, providers: Optional[Mapping[str, Callable[[], Iterable[str]]]] = None) -> dict:
        """JSON schema of the parameters, with units and resolved choices."""
        schema = self.params_model.model_json_schema()
        props = schema.get("properties", {})
        for name, unit in self.units.items():
            if name in props:
                props[name]["x-unit"] = unit
        for name, provider in self.choices.items():
            if name in props:
                props[name]["x-choices-from"] = provider
                if providers and provider in providers:
                    props[name]["enum"] = list(providers[provider]())
        return schema

    def entry(self, providers=None) -> dict:
        return {
            "id": self.id, "title": self.title, "category": self.category,
            "description": self.description, "kind": "task",
            "schema": self.schema(providers), "locks": list(self.locks),
            "effects": self.effects,
            "flags": {**{k: getattr(self.flags, k) for k in
                         ("moves_xy", "moves_z", "homes", "emits_light", "external", "runs_code")},
                      "needs": list(self.flags.needs), "ask_first": self.flags.ask_first},
        }


REGISTRY: Dict[str, TaskSpec] = {}


def parse_docstring(doc: Optional[str]) -> Tuple[str, str, Dict[str, str]]:
    """(title, description, {param: text}) from a Google-style docstring."""
    if not doc:
        return "", "", {}
    lines = inspect.cleandoc(doc).splitlines()
    title = lines[0].strip() if lines else ""
    body, args, current, section, arg_indent = [], {}, None, "body", None
    for line in lines[1:]:
        stripped = line.strip()
        if re.fullmatch(r"(Args|Arguments|Parameters):", stripped):
            section, arg_indent = "args", None
            continue
        if re.fullmatch(r"(Returns|Yields|Raises|Examples?|Notes?):", stripped):
            section = "other"
            continue
        if section == "body":
            body.append(line)
        elif section == "args" and stripped:
            indent = len(line) - len(line.lstrip())
            m = re.match(r"^(\w+)(?:\s*\([^)]*\))?:\s*(.*)$", stripped)
            if arg_indent is None and m:
                arg_indent = indent
            if m and indent == arg_indent:
                current = m.group(1)
                args[current] = m.group(2).strip()
            elif current:
                args[current] = (args[current] + " " + stripped).strip()
    return title, "\n".join(body).strip(), args


def _unit(name: str) -> Optional[str]:
    for suffix, unit in UNIT_SUFFIXES:
        if name.endswith(suffix):
            return unit
    return None


def _injected_kind(annotation: Any) -> Optional[str]:
    """'ctx' for a TaskContext (or any Arkitekt task type), 'hw' for the Microscope."""
    origin = typing.get_origin(annotation)
    if origin is typing.Annotated:
        annotation = typing.get_args(annotation)[0]
    if origin is typing.Union:
        args = [a for a in typing.get_args(annotation) if a is not type(None)]
        annotation = args[0] if len(args) == 1 else annotation
    if not isinstance(annotation, type):
        return None
    if getattr(annotation, "__arkitekt_task__", False) is True:
        return "ctx"
    return "hw" if annotation in INJECTED_TYPES else None


def build_spec(fn: Callable[..., Any], *, id: Optional[str] = None, title: Optional[str] = None,
               category: str = "General", locks: Iterable[str] = (), effects: Effects = "UNKNOWN",
               choices: Optional[Mapping[str, str]] = None,
               flags: Optional[TaskFlags] = None) -> TaskSpec:
    doc_title, description, param_docs = parse_docstring(fn.__doc__)
    hints = typing.get_type_hints(fn, include_extras=True)
    fields: Dict[str, Any] = {}
    injected: Dict[str, str] = {}
    units: Dict[str, str] = {}
    for name, param in inspect.signature(fn).parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            raise TypeError(f"task {fn.__name__}: *args/**kwargs are not allowed")
        annotation = hints.get(name, Any)
        kind = _injected_kind(annotation)
        if kind:
            if param.kind is not param.KEYWORD_ONLY:
                raise TypeError(
                    f"task {fn.__name__}: injected argument {name!r} must be keyword-only")
            injected[name] = kind
            continue
        if name in RESERVED_PARAM_NAMES:
            raise TypeError(f"task {fn.__name__}: parameter name {name!r} is reserved")
        if name not in param_docs:
            raise TypeError(f"task {fn.__name__}: parameter {name!r} has no Args: description")
        default = ... if param.default is param.empty else param.default
        fields[name] = (annotation, Field(default, description=param_docs[name]))
        if (unit := _unit(name)):
            units[name] = unit
    for name in (choices or {}):
        if name not in fields:
            raise TypeError(f"task {fn.__name__}: choices for unknown parameter {name!r}")
    task_id = id or fn.__name__
    model = create_model(f"{task_id}_params", __config__=ConfigDict(extra="forbid"), **fields)
    return TaskSpec(
        id=task_id, title=title or doc_title or task_id.replace("_", " ").capitalize(),
        category=category, description=description, function=fn, params_model=model,
        param_docs={k: v for k, v in param_docs.items() if k in fields}, units=units,
        injected=injected, choices=dict(choices or {}), locks=tuple(locks),
        effects=effects, flags=flags or TaskFlags(),
    )


def task(**options: Any) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as a task. Returns the function unchanged.

    Options: id, title, category, locks, effects, choices, flags (see build_spec).
    """
    def mark(fn: Callable[..., Any]) -> Callable[..., Any]:
        spec = build_spec(fn, **options)
        if spec.id in REGISTRY and REGISTRY[spec.id].function is not fn:
            raise ValueError(f"task id {spec.id!r} is already registered")
        REGISTRY[spec.id] = spec
        fn.__aufgaben_task__ = spec
        return fn
    return mark


def catalogue(capabilities: Optional[Iterable[str]] = None,
              providers: Optional[Mapping[str, Callable[[], Iterable[str]]]] = None) -> List[dict]:
    """Palette entries for the control nodes and every registered task.

    A task whose ``needs`` the host lacks is listed with ``available: false`` and
    ``missing`` (the capabilities it lacks), so an editor can still show the
    step of a job written for other hardware; the validator refuses to run it.
    """
    from .control import CONTROL_SPECS
    caps = set(capabilities) if capabilities is not None else None
    entries = []
    for spec in CONTROL_SPECS.values():
        entries.append({
            "id": spec.id, "title": spec.title, "category": spec.category, "kind": "control",
            "description": spec.description, "branches": list(spec.branches), "loop": spec.loop,
            "axis": spec.axis, "ref_kinds": {k: list(v) for k, v in spec.ref_kinds.items()},
            "expr_fields": list(spec.expr_fields), "schema": spec.params.model_json_schema(),
        })
    for spec in REGISTRY.values():
        missing = sorted(set(spec.flags.needs) - caps) if caps is not None else []
        entries.append({**spec.entry(providers), "available": not missing, "missing": missing})
    return entries
