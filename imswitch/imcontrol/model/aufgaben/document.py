"""The job document: what an Aufgaben job is, as data.

A job is a tree of nodes that runs top to bottom. Container nodes (loops,
conditions, blocks) hold child nodes in ``body``. Resources are declared once at
the top and referenced by name with an ``"@name"`` string. ``"${...}"`` strings
are expressions evaluated at run time.

Acquisition geometry reuses useq-schema models unchanged: positions, well-plate
plans, grid plans, z plans, time plans and channels are the same objects that
pymmcore-plus and other useq consumers read. A job without control flow,
variables or peripheral tasks can be exported to a plain useq ``MDASequence``
(see ``useq_bridge``), and any ``MDASequence`` imports as a job.

This module depends on pydantic and useq only, so the document format can move
to another host application unchanged.
"""
from __future__ import annotations

from typing import Annotated, Any, Dict, Iterator, List, Literal, Optional, Tuple, Union

import useq
from pydantic import BaseModel, ConfigDict, Field, model_validator

SCHEMA_ID = "aufgaben.job/1"
NODE_ID_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"
NAME_PATTERN = r"^[A-Za-z_][A-Za-z0-9_]*$"

Scalar = Union[bool, int, float, str]


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)


# --------------------------------------------------------------------------- #
# Job parameters (asked for in a wizard or before a run) and variables
# --------------------------------------------------------------------------- #
class Parameter(_Model):
    """A value the operator sets before a run; referenced as ``param.<name>``."""

    type: Literal["integer", "number", "boolean", "string", "choice"]
    default: Optional[Scalar] = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    choices: List[Scalar] = Field(default_factory=list)
    unit: Optional[str] = None
    title: Optional[str] = None
    description: str = ""
    ui: Literal["wizard", "runtime", "both", "hidden"] = "runtime"

    @model_validator(mode="after")
    def _choices_for_choice(self) -> "Parameter":
        if self.type == "choice" and not self.choices:
            raise ValueError("a 'choice' parameter needs choices")
        return self


# --------------------------------------------------------------------------- #
# Resources
# --------------------------------------------------------------------------- #
GridPlan = Union[
    useq.GridRowsColumns,
    useq.GridWidthHeight,
    useq.GridFromEdges,
    useq.GridFromPolygon,
    useq.RandomPoints,
]


class CaptureChannel(useq.Channel):
    """A useq ``Channel`` plus what ImSwitch needs to switch the light.

    ``config`` names the illumination source as configured in the setup (a
    laser, LED or LED-matrix pattern). ``exposure`` is in ms, ``z_offset`` in
    µm, as in useq. The extra fields are dropped on export to useq.
    """

    intensity: Optional[float] = Field(
        None, description="Source intensity in the source's own units; None keeps it")
    gain: Optional[float] = Field(None, description="Camera gain; None keeps it")
    kind: Literal["default", "ring", "dpc"] = "default"
    led_pattern: Optional[Dict[str, Any]] = Field(
        None, description="LED-matrix pattern parameters for kind ring/dpc")


class PositionsResource(_Model):
    """Named stage positions in µm (user frame). Tasks may add or remove points."""

    kind: Literal["positions"] = "positions"
    positions: List[useq.AbsolutePosition] = Field(default_factory=list)
    labels: Dict[str, str] = Field(default_factory=dict, description="position name -> label")


class WellPlateResource(_Model):
    """A plate, its alignment and the selected wells (useq ``WellPlatePlan``)."""

    kind: Literal["well_plate"] = "well_plate"
    plan: useq.WellPlatePlan
    labels: Dict[str, str] = Field(default_factory=dict, description="well name -> label")


class GridResource(_Model):
    """A tile pattern relative to a position (useq grid plans, µm)."""

    kind: Literal["grid"] = "grid"
    plan: GridPlan


class ZPlanResource(_Model):
    kind: Literal["z_plan"] = "z_plan"
    plan: useq.AnyZPlan


class TimePlanResource(_Model):
    kind: Literal["time_plan"] = "time_plan"
    plan: useq.AnyTimePlan


class CaptureResource(_Model):
    """The channels one capture takes, in order (JOBS: capture definition)."""

    kind: Literal["capture"] = "capture"
    channels: List[CaptureChannel] = Field(..., min_length=1)
    keep_light_on_between_channels: bool = False


class AutofocusResource(_Model):
    """How to focus. ``range_um`` is the half-range around the current Z."""

    kind: Literal["autofocus"] = "autofocus"
    mode: Literal["software", "hardware"] = "software"
    method: Literal["scan", "hill_climbing"] = "scan"
    range_um: float = Field(50.0, gt=0)
    step_um: float = Field(5.0, gt=0)
    channel: Optional[str] = Field(None, description="Illumination source used while focusing")
    intensity: Optional[float] = Field(None, description="Its intensity; None keeps the value")
    algorithm: str = "LAPE"
    settle_s: float = Field(0.1, ge=0)
    max_attempts: int = Field(2, ge=1)


class Region(_Model):
    name: str
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    vertices: Optional[List[Tuple[float, float]]] = None
    label: Optional[str] = None


class RegionsResource(_Model):
    """Areas in stage µm, drawn by a user or produced by an analysis task."""

    kind: Literal["regions"] = "regions"
    regions: List[Region] = Field(default_factory=list)


class FocusMapResource(_Model):
    """A focus surface, filled at run time by a create-focus-map task."""

    kind: Literal["focus_map"] = "focus_map"
    method: Literal["spline", "rbf", "constant"] = "spline"
    z_offset_um: float = 0.0


Resource = Annotated[
    Union[
        PositionsResource,
        WellPlateResource,
        GridResource,
        ZPlanResource,
        TimePlanResource,
        CaptureResource,
        AutofocusResource,
        RegionsResource,
        FocusMapResource,
    ],
    Field(discriminator="kind"),
]


# --------------------------------------------------------------------------- #
# Storage
# --------------------------------------------------------------------------- #
class Storage(_Model):
    """Where and how frames are written. Both formats are plain OME."""

    formats: List[Literal["ome-zarr", "ome-tiff"]] = Field(default_factory=lambda: ["ome-zarr"])
    layout: Literal["auto", "positions", "plate"] = Field(
        "auto",
        description="positions: one OME image per stage position; plate: an OME-NGFF "
                    "plate (wells/fields). auto: plate when a well loop is used.")
    folder: Optional[str] = Field(
        None, description="Relative to the data folder; default by job name")
    stitched_preview: bool = Field(False, description="Also write a stitched mosaic per region")


# --------------------------------------------------------------------------- #
# Nodes
# --------------------------------------------------------------------------- #
class Every(_Model):
    """Run a node only on every n-th iteration of a loop (JOBS: Every nth)."""

    nth: int = Field(..., ge=1)
    of: Optional[str] = Field(None, description="Loop node id; default the innermost loop")
    include_first: bool = True


class Retry(_Model):
    retry: int = Field(..., ge=1)
    then: Literal["abort", "skip", "continue_loop"] = "abort"


OnError = Union[Literal["abort", "skip", "continue_loop"], Retry]


class Node(_Model):
    """One step of a job.

    ``task`` names a task type from the catalogue. ``params`` are validated
    against that task's parameter model. Containers use ``body``; a condition
    also uses ``orelse``; a time lapse may use ``idle`` for work done while it
    waits for the next round.
    """

    id: str = Field(..., pattern=NODE_ID_PATTERN)
    task: str
    params: Dict[str, Any] = Field(default_factory=dict)
    body: List["Node"] = Field(default_factory=list)
    orelse: List["Node"] = Field(default_factory=list)
    idle: List["Node"] = Field(default_factory=list)
    enabled: Union[bool, str] = True
    every: Optional[Every] = None
    on_error: OnError = "abort"
    label: Optional[str] = None


Node.model_rebuild()


class Job(_Model):
    """An Aufgaben job document."""

    schema_: Literal["aufgaben.job/1"] = Field(SCHEMA_ID, alias="schema")
    name: str = Field(..., min_length=1)
    description: str = ""
    parameters: Dict[str, Parameter] = Field(default_factory=dict)
    variables: Dict[str, Scalar] = Field(default_factory=dict)
    resources: Dict[str, Resource] = Field(default_factory=dict)
    storage: Storage = Field(default_factory=Storage)
    tasks: List[Node] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _names_are_identifiers(self) -> "Job":
        import re
        for group, names in (("parameter", self.parameters), ("variable", self.variables),
                             ("resource", self.resources)):
            for name in names:
                if not re.match(NAME_PATTERN, name):
                    raise ValueError(f"{group} name {name!r} is not an identifier")
        return self

    def walk(self) -> Iterator[Tuple[Node, Tuple[Node, ...]]]:
        """Yield every node with its chain of ancestors, depth first."""

        def _walk(nodes, ancestors):
            for node in nodes:
                yield node, ancestors
                for branch in (node.body, node.orelse, node.idle):
                    yield from _walk(branch, ancestors + (node,))

        yield from _walk(self.tasks, ())

    def to_json(self, **kw) -> str:
        return self.model_dump_json(by_alias=True, exclude_defaults=False, **kw)


def resource_ref(value: Any) -> Optional[str]:
    """Return the resource name if ``value`` is an ``"@name"`` reference."""
    if isinstance(value, str) and value.startswith("@") and not value.startswith("@@"):
        return value[1:]
    return None


def expression(value: Any) -> Optional[str]:
    """Return the expression text if ``value`` is a ``"${...}"`` string."""
    if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
        return value[2:-1].strip()
    return None
