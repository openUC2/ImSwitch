"""Control nodes: loops, conditions and flow steps the engine itself runs.

These are not hardware tasks. Loops iterate and give each captured frame its
index on a useq axis (``t``, ``p``, ``g``, ``z``, ``c``), so a frame taken inside
``time_lapse > loop_positions > zstack > capture`` carries the same index an
``MDAEvent`` would. Position, well, tile and Z loops move to each item before
running their body (``move: true``). With ``move: false`` the loop only
iterates and a ``move_to_item`` node moves, e.g. inside a condition.

Parameter fields can hold a value, an ``"@resource"`` reference of an allowed
kind, or a ``"${expression}"``. Fields listed in ``expr_fields`` are plain
expression text without ``${}``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Type

import useq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .document import CaptureChannel, GridPlan, Scalar


class _Params(BaseModel):
    model_config = ConfigDict(extra="forbid")


# --------------------------------------------------------------------------- #
# Loops
# --------------------------------------------------------------------------- #
class LoopPositionsParams(_Params):
    source: List[useq.AbsolutePosition] = Field(
        ..., description="Positions to visit, or '@name' of a positions resource")
    move: bool = Field(True, description="Move to each position (XY and its Z, if set)")
    split_storage: bool = Field(False, description="One output per position")


class LoopWellsParams(_Params):
    plate: useq.WellPlatePlan = Field(..., description="'@name' of a well_plate resource")
    label: Optional[str] = Field(None, description="Only wells with this label")
    move: bool = Field(True, description="Move to each well centre")
    split_storage: bool = False


class LoopGridParams(_Params):
    plan: GridPlan = Field(..., description="Tiles relative to the current loop position")
    move: bool = True


class ZStackParams(_Params):
    plan: useq.AnyZPlan = Field(..., description="Z positions; relative plans are relative "
                                                 "to the Z where the loop starts")
    move: bool = True


class LoopChannelsParams(_Params):
    channels: List[CaptureChannel] = Field(..., description="'@name' of a capture resource")


class TimeLapseParams(_Params):
    plan: useq.AnyTimePlan = Field(..., description="Interval and count or duration")
    on_overrun: Literal["run_now", "skip"] = Field(
        "run_now", description="When a round ends after the next start time: start the next "
                               "round at once, or skip the missed start times")
    until: Optional[str] = Field(None, description="Stop early when this expression is true")
    split_storage: bool = False


class RepeatParams(_Params):
    count: Optional[int] = Field(None, ge=1)
    until: Optional[str] = Field(None, description="Stop when this expression is true")

    @model_validator(mode="after")
    def _needs_a_limit(self) -> "RepeatParams":
        if self.count is None and self.until is None:
            raise ValueError("repeat needs count, until, or both")
        return self


# --------------------------------------------------------------------------- #
# Grouping and branching
# --------------------------------------------------------------------------- #
class ConditionParams(_Params):
    expr: str


class BlockParams(_Params):
    collapsed: bool = False


class PhaseParams(_Params):
    name: str


# --------------------------------------------------------------------------- #
# Acquisition nodes the engine expands itself
# --------------------------------------------------------------------------- #
class CaptureParams(_Params):
    channels: Optional[List[CaptureChannel]] = Field(
        None, description="'@name' of a capture resource; omitted inside loop_channels, "
                          "where the loop's current channel is taken")
    save: bool = Field(True, description="False keeps the frames for analysis tasks only")


class MoveToItemParams(_Params):
    loop: Optional[str] = Field(None, description="Loop whose current item to move to; "
                                                  "default the innermost position/well/tile/Z loop")


class MdaParams(_Params):
    """A whole useq MDASequence run as one node (the useq import path)."""

    sequence: useq.MDASequence


# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #
class BreakParams(_Params):
    loop: Optional[str] = Field(None, description="Id of the loop to leave; default the innermost")


class EmptyParams(_Params):
    pass


class MessageParams(_Params):
    message: str = ""


class WaitParams(_Params):
    seconds: float = Field(..., ge=0)


class WaitUntilParams(_Params):
    expr: str
    timeout_s: Optional[float] = Field(None, gt=0)
    poll_s: float = Field(0.5, gt=0)
    on_timeout: Literal["abort", "continue"] = "abort"


class QuestionParams(_Params):
    text: str
    buttons: List[str] = Field(default_factory=lambda: ["Continue"], min_length=1)
    store_as: Optional[str] = Field(None, description="Variable that receives the button text")
    timeout_s: Optional[float] = Field(None, gt=0)


class SetVariableParams(_Params):
    name: str
    value: Scalar


class LogParams(_Params):
    message: str
    level: Literal["info", "warning", "error"] = "info"


# --------------------------------------------------------------------------- #
# The table the validator and the engine read
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ControlSpec:
    id: str
    title: str
    category: str
    params: Type[BaseModel]
    branches: Tuple[str, ...] = ()          # which of body / orelse / idle it may hold
    loop: bool = False                      # break/continue/every can target it
    axis: Optional[str] = None              # useq axis it indexes frames on
    ref_kinds: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    expr_fields: Tuple[str, ...] = ()
    description: str = ""


CONTROL_SPECS: Dict[str, ControlSpec] = {s.id: s for s in [
    ControlSpec("loop_positions", "Loop over positions", "Loops", LoopPositionsParams,
                ("body",), loop=True, axis="p", ref_kinds={"source": ("positions",)}),
    ControlSpec("loop_wells", "Loop over wells", "Loops", LoopWellsParams,
                ("body",), loop=True, axis="p", ref_kinds={"plate": ("well_plate",)}),
    ControlSpec("loop_grid", "Loop over tiles", "Loops", LoopGridParams,
                ("body",), loop=True, axis="g", ref_kinds={"plan": ("grid",)}),
    ControlSpec("zstack", "Z-stack", "Loops", ZStackParams,
                ("body",), loop=True, axis="z", ref_kinds={"plan": ("z_plan",)}),
    ControlSpec("loop_channels", "Loop over channels", "Loops", LoopChannelsParams,
                ("body",), loop=True, axis="c", ref_kinds={"channels": ("capture",)}),
    ControlSpec("time_lapse", "Time lapse", "Loops", TimeLapseParams,
                ("body", "idle"), loop=True, axis="t", ref_kinds={"plan": ("time_plan",)},
                expr_fields=("until",)),
    ControlSpec("repeat", "Repeat", "Loops", RepeatParams,
                ("body",), loop=True, expr_fields=("until",)),
    ControlSpec("condition", "If", "Flow", ConditionParams,
                ("body", "orelse"), expr_fields=("expr",)),
    ControlSpec("block", "Block", "Flow", BlockParams, ("body",)),
    ControlSpec("phase", "Phase", "Flow", PhaseParams, ("body",)),
    ControlSpec("capture", "Capture", "Acquisition", CaptureParams,
                axis="c", ref_kinds={"channels": ("capture",)}),
    ControlSpec("mda", "useq sequence", "Acquisition", MdaParams),
    ControlSpec("move_to_item", "Move to current item", "Acquisition", MoveToItemParams,
                description="Move to the current position, well, tile or Z of a loop "
                            "that runs with move: false"),
    ControlSpec("break", "Break", "Flow", BreakParams),
    ControlSpec("continue", "Continue", "Flow", EmptyParams),
    ControlSpec("finish", "Finish", "Flow", MessageParams,
                description="End the run normally; data so far is kept"),
    ControlSpec("abort", "Abort", "Flow", MessageParams,
                description="End the run as aborted; data so far is kept and marked"),
    ControlSpec("exit_with_error", "Exit with error", "Flow", MessageParams),
    ControlSpec("wait", "Wait", "Flow", WaitParams),
    ControlSpec("wait_until", "Wait until", "Flow", WaitUntilParams, expr_fields=("expr",)),
    ControlSpec("question", "Question", "Flow", QuestionParams),
    ControlSpec("set_variable", "Set variable", "Flow", SetVariableParams),
    ControlSpec("log", "Log", "Flow", LogParams),
]}

LOOP_AXES = {s.id: s.axis for s in CONTROL_SPECS.values() if s.loop and s.axis}
MOVING_LOOPS = ("loop_positions", "loop_wells", "loop_grid", "zstack")
