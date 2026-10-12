"""Convert between useq ``MDASequence`` and job documents.

Import: a sequence that is a plain product of axes (positions without
sub-sequences, a relative grid, z, channels without ``do_stack``/``acquire_every``,
a time plan, autofocus once per position) becomes nested loop nodes, one per
axis in ``axis_order``. Anything else is kept whole as one ``mda`` node, so the
import never loses information.

Export: a job whose tree is a single chain of those loops ending in a capture
(or a single ``mda`` node) becomes an ``MDASequence``. Conditions, variables,
peripheral tasks and ``move: false`` loops have no useq equivalent and raise
``NotExportable``.

Loop nodes index frames on the same axes as useq (t, p, g, z, c), so a capture
inside the loops gets the index the matching ``MDAEvent`` would have.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import useq

from .document import CaptureChannel, Job, Node, resource_ref

AXIS_LOOP = {"t": "time_lapse", "p": "loop_positions", "g": "loop_grid",
             "z": "zstack", "c": "loop_channels"}
LOOP_AXIS = {v: k for k, v in AXIS_LOOP.items()}
LOOP_PARAM = {"time_lapse": "plan", "loop_positions": "source", "loop_grid": "plan",
              "zstack": "plan", "loop_channels": "channels"}
RESOURCE = {"t": ("time", "time_plan"), "p": ("positions", "positions"),
            "g": ("grid", "grid"), "z": ("z", "z_plan"), "c": ("capture", "capture")}


class NotExportable(ValueError):
    """The job uses something a useq MDASequence cannot express."""


def _expandable(seq: useq.MDASequence) -> Optional[str]:
    """None if the sequence maps onto plain nested loops, else the reason."""
    if isinstance(seq.stage_positions, useq.WellPlatePlan):
        return "well-plate plans are kept as one mda node"
    if any(getattr(p, "sequence", None) is not None for p in seq.stage_positions):
        return "positions with sub-sequences"
    if any(not c.do_stack or c.acquire_every != 1 for c in seq.channels):
        return "channels with do_stack=False or acquire_every"
    if seq.grid_plan is not None and seq.stage_positions and not seq.grid_plan.is_relative:
        return "an absolute grid combined with stage positions"
    af = seq.autofocus_plan
    if af is not None and (not isinstance(af, useq.AxesBasedAF) or set(af.axes) != {"p"}):
        return "autofocus other than once per position"
    return None


def _sequence_resources(seq: useq.MDASequence, used: Dict[str, bool]) -> Dict[str, Any]:
    resources: Dict[str, Any] = {}
    if used["t"]:
        resources["time"] = {"kind": "time_plan", "plan": seq.time_plan}
    if used["p"]:
        resources["positions"] = {"kind": "positions", "positions": [
            useq.AbsolutePosition(**p.model_dump(exclude={"sequence"}))
            for p in seq.stage_positions]}
    if used["g"]:
        resources["grid"] = {"kind": "grid", "plan": seq.grid_plan}
    if used["z"]:
        resources["z"] = {"kind": "z_plan", "plan": seq.z_plan}
    if used["c"]:
        resources["capture"] = {"kind": "capture", "channels": [
            CaptureChannel(**c.model_dump()) for c in seq.channels]}
    return resources


def _whole(seq: useq.MDASequence, name: str, reason: str) -> Job:
    dumped = seq.model_dump(mode="json", exclude_unset=True)
    return Job(name=name, metadata={"source": "useq", "kept_whole_because": reason},
               tasks=[Node(id="mda", task="mda", params={"sequence": dumped})])


def job_from_sequence(seq: useq.MDASequence, name: str = "useq sequence",
                      expand: bool = True) -> Job:
    """A job equivalent to ``seq``."""
    reason = _expandable(seq) if expand else "expand=False"
    if reason is not None:
        return _whole(seq, name, reason)

    used = {"t": seq.time_plan is not None, "p": bool(seq.stage_positions),
            "g": seq.grid_plan is not None, "z": seq.z_plan is not None,
            "c": bool(seq.channels)}
    order = [a for a in seq.axis_order if used.get(a)]

    # innermost first: the capture, then wrap it in each loop outwards
    c_innermost = bool(order) and order[-1] == "c"
    if c_innermost:
        leaf_params: Dict[str, Any] = {"channels": "@capture"}
    else:
        leaf_params = {} if used["c"] else {"channels": []}
    inner: List[Node] = [Node(id="capture", task="capture", params=leaf_params)]
    for axis in reversed(order[:-1] if c_innermost else order):
        loop = AXIS_LOOP[axis]
        body = inner
        if axis == "p" and seq.autofocus_plan is not None:
            af = Node(id="autofocus", task="autofocus", params={"mode": "hardware"})
            body = [af] + body
        ref = f"@{RESOURCE[axis][0]}"
        inner = [Node(id=f"loop_{axis}", task=loop, params={LOOP_PARAM[loop]: ref}, body=body)]

    meta: Dict[str, Any] = {"source": "useq"}
    if seq.keep_shutter_open_across:
        meta["keep_shutter_open_across"] = list(seq.keep_shutter_open_across)
    if seq.metadata:
        meta["useq_metadata"] = dict(seq.metadata)
    return Job(name=name, resources=_sequence_resources(seq, used), tasks=inner, metadata=meta)


def _resource(job: Job, value: Any, kind: str) -> Any:
    name = resource_ref(value)
    if name is None:
        return value
    res = job.resources.get(name)
    if res is None or res.kind != kind:
        raise NotExportable(f"'@{name}' is not a {kind} resource")
    return res


def _useq_channels(chans) -> List[useq.Channel]:
    fields = set(useq.Channel.model_fields)
    return [useq.Channel(**(c if isinstance(c, dict) else c.model_dump(include=fields)))
            for c in chans]


# axis -> (resource kind, MDASequence field, attribute holding the value)
_AXIS_FIELD = {"t": ("time_plan", "time_plan", "plan"),
               "p": ("positions", "stage_positions", "positions"),
               "g": ("grid", "grid_plan", "plan"),
               "z": ("z_plan", "z_plan", "plan"),
               "c": ("capture", "channels", "channels")}


def _loop_value(job: Job, node: Node, axis: str) -> Any:
    kind, _, attr = _AXIS_FIELD[axis]
    res = _resource(job, node.params.get(LOOP_PARAM[node.task]), kind)
    value = getattr(res, attr, res)
    if axis == "c":
        return _useq_channels(value)
    return list(value) if axis == "p" else value


def _plain_body(nodes: List[Node], order: List[str], kwargs: Dict[str, Any]) -> Node:
    """The single node of a chain level (after an optional per-position autofocus)."""
    body = [n for n in nodes if n.enabled is not False]
    if len(body) == 2 and body[0].task == "autofocus" and order and order[-1] == "p":
        if body[0].params.get("mode") != "hardware":
            raise NotExportable("only hardware autofocus once per position maps to useq")
        kwargs["autofocus_plan"] = useq.AxesBasedAF(axes=("p",))
        body = body[1:]
    if len(body) != 1:
        raise NotExportable(f"expected one node at depth {len(order)}, found {len(body)}")
    node = body[0]
    if node.every is not None or node.orelse or node.idle or isinstance(node.enabled, str):
        raise NotExportable(f"node {node.id!r} uses every/else/idle/conditional enable")
    return node


def _capture(job: Job, node: Node, order: List[str], kwargs: Dict[str, Any]) -> None:
    if node.params.get("save") is False:
        raise NotExportable("capture without saving has no useq equivalent")
    chans = node.params.get("channels")
    if chans:
        res = _resource(job, chans, "capture")
        kwargs["channels"] = _useq_channels(getattr(res, "channels", res))
        order.append("c")


def sequence_from_job(job: Job) -> useq.MDASequence:
    """The useq ``MDASequence`` equivalent to ``job``, if there is one."""
    if job.parameters or job.variables:
        raise NotExportable("job parameters and variables have no useq equivalent")
    if len(job.tasks) == 1 and job.tasks[0].task == "mda":
        return useq.MDASequence(**job.tasks[0].params["sequence"])

    kwargs: Dict[str, Any] = {}
    order: List[str] = []
    nodes = job.tasks
    while True:
        node = _plain_body(nodes, order, kwargs)
        if node.task == "capture":
            _capture(job, node, order, kwargs)
            break
        axis = LOOP_AXIS.get(node.task)
        if axis is None:
            raise NotExportable(f"task {node.task!r} has no useq equivalent")
        if axis in order:
            raise NotExportable(f"axis {axis!r} is looped twice")
        if node.params.get("move", True) is False:
            raise NotExportable(f"{node.id!r} runs with move: false")
        kwargs[_AXIS_FIELD[axis][1]] = _loop_value(job, node, axis)
        order.append(axis)
        nodes = node.body

    # useq wants every axis in axis_order; put unused ones after the used ones
    order += [a for a in "tpgcz" if a not in order]
    return useq.MDASequence(axis_order="".join(order), **kwargs)
