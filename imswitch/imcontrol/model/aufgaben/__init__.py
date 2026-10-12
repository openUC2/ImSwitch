"""Aufgaben: composable acquisition jobs (a JOBS-style workflow model).

Independent of ImSwitch: this package imports only pydantic, useq-schema,
numpy and the standard library, so it can move to newswitch unchanged (see
docs/AUFGABEN_ROADMAP.md). The ImSwitch side is a host adapter implementing
``Microscope`` plus ``AufgabenController`` (Phase 1).

- document:     the job format (``Job``, ``Node``, resources, storage)
- control:      loops, conditions and flow steps the engine runs itself
- registry:     ``@task`` and the palette catalogue
- primitives:   the v1 hardware tasks, written against ``Microscope``
- hardware:     the ``Microscope`` protocol a host implements
- expressions:  the safe expression language
- validate:     checks a job before it runs
- useq_bridge:  MDASequence <-> job
"""
import json
from pathlib import Path

from . import primitives  # noqa: F401  (registers the v1 tasks)
from .control import CONTROL_SPECS, ControlSpec
from .document import SCHEMA_ID, CaptureChannel, Job, Node, Parameter, Storage
from .hardware import CAPABILITIES, Microscope
from .registry import REGISTRY, TaskContext, TaskFlags, TaskSpec, catalogue, task
from .useq_bridge import NotExportable, job_from_sequence, sequence_from_job
from .validate import Report, validate

EXAMPLES_DIR = Path(__file__).parent / "examples"


def job_json_schema() -> dict:
    """JSON Schema of a saved job (draft 2020-12, from pydantic).

    It describes the canonical form ``Job.to_json()`` writes. Loading also
    accepts useq's shorthands (a plate name instead of a plate object, a scalar
    grid overlap), which useq's validators coerce but its schema does not list.
    """
    schema = Job.model_json_schema(by_alias=True, mode="serialization")
    schema["$id"] = f"https://openuc2.github.io/schemas/{SCHEMA_ID}.json"
    _durations_accept_seconds(schema)
    return schema


def _durations_accept_seconds(node) -> None:
    """useq writes time-plan durations as seconds but declares ISO strings."""
    if isinstance(node, dict):
        for key, value in list(node.items()):
            if isinstance(value, dict) and value.get("format") == "duration" \
                    and value.get("type") == "string":
                rest = {k: v for k, v in value.items() if k not in ("type", "format")}
                node[key] = {**rest, "anyOf": [{"type": "number", "description": "seconds"},
                                               {"type": "string", "format": "duration"}]}
            else:
                _durations_accept_seconds(value)
    elif isinstance(node, list):
        for item in node:
            _durations_accept_seconds(item)


def load_job(path) -> Job:
    return Job.model_validate(json.loads(Path(path).read_text()))


def examples() -> dict:
    """The bundled example jobs by file stem."""
    return {p.stem: load_job(p) for p in sorted(EXAMPLES_DIR.glob("*.json"))}


__all__ = [
    "SCHEMA_ID", "Job", "Node", "Parameter", "Storage", "CaptureChannel",
    "CONTROL_SPECS", "ControlSpec", "REGISTRY", "TaskContext", "TaskFlags", "TaskSpec",
    "catalogue", "task", "Microscope", "CAPABILITIES", "validate", "Report",
    "job_from_sequence", "sequence_from_job", "NotExportable",
    "job_json_schema", "load_job", "examples", "EXAMPLES_DIR",
]
