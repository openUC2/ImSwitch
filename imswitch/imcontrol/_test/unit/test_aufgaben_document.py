"""Aufgaben job documents: format, validation, examples, portability."""
import ast
import json
from pathlib import Path

import jsonschema
import pytest

from imswitch.imcontrol.model import aufgaben as A
from imswitch.imcontrol.model.aufgaben.document import expression, resource_ref

PKG = Path(A.__file__).parent
SCHEMA_FILE = Path(__file__).parents[4] / "docs" / "aufgaben" / "job.schema.json"


def test_package_never_imports_imswitch():
    # The engine core must move to newswitch unchanged.
    for path in PKG.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                assert not name.startswith("imswitch"), f"{path.name} imports {name}"


@pytest.mark.parametrize("name", sorted(A.examples()))
def test_examples_validate(name):
    report = A.validate(A.examples()[name])
    assert report.ok, [i.message for i in report.errors]


@pytest.mark.parametrize("name", sorted(A.examples()))
def test_saved_jobs_match_the_json_schema(name):
    # The schema describes the canonical form a job is saved in.
    saved = json.loads(A.examples()[name].to_json())
    jsonschema.validate(saved, A.job_json_schema())
    assert A.Job.model_validate(saved) == A.examples()[name]


def test_committed_schema_is_current():
    assert json.loads(SCHEMA_FILE.read_text()) == json.loads(json.dumps(A.job_json_schema())), \
        "regenerate docs/aufgaben/job.schema.json from aufgaben.job_json_schema()"


def test_json_round_trip():
    job = A.examples()["tile_scan_focus_check"]
    again = A.Job.model_validate_json(job.to_json())
    assert again == job
    assert json.loads(job.to_json())["schema"] == A.SCHEMA_ID


def test_ref_and_expression_markers():
    assert resource_ref("@z") == "z"
    assert resource_ref("@@literal") is None
    assert resource_ref("z") is None
    assert expression("${a + 1}") == "a + 1"
    assert expression("a + 1") is None


def _job(tasks, **kw):
    return A.Job.model_validate({"name": "t", **kw, "tasks": tasks})


def _messages(job):
    return [(i.node, i.message) for i in A.validate(job).errors]


def test_duplicate_ids_and_unknown_tasks():
    job = _job([{"id": "a", "task": "wait", "params": {"seconds": 1}},
                {"id": "a", "task": "teleport"}])
    msgs = _messages(job)
    assert ("a", "duplicate node id 'a'") in msgs
    assert ("a", "unknown task 'teleport'") in msgs


def test_branches_only_where_allowed():
    job = _job([{"id": "w", "task": "wait", "params": {"seconds": 1},
                 "body": [{"id": "x", "task": "log", "params": {"message": "hi"}}]}])
    assert ("w", "wait cannot have 'body'") in _messages(job)


def test_resource_kind_and_existence():
    job = _job([{"id": "z", "task": "zstack", "params": {"plan": "@nothing"},
                 "body": [{"id": "c", "task": "capture", "params": {"channels": "@cap"}}]}],
               resources={"cap": {"kind": "capture", "channels": [{"config": "LED"}]}})
    assert ("z", "plan: no resource 'nothing'") in _messages(job)


def test_break_and_every_need_their_loop():
    job = _job([{"id": "b", "task": "break"},
                {"id": "r", "task": "repeat", "params": {"count": 2}, "body": [
                    {"id": "l", "task": "log", "params": {"message": "x"},
                     "every": {"nth": 2, "of": "elsewhere"}}]}])
    msgs = _messages(job)
    assert ("b", "break outside a loop") in msgs
    assert ("l", "every.of 'elsewhere' is not an enclosing loop") in msgs


def test_repeat_needs_a_limit():
    log = {"id": "l", "task": "log", "params": {"message": "x"}}
    job = _job([{"id": "r", "task": "repeat", "body": [log]}])
    assert any("repeat needs count, until, or both" in m for _, m in _messages(job))


def test_expressions_are_checked_against_the_document():
    job = _job([{"id": "c", "task": "condition", "params": {"expr": "var.n > param.limit"},
                 "body": [{"id": "l", "task": "log", "params": {"message": "x"}}]},
                {"id": "s", "task": "set_variable", "params": {"name": "undeclared", "value": 1}}],
               variables={"n": 0})
    msgs = _messages(job)
    assert ("c", "unknown parameter param.limit") in msgs
    assert ("s", "set_variable: 'undeclared' is not a declared variable") in msgs


def test_task_parameters_are_type_checked():
    job = _job([{"id": "m", "task": "move_relative", "params": {"dx_um": "far"}}])
    assert any(n == "m" and m.startswith("dx_um:") for n, m in _messages(job))


def test_from_resource_fills_task_parameters():
    job = _job([{"id": "f", "task": "autofocus", "params": {"$from": "@af", "range_um": 20}}],
               resources={"af": {"kind": "autofocus", "range_um": 40, "step_um": 2}})
    assert A.validate(job).ok


def test_safety_summary():
    job = _job([{"id": "h", "task": "home_axis", "params": {"axis": "Z"}},
                {"id": "o", "task": "set_objective", "params": {"slot": 1}},
                {"id": "l", "task": "set_illumination",
                 "params": {"source": "LED", "intensity": 10}}])
    report = A.validate(job)
    assert report.ask_first == ["h", "o"]
    assert report.moves_z and report.emits_light


def test_capabilities_hide_unavailable_tasks():
    job = _job([{"id": "p", "task": "pump_volume", "params": {"pump": "p1", "volume_ul": 10}}])
    assert A.validate(job).ok
    report = A.validate(job, capabilities={"stage", "camera"})
    assert any("needs ['pump']" in i.message for i in report.errors)
    ids = {e["id"] for e in A.catalogue(capabilities={"stage", "camera"})}
    assert "move_to_position" in ids and "pump_volume" not in ids and "zstack" in ids
