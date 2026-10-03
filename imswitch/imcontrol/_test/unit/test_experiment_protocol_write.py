#!/usr/bin/env python3
"""WP-09 — the protocol file must never be left truncated.

save_experiment_protocol used to open the file with 'w' (which truncates) and
only then serialize, so anything non-serializable in snake_tiles or
workflow_steps raised mid-write and left a 0-byte protocol behind. The caller
swallowed the error, so a run that never really started looked like a run that
produced nothing.

Runs standalone: `.venv/bin/python <this file>`.
"""

import json
import os
import sys
import tempfile
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.controller.controllers.experiment_controller.experiment_mode_base import (  # noqa: E402
    ExperimentModeBase,
)


class _Uncooperative:
    """Serializing this raises: it is callable but has no __name__, which is
    the branch _json_serializer trips over."""

    def __call__(self):  # pragma: no cover - never called
        pass


def _saver():
    """ExperimentModeBase.save_experiment_protocol bound to a bare stub."""
    stub = types.SimpleNamespace(
        controller=types.SimpleNamespace(version="test"),
        _logger=types.SimpleNamespace(info=lambda *a, **k: None,
                                      error=lambda *a, **k: None),
    )
    for name in ("save_experiment_protocol", "_json_serializer"):
        setattr(stub, name, types.MethodType(getattr(ExperimentModeBase, name), stub))
    return stub


def test_valid_protocol_is_written_and_reparses():
    stub = _saver()
    base = os.path.join(tempfile.mkdtemp(), "run")
    path = stub.save_experiment_protocol({"snake_tiles": [[{"x": 1.0}]]}, base, mode="normal")

    assert path == base + "_protocol.json"
    written = json.loads(open(path).read())
    assert written["snake_tiles"] == [[{"x": 1.0}]]
    assert written["mode"] == "normal"
    assert written["timestamp"]


def test_unserializable_field_raises_and_leaves_no_file():
    stub = _saver()
    base = os.path.join(tempfile.mkdtemp(), "run")

    try:
        stub.save_experiment_protocol({"workflow_steps": [_Uncooperative()]}, base)
    except Exception:
        pass
    else:
        raise AssertionError(
            "an unserializable protocol must fail loudly — the run is not "
            "reproducible without it"
        )

    assert not os.path.exists(base + "_protocol.json"), (
        "a truncated protocol was left on disk"
    )


def test_existing_protocol_survives_a_failed_rewrite():
    """The worst case: a good protocol replaced by an empty one."""
    stub = _saver()
    base = os.path.join(tempfile.mkdtemp(), "run")
    stub.save_experiment_protocol({"snake_tiles": [[{"x": 1.0}]]}, base)

    try:
        stub.save_experiment_protocol({"workflow_steps": [_Uncooperative()]}, base)
    except Exception:
        pass

    assert json.loads(open(base + "_protocol.json").read())["snake_tiles"]


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
