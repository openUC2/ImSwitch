#!/usr/bin/env python3
"""WP-06 — autofocus must report failure, and its offset belongs to one region.

Two field symptoms shared one cause. The experiment read the focus result from
`self.mStage.getPosition()["Z"]` instead of autofocus's own return value, so a
failed or timed-out autofocus recorded wherever the stage happened to sit —
usually the start of the scan range. That value became a single global Z offset
added to every later capture move, so one bad point silently defocused the rest
of the run. Squid keeps these per region and gates on an explicit success flag.

Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys
import types

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.controller.controllers.ExperimentController import (  # noqa: E402
    ExperimentController,
)


def _controller():
    """The Z-offset bookkeeping bound to a bare stub."""
    moves = []
    stub = types.SimpleNamespace(
        _experiment_af_offsets={},
        _af_successes=0,
        _af_failures=0,
        SPEED_Z=1000,
        ACCELERATION_Z=1000,
        moves=moves,
        mStage=types.SimpleNamespace(
            move=lambda value, **kw: moves.append(value),
            getPosition=lambda: {"Z": -999.0},  # never the right answer
        ),
        _logger=types.SimpleNamespace(
            info=lambda *a, **k: None,
            debug=lambda *a, **k: None,
            warning=lambda *a, **k: None,
        ),
    )
    for name in ("move_stage_z", "update_af_offset"):
        setattr(stub, name, types.MethodType(getattr(ExperimentController, name), stub))
    return stub


def test_offset_is_stored_per_region():
    c = _controller()
    c.update_af_offset(metadata={"result": 110.0}, expected_z=100.0, region_id="area_0")
    c.update_af_offset(metadata={"result": 95.0}, expected_z=100.0, region_id="area_1")
    assert c._experiment_af_offsets == {"area_0": 10.0, "area_1": -5.0}
    assert (c._af_successes, c._af_failures) == (2, 0)


def test_a_region_only_ever_gets_its_own_offset():
    """The whole point: one bad/absent region must not move another."""
    c = _controller()
    c.update_af_offset(metadata={"result": 110.0}, expected_z=100.0, region_id="area_0")

    c.move_stage_z(50.0, af_region_id="area_0")
    c.move_stage_z(50.0, af_region_id="area_1")   # never autofocused
    c.move_stage_z(50.0)                          # not a capture move
    assert c.moves == [60.0, 50.0, 50.0]


def test_failed_autofocus_keeps_the_previous_offset_and_is_counted():
    c = _controller()
    c.update_af_offset(metadata={"result": 110.0}, expected_z=100.0, region_id="area_0")

    assert c.update_af_offset(metadata={"result": None}, expected_z=100.0,
                              region_id="area_0") is None
    assert c._experiment_af_offsets["area_0"] == 10.0   # untouched
    assert (c._af_successes, c._af_failures) == (1, 1)

    # ... and the run continues at that unchanged offset.
    c.move_stage_z(50.0, af_region_id="area_0")
    assert c.moves == [60.0]


def test_relative_moves_never_get_an_offset():
    c = _controller()
    c.update_af_offset(metadata={"result": 110.0}, expected_z=100.0, region_id="area_0")
    c.move_stage_z(5.0, relative=True, af_region_id="area_0")
    assert c.moves == [5.0]


def test_offset_is_not_taken_when_the_caller_opted_out():
    c = _controller()
    assert c.update_af_offset(metadata={"result": 110.0}, expected_z=100.0,
                              region_id="area_0", apply_global_offset=False) is None
    assert c._experiment_af_offsets == {}


def test_autofocus_timeout_scales_with_the_scan_instead_of_a_flat_two_minutes():
    timeout = ExperimentController._autofocus_timeout_s

    # A short, fast scan must not cost anywhere near the old flat 120 s.
    quick = timeout(af_range=20.0, af_resolution=10.0, af_settle_time=0.0, two_stage=False)
    assert 10.0 <= quick < 30.0

    # A longer scan gets proportionally longer...
    longer = timeout(af_range=200.0, af_resolution=5.0, af_settle_time=0.2, two_stage=False)
    assert longer > quick

    # ... two-stage more still, and everything stays under the hard ceiling.
    assert timeout(200.0, 5.0, 0.2, True) >= longer
    assert timeout(10000.0, 0.1, 5.0, True) == 120.0


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all passed")
