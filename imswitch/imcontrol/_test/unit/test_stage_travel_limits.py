"""
Travel limits (no-go area) of ESP32StageManager in the device frame, jog clamping and
the FRAME-calibration travel probe, against uc2rest with a fake serial link (no hardware).
"""
from types import SimpleNamespace

import pytest

from imswitch.imcontrol.controller.controllers import frame_calibration_tools as fct
from imswitch.imcontrol.model.managers.positioners.PositionerManager import StageLimitError
from imswitch.imcontrol._test.unit.test_esp32_stage_axes import FakeParent, make_stage
from uc2rest.home import Home
from uc2rest.motor import Motor

LIMITS = dict(minX=0, maxX=10000, limitXenabled=True)


def stage_with_signal(**props):
    parent = FakeParent()
    parent.motor = Motor(parent=parent)
    parent.home = Home(parent)
    stage, parent, _ = make_stage(["X", "Y", "Z", "A"], parent=parent, **props)
    refused = []
    stage._commChannel.sigStageLimit = SimpleNamespace(emit=refused.append)
    return stage, parent, refused


def sent_moves(parent):
    return [p for p in parent.sent
            if "motor" in p and any("position" in s for s in p["motor"]["steppers"])]


def test_absolute_move_is_checked_in_the_device_frame():
    stage, parent, refused = stage_with_signal(**LIMITS)
    stage.stageOffsetPositions["X"] = 3000          # user 0 = device 3000
    stage.move(6000, "X", is_absolute=True)          # device 9000: inside
    assert stage.position["X"] == 6000
    n = len(sent_moves(parent))
    with pytest.raises(StageLimitError, match="9000|10000"):
        stage.move(8000, "X", is_absolute=True)      # device 11000: user-frame check would pass
    assert len(sent_moves(parent)) == n              # nothing sent
    assert refused and "outside the travel range" in refused[-1]["message"]
    with pytest.raises(StageLimitError):
        stage.move(-3500, "X", is_absolute=True)     # device -500
    stage.move(-3000, "X", is_absolute=True)         # device 0: on the limit is allowed


def test_relative_and_combined_moves_are_refused():
    stage, _, _ = stage_with_signal(**LIMITS, minY=0, maxY=5000, limitYenabled=True)
    stage.move(9000, "X")
    with pytest.raises(StageLimitError):
        stage.move(2000, "X")                        # 11000
    with pytest.raises(StageLimitError):
        stage.move((100, 6000), "XY", is_absolute=True)
    assert stage.position["X"] == 9000


def test_disabled_limits_do_not_refuse():
    stage, _, _ = stage_with_signal(minX=0, maxX=10000)
    stage.move(50000, "X", is_absolute=True)
    assert stage.position["X"] == 50000


def last_x(parent):
    return next(st for p in reversed(parent.sent) if "motor" in p
                for st in p["motor"]["steppers"] if st["stepperid"] == 1)


def test_jog_becomes_an_absolute_move_to_the_limit_and_stops():
    stage, parent, refused = stage_with_signal(**LIMITS)
    stage.stageOffsetPositions["X"] = 2000
    stage.move(1000, "X", is_absolute=True)          # device 3000
    stage.moveForever(speed=(0, 5000, 0, 0))
    x = last_x(parent)
    assert x["position"] == 10000 and x["isabs"] and not x.get("isforever")
    stage.moveForever(speed=(0, 0, 0, 0), is_stop=True)
    assert last_x(parent).get("isstop")

    stage.moveForever(speed=(0, -5000, 0, 0))        # towards min: device 0
    x = last_x(parent)
    assert x["position"] == 0 and x["isabs"]

    stage._position["X"] = -2000                     # device 0, on the min limit
    n = len(parent.sent)
    stage.moveForeverByAxis(speed=-5000, axis="X")
    assert len(parent.sent) == n and "jog refused" in refused[-1]["message"]


def test_unlimited_axis_still_jogs_forever():
    stage, parent, _ = stage_with_signal(**LIMITS)
    stage.moveForeverByAxis(speed=3000, axis="Y")
    y = next(s for s in parent.last("motor") if s["stepperid"] == 2)
    assert y["isforever"] == 1 and y["speed"] == 3000


def test_startup_dummy_move_bypasses_limits():
    # a stage parked exactly on its min limit must still start (the ±1 µm nudge)
    stage, _, _ = stage_with_signal(minX=0, maxX=10, limitXenabled=True)
    assert stage.limitEnabled["X"]


def test_probe_reports_where_the_end_switch_stopped_the_axis():
    stage, parent, _ = stage_with_signal(homeDirectionX=-1)
    readings = iter([0, 120000])                     # before, after the hard-limit stop
    stage.getDevicePositionAxis = lambda axis: next(readings)
    start, end = stage.probeTravelEnd("X", 200000, backoffUm=500)
    assert (start, end) == (0, 120000)
    moves = [next(s for s in p["motor"]["steppers"] if s["stepperid"] == 1)
             for p in sent_moves(parent)[-2:]]
    assert moves[0]["position"] > 0 and moves[1]["position"] < 0   # away from home, then back off

    readings = iter([0, 200000])                     # ran the full distance: no switch
    stage.getDevicePositionAxis = lambda axis: next(readings)
    assert stage.probeTravelEnd("X", 200000)[1] is None

    stage.hardLimitsEnabled["X"] = False
    with pytest.raises(RuntimeError, match="hard limits"):
        stage.probeTravelEnd("X", 200000)


def test_travel_limits_from_a_probe():
    assert fct.travel_limits(0, 120000, 500) == (0, 119500)
    assert fct.travel_limits(1000, -80000, 500) == (-79500, 1000)   # home at +, far end at -
    assert fct.travel_limits(0, None, 500) == (None, None)
    assert fct.travel_limits(0, 300, 500) == (None, None)           # switch still pressed


def test_frame_calibration_travel_step_stores_device_frame_limits():
    from imswitch.imcontrol.controller.controllers.FrameCalibrationController import (
        DEFAULT_SETTINGS, FrameCalibrationController)
    stage, _, _ = stage_with_signal(minZ=-100, maxZ=5000, limitZenabled=True)
    ends = {"X": (0, 120000), "Y": (0, None)}       # Y: no end switch found
    stage.probeTravelEnd = lambda ax, maxUm, backoffUm: ends[ax]
    props, warnings, saved = {}, [], []
    fake = SimpleNamespace(
        _stageManager=lambda: stage, _checkAbort=lambda: None, _warn=warnings.append,
        _saveSetup=lambda: saved.append(True), _ctx={},
        _setupInfo=SimpleNamespace(
            positioners={stage.name: SimpleNamespace(managerProperties=props)}))
    detail = FrameCalibrationController._step_travel(fake, dict(DEFAULT_SETTINGS))
    assert props == {"minX": 0, "maxX": 119500, "limitXenabled": True}
    assert stage.limitEnabled["X"] and stage.maxPos["X"] == 119500 and not stage.limitEnabled["Y"]
    assert saved and len(warnings) == 1 and "Y" in warnings[0]
    assert "X 0 .. 119500" in detail and "Z from the config: -100 .. 5000" in detail
    with pytest.raises(StageLimitError):
        stage.move(119600, "X", is_absolute=True)


def test_limits_reported_in_the_user_frame():
    import types
    from imswitch.imcontrol.model.managers.positioners.PositionerManager import INF, PositionerManager
    p = types.SimpleNamespace(minPos={"X": 0.0, "Z": -INF}, maxPos={"X": 90000.0, "Z": INF},
                              limitEnabled={"X": True, "Z": False}, stageOffsetPositions={"X": 1000})
    limits = PositionerManager.getTravelLimitsUser(p)
    assert limits["X"] == {"min": -1000.0, "max": 89000.0, "enabled": True}
    assert limits["Z"] == {"min": None, "max": None, "enabled": False}
