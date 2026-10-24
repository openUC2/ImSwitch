"""
ESP32StageManager with arbitrary axis letters mapped to firmware stepperids,
against the real uc2rest Motor/Home and a fake serial link (no hardware).
"""
from types import SimpleNamespace
from unittest import mock

import pytest
from uc2rest.home import Home
from uc2rest.motor import Motor

from imswitch.imcontrol.model.managers.positioners.ESP32StageManager import ESP32StageManager


class FakeParent:
    """Just enough of UC2Client for uc2rest: records every command, answers /motor_get."""

    def __init__(self):
        self.serial = mock.Mock(is_connected=False, use_qid_done=True, DEBUG=False)
        self.logger = mock.Mock()
        self.sent = []
        self.positions = {}  # stepperid -> firmware steps

    def post_json(self, path, payload, **kwargs):
        self.sent.append(payload)
        if path == "/motor_get":
            return [{"steppers": [{"stepperid": i, "position": p} for i, p in self.positions.items()]}]
        return {}

    def last(self, key):
        return next(p[key] for p in reversed(self.sent) if key in p)["steppers"]


def make_stage(axes, name="ESP32Stage", parent=None, **props):
    if parent is None:
        parent = FakeParent()
        parent.motor = Motor(parent=parent)
        parent.home = Home(parent)
    info = SimpleNamespace(axes=axes, managerProperties={"rs232device": "ESP32", **props}, stageOffsets={},
                           forPositioning=True, forScanning=False, resetOnClose=True)
    emitted = []
    comm = SimpleNamespace(sigUpdateMotorPosition=SimpleNamespace(emit=emitted.append),
                           sigHomingState=SimpleNamespace(emit=lambda *a: None))
    esp32 = SimpleNamespace(motor=parent.motor, home=parent.home)
    stage = ESP32StageManager(info, name, rs232sManager={"ESP32": SimpleNamespace(_esp32=esp32)}, commChannel=comm)
    return stage, parent, emitted


def ids(steppers):
    return sorted(s["stepperid"] for s in steppers)


def test_legacy_xyza_stage_keeps_its_ids():
    stage, parent, _ = make_stage(["X", "Y", "Z", "A"], stepsizeX=0.5)
    stage.move(100, "X")
    assert [(s["stepperid"], s["position"]) for s in parent.last("motor")] == [(1, 200)]
    stage.move((10, 20), "XY")
    assert ids(parent.last("motor")) == [1, 2]


def test_dual_axis_z_drives_a_too():
    stage, parent, _ = make_stage(["X", "Y", "Z"], isDualaxis=True)
    stage.move(50, "Z")
    assert ids(parent.last("motor")) == [0, 3]


def test_satellite_stage_shares_the_board():
    main, parent, mainEmitted = make_stage(["X", "Y", "Z", "A"])
    sat, _, satEmitted = make_stage(["B", "C", "D", "E"], name="Satellite", parent=parent,
                                    stepsizeC=0.5, homeBenabled=True, homeSpeedB=4000)
    sat.move(10, "C")
    assert [(s["stepperid"], s["position"]) for s in parent.last("motor")] == [(5, 20)]

    parent.motor._callback_motor_status({"steppers": [{"stepperid": 5, "position": 40}]})
    assert satEmitted[-1]["Satellite"]["C"] == 20 and "ESP32Stage" in mainEmitted[-1]

    parent.positions = {4: 7}
    assert sat.getPosition()["B"] == 7
    assert ids(parent.sent[-1]["motor"]["steppers"]) == [4, 5, 6, 7]   # CAN ids asked for explicitly

    sat.doHome("B")
    home = parent.last("home")[0]
    assert (home["stepperid"], home["speed"]) == (4, 4000)

    sat.stopAll()
    assert ids(parent.last("motor")) == [4, 5, 6, 7]
    assert list(sat.getMotorSettings()["axes"]) == ["B", "C", "D", "E"]
    assert sat.getMotorSettingsForAxis("C")["stepperId"] == 5


def test_custom_letters_need_an_id():
    with pytest.raises(ValueError):
        make_stage(["Q"])
    stage, parent, _ = make_stage(["Q"], axisIds={"Q": 8})
    stage.move(5, "Q")
    assert ids(parent.last("motor")) == [8]


def test_relative_move_is_counted_once_when_the_board_reports_meanwhile():
    """uc2rest counts a move into its position before sending it; a status message that
    arrives during the send pushed that into the stage, and move() added the step again."""
    stage, parent, _ = make_stage(["X", "Y", "Z", "A"])
    send = parent.post_json

    def post_json(path, payload, **kwargs):
        parent.motor._callback_motor_status({"steppers": [{"stepperid": 0, "position": 0}]})
        return send(path, payload, **kwargs)
    parent.post_json = post_json
    stage.move(1000, "Z", is_blocking=False)
    assert stage.position["Z"] == 1000
    stage.move(1000, "Z", is_blocking=False)
    assert stage.position["Z"] == 2000
