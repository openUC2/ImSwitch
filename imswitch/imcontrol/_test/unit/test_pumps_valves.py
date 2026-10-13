"""Pump and valve managers: unit conversion, blocking and stopping, positions."""
import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from imswitch.imcontrol.model.SetupInfo import PeripheralInfo, SetupInfo
from imswitch.imcontrol.model.managers.PumpsManager import PumpsManager
from imswitch.imcontrol.model.managers.ValvesManager import ValvesManager
from imswitch.imcontrol.model.managers.pumps.StepperPumpManager import StepperPumpManager
from imswitch.imcontrol.model.managers.pumps.VirtualPumpManager import VirtualPumpManager
from imswitch.imcontrol.model.managers.valves.DigitalOutValveManager import DigitalOutValveManager
from imswitch.imcontrol.model.managers.valves.SerialValveManager import SerialValveManager
from imswitch.imcontrol.model.managers.valves.ServoValveManager import ServoValveManager


class FakeMotor:
    def __init__(self):
        self.stepSize = np.array([2.0, 1.0, 1.0, 1.0])   # A has 2 physical units per step
        self.calls = []

    def axis_id(self, axis):
        return "AXYZ".index(axis)

    def move_axis_by_name(self, **kw):
        self.calls.append(("move", kw))

    def move_forever(self, speed):
        self.calls.append(("forever", speed))

    def stop(self, axis):
        self.calls.append(("stop", axis))


class FakeBoard:
    def __init__(self):
        self.motor = FakeMotor()
        self.digitalout = SimpleNamespace(calls=[])
        self.digitalout.set_digitalout = lambda **kw: self.digitalout.calls.append(kw)
        self.laser = SimpleNamespace(calls=[])
        self.laser.set_servo = lambda **kw: self.laser.calls.append(kw)


def low_level(board=None, port=None):
    return {"rs232sManager": {"ESP32": SimpleNamespace(_esp32=board), "valve": port}}


def info(manager, **props):
    return PeripheralInfo(managerName=manager, managerProperties=props)


def test_stepper_pump_converts_volume_and_rate_to_steps():
    board = FakeBoard()
    pump = StepperPumpManager(info("StepperPumpManager", axis="A", stepsPerUl=100,
                                   maxRateUlMin=600), "p", **low_level(board))
    pump.pump_volume(-5, 300)                       # backwards, 300 µL/min
    kind, kw = board.motor.calls[0]
    assert kind == "move" and kw["axis"] == "A" and kw["is_blocking"] is True
    assert kw["steps"] == -500 * 2.0                # steps x stepSize: uc2rest's units
    assert kw["speed"] == 500                       # 300/60 µL/s x 100 steps/µL
    assert kw["timeout"] > 1.0                      # one second of pumping
    pump.run(-60)
    assert board.motor.calls[1] == ("forever", {"A": -100})
    pump.stop()
    assert board.motor.calls[2] == ("stop", "A")
    with pytest.raises(ValueError):
        pump.pump_volume(1, 601)
    with pytest.raises(ValueError):
        pump.run(0)


def test_virtual_pump_counts_volume_and_stops_early():
    pump = VirtualPumpManager(info("VirtualPumpManager", speedup=600), "v")
    pump.pump_volume(100, 600)                      # 10 s of pumping in 1/60 s
    assert pump.pumpedUl == pytest.approx(100)
    slow = VirtualPumpManager(info("VirtualPumpManager"), "s")
    threading.Timer(0.2, slow.stop).start()
    t0 = time.monotonic()
    slow.pump_volume(1000, 60)                      # would take 1000 s
    assert time.monotonic() - t0 < 2
    assert 0 < slow.pumpedUl < 10 and not slow.running


def test_valves_switch_by_position_name():
    board = FakeBoard()
    multi = DigitalOutValveManager(info("DigitalOutValveManager", positions={
        "sample": {"1": 1, "2": 0}, "waste": {"1": 0, "2": 1}}), "inlet", **low_level(board))
    multi.set_position("waste")
    assert board.digitalout.calls == [{"digitaloutid": 1, "digitaloutval": 0},
                                      {"digitaloutid": 2, "digitaloutval": 1}]
    single = DigitalOutValveManager(info("DigitalOutValveManager", line=3,
                                         positions={"open": 1, "closed": 0}),
                                    "v", **low_level(board))
    single.set_position("open")
    assert board.digitalout.calls[-1] == {"digitaloutid": 3, "digitaloutval": 1}
    assert single.position == "open" and single.positions == ["open", "closed"]
    with pytest.raises(ValueError, match="no position"):
        single.set_position("half")
    servo = ServoValveManager(info("ServoValveManager", channel=2, positions={"a": 0, "b": 90}),
                              "s", **low_level(board))
    servo.set_position("b")
    assert board.laser.calls == [{"channel": 2, "value": 90, "is_blocking": True}]
    port = SimpleNamespace(sent=[])
    port.query = lambda cmd: port.sent.append(cmd) or "OK"
    serial = SerialValveManager(info("SerialValveManager", rs232device="valve",
                                     positions={"1": "GO01\r"}), "sel", **low_level(port=port))
    serial.set_position("1")
    assert port.sent == ["GO01\r"]


def test_setup_blocks_become_managers():
    setup = SetupInfo.from_dict({
        "pumps": {"peristaltic": {"managerName": "VirtualPumpManager",
                                  "managerProperties": {"speedup": 60}}},
        "valves": {"inlet": {"managerName": "VirtualValveManager",
                             "managerProperties": {"positions": {"sample": 1, "waste": 0}}}},
    })
    pumps = PumpsManager(setup.pumps)
    valves = ValvesManager(setup.valves)
    assert pumps.getAllDeviceNames() == ["peristaltic"]
    valves["inlet"].set_position("sample")
    assert valves["inlet"].history == [1]
    assert SetupInfo.from_dict({}).pumps == {}
