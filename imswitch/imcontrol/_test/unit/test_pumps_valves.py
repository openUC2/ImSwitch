"""Pump and valve managers: unit conversion, blocking and stopping, positions."""
import threading
import time
from types import SimpleNamespace

import pytest

from imswitch.imcontrol.model.SetupInfo import PeripheralInfo, SetupInfo
from imswitch.imcontrol.model.managers.PumpsManager import PumpsManager
from imswitch.imcontrol.model.managers.ValvesManager import ValvesManager
from imswitch.imcontrol.model.managers.pumps.LaserPumpManager import LaserPumpManager
from imswitch.imcontrol.model.managers.pumps.PositionerPumpManager import PositionerPumpManager
from imswitch.imcontrol.model.managers.pumps.VirtualPumpManager import VirtualPumpManager
from imswitch.imcontrol.model.managers.valves.DigitalOutValveManager import DigitalOutValveManager
from imswitch.imcontrol.model.managers.valves.LaserValveManager import LaserValveManager


class Devices(dict):
    """A MultiManager stand-in: name -> device."""

    def getAllDeviceNames(self):
        return list(self)


class FakeLaser:
    valueRangeMax = 1023

    def __init__(self):
        self.calls = []

    def setValue(self, value):
        self.calls.append(("value", value))

    def setEnabled(self, enabled):
        self.calls.append(("enabled", enabled))


class FakeStage:
    axes = ["X", "Y", "Z", "A"]
    stepSizes = {"A": 2.0}

    def __init__(self):
        self.calls = []

    def move(self, **kw):
        self.calls.append(("move", kw))

    def moveForever(self, speed):
        self.calls.append(("forever", speed))

    def forceStop(self, axis):
        self.calls.append(("stop", axis))


def info(manager, **props):
    return PeripheralInfo(managerName=manager, managerProperties=props)


def test_laser_pump_sets_pwm_for_the_rate_and_times_the_volume():
    laser = FakeLaser()
    pump = LaserPumpManager(info("LaserPumpManager", laser="pump", maxRateUlMin=600), "p",
                            lasersManager=Devices(pump=laser))
    pump.pump_volume(1, 300)                         # 0.2 s at half speed
    assert laser.calls[:2] == [("value", 1023 / 2), ("enabled", True)]
    assert laser.calls[-2:] == [("enabled", False), ("value", 0)]
    pump.run(60)
    assert laser.calls[-2][1] == pytest.approx(102.3) and laser.calls[-1] == ("enabled", True)
    with pytest.raises(ValueError, match="one way"):
        pump.pump_volume(-1, 60)
    with pytest.raises(ValueError, match="one way"):
        pump.run(-60)
    threading.Timer(0.1, pump.stop).start()
    t0 = time.monotonic()
    pump.pump_volume(1000, 60)                       # would take 1000 s
    assert time.monotonic() - t0 < 2 and laser.calls[-1] == ("value", 0)


def test_positioner_pump_moves_its_axis_in_positioner_units():
    stage = FakeStage()
    pump = PositionerPumpManager(info("PositionerPumpManager", axis="A", stepsPerUl=100,
                                      maxRateUlMin=600), "p", positionersManager=Devices(s=stage))
    pump.pump_volume(-5, 300)                        # backwards, 300 µL/min
    kind, kw = stage.calls[0]
    assert kind == "move" and kw["axis"] == "A" and kw["is_blocking"] is True
    assert kw["value"] == -500 * 2.0                 # steps x stepSize
    assert kw["speed"] == 500                        # steps/s
    pump.run(-60)
    assert stage.calls[1] == ("forever", {"A": -100})
    pump.stop()
    assert stage.calls[2] == ("stop", "A")
    with pytest.raises(ValueError):
        pump.pump_volume(1, 601)
    with pytest.raises(ValueError, match="no axis B"):
        PositionerPumpManager(info("PositionerPumpManager", axis="B", stepsPerUl=1), "q",
                              positionersManager=Devices(s=stage))


def test_virtual_pump_counts_volume_and_stops_early():
    pump = VirtualPumpManager(info("VirtualPumpManager", speedup=600), "v")
    pump.pump_volume(100, 600)                       # 10 s of pumping in 1/60 s
    assert pump.pumpedUl == pytest.approx(100)
    slow = VirtualPumpManager(info("VirtualPumpManager"), "s")
    threading.Timer(0.2, slow.stop).start()
    t0 = time.monotonic()
    slow.pump_volume(1000, 60)                       # would take 1000 s
    assert time.monotonic() - t0 < 2
    assert 0 < slow.pumpedUl < 10 and not slow.running


def test_valves_switch_by_position_name():
    laser = FakeLaser()
    valve = LaserValveManager(info("LaserValveManager", laser="v",
                                   positions={"closed": 0, "open": 800}), "inlet",
                              lasersManager=Devices(v=laser))
    valve.set_position("open")
    valve.set_position("closed")
    assert laser.calls == [("value", 800.0), ("enabled", True), ("value", 0.0), ("enabled", False)]
    assert valve.position == "closed" and valve.positions == ["closed", "open"]
    with pytest.raises(ValueError, match="no position"):
        valve.set_position("half")

    out = SimpleNamespace(calls=[])
    out.set_digitalout = lambda **kw: out.calls.append(kw)
    board = {"rs232sManager": {"ESP32": SimpleNamespace(_esp32=SimpleNamespace(digitalout=out))}}
    multi = DigitalOutValveManager(info("DigitalOutValveManager", positions={
        "sample": {"1": 1, "2": 0}, "waste": {"1": 0, "2": 1}}), "inlet", **board)
    multi.set_position("waste")
    assert out.calls == [{"digitaloutid": 1, "digitaloutval": 0},
                         {"digitaloutid": 2, "digitaloutval": 1}]


def test_setup_blocks_become_managers():
    setup = SetupInfo.from_dict({
        "pumps": {"peristaltic": {"managerName": "LaserPumpManager",
                                  "managerProperties": {"laser": "pwm1", "maxRateUlMin": 500}}},
        "valves": {"inlet": {"managerName": "VirtualValveManager",
                             "managerProperties": {"positions": {"sample": 1, "waste": 0}}}},
    })
    pumps = PumpsManager(setup.pumps, lasersManager=Devices(pwm1=FakeLaser()))
    valves = ValvesManager(setup.valves)
    assert pumps.getAllDeviceNames() == ["peristaltic"]
    assert pumps["peristaltic"].laserName == "pwm1"
    valves["inlet"].set_position("sample")
    assert valves["inlet"].history == [1]
    assert SetupInfo.from_dict({}).pumps == {}
