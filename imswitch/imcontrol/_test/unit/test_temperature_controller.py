"""TemperatureController over the virtual UC2 board: typed setpoint and on/off."""
from types import SimpleNamespace

import pytest

from imswitch.imcontrol.controller.controllers.TemperatureController import TemperatureController
from imswitch.imcontrol.model.SetupInfo import RS232Info
from imswitch.imcontrol.model.managers.rs232.VirtualESP32Manager import VirtualESP32Manager


@pytest.fixture
def controller():
    board = VirtualESP32Manager(RS232Info(managerName="VirtualESP32Manager",
                                          managerProperties={"noiseC": 0.0}), "ESP32")
    c = TemperatureController.__new__(TemperatureController)   # skip the widget base
    c._master = SimpleNamespace(rs232sManager={"ESP32": board})
    c.Kp, c.Ki, c.Kd, c.controlTarget = 100, 0.1, 0.5, 0.0
    c.temperatureController = board._esp32.temperature
    yield c
    board.finalize()


def test_setpoint_and_switching(controller):
    state = controller.set_temperature(active=True, target=37.0, sensorOffset=2.0)
    assert state["target_c"] == 37.0 and state["temperature_c"] is not None
    heater = controller.temperatureController
    assert heater.target_c == 39.0 and heater.active
    controller.set_temperature(active=False)
    assert not heater.active and heater.target_c == 40.0     # keeps the setpoint, default offset


def test_http_parameters_are_typed():
    import inspect
    sig = inspect.signature(TemperatureController.set_temperature)
    assert sig.parameters["active"].annotation in (bool, "bool")
    assert "float" in str(sig.parameters["target"].annotation)
