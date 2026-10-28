"""The virtual ESP32: simulated temperature, digital I/O and sensors.

It stands in for the uc2rest client on the virtual microscope, so UC2Config,
Temperature and I2CSensor work without hardware. These tests pin the model and
the uc2rest-shaped return values the controllers read.
"""
import json
import math
import types

import pytest

from imswitch.imcontrol.model.managers.UC2ConfigManager import UC2ConfigManager
from imswitch.imcontrol.model.managers.rs232.VirtualESP32Manager import (
    VirtualESP32Manager, VirtualUC2Client,
)


class Clock:
    def __init__(self):
        self.t = 1000.0

    def __call__(self):
        return self.t


@pytest.fixture
def client():
    clock = Clock()
    c = VirtualUC2Client({"ambientC": 20.0, "heaterTauS": 50.0, "heaterMaxRiseC": 25.0,
                          "noiseC": 0.0, "digitalIn": {"1": {"mirrorOut": 2},
                                                       "3": {"schedule": [[10, 1], [20, 0]]}}},
                         clock=clock)
    c.test_clock = clock
    return c


def test_heater_follows_a_first_order_lag(client):
    t = client.temperature
    assert t.get_temperature() == pytest.approx(20.0)
    t.set_temperature(active=1, Kp=100, Ki=0.1, Kd=0.5, target=37)
    client.test_clock.t += 50.0                       # one time constant
    assert t.get_temperature() == pytest.approx(37 - 17 * math.exp(-1), abs=1e-3)
    client.test_clock.t += 1000.0
    assert t.get_temperature() == pytest.approx(37.0, abs=1e-3)
    t.stop_heating()
    client.test_clock.t += 1000.0
    assert t.get_temperature() == pytest.approx(20.0, abs=1e-3)


def test_heater_is_limited_by_its_power(client):
    client.temperature.set_temperature(active=1, target=80)
    client.test_clock.t += 10_000.0
    assert client.temperature.get_temperature() == pytest.approx(45.0, abs=1e-3)


def test_digital_outputs_latch_and_pulse(client):
    out = client.digitalout
    assert out.set_digitalout(digitaloutid=2, digitaloutval=1)["digitalout"]["digitaloutval"] == 1
    assert out.get_digitalout(2)["digitalout"]["digitaloutval"] == 1
    out.set_digitalout(digitaloutid=2, digitaloutval=-1)
    assert out.pulses[2] == 1 and out.values[2] == 0
    with pytest.raises(ValueError):
        out.set_digitalout(digitaloutid=7, digitaloutval=1)


def test_digital_inputs_loopback_schedule_and_callbacks(client):
    din, seen = client.digitalin, []
    din.register_callback(0, seen.append)
    din.act_digitalin()                                   # establish the baseline
    client.digitalout.set_digitalout(digitaloutid=2, digitaloutval=1)
    assert din.get_digitalin(1)["digitalin"]["digitalinval"] == 1   # mirrors output 2
    assert seen and seen[-1][0] == 1
    assert din.get_digitalin(3)["digitalin"]["digitalinval"] == 0
    client.test_clock.t += 15
    assert din.get_digitalin(3)["digitalin"]["digitalinval"] == 1
    client.test_clock.t += 10
    assert din.get_digitalin(3)["digitalin"]["digitalinval"] == 0


def test_simulation_messages_through_write_serial(client):
    reply = client.serial.writeSerial(json.dumps(
        {"task": "/digitalin_sim", "digitalinid": 2, "digitalinval": 1}))
    assert reply["digitalin"] == {"digitalinid": 2, "digitalinval": 1}
    events = []
    client.state.register_emergency_callback(events.append)
    client.serial.writeSerial({"task": "/estop_sim", "active": 1})
    assert client.state.is_emergency_active() and client.state.get_power() == 0
    assert events[-1]["active"] == 1
    assert "error" in client.serial.writeSerial({"task": "/motor_act"})


def test_sensors_follow_the_model(client):
    client.temperature.set_temperature(active=1, target=30)
    client.test_clock.t += 10_000
    assert client.i2c.read_sht45()["temperature_c"] == pytest.approx(30.0, abs=0.2)
    assert client.i2c.read_tsl2591()["lux"] > 0
    assert client.fan.set_mode("manual", wiper=127)["fan"]["wiper"] == 127
    assert client.fan.get_fan()["rpm"] == 3000


def test_unmodelled_setters_refuse(client):
    assert client.gpio.get_status() == {}
    with pytest.raises(NotImplementedError):
        client.gpio.set_threshold(5)


def test_uc2config_manager_uses_it():
    info = types.SimpleNamespace(managerProperties={"ambientC": 21.0, "noiseC": 0.0})
    esp = VirtualESP32Manager(info, "ESP32")
    manager = UC2ConfigManager(None, {"rs232sManager": {"ESP32": esp}})
    assert manager.isConnected() is True
    assert manager._digitalOut is esp._esp32.digitalout
    assert manager.getFirmwareInfo().get("name", "UC2_Virtual") == "UC2_Virtual"
    esp.finalize()
    assert manager.isConnected() is False


def test_ids_from_query_strings_are_accepted(client):
    # UC2ConfigController's GPIO endpoints are untyped: ids arrive as strings.
    client.digitalout.set_digitalout(digitaloutid="2", digitaloutval="1")
    assert client.digitalout.get_digitalout("2")["digitalout"]["digitaloutval"] == 1
    assert client.digitalin.get_digitalin("1")["digitalin"]["digitalinval"] == 1
