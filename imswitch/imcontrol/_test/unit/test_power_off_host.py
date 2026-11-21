"""powerOffHost: motors stop and lights go off before logind is asked to power off."""
import types

from imswitch.imcontrol.controller.controllers.UC2ConfigController import UC2ConfigController


class _Devices(dict):
    def getAllDeviceNames(self):
        return list(self)


def _fake(calls, fail=False):
    laser = types.SimpleNamespace(setEnabled=lambda on: calls.append(("laser", on)))
    ctl = types.SimpleNamespace(
        _master=types.SimpleNamespace(lasersManager=_Devices(l1=laser), LEDsManager=_Devices()),
        _logger=types.SimpleNamespace(error=lambda msg: calls.append(("error", msg))),
        _stopAllMotors=lambda: calls.append("stop"),
    )
    ctl._lightsOff = types.MethodType(UC2ConfigController._lightsOff, ctl)

    def call(method, signature, body, **kw):
        if fail:
            raise ConnectionError("no system bus")
        calls.append((method, body, kw["bus_name"]))
    ctl._callSystemd = call
    return ctl


def _power_off(ctl, reboot):
    fn = UC2ConfigController.powerOffHost
    return getattr(fn, "__wrapped__", fn)(ctl, reboot=reboot)


def test_stops_motors_and_lights_before_power_off():
    calls = []
    assert _power_off(_fake(calls), False) == {"status": "shutting down"}
    assert calls == ["stop", ("laser", False), ("PowerOff", (False,), "org.freedesktop.login1")]


def test_reboot_and_missing_bus():
    calls = []
    assert _power_off(_fake(calls), True)["status"] == "rebooting"
    assert calls[-1][0] == "Reboot"
    calls = []
    assert _power_off(_fake(calls, fail=True), False)["status"] == "error"
