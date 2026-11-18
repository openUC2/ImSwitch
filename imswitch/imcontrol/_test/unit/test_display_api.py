"""Status OLED on the CAN-HAT master: device name lookup, throttled status line, API.

No hardware: the ESP32 is the virtual one (its ``display`` mirrors uc2rest's Display).
"""
import time
import types

import pytest

from imswitch.imcontrol.controller.controllers.uc2config import display_api
from imswitch.imcontrol.controller.controllers.uc2config.display_api import (
    DisplayApiMixin, resolve_device_name,
)
from imswitch.imcontrol.model.managers.UC2ConfigManager import UC2ConfigManager
from imswitch.imcontrol.model.managers.rs232.VirtualESP32Manager import VirtualESP32Manager


def _boom():
    raise OSError("no such file")


# -- device name -------------------------------------------------------------

def test_hostname_file_is_the_device_name():
    assert resolve_device_name({}, lambda: "openuc2-metal-slope-23501", lambda: "3f9a1c2b7d4e") \
        == "openuc2-metal-slope-23501"


def test_env_overrides_everything():
    assert resolve_device_name({"UC2_DEVICE_NAME": "my-frame"}, lambda: "openuc2-x", lambda: "y") == "my-frame"


def test_falls_back_to_the_local_hostname_without_the_file():
    assert resolve_device_name({}, _boom, lambda: "openuc2-lab") == "openuc2-lab"


def test_container_id_and_localhost_are_no_names():
    assert resolve_device_name({}, _boom, lambda: "3f9a1c2b7d4e") is None
    assert resolve_device_name({}, lambda: "localhost", lambda: "localhost") is None
    assert resolve_device_name({}, lambda: "  ", _boom) is None


def test_hostname_file_is_read_with_its_newline(tmp_path):
    hostname = tmp_path / "hostname"
    hostname.write_text("openuc2-metal-slope-23501\n")
    assert resolve_device_name({}, lambda: display_api._file_hostname(str(hostname)), _boom) \
        == "openuc2-metal-slope-23501"
    assert resolve_device_name({}, lambda: display_api._file_hostname(str(tmp_path / "missing")),
                               lambda: "openuc2-lab") == "openuc2-lab"


def test_name_is_cut_like_the_hotspot_ssid():
    # {hostname-32} in the pallet's hotspot template
    assert resolve_device_name({}, lambda: "openuc2-" + "x" * 40, _boom) == ("openuc2-" + "x" * 40)[:32]


# -- controller part (mixin on a stub controller) ------------------------------

class _Channel:
    def __init__(self):
        self.handlers = []
        self.sigDisplayStatus = types.SimpleNamespace(connect=self.handlers.append, emit=self._emit)

    def _emit(self, *args):
        for handler in self.handlers:
            handler(*args)


class _Controller(DisplayApiMixin):
    """Just the attributes the mixin reads from UC2ConfigController."""

    def __init__(self, esp=None):
        self._commChannel = _Channel()
        self._logger = types.SimpleNamespace(error=print, debug=print, warning=print)
        self.esp = esp
        # without an ESP the manager is created without its ``ESP32`` attribute
        lowLevel = {"rs232sManager": {"ESP32": esp}} if esp is not None else {}
        self._master = types.SimpleNamespace(UC2ConfigManager=UC2ConfigManager(None, lowLevel))
        self._pushDeviceNameSoon = lambda: None  # no background retry thread in unit tests
        self._init_display()


@pytest.fixture
def controller(monkeypatch):
    monkeypatch.setattr(display_api, "DISPLAY_MIN_INTERVAL_S", 0.05)
    monkeypatch.setattr(display_api, "resolve_device_name", lambda *a, **k: "openuc2-metal-slope-23501")
    info = types.SimpleNamespace(managerProperties={"ambientC": 21.0, "noiseC": 0.0})
    esp = VirtualESP32Manager(info, "ESP32")
    c = _Controller(esp)
    yield c
    esp.finalize()


def _screen(controller):
    return controller.esp._esp32.display.state


def test_device_name_is_pushed_to_the_display(controller):
    assert controller.updateDisplayDeviceName() == {"status": "success", "name": "openuc2-metal-slope-23501"}
    assert _screen(controller)["name"] == "openuc2-metal-slope-23501"
    assert controller.getDeviceName() == {"name": "openuc2-metal-slope-23501"}


def test_firmware_without_display_module_is_reported(controller):
    controller.esp._esp32.display.get_display = lambda timeout=1: {}
    result = controller.updateDisplayDeviceName()
    assert result["status"] == "unsupported"
    # nothing is sent to a firmware that would answer "unknown task"
    controller._commChannel.sigDisplayStatus.emit("Scanning", 5)
    assert controller._displayTimer is None


def _wait_for(condition, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.01)
    return False


def test_status_bursts_are_coalesced_to_the_latest_value(controller):
    sent = []
    real = controller.esp._esp32.display.set_status
    controller.esp._esp32.display.set_status = lambda s, progress=None: (sent.append((s, progress)), real(s, progress))

    for step in range(20):
        controller._commChannel.sigDisplayStatus.emit(f"Acquire frame {step}", step)

    assert _wait_for(lambda: _screen(controller)["status"] == "Acquire frame 19")
    assert _screen(controller)["progress"] == 19
    # the first step goes out at once, the rest of the burst collapses into one trailing update
    assert 1 <= len(sent) <= 3 and sent[-1] == ("Acquire frame 19", 19)


def test_api_switches(controller):
    assert controller.getDisplayState()["on"] == 1
    assert controller.setDisplayEnabled(False)["status"] == "success"
    assert _screen(controller)["on"] == 0
    controller.setDisplayEnabled(True)
    controller.setDisplayFlip(True)
    controller.setDisplayName("frame-1")
    assert (_screen(controller)["on"], _screen(controller)["flip"], _screen(controller)["name"]) == (1, 1, "frame-1")
    # setDisplayStatus goes through the same throttled signal as the experiment steps
    assert controller.setDisplayStatus("Hello", 40) == {"status": "success"}
    assert _wait_for(lambda: (_screen(controller)["status"], _screen(controller)["progress"]) == ("Hello", 40))


def test_no_esp_means_errors_not_exceptions(monkeypatch):
    monkeypatch.setattr(display_api, "resolve_device_name", lambda *a, **k: "x")
    c = _Controller()
    assert c.getDisplayState() == {}
    assert c.setDisplayEnabled(False)["status"] == "error"
    assert c.updateDisplayDeviceName()["status"] in ("error", "unsupported")
