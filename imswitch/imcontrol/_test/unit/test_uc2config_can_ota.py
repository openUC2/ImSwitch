"""CAN streaming OTA orchestration in UC2ConfigController (no hardware).

Covers: intermediate failures are not reported as the terminal "error" (the
wizard counted them as failed devices), one upload at a time on the serial
port, and the streaming baud defaulting to the live link's baud.
"""
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

from imswitch.imcontrol.controller.controllers.UC2ConfigController import UC2ConfigController


class FakeCanOta:
    """Stands in for uc2rest's canota: each attempt reports a failure through
    status_callback (as uc2rest does) unless its outcome is True."""

    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.bauds = []
        self._cancel_event = threading.Event()

    def start_can_streaming_ota_blocking(self, can_id, firmware_path, progress_callback,
                                         status_callback, baud):
        self.bauds.append(baud)
        ok = self.outcomes.pop(0)
        if not ok:
            status_callback("Master reported error at chunk #1: size_write_failed", False)
        return ok


def make_controller(canota, link_baud=115200):
    c = object.__new__(UC2ConfigController)
    c._UC2ConfigController__logger = c._logger = logging.getLogger("test")
    c._firmware_server_url = ""  # no version.json -> transfer result is not re-verified
    c._ota_status = {}
    c._can_ota_lock = threading.Lock()
    c._usb_flash_lock = threading.Lock()
    c._download_firmware_for_device = lambda can_id: Path("/tmp/fw.bin")
    esp32 = SimpleNamespace(canota=canota, serial=SimpleNamespace(baudrate=link_baud))
    c._master = SimpleNamespace(UC2ConfigManager=SimpleNamespace(ESP32=esp32))
    c.emitted = []
    c.sigOTAStatusUpdate = SimpleNamespace(emit=c.emitted.append)
    return c


@pytest.fixture(autouse=True)
def no_retry_sleep(monkeypatch):
    monkeypatch.setattr(time, "sleep", lambda s: None)


def statuses(c):
    return [e["status"] for e in c.emitted]


def test_failed_attempt_then_success_is_never_reported_as_error():
    c = make_controller(FakeCanOta([False, True]))
    assert c.startCANStreamingOTA(can_id=11)["status"] == "success"
    s = statuses(c)
    assert "error" not in s
    assert "attempt_failed" in s and "retrying" in s
    assert s[-1] == "success"


def test_all_attempts_failing_reports_one_final_error():
    c = make_controller(FakeCanOta([False, False, False]))
    assert c.startCANStreamingOTA(can_id=11)["status"] == "error"
    s = statuses(c)
    assert s.count("error") == 1 and s[-1] == "error"
    assert not c._can_ota_lock.locked()


def test_baud_defaults_to_the_live_link():
    canota = FakeCanOta([True])
    make_controller(canota, link_baud=115200).startCANStreamingOTA(can_id=11)
    assert canota.bauds == [115200]


def test_second_upload_is_refused_while_one_runs():
    c = make_controller(FakeCanOta([True]))
    c._can_ota_lock.acquire()
    assert c.startCANStreamingOTA(can_id=11)["status"] == "busy"
    assert c.startMultipleCANStreamingOTA([11, 12])["status"] == "busy"
    assert c.flashMasterFirmwareUSB()["status"] == "busy"
    assert c.emitted == []


def test_can_upload_is_refused_during_usb_flash():
    c = make_controller(FakeCanOta([True]))
    c._usb_flash_lock.acquire()
    assert c.startCANStreamingOTA(can_id=11)["status"] == "busy"


def test_multiple_devices_run_under_one_lock():
    canota = FakeCanOta([True, True])
    c = make_controller(canota)
    result = c.startMultipleCANStreamingOTA([11, 12], delay_between=0)
    assert [r["result"]["status"] for r in result["results"]] == ["success", "success"]
    assert not c._can_ota_lock.locked()
