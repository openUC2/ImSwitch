"""Prompted firmware update (uc2config/firmware_update.py), without hardware.

The controller is built without __init__; the bus scan, the USB identity, the
firmware server and the flashing primitives are fakes, so these tests pin the
orchestration rules: which image a board gets, when an update is refused, the
order and stop-at-first-failure, and that "done" means "re-scan shows it".
"""
import hashlib
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from imswitch.imcontrol.controller.controllers.UC2ConfigController import UC2ConfigController
from imswitch.imcontrol.controller.controllers.uc2config import firmware_update

OLD = "UC2-ESP v2.0"
NEW = "v2026.0.0-beta.4-7-gbd08544-t20260930125929"
MASTER_IMG = "esp32_UC2_canopen_master_release.bin"
MOTOR_Y_IMG = "esp32_UC2_canopen_slave_motor_release_motY.bin"
LED_IMG = "esp32_UC2_canopen_slave_led_release.bin"
MANIFEST = {"version": NEW, "files": {n: {"size": 3, "sha256": hashlib.sha256(b"abc").hexdigest()}
                                      for n in (MASTER_IMG, MOTOR_Y_IMG, LED_IMG)}}


def usb_info(version=OLD, image=None):
    return {"pindef": "UC2_canopen_master", "isMaster": True, "connected": True,
            "fwVersion": version, "fwImage": image, "serialport": "/dev/ttyUSB0"}


def bus(*nodes, master_version=OLD):
    return {"master": {"canId": 1, "fwVersion": master_version}, "scan": list(nodes)}


def node(can_id, kind, version=OLD, image=None, reachable=True):
    return {"canId": can_id, "deviceTypeStr": kind, "fwVersion": version, "fwImage": image,
            "statusStr": "idle" if reachable else "unreachable"}


class Lasers:
    def __init__(self):
        self.enabled = {"488": True}

    def getAllDeviceNames(self):
        return list(self.enabled)

    def __getitem__(self, name):
        return SimpleNamespace(setEnabled=lambda on: self.enabled.__setitem__(name, on))


def make_controller(scan, usb=None, manifest=MANIFEST, controllers=None):
    c = object.__new__(UC2ConfigController)
    c._UC2ConfigController__logger = c._logger = logging.getLogger("test")
    c._firmware_server_url = "http://fw/firmware"
    c._can_ota_lock, c._usb_flash_lock = threading.Lock(), threading.Lock()
    c._usb_flash_cancel_event = threading.Event()
    c._fw_update_state = {"state": "idle", "steps": []}
    c._fw_update_cancel = threading.Event()
    c._fw_prompt = None
    c._fetch_firmware_manifest = lambda: manifest
    c.getFirmwareInfo = lambda: usb if usb is not None else usb_info()
    c.scan_canbus = lambda timeout=5, probe_range=False: scan
    c._get_can_id_firmware_mapping = lambda: {1: MASTER_IMG, 12: MOTOR_Y_IMG, 30: LED_IMG}
    registry = controllers or {}
    c.lasers = Lasers()
    c._master = SimpleNamespace(getController=registry.get, lasersManager=c.lasers,
                                positionersManager=None,
                                UC2ConfigManager=SimpleNamespace(ESP32=SimpleNamespace(
                                    canota=SimpleNamespace(cancel_streaming_ota=lambda: None))))
    return c


def run_update(c, **kwargs):
    started = c.startFirmwareUpdate(**kwargs)
    if started["status"] == "started":
        c._fw_update_thread.join(timeout=5)
    return started, c.getFirmwareUpdateStatus()


# ── which image a board gets ────────────────────────────────────────────────

def test_reported_image_wins_over_the_can_id_table():
    # Node 12 reports it was built as the axis-Y image; the table would agree
    # here, but node 14 (no table entry) is only resolvable via its report.
    scan = bus(node(12, "motor", image=MOTOR_Y_IMG), node(14, "motor", image=MOTOR_Y_IMG),
               node(30, "led"))
    devices = {d["canId"]: d for d in make_controller(scan).checkFirmwareUpdates()["devices"]}
    assert devices[14]["filename"] == MOTOR_Y_IMG and devices[14]["image_source"] == "reported"
    assert devices[30]["filename"] == LED_IMG and devices[30]["image_source"] == "mapping"
    assert devices[1]["filename"] == MASTER_IMG  # USB master via its pindef


# ── download check ──────────────────────────────────────────────────────────

def test_download_must_match_version_json(tmp_path):
    c = make_controller(bus())
    good = tmp_path / MOTOR_Y_IMG
    good.write_bytes(b"abc")
    assert c._verify_firmware_download(good, MOTOR_Y_IMG, MANIFEST)
    bad = tmp_path / LED_IMG
    bad.write_bytes(b"abd")
    assert not c._verify_firmware_download(bad, LED_IMG, MANIFEST)
    assert not bad.exists()  # never flash a corrupted image
    unlisted = tmp_path / "custom.bin"
    unlisted.write_bytes(b"x")
    assert c._verify_firmware_download(unlisted, "custom.bin", MANIFEST)


# ── "done" means the re-scan shows the new version ─────────────────────────

def test_wait_for_node_version_ignores_the_old_version_until_reboot(monkeypatch):
    monkeypatch.setattr(firmware_update, "VERIFY_SETTLE_S", 0)
    monkeypatch.setattr(time, "sleep", lambda s: None)
    answers = iter([bus(node(12, "motor", OLD)), bus(node(12, "motor", reachable=False)),
                    bus(node(12, "motor", NEW, MOTOR_Y_IMG))])
    c = make_controller(None)
    c.scan_canbus = lambda timeout=5: next(answers)
    assert c._wait_for_node_version(12, NEW) == (True, NEW)


def test_wait_for_node_version_reports_what_the_node_runs(monkeypatch):
    monkeypatch.setattr(firmware_update, "VERIFY_SETTLE_S", 0)
    monkeypatch.setattr(firmware_update, "VERIFY_TIMEOUT_S", 0.05)
    c = make_controller(bus(node(12, "motor", OLD)))
    assert c._wait_for_node_version(12, NEW, timeout=0.05) == (False, OLD)


# ── refusals ────────────────────────────────────────────────────────────────

def test_refused_while_an_experiment_runs():
    experiment = SimpleNamespace(getExperimentStatus=lambda: {"status": "running"})
    c = make_controller(bus(node(12, "motor")), controllers={"Experiment": experiment})
    result = c.startFirmwareUpdate()
    assert result["status"] == "refused"
    assert "An experiment is running." in result["reasons"]


def test_refused_while_a_stage_moves():
    motor = SimpleNamespace(isBusy=lambda steps, timeout=1: True)
    positioners = {"ESP32Stage": SimpleNamespace(_motor=motor)}
    c = make_controller(bus(node(12, "motor")))
    c._master.positionersManager = type("PM", (), {
        "getAllDeviceNames": lambda self: list(positioners),
        "__getitem__": lambda self, k: positioners[k]})()
    assert "A stage is moving or homing." in c.startFirmwareUpdate()["reasons"]


def test_refused_without_version_json():
    c = make_controller(bus(node(12, "motor")), manifest={})
    assert c.startFirmwareUpdate()["status"] == "refused"


def test_refused_for_a_node_that_is_not_on_the_bus():
    c = make_controller(bus(node(12, "motor"), node(20, "laser", reachable=False)))
    result = c.startFirmwareUpdate(can_ids=[12, 20])
    assert result["status"] == "refused"
    assert "CAN node 20 is not on the bus." in result["reasons"]
    assert not c._can_ota_lock.locked()


# ── the run ─────────────────────────────────────────────────────────────────

def test_updates_outdated_nodes_then_the_master_last():
    scan = bus(node(12, "motor", image=MOTOR_Y_IMG), node(30, "led", NEW, LED_IMG))
    c = make_controller(scan)
    order = []
    ok = {"status": "success", "message": "ok"}
    c._can_streaming_ota = lambda can_id, filename, expected_version: (
        order.append((can_id, filename, expected_version)) or ok)
    c._do_flash = lambda **kw: (
        order.append(("usb", kw["firmware_filename"], kw["port"])) or {"status": "success"})
    c.getFirmwareInfo = lambda: usb_info(NEW if len(order) == 2 else OLD, MASTER_IMG)

    started, status = run_update(c, include_master=True)

    assert started["status"] == "started"
    # node 30 is already up to date -> only 12, then the USB master
    assert order == [(12, MOTOR_Y_IMG, NEW), ("usb", MASTER_IMG, "/dev/ttyUSB0")]
    assert status["state"] == "success"
    assert [s["status"] for s in status["steps"]] == ["done", "done"]
    assert status["homing_required"] is True  # motor 12 rebooted
    assert c.lasers.enabled == {"488": False}
    assert not c._can_ota_lock.locked() and not c._usb_flash_lock.locked()


def test_stops_at_the_first_failure():
    scan = bus(node(12, "motor", image=MOTOR_Y_IMG), node(30, "led", image=LED_IMG))
    c = make_controller(scan)
    tried = []
    c._can_streaming_ota = lambda can_id, filename, expected_version: (
        tried.append(can_id) or {"status": "error", "message": "size_write_failed"})
    c._do_flash = lambda **kw: pytest.fail("the master must not be flashed after a failure")

    _, status = run_update(c, include_master=True)

    assert tried == [12]
    assert status["state"] == "failed"
    assert [s["status"] for s in status["steps"]] == ["failed", "skipped", "skipped"]
    assert status["homing_required"] is False
    assert not c._can_ota_lock.locked()


def test_master_counts_as_done_only_with_the_new_version():
    c = make_controller(bus())
    c._do_flash = lambda **kw: {"status": "success"}
    c.getFirmwareInfo = lambda: usb_info(OLD, MASTER_IMG)  # still the old firmware afterwards
    original_sleep = time.sleep
    time.sleep = lambda s: None
    try:
        _, status = run_update(c, can_ids=[], include_master=True)
    finally:
        time.sleep = original_sleep
    assert status["state"] == "failed"
    assert "reports" in status["steps"][0]["message"]


def test_startup_check_offers_the_update():
    c = make_controller(bus(node(12, "motor", image=MOTOR_Y_IMG)))
    c._master.UC2ConfigManager.isConnected = lambda: True
    emitted = []
    c.sigFirmwareUpdatesAvailable = SimpleNamespace(emit=emitted.append)
    c._startup_firmware_check()
    assert emitted and emitted[0]["updates_available"] == 2  # master + node 12
    assert c.getFirmwareUpdatePrompt()["updates_available"] == 2
