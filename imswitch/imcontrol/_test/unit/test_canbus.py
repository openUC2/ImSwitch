"""imswitch.imcontrol.model.canbus without hardware or network.

The firmware server is a fake requests.get, the master a fake uc2rest client.
These pin the rules the update relies on: which image a board gets, that
downloads are verified, that a failed attempt is not a terminal error, that
"done" means the rebooted board reports the new version, and that one writer
at a time uses the master's serial port.
"""
import hashlib
import io
import json
import logging
import threading
import time
from types import SimpleNamespace

import pytest

from imswitch.imcontrol.model.canbus import (CanNetwork, FirmwareServer, UpdateHooks,
                                             device_mapping, update_status)
from imswitch.imcontrol.model.canbus import bus as bus_module
from imswitch.imcontrol.model.canbus import firmware_server as server_module

OLD = "UC2-ESP v2.0"
NEW = "v2026.0.0-beta.4-7-gbd08544-t20260930125929"
MASTER = "esp32_UC2_canopen_master_release.bin"
MOTOR_Y = "esp32_UC2_canopen_slave_motor_release_motY.bin"
LED = "esp32_UC2_canopen_slave_led_release.bin"
IMAGE = b"\xe9" + bytes(99)
URL = "http://fw/firmware"


class FakeServer:
    """requests.get for the firmware server: listing, version.json, files."""

    def __init__(self, files=None, version=NEW, manifest=True):
        self.files = dict(files or {MASTER: IMAGE, MOTOR_Y: IMAGE, LED: IMAGE})
        self.manifest = {"version": version, "commit_time": "2026-09-30T12:59:29Z", "files": {
            n: {"size": len(b), "sha256": hashlib.sha256(b).hexdigest()}
            for n, b in self.files.items()}} if manifest else None
        self.downloads = []
        self.down = False

    def get(self, url, headers=None, timeout=None, stream=False):
        if self.down:
            raise server_module.requests.exceptions.ConnectionError("server down")
        path = url[len(URL):].strip("/")
        if path == "":
            body = [{"name": n, "size": len(b), "mod_time": "t"} for n, b in self.files.items()]
        elif path == "version.json" and self.manifest is not None:
            body = self.manifest
        elif path in self.files:
            self.downloads.append(path)
            return Response(raw=self.files[path])
        else:
            return Response(status=404)
        return Response(body=body)


class Response:
    def __init__(self, body=None, raw=b"", status=200):
        self._body, self.raw, self.status = body, io.BytesIO(raw), status

    def json(self):
        return json.loads(json.dumps(self._body))

    def raise_for_status(self):
        if self.status != 200:
            raise server_module.requests.exceptions.HTTPError(str(self.status))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeCanOta:
    """uc2rest canota: an attempt reports failure via status_callback unless its outcome is True."""

    def __init__(self, outcomes):
        self.outcomes, self.bauds = list(outcomes), []
        self._cancel_event = threading.Event()

    def start_can_streaming_ota_blocking(self, can_id, firmware_path, progress_callback,
                                         status_callback, baud):
        self.bauds.append(baud)
        ok = self.outcomes.pop(0)
        if ok:
            progress_callback(1, 1, 100, 20.0)
        else:
            status_callback("Master reported error at chunk #1: size_write_failed", False)
        return ok

    def cancel_streaming_ota(self):
        self._cancel_event.set()


def scan_result(*nodes, master_version=OLD, master_image=None):
    return {"master": {"canId": 1, "fwVersion": master_version, "fwImage": master_image},
            "scan": list(nodes)}


def node(can_id, kind, version=OLD, image=None, reachable=True):
    return {"canId": can_id, "deviceTypeStr": kind, "fwVersion": version, "fwImage": image,
            "statusStr": "idle" if reachable else "unreachable"}


class FakeCan:
    def __init__(self, *scans):
        self.scans = list(scans)

    def scan(self, timeout=5, probe_range=False):
        return self.scans.pop(0) if len(self.scans) > 1 else self.scans[0]


class Link:
    def __init__(self):
        self.calls = []

    def release(self):
        self.calls.append("release")

    def restore(self):
        self.calls.append("restore")

    def current_port(self):
        return "/dev/ttyUSB0"


@pytest.fixture
def server(monkeypatch):
    fake = FakeServer()
    monkeypatch.setattr(server_module.requests, "get", fake.get)
    return fake


@pytest.fixture(autouse=True)
def fast(monkeypatch):
    real_sleep = time.sleep
    monkeypatch.setattr(time, "sleep", lambda s: real_sleep(0) if s else None)


def network(tmp_path, *scans, outcomes=(True,), usb=None, hooks=None, baud=115200):
    client = SimpleNamespace(can=FakeCan(*(scans or [scan_result()])), canota=FakeCanOta(outcomes),
                             serial=SimpleNamespace(baudrate=baud))
    emitted = []
    hooks = hooks or UpdateHooks(usb_info=lambda: usb if usb is not None else {
        "connected": True, "pindef": "UC2_canopen_master", "isMaster": True,
        "fwVersion": OLD, "serialport": "/dev/ttyUSB0"})
    net = CanNetwork(lambda: client, URL, tmp_path, Link(), hooks, emit_ota=emitted.append,
                     logger=logging.getLogger("test"))
    return net, client, emitted


# ── versions and images ─────────────────────────────────────────────────────

@pytest.mark.parametrize("installed, available, expected", [
    (NEW, NEW, "up_to_date"),
    (OLD, NEW, "update_available"),          # firmware before version reporting
    (None, NEW, "update_available"),
    ("v2026.0.0-beta.4-5-gaaaaaaa-t20260901000000", NEW, "update_available"),
    ("v2026.0.0-beta.4-8-gbbbbbbb-t20261001000000-dirty", NEW, "device_newer"),
    ("v2026.1.0", NEW, "update_available"),  # tag build: no timestamp to order by
    (NEW, None, "unknown"),                  # server has no version.json
])
def test_update_status(installed, available, expected):
    assert update_status(installed, available) == expected


def test_device_mapping_comes_from_the_one_can_table():
    mapping = device_mapping()
    assert mapping["master"] == 1
    assert mapping["motors"]["Y"] == 12 and mapping["laser"]["laser_0"] == 20


# ── firmware server ─────────────────────────────────────────────────────────

def test_files_carry_version_and_sha256(server, tmp_path):
    files = FirmwareServer(URL, tmp_path).files()
    assert files["server_version"] == NEW
    motor = next(f for f in files["files"] if f["filename"] == MOTOR_Y)
    assert motor["version"] == NEW and motor["sha256"] == hashlib.sha256(IMAGE).hexdigest()
    by_id = FirmwareServer(URL, tmp_path).images_by_can_id([12])["firmware"]
    assert by_id[12]["filename"] == MOTOR_Y


def test_download_is_verified_and_reused_from_cache(server, tmp_path):
    fw = FirmwareServer(URL, tmp_path)
    assert fw.download(MOTOR_Y).read_bytes() == IMAGE
    assert fw.download(MOTOR_Y) is not None
    assert server.downloads == [MOTOR_Y]  # second time: cached copy matched version.json
    server.files[LED] = IMAGE[:-1] + b"\x01"  # served bytes no longer match version.json
    assert fw.download(LED) is None
    assert not (tmp_path / LED).exists()     # never flash a corrupted image


def test_image_choice(server, tmp_path):
    fw = FirmwareServer(URL, tmp_path)
    assert fw.can_image(12) == MOTOR_Y
    assert fw.can_image(13) is None            # motZ is not on this server
    server.files["id_12_custom.bin"] = IMAGE
    assert fw.can_image(12) == "id_12_custom.bin"
    assert fw.master_image() == MASTER         # the CANopen master before the old HAT names


# ── CAN OTA ─────────────────────────────────────────────────────────────────

def test_failed_attempt_then_success_is_not_an_error(server, tmp_path, monkeypatch):
    net, client, emitted = network(tmp_path, scan_result(node(12, "motor", NEW, MOTOR_Y)),
                                   outcomes=(False, True))
    monkeypatch.setattr(net.bus, "wait_for_version", lambda *a, **k: (True, NEW))
    result = net.ota.start(12)
    statuses = [e["status"] for e in emitted]
    assert result["status"] == "success"
    assert "error" not in statuses
    assert {"attempt_failed", "retrying", "verifying"} <= set(statuses)
    assert statuses[-1] == "success"
    assert client.canota.bauds == [115200, 115200]  # the link's baud, not a fixed 921600


def test_flashed_but_old_version_is_an_error(server, tmp_path, monkeypatch):
    net, _, emitted = network(tmp_path)
    monkeypatch.setattr(net.bus, "wait_for_version", lambda *a, **k: (False, OLD))
    result = net.ota.start(12)
    assert result["status"] == "error" and "reports" in result["message"]
    assert [e["status"] for e in emitted].count("error") == 1


def test_wait_for_version_ignores_the_old_version_until_reboot(tmp_path):
    answers = [scan_result(node(12, "motor", OLD)), scan_result(node(12, "motor", reachable=False)),
               scan_result(node(12, "motor", NEW, MOTOR_Y))]
    net, client, _ = network(tmp_path, *answers)
    assert net.bus.wait_for_version(12, NEW, settle=0) == (True, NEW)


def test_one_writer_on_the_serial_port(server, tmp_path):
    net, _, _ = network(tmp_path)
    net.guard.usb.acquire()
    assert net.ota.start(12)["status"] == "busy"
    net.guard.usb.release()
    net.guard.can.acquire()
    assert net.usb.flash()["status"] == "busy"
    assert net.ota.start_many([12])["status"] == "busy"


def test_merged_image_with_wrong_boot_address_is_rejected(tmp_path):
    net, _, _ = network(tmp_path)
    bad = tmp_path / "x_merged.bin"
    bad.write_bytes(b"\xff" * 0x1000 + b"\xe9\x00\x02\x20")
    assert not net.usb.validate(bad, is_merged=True, chip="esp32s3")["valid"]
    # 0x1000 is the classic ESP32's boot address
    assert net.usb.validate(bad, is_merged=True, chip="esp32")["valid"]


# ── check and update ────────────────────────────────────────────────────────

def test_check_prefers_the_reported_image(server, tmp_path):
    net, _, _ = network(tmp_path, scan_result(node(14, "motor", image=MOTOR_Y), node(30, "led"),
                                              node(20, "laser", reachable=False)))
    devices = {d["canId"]: d for d in net.updater.check()["devices"]}
    assert devices[14]["filename"] == MOTOR_Y and devices[14]["image_source"] == "reported"
    assert devices[30]["filename"] == LED and devices[30]["image_source"] == "mapping"
    assert devices[20]["update_status"] == "unreachable"
    assert devices[1]["filename"] == MASTER and devices[1]["connection"] == "usb"


def test_standalone_board_is_not_can_scanned(server, tmp_path):
    usb = {"connected": True, "pindef": "UC2_3", "isMaster": False, "fwVersion": OLD}
    net, client, _ = network(tmp_path, usb=usb)
    client.can = None  # a scan would fail loudly
    [board] = net.updater.check()["devices"]
    assert board["filename"] == "esp32_UC2_3_release.bin"


def test_update_refusals(server, tmp_path):
    net, _, _ = network(tmp_path, scan_result(node(12, "motor"),
                                              node(20, "laser", reachable=False)))
    assert "CAN node 20 is not on the bus." in net.updater.start(can_ids=[12, 20])["reasons"]
    net.updater.hooks.blockers = lambda: ["An experiment is running."]
    assert net.updater.start()["reasons"] == ["An experiment is running."]
    server.down = True
    net.updater.hooks.blockers = list
    assert "not reachable" in net.updater.start(can_ids=[12])["reasons"][0]
    assert not net.guard.can.locked()


def test_without_version_json_boards_are_chosen_by_hand_and_not_verified(server, tmp_path,
                                                                        monkeypatch):
    server.manifest = None  # an older firmware server: only the file listing
    net, _, _ = network(tmp_path, scan_result(node(12, "motor"), node(13, "motor")))
    devices = {d["canId"]: d for d in net.updater.check()["devices"]}
    assert devices[12]["update_status"] == "unknown"      # image there, no version to compare
    assert devices[13]["update_status"] == "no_firmware"  # motZ is not on the server
    assert net.updater.start()["reasons"] == ["Nothing to update."]  # nothing preselected
    monkeypatch.setattr(net.ota, "upload", lambda can_id, filename, expected_version, cancel: (
        {"status": "success", "message": f"expected {expected_version}"}))
    monkeypatch.setattr(net.usb, "run", lambda **kw: {"status": "success"})
    _, state = run(net, can_ids=[12], include_master=True)
    assert state["state"] == "success"
    assert state["steps"][0]["message"] == "expected None"  # the OTA cannot verify
    assert "not verified" in state["steps"][1]["message"]


def run(net, **kwargs):
    started = net.updater.start(**kwargs)
    if started["status"] == "started":
        net.updater.thread.join(timeout=5)
    return started, net.updater.state


def test_update_runs_nodes_then_master_last(server, tmp_path, monkeypatch):
    calls = []
    usb = {"connected": True, "pindef": "UC2_canopen_master", "isMaster": True, "fwVersion": OLD,
           "fwImage": MASTER, "serialport": "/dev/ttyUSB0"}
    hooks = UpdateHooks(usb_info=lambda: usb, before_update=lambda: calls.append("lasers off"),
                        motor_restarted=lambda: calls.append("homing"))
    net, _, _ = network(tmp_path, scan_result(node(12, "motor", image=MOTOR_Y),
                                              node(30, "led", NEW, LED)), hooks=hooks)
    monkeypatch.setattr(net.ota, "upload", lambda can_id, filename, expected_version, cancel: (
        calls.append((can_id, filename)) or {"status": "success", "message": "ok"}))

    def usb_run(**kw):
        calls.append(("usb", kw["firmware_filename"], kw["port"]))
        usb["fwVersion"] = NEW
        return {"status": "success"}
    monkeypatch.setattr(net.usb, "run", usb_run)

    started, state = run(net, include_master=True)

    assert started["status"] == "started"
    # the LED is up to date, so: lasers off, node 12, homing, then the master
    assert calls == ["lasers off", (12, MOTOR_Y), "homing", ("usb", MASTER, "/dev/ttyUSB0")]
    assert state["state"] == "success" and state["homing_required"]
    assert [s["status"] for s in state["steps"]] == ["done", "done"]
    assert not net.guard.can.locked() and not net.guard.usb.locked()


def test_update_stops_at_the_first_failure(server, tmp_path, monkeypatch):
    net, _, _ = network(tmp_path, scan_result(node(12, "motor", image=MOTOR_Y), node(30, "led")))
    monkeypatch.setattr(net.ota, "upload", lambda *a, **k: {"status": "error", "message": "boom"})
    monkeypatch.setattr(net.usb, "run", lambda **kw: pytest.fail("master flashed after a failure"))
    _, state = run(net, include_master=True)
    assert state["state"] == "failed"
    assert [s["status"] for s in state["steps"]] == ["failed", "skipped", "skipped"]
    assert not state["homing_required"] and not net.guard.can.locked()


def test_master_counts_as_done_only_with_the_new_version(server, tmp_path, monkeypatch):
    net, _, _ = network(tmp_path)
    monkeypatch.setattr(net.usb, "run", lambda **kw: {"status": "success"})
    _, state = run(net, can_ids=[], include_master=True)
    assert state["state"] == "failed" and "reports" in state["steps"][0]["message"]


def test_bus_module_has_no_imswitch_dependency():
    import ast
    import inspect
    from imswitch.imcontrol.model import canbus
    for module in (bus_module, server_module, canbus.ota, canbus.usb, canbus.updater,
                   canbus.images, canbus.guard, canbus.network):
        imports = [n for n in ast.walk(ast.parse(inspect.getsource(module)))
                   if isinstance(n, (ast.Import, ast.ImportFrom))]
        names = [a.name for n in imports for a in n.names] + [n.module or "" for n in imports
                                                               if isinstance(n, ast.ImportFrom)]
        assert not any(name.startswith("imswitch") for name in names), module.__name__
