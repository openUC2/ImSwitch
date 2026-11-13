"""
Parking and arbitrary-point scanning of the ESP32 galvo manager and controller.

A park or a points upload that the board did not acknowledge must come back as
an error all the way up to the UI, and the park position must survive a restart.
Driven with a fake UC2-REST ``galvo``; no board, no serial port. ``HOME`` is
pointed at a scratch folder because the park position is saved under ~/.imswitch.
"""

from __future__ import annotations

import json
import logging
import os
from types import SimpleNamespace

import pytest

from imswitch.imcontrol.controller.controllers.GalvoScannerController import (
    GalvoScannerController,
)
from imswitch.imcontrol.model.managers.galvoscanners.ESP32GalvoScannerManager import (
    ESP32GalvoScannerManager,
)

ACK_REPLIES = [
    {"return": 1, "qid": 5},           # CAN master
    [{"return": 1, "qid": 5}],         # post_json returns a list
    {"success": True},                 # standalone board
    17,                                # bare qid: sent, no reply awaited
]
NACK_REPLIES = [
    "No response received",
    "communication interrupted by timeout or reset: 5 and code:{}",
    "Failed to Send",
    [{"return": 0, "qid": 5}],         # master: no route / SDO write timed out
    {"success": False, "error": "Invalid configuration values"},
    [{"return": 1}, {"error": "node 0x15 unreachable"}],
]


class FakeGalvo:
    """Stands in for uc2rest.Galvo; records calls, replies as scripted."""

    def __init__(self):
        self.calls = []
        self.replies = {}      # method name -> reply (or Exception to raise)

    def _do(self, name, *args):
        self.calls.append((name, args))
        reply = self.replies.get(name, {"return": 1, "qid": 1})
        if isinstance(reply, Exception):
            raise reply
        return reply

    def set_position(self, x, y, timeout=1):
        return self._do("set_position", x, y)

    def stop_galvo_scan(self, timeout=1):
        return self._do("stop_galvo_scan")

    def set_arbitrary_points(self, points, laser_trigger="AUTO", timeout=1):
        return self._do("set_arbitrary_points", points, laser_trigger)

    def names(self):
        return [c[0] for c in self.calls]


@pytest.fixture(autouse=True)
def scratch_home(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    return tmp_path


def make_manager(name="galvo", **props):
    galvo = FakeGalvo()
    info = SimpleNamespace(managerProperties={"rs232device": "ESP32", **props})
    low = {"rs232sManager": {"ESP32": SimpleNamespace(_esp32=SimpleNamespace(galvo=galvo))}}
    return ESP32GalvoScannerManager(info, name, **low), galvo


POINTS = [{"x": 100, "y": 200, "dwell_us": 500}, {"x": 300, "y": 400, "dwell_us": 500}]


# --------------------------------------------------------------------------
# park
# --------------------------------------------------------------------------

@pytest.mark.parametrize("reply", ACK_REPLIES)
def test_park_acknowledged(reply):
    mgr, galvo = make_manager(park_x=1000, park_y=3000)
    galvo.replies["set_position"] = reply
    out = mgr.park()
    assert out["status"] == "parked" and (out["x"], out["y"]) == (1000, 3000)
    assert galvo.calls == [("set_position", (1000, 3000))]


@pytest.mark.parametrize("reply", NACK_REPLIES + [RuntimeError("serial closed")])
def test_park_not_acknowledged_is_an_error(reply):
    mgr, galvo = make_manager()
    mgr._is_scanning = True
    galvo.replies["set_position"] = reply
    out = mgr.park()
    assert "error" in out and "status" not in out
    assert mgr._is_scanning is True          # the board never confirmed it stopped


def test_park_without_board_is_an_error():
    mgr, _ = make_manager()
    mgr._galvo = None
    assert mgr.park() == {"error": "Galvo not initialized"}


# --------------------------------------------------------------------------
# stop + park-on-stop
# --------------------------------------------------------------------------

def test_stop_scan_parks_afterwards():
    mgr, galvo = make_manager(park_x=10, park_y=20)
    mgr._is_scanning = True
    out = mgr.stop_scan()
    assert out["status"] == "stopped" and out["park"]["status"] == "parked"
    assert galvo.names() == ["stop_galvo_scan", "set_position"]
    assert mgr._is_scanning is False


def test_stop_scan_without_park_on_stop_does_not_move_the_beam():
    mgr, galvo = make_manager(park_on_stop=False)
    assert mgr.stop_scan()["status"] == "stopped"
    assert galvo.names() == ["stop_galvo_scan"]


def test_unacknowledged_stop_is_an_error_and_does_not_park():
    mgr, galvo = make_manager()
    mgr._is_scanning = True
    galvo.replies["stop_galvo_scan"] = "No response received"
    out = mgr.stop_scan()
    assert "error" in out
    assert galvo.names() == ["stop_galvo_scan"]
    assert mgr._is_scanning is True


def test_failed_park_after_stop_is_reported():
    mgr, galvo = make_manager()
    mgr._is_scanning = True
    galvo.replies["set_position"] = [{"return": 0, "qid": 3}]
    out = mgr.stop_scan()
    assert "error" in out and "parking failed" in out["error"]
    assert mgr._is_scanning is False         # the stop itself did succeed


# --------------------------------------------------------------------------
# park position persistence
# --------------------------------------------------------------------------

def test_park_config_survives_a_restart():
    mgr, _ = make_manager(park_x=100, park_y=100, park_on_stop=True)
    mgr.set_park_config(park_x=1234, park_y=3210, park_on_stop=False)

    restarted, _ = make_manager(park_x=100, park_y=100, park_on_stop=True)
    assert restarted.get_park_config() == {"park_x": 1234, "park_y": 3210,
                                           "park_on_stop": False}


def test_park_config_is_per_scanner_and_clamped():
    mgr, _ = make_manager(name="a")
    mgr.set_park_config(park_x=-5, park_y=99999)
    assert mgr.get_park_config()["park_x"] == 0 and mgr.get_park_config()["park_y"] == 4095
    other, _ = make_manager(name="b")
    assert other.get_park_config() == {"park_x": 2048, "park_y": 2048, "park_on_stop": True}


def test_setup_json_default_is_used_until_the_ui_edits_it():
    mgr, _ = make_manager(park_x=700)
    assert mgr.get_park_config()["park_x"] == 700
    assert not os.path.exists(mgr._park_config_path())     # nothing saved yet


def test_unreadable_park_file_falls_back_to_defaults():
    mgr, _ = make_manager()
    path = mgr._park_config_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write("{not json")
    again, _ = make_manager()
    assert again.get_park_config()["park_x"] == 2048


# --------------------------------------------------------------------------
# arbitrary points
# --------------------------------------------------------------------------

def test_points_started_once_acknowledged():
    mgr, galvo = make_manager()
    out = mgr.set_arbitrary_points(POINTS, "HIGH")
    assert out["status"] == "started" and out["num_points"] == 2
    assert mgr.get_arbitrary_scan_state()["running"] is True
    assert galvo.calls[0] == ("set_arbitrary_points", (POINTS, "HIGH"))


@pytest.mark.parametrize("reply", NACK_REPLIES)
def test_points_not_acknowledged_is_an_error(reply):
    mgr, galvo = make_manager()
    galvo.replies["set_arbitrary_points"] = reply
    assert "error" in mgr.set_arbitrary_points(POINTS)
    assert mgr.get_arbitrary_scan_state()["running"] is False


def test_firmware_point_limit_is_256_not_265():
    mgr, galvo = make_manager()
    too_many = [{"x": 1, "y": 1, "dwell_us": 1}] * 257
    assert "error" in mgr.set_arbitrary_points(too_many)
    assert galvo.calls == []                  # never sent: the board would reject it
    assert mgr.set_arbitrary_points(too_many[:256])["status"] == "started"


def test_stop_arbitrary_really_stops_then_parks():
    mgr, galvo = make_manager(park_x=5, park_y=6)
    mgr.set_arbitrary_points(POINTS)
    galvo.calls.clear()
    out = mgr.stop_arbitrary_scan()
    assert out["status"] == "stopped"
    # a real stop, not a fake "scan" of one point at the DAC origin
    assert galvo.names() == ["stop_galvo_scan", "set_position"]
    assert galvo.calls[1][1] == (5, 6)
    assert mgr.get_arbitrary_scan_state()["running"] is False


def test_pause_stops_without_parking_and_resume_resends_the_points():
    mgr, galvo = make_manager()
    mgr.set_arbitrary_points(POINTS, "CONTINUOUS")
    galvo.calls.clear()

    assert mgr.pause_arbitrary_scan()["status"] == "paused"
    assert galvo.names() == ["stop_galvo_scan"]
    assert mgr.get_arbitrary_scan_state()["paused"] is True

    assert mgr.resume_arbitrary_scan()["status"] == "resumed"
    assert galvo.calls[-1] == ("set_arbitrary_points", (POINTS, "CONTINUOUS"))
    state = mgr.get_arbitrary_scan_state()
    assert state["running"] is True and state["paused"] is False


def test_resume_without_pause_is_an_error():
    mgr, galvo = make_manager()
    assert "error" in mgr.resume_arbitrary_scan()
    mgr.set_arbitrary_points(POINTS)
    galvo.calls.clear()
    assert "error" in mgr.resume_arbitrary_scan()       # running, not paused
    assert galvo.calls == []


def test_unacknowledged_pause_keeps_the_scan_marked_running():
    mgr, galvo = make_manager()
    mgr.set_arbitrary_points(POINTS)
    galvo.replies["stop_galvo_scan"] = "No response received"
    assert "error" in mgr.pause_arbitrary_scan()
    state = mgr.get_arbitrary_scan_state()
    assert state["running"] is True and state["paused"] is False


# --------------------------------------------------------------------------
# controller: errors reach the HTTP caller
# --------------------------------------------------------------------------

class OneScannerMaster:
    def __init__(self, scanner):
        self._scanner = scanner

    def getAllDeviceNames(self):
        return ["galvo"]

    def __getitem__(self, name):
        return self._scanner


def make_controller(mgr):
    ctrl = GalvoScannerController.__new__(GalvoScannerController)
    ctrl._master = SimpleNamespace(galvoScannersManager=OneScannerMaster(mgr))
    ctrl._GalvoScannerController__logger = logging.getLogger("test-galvo")
    return ctrl


def test_parkGalvo_reports_success_with_position():
    mgr, _ = make_manager(park_x=1, park_y=2)
    out = make_controller(mgr).parkGalvo()
    assert out["status"] == "parked" and (out["x"], out["y"]) == (1, 2)
    json.dumps(out)                                   # must be serialisable for FastAPI


def test_parkGalvo_reports_an_unacknowledged_park_as_error():
    mgr, galvo = make_manager()
    galvo.replies["set_position"] = "No response received"
    out = make_controller(mgr).parkGalvo()
    assert "error" in out and "status" not in out


def test_parkGalvo_without_a_board_is_an_error():
    mgr, _ = make_manager()
    mgr._galvo = None
    assert "error" in make_controller(mgr).parkGalvo()


def test_stopGalvoScan_surfaces_a_failed_stop():
    mgr, galvo = make_manager()
    galvo.replies["stop_galvo_scan"] = [{"return": 0, "qid": 1}]
    assert "error" in make_controller(mgr).stopGalvoScan()


def test_setArbitraryPoints_surfaces_errors_and_enforces_the_firmware_limit():
    mgr, galvo = make_manager()
    ctrl = make_controller(mgr)
    pts = json.dumps([{"x": 1, "y": 1, "dwell_us": 1}] * 257)
    assert "256" in ctrl.setArbitraryPoints(pts, apply_affine=False)["error"]

    galvo.replies["set_arbitrary_points"] = "No response received"
    assert "error" in ctrl.setArbitraryPoints(json.dumps(POINTS), apply_affine=False)

    galvo.replies["set_arbitrary_points"] = {"return": 1, "qid": 2}
    out = ctrl.setArbitraryPoints(json.dumps(POINTS), apply_affine=False)
    assert out["status"] == "points_set" and out["num_points"] == 2


@pytest.mark.parametrize("method", ["stopArbitraryScan", "pauseArbitraryScan"])
def test_arbitrary_stop_and_pause_surface_errors(method):
    mgr, galvo = make_manager()
    ctrl = make_controller(mgr)
    ctrl.setArbitraryPoints(json.dumps(POINTS), apply_affine=False)
    galvo.replies["stop_galvo_scan"] = "No response received"
    assert "error" in getattr(ctrl, method)()
