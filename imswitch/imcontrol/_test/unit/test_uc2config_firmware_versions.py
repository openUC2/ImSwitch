"""Firmware version comparison in UC2ConfigController.checkFirmwareUpdates.

No hardware, no server: the controller is built without __init__ and its three
inputs (version.json, the USB board's identity, the CAN bus scan) are stubbed.
"""
import logging

import pytest

from imswitch.imcontrol.controller.controllers.UC2ConfigController import UC2ConfigController

SERVER = "v2026.0.0-beta.4-6-g2108590-t20260930092622"
MANIFEST = {
    "version": SERVER,
    "commit_time": "2026-09-30T09:26:22Z",
    "files": {
        "esp32_UC2_canopen_master_release.bin": {"size": 1, "sha256": "aa"},
        "esp32_UC2_canopen_slave_motor_release_motX.bin": {"size": 1, "sha256": "bb"},
        "esp32_UC2_canopen_slave_laser_release.bin": {"size": 1, "sha256": "cc"},
    },
}


def make_controller(manifest, usb, scan):
    c = object.__new__(UC2ConfigController)
    c._UC2ConfigController__logger = c._logger = logging.getLogger("test")
    c._firmware_server_url = "http://fw/firmware"
    c._fetch_firmware_manifest = lambda: manifest
    c.getFirmwareInfo = lambda: usb
    c.scan_canbus = lambda timeout=5, probe_range=False: scan
    return c


@pytest.mark.parametrize("installed, available, expected", [
    (SERVER, SERVER, "up_to_date"),
    ("UC2-ESP v2.0", SERVER, "update_available"),       # firmware before version reporting
    (None, SERVER, "update_available"),
    ("v2026.0.0-beta.4-5-gaaaaaaa-t20260901000000", SERVER, "update_available"),
    ("v2026.0.0-beta.4-7-gbbbbbbb-t20261001000000-dirty", SERVER, "device_newer"),
    ("v2026.1.0", SERVER, "update_available"),          # tag build: no timestamp to order by
    (SERVER, None, "unknown"),                          # server has no version.json
])
def test_update_status(installed, available, expected):
    assert UC2ConfigController._firmware_update_status(installed, available) == expected


def test_check_can_master_and_nodes():
    usb = {"pindef": "UC2_canopen_master", "isMaster": True, "connected": True,
           "fwVersion": SERVER, "date": "Sep 30 2026 09:40:00"}
    scan = {
        "master": {"canId": 1, "fwVersion": SERVER, "mac": "AA"},
        "scan": [
            {"canId": 11, "deviceTypeStr": "motor", "statusStr": "idle",
             "fwVersion": "UC2-ESP v2.0"},
            {"canId": 20, "deviceTypeStr": "laser", "statusStr": "unreachable"},
            {"canId": 40, "deviceTypeStr": "galvo", "statusStr": "idle", "fwVersion": SERVER},
        ],
    }
    result = make_controller(MANIFEST, usb, scan).checkFirmwareUpdates()

    by_id = {d["canId"]: d for d in result["devices"]}
    assert result["server_version"] == SERVER
    assert by_id[1]["connection"] == "usb"
    assert by_id[1]["filename"] == "esp32_UC2_canopen_master_release.bin"
    assert by_id[1]["update_status"] == "up_to_date"
    assert by_id[11]["update_status"] == "update_available"
    assert by_id[11]["available_version"] == SERVER
    assert by_id[20]["update_status"] == "unreachable"
    assert by_id[40]["update_status"] == "no_firmware"      # galvo image not on this server
    assert result["updates_available"] == 1


def test_check_standalone_board_skips_can_scan():
    usb = {"pindef": "UC2_3", "isMaster": False, "connected": True, "fwVersion": "UC2-ESP v2.0"}
    manifest = {"version": SERVER, "files": {"esp32_UC2_3.bin": {"size": 1, "sha256": "dd"}}}

    def no_scan(**_):
        raise AssertionError("standalone boards must not be CAN-scanned")

    c = make_controller(manifest, usb, {})
    c.scan_canbus = no_scan
    [board] = c.checkFirmwareUpdates()["devices"]
    assert board["filename"] == "esp32_UC2_3.bin"
    assert board["update_status"] == "update_available"


def test_check_without_manifest_or_board():
    usb = {"status": "error", "message": "no serial"}
    result = make_controller({}, usb, {}).checkFirmwareUpdates()
    assert result["devices"] == []
    assert result["server_version"] is None
