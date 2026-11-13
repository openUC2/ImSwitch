"""The firmware server is usable and every board runs its version. Only reads, flashes nothing.

checkFirmwareUpdates is ImSwitch's own comparison of the server's version.json with what each
board reports. To flash before the tests, start the suite with --sync-firmware.
"""
import pytest

import hil_client as client

SCAN_TIMEOUT = 5  # seconds checkFirmwareUpdates waits for the boards on the CAN bus
# update_status values of a board that differs from the server and has an image to flash.
OUT_OF_SYNC = ("update_available", "device_newer")


def label(device):
    return f"CAN {device.get('canId')} ({device.get('deviceTypeStr')}, {device['connection']})"


def scan_came_back_empty(result):
    """True if the CAN scan found nothing: then even the master is listed without a canId."""
    usb = next((d for d in result.get("devices") or [] if d["connection"] == "usb"), None)
    return (usb is not None and "can" in (usb.get("deviceTypeStr") or "").lower()
            and usb.get("canId") is None)


@pytest.fixture(scope="module")
def check():
    """The answer of checkFirmwareUpdates, asked once: it scans the CAN bus."""
    result = client.get_json("/UC2ConfigController/checkFirmwareUpdates",
                             {"timeout": SCAN_TIMEOUT}, timeout=60)
    assert not scan_came_back_empty(result), (
        "the CAN scan came back empty: no node listed, the master without canId")
    print()
    for device in result["devices"]:
        print(f"{label(device)}: {device.get('installed_version') or 'no version'}, "
              f"image {device.get('filename') or 'none'} [{device['update_status']}]")
    return result


@pytest.fixture(scope="module")
def answering(check):
    """The boards that answered the scan. The others may just be powered down."""
    boards = [d for d in check["devices"] if d["update_status"] != "unreachable"]
    if not boards:
        pytest.skip("no board answered")
    return boards


def test_master_firmware_is_reported():
    info = client.get_json("/UC2ConfigController/getFirmwareInfo", timeout=60)
    if info.get("status") == "error":
        pytest.skip(f"getFirmwareInfo failed: {info.get('message')}")
    print(f"\nmaster: {info}")
    assert info.get("connected") is True, f"master board is not connected: {info}"
    for field in ("name", "version", "date", "pindef"):
        assert info.get(field), f"master firmware reports no {field}: {info}"


def test_firmware_server_is_configured():
    url = client.get_json("/UC2ConfigController/getOTAFirmwareServer").get("firmware_server_url")
    print(f"\nfirmware server: {url}")
    assert url, "no OTA firmware server configured"


def test_firmware_server_lists_binaries():
    listing = client.get_json("/UC2ConfigController/listAllFirmwareFiles", timeout=60)
    if listing.get("status") != "success":
        pytest.skip(f"firmware server unusable: {listing.get('message')}")
    files = listing.get("files") or []
    print(f"\n{len(files)} firmware files on {listing.get('firmware_server')}")
    assert files, f"firmware server {listing.get('firmware_server')} offers no .bin files"
    for entry in files:
        assert entry.get("filename", "").endswith(".bin"), entry
        assert entry.get("url"), entry


def test_answering_boards_have_firmware(check, answering):
    missing = [label(d) for d in answering
               if d["update_status"] == "no_firmware" or not d.get("filename")]
    assert not missing, f"no image on {check.get('firmware_server')} for: {', '.join(missing)}"


def test_boards_run_server_version(check, answering):
    """Exactly the server's version: a board with a newer build fails too."""
    server = check.get("server_version")
    if not server:
        pytest.skip(f"{check.get('firmware_server')} names no version (no version.json?)")
    stale = [f"{label(d)} runs {d.get('installed_version')}"
             for d in answering if d["update_status"] in OUT_OF_SYNC]
    assert not stale, f"not on the server's version {server}: {', '.join(stale)}"
