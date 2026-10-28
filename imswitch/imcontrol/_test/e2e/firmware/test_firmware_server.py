"""Check the firmware server and whether every board runs its version.

Read-only: nothing here flashes anything. ci/sync_firmware.py does that, and
run_firmware_test.sh and run_all.sh run it before these tests.

The version checks use ImSwitch's own comparison (checkFirmwareUpdates): the
version in the firmware server's version.json against what each board reports.
An ImSwitch without that endpoint skips them.

- test_master_firmware_is_reported: the USB-connected master identifies itself
- test_firmware_server_is_configured: an OTA firmware server URL is set
- test_firmware_server_lists_binaries: that server answers with .bin files
- test_answering_boards_have_firmware: every board that answers has an image
  on the server
- test_boards_run_server_version: and runs the server's version
"""

import os

import pytest
import requests


BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")

# Scanning the bus takes a few seconds on the firmware side.
SCAN_TIMEOUT = int(os.environ.get("FIRMWARE_SCAN_TIMEOUT", "5"))

# update_status values that differ from the server's version.
OUT_OF_SYNC = ("update_available", "device_newer")


def api(method, **params):
    """Call one UC2ConfigController endpoint, skipping when ImSwitch cannot answer.

    A refused connection says nothing about the firmware, and the suite is
    meant to be runnable without a rig. A 404 means this ImSwitch has no such
    endpoint (or no UC2ConfigController at all), which is just as absent.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/api/UC2ConfigController/{method}",
            params=params,
            timeout=60,
        )
    except requests.RequestException as exc:
        pytest.skip(f"ImSwitch not reachable at {BASE_URL}: {exc}")

    if response.status_code == 404:
        pytest.skip(f"this ImSwitch has no {method}")

    assert response.status_code == 200, (
        f"{method} -> {response.status_code}: {response.text}"
    )

    return response.json()


def firmware_files():
    """The flat .bin list from the server, skipping when it is unreachable.

    An unreachable or unset server is a missing service rather than a broken
    one, so it skips like every other absent-hardware case in this suite.
    """
    listing = api("listAllFirmwareFiles")

    if listing.get("status") != "success":
        pytest.skip(f"firmware server unusable: {listing.get('message')}")

    return listing


def label(device):
    return f"CAN {device.get('canId')} ({device.get('deviceTypeStr')}, {device['connection']})"


@pytest.fixture(scope="module")
def firmware_check():
    """checkFirmwareUpdates, once for both tests: it scans the CAN bus.

    A CAN master always appears in its own scan with canId 1. Without it the
    scan answered nothing, and both tests would pass with no node compared.
    """
    check = api("checkFirmwareUpdates", timeout=SCAN_TIMEOUT)
    usb = next(
        (device for device in check.get("devices") or [] if device["connection"] == "usb"),
        None,
    )

    if usb and "can" in (usb.get("deviceTypeStr") or "").lower() and usb.get("canId") is None:
        pytest.fail("the CAN scan came back empty: no node listed, the master without canId")

    return check


def answering(check):
    """The master and every CAN node that answered, skipping when none did."""
    devices = [
        device for device in check.get("devices") or []
        if device["update_status"] != "unreachable"
    ]

    if not devices:
        pytest.skip("no board answered")

    return devices


@pytest.mark.hardware
def test_master_firmware_is_reported():
    """The USB-connected ESP32 master must report its own identity."""
    info = api("getFirmwareInfo")

    if info.get("status") == "error":
        pytest.skip(f"getFirmwareInfo failed: {info.get('message')}")

    assert info.get("connected") is True, f"master board is not connected: {info}"

    # The build date and pindef are what actually tell two firmwares apart, so
    # a reply without them is not proof that a master firmware is running.
    for field in ("name", "version", "date", "pindef"):
        assert info.get(field), f"master firmware reports no {field}: {info}"

    print(
        f"\nmaster: {info.get('name')} {info.get('version')} "
        f"({info.get('pindef')}, built {info.get('date')}) "
        f"on {info.get('serialport')}"
    )


@pytest.mark.hardware
def test_firmware_server_is_configured():
    """ImSwitch must know where to fetch firmware from."""
    url = api("getOTAFirmwareServer").get("firmware_server_url")

    assert url, "no OTA firmware server configured"

    print(f"\nfirmware server: {url}")


@pytest.mark.hardware
def test_firmware_server_lists_binaries():
    """The server must answer with a usable list of .bin files."""
    listing = firmware_files()
    files = listing.get("files") or []

    assert files, f"firmware server {listing.get('firmware_server')} offers no .bin files"

    # Without a name and a URL an entry cannot be downloaded, which is the
    # only thing the list is good for.
    for entry in files:
        assert entry.get("filename", "").endswith(".bin"), entry
        assert entry.get("url"), entry

    print(f"\n{len(files)} firmware files on {listing.get('firmware_server')}")


@pytest.mark.hardware
def test_answering_boards_have_firmware(firmware_check):
    """Every board that answers must have an image on the firmware server.

    Without one, neither the sync nor ImSwitch's update can flash it. The image
    is the one the board names itself (image_source "reported"), or for
    firmware too old to name one, ImSwitch's CAN-id table ("mapping").

    Unreachable nodes are printed, not asserted: they may be powered down.
    """
    print()
    for device in firmware_check.get("devices") or []:
        print(
            f"{label(device)}: {device.get('filename') or 'no image'} "
            f"[{device.get('image_source')}, {device['update_status']}]"
        )

    missing = [
        device for device in answering(firmware_check)
        if device["update_status"] == "no_firmware" or not device.get("filename")
    ]

    assert not missing, (
        f"CAN {[device.get('canId') for device in missing]} answer but have no image on "
        f"{firmware_check.get('firmware_server')} | "
        + " | ".join(label(device) for device in missing)
    )


@pytest.mark.hardware
def test_boards_run_server_version(firmware_check):
    """The master and every CAN node that answers must run the server's version.

    ImSwitch compares the exact strings: "version" in the server's version.json
    and what each board reports (CANopen OD 0x2500 / identifier_version). A
    newer developer build ("device_newer") counts as out of sync too: the sync
    flashes it back to the server's version.
    """
    server = firmware_check.get("server_version")

    if not server:
        pytest.skip(
            f"{firmware_check.get('firmware_server')} names no version: "
            + ("no version.json" if firmware_check.get("server_reachable") else "unreachable")
        )

    devices = answering(firmware_check)

    print(f"\nserver offers {server}")
    for device in devices:
        print(
            f"{label(device)}: {device.get('installed_version') or 'reports no version'} "
            f"[{device['update_status']}]"
        )

    stale = [device for device in devices if device["update_status"] in OUT_OF_SYNC]

    # Everything on one line, ids first: the runners pass --tb=line, which
    # shows nothing but the first line of the message.
    assert not stale, (
        f"CAN {[device.get('canId') for device in stale]} out of sync ({len(stale)} of "
        f"{len(devices)} boards do not run {server}) | "
        + " | ".join(
            f"{label(device)}: {device.get('installed_version')}" for device in stale
        )
    )
