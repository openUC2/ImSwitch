"""Check the firmware server and whether the boards run the newest firmware.

Read-only: nothing here flashes anything. The OTA endpoints that would are
deliberately not touched.

- test_master_firmware_is_reported: the USB-connected master identifies itself
- test_firmware_server_is_configured: an OTA firmware server URL is set
- test_firmware_server_lists_binaries: that server answers with .bin files
- test_reachable_bus_devices_have_firmware: every CAN node the master can
  actually talk to has a firmware file mapped to its id
- test_boards_run_newest_firmware: the master and every reachable node run
  the version the server offers
"""

import os

import pytest
import requests


BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")

# Scanning the bus takes a few seconds on the firmware side.
SCAN_TIMEOUT = int(os.environ.get("FIRMWARE_SCAN_TIMEOUT", "5"))


def api(method, **params):
    """Call one UC2ConfigController endpoint, skipping when ImSwitch is down.

    A refused connection says nothing about the firmware, and the suite is
    meant to be runnable without a rig.
    """
    try:
        response = requests.get(
            f"{BASE_URL}/api/UC2ConfigController/{method}",
            params=params,
            timeout=60,
        )
    except requests.RequestException as exc:
        pytest.skip(f"ImSwitch not reachable at {BASE_URL}: {exc}")

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


def fetch_manifest(server_url):
    """The firmware server's version.json, or None when it publishes none.

    The firmware build writes it next to the images (youseetoo/uc2-esp32,
    tools/write_fw_manifest.py). Its "version" is the exact string every image
    of that build reports as fwVersion (CANopen OD 0x2500), so it is the newest
    version the server can hand out. Servers built before versioned firmware
    have no version.json. Raises requests.RequestException when unreachable.
    """
    response = requests.get(f"{server_url.rstrip('/')}/version.json", timeout=30)

    if response.status_code == 404:
        return None

    response.raise_for_status()
    return response.json()


def boards(scan):
    """The master and every CAN node that answered the scan."""
    nodes = [
        device for device in scan.get("scan") or []
        if device.get("statusStr") != "unreachable"
    ]
    return ([scan["master"]] if scan.get("master") else []) + nodes


def outdated(scan, newest):
    """The boards that report a fwVersion other than the newest one.

    A board that reports no fwVersion cannot be compared, which is not the
    same as outdated, so it is left out here.
    """
    return [
        board for board in boards(scan)
        if board.get("fwVersion") and board["fwVersion"] != newest
    ]


def label(board):
    """'CAN 10 (motor)'; the master's scan entry carries no device type."""
    return f"CAN {board['canId']} ({board.get('deviceTypeStr', 'master')})"


def newest_firmware():
    """The server's version.json, skipping when the server cannot name a version."""
    url = api("getOTAFirmwareServer").get("firmware_server_url")

    if not url:
        pytest.skip("no OTA firmware server configured")

    try:
        manifest = fetch_manifest(url)
    except requests.RequestException as exc:
        pytest.skip(f"firmware server not reachable from this host: {exc}")

    if not (manifest or {}).get("version"):
        pytest.skip(f"{url} publishes no version.json: firmware built before versioning")

    return manifest


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
def test_reachable_bus_devices_have_firmware():
    """Every CAN node the master can talk to must have a firmware mapped.

    The mapping in UC2ConfigController._get_can_id_firmware_mapping is a fixed
    table, so a node whose id is missing from it cannot be updated through the
    CAN OTA wizard even when its binary sits on the server.

    Only nodes that answered the scan are required: an unreachable one may be
    powered down or unrouted, which says nothing about the mapping. Those are
    printed instead, because they are worth a look.
    """
    scan = api("scan_canbus", timeout=SCAN_TIMEOUT)
    devices = scan.get("scan") or []

    if not devices:
        pytest.skip("no devices on the CAN bus")

    # listAvailableFirmware keys its result by CAN id, but JSON object keys are
    # strings, so they have to be converted before comparing with the scan.
    mapped = {
        int(can_id) for can_id in (api("listAvailableFirmware").get("firmware") or {})
    }

    reachable = [
        device for device in devices
        if device.get("statusStr") != "unreachable"
    ]

    if not reachable:
        pytest.skip(f"no reachable CAN node: {scan.get('detected_ids')}")

    missing = [
        device["canId"] for device in reachable
        if device["canId"] not in mapped
    ]

    unreachable_unmapped = [
        device["canId"] for device in devices
        if device.get("statusStr") == "unreachable"
        and device["canId"] not in mapped
    ]

    print(
        f"\nbus: {scan.get('detected_ids')}, "
        f"reachable: {[d['canId'] for d in reachable]}, "
        f"firmware mapped for: {sorted(mapped)}"
    )

    if unreachable_unmapped:
        print(
            f"unreachable and unmapped (not asserted): {unreachable_unmapped}"
        )

    assert not missing, (
        f"CAN nodes {missing} answer on the bus but have no firmware mapped; "
        f"add them to _get_can_id_firmware_mapping in UC2ConfigController"
    )


@pytest.mark.hardware
def test_boards_run_newest_firmware():
    """The master and every reachable CAN node must run the newest firmware.

    Newest is what the firmware server offers: the "version" in its
    version.json, the same string a board built from those images reports as
    fwVersion. So this is an exact string comparison, no dates involved.

    Boards that answer but report no fwVersion are printed, not counted.
    """
    manifest = newest_firmware()
    newest = manifest["version"]
    scan = api("scan_canbus", timeout=SCAN_TIMEOUT)
    answering = boards(scan)

    if not answering:
        pytest.skip(f"no board answered the scan: {scan.get('detected_ids')}")

    print(f"\nserver offers {newest} (commit {manifest.get('commit') or '?'})")

    for board in answering:
        print(f"{label(board)}: {board.get('fwVersion') or 'reports no fwVersion'}")

    compared = [board for board in answering if board.get("fwVersion")]

    if not compared:
        pytest.skip("no answering board reports a fwVersion")

    stale = outdated(scan, newest)

    # Everything on one line, ids first: the runners pass --tb=line, which
    # shows nothing but the first line of the message.
    assert not stale, (
        f"CAN {[board['canId'] for board in stale]} outdated ({len(stale)} of "
        f"{len(compared)} boards do not run {newest}) | "
        + " | ".join(f"{label(board)}: {board['fwVersion']}" for board in stale)
    )
