"""Flash every reachable CAN node that does not run the newest firmware.

Newest is the version the firmware server names in its version.json; the
check is outdated() from test_firmware_server.py. Outdated nodes are flashed
one by one over CAN (startCANStreamingOTA, blocking), then the bus is scanned
again. The USB master is only reported, never flashed. Exits 1 if a node is
still outdated afterwards. A server without version.json cannot name a
version, so nothing is flashed.

Runs inside the container, where the firmware server is reachable:

    docker exec imswitch-server-1 python3 /tmp/e2e/firmware/update_firmware.py
"""
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from test_firmware_server import (
    BASE_URL, SCAN_TIMEOUT, boards, fetch_manifest, label, outdated,
)


def api(method, wait=60, **params):
    """Call one UC2ConfigController endpoint and return its JSON.

    `wait` is the HTTP timeout; every other keyword is a query parameter.
    """
    response = requests.get(
        f"{BASE_URL}/api/UC2ConfigController/{method}", params=params, timeout=wait
    )
    response.raise_for_status()
    return response.json()


def scan():
    return api("scan_canbus", wait=SCAN_TIMEOUT + 30, timeout=SCAN_TIMEOUT)


if os.environ.get("FIRMWARE_UPDATE", "on") == "off":
    sys.exit(print("firmware: update skipped (FIRMWARE_UPDATE=off)"))

server = api("getOTAFirmwareServer").get("firmware_server_url")
try:
    manifest = fetch_manifest(server) if server else None
except requests.RequestException as exc:
    sys.exit(f"firmware: server {server} not reachable: {exc}")

if not (manifest or {}).get("version"):
    sys.exit(print(f"firmware: {server or 'no server'} names no version (no version.json)"))

newest = manifest["version"]
print(f"firmware: server offers {newest}", flush=True)

first = scan()
master_id = (first.get("master") or {}).get("canId")
mapped = api("listAvailableFirmware").get("firmware") or {}
todo = []

for board in outdated(first, newest):
    if board["canId"] == master_id:
        print(f"{label(board)}: runs {board['fwVersion']}, flash it over USB (not done here)")
    elif str(board["canId"]) not in mapped:
        print(f"{label(board)}: runs {board['fwVersion']}, but has no firmware mapped")
    else:
        todo.append(board)

if not todo:
    sys.exit(print("firmware: no CAN node to update"))

for board in todo:
    print(f"{label(board)}: {board['fwVersion']} -> {newest} ...", flush=True)
    print(f"{label(board)}: {api('startCANStreamingOTA', wait=900, can_id=board['canId'])}")
    time.sleep(10)  # reboot before the next node or the re-scan

# A node that does not answer the re-scan is not verified either.
running = {board["canId"]: board.get("fwVersion") for board in boards(scan())}
flashed = sorted(board["canId"] for board in todo)
still = [can_id for can_id in flashed if running.get(can_id) != newest]
if still:
    sys.exit(f"firmware: CAN {still} not on {newest} after flashing")
print(f"firmware: CAN {flashed} updated to {newest} and verified")
