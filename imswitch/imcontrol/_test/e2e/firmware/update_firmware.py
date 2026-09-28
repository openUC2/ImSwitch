"""Flash every reachable CAN node that does not run the firmware the server offers.

Runs inside the container, where the firmware server is reachable. Same
currency check as test_reachable_bus_devices_run_current_firmware; outdated
nodes are flashed one by one over CAN (startCANStreamingOTA, blocking). The
USB master is not touched. Exits 1 if a flash failed or a node is still
outdated afterwards.

    docker exec imswitch-server-1 python3 /tmp/e2e/firmware/update_firmware.py
"""
import os
import sys
import time

import requests

sys.path.insert(0, os.path.dirname(__file__))
from test_firmware_server import BASE_URL, IDENTITY_PATTERN, SCAN_TIMEOUT


def api(method, timeout=60, **params):
    """Call one UC2ConfigController endpoint and return its JSON."""
    response = requests.get(
        f"{BASE_URL}/api/UC2ConfigController/{method}", params=params, timeout=timeout
    )
    response.raise_for_status()
    return response.json()


def outdated():
    """{can_id: filename} of reachable nodes whose build differs from their .bin."""
    mapped = api("listAvailableFirmware").get("firmware") or {}
    result = {}

    for device in api("scan_canbus", timeout=SCAN_TIMEOUT)["scan"] or []:
        entry = mapped.get(str(device["canId"]))
        if device.get("statusStr") == "unreachable" or entry is None:
            continue

        found = {
            (m.group(1).decode(), f"{m.group(3).decode()} {m.group(2).decode()}")
            for m in IDENTITY_PATTERN.finditer(requests.get(entry["url"], timeout=60).content)
        }
        running = (device.get("fwVersion"), device.get("build"))

        # No unique identity or no reported build: undecidable, so left alone.
        if len(found) == 1 and all(running) and running != found.pop():
            print(f"CAN {device['canId']}: running {running[1]!r}, server has {entry['filename']}")
            result[device["canId"]] = entry["filename"]

    return result


if os.environ.get("FIRMWARE_UPDATE", "on") == "off":
    sys.exit(print("firmware update skipped (FIRMWARE_UPDATE=off)"))

print("firmware: checking the CAN nodes against the firmware server ...", flush=True)
todo = outdated()
if not todo:
    sys.exit(print("firmware: all reachable CAN nodes are current"))

for can_id, filename in todo.items():
    print(f"CAN {can_id}: flashing {filename} ...", flush=True)
    print(f"CAN {can_id}: {api('startCANStreamingOTA', timeout=900, can_id=can_id)}")
    time.sleep(10)  # reboot before the next node or the re-scan

still = outdated()
if still:
    sys.exit(f"firmware: still outdated after flashing: {sorted(still)}")
print(f"firmware: CAN {sorted(todo)} updated and verified")
