"""Bring every board to the firmware server's version: a firmware sync.

Sync, not update: a board whose version differs from the server's is flashed
either way, also when it runs a newer developer build (update_status
"device_newer"). There is no developer mode that keeps such a build.

Uses ImSwitch's own update (UC2ConfigController: checkFirmwareUpdates,
startFirmwareUpdate, getFirmwareUpdateStatus). It downloads and sha256-checks
each image, switches the lasers off and counts a board as done only when it
reports the new version. Two runs instead of one, because that update flashes
the USB master last: a master built before versioned firmware reads at most 39
characters of a node's version, so no updated node would ever verify. Hence
the master first, then the nodes from a fresh check.

Without --yes it only prints what it would flash.

    python3 sync_firmware.py          # plan only
    python3 sync_firmware.py --yes    # flash

Exit codes: 0 every board with an image runs the server's version, 1 a board
does not after the sync, 2 the sync could not run (ImSwitch unreachable, no
version.json on the firmware server, ImSwitch refused to start).
"""
import argparse
import os
import sys
import time

import requests


BASE_URL = os.environ.get("IMSWITCH_URL", "http://localhost:8001")

# Differs from the server's version and has an image there to flash.
OUT_OF_SYNC = ("update_available", "device_newer")


def die(message):
    print(message, file=sys.stderr)
    sys.exit(2)


def api(method, post=False, body=None, **params):
    """Call one UC2ConfigController endpoint and return its JSON."""
    url = f"{BASE_URL}/api/UC2ConfigController/{method}"
    if post:
        response = requests.post(url, params=params, json=body, timeout=120)
    else:
        response = requests.get(url, params=params, timeout=120)
    response.raise_for_status()
    return response.json()


def label(device):
    return f"CAN {device.get('canId')} ({device.get('deviceTypeStr')}, {device['connection']})"


def check():
    """checkFirmwareUpdates; a server without version.json gives nothing to sync to."""
    result = api("checkFirmwareUpdates", timeout=5)
    if not result.get("server_version"):
        die(f"sync: {result.get('firmware_server')} publishes no version.json, nothing to sync to")
    return result


def out_of_sync(result, connection):
    return [
        device for device in result["devices"]
        if device["connection"] == connection and device["update_status"] in OUT_OF_SYNC
    ]


def run(can_ids, include_master):
    """One ImSwitch update, waited for; True when every step ended done."""
    started = api(
        "startFirmwareUpdate", post=True, body=can_ids,
        include_master="true" if include_master else "false",
    )
    if started.get("status") != "started":
        die("sync: ImSwitch refused: " + "; ".join(started.get("reasons") or [str(started)]))

    message = None
    while True:
        time.sleep(5)
        try:
            status = api("getFirmwareUpdateStatus")
        except requests.RequestException:
            continue  # the update runs on in ImSwitch; ask again
        if status.get("message") != message:
            message = status.get("message")
            print(f"sync:   {message}", flush=True)
        if status.get("state") != "running":
            break

    for step in status.get("steps") or []:
        print(f"sync:   {label(step)}: {step['status']} {step.get('message') or ''}")
    if status.get("homing_required"):
        print("sync: updated motors restarted and need homing")
    return status.get("state") == "success"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--yes", action="store_true", help="flash; otherwise only print the plan")
    args = parser.parse_args()

    result = check()
    server = result["server_version"]
    print(f"sync: server version {server}")
    for device in result["devices"]:
        installed = device.get("installed_version") or "no version"
        print(f"sync:   {label(device)}: {installed} [{device['update_status']}]")

    master = out_of_sync(result, "usb")
    nodes = out_of_sync(result, "can")
    if not master and not nodes:
        sys.exit(print("sync: every board with an image runs the server's version"))
    if not args.yes:
        planned = ", ".join(label(device) for device in master + nodes)
        sys.exit(print(f"sync: would flash {planned}; pass --yes to flash"))

    if master:
        print("sync: the master first", flush=True)
        if not run([], include_master=True):
            sys.exit("sync: the master did not end on the server's version")
        nodes = out_of_sync(check(), "can")  # the new master reads full versions

    if nodes:
        print(f"sync: {len(nodes)} CAN node(s)", flush=True)
        if not run([device["canId"] for device in nodes], include_master=False):
            sys.exit("sync: a CAN node did not end on the server's version")

    final = check()
    left = out_of_sync(final, "usb") + out_of_sync(final, "can")
    if left:
        sys.exit("sync: still out of sync: " + ", ".join(label(device) for device in left))
    print(f"sync: every board with an image runs {server}")


if __name__ == "__main__":
    try:
        main()
    except requests.RequestException as exc:
        die(f"sync: ImSwitch not reachable at {BASE_URL}: {exc}")
