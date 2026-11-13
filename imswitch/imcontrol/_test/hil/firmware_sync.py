"""Bring every board to the firmware server's version. FLASHES FIRMWARE.

A sync, not an update: a board with a newer developer build is flashed back as well. The
flashing itself is ImSwitch's own firmware update (startFirmwareUpdate), which checks every
image and counts a board as done only when it reports the new version.
"""
import time

import requests

import hil_client as client

SCAN_TIMEOUT = 5     # seconds checkFirmwareUpdates waits for the boards on the CAN bus
SYNC_TIMEOUT = 1800  # seconds one firmware update may take

# update_status values of a board that differs from the server and has an image to flash.
OUT_OF_SYNC = ("update_available", "device_newer")


class SyncFailed(Exception):
    pass


def label(device):
    return f"CAN {device.get('canId')} ({device.get('deviceTypeStr')}, {device['connection']})"


def scan_came_back_empty(result):
    """True if the CAN scan found nothing: then even the master is listed without a canId."""
    usb = next((d for d in result.get("devices") or [] if d["connection"] == "usb"), None)
    return (usb is not None and "can" in (usb.get("deviceTypeStr") or "").lower()
            and usb.get("canId") is None)


def check():
    """checkFirmwareUpdates; asked up to three times, because the CAN scan sometimes is empty."""
    for _ in range(3):
        result = client.get_json("/UC2ConfigController/checkFirmwareUpdates",
                                 {"timeout": SCAN_TIMEOUT}, timeout=120)
        if not scan_came_back_empty(result):
            break
        time.sleep(3)
    else:
        raise SyncFailed("the CAN scan came back empty three times")
    if not result.get("server_version"):
        raise SyncFailed(f"{result.get('firmware_server')} publishes no version.json")
    return result


def out_of_sync(result, connection):
    """The boards on "usb" (the master) or "can" (the nodes) that need flashing."""
    return [d for d in result["devices"]
            if d["connection"] == connection and d["update_status"] in OUT_OF_SYNC]


def flash(can_ids, include_master):
    """Start one ImSwitch firmware update and wait until it ended; raises unless it succeeded."""
    started = client.post_json("/UC2ConfigController/startFirmwareUpdate", json=can_ids,
                               params={"include_master": include_master}, timeout=120)
    if started.get("status") != "started":
        raise SyncFailed(f"ImSwitch refused the update: {started}")

    deadline = time.monotonic() + SYNC_TIMEOUT
    status, message = {"state": "running"}, None
    while status.get("state") == "running":
        if time.monotonic() > deadline:
            raise SyncFailed(f"the update still runs after {SYNC_TIMEOUT}s")
        time.sleep(5)
        try:
            status = client.get_json("/UC2ConfigController/getFirmwareUpdateStatus", timeout=120)
        except requests.RequestException:
            continue  # ImSwitch can be too busy flashing to answer; the update goes on
        if status.get("message") != message:
            message = status.get("message")
            print(f"sync:   {message}", flush=True)

    for step in status.get("steps") or []:
        print(f"sync:   {label(step)}: {step['status']} {step.get('message') or ''}", flush=True)
    if status.get("homing_required"):
        print("sync: the updated motors restarted and need homing", flush=True)
    if status.get("state") != "success":
        raise SyncFailed(f"the update ended {status.get('state')!r}")


def sync():
    """Flash every board that differs from the server; raises SyncFailed if one still does."""
    result = check()
    print(f"sync: server version {result['server_version']}", flush=True)
    for device in result["devices"]:
        print(f"sync:   {label(device)}: {device.get('installed_version') or 'no version'} "
              f"[{device['update_status']}]", flush=True)

    # The master first, on its own: old master firmware cuts off the version the nodes report,
    # so behind it no node would count as updated.
    if out_of_sync(result, "usb"):
        flash([], include_master=True)
        result = check()
    nodes = out_of_sync(result, "can")
    if nodes:
        flash([d["canId"] for d in nodes], include_master=False)
        result = check()

    left = out_of_sync(result, "usb") + out_of_sync(result, "can")
    if left:
        raise SyncFailed("still out of sync: " + ", ".join(label(d) for d in left))
    print(f"sync: every board runs {result['server_version']}", flush=True)
