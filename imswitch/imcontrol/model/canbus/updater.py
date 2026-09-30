"""Compare every board with the firmware server and update the outdated ones.

check(): read-only — version.json, the USB board's identity and a bus scan.
start(): an unattended run in a background thread — refuses while the host
says it is busy, then before_update (e.g. lasers off), each CAN node one at a
time, the USB master last, and stops at the first failure. A board counts as
done only when it reports the new version afterwards.
"""
import datetime
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from .images import legacy_image, update_status


@dataclass
class UpdateHooks:
    """What the host (ImSwitch) contributes."""
    usb_info: Callable[[], dict] = dict
    """Identity of the USB board: {connected, pindef, isMaster, fwVersion, fwImage, date,
    serialport}."""
    blockers: Callable[[], list] = list
    """Reasons not to start now, e.g. an experiment is running."""
    before_update: Callable[[], None] = field(default=lambda: None)
    """Runs once before the first board, e.g. switch lasers off."""
    motor_restarted: Callable[[], None] = field(default=lambda: None)
    """A motor node rebooted and lost its position."""


class FirmwareUpdater:
    def __init__(self, server, bus, ota, usb, guard, hooks: UpdateHooks, logger=None):
        self.server, self.bus, self.ota, self.usb, self.guard = server, bus, ota, usb, guard
        self.hooks = hooks
        self._log = logger or logging.getLogger(__name__)
        self.state = {"state": "idle", "steps": []}
        self.cancel_event = threading.Event()
        self.thread = None

    # ── check ───────────────────────────────────────────────────────────────

    def check(self, timeout=5, probe_range=False) -> dict:
        """Installed vs server version per board. Each board is matched to the
        image it reports it was built as (image_source "reported"); older
        firmware falls back to the pindef (USB) or its CAN id ("mapping").
        update_status: up_to_date | update_available | device_newer | unknown
        (no version.json) | no_firmware (no image on the server) | unreachable."""
        manifest = self.server.manifest()
        manifest_files = manifest.get("files", {})

        def entry(installed, filename, source, reachable=True, **fields):
            available = self.server.file_version(manifest, filename)
            if not reachable:
                status = "unreachable"
            elif manifest_files and filename not in manifest_files:
                status = "no_firmware"
            else:
                status = update_status(installed, available)
            return {**fields, "installed_version": installed or None, "filename": filename,
                    "image_source": source, "available_version": available,
                    "update_status": status}

        usb = self.hooks.usb_info() or {}
        pindef = str(usb.get("pindef") or "")
        is_can_master = bool(usb.get("isMaster")) or "can" in pindef.lower()
        scan = self.bus.scan(timeout=timeout, probe_range=probe_range) if is_can_master else {}
        master = scan.get("master") or {}

        devices = []
        if usb.get("connected") or master:
            image = usb.get("fwImage") or master.get("fwImage")
            source = "reported"
            if not image:  # built before the image was reported: named after the pindef (= env)
                candidates = ([f"esp32_{pindef}_release.bin", f"esp32_{pindef}.bin"]
                              if pindef else [])
                if legacy_image(master.get("canId")):
                    candidates.append(legacy_image(master.get("canId")))
                image = next((c for c in candidates if c in manifest_files),
                             candidates[0] if candidates else None)
                source = "mapping"
            devices.append(entry(usb.get("fwVersion") or master.get("fwVersion"), image, source,
                                 canId=master.get("canId"), deviceTypeStr=pindef or "usb",
                                 connection="usb", build=usb.get("date") or master.get("build"),
                                 mac=master.get("mac")))
        for node in scan.get("scan", []):
            devices.append(entry(node.get("fwVersion"),
                                 node.get("fwImage") or legacy_image(node.get("canId")),
                                 "reported" if node.get("fwImage") else "mapping",
                                 reachable=node.get("statusStr") != "unreachable",
                                 canId=node.get("canId"), deviceTypeStr=node.get("deviceTypeStr"),
                                 connection="can", build=node.get("build"), mac=node.get("mac")))
        return {"status": "success", "firmware_server": self.server.url,
                "server_version": manifest.get("version"),
                "server_commit_time": manifest.get("commit_time"),
                "updates_available": sum(d["update_status"] == "update_available" for d in devices),
                "devices": devices}

    # ── update ──────────────────────────────────────────────────────────────

    def blockers(self) -> list:
        return self.guard.busy_reasons() + list(self.hooks.blockers())

    def start(self, can_ids=None, include_master=False) -> dict:
        """{"status": "started", "steps"} or {"status": "refused", "reasons"}.
        *can_ids* default: every node whose status is update_available;
        explicitly listed nodes are updated whatever their status."""
        if self.state.get("state") == "running":
            return {"status": "refused", "reasons": ["A firmware update is already running."]}
        reasons = self.blockers()
        if reasons:
            return {"status": "refused", "reasons": reasons}
        check = self.check()
        if not check.get("server_version"):
            return {"status": "refused", "reasons": [
                "The firmware server has no version.json, so the result could not be verified."]}
        steps, problems = self._plan(check, can_ids, include_master)
        if problems or not steps:
            return {"status": "refused", "reasons": problems or ["Nothing to update."]}
        busy = self.guard.claim_can()
        if busy:
            return {"status": "refused", "reasons": [busy]}
        self.cancel_event.clear()
        self.state = {"state": "running", "server_version": check["server_version"], "steps": steps,
                      "current": None, "homing_required": False, "message": "Starting…",
                      "started": _now(), "finished": None}
        self.thread = threading.Thread(target=self._run, name="FirmwareUpdate", daemon=True)
        self.thread.start()
        return {"status": "started", "steps": steps}

    def cancel(self) -> dict:
        """Stop after the current board. A CAN transfer in progress is aborted
        (the node keeps its old firmware); a USB flash of the master is NOT
        interrupted — a half-written master needs a manual reflash."""
        if self.state.get("state") != "running":
            return {"status": "idle"}
        self.cancel_event.set()
        self.ota.cancel()
        self.state["message"] = "Cancelling after the current board…"
        return {"status": "cancelling"}

    @staticmethod
    def _plan(check, can_ids, include_master):
        by_id = {d.get("canId"): d for d in check["devices"] if d["connection"] == "can"}
        wanted = can_ids if can_ids is not None else [
            cid for cid, d in by_id.items() if d["update_status"] == "update_available"]
        steps, problems = [], []
        for cid in wanted:
            d = by_id.get(cid)
            if d is None or d["update_status"] == "unreachable":
                problems.append(f"CAN node {cid} is not on the bus.")
            elif not d["available_version"]:
                problems.append(f"No image for CAN node {cid} ({d['filename']}) on the server.")
            else:
                steps.append(_step(d))
        if include_master:
            usb = next((d for d in check["devices"] if d["connection"] == "usb"), None)
            if usb is None or not usb["available_version"]:
                problems.append("No image for the USB master on the server.")
            else:
                steps.append(_step(usb))
        return steps, problems

    def _run(self):
        """Background thread; holds guard.can (taken by start) throughout."""
        state = self.state
        try:
            state["message"] = "Preparing (lasers off)…"
            self.hooks.before_update()
            for index, step in enumerate(state["steps"]):
                if self.cancel_event.is_set():
                    break
                state.update(current=index, message=f"Updating {_label(step)}…")
                step["status"] = "running"
                ok, message = (self._update_can(step) if step["connection"] == "can"
                               else self._update_master(step))
                step["status"], step["message"] = ("done" if ok else "failed"), message
                if ok and step["deviceTypeStr"] == "motor":
                    state["homing_required"] = True
                    self.hooks.motor_restarted()
                if not ok:
                    break
            for step in state["steps"]:
                if step["status"] == "pending":
                    step["status"] = "skipped"
            failed = any(s["status"] == "failed" for s in state["steps"])
            if failed:
                final = "failed"
                message = "Stopped at the first failure; later boards were not touched."
            elif self.cancel_event.is_set():
                final, message = "cancelled", "Cancelled — remaining boards were not updated."
            else:
                final, message = "success", "All selected boards run the server's firmware."
        except Exception as e:  # never leave the state "running"
            self._log.error(f"Firmware update aborted: {e}", exc_info=True)
            final, message = "failed", f"Aborted: {e}"
        finally:
            self.guard.can.release()
        state.update(state=final, message=message, current=None, finished=_now())
        self._log.info(f"Firmware update finished: {final} — {message}")

    def _update_can(self, step):
        result = self.ota.upload(step["canId"], filename=step["filename"],
                                 expected_version=step["to_version"], cancel=self.cancel_event)
        return result.get("status") == "success", result.get("message", "")

    def _update_master(self, step):
        """esptool drops the host's link; run() restores it, then the version
        is read again over the new link."""
        if not self.guard.usb.acquire(blocking=False):
            return False, "A USB flash is already running."
        try:
            self.usb.cancel_event.clear()
            result = self.usb.run(port=(self.hooks.usb_info() or {}).get("serialport"), match="HAT",
                                  baud=921600, firmware_filename=step["filename"],
                                  reconnect_after=True, chip="auto", erase_flash=False,
                                  skip_disconnect=False)
        finally:
            self.guard.usb.release()
        if result.get("status") != "success":
            detail = f"{result.get('message')} {result.get('details') or ''}".strip()
            return False, f"USB flash: {detail}"
        time.sleep(2)
        installed = (self.hooks.usb_info() or {}).get("fwVersion") or (
            (self.bus.scan(timeout=5).get("master") or {}).get("fwVersion"))
        if installed != step["to_version"]:
            return False, f"Flashed, but the master reports {installed!r}."
        return True, f"Running {installed}"


def _step(device):
    return {"canId": device.get("canId"), "connection": device["connection"],
            "deviceTypeStr": device.get("deviceTypeStr"), "filename": device["filename"],
            "from_version": device.get("installed_version"),
            "to_version": device["available_version"], "status": "pending", "message": ""}


def _label(step):
    if step["connection"] == "usb":
        return "the USB master"
    return f"{step['deviceTypeStr']} {step['canId']}"


def _now():
    return datetime.datetime.now().isoformat(timespec="seconds")
