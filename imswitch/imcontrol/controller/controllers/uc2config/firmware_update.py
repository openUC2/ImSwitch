"""Firmware versions, update checks and the prompted update of all boards.

Mixed into UC2ConfigController, so the routes stay /UC2ConfigController/...
The OTA primitives it drives (_can_streaming_ota, _do_flash, the download
helpers, the locks) live in UC2ConfigController.

The uc2-esp32 frame build bakes one release string (UC2_FW_VERSION, e.g.
"v2026.0.0-beta.4-6-g2108590-t20260930092622") and the image name it is
published as (UC2_FW_IMAGE) into every image, and publishes
<server>/version.json. Boards report both: the USB board via /state_get
(uc2rest "fwVersion"/"fwImage"), CAN nodes via the bus scan (OD 0x2500/0x2501).
Equal version strings = same build. See docs/FIRMWARE_VERSIONING.md.
"""
import datetime
import hashlib
import re
import threading
import time

import requests

from imswitch.imcommon.model import APIExport

# Seconds after startup before the opt-in check runs: lets the serial link,
# the CAN nodes' boot-up and the HTTP server settle.
STARTUP_CHECK_DELAY_S = 20
# A slave reboots 2 s after a successful OTA; give it time to go down before
# asking for its version, then this long to come back with the new one.
VERIFY_SETTLE_S = 4
VERIFY_TIMEOUT_S = 40
BUSY_EXPERIMENT_STATES = ("running", "paused", "stopping")


class FirmwareUpdateMixin:
    """Version check, update planning/orchestration and the on-connect prompt."""

    _FW_TIMESTAMP = re.compile(r"-t(\d{14})(?:-|$)")

    def _init_firmware_update(self):
        self._fw_update_state = {"state": "idle", "steps": []}
        self._fw_update_cancel = threading.Event()
        self._fw_update_thread = None
        self._fw_prompt = None  # result of the opt-in check after connect
        if self._check_firmware_on_connect():
            timer = threading.Timer(STARTUP_CHECK_DELAY_S, self._startup_firmware_check)
            timer.daemon = True
            timer.start()

    # ── server manifest ──────────────────────────────────────────────────────

    def _fetch_firmware_manifest(self) -> dict:
        """version.json from the firmware server: {version, commit, commit_time,
        run_url, files: {filename: {size, sha256}}}. Empty dict when the server
        has none (images built before it existed) or is unreachable."""
        server_url = (self._firmware_server_url or "").rstrip("/")
        if not server_url:
            return {}
        try:
            response = requests.get(f"{server_url}/version.json", timeout=5)
            response.raise_for_status()
            manifest = response.json()
            return manifest if isinstance(manifest, dict) else {}
        except (requests.exceptions.RequestException, ValueError) as e:
            self._logger.debug(f"No firmware manifest at {server_url}/version.json: {e}")
            return {}

    @staticmethod
    def _firmware_file_version(manifest: dict, filename: str):
        """Release string of *filename* on the server; None if version.json does not list it."""
        if filename and filename in manifest.get("files", {}):
            return manifest.get("version") or None
        return None

    def _verify_firmware_download(self, path, filename: str, manifest: dict = None) -> bool:
        """Compare a downloaded image with its version.json entry (size + sha256).
        True when it matches or the server does not list the file (no manifest:
        nothing to check against). A mismatch is logged and the file deleted."""
        manifest = self._fetch_firmware_manifest() if manifest is None else manifest
        entry = manifest.get("files", {}).get(filename)
        if not entry:
            self._logger.warning(f"{filename} is not in version.json — download not verified")
            return True
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) == entry.get("size", len(data)) and digest == entry.get("sha256"):
            return True
        self._logger.error(
            f"Download of {filename} does not match version.json "
            f"(size {len(data)} vs {entry.get('size')}, sha256 {digest[:12]}… vs "
            f"{str(entry.get('sha256'))[:12]}…) — deleted")
        path.unlink(missing_ok=True)
        return False

    @classmethod
    def _firmware_update_status(cls, installed: str, available: str) -> str:
        """Compare a board's version with the server's. Firmware older than
        version reporting says "UC2-ESP v2.0" or nothing, which never matches, so
        it shows as update_available. "device_newer" only when both strings carry
        a -t<timestamp> (non-tag builds) and the board's is later — e.g. a
        developer's local build, which should not be offered a downgrade."""
        if not available:
            return "unknown"
        if installed == available:
            return "up_to_date"
        t_inst = cls._FW_TIMESTAMP.search(installed or "")
        t_avail = cls._FW_TIMESTAMP.search(available)
        if t_inst and t_avail and t_inst.group(1) > t_avail.group(1):
            return "device_newer"
        return "update_available"

    # ── version check ────────────────────────────────────────────────────────

    @APIExport(runOnUIThread=False)
    def checkFirmwareUpdates(self, timeout: int = 5, probe_range: bool = False) -> dict:
        """
        Compare the firmware on every connected board with the firmware server.

        Read-only: reads <server>/version.json, the USB board's identity
        (/state_get) and — on a CAN master — every node's version and image
        from a bus scan (SDO reads of OD 0x2500/0x2501). Nothing is flashed.

        Each board is matched to the image it reports it was built as
        (image_source "reported"); boards built before that was reported fall
        back to the pindef (USB) or the CAN-id table (image_source "mapping").

        :param timeout: CAN scan timeout in seconds
        :param probe_range: also probe CAN ids 1..127 for unrouted nodes (slower)
        :return: {status, firmware_server, server_version, server_commit_time,
                  updates_available, devices: [{canId, deviceTypeStr,
                  connection ("usb"|"can"), installed_version, build, mac,
                  filename, image_source, available_version, update_status}]}
                  update_status is one of "up_to_date", "update_available",
                  "device_newer", "unknown" (server has no version.json),
                  "no_firmware" (no image for this board on the server),
                  "unreachable" (node did not answer the scan).
        """
        manifest = self._fetch_firmware_manifest()
        manifest_files = manifest.get("files", {})
        can_map = self._get_can_id_firmware_mapping()

        def entry(installed, filename, source, reachable=True, **fields):
            available = self._firmware_file_version(manifest, filename)
            if not reachable:
                status = "unreachable"
            elif manifest_files and filename not in manifest_files:
                status = "no_firmware"
            else:
                status = self._firmware_update_status(installed, available)
            return {**fields, "installed_version": installed or None,
                    "filename": filename, "image_source": source,
                    "available_version": available, "update_status": status}

        usb = self.getFirmwareInfo() or {}
        pindef = str(usb.get("pindef") or "")
        is_can_master = bool(usb.get("isMaster")) or "can" in pindef.lower()
        scan = self.scan_canbus(timeout=timeout, probe_range=probe_range) if is_can_master else {}
        master = scan.get("master") or {}

        devices = []
        if usb.get("connected") or master:
            reported = usb.get("fwImage") or master.get("fwImage")
            if reported:
                usb_file, source = reported, "reported"
            else:
                # Built before the image was reported: named after the pindef
                # (= PlatformIO env); a CAN master is also in the CAN-id table.
                candidates = ([f"esp32_{pindef}_release.bin", f"esp32_{pindef}.bin"]
                              if pindef else [])
                if master.get("canId") in can_map:
                    candidates.append(can_map[master["canId"]])
                usb_file = next((c for c in candidates if c in manifest_files),
                                candidates[0] if candidates else None)
                source = "mapping"
            devices.append(entry(usb.get("fwVersion") or master.get("fwVersion"), usb_file, source,
                                 canId=master.get("canId"), deviceTypeStr=pindef or "usb",
                                 connection="usb", build=usb.get("date") or master.get("build"),
                                 mac=master.get("mac")))

        for node in scan.get("scan", []):
            filename = node.get("fwImage") or can_map.get(node.get("canId"))
            devices.append(entry(node.get("fwVersion"), filename,
                                 "reported" if node.get("fwImage") else "mapping",
                                 reachable=node.get("statusStr") != "unreachable",
                                 canId=node.get("canId"), deviceTypeStr=node.get("deviceTypeStr"),
                                 connection="can", build=node.get("build"), mac=node.get("mac")))

        return {
            "status": "success",
            "firmware_server": self._firmware_server_url,
            "server_version": manifest.get("version"),
            "server_commit_time": manifest.get("commit_time"),
            "updates_available": sum(d["update_status"] == "update_available" for d in devices),
            "devices": devices,
        }

    def _wait_for_node_version(self, can_id: int, expected: str, timeout=VERIFY_TIMEOUT_S):
        """After an OTA: wait until CAN node *can_id* answers the bus scan with
        *expected*. Returns (ok, last_seen_version). The node reports its old
        version until its deferred reboot, so a match only counts after
        VERIFY_SETTLE_S."""
        time.sleep(VERIFY_SETTLE_S)
        deadline = time.time() + timeout
        seen = None
        while time.time() < deadline and not self._fw_update_cancel.is_set():
            scan = self.scan_canbus(timeout=5) or {}
            node = next((n for n in scan.get("scan", []) if n.get("canId") == can_id), {})
            seen = node.get("fwVersion") or seen
            if node.get("statusStr") != "unreachable" and node.get("fwVersion") == expected:
                return True, seen
            time.sleep(2)
        return False, seen

    # ── orchestrated update ──────────────────────────────────────────────────

    def _firmware_update_blockers(self) -> list:
        """Reasons an unattended update must not start now (empty = go)."""
        reasons = [label for label, busy in (
            ("A CAN OTA upload is running.", self._can_ota_lock.locked()),
            ("A USB flash is running.", self._usb_flash_lock.locked())) if busy]
        for label, predicate in self._busy_checks():
            try:
                if predicate():
                    reasons.append(label)
            except Exception as e:  # a broken check must not block every update
                self._logger.warning(f"Firmware update pre-check '{label}' failed: {e}")
        return reasons

    def _busy_checks(self):
        """(reason, predicate) for everything that must be idle during an
        update. Controllers missing from this setup are skipped."""
        get = getattr(self._master, "getController", lambda name: None)
        checks = [("A stage is moving or homing.", self._stage_moving)]
        experiment, recording, objective = get("Experiment"), get("Recording"), get("Objective")
        if experiment is not None:
            checks.append(("An experiment is running.", lambda: (
                experiment.getExperimentStatus().get("status") in BUSY_EXPERIMENT_STATES
                or getattr(experiment, "_experiment_starting", False))))
        if recording is not None:
            checks.append(("A recording is running.", recording.isRecording))
        for name in ("Workflow", "Timelapse"):
            manager = getattr(get(name), "workflow_manager", None)
            if manager is not None:
                checks.append((f"A {name.lower()} is running.", lambda m=manager: (
                    m.get_status().get("status") in BUSY_EXPERIMENT_STATES)))
        if objective is not None:
            checks.append(("The objective turret is moving.",
                           lambda: getattr(objective, "_isMovingObjective", False)))
        return checks

    def _stage_moving(self) -> bool:
        """Frame homing, or a motor the firmware reports busy (/motor_get isbusy)."""
        manager = getattr(self._master, "positionersManager", None)
        for name in (manager.getAllDeviceNames() if manager else []):
            positioner = manager[name]
            if getattr(positioner, "isFrameHomingActive", lambda: False)():
                return True
            motor = getattr(positioner, "_motor", None)
            if motor is not None and hasattr(motor, "isBusy") and motor.isBusy(None):
                return True
        return False

    def _lasers_off(self):
        """Switch every laser off before boards reboot (a laser node restarting
        mid-exposure must not come back into an undefined state)."""
        manager = getattr(self._master, "lasersManager", None)
        if manager is None:
            return
        laser_controller = getattr(self._master, "getController", lambda name: None)("Laser")
        for name in manager.getAllDeviceNames():
            try:
                if laser_controller is not None and hasattr(laser_controller, "setLaserActive"):
                    laser_controller.setLaserActive(name, False)  # also updates the UI state
                else:
                    manager[name].setEnabled(False)
            except Exception as e:
                self._logger.warning(f"Could not switch laser {name} off: {e}")

    def _publish_fw_update(self, **changes):
        self._fw_update_state.update(changes)

    @APIExport(runOnUIThread=False, requestType="POST")
    def startFirmwareUpdate(self, can_ids: list[int] = None, include_master: bool = False) -> dict:
        """
        Update outdated boards to the firmware server's version, unattended.

        Refused (nothing started) while an experiment, recording, workflow or
        timelapse runs, a stage or the objective turret moves, another OTA/USB
        flash runs, or when the server has no version.json (the result could
        not be verified). Then, in a background thread:
        lasers off → each selected CAN node one at a time (download, sha256
        check against version.json, CAN streaming OTA, re-scan until the node
        reports the new version) → the USB master last (esptool; ImSwitch
        reconnects and re-reads the version). Stops at the first failure.
        Updated motor nodes have lost their position: "homing_required" is set.

        :param can_ids: CAN nodes to update (default: every node whose status
                        is update_available). Body: JSON array.
        :param include_master: also flash the USB-connected master (last).
        :return: {"status": "started", "steps": [...]} or
                 {"status": "refused", "reasons": [...]}; progress via
                 getFirmwareUpdateStatus.
        """
        if self._fw_update_state.get("state") == "running":
            return {"status": "refused", "reasons": ["A firmware update is already running."]}
        reasons = self._firmware_update_blockers()
        if reasons:
            return {"status": "refused", "reasons": reasons}

        check = self.checkFirmwareUpdates()
        if not check.get("server_version"):
            return {"status": "refused", "reasons": [
                "The firmware server has no version.json, so the result could not be verified."]}
        steps, problems = self._plan_firmware_update(check, can_ids, include_master)
        if problems or not steps:
            return {"status": "refused", "reasons": problems or ["Nothing to update."]}

        busy = self._acquire_can_ota_lock()
        if busy:
            return {"status": "refused", "reasons": [busy["message"]]}
        self._fw_update_cancel.clear()
        self._fw_update_state = {
            "state": "running", "server_version": check["server_version"], "steps": steps,
            "current": None, "homing_required": False, "message": "Starting…",
            "started": datetime.datetime.now().isoformat(timespec="seconds"), "finished": None,
        }
        self._fw_update_thread = threading.Thread(
            target=self._run_firmware_update, name="FirmwareUpdate", daemon=True)
        self._fw_update_thread.start()
        return {"status": "started", "steps": steps}

    def _plan_firmware_update(self, check, can_ids, include_master):
        """Steps (CAN nodes in the given order, then the USB master) and the
        problems that stop the whole update before anything is flashed."""
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
                steps.append(self._fw_step(d))
        if include_master:
            usb = next((d for d in check["devices"] if d["connection"] == "usb"), None)
            if usb is None or not usb["available_version"]:
                problems.append("No image for the USB master on the server.")
            else:
                steps.append(self._fw_step(usb))
        return steps, problems

    @staticmethod
    def _fw_step(device: dict) -> dict:
        return {"canId": device.get("canId"), "connection": device["connection"],
                "deviceTypeStr": device.get("deviceTypeStr"), "filename": device["filename"],
                "from_version": device.get("installed_version"),
                "to_version": device["available_version"], "status": "pending", "message": ""}

    def _run_firmware_update(self):
        """Background thread of startFirmwareUpdate. Holds the CAN OTA lock
        (acquired by the caller) for the whole run."""
        state = self._fw_update_state
        try:
            self._publish_fw_update(message="Switching lasers off…")
            self._lasers_off()
            for index, step in enumerate(state["steps"]):
                if self._fw_update_cancel.is_set():
                    break
                self._publish_fw_update(current=index,
                                        message=f"Updating {self._fw_step_label(step)}…")
                step["status"] = "running"
                ok, message = (self._fw_update_can_node(step) if step["connection"] == "can"
                               else self._fw_update_usb_master(step))
                step["status"], step["message"] = ("done" if ok else "failed"), message
                if ok and step["deviceTypeStr"] == "motor":
                    self._mark_homing_required()
                if not ok:
                    break
            for step in state["steps"]:
                if step["status"] == "pending":
                    step["status"] = "skipped"
            failed = any(s["status"] == "failed" for s in state["steps"])
            if self._fw_update_cancel.is_set() and not failed:
                final, message = "cancelled", "Cancelled — remaining boards were not updated."
            elif failed:
                final = "failed"
                message = "Stopped at the first failure; later boards were not touched."
            else:
                final, message = "success", "All selected boards run the server's firmware."
        except Exception as e:  # never leave the state "running"
            self._logger.error(f"Firmware update aborted: {e}", exc_info=True)
            final, message = "failed", f"Aborted: {e}"
        finally:
            self._can_ota_lock.release()
        self._publish_fw_update(state=final, message=message, current=None,
                                finished=datetime.datetime.now().isoformat(timespec="seconds"))
        self._logger.info(f"Firmware update finished: {final} — {message}")

    @staticmethod
    def _fw_step_label(step):
        if step["connection"] == "usb":
            return "the USB master"
        return f"{step['deviceTypeStr'] or 'node'} {step['canId']}"

    def _fw_update_can_node(self, step):
        """One CAN node: _can_streaming_ota downloads + verifies the image,
        streams it and waits for the node to report the new version."""
        result = self._can_streaming_ota(step["canId"], filename=step["filename"],
                                         expected_version=step["to_version"])
        return result.get("status") == "success", result.get("message", "")

    def _fw_update_usb_master(self, step):
        """The USB master, last: esptool drops the serial link; _do_flash
        reconnects, then the version is re-read over the new link."""
        if not self._usb_flash_lock.acquire(blocking=False):
            return False, "A USB flash is already running."
        try:
            self._usb_flash_cancel_event.clear()
            port = (self.getFirmwareInfo() or {}).get("serialport")
            result = self._do_flash(port=port, match="HAT", baud=921600,
                                    firmware_filename=step["filename"], reconnect_after=True,
                                    chip="auto", erase_flash=False, skip_disconnect=False)
        finally:
            self._usb_flash_lock.release()
        if result.get("status") != "success":
            details = f"{result.get('message')} {result.get('details') or ''}".strip()
            return False, f"USB flash: {details}"
        time.sleep(2)
        installed = (self.getFirmwareInfo() or {}).get("fwVersion")
        if not installed:  # uc2rest without fwVersion support: ask the bus scan
            installed = ((self.scan_canbus(timeout=5) or {}).get("master") or {}).get("fwVersion")
        if installed != step["to_version"]:
            return False, f"Flashed, but the master reports {installed!r}."
        return True, f"Running {installed}"

    def _mark_homing_required(self):
        """A motor node rebooted and forgot its position: re-arm the homing
        recommendation (the frontend reads it on reconnect) and report it."""
        self._publish_fw_update(homing_required=True)
        get = getattr(self._master, "getController", lambda name: None)
        positioner_controller = get("Positioner")
        if positioner_controller is not None:
            positioner_controller._hasHomedSinceStartup = False
            positioner_controller._homingRecommendationDismissed = False

    @APIExport(runOnUIThread=False)
    def getFirmwareUpdateStatus(self) -> dict:
        """State of the last/current startFirmwareUpdate run:
        {state: idle|running|success|failed|cancelled, server_version, message,
         current (step index), homing_required, started, finished,
         steps: [{canId, connection, deviceTypeStr, filename, from_version,
                  to_version, status: pending|running|done|failed|skipped, message}]}.
        Per-node transfer progress is on sigOTAStatusUpdate / sigUSBFlashStatusUpdate."""
        return self._fw_update_state

    @APIExport(runOnUIThread=False, requestType="POST")
    def cancelFirmwareUpdate(self) -> dict:
        """Stop the running update after the current board. A CAN transfer in
        progress is aborted (the node keeps its old firmware); a USB flash of
        the master is NOT interrupted — a half-written master needs a manual
        USB reflash."""
        if self._fw_update_state.get("state") != "running":
            return {"status": "idle"}
        self._fw_update_cancel.set()
        try:
            self._master.UC2ConfigManager.ESP32.canota.cancel_streaming_ota()
        except Exception as e:
            self._logger.debug(f"No CAN transfer to cancel: {e}")
        self._publish_fw_update(message="Cancelling after the current board…")
        return {"status": "cancelling"}

    # ── optional check after connect ─────────────────────────────────────────

    def _check_firmware_on_connect(self) -> bool:
        info = getattr(getattr(self, "_setupInfo", None), "uc2Config", None)
        return bool(getattr(info, "checkFirmwareOnConnect", False))

    def _startup_firmware_check(self):
        """Opt-in (uc2Config.checkFirmwareOnConnect): once after startup, compare
        the boards with the server and offer the update. Read-only."""
        try:
            if not self._master.UC2ConfigManager.isConnected():
                return
            if self._firmware_update_blockers():
                self._logger.info("Skipping the firmware check after connect: hardware busy")
                return
            result = self.checkFirmwareUpdates()
            if result.get("updates_available"):
                self._fw_prompt = result
                self.sigFirmwareUpdatesAvailable.emit(result)
                self._logger.info(f"{result['updates_available']} board(s) can be updated "
                                  f"to {result.get('server_version')}")
        except Exception as e:
            self._logger.warning(f"Firmware check after connect failed: {e}")

    @APIExport(runOnUIThread=False)
    def getFirmwareUpdatePrompt(self) -> dict:
        """Result of the check after connect when it found updates, else
        {"updates_available": 0}. The frontend reads it when it connects
        (sigFirmwareUpdatesAvailable is lost if no browser was open)."""
        return self._fw_prompt or {"updates_available": 0}

    @APIExport(runOnUIThread=False)
    def getFirmwareCheckOnConnect(self) -> dict:
        """Whether ImSwitch compares the boards with the firmware server after startup."""
        return {"enabled": self._check_firmware_on_connect()}

    @APIExport(runOnUIThread=False)
    def setFirmwareCheckOnConnect(self, enabled: bool = True) -> dict:
        """Enable/disable the firmware check after startup; persisted in the
        setup JSON as uc2Config.checkFirmwareOnConnect. Takes effect on the next start."""
        from imswitch.imcontrol.model import SetupInfo as setup_info_module
        from imswitch.imcontrol.model import configfiletools
        if self._setupInfo.uc2Config is None:
            self._setupInfo.uc2Config = setup_info_module.UC2ConfigInfo()
        self._setupInfo.uc2Config.checkFirmwareOnConnect = bool(enabled)
        options, _ = configfiletools.loadOptions()
        configfiletools.saveSetupInfo(options, self._setupInfo)
        return {"enabled": bool(enabled)}
