"""UC2ConfigController endpoints for the CAN network and its firmware.

Mixed into UC2ConfigController, so every route stays
/UC2ConfigController/<name> with its parameters and response shape. The work
is done by imswitch.imcontrol.model.canbus (independent of ImSwitch); this
file only adds what ImSwitch knows: the serial link to the master, when the
hardware is busy, switching lasers off, re-arming the homing prompt, the
setup-JSON options and the optional check after connect.
"""
import tempfile
import threading
from pathlib import Path

from imswitch.imcommon.model import APIExport
from imswitch.imcontrol.model.canbus import CanNetwork, UpdateHooks, device_mapping
# The module, not the package attribute: imswitch.imcontrol.model re-exports the
# SetupInfo *class* under the module's name.
from imswitch.imcontrol.model.SetupInfo import UC2ConfigInfo

DEFAULT_FIRMWARE_URL = "http://host.docker.internal/firmware"
# Seconds after startup before the opt-in firmware check: lets the serial
# link, the CAN nodes' boot-up and the HTTP server settle.
STARTUP_CHECK_DELAY_S = 20
BUSY_STATES = ("running", "paused", "stopping")


class _ManagerLink:
    """ImSwitch's own serial connection to the master (UC2ConfigManager), as
    CanNetwork needs it: esptool must have the port to itself."""

    def __init__(self, manager, logger):
        self._manager, self._log = manager, logger

    def release(self):
        for step in ("interruptSerialCommunication", "closeSerial"):
            try:
                getattr(self._manager, step)()
            except Exception as e:
                self._log.warning(f"{step} before USB flashing failed (non-fatal): {e}")

    def restore(self):
        self._log.info("Reconnecting ImSwitch to master after flashing…")
        self._manager.initSerial(baudrate=None)

    def current_port(self):
        return getattr(self._manager, "serialport", None)


class CanNetworkApiMixin:

    def _init_can_network(self):
        manager = self._master.UC2ConfigManager
        uc2 = getattr(self._setupInfo, "uc2Config", None)
        self._can_network = CanNetwork(
            client=lambda: getattr(manager, "ESP32", None),
            firmware_url=getattr(uc2, "firmwareServerUrl", None) or DEFAULT_FIRMWARE_URL,
            cache_dir=Path(tempfile.gettempdir()) / "uc2_ota_firmware_cache",
            link=_ManagerLink(manager, self._logger),
            hooks=UpdateHooks(usb_info=self.getFirmwareInfo, blockers=self._hardware_busy_reasons,
                              before_update=self._lasers_off, motor_restarted=self._rearm_homing),
            emit_ota=self.sigOTAStatusUpdate.emit, emit_usb=self.sigUSBFlashStatusUpdate.emit,
            logger=self._logger)
        self._fw_prompt = None  # result of the check after connect
        self._register_can_scan_callback()
        if self._check_firmware_on_connect():
            timer = threading.Timer(STARTUP_CHECK_DELAY_S, self._startup_firmware_check)
            timer.daemon = True
            timer.start()

    def _register_can_scan_callback(self):
        """Forward scan results the firmware pushes to sigUpdateCANDevices."""
        can = getattr(getattr(self._master.UC2ConfigManager, "ESP32", None), "can", None)
        if can is None:
            self._logger.warning("ESP32 CAN not available - CAN scan callbacks won't work")
            return
        try:
            can.register_callback(0, self._commChannel.sigUpdateCANDevices.emit)
        except Exception as e:
            self._logger.error(f"Could not register CAN callback: {e}")

    # ── bus ─────────────────────────────────────────────────────────────────

    @APIExport(runOnUIThread=False)
    def scan_canbus(self, timeout: int = 5, probe_range: bool = False) -> dict:
        """Scan the CAN bus. Reachable nodes report build, fwVersion, fwImage
        and mac; the master's own identity is under "master". probe_range also
        probes ids 1..127 for unrouted / freshly flashed boards
        (deviceTypeStr "unrouted"). Returns {} without a CAN master."""
        return self._can_network.bus.scan(timeout=timeout, probe_range=probe_range)

    @APIExport(runOnUIThread=False)
    def get_canbus_devices(self, timeout: int = 2):
        """CAN ids of the nodes on the bus (from a scan)."""
        return self._can_network.bus.scan(timeout=timeout).get("detected_ids", [])

    @APIExport(runOnUIThread=False, requestType="POST")
    def reassignCANId(self, new_id: int, mac: str = None, target: int = None,
                      expect_mac: str = None, timeout: int = 5) -> dict:
        """Move a node to *new_id* without reflashing, identified by *mac*
        (preferred) or its current id *target* (guarded by *expect_mac*). It
        persists the id and reappears after ~0.3 s; re-scan to confirm."""
        return self._can_network.bus.reassign(new_id, mac=mac, target=target,
                                              expect_mac=expect_mac, timeout=timeout)

    @APIExport(runOnUIThread=True)
    def restartCANDevice(self, device_id=0):
        """Reboot CAN node *device_id* (0 = the master)."""
        self._can_network.bus.restart(device_id)

    # ── firmware server ─────────────────────────────────────────────────────

    @APIExport(runOnUIThread=False)
    def setOTAFirmwareServer(self, server_url=DEFAULT_FIRMWARE_URL):
        """Use *server_url* as firmware server (checked first). Not persisted —
        for startup use the setup JSON's uc2Config.firmwareServerUrl."""
        return self._can_network.server.set_url(server_url)

    @APIExport(runOnUIThread=False)
    def getOTAFirmwareServer(self):
        return {"firmware_server_url": self._can_network.server.url}

    @APIExport(runOnUIThread=False)
    def listAllFirmwareFiles(self):
        """Every .bin on the server: {status, firmware_server, server_version,
        server_commit_time, files: [{filename, size, mod_time, url, version, sha256}]}."""
        return self._can_network.server.files()

    @APIExport(runOnUIThread=False)
    def listAvailableFirmware(self, can_ids: list = None):
        """The fixed-role images on the server by CAN id: {status,
        firmware_server, server_version, firmware_count, firmware: {can_id:
        {filename, url, can_id, size, mod_time, version, sha256}}}."""
        return self._can_network.server.images_by_can_id(can_ids)

    @APIExport(runOnUIThread=False)
    def clearOTAFirmwareCache(self):
        return self._can_network.server.clear_cache()

    @APIExport(runOnUIThread=False)
    def getOTAFirmwareCacheStatus(self):
        return self._can_network.server.cache_status()

    # ── CAN streaming OTA ───────────────────────────────────────────────────

    @APIExport(runOnUIThread=False)
    def startCANStreamingOTA(self, can_id: int, firmware_url: str = None, baud: int = None):
        """Update one node over CAN (blocks until done): download + sha256
        check, stream with up to 3 attempts, wait for the node to report the
        new version. {"status": "busy"} while another upload or a USB flash
        uses the port. *firmware_url* is ignored; *baud* defaults to the link's.
        Progress on sigOTAStatusUpdate."""
        return self._can_network.ota.start(int(can_id), baud=baud)

    @APIExport(runOnUIThread=False, requestType="POST")
    def startMultipleCANStreamingOTA(self, can_ids: list[int], delay_between: int = 5):
        """startCANStreamingOTA for several nodes, one after another; returns
        when all are done (or {"status": "busy"})."""
        if not isinstance(can_ids, list):
            return {"status": "error", "message": "can_ids must be a list"}
        return self._can_network.ota.start_many(can_ids, delay_between=delay_between)

    @APIExport(runOnUIThread=False, requestType="POST")
    def cancelCANStreamingOTA(self):
        """Abort the running upload at the next chunk; the node keeps its old firmware."""
        return self._can_network.ota.cancel()

    @APIExport(runOnUIThread=False)
    def getOTAStatus(self, can_id=None):
        """Last CAN OTA status of one node or of all."""
        ota = self._can_network.ota
        with ota.status_lock:
            if can_id is None:
                return {"status": "success", "device_count": len(ota.status),
                        "devices": dict(ota.status)}
            status = ota.status.get(int(can_id))
        if status is None:
            return {"status": "error", "message": f"No OTA status available for device {can_id}"}
        return {"status": "success", "can_id": int(can_id), "ota_status": status}

    @APIExport(runOnUIThread=False)
    def clearOTAStatus(self, can_id=None):
        ota = self._can_network.ota
        with ota.status_lock:
            if can_id is None:
                count = len(ota.status)
                ota.status.clear()
                return {"status": "success", "message": f"Cleared OTA status for {count} devices"}
            if ota.status.pop(int(can_id), None) is None:
                return {"status": "error", "message": f"No OTA status found for device {can_id}"}
        return {"status": "success", "message": f"Cleared OTA status for device {can_id}"}

    @APIExport(runOnUIThread=False)
    def getOTADeviceMapping(self):
        """The fixed CAN-id roles: {"master": 1, "motors": {"A": 10, ...}, ...}."""
        return {"status": "success", "mapping": device_mapping(),
                "description": "CAN ID mapping for UC2 devices"}

    # ── USB flashing and bring-up ───────────────────────────────────────────

    @APIExport(runOnUIThread=False)
    def listSerialPorts(self):
        return self._can_network.usb.list_ports()

    @APIExport(runOnUIThread=False)
    def getUSBFlashStatus(self):
        return self._can_network.usb.status

    @APIExport(runOnUIThread=False, requestType="POST")
    def flashMasterFirmwareUSB(self, port: str | None = None, match: str = "HAT",
                               baud: int = 921600, firmware_filename: str | None = None,
                               reconnect_after: bool = True,
                               chip: str = "auto", erase_flash: bool = False,
                               skip_disconnect: bool = False):
        """Flash a board over USB with esptool (blocks until done).

        port: explicit serial port, else found by *match* in the port metadata.
        firmware_filename: image on the server, else the master image.
        reconnect_after: reconnect ImSwitch to the master afterwards.
        chip: "auto" (from the USB VID:PID), "esp32", "esp32s3", "esp32s2", "esp32c3".
        erase_flash: erase first (merged images only).
        skip_disconnect: keep ImSwitch's link open (flashing another board, e.g. a XIAO).
        Refused ({"status": "busy"}) during a CAN OTA upload. Progress on
        sigUSBFlashStatusUpdate."""
        return self._can_network.usb.flash(
            port=port, match=match, baud=baud, firmware_filename=firmware_filename,
            reconnect_after=reconnect_after, chip=chip, erase_flash=erase_flash,
            skip_disconnect=skip_disconnect)

    @APIExport(runOnUIThread=False, requestType="POST")
    def cancelUSBFlash(self):
        """Cancel a running USB flash ({"status": "idle"} if none)."""
        return self._can_network.usb.cancel_request()

    @APIExport(runOnUIThread=False, requestType="POST")
    def sendCanAddress(self, port: str = "", address: int = 1, baud: int = 115200,
                       timeout: float = 2.0, skip_flash: bool = False):
        """Assign CAN *address* to a freshly flashed board on its own serial
        *port*; detects boot loops and checks /state_get afterwards.
        *skip_flash* is accepted for compatibility and ignored."""
        return self._can_network.usb.send_can_address(port, address=address, baud=baud,
                                                      timeout=timeout)

    @APIExport(runOnUIThread=False)
    def probeDeviceState(self, port: str = "", baud: int = 115200, timeout: float = 2.0):
        """/state_get on a board's own serial port: is the firmware running?"""
        return self._can_network.usb.probe_state(port, baud=baud, timeout=timeout)

    @APIExport(runOnUIThread=False)
    def getRecommendedFirmware(self, port: str = "", baud: int = 115200, timeout: float = 2.0):
        """Which image on the firmware server fits a board, from what its
        firmware reports over serial (/state_get: fwImage, pindef, CAN id).

        port: empty, or ImSwitch's own port = the board ImSwitch is connected
        to, read over the open link. Any other port is opened and probed,
        which may reset that board; refused ({"status": "busy"}) while a
        flash or CAN OTA runs. Read-only otherwise: nothing is flashed.
        Returns {status, source: "imswitch"|"port", port, chip, identity:
        {fwImage, fwVersion, pindef, isMaster, canId, ...}, firmware_server,
        server_version, recommended: {filename, merged, source, reason,
        candidates: [{filename, source, on_server}], file}}; recommended.filename
        is None when no candidate is on the server."""
        usb = self._can_network.usb
        own_port = usb.link.current_port()
        if not port or port == own_port:
            identity = self.getFirmwareInfo() or {}
            if identity.get("status") == "error" or not identity.get("connected"):
                return {"status": "error", "source": "imswitch", "port": own_port,
                        "message": "ImSwitch is not connected to a board. Pick the board's "
                                   "serial port to probe it directly."}
            if identity.get("isMaster") and not identity.get("canId"):
                identity = {**identity, "canId": 1}
            found = {"status": "success", "source": "imswitch", "port": own_port,
                     "chip": usb.detect_chip(own_port) if own_port else None,
                     "identity": identity}
        else:
            found = usb.identify(port, baud=baud, timeout=timeout)
            if found.get("status") != "success":
                return {**found, "source": "port"}
            found["source"] = "port"
        return {**found, **self._can_network.server.recommend(found["identity"])}

    @APIExport(runOnUIThread=False, requestType="POST")
    def testDeviceAction(self, port: str = "", device_type: str = "motor", baud: int = 115200,
                         timeout: float = 2.0, stepperid: int = 1, speed: int = 2000,
                         position: int = 1000, isabs: int = 0, r: int = 25, g: int = 25,
                         b: int = 25, led_action: str = "fill", laserid: int = 1,
                         laserval: int = 118):
        """Make a freshly flashed slave on its own serial *port* do something
        visible: device_type "motor" (moves!), "ledarray" or "laser"."""
        return self._can_network.usb.test_action(
            port, device_type=device_type, baud=baud, timeout=timeout, stepperid=stepperid,
            speed=speed, position=position, isabs=isabs, r=r, g=g, b=b, led_action=led_action,
            laserid=laserid, laserval=laserval)

    # ── versions and the prompted update ────────────────────────────────────

    @APIExport(runOnUIThread=False)
    def checkFirmwareUpdates(self, timeout: int = 5, probe_range: bool = False) -> dict:
        """Installed vs firmware-server version per board (read-only):
        {status, firmware_server, server_version, server_commit_time,
        updates_available, devices: [{canId, deviceTypeStr, connection
        ("usb"|"can"), installed_version, build, mac, filename, image_source,
        available_version, update_status}]}. update_status: up_to_date |
        update_available | device_newer | unknown | no_firmware | unreachable."""
        return self._can_network.updater.check(timeout=timeout, probe_range=probe_range)

    @APIExport(runOnUIThread=False, requestType="POST")
    def startFirmwareUpdate(self, can_ids: list[int] = None, include_master: bool = False) -> dict:
        """Update boards to the server's version, unattended (background).

        Refused ({"status": "refused", "reasons"}) while an experiment,
        recording, workflow or timelapse runs, a stage or the objective moves,
        another OTA/USB flash runs, or the firmware server is unreachable.
        Otherwise: lasers off → each CAN node (download + sha256 check, OTA,
        re-scan until it reports the new version) → the USB master last
        (esptool, reconnect, re-read). Stops at the first failure; updated
        motors need homing. A server without version.json can still be used:
        boards must then be listed explicitly and are not verified.

        :param can_ids: nodes to update (body: JSON array; default: every
                        node whose status is update_available)
        :param include_master: also flash the USB-connected master, last
        """
        return self._can_network.updater.start(can_ids=can_ids, include_master=include_master)

    @APIExport(runOnUIThread=False)
    def getFirmwareUpdateStatus(self) -> dict:
        """{state: idle|running|success|failed|cancelled, server_version,
        message, current, homing_required, started, finished, steps: [{canId,
        connection, deviceTypeStr, filename, from_version, to_version, status:
        pending|running|done|failed|skipped, message}]}."""
        return self._can_network.updater.state

    @APIExport(runOnUIThread=False, requestType="POST")
    def cancelFirmwareUpdate(self) -> dict:
        """Stop after the current board (a USB flash of the master is not interrupted)."""
        return self._can_network.updater.cancel()

    # ── ImSwitch hooks of the update ────────────────────────────────────────

    def _hardware_busy_reasons(self) -> list:
        """Why an unattended update must not start now (empty = go). A check
        that fails must not block every update, so it only logs."""
        get = getattr(self._master, "getController", lambda name: None)
        experiment, recording, objective = get("Experiment"), get("Recording"), get("Objective")
        checks = [("A stage is moving or homing.", self._stage_moving)]
        if experiment is not None:
            checks.append(("An experiment is running.", lambda: (
                experiment.getExperimentStatus().get("status") in BUSY_STATES
                or getattr(experiment, "_experiment_starting", False))))
        if recording is not None:
            checks.append(("A recording is running.", recording.isRecording))
        for name in ("Workflow", "Timelapse"):
            manager = getattr(get(name), "workflow_manager", None)
            if manager is not None:
                checks.append((f"A {name.lower()} is running.", lambda m=manager: (
                    m.get_status().get("status") in BUSY_STATES)))
        if objective is not None:
            checks.append(("The objective turret is moving.",
                           lambda: getattr(objective, "_isMovingObjective", False)))
        reasons = []
        for label, predicate in checks:
            try:
                if predicate():
                    reasons.append(label)
            except Exception as e:
                self._logger.warning(f"Firmware update pre-check '{label}' failed: {e}")
        return reasons

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
        """Before boards reboot: every laser off, through LaserController when
        present so the UI state follows."""
        manager = getattr(self._master, "lasersManager", None)
        if manager is None:
            return
        laser_controller = getattr(self._master, "getController", lambda name: None)("Laser")
        for name in manager.getAllDeviceNames():
            try:
                if laser_controller is not None and hasattr(laser_controller, "setLaserActive"):
                    laser_controller.setLaserActive(name, False)
                else:
                    manager[name].setEnabled(False)
            except Exception as e:
                self._logger.warning(f"Could not switch laser {name} off: {e}")

    def _rearm_homing(self):
        """A motor node rebooted and forgot its position: the homing prompt
        appears again (the frontend reads it on reconnect)."""
        positioner = getattr(self._master, "getController", lambda name: None)("Positioner")
        if positioner is not None:
            positioner._hasHomedSinceStartup = False
            positioner._homingRecommendationDismissed = False

    # ── optional check after connect ────────────────────────────────────────

    def _check_firmware_on_connect(self) -> bool:
        return bool(getattr(getattr(self._setupInfo, "uc2Config", None),
                            "checkFirmwareOnConnect", False))

    def _startup_firmware_check(self):
        """Opt-in (uc2Config.checkFirmwareOnConnect), once after startup and
        only while idle: offer the update when boards are outdated. Read-only."""
        try:
            if not self._master.UC2ConfigManager.isConnected():
                return
            if self._can_network.updater.blockers():
                self._logger.info("Skipping the firmware check after connect: hardware busy")
                return
            result = self.checkFirmwareUpdates()
            if result.get("updates_available"):
                self._fw_prompt = result
                self.sigFirmwareUpdatesAvailable.emit(result)
                self._logger.info(f"{result['updates_available']} board(s) can be updated to "
                                  f"{result.get('server_version')}")
        except Exception as e:
            self._logger.warning(f"Firmware check after connect failed: {e}")

    @APIExport(runOnUIThread=False)
    def getFirmwareUpdatePrompt(self) -> dict:
        """Result of the check after connect when it found updates, else
        {"updates_available": 0}. The frontend reads it on connect
        (sigFirmwareUpdatesAvailable is lost if no browser was open)."""
        return self._fw_prompt or {"updates_available": 0}

    @APIExport(runOnUIThread=False)
    def getFirmwareCheckOnConnect(self) -> dict:
        """Whether ImSwitch compares the boards with the firmware server after startup."""
        return {"enabled": self._check_firmware_on_connect()}

    @APIExport(runOnUIThread=False)
    def setFirmwareCheckOnConnect(self, enabled: bool = True) -> dict:
        """Enable/disable the check after startup; saved in the setup JSON as
        uc2Config.checkFirmwareOnConnect, effective on the next start. Returns
        {"enabled"} or {"status": "error", "message", "enabled": <unchanged>}."""
        import imswitch.imcontrol.model.configfiletools as configfiletools
        try:
            if self._setupInfo.uc2Config is None:
                self._setupInfo.uc2Config = UC2ConfigInfo()
            self._setupInfo.uc2Config.checkFirmwareOnConnect = bool(enabled)
            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)
        except Exception as e:
            self._logger.error(f"Could not save checkFirmwareOnConnect: {e}", exc_info=True)
            return {"status": "error", "message": f"Could not save the setting: {e}",
                    "enabled": self._check_firmware_on_connect()}
        return {"enabled": bool(enabled)}
