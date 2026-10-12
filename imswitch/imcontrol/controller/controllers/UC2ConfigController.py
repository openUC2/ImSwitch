import os
import threading
import datetime
from imswitch.imcommon.model import APIExport, initLogger, dirtools, ostools
from imswitch.imcommon.framework import Signal
from ..basecontrollers import ImConWidgetController
from .uc2config.can_network_api import CanNetworkApiMixin
import tifffile as tif


class UC2ConfigController(CanNetworkApiMixin, ImConWidgetController):
    """Linked to UC2ConfigWidget."""

    sigUC2SerialReadMessage = Signal(str)
    sigUC2SerialWriteMessage = Signal(str)
    sigUC2SerialIsConnected = Signal(bool)
    sigOTAStatusUpdate = Signal(object)  # CAN streaming OTA status (canbus.CanOta)
    sigUSBFlashStatusUpdate = Signal(object)  # USB flash status (canbus.UsbFlasher)
    sigCameraTrigger = Signal(object)  # Emits camera trigger events from hardware
    sigBusStatusUpdate = Signal(object)  # Emits CAN-bus power / emergency-stop status changes
    sigCollisionStatusUpdate = Signal(object)  # Emits collision-detector events/state (GPIO slave)
    sigPtzEvent = Signal(object)  # Emits PTZ keyboard key events + the action they triggered
    sigFirmwareUpdatesAvailable = Signal(object)  # checkFirmwareUpdates result of the check after connect


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        self._logger = self.__logger  # for the mixins (and the serial-callback error path below)

        try:
            self.stages = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]
        except Exception as e:
            self.__logger.error("No Stages found in the config file? ", e )
            self.stages = None

        #
        # register the callback to take a snapshot triggered by the ESP32
        self.registerCaptureCallback()

        # register camera trigger callback for performance mode
        self.registerCameraTriggerCallback()

        # register emergency-stop callback (CAN-bus power / E-stop button)
        self.registerEmergencyCallback()

        # collision detector (GPIO slave): latched crash state + optional
        # armed auto-stop of all motors. A crash also cuts bus power (see
        # collision_callback), so recovery requires the user to reset the
        # alarm (restores power) AND run a safe frame-homing before the stage
        # position can be trusted again — tracked by _collisionRequiresHoming.
        self._collisionArmed = False
        self._collisionLatched = False
        self._collisionRequiresHoming = False
        self._lastCollisionEvent = None
        self.registerCollisionCallback()

        # PTZ keyboard bridge (CAN node 61): map discrete key events (presets,
        # AUX, iris) to microscope functions. The action table (name->callable)
        # is fixed; the mapping (eventKey->action+params) is user-configurable
        # at runtime via the setPtz* API so bindings stay flexible.
        self._ptzActions = self._buildPtzActions()
        self._ptzMapping = self._defaultPtzMapping()
        self._lastPtzEvent = None
        self.registerPtzCallback()

        # register the callbacks for emitting serial-related signals
        if hasattr(self._master.UC2ConfigManager, "ESP32"):
            try:
                self._master.UC2ConfigManager.ESP32.serial.setWriteCallback(self.processSerialWriteMessage)
                self._master.UC2ConfigManager.ESP32.serial.setReadCallback(self.processSerialReadMessage)
            except Exception as e:
                self._logger.error(f"Could not register serial callbacks: {e}")

        # CAN network: bus scan, firmware (CAN OTA, USB flash), prompted
        # update — model/canbus, endpoints in uc2config/can_network_api.py
        self._init_can_network()

    def processSerialWriteMessage(self, message):
        self.sigUC2SerialWriteMessage.emit(message)

    def processSerialReadMessage(self, message):
        self.sigUC2SerialReadMessage.emit(message)

    def registerCaptureCallback(self):
        # This will capture an image based on a signal coming from the ESP32
        def snapImage(value):
            self.detector_names = self._master.detectorsManager.getAllDeviceNames()
            self.detector = self._master.detectorsManager[self.detector_names[0]]
            mImage = self.detector.getLatestFrame()
            # save image
            drivePath = dirtools.UserFileDirs.getValidatedDataPath()
            timeStamp = datetime.datetime.now().strftime("%Y_%m_%d")
            dirPath = os.path.join(drivePath, 'recordings', timeStamp)
            fileName  = "Snapshot_"+datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
            if not os.path.exists(dirPath):
                os.makedirs(dirPath)
            filePath = os.path.join(dirPath, fileName)
            self.__logger.debug(f"Saving image to {filePath}.tif")
            if mImage is not None:
                if mImage.ndim == 2:
                    tif.imwrite(filePath + ".tif", mImage)
                elif mImage.ndim == 3:
                    tif.imwrite(filePath + ".tif", mImage[0])
                else:
                    self.__logger.error("Image is not 2D or 3D")
            else:
                self.__logger.error("Image is None")

            # (detectorName, image, init, scale, isCurrentDetector)
            self._commChannel.sigUpdateImage.emit('Image', mImage, True, 1, False)

        def printCallback(value):
            self.__logger.debug(f"Callback called with value: {value}")
        try:
            self.__logger.debug("Registering callback for snapshot")
            # register default callback
            for i in range(1, self._master.UC2ConfigManager.ESP32.message.nCallbacks):
                self._master.UC2ConfigManager.ESP32.message.register_callback(i, printCallback)
            self._master.UC2ConfigManager.ESP32.message.register_callback(1, snapImage) # FIXME: Too hacky?

        except Exception as e:
            self.__logger.error(f"Could not register callback: {e}")



    def registerCameraTriggerCallback(self):
        """
        Register callback for camera trigger events from firmware.
        
        When the firmware sends {"cam":1}, this callback emits a signal
        that can be used by the ExperimentController for software-triggered
        acquisition in performance mode.
        """
        def camera_trigger_callback(trigger_info):
            """
            Handle camera trigger events from ESP32.
            
            :param trigger_info: Dictionary containing:
                - trigger: Trigger signal (1 = trigger)
                - frame_id: Frame number
                - timestamp: Unix timestamp
                - illumination: Optional illumination channel index
            """
            try:
                self.__logger.debug(f"Camera trigger received: frame {trigger_info.get('frame_id')}")
                
                # Emit signal for ExperimentController or other listeners
                self.sigCameraTrigger.emit(trigger_info)
                
            except Exception as e:
                self.__logger.error(f"Error in camera trigger callback: {e}")

        try:
            if hasattr(self._master.UC2ConfigManager, "ESP32") and hasattr(self._master.UC2ConfigManager.ESP32, "camera_trigger"):
                self._master.UC2ConfigManager.ESP32.camera_trigger.register_callback(0, camera_trigger_callback)
                self.__logger.debug("Camera trigger callback registered successfully")
            else:
                self.__logger.debug("ESP32 camera_trigger module not available - hardware camera triggers won't work")
        except Exception as e:
            self.__logger.error(f"Could not register camera trigger callback: {e}")

    def registerEmergencyCallback(self):
        """
        Register a callback for emergency-stop (E-stop) events.

        The CANopen master firmware pushes an unsolicited block when the
        hardware E-stop is pressed/released:
            {"emergency":{"active":1,"reason":"estop","msg":"..."}}
        We forward it to the frontend via sigBusStatusUpdate so the UI can
        warn the user and reflect that the CAN-bus is no longer available.
        """
        def emergency_callback(em):
            try:
                emergency_active = bool(em.get("active", 0))
                status = {
                    "emergencyActive": emergency_active,
                    # The high-current bus is unavailable while the E-stop is asserted.
                    "available": not emergency_active,
                    "reason": em.get("reason"),
                    "msg": em.get("msg"),
                    "timestamp": datetime.datetime.now().isoformat(),
                }
                self.sigBusStatusUpdate.emit(status)
            except Exception as e:
                self.__logger.error(f"Error in emergency_callback: {e}")

        try:
            registered = self._master.UC2ConfigManager.registerEmergencyCallback(emergency_callback)
            if registered:
                self.__logger.debug("Emergency-stop callback registered successfully")
            else:
                self.__logger.debug("Emergency-stop callback not registered (state module unavailable)")
        except Exception as e:
            self.__logger.error(f"Could not register emergency callback: {e}")

    def registerCollisionCallback(self):
        """
        Register a callback for collision-detector events from the GPIO slave.

        The CANopen master pushes an unsolicited frame on every collision
        trip/clear edge:
            {"gpio":{"event":1,"node":60,"trip":1,"estop":0,
                     "filtered":2365,"raw":2937,...}}
        On trip we latch the crash state (cleared only by resetCollisionAlarm)
        and — if armed via armCollisionProtection — immediately stop ALL
        motors. The event is always forwarded to the frontend via
        sigCollisionStatusUpdate.
        """
        def collision_callback(event):
            try:
                tripped = bool(event.get("trip", 0))
                self._lastCollisionEvent = event
                stopped = False
                if tripped:
                    # Latch: stays set until the user confirms the situation
                    # is cleared via resetCollisionAlarm(). A crash always
                    # invalidates the stage position → require safe homing.
                    self._collisionLatched = True
                    self._collisionRequiresHoming = True
                    if self._collisionArmed:
                        # Cut bus power AND stop motors immediately.
                        self.setBusPower(enable=False)
                        stopped = self._stopAllMotors()
                        self.__logger.warning(
                            f"COLLISION detected (filtered={event.get('filtered')}) — "
                            f"bus power cut, motors stopped: {stopped}")
                    else:
                        self.__logger.warning(
                            f"COLLISION detected (filtered={event.get('filtered')}) — "
                            "auto-stop not armed")
                self.sigCollisionStatusUpdate.emit(self._collisionStateDict(
                    event=event, motorsStopped=stopped))
            except Exception as e:
                self.__logger.error(f"Error in collision_callback: {e}")

        try:
            registered = self._master.UC2ConfigManager.registerCollisionCallback(collision_callback)
            if registered:
                self.__logger.debug("Collision callback registered successfully")
            else:
                self.__logger.debug("Collision callback not registered (gpio module unavailable)")
        except Exception as e:
            self.__logger.error(f"Could not register collision callback: {e}")

    def _stopAllMotors(self):
        """Immediately stop every positioner. Returns True if at least one
        stop command went out."""
        anyStopped = False
        try:
            for name in self._master.positionersManager.getAllDeviceNames():
                try:
                    p = self._master.positionersManager[name]
                    if hasattr(p, "stopAll"):
                        p.stopAll()
                    elif hasattr(p, "forceStop"):
                        for ax in getattr(p, "axes", []):
                            p.forceStop(ax)
                    else:
                        continue
                    anyStopped = True
                except Exception as e:
                    self.__logger.error(f"Could not stop positioner {name}: {e}")
        except Exception as e:
            self.__logger.error(f"_stopAllMotors failed: {e}")
        return anyStopped

    def _collisionStateDict(self, event=None, motorsStopped=False):
        return {
            "trip": bool((event or self._lastCollisionEvent or {}).get("trip", 0)),
            "latched": self._collisionLatched,
            "armed": self._collisionArmed,
            "requiresHoming": self._collisionRequiresHoming,
            "motorsStopped": motorsStopped,
            "event": event or self._lastCollisionEvent,
            "timestamp": datetime.datetime.now().isoformat(),
        }

    # ══════════════════════════════════════════════════════════════════════
    # PTZ keyboard bridge (CAN node 61) → microscope-function mapping
    #
    # The keyboard bridge forwards discrete keys to the master, which pushes
    #   {"ptz":{"event":1,"name":"preset_call","arg":1,...}}
    # over serial. We turn each key into an "event key" string and look it up
    # in a user-configurable mapping to run a named action (snap, objective,
    # laser, led, …). This mirrors the snap callback (registerCaptureCallback)
    # but is table-driven so bindings can be changed at runtime via the API.
    # ══════════════════════════════════════════════════════════════════════
    @staticmethod
    def _ptzEventKey(name, arg):
        """Canonical binding key for a PTZ event. Keys with a meaningful arg
        (presets 1..255, AUX 1..6) are addressed as "name:arg"; arg-less keys
        (iris_open, autoscan, …) as plain "name". Dispatch tries "name:arg"
        first, then falls back to "name" so a binding can be arg-agnostic."""
        try:
            arg = int(arg)
        except (TypeError, ValueError):
            arg = 0
        return f"{name}:{arg}" if arg else str(name)

    def _buildPtzActions(self):
        """Return the registry of named actions {name: (fn, help)}.

        Each fn takes a params dict (from the binding) plus the raw event, and
        performs one microscope operation. Add new actions here; the mapping
        API only references them by name so the frontend can offer a dropdown.
        """
        return {
            "none":      (self._ptzActionNone,      "Do nothing (unbind a key)."),
            "snap":      (self._ptzActionSnap,      "Snap an image and save it (like the ESP32 snap callback)."),
            "objective": (self._ptzActionObjective, "Switch objective. params: {'slot': 0|1}."),
            "laser":     (self._ptzActionLaser,     "Set a laser/LED device. params: {'name': <deviceName>, 'value': 0..1023, 'active': bool}."),
            "led":       (self._ptzActionLaser,     "Alias of 'laser' for LED-array devices in lasersManager."),
        }

    def _defaultPtzMapping(self):
        """Sensible starting bindings. Objective + snap are unambiguous; the
        laser/LED example is left with a placeholder device name the user
        edits via setPtzAction (device names are setup-specific)."""
        return {
            "preset_set": {"action": "objective", "params": {"slot": 0}},
            "preset_call": {"action": "objective", "params": {"slot": 1}},
            "iris_open":     {"action": "snap",      "params": {}},
            # Example LED binding — set the real device name for your setup:
            #   setPtzAction("aux_on:4",  "led", {"name":"LED", "value":512, "active":True})
            #   setPtzAction("aux_off:4", "led", {"name":"LED", "value":0,   "active":False})
        }

    # ── individual actions ────────────────────────────────────────────────
    def _ptzActionNone(self, params, event):
        return {"ok": True, "action": "none"}

    def _ptzActionSnap(self, params, event):
        """Snap + save from the first detector, and push it to the viewer.
        Same operation as the ESP32-triggered snapImage callback."""
        try:
            detector_names = self._master.detectorsManager.getAllDeviceNames()
            detector = self._master.detectorsManager[detector_names[0]]
            mImage = detector.getLatestFrame()
            drivePath = dirtools.UserFileDirs.getValidatedDataPath()
            timeStamp = datetime.datetime.now().strftime("%Y_%m_%d")
            dirPath = os.path.join(drivePath, 'recordings', timeStamp)
            fileName = "PTZSnap_" + datetime.datetime.now().strftime("%Y_%m_%d-%H-%M-%S")
            os.makedirs(dirPath, exist_ok=True)
            filePath = os.path.join(dirPath, fileName)
            if mImage is not None:
                tif.imwrite(filePath + ".tif", mImage[0] if mImage.ndim == 3 else mImage)
                self._commChannel.sigUpdateImage.emit('Image', mImage, True, 1, False)
            return {"ok": True, "action": "snap", "path": filePath + ".tif"}
        except Exception as e:
            self.__logger.error(f"PTZ snap action failed: {e}")
            return {"ok": False, "action": "snap", "error": str(e)}

    def _ptzActionObjective(self, params, event):
        """Switch objective slot via the ObjectiveController (commChannel)."""
        try:
            slot = int(params.get("slot", 0))
            self._commChannel.sigSetObjectiveByID.emit(str(slot))
            return {"ok": True, "action": "objective", "slot": slot}
        except Exception as e:
            self.__logger.error(f"PTZ objective action failed: {e}")
            return {"ok": False, "action": "objective", "error": str(e)}

    def _ptzActionLaser(self, params, event):
        """Enable/disable + set a laser or LED device in lasersManager.
        params: {'name': deviceName, 'value': 0..1023, 'active': bool}."""
        try:
            name = params.get("name")
            if not name:
                return {"ok": False, "action": "laser", "error": "no device name in params"}
            names = self._master.lasersManager.getAllDeviceNames()
            if name not in names:
                return {"ok": False, "action": "laser",
                        "error": f"device '{name}' not found in {names}"}
            device = self._master.lasersManager[name]
            active = bool(params.get("active", True))
            device.setEnabled(active)
            if "value" in params and active:
                device.setValue(int(params["value"]))
            return {"ok": True, "action": "laser", "name": name, "active": active,
                    "value": params.get("value")}
        except Exception as e:
            self.__logger.error(f"PTZ laser action failed: {e}")
            return {"ok": False, "action": "laser", "error": str(e)}

    # ── dispatch + registration ───────────────────────────────────────────
    def _dispatchPtzEvent(self, event):
        """Look the event up in the mapping and run the bound action. Emits
        sigPtzEvent with {event, binding, result} either way so a UI can show
        activity and unmapped keys (helpful when building a binding table)."""
        name = event.get("name", "")
        arg = event.get("arg", 0)
        exactKey = self._ptzEventKey(name, arg)
        binding = self._ptzMapping.get(exactKey) or self._ptzMapping.get(str(name))
        result = None
        if binding:
            actionName = binding.get("action", "none")
            entry = self._ptzActions.get(actionName)
            if entry is None:
                result = {"ok": False, "error": f"unknown action '{actionName}'"}
            else:
                fn = entry[0]
                result = fn(binding.get("params", {}) or {}, event)
            self.__logger.info(f"PTZ key {exactKey} -> {actionName}: {result}")
        else:
            self.__logger.debug(f"PTZ key {exactKey} unmapped")
        self.sigPtzEvent.emit({
            "event": event,
            "eventKey": exactKey,
            "binding": binding,
            "result": result,
            "timestamp": datetime.datetime.now().isoformat(),
        })

    def registerPtzCallback(self):
        """Register the PTZ key dispatcher with the manager (ESP32.ptz)."""
        def ptz_callback(event):
            try:
                self._lastPtzEvent = event
                self._dispatchPtzEvent(event)
            except Exception as e:
                self.__logger.error(f"Error in ptz_callback: {e}")
        try:
            registered = self._master.UC2ConfigManager.registerPtzCallback(ptz_callback)
            if registered:
                self.__logger.debug("PTZ callback registered successfully")
            else:
                self.__logger.debug("PTZ callback not registered (ptz module unavailable)")
        except Exception as e:
            self.__logger.error(f"Could not register PTZ callback: {e}")

    ''' PTZ keyboard bridge (CAN node 61) '''

    @APIExport(runOnUIThread=False)
    def getPtzMapping(self):
        """Return the current PTZ eventKey -> {action, params} bindings."""
        return dict(self._ptzMapping)

    @APIExport(runOnUIThread=False)
    def listPtzActions(self):
        """Available action names and their help text (for a binding UI)."""
        return {name: help for name, (fn, help) in self._ptzActions.items()}

    @APIExport(runOnUIThread=False)
    def setPtzAction(self, eventKey: str, action: str, params: dict = None):
        """Bind one key. eventKey e.g. "preset_call:1", "aux_on:4", "iris_open".
        action must be one of listPtzActions(). params is action-specific,
        e.g. {"slot":1} for objective, {"name":"LED","value":512} for led."""
        if action not in self._ptzActions:
            return {"status": "error",
                    "message": f"unknown action '{action}'; choose from {list(self._ptzActions)}"}
        self._ptzMapping[str(eventKey)] = {"action": action, "params": dict(params or {})}
        return {"status": "ok", "mapping": self._ptzMapping[str(eventKey)]}

    @APIExport(runOnUIThread=False)
    def setPtzMapping(self, mapping: dict):
        """Replace the whole binding table. Values are {"action","params"};
        unknown actions are rejected."""
        clean = {}
        for k, v in (mapping or {}).items():
            action = (v or {}).get("action", "none")
            if action not in self._ptzActions:
                return {"status": "error", "message": f"unknown action '{action}' for key '{k}'"}
            clean[str(k)] = {"action": action, "params": dict((v or {}).get("params", {}) or {})}
        self._ptzMapping = clean
        return {"status": "ok", "mapping": self._ptzMapping}

    @APIExport(runOnUIThread=False)
    def clearPtzAction(self, eventKey: str):
        """Remove one binding."""
        self._ptzMapping.pop(str(eventKey), None)
        return {"status": "ok", "mapping": self._ptzMapping}

    @APIExport(runOnUIThread=False)
    def triggerPtzEvent(self, name: str, arg: int = 0):
        """Simulate a keyboard key (runs the bound action) — test the mapping
        without the hardware. e.g. triggerPtzEvent("preset_call", 1)."""
        event = {"event": 1, "name": name, "arg": int(arg), "simulated": True}
        self._lastPtzEvent = event
        self._dispatchPtzEvent(event)
        return {"status": "ok", "event": event}

    @APIExport(runOnUIThread=False)
    def getPtzStatus(self):
        """PTZ bridge diagnostics (parser stats + last frame + motion) plus the
        last key event and the current binding table."""
        try:
            status = self._master.UC2ConfigManager.getPtzStatus() or {}
        except Exception as e:
            self.__logger.error(f"getPtzStatus failed: {e}")
            status = {"status": "error", "message": str(e)}
        status["lastEvent"] = self._lastPtzEvent
        status["mapping"] = self._ptzMapping
        return status

    @APIExport(runOnUIThread=False)
    def setPtzDebug(self, level: int):
        """Bridge serial debug verbosity: 0 quiet, 1 decoded frames, 2 + raw
        RS-485 hex (wiring / baud bring-up)."""
        try:
            return self._master.UC2ConfigManager.setPtzDebug(int(level))
        except Exception as e:
            self.__logger.error(f"setPtzDebug failed: {e}")
            return {"status": "error", "message": str(e)}

    ''' Collision detector (GPIO slave) '''

    @APIExport(runOnUIThread=False)
    def getGpioStatus(self):
        """
        Poll the collision detector on the GPIO slave (SDO reads over CAN).

        :return: {"mode" (0=auto/1=manual), "mean"/"baseline", "sigma",
                  "deviation", "filtered", "raw", "reference", "threshold",
                  "sensitivity", "trip", "estop"} plus the software crash
                  state {"latched", "armed", "requiresHoming"}.
        """
        try:
            status = self._master.UC2ConfigManager.getGpioStatus() or {}
            status["latched"] = self._collisionLatched
            status["armed"] = self._collisionArmed
            status["requiresHoming"] = self._collisionRequiresHoming
            return status
        except Exception as e:
            self.__logger.error(f"getGpioStatus failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def getCollisionState(self):
        """Software crash state without touching the CAN bus:
        {"trip", "latched", "armed", "event", "timestamp"}."""
        return self._collisionStateDict()

    @APIExport(runOnUIThread=False)
    def setCollisionThreshold(self, threshold: int):
        """Deviation band (ADC counts) around the reference; persisted on the
        slave in NVS."""
        try:
            return self._master.UC2ConfigManager.setCollisionThreshold(int(threshold))
        except Exception as e:
            self.__logger.error(f"setCollisionThreshold failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def setCollisionSensitivity(self, sensitivity: int):
        """Consecutive out-of-band samples (50 Hz) required to trip/clear —
        rejects single-sample spikes. Persisted on the slave in NVS."""
        try:
            return self._master.UC2ConfigManager.setCollisionSensitivity(int(sensitivity))
        except Exception as e:
            self.__logger.error(f"setCollisionSensitivity failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def setCollisionReference(self, reference: int):
        """Explicitly set the idle baseline (ADC counts) — typically the
        polled "mean" value. Persisted on the slave in NVS."""
        try:
            return self._master.UC2ConfigManager.setCollisionReference(int(reference))
        except Exception as e:
            self.__logger.error(f"setCollisionReference failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def calibrateCollisionReference(self):
        """Tell the slave to take its CURRENT rolling mean as the new
        reference. MANUAL mode only. Call while idle and collision-free."""
        try:
            return self._master.UC2ConfigManager.calibrateCollisionReference()
        except Exception as e:
            self.__logger.error(f"calibrateCollisionReference failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def setCollisionMode(self, mode: str = "auto"):
        """
        Select the collision-detection algorithm on the slave (persisted NVS):
          "auto"   — adaptive baseline + robust-sigma z-score. Parameter-free:
                     tracks a slowly drifting background and trips on a fast
                     deflection. Recommended; no calibration needed.
          "manual" — fixed reference +/- threshold. Deterministic; requires the
                     reference to be calibrated to the idle level.
        """
        try:
            return self._master.UC2ConfigManager.setCollisionMode(mode)
        except Exception as e:
            self.__logger.error(f"setCollisionMode failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def armCollisionProtection(self, arm: bool = True):
        """
        Arm/disarm automatic motor stop on collision. While armed, a pushed
        collision event immediately stops ALL positioners; the crash state
        latches until resetCollisionAlarm() is called.
        """
        self._collisionArmed = bool(arm)
        self.__logger.info(f"Collision auto-stop {'ARMED' if self._collisionArmed else 'disarmed'}")
        state = self._collisionStateDict()
        self.sigCollisionStatusUpdate.emit(state)
        return state

    @APIExport(runOnUIThread=False)
    def resetCollisionAlarm(self, restorePower: bool = True):
        """
        Clear the latched crash state after the user has verified the
        situation is resolved. By default this also restores CAN-bus power
        (which a collision cuts) so the stage can move again — but it does
        NOT move any motor. The `requiresHoming` flag stays set: the position
        reference is lost after a crash, so a safe frame-homing must be run
        (and confirmed via confirmSafeHoming) before normal operation.
        """
        self._collisionLatched = False
        if restorePower:
            try:
                self.setBusPower(enable=True)
            except Exception as e:
                self.__logger.error(f"Could not restore bus power on reset: {e}")
        self.__logger.info(
            "Collision alarm reset by user (bus power restored=%s); safe homing still required"
            % bool(restorePower))
        state = self._collisionStateDict()
        self.sigCollisionStatusUpdate.emit(state)
        return state

    @APIExport(runOnUIThread=False)
    def confirmSafeHoming(self):
        """
        Clear the `requiresHoming` flag after a safe frame-homing has been
        completed following a crash. Call this once the stage has been
        re-homed and its position is trustworthy again.
        """
        self._collisionRequiresHoming = False
        self.__logger.info("Collision post-crash homing confirmed by user")
        state = self._collisionStateDict()
        self.sigCollisionStatusUpdate.emit(state)
        return state

    ''' CAN-bus power & emergency-stop (safety) '''

    @APIExport(runOnUIThread=False)
    def setBusPower(self, enable: bool = True):
        """
        Enable/disable the high-current CAN-bus power that feeds the slaves.

        :param enable: True to power the bus, False to cut power.
        :return: Status dict with the requested power state.
        """
        try:
            self._master.UC2ConfigManager.setBusPower(enable)
            return {"status": "success", "power": int(bool(enable))}
        except Exception as e:
            self.__logger.error(f"setBusPower failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def getBusStatus(self):
        """
        Query CAN-bus power and emergency-stop status.

        :return: {
            "power": 1|0|None,           # high-current bus power state
            "emergencyActive": bool,     # E-stop currently asserted
            "available": bool,           # bus usable (powered and not in E-stop)
            "estop": {...}               # raw E-stop diagnostics (best-effort)
        }
        """
        try:
            power = self._master.UC2ConfigManager.getBusPower()
            emergency_active = self._master.UC2ConfigManager.isEmergencyActive()
            estop = self._master.UC2ConfigManager.getEstop()
            return {
                "power": power,
                "emergencyActive": bool(emergency_active),
                "available": (power == 1) and not emergency_active,
                "estop": estop,
            }
        except Exception as e:
            self.__logger.error(f"getBusStatus failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def getFirmwareInfo(self):
        """
        Identity of the USB-connected ESP32 master firmware.

        :return: {name, version, fwVersion, date, author, pindef, isMaster,
                  connected, serialport}. ``fwVersion`` is the release the
                  firmware was built from (same string as version.json on the
                  firmware server; empty on firmware/uc2rest predating it),
                  ``version`` the fixed API generation ("V2.0").
        """
        try:
            return self._master.UC2ConfigManager.getFirmwareInfo()
        except Exception as e:
            self.__logger.error(f"getFirmwareInfo failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def getMicroscopeStandName(self):
        """Human-readable microscope stand model (default 'openUC2 FRAME')."""
        try:
            if getattr(self, "_setupInfo", None) is not None:
                return {"name": self._setupInfo.getMicroscopeStandName()}
        except Exception as e:
            self.__logger.error(f"getMicroscopeStandName failed: {e}")
        return {"name": "openUC2 FRAME"}

    ''' PS-controller joystick direction '''

    @APIExport(runOnUIThread=False, requestType="GET")
    def setJoystickDirection(self, axis: str = "X", inverted: bool = False):
        """
        Invert (or un-invert) the PS-controller joystick for one motor axis.
        Updates the in-memory cache in the stage manager and persists the new
        value to the JSON setup file so the setting survives a restart.

        :param axis: Axis name ("A", "X", "Y", "Z").
        :param inverted: True to reverse joystick movement for that axis.
        """
        try:
            # Update device + stage-manager cache (single source of truth).
            if self.stages is not None and hasattr(self.stages, "setJoystickDirectionSettings"):
                result = self.stages.setJoystickDirectionSettings(axis=axis, inverted=inverted)
                if not result.get("success"):
                    return {"status": "error", "message": result.get("error", "unknown")}
            else:
                # Fallback: direct device call when stages are unavailable.
                self._master.UC2ConfigManager.setJoystickDirection(axis=axis, inverted=inverted)
            # Persist the updated settings to the JSON config.
            self._saveJoystickSettings()
            return {"status": "success", "axis": str(axis).upper(), "inverted": bool(inverted)}
        except Exception as e:
            self.__logger.error(f"setJoystickDirection failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def getJoystickDirection(self):
        """
        Read the joystick inversion per axis.

        Reads from the in-memory stage-manager cache (no device round-trip).
        This endpoint is intended for frontend initialisation: call it once on
        page load to reflect the persisted config state without querying the
        device. Falls back to a live device query if the cache is unavailable.

        :return: {"status": "ok", "axes": {"A": bool, "X": bool, "Y": bool, "Z": bool}}
        """
        try:
            # Prefer the in-memory cache (populated from config on startup).
            if self.stages is not None and hasattr(self.stages, "getJoystickDirectionSettings"):
                axes = self.stages.getJoystickDirectionSettings()
                return {"status": "ok", "axes": axes}
            # Fallback: live device query.
            idx_to_name = {0: "A", 1: "X", 2: "Y", 3: "Z"}
            raw = self._master.UC2ConfigManager.getJoystickDirection(axis=None)
            axes = {}
            if isinstance(raw, list):
                for entry in raw:
                    name = idx_to_name.get(entry.get("axis"))
                    if name:
                        axes[name] = bool(entry.get("inverted", False))
            return {"status": "ok", "axes": axes}
        except Exception as e:
            self.__logger.error(f"getJoystickDirection failed: {e}")
            return {"status": "error", "message": str(e)}

    def _saveJoystickSettings(self):
        """Persist the current joystick inversion cache to the JSON setup file.

        Follows the same pattern as ``saveMotorSettings`` — reads the current
        in-memory cache from the stage manager and writes the per-axis
        ``joystickInverted<AXIS>`` keys into ``managerProperties``.
        """
        try:
            if self.stages is None or not hasattr(self.stages, "getJoystickDirectionSettings"):
                self.__logger.warning("Cannot save joystick settings: stages not available")
                return
            if not hasattr(self, "_setupInfo") or self._setupInfo is None:
                self.__logger.warning("Cannot save joystick settings: _setupInfo not available")
                return

            positionerName = self.stages._name
            joystick = self.stages.getJoystickDirectionSettings()
            props_update = {f'joystickInverted{ax}': bool(v) for ax, v in joystick.items()}

            if hasattr(self._setupInfo, "positioners") and positionerName in self._setupInfo.positioners:
                for key, value in props_update.items():
                    self._setupInfo.positioners[positionerName].managerProperties[key] = value

                import imswitch.imcontrol.model.configfiletools as configfiletools
                mOptions, _ = configfiletools.loadOptions()
                configfiletools.saveSetupInfo(mOptions, self._setupInfo)
                self.__logger.info(f"Joystick settings saved for positioner '{positionerName}'")
            else:
                self.__logger.warning(
                    f"Positioner '{positionerName}' not found in setupInfo.positioners"
                )
        except Exception as e:
            self.__logger.error(f"Could not save joystick settings: {e}")

    ''' Joystick-jog speed multiplier '''

    @APIExport(runOnUIThread=False, requestType="GET")
    def setSpeedMultiplier(self, axis: str = "X", multiplier: float = 1):
        """
        Set the joystick-jog speed multiplier for one motor axis.
        Updates the in-memory cache in the stage manager and persists the new
        value to the JSON setup file so the setting survives a restart.

        :param axis: Axis name ("A", "X", "Y", "Z").
        :param multiplier: Speed multiplier applied on the device for joystick jogging.
        """
        try:
            if self.stages is not None and hasattr(self.stages, "setSpeedMultiplierSettings"):
                result = self.stages.setSpeedMultiplierSettings(axis=axis, multiplier=multiplier)
                if not result.get("success"):
                    return {"status": "error", "message": result.get("error", "unknown")}
            else:
                # Fallback: direct device call when stages are unavailable.
                self._master.UC2ConfigManager.setSpeedMultiplier(axis=axis, multiplier=multiplier)
            # Persist the updated settings to the JSON config.
            self._saveSpeedMultiplierSettings()
            return {"status": "success", "axis": str(axis).upper(), "multiplier": multiplier}
        except Exception as e:
            self.__logger.error(f"setSpeedMultiplier failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def getSpeedMultiplier(self):
        """
        Read the joystick-jog speed multiplier per axis.

        Reads from the in-memory stage-manager cache (no device round-trip).
        Falls back to a live device query if the cache is unavailable.

        :return: {"status": "ok", "axes": {"A": num, "X": num, "Y": num, "Z": num}}
        """
        try:
            if self.stages is not None and hasattr(self.stages, "getSpeedMultiplierSettings"):
                axes = self.stages.getSpeedMultiplierSettings()
                return {"status": "ok", "axes": axes}
            # Fallback: live device query.
            idx_to_name = {0: "A", 1: "X", 2: "Y", 3: "Z"}
            raw = self._master.UC2ConfigManager.getSpeedMultiplier(axis=None)
            axes = {}
            if isinstance(raw, list):
                for entry in raw:
                    name = idx_to_name.get(entry.get("axis"))
                    if name:
                        axes[name] = entry.get("multiplier", 1)
            return {"status": "ok", "axes": axes}
        except Exception as e:
            self.__logger.error(f"getSpeedMultiplier failed: {e}")
            return {"status": "error", "message": str(e)}

    def _saveSpeedMultiplierSettings(self):
        """Persist the current speed-multiplier cache to the JSON setup file.

        Follows the same pattern as ``_saveJoystickSettings``.
        """
        try:
            if self.stages is None or not hasattr(self.stages, "getSpeedMultiplierSettings"):
                self.__logger.warning("Cannot save speed multiplier settings: stages not available")
                return
            if not hasattr(self, "_setupInfo") or self._setupInfo is None:
                self.__logger.warning("Cannot save speed multiplier settings: _setupInfo not available")
                return

            positionerName = self.stages._name
            speedMult = self.stages.getSpeedMultiplierSettings()
            props_update = {f'speedMultiplier{ax}': v for ax, v in speedMult.items()}

            if hasattr(self._setupInfo, "positioners") and positionerName in self._setupInfo.positioners:
                for key, value in props_update.items():
                    self._setupInfo.positioners[positionerName].managerProperties[key] = value

                import imswitch.imcontrol.model.configfiletools as configfiletools
                mOptions, _ = configfiletools.loadOptions()
                configfiletools.saveSetupInfo(mOptions, self._setupInfo)
                self.__logger.info(f"Speed multiplier settings saved for positioner '{positionerName}'")
            else:
                self.__logger.warning(
                    f"Positioner '{positionerName}' not found in setupInfo.positioners"
                )
        except Exception as e:
            self.__logger.error(f"Could not save speed multiplier settings: {e}")

    ''' Fan & board temperature '''

    @APIExport(runOnUIThread=False)
    def getFanState(self):
        """
        Read the current fan state from the firmware.

        :return: {mode, wiper, manual, rpm, stalled, kick, tempC, curve}
        """
        try:
            return self._master.UC2ConfigManager.getFan()
        except Exception as e:
            self.__logger.error(f"getFanState failed: {e}")
            return {}

    @APIExport(runOnUIThread=False)
    def setFanMode(self, mode: str = "auto", wiper: int = None):
        """
        Set the fan operating mode.

        :param mode: 'auto' (curve-driven), 'manual' (fixed wiper) or 'off'.
        :param wiper: 0-127 PWM wiper, required/used for 'manual' mode.
        :return: Status dict.
        """
        try:
            w = int(wiper) if wiper is not None else None
        except (TypeError, ValueError):
            w = None
        try:
            self._master.UC2ConfigManager.setFanMode(mode=mode, wiper=w)
            return {"status": "success", "mode": mode, "wiper": w}
        except Exception as e:
            self.__logger.error(f"setFanMode failed: {e}")
            return {"status": "error", "message": str(e)}

    @APIExport(runOnUIThread=False)
    def getBoardTemperature(self):
        """
        Read board/air temperatures from the firmware.

        :return: {pcb, air, esp, pcb_ok, air_ok}
        """
        try:
            return self._master.UC2ConfigManager.getTemperature()
        except Exception as e:
            self.__logger.error(f"getBoardTemperature failed: {e}")
            return {}






    def set_motor_positions(self, a, x, y, z):
        # Add your logic to set motor positions here.
        self.__logger.debug(f"Setting motor positions: A={a}, X={x}, Y={y}, Z={z}")
        # push the positions to the motor controller
        if a is not None: self.stages.setPositionOnDevice(value=float(a), axis="A")
        if x is not None:  self.stages.setPositionOnDevice(value=float(x), axis="X")
        if y is not None: self.stages.setPositionOnDevice(value=float(y), axis="Y")
        if z is not None: self.stages.setPositionOnDevice(value=float(z), axis="Z")

        # retrieve the positions from the motor controller
        positions = self.stages.getPosition()
        # update the GUI
        self._commChannel.sigUpdateMotorPosition.emit()

    def interruptSerialCommunication(self):
        self._master.UC2ConfigManager.interruptSerialCommunication()

    def set_auto_enable(self):
        # Add your logic to auto-enable the motors here.
        # get motor controller
        self.stages.enalbeMotors(enableauto=True)

    def unset_auto_enable(self):
        # Add your logic to unset auto-enable for the motors here.
        self.stages.enalbeMotors(enable=True, enableauto=False)

    def reconnectThread(self, baudrate=None, port=None):
        try:
            self._master.UC2ConfigManager.initSerial(port=port, baudrate=baudrate)
        except TypeError:
            # Older managers may not accept the port kwarg
            self._master.UC2ConfigManager.initSerial(baudrate=baudrate)
        self.__logger.debug("We are connected: "+str(self._master.UC2ConfigManager.isConnected()))
        self.sigUC2SerialIsConnected.emit(self._master.UC2ConfigManager.isConnected())

    def closeConnection(self):
        self._master.UC2ConfigManager.closeSerial()

    @APIExport(runOnUIThread=True)
    def moveToSampleMountingPosition(self):
        self._logger.debug('Moving to sample loading position.')
        self.stages.moveToSampleMountingPosition()

    @APIExport(runOnUIThread=False)
    def stopImSwitch(self):
        self._commChannel.sigExperimentStop.emit()
        return {"message": "ImSwitch is shutting down"}

    @APIExport(runOnUIThread=False)
    def restartImSwitch(self):
        ostools.restartSoftware()
        return {"message": "ImSwitch is restarting"}

    @APIExport(runOnUIThread=False)
    def isImSwitchRunning(self):
        return True

    # --- Forklift pallet upgrade (openUC2 OS only, EXPERIMENTAL) -------------
    # The upgrade runs in the host's systemd unit imswitch-pallet-upgrade.service (from the
    # openUC2/os-rpi pallet), not in this container, because `forklift stage apply` recreates
    # this container. We reach systemd through the host D-Bus socket mounted into the container.
    _PALLET_UPGRADE_UNIT = "imswitch-pallet-upgrade.service"

    def _callSystemd(self, method, signature, body, path="/org/freedesktop/systemd1",
                     interface="org.freedesktop.systemd1.Manager"):
        from jeepney import DBusAddress, new_method_call
        from jeepney.io.blocking import open_dbus_connection
        from jeepney.wrappers import unwrap_msg
        address = DBusAddress(path, bus_name="org.freedesktop.systemd1", interface=interface)
        with open_dbus_connection(bus="SYSTEM") as conn:
            return unwrap_msg(conn.send_and_get_reply(
                new_method_call(address, method, signature, body), timeout=5))

    @APIExport(runOnUIThread=False)
    def startPalletUpgrade(self):
        """EXPERIMENTAL: run `forklift plt upgrade --force && forklift stage apply` on the host.

        Returns once systemd has queued the job. The upgrade then restarts this ImSwitch
        container; poll getPalletUpgradeStatus for its state and terminal output.
        """
        try:
            self._callSystemd("StartUnit", "ss", (self._PALLET_UPGRADE_UNIT, "replace"))
        except Exception as e:
            self._logger.error(f"Could not start the pallet upgrade: {e}")
            return {"status": "error", "message": str(e)}
        return {"status": "started"}

    @APIExport(runOnUIThread=False)
    def getPalletUpgradeStatus(self):
        """systemd state of the pallet upgrade ("activating" while it runs, then "inactive" or
        "failed") and the last 500 lines of its terminal output."""
        try:
            (unitPath,) = self._callSystemd("LoadUnit", "s", (self._PALLET_UPGRADE_UNIT,))
            ((_, state),) = self._callSystemd(
                "Get", "ss", ("org.freedesktop.systemd1.Unit", "ActiveState"),
                path=unitPath, interface="org.freedesktop.DBus.Properties")
        except Exception as e:
            state = f"unknown: {e}"
        try:
            # Written by the unit's StandardOutput= into the bind-mounted config folder
            with open(os.path.join(dirtools.UserFileDirs.Root, "pallet-upgrade.log"),
                      errors="replace") as f:
                log = "".join(f.readlines()[-500:])
        except OSError:
            log = ""
        return {"state": state, "log": log}

    @APIExport(runOnUIThread=False)
    def getDataPath(self):
        return dirtools.UserFileDirs.getValidatedDataPath()

    @APIExport(runOnUIThread=False)
    def setDataPathFolder(self, path):
        dirtools.UserFileDirs.Data = path
        self._logger.debug(f"Data path set to {path}")
        return {"message": f"Data path set to {path}"}

    @APIExport(runOnUIThread=True)
    def reconnect(self, port: str = None, baudrate: int = None):
        """Reconnect to the ESP32 over serial.

        Optionally override the port and/or baudrate. When both are omitted
        the values from the setup JSON / current runtime are used.
        """
        self._logger.debug(
            f'Reconnecting to ESP32 device (port={port}, baudrate={baudrate}).'
        )
        try:
            baudrate_int = int(baudrate) if baudrate is not None else None
        except (TypeError, ValueError):
            baudrate_int = None
        port_str = port if port else None
        mThread = threading.Thread(
            target=self.reconnectThread,
            kwargs={"baudrate": baudrate_int, "port": port_str},
        )
        mThread.start()
        return {
            "status": "started",
            "port": port_str,
            "baudrate": baudrate_int,
        }


    @APIExport(runOnUIThread=True)
    def writeSerial(self, payload):
        return self._master.UC2ConfigManager.ESP32.serial.writeSerial(payload)

    @APIExport(runOnUIThread=True)
    def uc2_board_is_connected(self, strict: bool = False):
        """Check whether the ESP32 board is reachable.

        - ``strict=False`` (default): cheap flag-only check used by the
          polling loop in the frontend.
        - ``strict=True``: actively pings the firmware (writes /state_get)
          and reports True only if the device responds within ~0.5s.
        """
        if not strict:
            return self._master.UC2ConfigManager.isConnected()
        try:
            return bool(self._master.UC2ConfigManager.ping(timeout=0.5))
        except Exception as e:
            self.__logger.debug(f"strict uc2_board_is_connected ping failed: {e}")
            return False

    @APIExport(runOnUIThread=True)
    def btpairing(self):
        self._logger.debug('Pairing BT device.')
        mThread = threading.Thread(target=self._master.UC2ConfigManager.pairBT)
        mThread.start()
        mThread.join()


    @APIExport(runOnUIThread=False)
    def espRestart(self):
        try:
            self._master.UC2ConfigManager.restartESP()
            return {"status": "ESP32 restarted successfully"}
        except Exception as e:
            return {"error": str(e)}














    
    


    





    # USB flash status tracking




    # -----------------------------
    # USB flashing for CAN HAT (master)
    # -----------------------------





    # VID:PID to chip type mapping for auto-detection

    # Chip-specific esptool flash parameters

    # Known CAN bus addresses








    @APIExport(runOnUIThread=False, requestType="POST")
    def setSerialConfig(self, port: str = "", baudrate: int = 115200, persist: bool = True):
        """Apply (and optionally persist) the serial port / baudrate used to talk to the ESP32.

        Parameters:
          port: serial port device path (e.g. "/dev/ttyACM0"). Empty/"auto" keeps current.
          baudrate: serial baudrate (default 115200). Pass 0/None to keep current.
          persist: when True, write the change back to the setup JSON.
        """
        # Normalize optional inputs coming from query params
        port_arg = port.strip() if isinstance(port, str) else port
        if not port_arg or str(port_arg).lower() == "auto":
            port_arg = None
        baud_arg = int(baudrate) if baudrate not in (None, 0, "", "null") else None

        try:
            result = self._master.UC2ConfigManager.setSerialConfig(
                port=port_arg,
                baudrate=baud_arg,
                persist=bool(persist),
            )
            return {"status": "success", **result, "persisted": bool(persist)}
        except Exception as e:
            self.__logger.error(f"Failed to set serial config: {e}", exc_info=True)
            return {"status": "error", "message": str(e)}
        
    # Digital Input/Output API Methods


    # Digital Input/Output API Methods
    @APIExport(runOnUIThread=False)
    def getDigitalIn(self, digitalinid=1, timeout=1, is_blocking=True):
        """
        Get the current value of a digital input.
        
        :param digitalinid: ID of the digital input (1, 2, or 3)
        :param timeout: Timeout for the request in seconds
        :param is_blocking: Whether to wait for a response
        :return: Response from the device containing digitalin status
        """
        try:
            digitalinid = int(digitalinid)
            timeout = float(timeout)
            return self._master.UC2ConfigManager._digitalIn.get_digitalin(
                digitalinid=digitalinid,
                timeout=timeout,
                is_blocking=is_blocking
            )
        except Exception as e:
            self.__logger.error(f"Error getting digital input: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False)
    def actDigitalIn(self, timeout=1, is_blocking=False):
        """
        Trigger the digitalin act function.
        
        :param timeout: Timeout for the request in seconds
        :param is_blocking: Whether to wait for a response
        :return: Response from the device
        """
        try:
            return self._master.UC2ConfigManager._digitalIn.act_digitalin(
                timeout=timeout,
                is_blocking=is_blocking
            )
        except Exception as e:
            self.__logger.error(f"Error acting digital input: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False)
    def getDigitalOut(self, digitaloutid=1, timeout=1, is_blocking=True):
        """
        Get the current value and pin of a digital output.
        
        :param digitaloutid: ID of the digital output (1, 2, or 3)
        :param timeout: Timeout for the request in seconds
        :param is_blocking: Whether to wait for a response
        :return: Response from the device containing digitalout status (id, val, pin)
        """
        try:
            return self._master.UC2ConfigManager._digitalOut.get_digitalout(
                digitaloutid=digitaloutid,
                timeout=timeout,
                is_blocking=is_blocking
            )
        except Exception as e:
            self.__logger.error(f"Error getting digital output: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False)
    def setDigitalOut(self, digitaloutid=1, digitaloutval=0, timeout=1, is_blocking=False):
        """
        Set the value of a digital output. Use digitaloutval=-1 to trigger a pulse (HIGH->LOW).
        
        :param digitaloutid: ID of the digital output (1, 2, or 3)
        :param digitaloutval: Value to set (0=LOW, 1=HIGH, -1=pulse/trigger)
        :param timeout: Timeout for the request in seconds
        :param is_blocking: Whether to wait for a response
        :return: Response from the device
        """
        try:
            return self._master.UC2ConfigManager._digitalOut.set_digitalout(
                digitaloutid=digitaloutid,
                digitaloutval=digitaloutval,
                timeout=timeout,
                is_blocking=is_blocking
            )
        except Exception as e:
            self.__logger.error(f"Error setting digital output: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False)
    def setupDigitalOutPin(self, digitaloutid=1, pin=4):
        """
        Setup the pin for a digital output.
        
        :param digitaloutid: ID of the digital output (1, 2, or 3)
        :param pin: GPIO pin number to use
        :return: Response from the device
        """
        try:
            return self._master.UC2ConfigManager._digitalOut.setup_digitaloutpin(
                id=digitaloutid,
                pin=pin
            )
        except Exception as e:
            self.__logger.error(f"Error setting up digital output pin: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False)
    def resetTriggerTable(self):
        """
        Reset the trigger table for digital outputs.
        
        :return: Response from the device
        """
        try:
            return self._master.UC2ConfigManager._digitalOut.reset_triggertable()
        except Exception as e:
            self.__logger.error(f"Error resetting trigger table: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False)
    def setTrigger(self, trigger1=False, delayOn1=0, delayOff1=0, 
                   trigger2=False, delayOn2=0, delayOff2=0,
                   trigger3=False, delayOn3=0, delayOff3=0):
        """
        Set up a trigger table with 3 triggers.
        
        :param trigger1: Enable trigger 1
        :param delayOn1: Delay before turning on trigger 1 (ms)
        :param delayOff1: Delay before turning off trigger 1 (ms)
        :param trigger2: Enable trigger 2
        :param delayOn2: Delay before turning on trigger 2 (ms)
        :param delayOff2: Delay before turning off trigger 2 (ms)
        :param trigger3: Enable trigger 3
        :param delayOn3: Delay before turning on trigger 3 (ms)
        :param delayOff3: Delay before turning off trigger 3 (ms)
        :return: Response from the device
        """
        try:
            return self._master.UC2ConfigManager._digitalOut.set_trigger(
                trigger1=trigger1, delayOn1=delayOn1, delayOff1=delayOff1,
                trigger2=trigger2, delayOn2=delayOn2, delayOff2=delayOff2,
                trigger3=trigger3, delayOn3=delayOn3, delayOff3=delayOff3
            )
        except Exception as e:
            self.__logger.error(f"Error setting trigger: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False)
    def sendTrigger(self, triggerId=0):
        """
        Send a trigger pulse on the specified digital output.
        
        :param triggerId: ID of the trigger to send (0, 1, 2, or 3)
        :return: Response from the device
        """
        try:
            return self._master.UC2ConfigManager._digitalOut.sendTrigger(
                triggerId=triggerId
            )
        except Exception as e:
            self.__logger.error(f"Error sending trigger: {e}")
            return {"error": str(e)}

    # ============================================================================
    # Motor Settings API - Unified configuration interface for motor parameters
    # ============================================================================

    @APIExport(runOnUIThread=False)
    def getMotorSettings(self):
        """
        Get all motor settings for all axes in a unified format.
        
        Returns a dictionary with global settings and per-axis configurations
        including motion parameters, homing settings, and limit configurations.
        
        :return: Dictionary with motor settings
        """
        try:
            if self.stages is None:
                return {"error": "No stages configured", "settings": None}
            return self.stages.getMotorSettings()
        except Exception as e:
            self.__logger.error(f"Error getting motor settings: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False)
    def getMotorSettingsForAxis(self, axis: str = "X"):
        """
        Get motor settings for a specific axis.
        
        :param axis: Axis name (X, Y, Z, or A)
        :return: Dictionary with axis-specific motor settings
        """
        try:
            if self.stages is None:
                return {"error": "No stages configured", "settings": None}
            return self.stages.getMotorSettingsForAxis(axis.upper())
        except Exception as e:
            self.__logger.error(f"Error getting motor settings for axis {axis}: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False)
    def getTMCSettingsForAxis(self, axis: str = "X"):
        """
        Get TMC stepper driver settings for a specific axis from the device.
        
        :param axis: Axis name (X, Y, Z, or A)
        :return: Dictionary with TMC settings
        """
        try:
            if self.stages is None:
                return {"error": "No stages configured", "settings": None}
            return self.stages.getTMCSettingsForAxis(axis.upper())
        except Exception as e:
            self.__logger.error(f"Error getting TMC settings for axis {axis}: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=False, requestType="POST")
    def setMotorSettingsForAxis(self, axis: str, settings: dict):
        """
        Set motor settings for a specific axis.
        
        Updates both in-memory values and device configuration.
        Settings should include 'motion', 'homing', and/or 'limits' sub-dictionaries.
        
        Example settings:
        {
            "motion": {
                "stepSize": 1.0,
                "maxSpeed": 10000,
                "acceleration": 1000000,
                "backlash": 0
            },
            "homing": {
                "enabled": true,
                "speed": 15000,
                "direction": -1,
                "endstopPolarity": 1,
                "endposRelease": 3000,
                "timeout": 20000
            }
        }
        
        :param axis: Axis name (X, Y, Z, or A)
        :param settings: Dictionary with settings to update
        :return: Result dictionary with updated fields and any errors
        """
        try:
            if self.stages is None:
                return {"error": "No stages configured", "success": False}
            result = self.stages.setMotorSettingsForAxis(axis.upper(), settings)
            # Save to config after successful update
            if result.get('success', False):
                self.saveMotorSettings()
            return result
        except Exception as e:
            self.__logger.error(f"Error setting motor settings for axis {axis}: {e}")
            return {"error": str(e), "success": False}

    @APIExport(runOnUIThread=False, requestType="POST")
    def setMotorSettings(self, settings: dict):
        """
        Set motor settings for all axes at once.
        
        Settings should include 'global' and/or 'axes' sub-dictionaries.
        
        Example settings:
        {
            "global": {
                "axisOrder": [0, 1, 2, 3],
                "isCoreXY": false,
                "isEnabled": true,
                "enableAuto": true
            },
            "axes": {
                "X": { "motion": {...}, "homing": {...} },
                "Y": { "motion": {...}, "homing": {...} },
                ...
            }
        }
        
        :param settings: Dictionary with settings to update
        :return: Result dictionary with updated fields and any errors
        """
        try:
            if self.stages is None:
                return {"error": "No stages configured", "success": False}
            
            result = {"success": True, "results": {}}
            
            # Update global settings
            if 'global' in settings:
                global_result = self.stages.setGlobalMotorSettings(settings['global'])
                result['results']['global'] = global_result
                if not global_result.get('success', False):
                    result['success'] = False
            
            # Update per-axis settings
            if 'axes' in settings:
                for axis, axis_settings in settings['axes'].items():
                    axis_result = self.stages.setMotorSettingsForAxis(axis.upper(), axis_settings)
                    result['results'][axis] = axis_result
                    if not axis_result.get('success', False):
                        result['success'] = False
            
            return result
            
        except Exception as e:
            self.__logger.error(f"Error setting motor settings: {e}")
            return {"error": str(e), "success": False}

    @APIExport(runOnUIThread=False, requestType="POST")
    def setTMCSettingsForAxis(self, axis: str, settings: dict):
        """
        Set TMC stepper driver settings for a specific axis.
        
        Example settings:
        {
            "msteps": 16,
            "rmsCurrent": 500,
            "sgthrs": 10,
            "semin": 5,
            "semax": 2,
            "blankTime": 24,
            "toff": 3
        }
        
        :param axis: Axis name (X, Y, Z, or A)
        :param settings: Dictionary with TMC settings
        :return: Result dictionary
        """
        try:
            if self.stages is None:
                return {"error": "No stages configured", "success": False}
            result = self.stages.setTMCSettingsForAxis(axis.upper(), settings)
            # Save to config after successful update (including TMC settings from device)
            if result.get('success', False):
                self.saveMotorSettings(includeTMC=True)
            return result
        except Exception as e:
            self.__logger.error(f"Error setting TMC settings for axis {axis}: {e}")
            return {"error": str(e), "success": False}

    @APIExport(runOnUIThread=False, requestType="POST")
    def setGlobalMotorSettings(self, settings: dict):
        """
        Set global motor settings (axis order, CoreXY mode, enable settings).
        
        Example settings:
        {
            "axisOrder": [0, 1, 2, 3],
            "isCoreXY": false,
            "isEnabled": true,
            "enableAuto": true,
            "isDualAxis": false
        }
        
        :param settings: Dictionary with global settings
        :return: Result dictionary
        """
        try:
            if self.stages is None:
                return {"error": "No stages configured", "success": False}
            return self.stages.setGlobalMotorSettings(settings)
        except Exception as e:
            self.__logger.error(f"Error setting global motor settings: {e}")
            return {"error": str(e), "success": False}

    @APIExport(runOnUIThread=False, requestType="POST") 
    def applyMotorSettingsToDevice(self, axis: str = None):
        """
        Apply current in-memory motor settings to the device.
        
        This is useful after making multiple setting changes to push
        all changes to the hardware at once.
        
        :param axis: Specific axis to apply (X, Y, Z, A), or None for all axes
        :return: Result dictionary
        """
        try:
            if self.stages is None:
                return {"error": "No stages configured", "success": False}
            
            axes_to_apply = [axis.upper()] if axis else ['X', 'Y', 'Z', 'A']
            results = {}
            
            for ax in axes_to_apply:
                try:
                    # Re-apply motor setup
                    minPos = getattr(self.stages, f'min{ax}', float('-inf'))
                    maxPos = getattr(self.stages, f'max{ax}', float('inf'))
                    stepSize = self.stages.stepSizes.get(ax, 1)
                    backlash = getattr(self.stages, f'backlash{ax}', 0)
                    
                    self.stages.setupMotor(minPos, maxPos, stepSize, backlash, ax)
                    results[ax] = {"success": True}
                except Exception as e:
                    results[ax] = {"success": False, "error": str(e)}
            
            return {"success": all(r.get("success", False) for r in results.values()), "results": results}
            
        except Exception as e:
            self.__logger.error(f"Error applying motor settings to device: {e}")
            return {"error": str(e), "success": False}

    def saveMotorSettings(self, includeTMC: bool = False):
        """
        Save current motor settings to the config file.
        
        This follows the same pattern as PositionerController.saveStageOffset for config persistence.
        The controller has access to _setupInfo which the manager does not.
        
        Args:
            includeTMC: If True, also fetch and save TMC settings from the device
        """
        try:
            if self.stages is None:
                self.__logger.warning("Cannot save motor settings: no stages configured")
                return

            positionerName = self.stages._name
            
            # Build motor settings dict from current manager state
            axes = ["X", "Y", "Z", "A"]
            motorSettings = {}
            
            # Save step sizes
            for ax in axes:
                motorSettings[f'stepsize{ax}'] = self.stages.stepSizes.get(ax, 1)
            
            # Save max speeds
            for ax in axes:
                motorSettings[f'maxSpeed{ax}'] = self.stages.maxSpeed.get(ax, 10000)
            
            # Save initial speeds (current speeds)
            for ax in axes:
                motorSettings[f'initialSpeed{ax}'] = self.stages._speed.get(ax, 10000)
            
            # Save min/max positions
            motorSettings['minX'] = self.stages.minX if self.stages.minX != float('-inf') else None
            motorSettings['maxX'] = self.stages.maxX if self.stages.maxX != float('inf') else None
            motorSettings['minY'] = self.stages.minY if self.stages.minY != float('-inf') else None
            motorSettings['maxY'] = self.stages.maxY if self.stages.maxY != float('inf') else None
            motorSettings['minZ'] = self.stages.minZ if self.stages.minZ != float('-inf') else None
            motorSettings['maxZ'] = self.stages.maxZ if self.stages.maxZ != float('inf') else None
            motorSettings['minA'] = self.stages.minA if self.stages.minA != float('-inf') else None
            motorSettings['maxA'] = self.stages.maxA if self.stages.maxA != float('inf') else None
            
            # Save backlash
            motorSettings['backlashX'] = self.stages.backlashX
            motorSettings['backlashY'] = self.stages.backlashY
            motorSettings['backlashZ'] = self.stages.backlashZ
            motorSettings['backlashA'] = self.stages.backlashA
            
            # Save homing parameters
            for ax in axes:
                motorSettings[f'homeSpeed{ax}'] = getattr(self.stages, f'homeSpeed{ax}', 15000)
                motorSettings[f'homeDirection{ax}'] = getattr(self.stages, f'homeDirection{ax}', -1)
                motorSettings[f'homeEndstoppolarity{ax}'] = getattr(self.stages, f'homeEndstoppolarity{ax}', 1)
                motorSettings[f'homeEndposRelease{ax}'] = getattr(self.stages, f'homeEndposRelease{ax}', 1)
                motorSettings[f'homeTimeout{ax}'] = getattr(self.stages, f'homeTimeout{ax}', 20000)
            
            # Save TMC settings if requested (fetch from device)
            if includeTMC:
                for ax in axes:
                    try:
                        tmc_result = self.stages.getTMCSettingsForAxis(ax)
                        if tmc_result.get('success') and 'settings' in tmc_result:
                            tmc = tmc_result['settings']
                            motorSettings[f'msteps{ax}'] = tmc.get('msteps', 16)
                            motorSettings[f'rms_current{ax}'] = tmc.get('rmsCurrent', 500)
                            motorSettings[f'sgthrs{ax}'] = tmc.get('sgthrs', 10)
                            motorSettings[f'semin{ax}'] = tmc.get('semin', 5)
                            motorSettings[f'semax{ax}'] = tmc.get('semax', 2)
                            motorSettings[f'blank_time{ax}'] = tmc.get('blankTime', 24)
                            motorSettings[f'toff{ax}'] = tmc.get('toff', 3)
                    except Exception as e:
                        self.__logger.warning(f"Could not fetch TMC settings for axis {ax}: {e}")

            # Always save joystick direction settings from the in-memory cache.
            if hasattr(self.stages, "getJoystickDirectionSettings"):
                try:
                    joystick = self.stages.getJoystickDirectionSettings()
                    for ax, inv in joystick.items():
                        motorSettings[f'joystickInverted{ax}'] = bool(inv)
                except Exception as e:
                    self.__logger.warning(f"Could not fetch joystick settings: {e}")

            # Always save joystick-jog speed multiplier settings from the in-memory cache.
            if hasattr(self.stages, "getSpeedMultiplierSettings"):
                try:
                    speedMult = self.stages.getSpeedMultiplierSettings()
                    for ax, mult in speedMult.items():
                        motorSettings[f'speedMultiplier{ax}'] = mult
                except Exception as e:
                    self.__logger.warning(f"Could not fetch speed multiplier settings: {e}")

            # Update setupInfo and save to config file
            if hasattr(self, '_setupInfo') and self._setupInfo is not None:
                # Update the positioner's managerProperties in setupInfo
                if hasattr(self._setupInfo, 'positioners') and positionerName in self._setupInfo.positioners:
                    # Update existing properties
                    for key, value in motorSettings.items():
                        if value is not None:
                            self._setupInfo.positioners[positionerName].managerProperties[key] = value

                    # Save the updated setupInfo to disk
                    import imswitch.imcontrol.model.configfiletools as configfiletools
                    mOptions, _ = configfiletools.loadOptions()
                    configfiletools.saveSetupInfo(mOptions, self._setupInfo)
                    self.__logger.info(f"Saved motor settings for {positionerName}")
                else:
                    self.__logger.warning(f"Positioner {positionerName} not found in setupInfo.positioners")
            else:
                self.__logger.warning("Cannot save motor settings: _setupInfo not available")

        except Exception as e:
            self.__logger.error(f"Could not save motor settings: {e}")
            import traceback
            traceback.print_exc()


# Copyright (C) Benedict Diederich
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.