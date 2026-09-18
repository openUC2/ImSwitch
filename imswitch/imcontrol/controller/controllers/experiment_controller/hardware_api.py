"""Camera, illumination, stage and sensor primitives the scan steps call.

Mixed into ExperimentController; the methods keep their state on ``self``.
"""
import time
import numpy as np
from typing import Optional, Dict, Any

import os

from imswitch.imcontrol.model.managers.WorkflowManager import WorkflowContext
from imswitch.imcommon.model import APIExport

from imswitch.imcontrol.controller.controllers.experiment_controller import ExperimentWorkflowParams, SyntheticChannel


class HardwareMixin:
    """Camera, illumination, stage and sensor primitives the scan steps call."""

    def _init_hardware(self):
        """Detector, lasers, LED matrix, stage — and the capability list the designer shows."""
        # select detectors
        allDetectorNames = self._master.detectorsManager.getAllDeviceNames()
        self.mDetector = self._master.detectorsManager[allDetectorNames[0]]
        self.isRGB = self.mDetector._camera.isRGB
        self.detectorPixelSize = self.mDetector.pixelSizeUm

        # select lasers
        self.allIlluNames = self._master.lasersManager.getAllDeviceNames()
        self.availableIlluminations = []
        for iDevice in self.allIlluNames:
            # laser maanger
            self.availableIlluminations.append(self._master.lasersManager[iDevice])

        # Detect LED matrix presence so we can offer synthetic "Ring" and
        # "DPC" channels in the Wellplate designer.  We grab the *first*
        # LED matrix (multi-matrix setups are out of scope for now).
        self._ledMatrix = None
        self._ledMatrixName = None
        try:
            ledMatrixManager = getattr(self._master, "LEDMatrixsManager", None)
            if ledMatrixManager is not None:
                ledMatrixNames = ledMatrixManager.getAllDeviceNames()
                if ledMatrixNames:
                    self._ledMatrixName = ledMatrixNames[0]
                    self._ledMatrix = ledMatrixManager[self._ledMatrixName]
        except Exception as e:  # noqa: BLE001 - best-effort detection
            self._logger.warning(f"LEDMatrix detection failed: {e}")
            self._ledMatrix = None

        # Names for the synthetic LED-matrix channels.  Kept as class-level
        # constants so the workflow builder and channel-list endpoint use
        # identical strings.
        self.LED_MATRIX_RING_CHANNEL = "LED Matrix Ring"
        self.LED_MATRIX_DPC_CHANNEL = "LED Matrix DPC"
        self.DPC_SUB_CHANNELS = ("top", "bottom", "left", "right")

        # select stage
        self.allPositionerNames = self._master.positionersManager.getAllDeviceNames()[0]
        try:
            self.mStage = self._master.positionersManager[self._master.positionersManager.getAllDeviceNames()[0]]
        except (KeyError, IndexError):
            self.mStage = None

        # TODO: Adjust parameters
        # define changeable Experiment parameters as ExperimentWorkflowParams
        self.ExperimentParams = ExperimentWorkflowParams()
        self.ExperimentParams.illuSources = list(self.allIlluNames)
        # Per-source kind tag, parallel to illuSources.  Real lasers/LEDs are
        # "default"; the LED-matrix synthetic channels appended below add
        # "ring" and "dpc" entries.  Frontend renders kind-specific controls.
        self.ExperimentParams.illuSourceKinds = ["default"] * len(self.allIlluNames)
        self.ExperimentParams.illuSourceMinIntensities = []
        self.ExperimentParams.illuSourceMaxIntensities = []
        self.ExperimentParams.illuIntensities = [0]*len(self.allIlluNames)
        self.ExperimentParams.exposureTimes = [0]*len(self.allIlluNames)
        self.ExperimentParams.gains = [0]*len(self.allIlluNames)
        self.ExperimentParams.isDPCpossible = False
        self.ExperimentParams.isDarkfieldpossible = False
        self.ExperimentParams.performanceMode = False
        for laserN in self.availableIlluminations:
            self.ExperimentParams.illuSourceMinIntensities.append(laserN.valueRangeMin)
            self.ExperimentParams.illuSourceMaxIntensities.append(laserN.valueRangeMax)

        # Append LED-matrix synthetic channels when hardware is present.
        # Intensity here is reported as the RGB byte range (0..255); the
        # actual per-frame patterns (radius, RGB) come from
        # parameterValue.illuminationParams[channel_name].
        if self._ledMatrix is not None:
            # Discover matrix size so the frontend can clamp the ring-radius
            # slider to physically meaningful values.  Different LED matrix
            # managers expose dimensions slightly differently, so we probe a
            # few attribute names and fall back to a conservative 4×4 if all
            # else fails.
            _nx = (
                getattr(self._ledMatrix, "Nx", None)
                or getattr(self._ledMatrix, "nLedsX", None)
                or getattr(self._ledMatrix, "Ncols", None)
                or 4
            )
            _ny = (
                getattr(self._ledMatrix, "Ny", None)
                or getattr(self._ledMatrix, "nLedsY", None)
                or getattr(self._ledMatrix, "Nrows", None)
                or 4
            )
            try:
                _nx = int(_nx)
                _ny = int(_ny)
            except (TypeError, ValueError):
                _nx, _ny = 4, 4
            # A ring of radius r needs r LEDs from the centre to the edge,
            # so max usable radius is (min(N) - 1) // 2.  For an 8×8 matrix
            # this gives 3 (rings at radii 0,1,2,3 → 4 distinct rings).
            _max_ring_radius = 4# TODO:Hardcoded max(0, (min(_nx, _ny) - 1) // 2)
            self.ExperimentParams.ledMatrixInfo = {
                "nLedsX": _nx,
                "nLedsY": _ny,
                "maxRingRadius": _max_ring_radius,
            }
            # Synthetic LED-matrix channels are advertised in their OWN list
            # (illuSources stays conventional/default-only).  The frontend
            # renders ring/DPC controls from this list and sends back enabled
            # entries in ParameterValue.syntheticChannels.
            self.ExperimentParams.syntheticChannels = [
                SyntheticChannel(
                    name=self.LED_MATRIX_RING_CHANNEL,
                    kind="ring",
                    enabled=False,
                    intensityR=0, intensityG=0, intensityB=0,
                    radius=_max_ring_radius,
                ),
                SyntheticChannel(
                    name=self.LED_MATRIX_DPC_CHANNEL,
                    kind="dpc",
                    enabled=False,
                    # DPC conventionally lights one colour; default to green.
                    intensityR=0, intensityG=255, intensityB=0,
                ),
            ]
            # Surface DPC capability separately so existing UI bits that
            # only look at isDPCpossible can still gate features.
            self.ExperimentParams.isDPCpossible = True
            self._logger.info(
                f"LEDMatrix '{self._ledMatrixName}' detected ({_nx}x{_ny}, "
                f"maxRingRadius={_max_ring_radius}) — exposing synthetic "
                f"channels: '{self.LED_MATRIX_RING_CHANNEL}', '{self.LED_MATRIX_DPC_CHANNEL}'."
            )

    @APIExport(requestType="GET")
    def getHardwareParameters(self):
        try:
            det_px = self.mDetector.pixelSizeUm  # [Z, Y, X]
            self.ExperimentParams.pixel_size_um = float(det_px[1]) if len(det_px) > 1 else 1.0
        except Exception:
            pass
        return self.ExperimentParams

    @APIExport(requestType="GET")
    def getDetectorPixelSize(self) -> dict:
        """Return the calibrated pixel size (µm/px) for the primary acquisition detector.

        PixelCalibrationController injects the per-objective value from
        PixelCalibration.affineCalibrations into the detector at startup and
        whenever the objective changes, so this always reflects the current
        calibration.
        """
        try:
            px = self.mDetector.pixelSizeUm  # [Z, Y, X]
            value = float(px[1]) if len(px) > 1 else 1.0
        except Exception:
            value = 1.0
        return {"pixel_size_um": value}

    # ------------------------------------------------------------------
    # Wellplate / labware endpoints
    #
    # All layouts are now served by the Opentrons-style ``LabwareManager``
    # (see ``imswitch.imcontrol.model.labware``). Use ``getLabwareList`` /
    # ``getLabwareDefinition`` from new clients. The internal helper
    # ``_labware_to_layout_dict`` is still needed by the overview-camera
    # registration code, which consumes the canvas-style dict.
    # ------------------------------------------------------------------

    @APIExport(requestType="GET")
    def set_led_status(self, status: str = "idle"):
        """
        Set LED matrix status if available.

        Args:
            status: Status string - "idle", "rainbow" (busy), "error", etc.
        """
        try:
            # Check if LED matrix manager is available
            if hasattr(self._master, 'LEDMatrixsManager'):
                led_names = self._master.LEDMatrixsManager.getAllDeviceNames()
                if led_names and len(led_names) > 0:
                    # Set status on first LED matrix
                    led_matrix = self._master.LEDMatrixsManager[led_names[0]]
                    led_matrix.setStatus(status=status)
                    self._logger.debug(f"LED status set to: {status}")
        except Exception as e:
            self._logger.debug(f"Could not set LED status: {e}")

    def _getExposureTimeSec(self) -> float:
        """Return the current detector exposure time in seconds (best effort).

        Priority order:
        1. DetectorManager.getParameter("exposure") — returns ms, clean API.
        2. _camera.exposure_time raw attribute — µs, camera-specific fallback.
        3. Hard-coded 0.1 s safe default.
        """
        try:
            exp_ms = self.mDetector.getParameter("exposure")
            if exp_ms is not None and float(exp_ms) > 0:
                return float(exp_ms) / 1000.0
        except Exception:
            pass
        try:
            return self.mDetector._camera.exposure_time / 1e6
        except Exception:
            return 0.1

    def _beginTriggeredAcquisition(self) -> bool:
        """Switch the detector to software-trigger mode for deterministic post-move grabs.

        In continuous (free-run) mode the camera ring-buffer may hold frames
        captured before the stage finished settling, so getLatestFrame() can
        return a stale frame.  In software-trigger mode every exposure is
        initiated by an explicit trigger fired *after* the stage move, making
        the returned frame guaranteed to be post-move.

        Returns True when the camera was successfully switched to software-
        trigger mode; False when the camera does not support it (graceful
        degradation — callers fall back to the frame-number polling path).
        """
        if not hasattr(self.mDetector, "setTriggerSource") or not hasattr(self.mDetector, "snapSync"):
            self._useTriggeredGrab = False
            return False
        try:
            ok = bool(self.mDetector.setTriggerSource("software"))
        except Exception as e:
            self._logger.warning(f"Could not enable software trigger: {e}")
            ok = False
        self._useTriggeredGrab = ok
        if ok:
            self._logger.info("ExperimentController: camera in software-trigger mode (deterministic grabs)")
        else:
            self._logger.warning("ExperimentController: software trigger not supported, using freerun fallback")
        return ok

    def _endTriggeredAcquisition(self) -> None:
        """Restore continuous (free-run) acquisition after a triggered sequence.

        Safe to call even when _beginTriggeredAcquisition() was not called or
        returned False — it is a no-op in that case.
        """
        if not getattr(self, "_useTriggeredGrab", False):
            return
        self._useTriggeredGrab = False
        try:
            if hasattr(self.mDetector, "setTriggerSource"):
                self.mDetector.setTriggerSource("continuous")
                self._logger.info("ExperimentController: camera restored to continuous mode")
        except Exception as e:
            self._logger.warning(f"Could not restore continuous mode: {e}")

    def grabCameraFrame(self, frameSync: int = 2, returnFrameNumber: bool = False):
        """Return a single camera frame guaranteed to be captured after this call.

        Fast path — software trigger (when _beginTriggeredAcquisition succeeded):
            Flushes the ring-buffer, fires one software trigger via
            detector.snapSync(), and returns exactly the frame that results.
            No pre-trigger frame can slip through.

        Fallback path — freerun frame-number polling:
            Flushes the ring-buffer, then polls getLatestFrame() until the
            frame-number has advanced by at least ``frameSync`` counts.
            Timeout is proportional to the current exposure time so long
            exposures do not abort prematurely.

        Args:
            frameSync: Minimum frame-ID increment to wait for in the fallback
                path.  2 is a safe default; increase if the illumination/gain
                change takes more than one frame period to settle.
            returnFrameNumber: When True, return a ``(frame, frame_number)``
                tuple instead of just the frame array.
        """
        # ── Fast path: triggered grab ─────────────────────────────────────
        if getattr(self, "_useTriggeredGrab", False) and hasattr(self.mDetector, "snapSync"):
            mFrame = self.mDetector.snapSync(timeout=max(2.0, self._getExposureTimeSec() * 4 + 1.0))
            if returnFrameNumber:
                fn = self.mDetector.getFrameNumber() if hasattr(self.mDetector, "getFrameNumber") else -1
                return mFrame, fn
            return mFrame

        # ── Fallback path: flush + frame-number polling ───────────────────
        # Discard frames captured before this call (e.g. during a stage move)
        # so the very next frame we receive is post-move.
        try:
            if hasattr(self.mDetector, "flushBuffer"):
                self.mDetector.flushBuffer()
        except Exception:
            pass

        exposure_s = self._getExposureTimeSec()
        timeoutFrameRequest = max(1.0, (frameSync + 2) * exposure_s + 0.5)
        cTime = time.time()
        lastFrameNumber = -1
        currentFrameNumber = -1
        mFrame = None

        while True:
            mFrame, currentFrameNumber = self.mDetector.getLatestFrame(returnFrameNumber=True)
            if lastFrameNumber == -1:
                # Anchor: remember the frame number at the start of the wait.
                lastFrameNumber = currentFrameNumber
            if time.time() - cTime > timeoutFrameRequest:
                # Timeout — return whatever we have rather than blocking forever.
                if mFrame is None:
                    mFrame = self.mDetector.getLatestFrame(returnFrameNumber=False)
                self._logger.warning(
                    f"grabCameraFrame: timed out after {timeoutFrameRequest:.1f}s "
                    f"(exposure={exposure_s*1000:.0f}ms, frameSync={frameSync})"
                )
                break
            if currentFrameNumber <= lastFrameNumber + frameSync:
                time.sleep(0.01)  # yield CPU while waiting
            else:
                break

        if returnFrameNumber:
            return mFrame, currentFrameNumber
        return mFrame

    def acquire_frame(self, channel: str, frameSync: int = 0):
        """Acquire a single frame deterministically.

        Delegates entirely to grabCameraFrame().  When software-trigger mode
        was activated via _beginTriggeredAcquisition(), the frame is guaranteed
        to have been exposed after this call (atomic fire-and-return).
        Otherwise the freerun fallback is used (flush + frame-number wait).

        Args:
            channel: Illumination channel name (used for logging only; the
                caller is expected to have already configured illumination).
            frameSync: Passed through to grabCameraFrame's fallback path.
        """
        self._logger.debug(f"Acquiring frame on channel {channel}")

        # CAN-bus power darkness: cut bus power (→ complete darkness), let it
        # settle, expose, then restore power and settle again. Wraps each frame
        # so long luminescence exposures see zero stray light. Motors keep their
        # position across the power cycle, so no re-homing is needed.
        if getattr(self, "_busPowerDarkness", False):
            settle = getattr(self, "_busPowerSettle", 2.0)
            self._setBusPower(False)
            if settle > 0:
                time.sleep(settle)
            try:
                frame = self.grabCameraFrame(frameSync=frameSync)
            finally:
                self._setBusPower(True)
                if settle > 0:
                    time.sleep(settle)
            return frame

        return self.grabCameraFrame(frameSync=frameSync)

    def _setBusPower(self, enable: bool) -> None:
        """Enable/disable the high-current CAN-bus power that feeds the slaves.

        Best-effort: routed through the UC2Config controller. Logged and
        swallowed when UC2Config/ESP32 is unavailable (e.g. simulation) so the
        acquisition is never aborted by a missing bus."""
        try:
            uc2 = self._master.getController('UC2Config')
        except Exception:
            uc2 = None
        if uc2 is None or not hasattr(uc2, "setBusPower"):
            self._logger.warning(
                "Bus-power darkness requested but UC2Config.setBusPower is "
                "unavailable — skipping.")
            return
        try:
            uc2.setBusPower(enable=bool(enable))
            self._logger.debug(f"CAN-bus power {'ON' if enable else 'OFF'}")
        except Exception as e:
            self._logger.warning(f"setBusPower({enable}) failed: {e}")

    def set_exposure_time_gain(self, exposure_time: float, gain: float, context: WorkflowContext, metadata: Dict[str, Any]):
        # Set gain and exposure via the shared attribute signal.
        # The signal triggers the detector driver to apply the new values,
        # but this is asynchronous – we need to wait briefly so the detector
        # registers the change before the next frame is captured.
        changed = False
        if gain is not None and gain >= 0:
            self._commChannel.sharedAttrs.sigAttributeSet(['Detector', None, None, "gain"], gain)
            self._master.getController('Settings').setDetectorGain(None, gain)  # Ensure SettingsController is updated, TODO: we have to pass the correct detectorname in the future
            self._logger.debug(f"Setting gain to {gain}")
            changed = True
        if exposure_time is not None and exposure_time > 0:
            self._commChannel.sharedAttrs.sigAttributeSet(['Detector', None, None, "exposureTime"], exposure_time)
            self._master.getController('Settings').setDetectorExposureTime(None, exposure_time)  # Ensure SettingsController is updated, TODO: we have to pass the correct detectorname in the future
            self._logger.debug(f"Setting exposure time to {exposure_time}")
            changed = True

        # Give the detector enough time to apply the new register values.
        # Most camera drivers need at least one frame period to latch new settings.
        if False and changed: # TODO: we should probably make this dependent on the exposure time - for very short exposures we can get away with a shorter wait, but for long exposures we need to wait longer - maybe something like max(0.1, exposure_time*1.5) or similar
            # Wait proportional to exposure time so slow exposures have enough
            # settle time, but cap at a reasonable maximum.
            settle_time = min(max(0.05, (exposure_time or 50) / 1000.0), 0.5)
            time.sleep(settle_time)
            # Discard one stale frame that was captured with old settings
            try:
                self.mDetector.getLatestFrame(returnFrameNumber=False)
            except Exception:
                pass

    def set_laser_power(self, power: float, channel: str):
        if channel not in self.allIlluNames:
            self._logger.error(f"Channel {channel} not found in available lasers: {self.allIlluNames}")
            return None
        self._master.lasersManager[channel].setValue(power, getReturn=True)
        # Use `not enabled` instead of `== 0` to handle False/None/0 returned
        # after _switch_off_all_illumination calls setEnabled(False).
        if power > 0 and not self._master.lasersManager[channel].enabled:
            self._master.lasersManager[channel].setEnabled(1, getReturn=True)
        self._logger.debug(f"Setting laser power to {power} for channel {channel}")
        time.sleep(0.04)  # Short delay to ensure power is set before next acquisition # TODO: Necessary?
        return power

    def set_led_matrix_pattern(
        self,
        kind: str,
        direction: Optional[str] = None,
        radius: Optional[int] = None,
        intensity_r: int = 0,
        intensity_g: int = 0,
        intensity_b: int = 0,
        settle_s: float = 0.05,
    ):
        """Drive the LED matrix for a Wellplate-designer synthetic channel.

        Args:
            kind: "ring" | "halves" | "off".  "halves" is the DPC quadrant
                primitive (combined with ``direction``).
            direction: For ``kind="halves"``: "top" | "bottom" | "left" | "right".
            radius: For ``kind="ring"``: ring radius in LED units.
            intensity_r/g/b: 0..255 per colour channel.
            settle_s: Sleep after setting so illumination stabilises before
                the camera trigger fires.  setHalves/setRing return as soon
                as the command is acked over serial, so a short settle is
                important to avoid the first row of the rolling-shutter
                frame capturing the previous pattern.
        """
        if self._ledMatrix is None:
            self._logger.warning(
                f"set_led_matrix_pattern({kind}) called but no LED matrix is configured."
            )
            return None
        try:
            if kind == "off":
                self._ledMatrix.setAll(state=(0, 0, 0), getReturn=False)
            elif kind == "ring":
                if radius is None:
                    self._logger.error("set_led_matrix_pattern(ring): missing 'radius'")
                    return None
                self._ledMatrix.setRing(
                    radius=int(radius),
                    intensity=(int(intensity_r), int(intensity_g), int(intensity_b)),
                )
            elif kind == "halves":
                if direction is None or direction not in self.DPC_SUB_CHANNELS:
                    self._logger.error(
                        f"set_led_matrix_pattern(halves): direction must be one of "
                        f"{self.DPC_SUB_CHANNELS}, got {direction!r}"
                    )
                    return None
                self._ledMatrix.setHalves(
                    intensity=(int(intensity_r), int(intensity_g), int(intensity_b)),
                    region=direction,
                )
            else:
                self._logger.error(f"set_led_matrix_pattern: unknown kind {kind!r}")
                return None
        except Exception as e:  # noqa: BLE001 - hardware errors surfaced as log warnings
            self._logger.warning(f"set_led_matrix_pattern({kind}) failed: {e}")
            return None

        # Settle so the camera doesn't read mid-transition (rolling shutter
        # captures the previous pattern on the top rows otherwise).
        if settle_s > 0:
            time.sleep(settle_s)
        return kind


    def home_axis(self, axis: str, isBlocking: bool = True):
        if axis not in ["X", "Y", "Z"]:
            self._logger.error(f"Invalid axis '{axis}' specified for movement")
            return None
        self._logger.debug(f"Moving axis {axis} to home position with blocking={isBlocking}")
        self.mStage.doHome(axis=axis, isBlocking=isBlocking)

    @APIExport(requestType="POST")
    def homeAllAxes(self):
        """Home all stage axes (X, Y, Z) sequentially. Blocks until complete."""
        self._logger.info("Homing all axes before experiment...")
        for axis in ["X", "Y", "Z"]:
            self.home_axis(axis=axis, isBlocking=True)
        self._logger.info("All axes homed successfully.")
        return {"status": "ok", "message": "All axes homed"}


    def move_stage_xy(self, posX: float = None, posY: float = None, relative: bool = False):
        # {"task":"/motor_act",     "motor":     {         "steppers": [             { "stepperid": 1, "position": -1000, "speed": 30000, "isabs": 0, "isaccel":1, "isen":0, "accel":500000}     ]}}
        self._logger.info(f"Moving stage to X={posX}, Y={posY}")
        #if posY and posX is None:
        # Use the experiment-configured scan speed (set from the frontend in
        # startWellplateExperiment); falls back to the defaults in __init__ when
        # no experiment has overridden it.
        # A single coordinated XY command is preferred (no dog-leg between
        # tiles), but stages that cannot do it — e.g. MMCore — would silently
        # drop an axis="XY" move, so those fall back to sequential axes.
        if "XY" in (getattr(self.mStage, "combinedAxes", None) or []):
            self.mStage.move(value=(posX, posY), speed=(self.SPEED_X, self.SPEED_Y), axis="XY", is_absolute=not relative, is_blocking=True, acceleration=(self.ACCELERATION, self.ACCELERATION))
        else:
            for axis, value, speed in (("X", posX, self.SPEED_X), ("Y", posY, self.SPEED_Y)):
                if value is None:
                    continue
                self.mStage.move(value=value, speed=speed, axis=axis, is_absolute=not relative, is_blocking=True, acceleration=self.ACCELERATION)
        #newPosition = self.mStage.getPosition()
        #self._commChannel.sigUpdateMotorPosition.emit([posX, posY])
        return (posX, posY) # TODO: Need to adjust in case of relative move

    def move_stage_z(self, posZ: float, relative: bool = False, maxSpeedZ=5000,
                     af_region_id: Optional[str] = None):
        # When af_region_id is given (used by the per-plane capture moves in
        # the acquisition workflow), add the runtime autofocus Z offset on top
        # of the absolute target so the measured focus is actually applied to
        # acquisitions instead of being overwritten by this absolute move.
        if af_region_id is not None and not relative:
            posZ = posZ + float(self._experiment_af_offsets.get(af_region_id, 0.0))
        self._logger.info(f"Moving stage to Z={posZ}")
        self.mStage.move(value=posZ, speed=np.min((self.SPEED_Z, maxSpeedZ)), axis="Z", is_absolute=not relative, is_blocking=True, acceleration=self.ACCELERATION_Z)
        #newPosition = self.mStage.getPosition()
        #self._commChannel.sigUpdateMotorPosition.emit([newPosition["Z"]])
        return posZ # TODO: Need to adjust in case of relative move

    def set_detector_parameter(self, parameter: str, value: Any):
        """Set a detector parameter."""
        try:
            if hasattr(self.mDetector, 'setParameter'):
                self.mDetector.setParameter(parameter, value)
            elif hasattr(self.mDetector, '_camera') and hasattr(self.mDetector._camera, 'setParameter'):
                self.mDetector._camera.setParameter(parameter, value)
            else:
                self._logger.warning(f"Cannot set detector parameter {parameter} - method not available")
        except Exception as e:
            self._logger.error(f"Error setting detector parameter {parameter} to {value}: {e}")

    def return_to_initial_position(self, include_z_position: bool = False):
        """Return the stage to the position stored at experiment start."""
        try:
            if hasattr(self, "_initial_experiment_position") and self._initial_experiment_position:
                pos = self._initial_experiment_position
                self._logger.info(
                    "Returning to initial position: X=%.2f, Y=%.2f, Z=%.2f",
                    pos["X"], pos["Y"], pos["Z"],
                )
                self.move_stage_xy(pos["X"], pos["Y"], relative=False)
                if include_z_position: self.move_stage_z(pos["Z"], relative=False)
                self._initial_experiment_position = None
            else:
                self._logger.debug("No initial experiment position stored, skipping return.")
        except Exception as e:
            self._logger.warning("Failed to return to initial position: %s", e)

    def _switch_off_all_illumination(self) -> None:
        """
        Turn off all illumination sources before starting scan.
        This ensures clean state for hardware-controlled illumination.
        """
        try:
            # Try to access laser manager
            if hasattr(self, '_master') and hasattr(self._master, 'lasersManager'):
                for laser_name in self._master.lasersManager.getAllDeviceNames():
                    try:
                        self._master.lasersManager[laser_name].setEnabled(False)
                        self._master.lasersManager[laser_name].setValue(0)
                    except Exception as e:
                        self._logger.debug(f"Could not turn off laser {laser_name}: {e}")

            # Also clear the LED matrix/ring so a leftover ring/DPC pattern from
            # a previous acquisition does not survive into this scan. Normal
            # mode previously only reset the lasers here (performance mode reset
            # the matrix separately); this brings normal mode to parity.
            # set_led_matrix_pattern self-guards when no matrix is configured.
            if getattr(self, "_ledMatrix", None) is not None:
                self.set_led_matrix_pattern(kind="off", settle_s=0)

            self._logger.debug("All illumination sources switched off before scan")
        except Exception as e:
            self._logger.warning(f"Error switching off illumination: {e}")

    def _read_i2c_snapshot(self, min_interval_s: float = 1.0) -> Optional[Dict[str, Any]]:
        """Read the latest I2C environmental sensor values, or None.

        Returns the reading dict from
        ``I2CSensorController.getLatestI2CSensorValues()`` (keys such as
        ``temperature_c`` / ``humidity_pct`` / ``lux``), or ``None`` when no
        I2C sensor controller is configured or the read fails.

        Reads are throttled: within ``min_interval_s`` of the last read the
        cached snapshot is returned instead of hitting the hardware again, so
        embedding this per-frame never bottlenecks a fast raster scan. Never
        raises.
        """
        now = time.monotonic()
        cached = getattr(self, "_i2c_last_snapshot", None)
        last_ts = getattr(self, "_i2c_last_read_ts", None)
        if cached is not None and last_ts is not None and (now - last_ts) < min_interval_s:
            return cached
        try:
            i2c = self._master.getController('I2CSensor')
        except Exception:
            i2c = None
        reading = None
        if i2c is not None:
            try:
                reading = i2c.getLatestI2CSensorValues()
            except Exception as e:
                self._logger.debug(f"I2C sensor read failed: {e}")
                reading = None
        self._i2c_last_snapshot = reading
        self._i2c_last_read_ts = now
        self._i2c_read_seq = getattr(self, "_i2c_read_seq", 0) + 1
        return reading

    def _log_i2c_row(self, reading: Dict[str, Any], time_index: int = 0) -> None:
        """Append one environmental-sensor row to the experiment sidecar CSV.

        No-op when no CSV path has been set up for this experiment (i.e. no
        I2C controller was detected at start) or the reading is empty.
        """
        csv_path = getattr(self, "_i2c_csv_path", None)
        if not csv_path or not reading:
            return
        try:
            import csv as _csv
            write_header = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as fh:
                writer = _csv.writer(fh)
                if write_header:
                    writer.writerow([
                        "datetime", "timestamp", "time_index",
                        "temperature_c", "humidity_pct", "lux", "ch0_full", "ch1_ir",
                    ])
                writer.writerow([
                    reading.get("datetime"), reading.get("timestamp"), time_index,
                    reading.get("temperature_c"), reading.get("humidity_pct"),
                    reading.get("lux"), reading.get("ch0_full"), reading.get("ch1_ir"),
                ])
        except Exception as e:
            self._logger.debug(f"Could not append I2C sensor CSV row: {e}")

    # -------------------------------------------------------------------------
    # internal helpers
    # -------------------------------------------------------------------------
    def _start_i2c_logging(self, dir_path: str) -> None:
        """Optional I2C sensor sidecar CSV next to the data; a no-op without a sensor controller."""
        self._i2c_csv_path = None
        self._i2c_last_snapshot = None
        self._i2c_last_read_ts = None
        self._i2c_read_seq = 0
        self._i2c_logged_seq = -1
        try:
            if self._read_i2c_snapshot(min_interval_s=0.0) is not None:
                self._i2c_csv_path = os.path.join(dir_path, "i2c_sensor_log.csv")
                self._logger.info(f"I2C sensor logging enabled → {self._i2c_csv_path}")
        except Exception:
            pass
