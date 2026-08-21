"""
ImSwitch :class:`DetectorManager` that wraps a Micro-Manager camera device
through ``pymmcore-plus``.

Two configuration modes are supported via the ``managerProperties`` block of
the setup JSON:

* **Config mode** – set ``cfgPath`` to a Micro-Manager ``.cfg`` file. The
  file is loaded once and reused by every MMCore* manager pointing at it.
* **Manual mode** – set ``adapterName`` and ``deviceName`` (e.g.
  ``"DemoCamera"`` / ``"DCam"``). The device is loaded with
  ``loadDevice`` + ``initializeDevice`` directly. Useful for quick demos and
  for setups where a single device is needed without authoring a ``.cfg``.

Recognised ``managerProperties`` keys:

============  ========  =========================================================
Key           Required  Description
============  ========  =========================================================
cfgPath       no*       Path to a ``.cfg`` file (mutually exclusive with the
                        manual-mode keys).
adapterPath   no        Override the adapter search directory.
adapterName   no*       Adapter to load in manual mode, e.g. ``"DemoCamera"``.
deviceName    no*       Device name inside the adapter, e.g. ``"DCam"``.
deviceLabel   no        Label assigned to the loaded device (default
                        ``"Camera"``).
============  ========  =========================================================

\\* Either ``cfgPath`` or both ``adapterName`` + ``deviceName`` must be
provided.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

import numpy as np

from imswitch.imcommon.model import initLogger
from imswitch.imcontrol.model.managers import MMCoreManager
from .DetectorManager import (
    DetectorManager,
    DetectorListParameter,
    DetectorNumberParameter,
    DetectorParameter,
)
import pymmcore 

# Property names we never expose through the parameter UI – internal MMCore
# bookkeeping or things that would confuse the generic editor.
_SKIP_PROPERTY_PREFIXES = ("On",)
_SKIP_PROPERTY_SUBSTRINGS = ("TransposeCorrection",)

# Properties surfaced in the "base" (non-expert) view. Everything else is
# flagged ``advanced`` so the frontend can collapse it behind an "Expert" toggle.
# Matching is case-insensitive on the raw MMCore property name.
_BASE_PROPERTY_NAMES = {
    "exposure",
    "gain",
    "emgain",
    "binning",
    "pre-amp-gain",
    "output_amplifier",
    "readoutmode",
}

# Property names (case-insensitive) that report the sensor/CCD temperature.
_TEMPERATURE_PROPERTY_NAMES = ("ccdtemperature", "temperature", "sensortemperature")
_TEMPERATURE_SETPOINT_NAMES = ("ccdtemperaturesetpoint", "temperaturesetpoint")

# Longest exposure for which the live-view poll may still fall back to a
# blocking ``core.snap()``. ``snapImage()`` sits inside MMCore's device lock
# for the whole integration, so anything longer would stall every other MMCore
# call in the process (temperature polls, parameter reads, a running snap job)
# and make the backend look frozen. Longer exposures must go through
# ``acquireSingleFrame()`` / ``snapSync()`` instead.
_IDLE_SNAP_MAX_EXPOSURE_S = 2.0

# Minimum spacing between two "the idle snap failed" log lines. The live-view
# loop polls many times a second; unthrottled, a camera that rejects the snap
# (Andor DRV_ACQUIRING/DRV_IDLE) buries every other message in the console.
_SNAP_ERROR_LOG_INTERVAL_S = 5.0


def _is_internal_property(prop: str) -> bool:
    if any(prop.startswith(p) for p in _SKIP_PROPERTY_PREFIXES):
        return True
    if any(s in prop for s in _SKIP_PROPERTY_SUBSTRINGS):
        return True
    return False


def _is_base_property(prop: str) -> bool:
    return prop.strip().lower() in _BASE_PROPERTY_NAMES


class _MMCoreCameraProxy:
    """Live compatibility shim for the hardware ``_camera`` attribute surface.

    The rest of ImSwitch (ExperimentController, ObjectiveController,
    WorkflowController, LightsheetController, ...) reaches into
    ``detector._camera`` expecting the same handful of attributes that the
    concrete hardware interfaces (``hikcamera``, ``gxipycamera``, ...) expose.
    Previously this was a static dummy object with hard-coded values that never
    tracked the real device state. This proxy forwards every access to the live
    MMCore state instead, so the values are always current.

    IMPORTANT — units: ``exposure_time`` is expressed in **microseconds**, to
    match every consumer in the codebase. They all compute
    ``_camera.exposure_time / 1e6`` to get seconds, mirroring ``hikcamera``
    where the default ``10000`` means 10 ms. MMCore's own ``getExposure()``
    returns *milliseconds*, so we convert here. Getting this wrong makes the
    ExperimentController mis-estimate frame timing by 1000x (e.g. long
    exposures appear not to block).
    """

    def __init__(self, manager: "MMCoreDetectorManager"):
        self._m = manager

    @property
    def SensorWidth(self) -> int:
        return int(self._m.fullShape[0])

    @property
    def SensorHeight(self) -> int:
        return int(self._m.fullShape[1])

    @property
    def isRGB(self) -> bool:
        return bool(self._m.isRGB)

    @property
    def exposure_time(self) -> float:
        """Current exposure in **microseconds** (see class docstring)."""
        try:
            return float(self._m._core.getExposure()) * 1000.0
        except Exception:
            return 10000.0

    @exposure_time.setter
    def exposure_time(self, value_us: float) -> None:
        try:
            self._m._core.setExposure(float(value_us) / 1000.0)
        except Exception:
            self._m._logger.error("Failed to set exposure via proxy", exc_info=True)

    def getLatestFrame(self, *args, **kwargs):
        return self._m.getLatestFrame(*args, **kwargs)

    def setParameter(self, name, value):
        return self._m.setParameter(name, value)

    def getStreamDiagnostics(self) -> Dict[str, Any]:
        return self._m.getStreamDiagnostics()


class MMCoreDetectorManager(DetectorManager):
    """Detector manager backed by a Micro-Manager camera device."""

    def __init__(self, detectorInfo, name, **lowLevelManagers):
        self._logger = initLogger(self, instanceName=name)
        self._props: Dict = dict(detectorInfo.managerProperties or {})

        cfg_path = self._props.get("cfgPath")
        adapter_name = self._props.get("adapterName")
        device_name = self._props.get("deviceName")
        adapter_path = self._props.get("adapterPath")
        adapter_paths = [adapter_path] if adapter_path else None
        # Live compatibility shim used by the rest of ImSwitch (see class docstring).
        self._camera = _MMCoreCameraProxy(self)
        # Per-parameter metadata (min/max limits, advanced flag, MMCore type)
        # keyed by property name. Populated by _build_parameters().
        self._paramMeta: Dict[str, Dict[str, Any]] = {}
        # Device values captured at first init, used by "reset to default".
        self._factoryDefaults: Dict[str, str] = {}
        self._label: str = self._props.get("deviceLabel", "Camera")

        if cfg_path:
            self._core = MMCoreManager.ensure_loaded(cfg_path, adapter_paths)
            cam = self._core.getCameraDevice()
            if cam:
                self._label = cam
            try:
                self._core.setCameraDevice(self._label)
            except Exception:
                self._logger.warning(
                    f"Could not set camera device to '{self._label}'", exc_info=True,
                )
        elif adapter_name and device_name:
            self._core = MMCoreManager.ensure_core(adapter_paths)
            if self._label not in self._core.getLoadedDevices():
                self._core.loadDevice(self._label, adapter_name, device_name)
                self._core.initializeDevice(self._label)
            self._core.setCameraDevice(self._label)
        else:
            raise ValueError(
                f"MMCoreDetectorManager '{name}' requires either 'cfgPath' or "
                "both 'adapterName' and 'deviceName' in managerProperties."
            )

        # Sensor info
        try:
            full_shape = (int(self._core.getImageWidth()), int(self._core.getImageHeight()))
        except Exception:
            # Snap once to make sure the camera reports its geometry.
            self._core.snap()
            full_shape = (int(self._core.getImageWidth()), int(self._core.getImageHeight()))

        # Real sensor geometry drives fullShape / _camera.SensorWidth|Height
        # (the proxy reads them back from fullShape). pixelSize is [Z, Y, X].
        self.pixelSize = [1.0, 1.0, 1.0]

        # Binning options
        supported_binnings: List[int] = self._read_supported_binnings()

        # Build parameter dict
        parameters = self._build_parameters()

        try:
            current_exposure = float(self._core.getExposure())
        except Exception:
            current_exposure = 10.0
        parameters["Exposure"] = DetectorNumberParameter(
            group="Acquisition",
            value=current_exposure,
            editable=True,
            valueUnits="ms",
        )
        # Exposure is a base (non-expert) parameter; expose limits if the
        # adapter advertises an "Exposure" property with limits.
        exposure_limits = self._read_property_limits("Exposure")
        self._paramMeta["Exposure"] = {
            "advanced": False,
            "min": exposure_limits["min"],
            "max": exposure_limits["max"],
        }

        # additional flags
        self._running = False
        self._frameNunber = 0

        # Concurrency + trigger-mode state.
        #
        # ``_grabLock`` serializes ALL access to the MMCore core object across
        # threads (the live-view StreamWorker polling getLatestFrame vs. an
        # experiment thread firing snapSync). Without it the Andor adapter
        # raises DRV_ACQUIRING (20072) when two snaps overlap.
        #
        # ``_triggerMode`` says which acquisition style is in effect (a
        # background job takes ownership through acquireExclusive() instead,
        # see below):
        #   * "continuous" (default) — live view; getLatestFrame reads the
        #     circular buffer.
        #   * "software" — an experiment owns the camera and fires explicit
        #     snapSync() exposures. getLatestFrame then NEVER touches the
        #     hardware; it just replays the last cached frame so the live view
        #     keeps displaying without competing for the sensor.
        #
        # ``_latestFrame`` caches the most recent frame (from either the live
        # buffer or a snapSync) so the live view has something to show even
        # while the experiment holds the camera.
        self._grabLock = threading.RLock()
        self._triggerMode = "continuous"
        self._latestFrame: Optional[np.ndarray] = None

        # Rolling measurement of how fast the CAMERA actually delivers frames
        # into MMCore's circular buffer (frames popped / wall-clock). This is
        # the true camera rate, independent of our poll/encode rate — use it to
        # tell "the camera is slow" (cameraFps low) from "downstream is slow".
        self._camStats: Dict[str, Any] = {"fps": 0.0, "delivered": 0, "last_t": None}

        # Exclusive-access bookkeeping. A background job (long-exposure snap,
        # experiment step, ...) claims the sensor through acquireExclusive();
        # while the claim is held, getLatestFrame() replays the cache and
        # issues no hardware call at all.
        #
        # Without this the live-view poll thread and the job drove the same
        # MMCore instance concurrently, which broke both of them:
        #   * with a sequence running, the poll drained the circular buffer and
        #     swallowed the job's single frame, so the job waited for an image
        #     that was already gone -> "stuck" acquisition;
        #   * with the sequence stopped, the poll's fallback core.snap() either
        #     collided with the job (Andor 20072/20073 error spam) or blocked
        #     inside MMCore's device lock for the entire exposure, stalling
        #     every other API request -> frozen backend, no progress updates.
        self._exclusiveLock = threading.Lock()
        self._exclusiveOwner: Optional[str] = None
        self._exclusiveSince: float = 0.0
        self._exclusiveRestoreLive = False

        # Throttling state for the idle-mode fallback snap and its error log.
        self._lastIdleSnap = 0.0
        self._lastSnapErrorLog = 0.0
        self._snapErrorCount = 0

        super().__init__(
            detectorInfo,
            name,
            fullShape=full_shape,
            supportedBinnings=supported_binnings,
            model=self._label or "MMCore Camera",
            parameters=parameters,
            croppable=True,
        )
        #
        self._logger.info(f"MMCoreDetectorManager '{name}' initialized for device '{self._label}'")

    # ------------------------------------------------------------------
    # Parameter discovery helpers
    # ------------------------------------------------------------------
    def _read_supported_binnings(self) -> List[int]:
        try:
            allowed = list(self._core.getAllowedPropertyValues(self._label, "Binning"))
        except Exception:
            return [1]

        binnings: List[int] = []
        for entry in allowed:
            try:
                # Binning is sometimes "1" and sometimes "1x1".
                txt = str(entry).lower().split("x")[0]
                binnings.append(int(txt))
            except (TypeError, ValueError):
                continue
        return sorted(set(binnings)) or [1]

    def _read_property_limits(self, prop: str) -> Dict[str, Optional[float]]:
        """Return {'min': .., 'max': ..} for a numeric property, if MMCore
        advertises limits for it. Values are ``None`` when not available."""
        lo: Optional[float] = None
        hi: Optional[float] = None
        try:
            if bool(self._core.hasPropertyLimits(self._label, prop)):
                lo = float(self._core.getPropertyLowerLimit(self._label, prop))
                hi = float(self._core.getPropertyUpperLimit(self._label, prop))
        except Exception:
            lo = hi = None
        return {"min": lo, "max": hi}

    def _build_parameters(self) -> Dict[str, DetectorParameter]:
        parameters: Dict[str, DetectorParameter] = {}
        self._paramMeta = {}
        self._factoryDefaults = {}
        try:
            prop_names = list(self._core.getDevicePropertyNames(self._label))
        except Exception:
            return parameters

        for prop in prop_names:
            if _is_internal_property(prop):
                continue
            try:
                value = self._core.getProperty(self._label, prop)
            except Exception:
                continue

            try:
                read_only = bool(self._core.isPropertyReadOnly(self._label, prop))
            except Exception:
                read_only = False

            allowed: List[str] = []
            try:
                allowed = list(self._core.getAllowedPropertyValues(self._label, prop))
            except Exception:
                allowed = []

            # Remember the device's own default so "reset to default" can
            # restore it later (editable properties only).
            if not read_only:
                self._factoryDefaults[prop] = str(value)

            meta: Dict[str, Any] = {
                "advanced": not _is_base_property(prop),
                "min": None,
                "max": None,
            }

            if allowed:
                parameters[prop] = DetectorListParameter(
                    group="MMCore",
                    value=str(value),
                    editable=not read_only,
                    options=[str(a) for a in allowed],
                )
                self._paramMeta[prop] = meta
                continue

            try:
                num_value = float(value)
            except (TypeError, ValueError):
                # Skip free-form strings – they don't map onto our UI widgets.
                continue

            limits = self._read_property_limits(prop)
            meta.update(limits)
            parameters[prop] = DetectorNumberParameter(
                group="MMCore",
                value=num_value,
                editable=not read_only,
                valueUnits="",
            )
            self._paramMeta[prop] = meta
        return parameters

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------
    def _store_latest(self, frame: np.ndarray, number: Optional[int] = None) -> None:
        """Cache the most recent frame + its frame number."""
        self._latestFrame = frame
        if number is not None:
            self._frameNunber = int(number)
        else:
            self._frameNunber += 1

    def _update_camera_fps(self, popped: int) -> None:
        """Update the rolling estimate of the true camera delivery rate from
        the number of frames drained in this poll."""
        if popped <= 0:
            return
        now = time.time()
        st = self._camStats
        st["delivered"] += popped
        last_t = st["last_t"]
        if last_t is not None:
            dt = now - last_t
            if dt > 0:
                inst = popped / dt
                st["fps"] = inst if st["fps"] == 0.0 else 0.8 * st["fps"] + 0.2 * inst
        st["last_t"] = now

    def _return_stored(self, returnFrameNumber: bool):
        """Return the cached frame (or a black placeholder if none yet)."""
        frame = self._latestFrame
        if frame is None:
            placeholder = np.zeros(self._shape, dtype=np.uint16)
            return (placeholder, -1) if returnFrameNumber else placeholder
        return (frame, self._frameNunber) if returnFrameNumber else frame

    def _log_snap_error(self, exc: Exception) -> None:
        """Log a failed idle snap at most once per ``_SNAP_ERROR_LOG_INTERVAL_S``.

        The live-view loop calls the idle path many times a second. When the
        device rejects the snap — Andor raises 20072/20073 whenever another
        acquisition owns the sensor — an unthrottled logger produces hundreds
        of identical tracebacks per second and hides everything else.
        """
        self._snapErrorCount += 1
        now = time.time()
        if now - self._lastSnapErrorLog < _SNAP_ERROR_LOG_INTERVAL_S:
            return
        suppressed = self._snapErrorCount - 1
        self._lastSnapErrorLog = now
        self._snapErrorCount = 0
        extra = f" ({suppressed} further failures suppressed)" if suppressed > 0 else ""
        self._logger.warning(f"Idle snap from MMCore failed: {exc}{extra}")

    def getLatestFrame(self, returnFrameNumber=False) -> np.ndarray:
        """Return the most recent frame for display / non-deterministic reads.

        This method NEVER fires a new exposure while somebody else owns the
        camera — an experiment (``_triggerMode == "software"``) or a background
        job holding an exclusive claim (``acquireExclusive``). In those cases it
        simply replays the last cached frame, so the live-view poll loop cannot
        compete with the owner for the sensor. That competition was what
        produced the Andor DRV_ACQUIRING (20072/20073) spam, the endless
        ``Frame number: -1``, and snap jobs that never finished because the poll
        had already drained their frame out of the circular buffer.

        In continuous (live) mode it drains the circular buffer down to the
        NEWEST frame (discarding stale FIFO frames) so live view never
        accumulates latency. Only when the camera is genuinely idle (no live
        sequence, no owner) does it fall back to a single snap — throttled, and
        only for exposures short enough that blocking in the device lock is
        harmless.
        """
        # ── Somebody else owns the sensor: replay cache, touch no hardware ──
        if self._triggerMode == "software" or self._exclusiveOwner is not None:
            return self._return_stored(returnFrameNumber)

        # Never queue the poll thread up behind a grab that is already in
        # flight — piling threads on the lock only adds latency, and the frame
        # they would eventually read is the one we already have cached.
        if not self._grabLock.acquire(blocking=False):
            return self._return_stored(returnFrameNumber)
        try:
            try:
                sequence_running = bool(self._core.isSequenceRunning())
            except Exception:
                sequence_running = False

            if sequence_running:
                # Drain to the newest frame; keep only the last one.
                newest = None
                newest_num = None
                popped = 0
                try:
                    while self._core.getRemainingImageCount() > 0:
                        md = pymmcore.Metadata()
                        # popNextImageMD pops the OLDEST image and fills md.
                        newest = self._core.popNextImageMD(md)
                        popped += 1
                        if md.HasTag(pymmcore.g_Keyword_Metadata_ImageNumber):
                            try:
                                newest_num = int(
                                    md.GetSingleTag(
                                        pymmcore.g_Keyword_Metadata_ImageNumber
                                    ).GetValue()
                                )
                            except Exception:
                                newest_num = None
                except Exception:
                    self._logger.debug(
                        "Failed to drain sequence buffer", exc_info=True)
                if newest is not None:
                    # Without a camera ImageNumber tag, advance by the number of
                    # frames actually drained (not just +1) so the counter
                    # reflects true throughput.
                    if newest_num is None:
                        newest_num = self._frameNunber + popped
                    self._store_latest(np.asarray(newest), newest_num)
                # Measure real camera delivery rate from frames actually drained.
                self._update_camera_fps(popped)
                # Return the cached latest whether or not a new frame arrived,
                # so live view shows a stable image instead of black flicker.
                return self._return_stored(returnFrameNumber)

            # Idle (no live sequence, nobody owns the sensor): fall back to a
            # single fresh snap so occasional on-demand callers still get real
            # data. Guarded twice over, because core.snap() blocks inside
            # MMCore's device lock for the whole integration:
            #   * skip it entirely for long exposures — a multi-minute Andor
            #     snap fired from the poll thread freezes every other MMCore
            #     call in the process;
            #   * fire at most one per exposure period, so a camera that keeps
            #     refusing cannot be hammered at poll rate.
            try:
                exposure_s = float(self._core.getExposure()) / 1000.0
            except Exception:
                exposure_s = 0.1
            if exposure_s > _IDLE_SNAP_MAX_EXPOSURE_S:
                return self._return_stored(returnFrameNumber)
            now = time.time()
            if now - self._lastIdleSnap < exposure_s:
                return self._return_stored(returnFrameNumber)
            self._lastIdleSnap = now
            try:
                image = np.asarray(self._core.snap())
                self._store_latest(image)
            except Exception as exc:  # noqa: BLE001 - throttled below
                self._log_snap_error(exc)
            return self._return_stored(returnFrameNumber)
        finally:
            self._grabLock.release()

    def getChunk(self) -> np.ndarray:
        frames = []
        try:
            while self._core.getRemainingImageCount() > 0:
                frames.append(self._core.popNextImage())
        except Exception:
            self._logger.error("Failed to drain MMCore buffer", exc_info=True)

        if not frames:
            return np.empty((0, self._shape[1], self._shape[0]))
        return np.stack(frames, axis=0)

    def flushBuffers(self) -> None:
        try:
            self._core.clearCircularBuffer()
        except Exception:
            self._logger.warning("Could not clear circular buffer", exc_info=True)

    # ------------------------------------------------------------------
    # Acquisition control
    # ------------------------------------------------------------------
    def startAcquisition(self) -> None:
        # Safety rule: while an experiment owns the camera (software-trigger
        # mode) we must NOT start a competing continuous sequence — that is the
        # exact collision that crashed the Andor. Live view still works: it
        # replays cached frames via getLatestFrame() in software mode.
        if getattr(self, "_triggerMode", "continuous") == "software":
            self._logger.debug(
                "startAcquisition ignored — camera in software-trigger mode")
            return
        # Same rule for a background job that has claimed the sensor: starting
        # a continuous sequence underneath it would feed the live-view poll the
        # job's own frames and leave the job waiting for an image that was
        # already consumed.
        if getattr(self, "_exclusiveOwner", None) is not None:
            self._logger.debug(
                f"startAcquisition ignored — camera claimed by "
                f"'{self._exclusiveOwner}'")
            return
        with self._grabLock:
            if not self._core.isSequenceRunning():
                # Reset the delivery-rate meter for this live session.
                self._camStats = {"fps": 0.0, "delivered": 0, "last_t": None}
                self._core.startContinuousSequenceAcquisition(0)
                self._running = True
                self._logAcquisitionTiming()

    def _logAcquisitionTiming(self) -> None:
        """Log the exposure + the camera properties that actually govern the
        continuous frame rate. When live FPS is far below ``1000/exposure_ms``,
        the culprit is almost always one of these (slow readout/pixel-clock, or
        Overlap/Frame-Transfer disabled) — not ImSwitch. Read-only; best-effort.
        """
        try:
            exp = float(self._core.getExposure())
        except Exception:
            exp = None
        # Substring needles — Andor prefixes property names (e.g.
        # "Andor sCMOS: ReadoutTime"), so exact matching misses them.
        needles = (
            "readout", "framerate", "frame rate", "frames per second",
            "overlap", "frametransfer", "frame transfer", "pixel clock",
            "pixelclock", "hsspeed", "vsspeed", "amplifier",
            "acquisitionmode", "acquisition mode", "sensor readout",
        )
        found: Dict[str, Any] = {}
        try:
            names = list(self._core.getDevicePropertyNames(self._label))
        except Exception:
            names = []
        for name in names:
            low = name.lower()
            if any(n in low for n in needles):
                try:
                    found[name] = self._core.getProperty(self._label, name)
                except Exception:
                    pass
        expected = f"{1000.0 / exp:.0f}" if exp else "?"
        self._logger.info(
            f"MMCore continuous acquisition started (exposure={exp} ms, "
            f"theoretical max ≈ {expected} fps). Frame-rate-governing camera "
            f"properties: {found or '(none exposed by adapter)'}. If the live "
            f"FPS is far below that, enable Overlap/FrameTransfer and select the "
            f"fastest readout rate in the camera's Expert settings — the "
            f"exposure alone does not set the frame rate."
        )

    def stopAcquisition(self) -> None:
        # Stop the continuous sequence AND clear the circular buffer, so a
        # subsequent snap() reads a fresh exposure taken with the current
        # parameters rather than a stale frame left over from live view.
        #
        # Never wait indefinitely for _grabLock: a single-frame acquisition
        # holds it for the whole exposure, and "stop live view" must not block
        # an API worker for minutes behind a long snap. If the lock is busy we
        # stop the sequence anyway — stopSequenceAcquisition is exactly the
        # call that unblocks the holder.
        locked = self._grabLock.acquire(timeout=1.0)
        try:
            try:
                if self._core.isSequenceRunning():
                    self._core.stopSequenceAcquisition()
                self._running = False
            except Exception:
                self._logger.warning("Failed to stop MMCore acquisition", exc_info=True)
            if not locked:
                # Someone is mid-acquisition; clearing the buffer under them
                # would drop the frame they are waiting for.
                return
            try:
                self._core.clearCircularBuffer()
            except Exception:
                self._logger.debug("Could not clear circular buffer on stop", exc_info=True)
        finally:
            if locked:
                self._grabLock.release()

    def flushBuffer(self) -> None:
        """Alias for :meth:`flushBuffers` — ExperimentController.grabCameraFrame
        calls the singular name."""
        self.flushBuffers()

    def getFrameNumber(self) -> int:
        """Monotonic frame counter, incremented on every snap/buffer read.

        Used by ExperimentController.grabCameraFrame's ``returnFrameNumber``
        path and the free-run frame-sync fallback."""
        return int(self._frameNunber)

    # ------------------------------------------------------------------
    # Software-triggered single-shot acquisition
    # ------------------------------------------------------------------
    # These make the detector compatible with ExperimentController's
    # deterministic-grab machinery (_beginTriggeredAcquisition + snapSync),
    # which previously never activated for MMCore/Andor. In software-trigger
    # mode every exposure is an explicit, blocking snap — ideal for very long
    # exposures (minutes), where the free-run frame-number polling path is
    # both flaky and effectively blocking anyway.
    def setTriggerSource(self, source: str) -> bool:
        """Switch between deterministic single-shot ("software"/"internal") and
        continuous live ("continuous") acquisition.

        Returns True on success so ExperimentController activates the
        triggered fast-path.
        """
        src = str(source).strip().lower()
        with self._grabLock:
            if src in ("software", "internal", "int", "swtrigger"):
                # Take ownership: from now on getLatestFrame() replays the cache
                # and never touches the sensor; only snapSync() exposes.
                self._triggerMode = "software"
                # Stop the live sequence so snaps take fresh, exclusive
                # exposures (MMCore forbids snap() while a sequence runs).
                try:
                    if self._core.isSequenceRunning():
                        self._core.stopSequenceAcquisition()
                    self._running = False
                except Exception:
                    self._logger.debug(
                        "setTriggerSource: stop sequence failed", exc_info=True)
                try:
                    self._core.clearCircularBuffer()
                except Exception:
                    pass
                # Best-effort: put the camera adapter into an internal/software
                # trigger mode if it exposes such a property. Harmless no-op
                # when the property (or an allowed value) is absent.
                self._try_set_trigger_property(("Internal", "Software"))
                self._logger.info(
                    "MMCore: software-trigger mode ON "
                    "(live view now replays cached frames)")
                return True

            # Any other value -> restore continuous live acquisition.
            self._triggerMode = "continuous"
            self._try_set_trigger_property(("Internal",))
            try:
                # startAcquisition() now that _triggerMode is continuous — it
                # resets the FPS meter and logs the acquisition timing.
                self.startAcquisition()
            except Exception:
                self._logger.debug(
                    "setTriggerSource: start sequence failed", exc_info=True)
            self._logger.info("MMCore: continuous live mode restored")
            return True

    def _try_set_trigger_property(self, preferred_values) -> None:
        """Best-effort set of a trigger-mode property on the camera adapter.

        Different MMCore adapters name this differently ("TriggerMode",
        "Trigger", "Trigger Source", ...); many Andor configs have no such
        property at all (internal trigger is the only mode). We try a few
        common names and silently give up if none apply."""
        candidate_props = ("TriggerMode", "Trigger", "Trigger Source", "TriggerSource")
        for prop in candidate_props:
            try:
                if not self._core.hasProperty(self._label, prop):
                    continue
                allowed = []
                try:
                    allowed = [str(a) for a in
                               self._core.getAllowedPropertyValues(self._label, prop)]
                except Exception:
                    allowed = []
                target = None
                for want in preferred_values:
                    if not allowed:
                        target = want
                        break
                    for a in allowed:
                        if a.strip().lower() == want.strip().lower():
                            target = a
                            break
                    if target is not None:
                        break
                if target is not None:
                    self._core.setProperty(self._label, prop, str(target))
                return
            except Exception:
                self._logger.debug(f"Could not set trigger property {prop}",
                                   exc_info=True)

    def snapSync(self, timeout: float = None) -> np.ndarray:
        """Fire one exposure and block until the resulting frame is available.

        The single deterministic-grab primitive used by the ExperimentController
        for every acquisition. It delegates to :meth:`acquireSingleFrame`, so:

        1. ``_grabLock`` keeps any live-view poll off the sensor for the whole
           exposure (that overlap is what raised Andor DRV_ACQUIRING/20072).
        2. Any running sequence is stopped and stale buffered frames cleared
           before the exposure is triggered.
        3. The wait is a polling loop, not a blocking ``core.snap()``. Only
           this thread blocks; MMCore's device lock stays free, so parameter
           reads, temperature polls and the REST API keep working through a
           multi-minute exposure instead of freezing the backend.
        4. A stuck camera is not waited on forever — the timeout defaults to
           ``exposure x 4 + 5 s``, generous enough that legitimate long
           exposures never trip it.
        5. The frame is cached as the latest so the live view can display it.

        On timeout or failure the last cached frame is returned, matching the
        previous behaviour (callers treat a repeated frame as "no new data").
        """
        if timeout is None or timeout <= 0:
            try:
                exp_s = float(self._core.getExposure()) / 1000.0
            except Exception:
                exp_s = 0.1
            timeout = max(5.0, exp_s * 4.0 + 5.0)

        image, status = self.acquireSingleFrame(timeout=timeout)
        if status == "done" and image is not None:
            return image

        self._logger.error(
            f"snapSync: no frame ({status}) — returning last cached frame.")
        return self._return_stored(False)

    # ------------------------------------------------------------------
    # Exclusive access + cancellable single-frame acquisition
    # ------------------------------------------------------------------
    # Long acquisitions (a snap job, an experiment step) must own the sensor
    # for their whole duration. These primitives are what makes that safe: the
    # owner is announced to getLatestFrame() (which then replays its cache
    # instead of touching MMCore), the exposure is driven through a polling
    # loop rather than a blocking snap (so the interpreter and MMCore's device
    # lock stay free and the backend keeps answering requests), and the whole
    # thing can be aborted from another thread.
    @property
    def exclusiveOwner(self) -> Optional[str]:
        """Name of the caller currently owning the sensor, or ``None``."""
        return self._exclusiveOwner

    def acquireExclusive(self, owner: str, timeout: float = 0.0) -> bool:
        """Claim the camera for ``owner``. Returns False if already claimed.

        Stops a running live sequence and remembers that it did, so
        ``releaseExclusive`` can put live view back exactly as it found it.
        ``timeout`` > 0 waits that many seconds for a competing claim to end.
        """
        if timeout and timeout > 0:
            acquired = self._exclusiveLock.acquire(timeout=timeout)
        else:
            acquired = self._exclusiveLock.acquire(blocking=False)
        if not acquired:
            return False

        self._exclusiveOwner = str(owner)
        self._exclusiveSince = time.time()
        try:
            self._exclusiveRestoreLive = bool(self._core.isSequenceRunning())
        except Exception:
            self._exclusiveRestoreLive = False
        if self._exclusiveRestoreLive:
            try:
                self._core.stopSequenceAcquisition()
                self._running = False
            except Exception:
                self._logger.debug(
                    "acquireExclusive: could not stop live sequence",
                    exc_info=True)
        self._logger.debug(f"Camera claimed by '{owner}'")
        return True

    def releaseExclusive(self, owner: Optional[str] = None,
                         restoreLive: Optional[bool] = None) -> None:
        """Release a claim taken with :meth:`acquireExclusive`.

        ``restoreLive`` defaults to whatever the live state was when the claim
        was taken; pass False to leave the camera idle.
        """
        if self._exclusiveOwner is None:
            return
        if owner is not None and self._exclusiveOwner != str(owner):
            self._logger.warning(
                f"releaseExclusive('{owner}') ignored -- camera is owned by "
                f"'{self._exclusiveOwner}'")
            return

        shouldRestore = (self._exclusiveRestoreLive if restoreLive is None
                         else bool(restoreLive))
        self._exclusiveOwner = None
        self._exclusiveRestoreLive = False
        try:
            self._exclusiveLock.release()
        except RuntimeError:
            pass

        # Only bring live view back if the camera isn't owned by an experiment
        # through the software-trigger path.
        if shouldRestore and self._triggerMode != "software":
            try:
                self.startAcquisition()
            except Exception:
                self._logger.debug(
                    "releaseExclusive: could not restart live acquisition",
                    exc_info=True)

    @contextmanager
    def exclusiveAccess(self, owner: str, timeout: float = 0.0,
                        restoreLive: Optional[bool] = None):
        """Context manager form of :meth:`acquireExclusive`.

        Raises ``RuntimeError`` if the camera is already claimed, so a caller
        never silently runs an acquisition on a busy sensor.
        """
        if not self.acquireExclusive(owner, timeout=timeout):
            raise RuntimeError(
                f"Camera '{self.name}' is busy -- currently owned by "
                f"'{self._exclusiveOwner}'")
        try:
            yield self
        finally:
            self.releaseExclusive(owner, restoreLive=restoreLive)

    def abortAcquisition(self) -> None:
        """Stop whatever the camera is doing *right now*.

        Ends an integration in the driver instead of waiting it out, which is
        what makes a long exposure cancellable. Deliberately does NOT take
        ``_grabLock``: the whole point is to interrupt the thread that holds it.
        """
        try:
            if self._core.isSequenceRunning():
                self._core.stopSequenceAcquisition()
        except Exception:
            self._logger.debug("abortAcquisition: stop failed", exc_info=True)
        self._running = False

    def acquireSingleFrame(self, cancelEvent: Optional[threading.Event] = None,
                           timeout: Optional[float] = None,
                           pollInterval: float = 0.05):
        """Acquire exactly one frame, cancellably, without freezing the process.

        Returns ``(image, status)`` where ``status`` is one of ``"done"``,
        ``"cancelled"``, ``"timeout"`` or ``"error"``; ``image`` is ``None``
        unless the status is ``"done"``.

        Uses ``startSequenceAcquisition(1)`` + polling rather than
        ``core.snap()``. ``snapImage()`` sits inside MMCore's device lock for
        the entire integration, so a multi-minute Andor exposure would block
        every other MMCore call in the process -- temperature polls, parameter
        reads, the live view, and the very status endpoint the UI polls for
        progress. The sequence variant returns immediately, so we can sleep
        between polls, keep the backend responsive, honour ``cancelEvent``, and
        give up on a camera that never delivers instead of hanging forever.

        The caller is expected to hold an exclusive claim
        (:meth:`acquireExclusive`); otherwise the live-view poll can drain the
        frame out of the circular buffer before we read it.
        """
        with self._grabLock:
            try:
                exposure_s = float(self._core.getExposure()) / 1000.0
            except Exception:
                exposure_s = 0.1
            if timeout is None or timeout <= 0:
                # Generous: readout and adapter overhead scale with exposure,
                # but a stuck camera must not be waited on forever.
                timeout = max(15.0, exposure_s * 2.0 + 15.0)

            # Start from a clean slate: no sequence running, no stale frames.
            try:
                if self._core.isSequenceRunning():
                    self._core.stopSequenceAcquisition()
                self._running = False
            except Exception:
                self._logger.debug(
                    "acquireSingleFrame: could not stop running sequence",
                    exc_info=True)
            try:
                self._core.clearCircularBuffer()
            except Exception:
                pass

            try:
                self._core.startSequenceAcquisition(1, 0, True)
            except Exception:
                self._logger.error(
                    "acquireSingleFrame: could not start acquisition",
                    exc_info=True)
                return None, "error"

            deadline = time.time() + timeout
            status = "done"
            sawRunning = False
            try:
                while True:
                    try:
                        remaining = int(self._core.getRemainingImageCount())
                    except Exception:
                        remaining = 0
                    if remaining > 0:
                        break
                    if cancelEvent is not None and cancelEvent.is_set():
                        status = "cancelled"
                        break
                    if time.time() > deadline:
                        status = "timeout"
                        self._logger.error(
                            f"acquireSingleFrame: no frame after {timeout:.1f}s "
                            f"(exposure {exposure_s:.3f}s) -- giving up")
                        break
                    # The driver dropping out of "running" with an empty buffer
                    # means the adapter aborted; don't wait out the timeout.
                    # Only trusted once we have actually seen it run, since
                    # isSequenceRunning() can lag the start call.
                    try:
                        running = bool(self._core.isSequenceRunning())
                    except Exception:
                        running = True
                    if running:
                        sawRunning = True
                    elif sawRunning:
                        status = "error"
                        self._logger.error(
                            "acquireSingleFrame: acquisition stopped without "
                            "delivering a frame")
                        break
                    time.sleep(pollInterval)
            finally:
                try:
                    if self._core.isSequenceRunning():
                        self._core.stopSequenceAcquisition()
                except Exception:
                    pass

            if status != "done":
                return None, status

            try:
                image = np.asarray(self._core.popNextImage())
            except Exception:
                self._logger.error(
                    "acquireSingleFrame: failed to read the delivered frame",
                    exc_info=True)
                return None, "error"

            self._store_latest(image)
            return image, "done"

    def crop(self, hpos: int, vpos: int, hsize: int, vsize: int) -> None:
        try:
            self._core.setROI(self._label, int(hpos), int(vpos), int(hsize), int(vsize))
            self._refresh_shape()
        except Exception:
            self._logger.error("setROI failed", exc_info=True)

    def _refresh_shape(self) -> None:
        """Re-read the current image geometry from MMCore (reflects ROI and
        binning) and update ``self._shape``."""
        try:
            self._shape = (int(self._core.getImageWidth()), int(self._core.getImageHeight()))
        except Exception:
            self._logger.debug("Could not refresh image shape", exc_info=True)

    @property
    def pixelSizeUm(self) -> List[float]:
        try:
            ps = float(self._core.getPixelSizeUm())
        except Exception:
            ps = 0.0
        if ps <= 0:
            # Fall back to the manager-level override (settable via
            # setPixelSizeUm) before defaulting to 1 µm.
            try:
                ps = float(self.pixelSize[1])
            except Exception:
                ps = 1.0
        if ps <= 0:
            ps = 1.0
        return [1.0, ps, ps]

    # ------------------------------------------------------------------
    # Parameter / binning / lifecycle
    # ------------------------------------------------------------------
    def _resolve_param_key(self, name):
        """Map a logical/loosely-cased parameter name onto the actual MMCore
        property key stored in ``self.parameters``.

        Callers like ``SettingsController.setDetectorGain`` pass lowercase
        logical names ("gain"), but MMCore adapters use device-specific,
        case-sensitive property names (Andor: "Gain" / "Pre-Amp-Gain" / ...).
        Returns the resolved key, or ``None`` if nothing matches.
        """
        if not isinstance(name, str):
            return None
        if name in self.parameters:
            return name
        lname = name.strip().lower()
        # Case-insensitive exact match (handles "gain" -> "Gain").
        for key in self.parameters:
            if key.strip().lower() == lname:
                return key
        # Gain family: try the common Andor/EM aliases, then any gain-ish key.
        if "gain" in lname:
            candidates = (
                "gain", "pre-amp-gain", "pre-amp gain", "preampgain",
                "emgain", "em gain", "gain multiplier", "sensitivity",
            )
            for cand in candidates:
                for key in self.parameters:
                    if key.strip().lower() == cand:
                        return key
            for key in self.parameters:
                if "gain" in key.strip().lower():
                    return key
        return None

    def setParameter(self, name, value):
        # Accept any case variant of Exposure/exposure/exposureTime from callers.
        # The base class lowercases names containing "posure" to "exposure",
        # which would miss our "Exposure" key — handle it here and skip super().
        if isinstance(name, str) and "posure" in name.lower():
            try:
                self._core.setExposure(float(value))
            except Exception:
                self._logger.error(f"Failed to set exposure to {value}", exc_info=True)
            try:
                actual = float(self._core.getExposure())
            except Exception:
                actual = float(value)
            exposure_key = "Exposure" if "Exposure" in self.parameters else (
                "exposure" if "exposure" in self.parameters else None
            )
            if exposure_key is not None:
                self.parameters[exposure_key].value = actual
            # NB: no need to mirror into self._camera.exposure_time — the proxy
            # reads it live from the core (in microseconds).
            return self.parameters

        # Resolve loosely-cased/logical names (e.g. "gain") to the real MMCore
        # property key before the exact-match lookup below. Without this, a
        # SettingsController.setDetectorGain(None, gain) call would fall through
        # to super() and raise "Non-existent parameter gain" on the Andor.
        if isinstance(name, str) and name not in self.parameters:
            resolved = self._resolve_param_key(name)
            if resolved is not None:
                name = resolved
            else:
                self._logger.warning(
                    f"MMCore detector has no property matching '{name}'; "
                    f"ignoring setParameter."
                )
                return self.parameters

        if name in self.parameters:
            try:
                self._core.setProperty(self._label, name, str(value))
            except Exception:
                self._logger.error(
                    f"Failed to set MMCore property {name}={value}", exc_info=True
                )
            # Echo back what the device actually accepted — MMCore may clamp or
            # snap to allowed values, and the UI needs the truth.
            try:
                actual = self._core.getProperty(self._label, name)
            except Exception:
                actual = value
            param = self.parameters[name]
            if isinstance(param, DetectorNumberParameter):
                try:
                    param.value = float(actual)
                except (TypeError, ValueError):
                    param.value = actual
            else:
                param.value = str(actual)
            # Changing binning (or ROI-ish properties) alters the frame
            # geometry — keep _shape in sync so downstream consumers and the
            # live loop allocate the right buffer size.
            if name.strip().lower() == "binning":
                self._refresh_shape()
                try:
                    super().setBinning(int(float(actual)))
                except Exception:
                    pass
            return self.parameters

        return super().setParameter(name, value)


    def getParameter(self, name):
        """Return the current value of a parameter, resolving exposure through
        the live core so callers (e.g. ExperimentController) get milliseconds
        rather than a possibly-stale cached value.

        Raises ``AttributeError`` for unknown parameters, matching the other
        detector managers.
        """
        if isinstance(name, str) and "posure" in name.lower():
            try:
                return float(self._core.getExposure())
            except Exception:
                pass
        if name in self.parameters:
            return self.parameters[name].value
        # Resolve loosely-cased/logical names (e.g. "gain" -> "Gain") so
        # callers reading gain back don't hit a spurious AttributeError.
        resolved = self._resolve_param_key(name)
        if resolved is not None:
            return self.parameters[resolved].value
        raise AttributeError(f'Non-existent parameter "{name}" specified')

    def setBinning(self, binning: int) -> None:
        try:
            self._core.setProperty(self._label, "Binning", str(binning))
        except Exception:
            # Not all cameras expose a Binning property – fall back silently.
            pass
        super().setBinning(binning)
        # Reflect the new geometry and keep the "Binning" list parameter in sync.
        self._refresh_shape()
        if "Binning" in self.parameters:
            try:
                self.parameters["Binning"].value = str(binning)
            except Exception:
                pass

    def finalize(self) -> None:
        self.stopAcquisition()

    def setPixelSizeUm(self, pixelSizeUm) -> None:
        # MMCore owns the physical pixel size via its pixel-size config, but we
        # allow an ImSwitch-side override for setups without a configured
        # pixel-size group. Accept either a scalar or a [Z, Y, X] list.
        if isinstance(pixelSizeUm, (list, tuple)):
            ps = float(pixelSizeUm[-1])
        else:
            ps = float(pixelSizeUm)
        self.pixelSize = [1.0, ps, ps]

    # ------------------------------------------------------------------
    # Temperature / diagnostics / metadata
    # ------------------------------------------------------------------
    def _find_property(self, candidates) -> Optional[str]:
        """Return the actual property name whose lowercase form matches one of
        ``candidates`` (already lowercase), or ``None``."""
        try:
            names = list(self._core.getDevicePropertyNames(self._label))
        except Exception:
            return None
        lut = {n.lower(): n for n in names}
        for cand in candidates:
            if cand in lut:
                return lut[cand]
        return None

    def getTemperatureC(self) -> Optional[float]:
        """Current sensor/CCD temperature in °C, or ``None`` if unavailable."""
        prop = self._find_property(_TEMPERATURE_PROPERTY_NAMES)
        if prop is None:
            return None
        try:
            return float(self._core.getProperty(self._label, prop))
        except Exception:
            return None

    def getTemperatureSetpointC(self) -> Optional[float]:
        prop = self._find_property(_TEMPERATURE_SETPOINT_NAMES)
        if prop is None:
            return None
        try:
            return float(self._core.getProperty(self._label, prop))
        except Exception:
            return None

    def getStreamDiagnostics(self) -> Dict[str, Any]:
        """Lightweight buffer/acquisition stats, mirroring the shape returned by
        the hardware interfaces so LiveViewController can render it uniformly."""
        diag: Dict[str, Any] = {
            "backend": "mmcore",
            "label": self._label,
            "running": bool(self._running),
        }
        try:
            diag["sequenceRunning"] = bool(self._core.isSequenceRunning())
        except Exception:
            diag["sequenceRunning"] = None
        try:
            diag["remainingImages"] = int(self._core.getRemainingImageCount())
        except Exception:
            diag["remainingImages"] = None
        try:
            diag["bufferTotalCapacity"] = int(self._core.getBufferTotalCapacity())
            diag["bufferFreeCapacity"] = int(self._core.getBufferFreeCapacity())
        except Exception:
            pass
        try:
            diag["exposureMs"] = float(self._core.getExposure())
        except Exception:
            diag["exposureMs"] = None
        # True camera delivery rate (frames the sensor actually pushes into the
        # circular buffer per second). If this is far below 1000/exposureMs the
        # bottleneck is the camera/readout config, NOT the ImSwitch pipeline.
        diag["cameraFps"] = round(float(self._camStats.get("fps", 0.0)), 2)
        diag["framesDelivered"] = int(self._camStats.get("delivered", 0))
        temp = self.getTemperatureC()
        if temp is not None:
            diag["temperatureC"] = temp
        return diag

    def getMetadataSnapshot(self) -> Dict[str, Any]:
        """Return a flat dict of *all* current MMCore property values for the
        camera device, suitable for embedding in acquisition metadata."""
        meta: Dict[str, Any] = {}
        try:
            for prop in self._core.getDevicePropertyNames(self._label):
                try:
                    meta[prop] = self._core.getProperty(self._label, prop)
                except Exception:
                    continue
        except Exception:
            self._logger.debug("Could not build MMCore metadata snapshot", exc_info=True)
        try:
            meta["Exposure_ms"] = float(self._core.getExposure())
        except Exception:
            pass
        meta["_pixelSizeUm"] = self.pixelSizeUm
        meta["_binning"] = self.binning
        meta["_model"] = self.model
        return meta

    # ------------------------------------------------------------------
    # Persistence helpers (driven by MMCoreController)
    # ------------------------------------------------------------------
    def applySavedProperties(self, saved: Dict[str, Any]) -> Dict[str, Any]:
        """Apply a dict of {propertyName: value} to the device. Returns a dict
        of {propertyName: appliedValue} for the ones that were applied."""
        applied: Dict[str, Any] = {}
        if not saved:
            return applied
        for name, value in saved.items():
            try:
                self.setParameter(name, value)
                applied[name] = value
            except Exception:
                self._logger.warning(
                    f"Could not apply saved property {name}={value}", exc_info=True
                )
        return applied

    def resetToFactoryDefaults(self) -> Dict[str, Any]:
        """Restore every editable property to the value captured at init."""
        applied: Dict[str, Any] = {}
        for name, value in self._factoryDefaults.items():
            try:
                self.setParameter(name, value)
                applied[name] = value
            except Exception:
                self._logger.warning(
                    f"Could not reset property {name} to default {value}", exc_info=True
                )
        return applied

    def getFactoryDefaults(self) -> Dict[str, str]:
        return dict(self._factoryDefaults)

    # ------------------------------------------------------------------
    # Status (enriched with limits / advanced flag / temperature)
    # ------------------------------------------------------------------
    def getCameraStatus(self) -> Dict[str, Any]:
        status = super().getCameraStatus()
        # Enrich each parameter with min/max limits and the base/expert flag.
        params = status.get("parameters", {})
        for name, info in params.items():
            meta = self._paramMeta.get(name)
            if not meta:
                info.setdefault("advanced", False)
                continue
            info["advanced"] = bool(meta.get("advanced", True))
            if meta.get("min") is not None:
                info["min"] = meta["min"]
            if meta.get("max") is not None:
                info["max"] = meta["max"]
        # Top-level temperature + geometry extras.
        status["temperatureC"] = self.getTemperatureC()
        status["temperatureSetpointC"] = self.getTemperatureSetpointC()
        status["exposureMs"] = params.get("Exposure", {}).get("value")
        # Who currently owns the sensor, so a client can explain "camera busy"
        # instead of just failing.
        status["exclusiveOwner"] = self._exclusiveOwner
        status["busy"] = self._exclusiveOwner is not None
        return status



# Copyright (C) 2020-2026 ImSwitch developers
# This file is part of ImSwitch and licensed under GPL-3.0-or-later.
