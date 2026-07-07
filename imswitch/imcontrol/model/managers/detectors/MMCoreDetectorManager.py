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
    def getLatestFrame(self, returnFrameNumber=False) -> np.ndarray:
        # Two cases:
        #   1) A continuous sequence acquisition is running (live mode). We
        #      MUST NOT call snap() — MMCore raises "This operation can not
        #      be executed while sequence acquisition is running". Read from
        #      the circular buffer instead.
        #   2) No sequence is running (idle, or just after stopAcquisition).
        #      Take a one-shot snap.
        try:
            sequence_running = bool(self._core.isSequenceRunning())
        except Exception:
            sequence_running = False

        if sequence_running:
            try:
                if self._core.getRemainingImageCount() > 0:
                    frame = np.asarray(self._core.getLastImage())
                    self._frameNunber += 1
                    if returnFrameNumber:
                        return frame, self._frameNunber
                    return frame
            except Exception:
                self._logger.debug(
                    "Failed to read latest image from sequence buffer",
                    exc_info=True,
                )
            # No frame in the buffer yet — return a placeholder rather than
            # blocking the live-view loop.
            if returnFrameNumber:
                return np.zeros(self._shape, dtype=np.uint16), -1
            return np.zeros(self._shape, dtype=np.uint16)

        # No sequence: try the most recent buffered frame, then fall back to
        # a one-shot snap.
        try:
            if self._core.getRemainingImageCount() > 0:
                return np.asarray(self._core.getLastImage())
        except Exception:
            pass
        try:
            # CMMCorePlus.snap() returns the numpy array directly (it does
            # snapImage() + getImage() under the hood). Calling getImage()
            # again on top fails on the Andor adapter with
            # "Camera image buffer read failed".
            image = np.asarray(self._core.snap())
            self._frameNunber += 1
            if returnFrameNumber:
                return image, self._frameNunber
            return image
        except Exception:
            self._logger.error("Failed to snap a frame from MMCore", exc_info=True)
            if returnFrameNumber:
                return np.zeros(self._shape, dtype=np.uint16), -1
            return np.zeros(self._shape, dtype=np.uint16)

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
        if not self._core.isSequenceRunning():
            self._core.startContinuousSequenceAcquisition(0)
            self._running = True

    def stopAcquisition(self) -> None:
        # Stop the continuous sequence AND clear the circular buffer, so a
        # subsequent snap() reads a fresh exposure taken with the current
        # parameters rather than a stale frame left over from live view.
        try:
            if self._core.isSequenceRunning():
                self._core.stopSequenceAcquisition()
            self._running = False
        except Exception:
            self._logger.warning("Failed to stop MMCore acquisition", exc_info=True)
        try:
            self._core.clearCircularBuffer()
        except Exception:
            self._logger.debug("Could not clear circular buffer on stop", exc_info=True)

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
        return status



# Copyright (C) 2020-2026 ImSwitch developers
# This file is part of ImSwitch and licensed under GPL-3.0-or-later.
