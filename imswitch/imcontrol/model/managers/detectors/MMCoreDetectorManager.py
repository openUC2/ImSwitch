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

from typing import Dict, List

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


def _is_internal_property(prop: str) -> bool:
    if any(prop.startswith(p) for p in _SKIP_PROPERTY_PREFIXES):
        return True
    if any(s in prop for s in _SKIP_PROPERTY_SUBSTRINGS):
        return True
    return False


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

        super().__init__(
            detectorInfo,
            name,
            fullShape=full_shape,
            supportedBinnings=supported_binnings,
            model=self._label or "MMCore Camera",
            parameters=parameters,
            croppable=True,
        )

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

    def _build_parameters(self) -> Dict[str, DetectorParameter]:
        parameters: Dict[str, DetectorParameter] = {}
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

            if allowed:
                parameters[prop] = DetectorListParameter(
                    group="MMCore",
                    value=str(value),
                    editable=not read_only,
                    options=[str(a) for a in allowed],
                )
                continue

            try:
                num_value = float(value)
            except (TypeError, ValueError):
                # Skip free-form strings – they don't map onto our UI widgets.
                continue
            parameters[prop] = DetectorNumberParameter(
                group="MMCore",
                value=num_value,
                editable=not read_only,
                valueUnits="",
            )
        return parameters

    # ------------------------------------------------------------------
    # Frame access
    # ------------------------------------------------------------------
    def getLatestFrame(self) -> np.ndarray:
        try:
            if self._core.getRemainingImageCount() > 0:
                return self._core.getLastImage()
        except Exception:
            pass
        try:
            self._core.snap()
            return self._core.getImage()
        except Exception:
            self._logger.error("Failed to snap a frame from MMCore", exc_info=True)
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

    def stopAcquisition(self) -> None:
        try:
            if self._core.isSequenceRunning():
                self._core.stopSequenceAcquisition()
        except Exception:
            self._logger.warning("Failed to stop MMCore acquisition", exc_info=True)

    def crop(self, hpos: int, vpos: int, hsize: int, vsize: int) -> None:
        try:
            self._core.setROI(self._label, int(hpos), int(vpos), int(hsize), int(vsize))
            self._shape = (int(hsize), int(vsize))
        except Exception:
            self._logger.error("setROI failed", exc_info=True)

    @property
    def pixelSizeUm(self) -> List[float]:
        try:
            ps = float(self._core.getPixelSizeUm())
        except Exception:
            ps = 0.0
        if ps <= 0:
            ps = 1.0
        return [1.0, ps, ps]

    # ------------------------------------------------------------------
    # Parameter / binning / lifecycle
    # ------------------------------------------------------------------
    def setParameter(self, name, value):
        if name == "Exposure":
            try:
                self._core.setExposure(float(value))
            except Exception:
                self._logger.error(f"Failed to set exposure to {value}", exc_info=True)
        elif name in self.parameters:
            try:
                self._core.setProperty(self._label, name, str(value))
            except Exception:
                self._logger.error(
                    f"Failed to set MMCore property {name}={value}", exc_info=True
                )
        return super().setParameter(name, value)

    def setBinning(self, binning: int) -> None:
        try:
            self._core.setProperty(self._label, "Binning", str(binning))
        except Exception:
            # Not all cameras expose a Binning property – fall back silently.
            pass
        super().setBinning(binning)

    def finalize(self) -> None:
        self.stopAcquisition()


# Copyright (C) 2020-2026 ImSwitch developers
# This file is part of ImSwitch and licensed under GPL-3.0-or-later.
