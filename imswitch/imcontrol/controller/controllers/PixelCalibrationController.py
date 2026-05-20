"""
PixelCalibrationController
==========================

Single source of truth for per-detector pixel size and image flip in ImSwitch.

Design
------
* Calibrations are keyed by **(detector name, objective id)** in the form
  ``"<detectorName>::<objectiveId>"``. Legacy entries that only store the
  detector name are still recognised and treated as objective ``"default"``.
* On startup, and whenever the active objective changes, the controller reads
  ``setupInfo.PixelCalibration.affineCalibrations`` and pushes the derived
  ``flipX``/``flipY`` and average pixel size into every matching
  ``DetectorManager`` via ``setFlipImage()`` / ``setPixelSizeUm()``.
* New calibrations run via ``calibrateStageAffine`` are stored in a *pending*
  state. The frontend reviews the result and may edit pixel size / flip / matrix
  before calling ``applyPendingCalibration`` to persist and apply it. A
  ``discardPendingCalibration`` endpoint discards without saving.
* The controller is **always loaded** — no ``PixelCalibration`` config block is
  required for ImSwitch to start.
* AprilTag / overview-camera helpers have been removed; this controller is now
  a focused addon for the DetectorManager.
"""
import math
import threading
import time

import numpy as np
import NanoImagingPack as nip

from imswitch.imcommon.framework import Signal
from imswitch.imcommon.model import APIExport, initLogger
from imswitch.imcontrol.controller.controllers.camera_stage_mapping.affine_stage_calibration import compute_affine_matrix, measure_pixel_shift

from ..basecontrollers import LiveUpdatedController


def _convert_to_native(obj):
    """Recursively convert numpy types to plain Python so values are JSON-safe.

    Also replaces NaN / +-Inf with ``None`` so that FastAPI / json.dumps does not
    raise ``ValueError: Out of range float values are not JSON compliant``.
    """
    if isinstance(obj, np.ndarray):
        return _convert_to_native(obj.tolist())
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _convert_to_native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_convert_to_native(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_convert_to_native(v) for v in obj)
    return obj


class PixelCalibrationController(LiveUpdatedController):
    """Controller exposing per-detector affine calibration as a DetectorManager addon."""

    sigImageReceived = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        # Default acquisition detector (used by PixelCalibrationClass for image grabs).
        all_detector_names = self._master.detectorsManager.getAllDeviceNames()
        if not all_detector_names:
            raise RuntimeError("PixelCalibrationController requires at least one detector")
        self.detector = self._master.detectorsManager[all_detector_names[0]]

        # In-memory calibration store, keyed by ``"<detectorName>::<objectiveId>"``.
        self.affineCalibrations: dict = {}
        # Pending (not yet applied) calibrations awaiting frontend approval.
        # Also keyed by the composite key so multiple objectives can be in flight.
        self._pendingCalibration: dict = {}

        # Currently selected objective (string key).
        self.currentObjective = self._getCurrentObjectiveId()

        self._loadAffineCalibrations()

        # React to objective changes -> re-apply matching calibration per detector.
        try:
            obj_ctrl = self._master._controllersRegistry.get("Objective", None)
            if obj_ctrl is not None and hasattr(obj_ctrl, "sigObjectiveChanged"):
                obj_ctrl.sigObjectiveChanged.connect(self._onObjectiveChanged)
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.debug(f"Could not subscribe to objective changes: {exc}")

    # ------------------------------------------------------------------ #
    # Startup helpers                                                     #
    # ------------------------------------------------------------------ #

    def _getCurrentObjectiveId(self) -> str:
        """Best-effort lookup of the currently selected objective slot as a string."""
        try:
            obj_ctrl = self._master._controllersRegistry.get("Objective", None)
            if obj_ctrl is not None:
                slot = getattr(obj_ctrl, "_currentObjective", None)
                if slot is not None:
                    return str(int(slot))
        except Exception as exc:  # pragma: no cover - defensive
            self._logger.debug(f"Could not resolve current objective: {exc}")
        return "default"

    def _resolveObjectiveId(self, objectiveId) -> str:
        """Normalise an objective id; ``None``/``"current"`` -> currently active slot."""
        if objectiveId is None or objectiveId == "" or objectiveId == "current":
            return self._getCurrentObjectiveId()
        return str(objectiveId)

    def _makeKey(self, detectorName: str, objectiveId: str) -> str:
        return f"{detectorName}::{objectiveId}"

    def _splitKey(self, key: str):
        """Split a composite key into ``(detectorName, objectiveId)``; legacy keys -> ``"default"``."""
        if "::" in key:
            det, obj = key.split("::", 1)
            return det, obj
        return key, "default"

    def _onObjectiveChanged(self, _status: dict = None):
        """Re-apply per-detector calibration matching the newly selected objective."""
        new_obj = self._getCurrentObjectiveId()
        if new_obj == self.currentObjective:
            return
        self.currentObjective = new_obj
        self._logger.info(f"Objective changed to '{new_obj}' - re-applying calibrations")
        for detector_name in self._master.detectorsManager.getAllDeviceNames():
            key = self._makeKey(detector_name, new_obj)
            calib = self.affineCalibrations.get(key)
            if calib is None:
                # Try legacy detector-only key as a fallback.
                calib = self.affineCalibrations.get(detector_name)
            if calib is not None:
                self._applyCalibrationToDetector(detector_name, calib)

    def _loadAffineCalibrations(self):
        """Load persisted calibrations and push them into matching detectors."""
        pixel_calibration = getattr(self._setupInfo, "PixelCalibration", None)
        if pixel_calibration is None or not getattr(pixel_calibration, "affineCalibrations", None):
            self._logger.info("No PixelCalibration data on disk - all detectors start uncalibrated")
            self.affineCalibrations = {}
            return

        self.affineCalibrations = dict(pixel_calibration.affineCalibrations)
        self._logger.info(
            f"Loaded {len(self.affineCalibrations)} affine calibration(s) from setup config"
        )

        known_detectors = set(self._master.detectorsManager.getAllDeviceNames())
        current_obj = self.currentObjective
        # Apply only the calibration that matches the *current* objective for each detector.
        for detector_name in known_detectors:
            key = self._makeKey(detector_name, current_obj)
            calib = self.affineCalibrations.get(key)
            if calib is None:
                # Legacy detector-only entry (no objective suffix).
                calib = self.affineCalibrations.get(detector_name)
            if calib is not None:
                self._applyCalibrationToDetector(detector_name, calib)
            else:
                self._logger.info(
                    f"No calibration for detector '{detector_name}' @ objective '{current_obj}' - defaults kept"
                )

    def _applyCalibrationToDetector(self, detector_name: str, calib_data: dict):
        """Push pixel size + flip from a calibration entry to the named detector."""
        try:
            detector = self._master.detectorsManager[detector_name]
        except Exception:
            self._logger.warning(
                f"Detector '{detector_name}' not found - skipping calibration apply"
            )
            return

        metrics = calib_data.get("metrics", {}) or {}
        scale_x = float(metrics.get("scale_x_um_per_pixel", 1.0))
        scale_y = float(metrics.get("scale_y_um_per_pixel", 1.0))
        avg_pixel_size = (abs(scale_x) + abs(scale_y)) / 2.0
        flip_x = scale_x > 0
        flip_y = scale_y > 0

        if hasattr(detector, "setFlipImage"):
            try:
                detector.setFlipImage(flip_y, flip_x)
                self._logger.info(
                    f"[{detector_name}] flip set: Y={flip_y}, X={flip_x}"
                )
            except Exception as exc:
                self._logger.warning(f"[{detector_name}] setFlipImage failed: {exc}")

        if hasattr(detector, "setPixelSizeUm"):
            try:
                detector.setPixelSizeUm(avg_pixel_size)
                self._logger.info(
                    f"[{detector_name}] pixel size set: {avg_pixel_size:.4f} µm/px"
                )
            except Exception as exc:
                self._logger.warning(f"[{detector_name}] setPixelSizeUm failed: {exc}")

    # ------------------------------------------------------------------ #
    # Lookup helpers (used by other controllers)                          #
    # ------------------------------------------------------------------ #

    def _resolveCalibration(self, detectorName: str, objectiveId: str):
        """Return the stored calibration for a (detector, objective) pair, or ``None``."""
        key = self._makeKey(detectorName, objectiveId)
        calib = self.affineCalibrations.get(key)
        if calib is None:
            # Legacy detector-only key.
            calib = self.affineCalibrations.get(detectorName)
        return calib

    def getAffineMatrix(self, detectorName: str = None, objectiveId: str = None) -> np.ndarray:
        """Return the persisted affine matrix for a (detector, objective) pair."""
        if detectorName is None:
            detectorName = self._defaultDetectorName()
        objectiveId = self._resolveObjectiveId(objectiveId)
        calib = self._resolveCalibration(detectorName, objectiveId)
        if calib and "affine_matrix" in calib:
            return np.array(calib["affine_matrix"], dtype=float)
        return self._setupInfo.getAffineMatrix(self._makeKey(detectorName, objectiveId))

    def getPixelSize(self, detectorName: str = None, objectiveId: str = None) -> tuple:
        """Return ``(scale_x, scale_y)`` in µm/px (signed)."""
        if detectorName is None:
            detectorName = self._defaultDetectorName()
        objectiveId = self._resolveObjectiveId(objectiveId)
        calib = self._resolveCalibration(detectorName, objectiveId) or {}
        metrics = calib.get("metrics", {}) if calib else {}
        return (
            float(metrics.get("scale_x_um_per_pixel", 1.0)),
            float(metrics.get("scale_y_um_per_pixel", 1.0)),
        )

    @APIExport()
    def getPixelSizeUm(self, detectorName: str = None, objectiveId: str = None):
        """Public lookup used by other controllers (e.g. ObjectiveController).

        Returns the *unsigned* average pixel size for the (detector, objective)
        pair, or ``None`` if no calibration is stored.
        """
        if detectorName is None:
            detectorName = self._defaultDetectorName()
        objectiveId = self._resolveObjectiveId(objectiveId)
        calib = self._resolveCalibration(detectorName, objectiveId)
        if calib is None:
            return {"success": False, "pixelSizeUm": None}
        sx, sy = self.getPixelSize(detectorName, objectiveId)
        return {
            "success": True,
            "detectorName": detectorName,
            "objectiveId": objectiveId,
            "pixelSizeUm": (abs(sx) + abs(sy)) / 2.0,
            "scaleXUmPerPixel": sx,
            "scaleYUmPerPixel": sy,
        }

    def _defaultDetectorName(self) -> str:
        names = self._master.detectorsManager.getAllDeviceNames()
        return names[0] if names else "default"

    # ------------------------------------------------------------------ #
    # API: calibration lifecycle                                          #
    # ------------------------------------------------------------------ #

    @APIExport(runOnUIThread=True, requestType="POST")
    def calibrateStageAffine(
        self,
        detectorName: str = None,
        objectiveId: str = None,
        stepSizeUm: float = 100.0,
        pattern: str = "cross",
        nSteps: int = 1,
        crop_size: int = 1024,
        isDEBUG: bool = False,
        backlashUm: float = 0.0,
    ):
        """Run an affine stage-to-camera calibration in a background thread.

        The result is stored as **pending** for the (detector, objective) pair
        and must be reviewed via ``getPendingCalibration`` and confirmed via
        ``applyPendingCalibration`` (or dropped via
        ``discardPendingCalibration``) before it takes effect.
        """
        if detectorName is None or detectorName == "":
            detectorName = self._defaultDetectorName()
        objectiveId = self._resolveObjectiveId(objectiveId)

        if detectorName not in self._master.detectorsManager.getAllDeviceNames():
            return {
                "success": False,
                "error": f"Unknown detector '{detectorName}'",
            }

        if not self._validateCameraIntensity(detectorName):
            return {
                "success": False,
                "error": (
                    "Camera intensity out of range (saturated or too dark). "
                    "Adjust exposure or lighting before calibration."
                ),
            }

        thread = threading.Thread(
            target=self._calibrateStageAffineInThread,
            args=(detectorName, objectiveId, stepSizeUm, pattern, nSteps, crop_size, isDEBUG, backlashUm),
            daemon=True,
        )
        thread.start()
        return {
            "success": True,
            "pending": False,
            "detectorName": detectorName,
            "objectiveId": objectiveId,
            "message": "Calibration started in background thread. Poll getPendingCalibration for results.",
        }

    def _calibrateStageAffineInThread(
        self,
        detectorName: str,
        objectiveId: str,
        stepSizeUm: float,
        pattern: str,
        nSteps: int,
        crop_size: int,
        isDEBUG: bool,
        backlashUm: float = 0.0,
    ):
        key = self._makeKey(detectorName, objectiveId)
        try:
            helper = PixelCalibrationClass(self, detectorName=detectorName)
            result = helper.calibrate_affine(
                backlash_um=backlashUm,
                detector_name=detectorName,
                step_size_um=stepSizeUm,
                pattern=pattern,
                n_steps=nSteps,
                crop_size=crop_size,
                isDEBUG=isDEBUG,
            )
            result_serializable = _convert_to_native(result)

            self._pendingCalibration[key] = {
                "affine_matrix": result_serializable.get("affine_matrix", []),
                "metrics": result_serializable.get("metrics", {}),
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
                "detector_name": detectorName,
                "objective_id": objectiveId,
            }
            self._logger.info(
                f"Calibration pending approval for '{detectorName}' @ objective '{objectiveId}'"
            )
            return self._pendingCalibration[key]
        except Exception as exc:
            self._logger.error(
                f"Calibration thread failed for '{key}': {exc}", exc_info=True
            )
            self._pendingCalibration[key] = {
                "error": str(exc),
                "detector_name": detectorName,
                "objective_id": objectiveId,
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            return None

    @APIExport()
    def getPendingCalibration(self, detectorName: str = None, objectiveId: str = None):
        """Return pending (un-applied) calibration data for one or all entries."""
        if detectorName:
            objectiveId = self._resolveObjectiveId(objectiveId)
            key = self._makeKey(detectorName, objectiveId)
            data = self._pendingCalibration.get(key)
            if data is None:
                return {
                    "success": False,
                    "message": f"No pending calibration for '{detectorName}' @ '{objectiveId}'",
                }
            # Always sanitise: stored payloads may carry NaN/Inf from numpy math.
            return _convert_to_native({"success": True, **data})
        return _convert_to_native({"success": True, "pending": self._pendingCalibration})

    @APIExport(requestType="POST")
    def applyPendingCalibration(
        self,
        detectorName: str,
        objectiveId: str = None,
        affineMatrix: list = None,
        metrics: dict = None,
    ):
        """Persist (and apply) a pending calibration, optionally with edited values."""
        objectiveId = self._resolveObjectiveId(objectiveId)
        key = self._makeKey(detectorName, objectiveId)
        pending = self._pendingCalibration.get(key)
        if pending is None:
            return {
                "success": False,
                "message": f"No pending calibration for '{detectorName}' @ '{objectiveId}'",
            }
        if "error" in pending:
            return {
                "success": False,
                "message": f"Pending calibration has error: {pending['error']}",
            }

        final_matrix = affineMatrix if affineMatrix is not None else pending["affine_matrix"]
        final_metrics = metrics if metrics is not None else pending["metrics"]

        calib_data = {
            "affine_matrix": final_matrix,
            "metrics": final_metrics,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "detector_name": detectorName,
            "objective_id": objectiveId,
        }

        self._setupInfo.setAffineCalibration(key, calib_data)
        try:
            import imswitch.imcontrol.model.configfiletools as configfiletools
            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)
        except Exception as exc:
            self._logger.warning(f"Could not save calibration to disk: {exc}")

        self.affineCalibrations[key] = calib_data
        # Apply only if this matches the currently selected objective.
        if objectiveId == self.currentObjective:
            self._applyCalibrationToDetector(detectorName, calib_data)

        del self._pendingCalibration[key]

        return _convert_to_native({
            "success": True,
            "detectorName": detectorName,
            "objectiveId": objectiveId,
            "message": f"Calibration applied and saved for '{detectorName}' @ '{objectiveId}'",
            "affineMatrix": final_matrix,
            "metrics": final_metrics,
        })

    @APIExport(requestType="POST")
    def discardPendingCalibration(self, detectorName: str, objectiveId: str = None):
        """Drop a pending calibration without persisting or applying it."""
        objectiveId = self._resolveObjectiveId(objectiveId)
        key = self._makeKey(detectorName, objectiveId)
        if key in self._pendingCalibration:
            del self._pendingCalibration[key]
            return {
                "success": True,
                "message": f"Pending calibration discarded for '{detectorName}' @ '{objectiveId}'",
            }
        return {
            "success": False,
            "message": f"No pending calibration for '{detectorName}' @ '{objectiveId}'",
        }

    # ------------------------------------------------------------------ #
    # API: read / write persisted calibrations                            #
    # ------------------------------------------------------------------ #

    @APIExport()
    def getCalibratedDetectors(self):
        """Return a list of (detector, objective) entries that have a persisted calibration."""
        entries = []
        for key in self.affineCalibrations.keys():
            det, obj = self._splitKey(key)
            entries.append({"detectorName": det, "objectiveId": obj, "key": key})
        return {
            "success": True,
            "entries": entries,
            "detectors": sorted({e["detectorName"] for e in entries}),
        }

    @APIExport()
    def getAvailableDetectors(self):
        """List every detector known to the system plus its per-objective calibration status.

        Used by the frontend to populate detector dropdowns in both the automatic
        and manual calibration tabs.
        """
        names = list(self._master.detectorsManager.getAllDeviceNames())
        out = []
        for name in names:
            calibrated_objectives = []
            for key in self.affineCalibrations.keys():
                det, obj = self._splitKey(key)
                if det == name:
                    calibrated_objectives.append(obj)
            out.append({
                "detectorName": name,
                "calibratedObjectives": sorted(set(calibrated_objectives)),
            })
        return {
            "success": True,
            "detectors": out,
            "detectorNames": names,
            "currentObjective": self.currentObjective,
        }

    @APIExport()
    def getCalibrationData(self, detectorName: str = "default", objectiveId: str = None):
        """Return persisted calibration data for the given (detector, objective) pair."""
        objectiveId = self._resolveObjectiveId(objectiveId)
        calib = self._resolveCalibration(detectorName, objectiveId)
        if calib is None:
            return {
                "success": False,
                "error": f"No calibration found for '{detectorName}' @ '{objectiveId}'",
            }
        return _convert_to_native({
            "success": True,
            "detectorName": detectorName,
            "objectiveId": objectiveId,
            "affineMatrix": calib.get("affine_matrix", [[1, 0, 0], [0, 1, 0]]),
            "metrics": calib.get("metrics", {}),
            "timestamp": calib.get("timestamp", "unknown"),
        })

    @APIExport(requestType="POST")
    def setCalibrationData(
        self,
        detectorName: str,
        affineMatrix: list,
        objectiveId: str = None,
        metrics: dict = None,
    ):
        """Directly set and apply a calibration (e.g. from a manual workflow)."""
        objectiveId = self._resolveObjectiveId(objectiveId)
        if not isinstance(affineMatrix, list) or len(affineMatrix) != 2:
            return {"success": False, "error": "affineMatrix must be a 2-row nested list"}
        for row in affineMatrix:
            if not isinstance(row, list) or len(row) != 3:
                return {
                    "success": False,
                    "error": "affineMatrix rows must each have 3 numeric entries",
                }
        try:
            affineMatrix = [[float(v) for v in row] for row in affineMatrix]
        except (ValueError, TypeError) as exc:
            return {"success": False, "error": f"affineMatrix contains non-numeric values: {exc}"}

        key = self._makeKey(detectorName, objectiveId)
        calib_data = {
            "affine_matrix": affineMatrix,
            "metrics": metrics if metrics is not None else {},
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "detector_name": detectorName,
            "objective_id": objectiveId,
        }
        self._setupInfo.setAffineCalibration(key, calib_data)
        try:
            import imswitch.imcontrol.model.configfiletools as configfiletools
            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)
        except Exception as exc:
            self._logger.warning(f"Could not save calibration to disk: {exc}")

        self.affineCalibrations[key] = calib_data
        if objectiveId == self.currentObjective:
            self._applyCalibrationToDetector(detectorName, calib_data)
        # Surface common derived values so the frontend can render them directly.
        m = calib_data["metrics"] or {}
        sx = float(m.get("scale_x_um_per_pixel", 1.0))
        sy = float(m.get("scale_y_um_per_pixel", 1.0))
        return _convert_to_native({
            "success": True,
            "detectorName": detectorName,
            "objectiveId": objectiveId,
            "message": f"Calibration data saved for '{detectorName}' @ '{objectiveId}'",
            "pixelSizeUm": (abs(sx) + abs(sy)) / 2.0,
            "scaleXUmPerPixel": sx,
            "scaleYUmPerPixel": sy,
            "affineMatrix": calib_data["affine_matrix"],
            "metrics": calib_data["metrics"],
            "displacementPx": m.get("displacement_px"),
            "movementDistanceUm": m.get("movement_distance_um"),
            "movementAxis": m.get("movement_axis"),
        })

    @APIExport()
    def deleteCalibration(self, detectorName: str, objectiveId: str = None):
        """Delete a calibration and reset the detector to defaults."""
        objectiveId = self._resolveObjectiveId(objectiveId)
        key = self._makeKey(detectorName, objectiveId)
        # Look up the right key (composite or legacy detector-only).
        store_key = key if key in self.affineCalibrations else (
            detectorName if detectorName in self.affineCalibrations else None
        )
        if store_key is None:
            return {
                "success": False,
                "message": f"No calibration found for '{detectorName}' @ '{objectiveId}'",
            }

        del self.affineCalibrations[store_key]
        if (
            self._setupInfo.PixelCalibration is not None
            and store_key in self._setupInfo.PixelCalibration.affineCalibrations
        ):
            del self._setupInfo.PixelCalibration.affineCalibrations[store_key]
        try:
            import imswitch.imcontrol.model.configfiletools as configfiletools
            options, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(options, self._setupInfo)
        except Exception as exc:
            self._logger.warning(f"Could not persist deletion: {exc}")

        identity = {
            "affine_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            "metrics": {
                "scale_x_um_per_pixel": 1.0,
                "scale_y_um_per_pixel": 1.0,
                "rotation_deg": 0.0,
            },
        }
        if objectiveId == self.currentObjective:
            self._applyCalibrationToDetector(detectorName, identity)
        return {
            "success": True,
            "detectorName": detectorName,
            "objectiveId": objectiveId,
            "message": f"Calibration deleted for '{detectorName}' @ '{objectiveId}' (reset to identity)",
        }

    # ------------------------------------------------------------------ #
    # API: manual two-point pixel-size calibration                        #
    # ------------------------------------------------------------------ #

    @APIExport(requestType="POST")
    def manualPixelSizeCalibration(
        self,
        point1X: float = 0.0,
        point1Y: float = 0.0,
        point2X: float = 0.0,
        point2Y: float = 0.0,
        movementDistanceUm: float = 100.0,
        movementAxis: str = "X",
        detectorName: str = None,
        objectiveId: str = None,
        previewSubsamplingFactor: float = 1.0,
    ):
        """Compute and store a pixel size from two image points and a known stage move.

        ``previewSubsamplingFactor`` is the ratio ``sensor_pixels / preview_pixels``
        of the live stream that the user clicked on. If the frontend already scaled
        the click coordinates up to full sensor resolution, leave it at ``1.0``.
        Otherwise pass the live-stream subsampling factor and the controller will
        scale the displacement accordingly so the pixel size refers to *sensor*
        pixels (which is what every other controller expects).
        """
        if detectorName is None or detectorName == "":
            detectorName = self._defaultDetectorName()
        objectiveId = self._resolveObjectiveId(objectiveId)

        if movementDistanceUm <= 0:
            return {"success": False, "error": "movementDistanceUm must be > 0"}
        if movementAxis not in ("X", "Y"):
            return {"success": False, "error": "movementAxis must be 'X' or 'Y'"}

        try:
            sub = float(previewSubsamplingFactor) if previewSubsamplingFactor else 1.0
        except (TypeError, ValueError):
            sub = 1.0
        if sub <= 0:
            sub = 1.0

        dx = (point2X - point1X) * sub
        dy = (point2Y - point1Y) * sub
        displacement_px = float(np.sqrt(dx * dx + dy * dy))
        if displacement_px < 1.0:
            return {
                "success": False,
                "error": "Pixel displacement < 1 px - mark two distinct points or move further.",
            }

        pixel_size_um = abs(movementDistanceUm) / displacement_px
        # Preserve previous rotation/flip if a calibration already exists for this (detector, objective).
        existing_calib = self._resolveCalibration(detectorName, objectiveId) or {}
        existing = existing_calib.get("affine_matrix")
        rotation_deg = 0.0
        if existing is not None and len(existing) >= 2:
            a = np.array(existing, dtype=float)
            m = a[:2, :2]
            old_sx = float(np.linalg.norm(m[:, 0]))
            old_sy = float(np.linalg.norm(m[:, 1]))
            if old_sx > 0 and old_sy > 0:
                direction = np.array(
                    [
                        [m[0, 0] / old_sx, m[0, 1] / old_sy],
                        [m[1, 0] / old_sx, m[1, 1] / old_sy],
                    ]
                )
                new_m = direction * pixel_size_um
                affine_matrix = [
                    [float(new_m[0, 0]), float(new_m[0, 1]), float(a[0, 2])],
                    [float(new_m[1, 0]), float(new_m[1, 1]), float(a[1, 2])],
                ]
                rotation_deg = float(np.degrees(np.arctan2(direction[1, 0], direction[0, 0])))
            else:
                affine_matrix = [
                    [pixel_size_um, 0.0, 0.0],
                    [0.0, pixel_size_um, 0.0],
                ]
        else:
            affine_matrix = [
                [pixel_size_um, 0.0, 0.0],
                [0.0, pixel_size_um, 0.0],
            ]

        metrics = {
            "scale_x_um_per_pixel": pixel_size_um,
            "scale_y_um_per_pixel": pixel_size_um,
            "rotation_deg": rotation_deg,
            "quality": "manual",
            "method": "manual_two_point",
            "displacement_px": displacement_px,
            "movement_distance_um": float(movementDistanceUm),
            "movement_axis": movementAxis,
            "point1": [float(point1X), float(point1Y)],
            "point2": [float(point2X), float(point2Y)],
        }
        return self.setCalibrationData(
            detectorName=detectorName,
            affineMatrix=affine_matrix,
            objectiveId=objectiveId,
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    # API: manual four-point affine calibration                           #
    # ------------------------------------------------------------------ #

    @APIExport(requestType="POST")
    def manualFourPointCalibration(
        self,
        pointA1X: float = 0.0,
        pointA1Y: float = 0.0,
        pointA2X: float = 0.0,
        pointA2Y: float = 0.0,
        movementDistanceXUm: float = 100.0,
        pointB1X: float = 0.0,
        pointB1Y: float = 0.0,
        pointB2X: float = 0.0,
        pointB2Y: float = 0.0,
        movementDistanceYUm: float = 100.0,
        detectorName: str = None,
        objectiveId: str = None,
        previewSubsamplingFactor: float = 1.0,
    ):
        """Compute a full affine calibration from two point-pairs (one per axis).

        Workflow (frontend drives the stage moves):
          1. Mark feature → P_A1, move stage +X, mark same feature → P_A2
          2. Mark feature → P_B1, move stage +Y, mark same feature → P_B2

        The two pixel-shift / stage-shift pairs fully determine a 2×2 affine
        matrix that encodes scale, rotation *and* flip — consistent with the
        automatic ``calibrateStageAffine`` output.

        ``previewSubsamplingFactor``: ratio sensor_pixels / preview_pixels.
        Pass the live-stream subsampling factor so pixel coordinates are
        scaled to full sensor resolution before computation.
        """
        if detectorName is None or detectorName == "":
            detectorName = self._defaultDetectorName()
        objectiveId = self._resolveObjectiveId(objectiveId)

        if movementDistanceXUm <= 0:
            return {"success": False, "error": "movementDistanceXUm must be > 0"}
        if movementDistanceYUm <= 0:
            return {"success": False, "error": "movementDistanceYUm must be > 0"}

        try:
            sub = float(previewSubsamplingFactor) if previewSubsamplingFactor else 1.0
        except (TypeError, ValueError):
            sub = 1.0
        if sub <= 0:
            sub = 1.0

        # Pixel shifts scaled to sensor coordinates
        pixel_shift_x = np.array([
            (pointA2X - pointA1X) * sub,
            (pointA2Y - pointA1Y) * sub,
        ])
        pixel_shift_y = np.array([
            (pointB2X - pointB1X) * sub,
            (pointB2Y - pointB1Y) * sub,
        ])

        min_disp = 1.0
        if np.linalg.norm(pixel_shift_x) < min_disp:
            return {"success": False, "error": "X-axis pixel displacement < 1 px — mark two distinct points or move further."}
        if np.linalg.norm(pixel_shift_y) < min_disp:
            return {"success": False, "error": "Y-axis pixel displacement < 1 px — mark two distinct points or move further."}

        pixel_shifts = np.array([pixel_shift_x, pixel_shift_y])  # (2, 2)
        stage_shifts = np.array([
            [movementDistanceXUm, 0.0],
            [0.0, movementDistanceYUm],
        ])  # (2, 2)

        # Solve for A: stage = pixel @ A^T  →  A^T = inv(pixel_shifts) @ stage_shifts
        try:
            A_T = np.linalg.solve(pixel_shifts, stage_shifts)
        except np.linalg.LinAlgError:
            return {
                "success": False,
                "error": "Pixel shifts are collinear — the four points don't span both axes.",
            }
        A = A_T.T  # 2×2

        # SVD decomposition — same extraction as compute_affine_matrix
        U, singular_values, Vt = np.linalg.svd(A)
        rotation_deg = float(np.degrees(np.arctan2(U[1, 0], U[0, 0])))
        scale_x = float(singular_values[0])
        scale_y = float(singular_values[1])

        # Encode reflection (flip) via determinant sign.
        # det(A) < 0 means the mapping includes a mirror.
        det_A = float(np.linalg.det(A))
        if det_A < 0:
            scale_y = -scale_y

        # Build 2×3 affine matrix (zero translation — we measured shifts, not positions)
        affine_matrix = [
            [float(A[0, 0]), float(A[0, 1]), 0.0],
            [float(A[1, 0]), float(A[1, 1]), 0.0],
        ]

        # Quality: with exactly 2 data points the fit is exact (RMSE = 0)
        predicted = pixel_shifts @ A.T
        residuals = stage_shifts - predicted
        rmse = float(np.sqrt(np.mean(np.sum(residuals ** 2, axis=1))))

        metrics = {
            "scale_x_um_per_pixel": scale_x,
            "scale_y_um_per_pixel": scale_y,
            "rotation_deg": rotation_deg,
            "rmse_um": rmse,
            "quality": "manual",
            "method": "manual_four_point",
            "condition_number": float(np.linalg.cond(A)),
            "determinant": det_A,
            "point_a1": [float(pointA1X), float(pointA1Y)],
            "point_a2": [float(pointA2X), float(pointA2Y)],
            "point_b1": [float(pointB1X), float(pointB1Y)],
            "point_b2": [float(pointB2X), float(pointB2Y)],
            "movement_distance_x_um": float(movementDistanceXUm),
            "movement_distance_y_um": float(movementDistanceYUm),
            "pixel_shift_x": [float(pixel_shift_x[0]), float(pixel_shift_x[1])],
            "pixel_shift_y": [float(pixel_shift_y[0]), float(pixel_shift_y[1])],
        }
        return self.setCalibrationData(
            detectorName=detectorName,
            affineMatrix=affine_matrix,
            objectiveId=objectiveId,
            metrics=metrics,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _validateCameraIntensity(self, detectorName: str) -> bool:
        """Reject calibrations when the detector frame is saturated or too dark."""
        try:
            detector = self._master.detectorsManager[detectorName]
            if not hasattr(detector, "getLatestFrame"):
                return True
            frame = detector.getLatestFrame()
            if frame is None:
                return True
            max_val = float(np.max(frame))
            mean_val = float(np.mean(frame))
            if frame.dtype == np.uint8:
                min_threshold = 10
            elif frame.dtype == np.uint16:
                min_threshold = 50 if max_val < 4096 else 500
            else:
                min_threshold = 0.02
            if mean_val <= min_threshold:
                self._logger.warning(
                    f"[{detectorName}] frame too dark (mean={mean_val:.1f}, threshold={min_threshold})"
                )
                return False
            self._logger.info(
                f"[{detectorName}] intensity OK: mean={mean_val:.1f}, max={max_val:.1f}"
            )
            return True
        except Exception as exc:
            self._logger.warning(f"Intensity validation failed: {exc}")
            return True


class PixelCalibrationClass:
    """Camera-to-stage affine calibration math (kept verbatim from the legacy controller)."""

    def __init__(self, parent: PixelCalibrationController, detectorName: str = None):
        self._parent = parent
        self._detectorName = detectorName or parent._defaultDetectorName()

    # ---------- detector access -------------------------------------------------
    @property
    def _detector(self):
        return self._parent._master.detectorsManager[self._detectorName]

    def _grab_image(self, crop_size: int = 1024, frameSync: int = 3, returnFrameNumber: bool = False):
        timeout_s = 1.0  # TODO: scale with exposure
        t0 = time.time()
        last_fn = -1
        cur_fn = None
        mFrame = None
        while True:
            mFrame, cur_fn = self._detector.getLatestFrame(returnFrameNumber=True)
            if last_fn == -1:
                last_fn = cur_fn
            if time.time() - t0 > timeout_s:
                if mFrame is None:
                    mFrame = self._detector.getLatestFrame(returnFrameNumber=False)
                break
            if cur_fn <= last_fn + frameSync:
                time.sleep(0.01)
            else:
                break

        if mFrame.ndim > 2:
            mFrame = np.mean(mFrame, axis=2)
        cropped = np.array(nip.extract(mFrame, (crop_size, crop_size)))
        if returnFrameNumber:
            return cropped, cur_fn
        return cropped

    def _get_stage_position(self):
        stage = self._parent._master.positionersManager[
            self._parent._master.positionersManager.getAllDeviceNames()[0]
        ]
        pos = stage.getPosition()
        return np.array([pos["X"], pos["Y"], pos["Z"]])

    def _move_stage(self, position_um):
        stage = self._parent._master.positionersManager[
            self._parent._master.positionersManager.getAllDeviceNames()[0]
        ]
        stage.move(value=position_um, axis="XY", is_absolute=True, is_blocking=True)
        if len(position_um) > 2:
            stage.move(value=position_um[2], axis="Z", is_absolute=True, is_blocking=True)

    # ---------- main routine ----------------------------------------------------
    def calibrate_affine(
        self,
        detector_name: str = None,
        step_size_um: float = 100.0,
        pattern: str = "cross",
        n_steps: int = 4,
        settle_time: float = 0.2,
        crop_size: int = 1024,
        isDEBUG: bool = False,
        backlash_um: float = 0.0,
    ):
        """Capture phase-correlation samples and compute an affine matrix."""
        if detector_name and detector_name != self._detectorName:
            self._detectorName = detector_name
        self._parent._logger.info(
            f"Starting affine calibration on detector '{self._detectorName}'"
        )

        start_position = self._get_stage_position()
        self._parent._logger.info(f"Starting position: {start_position[:2]} µm")

        # Backlash compensation: move away in X and Y, then back to start
        if backlash_um > 0:
            self._parent._logger.info(f"Backlash compensation: {backlash_um} µm in X and Y")
            backlash_target = start_position + np.array([backlash_um, backlash_um, 0])
            self._move_stage(backlash_target)
            time.sleep(settle_time)
            self._move_stage(start_position)
            time.sleep(settle_time)

        time.sleep(settle_time)
        ref_image = self._grab_image(crop_size=crop_size)

        if pattern == "cross":
            offsets = [
                (0, 0),
                (step_size_um, 0),
                (-step_size_um, 0),
                (0, 0),
                (0, step_size_um),
                (0, -step_size_um),
                (0, 0),
            ]
        elif pattern == "grid":
            offsets = []
            half_range = (n_steps - 1) / 2.0
            for i in range(n_steps):
                for j in range(n_steps):
                    offsets.append(((i - half_range) * step_size_um, (j - half_range) * step_size_um))
        else:
            raise ValueError(f"Unknown pattern: {pattern}")

        pixel_shifts = []
        stage_shifts = []
        correlations = []
        try:
            for idx, (dx, dy) in enumerate(offsets):
                target = start_position + np.array([dx, dy, 0])
                self._move_stage(target)
                time.sleep(settle_time)
                image = self._grab_image(crop_size=crop_size)
                shift, corr = measure_pixel_shift(np.array(ref_image), np.array(image))
                pixel_shifts.append(shift)
                stage_shifts.append([dx, dy])
                correlations.append(corr)
                self._parent._logger.debug(
                    f"Step {idx + 1}/{len(offsets)}: stage=({dx:.1f},{dy:.1f}) "
                    f"px=({shift[0]:.2f},{shift[1]:.2f}) corr={corr:.3f}"
                )
                if isDEBUG:
                    import tifffile
                    tifffile.imwrite("affine_calib_image_.tif", image.astype(np.float32), append=True)
        finally:
            try:
                self._move_stage(start_position)
            except Exception:  # pragma: no cover — best-effort recovery
                pass

        pixel_shifts = np.array(pixel_shifts)
        stage_shifts = np.array(stage_shifts)
        correlations = np.array(correlations)

        affine_matrix, inlier_mask, metrics = compute_affine_matrix(pixel_shifts, stage_shifts)
        metrics["mean_correlation"] = float(np.mean(correlations))
        metrics["min_correlation"] = float(np.min(correlations))

        self._parent._logger.info(
            f"Calibration done: quality={metrics.get('quality', 'n/a')} "
            f"rmse={metrics.get('rmse_um', 0):.3f}µm "
            f"rot={metrics.get('rotation_deg', 0):.2f}° "
            f"scale=({metrics.get('scale_x_um_per_pixel', 0):.3f},{metrics.get('scale_y_um_per_pixel', 0):.3f})"
        )

        return {
            "affine_matrix": affine_matrix,
            "metrics": metrics,
            "pixel_displacements": pixel_shifts,
            "stage_displacements": stage_shifts,
            "correlation_values": correlations,
            "inlier_mask": inlier_mask,
            "starting_position": start_position,
        }


# Copyright (C) 2020-2024 ImSwitch developers
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
import os
