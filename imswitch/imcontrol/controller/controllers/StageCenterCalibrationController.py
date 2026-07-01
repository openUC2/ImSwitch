"""StageCenterCalibrationController.

Stage-center calibration via XY raster scan.

Replaces the legacy out-growing square spiral with a deterministic raster
scan around a user supplied start position. For every visited grid node a
mean intensity sample is recorded (subsampled for speed) so the frontend
can render a heatmap. The brightest spot then defines the stage position
of the well-defined hole on the openUC2 calibration slide and is used to
write the stage offset.

Workflow (driven by the wizard in FRAME Settings -> Stage Offset Calibration):
  1. The user inserts the calibration slide into slot 1 and roughly centres
     the well above the objective.
  2. ``performCalibration`` performs the raster scan and stores
     ``(x_um, y_um, mean_intensity)`` tuples.
  3. ``getCalibrationHeatmap`` returns the data plus the brightest spot --
     this gives the mechanical zero of the reference slide.
  4. The frontend lets the user accept / override the value before it is
     persisted into the JSON setup file via ``PositionerController``.

The legacy brightness-spiral worker is preserved (commented out) so it can
be reactivated for the planned "stage 2" automatic refinement.
"""

import json
import os
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np

from imswitch.imcommon.framework import Signal, Mutex
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController


# Single canonical location for the most recent heatmap. The CSV is
# overwritten on every run so the frontend can load "the last scan" without
# walking a tree of timestamped folders.
_LATEST_HEATMAP_DIR = os.path.join(os.path.expanduser("~"), "imswitch_calibrations")
_LATEST_HEATMAP_CSV = os.path.join(_LATEST_HEATMAP_DIR, "stage_center_latest.csv")
_LATEST_HEATMAP_JSON = os.path.join(_LATEST_HEATMAP_DIR, "stage_center_latest.json")


class StageCenterCalibrationController(ImConWidgetController):

    sigImageReceived = Signal(np.ndarray)
    # x_um, y_um, mean_intensity
    sigCalibrationProgress = Signal(float, float, float)

    # ---------------------------- initialisation -----------------------------

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        self._task: Optional[threading.Thread] = None
        self._is_running: bool = False
        # Each entry: (x_um, y_um, mean_intensity). Tuples of floats keep the
        # in-memory footprint small, even for large rasters.
        self._samples: list[tuple[float, float, float]] = []
        self._scan_meta: dict = {}
        self._run_mutex = Mutex()
        self._useTriggeredGrab: bool = False
        # Restore the last heatmap from disk so the tab repopulates after a
        # frontend reload.
        self._restoreLatestHeatmap()

    # ------------------------------ helpers ----------------------------------

    def getDetector(self):
        return self._master.detectorsManager[
            self._master.detectorsManager.getAllDeviceNames()[0]
        ]

    def getStage(self):
        stageName = self._master.positionersManager.getAllDeviceNames()[0]
        return self._master.positionersManager[stageName]

    def _hasPixelCalibration(self) -> bool:
        """Return True when at least one detector has a usable pixel size.

        We consider the calibration valid when the configured PixelCalibration
        controller can resolve a non-default pixel size for the active
        detector. Falls back to ``True`` when no PixelCalibration controller
        is present so we do not break headless / minimal setups.
        """
        try:
            pc = self._master.subControllers.get("PixelCalibration") \
                if hasattr(self._master, "subControllers") else None
        except Exception:
            pc = None

        # Fallback: try the parent _moduleCommChannel registry
        try:
            if pc is None:
                pc = self._commChannel.getController("PixelCalibration")
        except Exception:
            pc = None

        if pc is None:
            # No controller -> cannot enforce; allow calibration to run.
            return True
        try:
            info = pc.getPixelSizeUm(detectorName=None, objectiveId=None)
            return bool(info and info.get("success") and info.get("pixelSizeUm"))
        except Exception:
            return True

    def _beginTriggeredAcquisition(self) -> bool:
        """Switch the camera to software trigger for in-sync, post-move grabs.

        Free-running acquisition buffers frames asynchronously, so a grab can
        return a frame exposed before the stage settled. In software-trigger
        mode each frame is exposed in response to an explicit trigger fired
        after the move. Returns ``True`` when active, ``False`` when the
        detector does not support software triggering (fallback stays active).
        """
        detector = self.getDetector()
        if not hasattr(detector, "snapSync") or not hasattr(detector, "setTriggerSource"):
            self._useTriggeredGrab = False
            return False
        try:
            ok = bool(detector.setTriggerSource("software"))
        except Exception as e:
            self._logger.warning(f"Could not enable software trigger: {e}")
            ok = False
        self._useTriggeredGrab = ok
        if ok:
            self._logger.info("Calibration: camera in software-trigger mode (in-sync grabs)")
        return ok

    def _endTriggeredAcquisition(self) -> None:
        """Restore continuous (free-run) acquisition after a triggered scan."""
        if not self._useTriggeredGrab:
            return
        self._useTriggeredGrab = False
        try:
            detector = self.getDetector()
            if hasattr(detector, "setTriggerSource"):
                detector.setTriggerSource("continuous")
                self._logger.info("Calibration: camera restored to continuous mode")
        except Exception as e:
            self._logger.warning(f"Could not restore continuous mode: {e}")

    def _grabFrame(self):
        """Return one camera frame.

        Triggered path (preferred): fire a software trigger via ``snapSync``
        so the exposure is guaranteed to happen after the stage settled.

        Fallback: flush any buffered frames and return the latest frame from
        ``getLatestFrame`` (original behaviour, used when the detector does
        not support software triggering).
        """
        detector = self.getDetector()
        # Triggered path.
        if self._useTriggeredGrab and hasattr(detector, "snapSync"):
            return detector.snapSync(timeout=2.0)
        # Fallback: discard frames accumulated during the move, then grab.
        try:
            if hasattr(detector, "flushBuffer"):
                detector.flushBuffer()
        except Exception:
            pass
        return detector.getLatestFrame()

    def _grabMeanFrame(self) -> Optional[float]:
        frame = self._grabFrame()
        if frame is None or getattr(frame, "size", 0) == 0:
            return None
        # Subsample heavily for speed (every 20th pixel in each direction).
        return float(np.mean(frame[::20, ::20]))

    # ------------------------------- API -------------------------------------

    @APIExport()
    def performCalibration(
        self,
        start_x: float,
        start_y: float,
        exposure_time_us: int = 3000,
        speed: int = 15000,
        step_um: float = 250.0,
        max_radius_um: float = 5000.0,
        brightness_factor: float = 1.4,  # kept for backward-compat / stage-2
        require_pixel_calibration: bool = True,
        move_to_brightest: bool = True,
    ) -> dict:
        """Run an XY raster scan and record (x, y, intensity) at each node.

        The scan covers a square of side ``2 * max_radius_um`` around
        ``(start_x, start_y)`` with grid spacing ``step_um``. When
        ``move_to_brightest`` is true the stage parks on top of the brightest
        sample at the end of the scan so the frontend can capture / confirm
        before storing the offset.
        """
        if self._is_running:
            return self._currentResult()

        if require_pixel_calibration and not self._hasPixelCalibration():
            return {
                "success": False,
                "error": "Pixel calibration is required before stage centering.",
            }

        self._is_running = True
        self._samples.clear()
        self._scan_meta = {
            "start_x": float(start_x),
            "start_y": float(start_y),
            "step_um": float(step_um),
            "max_radius_um": float(max_radius_um),
            "speed": int(speed),
            "exposure_time_us": int(exposure_time_us),
            "started_at": datetime.now().isoformat(timespec="seconds"),
            "move_to_brightest": bool(move_to_brightest),
        }

        self._task = threading.Thread(
            target=self._rasterWorker,
            args=(start_x, start_y, speed, step_um, max_radius_um, move_to_brightest),
        )
        self._task.start()

        return self._currentResult()

    @APIExport()
    def getIsCalibrationRunning(self) -> bool:
        return self._is_running

    @APIExport()
    def stopCalibration(self) -> dict:
        """Stop the raster scan and persist whatever samples were collected.

        The user often stops a scan early after visually identifying the
        bright spot - we should keep the partial result on disk so the
        frontend can still accept the brightest sample (or reload it later).
        """
        self._is_running = False
        if self._task is not None:
            self._task.join()
            self._task = None
        # Persist whatever samples we have so a "stop early" workflow still
        # leaves a usable heatmap behind.
        try:
            self._persistLatestHeatmap()
        except Exception as e:
            self._logger.warning(f"Persist on stop failed: {e}")
        self._logger.info("Calibration stopped.")
        return self._currentResult()

    @APIExport()
    def getCalibrationHeatmap(self) -> dict:
        """Return the raster samples plus the brightest spot."""
        return self._currentResult()

    @APIExport()
    def getRecommendedScanParameters(self) -> dict:
        """Suggest ``step_um`` based on the active detector's FOV.

        The frontend uses this to populate the scan inputs adaptively (so a
        4x objective scans coarsely, a 60x scans finely) without having to
        plumb pixel size + camera shape through the UI.

        Returns:
            ``{success, pixelSizeUm, frameWidth, frameHeight, fovXUm,
              fovYUm, recommendedStepUm, recommendedMaxRadiusUm}``
        """
        detector = self.getDetector()
        # Detector frame shape (height, width). Fall back to common defaults.
        frame_h, frame_w = 2048, 2048
        try:
            shape = getattr(detector, "shape", None)
            if shape is None:
                latest = detector.getLatestFrame()
                if latest is not None and getattr(latest, "shape", None):
                    shape = latest.shape
            if shape is not None and len(shape) >= 2:
                frame_h, frame_w = int(shape[0]), int(shape[1])
        except Exception:
            pass

        # Pixel size: prefer the PixelCalibration controller, fall back to the
        # detector's reported value.
        pixel_size_um = None
        try:
            pc = None
            if hasattr(self._master, "subControllers"):
                pc = self._master.subControllers.get("PixelCalibration")
            if pc is None:
                try:
                    pc = self._commChannel.getController("PixelCalibration")
                except Exception:
                    pc = None
            if pc is not None:
                info = pc.getPixelSizeUm(detectorName=None, objectiveId=None)
                if info and info.get("success"):
                    pixel_size_um = float(info.get("pixelSizeUm") or 0.0) or None
        except Exception:
            pixel_size_um = None

        if pixel_size_um is None:
            try:
                pixel_size_um = float(getattr(detector, "pixelSizeUm", 0.0)) or None
            except Exception:
                pixel_size_um = None

        # Final fallback: assume 1 um/px so the UI gets *some* defaults.
        if not pixel_size_um or pixel_size_um <= 0:
            pixel_size_um = 1.0

        fov_x_um = pixel_size_um * frame_w
        fov_y_um = pixel_size_um * frame_h

        # Step ~ 80% of the smaller FOV side, so neighbouring grid nodes overlap
        # slightly and we never miss the bright spot between two samples.
        step_um = max(10.0, 0.8 * min(fov_x_um, fov_y_um))
        # Default radius: enough to cover ~10 FOVs in each direction by default.
        max_radius_um = max(step_um * 4.0, 5000.0)

        return {
            "success": True,
            "pixelSizeUm": float(pixel_size_um),
            "frameWidth": int(frame_w),
            "frameHeight": int(frame_h),
            "fovXUm": float(fov_x_um),
            "fovYUm": float(fov_y_um),
            "recommendedStepUm": float(step_um),
            "recommendedMaxRadiusUm": float(max_radius_um),
        }

    # --------------------------- raster worker -------------------------------

    def _rasterWorker(self, cx: float, cy: float, speed: int,
                      step_um: float, max_r: float,
                      move_to_brightest: bool = True) -> None:
        try:
            # Prefer software-triggered grabs so each frame is exposed after
            # the stage has finished its move. Falls back to getLatestFrame
            # transparently when the detector does not support triggering.
            self._beginTriggeredAcquisition()
            stage = self.getStage()
            stage.move(axis="XY", value=(cx,cy), is_absolute=True, is_blocking=True)

            # Build symmetric grid around the start position. Use float steps so
            # the scan matches ``max_r`` exactly when it is divisible by
            # ``step_um``. ``np.arange`` with float endpoint can drift; use
            # ``linspace`` based on integer node count instead.
            n_half = max(1, int(round(max_r / max(step_um, 1.0))))
            offsets = np.linspace(-n_half * step_um, n_half * step_um,
                                  2 * n_half + 1)

            for row_idx, dy in enumerate(offsets):
                if not self._is_running:
                    break
                # Boustrophedon raster: alternate direction each row to keep
                # travel short.
                row_offsets = offsets if row_idx % 2 == 0 else offsets[::-1]
                target_y = cy + float(dy)
                stage.move(axis="Y", value=target_y, is_absolute=True, is_blocking=True)
                for dx in row_offsets:
                    if not self._is_running:
                        break
                    target_x = cx + float(dx)
                    stage.move(axis="X", value=target_x, is_absolute=True, is_blocking=True)
                    intensity = self._grabMeanFrame()
                    if intensity is None:
                        continue
                    self._samples.append((target_x, target_y, intensity))
                    try:
                        self.sigCalibrationProgress.emit(
                            target_x, target_y, intensity
                        )
                    except Exception:
                        pass

            # Park the stage at the brightest sample so the user can validate
            # the calibration target visually before persisting an offset.
            if move_to_brightest and self._samples:
                arr = np.asarray(self._samples, dtype=float)
                idx = int(np.argmax(arr[:, 2]))
                bx, by = float(arr[idx, 0]), float(arr[idx, 1])
                try:
                    stage.move(axis="XY", value=(bx,by), is_absolute=True, is_blocking=True)
                    self._scan_meta["parked_at"] = {"x": bx, "y": by}
                except Exception as e:
                    self._logger.warning(f"move-to-brightest failed: {e}")

            self._persistLatestHeatmap()
        finally:
            self._endTriggeredAcquisition()
            self._is_running = False

    # ----------------------------- result --------------------------------

    def _currentResult(self) -> dict:
        if not self._samples:
            return {
                "success": True,
                "running": self._is_running,
                "samples": [],
                "brightest": None,
                "meta": dict(self._scan_meta),
            }
        arr = np.asarray(self._samples, dtype=float)
        idx = int(np.argmax(arr[:, 2]))
        bx, by, bi = float(arr[idx, 0]), float(arr[idx, 1]), float(arr[idx, 2])
        return {
            "success": True,
            "running": self._is_running,
            "samples": [
                {"x": float(x), "y": float(y), "intensity": float(i)}
                for x, y, i in self._samples
            ],
            "brightest": {"x": bx, "y": by, "intensity": bi},
            "meta": dict(self._scan_meta),
        }

    # ------------------------- persistence helpers ---------------------------

    def _persistLatestHeatmap(self) -> None:
        """Overwrite a single canonical CSV + JSON sidecar with the latest scan.

        The CSV stays compatible with the legacy path (one timestamped copy
        is also written for archival). The JSON sidecar is what the frontend
        loads on mount - it contains the samples, the brightest spot, and the
        scan metadata in the exact shape ``getCalibrationHeatmap`` returns.
        """
        if not self._samples:
            return
        try:
            os.makedirs(_LATEST_HEATMAP_DIR, exist_ok=True)
            arr = np.asarray(self._samples, dtype=float)
            np.savetxt(
                _LATEST_HEATMAP_CSV,
                arr,
                delimiter=",",
                header="X(um),Y(um),mean_intensity",
            )
            payload = self._currentResult()
            with open(_LATEST_HEATMAP_JSON, "w") as f:
                json.dump(payload, f, indent=2)
            self._logger.info(
                f"Heatmap saved to {_LATEST_HEATMAP_CSV} and {_LATEST_HEATMAP_JSON}"
            )
        except Exception as e:
            self._logger.warning(f"Could not persist latest heatmap: {e}")

    def _restoreLatestHeatmap(self) -> None:
        """Load the latest heatmap from disk (if any) so the tab can repopulate."""
        try:
            if not os.path.isfile(_LATEST_HEATMAP_JSON):
                return
            with open(_LATEST_HEATMAP_JSON, "r") as f:
                data = json.load(f)
            samples = data.get("samples") or []
            self._samples = [
                (float(s["x"]), float(s["y"]), float(s["intensity"]))
                for s in samples
            ]
            self._scan_meta = dict(data.get("meta") or {})
        except Exception as e:
            self._logger.warning(f"Could not restore latest heatmap: {e}")
            self._samples = []
            self._scan_meta = {}

    @APIExport()
    def getLatestHeatmap(self) -> dict:
        """Return the most recent heatmap (loaded from disk if necessary).

        Equivalent to ``getCalibrationHeatmap`` while the controller has live
        samples in memory, but transparently restores from
        ``stage_center_latest.json`` so the heatmap survives a frontend
        reload.
        """
        if not self._samples:
            self._restoreLatestHeatmap()
        return self._currentResult()


# ---------------------------------------------------------------------------
# Stage 2 (planned, currently disabled): automatic brightest-spot refinement
# via the original out-growing square spiral. The implementation below is
# preserved verbatim from the previous version of this controller and can be
# re-enabled once the user wants to chain the raster scan with a fast spiral
# refinement around the brightest grid node.
# ---------------------------------------------------------------------------
#
# def _spiralWorker(self, cx, cy, speed, step_um, max_r, bf):
#     self.getStage().move("X", cx, True, True)
#     self.getStage().move("Y", cy, True, True)
#     baseline = self._grabMeanFrame()
#     if baseline is None:
#         self._logger.error("No detector image - aborting")
#         self._is_running = False
#         return
#     directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, N, W, S
#     dir_idx = 0
#     run_len = 1
#     legs_done = 0
#     off_x = off_y = 0.0
#     while self._is_running:
#         dx, dy = directions[dir_idx]
#         axis = "X" if dx else "Y"
#         for _ in range(run_len):
#             if not self._is_running:
#                 break
#             off_x += dx * step_um
#             off_y += dy * step_um
#             if max(abs(off_x), abs(off_y)) > max_r:
#                 self._is_running = False
#                 break
#             target = (cx + off_x) if axis == "X" else (cy + off_y)
#             ctrl = MovementController(self.getStage())
#             ctrl.move_to_position(target, axis=axis, speed=speed,
#                                   is_absolute=True)
#             while not ctrl.is_target_reached() and self._is_running:
#                 m = self._grabMeanFrame()
#                 p = self.getStage().getPosition()
#                 if m is not None:
#                     self._samples.append((p["X"], p["Y"], m))
#                 if m is not None and m >= baseline * bf:
#                     self._is_running = False
#                     break
#                 time.sleep(0.002)
#         dir_idx = (dir_idx + 1) % 4
#         legs_done += 1
#         if legs_done == 2:
#             legs_done = 0
#             run_len += 1


class MovementController:
    """Tiny helper that moves one axis asynchronously.

    Retained for the (currently disabled) stage-2 spiral refinement worker.
    """

    def __init__(self, stage):
        self.stage = stage
        self._done = False

    def move_to_position(self, value, axis, speed, is_absolute):
        self._done = False
        threading.Thread(
            target=self._move,
            args=(value, axis, speed, is_absolute),
            daemon=True,
        ).start()

    def _move(self, value, axis, speed, is_absolute):
        self.stage.move(
            axis=axis,
            value=value,
            speed=speed,
            is_absolute=is_absolute,
            is_blocking=True,
        )
        self._done = True

    def is_target_reached(self):
        return self._done
