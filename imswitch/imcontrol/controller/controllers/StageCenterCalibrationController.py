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

import os
import threading
import time
from datetime import datetime
from typing import Optional

import numpy as np

from imswitch.imcommon.framework import Signal, Mutex
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController


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

    def _grabMeanFrame(self) -> Optional[float]:
        frame = self.getDetector().getLatestFrame()
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
        speed: int = 5000,
        step_um: float = 250.0,
        max_radius_um: float = 5000.0,
        brightness_factor: float = 1.4,  # kept for backward-compat / stage-2
        require_pixel_calibration: bool = True,
    ) -> dict:
        """Run an XY raster scan and record (x, y, intensity) at each node.

        The scan covers a square of side ``2 * max_radius_um`` around
        ``(start_x, start_y)`` with grid spacing ``step_um``. Returns the
        sampled positions, the brightest position, and scan metadata.
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
        }

        try:
            pass # self.getDetector().setExposure(exposure_time_us)
        except AttributeError:
            pass

        self._task = threading.Thread(
            target=self._rasterWorker,
            args=(start_x, start_y, speed, step_um, max_radius_um)
            )
        self._task.start()

        return self._currentResult()

    @APIExport()
    def getIsCalibrationRunning(self) -> bool:
        return self._is_running

    @APIExport()
    def stopCalibration(self) -> dict:
        self._is_running = False
        if self._task is not None:
            self._task.join()
            self._task = None
        self._logger.info("Calibration stopped.")
        return self._currentResult()

    @APIExport()
    def getCalibrationHeatmap(self) -> dict:
        """Return the raster samples plus the brightest spot."""
        return self._currentResult()

    # --------------------------- raster worker -------------------------------

    def _rasterWorker(self, cx: float, cy: float, speed: int,
                      step_um: float, max_r: float) -> None:
        try:
            stage = self.getStage()
            stage.move("X", cx, True, True)
            stage.move("Y", cy, True, True)

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
                stage.move("Y", target_y, True, True)
                for dx in row_offsets:
                    if not self._is_running:
                        break
                    target_x = cx + float(dx)
                    stage.move("X", target_x, True, True)
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

            self._saveSamplesCsv()
        finally:
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

    def _saveSamplesCsv(self) -> None:
        if not self._samples:
            return
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_path = os.path.join(
            os.path.expanduser("~"), "imswitch_calibrations", ts
        )
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, "stage_center_raster.csv")
        np.savetxt(
            path,
            np.asarray(self._samples, dtype=float),
            delimiter=",",
            header="X(um),Y(um),mean_intensity",
        )
        self._logger.info(f"Raster samples saved to {path}")


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
