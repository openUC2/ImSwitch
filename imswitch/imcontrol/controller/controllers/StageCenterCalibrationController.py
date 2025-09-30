import os
import threading
import time
from datetime import datetime

import numpy as np
import tifffile as tif

from imswitch import IS_HEADLESS
from imswitch.imcommon.framework import Signal, Mutex
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController


class StageCenterCalibrationController(ImConWidgetController):
    """Out‑growing square‑spiral search for the sample’s centre.

    * Each leg is executed as a **continuous move** on the respective axis using
      ``MovementController`` so that frames are grabbed **while** the stage is
      travelling – for **both X *and* Y directions**.
    * Spiral pitch = ``step_um``; side length increases after every second turn
      (E→N→W→S cycle).
    * Acquisition stops when the mean intensity (20‑pixel subsampling) rises by
      ``brightness_factor`` or when the requested ``max_radius_um`` is reached.
    * All visited (x, y) coordinates are stored and returned; a CSV copy is also
      written for record keeping.
    """

    sigImageReceived = Signal(np.ndarray)  # optional live‑view

    # ─────────────────────────── initialisation ────────────────────────────

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)


        # state
        self._task = None
        self._is_running = False
        self._positions: list[tuple[float, float]] = []
        self._run_mutex = Mutex()

    # ───────────────────────────── API ──────────────────────────────────────
    def getDetector(self):
        # devices
        return self._master.detectorsManager[self._master.detectorsManager.getAllDeviceNames()[0]]
        
    def getStage(self):
        stageName = self._master.positionersManager.getAllDeviceNames()[0]
        return self._master.positionersManager[stageName]

    @APIExport()
    def performCalibration(
        self,
        start_x: float,
        start_y: float,
        exposure_time_us: int = 3000,
        speed: int = 5000,
        step_um: float = 50.0,
        max_radius_um: float = 2000.0,
        brightness_factor: float = 1.4,
    ) -> list[tuple[float, float]]:
        if self._is_running:
            return self._positions

        self._is_running = True
        self._positions.clear()

        try:
            self.getDetector().setExposure(exposure_time_us)
        except AttributeError:
            pass

        self._task = threading.Thread(
            target=self._worker,
            args=(
                start_x,
                start_y,
                speed,
                step_um,
                max_radius_um,
                brightness_factor,
            ),
            daemon=True,
        )
        self._task.start()
        self._task.join()
        return self._positions.copy()

    @APIExport()
    def getIsCalibrationRunning(self):
        return self._is_running

    # ──────────────────────────── worker ────────────────────────────────────

    def _worker(self, cx, cy, speed, step_um, max_r, bf):
        self.getStage().move("X", cx, True, True)
        self.getStage().move("Y", cy, True, True)

        baseline = self._grabMeanFrame()
        if baseline is None:
            self._logger.error("No detector image – aborting")
            self._is_running = False
            return

        directions = [(1, 0), (0, 1), (-1, 0), (0, -1)]  # E, N, W, S
        dir_idx = 0
        run_len = 1
        legs_done = 0
        off_x = off_y = 0.0

        while self._is_running:
            dx, dy = directions[dir_idx]
            axis = "X" if dx else "Y"

            for _ in range(run_len):
                if not self._is_running:
                    break
                off_x += dx * step_um
                off_y += dy * step_um

                if max(abs(off_x), abs(off_y)) > max_r:
                    self._logger.info("Max radius reached – stop")
                    self._is_running = False
                    break

                target = (cx + off_x) if axis == "X" else (cy + off_y)
                ctrl = MovementController(self.getStage())
                ctrl.move_to_position(target, axis=axis, speed=speed, is_absolute=True)

                # ───── grab frames while travelling ─────
                while not ctrl.is_target_reached() and self._is_running:
                    m = self._grabMeanFrame()
                    p = self.getStage().getPosition()
                    self._positions.append((p["X"], p["Y"]))
                    if m is not None and m >= baseline * bf:
                        self._logger.info("Brightness threshold hit – done")
                        self._is_running = False
                        break
                    time.sleep(0.002)  # mild CPU relief

                if not self._is_running:
                    break

            if not self._is_running:
                break

            dir_idx = (dir_idx + 1) % 4
            legs_done += 1
            if legs_done == 2:
                legs_done = 0
                run_len += 1  # enlarge spiral

        self._savePositionsCsv()
        self._is_running = False

    @APIExport()
    def stopCalibration(self):
        """Stops the calibration process."""
        self._is_running = False
        if self._task is not None:
            self._task.join()
            self._task = None
        self._logger.info("Calibration stopped.")
    # ─────────────────────── helpers ────────────────────────────────────────

    def _grabMeanFrame(self):
        frame = self.getDetector().getLatestFrame()
        if frame is None or frame.size == 0:
            return None
        meanValue = np.mean(frame[::20, ::20])  # subsample for speed
        self._logger.debug(f"Mean value of frame: {meanValue}") 
        return meanValue

    def _savePositionsCsv(self):
        if not self._positions:
            return
        ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_path = os.path.join(os.path.expanduser("~"), "imswitch_calibrations", ts)
        os.makedirs(dir_path, exist_ok=True)
        path = os.path.join(dir_path, "stage_center_spiral.csv")
        np.savetxt(path, np.array(self._positions), delimiter=",", header="X(µm),Y(µm)")
        self._logger.info(f"Positions saved to {path}")

    # ─────────────────────── GUI convenience ───────────────────────────────

    def _startCalibrationFromGui(self):
        if IS_HEADLESS:
            return
        w = self._widget
        pos = self.performCalibration(w.spinStartX.value(), w.spinStartY.value(),
                                       w.spinExposure.value(), w.spinSpeed.value(),
                                       w.spinPitch.value())
        w.showPositions(pos)

    def displayImage(self, frame):
        if IS_HEADLESS:
            return
        self._widget.setImage(np.uint16(frame), colormap="gray", name="Calib", pixelsize=(1, 1))


class MovementController:
    """Tiny helper that moves one axis asynchronously."""

    def __init__(self, stage):
        self.stage = stage
        self._done = False

    def move_to_position(self, value, axis, speed, is_absolute):
        self._done = False
        threading.Thread(target=self._move, args=(value, axis, speed, is_absolute), daemon=True).start()

    def _move(self, value, axis, speed, is_absolute):
        self.stage.move(axis=axis, value=value, speed=speed, is_absolute=is_absolute, is_blocking=True)
        self._done = True

    def is_target_reached(self):
        return self._done
