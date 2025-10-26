from imswitch import IS_HEADLESS
import time
import numpy as np
import scipy.ndimage as ndi
import threading
from scipy.optimize import curve_fit
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from skimage.filters import gaussian
from imswitch.imcommon.framework import Signal
import cv2
import queue

# Global axis for Z-positioning
gAxis = "Z"


def _gaussian(x, a, x0, sigma, c):
    return a * np.exp(-0.5 * ((x - x0) / (sigma + 1e-12)) ** 2) + c


def _robust_gaussian_fit(x, y):
    """
    Fit a 1D Gaussian to (x, y). Returns (x0, fit_y) where:
      - x0 is the fitted center
      - fit_y is the Gaussian evaluated on x (or None on failure)
    Falls back to argmax if fit fails.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    a0 = float(np.max(y) - np.min(y) + 1e-9)
    x0_0 = float(x[np.argmax(y)])
    sigma0 = max((np.max(x) - np.min(x)) / 5.0, 1e-6)
    c0 = float(np.min(y))

    p0 = [a0, x0_0, sigma0, c0]
    bounds = ([0.0, np.min(x) - abs(3 * sigma0), 1e-6, -np.inf],
              [np.inf, np.max(x) + abs(3 * sigma0), np.inf, np.inf])

    try:
        popt, _ = curve_fit(_gaussian, x, y, p0=p0, bounds=bounds, maxfev=20000)
        a, x0, sigma, c = popt
        fit_y = _gaussian(x, a, x0, sigma, c)
        return float(x0), fit_y
    except Exception:
        return float(x0_0), None


class MovementController:
    """
    Simple asynchronous mover:
      .move_to_position(value, axis, speed, is_absolute)
      .is_target_reached()
    """
    def __init__(self, stages):
        self.stages = stages
        self._lock = threading.Lock()
        self.target_reached = True
        self.target_position = None
        self.axis = None
        self.speed = None
        self.is_absolute = True
        self._thread = None

    def move_to_position(self, value, axis=gAxis, speed=None, is_absolute=True):
        with self._lock:
            self.target_reached = False
            self.target_position = value
            self.axis = axis
            self.speed = speed
            self.is_absolute = is_absolute
        self._thread = threading.Thread(target=self._move, daemon=True)
        self._thread.start()

    def _move(self):
        with self._lock:
            value = self.target_position
            axis = self.axis
            speed = self.speed
            is_absolute = self.is_absolute
        try:
            self.stages.move(value=value, axis=axis, speed=speed, is_absolute=is_absolute, is_blocking=True)
        finally:
            with self._lock:
                self.target_reached = True

    def is_target_reached(self):
        with self._lock:
            return bool(self.target_reached)


class AutofocusController(ImConWidgetController):
    """Linked to AutofocusWidget."""
    sigUpdateFocusPlot = Signal(object, object)   # x, y
    sigUpdateFocusValue = Signal(object)          # {"bestzpos": float}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        self.isAutofusRunning = False  # keep attribute name for compatibility

        if self._setupInfo.autofocus is not None:
            self.cameraName = self._setupInfo.autofocus.camera
            self.stageName = self._setupInfo.autofocus.positioner
        else:
            self.cameraName = self._master.detectorsManager.getAllDeviceNames()[0]
            self.stageName = self._master.positionersManager.getAllDeviceNames()[0]

        self.camera = self._master.detectorsManager[self.cameraName]
        self.stages = self._master.positionersManager[self.stageName]

        self._commChannel.sigAutoFocus.connect(self.autoFocus)
        if not IS_HEADLESS:
            self._widget.focusButton.clicked.connect(self.focusButton)

        self._moveController = MovementController(self.stages)

    def __del__(self):
        try:
            self.isAutofusRunning = False
            if hasattr(self, '_AutofocusThead') and self._AutofocusThead and self._AutofocusThead.is_alive():
                self._AutofocusThead.join(timeout=1.0)
        except Exception:
            pass
        if hasattr(super(), '__del__'):
            super().__del__()

    def focusButton(self):
        if not self.isAutofusRunning:
            rangez = float(self._widget.zStepRangeEdit.text())
            resolutionz = float(self._widget.zStepSizeEdit.text())
            defocusz = float(self._widget.zBackgroundDefocusEdit.text())
            self._widget.focusButton.setText('Stop')
            self.autoFocus(rangez, resolutionz, defocusz)
        else:
            self.isAutofusRunning = False

    @APIExport(runOnUIThread=True)
    def autoFocus(self, rangez: int = 100, resolutionz: int = 10, defocusz: int = 0):
        """Step-scan autofocus with Gaussian peak fit."""
        if self.isAutofusRunning:
            return
        self.isAutofusRunning = True
        self._AutofocusThead = threading.Thread(
            target=self._doAutofocusBackground,
            args=(rangez, resolutionz, defocusz),
            daemon=True
        )
        self._AutofocusThead.start()

    @APIExport(runOnUIThread=True)
    def autoFocusFast(self, sweep_range: float = 150.0, speed: float = None, defocusz: int = 0, axis: str = gAxis):
        """
        Continuous fast-sweep autofocus WITHOUT continuous Z-readback:
          - Move to z0 + sweep_range, then sweep down to z0 - sweep_range.
          - Record a timestamp for every image.
          - After the sweep, build Z positions by linear time mapping using
            z(t) = z_start + v_eff * (t - t_start), where
            v_eff = (z_end - z_start) / (t_end - t_start).
          - Fit Gaussian to (z, focus) and move to center.
        """
        if self.isAutofusRunning:
            return
        self.isAutofusRunning = True
        self._AutofocusThead = threading.Thread(
            target=self._doAutofocusFastBackground_timeMapped,
            args=(sweep_range, speed, defocusz, axis),
            daemon=True
        )
        self._AutofocusThead.start()

    @APIExport(runOnUIThread=True)
    def stopAutofocus(self):
        self.isAutofusRunning = False

    def grabCameraFrame(self, returnFrameNumber=False):
        try:
            return self.camera.getLatestFrame(returnFrameNumber=returnFrameNumber)
        except TypeError:
            if returnFrameNumber:
                f = self.camera.getLatestFrame()
                return f, None
            return self.camera.getLatestFrame()

    def recordFlatfield(self, nFrames=10, nGauss=16, defocusPosition=200, defocusAxis=gAxis):
        flatfield = []
        time.sleep(0.05)
        self.stages.move(value=defocusPosition, axis=defocusAxis, is_absolute=False, is_blocking=True)
        for _ in range(nFrames):
            flatfield.append(self.grabCameraFrame())
        flatfield = np.mean(np.array(flatfield), 0)
        flatfield = gaussian(flatfield, sigma=nGauss)
        self.stages.move(value=-defocusPosition, axis=defocusAxis, is_absolute=False, is_blocking=True)
        time.sleep(0.05)
        return flatfield

    # ---------- Step-scan autofocus with Gaussian fit ----------
    def _doAutofocusBackground(self, rangez=100, resolutionz=10, defocusz=0, axis=gAxis):
        self._commChannel.sigAutoFocusRunning.emit(True)
        mProcessor = FrameProcessor()

        if defocusz != 0:
            flatfieldImage = self.recordFlatfield(defocusPosition=defocusz, defocusAxis=axis)
            mProcessor.setFlatfieldFrame(flatfieldImage)

        initialPosition = float(self.stages.getPosition()[axis])

        Nz = int(max(5, np.floor((2 * abs(rangez)) / max(1e-6, abs(resolutionz))) + 1))
        relative_positions = np.linspace(-abs(rangez), abs(rangez), Nz).astype(float)

        self.stages.move(value=relative_positions[0], axis=axis, is_absolute=False, is_blocking=True)

        for iz in range(Nz):
            if not self.isAutofusRunning:
                break
            if iz != 0:
                step = relative_positions[iz] - relative_positions[iz - 1]
                self.stages.move(value=step, axis=axis, is_absolute=False, is_blocking=True)
            frame = self.grabCameraFrame()
            mProcessor.add_frame(frame, iz)

        allfocusvals = np.array(mProcessor.getFocusValueList(Nz))
        mProcessor.stop()

        if not self.isAutofusRunning:
            self.stages.move(value=initialPosition, axis=axis, is_absolute=True, is_blocking=True)
            self._commChannel.sigAutoFocusRunning.emit(False)
            self.isAutofusRunning = False
            if not IS_HEADLESS:
                self._widget.focusButton.setText('Autofocus')
            self.sigUpdateFocusValue.emit({"bestzpos": initialPosition})
            return initialPosition

        coordinates_abs = relative_positions + initialPosition

        try:
            if not IS_HEADLESS and hasattr(self._widget, "focusPlotCurve"):
                self._widget.focusPlotCurve.setData(coordinates_abs[:len(allfocusvals)], allfocusvals)
            else:
                self.sigUpdateFocusPlot.emit(coordinates_abs[:len(allfocusvals)], allfocusvals)
        except Exception:
            pass

        x0_fit, fit_y = _robust_gaussian_fit(coordinates_abs[:len(allfocusvals)], allfocusvals)

        try:
            if fit_y is not None and not IS_HEADLESS and hasattr(self._widget, "focusPlotFitCurve"):
                self._widget.focusPlotFitCurve.setData(coordinates_abs[:len(allfocusvals)], fit_y)
        except Exception:
            pass

        best_target = float(x0_fit)
        self.stages.move(value=best_target, axis=axis, is_absolute=True, is_blocking=True)

        final_z = best_target
        self._commChannel.sigAutoFocusRunning.emit(False)
        self.isAutofusRunning = False
        if not IS_HEADLESS:
            self._widget.focusButton.setText('Autofocus')
        self.sigUpdateFocusValue.emit({"bestzpos": final_z})
        return final_z

    # ---------- Continuous fast-sweep autofocus with time→Z mapping (no continuous Z readback) ----------
    def _doAutofocusFastBackground_timeMapped(self, sweep_range=150.0, speed=None, defocusz=0, axis=gAxis):
        self._commChannel.sigAutoFocusRunning.emit(True)

        # Optional flatfield
        flatfieldImage = None
        if defocusz != 0:
            flatfieldImage = self.recordFlatfield(defocusPosition=defocusz, defocusAxis=axis)

        # Setup sweep
        z0 = float(self.stages.getPosition()[axis])  # single read at start
        z_start = z0 + abs(sweep_range)
        z_end = z0 - abs(sweep_range)
        total_dist = float(z_end - z_start)  # negative for downward sweep

        # Move to start
        self.stages.move(value=z_start, axis=axis, is_absolute=True, is_blocking=True)
        time.sleep(0.005)

        # Launch continuous move to end
        self._moveController.move_to_position(z_end, axis=axis, speed=speed, is_absolute=True)
        t_start = time.time()

        # Accumulate timestamps and focus values; no Z readbacks here
        t_rel_list = []
        fvals = []
        last_fn = None

        while not self._moveController.is_target_reached() and self.isAutofusRunning:
            frame, fn = self.grabCameraFrame(returnFrameNumber=True)
            if frame is None:
                time.sleep(0.001)
                continue

            # dedupe by frame number if available
            if fn is not None:
                if last_fn is None:
                    last_fn = fn
                if fn <= last_fn:
                    time.sleep(0.0005)
                    continue
                last_fn = fn

            img = frame
            if flatfieldImage is not None:
                img = img / (flatfieldImage + 1e-12)

            img_proc = FrameProcessor.extract(img, min(img.shape[0], img.shape[1], 2048))
            if img_proc.ndim == 3:
                img_proc = np.mean(img_proc, axis=-1)

            f_measure = FrameProcessor.calculate_focus_measure_static(img_proc, method="LAPE")
            t_rel_list.append(time.time() - t_start)
            fvals.append(f_measure)

            time.sleep(0.0005)  # reduce CPU

        # Ensure motion fully completed; capture total time
        while not self._moveController.is_target_reached() and self.isAutofusRunning:
            time.sleep(0.001)
        t_end = time.time()
        total_time = max(1e-6, t_end - t_start)

        if not self.isAutofusRunning:
            # aborted: return to original position
            self.stages.move(value=z0, axis=axis, is_absolute=True, is_blocking=True)
            self._commChannel.sigAutoFocusRunning.emit(False)
            self.isAutofusRunning = False
            if not IS_HEADLESS:
                self._widget.focusButton.setText('Autofocus')
            self.sigUpdateFocusValue.emit({"bestzpos": z0})
            return z0

        # Map times -> Z using effective constant velocity
        v_eff = total_dist / total_time  # units of stage axis per second (likely µm/s)
        t_rel = np.asarray(t_rel_list, dtype=float)
        zs = z_start + v_eff * t_rel
        fvals = np.asarray(fvals, dtype=float)

        # Clip to [min(z_start,z_end), max(...)] to avoid small drift after finish
        zmin, zmax = (z_end, z_start) if z_end < z_start else (z_start, z_end)
        zs = np.clip(zs, zmin, zmax)

        # Plot raw
        try:
            if not IS_HEADLESS and hasattr(self._widget, "focusPlotCurve"):
                self._widget.focusPlotCurve.setData(zs, fvals)
            else:
                self.sigUpdateFocusPlot.emit(zs, fvals)
        except Exception:
            pass

        # Fit Gaussian
        if len(zs) >= 5:
            x0_fit, fit_y = _robust_gaussian_fit(zs, fvals)
        else:
            x0_fit, fit_y = (float(zs[np.argmax(fvals)]) if len(zs) else z0, None)

        # Optional fit plot
        try:
            if fit_y is not None and not IS_HEADLESS and hasattr(self._widget, "focusPlotFitCurve"):
                self._widget.focusPlotFitCurve.setData(zs, fit_y)
        except Exception:
            pass

        # Move to best focus (absolute)
        best_target = float(x0_fit)
        self.stages.move(value=best_target, axis=axis, is_absolute=True, is_blocking=True)

        final_z = best_target
        self._commChannel.sigAutoFocusRunning.emit(False)
        self.isAutofusRunning = False
        if not IS_HEADLESS:
            self._widget.focusButton.setText('Autofocus')
        self.sigUpdateFocusValue.emit({"bestzpos": final_z})
        return final_z


class FrameProcessor:
    def __init__(self, nGauss=7, nCropsize=2048):
        self.isRunning = True
        self.frame_queue = queue.Queue()
        self.allfocusvals = []
        self.worker_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.worker_thread.start()
        self.flatFieldFrame = None
        self.nGauss = nGauss
        self.nCropsize = nCropsize

    def setFlatfieldFrame(self, flatfieldFrame):
        self.flatFieldFrame = flatfieldFrame

    def add_frame(self, img, iz):
        self.frame_queue.put((img, iz))

    def process_frames(self):
        while self.isRunning:
            try:
                img, iz = self.frame_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            self.process_frame(img, iz)

    def process_frame(self, img, iz):
        if self.flatFieldFrame is not None:
            img = img / (self.flatFieldFrame + 1e-12)
        img = self.extract(img, self.nCropsize)
        if len(img.shape) > 2:
            img = np.mean(img, -1)
        focusquality = self.calculate_focus_measure(img)
        self.allfocusvals.append(focusquality)

    @staticmethod
    def calculate_focus_measure_static(image, method="LAPE"):
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if method == "LAPE":
            if image.dtype == np.uint16:
                lap = cv2.Laplacian(image, cv2.CV_32F)
            else:
                lap = cv2.Laplacian(image, cv2.CV_16S)
            return float(np.mean(np.square(lap)))
        elif method == "GLVA":
            return float(np.std(image, axis=None))
        else:
            return float(np.std(image, axis=None))

    def calculate_focus_measure(self, image, method="LAPE"):
        return self.calculate_focus_measure_static(image, method=method)

    @staticmethod
    def extract(marray, crop_size):
        h, w = marray.shape[0], marray.shape[1]
        cs = int(min(crop_size, h, w))
        center_x, center_y = w // 2, h // 2
        x_start = max(0, center_x - cs // 2)
        x_end = x_start + cs
        y_start = max(0, center_y - cs // 2)
        y_end = y_start + cs
        return marray[y_start:y_end, x_start:x_end]

    def getFocusValueList(self, nFrameExpected, timeout=5):
        t0 = time.time()
        while len(self.allfocusvals) < nFrameExpected:
            time.sleep(0.005)
            if time.time() - t0 > timeout:
                break
        return self.allfocusvals

    def stop(self):
        self.isRunning = False
        try:
            self.worker_thread.join(timeout=0.5)
        except Exception:
            pass


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <https://www.gnu.org/licenses/>.
