import io
import time
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple, List

import cv2
import numpy as np
import scipy.ndimage as ndi
from PIL import Image, ImageFile
from fastapi import Response
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter
from skimage.feature import peak_local_max

from imswitch.imcommon.framework import Thread, Signal  # noqa: F401 (Timer kept for context)
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import LiveUpdatedController
from imswitch import IS_HEADLESS

ImageFile.LOAD_TRUNCATED_IMAGES = True


class FocusLockController(LiveUpdatedController):
    """Linked to FocusLockWidget."""

    sigUpdateFocusValue = Signal()
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        if self._setupInfo.focusLock is None:
            return

        self.camera = self._setupInfo.focusLock.camera
        self.positioner = self._setupInfo.focusLock.positioner
        self.updateFreq = self._setupInfo.focusLock.updateFreq or 10
        
        # load parameters from setupInfo
        self.ki = self._setupInfo.focusLock.piKi
        self.kp = self._setupInfo.focusLock.piKp

        self.focusLockMetric = getattr(self._setupInfo.focusLock, "focusLockMetric", "JPG")
        self.cropCenter = getattr(self._setupInfo.focusLock, "cropCenter", None)
        self.cropSize = getattr(self._setupInfo.focusLock, "cropSize", None)

        # Initial Parameters for Focus Lock
        self.setPointSignal = 0.0
        self.locked = False
        self.aboutToLock = False
        self.zStackVar = False
        self.twoFociVar = False
        self.noStepVar = True
        self.focusTime = 1000 / self.updateFreq  # focus signal update interval (ms)
        self.zStepLimLo = 0.0
        self.aboutToLockDiffMax = 0.4
        self.lockPosition = 0.0
        self.currentPosition = 0.0
        self.lastPosition = 0.0
        self.buffer = 40
        self.currPoint = 0
        self.setPointData = np.zeros(self.buffer, dtype=float)
        self.timeData = np.zeros(self.buffer, dtype=float)
        self.gaussianSigma = 11.0
        self.backgroundThreshold = 40

        # Threads and Workers for focus lock
        try:
            self._master.detectorsManager[self.camera].startAcquisition()
        except Exception as e:
            self._logger.error(f"Failed to start acquisition on camera '{self.camera}': {e}")

        self.__processDataThread = ProcessDataThread(self)
        self.__focusCalibThread = FocusCalibThread(self)
        self.__processDataThread.setFocusLockMetric(self.focusLockMetric)

        # connect frame-update signal to function
        self._commChannel.sigUpdateImage.connect(self.update)

        # In case we run on QT, assign the widgets
        if IS_HEADLESS:
            return
        self._widget.setKp(self._setupInfo.focusLock.piKp)
        self._widget.setKi(self._setupInfo.focusLock.piKi)

        # Connect FocusLockWidget buttons
        self._widget.kpEdit.textChanged.connect(self.unlockFocus)
        self._widget.kiEdit.textChanged.connect(self.unlockFocus)

        self._widget.lockButton.clicked.connect(self.toggleFocus)
        self._widget.camDialogButton.clicked.connect(self.cameraDialog)
        self._widget.focusCalibButton.clicked.connect(self.focusCalibrationStart)
        self._widget.calibCurveButton.clicked.connect(self.showCalibrationCurve)

        self._widget.zStackBox.stateChanged.connect(self.zStackVarChange)
        self._widget.twoFociBox.stateChanged.connect(self.twoFociVarChange)

        self._widget.sigSliderExpTValueChanged.connect(self.setExposureTime)
        self._widget.sigSliderGainValueChanged.connect(self.setGain)

    def __del__(self):
        try:
            self.__processDataThread.quit()
            self.__processDataThread.wait()
        except Exception:
            pass
        try:
            self.__focusCalibThread.quit()
            self.__focusCalibThread.wait()
        except Exception:
            pass
        try:
            if hasattr(self, "_master") and hasattr(self, "camera"):
                self._master.detectorsManager[self.camera].stopAcquisition()
        except Exception:
            pass
        try:
            if hasattr(self, "ESP32Camera"):
                self.ESP32Camera.stopStreaming()
        except Exception:
            pass
        if hasattr(super(), "__del__"):
            try:
                super().__del__()
            except Exception:
                pass

    @APIExport(runOnUIThread=True)
    def unlockFocus(self):
        if self.locked:
            self.locked = False
            if not IS_HEADLESS:
                self._widget.lockButton.setChecked(False)
                try:
                    self._widget.focusPlot.removeItem(self._widget.focusLockGraph.lineLock)
                except Exception:
                    pass

    @APIExport(runOnUIThread=True)  
    def toggleFocus(self, toLock:bool=None):
        self.aboutToLock = False
        if (not IS_HEADLESS and self._widget.lockButton.isChecked()) or (toLock is not None and toLock and self.locked is False):
            zpos = self._master.positionersManager[self.positioner].get_abs()
            self.lockFocus(zpos)
            if not IS_HEADLESS: self._widget.lockButton.setText("Unlock")
        else:
            self.unlockFocus()
            if not IS_HEADLESS:  self._widget.lockButton.setText("Lock")

    def cameraDialog(self):
        try:
            self._master.detectorsManager[self.camera].openPropertiesDialog()
        except Exception as e:
            self._logger.error(f"Failed to open camera dialog: {e}")

    @APIExport(runOnUIThread=True)
    def focusCalibrationStart(self):
        self.__focusCalibThread.start()

    def showCalibrationCurve(self):
        if not IS_HEADLESS:
            self._widget.showCalibrationCurve(self.__focusCalibThread.getData())

    def zStackVarChange(self):
        self.zStackVar = not self.zStackVar

    def twoFociVarChange(self):
        self.twoFociVar = not self.twoFociVar

    def update(self, detectorName, im, _init, _scale, _isCurrentDetector):
        # get data
        if detectorName != self.camera:
            return
        self.setPointSignal = self.__processDataThread.update(im, self.twoFociVar)
        self.sigUpdateFocusValue.emit((self.setPointSignal, time.time()))
        # move
        if self.locked:
            value_move = self.updatePI()
            if self.noStepVar and abs(value_move) > 0.002:
                self._master.positionersManager[self.positioner].move(value_move, 0)
        elif self.aboutToLock:
            if not hasattr(self, "aboutToLockDataPoints"):
                self.aboutToLockDataPoints = np.zeros(5, dtype=float)
            self.aboutToLockUpdate()

        # update graphics
        self.updateSetPointData()
        if IS_HEADLESS:
            return
        try:
            self._widget.camImg.setImage(im)
            if self.currPoint < self.buffer:
                self._widget.focusPlotCurve.setData(
                    self.timeData[1:self.currPoint],
                    self.setPointData[1:self.currPoint],
                )
            else:
                self._widget.focusPlotCurve.setData(self.timeData, self.setPointData)
        except Exception:
            pass

    @APIExport(runOnUIThread=True)
    def setParamsAstigmatism(
        self,
        gaussianSigma: float,
        backgroundThreshold: float,
        cropSize: int,
        cropCenter: Optional[List[int]] = None,
    ):
        """Set parameters for astigmatism focus metric."""
        self.gaussianSigma = float(gaussianSigma)
        self.backgroundThreshold = float(backgroundThreshold)
        self.cropSize = int(cropSize)
        if cropCenter is None:
            cropCenter = [self.cropSize // 2, self.cropSize // 2]
        self.cropCenter = np.asarray(cropCenter, dtype=int)

    @APIExport(runOnUIThread=True)
    def getParamsAstigmatism(self):
        """Get parameters for astigmatism focus metric."""
        return {
            "gaussianSigma": self.gaussianSigma,
            "backgroundThreshold": self.backgroundThreshold,
            "cropSize": self.cropSize,
            "cropCenter": self.cropCenter,
        }

    def aboutToLockUpdate(self):
        self.aboutToLockDataPoints = np.roll(self.aboutToLockDataPoints, 1)
        self.aboutToLockDataPoints[0] = float(self.setPointSignal)
        averageDiff = float(np.std(self.aboutToLockDataPoints))
        if averageDiff < self.aboutToLockDiffMax:
            zpos = self._master.positionersManager[self.positioner].get_abs()
            self.lockFocus(zpos)
            self.aboutToLock = False

    def updateSetPointData(self):
        if self.currPoint < self.buffer:
            self.setPointData[self.currPoint] = self.setPointSignal
            self.timeData[self.currPoint] = 0.0  # placeholder for potential timing
        else:
            self.setPointData = np.roll(self.setPointData, -1)
            self.setPointData[-1] = self.setPointSignal
            self.timeData = np.roll(self.timeData, -1)
            self.timeData[-1] = 0.0
        self.currPoint += 1

    @APIExport(runOnUIThread=True)
    def setPIParameters(self, kp: float, ki: float):
        """Set parameters for the PI controller."""
        if not hasattr(self, "pi"):
            self.pi = PI(self.setPointSignal, 0.001, kp,  ki)
            self.ki = ki
            self.kp = kp
        else:
            self.pi.setParameters(kp, ki)
        if not IS_HEADLESS:
            self._widget.setKp(kp)
            self._widget.setKi(ki)
         
    @APIExport(runOnUIThread=True)
    def getPIParameters(self) -> Tuple[float, float, float]:
        """Get parameters for the PI controller."""
        if hasattr(self, "pi"):
            return self.pi.kp, self.pi.ki
        return self.kp, self.ki
       
    def updatePI(self):
        if not self.noStepVar:
            self.noStepVar = True
        self.currentPosition = float(self._master.positionersManager[self.positioner].get_abs())
        self.stepDistance = abs(self.currentPosition - self.lastPosition)
        distance = self.currentPosition - self.lockPosition
        move = self.pi.update(self.setPointSignal)
        self.lastPosition = self.currentPosition

        if abs(distance) > 5 or abs(move) > 3: # TODO: make this a setting adaptive by GUI/API
            self._logger.warning(
                f"Safety unlocking! Distance to lock: {distance:.3f}, current move step: {move:.3f}."
            ) # TODO: If this happens, shoot a signal to the GUI/API via signal
            self.unlockFocus()
        elif self.zStackVar:
            if self.stepDistance > self.zStepLimLo:
                self.unlockFocus()
                self.aboutToLockDataPoints = np.zeros(5, dtype=float)
                self.aboutToLock = True
                self.noStepVar = False
        return move

    def lockFocus(self, zpos):
        if not self.locked:
            if IS_HEADLESS:
                kp, ki = self.kp, self.ki
            else:
                kp = float(self._widget.kpEdit.text())
                ki = float(self._widget.kiEdit.text())
            self.pi = PI(self.setPointSignal, kp, ki)
            self.lockPosition = float(zpos)
            self.locked = True
            if not IS_HEADLESS:
                try:
                    self._widget.focusLockGraph.lineLock = self._widget.focusPlot.addLine(
                        y=self.setPointSignal, pen="r"
                    )
                    self._widget.lockButton.setChecked(True)
                except Exception:
                    pass
            self.updateZStepLimits()

    def updateZStepLimits(self):
        try:
            self.zStepLimLo = 0.001 * float(self._widget.zStepFromEdit.text())
        except Exception:
            self.zStepLimLo = 0.0 # TODO: Make this a setting in the setupInfo or the GUI/API

    @APIExport(runOnUIThread=True)
    def returnLastCroppedImage(self) -> Response:
        """Returns the last cropped image from the camera."""
        try:
            arr = self.__processDataThread.getCroppedImage()
            im = Image.fromarray(arr.astype(np.uint8))
            with io.BytesIO() as buf:
                im = im.convert("L")  # ensure grayscale
                im.save(buf, format="PNG")
                im_bytes = buf.getvalue()
            headers = {"Content-Disposition": 'inline; filename="crop.png"'}
            return Response(im_bytes, headers=headers, media_type="image/png")
        except Exception as e:
            raise RuntimeError("No cropped image available. Please run update() first.") from e

    @APIExport(runOnUIThread=True)
    def returnLastImage(self) -> Response:
        lastFrame = self._master.detectorsManager[self.camera].getLatestFrame()
        if lastFrame is None:
            raise RuntimeError("No image available. Please run update() first.")
        try:
            im = Image.fromarray(lastFrame.astype(np.uint8))
            with io.BytesIO() as buf:
                im.save(buf, format="PNG")
                im_bytes = buf.getvalue()
            headers = {"Content-Disposition": 'inline; filename="last_image.png"'}
            return Response(im_bytes, headers=headers, media_type="image/png")
        except Exception as e:
            raise RuntimeError("Failed to convert last image to PNG.") from e

    @APIExport(runOnUIThread=True)
    def setCropFrameParameters(self, cropSize: int, cropCenter: Optional[List[int]] = None):
        """Set the crop frame parameters for the camera."""
        self.__processDataThread.setCropFrameParameters(cropSize, cropCenter)


class ProcessDataThread(Thread):
    def __init__(self, controller, *args, **kwargs):
        self._controller = controller
        super().__init__(*args, **kwargs)
        self.focusLockMetric: Optional[str] = None
        self.cropSize: Optional[int] = None
        self.cropCenter: Optional[np.ndarray] = None

    def setFocusLockMetric(self, focuslockMetric: str):
        self.focusLockMetric = focuslockMetric

    def getCroppedImage(self) -> np.ndarray:
        """Returns the last processed (cropped) image array."""
        if hasattr(self, "imagearraygf"):
            return self.imagearraygf
        raise RuntimeError("No image processed yet. Please run update() first.")

    @staticmethod
    def extract(marray: np.ndarray, crop_size: Optional[int] = None, crop_center: Optional[List[int]] = None) -> np.ndarray:
        h, w = marray.shape[:2]
        if crop_center is None:
            center_x, center_y = w // 2, h // 2
        else:
            center_x, center_y = int(crop_center[0]), int(crop_center[1])

        if crop_size is None:
            crop_size = min(h, w) // 2
        crop_size = int(crop_size)

        half = crop_size // 2
        x_start = max(0, center_x - half)
        y_start = max(0, center_y - half)
        x_end = min(w, x_start + crop_size)
        y_end = min(h, y_start + crop_size)

        # Adjust starts if we hit a boundary on the end
        x_start = max(0, x_end - crop_size)
        y_start = max(0, y_end - crop_size)

        return marray[y_start:y_end, x_start:x_end]

    def _jpeg_size_metric(self, img: np.ndarray) -> int:
        # Ensure uint8 grayscale for JPEG encode
        if img.dtype != np.uint8:
            img_u8 = np.clip(img, 0, 255).astype(np.uint8)
        else:
            img_u8 = img
        success, buffer = cv2.imencode(".jpg", img_u8, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
        if not success:
            self._controller._logger.warning("Failed to encode image to JPEG.")
            return 0
        return int(len(buffer))

    def update(self, latestImg: np.ndarray, twoFociVar: bool) -> float:
        if self.focusLockMetric == "JPG":
            self.imagearraygf = self.extract(
                latestImg,
                crop_center=self._controller.cropCenter,
                crop_size=self._controller.cropSize,
            )
            focusMetricGlobal = float(self._jpeg_size_metric(self.imagearraygf))
        elif self.focusLockMetric == "astigmatism":
            config = FocusConfig(
                gaussian_sigma=float(self._controller.gaussianSigma),
                background_threshold=int(self._controller.backgroundThreshold),
                crop_radius=int(self._controller.cropSize or 300),
                enable_gaussian_blur=True,
            )
            focus_metric = FocusMetric(config)
            self.imagearraygf = self.extract(
                latestImg,
                crop_center=self._controller.cropCenter,
                crop_size=self._controller.cropSize,
            )
            result = focus_metric.compute(self.imagearraygf)
            focusMetricGlobal = float(result["focus"])
            self._controller._logger.debug(
                f"Focus computation result: {result}, Focus value: {result['focus']:.4f}, Timestamp: {result['t']}"
            )
        else:
            # Gaussian filter to remove noise for better center estimate
            self.imagearraygf = gaussian_filter(latestImg.astype(float), 7)

            # Update the focus signal
            if twoFociVar:
                allmaxcoords = peak_local_max(self.imagearraygf, min_distance=60)
                size = allmaxcoords.shape[0]
                if size >= 2:
                    maxvals = np.full(2, -np.inf)
                    maxvalpos = np.zeros(2, dtype=int)
                    for n in range(size):
                        val = self.imagearraygf[allmaxcoords[n][0], allmaxcoords[n][1]]
                        if val > maxvals[0]:
                            if val > maxvals[1]:
                                maxvals[0] = maxvals[1]
                                maxvals[1] = val
                                maxvalpos[0] = maxvalpos[1]
                                maxvalpos[1] = n
                            else:
                                maxvals[0] = val
                                maxvalpos[0] = n
                    xcenter = allmaxcoords[maxvalpos[0]][0]
                    ycenter = allmaxcoords[maxvalpos[0]][1]
                    if allmaxcoords[maxvalpos[1]][1] < ycenter:
                        xcenter = allmaxcoords[maxvalpos[1]][0]
                        ycenter = allmaxcoords[maxvalpos[1]][1]
                    centercoords2 = np.array([xcenter, ycenter])
                else:
                    # Fallback to global max if not enough peaks
                    centercoords = np.where(self.imagearraygf == np.max(self.imagearraygf))
                    centercoords2 = np.array([centercoords[0][0], centercoords[1][0]])
            else:
                centercoords = np.where(self.imagearraygf == np.max(self.imagearraygf))
                centercoords2 = np.array([centercoords[0][0], centercoords[1][0]])

            subsizey = 50
            subsizex = 50
            h, w = self.imagearraygf.shape[:2]
            xlow = max(0, int(centercoords2[0] - subsizex))
            xhigh = min(h, int(centercoords2[0] + subsizex))
            ylow = max(0, int(centercoords2[1] - subsizey))
            yhigh = min(w, int(centercoords2[1] + subsizey))

            self.imagearraygfsub = self.imagearraygf[xlow:xhigh, ylow:yhigh]
            massCenter = np.array(ndi.center_of_mass(self.imagearraygfsub))
            # Add the information about where the center of the subarray is
            focusMetricGlobal = float(massCenter[1] + centercoords2[1])

        return focusMetricGlobal

    def setCropFrameParameters(self, cropSize: int, cropCenter: Optional[List[int]] = None):
        """Set the crop frame parameters for the camera."""
        self.cropSize = int(cropSize)
        detectorSize = self._controller._master.detectorsManager[self._controller.camera].getDetectorSize()
        if self.cropSize > detectorSize[0] or self.cropSize > detectorSize[1]:
            raise ValueError(f"Crop size {self.cropSize} exceeds detector size {detectorSize}.")
        if cropCenter is None:
            cropCenter = [self.cropSize // 2, self.cropSize // 2]
        self.cropCenter = np.asarray(cropCenter, dtype=int)


class FocusCalibThread(Thread):
    def __init__(self, controller, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._controller = controller

    def run(self):
        self.signalData: List[float] = []
        self.positionData: List[float] = []
        self.fromVal = float(self._controller._widget.calibFromEdit.text())
        self.toVal = float(self._controller._widget.calibToEdit.text())
        self.scan_list = np.round(np.linspace(self.fromVal, self.toVal, 20), 2)
        for z in self.scan_list:
            self._controller._master.positionersManager[self._controller.positioner].setPosition(z, 0)
            time.sleep(0.5)
            self.focusCalibSignal = float(self._controller.setPointSignal)
            self.signalData.append(self.focusCalibSignal)
            self.positionData.append(
                float(self._controller._master.positionersManager[self._controller.positioner].get_abs())
            )
        self.poly = np.polyfit(self.positionData, self.signalData, 1)
        self.calibrationResult = np.around(self.poly, 4)
        self.show()

    def show(self):
        if self.poly[0] == 0:
            calText = "Calibration invalid"
        else:
            cal_nm = np.round(1000 / self.poly[0], 1)
            calText = f"1 px --> {cal_nm} nm"
        self._controller._widget.calibrationDisplay.setText(calText)

    def getData(self):
        return {
            "signalData": self.signalData,
            "positionData": self.positionData,
            "poly": self.poly,
        }


class PI:
    """Simple implementation of a discrete PI controller.
    Taken from http://code.activestate.com/recipes/577231-discrete-pid-controller/
    Author: Federico Barabas
    """

    def __init__(self, setPoint: float, kp: float = 0.0, ki: float = 0.0):
        self._kp = kp
        self._ki = ki
        self._setPoint = float(setPoint)
        self.error = 0.0
        self._started = False
        self.out = 0.0
        self.lastError = 0.0

    def setParameters(self, kp: float, ki: float):
        self.kp = kp
        self.ki = ki

    def update(self, currentValue: float) -> float:
        """Calculate PI output value for given reference input and feedback.
        Using the iterative formula to avoid integrative part building."""
        self.error = self.setPoint - float(currentValue)
        if self.started:
            self.dError = self.error - self.lastError
            self.out = self.out + self.kp * self.dError + self.ki * self.error
        else:
            self.out = self.kp * self.error
            self.started = True
        self.lastError = self.error
        return self.out

    def restart(self):
        self.started = False
        self.out = 0.0
        self.lastError = 0.0

    @property
    def started(self) -> bool:
        return self._started

    @started.setter
    def started(self, value: bool):
        self._started = bool(value)

    @property
    def setPoint(self) -> float:
        return self._setPoint

    @setPoint.setter
    def setPoint(self, value: float):
        self._setPoint = float(value)

    @property
    def kp(self) -> float:
        return self._kp

    @kp.setter
    def kp(self, value: float):
        self._kp = float(value)

    @property
    def ki(self) -> float:
        return self._ki

    @ki.setter
    def ki(self, value: float):
        self._ki = float(value)


"""
Focus Metric Algorithm Implementation

Based on the specification in section 5:
1. Convert frame to grayscale (numpy uint8)
2. Optional Gaussian blur σ ≈ 11 px to suppress noise
3. Threshold: im[im < background] = 0, background configurable
4. Compute mean projections projX, projY
5. Fit projX with double-Gaussian, projY with single-Gaussian (SciPy curve_fit)
6. Focus value F = σx / σy (float32)
7. Return timestamped JSON {"t": timestamp, "focus": F}
"""


@dataclass
class FocusConfig:
    """Configuration for focus metric computation"""
    gaussian_sigma: float = 11.0  # Gaussian blur sigma
    background_threshold: int = 40  # Background threshold value
    crop_radius: int = 300  # Radius for cropping around max intensity
    enable_gaussian_blur: bool = True  # Enable/disable Gaussian preprocessing


class FocusMetric:
    """Focus metric computation using double/single Gaussian fitting"""

    def __init__(self, config: Optional[FocusConfig] = None):
        self.config = config or FocusConfig()

    @staticmethod
    def gaussian_1d(xdata: np.ndarray, i0: float, x0: float, sigma: float, amp: float) -> np.ndarray:
        """Single Gaussian model function"""
        x = xdata
        x0 = float(x0)
        return i0 + amp * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))

    @staticmethod
    def double_gaussian_1d(
        xdata: np.ndarray, i0: float, x0: float, sigma: float, amp: float, dist: float
    ) -> np.ndarray:
        """Double Gaussian model function"""
        x = xdata
        x0 = float(x0)
        return (
            i0
            + amp * np.exp(-((x - (x0 - dist / 2)) ** 2) / (2 * sigma ** 2))
            + amp * np.exp(-((x - (x0 + dist / 2)) ** 2) / (2 * sigma ** 2))
        )

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame according to specification steps 1-3

        Args:
            frame: Input frame (can be RGB or grayscale)

        Returns:
            Preprocessed grayscale frame
        """
        # Step 1: Convert to grayscale if needed
        if frame.ndim == 3:
            im = np.mean(frame, axis=-1).astype(np.uint8)
        else:
            im = frame.astype(np.uint8)

        # Convert to float for processing
        im = im.astype(float)

        # Find maximum intensity location for cropping
        if self.config.crop_radius > 0:
            # Apply heavy Gaussian blur to find general maximum location
            im_gauss = gaussian_filter(im, sigma=111)
            max_coord = np.unravel_index(np.argmax(im_gauss), im_gauss.shape)

            # Crop around maximum with specified radius
            h, w = im.shape
            y_min = max(0, max_coord[0] - self.config.crop_radius)
            y_max = min(h, max_coord[0] + self.config.crop_radius)
            x_min = max(0, max_coord[1] - self.config.crop_radius)
            x_max = min(w, max_coord[1] + self.config.crop_radius)

            im = im[y_min:y_max, x_min:x_max]

        # Step 2: Optional Gaussian blur to suppress noise
        if self.config.enable_gaussian_blur:
            im = gaussian_filter(im, sigma=self.config.gaussian_sigma)

        # Mean subtraction
        im = im - np.mean(im) / 2.0

        # Step 3: Threshold background
        im[im < self.config.background_threshold] = 0

        return im

    def preprocess_frame_rainer(self, frame: np.ndarray) -> np.ndarray:
        """
        Alternate preprocessor (kept for context). Implemented as a thin wrapper
        around `preprocess_frame` to avoid undefined references in the original.
        """
        return self.preprocess_frame(frame)

    def compute_projections(self, im: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Step 4: Compute mean projections projX, projY

        Args:
            im: Preprocessed image

        Returns:
            (projX, projY) - mean projections along y and x axes
        """
        projX = np.mean(im, axis=0)  # Project along y-axis
        projY = np.mean(im, axis=1)  # Project along x-axis
        return projX, projY

    def fit_projections(
        self, projX: np.ndarray, projY: np.ndarray, isDoubleGaussX: bool = False
    ) -> Tuple[float, float]:
        """
        Steps 5-6: Fit projections and compute focus value

        Args:
            projX: X projection (fit with double-Gaussian if requested)
            projY: Y projection (fit with single-Gaussian)

        Returns:
            (sigma_x, sigma_y) - fitted standard deviations
        """
        h1, w1 = len(projY), len(projX)
        x = np.arange(w1)
        y = np.arange(h1)

        # Initial guess parameters
        i0_x = float(np.mean(projX))
        amp_x = float(np.max(projX) - i0_x)
        sigma_x_init = float(np.std(projX))
        i0_y = float(np.mean(projY))
        amp_y = float(np.max(projY) - i0_y)
        sigma_y_init = float(np.std(projY))

        if isDoubleGaussX:
            init_guess_x = [i0_x, w1 / 2, sigma_x_init, amp_x, 100.0]
        else:
            init_guess_x = [i0_x, w1 / 2, sigma_x_init, amp_x]
        init_guess_y = [i0_y, h1 / 2, sigma_y_init, amp_y]

        try:
            if isDoubleGaussX:
                popt_x, _ = curve_fit(self.double_gaussian_1d, x, projX, p0=init_guess_x, maxfev=50000)
                sigma_x = abs(float(popt_x[2]))
            else:
                popt_x, _ = curve_fit(self.gaussian_1d, x, projX, p0=init_guess_x, maxfev=50000)
                sigma_x = abs(float(popt_x[2]))

            popt_y, _ = curve_fit(self.gaussian_1d, y, projY, p0=init_guess_y, maxfev=50000)
            sigma_y = abs(float(popt_y[2]))
        except Exception:
            # Fallback to standard deviation if fitting fails
            sigma_x = float(np.std(projX))
            sigma_y = float(np.std(projY))

        return sigma_x, sigma_y

    def compute(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Main computation method - implements complete focus metric algorithm

        Args:
            frame: Input camera frame (RGB or grayscale)

        Returns:
            Timestamped JSON with focus value: {"t": timestamp, "focus": focus_value}
        """
        timestamp = time.time()

        try:
            im = self.preprocess_frame(frame)
            projX, projY = self.compute_projections(im)
            sigma_x, sigma_y = self.fit_projections(projX, projY)
            focus_value = float("inf") if sigma_y == 0 else float(sigma_x / sigma_y)
        except Exception:
            focus_value = float("nan")

        return {"t": timestamp, "focus": focus_value}

    def update_config(self, **kwargs) -> None:
        """Update configuration parameters"""
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")


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
