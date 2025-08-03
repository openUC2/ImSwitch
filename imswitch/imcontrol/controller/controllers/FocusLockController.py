import time
import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage.feature import peak_local_max
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from dataclasses import dataclass
from typing import Optional, Dict, Any
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from fastapi import Response
from imswitch.imcommon.framework import Thread, Timer
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import LiveUpdatedController
import io
from imswitch import IS_HEADLESS

class FocusLockController(LiveUpdatedController):
    """Linked to FocusLockWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)

        if self._setupInfo.focusLock is None:
            return

        self.camera = self._setupInfo.focusLock.camera
        self.positioner = self._setupInfo.focusLock.positioner
        self.updateFreq = self._setupInfo.focusLock.updateFreq
        if self.updateFreq is None:
            self.updateFreq = 10

        try:
            self.focusLockMetric = self._setupInfo.focusLock.focusLockMetric
        except:
            self.focusLockMetric = "JPG"

        try:
            self.cropCenter = self._setupInfo.focusLock.cropCenter
        except:
            self.cropCenter = None
        try:
            self.cropSize = self._setupInfo.focusLock.cropSize
        except:
            self.cropSize = None

        # Initial Parameters for Focus Lock
        self.setPointSignal = 0
        self.locked = False
        self.aboutToLock = False
        self.zStackVar = False
        self.twoFociVar = False
        self.noStepVar = True
        self.focusTime = 1000 / self.updateFreq  # focus signal update interval (ms)
        self.zStepLimLo = 0
        self.aboutToLockDiffMax = 0.4
        self.lockPosition = 0
        self.currentPosition = 0
        self.lastPosition = 0
        self.buffer = 40
        self.currPoint = 0
        self.setPointData = np.zeros(self.buffer)
        self.timeData = np.zeros(self.buffer)
        self.gaussianSigma = 11
        self.backgroundThreshold = 40
        # Threads and Workers for focus lock
        self._master.detectorsManager[self.camera].startAcquisition()
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
        self.__processDataThread.quit()
        self.__processDataThread.wait()
        self.__focusCalibThread.quit()
        self.__focusCalibThread.wait()
        self.ESP32Camera.stopStreaming()
        if hasattr(super(), '__del__'):
            super().__del__()

    @APIExport(runOnUIThread=True)
    def unlockFocus(self):
        if self.locked:
            self.locked = False
            self._widget.lockButton.setChecked(False)
            self._widget.focusPlot.removeItem(self._widget.focusLockGraph.lineLock)

    @APIExport(runOnUIThread=True)
    def toggleFocus(self):
        self.aboutToLock = False
        if self._widget.lockButton.isChecked():
            zpos = self._master.positionersManager[self.positioner].get_abs()
            self.lockFocus(zpos)
            self._widget.lockButton.setText('Unlock')
        else:
            self.unlockFocus()
            self._widget.lockButton.setText('Lock')

    def cameraDialog(self):
        self._master.detectorsManager[self.camera].openPropertiesDialog()

    @APIExport(runOnUIThread=True)
    def focusCalibrationStart(self):
        self.__focusCalibThread.start()

    def showCalibrationCurve(self):
        self._widget.showCalibrationCurve(self.__focusCalibThread.getData())

    def zStackVarChange(self):
        if self.zStackVar:
            self.zStackVar = False
        else:
            self.zStackVar = True

    def twoFociVarChange(self):
        if self.twoFociVar:
            self.twoFociVar = False
        else:
            self.twoFociVar = True

    def update(self, detectorName, im, init, scale, isCurrentDetector):
        # get data
        if detectorName != self.camera: return
        self.setPointSignal = self.__processDataThread.update(im, self.twoFociVar)
        # move
        if self.locked:
            value_move = self.updatePI()
            if self.noStepVar and abs(value_move) > 0.002:
                self._master.positionersManager[self.positioner].move(value_move, 0)
        elif self.aboutToLock:
           self.aboutToLockUpdate()
        # udpate graphics
        self.updateSetPointData()
        if IS_HEADLESS: 
            return
        self._widget.camImg.setImage(im)
        if self.currPoint < self.buffer:
            self._widget.focusPlotCurve.setData(self.timeData[1:self.currPoint],
                                                self.setPointData[1:self.currPoint])
        else:
            self._widget.focusPlotCurve.setData(self.timeData, self.setPointData)

    def setParamsAstigmatism(self, gaussianSigma: float, backgroundThreshold: float, cropSize: int, cropCenter: Optional[list] = None):
        """ Set parameters for astigmatism focus metric. """
        self.gaussianSigma = gaussianSigma
        self.backgroundThreshold = backgroundThreshold
        self.cropSize = cropSize
        if cropCenter is None:
            cropCenter = np.array([cropSize // 2, cropSize // 2])
        self.cropCenter = cropCenter
        
    def getParamsAstigmatism(self):
        """ Get parameters for astigmatism focus metric. """
        return {
            "gaussianSigma": self.gaussianSigma,
            "backgroundThreshold": self.backgroundThreshold,
            "cropSize": self.cropSize,
            "cropCenter": self.cropCenter
        }

    def aboutToLockUpdate(self):
        self.aboutToLockDataPoints = np.roll(self.aboutToLockDataPoints,1)
        self.aboutToLockDataPoints[0] = self.setPointSignal
        averageDiff = np.std(self.aboutToLockDataPoints)
        if averageDiff < self.aboutToLockDiffMax:
            zpos = self._master.positionersManager[self.positioner].get_abs()
            self.lockFocus(zpos)
            self.aboutToLock = False

    def updateSetPointData(self):
        if self.currPoint < self.buffer:
            self.setPointData[self.currPoint] = self.setPointSignal
            self.timeData[self.currPoint] =  0 # perf_counter() - self.startTime
        else:
            self.setPointData = np.roll(self.setPointData, -1)
            self.setPointData[-1] = self.setPointSignal
            self.timeData = np.roll(self.timeData, -1)
            self.timeData[-1] = 0 # perf_counter() - self.startTime
        self.currPoint += 1

    def updatePI(self):
        if not self.noStepVar:
            self.noStepVar = True
        self.currentPosition = self._master.positionersManager[self.positioner].get_abs()
        self.stepDistance = np.abs(self.currentPosition - self.lastPosition)
        distance = self.currentPosition - self.lockPosition
        move = self.pi.update(self.setPointSignal)
        self.lastPosition = self.currentPosition

        if abs(distance) > 5 or abs(move) > 3:
            self._logger.warning(f'Safety unlocking! Distance to lock: {distance:.3f}, current move step: {move:.3f}.')
            self.unlockFocus()
        elif self.zStackVar:
            if self.stepDistance > self.zStepLimLo:
                self.unlockFocus()
                self.aboutToLockDataPoints = np.zeros(5)
                self.aboutToLock = True
                self.noStepVar = False
        return move

    def lockFocus(self, zpos):
        if not self.locked:
            kp = float(self._widget.kpEdit.text())
            ki = float(self._widget.kiEdit.text())
            self.pi = PI(self.setPointSignal, 0.001, kp, ki)
            self.lockPosition = zpos
            self.locked = True
            self._widget.focusLockGraph.lineLock = self._widget.focusPlot.addLine(
                y=self.setPointSignal, pen='r'
            )
            self._widget.lockButton.setChecked(True)
            self.updateZStepLimits()

    def updateZStepLimits(self):
        self.zStepLimLo = 0.001 * float(self._widget.zStepFromEdit.text())

    @APIExport(runOnUIThread=True)
    def returnLastCroppedImage(self) -> Response:
        """ Returns the last cropped image from the camera. """
        try:

            # using an in-memory image
            im = Image.fromarray(self.__processDataThread.getCroppedImage())

            # save image to an in-memory bytes buffer
            # save image to an in-memory bytes buffer
            with io.BytesIO() as buf:
                im = im.convert('L')  # convert image to 'L' mode
                im.save(buf, format='PNG')
                im_bytes = buf.getvalue()

            headers = {'Content-Disposition': 'inline; filename="test.png"'}
            return Response(im_bytes, headers=headers, media_type='image/png')

        except Exception as e:
            raise RuntimeError('No cropped image available. Please run update() first.')

    @APIExport(runOnUIThread=True)
    def setCropFrameParameters(self, cropSize: int, cropCenter: Optional[list] = None):
        """ Set the crop frame parameters for the camera. """
        self.__processDataThread.setCropFrameParameters(cropSize, cropCenter)


class ProcessDataThread(Thread):
    def __init__(self, controller, *args, **kwargs):
        self._controller = controller
        super().__init__(*args, **kwargs)
        self.focusLockMetric = None

    def setFocusLockMetric(self, focuslockMetric):
        self.focusLockMetric = focuslockMetric

    def getCroppedImage(self):
        """ Returns the last processed image array. """
        if hasattr(self, 'imagearraygf'):
            return self.imagearraygf
        else:
            raise RuntimeError('No image processed yet. Please run update() first.')
    
    @staticmethod
    def extract(marray, crop_size=None, crop_center=None):
        if crop_center is None:
            center_x, center_y = marray.shape[1] // 2, marray.shape[0] // 2
        else:
            center_x, center_y = crop_center

        if crop_size is None:
            crop_size = np.min(marray.shape)//2
        # Calculate the starting and ending indices for cropping
        x_start = center_x - crop_size // 2
        x_end = x_start + crop_size
        y_start = center_y - crop_size // 2
        y_end = y_start + crop_size

        # Crop the center region
        return marray[y_start:y_end, x_start:x_end]
    
    def update(self, latestImg, twoFociVar):
        if self.focusLockMetric == "JPG":
            self.imagearraygf = self.extract(latestImg,  crop_center=self._controller.cropCenter, crop_size=self._controller.cropSize)
            is_success, buffer = cv2.imencode(".jpg", self.imagearraygf, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            # Check if encoding was successful
            if is_success:
                # Get the size of the JPEG image
                focusquality = len(buffer)
            else:
                focusquality = 0
                print("Failed to encode image")
            focusMetricGlobal = focusquality
        elif self.focusLockMetric == "astigmatism":
            # Create focus metric instance
            # TODO: We should have this globally accessible, not created every time
            # TODO: We should expose the focus metric configuration in the setup JSON and make it adjustable via the REST API
            config = FocusConfig(gaussian_sigma=self._controller.gaussianSigma,
                                    background_threshold=self._controller.backgroundThreshold,
                                    crop_radius=self._controller.cropSize,
                                    enable_gaussian_blur=True)
    
            
            focus_metric = FocusMetric(config)
            
            # Compute focus metric
            self.imagearraygf = self.extract(latestImg, crop_center=self._controller.cropCenter, crop_size=self._controller.cropSize)
            result = focus_metric.compute(self.imagearraygf)
            focusMetricGlobal = result['focus']
            print(f"Focus computation result: {result}, Focus value: {result['focus']:.4f}, Timestamp: {result['t']}")
        else:
            # Gaussian filter the image, to remove noise and so on, to get a better center estimate
            self.imagearraygf = ndi.filters.gaussian_filter(latestImg, 7)

            # Update the focus signal
            if twoFociVar:
                    allmaxcoords = peak_local_max(self.imagearraygf, min_distance=60)
                    size = allmaxcoords.shape
                    maxvals = np.zeros(size[0])
                    maxvalpos = np.zeros(2)
                    for n in range(0, size[0]):
                        if self.imagearraygf[allmaxcoords[n][0], allmaxcoords[n][1]] > maxvals[0]:
                            if self.imagearraygf[allmaxcoords[n][0], allmaxcoords[n][1]] > maxvals[1]:
                                tempval = maxvals[1]
                                maxvals[0] = tempval
                                maxvals[1] = self.imagearraygf[allmaxcoords[n][0], allmaxcoords[n][1]]
                                tempval = maxvalpos[1]
                                maxvalpos[0] = tempval
                                maxvalpos[1] = n
                            else:
                                maxvals[0] = self.imagearraygf[allmaxcoords[n][0], allmaxcoords[n][1]]
                                maxvalpos[0] = n
                    xcenter = allmaxcoords[maxvalpos[0]][0]
                    ycenter = allmaxcoords[maxvalpos[0]][1]
                    if allmaxcoords[maxvalpos[1]][1] < ycenter:
                        xcenter = allmaxcoords[maxvalpos[1]][0]
                        ycenter = allmaxcoords[maxvalpos[1]][1]
                    centercoords2 = np.array([xcenter, ycenter])
            else:
                centercoords = np.where(self.imagearraygf == np.array(self.imagearraygf.max()))
                centercoords2 = np.array([centercoords[0][0], centercoords[1][0]])

            subsizey = 50
            subsizex = 50
            xlow = max(0, (centercoords2[0] - subsizex))
            xhigh = min(1024, (centercoords2[0] + subsizex))
            ylow = max(0, (centercoords2[1] - subsizey))
            yhigh = min(1280, (centercoords2[1] + subsizey))

            self.imagearraygfsub = self.imagearraygf[xlow:xhigh, ylow:yhigh]
            massCenter = np.array(ndi.measurements.center_of_mass(self.imagearraygfsub))
            # add the information about where the center of the subarray is
            focusMetricGlobal = massCenter[1] + centercoords2[1]  # - subsizey - self.sensorSize[1] / 2
        return focusMetricGlobal

    def setCropFrameParameters(self, cropSize: int, cropCenter: Optional[list] = None):
        """ Set the crop frame parameters for the camera. """
        self.cropSize = cropSize
        detectorSize = self._controller._master.detectorsManager[self._controller.camera].getDetectorSize()
        if cropSize > detectorSize[0] or cropSize > detectorSize[1]:
            raise ValueError(f"Crop size {cropSize} exceeds detector size {detectorSize}.")
        if cropCenter is None:
            cropCenter = np.array([cropSize // 2, cropSize // 2])
        self.cropCenter = cropCenter


class FocusCalibThread(Thread):
    def __init__(self, controller, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._controller = controller

    def run(self):
        self.signalData = []
        self.positionData = []
        self.fromVal = float(self._controller._widget.calibFromEdit.text())
        self.toVal = float(self._controller._widget.calibToEdit.text())
        self.scan_list = np.round(np.linspace(self.fromVal, self.toVal, 20), 2)
        for z in self.scan_list:
            self._controller._master.positionersManager[self._controller.positioner].setPosition(z, 0)
            time.sleep(0.5)
            self.focusCalibSignal = self._controller.setPointSignal
            self.signalData.append(self.focusCalibSignal)
            self.positionData.append(self._controller._master.positionersManager[self._controller.positioner].get_abs())
        self.poly = np.polyfit(self.positionData, self.signalData, 1)
        self.calibrationResult = np.around(self.poly, 4)
        self.show()

    def show(self):
        cal_nm = np.round(1000 / self.poly[0], 1)
        calText = f'1 px --> {cal_nm} nm'
        self._controller._widget.calibrationDisplay.setText(calText)

    def getData(self):
        data = {
            'signalData': self.signalData,
            'positionData': self.positionData,
            'poly': self.poly
        }
        return data


class PI:
    """Simple implementation of a discrete PI controller.
    Taken from http://code.activestate.com/recipes/577231-discrete-pid-controller/
    Author: Federico Barabas"""
    def __init__(self, setPoint, multiplier=1, kp=0, ki=0):
        self._kp = multiplier * kp
        self._ki = multiplier * ki
        self._setPoint = setPoint
        self.multiplier = multiplier
        self.error = 0.0
        self._started = False

    def update(self, currentValue):
        """ Calculate PI output value for given reference input and feedback.
        Using the iterative formula to avoid integrative part building. """
        self.error = self.setPoint - currentValue
        if self.started:
            self.dError = self.error - self.lastError
            self.out = self.out + self.kp * self.dError + self.ki * self.error
        else:
            # This only runs in the first step
            self.out = self.kp * self.error
            self.started = True
        self.lastError = self.error
        return self.out

    def restart(self):
        self.started = False

    @property
    def started(self):
        return self._started

    @started.setter
    def started(self, value):
        self._started = value

    @property
    def setPoint(self):
        return self._setPoint

    @setPoint.setter
    def setPoint(self, value):
        self._setPoint = value

    @property
    def kp(self):
        return self._kp

    @kp.setter
    def kp(self, value):
        self._kp = value

    @property
    def ki(self):
        return self._ki

    @ki.setter
    def ki(self, value):
        self._ki = value
        
    



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
    def double_gaussian_1d(xdata: np.ndarray, i0: float, x0: float, sigma: float, 
                          amp: float, dist: float) -> np.ndarray:
        """Double Gaussian model function"""
        x = xdata
        x0 = float(x0)
        return (i0 + amp * np.exp(-((x - (x0 - dist/2)) ** 2) / (2 * sigma ** 2)) +
                amp * np.exp(-((x - (x0 + dist/2)) ** 2) / (2 * sigma ** 2)))
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        Preprocess frame according to specification steps 1-3
        
        Args:
            frame: Input frame (can be RGB or grayscale)
            
        Returns:
            Preprocessed grayscale frame
        """
        # Step 1: Convert to grayscale if needed
        if len(frame.shape) == 3:
            # Convert RGB to grayscale by averaging channels
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
            
        # Apply mean subtraction (from original code)
        im = im - np.mean(im) / 2
        
        # Step 3: Threshold background
        im[im < self.config.background_threshold] = 0
        
        return im
        
    def preprocess_frame_rainer(self, frame: np.ndarray) -> np.ndarray:        
        if len(frame.shape) == 3:
            # Convert RGB to grayscale by averaging channels
            im = np.mean(frame, axis=-1).astype(np.uint8)
        else:
            im = frame.astype(np.uint8)
            
        # Convert to float for processing
        im = im.astype(float)            
        sum(nip.abs2(nip.rr2(ez) * np.max(0, intensityarray - maximum(intensityarray)/10)))
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
            
        # Apply mean subtraction (from original code)
        im = im - np.mean(im) / 2
        
        # Step 3: Threshold background
        im[im < self.config.background_threshold] = 0
        
        return im
    
    def compute_projections(self, im: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
    
    def fit_projections(self, projX: np.ndarray, projY: np.ndarray, isDoubleGaussX = False) -> tuple[float, float]:
        """
        Steps 5-6: Fit projections and compute focus value
        
        Args:
            projX: X projection (fit with double-Gaussian)
            projY: Y projection (fit with single-Gaussian)
            
        Returns:
            (sigma_x, sigma_y) - fitted standard deviations
        """
        h1, w1 = len(projY), len(projX)
        x = np.arange(w1)
        y = np.arange(h1)
        
        # Initial guess parameters for X fit (double Gaussian)
        i0_x = np.mean(projX)
        amp_x = np.max(projX) - i0_x
        sigma_x_init = np.std(projX)
        if isDoubleGaussX:
            init_guess_x = [i0_x, w1/2, sigma_x_init, amp_x, 100]
        else:
            init_guess_x = [i0_x, w1/2, sigma_x_init, amp_x]
        
        # Initial guess parameters for Y fit (single Gaussian)
        i0_y = np.mean(projY)
        amp_y = np.max(projY) - i0_y
        sigma_y_init = np.std(projY)
        init_guess_y = [i0_y, h1/2, sigma_y_init, amp_y]
        
        try:
            # Fit X projection with double Gaussian
            if isDoubleGaussX:
                popt_x, _ = curve_fit(self.double_gaussian_1d, x, projX, 
                                    p0=init_guess_x, maxfev=50000)
                sigma_x = abs(popt_x[2])  # Ensure positive sigma
            else:
                popt_x, _ = curve_fit(self.gaussian_1d, x, projX,
                                     p0=init_guess_x, maxfev=50000)
                sigma_x = abs(popt_x[2])  # Ensure positive sigma
                
            # Fit Y projection with single Gaussian  
            popt_y, _ = curve_fit(self.gaussian_1d, y, projY,
                                 p0=init_guess_y, maxfev=50000)
            sigma_y = abs(popt_y[2])  # Ensure positive sigma
            
        except Exception as e:
            # Fallback to standard deviation if fitting fails
            sigma_x = np.std(projX)
            sigma_y = np.std(projY)
            
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
            # Steps 1-3: Preprocess frame
            im = self.preprocess_frame(frame)
            
            # Step 4: Compute projections
            projX, projY = self.compute_projections(im)
            
            # Steps 5-6: Fit projections and compute focus value
            sigma_x, sigma_y = self.fit_projections(projX, projY)
            
            # Avoid division by zero
            if sigma_y == 0:
                focus_value = float('inf')
            else:
                focus_value = float(sigma_x / sigma_y)
                
        except Exception as e:
            # Return invalid focus value on error
            focus_value = float('nan')
            
        # Step 7: Return timestamped JSON
        return {
            "t": timestamp,
            "focus": focus_value
        }
    
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

