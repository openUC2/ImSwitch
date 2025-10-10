from imswitch import IS_HEADLESS
import time
import numpy as np
import scipy.ndimage as ndi
import threading
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController
from skimage.filters import gaussian
from imswitch.imcommon.framework import Signal
import cv2
import queue

# Global axis for Z-positioning - should be Z
gAxis = "Z"

class AutofocusController(ImConWidgetController):
    """Linked to AutofocusWidget."""
    sigUpdateFocusPlot = Signal(object, object)
    sigUpdateFocusValue = Signal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        self.isAutofusRunning = False

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

    def __del__(self):
        self._AutofocusThead.quit()
        self._AutofocusThead.wait()
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
    def autoFocus(self, rangez:int=100, resolutionz:int=10, defocusz:int=0):
        self.isAutofusRunning = True
        self._AutofocusThead = threading.Thread(
            target=self.doAutofocusBackground,
            args=(rangez, resolutionz, defocusz),
            daemon=True
        )
        self._AutofocusThead.start()

    @APIExport(runOnUIThread=True)
    def stopAutofocus(self):
        self.isAutofusRunning = False

    def grabCameraFrame(self):
        return self.camera.getLatestFrame()

    def recordFlatfield(self, nFrames=10, nGauss=16, defocusPosition=200, defocusAxis="Z"):
        flatfield = []
        posStart = self.stages.getPosition()[defocusAxis]
        time.sleep(1)
        self.stages.move(value=defocusPosition, axis=defocusAxis, is_absolute=False, is_blocking=True)
        for _ in range(nFrames):
            flatfield.append(self.grabCameraFrame())
        flatfield = np.mean(np.array(flatfield), 0)
        flatfield = gaussian(flatfield, sigma=nGauss)
        self.stages.move(value=-defocusPosition, axis=defocusAxis, is_absolute=False, is_blocking=True)
        time.sleep(1)
        return flatfield

    def doAutofocusBackground(self, rangez=100, resolutionz=10, defocusz=0):
        self._commChannel.sigAutoFocusRunning.emit(True)
        mProcessor = FrameProcessor()
        if defocusz != 0:
            flatfieldImage = self.recordFlatfield(defocusPosition=defocusz)
            mProcessor.setFlatfieldFrame(flatfieldImage)

        initialPosition = self.stages.getPosition()["Z"]
        Nz = int(2 * rangez // resolutionz)
        relative_positions = np.int32(np.linspace(-abs(rangez), abs(rangez), Nz))

        # Move to the first relative position
        self.stages.move(value=relative_positions[0], axis="Z", is_absolute=False, is_blocking=True)

        for iz in range(Nz):
            if not self.isAutofusRunning:
                break
            if iz != 0:
                step = relative_positions[iz] - relative_positions[iz - 1]
                self.stages.move(value=step, axis="Z", is_absolute=False, is_blocking=True)
            frame = self.grabCameraFrame()
            mProcessor.add_frame(frame, iz)

        allfocusvals = np.array(mProcessor.getFocusValueList(Nz))
        mProcessor.stop()

        if self.isAutofusRunning:
            coordinates = relative_positions + initialPosition
            if not IS_HEADLESS:
                self._widget.focusPlotCurve.setData(coordinates[:len(allfocusvals)], allfocusvals)
            else:
                self.sigUpdateFocusPlot.emit(coordinates[:len(allfocusvals)], allfocusvals)

            best_index = np.argmax(allfocusvals)
            bestzpos_rel = relative_positions[best_index]

            # Move to best focus
            self.stages.move(value=-2 * rangez, axis="Z", is_absolute=False, is_blocking=True)
            self.stages.move(value=(rangez + bestzpos_rel), axis="Z", is_absolute=False, is_blocking=True)
        else:
            # Return to initial absolute position if stopped
            self.stages.move(value=initialPosition, axis="Z", is_absolute=True, is_blocking=True)

        self._commChannel.sigAutoFocusRunning.emit(False)
        self.isAutofusRunning = False
        if not IS_HEADLESS:
            self._widget.focusButton.setText('Autofocus')

        final_z = bestzpos_rel + initialPosition if self.isAutofusRunning else initialPosition
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
            img, iz = self.frame_queue.get()
            self.process_frame(img, iz)

    def process_frame(self, img, iz):
        if self.flatFieldFrame is not None:
            img = img / self.flatFieldFrame
        img = self.extract(img, self.nCropsize)
        if len(img.shape) > 2:
            img = np.mean(img, -1)
        if 0:
            imagearraygf = ndi.gaussian_filter(img, self.nGauss)
            is_success, buffer = cv2.imencode(".jpg", imagearraygf, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            focusquality = len(buffer) if is_success else 0
        else:
            focusquality = self.calculate_focus_measure(img)
        self.allfocusvals.append(focusquality)

    def calculate_focus_measure(self, image, method="LAPE"):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)  # optional
        if method == "LAPE":
            if image.dtype == np.uint16:
                lap = cv2.Laplacian(image, cv2.CV_32F)
            else:
                lap = cv2.Laplacian(image, cv2.CV_16S)
            focus_measure = np.mean(np.square(lap))
        elif method == "GLVA":
            focus_measure = np.std(image, axis=None)  # GLVA
        else:
            focus_measure = np.std(image, axis=None)  # GLVA
        return focus_measure




    @staticmethod
    def extract(marray, crop_size):
        center_x, center_y = marray.shape[1] // 2, marray.shape[0] // 2
        x_start = center_x - crop_size // 2
        x_end = x_start + crop_size
        y_start = center_y - crop_size // 2
        y_end = y_start + crop_size
        return marray[y_start:y_end, x_start:x_end]

    def getFocusValueList(self, nFrameExpected, timeout=5):
        t0 = time.time()
        while len(self.allfocusvals) < nFrameExpected:
            time.sleep(0.01)
            if time.time() - t0 > timeout:
                break
        return self.allfocusvals

    def stop(self):
        self.isRunning = False

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
