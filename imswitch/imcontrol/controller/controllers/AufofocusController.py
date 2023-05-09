import time

import numpy as np
from time import perf_counter
import scipy.ndimage as ndi
from scipy.ndimage.filters import laplace

from imswitch.imcommon.framework import Thread, Timer
from imswitch.imcommon.model import initLogger, APIExport
from ..basecontrollers import ImConWidgetController

# global axis for Z-positioning - should be Z
gAxis = "Z" 
T_DEBOUNCE = .2
class AutofocusController(ImConWidgetController):
    """Linked to AutofocusWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)

        if self._setupInfo.autofocus is None:
            return

        self.camera = self._setupInfo.autofocus.camera
        self.positioner = self._setupInfo.autofocus.positioner
        #self._master.detectorsManager[self.camera].crop(*self.cropFrame)

        # Connect AutofocusWidget buttons
        self._widget.focusButton.clicked.connect(self.focusButton)

        self._master.detectorsManager[self.camera].startAcquisition()
        self.__processDataThread = ProcessDataThread(self)

    def __del__(self):
        self.__processDataThread.quit()
        self.__processDataThread.wait()
        if hasattr(super(), '__del__'):
            super().__del__()

    def focusButton(self):
        rangez = float(self._widget.zStepRangeEdit.text())
        resolutionz = float(self._widget.zStepSizeEdit.text())
        self._widget.focusButton.setText('Stop')
        self.autoFocus(rangez,resolutionz)
        self._widget.focusButton.setText('Autofocus')

    @APIExport(runOnUIThread=True)
    # Update focus lock
    def autoFocus(self, rangez=100, resolutionz=10):

        '''
        The stage moves from -rangez...+rangez with a resolution of resolutionz
        For every stage-position a camera frame is captured and a contrast curve is determined

        '''
        # determine optimal focus position by stepping through all z-positions and cacluate the focus metric
        self.focusPointSignal = self.__processDataThread.update(rangez,resolutionz)

class ProcessDataThread(Thread):
    def __init__(self, controller, *args, **kwargs):
        self._controller = controller
        super().__init__(*args, **kwargs)

    def grabCameraFrame(self):
        detectorManager = self._controller._master.detectorsManager[self._controller.camera]
        self.latestimg = detectorManager.getLatestFrame()
        return self.latestimg

    def update(self, rangez, resolutionz):

        allfocusvals = []
        allfocuspositions = []

        
        # 0 move focus to initial position
        self._controller._master.positionersManager[self._controller.positioner].move(-rangez, axis=gAxis)
        img = self.grabCameraFrame()     # grab dummy frame?
        # store data
        Nz = int(2*rangez//resolutionz)
        allfocusvals = np.zeros(Nz)
        allfocuspositions  = np.zeros(Nz)
        allfocusimages = []

<<<<<<< Updated upstream
        # 1 compute focus for every z position
        for iz in range(Nz):

            # 0 Move stage to the predefined position - remember: stage moves in relative coordinates
            self._controller._master.positionersManager[self._controller.positioner].move(resolutionz, axis=gAxis)
            time.sleep(T_DEBOUNCE)
            positionz = iz*resolutionz
            self._controller._logger.debug(f'Moving focus to {positionz}')

            # 1 Grab camera frame
            self._controller._logger.debug("Grabbing Frame")
            img = self.grabCameraFrame()
            allfocusimages.append(img)

            # 2 Gaussian filter the image, to remove noise
            self._controller._logger.debug("Processing Frame")
            #img_norm = img-np.min(img)
            #img_norm = img_norm/np.mean(img_norm)
            imagearraygf = ndi.filters.gaussian_filter(img, 3)

            # 3 compute focus metric
            focusquality = np.mean(ndi.filters.laplace(imagearraygf))
            allfocusvals[iz]=focusquality
            allfocuspositions[iz] = positionz

        # display the curve
        self._controller._widget.focusPlotCurve.setData(allfocuspositions,allfocusvals)

        # 4 find maximum focus value and move stage to this position
        allfocusvals=np.array(allfocusvals)
        zindex=np.where(np.max(allfocusvals)==allfocusvals)[0]
        bestzpos = allfocuspositions[np.squeeze(zindex)]

         # 5 move focus back to initial position (reduce backlash)
        self._controller._master.positionersManager[self._controller.positioner].move(-Nz*resolutionz, axis=gAxis)
=======
        if 1:
            # 0 move focus to initial position
            self.stages.move(value=allfocuspositions[0], axis="Z", is_absolute=True, is_blocking=True)
            import cv2
            frameStack = []
            tStart = time.time()
            # alternative route - measure while moving
            self.doneMovingBackground = False
            def moveBackground():
                self.stages.move(value=allfocuspositions[-1], axis="Z", speed=30, is_absolute=True, is_blocking=True)
                self.doneMovingBackground = True
                
            # start thread to move stage 
            threading.Thread(target=moveBackground, args=(), daemon=True).start()

            # capture images until we arrive at the destination 
            while not self.doneMovingBackground:
                frameStack.append(cv2.resize(self.grabCameraFrame(), None, None, fx=.1, fy=.1))
                print(len(frameStack))
                
            tEnd = time.time()
            frameStack = np.array(frameStack)
    
            # move back to initial position
            self.stages.move(value=allfocuspositions[0], axis="Z", is_absolute=True, is_blocking=True)

            # compute best focus and relate to the timing
            focusMetric = []
            for iFrame in frameStack:
                # blur
                imagearraygf = ndi.filters.gaussian_filter(iFrame, 3)

                # compute focus metric
                focusquality = np.mean(ndi.filters.laplace(imagearraygf))
                focusMetric.append(focusquality)
            
            # identify position with max focus
            indexMaxFocus = np.argmax(np.array(focusMetric))
            maxFocusPosition = allfocuspositions[0]+np.abs(allfocuspositions[-1]-allfocuspositions[0])/len(focusMetric)*indexMaxFocus
            self.stages.move(value=maxFocusPosition, axis="Z", is_absolute=True, is_blocking=True)

        
            # We are done!
            self._commChannel.sigAutoFocusRunning.emit(False) # inidicate that we are running the autofocus
            self.isAutofusRunning = False

            self._widget.focusButton.setText('Autofocus')
            allfocuspositions = np.linspace(allfocuspositions[0],allfocuspositions[-1],len(focusMetric))
            allfocusvals = np.array(focusMetric)
            self._widget.focusPlotCurve.setData(allfocuspositions,allfocusvals)

            return maxFocusPosition

        else:


            # 1 compute focus for every z position
            for iz in range(Nz):

                # exit autofocus if necessary
                if not self.isAutofusRunning:
                    break
                # 0 Move stage to the predefined position - remember: stage moves in relative coordinates
                self.stages.move(value=allfocuspositions[iz], axis="Z", is_absolute=True, is_blocking=True)

                time.sleep(T_DEBOUNCE)
                self._logger.debug(f'Moving focus to {allfocuspositions[iz]}')

                # 1 Grab camera frame
                self._logger.debug("Grabbing Frame")
                img = self.grabCameraFrame()
                # crop frame, only take inner 40%
                if isNIP:
                    img = nip.extract(img, (int(img.shape[0]*0.4),int(img.shape[1]*0.4)))
                allfocusimages.append(img)

                # 2 Gaussian filter the image, to remove noise
                self._logger.debug("Processing Frame")
                #img_norm = img-np.min(img)
                #img_norm = img_norm/np.mean(img_norm)
                imagearraygf = ndi.filters.gaussian_filter(img, 3)

                # 3 compute focus metric
                focusquality = np.mean(ndi.filters.laplace(imagearraygf))
                allfocusvals[iz]=focusquality

            if self.isAutofusRunning:
                # display the curve
                self._widget.focusPlotCurve.setData(allfocuspositions,allfocusvals)

                # 4 find maximum focus value and move stage to this position
                allfocusvals=np.array(allfocusvals)
                zindex=np.where(np.max(allfocusvals)==allfocusvals)[0]
                bestzpos = allfocuspositions[np.squeeze(zindex)]

                # 5 move focus back to initial position (reduce backlash)
                self.stages.move(value=allfocuspositions[0], axis="Z", is_absolute=True, is_blocking=True)

                # 6 Move stage to the position with max focus value
                self._logger.debug(f'Moving focus to {zindex*resolutionz}')
                self.stages.move(value=bestzpos, axis="Z", is_absolute=True, is_blocking=True)

                if False:
                    allfocusimages=np.array(allfocusimages)
                    np.save('allfocusimages.npy', allfocusimages)
                    import tifffile as tif
                    tif.imsave("llfocusimages.tif", allfocusimages)
                    np.save('allfocuspositions.npy', allfocuspositions)
                    np.save('allfocusvals.npy', allfocusvals)

            else:
                self.stages.move(value=initialPosition, axis="Z", is_absolute=True, is_blocking=True)
>>>>>>> Stashed changes

        # 6 Move stage to the position with max focus value
        self._controller._logger.debug(f'Moving focus to {zindex*resolutionz}')
        self._controller._master.positionersManager[self._controller.positioner].move(zindex*resolutionz, axis=gAxis)


<<<<<<< Updated upstream
        # DEBUG
        allfocusimages=np.array(allfocusimages)
        np.save('allfocusimages.npy', allfocusimages)
        import tifffile as tif
        tif.imsave("llfocusimages.tif", allfocusimages)
        np.save('allfocuspositions.npy', allfocuspositions)
        np.save('allfocusvals.npy', allfocusvals)

        return bestzpos
=======
            # DEBUG

            # We are done!
            self._commChannel.sigAutoFocusRunning.emit(False) # inidicate that we are running the autofocus
            self.isAutofusRunning = False

            self._widget.focusButton.setText('Autofocus')
            return bestzpos
>>>>>>> Stashed changes

# Copyright (C) 2020-2021 ImSwitch developers
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
