import numpy as np
import pytest
import time
import threading

from imswitch.imcontrol.model import DetectorsManager, RS232sManager
from imswitch.imcontrol.model.SetupInfo import RS232Info
from imswitch.imcontrol._test.unit import detectorInfosBasic, detectorInfosMulti, detectorInfosNonSquare


def getImage(detectorsManager, timeout_seconds=30):
    """Get image from detector without Qt dependencies."""
    receivedImage = None
    numImagesReceived = 0
    error_occurred = None

    def imageUpdated(detectorName, img, frameName, frameNumber, isCurrentDetector):
        nonlocal receivedImage, numImagesReceived
        if isCurrentDetector:
            receivedImage = img
            numImagesReceived += 1

    # Connect signal handler
    detectorsManager.sigImageUpdated.connect(imageUpdated)

    try:
        # Start acquisition
        handle = detectorsManager.startAcquisition(liveView=True)
        
        # Wait for images with timeout
        start_time = time.time()
        while numImagesReceived < 3 and (time.time() - start_time) < timeout_seconds:
            time.sleep(0.1)  # Small sleep to prevent busy waiting
            
        # Check if we got the required images
        if numImagesReceived < 3:
            raise TimeoutError(f"Only received {numImagesReceived} images within {timeout_seconds}s timeout")
            
    finally:
        # Clean up
        try:
            detectorsManager.sigImageUpdated.disconnect(imageUpdated)
            detectorsManager.stopAcquisition(handle, liveView=True)
        except Exception as e:
            print(f"Warning: Error during cleanup: {e}")

    return receivedImage


@pytest.mark.parametrize('detectorInfos', [detectorInfosBasic, detectorInfosNonSquare])
def test_acquisition_liveview_single(detectorInfos):
    # initialize the lowLevelManager VirtualMicroscope
    VirtualMicroscope = RS232sManager({'VirtualMicroscope': RS232Info(managerName='VirtualMicroscopeManager', managerProperties={'imagePath_': 'simplant', 'imagePath__': 'smlm'})})
    # TODO: In the second - multicamera run this function is also called - not sure why!
    lowLevelManagers = {
        'rs232sManager': VirtualMicroscope
    }
    detectorsManager = DetectorsManager(detectorInfos, updatePeriod=100, **lowLevelManagers)
    receivedImage = getImage(detectorsManager)

    assert receivedImage is not None
    assert receivedImage.shape == (
        detectorInfos['CAM'].managerProperties['virtcam']['image_height'],
        detectorInfos['CAM'].managerProperties['virtcam']['image_width']
    )
    assert not np.all(receivedImage == receivedImage[0, 0])  # Assert that not all pixels are same


@pytest.mark.parametrize('currentDetector', ['Camera 1', 'Camera 2'])
def test_acquisition_liveview_multi(currentDetector):
    # initialize the lowLevelManager VirtualMicroscope for multi-camera test
    VirtualMicroscope = RS232sManager({'VirtualMicroscope': RS232Info(managerName='VirtualMicroscopeManager', managerProperties={'imagePath_': 'simplant', 'imagePath__': 'smlm'})})

    lowLevelManagers = {
        'rs232sManager': VirtualMicroscope
    }
    detectorsManager = DetectorsManager(detectorInfosMulti, updatePeriod=100, **lowLevelManagers)
    detectorsManager.setCurrentDetector(currentDetector)
    receivedImage = getImage(detectorsManager)

    assert receivedImage is not None
    assert receivedImage.shape == (
        detectorInfosMulti[currentDetector].managerProperties['virtcam']['image_height'],
        detectorInfosMulti[currentDetector].managerProperties['virtcam']['image_width']
    )
    assert not np.all(receivedImage == receivedImage[0, 0])  # Assert that not all pixels are same
    
    
    '''
    TODO: Weget the following issue:
    
self = <imswitch.imcontrol.model.managers.DetectorsManager.DetectorsManager object at 0x15a242750>
detectorName = 'Camera 2'

    def setCurrentDetector(self, detectorName):
        """ Sets the current detector by its name. """
    
        self._validateManagedDeviceName(detectorName)
    
        oldDetectorName = self._currentDetectorName
        self._currentDetectorName = detectorName
        self.sigDetectorSwitched.emit(detectorName, oldDetectorName)
    
>       if self._thread.isRunning():
E       AttributeError: 'Thread' object has no attribute 'isRunning'

'''

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
