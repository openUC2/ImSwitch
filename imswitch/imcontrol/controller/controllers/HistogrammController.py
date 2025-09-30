import numpy as np
try:
    import NanoImagingPack as nip
    isNIP = True
except:
    isNIP = False

from imswitch import IS_HEADLESS
from imswitch.imcommon.framework import Signal, Thread, Worker, Mutex
from imswitch.imcontrol.view import guitools
from imswitch.imcommon.model import initLogger
from imswitch.imcommon.model import APIExport
from ..basecontrollers import LiveUpdatedController
import cv2

class HistogrammController(LiveUpdatedController):
    """ Linked to HistogrammWidget."""

    sigHistogramComputed = Signal(np.ndarray, np.ndarray)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.updateRate = 10
        self.it = 0

        # Connect CommunicationChannel signals
        self._commChannel.sigUpdateImage.connect(self.update)


    def update(self, detectorName, image, init, isCurrentDetector):
        """ Update with new detector frame. """

        # bin image to reduce the number of pixels and off-load CPU usage
        if image is None:
            return
        binned_image = cv2.resize(image, (0, 0), fx=0.25, fy=0.25)

        # Now calculate the histogram for the binned image
        nBins = 256
        hist, bins = np.histogram(binned_image.astype(np.uint8).ravel(), bins=nBins)
        units = np.linspace(0, 2**10, nBins)
        self.sigHistogramComputed.emit(units, hist)

        # display the curve
        if not IS_HEADLESS:
            self._widget.setHistogrammData(units,hist)

    @APIExport(runOnUIThread=False)
    def histogrammActive(self):
        '''just a dummy endpoint to check if the histogramm is active'''
        return True


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
