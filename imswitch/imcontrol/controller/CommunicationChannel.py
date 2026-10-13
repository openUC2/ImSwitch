from typing import Mapping

import numpy as np
from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import pythontools, APIExport, SharedAttributes
from imswitch.imcommon.model import initLogger


class CommunicationChannel(SignalInterface):
    """
    Communication Channel is a class that handles the communication between Master Controller
    and Widgets, or between Widgets.
    """

    sigUpdateStreamFrame = Signal()

    sigDetectorParametersUpdated = Signal(dict)  # Emits: {'detectorName': str, 'parameters': dict}

    sigUpdateImage = Signal(
        str, np.ndarray, bool, list, bool
    )  # (detectorName, image, init, scale, isCurrentDetector)

    sigAcquisitionStarted = Signal()

    sigAcquisitionStopped = Signal()


    sigAdjustFrame = Signal(object)  # (shape)

    sigDetectorSwitched = Signal(str, str)  # (newDetectorName, oldDetectorName)

    sigGridToggled = Signal(bool)  # (enabled)

    sigCrosshairToggled = Signal(bool)  # (enabled)



    sigRecordingStarted = Signal()

    sigRecordingEnded = Signal()

    sigUpdateRecFrameNum = Signal(int)  # (frameNumber)

    sigUpdateRecTime = Signal(int)  # (recTime)

    sigMemorySnapAvailable = Signal(
        str, np.ndarray, object, bool
    )  # (name, image, filePath, savedToDisk)



    sigScanStarting = Signal()

    sigScanBuilt = Signal(object)  # (deviceList)

    sigScanStarted = Signal()

    sigScanDone = Signal()

    sigScanEnded = Signal()

    sigSLMMaskUpdated = Signal(object)  # (mask)
    sigSIMMaskUpdated = Signal(object) # (mask)


    sigSnapImg = Signal()

    sigSnapImgPrev = Signal(str, np.ndarray, str)  # (detector, image, nameSuffix)




    sigStartRecordingExternal = Signal()




    sigExperimentStop = Signal()
    #sigRequestScannersInScan = Signal()


    sigAutoFocus =  Signal(float, float) # scanrange and stepsize
    sigAutoFocusRunning = Signal(bool) # indicate if autofocus is running or not
    sigAutoFocusLiveValue = Signal(object) # live focus value during monitoring mode {"focus_value": float, "timestamp": float}

    # Objective
    sigToggleObjective = Signal(int) # objective slot number 1,2
    sigSetObjectiveByName = Signal(str) # objective name (e.g., "10x", "20x")
    sigSetObjectiveByID = Signal(str) # objective ID
    sigStartLiveAcquistion = Signal(bool)
    sigStopLiveAcquisition = Signal(bool)






    sigUpdateMotorPosition = Signal(list)  # # TODO: Just forcely update the positoin in the GUI

    sigHomingState = Signal(dict)  # (homingStateDict) - per-axis frame-homing progress for the frontend

    sigDiskFull = Signal(dict)  # (diskInfoDict) - storage is (nearly) full; tell the user to free space

    sigUpdateLaserPower = Signal(dict)  # (laserPowerDict) - updates laser power values from device callbacks

    sigUpdateCANDevices = Signal(list)  # (canDevicesList) - updates CAN device scan results from callbacks


    sigNewFrame = Signal()

    # signal to control actions from the ESP32

    # useq-schema related signals

    # light-sheet related signals
    sigStartLightSheet = Signal(float, float, float, str, str, float) # (startX, startY, speed, axis, lightsource, lightsourceIntensity)
    sigStopLightSheet = Signal()


    @property
    def sharedAttrs(self):
        return self.__sharedAttrs

    def __init__(self, main, setupInfo):
        super().__init__()
        self.__main = main
        self.__sharedAttrs = SharedAttributes()
        self.__logger = initLogger(self)
        self._scriptExecution = False
        self.__main._moduleCommChannel.sigExecutionFinished.connect(self.executionFinished)
        self.output = []

        self.streamstarted = False

    def getCenterViewbox(self):
        """ Returns the center point of the viewbox, as an (x, y) tuple. """
        if 'Image' in self.__main.controllers:
            return self.__main.controllers['Image'].getCenterViewbox()
        else:
            raise RuntimeError('Required image widget not available')

    def getDimsScan(self):
        if 'Scan' in self.__main.controllers:
            return self.__main.controllers['Scan'].getDimsScan()
        else:
            raise RuntimeError('Required scan widget not available')

    def getNumScanPositions(self):
        if 'Scan' in self.__main.controllers:
            return self.__main.controllers['Scan'].getNumScanPositions()
        else:
            raise RuntimeError('Required scan widget not available')

    def get_image(self, detectorName=None):
        return self.__main.controllers['View'].get_image(detectorName)

    def move(self, positionerName, axis="X", dist=0):
        return self.__main.controllers['Positioner'].move(positionerName, axis=axis, dist=dist)

    @APIExport(runOnUIThread=True)
    def acquireImage(self) -> None:
        image = self.get_image()
        self.output.append(image)

    def runScript(self, text):
        self.output = []
        self._scriptExecution = True
        self.__main._moduleCommChannel.sigRunScript.emit(text)

    def executionFinished(self):
        self.sigScriptExecutionFinished.emit()
        self._scriptExecution = False

    def isExecuting(self):
        return self._scriptExecution

    #@APIExport()
    def signals(self) -> Mapping[str, Signal]:
        """ Returns signals that can be used with e.g. the getWaitForSignal
        action. Currently available signals are:

         - acquisitionStarted
         - acquisitionStopped
         - recordingStarted
         - recordingEnded
         - scanEnded

        They can be accessed like this: api.imcontrol.signals().scanEnded
        """

        return pythontools.dictToROClass({
            'acquisitionStarted': self.sigAcquisitionStarted,
            'acquisitionStopped': self.sigAcquisitionStopped,
            'recordingStarted': self.sigRecordingStarted,
            'recordingEnded': self.sigRecordingEnded,
            'scanEnded': self.sigScanEnded,
            'saveFocus': self.sigSaveFocus
        })


# Copyright (C) 2020-2022 ImSwitch developers
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
