import asyncio
from typing import Mapping

import numpy as np
from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import pythontools, APIExport, SharedAttributes
from imswitch.imcommon.model import initLogger

import numpy as np
from PIL import Image
from io import BytesIO
from fastapi.responses import StreamingResponse
from fastapi import FastAPI, Response
import cv2

class CommunicationChannel(SignalInterface):
    """
    Communication Channel is a class that handles the communication between Master Controller
    and Widgets, or between Widgets.
    """
    #sigRequestScannersInScan = Signal()

    #sigSendScannersInScan = Signal(object)  # (scannerList)

    # Objective 

    # signal to control actions from the ESP32

    # useq-schema related signals

    # light-sheet related signals

    # scanning-related signals


    @property
    def sharedAttrs(self):
        return self.__sharedAttrs

    def __init__(self, main, setupInfo):
super().__init__()
        self.sigUpdateImage = Signal(
        str, np.ndarray, bool, list, bool
    )  # (detectorName, image, init, scale, isCurrentDetector)  # (detectorName, image, init, scale, isCurrentDetector)
        self.sigAcquisitionStarted = Signal()
        self.sigAcquisitionStopped = Signal()
        self.sigScriptExecutionFinished = Signal()
        self.sigAdjustFrame = Signal(object)  # (shape)  # (shape)
        self.sigDetectorSwitched = Signal(str, str)  # (newDetectorName, oldDetectorName)  # (newDetectorName, oldDetectorName)
        self.sigGridToggled = Signal(bool)  # (enabled)  # (enabled)
        self.sigCrosshairToggled = Signal(bool)  # (enabled)  # (enabled)
        self.sigAddItemToVb = Signal(object)  # (item)  # (item)
        self.sigRemoveItemFromVb = Signal(object)  # (item)  # (item)
        self.sigRecordingStarted = Signal()
        self.sigRecordingEnded = Signal()
        self.sigUpdateRecFrameNum = Signal(int)  # (frameNumber)  # (frameNumber)
        self.sigUpdateRecTime = Signal(int)  # (recTime)  # (recTime)
        self.sigMemorySnapAvailable = Signal(
        str, np.ndarray, object, bool
    )  # (name, image, filePath, savedToDisk)  # (name, image, filePath, savedToDisk)
        self.sigRunScan = Signal(bool, bool)  # (recalculateSignals, isNonFinalPartOfSequence)  # (recalculateSignals, isNonFinalPartOfSequence)
        self.sigAbortScan = Signal()
        self.sigScanStarting = Signal()
        self.sigScanBuilt = Signal(object)  # (deviceList)  # (deviceList)
        self.sigScanStarted = Signal()
        self.sigScanDone = Signal()
        self.sigScanEnded = Signal()
        self.sigSLMMaskUpdated = Signal(object)  # (mask)  # (mask)
        self.sigSIMMaskUpdated = Signal(object) # (mask)  # (mask)
        self.sigToggleBlockScanWidget = Signal(bool)
        self.sigSnapImg = Signal()
        self.sigSnapImgPrev = Signal(str, np.ndarray, str)  # (detector, image, nameSuffix)  # (detector, image, nameSuffix)
        self.sigRequestScanParameters = Signal()
        self.sigSendScanParameters = Signal(dict, dict, object)  # (analogParams, digitalParams, scannerList)  # (analogParams, digitalParams, scannerList)
        self.sigSetAxisCenters = Signal(object, object)  # (axisDeviceList, axisCenterList)  # (axisDeviceList, axisCenterList)
        self.sigStartRecordingExternal = Signal()
        self.sigRequestScanFreq = Signal()
        self.sigSendScanFreq = Signal(float)  # (scanPeriod)  # (scanPeriod)
        self.sigPixelSizeChange = Signal(float)  # (pixelSize)  # (pixelSize)
        self.sigExperimentStop = Signal()
        self.sigFlatFieldRunning = Signal(bool)
        self.sigFlatFieldImage = Signal(object)
        self.sigAutoFocus = Signal(float, float) # scanrange and stepsize  # scanrange and stepsize
        self.sigAutoFocusRunning = Signal(bool) # indicate if autofocus is running or not  # indicate if autofocus is running or not
        self.sigToggleObjective = Signal(int) # objective slot number 1,2  # objective slot number 1,2
        self.sigStartLiveAcquistion = Signal(bool)
        self.sigStopLiveAcquisition = Signal(bool)
        self.sigInitialFocalPlane = Signal(float) # initial focal plane for DeckScanController  # initial focal plane for DeckScanController
        self.sigBroadcast = Signal(str, str, object)
        self.sigSaveFocus = Signal()
        self.sigScanFrameFinished = Signal()  # TODO: emit this signal when a scanning frame finished, maybe in scanController if possible? Otherwise in APDManager for now, even if that is not general if you want to do camera-based experiments. Could also create a signal specifically for this from the scan curve generator perhaps, specifically for the rotation experiments, would that be smarter?  # TODO: emit this signal when a scanning frame finished, maybe in scanController if possible? Otherwise in APDManager for now, even if that is not general if you want to do camera-based experiments. Could also create a signal specifically for this from the scan curve generator perhaps, specifically for the rotation experiments, would that be smarter?
        self.sigUpdateRotatorPosition = Signal(str, str)  # (rotatorName)  # (rotatorName)
        self.sigUpdateMotorPosition = Signal(list)  # # TODO: Just forcely update the positoin in the GUI  # # TODO: Just forcely update the positoin in the GUI
        self.sigSetSyncInMovementSettings = Signal(str, float)  # (rotatorName, position)  # (rotatorName, position)
        self.sigNewFrame = Signal()
        self.sigESP32Message = Signal(str, str)  # (key, message)  # (key, message)
        self.sigSetXYPosition = Signal(float, float)
        self.sigSetZPosition = Signal(float)
        self.sigSetExposure = Signal(float)
        self.sigSetSpeed = Signal(float)
        self.sigStartLightSheet = Signal(float, float, float, str, str, float) # (startX, startY, speed, axis, lightsource, lightsourceIntensity)  # (startX, startY, speed, axis, lightsource, lightsourceIntensity)
        self.sigStopLightSheet = Signal()
        self.sigStartTileBasedTileScanning = Signal(int, int, int, int, int, int, str, int, int, bool, bool, bool) # (numb erTilesX, numberTilesY, stepSizeX, stepSizeY, nTimes, tPeriod, illuSource, initPosX, initPosY, isStitchAshlar, isStitchAshlarFlipX, isStitchAshlarFlipY)  # (numb erTilesX, numberTilesY, stepSizeX, stepSizeY, nTimes, tPeriod, illuSource, initPosX, initPosY, isStitchAshlar, isStitchAshlarFlipX, isStitchAshlarFlipY)
        self.sigStopTileBasedTileScanning = Signal()
        self.sigOnResultTileBasedTileScanning = Signal(np.ndarray, np.ndarray) # (tiles, postions)  # (tiles, postions)
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
