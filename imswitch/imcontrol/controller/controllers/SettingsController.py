from dataclasses import dataclass
from typing import Any, List, Tuple

import numpy as np

from imswitch import IS_HEADLESS
from imswitch.imcommon.model import APIExport
from imswitch.imcontrol.model import configfiletools
from imswitch.imcontrol.view import guitools as guitools
from ..basecontrollers import ImConWidgetController


@dataclass
class SettingsControllerParams:
    model: Any
    binning: Any
    frameMode: Any
    x0: Any
    y0: Any
    width: Any
    height: Any
    applyROI: Any
    newROI: Any
    abortROI: Any
    saveMode: Any
    deleteMode: Any
    allDetectorsFrame: Any



class SettingsController(ImConWidgetController):
    """ Linked to SettingsWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.settingAttr = False
        self.allParams = {}

        if not self._master.detectorsManager.hasDevices():
            return


        # Connect CommunicationChannel signals
        self._commChannel.sigDetectorSwitched.connect(self.detectorSwitched)

        self.roiAdded = False

        if not IS_HEADLESS:
            # Set up detectors
            for dName, dManager in self._master.detectorsManager:
                if not dManager.forAcquisition:
                    continue

                self._widget.addDetector(
                    dName, dManager.model, dManager.parameters, dManager.actions,
                    dManager.supportedBinnings, self._setupInfo.rois
                )
            self.initParameters()
        self._commChannel.sharedAttrs.sigAttributeSet.connect(self.attrChanged)
        self.detectorSwitched(self._master.detectorsManager.getCurrentDetectorName())
        self.updateSharedAttrs()
        if IS_HEADLESS: return


        execOnAll = self._master.detectorsManager.execOnAll
        execOnAll(lambda c: (self.updateParamsFromDetector(detector=c)),
                  condition=lambda c: c.forAcquisition)
        execOnAll(lambda c: (self.adjustFrame(detector=c)),
                  condition=lambda c: c.forAcquisition)
        execOnAll(lambda c: (self.updateFrame(detector=c)),
                  condition=lambda c: c.forAcquisition)
        execOnAll(lambda c: (self.updateFrameActionButtons(detector=c)),
                  condition=lambda c: c.forAcquisition)


        # Connect SettingsWidget signals
        self._widget.sigROIChanged.connect(self.ROIchanged)
        self._widget.sigDetectorChanged.connect(self.detectorSwitchClicked)
        self._widget.sigNextDetectorClicked.connect(self.detectorNextClicked)

    def addROI(self):
        """ Adds the ROI to ImageWidget viewbox through the CommunicationChannel. """
        if not self.roiAdded:
            self._commChannel.sigAddItemToVb.emit(self._widget.getROIGraphicsItem())
            self.roiAdded = True

    def toggleROI(self, b, position=None, size=None):
        """ Show or hide ROI. """
        if b:
            self.addROI()
            self._widget.showROI(position, size)
        else:
            self._widget.hideROI()

    def initParameters(self):
        """ Take parameters from the detector Tree map. """
        for detectorName in self._master.detectorsManager.getAllDeviceNames():
            if self._master.detectorsManager[detectorName].forAcquisition:
                detectorTree = self._widget.trees[detectorName]
                framePar = detectorTree.p.param('Image frame')
                self.allParams[detectorName] = SettingsControllerParams(
                    model=detectorTree.p.param('Model'),
                    binning=framePar.param('Binning'),
                    frameMode=framePar.param('Mode'),
                    x0=framePar.param('X0'),
                    y0=framePar.param('Y0'),
                    width=framePar.param('Width'),
                    height=framePar.param('Height'),
                    applyROI=framePar.param('Apply'),
                    newROI=framePar.param('New ROI'),
                    abortROI=framePar.param('Abort ROI'),
                    saveMode=framePar.param('Save mode'),
                    deleteMode=framePar.param('Delete mode'),
                    allDetectorsFrame=framePar.param('Update all detectors')
                )

                params = self.allParams[detectorName]
                params.binning.sigValueChanged.connect(self.updateBinning)
                params.frameMode.sigValueChanged.connect(self.updateFrame)
                params.applyROI.sigActivated.connect(self.adjustFrame)
                params.newROI.sigActivated.connect(self.updateFrame)
                params.abortROI.sigActivated.connect(self.abortROI)
                params.saveMode.sigActivated.connect(self.saveMode)
                params.deleteMode.sigActivated.connect(self.deleteMode)

                def syncFrameParamsWithoutUpdates(): self.syncFrameParams(False, False)
                params.x0.sigValueChanged.connect(syncFrameParamsWithoutUpdates)
                params.y0.sigValueChanged.connect(syncFrameParamsWithoutUpdates)
                params.width.sigValueChanged.connect(syncFrameParamsWithoutUpdates)
                params.height.sigValueChanged.connect(syncFrameParamsWithoutUpdates)
                params.allDetectorsFrame.sigValueChanged.connect(self.syncFrameParams)

        detectorsParameters = self._master.detectorsManager.execOnAll(
            lambda c: c.parameters, condition=lambda c: c.forAcquisition
        )
        for detectorName, detectorParameters in detectorsParameters.items():
            for parameterName, parameter in detectorParameters.items():
                paramInWidget = self._widget.trees[detectorName].p.param(parameter.group).param(
                    parameterName
                )
                paramInWidget.sigValueChanged.connect(
                    lambda _, value, detectorName=detectorName, parameterName=parameterName:
                    self.setDetectorParameter(detectorName, parameterName, value)
                )

        detectorsActions = self._master.detectorsManager.execOnAll(
            lambda c: c.actions, condition=lambda c: c.forAcquisition
        )
        for detectorName, detectorActions in detectorsActions.items():
            for actionName, action in detectorActions.items():
                paramInWidget = self._widget.trees[detectorName].p.param(action.group).param(
                    actionName
                )
                paramInWidget.sigActivated.connect(action.func)

    def adjustFrame(self, *, detector=None):
        """ Crop detector and adjust frame. """

        if detector is None:
            self.getDetectorManagerFrameExecFunc()(lambda c: self.adjustFrame(detector=c))
            return

        # Adjust frame
        params = self.allParams[detector.name]
        try:
            binning = int(params.binning.value())
        except:
            binning = 1
        width = params.width.value()
        height = params.height.value()
        x0 = params.x0.value()
        y0 = params.y0.value()

        # Round to closest "divisable by 4" value.
        hpos = binning * x0
        vpos = binning * y0
        hsize = binning * width
        vsize = binning * height

        hmodulus = 4
        vmodulus = 4
        vpos = int(vmodulus * np.ceil(vpos / vmodulus))
        hpos = int(hmodulus * np.ceil(hpos / hmodulus))
        vsize = int(vmodulus * np.ceil(vsize / vmodulus))
        hsize = int(hmodulus * np.ceil(hsize / hmodulus))

        detector.crop(hpos, vpos, hsize, vsize)

        # Final shape values might differ from the user-specified one because of detector limitation
        # x128
        if detector.name == self._master.detectorsManager.getCurrentDetectorName():
            self._commChannel.sigAdjustFrame.emit(detector.shape)
            self._widget.hideROI()

        self.updateParamsFromDetector(detector=detector)
        self.updateSharedAttrs()

    def ROIchanged(self):
        """ Update parameters according to ROI. """
        frameStart = self._master.detectorsManager.execOnCurrent(lambda c: c.frameStart)
        ROI = self._widget.getROIGraphicsItem()
        pos = ROI.position
        size = ROI.size

        currentParams = self.getCurrentParams()
        currentParams.x0.setValue(frameStart[0] + int(pos[0]))
        currentParams.y0.setValue(frameStart[1] + int(pos[1]))
        currentParams.width.setValue(size[0])  # [0] is Width
        currentParams.height.setValue(size[1])  # [1] is Height

    def updateFrameActionButtons(self, *, detector=None):
        """ Shows the frame-related buttons appropriate for the current frame
        mode, and hides the others. """

        if detector is None:
            self.getDetectorManagerFrameExecFunc()(
                lambda c: self.updateFrameActionButtons(detector=c)
            )
            return

        params = self.allParams[detector.name]

        params.applyROI.hide()
        params.newROI.hide()
        params.abortROI.hide()
        params.saveMode.hide()
        params.deleteMode.hide()

        if params.frameMode.value() == 'Custom':
            params.applyROI.show()
            params.newROI.show()
            params.abortROI.show()
            params.saveMode.show()
        elif params.frameMode.value() != 'Full chip':
            params.deleteMode.show()

    def abortROI(self):
        """ Cancel and reset parameters of the ROI. """
        self.toggleROI(False)
        frameStart = self._master.detectorsManager.execOnCurrent(lambda c: c.frameStart)
        shapes = self._master.detectorsManager.execOnCurrent(lambda c: c.shape)

        currentParams = self.getCurrentParams()
        currentParams.x0.setValue(frameStart[0])
        currentParams.y0.setValue(frameStart[1])
        currentParams.width.setValue(shapes[0])
        currentParams.height.setValue(shapes[1])

    def saveMode(self):
        """ Save the current frame mode parameters to the mode list. """

        currentParams = self.getCurrentParams()
        x0, y0, width, height = (currentParams.x0.value(), currentParams.y0.value(),
                                 currentParams.width.value(), currentParams.height.value())

        name = guitools.askForTextInput(
            self._widget,
            'Add frame mode',
            f'Enter a name for this mode:\n(X0: {x0}; Y0: {y0}; Width: {width}; Height: {height})')

        if not name:  # No name provided
            return

        add = True
        alreadyExists = False
        if name in self._setupInfo.rois:
            alreadyExists = True
            add = guitools.askYesNoQuestion(
                self._widget,
                'Frame mode already exists',
                f'A frame mode with the name "{name}" already exists. Do you want to overwrite it"?'
            )

        if add:
            # Add in GUI
            if not alreadyExists:
                for params in self.allParams.values():
                    newModeItems = params.frameMode.opts['limits'].copy()
                    newModeItems.insert(len(newModeItems) - 1, name)
                    params.frameMode.setLimits(newModeItems)

            # Set in setup info
            self._setupInfo.setROI(name, x0, y0, width, height)
            configfiletools.saveSetupInfo(configfiletools.loadOptions()[0], self._setupInfo)

            # Update selected ROI in GUI
            for params in self.allParams.values():
                params.frameMode.setValue(name)

    def deleteMode(self):
        """ Delete the current frame mode from the mode list (if it's a saved
        custom ROI). """

        currentParams = self.getCurrentParams()
        modeToDelete = currentParams.frameMode.value()

        confirmationResult = guitools.askYesNoQuestion(
            self._widget,
            'Delete frame mode?',
            f'Are you sure you want to delete the mode "{modeToDelete}"?'
        )

        if confirmationResult:
            # Remove in GUI
            for params in self.allParams.values():
                newModeItems = params.frameMode.opts['limits'].copy()
                newModeItems = [value for value in newModeItems if value != modeToDelete]
                params.frameMode.setLimits(newModeItems)

            # Remove from setup info
            self._setupInfo.removeROI(modeToDelete)
            configfiletools.saveSetupInfo(configfiletools.loadOptions()[0], self._setupInfo)

    def updateBinning(self):
        """ Update a new binning to the detector. """
        def set_binning(c):
            if type(self.allParams[c.name].binning.value())!=int:
                c.setBinning(1)
            else:
                c.setBinning(int(self.allParams[c.name].binning.value()))

        self.getDetectorManagerFrameExecFunc()(set_binning)
        self.updateSharedAttrs()

    def updateParamsFromDetector(self, *, detector):
        """ Update the parameter values from the detector. """
        if IS_HEADLESS: return
        params = self.allParams[detector.name]

        # Detector parameters
        for parameterName, parameter in detector.parameters.items():
            paramInWidget = self._widget.trees[detector.name].p.param(parameter.group).param(
                parameterName
            )
            paramInWidget.setValue(parameter.value)

        # Frame
        params.binning.setValue(detector.binning)
        frameStart = detector.frameStart
        shape = detector.shape
        fullShape = detector.fullShape # width, height of the full chip
        params.x0.setValue(frameStart[0])
        params.y0.setValue(frameStart[1])
        params.width.setValue(shape[0])
        params.width.setLimits((1, fullShape[0]))
        params.height.setValue(shape[1])
        params.height.setLimits((1, fullShape[1]))

        # Model
        params.model.setValue(detector.model)

    def updateFrame(self, *, detector=None):
        """ Change the image frame size and position in the sensor. """

        if detector is None:
            self.getDetectorManagerFrameExecFunc()(lambda c: self.updateFrame(detector=c))
            return

        params = self.allParams[detector.name]
        frameMode = self.getCurrentParams().frameMode.value()
        customFrame = frameMode == 'Custom'

        params.x0.setWritable(customFrame)
        params.y0.setWritable(customFrame)
        params.width.setWritable(customFrame)
        params.height.setWritable(customFrame)
        # Call .show() to prevent view alignment issues
        params.x0.show()
        params.y0.show()
        params.width.show()
        params.height.show()

        if customFrame:
            ROIsize = (64, 64)
            ROIcenter = self._commChannel.getCenterViewbox()

            ROIpos = (ROIcenter[0] - 0.5 * ROIsize[0],
                      ROIcenter[1] - 0.5 * ROIsize[1])

            self.toggleROI(True, ROIpos, ROIsize)
            self.ROIchanged()

        else:
            if frameMode == "":
                pass
            else:
                if frameMode == 'Full chip':
                    fullChipShape = detector.fullShape
                    params.x0.setValue(0)
                    params.y0.setValue(0)
                    params.width.setValue(fullChipShape[0])
                    params.height.setValue(fullChipShape[1])
                else:
                    roiInfo = self._setupInfo.rois[frameMode]
                    params.x0.setValue(roiInfo.x)
                    params.y0.setValue(roiInfo.y)
                    params.width.setValue(roiInfo.w)
                    params.height.setValue(roiInfo.h)

                self.adjustFrame(detector=detector)

        self.syncFrameParams(doAdjustFrame=False)

    def detectorSwitched(self, newDetectorName, _=None):
        """ Called when the user switches to another detector. """
        newDetectorShape = self._master.detectorsManager[newDetectorName].shape
        self._commChannel.sigAdjustFrame.emit(newDetectorShape)
        if IS_HEADLESS: return
        self._widget.setDisplayedDetector(newDetectorName)
        self._widget.setImageFrameVisible(self._master.detectorsManager[newDetectorName].croppable)

    def detectorSwitchClicked(self, detectorName):
        """ Changes the current detector to the selected detector. """
        self._master.detectorsManager.setCurrentDetector(detectorName)

    def detectorNextClicked(self):
        """ Changes the current detector to the next detector. """
        self._widget.selectNextDetector()

    def syncFrameParams(self, doAdjustFrame=True, doUpdateFrameActionButtons=True):
        currentParams = self.getCurrentParams()
        shouldSync = currentParams.allDetectorsFrame.value()

        for params in self.allParams.values():
            params.allDetectorsFrame.setValue(shouldSync)
            if shouldSync:
                params.frameMode.setValue(currentParams.frameMode.value())
                params.x0.setValue(currentParams.x0.value())
                params.y0.setValue(currentParams.y0.value())
                params.width.setValue(currentParams.width.value())
                params.height.setValue(currentParams.height.value())

        if shouldSync and doAdjustFrame:
            self.adjustFrame()

        if doUpdateFrameActionButtons:
            self.updateFrameActionButtons()

    def getCurrentParams(self):
        return self.allParams[self._master.detectorsManager.getCurrentDetectorName()]

    def getDetectorManagerFrameExecFunc(self):
        """ Returns the detector manager exec function that should be used for
        frame-related changes. """
        currentParams = self.getCurrentParams()
        detectorsManager = self._master.detectorsManager
        return (detectorsManager.execOnAll if currentParams.allDetectorsFrame.value()
                else detectorsManager.execOnCurrent)

    def attrChanged(self, key, value):
        if self.settingAttr or len(key) < 3 or key[0] != _attrCategory:
            return

        detectorName = key[1]
        if detectorName is None:
            detectorName = self._master.detectorsManager.getAllDeviceNames()[0]
        if len(key) == 3:
            if key[2] == _binningAttr:
                self.setDetectorBinning(detectorName, value)
            elif key[2] == _ROIAttr:
                self.setDetectorROI(detectorName, (value[0], value[1]), (value[2], value[3]))
        if len(key) == 4:
            if key[2] == _detectorParameterSubCategory:
                self.setDetectorParameter(detectorName, key[3], value)

    def setSharedAttr(self, detectorName, attr, value, *, isDetectorParameter=False):
        self.settingAttr = True
        try:
            if not isDetectorParameter:
                key = (_attrCategory, detectorName, attr)
            else:
                key = (_attrCategory, detectorName, _detectorParameterSubCategory, attr)
            self._commChannel.sharedAttrs[key] = value
        finally:
            self.settingAttr = False

    def updateSharedAttrs(self):
        for dName, dManager in self._master.detectorsManager:
            self.setSharedAttr(dName, _modelAttr, dManager.model)
            self.setSharedAttr(dName, _pixelSizeAttr, dManager.pixelSizeUm)
            self.setSharedAttr(dName, _binningAttr, dManager.binning)
            self.setSharedAttr(dName, _ROIAttr, [*dManager.frameStart, *dManager.shape])

            for parameterName, parameter in dManager.parameters.items():
                self.setSharedAttr(dName, parameterName, parameter.value, isDetectorParameter=True)
    @APIExport()
    def sendSoftwareTrigger(self):
        """ Sends a software trigger to the current detector. """
        self._master.detectorsManager.execOnCurrent(lambda c: c.sendSoftwareTrigger())

    @APIExport()
    def setDetectorCompressionrate(self, compressionrate:int=80):
        self._master.detectorsManager.updateGlobalDetectorParams({"compressionlevel":compressionrate})

    @APIExport()
    def getDetectorGlobalParameters(self):
        return self._master.detectorsManager.getGlobalDetectorParams()

    @APIExport()
    def getDetectorParameters(self) -> dict:
        """ Returns the current parameters of the current detector. """
        # collect exposure time
        try: mExposureTime = self._master.detectorsManager.getCurrentDetector().parameters['exposure'].value
        except: mExposureTime = 1
        # collect gain
        try: mGain = self._master.detectorsManager.getCurrentDetector().parameters['gain'].value
        except: mGain = 0
        # collect pixelSize
        try: mPixelSize = self._master.detectorsManager.getCurrentDetector().pixelSizeUm[-1]
        except: mPixelSize = 1
        # collect binning
        try: mBinning = self._master.detectorsManager.getCurrentDetector().binning
        except: mBinning = 1
        # get Black Level
        try: mBlacklevel = self._master.detectorsManager.getCurrentDetector().parameters['blacklevel'].value
        except: mBlacklevel = 0
        # get rgb
        try: mRGB = self._master.detectorsManager.getCurrentDetector()._isRGB
        except: mRGB = 0
        # collect mode (auto/manual)
        try: camMode = self._master.detectorsManager.getCurrentDetector().parameters['exposure_mode'].value
        except: camMode = 'manual'
        mParameterDict = {
            'exposure': mExposureTime,
            'gain': mGain,
            'pixelSize': mPixelSize,
            'binning': mBinning,
            'blacklevel': mBlacklevel,
            'isRGB': mRGB,
            'mode': camMode
        }
        return mParameterDict

    @APIExport()
    def getDetectorNames(self) -> List[str]:
        """ Returns the device names of all detectors. These device names can
        be passed to other detector-related functions. """
        return self._master.detectorsManager.getAllDeviceNames()

    @APIExport(runOnUIThread=True)
    def setDetectorBinning(self, detectorName: str, binning: int) -> None:
        """ Sets binning value for the specified detector. """
        self.allParams[detectorName].binning.setValue(binning)
        self._master.detectorsManager[detectorName].setBinning(binning)

    @APIExport(runOnUIThread=True)
    def setDetectorROI(self, detectorName: str, frameStart: Tuple[int, int],
                       shape: Tuple[int, int]) -> None:
        """ Sets the ROI for the specified detector. frameStart is a tuple
        (x0, y0) and shape is a tuple (width, height). """

        detector = self._master.detectorsManager[detectorName]

        self.allParams[detectorName].frameMode.setValue('Custom')
        self.updateFrame(detector=detector)

        self.allParams[detectorName].x0.setValue(frameStart[0])
        self.allParams[detectorName].y0.setValue(frameStart[1])
        self.allParams[detectorName].width.setValue(shape[0])
        self.allParams[detectorName].height.setValue(shape[1])
        self.adjustFrame(detector=detector)

    @APIExport(runOnUIThread=True)
    def setDetectorParameter(self, detectorName: str, parameterName: str, value: Any) -> None:
        """ Sets the specified detector-specific parameter to the specified
        value. """

        if (parameterName in ['Trigger source'] and
                self.getCurrentParams().allDetectorsFrame.value()):
            # Special case for certain parameters that will follow the "update all detectors" option
            execFunc = self._master.detectorsManager.execOnAll
        else:
            def execFunc(f):
                self._master.detectorsManager.execOn(detectorName, f)

        execFunc(
            lambda c: (c.setParameter(parameterName, value) and
                       self.updateParamsFromDetector(detector=c))
        )
        self.updateSharedAttrs()

    @APIExport(runOnUIThread=True)
    def setDetectorMode(self, detectorName: str=None, isAuto: bool=True) -> None:
        """ Sets the detector mode for the specified detector. """
        if detectorName is None:
            detectorName = self._master.detectorsManager.getCurrentDetectorName()
        try:
            self.setDetectorParameter(detectorName, 'mode', 'Auto' if isAuto else 'Manual')
        except Exception as e:
            pass

    @APIExport(runOnUIThread=True)
    def setDetectorExposureTime(self, detectorName: str=None, exposureTime: float=1) -> None:
        """ Sets the exposure time for the specified detector. """
        if detectorName is None:
            detectorName = self._master.detectorsManager.getCurrentDetectorName()
        self.setDetectorParameter(detectorName, 'exposure', exposureTime)

    @APIExport(runOnUIThread=True)
    def setDetectorTriggerType(self, detectorName: str=None, triggerType: str='Software') -> None:
        """ Sets the trigger type for the specified detector. """
        if detectorName is None:
            detectorName = self._master.detectorsManager.getCurrentDetectorName()
        self.setDetectorParameter(detectorName, 'trigger_source', triggerType)

    @APIExport(runOnUIThread=True)
    def getDetectorTriggerTypes(self, detectorName: str=None) -> List[str]:
        """ Returns the available trigger types for the specified detector. """
        if detectorName is None:
            detectorName = self._master.detectorsManager.getCurrentDetectorName()
        return self._master.detectorsManager[detectorName].getTriggerTypes()

    @APIExport(runOnUIThread=True)
    def getDetectorCurrentTriggerType(self, detectorName: str=None) -> str:
        """ Returns the current trigger type for the specified detector. """
        if detectorName is None:
            detectorName = self._master.detectorsManager.getCurrentDetectorName()
        return self._master.detectorsManager[detectorName].getCurrentTriggerType()

    @APIExport(runOnUIThread=True)
    def setDetectorPreviewMinMaxValue(self, detectorName: str=None, minValue: int=0, maxValue: int = 1024) -> None:
        """ Sets the preview minimum value for the specified detector. """
        if detectorName is None:
            detectorName = self._master.detectorsManager.getCurrentDetectorName()
        self.setDetectorParameter(detectorName, 'previewMinValue', minValue)
        self.setDetectorParameter(detectorName, 'previewMaxValue', maxValue)


    @APIExport(runOnUIThread=True)
    def setDetectorGain(self, detectorName: str=None, gain: float=0) -> None:
        """ Sets the gain for the specified detector. """
        if detectorName is None:
            detectorName = self._master.detectorsManager.getCurrentDetectorName()
        self.setDetectorParameter(detectorName, 'gain', gain)

    @APIExport(runOnUIThread=True)
    def setDetectorPreviewMinValue(self, detectorName: str=None, minValue: int=0) -> None:
        """ Sets the preview minimum value for the specified detector. """
        if detectorName is None:
            detectorName = self._master.detectorsManager.getCurrentDetectorName()
        self.setDetectorParameter(detectorName, 'previewMinValue', minValue)

    @APIExport(runOnUIThread=True)
    def setDetectorPreviewMaxValue(self, detectorName: str=None, maxValue: int=4095) -> None:
        """ Sets the preview maximum value for the specified detector. """
        if detectorName is None:
            detectorName = self._master.detectorsManager.getCurrentDetectorName()
        self.setDetectorParameter(detectorName, 'previewMaxValue', maxValue)

_attrCategory = 'Detector'
_modelAttr = 'Model'
_pixelSizeAttr = 'Pixel size'
_binningAttr = 'Binning'
_ROIAttr = 'ROI'
_detectorParameterSubCategory = 'Param'


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

