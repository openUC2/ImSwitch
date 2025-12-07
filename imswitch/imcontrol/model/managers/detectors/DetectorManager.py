import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import threading
import cv2
import numpy as np

from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger


@dataclass
class DetectorAction:
    """ An action that is made available for the user to execute. """

    group: str
    """ The group to place the action in (does not need to be
    pre-defined). """

    func: callable
    """ The function that is called when the action is executed. """


@dataclass
class DetectorParameter(ABC):
    """ Abstract base class for detector parameters that are made available for
    the user to view/edit. """

    group: str
    """ The group to place the parameter in (does not need to be
    pre-defined). """

    value: Any
    """ The value of the parameter. """

    editable: bool
    """ Whether it is possible to edit the value of the parameter. """


@dataclass
class DetectorNumberParameter(DetectorParameter):
    """ A detector parameter with a numerical value. """

    value: float
    """ The value of the parameter. """

    valueUnits: str
    """ Parameter value units, e.g. "nm" or "fps". """

@dataclass
class DetectorBooleanParameter(DetectorParameter):
    """ A detector parameter with a boolean value. """

    value: bool
    """ The value of the parameter. """
@dataclass
class DetectorListParameter(DetectorParameter):
    """ A detector parameter with a value from a list of options. """

    value: str
    """ The value of the parameter. """

    options: List[str]
    """ The available values to pick from. """


class DetectorManager(SignalInterface):
    """ Abstract base class for managers that control detectors. Each type of
    detector corresponds to a manager derived from this class. """

    sigImageUpdated = Signal(np.ndarray, bool, list)
    sigNewFrame = Signal()
    
    @abstractmethod
    def __init__(self, detectorInfo, name: str, fullShape: Tuple[int, int],
                 supportedBinnings: List[int], model: str, *,
                 parameters: Optional[Dict[str, DetectorParameter]] = None,
                 actions: Optional[Dict[str, DetectorAction]] = None,
                 croppable: bool = True,
                 isRGB: bool = False) -> None:
        """
        Args:
            detectorInfo: See setup file documentation.
            name: The unique name that the device is identified with in the
              setup file.
            fullShape: Maximum image size as a tuple ``(width, height)``.
            supportedBinnings: Supported binnings as a list.
            model: Detector device model name.
            parameters: Parameters to make available to the user to view/edit.
            actions: Actions to make available to the user to execute.
            croppable: Whether the detector image can be cropped.
            isRGB: color non monochromatic camera
        """

        super().__init__()
        self.__logger = initLogger(self, instanceName=name)

        self._detectorInfo = detectorInfo

        self._isMock = False
        
        self._frameStart = (0, 0)
        self._shape = fullShape

        self.__name = name
        self.__model = model
        self.__parameters = parameters if parameters is not None else {}
        self.__actions = actions if actions is not None else {}
        self.__croppable = croppable

        self.__flatfieldImage = None
        self.__isFlatfielding = False

        self._minValueFramePreview = -1
        self._maxValueFramePreview = -1

        self.__fullShape = fullShape
        self.__supportedBinnings = supportedBinnings
        self.__image = np.array([])

        self.__forAcquisition = detectorInfo.forAcquisition
        self.__forFocusLock = detectorInfo.forFocusLock
        #if not detectorInfo.forAcquisition and not detectorInfo.forFocusLock:
        #    raise ValueError('At least one of forAcquisition and forFocusLock must be set in'
        #                     ' DetectorInfo.')

        # set RGB if information is available
        try:
            isRGB = self._detectorInfo.managerProperties["isRGB"] #parameters['isRGB'].value
        except:
            isRGB = False
        self.setRGB(isRGB)
        self.setBinning(supportedBinnings[0])

    def updateLatestFrame(self, init):
        """ :meta private: """
        try:
            self.__image = self.getLatestFrame()
        except Exception:
            self.__logger.error(traceback.format_exc())
        else:
            if self.__image is not None:
                self.sigImageUpdated.emit(self.__image, init, self.scale) 
    def setMinValueFramePreview(self, value):
        """ Sets the minimum value for the frame preview to display via a jpeg image """
        self._minValueFramePreview = value

    def setMaxValueFramePreview(self, value):
        """ Sets the maximum value for the frame preview to display via a jpeg image """
        self._maxValueFramePreview = value

    def setParameter(self, name: str, value: Any) -> Dict[str, DetectorParameter]:
        """ Sets a parameter value and returns the updated list of parameters.
        If the parameter doesn't exist, i.e. the parameters field doesn't
        contain a key with the specified parameter name, an AttributeError will
        be raised. """
        if name == 'previewMinValue':
            self.setMinValueFramePreview(value)
            return
        elif name == 'previewMaxValue':
            self.setMaxValueFramePreview(value)
            return
        if name not in self.__parameters:
            raise AttributeError(f'Non-existent parameter "{name}" specified')
        self.__parameters[name].value = value
        return self.parameters

    def sendSoftwareTrigger(self) -> None:
        """Trigger a software trigger on the detector, if supported.
        This is a no-op for detectors that do not support software triggering.
        """
        pass

    def setTriggerSource(self, source: str) -> None:
        """Set the trigger source for the detector.
        This is a base implementation that updates the trigger_source parameter if it exists.
        Subclasses should override this method to implement hardware-specific trigger control.
        
        Args:
            source: The trigger source ("Continous", "Internal trigger", "External trigger")
        """
        if 'trigger_source' in self.parameters:
            self.setParameter('trigger_source', source)
        else:
            self.__logger.warning(f"Trigger source parameter not available for {self.__class__.__name__}")

    def getCurrentTriggerType(self) -> str:
        """availalbe trigger types from the camera"""
        return "Software"

    def getTriggerTypes(self)  -> List[str]:
        """ Returns a list of available trigger types for the detector. """
        return ["Software", "External", "Continuous"]

    def setRGB(self, isRGB: bool) -> None:
        """ Sets the sensortype of the camera """
        self._isRGB = isRGB

    def setBinning(self, binning: int) -> None:
        """ Sets the detector's binning. """

        if binning not in self.__supportedBinnings:
            raise ValueError(f'Specified binning value "{binning}" not supported by the detector')

        self._binning = binning

    def setFlatfieldImage(self, flatieldImage, setFlatfielding):
        pass

    @property
    def name(self) -> str:
        """ Unique detector name, defined in the detector's setup info. """
        return self.__name

    @property
    def isRGB(self) -> bool:
        return self.isRGB

    @property
    def model(self) -> str:
        """ Detector model name. """
        return self.__model

    @property
    def binning(self) -> int:
        """ Current binning. """
        return self._binning

    @property
    def supportedBinnings(self) -> List[int]:
        """ Supported binnings as a list. """
        return self.__supportedBinnings

    @property
    def frameStart(self) -> Tuple[int, int]:
        """ Position of the top left corner of the current frame as a tuple
        ``(x, y)``. """
        return self._frameStart

    @property
    def shape(self) -> Tuple[int, ...]:
        """ Current image size as a tuple ``(width, height, ...)``. """
        return self._shape

    @property
    def fullShape(self) -> Tuple[int, ...]:
        """ Maximum image size as a tuple ``(width, height, ...)``. """
        return self.__fullShape

    @property
    def image(self) -> np.ndarray:
        """ Latest LiveView image. """
        return self.__image

    @property
    def parameters(self) -> Dict[str, DetectorParameter]:
        """ Dictionary of available parameters. """
        return self.__parameters

    @property
    def actions(self) -> Dict[str, DetectorAction]:
        """ Dictionary of available actions. """
        return self.__actions

    @property
    def croppable(self) -> bool:
        """ Whether the detector supports frame cropping. """
        return self.__croppable

    @property
    def forAcquisition(self) -> bool:
        """ Whether the detector is used for acquisition. """
        return self.__forAcquisition

    @property
    def forFocusLock(self) -> bool:
        """ Whether the detector is used for focus lock. """
        return self.__forFocusLock

    @property
    def scale(self) -> List[int]:
        """ The pixel sizes in micrometers, all axes, in the format high dim
        to low dim (ex. [..., 'Z', 'Y', 'X']). Override in managers handling
        >3 dim images (e.g. APDManager). """
        return self.pixelSizeUm[1:]

    @property
    @abstractmethod
    def pixelSizeUm(self) -> List[int]:
        """ The pixel size in micrometers, in 3D, in the format
        ``[Z, Y, X]``. Non-scanned ``Z`` set to 1. """
        pass

    @abstractmethod
    def crop(self, hpos: int, vpos: int, hsize: int, vsize: int) -> None:
        """ Crop the frame read out by the detector. """
        pass

    @abstractmethod
    def getLatestFrame(self) -> np.ndarray:
        """ Returns the frame that represents what the detector currently is
        capturing. The returned object is a numpy array of shape
        (height, width). """
        pass

    @abstractmethod
    def getChunk(self) -> np.ndarray:
        """ Returns the frames captured by the detector since getChunk was last
        called, or since the buffers were last flushed (whichever happened
        last). The returned object is a numpy array of shape
        (numFrames, height, width). """
        pass

    @abstractmethod
    def flushBuffers(self) -> None:
        """ Flushes the detector buffers so that getChunk starts at the last
        frame captured at the time that this function was called. """
        pass

    @abstractmethod
    def startAcquisition(self) -> None:
        """ Starts image acquisition. """
        pass

    @abstractmethod
    def stopAcquisition(self) -> None:
        """ Stops image acquisition. """
        pass

    def finalize(self) -> None:
        """ Close/cleanup detector. """
        pass

    def recordFlatfieldImage(self, image: np.ndarray) -> np.ndarray:
        """ Performs flatfield correction on the specified image. """
        return image

    def getIsRGB(self):
        return self.isRGB

    def getCameraStatus(self) -> Dict[str, Any]:
        """ Returns comprehensive camera status information.
        This method collects all available camera information including hardware specs,
        current settings, and operational status.
        
        Returns:
            Dictionary containing camera status with the following keys:
            - model: Camera model name
            - isMock: Whether this is a mock/dummy camera
            - isConnected: Connection status
            - isRGB: Whether camera is RGB/color
            - sensorWidth: Full sensor width in pixels
            - sensorHeight: Full sensor height in pixels
            - currentWidth: Current frame width (after ROI/binning)
            - currentHeight: Current frame height (after ROI/binning)
            - pixelSizeUm: Physical pixel size in micrometers
            - binning: Current binning value
            - supportedBinnings: List of supported binning values
            - frameStart: Current ROI position as (x, y)
            - croppable: Whether ROI/cropping is supported
            - forAcquisition: Whether detector is used for acquisition
            - forFocusLock: Whether detector is used for focus lock
            - parameters: Dictionary of all detector parameters with values and metadata
        """
        status = {
            'model': self.__model,
            'isMock': self._isMock,
            'isConnected': not self._isMock,  # Override in subclasses if needed
            'isRGB': self._isRGB,
            'sensorWidth': self.__fullShape[0],
            'sensorHeight': self.__fullShape[1],
            'currentWidth': self._shape[0],
            'currentHeight': self._shape[1],
            'pixelSizeUm': self.pixelSizeUm,
            'binning': self._binning,
            'supportedBinnings': self.__supportedBinnings,
            'frameStart': self._frameStart,
            'croppable': self.__croppable,
            'forAcquisition': self.__forAcquisition,
            'forFocusLock': self.__forFocusLock,
            'parameters': {}
        }
        
        # Add all parameters with their current values and metadata
        for param_name, param_obj in self.__parameters.items():
            param_info = {
                'value': param_obj.value,
                'group': param_obj.group,
                'editable': param_obj.editable
            }
            
            # Add type-specific metadata
            if isinstance(param_obj, DetectorNumberParameter):
                param_info['type'] = 'number'
                param_info['units'] = param_obj.valueUnits
            elif isinstance(param_obj, DetectorListParameter):
                param_info['type'] = 'list'
                param_info['options'] = param_obj.options
            elif isinstance(param_obj, DetectorBooleanParameter):
                param_info['type'] = 'boolean'
            else:
                param_info['type'] = 'unknown'
            
            status['parameters'][param_name] = param_info
        
        return status

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
