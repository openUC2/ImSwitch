import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
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
                '''
                eventually manipulate the image before sending it
                we have 255 levels of gray, so we can use the min and max values to scale the image

                TODO: not ideal as we scale noise, but we need to do this for the preview
                '''
                def stretch_pixels(image_12bit, lower_clip, upper_clip):
                    # Clamping to the range [lower_clip, upper_clip]
                    clamped = np.clip(image_12bit, lower_clip, upper_clip)
                    # Mapping the clamped values to [0, 255]
                    scaled = (clamped - lower_clip) * 255.0 / (upper_clip - lower_clip)
                    # Converting to 8-bit
                    return np.clip(scaled, 0, 255).astype(np.uint8)
                if self._maxValueFramePreview != -1 and self._minValueFramePreview != -1:
                    self.__image = stretch_pixels(self.__image, self._minValueFramePreview, self._maxValueFramePreview)
                self.sigImageUpdated.emit(self.__image, init, self.scale) # TODO - inject compressionrate?

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
