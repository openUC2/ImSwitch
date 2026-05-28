from abc import ABC, abstractmethod

from typing import Dict, List, Optional


class PositionerManager(ABC):
    """ Abstract base class for managers that control positioners. Each type of
    positioner corresponds to a manager derived from this class. """

    @abstractmethod
    def __init__(self, positionerInfo, name: str, initialPosition: Dict[str, float], initialSpeed: Optional[Dict[str, float]]=None):
        """
        Args:
            positionerInfo: See setup file documentation.
            name: The unique name that the device is identified with in the
              setup file.
            initialPosition: The initial position for each axis. This is a dict
              in the format ``{ axis: position }``.
        """

        self._positionerInfo = positionerInfo
        self._position = initialPosition
        self.__axes = positionerInfo.axes
        # SPEED
        if initialSpeed is None:
            # assign default speeds
            initialSpeed = {axis: 0 for axis in positionerInfo.axes} # TODO: Hardcoded - should be updated according to JSon?
        self._speed = initialSpeed
        try: self.setSpeed(positionerInfo.managerProperties.get("initialSpeed")) # force to display the correct values
        except: pass

        # HOME
        initialHome={
            axis: False for axis in positionerInfo.axes 
        }
        self._home = initialHome # is homed?

        # settings for stopping the axis
        initialStop={
            axis: False for axis in positionerInfo.axes # TODO: Hardcoded - hsould be updated according to JSon?
        }
        self._stop = initialStop # is stopped?

        self.__name = name
        try:self.setSpeed(positionerInfo.managerProperties.get("initialSpeed"))
        except: pass
        self.__forPositioning = positionerInfo.forPositioning
        self.__forScanning = positionerInfo.forScanning
        self.__resetOnClose = positionerInfo.resetOnClose
        if not positionerInfo.forPositioning and not positionerInfo.forScanning:
            raise ValueError('At least one of forPositioning and forScanning must be set in'
                             ' PositionerInfo.')

    @property
    def name(self) -> str:
        """ Unique positioner name, defined in the positioner's setup info. """
        return self.__name

    @property
    def position(self) -> Dict[str, float]:
        """ The position of each axis. This is a dict in the format
        ``{ axis: position }``. """
        return self._position

    @property
    def speed(self) -> Dict[str, float]:
        """ The speed of each axis. This is a dict in the format
        ``{ axis: position }``. """
        return self._speed

    @property
    def home(self) -> Dict[str, bool]:
        """ The home of each axis. This is a dict in the format
        ``{ axis: homed }``. """
        return self._home

    @property
    def stop(self) -> Dict[str, bool]:
        """ The stop of each axis. This is a dict in the format
        ``{ axis: stopped }``. """
        return self._stop

    @property
    def axes(self) -> List[str]:
        """ The list of axes that are controlled by this positioner. """
        return self.__axes

    @property
    def forPositioning(self) -> bool:
        """ Whether the positioner is used for manual positioning. """
        return self.__forPositioning

    @property
    def forScanning(self) -> bool:
        """ Whether the positioner is used for scanning. """
        return self.__forScanning

    @property
    def resetOnClose(self) -> bool:
        """ Whether the positioner should be reset to 0-position upon closing. """
        return self.__resetOnClose

    @abstractmethod
    def move(self, dist: float, axis: str):
        """ Moves the positioner by the specified distance and returns the new
        position. Derived classes will update the position field manually. If
        the positioner controls multiple axes, the axis must be specified. """
        pass

    @abstractmethod
    def moveForever(self, speed=(0, 0, 0, 0), is_stop=False):
        ''' Moves the positioner infinitely at a given speed'''
        pass


    # @abstractmethod
    # def _set_position(self, pos, axis):
    #     pass

    def setPosition(self, position: float, axis: str):
        """ Adjusts the positioner to the specified position and returns the
        new position. Derived classes will update the position field manually.
        If the positioner controls multiple axes, the axis must be specified.
        """
        # result_pos = self._set_position(position, axis)
        # self._position[axis] = result_pos
        pass

    def getPosition(self, axis: str) -> float:
        """ Returns the current position of the specified axis. """
        pass

    def finalize(self) -> None:
        """ Close/cleanup positioner. """
        pass

    def enableMotors(self, enable: bool=None, autoenable:bool=None) -> None:
        """ Enable/disable motors. """
        pass

    def moveToSampleMountingPosition(self) -> None:
        """ Move to sample mounting position. """
        pass

    def getDevicePositionAxis(self, axis="X"):
        """Raw device position (no offset applied) for the given axis.

        The base implementation reads ``self.position[axis]`` and re-adds the
        offset, which is correct only when ``self.position`` already stores the
        offset-corrected (user) value. Concrete managers should override this
        with a direct read from the hardware whenever possible so that the
        controller can compute a stable offset atomically.
        """
        userPos = self.position.get(axis, 0) if hasattr(self, 'position') else 0
        if hasattr(self, 'stageOffsetPositions'):
            return userPos + self.stageOffsetPositions.get(axis, 0)
        return userPos

    def resetStageOffsetAxis(self, axis="X"):
        """Reset the stage offset for the given axis to 0 (in-memory only)."""
        if hasattr(self, '_logger'):
            self._logger.debug(f'Resetting stage offset for {axis} axis.')
        if hasattr(self, 'stageOffsetPositions'):
            self.stageOffsetPositions[axis] = 0

    def setStageOffsetAxis(self, knownPosition=0, currentDevicePosition=None,
                            knownOffset=None, axis="X"):
        """Set the stage offset using a single canonical contract.

        The offset is defined as ``device_position - user_position`` so that
        ``user_position = device_position - offset`` everywhere. Given a known
        user-coordinate ``knownPosition`` and the raw device position at the
        same physical point (``currentDevicePosition``) the new offset is::

            offset = currentDevicePosition - knownPosition

        When ``knownOffset`` is passed it is stored verbatim (escape hatch for
        callers that already know the desired offset). When
        ``currentDevicePosition`` is omitted the manager reads it via
        :meth:`getDevicePositionAxis`.
        """
        if hasattr(self, '_logger'):
            self._logger.debug(f'Setting stage offset for {axis} axis.')
        if knownOffset is not None:
            offset = float(knownOffset)
        else:
            if currentDevicePosition is None:
                currentDevicePosition = self.getDevicePositionAxis(axis)
            offset = float(currentDevicePosition) - float(knownPosition)
        if hasattr(self, 'stageOffsetPositions'):
            self.stageOffsetPositions[axis] = offset

    def getStageOffsetAxis(self, axis="X"):
        """Return the stage offset for the given axis."""
        if hasattr(self, '_logger'):
            self._logger.debug(f'Getting stage offset for {axis} axis.')
        if hasattr(self, 'stageOffsetPositions'):
            return self.stageOffsetPositions.get(axis, 0)
        return 0

    def moveToSampleLoadingPosition(self, speed=10000, is_blocking=True):
        """ Move to sample loading position. """
        pass

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
