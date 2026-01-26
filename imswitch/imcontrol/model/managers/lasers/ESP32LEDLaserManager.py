from imswitch.imcommon.model import initLogger
from .LaserManager import LaserManager
import numpy as np

class ESP32LEDLaserManager(LaserManager):
    """ LaserManager for controlling LEDs and LAsers connected to an
    ESP32 exposing a REST API
    Each LaserManager instance controls one LED/Laser channel on the ESP32.

    Manager properties:

    - ``rs232device`` -- name of the defined rs232 communication channel
      through which the communication should take place

    """

    def __init__(self, laserInfo, name, **lowLevelManagers):
        super().__init__(laserInfo, name, isBinary=False, valueUnits='mW', valueDecimals=0)
        self._rs232manager = lowLevelManagers['rs232sManager'][
            laserInfo.managerProperties['rs232device']
        ]
        if type(laserInfo.managerProperties['channel_index']) != int:
            # In older Versions we allowed the Ledmatrix to be accessible via the name 'LED', this is now deprecated
            raise ValueError("channel_index must be an integer. Most likely you have an old config file, please update it to use the channel index (0-3...) instead of the name 'LED'.")
        
        self.__logger = initLogger(self, instanceName=name)

        # Try to get commChannel if available (for emitting signals)
        self._commChannel = lowLevelManagers.get('commChannel', None)
        if self._commChannel is None:
            self.__logger.warning("Communication channel not available - laser status updates won't emit signals")

        self._esp32 = self._rs232manager._esp32
        self._laser = self._rs232manager._esp32.laser
        self._motor = self._rs232manager._esp32.motor
        self._led = self._rs232manager._esp32.led

        # preset the pattern
        self.power = 0
        self.channel_index = laserInfo.managerProperties['channel_index']

        # do we want to vary the laser intensity to despeckle the image?
        try:
            self.laser_despeckle_amplitude = laserInfo.managerProperties['laser_despeckle_amplitude']
            self.laser_despeckle_period = laserInfo.managerProperties['laser_despeckle_period']
            self.__logger.debug("Laser despeckle enabled")
        except:
            self.laser_despeckle_amplitude = 0. # %
            self.laser_despeckle_period = 10 # ms
            self.__logger.debug("Laser despeckle disabled")

        # set the laser to 0
        self.enabled = False
        self.setEnabled(self.enabled)

        # Register callback for laser status updates
        try:
            # Register callback with key 0 (similar to motor callback registration)
            self._laser.register_callback(0, callbackfct=self._callback_laser_status)
            self.__logger.debug(f"Laser status callback registered for channel {self.channel_index}")
        except Exception as e:
            self.__logger.error(f"Could not register laser status callback: {e}")

    def _callback_laser_status(self, laserValues):
        """ Callback function to handle laser status updates from the ESP32.
        Updates the internal state of the laser manager based on the received laser values array.
        
        Args:
            laserValues: numpy array containing the laser values for all channels
                         [laser0, laser1, laser2, laser3]
        """
        try:
            self.__logger.debug(f"Received laser status update: {laserValues}")

            # Update power for this specific laser channel
            if 0 <= self.channel_index < len(laserValues):
                new_power = laserValues[self.channel_index]
                if new_power != self.power:
                    self.power = new_power
                    self.__logger.info(f"Laser channel {self.channel_index} power updated to {self.power} mW")
                    # Update enabled state based on power value
                    self.enabled = (self.power > 0)

                    # Emit signal to update GUI if commChannel is available
                    if self._commChannel is not None:
                        laserDict = {self.name: {"power": self.power, "enabled": bool(self.enabled)}}
                        self._commChannel.sigUpdateLaserPower.emit(laserDict)

        except Exception as e:
            self.__logger.error(f"Error in _callback_laser_status: {e}")

    def setEnabled(self, enabled,  getReturn=False):
        """Turn on (N) or off (F) laser emission"""
        self.enabled = enabled
        self._laser.set_laser(self.channel_index,
                                            int(self.power*self.enabled),
                                            despeckleAmplitude = self.laser_despeckle_amplitude,
                                            despecklePeriod = self.laser_despeckle_period,
                                            is_blocking=getReturn)

    def setValue(self, power, getReturn=False):
        """Handles output power.
        Sends a RS232 command to the laser specifying the new intensity.
        """
        self.power = power
        if self.enabled:
            self._laser.set_laser(self.channel_index,
                                int(self.power),
                                despeckleAmplitude = self.laser_despeckle_amplitude,
                                despecklePeriod = self.laser_despeckle_period,
                                is_blocking=getReturn)

    def sendTrigger(self, triggerId):
        self._esp32.digital.sendTrigger(triggerId)

    def setGalvo(self,channel=1, frequency=1, offset=0, amplitude=1, clk_div=0, phase=0, invert=1, timeout=1):
        self._rs232manager._esp32.galvo.set_dac(
            channel=channel, frequency=frequency, offset=offset, amplitude=amplitude, clk_div=clk_div,
            phase=phase, invert=invert, timeout=timeout)


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
