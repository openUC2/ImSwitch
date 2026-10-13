from typing import List, Union
from imswitch.imcommon.model import APIExport, initLogger
from imswitch.imcontrol.model import configfiletools
from imswitch.imcontrol.view.guitools.ViewSetupInfo import LaserPresetInfo
from ..basecontrollers import ImConWidgetController


class LaserController(ImConWidgetController):
    """ Linked to LaserWidget."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self._logger = initLogger(self)
        self.settingAttr = False
        self.presetBeforeScan = None

        # Set up lasers
        for lName, lManager in self._master.lasersManager:
            # Set wavelength metadata (static property from config)
            if lManager.wavelength:
                self.setSharedAttr(lName, _wavelengthAttr, lManager.wavelength)
            
            if not lManager.isBinary:
                self.valueChanged(lName, lManager.valueRangeMin)

        # Connect CommunicationChannel signals
        self._commChannel.sharedAttrs.sigAttributeSet.connect(self.attrChanged)
        self._commChannel.sigScanStarting.connect(lambda: self.scanChanged(True))
        self._commChannel.sigScanEnded.connect(lambda: self.scanChanged(False))

    def closeEvent(self):
        self._master.lasersManager.execOnAll(lambda l: l.setScanModeActive(False))
        self._master.lasersManager.execOnAll(lambda l: l.setValue(0))

    def toggleLaser(self, laserName, enabled):
        """ Enable or disable laser (on/off)."""
        self._master.lasersManager[laserName].setEnabled(enabled)
        self.setSharedAttr(laserName, _enabledAttr, enabled)

    def valueChanged(self, laserName, magnitude):
        """ Change magnitude. """
        self._master.lasersManager[laserName].setValue(magnitude)
        self.setSharedAttr(laserName, _valueAttr, magnitude)

    def toggleModulation(self, laserName, enabled):
        """ Enable or disable laser modulation (on/off). """
        self._master.lasersManager[laserName].setModulationEnabled(enabled)
        self.setSharedAttr(laserName, _freqEnAttr, enabled)

    def frequencyChanged(self, laserName, frequency):
        """ Change modulation frequency. """
        self._master.lasersManager[laserName].setModulationFrequency(frequency)
        self.setSharedAttr(laserName, _freqAttr, frequency)

    def dutyCycleChanged(self, laserName, dutyCycle):
        """ Change modulation duty cycle. """
        self._master.lasersManager[laserName].setModulationDutyCycle(dutyCycle)
        self.setSharedAttr(laserName, _dcAttr, dutyCycle)

    def makePreset(self):
        """ Returns a preset object corresponding to the current laser values.
        """
        attrs = self._commChannel.sharedAttrs
        return {lName: LaserPresetInfo(value=attrs[(_attrCategory, lName, _valueAttr)])
                for lName, lManager in self._master.lasersManager
                if not lManager.isBinary and (_attrCategory, lName, _valueAttr) in attrs}

    def applyPreset(self, laserPreset):
        """ Loads a preset object into the current values. """
        for laserName, laserPresetInfo in laserPreset.items():
            self.setLaserValue(laserName, laserPresetInfo.value)

    def scanChanged(self, isScanning):
        """ Handles what happens when a scan is started/stopped. """
        self._master.lasersManager.execOnAll(lambda l: l.setScanModeActive(isScanning))

        defaultScanPresetName = self._setupInfo.defaultLaserPresetForScan
        if defaultScanPresetName in self._setupInfo.laserPresets:
            if isScanning and self.presetBeforeScan is None:
                # Scan started, save current values and apply default scan preset
                self.presetBeforeScan = self.makePreset()
                self.applyPreset(self._setupInfo.laserPresets[defaultScanPresetName])
            elif self.presetBeforeScan is not None:
                # Scan finished, restore the values that were set before the scan started
                self.applyPreset(self.presetBeforeScan)
                self.presetBeforeScan = None

    def attrChanged(self, key, value):
        if self.settingAttr or len(key) != 3 or key[0] != _attrCategory:
            return

        laserName = key[1]
        if key[2] == _enabledAttr:
            self.setLaserActive(laserName, value)
        elif key[2] == _valueAttr:
            self.setLaserValue(laserName, value)

    def setSharedAttr(self, laserName, attr, value):
        self.settingAttr = True
        try:
            self._commChannel.sharedAttrs[(_attrCategory, laserName, attr)] = value
        finally:
            self.settingAttr = False

    @APIExport()
    def getLaserNames(self) -> List[str]:
        """ Returns the device names of all lasers. These device names can be
        passed to other laser-related functions. """
        return self._master.lasersManager.getAllDeviceNames()

    @APIExport()
    def getLaserValueRanges(self, laserName: str) -> List[Union[int, float, None]]:
        """ Returns the value range of the specified laser as a tuple
        (min, max). If the laser can only be turned on/off, returns None. """
        try:
            lManager = self._master.lasersManager[laserName]
            if lManager.isBinary:
                return [0, 1]
            else:
                return (lManager.valueRangeMin, lManager.valueRangeMax)
        except KeyError:
            return None

    @APIExport()
    def getLaserValue(self, laserName: str) -> Union[int, float]:
        """ Returns the value of the specified laser, in the units that the
        laser uses. """
        try:
            return self._master.lasersManager[laserName].power
        except Exception:
            return 0

    @APIExport(runOnUIThread=True)
    def getLaserActive(self, laserName: str) -> bool:
        """ Returns whether the specified laser is powered on. """
        try:
            return self._master.lasersManager[laserName].enabled
        except Exception:
            return False

    @APIExport(runOnUIThread=True)
    def setLaserActive(self, laserName: str, active: bool) -> None:
        """ Sets whether the specified laser is powered on. """
        self.toggleLaser(laserName, active)


    @APIExport(runOnUIThread=True)
    def setLaserValue(self, laserName: str, value: Union[int, float]) -> None:
        """ Sets the value of the specified laser, in the units that the laser
        uses. """
        self.valueChanged(laserName, value)

    @APIExport()
    def changeScanPower(self, laserName, laserValue):
        defaultPreset = self._setupInfo.laserPresets[self._setupInfo.defaultLaserPresetForScan]
        defaultPreset[laserName] = LaserPresetInfo(value=laserValue)

    @APIExport()
    def setLaserChannelIndex(self, laserName: str, channelIndex: Union[int, str]) -> bool:
        """ Sets the channel_index for a specific laser in managerProperties 
        and saves it persistently to the configuration file.
        
        Args:
            laserName: The name of the laser to modify (e.g., '635', '488', 'LED Array', 'LED')
            channelIndex: The new channel index value (can be int or string like "LED")
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Check if laser exists in setup
            if laserName not in self._setupInfo.lasers:
                self._logger.error(f"Laser '{laserName}' not found in setup")
                return False

            # Get the laser info
            laserInfo = self._setupInfo.lasers[laserName]

            # Check if managerProperties exists and has channel_index
            if 'channel_index' not in laserInfo.managerProperties and channelIndex == "LED":
                self._logger.warning(f"Laser '{laserName}' does not have 'channel_index' in managerProperties")
                return False

            # Store old value for logging
            oldChannelIndex = laserInfo.managerProperties['channel_index']

            # update internal state
            self._master.lasersManager[laserName].channel_index = int(channelIndex)

            # Update the channel index
            laserInfo.managerProperties['channel_index'] = int(channelIndex)

            # Save to configuration file
            configfiletools.saveSetupInfo(configfiletools.loadOptions()[0], self._setupInfo)

            self._logger.info(f"Successfully changed laser '{laserName}' channel_index from {oldChannelIndex} to {channelIndex}")

            # Update the laser manager's channel_index if it has this attribute
            if hasattr(self._master.lasersManager[laserName], 'channel_index'):
                self._master.lasersManager[laserName].channel_index = channelIndex
                self._logger.info(f"Updated runtime channel_index for laser '{laserName}'")

            return True

        except Exception as e:
            self._logger.error(f"Error setting channel_index for laser '{laserName}': {str(e)}")
            return False

    @APIExport()
    def getLaserChannelIndex(self, laserName: str) -> Union[int, str, None]:
        """ Gets the current channel_index for a specific laser.
        
        Args:
            laserName: The name of the laser
            
        Returns:
            The channel_index value, or None if not found
        """
        try:
            if laserName not in self._setupInfo.lasers:
                self._logger.error(f"Laser '{laserName}' not found in setup")
                return None

            laserInfo = self._setupInfo.lasers[laserName]

            if 'channel_index' in laserInfo.managerProperties:
                return laserInfo.managerProperties['channel_index']
            else:
                self._logger.warning(f"Laser '{laserName}' does not have 'channel_index' in managerProperties")
                return None

        except Exception as e:
            self._logger.error(f"Error getting channel_index for laser '{laserName}': {str(e)}")
            return None

_attrCategory = 'Laser'
_enabledAttr = 'Enabled'
_valueAttr = 'Value'
_freqEnAttr = "ModulationEnabled"
_freqAttr = "Frequency"
_dcAttr = "DutyCycle"
_wavelengthAttr = "WavelengthNm"
_powerMwAttr = "PowerMw"


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
