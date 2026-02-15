"""
ESP32-based galvo scanner manager.

This module implements the galvo scanner manager for ESP32/UC2 hardware
using the UC2-REST galvo.py interface.
"""

from typing import Dict, Any, Optional
from imswitch.imcommon.model import initLogger
from .GalvoScannerManager import GalvoScannerManager


class ESP32GalvoScannerManager(GalvoScannerManager):
    """
    Galvo scanner manager for ESP32/UC2 hardware.
    
    This manager interfaces with the UC2-REST galvo module to control
    galvo mirror scanners connected to an ESP32 microcontroller.
    
    The galvo scanner uses DAC outputs to drive X and Y galvo mirrors
    for high-speed laser scanning applications.
    
    Configuration in setup JSON:
    {
        "galvoScanners": {
            "ESP32Galvo": {
                "managerName": "ESP32GalvoScannerManager",
                "managerProperties": {
                    "rs232device": "ESP32",
                    "nx": 256,
                    "ny": 256,
                    "x_min": 500,
                    "x_max": 3500,
                    "y_min": 500,
                    "y_max": 3500,
                    "sample_period_us": 1,
                    "frame_count": 0,
                    "bidirectional": false
                }
            }
        }
    }
    """

    def __init__(self, galvoScannerInfo, name: str, **lowLevelManagers):
        """
        Initialize the ESP32 galvo scanner manager.
        
        Args:
            galvoScannerInfo: Configuration from setup file
            name: Unique device name
            **lowLevelManagers: Low-level managers including rs232sManager
        """
        super().__init__(galvoScannerInfo, name)
        
        self.__logger = initLogger(self, instanceName=name)
        
        # Get the RS232/serial manager for ESP32 communication
        try:
            rs232device = galvoScannerInfo.managerProperties.get('rs232device', 'ESP32')
            self._rs232manager = lowLevelManagers['rs232sManager'][rs232device]
            self._galvo = self._rs232manager._esp32.galvo
            self.__logger.info(f"ESP32 Galvo Scanner '{name}' initialized successfully")
        except KeyError as e:
            self.__logger.error(f"Could not find RS232 device: {e}")
            self._galvo = None
        except Exception as e:
            self.__logger.error(f"Failed to initialize ESP32 Galvo Scanner: {e}")
            self._galvo = None

    def start_scan(self, nx: int = None, ny: int = None,
                   x_min: int = None, x_max: int = None,
                   y_min: int = None, y_max: int = None,
                   sample_period_us: int = None, frame_count: int = None,
                   bidirectional: bool = None,
                   pre_samples: int = None, fly_samples: int = None,
                   trig_delay_us: int = None, trig_width_us: int = None,
                   line_settle_samples: int = None, enable_trigger: int = None,
                   apply_x_lut: int = None,
                   timeout: int = 1) -> Dict[str, Any]:
        """
        Start galvo scanning with the specified parameters.
        
        Updates the internal configuration with any provided parameters,
        then starts the scan on the ESP32 hardware.
        
        Args:
            nx: Number of X samples per line
            ny: Number of Y lines
            x_min: Min X position 0-4095
            x_max: Max X position 0-4095
            y_min: Min Y position 0-4095
            y_max: Max Y position 0-4095
            sample_period_us: Microseconds per sample, 0=max speed
            frame_count: Number of frames, 0=infinite
            bidirectional: Enable bidirectional scanning
            pre_samples: Pre-scan samples
            fly_samples: Fly-back samples
            trig_delay_us: Trigger delay in microseconds
            trig_width_us: Trigger width in microseconds
            line_settle_samples: Line settling samples
            enable_trigger: Enable trigger output 0/1
            apply_x_lut: Apply X lookup table 0/1
            timeout: Request timeout in seconds
            
        Returns:
            dict: Response from ESP32 hardware
        """
        if self._galvo is None:
            self.__logger.error("Galvo not initialized")
            return {"error": "Galvo not initialized"}
        
        # Update config with provided values
        self.update_config(
            nx=nx, ny=ny,
            x_min=x_min, x_max=x_max,
            y_min=y_min, y_max=y_max,
            sample_period_us=sample_period_us,
            frame_count=frame_count,
            bidirectional=bidirectional,
            pre_samples=pre_samples,
            fly_samples=fly_samples,
            trig_delay_us=trig_delay_us,
            trig_width_us=trig_width_us,
            line_settle_samples=line_settle_samples,
            enable_trigger=enable_trigger,
            apply_x_lut=apply_x_lut
        )
        
        try:
            result = self._galvo.set_galvo_scan(
                nx=self._config.nx,
                ny=self._config.ny,
                x_min=self._config.x_min,
                x_max=self._config.x_max,
                y_min=self._config.y_min,
                y_max=self._config.y_max,
                sample_period_us=self._config.sample_period_us,
                frame_count=self._config.frame_count,
                bidirectional=self._config.bidirectional,
                pre_samples=self._config.pre_samples,
                fly_samples=self._config.fly_samples,
                trig_delay_us=self._config.trig_delay_us,
                trig_width_us=self._config.trig_width_us,
                line_settle_samples=self._config.line_settle_samples,
                enable_trigger=self._config.enable_trigger,
                apply_x_lut=self._config.apply_x_lut,
                timeout=timeout
            )
            self._is_scanning = True
            self.__logger.info(f"Started galvo scan: {self._config.nx}x{self._config.ny}")
            return result or {"status": "started", "config": self.get_config_dict()}
        except Exception as e:
            self.__logger.error(f"Failed to start galvo scan: {e}")
            return {"error": str(e)}

    def stop_scan(self, timeout: int = 1) -> Dict[str, Any]:
        """
        Stop the galvo scanner.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            dict: Response from ESP32 hardware
        """
        if self._galvo is None:
            self.__logger.error("Galvo not initialized")
            return {"error": "Galvo not initialized"}
        
        try:
            result = self._galvo.stop_galvo_scan(timeout=timeout)
            self._is_scanning = False
            self.__logger.info("Stopped galvo scan")
            return result or {"status": "stopped"}
        except Exception as e:
            self.__logger.error(f"Failed to stop galvo scan: {e}")
            return {"error": str(e)}

    def get_status(self, timeout: int = 1) -> Dict[str, Any]:
        """
        Get the current galvo scanner status.
        
        Args:
            timeout: Request timeout in seconds
            
        Returns:
            dict: Status including running, current_frame, current_line, config
        """
        if self._galvo is None:
            return {
                "error": "Galvo not initialized",
                "running": False,
                "config": self.get_config_dict()
            }
        
        try:
            result = self._galvo.get_galvo_status(timeout=timeout)
            
            # Update internal state from hardware response
            if result:
                self._is_scanning = result.get('running', False)
                self._current_frame = result.get('current_frame', 0)
                self._current_line = result.get('current_line', 0)
            
            return {
                "running": self._is_scanning,
                "current_frame": self._current_frame,
                "current_line": self._current_line,
                "config": self.get_config_dict(),
                "hardware_response": result
            }
        except Exception as e:
            self.__logger.error(f"Failed to get galvo status: {e}")
            return {
                "error": str(e),
                "running": self._is_scanning,
                "config": self.get_config_dict()
            }

    def set_dac(self, channel: int = 1, frequency: int = 1, offset: int = 0,
                amplitude: float = 1.0, clk_div: int = 0, phase: int = 0,
                invert: int = 1, timeout: int = 1) -> Dict[str, Any]:
        """
        Configure the DAC for analog waveform generation.
        
        This is used for continuous waveform generation mode rather than
        point-by-point scanning.
        
        Args:
            channel: DAC channel (1 or 2)
            frequency: Waveform frequency in Hz
            offset: DC offset
            amplitude: Waveform amplitude
            clk_div: Clock divider
            phase: Phase offset
            invert: Invert output (0 or 1)
            timeout: Request timeout in seconds
            
        Returns:
            dict: Response from hardware
        """
        if self._galvo is None:
            return {"error": "Galvo not initialized"}
        
        try:
            self._galvo.set_dac(
                channel=channel,
                frequency=frequency,
                offset=offset,
                amplitude=amplitude,
                clk_div=clk_div,
                phase=phase,
                invert=invert,
                timeout=timeout
            )
            return {"status": "dac_configured", "channel": channel}
        except Exception as e:
            self.__logger.error(f"Failed to set DAC: {e}")
            return {"error": str(e)}


# Copyright (C) 2020-2025 ImSwitch developers
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
