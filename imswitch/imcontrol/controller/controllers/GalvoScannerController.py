"""
Galvo Scanner Controller for ImSwitch.

This controller provides the API interface for controlling galvo scanners,
exposing scan parameters and control functions via REST API.
"""

from typing import Dict, List, Optional, Any
from imswitch import IS_HEADLESS
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController


class GalvoScannerController(ImConWidgetController):
    """
    Controller for galvo scanner devices.
    
    Provides REST API endpoints for:
    - Starting/stopping galvo scans
    - Configuring scan parameters (nx, ny, x/y ranges, timing)
    - Getting scanner status
    - Managing multiple galvo scanner devices
    
    API Endpoints (via @APIExport):
    - GET /GalvoScannerController/getGalvoScannerNames
    - GET /GalvoScannerController/getGalvoScannerConfig
    - GET /GalvoScannerController/getGalvoScannerStatus
    - POST /GalvoScannerController/startGalvoScan
    - POST /GalvoScannerController/stopGalvoScan
    - POST /GalvoScannerController/setGalvoScanConfig
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self, tryInheritParent=True)
        
        # Check if galvo scanners are available
        if not hasattr(self._master, 'galvoScannersManager'):
            self.__logger.warning("No galvo scanners manager found in master controller")
            return
            
        if not self._master.galvoScannersManager.hasDevices():
            self.__logger.info("No galvo scanner devices configured")
            return
        
        self.__logger.info(f"GalvoScannerController initialized with devices: "
                          f"{self._master.galvoScannersManager.getAllDeviceNames()}")
        
        # Set up widget if not headless
        if not IS_HEADLESS:
            self._setupWidget()

    def _setupWidget(self):
        """Set up widget connections if available."""
        if self._widget is None:
            return
        # Connect widget signals here when widget is implemented
        pass

    # ========================
    # API Export Methods
    # ========================

    @APIExport()
    def getGalvoScannerNames(self) -> List[str]:
        """
        Get the names of all configured galvo scanner devices.
        
        Returns:
            List of galvo scanner device names
            
        Example:
            GET /api/GalvoScannerController/getGalvoScannerNames
            Response: ["ESP32Galvo"]
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return []
        return self._master.galvoScannersManager.getAllDeviceNames()

    @APIExport()
    def getGalvoScannerConfig(self, scannerName: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current configuration for a galvo scanner.
        
        Args:
            scannerName: Name of the galvo scanner. If None, uses the first available.
            
        Returns:
            Dictionary with scan configuration parameters:
            - nx: Number of X samples per line
            - ny: Number of Y lines
            - x_min, x_max: X position range (0-4095)
            - y_min, y_max: Y position range (0-4095)
            - sample_period_us: Microseconds per sample
            - frame_count: Number of frames (0=infinite)
            - bidirectional: Whether bidirectional scanning is enabled
            
        Example:
            GET /api/GalvoScannerController/getGalvoScannerConfig?scannerName=ESP32Galvo
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            return {
                "scannerName": scannerName,
                "config": scanner.get_config_dict()
            }
        except Exception as e:
            self.__logger.error(f"Error getting config for {scannerName}: {e}")
            return {"error": str(e)}

    @APIExport()
    def getGalvoScannerStatus(self, scannerName: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current status of a galvo scanner.
        
        Args:
            scannerName: Name of the galvo scanner. If None, uses the first available.
            
        Returns:
            Dictionary with status information:
            - running: Whether scanner is currently active
            - current_frame: Current frame number
            - current_line: Current line number
            - config: Current configuration
            
        Example:
            GET /api/GalvoScannerController/getGalvoScannerStatus
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            status = scanner.get_status()
            status["scannerName"] = scannerName
            return status
        except Exception as e:
            self.__logger.error(f"Error getting status for {scannerName}: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def setGalvoScanConfig(self, scannerName: Optional[str] = None,
                           nx: Optional[int] = None, ny: Optional[int] = None,
                           x_min: Optional[int] = None, x_max: Optional[int] = None,
                           y_min: Optional[int] = None, y_max: Optional[int] = None,
                           sample_period_us: Optional[int] = None,
                           frame_count: Optional[int] = None,
                           bidirectional: Optional[bool] = None,
                           pre_samples: Optional[int] = None,
                           fly_samples: Optional[int] = None,
                           trig_delay_us: Optional[int] = None,
                           trig_width_us: Optional[int] = None,
                           line_settle_samples: Optional[int] = None,
                           enable_trigger: Optional[int] = None,
                           apply_x_lut: Optional[int] = None) -> Dict[str, Any]:
        """
        Update the configuration for a galvo scanner without starting a scan.
        
        Args:
            scannerName: Name of the galvo scanner. If None, uses the first available.
            nx: Number of X samples per line (1-4096)
            ny: Number of Y lines (1-4096)
            x_min: Minimum X position (0-4095)
            x_max: Maximum X position (0-4095)
            y_min: Minimum Y position (0-4095)
            y_max: Maximum Y position (0-4095)
            sample_period_us: Microseconds per sample (0=max speed)
            frame_count: Number of frames (0=infinite)
            bidirectional: Enable bidirectional scanning
            pre_samples: Pre-scan samples
            fly_samples: Fly-back samples
            trig_delay_us: Trigger delay in microseconds
            trig_width_us: Trigger width in microseconds
            line_settle_samples: Line settling samples
            enable_trigger: Enable trigger output (0/1)
            apply_x_lut: Apply X lookup table (0/1)
            
        Returns:
            Updated configuration dictionary
            
        Example:
            POST /api/GalvoScannerController/setGalvoScanConfig?nx=512&ny=512&bidirectional=true
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        self.__logger.info(f"setGalvoScanConfig called with: scannerName={scannerName}, "
                           f"nx={nx}, ny={ny}, x_min={x_min}, x_max={x_max}, "
                           f"y_min={y_min}, y_max={y_max}, sample_period_us={sample_period_us}, "
                           f"frame_count={frame_count}, bidirectional={bidirectional}, "
                           f"pre_samples={pre_samples}, fly_samples={fly_samples}, "
                           f"trig_delay_us={trig_delay_us}, trig_width_us={trig_width_us}, "
                           f"line_settle_samples={line_settle_samples}, enable_trigger={enable_trigger}, "
                           f"apply_x_lut={apply_x_lut}")
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            
            # Update configuration with provided values
            scanner.update_config(
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
            
            self.__logger.info(f"Updated config for {scannerName}")
            return {
                "status": "config_updated",
                "scannerName": scannerName,
                "config": scanner.get_config_dict()
            }
        except Exception as e:
            self.__logger.error(f"Error setting config for {scannerName}: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def startGalvoScan(self, scannerName: Optional[str] = None,
                       nx: Optional[int] = None, ny: Optional[int] = None,
                       x_min: Optional[int] = None, x_max: Optional[int] = None,
                       y_min: Optional[int] = None, y_max: Optional[int] = None,
                       sample_period_us: Optional[int] = None,
                       frame_count: Optional[int] = None,
                       bidirectional: Optional[bool] = None,
                       pre_samples: Optional[int] = None,
                       fly_samples: Optional[int] = None,
                       trig_delay_us: Optional[int] = None,
                       trig_width_us: Optional[int] = None,
                       line_settle_samples: Optional[int] = None,
                       enable_trigger: Optional[int] = None,
                       apply_x_lut: Optional[int] = None,
                       timeout: int = 1) -> Dict[str, Any]:
        """
        Start a galvo scan with the specified parameters.
        
        Parameters that are not provided will use the current configuration values.
        
        Args:
            scannerName: Name of the galvo scanner. If None, uses the first available.
            nx: Number of X samples per line (1-4096)
            ny: Number of Y lines (1-4096)
            x_min: Minimum X position (0-4095)
            x_max: Maximum X position (0-4095)
            y_min: Minimum Y position (0-4095)
            y_max: Maximum Y position (0-4095)
            sample_period_us: Microseconds per sample (0=max speed)
            frame_count: Number of frames (0=infinite)
            bidirectional: Enable bidirectional scanning
            pre_samples: Pre-scan samples
            fly_samples: Fly-back samples
            trig_delay_us: Trigger delay in microseconds
            trig_width_us: Trigger width in microseconds
            line_settle_samples: Line settling samples
            enable_trigger: Enable trigger output (0/1)
            apply_x_lut: Apply X lookup table (0/1)
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with scan start status
            
        Example:
            POST /api/GalvoScannerController/startGalvoScan?nx=256&ny=256&frame_count=10
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            scanner.start_scan(
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
                apply_x_lut=apply_x_lut,
                timeout=timeout
            )
            
            self.__logger.info(f"Started scan on {scannerName}")
            result = {"status": "started", "scannerName": scannerName}
            return result
        except Exception as e:
            self.__logger.error(f"Error starting scan on {scannerName}: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def stopGalvoScan(self, scannerName: Optional[str] = None,
                      timeout: int = 1) -> Dict[str, Any]:
        """
        Stop an active galvo scan.
        
        Args:
            scannerName: Name of the galvo scanner. If None, uses the first available.
            timeout: Request timeout in seconds
            
        Returns:
            Dictionary with stop status
            
        Example:
            POST /api/GalvoScannerController/stopGalvoScan
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            scanner.stop_scan(timeout=timeout)
            
            self.__logger.info(f"Stopped scan on {scannerName}")
            result = {"status": "stopped", "scannerName": scannerName}
            return result
        except Exception as e:
            self.__logger.error(f"Error stopping scan on {scannerName}: {e}")
            return {"error": str(e)}

    @APIExport()
    def getAllGalvoScannersStatus(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all galvo scanners.
        
        Returns:
            Dictionary mapping scanner names to their status
            
        Example:
            GET /api/GalvoScannerController/getAllGalvoScannersStatus
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        return self._master.galvoScannersManager.get_all_status()

    @APIExport(runOnUIThread=True)
    def stopAllGalvoScans(self) -> Dict[str, Dict[str, Any]]:
        """
        Stop all active galvo scans.
        
        Returns:
            Dictionary mapping scanner names to their stop status
            
        Example:
            POST /api/GalvoScannerController/stopAllGalvoScans
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        return self._master.galvoScannersManager.stop_all_scans()

    # ========================
    # Helper Methods
    # ========================

    def _resolveScanner(self, scannerName: Optional[str]) -> Optional[str]:
        """
        Resolve the scanner name, returning the first available if None.
        
        Args:
            scannerName: Provided scanner name or None
            
        Returns:
            Resolved scanner name or None if no scanners available
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return None
        
        names = self._master.galvoScannersManager.getAllDeviceNames()
        if not names:
            return None
        
        if scannerName is None or scannerName not in names:
            return names[0]
        
        return scannerName


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
