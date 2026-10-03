"""
Galvo Scanner Controller for ImSwitch.

This controller provides the API interface for controlling galvo scanners,
exposing scan parameters and control functions via REST API.
"""

from typing import Dict, List, Optional, Any
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController
import base64
import io
import json
import math
import time

import numpy as np


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

        self._bindGalvoToFlimDetectors()
        self._autoStartScanners()

    def _autoStartScanners(self):
        """Start scanning at boot for scanners with autoStartScan enabled
        (default). The FLIM rig wants the raster + trigger pattern running from
        the start; disable with "autoStartScan": false in managerProperties.
        """
        for name in self._master.galvoScannersManager.getAllDeviceNames():
            scanner = self._master.galvoScannersManager[name]
            if not getattr(scanner, '_autoStartScan', False):
                continue
            try:
                scanner.start_scan()
                self.__logger.info(
                    f"Auto-started galvo scan on '{name}' with default config")
            except Exception as e:
                self.__logger.warning(f"Auto-start of '{name}' failed: {e}")

    def _bindGalvoToFlimDetectors(self):
        """Late-bind galvo scanners into FLIM detectors.

        Detectors are constructed before galvoScannersManager exists, so a
        FLIMLabsDetectorManager cannot resolve its ``galvoScanner`` property at
        construction time. Do it here, once both managers are up, so the FLIM
        detector can start/stop the scan around its own acquisitions.
        """
        try:
            detectors = self._master.detectorsManager
            scanners = self._master.galvoScannersManager
        except Exception:
            return
        for detName in detectors.getAllDeviceNames():
            det = detectors[detName]
            if not hasattr(det, 'setGalvoScanner'):
                continue
            scannerName = getattr(det, 'galvoScannerName', None)
            names = scanners.getAllDeviceNames()
            if scannerName is None and names:
                scannerName = names[0]
            if scannerName and scannerName in names:
                det.setGalvoScanner(scanners[scannerName])
                self.__logger.info(
                    f"Bound galvo scanner '{scannerName}' to detector '{detName}'")
                # A persisted camera<->scanner calibration already tells us
                # the µm per DAC count - hand it over so FOV/pixel size are
                # physical from the first frame on.
                self._propagateUmPerDacToFlim(scannerName)
            else:
                self.__logger.warning(
                    f"Detector '{detName}' requests galvo scanner '{scannerName}' "
                    "which is not configured")


    def _propagateGeometryToFlim(self, scannerName):
        """Push the current scan geometry into every FLIM detector bound to
        this scanner.

        The FLIM card has no geometry of its own - frame/line/pixel boundaries
        come from this scanner's trigger pattern. If nx/ny or the scanned DAC
        range changes while the card is armed for the old shape, lines fall
        outside the frame buffer and are dropped silently, so the live image
        just stops updating. Pushing it through keeps the two in step without
        re-arming the card (scouting applies it live).
        """
        results = {}
        try:
            detectors = self._master.detectorsManager
        except Exception:
            return results
        for detName in detectors.getAllDeviceNames():
            det = detectors[detName]
            if not hasattr(det, 'applyGeometryLive'):
                continue
            if getattr(det, 'galvoScannerName', None) not in (None, scannerName):
                continue
            try:
                results[detName] = det.applyGeometryLive()
            except Exception as e:
                self.__logger.warning(
                    f"Could not propagate geometry to '{detName}': {e}")
                results[detName] = {'error': str(e)}
        return results

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
                           apply_x_lut: Optional[int] = None,
                           overscan_samples: Optional[int] = None,
                           laser_blanking: Optional[int] = None,
                           hw_pixel_clock: Optional[int] = None) -> Dict[str, Any]:
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
                apply_x_lut=apply_x_lut,
                overscan_samples=overscan_samples,
                laser_blanking=laser_blanking,
                hw_pixel_clock=hw_pixel_clock
            )
            
            self.__logger.info(f"Updated config for {scannerName}")
            # The FLIM card derives its frame purely from this scan pattern, so
            # a geometry change has to reach it too - otherwise every line
            # lands out of range and the image silently goes blank.
            flim = self._propagateGeometryToFlim(scannerName)
            return {
                "status": "config_updated",
                "scannerName": scannerName,
                "config": scanner.get_config_dict(),
                "flimDetectors": flim,
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
                       overscan_samples: Optional[int] = None,
                       laser_blanking: Optional[int] = None,
                       hw_pixel_clock: Optional[int] = None,
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
                overscan_samples=overscan_samples,
                laser_blanking=laser_blanking,
                hw_pixel_clock=hw_pixel_clock,
                timeout=timeout
            )
            
            self.__logger.info(f"Started scan on {scannerName}")
            # start_scan() also commits any nx/ny/range overrides passed here,
            # so the FLIM detectors need the same update as setGalvoScanConfig.
            result = {"status": "started", "scannerName": scannerName,
                      "flimDetectors": self._propagateGeometryToFlim(scannerName)}
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

    @APIExport(runOnUIThread=True)
    def getGalvoParkConfig(self, scannerName: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the parking configuration (park_x, park_y, park_on_stop).

        Example:
            GET /api/GalvoScannerController/getGalvoParkConfig
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            cfg = scanner.get_park_config()
            cfg["scannerName"] = scannerName
            return cfg
        except Exception as e:
            self.__logger.error(f"Error getting park config for {scannerName}: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def setGalvoParkConfig(self, scannerName: Optional[str] = None,
                           park_x: Optional[int] = None,
                           park_y: Optional[int] = None,
                           park_on_stop: Optional[bool] = None) -> Dict[str, Any]:
        """
        Update the parking configuration.

        Args:
            park_x: Park X position in DAC counts (0-4095)
            park_y: Park Y position in DAC counts (0-4095)
            park_on_stop: Whether to move to the park position when a scan stops

        Example:
            POST /api/GalvoScannerController/setGalvoParkConfig?park_x=2048&park_y=2048&park_on_stop=true
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            cfg = scanner.set_park_config(park_x=park_x, park_y=park_y,
                                          park_on_stop=park_on_stop)
            cfg["scannerName"] = scannerName
            return cfg
        except Exception as e:
            self.__logger.error(f"Error setting park config for {scannerName}: {e}")
            return {"error": str(e)}

    @APIExport()
    def getFlimLabsConfig(self) -> Dict[str, Any]:
        """
        Return the persisted FLIM LABS bridge settings from the setup file.

        Returns an empty dict when nothing has been saved yet.

        Example:
            GET /api/GalvoScannerController/getFlimLabsConfig
        """
        cfg = getattr(self._setupInfo, 'flimLabs', None)
        return cfg if isinstance(cfg, dict) else {}

    @APIExport(requestType="POST")
    def setFlimLabsConfig(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Persist FLIM LABS bridge settings into the setup configuration file.

        The dict is stored as-is under the top-level ``flimLabs`` key and
        survives a restart. The frontend loads it on mount via
        getFlimLabsConfig.

        Example:
            POST /api/GalvoScannerController/setFlimLabsConfig
            body: {"host": "192.168.2.100", "frequencyMhz": 40, ...}
            (FastAPI maps the single dict parameter to the raw JSON body)
        """
        try:
            self._setupInfo.flimLabs = dict(config or {})
            import imswitch.imcontrol.model.configfiletools as configfiletools
            mOptions, _ = configfiletools.loadOptions()
            configfiletools.saveSetupInfo(mOptions, self._setupInfo)
            self.__logger.info("FLIM LABS bridge settings saved to setup file")
            return {"status": "saved", "config": self._setupInfo.flimLabs}
        except Exception as e:
            self.__logger.error(f"Failed to save FLIM LABS settings: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def parkGalvo(self, scannerName: Optional[str] = None,
                  timeout: int = 1) -> Dict[str, Any]:
        """
        Immediately move the beam to the configured park position.

        Example:
            POST /api/GalvoScannerController/parkGalvo
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            result = scanner.park(timeout=timeout)
            self.__logger.info(f"Parked {scannerName}")
            # The ESP32 answers with a LIST (e.g. [{'return': 1, 'qid': 51}]);
            # returning it raw fails FastAPI's Dict[str, Any] response model
            # with a 500 before the caller ever sees it.
            return {"status": "parked", "scannerName": scannerName,
                    "result": result}
        except Exception as e:
            self.__logger.error(f"Error parking {scannerName}: {e}")
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
    # Arbitrary Points API
    # ========================

    @APIExport(runOnUIThread=True)
    def setArbitraryPoints(self, points: str,
                           scannerName: Optional[str] = None,
                           laser_trigger: str = "AUTO",
                           apply_affine: bool = True,
                           timeout: int = 1) -> Dict[str, Any]:
        """
        Send arbitrary points to the galvo scanner.
        
        Points are given in camera coordinates if apply_affine is True (default),
        or in DAC coordinates (0-4095) if apply_affine is False.
        
        Args:
            points: JSON string of point list, each with:
                    - x (int): X coordinate
                    - y (int): Y coordinate
                    - dwell_us (int): Dwell time in microseconds
                    - laser_intensity (int, optional): 0-255
            scannerName: Scanner device name (optional)
            laser_trigger: Trigger mode - AUTO, HIGH, LOW, CONTINUOUS
            apply_affine: Whether to apply affine transform (camera→galvo)
            timeout: Request timeout
            
        Returns:
            dict: Status and transformed points
            
        Example:
            POST /api/GalvoScannerController/setArbitraryPoints?points=[{"x":100,"y":200,"dwell_us":500}]
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            # Parse points from JSON string
            if isinstance(points, str):
                point_list = json.loads(points)
            else:
                point_list = points
            
            if not isinstance(point_list, list):
                return {"error": "Points must be a JSON array"}
            
            if len(point_list) > 265:
                return {"error": f"Maximum 265 points supported, got {len(point_list)}"}
            
            scanner = self._master.galvoScannersManager[scannerName]
            
            # Apply affine transform if requested (camera coords → galvo DAC coords)
            if apply_affine:
                point_list = scanner.affine_transform.transform_points(point_list)
            
            result = scanner.set_arbitrary_points(
                points=point_list, laser_trigger=laser_trigger, timeout=timeout
            )
            
            self.__logger.info(f"Set {len(point_list)} arbitrary points on {scannerName}")
            return {
                "status": "points_set",
                "scannerName": scannerName,
                "num_points": len(point_list),
                "transformed_points": point_list,
                "result": result
            }
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON for points: {e}"}
        except Exception as e:
            self.__logger.error(f"Error setting arbitrary points: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def startArbitraryScan(self, points: str,
                           scannerName: Optional[str] = None,
                           laser_trigger: str = "AUTO",
                           apply_affine: bool = True,
                           timeout: int = 1) -> Dict[str, Any]:
        """
        Start arbitrary point scanning (alias for setArbitraryPoints).
        
        Args:
            points: JSON string of point list
            scannerName: Scanner device name
            laser_trigger: Trigger mode
            apply_affine: Apply affine transform
            timeout: Request timeout
            
        Returns:
            dict: Scan start status
        """
        return self.setArbitraryPoints(
            points=points, scannerName=scannerName,
            laser_trigger=laser_trigger, apply_affine=apply_affine,
            timeout=timeout
        )

    @APIExport(runOnUIThread=True)
    def stopArbitraryScan(self, scannerName: Optional[str] = None,
                          timeout: int = 1) -> Dict[str, Any]:
        """
        Stop arbitrary point scanning.
        
        Args:
            scannerName: Scanner device name
            timeout: Request timeout
            
        Returns:
            dict: Stop status
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            result = scanner.stop_arbitrary_scan(timeout=timeout)
            return {"status": "stopped", "scannerName": scannerName, "result": result}
        except Exception as e:
            self.__logger.error(f"Error stopping arbitrary scan: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def pauseArbitraryScan(self, scannerName: Optional[str] = None,
                           timeout: int = 1) -> Dict[str, Any]:
        """
        Pause arbitrary point scanning (keeps current index).
        
        Args:
            scannerName: Scanner device name
            timeout: Request timeout
            
        Returns:
            dict: Pause status
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            result = scanner.pause_arbitrary_scan(timeout=timeout)
            return {"status": "paused", "scannerName": scannerName, "result": result}
        except Exception as e:
            self.__logger.error(f"Error pausing arbitrary scan: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def resumeArbitraryScan(self, scannerName: Optional[str] = None,
                            timeout: int = 1) -> Dict[str, Any]:
        """
        Resume arbitrary point scanning from paused position.
        
        Args:
            scannerName: Scanner device name
            timeout: Request timeout
            
        Returns:
            dict: Resume status
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            result = scanner.resume_arbitrary_scan(timeout=timeout)
            return {"status": "resumed", "scannerName": scannerName, "result": result}
        except Exception as e:
            self.__logger.error(f"Error resuming arbitrary scan: {e}")
            return {"error": str(e)}

    @APIExport()
    def getArbitraryScanState(self, scannerName: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current state of arbitrary point scanning.
        
        Returns:
            dict: Scan state including running, paused, num_points, points
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            state = scanner.get_arbitrary_scan_state()
            state["scannerName"] = scannerName
            return state
        except Exception as e:
            self.__logger.error(f"Error getting arbitrary scan state: {e}")
            return {"error": str(e)}

    # ========================
    # Affine Transform API
    # ========================

    @APIExport()
    def getAffineTransform(self, scannerName: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current affine transformation matrix for camera-to-galvo mapping.
        
        Returns:
            dict: Affine transform with keys a11, a12, tx, a21, a22, ty
            
        Example:
            GET /api/GalvoScannerController/getAffineTransform
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
                "affine_transform": scanner.get_affine_transform_dict()
            }
        except Exception as e:
            self.__logger.error(f"Error getting affine transform: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def setAffineTransform(self, scannerName: Optional[str] = None,
                           a11: Optional[float] = None, a12: Optional[float] = None,
                           tx: Optional[float] = None, a21: Optional[float] = None,
                           a22: Optional[float] = None, ty: Optional[float] = None,
                           save: bool = True) -> Dict[str, Any]:
        """
        Set the affine transformation matrix for camera-to-galvo mapping.
        
        Args:
            scannerName: Scanner device name
            a11, a12, tx, a21, a22, ty: Affine matrix elements
            save: Whether to persist the transform to disk
            
        Returns:
            dict: Updated affine transform
            
        Example:
            POST /api/GalvoScannerController/setAffineTransform?a11=2.0&a22=2.0&tx=100&ty=200
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            transform = scanner.set_affine_transform(
                a11=a11, a12=a12, tx=tx, a21=a21, a22=a22, ty=ty
            )
            
            if save:
                config_path = scanner.save_affine_config()
                self.__logger.info(f"Saved affine config to {config_path}")
            
            return {
                "status": "transform_updated",
                "scannerName": scannerName,
                "affine_transform": transform,
                "saved": save,
                "flim": self._propagateUmPerDacToFlim(scannerName),
            }
        except Exception as e:
            self.__logger.error(f"Error setting affine transform: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def resetAffineTransform(self, scannerName: Optional[str] = None,
                             save: bool = True) -> Dict[str, Any]:
        """
        Reset affine transform to identity matrix.
        
        Args:
            scannerName: Scanner device name
            save: Whether to persist
            
        Returns:
            dict: Reset affine transform
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            scanner.reset_affine_transform()
            
            if save:
                scanner.save_affine_config()
            
            return {
                "status": "transform_reset",
                "scannerName": scannerName,
                "affine_transform": scanner.get_affine_transform_dict(),
                "flim": self._propagateUmPerDacToFlim(scannerName),
            }
        except Exception as e:
            self.__logger.error(f"Error resetting affine transform: {e}")
            return {"error": str(e)}

    @APIExport(runOnUIThread=True)
    def runAffineCalibration(self, calibration_data: str,
                             scannerName: Optional[str] = None,
                             save: bool = True) -> Dict[str, Any]:
        """
        Compute affine transform from calibration point pairs.
        
        The calibration_data must contain at least 3 corresponding pairs
        of camera pixel coordinates and galvo DAC coordinates.
        
        Args:
            calibration_data: JSON string with format:
                {
                    "cam_points": [[cx1,cy1], [cx2,cy2], [cx3,cy3]],
                    "galvo_points": [[gx1,gy1], [gx2,gy2], [gx3,gy3]]
                }
            scannerName: Scanner device name
            save: Whether to persist the computed transform
            
        Returns:
            dict: Computed affine transform
            
        Example:
            POST /api/GalvoScannerController/runAffineCalibration?calibration_data={"cam_points":[[100,100],[400,100],[100,400]],"galvo_points":[[500,500],[3500,500],[500,3500]]}
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        
        try:
            if isinstance(calibration_data, str):
                cal_data = json.loads(calibration_data)
            else:
                cal_data = calibration_data
            
            cam_points = [tuple(p) for p in cal_data.get('cam_points', [])]
            galvo_points = [tuple(p) for p in cal_data.get('galvo_points', [])]
            
            if len(cam_points) < 3 or len(galvo_points) < 3:
                return {"error": "At least 3 point pairs required for calibration"}
            
            if len(cam_points) != len(galvo_points):
                return {"error": "Number of camera and galvo points must match"}
            
            scanner = self._master.galvoScannersManager[scannerName]
            transform = scanner.compute_affine_from_calibration(cam_points, galvo_points)
            
            if save:
                config_path = scanner.save_affine_config()
                self.__logger.info(f"Saved calibration to {config_path}")
            
            return {
                "status": "calibration_complete",
                "scannerName": scannerName,
                "affine_transform": transform,
                "num_points_used": len(cam_points),
                "saved": save,
                "flim": self._propagateUmPerDacToFlim(scannerName),
            }
        except json.JSONDecodeError as e:
            return {"error": f"Invalid calibration data JSON: {e}"}
        except Exception as e:
            self.__logger.error(f"Error running affine calibration: {e}")
            return {"error": str(e)}

    @APIExport()
    def getCalibrationPoints(self, scannerName: Optional[str] = None) -> Dict[str, Any]:
        """
        Get suggested galvo positions for the 3-point calibration workflow.
        
        Returns 3 well-separated galvo DAC positions for calibration.
        
        Returns:
            dict: Three galvo positions for calibration
        """
        # Return 3 well-separated points across the DAC range
        return {
            "calibration_galvo_points": [
                {"x": 1024, "y": 1024, "label": "Top-Left"},
                {"x": 3072, "y": 1024, "label": "Top-Right"},
                {"x": 2048, "y": 3072, "label": "Bottom-Center"}
            ],
            "instructions": (
                "For each point: 1) Galvo moves to position, "
                "2) Laser turns on, 3) Click the bright spot in camera view, "
                "4) Click 'Confirm' to record the pair."
            )
        }

    # ========================
    # Camera <-> scanner calibration
    # ========================
    #
    # The affine transform maps camera pixels -> DAC counts. Together with the
    # camera's (calibrated) pixel size that is the only bridge between the
    # 0..4095 scanner units and micrometres at the sample, which is what the
    # raster tab needs to draw the camera image behind the scan region and to
    # label the scan range in µm.

    DAC_FULL_SCALE = 4096
    _BACKGROUND_MAX_DIM = 1024

    def _cameraDetectorNames(self) -> List[str]:
        """Detectors that *look at* the sample (widefield cameras).

        A FLIM/scan detector's frame IS the scan, so it can't serve as the
        background behind it; those are recognised by their galvo binding.
        """
        try:
            detectors = self._master.detectorsManager
        except Exception:
            return []
        return [name for name in detectors.getAllDeviceNames()
                if not hasattr(detectors[name], 'setGalvoScanner')]

    def _resolveCamera(self, detectorName: Optional[str]) -> Optional[str]:
        names = self._cameraDetectorNames()
        if not names:
            return None
        if detectorName in names:
            return detectorName
        try:
            current = self._master.detectorsManager.getCurrentDetectorName()
            if current in names:
                return current
        except Exception:
            pass
        return names[0]

    @staticmethod
    def _cameraPixelSizeUm(detector) -> tuple:
        """(x, y) µm per camera pixel; pixelSizeUm is [Z, Y, X]."""
        try:
            ps = list(detector.pixelSizeUm)
            x = float(ps[-1])
            y = float(ps[-2]) if len(ps) >= 2 else x
            if x <= 0 or y <= 0:
                raise ValueError
            return x, y
        except Exception:
            return 1.0, 1.0

    @staticmethod
    def _isIdentityAffine(affine: Dict[str, float]) -> bool:
        ident = {'a11': 1.0, 'a12': 0.0, 'tx': 0.0, 'a21': 0.0, 'a22': 1.0, 'ty': 0.0}
        return all(abs(float(affine.get(k, v)) - v) < 1e-9 for k, v in ident.items())

    @staticmethod
    def _umPerDacFromAffine(affine: Dict[str, float], pixelUmX: float,
                            pixelUmY: float) -> tuple:
        """µm at the sample per DAC count along scanner X and Y.

        Invert the 2x2 part of the camera->DAC affine: its columns are the
        camera-pixel displacement of the spot for +1 count on X resp. Y.
        Scale to µm with the (possibly anisotropic) camera pixel size.
        Returns (None, None) for a singular transform.
        """
        a11, a12 = float(affine['a11']), float(affine['a12'])
        a21, a22 = float(affine['a21']), float(affine['a22'])
        det = a11 * a22 - a12 * a21
        if abs(det) < 1e-12:
            return None, None
        dxX, dyX = a22 / det, -a21 / det     # +1 DAC on X -> camera (dx, dy)
        dxY, dyY = -a12 / det, a11 / det     # +1 DAC on Y
        return (math.hypot(dxX * pixelUmX, dyX * pixelUmY),
                math.hypot(dxY * pixelUmX, dyY * pixelUmY))

    def _flimDetectorsFor(self, scannerName: str):
        try:
            detectors = self._master.detectorsManager
        except Exception:
            return []
        out = []
        for detName in detectors.getAllDeviceNames():
            det = detectors[detName]
            if not hasattr(det, 'setUmPerDac'):
                continue
            if getattr(det, 'galvoScannerName', None) not in (None, scannerName):
                continue
            out.append((detName, det))
        return out

    def _propagateUmPerDacToFlim(self, scannerName: str) -> Dict[str, Any]:
        """Push the affine-derived µm/DAC into the FLIM detectors bound to
        this scanner, or revert them to their setup values when the affine
        is identity (uncalibrated)."""
        results = {}
        flim = self._flimDetectorsFor(scannerName)
        if not flim:
            return results
        try:
            scanner = self._master.galvoScannersManager[scannerName]
            affine = scanner.get_affine_transform_dict()
        except Exception as e:
            self.__logger.warning(f"Could not read affine for '{scannerName}': {e}")
            return results
        umX = umY = None
        if not self._isIdentityAffine(affine):
            camName = self._resolveCamera(None)
            if camName is not None:
                cam = self._master.detectorsManager[camName]
                umX, umY = self._umPerDacFromAffine(
                    affine, *self._cameraPixelSizeUm(cam))
        for detName, det in flim:
            try:
                results[detName] = det.setUmPerDac(umX, umY, source='affine')
            except Exception as e:
                self.__logger.warning(
                    f"Could not apply scanner calibration to '{detName}': {e}")
                results[detName] = {'error': str(e)}
        return results

    @APIExport()
    def getGalvoCameraCalibration(self, scannerName: Optional[str] = None,
                                  detectorName: Optional[str] = None) -> Dict[str, Any]:
        """
        Summarise the camera<->scanner calibration in physical units.

        Combines the affine transform (camera px -> DAC counts, from the
        3-point wizard) with the camera's pixel size to give µm per DAC count
        and the field of view / pixel size of the current raster scan. With an
        identity affine the scanner is *uncalibrated* and only the raw DAC
        numbers are meaningful.

        Args:
            scannerName: Scanner device name (default: first)
            detectorName: 2D camera to use (default: current/first camera)

        Example:
            GET /api/GalvoScannerController/getGalvoCameraCalibration
        """
        if not hasattr(self._master, 'galvoScannersManager'):
            return {"error": "No galvo scanners manager available"}
        scannerName = self._resolveScanner(scannerName)
        if scannerName is None:
            return {"error": "No galvo scanner available"}
        scanner = self._master.galvoScannersManager[scannerName]
        affine = scanner.get_affine_transform_dict()
        calibrated = not self._isIdentityAffine(affine)

        cameras = self._cameraDetectorNames()
        camName = self._resolveCamera(detectorName)
        info: Dict[str, Any] = {
            'scannerName': scannerName,
            'affine_transform': affine,
            'calibrated': calibrated,
            'cameraDetectors': cameras,
            'detectorName': camName,
            'frameWidth': None, 'frameHeight': None,
            'pixelSizeUmX': None, 'pixelSizeUmY': None,
            'umPerDacX': None, 'umPerDacY': None,
            'scan': None,
        }
        if camName is None:
            info['hint'] = 'No 2D camera configured - cannot relate DAC counts to µm.'
            return info

        cam = self._master.detectorsManager[camName]
        try:
            w, h = int(cam.shape[0]), int(cam.shape[1])
        except Exception:
            w = h = None
        psX, psY = self._cameraPixelSizeUm(cam)
        info.update({'frameWidth': w, 'frameHeight': h,
                     'pixelSizeUmX': psX, 'pixelSizeUmY': psY})
        if not calibrated:
            info['hint'] = ('Scanner is not calibrated to the camera: run the '
                            '3-point affine wizard (Arbitrary Points tab).')
            return info

        umX, umY = self._umPerDacFromAffine(affine, psX, psY)
        info['umPerDacX'], info['umPerDacY'] = umX, umY
        if umX is None:
            info['hint'] = 'Affine transform is singular - re-run the calibration.'
            return info
        cfg = scanner.config
        spanX = abs(int(cfg.x_max) - int(cfg.x_min))
        spanY = abs(int(cfg.y_max) - int(cfg.y_min))
        nx, ny = max(1, int(cfg.nx)), max(1, int(cfg.ny))
        info['scan'] = {
            'nx': nx, 'ny': ny,
            'x_min': cfg.x_min, 'x_max': cfg.x_max,
            'y_min': cfg.y_min, 'y_max': cfg.y_max,
            'fovUmX': umX * spanX, 'fovUmY': umY * spanY,
            'pixelUmX': umX * spanX / nx, 'pixelUmY': umY * spanY / ny,
            'fullScaleUmX': umX * self.DAC_FULL_SCALE,
            'fullScaleUmY': umY * self.DAC_FULL_SCALE,
        }
        # Rotation of the scanner's X axis as seen by the camera, for the eye
        a11, a21 = float(affine['a11']), float(affine['a21'])
        a12, a22 = float(affine['a12']), float(affine['a22'])
        det = a11 * a22 - a12 * a21
        info['rotationDeg'] = math.degrees(math.atan2(-a21 / det, a22 / det))
        info['flim'] = {name: det_.umPerDacInfo
                        for name, det_ in self._flimDetectorsFor(scannerName)}
        return info

    @APIExport(runOnUIThread=False)
    def snapGalvoCameraBackground(self, scannerName: Optional[str] = None,
                                  detectorName: Optional[str] = None,
                                  maxDim: int = 1024) -> Dict[str, Any]:
        """
        Grab one frame from the 2D camera as a PNG to draw behind the scan
        pattern preview.

        The frame is contrast-stretched (1st-99th percentile) to 8 bit and
        downsampled so its longer edge is at most ``maxDim`` px. The returned
        ``frameWidth``/``frameHeight`` are the camera frame's real size, i.e.
        the coordinate system the affine calibration was measured in;
        ``subsampling`` is the integer stride applied to get ``width`` x
        ``height``.

        Example:
            GET /api/GalvoScannerController/snapGalvoCameraBackground?maxDim=800
        """
        camName = self._resolveCamera(detectorName)
        if camName is None:
            return {"error": "No 2D camera configured"}
        cam = self._master.detectorsManager[camName]

        frame = None
        try:
            frame = cam.getLatestFrame()
        except Exception:
            frame = None
        if frame is None or getattr(frame, 'size', 0) == 0:
            # Live view isn't running - start the camera and wait for a frame
            try:
                cam.startAcquisition()
            except Exception as e:
                return {"error": f"Camera '{camName}' could not start: {e}"}
            deadline = time.time() + 3.0
            while time.time() < deadline:
                time.sleep(0.1)
                try:
                    frame = cam.getLatestFrame()
                except Exception:
                    frame = None
                if frame is not None and getattr(frame, 'size', 0) > 0:
                    break
        if frame is None or getattr(frame, 'size', 0) == 0:
            return {"error": f"No frame available from '{camName}'"}

        try:
            img = np.asarray(frame)
            if img.ndim == 3:
                img = img.mean(axis=-1)      # RGB -> gray for the backdrop
            img = np.squeeze(img)
            if img.ndim != 2:
                return {"error": f"Unexpected frame shape {img.shape}"}
            frameH, frameW = img.shape
            maxDim = max(64, int(maxDim or self._BACKGROUND_MAX_DIM))
            stride = max(1, int(math.ceil(max(frameH, frameW) / maxDim)))
            small = img[::stride, ::stride]
            lo, hi = np.percentile(small, (1, 99))
            if hi <= lo:
                hi = lo + 1
            scaled = np.clip((small.astype(np.float32) - lo) * 255.0 / (hi - lo),
                             0, 255).astype(np.uint8)
            from PIL import Image
            buf = io.BytesIO()
            Image.fromarray(scaled, mode='L').save(buf, format='PNG')
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
        except Exception as e:
            self.__logger.error(f"Failed to render camera background: {e}")
            return {"error": f"Failed to render camera background: {e}"}

        psX, psY = self._cameraPixelSizeUm(cam)
        return {
            'detectorName': camName,
            'image': f'data:image/png;base64,{b64}',
            'width': int(scaled.shape[1]), 'height': int(scaled.shape[0]),
            'frameWidth': int(frameW), 'frameHeight': int(frameH),
            'subsampling': stride,
            'pixelSizeUmX': psX, 'pixelSizeUmY': psY,
            'timestamp': time.time(),
        }

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
