"""
Galvo Scanner Controller for ImSwitch.

This controller provides the API interface for controlling galvo scanners,
exposing scan parameters and control functions via REST API.
"""

from typing import Dict, List, Optional, Any
from imswitch import IS_HEADLESS
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController
import json


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
                "saved": save
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
                "affine_transform": scanner.get_affine_transform_dict()
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
                "saved": save
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
