"""
Performance mode implementation for ExperimentController.

This module handles experiment execution where parameters are sent to hardware
directly for time-critical operations with hardware triggering.
"""

import time
import threading
from typing import List, Dict, Any
from fastapi import HTTPException

from .experiment_mode_base import ExperimentModeBase


class ExperimentPerformanceMode(ExperimentModeBase):
    """
    Performance mode experiment execution.
    
    In performance mode, the microcontroller handles stage movement, triggering,
    and illumination directly for optimal timing performance. ImSwitch mainly
    listens to the camera and stores images.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._scan_thread = None
        self._scan_running = False
    
    def execute_experiment(self, 
                         snake_tiles: List[List[Dict]],
                         illumination_intensities: List[float],
                         experiment_params: Dict[str, Any],
                         **kwargs) -> Dict[str, Any]:
        """
        Execute experiment in performance mode.
        
        Args:
            snake_tiles: List of tiles containing scan points
            illumination_intensities: List of illumination values
            experiment_params: Dictionary containing experiment parameters
            **kwargs: Additional parameters
            
        Returns:
            Dictionary with execution results
        """
        self._logger.debug("Performance mode is enabled. Executing on hardware directly.")
        
        # Start the scan in a background thread to make it non-blocking
        if self._scan_running:
            raise HTTPException(status_code=400, detail="Performance mode scan is already running.")
        
        # Start background thread to execute the scan
        self._scan_thread = threading.Thread(
            target=self._execute_scan_background,
            args=(snake_tiles, illumination_intensities, experiment_params),
            daemon=True
        )
        self._scan_running = True
        self._scan_thread.start()
        
        return {"status": "running", "mode": "performance"}
    
    def _execute_scan_background(self, 
                               snake_tiles: List[List[Dict]],
                               illumination_intensities: List[float],
                               experiment_params: Dict[str, Any]) -> None:
        """
        Execute the scan in background thread.
        
        Args:
            snake_tiles: List of tiles containing scan points
            illumination_intensities: List of illumination values
            experiment_params: Dictionary containing experiment parameters
        """
        try:
            t_period = experiment_params.get('tPeriod', 1)
            n_times = experiment_params.get('nTimes', 1)
            
            for snake_tile in snake_tiles:
                # Wait if another fast stage scan is running
                while self.controller.fastStageScanIsRunning:
                    self._logger.debug("Waiting for fast stage scan to finish...")
                    time.sleep(0.1)
                    
                # Compute scan parameters
                scan_params = self._compute_scan_parameters(snake_tile, illumination_intensities, experiment_params)
                
                # Validate scan parameters
                if scan_params['nx'] > 100 or scan_params['ny'] > 100:
                    self._logger.error("Too many points in X/Y direction. Please reduce the number of points.")
                    return
                
                # Execute fast stage scan
                zarr_url = self._execute_fast_stage_scan(scan_params, t_period, n_times)
                self._logger.info(f"Performance mode scan completed. Data saved to: {zarr_url}")
            
        except Exception as e:
            self._logger.error(f"Error in performance mode scan: {str(e)}")
        finally:
            self._scan_running = False
    
    def _compute_scan_parameters(self, 
                                snake_tile: List[Dict],
                                illumination_intensities: List[float],
                                experiment_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compute scan parameters for hardware execution.
        
        Args:
            snake_tile: Single tile containing scan points
            illumination_intensities: List of illumination values
            experiment_params: Dictionary containing experiment parameters
            
        Returns:
            Dictionary with computed scan parameters
        """
        # Compute scan ranges
        xStart, xEnd, yStart, yEnd, xStep, yStep = self.compute_scan_ranges([snake_tile])
        
        # Calculate number of steps
        nx = int((xEnd - xStart) // xStep) + 1 if xStep != 0 else 1
        ny = int((yEnd - yStart) // yStep) + 1 if yStep != 0 else 1
        
        # Prepare illumination parameters
        illum_params = self.prepare_illumination_parameters(illumination_intensities)
        
        # Handle LED parameter if present
        led_value = self._extract_led_value(experiment_params)
        
        return {
            'xstart': xStart,
            'xstep': xStep,
            'nx': nx,
            'ystart': yStart,
            'ystep': yStep,
            'ny': ny,
            'illumination0': illum_params['illumination0'],
            'illumination1': illum_params['illumination1'],
            'illumination2': illum_params['illumination2'],
            'illumination3': illum_params['illumination3'],
            'led': led_value
        }
    
    def _extract_led_value(self, experiment_params: Dict[str, Any]) -> float:
        """
        Extract LED value from experiment parameters.
        
        Args:
            experiment_params: Dictionary containing experiment parameters
            
        Returns:
            LED intensity value (0-255)
        """
        m_experiment = experiment_params.get('mExperiment')
        if not m_experiment:
            return 0
            
        illumination_sources = getattr(m_experiment.parameterValue, 'illumination', [])
        illumination_intensities = getattr(m_experiment.parameterValue, 'illuIntensities', [])
        
        if not isinstance(illumination_sources, list):
            illumination_sources = [illumination_sources] if illumination_sources else []
        if not isinstance(illumination_intensities, list):
            illumination_intensities = [illumination_intensities] if illumination_intensities else []
            
        # Find LED index
        led_index = next((i for i, item in enumerate(illumination_sources) 
                         if item and "led" in item.lower()), None)
        
        if led_index is not None and led_index < len(illumination_intensities):
            # Limit LED intensity to 255
            return min(illumination_intensities[led_index], 255)
        
        return 0
    
    def _execute_fast_stage_scan(self, 
                               scan_params: Dict[str, Any],
                               t_period: float,
                               n_times: int) -> str:
        """
        Execute the fast stage scan with hardware triggering.
        
        Args:
            scan_params: Dictionary with scan parameters
            t_period: Period between scans
            n_times: Number of time points
            
        Returns:
            OME-Zarr URL for the saved data
        """
        # Move to initial position first
        self.controller.move_stage_xy(
            posX=scan_params['xstart'], 
            posY=scan_params['ystart'], 
            relative=False
        )
        
        # Execute the fast stage scan acquisition
        zarr_url = self.controller.startFastStageScanAcquisition(
            xstart=scan_params['xstart'],
            xstep=scan_params['xstep'],
            nx=scan_params['nx'],
            ystart=scan_params['ystart'],
            ystep=scan_params['ystep'],
            ny=scan_params['ny'],
            tsettle=90,  # TODO: make these parameters adjustable
            tExposure=50,  # TODO: make these parameters adjustable
            illumination0=scan_params['illumination0'],
            illumination1=scan_params['illumination1'],
            illumination2=scan_params['illumination2'],
            illumination3=scan_params['illumination3'],
            led=scan_params['led'],
            tPeriod=t_period,
            nTimes=n_times
        )
        
        return zarr_url
    
    def is_hardware_capable(self) -> bool:
        """
        Check if hardware supports performance mode execution.
        
        Returns:
            True if hardware supports performance mode, False otherwise
        """
        return (hasattr(self.controller.mStage, "start_stage_scanning") and 
                hasattr(self.controller.mDetector, "setTriggerSource"))
    
    def is_scan_running(self) -> bool:
        """
        Check if a performance mode scan is currently running.
        
        Returns:
            True if scan is running, False otherwise
        """
        return self._scan_running
    
    def get_scan_status(self) -> Dict[str, Any]:
        """
        Get the current status of the performance mode scan.
        
        Returns:
            Dictionary with scan status information
        """
        return {
            "running": "idle" if self._scan_running else "running",
            "mode": "performance"
        }