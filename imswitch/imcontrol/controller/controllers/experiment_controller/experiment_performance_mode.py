"""
Performance mode implementation for ExperimentController.

This module handles experiment execution where parameters are sent to hardware
directly for time-critical operations with hardware triggering.
"""

import os
import time
import threading
from typing import List, Dict, Any
from fastapi import HTTPException
import numpy as np

from .experiment_mode_base import ExperimentModeBase
from .ome_writer import OMEWriter


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

            # Check if single TIFF writing is enabled
            is_single_tiff_mode = getattr(self.controller, '_ome_write_single_tiff', False)
            file_writers = []

            if is_single_tiff_mode:
                # Set up OME writers similar to normal mode for single TIFF output
                file_writers = self._setup_ome_writers(
                    snake_tiles, illumination_intensities, experiment_params
                )

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

            # Finalize OME writers if they were created
            if file_writers:
                self._finalize_ome_writers(file_writers)

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

    def _setup_ome_writers(self,
                          snake_tiles: List[List[Dict]],
                          illumination_intensities: List[float],
                          experiment_params: Dict[str, Any]) -> List[OMEWriter]:
        """
        Set up OME writers for single TIFF output in performance mode.
        
        Args:
            snake_tiles: List of tiles containing scan points
            illumination_intensities: List of illumination values
            experiment_params: Dictionary containing experiment parameters
            
        Returns:
            List of OMEWriter instances
        """
        file_writers = []
        
        # Only create writers if single TIFF mode is enabled
        is_single_tiff_mode = getattr(self.controller, '_ome_write_single_tiff', False)
        if not is_single_tiff_mode:
            return file_writers

        self._logger.debug("Setting up OME writers for single TIFF output in performance mode")

        # Create experiment directory and file paths
        timeStamp, dirPath, mFileName = self.create_experiment_directory("performance_scan")
        
        # Create a single OME writer for all tiles in single TIFF mode
        experiment_name = f"0_performance_scan"
        m_file_path = os.path.join(dirPath, f"{mFileName}_{experiment_name}.ome.tif")
        self._logger.debug(f"Performance mode single TIFF path: {m_file_path}")
        
        # Create file paths
        file_paths = self.create_ome_file_paths(m_file_path.replace(".ome.tif", ""))
        
        # Calculate combined tile and grid parameters for all positions
        all_tiles = [tile for tiles in snake_tiles for tile in tiles]  # Flatten all tiles
        if hasattr(self.controller, 'mDetector') and hasattr(self.controller.mDetector, '_shape'):
            tile_shape = (self.controller.mDetector._shape[-1], self.controller.mDetector._shape[-2])
        else:
            tile_shape = (512, 512)  # Default shape
        grid_shape, grid_geometry = self.calculate_grid_parameters(all_tiles)
        
        # Create writer configuration for single TIFF mode
        n_channels = sum(np.array(illumination_intensities) > 0)
        writer_config = self.create_writer_config(
            write_tiff=False,  # Disable individual TIFF files
            write_zarr=getattr(self.controller, '_ome_write_zarr', True),
            write_stitched_tiff=False,  # Disable stitched TIFF
            write_tiff_single=True,  # Enable single TIFF writing
            min_period=0.1,
            n_time_points=1,
            n_z_planes=1,  # Performance mode typically single Z
            n_channels=n_channels
        )
        
        # Create single OME writer for all positions
        ome_writer = OMEWriter(
            file_paths=file_paths,
            tile_shape=tile_shape,
            grid_shape=grid_shape,
            grid_geometry=grid_geometry,
            config=writer_config,
            logger=self._logger
        )
        file_writers.append(ome_writer)
        
        return file_writers

    def _finalize_ome_writers(self, file_writers: List[OMEWriter]) -> None:
        """
        Finalize OME writers after scan completion.
        
        Args:
            file_writers: List of OMEWriter instances to finalize
        """
        for writer in file_writers:
            try:
                writer.finalize()
                self._logger.debug("OME writer finalized successfully")
            except Exception as e:
                self._logger.error(f"Error finalizing OME writer: {str(e)}")

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
        status = "running" if self._scan_running else "idle"
        return {
            "status": status,
            "running": self._scan_running,
            "mode": "performance"
        }

    def stop_scan(self) -> Dict[str, Any]:
        """
        Stop the performance mode scan.

        Returns:
            Dictionary with stop result
        """
        if self._scan_running:
            self._scan_running = False
            if self._scan_thread and self._scan_thread.is_alive():
                self._scan_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish
            return {"status": "stopped", "message": "Performance mode scan stopped"}
        else:
            return {"status": "not_running", "message": "No performance mode scan is running"}

    def force_stop_scan(self) -> Dict[str, Any]:
        """
        Force stop the performance mode scan.

        Returns:
            Dictionary with force stop result
        """
        self._scan_running = False
        if self._scan_thread and self._scan_thread.is_alive():
            # Don't wait for thread to finish gracefully in force stop
            pass
        return {"status": "force_stopped", "message": "Performance mode scan force stopped"}

    def pause_scan(self) -> Dict[str, Any]:
        """
        Pause is not supported in performance mode.

        Returns:
            Dictionary indicating pause is not supported
        """
        return {"status": "not_supported", "message": "Pause is not supported in performance mode"}

    def resume_scan(self) -> Dict[str, Any]:
        """
        Resume is not supported in performance mode.

        Returns:
            Dictionary indicating resume is not supported
        """
        return {"status": "not_supported", "message": "Resume is not supported in performance mode"}
