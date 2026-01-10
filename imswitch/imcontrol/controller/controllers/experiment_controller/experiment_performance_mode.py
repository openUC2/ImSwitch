"""
Performance mode implementation for ExperimentController.

This module handles experiment execution where parameters are sent to hardware
directly for time-critical operations with hardware triggering.
"""

import os
import time
import threading
from typing import List, Dict, Any, Optional
from fastapi import HTTPException
import numpy as np

from .experiment_mode_base import ExperimentModeBase
from .ome_writer import OMEWriter


# Default timeout multiplier for scan operations
SCAN_TIMEOUT_MULTIPLIER = 1.25
# Default margin for exposure time calculations
EXPOSURE_TIME_MARGIN = 1.25
# Frame timeout - if no frames received within this time, abort scan
FRAME_TIMEOUT_SECONDS = 10.0


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
        self._last_frame_time = None
        self._frame_count = 0
        self._expected_frames = 0
        self._use_software_trigger = False  # Option to use software trigger via callback
        self._camera_trigger_callback_registered = False

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
                # Calculate timeout based on scan parameters
                scan_timeout = self._calculate_scan_timeout(snake_tile, illumination_intensities, experiment_params)
                
                # Stop any previous fast stage scan if still running (with timeout)
                self._stop_previous_scan_with_timeout(scan_timeout)

                # Compute scan parameters
                scan_params = self._compute_scan_parameters(snake_tile, illumination_intensities, experiment_params)

                # Validate scan parameters
                if scan_params['nx'] > 100 or scan_params['ny'] > 100:
                    self._logger.error("Too many points in X/Y direction. Please reduce the number of points.")
                    return

                # Switch off all illumination levels before starting scan
                self._switch_off_all_illumination()

                # Execute fast stage scan
                zarr_url = self._execute_fast_stage_scan(scan_params, t_period, n_times, experiment_params)
                self._logger.info(f"Performance mode scan completed. Data saved to: {zarr_url}")

            # Finalize OME writers if they were created
            if file_writers:
                self._finalize_ome_writers(file_writers)

            # Reset camera to software triggering after scan
            self._reset_camera_to_software_trigger()

            # Set LED status to idle when scan completes successfully
            self.controller.set_led_status("idle")

        except Exception as e:
            self._logger.error(f"Error in performance mode scan: {str(e)}")
            # Set LED status to error
            self.controller.set_led_status("error")
            # Ensure camera is reset on error
            self._reset_camera_to_software_trigger()
        finally:
            self._scan_running = False

    def _calculate_scan_timeout(self,
                               snake_tile: List[Dict],
                               illumination_intensities: List[float],
                               experiment_params: Dict[str, Any]) -> float:
        """
        Calculate timeout for scan operation based on parameters.
        
        Formula: nScans * nChannels * (tPre + tPost) * 1.25
        
        Args:
            snake_tile: Single tile containing scan points
            illumination_intensities: List of illumination values
            experiment_params: Dictionary containing experiment parameters
            
        Returns:
            Timeout in seconds
        """
        # Calculate number of positions
        n_positions = len(snake_tile) if snake_tile else 1
        
        # Calculate number of active channels
        n_channels = sum(1 for i in illumination_intensities if i > 0) or 1
        
        # Get timing parameters
        m_experiment = experiment_params.get('mExperiment')
        if m_experiment and hasattr(m_experiment, 'parameterValue'):
            exposure_times = getattr(m_experiment.parameterValue, 'illuExposures', [50])
            if not isinstance(exposure_times, list):
                exposure_times = [exposure_times]
            t_exposure = max(exposure_times) if exposure_times else 50
        else:
            t_exposure = 50
        
        t_settle = experiment_params.get('tSettle', 90)
        
        # Calculate timeout with margin
        timeout_ms = n_positions * n_channels * (t_settle + t_exposure) * SCAN_TIMEOUT_MULTIPLIER
        timeout_seconds = timeout_ms / 1000.0
        
        # Add minimum timeout
        return max(timeout_seconds, FRAME_TIMEOUT_SECONDS)

    def _stop_previous_scan_with_timeout(self, timeout: float) -> None:
        """
        Wait for previous fast stage scan to finish, with timeout.
        If timeout is reached, force stop the scan.
        
        Args:
            timeout: Maximum time to wait in seconds
        """
        start_time = time.time()
        
        while self.controller.fastStageScanIsRunning:
            elapsed = time.time() - start_time
            
            if elapsed > timeout:
                self._logger.warning(f"Previous scan timeout after {elapsed:.1f}s. Force stopping...")
                try:
                    self.controller.mStage.stop_stage_scanning()
                    self.controller.fastStageScanIsRunning = False
                except Exception as e:
                    self._logger.error(f"Error stopping previous scan: {e}")
                break
                
            # Check if frames are being produced
            if self._last_frame_time is not None:
                time_since_last_frame = time.time() - self._last_frame_time
                if time_since_last_frame > FRAME_TIMEOUT_SECONDS:
                    self._logger.warning(
                        f"No frames received for {time_since_last_frame:.1f}s. "
                        "Hardware trigger may not be working. Stopping scan..."
                    )
                    try:
                        self.controller.mStage.stop_stage_scanning()
                        self.controller.fastStageScanIsRunning = False
                    except Exception as e:
                        self._logger.error(f"Error stopping scan due to frame timeout: {e}")
                    break
            
            self._logger.debug("Waiting for previous fast stage scan to finish...")
            time.sleep(0.1)

    def _switch_off_all_illumination(self) -> None:
        """
        Turn off all illumination sources before starting scan.
        This ensures clean state for hardware-controlled illumination.
        """
        try:
            # Try to access laser manager
            if hasattr(self.controller, '_master') and hasattr(self.controller._master, 'lasersManager'):
                for laser_name in self.controller._master.lasersManager.getAllDeviceNames():
                    try:
                        self.controller._master.lasersManager[laser_name].setEnabled(False)
                        self.controller._master.lasersManager[laser_name].setValue(0)
                    except Exception as e:
                        self._logger.debug(f"Could not turn off laser {laser_name}: {e}")
            
            # Try to turn off LED via UC2 interface
            if hasattr(self.controller, 'mStage') and hasattr(self.controller.mStage, '_motor'):
                try:
                    # Send zero illumination to all channels
                    esp32 = self.controller.mStage._rs232manager._esp32
                    if hasattr(esp32, 'led'):
                        esp32.led.send_LEDMatrix_array(intensity=0, ids=[i for i in range(64)])
                except Exception as e:
                    self._logger.debug(f"Could not turn off LED matrix: {e}")
                    
            self._logger.debug("All illumination sources switched off before scan")
        except Exception as e:
            self._logger.warning(f"Error switching off illumination: {e}")

    def _reset_camera_to_software_trigger(self) -> None:
        """
        Reset camera to software/continuous triggering after scan completion.
        """
        try:
            if hasattr(self.controller, 'mDetector'):
                self.controller.mDetector.stopAcquisition()
                self.controller.mDetector.setTriggerSource("Continuous")
                self.controller.mDetector.flushBuffers()
                self.controller.mDetector.startAcquisition()
                self._logger.debug("Camera reset to continuous/software trigger mode")
        except Exception as e:
            self._logger.error(f"Error resetting camera trigger: {e}")

    def _calculate_tpre_from_exposures(self, experiment_params: Dict[str, Any]) -> float:
        """
        Calculate tPre (settle time) from maximum exposure time with margin.
        
        tPre = max(exposure_times) * 1.25
        
        Args:
            experiment_params: Dictionary containing experiment parameters
            
        Returns:
            tPre value in milliseconds
        """
        m_experiment = experiment_params.get('mExperiment')
        
        if m_experiment and hasattr(m_experiment, 'parameterValue'):
            exposure_times = getattr(m_experiment.parameterValue, 'illuExposures', [])
            if not isinstance(exposure_times, list):
                exposure_times = [exposure_times] if exposure_times else []
            
            if exposure_times:
                max_exposure = max(exposure_times)
                t_pre = max_exposure * EXPOSURE_TIME_MARGIN
                self._logger.debug(f"Calculated tPre from max exposure {max_exposure}ms: {t_pre}ms")
                return t_pre
        
        # Default settle time if no exposure info available
        return 90

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

        # Extract Z-stack parameters
        z_params = self._extract_z_stack_parameters(experiment_params)
        
        # Calculate tPre from maximum exposure time
        t_pre = self._calculate_tpre_from_exposures(experiment_params)

        return {
            'xstart': xStart,
            'xstep': xStep,
            'nx': nx,
            'ystart': yStart,
            'ystep': yStep,
            'ny': ny,
            'zstart': z_params['zstart'],
            'zstep': z_params['zstep'],
            'nz': z_params['nz'],
            'tsettle': t_pre,
            'illumination0': illum_params['illumination0'],
            'illumination1': illum_params['illumination1'],
            'illumination2': illum_params['illumination2'],
            'illumination3': illum_params['illumination3'],
            'illumination4': illum_params['illumination4'],
            'led': led_value
        }

    def _extract_z_stack_parameters(self, experiment_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract Z-stack parameters from experiment configuration.
        
        Args:
            experiment_params: Dictionary containing experiment parameters
            
        Returns:
            Dictionary with zstart, zstep, nz
        """
        m_experiment = experiment_params.get('mExperiment')
        
        if m_experiment and hasattr(m_experiment, 'parameterValue'):
            param_value = m_experiment.parameterValue
            
            # Check if Z-stack is enabled
            z_stack_enabled = getattr(param_value, 'zStackEnabled', False)
            
            if z_stack_enabled:
                z_start = getattr(param_value, 'zStackStart', 0)
                z_step = getattr(param_value, 'zStackStep', 0)
                n_z = getattr(param_value, 'zStackSteps', 1)
                
                return {
                    'zstart': z_start,
                    'zstep': z_step,
                    'nz': max(1, n_z)
                }
        
        # Default: no Z-stacking
        return {
            'zstart': 0,
            'zstep': 0,
            'nz': 1
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
                               n_times: int,
                               experiment_params: Dict[str, Any]) -> str:
        """
        Execute the fast stage scan with hardware triggering.

        Args:
            scan_params: Dictionary with scan parameters
            t_period: Period between scans
            n_times: Number of time points
            experiment_params: Full experiment parameters

        Returns:
            OME-Zarr URL for the saved data
        """
        # Move to initial position first
        self.controller.move_stage_xy(
            posX=scan_params['xstart'],
            posY=scan_params['ystart'],
            relative=False
        )

        # Get exposure time
        m_experiment = experiment_params.get('mExperiment')
        if m_experiment and hasattr(m_experiment, 'parameterValue'):
            exposure_times = getattr(m_experiment.parameterValue, 'illuExposures', [50])
            if isinstance(exposure_times, list) and exposure_times:
                t_exposure = exposure_times[0]
            else:
                t_exposure = exposure_times if exposure_times else 50
        else:
            t_exposure = 50

        # Build illumination tuple
        illumination = (
            scan_params.get('illumination0') or 0,
            scan_params.get('illumination1') or 0,
            scan_params.get('illumination2') or 0,
            scan_params.get('illumination3') or 0
        )

        # Reset frame tracking
        self._last_frame_time = time.time()
        self._frame_count = 0
        self._expected_frames = scan_params['nx'] * scan_params['ny'] * scan_params['nz']
        
        # Register camera trigger callback if using software trigger mode
        if self._use_software_trigger:
            self._register_camera_trigger_callback()

        # Execute the fast stage scan acquisition with Z-stacking support
        zarr_url = self.controller.startFastStageScanAcquisition(
            xstart=scan_params['xstart'],
            xstep=scan_params['xstep'],
            nx=scan_params['nx'],
            ystart=scan_params['ystart'],
            ystep=scan_params['ystep'],
            ny=scan_params['ny'],
            zstart=scan_params.get('zstart', 0),
            zstep=scan_params.get('zstep', 0),
            nz=scan_params.get('nz', 1),
            tsettle=scan_params['tsettle'],
            tExposure=t_exposure,
            illumination0=scan_params['illumination0'],
            illumination1=scan_params['illumination1'],
            illumination2=scan_params['illumination2'],
            illumination3=scan_params['illumination3'],
            illumination4=scan_params['illumination4'],
            led=scan_params['led'],
            tPeriod=t_period,
            nTimes=n_times
        )

        return zarr_url

    def _register_camera_trigger_callback(self) -> None:
        """
        Register callback for camera trigger signal from firmware.
        This enables software-triggered acquisition based on hardware events.
        """
        if self._camera_trigger_callback_registered:
            return
            
        try:
            if hasattr(self.controller, 'mStage') and hasattr(self.controller.mStage, '_rs232manager'):
                esp32 = self.controller.mStage._rs232manager._esp32
                if hasattr(esp32, 'camera_trigger'):
                    esp32.camera_trigger.register_callback(0, self._on_camera_trigger)
                    self._camera_trigger_callback_registered = True
                    self._logger.debug("Camera trigger callback registered")
        except Exception as e:
            self._logger.warning(f"Could not register camera trigger callback: {e}")

    def _unregister_camera_trigger_callback(self) -> None:
        """
        Unregister camera trigger callback.
        """
        try:
            if hasattr(self.controller, 'mStage') and hasattr(self.controller.mStage, '_rs232manager'):
                esp32 = self.controller.mStage._rs232manager._esp32
                if hasattr(esp32, 'camera_trigger'):
                    esp32.camera_trigger.unregister_callback(0)
                    self._camera_trigger_callback_registered = False
        except Exception as e:
            self._logger.debug(f"Could not unregister camera trigger callback: {e}")

    def _on_camera_trigger(self, trigger_info: Dict[str, Any]) -> None:
        """
        Callback function for camera trigger events from firmware.
        
        Args:
            trigger_info: Dictionary with trigger information
        """
        self._last_frame_time = time.time()
        self._frame_count += 1
        
        if self._use_software_trigger:
            # Trigger software capture
            try:
                if hasattr(self.controller, 'mDetector'):
                    self.controller.mDetector.softwareTrigger()
                    self._logger.debug(f"Software trigger sent for frame {self._frame_count}")
            except Exception as e:
                self._logger.error(f"Error during software trigger: {e}")

    def set_software_trigger_mode(self, enabled: bool) -> None:
        """
        Enable or disable software trigger mode.
        
        When enabled, the camera will be triggered via software when the
        firmware sends {"cam":1} signals, instead of relying on hardware triggers.
        
        Args:
            enabled: True to use software triggering, False for hardware triggering
        """
        self._use_software_trigger = enabled
        self._logger.info(f"Software trigger mode {'enabled' if enabled else 'disabled'}")
        
        if enabled:
            self._register_camera_trigger_callback()
        else:
            self._unregister_camera_trigger_callback()

    def _setup_ome_writers(self,
                          snake_tiles: List[List[Dict]],
                          illumination_intensities: List[float],
                          experiment_params: Dict[str, Any]) -> List[OMEWriter]:
        """
        Set up OME writers for single TIFF output in performance mode.
        
        This method mirrors the setup in normal mode to ensure consistent
        file saving interface for timelapses and other experiments.
        
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

        # Create shared individual_tiffs directory at the experiment root level
        shared_individual_tiffs_dir = os.path.join(dirPath, "individual_tiffs")
        os.makedirs(shared_individual_tiffs_dir, exist_ok=True)

        # Create a single OME writer for all tiles in single TIFF mode
        experiment_name = "0_performance_scan"
        m_file_path = os.path.join(dirPath, f"{mFileName}_{experiment_name}.ome.tif")
        self._logger.debug(f"Performance mode single TIFF path: {m_file_path}")

        # Create file paths with shared individual_tiffs directory
        file_paths = self.create_ome_file_paths(m_file_path.replace(".ome.tif", ""), shared_individual_tiffs_dir)

        # Calculate combined tile and grid parameters for all positions
        all_tiles = [tile for tiles in snake_tiles for tile in tiles]  # Flatten all tiles
        if hasattr(self.controller, 'mDetector') and hasattr(self.controller.mDetector, '_shape'):
            tile_shape = (self.controller.mDetector._shape[-1], self.controller.mDetector._shape[-2])
        else:
            tile_shape = (512, 512)  # Default shape
        grid_shape, grid_geometry = self.calculate_grid_parameters(all_tiles)

        # Extract Z-stack parameters for writer config
        z_params = self._extract_z_stack_parameters(experiment_params)
        n_z_planes = z_params['nz']
        
        # Get time points
        n_time_points = experiment_params.get('nTimes', 1)

        # Create writer configuration for single TIFF mode
        n_channels = sum(np.array(illumination_intensities) > 0) or 1
        writer_config = self.create_writer_config(
            write_tiff=False,  # Disable individual TIFF files
            write_zarr=getattr(self.controller, '_ome_write_zarr', True),
            write_stitched_tiff=False,  # Disable stitched TIFF
            write_tiff_single=True,  # Enable single TIFF writing
            write_individual_tiffs=getattr(self.controller, '_ome_write_individual_tiffs', False),
            min_period=0.1,
            n_time_points=n_time_points,
            n_z_planes=n_z_planes,
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
            "mode": "performance",
            "frames_received": self._frame_count,
            "expected_frames": self._expected_frames,
            "software_trigger_mode": self._use_software_trigger
        }

    def stop_scan(self) -> Dict[str, Any]:
        """
        Stop the performance mode scan.

        Returns:
            Dictionary with stop result
        """
        if self._scan_running:
            self._scan_running = False
            
            # Stop hardware scan
            try:
                self.controller.mStage.stop_stage_scanning()
            except Exception as e:
                self._logger.warning(f"Error stopping stage scan: {e}")
            
            if self._scan_thread and self._scan_thread.is_alive():
                self._scan_thread.join(timeout=5.0)  # Wait up to 5 seconds for thread to finish

            # Reset camera to software trigger
            self._reset_camera_to_software_trigger()

            # Set LED status to idle
            self.controller.set_led_status("idle")

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
        
        # Force stop hardware
        try:
            self.controller.mStage.stop_stage_scanning()
            self.controller.fastStageScanIsRunning = False
        except Exception as e:
            self._logger.warning(f"Error force stopping scan: {e}")
        
        # Reset camera
        self._reset_camera_to_software_trigger()
        
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
