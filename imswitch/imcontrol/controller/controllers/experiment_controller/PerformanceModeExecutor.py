"""
Performance Mode Executor for Experiment Controller

Handles high-performance scanning that executes directly on hardware/microcontroller
for time-critical operations with external triggering.
"""
import os
import time
import threading
import tifffile as tif
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from imswitch.imcommon.model import initLogger, dirtools


class PerformanceModeExecutor:
    """Manages performance mode (hardware-triggered) experiment execution."""

    def __init__(self, hardware_interface, save_dir: str = None):
        """
        Initialize performance mode executor.
        
        Args:
            hardware_interface: HardwareInterface instance
            save_dir: Directory to save files, uses default if None
        """
        self._logger = initLogger(self)
        self._hardware = hardware_interface
        
        # Setup save directory
        if save_dir is None:
            save_dir = dirtools.UserFileDirs.Data
        self.save_dir = os.path.join(save_dir, "ExperimentController")
        os.makedirs(self.save_dir, exist_ok=True)

        # Writer thread control
        self._writer_thread = None
        self._stop_writer_evt = threading.Event()
        
        # Fast stage scanning state
        self.fast_stage_scan_is_running = False

    def start_fast_stage_scan_acquisition(self,
                                        xstart: float = 0, 
                                        xstep: float = 500, 
                                        nx: int = 10,
                                        ystart: float = 0, 
                                        ystep: float = 500, 
                                        ny: int = 10,
                                        tsettle: float = 90, 
                                        tExposure: float = 50,
                                        illumination0: int = None, 
                                        illumination1: int = None,
                                        illumination2: int = None, 
                                        illumination3: int = None, 
                                        led: float = None):
        """
        Execute fast stage scan with hardware triggering.
        
        This couples 2D stage scan with external-trigger camera acquisition.
        Puts camera into external trigger mode and runs stage scanning with
        microcontroller TTL pulses for each grid coordinate.
        
        Args:
            xstart: Starting X position  
            xstep: X step size
            nx: Number of X steps
            ystart: Starting Y position
            ystep: Y step size  
            ny: Number of Y steps
            tsettle: Settling time at each position (ms)
            tExposure: Exposure time (ms)
            illumination0-3: Illumination channel intensities
            led: LED intensity
        """
        self.fast_stage_scan_is_running = True
        self._stop()  # Ensure all prior runs are stopped
        
        # Move to starting position
        self._hardware.move_stage_xy(posX=xstart, posY=ystart, relative=False)

        # Prepare camera for external triggering
        self._hardware.stop_detector_acquisition()
        self._hardware.set_trigger_source("External trigger")
        self._hardware.flush_detector_buffers()
        self._hardware.start_detector_acquisition()

        # Prepare illumination configuration
        illum_dict = {
            "illumination0": illumination0,
            "illumination1": illumination1,
            "illumination2": illumination2,
            "illumination3": illumination3,
            "led": led
        }

        # Count valid illumination channels
        n_illuminations = sum(
            val is not None and val > 0 
            for val in illum_dict.values()
        )
        n_scan = max(n_illuminations, 1)
        total_frames = nx * ny * n_scan
        
        self._logger.info(f"Stage-scan: {nx}×{ny} ({total_frames} frames)")

        # Generate metadata for each position
        metadata_list = self._generate_scan_metadata(
            xstart, xstep, nx, ystart, ystep, ny, illum_dict, n_illuminations
        )

        # Start writer thread
        self._start_writer_thread(total_frames, metadata_list)

        # Execute stage scan (blocks until finished)
        illumination_tuple = (
            illumination0 or 0, illumination1 or 0, 
            illumination2 or 0, illumination3 or 0
        ) if n_illuminations > 0 else (0, 0, 0, 0)
        
        self._hardware.mStage.start_stage_scanning(
            xstart=0, xstep=xstep, nx=nx,
            ystart=0, ystep=ystep, ny=ny,
            tsettle=tsettle, tExposure=tExposure,
            illumination=illumination_tuple, led=led or 0,
        )

    def stop_fast_stage_scan_acquisition(self):
        """Stop the stage scan acquisition and writer thread."""
        if self._hardware.mStage:
            self._hardware.mStage.stop_stage_scanning()
        self.fast_stage_scan_is_running = False
        self._logger.info("Stopping stage scan acquisition...")
        self._stop()
        self._logger.info("Stage scan acquisition stopped.")

    def _generate_scan_metadata(self, 
                               xstart: float, xstep: float, nx: int,
                               ystart: float, ystep: float, ny: int,
                               illum_dict: Dict[str, Optional[int]], 
                               n_illuminations: int) -> List[Dict[str, Any]]:
        """
        Generate metadata for each position in the scan.
        
        Args:
            xstart, xstep, nx: X axis scan parameters
            ystart, ystep, ny: Y axis scan parameters  
            illum_dict: Dictionary of illumination channel values
            n_illuminations: Number of valid illumination channels
            
        Returns:
            List of metadata dictionaries for each frame
        """
        def add_data_point(metadata_list: List[Dict], x: float, y: float, 
                          illumination_channel: str, illumination_value: int, 
                          running_number: int) -> List[Dict]:
            """Helper function to add metadata for each position."""
            metadata_list.append({
                "x": x,
                "y": y,
                "illuminationChannel": illumination_channel,
                "illuminationValue": illumination_value,
                "runningNumber": running_number
            })
            return metadata_list

        metadata_list = []
        running_number = 0
        
        for iy in range(ny):
            for ix in range(nx):
                x = xstart + ix * xstep
                y = ystart + iy * ystep
                
                # Snake pattern
                if iy % 2 == 1:
                    x = xstart + (nx - 1 - ix) * xstep

                # Generate frame(s) for this position
                if n_illuminations == 0:
                    running_number += 1
                    add_data_point(metadata_list, x, y, "default", -1, running_number)
                else:
                    # Take image for each illumination channel > 0
                    for channel, value in illum_dict.items():
                        if value is not None and value > 0:
                            running_number += 1
                            add_data_point(metadata_list, x, y, channel, value, running_number)
                            
        return metadata_list

    def _start_writer_thread(self, total_frames: int, metadata_list: List[Dict]):
        """
        Start the background writer thread.
        
        Args:
            total_frames: Expected number of frames to write
            metadata_list: Metadata for each frame
        """
        self._stop_writer_evt.clear()
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            args=(total_frames, metadata_list),
            daemon=True,
        )
        self._writer_thread.start()

    def _writer_loop(self, n_expected: int, metadata_list: List[Dict], 
                    min_period: float = 0.2):
        """
        Background writer loop that saves frames as they arrive.
        
        Uses non-blocking camera.getChunk() to get frames from ring buffer
        and saves them with metadata-based filenames.
        
        Args:
            n_expected: Expected number of frames to save
            metadata_list: Metadata for each frame 
            min_period: Minimum period between status updates
        """
        saved = 0
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(self.save_dir, timestamp + "_FastStageScan")
        os.makedirs(file_path, exist_ok=True)
        
        self._logger.info(f"Writer thread started, saving to {file_path}...")
        
        while saved < n_expected and not self._stop_writer_evt.is_set():
            frames, ids = self._hardware.get_detector_chunk()

            if frames.size == 0:  # No new frames yet
                time.sleep(0.005)
                continue

            for frame, fid in zip(frames, ids):
                if fid >= len(metadata_list):
                    self._logger.warning(f"Frame ID {fid} exceeds metadata list length")
                    break
                    
                current_metadata = metadata_list[fid]
                if not current_metadata:
                    self._logger.warning(f"Metadata for frame {fid} is empty")
                    break

                # Generate filename from metadata
                filename = self._generate_frame_filename(file_path, current_metadata, fid)
                
                try:
                    tif.imwrite(filename, frame)
                    saved += 1
                    self._logger.debug(f"Saved {saved}/{n_expected} as {filename}")
                except Exception as e:
                    self._logger.error(f"Error saving frame {fid}: {e}")

        self._logger.info(f"Writer thread finished ({saved} images).")
        self._cleanup_after_scan()

    def _generate_frame_filename(self, base_path: str, metadata: Dict[str, Any], 
                                frame_id: int) -> str:
        """
        Generate filename for a frame based on metadata.
        
        Args:
            base_path: Base directory path
            metadata: Frame metadata dictionary
            frame_id: Frame ID number
            
        Returns:
            Full path for the frame file
        """
        filename = (
            f"FastScan_{metadata['runningNumber']}_"
            f"x_{round(metadata['x'], 1)}_"
            f"y_{round(metadata['y'], 1)}_"
            f"illu-{metadata['illuminationChannel']}_"
            f"{metadata['illuminationValue']}"
            f"{frame_id:06d}.tif"
        )
        return os.path.join(base_path, filename)

    def _cleanup_after_scan(self):
        """Clean up after scan completion."""
        self._logger.info("Grid-scan completed and all images saved.")
        
        # Reset camera to continuous mode
        self._hardware.stop_detector_acquisition()
        self._hardware.set_trigger_source("Continuous")
        self._hardware.flush_detector_buffers()
        self._hardware.start_detector_acquisition()
        
        self._logger.info("Camera reset to continuous mode.")
        self.fast_stage_scan_is_running = False

    def _stop(self):
        """Abort the acquisition gracefully."""
        self._stop_writer_evt.set()
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=2)
        self._hardware.stop_detector_acquisition()

    def is_scan_running(self) -> bool:
        """Check if a fast stage scan is currently running."""
        return self.fast_stage_scan_is_running

    def wait_for_scan_completion(self, timeout: float = None):
        """
        Wait for current scan to complete.
        
        Args:
            timeout: Maximum time to wait in seconds, None for no timeout
        """
        start_time = time.time()
        while self.fast_stage_scan_is_running:
            if timeout and (time.time() - start_time) > timeout:
                self._logger.warning("Timeout waiting for scan completion")
                break
            time.sleep(0.1)