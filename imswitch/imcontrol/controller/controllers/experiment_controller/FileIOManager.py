"""
File I/O Manager for Experiment Controller

Centralizes all file operations including different file formats (TIFF, OME-Zarr, etc.)
and provides unified interface for different file writers.
"""
import os
import time
import threading
import numpy as np
import tifffile as tif
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from imswitch.imcommon.model import initLogger, dirtools
from .OmeTiffStitcher import OmeTiffStitcher

# Try to import OME-Zarr support
try:
    from .zarr_data_source import MinimalZarrDataSource
    from .single_multiscale_zarr_data_source import SingleMultiscaleZarrWriter
    IS_OMEZARR_AVAILABLE = False  # TODO: Set to True when ready
except ImportError:
    IS_OMEZARR_AVAILABLE = False


class FileIOManager:
    """Manages file I/O operations for different formats and writers."""

    def __init__(self, base_save_dir: str = None):
        """
        Initialize file I/O manager.
        
        Args:
            base_save_dir: Base directory for saving files
        """
        self._logger = initLogger(self)
        
        # Setup save directory
        if base_save_dir is None:
            base_save_dir = dirtools.UserFileDirs.Data
        self.base_save_dir = base_save_dir
        
        # Active writers
        self._tiff_writers = []
        self._omezarr_store = None
        
        # File paths tracking
        self._file_paths = []

    def create_experiment_directory(self, experiment_name: str) -> str:
        """
        Create directory for experiment files.
        
        Args:
            experiment_name: Name of the experiment
            
        Returns:
            Path to created directory
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dir_path = os.path.join(
            self.base_save_dir, 'recordings', 
            f"{timestamp}_{experiment_name}"
        )
        os.makedirs(dir_path, exist_ok=True)
        return dir_path

    def setup_tiff_writers(self, experiment_points: List, 
                          base_path: str, experiment_name: str) -> List[OmeTiffStitcher]:
        """
        Setup TIFF writers for experiment points.
        
        Args:
            experiment_points: List of experiment points
            base_path: Base directory path
            experiment_name: Name of experiment
            
        Returns:
            List of initialized TIFF writers
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tiff_writers = []
        file_paths = []
        
        for index, point in enumerate(experiment_points):
            point_name = point.name
            filename = f"{timestamp}_{experiment_name}{index}_{point_name}_.ome.tif"
            file_path = os.path.join(base_path, filename)
            
            file_paths.append(file_path)
            self._logger.debug(f"OME-TIFF path: {file_path}")
            
            # Create OmeTiffStitcher instance
            tiff_writer = OmeTiffStitcher(file_path)
            tiff_writers.append(tiff_writer)
        
        self._tiff_writers = tiff_writers
        self._file_paths = file_paths
        return tiff_writers

    def setup_omezarr_store(self, zarr_path: str, n_times: int, n_z_steps: int,
                           fov_x: int, fov_y: int) -> Optional[Any]:
        """
        Setup OME-Zarr store for writing.
        
        Args:
            zarr_path: Path for Zarr store
            n_times: Number of timepoints
            n_z_steps: Number of Z steps
            fov_x: Field of view X size
            fov_y: Field of view Y size
            
        Returns:
            Initialized Zarr store or None if not available
        """
        if not IS_OMEZARR_AVAILABLE:
            self._logger.warning("OME-Zarr not available")
            return None
            
        try:
            ome_store = SingleMultiscaleZarrWriter(zarr_path, "w")
            ome_store.set_metadata(t=n_times, c=1, z=n_z_steps, 
                                  bigY=fov_y, bigX=fov_x)
            ome_store.open_store()
            self._omezarr_store = ome_store
            self._logger.debug(f"OME-Zarr store created at: {zarr_path}")
            return ome_store
        except Exception as e:
            self._logger.error(f"Failed to create OME-Zarr store: {e}")
            return None

    def start_tiff_writer(self, writer_index: int):
        """
        Start a specific TIFF writer.
        
        Args:
            writer_index: Index of the writer to start
        """
        if 0 <= writer_index < len(self._tiff_writers):
            writer = self._tiff_writers[writer_index]
            writer.start()
            self._logger.debug(f"Started TIFF writer {writer_index}")
        else:
            self._logger.error(f"Invalid TIFF writer index: {writer_index}")

    def close_tiff_writer(self, writer_index: int):
        """
        Close a specific TIFF writer.
        
        Args:
            writer_index: Index of the writer to close
        """
        if 0 <= writer_index < len(self._tiff_writers):
            writer = self._tiff_writers[writer_index]
            writer.close()
            self._logger.debug(f"Closed TIFF writer {writer_index}")
        else:
            self._logger.error(f"Invalid TIFF writer index: {writer_index}")

    def save_frame_tiff(self, image: np.ndarray, writer_index: int,
                       position_x: float, position_y: float, position_z: float = 0,
                       index_x: int = 0, index_y: int = 0, 
                       pixel_size: float = 1.0) -> bool:
        """
        Save frame to TIFF file.
        
        Args:
            image: Image array to save
            writer_index: Index of TIFF writer to use
            position_x: Stage X position
            position_y: Stage Y position
            position_z: Stage Z position
            index_x: Tile index X
            index_y: Tile index Y
            pixel_size: Pixel size in microns
            
        Returns:
            True if successful, False otherwise
        """
        if not (0 <= writer_index < len(self._tiff_writers)):
            self._logger.error(f"Invalid TIFF writer index: {writer_index}")
            return False
            
        writer = self._tiff_writers[writer_index]
        
        try:
            writer.add_image(
                image=image,
                position_x=position_x,
                position_y=position_y,
                index_x=index_x,
                index_y=index_y,
                pixel_size=pixel_size
            )
            return True
        except Exception as e:
            self._logger.error(f"Error saving TIFF: {e}")
            return False

    def save_frame_omezarr(self, image: np.ndarray, 
                          position_x: float, position_y: float, position_z: float = 0,
                          time_index: int = 0, channel_index: int = 0, 
                          z_index: int = 0) -> bool:
        """
        Save frame to OME-Zarr store.
        
        Args:
            image: Image array to save
            position_x: Stage X position
            position_y: Stage Y position  
            position_z: Stage Z position
            time_index: Time point index
            channel_index: Channel index
            z_index: Z slice index
            
        Returns:
            True if successful, False otherwise
        """
        if not self._omezarr_store:
            return False
            
        try:
            self._omezarr_store.write_tile(
                image, t=time_index, c=channel_index, z=z_index,
                y_start=int(position_y), x_start=int(position_x)
            )
            return True
        except Exception as e:
            self._logger.error(f"Error saving to OME-Zarr: {e}")
            return False

    def save_canvas_image(self, canvas: np.ndarray, filename: str = "final_canvas.tif") -> bool:
        """
        Save canvas image to file.
        
        Args:
            canvas: Canvas image array
            filename: Filename to save as
            
        Returns:
            True if successful, False otherwise
        """
        try:
            tif.imwrite(filename, canvas)
            self._logger.debug(f"Saved canvas to {filename}")
            return True
        except Exception as e:
            self._logger.error(f"Error saving canvas: {e}")
            return False

    def close_omezarr_store(self) -> bool:
        """
        Close OME-Zarr store.
        
        Returns:
            True if successful, False otherwise
        """
        if not self._omezarr_store:
            return True
            
        try:
            self._omezarr_store.close()
            self._omezarr_store = None
            self._logger.debug("Closed OME-Zarr store")
            return True
        except Exception as e:
            self._logger.error(f"Error closing OME-Zarr store: {e}")
            return False

    def close_all_writers(self):
        """Close all active writers."""
        # Close TIFF writers
        for i, writer in enumerate(self._tiff_writers):
            try:
                writer.close()
            except Exception as e:
                self._logger.error(f"Error closing TIFF writer {i}: {e}")
        
        # Close OME-Zarr store
        self.close_omezarr_store()
        
        self._tiff_writers = []

    def get_file_paths(self) -> List[str]:
        """
        Get list of file paths created.
        
        Returns:
            List of file paths
        """
        return self._file_paths.copy()

    def compute_canvas_dimensions(self, min_x: float, max_x: float, 
                                min_y: float, max_y: float,
                                diff_x: float, diff_y: float, 
                                pixel_size: float) -> tuple:
        """
        Compute canvas dimensions for stitching.
        
        Args:
            min_x, max_x: X coordinate range
            min_y, max_y: Y coordinate range
            diff_x, diff_y: Step sizes
            pixel_size: Pixel size in microns
            
        Returns:
            Tuple of (width_pixels, height_pixels)
        """
        width_pixels = int(np.ceil((max_x - min_x + diff_x) / pixel_size))
        height_pixels = int(np.ceil((max_y - min_y + diff_y) / pixel_size))
        return width_pixels, height_pixels

    def create_canvas(self, width: int, height: int, 
                     is_rgb: bool = False, dtype=np.uint16) -> np.ndarray:
        """
        Create blank canvas for image stitching.
        
        Args:
            width: Canvas width in pixels
            height: Canvas height in pixels
            is_rgb: Whether canvas is RGB
            dtype: Data type for canvas
            
        Returns:
            Blank canvas array
        """
        if is_rgb:
            canvas = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            canvas = np.zeros((height, width), dtype=dtype)
        return canvas

    def add_image_to_canvas(self, canvas: np.ndarray, image: np.ndarray,
                           pos_x: float, pos_y: float, min_x: float, min_y: float,
                           pixel_size: float) -> np.ndarray:
        """
        Add image to canvas at specified position.
        
        Args:
            canvas: Canvas array to add to
            image: Image to add
            pos_x, pos_y: Position coordinates
            min_x, min_y: Minimum coordinates for offset calculation
            pixel_size: Pixel size for coordinate conversion
            
        Returns:
            Updated canvas
        """
        try:
            # Calculate pixel position
            pos_pixels = (
                int((pos_y - min_y) / pixel_size),
                int((pos_x - min_x) / pixel_size)
            )
            
            # Use maximum blending (assuming utils.paste is available)
            # For now, simple placement
            h, w = image.shape[:2]
            y_start, x_start = pos_pixels
            
            # Bounds checking
            y_end = min(y_start + h, canvas.shape[0])
            x_end = min(x_start + w, canvas.shape[1])
            
            if y_start >= 0 and x_start >= 0:
                img_h = y_end - y_start
                img_w = x_end - x_start
                
                if len(canvas.shape) == 3:  # RGB
                    canvas[y_start:y_end, x_start:x_end] = np.maximum(
                        canvas[y_start:y_end, x_start:x_end],
                        image[:img_h, :img_w]
                    )
                else:  # Grayscale
                    canvas[y_start:y_end, x_start:x_end] = np.maximum(
                        canvas[y_start:y_end, x_start:x_end],
                        image[:img_h, :img_w]
                    )
                    
        except Exception as e:
            self._logger.error(f"Error adding image to canvas: {e}")
            
        return canvas