"""
Single TIFF writer for appending multiple tiles with x/y/z locations and channels.

This module provides a writer that appends all tiles from a single tile scan
to a single TIFF file with proper OME metadata including spatial coordinates
and channel information.
"""

import time
import threading
import numpy as np
import tifffile
from collections import deque
import os
from typing import Dict, Any, Optional


class SingleTiffWriter:
    """
    Single TIFF writer that appends tiles as a multi-page TIFF timelapse series with xyz positions.
    
    This writer is specifically designed for single tile scans where each scan position
    contains only one tile, and all tiles should be appended to a single TIFF file
    as a timelapse series. Each tile becomes a timepoint in the series, with spatial
    coordinates stored in metadata for Fiji/napari compatibility.
    """
    
    def __init__(self, file_path: str, bigtiff: bool = False):
        """
        Initialize the single TIFF writer.
        
        Args:
            file_path: Path where the TIFF file will be written
            bigtiff: Whether to use BigTIFF format (False for Fiji compatibility)
        """
        self.file_path = file_path
        self.bigtiff = bigtiff
        self.queue = deque()  # Holds (image_array, metadata_dict)
        self.lock = threading.Lock()
        self.is_running = False
        self._thread = None
        self.image_count = 0
        self.timepoint_index = 0  # Track current timepoint in series
        
    def start(self):
        """Begin the background thread that writes images to disk as they arrive."""
        self.is_running = True
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()
    
    def stop(self):
        """Signal the thread to stop, then join it."""
        self.is_running = False
        if self._thread is not None:
            self._thread.join()
    
    def add_image(self, image: np.ndarray, metadata: Dict[str, Any]):
        """
        Enqueue an image for writing with metadata.
        
        Args:
            image: 2D NumPy array (grayscale image)
            metadata: Dictionary containing position and other metadata
        """
        with self.lock:
            self.queue.append((image, metadata))
    
    def _process_queue(self):
        """
        Background loop: write images as pages in a multi-page TIFF timelapse series.
        """
        # Ensure the folder exists
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
        # Open TiffWriter to create proper multi-page TIFF
        with tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff) as tif:
            while self.is_running or len(self.queue) > 0:
                with self.lock:
                    if self.queue:
                        image, metadata = self.queue.popleft()
                    else:
                        image = None
                
                if image is not None:
                    try:
                        # Extract position information for description
                        position_x = metadata.get("PositionX", metadata.get("TilePosition", {}).get("X", 0.0))
                        position_y = metadata.get("PositionY", metadata.get("TilePosition", {}).get("Y", 0.0))
                        position_z = metadata.get("PositionZ", metadata.get("TilePosition", {}).get("Z", 0.0))
                        pixel_size = metadata.get("PhysicalSizeX", metadata.get("pixel_size", 1.0))
                        
                        # Create description with position information
                        description = f"T={self.image_count} X={position_x:.1f} Y={position_y:.1f} Z={position_z:.1f} pixel_size={pixel_size}"
                        
                        # Write image as a page in the multi-page TIFF
                        # Use simple metadata that won't cause conflicts
                        tif.write(
                            data=image,
                            photometric='minisblack',
                            contiguous=True,
                            description=description,
                            resolution=(1/pixel_size, 1/pixel_size),  # Set resolution for scale
                            resolutionunit='MICROMETER'
                        )
                        
                        self.image_count += 1
                        self.timepoint_index += 1
                        print(f"Wrote image to timelapse TIFF as T=%d, to path: %s" % (self.image_count-1, self.file_path))
                    except Exception as e:
                        print(f"Error writing image to single TIFF: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Sleep briefly to avoid busy loop when queue is empty
                    time.sleep(0.01)
    
    def close(self):
        """Close the single TIFF writer."""
        self.stop()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.is_running = False
        self.queue.clear()
        print(f"Single TIFF timelapse writer closed. Total timepoints written: {self.image_count}")