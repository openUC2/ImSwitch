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
    Single TIFF writer that appends tiles with x/y/z locations and channel stacking.
    
    This writer is specifically designed for single tile scans where each scan position
    contains only one tile, and all tiles should be appended to a single TIFF file
    with proper OME metadata including spatial coordinates.
    """
    
    def __init__(self, file_path: str, bigtiff: bool = True):
        """
        Initialize the single TIFF writer.
        
        Args:
            file_path: Path where the TIFF file will be written
            bigtiff: Whether to use BigTIFF format (recommended for large files)
        """
        self.file_path = file_path
        self.bigtiff = bigtiff
        self.queue = deque()  # Holds (image_array, metadata_dict)
        self.lock = threading.Lock()
        self.is_running = False
        self._thread = None
        self.image_count = 0
        
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
        Enqueue an image for writing with full metadata.
        
        Args:
            image: 2D NumPy array (grayscale image)
            metadata: Dictionary containing position and other metadata
        """
        # Extract relevant metadata for OME-TIFF
        ome_metadata = self._create_ome_metadata(image, metadata)
        
        with self.lock:
            self.queue.append((image, ome_metadata))
    
    def _create_ome_metadata(self, image: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create OME metadata dictionary from tile metadata.
        
        Args:
            image: Image array
            metadata: Original metadata dictionary
            
        Returns:
            OME-compatible metadata dictionary
        """
        # Extract positions and indices
        position_x = metadata.get("x", 0.0)
        position_y = metadata.get("y", 0.0)
        z_index = metadata.get("z_index", 0)
        t_index = metadata.get("time_index", 0)
        c_index = metadata.get("channel_index", 0)
        
        # Get pixel size from metadata or use default
        pixel_size = metadata.get("pixel_size", 1.0)
        
        # Create OME metadata structure
        ome_metadata = {
            "Pixels": {
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
                "SizeX": image.shape[1],
                "SizeY": image.shape[0],
                "SizeZ": 1,
                "SizeT": 1,
                "SizeC": 1,
                "Type": "uint16",
                "DimensionOrder": "XYZCT"
            },
            "Plane": {
                "PositionX": position_x,
                "PositionY": position_y,
                "PositionZ": z_index,
                "TheZ": z_index,
                "TheT": t_index,
                "TheC": c_index
            },
            "Channel": {
                "ID": f"Channel:{c_index}",
                "SamplesPerPixel": 1,
                "IlluminationType": metadata.get("illuminationChannel", "Unknown")
            }
        }
        
        return ome_metadata
    
    def _process_queue(self):
        """
        Background loop: open the OME-TIFF in append mode, pop images from queue,
        and write them with embedded metadata.
        """
        # Ensure the folder exists
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
        # Use append mode to add multiple images to single file
        with tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff, append=True) as tif:
            while self.is_running or len(self.queue) > 0:
                with self.lock:
                    if self.queue:
                        image, metadata = self.queue.popleft()
                    else:
                        image = None
                
                if image is not None:
                    try:
                        # Write each image as a new page in the TIFF file
                        tif.write(
                            data=image,
                            metadata=metadata,
                            contiguous=True
                        )
                        self.image_count += 1
                    except Exception as e:
                        print(f"Error writing image to single TIFF: {e}")
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
        print(f"Single TIFF writer closed. Total images written: {self.image_count}")