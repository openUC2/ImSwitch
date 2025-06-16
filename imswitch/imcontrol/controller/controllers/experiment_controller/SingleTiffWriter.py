"""
Single TIFF writer for appending multiple tiles with x/y/z locations and channels.

This module provides a writer that appends all tiles from a single tile scan
to a single TIFF file with proper metadata including spatial coordinates
and channel information. Uses the same queue-based processing pattern as
HistoScanController for memory efficiency.
"""

import time
import threading
import numpy as np
import tifffile
from collections import deque
import os
from typing import Dict, Any


class SingleTiffWriter:
    """
    Single TIFF writer that writes images directly to disk using queue processing.

    This writer follows the same pattern as HistoScanController - it writes each 
    image immediately to disk as it's dequeued, without storing images in memory.
    This ensures memory efficiency for large datasets.
    """

    def __init__(self, file_path: str, bigtiff: bool = True):
        """
        Initialize the single TIFF writer.

        Args:
            file_path: Path where the TIFF file will be written
            bigtiff: Whether to use BigTIFF format
        """
        self.file_path = file_path
        self.bigtiff = bigtiff
        self.queue = deque()  # Holds (image_array, metadata_dict)
        self.lock = threading.Lock()
        self.is_running = False
        self._thread = None
        self.image_count = 0
        self.tiff_writer = None

    def start(self):
        """Begin the background thread that collects images and metadata."""
        self.is_running = True
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the thread to stop and write the final TIFF file."""
        self.is_running = False
        if self._thread is not None:
            self._thread.join()

    def add_image(self, image: np.ndarray, metadata: Dict[str, Any]):
        """
        Add an image with metadata to be written.

        Args:
            image: 2D NumPy array (grayscale image)
            metadata: Dictionary containing position and other metadata
        """
        with self.lock:
            self.queue.append((image, metadata))

    def _process_queue(self):
        """
        Background loop: write images directly to disk using queue processing like HistoScanController.
        """
        # Ensure the folder exists
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
        # Open TiffWriter once with append=True, like HistoScanController
        with tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff, append=True) as tif:
            self.tiff_writer = tif
            
            while self.is_running or len(self.queue) > 0:
                with self.lock:
                    if self.queue:
                        image, metadata = self.queue.popleft()
                    else:
                        image = None

                if image is not None:
                    try:
                        # Create OME metadata and write image immediately to disk
                        tiff_metadata = self._create_tiff_metadata(image, metadata)
                        
                        # Write directly to disk like HistoScanController
                        tif.write(data=image, metadata=tiff_metadata)
                        
                        print(f"Wrote image {self.image_count} to single TIFF at position "
                              f"({metadata.get('x', 0)}, {metadata.get('y', 0)}) to: {self.file_path}")
                        
                        self.image_count += 1
                        
                    except Exception as e:
                        print(f"Error writing image to single TIFF: {e}")
                        import traceback
                        traceback.print_exc()
                else:
                    # Sleep briefly to avoid busy loop when queue is empty
                    time.sleep(0.01)
            
            self.tiff_writer = None
            
    def _create_tiff_metadata(self, image: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create TIFF metadata in the same format as HistoScanController.
        
        Args:
            image: Image data as numpy array
            metadata: Image metadata including position information
            
        Returns:
            Dictionary with TIFF metadata in HistoScanController format
        """
        # Extract metadata
        pixel_size = metadata.get("pixel_size", 1.0)
        pos_x = metadata.get("x", 0)
        pos_y = metadata.get("y", 0)
        
        # Calculate index coordinates (same as HistoScanController)
        index_x = self.image_count
        index_y = 0
        
        # Create metadata in EXACT same format as working HistoScanController
        tiff_metadata = {
            'Pixels': {
                'PhysicalSizeX': float(pixel_size),
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': float(pixel_size), 
                'PhysicalSizeYUnit': 'µm'
            },
            'Plane': {
                'PositionX': float(pos_x),
                'PositionY': float(pos_y),
                'IndexX': int(index_x),
                'IndexY': int(index_y)
            }
        }
        
        return tiff_metadata



    def close(self):
        """Close the single TIFF writer."""
        self.stop()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        
        print(f"Single TIFF writer completed. Total images written: {self.image_count}")
        
        self.is_running = False
        self.queue.clear()
