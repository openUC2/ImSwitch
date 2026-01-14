"""
Single TIFF writer for appending multiple tiles with x/y/z locations and channels.

This module provides a writer that appends all tiles from a single tile scan
to a single TIFF file with proper OME metadata including spatial coordinates
and channel information.
"""

import threading
import numpy as np
import tifffile
from collections import deque
import os
from typing import Dict, Any


class SingleTiffWriter:
    """
    Single TIFF writer that appends tiles as separate images with individual position metadata.
    
    This writer is specifically designed for single tile scans where each scan position
    contains only one tile, and all tiles should be appended to a single TIFF file
    as separate images. Each tile becomes a separate image element in the TIFF with
    proper position metadata that Fiji can read correctly.
    
    Uses the exact same pattern as the working HistoScanController.
    """

    def __init__(self, file_path: str, bigtiff: bool = True):
        """
        Initialize the single TIFF writer.
        
        Args:
            file_path: Path where the TIFF file will be written
            bigtiff: Whether to use BigTIFF format (True to match HistoScanController)
        """
        self.file_path = file_path
        self.bigtiff = bigtiff
        self.queue = deque()  # Holds (image_array, metadata_dict)
        self.lock = threading.Lock()
        self.is_running = False
        self._thread = None
        self.image_count = 0
        self._finalize_requested = False
        self._tiff_writer = None  # Keep writer reference for synchronous mode

    def add_image(self, image: np.ndarray, metadata: Dict[str, Any]):
        """
        Write an image synchronously with metadata.
        
        Args:
            image: 2D NumPy array (grayscale image)
            metadata: Dictionary containing position and other metadata
        """
        # Write synchronously instead of queueing
        self._write_image_sync(image, metadata)

    def _write_image_sync(self, image: np.ndarray, input_metadata: Dict[str, Any]):
        """
        Write image synchronously without using a background thread.
        
        Args:
            image: 2D NumPy array (grayscale image)
            input_metadata: Dictionary containing position and other metadata
        """
        # Ensure the folder exists
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Initialize TiffWriter on first write (without append mode to create a new file)
        if self._tiff_writer is None:
            self._tiff_writer = tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff)

        try:
            # Extract metadata in EXACT same format as HistoScanController
            pixel_size = input_metadata.get("pixel_size", 1.0)
            pos_x = input_metadata.get("x", 0)
            pos_y = input_metadata.get("y", 0)

            # Calculate index coordinates from position and grid step
            index_x = self.image_count # Assuming image_count corresponds to the x index
            index_y = 0

            # Create metadata in EXACT same format as working HistoScanController
            metadata = {'Pixels': {
                'ImageDescription': f"ImageID={self.image_count}",
                'PhysicalSizeX': float(pixel_size),
                'PhysicalSizeXUnit': 'µm',
                'PhysicalSizeY': float(pixel_size),
                'PhysicalSizeYUnit': 'µm'},
                'axes': 'ZYX',
                'Plane': {
                    'PositionX': float(pos_x),
                    'PositionY': float(pos_y)
            }}

            # Write image using EXACT same method as HistoScanController
            self._tiff_writer.write(data=image, metadata=metadata)

            self.image_count += 1
            #print(f"Wrote image {self.image_count} to single TIFF at position ({pos_x}, {pos_y}) with index ({index_x}, {index_y})")
        except Exception as e:
            print(f"Error writing image to single TIFF: {e}")
            import traceback
            traceback.print_exc()

    def _process_queue(self):
        """
        DEPRECATED: Background loop for asynchronous writing.
        This method is kept for backward compatibility but is no longer used.
        """
        pass


    def close(self):
        """Close the single TIFF writer."""
        # Close the TiffWriter if it exists
        if self._tiff_writer is not None:
            try:
                self._tiff_writer.close()
            except Exception as e:
                print(f"Error closing TIFF writer: {e}")
            finally:
                self._tiff_writer = None

        print(f"Single TIFF writer completed. Total images written: {self.image_count}")

        self.is_running = False
        self._finalize_requested = True
        self.queue.clear()
