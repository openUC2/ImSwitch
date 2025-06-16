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
        Background loop: write images exactly like working HistoScanController.
        Use append=True and pass metadata directly like HistoScanController does.
        """
        # Ensure the folder exists
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Initialize TiffWriter with append=True exactly like HistoScanController
        with tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff, append=True, ome=True) as tif:
            while self.is_running or len(self.queue) > 0:
                with self.lock:
                    if self.queue:
                        image, input_metadata = self.queue.popleft()
                    else:
                        image = None

                if image is not None:
                    try:
                        # Extract position and pixel size from input metadata
                        pixel_size = input_metadata.get("pixel_size", 1.0)
                        pos_x = input_metadata.get("x", 0)
                        pos_y = input_metadata.get("y", 0)

                        # Calculate index coordinates from image count for now
                        index_x = self.image_count
                        index_y = 0

                        # Create metadata in EXACT same format as working HistoScanController
                        # with IndexX and IndexY included (they were commented out before)
                        metadata = {'Pixels': {
                            'PhysicalSizeX': float(pixel_size),
                            'PhysicalSizeXUnit': 'µm',
                            'PhysicalSizeY': float(pixel_size),
                            'PhysicalSizeYUnit': 'µm'},

                            'Plane': {
                                'PositionX': float(pos_x),
                                'PositionY': float(pos_y),
                                'IndexX': int(index_x),
                                'IndexY': int(index_y)
                        }}

                        # Write image using EXACT same method as HistoScanController
                        tif.write(data=image, metadata=metadata)

                        self.image_count += 1
                        print(f"Wrote image {self.image_count} to single TIFF at position "
                              f"({pos_x}, {pos_y}) with index ({index_x}, {index_y})")
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
        print(f"Single TIFF writer closed. Total images written: {self.image_count}")
