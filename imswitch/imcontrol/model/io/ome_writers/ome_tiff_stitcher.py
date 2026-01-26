"""
OME-TIFF stitcher for assembling mosaic/tiled images.

This module provides a background-threaded writer that appends images to an OME-TIFF
file with proper positional metadata for Fiji/ImageJ compatibility.

Migrated from: imswitch/imcontrol/controller/controllers/experiment_controller/OmeTiffStitcher.py
"""

import time
import threading
import tifffile
from collections import deque
import os
from typing import Optional


class OmeTiffStitcher:
    """
    Background-threaded OME-TIFF stitcher for assembling tiled mosaic images.
    
    This class maintains a queue of images with positional metadata and writes
    them to a single OME-TIFF file in a background thread. The resulting file
    can be read by Fiji and other image analysis tools with proper tile
    positioning metadata.
    
    Example:
        >>> from imswitch.imcontrol.model.io.writers import OmeTiffStitcher
        >>> stitcher = OmeTiffStitcher("/path/to/output.ome.tif")
        >>> stitcher.start()
        >>> stitcher.add_image(image_array, pos_x=100.0, pos_y=200.0, 
        ...                    index_x=0, index_y=1, pixel_size=0.325)
        >>> stitcher.stop()
    """

    def __init__(self, file_path: str, bigtiff: bool = True, isRGB: bool = False):
        """
        Initialize the OME-TIFF stitcher.
        
        Args:
            file_path: Path where the OME-TIFF file will be written
            bigtiff: Whether to use BigTIFF format (recommended for large files >4GB)
            isRGB: Whether images are RGB format (vs grayscale)
        """
        self.file_path = file_path
        self.bigtiff = bigtiff
        self.queue = deque()  # Holds (image_array, metadata_dict)
        self.lock = threading.Lock()
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.isRGB = isRGB

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

    def add_image(self, image, position_x: float, position_y: float, 
                  index_x: int, index_y: int, pixel_size: float):
        """
        Enqueue an image for writing with positional metadata.
        
        Args:
            image: 2D or 3D NumPy array (grayscale or RGB)
            position_x: Stage X coordinate in microns
            position_y: Stage Y coordinate in microns
            index_x: Tile index X position in the grid
            index_y: Tile index Y position in the grid
            pixel_size: Pixel size in microns
        """
        # A minimal OME-like metadata block that Fiji can interpret
        metadata = {
            "Pixels": {
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
            },
            "Plane": {
                "PositionX": position_x,
                "PositionY": position_y,
                "IndexX": index_x,
                "IndexY": index_y
            },
        }
        with self.lock:
            self.queue.append((image, metadata))

    def _process_queue(self):
        """
        Background loop: open the OME-TIFF in append mode, pop images from queue,
        and write them with embedded metadata.
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        photometric = "rgb" if self.isRGB else None
        
        with tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff, append=True) as tif:
            # Keep running until stop() is called AND the queue is empty
            while self.is_running or len(self.queue) > 0:
                with self.lock:
                    if self.queue:
                        image, metadata = self.queue.popleft()
                    else:
                        image = None

                if image is not None:
                    # Each call writes a new series/plane in append mode
                    try:
                        if len(image.shape) == 2:
                            # Grayscale image
                            tif.write(data=image, metadata=metadata)
                        elif len(image.shape) == 3:
                            # Color image (RGB or RGBA)
                            tif.write(data=image, metadata=metadata, photometric=photometric)
                    except Exception as e:
                        print(f"Error writing image to stitched TIFF: {e}")
                else:
                    # Sleep briefly to avoid busy loop when queue is empty
                    time.sleep(0.01)

    def close(self):
        """Close the OME-TIFF stitcher and cleanup resources."""
        self.stop()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        self.is_running = False
        self.queue.clear()
