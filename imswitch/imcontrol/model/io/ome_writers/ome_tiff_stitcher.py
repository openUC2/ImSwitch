"""
OME-TIFF stitcher for assembling mosaic/tiled images.

This module provides a background-threaded writer that appends images to an OME-TIFF
file with proper positional metadata for Fiji/ImageJ compatibility.

Migrated from: imswitch/imcontrol/controller/controllers/experiment_controller/OmeTiffStitcher.py
"""

import threading
import tifffile
from collections import deque
import os
from typing import Optional

from imswitch.imcommon.model import initLogger


class OmeTiffStitcher:
    """
    Background-threaded OME-TIFF stitcher for assembling tiled mosaic images.

    This class maintains a queue of images with positional metadata and writes
    them to a single OME-TIFF file in a background thread. The resulting file
    can be read by Fiji and other image analysis tools with proper tile
    positioning metadata.

    It makes no assumption about how many images it will receive: it writes
    whatever is queued until stop() is called and the queue has drained.
    Exiting on an expected tile count instead hung stop() forever on a short
    run and discarded the tail of the queue on a long one.

    Example:
        >>> from imswitch.imcontrol.model.io.writers import OmeTiffStitcher
        >>> stitcher = OmeTiffStitcher("/path/to/output.ome.tif")
        >>> stitcher.start()
        >>> stitcher.add_image(image_array, pos_x=100.0, pos_y=200.0,
        ...                    index_x=0, index_y=1, pixel_size=0.325)
        >>> stitcher.stop()
    """

    def __init__(self, file_path: str, bigtiff: bool = True, isRGB: bool = False,
                 tile_w: Optional[int] = None, tile_h: Optional[int] = None,
                 write_kwargs: Optional[dict] = None):
        """
        Initialize the OME-TIFF stitcher.

        Args:
            file_path: Path where the OME-TIFF file will be written
            bigtiff: Whether to use BigTIFF format (recommended for large files >4GB)
            isRGB: Whether images are RGB format (vs grayscale)
            tile_w: Width of each tile in pixels (optional, for metadata)
            tile_h: Height of each tile in pixels (optional, for metadata)
            write_kwargs: Compression kwargs passed to every tifffile write
                (see OMEWriterConfig.tiff_write_kwargs). This is the largest
                file a run produces and it used to be written uncompressed.
        """
        self._logger = initLogger(self)
        self.file_path = file_path
        self.bigtiff = bigtiff
        self.queue = deque()  # Holds (image_array, metadata_dict)
        self.lock = threading.Lock()
        # Set before the thread starts and cleared by stop(); the thread also
        # waits on _wake so an idle queue costs nothing.
        self.is_running = False
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.isRGB = isRGB
        self.tile_w = tile_w
        self.tile_h = tile_h
        self.images_written = 0
        self._write_kwargs = write_kwargs or {}

    def start(self):
        """Begin the background thread that writes images to disk as they arrive."""
        if self._thread is not None:
            return
        self.is_running = True
        # Not a daemon: interpreter shutdown killing this thread inside the
        # TiffWriter context is what produced blank stitched files.
        self._thread = threading.Thread(
            target=self._process_queue, daemon=False, name="OmeTiffStitcher"
        )
        self._thread.start()

    def stop(self):
        """Signal the thread to finish draining the queue, then join it."""
        self.is_running = False
        self._wake.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

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
        self._wake.set()

    def _process_queue(self):
        """
        Background loop: open the OME-TIFF in append mode, pop images from the
        queue, and write them with embedded metadata. Exits once stop() has been
        called and the queue is empty — never before, never on a tile count.
        """
        # Ensure the output directory exists
        output_dir = os.path.dirname(self.file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        photometric = "rgb" if self.isRGB else None

        with tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff, append=True) as tif:
            while True:
                with self.lock:
                    image, metadata = self.queue.popleft() if self.queue else (None, None)

                if image is None:
                    if not self.is_running:
                        break
                    if not threading.main_thread().is_alive():
                        # Shutting down with nobody to close us. Non-daemon,
                        # so drain and close rather than block the exit.
                        self._logger.warning(
                            "Interpreter shutting down with the stitcher still "
                            "open — draining and closing"
                        )
                        self.is_running = False
                        continue
                    # Sleep until something arrives (or stop() wakes us).
                    self._wake.wait(0.1)
                    self._wake.clear()
                    continue

                try:
                    if image.ndim == 2:
                        tif.write(data=image, metadata=metadata, **self._write_kwargs)
                    else:
                        tif.write(data=image, metadata=metadata,
                                  photometric=photometric, **self._write_kwargs)
                    self.images_written += 1
                except Exception as e:
                    self._logger.error(f"Error writing image to stitched TIFF: {e}")

        self._logger.info(
            f"Stitched TIFF closed: {self.images_written} images → {self.file_path}"
        )

    def close(self):
        """Close the OME-TIFF stitcher and cleanup resources."""
        self.stop()
        self.queue.clear()
