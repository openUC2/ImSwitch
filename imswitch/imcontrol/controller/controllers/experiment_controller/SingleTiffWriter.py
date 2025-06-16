"""
Single TIFF writer for appending multiple tiles with x/y/z locations and channels.

This module provides a writer that appends all tiles from a single tile scan
to a single TIFF file with proper OME metadata including spatial coordinates
and channel information using ome-types for correct XML structure.
"""

import time
import threading
import numpy as np
import tifffile
from collections import deque
import os
from typing import Dict, Any, List

try:
    import ome_types
    from ome_types.model import OME, Image, Pixels, Plane, Channel, TiffData, UnitsLength
    HAS_OME_TYPES = True
except ImportError:
    HAS_OME_TYPES = False


class SingleTiffWriter:
    """
    Single TIFF writer that creates proper multi-image OME-TIFF with spatial metadata.

    This writer uses ome-types to generate proper OME XML metadata structure
    that creates separate Image elements for each tile with position information
    that Fiji can read correctly.
    """

    def __init__(self, file_path: str, bigtiff: bool = True):
        """
        Initialize the single TIFF writer.

        Args:
            file_path: Path where the TIFF file will be written
            bigtiff: Whether to use BigTIFF format
        """
        if not HAS_OME_TYPES:
            raise ImportError("ome-types is required for SingleTiffWriter. Install with: pip install ome-types")
            
        self.file_path = file_path
        self.bigtiff = bigtiff
        self.queue = deque()  # Holds (image_array, metadata_dict)
        self.lock = threading.Lock()
        self.is_running = False
        self._thread = None
        self.image_count = 0
        
        # OME structure building
        self.ome = OME()
        self.images_data = []  # Store (image, metadata) pairs to write at the end

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
        Background loop: collect images and build OME structure.
        """
        while self.is_running or len(self.queue) > 0:
            with self.lock:
                if self.queue:
                    image, metadata = self.queue.popleft()
                else:
                    image = None

            if image is not None:
                try:
                    # Store image and create OME Image element
                    self._add_ome_image(image, metadata)
                    print(f"Added image {self.image_count} to OME structure at position "
                          f"({metadata.get('x', 0)}, {metadata.get('y', 0)})")
                except Exception as e:
                    print(f"Error adding image to OME structure: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                # Sleep briefly to avoid busy loop when queue is empty
                time.sleep(0.01)

    def _add_ome_image(self, image: np.ndarray, metadata: Dict[str, Any]):
        """
        Add an image to the OME structure and store the image data.
        
        Args:
            image: Image data as numpy array
            metadata: Image metadata including position information
        """
        # Extract metadata
        pixel_size = metadata.get("pixel_size", 1.0)
        pos_x = metadata.get("x", 0)
        pos_y = metadata.get("y", 0)
        
        # Create Pixels element
        pixels = Pixels(
            id=f"Pixels:{self.image_count}",
            dimension_order="XYCZT",
            size_x=image.shape[1],
            size_y=image.shape[0], 
            size_c=1,
            size_t=1,
            size_z=1,
            type="uint16",
            big_endian=False,
            physical_size_x=float(pixel_size),
            physical_size_x_unit=UnitsLength.MICROMETER,
            physical_size_y=float(pixel_size),
            physical_size_y_unit=UnitsLength.MICROMETER
        )
        
        # Create Channel
        channel = Channel(
            id=f"Channel:{self.image_count}:0",
            samples_per_pixel=1
        )
        pixels.channels = [channel]
        
        # Create TiffData referencing the IFD where this image will be stored
        tiff_data = TiffData(
            ifd=self.image_count,
            plane_count=1
        )
        pixels.tiff_data_blocks = [tiff_data]
        
        # Create Plane with position metadata
        plane = Plane(
            the_c=0,
            the_t=0,
            the_z=0,
            position_x=float(pos_x),
            position_x_unit="reference frame",
            position_y=float(pos_y),
            position_y_unit="reference frame"
        )
        pixels.planes = [plane]
        
        # Create Image element
        ome_image = Image(
            id=f"Image:{self.image_count}",
            name=f"Image{self.image_count}",
            pixels=pixels
        )
        
        # Add to OME structure
        self.ome.images.append(ome_image)
        
        # Store the image data to write later
        self.images_data.append((image, metadata))
        
        self.image_count += 1

    def _write_tiff_file(self):
        """
        Write the final TIFF file with all images and OME metadata.
        """
        if not self.images_data:
            print("No images to write")
            return
            
        # Ensure the folder exists
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)

        # Generate the OME XML metadata
        xml_metadata = ome_types.to_xml(self.ome)
        # Replace µm with um to avoid ASCII encoding issues in TIFF
        xml_metadata = xml_metadata.replace('µm', 'um')
        
        print(f"Writing {len(self.images_data)} images to {self.file_path}")
        print(f"Generated OME XML with {len(self.ome.images)} image elements")
        
        # Write TIFF file with OME metadata
        with tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff) as tif:
            for i, (image, metadata) in enumerate(self.images_data):
                # Only add description to the first image
                description = xml_metadata if i == 0 else None
                tif.write(
                    data=image,
                    description=description
                )

    def close(self):
        """Close the single TIFF writer and write the final file."""
        self.stop()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        
        # Write the final TIFF file with all collected images
        try:
            self._write_tiff_file()
            print(f"Single TIFF writer completed. Total images written: {self.image_count}")
        except Exception as e:
            print(f"Error writing final TIFF file: {e}")
            import traceback
            traceback.print_exc()
        
        self.is_running = False
        self.queue.clear()
        self.images_data.clear()
        self.ome = OME()  # Reset OME structure
