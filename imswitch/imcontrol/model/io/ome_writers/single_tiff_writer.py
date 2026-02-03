"""
Single TIFF writer for appending multiple tiles with positional metadata.

This module provides a synchronous TIFF writer that appends all tiles from a scan
to a single TIFF file with proper OME metadata including spatial coordinates.
Each tile is stored as a separate image within the TIFF.

Migrated from: imswitch/imcontrol/controller/controllers/experiment_controller/SingleTiffWriter.py
"""

import os
from typing import Dict, Any, Optional
import numpy as np
import tifffile


class SingleTiffWriter:
    """
    Single TIFF writer that appends tiles as separate images with individual position metadata.
    
    This writer is specifically designed for tile scans where each scan position
    contains one tile, and all tiles should be appended to a single TIFF file
    as separate images. Each tile becomes a separate image element in the TIFF with
    proper position metadata that Fiji can read correctly.
    
    Unlike OmeTiffStitcher, this writer operates synchronously (no background thread)
    which is simpler and suitable for most use cases.
    
    Example:
        >>> from imswitch.imcontrol.model.io.writers import SingleTiffWriter
        >>> writer = SingleTiffWriter("/path/to/tiles.ome.tif")
        >>> writer.add_image(image_array, {"x": 100.0, "y": 200.0, "pixel_size": 0.325})
        >>> writer.add_image(image_array2, {"x": 100.0, "y": 300.0, "pixel_size": 0.325})
        >>> writer.close()
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
        self.image_count = 0
        self._tiff_writer: Optional[tifffile.TiffWriter] = None

    def add_image(self, image: np.ndarray, metadata: Dict[str, Any]):
        """
        Write an image synchronously with metadata.
        
        Args:
            image: 2D NumPy array (grayscale image)
            metadata: Dictionary containing position and other metadata.
                      Expected keys: 'x', 'y', 'pixel_size'
        """
        self._write_image_sync(image, metadata)

    def _write_image_sync(self, image: np.ndarray, input_metadata: Dict[str, Any]):
        """
        Write image synchronously.
        
        Args:
            image: 2D NumPy array (grayscale image)
            input_metadata: Dictionary containing position and other metadata
        """
        # Ensure the folder exists
        output_dir = os.path.dirname(self.file_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        # Initialize TiffWriter on first write (creates a new file)
        if self._tiff_writer is None:
            self._tiff_writer = tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff)

        try:
            # Extract metadata
            pixel_size = input_metadata.get("pixel_size", 1.0)
            pos_x = input_metadata.get("x", 0)
            pos_y = input_metadata.get("y", 0)

            # Create metadata in format compatible with Fiji/ImageJ
            metadata = {
                'Pixels': {
                    'ImageDescription': f"ImageID={self.image_count}",
                    'PhysicalSizeX': float(pixel_size),
                    'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': float(pixel_size),
                    'PhysicalSizeYUnit': 'µm'
                },
                'axes': 'ZYX',
                'Plane': {
                    'PositionX': float(pos_x),
                    'PositionY': float(pos_y)
                }
            }

            # Write image with metadata
            self._tiff_writer.write(data=image, metadata=metadata)
            self.image_count += 1

        except Exception as e:
            print(f"Error writing image to single TIFF: {e}")
            import traceback
            traceback.print_exc()

    def close(self):
        """Close the single TIFF writer and finalize the file."""
        if self._tiff_writer is not None:
            try:
                self._tiff_writer.close()
            except Exception as e:
                print(f"Error closing TIFF writer: {e}")
            finally:
                self._tiff_writer = None

        print(f"Single TIFF writer completed. Total images written: {self.image_count}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures proper cleanup."""
        self.close()
        return False
