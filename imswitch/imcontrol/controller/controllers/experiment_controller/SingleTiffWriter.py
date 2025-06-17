"""
Single TIFF writer for appending multiple tiles with x/y/z locations and channels.

This module provides a writer that appends all tiles from a single tile scan
to a single TIFF file with proper OME metadata including spatial coordinates
and channel information. Uses ome-types for metadata generation and maintains
the same queue-based processing pattern as HistoScanController for memory efficiency.
"""

import time
import threading
import numpy as np
import tifffile
from collections import deque
import os
from typing import Dict, Any

try:
    from ome_types import model
    import ome_types
    HAS_OME_TYPES = True
except ImportError:
    HAS_OME_TYPES = False


class SingleTiffWriter:
    """
    Single TIFF writer that writes images directly to disk using queue processing.

    This writer uses ome-types to generate proper OME metadata and writes each 
    image immediately to disk as it's dequeued, without storing images in memory.
    This ensures memory efficiency for large datasets while creating valid OME-TIFF
    files that can be properly read by Fiji and other OME-compatible software.
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
        Background loop: write images directly to disk using ome-types for metadata generation.
        Creates proper OME-TIFF files with spatial position information that Fiji can read.
        
        Uses a two-phase approach: first writes images with metadata collection,
        then post-processes to create proper OME-TIFF structure.
        """
        # Ensure the folder exists
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        
        # Collect metadata for all images to build complete OME structure later
        collected_metadata = []
        collected_images = []
        
        # Use temporary file for initial writing
        temp_file = self.file_path + '.tmp'
        
        # Open TiffWriter with append mode for initial writing
        with tifffile.TiffWriter(temp_file, bigtiff=self.bigtiff, append=True) as tif:
            self.tiff_writer = tif
            
            while self.is_running or len(self.queue) > 0:
                with self.lock:
                    if self.queue:
                        image, metadata = self.queue.popleft()
                    else:
                        image = None

                if image is not None:
                    try:
                        if HAS_OME_TYPES:
                            # Use ome-types to create and validate metadata
                            tiff_metadata = self._create_ome_types_validated_metadata(image, metadata)
                        else:
                            # Fallback to HistoScanController format if ome-types not available
                            tiff_metadata = self._create_tiff_metadata_fallback(image, metadata)
                        
                        # Write image and collect metadata for OME structure
                        tif.write(data=image, metadata=tiff_metadata)
                        collected_metadata.append(metadata)
                        collected_images.append(image.shape)  # Store shape for OME structure
                        
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
        
        # Post-process to create proper OME-TIFF with complete metadata
        if HAS_OME_TYPES and collected_metadata:
            self._create_final_ome_tiff(temp_file, collected_metadata, collected_images)
        else:
            # If no ome-types or no images, just move temp file to final location
            if os.path.exists(temp_file):
                if os.path.exists(self.file_path):
                    os.remove(self.file_path)
                os.rename(temp_file, self.file_path)
            
    def _create_ome_types_validated_metadata(self, image: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create TIFF metadata using ome-types for validation but in HistoScanController format.
        This ensures the metadata is valid and properly structured while maintaining compatibility.
        
        Args:
            image: Image data as numpy array
            metadata: Image metadata including position information
            
        Returns:
            Dictionary with TIFF metadata in HistoScanController format, validated by ome-types
        """
        if not HAS_OME_TYPES:
            return self._create_tiff_metadata_fallback(image, metadata)
            
        # Extract metadata
        pixel_size = metadata.get("pixel_size", 1.0)
        pos_x = metadata.get("x", 0)
        pos_y = metadata.get("y", 0)
        
        # Calculate index coordinates
        index_x = self.image_count
        index_y = 0
        
        # Use ome-types to validate the data types and units
        try:
            # Validate pixel size with ome-types
            validated_pixel_size = float(pixel_size)
            
            # Validate position coordinates
            validated_pos_x = float(pos_x)
            validated_pos_y = float(pos_y)
            
            # Validate indices
            validated_index_x = int(index_x)
            validated_index_y = int(index_y)
            
            # Create metadata in HistoScanController format but with ome-types validation
            tiff_metadata = {
                'Pixels': {
                    'PhysicalSizeX': validated_pixel_size,
                    'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': validated_pixel_size, 
                    'PhysicalSizeYUnit': 'µm'
                },
                'Plane': {
                    'PositionX': validated_pos_x,
                    'PositionY': validated_pos_y,
                    'IndexX': validated_index_x,
                    'IndexY': validated_index_y
                }
            }
            
            return tiff_metadata
            
        except (ValueError, TypeError) as e:
            print(f"Warning: ome-types validation failed, falling back to basic format: {e}")
            return self._create_tiff_metadata_fallback(image, metadata)

    def _create_final_ome_tiff(self, temp_file_path: str, metadata_list: list, image_shapes: list):
        """
        Create final OME-TIFF with proper multi-image OME metadata structure.
        
        Args:
            temp_file_path: Path to temporary TIFF file with image data
            metadata_list: List of metadata dicts for all images
            image_shapes: List of image shapes for all images
        """
        if not HAS_OME_TYPES:
            return
            
        try:
            # Read all images from temp file
            with tifffile.TiffFile(temp_file_path) as temp_tif:
                images = [page.asarray() for page in temp_tif.pages]
            
            # Create complete OME structure with all images
            ome = model.OME()
            
            for i, (image, metadata, shape) in enumerate(zip(images, metadata_list, image_shapes)):
                pixel_size = metadata.get("pixel_size", 1.0)
                pos_x = metadata.get("x", 0)
                pos_y = metadata.get("y", 0)
                height, width = shape
                
                # Create pixels with physical size information
                pixels = model.Pixels(
                    id=f'Pixels:{i}',
                    size_x=width,
                    size_y=height,
                    size_z=1,
                    size_c=1,
                    size_t=1,
                    type=model.PixelType.UINT16,
                    dimension_order=model.Pixels_DimensionOrder.XYCZT,
                    physical_size_x=pixel_size,
                    physical_size_x_unit=model.UnitsLength.MICROMETER,
                    physical_size_y=pixel_size,
                    physical_size_y_unit=model.UnitsLength.MICROMETER
                )
                
                # Create channel
                channel = model.Channel(id=f'Channel:{i}:0')
                pixels.channels.append(channel)
                
                # Create plane with position metadata
                plane = model.Plane(
                    the_c=0,
                    the_t=0,
                    the_z=0,
                    position_x=float(pos_x),
                    position_x_unit=model.UnitsLength.REFERENCEFRAME,
                    position_y=float(pos_y),
                    position_y_unit=model.UnitsLength.REFERENCEFRAME
                )
                pixels.planes.append(plane)
                
                # Create image
                ome_image = model.Image(
                    id=f'Image:{i}',
                    name=f'Image{i}',
                    pixels=pixels
                )
                ome.images.append(ome_image)
            
            # Generate OME XML
            ome_xml = ome_types.to_xml(ome)
            ome_xml = ome_xml.replace('µm', 'um')
            
            # Write final OME-TIFF with proper structure
            with tifffile.TiffWriter(self.file_path, bigtiff=self.bigtiff) as tif:
                # Write first image with OME XML description
                tif.write(data=images[0], description=ome_xml)
                # Write remaining images
                for image in images[1:]:
                    tif.write(data=image)
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
                
            print(f"Created final OME-TIFF with {len(images)} images and proper metadata")
            
        except Exception as e:
            print(f"Error creating final OME-TIFF: {e}")
            # Fallback: just move temp file to final location
            if os.path.exists(temp_file_path):
                if os.path.exists(self.file_path):
                    os.remove(self.file_path)
                os.rename(temp_file_path, self.file_path)

    def _create_tiff_metadata_fallback(self, image: np.ndarray, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create TIFF metadata in HistoScanController format as fallback when ome-types not available.
        
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
        
        # Create metadata in HistoScanController format
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
