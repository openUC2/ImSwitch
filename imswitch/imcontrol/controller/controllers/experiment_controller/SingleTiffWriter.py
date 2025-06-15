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
    Single TIFF writer that appends tiles as a timelapse series with xyz positions and channels.
    
    This writer is specifically designed for single tile scans where each scan position
    contains only one tile, and all tiles should be appended to a single TIFF file
    as a timelapse series. Each tile becomes a timepoint in the series, with spatial
    coordinates stored in metadata for Fiji compatibility.
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
        self.images_buffer = []  # Buffer to collect images before writing
        self.metadata_buffer = []  # Buffer to collect metadata
        
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
        Create OME metadata dictionary for timelapse series with spatial coordinates.
        
        Args:
            image: Image array
            metadata: Original metadata dictionary
            
        Returns:
            OME-compatible metadata dictionary for timelapse series
        """
        # Extract positions and indices
        position_x = metadata.get("x", 0.0)
        position_y = metadata.get("y", 0.0)
        position_z = metadata.get("z", 0.0)
        z_index = metadata.get("z_index", 0)
        c_index = metadata.get("channel_index", 0)
        
        # Get pixel size from metadata or use default
        pixel_size = metadata.get("pixel_size", 1.0)
        
        # Create OME metadata structure for timelapse series
        # Each tile becomes a timepoint with spatial coordinates in metadata
        ome_metadata = {
            "Pixels": {
                "PhysicalSizeX": pixel_size,
                "PhysicalSizeXUnit": "µm",
                "PhysicalSizeY": pixel_size,
                "PhysicalSizeYUnit": "µm",
                "SizeX": image.shape[1],
                "SizeY": image.shape[0],
                "SizeZ": 1,
                "SizeT": 1,  # Each image is one timepoint
                "SizeC": 1,
                "Type": "uint16",
                "DimensionOrder": "XYZCT"
            },
            "Plane": {
                # Store spatial coordinates for reconstruction
                "PositionX": position_x,
                "PositionY": position_y, 
                "PositionZ": position_z,
                "TheZ": z_index,
                "TheT": self.timepoint_index,  # Use sequential timepoint indexing
                "TheC": c_index
            },
            "Channel": {
                "ID": f"Channel:{c_index}",
                "SamplesPerPixel": 1,
                "IlluminationType": metadata.get("illuminationChannel", "Unknown")
            },
            # Add spatial metadata for reconstruction in Fiji
            "TilePosition": {
                "X": position_x,
                "Y": position_y,
                "Z": position_z,
                "ZIndex": z_index,
                "ChannelIndex": c_index,
                "TimeIndex": self.timepoint_index
            }
        }
        
        return ome_metadata
    
    def _process_queue(self):
        """
        Background loop: collect images from queue and write them as a multi-page TIFF stack.
        """
        # Ensure the folder exists
        if not os.path.exists(os.path.dirname(self.file_path)):
            os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
            
        # Process images from queue and buffer them
        while self.is_running or len(self.queue) > 0:
            with self.lock:
                if self.queue:
                    image, metadata = self.queue.popleft()
                else:
                    image = None
            
            if image is not None:
                try:
                    # Add image and metadata to buffers
                    self.images_buffer.append(image)
                    self.metadata_buffer.append(metadata)
                    self.image_count += 1
                    self.timepoint_index += 1
                    print(f"Buffered image as timepoint %d for single TIFF: %s" % (metadata["TilePosition"]["TimeIndex"], self.file_path))
                except Exception as e:
                    print(f"Error buffering image for single TIFF: {e}")
            else:
                # Sleep briefly to avoid busy loop when queue is empty
                time.sleep(0.01)
        
        # Write all buffered images as a single multi-page TIFF
        self._write_buffered_images()
    
    def _write_buffered_images(self):
        """
        Write all buffered images as a single multi-page TIFF stack.
        """
        if not self.images_buffer:
            print("No images to write to single TIFF")
            return
            
        try:
            # Stack all images into a 3D array (timepoints, height, width)
            image_stack = np.stack(self.images_buffer, axis=0)
            
            # Create combined metadata for the entire stack
            # Use the metadata from the first image as template
            if self.metadata_buffer:
                first_metadata = self.metadata_buffer[0]
                
                # Create OME metadata for the entire stack
                ome_metadata = {
                    'axes': 'TYX',  # Time, Y, X dimensions
                    'PhysicalSizeX': first_metadata.get("pixel_size", 1.0),
                    'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': first_metadata.get("pixel_size", 1.0), 
                    'PhysicalSizeYUnit': 'µm',
                    'SizeT': len(self.images_buffer),
                    'SizeX': image_stack.shape[2],
                    'SizeY': image_stack.shape[1],
                    'SizeZ': 1,
                    'SizeC': 1
                }
                
                # Add position information for each timepoint
                positions = []
                for i, metadata in enumerate(self.metadata_buffer):
                    tile_pos = metadata.get("TilePosition", {})
                    positions.append({
                        'TimeIndex': i,
                        'PositionX': tile_pos.get("X", 0.0),
                        'PositionY': tile_pos.get("Y", 0.0),
                        'PositionZ': tile_pos.get("Z", 0.0)
                    })
                
                # Add position metadata
                ome_metadata['Positions'] = positions
            else:
                ome_metadata = None
            
            # Write the entire stack as a single multi-page TIFF
            tifffile.imwrite(
                self.file_path,
                image_stack,
                metadata=ome_metadata,
                bigtiff=self.bigtiff,
                compression='zlib'
            )
            
            print(f"Wrote {len(self.images_buffer)} images to single TIFF stack: {self.file_path}")
            
        except Exception as e:
            print(f"Error writing buffered images to single TIFF: {e}")
        finally:
            # Clear buffers
            self.images_buffer.clear()
            self.metadata_buffer.clear()
    
    def close(self):
        """Close the single TIFF writer and ensure all buffered images are written."""
        self.stop()
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        
        # Write any remaining buffered images
        if self.images_buffer:
            self._write_buffered_images()
            
        self.is_running = False
        self.queue.clear()
        print(f"Single TIFF timelapse writer closed. Total timepoints written: {self.image_count}")