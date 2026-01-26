"""
Stitched TIFF writer for 2D mosaic/tile acquisitions.

Provides writers for creating stitched mosaic images from tile scans,
supporting both in-memory stitching and file-based streaming.
"""

import os
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
import numpy as np

try:
    import tifffile as tiff
except ImportError:
    tiff = None

logger = logging.getLogger(__name__)


@dataclass
class TileInfo:
    """Information about a single tile in a mosaic."""
    row: int
    col: int
    x_pos_um: float  # Stage X position in microns
    y_pos_um: float  # Stage Y position in microns
    z_pos_um: float = 0.0  # Stage Z position in microns
    channel: str = "default"
    time_point: int = 0
    z_plane: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MosaicConfig:
    """Configuration for a mosaic acquisition."""
    nx: int  # Number of tiles in X
    ny: int  # Number of tiles in Y
    tile_width: int  # Tile width in pixels
    tile_height: int  # Tile height in pixels
    overlap_x: float = 0.0  # Overlap fraction in X (0-1)
    overlap_y: float = 0.0  # Overlap fraction in Y (0-1)
    pixel_size_um: float = 1.0  # Pixel size in microns
    origin_x_um: float = 0.0  # X origin of mosaic in microns
    origin_y_um: float = 0.0  # Y origin of mosaic in microns
    
    @property
    def step_x_pixels(self) -> int:
        """Step size in pixels between tile centers in X."""
        return int(self.tile_width * (1 - self.overlap_x))
    
    @property
    def step_y_pixels(self) -> int:
        """Step size in pixels between tile centers in Y."""
        return int(self.tile_height * (1 - self.overlap_y))
    
    @property
    def canvas_width(self) -> int:
        """Total canvas width in pixels."""
        return self.tile_width + (self.nx - 1) * self.step_x_pixels
    
    @property
    def canvas_height(self) -> int:
        """Total canvas height in pixels."""
        return self.tile_height + (self.ny - 1) * self.step_y_pixels


class StitchedTiffWriter:
    """
    Writer for creating 2D stitched OME-TIFF mosaics.
    
    This writer collects tiles from a scan and stitches them into
    a single large image with proper OME metadata.
    
    Usage:
        config = MosaicConfig(nx=5, ny=5, tile_width=2048, tile_height=2048)
        writer = StitchedTiffWriter('/path/to/mosaic.ome.tiff', config)
        writer.open()
        
        for tile_info, tile_image in tiles:
            writer.add_tile(tile_info, tile_image)
        
        writer.close()  # Writes the stitched image
    """
    
    def __init__(self, filepath: str, config: MosaicConfig, dtype=np.uint16):
        """
        Initialize the stitched TIFF writer.
        
        Args:
            filepath: Output file path
            config: MosaicConfig with mosaic dimensions
            dtype: Data type for the canvas (default: uint16)
        """
        self._filepath = filepath
        self._config = config
        self._dtype = dtype
        self._logger = logging.getLogger(self.__class__.__name__)
        
        # Canvas for stitching
        self._canvas: Optional[np.ndarray] = None
        self._tile_count = 0
        self._is_open = False
        self._lock = threading.Lock()
        
        # Tile tracking
        self._received_tiles: Dict[Tuple[int, int], bool] = {}
        
        # Metadata
        self._metadata: Dict[str, Any] = {}
        self._tile_metadata: List[Dict[str, Any]] = []
    
    def open(self):
        """Initialize the canvas for stitching."""
        if self._is_open:
            return
        
        # Create canvas
        canvas_shape = (self._config.canvas_height, self._config.canvas_width)
        self._canvas = np.zeros(canvas_shape, dtype=self._dtype)
        
        # Initialize tile tracking
        for row in range(self._config.ny):
            for col in range(self._config.nx):
                self._received_tiles[(row, col)] = False
        
        self._is_open = True
        self._tile_count = 0
        self._logger.info(f"StitchedTiffWriter opened: canvas {canvas_shape}")
    
    def add_tile(self, tile_info: TileInfo, image: np.ndarray):
        """
        Add a tile to the mosaic.
        
        Args:
            tile_info: TileInfo with position information
            image: Tile image data
        """
        if not self._is_open:
            raise RuntimeError("Writer not open")
        
        with self._lock:
            row, col = tile_info.row, tile_info.col
            
            if row >= self._config.ny or col >= self._config.nx:
                self._logger.warning(f"Tile ({row}, {col}) outside mosaic bounds")
                return
            
            # Calculate position in canvas
            x_start = col * self._config.step_x_pixels
            y_start = row * self._config.step_y_pixels
            
            # Handle size mismatches
            img_h, img_w = image.shape[:2]
            x_end = min(x_start + img_w, self._config.canvas_width)
            y_end = min(y_start + img_h, self._config.canvas_height)
            
            src_w = x_end - x_start
            src_h = y_end - y_start
            
            # Place tile in canvas
            if image.ndim == 2:
                self._canvas[y_start:y_end, x_start:x_end] = image[:src_h, :src_w]
            else:
                # Take first channel if multi-channel
                self._canvas[y_start:y_end, x_start:x_end] = image[:src_h, :src_w, 0]
            
            # Track tile
            self._received_tiles[(row, col)] = True
            self._tile_count += 1
            
            # Store tile metadata
            self._tile_metadata.append({
                'row': row,
                'col': col,
                'x_um': tile_info.x_pos_um,
                'y_um': tile_info.y_pos_um,
                'z_um': tile_info.z_pos_um,
            })
            
            self._logger.debug(f"Added tile ({row}, {col}) at ({x_start}, {y_start})")
    
    def add_tile_at_position(self, image: np.ndarray, 
                             x_pos_um: float, y_pos_um: float,
                             z_pos_um: float = 0.0):
        """
        Add a tile using absolute stage position.
        
        Calculates row/col from stage coordinates.
        
        Args:
            image: Tile image data
            x_pos_um: Stage X position in microns
            y_pos_um: Stage Y position in microns
            z_pos_um: Stage Z position in microns
        """
        # Calculate row/col from position
        step_x_um = self._config.step_x_pixels * self._config.pixel_size_um
        step_y_um = self._config.step_y_pixels * self._config.pixel_size_um
        
        col = round((x_pos_um - self._config.origin_x_um) / step_x_um)
        row = round((y_pos_um - self._config.origin_y_um) / step_y_um)
        
        tile_info = TileInfo(
            row=row, col=col,
            x_pos_um=x_pos_um, y_pos_um=y_pos_um, z_pos_um=z_pos_um
        )
        self.add_tile(tile_info, image)
    
    def set_metadata(self, **kwargs):
        """Set metadata for the stitched image."""
        self._metadata.update(kwargs)
    
    def close(self):
        """Write the stitched image and close."""
        if not self._is_open:
            return
        
        try:
            # Build OME metadata
            ome_metadata = self._build_ome_metadata()
            
            # Ensure directory exists
            dirname = os.path.dirname(self._filepath)
            if dirname:
                os.makedirs(dirname, exist_ok=True)
            
            # Write stitched image
            tiff.imwrite(
                self._filepath,
                self._canvas,
                metadata=ome_metadata,
                imagej=False,
            )
            
            self._logger.info(
                f"StitchedTiffWriter closed: {self._tile_count} tiles, "
                f"canvas {self._canvas.shape}"
            )
            
        except Exception as e:
            self._logger.error(f"Error writing stitched TIFF: {e}")
            raise
        
        finally:
            self._canvas = None
            self._is_open = False
            self._received_tiles.clear()
            self._tile_metadata.clear()
    
    def _build_ome_metadata(self) -> Dict[str, Any]:
        """Build OME-TIFF metadata for the stitched image."""
        metadata = {
            'ImageDescription': 'Stitched mosaic from tile scan',
            'PhysicalSizeX': self._config.pixel_size_um,
            'PhysicalSizeY': self._config.pixel_size_um,
            'PhysicalSizeXUnit': 'µm',
            'PhysicalSizeYUnit': 'µm',
            'MosaicNX': self._config.nx,
            'MosaicNY': self._config.ny,
            'TileWidth': self._config.tile_width,
            'TileHeight': self._config.tile_height,
            'OverlapX': self._config.overlap_x,
            'OverlapY': self._config.overlap_y,
            'TileCount': self._tile_count,
            'OriginXum': self._config.origin_x_um,
            'OriginYum': self._config.origin_y_um,
        }
        
        # Add user-provided metadata
        metadata.update(self._metadata)
        
        return metadata
    
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    @property
    def tile_count(self) -> int:
        return self._tile_count
    
    @property
    def expected_tiles(self) -> int:
        return self._config.nx * self._config.ny
    
    @property
    def missing_tiles(self) -> List[Tuple[int, int]]:
        """Get list of missing tile positions."""
        return [pos for pos, received in self._received_tiles.items() if not received]
    
    def get_canvas(self) -> Optional[np.ndarray]:
        """Get the current stitched canvas (for preview)."""
        return self._canvas.copy() if self._canvas is not None else None


class StreamingStitchedTiffWriter:
    """
    Streaming writer that writes tiles directly to disk.
    
    For very large mosaics where the full canvas doesn't fit in memory,
    this writer uses tifffile's BigTIFF support to write tiles progressively.
    
    Note: This creates a multi-page TIFF where each tile is a separate page,
    with position metadata for later stitching.
    """
    
    def __init__(self, filepath: str, config: MosaicConfig, bigtiff: bool = True):
        """
        Initialize the streaming stitched TIFF writer.
        
        Args:
            filepath: Output file path
            config: MosaicConfig with mosaic dimensions
            bigtiff: Use BigTIFF format (required for >4GB files)
        """
        self._filepath = filepath
        self._config = config
        self._bigtiff = bigtiff
        self._logger = logging.getLogger(self.__class__.__name__)
        
        self._writer = None
        self._tile_count = 0
        self._is_open = False
        self._lock = threading.Lock()
    
    def open(self):
        """Open the TIFF file for writing."""
        if self._is_open:
            return
        
        dirname = os.path.dirname(self._filepath)
        if dirname:
            os.makedirs(dirname, exist_ok=True)
        
        self._writer = tiff.TiffWriter(self._filepath, bigtiff=self._bigtiff)
        self._is_open = True
        self._tile_count = 0
        self._logger.info(f"StreamingStitchedTiffWriter opened: {self._filepath}")
    
    def add_tile(self, tile_info: TileInfo, image: np.ndarray):
        """
        Write a tile to the TIFF file.
        
        Each tile is written as a separate page with position metadata.
        """
        if not self._is_open or self._writer is None:
            raise RuntimeError("Writer not open")
        
        with self._lock:
            # Build per-tile metadata
            metadata = {
                'Pixels': {
                    'PhysicalSizeX': self._config.pixel_size_um,
                    'PhysicalSizeXUnit': 'µm',
                    'PhysicalSizeY': self._config.pixel_size_um,
                    'PhysicalSizeYUnit': 'µm',
                },
                'Plane': {
                    'PositionX': tile_info.x_pos_um,
                    'PositionY': tile_info.y_pos_um,
                    'PositionZ': tile_info.z_pos_um,
                },
                'TileRow': tile_info.row,
                'TileCol': tile_info.col,
                'ImageID': self._tile_count,
            }
            
            # Write tile as a page
            self._writer.write(data=image, metadata=metadata)
            self._tile_count += 1
            
            self._logger.debug(f"Written tile ({tile_info.row}, {tile_info.col})")
    
    def close(self):
        """Close the TIFF file."""
        if not self._is_open:
            return
        
        if self._writer is not None:
            self._writer.close()
            self._writer = None
        
        self._is_open = False
        self._logger.info(f"StreamingStitchedTiffWriter closed: {self._tile_count} tiles")
    
    @property
    def is_open(self) -> bool:
        return self._is_open
    
    @property
    def tile_count(self) -> int:
        return self._tile_count


# =============================================================================
# Factory function
# =============================================================================

def create_stitched_writer(filepath: str, config: MosaicConfig,
                           streaming: bool = False, **kwargs):
    """
    Create a stitched TIFF writer.
    
    Args:
        filepath: Output file path
        config: MosaicConfig with mosaic dimensions
        streaming: If True, use streaming writer (for large mosaics)
        **kwargs: Additional arguments for the writer
        
    Returns:
        StitchedTiffWriter or StreamingStitchedTiffWriter
    """
    if streaming:
        return StreamingStitchedTiffWriter(filepath, config, **kwargs)
    else:
        return StitchedTiffWriter(filepath, config, **kwargs)


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
