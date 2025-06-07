"""
Unified OME writer for both TIFF and OME-Zarr formats.

This module provides a reusable writer that can handle both individual TIFF files
and OME-Zarr mosaics, supporting both fast stage scan and normal stage scan modes.
"""

import os
import time
import zarr
import numcodecs
import tifffile as tif
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

try:
    from .OmeTiffStitcher import OmeTiffStitcher
except ImportError:
    from OmeTiffStitcher import OmeTiffStitcher


@dataclass
class OMEWriterConfig:
    """Configuration for OME writer behavior."""
    write_tiff: bool = False
    write_zarr: bool = True
    write_stitched_tiff: bool = False  # New option for stitched TIFF
    min_period: float = 0.2
    compression: str = "zlib"
    zarr_compressor = None
    pixel_size: float = 1.0  # pixel size in microns
    
    def __post_init__(self):
        if self.zarr_compressor is None:
            self.zarr_compressor = numcodecs.Blosc("zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)


class OMEWriter:
    """
    Unified writer for OME-TIFF and OME-Zarr formats.
    
    This class extracts the reusable logic from _writer_loop_ome to support
    both fast stage scan and normal stage scan writing operations.
    """
    
    def __init__(self, file_paths, tile_shape, grid_shape, grid_geometry, config: OMEWriterConfig, logger=None):
        """
        Initialize the OME writer.
        
        Args:
            file_paths: OMEFileStorePaths object with tiff_dir, zarr_dir, base_dir
            tile_shape: (height, width) of individual tiles
            grid_shape: (nx, ny) grid dimensions
            grid_geometry: (x_start, y_start, x_step, y_step) for positioning
            config: OMEWriterConfig for writer behavior
            logger: Logger instance for debugging
        """
        self.file_paths = file_paths
        self.tile_h, self.tile_w = tile_shape
        self.nx, self.ny = grid_shape
        self.x_start, self.y_start, self.x_step, self.y_step = grid_geometry
        self.config = config
        self.logger = logger
        
        # Zarr components
        self.store = None
        self.root = None
        self.canvas = None
        
        # Stitched TIFF writer
        self.tiff_stitcher = None
        
        # Timing
        self.t_last = time.time()
        
        # Initialize storage if needed
        if config.write_zarr:
            self._setup_zarr_store()
        
        if config.write_stitched_tiff:
            self._setup_tiff_stitcher()
    
    def _setup_zarr_store(self):
        """Set up the OME-Zarr store and canvas."""
        self.store = zarr.DirectoryStore(str(self.file_paths.zarr_dir))
        self.root = zarr.group(self.store, overwrite=True)
        self.canvas = self.root.create_dataset(
            "0",
            shape=(1, 1, 1, self.ny * self.tile_h, self.nx * self.tile_w),  # t c z y x
            chunks=(1, 1, 1, self.tile_h, self.tile_w),
            dtype="uint16",
            compressor=self.config.zarr_compressor,
            dimension_separator="/"
        )
        
        # Set OME-Zarr metadata
        self.root.attrs["multiscales"] = [{
            "version": "0.4",
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [1, 1, 1, 1, 1]}
                    ]
                }
            ],
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }]
    
    def _setup_tiff_stitcher(self):
        """Set up the TIFF stitcher for creating stitched OME-TIFF files."""
        stitched_tiff_path = os.path.join(self.file_paths.base_dir, "stitched.ome.tif")
        self.tiff_stitcher = OmeTiffStitcher(stitched_tiff_path, bigtiff=True)
        self.tiff_stitcher.start()
        if self.logger:
            self.logger.debug(f"TIFF stitcher initialized: {stitched_tiff_path}")
    
    def write_frame(self, frame, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Write a single frame to both TIFF and/or Zarr formats.
        
        Args:
            frame: Image data as numpy array
            metadata: Dictionary containing position and other metadata
            
        Returns:
            Dictionary with information about the written chunk (for Zarr)
        """
        result = {}
        
        # Write individual TIFF file if requested
        if self.config.write_tiff:
            self._write_tiff_tile(frame, metadata)
        
        # Write to Zarr canvas if requested
        if self.config.write_zarr and self.canvas is not None:
            chunk_info = self._write_zarr_tile(frame, metadata)
            result.update(chunk_info)
        
        # Write to stitched TIFF if requested
        if self.config.write_stitched_tiff and self.tiff_stitcher is not None:
            self._write_stitched_tiff_tile(frame, metadata)
        
        # Throttle writes if needed
        self._throttle_writes()
        
        return result
    
    def _write_tiff_tile(self, frame, metadata: Dict[str, Any]):
        """Write individual TIFF tile."""
        tiff_name = (
            f"F{metadata['runningNumber']:06d}_"
            f"x{metadata['x']:.1f}_y{metadata['y']:.1f}_"
            f"{metadata['illuminationChannel']}_{metadata['illuminationValue']}.ome.tif"
        )
        tiff_path = os.path.join(self.file_paths.tiff_dir, tiff_name)
        tif.imwrite(tiff_path, frame, compression=self.config.compression)
    
    def _write_zarr_tile(self, frame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Write tile to Zarr canvas and return chunk information."""
        # Calculate grid position
        ix = int(round((metadata["x"] - self.x_start) / self.x_step))
        iy = int(round((metadata["y"] - self.y_start) / self.y_step))
        
        # Calculate canvas coordinates
        y0, y1 = iy * self.tile_h, (iy + 1) * self.tile_h
        x0, x1 = ix * self.tile_w, (ix + 1) * self.tile_w
        
        # Write to canvas
        self.canvas[0, 0, 0, y0:y1, x0:x1] = frame
        
        # Return chunk information for frontend updates
        rel_chunk = f"0/{iy}.{ix}"  # NGFF v0.4 layout
        return {
            "rel_chunk": rel_chunk,
            "grid_pos": (ix, iy),
            "canvas_bounds": (x0, x1, y0, y1)
        }
    
    def _write_stitched_tiff_tile(self, frame, metadata: Dict[str, Any]):
        """Write tile to stitched TIFF using OmeTiffStitcher."""
        # Calculate grid index from position
        ix = int(round((metadata["x"] - self.x_start) / self.x_step))
        iy = int(round((metadata["y"] - self.y_start) / self.y_step))
        
        self.tiff_stitcher.add_image(
            image=frame,
            position_x=metadata["x"],
            position_y=metadata["y"],
            index_x=ix,
            index_y=iy,
            pixel_size=self.config.pixel_size
        )
    
    def _throttle_writes(self):
        """Throttle disk writes if needed."""
        t_now = time.time()
        if t_now - self.t_last < self.config.min_period:
            time.sleep(self.config.min_period - (t_now - self.t_last))
        self.t_last = t_now
    
    def finalize(self):
        """Finalize the writing process and optionally build pyramids."""
        if self.config.write_zarr and self.store is not None:
            try:
                from ome_zarr.writer import to_multiscales
                to_multiscales(self.store, scale_factors=[[2, 2, 2]])
                if self.logger:
                    self.logger.info("OME-Zarr pyramid generated successfully")
            except Exception as err:
                if self.logger:
                    self.logger.warning(f"Pyramid generation failed: {err}")
        
        # Close stitched TIFF writer
        if self.config.write_stitched_tiff and self.tiff_stitcher is not None:
            self.tiff_stitcher.close()
            if self.logger:
                self.logger.info("Stitched TIFF file completed")
        
        if self.logger:
            self.logger.info(f"OME writer finalized for {self.file_paths.base_dir}")
    
    def get_zarr_url(self) -> Optional[str]:
        """Get the relative Zarr URL for frontend streaming."""
        if self.config.write_zarr:
            return str(self.file_paths.zarr_dir)
        return None