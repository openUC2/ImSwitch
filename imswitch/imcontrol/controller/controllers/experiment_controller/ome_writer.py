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
import threading
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import numpy as np
try:
    from .OmeTiffStitcher import OmeTiffStitcher
    from .SingleTiffWriter import SingleTiffWriter
except ImportError:
    from OmeTiffStitcher import OmeTiffStitcher
    from SingleTiffWriter import SingleTiffWriter


@dataclass
class OMEWriterConfig:
    """Configuration for OME writer behavior."""
    write_tiff: bool = False
    write_zarr: bool = True
    write_stitched_tiff: bool = False  # New option for stitched TIFF
    write_tiff_single: bool = False  # Append images in a single TIFF file
    min_period: float = 0.2
    compression: str = "zlib"
    zarr_compressor = None
    pixel_size: float = 1.0  # pixel size in microns
    dimension_seperator: str = "/"
    # Multi-dimensional support
    n_time_points: int = 1  # Number of time points
    n_z_planes: int = 1     # Number of z planes
    n_channels: int = 1     # Number of channels
    
    
    def __post_init__(self):
        def get_zarr_compressor(zarr_version="3"):
            if zarr_version == "3":
                # Use the v3 codec
                return zarr.codecs.BloscCodec(cname="zstd", clevel=3, shuffle=zarr.codecs.BloscShuffle.bitshuffle)
            else:
                # Use the v2 codec
                import numcodecs
                return numcodecs.Blosc("zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)
            
        if self.zarr_compressor is None:
            if zarr.__version__.startswith("3"):
                self.zarr_compressor = get_zarr_compressor("3")
            else:
                self.zarr_compressor = get_zarr_compressor("2")
            


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
        self.x_start, self.y_start, self.x_step, self.y_step = grid_geometry # TODO: this should be set per each grid 
        self.config = config
        self.logger = logger
        
        # Zarr components
        self.store = None
        self.root = None
        self.canvas = None
        
        # Stitched TIFF writer
        self.tiff_stitcher = None
        
        # Single TIFF writer for appending tiles
        self.single_tiff_writer = None
        
        # Timing
        self.t_last = time.time()
        
        # Initialize storage if needed
        if config.write_zarr:
            self._setup_zarr_store()
        
        if config.write_stitched_tiff:
            self._setup_tiff_stitcher()
            
        if config.write_tiff_single:
            self._setup_single_tiff_writer()
    
    def _setup_zarr_store(self):
        """Set up the OME-Zarr store and canvas."""
        # Use path string for Zarr v3 compatibility
        self.store = str(self.file_paths.zarr_dir)
        # Use context manager to ensure proper closing (if supported)
        self.root = zarr.open_group(store=self.store, mode="w")
        self.canvas = self.root.create_array(
            name="0",
            shape=(
                int(self.config.n_time_points),
                int(self.config.n_channels),
                int(self.config.n_z_planes),
                int(self.ny * self.tile_h),
                int(self.nx * self.tile_w)
            ),  # t c z y x
            chunks=(1, 1, 1, int(self.tile_h), int(self.tile_w)),
            dtype="uint16",
            compressor=self.config.zarr_compressor # Has proper BLOSC compression (based on v3)
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
    
    def _setup_single_tiff_writer(self):
        """Set up the single TIFF writer for appending tiles with metadata."""
        single_tiff_path = os.path.join(self.file_paths.base_dir, "single_tiles.ome.tif")
        self.single_tiff_writer = SingleTiffWriter(single_tiff_path, bigtiff=True)
        self.single_tiff_writer.start()
        if self.logger:
            self.logger.debug(f"Single TIFF writer initialized: {single_tiff_path}")
    
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
        
        # Write to single TIFF if requested
        if self.config.write_tiff_single and self.single_tiff_writer is not None:
            self._write_single_tiff_tile(frame, metadata)
        
        # Throttle writes if needed
        self._throttle_writes()
        
        return result
    
    def _write_tiff_tile(self, frame, metadata: Dict[str, Any]):
        """Write individual TIFF tile."""
        # Include time and z information in filename
        t_idx = metadata.get("time_index", 0)
        z_idx = metadata.get("z_index", 0)
        c_idx = metadata.get("channel_index", 0)
        
        tiff_name = (
            f"F{metadata['runningNumber']:06d}_"
            f"t{t_idx:03d}_c{c_idx:03d}_z{z_idx:03d}_"
            f"x{metadata['x']:.1f}_y{metadata['y']:.1f}_"
            f"{metadata['illuminationChannel']}_{metadata['illuminationValue']}.ome.tif"
        )
        tiff_path = os.path.join(self.file_paths.tiff_dir, tiff_name)
        tif.imwrite(tiff_path, frame, compression=self.config.compression)
    
    def _write_zarr_tile(self, frame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Write tile to Zarr canvas and return chunk information."""
        # Calculate grid position
        ix = int(round((metadata["x"] - self.x_start) / np.max((self.x_step,1))))
        iy = int(round((metadata["y"] - self.y_start) / np.max((self.y_step,1))))
        
        # Get time, channel, and z indices from metadata
        t_idx = metadata.get("time_index", 0)
        c_idx = metadata.get("channel_index", 0) 
        z_idx = metadata.get("z_index", 0)
        
        # Validate indices are within bounds
        t_idx = min(t_idx, self.config.n_time_points - 1)
        c_idx = min(c_idx, self.config.n_channels - 1)
        z_idx = min(z_idx, self.config.n_z_planes - 1)
        
        # Calculate canvas coordinates
        y0, y1 = iy * self.tile_h, (iy + 1) * self.tile_h
        x0, x1 = ix * self.tile_w, (ix + 1) * self.tile_w
        
        # Write to canvas with proper indexing
        self.canvas[t_idx, c_idx, z_idx, y0:y1, x0:x1] = frame
        
        # Return chunk information for frontend updates
        rel_chunk = f"0/{iy}.{ix}"  # NGFF v0.4 layout
        return {
            "rel_chunk": rel_chunk,
            "grid_pos": (ix, iy),
            "canvas_bounds": (x0, x1, y0, y1),
            "t_idx": t_idx,
            "c_idx": c_idx,
            "z_idx": z_idx
        }
    
    def _write_stitched_tiff_tile(self, frame, metadata: Dict[str, Any]):
        """Write tile to stitched TIFF using OmeTiffStitcher."""
        # Calculate grid index from position
        ix = int(round((metadata["x"] - self.x_start) / np.max((self.x_step,1))))
        iy = int(round((metadata["y"] - self.y_start) / np.max((1,self.y_step))))
        
        self.tiff_stitcher.add_image(
            image=frame,
            position_x=metadata["x"],
            position_y=metadata["y"],
            index_x=ix,
            index_y=iy,
            pixel_size=self.config.pixel_size
        )
    
    def _write_single_tiff_tile(self, frame, metadata: Dict[str, Any]):
        """Write tile to single TIFF using SingleTiffWriter."""
        # Add pixel size to metadata for the single TIFF writer
        metadata_with_pixel_size = metadata.copy()
        metadata_with_pixel_size["pixel_size"] = self.config.pixel_size
        
        self.single_tiff_writer.add_image(
            image=frame,
            metadata=metadata_with_pixel_size
        )
    
    def _throttle_writes(self):
        """Throttle disk writes if needed."""
        t_now = time.time()
        if t_now - self.t_last < self.config.min_period:
            time.sleep(self.config.min_period - (t_now - self.t_last))
        self.t_last = t_now
    
    def _build_vanilla_zarr_pyramids(self):
        """
        Build pyramid levels for OME-Zarr format using memory-efficient processing.
        This creates downsampled versions of the full resolution data.
        """
        if self.canvas is None:
            return
            
        # Start pyramid generation in background thread for better performance
        def _generate_pyramids():
            try:
                self._generate_pyramids_sync()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Pyramid generation failed: {e}")
        
        # Run in background thread to avoid blocking
        pyramid_thread = threading.Thread(target=_generate_pyramids, daemon=True)
        pyramid_thread.start()
        
        # For now, wait for completion to maintain backward compatibility
        # In future versions, this could be made fully asynchronous
        pyramid_thread.join()
    
    def _generate_pyramids_sync(self):
        """
        Synchronous pyramid generation with memory-efficient processing.
        """
        # Get dimensions of the full resolution data
        full_shape = self.canvas.shape[-2:]  # Get y,x dimensions from t,c,z,y,x
        
        # Create pyramid levels with 2x downsampling
        max_levels = 4  # Create up to 4 pyramid levels
        
        for level in range(1, max_levels):
            # Calculate new shape for this level (2x downsampling)
            new_shape = (full_shape[0] // (2**level), full_shape[1] // (2**level))
            
            # Stop if the image becomes too small
            if new_shape[0] < 64 or new_shape[1] < 64:
                break
            
            # Create new dataset for this pyramid level
            level_canvas = self.root.create_dataset(
                str(level),
                shape=(1, 1, 1, int(new_shape[0]), int(new_shape[1])),  # t c z y x
                chunks=(1, 1, 1, int(min(self.tile_h, new_shape[0])), int(min(self.tile_w, new_shape[1]))),
                dtype="uint16",
                compressor=self.config.zarr_compressor
            )
            
            # Process data in chunks to avoid loading entire array into memory
            self._downsample_in_chunks(self.canvas, level_canvas, level)
            
            if self.logger:
                self.logger.debug(f"Created pyramid level {level} with shape {new_shape}")
        
        # Update the multiscales metadata to include all pyramid levels
        self._update_multiscales_metadata()
    
    def _downsample_in_chunks(self, source_canvas, target_canvas, level):
        """
        Downsample data in chunks to avoid memory issues with large arrays.
        
        Args:
            source_canvas: Source zarr array
            target_canvas: Target zarr array for downsampled data
            level: Pyramid level (1, 2, 3, ...)
        """
        # Get source and target shapes
        source_shape = source_canvas.shape[-2:]  # y, x
        target_shape = target_canvas.shape[-2:]  # y, x
        
        # Define chunk size for processing (adjust based on available memory)
        chunk_size = min(1024, source_shape[0], source_shape[1])
        
        # Process in overlapping chunks to handle downsampling
        downsample_factor = 2 ** level
        
        for y_start in range(0, target_shape[0], chunk_size):
            y_end = min(y_start + chunk_size, target_shape[0])
            
            for x_start in range(0, target_shape[1], chunk_size):
                x_end = min(x_start + chunk_size, target_shape[1])
                
                # Calculate corresponding region in source
                src_y_start = y_start * downsample_factor
                src_y_end = min(y_end * downsample_factor, source_shape[0])
                src_x_start = x_start * downsample_factor
                src_x_end = min(x_end * downsample_factor, source_shape[1])
                
                # Read source data chunk
                source_chunk = source_canvas[0, 0, 0, src_y_start:src_y_end, src_x_start:src_x_end]
                
                # Downsample using simple subsampling
                downsampled_chunk = source_chunk[::downsample_factor, ::downsample_factor]
                
                # Calculate actual target region size
                actual_y_size = min(downsampled_chunk.shape[0], y_end - y_start)
                actual_x_size = min(downsampled_chunk.shape[1], x_end - x_start)
                
                # Write downsampled chunk to target
                target_canvas[0, 0, 0, y_start:y_start+actual_y_size, x_start:x_start+actual_x_size] = \
                    downsampled_chunk[:actual_y_size, :actual_x_size]
    
    def _update_multiscales_metadata(self):
        """Update the multiscales metadata to include all pyramid levels."""
        datasets = []
        for level_name in sorted(self.root.keys(), key=int):
            level_int = int(level_name)
            scale_factor = 2 ** level_int
            datasets.append({
                "path": level_name,
                "coordinateTransformations": [
                    {"type": "scale", "scale": [1, 1, 1, scale_factor, scale_factor]}
                ]
            })
        
        # Update multiscales metadata with all levels
        self.root.attrs["multiscales"] = [{
            "version": "0.4",
            "datasets": datasets,
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space"},
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
        }]
    
    def finalize(self):
        """Finalize the writing process and optionally build pyramids."""
        if self.config.write_zarr and self.store is not None:
            try:
                self._build_vanilla_zarr_pyramids()
                if self.logger:
                    self.logger.info("Vanilla Zarr pyramid generated successfully")
            except Exception as err:
                if self.logger:
                    self.logger.warning(f"Pyramid generation failed: {err}")
        
        # Close stitched TIFF writer
        if self.config.write_stitched_tiff and self.tiff_stitcher is not None:
            self.tiff_stitcher.close()
            if self.logger:
                self.logger.info("Stitched TIFF file completed")
        
        # Close single TIFF writer
        if self.config.write_tiff_single and self.single_tiff_writer is not None:
            self.single_tiff_writer.close()
            if self.logger:
                self.logger.info("Single TIFF file completed")
        
        if self.logger:
            self.logger.info(f"OME writer finalized for {self.file_paths.base_dir}")
    
    def get_zarr_url(self) -> Optional[str]:
        """Get the relative Zarr URL for frontend streaming."""
        if self.config.write_zarr:
            return str(self.file_paths.zarr_dir)
        return None