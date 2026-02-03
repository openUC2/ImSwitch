"""
Unified OME writer for both TIFF and OME-Zarr formats.

This module provides a comprehensive writer that handles both individual TIFF files
and OME-Zarr mosaics, supporting multi-dimensional data (time, channel, z-stack).
It includes proper OME-NGFF metadata for both formats, as well as optional
streaming to OMERO servers.

Migrated from: imswitch/imcontrol/controller/controllers/experiment_controller/ome_writer.py

Features:
    - OME-Zarr with proper NGFF 0.4 metadata and pyramid generation
    - Stitched OME-TIFF via OmeTiffStitcher
    - Single TIFF files via SingleTiffWriter  
    - Individual TIFF files with position-based naming
    - Multi-dimensional support (TCZYX)
    - Physical coordinate transformations
    - Channel metadata (names, colors)
    - OMERO streaming upload for real-time server storage
"""

import os
import time
import threading
from typing import Optional, Dict, Any, List, TYPE_CHECKING
from dataclasses import dataclass

import numpy as np
import zarr
import tifffile as tif

# Import from local writers module
from .ome_tiff_stitcher import OmeTiffStitcher
from .single_tiff_writer import SingleTiffWriter
from .omero_uploader import (
    OMEROUploader,
    OMEROConnectionParams,
    TileMetadata,
    is_omero_available,
)

# Global registry for shared OMERO uploaders (for timelapse experiments)
_shared_omero_uploaders: Dict[str, OMEROUploader] = {}


@dataclass
class OMEWriterConfig:
    """
    Configuration for OME writer behavior.
    
    Controls which output formats are enabled and their parameters.
    
    Attributes:
        write_tiff: DEPRECATED - Use write_individual_tiffs instead.
                   Legacy mode that writes TIFFs without OME metadata.
        write_zarr: Write OME-Zarr format with pyramids
        write_stitched_tiff: Write stitched OME-TIFF mosaic
        write_tiff_single: Append tiles to a single TIFF file
        write_individual_tiffs: Write individual OME-TIFFs with position naming
                               and proper OME-XML metadata (RECOMMENDED)
        write_omero: Stream tiles to OMERO server
        omero_queue_size: Max tiles to queue for OMERO upload
        min_period: Minimum time between writes (throttling)
        compression: Compression algorithm for TIFF files
        zarr_compressor: Compressor for Zarr arrays
        pixel_size: Pixel size in microns (X/Y)
        pixel_size_z: Z-step size in microns
        dimension_separator: Separator for Zarr chunks
        n_time_points: Number of time points
        n_z_planes: Number of Z planes
        n_channels: Number of channels
        channel_names: List of channel names
        channel_colors: List of channel colors (hex without #)
        x_start: Starting X position in microns
        y_start: Starting Y position in microns
        z_start: Starting Z position in microns
        time_interval: Time interval in seconds
    """
    write_tiff: bool = False  # DEPRECATED - use write_individual_tiffs
    write_zarr: bool = True
    write_stitched_tiff: bool = False
    write_tiff_single: bool = False
    write_individual_tiffs: bool = False
    write_omero: bool = False
    omero_queue_size: int = 100
    min_period: float = 0.2
    compression: str = "zlib"
    zarr_compressor = None
    pixel_size: float = 1.0
    pixel_size_z: float = 1.0
    dimension_separator: str = "/"
    n_time_points: int = 1
    n_z_planes: int = 1
    n_channels: int = 1
    channel_names: Optional[List[str]] = None
    channel_colors: Optional[List[str]] = None
    x_start: float = 0.0
    y_start: float = 0.0
    z_start: float = 0.0
    time_interval: float = 1.0

    def __post_init__(self):
        """Initialize default compressor and channel metadata."""
        if self.zarr_compressor is None:
            self.zarr_compressor = _get_zarr_compressor()

        # Initialize default channel names if not provided
        if self.channel_names is None:
            self.channel_names = [f"Channel_{i}" for i in range(self.n_channels)]

        # Initialize default channel colors if not provided
        if self.channel_colors is None:
            # Default colors: green, red, blue, cyan, magenta, yellow
            default_colors = ["00FF00", "FF0000", "0000FF", "00FFFF", "FF00FF", "FFFF00"]
            self.channel_colors = [default_colors[i % len(default_colors)] for i in range(self.n_channels)]


def _get_zarr_compressor():
    """Get the appropriate Zarr compressor based on zarr version."""
    if zarr.__version__.startswith("3"):
        # Zarr v3 codec
        return zarr.codecs.BloscCodec(
            cname="zstd", clevel=3, 
            shuffle=zarr.codecs.BloscShuffle.bitshuffle
        )
    else:
        # Zarr v2 codec (using numcodecs)
        import numcodecs
        return numcodecs.Blosc("zstd", clevel=3, shuffle=numcodecs.Blosc.BITSHUFFLE)


class OMEFileStorePaths:
    """
    Helper class for managing OME file storage paths.
    
    Organizes output directories in a single timestamped folder structure:
    - base_dir/: Root experiment folder
    - base_dir/tiles/: TIFF tiles organized by timepoint
    - base_dir.ome.zarr: OME-Zarr store
    
    All TIFF files (tiles and individual images) go into the same tiles directory,
    organized by timepoint subfolders to avoid duplication.
    
    Attributes:
        base_dir: Base directory for all outputs
        tiff_dir: Directory for all TIFF files (tiles and individual)
        zarr_dir: Path for OME-Zarr store
    """
    
    def __init__(self, base_dir: str, shared_individual_tiffs_dir: Optional[str] = None):
        """
        Initialize OME file storage paths.
        
        Args:
            base_dir: Base directory for this writer's files
            shared_individual_tiffs_dir: Deprecated - ignored. All TIFFs go to tiff_dir.
        """
        self.base_dir = base_dir
        self.tiff_dir = os.path.join(base_dir, "tiles")
        self.zarr_dir = os.path.join(base_dir + ".ome.zarr")

        os.makedirs(self.tiff_dir, exist_ok=True)

    def get_timepoint_dir(self, timepoint_index: int) -> str:
        """
        Get or create directory for a specific timepoint.
        
        Args:
            timepoint_index: Zero-based index of the timepoint
            
        Returns:
            Path to the timepoint directory
        """
        timepoint_dir = os.path.join(
            self.tiff_dir,  # Use unified tiff_dir instead of separate individual_tiffs_dir
            f"timepoint_{timepoint_index:04d}"
        )
        os.makedirs(timepoint_dir, exist_ok=True)
        return timepoint_dir


class OMEWriter:
    """
    Unified writer for OME-TIFF and OME-Zarr formats.
    
    This class provides comprehensive support for writing microscopy data in
    multiple formats simultaneously. It handles:
    - OME-Zarr with NGFF 0.4 metadata and pyramid levels
    - Stitched OME-TIFF mosaics
    - Single TIFF files with appended tiles
    - Individual TIFF files with position-based naming
    - OMERO streaming upload for real-time server storage
    
    The writer supports multi-dimensional data (TCZYX) with proper physical
    coordinate transformations and channel metadata.
    
    Example:
        >>> from imswitch.imcontrol.model.io.ome_writers import (
        ...     OMEWriter, OMEWriterConfig, OMEFileStorePaths,
        ...     OMEROConnectionParams
        ... )
        >>> 
        >>> # Configure output formats
        >>> config = OMEWriterConfig(
        ...     write_zarr=True,
        ...     write_stitched_tiff=True,
        ...     write_omero=True,
        ...     pixel_size=0.325,
        ...     n_channels=2,
        ...     channel_names=["DAPI", "GFP"]
        ... )
        >>> 
        >>> # Setup file paths
        >>> file_paths = OMEFileStorePaths("/path/to/experiment")
        >>> 
        >>> # OMERO connection (optional)
        >>> omero_params = OMEROConnectionParams(
        ...     host="omero.server.com",
        ...     username="user",
        ...     password="pass"
        ... )
        >>> 
        >>> # Create writer
        >>> writer = OMEWriter(
        ...     file_paths=file_paths,
        ...     tile_shape=(512, 512),
        ...     grid_shape=(10, 10),
        ...     grid_geometry=(0, 0, 500, 500),  # x_start, y_start, x_step, y_step
        ...     config=config,
        ...     omero_connection_params=omero_params
        ... )
        >>> 
        >>> # Write frames
        >>> for i, frame in enumerate(frames):
        ...     writer.write_frame(frame, {"x": ..., "y": ..., ...})
        >>> 
        >>> # Finalize (builds pyramids, waits for OMERO upload)
        >>> writer.finalize()
    """

    def __init__(
        self, 
        file_paths: OMEFileStorePaths, 
        tile_shape: tuple, 
        grid_shape: tuple, 
        grid_geometry: tuple, 
        config: OMEWriterConfig, 
        logger=None, 
        isRGB: bool = False,
        omero_connection_params: Optional[OMEROConnectionParams] = None,
        shared_omero_key: Optional[str] = None,
    ):
        """
        Initialize the OME writer.
        
        Args:
            file_paths: OMEFileStorePaths object with output directories
            tile_shape: (height, width) of individual tiles in pixels
            grid_shape: (nx, ny) grid dimensions (number of tiles)
            grid_geometry: (x_start, y_start, x_step, y_step) for tile positioning
            config: OMEWriterConfig for writer behavior
            logger: Logger instance for debugging
            isRGB: Whether images are RGB format
            omero_connection_params: OMERO connection parameters (required if write_omero=True)
            shared_omero_key: Key for shared OMERO uploader (for timelapse experiments).
                             If provided, reuses an existing uploader or creates one to share.
        """
        self.file_paths = file_paths
        self.tile_h, self.tile_w = tile_shape
        self.nx, self.ny = grid_shape
        self.x_start, self.y_start, self.x_step, self.y_step = grid_geometry
        self.config = config
        self.logger = logger
        self.isRGB = isRGB
        self.omero_connection_params = omero_connection_params
        self.shared_omero_key = shared_omero_key

        # Zarr components
        self.store = None
        self.root = None
        self.canvas = None

        # TIFF writers
        self.tiff_stitcher: Optional[OmeTiffStitcher] = None
        self.single_tiff_writer: Optional[SingleTiffWriter] = None

        # OMERO uploader
        self.omero_uploader: Optional[OMEROUploader] = None
        self._owns_omero_uploader = False  # Track if we own the uploader

        # Timing for throttling
        self.t_last = time.time()

        # Initialize storage backends
        if config.write_zarr:
            self._setup_zarr_store()

        if config.write_stitched_tiff:
            self._setup_tiff_stitcher()

        if config.write_tiff_single:
            self._setup_single_tiff_writer()

        if config.write_omero:
            self._setup_omero_uploader()

    def _setup_zarr_store(self):
        """Set up the OME-Zarr store and canvas with proper OME-NGFF metadata."""
        # Use path string for Zarr v3 compatibility
        self.store = str(self.file_paths.zarr_dir)
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
            compressor=self.config.zarr_compressor
        )

        # Set OME-Zarr metadata
        self._set_ome_ngff_metadata()

    def _set_ome_ngff_metadata(self):
        """
        Set OME-NGFF compliant metadata for the Zarr store.
        
        This sets up:
        - multiscales: Image pyramid metadata with physical coordinates
        - omero: Channel visualization metadata (names, colors)
        """
        pixel_size_x = self.config.pixel_size
        pixel_size_y = self.config.pixel_size
        pixel_size_z = self.config.pixel_size_z
        time_interval = self.config.time_interval

        # Set multiscales metadata with physical coordinate transformations
        self.root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": "experiment",
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": [time_interval, 1, pixel_size_z, pixel_size_y, pixel_size_x]},
                        {"type": "translation", "translation": [0, 0, self.config.z_start, self.y_start, self.x_start]}
                    ]
                }
            ],
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "coordinateTransformations": [
                {"type": "scale", "scale": [time_interval, 1, pixel_size_z, pixel_size_y, pixel_size_x]}
            ]
        }]

        # Set omero metadata for channel visualization
        channels = []
        for i in range(self.config.n_channels):
            channel_name = self.config.channel_names[i] if i < len(self.config.channel_names) else f"Channel_{i}"
            channel_color = self.config.channel_colors[i] if i < len(self.config.channel_colors) else "FFFFFF"
            channels.append({
                "label": channel_name,
                "color": channel_color,
                "active": True,
                "coefficient": 1.0,
                "family": "linear",
                "inverted": False,
                "window": {
                    "start": 0,
                    "end": 65535,  # 16-bit max
                    "min": 0,
                    "max": 65535
                }
            })

        self.root.attrs["omero"] = {
            "id": 1,
            "name": os.path.basename(self.file_paths.zarr_dir),
            "version": "0.4",
            "channels": channels,
            "rdefs": {
                "defaultT": 0,
                "defaultZ": self.config.n_z_planes // 2,
                "model": "color"
            }
        }

    def _setup_tiff_stitcher(self):
        """Set up the TIFF stitcher for creating stitched OME-TIFF files."""
        stitched_tiff_path = os.path.join(self.file_paths.base_dir, "stitched.ome.tif")
        self.tiff_stitcher = OmeTiffStitcher(stitched_tiff_path, bigtiff=True, isRGB=self.isRGB)
        self.tiff_stitcher.start()
        if self.logger:
            self.logger.debug(f"TIFF stitcher initialized: {stitched_tiff_path}")

    def _setup_single_tiff_writer(self):
        """Set up the single TIFF writer for appending tiles with metadata."""
        single_tiff_path = os.path.join(self.file_paths.base_dir, "single_tiles.ome.tif")
        self.single_tiff_writer = SingleTiffWriter(single_tiff_path, bigtiff=True)
        if self.logger:
            self.logger.debug(f"Single TIFF writer initialized: {single_tiff_path}")

    def _setup_omero_uploader(self):
        """
        Set up the OMERO uploader for streaming tiles to OMERO server.
        
        If shared_omero_key is provided, attempts to reuse an existing uploader
        or creates a new one that can be shared with subsequent writers.
        """
        global _shared_omero_uploaders

        if not is_omero_available():
            if self.logger:
                self.logger.warning("OMERO upload requested but omero-py not available. Disabling.")
            self.config.write_omero = False
            return

        if self.omero_connection_params is None:
            if self.logger:
                self.logger.warning("OMERO upload requested but no connection params provided. Disabling.")
            self.config.write_omero = False
            return

        # Check for shared uploader
        if self.shared_omero_key and self.shared_omero_key in _shared_omero_uploaders:
            self.omero_uploader = _shared_omero_uploaders[self.shared_omero_key]
            self._owns_omero_uploader = False
            if self.logger:
                self.logger.debug(f"Reusing shared OMERO uploader: {self.shared_omero_key}")
            return

        try:
            self.omero_uploader = OMEROUploader(
                connection_params=self.omero_connection_params,
                image_name=os.path.basename(self.file_paths.base_dir),
                dtype=np.uint16,
                size_x=self.nx * self.tile_w,
                size_y=self.ny * self.tile_h,
                size_z=self.config.n_z_planes,
                size_c=self.config.n_channels,
                size_t=self.config.n_time_points,
                tile_width=self.tile_w,
                tile_height=self.tile_h,
                pixel_size_um=self.config.pixel_size,
                channel_names=self.config.channel_names,
                logger=self.logger,
                queue_maxsize=self.config.omero_queue_size,
            )
            self.omero_uploader.start()
            self._owns_omero_uploader = True

            # Register shared uploader if key provided
            if self.shared_omero_key:
                _shared_omero_uploaders[self.shared_omero_key] = self.omero_uploader
                if self.logger:
                    self.logger.debug(f"Registered shared OMERO uploader: {self.shared_omero_key}")

            if self.logger:
                self.logger.info(f"OMERO uploader initialized for {self.omero_connection_params.host}")

        except Exception as e:
            if self.logger:
                self.logger.error(f"Failed to initialize OMERO uploader: {e}")
            self.config.write_omero = False
            self.omero_uploader = None

    def write_frame(self, frame, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Write a single frame to enabled output formats.
        
        Args:
            frame: Image data as numpy array
            metadata: Dictionary containing position and other metadata.
                     Required keys: 'x', 'y'
                     Optional keys: 'time_index', 'channel_index', 'z_index',
                                   'runningNumber', 'illuminationChannel', 
                                   'illuminationValue', 'z'
            
        Returns:
            Dictionary with information about the written chunk (for Zarr)
        """
        result = {}

        # Legacy write_tiff is deprecated - use write_individual_tiffs instead
        # which includes proper OME-XML metadata
        # if self.config.write_tiff:
        #     self._write_tiff_tile(frame, metadata)

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

        # Write individual TIFF files with position-based naming if requested
        if self.config.write_individual_tiffs:
            self._write_individual_tiff(frame, metadata)

        # Write to OMERO if requested
        if self.config.write_omero and self.omero_uploader is not None:
            self._write_omero_tile(frame, metadata)

        # Throttle writes if needed
        self._throttle_writes()

        return result

    def _write_tiff_tile(self, frame, metadata: Dict[str, Any]):
        """Write individual TIFF tile."""
        t_idx = metadata.get("time_index", 0)
        z_idx = metadata.get("z_index", 0)
        c_idx = metadata.get("channel_index", 0)

        tiff_name = (
            f"F{metadata.get('runningNumber', 0):06d}_"
            f"t{t_idx:03d}_c{c_idx:03d}_z{z_idx:03d}_"
            f"x{metadata['x']:.1f}_y{metadata['y']:.1f}_"
            f"{metadata.get('illuminationChannel', 'unknown')}_"
            f"{metadata.get('illuminationValue', 0)}.ome.tif"
        )
        tiff_path = os.path.join(self.file_paths.tiff_dir, tiff_name)
        tif.imwrite(tiff_path, frame, compression=self.config.compression)

    def _write_zarr_tile(self, frame, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Write tile to Zarr canvas and return chunk information."""
        # Calculate grid position
        ix = int(round((metadata["x"] - self.x_start) / max(self.x_step, 1)))
        iy = int(round((metadata["y"] - self.y_start) / max(self.y_step, 1)))

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
        ix = int(round((metadata["x"] - self.x_start) / max(self.x_step, 1)))
        iy = int(round((metadata["y"] - self.y_start) / max(self.y_step, 1)))

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
        metadata_with_pixel_size = metadata.copy()
        metadata_with_pixel_size["pixel_size"] = self.config.pixel_size
        self.single_tiff_writer.add_image(image=frame, metadata=metadata_with_pixel_size)

    def _write_individual_tiff(self, frame, metadata: Dict[str, Any]):
        """
        Write individual TIFF file with position-based naming and OME-XML metadata.
        
        Files are organized in folders by timepoint, with filenames indicating:
        - Position in XYZ (in microns * 1000 for sub-micron precision)
        - Channel index and name
        - Iterator (running number)
        - Laser power
        
        Each file includes proper OME-XML metadata for compatibility with 
        ImageJ, OMERO, and other OME tools.
        """
        t_idx = metadata.get("time_index", 0)
        c_idx = metadata.get("channel_index", 0)

        # Get position in microns (multiply by 1000 for sub-micron precision)
        x_microns = int(metadata.get("x", 0) * 1000)
        y_microns = int(metadata.get("y", 0) * 1000)
        z_microns = int(metadata.get("z", 0) * 1000)

        channel = metadata.get("illuminationChannel", "unknown")
        laser_power = int(metadata.get("illuminationValue", 0))
        iterator = metadata.get("runningNumber", 0)

        timepoint_dir = self.file_paths.get_timepoint_dir(t_idx)
        current_time = time.strftime("%Y%m%d_%H%M%S")
        filename = f"t{current_time}_x{x_microns}_y{y_microns}_z{z_microns}_c{c_idx}_{channel}_i{iterator:04d}_p{laser_power}.tif"
        filepath = os.path.join(timepoint_dir, filename)

        # Build OME-XML metadata for this individual TIFF
        try:
            from .ome_tiff_metadata import build_ome_metadata_from_dict, OME_TYPES_AVAILABLE
            
            if OME_TYPES_AVAILABLE:
                # Prepare metadata dict with image dimensions and pixel info
                ome_metadata = metadata.copy()
                ome_metadata["height"] = frame.shape[0]
                ome_metadata["width"] = frame.shape[1] if frame.ndim > 1 else 1
                ome_metadata["dtype"] = str(frame.dtype)
                ome_metadata["pixel_size"] = self.config.pixel_size
                ome_metadata["channel_name"] = channel
                
                # Build OME-XML string
                ome_xml = build_ome_metadata_from_dict(ome_metadata)
                if ome_xml:
                    tif.imwrite(filepath, frame, compression=self.config.compression, description=ome_xml)
                else:
                    tif.imwrite(filepath, frame, compression=self.config.compression)
            else:
                tif.imwrite(filepath, frame, compression=self.config.compression)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to write OME metadata to individual TIFF: {e}")
            tif.imwrite(filepath, frame, compression=self.config.compression)

    def _write_omero_tile(self, frame, metadata: Dict[str, Any]):
        """
        Write tile to OMERO server via the uploader queue.
        
        Creates TileMetadata from the frame metadata and queues it for upload.
        """
        if self.omero_uploader is None:
            return

        # Calculate grid position
        ix = int(round((metadata["x"] - self.x_start) / max(self.x_step, 1)))
        iy = int(round((metadata["y"] - self.y_start) / max(self.y_step, 1)))

        # Get dimension indices
        t_idx = metadata.get("time_index", 0)
        c_idx = metadata.get("channel_index", 0)
        z_idx = metadata.get("z_index", 0)

        # Create tile metadata
        tile_meta = TileMetadata(
            ix=ix,
            iy=iy,
            z=z_idx,
            c=c_idx,
            t=t_idx,
            tile_data=frame,
            pixel_size_um=self.config.pixel_size,
            channel_name=metadata.get("illuminationChannel", f"Channel_{c_idx}"),
        )

        # Queue the tile for upload
        self.omero_uploader.queue_tile(tile_meta)

    def _throttle_writes(self):
        """Throttle disk writes if needed."""
        t_now = time.time()
        if t_now - self.t_last < self.config.min_period:
            time.sleep(self.config.min_period - (t_now - self.t_last))
        self.t_last = t_now

    def finalize(self):
        """
        Finalize the writing process.
        
        - Builds pyramid levels for OME-Zarr
        - Closes all TIFF writers
        - Waits for OMERO upload to complete (if owned)
        """
        if self.config.write_zarr and self.store is not None:
            try:
                self._build_vanilla_zarr_pyramids()
                if self.logger:
                    self.logger.info("Vanilla Zarr pyramid generated successfully")
            except Exception as err:
                if self.logger:
                    self.logger.warning(f"Pyramid generation failed: {err}")

        if self.config.write_stitched_tiff and self.tiff_stitcher is not None:
            self.tiff_stitcher.close()
            if self.logger:
                self.logger.info("Stitched TIFF file completed")

        if self.config.write_tiff_single and self.single_tiff_writer is not None:
            self.single_tiff_writer.close()
            if self.logger:
                self.logger.info("Single TIFF file completed")

        # Finalize OMERO uploader if we own it
        if self.config.write_omero and self.omero_uploader is not None and self._owns_omero_uploader:
            try:
                self.omero_uploader.stop_and_wait()
                if self.logger:
                    self.logger.info("OMERO upload completed")
            except Exception as e:
                if self.logger:
                    self.logger.error(f"OMERO finalization error: {e}")

        if self.logger:
            self.logger.info(f"OME writer finalized for {self.file_paths.base_dir}")

    def _build_vanilla_zarr_pyramids(self):
        """Build pyramid levels for OME-Zarr format."""
        if self.canvas is None:
            return

        def _generate_pyramids():
            try:
                self._generate_pyramids_sync()
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Pyramid generation failed: {e}")

        pyramid_thread = threading.Thread(target=_generate_pyramids, daemon=True)
        pyramid_thread.start()
        pyramid_thread.join()  # Wait for completion

    def _generate_pyramids_sync(self):
        """Synchronous pyramid generation with memory-efficient processing."""
        full_shape = self.canvas.shape
        n_t, n_c, n_z = full_shape[0], full_shape[1], full_shape[2]
        spatial_shape = full_shape[-2:]

        max_levels = 4

        for level in range(1, max_levels):
            new_y = spatial_shape[0] // (2 ** level)
            new_x = spatial_shape[1] // (2 ** level)

            if new_y < 64 or new_x < 64:
                break

            level_name = str(level)
            
            # Check if array already exists and delete it to avoid "array exists in store" error
            if level_name in self.root:
                try:
                    del self.root[level_name]
                    if self.logger:
                        self.logger.debug(f"Deleted existing pyramid level {level} for regeneration")
                except Exception as e:
                    if self.logger:
                        self.logger.warning(f"Could not delete existing pyramid level {level}: {e}")
                    continue

            level_canvas = self.root.create_array(
                name=level_name,
                shape=(n_t, n_c, n_z, int(new_y), int(new_x)),
                chunks=(1, 1, 1, int(min(self.tile_h, new_y)), int(min(self.tile_w, new_x))),
                dtype="uint16",
                compressor=self.config.zarr_compressor
            )

            self._downsample_all_dimensions(self.canvas, level_canvas, level, n_t, n_c, n_z)

            if self.logger:
                self.logger.debug(f"Created pyramid level {level} with shape ({n_t}, {n_c}, {n_z}, {new_y}, {new_x})")

        self._update_multiscales_metadata()

    def _downsample_all_dimensions(self, source_canvas, target_canvas, level, n_t, n_c, n_z):
        """Downsample data for all t, c, z dimensions."""
        downsample_factor = 2 ** level

        for t_idx in range(n_t):
            for c_idx in range(n_c):
                for z_idx in range(n_z):
                    try:
                        source_data = np.array(source_canvas[t_idx, c_idx, z_idx, :, :])
                        downsampled = source_data[::downsample_factor, ::downsample_factor]
                        target_canvas[t_idx, c_idx, z_idx, :, :] = downsampled
                    except Exception as e:
                        if self.logger:
                            self.logger.warning(f"Failed to downsample t={t_idx}, c={c_idx}, z={z_idx}: {e}")

    def _update_multiscales_metadata(self):
        """Update the multiscales metadata to include all pyramid levels."""
        pixel_size_x = self.config.pixel_size
        pixel_size_y = self.config.pixel_size
        pixel_size_z = self.config.pixel_size_z
        time_interval = self.config.time_interval

        datasets = []
        for level_name in sorted([k for k in self.root.keys() if k.isdigit()], key=int):
            level_int = int(level_name)
            scale_factor = 2 ** level_int
            datasets.append({
                "path": level_name,
                "coordinateTransformations": [
                    {"type": "scale", "scale": [
                        time_interval, 1, pixel_size_z,
                        pixel_size_y * scale_factor,
                        pixel_size_x * scale_factor
                    ]},
                    {"type": "translation", "translation": [
                        0, 0, self.config.z_start, self.y_start, self.x_start
                    ]}
                ]
            })

        self.root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": "experiment",
            "datasets": datasets,
            "axes": [
                {"name": "t", "type": "time", "unit": "second"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "coordinateTransformations": [
                {"type": "scale", "scale": [time_interval, 1, pixel_size_z, pixel_size_y, pixel_size_x]}
            ]
        }]

        # Preserve/set omero metadata
        if "omero" not in self.root.attrs:
            channels = []
            for i in range(self.config.n_channels):
                channel_name = self.config.channel_names[i] if i < len(self.config.channel_names) else f"Channel_{i}"
                channel_color = self.config.channel_colors[i] if i < len(self.config.channel_colors) else "FFFFFF"
                channels.append({
                    "label": channel_name,
                    "color": channel_color,
                    "active": True,
                    "coefficient": 1.0,
                    "family": "linear",
                    "inverted": False,
                    "window": {"start": 0, "end": 65535, "min": 0, "max": 65535}
                })

            self.root.attrs["omero"] = {
                "id": 1,
                "name": os.path.basename(self.file_paths.zarr_dir),
                "version": "0.4",
                "channels": channels,
                "rdefs": {"defaultT": 0, "defaultZ": self.config.n_z_planes // 2, "model": "color"}
            }

    def get_zarr_url(self) -> Optional[str]:
        """Get the Zarr store path for frontend streaming."""
        if self.config.write_zarr:
            return str(self.file_paths.zarr_dir)
        return None

    @classmethod
    def cleanup_shared_omero_uploaders(cls, key: Optional[str] = None, logger=None):
        """
        Clean up shared OMERO uploaders.
        
        Call this after all timepoints in a timelapse experiment are complete.
        
        Args:
            key: Specific key to clean up. If None, cleans up all shared uploaders.
            logger: Logger for status messages.
        """
        global _shared_omero_uploaders

        if key is not None:
            if key in _shared_omero_uploaders:
                try:
                    _shared_omero_uploaders[key].stop_and_wait()
                    if logger:
                        logger.info(f"Cleaned up shared OMERO uploader: {key}")
                except Exception as e:
                    if logger:
                        logger.error(f"Error cleaning up OMERO uploader {key}: {e}")
                finally:
                    del _shared_omero_uploaders[key]
        else:
            # Clean up all shared uploaders
            for k in list(_shared_omero_uploaders.keys()):
                try:
                    _shared_omero_uploaders[k].stop_and_wait()
                    if logger:
                        logger.info(f"Cleaned up shared OMERO uploader: {k}")
                except Exception as e:
                    if logger:
                        logger.error(f"Error cleaning up OMERO uploader {k}: {e}")
            _shared_omero_uploaders.clear()

    def get_omero_image_id(self) -> Optional[int]:
        """
        Get the OMERO image ID created by this writer.
        
        Returns:
            Image ID if available, None otherwise.
        """
        if self.omero_uploader is not None:
            return self.omero_uploader.get_image_id()
        return None

    def get_omero_dataset_id(self) -> Optional[int]:
        """
        Get the OMERO dataset ID used by this writer.
        
        Returns:
            Dataset ID if available, None otherwise.
        """
        if self.omero_uploader is not None:
            return self.omero_uploader.get_dataset_id()
        return None
