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
        write_tiff: Write a single multi-dimensional OME-TIFF hyperstack
                   (TCZYX / TCZYXS for RGB) with full OME-XML metadata at
                   finalize(). Fiji-readable ("Bio-Formats") with correct
                   dimensions, physical pixel sizes and channel names.
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
    write_tiff: bool = False  # Single multi-dimensional OME-TIFF hyperstack
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

        # NOTE: directories are created lazily (only when a TIFF is actually
        # written – see ``ensure_tiff_dir`` / ``get_timepoint_dir``).  Creating
        # ``tiff_dir`` eagerly here produced an empty ``<area>/tiles`` folder for
        # every scan-area writer even when individual-TIFF output is disabled
        # (the default), littering each multi-area / multi-position acquisition
        # with dozens of empty directories.

    def ensure_tiff_dir(self) -> str:
        """Create and return the TIFF output directory on first use."""
        os.makedirs(self.tiff_dir, exist_ok=True)
        return self.tiff_dir

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
        well_metadata: Optional[Dict[str, Any]] = None,
        image_name: Optional[str] = None,
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
            well_metadata: Optional dict describing the labware well backing this image.
                Recognised keys: ``wellRow`` (str, e.g. "A"), ``wellColumn`` (int, 1-based),
                ``labwareLoadName`` (str), ``conditionLabel`` (str). When provided, the OME-NGFF
                ``well`` group attrs and a custom ``imswitch_well`` block are emitted.
            image_name: Optional clean image name (e.g. the position/area name) used
                for OME/OMERO image metadata. Falls back to the output file basename.
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
        self.well_metadata = well_metadata or None
        # Optional clean image name for OME/OMERO metadata. Falls back to the
        # file basename (legacy behaviour) when not provided.
        self._image_name = image_name or None

        # Zarr components
        self.store = None
        self.root = None
        self.canvas = None

        # TIFF writers
        self.tiff_stitcher: Optional[OmeTiffStitcher] = None
        self.single_tiff_writer: Optional[SingleTiffWriter] = None

        # In-memory mosaic for the unified multi-dimensional OME-TIFF hyperstack
        # (write_tiff). Filled tile-by-tile in write_frame() and flushed to a
        # single Fiji-readable OME-TIFF at finalize(). Allocated lazily so we
        # never reserve RAM when the option is disabled.
        self.tiff_mosaic: Optional[np.ndarray] = None
        self._tiff_frames_written = 0

        # OMERO uploader
        self.omero_uploader: Optional[OMEROUploader] = None
        self._owns_omero_uploader = False  # Track if we own the uploader

        # Timing for throttling
        self.t_last = time.time()

        # Initialize storage backends
        if config.write_zarr:
            self._setup_zarr_store()

        if config.write_tiff:
            self._setup_tiff_mosaic()

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
        
        # Colour (RGB) detectors deliver (H, W, 3) tiles. Add a trailing
        # "samples" axis so the canvas can physically store them, and keep the
        # native 8-bit range (Picamera2/Hik RGB are uint8) so viewers render
        # true colour instead of a near-black 16-bit image. Mono detectors keep
        # the original 5-D uint16 layout unchanged.
        base_shape = (
            int(self.config.n_time_points),
            int(self.config.n_channels),
            int(self.config.n_z_planes),
            int(self.ny * self.tile_h),
            int(self.nx * self.tile_w),
        )
        base_chunks = (1, 1, 1, int(self.tile_h), int(self.tile_w))
        if self.isRGB:
            canvas_shape = base_shape + (3,)
            canvas_chunks = base_chunks + (3,)
            canvas_dtype = "uint8"
        else:
            canvas_shape = base_shape
            canvas_chunks = base_chunks
            canvas_dtype = "uint16"

        self.canvas = self.root.create_array(
            name="0",
            shape=canvas_shape,   # t c z y x (s)
            chunks=canvas_chunks,
            dtype=canvas_dtype,
            compressor=self.config.zarr_compressor
        )

        # Set OME-Zarr metadata
        self._set_ome_ngff_metadata()

    def _setup_tiff_mosaic(self):
        """Allocate the in-memory mosaic backing the multi-dimensional OME-TIFF.

        The mosaic mirrors the OME-Zarr canvas layout (TCZYX, plus a trailing
        samples axis for RGB) so a full scan lands in a single Fiji-readable
        hyperstack. Uint16 for mono detectors, uint8 for RGB (matching the
        native camera range so colour renders correctly).
        """
        base_shape = (
            int(self.config.n_time_points),
            int(self.config.n_channels),
            int(self.config.n_z_planes),
            int(self.ny * self.tile_h),
            int(self.nx * self.tile_w),
        )
        try:
            if self.isRGB:
                self.tiff_mosaic = np.zeros(base_shape + (3,), dtype=np.uint8)
            else:
                self.tiff_mosaic = np.zeros(base_shape, dtype=np.uint16)
            if self.logger:
                self.logger.debug(
                    f"OME-TIFF mosaic allocated: shape={self.tiff_mosaic.shape}, "
                    f"dtype={self.tiff_mosaic.dtype}"
                )
        except MemoryError:
            # Refuse gracefully rather than crashing the acquisition thread.
            self.tiff_mosaic = None
            self.config.write_tiff = False
            if self.logger:
                self.logger.error(
                    "Not enough memory to build the multi-dimensional OME-TIFF "
                    f"mosaic (shape {base_shape}); disabling OME-TIFF output. "
                    "Use OME-Zarr or Stitched OME-TIFF for very large scans."
                )

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

        # 8-bit display range for RGB (uint8) canvases, 16-bit for mono.
        display_max = 255 if self.isRGB else 65535

        # Axes / scales. RGB canvases carry a trailing "samples" axis; mono
        # canvases keep the classic 5-D TCZYX layout.
        axes = [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        scale = [time_interval, 1, pixel_size_z, pixel_size_y, pixel_size_x]
        translation = [0, 0, self.config.z_start, self.y_start, self.x_start]
        if self.isRGB:
            axes.append({"name": "s", "type": "channel"})
            scale = scale + [1]
            translation = translation + [0]

        # Set multiscales metadata with physical coordinate transformations
        self.root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": self._image_name or "experiment",
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [
                        {"type": "scale", "scale": list(scale)},
                        {"type": "translation", "translation": list(translation)}
                    ]
                }
            ],
            "axes": axes,
            "coordinateTransformations": [
                {"type": "scale", "scale": list(scale)}
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
                    "end": display_max,
                    "min": 0,
                    "max": display_max
                }
            })

        self.root.attrs["omero"] = {
            "id": 1,
            "name": self._image_name or os.path.basename(self.file_paths.zarr_dir),
            "version": "0.4",
            "channels": channels,
            "rdefs": {
                "defaultT": 0,
                "defaultZ": self.config.n_z_planes // 2,
                "model": "color"
            }
        }

        # OME-NGFF "well" group attrs + ImSwitch labware metadata.
        # Emitted only when the caller passed structured well metadata, so the
        # plain (non-plate) acquisition path stays byte-identical.
        if self.well_metadata:
            wm = self.well_metadata
            self.root.attrs["well"] = {
                "version": "0.4",
                "images": [{"path": "0", "acquisition": 0}],
            }
            self.root.attrs["imswitch_well"] = {
                "wellRow": wm.get("wellRow"),
                "wellColumn": wm.get("wellColumn"),
                "labwareLoadName": wm.get("labwareLoadName"),
                "conditionLabel": wm.get("conditionLabel"),
            }

    def _setup_tiff_stitcher(self):
        """Set up the TIFF stitcher for creating stitched OME-TIFF files."""
        # base_dir is created lazily (see OMEFileStorePaths); ensure it exists
        # before the stitcher opens its output file inside it.
        os.makedirs(self.file_paths.base_dir, exist_ok=True)
        stitched_tiff_path = os.path.join(self.file_paths.base_dir, "stitched.ome.tif")
        self.tiff_stitcher = OmeTiffStitcher(stitched_tiff_path, bigtiff=True, isRGB=self.isRGB, nx=self.nx, ny=self.ny, tile_w=self.tile_w, tile_h=self.tile_h)
        self.tiff_stitcher.start()
        if self.logger:
            self.logger.debug(f"TIFF stitcher initialized: {stitched_tiff_path}")

    def _setup_single_tiff_writer(self):
        """Set up the single TIFF writer for appending tiles with metadata."""
        # base_dir is created lazily (see OMEFileStorePaths); ensure it exists
        # before the single-TIFF writer opens its output file inside it.
        os.makedirs(self.file_paths.base_dir, exist_ok=True)
        single_tiff_path = os.path.join(self.file_paths.base_dir, "single_tiles.ome.tif")
        self.single_tiff_writer = SingleTiffWriter(single_tiff_path, bigtiff=True, isRGB=self.isRGB)
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
                image_name=self._image_name or os.path.basename(self.file_paths.base_dir),
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

        # Accumulate into the in-memory multi-dimensional OME-TIFF mosaic.
        # The full hyperstack is flushed to disk once in finalize().
        if self.config.write_tiff and self.tiff_mosaic is not None:
            self._write_tiff_mosaic_tile(frame, metadata)

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

    def _write_tiff_mosaic_tile(self, frame, metadata: Dict[str, Any]):
        """Place a single tile into the in-memory multi-dimensional OME-TIFF mosaic.

        Uses the same grid/index maths as the Zarr canvas so the two outputs
        stay pixel-aligned. Handles RGB/mono coercion identically to
        ``_write_zarr_tile``.
        """
        # Calculate grid position
        ix = int(round((metadata["x"] - self.x_start) / max(self.x_step, 1)))
        iy = int(round((metadata["y"] - self.y_start) / max(self.y_step, 1)))

        # Dimension indices, clamped to the allocated extents
        t_idx = min(metadata.get("time_index", 0), self.config.n_time_points - 1)
        c_idx = min(metadata.get("channel_index", 0), self.config.n_channels - 1)
        z_idx = min(metadata.get("z_index", 0), self.config.n_z_planes - 1)

        # Guard against out-of-range tiles (defensive; grid should already fit)
        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            if self.logger:
                self.logger.warning(
                    f"OME-TIFF tile ({ix},{iy}) outside grid ({self.nx}x{self.ny}); skipped"
                )
            return

        y0, y1 = iy * self.tile_h, (iy + 1) * self.tile_h
        x0, x1 = ix * self.tile_w, (ix + 1) * self.tile_w

        if self.isRGB:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            self.tiff_mosaic[t_idx, c_idx, z_idx, y0:y1, x0:x1, :] = frame
        else:
            if frame.ndim == 3:
                frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(self.tiff_mosaic.dtype)
            self.tiff_mosaic[t_idx, c_idx, z_idx, y0:y1, x0:x1] = frame

        self._tiff_frames_written += 1

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

        # Write to canvas with proper indexing. RGB canvases carry a trailing
        # samples axis; a stray 2-D frame is broadcast across the 3 samples,
        # and a stray 3-D frame on a mono canvas is collapsed to luminance.
        if self.isRGB:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            self.canvas[t_idx, c_idx, z_idx, y0:y1, x0:x1, :] = frame
        else:
            if frame.ndim == 3:
                frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(self.canvas.dtype)
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

        # RGB tiles (H, W, 3) must be tagged photometric="rgb" or tifffile would
        # store them as a 3-plane grayscale stack, which is why colour tiles were
        # being saved as grayscale. Grayscale tiles keep photometric=None (auto).
        photometric = "rgb" if (self.isRGB and frame.ndim == 3) else None

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
                    tif.imwrite(filepath, frame, compression=self.config.compression, description=ome_xml, photometric=photometric)
                else:
                    tif.imwrite(filepath, frame, compression=self.config.compression, photometric=photometric)
            else:
                tif.imwrite(filepath, frame, compression=self.config.compression, photometric=photometric)
        except Exception as e:
            if self.logger:
                self.logger.warning(f"Failed to write OME metadata to individual TIFF: {e}")
            tif.imwrite(filepath, frame, compression=self.config.compression, photometric=photometric)

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

        # The OMERO uploader is built for the mono plane model (uint16, no
        # samples axis) and is not RGB-aware. For colour experiments, downgrade
        # the OMERO tile to grayscale so the upload stays functional instead of
        # crashing/corrupting mid-run. The OME-Zarr and TIFF outputs still keep
        # full colour. (Mono frames are 2-D and skip this branch unchanged.)
        if self.isRGB and getattr(frame, "ndim", 0) == 3:
            if not getattr(self, "_warned_omero_rgb", False):
                if self.logger:
                    self.logger.warning(
                        "OMERO upload is not RGB-capable; uploading grayscale "
                        "tiles to OMERO (OME-Zarr/TIFF outputs keep colour)."
                    )
                self._warned_omero_rgb = True
            frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(np.uint16)

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

        if self.config.write_tiff and self.tiff_mosaic is not None:
            try:
                self._finalize_ome_tiff()
            except Exception as err:
                if self.logger:
                    self.logger.error(f"Failed to write multi-dimensional OME-TIFF: {err}")

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

    def _finalize_ome_tiff(self):
        """Flush the accumulated mosaic to a single multi-dimensional OME-TIFF.

        Produces a Fiji/Bio-Formats-readable hyperstack with the proper OME
        metadata model: TCZYX axes (plus a trailing samples axis for RGB),
        physical pixel sizes, Z-step, time increment and channel names. Written
        via tifffile's native OME support (``ome=True``).
        """
        if self.tiff_mosaic is None:
            return

        if self._tiff_frames_written == 0:
            if self.logger:
                self.logger.warning(
                    "OME-TIFF requested but no frames were written; skipping empty file"
                )
            return

        # Write the single hyperstack next to the OME-Zarr store so the two
        # outputs live side by side under the same experiment folder.
        os.makedirs(self.file_paths.base_dir, exist_ok=True)
        out_path = os.path.join(
            self.file_paths.base_dir,
            os.path.basename(self.file_paths.base_dir) + ".ome.tif",
        )

        # RGB hyperstacks carry a trailing samples axis and must be tagged
        # photometric="rgb"; mono stacks stay minisblack.
        axes = "TCZYXS" if self.isRGB else "TCZYX"
        photometric = "rgb" if self.isRGB else "minisblack"

        # OME metadata model. tifffile maps these keys onto the OME-XML schema.
        channel_names = list(self.config.channel_names or [])
        ome_metadata = {
            "axes": axes,
            "PhysicalSizeX": float(self.config.pixel_size),
            "PhysicalSizeXUnit": "µm",
            "PhysicalSizeY": float(self.config.pixel_size),
            "PhysicalSizeYUnit": "µm",
            "PhysicalSizeZ": float(self.config.pixel_size_z),
            "PhysicalSizeZUnit": "µm",
            "TimeIncrement": float(self.config.time_interval),
            "TimeIncrementUnit": "s",
        }
        if self._image_name:
            ome_metadata["Name"] = self._image_name
        if channel_names:
            ome_metadata["Channel"] = {"Name": channel_names}

        tif.imwrite(
            out_path,
            self.tiff_mosaic,
            photometric=photometric,
            metadata=ome_metadata,
            compression=self.config.compression,
            ome=True,
            bigtiff=True,
        )

        # Release the buffer promptly — a full mosaic can be large.
        self.tiff_mosaic = None

        if self.logger:
            self.logger.info(
                f"Multi-dimensional OME-TIFF written ({self._tiff_frames_written} frames): {out_path}"
            )

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
        # RGB canvases are 6-D (…, Y, X, 3); mono are 5-D (…, Y, X).
        spatial_shape = full_shape[3:5] if self.isRGB else full_shape[-2:]

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

            if self.isRGB:
                level_shape = (n_t, n_c, n_z, int(new_y), int(new_x), 3)
                level_chunks = (1, 1, 1, int(min(self.tile_h, new_y)), int(min(self.tile_w, new_x)), 3)
                level_dtype = "uint8"
            else:
                level_shape = (n_t, n_c, n_z, int(new_y), int(new_x))
                level_chunks = (1, 1, 1, int(min(self.tile_h, new_y)), int(min(self.tile_w, new_x)))
                level_dtype = "uint16"
            level_canvas = self.root.create_array(
                name=level_name,
                shape=level_shape,
                chunks=level_chunks,
                dtype=level_dtype,
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
                        if self.isRGB:
                            source_data = np.array(source_canvas[t_idx, c_idx, z_idx, :, :, :])
                            downsampled = source_data[::downsample_factor, ::downsample_factor, :]
                            target_canvas[t_idx, c_idx, z_idx, :, :, :] = downsampled
                        else:
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

        rgb = self.isRGB
        display_max = 255 if rgb else 65535

        datasets = []
        for level_name in sorted([k for k in self.root.keys() if k.isdigit()], key=int):
            level_int = int(level_name)
            scale_factor = 2 ** level_int
            ds_scale = [
                time_interval, 1, pixel_size_z,
                pixel_size_y * scale_factor,
                pixel_size_x * scale_factor,
            ]
            ds_translation = [0, 0, self.config.z_start, self.y_start, self.x_start]
            if rgb:
                ds_scale = ds_scale + [1]
                ds_translation = ds_translation + [0]
            datasets.append({
                "path": level_name,
                "coordinateTransformations": [
                    {"type": "scale", "scale": ds_scale},
                    {"type": "translation", "translation": ds_translation}
                ]
            })

        axes = [
            {"name": "t", "type": "time", "unit": "second"},
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space", "unit": "micrometer"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
        ]
        top_scale = [time_interval, 1, pixel_size_z, pixel_size_y, pixel_size_x]
        if rgb:
            axes.append({"name": "s", "type": "channel"})
            top_scale = top_scale + [1]

        self.root.attrs["multiscales"] = [{
            "version": "0.4",
            "name": self._image_name or "experiment",
            "datasets": datasets,
            "axes": axes,
            "coordinateTransformations": [
                {"type": "scale", "scale": top_scale}
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
                    "window": {"start": 0, "end": display_max, "min": 0, "max": display_max}
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
