"""
Unified OME writer for both TIFF and OME-Zarr formats.

This module provides a comprehensive writer that handles both individual TIFF files
and OME-Zarr mosaics, supporting multi-dimensional data (time, channel, z-stack).
It includes proper OME-NGFF metadata for both formats, as well as optional
streaming to OMERO servers.

Migrated from: imswitch/imcontrol/controller/controllers/experiment_controller/ome_writer.py

Features:
    - OME-Zarr canvas of a scan region (NGFF 0.4, pyramid) through the shared
      ``NgffImage``; ``stitched.ome.tif`` is that canvas exported as OME-TIFF
    - Individual OME-TIFF tiles with position-based naming (``save_image``)
    - Multi-dimensional support (TCZYX)
    - Physical coordinate transformations
    - Channel metadata (names, colors)
    - OMERO streaming upload for real-time server storage
"""

import os
import shutil
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass

import numpy as np

from imswitch.imcontrol.model.aufgaben.storage import ImageSpec, NgffImage, save_image
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
        write_tiff, write_stitched_tiff, write_tiff_single: Any of them writes
                   the region canvas as ``stitched.ome.tif`` (TCZYX, pyramid,
                   OME-XML) at finalize(). They were three writers once; now
                   they are one export of the canvas.
        write_zarr: Keep the canvas as OME-Zarr (otherwise it is temporary)
        write_individual_tiffs: Write individual OME-TIFFs with position naming
                               and proper OME-XML metadata (RECOMMENDED)
        write_omero: Stream tiles to OMERO server
        omero_queue_size: Max tiles to queue for OMERO upload
        min_period: Minimum time between writes (throttling)
        pixel_size: Pixel size in microns (X/Y)
        pixel_size_z: Z-step size in microns
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
    min_period: float = 0.2   # inert: the writer no longer throttles itself
    pixel_size: float = 1.0
    pixel_size_z: float = 1.0
    n_time_points: int = 1
    n_z_planes: int = 1
    n_channels: int = 1
    channel_names: Optional[List[str]] = None
    channel_colors: Optional[List[str]] = None
    x_start: float = 0.0
    y_start: float = 0.0
    z_start: float = 0.0
    time_interval: float = 1.0

    @property
    def wants_tiff(self) -> bool:
        return self.write_tiff or self.write_stitched_tiff or self.write_tiff_single

    def __post_init__(self):
        """Initialize default channel metadata."""
        # Initialize default channel names if not provided
        if self.channel_names is None:
            self.channel_names = [f"Channel_{i}" for i in range(self.n_channels)]

        # Initialize default channel colors if not provided
        if self.channel_colors is None:
            # Default colors: green, red, blue, cyan, magenta, yellow
            default_colors = ["00FF00", "FF0000", "0000FF", "00FFFF", "FF00FF", "FFFF00"]
            self.channel_colors = [default_colors[i % len(default_colors)] for i in range(self.n_channels)]


def _sanitize_region_tag(region_id: Optional[str]) -> str:
    """Region id reduced to what a filename can carry, or "" when unset."""
    return "".join(c for c in (region_id or "") if c.isalnum())[:24]


def _sanitize_channel_name(channel) -> str:
    """Channel name as the tile filename pattern accepts it: ``[A-Za-z0-9_]``."""
    cleaned = "".join(c if (c.isascii() and c.isalnum()) else "_" for c in str(channel or ""))
    return cleaned.strip("_") or "unknown"


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
    - The region canvas as OME-Zarr (NGFF 0.4, pyramid) and/or OME-TIFF
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
        region_id: Optional[str] = None,
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

        # OME-Zarr canvas (NGFF 0.4 on zarr v2, shared with Aufgaben)
        self.store = None
        self.zarr_image: Optional[NgffImage] = None
        self.canvas = None

        self._finalized = False
        self._frames_written = 0
        # Goes into every individual-TIFF filename so downstream tooling can
        # tell one scan region's tiles from another's without relying on the
        # directory layout.
        self._region_tag = _sanitize_region_tag(region_id)

        # OMERO uploader
        self.omero_uploader: Optional[OMEROUploader] = None
        self._owns_omero_uploader = False  # Track if we own the uploader

        # Per-output write timings, summarised to the log every N frames
        # (see write_frame). Cheap, and the only way to see what a slow Pi is
        # actually spending its time on from a field log.
        self._timing_ms = {}
        self._timing_frames = 0

        # A stitched canvas (zarr, and the OME-TIFF exported from it) costs at
        # least one full pass over the whole scan at the end. On a Pi that is
        # minutes of I/O for a slide. Above the budget the run keeps only the
        # per-tile outputs; downstream tools (napari plugin, Ashlar) stitch
        # from those on a machine that can afford it.
        self._apply_large_scan_policy()

        # The canvas: kept as OME-Zarr, or temporary when only the TIFF is wanted
        if config.write_zarr or config.wants_tiff:
            self._setup_zarr_store()

        if config.write_omero:
            self._setup_omero_uploader()

    TIMING_LOG_EVERY = 100

    LARGE_SCAN_CANVAS_BYTES = 2 * 1024 ** 3      # 2 GiB, any stitched canvas

    def canvas_bytes(self) -> int:
        """Size of one full stitched canvas for this writer's geometry."""
        per_px = 3 if self.isRGB else 2
        return (int(self.config.n_time_points) * int(self.config.n_channels)
                * int(self.config.n_z_planes) * int(self.ny) * int(self.tile_h)
                * int(self.nx) * int(self.tile_w) * per_px)

    def _apply_large_scan_policy(self):
        size = self.canvas_bytes()
        self.large_scan = size > self.LARGE_SCAN_CANVAS_BYTES
        dropped = []
        if self.large_scan:
            for flag, label in (("write_zarr", "OME-Zarr canvas"),
                                ("write_tiff", "OME-TIFF canvas"),
                                ("write_stitched_tiff", "stitched TIFF"),
                                ("write_tiff_single", "single TIFF")):
                if getattr(self.config, flag):
                    setattr(self.config, flag, False)
                    dropped.append(label)
            if not self.config.write_individual_tiffs:
                self.config.write_individual_tiffs = True
                dropped.append("→ individual TIFFs enabled instead")
        if dropped and self.logger:
            self.logger.warning(
                f"Large scan: stitched canvas would be {size / 1024 ** 3:.1f} GiB "
                f"({self.nx}x{self.ny} tiles). Skipping: {', '.join(dropped)}. "
                "Stitch offline from the individual tiles."
            )

    def _image_spec(self) -> ImageSpec:
        """The canvas of this region (TCZYX) for both the OME-Zarr and the OME-TIFF.

        RGB tiles become three uint8 channels per illumination channel; the
        translation is the top-left pixel of the first tile.
        """
        cfg, samples = self.config, 3 if self.isRGB else 1
        names = list(cfg.channel_names or [])[:cfg.n_channels]
        names += [f"Channel_{i}" for i in range(len(names), cfg.n_channels)]
        if self.isRGB:
            names = [f"{n}_rgb_{c}" for n in names for c in "rgb"]
        px = cfg.pixel_size
        return ImageSpec(
            shape=(int(cfg.n_time_points), int(cfg.n_channels) * samples, int(cfg.n_z_planes),
                   int(self.ny * self.tile_h), int(self.nx * self.tile_w)),
            dtype="uint8" if self.isRGB else "uint16", pixel_size_um=px,
            z_step_um=cfg.pixel_size_z, t_step_s=cfg.time_interval,
            origin_um=(self.x_start - self.tile_w * px / 2, self.y_start - self.tile_h * px / 2,
                       cfg.z_start),
            channel_names=names, channel_colors=[] if self.isRGB else list(cfg.channel_colors or []),
            name=self._image_name or os.path.basename(self.file_paths.base_dir),
            chunks_yx=(int(self.tile_h), int(self.tile_w)))

    def _setup_zarr_store(self):
        """One NGFF image per writer: the stitched canvas of this region.

        Without ``write_zarr`` it lives in a hidden folder and is removed after
        the OME-TIFF export.
        """
        self.store = str(self.file_paths.zarr_dir) if self.config.write_zarr else \
            os.path.join(self.file_paths.base_dir, ".canvas.ome.zarr")
        self.zarr_image = NgffImage.create(self.store, self._image_spec())
        self.canvas = self.zarr_image.array
        if self.well_metadata:   # custom attrs; NGFF plates come from Aufgaben
            wm = self.well_metadata
            self.zarr_image.group.attrs["imswitch_well"] = {
                k: wm.get(k) for k in ("wellRow", "wellColumn", "labwareLoadName",
                                       "conditionLabel")}

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
        cfg = self.config
        outputs = (
            ("canvas", self.zarr_image is not None, self._write_zarr_tile),
            ("individual", cfg.write_individual_tiffs, self._write_individual_tiff),
            ("omero", cfg.write_omero and self.omero_uploader is not None, self._write_omero_tile),
        )
        for name, enabled, write in outputs:
            if not enabled:
                continue
            t0 = time.perf_counter()
            out = write(frame, metadata)
            self._timing_ms[name] = self._timing_ms.get(name, 0.0) + (time.perf_counter() - t0) * 1e3
            if name == "canvas" and out and cfg.write_zarr:   # chunk info for the live viewer
                result.update(out)
        self._frames_written += 1

        self._timing_frames += 1
        if self._timing_frames % self.TIMING_LOG_EVERY == 0 and self.logger:
            n = self.TIMING_LOG_EVERY
            parts = ", ".join(f"{k}={v / n:.0f}" for k, v in self._timing_ms.items())
            self.logger.info(f"Writer timing, ms/frame over last {n}: {parts}")
            self._timing_ms = {}
        return result

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

        # RGB tiles fill three channels; a stray 2-D frame on an RGB canvas is
        # copied to all three, a stray 3-D frame on a mono canvas becomes luminance.
        if self.isRGB:
            if frame.ndim == 2:
                frame = np.stack([frame] * 3, axis=-1)
            for k in range(3):
                self.zarr_image.write(frame[..., k], t_idx, c_idx * 3 + k, z_idx, y0, x0)
        else:
            if frame.ndim == 3:
                frame = np.dot(frame[..., :3], [0.299, 0.587, 0.114]).astype(self.canvas.dtype)
            self.zarr_image.write(frame, t_idx, c_idx, z_idx, y0, x0)

        # The chunk key of this tile's first plane (zarr v2, "/" separator)
        ch, cw = self.zarr_image.chunks_yx
        c_first = c_idx * 3 if self.isRGB else c_idx
        return {
            "rel_chunk": f"0/{t_idx}/{c_first}/{z_idx}/{y0 // ch}/{x0 // cw}",
            "grid_pos": (ix, iy),
            "canvas_bounds": (x0, x1, y0, y1),
            "t_idx": t_idx,
            "c_idx": c_idx,
            "z_idx": z_idx
        }

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

        # The name goes into a filename that downstream stitchers (the napari
        # openUC2 processor, ashlar) parse with ``_c<idx>_<[A-Za-z0-9_]+>_i``:
        # a space or hyphen ("LED Matrix", "Laser-488") would make every tile
        # of that channel unparseable, so reduce it to what the pattern takes.
        channel = _sanitize_channel_name(metadata.get("illuminationChannel", "unknown"))
        # -1 marks "no light"; "_p-1" does not match the stitchers' "_p(\d+)",
        # so such tiles were skipped. Write 0 instead.
        laser_power = max(0, int(metadata.get("illuminationValue", 0)))
        # ``_i`` is the tile's index within its scan region — the same
        # ``iterator`` the protocol JSON lists — so stitchers can look tiles
        # up in it. Older callers only sent the workflow step id.
        iterator = metadata.get("iterator", metadata.get("runningNumber", 0))

        timepoint_dir = self.file_paths.get_timepoint_dir(t_idx)
        current_time = time.strftime("%Y%m%d_%H%M%S")
        region = f"_r{self._region_tag}" if self._region_tag else ""
        filename = (
            f"t{current_time}{region}_x{x_microns}_y{y_microns}_z{z_microns}"
            f"_c{c_idx}_{channel}_i{iterator:04d}_p{laser_power}.tif"
        )
        filepath = os.path.join(timepoint_dir, filename)

        # RGB tiles (H, W, 3) are interleaved RGB; mono tiles one plane.
        rgb = self.isRGB and frame.ndim == 3
        h, w = frame.shape[:2]
        px = self.config.pixel_size
        x, y, z = (float(metadata.get(k) or 0.0) for k in ("x", "y", "z"))
        exposure = metadata.get("exposure_time_ms", metadata.get("exposure"))
        save_image(filepath, frame, "YXS" if rgb else "YX", pixel_size_um=px,
                   origin_um=(x - w * px / 2, y - h * px / 2, z), channel_names=[channel],
                   exposure_ms=[exposure] if exposure is not None else [],
                   name=os.path.splitext(filename)[0])

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

    def finalize(self):
        """
        Finalize the writing process.
        
        - Builds pyramid levels for OME-Zarr
        - Closes all TIFF writers
        - Waits for OMERO upload to complete (if owned)

        Idempotent: the workflow finalizes each tile writer and then makes a
        final cleanup pass over all of them, and an aborted run finalizes
        whatever the workflow never reached. Rebuilding pyramids on the second
        call is pure waste.
        """
        if self._finalized:
            return
        self._finalized = True

        if self.zarr_image is not None:
            self._finalize_canvas()

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

    def _finalize_canvas(self):
        """Pyramid for the canvas, its OME-TIFF export, and the temporary store removed."""
        try:
            self.zarr_image.finalize(min_size=64, max_levels=4)
            if self.logger:
                self.logger.info(f"Canvas pyramid: {self.zarr_image.levels} level(s)")
        except Exception as err:
            if self.logger:
                self.logger.error(f"Pyramid generation failed (full resolution is intact): {err}")
        if self.config.wants_tiff and self._frames_written:
            path = os.path.join(self.file_paths.base_dir, "stitched.ome.tif")
            try:
                self.zarr_image.export_ome_tiff(path)
                if self.logger:
                    self.logger.info(f"OME-TIFF written ({self._frames_written} frames): {path}")
            except Exception as err:
                if self.logger:
                    self.logger.error(f"OME-TIFF export failed: {err}")
        if not self.config.write_zarr:
            shutil.rmtree(self.store, ignore_errors=True)

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
