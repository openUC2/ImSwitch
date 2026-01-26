"""
Minimal OME-Zarr data source for streaming acquisition data.

This module provides a simple writer for streaming 2D frames into an OME-Zarr
store with multi-resolution support.

Migrated from: imswitch/imcontrol/controller/controllers/experiment_controller/zarr_data_source.py
"""

import numpy as np
import zarr
import zarr.storage
from typing import Optional, Dict, Any, List

from .metadata import MinimalMetadata


GROUP_PREFIX = "Pos_"


class MinimalZarrDataSource:
    """
    Writes frames to an OME-Zarr store with multi-resolution support.
    
    This class provides a simple interface for streaming 2D frames into
    a proper OME-Zarr 0.4 compliant store with automatic pyramid generation.
    
    Features:
        - Multi-position support (each position as separate group)
        - Multi-resolution pyramid generation
        - Automatic indexing based on per_stack or per_channel scanning order
        - Compatible with napari-ome-zarr viewer
    
    Example:
        >>> source = MinimalZarrDataSource("/path/to/data.zarr")
        >>> source.set_metadata_from_configuration_experiment(config)
        >>> source.write(frame_data)  # Automatically indexes into t,c,z,p
        >>> source.close()
    """

    def __init__(self, file_name: str, mode: str = "w"):
        """
        Initialize the Zarr data source.
        
        Args:
            file_name: Path to the .zarr directory
            mode: File mode ('w' for write, 'a' for append)
        """
        self.file_name = file_name
        self.mode = mode
        self._store = None
        self.image = None

        # Shape placeholders
        self.shape_t = 1
        self.shape_c = 1
        self.shape_z = 1
        self.shape_y = 64
        self.shape_x = 64
        self.per_stack = True
        self.dtype = np.uint16

        # Multi-resolution arrays
        self.shapes: List[tuple] = []
        self.resolutions: List[tuple] = []

        self.metadata: Optional[MinimalMetadata] = None

        # Bookkeeping for scanning
        self._current_position = -1
        self._current_frame = 0

    def set_metadata_from_configuration_experiment(self, config: Dict[str, Any]):
        """
        Configure shapes and metadata from experiment configuration.
        
        Args:
            config: Dictionary containing experiment configuration with
                   'experiment.MicroscopeState' and 'experiment.CameraParameters'
        """
        # Get MicroscopeState
        ms = config.get("experiment", {}).get("MicroscopeState", {})

        # Z-steps default to 1 if no Z-stack configured
        self.shape_z = 1
        self.shape_t = ms.get("timepoints", 1)

        # Count selected channels
        channels_dict = ms.get("channels", {})
        channels = [ck for ck, cinfo in channels_dict.items() 
                   if cinfo.get("is_selected", False)]
        self.shape_c = max(len(channels), 1)

        # Get camera parameters
        cam_key = ms.get("microscope_name", "default_cam")
        cp = config.get("experiment", {}).get("CameraParameters", {}).get(cam_key, {})
        self.shape_x = cp.get("x_pixels", 64)
        self.shape_y = cp.get("y_pixels", 64)

        # Scanning order
        self.per_stack = ms.get("stack_cycling_mode", "per_stack") == "per_stack"

        # 2-level pyramid
        shape0 = (self.shape_z, self.shape_y, self.shape_x)
        shape1 = (self.shape_z, max(1, self.shape_y // 2), max(1, self.shape_x // 2))
        self.shapes = [shape0, shape1]
        self.resolutions = [(1, 1, 1), (1, 2, 2)]

        self.metadata = MinimalMetadata(per_stack=self.per_stack)

    def configure(
        self,
        shape_t: int = 1,
        shape_c: int = 1,
        shape_z: int = 1,
        shape_y: int = 512,
        shape_x: int = 512,
        per_stack: bool = True,
        dtype=np.uint16,
        num_pyramid_levels: int = 2
    ):
        """
        Directly configure the data source dimensions.
        
        Args:
            shape_t: Number of time points
            shape_c: Number of channels
            shape_z: Number of Z planes
            shape_y: Image height in pixels
            shape_x: Image width in pixels
            per_stack: If True, Z changes faster than C
            dtype: Data type for arrays
            num_pyramid_levels: Number of pyramid resolution levels
        """
        self.shape_t = shape_t
        self.shape_c = shape_c
        self.shape_z = shape_z
        self.shape_y = shape_y
        self.shape_x = shape_x
        self.per_stack = per_stack
        self.dtype = dtype

        # Build pyramid shapes
        self.shapes = []
        self.resolutions = []
        for level in range(num_pyramid_levels):
            factor = 2 ** level
            shape = (shape_z, max(1, shape_y // factor), max(1, shape_x // factor))
            self.shapes.append(shape)
            self.resolutions.append((1, factor, factor))

        self.metadata = MinimalMetadata(per_stack=self.per_stack)

    def _setup_store(self):
        """Create the Zarr store and top-level group."""
        # Zarr 3.0 compatibility
        if hasattr(zarr.storage, 'FSStore'):
            try:
                store = zarr.storage.FSStore(
                    self.file_name, mode=self.mode, dimension_separator="/"
                )
            except TypeError:
                store = zarr.storage.FSStore(self.file_name, mode=self.mode)
        else:
            store = self.file_name

        self.image = zarr.group(store=store, overwrite=True)
        self._store = store
        self.image.attrs["description"] = "OME-Zarr from MinimalZarrDataSource"

    def new_position(self, pos_index: int, **kwargs):
        """
        Create arrays for a new position with multi-resolution levels.
        
        Args:
            pos_index: Position index
            **kwargs: Additional metadata (x, y, z, theta, f, view)
        """
        name = f"{GROUP_PREFIX}{pos_index}"
        paths = []
        
        for i, zyx in enumerate(self.shapes):
            shape_5d = (self.shape_t, self.shape_c, *zyx)
            arr_name = f"{name}_{i}"

            arr = self.image.create(
                name=arr_name,
                shape=shape_5d,
                chunks=(1, 1) + zyx,
                dtype=self.dtype,
            )
            paths.append(arr.path)
            arr.attrs["_ARRAY_DIMENSIONS"] = ["t", "c", "z", "y", "x"]

        # Update multiscales metadata
        ms_list = self.image.attrs.get("multiscales", [])
        ms_entry = self.metadata.multiscales_dict(
            name, paths, self.resolutions, view=kwargs.get("view", "")
        )
        ms_list.append(ms_entry)
        self.image.attrs["multiscales"] = ms_list

    def _cztp_indices(self, frame_id: int):
        """
        Map overall frame index to (c, z, t, position).
        
        Args:
            frame_id: Overall frame index
            
        Returns:
            Tuple of (channel, z_slice, time, position) indices
        """
        if self.per_stack:
            c = (frame_id // self.shape_z) % self.shape_c
            z = frame_id % self.shape_z
        else:
            c = frame_id % self.shape_c
            z = (frame_id // self.shape_c) % self.shape_z
        t = (frame_id // (self.shape_c * self.shape_z)) % self.shape_t
        p = frame_id // (self.shape_c * self.shape_z * self.shape_t)
        return c, z, t, p

    def write(
        self, 
        data: np.ndarray, 
        x: Optional[float] = None, 
        y: Optional[float] = None, 
        z: Optional[float] = None, 
        theta: Optional[float] = None, 
        f: Optional[float] = None, 
        ti: Optional[int] = None, 
        ci: Optional[int] = None
    ):
        """
        Write a 2D frame to the next (t, c, z, p) position.
        
        Args:
            data: 2D image data
            x, y, z: Stage position (for metadata)
            theta, f: Additional position info
            ti: Override time index
            ci: Override channel index
        """
        if self._store is None:
            self._setup_store()

        c, zslice, t, p = self._cztp_indices(self._current_frame)
        
        # Allow overriding indices
        if ti is not None:
            t = ti
        if ci is not None:
            c = ci
            
        # Create new position if needed
        if p != self._current_position:
            self._current_position = p
            self.new_position(p, x=x, y=y, z=z, theta=theta, f=f)

        # Store in each resolution level
        for i, (dz, dy, dx) in enumerate(self.resolutions):
            arr_name = f"{GROUP_PREFIX}{p}_{i}"
            zs = zslice // dz
            ds_data = data[::dy, ::dx]
            self.image[arr_name][t, c, zs, :, :] = ds_data.astype(self.dtype)

        self._current_frame += 1

    def write_frame(
        self,
        data: np.ndarray,
        t: int = 0,
        c: int = 0,
        z: int = 0,
        position: int = 0
    ):
        """
        Write a frame with explicit indices (alternative to automatic indexing).
        
        Args:
            data: 2D image data
            t: Time index
            c: Channel index
            z: Z-slice index
            position: Position index
        """
        if self._store is None:
            self._setup_store()

        # Create position if needed
        if position != self._current_position:
            self._current_position = position
            self.new_position(position)

        # Store in each resolution level
        for i, (dz, dy, dx) in enumerate(self.resolutions):
            arr_name = f"{GROUP_PREFIX}{position}_{i}"
            zs = z // dz
            ds_data = data[::dy, ::dx]
            self.image[arr_name][t, c, zs, :, :] = ds_data.astype(self.dtype)

    def close(self):
        """Close the Zarr store."""
        if self._store is not None:
            if hasattr(self._store, 'close'):
                self._store.close()
            self._store = None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
        return False
