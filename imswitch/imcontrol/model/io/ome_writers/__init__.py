"""
Writers submodule for microscopy data output.

This module provides specialized writers for various microscopy data formats:

- **OmeTiffStitcher**: Background-threaded writer for tiled mosaic OME-TIFF files
- **SingleTiffWriter**: Synchronous writer for multi-image TIFF files
- **OMEWriter**: Unified writer supporting OME-Zarr, stitched TIFF, and individual TIFFs
- **OMEWriterConfig**: Configuration dataclass for OMEWriter
- **OMEFileStorePaths**: Helper for managing output directory structure

Example Usage:
    >>> from imswitch.imcontrol.model.io.writers import (
    ...     OMEWriter, OMEWriterConfig, OMEFileStorePaths,
    ...     OmeTiffStitcher, SingleTiffWriter
    ... )
    >>> 
    >>> # For simple tile stitching:
    >>> stitcher = OmeTiffStitcher("/path/to/mosaic.ome.tif")
    >>> stitcher.start()
    >>> stitcher.add_image(img, pos_x=0, pos_y=0, index_x=0, index_y=0, pixel_size=0.325)
    >>> stitcher.stop()
    >>> 
    >>> # For comprehensive OME-Zarr + TIFF output:
    >>> config = OMEWriterConfig(write_zarr=True, write_stitched_tiff=True)
    >>> paths = OMEFileStorePaths("/path/to/experiment")
    >>> writer = OMEWriter(paths, tile_shape=(512, 512), grid_shape=(5, 5),
    ...                    grid_geometry=(0, 0, 500, 500), config=config)
    >>> writer.write_frame(image, {"x": 100, "y": 200})
    >>> writer.finalize()

Migration Notes:
    These classes were migrated from:
    - imswitch.imcontrol.controller.controllers.experiment_controller.OmeTiffStitcher
    - imswitch.imcontrol.controller.controllers.experiment_controller.SingleTiffWriter
    - imswitch.imcontrol.controller.controllers.experiment_controller.ome_writer
    
    The old locations are deprecated and will be removed in a future version.
"""

from .ome_tiff_stitcher import OmeTiffStitcher
from .single_tiff_writer import SingleTiffWriter
from .ome_writer import OMEWriter, OMEWriterConfig, OMEFileStorePaths
from .metadata import MinimalMetadata
from .minimal_zarr_source import MinimalZarrDataSource

__all__ = [
    # Primary unified writer
    'OMEWriter',
    'OMEWriterConfig',
    'OMEFileStorePaths',
    # Specialized writers
    'OmeTiffStitcher',
    'SingleTiffWriter',
    # Zarr streaming
    'MinimalZarrDataSource',
    # Metadata utilities
    'MinimalMetadata',
]
