"""
Writers submodule for microscopy data output.

This module provides specialized writers for various microscopy data formats:

- **OmeTiffStitcher**: Background-threaded writer for tiled mosaic OME-TIFF files
- **SingleTiffWriter**: Synchronous writer for multi-image TIFF files
- **OMEWriter**: Unified writer supporting OME-Zarr, stitched TIFF, individual TIFFs, and OMERO
- **OMEWriterConfig**: Configuration dataclass for OMEWriter
- **OMEFileStorePaths**: Helper for managing output directory structure
- **OMEROUploader**: Thread-safe OMERO uploader for streaming tiles
- **OMEROConnectionParams**: OMERO connection parameters dataclass
- **TileMetadata**: Metadata for OMERO tile uploads

Example Usage:
    >>> from imswitch.imcontrol.model.io.ome_writers import (
    ...     OMEWriter, OMEWriterConfig, OMEFileStorePaths,
    ...     OmeTiffStitcher, SingleTiffWriter,
    ...     OMEROUploader, OMEROConnectionParams, TileMetadata
    ... )
    >>> 
    >>> # For simple tile stitching:
    >>> stitcher = OmeTiffStitcher("/path/to/mosaic.ome.tif")
    >>> stitcher.start()
    >>> stitcher.add_image(img, pos_x=0, pos_y=0, index_x=0, index_y=0, pixel_size=0.325)
    >>> stitcher.stop()
    >>> 
    >>> # For comprehensive OME-Zarr + TIFF + OMERO output:
    >>> config = OMEWriterConfig(write_zarr=True, write_stitched_tiff=True, write_omero=True)
    >>> paths = OMEFileStorePaths("/path/to/experiment")
    >>> omero_params = OMEROConnectionParams(host="omero.server.com", username="user", password="pass")
    >>> writer = OMEWriter(paths, tile_shape=(512, 512), grid_shape=(5, 5),
    ...                    grid_geometry=(0, 0, 500, 500), config=config,
    ...                    omero_connection_params=omero_params)
    >>> writer.write_frame(image, {"x": 100, "y": 200})
    >>> writer.finalize()

Migration Notes:
    These classes were migrated from:
    - imswitch.imcontrol.controller.controllers.experiment_controller.OmeTiffStitcher
    - imswitch.imcontrol.controller.controllers.experiment_controller.SingleTiffWriter
    - imswitch.imcontrol.controller.controllers.experiment_controller.ome_writer
    - imswitch.imcontrol.model.writers.omero_uploader
    
    The old locations are deprecated and will be removed in a future version.
"""

from .ome_tiff_stitcher import OmeTiffStitcher
from .single_tiff_writer import SingleTiffWriter
from .ome_writer import OMEWriter, OMEWriterConfig, OMEFileStorePaths
from .metadata import MinimalMetadata
from .minimal_zarr_source import MinimalZarrDataSource
from .omero_uploader import (
    OMEROUploader,
    OMEROConnectionParams,
    TileMetadata,
    is_omero_available,
)
from .ome_tiff_metadata import (
    build_ome_metadata,
    build_ome_metadata_from_dict,
    build_ome_instrument,
    OMEMetadataParams,
    OMEInstrumentTemplate,
    OME_TYPES_AVAILABLE,
)

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
    # OME-TIFF metadata (using ome_types library)
    'build_ome_metadata',
    'build_ome_metadata_from_dict',
    'build_ome_instrument',
    'OMEMetadataParams',
    'OMEInstrumentTemplate',
    'OME_TYPES_AVAILABLE',
    # OMERO uploader
    'OMEROUploader',
    'OMEROConnectionParams',
    'TileMetadata',
    'is_omero_available',
]
