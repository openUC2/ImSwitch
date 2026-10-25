"""
Writers for experiment output.

- **OMEWriter**: the region canvas as OME-Zarr and/or OME-TIFF (``stitched.ome.tif``),
  individual OME-TIFF tiles, and OMERO upload
- **OMEWriterConfig**, **OMEFileStorePaths**: its configuration and folder layout
- **OMEROUploader**, **OMEROConnectionParams**, **TileMetadata**: OMERO streaming

All image files go through the shared OME writer in
``imswitch.imcontrol.model.aufgaben.storage`` (``NgffImage``, ``save_image``).

Example:
    >>> config = OMEWriterConfig(write_zarr=True, write_stitched_tiff=True)
    >>> paths = OMEFileStorePaths("/path/to/experiment")
    >>> writer = OMEWriter(paths, tile_shape=(512, 512), grid_shape=(5, 5),
    ...                    grid_geometry=(0, 0, 500, 500), config=config)
    >>> writer.write_frame(image, {"x": 100, "y": 200})
    >>> writer.finalize()
"""

from .ome_writer import OMEWriter, OMEWriterConfig, OMEFileStorePaths
from .metadata import MinimalMetadata
from .omero_uploader import (
    OMEROUploader,
    OMEROConnectionParams,
    TileMetadata,
    is_omero_available,
)
from .plate_metadata import build_plate_metadata, write_plate_metadata_sidecar

__all__ = [
    'OMEWriter',
    'OMEWriterConfig',
    'OMEFileStorePaths',
    'MinimalMetadata',
    # OME-NGFF plate metadata sidecar
    'build_plate_metadata',
    'write_plate_metadata_sidecar',
    # OMERO uploader
    'OMEROUploader',
    'OMEROConnectionParams',
    'TileMetadata',
    'is_omero_available',
]
