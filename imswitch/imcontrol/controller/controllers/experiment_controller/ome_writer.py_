"""
DEPRECATED: This module has been migrated to imswitch.imcontrol.model.io.writers

Please use the new location for all new code:
    from imswitch.imcontrol.model.io import OMEWriter, OMEWriterConfig, OMEFileStorePaths

This file is maintained for backward compatibility only and will be removed
in a future version.

Original description:
Unified OME writer for both TIFF and OME-Zarr formats.
This module provides a reusable writer that can handle both individual TIFF files
and OME-Zarr mosaics, supporting both fast stage scan and normal stage scan modes.
"""

import warnings

# Emit deprecation warning when this module is imported
warnings.warn(
    "Importing from experiment_controller.ome_writer is deprecated. "
    "Use: from imswitch.imcontrol.model.io import OMEWriter, OMEWriterConfig, OMEFileStorePaths",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from imswitch.imcontrol.model.io import (
    OMEWriter,
    OMEWriterConfig,
    OMEFileStorePaths,
    OmeTiffStitcher,
    SingleTiffWriter,
)

__all__ = ['OMEWriter', 'OMEWriterConfig', 'OMEFileStorePaths', 'OmeTiffStitcher', 'SingleTiffWriter']
