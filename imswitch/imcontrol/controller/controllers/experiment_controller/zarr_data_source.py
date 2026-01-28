"""
DEPRECATED: This module has been moved to imswitch.imcontrol.model.io.ome_writers.minimal_zarr_source

For new code, use:
    from imswitch.imcontrol.model.io import MinimalZarrDataSource
"""
import warnings

warnings.warn(
    "Importing from experiment_controller.zarr_data_source is deprecated. "
    "Use: from imswitch.imcontrol.model.io import MinimalZarrDataSource",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backwards compatibility
from imswitch.imcontrol.model.io import MinimalZarrDataSource

__all__ = ['MinimalZarrDataSource']
