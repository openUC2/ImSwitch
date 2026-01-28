"""
DEPRECATED: This module has been moved to imswitch.imcontrol.model.io.ome_writers.metadata

For new code, use:
    from imswitch.imcontrol.model.io import MinimalMetadata
"""
import warnings

warnings.warn(
    "Importing from experiment_controller.metadata is deprecated. "
    "Use: from imswitch.imcontrol.model.io import MinimalMetadata",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backwards compatibility
from imswitch.imcontrol.model.io import MinimalMetadata

__all__ = ['MinimalMetadata']
