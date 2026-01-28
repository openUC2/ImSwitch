"""
DEPRECATED: This module has been migrated to imswitch.imcontrol.model.io.writers

Please use the new location for all new code:
    from imswitch.imcontrol.model.io import OmeTiffStitcher

This file is maintained for backward compatibility only and will be removed
in a future version.
"""

import warnings

# Emit deprecation warning when this module is imported
warnings.warn(
    "Importing from experiment_controller.OmeTiffStitcher is deprecated. "
    "Use: from imswitch.imcontrol.model.io import OmeTiffStitcher",
    DeprecationWarning,
    stacklevel=2
)

# Re-export from new location for backward compatibility
from imswitch.imcontrol.model.io import OmeTiffStitcher

__all__ = ['OmeTiffStitcher']
