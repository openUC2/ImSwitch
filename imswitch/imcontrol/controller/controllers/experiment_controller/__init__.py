"""
Experiment controller modules for structured experiment execution.

This package provides modular components for handling different experiment
execution modes and shared functionality.

NOTE: The writer classes (OMEWriter, OMEWriterConfig, OMEFileStorePaths,
OmeTiffStitcher, SingleTiffWriter) have been migrated to the io/writers module.
For new code, please import from:
    from imswitch.imcontrol.model.io import OMEWriter, OMEWriterConfig, OMEFileStorePaths

The imports below are maintained for backward compatibility but are DEPRECATED.
"""

from .experiment_mode_base import ExperimentModeBase
from .experiment_performance_mode import ExperimentPerformanceMode
from .experiment_normal_mode import ExperimentNormalMode

# DEPRECATED: Import from imswitch.imcontrol.model.io instead
# These imports are maintained for backward compatibility only
from imswitch.imcontrol.model.io import (
    OMEWriter,
    OMEWriterConfig,
    OMEFileStorePaths,
    OmeTiffStitcher,
    SingleTiffWriter,
)

import warnings


def __getattr__(name):
    """Emit deprecation warning when accessing writer classes."""
    deprecated_classes = ['OMEWriter', 'OMEWriterConfig', 'OMEFileStorePaths', 
                          'OmeTiffStitcher', 'SingleTiffWriter']
    if name in deprecated_classes:
        warnings.warn(
            f"Importing {name} from experiment_controller is deprecated. "
            f"Use: from imswitch.imcontrol.model.io import {name}",
            DeprecationWarning,
            stacklevel=2
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    'ExperimentModeBase',
    'ExperimentPerformanceMode',
    'ExperimentNormalMode',
    # Deprecated - use imswitch.imcontrol.model.io instead
    'OMEFileStorePaths',
    'OMEWriter',
    'OMEWriterConfig',
]
