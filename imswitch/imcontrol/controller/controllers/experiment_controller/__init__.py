"""
Experiment controller modules for structured experiment execution.

This package provides modular components for handling different experiment
execution modes and shared functionality.

NOTE: The writer classes (OMEWriter, OMEWriterConfig, OMEFileStorePaths,
OmeTiffStitcher, SingleTiffWriter, MinimalMetadata, MinimalZarrDataSource) 
have been migrated to the io/ome_writers module.

For new code, please import from:
    from imswitch.imcontrol.model.io import (
        OMEWriter, OMEWriterConfig, OMEFileStorePaths,
        OmeTiffStitcher, SingleTiffWriter,
        MinimalMetadata, MinimalZarrDataSource,
    )

The imports below are maintained for backward compatibility but are DEPRECATED.
"""

from .experiment_mode_base import ExperimentModeBase
from .experiment_performance_mode import ExperimentPerformanceMode
from .experiment_normal_mode import ExperimentNormalMode

# Pydantic data models (split out of ExperimentController.py for readability).
from .models import (
    AutoFocusMode,
    AutoFocusSoftwareMethod,
    CenterPosition,
    Experiment,
    ExperimentWorkflowParams,
    FocusAlgorithm,
    FocusFitMethod,
    FocusMapConfig,
    FocusMapFromPointsRequest,
    KeepIlluminationMode,
    MDAChannelConfig,
    MDASequenceInfo,
    MDASequenceRequest,
    NeighborPoint,
    ParameterValue,
    Point,
    ScanArea,
    ScanBounds,
    ScanMetadata,
    ScanPattern,
    ScanPosition,
    StartExperimentResponse,
    TriggerMode,
)
from .execution_context import ExecutionContext


'''
from imswitch.imcontrol.model.io import (
    OMEWriter,
    OMEWriterConfig,
    OMEFileStorePaths,
    OmeTiffStitcher,
    SingleTiffWriter,
    MinimalMetadata,
    MinimalZarrDataSource,
)

import warnings


def __getattr__(name):
    """Emit deprecation warning when accessing migrated classes."""
    deprecated_classes = [
        'OMEWriter', 'OMEWriterConfig', 'OMEFileStorePaths', 
        'OmeTiffStitcher', 'SingleTiffWriter',
        'MinimalMetadata', 'MinimalZarrDataSource',
    ]
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
    'OmeTiffStitcher',
    'SingleTiffWriter',
    'MinimalMetadata',
    'MinimalZarrDataSource',
]

'''
