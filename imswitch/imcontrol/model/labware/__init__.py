"""Opentrons-compatible labware definition layer for ImSwitch.

Public surface:
  - LabwareDefinition, WellPosition, WellGeometry, LabwareDimensions
  - WellSelectionPattern, resolve_pattern
  - LabwareManager
  - load_labware_from_dict, load_labware_from_file, LabwareValidationError
  - generate_sbs_wellplate
"""

from .models import (
    LabwareDefinition,
    LabwareDimensions,
    OpentronsLabwareV2,
    WellGeometry,
    WellPosition,
)
from .loader import (
    LabwareValidationError,
    MM_TO_UM,
    load_labware_from_dict,
    load_labware_from_file,
)
from .selector import WellSelectionPattern, resolve_pattern
from .manager import LabwareManager
from .generators import generate_sbs_wellplate

__all__ = [
    "LabwareDefinition",
    "LabwareDimensions",
    "OpentronsLabwareV2",
    "WellGeometry",
    "WellPosition",
    "LabwareValidationError",
    "MM_TO_UM",
    "load_labware_from_dict",
    "load_labware_from_file",
    "WellSelectionPattern",
    "resolve_pattern",
    "LabwareManager",
    "generate_sbs_wellplate",
]
