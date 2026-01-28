"""
Metadata Hub and OME-types integration for ImSwitch.

This package provides:
- MetadataHub: Central aggregator for hardware state and detector metadata
- DetectorContext: Detector-specific metadata (pixel size, shape, transforms)
- Schema: Standardized metadata keys and normalization
- SharedAttrs Bridge: Connects legacy SharedAttributes to MetadataHub
"""

from .metadata_hub import MetadataHub, DetectorContext, FrameEvent
from .schema import MetadataSchema, MetadataCategory, SharedAttrValue
from .sharedattrs_bridge import SharedAttrsMetadataBridge

__all__ = [
    'MetadataHub',
    'DetectorContext',
    'FrameEvent',
    'MetadataSchema',
    'MetadataCategory',
    'SharedAttrValue',
    'SharedAttrsMetadataBridge',
]
