"""
Unified writers for acquisition data.

Re-exports writers from the model/writers module and provides
additional utilities for writer selection and configuration.
"""

import logging
from typing import Dict, Type, Optional, List

# Import from existing writers module
from ..writers.base import (
    WriterBase,
    SessionContext,
    DetectorContext as WriterDetectorContext,
    FrameEvent as WriterFrameEvent,
    WriterCapabilities,
)
from ..writers.registry import get_writer, WriterRegistry
from ..writers.ome_tiff_writer import OMETiffWriter
from ..writers.ome_zarr_writer import OMEZarrWriter

# Re-export metadata types from metadata module
try:
    from ..metadata import MetadataHub, DetectorContext, FrameEvent
    HAS_METADATA_HUB = True
except ImportError:
    HAS_METADATA_HUB = False
    DetectorContext = None
    FrameEvent = None
    MetadataHub = None

logger = logging.getLogger(__name__)


def convert_detector_context(ctx) -> WriterDetectorContext:
    """
    Convert MetadataHub DetectorContext to Writer DetectorContext.
    
    This allows writers to work with both metadata hub contexts
    and standalone writer contexts.
    """
    if isinstance(ctx, WriterDetectorContext):
        return ctx
    
    # Convert from MetadataHub DetectorContext
    return WriterDetectorContext(
        name=ctx.name,
        shape_px=ctx.shape_px,
        pixel_size_um=ctx.pixel_size_um,
        dtype=ctx.dtype,
        fov_um=ctx.fov_um,
        binning=ctx.binning,
        roi=ctx.roi,
        channel_name=ctx.channel_name,
        channel_color=ctx.channel_color,
        wavelength_nm=ctx.wavelength_nm,
        exposure_ms=ctx.exposure_ms,
        gain=ctx.gain,
    )


def convert_frame_event(event) -> WriterFrameEvent:
    """
    Convert MetadataHub FrameEvent to Writer FrameEvent.
    """
    if isinstance(event, WriterFrameEvent):
        return event
    
    # Convert from MetadataHub FrameEvent
    return WriterFrameEvent(
        frame_number=event.frame_number,
        timestamp=event.timestamp,
        detector_name=event.detector_name,
        stage_x_um=event.stage_x_um,
        stage_y_um=event.stage_y_um,
        stage_z_um=event.stage_z_um,
        exposure_ms=event.exposure_ms,
        laser_power_mw=getattr(event, 'laser_power_mw', None),
        metadata=event.metadata if hasattr(event, 'metadata') else {},
    )


def list_available_writers() -> List[str]:
    """List all registered writer formats."""
    return WriterRegistry.list_formats()


__all__ = [
    'WriterBase',
    'SessionContext',
    'WriterDetectorContext',
    'WriterFrameEvent',
    'WriterCapabilities',
    'OMETiffWriter',
    'OMEZarrWriter',
    'get_writer',
    'convert_detector_context',
    'convert_frame_event',
    'list_available_writers',
]


# Copyright (C) 2020-2024 ImSwitch developers
# This file is part of ImSwitch.
#
# ImSwitch is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# ImSwitch is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
