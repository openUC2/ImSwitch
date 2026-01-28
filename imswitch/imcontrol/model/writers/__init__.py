"""
Writers package for ImSwitch - Unified I/O for all file formats.

This package provides a common interface for writing acquisition data
in various formats (OME-TIFF, OME-Zarr, PNG, JPG, MP4, etc.).

All writers implement the WriterBase interface and can be used from
both RecordingManager and ExperimentController.
"""

from .base import WriterBase, SessionContext, WriterCapabilities, DetectorContext, FrameEvent
from .registry import WriterRegistry, get_writer, register_writer
from .uuid_gen import compute_content_id, compute_uuid5, generate_session_uuid
from .ome_tiff_writer import OMETiffWriter
from .ome_zarr_writer import OMEZarrWriter

__all__ = [
    'WriterBase',
    'SessionContext',
    'DetectorContext',
    'FrameEvent',
    'WriterCapabilities',
    'WriterRegistry',
    'get_writer',
    'register_writer',
    'compute_content_id',
    'compute_uuid5',
    'generate_session_uuid',
    'OMETiffWriter',
    'OMEZarrWriter',
]

