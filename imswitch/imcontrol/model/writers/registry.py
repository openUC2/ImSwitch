"""
Writer registry for mapping file formats to writer implementations.

Replaces the old DEFAULT_STORER_MAP pattern with a more flexible
registration system.
"""

from typing import Dict, Type, Optional
from .base import WriterBase
import logging

logger = logging.getLogger(__name__)


class WriterRegistry:
    """
    Registry for file format writers.
    
    Allows writers to be registered by format name and retrieved
    for use by RecordingManager, ExperimentController, etc.
    """
    
    _writers: Dict[str, Type[WriterBase]] = {}
    
    @classmethod
    def register(cls, format_name: str, writer_class: Type[WriterBase]):
        """
        Register a writer for a format.
        
        Args:
            format_name: Format identifier (e.g., 'OME_TIFF', 'OME_ZARR', 'PNG')
            writer_class: Writer class (subclass of WriterBase)
        """
        if not issubclass(writer_class, WriterBase):
            raise TypeError(f"{writer_class} must be a subclass of WriterBase")
        
        cls._writers[format_name.upper()] = writer_class
        logger.debug(f"Registered writer {writer_class.__name__} for format {format_name}")
    
    @classmethod
    def get(cls, format_name: str) -> Optional[Type[WriterBase]]:
        """
        Get writer class for a format.
        
        Args:
            format_name: Format identifier
        
        Returns:
            Writer class or None if not found
        """
        return cls._writers.get(format_name.upper())
    
    @classmethod
    def list_formats(cls) -> list:
        """List all registered formats."""
        return list(cls._writers.keys())
    
    @classmethod
    def clear(cls):
        """Clear all registered writers (mainly for testing)."""
        cls._writers.clear()


def register_writer(format_name: str):
    """
    Decorator to register a writer class.
    
    Usage:
        @register_writer('OME_TIFF')
        class OMETiffWriter(WriterBase):
            ...
    """
    def decorator(writer_class: Type[WriterBase]):
        WriterRegistry.register(format_name, writer_class)
        return writer_class
    return decorator


def get_writer(format_name: str) -> Optional[Type[WriterBase]]:
    """
    Get writer class for a format.
    
    Args:
        format_name: Format identifier
    
    Returns:
        Writer class or None if not found
    """
    return WriterRegistry.get(format_name)


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
