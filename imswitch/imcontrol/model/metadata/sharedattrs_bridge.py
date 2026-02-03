"""
Bridge between legacy SharedAttributes and MetadataHub.

Subscribes to SharedAttributes signal updates and forwards them to the
MetadataHub with validation and normalization.
"""

from typing import Optional, List
import logging

from imswitch.imcommon.model import SharedAttributes
from .metadata_hub import MetadataHub
from .schema import MetadataSchema, MetadataCategory


logger = logging.getLogger(__name__)


class SharedAttrsMetadataBridge:
    """
    Bridge to connect SharedAttributes to MetadataHub.
    
    Listens to sharedAttrs.sigAttributeSet signal and pushes
    validated/normalized updates to the hub.
    
    This provides backwards compatibility while enabling the new
    metadata infrastructure.
    """
    
    def __init__(self, 
                 shared_attrs: SharedAttributes,
                 hub: MetadataHub,
                 categories: Optional[List[str]] = None):
        """
        Initialize the bridge.
        
        Args:
            shared_attrs: SharedAttributes instance to monitor
            hub: MetadataHub instance to update
            categories: Optional list of categories to forward (None = all)
        """
        self.shared_attrs = shared_attrs
        self.hub = hub
        self.categories = set(categories) if categories else None
        
        # Subscribe to attribute changes
        try:
            self.shared_attrs.sigAttributeSet.connect(self._on_attribute_set)
            logger.info("SharedAttrsMetadataBridge initialized")
        except Exception as e:
            logger.error(f"Failed to connect to SharedAttributes signal: {e}")
            raise
    
    def _on_attribute_set(self, key, value):
        """
        Handle SharedAttributes update signal.
        
        Args:
            key: Tuple key from SharedAttributes
            value: Raw value
        """
        try:
            # Validate key
            if not MetadataSchema.validate_key(key):
                logger.debug(f"Skipping invalid key: {key}")
                return
            
            # Filter by category if specified
            if self.categories and key[0] not in self.categories:
                return
            
            # Normalize and forward to hub
            # Hub will handle schema normalization internally
            self.hub.update(key, value, source='SharedAttributes')
            
        except Exception as e:
            logger.error(f"Error bridging attribute {key}: {e}")
    
    def disconnect(self):
        """Disconnect from SharedAttributes signals."""
        try:
            self.shared_attrs.sigAttributeSet.disconnect(self._on_attribute_set)
            logger.info("SharedAttrsMetadataBridge disconnected")
        except Exception as e:
            logger.error(f"Error disconnecting bridge: {e}")


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
