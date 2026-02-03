"""
InstrumentMetadataManager - Manager for microscope instrument metadata.

Provides comprehensive instrument information for OME-types metadata,
including UC2 components, firmware version, optical configuration, etc.

This integrates with the MetadataHub to provide a complete picture of
the instrument state at acquisition time.
"""

import json
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional
from pathlib import Path

from imswitch.imcommon.framework import Signal, SignalInterface
from imswitch.imcommon.model import initLogger


@dataclass
class OpticalComponent:
    """
    Represents an optical component in the microscope setup.
    
    Compatible with UC2 OptiKit JSON format and OME-types Instrument model.
    """
    name: str
    module_id: str
    description: str = ""
    grid_position: tuple = (0, 0, 0)
    rotation: tuple = (0, 0, 0)
    params: Dict[str, Any] = field(default_factory=dict)
    # OME-compatible fields
    manufacturer: str = "openUC2"
    model: str = ""
    serial_number: str = ""


@dataclass
class FilterInfo:
    """Filter set information for OME metadata."""
    name: str
    filter_type: str  # "Excitation", "Emission", "Dichroic"
    wavelength_nm: Optional[float] = None
    bandwidth_nm: Optional[float] = None
    manufacturer: str = ""
    model: str = ""


@dataclass
class InstrumentInfo:
    """
    Complete instrument metadata for OME-types integration.
    
    Maps to ome_types.model.Instrument structure.
    """
    # Microscope identification
    name: str = "openUC2 Microscope"
    microscope_type: str = "Inverted"  # or "Upright", "Other"
    manufacturer: str = "openUC2"
    model: str = "UC2 Frame"
    serial_number: str = ""
    firmware_version: str = ""
    
    # Configuration UUID - unique identifier for this optical configuration
    configuration_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # UC2 specific
    uc2_frame_name: str = ""
    uc2_frame_author: str = ""
    uc2_frame_version: str = "1.0.0"
    uc2_verified: bool = False
    
    # Components
    components: List[OpticalComponent] = field(default_factory=list)
    filters: List[FilterInfo] = field(default_factory=list)
    
    # Tube lens
    tube_lens_focal_length_mm: float = 180.0  # Standard Nikon tube lens
    tube_lens_magnification: float = 1.0
    
    # Timestamps
    created_at: str = ""
    last_modified: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'name': self.name,
            'microscope_type': self.microscope_type,
            'manufacturer': self.manufacturer,
            'model': self.model,
            'serial_number': self.serial_number,
            'firmware_version': self.firmware_version,
            'configuration_uuid': self.configuration_uuid,
            'uc2_frame_name': self.uc2_frame_name,
            'uc2_frame_author': self.uc2_frame_author,
            'uc2_frame_version': self.uc2_frame_version,
            'uc2_verified': self.uc2_verified,
            'components': [asdict(c) for c in self.components],
            'filters': [asdict(f) for f in self.filters],
            'tube_lens_focal_length_mm': self.tube_lens_focal_length_mm,
            'tube_lens_magnification': self.tube_lens_magnification,
            'created_at': self.created_at,
            'last_modified': self.last_modified,
        }


class InstrumentMetadataManager(SignalInterface):
    """
    Manager for microscope instrument metadata.
    
    Collects and provides instrument information from:
    - UC2 OptiKit configuration files
    - ESP32 firmware version
    - Setup JSON configuration
    - Runtime state
    
    This information is used to populate OME-types Instrument metadata
    during image acquisition and storage.
    """
    
    sigInstrumentInfoUpdated = Signal(object)  # InstrumentInfo
    
    def __init__(self, instrumentInfo=None, setupInfo=None, lowLevelManagers=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        
        self._setupInfo = setupInfo
        self._lowLevelManagers = lowLevelManagers or {}
        
        # Initialize instrument info
        self._instrument_info = InstrumentInfo(
            created_at=time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            last_modified=time.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        )
        
        # Load configuration if provided
        if instrumentInfo is not None:
            self._load_from_config(instrumentInfo)
        
        # Try to get firmware version from ESP32
        self._load_firmware_version()
        
        self.__logger.info(f"InstrumentMetadataManager initialized: {self._instrument_info.name}")
    
    def _load_from_config(self, config):
        """
        Load instrument info from SetupInfo.InstrumentInfo configuration.
        
        Handles the new config.json format with camelCase field names.
        """
        try:
            # Map SetupInfo.InstrumentInfo fields (camelCase) to InstrumentInfo fields (snake_case)
            if hasattr(config, 'name') and config.name:
                self._instrument_info.name = config.name
            if hasattr(config, 'microscopeType') and config.microscopeType:
                self._instrument_info.microscope_type = config.microscopeType
            if hasattr(config, 'manufacturer') and config.manufacturer:
                self._instrument_info.manufacturer = config.manufacturer
            if hasattr(config, 'model') and config.model:
                self._instrument_info.model = config.model
            if hasattr(config, 'serialNumber') and config.serialNumber:
                self._instrument_info.serial_number = config.serialNumber
            
            # Optical configuration
            if hasattr(config, 'tubeLensFocalLengthMm') and config.tubeLensFocalLengthMm:
                self._instrument_info.tube_lens_focal_length_mm = config.tubeLensFocalLengthMm
            if hasattr(config, 'tubeLensMagnification') and config.tubeLensMagnification:
                self._instrument_info.tube_lens_magnification = config.tubeLensMagnification
            
            # UC2 specific
            if hasattr(config, 'uc2FrameName') and config.uc2FrameName:
                self._instrument_info.uc2_frame_name = config.uc2FrameName
            if hasattr(config, 'uc2FrameAuthor') and config.uc2FrameAuthor:
                self._instrument_info.uc2_frame_author = config.uc2FrameAuthor
            if hasattr(config, 'uc2FrameVersion') and config.uc2FrameVersion:
                self._instrument_info.uc2_frame_version = config.uc2FrameVersion
            if hasattr(config, 'uc2Verified'):
                self._instrument_info.uc2_verified = config.uc2Verified
            
            # Load filters from config
            if hasattr(config, 'filters') and config.filters:
                for f in config.filters:
                    filter_info = FilterInfo(
                        name=f.get('name', ''),
                        filter_type=f.get('filterType', 'Emission'),
                        wavelength_nm=f.get('wavelengthNm'),
                        bandwidth_nm=f.get('bandwidthNm'),
                        manufacturer=f.get('manufacturer', ''),
                        model=f.get('model', ''),
                    )
                    self._instrument_info.filters.append(filter_info)
            
            # Load UC2 OptiKit config if path specified
            if hasattr(config, 'uc2OptiKitConfigPath') and config.uc2OptiKitConfigPath:
                self.load_uc2_optikit_config(config.uc2OptiKitConfigPath)
            
            self.__logger.info(f"Loaded instrument config: {self._instrument_info.name}")
            
        except Exception as e:
            self.__logger.warning(f"Error loading instrument config: {e}")
    
    def _load_firmware_version(self):
        """Try to load firmware version from ESP32."""
        try:
            if 'rs232sManager' in self._lowLevelManagers:
                rs232_manager = self._lowLevelManagers['rs232sManager']
                if 'ESP32' in rs232_manager:
                    esp32 = rs232_manager['ESP32']._esp32
                    if hasattr(esp32, 'state') and hasattr(esp32.state, 'get_state'):
                        state = esp32.state.get_state()
                        if 'firmware_version' in state:
                            self._instrument_info.firmware_version = state['firmware_version']
                            self.__logger.info(f"ESP32 firmware version: {self._instrument_info.firmware_version}")
        except Exception as e:
            self.__logger.debug(f"Could not get ESP32 firmware version: {e}")
    
    def load_uc2_optikit_config(self, config_path: str) -> bool:
        """
        Load UC2 OptiKit configuration from JSON file.
        
        Args:
            config_path: Path to UC2 OptiKit JSON configuration file
            
        Returns:
            True if loaded successfully
        """
        try:
            path = Path(config_path)
            if not path.exists():
                self.__logger.warning(f"OptiKit config not found: {config_path}")
                return False
            
            with open(path, 'r') as f:
                data = json.load(f)
            
            return self.load_uc2_optikit_dict(data)
            
        except Exception as e:
            self.__logger.error(f"Error loading OptiKit config: {e}")
            return False
    
    def load_uc2_optikit_dict(self, data: Dict[str, Any]) -> bool:
        """
        Load UC2 OptiKit configuration from dictionary.
        
        Parses the UC2 OptiKit JSON format and converts to InstrumentInfo.
        
        Args:
            data: Dictionary from UC2 OptiKit JSON
            
        Returns:
            True if loaded successfully
        """
        try:
            # Extract top-level metadata
            self._instrument_info.uc2_frame_name = data.get('name', '')
            self._instrument_info.uc2_frame_author = data.get('author', '')
            self._instrument_info.uc2_frame_version = data.get('version', '1.0.0')
            self._instrument_info.uc2_verified = data.get('uc2_verified', False)
            
            # Update name if frame name is provided
            if self._instrument_info.uc2_frame_name:
                self._instrument_info.name = f"openUC2: {self._instrument_info.uc2_frame_name}"
            
            # Parse metadata section
            if 'metadata' in data:
                meta = data['metadata']
                self._instrument_info.created_at = meta.get('created', '')
            
            # Parse UC2 components
            self._instrument_info.components = []
            for comp_data in data.get('uc2_components', []):
                component = OpticalComponent(
                    name=comp_data.get('name', ''),
                    module_id=comp_data.get('moduleId', ''),
                    description=comp_data.get('description', ''),
                    grid_position=tuple(comp_data.get('grid_pos', [0, 0, 0])),
                    rotation=tuple(comp_data.get('rotation', [0, 0, 0])),
                    params=comp_data.get('params', {}),
                    model=comp_data.get('originalName', ''),
                )
                self._instrument_info.components.append(component)
            
            # Extract filter information from components
            self._extract_filters_from_components()
            
            # Generate configuration UUID based on component hash
            self._generate_configuration_uuid()
            
            self._instrument_info.last_modified = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
            
            self.sigInstrumentInfoUpdated.emit(self._instrument_info)
            self.__logger.info(f"Loaded UC2 OptiKit config with {len(self._instrument_info.components)} components")
            
            return True
            
        except Exception as e:
            self.__logger.error(f"Error parsing OptiKit config: {e}")
            return False
    
    def _extract_filters_from_components(self):
        """Extract filter information from UC2 components."""
        self._instrument_info.filters = []
        
        for comp in self._instrument_info.components:
            module_id = comp.module_id.lower()
            
            # Dichroic filters
            if 'dichroic' in module_id or 'filter-dichroic' in module_id:
                filter_info = FilterInfo(
                    name=comp.name,
                    filter_type="Dichroic",
                    manufacturer="openUC2",
                    model=comp.model,
                )
                # Try to extract wavelength from description or name
                self._extract_wavelength(filter_info, comp)
                self._instrument_info.filters.append(filter_info)
            
            # Emission/Excitation filters
            elif 'emifil' in module_id or 'emission' in module_id:
                filter_info = FilterInfo(
                    name=comp.name,
                    filter_type="Emission",
                    manufacturer="openUC2",
                    model=comp.model,
                )
                self._extract_wavelength(filter_info, comp)
                self._instrument_info.filters.append(filter_info)
            
            elif 'excfil' in module_id or 'excitation' in module_id:
                filter_info = FilterInfo(
                    name=comp.name,
                    filter_type="Excitation",
                    manufacturer="openUC2",
                    model=comp.model,
                )
                self._extract_wavelength(filter_info, comp)
                self._instrument_info.filters.append(filter_info)
    
    def _extract_wavelength(self, filter_info: FilterInfo, component: OpticalComponent):
        """Try to extract wavelength from component name/description."""
        import re
        
        # Look for patterns like "532", "WLS532", "488nm", etc.
        text = f"{component.name} {component.description} {component.model}"
        
        # Match wavelength patterns
        match = re.search(r'(\d{3,4})\s*nm?', text, re.IGNORECASE)
        if match:
            filter_info.wavelength_nm = float(match.group(1))
    
    def _generate_configuration_uuid(self):
        """Generate a deterministic UUID based on component configuration."""
        import hashlib
        
        # Create a hash of component names and module IDs
        components_str = "|".join(
            f"{c.module_id}:{c.name}" 
            for c in sorted(self._instrument_info.components, key=lambda x: x.name)
        )
        
        hash_bytes = hashlib.sha256(components_str.encode()).digest()[:16]
        self._instrument_info.configuration_uuid = str(uuid.UUID(bytes=hash_bytes))
    
    # === Public API ===
    
    @property
    def instrument_info(self) -> InstrumentInfo:
        """Get current instrument info."""
        return self._instrument_info
    
    def get_ome_instrument_dict(self) -> Dict[str, Any]:
        """
        Get instrument information formatted for OME-types.
        
        Returns dictionary compatible with ome_types.model.Instrument.
        """
        return {
            'id': f"Instrument:{self._instrument_info.configuration_uuid}",
            'name': self._instrument_info.name,
            'microscope': {
                'type': self._instrument_info.microscope_type,
                'manufacturer': self._instrument_info.manufacturer,
                'model': self._instrument_info.model,
                'serial_number': self._instrument_info.serial_number,
            },
            # Custom annotations for UC2-specific data
            'annotation': {
                'firmware_version': self._instrument_info.firmware_version,
                'configuration_uuid': self._instrument_info.configuration_uuid,
                'uc2_frame_name': self._instrument_info.uc2_frame_name,
                'uc2_frame_author': self._instrument_info.uc2_frame_author,
                'tube_lens_focal_length_mm': self._instrument_info.tube_lens_focal_length_mm,
            },
        }
    
    def get_filters_for_channel(self, channel_name: str) -> List[FilterInfo]:
        """
        Get filters associated with a channel name.
        
        Args:
            channel_name: Name of the imaging channel
            
        Returns:
            List of FilterInfo objects for this channel
        """
        # Simple matching based on wavelength in channel name
        import re
        
        match = re.search(r'(\d{3,4})', channel_name)
        if not match:
            return []
        
        wavelength = float(match.group(1))
        
        # Return filters with similar wavelength (within 50nm)
        return [
            f for f in self._instrument_info.filters
            if f.wavelength_nm and abs(f.wavelength_nm - wavelength) < 50
        ]
    
    def set_firmware_version(self, version: str):
        """Update firmware version."""
        self._instrument_info.firmware_version = version
        self._instrument_info.last_modified = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        self.sigInstrumentInfoUpdated.emit(self._instrument_info)
    
    def set_tube_lens(self, focal_length_mm: float, magnification: float = 1.0):
        """Set tube lens parameters."""
        self._instrument_info.tube_lens_focal_length_mm = focal_length_mm
        self._instrument_info.tube_lens_magnification = magnification
        self._instrument_info.last_modified = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        self.sigInstrumentInfoUpdated.emit(self._instrument_info)
    
    def add_filter(self, filter_info: FilterInfo):
        """Add a filter to the instrument configuration."""
        self._instrument_info.filters.append(filter_info)
        self._instrument_info.last_modified = time.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        self.sigInstrumentInfoUpdated.emit(self._instrument_info)
    
    def to_json(self) -> str:
        """Serialize instrument info to JSON string."""
        return json.dumps(self._instrument_info.to_dict(), indent=2)
    
    def save_to_file(self, filepath: str):
        """Save instrument info to JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
        self.__logger.info(f"Saved instrument metadata to {filepath}")


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
