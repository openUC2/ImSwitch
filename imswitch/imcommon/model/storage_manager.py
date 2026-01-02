"""
Storage path manager - Single source of truth for all storage path management.

This module provides a centralized manager for handling data storage paths,
including support for external drives, automatic fallback, and path validation.
"""

import os
import json
from typing import Optional, List, Dict, Any
from dataclasses import dataclass, asdict

from .storage_scanner import StorageScanner, ExternalStorage


@dataclass
class StorageConfiguration:
    """Configuration for storage path management."""

    # Primary data storage location
    default_data_path: Optional[str] = None

    # Configuration file location
    config_path: Optional[str] = None

    # External mount point scanning
    enable_external_scanning: bool = False
    external_mount_paths: List[str] = None

    # Current active storage (runtime state)
    active_data_path: Optional[str] = None

    # Fallback when external storage unavailable
    fallback_data_path: Optional[str] = None

    # Persistence
    persist_storage_preferences: bool = False

    def __post_init__(self):
        """Initialize default values after dataclass initialization."""
        if self.external_mount_paths is None:
            self.external_mount_paths = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StorageConfiguration':
        """Create StorageConfiguration from dictionary."""
        return cls(**data)


class StoragePathManager:
    """
    Single source of truth for all storage path management.
    
    This class manages both configuration and data paths, handles external
    storage detection, and provides path validation.
    """

    def __init__(self, config: Optional[StorageConfiguration] = None):
        """
        Initialize the storage path manager.
        
        Args:
            config: Storage configuration. If None, uses defaults.
        """
        self.config = config or StorageConfiguration()
        self.scanner = StorageScanner()
        self._preference_file = None

    def get_active_data_path(self) -> str:
        """
        Get the current active data storage path.
        
        This method delegates to UserFileDirs.getValidatedDataPath() which is
        the central source of truth for data path resolution.
        
        Returns:
            Current active data storage path (guaranteed to exist)
        """
        from imswitch.imcommon.model.dirtools import UserFileDirs

        # Delegate to dirtools as the source of truth
        validated_path = UserFileDirs.getValidatedDataPath()

        # Update our internal state to match
        self.config.active_data_path = validated_path

        return validated_path

    def get_config_path(self) -> str:
        """
        Get the configuration file path.
        
        Returns:
            Configuration file path
        """
        if self.config.config_path:
            return self.config.config_path

        # Default to user's config directory
        '''
        TODO: This is probably wrong as we also set it in the dirtools 
        we probably need to merge it 
        # TODO: Is this actually needed at all? The _baseDataFilesDir should point to the dataset directoy, right now it doesn't 
        _baseDataFilesDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '_data') # User defaults
        _baseUserFilesDir = os.path.join(getSystemUserDir()) # User configuration files
        # TODO: Shall we have global a dataset directory here too?
        '''
        return os.path.expanduser("~/ImSwitchConfig")


    def set_data_path(self, path: str, persist: bool = False) -> tuple[bool, str]:
        """
        Set a new data path with validation.
        
        Args:
            path: New data path to set
            persist: Whether to persist this preference
            
        Returns:
            Tuple of (success, error_message)
            
        Security: Path is normalized to absolute path to prevent directory traversal.
        """
        # Normalize path to absolute path
        normalized_path = os.path.abspath(path)

        # Validate the normalized path # TODO: If it does not exist, we should create it if possible
        is_valid, error_msg = self.scanner.validate_storage_path(normalized_path)
        if not is_valid:
            return False, error_msg

        # Set as active path (using normalized path)
        self.config.active_data_path = normalized_path

        # Persist if requested (use normalized path)
        if persist and self.config.persist_storage_preferences:
            self._save_preference(normalized_path)

        return True, ""

    def scan_external_drives(self) -> List[ExternalStorage]:
        """
        Scan for available external storage devices.
        
        Returns:
            List of detected external storage devices
        """
        if not self.config.enable_external_scanning: # TODO: We should scan for external drives in any way (either natively on OS level or on the dedicated volume mount provided by docker); Also why not enable => should rather be enable?
            return []

        drives = self.scanner.scan_external_mounts(self.config.external_mount_paths) # TODO: this should depend on OS configuration (e.g. /media in case of linux, etc. ); for the docker case we should mount that under a new volume that is available under root e.g. /datasets - if this folder exists we should scan this

        # Mark the active drive if it matches
        active_path = self.config.active_data_path
        if active_path:
            for drive in drives:
                if active_path.startswith(drive.path):
                    drive.is_active = True
                    break

        return drives

    def is_path_valid(self, path: str) -> bool:
        """
        Validate if a path exists and is writable.
        
        Args:
            path: Path to validate
            
        Returns:
            True if path is valid, False otherwise
        """
        is_valid, _ = self.scanner.validate_storage_path(path)
        return is_valid

    def get_storage_status(self) -> Dict[str, Any]:
        """
        Get comprehensive storage status information.
        
        Returns:
            Dictionary with storage status information
        """
        active_path = self.get_active_data_path()

        # Get disk usage for active path
        free_gb, total_gb = self.scanner.get_disk_usage(active_path)

        return {
            "active_path": active_path,
            "fallback_path": self.config.fallback_data_path,
            "available_external_drives": [d.to_dict() for d in self.scan_external_drives()],
            "scan_enabled": self.config.enable_external_scanning,
            "mount_paths": self.config.external_mount_paths,
            "free_space_gb": round(free_gb, 2),
            "total_space_gb": round(total_gb, 2),
            "percent_used": round((1 - free_gb / total_gb) * 100, 2) if total_gb > 0 else 0
        }

    def get_config_paths(self) -> Dict[str, str]:
        """
        Get all configuration-related paths.
        
        Returns:
            Dictionary with configuration paths
        """
        return {
            "config_path": self.get_config_path(),
            "data_path": self.config.default_data_path or "",
            "active_data_path": self.get_active_data_path()
        }

    def update_config_paths(self, config_path: Optional[str] = None,
                           data_path: Optional[str] = None,
                           persist: bool = False) -> tuple[bool, str]:
        """
        Update configuration paths.
        
        Args:
            config_path: New configuration path
            data_path: New data path
            persist: Whether to persist these changes
            
        Returns:
            Tuple of (success, error_message)
            
        Security: Paths are validated and normalized to prevent directory traversal.
        """
        if config_path:
            # Normalize and validate the path
            normalized_path = os.path.abspath(config_path)

            # Additional security: check path doesn't contain suspicious patterns
            if '..' in os.path.normpath(config_path):
                return False, "Invalid path: directory traversal detected"

            # Use the normalized path for all operations
            if not os.path.isdir(normalized_path):
                return False, f"Configuration path does not exist: {normalized_path}"
            self.config.config_path = normalized_path

        if data_path:
            # Path validation is done in set_data_path
            success, error_msg = self.set_data_path(data_path, persist=persist)
            if not success:
                return False, error_msg
            self.config.default_data_path = os.path.abspath(data_path)

        return True, ""

    def _get_preference_file_path(self) -> str:
        """
        Get the path to the storage preference file.
        
        Security: Returns an absolute, normalized path to prevent directory traversal.
        """
        if self._preference_file:
            return os.path.abspath(self._preference_file)

        config_dir = self.get_config_path()
        # Normalize and make absolute to prevent directory traversal
        preference_file = os.path.abspath(os.path.join(config_dir, "storage_preferences.json"))

        # Verify the preference file is within the config directory
        if not preference_file.startswith(os.path.abspath(config_dir)):
            raise ValueError("Invalid preference file path - directory traversal detected")

        return preference_file

    def _save_preference(self, path: str) -> None:
        """
        Save storage path preference to file.
        
        Args:
            path: Path to save as preference
        """
        try:
            preference_file = self._get_preference_file_path()
            os.makedirs(os.path.dirname(preference_file), exist_ok=True)

            with open(preference_file, 'w') as f:
                json.dump({"preferred_data_path": path}, f, indent=2)
        except Exception as e:
            print(f"Warning: Failed to save storage preference: {e}")

    def _load_preference(self) -> Optional[str]:
        """
        Load storage path preference from file.
        
        Returns:
            Preferred path if found, None otherwise
        """
        try:
            preference_file = self._get_preference_file_path()
            if os.path.exists(preference_file):
                with open(preference_file, 'r') as f:
                    data = json.load(f)
                    return data.get("preferred_data_path")
        except Exception as e:
            print(f"Warning: Failed to load storage preference: {e}")

        return None

    def initialize_from_legacy_globals(self, default_config_path: Optional[str],
                                      default_data_path: Optional[str],
                                      scan_ext_data_path: bool,
                                      ext_data_path: Optional[str]) -> None:
        """
        Initialize storage manager from legacy global variables.
        
        This method provides backward compatibility with the old configuration system.
        
        Args:
            default_config_path: Legacy DEFAULT_CONFIG_PATH
            default_data_path: Legacy DEFAULT_DATA_PATH
            scan_ext_data_path: Legacy SCAN_EXT_DATA_PATH
            ext_data_path: Legacy EXT_DATA_PATH
        """
        # Set configuration path
        if default_config_path:
            self.config.config_path = default_config_path

        # Set default data path
        if default_data_path:
            self.config.default_data_path = default_data_path
            self.config.fallback_data_path = default_data_path

        # Set external scanning
        self.config.enable_external_scanning = scan_ext_data_path

        # Set external mount paths
        if ext_data_path:
            if isinstance(ext_data_path, str):
                self.config.external_mount_paths = [ext_data_path]
            else:
                self.config.external_mount_paths = ext_data_path

        # Try to load saved preference
        if self.config.persist_storage_preferences:
            preferred_path = self._load_preference()
            if preferred_path and self.is_path_valid(preferred_path):
                self.config.active_data_path = preferred_path


# Global storage manager instance
_storage_manager: Optional[StoragePathManager] = None


def get_storage_manager() -> StoragePathManager:
    """
    Get the global storage manager instance.
    
    Returns:
        Global StoragePathManager instance
    """
    global _storage_manager
    if _storage_manager is None:
        _storage_manager = StoragePathManager()
    return _storage_manager


def set_storage_manager(manager: StoragePathManager) -> None:
    """
    Set the global storage manager instance.
    
    Args:
        manager: StoragePathManager instance to set as global
    """
    global _storage_manager
    _storage_manager = manager


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
