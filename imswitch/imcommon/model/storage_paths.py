"""
Simple storage path management utilities.

# WebSocket events when drives are mounted or unmounted.This module provides background monitoring of storage devices and emits
This module provides straightforward functions for resolving storage paths
with clear precedence rules:
1. Runtime override (set via API)
2. Configuration/CLI argument
3. Default fallback

No complex abstractions - just simple, testable functions.
"""

import os
import shutil
from typing import Optional, List, Dict, Any, Tuple


def get_data_path() -> str:
    """
    Get the current data storage path.
    
    Precedence:
    1. Runtime override (set via API)
    2. Config data_folder (from CLI args or environment)
    3. Default: ~/ImSwitchConfig/data
    
    Returns:
        Absolute path to data storage directory
    """
    from imswitch.config import get_config

    config = get_config()

    # 1. Check runtime override (set via API)
    if hasattr(config, '_runtime_data_path') and config._runtime_data_path:
        return config._runtime_data_path

    # 2. Check config/CLI argument
    if config.data_folder and os.path.isdir(config.data_folder):
        return config.data_folder

    # 3. Fallback to default
    default = os.path.join(os.path.expanduser('~'), 'ImSwitchConfig', 'data')
    os.makedirs(default, exist_ok=True)
    return default


def get_config_path() -> str:
    """
    Get the configuration file path.
    
    Precedence:
    1. Config config_folder (from CLI args or environment)
    2. Default: ~/ImSwitchConfig
    
    Returns:
        Absolute path to configuration directory
    """
    from imswitch.config import get_config

    config = get_config()

    if config.config_folder and os.path.isdir(config.config_folder):
        return config.config_folder

    default = os.path.join(os.path.expanduser('~'), 'ImSwitchConfig')
    os.makedirs(default, exist_ok=True)
    return default


def set_data_path(path: str) -> Tuple[bool, str]:
    """
    Set runtime data path override.
    
    This sets a runtime override that takes precedence over configuration.
    The override persists until the application restarts.
    
    Args:
        path: Absolute path to new data directory
    
    Returns:
        Tuple of (success, error_message)
        - success: True if path was set successfully
        - error_message: Empty string on success, error description on failure
    """
    from imswitch.config import get_config

    # Validate path exists
    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"

    # Validate it's a directory
    if not os.path.isdir(path):
        return False, f"Path is not a directory: {path}"

    # Validate it's writable
    if not os.access(path, os.W_OK):
        return False, f"Path is not writable: {path}"

    # Set runtime override
    config = get_config()
    config._runtime_data_path = path

    return True, ""


def get_storage_info(path: Optional[str] = None) -> Dict[str, Any]:
    """
    Get storage information for a given path.
    
    Args:
        path: Path to check. If None, uses current data path.
    
    Returns:
        Dictionary with storage information:
        - path: Absolute path
        - exists: Whether path exists
        - writable: Whether path is writable
        - free_space_gb: Free space in GB
        - total_space_gb: Total space in GB
        - percent_used: Percentage of space used
    """
    if path is None:
        path = get_data_path()

    info = {
        "path": path,
        "exists": os.path.exists(path),
        "writable": False,
        "free_space_gb": 0.0,
        "total_space_gb": 0.0,
        "percent_used": 0.0
    }

    if not info["exists"]:
        return info

    # Check writability
    info["writable"] = os.access(path, os.W_OK)

    # Get disk usage
    try:
        usage = shutil.disk_usage(path)
        info["free_space_gb"] = round(usage.free / (1024**3), 2)
        info["total_space_gb"] = round(usage.total / (1024**3), 2)
        info["percent_used"] = round((usage.used / usage.total) * 100, 2)
    except Exception:
        pass

    return info


def scan_external_drives(base_paths: List[str]) -> List[Dict[str, Any]]:
    """
    Scan for external drives at given mount points.
    
    This function scans the specified mount paths for external storage devices.
    In Docker environments, this typically scans /media, /Volumes, or /datasets.
    In native environments, it scans OS-level mount points.
    
    No configuration flag required - scanning happens whenever mount paths are provided.
    
    Args:
        base_paths: List of mount directories to scan (e.g., ['/media', '/Volumes', '/datasets'])
    
    Returns:
        List of drive information dictionaries, empty list if no drives found
    
    Example:
        >>> drives = scan_external_drives(['/media', '/Volumes'])
        >>> for drive in drives:
        ...     print(f"{drive['label']}: {drive['free_space_gb']} GB free")
    """
    from .storage_scanner import StorageScanner

    scanner = StorageScanner()
    drives = scanner.scan_external_mounts(base_paths)
    return [drive.to_dict() for drive in drives]


def validate_path(path: str, min_free_gb: float = 1.0) -> Tuple[bool, str]:
    """
    Validate a path for use as storage location.
    
    Args:
        path: Path to validate
        min_free_gb: Minimum free space required in GB
    
    Returns:
        Tuple of (is_valid, error_message)
    """
    if not path:
        return False, "Path is empty"

    if not os.path.exists(path):
        return False, f"Path does not exist: {path}"

    if not os.path.isdir(path):
        return False, f"Path is not a directory: {path}"

    if not os.access(path, os.W_OK):
        return False, f"Path is not writable: {path}"

    # Check free space
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024**3)
        if free_gb < min_free_gb:
            return False, f"Insufficient free space: {free_gb:.2f} GB < {min_free_gb} GB"
    except Exception as e:
        return False, f"Could not check disk usage: {e}"

    return True, ""


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
