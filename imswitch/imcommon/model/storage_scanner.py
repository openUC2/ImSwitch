"""
Storage scanner for detecting and managing external storage devices.

This module provides functionality to scan for external storage devices
(USB drives, SD cards, etc.) and validate their writability and available space.
"""

import os
import shutil
from typing import List, Optional, Dict, Any
from dataclasses import dataclass, asdict


@dataclass
class ExternalStorage:
    """Represents an external storage device."""
    path: str
    label: str
    writable: bool
    free_space_gb: float
    total_space_gb: float
    filesystem: str = "unknown"
    is_active: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return asdict(self)


class StorageScanner:
    """Scanner for external storage devices."""

    # System volumes to exclude from scanning
    SYSTEM_VOLUMES = {
        "Macintosh HD",
        "System Volume Information",
        "Recovery",
        "Preboot",
        "VM",
        ".Spotlight-V100",
        ".fseventsd",
        ".Trashes"
    }

    def __init__(self):
        """Initialize the storage scanner."""
        pass

    def is_writable_directory(self, path: str) -> bool:
        """
        Check if a directory is writable by attempting to create and remove a test file.
        
        Args:
            path: Path to check
            
        Returns:
            True if the directory is writable, False otherwise
        """
        if not path or not os.path.isdir(path):
            return False
        try:
            test_file = os.path.join(path, ".write_test")
            with open(test_file, "w") as f:
                f.write("test")
            os.remove(test_file)
            return True
        except Exception:
            return False

    def get_disk_usage(self, path: str) -> tuple[float, float]:
        """
        Get disk usage for a given path.
        
        Args:
            path: Path to check disk usage
            
        Returns:
            Tuple of (free_space_gb, total_space_gb)
        """
        try:
            usage = shutil.disk_usage(path)
            free_gb = usage.free / (1024 ** 3)  # Convert bytes to GB
            total_gb = usage.total / (1024 ** 3)
            return free_gb, total_gb
        except Exception:
            return 0.0, 0.0

    def get_filesystem_type(self, path: str) -> str:
        """
        Get the filesystem type for a given path.
        
        Args:
            path: Path to check
            
        Returns:
            Filesystem type as string
        """
        # This is a simplified implementation
        # In production, you might want to use platform-specific tools
        try:
            if os.name == 'posix':
                # On Linux/Mac, we could parse /proc/mounts or use statvfs
                # For now, return a generic value
                return "ext4/ntfs/exfat"
            else:
                return "ntfs"
        except Exception:
            return "unknown"

    def is_system_volume(self, name: str) -> bool:
        """
        Check if a directory name represents a system volume.
        
        Args:
            name: Directory name to check
            
        Returns:
            True if this is a system volume, False otherwise
        """
        return name in self.SYSTEM_VOLUMES or name.startswith('.')

    def scan_external_mounts(self, base_paths: List[str]) -> List[ExternalStorage]:
        """
        Scan mount directories for external storage devices.
        
        Args:
            base_paths: List of mount point directories to scan (e.g., ["/media", "/Volumes"])
            
        Returns:
            List of detected external storage devices
        """
        detected_drives = []

        for base_path in base_paths:
            if not base_path or not os.path.exists(base_path):
                continue

            try:
                for entry in sorted(os.listdir(base_path)):
                    full_path = os.path.join(base_path, entry)

                    # Skip if not a directory
                    if not os.path.isdir(full_path):
                        continue

                    # Skip system volumes and hidden directories
                    if self.is_system_volume(entry):
                        continue

                    # Check if writable
                    writable = self.is_writable_directory(full_path)

                    # Get disk usage
                    free_gb, total_gb = self.get_disk_usage(full_path)

                    # Get filesystem type
                    filesystem = self.get_filesystem_type(full_path)

                    # Create ExternalStorage object
                    storage = ExternalStorage(
                        path=full_path,
                        label=entry,
                        writable=writable,
                        free_space_gb=round(free_gb, 2),
                        total_space_gb=round(total_gb, 2),
                        filesystem=filesystem,
                        is_active=False
                    )
                    if writable:
                        detected_drives.append(storage)

            except Exception as e:
                print(f"Error scanning {base_path}: {e}")
                continue

        return detected_drives

    def pick_first_external_folder(self, base_path: str) -> Optional[str]:
        """
        Pick the first suitable external folder from a mount directory.
        
        This is a legacy compatibility method that returns the first writable,
        non-system directory found in the base path.
        
        Args:
            base_path: Mount directory to scan
            
        Returns:
            Path to first suitable external folder, or None if none found
        """
        if not base_path or not os.path.exists(base_path):
            return None

        for d in sorted(os.listdir(base_path)):
            full_path = os.path.join(base_path, d)
            if not os.path.isdir(full_path):
                continue

            # Exclude system volumes and hidden directories
            if not self.is_system_volume(d) and self.is_writable_directory(full_path):
                return full_path

        return None

    def validate_storage_path(self, path: str, min_free_gb: float = 1.0) -> tuple[bool, str]:
        """
        Validate a storage path for use.
        
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

        if not self.is_writable_directory(path):
            return False, f"Path is not writable: {path}"

        free_gb, _ = self.get_disk_usage(path)
        if free_gb < min_free_gb:
            return False, f"Insufficient free space: {free_gb:.2f} GB < {min_free_gb} GB"

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
