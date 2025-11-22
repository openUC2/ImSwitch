"""
Unit tests for storage management system.

Tests the StorageScanner and StoragePathManager classes.
"""
import pytest
import os
import tempfile
import shutil
from pathlib import Path


# Import the modules to test
# We need to be careful with imports since they might depend on other imswitch modules
def test_storage_scanner_basic():
    """Test basic StorageScanner functionality."""
    from imswitch.imcommon.model.storage_scanner import StorageScanner, ExternalStorage
    
    scanner = StorageScanner()
    assert scanner is not None


def test_storage_scanner_is_writable():
    """Test the is_writable_directory method."""
    from imswitch.imcommon.model.storage_scanner import StorageScanner
    
    scanner = StorageScanner()
    
    # Test with a writable temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        assert scanner.is_writable_directory(tmpdir) is True
    
    # Test with non-existent path
    assert scanner.is_writable_directory("/nonexistent/path") is False
    
    # Test with a file (not a directory)
    with tempfile.NamedTemporaryFile() as tmpfile:
        assert scanner.is_writable_directory(tmpfile.name) is False


def test_storage_scanner_disk_usage():
    """Test disk usage retrieval."""
    from imswitch.imcommon.model.storage_scanner import StorageScanner
    
    scanner = StorageScanner()
    
    # Test with temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        free_gb, total_gb = scanner.get_disk_usage(tmpdir)
        
        assert free_gb >= 0
        assert total_gb > 0
        assert free_gb <= total_gb


def test_storage_scanner_is_system_volume():
    """Test system volume detection."""
    from imswitch.imcommon.model.storage_scanner import StorageScanner
    
    scanner = StorageScanner()
    
    # Test known system volumes
    assert scanner.is_system_volume("Macintosh HD") is True
    assert scanner.is_system_volume("System Volume Information") is True
    assert scanner.is_system_volume(".hidden") is True
    
    # Test regular names
    assert scanner.is_system_volume("USB_DRIVE") is False
    assert scanner.is_system_volume("data") is False


def test_storage_scanner_validate_path():
    """Test path validation."""
    from imswitch.imcommon.model.storage_scanner import StorageScanner
    
    scanner = StorageScanner()
    
    # Test with valid writable directory
    with tempfile.TemporaryDirectory() as tmpdir:
        is_valid, error_msg = scanner.validate_storage_path(tmpdir)
        assert is_valid is True
        assert error_msg == ""
    
    # Test with non-existent path
    is_valid, error_msg = scanner.validate_storage_path("/nonexistent/path")
    assert is_valid is False
    assert "does not exist" in error_msg.lower()
    
    # Test with empty path
    is_valid, error_msg = scanner.validate_storage_path("")
    assert is_valid is False
    assert "empty" in error_msg.lower()


def test_storage_scanner_scan_external_mounts():
    """Test scanning for external mounts."""
    from imswitch.imcommon.model.storage_scanner import StorageScanner
    
    scanner = StorageScanner()
    
    # Create a mock external mount structure
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create some mock mount points
        mount_base = os.path.join(tmpdir, "media")
        os.makedirs(mount_base)
        
        # Create a mock USB drive
        usb_drive = os.path.join(mount_base, "USB_DRIVE")
        os.makedirs(usb_drive)
        
        # Create a system volume (should be ignored)
        system_vol = os.path.join(mount_base, "System Volume Information")
        os.makedirs(system_vol)
        
        # Scan for drives
        drives = scanner.scan_external_mounts([mount_base])
        
        # Should find USB_DRIVE but not System Volume Information
        drive_names = [d.label for d in drives]
        assert "USB_DRIVE" in drive_names
        assert "System Volume Information" not in drive_names
        
        # Verify USB_DRIVE properties
        usb = next(d for d in drives if d.label == "USB_DRIVE")
        assert usb.path == usb_drive
        assert usb.writable is True
        assert usb.free_space_gb > 0


def test_external_storage_dataclass():
    """Test ExternalStorage dataclass."""
    from imswitch.imcommon.model.storage_scanner import ExternalStorage
    
    storage = ExternalStorage(
        path="/media/usb",
        label="USB_DRIVE",
        writable=True,
        free_space_gb=128.5,
        total_space_gb=256.0,
        filesystem="ext4",
        is_active=False
    )
    
    assert storage.path == "/media/usb"
    assert storage.label == "USB_DRIVE"
    assert storage.writable is True
    assert storage.free_space_gb == 128.5
    assert storage.total_space_gb == 256.0
    assert storage.filesystem == "ext4"
    assert storage.is_active is False
    
    # Test to_dict conversion
    storage_dict = storage.to_dict()
    assert isinstance(storage_dict, dict)
    assert storage_dict["path"] == "/media/usb"
    assert storage_dict["free_space_gb"] == 128.5


def test_storage_configuration():
    """Test StorageConfiguration dataclass."""
    from imswitch.imcommon.model.storage_manager import StorageConfiguration
    
    config = StorageConfiguration(
        default_data_path="/data",
        config_path="/config",
        enable_external_scanning=True,
        external_mount_paths=["/media", "/Volumes"]
    )
    
    assert config.default_data_path == "/data"
    assert config.config_path == "/config"
    assert config.enable_external_scanning is True
    assert config.external_mount_paths == ["/media", "/Volumes"]
    
    # Test to_dict
    config_dict = config.to_dict()
    assert isinstance(config_dict, dict)
    assert config_dict["default_data_path"] == "/data"
    
    # Test from_dict
    config2 = StorageConfiguration.from_dict(config_dict)
    assert config2.default_data_path == "/data"


def test_storage_path_manager_basic():
    """Test basic StoragePathManager functionality."""
    from imswitch.imcommon.model.storage_manager import StoragePathManager, StorageConfiguration
    
    config = StorageConfiguration(
        default_data_path=tempfile.gettempdir(),
        enable_external_scanning=False
    )
    
    manager = StoragePathManager(config)
    assert manager is not None
    
    # Test getting active data path
    active_path = manager.get_active_data_path()
    assert active_path is not None
    assert isinstance(active_path, str)


def test_storage_path_manager_get_config_path():
    """Test getting configuration path."""
    from imswitch.imcommon.model.storage_manager import StoragePathManager, StorageConfiguration
    
    config = StorageConfiguration(
        config_path="/custom/config"
    )
    
    manager = StoragePathManager(config)
    config_path = manager.get_config_path()
    assert config_path == "/custom/config"


def test_storage_path_manager_set_data_path():
    """Test setting data path."""
    from imswitch.imcommon.model.storage_manager import StoragePathManager, StorageConfiguration
    
    manager = StoragePathManager()
    
    # Set to a valid path
    with tempfile.TemporaryDirectory() as tmpdir:
        success, error_msg = manager.set_data_path(tmpdir, persist=False)
        assert success is True
        assert error_msg == ""
        assert manager.get_active_data_path() == tmpdir
    
    # Try to set to invalid path
    success, error_msg = manager.set_data_path("/nonexistent/path", persist=False)
    assert success is False
    assert error_msg != ""


def test_storage_path_manager_is_path_valid():
    """Test path validation through manager."""
    from imswitch.imcommon.model.storage_manager import StoragePathManager
    
    manager = StoragePathManager()
    
    # Valid path
    with tempfile.TemporaryDirectory() as tmpdir:
        assert manager.is_path_valid(tmpdir) is True
    
    # Invalid path
    assert manager.is_path_valid("/nonexistent/path") is False


def test_storage_path_manager_get_storage_status():
    """Test getting storage status."""
    from imswitch.imcommon.model.storage_manager import StoragePathManager, StorageConfiguration
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfiguration(
            default_data_path=tmpdir,
            enable_external_scanning=False
        )
        
        manager = StoragePathManager(config)
        status = manager.get_storage_status()
        
        assert isinstance(status, dict)
        assert "active_path" in status
        assert "fallback_path" in status
        assert "available_external_drives" in status
        assert "scan_enabled" in status
        assert "mount_paths" in status
        assert "free_space_gb" in status
        assert "total_space_gb" in status
        assert "percent_used" in status


def test_storage_path_manager_get_config_paths():
    """Test getting all configuration paths."""
    from imswitch.imcommon.model.storage_manager import StoragePathManager, StorageConfiguration
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfiguration(
            default_data_path=tmpdir,
            config_path="/config"
        )
        
        manager = StoragePathManager(config)
        paths = manager.get_config_paths()
        
        assert isinstance(paths, dict)
        assert "config_path" in paths
        assert "data_path" in paths
        assert "active_data_path" in paths
        assert paths["config_path"] == "/config"


def test_storage_path_manager_update_config_paths():
    """Test updating configuration paths."""
    from imswitch.imcommon.model.storage_manager import StoragePathManager
    
    manager = StoragePathManager()
    
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        # Update both paths
        success, error_msg = manager.update_config_paths(
            config_path=tmpdir1,
            data_path=tmpdir2,
            persist=False
        )
        
        assert success is True
        assert error_msg == ""
        
        # Verify paths were updated
        paths = manager.get_config_paths()
        assert paths["config_path"] == tmpdir1
        assert paths["active_data_path"] == tmpdir2


def test_storage_path_manager_fallback():
    """Test fallback behavior when preferred path is unavailable."""
    from imswitch.imcommon.model.storage_manager import StoragePathManager, StorageConfiguration
    
    with tempfile.TemporaryDirectory() as tmpdir:
        config = StorageConfiguration(
            default_data_path="/nonexistent/path",
            fallback_data_path=tmpdir
        )
        
        manager = StoragePathManager(config)
        active_path = manager.get_active_data_path()
        
        # Should fall back to fallback_data_path
        assert active_path == tmpdir


def test_get_storage_manager_singleton():
    """Test that get_storage_manager returns a singleton."""
    from imswitch.imcommon.model.storage_manager import get_storage_manager, set_storage_manager
    
    manager1 = get_storage_manager()
    manager2 = get_storage_manager()
    
    # Should be the same instance
    assert manager1 is manager2


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
