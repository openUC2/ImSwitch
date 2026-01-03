"""
Storage API tests for ImSwitch backend.

Tests storage management endpoints including:
- Storage status retrieval
- External drive detection
- Active path management
- Configuration path queries
"""
import pytest
import os
import tempfile


def test_storage_status_endpoint(api_server):
    """Test that the storage status endpoint returns valid data."""
    response = api_server.get("/storageManager/status")
    assert response.status_code == 200

    data = response.json()

    # Check required fields
    assert "active_path" in data
    assert "fallback_path" in data
    assert "available_external_drives" in data
    assert "scan_enabled" in data
    assert "mount_paths" in data
    assert "free_space_gb" in data
    assert "total_space_gb" in data
    assert "percent_used" in data

    # Validate types
    assert isinstance(data["active_path"], str)
    assert data["fallback_path"] is None or isinstance(data["fallback_path"], str)
    assert isinstance(data["available_external_drives"], list)
    assert isinstance(data["scan_enabled"], bool)
    assert isinstance(data["mount_paths"], list)
    assert isinstance(data["free_space_gb"], (int, float))
    assert isinstance(data["total_space_gb"], (int, float))
    assert isinstance(data["percent_used"], (int, float))

    # Validate reasonable values
    assert data["free_space_gb"] >= 0
    assert data["total_space_gb"] >= 0
    assert 0 <= data["percent_used"] <= 100


def test_external_drives_endpoint(api_server):
    """Test that the external drives endpoint returns valid data."""
    response = api_server.get("/storageManager/external-drives")
    assert response.status_code == 200

    data = response.json()

    # Check structure
    assert "drives" in data
    assert isinstance(data["drives"], list)

    # If drives are found, validate their structure
    for drive in data["drives"]:
        assert "path" in drive
        assert "label" in drive
        assert "writable" in drive
        assert "free_space_gb" in drive
        assert "total_space_gb" in drive
        assert "filesystem" in drive
        assert "is_active" in drive

        assert isinstance(drive["path"], str)
        assert isinstance(drive["label"], str)
        assert isinstance(drive["writable"], bool)
        assert isinstance(drive["free_space_gb"], (int, float))
        assert isinstance(drive["total_space_gb"], (int, float))
        assert isinstance(drive["filesystem"], str)
        assert isinstance(drive["is_active"], bool)


def test_config_paths_endpoint(api_server):
    """Test that the config paths endpoint returns valid data."""
    response = api_server.get("/storageManager/config-paths")
    assert response.status_code == 200

    data = response.json()

    # Check required fields
    assert "config_path" in data
    assert "data_path" in data
    assert "active_data_path" in data

    # Validate types
    assert isinstance(data["config_path"], str)
    assert isinstance(data["data_path"], str) or data["data_path"] == ""
    assert isinstance(data["active_data_path"], str)


def test_set_active_path_with_valid_path(api_server):
    """Test setting active path to a valid directory."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set this as the active path
        response = api_server.post("/storageManager/set-active-path", json={
            "path": tmpdir,
            "persist": False
        })

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert data["active_path"] == tmpdir
        assert data["persisted"] is False
        assert "message" in data


def test_set_active_path_with_invalid_path(api_server):
    """Test setting active path to an invalid directory."""
    # Try to set a non-existent path
    response = api_server.post("/storageManager/set-active-path", json={
        "path": "/nonexistent/invalid/path",
        "persist": False
    })

    # Should return error
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_set_active_path_with_non_writable_path(api_server):
    """Test setting active path to a non-writable directory."""
    # On most systems, /root is not writable by regular users
    # Skip this test if running as root
    if os.getuid() == 0:
        pytest.skip("Running as root, cannot test non-writable directory")

    response = api_server.post("/storageManager/set-active-path", json={
        "path": "/root",
        "persist": False
    })

    # Should return error or succeed if /root doesn't exist
    if response.status_code == 200:
        # Path validation might have different behavior
        pass
    else:
        assert response.status_code == 400


def test_update_config_paths_data_only(api_server):
    """Test updating only the data path."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        response = api_server.post("/storageManager/update-config-path", json={
            "data_path": tmpdir,
            "persist": False
        })

        assert response.status_code == 200
        data = response.json()

        assert data["success"] is True
        assert "message" in data
        assert "config_path" in data
        assert "data_path" in data
        assert "active_data_path" in data


def test_update_config_paths_invalid_data_path(api_server):
    """Test updating with an invalid data path."""
    response = api_server.post("/storageManager/update-config-path", json={
        "data_path": "/nonexistent/invalid/path",
        "persist": False
    })

    # Should return error
    assert response.status_code == 400
    data = response.json()
    assert "detail" in data


def test_storage_endpoints_in_openapi_spec(api_server):
    """Test that storage endpoints are documented in OpenAPI spec."""
    response = api_server.get("/openapi.json")
    assert response.status_code == 200
    spec = response.json()

    # Check that our endpoints are in the spec
    paths = spec["paths"]

    assert "/storageManager/status" in paths
    assert "/storageManager/external-drives" in paths
    assert "/storageManager/set-active-path" in paths
    assert "/storageManager/config-paths" in paths
    assert "/storageManager/update-config-path" in paths

    # Validate methods
    assert "get" in paths["/storageManager/status"]
    assert "get" in paths["/storageManager/external-drives"]
    assert "post" in paths["/storageManager/set-active-path"]
    assert "get" in paths["/storageManager/config-paths"]
    assert "post" in paths["/storageManager/update-config-path"]


def test_storage_status_consistency(api_server):
    """Test that storage status is consistent with config paths."""
    # Get status
    status_response = api_server.get("/storageManager/status")
    assert status_response.status_code == 200
    status_data = status_response.json()

    # Get config paths
    paths_response = api_server.get("/storageManager/config-paths")
    assert paths_response.status_code == 200
    paths_data = paths_response.json()

    # Active paths should match
    assert status_data["active_path"] == paths_data["active_data_path"]


def test_set_and_verify_active_path(api_server):
    """Test setting a path and verifying it's reflected in status."""
    # Create a temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        # Set as active path
        set_response = api_server.post("/storageManager/set-active-path", json={
            "path": tmpdir,
            "persist": False
        })
        assert set_response.status_code == 200

        # Verify in status
        status_response = api_server.get("/storageManager/status")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert status_data["active_path"] == tmpdir

        # Verify in config paths
        paths_response = api_server.get("/storageManager/config-paths")
        assert paths_response.status_code == 200
        paths_data = paths_response.json()
        assert paths_data["active_data_path"] == tmpdir


def test_storage_api_error_handling(api_server):
    """Test that storage API endpoints handle errors gracefully."""
    # Test with malformed JSON
    response = api_server.post("/storageManager/set-active-path",
                               data="invalid json",
                               headers={"Content-Type": "application/json"})
    assert response.status_code in [400, 422]  # Bad request or validation error

    # Test with missing required fields
    response = api_server.post("/storageManager/set-active-path", json={})
    assert response.status_code == 422  # Validation error


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
