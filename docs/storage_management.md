# Storage Management System

This document describes the new unified storage management system in ImSwitch, which provides centralized control over data and configuration paths with support for external storage devices.

## Overview

The storage management system consists of three main components:

1. **StorageScanner** - Detects and validates storage devices
2. **StoragePathManager** - Manages active storage paths
3. **StorageController** - Provides REST API endpoints

## Architecture

### StorageScanner

Located in `imswitch/imcommon/model/storage_scanner.py`

**Purpose**: Scan for and validate storage devices (USB drives, SD cards, etc.)

**Key Features**:
- Detect external storage devices in mount directories
- Check if directories are writable
- Calculate disk usage statistics
- Filter out system volumes
- Validate storage paths before use

**Usage**:
```python
from imswitch.imcommon.model.storage_scanner import StorageScanner

scanner = StorageScanner()

# Scan for external drives
drives = scanner.scan_external_mounts(["/media", "/Volumes"])

# Check if path is writable
is_writable = scanner.is_writable_directory("/path/to/check")

# Get disk usage
free_gb, total_gb = scanner.get_disk_usage("/path")

# Validate path for use
is_valid, error_msg = scanner.validate_storage_path("/path", min_free_gb=1.0)
```

### StoragePathManager

Located in `imswitch/imcommon/model/storage_manager.py`

**Purpose**: Single source of truth for all storage path management

**Key Features**:
- Unified configuration management
- Automatic external storage detection
- Fallback path handling
- Path validation and verification
- Preference persistence (optional)

**Usage**:
```python
from imswitch.imcommon.model.storage_manager import (
    get_storage_manager,
    StorageConfiguration
)

# Get the global storage manager
manager = get_storage_manager()

# Get active data path
data_path = manager.get_active_data_path()

# Set new data path
success, error = manager.set_data_path("/new/path", persist=True)

# Get storage status
status = manager.get_storage_status()

# Scan for external drives
drives = manager.scan_external_drives()
```

### StorageController

Located in `imswitch/imcontrol/controller/controllers/StorageController.py`

**Purpose**: Expose storage management via REST API

**Key Features**:
- Query storage status
- List external drives
- Change active storage path
- Update configuration paths

## REST API Endpoints

### GET /api/storage/status

Get comprehensive storage status.

**Response**:
```json
{
  "active_path": "/media/usb-drive-1/datasets",
  "fallback_path": "/datasets",
  "available_external_drives": [
    {
      "path": "/media/usb-drive-1",
      "label": "USB_DRIVE",
      "writable": true,
      "free_space_gb": 128.5,
      "total_space_gb": 256.0,
      "filesystem": "ext4",
      "is_active": true
    }
  ],
  "scan_enabled": true,
  "mount_paths": ["/media", "/Volumes"],
  "free_space_gb": 128.5,
  "total_space_gb": 256.0,
  "percent_used": 49.8
}
```

### GET /api/storage/external-drives

List detected external storage devices.

**Response**:
```json
{
  "drives": [
    {
      "path": "/media/usb-drive-1",
      "label": "USB_DRIVE",
      "writable": true,
      "free_space_gb": 128.5,
      "total_space_gb": 256.0,
      "filesystem": "ext4",
      "is_active": true
    }
  ]
}
```

### POST /api/storage/set-active-path

Set the active storage path.

**Request**:
```json
{
  "path": "/media/usb-drive-1/datasets",
  "persist": true
}
```

**Response**:
```json
{
  "success": true,
  "active_path": "/media/usb-drive-1/datasets",
  "persisted": true,
  "message": "Storage path updated successfully"
}
```

### GET /api/storage/config-paths

Get all configuration paths.

**Response**:
```json
{
  "config_path": "/home/user/ImSwitchConfig",
  "data_path": "/datasets",
  "active_data_path": "/media/usb-drive-1/datasets"
}
```

### POST /api/storage/update-config-path

Update configuration paths.

**Request**:
```json
{
  "config_path": "/custom/config/path",
  "data_path": "/custom/data/path",
  "persist": true
}
```

**Response**:
```json
{
  "success": true,
  "message": "Configuration paths updated successfully",
  "config_path": "/custom/config/path",
  "data_path": "/custom/data/path",
  "active_data_path": "/custom/data/path"
}
```

## Configuration

### Python API

```python
from imswitch.imcommon.model.storage_manager import (
    StoragePathManager,
    StorageConfiguration
)

config = StorageConfiguration(
    default_data_path="/datasets",
    config_path="/config",
    enable_external_scanning=True,
    external_mount_paths=["/media", "/Volumes"],
    fallback_data_path="/datasets",
    persist_storage_preferences=True
)

manager = StoragePathManager(config)
```

### Environment Variables (Docker)

When running in Docker, configure via environment variables:

```bash
# Enable external storage scanning
SCAN_EXT_DATA_PATH=true

# Mount point directory (where external drives are mounted)
EXT_DATA_PATH=/media

# Default data path
DATA_PATH=/datasets

# Configuration path
CONFIG_PATH=/config
```

See `docker/entrypoint.sh` for complete documentation.

## Integration with Legacy Code

The storage management system maintains backward compatibility:

### In dirtools.py

```python
from imswitch.imcommon.model.storage_manager import get_storage_manager

# Legacy code using UserFileDirs.Data still works
data_path = UserFileDirs.Data

# Internally uses storage manager if available
```

### In __main__.py

```python
# Storage manager is initialized early
storage_manager = get_storage_manager()
storage_manager.initialize_from_legacy_globals(
    config.config_folder,
    config.data_folder,
    config.scan_ext_data_folder,
    config.ext_data_folder
)
```

## Workflow Examples

### Basic Usage (No External Storage)

1. User starts ImSwitch
2. Storage manager initializes with default paths
3. Data saves to configured default location
4. API endpoints available for querying status

### External Storage Detection

1. User starts ImSwitch with `SCAN_EXT_DATA_PATH=true`
2. Storage manager scans mount directories
3. First writable external drive automatically selected
4. Data saves to external drive
5. If external drive removed, automatically falls back to default

### Runtime Path Switching

1. User inserts USB drive
2. Frontend calls `GET /api/storage/external-drives`
3. User sees available drives in UI
4. User clicks "Switch to USB"
5. Frontend calls `POST /api/storage/set-active-path`
6. New data saves to USB drive
7. With `persist=true`, preference saved for next session

## Testing

Comprehensive test coverage (32 tests):

### Unit Tests

Located in `imswitch/imcontrol/_test/unit/test_storage_manager.py`

Run with:
```bash
pytest imswitch/imcontrol/_test/unit/test_storage_manager.py -v
```

### API Tests

Located in `imswitch/imcontrol/_test/api/test_storage_api.py`

Run with:
```bash
pytest imswitch/imcontrol/_test/api/test_storage_api.py -v
```

## Migration Guide

### For Existing Deployments

The storage management system is **fully backward compatible**. No changes required.

### For New Features

To enable external storage detection:

**Native Python**:
```bash
python -m imswitch --headless --scan-ext-data-folder --ext-data-folder /media
```

**Docker**:
```bash
docker run -e SCAN_EXT_DATA_PATH=true -e EXT_DATA_PATH=/media ...
```

### For Frontend Integration

Use the REST API endpoints to:
1. Query available storage: `GET /api/storage/external-drives`
2. Display to user with current selection
3. Allow user to switch: `POST /api/storage/set-active-path`
4. Show storage status: `GET /api/storage/status`

## Troubleshooting

### External drives not detected

1. Check mount path is correct (`/media` on Linux, `/Volumes` on macOS)
2. Verify `SCAN_EXT_DATA_PATH=true` is set
3. Check logs for "Storage manager initialized" message
4. Ensure drive is writable

### Path validation fails

1. Check path exists: `ls -la /path`
2. Check permissions: `touch /path/test && rm /path/test`
3. Check disk space: `df -h /path`
4. Review logs for validation error details

### API endpoints not available

1. Verify ImSwitch started successfully
2. Check OpenAPI docs: `http://localhost:8001/docs`
3. Look for storage endpoints in `/api/storage/*`
4. Check server logs for errors

## Future Enhancements

Potential improvements for future versions:

1. **Real-time Monitoring**: Filesystem watching for mount/unmount events
2. **WebSocket Notifications**: Push notifications when storage changes
3. **Auto-Switch**: Automatically switch to external storage when available
4. **Space Warnings**: Notify when storage is low
5. **Multiple Drives**: Support for multiple external drives
6. **Drive Health**: SMART status monitoring
7. **Frontend UI**: Complete storage management interface

## References

- Issue: #[issue-number]
- Implementation PR: [PR-number]
- API Documentation: http://localhost:8001/docs
- Docker Documentation: `docker/entrypoint.sh`

## License

Copyright (C) 2020-2024 ImSwitch developers

This file is part of ImSwitch.

ImSwitch is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.
