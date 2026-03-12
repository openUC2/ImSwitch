"""
Storage Controller - REST API endpoints for storage management.

This controller provides API endpoints for querying storage status,
listing external drives, managing storage paths, and monitoring USB drives
using the simplified storage_paths module.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import HTTPException
from imswitch.imcommon.framework import Signal, Timer
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController
import time

from imswitch.imcommon.model.storage_paths import (
    get_data_path,
    get_config_path,
    set_data_path,
    get_storage_info,
    scan_external_drives,
    validate_path
)
from imswitch.imcommon.model.storage_monitor import (
    get_storage_monitor,
    start_storage_monitoring
)
from imswitch.config import get_config


class SetActivePathRequest(BaseModel):
    """Request model for setting active storage path."""
    path: str
    persist: bool = False


class UpdateConfigPathRequest(BaseModel):
    """Request model for updating configuration paths."""
    config_path: Optional[str] = None
    data_path: Optional[str] = None
    persist: bool = False


class StorageController(ImConWidgetController):
    """
    Controller for storage management API endpoints.
    
    Provides API endpoints for storage path management, external drive detection,
    and USB drive monitoring with Signal-based notifications.
    """

    sigStorageDeviceChanged = Signal(dict)  # Emits: {'event': 'mounted'/'unmounted', 'timestamp': str, 'data': drive_info}
    sigStorageStatusUpdate = Signal(dict)   # Emits the current storage snapshot for frontend Redux/WebSocket consumers

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)
        self._monitor_started = False
        self._status_emit_interval_ms = 5000
        self._storage_status_timer = Timer()
        self._storage_status_timer.timeout.connect(self._emit_storage_status_update)

        # Check if there's a saved storage path in setup config and use it
        self._load_persisted_path()

        # Periodically push storage status to Socket.IO clients so the frontend
        # does not need to poll via HTTP.
        self._storage_status_timer.start(self._status_emit_interval_ms)
        self._emit_storage_status_update()

        # Start USB monitoring automatically on initialization
        if False: # TODO: We should have this enablable through the config?
            self._start_monitoring()

    def _load_persisted_path(self):
        """Load persisted storage path from setup config on startup."""
        try:
            if hasattr(self._setupInfo, 'storage') and self._setupInfo.storage:
                saved_path = self._setupInfo.storage.activeDataPath
                if saved_path:
                    # Validate and set the saved path
                    is_valid, error_msg = validate_path(saved_path)
                    if is_valid:
                        success, error_msg = set_data_path(saved_path)
                        if success:
                            self._logger.info(f"Loaded persisted storage path: {saved_path}")
                            # Refresh UserFileDirs with the loaded path
                            from imswitch.imcommon.model.dirtools import UserFileDirs
                            UserFileDirs.refresh_paths()
                        else:
                            self._logger.warning(f"Failed to set persisted path: {error_msg}")
                    else:
                        self._logger.warning(f"Persisted path no longer valid: {error_msg}")
        except Exception as e:
            self._logger.warning(f"Error loading persisted storage path: {e}")

    def _start_monitoring(self):
        """
        Start USB storage monitoring and setup WebSocket event callbacks.
        
        Called automatically during controller initialization.
        """
        if self._monitor_started:
            return

        config = get_config()

        # Get mount paths from config or use defaults
        if config.ext_data_folder:
            mount_paths = [p.strip() for p in config.ext_data_folder.split(',')]
        else:
            mount_paths = None  # Will use platform defaults

        # Start monitoring # TODO: @ethanjli maybe it's not necesaary as we explicitly call it from the API?
        if get_storage_monitor() is None:
            monitor = start_storage_monitoring(mount_paths=mount_paths, poll_interval=5)

        # Add callback to emit Signal when storage changes
        def on_storage_change(event_type: str, drive_info: Dict[str, Any]):
            """Emit Signal when storage changes."""
            try:
                event_data = {
                    'event': f'storage:device-{event_type}',
                    'timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    'data': drive_info
                }
                self.sigStorageDeviceChanged.emit(event_data)
                self._logger.info(f"Emitted storage event: {event_type} - {drive_info.get('path', 'unknown')}")
            except Exception as e:
                self._logger.error(f"Error emitting storage event: {e}")

        monitor.add_callback(on_storage_change)

        self._monitor_started = True
        self._logger.info(f"Storage monitoring started with mount paths: {mount_paths}")

    def _get_mount_paths(self):
        """Return configured storage mount roots, or sensible platform defaults."""
        config = get_config()
        if config.ext_data_folder:
            return [p.strip() for p in config.ext_data_folder.split(',') if p.strip()]
        return ['/media', '/Volumes', '/datasets']

    def _get_external_drives(self, active_path: Optional[str] = None):
        """Scan and annotate available external drives."""
        active_path = active_path or get_data_path()
        drives = scan_external_drives(self._get_mount_paths())

        for drive in drives:
            drive_path = drive.get("path") or drive.get("mount_point")
            if drive_path and "mount_point" not in drive:
                drive["mount_point"] = drive_path
            drive["is_active"] = bool(
                active_path and drive_path and active_path.startswith(drive_path)
            )

        return drives

    def _build_storage_status(self) -> Dict[str, Any]:
        """Build a normalized storage snapshot for both HTTP and WebSocket consumers."""
        config = get_config()
        active_path = get_data_path()
        config_path = get_config_path()
        storage_info = get_storage_info(active_path)

        total_bytes = int(round(storage_info.get("total_space_gb", 0.0) * (1024 ** 3)))
        free_bytes = int(round(storage_info.get("free_space_gb", 0.0) * (1024 ** 3)))
        used_bytes = max(total_bytes - free_bytes, 0)

        return {
            "active_path": active_path,
            "active_data_path": active_path,
            "config_path": config_path,
            "fallback_path": config.data_folder if config.data_folder else None,
            "scan_enabled": config.scan_ext_data_folder,
            "mount_paths": self._get_mount_paths(),
            "exists": storage_info["exists"],
            "writable": storage_info["writable"],
            "free_space_gb": storage_info["free_space_gb"],
            "total_space_gb": storage_info["total_space_gb"],
            "percent_used": storage_info["percent_used"],
            "disk_usage": {
                "free": free_bytes,
                "used": used_bytes,
                "total": total_bytes,
                "free_gb": storage_info["free_space_gb"],
                "total_gb": storage_info["total_space_gb"],
                "used_gb": round(used_bytes / (1024 ** 3), 2),
                "percent_used": storage_info["percent_used"],
            },
            "available_drives": self._get_external_drives(active_path),
            "updated_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        }

    def _emit_storage_status_update(self):
        """Push the latest storage snapshot over Socket.IO via the signal bridge."""
        try:
            self.sigStorageStatusUpdate.emit(self._build_storage_status())
        except Exception as e:
            self._logger.error(f"Error emitting storage status update: {e}")

    @APIExport(runOnUIThread=False)
    def get_storage_status(self) -> Dict:
        """
        Get current storage status including active path and available drives.
        
        Returns:
            Dictionary with storage status information
        """
        try:
            status = self._build_storage_status()

            self._logger.info(
                f"Storage status requested - active path: {status['active_path']}"
            )

            return status
        except Exception as e:
            self._logger.error(f"Error getting storage status: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @APIExport(runOnUIThread=False)
    def list_external_drives(self) -> Dict[str, Any]:
        """
        List all detected external storage drives.
        
        Scans configured mount paths for external drives. In Docker, this typically
        scans /media or /datasets. In native environments, scans OS-level mount points.
        
        Returns:
            Dictionary with list of external drives
        """
        try:
            drives = self._get_external_drives()

            self._logger.info(
                f"External drives listed - found {len(drives)} drive(s)"
            )

            return {"drives": drives}
        except Exception as e:
            self._logger.error(f"Error listing external drives: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @APIExport(runOnUIThread=False, requestType="POST")
    def set_active_path(self, path: str, persist: bool = False) -> Dict[str, Any]:
        """
        Set the active storage path.
        
        Creates an 'ImSwitchData' subfolder within the selected path where all
        ImSwitch data will be stored.
        
        Args:
            path: Base path to set as active storage location (e.g., USB drive path)
            persist: Whether to persist this setting to configuration
            
        Returns:
            Dictionary with success status and active path
        """
        try:
            import os

            # Validate the base path first
            is_valid, error_msg = validate_path(path)
            if not is_valid:
                self._logger.warning(
                    f"Invalid path {path}: {error_msg}"
                )
                raise HTTPException(status_code=400, detail=error_msg)

            # Create ImSwitchData subfolder within the selected path
            imswitch_data_path = os.path.join(path, "ImSwitchData")

            try:
                os.makedirs(imswitch_data_path, exist_ok=True)
                self._logger.info(f"Created/verified ImSwitchData folder at: {imswitch_data_path}")
            except Exception as e:
                error_msg = f"Failed to create ImSwitchData folder: {str(e)}"
                self._logger.error(error_msg)
                raise HTTPException(status_code=500, detail=error_msg)

            # Set the data path to the ImSwitchData subfolder
            success, error_msg = set_data_path(imswitch_data_path)

            if not success:
                self._logger.warning(
                    f"Failed to set active path to {imswitch_data_path}: {error_msg}"
                )
                raise HTTPException(status_code=400, detail=error_msg)

            # Refresh UserFileDirs paths
            from imswitch.imcommon.model.dirtools import UserFileDirs
            UserFileDirs.refresh_paths()

            # Persist to setup config if requested
            if persist:
                try:
                    # Ensure storage section exists in setup config
                    from imswitch.imcontrol.model.SetupInfo import StorageInfo
                    if not hasattr(self._setupInfo, 'storage') or self._setupInfo.storage is None:
                        self._setupInfo.storage = StorageInfo()

                    # Save the ImSwitchData path (not the base path)
                    self._setupInfo.storage.activeDataPath = imswitch_data_path

                    # Save setup config to disk
                    from imswitch.imcontrol.model import configfiletools
                    options, _ = configfiletools.loadOptions()
                    configfiletools.saveSetupInfo(options, self._setupInfo)

                    self._logger.info(f"Persisted storage path to setup config: {imswitch_data_path}")
                except Exception as e:
                    self._logger.error(f"Failed to persist storage path: {e}")
                    # Don't fail the whole operation if persist fails

            self._logger.info(
                f"Active storage path set to: {imswitch_data_path} (base: {path}, persist={persist})"
            )

            self._emit_storage_status_update()

            return {
                "success": True,
                "active_path": get_data_path(),
                "base_path": path,
                "persisted": persist,
                "message": "Storage path updated successfully (ImSwitchData folder created)"
            }
        except HTTPException:
            raise
        except Exception as e:
            self._logger.error(f"Error setting active path: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @APIExport(runOnUIThread=False)
    def get_config_paths(self) -> Dict:
        """
        Get all configuration-related paths.
        
        Returns:
            Dictionary with configuration paths:
            - config_path: Where ImSwitch configuration files are stored
            - data_path: Configured fallback data path from config file (may be None)
            - active_data_path: Currently active runtime data path (may differ from config)
        """
        try:
            paths = {
                "config_path": get_config_path(),
                "data_path": get_config().data_folder if get_config().data_folder else None,
                "active_data_path": get_data_path()
            }

            self._logger.info("Configuration paths requested")

            return paths
        except Exception as e:
            self._logger.error(f"Error getting config paths: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @APIExport(runOnUIThread=False, requestType="POST")
    def update_config_paths(self, config_path: Optional[str] = None,
                           data_path: Optional[str] = None,
                           persist: bool = False) -> Dict[str, Any]:
        """
        Update configuration paths.
        
        Args:
            config_path: Optional path to configuration directory
            data_path: Optional path to data directory
            persist: Whether to persist changes to configuration
            
        Returns:
            Dictionary with success status and updated paths
        """
        try:
            config = get_config()

            # Update configuration
            if config_path:
                config.config_folder = config_path

            if data_path:
                config.data_folder = data_path
                # Also set as runtime override
                success, error_msg = set_data_path(data_path)
                if not success:
                    raise HTTPException(status_code=400, detail=error_msg)

            # Refresh paths
            from imswitch.imcommon.model.dirtools import UserFileDirs
            UserFileDirs.refresh_paths()

            self._logger.info(
                f"Configuration paths updated - config: {config_path}, "
                f"data: {data_path}"
            )

            paths = {
                "config_path": get_config_path(),
                "data_path": config.data_folder,
                "active_data_path": get_data_path()
            }

            return {
                "success": True,
                "message": "Configuration paths updated successfully",
                **paths
            }
        except HTTPException:
            raise
        except Exception as e:
            self._logger.error(f"Error updating config paths: {e}")
            raise HTTPException(status_code=500, detail=str(e))


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
