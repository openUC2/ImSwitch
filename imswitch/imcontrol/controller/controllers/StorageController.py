"""
Storage Controller - REST API endpoints for storage management.

This controller provides API endpoints for querying storage status,
listing external drives, managing storage paths, and monitoring USB drives
using the simplified storage_paths module.
"""

from typing import Optional, Dict, Any
from pydantic import BaseModel
from fastapi import HTTPException
from imswitch.imcommon.framework import Signal
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self, tryInheritParent=False)
        self._monitor_started = False

        # Check if there's a saved storage path in setup config and use it
        self._load_persisted_path()

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

    @APIExport(runOnUIThread=False)
    def get_storage_status(self) -> Dict:
        """
        Get current storage status including active path and available drives.
        
        Returns:
            Dictionary with storage status information
        """
        try:
            config = get_config()
            data_path = get_data_path()
            config_path = get_config_path()

            # Get storage info for current data path
            storage_info = get_storage_info(data_path)

            # Build response
            status = {
                "active_path": data_path,
                "config_path": config_path,
                "fallback_path": config.data_folder if config.data_folder else None,
                "scan_enabled": config.scan_ext_data_folder,
                "mount_paths": config.ext_data_folder.split(',') if config.ext_data_folder else [],
                "exists": storage_info["exists"],
                "writable": storage_info["writable"],
                "free_space_gb": storage_info["free_space_gb"],
                "total_space_gb": storage_info["total_space_gb"],
                "percent_used": storage_info["percent_used"]
            }

            self._logger.info(
                f"Storage status requested - active path: {data_path}"
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
            config = get_config()

            # Get mount paths - use ext_data_folder if configured, otherwise use sensible defaults
            if config.ext_data_folder:
                mount_paths = [p.strip() for p in config.ext_data_folder.split(',')]
            else:
                # Default mount points for Docker and native environments
                mount_paths = ['/media', '/Volumes', '/datasets']

            # Scan for external drives
            drives = scan_external_drives(mount_paths)

            # Mark active drive
            active_path = get_data_path()
            for drive in drives:
                drive["is_active"] = (drive["path"] == active_path)

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
