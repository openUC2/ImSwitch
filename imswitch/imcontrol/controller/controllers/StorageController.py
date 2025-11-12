"""
Storage Controller - REST API endpoints for storage management.

This controller provides API endpoints for querying storage status,
listing external drives, and managing storage paths.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import HTTPException

from imswitch.imcommon.model.storage_manager import get_storage_manager
from imswitch.imcommon.model import initLogger


class SetActivePathRequest(BaseModel):
    """Request model for setting active storage path."""
    path: str
    persist: bool = False


class UpdateConfigPathRequest(BaseModel):
    """Request model for updating configuration paths."""
    config_path: Optional[str] = None
    data_path: Optional[str] = None
    persist: bool = False


class StorageController:
    """
    Controller for storage management API endpoints.
    
    This controller is not a traditional ImSwitch controller but provides
    static methods to be registered with the FastAPI server.
    """
    
    _logger = None
    
    @classmethod
    def _get_logger(cls):
        """Get or create logger instance."""
        if cls._logger is None:
            cls._logger = initLogger('StorageController')
        return cls._logger
    
    @staticmethod
    def get_storage_status() -> Dict[str, Any]:
        """
        Get current storage status including active path and available drives.
        
        Returns:
            Dictionary with storage status information
            
        Example response:
        {
            "active_path": "/media/usb-drive-1/datasets",
            "fallback_path": "/datasets",
            "available_external_drives": [...],
            "scan_enabled": true,
            "mount_paths": ["/media", "/Volumes"],
            "free_space_gb": 128.5,
            "total_space_gb": 256.0,
            "percent_used": 49.8
        }
        """
        try:
            storage_manager = get_storage_manager()
            status = storage_manager.get_storage_status()
            
            StorageController._get_logger().info(
                f"Storage status requested - active path: {status['active_path']}"
            )
            
            return status
        except Exception as e:
            StorageController._get_logger().error(f"Error getting storage status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    def list_external_drives() -> Dict[str, List[Dict[str, Any]]]:
        """
        List all detected external storage drives.
        
        Returns:
            Dictionary with list of external drives
            
        Example response:
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
        """
        try:
            storage_manager = get_storage_manager()
            drives = storage_manager.scan_external_drives()
            
            StorageController._get_logger().info(
                f"External drives listed - found {len(drives)} drive(s)"
            )
            
            return {
                "drives": [drive.to_dict() for drive in drives]
            }
        except Exception as e:
            StorageController._get_logger().error(f"Error listing external drives: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    def set_active_path(request: SetActivePathRequest) -> Dict[str, Any]:
        """
        Set the active storage path.
        
        Args:
            request: Request with path and persist flag
            
        Returns:
            Dictionary with success status and active path
            
        Example response:
        {
            "success": true,
            "active_path": "/media/usb-drive-1/datasets",
            "persisted": true,
            "message": "Storage path updated successfully"
        }
        """
        try:
            storage_manager = get_storage_manager()
            success, error_msg = storage_manager.set_data_path(
                request.path,
                persist=request.persist
            )
            
            if not success:
                StorageController._get_logger().warning(
                    f"Failed to set active path to {request.path}: {error_msg}"
                )
                raise HTTPException(status_code=400, detail=error_msg)
            
            StorageController._get_logger().info(
                f"Active storage path set to: {request.path} (persist={request.persist})"
            )
            
            return {
                "success": True,
                "active_path": storage_manager.get_active_data_path(),
                "persisted": request.persist,
                "message": "Storage path updated successfully"
            }
        except HTTPException:
            raise
        except Exception as e:
            StorageController._get_logger().error(f"Error setting active path: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    def get_config_paths() -> Dict[str, str]:
        """
        Get all configuration-related paths.
        
        Returns:
            Dictionary with configuration paths
            
        Example response:
        {
            "config_path": "/home/user/ImSwitchConfig",
            "data_path": "/datasets",
            "active_data_path": "/media/usb-drive-1/datasets"
        }
        """
        try:
            storage_manager = get_storage_manager()
            paths = storage_manager.get_config_paths()
            
            StorageController._get_logger().info("Configuration paths requested")
            
            return paths
        except Exception as e:
            StorageController._get_logger().error(f"Error getting config paths: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @staticmethod
    def update_config_paths(request: UpdateConfigPathRequest) -> Dict[str, Any]:
        """
        Update configuration paths.
        
        Args:
            request: Request with config_path and/or data_path
            
        Returns:
            Dictionary with success status and updated paths
            
        Example response:
        {
            "success": true,
            "message": "Configuration paths updated successfully",
            "config_path": "/custom/config/path",
            "data_path": "/custom/data/path"
        }
        """
        try:
            storage_manager = get_storage_manager()
            success, error_msg = storage_manager.update_config_paths(
                config_path=request.config_path,
                data_path=request.data_path,
                persist=request.persist
            )
            
            if not success:
                StorageController._get_logger().warning(
                    f"Failed to update config paths: {error_msg}"
                )
                raise HTTPException(status_code=400, detail=error_msg)
            
            StorageController._get_logger().info(
                f"Configuration paths updated - config: {request.config_path}, "
                f"data: {request.data_path}"
            )
            
            paths = storage_manager.get_config_paths()
            
            return {
                "success": True,
                "message": "Configuration paths updated successfully",
                "config_path": paths["config_path"],
                "data_path": paths["data_path"],
                "active_data_path": paths["active_data_path"]
            }
        except HTTPException:
            raise
        except Exception as e:
            StorageController._get_logger().error(f"Error updating config paths: {e}")
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
