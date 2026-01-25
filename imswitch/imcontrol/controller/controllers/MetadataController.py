"""
MetadataController - API endpoints for the MetadataHub and InstrumentMetadataManager.

Provides REST API endpoints to:
- Query current metadata state
- Get instrument information
- Access detector contexts
- Get frame events
"""

from typing import Any, Dict, List, Optional
import json

from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController


class MetadataController(ImConWidgetController):
    """
    Controller providing API access to MetadataHub and InstrumentMetadataManager.
    
    This controller exposes the metadata state via REST API endpoints,
    enabling external tools and frontends to visualize the metadata.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.__logger = initLogger(self)
        
        # Get references to metadata managers from MasterController
        self._metadata_hub = getattr(self._master, 'metadataHub', None)
        self._instrument_manager = getattr(self._master, 'instrumentMetadataManager', None)
        
        if self._metadata_hub is None:
            self.__logger.warning("MetadataHub not available")
        else:
            self.__logger.info("MetadataController initialized with MetadataHub")
        
        if self._instrument_manager is None:
            self.__logger.warning("InstrumentMetadataManager not available")

    # === Global Metadata API ===
    
    @APIExport()
    def getMetadataSnapshot(self, flat: bool = False, category: str = None) -> Dict[str, Any]:
        """
        Get a snapshot of the current global metadata state.
        
        Args:
            flat: If True, return flat dict with ':' separated keys
            category: Optional category filter (e.g., "Positioner", "Illumination")
        
        Returns:
            Dictionary containing current metadata values with timestamps
        """
        if self._metadata_hub is None:
            return {"error": "MetadataHub not available"}
        
        return self._metadata_hub.get_latest(flat=flat, filter_category=category)
    
    @APIExport()
    def getMetadataCategories(self) -> List[str]:
        """
        Get list of available metadata categories.
        
        Returns:
            List of category names (e.g., ["Positioner", "Illumination", "Detector"])
        """
        if self._metadata_hub is None:
            return []
        
        metadata = self._metadata_hub.get_latest(flat=True)
        categories = set()
        for key in metadata.keys():
            parts = key.split(':')
            if parts:
                categories.add(parts[0])
        return sorted(list(categories))
    
    @APIExport()
    def getMetadataJSON(self) -> str:
        """
        Get metadata as JSON string for external consumption.
        
        Returns:
            JSON string of current metadata state
        """
        if self._metadata_hub is None:
            return json.dumps({"error": "MetadataHub not available"})
        
        return self._metadata_hub.to_json()
    
    # === Detector Context API ===
    
    @APIExport()
    def getDetectorContext(self, detectorName: str) -> Dict[str, Any]:
        """
        Get the metadata context for a specific detector.
        
        Args:
            detectorName: Name of the detector
        
        Returns:
            Dictionary with detector context (shape, pixel size, exposure, etc.)
        """
        if self._metadata_hub is None:
            return {"error": "MetadataHub not available"}
        
        ctx = self._metadata_hub.get_detector(detectorName)
        if ctx is None:
            return {"error": f"Detector '{detectorName}' not registered"}
        
        return ctx.to_dict()
    
    @APIExport()
    def getAllDetectorContexts(self) -> Dict[str, Dict[str, Any]]:
        """
        Get metadata contexts for all registered detectors.
        
        Returns:
            Dictionary mapping detector names to their contexts
        """
        if self._metadata_hub is None:
            return {"error": "MetadataHub not available"}
        
        return self._metadata_hub.export_detector_contexts()
    
    # === Frame Events API ===
    
    @APIExport()
    def getFrameEvents(self, detectorName: str, maxEvents: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent frame events for a detector.
        
        Args:
            detectorName: Name of the detector
            maxEvents: Maximum number of events to return
        
        Returns:
            List of frame event dictionaries
        """
        if self._metadata_hub is None:
            return []
        
        events = self._metadata_hub.get_frame_events(detectorName, limit=maxEvents)
        return [e.to_dict() for e in events]
    
    @APIExport()
    def getLatestFrameEvent(self, detectorName: str) -> Dict[str, Any]:
        """
        Get the most recent frame event for a detector.
        
        Args:
            detectorName: Name of the detector
        
        Returns:
            Dictionary with latest frame event or empty dict if none
        """
        if self._metadata_hub is None:
            return {}
        
        events = self._metadata_hub.get_frame_events(detectorName, limit=1)
        if events:
            return events[0].to_dict()
        return {}
    
    # === Instrument Metadata API ===
    
    @APIExport()
    def getInstrumentInfo(self) -> Dict[str, Any]:
        """
        Get complete instrument metadata (microscope configuration).
        
        Returns:
            Dictionary with instrument info including UC2 components, filters, etc.
        """
        if self._instrument_manager is None:
            return {"error": "InstrumentMetadataManager not available"}
        
        return self._instrument_manager.instrument_info.to_dict()
    
    @APIExport()
    def getOMEInstrument(self) -> Dict[str, Any]:
        """
        Get instrument metadata formatted for OME-types.
        
        Returns:
            Dictionary compatible with ome_types.model.Instrument
        """
        if self._instrument_manager is None:
            return {"error": "InstrumentMetadataManager not available"}
        
        return self._instrument_manager.get_ome_instrument_dict()
    
    @APIExport()
    def getInstrumentComponents(self) -> List[Dict[str, Any]]:
        """
        Get list of UC2 optical components.
        
        Returns:
            List of component dictionaries
        """
        if self._instrument_manager is None:
            return []
        
        from dataclasses import asdict
        return [asdict(c) for c in self._instrument_manager.instrument_info.components]
    
    @APIExport()
    def getInstrumentFilters(self) -> List[Dict[str, Any]]:
        """
        Get list of optical filters.
        
        Returns:
            List of filter dictionaries
        """
        if self._instrument_manager is None:
            return []
        
        from dataclasses import asdict
        return [asdict(f) for f in self._instrument_manager.instrument_info.filters]
    
    @APIExport()
    def loadUC2OptiKitConfig(self, configPath: str) -> bool:
        """
        Load UC2 OptiKit configuration from a JSON file.
        
        Args:
            configPath: Path to UC2 OptiKit JSON configuration file
        
        Returns:
            True if loaded successfully
        """
        if self._instrument_manager is None:
            return False
        
        return self._instrument_manager.load_uc2_optikit_config(configPath)
    
    @APIExport()
    def setFirmwareVersion(self, version: str) -> None:
        """
        Set the firmware version string.
        
        Args:
            version: Firmware version string
        """
        if self._instrument_manager is not None:
            self._instrument_manager.set_firmware_version(version)
    
    @APIExport()
    def setTubeLens(self, focalLengthMm: float, magnification: float = 1.0) -> None:
        """
        Set tube lens parameters.
        
        Args:
            focalLengthMm: Focal length in millimeters
            magnification: Tube lens magnification factor
        """
        if self._instrument_manager is not None:
            self._instrument_manager.set_tube_lens(focalLengthMm, magnification)
    
    # === Shared Attributes Integration ===
    
    @APIExport()
    def getSharedAttributes(self) -> Dict[str, Any]:
        """
        Get the shared attributes from the communication channel.
        
        Returns:
            Dictionary of shared attributes
        """
        try:
            return json.loads(self._commChannel.sharedAttrs.getJSON())
        except Exception as e:
            self.__logger.error(f"Error getting shared attributes: {e}")
            return {}
    
    @APIExport()
    def getSharedAttributesFlat(self) -> Dict[str, Any]:
        """
        Get shared attributes in flat format (HDF5 style).
        
        Returns:
            Dictionary with ':' separated keys
        """
        try:
            return self._commChannel.sharedAttrs.getSharedAttributes()
        except Exception as e:
            self.__logger.error(f"Error getting shared attributes: {e}")
            return {}
    
    # === Session Discovery API (Multi-instance access) ===
    
    @APIExport()
    def listActiveSessions(self, basePath: str = None, limit: int = 50) -> List[Dict[str, Any]]:
        """
        List active and recent acquisition sessions.
        
        This enables multi-instance access: another ImSwitch can query
        available sessions and open them read-only.
        
        Args:
            basePath: Base directory to search (defaults to user data dir)
            limit: Maximum number of sessions to return
        
        Returns:
            List of session summaries with path, status, and basic metadata
        """
        from imswitch.imcontrol.model.io import SessionManager
        from imswitch.imcommon.model import dirtools
        
        if basePath is None:
            basePath = dirtools.UserFileDirs.Data
        
        try:
            manager = SessionManager(basePath)
            return manager.list_sessions(limit=limit)
        except Exception as e:
            self.__logger.error(f"Error listing sessions: {e}")
            return []
    
    @APIExport()
    def getSessionMetadata(self, sessionId: str, basePath: str = None) -> Dict[str, Any]:
        """
        Get full metadata for a specific session.
        
        Args:
            sessionId: Session UUID
            basePath: Base directory to search
        
        Returns:
            Full session metadata including hub snapshot and detector contexts
        """
        from imswitch.imcontrol.model.io import SessionManager
        from imswitch.imcommon.model import dirtools
        
        if basePath is None:
            basePath = dirtools.UserFileDirs.Data
        
        try:
            manager = SessionManager(basePath)
            return manager.get_session(sessionId) or {"error": f"Session {sessionId} not found"}
        except Exception as e:
            self.__logger.error(f"Error getting session: {e}")
            return {"error": str(e)}
    
    @APIExport()
    def getZarrStoreUrl(self, sessionPath: str) -> str:
        """
        Get the URL/path to a session's OME-Zarr store.
        
        Useful for opening the Zarr store in another ImSwitch instance
        or external viewers like napari.
        
        Args:
            sessionPath: Path to session directory
        
        Returns:
            Path to the OME-Zarr store
        """
        from pathlib import Path
        from imswitch.imcontrol.model.io.session import SessionManager
        
        zarr_path = Path(sessionPath) / SessionManager.ZARR_DIR
        if zarr_path.exists():
            return str(zarr_path)
        return ""
    
    @APIExport()
    def getCurrentSessionInfo(self) -> Dict[str, Any]:
        """
        Get info about the current acquisition session if one is active.
        
        Returns:
            Current session info or empty dict if no active session
        """
        # Check if RecordingManager has an active session
        try:
            rec_manager = self._master.recordingManager
            if rec_manager and hasattr(rec_manager, '_active_data_store'):
                store = rec_manager._active_data_store
                if store:
                    return {
                        'session_path': str(store.get_session_path()),
                        'zarr_path': str(store.get_zarr_path()),
                        'statistics': store.get_statistics(),
                    }
        except Exception as e:
            self.__logger.debug(f"Error getting current session: {e}")
        
        return {}


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
