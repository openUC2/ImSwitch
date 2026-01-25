"""
Unified I/O module for ImSwitch acquisition data.

Provides a centralized AcquisitionDataStore interface that can be used by
both RecordingManager/Controller and ExperimentController for consistent
data writing with OME-compliant metadata.

Main components:
- AcquisitionDataStore: High-level service coordinating all writers
- Writers: OME-Zarr and OME-TIFF writers with proper metadata
- SessionManager: Session directory layout and multi-instance access
- Adapters: Compatibility adapters for existing RecordingManager

Usage:
    from imswitch.imcontrol.model.io import AcquisitionDataStore, SessionInfo
    
    # Create session
    session = SessionInfo(
        session_id=str(uuid.uuid4()),
        base_path="/path/to/data",
        project="MyProject"
    )
    
    # Open data store
    store = AcquisitionDataStore(session, hub_snapshot, write_zarr=True, write_tiff=True)
    store.open(detector_contexts)
    
    # Write frames with aligned metadata
    store.write_frame(detector_name, frame, frame_event)
    
    # Close session
    store.close()
    
For RecordingManager compatibility:
    from imswitch.imcontrol.model.io import DataStoreAdapter
    
    adapter = DataStoreAdapter(filepath, detectorsManager, metadata_hub=hub)
    adapter.snap(images, attrs)
"""

from .data_store import AcquisitionDataStore, create_data_store
from .session import SessionInfo, SessionManager, SessionFileLock
from .writers import (
    WriterBase,
    OMEZarrWriter,
    OMETiffWriter,
    get_writer,
    convert_detector_context,
    convert_frame_event,
    list_available_writers,
)
from .adapters import DataStoreAdapter, StreamingDataStoreAdapter

__all__ = [
    # Core data store
    'AcquisitionDataStore',
    'create_data_store',
    
    # Session management
    'SessionInfo',
    'SessionManager',
    'SessionFileLock',
    
    # Writers
    'WriterBase',
    'OMEZarrWriter',
    'OMETiffWriter',
    'get_writer',
    'convert_detector_context',
    'convert_frame_event',
    'list_available_writers',
    
    # Adapters for compatibility
    'DataStoreAdapter',
    'StreamingDataStoreAdapter',
]


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
