"""
Unified I/O module for ImSwitch acquisition data.

This module centralizes ALL data storage functionality and replaces
the legacy RecordingManager/RecordingController pattern.

Supported Operations:
=====================
- Single image capture (snap): PNG, JPEG, TIFF with metadata
- Video recording: MP4 with start/stop
- Streaming recordings: OME-Zarr, OME-TIFF
- Stitched mosaics: 2D stitched OME-TIFF

Main Entry Point: RecordingService
===================================
The RecordingService is the central service that replaces RecordingManager.
It provides unified access to all recording operations with MetadataHub integration.

    from imswitch.imcontrol.model.io import RecordingService, SaveFormat, SaveMode
    
    # Create service
    service = RecordingService(detectors_manager)
    service.set_metadata_hub(metadata_hub)
    
    # Snap single images
    results = service.snap(format=SaveFormat.TIFF, save_mode=SaveMode.Disk)
    
    # Video recording
    service.start_video_recording('/path/to/video.mp4', fps=30)
    service.add_video_frame(frame)
    service.stop_video_recording()
    
    # Streaming to OME-Zarr
    service.start_streaming('/path/to/data.zarr', format=SaveFormat.OME_ZARR)
    service.write_streaming_frame(detector_name, frame, metadata)
    service.stop_streaming()

Alternative APIs:
=================

1. SnapService - Lightweight snap-only service
       from imswitch.imcontrol.model.io import get_snap_service
       snap_service = get_snap_service()
       results = snap_service.snap(savepath='/data/snap', format='tiff')

2. AcquisitionDataStore - Low-level streaming with full control
       from imswitch.imcontrol.model.io import AcquisitionDataStore, SessionInfo
       store = AcquisitionDataStore(session, hub_snapshot, write_zarr=True)
       store.open(detector_contexts)
       store.write_frame(detector_name, frame, frame_event)
       store.close()

3. StitchedTiffWriter - For mosaic/tile acquisitions
       from imswitch.imcontrol.model.io import StitchedTiffWriter, MosaicConfig
       config = MosaicConfig(nx=5, ny=5, tile_width=2048, tile_height=2048)
       writer = StitchedTiffWriter('/path/to/mosaic.ome.tiff', config)
       writer.open()
       writer.add_tile(tile_info, image)
       writer.close()

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
from .snap_service import (
    SnapService,
    SnapResult as SnapServiceResult,
    get_snap_service,
    shutdown_snap_service,
)
from .recording_service import (
    RecordingService,
    SaveMode,
    SaveFormat,
    RecMode,
    SnapResult,
    RecordingStatus,
    MP4Writer,
    get_recording_service,
    create_recording_service,
    shutdown_recording_service,
)
from .stitched_tiff_writer import (
    StitchedTiffWriter,
    StreamingStitchedTiffWriter,
    MosaicConfig,
    TileInfo,
    create_stitched_writer,
)

# OME Writers (migrated from experiment_controller)
from .ome_writers import (
    OMEWriter,
    OMEWriterConfig,
    OMEFileStorePaths,
    OmeTiffStitcher,
    SingleTiffWriter,
    MinimalMetadata,
    MinimalZarrDataSource,
    # OMERO uploader
    OMEROUploader,
    OMEROConnectionParams,
    TileMetadata,
    is_omero_available,
)

__all__ = [
    # === Main Entry Point ===
    'RecordingService',
    'get_recording_service',
    'create_recording_service',
    'shutdown_recording_service',
    
    # === Enums (shared) ===
    'SaveMode',
    'SaveFormat',
    'RecMode',
    
    # === Result Types ===
    'SnapResult',
    'RecordingStatus',
    
    # === Specialized Writers ===
    'MP4Writer',
    'StitchedTiffWriter',
    'StreamingStitchedTiffWriter',
    'MosaicConfig',
    'TileInfo',
    'create_stitched_writer',
    
    # === Core data store (low-level) ===
    'AcquisitionDataStore',
    'create_data_store',
    
    # === Session management ===
    'SessionInfo',
    'SessionManager',
    'SessionFileLock',
    
    # === Base Writers ===
    'WriterBase',
    'OMEZarrWriter',
    'OMETiffWriter',
    'get_writer',
    'convert_detector_context',
    'convert_frame_event',
    'list_available_writers',
    
    # === Adapters for compatibility ===
    'DataStoreAdapter',
    'StreamingDataStoreAdapter',
    
    # === Snap service (lightweight alternative) ===
    'SnapService',
    'get_snap_service',
    'shutdown_snap_service',
    
    # === OME Writers (migrated from experiment_controller) ===
    'OMEWriter',
    'OMEWriterConfig',
    'OMEFileStorePaths',
    'OmeTiffStitcher',
    'SingleTiffWriter',
    'MinimalMetadata',
    'MinimalZarrDataSource',
    # OMERO uploader
    'OMEROUploader',
    'OMEROConnectionParams',
    'TileMetadata',
    'is_omero_available',
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
