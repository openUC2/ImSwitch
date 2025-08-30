"""
OMERO uploader for streaming tiles to OMERO server in parallel to acquisition.

This module provides a thread-safe uploader that consumes tiles from a bounded queue
and writes them to OMERO using either tile writes (pyramid backend) or full-plane writes
(ROMIO backend), with automatic backend detection.
"""

import time
import queue
import threading
import tempfile
import pickle
import os
from typing import Dict, Any, Optional, Tuple, List, Union
import numpy as np
from dataclasses import dataclass
import logging

# OMERO imports with graceful fallback
try:
    import omero
    from omero.gateway import BlitzGateway
    from omero.rtypes import rstring
    from omero.model import (
        DatasetI, ImageI, DatasetImageLinkI, LengthI,
        MapAnnotationI, NamedValue, ImageAnnotationLinkI, CommentAnnotationI,
    )
    from omero.model.enums import UnitsLength
    OMERO_AVAILABLE = True
except ImportError:
    OMERO_AVAILABLE = False
    # Mock classes for when OMERO is not available
    class BlitzGateway:
        pass
    class omero:
        class client:
            pass


@dataclass
class OMEROConnectionParams:
    """OMERO connection parameters."""
    host: str
    port: int = 4064
    username: str = ""
    password: str = ""
    group_id: int = -1
    project_id: int = -1
    dataset_id: int = -1
    connection_timeout: int = 30
    upload_timeout: int = 300


@dataclass
class TileMetadata:
    """Metadata for a single tile."""
    ix: int  # X grid index
    iy: int  # Y grid index
    z: int = 0  # Z plane index
    c: int = 0  # Channel index
    t: int = 0  # Time point index
    tile_data: np.ndarray = None  # Tile pixel data
    experiment_id: str = ""  # Local experiment identifier
    pixel_size_um: float = 1.0  # Pixel size in micrometers


class DiskSpilloverQueue:
    """Bounded queue with disk spillover for robust tile handling."""
    
    def __init__(self, max_memory_items: int = 100, spill_dir: Optional[str] = None):
        self.max_memory_items = max_memory_items
        self.memory_queue = queue.Queue(maxsize=max_memory_items)
        self.spill_dir = spill_dir or tempfile.mkdtemp(prefix="imswitch_omero_")
        self.spill_counter = 0
        self.spilled_files = queue.Queue(maxsize=max_memory_items)  # Limit number of spilled files tracked
        self._lock = threading.Lock()
        self._closed = False
        
        # Ensure spill directory exists
        os.makedirs(self.spill_dir, exist_ok=True)
        
    def put(self, item: TileMetadata) -> bool:
        """Put item in queue, spilling to disk if memory queue is full."""
        if self._closed:
            return False
            
        try:
            # Try memory queue first
            self.memory_queue.put_nowait(item)
            return True
        except queue.Full:
            # Spill to disk
            with self._lock:
                if self._closed:
                    return False
                spill_file = os.path.join(self.spill_dir, f"tile_{self.spill_counter}.pkl")
                self.spill_counter += 1
                
                try:
                    with open(spill_file, 'wb') as f:
                        pickle.dump(item, f)
                    self.spilled_files.put(spill_file)
                    return True
                except Exception as e:
                    logging.error(f"Failed to spill tile to disk: {e}")
                    return False
    
    def get(self, timeout: Optional[float] = None) -> Optional[TileMetadata]:
        """Get item from queue, checking memory first then disk."""
        if self._closed and self.empty():
            return None
            
        start_time = time.time() if timeout else None
        
        while True:
            try:
                # Try memory queue first
                return self.memory_queue.get_nowait()
            except queue.Empty:
                pass
                
            # Try spilled files
            with self._lock:
                try:
                    spill_file = self.spilled_files.get_nowait()
                    with open(spill_file, 'rb') as f:
                        item = pickle.load(f)
                    os.remove(spill_file)  # Clean up
                    return item
                except queue.Empty:
                    pass
                except Exception as e:
                    logging.error(f"Failed to read spilled tile: {e}")
                    
            # Check timeout
            if timeout is not None:
                elapsed = time.time() - start_time
                if elapsed >= timeout:
                    return None
                    
            # If closed and empty, return None
            if self._closed and self.empty():
                return None
                
            # Small delay to prevent busy waiting
            time.sleep(0.001)
            
            # For non-blocking calls without timeout, return None if nothing available
            if timeout is None:
                return None
    
    def empty(self) -> bool:
        """Check if queue is empty."""
        return self.memory_queue.empty() and self.spilled_files.empty()
    
    def close(self):
        """Close queue and clean up spilled files."""
        self._closed = True
        with self._lock:
            # Clean up any remaining spilled files
            while not self.spilled_files.empty():
                try:
                    spill_file = self.spilled_files.get_nowait()
                    if os.path.exists(spill_file):
                        os.remove(spill_file)
                except:
                    pass
            # Remove spill directory if empty
            try:
                os.rmdir(self.spill_dir)
            except:
                pass

    def len(self) -> int:
        """Get the current length of the queue."""
        return self.memory_queue.qsize() + self.spilled_files.qsize()

class OMEROUploader:
    """Thread-safe OMERO uploader for streaming tiles."""
    
    def __init__(self, connection_params: OMEROConnectionParams, 
                 mosaic_config: Dict[str, Any],
                 queue_size: int = 100):
        self.connection_params = connection_params
        self.mosaic_config = mosaic_config
        self.tile_queue = DiskSpilloverQueue(max_memory_items=queue_size)
        self.uploader_thread = None
        self.running = False
        self.connection = None
        self.client = None
        self.store = None
        self._connection_cleaned = False
        self._finalized = False
        self._stop_requested = False
        self._worker_finished = threading.Event()
        
        # Mosaic configuration
        self.nx = mosaic_config.get('nx', 1)
        self.ny = mosaic_config.get('ny', 1)
        self.tile_w = mosaic_config.get('tile_w', 512)
        self.tile_h = mosaic_config.get('tile_h', 512)
        self.size_z = mosaic_config.get('size_z', 1)
        self.size_c = mosaic_config.get('size_c', 1)
        self.size_t = mosaic_config.get('size_t', 1)
        self.pixel_type = mosaic_config.get('pixel_type', 'uint16')
        self.dataset_name = mosaic_config.get('dataset_name', 'ImSwitch-StageScan')
        self.image_name = mosaic_config.get('image_name', 'mosaic')
        
        # OMERO objects
        self.dataset_id = None
        self.image_id = None
        self.pixels_id = None
        self.use_tile_writes = True  # Will be determined by backend detection
        
        # Row buffering for ROMIO backend (requires full row writes)
        self.row_buffers = {}  # Key: (iy, z, c, t), Value: dict of tiles indexed by ix
        self.row_buffer_lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
        
        # Log mosaic configuration for debugging
        self.logger.info(f"OMERO uploader initialized with mosaic config: "
                        f"{self.nx}x{self.ny} tiles, {self.tile_w}x{self.tile_h} pixels per tile, "
                        f"Z={self.size_z}, C={self.size_c}, T={self.size_t}, dtype={self.pixel_type}")
        
    def start(self) -> bool:
        """Start the uploader thread."""
        self._finalized = False
        if not OMERO_AVAILABLE:
            self.logger.warning("OMERO Python not available, uploader cannot start")
            return False
        

        if self.running:
            return True
            
        # Initialize OMERO connection outside the thread to block the outer runtime
        if not self._connect_to_omero():
            self.logger.error("Failed to connect to OMERO, uploader stopping")
            return
                
        if self.connection is None: 
            self.logger.warning("OMERO Python not available, uploader cannot start")
            return
    
        # Create dataset and image
        if not self._setup_omero_objects():
            self.logger.error("Failed to setup OMERO objects, uploader stopping") 
            return
                

        self.running = True
        self.uploader_thread = threading.Thread(target=self._uploader_worker, daemon=True)
        self.uploader_thread.start()
        return True
    
    def stop(self):
        """Stop the uploader thread and clean up."""
        self.logger.info("Stopping OMERO uploader...")
        
        # Signal the worker to stop accepting new tiles
        self._stop_requested = True
        self.running = False
        
        # Wait for worker to finish processing remaining tiles
        if self.uploader_thread:
            self.logger.info("Waiting for uploader thread to complete...")
            self._worker_finished.wait(timeout=30)  # Wait up to 30 seconds
            self.uploader_thread.join(timeout=5)  # Additional join with short timeout
        
        # Finalize any remaining uploads after worker finishes
        try:
            self.finalize()
        except Exception as e:
            self.logger.warning(f"Error during finalization: {e}")
            
        self._cleanup_row_buffers()
        self._cleanup_connection()
        self.tile_queue.close() 
    
    def enqueue_tile(self, tile_metadata: TileMetadata) -> bool:
        """Enqueue a tile for upload."""
        if not self.running or self._stop_requested:
            self.logger.warning(f"Cannot enqueue tile ({tile_metadata.ix}, {tile_metadata.iy}) - uploader stopping or not running")
            return False
        
        self.logger.debug(f"Enqueueing tile ({tile_metadata.ix}, {tile_metadata.iy}) for upload (mosaic size: {self.nx}x{self.ny})")
        success = self.tile_queue.put(tile_metadata)
        if not success:
            self.logger.warning(f"Failed to enqueue tile ({tile_metadata.ix}, {tile_metadata.iy}) - queue full or closed")
        return success
    
    def signal_completion(self):
        """Signal that no more tiles will be enqueued."""
        self.logger.info("Signaling completion - no more tiles will be enqueued")
        self._stop_requested = True
    
    def _uploader_worker(self):
        """Main uploader thread worker."""
        self.logger.info("OMERO uploader thread started")
        
        try:
            # Process tiles from queue until stop requested AND queue is empty
            while True:
                # Get tile with short timeout to allow checking stop condition
                tile = self.tile_queue.get(timeout=0.1)
                if tile is None:
                    # No tile available, check if we should continue waiting
                    if self._stop_requested and self.tile_queue.empty():
                        self.logger.info("Stop requested and queue empty, worker finishing")
                        break
                    # If stop not requested or queue not empty, continue polling
                    continue
                    
                try:
                    self._upload_tile(tile)
                except Exception as e:
                    self.logger.error(f"Failed to upload tile ({tile.ix}, {tile.iy}): {e}")
                    # TODO: Implement fallback to local storage
                    
        except Exception as e:
            self.logger.error(f"OMERO uploader worker error: {e}")
        finally:
            # Signal that worker is finished
            self._worker_finished.set()
            self.logger.info("OMERO uploader thread stopped")
    
    def _connect_to_omero(self) -> bool:
        """Establish connection to OMERO server."""
        try:
            self.client = omero.client(
                self.connection_params.host, 
                self.connection_params.port
            )
            #self.client.setDefaultContextTimeout(self.connection_params.connection_timeout * 1000)
            
            session = self.client.createSession(
                self.connection_params.username,
                self.connection_params.password
            )
            # session.ice_timeout(self.connection_params.connection_timeout * 1000) # TODO: Not sure what'S the equivalent for the timeout, setDefaultContextTimeout doesn't exist
            self.connection = BlitzGateway(client_obj=self.client)
            
            # Test connection
            user = self.connection.getUser()
            self.logger.info(f"Connected to OMERO as {user.getName()}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to OMERO: {e}")
            return False
    
    def _setup_omero_objects(self) -> bool:
        """Create dataset and image in OMERO."""
        try:
            # Create or get dataset
            if self.connection_params.dataset_id > 0:
                # Check if existing dataset exists
                dataset_obj = self.connection.getObject("Dataset", self.connection_params.dataset_id)
                if dataset_obj is not None:
                    # Use existing dataset
                    self.dataset_id = self.connection_params.dataset_id
                    self.logger.info(f"Using existing dataset {self.dataset_id}: {dataset_obj.getName()}")
                else:
                    # Dataset doesn't exist, create new one
                    self.logger.warning(f"Dataset {self.connection_params.dataset_id} not found, creating new dataset")
                    ds = DatasetI()
                    ds.setName(rstring(self.dataset_name))
                    ds.setDescription(rstring("ImSwitch mosaic acquisition"))
                    ds = self.connection.getUpdateService().saveAndReturnObject(ds, self.connection.SERVICE_OPTS)
                    self.dataset_id = ds.getId().getValue()
            else:
                # Create new dataset
                ds = DatasetI()
                ds.setName(rstring(self.dataset_name))
                ds.setDescription(rstring("ImSwitch mosaic acquisition"))
                ds = self.connection.getUpdateService().saveAndReturnObject(ds, self.connection.SERVICE_OPTS)
                self.dataset_id = ds.getId().getValue()
            
            # Create image
            size_x = self.nx * self.tile_w
            size_y = self.ny * self.tile_h
            
            pixels_service = self.connection.getPixelsService()
            query_service = self.connection.getQueryService()
            pixels_type = query_service.findByString("PixelsType", "value", self.pixel_type)
            
            self.image_id = pixels_service.createImage(
                size_x, size_y, self.size_z, self.size_t, 
                list(range(self.size_c)), pixels_type,
                self.image_name, "ImSwitch streamed mosaic"
            ).getValue()
            
            # Link image to dataset
            link = DatasetImageLinkI()
            link.setParent(DatasetI(self.dataset_id, False))
            link.setChild(ImageI(self.image_id, False))
            self.connection.getUpdateService().saveAndReturnObject(link, self.connection.SERVICE_OPTS)
            
            # Set pixel size
            img = self.connection.getObject("Image", self.image_id)
            pixels = query_service.get("Pixels", img.getPixelsId())
            self.pixels_id = pixels.getId().getValue()
            
            pixel_size = self.mosaic_config.get('pixel_size_um', 1.0)
            pixels.setPhysicalSizeX(LengthI(pixel_size, UnitsLength.MICROMETER))
            pixels.setPhysicalSizeY(LengthI(pixel_size, UnitsLength.MICROMETER))
            self.connection.getUpdateService().saveAndReturnObject(pixels, self.connection.SERVICE_OPTS)
            
            # Setup raw pixels store and detect backend
            self._setup_raw_pixels_store()
            
            self.logger.info(f"Created OMERO image {self.image_id} in dataset {self.dataset_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup OMERO objects: {e}")
            return False
    
    def _setup_raw_pixels_store(self):
        """Setup raw pixels store and detect backend capabilities."""
        try:
            self.store = self.connection.createRawPixelsStore()
            self.store.setPixelsId(self.pixels_id, True)
            
            # Detect backend type
            srv_tw, srv_th = self.store.getTileSize()
            srv_tw, srv_th = int(srv_tw), int(srv_th)
            
            size_x = self.nx * self.tile_w
            
            # Heuristic: if server tile width >= image width, use row-based writes
            self.use_tile_writes = (srv_tw > 0 and srv_tw < size_x)
            
            backend_type = "tile writes" if self.use_tile_writes else "row-stripe writes"
            self.logger.info(f"OMERO backend detected: {backend_type} (server tile size: {srv_tw}x{srv_th})")
            
        except Exception as e:
            self.logger.error(f"Failed to setup raw pixels store: {e}")
            raise
    
    def _upload_tile(self, tile: TileMetadata):
        """Upload a single tile to OMERO."""
        if self.use_tile_writes:
            self._upload_tile_tiled(tile)
        else:
            self._upload_tile_row_stripe(tile)
    
    def _upload_tile_tiled(self, tile: TileMetadata):
        """Upload tile using tile-based writes (pyramid backend)."""
        try:
            srv_tw, srv_th = self.store.getTileSize()
            srv_tw, srv_th = int(srv_tw), int(srv_th)
            
            x0 = tile.ix * self.tile_w
            y0 = tile.iy * self.tile_h
            
            # Sub-tile to server tile size
            for off_y in range(0, self.tile_h, srv_th):
                for off_x in range(0, self.tile_w, srv_tw):
                    sub_tile = tile.tile_data[off_y:off_y+srv_th, off_x:off_x+srv_tw]
                    h, w = sub_tile.shape[0], sub_tile.shape[1]
                    if h > 0 and w > 0:
                        self.logger.debug(f"Uploading sub-tile {x0 + off_x},{y0 + off_y} ({w}x{h})")
                        buf = np.ascontiguousarray(sub_tile).tobytes()
                        self.store.setTile(buf, tile.z, tile.c, tile.t, 
                                         x0 + off_x, y0 + off_y, w, h)
                                         
        except Exception as e:
            self.logger.error(f"Failed to upload tile {tile.ix},{tile.iy} using tile writes: {e}")
            raise
    
    def _upload_tile_row_stripe(self, tile: TileMetadata):
        """Upload tile using row-stripe writes (ROMIO backend)."""
        # ROMIO backend requires full row writes, so we need to buffer tiles
        # by row until we have a complete row, then merge and upload
        try:
            row_key = (tile.iy, tile.z, tile.c, tile.t)
            self.logger.debug(f"Processing tile ({tile.ix}, {tile.iy}) for row buffer, key: {row_key}")
            
            with self.row_buffer_lock:
                # Initialize row buffer if needed
                if row_key not in self.row_buffers:
                    self.row_buffers[row_key] = {}
                    self.logger.debug(f"Created new row buffer for row {tile.iy}")
                
                # Add tile to row buffer
                self.row_buffers[row_key][tile.ix] = tile
                current_tiles_in_row = len(self.row_buffers[row_key])
                self.logger.debug(f"Added tile {tile.ix} to row {tile.iy}, now has {current_tiles_in_row}/{self.nx} tiles")
                
                # Check if row is complete
                if len(self.row_buffers[row_key]) == self.nx:
                    # Row is complete, merge tiles and upload
                    self._upload_complete_row(row_key, self.row_buffers[row_key])
                    # Clean up completed row
                    del self.row_buffers[row_key]
                    self.logger.debug(f"Cleaned up completed row buffer for row {tile.iy}")
                else:
                    self.logger.debug(f"Row {tile.iy} still incomplete ({current_tiles_in_row}/{self.nx} tiles), waiting for more tiles...")
                    
        except Exception as e:
            self.logger.error(f"Failed to buffer tile {tile.ix},{tile.iy} for row upload: {e}")
            raise
    
    def _upload_complete_row(self, row_key: Tuple[int, int, int, int], row_tiles: Dict[int, TileMetadata]):
        """Upload a complete row of tiles to OMERO."""
        iy, z, c, t = row_key
        
        try:
            # Sort tiles by ix to ensure proper order
            sorted_tiles = [row_tiles[ix] for ix in sorted(row_tiles.keys())]
            
            # Merge tiles horizontally to create full row
            row_data = np.concatenate([tile.tile_data for tile in sorted_tiles], axis=1)
            
            # Calculate row position
            y_start = iy * self.tile_h
            row_height = self.tile_h
            row_width = self.nx * self.tile_w
            
            self.logger.debug(f"Uploading complete row {iy} at y={y_start}, size={row_width}x{row_height}")
            
            # Upload row stripe by stripe (each row of pixels in the tile height)
            
            buf = np.ascontiguousarray(row_data).tobytes()
            self.store.setTile(buf, z, c, t, 
                                     0, y_start, row_width, 1)

        except Exception as e:
            self.logger.error(f"Failed to upload complete row {iy}: {e}")
            raise
    
    def _cleanup_row_buffers(self):
        """Clean up any remaining row buffers."""
        with self.row_buffer_lock:
            if self.row_buffers:
                self.logger.warning(f"Cleaning up {len(self.row_buffers)} incomplete row buffers")
                self.row_buffers.clear()
    
    def finalize(self):
        """Finalize the OMERO upload."""
        # Check if already finalized
        if hasattr(self, '_finalized') and self._finalized:
            return
            
        self._finalized = True
        
        # Upload any remaining incomplete rows before finalizing
        self._upload_remaining_rows()
        
        if self.store:
            try:
                self.store.save()
                self.logger.info("OMERO upload finalized successfully")
            except Exception as e:
                self.logger.error(f"Failed to finalize OMERO upload: {e}")
    
    def _upload_remaining_rows(self):
        """Upload any remaining incomplete rows (for edge cases where not all tiles arrive)."""
        self.logger.info("Checking for remaining rows to upload...")
        
        # Try to acquire the lock with timeout to avoid deadlock
        lock_acquired = self.row_buffer_lock.acquire(timeout=5.0)
        if not lock_acquired:
            self.logger.error("Could not acquire row buffer lock for uploading remaining rows")
            return
            
        try:
            if self.row_buffers:
                self.logger.warning(f"Uploading {len(self.row_buffers)} incomplete rows during finalization")
                
                for row_key, row_tiles in list(self.row_buffers.items()):
                    iy, z, c, t = row_key
                    tiles_count = len(row_tiles) if row_tiles else 0
                    self.logger.info(f"Processing incomplete row {iy} with {tiles_count}/{self.nx} tiles")
                    
                    if row_tiles:  # Only upload if we have at least some tiles
                        try:
                            # Create padded row if incomplete
                            if len(row_tiles) < self.nx:
                                self.logger.warning(f"Row {iy} incomplete ({tiles_count}/{self.nx} tiles), creating padded row")
                                # Fill missing tiles with zeros or duplicate last available tile
                                padded_tiles = {}
                                for ix in range(self.nx):
                                    if ix in row_tiles:
                                        padded_tiles[ix] = row_tiles[ix]
                                    else:
                                        # Create empty tile with same properties as existing tiles
                                        if row_tiles:
                                            sample_tile = next(iter(row_tiles.values()))
                                            empty_tile = TileMetadata(
                                                ix=ix, iy=iy, z=z, c=c, t=t,
                                                tile_data=np.zeros_like(sample_tile.tile_data),
                                                experiment_id=sample_tile.experiment_id,
                                                pixel_size_um=sample_tile.pixel_size_um
                                            )
                                            padded_tiles[ix] = empty_tile
                                row_tiles = padded_tiles
                            
                            self._upload_complete_row(row_key, row_tiles)
                            self.logger.info(f"Successfully uploaded incomplete row {iy}")
                        except Exception as e:
                            self.logger.error(f"Failed to upload incomplete row {row_key}: {e}")
                    else:
                        self.logger.warning(f"Skipping empty row buffer for row {iy}")
                
                self.row_buffers.clear()
            else:
                self.logger.info("No incomplete rows found during finalization")
        finally:
            self.row_buffer_lock.release()
    
    def _cleanup_connection(self):
        """Clean up OMERO connection."""
        # Use a flag to ensure cleanup only happens once
        if hasattr(self, '_connection_cleaned') and self._connection_cleaned:
            return
            
        self._connection_cleaned = True
        
        # Close store first
        if self.store:
            try:
                self.store.close()
                self.logger.debug("OMERO store closed")
            except Exception as e:
                self.logger.warning(f"Error closing OMERO store: {e}")
            self.store = None
            
        # Close connection 
        if self.connection:
            try:
                self.connection.close()
                self.logger.debug("OMERO connection closed")
            except Exception as e:
                self.logger.warning(f"Error closing OMERO connection: {e}")
            self.connection = None
            
        # Close client session last
        if self.client:
            try:
                self.client.closeSession()
                self.logger.debug("OMERO client session closed")
            except Exception as e:
                self.logger.warning(f"Error closing OMERO client session: {e}")
            self.client = None