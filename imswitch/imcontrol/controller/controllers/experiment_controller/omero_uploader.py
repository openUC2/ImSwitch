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
        self.spilled_files = queue.Queue()
        self._lock = threading.Lock()
        self._closed = False
        
        # Ensure spill directory exists
        os.makedirs(self.spill_dir, exist_ok=True)
        
    def put(self, item: TileMetadata, timeout: Optional[float] = None) -> bool:
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
            
        try:
            # Try memory queue first
            return self.memory_queue.get_nowait()
        except queue.Empty:
            # Try spilled files
            with self._lock:
                try:
                    spill_file = self.spilled_files.get_nowait()
                    with open(spill_file, 'rb') as f:
                        item = pickle.load(f)
                    os.remove(spill_file)  # Clean up
                    return item
                except queue.Empty:
                    return None
                except Exception as e:
                    logging.error(f"Failed to read spilled tile: {e}")
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
        
        self.logger = logging.getLogger(__name__)
        
    def start(self) -> bool:
        """Start the uploader thread."""
        if not OMERO_AVAILABLE:
            self.logger.warning("OMERO Python not available, uploader cannot start")
            return False
            
        if self.running:
            return True
            
        self.running = True
        self.uploader_thread = threading.Thread(target=self._uploader_worker, daemon=True)
        self.uploader_thread.start()
        return True
    
    def stop(self):
        """Stop the uploader thread and clean up."""
        self.running = False
        if self.uploader_thread:
            self.uploader_thread.join(timeout=10)
        self.tile_queue.close()
        self._cleanup_connection()
    
    def enqueue_tile(self, tile_metadata: TileMetadata) -> bool:
        """Enqueue a tile for upload."""
        if not self.running:
            return False
        return self.tile_queue.put(tile_metadata)
    
    def _uploader_worker(self):
        """Main uploader thread worker."""
        self.logger.info("OMERO uploader thread started")
        
        try:
            # Initialize OMERO connection
            if not self._connect_to_omero():
                self.logger.error("Failed to connect to OMERO, uploader stopping")
                return
                
            # Create dataset and image
            if not self._setup_omero_objects():
                self.logger.error("Failed to setup OMERO objects, uploader stopping") 
                return
                
            # Process tiles from queue
            while self.running:
                tile = self.tile_queue.get(timeout=1.0)
                if tile is None:
                    continue
                    
                try:
                    self._upload_tile(tile)
                except Exception as e:
                    self.logger.error(f"Failed to upload tile ({tile.ix}, {tile.iy}): {e}")
                    # TODO: Implement fallback to local storage
                    
        except Exception as e:
            self.logger.error(f"OMERO uploader worker error: {e}")
        finally:
            self._cleanup_connection()
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
                # Use existing dataset
                self.dataset_id = self.connection_params.dataset_id
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
            
            '''TODO: The above line fails with : 
            
            WARNING:omero.gateway:ValidationException on <class 'omero.gateway.OmeroGatewaySafeCallWrapper'> to <eb90adcb-6699-42a3-bb22-39a3b744acf2omero.api.IUpdate> saveAndReturnObject((object #0 (::omero::model::DatasetImageLink)
{
    _id = <nil>
    _details = object #1 (::omero::model::Details)
    {
        _owner = <nil>
        _group = <nil>
        _creationEvent = <nil>
        _updateEvent = <nil>
        _permissions = <nil>
        _externalInfo = <nil>
        _call = {}
        _event = <nil>
    }
    _loaded = True
    _version = <nil>
    _parent = object #2 (::omero::model::Dataset)
    {
        _id = object #3 (::omero::RLong)
        {
            _val = 200
        }
        _details = <nil>
        _loaded = False
        _version = <nil>
        _projectLinksSeq = {}
        _projectLinksLoaded = False
        _projectLinksCountPerOwner = {}
        _imageLinksSeq = {}
        _imageLinksLoaded = False
        _imageLinksCountPerOwner = {}
        _annotationLinksSeq = {}
        _annotationLinksLoaded = False
        _annotationLinksCountPerOwner = {}
        _name = <nil>
        _description = <nil>
    }
    _child = object #4 (::omero::model::Image)
    {
        _id = object #5 (::omero::RLong)
        {
            _val = 54
        }
        _details = <nil>
        _loaded = False
        _version = <nil>
        _series = <nil>
        _acquisitionDate = <nil>
        _archived = <nil>
        _partial = <nil>
        _format = <nil>
        _imagingEnvironment = <nil>
        _objectiveSettings = <nil>
        _instrument = <nil>
        _stageLabel = <nil>
        _experiment = <nil>
        _pixelsSeq = {}
        _pixelsLoaded = False
        _wellSamplesSeq = {}
        _wellSamplesLoaded = False
        _roisSeq = {}
        _roisLoaded = False
        _datasetLinksSeq = {}
        _datasetLinksLoaded = False
        _datasetLinksCountPerOwner = {}
        _folderLinksSeq = {}
        _folderLinksLoaded = False
        _folderLinksCountPerOwner = {}
        _fileset = <nil>
        _annotationLinksSeq = {}
        _annotationLinksLoaded = False
        _annotationLinksCountPerOwner = {}
        _name = <nil>
        _description = <nil>
    }
}, <ServiceOptsDict: {'omero.client.uuid': 'eb90adcb-6699-42a3-bb22-39a3b744acf2', 'omero.session.uuid': '9823d4ce-95e1-425d-9f77-65fb96ec9642'}>), {})
Traceback (most recent call last):
  File "/Users/bene/mambaforge/envs/microeye/lib/python3.10/site-packages/omero/gateway/__init__.py", line 4870, in __call__
    return self.f(*args, **kwargs)
  File "/Users/bene/mambaforge/envs/microeye/lib/python3.10/site-packages/omero_api_IUpdate_ice.py", line 163, in saveAndReturnObject
    return _M_omero.api.IUpdate._op_saveAndReturnObject.invoke(self, ((obj, ), _ctx))
omero.ValidationException: exception ::omero::ValidationException
{
    serverStackTrace = ome.conditions.ValidationException: No row with the given identifier exists: [ome.model.containers.Dataset#200]; nested exception is org.hibernate.ObjectNotFoundException: No row with the given identifier exists: [ome.model.containers.Dataset#200]
        at org.springframework.orm.hibernate3.SessionFactoryUtils.convertHibernateAccessException(SessionFactoryUtils.java:693)
        at org.springframework.orm.hibernate3.HibernateAccessor.convertHibernateAccessException(HibernateAccessor.java:416)
        at org.springframework.orm.hibernate3.HibernateInterceptor.invoke(HibernateInterceptor.java:125)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:179)
        at org.springframework.transaction.interceptor.TransactionInterceptor$1.proceedWithInvocation(TransactionInterceptor.java:99)
        at org.springframework.transaction.interceptor.TransactionAspectSupport.invokeWithinTransaction(TransactionAspectSupport.java:283)
        at org.springframework.transaction.interceptor.TransactionInterceptor.invoke(TransactionInterceptor.java:96)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:179)
        at ome.tools.hibernate.ProxyCleanupFilter$Interceptor.invoke(ProxyCleanupFilter.java:249)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:179)
        at ome.services.util.ServiceHandler.invoke(ServiceHandler.java:121)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:179)
        at org.springframework.aop.framework.JdkDynamicAopProxy.invoke(JdkDynamicAopProxy.java:213)
        at com.sun.proxy.$Proxy100.saveAndReturnObject(Unknown Source)
        at jdk.internal.reflect.GeneratedMethodAccessor420.invoke(Unknown Source)
        at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.base/java.lang.reflect.Method.invoke(Method.java:566)
        at org.springframework.aop.support.AopUtils.invokeJoinpointUsingReflection(AopUtils.java:333)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.invokeJoinpoint(ReflectiveMethodInvocation.java:190)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:157)
        at ome.security.basic.BasicSecurityWiring.invoke(BasicSecurityWiring.java:93)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:179)
        at ome.services.blitz.fire.AopContextInitializer.invoke(AopContextInitializer.java:43)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:179)
        at org.springframework.aop.framework.JdkDynamicAopProxy.invoke(JdkDynamicAopProxy.java:213)
        at com.sun.proxy.$Proxy100.saveAndReturnObject(Unknown Source)
        at jdk.internal.reflect.GeneratedMethodAccessor444.invoke(Unknown Source)
        at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.base/java.lang.reflect.Method.invoke(Method.java:566)
        at ome.services.blitz.util.IceMethodInvoker.invoke(IceMethodInvoker.java:172)
        at ome.services.throttling.Callback.run(Callback.java:56)
        at ome.services.throttling.InThreadThrottlingStrategy.callInvokerOnRawArgs(InThreadThrottlingStrategy.java:56)
        at ome.services.blitz.impl.AbstractAmdServant.callInvokerOnRawArgs(AbstractAmdServant.java:140)
        at ome.services.blitz.impl.UpdateI.saveAndReturnObject_async(UpdateI.java:62)
        at jdk.internal.reflect.GeneratedMethodAccessor443.invoke(Unknown Source)
        at java.base/jdk.internal.reflect.DelegatingMethodAccessorImpl.invoke(DelegatingMethodAccessorImpl.java:43)
        at java.base/java.lang.reflect.Method.invoke(Method.java:566)
        at org.springframework.aop.support.AopUtils.invokeJoinpointUsingReflection(AopUtils.java:333)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.invokeJoinpoint(ReflectiveMethodInvocation.java:190)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:157)
        at omero.cmd.CallContext.invoke(CallContext.java:85)
        at org.springframework.aop.framework.ReflectiveMethodInvocation.proceed(ReflectiveMethodInvocation.java:179)
        at org.springframework.aop.framework.JdkDynamicAopProxy.invoke(JdkDynamicAopProxy.java:213)
        at com.sun.proxy.$Proxy101.saveAndReturnObject_async(Unknown Source)
        at omero.api._IUpdateTie.saveAndReturnObject_async(_IUpdateTie.java:92)
        at omero.api._IUpdateDisp.___saveAndReturnObject(_IUpdateDisp.java:229)
        at omero.api._IUpdateDisp.__dispatch(_IUpdateDisp.java:423)
        at IceInternal.Incoming.invoke(Incoming.java:221)
        at Ice.ConnectionI.invokeAll(ConnectionI.java:2536)
        at Ice.ConnectionI.dispatch(ConnectionI.java:1145)
        at Ice.ConnectionI.message(ConnectionI.java:1056)
        at IceInternal.ThreadPool.run(ThreadPool.java:395)
        at IceInternal.ThreadPool.access$300(ThreadPool.java:12)
        at IceInternal.ThreadPool$EventHandlerThread.run(ThreadPool.java:832)
        at java.base/java.lang.Thread.run(Thread.java:829)

    serverExceptionClass = ome.conditions.ValidationException
    message = No row with the given identifier exists: [ome.model.containers.Dataset#200]; nested exception is org.hibernate.ObjectNotFoundException: No row with the given identifier exists: [ome.model.containers.Dataset#200]
}

'''
            
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
                        buf = np.ascontiguousarray(sub_tile).tobytes()
                        self.store.setTile(buf, tile.z, tile.c, tile.t, 
                                         x0 + off_x, y0 + off_y, w, h)
                                         
        except Exception as e:
            self.logger.error(f"Failed to upload tile {tile.ix},{tile.iy} using tile writes: {e}")
            raise
    
    def _upload_tile_row_stripe(self, tile: TileMetadata):
        """Upload tile using row-stripe writes (ROMIO backend)."""
        # For row-stripe mode, we need to collect all tiles in a row before writing
        # This is a simplified implementation - in practice, you'd want to buffer row data
        try:
            x0 = tile.ix * self.tile_w
            y0 = tile.iy * self.tile_h
            
            # Write as a single row stripe
            buf = np.ascontiguousarray(tile.tile_data).tobytes()
            self.store.setTile(buf, tile.z, tile.c, tile.t, 
                             x0, y0, self.tile_w, self.tile_h)
                             
        except Exception as e:
            self.logger.error(f"Failed to upload tile {tile.ix},{tile.iy} using row writes: {e}")
            raise
    
    def finalize(self):
        """Finalize the OMERO upload."""
        if self.store:
            try:
                self.store.save()
                self.logger.info("OMERO upload finalized successfully")
            except Exception as e:
                self.logger.error(f"Failed to finalize OMERO upload: {e}")
    
    def _cleanup_connection(self):
        """Clean up OMERO connection."""
        if self.store:
            try:
                self.store.close()
            except:
                pass
            self.store = None
            
        if self.connection:
            try:
                self.connection.close()
            except:
                pass
            self.connection = None
            
        if self.client:
            try:
                self.client.closeSession()
            except:
                pass
            self.client = None