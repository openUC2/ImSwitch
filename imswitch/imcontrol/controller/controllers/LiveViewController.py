"""
LiveViewController - Centralized controller for all live streaming functionality.

This controller manages per-detector streaming with dedicated worker threads and supports
multiple streaming protocols (binary, JPEG, MJPEG, WebRTC).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from abc import abstractmethod
import numpy as np
import threading
import queue
import time

from imswitch import IS_HEADLESS
from imswitch.imcommon.framework import Signal, Timer, Worker
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import LiveUpdatedController


@dataclass
class StreamParams:
    """Unified dataclass for stream parameters that can be interpreted by frontend/backend."""
    
    # Common parameters
    detector_name: Optional[str] = None  # None means use first available detector
    protocol: str = "binary"  # binary, jpeg, mjpeg, webrtc
    
    # Binary stream parameters
    compression_algorithm: str = "lz4"  # lz4, zstandard
    compression_level: int = 0
    subsampling_factor: int = 4
    throttle_ms: int = 50
    
    # JPEG/MJPEG parameters
    jpeg_quality: int = 80
    
    # WebRTC parameters
    stun_servers: list = field(default_factory=lambda: ["stun:stun.l.google.com:19302"])
    turn_servers: list = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "detector_name": self.detector_name,
            "protocol": self.protocol,
            "compression_algorithm": self.compression_algorithm,
            "compression_level": self.compression_level,
            "subsampling_factor": self.subsampling_factor,
            "throttle_ms": self.throttle_ms,
            "jpeg_quality": self.jpeg_quality,
            "stun_servers": self.stun_servers,
            "turn_servers": self.turn_servers,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamParams':
        """Create StreamParams from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class StreamWorker(Worker):
    """
    Base class for protocol-specific stream workers.
    Each worker runs in its own thread and waits for new frames without using a timer.
    This avoids skipping frames and ensures consistent frame rate.
    Workers do the actual encoding and emit encoded bytes that match socket.io message format.
    """
    
    sigStreamFrame = Signal(dict)  # Emits pre-formatted message dict ready for socket.io emission
    
    def __init__(self, detectorManager, updatePeriodMs: int, streamParams: StreamParams):
        super().__init__()
        self._detector = detectorManager
        self._updatePeriod = updatePeriodMs / 1000.0  # Convert to seconds
        self._params = streamParams
        self._running = False
        self._thread = None
        self._logger = initLogger(self)
        self._last_frame_time = 0
    
    def run(self):
        """Start polling frames without timer - wait and push immediately."""
        self._running = True
        self._logger.debug(f"StreamWorker started with update period {self._updatePeriod}s")
        
        while self._running:
            try:
                # Wait for minimum period
                elapsed = time.time() - self._last_frame_time
                if elapsed < self._updatePeriod:
                    time.sleep(self._updatePeriod - elapsed)
                
                # Capture and emit immediately
                self._captureAndEmit()
                self._last_frame_time = time.time()
                
            except Exception as e:
                self._logger.error(f"Error in StreamWorker loop: {e}")
                time.sleep(0.1)  # Brief pause on error
    
    def stop(self):
        """Stop polling frames."""
        self._running = False
        self._logger.debug("StreamWorker stopped")
    
    @abstractmethod
    def _captureAndEmit(self):
        """Capture frame from detector, encode it, and emit processed data."""
        pass


class BinaryStreamWorker(StreamWorker):
    """Worker for binary frame streaming with LZ4/Zstandard compression."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image_id = 0
        # Import compression libraries on demand
        self._encoder = None
        
    def _get_encoder(self):
        """Get or create encoder with current parameters."""
        try:
            from imswitch.imcommon.framework.binary_streaming import BinaryFrameEncoder
            return BinaryFrameEncoder(
                compression_algorithm=self._params.compression_algorithm,
                compression_level=self._params.compression_level,
                subsampling_factor=self._params.subsampling_factor,
            )
        except ImportError:
            self._logger.error("BinaryFrameEncoder not available")
            return None
    
    def _captureAndEmit(self):
        """Capture frame from detector, compress it, and emit pre-formatted socket.io message."""
        try:
            frame = self._detector.getLatestFrame()
            if frame is None:
                return
            
            # Get detector info
            detector_name = self._detector.name
            pixel_size = 1.0
            try:
                pixel_size = self._detector.pixelSizeUm[-1]
            except:
                pass
            
            # Ensure frame is contiguous and proper type
            frame = np.ascontiguousarray(frame)
            if frame.dtype == np.float32 or frame.dtype == np.float64:
                frame = np.uint8(frame * 255)
            if frame.dtype not in [np.uint8, np.uint16]:
                frame = np.uint8(frame)
            
            # Get encoder
            encoder = self._get_encoder()
            if encoder is None:
                return
            
            # Encode frame
            packet, metadata = encoder.encode_frame(frame)
            
            # Add metadata
            metadata['server_timestamp'] = time.time()
            metadata['image_id'] = self._image_id
            metadata['detectorname'] = detector_name
            metadata['pixelsize'] = int(pixel_size)
            metadata['format'] = 'binary'
            
            self._image_id += 1
            
            # Create pre-formatted message for socket.io
            message = {
                'type': 'binary_frame',
                'event': 'frame',
                'data': packet,
                'metadata': metadata
            }
            
            # Emit pre-formatted message
            self.sigStreamFrame.emit(message)
            
        except Exception as e:
            self._logger.error(f"Error in BinaryStreamWorker: {e}")


class JPEGStreamWorker(StreamWorker):
    """Worker for JPEG frame streaming with compression."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image_id = 0
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            self._logger.error("opencv-python required for JPEG streaming")
            self._cv2 = None
    
    def _captureAndEmit(self):
        """Capture frame from detector, encode as JPEG, and emit pre-formatted socket.io message."""
        if self._cv2 is None:
            return
        
        try:
            frame = self._detector.getLatestFrame()
            if frame is None:
                return
            
            # Get detector info
            detector_name = self._detector.name
            pixel_size = 1.0
            try:
                pixel_size = self._detector.pixelSizeUm[-1]
            except:
                pass
            
            # Normalize to uint8 if needed
            if frame.dtype != np.uint8:
                vmin = float(np.min(frame))
                vmax = float(np.max(frame))
                if vmax > vmin:
                    frame = ((frame - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
            
            # Apply subsampling if needed
            if self._params.subsampling_factor > 1:
                frame = frame[::self._params.subsampling_factor, ::self._params.subsampling_factor]
            
            # Encode as JPEG
            encode_params = [self._cv2.IMWRITE_JPEG_QUALITY, self._params.jpeg_quality]
            success, encoded = self._cv2.imencode('.jpg', frame, encode_params)

            if success: # TODO: Eventually think about messagepack instead of base64
                import base64
                jpeg_bytes = encoded.tobytes()
                encoded_image = base64.b64encode(jpeg_bytes).decode('utf-8')
                
                # Create metadata
                metadata = {
                    'server_timestamp': time.time(),
                    'image_id': self._image_id,
                    'detectorname': detector_name,
                    'pixelsize': int(pixel_size),
                    'format': 'jpeg'
                }
                
                self._image_id += 1
                
                # Create pre-formatted message for socket.io (JSON signal format)
                message = {
                    'type': 'jpeg_frame',
                    'event': 'signal',
                    'data': {
                        'name': 'sigUpdateImage',
                        'detectorname': detector_name,
                        'pixelsize': int(pixel_size),
                        'format': 'jpeg',
                        'image': encoded_image,
                        'server_timestamp': metadata['server_timestamp'],
                        'image_id': self._image_id
                    }
                }
                
                # Emit pre-formatted message
                self.sigStreamFrame.emit(message)
                
        except Exception as e:
            self._logger.error(f"Error in JPEGStreamWorker: {e}")


class MJPEGStreamWorker(StreamWorker):
    """
    Worker for MJPEG HTTP streaming.
    Replaces RecordingController.video_feeder functionality.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_queue = queue.Queue(maxsize=10)
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            self._logger.error("opencv-python required for MJPEG streaming")
            self._cv2 = None
    
    def _captureAndEmit(self):
        """Capture frame and put in queue for MJPEG streaming."""
        if self._cv2 is None:
            return
        
        try:
            frame = self._detector.getLatestFrame()
            if frame is None:
                return
            
            # Normalize to uint8 if needed
            if frame.dtype != np.uint8:
                vmin = float(np.min(frame))
                vmax = float(np.max(frame))
                if vmax > vmin:
                    frame = ((frame - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
            
            # Encode as JPEG
            encode_params = [self._cv2.IMWRITE_JPEG_QUALITY, self._params.jpeg_quality]
            success, encoded = self._cv2.imencode('.jpg', frame, encode_params)
            
            if success:
                jpeg_bytes = encoded.tobytes()
                # Build MJPEG frame with proper headers
                header = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                )
                content_length = f'Content-Length: {len(jpeg_bytes)}\r\n\r\n'.encode('ascii')
                mjpeg_frame = header + content_length + jpeg_bytes + b'\r\n'
                
                # Put in queue, drop frame if full
                try:
                    self._frame_queue.put_nowait(mjpeg_frame)
                except queue.Full:
                    pass  # Drop frame if queue is full
        except Exception as e:
            self._logger.error(f"Error in MJPEGStreamWorker: {e}")
    
    def get_frame(self, timeout=1.0):
        """Get next frame from queue (for HTTP streaming)."""
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None


class WebRTCStreamWorker(StreamWorker):
    """
    Worker for WebRTC streaming using aiortc.
    Provides low-latency real-time streaming using WebRTC protocol.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._video_track = None
        self._frame_queue = queue.Queue(maxsize=2)  # Small queue for latest frames
        
        try:
            from aiortc import VideoStreamTrack
            import av
            self._VideoStreamTrack = VideoStreamTrack
            self._av = av
            self._has_webrtc = True
        except ImportError:
            self._logger.error("aiortc and av required for WebRTC streaming")
            self._has_webrtc = False
    
    def _captureAndEmit(self):
        """Capture frame and put in queue for WebRTC streaming."""
        if not self._has_webrtc:
            return
        
        try:
            frame = self._detector.getLatestFrame()
            if frame is None:
                return
            
            # Normalize to uint8 if needed
            if frame.dtype != np.uint8:
                vmin = float(np.min(frame))
                vmax = float(np.max(frame))
                if vmax > vmin:
                    frame = ((frame - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                else:
                    frame = np.zeros_like(frame, dtype=np.uint8)
            
            # Put frame in queue, replacing old frame if full
            try:
                # Try to clear old frame
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    pass
                self._frame_queue.put_nowait(frame)
            except queue.Full:
                pass  # Drop frame if queue is full
                
        except Exception as e:
            self._logger.error(f"Error in WebRTCStreamWorker: {e}")
    
    def get_video_track(self):
        """Get or create video track for WebRTC."""
        if self._video_track is None and self._has_webrtc:
            self._video_track = DetectorVideoTrack(
                self._frame_queue, 
                self._detector.name,
                self._VideoStreamTrack,
                self._av
            )
        return self._video_track


class DetectorVideoTrack:
    """Custom video track that reads frames from detector queue."""
    
    def __init__(self, frame_queue, detector_name, VideoStreamTrack, av):
        self._queue = frame_queue
        self._detector_name = detector_name
        self._av = av
        
        # Create custom track class that inherits from VideoStreamTrack
        class CustomVideoTrack(VideoStreamTrack):
            def __init__(inner_self):
                super().__init__()
                inner_self.kind = "video"
                inner_self._queue = frame_queue
                inner_self._av = av
                inner_self._timestamp = 0
                inner_self._time_base = 1 / 30  # 30 fps
            
            async def recv(inner_self):
                """Receive next video frame."""
                import asyncio
                
                # Wait for frame from queue (with timeout)
                try:
                    frame = await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(
                            None, inner_self._queue.get, True, 1.0
                        ),
                        timeout=1.0
                    )
                except (asyncio.TimeoutError, queue.Empty):
                    # Return black frame if no frame available
                    frame = np.zeros((480, 640), dtype=np.uint8)
                
                # Ensure frame is the right shape
                if len(frame.shape) == 2:
                    # Grayscale - convert to RGB
                    frame = np.stack([frame, frame, frame], axis=2)
                elif frame.shape[2] == 1:
                    frame = np.repeat(frame, 3, axis=2)
                
                # Create av.VideoFrame
                new_frame = inner_self._av.VideoFrame.from_ndarray(frame, format="rgb24")
                new_frame.pts = inner_self._timestamp
                new_frame.time_base = inner_self._time_base
                inner_self._timestamp += 1
                
                return new_frame
        
        self._track = CustomVideoTrack()
    
    def __getattr__(self, name):
        """Delegate attribute access to the underlying track."""
        return getattr(self._track, name)


class LiveViewController(LiveUpdatedController):
    """
    Centralized controller for all live streaming functionality.
    Manages per-detector streaming with dedicated worker threads.
    Only one protocol can be active per detector at a time.
    """
    
    sigStreamStarted = Signal(str, str)      # (detectorName, protocol)
    sigStreamStopped = Signal(str, str)      # (detectorName, protocol)
    sigStreamFrame = Signal(dict)            # Pre-formatted socket.io message
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        
        # Active streams: {detectorName: (protocol, StreamWorker)}
        # Only one protocol per detector allowed
        self._activeStreams: Dict[str, tuple] = {}
        self._streamThreads: Dict[str, threading.Thread] = {}
        self._streamIsRunning = False
        
        # Global stream parameters per protocol
        self._streamParams: Dict[str, StreamParams] = {
            'binary': StreamParams(protocol='binary'),
            'jpeg': StreamParams(protocol='jpeg'),
            'mjpeg': StreamParams(protocol='mjpeg'),
            'webrtc': StreamParams(protocol='webrtc'),
        }
        
        # Connect to communication channel signals
        self._commChannel.sigStartLiveAcquistion.connect(self._onStartLiveAcquisition)
        self._commChannel.sigStopLiveAcquisition.connect(self._onStopLiveAcquisition)
        '''
        TODO: need to handle 
        
            self.execOnAll(lambda c: c.stopAcquisition(), condition=lambda c: c.forAcquisition)
            self.sigAcquisitionStopped.emit()
            self.execOnAll(lambda c: c.startAcquisition(), condition=lambda c: c.forAcquisition)
            self.sigAcquisitionStarted.emit()
            '''
        self._logger.info("LiveViewController initialized")
    
    def _onStartLiveAcquisition(self, start: bool):
        """Handle start live acquisition signal."""
        if start:
            self._logger.debug("Received start live acquisition signal")
            # This can be used to automatically start default streaming
            # For now, streaming is explicitly controlled via API
        else:
            self._logger.debug("Received stop live acquisition signal")
    
    def _onStopLiveAcquisition(self, stop: bool):
        """Handle stop live acquisition signal."""
        if stop:
            self._logger.debug("Stopping all live acquisitions")
            # Stop all active streams
            for detector_name in list(self._activeStreams.keys()):
                self.stopLiveView(detector_name)
    
    @APIExport()
    def getLiveViewActive(self) -> bool:
        """Check if any live view stream is currently active."""
        return len(self._activeStreams) > 0
    
    @APIExport(requestType="POST")
    def startLiveView(self, detectorName: Optional[str] = None, protocol: str = "binary",
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start live streaming for a specific detector.
        Only one protocol can be active per detector at a time.
        
        Args:
            detectorName: Name of detector to stream from (None = first available)
            protocol: Streaming protocol (binary, jpeg, mjpeg, webrtc)
            params: Optional parameters to override defaults
        
        Returns:
            Dictionary with status and stream info
        """
        try:
            # Get detector
            if detectorName is None:
                detectorName = self._master.detectorsManager.getAllDeviceNames()[0]
            
            detector = self._master.detectorsManager[detectorName]
            
            # Check if detector already has an active stream
            if detectorName in self._activeStreams:
                old_protocol, old_worker = self._activeStreams[detectorName]
                return {
                    "status": "already_running",
                    "detector": detectorName,
                    "protocol": old_protocol,
                    "message": f"Stream already active for {detectorName} with protocol {old_protocol}. Stop it first."
                }
            
            # Get stream parameters and update global params
            stream_params = self._streamParams.get(protocol, StreamParams(protocol=protocol))
            if params:
                # Update with provided parameters
                for key, value in params.items():
                    if hasattr(stream_params, key):
                        setattr(stream_params, key, value)
            
            # Update DetectorsManager global params for streaming configuration
            update_params = {}
            if protocol in ["binary", "jpeg"]:
                update_params['stream_compression_algorithm'] = stream_params.compression_algorithm if protocol == "binary" else "jpeg"
                update_params['stream_compression_level'] = stream_params.compression_level
                update_params['stream_subsampling_factor'] = stream_params.subsampling_factor
                update_params['stream_throttle_ms'] = stream_params.throttle_ms
                if protocol == "jpeg":
                    update_params['compressionlevel'] = stream_params.jpeg_quality
                
                self._master.detectorsManager.updateGlobalDetectorParams(update_params) # TOOD: I think this is still not needed anymore
            
            # Create appropriate worker
            worker = self._createWorker(detector, protocol, stream_params)
            if worker is None:
                return {
                    "status": "error",
                    "message": f"Failed to create worker for protocol {protocol}"
                }
            
            # Connect worker signal to controller's signal, which is then handled by noqt
            # The worker emits pre-formatted messages ready for socket.io emission
            worker.sigStreamFrame.connect(self._commChannel.sigUpdateImage)
            
            # Start worker in thread
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            
            # Store worker and thread (only protocol and worker, not tuple key)
            self._activeStreams[detectorName] = (protocol, worker)
            self._streamThreads[detectorName] = thread
            
            # Emit signal
            self.sigStreamStarted.emit(detectorName, protocol)
            
            self._logger.info(f"Started {protocol} stream for detector {detectorName}")
            
            return {
                "status": "success",
                "detector": detectorName,
                "protocol": protocol,
                "params": stream_params.to_dict()
            }
        except Exception as e:
            self._logger.error(f"Error starting live view: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @APIExport()
    def stopLiveView(self, detectorName: Optional[str] = None) -> Dict[str, Any]:
        """
        Stop live streaming for a specific detector.
        If detectorName is None, stops the first active stream.
        Protocol is ignored - we only care about the detector.
        
        Args:
            detectorName: Name of detector (None = stop first active detector)
        
        Returns:
            Dictionary with status
        """
        try:
            # If no detector specified, stop first active stream
            if detectorName is None:
                if not self._activeStreams:
                    return {
                        "status": "not_running",
                        "message": "No active streams to stop"
                    }
                detectorName = list(self._activeStreams.keys())[0]
            
            # Check if detector has active stream
            if detectorName not in self._activeStreams:
                return {
                    "status": "not_running",
                    "detector": detectorName,
                    "message": f"No active stream for detector {detectorName}"
                }
            
            # Get protocol and worker
            protocol, worker = self._activeStreams[detectorName]
            
            # Stop worker
            worker.stop()
            
            # Clean up
            del self._activeStreams[detectorName]
            if detectorName in self._streamThreads:
                del self._streamThreads[detectorName]
            
            # Emit signal
            self.sigStreamStopped.emit(detectorName, protocol)
            
            self._logger.info(f"Stopped {protocol} stream for detector {detectorName}")
            
            return {
                "status": "success",
                "detector": detectorName,
                "protocol": protocol
            }
        except Exception as e:
            self._logger.error(f"Error stopping live view: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @APIExport()
    def setStreamParameters(self, protocol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure streaming parameters for a protocol (global settings).
        
        Args:
            protocol: Streaming protocol (binary, jpeg, mjpeg, webrtc)
            params: Dictionary of parameters to set
        
        Returns:
            Dictionary with status and updated params
            
        example params:
            For binary:
                {'compression_algorithm': 'lz4', 'compression_level': 0, 'subsampling_factor': 4, 'throttle_ms': 50}
            For jpeg:
                {'jpeg_quality': 80, 'subsampling_factor': 4, 'throttle_ms': 50}
            For mjpeg:
                {'jpeg_quality': 80, 'subsampling_factor': 4, 'throttle_ms': 50}
            For webrtc:
                {'ice_servers': [{'urls': ['stun:stun.l.google.com:19302']}], 'media_constraints': {'video': True, 'audio': True}}
        """
        try:
            if protocol not in self._streamParams:
                self._streamParams[protocol] = StreamParams(protocol=protocol)
            
            # Update parameters
            stream_params = self._streamParams[protocol]
            for key, value in params.items():
                if hasattr(stream_params, key):
                    setattr(stream_params, key, value)
            
            # Also update DetectorsManager global params for backward compatibility
            if protocol == "binary":
                update_params = {}
                if 'compression_algorithm' in params:
                    update_params['stream_compression_algorithm'] = params['compression_algorithm']
                if 'compression_level' in params:
                    update_params['stream_compression_level'] = params['compression_level']
                if 'subsampling_factor' in params:
                    update_params['stream_subsampling_factor'] = params['subsampling_factor']
                if 'throttle_ms' in params:
                    update_params['stream_throttle_ms'] = params['throttle_ms']
                # TOOD: I think this is still not needed anymore
                self._master.detectorsManager.updateGlobalDetectorParams(update_params)
            elif protocol == "jpeg":
                if 'jpeg_quality' in params: 
                    self._master.detectorsManager.updateGlobalDetectorParams({ # TOOD: I think this is still not needed anymore
                        'compressionlevel': params['jpeg_quality']
                    })
            
            return {
                "status": "success",
                "protocol": protocol,
                "params": stream_params.to_dict()
            }
        except Exception as e:
            self._logger.error(f"Error setting stream params: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @APIExport()
    def getStreamParameters(self, protocol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current streaming parameters.
        
        Args:
            protocol: Optional protocol to get params for (None = all protocols)
        
        Returns:
            Dictionary with streaming parameters
        """
        try:
            if protocol:
                if protocol in self._streamParams:
                    return {
                        "status": "success",
                        "protocol": protocol,
                        "params": self._streamParams[protocol].to_dict()
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Unknown protocol: {protocol}"
                    }
            else:
                # Return all protocols
                return {
                    "status": "success",
                    "protocols": {
                        p: params.to_dict() 
                        for p, params in self._streamParams.items()
                    }
                }
        except Exception as e:
            self._logger.error(f"Error getting stream params: {e}")
            return {
                "status": "error",
                "message": str(e)
            }
    
    @APIExport()
    def getActiveStreams(self) -> Dict[str, Any]:
        """
        Get list of currently active streams.
        
        Returns:
            Dictionary with active stream information
        """
        return {
            "status": "success",
            "active_streams": [
                {
                    "detector": detector_name,
                    "protocol": protocol
                }
                for detector_name, (protocol, worker) in self._activeStreams.items()
            ]
        }
    
    def _createWorker(self, detector, protocol: str, params: StreamParams) -> Optional[StreamWorker]:
        """Create appropriate worker for the given protocol."""
        update_period = params.throttle_ms
        
        if protocol == "binary":
            return BinaryStreamWorker(detector, update_period, params)
        elif protocol == "jpeg":
            return JPEGStreamWorker(detector, update_period, params)
        elif protocol == "mjpeg":
            return MJPEGStreamWorker(detector, update_period, params)
        elif protocol == "webrtc":
            return WebRTCStreamWorker(detector, update_period, params)
        else:
            self._logger.error(f"Unknown protocol: {protocol}")
            return None
    
    def getMJPEGWorker(self, detectorName: str) -> Optional[MJPEGStreamWorker]:
        """Get MJPEG worker for HTTP streaming (used by video_feeder endpoint)."""
        if detectorName in self._activeStreams:
            protocol, worker = self._activeStreams[detectorName]
            if protocol == "mjpeg" and isinstance(worker, MJPEGStreamWorker):
                return worker
        return None
    
    @APIExport(runOnUIThread=False)
    def mjpeg_stream(self, startStream: bool = True, detectorName: Optional[str] = None):
        """
        HTTP endpoint for MJPEG streaming.
        Replaces RecordingController.video_feeder with LiveViewController implementation.
        
        Args:
            startStream: Whether to start streaming
            detectorName: Name of detector (None = first available)
        
        Returns:
            StreamingResponse with MJPEG data or status message
        """
        try:
            from fastapi.responses import StreamingResponse
        except ImportError:
            return {"status": "error", "message": "FastAPI not available"}
        
        if not startStream:
            # Stop the stream - only care about detector
            self.stopLiveView(detectorName)
            return {"status": "success", "message": "stream stopped"}
        
        # Start streaming
        if detectorName is None:
            detectorName = self._master.detectorsManager.getAllDeviceNames()[0]
        
        # Check if stream already exists for this detector
        if detectorName not in self._activeStreams:
            # Start the MJPEG stream
            result = self.startLiveView(detectorName, "mjpeg")
            if result['status'] != 'success':
                return result
        
        # Get the worker
        worker = self.getMJPEGWorker(detectorName)
        if worker is None:
            return {"status": "error", "message": "Failed to get MJPEG worker"}
        
        # Create generator for streaming response
        def frame_generator():
            """Generator that yields MJPEG frames."""
            try:
                while detectorName in self._activeStreams:
                    frame = worker.get_frame(timeout=1.0)
                    if frame:
                        yield frame
            except GeneratorExit:
                self._logger.debug("MJPEG stream connection closed by client")
            except Exception as e:
                self._logger.error(f"Error in MJPEG frame generator: {e}")
        
        # Return streaming response with proper headers
        headers = {
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        }
        
        return StreamingResponse(
            frame_generator(),
            media_type="multipart/x-mixed-replace;boundary=frame",
            headers=headers
        )
    
    def getWebRTCWorker(self, detectorName: str) -> Optional[WebRTCStreamWorker]:
        """Get WebRTC worker for signaling (used by WebRTC endpoints)."""
        if detectorName in self._activeStreams:
            protocol, worker = self._activeStreams[detectorName]
            if protocol == "webrtc" and isinstance(worker, WebRTCStreamWorker):
                return worker
        return None
    
    @APIExport(runOnUIThread=False, requestType="POST")
    def webrtc_offer(self, detectorName: Optional[str] = None, sdp: str = None, type: str = None):
        """
        Handle WebRTC offer from client and return answer.
        
        Args:
            detectorName: Name of detector (None = first available)
            sdp: Session Description Protocol from client
            type: Type of SDP (should be "offer")
        
        Returns:
            Dictionary with answer SDP
        """
        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
            import asyncio
            import json
        except ImportError:
            return {"status": "error", "message": "aiortc not available"}
        
        # Get detector name
        if detectorName is None:
            detectorName = self._master.detectorsManager.getAllDeviceNames()[0]
        
        # Start WebRTC stream if not already active
        if detectorName not in self._activeStreams:
            result = self.startLiveView(detectorName, "webrtc")
            if result['status'] != 'success':
                return result
        
        # Get the worker
        worker = self.getWebRTCWorker(detectorName)
        if worker is None:
            return {"status": "error", "message": "Failed to get WebRTC worker"}
        
        # Create peer connection
        pc = RTCPeerConnection()
        
        # Store PC for cleanup
        if not hasattr(self, '_webrtc_peers'):
            self._webrtc_peers = set()
        self._webrtc_peers.add(pc)
        
        @pc.on("connectionstatechange")
        async def on_connectionstatechange():
            self._logger.info(f"WebRTC connection state: {pc.connectionState}")
            if pc.connectionState == "failed" or pc.connectionState == "closed":
                await pc.close()
                if pc in self._webrtc_peers:
                    self._webrtc_peers.discard(pc)
        
        # Add video track
        video_track = worker.get_video_track()
        if video_track:
            pc.addTrack(video_track)
        
        # Handle offer and create answer
        async def process_offer():
            offer_desc = RTCSessionDescription(sdp=sdp, type=type)
            await pc.setRemoteDescription(offer_desc)
            
            answer = await pc.createAnswer()
            await pc.setLocalDescription(answer)
            
            return {
                "status": "success",
                "sdp": pc.localDescription.sdp,
                "type": pc.localDescription.type
            }
        
        # Run async function
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(process_offer())
            return result
        finally:
            loop.close()
    
    @APIExport(runOnUIThread=False)
    def webrtc_ice_candidate(self, detectorName: Optional[str] = None, candidate: dict = None):
        """
        Handle ICE candidate from client.
        
        Args:
            detectorName: Name of detector
            candidate: ICE candidate information
        
        Returns:
            Status dictionary
        """
        # This is a simplified version - full implementation would need to store
        # and manage ICE candidates per peer connection
        return {"status": "success", "message": "ICE candidate received"}


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
