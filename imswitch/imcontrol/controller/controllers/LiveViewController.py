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
from ..basecontrollers import ImConWidgetController


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
    Each worker runs in its own thread and polls frames at the configured rate.
    """
    
    sigFrameReady = Signal(bytes)  # Emits processed frame data
    
    def __init__(self, detectorManager, updatePeriodMs: int, streamParams: StreamParams):
        super().__init__()
        self._detector = detectorManager
        self._updatePeriod = updatePeriodMs
        self._params = streamParams
        self._running = False
        self._timer = None
        self._logger = initLogger(self)
    
    def run(self):
        """Start polling frames at configured rate."""
        self._running = True
        self._timer = Timer()
        self._timer.timeout.connect(self._captureAndEmit)
        self._timer.start(self._updatePeriod)
        self._logger.debug(f"StreamWorker started with update period {self._updatePeriod}ms")
    
    def stop(self):
        """Stop polling frames."""
        self._running = False
        if self._timer:
            self._timer.stop()
        self._logger.debug("StreamWorker stopped")
    
    @abstractmethod
    def _captureAndEmit(self):
        """Capture frame from detector and emit processed data."""
        pass


class BinaryStreamWorker(StreamWorker):
    """Worker for binary frame streaming with LZ4/Zstandard compression."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Import compression libraries on demand
        if self._params.compression_algorithm == "lz4":
            try:
                import lz4.frame
                self._compress = lz4.frame.compress
            except ImportError:
                self._logger.warning("lz4 not available, falling back to no compression")
                self._compress = lambda x: x.tobytes()
        elif self._params.compression_algorithm == "zstandard":
            try:
                import zstandard as zstd
                self._compressor = zstd.ZstdCompressor(level=self._params.compression_level)
                self._compress = self._compressor.compress
            except ImportError:
                self._logger.warning("zstandard not available, falling back to no compression")
                self._compress = lambda x: x.tobytes()
        else:
            self._compress = lambda x: x.tobytes()
    
    def _captureAndEmit(self):
        """Capture frame from detector and emit compressed binary data."""
        try:
            frame = self._detector.getLatestFrame()
            if frame is None:
                return
            
            # Apply subsampling if configured
            if self._params.subsampling_factor > 1:
                frame = frame[::self._params.subsampling_factor, 
                             ::self._params.subsampling_factor]
            
            # Compress frame
            compressed = self._compress(frame)
            
            # Emit compressed frame
            self.sigFrameReady.emit(compressed)
        except Exception as e:
            self._logger.error(f"Error in BinaryStreamWorker: {e}")


class JPEGStreamWorker(StreamWorker):
    """Worker for JPEG frame streaming."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            self._logger.error("opencv-python required for JPEG streaming")
            self._cv2 = None
    
    def _captureAndEmit(self):
        """Capture frame from detector and emit JPEG data."""
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
                self.sigFrameReady.emit(encoded.tobytes())
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
                mjpeg_frame = (
                    b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n'
                    f'Content-Length: {len(jpeg_bytes)}\r\n\r\n'.encode('ascii') +
                    jpeg_bytes + b'\r\n'
                )
                
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
    Provides foundation for low-latency real-time streaming.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._rtcConnection = None
        self._videoTrack = None
        
        try:
            from aiortc import RTCPeerConnection, VideoStreamTrack
            import av
            self._RTCPeerConnection = RTCPeerConnection
            self._VideoStreamTrack = VideoStreamTrack
            self._av = av
        except ImportError:
            self._logger.error("aiortc and av required for WebRTC streaming")
            self._RTCPeerConnection = None
    
    def _captureAndEmit(self):
        """Capture frame and push to WebRTC track."""
        if self._RTCPeerConnection is None or self._videoTrack is None:
            return
        
        try:
            frame = self._detector.getLatestFrame()
            if frame is None:
                return
            
            # Convert numpy array to av.VideoFrame
            # This is a placeholder - actual implementation would need proper frame conversion
            # video_frame = self._convertToAVFrame(frame)
            # self._videoTrack.putFrame(video_frame)
            
            # For now, just log that we would send a frame
            # Full implementation requires more complex async handling
            pass
        except Exception as e:
            self._logger.error(f"Error in WebRTCStreamWorker: {e}")


class LiveViewController(ImConWidgetController):
    """
    Centralized controller for all live streaming functionality.
    Manages per-detector streaming with dedicated worker threads.
    """
    
    sigFrameReady = Signal(str, str, bytes)  # (detectorName, protocol, frameData)
    sigStreamStarted = Signal(str, str)      # (detectorName, protocol)
    sigStreamStopped = Signal(str, str)      # (detectorName, protocol)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        
        # Active streams: {(detectorName, protocol): StreamWorker}
        self._activeStreams: Dict[tuple, StreamWorker] = {}
        self._streamThreads: Dict[tuple, threading.Thread] = {}
        
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
        
        # Connect widget signals if not headless
        if not IS_HEADLESS and self._widget is not None:
            # Populate detector list
            detectorNames = self._master.detectorsManager.getAllDeviceNames()
            self._widget.setDetectorList(detectorNames)
            
            # Connect widget signals
            self._widget.sigStartStream.connect(self._onWidgetStartStream)
            self._widget.sigStopStream.connect(self._onWidgetStopStream)
        
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
            for key in list(self._activeStreams.keys()):
                detector_name, protocol = key
                self.stopLiveView(detector_name, protocol)
    
    def _onWidgetStartStream(self, detectorName: str, protocol: str):
        """Handle start stream from widget."""
        if not IS_HEADLESS and self._widget is not None:
            result = self.startLiveView(detectorName, protocol)
            if result['status'] == 'success':
                self._widget.setStatus(f"Streaming {protocol} from {detectorName}")
                self._widget.setButtonsEnabled(False, True)
                self._updateWidgetActiveStreams()
            else:
                self._widget.setStatus(f"Error: {result.get('message', 'Unknown error')}")
    
    def _onWidgetStopStream(self, detectorName: str, protocol: str):
        """Handle stop stream from widget."""
        if not IS_HEADLESS and self._widget is not None:
            result = self.stopLiveView(detectorName, protocol)
            if result['status'] == 'success':
                self._widget.setStatus("Stream stopped")
                self._widget.setButtonsEnabled(True, False)
                self._updateWidgetActiveStreams()
            else:
                self._widget.setStatus(f"Error: {result.get('message', 'Unknown error')}")
    
    def _updateWidgetActiveStreams(self):
        """Update widget with current active streams."""
        if not IS_HEADLESS and self._widget is not None:
            streams = self.getActiveStreams()
            self._widget.updateActiveStreams(streams.get('active_streams', []))
    
    @APIExport()
    def startLiveView(self, detectorName: Optional[str] = None, protocol: str = "binary", 
                      params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Start live streaming for a specific detector.
        
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
            
            # Check if stream already active
            stream_key = (detectorName, protocol)
            if stream_key in self._activeStreams:
                return {
                    "status": "already_running",
                    "detector": detectorName,
                    "protocol": protocol,
                    "message": f"Stream already active for {detectorName} with protocol {protocol}"
                }
            
            # Get stream parameters
            stream_params = self._streamParams.get(protocol, StreamParams(protocol=protocol))
            if params:
                # Update with provided parameters
                for key, value in params.items():
                    if hasattr(stream_params, key):
                        setattr(stream_params, key, value)
            
            # Create appropriate worker
            worker = self._createWorker(detector, protocol, stream_params)
            if worker is None:
                return {
                    "status": "error",
                    "message": f"Failed to create worker for protocol {protocol}"
                }
            
            # Connect worker signals
            worker.sigFrameReady.connect(
                lambda data, dn=detectorName, p=protocol: self.sigFrameReady.emit(dn, p, data)
            )
            
            # Start worker in thread
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()
            
            # Store worker and thread
            self._activeStreams[stream_key] = worker
            self._streamThreads[stream_key] = thread
            
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
    def stopLiveView(self, detectorName: str, protocol: str) -> Dict[str, Any]:
        """
        Stop live streaming for a specific detector.
        
        Args:
            detectorName: Name of detector
            protocol: Streaming protocol
        
        Returns:
            Dictionary with status
        """
        try:
            stream_key = (detectorName, protocol)
            
            if stream_key not in self._activeStreams:
                return {
                    "status": "not_running",
                    "detector": detectorName,
                    "protocol": protocol,
                    "message": f"No active stream for {detectorName} with protocol {protocol}"
                }
            
            # Stop worker
            worker = self._activeStreams[stream_key]
            worker.stop()
            
            # Clean up
            del self._activeStreams[stream_key]
            if stream_key in self._streamThreads:
                del self._streamThreads[stream_key]
            
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
    def setStreamParams(self, protocol: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Configure streaming parameters for a protocol (global settings).
        
        Args:
            protocol: Streaming protocol (binary, jpeg, mjpeg, webrtc)
            params: Dictionary of parameters to set
        
        Returns:
            Dictionary with status and updated params
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
                
                self._master.detectorsManager.updateGlobalDetectorParams(update_params)
            elif protocol == "jpeg":
                if 'jpeg_quality' in params:
                    self._master.detectorsManager.updateGlobalDetectorParams({
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
    def getStreamParams(self, protocol: Optional[str] = None) -> Dict[str, Any]:
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
                for detector_name, protocol in self._activeStreams.keys()
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
        stream_key = (detectorName, "mjpeg")
        worker = self._activeStreams.get(stream_key)
        if isinstance(worker, MJPEGStreamWorker):
            return worker
        return None


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
