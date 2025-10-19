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
    """
    
    sigFrameReady = Signal(str, np.ndarray, bool, list, bool)  # Reuse sigUpdateImage format
    
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
        """Capture frame from detector and emit processed data."""
        pass


class BinaryStreamWorker(StreamWorker):
    """Worker for binary frame streaming - processing happens in noqt framework."""
    
    def _captureAndEmit(self):
        """Capture frame from detector and emit in sigUpdateImage format."""
        try:
            frame = self._detector.getLatestFrame()
            if frame is None:
                return
            
            # Get detector info
            detector_name = self._detector.name
            pixel_size = [1.0]  # Default pixel size
            try:
                pixel_size = [self._detector.pixelSizeUm[-1]]
            except:
                pass
            
            # Emit in sigUpdateImage format: (detectorName, image, init, scale, isCurrentDetector)
            # The noqt framework will handle the actual compression based on stream params
            self.sigFrameReady.emit(detector_name, frame, False, pixel_size, True)
            
        except Exception as e:
            self._logger.error(f"Error in BinaryStreamWorker: {e}")


class JPEGStreamWorker(StreamWorker):
    """Worker for JPEG frame streaming - processing happens in noqt framework."""
    
    def _captureAndEmit(self):
        """Capture frame from detector and emit in sigUpdateImage format."""
        try:
            frame = self._detector.getLatestFrame()
            if frame is None:
                return
            
            # Get detector info
            detector_name = self._detector.name
            pixel_size = [1.0]  # Default pixel size
            try:
                pixel_size = [self._detector.pixelSizeUm[-1]]
            except:
                pass
            
            # Emit in sigUpdateImage format: (detectorName, image, init, scale, isCurrentDetector)
            # The noqt framework will handle the actual JPEG encoding based on stream params
            self.sigFrameReady.emit(detector_name, frame, False, pixel_size, True)
            
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


class LiveViewController(LiveUpdatedController):
    """
    Centralized controller for all live streaming functionality.
    Manages per-detector streaming with dedicated worker threads.
    Only one protocol can be active per detector at a time.
    """
    
    sigStreamStarted = Signal(str, str)      # (detectorName, protocol)
    sigStreamStopped = Signal(str, str)      # (detectorName, protocol)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        
        # Active streams: {detectorName: (protocol, StreamWorker)}
        # Only one protocol per detector allowed
        self._activeStreams: Dict[str, tuple] = {}
        self._streamThreads: Dict[str, threading.Thread] = {}
        
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
            
            self._master.detectorsManager.updateGlobalDetectorParams(update_params)
            
            # Create appropriate worker
            worker = self._createWorker(detector, protocol, stream_params)
            if worker is None:
                return {
                    "status": "error",
                    "message": f"Failed to create worker for protocol {protocol}"
                }
            
            # Connect worker to emit sigUpdateImage through communication channel
            worker.sigFrameReady.connect(
                lambda detectorName, image, init, scale, isCurrentDetector: 
                self._commChannel.sigUpdateImage.emit(detectorName, image, init, scale, isCurrentDetector)
            )
            
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
