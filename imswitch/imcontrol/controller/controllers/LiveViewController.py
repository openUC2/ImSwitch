"""
LiveViewController - Centralized controller for all live streaming functionality.

This controller manages per-detector streaming with dedicated worker threads and supports
multiple streaming protocols (binary, JPEG, MJPEG, WebRTC).
"""
import asyncio
from dataclasses import dataclass, field
from typing import Dict, Optional, Any
from abc import abstractmethod
import numpy as np
import threading
import queue
import time
from pydantic import BaseModel

from imswitch.imcommon.framework import Signal, Worker
from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import LiveUpdatedController


@dataclass
class StreamParams:
    """Unified dataclass for stream parameters that can be interpreted by frontend/backend."""

    # Common parameters
    detector_name: Optional[str] = None  # None means use first available detector
    protocol: str = "jpeg"  # binary, jpeg, mjpeg, webrtc

    # Binary stream parameters
    compression_algorithm: str = "lz4"  # lz4, zstandard
    compression_level: int = 0
    subsampling_factor: int = 4
    throttle_ms: int = 50

    # Crop parameters (applied before subsampling)
    crop_size: int = 0  # 0 means no crop (full FOV), >0 crops quadratic region around center

    # JPEG/MJPEG parameters
    jpeg_quality: int = 80

    # WebRTC parameters
    stun_servers: list = field(default_factory=list)  # Empty by default - works without internet!
    turn_servers: list = field(default_factory=list)
    max_width: int = 1280  # Maximum frame width in pixels, 0 = no limit

    # Fan-out control: when True, the StreamWorker also emits raw frames on
    # ``sigUpdateFrame`` (wired to ``CommunicationChannel.sigUpdateImage``),
    # which then triggers every connected controller's ``update()``
    # synchronously on the streaming thread (HistogrammController,
    # FFTController, InLineHoloController, OffAxisHoloController,
    # ImageController, RecordingService for streaming recordings, …).
    # For large sensors on a Pi 5 this is a major source of streaming lag:
    # the broadcast cost is linear in (n_listeners × pixel_count). Set to
    # False on detectors that don't need realtime per-frame analysis. Can be
    # overridden per-detector via ``DetectorInfo.defaultStreamSettings`` in
    # the setup JSON, e.g. ``{"broadcast_frames": false}``.
    broadcast_frames: bool = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses."""
        return {
            "detector_name": self.detector_name,
            "protocol": self.protocol,
            "compression_algorithm": self.compression_algorithm,
            "compression_level": self.compression_level,
            "subsampling_factor": self.subsampling_factor,
            "throttle_ms": self.throttle_ms,
            "crop_size": self.crop_size,
            "jpeg_quality": self.jpeg_quality,
            "stun_servers": self.stun_servers,
            "turn_servers": self.turn_servers,
            "max_width": self.max_width,
            "broadcast_frames": self.broadcast_frames,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StreamParams':
        """Create StreamParams from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class _JpegEncoder:
    """JPEG encoder that prefers PyTurboJPEG, falls back to cv2.imencode.

    PyTurboJPEG is a thin Python wrapper over libjpeg-turbo. On a Pi 5
    (Cortex-A76 with NEON) it is typically 1.5-2x faster than
    ``cv2.imencode('.jpg', ...)`` at the same quality, because it avoids
    OpenCV's internal copy and goes straight to libjpeg-turbo's NEON
    paths. If the ``turbojpeg`` package is not installed, or its native
    library cannot be loaded, or a single encode call fails, the encoder
    falls back to ``cv2.imencode`` automatically.

    Reusing one instance per worker amortises the small TurboJPEG
    bootstrap cost across all frames.
    """

    def __init__(self, logger=None):
        self._logger = logger
        self._turbo = None
        self._fmt_bgr = None
        self._fmt_gray = None
        self._cv2 = None

        try:
            from turbojpeg import TurboJPEG, TJPF_BGR, TJPF_GRAY
            self._turbo = TurboJPEG()
            self._fmt_bgr = TJPF_BGR
            self._fmt_gray = TJPF_GRAY
            if logger is not None:
                logger.info("JPEG encoder: PyTurboJPEG (libjpeg-turbo)")
        except Exception as e:
            # ImportError, OSError (native lib missing), RuntimeError, etc.
            if logger is not None:
                logger.info(
                    f"PyTurboJPEG not available ({e!r}); "
                    f"using cv2.imencode fallback"
                )

        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            pass

    @property
    def backend(self) -> str:
        if self._turbo is not None:
            return "turbojpeg"
        if self._cv2 is not None:
            return "cv2"
        return "none"

    def encode(self, img: np.ndarray, quality: int) -> bytes:
        """Encode ``img`` (uint8, BGR or grayscale) as a JPEG byte string.

        Raises ``RuntimeError`` if no backend is available or both fail.
        """
        quality = max(1, min(100, int(quality)))

        if self._turbo is not None:
            try:
                if img.ndim == 2:
                    return self._turbo.encode(
                        img, quality=quality, pixel_format=self._fmt_gray
                    )
                else:
                    return self._turbo.encode(
                        img, quality=quality, pixel_format=self._fmt_bgr
                    )
            except Exception as e:
                if self._logger is not None:
                    self._logger.warning(
                        f"PyTurboJPEG encode failed ({e!r}); "
                        f"falling back to cv2 for the rest of this session."
                    )
                # Disable turbo for subsequent frames so we don't keep
                # paying for the failed-path overhead.
                self._turbo = None

        if self._cv2 is not None:
            ok, encoded = self._cv2.imencode(
                '.jpg', img,
                [int(self._cv2.IMWRITE_JPEG_QUALITY), quality],
            )
            if ok:
                return bytes(encoded)
            raise RuntimeError("cv2.imencode returned False")

        raise RuntimeError(
            "No JPEG encoder available (neither PyTurboJPEG nor cv2)"
        )


def apply_center_crop(frame: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Apply quadratic (square) center crop to frame.
    
    Args:
        frame: Input image (2D or 3D array)
        crop_size: Size of the square crop region. If 0 or >= min(height, width), returns original frame.
    
    Returns:
        Cropped frame
    """
    if crop_size <= 0:
        return frame

    height, width = frame.shape[:2]
    min_dim = min(height, width)

    # Clamp crop_size to image dimensions
    crop_size = min(crop_size, min_dim)

    # Calculate center crop coordinates
    center_y, center_x = height // 2, width // 2
    half_crop = crop_size // 2

    y_start = center_y - half_crop
    y_end = y_start + crop_size
    x_start = center_x - half_crop
    x_end = x_start + crop_size

    # Crop the frame
    if len(frame.shape) == 2:
        return frame[y_start:y_end, x_start:x_end]
    else:
        return frame[y_start:y_end, x_start:x_end, :]


class StreamWorker(Worker):
    """
    Base class for protocol-specific stream workers.
    Each worker runs in its own thread and waits for new frames without using a timer.
    This avoids skipping frames and ensures consistent frame rate.
    Workers do the actual encoding and emit encoded bytes that match socket.io message format.
    
    In headless mode, this worker is responsible for broadcasting frames to other controllers
    via sigUpdateFrame, which should be connected to CommunicationChannel.sigUpdateImage.
    """

    sigStreamFrame = Signal(dict)  # Emits pre-formatted message dict ready for socket.io emission
    sigUpdateFrame = Signal(str, np.ndarray, bool, bool, float, bool)  # (detectorName, image, init, scale, isCurrentDetector) - for broadcasting to other controllers
    def __init__(self, detectorManager, updatePeriodMs: int, streamParams: StreamParams):
        super().__init__()
        self._detector = detectorManager
        self._updatePeriod = updatePeriodMs / 1000.0  # Convert to seconds
        self._params = streamParams
        self._running = False
        self._thread = None
        self._logger = initLogger(self)
        self._last_frame_time = 0
        self._was_running = False

        # Frame broadcasting is disabled by default to avoid performance issues
        # Enable via enableFrameBroadcast() if controllers need frame updates
        self._broadcast_frames = True

        # Preallocated buffers for the uint16 → uint8 cast used by
        # subclasses' ``_u16_to_u8_no_alloc`` (§4.B.8). Sized to the most
        # recently seen post-crop+subsample shape; reallocated only when
        # the shape changes. ``None`` until first use.
        self._u8_buf: Optional[np.ndarray] = None
        self._u16_tmp: Optional[np.ndarray] = None

    def _u16_to_u8_no_alloc(self, src: np.ndarray) -> np.ndarray:
        """Convert ``src`` (uint16) → uint8 via ``>> 4`` without allocating.

        After §4.A the cast already runs on a small post-crop+subsample
        frame (~1 MB for a 4× subsampled 9 MP sensor), so the residual
        per-frame allocation is small. This helper removes it anyway by
        reusing two preallocated buffers across calls — useful when the
        worker is steady-state and Python's allocator would otherwise
        churn on the matched alloc/free pair.

        The returned buffer is **reused** across calls. Anything that
        needs to outlive the next ``_captureAndEmit`` iteration must copy
        it first (cv2/PyTurboJPEG encode produces a fresh ``bytes``
        object, so the workers' main paths are safe).
        """
        if (self._u8_buf is None
                or self._u8_buf.shape != src.shape):
            self._u8_buf = np.empty(src.shape, dtype=np.uint8)
            self._u16_tmp = np.empty(src.shape, dtype=np.uint16)
        # Two passes over the data, zero allocations. ``np.right_shift``
        # requires matching dtypes for ``out``, hence the u16 scratch
        # buffer plus ``np.copyto`` with ``casting='unsafe'`` for the
        # truncation to u8.
        np.right_shift(src, 4, out=self._u16_tmp)
        np.copyto(self._u8_buf, self._u16_tmp, casting='unsafe')
        return self._u8_buf

    def run(self):
        """Start polling frames without timer - wait and push immediately."""
        self._running = True
        self._logger.info(f"StreamWorker started with update period {self._updatePeriod}s")
        frameReadAttempts = 0
        last_detector_frame_number = -1
        while self._running:
            try:
                self._updatePeriod = self._params.throttle_ms / 1000.0  # Update in case params changed # TODO: This is a weird place to change it
                # Check if enough time has passed since last frame
                if (time.time() - self._last_frame_time) >= self._updatePeriod:  # TODO: and  imswitch.__is_stream_ready_for_sending__
                    # Capture and emit frame

                    # Get frame with actual detector frame number
                    result = self._detector.getLatestFrame(returnFrameNumber=True)
                    if isinstance(result, tuple) and len(result) == 2:
                        frame, detector_frame_number = result
                    else:
                        frame = result
                        detector_frame_number = None

                    # Broadcast frame to other controllers (HistogrammController, InLineHoloController, etc.)
                    # This is DISABLED by default for performance - enable via enableFrameBroadcast()
                    # Controllers that need frames should use getCachedFrame() or subscribe explicitly
                    if frame is not None and self._broadcast_frames:
                        self.sigUpdateFrame.emit(self._detector.name, frame, False, False, self._detector.pixelSizeUm, True) # (str, np.ndarray, bool, bool, float, bool)

                    if (frame is None and last_detector_frame_number == detector_frame_number) and frameReadAttempts > 3:
                        self._logger.warning("Frame capture failed, stopping worker (Frame none or no new frame: {})".format(detector_frame_number==last_detector_frame_number))
                        self._running = False
                        break  # No frame available, but not an error - keep running
                    else:
                        frameReadAttempts = 0  # Reset attempts on successful read
                    
                    last_detector_frame_number = detector_frame_number

                    frameResult = self._captureAndEmit(frame, detector_frame_number)
                    self._last_frame_time = time.time()

                    # Only stop on explicit failure, not on None frames
                    if frameResult is False and frameReadAttempts > 3:  # Explicitly check for False, not None
                        self._logger.warning("Frame capture failed, stopping worker")
                        self._running = False
                        break
                    else:
                        frameReadAttempts = 0  # Reset attempts on successful read
                        
                    frameReadAttempts += 1
                else:
                    # Sleep for a small amount to avoid busy waiting
                    time.sleep(0.01)

            except Exception as e:
                self._logger.error(f"Error in StreamWorker loop: {e}")
                time.sleep(0.1)  # Brief pause on error
                # Continue running unless explicitly stopped

        self._logger.info("StreamWorker run loop exited")

    def stop(self):
        """Stop polling frames."""
        self._running = False
        self._logger.info("StreamWorker stopped")

    def enableFrameBroadcast(self, enable: bool = True):
        """
        Enable or disable frame broadcasting to other controllers.
        
        When enabled, sigUpdateFrame is emitted for each captured frame,
        allowing controllers like HistogrammController to receive updates.
        
        WARNING: This can significantly impact performance if many controllers
        are connected. Consider using getCachedFrame() instead for on-demand access.
        
        Args:
            enable: True to enable broadcasting, False to disable
        """
        self._broadcast_frames = enable
        self._logger.info(f"Frame broadcasting {'enabled' if enable else 'disabled'}")

    @abstractmethod
    def _captureAndEmit(self, frame, detector_frame_number=None):
        """Capture frame from detector, encode it, and emit processed data."""
        pass


class BinaryStreamWorker(StreamWorker):
    """Worker for binary frame streaming with LZ4/Zstandard compression."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_id = 0  # Unified frame counter (will sync with detector frame number)
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

    def _captureAndEmit(self, frame, detector_frame_number=None):
        """Capture frame from detector, compress it, and emit pre-formatted socket.io message."""
        try:

            # Use detector frame number if available, otherwise use our counter
            if detector_frame_number is not None:
                self._frame_id = detector_frame_number
            else:
                self._frame_id = (self._frame_id + 1) % 65536  # Handle rollover at 16-bit boundary

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

            # Apply center crop if specified (before subsampling)
            if self._params.crop_size > 0:
                frame = apply_center_crop(frame, self._params.crop_size)

            # Get encoder
            encoder = self._get_encoder()
            if encoder is None:
                self._logger.error("Failed to get encoder")
                return False  # This is a real error

            # Encode frame
            packet, encoding_metadata = encoder.encode_frame(frame)

            # Create unified metadata structure
            metadata = {
                'server_timestamp': time.time(),
                'frame_id': self._frame_id,  # Unified field name
                'detector_name': detector_name,
                'pixel_size': float(pixel_size),
                'format': 'binary',
                'protocol': 'binary'
            }
            # Merge encoding metadata (compression info, etc.)
            metadata.update(encoding_metadata)

            # Create pre-formatted message for socket.io
            message = {
                'type': 'binary_frame',
                'event': 'frame',
                'data': packet,
                'metadata': metadata
            }

            # Emit pre-formatted message
            self.sigStreamFrame.emit(message)

            return True

        except Exception as e:
            self._logger.error(f"Error in BinaryStreamWorker: {e}")
            return False  # Real error - stop worker


class JPEGStreamWorker(StreamWorker):
    """Worker for JPEG frame streaming with compression."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_id = 0  # Unified frame counter (will sync with detector frame number)
        try:
            import cv2
            self._cv2 = cv2
        except ImportError:
            self._logger.error("opencv-python required for JPEG streaming")
            self._cv2 = None
        # Prefer PyTurboJPEG; falls back to cv2.imencode automatically.
        self._jpeg = _JpegEncoder(logger=self._logger)
        if self._jpeg.backend == "none" and self._cv2 is None:
            self._logger.error(
                "No JPEG encoder available (neither PyTurboJPEG nor cv2)."
            )

    def _captureAndEmit(self, frame, detector_frame_number=None):
        """Capture frame from detector, encode as JPEG, and emit pre-formatted socket.io message."""
        if self._cv2 is None:
            return False  # This is a configuration error

        try:
            # Use detector frame number if available, otherwise use our counter
            if detector_frame_number is not None:
                self._frame_id = detector_frame_number
            else:
                self._frame_id = (self._frame_id + 1) % 65536  # Handle rollover at 16-bit boundary

            # Get detector info
            detector_name = self._detector.name
            pixel_size = 1.0
            try:
                pixel_size = self._detector.pixelSizeUm[-1]
            except:
                pass

            if frame is None:
                self._logger.warning("Received None frame, skipping")
                return None  # Not an error, but skip processing

            # Order matters for large sensors: shrink the frame BEFORE
            # converting dtype, so the dtype cast runs on the small frame,
            # not the full sensor. ``crop_size`` is interpreted in sensor
            # pixels (unchanged semantics from the previous implementation).
            #
            # 1) Crop (slice view, no copy).
            if self._params.crop_size > 0:
                frame = apply_center_crop(frame, self._params.crop_size)

            # 2) Subsample (stride view, no copy).
            if self._params.subsampling_factor > 1:
                frame = frame[::self._params.subsampling_factor, ::self._params.subsampling_factor]

            # 3) Convert uint16 → uint8 via right-shift, into a preallocated
            # buffer (see StreamWorker._u16_to_u8_no_alloc). Avoids both the
            # float64 intermediate the old ``frame / 16`` produced AND the
            # per-frame alloc that ``(frame >> 4).astype(np.uint8)`` does.
            if frame.dtype == np.uint16:
                frame = self._u16_to_u8_no_alloc(frame)

            # 4) Ensure C-contiguous for the encoder. The preallocated
            # buffer above is contiguous, so this is a no-op on the u16
            # path; it covers the uint8-input path where step 2 produced
            # a strided view.
            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)

            # 5) Convert RGB → BGR for JPEG encoding (both libjpeg-turbo and
            # OpenCV expect BGR for 3-channel input).
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                if self._cv2 is not None:
                    frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
                else:
                    # No cv2 — flip channel order in NumPy.
                    frame = np.ascontiguousarray(frame[..., ::-1])

            # 6) Encode as JPEG via _JpegEncoder (PyTurboJPEG → cv2 fallback).
            try:
                jpeg_bytes = self._jpeg.encode(frame, self._params.jpeg_quality)
                success = True
            except Exception as enc_err:
                self._logger.warning(f"JPEG encode failed: {enc_err!r}")
                jpeg_bytes = b''
                success = False

            if success:
                # Emit raw JPEG bytes. The downstream socket.io path
                # (noqt.py::_handle_stream_frame, 'jpeg_frame' branch) packs
                # this with msgpack using ``use_bin_type=True``, which serialises
                # ``bytes`` as a MessagePack ``bin`` value (Uint8Array on the
                # client). Previously we base64-encoded these bytes into a
                # JSON-safe string, which inflated the payload ~33% and cost
                # ~5-10 ms per frame on the Pi 5. The React frontend must wrap
                # the received Uint8Array as a Blob/ObjectURL instead of using
                # a ``data:image/jpeg;base64,...`` URI.

                # Create unified metadata structure
                metadata = {
                    'server_timestamp': time.time(),
                    'frame_id': self._frame_id,  # Unified field name
                    'detector_name': detector_name,
                    'pixel_size': float(pixel_size),
                    'format': 'jpeg',
                    'protocol': 'jpeg',
                    'jpeg_quality': self._params.jpeg_quality
                }

                # Create pre-formatted message for socket.io
                # Use unified 'frame' event for consistency with binary
                message = {
                    'type': 'jpeg_frame',
                    'event': 'frame',
                    'data': {
                        'image': jpeg_bytes,
                        'metadata': metadata
                    }
                }

                # Emit pre-formatted message
                self.sigStreamFrame.emit(message)

                return True
            else:
                self._logger.warning("JPEG encoding failed")
                return None  # Encoding failed, but keep trying

        except Exception as e:
            self._logger.error(f"Error in JPEGStreamWorker: {e}")
            return False  # Real error - stop worker


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
        # Prefer PyTurboJPEG; falls back to cv2.imencode automatically.
        self._jpeg = _JpegEncoder(logger=self._logger)
        if self._jpeg.backend == "none" and self._cv2 is None:
            self._logger.error(
                "No JPEG encoder available (neither PyTurboJPEG nor cv2)."
            )

    def _captureAndEmit(self, frame, detector_frame_number=None):
        """Capture frame and put in queue for MJPEG streaming."""
        if self._cv2 is None and self._jpeg.backend == "none":
            return False  # This is a configuration error

        try:
            if frame is None:
                self._logger.warning("Received None frame, skipping")
                return None

            # Same ordering as JPEGStreamWorker: shrink first, then convert
            # dtype, so the cast runs on the small frame. The previous
            # ``np.min / np.max → float → cast`` did 4+ full-frame passes in
            # float64 — the worst case for large sensors. We use the same
            # fixed right-shift as JPEGStreamWorker for consistency. This
            # trades per-frame auto-stretch for ~10x lower CPU; if
            # auto-stretch is needed in MJPEG specifically, compute
            # vmin/vmax on the already-subsampled thumbnail (step 2 below).

            # 1) Crop (view).
            if self._params.crop_size > 0:
                frame = apply_center_crop(frame, self._params.crop_size)

            # 2) Subsample (view). The MJPEG worker previously skipped this
            # entirely, encoding the full sensor every frame — single biggest
            # win on this path.
            if self._params.subsampling_factor > 1:
                frame = frame[::self._params.subsampling_factor, ::self._params.subsampling_factor]

            # 3) Convert to uint8.
            if frame.dtype == np.uint16:
                # Preallocated-buffer right-shift; matches JPEGStreamWorker.
                frame = self._u16_to_u8_no_alloc(frame)
            elif frame.dtype != np.uint8:
                # Float or other dtype: rare path. Keep the previous min/max
                # stretch, but it now runs on the already-cropped+subsampled
                # frame (≪ 1/16 the cost) instead of the full sensor.
                vmin = float(frame.min())
                vmax = float(frame.max())
                if vmax > vmin:
                    frame = ((frame - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                else:
                    frame = np.zeros(frame.shape, dtype=np.uint8)

            # 4) Ensure C-contiguous for cv2.imencode.
            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)

            # 5) Convert RGB → BGR for JPEG encoding (both libjpeg-turbo and
            # OpenCV expect BGR for 3-channel input).
            if len(frame.shape) == 3 and frame.shape[2] == 3:  # TODO: Is this correct for all RGB cameras?
                if self._cv2 is not None:
                    frame = self._cv2.cvtColor(frame, self._cv2.COLOR_RGB2BGR)
                else:
                    frame = np.ascontiguousarray(frame[..., ::-1])

            # 6) Encode as JPEG via _JpegEncoder (PyTurboJPEG → cv2 fallback).
            try:
                jpeg_bytes = self._jpeg.encode(frame, self._params.jpeg_quality)
                success = True
            except Exception as enc_err:
                self._logger.warning(f"MJPEG encode failed: {enc_err!r}")
                jpeg_bytes = b''
                success = False

            if success:
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
                return True
            else:
                self._logger.warning("MJPEG encoding failed")
                return None  # Encoding failed, but keep trying

        except Exception as e:
            self._logger.error(f"Error in MJPEGStreamWorker: {e}")
            return False  # Real error - stop worker

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

    def _captureAndEmit(self, frame, detector_frame_number=None):
        """Capture frame, prepare it fully for av.VideoFrame, and enqueue.

        All the expensive preprocessing (dtype cast, crop, subsample,
        grayscale→RGB, target-size resize, even-dim alignment) runs *here*
        on the StreamWorker thread instead of inside ``recv()`` on the
        aiortc event loop. The queue therefore holds frames that are
        rgb24, contiguous, even-dimensioned, and already at the target
        size — ``recv()`` only needs to wrap them in ``av.VideoFrame``.
        """
        if not self._has_webrtc:
            return False  # This is a configuration error

        try:
            if frame is None:
                return None

            # 1) Crop (view).
            if self._params.crop_size > 0:
                frame = apply_center_crop(frame, self._params.crop_size)

            # 2) Subsample via stride slice (view; cheap).
            sub = max(1, int(getattr(self._params, "subsampling_factor", 1)))
            if sub > 1:
                frame = frame[::sub, ::sub]

            # 3) uint16 → uint8. Preallocated-buffer right-shift; matches
            # JPEG/MJPEG workers. (For genuinely 16-bit data this
            # discards 4 low bits — fine for a lossy WebRTC codec.)
            if frame.dtype == np.uint16:
                frame = self._u16_to_u8_no_alloc(frame)
            elif frame.dtype != np.uint8:
                # Rare path: float or other dtype. Run min/max stretch on
                # the already-cropped+subsampled frame, not the full sensor.
                vmin = float(frame.min())
                vmax = float(frame.max())
                if vmax > vmin:
                    frame = ((frame - vmin) / (vmax - vmin) * 255.0).astype(np.uint8)
                else:
                    frame = np.zeros(frame.shape, dtype=np.uint8)

            # 4) Make sure it's RGB-channels-last for av.VideoFrame("rgb24").
            if frame.ndim == 2:
                # Grayscale → RGB via broadcasting (no extra alloc beyond
                # the final contiguous copy below).
                frame = np.repeat(frame[..., None], 3, axis=2)
            elif frame.ndim == 3 and frame.shape[2] == 1:
                frame = np.repeat(frame, 3, axis=2)
            elif frame.ndim == 3 and frame.shape[2] == 4:
                frame = frame[:, :, :3]

            # 5) Target size: respect ``max_width`` and a hard 720p cap so
            # software libx264 inside aiortc can keep up on a Pi 5. This
            # used to run inside ``recv()`` — moving it here means the
            # asyncio loop only does the lightweight VideoFrame wrap.
            orig_h, orig_w = frame.shape[:2]
            max_width = int(getattr(self._params, "max_width", 1280)) or orig_w
            # Hard cap to 720p — pi5 software h264 can't handle more
            # at any reasonable framerate.
            hard_w_cap = 1280
            target_w = min(max_width, hard_w_cap)

            if orig_w > target_w:
                scale = target_w / orig_w
                new_w = int(orig_w * scale)
                new_h = int(orig_h * scale)
                # Even dims (codec requirement).
                new_w -= new_w % 2
                new_h -= new_h % 2
                try:
                    import cv2
                    frame = cv2.resize(
                        frame, (new_w, new_h), interpolation=cv2.INTER_AREA
                    )
                except ImportError:
                    # No cv2: stride-subsample down by a power of 2 as a
                    # last resort. Loses aspect-ratio precision but
                    # functional.
                    step = max(1, orig_w // new_w)
                    frame = frame[::step, ::step]

            # 6) Strip a row/col if necessary to land on even dims.
            h, w = frame.shape[:2]
            if w % 2:
                frame = frame[:, : w - 1]
            if h % 2:
                frame = frame[: h - 1, :]

            # 7) Contiguous, ready for av.VideoFrame.from_ndarray("rgb24").
            if not frame.flags.c_contiguous:
                frame = np.ascontiguousarray(frame)

            # Single-slot semantics: drop any stale frame, push the new
            # one. Lowest latency at the cost of dropping intermediate
            # frames when the consumer is slow — exactly what WebRTC
            # wants for live preview.
            while not self._frame_queue.empty():
                try:
                    self._frame_queue.get_nowait()
                except queue.Empty:
                    break
            try:
                self._frame_queue.put_nowait(frame)
                return True
            except queue.Full:
                return None

        except Exception as e:
            self._logger.error(f"Error in WebRTCStreamWorker: {e}")
            return False  # Real error - stop worker

    def get_video_track(self):
        """Get or create video track for WebRTC."""
        if self._video_track is None and self._has_webrtc:
            track_wrapper = DetectorVideoTrack(
                self._frame_queue,
                self._detector.name,
                self._VideoStreamTrack,
                self._av,
                self._params  # Pass stream parameters
            )
            # Return the actual track, not the wrapper
            self._video_track = track_wrapper._track
        return self._video_track


class DetectorVideoTrack:
    """Custom video track that reads frames from detector queue."""

    def __init__(self, frame_queue, detector_name, VideoStreamTrack, av, stream_params=None):
        self._queue = frame_queue
        self._detector_name = detector_name
        self._av = av
        self._stream_params = stream_params or StreamParams(protocol='webrtc')
        self._logger = initLogger(self)

        # Create custom track class that inherits from VideoStreamTrack
        class CustomVideoTrack(VideoStreamTrack):
            def __init__(inner_self):
                super().__init__()
                inner_self.kind = "video"
                inner_self._queue = frame_queue
                inner_self._av = av
                inner_self._timestamp = 0
                inner_self._stream_params = stream_params or StreamParams(protocol='webrtc')
                inner_self._logger = initLogger(inner_self)
                # time_base must be a Fraction, not a float
                from fractions import Fraction
                inner_self._time_base = Fraction(1, 30)  # 30 fps

            async def recv(inner_self):
                """Receive next video frame."""

                # Try to get frame from queue with shorter timeout for faster response
                frame = None
                start_time = asyncio.get_event_loop().time()
                timeout = 0.1  # Reduced from 0.5 to 0.1 seconds for faster response

                # Try to get the latest frame immediately
                try:
                    frame = inner_self._queue.get_nowait()
                except queue.Empty:
                    # If no frame available, wait briefly
                    for _ in range(3):  # Max 3 attempts, 30ms total
                        await asyncio.sleep(0.01)
                        try:
                            frame = inner_self._queue.get_nowait()
                            break
                        except queue.Empty:
                            continue

                if frame is None:
                    # Use last frame if available, otherwise create a small placeholder frame.
                    if hasattr(inner_self, '_last_frame') and inner_self._last_frame is not None:
                        frame = inner_self._last_frame  # already prepared by producer
                    else:
                        # Create a small black frame to reduce bandwidth.
                        frame = np.zeros((240, 320, 3), dtype=np.uint8)
                        inner_self._logger.info("Using placeholder frame")
                else:
                    # Frames coming out of the queue are already rgb24,
                    # contiguous, even-dimensioned, and sized to the WebRTC
                    # target by WebRTCStreamWorker._captureAndEmit. We just
                    # need to wrap them. Keep one as a fallback for the
                    # next empty-queue tick.
                    inner_self._last_frame = frame

                # Defensive sanity checks — cheap and protect against an
                # accidental upstream change. If any of these trigger
                # we'd rather pay a one-frame cv2 hit than send a malformed
                # buffer to libx264.
                if frame.ndim == 2:
                    frame = np.repeat(frame[..., None], 3, axis=2)
                elif frame.ndim == 3 and frame.shape[2] == 1:
                    frame = np.repeat(frame, 3, axis=2)
                elif frame.ndim == 3 and frame.shape[2] == 4:
                    frame = frame[:, :, :3]
                if not frame.flags.c_contiguous:
                    frame = np.ascontiguousarray(frame)
                h, w = frame.shape[:2]
                if (w % 2) or (h % 2):
                    frame = frame[: h - (h % 2), : w - (w % 2)]

                try:
                    new_frame = inner_self._av.VideoFrame.from_ndarray(frame, format="rgb24")
                    new_frame.pts = inner_self._timestamp
                    new_frame.time_base = inner_self._time_base
                    inner_self._timestamp += 1
                    return new_frame
                except Exception as e:
                    inner_self._logger.warning(f"Error creating av.VideoFrame: {e}, using fallback frame")
                    # If frame creation fails, create a minimal frame
                    fallback_frame = np.zeros((240, 320, 3), dtype=np.uint8)
                    new_frame = inner_self._av.VideoFrame.from_ndarray(fallback_frame, format="rgb24")
                    new_frame.pts = inner_self._timestamp
                    new_frame.time_base = inner_self._time_base
                    inner_self._timestamp += 1
                    return new_frame

        self._track = CustomVideoTrack()

    def __getattr__(self, name):
        """Delegate attribute access to the underlying track."""
        return getattr(self._track, name)


class WebRTCOfferRequest(BaseModel):
    """Request model for WebRTC offer."""
    sdp: str
    sdp_type: str  # Should be "offer"
    detectorName: Optional[str] = None
    params: Optional[Dict[str, Any]] = None  # Stream parameters


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
        self._streamLock = threading.Lock()  # Protects _activeStreams and _streamThreads

        # WebRTC peer connections: {detectorName: RTCPeerConnection}
        self._webrtc_peers: Dict[str, Any] = {}
        self._webrtc_loop = None
        self._webrtc_loop_thread = None

        # Global stream parameters per protocol (defaults)
        self._streamParams: Dict[str, StreamParams] = {
            'binary': StreamParams(protocol='binary'),
            'jpeg': StreamParams(protocol='jpeg'),
            'mjpeg': StreamParams(protocol='mjpeg'),
            'webrtc': StreamParams(protocol='webrtc'),
        }

        # Per-detector stream parameters: {detectorName: StreamParams}
        # Stores the last-used params for each detector so they survive detector switches.
        self._detectorParams: Dict[str, StreamParams] = {}

        # Load per-detector defaults from the setup config (DetectorInfo.defaultStreamSettings).
        # These are *fixed* defaults — the frontend can override them for the
        # current session via setStreamParameters but cannot persist changes
        # back to the config. They are seeded into _detectorParams here so
        # the first startLiveView() call uses them.
        self._loadDetectorDefaultsFromConfig()

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

    def _loadDetectorDefaultsFromConfig(self) -> None:
        """Seed per-detector stream params from ``DetectorInfo.defaultStreamSettings``.

        Called once during ``__init__``. Each detector's defaultStreamSettings
        dict (from setup JSON) is merged on top of the global protocol
        defaults and stored in ``_detectorParams[detectorName]``. Any later
        ``startLiveView`` call without explicit params will use this entry,
        giving each camera its own sensible default protocol + subsampling.
        default dict structure example:
        "defaultStreamSettings": {
            "protocol": "jpeg",
            "jpeg_quality": 80,
            "subsampling_factor": 4,
            "throttle_ms": 50,
            "broadcast_frames": false   // optional; default true. Set false
                                        // on large sensors to skip the
                                        // per-frame fan-out to Histogram/
                                        // FFT/Holo/RecordingService.
        }
        """
        detector_infos = {}
        try:
            if hasattr(self, '_setupInfo') and self._setupInfo is not None:
                detector_infos = getattr(self._setupInfo, 'detectors', None) or {}
        except Exception:
            detector_infos = {}

        for detector_name, info in detector_infos.items():
            try:
                defaults = getattr(info, 'defaultStreamSettings', None) or {}
                if not isinstance(defaults, dict) or not defaults:
                    continue
                protocol = str(defaults.get('protocol') or 'binary').lower()
                if protocol not in self._streamParams:
                    self._logger.warning(
                        f"Detector {detector_name}: unknown defaultStreamSettings.protocol "
                        f"'{protocol}', falling back to 'binary'."
                    )
                    protocol = 'binary'

                # Start from the global protocol baseline so we don't drop
                # required fields, then layer the detector-specific defaults
                # on top.
                base = self._streamParams[protocol].to_dict()
                base.update({k: v for k, v in defaults.items() if k != 'protocol'})
                base['protocol'] = protocol

                # StreamParams.from_dict is lenient about unknown keys.
                try:
                    params = StreamParams.from_dict(base)
                except Exception:
                    params = StreamParams(protocol=protocol)
                    for k, v in defaults.items():
                        if k != 'protocol' and hasattr(params, k):
                            setattr(params, k, v)

                self._detectorParams[detector_name] = params
                self._logger.info(
                    f"Detector {detector_name}: loaded default stream settings "
                    f"protocol={protocol}, subsampling={getattr(params, 'subsampling_factor', '?')}"
                )
            except Exception as e:
                self._logger.warning(
                    f"Failed to load defaultStreamSettings for detector {detector_name}: {e}"
                )

    def _get_or_create_webrtc_loop(self):
        """Get or create a persistent event loop for WebRTC in a separate thread."""
        if self._webrtc_loop is None or not self._webrtc_loop.is_running():
            import asyncio

            def run_loop():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                self._webrtc_loop = loop
                self._logger.info("WebRTC event loop started")
                loop.run_forever()

            self._webrtc_loop_thread = threading.Thread(target=run_loop, daemon=True)
            self._webrtc_loop_thread.start()

            # Wait for loop to be ready
            import time
            max_wait = 2.0
            start = time.time()
            while self._webrtc_loop is None and (time.time() - start) < max_wait:
                time.sleep(0.01)

        return self._webrtc_loop

    def _stop_webrtc_loop(self):
        """Stop the WebRTC event loop."""
        if self._webrtc_loop and self._webrtc_loop.is_running():
            self._webrtc_loop.call_soon_threadsafe(self._webrtc_loop.stop)
            if self._webrtc_loop_thread:
                self._webrtc_loop_thread.join(timeout=1.0)
            self._webrtc_loop = None
            self._webrtc_loop_thread = None
            self._logger.info("WebRTC event loop stopped")

    def _onStartLiveAcquisition(self, start: bool):
        """Handle start live acquisition signal."""
        if start:
            self._logger.info("Received start live acquisition signal")
            # This can be used to automatically start default streaming
            # For now, streaming is explicitly controlled via API
        else:
            self._logger.info("Received stop live acquisition signal")

    def _onStopLiveAcquisition(self, stop: bool):
        """Handle stop live acquisition signal."""
        if stop:
            self._logger.info("Stopping all live acquisitions")
            # Stop all active streams (take snapshot of keys under lock)
            with self._streamLock:
                detector_names = list(self._activeStreams.keys())
            for detector_name in detector_names:
                self.stopLiveView(detector_name, stopCamera=False)

    @APIExport()
    def getLiveViewActive(self) -> bool:
        """Check if any live view stream is currently active."""

        return bool(len(self._activeStreams) > 0)

    @APIExport(requestType="POST")
    def startLiveView(self, detectorName: Optional[str] = None, protocol: str = "jpeg",
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
            
        Example:
            -> binary
            params: {"compression_algorithm": "lz4", "compression_level": 0, "subsampling_factor": 4, "throttle_ms": 50}
            -> jpeg
            params: {"jpeg_quality": 80, "subsampling_factor": 4, "throttle_ms": 50}
            -> mjpeg
            params: {"jpeg_quality": 80, "throttle_ms": 50}
            -> webrtc
            params: {"stun_servers": ["stun:stun.l.google.com:19302"], "turn_servers": []}
        """
        try:
            # Get detector
            if detectorName is None:
                detectorName = self._master.detectorsManager.getAllDeviceNames()[0]
            detector = self._master.detectorsManager[detectorName]

            # ensure the detector is actually started
            if not detector._running:
                detector.startAcquisition()

            with self._streamLock:
                # Check if detector already has an active stream
                if detectorName in self._activeStreams:
                    old_protocol, old_worker = self._activeStreams[detectorName]
                    return {
                        "status": "already_running",
                        "detector": detectorName,
                        "protocol": old_protocol,
                        "message": f"Stream already active for {detectorName} with protocol {old_protocol}. Stop it first."
                    }

            # Resolve effective stream parameters. Priority:
            # 1. Explicit params from this call
            # 2. Previously saved per-detector params (matching the requested protocol)
            # 3. Global protocol defaults
            global_params = self._streamParams.get(protocol, StreamParams(protocol=protocol))
            saved_detector = self._detectorParams.get(detectorName)

            if params:
                # Caller provided explicit overrides — start from global and apply
                stream_params = StreamParams(**global_params.to_dict())
                for key, value in params.items():
                    if hasattr(stream_params, key):
                        setattr(stream_params, key, value)
            elif saved_detector is not None and saved_detector.protocol == protocol:
                # Re-use the params from the last time this detector was streamed
                # with the same protocol
                stream_params = StreamParams(**saved_detector.to_dict())
            elif saved_detector is not None and saved_detector.protocol != protocol:
                # Detector has saved params but for a different protocol —
                # start from global defaults but keep detector-specific settings
                # like subsampling_factor, throttle_ms that are protocol-agnostic
                stream_params = StreamParams(**global_params.to_dict())
                # Carry over common detector-specific settings
                for common_key in ('subsampling_factor', 'throttle_ms', 'crop_size'):
                    if hasattr(saved_detector, common_key):
                        setattr(stream_params, common_key, getattr(saved_detector, common_key))
            else:
                stream_params = StreamParams(**global_params.to_dict())

            stream_params.detector_name = detectorName
            stream_params.protocol = protocol

            # Persist effective params for this detector
            self._detectorParams[detectorName] = StreamParams(**stream_params.to_dict())

            # Create appropriate worker
            worker = self._createWorker(detector, protocol, stream_params)
            if worker is None:
                return {
                    "status": "error",
                    "message": f"Failed to create worker for protocol {protocol}"
                }

            # Connect worker signal to controller's signal, which is then handled by noqt
            # The worker emits pre-formatted messages ready for socket.io emission
            worker.sigStreamFrame.connect(self._commChannel.sigUpdateStreamFrame)

            # Frame broadcasting (sigUpdateImage fan-out to every controller's
            # update() — Histogram, FFT, Holo, RecordingService for streaming
            # recordings, ImageController, …) is gated by
            # ``stream_params.broadcast_frames``. Default True preserves prior
            # behaviour; set ``"broadcast_frames": false`` in a detector's
            # ``defaultStreamSettings`` (setup JSON) to turn it off for big
            # sensors where the per-frame fan-out cost is dominant.
            worker.sigUpdateFrame.connect(self._commChannel.sigUpdateImage)
            worker.enableFrameBroadcast(bool(stream_params.broadcast_frames))

            # Start worker in thread
            thread = threading.Thread(target=worker.run, daemon=True)
            thread.start()

            with self._streamLock:
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
    def stopLiveView(self, detectorName: Optional[str] = None, stopCamera: bool=True) -> Dict[str, Any]:
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
            with self._streamLock:
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

            # Stop worker (outside lock — worker.stop() may take time)
            worker.stop()

            # If it's WebRTC, close the peer connection
            if protocol == "webrtc" and detectorName in self._webrtc_peers:
                import asyncio
                loop = self._get_or_create_webrtc_loop()

                async def close_pc():
                    pc = self._webrtc_peers[detectorName]
                    await pc.close()
                    del self._webrtc_peers[detectorName]
                    self._logger.info(f"Closed WebRTC peer connection for {detectorName}")

                # Schedule close on the event loop
                future = asyncio.run_coroutine_threadsafe(close_pc(), loop)
                try:
                    future.result(timeout=5.0)
                except Exception as e:
                    self._logger.error(f"Error closing WebRTC peer: {e}")

            with self._streamLock:
                # Clean up
                if detectorName in self._activeStreams:
                    del self._activeStreams[detectorName]
                if detectorName in self._streamThreads:
                    del self._streamThreads[detectorName]

            # Emit signal
            self.sigStreamStopped.emit(detectorName, protocol)

            self._logger.info(f"Stopped {protocol} stream for detector {detectorName}")

            # Optionally stop camera acquisition
            if stopCamera:
                detector = self._master.detectorsManager[detectorName]
                detector.stopAcquisition()
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

    def detectorIsRunning(self, detectorName: str=None) -> bool:
        """Check if a detector is currently running acquisition."""
        if detectorName is None:
            detectorName = self._master.detectorsManager.getAllDeviceNames()[0]
        detector = self._master.detectorsManager[detectorName]
        return detector._running

    @APIExport(requestType="POST")
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

            # Update global parameters
            stream_params = self._streamParams[protocol]
            for key, value in params.items():
                if hasattr(stream_params, key):
                    setattr(stream_params, key, value)
                    print(f"Set {protocol} param {key} to {value}")
            # Restart active streams that use the updated protocol
            detectors_to_restart = []
            for detector_name, (active_protocol, worker) in self._activeStreams.items():
                if active_protocol == protocol:
                    detectors_to_restart.append(detector_name)

            # Restart streams with updated parameters
            restarted_streams = []
            for detector_name in detectors_to_restart:
                self._logger.info(f"Restarting {protocol} stream for {detector_name} with updated parameters")
                # Stop the current stream
                self.stopLiveView(detectorName=detector_name, stopCamera=False)

                # Start with new parameters
                result = self.startLiveView(detector_name, protocol, params)
                if result['status'] == 'success':
                    restarted_streams.append(detector_name)


            response = {
                "status": "success",
                "protocol": protocol,
                "params": stream_params.to_dict()
            }

            if restarted_streams:
                response["restarted_detectors"] = restarted_streams
                response["message"] = f"Parameters updated and {len(restarted_streams)} stream(s) restarted"
            else:
                response["message"] = "Parameters updated (no active streams to restart)"

            return response

        except Exception as e:
            self._logger.error(f"Error setting stream params: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    @APIExport()
    def getStreamParameters(self, protocol: Optional[str] = None, detectorName: Optional[str] = None) -> Dict[str, Any]:
        """
        Get current streaming parameters.
        
        Args:
            protocol: Optional protocol to get params for (None = all protocols)
            detectorName: Optional detector to get current protocol for (None = first active)
        
        Returns:
            Dictionary with streaming parameters
            
        Example return:
        {
            "status": "success",
            "current_active_protocols": {"detector1": "binary", "detector2": "jpeg"},
            "requested_protocol": "binary",
            "protocols": {
                "binary": {"detector_name": null, "protocol": "binary", ...},
                "jpeg": {"detector_name": null, "protocol": "jpeg", ...}
            }
        }
        """
        try:
            # Build active protocols mapping
            active_protocols = {
                det_name: prot for det_name, (prot, worker) in self._activeStreams.items()
            }

            if protocol:
                # Return specific protocol parameters
                if protocol in self._streamParams:
                    # Check which detectors are using this protocol
                    active_detectors = [det for det, prot in active_protocols.items() if prot == protocol]

                    return {
                        "status": "success",
                        "protocol": protocol,
                        "params": self._streamParams[protocol].to_dict(),
                        "active_detectors": active_detectors,
                        "is_active": len(active_detectors) > 0
                    }
                else:
                    return {
                        "status": "error",
                        "message": f"Unknown protocol: {protocol}"
                    }
            else:
                # Return all protocols with current status
                current_protocol = None
                if detectorName and detectorName in active_protocols:
                    current_protocol = active_protocols[detectorName]
                elif active_protocols:
                    # Use first active protocol if no detector specified
                    current_protocol = next(iter(active_protocols.values()))

                return {
                    "status": "success",
                    "current_active_protocols": active_protocols,
                    "current_protocol": current_protocol,
                    "total_active_streams": len(active_protocols),
                    "protocols": {
                        p: {
                            **params.to_dict(),
                            "active_detectors": [det for det, prot in active_protocols.items() if prot == p],
                            "is_active": any(prot == p for prot in active_protocols.values())
                        }
                        for p, params in self._streamParams.items()
                    },
                    "per_detector": {
                        det: dp.to_dict() for det, dp in self._detectorParams.items()
                    }
                }
        except Exception as e:
            self._logger.error(f"Error getting stream params: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    @APIExport(requestType="POST")
    def setDetectorStreamParameters(self, detectorName: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Set stream parameters for a specific detector.
        These override the global protocol defaults when the detector is next started.
        If the detector is currently streaming, the stream is restarted with the new params.
        """
        try:
            saved = self._detectorParams.get(detectorName)
            if saved is None:
                saved = StreamParams()
            for key, value in params.items():
                if hasattr(saved, key):
                    setattr(saved, key, value)
            self._detectorParams[detectorName] = saved

            # Restart if currently active
            if detectorName in self._activeStreams:
                protocol, _ = self._activeStreams[detectorName]
                self.stopLiveView(detectorName=detectorName, stopCamera=False)
                result = self.startLiveView(detectorName, protocol, params)
                return {
                    "status": "success",
                    "detector": detectorName,
                    "params": saved.to_dict(),
                    "restarted": True,
                    "start_result": result
                }

            return {
                "status": "success",
                "detector": detectorName,
                "params": saved.to_dict(),
                "restarted": False
            }
        except Exception as e:
            self._logger.error(f"Error setting detector stream params: {e}")
            return {"status": "error", "message": str(e)}

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

    @APIExport()
    def getCurrentStreamProtocol(self, detectorName: Optional[str] = None) -> Dict[str, Any]:
        """
        Get the current streaming protocol for a specific detector.
        
        Args:
            detectorName: Name of detector (None = first available or first active)
        
        Returns:
            Dictionary with current protocol information
        """
        try:
            # If no detector specified, use first active stream or first available detector
            if detectorName is None:
                if self._activeStreams:
                    detectorName = list(self._activeStreams.keys())[0]
                else:
                    detectorName = self._master.detectorsManager.getAllDeviceNames()[0]

            # Check if detector has active stream
            if detectorName in self._activeStreams:
                protocol, worker = self._activeStreams[detectorName]
                if protocol is None:
                    protocol = "binary" # Default protocol if none set
                return {
                    "status": "success",
                    "detector": detectorName,
                    "protocol": protocol,
                    "is_streaming": True,
                    "params": self._streamParams[protocol].to_dict() if protocol in self._streamParams else {}
                }
            else:
                return {
                    "status": "success",
                    "detector": detectorName,
                    "protocol": None,
                    "is_streaming": False,
                    "message": f"No active stream for detector {detectorName}"
                }
        except Exception as e:
            self._logger.error(f"Error getting current stream protocol: {e}")
            return {
                "status": "error",
                "message": str(e)
            }

    @APIExport()
    def getStreamStatus(self) -> Dict[str, Any]:
        """
        Get comprehensive streaming status for all detectors.
        
        Returns:
            Dictionary with complete streaming status
        """
        try:
            all_detectors = self._master.detectorsManager.getAllDeviceNames()
            detector_status = {}

            for detector_name in all_detectors:
                if detector_name in self._activeStreams:
                    protocol, worker = self._activeStreams[detector_name]
                    detector_status[detector_name] = {
                        "is_streaming": True,
                        "protocol": protocol,
                        "params": self._streamParams[protocol].to_dict() if protocol in self._streamParams else {}
                    }
                else:
                    detector_status[detector_name] = {
                        "is_streaming": False,
                        "protocol": None,
                        "params": {}
                    }

            return {
                "status": "success",
                "total_detectors": len(all_detectors),
                "active_streams": len(self._activeStreams),
                "detectors": detector_status,
                "available_protocols": list(self._streamParams.keys())
            }
        except Exception as e:
            self._logger.error(f"Error getting stream status: {e}")
            return {
                "status": "error",
                "message": str(e)
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
            self.stopLiveView(detectorName=detectorName, stopCamera=False)
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
                self._logger.info("MJPEG stream connection closed by client")
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
    def webrtc_offer(self, request: WebRTCOfferRequest):
        """
        Handle WebRTC offer from client and return answer.
        
        Args:
            request: WebRTCOfferRequest containing SDP offer and detector name
        
        Returns:
            Dictionary with answer SDP
        """
        import time
        start_time = time.time()
        timing = {}

        try:
            from aiortc import RTCPeerConnection, RTCSessionDescription
            import asyncio
            import json
        except ImportError:
            return {"status": "error", "message": "aiortc not available"}

        timing['imports'] = time.time() - start_time
        self._logger.info("⚡ WebRTC offer processing started")

        # Extract parameters from request
        sdp = request.sdp
        sdp_type = request.sdp_type
        detectorName = request.detectorName
        params = request.params or {}

        timing['params_extracted'] = time.time() - start_time
        self._logger.info(f"Received WebRTC offer: type={sdp_type}, sdp length={len(sdp)}, params={params}")

        # Get detector name
        if detectorName is None:
            detectorName = self._master.detectorsManager.getAllDeviceNames()[0]

        # Update stream parameters if provided
        if params:
            try:
                current_params = self._streamParams.get('webrtc', StreamParams(protocol='webrtc'))
                # Update with new parameters
                for key, value in params.items():
                    if hasattr(current_params, key):
                        setattr(current_params, key, value)
                        self._logger.info(f"Updated WebRTC param {key}={value}")
                self._streamParams['webrtc'] = current_params
                timing['params_updated'] = time.time() - start_time
            except Exception as e:
                self._logger.warning(f"Failed to update stream parameters: {e}")

        # Start WebRTC stream if not already active
        if detectorName not in self._activeStreams:
            self._logger.info(f"🚀 Starting WebRTC stream for {detectorName}")
            result = self.startLiveView(detectorName, "webrtc", params)
            if result['status'] != 'success':
                return result
            timing['stream_started'] = time.time() - start_time
        else:
            timing['stream_started'] = time.time() - start_time
            self._logger.info(f"🔄 WebRTC stream already active for {detectorName}")

        # Get the worker
        worker = self.getWebRTCWorker(detectorName)
        if worker is None:
            return {"status": "error", "message": "Failed to get WebRTC worker"}

        timing['worker_ready'] = time.time() - start_time

        # Get or create persistent event loop
        loop = self._get_or_create_webrtc_loop()
        timing['loop_ready'] = time.time() - start_time

        # Handle offer and create answer in async context
        async def process_offer():
            offer_start = time.time()
            offer_timing = {}

            # Close existing peer connection for this detector if any
            if detectorName in self._webrtc_peers:
                old_pc = self._webrtc_peers[detectorName]
                try:
                    await old_pc.close()
                    self._logger.info(f"Closed old peer connection for {detectorName}")
                except Exception as e:
                    self._logger.warning(f"Error closing old peer: {e}")

            offer_timing['cleanup'] = time.time() - offer_start

            # Create new peer connection with ICE servers
            from aiortc import RTCConfiguration

            # For local connections, we don't need STUN servers (works without internet)
            # Explicitly disable ICE servers to prevent STUN timeouts
            stream_params = self._streamParams.get('webrtc', StreamParams(protocol='webrtc'))

            # Force empty ICE servers list for local connections to prevent STUN timeouts
            self._logger.info("✅ Disabling ICE servers for local connection (prevents STUN timeouts)")
            config = RTCConfiguration(iceServers=[])  # Only iceServers parameter supported
            pc = RTCPeerConnection(configuration=config)

            # Store PC for this detector
            self._webrtc_peers[detectorName] = pc

            offer_timing['pc_created'] = time.time() - offer_start

            @pc.on("connectionstatechange")
            async def on_connectionstatechange():
                self._logger.info(f"WebRTC connection state for {detectorName}: {pc.connectionState}")
                if pc.connectionState == "failed":
                    self._logger.error(f"WebRTC connection failed for {detectorName}")
                elif pc.connectionState == "closed":
                    self._logger.info(f"WebRTC connection closed for {detectorName}")
                    if detectorName in self._webrtc_peers and self._webrtc_peers[detectorName] == pc:
                        del self._webrtc_peers[detectorName]

            @pc.on("iceconnectionstatechange")
            async def on_iceconnectionstatechange():
                self._logger.info(f"ICE connection state for {detectorName}: {pc.iceConnectionState}")

            # Add video track
            video_track = worker.get_video_track()
            offer_timing['got_track'] = time.time() - offer_start
            self._logger.info(f"Got video track from worker: {video_track}, type: {type(video_track)}")
            if video_track:
                # Verify the track has the recv method
                if hasattr(video_track, 'recv'):
                    self._logger.info(f"Video track has recv method: {video_track.recv}")
                else:
                    self._logger.error("Video track missing recv method!")

                pc.addTrack(video_track)
                offer_timing['track_added'] = time.time() - offer_start
                self._logger.info(f"Added video track to peer connection for {detectorName}")
                self._logger.info(f"Peer connection tracks: {pc.getTransceivers()}")
            else:
                self._logger.error("Failed to get video track from worker")
                raise Exception("No video track available")

            # Process offer
            try:
                self._logger.info("🔄 Processing SDP offer...")
                offer_desc = RTCSessionDescription(sdp=sdp, type=sdp_type)
                await pc.setRemoteDescription(offer_desc)
                offer_timing['remote_desc_set'] = time.time() - offer_start
                self._logger.info(f"Set remote description for {detectorName}")

                # Create and set answer
                self._logger.info("🔄 Creating SDP answer...")
                answer = await pc.createAnswer()
                offer_timing['answer_created'] = time.time() - offer_start

                # Pre-warm the video track by getting a frame
                self._logger.info("🔄 Pre-warming video track...")
                try:
                    if video_track and hasattr(video_track, 'recv'):
                        # Try to get a frame to ensure the track is ready
                        await asyncio.wait_for(video_track.recv(), timeout=0.5)
                        self._logger.info("✅ Video track pre-warmed successfully")
                except (asyncio.TimeoutError, Exception) as e:
                    self._logger.warning(f"⚠️ Video track pre-warm failed (continuing anyway): {e}")

                offer_timing['track_prewarmed'] = time.time() - offer_start

                # Set local description with timeout to prevent hanging
                self._logger.info("🔄 Setting local description...")
                local_desc_success = True
                try:
                    await asyncio.wait_for(pc.setLocalDescription(answer), timeout=2.0)
                    offer_timing['local_desc_set'] = time.time() - offer_start
                    self._logger.info("✅ Local description set successfully")
                except asyncio.TimeoutError:
                    self._logger.error("❌ Timeout setting local description - using answer SDP directly")
                    local_desc_success = False
                    offer_timing['local_desc_set'] = time.time() - offer_start
                except Exception as local_desc_error:
                    self._logger.error(f"❌ Error setting local description: {local_desc_error}")
                    local_desc_success = False
                    offer_timing['local_desc_set'] = time.time() - offer_start

                # Check if we have a valid local description, otherwise use the answer directly
                if pc.localDescription and pc.localDescription.type:
                    self._logger.info(f"WebRTC answer created for {detectorName}: type={pc.localDescription.type}")
                    answer_sdp = pc.localDescription.sdp
                    answer_type = pc.localDescription.type
                else:
                    self._logger.warning("⚠️ Local description not available, using answer SDP directly")
                    answer_sdp = answer.sdp
                    answer_type = answer.type

                # Log detailed timing breakdown
                total_offer_time = time.time() - offer_start
                self._logger.info("🕐 Backend SDP processing timing:")
                self._logger.info(f"   Cleanup: {offer_timing.get('cleanup', 0)*1000:.1f}ms")
                self._logger.info(f"   PC created: {offer_timing.get('pc_created', 0)*1000:.1f}ms")
                self._logger.info(f"   Got track: {(offer_timing.get('got_track', 0) - offer_timing.get('pc_created', 0))*1000:.1f}ms")
                self._logger.info(f"   Track added: {(offer_timing.get('track_added', 0) - offer_timing.get('got_track', 0))*1000:.1f}ms")
                self._logger.info(f"   Remote desc: {(offer_timing.get('remote_desc_set', 0) - offer_timing.get('track_added', 0))*1000:.1f}ms")
                self._logger.info(f"   Answer created: {(offer_timing.get('answer_created', 0) - offer_timing.get('remote_desc_set', 0))*1000:.1f}ms")
                self._logger.info(f"   Track prewarmed: {(offer_timing.get('track_prewarmed', 0) - offer_timing.get('answer_created', 0))*1000:.1f}ms")
                self._logger.info(f"   Local desc: {(offer_timing.get('local_desc_set', 0) - offer_timing.get('track_prewarmed', 0))*1000:.1f}ms")
                self._logger.info(f"   Total SDP processing: {total_offer_time*1000:.1f}ms")

                return {
                    "status": "success",
                    "sdp": answer_sdp,
                    "type": answer_type
                }
            except Exception as offer_error:
                self._logger.error(f"❌ Error processing SDP offer for {detectorName}: {offer_error}")
                # Try to close the peer connection on error
                try:
                    await pc.close()
                    if detectorName in self._webrtc_peers:
                        del self._webrtc_peers[detectorName]
                except Exception as close_error:
                    self._logger.warning(f"Error closing peer connection: {close_error}")

                # Return a fallback error response
                return {
                    "status": "error",
                    "message": f"SDP processing failed: {str(offer_error)}"
                }

        # Run async function on persistent event loop
        try:
            self._logger.info("🔄 Running async SDP processing...")
            timing['async_start'] = time.time() - start_time

            future = asyncio.run_coroutine_threadsafe(process_offer(), loop)
            result = future.result(timeout=15.0)  # Increased timeout for better reliability

            timing['async_complete'] = time.time() - start_time
            total_time = time.time() - start_time

            # Log complete backend timing breakdown
            self._logger.info("🕐 Complete backend timing breakdown:")
            self._logger.info(f"   Total backend time: {total_time*1000:.1f}ms")
            self._logger.info(f"   Imports: {timing.get('imports', 0)*1000:.1f}ms")
            self._logger.info(f"   Params extracted: {timing.get('params_extracted', 0)*1000:.1f}ms")
            self._logger.info(f"   Params updated: {timing.get('params_updated', timing.get('params_extracted', 0))*1000:.1f}ms")
            self._logger.info(f"   Stream started: {(timing.get('stream_started', 0) - timing.get('params_updated', timing.get('params_extracted', 0)))*1000:.1f}ms")
            self._logger.info(f"   Worker ready: {(timing.get('worker_ready', 0) - timing.get('stream_started', 0))*1000:.1f}ms")
            self._logger.info(f"   Loop ready: {(timing.get('loop_ready', 0) - timing.get('worker_ready', 0))*1000:.1f}ms")
            self._logger.info(f"   Async setup: {(timing.get('async_start', 0) - timing.get('loop_ready', 0))*1000:.1f}ms")
            self._logger.info(f"   Async processing: {(timing.get('async_complete', 0) - timing.get('async_start', 0))*1000:.1f}ms")

            self._logger.info(f"✅ WebRTC offer processed successfully in {total_time*1000:.1f}ms")
            return result

        except asyncio.TimeoutError:
            total_time = time.time() - start_time
            self._logger.error(f"❌ Timeout processing WebRTC offer for {detectorName} after {total_time*1000:.1f}ms")
            return {"status": "error", "message": "Timeout processing WebRTC offer"}
        except Exception as e:
            import traceback
            total_time = time.time() - start_time
            full_error = traceback.format_exc()
            self._logger.error(f"❌ Error processing WebRTC offer after {total_time*1000:.1f}ms: {e}\n{full_error}")
            return {"status": "error", "message": str(e)}

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
