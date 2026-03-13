# interfaces/andor_camera.py
"""
Low-level wrapper around pyAndorSDK3.

Exposes a HIK-like API so that ImSwitch's DetectorManager can control
the camera without knowing SDK specifics.

Key design decisions:
  - wait_buffer timeout is computed dynamically so very long exposures
    (> 30 s) never time out prematurely.
  - suspend_live() stops the hardware stream without closing the device,
    matching the HIK interface contract used by DetectorManager.
  - setPropertyValue / getPropertyValue provide a uniform dict-like API.
  - Single-frame snap() capture is available for long exposures.
  - Flatfield correction is applied on the fly (same as HIK).
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Optional, Tuple

import numpy as np
from imswitch.imcommon.model import initLogger

try:
    import pyAndorSDK3
    from pyAndorSDK3 import AndorSDK3
    IS_ANDOR_SDK3_AVAILABLE = True
except ImportError:
    IS_ANDOR_SDK3_AVAILABLE = False
    AndorSDK3 = None


# ---------------------------------------------------------------------------
# Pixel-encoding helpers
# ---------------------------------------------------------------------------
_ENCODING_MAX = {
    "mono8":        255,
    "mono12":       4095,
    "mono12packed": 4095,
    "mono16":       65535,
    "mono32":       65535,
}


class andorcamera:
    """
    Andor SDK3 interface delivering frames through a ring-buffer.

    Public attributes that match the HIK camera contract:
        model              str
        SensorWidth        int
        SensorHeight       int
        shape              (H, W)
        frameNumber        int   - last received frame id
        timestamp          float
        _activePixelFormat str   - e.g. "Mono16"
        isFlatfielding     bool
    """

    model: str = "AndorSDK3"

    def __init__(
        self,
        camera_no: int = 0,
        pixel_encoding: str = "Mono16",
        buffer_count: int = 10,
    ) -> None:
        self.__logger = initLogger(self, tryInheritParent=False)

        if not IS_ANDOR_SDK3_AVAILABLE:
            raise ImportError("pyAndorSDK3 is not installed")

        # ---- SDK objects ---------------------------------------------------
        self._sdk = AndorSDK3()
        self.cam = self._sdk.GetCamera(camera_no)
        self.cam.open()

        # ---- pixel encoding ------------------------------------------------
        self._activePixelFormat = pixel_encoding
        self.cam.PixelEncoding = pixel_encoding

        # ---- sensor geometry -----------------------------------------------
        self.SensorWidth  = int(getattr(self.cam, "SensorWidth",  self.cam.AOIWidth))
        self.SensorHeight = int(getattr(self.cam, "SensorHeight", self.cam.AOIHeight))
        self.shape: Tuple[int, int] = (self.SensorHeight, self.SensorWidth)

        # ---- ring-buffer ---------------------------------------------------
        self._buffer_count = buffer_count
        self._buf_ring: collections.deque[np.ndarray] = collections.deque(maxlen=buffer_count)
        self._fid_ring:  collections.deque[int]       = collections.deque(maxlen=buffer_count)
        self.frameNumber: int   = -1
        self.timestamp:   float = 0.0

        # ---- streaming state -----------------------------------------------
        self._worker:  Optional[threading.Thread] = None
        self._running  = False
        self._exit     = threading.Event()

        # ---- flatfield -----------------------------------------------------
        self.isFlatfielding   = False
        self._flatfieldImage: Optional[np.ndarray] = None

        # ---- ROI (full-frame defaults) -------------------------------------
        self._roi_hpos  = 0
        self._roi_vpos  = 0
        self._roi_hsize: Optional[int] = None
        self._roi_vsize: Optional[int] = None

        # ---- default parameters -------------------------------------------
        self.cam.CycleMode   = "Continuous"
        self.cam.TriggerMode = "Internal"
        self.set_exposure_time_ms(10.0)
        self.set_gain(0)

        self.__logger.info(
            f"Andor camera opened: {self.SensorWidth}\u00d7{self.SensorHeight}, "
            f"encoding={pixel_encoding}"
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _wait_timeout_ms(self) -> int:
        """
        Compute a safe wait_buffer() timeout in ms.

        Formula: 3 x exposure + 5 s, minimum 2 s.
        This ensures even exposures > 30 s never time out prematurely.
        """
        try:
            exp_ms = float(self.cam.ExposureTime) * 1000.0
        except Exception:
            exp_ms = 10_000.0
        return max(2_000, int(exp_ms * 3 + 5_000))

    def _decode_frame(self, raw: np.ndarray) -> np.ndarray:
        """Convert the raw SDK byte buffer into a shaped 2-D numpy array."""
        enc = self._activePixelFormat.lower()
        h = self._roi_vsize or self.SensorHeight
        w = self._roi_hsize or self.SensorWidth

        if enc in ("mono16", "mono12"):
            img = raw.view(np.uint16)
        elif enc == "mono32":
            img = raw.view(np.uint32)
        else:
            img = raw.view(np.uint8)

        # Reshape respecting stride padding when available
        try:
            stride_px = self.cam.AOIStride // img.itemsize
            img = img[: h * stride_px].reshape(h, stride_px)[:, :w]
        except Exception:
            img = img[: h * w].reshape(h, w)

        return img.copy()

    # -----------------------------------------------------------------------
    # Basic parameter helpers
    # -----------------------------------------------------------------------
    def set_exposure_time_ms(self, exp_ms: float) -> None:
        exp_s = max(
            float(self.cam.min_ExposureTime),
            min(float(self.cam.max_ExposureTime), exp_ms / 1000.0),
        )
        self.cam.ExposureTime = exp_s
        self.__logger.debug(f"Exposure set to {exp_s * 1000:.3f} ms")

    def get_exposure_time_ms(self) -> float:
        return float(self.cam.ExposureTime) * 1000.0

    def get_exposuretime(self) -> Tuple[float, float, float]:
        """Return (current_us, min_us, max_us) - HIK-compatible tuple."""
        cur = float(self.cam.ExposureTime)     * 1_000_000.0
        mn  = float(self.cam.min_ExposureTime) * 1_000_000.0
        mx  = float(self.cam.max_ExposureTime) * 1_000_000.0
        return (cur, mn, mx)

    def set_exposure_mode(self, mode: str = "manual") -> None:
        # Andor SDK3 has no automatic-exposure mode; method kept for API parity
        self.__logger.debug(f"set_exposure_mode({mode}) - Andor has no AE, ignored")

    def set_gain(self, gain: float) -> None:
        # Andor cameras expose gain via SimplePreAmpGainControl (string enum),
        # not a numeric register; silently accept and ignore.
        self.__logger.debug(f"set_gain({gain}) - no numeric gain knob on Andor SDK3")

    def get_gain(self) -> Tuple[float, float, float]:
        """Return (current, min, max) - HIK-compatible tuple."""
        return (0.0, 0.0, 0.0)

    def set_frame_rate(self, fps: float) -> None:
        if fps > 0:
            try:
                self.cam.FrameRate = min(fps, float(self.cam.max_FrameRate))
            except Exception as e:
                self.__logger.warning(f"set_frame_rate({fps}): {e}")

    def set_blacklevel(self, level: float) -> None:
        # Not a standard Andor SDK3 feature; kept for API parity
        self.__logger.debug(f"set_blacklevel({level}) - not supported by Andor SDK3, ignored")

    def set_pixel_format(self, fmt: str) -> None:
        """Change pixel encoding (e.g. 'Mono16', 'Mono12')."""
        self._activePixelFormat = fmt
        try:
            self.cam.PixelEncoding = fmt
            self.__logger.debug(f"Pixel encoding set to {fmt}")
        except Exception as e:
            self.__logger.error(f"set_pixel_format({fmt}): {e}")

    def setBinning(self, binning: int = 1) -> None:
        tag = f"{binning}x{binning}"
        try:
            self.cam.AOIBinning = tag
            self.SensorWidth  = int(self.cam.AOIWidth)
            self.SensorHeight = int(self.cam.AOIHeight)
            self.shape        = (self.SensorHeight, self.SensorWidth)
            self.__logger.debug(f"Binning set to {tag}")
        except Exception as e:
            self.__logger.warning(f"setBinning({binning}): {e}")

    def set_camera_mode(self, isAutomatic: bool) -> None:
        self.__logger.debug(f"set_camera_mode({isAutomatic}) - not applicable to Andor SDK3")

    # -----------------------------------------------------------------------
    # Trigger & acquisition mode
    # -----------------------------------------------------------------------
    def set_trigger_source(self, source: str) -> bool:
        lower = source.lower()
        try:
            if lower.startswith(("cont", "internal")):
                self.cam.TriggerMode = "Internal"
            elif lower.startswith("soft"):
                self.cam.TriggerMode = "Software"
            elif lower.startswith(("ext", "hard")):
                self.cam.TriggerMode = "External"
            else:
                self.__logger.warning(f"Unknown trigger source '{source}'")
                return False
            self.__logger.debug(f"Trigger mode set to {self.cam.TriggerMode}")
            return True
        except Exception as e:
            self.__logger.error(e)
            return False

    # HIK-compatible camelCase alias
    def setTriggerSource(self, source: str) -> bool:
        return self.set_trigger_source(source)

    def send_trigger(self) -> bool:
        try:
            self.cam.SoftwareTrigger()
            return True
        except Exception as e:
            self.__logger.error(e)
            return False

    def getTriggerTypes(self):
        return [
            "Continuous (Internal)",
            "Software Trigger",
            "External Trigger",
        ]

    def getTriggerSource(self) -> str:
        tm = self.cam.TriggerMode
        if tm == "Software":
            return "Software Trigger"
        if tm == "External":
            return "External Trigger"
        return "Continuous (Internal)"

    # -----------------------------------------------------------------------
    # Live (streaming) acquisition
    # -----------------------------------------------------------------------
    def start_live(self) -> None:
        if self._running:
            return
        self.__logger.debug("Starting live acquisition ...")
        self._queue_initial_buffers()
        self.cam.AcquisitionStart()
        self._exit.clear()
        self._worker = threading.Thread(target=self._acq_loop, daemon=True)
        self._worker.start()
        self._running = True

    def stop_live(self) -> None:
        """Full stop: halt the hardware stream and release all SDK buffers."""
        if not self._running:
            return
        self.__logger.debug("Stopping live acquisition ...")
        self._exit.set()
        if self._worker:
            # Give the thread time to finish the current wait_buffer call
            join_timeout = max(5.0, self._wait_timeout_ms() / 1000.0 + 2.0)
            self._worker.join(timeout=join_timeout)
            self._worker = None
        try:
            self.cam.AcquisitionStop()
            self.cam.flush()
        except Exception as e:
            self.__logger.warning(f"stop_live cleanup: {e}")
        self._running = False

    def suspend_live(self) -> None:
        """
        Stop the hardware stream but keep the device open.
        Mirrors the HIK interface; used by DetectorManager when changing
        parameters that require the stream to be idle.
        """
        self.stop_live()

    def prepare_live(self) -> None:
        """No-op - kept for HIK API compatibility."""
        pass

    def _queue_initial_buffers(self) -> None:
        bs = int(self.cam.ImageSizeBytes)
        for _ in range(self._buffer_count):
            buf = np.empty((bs,), dtype=np.uint8)
            self.cam.queue(buf, bs)

    def _acq_loop(self) -> None:
        """
        Background thread: pull completed frames from the SDK and push them
        into the ring buffer.

        The wait_buffer() timeout is recalculated on every iteration so that
        very long exposures (> 30 s) never falsely raise TimeoutError.
        """
        bs = int(self.cam.ImageSizeBytes)
        while not self._exit.is_set():
            timeout = self._wait_timeout_ms()
            try:
                acq   = self.cam.wait_buffer(timeout)
                raw   = np.frombuffer(acq._np_data, dtype=np.uint8)
                frame = self._decode_frame(raw)

                # Frame counter: prefer SDK metadata, fallback to increment
                try:
                    fid = int(acq.metadata.frame_count)
                except Exception:
                    fid = self.frameNumber + 1

                ts = time.time()

                # Optional flatfield correction
                if self.isFlatfielding and self._flatfieldImage is not None:
                    try:
                        corrected = frame.astype(np.float32) / self._flatfieldImage
                        max_val   = _ENCODING_MAX.get(self._activePixelFormat.lower(), 65535)
                        frame     = np.clip(corrected, 0, max_val).astype(frame.dtype)
                    except Exception:
                        pass  # Keep uncorrected frame on error

                self._buf_ring.append(frame)
                self._fid_ring.append(fid)
                self.frameNumber = fid
                self.timestamp   = ts

                # Re-queue the same buffer for the next frame
                try:
                    self.cam.queue(acq._np_data, bs)
                except Exception:
                    new_buf = np.empty((bs,), dtype=np.uint8)
                    self.cam.queue(new_buf, bs)

            except TimeoutError:
                continue
            except Exception as e:
                if not self._exit.is_set():
                    self.__logger.error(f"Acquisition loop error: {e}")
                break

    # -----------------------------------------------------------------------
    # Frame retrieval
    # -----------------------------------------------------------------------
    def getLast(
        self,
        returnFrameNumber: bool = False,
        timeout: Optional[float] = None,
        auto_trigger: bool = True,
    ):
        """
        Return the latest decoded frame as a 2-D numpy array.

        ``timeout`` defaults to the dynamic value computed from the current
        exposure time so long exposures always complete before giving up.
        A software trigger is sent automatically when in Software mode.
        """
        if timeout is None:
            timeout = self._wait_timeout_ms() / 1000.0

        # Send a software trigger when required
        if auto_trigger and self.cam.TriggerMode == "Software":
            self.send_trigger()

        t0 = time.time()
        while not self._buf_ring:
            if time.time() - t0 > timeout:
                return (None, None) if returnFrameNumber else None
            time.sleep(0.005)

        frame = self._buf_ring[-1]
        fid   = self._fid_ring[-1]
        return (frame, fid) if returnFrameNumber else frame

    def getLastChunk(self):
        """Return (array_of_frames, array_of_fids) and flush the ring buffer."""
        frames = list(self._buf_ring)
        fids   = list(self._fid_ring)
        self.flushBuffer()
        return np.array(frames), np.array(fids)

    def flushBuffer(self) -> None:
        self._buf_ring.clear()
        self._fid_ring.clear()

    def getFrameNumber(self) -> int:
        return self.frameNumber

    # -----------------------------------------------------------------------
    # Single-shot (non-live) capture
    # -----------------------------------------------------------------------
    def snap(self) -> Optional[np.ndarray]:
        """
        Capture a single frame outside the continuous live loop.

        Recommended for very long exposures (e.g. > 30 s) where running
        the full ring-buffer loop would waste bandwidth and CPU.
        Temporarily switches to Fixed CycleMode=1, waits for the frame
        with a dynamically-computed timeout, then restores settings.
        """
        timeout_ms   = self._wait_timeout_ms()
        bs           = int(self.cam.ImageSizeBytes)
        buf          = np.empty((bs,), dtype=np.uint8)
        saved_trigger = self.cam.TriggerMode
        saved_cycle   = self.cam.CycleMode
        frame         = None
        try:
            self.cam.CycleMode   = "Fixed"
            self.cam.FrameCount  = 1
            self.cam.TriggerMode = "Internal"
            self.cam.queue(buf, bs)
            self.cam.AcquisitionStart()
            acq   = self.cam.wait_buffer(timeout_ms)
            raw   = np.frombuffer(acq._np_data, dtype=np.uint8)
            frame = self._decode_frame(raw)
        except Exception as e:
            self.__logger.error(f"snap() failed: {e}")
        finally:
            try:
                self.cam.AcquisitionStop()
                self.cam.flush()
            except Exception:
                pass
            try:
                self.cam.TriggerMode = saved_trigger
                self.cam.CycleMode   = saved_cycle
            except Exception:
                pass
        return frame

    # -----------------------------------------------------------------------
    # ROI
    # -----------------------------------------------------------------------
    def setROI(
        self,
        hpos: int = 0,
        vpos: int = 0,
        hsize: Optional[int] = None,
        vsize: Optional[int] = None,
    ) -> None:
        if hsize is None:
            hsize = self.SensorWidth
        if vsize is None:
            vsize = self.SensorHeight

        self._roi_hpos  = hpos
        self._roi_vpos  = vpos
        self._roi_hsize = hsize
        self._roi_vsize = vsize

        try:
            # Andor AOI coordinates are 1-based
            self.cam.AOILeft   = hpos + 1
            self.cam.AOITop    = vpos + 1
            self.cam.AOIWidth  = hsize
            self.cam.AOIHeight = vsize
        except Exception as e:
            self.__logger.warning(f"setROI could not apply to SDK: {e}")
        self.__logger.debug(f"ROI set to ({hpos},{vpos}) {hsize}\u00d7{vsize}")

    # -----------------------------------------------------------------------
    # Flatfield correction
    # -----------------------------------------------------------------------
    def set_flatfielding(self, enabled: bool) -> None:
        self.isFlatfielding = enabled

    def setFlatfieldImage(
        self,
        flatfieldImage: Optional[np.ndarray],
        isFlatfieldEnabled: bool = True,
    ) -> None:
        if flatfieldImage is not None:
            self._flatfieldImage = flatfieldImage.astype(np.float32)
            self._flatfieldImage[self._flatfieldImage == 0] = 1.0
        self.isFlatfielding = isFlatfieldEnabled

    def recordFlatfieldImage(self, nFrames: int = 10) -> None:
        """Average nFrames to build a flatfield reference image."""
        frames  = []
        timeout = self._wait_timeout_ms() / 1000.0
        for _ in range(nFrames):
            f = self.getLast(auto_trigger=True, timeout=timeout)
            if f is not None:
                frames.append(f.astype(np.float32))
            time.sleep(0.02)
        if frames:
            avg = np.mean(frames, axis=0).astype(np.float32)
            avg[avg == 0] = 1.0
            self._flatfieldImage = avg
            self.__logger.info("Flatfield image recorded")

    # -----------------------------------------------------------------------
    # HIK-compatible property dictionary interface
    # -----------------------------------------------------------------------
    def setPropertyValue(self, name: str, value) -> object:
        """Map HIK property names to SDK3 setters."""
        n = name.lower()
        if n == "exposure":
            self.set_exposure_time_ms(float(value))
        elif n == "gain":
            self.set_gain(float(value))
        elif n in ("frame_rate", "framerate"):
            self.set_frame_rate(float(value))
        elif n == "blacklevel":
            self.set_blacklevel(float(value))
        elif n == "trigger_source":
            self.set_trigger_source(str(value))
        elif n == "pixel_format":
            self.set_pixel_format(str(value))
        elif n == "binning":
            self.setBinning(int(value))
        else:
            self.__logger.debug(f"setPropertyValue: unknown property '{name}'")
        return value

    def getPropertyValue(self, name: str) -> object:
        """Map HIK property names to SDK3 getters."""
        n = name.lower()
        if n == "exposure":
            return self.get_exposure_time_ms()
        if n == "gain":
            return self.get_gain()[0]
        if n == "frame_number":
            return self.frameNumber
        if n == "trigger_source":
            return self.getTriggerSource()
        if n == "image_width":
            return self.SensorWidth
        if n == "image_height":
            return self.SensorHeight
        self.__logger.debug(f"getPropertyValue: unknown property '{name}'")
        return None

    def get_camera_parameters(self) -> dict:
        """Return a summary dict (used by getCameraStatus in the manager)."""
        return {
            "model_name":     self.model,
            "isRGB":          False,
            "exposure_us":    self.get_exposuretime()[0],
            "trigger_source": self.getTriggerSource(),
            "pixel_encoding": self._activePixelFormat,
            "width":          self.SensorWidth,
            "height":         self.SensorHeight,
        }

    # -----------------------------------------------------------------------
    # GUI stub
    # -----------------------------------------------------------------------
    def openPropertiesGUI(self) -> None:
        self.__logger.info("openPropertiesGUI: no native GUI available for Andor SDK3 cameras")

    # -----------------------------------------------------------------------
    # Housekeeping
    # -----------------------------------------------------------------------
    def close(self) -> None:
        if self._running:
            self.stop_live()
        try:
            self.cam.close()
        except Exception as e:
            self.__logger.warning(f"close(): {e}")
        self.__logger.info("Andor camera closed")

    # ---- context-manager support ------------------------------------------
    def __enter__(self):
        self.start_live()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
