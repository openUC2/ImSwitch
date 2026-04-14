# interfaces/andorcamera2.py
"""
Low-level wrapper around pyAndorSDK2 (atmcd / classic DLL-based API).

Provides the same public interface as ``andorcamera.py`` (SDK3) so that
``AndorCamManager`` can drive either SDK version transparently.

SDK2 key differences vs SDK3:
  - Single global ``atmcd()`` object; no per-camera OO handle.
  - Acquisition mode 5 = Run Till Abort (continuous).
  - Read mode 4 = Image (full 2-D).
  - Trigger modes: 0=Internal, 1=External, 10=Software.
  - Bloking per-frame wait via ``WaitForAcquisitionTimeOut()``.
  - Image retrieval via ``GetMostRecentImage16()`` (16-bit).
  - ROI / binning via ``SetImage(hbin, vbin, hstart, hend, vstart, vend)``.
  - Coordinates are **1-based** in SDK2.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Optional, Tuple

import numpy as np
from imswitch.imcommon.model import initLogger

try:
    from imswitch.imcontrol.model.interfaces.pyAndorSDK2 import atmcd
    from imswitch.imcontrol.model.interfaces.pyAndorSDK2 import atmcd_codes, atmcd_errors

    IS_ANDOR_SDK2_AVAILABLE = True
except Exception:
    IS_ANDOR_SDK2_AVAILABLE = False
    atmcd = None  # type: ignore

# Alias for readability
_DRV_SUCCESS  = 20002  # atmcd_errors.Error_Codes.DRV_SUCCESS
_DRV_NO_NEW   = 20024  # DRV_NO_NEW_DATA
_DRV_ACQUIRING = 20072
_DRV_IDLE      = 20073

# Acquisition mode numbers (SDK2)
_ACQ_RUN_TILL_ABORT = 5
_ACQ_SINGLE_SCAN    = 1

# Read mode numbers (SDK2)
_READ_IMAGE = 4

# Trigger mode numbers (SDK2)
_TRIG_INTERNAL = 0
_TRIG_EXTERNAL = 1
_TRIG_SOFTWARE = 10

# Max pixel value for 16-bit
_MAX_16BIT = 65535


class andorcamera2:
    """
    Andor SDK2 interface delivering frames through a ring-buffer thread.

    Public attributes that match the HIK / SDK3 camera contract:
        model              str
        SensorWidth        int
        SensorHeight       int
        shape              (H, W)
        frameNumber        int   – last received frame counter
        timestamp          float
        _activePixelFormat str   – always "Mono16" for SDK2
        isFlatfielding     bool
    """

    model: str = "AndorSDK2"

    def __init__(
        self,
        camera_no: int = 0,
        buffer_count: int = 10,
    ) -> None:
        self.__logger = initLogger(self, tryInheritParent=False)

        if not IS_ANDOR_SDK2_AVAILABLE:
            raise ImportError(
                "pyAndorSDK2 is not available – ensure the Andor SDK2 DLL is installed"
            )

        # ---- SDK object ---------------------------------------------------
        self._sdk = atmcd()
        ret = self._sdk.Initialize("")
        if ret != _DRV_SUCCESS:
            raise RuntimeError(f"Andor SDK2 Initialize() failed with code {ret}")

        # ---- pixel format --------------------------------------------------
        # SDK2 only supports 16-bit readout in Image mode for scientific cameras
        self._activePixelFormat = "Mono16"

        # ---- sensor geometry -----------------------------------------------
        ret, xpix, ypix = self._sdk.GetDetector()
        if ret != _DRV_SUCCESS:
            raise RuntimeError(f"SDK2 GetDetector() failed with code {ret}")
        self.SensorWidth:  int = xpix
        self.SensorHeight: int = ypix
        self.shape: Tuple[int, int] = (self.SensorHeight, self.SensorWidth)

        # ---- ROI defaults (full frame, 1-based SDK2 coordinates) ----------
        self._hbin    = 1
        self._vbin    = 1
        self._hstart  = 1
        self._hend    = self.SensorWidth
        self._vstart  = 1
        self._vend    = self.SensorHeight

        # ---- ring-buffer ---------------------------------------------------
        self._buffer_count = buffer_count
        self._buf_ring: collections.deque[np.ndarray] = collections.deque(maxlen=buffer_count)
        self._fid_ring: collections.deque[int]        = collections.deque(maxlen=buffer_count)
        self.frameNumber: int   = -1
        self.timestamp:   float = 0.0

        # ---- streaming state -----------------------------------------------
        self._worker: Optional[threading.Thread] = None
        self._running  = False
        self._exit     = threading.Event()

        # ---- flatfield -----------------------------------------------------
        self.isFlatfielding    = False
        self._flatfieldImage: Optional[np.ndarray] = None

        # ---- initial HW configuration -------------------------------------
        self._sdk.SetReadMode(_READ_IMAGE)
        self._sdk.SetAcquisitionMode(_ACQ_RUN_TILL_ABORT)
        self._sdk.SetTriggerMode(_TRIG_INTERNAL)
        self._sdk.SetImage(
            self._hbin, self._vbin,
            self._hstart, self._hend,
            self._vstart, self._vend,
        )
        self.set_exposure_time_ms(10.0)

        # Model string
        ret2, head = self._sdk.GetHeadModel()
        if ret2 == _DRV_SUCCESS:
            self.model = str(head)

        self.__logger.info(
            f"Andor SDK2 camera opened: {self.SensorWidth}\u00d7{self.SensorHeight}"
        )

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------
    def _wait_timeout_ms(self) -> int:
        """Dynamic wait timeout: 3× exposure + 5 s, minimum 2 s."""
        try:
            _, exp_s, _, _ = self._sdk.GetAcquisitionTimings()
            exp_ms = exp_s * 1000.0
        except Exception:
            exp_ms = 10_000.0
        return max(2_000, int(exp_ms * 3 + 5_000))

    def _image_size(self) -> int:
        """Number of pixels in the current ROI / binning."""
        cols = (self._hend - self._hstart + 1) // self._hbin
        rows = (self._vend - self._vstart + 1) // self._vbin
        return rows * cols

    def _image_shape(self) -> Tuple[int, int]:
        rows = (self._vend - self._vstart + 1) // self._vbin
        cols = (self._hend - self._hstart + 1) // self._hbin
        return (rows, cols)

    # -----------------------------------------------------------------------
    # Basic parameter helpers
    # -----------------------------------------------------------------------
    def set_exposure_time_ms(self, exp_ms: float) -> None:
        exp_s = max(0.0, exp_ms / 1000.0)
        self._sdk.SetExposureTime(exp_s)
        self.__logger.debug(f"Exposure set to {exp_ms:.3f} ms")

    def get_exposure_time_ms(self) -> float:
        _, exp_s, _, _ = self._sdk.GetAcquisitionTimings()
        return float(exp_s) * 1000.0

    def get_exposuretime(self) -> Tuple[float, float, float]:
        """Return (current_us, min_us, max_us) – HIK-compatible tuple."""
        _, exp_s, _, _ = self._sdk.GetAcquisitionTimings()
        cur_us = float(exp_s) * 1_000_000.0
        # SDK2 doesn't expose min/max directly; return conservative values
        return (cur_us, 0.001 * 1_000_000.0, 3600.0 * 1_000_000.0)

    def set_exposure_mode(self, mode: str = "manual") -> None:
        self.__logger.debug(f"set_exposure_mode({mode}) – SDK2 has no AE, ignored")

    def set_gain(self, gain: float) -> None:
        """Set EM gain (ignored if camera has no EMCCD)."""
        try:
            g = max(0, int(gain))
            self._sdk.SetEMCCDGain(g)
        except Exception as e:
            self.__logger.debug(f"set_gain({gain}): {e}")

    def get_gain(self) -> Tuple[float, float, float]:
        try:
            ret, g = self._sdk.GetEMCCDGain()
            if ret == _DRV_SUCCESS:
                return (float(g), 0.0, 300.0)
        except Exception:
            pass
        return (0.0, 0.0, 0.0)

    def set_frame_rate(self, fps: float) -> None:
        """Approximate frame rate via kinetic cycle time."""
        if fps > 0:
            try:
                self._sdk.SetKineticCycleTime(1.0 / fps)
            except Exception as e:
                self.__logger.warning(f"set_frame_rate({fps}): {e}")

    def set_blacklevel(self, level: float) -> None:
        self.__logger.debug(f"set_blacklevel({level}) – not supported by SDK2, ignored")

    def set_pixel_format(self, fmt: str) -> None:
        self.__logger.debug(f"set_pixel_format({fmt}) – SDK2 always uses 16-bit, ignored")

    def setBinning(self, binning: int = 1) -> None:
        self._hbin = binning
        self._vbin = binning
        self._apply_image_settings()
        self.shape = self._image_shape()
        self.__logger.debug(f"Binning set to {binning}×{binning}")

    def set_camera_mode(self, isAutomatic: bool) -> None:
        self.__logger.debug(f"set_camera_mode({isAutomatic}) – not applicable to SDK2")

    def _apply_image_settings(self) -> None:
        self._sdk.SetImage(
            self._hbin, self._vbin,
            self._hstart, self._hend,
            self._vstart, self._vend,
        )

    # -----------------------------------------------------------------------
    # Trigger
    # -----------------------------------------------------------------------
    def set_trigger_source(self, source: str) -> bool:
        lower = source.lower()
        try:
            if lower.startswith(("cont", "internal")):
                mode = _TRIG_INTERNAL
            elif lower.startswith("soft"):
                mode = _TRIG_SOFTWARE
            elif lower.startswith(("ext", "hard")):
                mode = _TRIG_EXTERNAL
            else:
                self.__logger.warning(f"Unknown trigger source '{source}'")
                return False
            ret = self._sdk.SetTriggerMode(mode)
            if ret != _DRV_SUCCESS:
                self.__logger.error(f"SetTriggerMode returned {ret}")
                return False
            self._trigger_mode = mode
            self.__logger.debug(f"Trigger mode set to {mode} ({source})")
            return True
        except Exception as e:
            self.__logger.error(e)
            return False

    def setTriggerSource(self, source: str) -> bool:
        return self.set_trigger_source(source)

    def send_trigger(self) -> bool:
        try:
            ret = self._sdk.SendSoftwareTrigger()
            return ret == _DRV_SUCCESS
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
        mode = getattr(self, "_trigger_mode", _TRIG_INTERNAL)
        if mode == _TRIG_SOFTWARE:
            return "Software Trigger"
        if mode == _TRIG_EXTERNAL:
            return "External Trigger"
        return "Continuous (Internal)"

    # -----------------------------------------------------------------------
    # Live (streaming) acquisition
    # -----------------------------------------------------------------------
    def start_live(self) -> None:
        if self._running:
            return
        self.__logger.debug("Starting SDK2 live acquisition ...")
        self._sdk.SetAcquisitionMode(_ACQ_RUN_TILL_ABORT)
        self._sdk.SetReadMode(_READ_IMAGE)
        self._apply_image_settings()
        self._sdk.PrepareAcquisition()
        self._sdk.StartAcquisition()
        self._exit.clear()
        self._worker = threading.Thread(target=self._acq_loop, daemon=True)
        self._worker.start()
        self._running = True

    def stop_live(self) -> None:
        if not self._running:
            return
        self.__logger.debug("Stopping SDK2 live acquisition ...")
        self._exit.set()
        # Unblock the thread's WaitForAcquisitionTimeOut
        try:
            self._sdk.CancelWait()
        except Exception:
            pass
        if self._worker:
            join_timeout = max(5.0, self._wait_timeout_ms() / 1000.0 + 2.0)
            self._worker.join(timeout=join_timeout)
            self._worker = None
        try:
            self._sdk.AbortAcquisition()
        except Exception as e:
            self.__logger.warning(f"stop_live AbortAcquisition: {e}")
        self._running = False

    def suspend_live(self) -> None:
        """Stop stream but keep device open (HIK API contract)."""
        self.stop_live()

    def prepare_live(self) -> None:
        pass

    def _acq_loop(self) -> None:
        """
        Background thread: block on WaitForAcquisitionTimeOut(), then
        fetch the most recent 16-bit frame and push it to the ring buffer.
        """
        fid = 0
        while not self._exit.is_set():
            timeout_ms = self._wait_timeout_ms()
            ret = self._sdk.WaitForAcquisitionTimeOut(timeout_ms)
            if self._exit.is_set():
                break
            if ret == _DRV_NO_NEW:
                continue
            if ret != _DRV_SUCCESS:
                if not self._exit.is_set():
                    self.__logger.error(f"WaitForAcquisitionTimeOut returned {ret}")
                break

            # Retrieve the latest frame
            size = self._image_size()
            ret2, arr = self._sdk.GetMostRecentImage16(size)
            if ret2 != _DRV_SUCCESS:
                continue

            h, w  = self._image_shape()
            frame = arr.view(np.uint16)[: h * w].reshape(h, w).copy()
            fid  += 1
            ts    = time.time()

            if self.isFlatfielding and self._flatfieldImage is not None:
                try:
                    corrected = frame.astype(np.float32) / self._flatfieldImage
                    frame     = np.clip(corrected, 0, _MAX_16BIT).astype(np.uint16)
                except Exception:
                    pass

            self._buf_ring.append(frame)
            self._fid_ring.append(fid)
            self.frameNumber = fid
            self.timestamp   = ts

    # -----------------------------------------------------------------------
    # Frame retrieval
    # -----------------------------------------------------------------------
    def getLast(
        self,
        returnFrameNumber: bool = False,
        timeout: Optional[float] = None,
        auto_trigger: bool = True,
    ):
        if timeout is None:
            timeout = self._wait_timeout_ms() / 1000.0

        if auto_trigger and getattr(self, "_trigger_mode", _TRIG_INTERNAL) == _TRIG_SOFTWARE:
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
    # Single-shot capture
    # -----------------------------------------------------------------------
    def snap(self) -> Optional[np.ndarray]:
        """Capture one frame outside the continuous loop."""
        timeout_ms    = self._wait_timeout_ms()
        saved_acq_mode = _ACQ_SINGLE_SCAN
        frame          = None
        try:
            self._sdk.SetAcquisitionMode(_ACQ_SINGLE_SCAN)
            self._sdk.SetReadMode(_READ_IMAGE)
            self._apply_image_settings()
            self._sdk.PrepareAcquisition()
            self._sdk.StartAcquisition()
            ret = self._sdk.WaitForAcquisitionTimeOut(timeout_ms)
            if ret == _DRV_SUCCESS:
                size = self._image_size()
                ret2, arr = self._sdk.GetMostRecentImage16(size)
                if ret2 == _DRV_SUCCESS:
                    h, w  = self._image_shape()
                    frame = arr.view(np.uint16)[: h * w].reshape(h, w).copy()
        except Exception as e:
            self.__logger.error(f"snap() failed: {e}")
        finally:
            try:
                self._sdk.AbortAcquisition()
            except Exception:
                pass
            # Restore continuous mode
            try:
                self._sdk.SetAcquisitionMode(_ACQ_RUN_TILL_ABORT)
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

        # SDK2 uses 1-based coordinates
        self._hstart = hpos + 1
        self._hend   = hpos + hsize
        self._vstart = vpos + 1
        self._vend   = vpos + vsize
        self._apply_image_settings()
        self.shape = self._image_shape()
        self.__logger.debug(
            f"ROI set: hstart={self._hstart} hend={self._hend} "
            f"vstart={self._vstart} vend={self._vend}"
        )

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
            return self._image_shape()[1]
        if n == "image_height":
            return self._image_shape()[0]
        self.__logger.debug(f"getPropertyValue: unknown property '{name}'")
        return None

    def get_camera_parameters(self) -> dict:
        return {
            "model_name":     self.model,
            "isRGB":          False,
            "exposure_us":    self.get_exposuretime()[0],
            "trigger_source": self.getTriggerSource(),
            "pixel_encoding": self._activePixelFormat,
            "width":          self._image_shape()[1],
            "height":         self._image_shape()[0],
        }

    # -----------------------------------------------------------------------
    # GUI stub
    # -----------------------------------------------------------------------
    def openPropertiesGUI(self) -> None:
        self.__logger.info("openPropertiesGUI: no native GUI for Andor SDK2 cameras")

    # -----------------------------------------------------------------------
    # Housekeeping
    # -----------------------------------------------------------------------
    def close(self) -> None:
        if self._running:
            self.stop_live()
        try:
            self._sdk.ShutDown()
        except Exception as e:
            self.__logger.warning(f"close() ShutDown: {e}")
        self.__logger.info("Andor SDK2 camera closed")

    def __enter__(self):
        self.start_live()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
