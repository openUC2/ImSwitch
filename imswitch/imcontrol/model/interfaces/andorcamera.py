# interfaces/andor_camera.py
"""
Low-level wrapper around pyAndorSDK3.

Exposes a Hik-like API so that ImSwitch’s DetectorManager can control
the camera without knowing SDK specifics.
"""
from __future__ import annotations

import collections
import threading
import time
from typing import Optional, Tuple, List

import numpy as np
from imswitch.imcommon.model import initLogger

try:
    import pyAndorSDK3
    from pyAndorSDK3 import AndorSDK3
    IS_ANDOR_SDK3_AVAILABLE = True
except ImportError:
    IS_ANDOR_SDK3_AVAILABLE = False
    AndorSDK3 = None



class andorcamera:
    """Minimal Andor SDK3 interface delivering frames through a ring-buffer."""

    def __init__(
        self,
        camera_no: int = 0,
        pixel_encoding: str = "Mono16",
        buffer_count: int = 5,
    ) -> None:
        self.__logger = initLogger(self, tryInheritParent=False)

        # ---- SDK objects ---------------------------------------------------
        self._sdk = AndorSDK3()
        self.cam = self._sdk.GetCamera(camera_no)
        self.cam.open()

        # ---- sensor geometry ----------------------------------------------
        self.cam.PixelEncoding = pixel_encoding
        self.SensorWidth = int(getattr(self.cam, "SensorWidth", self.cam.WidthMax))
        self.SensorHeight = int(getattr(self.cam, "SensorHeight", self.cam.HeightMax))
        self.shape: Tuple[int, int] = (self.SensorHeight, self.SensorWidth)

        # ---- ring-buffer ---------------------------------------------------
        self._buf_ring: collections.deque[np.ndarray] = collections.deque(maxlen=buffer_count)
        self._fid_ring: collections.deque[int] = collections.deque(maxlen=buffer_count)
        self.frameNumber: int = -1
        self.timestamp: float = 0.0

        # ---- streaming -----------------------------------------------------
        self._worker: Optional[threading.Thread] = None
        self._running = False
        self._exit = threading.Event()

        # ---- default parameters -------------------------------------------
        self.set_exposure_time_ms(10.0)
        self.set_gain(0)
        self.set_trigger_source("Continuous")

    # ---------------------------------------------------------------------#
    # Basic parameter helpers                                              #
    # ---------------------------------------------------------------------#
    def set_exposure_time_ms(self, exp_ms: float) -> None:
        exp_s = max(self.cam.min_ExposureTime, min(self.cam.max_ExposureTime, exp_ms / 1000.0))
        self.cam.ExposureTime = exp_s
        self.__logger.debug(f"Exposure set to {exp_s*1000:.2f} ms")

    def get_exposure_time_ms(self) -> float:
        return float(self.cam.ExposureTime) * 1000.0

    def set_gain(self, gain: float) -> None:
        for prop in ("AOIGain", "Gain"):
            if hasattr(self.cam, prop):
                setattr(self.cam, prop, gain)
                break
        self.__logger.debug(f"Gain set to {gain}")

    def get_gain(self) -> float:
        for prop in ("AOIGain", "Gain"):
            if hasattr(self.cam, prop):
                return float(getattr(self.cam, prop))
        return 0.0

    # ---------------------------------------------------------------------#
    # Trigger & acquisition                                                #
    # ---------------------------------------------------------------------#
    def set_trigger_source(self, source: str) -> bool:
        lower = source.lower()
        try:
            if lower.startswith("cont"):
                self.cam.TriggerMode = "Internal"
            elif lower.startswith("soft"):
                self.cam.TriggerMode = "Software"
            elif lower.startswith(("ext", "hard")):
                self.cam.TriggerMode = "External"
            else:
                self.__logger.warning(f"Unknown trigger source '{source}'")
                return False
            self.__logger.debug(f"Trigger source set to {self.cam.TriggerMode}")
            return True
        except Exception as e:
            self.__logger.error(e)
            return False

    def send_trigger(self) -> bool:
        try:
            self.cam.SoftwareTrigger()
            return True
        except Exception as e:
            self.__logger.error(e)
            return False

    # ---------------------------------------------------------------------#
    # Live acquisition                                                     #
    # ---------------------------------------------------------------------#
    def start_live(self) -> None:
        if self._running:
            return
        self.__logger.debug("Starting live …")
        self._queue_initial_buffers()
        self.cam.AcquisitionStart()
        self._exit.clear()
        self._worker = threading.Thread(target=self._acq_loop, daemon=True)
        self._worker.start()
        self._running = True

    def stop_live(self) -> None:
        if not self._running:
            return
        self.__logger.debug("Stopping live …")
        self._exit.set()
        if self._worker:
            self._worker.join()
            self._worker = None
        self.cam.AcquisitionStop()
        self.cam.flush()
        self._running = False

    def _queue_initial_buffers(self) -> None:
        bs = self.cam.ImageSizeBytes
        for _ in range(self._buf_ring.maxlen):
            buf = np.empty((bs,), dtype=np.uint8)
            self.cam.queue(buf, bs)

    def _acq_loop(self) -> None:
        while not self._exit.is_set():
            try:
                acq = self.cam.wait_buffer(1000)  # ms
                frame = acq.image               # numpy view
                fid = int(acq.frame)
                ts = time.time()

                self._buf_ring.append(frame)
                self._fid_ring.append(fid)
                self.frameNumber = fid
                self.timestamp = ts

                self.cam.queue(acq._buf, len(acq._buf))  # re-queue
            except TimeoutError:
                continue
            except Exception as e:
                self.__logger.error(e)
                break

    # ---------------------------------------------------------------------#
    # Frame retrieval helpers                                              #
    # ---------------------------------------------------------------------#
    def getLast(
        self,
        returnFrameNumber: bool = False,
        timeout: float = 1.0,
        auto_trigger: bool = True,
    ):
        # one-shot trigger if required
        if auto_trigger and self.cam.TriggerMode == "Software":
            self.send_trigger()

        t0 = time.time()
        while not self._buf_ring:
            if time.time() - t0 > timeout:
                return (None, None) if returnFrameNumber else None
            time.sleep(0.005)

        if returnFrameNumber:
            return self._buf_ring[-1], self._fid_ring[-1]
        return self._buf_ring[-1]

    def getLastChunk(self):
        frames = list(self._buf_ring)
        fids = list(self._fid_ring)
        self.flushBuffer()
        return np.array(frames), np.array(fids)

    def flushBuffer(self) -> None:
        self._buf_ring.clear()
        self._fid_ring.clear()

    # ---------------------------------------------------------------------#
    # ROI – full frame by default                                          #
    # ---------------------------------------------------------------------#
    def setROI(self, hpos: int = 0, vpos: int = 0, hsize: int | None = None, vsize: int | None = None) -> None:
        if hsize is None:
            hsize = self.SensorWidth
        if vsize is None:
            vsize = self.SensorHeight
        self.cam.AOILeft = hpos
        self.cam.AOITop = vpos
        self.cam.AOIWidth = hsize
        self.cam.AOIHeight = vsize
        self.__logger.debug(f"ROI set to ({hpos},{vpos}) {hsize}×{vsize}")

    # ---------------------------------------------------------------------#
    # House-keeping                                                        #
    # ---------------------------------------------------------------------#
    def close(self) -> None:
        if self._running:
            self.stop_live()
        self.cam.close()
        self.__logger.info("Camera closed")

    # context-manager support
    def __enter__(self):
        self.start_live()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
