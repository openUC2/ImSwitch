"""Allied Vision camera interface (VmbPy, with a legacy VimbaPython fallback).

Feature-for-feature sibling of :class:`~.hikcamera.CameraHIK` so that
``AVManager`` can offer the same ImSwitch surface as ``HikCamManager``:
hardware ROI, binning, manual/auto/once exposure, frame-rate cap,
software/external trigger with synchronous snaps, flip, high-bit-depth mono
and Bayer → RGB conversion.

Both SDKs expose the same GenICam feature API (``get_feature_by_name(...).get()
/ .set() / .get_range()``) and the same ``Frame`` API; they only differ in the
system object, the frame-handler signature and the exception names, which are
normalised at import time below.
"""
import collections
import re
import threading
import time

import numpy as np

from imswitch.imcommon.model import initLogger

isVmbPy = False
isVimba = False
AllocationMode = None
PixelFormat = None
try:
    from vmbpy import VmbSystem, FrameStatus, PixelFormat, VmbCameraError
    try:
        from vmbpy import AllocationMode
    except ImportError:
        pass
    isVmbPy = True
except ImportError:
    try:
        from vimba import Vimba, FrameStatus, PixelFormat
        from vimba import VimbaCameraError as VmbCameraError
        try:
            from vimba import AllocationMode
        except ImportError:
            pass
        isVimba = True
    except ImportError:
        class VmbCameraError(Exception):
            pass

if not (isVmbPy or isVimba):
    print("Neither VmbPy nor legacy VimbaPython installed - Allied Vision cameras unavailable")


_COLOR_FORMAT_PREFIXES = ('Bayer', 'Rgb', 'Bgr', 'Rgba', 'Bgra', 'Yuv', 'YCbCr')
# Preferred formats, most desirable first. Packed formats are avoided because
# ``Frame.as_numpy_ndarray`` cannot map them; 8-bit Bayer is converted to Rgb8.
_MONO_FORMAT_PREFERENCE = ('Mono12', 'Mono10', 'Mono8')
_RGB_FORMAT_PREFERENCE = ('BayerRG8', 'BayerGR8', 'BayerGB8', 'BayerBG8', 'Rgb8', 'Bgr8')


def _bitDepthOfFormat(name) -> int:
    """'Mono12Packed' -> 12, 'BayerRG8' -> 8, unknown -> 8."""
    m = re.search(r'(\d+)', str(name))
    if not m:
        return 8
    bits = int(m.group(1))
    return bits if 8 <= bits <= 16 else 8


def _isColorFormat(name) -> bool:
    return str(name).startswith(_COLOR_FORMAT_PREFIXES)


class CameraAV:
    """Frame grabbing via the SDK's asynchronous streaming callback (no polling)."""

    def __init__(self, camera_id=None, isRGB=None, binning=1, flipImage=(False, False),
                 frame_rate=-1, pixel_format=None):
        """
        :param camera_id: Index (int) or ID (string) of the camera to open; None = first camera.
        :param isRGB: Force colour output; None = detect from the camera's pixel formats.
        :param binning: Startup binning factor (BinningHorizontal/BinningVertical).
        :param flipImage: (flipY, flipX) applied in software to every frame.
        :param frame_rate: Acquisition frame-rate cap in fps; <= 0 disables the limiter.
        :param pixel_format: Explicit PixelFormat name (e.g. "Mono8"); None = negotiate
                             (Mono12 > Mono10 > Mono8, or 8-bit Bayer/RGB for colour).
        """
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=True)

        if not (isVmbPy or isVimba):
            raise RuntimeError("Neither VmbPy nor legacy VimbaPython installed or found.")

        self.model = "AlliedVisionCamera"
        self.is_connected = False
        self.is_streaming = False
        self.flipImage = tuple(flipImage)
        self.binning = 1
        self.frame_rate = -1
        self.exposure_time = None
        self.gain = None
        self.blacklevel = None
        self.trigger_source = "Continuous"
        self.isRGB = False
        self.bitDepth = 8
        self._activePixelFormat = None

        self.SensorWidth = 0
        self.SensorHeight = 0
        # Current hardware ROI (full-sensor coordinates, binned pixels)
        self.ROI_hpos = 0
        self.ROI_vpos = 0
        self.ROI_width = 0
        self.ROI_height = 0
        # Only used when the camera refuses hardware ROI features
        self._softwareROI = False

        self.NBuffer = 3
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameid_buffer = collections.deque(maxlen=self.NBuffer)
        self._frame_lock = threading.Lock()
        self.frame = None
        self.frameNumber = -1
        self.timestamp = 0
        self.lastFrameFromBuffer = None
        self.lastFrameId = -1
        self._streamStats = self._newStreamStats()

        self._vmb_system = None
        self._vimba = None
        self._camera = None
        self._camera_open = False
        self._featExposure = None

        self._open_camera(camera_id)
        try:
            self._configure(isRGB, binning, frame_rate, pixel_format)
        except Exception:
            self._cleanup_context_managers()
            raise
        self.is_connected = True
        self.__logger.info(
            f"CameraAV ready: model={self.model}, isRGB={self.isRGB}, "
            f"format={self._activePixelFormat} ({self.bitDepth} bit), "
            f"sensor={self.SensorWidth}x{self.SensorHeight}, binning={self.binning}")

    def _configure(self, isRGB, binning, frame_rate, pixel_format):
        """Startup configuration once the camera context is open."""
        if isRGB is not None:
            self.isRGB = bool(isRGB)
        else:
            self.isRGB = self._detect_rgb()
        self._negotiatePixelFormat(pixel_format)
        if self.isRGB and not _isColorFormat(self._activePixelFormat):
            self.__logger.warning(
                f"isRGB requested but the camera delivers {self._activePixelFormat}; "
                "frames will be monochrome")

        # Exposure feature name differs between Alvium (SFNC) and older GigE models
        self._featExposure = self._feature('ExposureTime') or self._feature('ExposureTimeAbs')

        self.setBinning(binning)
        self.set_frame_rate(frame_rate)
        self.setTriggerSource("Continuous")

    # ------------------------------------------------------------------
    # Open / close
    # ------------------------------------------------------------------
    def _open_camera(self, camera_id):
        if isVmbPy:
            self._vmb_system = VmbSystem.get_instance()
            self._vmb_system.__enter__()
            system = self._vmb_system
        else:
            self._vimba = Vimba.get_instance()
            self._vimba.__enter__()
            system = self._vimba

        try:
            cams = system.get_all_cameras()
            if not cams:
                raise RuntimeError("No Allied Vision cameras found.")

            if isinstance(camera_id, str) and camera_id.strip().isdigit():
                camera_id = int(camera_id)
            if camera_id is None:
                self._camera = cams[0]
            elif isinstance(camera_id, int):
                if camera_id < 0 or camera_id >= len(cams):
                    raise RuntimeError(f"Invalid camera index: {camera_id} ({len(cams)} cameras found)")
                self._camera = cams[camera_id]
            else:
                try:
                    self._camera = system.get_camera_by_id(str(camera_id))
                except VmbCameraError as e:
                    raise RuntimeError(f"Failed to open camera with ID '{camera_id}'.") from e

            # Keep the camera context open for the lifetime of this object
            self._camera.__enter__()
            self._camera_open = True

            # GigE only: negotiate the packet size
            try:
                if isVmbPy:
                    streams = self._camera.get_streams()
                    feat = streams[0].get_feature_by_name('GVSPAdjustPacketSize') if streams else None
                else:
                    feat = self._camera.get_feature_by_name('GVSPAdjustPacketSize')
                if feat is not None:
                    feat.run()
                    while not feat.is_done():
                        time.sleep(0.001)
            except Exception:
                pass

            self._trySet('AcquisitionMode', 'Continuous')

            try:
                self.model = self._camera.get_name()
            except Exception:
                self.model = "AlliedVisionCamera"

            self._readSensorSize()
        except Exception:
            self._cleanup_context_managers()
            raise

        self.__logger.debug(f"Opened camera '{self.model}' (ID/index: {camera_id}).")

    def _readSensorSize(self):
        """Full frame size at the current binning (WidthMax/HeightMax shrink with binning)."""
        for wName, hName in (('WidthMax', 'HeightMax'), ('SensorWidth', 'SensorHeight'),
                             ('Width', 'Height')):
            fw, fh = self._feature(wName), self._feature(hName)
            if fw is None or fh is None:
                continue
            try:
                self.SensorWidth = int(fw.get())
                self.SensorHeight = int(fh.get())
                return
            except Exception:
                continue
        self.SensorWidth = self.SensorHeight = 0

    def _detect_rgb(self) -> bool:
        """A colour camera lists at least one Bayer/RGB/YUV pixel format."""
        try:
            fmts = self._camera.get_pixel_formats()
            return any(_isColorFormat(f) for f in fmts)
        except Exception as e:
            self.__logger.warning(f"Could not detect RGB capability: {e}")
            return False

    def _cleanup_context_managers(self):
        try:
            if self._camera is not None and self._camera_open:
                self._camera.__exit__(None, None, None)
        except Exception as e:
            self.__logger.warning(f"Error exiting camera context: {e}")
        self._camera_open = False
        try:
            if self._vmb_system is not None:
                self._vmb_system.__exit__(None, None, None)
            if self._vimba is not None:
                self._vimba.__exit__(None, None, None)
        except Exception as e:
            self.__logger.warning(f"Error exiting SDK context: {e}")
        self._vmb_system = None
        self._vimba = None

    def close(self):
        if self.is_streaming:
            self.stop_live()
        self._cleanup_context_managers()
        self.is_connected = False
        self.__logger.debug("Camera closed and SDK context cleaned up.")

    # ------------------------------------------------------------------
    # Feature helpers
    # ------------------------------------------------------------------
    def _feature(self, name):
        """Feature object or None if the camera does not have it."""
        if self._camera is None:
            return None
        try:
            return self._camera.get_feature_by_name(name)
        except Exception:
            return None

    def _trySet(self, name, value) -> bool:
        feat = self._feature(name)
        if feat is None:
            self.__logger.debug(f"Feature {name} not available")
            return False
        try:
            feat.set(value)
            return True
        except Exception as e:
            self.__logger.warning(f"Could not set {name}={value!r}: {e}")
            return False

    @staticmethod
    def _alignToFeature(feat, value):
        """Clamp ``value`` into the feature's range and round down to its increment."""
        lo, hi = feat.get_range()
        try:
            inc = feat.get_increment() or 1
        except Exception:
            inc = 1
        value = min(max(value, lo), hi)
        if inc and inc > 1:
            value = lo + int((value - lo) // inc) * inc
        return int(value)

    # ------------------------------------------------------------------
    # Pixel format
    # ------------------------------------------------------------------
    def _negotiatePixelFormat(self, requested=None):
        try:
            available = {str(f): f for f in self._camera.get_pixel_formats()}
        except Exception as e:
            self.__logger.warning(f"Could not list pixel formats: {e}")
            available = {}

        if requested:
            candidates = [str(requested)]
        elif self.isRGB:
            candidates = list(_RGB_FORMAT_PREFERENCE)
        else:
            candidates = list(_MONO_FORMAT_PREFERENCE)

        lower = {k.lower(): v for k, v in available.items()}
        for name in candidates:
            fmt = available.get(name) or lower.get(name.lower())
            if fmt is None:
                continue
            try:
                self._camera.set_pixel_format(fmt)
                self._activePixelFormat = str(fmt)
                self.__logger.info(f"Pixel format set to {self._activePixelFormat}")
                break
            except Exception as e:
                self.__logger.info(f"Pixel format {name} rejected: {e}")
        else:
            try:
                self._activePixelFormat = str(self._camera.get_pixel_format())
            except Exception:
                self._activePixelFormat = 'Mono8'
            self.__logger.warning(
                f"None of {candidates} could be set (available: {list(available)}); "
                f"keeping {self._activePixelFormat}")
        self.bitDepth = _bitDepthOfFormat(self._activePixelFormat)

    def set_pixel_format(self, fmt):
        """Switch the PixelFormat by name (e.g. "Mono8"); restarts the stream if needed."""
        if fmt is None:
            return
        was_streaming = self.is_streaming
        if was_streaming:
            self.stop_live()
        try:
            self._negotiatePixelFormat(str(fmt))
        finally:
            if was_streaming:
                self.start_live()

    # ------------------------------------------------------------------
    # Frame delivery
    # ------------------------------------------------------------------
    def _frameToNumpy(self, frame):
        """Owned (h, w) or (h, w, 3) array from an SDK frame, flip applied.

        ``as_numpy_ndarray`` is a zero-copy view of the SDK buffer, which is
        reused after ``queue_frame``; the result is therefore always copied.
        """
        name = str(frame.get_pixel_format())
        arr = None
        if name.startswith('Mono') and not (name.endswith('Packed') or name.endswith('p')):
            arr = frame.as_numpy_ndarray()
        elif name in ('Rgb8', 'Bgr8'):
            arr = frame.as_numpy_ndarray()
            if name == 'Bgr8':
                arr = arr[..., ::-1]
        else:
            # Bayer / packed / YUV: let the SDK's image-transform library convert
            target = PixelFormat.Rgb8 if (self.isRGB or _isColorFormat(name)) else PixelFormat.Mono8
            if name.startswith('Mono') and self.bitDepth > 8:
                target = PixelFormat.Mono16
            try:
                converted = frame.convert_pixel_format(target)
                arr = (converted if converted is not None else frame).as_numpy_ndarray()
            except Exception as e:
                if name.startswith('Bayer') and name.endswith('8'):
                    # OpenCV fallback for 8-bit Bayer (same mapping as hikcamera)
                    import cv2
                    bayer_map = {'BayerRG8': cv2.COLOR_BayerRG2RGB, 'BayerBG8': cv2.COLOR_BayerBG2RGB,
                                 'BayerGB8': cv2.COLOR_BayerGB2RGB, 'BayerGR8': cv2.COLOR_BayerGR2RGB}
                    raw = frame.as_numpy_ndarray()[:, :, 0]
                    arr = cv2.cvtColor(raw, bayer_map[name])
                else:
                    self.__logger.error(f"Cannot convert pixel format {name}: {e}")
                    return None

        if arr.ndim == 3 and arr.shape[2] == 1:
            arr = arr[:, :, 0]

        if self._softwareROI and self.ROI_width and self.ROI_height:
            cropped = arr[self.ROI_vpos:self.ROI_vpos + self.ROI_height,
                          self.ROI_hpos:self.ROI_hpos + self.ROI_width]
            if cropped.size:
                arr = cropped

        if self.flipImage[0]:
            arr = np.flip(arr, axis=0)
        if self.flipImage[1]:
            arr = np.flip(arr, axis=1)
        return arr.copy()

    def _frame_handler(self, *args):
        """VmbPy calls ``handler(cam, stream, frame)``, legacy Vimba ``handler(cam, frame)``."""
        cam, frame = args[0], args[-1]
        t_entry = time.time()
        try:
            if frame.get_status() == FrameStatus.Complete:
                data = self._frameToNumpy(frame)
                if data is not None:
                    fid = frame.get_id()
                    try:
                        ts = frame.get_timestamp()
                    except Exception:
                        ts = 0
                    with self._frame_lock:
                        self.frame_buffer.append(data)
                        self.frameid_buffer.append(fid)
                        self.frame = data
                        self.frameNumber = fid
                        self.timestamp = ts
                    self._updateStreamStats(t_entry)
            else:
                self._streamStats["incomplete_frames"] += 1
        except Exception as e:
            self.__logger.error(f"Error in frame handler: {e}")
        finally:
            try:
                cam.queue_frame(frame)
            except Exception as e:
                self.__logger.debug(f"queue_frame failed: {e}")

    def start_live(self):
        if self.is_streaming:
            return
        self.flushBuffer()
        self._streamStats = self._newStreamStats()
        kwargs = dict(handler=self._frame_handler, buffer_count=5)
        if AllocationMode is not None:
            kwargs['allocation_mode'] = AllocationMode.AnnounceFrame
        try:
            self._camera.start_streaming(**kwargs)
        except TypeError:
            kwargs.pop('allocation_mode', None)
            self._camera.start_streaming(**kwargs)
        self.is_streaming = True
        self.__logger.debug("Camera streaming started.")

    def stop_live(self):
        if not self.is_streaming:
            return
        try:
            self._camera.stop_streaming()
        except Exception as e:
            self.__logger.warning(f"Error stopping camera streaming: {e}")
        self.is_streaming = False
        self.__logger.debug("Camera streaming stopped.")

    def suspend_live(self):
        self.stop_live()

    def prepare_live(self):
        pass

    def getLast(self, returnFrameNumber: bool = False, timeout: float = 1.0,
                auto_trigger: bool = False):
        """Newest frame in the ring buffer; ``(None, None)``/``None`` after ``timeout``."""
        if auto_trigger and str(self.trigger_source).lower() in (
                "internal trigger", "software", "software trigger"):
            self.send_trigger()

        t0 = time.time()
        while not self.frame_buffer:
            if time.time() - t0 > timeout:
                return (None, None) if returnFrameNumber else None
            if self.lastFrameFromBuffer is not None:
                return ((self.lastFrameFromBuffer, self.lastFrameId) if returnFrameNumber
                        else self.lastFrameFromBuffer)
            time.sleep(0.005)

        with self._frame_lock:
            latest_frame = self.frame_buffer[-1]
            latest_frame_id = self.frameid_buffer[-1]
        self.lastFrameFromBuffer = latest_frame
        self.lastFrameId = latest_frame_id
        if returnFrameNumber:
            return latest_frame, latest_frame_id
        return latest_frame

    def getLastChunk(self):
        """Return *and clear* the ring buffer as ``(frames, ids)`` numpy stacks."""
        with self._frame_lock:
            frames = list(self.frame_buffer)
            ids = list(self.frameid_buffer)
        self.flushBuffer()
        self.lastFrameFromBuffer = frames[-1] if frames else None
        return np.array(frames), np.array(ids)

    def flushBuffer(self):
        with self._frame_lock:
            self.frame_buffer.clear()
            self.frameid_buffer.clear()
        self.lastFrameFromBuffer = None
        self.lastFrameId = -1

    def getFrameNumber(self):
        return self.frameNumber

    # ------------------------------------------------------------------
    # Exposure / gain / black level
    # ------------------------------------------------------------------
    def get_exposuretime(self):
        """(current, min, max) in microseconds, like CameraHIK."""
        feat = self._featExposure
        if feat is None:
            return (None, None, None)
        try:
            lo, hi = feat.get_range()
            return (float(feat.get()), float(lo), float(hi))
        except Exception as e:
            self.__logger.error(f"Get exposure time failed: {e}")
            return (None, None, None)

    def set_exposure_time(self, exposure_ms):
        """Exposure in milliseconds (the camera feature is in microseconds)."""
        feat = self._featExposure
        if feat is None:
            self.__logger.warning("Camera has no ExposureTime feature")
            return
        try:
            us = float(exposure_ms) * 1000.0
        except (TypeError, ValueError):
            self.__logger.error(f"set_exposure_time: {exposure_ms!r} is not a number")
            return
        try:
            lo, hi = feat.get_range()
            us = min(max(us, lo), hi)
            feat.set(us)
            self.exposure_time = us / 1000.0
        except Exception as e:
            self.__logger.error(f"Set exposure time {exposure_ms} ms failed: {e}")

    def set_exposure_mode(self, exposure_mode="manual"):
        """manual → ExposureAuto Off, auto → Continuous, once → Once."""
        mode = str(exposure_mode).lower()
        entry = {"manual": "Off", "off": "Off", "auto": "Continuous", "continuous": "Continuous",
                 "once": "Once", "single": "Once"}.get(mode)
        if entry is None:
            self.__logger.warning(f"Exposure mode not recognized: {exposure_mode}")
            return
        self._trySet('ExposureAuto', entry)

    def get_exposure_mode(self):
        feat = self._feature('ExposureAuto')
        if feat is None:
            return 'manual'
        try:
            return {'off': 'manual', 'once': 'manual', 'continuous': 'auto'}.get(
                str(feat.get()).lower(), 'manual')
        except Exception:
            return 'manual'

    def set_camera_mode(self, isAutomatic):
        self.set_exposure_mode(isAutomatic)

    def get_gain(self):
        """(current, min, max)."""
        feat = self._feature('Gain')
        if feat is None:
            return (None, None, None)
        try:
            lo, hi = feat.get_range()
            return (float(feat.get()), float(lo), float(hi))
        except Exception as e:
            self.__logger.error(f"Get gain failed: {e}")
            return (None, None, None)

    def set_gain(self, gain):
        feat = self._feature('Gain')
        if feat is None:
            self.__logger.warning("Camera has no Gain feature")
            return
        try:
            lo, hi = feat.get_range()
            value = min(max(float(gain), lo), hi)
            feat.set(value)
            self.gain = value
        except Exception as e:
            self.__logger.error(f"Set gain {gain} failed: {e}")

    def get_blacklevel(self):
        feat = self._feature('BlackLevel')
        if feat is None:
            return None
        try:
            return float(feat.get())
        except Exception:
            return None

    def set_blacklevel(self, blacklevel):
        feat = self._feature('BlackLevel')
        if feat is None:
            self.__logger.debug("Camera has no BlackLevel feature")
            return
        try:
            lo, hi = feat.get_range()
            value = min(max(float(blacklevel), lo), hi)
            feat.set(value)
            self.blacklevel = value
        except Exception as e:
            self.__logger.error(f"Set black level {blacklevel} failed: {e}")

    # ------------------------------------------------------------------
    # Frame rate
    # ------------------------------------------------------------------
    def set_frame_rate(self, frame_rate):
        """Cap the acquisition rate; ``frame_rate <= 0`` disables the limiter.

        Returns the rate the camera actually applied (-1 when free-running).
        """
        try:
            frame_rate = float(frame_rate)
        except (TypeError, ValueError):
            self.__logger.error(f"set_frame_rate: {frame_rate!r} is not a number; unchanged")
            return self.frame_rate

        fEnable = self._feature('AcquisitionFrameRateEnable')
        fRate = self._feature('AcquisitionFrameRate') or self._feature('AcquisitionFrameRateAbs')
        if fRate is None:
            self.__logger.debug("Camera has no AcquisitionFrameRate feature")
            self.frame_rate = -1
            return self.frame_rate

        try:
            if frame_rate <= 0:
                if fEnable is not None:
                    fEnable.set(False)
                self.frame_rate = -1
                self.__logger.info("Frame rate limiter disabled (free-run)")
                return self.frame_rate

            if fEnable is not None:
                fEnable.set(True)
            lo, hi = fRate.get_range()
            if frame_rate < lo or frame_rate > hi:
                clamped = min(max(frame_rate, lo), hi)
                self.__logger.warning(
                    f"Frame rate {frame_rate:.3f} fps outside the camera range "
                    f"[{lo:.3f}, {hi:.3f}] - using {clamped:.3f} fps")
                frame_rate = clamped
            fRate.set(frame_rate)
            self.frame_rate = float(fRate.get())
            self.__logger.info(f"Frame rate set to {self.frame_rate:.3f} fps (requested {frame_rate:.3f})")
        except Exception as e:
            self.__logger.error(f"Set frame rate {frame_rate} failed: {e}")
        return self.frame_rate

    def get_frame_rate(self):
        """What the camera holds now: -1 when the limiter is off."""
        fEnable = self._feature('AcquisitionFrameRateEnable')
        fRate = self._feature('AcquisitionFrameRate') or self._feature('AcquisitionFrameRateAbs')
        try:
            if fEnable is not None and not fEnable.get():
                return -1
            if fRate is not None:
                return float(fRate.get())
        except Exception:
            pass
        return self.frame_rate

    # ------------------------------------------------------------------
    # Binning / ROI
    # ------------------------------------------------------------------
    def getBinningRange(self):
        """(min, max) of BinningHorizontal, or (1, 1) if the camera cannot bin."""
        feat = self._feature('BinningHorizontal')
        if feat is None:
            return (1, 1)
        try:
            lo, hi = feat.get_range()
            return (int(lo), int(hi))
        except Exception:
            return (1, 1)

    def setBinning(self, binning=1):
        """Apply BinningHorizontal/BinningVertical and reset the ROI to the binned full frame.

        Must be called while not streaming. Returns the binning in effect.
        """
        binning = int(binning)
        fH, fV = self._feature('BinningHorizontal'), self._feature('BinningVertical')
        if fH is None:
            if binning != 1:
                self.__logger.warning("Camera has no binning feature; staying at 1x1")
            self.binning = 1
            self._readSensorSize()
            return self.binning
        try:
            # Offsets first: Width/Height maxima depend on binning and offset
            for name in ('OffsetX', 'OffsetY'):
                feat = self._feature(name)
                if feat is not None:
                    feat.set(feat.get_range()[0])
            lo, hi = fH.get_range()
            binning = int(min(max(binning, lo), hi))
            fH.set(binning)
            if fV is not None:
                fV.set(binning)
            # Deliver the full binned frame (some models keep the old, now
            # out-of-range Width/Height and silently revert the binning)
            for name in ('Width', 'Height'):
                feat = self._feature(name)
                if feat is not None:
                    feat.set(feat.get_range()[1])
            self.binning = int(fH.get())
        except Exception as e:
            self.__logger.error(f"Set binning {binning} failed: {e}")
        self._readSensorSize()
        self.ROI_hpos = self.ROI_vpos = 0
        self.ROI_width, self.ROI_height = self.SensorWidth, self.SensorHeight
        self.flushBuffer()
        self.__logger.debug(
            f"Binning set to {self.binning}x{self.binning}, "
            f"frame size {self.SensorWidth}x{self.SensorHeight}")
        return self.binning

    def setROI(self, hpos=None, vpos=None, hsize=None, vsize=None):
        """Hardware ROI via OffsetX/OffsetY/Width/Height (camera must not be streaming).

        Values are aligned to the feature increments; the tuple actually applied
        is returned. Falls back to a software crop if the camera refuses.
        """
        hpos = self.ROI_hpos if hpos is None else int(hpos)
        vpos = self.ROI_vpos if vpos is None else int(vpos)
        hsize = self.ROI_width if hsize is None else int(hsize)
        vsize = self.ROI_height if vsize is None else int(vsize)

        fW, fH = self._feature('Width'), self._feature('Height')
        fX, fY = self._feature('OffsetX'), self._feature('OffsetY')
        applied = None
        if None not in (fW, fH, fX, fY):
            try:
                # Shrink first (offset 0) so the requested size is always in range
                fX.set(fX.get_range()[0])
                fY.set(fY.get_range()[0])
                fW.set(self._alignToFeature(fW, hsize))
                fH.set(self._alignToFeature(fH, vsize))
                fX.set(self._alignToFeature(fX, hpos))
                fY.set(self._alignToFeature(fY, vpos))
                applied = (int(fX.get()), int(fY.get()), int(fW.get()), int(fH.get()))
                self._softwareROI = False
            except Exception as e:
                self.__logger.warning(f"Hardware ROI failed ({e}); using software crop")
        if applied is None:
            self._softwareROI = True
            hsize = min(hsize, max(self.SensorWidth - hpos, 0)) or self.SensorWidth
            vsize = min(vsize, max(self.SensorHeight - vpos, 0)) or self.SensorHeight
            applied = (hpos, vpos, hsize, vsize)

        self.ROI_hpos, self.ROI_vpos, self.ROI_width, self.ROI_height = applied
        self.flushBuffer()
        self.__logger.debug(
            f"ROI set to x={self.ROI_hpos}, y={self.ROI_vpos}, "
            f"w={self.ROI_width}, h={self.ROI_height}"
            + (" (software)" if self._softwareROI else ""))
        return applied

    # ------------------------------------------------------------------
    # Trigger
    # ------------------------------------------------------------------
    def getTriggerTypes(self):
        if not self.is_connected:
            return ["Camera not connected"]
        return ["Continuous (Free Run)", "Software Trigger", "External Trigger (Line0)"]

    def getTriggerSource(self) -> str:
        if not self.is_connected:
            return "Camera not connected"
        tlow = str(self.trigger_source).lower()
        if "soft" in tlow or "internal" in tlow:
            return "Software Trigger"
        if "ext" in tlow or "hard" in tlow or "line" in tlow:
            return "External Trigger (Line0)"
        return "Continuous (Free Run)"

    def _externalTriggerLine(self, requested: str):
        """Trigger input line: the one named in ``requested`` if the camera has it,
        else the first Line* entry (Alvium: Line0, older GigE: Line1)."""
        feat = self._feature('TriggerSource')
        names = []
        if feat is not None:
            try:
                names = [str(e) for e in feat.get_available_entries()]
            except Exception:
                names = []
        m = re.search(r'line\s*(\d+)', requested.lower())
        if m:
            wanted = f"Line{m.group(1)}"
            if wanted in names or not names:
                return wanted
        for n in names:
            if n.lower().startswith('line'):
                return n
        return 'Line0'

    def setTriggerSource(self, trigger_source):
        """
        Continous          -> free-run (TriggerMode Off)
        Internal trigger   -> software (SDK) trigger
        External trigger   -> hardware trigger on Line0 (or the line named)
        """
        was_streaming = self.is_streaming
        if was_streaming:
            self.suspend_live()
        tlow = str(trigger_source).lower()
        try:
            fMode, fSrc = self._feature('TriggerMode'), self._feature('TriggerSource')
            if fMode is None:
                self.__logger.warning("Camera has no TriggerMode feature")
                return False
            self._trySet('TriggerSelector', 'FrameStart')
            if "cont" in tlow or "free" in tlow:
                fMode.set('Off')
                self.__logger.info("Trigger source set to continuous (free run)")
            elif "soft" in tlow or "internal" in tlow:
                fMode.set('On')
                fSrc.set('Software')
                self.__logger.info("Trigger source set to software trigger")
            elif "ext" in tlow or "hard" in tlow or "line" in tlow:
                fMode.set('On')
                fSrc.set(self._externalTriggerLine(tlow))
                self.__logger.info(f"Trigger source set to external trigger ({fSrc.get()})")
            else:
                self.__logger.warning(f"Unknown trigger source: {trigger_source}")
                return False
            self.trigger_source = trigger_source
            return True
        except Exception as e:
            self.__logger.error(f"Set trigger source {trigger_source} failed: {e}")
            return False
        finally:
            if was_streaming:
                self.start_live()

    def send_trigger(self):
        """Fire one software trigger pulse (trigger source must be software)."""
        feat = self._feature('TriggerSoftware')
        if feat is None:
            self.__logger.error("Camera has no TriggerSoftware command")
            return False
        try:
            feat.run()
            return True
        except Exception as e:
            self.__logger.error(f"Software trigger failed: {e}")
            return False

    def snapSoftwareTrigger(self, timeout: float = 2.0):
        """Fire one software trigger and return the frame it produces.

        The frame is exposed *after* this call, so it is guaranteed to be
        post-move/post-settle (see CameraHIK.snapSoftwareTrigger).
        """
        self.flushBuffer()
        prev_id = self.frameNumber
        if not self.send_trigger():
            return self.getLast()
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.frame_buffer and self.frameid_buffer[-1] != prev_id:
                return self.frame_buffer[-1]
            time.sleep(0.003)
        self.__logger.warning("snapSoftwareTrigger: timed out waiting for triggered frame")
        return self.frame_buffer[-1] if self.frame_buffer else None

    # ------------------------------------------------------------------
    # Generic property access (ImSwitch manager API)
    # ------------------------------------------------------------------
    def setPropertyValue(self, property_name, property_value):
        if property_name == "gain":
            self.set_gain(property_value)
        elif property_name in ("exposure", "exposureTime"):
            self.set_exposure_time(property_value)
        elif property_name == "exposure_mode":
            self.set_exposure_mode(property_value)
        elif property_name == "blacklevel":
            self.set_blacklevel(property_value)
        elif property_name == "frame_rate":
            return self.set_frame_rate(property_value)
        elif property_name == "trigger_source":
            self.setTriggerSource(property_value)
        elif property_name == "mode":
            self.set_camera_mode(isAutomatic=property_value)
        elif property_name == "pixel_format":
            self.set_pixel_format(property_value)
        elif property_name == "isRGB":
            self.isRGB = bool(property_value)
        elif property_name == "roi_size":
            self.roi_size = property_value
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return False
        return property_value

    def getPropertyValue(self, property_name):
        if property_name == "gain":
            return self.get_gain()[0]
        elif property_name in ("exposure", "exposureTime"):
            cur = self.get_exposuretime()[0]
            return cur / 1000.0 if cur is not None else None
        elif property_name == "exposure_mode":
            return self.get_exposure_mode()
        elif property_name == "blacklevel":
            return self.get_blacklevel()
        elif property_name in ("image_width", "Width"):
            return self.ROI_width or self.SensorWidth
        elif property_name in ("image_height", "Height"):
            return self.ROI_height or self.SensorHeight
        elif property_name == "frame_number":
            return self.frameNumber
        elif property_name == "frame_rate":
            return self.get_frame_rate()
        elif property_name == "trigger_source":
            return self.trigger_source
        elif property_name == "pixel_format":
            return self._activePixelFormat
        elif property_name == "binning":
            return self.binning
        elif property_name == "isRGB":
            return self.isRGB
        elif property_name == "roi_size":
            return getattr(self, 'roi_size', None)
        else:
            self.__logger.warning(f'Property {property_name} does not exist')
            return None

    def get_camera_parameters(self):
        """Hardware snapshot for getCameraStatus()."""
        params = {"model_name": self.model, "isRGB": self.isRGB,
                  "pixel_format": self._activePixelFormat, "bit_depth": self.bitDepth,
                  "binning": self.binning, "width": self.ROI_width, "height": self.ROI_height,
                  "sensor_width": self.SensorWidth, "sensor_height": self.SensorHeight,
                  "trigger_source": self.getTriggerSource(), "frame_rate": self.get_frame_rate()}
        exp = self.get_exposuretime()
        if exp[0] is not None:
            params.update(exposure_current=exp[0], exposure_min=exp[1], exposure_max=exp[2])
        gain = self.get_gain()
        if gain[0] is not None:
            params.update(gain_current=gain[0], gain_min=gain[1], gain_max=gain[2])
        for name in ('DeviceSerialNumber', 'DeviceFirmwareVersion', 'DeviceTemperature'):
            feat = self._feature(name)
            if feat is not None:
                try:
                    params[name] = feat.get()
                except Exception:
                    pass
        return params

    def openPropertiesGUI(self):
        pass

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def _newStreamStats(self):
        now = time.time()
        return {"frames": 0, "incomplete_frames": 0, "fps_callback": 0.0,
                "_started": now, "_last_entry_t": None, "_window_t0": now, "_window_n": 0}

    def _updateStreamStats(self, t_entry):
        s = self._streamStats
        s["frames"] += 1
        s["_last_entry_t"] = t_entry
        s["_window_n"] += 1
        dt = t_entry - s["_window_t0"]
        if dt >= 1.0:
            s["fps_callback"] = s["_window_n"] / dt
            s["_window_t0"] = t_entry
            s["_window_n"] = 0

    def getStreamDiagnostics(self) -> dict:
        s = {k: v for k, v in self._streamStats.items() if not k.startswith("_")}
        last = self._streamStats.get("_last_entry_t")
        s["latest_frame_age_ms"] = None if last is None else (time.time() - last) * 1000.0
        s["uptime_s"] = time.time() - self._streamStats.get("_started", time.time())
        s["is_streaming"] = self.is_streaming
        s["buffered_frames"] = len(self.frame_buffer)
        return s

    def getDiagnostics(self) -> dict:
        frame = self.frame
        frame_info = {}
        if frame is not None:
            frame_info = {"shape": list(frame.shape), "dtype": str(frame.dtype),
                          "min": int(frame.min()), "max": int(frame.max())}
        return {"pixel_format": self._activePixelFormat, "bit_depth": self.bitDepth,
                "isRGB": self.isRGB, "is_streaming": self.is_streaming,
                "binning": self.binning, "frame": frame_info,
                "stream": self.getStreamDiagnostics()}

    def __enter__(self):
        self.start_live()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
