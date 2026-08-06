import collections
import threading
import time
from typing import List

import numpy as np

from imswitch.imcommon.model import initLogger

try:
    from imswitch.imcontrol.model.interfaces.toupcamsdk import toupcam
    TOUPCAM_AVAILABLE = True
    TOUPCAM_IMPORT_ERROR = None
except Exception as e:  # missing native library, broken install, ...
    toupcam = None
    TOUPCAM_AVAILABLE = False
    TOUPCAM_IMPORT_ERROR = e


class CameraToupcam:
    """ToupTek (Toupcam) camera wrapper that grabs frames via the SDK's
    pull-mode callback (no polling), mirroring the CameraHIK interface so the
    detector manager and the triggered-grab protocol work identically.

    Frames are pulled with PullImageV3 into a preallocated full-sensor buffer
    and stored in the ring buffer as owned copies (never views), so entries
    can not alias SDK memory that is overwritten by later frames.
    """

    def __init__(self, cameraNo=None, exposure_time=100, gain=0, frame_rate=-1,
                 blacklevel=0, isRGB=False, binning=1, flipImage=(False, False)):
        super().__init__()
        self.__logger = initLogger(self, tryInheritParent=False)

        if not TOUPCAM_AVAILABLE:
            raise RuntimeError(
                f"Toupcam SDK not available: {TOUPCAM_IMPORT_ERROR}. "
                "Install the native library (see toupcamsdk/install_toupcam_libs.py) "
                "or set TOUPCAM_SDK_DIR / IMSWITCH_TOUPCAM_LIB."
            )

        self.model = "CameraToupcam"
        self.shape = (0, 0)
        self.is_connected = False
        self.is_streaming = False
        self.downsamplepreview = 1

        self.blacklevel = blacklevel
        self.exposure_time = exposure_time  # ms (UI convention, like CameraHIK)
        self.gain = gain
        self.frame_rate = frame_rate
        self.cameraNo = cameraNo if cameraNo is not None else 0
        self.flipImage = flipImage  # (flipY, flipX)
        self.isRGB = bool(isRGB)
        self.binning = binning
        self.trigger_source = "Continous"

        self.NBuffer = 3
        self.frame_buffer = collections.deque(maxlen=self.NBuffer)
        self.frameid_buffer = collections.deque(maxlen=self.NBuffer)
        self.lastFrameFromBuffer = None
        self.lastFrameId = -1
        self.frameNumber = -1
        self.timestamp = 0

        self.SensorHeight = 0
        self.SensorWidth = 0
        self.max_adu = 255

        self.hcam = None
        self._pull_lock = threading.Lock()
        self._reconnect_lock = threading.Lock()
        self._reconnecting = False
        self._resume_stream_after_reconnect = False
        self._streamStats = self._newStreamStats()

        self._open_camera(self.cameraNo)

        # apply constructor defaults to the hardware
        try:
            if exposure_time and exposure_time > 0:
                self.set_exposure_time(exposure_time)
            if gain and gain > 0:
                self.set_gain(gain)
            if blacklevel and blacklevel > 0:
                self.set_blacklevel(blacklevel)
            if binning and binning > 1:
                self.setBinning(binning)
            self.set_frame_rate(frame_rate)
        except Exception as e:
            self.__logger.warning(f"Applying initial camera settings failed: {e}")

    # ---------------------------------------------------------------------
    # Camera discovery / opening
    # ---------------------------------------------------------------------
    def _open_camera(self, number: int):
        devices = toupcam.Toupcam.EnumV2()
        if not devices:
            raise RuntimeError("No Toupcam camera found (check USB / udev rules)")
        if number is None or number >= len(devices):
            self.__logger.warning(
                f"Camera index {number} out of range ({len(devices)} found), using 0"
            )
            number = 0
        dev = devices[number]
        self.deviceName = dev.displayname
        self._deviceFlags = dev.model.flag

        self.hcam = toupcam.Toupcam.Open(dev.id)
        if self.hcam is None:
            raise RuntimeError(f"Failed to open Toupcam camera {dev.displayname}")

        # capabilities from the model flag
        flag = self._deviceFlags
        self._hasTEC = bool(flag & toupcam.TOUPCAM_FLAG_TEC_ONOFF)
        self._hasGetTemperature = bool(flag & toupcam.TOUPCAM_FLAG_GETTEMPERATURE)
        self._hasFan = bool(flag & toupcam.TOUPCAM_FLAG_FAN)
        self._hasBlacklevel = bool(flag & toupcam.TOUPCAM_FLAG_BLACKLEVEL)
        self._hasSoftwareTrigger = bool(flag & toupcam.TOUPCAM_FLAG_TRIGGER_SOFTWARE)
        self._hasExternalTrigger = bool(flag & toupcam.TOUPCAM_FLAG_TRIGGER_EXTERNAL)
        self._isMonoSensor = bool(flag & toupcam.TOUPCAM_FLAG_MONO)
        highbitFlags = (toupcam.TOUPCAM_FLAG_RAW10 | toupcam.TOUPCAM_FLAG_RAW12
                        | toupcam.TOUPCAM_FLAG_RAW14 | toupcam.TOUPCAM_FLAG_RAW16)
        self._hasHighBitDepth = bool(flag & highbitFlags)

        # select the largest available resolution (full sensor)
        nRes = dev.model.preview
        best, bestArea = 0, 0
        for i in range(nRes):
            r = dev.model.res[i]
            if r.width * r.height > bestArea:
                best, bestArea = i, r.width * r.height
        try:
            self.hcam.put_eSize(best)
        except toupcam.HRESULTException as ex:
            self.__logger.warning(f"put_eSize({best}) failed hr=0x{ex.hr & 0xffffffff:x}")

        # ------------------------------------------------------------------
        # Configure image format. Order matters: auto-exposure must be off
        # before streaming starts, otherwise the SDK's DDR/queue defaults
        # serialize host pulls with sensor readout and throttle the fps.
        # ------------------------------------------------------------------
        try:
            self.hcam.put_AutoExpoEnable(0)
        except toupcam.HRESULTException:
            self.__logger.debug("put_AutoExpoEnable(0) not supported")

        if self.isRGB:
            # RGB mode: SDK delivers processed RGB24 rows
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_RAW, 0)
            try:
                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_RGB, 0)  # RGB24
            except toupcam.HRESULTException:
                pass
            self._bits = 24
            self._bytesPerPixel = 3
            self._npDtype = np.uint8
            self.max_adu = 255
        else:
            # RAW mode: unprocessed sensor data, highest available bit depth
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_RAW, 1)
            useHighBit = False
            if self._hasHighBitDepth:
                try:
                    self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BITDEPTH, 1)
                    useHighBit = True
                except toupcam.HRESULTException:
                    self.__logger.warning("16-bit mode rejected, falling back to 8 bit")
            if not useHighBit:
                try:
                    self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BITDEPTH, 0)
                except toupcam.HRESULTException:
                    pass
            self._bits = 16 if useHighBit else 8
            self._bytesPerPixel = 2 if useHighBit else 1
            self._npDtype = np.uint16 if useHighBit else np.uint8
            try:
                self.max_adu = (1 << self.hcam.MaxBitDepth()) - 1 if useHighBit else 255
            except toupcam.HRESULTException:
                self.max_adu = 65535 if useHighBit else 255

        width, height = self.hcam.get_Size()
        self.SensorWidth = width
        self.SensorHeight = height
        self.shape = (height, width)
        self.frame = np.zeros((height, width), dtype=self._npDtype)

        # full-sensor pull buffer; frames after ROI/binning are smaller and are
        # cut to info.width/height on every pull, so no reallocation is needed
        maxBytes = width * height * max(self._bytesPerPixel, 3)
        self._raw_buf = bytes(maxBytes)
        self._frame_info = toupcam.ToupcamFrameInfoV3()

        self.__logger.info(
            f"Opened {self.deviceName}: {width}x{height}, "
            f"{'RGB24' if self.isRGB else f'RAW{self._bits}'}, "
            f"TEC={self._hasTEC}, fan={self._hasFan}, blacklevel={self._hasBlacklevel}, "
            f"swTrigger={self._hasSoftwareTrigger}, extTrigger={self._hasExternalTrigger}"
        )
        self.is_connected = True

    def reconnectCamera(self):
        if self.hcam is not None:
            try:
                self.hcam.Close()
            except Exception as e:
                self.__logger.error(f"Error while closing camera handle: {e}")
            self.hcam = None
        self.is_streaming = False
        try:
            self._open_camera(self.cameraNo)
            self._reapply_settings()
            self.__logger.debug("Camera reconnected successfully.")
        except Exception as e:
            self.__logger.error(f"Failed to reconnect camera: {e}")

    def _reapply_settings(self):
        """Re-apply cached user settings after a reopen (USB replug etc.)."""
        try:
            if self.exposure_time and self.exposure_time > 0:
                self.set_exposure_time(self.exposure_time)
            if self.gain and self.gain > 0:
                self.set_gain(self.gain)
            if self.blacklevel and self.blacklevel > 0:
                self.set_blacklevel(self.blacklevel)
            if self.binning and self.binning > 1:
                self.setBinning(self.binning)
            self.set_frame_rate(self.frame_rate)
            self.setTriggerSource(self.trigger_source)
        except Exception as e:
            self.__logger.warning(f"Re-applying settings after reconnect failed: {e}")

    def _handleDisconnect(self):
        """Called from the SDK event callback when the camera drops off the bus.
        Spawns a background thread that keeps trying to reopen the device."""
        self.is_connected = False
        wasStreaming = self.is_streaming
        self.is_streaming = False
        with self._reconnect_lock:
            if self._reconnecting:
                return
            self._reconnecting = True
            self._resume_stream_after_reconnect = wasStreaming

        def _reconnectLoop():
            try:
                for attempt in range(30):
                    time.sleep(2.0)
                    self.__logger.info(f"Toupcam reconnect attempt {attempt + 1}")
                    try:
                        self.reconnectCamera()
                        if self.is_connected:
                            if self._resume_stream_after_reconnect:
                                self.start_live()
                            return
                    except Exception as e:
                        self.__logger.debug(f"Reconnect attempt failed: {e}")
                self.__logger.error("Giving up reconnecting to the Toupcam camera")
            finally:
                with self._reconnect_lock:
                    self._reconnecting = False

        threading.Thread(target=_reconnectLoop, daemon=True,
                         name="ToupcamReconnect").start()

    # ---------------------------------------------------------------------
    # SDK event callback (runs on the SDK's internal thread)
    # ---------------------------------------------------------------------
    @staticmethod
    def _eventCallback(nEvent, ctx):
        # The vast majority of callbacks come from the SDK's internal thread;
        # keep this dispatcher tiny and exception-free.
        try:
            if nEvent == toupcam.TOUPCAM_EVENT_IMAGE:
                ctx._onImageEvent(still=False)
            elif nEvent == toupcam.TOUPCAM_EVENT_STILLIMAGE:
                ctx._onImageEvent(still=True)
            elif nEvent == toupcam.TOUPCAM_EVENT_DISCONNECTED:
                ctx._CameraToupcam__logger.error("Camera disconnected!")
                ctx._handleDisconnect()
            elif nEvent == toupcam.TOUPCAM_EVENT_TRIGGERFAIL:
                ctx._CameraToupcam__logger.warning("Trigger failed")
            elif nEvent in (toupcam.TOUPCAM_EVENT_ERROR,
                            toupcam.TOUPCAM_EVENT_NOFRAMETIMEOUT):
                ctx._CameraToupcam__logger.error(f"Camera error event 0x{nEvent:x}")
        except Exception:
            pass

    def _onImageEvent(self, still=False):
        t_entry = time.time()
        with self._pull_lock:
            if self.hcam is None:
                return
            try:
                self.hcam.PullImageV3(self._raw_buf, 1 if still else 0,
                                      self._bits if self.isRGB else 0,
                                      -1,  # rowPitch -1 = tightly packed rows
                                      self._frame_info)
            except toupcam.HRESULTException as ex:
                self.__logger.error(f"PullImageV3 failed hr=0x{ex.hr & 0xffffffff:x}")
                return

            w = self._frame_info.width
            h = self._frame_info.height
            if w == 0 or h == 0:
                return

            if self.isRGB:
                view = np.frombuffer(self._raw_buf, dtype=np.uint8,
                                     count=w * h * 3).reshape(h, w, 3)
            else:
                view = np.frombuffer(self._raw_buf, dtype=self._npDtype,
                                     count=w * h).reshape(h, w)

            # the pull buffer is reused for every frame — store an owned copy
            frame = np.array(view, copy=True)

        if self.flipImage[0]:
            frame = np.flip(frame, axis=0)
        if self.flipImage[1]:
            frame = np.flip(frame, axis=1)

        fid = self._frame_info.seq
        ts = self._frame_info.timestamp  # µs
        self.frame_buffer.append(frame)
        self.frameid_buffer.append(fid)
        self.frameNumber = fid
        self.timestamp = ts
        self.frame = frame
        self.shape = frame.shape[:2]
        self._updateStreamStats(t_entry)

    # ---------------------------------------------------------------------
    # Streaming control
    # ---------------------------------------------------------------------
    def start_live(self):
        if self.is_streaming:
            return
        self.flushBuffer()
        if self.hcam is None:
            self.reconnectCamera()
        if self.hcam is None:
            raise RuntimeError("No Toupcam camera handle available")
        self._streamStats = self._newStreamStats()
        try:
            self.hcam.StartPullModeWithCallback(self._eventCallback, self)
        except toupcam.HRESULTException as ex:
            self.__logger.warning(
                f"StartPullMode failed hr=0x{ex.hr & 0xffffffff:x}, reconnecting once"
            )
            self.reconnectCamera()
            if self.hcam is None:
                raise RuntimeError("StartPullMode failed and reconnect did not recover")
            self.hcam.StartPullModeWithCallback(self._eventCallback, self)
        self.is_streaming = True

    def stop_live(self):
        if not self.is_streaming:
            return
        try:
            self.hcam.Stop()
        except Exception as e:
            self.__logger.warning(f"Stop() failed: {e}")
        self.is_streaming = False

    def suspend_live(self):
        self.stop_live()

    def prepare_live(self):
        pass

    def close(self):
        if self.is_streaming:
            self.stop_live()
        if self.hcam is not None:
            try:
                if self._hasFan:
                    self.hcam.put_Option(toupcam.TOUPCAM_OPTION_FAN, 0)
            except Exception:
                pass
            try:
                self.hcam.Close()
            except Exception as e:
                self.__logger.warning(f"Close() failed: {e}")
            self.hcam = None
        self.is_connected = False

    # ---------------------------------------------------------------------
    # Frame access (ring buffer)
    # ---------------------------------------------------------------------
    def getLast(self, returnFrameNumber: bool = False, timeout: float = 1.0,
                auto_trigger: bool = False):
        """Return the newest frame in the ring buffer (see CameraHIK.getLast)."""
        if auto_trigger and str(self.trigger_source).lower() in (
                "internal trigger", "software", "software trigger"):
            self.send_trigger()

        t0 = time.time()
        while not self.frame_buffer:
            if time.time() - t0 > timeout:
                return (None, None) if returnFrameNumber else None
            if self.lastFrameFromBuffer is not None:  # e.g. while in trigger mode
                if returnFrameNumber:
                    return self.lastFrameFromBuffer, self.lastFrameId
                return self.lastFrameFromBuffer
            time.sleep(0.005)

        latest_frame = self.frame_buffer[-1]
        latest_frame_id = self.frameid_buffer[-1]
        self.lastFrameFromBuffer = latest_frame
        self.lastFrameId = latest_frame_id
        if returnFrameNumber:
            return latest_frame, latest_frame_id
        return latest_frame

    def flushBuffer(self):
        # Clear the SDK-side queue too, so the next grab cannot return a frame
        # exposed before the flush (same rationale as MV_CC_ClearImageBuffer
        # in the HIK driver).
        try:
            if self.hcam is not None:
                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_FLUSH, 3)
        except Exception as e:
            self.__logger.debug(f"SDK flush failed: {e}")
        self.frameid_buffer.clear()
        self.frame_buffer.clear()
        self.lastFrameFromBuffer = None
        self.lastFrameId = -1

    def getLastChunk(self):
        """Return *and clear* the entire ring buffer as a numpy stack."""
        frames = list(self.frame_buffer)
        ids = list(self.frameid_buffer)
        self.flushBuffer()
        self.lastFrameFromBuffer = frames[-1] if frames else None
        return np.array(frames), np.array(ids)

    def getFrameNumber(self):
        return self.frameNumber

    # ---------------------------------------------------------------------
    # ROI / binning
    # ---------------------------------------------------------------------
    def setROI(self, hpos=None, vpos=None, hsize=None, vsize=None):
        """Hardware ROI via put_Roi. Offsets and sizes must be even; the SDK
        interprets (0, 0, 0, 0) as full frame."""
        if self.hcam is None:
            return hpos, vpos, hsize, vsize

        hpos = int(hpos) & ~1 if hpos is not None else 0
        vpos = int(vpos) & ~1 if vpos is not None else 0
        hsize = int(hsize) & ~1 if hsize is not None else 0
        vsize = int(vsize) & ~1 if vsize is not None else 0

        wasStreaming = self.is_streaming
        if wasStreaming:
            self.suspend_live()
        try:
            self.hcam.put_Roi(hpos, vpos, hsize, vsize)
            x, y, w, h = self.hcam.get_Roi()
            self.shape = (h, w)
            self.__logger.debug(f"ROI set to {w}x{h} at {x},{y}")
            return x, y, w, h
        except toupcam.HRESULTException as ex:
            self.__logger.error(
                f"put_Roi({hpos},{vpos},{hsize},{vsize}) failed "
                f"hr=0x{ex.hr & 0xffffffff:x}"
            )
            return hpos, vpos, hsize, vsize
        finally:
            if wasStreaming:
                self.start_live()

    def setBinning(self, binning=1):
        """Digital binning; averaged (0x80 | n) to keep the bit depth."""
        if self.hcam is None:
            return
        value = 1 if binning <= 1 else (0x80 | int(binning))
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BINNING, value)
            self.binning = binning
            self.__logger.debug(f"Binning set to {binning}x{binning}")
        except toupcam.HRESULTException as ex:
            self.__logger.warning(
                f"Binning {binning} not accepted hr=0x{ex.hr & 0xffffffff:x}"
            )

    # ---------------------------------------------------------------------
    # Exposure / gain / blacklevel / frame rate / format
    # ---------------------------------------------------------------------
    def set_exposure_time(self, exposure_time):
        """exposure_time in ms (UI convention)."""
        self.exposure_time = exposure_time
        try:
            self.hcam.put_ExpoTime(int(exposure_time * 1000))  # SDK wants µs
        except toupcam.HRESULTException as ex:
            self.__logger.error(
                f"Set exposure {exposure_time} ms failed hr=0x{ex.hr & 0xffffffff:x}"
            )

    def get_exposuretime(self):
        """Return (current, min, max) in µs, mirroring CameraHIK."""
        try:
            cur = self.hcam.get_ExpoTime()
            emin, emax, _ = self.hcam.get_ExpTimeRange()
            return (cur, emin, emax)
        except Exception as e:
            self.__logger.error(f"Get exposure time failed: {e}")
            return (None, None, None)

    def set_exposure_mode(self, exposure_mode="manual"):
        exposure_mode = str(exposure_mode).lower()
        try:
            if exposure_mode == "manual":
                self.hcam.put_AutoExpoEnable(0)
            elif exposure_mode in ("auto", "once"):
                # the SDK has no one-shot mode; "once" behaves like auto here
                self.hcam.put_AutoExpoEnable(1)
            else:
                self.__logger.warning("Exposure mode not recognized")
        except toupcam.HRESULTException as ex:
            self.__logger.error(f"Set exposure mode failed hr=0x{ex.hr & 0xffffffff:x}")

    def set_camera_mode(self, isAutomatic):
        self.set_exposure_mode(isAutomatic)

    def set_gain(self, gain):
        """Analog gain in Toupcam native percent (100 = 1x)."""
        try:
            gmin, gmax, _ = self.hcam.get_ExpoAGainRange()
            value = int(max(gmin, min(gmax, gain if gain and gain > 0 else gmin)))
            self.hcam.put_ExpoAGain(value)
            self.gain = value
        except Exception as e:
            self.__logger.error(f"Set gain {gain} failed: {e}")

    def get_gain(self):
        """Return (current, min, max) in native percent."""
        try:
            cur = self.hcam.get_ExpoAGain()
            gmin, gmax, _ = self.hcam.get_ExpoAGainRange()
            return (cur, gmin, gmax)
        except Exception as e:
            self.__logger.error(f"Get gain failed: {e}")
            return (None, None, None)

    def set_blacklevel(self, blacklevel):
        if not self._hasBlacklevel:
            self.__logger.debug("Camera does not support black level")
            return
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BLACKLEVEL, int(blacklevel))
            self.blacklevel = blacklevel
        except toupcam.HRESULTException as ex:
            self.__logger.error(f"Set blacklevel failed hr=0x{ex.hr & 0xffffffff:x}")

    def set_frame_rate(self, frame_rate):
        """Limit fps via TOUPCAM_OPTION_FRAMERATE; <= 0 means unlimited."""
        self.frame_rate = frame_rate
        try:
            value = 0 if frame_rate is None or frame_rate <= 0 else int(frame_rate)
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_FRAMERATE, value)
        except toupcam.HRESULTException as ex:
            self.__logger.debug(f"Set frame rate failed hr=0x{ex.hr & 0xffffffff:x}")

    def set_pixel_format(self, format):
        """'mono8' or 'mono16' (RAW container bit depth); no-op in RGB mode."""
        if self.isRGB or format is None:
            return
        fmt = str(format).lower()
        wasStreaming = self.is_streaming
        if wasStreaming:
            self.suspend_live()
        try:
            if fmt in ("mono8", "8", "raw8"):
                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BITDEPTH, 0)
                self._bits, self._bytesPerPixel, self._npDtype = 8, 1, np.uint8
                self.max_adu = 255
            elif fmt in ("mono10", "mono12", "mono14", "mono16", "16", "raw16"):
                self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BITDEPTH, 1)
                self._bits, self._bytesPerPixel, self._npDtype = 16, 2, np.uint16
                try:
                    self.max_adu = (1 << self.hcam.MaxBitDepth()) - 1
                except toupcam.HRESULTException:
                    self.max_adu = 65535
            else:
                self.__logger.warning(f"Unknown pixel format {format}")
        except toupcam.HRESULTException as ex:
            self.__logger.error(f"Set pixel format failed hr=0x{ex.hr & 0xffffffff:x}")
        finally:
            if wasStreaming:
                self.start_live()

    # ---------------------------------------------------------------------
    # Temperature / fan (TEC models)
    # ---------------------------------------------------------------------
    def get_temperature(self):
        """Sensor temperature in °C, or None if unsupported."""
        if not self._hasGetTemperature:
            return None
        try:
            return self.hcam.get_Temperature() / 10.0
        except Exception:
            return None

    def set_temperature(self, temperature_c):
        """TEC target temperature in °C (TEC models only)."""
        if not self._hasTEC:
            self.__logger.debug("Camera has no controllable TEC")
            return
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_TEC, 1)
            self.hcam.put_Temperature(int(temperature_c * 10))
        except toupcam.HRESULTException as ex:
            self.__logger.error(f"Set temperature failed hr=0x{ex.hr & 0xffffffff:x}")

    def set_fan_speed(self, speed):
        if not self._hasFan:
            return
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_FAN, int(speed))
        except toupcam.HRESULTException as ex:
            self.__logger.error(f"Set fan speed failed hr=0x{ex.hr & 0xffffffff:x}")

    # ---------------------------------------------------------------------
    # Trigger handling
    # ---------------------------------------------------------------------
    def getTriggerTypes(self) -> List[str]:
        if not self.is_connected:
            return ["Camera not connected"]
        types = ["Continuous (Free Run)"]
        if self._hasSoftwareTrigger:
            types.append("Software Trigger")
        if self._hasExternalTrigger:
            types.append("External Trigger")
        return types

    def getTriggerSource(self) -> str:
        if not self.is_connected:
            return "Camera not connected"
        tlow = str(self.trigger_source).lower()
        if "soft" in tlow or "internal" in tlow:
            return "Software Trigger"
        if "ext" in tlow or "hard" in tlow:
            return "External Trigger"
        return "Continuous (Free Run)"

    def setTriggerSource(self, trigger_source):
        """
        Continous          → free-run video mode          (TRIGGER option 0)
        Internal trigger   → software trigger             (TRIGGER option 1)
        External trigger   → hardware trigger input       (TRIGGER option 2)
        """
        if self.hcam is None:
            return False
        was_streaming = self.is_streaming
        if was_streaming:
            self.suspend_live()

        tlow = str(trigger_source).lower()
        try:
            if "cont" in tlow:
                mode = 0
            elif "soft" in tlow or tlow in ("internal trigger", "software trigger"):
                mode = 1
            elif "ext" in tlow or tlow in ("external trigger", "hardware", "line0"):
                mode = 2
            else:
                self.__logger.warning(f"Unknown trigger source: {trigger_source}")
                return False
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_TRIGGER, mode)
            self.trigger_source = trigger_source
            self.__logger.info(f"Trigger source set to {trigger_source} (mode {mode})")
            return True
        except toupcam.HRESULTException as ex:
            self.__logger.error(f"Set trigger failed hr=0x{ex.hr & 0xffffffff:x}")
            return False
        finally:
            if was_streaming:
                self.start_live()

    def send_trigger(self):
        """Fire one software trigger pulse (requires software-trigger mode)."""
        try:
            self.hcam.Trigger(1)
            return True
        except Exception as e:
            self.__logger.error(f"Software trigger failed: {e}")
            return False

    def snapSoftwareTrigger(self, timeout: float = 2.0):
        """Fire one software trigger and return the frame it produces.

        Requires software-trigger mode (``setTriggerSource('software')``); the
        returned image is guaranteed to be exposed AFTER this call — the basis
        for deterministic post-move grabs (see CameraHIK.snapSoftwareTrigger).
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

    # ---------------------------------------------------------------------
    # Property interface (used by the detector manager)
    # ---------------------------------------------------------------------
    def setPropertyValue(self, property_name, property_value):
        if property_name == "gain":
            self.set_gain(property_value)
        elif property_name in ("exposure", "exposureTime", "exposure_time"):
            self.set_exposure_time(property_value)
        elif property_name == "exposure_mode":
            self.set_exposure_mode(property_value)
        elif property_name == "blacklevel":
            self.set_blacklevel(property_value)
        elif property_name == "frame_rate":
            self.set_frame_rate(property_value)
        elif property_name == "trigger_source":
            self.setTriggerSource(property_value)
        elif property_name == "mode":
            self.set_camera_mode(isAutomatic=property_value)
        elif property_name == "pixel_format":
            self.set_pixel_format(property_value)
        elif property_name in ("target_temperature", "temperature"):
            self.set_temperature(property_value)
        elif property_name == "fan_speed":
            self.set_fan_speed(property_value)
        elif property_name == "binning":
            self.setBinning(property_value)
        else:
            self.__logger.warning(f"Property {property_name} does not exist")
            return False
        return property_value

    def getPropertyValue(self, property_name):
        if property_name == "gain":
            result = self.get_gain()
            return result[0] if result and result[0] is not None else None
        elif property_name == "exposure":
            result = self.get_exposuretime()
            # SDK returns µs, UI/manager expects ms
            return result[0] / 1000.0 if result and result[0] is not None else None
        elif property_name == "exposure_mode":
            try:
                return "auto" if self.hcam.get_AutoExpoEnable() else "manual"
            except Exception:
                return "manual"
        elif property_name == "blacklevel":
            try:
                return self.hcam.get_Option(toupcam.TOUPCAM_OPTION_BLACKLEVEL)
            except Exception:
                return self.blacklevel
        elif property_name in ("image_width", "Width"):
            return self.shape[1] if len(self.shape) > 1 else self.SensorWidth
        elif property_name in ("image_height", "Height"):
            return self.shape[0] if len(self.shape) > 0 else self.SensorHeight
        elif property_name == "frame_number":
            return self.frameNumber
        elif property_name == "frame_rate":
            return self.frame_rate
        elif property_name == "trigger_source":
            return self.trigger_source
        elif property_name == "temperature":
            return self.get_temperature()
        elif property_name == "binning":
            return self.binning
        else:
            self.__logger.warning(f"Property {property_name} does not exist")
            return None

    # ---------------------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------------------
    def get_camera_parameters(self):
        params = {}
        try:
            params["model_name"] = self.deviceName
            params["serial_number"] = self.hcam.SerialNumber()
            params["firmware_version"] = self.hcam.FwVersion()
            params["hardware_version"] = self.hcam.HwVersion()
        except Exception:
            pass
        try:
            params["sensor_width"] = self.SensorWidth
            params["sensor_height"] = self.SensorHeight
            params["bit_depth"] = self._bits
            params["isRGB"] = self.isRGB
            cur, emin, emax = self.get_exposuretime()
            params["exposure_us"] = cur
            params["exposure_range_us"] = (emin, emax)
            gcur, gmin, gmax = self.get_gain()
            params["gain_percent"] = gcur
            params["gain_range_percent"] = (gmin, gmax)
            temp = self.get_temperature()
            if temp is not None:
                params["temperature_c"] = temp
        except Exception:
            pass
        return params

    def _newStreamStats(self):
        return {
            "t_start": time.time(),
            "n_frames": 0,
            "t_last": 0.0,
            "dt_avg_ms": 0.0,
        }

    def _updateStreamStats(self, t_entry):
        s = self._streamStats
        if s["t_last"] > 0:
            dt_ms = (t_entry - s["t_last"]) * 1000.0
            alpha = 0.05
            s["dt_avg_ms"] = (1 - alpha) * s["dt_avg_ms"] + alpha * dt_ms
        s["t_last"] = t_entry
        s["n_frames"] += 1

    def getStreamDiagnostics(self) -> dict:
        s = self._streamStats
        elapsed = max(time.time() - s["t_start"], 1e-6)
        return {
            "fps_callback": s["n_frames"] / elapsed,
            "frame_interval_ms_avg": s["dt_avg_ms"],
            "frames_received": s["n_frames"],
            "buffer_fill": len(self.frame_buffer),
            "frame_number": self.frameNumber,
        }

    def getDiagnostics(self) -> dict:
        return {
            "model": self.deviceName if hasattr(self, "deviceName") else self.model,
            "is_connected": self.is_connected,
            "is_streaming": self.is_streaming,
            "trigger_source": self.trigger_source,
            "shape": tuple(self.shape),
            "bit_depth": getattr(self, "_bits", None),
            "stream": self.getStreamDiagnostics(),
        }

    def openPropertiesGUI(self):
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()


# Copyright (C) ImSwitch developers 2021
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
