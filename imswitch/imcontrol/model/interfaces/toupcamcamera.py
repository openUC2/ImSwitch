import collections
import csv
import datetime
import os
import threading
import time
from typing import List, Optional

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


# Exposure above which the live stream is no longer useful: the SDK delivers a
# frame only once per integration, so the preview would freeze for seconds at a
# time. The frontend stops (and does not auto-start) the stream above this and
# switches to snap-on-demand instead. Kept here so backend and UI agree on one
# number; see LiveViewController.LONG_EXPOSURE_THRESHOLD_MS.
LONG_EXPOSURE_THRESHOLD_MS = 2000.0

# The ImSwitch UI exposes gain as a small unitless 0..23 scale (inherited from
# the GenICam cameras, where it is dB). Toupcam instead uses *analog gain in
# percent*, 100 = 1x, with a per-model maximum read from get_ExpoAGainRange().
# We map the UI scale linearly onto that native range, so UI 0 is always the
# camera's minimum gain and UI 23 always its maximum.
UI_GAIN_MIN = 0
UI_GAIN_MAX = 23

# Sensor temperature monitor. The polling thread runs for as long as the
# camera object exists -- not only while acquiring -- because the cooler runs
# the whole time too. It does two jobs:
#   * safety: above TEMPERATURE_SHUTDOWN_C the cooler and the window heater are
#     switched off. A TEC dumps its own dissipation into the camera body, so a
#     stalled fan or a failing cooler keeps driving the sensor hotter, and
#     nothing else in the stack would notice.
#   * logging (readSaveTemperature): one CSV row per interval while the camera
#     is armed, appended to this file in the per-day recordings folder (the
#     same folder the snaps go to).
TEMPERATURE_LOG_INTERVAL_S = 5.0
TEMPERATURE_LOG_FILENAME = "toupcam_temperature_log.csv"
TEMPERATURE_LOG_COLUMNS = (
    "timestamp", "unix_time_s", "camera", "sensor_temperature_c",
    "tec_target_c", "tec_on", "fan_speed", "heat", "exposure_ms", "streaming",
    "over_temperature",
)
# Sensor temperature at which the cooler is cut out, the temperature it has to
# fall back to before it is re-armed, and how many consecutive samples either
# decision needs -- one bogus reading must neither trip nor clear the cutoff.
TEMPERATURE_SHUTDOWN_C = 30.0
TEMPERATURE_RECOVERY_C = TEMPERATURE_SHUTDOWN_C - 5.0
TEMPERATURE_SHUTDOWN_SAMPLES = 2
# How often the monitor re-arms the cooler on its own. A camera that keeps
# running hot has a hardware fault that power-cycling the TEC cannot fix, so
# after this many recoveries the cooler stays off until a target is set again.
TEMPERATURE_RECOVERY_ATTEMPTS = 3
# The TEC target is re-written to the camera at this interval. One write is not
# always enough: the camera can report the target back correctly, with the
# cooler switched on, and still leave the sensor sitting at ambient for hours.
# Re-asserting it costs one register write a minute and gets the regulation
# going again without anyone having to notice and fix it by hand.
TEMPERATURE_REASSERT_INTERVAL_S = 60.0
# A cooler that is regulating gets within a few degrees of its target in a
# minute or two. Staying this far above it for this long means it is not
# working: a target outside the model's range, a dead TEC, or no heat path.
TEMPERATURE_STALL_MARGIN_C = 5.0
TEMPERATURE_STALL_AFTER_S = 300.0


class CameraToupcam:
    """ToupTek (Toupcam) camera wrapper that grabs frames via the SDK's
    pull-mode callback (no polling), mirroring the CameraHIK interface so the
    detector manager and the triggered-grab protocol work identically.

    Frames are pulled with PullImageV3 into a preallocated full-sensor buffer
    and stored in the ring buffer as owned copies (never views), so entries
    can not alias SDK memory that is overwritten by later frames.

    Units at this boundary (the SDK uses different ones internally):

    - exposure: **milliseconds** in/out (ImSwitch convention), converted to the
      SDK's microseconds in :meth:`set_exposure_time`.
    - gain: **UI scale 0..23** in/out, mapped linearly onto the camera's native
      analog-gain percent range in :meth:`set_gain`.
    """

    def __init__(self, cameraNo=None, exposure_time=100, gain=0, frame_rate=-1,
                 blacklevel=200, isRGB=False, binning=1, flipImage=(False, False),
                 heat=False, lowNoise=True, conversionGain="HCG",
                 blacklevelAutoAdjust=None, readSaveTemperature=True):
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
        # Low-noise / long-exposure configuration. Each is applied only if the
        # camera advertises the corresponding capability flag; pass None to
        # leave the camera's own default alone.
        #   heat            -- window heater against condensation on a cooled
        #                      sensor (True = max level). Off by default: it
        #                      heats the camera from the inside, which works
        #                      against the cooler and the sensor temperature
        #                      that long exposures actually care about
        #   lowNoise        -- slower readout, higher SNR
        #   conversionGain  -- "LCG" / "HCG" / "HDR"; HCG gives the lower read
        #                      noise (at reduced full-well), which is what you
        #                      want for faint, long-exposure signal
        # blacklevelAutoAdjust re-derives the offset from the optical-black
        # pixels; it is switched off automatically whenever a manual black
        # level is set (see set_blacklevel).
        self.heat = heat
        self.lowNoise = lowNoise
        self.conversionGain = conversionGain
        self.blacklevelAutoAdjust = blacklevelAutoAdjust
        self.exposure_time = exposure_time  # ms (UI convention, like CameraHIK)
        # Cooling request, cached so a reopened handle (USB replug, failed
        # StartPullMode) gets the same TEC target back instead of the camera's
        # power-on default -- which is how a -30 °C sensor quietly warms up.
        self.targetTemperature = -30.0  # °C
        self.fanSpeed = 1  # -1 = the camera's default speed
        # readSaveTemperature: poll the sensor temperature every
        # TEMPERATURE_LOG_INTERVAL_S while the camera is armed and append it to
        # a CSV in the day's recordings folder -- the record that tells whether
        # a long exposure was actually taken at the requested TEC target.
        self.readSaveTemperature = bool(readSaveTemperature)
        self._tempMonitorThread: Optional[threading.Thread] = None
        self._tempMonitorStop = threading.Event()
        self._tempLogErrorLogged = False
        self._tempMonitorErrorLogged = False
        # Over-temperature cutoff state (see _checkTemperatureSafety).
        self._tecOverTemp = False
        self._tempOverSamples = 0
        self._tempRecoverySamples = 0
        self._tecRecoveryAttempts = 0
        self._tecRecoveryExhaustedLogged = False
        # TEC keep-alive / stall detection (see _maintainTecTarget).
        self._lastTecReassert = 0.0
        self._tecStallSince = None
        self._tecStallLogged = False
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

        # Native hardware ranges, filled in by _open_camera(). Cached because
        # set_gain()/set_exposure_time() are on the hot path of every UI slider
        # drag and the SDK getters round-trip over USB.
        self._gainRangeNative = (100, 100, 100)   # (min, max, default) percent
        self._expRangeUs = (1, 1_000_000, 10_000)  # (min, max, default) µs
        self._maxBitDepth = 8

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
            # set temperature/fan/TEC state if supported, otherwise ignore
            self.set_temperature(self.targetTemperature)
            self.set_fan_speed(self.fanSpeed)
            # Low-noise long-exposure configuration. Applied before the stream
            # starts because conversion gain and low-noise mode change the
            # sensor readout, which the SDK only picks up between streams.
            if conversionGain is not None:
                self.set_conversion_gain(conversionGain)
            if lowNoise is not None:
                self.set_low_noise(lowNoise)
            if heat is not None:
                self.set_heat(heat)
            if blacklevelAutoAdjust is not None:
                self.set_blacklevel_autoadjust(blacklevelAutoAdjust)
            self._logLowNoiseState()

        except Exception as e:
            self.__logger.warning(f"Applying initial camera settings failed: {e}")

        # Started here rather than with the stream: the over-temperature cutoff
        # is a safety function, not a logging option, and the cooler is already
        # running at this point.
        self._startTemperatureMonitor()

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
        # Settable TEC target range in °C. A target outside it is accepted by
        # the SDK and read back unchanged, but the camera may then not regulate
        # at all, so set_temperature() clamps to this.
        self._tecTargetRange = None
        self._hasGetTemperature = bool(flag & toupcam.TOUPCAM_FLAG_GETTEMPERATURE)
        self._hasFan = bool(flag & toupcam.TOUPCAM_FLAG_FAN)
        self._hasBlacklevel = bool(flag & toupcam.TOUPCAM_FLAG_BLACKLEVEL)
        self._hasSoftwareTrigger = bool(flag & toupcam.TOUPCAM_FLAG_TRIGGER_SOFTWARE)
        self._hasExternalTrigger = bool(flag & toupcam.TOUPCAM_FLAG_TRIGGER_EXTERNAL)
        self._isMonoSensor = bool(flag & toupcam.TOUPCAM_FLAG_MONO)
        # Long-exposure / low-noise capabilities.
        self._hasHeat = bool(flag & toupcam.TOUPCAM_FLAG_HEAT)
        self._hasLowNoise = bool(flag & toupcam.TOUPCAM_FLAG_LOW_NOISE)
        self._hasCGHDR = bool(flag & toupcam.TOUPCAM_FLAG_CGHDR)
        # TOUPCAM_FLAG_CG = LCG/HCG, TOUPCAM_FLAG_CGHDR adds an HDR step.
        self._hasCG = bool(flag & toupcam.TOUPCAM_FLAG_CG) or self._hasCGHDR
        highbitFlags = (toupcam.TOUPCAM_FLAG_RAW10 | toupcam.TOUPCAM_FLAG_RAW12
                        | toupcam.TOUPCAM_FLAG_RAW14 | toupcam.TOUPCAM_FLAG_RAW16)
        self._hasHighBitDepth = bool(flag & highbitFlags)

        # Native ranges, read once. Gain is in percent (100 = 1x) and exposure
        # in µs; both are mapped to the ImSwitch UI units by the setters below.
        try:
            self._gainRangeNative = self.hcam.get_ExpoAGainRange()
        except toupcam.HRESULTException as ex:
            self.__logger.warning(
                f"get_ExpoAGainRange failed hr=0x{ex.hr & 0xffffffff:x}, "
                f"assuming {self._gainRangeNative}"
            )
        try:
            self._expRangeUs = self.hcam.get_ExpTimeRange()
        except toupcam.HRESULTException as ex:
            self.__logger.warning(
                f"get_ExpTimeRange failed hr=0x{ex.hr & 0xffffffff:x}, "
                f"assuming {self._expRangeUs}"
            )
        try:
            self._maxBitDepth = int(self.hcam.MaxBitDepth())
        except Exception:
            self._maxBitDepth = 8
        if self._hasTEC:
            try:
                low, high = self.hcam.get_TecTargetRange()
                self._tecTargetRange = (low / 10.0, high / 10.0)
            except Exception:
                self.__logger.debug("TEC target range not reported by this model")

        # Heater levels are model-specific; TOUPCAM_OPTION_HEAT_MAX reports the
        # top of the range, and heat=True means "that level".
        self._heatMax = 0
        if self._hasHeat:
            try:
                self._heatMax = int(self.hcam.get_Option(toupcam.TOUPCAM_OPTION_HEAT_MAX))
            except Exception:
                self.__logger.debug("get HEAT_MAX failed, assuming on/off heater")
                self._heatMax = 1

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
            # RAW mode: unprocessed sensor data, highest available bit depth.
            # TOUPCAM_OPTION_BITDEPTH only picks the *container*: 0 = 8 bit,
            # 1 = "16 bit", which really means "the sensor's native depth in a
            # 16-bit word". MaxBitDepth() reports that native depth (14 on the
            # RAW14 models), so max_adu below is 16383 there, not 65535 — i.e.
            # yes, the full 14 bits are used, no truncation to 8 or 12.
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
                self.max_adu = (1 << self._maxBitDepth) - 1 if useHighBit else 255
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

        if toupcam.MISSING_EXPORTS:
            # native lib older than the wrapper — harmless as long as none of the
            # calls below is one of these (see _LibProxy in toupcam.py)
            self.__logger.debug(
                f"Toupcam SDK {toupcam.Toupcam.Version()} does not export: "
                f"{', '.join(sorted(toupcam.MISSING_EXPORTS))}"
            )

        try:
            pixelFormat = self.hcam.get_Option(toupcam.TOUPCAM_OPTION_PIXEL_FORMAT)
        except Exception:
            pixelFormat = None

        gmin, gmax, gdef = self._gainRangeNative
        emin, emax, _ = self._expRangeUs
        self.__logger.info(
            f"Opened {self.deviceName}: {width}x{height}, "
            f"{'RGB24' if self.isRGB else f'RAW{self._bits}'}, "
            f"TEC={self._hasTEC}, fan={self._hasFan}, blacklevel={self._hasBlacklevel}, "
            f"swTrigger={self._hasSoftwareTrigger}, extTrigger={self._hasExternalTrigger}, "
            f"heat={self._hasHeat}, lowNoise={self._hasLowNoise}, "
            f"conversionGain={self._hasCG}{' (+HDR)' if self._hasCGHDR else ''}"
            + (f", tecTarget {self._tecTargetRange[0]:.1f}.."
               f"{self._tecTargetRange[1]:.1f} °C"
               if self._tecTargetRange else "")
        )
        self.__logger.info(
            f"{self.deviceName} ranges: sensor bit depth={self._maxBitDepth} "
            f"(max ADU {self.max_adu}, pixelFormat={pixelFormat}), "
            f"exposure {emin / 1000.0:.3f}..{emax / 1000.0:.1f} ms, "
            f"gain {gmin}..{gmax}% (default {gdef}%) "
            f"mapped from UI {UI_GAIN_MIN}..{UI_GAIN_MAX}"
        )
        self.is_connected = True

    def reconnectCamera(self):
        self._stopTemperatureMonitor()
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
            self._startTemperatureMonitor()
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
            # Same for the cooler: a fresh handle does not remember the TEC
            # target, and nothing else would notice that the sensor is warming.
            self.set_temperature(self.targetTemperature)
            self.set_fan_speed(self.fanSpeed)
            # A reopened handle comes up with the camera's own defaults, so the
            # low-noise configuration has to be pushed again or a long
            # acquisition silently continues at LCG / normal-noise mode.
            if self.conversionGain is not None:
                self.set_conversion_gain(self.conversionGain)
            if self.lowNoise is not None:
                self.set_low_noise(self.lowNoise)
            if self.heat is not None:
                self.set_heat(self.heat)
            if self.blacklevelAutoAdjust is not None:
                self.set_blacklevel_autoadjust(self.blacklevelAutoAdjust)
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
        # The SDK numbers frames per stream, so a snap after a snap sees seq 1
        # again. Reset our counter too, so a caller that reads getFrameNumber()
        # before arming and waits for it to advance recognises the very first
        # frame of the new stream as fresh instead of sitting out a second
        # exposure (RecordingController._waitForFreshFrame).
        self.frameNumber = -1
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
        self._startTemperatureMonitor()
        if self.readSaveTemperature:
            # One row right at the start, so even a short acquisition leaves a
            # temperature on record instead of waiting out the first interval.
            self._writeTemperatureSample(self.get_temperature())

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
        self._stopTemperatureMonitor()
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
    def getLast(self, returnFrameNumber: bool = False, timeout: float = None,
                auto_trigger: bool = False):
        """Return the newest frame in the ring buffer (see CameraHIK.getLast).

        ``timeout`` defaults to one full integration plus a margin rather than a
        flat 1 s, so a multi-second exposure is actually waited out instead of
        reporting "no frame" while the sensor is still integrating.
        """
        if auto_trigger and str(self.trigger_source).lower() in (
                "internal trigger", "software", "software trigger"):
            self.send_trigger()

        if timeout is None:
            timeout = self._frameTimeout()

        t0 = time.time()
        while not self.frame_buffer:
            if time.time() - t0 > timeout:
                return (None, None) if returnFrameNumber else None
            if not self.is_streaming:
                # Acquisition was stopped while we were waiting (e.g. a snap was
                # cancelled): no further frame can arrive, so return now instead
                # of sitting out the rest of a multi-second exposure timeout.
                # This is what makes an in-progress long exposure abortable.
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
            return
        # Binning changes the delivered frame size. While streaming the next
        # pull updates self.shape anyway, but callers that set the binning on an
        # idle camera need the new size right away.
        try:
            width, height = self.hcam.get_Size()
            self.shape = (height, width)
        except toupcam.HRESULTException as ex:
            self.__logger.debug(f"get_Size after binning failed hr=0x{ex.hr & 0xffffffff:x}")

    # ---------------------------------------------------------------------
    # Exposure / gain / blacklevel / frame rate / format
    # ---------------------------------------------------------------------
    def set_exposure_time(self, exposure_time):
        """exposure_time in ms (ImSwitch UI convention).

        The SDK takes microseconds (``Toupcam_put_ExpoTime``), so the value is
        multiplied by 1000 here, then clamped to the model's ``get_ExpTimeRange``
        — passing an out-of-range value makes the SDK reject the call outright
        and silently leave the previous exposure in place.
        """
        emin, emax, _ = self._expRangeUs
        requestedUs = int(round(float(exposure_time) * 1000))
        appliedUs = int(max(emin, min(emax, requestedUs)))
        if appliedUs != requestedUs:
            self.__logger.warning(
                f"Exposure {exposure_time} ms is outside the camera range "
                f"{emin / 1000.0:.3f}..{emax / 1000.0:.1f} ms, clamped to "
                f"{appliedUs / 1000.0:.3f} ms"
            )
        try:
            self.hcam.put_ExpoTime(appliedUs)
            self.exposure_time = appliedUs / 1000.0
        except toupcam.HRESULTException as ex:
            self.__logger.error(
                f"Set exposure {exposure_time} ms failed hr=0x{ex.hr & 0xffffffff:x}"
            )

    def get_exposuretime(self):
        """Return (current, min, max) in µs, mirroring CameraHIK."""
        try:
            cur = self.hcam.get_ExpoTime()
            emin, emax, _ = self.hcam.get_ExpTimeRange()
            self._expRangeUs = (emin, emax, self._expRangeUs[2])
            return (cur, emin, emax)
        except Exception as e:
            self.__logger.error(f"Get exposure time failed: {e}")
            return (None, None, None)

    def getExposureSeconds(self) -> float:
        """Current exposure in seconds, from the cached UI value."""
        try:
            return max(float(self.exposure_time), 0.0) / 1000.0
        except (TypeError, ValueError):
            return 0.0

    def isLongExposure(self, thresholdMs: float = LONG_EXPOSURE_THRESHOLD_MS) -> bool:
        """True when a live stream would be pointless at the current exposure."""
        return self.getExposureSeconds() * 1000.0 > thresholdMs

    def _frameTimeout(self, base: float = 1.0) -> float:
        """Grab timeout that always leaves room for one full integration.

        A fixed 1 s (the old default) expires long before a multi-second
        exposure completes, so every long-exposure grab returned None.
        """
        return max(base, 2.0 * self.getExposureSeconds() + base)

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

    def _uiGainToNative(self, uiGain) -> int:
        """Map the UI's 0..23 scale linearly onto the camera's percent range.

        Toupcam analog gain is a percentage with 100 = 1x and a per-model
        maximum (often 2000–5000%). Feeding the raw UI number to
        ``put_ExpoAGain`` would land below ``gmin`` for every value the UI can
        produce, so the camera stayed pinned at minimum gain no matter what the
        user selected.
        """
        gmin, gmax, _ = self._gainRangeNative
        try:
            ui = float(uiGain)
        except (TypeError, ValueError):
            ui = UI_GAIN_MIN
        ui = max(UI_GAIN_MIN, min(UI_GAIN_MAX, ui))
        span = UI_GAIN_MAX - UI_GAIN_MIN
        frac = (ui - UI_GAIN_MIN) / span if span else 0.0
        return int(round(gmin + frac * (gmax - gmin)))

    def _nativeGainToUi(self, nativeGain) -> float:
        """Inverse of :meth:`_uiGainToNative`, so the UI round-trips."""
        gmin, gmax, _ = self._gainRangeNative
        if gmax <= gmin:
            return float(UI_GAIN_MIN)
        frac = (float(nativeGain) - gmin) / (gmax - gmin)
        frac = max(0.0, min(1.0, frac))
        return round(UI_GAIN_MIN + frac * (UI_GAIN_MAX - UI_GAIN_MIN), 2)

    def set_gain(self, gain):
        """Set analog gain from the UI's 0..23 scale (see _uiGainToNative)."""
        try:
            value = self._uiGainToNative(gain)
            self.hcam.put_ExpoAGain(value)
            self.gain = self._nativeGainToUi(value)
            self.__logger.debug(f"Gain UI {gain} -> {value}% native")
        except Exception as e:
            self.__logger.error(f"Set gain {gain} failed: {e}")

    def get_gain(self):
        """Return (current, min, max) on the UI's 0..23 scale."""
        try:
            self._gainRangeNative = self.hcam.get_ExpoAGainRange()
            cur = self.hcam.get_ExpoAGain()
            return (self._nativeGainToUi(cur), float(UI_GAIN_MIN), float(UI_GAIN_MAX))
        except Exception as e:
            self.__logger.error(f"Get gain failed: {e}")
            return (None, None, None)

    def get_gain_native(self):
        """Return (current, min, max) analog gain in Toupcam percent."""
        try:
            self._gainRangeNative = self.hcam.get_ExpoAGainRange()
            gmin, gmax, _ = self._gainRangeNative
            return (self.hcam.get_ExpoAGain(), gmin, gmax)
        except Exception as e:
            self.__logger.error(f"Get native gain failed: {e}")
            return (None, None, None)

    def blacklevel_max(self) -> int:
        """Largest black level accepted at the current output bit depth.

        The SDK's range is per bit depth (31 at 8 bit, 31*64 at 14 bit,
        31*256 at 16 bit, ...), because the value is expressed in ADU of the
        data actually being delivered. Writing a larger number is not an error
        -- the camera stores it and reads it back unchanged -- but the offset
        in the image is clamped, which looks exactly like "the setting does
        nothing".
        """
        bits = self._maxBitDepth if self._bits > 8 else 8
        return {
            8: toupcam.TOUPCAM_BLACKLEVEL8_MAX,
            10: toupcam.TOUPCAM_BLACKLEVEL10_MAX,
            11: toupcam.TOUPCAM_BLACKLEVEL11_MAX,
            12: toupcam.TOUPCAM_BLACKLEVEL12_MAX,
            14: toupcam.TOUPCAM_BLACKLEVEL14_MAX,
            16: toupcam.TOUPCAM_BLACKLEVEL16_MAX,
        }.get(int(bits), toupcam.TOUPCAM_BLACKLEVEL16_MAX)

    def set_blacklevel_autoadjust(self, enable):
        """Enable/disable the optical-black based automatic black level.

        While this is on the camera keeps re-deriving the offset from its OB
        pixels, so a manually written black level is overwritten behind your
        back -- the read-back still returns what you wrote, but the pixel
        values do not move. The SDK documents it as the knob to turn off
        precisely for long exposures, where OB leakage skews the adjustment.
        """
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST,
                                 1 if enable else 0)
            self.blacklevelAutoAdjust = bool(enable)
            return True
        except toupcam.HRESULTException as ex:
            # Not every model exposes it; that is not an error, it just means
            # there is no auto-adjust to fight with.
            self.__logger.debug(
                f"Black level auto-adjust not settable hr=0x{ex.hr & 0xffffffff:x}")
            return False

    def get_blacklevel_autoadjust(self):
        """Current auto-adjust state, or None if the model has no such option."""
        try:
            return bool(self.hcam.get_Option(
                toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST))
        except Exception:
            return None

    def set_blacklevel(self, blacklevel):
        """Set the black level (offset) in ADU of the current bit depth.

        Clamps to the bit-depth-dependent maximum and turns the OB auto-adjust
        off first -- both are silent failure modes: the value is accepted and
        reads back unchanged while the image offset does not move.
        """
        if not self._hasBlacklevel:
            self.__logger.warning(
                "Camera does not advertise TOUPCAM_FLAG_BLACKLEVEL; "
                "ignoring black level request")
            return

        requested = int(blacklevel)
        maximum = self.blacklevel_max()
        value = max(toupcam.TOUPCAM_BLACKLEVEL_MIN, min(requested, maximum))
        if value != requested:
            self.__logger.warning(
                f"Black level {requested} is outside 0..{maximum} for "
                f"{self._maxBitDepth if self._bits > 8 else 8}-bit output; "
                f"using {value}. (The camera accepts and reads back the larger "
                f"number, but the offset saturates at {maximum}.)"
            )

        # Manual black level and the OB auto-adjust are mutually exclusive in
        # practice -- leave auto-adjust on and it just overwrites this.
        if self.get_blacklevel_autoadjust():
            if self.set_blacklevel_autoadjust(False):
                self.__logger.info(
                    "Disabled black level auto-adjust so the manual black "
                    "level takes effect")

        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_BLACKLEVEL, value)
            self.blacklevel = value
        except toupcam.HRESULTException as ex:
            self.__logger.error(f"Set blacklevel failed hr=0x{ex.hr & 0xffffffff:x}")
            return

        # Read back: a mismatch here means the camera silently substituted a
        # value, which is worth seeing in the log rather than guessing at from
        # the image.
        try:
            actual = int(self.hcam.get_Option(toupcam.TOUPCAM_OPTION_BLACKLEVEL))
            if actual != value:
                self.__logger.warning(
                    f"Black level written as {value} but reads back {actual}")
            self.blacklevel = actual
        except Exception:
            pass

    # ---------------------------------------------------------------------
    # Low-noise / long-exposure configuration
    # ---------------------------------------------------------------------
    def set_conversion_gain(self, mode):
        """Select the sensor conversion gain: "LCG", "HCG" or "HDR"/"MCG".

        HCG is the low-read-noise mode (smaller full well) and is what you
        want for faint long-exposure signal; LCG keeps the full well for bright
        scenes. Accepts the names or the raw SDK integers 0/1/2.
        """
        if not self._hasCG:
            self.__logger.debug("Camera does not support conversion gain modes")
            return
        names = {"lcg": 0, "hcg": 1, "hdr": 2, "mcg": 2}
        if isinstance(mode, str):
            value = names.get(mode.strip().lower())
            if value is None:
                self.__logger.warning(f"Unknown conversion gain '{mode}'")
                return
        else:
            value = int(mode)
        if value == 2 and not self._hasCGHDR:
            self.__logger.warning(
                "Camera has no HDR/MCG conversion gain; keeping HCG")
            value = 1

        # Conversion gain switches the sensor's readout chain; do it while the
        # stream is down so the SDK reconfigures cleanly.
        wasStreaming = self.is_streaming
        if wasStreaming:
            self.suspend_live()
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_CG, value)
            self.conversionGain = {0: "LCG", 1: "HCG", 2: "HDR"}[value]
            self.__logger.info(f"Conversion gain set to {self.conversionGain}")
        except toupcam.HRESULTException as ex:
            self.__logger.error(
                f"Set conversion gain failed hr=0x{ex.hr & 0xffffffff:x}")
        finally:
            if wasStreaming:
                self.start_live()

    def get_conversion_gain(self):
        """Current conversion gain as "LCG"/"HCG"/"HDR", or None."""
        if not self._hasCG:
            return None
        try:
            return {0: "LCG", 1: "HCG", 2: "HDR"}.get(
                int(self.hcam.get_Option(toupcam.TOUPCAM_OPTION_CG)))
        except Exception:
            return None

    def set_low_noise(self, enable):
        """Enable the sensor's low-noise readout (higher SNR, lower fps)."""
        if not self._hasLowNoise:
            self.__logger.debug("Camera does not support low noise mode")
            return
        wasStreaming = self.is_streaming
        if wasStreaming:
            self.suspend_live()
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_LOW_NOISE,
                                 1 if enable else 0)
            self.lowNoise = bool(enable)
            self.__logger.info(f"Low noise mode {'on' if enable else 'off'}")
        except toupcam.HRESULTException as ex:
            self.__logger.error(
                f"Set low noise mode failed hr=0x{ex.hr & 0xffffffff:x}")
        finally:
            if wasStreaming:
                self.start_live()

    def get_low_noise(self):
        """Current low-noise state, or None if unsupported."""
        if not self._hasLowNoise:
            return None
        try:
            return bool(self.hcam.get_Option(toupcam.TOUPCAM_OPTION_LOW_NOISE))
        except Exception:
            return None

    def set_heat(self, level):
        """Window heater level, ``True`` meaning the camera's maximum.

        The heater keeps the sensor window from fogging up on a cooled camera
        -- without it, condensation during a long cooled acquisition shows up
        as drifting blobs in the image. Levels are model-specific and clamped
        to TOUPCAM_OPTION_HEAT_MAX.
        """
        if not self._hasHeat:
            self.__logger.debug("Camera does not support the window heater")
            return
        if isinstance(level, bool):
            value = self._heatMax if level else 0
        else:
            value = max(0, min(int(level), self._heatMax))
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_HEAT, value)
            self.heat = value
            self.__logger.info(f"Window heater set to {value}/{self._heatMax}")
        except toupcam.HRESULTException as ex:
            self.__logger.error(f"Set heat failed hr=0x{ex.hr & 0xffffffff:x}")

    def get_heat(self):
        """Current heater level, or None if unsupported."""
        if not self._hasHeat:
            return None
        try:
            return int(self.hcam.get_Option(toupcam.TOUPCAM_OPTION_HEAT))
        except Exception:
            return None

    def _logLowNoiseState(self):
        """Log what the camera actually ended up with, capability by capability.

        Worth one line at startup: every one of these is silently ignored when
        the model does not support it, so the log is the only way to tell
        "HCG is on" from "this camera has no HCG".
        """
        parts = []
        parts.append(f"conversionGain={self.get_conversion_gain() or 'n/a'}")
        lowNoise = self.get_low_noise()
        parts.append(f"lowNoise={'n/a' if lowNoise is None else lowNoise}")
        heat = self.get_heat()
        parts.append(
            "heat=n/a" if heat is None else f"heat={heat}/{self._heatMax}")
        auto = self.get_blacklevel_autoadjust()
        parts.append(f"blacklevelAutoAdjust={'n/a' if auto is None else auto}")
        if self._hasBlacklevel:
            parts.append(f"blacklevel={self.blacklevel}/{self.blacklevel_max()}")
        self.__logger.info(f"{self.deviceName} low-noise config: {', '.join(parts)}")

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

    def _applyTecTarget(self, temperature_c, quiet: bool = False) -> bool:
        """Switch the cooler on and push ``temperature_c`` (no safety checks).

        ``quiet`` drops the log line to debug, for the periodic keep-alive that
        would otherwise write an INFO line every minute.
        """
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_TEC, 1)
            self.hcam.put_Temperature(int(round(temperature_c * 10)))
            self._lastTecReassert = time.time()
            message = (f"TEC on, target {temperature_c:.1f} °C "
                       f"(sensor now {self.get_temperature()} °C)")
            if quiet:
                self.__logger.debug(f"Re-asserted: {message}")
            else:
                self.__logger.info(message)
            return True
        except Exception as e:
            self.__logger.error(f"Set temperature failed: {e}")
            return False

    def _disableTec(self, reason: str) -> bool:
        """Switch the cooler off, keeping the cached target for a later re-arm."""
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_TEC, 0)
            self.__logger.warning(f"TEC switched off: {reason}")
            return True
        except Exception as e:
            self.__logger.error(f"Switching the TEC off failed: {e}")
            return False

    def set_temperature(self, temperature_c):
        """TEC target temperature in °C (TEC models only); also switches the
        cooler on. Logged at INFO because an unexpected warm-up is otherwise
        impossible to trace back to whoever changed the target.

        An explicit call is also the override for a tripped over-temperature
        cutoff, so it clears it. The cooler is still left off while the sensor
        is at or above TEMPERATURE_SHUTDOWN_C; the monitor applies this target
        once the camera has cooled down.
        """
        if not self._hasTEC:
            self.__logger.debug("Camera has no controllable TEC")
            return
        try:
            temperature_c = float(temperature_c)
        except (TypeError, ValueError):
            self.__logger.warning(f"Ignoring invalid TEC target {temperature_c!r}")
            return

        if self._tecTargetRange is not None:
            low, high = self._tecTargetRange
            clamped = max(low, min(high, temperature_c))
            if clamped != temperature_c:
                self.__logger.warning(
                    f"TEC target {temperature_c:.1f} °C is outside this "
                    f"camera's {low:.1f}..{high:.1f} °C range, using "
                    f"{clamped:.1f} °C instead. An out-of-range target is "
                    f"accepted by the SDK and read back unchanged, but the "
                    f"cooler may then not regulate at all and the sensor "
                    f"stays at ambient.")
                temperature_c = clamped

        # Cached either way, so a reconnect or a cutoff recovery re-applies what
        # was last asked for rather than the camera's power-on default.
        self.targetTemperature = temperature_c
        self._tecOverTemp = False
        self._tempOverSamples = 0
        self._tempRecoverySamples = 0
        self._tecRecoveryAttempts = 0
        self._tecRecoveryExhaustedLogged = False
        self._tecStallSince = None
        self._tecStallLogged = False

        current = self.get_temperature()
        if current is not None and current >= TEMPERATURE_SHUTDOWN_C:
            self._tecOverTemp = True
            self.__logger.warning(
                f"Sensor at {current:.1f} °C is at or above the "
                f"{TEMPERATURE_SHUTDOWN_C:.0f} °C cutoff: target "
                f"{temperature_c:.1f} °C stored, but the cooler stays off "
                f"until the camera drops below {TEMPERATURE_RECOVERY_C:.0f} °C")
            self._disableTec("over temperature")
            return
        self._applyTecTarget(temperature_c)

    def get_target_temperature(self):
        """TEC target in °C as the camera reports it, or None if unsupported."""
        if not self._hasTEC:
            return None
        try:
            return self.hcam.get_Option(toupcam.TOUPCAM_OPTION_TECTARGET) / 10.0
        except Exception:
            return None

    def get_tec_enabled(self):
        """True/False for the cooler state, or None if unsupported."""
        if not self._hasTEC:
            return None
        try:
            return bool(self.hcam.get_Option(toupcam.TOUPCAM_OPTION_TEC))
        except Exception:
            return None

    def set_fan_speed(self, speed):
        """Fan speed: 0 = off, 1..max, -1 = the camera's default speed."""
        if not self._hasFan:
            return
        try:
            self.hcam.put_Option(toupcam.TOUPCAM_OPTION_FAN, int(speed))
            self.fanSpeed = int(speed)
        except toupcam.HRESULTException as ex:
            self.__logger.error(f"Set fan speed failed hr=0x{ex.hr & 0xffffffff:x}")

    def get_fan_speed(self):
        """Current fan speed as the camera reports it, or None if unsupported."""
        if not self._hasFan:
            return None
        try:
            return int(self.hcam.get_Option(toupcam.TOUPCAM_OPTION_FAN))
        except Exception:
            return None

    # ---------------------------------------------------------------------
    # Temperature monitor: over-temperature cutoff + optional CSV log
    # ---------------------------------------------------------------------
    def _temperatureLogPath(self) -> str:
        """CSV path in today's recordings folder (created on demand).

        Resolved per sample rather than once at start so a log that runs past
        midnight continues in the new day's folder, next to that day's snaps.
        """
        from imswitch.imcommon.model import dirtools
        day = datetime.date.today().strftime("%Y-%m-%d")
        folder = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(),
                              "recordings", day)
        os.makedirs(folder, exist_ok=True)
        return os.path.join(folder, TEMPERATURE_LOG_FILENAME)

    def _writeTemperatureSample(self, temperature) -> bool:
        """Append one row to the CSV. Returns False if nothing was written.

        ``temperature`` is passed in rather than read here so the monitor's
        safety check and the log row describe the very same sample.
        """
        if temperature is None:
            # Handle gone (reconnecting) or read failed -- no point in a row
            # of blanks.
            return False
        target = self.get_target_temperature()
        tecOn = self.get_tec_enabled()
        fan = self.get_fan_speed()
        heat = self.get_heat()
        now = datetime.datetime.now()
        row = (
            now.isoformat(timespec="seconds"),
            f"{now.timestamp():.1f}",
            getattr(self, "deviceName", self.model),
            f"{temperature:.1f}",
            "" if target is None else f"{target:.1f}",
            "" if tecOn is None else int(tecOn),
            "" if fan is None else fan,
            "" if heat is None else heat,
            f"{float(self.exposure_time):.3f}",
            int(self.is_streaming),
            int(self._tecOverTemp),
        )
        try:
            path = self._temperatureLogPath()
            writeHeader = not os.path.exists(path) or os.path.getsize(path) == 0
            with open(path, "a", newline="") as f:
                writer = csv.writer(f)
                if writeHeader:
                    writer.writerow(TEMPERATURE_LOG_COLUMNS)
                writer.writerow(row)
            self._tempLogErrorLogged = False
            return True
        except Exception as e:
            # Once per failure streak: a full disk or a vanished folder would
            # otherwise repeat every 5 s for the whole acquisition.
            if not self._tempLogErrorLogged:
                self.__logger.error(f"Temperature log write failed: {e}")
                self._tempLogErrorLogged = True
            return False

    def _checkTemperatureSafety(self, temperature: float) -> None:
        """Cut the cooler out above TEMPERATURE_SHUTDOWN_C, re-arm when cooled.

        The sensor of a cooled camera sits well below ambient in normal
        operation, so a reading this high means the heat is not getting out
        (fan stalled, ventilation blocked, TEC failing) -- and from there the
        cooler's own dissipation only makes it worse. The window heater is
        switched off with it for the same reason.
        """
        if not self._hasTEC:
            return

        if temperature >= TEMPERATURE_SHUTDOWN_C:
            self._tempRecoverySamples = 0
            self._tempOverSamples += 1
            if (self._tecOverTemp
                    or self._tempOverSamples < TEMPERATURE_SHUTDOWN_SAMPLES):
                return
            self._tecOverTemp = True
            self.__logger.error(
                f"Sensor temperature {temperature:.1f} °C reached the "
                f"{TEMPERATURE_SHUTDOWN_C:.0f} °C limit: switching the cooler "
                f"off. Check the fan and the ventilation.")
            self._disableTec(f"sensor at {temperature:.1f} °C")
            if self._hasHeat and (self.get_heat() or 0) > 0:
                self.set_heat(0)
            return

        self._tempOverSamples = 0
        if not self._tecOverTemp or temperature > TEMPERATURE_RECOVERY_C:
            return

        self._tempRecoverySamples += 1
        if self._tempRecoverySamples < TEMPERATURE_SHUTDOWN_SAMPLES:
            return
        self._tempRecoverySamples = 0
        if self._tecRecoveryAttempts >= TEMPERATURE_RECOVERY_ATTEMPTS:
            if not self._tecRecoveryExhaustedLogged:
                self.__logger.error(
                    f"Cooler tripped on temperature "
                    f"{TEMPERATURE_RECOVERY_ATTEMPTS} times; leaving it off. "
                    f"Fix the cooling, then set the target temperature again.")
                self._tecRecoveryExhaustedLogged = True
            return
        self._tecRecoveryAttempts += 1
        self._tecOverTemp = False
        self.__logger.info(
            f"Sensor back down to {temperature:.1f} °C, re-arming the cooler "
            f"at {self.targetTemperature:.1f} °C (attempt "
            f"{self._tecRecoveryAttempts}/{TEMPERATURE_RECOVERY_ATTEMPTS})")
        self._applyTecTarget(self.targetTemperature)

    def _maintainTecTarget(self, temperature: float) -> None:
        """Re-assert the TEC target periodically and warn when it is not working.

        Writing the target once is not reliable: the camera can report the
        requested target back on a read, with the cooler on, and still leave
        the sensor at ambient indefinitely. Re-writing it every
        TEMPERATURE_REASSERT_INTERVAL_S is what gets it regulating again.

        Skipped while the over-temperature cutoff is tripped: there the cooler
        is off deliberately and _checkTemperatureSafety owns it.
        """
        if not self._hasTEC or self._tecOverTemp:
            return

        now = time.time()
        if now - self._lastTecReassert >= TEMPERATURE_REASSERT_INTERVAL_S:
            self._applyTecTarget(self.targetTemperature, quiet=True)

        if temperature <= self.targetTemperature + TEMPERATURE_STALL_MARGIN_C:
            self._tecStallSince = None
            if self._tecStallLogged:
                self.__logger.info(
                    f"Cooler is regulating again: sensor {temperature:.1f} °C, "
                    f"target {self.targetTemperature:.1f} °C")
                self._tecStallLogged = False
            return

        if self._tecStallSince is None:
            self._tecStallSince = now
            return
        if (self._tecStallLogged
                or now - self._tecStallSince < TEMPERATURE_STALL_AFTER_S):
            return
        self._tecStallLogged = True
        rangeHint = ""
        if self._tecTargetRange is not None:
            rangeHint = (f" This model accepts {self._tecTargetRange[0]:.1f}.."
                         f"{self._tecTargetRange[1]:.1f} °C.")
        self.__logger.warning(
            f"Cooler is not regulating: the sensor has been at "
            f"{temperature:.1f} °C for "
            f"{(now - self._tecStallSince) / 60.0:.0f} min with the TEC on and "
            f"a target of {self.targetTemperature:.1f} °C (the camera reports "
            f"{self.get_target_temperature()}). Check the target, the fan and "
            f"the heat path.{rangeHint}")

    def _temperatureMonitorLoop(self):
        while True:
            try:
                temperature = self.get_temperature()
                if temperature is not None:
                    self._checkTemperatureSafety(temperature)
                    self._maintainTecTarget(temperature)
                    # Only while armed: the log is meant to describe
                    # acquisitions, not the hours the camera idles in between.
                    if self.readSaveTemperature and self.is_streaming:
                        self._writeTemperatureSample(temperature)
            except Exception as e:
                # Never let one bad sample end the thread -- it is what keeps
                # the over-temperature cutoff alive. Loud once, quiet after.
                if not self._tempMonitorErrorLogged:
                    self.__logger.error(f"Temperature monitor sample failed: {e}")
                    self._tempMonitorErrorLogged = True
                else:
                    self.__logger.debug(f"Temperature monitor sample failed: {e}")
            else:
                self._tempMonitorErrorLogged = False
            if self._tempMonitorStop.wait(TEMPERATURE_LOG_INTERVAL_S):
                return

    def _startTemperatureMonitor(self):
        """Start the polling thread (no-op on models without a temperature
        sensor, or when it is already running).

        Independent of readSaveTemperature: the over-temperature cutoff has to
        work whether or not anybody asked for a log.
        """
        if not self._hasGetTemperature:
            return
        if (self._tempMonitorThread is not None
                and self._tempMonitorThread.is_alive()):
            return
        self._tempMonitorStop.clear()
        self._tempMonitorThread = threading.Thread(
            target=self._temperatureMonitorLoop,
            name="ToupcamTemperatureMonitor", daemon=True)
        self._tempMonitorThread.start()
        self.__logger.info(
            f"Monitoring sensor temperature every "
            f"{TEMPERATURE_LOG_INTERVAL_S:.0f} s; cooler off above "
            f"{TEMPERATURE_SHUTDOWN_C:.0f} °C")
        if self.readSaveTemperature:
            self.__logger.info(
                f"Logging it while armed to {self._temperatureLogPath()}")

    def _stopTemperatureMonitor(self):
        thread = self._tempMonitorThread
        if thread is None:
            return
        self._tempMonitorStop.set()
        if thread is not threading.current_thread():
            thread.join(timeout=2.0)
        self._tempMonitorThread = None

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

    def snapSoftwareTrigger(self, timeout: float = None):
        """Fire one software trigger and return the frame it produces.

        Requires software-trigger mode (``setTriggerSource('software')``); the
        returned image is guaranteed to be exposed AFTER this call — the basis
        for deterministic post-move grabs (see CameraHIK.snapSoftwareTrigger).

        ``timeout`` defaults to one integration plus a margin so long exposures
        are waited out; pass a value to override.
        """
        if timeout is None:
            timeout = self._frameTimeout(base=2.0)
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
        elif property_name in ("conversion_gain", "conversionGain"):
            self.set_conversion_gain(property_value)
        elif property_name in ("low_noise", "lowNoise"):
            self.set_low_noise(property_value)
        elif property_name == "heat":
            self.set_heat(property_value)
        elif property_name in ("blacklevel_autoadjust", "blacklevelAutoAdjust"):
            self.set_blacklevel_autoadjust(property_value)
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
        elif property_name == "target_temperature":
            return self.get_target_temperature()
        elif property_name == "fan_speed":
            return self.get_fan_speed()
        elif property_name == "binning":
            return self.binning
        elif property_name == "pixel_format":
            if self.isRGB:
                return "rgb24"
            return "mono16" if self._bits > 8 else "mono8"
        elif property_name in ("conversion_gain", "conversionGain"):
            return self.get_conversion_gain()
        elif property_name in ("low_noise", "lowNoise"):
            return self.get_low_noise()
        elif property_name == "heat":
            return self.get_heat()
        elif property_name in ("blacklevel_autoadjust", "blacklevelAutoAdjust"):
            return self.get_blacklevel_autoadjust()
        elif property_name == "blacklevel_max":
            return self.blacklevel_max() if self._hasBlacklevel else None
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
            # container depth (8/16/24) vs. what the sensor actually delivers
            params["bit_depth"] = self._bits
            params["sensor_bit_depth"] = self._maxBitDepth
            params["max_adu"] = self.max_adu
            params["isRGB"] = self.isRGB
            cur, emin, emax = self.get_exposuretime()
            params["exposure_us"] = cur
            params["exposure_range_us"] = (emin, emax)
            params["exposure_ms"] = None if cur is None else cur / 1000.0
            params["exposure_range_ms"] = (
                None if emin is None else emin / 1000.0,
                None if emax is None else emax / 1000.0,
            )
            gcur, gmin, gmax = self.get_gain()
            params["gain"] = gcur
            params["gain_range"] = (gmin, gmax)
            ncur, nmin, nmax = self.get_gain_native()
            params["gain_percent"] = ncur
            params["gain_range_percent"] = (nmin, nmax)
            params["long_exposure_threshold_ms"] = LONG_EXPOSURE_THRESHOLD_MS
            params["is_long_exposure"] = self.isLongExposure()
            temp = self.get_temperature()
            if temp is not None:
                params["temperature_c"] = temp
            if self._hasTEC:
                params["tec_on"] = self.get_tec_enabled()
                params["tec_target_c"] = self.get_target_temperature()
                params["tec_over_temperature"] = self._tecOverTemp
                params["tec_shutdown_c"] = TEMPERATURE_SHUTDOWN_C
                params["tec_target_range_c"] = self._tecTargetRange
                params["tec_stalled"] = self._tecStallLogged
            if self._hasFan:
                params["fan_speed"] = self.get_fan_speed()
        except Exception:
            pass
        # Low-noise configuration, read back from the device rather than from
        # our cached request -- these are the values that explain a noise floor
        # or a black level that will not move.
        try:
            params["supports_heat"] = self._hasHeat
            params["supports_low_noise"] = self._hasLowNoise
            params["supports_conversion_gain"] = self._hasCG
            params["supports_blacklevel"] = self._hasBlacklevel
            params["conversion_gain"] = self.get_conversion_gain()
            params["low_noise"] = self.get_low_noise()
            params["heat"] = self.get_heat()
            params["heat_max"] = self._heatMax
            params["blacklevel_autoadjust"] = self.get_blacklevel_autoadjust()
            if self._hasBlacklevel:
                params["blacklevel"] = self.getPropertyValue("blacklevel")
                params["blacklevel_max"] = self.blacklevel_max()
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
            "sensor_bit_depth": self._maxBitDepth,
            "max_adu": self.max_adu,
            "exposure_ms": self.exposure_time,
            "is_long_exposure": self.isLongExposure(),
            "gain": self.gain,
            "gain_percent_native": self._uiGainToNative(self.gain),
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
