"""
Tests for the ToupTek low-noise / long-exposure configuration:
window heater, conversion gain (HCG), low-noise readout and black level.

They run against a fake SDK handle, so no camera and no native library call is
involved. ``CameraToupcam`` is built with ``object.__new__`` because its
``__init__`` opens a device; the attributes set here are the ones the tested
methods touch.
"""

from __future__ import annotations

import csv
import threading
import time

import pytest


toupcam = pytest.importorskip(
    "imswitch.imcontrol.model.interfaces.toupcamsdk.toupcam")

from imswitch.imcontrol.model.interfaces.toupcamcamera import (  # noqa: E402
    CameraToupcam,
    TEMPERATURE_RECOVERY_C as _RECOVERY_C,
    TEMPERATURE_SHUTDOWN_C as _SHUTDOWN_C,
)


class FakeHcam:
    """Records put_Option writes and serves get_Option from the same store."""

    def __init__(self, options=None, unsupported=()):
        self.options = {
            toupcam.TOUPCAM_OPTION_HEAT_MAX: 5,
            toupcam.TOUPCAM_OPTION_HEAT: 0,
            toupcam.TOUPCAM_OPTION_LOW_NOISE: 0,
            toupcam.TOUPCAM_OPTION_CG: 0,
            toupcam.TOUPCAM_OPTION_BLACKLEVEL: 0,
            toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST: 1,
        }
        if options:
            self.options.update(options)
        # Options this model does not implement -- the real SDK raises for these.
        self.unsupported = set(unsupported)
        self.writes = []

    def put_Option(self, option, value):
        if option in self.unsupported:
            raise toupcam.HRESULTException(0x80004001)  # E_NOTIMPL
        self.writes.append((option, value))
        self.options[option] = value

    def get_Option(self, option):
        if option in self.unsupported:
            raise toupcam.HRESULTException(0x80004001)
        return self.options[option]


class _SilentLogger:
    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, msg, *a, **k):
        self.warnings.append(msg)

    def info(self, msg, *a, **k):
        self.infos.append(msg)

    def debug(self, *a, **k):
        pass

    def error(self, msg, *a, **k):
        self.warnings.append(msg)


def _make_camera(hcam=None, *, bits=16, maxBitDepth=14, heat=True,
                 lowNoise=True, cg=True, cghdr=False, blacklevel=True):
    cam = object.__new__(CameraToupcam)
    cam.hcam = hcam if hcam is not None else FakeHcam()
    cam._CameraToupcam__logger = _SilentLogger()
    cam.deviceName = "FakeToupcam"
    cam.is_streaming = False
    cam.isRGB = False
    cam._bits = bits
    cam._maxBitDepth = maxBitDepth
    cam._hasHeat = heat
    cam._hasLowNoise = lowNoise
    cam._hasCG = cg
    cam._hasCGHDR = cghdr
    cam._hasBlacklevel = blacklevel
    cam._hasTEC = False
    cam._hasGetTemperature = False
    cam._hasFan = False
    cam.targetTemperature = -10.0
    cam.fanSpeed = -1
    cam.readSaveTemperature = False
    cam._tempMonitorThread = None
    cam._tempMonitorStop = threading.Event()
    cam._tempLogErrorLogged = False
    cam._tempMonitorErrorLogged = False
    cam._tecOverTemp = False
    cam._tempOverSamples = 0
    cam._tempRecoverySamples = 0
    cam._tecRecoveryAttempts = 0
    cam._tecRecoveryExhaustedLogged = False
    cam._tecTargetRange = None
    cam._lastTecReassert = 0.0
    cam._tecStallSince = None
    cam._tecStallLogged = False
    cam.exposure_time = 100
    cam.model = "CameraToupcam"
    cam._heatMax = 5
    cam.blacklevel = 0
    cam.heat = None
    cam.lowNoise = None
    cam.conversionGain = None
    cam.blacklevelAutoAdjust = None
    return cam


class TestLowNoiseConfiguration:
    def test_hcg_low_noise_and_heater_are_applied(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam)

        cam.set_conversion_gain("HCG")
        cam.set_low_noise(True)
        cam.set_heat(True)

        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_LOW_NOISE] == 1
        # heat=True means the camera's maximum level, not "1".
        assert hcam.options[toupcam.TOUPCAM_OPTION_HEAT] == 5
        assert cam.get_conversion_gain() == "HCG"
        assert cam.get_low_noise() is True
        assert cam.get_heat() == 5

    def test_conversion_gain_accepts_names_and_numbers(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam)

        cam.set_conversion_gain("lcg")
        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 0
        cam.set_conversion_gain(1)
        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 1

    def test_hdr_falls_back_to_hcg_without_the_flag(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam, cghdr=False)

        cam.set_conversion_gain("HDR")

        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 1
        assert any("HDR" in w for w in cam._CameraToupcam__logger.warnings)

    def test_unsupported_capabilities_are_skipped_not_written(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam, heat=False, lowNoise=False, cg=False)

        cam.set_conversion_gain("HCG")
        cam.set_low_noise(True)
        cam.set_heat(True)

        # Nothing written, nothing raised: the camera simply has no such mode.
        assert hcam.writes == []
        assert cam.get_conversion_gain() is None
        assert cam.get_low_noise() is None
        assert cam.get_heat() is None

    def test_heat_level_is_clamped_to_the_camera_maximum(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam)

        cam.set_heat(99)

        assert hcam.options[toupcam.TOUPCAM_OPTION_HEAT] == 5

    def test_settings_are_reapplied_after_a_reconnect(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam)
        cam.exposure_time = 0
        cam.gain = 0
        cam.binning = 1
        cam.frame_rate = -1
        cam.trigger_source = "Continous"
        cam.conversionGain = "HCG"
        cam.lowNoise = True
        cam.heat = True
        cam.set_frame_rate = lambda *a, **k: None
        cam.setTriggerSource = lambda *a, **k: None

        cam._reapply_settings()

        # A reopened handle comes up with the camera defaults, so all three
        # have to be pushed again.
        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_LOW_NOISE] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_HEAT] == 5


class TestBlackLevel:
    def test_maximum_follows_the_output_bit_depth(self):
        assert _make_camera(bits=8).blacklevel_max() == 31
        assert _make_camera(bits=16, maxBitDepth=12).blacklevel_max() == 31 * 16
        assert _make_camera(bits=16, maxBitDepth=14).blacklevel_max() == 31 * 64
        assert _make_camera(bits=16, maxBitDepth=16).blacklevel_max() == 31 * 256

    def test_out_of_range_value_is_clamped_and_reported(self):
        hcam = FakeHcam()
        # 8-bit output: the maximum is 31, so a requested 100 cannot show up.
        cam = _make_camera(hcam, bits=8)

        cam.set_blacklevel(100)

        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL] == 31
        assert cam.blacklevel == 31
        assert any("outside 0..31" in w for w in cam._CameraToupcam__logger.warnings)

    def test_auto_adjust_is_turned_off_so_the_value_takes_effect(self):
        # Auto-adjust on: the camera keeps re-deriving the offset from its
        # optical-black pixels and overwrites the manual value.
        hcam = FakeHcam({toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST: 1})
        cam = _make_camera(hcam)

        cam.set_blacklevel(100)

        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST] == 0
        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL] == 100
        # ...and the disable happens before the write, or the camera would
        # re-adjust in between.
        order = [opt for opt, _ in hcam.writes]
        assert order.index(toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST) < \
            order.index(toupcam.TOUPCAM_OPTION_BLACKLEVEL)

    def test_in_range_value_is_written_unchanged_at_high_bit_depth(self):
        hcam = FakeHcam()
        cam = _make_camera(hcam, bits=16, maxBitDepth=14)   # max 1984

        cam.set_blacklevel(100)

        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL] == 100
        assert cam.blacklevel == 100
        assert not any("outside" in w for w in cam._CameraToupcam__logger.warnings)

    def test_models_without_the_auto_adjust_option_still_work(self):
        hcam = FakeHcam(unsupported={toupcam.TOUPCAM_OPTION_BLACKLEVEL_AUTOADJUST})
        cam = _make_camera(hcam)

        cam.set_blacklevel(100)

        assert cam.get_blacklevel_autoadjust() is None
        assert hcam.options[toupcam.TOUPCAM_OPTION_BLACKLEVEL] == 100

    def test_read_back_mismatch_is_reported(self):
        class StickyHcam(FakeHcam):
            def put_Option(self, option, value):
                # Camera accepts the write but keeps its own value -- exactly
                # the silent failure this read-back exists to catch.
                if option == toupcam.TOUPCAM_OPTION_BLACKLEVEL:
                    self.writes.append((option, value))
                    return
                super().put_Option(option, value)

        hcam = StickyHcam()
        cam = _make_camera(hcam)

        cam.set_blacklevel(100)

        assert any("reads back" in w for w in cam._CameraToupcam__logger.warnings)


class FakeTecHcam(FakeHcam):
    """FakeHcam with the TEC / temperature calls of a cooled model."""

    def __init__(self, **kw):
        super().__init__(**kw)
        self.options.setdefault(toupcam.TOUPCAM_OPTION_TEC, 0)
        self.options.setdefault(toupcam.TOUPCAM_OPTION_TECTARGET, 100)
        self.options.setdefault(toupcam.TOUPCAM_OPTION_FAN, 0)
        self.sensorTemperature = 215  # 0.1 °C

    def put_Temperature(self, nTemperature):
        self.options[toupcam.TOUPCAM_OPTION_TECTARGET] = int(nTemperature)
        self.temperatureWrites = getattr(self, 'temperatureWrites', 0) + 1

    def get_TecTargetRange(self):
        return (-300, 500)  # 0.1 degC units, like the SDK

    def get_Temperature(self):
        return self.sensorTemperature


def _make_cooled_camera(hcam=None):
    hcam = hcam if hcam is not None else FakeTecHcam()
    cam = _make_camera(hcam)
    cam._hasTEC = True
    cam._hasGetTemperature = True
    cam._hasFan = True
    cam.targetTemperature = -10.0
    cam.fanSpeed = -1
    cam._tecTargetRange = (-30.0, 50.0)
    return cam


class TestCooling:
    def test_set_temperature_turns_tec_on_and_caches_target(self):
        hcam = FakeTecHcam()
        cam = _make_cooled_camera(hcam)

        cam.set_temperature(-30)

        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_TECTARGET] == -300
        assert cam.targetTemperature == -30.0
        assert cam.get_target_temperature() == -30.0
        assert cam.get_tec_enabled() is True
        assert cam.getPropertyValue("target_temperature") == -30.0
        assert cam.getPropertyValue("temperature") == 21.5

    def test_reapply_settings_restores_tec_target_after_reopen(self):
        # A reopened handle comes up with the model default (here +10 °C, TEC
        # off); the cached user target has to be pushed again.
        cam = _make_cooled_camera()
        cam.set_temperature(-30)
        cam.set_fan_speed(2)
        fresh = FakeTecHcam()
        cam.hcam = fresh
        cam.exposure_time = 0
        cam.gain = 0
        cam.binning = 1
        cam.frame_rate = -1
        cam.trigger_source = "Continous"
        cam.set_frame_rate = lambda *_: None
        cam.setTriggerSource = lambda *_: None

        cam._reapply_settings()

        assert fresh.options[toupcam.TOUPCAM_OPTION_TEC] == 1
        assert fresh.options[toupcam.TOUPCAM_OPTION_TECTARGET] == -300
        assert fresh.options[toupcam.TOUPCAM_OPTION_FAN] == 2

    def test_invalid_target_is_ignored(self):
        hcam = FakeTecHcam()
        cam = _make_cooled_camera(hcam)
        cam.set_temperature(-30)

        cam.set_temperature("not a number")

        assert hcam.options[toupcam.TOUPCAM_OPTION_TECTARGET] == -300
        assert cam.targetTemperature == -30.0


class TestTemperatureMonitor:
    """readSaveTemperature CSV rows plus the over-temperature cutoff."""

    def _cooled_logging_camera(self, tmp_path, monkeypatch):
        from imswitch.imcommon.model import dirtools
        from imswitch.imcontrol.model.interfaces import toupcamcamera

        monkeypatch.setattr(dirtools.UserFileDirs, "getValidatedDataPath",
                            classmethod(lambda cls: str(tmp_path)))
        monkeypatch.setattr(toupcamcamera, "TEMPERATURE_LOG_INTERVAL_S", 0.05)
        cam = _make_cooled_camera()
        cam.readSaveTemperature = True
        # start_live/stop_live of the real driver touch the SDK handle; the
        # monitor is driven directly here, the way those two call it.
        return cam

    def _log_file(self, tmp_path):
        from imswitch.imcontrol.model.interfaces.toupcamcamera import (
            TEMPERATURE_LOG_FILENAME)
        day = time.strftime("%Y-%m-%d")
        return tmp_path / "recordings" / day / TEMPERATURE_LOG_FILENAME

    def _rows(self, tmp_path):
        rows = list(csv.reader(self._log_file(tmp_path).open()))
        return rows[0], rows[1:]

    def test_rows_are_appended_while_armed(self, tmp_path, monkeypatch):
        cam = self._cooled_logging_camera(tmp_path, monkeypatch)
        cam.set_temperature(-30)
        cam.is_streaming = True

        cam._startTemperatureMonitor()
        time.sleep(0.3)
        cam._stopTemperatureMonitor()
        assert cam._tempMonitorThread is None

        header, body = self._rows(tmp_path)
        assert header[0] == "timestamp"
        assert len(body) >= 3
        sample = dict(zip(header, body[-1]))
        assert sample["sensor_temperature_c"] == "21.5"
        assert sample["tec_target_c"] == "-30.0"
        assert sample["tec_on"] == "1"
        assert sample["streaming"] == "1"
        assert sample["over_temperature"] == "0"

    def test_no_rows_while_idle_but_monitor_keeps_running(self, tmp_path, monkeypatch):
        # The cooler runs when the camera is idle, so the safety monitor has to
        # keep polling -- it just must not log acquisition rows.
        cam = self._cooled_logging_camera(tmp_path, monkeypatch)
        cam.is_streaming = False

        cam._startTemperatureMonitor()
        time.sleep(0.2)
        assert cam._tempMonitorThread.is_alive()
        cam._stopTemperatureMonitor()

        assert not self._log_file(tmp_path).exists()

    def test_second_session_appends_to_the_same_file(self, tmp_path, monkeypatch):
        cam = self._cooled_logging_camera(tmp_path, monkeypatch)
        cam.is_streaming = True
        for _ in range(2):
            cam._startTemperatureMonitor()
            time.sleep(0.12)
            cam._stopTemperatureMonitor()
        from imswitch.imcontrol.model.interfaces.toupcamcamera import (
            TEMPERATURE_LOG_COLUMNS)
        header, body = self._rows(tmp_path)
        assert header == list(TEMPERATURE_LOG_COLUMNS)
        assert len(body) >= 4

    def test_no_sensor_starts_nothing(self, tmp_path, monkeypatch):
        cam = self._cooled_logging_camera(tmp_path, monkeypatch)
        cam._hasGetTemperature = False
        cam._startTemperatureMonitor()
        assert cam._tempMonitorThread is None
        assert not self._log_file(tmp_path).exists()


class TestOverTemperatureCutoff:
    def _hot_camera(self, temperature_c=None, heat=0):
        if temperature_c is None:
            temperature_c = _SHUTDOWN_C + 5.0
        hcam = FakeTecHcam()
        hcam.options[toupcam.TOUPCAM_OPTION_HEAT] = heat
        cam = _make_cooled_camera(hcam)
        cam.set_temperature(-30)
        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 1
        hcam.sensorTemperature = int(temperature_c * 10)
        return cam, hcam

    def test_one_hot_sample_does_not_trip(self):
        cam, hcam = self._hot_camera()

        cam._checkTemperatureSafety(cam.get_temperature())

        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 1
        assert cam._tecOverTemp is False

    def test_two_hot_samples_switch_the_cooler_and_heater_off(self):
        cam, hcam = self._hot_camera(heat=5)

        for _ in range(2):
            cam._checkTemperatureSafety(cam.get_temperature())

        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 0
        assert hcam.options[toupcam.TOUPCAM_OPTION_HEAT] == 0
        assert cam._tecOverTemp is True
        # The target is kept, so a recovery can restore it.
        assert cam.targetTemperature == -30.0

    def test_cooler_is_re_armed_once_the_camera_cooled_down(self):
        cam, hcam = self._hot_camera()
        for _ in range(2):
            cam._checkTemperatureSafety(cam.get_temperature())
        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 0

        # Still too warm for the hysteresis band: below the cutoff, but not
        # yet back down to the re-arm point.
        hcam.sensorTemperature = int((_SHUTDOWN_C - 1.0) * 10)
        for _ in range(2):
            cam._checkTemperatureSafety(cam.get_temperature())
        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 0

        hcam.sensorTemperature = int((_RECOVERY_C - 1.0) * 10)
        cam._checkTemperatureSafety(cam.get_temperature())
        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 0  # needs two samples
        cam._checkTemperatureSafety(cam.get_temperature())

        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_TECTARGET] == -300
        assert cam._tecOverTemp is False

    def test_repeated_trips_eventually_leave_the_cooler_off(self):
        from imswitch.imcontrol.model.interfaces.toupcamcamera import (
            TEMPERATURE_RECOVERY_ATTEMPTS)
        cam, hcam = self._hot_camera()

        for _ in range(TEMPERATURE_RECOVERY_ATTEMPTS + 1):
            hcam.sensorTemperature = int((_SHUTDOWN_C + 5.0) * 10)
            for _ in range(2):
                cam._checkTemperatureSafety(cam.get_temperature())
            hcam.sensorTemperature = int((_RECOVERY_C - 1.0) * 10)
            for _ in range(2):
                cam._checkTemperatureSafety(cam.get_temperature())

        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 0
        assert cam._tecOverTemp is True
        assert cam._tecRecoveryAttempts == TEMPERATURE_RECOVERY_ATTEMPTS

        # An explicit target is the user override and brings the cooler back.
        cam.set_temperature(-30)
        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 1
        assert cam._tecOverTemp is False

    def test_setting_a_target_while_hot_stores_it_without_cooling(self):
        cam, hcam = self._hot_camera()
        hcam.options[toupcam.TOUPCAM_OPTION_TEC] = 1

        cam.set_temperature(-20)

        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 0
        assert cam.targetTemperature == -20.0
        assert cam._tecOverTemp is True

        # ... and the monitor applies it once the camera has cooled down.
        hcam.sensorTemperature = int((_RECOVERY_C - 1.0) * 10)
        for _ in range(2):
            cam._checkTemperatureSafety(cam.get_temperature())
        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_TECTARGET] == -200

    def test_over_temperature_is_recorded_in_the_log_row(self, tmp_path, monkeypatch):
        from imswitch.imcommon.model import dirtools
        monkeypatch.setattr(dirtools.UserFileDirs, "getValidatedDataPath",
                            classmethod(lambda cls: str(tmp_path)))
        cam, hcam = self._hot_camera()
        cam.readSaveTemperature = True
        cam.is_streaming = True
        for _ in range(2):
            cam._checkTemperatureSafety(cam.get_temperature())

        cam._writeTemperatureSample(cam.get_temperature())

        from imswitch.imcontrol.model.interfaces.toupcamcamera import (
            TEMPERATURE_LOG_FILENAME)
        path = (tmp_path / "recordings" / time.strftime("%Y-%m-%d")
                / TEMPERATURE_LOG_FILENAME)
        rows = list(csv.reader(path.open()))
        sample = dict(zip(rows[0], rows[-1]))
        assert sample["over_temperature"] == "1"
        assert sample["tec_on"] == "0"
        assert sample["sensor_temperature_c"] == f"{_SHUTDOWN_C + 5.0:.1f}"


class TestTecKeepAlive:
    """The TEC target is re-written periodically, and a stall is reported.

    A single write is not reliable on this family: the camera reports the
    target back correctly, with the cooler on, and can still sit at ambient
    for hours (observed in a real temperature log at target -40 degC).
    """

    def test_target_below_the_model_range_is_clamped(self):
        hcam = FakeTecHcam()
        cam = _make_cooled_camera(hcam)

        cam.set_temperature(-40)

        assert cam.targetTemperature == -30.0
        assert hcam.options[toupcam.TOUPCAM_OPTION_TECTARGET] == -300
        assert any("outside" in w for w in cam._CameraToupcam__logger.warnings)

    def test_target_inside_the_range_is_written_unchanged(self):
        hcam = FakeTecHcam()
        cam = _make_cooled_camera(hcam)

        cam.set_temperature(-25)

        assert cam.targetTemperature == -25.0
        assert hcam.options[toupcam.TOUPCAM_OPTION_TECTARGET] == -250

    def test_target_is_re_asserted_after_the_interval(self, monkeypatch):
        from imswitch.imcontrol.model.interfaces import toupcamcamera
        hcam = FakeTecHcam()
        cam = _make_cooled_camera(hcam)
        cam.set_temperature(-30)
        hcam.sensorTemperature = -300
        writes = hcam.temperatureWrites

        # Immediately after setting it, nothing is re-written.
        cam._maintainTecTarget(cam.get_temperature())
        assert hcam.temperatureWrites == writes

        # Once the interval has passed, the target goes out again.
        cam._lastTecReassert -= toupcamcamera.TEMPERATURE_REASSERT_INTERVAL_S + 1
        cam._maintainTecTarget(cam.get_temperature())
        assert hcam.temperatureWrites == writes + 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 1
        assert hcam.options[toupcam.TOUPCAM_OPTION_TECTARGET] == -300

    def test_no_re_assert_while_the_cutoff_is_tripped(self):
        from imswitch.imcontrol.model.interfaces import toupcamcamera
        hcam = FakeTecHcam()
        cam = _make_cooled_camera(hcam)
        cam.set_temperature(-30)
        hcam.sensorTemperature = 450
        for _ in range(2):
            cam._checkTemperatureSafety(cam.get_temperature())
        assert cam._tecOverTemp is True
        writes = hcam.temperatureWrites

        cam._lastTecReassert -= toupcamcamera.TEMPERATURE_REASSERT_INTERVAL_S + 1
        cam._maintainTecTarget(cam.get_temperature())

        assert hcam.temperatureWrites == writes
        assert hcam.options[toupcam.TOUPCAM_OPTION_TEC] == 0

    def test_stall_is_reported_once_and_cleared_when_it_cools(self):
        from imswitch.imcontrol.model.interfaces import toupcamcamera
        hcam = FakeTecHcam()
        cam = _make_cooled_camera(hcam)
        cam.set_temperature(-30)
        # Sitting at room temperature with the cooler on and a -30 target,
        # which is what the real log showed for hours on end.
        roomTemperature = min(24.6, _SHUTDOWN_C - 1.0)
        hcam.sensorTemperature = int(roomTemperature * 10)

        cam._maintainTecTarget(cam.get_temperature())
        assert cam._tecStallLogged is False  # a cool-down takes a while

        cam._tecStallSince -= toupcamcamera.TEMPERATURE_STALL_AFTER_S + 1
        cam._maintainTecTarget(cam.get_temperature())
        warnings = [w for w in cam._CameraToupcam__logger.warnings
                    if "not regulating" in w]
        assert len(warnings) == 1
        assert f"{roomTemperature:.1f}" in warnings[0]

        # ... and it is not repeated every sample.
        cam._maintainTecTarget(cam.get_temperature())
        assert len([w for w in cam._CameraToupcam__logger.warnings
                    if "not regulating" in w]) == 1

        hcam.sensorTemperature = -300
        cam._maintainTecTarget(cam.get_temperature())
        assert cam._tecStallLogged is False
        assert any("regulating again" in i
                   for i in cam._CameraToupcam__logger.infos)

    def test_a_normal_cooldown_does_not_report_a_stall(self):
        hcam = FakeTecHcam()
        cam = _make_cooled_camera(hcam)
        cam.set_temperature(-30)

        # ~2 minutes of samples on the way down, as in the real log.
        for tenths in range(int(min(24.6, _SHUTDOWN_C - 1.0) * 10), -300, -25):
            hcam.sensorTemperature = tenths
            cam._maintainTecTarget(cam.get_temperature())

        assert cam._tecStallLogged is False
        assert not [w for w in cam._CameraToupcam__logger.warnings
                    if "not regulating" in w]
