"""
Tests for keeping a TEC-cooled Toupcam's stream running between acquisitions.

Toupcam_Stop() cuts the TEC drive (measured: 5.8 V -> 0 V, sensor -30 -> +6 °C
within two minutes of snaps), so on TEC models stop_live() parks the stream in
software-trigger mode instead of stopping it. These tests run against a fake
SDK handle and check that no Stop() happens in normal use, that frames exposed
before a resume are dropped, and that models without a TEC behave as before.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

toupcam = pytest.importorskip(
    "imswitch.imcontrol.model.interfaces.toupcamsdk.toupcam")

from imswitch.imcontrol.model.interfaces.toupcamcamera import (  # noqa: E402
    PARK_TRIGGER_MODE,
)
from imswitch.imcontrol._test.unit.test_toupcam_lownoise import (  # noqa: E402
    FakeHcam,
    _make_camera,
)


class FakeStreamHcam(FakeHcam):
    """FakeHcam plus the stream calls, counted."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.options.setdefault(toupcam.TOUPCAM_OPTION_TRIGGER, 0)
        self.options.setdefault(toupcam.TOUPCAM_OPTION_BITDEPTH, 1)
        self.options.setdefault(toupcam.TOUPCAM_OPTION_TEC_VOLTAGE, 57)
        self.options.setdefault(toupcam.TOUPCAM_OPTION_TEC_VOLTAGE_MAX, 61)
        self.starts = 0
        self.stops = 0
        self.running = False

    def StartPullModeWithCallback(self, callback, ctx):
        assert not self.running, "StartPullMode on a running stream"
        self.starts += 1
        self.running = True

    def Stop(self):
        self.stops += 1
        self.running = False

    def get_Roi(self):
        return (0, 0, 2992, 3000)

    def put_Roi(self, *args):
        pass

    def MaxBitDepth(self):
        return 14

    def trigger_mode(self):
        return self.options[toupcam.TOUPCAM_OPTION_TRIGGER]


def _make_tec_camera(tec=True, swTrigger=True):
    hcam = FakeStreamHcam()
    cam = _make_camera(hcam)
    cam._hasTEC = tec
    cam._hasSoftwareTrigger = swTrigger
    cam.trigger_source = "Continous"
    cam.frameNumber = -1
    cam.frame_buffer = []
    cam.frameid_buffer = []
    cam.lastFrameFromBuffer = None
    cam.lastFrameId = -1
    cam._streamStats = None
    cam._newStreamStats = lambda: None
    cam._startTemperatureMonitor = lambda: None
    cam.exposure_time = 5000.0
    cam.SensorWidth, cam.SensorHeight = 2992, 3000
    return cam, hcam


class TestTecStreamIsParkedNotStopped:
    def test_idle_tec_camera_runs_parked(self):
        cam, hcam = _make_tec_camera()

        cam._startParked()

        assert hcam.running and hcam.starts == 1
        assert hcam.trigger_mode() == PARK_TRIGGER_MODE
        assert cam._parked and not cam.is_streaming

    def test_snap_cycles_never_stop_the_stream(self):
        cam, hcam = _make_tec_camera()
        cam._startParked()

        for _ in range(5):
            cam.start_live()
            assert cam.is_streaming and not cam._parked
            assert hcam.trigger_mode() == 0  # free run while acquiring
            cam.stop_live()
            assert cam._parked and hcam.trigger_mode() == PARK_TRIGGER_MODE

        assert hcam.stops == 0
        assert hcam.starts == 1

    def test_requested_trigger_mode_is_restored_on_resume(self):
        cam, hcam = _make_tec_camera()
        cam._startParked()

        cam.setTriggerSource("Software Trigger")
        # Still parked: the requested mode waits for the next start.
        assert hcam.trigger_mode() == PARK_TRIGGER_MODE
        cam.setTriggerSource("External Trigger")
        cam.start_live()

        assert hcam.trigger_mode() == 2
        assert hcam.stops == 0

    def test_trigger_change_while_acquiring_does_not_stop(self):
        cam, hcam = _make_tec_camera()
        cam._startParked()
        cam.start_live()

        cam.setTriggerSource("Software Trigger")

        assert cam.is_streaming and hcam.trigger_mode() == 1
        assert hcam.stops == 0

    def test_close_really_stops(self):
        cam, hcam = _make_tec_camera()
        cam._startParked()

        cam.close()

        assert hcam.stops == 1 and not hcam.running


def _wire_pull(cam, hcam):
    """Let _onImageEvent pull 4x4 frames from the fake handle."""
    info = SimpleNamespace(width=4, height=4, seq=0, timestamp=0)

    def pull(buf, still, bits, pitch, frameInfo):
        hcam.pulls += 1
        info.seq += 1

    hcam.pulls = 0
    hcam.PullImageV3 = pull
    cam._pull_lock = threading.Lock()
    cam._raw_buf = bytes(4 * 4 * 2)
    cam._frame_info = info
    cam._npDtype = np.uint16
    cam.flipImage = (False, False)
    cam._updateStreamStats = lambda t: None


class TestFramesAroundResume:
    def test_frames_while_parked_are_pulled_and_dropped(self):
        cam, hcam = _make_tec_camera()
        _wire_pull(cam, hcam)
        cam._startParked()

        cam._onImageEvent()

        assert hcam.pulls == 1  # taken off the SDK queue ...
        assert len(cam.frame_buffer) == 0  # ... but not handed out

    def test_frame_before_the_gate_is_dropped_after_it_kept(self):
        cam, hcam = _make_tec_camera()
        _wire_pull(cam, hcam)
        cam._startParked()
        cam.start_live()

        cam._onImageEvent()  # arrives right after the resume: stale
        assert len(cam.frame_buffer) == 0

        cam._acceptFramesAfter = time.time() - 1.0
        cam._onImageEvent()
        assert len(cam.frame_buffer) == 1

    def test_resume_drops_frames_exposed_before_it(self):
        cam, hcam = _make_tec_camera()
        cam._startParked()
        cam.exposure_time = 5000.0

        t0 = time.time()
        cam.start_live()

        # A frame needs a full 5 s exposure after the resume; anything earlier
        # was exposed (at least partly) while parked.
        assert cam._acceptFramesAfter >= t0 + 4.5
        assert cam._acceptFramesAfter <= time.time() + 5.0

    def test_short_exposure_gate_stays_short(self):
        cam, hcam = _make_tec_camera()
        cam._startParked()
        cam.exposure_time = 100.0

        t0 = time.time()
        cam.start_live()

        assert cam._acceptFramesAfter <= t0 + 0.1 + 0.05


class TestSettingsThatNeedAStoppedStream:
    def test_unchanged_value_does_not_restart(self):
        cam, hcam = _make_tec_camera()
        hcam.options[toupcam.TOUPCAM_OPTION_CG] = 1
        hcam.options[toupcam.TOUPCAM_OPTION_LOW_NOISE] = 1
        cam._bits = 16
        cam._startParked()

        cam.set_conversion_gain("HCG")
        cam.set_low_noise(True)
        cam.set_pixel_format("mono16")
        cam.setROI(0, 0, 0, 0)

        assert hcam.stops == 0
        assert cam.conversionGain == "HCG" and cam.lowNoise is True

    def test_changed_value_restarts_once_and_parks_again(self):
        cam, hcam = _make_tec_camera()
        hcam.options[toupcam.TOUPCAM_OPTION_CG] = 0
        cam._startParked()

        cam.set_conversion_gain("HCG")

        assert hcam.options[toupcam.TOUPCAM_OPTION_CG] == 1
        assert hcam.stops == 1 and hcam.starts == 2
        assert hcam.running and cam._parked
        assert hcam.trigger_mode() == PARK_TRIGGER_MODE

    def test_changed_value_while_acquiring_resumes_acquiring(self):
        cam, hcam = _make_tec_camera()
        hcam.options[toupcam.TOUPCAM_OPTION_LOW_NOISE] = 0
        cam._startParked()
        cam.start_live()

        cam.set_low_noise(True)

        assert cam.is_streaming and not cam._parked
        assert hcam.trigger_mode() == 0
        assert hcam.stops == 1


class TestCamerasWithoutTec:
    def test_stop_live_still_stops(self):
        cam, hcam = _make_tec_camera(tec=False)

        cam._startParked()  # no-op without a TEC
        assert hcam.starts == 0
        cam.start_live()
        cam.stop_live()

        assert hcam.starts == 1 and hcam.stops == 1
        assert not cam._parked

    def test_no_software_trigger_means_no_parking(self):
        cam, hcam = _make_tec_camera(swTrigger=False)

        cam.start_live()
        cam.stop_live()

        assert hcam.stops == 1


class TestTecVoltage:
    def test_voltage_is_reported_in_volts(self):
        cam, hcam = _make_tec_camera()

        assert cam.get_tec_voltage() == (5.7, 6.1)

    def test_no_tec_no_voltage(self):
        cam, hcam = _make_tec_camera(tec=False)

        assert cam.get_tec_voltage() == (None, None)
