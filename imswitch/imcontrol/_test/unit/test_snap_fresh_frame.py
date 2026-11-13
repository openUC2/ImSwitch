"""
Tests for the snap-on-demand arming logic in RecordingController:
``_armDetectorsForSnap`` / ``_waitForFreshFrame`` must cost exactly one
exposure, not two.

The fake detector below behaves like the Toupcam driver: the stream is
stopped between snaps, ``getLatestFrame`` blocks until the first frame of the
new stream arrives, and frame numbers restart at 1 on every stream start.
Before the pre-arm baseline was introduced, the baseline was taken from that
first (already fresh) frame and a *second* full exposure was waited out --
an hour for a 30-minute exposure.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from imswitch.imcontrol.controller.controllers.RecordingController import (
    RecordingController,
)


class FakeDetector:
    """Stream-restarting detector with a blocking first grab."""

    def __init__(self, exposureMs=50.0, resetCounterOnStart=True,
                 initialFrameNumber=-1):
        self.parameters = {"exposure": SimpleNamespace(value=exposureMs)}
        self._running = False
        self._frameNumber = initialFrameNumber
        self._resetCounterOnStart = resetCounterOnStart
        self.exposureSeconds = exposureMs / 1000.0
        self.framesExposed = 0
        self._streamStop = threading.Event()

    def getFrameNumber(self):
        return self._frameNumber

    def startAcquisition(self):
        self._running = True
        self._streamStop.clear()
        if self._resetCounterOnStart:
            self._frameNumber = -1

    def stopAcquisition(self):
        self._running = False
        self._streamStop.set()

    def getLatestFrame(self, returnFrameNumber=False):
        # One exposure per call once running: block for the integration, then
        # deliver the next frame of this stream (numbered from 1).
        if not self._running:
            return (None, None) if returnFrameNumber else None
        if self._streamStop.wait(self.exposureSeconds):
            return (None, None) if returnFrameNumber else None
        self._frameNumber = 1 if self._frameNumber < 0 else self._frameNumber + 1
        self.framesExposed += 1
        frame = object()
        return (frame, self._frameNumber) if returnFrameNumber else frame


def _make_controller(detector):
    ctrl = object.__new__(RecordingController)
    ctrl._RecordingController__logger = SimpleNamespace(
        debug=lambda *a, **k: None, info=lambda *a, **k: None,
        warning=lambda *a, **k: None, error=lambda *a, **k: None)
    ctrl._master = SimpleNamespace(detectorsManager={"cam": detector})
    return ctrl


class TestSnapCostsOneExposure:
    def test_first_snap_after_startup_waits_one_exposure(self):
        det = FakeDetector()
        ctrl = _make_controller(det)

        armed = ctrl._armDetectorsForSnap(["cam"])

        assert armed == ["cam"]
        assert det.framesExposed == 1

    def test_second_snap_with_renumbered_stream_waits_one_exposure(self):
        det = FakeDetector()
        ctrl = _make_controller(det)
        # First snap leaves the counter at 1 -- the same value the next
        # stream's first frame will carry.
        ctrl._disarmDetectorsAfterSnap(ctrl._armDetectorsForSnap(["cam"]))
        assert det.getFrameNumber() == 1

        det.framesExposed = 0
        ctrl._disarmDetectorsAfterSnap(ctrl._armDetectorsForSnap(["cam"]))

        assert det.framesExposed == 1

    def test_counter_restart_without_driver_reset_is_still_one_exposure(self):
        # Driver keeps the old number until the new stream's first frame, which
        # then comes in lower: a restarted counter can only mean a fresh frame.
        det = FakeDetector(resetCounterOnStart=False, initialFrameNumber=7)
        ctrl = _make_controller(det)
        det.getLatestFrame = _renumbering_grab(det)

        ctrl._armDetectorsForSnap(["cam"])

        assert det.framesExposed == 1

    def test_stale_frame_from_nonblocking_driver_is_skipped(self):
        # A driver that returns the last frame of the previous stream right
        # away (same number as before arming) must not have it accepted.
        det = FakeDetector(resetCounterOnStart=False, initialFrameNumber=3)
        ctrl = _make_controller(det)
        served = []

        def grab(returnFrameNumber=False):
            if not served:
                served.append(3)
                return (object(), 3) if returnFrameNumber else object()
            det.framesExposed += 1
            served.append(4)
            return (object(), 4) if returnFrameNumber else object()

        det.getLatestFrame = grab
        ctrl._armDetectorsForSnap(["cam"])

        assert served == [3, 4]
        assert det.framesExposed == 1

    def test_cancel_during_exposure_returns_without_frame(self):
        det = FakeDetector(exposureMs=5000)
        ctrl = _make_controller(det)
        cancel = threading.Event()

        def cancelSoon():
            cancel.wait(0.1)
            cancel.set()
            det.stopAcquisition()

        threading.Thread(target=cancelSoon, daemon=True).start()
        armed = ctrl._armDetectorsForSnap(["cam"], cancelEvent=cancel)

        assert armed == ["cam"]
        assert det.framesExposed == 0


def _make_snap_numpy_controller(detector):
    """Controller with what snapNumpy() needs; the recording service returns
    the detector's latest frame, as RecordingService.snap does."""
    ctrl = _make_controller(detector)
    ctrl.updateRecAttrs = lambda **kwargs: None
    ctrl.getDetectorNamesToCapture = lambda: ["cam"]
    ctrl._get_detector_attrs = lambda name: {}

    def snap_numpy(detector_names, attrs):
        frames = {name: ctrl._master.detectorsManager[name].getLatestFrame()
                  for name in detector_names}
        return {name: frame for name, frame in frames.items() if frame is not None}

    ctrl._recording_service = SimpleNamespace(snap_numpy=snap_numpy)
    return ctrl


class TestSnapNumpyArmsStoppedDetectors:
    def test_stopped_detector_is_armed_and_released(self):
        det = FakeDetector()
        ctrl = _make_snap_numpy_controller(det)

        images = ctrl.snapNumpy()

        assert images.get("cam") is not None
        assert det._running is False

    def test_running_stream_is_left_running(self):
        det = FakeDetector()
        det.startAcquisition()
        ctrl = _make_snap_numpy_controller(det)

        images = ctrl.snapNumpy()

        assert images.get("cam") is not None
        assert det._running is True

    def test_missing_frame_is_a_503_not_a_key_error(self):
        det = FakeDetector()
        det._running = None  # state unknown: not armed, and no frame comes
        ctrl = _make_snap_numpy_controller(det)

        with pytest.raises(HTTPException) as exc:
            ctrl.snapNumpyToFastAPI(detectorName="cam")

        assert exc.value.status_code == 503


def _renumbering_grab(det):
    def grab(returnFrameNumber=False):
        if det._streamStop.wait(det.exposureSeconds):
            return (None, None) if returnFrameNumber else None
        det._frameNumber = 1 if det.framesExposed == 0 else det._frameNumber + 1
        det.framesExposed += 1
        return (object(), det._frameNumber) if returnFrameNumber else object()
    return grab
