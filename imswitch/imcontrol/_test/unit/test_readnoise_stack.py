#!/usr/bin/env python3
"""Dark-stack capture in ReadNoiseCalibrationController.

Two things silently ruin a dark-frame calibration:
* a fixed per-frame timeout — above ~5 s exposure the loop gave up waiting and
  stored N copies of the same stale frame,
* the frame that was already integrating when the exposure changed, which still
  carries the old settings.

Runs standalone: `.venv/bin/python <this file>`.
"""

import os
import sys
import threading
import types

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../../.."))

from imswitch.imcontrol.controller.controllers.ReadNoiseCalibrationController import (  # noqa: E402
    ReadNoiseCalibrationController as RNC,
    _AcquisitionAborted,
)


class _Ctrl(RNC):
    """The capture loop only touches _progress and _abort - skip the real init."""

    def __init__(self):
        self._progress = {}
        self._abort = threading.Event()


class _Detector:
    """Emits a new, uniquely valued frame on every poll."""

    def __init__(self, exposure_ms=1.0):
        self.parameters = {"exposure": types.SimpleNamespace(value=exposure_ms)}
        self.frameNumber = 0
        self.flushed = False

    def getParameter(self, name):
        return self.parameters[name].value

    def flushBuffer(self):
        self.flushed = True

    def getLatestFrame(self, returnFrameNumber=False):
        self.frameNumber += 1
        frame = np.full((2, 2), self.frameNumber, dtype=np.uint16)
        return (frame, self.frameNumber) if returnFrameNumber else frame


def test_timeout_follows_exposure():
    ctrl = _Ctrl()
    assert ctrl._exposure_seconds(_Detector(10_000)) == 10.0
    # 5 s flat would have been shorter than a single exposure
    assert ctrl._frame_timeout(_Detector(10_000)) > 10.0
    assert ctrl._frame_timeout(_Detector(1_000_000)) > 1000.0
    assert ctrl._frame_timeout(_Detector(0.1)) >= 5.0


def test_no_exposure_reported_still_waits():
    detector = types.SimpleNamespace(parameters={})
    assert _Ctrl()._exposure_seconds(detector) == 0.0
    assert _Ctrl()._frame_timeout(detector) == 5.0


def test_warmup_frame_is_dropped():
    ctrl = _Ctrl()
    detector = _Detector()
    stack = ctrl._grab_stack(detector, 3, "dark")
    assert stack.shape == (3, 2, 2)
    assert detector.flushed
    # frame 1 is the warm-up; the kept frames are the 3 after it, all distinct
    assert [int(f[0, 0]) for f in stack] == [2, 3, 4]

    plain = _Ctrl()._grab_stack(_Detector(), 2, "dark", warmup=0)
    assert [int(f[0, 0]) for f in plain] == [1, 2]


def test_abort_raises():
    ctrl = _Ctrl()
    ctrl._abort.set()
    try:
        ctrl._grab_stack(_Detector(), 2, "dark")
    except _AcquisitionAborted:
        pass
    else:
        raise AssertionError("abort was ignored")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all good")
