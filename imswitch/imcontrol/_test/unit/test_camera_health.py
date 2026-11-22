"""CAM-LOST: a stand-in camera and a stalled live stream are reported; frame-taking work refuses a stand-in."""
import time
import types

from imswitch.imcontrol.controller.busy import busy_reasons, camera_reasons
from imswitch.imcontrol.controller.controllers.LiveViewController import LiveViewController


class _Detectors(dict):
    def getAllDeviceNames(self):
        return list(self)


def _camera(placeholder=False, trigger="Continuous (Free Run)", acquisition=True):
    return types.SimpleNamespace(isPlaceholder=placeholder, forAcquisition=acquisition, parameters={},
                                 getCurrentTriggerType=lambda: trigger)


def _worker(age_s, failed=False):
    return types.SimpleNamespace(lastNewFrameAt=time.monotonic() - age_s, stoppedOnFailure=failed)


def _health(detectors, streams):
    ctl = types.SimpleNamespace(_master=types.SimpleNamespace(detectorsManager=_Detectors(detectors)),
                                _activeStreams=streams)
    fn = LiveViewController.getCameraHealth
    return getattr(fn, "__wrapped__", fn)(ctl)


def test_stand_in_camera_and_stalled_stream_are_reported():
    h = _health({"Main": _camera(placeholder=True), "Side": _camera(), "Idle": _camera()},
                {"Side": ("jpeg", _worker(20))})
    assert h["Main"]["placeholder"] and "did not open" in h["Main"]["message"]
    assert h["Side"]["stalled"] and h["Side"]["secondsSinceNewFrame"] >= 20
    assert not h["Idle"]["stalled"] and h["Idle"]["message"] == ""   # no stream, nothing to judge


def test_fresh_frames_and_triggered_cameras_are_not_stalled():
    h = _health({"A": _camera(), "B": _camera(trigger="Software Trigger"), "C": _camera()},
                {"A": ("jpeg", _worker(0.5)), "B": ("jpeg", _worker(60)), "C": ("jpeg", _worker(0.5, failed=True))})
    assert not h["A"]["stalled"]
    assert not h["B"]["stalled"]   # waits for triggers on purpose
    assert h["C"]["stalled"]       # the stream gave up after failed grabs


def test_frame_taking_work_refuses_a_stand_in_camera():
    master = types.SimpleNamespace(detectorsManager=_Detectors(
        {"Main": _camera(placeholder=True), "Overview": _camera(placeholder=True, acquisition=False)}))
    assert len(camera_reasons(master)) == 1 and "Main" in camera_reasons(master)[0]
    assert busy_reasons(master) == []                 # non-camera work still runs
    assert busy_reasons(master, camera=True) == camera_reasons(master)
