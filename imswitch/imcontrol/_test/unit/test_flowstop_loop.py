"""FlowStop acquisition loop: it must acquire *every* requested frame and stop on request.

Regression guard for the stray `break` that ended the loop after a single image.
"""
import json
import os
from threading import Event, Thread

import numpy as np
import pytest

from imswitch.imcommon.model import dirtools
from imswitch.imcontrol.controller.controllers.FlowStopController import FlowStopController


class _FakeSignal:
    def __init__(self):
        self.calls = []

    def emit(self, *args):
        self.calls.append(args)


class _FakePositioner:
    def __init__(self):
        self.moves = []

    def move(self, value=0, axis="X", is_absolute=False, is_blocking=True, **kwargs):
        self.moves.append((value, axis))

    def stopAll(self):
        pass


class _FakeDetector:
    def getLatestFrame(self):
        return np.zeros((4, 4), dtype=np.uint8)


def _makeController(tmp_path, monkeypatch):
    monkeypatch.setattr(dirtools.UserFileDirs, "getValidatedDataPath",
                        classmethod(lambda cls: str(tmp_path)))
    monkeypatch.setattr(dirtools, "getDiskusage", lambda: 0.1)

    c = object.__new__(FlowStopController)
    c._logger = type("L", (), {"debug": lambda *a: None, "error": lambda *a: None,
                               "warning": lambda *a: None})()
    c._commChannel = type("C", (), {"sigStartLiveAcquistion": _FakeSignal()})()
    c.sigImagesTaken = _FakeSignal()
    c.sigIsRunning = _FakeSignal()
    c.positioner = _FakePositioner()
    c.detectorFlowCam = _FakeDetector()
    c.pumpAxis = "X"
    c.mMetadata = {"sample_project": "unit-test"}
    c.mExperimentParameters = {"numImages": -1}
    c.imagesTaken = 0
    c.is_measure = False
    c.isRecordVideo = False
    c.video_safe = None
    c._stopEvent = Event()
    c._startTime = 0.0
    c._relativePath = ""
    c._lastError = ""
    return c


BASE_PARAMS = {
    "timeStamp": "2026_01_01-00-00-00",
    "experimentName": "unit",
    "experimentDescription": "",
    "uniqueId": 42,
    "volumePerImage": 10,
    "timeToStabilize": 0.0,
    "pumpSpeed": 1000,
    "pumpTimeout": 1,
    "frameRate": 0,
    "fileFormat": "JPG",
    "isRecordVideo": False,
    "delayToStart": 0,
}


def test_acquires_every_requested_frame(tmp_path, monkeypatch):
    c = _makeController(tmp_path, monkeypatch)
    c.flowExperimentThread({**BASE_PARAMS, "numImages": 5})

    outDir = tmp_path / "recordings" / BASE_PARAMS["timeStamp"]
    jpgs = sorted(p.name for p in outDir.glob("*.jpg"))
    assert len(jpgs) == 5, jpgs
    assert c.imagesTaken == 5
    assert len(c.positioner.moves) == 5
    assert c.is_measure is False
    assert c.sigIsRunning.calls[-1] == (False,)


def test_writes_ecotaxa_metadata_file(tmp_path, monkeypatch):
    c = _makeController(tmp_path, monkeypatch)
    c.flowExperimentThread({**BASE_PARAMS, "numImages": 1})

    meta = json.loads((tmp_path / "recordings" / BASE_PARAMS["timeStamp"] / "metadata.json").read_text())
    assert meta["sample_project"] == "unit-test"
    assert meta["acq_id"] == 42
    assert meta["acq_nb_frame"] == 1


def test_unlimited_run_stops_on_request(tmp_path, monkeypatch):
    c = _makeController(tmp_path, monkeypatch)
    worker = Thread(target=c.flowExperimentThread, args=({**BASE_PARAMS, "numImages": -1},))
    worker.start()
    # let it take a few frames, then ask it to stop
    for _ in range(200):
        if c.imagesTaken >= 3:
            break
        Event().wait(0.01)
    c.stopFlowStopExperiment()
    worker.join(timeout=5)

    assert not worker.is_alive(), "loop did not honour the stop request"
    assert c.imagesTaken >= 3
