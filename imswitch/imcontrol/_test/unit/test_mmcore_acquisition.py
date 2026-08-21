"""
Regression tests for MMCoreDetectorManager's acquisition primitives.

These cover the failure mode where a long-exposure snap job and the live-view
poll drove the same MMCore instance at the same time: the poll drained the
job's single frame out of the circular buffer (so the job waited forever) and
its fallback ``core.snap()`` blocked inside MMCore's device lock for the whole
exposure, which stalled every other API call and froze the backend.

Everything runs against a fake core, so no Micro-Manager install is needed --
unlike ``test_mmcore_managers.py``, which drives the real DemoCamera adapter.
The manager is built with ``object.__new__`` because its ``__init__`` talks to
hardware; the attributes set here are exactly the ones the tested methods use.
"""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest


pytest.importorskip("pymmcore")

from imswitch.imcontrol.model.managers.detectors.MMCoreDetectorManager import (  # noqa: E402
    MMCoreDetectorManager,
)


class FakeCore:
    """Minimal stand-in for CMMCorePlus covering the acquisition calls.

    A frame appears in the buffer ``deliverAfter`` seconds after the sequence
    is armed, which is how a real camera behaves for an exposure of that
    length.
    """

    def __init__(self, exposureMs=50.0, deliverAfter=0.1, deliver=True):
        self.exposureMs = exposureMs
        self.deliverAfter = deliverAfter
        self.deliver = deliver
        self._running = False
        self._startedAt = None
        self._buffer = []
        self.snapCalls = 0
        self.startCalls = 0

    def _tick(self):
        if (self._running and self.deliver
                and time.time() - self._startedAt >= self.deliverAfter):
            self._buffer.append(np.full((4, 4), 7, dtype=np.uint16))
            self._running = False

    def isSequenceRunning(self):
        self._tick()
        return self._running

    def getRemainingImageCount(self):
        self._tick()
        return len(self._buffer)

    def popNextImage(self):
        return self._buffer.pop(0)

    def startSequenceAcquisition(self, numImages, intervalMs, stopOnOverflow):
        self.startCalls += 1
        self._running = True
        self._startedAt = time.time()

    def startContinuousSequenceAcquisition(self, intervalMs):
        self.startCalls += 1
        self._running = True
        self._startedAt = time.time()

    def stopSequenceAcquisition(self):
        self._running = False

    def getDevicePropertyNames(self, label):
        return ["Exposure"]

    def getProperty(self, label, name):
        return str(self.exposureMs)

    def clearCircularBuffer(self):
        self._buffer.clear()

    def getExposure(self):
        return self.exposureMs

    def snap(self):
        self.snapCalls += 1
        return np.full((4, 4), 3, dtype=np.uint16)


class _SilentLogger:
    def _noop(self, *args, **kwargs):
        pass

    debug = info = warning = error = _noop


def _make_manager(core):
    mgr = object.__new__(MMCoreDetectorManager)
    mgr._core = core
    mgr._logger = _SilentLogger()
    mgr._label = "Camera"
    mgr._grabLock = threading.RLock()
    mgr._triggerMode = "continuous"
    mgr._latestFrame = None
    mgr._frameNunber = 0
    mgr._running = False
    mgr._shape = (4, 4)
    mgr._camStats = {"fps": 0.0, "delivered": 0, "last_t": None}
    mgr._exclusiveLock = threading.Lock()
    mgr._exclusiveOwner = None
    mgr._exclusiveSince = 0.0
    mgr._exclusiveRestoreLive = False
    mgr._lastIdleSnap = 0.0
    mgr._lastSnapErrorLog = 0.0
    mgr._snapErrorCount = 0
    # DetectorManager.name reads a name-mangled private attribute, which the
    # skipped __init__ would normally have set.
    mgr._DetectorManager__name = "Camera"
    return mgr


class TestAcquireSingleFrame:
    def test_returns_the_frame_without_a_blocking_snap(self):
        core = FakeCore(deliverAfter=0.1)
        mgr = _make_manager(core)

        image, status = mgr.acquireSingleFrame(timeout=5.0)

        assert status == "done"
        assert image is not None and int(image[0, 0]) == 7
        assert core.startCalls == 1
        # core.snap() holds MMCore's device lock for the whole exposure and
        # must never be used for a job-driven acquisition.
        assert core.snapCalls == 0

    def test_gives_up_instead_of_hanging_forever(self):
        core = FakeCore(deliver=False)
        mgr = _make_manager(core)

        started = time.time()
        image, status = mgr.acquireSingleFrame(timeout=0.6)

        assert status == "timeout"
        assert image is None
        assert time.time() - started < 3.0

    def test_cancel_aborts_a_long_exposure_immediately(self):
        core = FakeCore(exposureMs=30_000.0, deliverAfter=30.0)
        mgr = _make_manager(core)
        cancelEvent = threading.Event()
        timer = threading.Timer(0.2, cancelEvent.set)
        timer.start()

        started = time.time()
        image, status = mgr.acquireSingleFrame(cancelEvent=cancelEvent,
                                               timeout=60.0)
        timer.cancel()

        assert status == "cancelled"
        assert image is None
        # The point of cancelling: it returns now, not in 30 s.
        assert time.time() - started < 3.0

    def test_abort_from_another_thread_unblocks_the_acquisition(self):
        core = FakeCore(exposureMs=30_000.0, deliverAfter=30.0)
        mgr = _make_manager(core)
        result = {}

        worker = threading.Thread(
            target=lambda: result.update(
                r=mgr.acquireSingleFrame(timeout=60.0)),
        )
        worker.start()
        time.sleep(0.2)
        # abortAcquisition must not need _grabLock -- the acquiring thread
        # holds it for the whole exposure.
        mgr.abortAcquisition()
        worker.join(5.0)

        assert not worker.is_alive(), "abortAcquisition did not unblock the grab"
        assert result["r"][1] == "error"

    def test_snapsync_uses_the_polling_path(self):
        core = FakeCore(deliverAfter=0.1)
        mgr = _make_manager(core)

        frame = mgr.snapSync()

        assert int(frame[0, 0]) == 7
        assert core.snapCalls == 0


class TestExclusiveAccess:
    def test_claim_is_exclusive(self):
        mgr = _make_manager(FakeCore())

        assert mgr.acquireExclusive("job") is True
        assert mgr.acquireExclusive("other") is False
        assert mgr.exclusiveOwner == "job"

        mgr.releaseExclusive("job")
        assert mgr.exclusiveOwner is None
        assert mgr.acquireExclusive("other") is True

    def test_live_poll_never_touches_the_sensor_while_claimed(self):
        core = FakeCore()
        mgr = _make_manager(core)
        mgr._store_latest(np.full((4, 4), 42, dtype=np.uint16))
        mgr.acquireExclusive("job")

        frame = None
        for _ in range(50):
            frame = mgr.getLatestFrame()

        # No snap, no sequence: the poll replays the cache instead of racing
        # the job for the frame it is waiting for.
        assert core.snapCalls == 0
        assert core.startCalls == 0
        assert int(frame[0, 0]) == 42

    def test_start_acquisition_is_a_noop_while_claimed(self):
        core = FakeCore()
        mgr = _make_manager(core)
        mgr.acquireExclusive("job")

        mgr.startAcquisition()

        assert core.isSequenceRunning() is False

    def test_release_restores_a_live_sequence_it_stopped(self):
        core = FakeCore()
        mgr = _make_manager(core)
        core.startSequenceAcquisition(0, 0, True)

        mgr.acquireExclusive("job")
        assert core.isSequenceRunning() is False   # stopped for the job

        mgr.releaseExclusive("job")
        assert core.isSequenceRunning() is True    # and put back afterwards

    def test_context_manager_releases_on_error(self):
        mgr = _make_manager(FakeCore())

        with pytest.raises(ValueError):
            with mgr.exclusiveAccess("job"):
                raise ValueError("boom")

        # A worker dying mid-job must not leave the camera claimed forever.
        assert mgr.exclusiveOwner is None

    def test_context_manager_refuses_a_busy_camera(self):
        mgr = _make_manager(FakeCore())
        mgr.acquireExclusive("job")

        with pytest.raises(RuntimeError, match="busy"):
            with mgr.exclusiveAccess("other"):
                pass


class TestIdleSnapGuards:
    def test_idle_snap_is_throttled_to_one_per_exposure(self):
        core = FakeCore(exposureMs=100.0)
        mgr = _make_manager(core)

        for _ in range(50):
            mgr.getLatestFrame()

        assert core.snapCalls == 1

    def test_long_exposures_are_never_snapped_from_the_poll(self):
        # A blocking snap of this length would stall every other MMCore call
        # in the process for a minute.
        core = FakeCore(exposureMs=60_000.0)
        mgr = _make_manager(core)

        for _ in range(50):
            mgr.getLatestFrame()

        assert core.snapCalls == 0

    def test_failing_snaps_are_logged_at_most_once_per_interval(self):
        core = FakeCore(exposureMs=1.0)
        mgr = _make_manager(core)
        logged = []
        attempts = []

        def boom():
            attempts.append(1)
            raise RuntimeError('Error in device "Andor": (20073)')

        core.snap = boom
        mgr._logger.warning = lambda msg, *a, **k: logged.append(msg)

        for _ in range(200):
            mgr.getLatestFrame()
            time.sleep(0.001)

        # Many failures, one log line: unthrottled this produced a full
        # traceback per poll and buried every other message in the console.
        assert len(attempts) > 5
        assert len(logged) == 1
        assert "20073" in logged[0]
