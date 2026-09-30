"""Strobed StageMap prescan (WP8): pairing, v·D placement, delay calibration,
the manager pass-throughs, and the whole strobed path in simulation.

In a strobed sweep the firmware triggers the camera, flashes the LED ``D`` µs
later and latches the stage X at every CANopen SYNC. Frame k (by the camera's
trigger index) belongs to SYNC n = k + 1 and was taken at ``x_n + v·D``.
"""

import logging
import threading
from types import SimpleNamespace

import numpy as np
import pytest

from imswitch.imcontrol.controller.controllers.StageMapController import (
    StageMapController, StageMapParams, estimateProfileShift, pairStrobeFrames,
    pickStrobeDelay, strobeAutoDelayUs, strobeFrameKeys, strobeRowScore, strobeSweepResult,
    strobeVelocity, applyStrobeSettings, strobeSettings, strobeFullScale, strobeLitRange,
    strobeWidthForLevel, strobeFrameCheck, strobeCheckHints, STROBE_PERSISTED_FIELDS)
from imswitch.imcontrol.model.managers.lasers.ESP32LEDLaserManager import ESP32LEDLaserManager
from imswitch.imcontrol.model.managers.positioners.ESP32StageManager import ESP32StageManager
from imswitch.imcontrol.model.managers.positioners.VirtualStageManager import VirtualStageManager

PERIOD_US = 33333.0


# --------------------------------------------------------------------------- #
# Pure functions
# --------------------------------------------------------------------------- #

def test_auto_delay_puts_the_flash_near_the_end_of_the_window():
    assert strobeAutoDelayUs(30000, 20) == 30000 - 20 - 1000
    assert strobeAutoDelayUs(500, 20) == 0.0            # clamped, never negative
    assert strobeAutoDelayUs(30000, 20, 1200) == 1200   # an explicit delay wins
    assert strobeAutoDelayUs(30000, 20, 0) == 0.0       # ... including zero


def test_pairing_is_relative_to_the_first_trigger_index():
    keys = np.arange(1000, 1010)        # camera counter does not start at 0
    n = np.arange(1, 11)
    x = 100.0 + 10.0 * (n - 1)
    pairs = pairStrobeFrames(keys, n, x, PERIOD_US, 0.0)
    assert [(i, fn) for i, fn, _ in pairs] == [(i, i + 1) for i in range(10)]
    assert [p[2] for p in pairs] == pytest.approx(list(x))


def test_pairing_tolerates_gaps_in_frames_and_in_reports():
    n = np.arange(1, 13)
    x = 5.0 * (n - 1)
    # frames for SYNC 3 and 8 never arrived; SYNC 12 has a frame but no report
    frameN = [1, 2, 4, 5, 6, 7, 9, 10, 11, 12]
    keys = np.array(frameN) + 499
    # reports for SYNC 5..6 were lost with a CAN frame, and 12 never came
    keep = ~np.isin(n, [5, 6, 12])
    pairs = pairStrobeFrames(keys, n[keep], x[keep], PERIOD_US, 0.0)
    got = {fn: xx for _, fn, xx in pairs}
    assert sorted(got) == [1, 2, 4, 5, 6, 7, 9, 10, 11]   # 12 is outside the reports
    assert got[5] == pytest.approx(20.0) and got[6] == pytest.approx(25.0)  # bridged
    assert all(got[k] == pytest.approx(5.0 * (k - 1)) for k in got)
    # frame indices point back into the input list
    assert [i for i, _, _ in pairs] == list(range(9))


@pytest.mark.parametrize("direction", [1.0, -1.0])
def test_positions_are_corrected_by_v_times_delay_in_both_directions(direction):
    speed = 20000.0                     # µm/s
    delayUs = 5000.0
    n = np.arange(1, 21)
    x = 1000.0 + direction * speed * (n - 1) * PERIOD_US * 1e-6
    v = strobeVelocity(n, x, PERIOD_US)
    assert v == pytest.approx(np.full(n.size, direction * speed))
    pairs = pairStrobeFrames(n + 7, n, x, PERIOD_US, delayUs)
    shift = direction * speed * delayUs * 1e-6    # 100 µm ahead in travel direction
    assert [p[2] for p in pairs] == pytest.approx(list(x + shift))


def test_velocity_is_taken_across_report_gaps():
    n = np.array([1, 2, 3, 7, 8, 9])
    x = 3.0 * (n - 1)
    assert strobeVelocity(n, x, 1e6) == pytest.approx(np.full(6, 3.0))
    assert strobeVelocity([4], [1.0], PERIOD_US) == pytest.approx([0.0])


def test_frame_keys_fall_back_when_the_camera_reports_no_trigger_index():
    keys, source = strobeFrameKeys([7, 8, 10], [1, 2, 3])
    assert source == "trigger" and list(keys) == [7, 8, 10]
    keys, source = strobeFrameKeys([0, 0, 0], [41, 42, 44])
    assert source == "frame" and list(keys) == [41, 42, 44]
    keys, source = strobeFrameKeys([0, 0, 0], [5, 5, 5])
    assert source == "arrival" and list(keys) == [0, 1, 2]


def test_sweep_events_are_merged_in_arrival_order():
    events = [{"type": "report", "n": [1, 2], "x": [0.0, 5.0], "x_steps": [0, 5]},
              {"type": "report", "n": [3, 4, 5], "x": [10.0, 15.0]},   # short x: cut
              {"type": "done", "frames": 5, "camera": 5}]
    n, x, done = strobeSweepResult(events)
    assert n == [1, 2, 3, 4] and x == [0.0, 5.0, 10.0, 15.0]
    assert done["frames"] == 5
    assert strobeSweepResult(events[:1])[2] is None


def rolling_shutter_frame(delayUs, rows=64, cols=16, lineUs=100.0, windowUs=10000.0,
                          widthUs=20.0, black=100.0, lit=1000.0, rng=None):
    """Row r is open from r·line to r·line + window; lit if the flash fits inside."""
    r = np.arange(rows)[:, None]
    open_ = (delayUs >= r * lineUs) & (delayUs + widthUs <= r * lineUs + windowUs)
    frame = np.where(open_, black + lit, black) * np.ones((rows, cols))
    if rng is not None:
        frame = frame + rng.normal(0, 5, frame.shape)
    return frame


def test_metric_prefers_the_bright_and_even_frame():
    rng = np.random.default_rng(0)
    dark = strobeRowScore(rolling_shutter_frame(-50000, rng=rng))
    half = strobeRowScore(rolling_shutter_frame(3200, rng=rng))    # rows 33.. dark
    full = strobeRowScore(rolling_shutter_frame(8000, rng=rng))    # every row open
    assert full["score"] > 5 * half["score"] and full["score"] > 5 * dark["score"]
    assert dark["uniformity"] > 0.9 and half["uniformity"] < 0.2
    assert full["uniformity"] > 0.95


def test_calibration_picks_a_delay_where_every_row_is_open():
    rng = np.random.default_rng(1)
    delays = np.arange(0, 10000, 250.0)
    scores = [strobeRowScore(rolling_shutter_frame(d, rng=rng))["score"] for d in delays]
    best = pickStrobeDelay(delays, scores)
    # all 64 rows are open for delays in [6300, 9980]; the middle keeps margin
    assert 6300 <= best <= 9980
    assert abs(best - (6300 + 9980) / 2) < 600
    assert pickStrobeDelay(delays, np.zeros(delays.size)) is None


# --------------------------------------------------------------------------- #
# Manager pass-throughs
# --------------------------------------------------------------------------- #

class FakeMotor:
    def __init__(self):
        self.started, self.callbacks, self.stopped = [], [], 0

    def has_strobe_sweep(self, timeout=1.0):
        return True

    def start_strobe_sweep(self, **kwargs):
        self.started.append(kwargs)

    def stop_strobe_sweep(self):
        self.stopped += 1

    def register_strobesweep_callback(self, cb):
        self.callbacks.append(cb)

    def unregister_strobesweep_callback(self, cb):
        self.callbacks.remove(cb)


def esp32_stage(motor, offsetX=0.0):
    stage = ESP32StageManager.__new__(ESP32StageManager)
    stage._motor = motor
    stage._ESP32StageManager__logger = logging.getLogger("test")
    stage.stageOffsetPositions = {"X": offsetX, "Y": 0, "Z": 0, "A": 0}
    stage._strobeAxis, stage._strobeCallbacks = "X", {}
    stage._speed = {"X": 1000, "Y": 1000, "Z": 1000, "A": 1000}
    stage.limitXenabled, stage.minX, stage.maxX = False, -np.inf, np.inf
    return stage


def test_esp32_stage_sweeps_in_the_user_frame():
    motor = FakeMotor()
    stage = esp32_stage(motor, offsetX=500.0)
    got = []
    assert stage.hasStrobeSweep() is True
    assert stage.registerStrobeSweepCallback(got.append) is True
    assert stage.startStrobeSweep(axis="X", target=1000.0, speed=15000, period_us=33333,
                                  laser=4, delay_us=1200, width_us=20, max_frames=0) is True
    sent = motor.started[0]
    assert sent["target"] == 1500.0 and sent["is_absolute"] is True   # device = user + offset
    assert (sent["speed"], sent["laser"]) == (15000, 4)
    assert (sent["delay_us"], sent["width_us"]) == (1200, 20)
    motor.callbacks[0]({"type": "report", "n": [1, 2], "x": [1500.0, 1600.0], "x_steps": [0, 0]})
    motor.callbacks[0]({"type": "done", "frames": 2})
    assert got[0]["x"] == [1000.0, 1100.0] and got[1]["type"] == "done"
    assert stage.unregisterStrobeSweepCallback(got.append) is True
    assert motor.callbacks == []


def test_esp32_stage_respects_the_soft_limits():
    motor = FakeMotor()
    stage = esp32_stage(motor)
    stage.limitXenabled, stage.minX, stage.maxX = True, 0.0, 5000.0
    assert stage.startStrobeSweep(axis="X", target=6000.0, speed=1000) is False
    assert motor.started == []


def test_old_uc2rest_reports_no_strobe_support():
    stage = esp32_stage(object())
    assert stage.hasStrobeSweep() is False
    assert stage.startStrobeSweep(target=10.0) is False
    assert stage.registerStrobeSweepCallback(lambda e: None) is False
    assert stage.stopStrobeSweep() is False


def esp32_laser(laser):
    manager = ESP32LEDLaserManager.__new__(ESP32LEDLaserManager)
    manager._laser = laser
    manager.channel_index = 4
    manager._ESP32LEDLaserManager__logger = logging.getLogger("test")
    return manager


def test_laser_set_strobe_passes_through_and_degrades():
    calls = []

    def set_strobe(channel, enable=True, delay_us=0, width_us=20, timeout=2.0):
        calls.append((channel, enable, delay_us, width_us))
        return {"strobe": {"supported": 1, "enabled": int(enable)}, "return": 1}

    reply = esp32_laser(SimpleNamespace(set_strobe=set_strobe)).setStrobe(True, 1200.4, 20)
    assert reply["strobe"]["supported"] == 1
    assert calls == [(4, True, 1200, 20)]
    assert esp32_laser(SimpleNamespace()).setStrobe(True) is None   # old uc2rest


# --------------------------------------------------------------------------- #
# Simulation: VirtualStageManager + a triggered fake camera + the controller
# --------------------------------------------------------------------------- #

MARGIN = 128


class FakePositioner:
    def __init__(self):
        self.position = {"X": 0.0, "Y": 0.0, "Z": 0.0, "A": 0.0}
        self.lock = threading.Lock()

    def move(self, x=None, y=None, z=None, a=None, is_absolute=False):
        with self.lock:
            for axis, v in (("X", x), ("Y", y), ("Z", z), ("A", a)):
                if v is not None:
                    self.position[axis] = v if is_absolute else self.position[axis] + v

    def get_position(self):
        with self.lock:
            return dict(self.position)


def virtual_stage():
    info = SimpleNamespace(
        axes=["X", "Y", "Z", "A"], stageOffsets={}, forPositioning=True,
        forScanning=False, resetOnClose=False,
        managerProperties={"stepsizeX": 1, "stepsizeY": 1, "stepsizeZ": 1, "stepsizeA": 1})
    comm = SimpleNamespace(sigUpdateMotorPosition=SimpleNamespace(emit=lambda *a: None))
    return VirtualStageManager(info, "VirtualStage", commChannel=comm,
                               rs232sManager={"VirtualMicroscope": SimpleNamespace(
                                   _positioner=FakePositioner())})


class TriggeredCamera:
    """Exposes one frame per simulated trigger while in external-trigger mode.

    ``scene`` mode shows the 1-D ``world`` around where the stage really was
    at the flash (latched X plus the motion during the delay). ``rolling``
    mode ignores the scene and lights the rows whose exposure contained the
    flash, as a rolling shutter would.
    """

    def __init__(self, stage, world, shape=(32, 64), trig0=1000, dropN=(), mode="scene",
                 lineUs=30.0, perUs=None, minPeriodUs=0.0):
        self.stage, self.world, self.mode, self.lineUs = stage, world, mode, lineUs
        # perUs: counts per µs of flash (None = fixed 1100); minPeriodUs: the
        # camera ignores triggers that come sooner after its last frame.
        self.perUs, self.minPeriodUs, self._lastN = perUs, minPeriodUs, None
        self.H, self.W = shape
        self.trig0, self.dropN = trig0, set(dropN)
        self.parameters = {"exposure": SimpleNamespace(value=10.0),
                           "trigger_source": SimpleNamespace(value="Continous")}
        self.pixelSizeUm = [1, 1.0, 1.0]
        self.shape = (self.W, self.H)
        self._buffer, self._lock, self.frameNum = [], threading.Lock(), 0
        self._start = 0.0
        stage.strobeTriggerListeners.append(self.onTrigger)

    def render(self, x):
        c = int(round(x)) + MARGIN
        return np.tile(self.world[c - self.W // 2:c + self.W // 2], (self.H, 1))

    def getLatestFrame(self, returnFrameNumber=False):
        frame = self.render(self.stage.getPosition()["X"])
        return (frame, self.frameNum) if returnFrameNumber else frame

    def getParameter(self, name):
        return self.parameters[name].value

    def setParameter(self, name, value):
        self.parameters[name].value = value

    def setTriggerSource(self, source):
        self.parameters["trigger_source"].value = source

    def flushBuffers(self):
        with self._lock:
            self._buffer = []

    def getChunkWithTriggerIndex(self):
        with self._lock:
            items, self._buffer = self._buffer, []
        if not items:
            return np.zeros((0,)), np.zeros((0,), int), np.zeros((0,), int)
        frames, ids, trigs = zip(*items)
        return np.array(frames), np.array(ids), np.array(trigs)

    def onTrigger(self, n, x):
        if self.parameters["trigger_source"].value != "External trigger":
            return
        req = self.stage.strobeSweepRequest
        delay = float(req["delay_us"] or 0.0)
        if self.mode == "rolling":
            windowUs = self.parameters["exposure"].value * 1000.0
            r = np.arange(self.H)[:, None]
            opens = r * self.lineUs
            lit = (delay >= opens) & (delay + req["width_us"] <= opens + windowUs)
            level = 1100 if self.perUs is None else min(100 + self.perUs * req["width_us"], 4095)
            frame = np.where(lit, level, 100) * np.ones((self.H, self.W))
        else:
            if n == 1:
                self._start = x
            dist = req["target"] - self._start
            t = (n - 1) * req["period_us"] * 1e-6 + delay * 1e-6
            frame = self.render(self._start + np.sign(dist) * min(abs(dist), req["speed"] * t))
        if n == 1:
            self._lastN = None
        if self.minPeriodUs and self._lastN is not None and \
                (n - self._lastN) * req["period_us"] < self.minPeriodUs:
            return                      # still reading out: the trigger is ignored
        self._lastN = n
        self.frameNum += 1
        if n in self.dropN:
            return                      # lost between camera and host
        with self._lock:
            self._buffer.append((frame, self.frameNum, self.trig0 + n))


class Lasers(dict):
    def getAllDeviceNames(self):
        return list(self.keys())


class StrobeLaser:
    def __init__(self, channel=4):
        self.channel_index, self.enabled, self.calls = channel, False, []

    def setStrobe(self, enable, delayUs=0, widthUs=20):
        self.calls.append((bool(enable), delayUs, widthUs))
        return {"strobe": {"supported": 1}}


def controller(stage, camera, lasers, **params):
    ctrl = StageMapController.__new__(StageMapController)
    ctrl._logger = logging.getLogger("test")
    ctrl.params = StageMapParams(**params)
    ctrl._detector, ctrl._detectorName, ctrl._stage = camera, "cam", stage
    ctrl._master = SimpleNamespace(
        detectorsManager=SimpleNamespace(startAcquisition=lambda: 1,
                                         stopAcquisition=lambda handle: None),
        lasersManager=Lasers(lasers))
    ctrl._tiles, ctrl._tilesLock, ctrl._nextTileId = [], threading.Lock(), 0
    ctrl._sessionPath, ctrl._acqHandle, ctrl._isRunning = None, None, False
    ctrl._shouldStop, ctrl._snapRequested = threading.Event(), threading.Event()
    ctrl._monitorThread = ctrl._prescanThread = None
    ctrl._prescanLagS, ctrl._strobeCalibrating, ctrl._lastError = 0.0, False, ""
    return ctrl


def world(rng, spanPx):
    x = np.arange(-MARGIN, spanPx + MARGIN, dtype=float)
    w = np.full_like(x, 200.0)
    for c in rng.uniform(0, spanPx, 30):
        w -= 120 * np.exp(-((x - c) ** 2) / (2 * rng.uniform(8, 40) ** 2))
    return np.clip(w, 20, 255).astype(np.uint16)


def test_strobed_prescan_fills_the_strip_where_the_frames_were_taken():
    rng = np.random.default_rng(5)
    span, speed, delayUs = 2000.0, 20000.0, 1500.0
    stage = virtual_stage()
    w = world(rng, int(span))
    camera = TriggeredCamera(stage, w, dropN={5, 17})
    laser = StrobeLaser()
    # v·D = 30 µm = 7.5 strip px: uncorrected strips would miss by that much,
    # in opposite directions on the two lines.
    ctrl = controller(stage, camera, {"LED": laser}, prescanStrobe=True,
                      strobeWindowMs=2.0, strobeDelayUs=delayUs)
    strips = []
    realAdd = ctrl._addStrip
    ctrl._addStrip = lambda strip, *a: (strips.append((strip, a)), realAdd(strip, *a))
    lagCalls = []
    ctrl._calibratePrescanLag = lambda *a: lagCalls.append(a)

    ctrl._prescanLoop(0.0, span, 0.0, 100.0, 100.0, speed, None, 0.0, 4)

    assert len(strips) == 2 and lagCalls == []
    assert stage.strobeSweepRequest["laser"] == 4
    assert stage.strobeSweepRequest["delay_us"] == delayUs
    assert stage.strobeSweepRequest["width_us"] == 20.0
    for strip, (minX, maxX, y, scale) in strips:
        assert (minX, maxX, scale) == (0.0, span, 4.0)
        assert strip.shape == (8, 500)
        assert (strip.max(axis=0) > 0).mean() > 0.9
        cols = np.round(np.linspace(0, span, strip.shape[1])).astype(int) + MARGIN
        truth = np.tile(w[cols], (strip.shape[0], 1))
        assert abs(estimateProfileShift(strip, truth, maxShiftPx=40)) <= 1
    # everything put back: camera free-running at its exposure, strobe off
    assert camera.parameters["trigger_source"].value == "Continous"
    assert camera.parameters["exposure"].value == 10.0
    assert laser.calls[-1][0] is False
    assert len(ctrl._tiles) == 2 and ctrl._tiles[0]["kind"] == "prescan"


def test_strobe_mode_falls_back_to_the_free_running_prescan():
    stage = virtual_stage()
    camera = TriggeredCamera(stage, world(np.random.default_rng(0), 100))
    laser = StrobeLaser()
    # off (the default): nothing is even asked
    ctrl = controller(stage, camera, {"LED": laser})
    ctrl._strobeSetup = lambda: pytest.fail("capability probed although strobe is off")
    assert ctrl._enterStrobeMode() is None
    # on, but no laser can strobe
    ctrl = controller(stage, camera, {"LED": SimpleNamespace(channel_index=0)},
                      prescanStrobe=True)
    assert ctrl._enterStrobeMode() is None
    # on, but the camera cannot report trigger indices
    plain = SimpleNamespace(getLatestFrame=lambda: np.zeros((4, 4)))
    ctrl = controller(stage, plain, {"LED": laser}, prescanStrobe=True)
    assert ctrl._strobeSetup()[0] is None and ctrl._enterStrobeMode() is None
    # on, but the firmware has no strobe sweep
    stage.hasStrobeSweep = lambda: False
    ctrl = controller(stage, camera, {"LED": laser}, prescanStrobe=True)
    laserFound, reason = ctrl._strobeSetup()
    assert laserFound is None and "strobesweep" in reason
    assert ctrl._enterStrobeMode() is None
    # the camera was never touched on the way out
    assert camera.parameters["trigger_source"].value == "Continous"
    assert camera.parameters["exposure"].value == 10.0



def test_old_laser_node_behind_a_new_master_falls_back():
    """New master firmware, old illumination-node firmware: the strobe-off probe
    answers supported=0 (or nothing), so the free-running prescan runs."""
    stage = virtual_stage()
    camera = TriggeredCamera(stage, world(np.random.default_rng(0), 100))

    class OldNodeLaser(StrobeLaser):
        def setStrobe(self, enable, delayUs=0, widthUs=20):
            self.calls.append((bool(enable), delayUs, widthUs))
            return {"strobe": {"supported": 0}, "return": 0,
                    "error": "laser node has no strobe support (firmware too old)"}

    old = OldNodeLaser()
    ctrl = controller(stage, camera, {"LED": old}, prescanStrobe=True)
    laserFound, reason = ctrl._strobeSetup()
    assert laserFound is None and "firmware too old" in reason
    assert old.calls == [(False, 0, 20)]           # the probe only ever switches the strobe off
    assert ctrl._enterStrobeMode() is None

    class SilentLaser(StrobeLaser):
        def setStrobe(self, enable, delayUs=0, widthUs=20):
            return None                              # old uc2rest or no answer
    ctrl = controller(stage, camera, {"LED": SilentLaser()}, prescanStrobe=True)
    assert ctrl._strobeSetup()[0] is None
    assert camera.parameters["trigger_source"].value == "Continous"


def test_delay_calibration_in_simulation_stores_the_best_delay():
    stage = virtual_stage()
    camera = TriggeredCamera(stage, world(np.random.default_rng(0), 100), mode="rolling")
    laser = StrobeLaser()
    ctrl = controller(stage, camera, {"LED": laser}, strobeWindowMs=2.0)
    result = ctrl.calibrateStageMapStrobeDelay(delayMinUs=0, delayMaxUs=-1,
                                               delayStepUs=100, framesPerDelay=2)
    assert result["success"], result
    assert len(result["table"]) == 20 and all(r["frames"] == 2 for r in result["table"])
    # rows open together for delays in [31 * 30, 2000 - 20] µs
    assert 930 <= result["bestDelayUs"] <= 1980
    assert ctrl.params.strobeDelayUs == result["bestDelayUs"]
    assert stage.strobeSweepRequest["max_frames"] == 2          # no motion, N frames
    assert stage.getPosition()["X"] == pytest.approx(0.0)
    assert camera.parameters["trigger_source"].value == "Continous"
    assert camera.parameters["exposure"].value == 10.0
    assert laser.calls[-1][0] is False


def test_stopping_mid_line_switches_the_strobe_off_and_restores_the_camera():
    stage = virtual_stage()
    camera = TriggeredCamera(stage, world(np.random.default_rng(2), 2000))
    laser = StrobeLaser()
    ctrl = controller(stage, camera, {"LED": laser}, prescanStrobe=True, strobeWindowMs=2.0)
    run = threading.Thread(target=ctrl._prescanLoop,
                           args=(0.0, 2000.0, 0.0, 1000.0, 100.0, 1000.0, None, 0.0, 4))
    run.start()
    deadline = threading.Event()
    while not stage.strobeSweepRequest and not deadline.wait(0.01):
        pass
    ctrl._shouldStop.set()
    run.join(timeout=10)
    assert not run.is_alive()
    assert not stage._strobeThread.is_alive()          # the sweep was stopped
    assert laser.calls and laser.calls[-1][0] is False
    assert camera.parameters["trigger_source"].value == "Continous"
    assert camera.parameters["exposure"].value == 10.0


def test_start_prescan_strobe_flag_reports_whether_strobing_runs(monkeypatch):
    """startPrescan(strobe=...) sets the param and answers up front whether the
    strobed sweep will run; the prescan thread reuses that check."""
    stage = virtual_stage()
    camera = TriggeredCamera(stage, world(np.random.default_rng(0), 100))
    laser = StrobeLaser()
    ctrl = controller(stage, camera, {"LED": laser})
    ctrl._createSession = lambda: setattr(ctrl, "_sessionPath", "/tmp/x")
    ctrl._dropTiles = lambda kind: None
    ctrl._emitStatus = lambda: None
    started = []
    monkeypatch.setattr(threading.Thread, "start", lambda self: started.append(self))

    res = ctrl.startPrescan(0, 100, 0, 0, dy=10, speedX=100, strobe=True)
    assert res["success"] and res["strobe"] is True and res["strobeReason"] == ""
    assert ctrl.params.prescanStrobe is True and ctrl._strobePreflight[0] is laser

    ctrl._prescanThread = None
    stage.hasStrobeSweep = lambda: False
    res = ctrl.startPrescan(0, 100, 0, 0, dy=10, speedX=100)       # keeps the setting
    assert res["strobe"] is False and "strobesweep" in res["strobeReason"]

    ctrl._prescanThread = None
    res = ctrl.startPrescan(0, 100, 0, 0, dy=10, speedX=100, strobe=False)
    assert "strobe" not in res and ctrl.params.prescanStrobe is False
    assert len(started) == 3



# --------------------------------------------------------------------------- #
# Strobe settings persistence and the full calibration
# --------------------------------------------------------------------------- #

def test_strobe_settings_roundtrip_and_bad_input():
    p = StageMapParams(prescanStrobe=True, strobeWidthUs=35, strobeDelayUs=21000,
                       strobeWindowMs=25, strobeMinPeriodMs=31.5)
    data = strobeSettings(p)
    assert set(data) == set(STROBE_PERSISTED_FIELDS)
    back = applyStrobeSettings(StageMapParams(), data)
    assert (back.prescanStrobe, back.strobeWidthUs, back.strobeDelayUs,
            back.strobeWindowMs, back.strobeMinPeriodMs) == (True, 35, 21000, 25, 31.5)
    # unknown keys ignored, a bad value leaves only that field at its default
    junk = applyStrobeSettings(StageMapParams(), {"strobeWidthUs": "wide", "strobeDelayUs": 900,
                                                  "minMoveFraction": 0.9, "nope": 1})
    assert junk.strobeWidthUs == 20.0 and junk.strobeDelayUs == 900
    assert junk.minMoveFraction == StageMapParams().minMoveFraction   # not a strobe field
    assert applyStrobeSettings(StageMapParams(), ["not", "a", "dict"]) == StageMapParams()


def test_settings_file_written_by_set_params_and_read_back(tmp_path):
    stage = virtual_stage()
    camera = TriggeredCamera(stage, world(np.random.default_rng(0), 100))
    ctrl = controller(stage, camera, {})
    ctrl._emitStatus = lambda: None
    ctrl._strobeSettingsFile = str(tmp_path / "config" / "stagemap_strobe.json")
    ctrl.setStageMapParams(StageMapParams(prescanStrobe=True, strobeWidthUs=42, strobeWindowMs=12))
    fresh = controller(stage, camera, {})
    fresh._strobeSettingsFile = ctrl._strobeSettingsFile
    fresh._loadStrobeSettings()
    assert fresh.params.prescanStrobe is True
    assert (fresh.params.strobeWidthUs, fresh.params.strobeWindowMs) == (42, 12)
    # a broken file is reported and ignored
    (tmp_path / "config" / "stagemap_strobe.json").write_text("{not json")
    broken = controller(stage, camera, {})
    broken._strobeSettingsFile = ctrl._strobeSettingsFile
    broken._loadStrobeSettings()
    assert broken.params == StageMapParams()


def test_full_scale_lit_range_and_width_rule():
    assert strobeFullScale(np.array([[3, 200]], np.uint8)) == 255
    assert strobeFullScale(np.array([[3, 4000]], np.uint16)) == 4095
    assert strobeFullScale(np.array([[3, 9000]], np.uint16)) == 16383
    assert strobeFullScale(np.array([[3, 60000]], np.uint16)) == 65535
    assert strobeLitRange([0, 1, 2, 3, 4], [0, 0.97, 1.0, 0.96, 0.2]) == (1.0, 3.0)
    assert strobeLitRange([0, 1], [0, 0]) is None
    # linear in width above the dark level: 20 us gives 100 counts over 100 dark
    assert strobeWidthForLevel(20, 200, 100, 4095, 0.6) == pytest.approx(200.0)      # capped x10
    assert strobeWidthForLevel(200, 1100, 100, 4095, 0.6) == pytest.approx(471.4, rel=1e-3)
    assert strobeWidthForLevel(900, 3000, 100, 4095, 0.9) == 1000.0                  # firmware limit
    assert strobeWidthForLevel(20, 100, 100, 4095, 0.6) == 20.0                      # no signal: keep


def test_frame_check_classifies_every_way_a_frame_can_miss_its_flash():
    ok = strobeFrameCheck([1100] * 5, [1.0] * 5, [11, 12, 13, 14, 15], 5, 1100, 100, 1.0)
    assert ok["ok"] and ok["missing"] == 0
    # a trigger gap (camera index 13 missing) counts even if pulses match frames
    gap = strobeFrameCheck([1100] * 4, [1.0] * 4, [11, 12, 14, 15], 4, 1100, 100, 1.0)
    assert gap["missing"] == 1 and not gap["ok"]
    mixed = strobeFrameCheck([1100, 150, 2300, 1100], [1.0, 1.0, 1.0, 0.5], [1, 2, 3, 4], 6,
                             1100, 100, 1.0)
    assert (mixed["dark"], mixed["double"], mixed["uneven"], mixed["missing"]) == (1, 1, 1, 2)
    hints = strobeCheckHints(dict(mixed, periodUs=31000), (20000, 21000), 30000, 0.5)
    text = " ".join(hints)
    assert "skipped 2 of 6" in text and "stayed dark" in text and "two flashes" in text
    assert "1.0 ms of timing margin" in text
    good = strobeCheckHints(dict(ok, periodUs=31000), (10000, 25000), 30000, 0.6)
    assert good == ["Every trigger gave one frame lit by one flash, at one frame per 31.0 ms."]
    tight = strobeCheckHints(dict(ok, periodUs=31000), (10000, 11000), 30000, 0.6)
    assert tight[0].startswith("Every trigger") and "timing margin" in tight[1]


def test_full_calibration_in_simulation_sets_delay_width_and_frame_rate(tmp_path):
    """Rolling shutter (31 rows x 30 us readout), 3 ms window, dim flash, and a
    camera that needs 6 ms between frames: the calibration finds a delay where
    all rows are lit, raises the width to 60 % of full scale, and lengthens
    the frame period until no trigger is skipped. Everything is persisted."""
    stage = virtual_stage()
    camera = TriggeredCamera(stage, world(np.random.default_rng(0), 100), mode="rolling",
                             perUs=5.0, minPeriodUs=6000.0)
    laser = StrobeLaser()
    ctrl = controller(stage, camera, {"LED": laser}, strobeWindowMs=3.0, strobeWidthUs=20)
    ctrl._emitStatus = lambda: None
    ctrl._strobeSettingsFile = str(tmp_path / "stagemap_strobe.json")
    result = ctrl.calibrateStageMapStrobe(adjustWidth=True, delayStepUs=250,
                                          framesPerDelay=2, testFrames=8)
    assert result["success"] and result["matched"], result
    lo, hi = result["litRangeUs"]
    assert 930 <= lo <= result["bestDelayUs"] <= hi
    assert result["widthFromUs"] == 20
    assert 400 <= result["widthUs"] <= 520                  # ~60 % of 4095 at 5 counts/us
    assert result["level"] == pytest.approx(0.6, abs=0.06)
    periods = [c["periodUs"] for c in result["checks"]]
    assert periods[0] == pytest.approx(4000) and result["checks"][0]["missing"] > 0
    assert result["check"]["missing"] == 0 and result["periodUs"] >= 6000
    assert ctrl.params.strobeMinPeriodMs == pytest.approx(result["periodUs"] / 1000)
    assert ctrl.params.strobeDelayUs == result["bestDelayUs"]
    assert ctrl.params.strobeWidthUs == result["widthUs"]
    assert "one frame lit by one flash" in result["hints"][0]
    # the wider flash shrank the all-rows range; the delay was re-centred inside it
    lo, hi = result["litRangeUs"]
    assert lo <= result["bestDelayUs"] <= hi and hi - lo >= 500
    assert any("timing margin" in h for h in result["hints"][1:]) or hi - lo >= 2000
    # persisted, and the camera is back as it was
    saved = __import__("json").loads((tmp_path / "stagemap_strobe.json").read_text())
    assert saved["strobeWidthUs"] == result["widthUs"]
    assert saved["strobeMinPeriodMs"] == pytest.approx(result["periodUs"] / 1000)
    assert camera.parameters["trigger_source"].value == "Continous"
    assert camera.parameters["exposure"].value == 10.0
    assert laser.calls[-1][0] is False
    # the prescan now uses the calibrated minimum period
    assert ctrl._strobeMinPeriodUs(3000) == pytest.approx(result["periodUs"])


def test_full_calibration_reports_when_nothing_lights():
    stage = virtual_stage()
    camera = TriggeredCamera(stage, world(np.random.default_rng(0), 100), mode="rolling",
                             lineUs=200.0)                # readout 6.2 ms > 3 ms window
    ctrl = controller(stage, camera, {"LED": StrobeLaser()}, strobeWindowMs=3.0)
    ctrl._emitStatus = lambda: None
    result = ctrl.calibrateStageMapStrobe(delayStepUs=500, framesPerDelay=1, testFrames=4)
    assert result["success"] is False and result["matched"] is False
    assert "every row" in result["error"] and "Lengthen the window" in result["error"]
    assert ctrl.params.strobeDelayUs == -1.0               # nothing stored
