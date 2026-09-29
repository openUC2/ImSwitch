"""
StepperXYStageManager against a fake i4 firmware.

The firmware answers MOVE immediately and runs the motion in the background;
BUSY reports whether it is still moving, and a new MOVE replaces a running one.
These tests pin down that the driver waits for BUSY=0 where it must.
"""

from __future__ import annotations

import time
from types import SimpleNamespace

from imswitch.imcontrol.model.managers.positioners.StepperXYStageManager import (
    StepperXYStageManager,
)

MOVE_S = 0.05  # fake move duration


class FakeFirmware:
    def __init__(self, has_busy=True, microsteps=32):
        self.has_busy = has_busy
        self.microsteps = microsteps
        self.busy_until = 0.0
        self.log = []  # (time, cmd, busy_at_that_time)

    def _busy(self):
        return time.time() < self.busy_until

    def query(self, cmd):
        self.log.append((time.time(), cmd, self._busy()))
        word = cmd.split()[0]
        if word == "MOVE":
            self.busy_until = time.time() + MOVE_S
            return "OK"
        if word == "BUSY" and self.has_busy:
            return "1" if self._busy() else "0"
        if word == "MICROSTEP" and len(cmd.split()) == 1:
            return str(self.microsteps)
        if word in ("STOP", "POWER", "PING", "V", "MICROSTEP", "ACCEL", "HOLD", "DITHER", "PHASE"):
            return "OK"
        return "ERR unknown cmd"

    def read(self, nbytes, timeout=None):
        return b""

    def cmds(self, word):
        return [entry for entry in self.log if entry[1].split()[0] == word]


def make_stage(fw, **props):
    props.setdefault("rs232device", "fw")
    props.setdefault("idleStopTimeoutS", 0)
    info = SimpleNamespace(
        managerProperties=props, axes=["X", "Y"],
        forPositioning=True, forScanning=False, resetOnClose=False,
    )
    comm = SimpleNamespace(sigUpdateMotorPosition=SimpleNamespace(emit=lambda *_: None))
    return StepperXYStageManager(
        info, "StepperXY", commChannel=comm, rs232sManager={"fw": fw},
    )


def test_blocking_move_returns_after_firmware_idle():
    fw = FakeFirmware()
    stage = make_stage(fw)
    stage.move(100, "X", is_blocking=True)
    assert not fw._busy()
    assert stage.getPosition()["X"] == 100


def test_next_move_waits_for_previous_non_blocking_move():
    fw = FakeFirmware()
    stage = make_stage(fw)
    stage.move(10, "X", is_blocking=False)
    stage.move(10, "Y", is_blocking=False)
    first, second = fw.cmds("MOVE")
    assert not second[2], "second MOVE sent while the first was still running"


def test_backlash_overshoot_finishes_before_return_move():
    fw = FakeFirmware()
    stage = make_stage(fw, backlashX=5)
    stage.move(10, "X")
    stage.move(-10, "X", is_blocking=False)  # reversal -> overshoot + return
    moves = fw.cmds("MOVE")
    assert [m[1] for m in moves][1:] == ["MOVE -15 0 800", "MOVE 5 0 800"]
    assert not moves[2][2]


def test_old_firmware_without_busy_falls_back_to_sleep():
    fw = FakeFirmware(has_busy=False)
    stage = make_stage(fw)
    stage.move(100, "X", is_blocking=True)  # 100 steps @ 800/s ~ 0.65 s wait
    assert not fw._busy()
    assert stage._has_busy is False


def test_step_size_derived_from_active_microsteps():
    fw = FakeFirmware(microsteps=64)
    stage = make_stage(fw, umPerElectricalCycleX=1600.0, umPerElectricalCycleY=800.0)
    assert stage._stepsizeX == 25.0
    assert stage._stepsizeY == 12.5


def test_power_sent_at_startup():
    fw = FakeFirmware()
    make_stage(fw, powerX=8000, powerY=18000)
    assert [c[1] for c in fw.cmds("POWER")] == ["POWER 8000 18000"]


def test_watchdog_does_not_stop_a_running_move():
    fw = FakeFirmware()
    stage = make_stage(fw, idleStopTimeoutS=0.1)
    fw.busy_until = time.time() + 10  # long move in progress
    time.sleep(1.2)
    assert fw.cmds("STOP") == []
    fw.busy_until = 0
    time.sleep(1.2)
    assert fw.cmds("STOP"), "idle watchdog never fired"
    stage.finalize()


def test_approach_direction_overshoots_only_against_it():
    fw = FakeFirmware()
    stage = make_stage(fw, approachDirectionX=1, approachDirectionY=1, approachOvershootSteps=8)
    stage.move(10, "X")          # with the approach direction: one move
    stage.move(-10, "X")         # against it: overshoot, then approach
    stage.move((-5, 3), "XY")    # mixed: only X overshoots
    moves = [m[1] for m in fw.cmds("MOVE")]
    assert moves == ["MOVE 10 0 800", "MOVE -18 0 800", "MOVE 8 0 800", "MOVE -13 3 800", "MOVE 8 0 800"]
    assert stage.getPosition()["X"] == -5 and stage.getPosition()["Y"] == 3


def test_approach_follows_axis_swap():
    fw = FakeFirmware()
    stage = make_stage(fw, approachDirectionX=1, swapXY=True)   # logical X = device Y
    stage.move(-4, "X")
    assert [m[1] for m in fw.cmds("MOVE")] == ["MOVE 0 -12 800", "MOVE 0 8 800"]


def test_speed_cap_and_settle():
    fw = FakeFirmware()
    stage = make_stage(fw, maxSpeedSteps=100, settleMs=100)
    t0 = time.time()
    stage.move(4, "X", speed=(20000, 20000))
    assert fw.cmds("MOVE")[0][1] == "MOVE 4 0 100"
    assert time.time() - t0 >= MOVE_S + 0.1


def test_firmware_settings_sent_at_startup():
    fw = FakeFirmware()
    make_stage(fw, firmware={"ACCEL": 1500, "HOLD": 0, "DITHER": "1 16 50", "PHASE_Y": [0, -3, 5]})
    sent = [c for _, c, _ in fw.log]
    assert "ACCEL 1500" in sent and "HOLD 0" in sent and "DITHER 1 16 50" in sent
    assert ["PHASE Y 0 0", "PHASE Y 1 -3", "PHASE Y 2 5"] == [c for c in sent if c.startswith("PHASE")]


def test_scan_hints_full_step():
    fw = FakeFirmware(microsteps=32)
    stage = make_stage(fw, umPerElectricalCycleX=4157.0, umPerElectricalCycleY=3908.0, approachDirectionX=1)
    h = stage.getScanHints()
    assert abs(h["fullStepUm"]["X"] - 4157.0 / 4) < 1e-9 and abs(h["fullStepUm"]["Y"] - 977.0) < 1e-9
    assert h["recommendedPattern"] == "raster"


def test_home_drives_negative_and_zeroes_only_homed_axes():
    fw = FakeFirmware()
    stage = make_stage(fw, homeSpeedSteps=50, homeTimeS=0.05)
    stage.move((40, 30), "XY")
    stage.home()
    assert fw.cmds("MOVE")[-1][1] == f"MOVE {-50 * 9999} {-50 * 9999} 50"
    assert stage.getPosition()["X"] == stage.getPosition()["Y"] == 0.0
    stage.move((40, 30), "XY")
    stage.doHome("Y")                      # per-axis: X keeps its position
    assert fw.cmds("MOVE")[-1][1] == f"MOVE 0 {-50 * 9999} 50"
    assert (stage.getPosition()["X"], stage.getPosition()["Y"]) == (40, 0.0)
    stage.doHome("Z")                      # no Z axis: no-op
    assert fw.cmds("MOVE")[-1][1] == f"MOVE 0 {-50 * 9999} 50"


def test_move_steps_is_raw_and_tracks_position():
    fw = FakeFirmware(microsteps=32)
    stage = make_stage(fw, approachDirectionX=1, umPerElectricalCycleX=3200.0, umPerElectricalCycleY=3200.0)
    stage.move_steps(-4, 3)                        # against the approach direction: still one raw move
    assert [m[1] for m in fw.cmds("MOVE")] == ["MOVE -4 3 800"]
    assert stage.getPosition()["X"] == -400.0 and stage.getPosition()["Y"] == 300.0


def test_apply_calibration_returns_properties():
    fw = FakeFirmware(microsteps=32)
    stage = make_stage(fw, swapXY=True)
    stage.get_power = lambda: (24000, 18000)       # device order (swapped)
    props = stage.apply_calibration(powerX=9000, powerY=15000, umPerStepX=125.0, approachOvershootSteps=5)
    assert fw.cmds("POWER")[-1][1] == "POWER 15000 9000"         # swapped on the wire
    assert props == {"powerX": 9000, "powerY": 15000, "stepsizeX": 125.0,
                     "umPerElectricalCycleX": 4000.0, "approachOvershootSteps": 5}
    assert stage._stepsizeX == 125.0 and stage._approach_overshoot == 5
