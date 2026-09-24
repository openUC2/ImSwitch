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
        if word in ("STOP", "POWER", "PING", "V", "MICROSTEP"):
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
