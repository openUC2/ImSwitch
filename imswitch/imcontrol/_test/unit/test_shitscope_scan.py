"""ShitScopeScan against a fake camera/stage that images a textured world at
the stage's *true* position (commanded + random placement error)."""
import json
import os
import threading
import time

import numpy as np
import pytest
from scipy.ndimage import gaussian_filter

from imswitch.imcontrol.model import shitscope_scan as ss

RNG = np.random.default_rng(5)
WORLD = 1000 + 300 * gaussian_filter(RNG.normal(size=(1600, 1600)), 3)
UM_PX, T = 2.0, 240          # µm/px, tile size px


class FakeStage:
    def __init__(self, error_um=40.0, delay=0.0):
        self.cmd = {"X": 0.0, "Y": 0.0}; self.true = dict(self.cmd)
        self.error_um, self.delay, self.moves = error_um, delay, 0

    def move(self, value, axis, is_absolute, is_blocking):
        assert axis == "XY" and is_absolute and is_blocking
        time.sleep(self.delay)
        self.cmd = {"X": value[0], "Y": value[1]}
        self.true = {k: v + RNG.normal(0, self.error_um) for k, v in self.cmd.items()}
        self.moves += 1

    def getPosition(self):
        return dict(self.cmd)


class FakeCamera:
    """New frame every 5 ms showing the world at the stage's true position."""
    def __init__(self, stage):
        self.stage, self.fid, self.flushed = stage, 0, 0

    def flushBuffer(self):
        self.flushed += 1

    def getLatestFrame(self, returnFrameNumber=False):
        time.sleep(0.005); self.fid += 1
        x = int(round(self.stage.true["X"] / UM_PX)) + 600
        y = int(round(self.stage.true["Y"] / UM_PX)) + 600
        f = WORLD[y:y + T, x:x + T].astype(np.uint16)
        return (f, self.fid) if returnFrameNumber else f


def run(tmp_path, nx=4, ny=3, **kw):
    stage = FakeStage(**kw); cam = FakeCamera(stage)
    step = ss.suggested_step(T * UM_PX, 100.0, 0.25)          # full step 100 µm, 25 % overlap
    plan = ss.plan_grid(0, 0, nx, ny, step, step, "raster", centered=True)
    scan = ss.ShitScopeScan(cam, stage, str(tmp_path), UM_PX, plan, analysis_downsample=1, stitch_downsample=1)
    return scan, stage, cam, plan


def test_plan_and_step():
    assert ss.suggested_step(1000, 130, 0.25) == 650          # 5 full steps <= 750
    assert ss.suggested_step(1000, None, 0.25) == 750
    p = ss.plan_grid(0, 0, 3, 2, 10, 20, "snake", centered=False)
    assert [(t["ix"], t["iy"]) for t in p] == [(0, 0), (1, 0), (2, 0), (2, 1), (1, 1), (0, 1)]
    r = ss.plan_grid(100, 100, 3, 3, 10, 10, "raster", centered=True)
    assert r[0]["x"] == 90 and r[-1]["y"] == 110 and all(t["ix"] == i % 3 for i, t in enumerate(r))


def test_full_scan_registers_and_stitches(tmp_path):
    scan, stage, cam, plan = run(tmp_path)
    scan.start(); scan._thread.join(60)
    st = scan.status()
    assert st["state"] == "done" and st["tile"] == len(plan) and st["error"] is None
    assert cam.flushed == len(plan)                           # a fresh frame per tile
    assert stage.cmd == {"X": 0.0, "Y": 0.0}                  # returned to start
    for name in ("scan.json", "tiles.json", "scan_quality.json", "stitched.tif"):
        assert os.path.exists(tmp_path / name), name
    assert len(os.listdir(tmp_path / "tiles")) == len(plan)
    s = scan.result["summary"]
    assert s["tiles_solved"] == len(plan) and s["pair_rms_um"] < 2 * UM_PX
    assert s["max_err_um"] > 20                               # the stage did misplace tiles...
    assert scan.result["error_map"].startswith("data:image/png") and scan.preview_png().startswith("data:image/png")


def test_stop_keeps_partial_scan(tmp_path):
    scan, stage, cam, plan = run(tmp_path, nx=5, ny=5, delay=0.05)
    scan.start(); time.sleep(0.4); scan.stop()
    st = scan.status()
    assert st["state"] in ("stopped", "done") and 0 < st["tile"] < len(plan)
    assert len(json.load(open(tmp_path / "tiles.json"))) == st["tile"]


def test_reanalyse_saved_scan(tmp_path):
    scan, *_ = run(tmp_path)
    scan.start(); scan._thread.join(60)
    again = ss.analyse_saved_scan(str(tmp_path), analysis_downsample=1)
    assert again["summary"]["tiles_solved"] == scan.result["summary"]["tiles_solved"]
    assert np.allclose(again["measured_um"], scan.result["measured_um"])


def test_camera_failure_reports_error(tmp_path):
    scan, stage, cam, plan = run(tmp_path)
    cam.getLatestFrame = lambda returnFrameNumber=False: (None, None)
    scan.frame_timeout_s = 0.2
    scan.start(); scan._thread.join(10)
    assert scan.status()["state"] == "error" and "no frame" in scan.status()["error"]


def test_home_first_then_scan_and_return(tmp_path):
    stage = FakeStage(error_um=0); cam = FakeCamera(stage)
    stage.cmd = {"X": 300.0, "Y": 200.0}; stage.true = dict(stage.cmd)
    homed = []
    def home():
        homed.append(scan.state); stage.cmd = {"X": 0.0, "Y": 0.0}
    plan = ss.plan_grid(300, 200, 2, 2, 100, 100, "raster", centered=False)
    scan = ss.ShitScopeScan(cam, stage, str(tmp_path), UM_PX, plan, analysis_downsample=1,
                            stitch_downsample=1, home=home)
    scan.start(); scan._thread.join(60)
    assert homed == ["homing"] and scan.status()["tile"] == 4
    assert [(r["x"], r["y"]) for r in scan.records] == [(300, 200), (400, 200), (300, 300), (400, 300)]
    assert stage.cmd == {"X": 300.0, "Y": 200.0}             # back to the pre-homing start
