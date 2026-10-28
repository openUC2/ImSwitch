"""ShitScopeController endpoints with a fake master (camera + stage from test_shitscope_scan)."""
import time
from types import SimpleNamespace

import pytest

from imswitch.imcommon.model import dirtools
from imswitch.imcontrol.controller.controllers.ShitScopeController import ShitScopeController
from imswitch.imcontrol._test.unit.test_shitscope_scan import FakeCamera, FakeStage, T, UM_PX


class FakeDetectors(dict):
    def __init__(self, cam):
        super().__init__(cam=cam); self.handles = []

    def getCurrentDetectorName(self):
        return "cam"

    def startAcquisition(self, liveView=False):
        self.handles.append("h"); return "h"

    def stopAcquisition(self, handle, liveView=False):
        self.handles.remove(handle)


class FakePositioners(dict):
    def getAllDeviceNames(self):
        return list(self)


@pytest.fixture
def controller(tmp_path, monkeypatch):
    monkeypatch.setattr(dirtools.UserFileDirs, "getValidatedDataPath", staticmethod(lambda: str(tmp_path)))
    stage = FakeStage(error_um=30.0)
    stage.getScanHints = lambda: {"fullStepUm": {"X": 100.0, "Y": 100.0}, "recommendedMinOverlap": 0.25,
                                  "recommendedPattern": "raster"}
    cam = FakeCamera(stage)
    cam.shape, cam.pixelSizeUm = (T, T), [1, UM_PX, UM_PX]
    master = SimpleNamespace(detectorsManager=FakeDetectors(cam), positionersManager=FakePositioners(xy=stage))
    return ShitScopeController(None, None, master, widget=None, factory=None, moduleCommChannel=None), master


def test_info_suggests_full_step_multiples(controller):
    c, _ = controller
    info = c.getShitScopeInfo()
    assert info["umPerPx"] == UM_PX and info["fovXUm"] == T * UM_PX
    assert info["suggestedStepXUm"] == 300.0 and info["pattern"] == "raster"   # 480 µm FOV, 25 % overlap -> 3 full steps


def test_scan_lifecycle(controller):
    c, master = controller
    assert c.getShitScopeStatus()["state"] == "idle" and not c.getShitScopeResult()["success"]
    r = c.startShitScopeScan(nx=3, ny=3)
    assert r["success"] and r["tiles"] == 9 and r["pattern"] == "raster"
    assert not c.startShitScopeScan(nx=2, ny=2)["success"]        # one scan at a time
    t0 = time.time()
    while c.getShitScopeStatus()["state"] in ("running", "analysing") and time.time() - t0 < 60:
        time.sleep(0.05)
    st = c.getShitScopeStatus()
    assert st["state"] == "done" and st["summary"]["tiles_solved"] == 9
    assert master.detectorsManager.handles == []                  # acquisition released
    assert c.getShitScopePreview()["image"].startswith("data:image/png")
    assert c.getShitScopeResult()["success"]
    assert c.analyzeShitScopeScan()["summary"]["tiles_solved"] == 9


def test_no_stage_is_reported_not_raised(controller):
    c, master = controller
    master.positionersManager.clear()
    info = c.getShitScopeInfo()
    assert info["stageAvailable"] is False and info["umPerPx"] == UM_PX
    r = c.startShitScopeScan(nx=2, ny=2)
    assert not r["success"] and "No stage" in r["error"]


class CalStage:
    """SimStage from the calibration test behind the StepperXYStageManager calls."""
    def __init__(self):
        from imswitch.imcontrol._test.unit.test_shitscope_calibration import SimStage
        self.sim, self.applied = SimStage(), None
        self.sim.set_power(18000, 24000)

    def move_steps(self, dx, dy, speed=None):
        self.sim.move_steps(dx, dy)

    def set_axis_power(self, x, y):
        self.sim.set_power(x, y)

    def get_axis_power(self):
        return self.sim.power["X"], self.sim.power["Y"]

    def apply_calibration(self, **kw):
        self.applied = kw
        return {"powerX": kw["powerX"], "powerY": kw["powerY"]}

    def getPosition(self):
        return {"X": 0.0, "Y": 0.0}


class CalCamera:
    def __init__(self, stage):
        self.stage, self.fid = stage, 0
        self.shape = (800, 600)
        from imswitch.imcontrol._test.unit.test_shitscope_calibration import UM_PX
        self.pixelSizeUm = [1, UM_PX, UM_PX]

    def getLatestFrame(self, returnFrameNumber=False):
        time.sleep(0.002); self.fid += 1
        f = self.stage.sim.grab()
        return (f, self.fid) if returnFrameNumber else f


def test_calibration_endpoints(tmp_path, monkeypatch):
    import imswitch.imcontrol.model.configfiletools as cft
    stage = CalStage(); cam = CalCamera(stage)
    master = SimpleNamespace(detectorsManager=FakeDetectors(cam), positionersManager=FakePositioners(xy=stage))
    setup = SimpleNamespace(positioners={"xy": SimpleNamespace(managerProperties={"powerX": 18000})})
    saved = []
    monkeypatch.setattr(cft, "loadOptions", lambda: ("opts", None))
    monkeypatch.setattr(cft, "saveSetupInfo", lambda opts, info: saved.append(info))
    c = ShitScopeController(setup, None, master, widget=None, factory=None, moduleCommChannel=None)
    assert c.getShitScopeCalibrationStatus() == {"state": "idle"}
    assert not c.applyShitScopeCalibration()["success"]
    assert c.startShitScopeCalibration()["startPower"] == [18000, 24000]
    assert not c.startShitScopeScan(nx=2, ny=2)["success"]          # blocked while calibrating
    t0 = time.time()
    while c.getShitScopeCalibrationStatus()["state"] == "running" and time.time() - t0 < 120:
        time.sleep(0.1)
    st = c.getShitScopeCalibrationStatus()
    assert st["state"] == "done" and st["result"]["ok"], st
    assert master.detectorsManager.handles == []                    # acquisition released
    r = c.applyShitScopeCalibration(persist=True)
    assert r["success"] and r["saved"] and stage.applied["powerX"] == st["result"]["recommended"]["powerX"]
    assert setup.positioners["xy"].managerProperties["powerX"] == r["applied"]["powerX"] and saved
