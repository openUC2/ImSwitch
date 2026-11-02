"""FrameCalibrationController end to end against a simulated FRAME: a stage with
backlash, a camera that renders the calibration plate (via, holes) through a known
rotated + mirrored camera matrix, and a plate sitting 0.2° rotated and ~1.5 mm off
in the holder. The routine must recover the camera matrix, the stage offset, the
plate rotation and the backlash."""
import math
import threading
from types import SimpleNamespace

import cv2
import numpy as np

from imswitch.imcontrol.controller.controllers import frame_calibration_tools as fct
from imswitch.imcontrol.controller.controllers.FrameCalibrationController import FrameCalibrationController

PS, ROT = 0.6, math.radians(30)                                  # µm/px, camera rotation
J_TRUE = (1 / PS) * np.array([[math.cos(ROT), -math.sin(ROT)], [-math.sin(ROT), -math.cos(ROT)]])  # mirrored
THETA = math.radians(0.2)                                          # plate rotation in the holder
D0 = np.array([1200.0, -900.0])                                    # device position of the plate origin
BACKLASH = {"X": 12.0, "Y": 30.0}
Z_FOCUS = -1500.0                                                  # plate in focus (homed Z = 0)
H, W = 900, 1200


class Stage:
    axes = ["X", "Y", "Z"]
    homeParams = {a: {"enabled": True, "steps": 0} for a in axes}

    def __init__(self):
        self.cmd = {a: 5000.0 for a in self.axes}          # commanded device position (what firmware reports)
        self.real = dict(self.cmd)                           # where the carriage really is
        self.offset = {a: 0.0 for a in self.axes}

    def move(self, value, axis, is_absolute=True, is_blocking=True, **_):
        x = value + self.offset[axis] if is_absolute else self.cmd[axis] + value
        self.cmd[axis] = x
        b, y = BACKLASH.get(axis, 0.0), self.real[axis]
        if x - y > b / 2:
            self.real[axis] = x - b / 2
        elif y - x > b / 2:
            self.real[axis] = x + b / 2

    def getPosition(self):
        return {a: self.cmd[a] - self.offset[a] for a in self.axes}

    def getStageOffsetAxis(self, axis):
        return self.offset[axis]

    def doHome(self, axis, isBlocking=False):
        self.cmd[axis] = self.real[axis] = 0.0
        return [{"qid": 1}, {"home": {"stepperid": 1, "status": "done", "pos": 0}, "qid": 1}]

    def stopAll(self):
        pass


class Camera:
    """Renders the plate: the via (~100x brighter than the plate), sunburst rays around
    it, the rect hole; blurred with defocus, scaled with the exposure."""

    def __init__(self, stage):
        self.stage, self.n = stage, 0
        self.yy, self.xx = np.mgrid[0:H, 0:W]
        self.parameters = {"exposure": SimpleNamespace(value=40.0)}

    def setParameter(self, name, value):
        self.parameters[name].value = value

    def getLatestFrame(self, returnFrameNumber=False):
        self.n += 1
        d = np.array([self.stage.real["X"], self.stage.real["Y"]])
        c = np.array([(W - 1) / 2, (H - 1) / 2])
        R = np.array([[math.cos(THETA), -math.sin(THETA)], [math.sin(THETA), math.cos(THETA)]])
        # plate coordinates of every pixel: stage-frame offset from the optical axis, rotated into the plate
        off = -np.linalg.solve(J_TRUE, np.stack([self.xx - c[0], self.yy - c[1]]).reshape(2, -1))
        P = (R.T @ (off + (d - D0)[:, None])).reshape(2, H, W)
        img = np.full((H, W), 200.0)
        vx, vy = fct.PLATE["via"]
        r, ang = np.hypot(P[0] - vx, P[1] - vy), np.arctan2(P[1] - vy, P[0] - vx)
        img[(r > 900) & (r < 4000) & ((ang * 12 / np.pi) % 2 < 0.6)] = 900
        rx, ry = fct.PLATE["rectHole"]
        rw, rh = fct.PLATE["rectHoleSize"]
        img[(r < fct.PLATE["viaDiameter"] / 2) | ((abs(P[0] - rx) < rw / 2) & (abs(P[1] - ry) < rh / 2))] = 30000
        sigma = abs(self.stage.real["Z"] - Z_FOCUS) / 40
        if sigma > 0.3:
            img = cv2.GaussianBlur(img, (0, 0), sigma)
        img = img * self.parameters["exposure"].value / 40 + 20 * np.random.default_rng(self.n).standard_normal((H, W))
        img = np.clip(img, 0, 4095)
        return (img, self.n) if returnFrameNumber else img


def test_frame_calibration_recovers_simulated_frame():
    stage = Stage()
    cam = Camera(stage)
    stored = {}
    pc = SimpleNamespace(
        _defaultDetectorName=lambda: "Cam", _validateCameraIntensity=lambda n: (True, ""),
        _resetDetectorFlip=lambda n: None, _restoreAppliedFlip=lambda n: None,
        getPixelSize=lambda det, obj: (PS, PS), applyBacklash=lambda **kw: None,
        setCalibrationData=lambda **kw: stored.update(kw))

    def set_offset(knownPosition, axis):
        stage.offset[axis] = stage.cmd[axis] - knownPosition

    controllers = {"PixelCalibration": pc, "Positioner": SimpleNamespace(setStageOffsetAxis=set_offset)}
    master = SimpleNamespace(
        getController=controllers.get,
        positionersManager=SimpleNamespace(getAllDeviceNames=lambda: ["S"], __getitem__=None),
        detectorsManager=SimpleNamespace(getAllDeviceNames=lambda: ["Cam"], startAcquisition=lambda: 1,
                                         stopAcquisition=lambda h: None),
        lasersManager=SimpleNamespace(getAllDeviceNames=lambda: []))
    master.positionersManager = type("PM", (), {"getAllDeviceNames": lambda s: ["S"], "__getitem__": lambda s, k: stage})()
    master.detectorsManager = type("DM", (), {"getAllDeviceNames": lambda s: ["Cam"], "__getitem__": lambda s, k: cam,
                                              "startAcquisition": lambda s: 1, "stopAcquisition": lambda s, h: None})()

    ctrl = FrameCalibrationController.__new__(FrameCalibrationController)
    ctrl._master = master
    ctrl._setupInfo = SimpleNamespace(
        frameCalibration={"settings": {"settleS": 0.0, "secondObjective": False}},
        objective=SimpleNamespace(magnifications=[4], zPositions=[0], pixelsizes=[PS], xyOffsets=[[0, 0], [0, 0]]))
    ctrl._logger = SimpleNamespace(info=print, warning=print, error=lambda *a, **k: print(*a))
    ctrl._abort, ctrl._continue, ctrl._thread, ctrl._ctx = threading.Event(), threading.Event(), None, {}
    ctrl._status = ctrl._newStatus("running")
    ctrl._saveSetup = lambda: None
    ctrl._writeReport = lambda: None
    ctrl._run()

    st = ctrl._status
    assert st["state"] == "finished", (st["message"], [(s["name"], s["status"], s["detail"]) for s in st["steps"]])
    # camera matrix in PixelCalibration's convention (= -inv(J))
    A = np.array(stored["affineMatrix"])[:, :2]
    assert np.allclose(A, -np.linalg.inv(J_TRUE), rtol=0, atol=2e-3), A
    # stage offset: the via is now at its plate coordinates. Positions are commanded ones;
    # arriving from the negative side leaves the carriage b/2 behind the command.
    via_dev = D0 + np.array([[math.cos(THETA), -math.sin(THETA)], [math.sin(THETA), math.cos(THETA)]]) @ fct.PLATE["via"]
    lag = np.array([BACKLASH["X"], BACKLASH["Y"]]) / 2
    assert np.allclose(np.array([stage.offset["X"], stage.offset["Y"]]) + fct.PLATE["via"], via_dev + lag, atol=2.0)
    res = st["result"]
    assert abs(res["placement"]["rotationDeg"] - math.degrees(THETA)) < 0.03, res["placement"]
    assert abs(res["backlash"]["X"] - BACKLASH["X"]) < 1.5 and abs(res["backlash"]["Y"] - BACKLASH["Y"]) < 1.5, res["backlash"]
    assert res["verifyUm"] < 3.0
    assert abs(res["focus"]["0"]["z"] - Z_FOCUS) <= 20, res["focus"]
    assert cam.parameters["exposure"].value == 40.0          # restored after the via work
