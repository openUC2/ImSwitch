"""SpectrometerController endpoints on the simulated spectrometer with a fake master."""
import io
import json
import os
import time
from types import SimpleNamespace

import numpy as np
import pytest

from imswitch.imcommon.model import dirtools
from imswitch.imcontrol.controller.controllers.SpectrometerController import (
    SpectrometerCalibrationBody, SpectrometerController, SpectrometerReferenceBody,
    SpectrometerSettingsBody, SpectrometerWizardBody)
from imswitch.imcontrol.model.SetupInfo import SpectrometerInfo
from imswitch.imcontrol.model.spectrometer import spectrum_io


class FakeLaser:
    def __init__(self, wavelength):
        self.wavelength, self.valueRangeMin, self.valueRangeMax = wavelength, 0, 1000
        self.isBinary, self.enabled, self.value = False, False, 0

    def setValue(self, v):
        self.value = v

    def setEnabled(self, on):
        self.enabled = bool(on)


class FakeLasers(dict):
    def getAllDeviceNames(self):
        return list(self)


class FakeMatrix:
    def __init__(self):
        self.calls = []

    def setAll(self, state=0, intensity=None):
        self.calls.append((state, intensity))


class FakeMatrices(dict):
    def getAllDeviceNames(self):
        return list(self)


def strict_json(obj):
    """What FastAPI does: no NaN, no numpy."""
    return json.loads(json.dumps(obj, allow_nan=False))


@pytest.fixture
def ctrl(tmp_path, monkeypatch):
    monkeypatch.setattr(dirtools.UserFileDirs, "Config", str(tmp_path / "config"))
    monkeypatch.setattr(dirtools.UserFileDirs, "getValidatedDataPath",
                        staticmethod(lambda: str(tmp_path / "data")))
    info = SpectrometerInfo(simulate=True, laserWavelengthNm=532.0, alignmentTargetNm=555.0,
                            design={"cameraFocalLengthMm": 32.0, "centerWavelengthNm": 550.0,
                                    "leverArmMm": 25.0},
                            processing={"rowStart": 57, "rowStop": 456})
    lasers = FakeLasers({"LASER488": FakeLaser(488), "LED635": FakeLaser(635),
                         "NOWL": FakeLaser(0)})
    matrix = FakeMatrix()
    master = SimpleNamespace(lasersManager=lasers, detectorsManager=None,
                             LEDMatrixsManager=FakeMatrices(ring=matrix),
                             getController=lambda name: None)
    setup = SimpleNamespace(spectrometer=info)
    c = SpectrometerController(setup, None, master, widget=None, factory=None,
                               moduleCommChannel=None)
    yield c, master, tmp_path
    c.closeEvent()


def wait(fn, done, timeout=60.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        out = fn()
        if done(out):
            return out
        time.sleep(0.05)
    raise TimeoutError(out)


def test_state_sources_and_live(ctrl):
    c, master, _ = ctrl
    st = strict_json(c.getSpectrometerState())
    assert st["simulated"] and st["processing"]["rowStart"] == 57
    assert st["calibration"]["model"] == "design" and not st["calibration"]["calibrated"]
    ids = {s["id"] for s in c.getSpectrometerSources()["sources"]}
    assert {"laser:LASER488", "laser:LED635", "led:r", "led:g", "led:b", "qbox520",
            "neon", "cfl", "sim:white"} <= ids and "laser:NOWL" not in ids
    assert c.setSpectrometerSource("laser:LASER488", True)["success"]
    assert master.lasersManager["LASER488"].enabled
    assert master.lasersManager["LASER488"].value == pytest.approx(50.0)   # 5 % of 1000
    assert c.startSpectrometerLive(periodMs=60)["success"]
    sp = wait(lambda: c.getSpectrometerSpectrum(maxPoints=1024),
              lambda s: s.get("available") and s["peaks"])
    sp = strict_json(sp)
    assert sp["which"] == "live"
    assert len(sp["counts"]) == 1024 and sp["ramanShiftCm1"] is not None
    assert abs(sp["peaks"][0]["nm"] - 488.0) < 6.0          # design axis, not calibrated yet
    img = c.getSpectrometerImage()
    assert img["image"].startswith("data:image/png")
    full = c.getSpectrometerImage(full=True)
    assert full["rows"] == [57, 456] and full["full"]
    c.stopSpectrometerLive()
    c.allSpectrometerSourcesOff()
    assert not master.lasersManager["LASER488"].enabled


def test_settings_roi_camera_and_resolution(ctrl):
    c, _, _ = ctrl
    r = c.setSpectrometerSettings(SpectrometerSettingsBody(collapse="sum", liveAverage=3))
    assert r["success"] and r["processing"]["collapse"] == "sum" and r["liveAverage"] == 3
    assert not c.setSpectrometerSettings(SpectrometerSettingsBody(collapse="max"))["success"]
    c.setSpectrometerSource("sim:white", True)
    roi = c.autoSpectrometerRoi()
    assert roi["success"] and abs(roi["rowStart"] - 57) < 25 and abs(roi["rowStop"] - 456) < 25
    cam = c.setSpectrometerCamera(exposureMs=20.0, gain=1.5, bitDepth=8)
    assert cam["exposureMs"] == 20.0 and cam["gain"] == 1.5 and cam["notes"]
    c.setSpectrometerSource("sim:white", False)
    c.setSpectrometerSource("laser:LED635", True)
    assert c.acquireSpectrometerSpectrum(nAverage=3)["success"]
    job = wait(c.getSpectrometerAcquisition, lambda j: j["state"] != "running")
    assert job["state"] == "done" and job["result"]["frames"] == 3
    res = strict_json(c.checkSpectrometerResolution(which="acquired"))
    assert res["success"] and 0.5 < res["widthOverSlitImage"] < 1.6 and res["fwhmCm1"] > 0


def test_dark_acquire_save(ctrl):
    c, _, tmp = ctrl
    assert c.takeSpectrometerDark(nAverage=3)["success"]
    assert wait(c.getSpectrometerAcquisition, lambda j: j["state"] != "running")["state"] == "done"
    assert c.getSpectrometerState()["dark"]["frames"] == 3
    c.setSpectrometerSource("laser:LASER488", True)
    c.acquireSpectrometerSpectrum(nAverage=4)
    wait(c.getSpectrometerAcquisition, lambda j: j["state"] != "running")
    sp = c.getSpectrometerSpectrum(which="acquired")
    assert sp["metadata"]["darkSubtracted"] and sp["metadata"]["frames"] == 4
    h5 = c.saveSpectrometerSpectrum(name="led 488/test", fileFormat="hdf5", which="acquired")
    assert h5["success"] and h5["files"][0].endswith(".h5")
    back = spectrum_io.load_hdf5(h5["files"][0])
    assert back["roi"].shape == (399, 3072) and back["metadata"]["frames"] == 4
    assert back["metadata"]["activeSources"] == ["laser:LASER488"]
    csv = c.saveSpectrometerSpectrum(name="x", fileFormat="csv", which="acquired")
    assert len(csv["files"]) == 2 and all(os.path.isfile(f) for f in csv["files"])
    c.clearSpectrometerDark()
    assert c.getSpectrometerState()["dark"] is None


def test_frame_adding_and_region_array(ctrl):
    c, _, _ = ctrl
    c.setSpectrometerSource("laser:LASER488", True)
    mean_peak = {}
    for mode in ("mean", "sum"):
        assert c.setSpectrometerSettings(SpectrometerSettingsBody(frameCombine=mode))["success"]
        c.acquireSpectrometerSpectrum(nAverage=4)
        wait(c.getSpectrometerAcquisition, lambda j: j["state"] != "running")
        sp = c.getSpectrometerSpectrum(which="acquired", maxPoints=0)
        mean_peak[mode] = max(sp["counts"])
        assert sp["metadata"]["frameCombine"] == mode
        assert sp["metadata"]["integrationMs"] == pytest.approx(10.0 * (4 if mode == "sum" else 1))
    assert mean_peak["sum"] / mean_peak["mean"] == pytest.approx(4.0, rel=0.05)
    assert not c.setSpectrometerSettings(SpectrometerSettingsBody(frameCombine="max"))["success"]
    r = c.getSpectrometerRoiArray(which="acquired")
    arr = np.load(io.BytesIO(r.body))
    assert arr.dtype == np.float32 and arr.shape == (399, 3072)
    assert r.headers["X-Rows"] == "57,456" and r.headers["X-Cols"] == "0,3072"
    corr = np.load(io.BytesIO(c.getSpectrometerRoiArray(which="acquired", corrected=True).body))
    assert corr.shape == arr.shape


def test_wizard_with_lasers_qbox_and_neopixels(ctrl):
    c, master, tmp = ctrl
    body = SpectrometerWizardBody(
        sources=["laser:LASER488", "qbox520", "laser:LED635", "led:r", "led:g", "led:b"],
        nAverage=3, nDark=2, settleS=0.0, sigmas={"laser:LASER488": 0.05,
                                                  "qbox520": 0.05, "laser:LED635": 0.05})
    assert c.startSpectrometerWizard(body)["success"]
    st = wait(c.getSpectrometerWizard, lambda s: s["state"] in ("waiting", "review", "error"))
    assert st["state"] == "waiting" and "Q-Box" in st["message"]
    assert not master.lasersManager["LASER488"].enabled                # 488 done and off
    c.continueSpectrometerWizard()
    st = strict_json(wait(c.getSpectrometerWizard,
                          lambda s: s["state"] in ("review", "error", "cancelled")))
    assert st["state"] == "review", st["message"]
    assert [s["status"] for s in st["steps"]] == ["done"] * 6
    assert st["smile"]["bowPx"] == pytest.approx(16, abs=1.5)
    assert not any(laser.enabled for laser in master.lasersManager.values())
    assert master.LEDMatrixsManager["ring"].calls[-1][0] == 0          # ring off at the end
    assert c.applySpectrometerWizard(save=True)["success"]
    cal = c.getSpectrometerCalibration()
    assert cal["calibrated"] and cal["model"] == "polynomial"         # 6 points -> auto
    sim = c._mgr.simulator
    grid = np.linspace(0, 3071, 30)
    assert np.max(np.abs(c._mgr.calibration.px_to_nm(grid) - sim.px_to_nm(grid))) < 4.0
    lasers_only = [p for p in cal["points"] if p["kind"] == "laser"]
    assert all(abs(p["residualNm"]) < 2.0 for p in lasers_only)
    # the alignment uses the grating fit stored next to the polynomial, not the design
    win = c.getSpectrometerAlignment(lineNm=520.0, targetCenterNm=555.0)["windowAtTargetNm"]
    true_win = sorted(sim.px_to_nm([0, 3071]))
    assert abs(win[0] - true_win[0]) < 1.5 and abs(win[1] - true_win[1]) < 1.5
    saved = json.load(open(c._mgr.stateFile))
    assert saved["calibration"]["model"] == "polynomial"


def test_laser_only_wizard_is_accurate_and_alignment_uses_it(ctrl):
    c, _, _ = ctrl
    body = SpectrometerWizardBody(sources=["laser:LASER488", "laser:LED635"], nAverage=3,
                                  takeDark=False, settleS=0.0,
                                  sigmas={"laser:LASER488": 0.05, "laser:LED635": 0.05})
    c.startSpectrometerWizard(body)
    st = wait(c.getSpectrometerWizard, lambda s: s["state"] in ("review", "error"))
    assert st["state"] == "review" and st["fit"]["model"] == "grating"
    c.applySpectrometerWizard(save=False)
    sim = c._mgr.simulator
    grid = np.linspace(0, 3071, 30)
    assert np.max(np.abs(c._mgr.calibration.px_to_nm(grid) - sim.px_to_nm(grid))) < 0.5
    c.setSpectrometerSource("laser:LED635", True)
    c.acquireSpectrometerSpectrum(nAverage=2)
    wait(c.getSpectrometerAcquisition, lambda j: j["state"] != "running")
    plan = strict_json(c.getSpectrometerAlignment(lineNm=635.0, which="acquired"))
    assert plan["success"] and plan["lineOnSensorAtTarget"]
    assert abs(plan["measuredPx"] - float(sim.nm_to_px(635.0))) < 1.0
    assert "column" in plan["advice"]
    assert plan["lineFound"] == "calibrated"
    far = c.getSpectrometerAlignment(lineNm=520.0, targetCenterNm=700.0, which="acquired")
    assert not far["lineOnSensorAtTarget"] and "outside" in far["advice"]
    assert far["lineFound"] == "not found" and far["measuredPx"] is None   # only 635 is on
    assert far["advice"].startswith("No line near 520")


def test_wizard_cancel_and_manual_calibration(ctrl):
    c, master, _ = ctrl
    c.startSpectrometerWizard(SpectrometerWizardBody(sources=["qbox520", "laser:LASER488"],
                                                     takeDark=False, settleS=0.0))
    wait(c.getSpectrometerWizard, lambda s: s["state"] == "waiting")
    c.cancelSpectrometerWizard()
    st = wait(c.getSpectrometerWizard, lambda s: s["state"] in ("cancelled", "error"))
    assert st["state"] == "cancelled" and not master.lasersManager["LASER488"].enabled
    assert not c.applySpectrometerWizard()["success"]
    pts = [{"px": 2900.0, "nm": 488.0, "sigmaNm": 0.1}, {"px": 1000.0, "nm": 610.0},
           {"px": 300.0, "nm": 650.0}]
    r = c.setSpectrometerCalibration(SpectrometerCalibrationBody(points=pts, model="grating"))
    assert r["success"] and r["summary"]["calibrated"]
    assert not c.setSpectrometerCalibration(SpectrometerCalibrationBody(points=[]))["success"]
    c.resetSpectrometerCalibration()
    assert not c.getSpectrometerCalibration()["calibrated"]


def test_reference_upload_replaces_the_placeholder(ctrl):
    c, _, tmp = ctrl
    nm = list(np.arange(450.0, 600.0, 0.5))
    g = list(np.exp(-0.5 * ((np.array(nm) - 527.0) / 12.0) ** 2))
    r = c.setSpectrometerReferenceSpectrum(SpectrometerReferenceBody(key="g", wavelengthNm=nm,
                                                                     intensity=g, label="ring G"))
    assert r["success"] and r["centerNm"] == pytest.approx(527.0, abs=0.2)
    src = {s["id"]: s for s in c.getSpectrometerSources()["sources"]}["led:g"]
    assert not src["reference"]["placeholder"] and src["sigmaNm"] == 1.5
    assert os.path.isfile(os.path.join(str(tmp / "config"), "spectrometer",
                                       "spectrometer_references.json"))
