"""One-button FRAME calibration with the FRAME calibration plate.

The user puts the plate in the holder (well-label side up, A1 notch at stage 0,0)
and presses start. The routine homes (verified), checks the motion directions,
finds the centre via, focuses, measures the camera<->stage affine per objective
from the tracked via (every measurement position is approached from the same
side, so backlash cannot bias it), sets the stage offset so that stage
coordinates are plate coordinates, checks the plate placement, measures backlash,
calibrates the second objective (parfocality, parcentricity, affine) and fits the
overview-camera homography. Everything is written to the setup JSON.

Geometry, detectors and fits: frame_calibration_tools.py. Plan:
docs/FRAME_AUTOCALIBRATION_PLAN.md
"""
import json
import math
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np

from imswitch.imcommon.model import APIExport, initLogger
from ..basecontrollers import ImConWidgetController
from ..busy import busy_reasons
from . import frame_calibration_tools as fct

_REPORT_DIR = os.path.join(os.path.expanduser("~"), "imswitch_calibrations", "frame_calibration")

DEFAULT_SETTINGS = {
    "detector": "",              # "" = PixelCalibration's default camera
    "overviewCamera": "",        # "" = experiment.overviewCameraName, else ObservationCamera
    "illumination": "LED",       # transmission light switched on at start ("" = leave as is)
    "illuminationValue": 0,      # 0 = keep its current value
    "overviewIlluminationValue": 60,  # overview frames by room light first (open rig); the LED is practically off below ~100
    "overviewExposureMs": 10.0,  # else full LED, overview camera at this manual exposure (then /2, x2): auto exposure saturates the holes
    "homeTurret": True,
    "secondObjective": True,
    "searchRadiusUm": 4000,      # plate placement tolerance around the nominal via
    "searchRadius2Um": 2000,     # parcentricity tolerance of the second objective (Pi rig 2026-11-02: ~1.5 mm)
    "axesStepUm": 2000,          # overview motion-direction check (< half the 4.5 mm well pitch)
    "focusSearchUm": 6000,       # first focus: sweep this far towards the sample from the homed Z
    "focusStepUm": 100,          # coarse focus step of the first objective (others: x mag ratio)
    "viaExposureFactor": 0.1,    # via work at this fraction of the exposure: the via is ~100x brighter than the plate
    "gridStepFov": 0.25,         # affine grid: +/- this fraction of the short FOV side
    "approachUm": 100,           # unidirectional approach distance, must exceed the backlash
    "zRetreatUm": 500,           # move away from the sample (+Z) before switching objectives
    "settleS": 0.2,
    "placementCheck": True,
    "maxRotationDeg": 0.3,
    "measureBacklash": True,
    "applyBacklash": False,
    "manualFocusFallback": True,
}

STEPS = [  # (name, title, critical)
    ("preflight", "Preflight: cameras, light, stage", True),
    ("homing", "Home Z, X, Y and turret (verified)", True),
    ("axes", "Motion directions (overview camera)", False),
    ("objective0", "First objective: find via, focus, affine", True),
    ("offset", "Stage offset: stage = plate coordinates", True),
    ("placement", "Plate placement in the holder", False),
    ("backlash", "Backlash X/Y", False),
    ("objective1", "Second objective: focus, parcentricity, affine", False),
    ("overview", "Overview camera mapping", False),
    ("save", "Verify and save", True),
]


class _Aborted(Exception):
    pass


class StepError(RuntimeError):
    """A step failed; the message is the hint shown to the user."""


class FrameCalibrationController(ImConWidgetController):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._abort = threading.Event()
        self._continue = threading.Event()
        self._thread = None
        self._ctx = {}
        self._status = self._newStatus("idle")
        self._status["result"] = self._fc().get("result")

    # ------------------------------------------------------------------ API
    @APIExport()
    def getFrameCalibrationSettings(self) -> dict:
        return self._settings()

    @APIExport(requestType="POST")
    def setFrameCalibrationSettings(self, settings: dict) -> dict:
        """Merge known keys into the persisted settings (setup JSON)."""
        merged = self._settings()
        for k, v in (settings or {}).items():
            if k in DEFAULT_SETTINGS:
                merged[k] = type(DEFAULT_SETTINGS[k])(v)
        self._fc()["settings"] = merged
        self._saveSetup()
        return merged

    @APIExport()
    def getFrameCalibrationPlate(self) -> dict:
        return {**fct.PLATE, "wrongOrientations": fct.WRONG_ORIENTATIONS}

    @APIExport()
    def getFrameCalibrationStatus(self) -> dict:
        return self._status

    @APIExport(requestType="POST")
    def startFrameCalibration(self) -> dict:
        if self.isFrameCalibrationRunning():
            return {"success": False, "error": "A FRAME calibration is already running."}
        busy = busy_reasons(self._master, self._logger, ignore=("FrameCalibration",))
        if busy:
            return {"success": False, "error": f"Busy: {', '.join(busy)}."}
        self._abort.clear()
        self._continue.clear()
        self._status = self._newStatus("running")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return {"success": True}

    @APIExport(requestType="POST")
    def stopFrameCalibration(self) -> dict:
        if not self.isFrameCalibrationRunning():
            return {"success": False, "message": "Not running."}
        self._abort.set()
        try:
            af = self._master.getController("Autofocus")
            if af is not None:
                af.stopAutofocus()
            self._stageManager().stopAll()
        except Exception as e:
            self._logger.warning(f"stop: {e}")
        return {"success": True}

    @APIExport(requestType="POST")
    def continueFrameCalibration(self) -> dict:
        """Resume after a manual step (e.g. manual focus)."""
        self._continue.set()
        return {"success": True}

    def isFrameCalibrationRunning(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    # ------------------------------------------------------------ run loop
    def _run(self):
        st = self._status
        s = self._ctx_init()
        try:
            for name, title, critical in STEPS:
                self._checkAbort()
                step = next(x for x in st["steps"] if x["name"] == name)
                step["status"], st["step"], t0 = "running", name, time.time()
                try:
                    detail = getattr(self, f"_step_{name}")(s)
                    step["status"] = "skipped" if detail == "skipped" else ("warn" if step["warnings"] else "ok")
                    step["detail"] = "" if detail == "skipped" else (detail or "")
                except _Aborted:
                    step["status"] = "aborted"
                    raise
                except Exception as e:
                    step["status"], step["detail"] = "failed", str(e)
                    self._logger.error(f"FRAME calibration step {name} failed: {e}", exc_info=not isinstance(e, StepError))
                    if critical:
                        raise StepError(f"{title}: {e}")
                finally:
                    step["seconds"] = round(time.time() - t0, 1)
            st["state"] = "finished"
            st["message"] = "Calibration complete" + (" (with warnings)" if any(x["warnings"] for x in st["steps"]) else "")
        except _Aborted:
            st["state"], st["message"] = "aborted", "Aborted by user"
        except Exception as e:
            st["state"], st["message"] = "failed", str(e)
        finally:
            st["step"] = ""
            st["finishedAt"] = datetime.now().isoformat(timespec="seconds")
            self._cleanup()
            self._writeReport()

    def _ctx_init(self):
        m = self._master
        s = self._settings()
        pc = m.getController("PixelCalibration")
        det = s["detector"] or (pc._defaultDetectorName() if pc else m.detectorsManager.getAllDeviceNames()[0])
        ov = s["overviewCamera"] or getattr(getattr(m, "experimentManager", None), "overviewCameraName", "") or "ObservationCamera"
        names = m.detectorsManager.getAllDeviceNames()
        self._ctx = {"s": s, "det": det, "ov": ov if ov in names and ov != det else None,
                     "obj": {}, "ovSamples": [], "saved": False, "images": {},
                     "zConfig": [float(z) for z in (getattr(self._setupInfo.objective, "zPositions", None) or [])]}
        return s

    def _cleanup(self):
        try:
            self._ctx["expBase"] = self._ctx.get("exp0")
            self._exposure(1.0)
        except Exception:
            pass
        pc = self._master.getController("PixelCalibration")
        if pc is not None and not self._ctx.get("saved") and self._ctx.get("flipReset"):
            try:
                pc._restoreAppliedFlip(self._ctx["det"])
            except Exception:
                pass
        if self._ctx.get("acq") is not None:
            try:
                self._master.detectorsManager.stopAcquisition(self._ctx["acq"])
            except Exception:
                pass

    # --------------------------------------------------------------- steps
    def _step_preflight(self, s):
        m = self._master
        name, value = s["illumination"], s["illuminationValue"]
        if name and name in m.lasersManager.getAllDeviceNames():
            if value:
                m.lasersManager[name].setValue(value)
            m.lasersManager[name].setEnabled(True)
            self._ctx["lightMain"] = getattr(m.lasersManager[name], "power", value)
        self._ctx["acq"] = m.detectorsManager.startAcquisition()
        img = self._grab()
        pc = m.getController("PixelCalibration")
        if pc is not None:
            ok, why = pc._validateCameraIntensity(self._ctx["det"])
            if not ok and "saturat" not in why:      # bright is fine here: the stage may sit over a hole
                g = fct.gray(img)
                if np.percentile(g, 99.5) <= 2 * np.percentile(g, 0.1) + 10:     # nothing above the black level
                    raise StepError(f"{self._ctx['det']}: {why} - is the {name or 'transmission light'} on, "
                                    "exposure and gain ok?")
                # light, but the stage sits over dark plate: the exposure is set at the via
                self._warn(f"{self._ctx['det']}: {why} at the start position - continuing")
            pc._resetDetectorFlip(self._ctx["det"])     # measure on the raw sensor; flips are re-applied on save
            self._ctx["flipReset"] = True
        params = getattr(m.detectorsManager[self._ctx["det"]], "parameters", {})
        if "exposure" in params:
            self._ctx["exp0"] = float(params["exposure"].value)
        out = [f"{self._ctx['det']} {img.shape[1]}x{img.shape[0]}"]
        if self._ctx["ov"]:
            try:
                ov = self._grab(self._ctx["ov"])
                out.append(f"overview {ov.shape[1]}x{ov.shape[0]}")
            except StepError as e:
                self._warn(f"{e}; continuing without overview camera")
                self._ctx["ov"] = None
        p = self._pos()
        out.append(f"stage at ({p['X']:.0f}, {p['Y']:.0f}, {p['Z']:.0f}) µm")
        return ", ".join(out)

    def _step_homing(self, s):
        stage = self._stageManager()
        out = []
        for ax in ["Z", "X", "Y"] + (["A"] if s["homeTurret"] and "A" in stage.axes else []):
            self._checkAbort()
            hp = getattr(stage, "homeParams", {}).get(ax, {})
            if not (hp.get("enabled") or hp.get("steps")):
                self._warn(f"homing disabled for {ax} in the config (home{ax}enabled)")
                continue
            t0 = time.time()
            status = fct.home_status(stage.doHome(ax, isBlocking=True))
            dt = time.time() - t0
            if status != "done":
                msg = (f"{ax} homing: {status} after {dt:.0f} s - check the {ax} end stop "
                       f"(wiring, homeEndstoppolarity{ax}, homeDirection{ax}, homeTimeout{ax})")
                if ax == "A":
                    self._warn(msg + "; turret positions stay open-loop")
                    continue
                raise StepError(msg)
            out.append(f"{ax} {dt:.1f} s")
            if ax == "Z":
                self._ctx["zTop"] = self._pos()["Z"]
        oc = self._master.getController("Objective")
        if oc is not None:
            oc._currentObjective = None   # turret re-referenced: force the next slot move
        return "homed " + ", ".join(out)

    def _step_axes(self, s):
        """Track the round hole in the overview camera for +/- moves on each axis. A
        landmark, not global correlation: the fixed LED glare pins a correlation at zero.
        The hole pair identifies it once; then only the round hole is followed."""
        if not self._ctx["ov"]:
            return "skipped"
        step = float(s["axesStepUm"])
        # the stage at the midpoint of the two holes puts both near the optical axis, inside
        # the LED cone (a dark enclosure lights only the cone); with full light the glare marks it
        ref = self._holePairMid()
        self._move(*ref, approach=0)
        img = self._grab(self._ctx["ov"])
        c = fct.glare_center(img)
        self._ctx["ovAxis"] = c or [img.shape[1] / 2, img.shape[0] / 2]
        self._ctx["ovRadius"] = img.shape[1] / (6 if c else 3)
        try:
            p0 = None
            for mode in self._overviewModes(s):
                if self._overviewMode(*mode):
                    p0 = self._holesAt(ref)
                if p0 is not None:
                    self._ctx["ovMode"] = mode
                    break
            if p0 is None:
                raise StepError("the overview camera does not see the plate holes with the stage at the round hole - "
                                "plate missing or not seated, or the camera moved")
            r0 = np.asarray(p0["round"])
            reach = 0.4 * float(np.linalg.norm(np.asarray(p0["rect"]) - r0))     # less than half way to the rect hole
            self._ctx["ovRef"] = {"u": ref.tolist(), "round": r0.tolist(), "reach": reach}
            shifts = {}
            for i, ax in enumerate("XY"):
                for sign in (1, -1):
                    t = ref.copy()
                    t[i] += sign * step
                    p = self._holesAt(t, near=r0, max_px=reach)
                    if p is None:
                        raise StepError(f"round hole lost in the overview image after {sign * step:+.0f} µm in {ax}")
                    shifts[("+" if sign > 0 else "-") + ax] = np.asarray(p["round"]) - r0
        finally:
            self._overviewMode(self._ctx.get("lightMain"), None)
        J, problems = fct.axes_check(shifts, step, min_px=2.0)
        ref_J = (self._fc().get("result") or {}).get("overviewJ")
        if ref_J and not problems:
            change = fct.direction_change_deg(J, ref_J)
            if change > 25:
                problems.append(f"image directions changed by {change:.0f}° since the last calibration - "
                                "axis/sign config changed or the overview camera moved")
        if problems:
            raise StepError("; ".join(problems))
        self._ctx["overviewJ"] = J.tolist()           # stored (reference for the next run) only when consistent
        out = ""
        Hprev = ((self._fc().get("result") or {}).get("overview") or {}).get("H")
        if Hprev or c:   # hole seen off the optical axis: the stage offset is off by that much
            # the last overview mapping knows the optical axis (plate point = stage position);
            # the glare is ~3.5 mm off it (rig 2026-10-30), so then only gross errors count
            axis = np.asarray(Hprev)[:, 2] if Hprev else None
            axis = axis[:2] / axis[2] if axis is not None else np.asarray(self._ctx["ovAxis"])
            # the round hole is seen at axis + J (u - P - shift) with the stage at u = ref
            shift = (ref - np.asarray(fct.PLATE["roundHole"])) - np.linalg.solve(J, r0 - axis)
            if np.linalg.norm(shift) > (1000 if Hprev else 6000):
                self._ctx["plateShift"] = shift.tolist()
                self._warn(f"the plate is ({shift[0]:+.0f}, {shift[1]:+.0f}) µm from where the stage offset expects it - "
                           "searching there")
            out = f"; plate seen ({shift[0]:+.0f}, {shift[1]:+.0f}) µm from nominal"
        return (f"+X moves the overview image {np.round(J[:, 0] * 1000, 1).tolist()} px/mm, "
                f"+Y {np.round(J[:, 1] * 1000, 1).tolist()} px/mm" + out)

    def _step_objective0(self, s):
        slots = self._slots()
        slot = slots[0]
        self._ctx["slot0"] = slot
        centre = np.add(fct.PLATE["via"], self._ctx.get("plateShift", (0, 0)))
        r = self._calibrateObjective(slot, s, centre, s["searchRadiusUm"])
        return self._objSummary(slot, r)

    def _step_offset(self, s):
        r0 = self._ctx["obj"][self._ctx["slot0"]]
        self._move(*r0["u"])                 # verified: the stage reads within 3 µm of u
        stage = self._stageManager()
        old = np.array([stage.getStageOffsetAxis(axis=a) for a in "XY"], float)
        posc = self._master.getController("Positioner")
        for i, ax in enumerate("XY"):
            # device position from the verified move target, not from one fresh read: the
            # firmware sometimes reports another axis (A under X, rig 2026-11-02 -> offset -63909)
            posc.setStageOffsetAxis(knownPosition=float(fct.PLATE["via"][i]),
                                    currentDevicePosition=float(r0["u"][i] + old[i]), axis=ax)
        new = np.array([stage.getStageOffsetAxis(axis=a) for a in "XY"], float)
        shift = old - new                  # user_new = user_old + (old - new)
        for t in self._status["trace"]:
            t["x"], t["y"] = t["x"] + shift[0], t["y"] + shift[1]
        for r in self._ctx["obj"].values():
            r["u"] = (np.asarray(r["u"]) + shift).tolist()
        found = np.asarray(r0["u"]) - np.array(fct.PLATE["via"])   # where the old offset expected it
        self._ctx["offset"] = {"offset": new.tolist(), "viaFoundAtUm": (found - shift).tolist()}
        return (f"offset ({new[0]:.0f}, {new[1]:.0f}) µm; the via was "
                f"({found[0] - shift[0]:+.0f}, {found[1] - shift[1]:+.0f}) µm from where the previous offset expected it")

    def _step_placement(self, s):
        if not s["placementCheck"]:
            return "skipped"
        r0 = self._ctx["obj"][self._ctx["slot0"]]
        J, plate_level = np.asarray(r0["J"]), r0["plateLevel"]

        def is_hole(xy):
            self._move(*xy)
            return float(np.median(self._grab())) > 1.5 * plate_level

        rect = np.array(fct.PLATE["rectHole"], float)
        if not is_hole(rect):
            found = next((n for n, xy in fct.WRONG_ORIENTATIONS.items() if is_hole(xy)), None)
            raise StepError(f"rectangular hole not at its nominal position - plate is {found or 'not seated or not the FRAME plate'}. "
                            "Insert it well-label side up with the A1 notch in the stage-origin corner.")
        w, h = fct.PLATE["rectHoleSize"]
        edges = []
        for dx, dy, k in ((-w / 2, 0, 0), (w / 2, 0, 0), (0, -h / 2, 1), (0, h / 2, 1)):
            u = (rect[0] + dx, rect[1] + dy)
            self._move(*u)                     # verified target, not a fresh read (firmware reports A under X)
            img = self._grab()
            self._ctx["images"][f"edge_{len(edges)}"] = img       # report (Pi run 7: all four edges ~300 µm off)
            pts = fct.boundary_points(img)
            if len(pts) < 50:
                raise StepError(f"no hole edge seen at ({u[0]:.0f}, {u[1]:.0f}) µm")
            edges.append(float(np.median(fct.px_to_plate(pts, J, u, img.shape[:2])[:, k])))
            self._trace(rect[0] + dx, rect[1] + dy, "edge")
        rc = [(edges[0] + edges[1]) / 2, (edges[2] + edges[3]) / 2]
        pl = fct.placement(fct.PLATE["via"], rc)
        pl["rectSize"] = [edges[1] - edges[0], edges[3] - edges[2]]
        self._ctx["placement"] = pl
        if abs(pl["rotationDeg"]) > s["maxRotationDeg"]:
            self._warn(f"plate is rotated {pl['rotationDeg']:.2f}° in the holder - push it into the A1 corner "
                       "against both stops and re-run")
        return (f"rotation {pl['rotationDeg']:+.2f}°, rect hole centre ({rc[0]:.0f}, {rc[1]:.0f}) µm "
                f"(nominal {fct.PLATE['rectHole']}), size {pl['rectSize'][0]:.0f} x {pl['rectSize'][1]:.0f} µm")

    def _step_backlash(self, s):
        if not s["measureBacklash"]:
            return "skipped"
        r0 = self._ctx["obj"][self._ctx["slot0"]]
        J, d_px = np.asarray(r0["J"]), r0["discPx"]
        via, a = np.array(fct.PLATE["via"], float), float(s["approachUm"])
        out = {}
        for i, ax in enumerate("XY"):
            self._move(*via)                               # arrive from -
            lo = self._disc(d_px)
            pre = via.copy()
            pre[i] += a
            self._move(*pre, approach=0)
            self._move(*via, approach=0)                   # arrive from +
            hi = self._disc(d_px)
            out[ax] = float(abs(np.linalg.solve(J, hi - lo)[i]))
        self._ctx["backlash"] = out
        if s["applyBacklash"]:
            pc = self._master.getController("PixelCalibration")
            for ax, b in out.items():
                pc.applyBacklash(axis=ax, backlashUm=b)
        return f"X {out['X']:.1f} µm, Y {out['Y']:.1f} µm" + (" (applied)" if s["applyBacklash"] else "")

    def _step_objective1(self, s):
        slots = self._slots()
        if not s["secondObjective"] or len(slots) < 2:
            return "skipped"
        k, k0 = slots[1], self._ctx["slot0"]
        r0 = self._ctx["obj"][k0]
        # back on the via with the first objective: absolute repeatability after the long moves
        self._move(*fct.PLATE["via"])
        self._ctx["repeatUm"] = self._residualUm(r0)
        self._move(z=self._pos()["Z"] + s["zRetreatUm"])
        try:
            r = self._calibrateObjective(k, s, fct.PLATE["via"], s["searchRadius2Um"])
        except Exception:
            # the following steps (overview, save) work with the first objective
            self._move(z=self._ctx.get("zTop", r0["z"] + s["zRetreatUm"]))   # homed top: always clear of the sample
            self._selectObjective(k0)
            self._move(z=r0["z"])
            raise
        dev0, dev1 = np.asarray(r0["dev"]), np.asarray(r["dev"])
        offs = [[0.0, 0.0], [0.0, 0.0]]
        offs[k] = (dev1 - dev0).tolist()
        self._setupInfo.objective.xyOffsets = offs
        self._ctx["xyOffsets"] = offs
        # switch back with Z + XY compensation: the via must stay centred and in focus
        self._selectObjective(k0, skipZ=False)
        time.sleep(0.5)
        self._ctx["switchBackUm"] = self._residualUm(r0)
        if self._ctx["switchBackUm"] > 10:
            self._warn(f"after switching back the via is {self._ctx['switchBackUm']:.1f} µm off centre")
        return (self._objSummary(k, r) + f"; parcentricity ({offs[k][0]:+.1f}, {offs[k][1]:+.1f}) µm; "
                f"switch-back residual {self._ctx['switchBackUm']:.1f} µm")

    def _step_overview(self, s):
        ref = self._ctx.get("ovRef")
        if not self._ctx["ov"] or ref is None:
            return "skipped"
        J = np.asarray(self._ctx["overviewJ"]) if "overviewJ" in self._ctx else np.zeros((2, 2))
        z_focus = self._ctx["obj"][self._ctx["slot0"]]["z"]
        # objective back at the homed top, as for the axes frames: at focus it sits under the
        # holes and its reflections shift the round hole's dark area (Pi run 5: 6 px)
        self._move(z=self._ctx.get("zTop", z_focus))
        self._overviewMode(*(self._ctx.get("ovMode") or (s["overviewIlluminationValue"], None)))
        try:
            for dx in (-4000, 0, 4000):         # 3x3 grid, holes stay in the LED cone
                for dy in (-4000, 0, 4000):
                    d = np.array([dx, dy], float)
                    self._holesAt(np.asarray(ref["u"]) + d, near=np.asarray(ref["round"]) + J @ d, max_px=ref["reach"])
        finally:
            self._overviewMode(self._ctx.get("lightMain"), None)
            self._move(z=z_focus)              # from above, like the focus sweep
        off = np.array([self._stageManager().getStageOffsetAxis(axis=a) for a in "XY"])
        samples = [(np.asarray(dev) - off, fct.PLATE["roundHole"], o) for dev, o in self._ctx["ovSamples"]]
        H, rms, n = fct.fit_overview_homography(samples)
        if H is None:
            raise StepError(f"round hole seen in only {len(samples)} overview frames (need 4)")
        self._ctx["overview"] = {"camera": self._ctx["ov"], "H": H.tolist(), "rmsPx": rms, "nPoints": n,
                                 "opticalAxisPx": self._ctx["ovAxis"]}
        if rms > 3:
            self._warn(f"overview mapping residual {rms:.1f} px")
        return f"homography from {n} points, residual {rms:.2f} px"

    def _step_save(self, s):
        k0 = self._ctx["slot0"]
        self._move(*fct.PLATE["via"])
        self._ctx["verifyUm"] = self._residualUm(self._ctx["obj"][k0])
        pc = self._master.getController("PixelCalibration")
        for slot, r in self._ctx["obj"].items():
            m = fct.J_metrics(np.asarray(r["J"]))
            m.update(method="frame_calibration_via_grid", n_inliers=9, rmse_px=r["rmsPx"],
                     rmse_um=r["rmsPx"] * (m["scale_x_um_per_pixel"] + m["scale_y_um_per_pixel"]) / 2,
                     validation_ok=True, validation_message="FRAME calibration plate, via tracked on a 3x3 grid")
            pc.setCalibrationData(detectorName=self._ctx["det"], affineMatrix=fct.affine_for_store(np.asarray(r["J"])),
                                  objectiveId=str(slot), metrics=m)
        self._ctx["saved"] = True
        result = self._summary()
        self._fc()["result"] = result
        self._status["result"] = result
        self._saveSetup()
        if self._ctx["verifyUm"] > 20:
            self._warn(f"going back to the via by plate coordinates leaves it {self._ctx['verifyUm']:.1f} µm off centre")
        return f"saved; via re-found {self._ctx['verifyUm']:.1f} µm from the image centre"

    # ------------------------------------------------- per-objective work
    def _calibrateObjective(self, slot, s, centre, radius):
        oc = self._master.getController("Objective")
        mags = list(getattr(self._setupInfo.objective, "magnifications", None) or [1, 1])
        k0 = self._ctx.get("slot0", slot)
        f = mags[k0] / mags[slot] if mags[slot] else 1.0
        if oc is not None:
            self._selectObjective(slot)
        ps0 = self._pixelSizeGuess(slot, mags)
        img = self._grab()
        self._move(*centre)                 # the sunburst around the via gives the focus contrast
        z = self._focus(slot, s, f)
        # the stage is trusted, the configured magnifications are not (Pi 2026-11-02: "20x" = 2.5x the 4x)
        ps = self._measurePixelSize(slot, centre, min(img.shape[:2]) * ps0) or ps0
        fov = min(img.shape[:2]) * ps
        d_px = fct.PLATE["viaDiameter"] / ps
        self._exposure(s["viaExposureFactor"])
        hit = self._searchVia(centre, radius, 0.7 * fov, d_px)    # overlap: some tile shows the via 0.4 r from its edge
        if hit is None:
            raise StepError(f"centre via not found within {radius:.0f} µm of ({centre[0]:.0f}, {centre[1]:.0f}) - "
                            "plate seated with the A1 notch at the stage origin? light on? try a larger search radius")
        J, rms, u, plate_level, d_px = self._centreAndFit(d_px, fov, s)
        if oc is not None:
            self._saveZ(oc, slot, z)
        # positions from the verified move targets (u, z), not from a fresh read (see _step_offset)
        dev = [u[0] + self._stageManager().getStageOffsetAxis(axis="X"),
               u[1] + self._stageManager().getStageOffsetAxis(axis="Y")]
        r = {"J": J.tolist(), "rmsPx": rms, "u": list(u), "dev": dev, "z": z, "plateLevel": plate_level,
             "discPx": d_px, "metrics": fct.J_metrics(J)}
        nominal = ps0 if slot == k0 else self._ctx["obj"][k0]["metrics"]["scale_x_um_per_pixel"] * f
        measured = r["metrics"]["scale_x_um_per_pixel"]
        if not 0.67 < measured / nominal < 1.5:
            self._warn(f"objective {slot}: {measured:.3f} µm/px is far from the expected {nominal:.3f} µm/px - "
                       "wrong objective in the slot or wrong magnification in the config?")
        self._ctx["obj"][slot] = r
        self._ctx["images"][f"via_objective{slot}"] = self._grab()
        return r

    def _measurePixelSize(self, slot, u, fov):
        """µm/px from +/- stage moves of a tenth of the FOV, tracked by image correlation on the
        focused sunburst. None (and a warning) when the correlation is weak or inconsistent."""
        step = 0.1 * fov
        self._move(*u)
        ref = self._grab()
        shifts = {}
        for i, ax in enumerate("XY"):
            for sign in (1, -1):
                du = np.zeros(2)
                du[i] = sign * step
                self._move(*(np.asarray(u, float) + du))
                sh, resp = fct.image_shift(ref, self._grab())
                if resp < 0.05:
                    self._warn(f"objective {slot}: pixel size not measured (no image correlation) - using the configured value")
                    return None
                shifts[("+" if sign > 0 else "-") + ax] = sh
        self._move(*u)
        J, problems = fct.axes_check(shifts, step)
        m = fct.J_metrics(J)
        if not problems and (abs(m["scale_x_um_per_pixel"] / m["scale_y_um_per_pixel"] - 1) > 0.1 or m["shear_deg"] > 5):
            # square pixels: anything else is 1-D texture (ray edges), correlation blind along it
            problems.append(f"scales {m['scale_x_um_per_pixel']:.3f}/{m['scale_y_um_per_pixel']:.3f} µm/px, "
                            f"shear {m['shear_deg']:.1f}° - too little texture")
        if problems:
            self._warn(f"objective {slot}: pixel size not measured ({'; '.join(problems)}) - using the configured value")
            return None
        ps = (m["scale_x_um_per_pixel"] + m["scale_y_um_per_pixel"]) / 2
        self._ctx.setdefault("pixelSizeMeasured", {})[str(slot)] = ps
        return ps

    def _holePairMid(self):
        return (np.array(fct.PLATE["roundHole"], float) + np.array(fct.PLATE["rectHole"], float)) / 2

    def _searchVia(self, centre, radius, step, d_px):
        self._ctx.pop("brightestTile", None)
        for dx, dy in fct.spiral(radius, step):
            x, y = centre[0] + dx, centre[1] + dy
            self._move(x, y, approach=0)
            img = self._grab()
            hit = fct.find_bright_disc(img, d_px)
            self._ctx["lastSearchImg"] = img
            self._trace(x, y, "tile")
            p = float(np.percentile(img, 99.5))
            if p > self._ctx.get("brightestTile", (-1,))[0]:
                self._ctx["brightestTile"] = (p, img)
            if hit:
                self._trace(x, y, "via")
                self._ctx["images"].setdefault("via_found", self._ctx.get("lastSearchImg"))
                return x, y, hit
        if "brightestTile" in self._ctx:      # report: the tile most likely to show the via
            self._ctx["images"]["via_search_brightest"] = self._ctx.pop("brightestTile")[1]
        return None

    def _centreAndFit(self, d_px, fov, s):
        step = s["gridStepFov"] * fov
        p = self._pos()
        u = self._roughCentre(np.array([p["X"], p["Y"]]), d_px, fov)
        self._disc(d_px, u)
        d_px = self._ctx["lastDisc"]["r"] * 2       # measured size from here on
        img = self._ctx["lastImg"]
        c = np.array([(img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2])
        # tiny steps first (the via may sit at the frame edge), centre, then the real direction check
        for frac in (0.05, 0.2):
            J = self._jacobian(d_px, u, frac * fov)
            for _ in range(6):                      # centre the via on the optical axis (image centre)
                e = c - self._disc(d_px, u)
                if np.linalg.norm(e) < 0.5:
                    break
                u = u + np.linalg.solve(J, e)
        # 3x3 grid around the centre -> final J
        stage, pix = [], []
        for gx in (-1, 0, 1):
            for gy in (-1, 0, 1):
                us = u + np.array([gx, gy]) * step
                pix.append(self._disc(d_px, us))
                stage.append(us)
        J, rms = fct.fit_J(stage, pix)
        self._disc(d_px, u)                          # back to the centre
        img = self._ctx["lastImg"]
        hit = self._ctx["lastDisc"]
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        outside = (xx - hit["x"]) ** 2 + (yy - hit["y"]) ** 2 > (1.5 * hit["r"]) ** 2
        plate_level = float(np.median(fct.gray(img)[outside])) if outside.any() else float(np.median(img))
        return J, rms, u, plate_level, d_px

    def _roughCentre(self, u, d_px, fov):
        """The search hit may sit at the frame edge. Without knowing the image orientation:
        try a 3x3 grid of positions and keep the one with the via nearest the image centre."""
        for frac in (0.3, 0.12):
            best = None
            for gx in (-1, 0, 1):
                for gy in (-1, 0, 1):
                    us = u + np.array([gx, gy]) * frac * fov
                    self._move(*us)
                    img = self._grab()
                    hit = fct.find_bright_disc(img, d_px)
                    self._trace(*us, "rough", hit=None if hit is None else [round(hit["x"]), round(hit["y"]), round(hit["r"])])
                    if hit:
                        e = math.hypot(hit["x"] - img.shape[1] / 2, hit["y"] - img.shape[0] / 2)
                        if best is None or e < best[0]:
                            best = (e, us)
            if best is None:
                raise StepError("the via vanished around the search hit - focus or exposure?")
            u = best[1]
        return u

    def _jacobian(self, d_px, u, step):
        """J from +/- steps on each axis; the via must move antisymmetrically (robust
        direction check: catches stale frames, slipping or swapped axes)."""
        p0 = self._disc(d_px, u)
        shifts = {}
        for i, ax in enumerate("XY"):
            for sign in (1, -1):
                du = np.zeros(2)
                du[i] = sign * step
                shifts[("+" if sign > 0 else "-") + ax] = self._disc(d_px, u + du) - p0
        J, problems = fct.axes_check(shifts, step)
        if problems:
            raise StepError("main camera motion check: " + "; ".join(problems))
        return J

    def _disc(self, d_px, u=None):
        """Move to u (unidirectional approach) and return the via centre in px."""
        if u is not None:
            self._move(*u)
        img = self._grab()
        hit = fct.find_bright_disc(img, d_px)
        if hit is None:
            p = self._pos()
            self._ctx["images"]["via_lost"] = img
            raise StepError(f"lost the via at ({p['X']:.0f}, {p['Y']:.0f}) µm - grid step too large or focus lost")
        self._ctx["lastImg"], self._ctx["lastDisc"] = img, hit
        return np.array([hit["x"], hit["y"]])

    def _residualUm(self, r):
        """Distance (µm) of the via from the image centre at the current position."""
        p = self._disc(r["discPx"])
        img = self._ctx["lastImg"]
        c = np.array([(img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2])
        return float(np.linalg.norm(np.linalg.solve(np.asarray(r["J"]), p - c)))

    def _focus(self, slot, s, f):
        """Sweep Z towards the sample until the focus metric has passed a clear peak, then
        a fine sweep around it; both from the same side (Z backlash). Normal exposure:
        the sunburst rays give the contrast. Start: the last FRAME calibration of this
        slot, the first objective + the configured parfocal offset, or the homed Z."""
        self._autoExposure()
        step = s["focusStepUm"] * f
        hint = self._zHint(slot)
        top = self._ctx.get("zTop", self._pos()["Z"])
        tries = [(hint + 25 * step, 50)] if hint is not None else []
        if slot == self._ctx.get("slot0", slot):     # never with the second (short working distance) objective
            tries.append((top, int(s["focusSearchUm"] / step)))
        for z0, n in tries:
            zs, vals = self._sweepZ(min(z0, top), -step, n, stop_at_peak=True)
            if fct.focus_peaked(vals):
                break
        else:
            if not s["manualFocusFallback"]:
                raise StepError("no focus found: the focus metric shows no clear peak (light, exposure, plate?)")
            self._waitUser("No clear focus found. Focus on the plate (the rays around the centre via) with the "
                           "Z buttons, then press Continue.")
            return self._pos()["Z"]
        zc = zs[int(np.argmax(vals))]
        zs, vals = self._sweepZ(zc + step, -step / 5, 11)
        zb = zs[int(np.argmax(vals))]
        self._move(z=zb + step)
        self._move(z=zb)
        self._ctx.setdefault("focus", {})[str(slot)] = {"z": zb, "stepUm": step}
        return zb

    def _sweepZ(self, z0, dz, n, stop_at_peak=False):
        zs, vals = [], []
        for i in range(n):
            self._move(z=z0 + i * dz)
            zs.append(z0 + i * dz)
            vals.append(fct.focus_metric(self._grab()))
            if stop_at_peak and fct.focus_peaked(vals):
                break
        return zs, vals

    def _zHint(self, slot):
        prev = ((self._fc().get("result") or {}).get("objectives") or {}).get(str(slot))
        if prev and prev.get("z") is not None:
            return float(prev["z"])
        k0, zcfg = self._ctx.get("slot0"), self._ctx.get("zConfig") or []
        if k0 is not None and slot != k0 and k0 in self._ctx["obj"]:
            delta = zcfg[slot] - zcfg[k0] if len(zcfg) > max(slot, k0) else 0.0
            return self._ctx["obj"][k0]["z"] + delta
        return None

    def _saveZ(self, oc, slot, z):
        """Store the focus Z of this slot; on the first objective shift the other slot by
        the same amount so the parfocal offset (the Z follow on objective switches) stays."""
        zcfg = self._ctx.get("zConfig") or []
        oc.saveCurrentZPosition(slot)
        oc._zPositions[slot] = z                        # the verified focus Z, not a fresh read
        oc._configManager.setObjectiveParameters(slot=slot, zPosition=z, emitSignal=True)
        if slot == self._ctx.get("slot0") and len(zcfg) == 2:
            k = 1 - slot
            oc._zPositions[k] = z + zcfg[k] - zcfg[slot]
            oc._configManager.setObjectiveParameters(slot=k, zPosition=oc._zPositions[k], emitSignal=True)

    def _exposure(self, factor):
        """Main-camera exposure relative to the auto exposure (else the user's)."""
        e = self._ctx.get("expBase") or self._ctx.get("exp0")
        if e:
            self._master.detectorsManager[self._ctx["det"]].setParameter("exposure", e * factor)
            time.sleep(0.2)

    def _autoExposure(self):
        """Exposure for focusing: only corrected when the plate and rays (95th percentile)
        are clearly too dark (< 5 % of full scale) or too bright (> 90 %); then towards 30 %,
        at most 100 ms (long exposures stalled the Hik stream). The via work runs at
        viaExposureFactor of it; gain is left alone."""
        e = self._ctx.get("exp0")
        if not e:
            return
        det = self._master.detectorsManager[self._ctx["det"]]
        for _ in range(6):
            det.setParameter("exposure", e)
            time.sleep(0.2)
            raw = self._grab()
            full = 255.0 if raw.dtype == np.uint8 else (4095.0 if raw.max() < 4096 else 65535.0)
            p95 = float(np.percentile(fct.gray(raw), 95))
            if 0.05 * full < p95 < 0.9 * full or (e >= 100 and p95 <= 0.05 * full):
                break
            e = float(np.clip(e * np.clip(0.3 * full / max(p95, 1.0), 0.1, 10), 0.05, 100))
        self._ctx["expBase"] = e

    def _pixelSizeGuess(self, slot, mags):
        k0 = self._ctx.get("slot0")
        if k0 is not None and k0 in self._ctx["obj"] and slot != k0:
            return self._ctx["obj"][k0]["metrics"]["scale_x_um_per_pixel"] * mags[k0] / mags[slot]
        pc = self._master.getController("PixelCalibration")
        if pc is not None:
            sx, sy = pc.getPixelSize(self._ctx["det"], str(slot))
            ps = (abs(sx) + abs(sy)) / 2
            if 0.02 < ps < 20 and ps != 1.0:
                return ps
        sizes = list(getattr(self._setupInfo.objective, "pixelsizes", None) or [])
        return float(sizes[slot]) if slot < len(sizes) and sizes[slot] else 1.0

    def _slots(self):
        oc = self._master.getController("Objective")
        if oc is None:
            return [0]
        configured = oc.getstatus().get("slotConfigured") or [True, True]
        return [k for k in (0, 1) if configured[k]]

    def _selectObjective(self, slot, skipZ=True):
        oc = self._master.getController("Objective")
        if oc is None:
            return
        r = oc.moveToObjective(slot, skipZ=skipZ)
        if not r.get("accepted"):
            raise StepError(f"objective switch to slot {slot} refused: {r.get('reason')}")
        t0 = time.time()
        while time.time() - t0 < 60:
            self._checkAbort()
            st = oc.getstatus()
            if not st.get("isMovingObjective") and st.get("lastCompletedObjectiveRequestId") == r["requestId"]:
                if st.get("lastObjectiveMoveError"):
                    raise StepError(f"objective switch: {st['lastObjectiveMoveError']}")
                return
            time.sleep(0.2)
        raise StepError(f"objective switch to slot {slot} did not finish within 60 s")

    # ------------------------------------------------- hardware helpers
    def _stageManager(self):
        pm = self._master.positionersManager
        return pm[pm.getAllDeviceNames()[0]]

    def _pos(self):
        """Stage position; re-read when the reply looks lost (device reads all zero)."""
        stage = self._stageManager()
        for _ in range(3):
            p = stage.getPosition()
            if any(abs(p.get(a, 0) + stage.getStageOffsetAxis(axis=a)) > 1e-6 for a in ("X", "Y", "Z")):
                return p
            time.sleep(0.2)
        return p

    def _move(self, x=None, y=None, z=None, approach=None, tol=3.0):
        """Blocking absolute move, one axis at a time, verified by reading the position back.
        ``approach`` (default: settings) > 0 arrives at X/Y from the negative side, so the
        backlash is always taken up the same way."""
        s = self._ctx["s"]
        a = s["approachUm"] if approach is None else approach
        if a and (x is not None or y is not None):
            self._move(None if x is None else x - a, None if y is None else y - a, approach=0, tol=tol)
        stage = self._stageManager()
        for ax, v in (("X", x), ("Y", y), ("Z", z)):
            if v is None:
                continue
            for attempt in (1, 2):
                self._checkAbort()
                stage.move(value=float(v), axis=ax, is_absolute=True, is_blocking=True)
                got = self._pos()[ax]
                if abs(got - v) <= tol:
                    break
                self._logger.warning(f"{ax} reads {got:.1f} after a move to {v:.1f} (attempt {attempt})")
            else:
                raise StepError(f"{ax} did not reach {v:.0f} µm (reads {got:.0f} µm) - stall, end stop or lost serial reply")
        time.sleep(s["settleS"])

    def _grab(self, name=None, frames=2, timeout=5.0):
        """A frame exposed after the last move: wait for the frame counter to advance."""
        det = self._master.detectorsManager[name or self._ctx["det"]]
        for owner in (det, getattr(det, "_camera", None)):
            if owner is not None and hasattr(owner, "flushBuffer"):
                try:
                    owner.flushBuffer()
                    break
                except Exception:
                    pass
        for attempt in (1, 2):
            _, n0 = det.getLatestFrame(returnFrameNumber=True)
            t0 = time.time()
            while time.time() - t0 < timeout:
                f, n = det.getLatestFrame(returnFrameNumber=True)
                if f is not None and n is not None and n0 is not None and n >= n0 + frames:
                    return np.array(f)
                self._checkAbort()
                time.sleep(0.01)
            if attempt == 1 and hasattr(det, "stopAcquisition"):
                self._logger.warning(f"{name or self._ctx['det']}: stream stalled, restarting it")
                try:
                    det.stopAcquisition()
                    det.startAcquisition()
                except Exception as e:
                    self._logger.warning(f"stream restart failed: {e}")
        raise StepError(f"{name or self._ctx['det']}: no new frame within {timeout:.0f} s "
                        f"(frame counter stuck at {n0}) - camera stalled or not streaming")

    def _holesAt(self, u, near=None, max_px=None):
        """Move to u and find the holes in the overview image: the pair, or with ``near``
        (px) only the round hole followed from there. Returns {"round": [x, y], ...} or None;
        keeps (device position, round-hole px) as homography samples."""
        self._move(*u, approach=0)
        img = self._grab(self._ctx["ov"])
        if near is None:
            holes = fct.find_plate_holes(img, self._ctx["ovAxis"], self._ctx["ovRadius"])
        else:
            r = fct.track_round_hole(img, self._ctx["ovAxis"], self._ctx["ovRadius"], near, max_px)
            holes = None if r is None else {"round": r}
        n = self._ctx["ovFrames"] = self._ctx.get("ovFrames", 0) + 1      # every frame goes to the report
        self._ctx["images"][f"overview_{n:02d}"] = img
        self._trace(*u, "overview", frame=n, holes=holes, mode=self._ctx.get("ovModeNow"))
        if holes is not None:      # the verified move target, not a fresh read (firmware reports A under X)
            stage = self._stageManager()
            dev = [u[0] + stage.getStageOffsetAxis(axis="X"), u[1] + stage.getStageOffsetAxis(axis="Y")]
            self._ctx["ovSamples"].append((dev, holes["round"]))
        return holes

    def _overviewModes(self, s):
        """(LED value, overview exposure in ms or None = camera auto) to try for the hole
        frames: room light with the LED dimmed (open rig, Mac 2026-10-30), then the full LED
        at a short manual exposure (dark enclosure, Pi 2026-11-02: the LED is off up to
        ~100 and nearly full from ~120, and auto exposure saturates its cone over the holes)."""
        e = float(s["overviewExposureMs"])
        return [(s["overviewIlluminationValue"], None)] + [(self._ctx.get("lightMain"), x) for x in (e, e / 2, 2 * e) if x > 0]

    def _overviewMode(self, light, exposureMs):
        """Set the LED and the overview camera exposure (None = auto). False when the camera
        has no manual exposure (no 'exposure_mode' parameter)."""
        det = self._master.detectorsManager[self._ctx["ov"]]
        params = getattr(det, "parameters", {}) or {}
        if exposureMs and "exposure_mode" not in params:
            return False
        self._light(light)
        if "exposure_mode" in params:
            det.setParameter("exposure_mode", "manual" if exposureMs else "auto")
            if exposureMs:
                det.setParameter("exposure", float(exposureMs))
            time.sleep(0.3)
        self._ctx["ovModeNow"] = [light, exposureMs]
        return True

    def _light(self, value):
        """Set the transmission light (0/None = leave it); the webcam needs a moment to re-expose."""
        name, lm = self._ctx["s"]["illumination"], self._master.lasersManager
        if name and value and name in lm.getAllDeviceNames():
            lm[name].setValue(value)
            time.sleep(1.0)

    # ------------------------------------------------- status / persistence
    def _newStatus(self, state):
        return {"state": state, "step": "", "message": "", "startedAt": datetime.now().isoformat(timespec="seconds"),
                "finishedAt": None, "trace": [], "result": self._status.get("result") if hasattr(self, "_status") else None,
                "steps": [{"name": n, "title": t, "critical": c, "status": "pending", "detail": "", "warnings": [],
                           "seconds": None} for n, t, c in STEPS]}

    def _trace(self, x, y, kind, **extra):
        tr = self._status["trace"]
        tr.append({"x": float(x), "y": float(y), "kind": kind, **extra})
        del tr[:-3000]

    def _warn(self, msg):
        self._logger.warning(msg)
        step = next((x for x in self._status["steps"] if x["status"] == "running"), None)
        if step is not None:
            step["warnings"].append(msg)

    def _waitUser(self, msg):
        self._continue.clear()
        self._status["state"], self._status["message"] = "waiting", msg
        while not self._continue.wait(0.2):
            self._checkAbort()
        self._status["state"], self._status["message"] = "running", ""

    def _checkAbort(self):
        if self._abort.is_set():
            raise _Aborted()

    def _objSummary(self, slot, r):
        m = r["metrics"]
        return (f"objective {slot}: {m['scale_x_um_per_pixel']:.4f} x {m['scale_y_um_per_pixel']:.4f} µm/px, "
                f"rotation {m['rotation_deg']:+.2f}°, shear {m['shear_deg']:.2f}°, fit residual {r['rmsPx']:.2f} px, "
                f"focus Z {r['z']:.0f} µm")

    def _summary(self):
        c = self._ctx
        return {"timestamp": datetime.now().isoformat(timespec="seconds"), "detector": c["det"],
                "objectives": {str(k): {"metrics": r["metrics"], "rmsPx": r["rmsPx"], "z": r["z"]} for k, r in c["obj"].items()},
                **{k: c[k] for k in ("offset", "placement", "backlash", "xyOffsets", "overview", "overviewJ", "focus", "plateShift",
                                     "pixelSizeMeasured",
                                     "repeatUm", "switchBackUm", "verifyUm") if k in c}}

    def _fc(self):
        if self._setupInfo.frameCalibration is None:
            self._setupInfo.frameCalibration = {}
        return self._setupInfo.frameCalibration

    def _settings(self):
        stored = self._fc().get("settings", {})
        return {k: stored.get(k, v) for k, v in DEFAULT_SETTINGS.items()}

    def _saveSetup(self):
        import imswitch.imcontrol.model.configfiletools as configfiletools
        options, _ = configfiletools.loadOptions()
        configfiletools.saveSetupInfo(options, self._setupInfo)

    def _writeReport(self):
        try:
            d = os.path.join(_REPORT_DIR, datetime.now().strftime("%Y%m%d_%H%M%S"))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "report.json"), "w") as fh:
                json.dump({"status": self._status, "settings": self._ctx.get("s")}, fh, indent=2, default=str)
            for name, img in self._ctx.get("images", {}).items():
                img = np.asarray(img)
                if img.dtype not in (np.uint8, np.uint16):
                    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                cv2.imwrite(os.path.join(d, f"{name}.png"), img)      # raw counts (16-bit PNG for 12-bit cameras)
            self._status["reportDir"] = d
        except Exception as e:
            self._logger.warning(f"Could not write the calibration report: {e}")
