"""One-button FRAME calibration with the FRAME calibration plate.

The user puts the plate in the holder (well-label side up, A1 notch at stage 0,0)
and presses start. The routine homes (verified), checks the motion directions,
finds the centre via, focuses, measures the camera<->stage affine per objective
from the tracked via (every measurement position is approached from the same
side, so backlash cannot bias it), sets the stage offset so that stage
coordinates are plate coordinates, checks the plate placement, measures backlash,
calibrates the second objective (parfocality, parcentricity, affine) and fits the
overview-camera homography. Everything is written to the setup JSON.

Alternative target (setting ``target``: "slides"): the 4-slide insert with pinhole
slides in slots 1 and 4. The 0.3 mm pinhole of slot 1 takes the via's place; the other
pinholes replace the rect hole in the placement check; the opaque slides give no focus
texture (the pinhole's light is found first, then focused) and no overview landmark
(motion check and overview mapping are skipped).

Geometry, detectors and fits: frame_calibration_tools.py. Plan:
docs/FRAME_AUTOCALIBRATION_PLAN.md
"""
import io
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
from . import frame_calibration_report as fcr
from . import frame_calibration_tools as fct

_REPORT_DIR = os.path.join(os.path.expanduser("~"), "imswitch_calibrations", "frame_calibration")

DEFAULT_SETTINGS = {
    "target": "plate",           # calibration target: "plate" (FRAME calibration plate) or "slides" (pinhole slides)
    "detector": "",              # "" = PixelCalibration's default camera
    "overviewCamera": "",        # "" = experiment.overviewCameraName, else ObservationCamera
    "illumination": "LED",       # transmission light switched on at start ("" = leave as is)
    "illuminationValue": 0,      # 0 = keep its current value
    "maxGain": True,             # main camera at its maximum gain during the run (short exposures; restored after)
    "autoExposure": True,        # main camera exposure set on the plate (else only a clearly wrong one is corrected)
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
    "pinholeSearchExposureMs": 0.3,  # slides, first objective at max gain: the hole ~65 % of full scale, its halo and ghost below light_spot
    "gridStepFov": 0.25,         # affine grid: +/- this fraction of the short FOV side
    "approachUm": 100,           # unidirectional approach distance, must exceed the backlash
    "zRetreatUm": 500,           # move away from the sample (towards the Z end stop) before switching objectives
    "zTowardsSample": 0,         # Z direction of the focus: 0 = away from the Z end stop (-homeDirectionZ), else +1/-1
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
    _speed = 20000   # stage speed for every calibration move

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._abort = threading.Event()
        self._continue = threading.Event()
        self._thread = None
        self._ctx = {}
        self._figures = []           # report figures of the running/last run: expected vs measured
        self._diskReport = None      # after a restart: the newest report on disk
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
        """Geometry of the selected calibration target, and the targets to choose from."""
        return {**self._T(), "wrongOrientations": fct.WRONG_ORIENTATIONS,
                "targets": {k: t["name"] for k, t in fct.TARGETS.items()}}

    @APIExport()
    def getFrameCalibrationStatus(self) -> dict:
        return self._status

    @APIExport()
    def getFrameCalibrationFigures(self) -> list:
        """Report figures of the running or last run: what was expected, what was measured."""
        return [{k: f.get(k) for k in ("id", "step", "title", "expected", "measured", "ok")} for f in self._lastReport()[2]]

    @APIExport(runOnUIThread=False)
    def getFrameCalibrationFigure(self, figureId: str = ""):
        """One report figure as JPEG (yellow = expected, green = measured)."""
        from fastapi.responses import Response
        f = next((f for f in self._lastReport()[2] if f["id"] == figureId), None)
        if f is None:
            return {"success": False, "error": f"no report figure {figureId!r}"}
        return Response(content=f["jpg"], media_type="image/jpeg")

    @APIExport(runOnUIThread=False)
    def getFrameCalibrationReportPdf(self):
        """The report of the running or last run as PDF: summary, steps, figures."""
        from fastapi.responses import Response
        status, settings, figures = self._lastReport()
        buf = io.BytesIO()
        fcr.write_pdf(buf, status, settings, figures)
        name = "frame_calibration_" + "".join(ch for ch in str(status.get("startedAt") or "") if ch.isalnum()) + ".pdf"
        return Response(content=buf.getvalue(), media_type="application/pdf",
                        headers={"Content-Disposition": f'attachment; filename="{name}"'})

    @APIExport(runOnUIThread=False)
    def getFrameCalibrationReportZip(self):
        """The last report folder as ZIP: report.json, raw 16-bit frames, figures, PDF."""
        from fastapi.responses import Response
        d = self._status.get("reportDir") or self._latestReportDir()
        if not d or not os.path.isdir(d):
            return {"success": False, "error": "no calibration report written yet"}
        return Response(content=fcr.zip_dir(d), media_type="application/zip",
                        headers={"Content-Disposition": f'attachment; filename="frame_calibration_{os.path.basename(d)}.zip"'})

    @APIExport(requestType="POST")
    def startFrameCalibration(self) -> dict:
        if self.isFrameCalibrationRunning():
            return {"success": False, "error": "A FRAME calibration is already running."}
        busy = busy_reasons(self._master, self._logger, ignore=("FrameCalibration",))
        if busy:
            return {"success": False, "error": f"Busy: {', '.join(busy)}."}
        self._abort.clear()
        self._continue.clear()
        self._figures, self._diskReport = [], None
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
            failed = [x["title"] for x in st["steps"] if x["status"] == "failed"]       # optional steps
            st["message"] = "Calibration complete" + (f" - failed: {'; '.join(failed)}" if failed else
                                                      " (with warnings)" if any(x["warnings"] for x in st["steps"]) else "")
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
            if self._ctx.get("gain0") is not None:
                self._master.detectorsManager[self._ctx["det"]].setParameter("gain", self._ctx["gain0"])
            self._cameraStatus("restored to the values before the run")
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
        det = m.detectorsManager[self._ctx["det"]]
        params = getattr(det, "parameters", {}) or {}
        if "exposure" in params:
            self._ctx["exp0"] = float(params["exposure"].value)
        if "gain" in params:                         # restored after the run
            self._ctx["gain0"] = float(params["gain"].value)
            gmax = getattr(params["gain"], "valueMax", None)
            if s["maxGain"] and gmax:                # independent of the user's gain (gain 1 after a restart: black)
                det.setParameter("gain", float(gmax))
                time.sleep(0.3)
        self._cameraStatus("start: " + ("maximum gain, " if s["maxGain"] else "") + "your exposure")
        img = self._grab()
        g = fct.gray(img)
        self._figure("Main camera at the start", img, expected="light reaches the camera: the brightest pixels clearly "
                     "above the black level (the stage may sit over dark plate)",
                     measured=f"mean {g.mean():.0f}, 99.5th percentile {np.percentile(g, 99.5):.0f}, "
                              f"black level {np.percentile(g, 0.1):.0f} counts; exposure {self._ctx.get('exp0')} ms, "
                              f"gain {(self._status.get('camera') or {}).get('gainDb')} dB")
        pc = m.getController("PixelCalibration")
        if pc is not None:
            ok, why = pc._validateCameraIntensity(self._ctx["det"])
            if not ok and "saturat" not in why:      # bright is fine here: the stage may sit over a hole
                g = fct.gray(img)
                if np.percentile(g, 99.5) <= 2 * np.percentile(g, 0.1) + 10 and self._T()["focusTexture"]:
                    # nothing above the black level (over an opaque pinhole slide that is normal)
                    raise StepError(f"{self._ctx['det']}: {why} - is the {name or 'transmission light'} on, "
                                    "exposure and gain ok?")
                # light, but the stage sits over dark plate: the exposure is set at the via;
                # over an opaque pinhole slide a black frame is expected, no warning
                if self._T()["focusTexture"]:
                    self._warn(f"{self._ctx['det']}: {why} at the start position - continuing")
            pc._resetDetectorFlip(self._ctx["det"])     # measure on the raw sensor; flips are re-applied on save
            self._ctx["flipReset"] = True
        out = [f"{self._ctx['det']} {img.shape[1]}x{img.shape[0]}"]
        if self._ctx["ov"]:
            try:
                ov = self._grab(self._ctx["ov"])
                out.append(f"overview {ov.shape[1]}x{ov.shape[0]}")
                self._figure("Overview camera at the start", ov, expected="the plate in view",
                             measured=f"{ov.shape[1]}x{ov.shape[0]} px")
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
        if not self._ctx["ov"] or "roundHole" not in self._T():      # the slides have no overview landmark
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
            roi = [("circle", self._ctx["ovAxis"], self._ctx["ovRadius"], fcr.OTHER, True),
                   ("cross", self._ctx["ovAxis"], 10, fcr.EXPECTED)]
            exp_pair = ("two dark holes, the round one and the larger rect one, inside the search region (cyan) "
                        "around the LED glare (yellow cross, near the optical axis)")
            if p0 is None:
                self._figure("Overview: no hole pair found", self._ctx.get("lastOv"), roi, expected=exp_pair,
                             measured=f"no pair in {len(self._overviewModes(s))} light/exposure settings", ok=False)
                raise StepError("the overview camera does not see the plate holes with the stage between them - "
                                "plate missing or not seated, or the camera moved")
            mode = self._ctx["ovMode"]
            self._ctx["ovRefImg"] = self._ctx["lastOv"]
            self._figure("Overview: hole pair at the reference position", self._ctx["ovRefImg"],
                         roi + [("circle", p0["round"], 14, fcr.MEASURED), ("text", np.add(p0["round"], (16, -8)), "round", fcr.MEASURED),
                                ("circle", p0["rect"], 16, fcr.MEASURED), ("text", np.add(p0["rect"], (18, 4)), "rect", fcr.MEASURED)],
                         expected=exp_pair,
                         measured=f"round {np.round(p0['round']).tolist()} px, rect {np.round(p0['rect']).tolist()} px; "
                                  f"LED {mode[0]}, exposure {f'{mode[1]:g} ms' if mode[1] else 'auto'}", ok=True)
            r0 = np.asarray(p0["round"])
            reach = 0.4 * float(np.linalg.norm(np.asarray(p0["rect"]) - r0))     # less than half way to the rect hole
            self._ctx["ovRef"] = {"u": ref.tolist(), "round": r0.tolist(), "reach": reach}
            shifts, seen = {}, {}
            for i, ax in enumerate("XY"):
                for sign in (1, -1):
                    t = ref.copy()
                    t[i] += sign * step
                    key = ("+" if sign > 0 else "-") + ax
                    p = self._holesAt(t, near=r0, max_px=reach)
                    if p is None:
                        self._figure(f"Overview: round hole lost after {key} {step:.0f} µm", self._ctx["lastOv"],
                                     [("circle", r0, reach, fcr.EXPECTED, True), ("cross", r0, 8, fcr.EXPECTED)],
                                     expected=f"the round hole within {reach:.0f} px of its reference position (yellow)",
                                     measured="no hole-sized dark blob there", ok=False)
                        raise StepError(f"round hole lost in the overview image after {sign * step:+.0f} µm in {ax}")
                    seen[key] = np.asarray(p["round"])
                    shifts[key] = seen[key] - r0
        finally:
            self._overviewMode(self._ctx.get("lightMain"), None)
        J, problems = fct.axes_check(shifts, step, min_px=2.0)
        ref_J = (self._fc().get("result") or {}).get("overviewJ")
        if ref_J and not problems:
            change = fct.direction_change_deg(J, ref_J)
            if change > 25:
                problems.append(f"image directions changed by {change:.0f}° since the last calibration - "
                                "axis/sign config changed or the overview camera moved")
        marks, k = [("circle", r0, 8, fcr.MEASURED)], 3.0          # shifts drawn 3x: ~11 px at 640 px
        for key, q in seen.items():
            opposite = ("-" if key[0] == "+" else "+") + key[1]
            marks += [("arrow", r0, r0 + k * (q - r0), fcr.MEASURED), ("text", r0 + k * 1.15 * (q - r0) + (-8, 5), key, fcr.MEASURED),
                      ("circle", r0 - k * (seen[opposite] - r0), 5, fcr.EXPECTED, True)]
        self._figure(f"Overview: round hole for +/-{step:.0f} µm moves", self._ctx["ovRefImg"], marks,
                     expected="each move and its opposite shift the hole by equal and opposite amounts (yellow: the "
                              "mirror of the opposite move); X and Y in clearly different directions. Shifts drawn 3x.",
                     measured="; ".join(f"{k} ({v[0]:+.1f}, {v[1]:+.1f}) px" for k, v in shifts.items())
                              + ("" if not problems else " - " + "; ".join(problems)), ok=not problems)
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
        centre = np.add(self._T()["via"], self._ctx.get("plateShift", (0, 0)))
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
            posc.setStageOffsetAxis(knownPosition=float(self._T()["via"][i]),
                                    currentDevicePosition=float(r0["u"][i] + old[i]), axis=ax)
        new = np.array([stage.getStageOffsetAxis(axis=a) for a in "XY"], float)
        shift = old - new                  # user_new = user_old + (old - new)
        for t in self._status["trace"]:
            t["x"], t["y"] = t["x"] + shift[0], t["y"] + shift[1]
        for r in self._ctx["obj"].values():
            r["u"] = (np.asarray(r["u"]) + shift).tolist()
        if "ovRef" in self._ctx:                     # the overview grid is placed around it
            self._ctx["ovRef"]["u"] = (np.asarray(self._ctx["ovRef"]["u"]) + shift).tolist()
        found = np.asarray(r0["u"]) - np.array(self._T()["via"])   # where the old offset expected it
        self._ctx["offset"] = {"offset": new.tolist(), "viaFoundAtUm": (found - shift).tolist()}
        return (f"offset ({new[0]:.0f}, {new[1]:.0f}) µm; the via was "
                f"({found[0] - shift[0]:+.0f}, {found[1] - shift[1]:+.0f}) µm from where the previous offset expected it")

    def _step_placement(self, s):
        if not s["placementCheck"]:
            return "skipped"
        if "checkPoints" in self._T():
            return self._placementByPinholes(s, self._T())
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
        edges, views = [], []
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
            views.append((img, u, k, edges[-1], rect[k] + (dx if k == 0 else dy)))
        rc = [(edges[0] + edges[1]) / 2, (edges[2] + edges[3]) / 2]
        pl = fct.placement(fct.PLATE["via"], rc)
        pl["rectSize"] = [edges[1] - edges[0], edges[3] - edges[2]]
        tiles = []
        for img, u, k, e, nominal in views:
            c = np.array([(img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2])
            L = 0.4 * min(img.shape[:2]) * r0["metrics"]["scale_x_um_per_pixel"]
            px = lambda P: c - np.asarray(J) @ (np.asarray(P, float) - np.asarray(u, float))
            seg = lambda v: [px((v, u[1] - L)), px((v, u[1] + L))] if k == 0 else [px((u[0] - L, v)), px((u[0] + L, v))]
            tiles.append(fcr.annotate(img, [("line", *seg(nominal), fcr.EXPECTED), ("line", *seg(e), fcr.MEASURED)], width=480))
        self._figure("Plate placement: rect hole edges", annotated=fcr.montage(tiles),
                     expected=f"edges of the {w:.0f} x {h:.0f} µm hole at their nominal plate positions (yellow); "
                              "order: left, right, top, bottom",
                     measured=f"edges (green) {', '.join(f'{e:.0f}' for e in edges)} µm: size {pl['rectSize'][0]:.0f} x "
                              f"{pl['rectSize'][1]:.0f} µm, rotation {pl['rotationDeg']:+.2f}°",
                     ok=abs(pl["rotationDeg"]) <= s["maxRotationDeg"])
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
        via, a = np.array(self._T()["via"], float), float(s["approachUm"])
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
        self._move(*self._T()["via"])
        self._ctx["repeatUm"] = self._residualUm(r0)
        self._move(z=self._at("Z") - self._zIn() * s["zRetreatUm"])
        # landmark: the via, or on the slides the smallest pinhole the placement check found (a
        # 0.3 mm one is ~2000 px wide at 20x: the frame height). Through the first objective it
        # sits at its measured plate position: that is where it is centred (device: + offset).
        land, diam, dev0 = self._T()["via"], None, np.asarray(r0["dev"])
        small = min(((v["plate"], v["diameterUm"]) for n, v in ((self._ctx.get("placement") or {}).get("pinholes") or {}).items()
                     if n.startswith("slot 1")), key=lambda t: t[1], default=None)
        if small is not None:
            land, diam = small
            dev0 = np.asarray(land) + [self._stageManager().getStageOffsetAxis(axis=a) for a in "XY"]
        try:
            # search where the stored parcentricity puts it (an objective switch moves the stage
            # by offs[k] - offs[k0]): the 20x tiles are small, the nominal via ~1.5 mm off (Pi)
            offs = getattr(self._setupInfo.objective, "xyOffsets", None) or [[0, 0], [0, 0]]
            centre = np.add(land, np.subtract(offs[k], offs[k0]))
            r = self._calibrateObjective(k, s, centre, s["searchRadius2Um"], diameter_um=diam)
        except Exception:
            # the following steps (overview, save) work with the first objective
            self._move(z=self._ctx.get("zTop", r0["z"] - self._zIn() * s["zRetreatUm"]))   # homed top: always clear of the sample
            self._selectObjective(k0)
            self._move(z=r0["z"])
            raise
        dev1 = np.asarray(r["dev"])
        offs = [[0.0, 0.0], [0.0, 0.0]]
        offs[k] = (dev1 - dev0).tolist()
        self._setupInfo.objective.xyOffsets = offs
        self._ctx["xyOffsets"] = offs
        # switch back with Z + XY compensation: the landmark must stay centred and in focus
        self._selectObjective(k0, skipZ=False)
        time.sleep(0.5)
        self._ctx["switchBackUm"] = self._residualUm(r0)
        self._viaFigure(f"Back on objective {k0} with the stored offsets", r0, self._ctx["switchBackUm"], 10)
        if self._ctx["switchBackUm"] > 10:
            self._warn(f"after switching back the via is {self._ctx['switchBackUm']:.1f} µm off centre - an objective "
                       "switch's X/Y compensation that reverses an axis adds its backlash (see 'Backlash X/Y'; "
                       "'Apply backlash' compensates it)")
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
        u0 = np.asarray(ref["u"])
        proj = lambda P: fct.plate_to_overview_px(H, P, u0)
        outline = [proj(P) for P in fct.PLATE["outline"]]
        rc, (rw, rh) = fct.PLATE["rectHole"], fct.PLATE["rectHoleSize"]
        rect = [proj((rc[0] + sx * rw / 2, rc[1] + sy * rh / 2)) for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
        marks = [("line", a, b, fcr.EXPECTED) for poly in (outline, rect) for a, b in zip(poly, poly[1:] + poly[:1])]
        marks += [("cross", proj(fct.PLATE["roundHole"]), 8, fcr.EXPECTED)]
        for (u, P, o) in samples:     # where each tracked hole was seen, against the fitted mapping
            marks += [("circle", o, 4, fcr.MEASURED), ("cross", fct.plate_to_overview_px(H, P, u), 4, fcr.EXPECTED)]
        self._figure("Overview camera mapping", self._ctx.get("ovRefImg"), marks,
                     expected="the plate outline, the rect hole and the round hole (yellow) on the plate seen at the "
                              "reference position; every tracked hole position on the fitted mapping (yellow crosses)",
                     measured=f"{n} of {len(samples)} positions used, residual {rms:.2f} px", ok=rms <= 3)
        if rms > 3:
            self._warn(f"overview mapping residual {rms:.1f} px")
        return f"homography from {n} points, residual {rms:.2f} px"

    def _step_save(self, s):
        k0 = self._ctx["slot0"]
        self._move(*self._T()["via"])
        self._ctx["verifyUm"] = self._residualUm(self._ctx["obj"][k0])
        self._viaFigure("Via re-found by its plate coordinates", self._ctx["obj"][k0], self._ctx["verifyUm"], 20)
        pc = self._master.getController("PixelCalibration")
        for slot, r in self._ctx["obj"].items():
            m = fct.J_metrics(np.asarray(r["J"]))
            m.update(method="frame_calibration_via_grid", n_inliers=9, rmse_px=r["rmsPx"],
                     rmse_um=r["rmsPx"] * (m["scale_x_um_per_pixel"] + m["scale_y_um_per_pixel"]) / 2,
                     validation_ok=True, validation_message=f"{self._T()['name']}: via tracked on a 3x3 grid")
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
    def _calibrateObjective(self, slot, s, centre, radius, diameter_um=None):
        oc = self._master.getController("Objective")
        mags = list(getattr(self._setupInfo.objective, "magnifications", None) or [1, 1])
        k0 = self._ctx.get("slot0", slot)
        f = mags[k0] / mags[slot] if mags[slot] else 1.0
        if oc is not None:
            self._selectObjective(slot)
        ps0 = self._pixelSizeGuess(slot, mags)
        img = self._grab()
        if self._T()["focusTexture"]:
            self._move(*centre)             # the sunburst around the via gives the focus contrast
        else:                               # opaque slide: the pinhole's light is all there is
            centre = self._findLight(slot, centre, radius, 0.7 * min(img.shape[:2]) * ps0, s, f)
        z = self._focus(slot, s, f)
        if not self._T()["focusTexture"]:   # via work relative to the focused pinhole, not a plate
            self._ctx["expBase"] = self._exposeOnLight(0.5) / s["viaExposureFactor"]
        # the stage is trusted, the configured magnifications are not (Pi 2026-11-02: "20x" = 2.5x the 4x)
        ps = self._measurePixelSize(slot, centre, min(img.shape[:2]) * ps0) or ps0
        fov = min(img.shape[:2]) * ps
        d_px = (diameter_um or self._T()["viaDiameter"]) / ps
        r0 = self._ctx["obj"].get(k0) if slot != k0 else None
        if r0 and r0.get("viaExposureMs"):
            # the via's brightness per pixel goes with 1/magnification^2: from the first objective's via
            # exposure, not from this objective's plate (10x, Pi 2026-11-03: the dark plate gave 10 ms
            # instead of ~0.3 ms, and ray tips came out as four false via candidates)
            e = r0["viaExposureMs"] * (r0["metrics"]["scale_x_um_per_pixel"] / ps) ** 2
            self._master.detectorsManager[self._ctx["det"]].setParameter("exposure", e)
            time.sleep(0.2)
            self._cameraStatus(f"via work at objective {k0}'s via exposure x (magnification ratio)^2")
        else:
            self._exposure(s["viaExposureFactor"])
        via_exposure = (self._status.get("camera") or {}).get("exposureMs")

        def centre_on(x, y, hit, img):
            self._viaSaturate(img, hit, via_exposure)
            try:
                return self._centreAndFit(d_px, fov, s, np.array([x, y], float))
            except StepError:
                self._setExposure(via_exposure, "via search")      # not the via: search on, rays dark
                raise
        fit = self._searchVia(centre, radius, 0.7 * fov, d_px, centre_on)   # overlap: some tile shows the via 0.4 r from its edge
        if fit is None:
            raise StepError(f"centre via not found within {radius:.0f} µm of ({centre[0]:.0f}, {centre[1]:.0f}) - "
                            "plate seated with the A1 notch at the stage origin? light on? try a larger search radius")
        J, rms, u, plate_level, d_px = fit
        img, disc = self._ctx["lastImg"], self._ctx["lastDisc"]
        c = ((img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2)
        S, Pp = (np.asarray(a, float) for a in self._ctx["grid"])
        pred = Pp.mean(axis=0) + (J @ (S - S.mean(axis=0)).T).T
        m = fct.J_metrics(J)
        self._figure(f"Via centred and 3x3 grid, objective {slot}", img,
                     [("cross", c, 40, fcr.EXPECTED), ("circle", (disc["x"], disc["y"]), disc["r"], fcr.MEASURED)]
                     + [("circle", q, 12, fcr.MEASURED) for q in Pp] + [("cross", q, 16, fcr.EXPECTED) for q in pred],
                     expected="the via at the image centre (yellow cross); its positions on the 3x3 stage grid on the "
                              "fitted affine (yellow crosses)",
                     measured=f"{m['scale_x_um_per_pixel']:.4f} x {m['scale_y_um_per_pixel']:.4f} µm/px, rotation "
                              f"{m['rotation_deg']:+.2f}°, shear {m['shear_deg']:.2f}°, fit residual {rms:.2f} px, "
                              f"via ⌀ {d_px:.0f} px", ok=True)
        if oc is not None:
            self._saveZ(oc, slot, z)
        # positions from the verified move targets (u, z), not from a fresh read (see _step_offset)
        dev = [u[0] + self._stageManager().getStageOffsetAxis(axis="X"),
               u[1] + self._stageManager().getStageOffsetAxis(axis="Y")]
        r = {"J": J.tolist(), "rmsPx": rms, "u": list(u), "dev": dev, "z": z, "plateLevel": plate_level,
             "discPx": d_px, "metrics": fct.J_metrics(J), "viaExposureMs": via_exposure,
             "centreExposureMs": (self._status.get("camera") or {}).get("exposureMs")}
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
        c = np.array([(ref.shape[1] - 1) / 2, (ref.shape[0] - 1) / 2])
        expect = f"+/-{step:.0f} µm moves shift the image by {step * min(ref.shape[:2]) / fov:.0f} px (yellow circle) " \
                 f"at the configured {fov / min(ref.shape[:2]):.4f} µm/px"
        marks = lambda sh: [("circle", c, step * min(ref.shape[:2]) / fov, fcr.EXPECTED, True)] + \
            [m for k, v in sh.items() for m in (("arrow", c, c + v, fcr.MEASURED), ("text", c + 1.1 * v, k, fcr.MEASURED))]
        shifts = {}
        for i, ax in enumerate("XY"):
            for sign in (1, -1):
                du = np.zeros(2)
                du[i] = sign * step
                self._move(*(np.asarray(u, float) + du))
                sh, resp = fct.image_shift(ref, self._grab())
                if resp < 0.05:
                    self._warn(f"objective {slot}: pixel size not measured (no image correlation) - using the configured value")
                    self._figure(f"Pixel size from stage moves, objective {slot}", ref, marks(shifts), expected=expect,
                                 measured=f"no image correlation after {sign * step:+.0f} µm in {ax}", ok=False)
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
            self._figure(f"Pixel size from stage moves, objective {slot}", ref, marks(shifts), expected=expect,
                         measured="; ".join(problems), ok=False)
            return None
        ps = (m["scale_x_um_per_pixel"] + m["scale_y_um_per_pixel"]) / 2
        self._ctx.setdefault("pixelSizeMeasured", {})[str(slot)] = ps
        ps0 = fov / min(ref.shape[:2])
        self._figure(f"Pixel size from stage moves, objective {slot}", ref, marks(shifts), expected=expect,
                     measured=f"{ps:.4f} µm/px, {ps0 / ps:.2f}x the configured magnification", ok=True)
        return ps

    def _findLight(self, slot, centre, radius, step, s, f):
        """Opaque target (pinhole slides): no texture to focus on. At the focus sweep's start
        height, spiral to the first tile with light, bring the light near the image centre by
        trial (no J yet) and set the exposure on it. Returns that stage position.

        The spiral runs short: on the Pi rig (2026-11-04) the hole stays a crisp disc over 5 mm of
        Z, but at a longer exposure the frame ~1 FOV beside it shows only its halo, cut by the edge,
        and a ~1/260 ghost of it that moves against the stage. Both pulled the light off the hole."""
        hint = self._zHint(slot)
        top = self._ctx.get("zTop", self._pos()["Z"])
        self._move(z=top if hint is None else self._notPast(hint, top))
        r0 = self._ctx["obj"].get(self._ctx.get("slot0")) if slot != self._ctx.get("slot0") else None
        self._setExposure(r0["viaExposureMs"] / f ** 2 if r0 and r0.get("viaExposureMs") else s["pinholeSearchExposureMs"],
                          "pinhole search: only the hole itself stands out")
        spot = lambda img: (lambda p: None if p is None else {"x": p[0], "y": p[1]})(fct.light_spot(img))
        e0 = float(self._master.detectorsManager[self._ctx["det"]].parameters["exposure"].value)
        for e in dict.fromkeys((e0, min(10 * e0, 100.0))):
            # second pass longer: a far-defocused pinhole is a dim blur, and an 8-bit camera
            # (FRAME 10, Hik RGB, 2026-11-12) has few counts above black at the short exposure
            if e != e0:
                self._setExposure(e, "pinhole search, second pass: longer exposure for a dim, defocused hole")
            u = self._spiralForLight(centre, radius, step, spot)
            if u is not None:
                break
        else:
            raise StepError(f"no light from the pinhole within {radius:.0f} µm of ({centre[0]:.0f}, {centre[1]:.0f}) µm - "
                            "pinhole slide in slot 1, LED on? try a larger search radius")
        u = self._roughCentre(u, 0, step / 0.7, find=spot, fracs=(0.3,))
        self._move(*u)                 # _roughCentre ends on its last trial position
        e = self._exposeOnLight(0.4)
        img = self._ctx["lastLightImg"]
        p = spot(img)
        self._figure(f"Pinhole light found, objective {slot}", img,
                     [("cross", (p["x"], p["y"]), 20, fcr.MEASURED)] if p else [],
                     expected="the (defocused) light of the 0.3 mm pinhole near the image centre",
                     measured=f"at stage ({u[0]:.0f}, {u[1]:.0f}) µm, Z {self._at('Z'):.0f} µm; exposure {e:.2f} ms",
                     ok=True)
        return u

    def _spiralForLight(self, centre, radius, step, spot):
        """First spiral tile (stage position) whose frame shows light, or None."""
        for dx, dy in fct.spiral(radius, step):
            u = np.array([centre[0] + dx, centre[1] + dy])
            self._move(*u, approach=0)
            img = self._grab()
            self._trace(*u, "tile")
            if spot(img):
                return u
        return None

    def _exposeOnLight(self, target):
        """Main-camera exposure with the brightest pixels (99.9th percentile) at ~``target``
        of full scale; returns it (ms)."""
        det = self._master.detectorsManager[self._ctx["det"]]
        e = float(det.parameters["exposure"].value)
        for _ in range(8):
            raw = self._ctx["lastLightImg"] = self._grab()
            full = 255.0 if raw.dtype == np.uint8 else (4095.0 if raw.max() < 4096 else 65535.0)
            p = float(np.percentile(fct.gray(raw), 99.9))
            if 0.6 * target * full < p < min(1.5 * target, 0.95) * full:
                break
            e = float(np.clip(e * (0.25 if p >= 0.95 * full else np.clip(target * full / max(p, 1.0), 0.1, 10)), 0.02, 100))
            self._setExposure(e, f"exposure set on the pinhole: brightest pixels towards {100 * target:.0f} % of full scale")
        return e

    def _placementByPinholes(self, s, T):
        """Pinhole slides: find the check-point pinholes (slot 4's centre one, each slide's outer
        two) in single frames, their plate positions from the first objective's camera matrix.
        Slot 4's centre gives the insert rotation and the stage scale over 90 mm; each slide's
        outer pair its rotation in the slot, and their sizes (0.2 / 0.4 mm) how it was inserted."""
        r0 = self._ctx["obj"][self._ctx["slot0"]]
        J, d_px = np.asarray(r0["J"]), r0["discPx"]
        ps = (r0["metrics"]["scale_x_um_per_pixel"] + r0["metrics"]["scale_y_um_per_pixel"]) / 2
        fov = min(self._ctx["lastImg"].shape[:2]) * ps
        if r0.get("viaExposureMs"):
            self._setExposure(r0["viaExposureMs"], "pinhole search at the via search exposure")
        found, tiles = {}, []
        for name, P in T["checkPoints"].items():
            hit = None
            for dx, dy in fct.spiral(s["searchRadiusUm"] / 2, 0.7 * fov):
                u = (P[0] + dx, P[1] + dy)
                self._move(*u, approach=0)
                self._trace(*u, "tile")
                if fct.find_bright_disc(self._grab(), d_px):
                    self._move(*u)                 # again from the negative side: backlash taken up
                    img = self._grab()
                    hit = fct.find_bright_disc(img, d_px)
                    if hit:
                        break
            if hit is None:
                self._warn(f"pinhole '{name}' not found within {s['searchRadiusUm'] / 2:.0f} µm of {P} µm")
                continue
            q = fct.px_to_plate((hit["x"], hit["y"]), J, u, img.shape[:2])[0]
            found[name] = {"plate": q.tolist(), "offsetUm": (q - np.asarray(P, float)).tolist(), "diameterUm": 2 * hit["r"] * ps}
            self._trace(*u, "edge")
            c = np.array([(img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2])
            tiles.append(fcr.annotate(img, [("cross", c - J @ (np.asarray(P, float) - np.asarray(u, float)), 30, fcr.EXPECTED),
                                            ("circle", (hit["x"], hit["y"]), hit["r"], fcr.MEASURED),
                                            ("text", (20, 40), name, fcr.MEASURED)], width=480))
        pl = {"pinholes": found}
        cp, via = T["checkPoints"], T["via"]
        if "slot 4, centre" in found:
            pl.update(fct.placement(via, found["slot 4, centre"]["plate"], cp["slot 4, centre"], via))
        pl["slides"] = {}
        for k in T["loadedSlots"]:
            lo, hi = found.get(f"slot {k}, -19 mm"), found.get(f"slot {k}, +19 mm")
            if lo and hi:
                pl["slides"][str(k)] = {
                    "rotationDeg": fct.placement(lo["plate"], hi["plate"], cp[f"slot {k}, +19 mm"], cp[f"slot {k}, -19 mm"])["rotationDeg"],
                    "smallEnd": "-y" if lo["diameterUm"] < hi["diameterUm"] else "+y"}      # where the 0.2 mm pinhole is
        rot = pl.get("rotationDeg")
        self._figure("Placement: pinholes of the slides", annotated=fcr.montage(tiles) if tiles else None,
                     expected="each check-point pinhole at its nominal plate position (yellow cross)",
                     measured="; ".join(f"{n}: {v['offsetUm'][0]:+.0f}, {v['offsetUm'][1]:+.0f} µm, ⌀ {v['diameterUm']:.0f} µm"
                                        for n, v in found.items()) or "none found",
                     ok=rot is not None and abs(rot) <= s["maxRotationDeg"])
        self._ctx["placement"] = pl
        if rot is None:
            raise StepError("slot 4's pinhole not found - is the second pinhole slide in slot 4?")
        if abs(rot) > s["maxRotationDeg"]:
            self._warn(f"the slide insert is rotated {rot:.2f}° against the stage axes")
        return (f"insert rotation {rot:+.2f}°, stage scale over 90 mm {pl['baselineScale']:.5f}"
                + "".join(f"; slot {k}: 0.2 mm pinhole at {v['smallEnd']}, slide rotation {v['rotationDeg']:+.2f}°"
                          for k, v in pl["slides"].items()))

    def _holePairMid(self):
        return (np.array(fct.PLATE["roundHole"], float) + np.array(fct.PLATE["rectHole"], float)) / 2

    def _searchVia(self, centre, radius, step, d_px, centre_on):
        """Spiral of tiles around ``centre``. A disc found is the via only if it can be
        centred (``centre_on(x, y, hit, img)`` returns the fit): a ray tip cut by the frame edge looks
        like a disc too (10x, Pi 2026-11-03: 9x the plate vs 35x for the via), so a candidate
        that cannot be centred is reported and the search goes on."""
        self._ctx.pop("brightestTile", None)
        rejected = []
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
                h = (hit["x"], hit["y"])
                self._figure(f"Via search: candidate {len(rejected) + 1}", img,
                             [("circle", h, hit["r"], fcr.MEASURED), ("circle", h, d_px / 2, fcr.EXPECTED, True)],
                             expected=f"a bright disc of ⌀ {d_px:.0f} px (yellow)",
                             measured=f"⌀ {2 * hit['r']:.0f} px at ({hit['x']:.0f}, {hit['y']:.0f}) px, "
                                      f"{hit['contrast']:.0f}x the plate; stage ({x:.0f}, {y:.0f}) µm")
                try:
                    return centre_on(x, y, hit, img)
                except StepError as e:
                    rejected.append(f"({x:.0f}, {y:.0f})")
                    self._warn(f"via candidate at ({x:.0f}, {y:.0f}) µm could not be centred ({e}) - not the via, searching on")
                    if len(rejected) >= 5:
                        raise StepError(f"{len(rejected)} via candidates could not be centred: {', '.join(rejected)} µm - "
                                        "focus, exposure, or the search radius?")
        if "brightestTile" in self._ctx:      # report: the tile most likely to show the via
            img = self._ctx["images"]["via_search_brightest"] = self._ctx.pop("brightestTile")[1]
            self._figure("Via search: brightest tile (no via found)", img,
                         [("circle", ((img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2), d_px / 2, fcr.EXPECTED, True)],
                         expected=f"a bright disc of ⌀ {d_px:.0f} px (yellow) in one of the tiles",
                         measured=f"none within {radius:.0f} µm of ({centre[0]:.0f}, {centre[1]:.0f}) µm", ok=False)
        return None

    def _centreAndFit(self, d_px, fov, s, u):
        """From the search tile u (its verified position; a fresh read once gave X -26 instead
        of 63910, Pi 2026-11-03): centre the via, then J from a 3x3 grid."""
        step = s["gridStepFov"] * fov
        u = self._roughCentre(u, d_px, fov)
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
        self._ctx["grid"] = (stage, pix)
        self._disc(d_px, u)                          # back to the centre
        img = self._ctx["lastImg"]
        hit = self._ctx["lastDisc"]
        yy, xx = np.ogrid[:img.shape[0], :img.shape[1]]
        outside = (xx - hit["x"]) ** 2 + (yy - hit["y"]) ** 2 > (1.5 * hit["r"]) ** 2
        plate_level = float(np.median(fct.gray(img)[outside])) if outside.any() else float(np.median(img))
        return J, rms, u, plate_level, d_px

    def _roughCentre(self, u, d_px, fov, find=None, fracs=(0.3, 0.12)):
        """The search hit may sit at the frame edge. Without knowing the image orientation:
        try a 3x3 grid of positions and keep the one with the via (``find(img)`` -> dict with
        x, y, or None) nearest the image centre."""
        find = find or (lambda img: fct.find_bright_disc(img, d_px))
        for frac in fracs:
            best = None
            for gx in (-1, 0, 1):
                for gy in (-1, 0, 1):
                    us = u + np.array([gx, gy]) * frac * fov
                    self._move(*us)
                    img = self._grab()
                    hit = find(img)
                    self._trace(*us, "rough", hit=None if hit is None else [round(hit["x"]), round(hit["y"]), round(hit.get("r", 0))])
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
            x, y = self._at("X"), self._at("Y")
            self._ctx["images"]["via_lost"] = img
            self._figure("Via lost", img, [("cross", ((img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2), 40, fcr.EXPECTED)],
                         expected=f"a bright disc of ⌀ {d_px:.0f} px in the frame", ok=False,
                         measured=f"no disc at stage ({x:.0f}, {y:.0f}) µm")
            raise StepError(f"lost the via at ({x:.0f}, {y:.0f}) µm - grid step too large or focus lost")
        self._ctx["lastImg"], self._ctx["lastDisc"] = img, hit
        return np.array([hit["x"], hit["y"]])

    def _viaFigure(self, title, r, residual_um, limit_um):
        img, d = self._ctx["lastImg"], self._ctx["lastDisc"]
        c = ((img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2)
        self._figure(title, img, [("cross", c, 40, fcr.EXPECTED), ("circle", (d["x"], d["y"]), d["r"], fcr.MEASURED)],
                     expected=f"the via at the image centre (yellow cross), within {limit_um} µm",
                     measured=f"{residual_um:.1f} µm from the centre", ok=residual_um <= limit_um)

    def _setExposure(self, e, note):
        self._master.detectorsManager[self._ctx["det"]].setParameter("exposure", float(e))
        time.sleep(0.2)
        self._cameraStatus(note)

    def _viaSaturate(self, img, hit, e):
        """Centring and the grid run on a saturated via, ~5x full scale from the candidate's
        core brightness (at most the plate exposure): its edge is then the hole wall. The core
        is not flat; unsaturated, the 10x via came out 799 px wide with a 25 px circle-fit error
        instead of 1000 px and 6 px (Pi 2026-11-03). The search itself stays short: rays dark."""
        g = fct.gray(img)
        full = 255.0 if img.dtype == np.uint8 else (4095.0 if img.max() < 4096 else 65535.0)
        yy, xx = np.ogrid[:g.shape[0], :g.shape[1]]
        core = g[(xx - hit["x"]) ** 2 + (yy - hit["y"]) ** 2 < (0.5 * hit["r"]) ** 2]
        med = float(np.median(core)) if core.size else full
        e_c = min(e * max(1.0, 5 * full / max(med, 1.0)), self._ctx.get("expBase") or e)
        self._setExposure(e_c, f"via centring at {e_c / e:.1f}x the search exposure (via saturated, sharp edge)")
        return e_c

    def _residualUm(self, r):
        """Distance (µm) of the via from the image centre at the current position."""
        if r.get("centreExposureMs"):          # the exposure the via was centred and measured at
            self._setExposure(r["centreExposureMs"], "via check at the objective's centring exposure")
        p = self._disc(r["discPx"])
        img = self._ctx["lastImg"]
        c = np.array([(img.shape[1] - 1) / 2, (img.shape[0] - 1) / 2])
        return float(np.linalg.norm(np.linalg.solve(np.asarray(r["J"]), p - c)))

    def _focus(self, slot, s, f):
        """Sweep Z towards the sample until the focus metric has passed a clear peak, then
        a fine sweep around it; both from the same side (Z backlash). Normal exposure:
        the sunburst rays give the contrast. Start: the last FRAME calibration of this
        slot, the first objective + the configured parfocal offset, or the homed Z.
        Slides, first objective: the whole range, then the highest peak. A hole through the PCB
        has two rims (Pi rig 2026-11-04: Z -2300 and -3970), and the first one met is the weaker.
        The second objective always stops at the peak: short working distance, and its window
        (+/- 25 steps around the hint) does not reach the other rim."""
        if self._T()["focusTexture"]:
            self._autoExposure()            # on the plate; the slides: set on the pinhole's light (_findLight)
        step = s["focusStepUm"] * f
        hint = self._zHint(slot)
        top = self._ctx.get("zTop", self._pos()["Z"])
        zin = self._zIn()                   # towards the sample: away from the homed Z end stop
        tries = [(hint - zin * 25 * step, 50)] if hint is not None else []
        if slot == self._ctx.get("slot0", slot):     # never with the second (short working distance) objective
            tries.append((top, int(s["focusSearchUm"] / step)))
        curves = []
        for z0, n in tries:
            zs, vals = self._sweepZ(self._notPast(z0, top), zin * step, n,
                                    stop_at_peak=self._T()["focusTexture"] or slot != self._ctx.get("slot0", slot))
            curves.append((zs, vals, f"coarse from {self._notPast(z0, top):.0f}"))
            if fct.focus_peaked(vals):
                break
        else:
            self._figure(f"Focus sweep, objective {slot}", annotated=fcr.plot_curves(curves, "Z (µm)", "focus metric"),
                         expected="one clear peak of the focus metric (the rays around the via come into focus)",
                         measured="no clear peak", ok=False)
            if not s["manualFocusFallback"]:
                raise StepError("no focus found: the focus metric shows no clear peak (light, exposure, plate?)")
            self._waitUser("No clear focus found. Focus on the " + ("plate (the rays around the centre via)"
                           if self._T()["focusTexture"] else "pinhole of the slide in slot 1")
                           + " with the Z buttons, then press Continue.")
            return self._pos()["Z"]
        zc = zs[int(np.argmax(vals))]
        zs, vals = self._sweepZ(zc - zin * step, zin * step / 5, 11)
        zb = zs[int(np.argmax(vals))]
        self._figure(f"Focus sweep, objective {slot}",
                     annotated=fcr.plot_curves(curves + [(zs, vals, "fine")], "Z (µm)", "focus metric", mark_x=zb),
                     expected="one clear peak of the focus metric (the rays around the via come into focus)",
                     measured=f"focus at Z {zb:.0f} µm (coarse step {step:.0f} µm, fine {step / 5:.0f} µm); "
                              f"exposure {self._ctx.get('expBase', 0):.2f} ms", ok=True)
        self._move(z=zb - zin * step)       # approached from the sweep's side (Z backlash)
        self._move(z=zb)
        self._ctx.setdefault("focus", {})[str(slot)] = {"z": zb, "stepUm": step}
        return zb

    def _zIn(self):
        """+1/-1: the Z direction towards the sample, i.e. of the focus sweeps. The homed Z sits at
        its end stop clear of the sample, so the focus lies away from it: -homeDirectionZ. The Pi rig
        homes to +Z (focus at -4420 µm, 2026-11-03); FRAME 10 (2026-11-12) homes to -Z, and a sweep
        to -Z ran into the end stop ("Z did not reach -234 µm (reads -134 µm)"). Setting zTowardsSample
        overrides it."""
        if "zIn" not in self._ctx:
            v = int((self._ctx.get("s") or {}).get("zTowardsSample", 0) or 0)
            if not v:
                hp = (getattr(self._stageManager(), "homeParams", None) or {}).get("Z") or {}
                d = hp.get("direction")
                v = -1 if d is None or float(d) > 0 else 1
            self._ctx["zIn"] = 1 if v > 0 else -1
        return self._ctx["zIn"]

    def _notPast(self, z, top):
        """``z``, but not beyond the homed Z on the side away from the sample (the end stop)."""
        return top if (z - top) * self._zIn() < 0 else z

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
        stored = (self._fc().get("result") or {}).get("objectives") or {}
        prev = stored.get(str(slot))
        k0, zcfg = self._ctx.get("slot0"), self._ctx.get("zConfig") or []
        if prev and prev.get("z") is not None:
            # another holder puts the sample at another height: what the first objective's focus
            # moved since the last calibration, the second one's moves too (its Z is unknown)
            prev0 = stored.get(str(k0)) or {}
            moved = slot != k0 and k0 in self._ctx["obj"] and prev0.get("z") is not None
            return float(prev["z"]) + (self._ctx["obj"][k0]["z"] - prev0["z"] if moved else 0.0)
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
            if factor != 1:
                self._cameraStatus(f"via work at {factor:g}x the plate exposure (the via is ~100x brighter)")

    def _autoExposure(self):
        """Exposure for focusing, set on the plate and rays: their 95th percentile towards 30 %
        of full scale (autoExposure), else only corrected when clearly too dark (< 5 %) or too
        bright (> 90 %). At most 100 ms (long exposures stalled the Hik stream). The via work
        runs at viaExposureFactor of it."""
        e = self._ctx.get("exp0")
        if not e:
            return
        lo, hi = (0.2, 0.45) if self._ctx["s"].get("autoExposure", True) else (0.05, 0.9)
        det = self._master.detectorsManager[self._ctx["det"]]
        for _ in range(8):
            det.setParameter("exposure", e)
            time.sleep(0.2)
            raw = self._grab()
            full = 255.0 if raw.dtype == np.uint8 else (4095.0 if raw.max() < 4096 else 65535.0)
            p95 = float(np.percentile(fct.gray(raw), 95))
            if lo * full < p95 < hi * full or (e >= 100 and p95 <= lo * full) or (e <= 0.05 and p95 >= hi * full):
                break
            e = float(np.clip(e * np.clip(0.3 * full / max(p95, 1.0), 0.1, 10), 0.05, 100))
        self._ctx["expBase"] = e
        self._cameraStatus(f"exposure set on the plate: 95th percentile at {100 * p95 / full:.0f} % of full scale")

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

    def _at(self, ax):
        """Where the stage is: the last verified move target (a fresh read can carry another
        axis's position, UC2 firmware 2026-11-02), else a read."""
        return self._ctx.get("at", {}).get(ax, self._pos()[ax])

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
                stage.move(value=float(v), axis=ax, is_absolute=True, is_blocking=True, speed=self._speed)
                got = self._pos()[ax]
                if abs(got - v) <= tol:
                    self._ctx.setdefault("at", {})[ax] = float(v)    # verified: later steps use this, not a fresh read
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
        img = self._ctx["lastOv"] = self._grab(self._ctx["ov"])
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
        frames: the full LED at a short manual exposure (dark enclosure, Pi 2026-11-02: the
        LED is off up to ~100 and nearly full from ~120, and auto exposure saturates its cone
        over the holes), then room light with the LED dimmed (open rig, Mac 2026-10-30)."""
        e = float(s["overviewExposureMs"])
        # room light last: in daylight it showed silkscreen wells as a hole pair (Pi 2026-11-03)
        return [(self._ctx.get("lightMain"), x) for x in (e, e / 2, 2 * e) if x > 0] + [(s["overviewIlluminationValue"], None)]

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
                "camera": None, "nFigures": 0,
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
        return {"timestamp": datetime.now().isoformat(timespec="seconds"), "detector": c["det"], "target": self._T()["key"],
                "objectives": {str(k): {"metrics": r["metrics"], "rmsPx": r["rmsPx"], "z": r["z"]} for k, r in c["obj"].items()},
                **{k: c[k] for k in ("offset", "placement", "backlash", "xyOffsets", "overview", "overviewJ", "focus", "plateShift",
                                     "pixelSizeMeasured",
                                     "repeatUm", "switchBackUm", "verifyUm") if k in c}}

    def _fc(self):
        if self._setupInfo.frameCalibration is None:
            self._setupInfo.frameCalibration = {}
        return self._setupInfo.frameCalibration

    def _T(self):
        """Geometry of the selected calibration target (frame_calibration_tools.TARGETS)."""
        return fct.TARGETS.get((self._ctx.get("s") or self._settings())["target"], fct.PLATE)

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
            meta = []
            for f in self._figures:
                with open(os.path.join(d, f"fig_{f['id']}.jpg"), "wb") as fh:
                    fh.write(f["jpg"])
                meta.append({k: f.get(k) for k in ("id", "step", "title", "expected", "measured", "ok")})
            with open(os.path.join(d, "figures.json"), "w") as fh:
                json.dump(meta, fh, indent=2)
            with open(os.path.join(d, "report.json"), "w") as fh:     # again, now with reportDir
                json.dump({"status": self._status, "settings": self._ctx.get("s")}, fh, indent=2, default=str)
            fcr.write_pdf(os.path.join(d, "report.pdf"), self._status, self._ctx.get("s"), self._figures)
        except Exception as e:
            self._logger.warning(f"Could not write the calibration report: {e}")

    def _latestReportDir(self):
        try:
            names = sorted(n for n in os.listdir(_REPORT_DIR) if os.path.isdir(os.path.join(_REPORT_DIR, n)))
            return os.path.join(_REPORT_DIR, names[-1]) if names else None
        except OSError:
            return None

    def _lastReport(self):
        """(status, settings, figures) of the running or last run; after a restart the newest
        report on disk."""
        if self._figures or self._status.get("state") != "idle":
            return self._status, self._ctx.get("s"), self._figures
        if self._diskReport is None:
            self._diskReport = (self._status, None, [])
            d = self._latestReportDir()
            try:
                with open(os.path.join(d, "report.json")) as fh:
                    rep = json.load(fh)
                figs = []
                if os.path.exists(os.path.join(d, "figures.json")):
                    with open(os.path.join(d, "figures.json")) as fh:
                        for m in json.load(fh):
                            with open(os.path.join(d, f"fig_{m['id']}.jpg"), "rb") as fj:
                                figs.append({**m, "jpg": fj.read()})
                self._diskReport = ({**rep["status"], "reportDir": d}, rep.get("settings"), figs)
            except Exception:
                pass
        return self._diskReport

    def _figure(self, title, img=None, marks=(), expected="", measured="", ok=None, annotated=None):
        """Add a report figure (never breaks the run): yellow = expected, green = measured."""
        try:
            disp = annotated if annotated is not None else fcr.annotate(img, marks)
            self._figures.append({"id": f"{len(self._figures) + 1:02d}", "step": self._status.get("step") or "",
                                  "title": title, "expected": expected, "measured": measured, "ok": ok,
                                  "jpg": fcr.jpeg(disp)})
            self._status["nFigures"] = len(self._figures)
        except Exception as e:
            self._logger.warning(f"report figure '{title}' failed: {e}")

    def _cameraStatus(self, note):
        """What the main camera is set to, for the UI."""
        try:
            p = self._master.detectorsManager[self._ctx["det"]].parameters
            self._status["camera"] = {"exposureMs": float(p["exposure"].value) if "exposure" in p else None,
                                      "gainDb": float(p["gain"].value) if "gain" in p else None, "note": note,
                                      "auto": bool(self._ctx["s"].get("autoExposure")),
                                      "maxGain": bool(self._ctx["s"].get("maxGain"))}
        except Exception:
            pass
