"""dvOPM digital twin and stage-scan volume acquisition.

The twin (``model/opm``) follows the live stages. A model axis linked to a stage axis
(setup JSON ``"opm": {"axisMap": ...}``) is read from and moved on that axis; an unlinked
one is set by hand on the instrument and typed into the UI. ``setOPMReference`` pairs
the current stage positions with model values ("we are here"); the twin is a straight
line through that pair, in mm and degrees, with the stage in µm.

The scan, like LightsheetController's continuous scan, drives sample X at constant
speed through the static sheet while the camera free-runs. Frames are placed by the
measured speed and frame rate (``twin.frame_positions``), saved raw as an OME image
(frame, tilted row, flat column) with a JSON of the frame positions, and optionally
deskewed into a (z, y, x) volume.
"""
import json
import math
import os
import threading
import time
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
from fastapi import HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from imswitch.imcommon.model import APIExport, dirtools, initLogger
from imswitch.imcontrol.model.SetupInfo import OPMInfo
from imswitch.imcontrol.model.aufgaben.storage import save_image
from imswitch.imcontrol.model.opm import twin
from imswitch.imcontrol.model.opm.kinematics import (
    Detection, Dish, MotorPose, deskew_affine, forward_kinematics, index_from_launch_angle,
    inverse_kinematics, launch_angle, paper_like_launch, plan_scan, sensitivities,
    sheet_angle, sheet_optics)
from imswitch.imcontrol.model.opm.optics import LaunchOptics, optiland_script, trace_launch
from imswitch.imcontrol.model.opm.rig import Rig, RigConfig, RigState
from ..basecontrollers import ImConWidgetController
from ..busy import busy_reasons

# ponytail: frames are buffered in RAM like Lightsheet; stream to NgffImage for longer scans
MAX_SCAN_BYTES = 4e9
SUFFIX = {"ome_zarr": (".ome.zarr",), "tiff": (".ome.tif",), "both": (".ome.zarr", ".ome.tif")}


class OPMOptics(BaseModel):
    """Launch optics for the trace and the rig drawing; live=True uses the twin's
    medium, sample Z and launch pose, live=False the what-if values below."""
    live: bool = True
    n: Optional[float] = None
    stage_z: float = 0.0
    phi_deg: Optional[float] = None     # None: the model's coplanar tilt for n
    fiber_na: float = 0.10
    f_col_mm: float = 8.0
    d_col_cyl_mm: float = 3.0
    f_cyl_mm: float = 9.9
    cyl_thickness_mm: float = 5.88
    cyl_R_mm: Optional[float] = 5.17
    f_obj_mm: float = 50.0
    wavelength_nm: float = 488.0
    fan_rays: int = 15
    fill: float = 1.0


class OPMMove(BaseModel):
    """Targets in model units: sample keys X, Y, Z (mm); launch keys x, y, z, focus (mm)
    and phi_deg. relative adds to the current values. track: a sample-Z move also moves
    the linked launch axes to the model pose for the new height. virtual: nothing moves,
    the drawing is shifted instead (align the twin with what you see). force: move even
    if a target is outside the rig's travel (the stage's own soft limits still apply)."""
    sample: Dict[str, float] = {}
    launch: Dict[str, float] = {}
    relative: bool = False
    track: bool = False
    virtual: bool = False
    force: bool = False


class OPMReference(BaseModel):
    """What the stages stand for right now. Missing sample axes are 0, missing launch
    axes the model pose for the current medium (the launch is aligned)."""
    sample: Dict[str, float] = {}
    launch: Dict[str, float] = {}


class OPMRigConfig(BaseModel):
    values: Dict[str, Any] = {}


class OPMScanRequest(BaseModel):
    length_mm: float = Field(1.0, gt=0, description="Scanned sample-X length")
    fps: float = Field(20.0, gt=0, description="Camera frame rate the speed is planned for")
    step_um: Optional[float] = Field(None, description="Step per frame; None: half the sheet FWHM")
    ramp_mm: float = Field(0.05, ge=0, description="Run-up at both ends for constant speed")
    illuSource: str = ""
    illuValue: float = 0.0              # in the source's own units (see getOPMState illumination)
    storageFormat: str = "ome_zarr"     # ome_zarr | tiff | both
    deskew: bool = True
    deskewFlip: bool = False            # the camera's tilted axis runs the other way
    experimentName: str = "opm_scan"


class OPMController(ImConWidgetController):
    """dvOPM launch twin, optics trace and stage-scan acquisition."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        info = getattr(self._setupInfo, "opm", None) or OPMInfo()
        self._det = Detection(**info.detection)
        self._dish = Dish(**info.dish)
        self._launch = paper_like_launch(self._det, self._dish)
        self._tiltedAxis = info.tiltedAxis
        pm = self._master.positionersManager
        self._links = self._checkLinks(
            twin.parse_axis_map(info.axisMap, info.positioner or pm.getAllDeviceNames()[0]))
        self._detectorName = info.detector or self._master.detectorsManager.getAllDeviceNames()[0]

        self._scan = {"isRunning": False, "progress": 0.0, "message": "Idle", "error": None,
                      "frames": 0, "last": None}
        self._scanThread = None
        self._stopScan = threading.Event()
        self._file = os.path.join(dirtools.UserFileDirs.Config, "opm.json")
        saved = self._load()
        self._n = float(saved.get("n", 1.45))
        self._limits = bool(saved.get("limits", True))
        self._rigCfg = RigConfig()
        self._rigCfg.update(saved.get("rig", {}))
        self._manual = {"sample.X": 0.0, "sample.Y": 0.0, "sample.Z": 0.0, "launch.y": 0.0,
                        **self._modelPose(0.0), **saved.get("manual", {})}
        self._ref = saved.get("reference")
        if not self._ref or set(self._ref.get("raw", {})) != set(self._links):
            self.setOPMReference(OPMReference())     # first start: the stage is at the nominal pose


    # ------------------------------------------------------------------ persistence
    def _load(self) -> dict:
        try:
            with open(self._file) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save(self):
        try:
            with open(self._file, "w") as f:
                json.dump({"n": self._n, "limits": self._limits, "reference": self._ref,
                           "manual": self._manual,
                           "rig": self._rigDict()}, f, indent=2)
        except OSError as e:
            self._logger.warning(f"Could not save {self._file}: {e}")

    def _rigDict(self) -> dict:
        return {k: getattr(self._rigCfg, k) for k in self._rigCfg.__dataclass_fields__}

    # ------------------------------------------------------------------ axes
    def _checkLinks(self, links):
        pm, ok = self._master.positionersManager, {}
        for t, (p, a, s) in links.items():
            if p in pm.getAllDeviceNames() and a in pm[p].axes:
                ok[t] = (p, a, s)
            else:
                self._logger.warning(f"OPM axis {t}: no stage axis {p}.{a}; it is set by hand")
        return ok

    def _raw(self) -> Dict[str, float]:
        """Positions (µm) of the linked axes as the stage manager last knew them.

        No serial query (getPosition() sends /motor_get on every call): the board pushes
        its positions and every move updates them. The twin only uses differences from
        its reference and moves relatively, so the manager's frame does not matter.
        """
        pm = self._master.positionersManager
        return {t: float(pm[p].position[a]) for t, (p, a, _) in self._links.items()}

    def _moveTo(self, t, value, blocking=False, speed=None):
        """Move linked axis t so the model reads ``value`` (relative move, see _raw)."""
        p, a, s = self._links[t]
        stage = self._master.positionersManager[p]
        target = twin.to_raw(t, value, s, self._ref["raw"][t], self._ref["model"][t])
        stage.move(value=target - float(stage.position[a]), axis=a, is_absolute=False,
                   is_blocking=blocking, speed=speed)

    def _values(self, raw=None) -> Dict[str, float]:
        """Every model axis: from its stage when linked, else the hand-set value."""
        raw = self._raw() if raw is None else raw
        out = dict(self._manual)
        for t, (_, _, s) in self._links.items():
            out[t] = twin.to_model(t, raw[t], s, self._ref["raw"][t], self._ref["model"][t])
        return out

    @staticmethod
    def _pose(v) -> MotorPose:
        return MotorPose(v["launch.x"], v["launch.z"], math.radians(v["launch.phi_deg"]),
                         v["launch.focus"])

    def _rigState(self, v) -> RigState:
        return RigState({ax: v[f"sample.{ax}"] for ax in "XYZ"}, self._pose(v), v["launch.y"])

    def _modelPose(self, stage_z: float, n: Optional[float] = None) -> Dict[str, float]:
        p = inverse_kinematics(self._n if n is None else n, stage_z, self._det, self._dish,
                               self._launch, "carriage_xz", 0.0)
        return {"launch.x": p.x, "launch.z": p.z, "launch.phi_deg": math.degrees(p.phi),
                "launch.focus": p.focus}

    def _drive(self, targets: Dict[str, float]):
        """Move linked axes (non-blocking) and store hand-set ones. Returns the hand-set ones."""
        manual = {}
        for t, v in targets.items():
            if t in self._links:
                self._moveTo(t, v)
            else:
                self._manual[t] = float(v)
                manual[t] = float(v)
        self._save()
        return manual

    def _refuseWhileScanning(self):
        if self._scan["isRunning"]:
            raise HTTPException(409, "The OPM scan is running. Stop it first.")

    # ------------------------------------------------------------------ state
    @APIExport()
    def getOPMState(self) -> dict:
        """Twin state: medium n, model axes (mm, degrees), stage positions of the linked
        axes (µm), the reference, the model's target pose and alignment errors, the
        illumination sources and the scan status."""
        raw = self._raw()
        v = self._values(raw)
        out = {"n": self._n, "limits": self._limits, "detector": self._detectorName,
               "links": {t: f"{'-' if s < 0 else ''}{p}.{a}"
                         for t, (p, a, s) in self._links.items()},
               "raw": raw, "values": v, "reference": self._ref, "scan": dict(self._scan),
               "illumination": self._illumination()}
        try:
            out["model"] = self._model(v)
        except ValueError as e:
            out["model_error"] = str(e)
        return _jsonable(out)

    def _model(self, v) -> dict:
        n, det, z, pose = self._n, self._det, v["sample.Z"], self._pose(v)
        target = self._modelPose(z)
        fk = forward_kinematics(pose, n, z, det, self._dish, self._launch)
        opt = sheet_optics(n, det, self._launch)
        return {
            "theta_deg": math.degrees(sheet_angle(n, det)),
            "phi_deg": math.degrees(launch_angle(n, det)),
            "phi_now_implies_n": index_from_launch_angle(pose.phi, det),
            "target": {k.split(".")[1]: val for k, val in target.items()},
            "err": {"tilt_deg": math.degrees(fk.angle_error), "offset_um": fk.offset_error_mm * 1e3,
                    "waist_mm": fk.waist_error_mm},
            "tol": {"tilt_deg": math.degrees((opt["fwhm_um"] * 1e-3 / 4)
                                             / (opt["fov_along_sheet_mm"] / 2)),
                    "offset_um": opt["fwhm_um"] / 4, "waist_mm": opt["rayleigh_mm"] / 4},
            "sheet": {"fwhm_um": opt["fwhm_um"], "confocal_mm": opt["confocal_mm"],
                      "fov_mm": opt["fov_along_sheet_mm"]},
            "gains": sensitivities(n, det),
        }

    def _illumination(self):
        lm = self._master.lasersManager
        return [{"name": k, "min": lm[k].valueRangeMin, "max": lm[k].valueRangeMax}
                for k in lm.getAllDeviceNames()]

    # ------------------------------------------------------------------ twin actions
    @APIExport()
    def setOPMMedium(self, n: float, move: bool = True) -> dict:
        """Set the medium index and put the launch on its model pose for the current
        sample Z: linked launch axes move (move=True), hand-set ones are updated and
        returned as "manual" so the user sets them on the instrument."""
        self._refuseWhileScanning()
        if not 1.0 < n < 2.0:
            raise HTTPException(400, "Index must be between 1.0 and 2.0.")
        try:
            launch_angle(n, self._det)
        except ValueError as e:
            raise HTTPException(400, str(e))
        try:
            pose = self._modelPose(self._values()["sample.Z"], n)
        except ValueError as e:
            raise HTTPException(400, f"No launch pose at this sample Z: {e}")
        self._n = float(n)
        targets = pose if move else {t: x for t, x in pose.items() if t not in self._links}
        return {"n": self._n, "manual": self._drive(targets)}

    @APIExport(requestType="POST")
    def moveOPMAxes(self, req: OPMMove) -> dict:
        """Move model axes (mm, degrees). Linked axes move their stage (non-blocking), the
        others are stored as set by hand. A target outside the rig's travel is refused
        before anything moves unless force=True. virtual=True moves nothing and shifts
        the twin instead, so the drawing can be matched to the real sheet."""
        self._refuseWhileScanning()
        cur = self._values()
        tgt = {}
        for grp, d in (("sample", req.sample), ("launch", req.launch)):
            for k, val in d.items():
                t = f"{grp}.{k.upper() if grp == 'sample' else k.lower()}"
                if t not in twin.TWIN_AXES:
                    raise HTTPException(400, f"Unknown axis: {grp} {k}")
                tgt[t] = cur[t] + val if req.relative else val
        note = None
        if req.track and "sample.Z" in tgt and not req.launch and not req.virtual:
            try:
                follow = self._modelPose(tgt["sample.Z"])
                tgt.update({t: x for t, x in follow.items() if t in self._links})
            except ValueError as e:      # no sheet pose at this height: move the sample alone
                note = f"The launch did not follow: {e}."
        if req.virtual:
            return {"moved": [], "manual": self._align(cur, tgt)}
        if self._limits and not req.force:
            self._checkTravel(cur, tgt)
        return {"moved": [t for t in tgt if t in self._links], "manual": self._drive(tgt),
                "note": note}

    @APIExport()
    def setOPMLimits(self, enabled: bool = True) -> dict:
        """Switch the twin's own travel check for moveOPMAxes on or off (saved). The stage
        managers' soft limits are not affected."""
        self._limits = bool(enabled)
        self._save()
        return {"limits": self._limits}

    def _checkTravel(self, cur, tgt):
        """Refuse targets outside the rig's travel; axes that are not moved are not checked."""
        fin = self._rigState({**cur, **tgt})
        bad = [f"{c['axis']} = {c['value']:.3f} (limits {c['min']:g} to {c['max']:g})"
               for c in Rig(self._rigCfg, self._det, self._dish, self._launch).check_limits(fin)
               if not c["ok"] and c["axis"] in tgt]
        if bad:
            raise HTTPException(400, "Outside travel: " + "; ".join(bad))

    def _align(self, cur, tgt):
        """Virtual move: shift the twin so the axes read ``tgt`` without moving anything."""
        for t, v in tgt.items():
            if t in self._links:
                self._ref["model"][t] += v - cur[t]
            else:
                self._manual[t] = float(v)
        self._save()
        return {}

    @APIExport(requestType="POST")
    def setOPMReference(self, req: OPMReference) -> dict:
        """Reset the twin: the current stage positions stand for these model values (sample
        default 0 mm, launch default the model pose for the current medium). Fine
        alignment afterwards: moveOPMAxes(virtual=True)."""
        self._refuseWhileScanning()
        sample = {f"sample.{k.upper()}": float(x) for k, x in req.sample.items()}
        launch = {f"launch.{k.lower()}": float(x) for k, x in req.launch.items()}
        try:
            pose = self._modelPose(sample.get("sample.Z", 0.0))
        except ValueError as e:
            raise HTTPException(400, f"No launch pose at this sample Z: {e}")
        model = {"sample.X": 0.0, "sample.Y": 0.0, "sample.Z": 0.0, "launch.y": 0.0,
                 **pose, **sample, **launch}
        self._ref = {"raw": self._raw(), "model": {t: model[t] for t in self._links},
                     "time": datetime.now().isoformat(timespec="seconds")}
        self._manual.update({t: x for t, x in {**sample, **launch}.items() if t not in self._links})
        self._save()
        return self._ref

    # ------------------------------------------------------------------ drawing and optics
    @APIExport(requestType="POST")
    def getOPMRigLayout(self, req: OPMOptics) -> dict:
        """Side and top view of the rig at the current axis values, plus collision,
        clipping and travel checks (all mm)."""
        rig = Rig(self._rigCfg, self._det, self._dish, self._launch, _launchOptics(req))
        try:
            return _jsonable(rig.layout(self._rigState(self._values()), self._n))
        except ValueError as e:
            raise HTTPException(400, str(e))

    def _opticsArgs(self, req: OPMOptics):
        o = _launchOptics(req)
        if req.live:
            v = self._values()
            return self._n, v["sample.Z"], self._pose(v), o
        n = req.n or self._n
        pose = inverse_kinematics(n, req.stage_z, self._det, self._dish, self._launch,
                                  "carriage_xz", 0.0)
        if req.phi_deg is not None:
            pose = MotorPose(pose.x, pose.z, math.radians(req.phi_deg), pose.focus)
        return n, req.stage_z, pose, o

    @APIExport(requestType="POST")
    def traceOPMOptics(self, req: OPMOptics) -> dict:
        """Ray trace of the launch: fibre, collimator, cylinder lens, objective, the
        tilted dish bottom and the medium, with the Gaussian sheet envelope."""
        n, z, pose, o = self._opticsArgs(req)
        try:
            return _jsonable(trace_launch(n, z, self._det, self._dish, self._launch, o, pose))
        except ValueError as e:
            raise HTTPException(400, str(e))

    @APIExport(requestType="POST")
    def exportOPMOptiland(self, req: OPMOptics):
        """The same launch as a stand-alone Optiland script (download)."""
        n, z, pose, o = self._opticsArgs(req)
        try:
            txt = optiland_script(n, z, self._det, self._dish, self._launch, o, pose,
                                  title=f"dvOPM light-sheet launch (n = {n:.3f})")
        except ValueError as e:
            raise HTTPException(400, str(e))
        return Response(txt, media_type="text/x-python", headers={
            "Content-Disposition": f'attachment; filename="dvopm_launch_n{n:.3f}_optiland.py"'})

    @APIExport()
    def getOPMRigConfig(self) -> dict:
        """Rig dimensions and travel limits (mm, degrees)."""
        return self._rigDict()

    @APIExport(requestType="POST")
    def setOPMRigConfig(self, req: OPMRigConfig) -> dict:
        """Change rig dimensions; saved with the twin state."""
        bad = self._rigCfg.update(req.values)
        if bad:
            raise HTTPException(400, f"Unknown field(s): {', '.join(bad)}")
        self._save()
        return self._rigDict()

    # ------------------------------------------------------------------ scan
    @APIExport()
    def getOPMScanPlan(self, length_mm: float = 1.0, fps: float = 20.0,
                       step_um: Optional[float] = None) -> dict:
        """Step per frame (µm), slice spacing (µm), stage speed (mm/s), frames and
        duration for a sample-X scan through the static sheet."""
        return plan_scan(self._n, self._det, self._launch, length_mm, fps,
                         step_mm=step_um * 1e-3 if step_um else None)

    @APIExport()
    def getOPMScanStatus(self) -> dict:
        return _jsonable(dict(self._scan))

    def isOPMScanRunning(self) -> bool:
        return bool(self._scan["isRunning"])

    @APIExport()
    def stopOPMScan(self) -> dict:
        self._stopScan.set()
        return {"success": True}

    @APIExport(requestType="POST")
    def startOPMScan(self, req: OPMScanRequest) -> dict:
        """Scan sample X through the sheet at constant speed with the camera free-running.
        Refused while another acquisition drives the microscope. Returns the plan."""
        self._refuseWhileScanning()
        reasons = busy_reasons(self._master, self._logger, ignore=("OPM",))
        if reasons:
            raise HTTPException(409, f"Busy: {', '.join(reasons)}.")
        if "sample.X" not in self._links:
            raise HTTPException(400, "Link sample.X to a stage axis (opm.axisMap) to scan.")
        if req.storageFormat not in SUFFIX:
            raise HTTPException(400, f"storageFormat must be one of {', '.join(SUFFIX)}")
        plan = self.getOPMScanPlan(req.length_mm, req.fps, req.step_um)
        frame = self._master.detectorsManager[self._detectorName].getLatestFrame()
        frames = plan["frames"] * (1 + 2 * req.ramp_mm / req.length_mm)
        if frame is not None and frames * np.asarray(frame).nbytes > MAX_SCAN_BYTES:
            raise HTTPException(400, f"About {frames:.0f} frames would not fit in memory; "
                                     "scan less or use a larger step.")
        self._stopScan.clear()
        self._scan = {"isRunning": True, "progress": 0.0, "message": "Starting", "error": None,
                      "frames": 0, "last": self._scan.get("last"), "plan": plan}
        self._scanThread = threading.Thread(target=self._runScan, args=(req, plan), daemon=True)
        self._scanThread.start()
        return {"success": True, "plan": plan}

    def _status(self, message, progress=None):
        self._scan["message"] = message
        if progress is not None:
            self._scan["progress"] = float(progress)

    def _runScan(self, req: OPMScanRequest, plan: dict):
        p, a, _ = self._links["sample.X"]
        stage = self._master.positionersManager[p]
        x0 = self._values()["sample.X"]
        laser = self._lightOn(req)
        try:
            frames, xs, stamps, numbers = self._sweep(stage, a, x0, req, plan)
            self._status(f"Writing {len(frames)} frames", 1.0)
            self._scan["last"] = self._write(req, plan, frames, xs, stamps, numbers)
            self._status(f"Saved {len(frames)} frames")
        except Exception as e:
            self._logger.error(f"OPM scan failed: {e}")
            self._scan["error"] = str(e)
            self._status("Stopped" if self._stopScan.is_set() else "Failed")
        finally:
            if laser is not None:
                laser.setValue(0)
                laser.setEnabled(False)
            self._moveTo("sample.X", x0)
            self._scan["isRunning"] = False

    def _lightOn(self, req: OPMScanRequest):
        lm = self._master.lasersManager
        if req.illuSource not in lm.getAllDeviceNames():
            return None
        laser = lm[req.illuSource]
        laser.setEnabled(True)
        laser.setValue(req.illuValue)
        return laser

    def _sweep(self, stage, axis, x0, req, plan):
        """Run-up, constant-speed move with frames grabbed on the way, crop to the scan."""
        xa, xb = x0 - req.ramp_mm, x0 + req.length_mm + req.ramp_mm
        det = self._master.detectorsManager[self._detectorName]
        det.startAcquisition()
        self._status("Moving to the start")
        self._moveTo("sample.X", xa, blocking=True)
        speed = plan["velocity_mm_s"] * 1e3       # µm/s; the speed that counts is measured below
        done = {}

        def move():
            self._moveTo("sample.X", xb, blocking=True, speed=speed)
            done["t1"] = time.time()
        t0 = time.time()
        mover = threading.Thread(target=move, daemon=True)
        mover.start()
        frames, stamps, numbers = self._grab(det, mover, t0, (xb - xa) / plan["velocity_mm_s"])
        if self._stopScan.is_set():
            getattr(stage, "forceStop", lambda _: None)(axis)
            raise RuntimeError("Stopped by user")
        mover.join()
        if None not in numbers:
            stamps = twin.frame_times(stamps, numbers)
        xs = twin.frame_positions(stamps, t0, done.get("t1", time.time()), xa, xb)
        keep = (xs >= x0) & (xs <= x0 + req.length_mm)
        if keep.sum() < 2:
            raise RuntimeError(f"Only {int(keep.sum())} frames inside the scan; lower the speed "
                               "or check that the camera runs.")
        idx = np.flatnonzero(keep)
        return ([frames[i] for i in idx], xs[idx], np.asarray(stamps)[idx],
                np.asarray(numbers)[idx] if None not in numbers else None)

    def _grab(self, det, mover, t0, duration):
        """Every new camera frame while the stage moves, with its arrival time."""
        frames, stamps, numbers, last = [], [], [], object()
        while mover.is_alive() and not self._stopScan.is_set():
            try:
                frame, number = det.getLatestFrame(returnFrameNumber=True)
            except TypeError:
                frame = det.getLatestFrame()
                number = None
            new = number != last if number is not None else frame is not last
            if frame is not None and new:
                frames.append(np.array(frame[..., 0] if frame.ndim == 3 else frame))
                stamps.append(time.time())
                numbers.append(number)
                last = number if number is not None else frame
                self._scan["frames"] = len(frames)
            self._status(f"Scanning, {len(frames)} frames", min(1.0, (time.time() - t0) / duration))
            time.sleep(0.002)
        return frames, stamps, numbers

    def _write(self, req, plan, frames, xs, stamps, numbers) -> dict:
        stack = np.stack(frames)
        if self._tiltedAxis == 1:
            stack = stack.transpose(0, 2, 1)
        summary = twin.scan_summary(stamps, xs, numbers)
        date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        folder = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), "OPM",
                              f"{date}_{req.experimentName}")
        os.makedirs(folder, exist_ok=True)
        det, n = self._det, self._n
        flat_um = det.pixel_um / det.M
        step_um = summary["step_um"] or plan["step_um"]
        meta = {"n": n, "theta_sheet_deg": math.degrees(sheet_angle(n, det)),
                "requested": plan, "measured": summary, "x_mm": xs.tolist(),
                "t_s": (np.asarray(stamps) - stamps[0]).tolist(),
                "frame_numbers": None if numbers is None else np.asarray(numbers).tolist(),
                "deskew_affine_frame_row_col_to_xyz_mm": deskew_affine(
                    n, det, step_um * 1e-3, stack.shape[1] / 2, stack.shape[2] / 2).tolist()}
        out = {"folder": folder, "frames": int(len(stack)), **summary, "raw": [], "deskewed": []}
        notes = {k: x for k, x in {"n": n, "fps_measured": summary["fps"],
                                   "step_um_measured": step_um}.items() if x is not None}
        for sfx in SUFFIX[req.storageFormat]:
            path = os.path.join(folder, "raw" + sfx)
            save_image(path, stack, "ZYX", pixel_size_um=flat_um, z_step_um=step_um,
                       channel_names=[req.illuSource or "OPM"], annotations=notes)
            out["raw"].append(path)
        if req.deskew:
            try:
                vol, vox = twin.deskew(stack, xs, n, det, flip=req.deskewFlip)
                meta["deskew"] = vox
                for sfx in SUFFIX[req.storageFormat]:
                    path = os.path.join(folder, "deskewed" + sfx)
                    save_image(path, vol, "ZYX", pixel_size_um=vox["dx_um"], z_step_um=vox["dz_um"],
                               channel_names=[req.illuSource or "OPM"], annotations=notes)
                    out["deskewed"].append(path)
            except ValueError as e:
                out["deskewError"] = str(e)
        with open(os.path.join(folder, "scan.json"), "w") as f:
            json.dump(_jsonable(meta), f, indent=1)
        return out


def _launchOptics(req: OPMOptics) -> LaunchOptics:
    known = {k: x for k, x in req.model_dump().items() if k in LaunchOptics.__dataclass_fields__}
    known.update(fan_rays=max(3, min(61, req.fan_rays)), fill=max(0.05, min(1.0, req.fill)))
    return LaunchOptics(**known)


def _jsonable(d):
    """numpy arrays and scalars -> plain JSON types."""
    return json.loads(json.dumps(
        d, default=lambda o: o.tolist() if hasattr(o, "tolist") else float(o)))
