"""Grating spectrometer: live spectrum, acquisition, dark, calibration wizard, alignment.

Loaded when "Spectrometer" is in the setup's availableWidgets; settings in the setup
block "spectrometer" (SpectrometerInfo, docs/SPECTROMETER.md). The instrument state
(camera, dark, processing, calibration) is SpectrometerManager's; this controller adds
the REST API, the live loop, background acquisitions and switching the calibration
sources (lasers through LaserController, NeoPixel channels through the LED matrix,
sources switched by hand through the wizard's Continue).

Units: wavelengths nm, Raman shift cm^-1, exposure ms, pixels are full-sensor columns
and rows. Lasers switched by the wizard run at ``laserPowerFraction`` of their range
and are always switched off at the end, on cancel and on errors.
"""
import math
import threading
import time
from collections import deque
from typing import Any, Dict, List, Optional

import numpy as np
from pydantic import BaseModel

from imswitch.imcommon.model import APIExport, initLogger
from imswitch.imcontrol.model.SetupInfo import SpectrometerInfo
from imswitch.imcontrol.model.spectrometer import peaks as K
from imswitch.imcontrol.model.spectrometer import pipeline as P
from imswitch.imcontrol.model.spectrometer import references as R
from imswitch.imcontrol.model.spectrometer.calibration import (alignment_plan,
                                                               make_calibration,
                                                               nm_to_raman_shift)
from imswitch.imcontrol.model.spectrometer.wizard import (CalibrationSource, CalibrationWizard,
                                                          WizardSettings)
from ..basecontrollers import ImConWidgetController

CHANNEL_NAMES = {"r": "red", "g": "green", "b": "blue"}


def _jsonable(o):
    """numpy -> Python, NaN/inf -> None (FastAPI refuses NaN)."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        v = float(o)
        return v if math.isfinite(v) else None
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def _rounded(a, digits: int):
    return [None if not math.isfinite(v) else round(v, digits) for v in np.asarray(a, float)]


class SpectrometerSettingsBody(BaseModel):
    rowStart: Optional[int] = None
    rowStop: Optional[int] = None
    colStart: Optional[int] = None
    colStop: Optional[int] = None
    collapse: Optional[str] = None
    nAverage: Optional[int] = None
    removeSpikes: Optional[bool] = None
    spikeThreshold: Optional[float] = None
    subtractDark: Optional[bool] = None
    smileCorrection: Optional[bool] = None
    liveAverage: Optional[int] = None


class SpectrometerCalibrationBody(BaseModel):
    """Points [{px, nm, sigmaNm?, source?}] to fit; model auto | grating | polynomial."""
    points: List[Dict[str, Any]] = []
    model: str = "auto"
    order: int = 2
    save: bool = True
    notes: str = ""


class SpectrometerReferenceBody(BaseModel):
    """Reference spectrum of a calibration LED, measured on another spectrometer."""
    key: str
    wavelengthNm: List[float]
    intensity: List[float]
    label: str = ""


class SpectrometerWizardBody(BaseModel):
    sources: List[str]
    model: str = "auto"
    order: int = 2
    nAverage: int = 5
    nDark: int = 10
    takeDark: bool = True
    measureSmile: bool = True
    settleS: float = 0.5
    power: Dict[str, float] = {}         # per source: fraction of the laser range / LED value
    wavelengths: Dict[str, float] = {}   # per source: measured line (replaces the nominal)
    sigmas: Dict[str, float] = {}        # per source: uncertainty of that wavelength (nm)


class SpectrometerController(ImConWidgetController):
    """Spectrometer page: live spectrum, camera, dark, calibration, alignment, saving."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._logger = initLogger(self)
        self._info = getattr(self._setupInfo, "spectrometer", None) or SpectrometerInfo()
        self._mgr = getattr(self._master, "spectrometerManager", None)
        if self._mgr is None:
            from imswitch.imcontrol.model.managers.SpectrometerManager import SpectrometerManager
            self._mgr = SpectrometerManager(self._info,
                                            getattr(self._master, "detectorsManager", None))
        self._lock = threading.RLock()
        self._camUsers, self._camHandle = set(), None
        self._live = {"running": False, "periodMs": 250, "frames": 0, "error": None,
                      "average": 1}
        self._liveStop = threading.Event()
        self._liveThread = None
        self._liveBuf = deque(maxlen=1)
        self._liveRec = None
        self._acquired = None
        self._job = {"state": "idle", "kind": None, "progress": 0.0, "message": "", "error": None}
        self._jobStop = threading.Event()
        self._wizard: Optional[CalibrationWizard] = None
        self._wizStop = threading.Event()
        self._active: Dict[str, float] = {}
        self._whiteRef = R.white_phosphor_led_spectrum()

    # ------------------------------------------------------------------ helpers
    def _needCamera(self, user: str, need: bool):
        """Keep the camera acquiring while anybody (live, job, wizard) needs frames."""
        if self._mgr.simulate:
            return
        dm = self._master.detectorsManager
        with self._lock:
            (self._camUsers.add if need else self._camUsers.discard)(user)
            det = self._mgr.detector()
            if self._camUsers and self._camHandle is None and not getattr(det, "_running", False):
                self._camHandle = dm.startAcquisition()
            elif not self._camUsers and self._camHandle is not None:
                try:
                    dm.stopAcquisition(self._camHandle)
                finally:
                    self._camHandle = None

    def _record(self, which: str = "auto") -> Optional[Dict]:
        if which == "live":
            return self._liveRec
        if which == "acquired":
            return self._acquired
        return self._liveRec if self._live["running"] or self._acquired is None else self._acquired

    def _peaks(self, rec: Dict, n: int = 8) -> List[Dict]:
        counts, px = rec["counts"], rec["px"]
        out = []
        for p in K.find_peaks(counts, max_peaks=n):
            pr = K.peak_properties(counts, p["index"], p["base"])
            x = float(px[0] + pr["center"])
            nm = float(self._mgr.calibration.px_to_nm(x))
            out.append({"px": x, "nm": nm, "counts": pr["height"], "fwhmPx": pr["fwhm"],
                        "fwhmNm": pr["fwhm"] * float(self._mgr.calibration.nm_per_px(x)),
                        "snr": pr["snr"]})
        return out

    # ------------------------------------------------------------------ state
    @APIExport()
    def getSpectrometerState(self) -> Dict:
        """Everything the page needs once: camera, processing, calibration summary, dark,
        sources, design, live/job/wizard status and warnings."""
        m = self._mgr
        warnings = list(m.designWarnings())
        if m.darkWarning():
            warnings.append(m.darkWarning())
        if m.calibration.smile and not m.calibration.calibrated:
            warnings.append("smile measured but no wavelength calibration yet")
        return _jsonable({
            "detector": m.detectorName, "simulated": m.simulate, "camera": m.cameraState(),
            "processing": m.settings.to_dict(), "liveAverage": self._live["average"],
            "calibration": m.calibration.summary(m.design.sensorWidthPx),
            "design": vars(m.design), "laserNm": m.laserNm,
            "smile": m.calibration.smile or None,
            "dark": ({k: v for k, v in m.darkMeta.items()} if m.dark is not None else None),
            "live": dict(self._live), "job": dict(self._job),
            "wizard": self._wizard.state if self._wizard else None,
            "alignmentTargetNm": self._info.alignmentTargetNm or m.design.centerWavelengthNm,
            "activeSources": sorted(self._active), "stateFile": m.stateFile,
            "warnings": warnings})

    # ------------------------------------------------------------------ live
    @APIExport()
    def startSpectrometerLive(self, periodMs: int = 250) -> Dict:
        """Process the camera's latest frame every periodMs (>= 50) into a live spectrum."""
        with self._lock:
            self._live["periodMs"] = int(max(50, periodMs))
            if self._live["running"]:
                return {"success": True, "live": dict(self._live)}
            self._needCamera("live", True)
            self._liveStop.clear()
            self._live.update(running=True, error=None)
            self._liveThread = threading.Thread(target=self._liveLoop, daemon=True,
                                                name="SpectrometerLive")
            self._liveThread.start()
        return {"success": True, "live": dict(self._live)}

    @APIExport()
    def stopSpectrometerLive(self) -> Dict:
        self._liveStop.set()
        if self._liveThread is not None:
            self._liveThread.join(timeout=5)
        with self._lock:
            self._live["running"] = False
            self._liveThread = None
        self._needCamera("live", False)
        return {"success": True}

    def _liveLoop(self):
        last_num = None
        while not self._liveStop.is_set():
            t0 = time.time()
            try:
                frame, num = self._mgr.grabFrame(fresh=False)
                if self._mgr.simulate or num < 0 or num != last_num:
                    last_num = num
                    rec = self._mgr.processFrame(frame, num)
                    self._liveRec = self._averaged(rec)
                    self._live["frames"] += 1
                    self._live["error"] = None
            except Exception as exc:
                self._live["error"] = str(exc)
                self._liveStop.wait(0.5)
            self._liveStop.wait(max(0.0, self._live["periodMs"] / 1000.0 - (time.time() - t0)))

    def _averaged(self, rec: Dict) -> Dict:
        n = int(max(1, self._live["average"]))
        if self._liveBuf.maxlen != n:
            self._liveBuf = deque(self._liveBuf, maxlen=n)
        if self._liveBuf and len(self._liveBuf[-1]) != len(rec["counts"]):
            self._liveBuf.clear()
        self._liveBuf.append(rec["counts"])
        if len(self._liveBuf) > 1:
            rec = {**rec, "counts": np.mean(np.stack(self._liveBuf), axis=0)}
            rec["metadata"] = {**rec["metadata"], "liveAverage": len(self._liveBuf)}
        return rec

    @APIExport()
    def getSpectrometerSpectrum(self, which: str = "auto", maxPoints: int = 1536,
                                withPeaks: bool = True) -> Dict:
        """Latest spectrum: which = auto (live if running, else the last acquisition) |
        live | acquired. Arrays px, nm (calibration or design), ramanShiftCm1 (if a
        laser wavelength is set) and counts, decimated to maxPoints keeping maxima."""
        rec = self._record(which)
        if rec is None:
            return {"available": False, "live": dict(self._live)}
        idx = np.arange(len(rec["counts"]))
        sel, counts = P.decimate(idx, rec["counts"], int(maxPoints or 0))
        meta = rec["metadata"]
        kind = "live" if rec is self._liveRec else "acquired"
        out = {"available": True, "which": kind, "px": rec["px"][sel].tolist(),
               "nm": _rounded(rec["nm"][sel], 4), "counts": _rounded(counts, 2),
               "ramanShiftCm1": (_rounded(rec["ramanShiftCm1"][sel], 2)
                                 if rec.get("ramanShiftCm1") is not None else None),
               "calibrated": bool(meta["calibration"].get("calibrated")),
               "model": meta["calibration"].get("model"),
               "metadata": {k: meta.get(k) for k in (
                   "timestamp", "exposureMs", "gain", "bitDepth", "frames", "rows", "cols",
                   "darkSubtracted", "smileCorrected", "spikesRemoved", "saturation",
                   "laserNm", "liveAverage", "simulatedSources")},
               "live": dict(self._live)}
        if withPeaks:
            out["peaks"] = self._peaks(rec)
        return _jsonable(out)

    @APIExport()
    def getSpectrometerImage(self, full: bool = False, corrected: bool = False,
                             maxWidth: int = 1536) -> Dict:
        """PNG (data URL) of the region of the last spectrum (raw, or dark/smile-corrected)
        or of a whole frame with the region's rows, to place the region on the slit."""
        try:
            return _jsonable(self._mgr.image(full=bool(full), max_width=int(maxWidth),
                                             corrected=bool(corrected)))
        except Exception as exc:
            return {"image": None, "error": str(exc)}

    # ------------------------------------------------------------------ settings
    @APIExport()
    def setSpectrometerCamera(self, exposureMs: float = -1.0, gain: float = -1.0,
                              bitDepth: int = 0) -> Dict:
        """Exposure (ms), gain and bit depth of the spectrometer camera; negative/0 keeps."""
        try:
            out = self._mgr.setCamera(exposureMs if exposureMs >= 0 else None,
                                      gain if gain >= 0 else None, bitDepth or None)
            return _jsonable({"success": True, **out})
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    @APIExport(requestType="POST")
    def setSpectrometerSettings(self, settings: SpectrometerSettingsBody) -> Dict:
        """Region of interest (full-sensor rows/columns, stop exclusive, 0 = to the end),
        collapse mean|sum, frames per acquisition, spikes, dark, smile, live average."""
        vals = {k: v for k, v in settings.model_dump().items() if v is not None}
        live_avg = vals.pop("liveAverage", None)
        with self._mgr._lock:
            cur = self._mgr.settings.to_dict()
            cur.update(vals)
            if cur.get("collapse") not in ("mean", "sum"):
                return {"success": False, "error": "collapse must be mean or sum"}
            self._mgr.settings = P.ProcessingSettings.from_dict(cur)
            self._mgr.save()
        if live_avg is not None:
            self._live["average"] = int(max(1, min(live_avg, 100)))
        return _jsonable({"success": True, "processing": self._mgr.settings.to_dict(),
                          "liveAverage": self._live["average"]})

    @APIExport()
    def autoSpectrometerRoi(self, frac: float = 0.5, pad: float = 0.05) -> Dict:
        """Find the rows lit by the slit in a fresh frame and use them as the region."""
        try:
            frame, _ = self._mgr.grabFrame(fresh=False)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        found = P.find_slit_rows(frame, frac=float(frac), pad=float(pad))
        if not found["found"]:
            return {"success": False, "error": "no slit image found (is light coming in?)"}
        body = SpectrometerSettingsBody(rowStart=found["rowStart"], rowStop=found["rowStop"])
        res = self.setSpectrometerSettings(body)
        prof = np.asarray(found["profile"])
        return _jsonable({**res, "rowStart": found["rowStart"], "rowStop": found["rowStop"],
                          "rowProfile": _rounded(prof[::max(1, prof.size // 512)], 1)})

    @APIExport()
    def setSpectrometerLaserWavelength(self, laserNm: float = 0.0) -> Dict:
        """Excitation wavelength (nm) for the Raman-shift axis; 0 clears it."""
        self._mgr.setLaserNm(laserNm if laserNm > 0 else None)
        return {"success": True, "laserNm": self._mgr.laserNm}

    # ------------------------------------------------------------------ jobs
    def _startJob(self, kind: str, fn) -> Dict:
        with self._lock:
            if self._job["state"] == "running":
                return {"success": False, "error": f"a {self._job['kind']} is running"}
            if self._wizard is not None and self._wizard.state in ("running", "waiting"):
                return {"success": False, "error": "the calibration wizard is running"}
            self._jobStop.clear()
            self._job = {"state": "running", "kind": kind, "progress": 0.0, "message": kind,
                         "error": None, "startedAt": time.time()}
        self._needCamera("job", True)

        def run():
            try:
                result = fn()
                self._job.update(state="done", progress=1.0, result=_jsonable(result),
                                 message=f"{kind} done")
            except InterruptedError:
                self._job.update(state="stopped", message=f"{kind} stopped")
            except Exception as exc:
                self._logger.error(f"Spectrometer {kind}: {exc}")
                self._job.update(state="error", error=str(exc), message=str(exc))
            finally:
                self._job["finishedAt"] = time.time()
                self._needCamera("job", False)

        threading.Thread(target=run, daemon=True, name=f"Spectrometer-{kind}").start()
        return {"success": True, "job": dict(self._job)}

    def _progress(self, i: int, n: int):
        self._job["progress"] = i / max(n, 1)
        self._job["message"] = f"{self._job['kind']}: frame {i} of {n}"

    @APIExport()
    def acquireSpectrometerSpectrum(self, nAverage: int = 0) -> Dict:
        """Average nAverage fresh frames (0: the processing setting) into one spectrum,
        in the background; poll getSpectrometerAcquisition."""
        n = int(nAverage) if nAverage > 0 else int(self._mgr.settings.nAverage)

        def job():
            acq = self._mgr.acquireRoi(n, fresh=True, progress=self._progress,
                                       stop=self._jobStop)
            rec = self._mgr.processAcquired(acq)
            self._acquired = rec
            return {"frames": n, "saturation": acq["saturation"],
                    "spikesRemoved": rec["metadata"]["spikesRemoved"],
                    "peaks": self._peaks(rec)}
        return self._startJob("acquisition", job)

    @APIExport()
    def takeSpectrometerDark(self, nAverage: int = 10) -> Dict:
        """Mean of nAverage full frames as the dark; block the light first. The
        calibration sources switched from this page are switched off first."""
        self._allSourcesOff()

        def job():
            return self._mgr.takeDark(int(max(1, nAverage)), progress=self._progress,
                                      stop=self._jobStop)
        return self._startJob("dark", job)

    @APIExport()
    def getSpectrometerAcquisition(self) -> Dict:
        """State of the running/last acquisition or dark job."""
        return _jsonable(dict(self._job))

    @APIExport()
    def stopSpectrometerAcquisition(self) -> Dict:
        self._jobStop.set()
        return {"success": True}

    @APIExport()
    def clearSpectrometerDark(self) -> Dict:
        self._mgr.clearDark()
        return {"success": True}

    @APIExport()
    def saveSpectrometerSpectrum(self, name: str = "spectrum", fileFormat: str = "hdf5",
                                 which: str = "auto", includeRoi: bool = True,
                                 notes: str = "") -> Dict:
        """Save a spectrum to <data>/Spectrometer: fileFormat hdf5 (one .h5 with the region
        frame) or csv (.csv + .json metadata). which: auto | live | acquired."""
        rec = self._record(which)
        if rec is None:
            return {"success": False, "error": "no spectrum yet (start live or acquire)"}
        extra = {"notes": notes, "activeSources": sorted(self._active)}
        try:
            paths = self._mgr.saveRecord(rec, name, fileFormat.lower(), bool(includeRoi), extra)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {"success": True, "files": paths}

    # ------------------------------------------------------------------ calibration
    @APIExport()
    def getSpectrometerCalibration(self) -> Dict:
        """The full calibration (model, parameters, points with residuals, smile)."""
        cal = self._mgr.calibration
        return _jsonable({**cal.to_dict(), "summary": cal.summary(self._mgr.design.sensorWidthPx),
                          "previous": self._mgr.previousCalibration})

    @APIExport(requestType="POST")
    def setSpectrometerCalibration(self, body: SpectrometerCalibrationBody) -> Dict:
        """Fit and (save=True) store a calibration from known lines [{px, nm, sigmaNm}]."""
        try:
            cal = make_calibration(body.points, self._mgr.design, model=body.model,
                                   order=body.order, start=self._mgr.calibration.grating,
                                   notes=body.notes, detector=self._mgr.detectorName or "")
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        cal.smile = dict(self._mgr.calibration.smile)
        self._mgr.setCalibration(cal, persist=bool(body.save))
        return _jsonable({"success": True, "calibration": cal.to_dict(),
                          "summary": cal.summary(self._mgr.design.sensorWidthPx)})

    @APIExport()
    def resetSpectrometerCalibration(self) -> Dict:
        """Back to the nominal design axis (the smile is kept)."""
        smile = dict(self._mgr.calibration.smile)
        self._mgr.resetCalibration(persist=False)
        self._mgr.setSmile(smile)
        return {"success": True}

    @APIExport()
    def measureSpectrometerSmile(self, nAverage: int = 5) -> Dict:
        """Measure the line curvature along the slit with one narrow line on (laser or lamp
        line); stored with the calibration."""
        try:
            self._needCamera("smile", True)
            m = self._mgr.wizardRoi(int(max(1, nAverage)))
            smile = P.measure_smile(m["roi"], m["rows"][0])
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        finally:
            self._needCamera("smile", False)
        self._mgr.setSmile(smile)
        return _jsonable({"success": True, "smile": smile})

    @APIExport()
    def clearSpectrometerSmile(self) -> Dict:
        self._mgr.setSmile(None)
        return {"success": True}

    @APIExport()
    def checkSpectrometerResolution(self, which: str = "auto") -> Dict:
        """Width of the strongest line: px, nm and cm^-1, against the slit image of the
        design (a well-focused instrument is close to it)."""
        rec = self._record(which)
        if rec is None:
            return {"success": False, "error": "no spectrum yet"}
        peaks = self._peaks(rec, n=3)
        if not peaks:
            return {"success": False, "error": "no line in the spectrum"}
        p = peaks[0]
        cal, d = self._mgr.calibration, self._mgr.design
        out = {"success": True, "line": p, "nmPerPx": float(cal.nm_per_px(p["px"])),
               "slitImagePx": d.slit_image_px(p["nm"], cal.grating["turnDeg"]),
               "calibrated": cal.calibrated}
        out["widthOverSlitImage"] = p["fwhmPx"] / out["slitImagePx"]
        if self._mgr.laserNm:
            lo = nm_to_raman_shift(p["nm"] - p["fwhmNm"] / 2, self._mgr.laserNm)
            hi = nm_to_raman_shift(p["nm"] + p["fwhmNm"] / 2, self._mgr.laserNm)
            out["fwhmCm1"] = float(abs(hi - lo))
        return _jsonable(out)

    @APIExport()
    def getSpectrometerAlignment(self, lineNm: float = 520.0, targetCenterNm: float = 0.0,
                                 which: str = "auto") -> Dict:
        """Grating alignment with a known line (e.g. the 520 nm Q-Box laser): where it
        sits now, where it must sit for the target window, and the turn between."""
        target = targetCenterNm or self._info.alignmentTargetNm or \
            self._mgr.design.centerWavelengthNm
        measured, found = self._findLine(self._record(which), float(lineNm))
        try:
            plan = alignment_plan(self._mgr.design, float(lineNm), float(target),
                                  self._mgr.calibration, measured, self._mgr.design.sensorWidthPx)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        plan["lineFound"] = found
        plan["advice"] = self._advice(plan)
        return _jsonable({"success": True, **plan})

    def _findLine(self, rec: Optional[Dict], line_nm: float):
        """Column of line_nm in a spectrum: with a calibration, the peak within
        max(2 nm, 3 FWHM) of the line ("calibrated"), else none ("not found"); without
        one, the strongest peak ("assumed strongest")."""
        peaks = self._peaks(rec, n=5) if rec is not None else []
        if not peaks:
            return None, "no line in the spectrum"
        if not self._mgr.calibration.calibrated:
            return peaks[0]["px"], "assumed strongest"
        near = min(peaks, key=lambda q: abs(q["nm"] - line_nm))
        if abs(near["nm"] - line_nm) <= max(2.0, 3.0 * near["fwhmNm"]):
            return near["px"], "calibrated"
        return None, "not found"

    @staticmethod
    def _advice(p: Dict) -> str:
        w0, w1 = p["windowAtTargetNm"]
        pre = ""
        if p.get("lineFound") == "not found":
            pre = f"No line near {p['lineNm']:g} nm in the spectrum (is it on?). "
        elif p.get("lineFound") == "assumed strongest":
            pre = (f"Not calibrated yet: taking the strongest line as {p['lineNm']:g} nm. ")
        return pre + SpectrometerController._plan_text(p, w0, w1)

    @staticmethod
    def _plan_text(p: Dict, w0: float, w1: float) -> str:
        if p["lineOnSensorAtTarget"]:
            s = (f"At the target the {p['lineNm']:g} nm line sits at column "
                 f"{p['linePxAtTarget']:.0f} (window {w0:.1f}-{w1:.1f} nm).")
            if p.get("measuredPx") is not None:
                s += (f" It is at {p['measuredPx']:.0f} now: turn the grating until it moves "
                      f"{p['linePxAtTarget'] - p['measuredPx']:+.0f} px.")
            return s
        s = (f"{p['lineNm']:g} nm lies outside the target window ({w0:.1f}-{w1:.1f} nm). Turn "
             f"the grating until the line sits at column {p['parkPx']:.0f} (grating turn "
             f"{p['turnAtParkDeg']:+.2f} deg), then turn by {p['deltaTurnFromParkDeg']:+.2f} deg")
        if p.get("leverTravelFromParkMm") is not None:
            s += f" ({p['leverTravelFromParkMm']:+.2f} mm at the lever)"
        return s + "."

    # ------------------------------------------------------------------ sources
    def _laserSources(self) -> List[CalibrationSource]:
        lm = getattr(self._master, "lasersManager", None)
        out = []
        if lm is None:
            return out
        allowed = self._info.calibrationLasers
        for name in lm.getAllDeviceNames():
            wl = getattr(lm[name], "wavelength", None)
            if not wl or (allowed is not None and name not in allowed):
                continue
            out.append(CalibrationSource(
                id=f"laser:{name}", label=f"{name} ({float(wl):g} nm)", kind="laser",
                control="laser", device=name, wavelengthNm=float(wl), sigmaNm=1.0,
                power=float(self._info.laserPowerFraction)))
        return out

    def _ledSources(self) -> List[CalibrationSource]:
        mm = getattr(self._master, "LEDMatrixsManager", None)
        names = list(mm.getAllDeviceNames()) if mm is not None else []
        device = self._info.ledMatrix or (names[0] if names else "")
        if not device and not self._mgr.simulate:
            return []
        out = []
        for ch in "rgb":
            ref = self._mgr.references.get(ch)
            if ref is None:
                continue
            out.append(CalibrationSource(
                id=f"led:{ch}", label=f"NeoPixel {CHANNEL_NAMES[ch]}"
                + (" (placeholder spectrum)" if ref.get("placeholder") else ""),
                kind="led", control="ledmatrix", device=device, channel=ch, reference=ref,
                sigmaNm=4.0 if ref.get("placeholder") else 1.5,
                power=float(self._info.ledIntensity) / 255.0))
        return out

    def _manualSources(self) -> List[CalibrationSource]:
        out = []
        for d in self._info.manualSources or []:
            d = dict(d)
            if d.get("lineList") and not d.get("lines"):
                d["lines"] = R.line_list(d["lineList"])
            kind = d.get("kind", "laser")
            d.setdefault("instructions",
                         f"Switch on {d.get('label', d['id'])} and let it shine into the "
                         "entrance (a laser: at a white card in front of it, low power), "
                         "then press Continue." if kind == "laser" else
                         f"Hold {d.get('label', d['id'])} in front of the entrance, then "
                         "press Continue.")
            out.append(CalibrationSource.from_dict({**d, "control": "manual", "kind": kind}))
        if self._mgr.simulate:
            out.append(CalibrationSource(id="sim:white", label="white LED (simulated, no "
                                         "calibration)", kind="led", control="manual",
                                         reference=None, instructions="demo source"))
        return out

    def _sources(self) -> Dict[str, CalibrationSource]:
        return {s.id: s for s in
                self._laserSources() + self._ledSources() + self._manualSources()}

    @APIExport()
    def getSpectrometerSources(self) -> Dict:
        """Calibration sources: lasers with a wavelength, the NeoPixel R/G/B channels (with
        the reference spectra in use) and sources switched by hand (Q-Box laser, lamps)."""
        out = []
        for s in self._sources().values():
            d = s.to_dict()
            d["forCalibration"] = bool(s.wavelengthNm or s.lines or s.reference)
            d["active"] = s.id in self._active
            if s.lines:
                d["lines"] = s.lines
            out.append(d)
        return _jsonable({"sources": out})

    def _emissionFor(self, src: CalibrationSource) -> List[Dict]:
        sim = self._info.simulation or {}
        true_nm = (sim.get("trueWavelengths") or {}).get(src.id)
        if src.id == "sim:white":
            return [{"type": "spectrum", **self._whiteRef, "brightness": 0.7}]
        if src.wavelengthNm:
            frac = src.power if src.control == "laser" else 0.05
            return [{"type": "line", "nm": true_nm or src.wavelengthNm,
                     "brightness": 0.5 * frac / 0.05}]
        if src.lines:
            n = len(src.lines)
            return [{"type": "lines", "nm": src.lines, "weights": list(np.linspace(1, 0.35, n)),
                     "brightness": 0.5}]
        if src.reference:
            return [{"type": "spectrum", **src.reference,
                     "brightness": 0.6 * src.power / (64 / 255)}]
        return []

    def _switch(self, src: CalibrationSource, on: bool):
        if self._mgr.simulate:
            self._mgr.setSimulatedEmission(src.id, self._emissionFor(src) if on else None)
        if src.control == "laser" and src.device:
            self._setLaser(src.device, on, src.power)
        elif src.control == "ledmatrix" and src.device:
            self._setLedChannel(src.device, src.channel, on, src.power)
        if on:
            self._active[src.id] = src.power
        else:
            self._active.pop(src.id, None)

    def _setLaser(self, name: str, on: bool, fraction: float):
        lm = self._master.lasersManager[name]
        lc = self._master.getController("Laser") if hasattr(self._master, "getController") \
            else None
        lo, hi = float(lm.valueRangeMin or 0), float(lm.valueRangeMax or 0)
        value = lo + float(np.clip(fraction, 0, 1)) * (hi - lo)
        if on:
            if not lm.isBinary:
                (lc.setLaserValue(name, value) if lc else lm.setValue(value))
            (lc.setLaserActive(name, True) if lc else lm.setEnabled(True))
        else:
            (lc.setLaserActive(name, False) if lc else lm.setEnabled(False))

    def _setLedChannel(self, name: str, ch: str, on: bool, fraction: float):
        mm = getattr(self._master, "LEDMatrixsManager", None)
        if mm is None or name not in mm.getAllDeviceNames():
            return
        v = int(round(255 * float(np.clip(fraction, 0, 1))))
        rgb = tuple(v if c == ch else 0 for c in "rgb") if on else (0, 0, 0)
        mm[name].setAll(state=1 if on else 0, intensity=rgb)

    def _allSourcesOff(self):
        srcs = self._sources()
        for sid in list(self._active):
            if sid in srcs:
                try:
                    self._switch(srcs[sid], False)
                except Exception as exc:
                    self._logger.warning(f"could not switch {sid} off: {exc}")
            self._active.pop(sid, None)

    @APIExport()
    def setSpectrometerSource(self, sourceId: str, active: bool = True,
                              power: float = -1.0) -> Dict:
        """Switch one calibration source (lasers at power x their range, LED channels at
        power x 255). Sources switched by hand only change the simulator's light."""
        srcs = self._sources()
        if sourceId not in srcs:
            return {"success": False, "error": f"unknown source {sourceId}"}
        src = srcs[sourceId]
        if power >= 0:
            src.power = float(power)
        try:
            self._switch(src, bool(active))
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        return {"success": True, "activeSources": sorted(self._active)}

    @APIExport()
    def allSpectrometerSourcesOff(self) -> Dict:
        self._allSourcesOff()
        return {"success": True}

    @APIExport(requestType="POST")
    def setSpectrometerReferenceSpectrum(self, body: SpectrometerReferenceBody) -> Dict:
        """Store the reference spectrum of an LED channel ("r", "g", "b") measured on
        another spectrometer (any sampling; normalised to its peak)."""
        try:
            spec = self._mgr.setReference(body.key, body.wavelengthNm, body.intensity,
                                          body.label)
        except Exception as exc:
            return {"success": False, "error": str(exc)}
        feat = K.reference_feature(spec["wavelengthNm"], spec["intensity"])
        return _jsonable({"success": True, "key": body.key, **feat})

    # ------------------------------------------------------------------ wizard
    @APIExport(requestType="POST")
    def startSpectrometerWizard(self, body: SpectrometerWizardBody) -> Dict:
        """Calibrate with the chosen sources, one after the other: dark, switch on,
        average, find the line(s), switch off; then fit and wait for apply/cancel."""
        if self._wizard is not None and self._wizard.state in ("running", "waiting"):
            return {"success": False, "error": "the wizard is already running"}
        if self._job["state"] == "running":
            return {"success": False, "error": f"a {self._job['kind']} is running"}
        srcs = self._sources()
        chosen = []
        for sid in body.sources:
            if sid not in srcs:
                return {"success": False, "error": f"unknown source {sid}"}
            s = srcs[sid]
            if sid in body.power:
                s.power = float(body.power[sid])
            if sid in body.wavelengths and s.wavelengthNm:
                s.wavelengthNm = float(body.wavelengths[sid])
            if sid in body.sigmas:
                s.sigmaNm = float(body.sigmas[sid])
            chosen.append(s)
        self._allSourcesOff()
        self._wizStop = threading.Event()
        n, n_dark = int(max(1, body.nAverage)), int(max(1, body.nDark))
        settings = WizardSettings(model=body.model, order=body.order,
                                  measureSmile=body.measureSmile, takeDark=body.takeDark,
                                  settleS=float(max(0.0, body.settleS)))
        try:
            self._wizard = CalibrationWizard(
                chosen, lambda: self._mgr.wizardRoi(n, stop=self._wizStop), self._switch,
                self._mgr.design, initial=self._mgr.calibration, settings=settings,
                take_dark=lambda: self._mgr.takeDark(n_dark, stop=self._wizStop),
                log=self._logger.info)
        except ValueError as exc:
            return {"success": False, "error": str(exc)}
        self._needCamera("wizard", True)
        self._wizard.start()
        threading.Thread(target=self._releaseAfterWizard, daemon=True).start()
        return {"success": True, "wizard": _jsonable(self._wizard.status())}

    def _releaseAfterWizard(self):
        wiz = self._wizard
        if wiz is not None:
            wiz.join()
        self._needCamera("wizard", False)

    @APIExport()
    def getSpectrometerWizard(self) -> Dict:
        """Wizard state: steps, message (what to do now), points, per-source spectra,
        smile, fit with residuals, resolution, hints."""
        if self._wizard is None:
            return {"state": "idle"}
        return _jsonable(self._wizard.status())

    @APIExport()
    def continueSpectrometerWizard(self) -> Dict:
        """The source asked for is on: measure it."""
        if self._wizard is None:
            return {"success": False, "error": "no wizard"}
        self._wizard.proceed()
        return {"success": True}

    @APIExport()
    def skipSpectrometerWizardStep(self) -> Dict:
        if self._wizard is None:
            return {"success": False, "error": "no wizard"}
        self._wizard.skip()
        return {"success": True}

    @APIExport()
    def cancelSpectrometerWizard(self) -> Dict:
        """Stop the wizard (sources off) or discard a fit waiting for review."""
        if self._wizard is None:
            return {"success": False, "error": "no wizard"}
        self._wizStop.set()
        self._wizard.cancel()
        if self._wizard.state == "review":
            self._wizard._set(state="cancelled", message="discarded")
        return {"success": True}

    @APIExport()
    def applySpectrometerWizard(self, save: bool = True) -> Dict:
        """Use the wizard's fit (and its smile) as the calibration; save=True stores it."""
        wiz = self._wizard
        if wiz is None or wiz.state != "review" or wiz.result is None:
            return {"success": False, "error": "no fit waiting for review"}
        cal = wiz.result
        cal.detector = self._mgr.detectorName or ""
        if self._mgr.last is not None:
            cal.frameShape = list(self._mgr.last["frameShape"])
        self._mgr.setCalibration(cal, persist=bool(save))
        wiz._set(state="saved", message="calibration applied" + (" and saved" if save else ""))
        return _jsonable({"success": True, "summary": cal.summary(self._mgr.design.sensorWidthPx)})

    # ------------------------------------------------------------------ shutdown
    def closeEvent(self):
        try:
            if self._wizard is not None:
                self._wizard.cancel()
            self._jobStop.set()
            self._liveStop.set()
            self._allSourcesOff()
        finally:
            self._needCamera("live", False)
