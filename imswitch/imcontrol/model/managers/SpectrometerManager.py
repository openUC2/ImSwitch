"""A grating spectrometer built from a camera: frames, dark, processing, calibration.

The manager owns what belongs to the instrument: which camera, the processing
settings, the dark frame, the pixel -> nm calibration (persisted in
<config>/spectrometer/<name>.json), the reference spectra of the calibration LEDs and,
with ``simulate``, a simulator that renders frames instead of the camera. Switching
lasers and LEDs and the REST API live in SpectrometerController. The numerics are in
imswitch.imcontrol.model.spectrometer (no ImSwitch imports, tested on their own).
"""
import base64
import io
import json
import math
import os
import threading
import time
from datetime import datetime
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

from imswitch.imcommon.model import dirtools, initLogger
from imswitch.imcontrol.model.SetupInfo import SpectrometerInfo
from imswitch.imcontrol.model.managers.detectors.DetectorManager import getDetectorExposureMs
from imswitch.imcontrol.model.spectrometer import pipeline as P
from imswitch.imcontrol.model.spectrometer import references as R
from imswitch.imcontrol.model.spectrometer import spectrum_io
from imswitch.imcontrol.model.spectrometer.calibration import (Calibration, GratingDesign,
                                                               nm_to_raman_shift)
from imswitch.imcontrol.model.spectrometer.simulator import (SimulationSettings,
                                                             SpectrometerSimulator)

CHUNK = 8          # frames median-cleaned together; longer averages go chunk by chunk


def _png_data_url(img: np.ndarray, max_width: int = 1536, lo=None, hi=None) -> Dict:
    """8-bit PNG (percentile stretch) of a 2-D array, downsampled to max_width."""
    from PIL import Image
    a = np.asarray(img, dtype=np.float32)
    step = max(1, int(math.ceil(a.shape[1] / max_width)))
    if step > 1:
        h, w = (a.shape[0] // step) * step, (a.shape[1] // step) * step
        a = a[:h, :w].reshape(h // step, step, w // step, step).mean(axis=(1, 3))
    lo = float(np.percentile(a, 1)) if lo is None else lo
    hi = float(np.percentile(a, 99.9)) if hi is None else hi
    u8 = np.clip((a - lo) / max(hi - lo, 1e-6) * 255, 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(u8).save(buf, format="PNG")
    return {"image": "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode(),
            "step": step, "levels": [lo, hi]}


class SpectrometerManager:
    def __init__(self, spectrometerInfo: Optional[SpectrometerInfo], detectorsManager=None,
                 configDir: Optional[str] = None):
        self._logger = initLogger(self)
        self.info = spectrometerInfo or SpectrometerInfo()
        self._detectors = detectorsManager
        names = list(detectorsManager.getAllDeviceNames()) if detectorsManager else []
        self.detectorName = self.info.detector or (names[0] if names else None)
        self.simulate = bool(self.info.simulate) or self.detectorName is None
        self._lock = threading.RLock()
        self._simLock = threading.Lock()
        self._dir = os.path.join(configDir or dirtools.UserFileDirs.Config, "spectrometer")
        self._file = os.path.join(self._dir, f"{self.info.name}.json")
        saved = self._load()

        self.design = GratingDesign.from_dict(self.info.design)
        self.simulator = None
        if self.simulate:
            self.simulator = SpectrometerSimulator(
                design=self.design, settings=SimulationSettings.from_dict(self.info.simulation))
            self.design = GratingDesign.from_dict(
                {**vars(self.design), "sensorWidthPx": self.simulator.settings.cols})
        self.settings = P.ProcessingSettings.from_dict(
            {**self.info.processing, **saved.get("processing", {})})
        self.laserNm = saved.get("laserNm", self.info.laserWavelengthNm)
        self.calibration = (Calibration.from_dict(saved["calibration"])
                            if saved.get("calibration") else self.designCalibration())
        self.previousCalibration = saved.get("previousCalibration")
        self.references = self._loadReferences()
        self.dark: Optional[np.ndarray] = None
        self.darkMeta: Dict = {}
        self._emission: Dict[str, List[Dict]] = {}
        self.last: Optional[Dict] = None          # last processed spectrum (live or acquired)
        self.lastAcquired: Optional[Dict] = None

    # ------------------------------------------------------------------ persistence
    def _load(self) -> Dict:
        try:
            with open(self._file, encoding="utf-8") as fh:
                return json.load(fh)
        except FileNotFoundError:
            return {}
        except Exception as exc:
            self._logger.warning(f"Could not read {self._file}: {exc}")
            return {}

    def save(self):
        os.makedirs(self._dir, exist_ok=True)
        doc = {"processing": self.settings.to_dict(), "laserNm": self.laserNm,
               "calibration": self.calibration.to_dict(),
               "previousCalibration": self.previousCalibration,
               "savedAt": datetime.now().isoformat(timespec="seconds")}
        tmp = self._file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(spectrum_io.to_json(doc))
        os.replace(tmp, self._file)

    @property
    def stateFile(self) -> str:
        return self._file

    # ------------------------------------------------------------------ calibration
    def designCalibration(self) -> Calibration:
        return Calibration(design=self.design, laserNm=self.laserNm,
                           detector=self.detectorName or "")

    def setCalibration(self, cal: Calibration, persist: bool = True):
        with self._lock:
            if self.calibration.calibrated:
                self.previousCalibration = self.calibration.to_dict()
            cal.laserNm = self.laserNm
            self.calibration = cal
            if persist:
                self.save()

    def resetCalibration(self, persist: bool = True):
        self.setCalibration(self.designCalibration(), persist)

    def setLaserNm(self, laser_nm: Optional[float]):
        with self._lock:
            self.laserNm = float(laser_nm) if laser_nm else None
            self.calibration.laserNm = self.laserNm
            self.save()

    def setSmile(self, smile: Optional[Dict], persist: bool = True):
        with self._lock:
            self.calibration.smile = dict(smile or {})
            if persist:
                self.save()

    def designWarnings(self) -> List[str]:
        out = []
        cal_design = vars(self.calibration.design)
        if self.calibration.calibrated and cal_design != vars(self.design):
            changed = [k for k, v in vars(self.design).items() if cal_design.get(k) != v]
            out.append("the setup's design changed since this calibration: " + ", ".join(changed))
        shape = self.calibration.frameShape
        if self.calibration.calibrated and shape and self.last and \
                list(self.last["frameShape"]) != list(shape):
            out.append(f"frame shape {self.last['frameShape']} differs from the calibration's "
                       f"{shape} (binning or camera crop changed?)")
        return out

    # ------------------------------------------------------------------ references
    def _referencesFile(self) -> Optional[str]:
        f = self.info.referenceSpectraFile
        if f:
            return f if os.path.isabs(f) else os.path.join(self._dir, f)
        own = os.path.join(self._dir, f"{self.info.name}_references.json")
        return own if os.path.isfile(own) else None

    def _loadReferences(self) -> Dict[str, Dict]:
        refs = R.placeholder_led_spectra()
        path = self._referencesFile()
        if path and os.path.isfile(path):
            try:
                refs.update(R.load_reference_spectra(path))
            except Exception as exc:
                self._logger.warning(f"Reference spectra {path}: {exc}")
        return refs

    def setReference(self, key: str, wavelength_nm: Sequence[float], intensity: Sequence[float],
                     label: str = "") -> Dict:
        spec = R.normalise_spectrum({"wavelengthNm": list(wavelength_nm),
                                     "intensity": list(intensity),
                                     "label": label or key, "placeholder": False})
        path = os.path.join(self._dir, f"{self.info.name}_references.json")
        os.makedirs(self._dir, exist_ok=True)
        doc = {}
        if os.path.isfile(path):
            with open(path, encoding="utf-8") as fh:
                doc = json.load(fh)
        doc.setdefault("sources", {})[key] = spec
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(spectrum_io.to_json(doc))
        self.references = self._loadReferences()
        return spec

    # ------------------------------------------------------------------ camera
    def detector(self):
        if self.simulate or self._detectors is None:
            return None
        return self._detectors[self.detectorName]

    def _parameterName(self, *keys: str) -> Optional[str]:
        det = self.detector()
        if det is None:
            return None
        want = {k.replace("_", "").lower() for k in keys}
        for name in det.parameters:
            if name.replace("_", "").replace(" ", "").lower() in want:
                return name
        return None

    def bitDepth(self, frame: Optional[np.ndarray] = None) -> int:
        if self.simulate:
            return int(self.simulator.settings.bits)
        det = self.detector()
        name = self.info.bitDepthParameter
        if name and name in det.parameters:
            v = det.parameters[name].value
            digits = "".join(ch for ch in str(v) if ch.isdigit())
            if digits:
                return int(digits)
        pmax = det.parameters.get("previewMaxValue")
        if pmax is not None and float(pmax.value) > 0:
            return int(round(math.log2(float(pmax.value) + 1)))
        if frame is not None:
            return 8 if np.asarray(frame).dtype == np.uint8 else 16
        return 16

    def cameraState(self) -> Dict:
        if self.simulate:
            sim = self.simulator
            return {"simulated": True, "exposureMs": sim.exposureMs, "gain": sim.gain,
                    "bitDepth": sim.settings.bits, "bitDepthSettable": False,
                    "frameShape": [sim.settings.rows, sim.settings.cols],
                    "model": "simulated spectrometer", "detector": None}
        det = self.detector()
        gain = self._parameterName("gain")
        return {"simulated": False, "exposureMs": getDetectorExposureMs(det),
                "gain": float(det.parameters[gain].value) if gain else None,
                "bitDepth": self.bitDepth(),
                "bitDepthSettable": bool(self.info.bitDepthParameter
                                         and self.info.bitDepthParameter in det.parameters),
                "frameShape": (list(self.last["frameShape"]) if self.last else None),
                "model": getattr(det, "model", ""), "detector": self.detectorName}

    def setCamera(self, exposureMs: Optional[float] = None, gain: Optional[float] = None,
                  bitDepth: Optional[int] = None) -> Dict:
        notes = (self._setSimulatedCamera(exposureMs, gain, bitDepth) if self.simulate
                 else self._setDetector(exposureMs, gain, bitDepth))
        return {**self.cameraState(), "notes": notes}

    def _setSimulatedCamera(self, exposureMs, gain, bitDepth) -> List[str]:
        if exposureMs is not None:
            self.simulator.exposureMs = float(max(0.01, exposureMs))
        if gain is not None:
            self.simulator.gain = float(max(0.0, gain))
        return ["the simulator is fixed at 12 bit"] if bitDepth else []

    def _setDetector(self, exposureMs, gain, bitDepth) -> List[str]:
        det = self.detector()
        for value, keys in ((exposureMs, ("exposure", "exposuretime", "exposure_time")),
                            (gain, ("gain",))):
            if value is None:
                continue
            name = self._parameterName(*keys)
            if name is None:
                raise ValueError(f"{self.detectorName} has no {keys[0]} parameter")
            det.setParameter(name, float(value))
        if not bitDepth:
            return []
        name = self.info.bitDepthParameter
        if name and name in det.parameters:
            det.setParameter(name, bitDepth)
            return []
        return [f"bit depth fixed by the camera driver ({self.bitDepth()} bit); set "
                "spectrometer.bitDepthParameter if it has a switch"]

    def exposureS(self) -> float:
        if self.simulate:
            return self.simulator.exposureMs / 1000.0
        return max(getDetectorExposureMs(self.detector()), 0.0) / 1000.0

    # ------------------------------------------------------------------ frames
    def setSimulatedEmission(self, source_id: str, components: Optional[List[Dict]]):
        with self._lock:
            if components:
                self._emission[source_id] = list(components)
            else:
                self._emission.pop(source_id, None)

    def simulatedSources(self) -> List[str]:
        return sorted(self._emission)

    def _latest(self):
        det = self.detector()
        try:
            out = det.getLatestFrame(returnFrameNumber=True)
            if isinstance(out, tuple) and len(out) == 2:
                return out[0], out[1]
        except TypeError:
            pass
        return det.getLatestFrame(), None

    def grabFrame(self, fresh: bool = False, after: Optional[int] = None,
                  timeout: float = 10.0) -> Tuple[np.ndarray, int]:
        """(frame, frame number). fresh: exposure began after this call (two frames
        on). after: the next frame after that frame number (for averaging)."""
        if self.simulate:
            with self._lock:
                emission = [c for v in self._emission.values() for c in v]
            if fresh or after is not None:
                time.sleep(self.simulator.exposureMs / 1000.0)
            with self._simLock:                  # the simulator's random state is shared
                return self.simulator.render(emission), self.simulator.frameNumber
        t0 = time.time()
        frame, n = self._latest()
        if fresh or after is not None:
            if n is None or n < 0:            # no frame counter: wait for new exposures
                time.sleep((2 if fresh else 1) * self.exposureS() + 0.05)
                frame, n = self._latest()
            else:
                target = n + 2 if fresh else after + 1
                limit = timeout + 3 * self.exposureS()
                while n < target:
                    if time.time() - t0 > limit:
                        raise TimeoutError("no new frame from the camera (is it running?)")
                    time.sleep(min(0.02, self.exposureS() / 4 + 0.005))
                    frame, n = self._latest()
        if frame is None or np.size(frame) == 0:
            raise RuntimeError(f"no frame from {self.detectorName} (is the camera running?)")
        return np.asarray(frame), (n if n is not None else -1)

    def bounds(self, shape) -> Tuple[int, int, int, int]:
        return self.settings.bounds(shape[:2])

    def acquireRoi(self, n: int, fresh: bool = True, progress: Optional[Callable] = None,
                   stop: Optional[threading.Event] = None) -> Dict:
        """Mean of n frames, cut to the region; spikes median-cleaned in chunks of 8.

        Returns {"roi", "rows", "cols", "frameShape", "saturation", "spikesRemoved",
        "frames", "frameNumbers"} (no dark subtraction, no smile correction)."""
        n = max(1, int(n))
        chunks, chunk, peak, nums, spikes = [], [], None, [], 0
        r0 = r1 = c0 = c1 = 0
        shape = None
        for i in range(n):
            if stop is not None and stop.is_set():
                raise InterruptedError("acquisition stopped")
            if i == 0:
                frame, num = self.grabFrame(fresh=fresh)
            else:
                frame, num = self.grabFrame(after=nums[-1] if nums[-1] >= 0 else None,
                                            fresh=nums[-1] < 0)
            mono = P.as_mono(frame)
            if shape is None:
                shape = mono.shape
                r0, r1, c0, c1 = self.bounds(shape)
            cut = np.asarray(frame)[r0:r1, c0:c1]
            m = float(np.max(cut))
            peak = m if peak is None else max(peak, m)
            chunk.append(mono[r0:r1, c0:c1].copy())
            nums.append(num)
            if len(chunk) == CHUNK or i == n - 1:
                mean, k = P.stack_mean(chunk, self.settings.removeSpikes,
                                       self.settings.spikeThreshold)
                chunks.append((mean, len(chunk)))
                spikes += k
                chunk = []
            if progress:
                progress(i + 1, n)
        total = sum(w for _, w in chunks)
        roi = sum(m * w for m, w in chunks) / total
        bits = self.bitDepth()
        top = (1 << bits) - 1
        sat = {"peak": peak, "fullScale": top, "peakFraction": peak / top, "saturated": peak >= top}
        return {"roi": roi.astype(np.float32), "rows": [r0, r1], "cols": [c0, c1],
                "frameShape": list(shape), "saturation": sat, "spikesRemoved": spikes,
                "frames": n, "frameNumbers": nums, "despiked": n >= 3}

    def takeDark(self, n: int = 10, progress: Optional[Callable] = None,
                 stop: Optional[threading.Event] = None) -> Dict:
        """Full-frame mean dark (so that a later change of the region keeps it valid)."""
        acc, shape = None, None
        for i in range(max(1, int(n))):
            if stop is not None and stop.is_set():
                raise InterruptedError("dark stopped")
            frame, _ = self.grabFrame(fresh=True)
            mono = P.as_mono(frame).astype(np.float64)
            acc = mono if acc is None else acc + mono
            shape = mono.shape
            if progress:
                progress(i + 1, n)
        dark = (acc / max(1, int(n))).astype(np.float32)
        meta = {"frames": int(n), "exposureMs": self.cameraState()["exposureMs"],
                "gain": self.cameraState()["gain"], "takenAt": time.time(),
                "frameShape": list(shape), "mean": float(dark.mean())}
        with self._lock:
            self.dark, self.darkMeta = dark, meta
        return meta

    def clearDark(self):
        with self._lock:
            self.dark, self.darkMeta = None, {}

    def darkWarning(self) -> Optional[str]:
        if self.dark is None or not self.settings.subtractDark:
            return None
        cam = self.cameraState()
        exp, gain = self.darkMeta.get("exposureMs"), self.darkMeta.get("gain")
        if exp and cam["exposureMs"] and abs(exp - cam["exposureMs"]) > 1e-3 * max(exp, 1):
            return f"dark taken at {exp:g} ms, camera now at {cam['exposureMs']:g} ms"
        if gain is not None and cam["gain"] is not None and gain != cam["gain"]:
            return f"dark taken at gain {gain:g}, camera now at {cam['gain']:g}"
        return None

    def _darkRoi(self, rows, cols, shape) -> Optional[np.ndarray]:
        if self.dark is None or not self.settings.subtractDark:
            return None
        if list(self.dark.shape) != list(shape[:2]):
            return None
        return self.dark[rows[0]:rows[1], cols[0]:cols[1]]

    # ------------------------------------------------------------------ spectra
    def axes(self, px: np.ndarray) -> Dict:
        nm = self.calibration.px_to_nm(px)
        out = {"nm": nm}
        if self.laserNm:
            out["ramanShiftCm1"] = nm_to_raman_shift(nm, self.laserNm)
        return out

    def processAcquired(self, acq: Dict) -> Dict:
        """Spectrum record from acquireRoi() (dark, smile, collapse)."""
        dark = self._darkRoi(acq["rows"], acq["cols"], acq["frameShape"])
        out = P.process_roi(acq["roi"], acq["rows"], acq["cols"], self.settings, dark,
                            self.calibration.smile, despiked=acq.get("despiked", False))
        # frame adding: the sum of the (dark-subtracted) frames instead of their mean
        k = float(acq["frames"]) if self.settings.frameCombine == "sum" else 1.0
        rec = {"px": out["px"], "counts": out["counts"] * k, **self.axes(out["px"]),
               "dark": (P.collapse_rows(dark, self.settings.collapse) * k if dark is not None
                        else None),
               "roi": out["roi"] * k, "rawRoi": acq["roi"] * k,
               "frameShape": acq["frameShape"],
               "metadata": self._metadata(acq, out, dark is not None)}
        with self._lock:
            self.last = rec
        return rec

    def processFrame(self, frame: np.ndarray, frame_number: int = -1) -> Dict:
        mono = P.as_mono(frame)
        r0, r1, c0, c1 = self.bounds(mono.shape)
        cut = np.asarray(frame)[r0:r1, c0:c1]
        bits = self.bitDepth(frame)
        top = (1 << bits) - 1
        peak = float(np.max(cut))
        acq = {"roi": mono[r0:r1, c0:c1], "rows": [r0, r1], "cols": [c0, c1],
               "frameShape": list(mono.shape), "frames": 1, "frameNumbers": [frame_number],
               "saturation": {"peak": peak, "fullScale": top, "peakFraction": peak / top,
                              "saturated": peak >= top},
               "spikesRemoved": 0, "despiked": False}
        return self.processAcquired(acq)

    def wizardRoi(self, n: int, stop: Optional[threading.Event] = None) -> Dict:
        """Dark-subtracted mean region without smile correction (for the wizard)."""
        acq = self.acquireRoi(n, fresh=True, stop=stop)
        dark = self._darkRoi(acq["rows"], acq["cols"], acq["frameShape"])
        roi = acq["roi"] - dark if dark is not None else acq["roi"]
        return {"roi": roi, "rows": acq["rows"], "cols": acq["cols"],
                "saturation": acq["saturation"]}

    def _metadata(self, acq: Dict, out: Dict, dark: bool) -> Dict:
        cam = self.cameraState()
        return {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "detector": self.detectorName, "simulated": self.simulate,
            "exposureMs": cam["exposureMs"], "gain": cam["gain"], "bitDepth": cam["bitDepth"],
            "frames": acq["frames"], "frameNumbers": acq["frameNumbers"][-8:],
            "rows": acq["rows"], "cols": acq["cols"], "frameShape": acq["frameShape"],
            "collapse": self.settings.collapse, "frameCombine": self.settings.frameCombine,
            "integrationMs": (cam["exposureMs"] or 0.0)
            * (acq["frames"] if self.settings.frameCombine == "sum" else 1),
            "darkSubtracted": dark,
            "dark": {k: v for k, v in self.darkMeta.items() if k != "mean"} if dark else None,
            "smileCorrected": bool(self.calibration.smile and self.settings.smileCorrection),
            "spikesRemoved": acq["spikesRemoved"] + out["spikesRemoved"],
            "saturation": acq["saturation"], "laserNm": self.laserNm,
            "calibration": self.calibration.to_dict(),
            "simulatedSources": self.simulatedSources() if self.simulate else None}

    def record(self, which: str = "last") -> Optional[Dict]:
        return self.lastAcquired if which == "acquired" else self.last

    def saveRecord(self, rec: Dict, name: str = "spectrum", fmt: str = "hdf5",
                   include_roi: bool = True, extra: Optional[Dict] = None) -> List[str]:
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)[:60] or "spectrum"
        folder = os.path.join(dirtools.UserFileDirs.getValidatedDataPath(), "Spectrometer")
        base = os.path.join(folder, f"{time.strftime('%Y%m%d_%H%M%S')}_{safe}")
        out = {k: rec.get(k) for k in ("px", "nm", "ramanShiftCm1", "counts", "dark")}
        out["metadata"] = {**rec["metadata"], **(extra or {})}
        if include_roi:
            out["roi"] = rec.get("rawRoi")
        if fmt in ("csv", "csv+json"):
            return spectrum_io.save_csv(base + ".csv", out)
        return [spectrum_io.save_hdf5(base + ".h5", out)]

    def image(self, full: bool = False, max_width: int = 1536, corrected: bool = False) -> Dict:
        """PNG preview: the region (raw or corrected) or a fresh full frame."""
        if full:
            frame, _ = self.grabFrame(fresh=False)
            mono = P.as_mono(frame)
            shape = mono.shape
            r0, r1, c0, c1 = self.bounds(shape)
            png = _png_data_url(mono, max_width)
            return {**png, "full": True, "frameShape": list(shape), "rows": [r0, r1],
                    "cols": [c0, c1]}
        rec = self.last
        if rec is None:
            return {"image": None, "full": False}
        img = rec["roi"] if corrected else rec["rawRoi"]
        png = _png_data_url(img, max_width)
        return {**png, "full": False, "frameShape": rec["frameShape"],
                "rows": rec["metadata"]["rows"], "cols": rec["metadata"]["cols"]}
