"""Calibration wizard: switch known sources one at a time, find their lines, fit.

Hardware-free: the caller passes three callables,

    switch(source, on)       turn a source on/off (lasers, LED-matrix channels). It is
                             called for "manual" sources too (after the user said the
                             source is on, and when it is done): hardware ignores it, a
                             simulator switches its emission
    measure()                {"roi": dark-subtracted mean frame region (no smile
                             correction), "rows": [r0, r1], "cols": [c0, c1]}
    take_dark()              optional, called once with every source off

Sources: a laser line (wavelengthNm), a line lamp (lines, e.g. neon) or a broad LED
with a reference spectrum measured elsewhere (reference). The first narrow line also
measures the smile. After the last source the points are fitted (see
calibration.make_calibration) and the wizard waits in "review" until the caller saves
or discards the result.
"""
import threading
import time
import traceback
from dataclasses import asdict, dataclass, fields
from typing import Callable, Dict, List, Optional

import numpy as np

from .calibration import Calibration, GratingDesign, make_calibration
from .peaks import (best_offset_scale, dominant_feature, find_peaks, match_lines,
                    peak_properties, reference_feature)
from .pipeline import apply_shifts, collapse_rows, decimate, measure_smile, smile_shifts


class _Cancelled(Exception):
    pass


@dataclass
class CalibrationSource:
    id: str
    label: str
    kind: str = "laser"            # laser | led | lamp
    control: str = "manual"        # laser | ledmatrix | led | manual
    device: str = ""
    channel: str = ""              # r | g | b for LED-matrix channels
    wavelengthNm: Optional[float] = None
    sigmaNm: float = 1.0           # uncertainty of the source's wavelength
    lines: Optional[List[float]] = None
    lineList: str = ""
    reference: Optional[Dict] = None
    power: float = 0.05            # lasers: fraction of the range; LEDs: 0..1
    instructions: str = ""
    trueNm: Optional[float] = None  # simulator only: the real line of a nominal laser

    @classmethod
    def from_dict(cls, d: Dict) -> "CalibrationSource":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in names})

    def to_dict(self, with_reference: bool = False) -> Dict:
        d = asdict(self)
        if not with_reference and d.get("reference"):
            ref = d["reference"]
            d["reference"] = {"label": ref.get("label", ""),
                              "placeholder": bool(ref.get("placeholder")),
                              **reference_feature(ref["wavelengthNm"], ref["intensity"])}
        return d


@dataclass
class WizardSettings:
    model: str = "auto"           # auto | grating | polynomial
    order: int = 2
    measureSmile: bool = True
    takeDark: bool = True
    settleS: float = 0.5
    ledFraction: float = 0.5      # LED centre: centroid above this fraction of the peak
    minSnr: float = 8.0

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "WizardSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


class CalibrationWizard:
    def __init__(self, sources: List[CalibrationSource], measure: Callable[[], Dict],
                 switch: Callable[[CalibrationSource, bool], None], design: GratingDesign,
                 initial: Optional[Calibration] = None, settings: Optional[WizardSettings] = None,
                 take_dark: Optional[Callable[[], None]] = None,
                 log: Callable[[str], None] = print):
        if not sources:
            raise ValueError("choose at least one calibration source")
        self.sources = sources
        self._measure, self._switch, self._take_dark = measure, switch, take_dark
        self.design = design
        self.initial = initial or Calibration(design=design)
        self.settings = settings or WizardSettings()
        self._log = log
        self._lock = threading.Lock()
        self._go, self._cancel = threading.Event(), threading.Event()
        self._skip = False
        self._thread: Optional[threading.Thread] = None
        self.result: Optional[Calibration] = None
        self._current = self.initial
        self._status = {
            "state": "idle", "message": "", "stepIndex": -1, "points": [], "spectra": {},
            "smile": None, "fit": None, "resolution": [], "hints": [],
            "steps": [{"id": s.id, "label": s.label, "kind": s.kind, "control": s.control,
                       "status": "pending", "detail": ""} for s in sources],
            "startedAt": None, "finishedAt": None}

    # ------------------------------------------------------------------ control
    def start(self):
        if self._thread is not None:
            raise RuntimeError("wizard already started")
        self._set(state="running", startedAt=time.time(), message="starting")
        self._thread = threading.Thread(target=self._run, daemon=True, name="SpectroWizard")
        self._thread.start()

    def proceed(self):
        """Continue after the user switched a manual source on."""
        self._go.set()

    def skip(self):
        """Skip the waiting (manual) step."""
        self._skip = True
        self._go.set()

    def cancel(self):
        self._cancel.set()
        self._go.set()

    def join(self, timeout: Optional[float] = None):
        if self._thread is not None:
            self._thread.join(timeout)

    @property
    def state(self) -> str:
        return self._status["state"]

    def status(self) -> Dict:
        with self._lock:
            return {**self._status, "points": [dict(p) for p in self._status["points"]],
                    "steps": [dict(s) for s in self._status["steps"]],
                    "hints": list(self._status["hints"])}

    # ------------------------------------------------------------------ internals
    def _set(self, **kw):
        with self._lock:
            self._status.update(kw)

    def _step(self, i: int, **kw):
        with self._lock:
            self._status["steps"][i].update(kw)

    def _hint(self, text: str):
        with self._lock:
            self._status["hints"].append(text)

    def _check(self):
        if self._cancel.is_set():
            raise _Cancelled()

    def _all_off(self):
        for s in self.sources:
            try:
                self._switch(s, False)
            except Exception as exc:         # never let a failed switch hide the real error
                self._log(f"could not switch {s.label} off: {exc}")

    def _wait_user(self, i: int, src: CalibrationSource) -> bool:
        msg = src.instructions or f"Switch on {src.label} and point it at the slit, then Continue."
        self._set(state="waiting", message=msg)
        self._step(i, status="waiting", detail=msg)
        self._go.clear()
        self._skip = False
        while not self._go.wait(0.2):
            pass
        self._check()
        self._set(state="running")
        return not self._skip

    def _run(self):
        try:
            self._all_off()
            if self.settings.takeDark and self._take_dark is not None:
                self._set(message="dark frame (all calibration sources off)")
                time.sleep(self.settings.settleS)
                self._take_dark()
            for i, src in enumerate(self.sources):
                self._check()
                self._set(stepIndex=i, message=f"{src.label}")
                self._one_source(i, src)
            self._check()
            self._finish()
        except (_Cancelled, InterruptedError):
            self._set(state="cancelled", message="cancelled", finishedAt=time.time())
        except Exception as exc:
            self._log(traceback.format_exc())
            self._set(state="error", message=str(exc), finishedAt=time.time())
        finally:
            self._all_off()

    def _one_source(self, i: int, src: CalibrationSource):
        if src.control == "manual" and not self._wait_user(i, src):
            self._step(i, status="skipped", detail="skipped")
            return
        self._step(i, status="switching", detail="switching on")
        self._switch(src, True)
        time.sleep(self.settings.settleS)
        self._check()
        self._step(i, status="measuring", detail="acquiring")
        try:
            m = self._measure()
        finally:
            self._switch(src, False)
        px, counts, roi_info = self._spectrum(m, src)
        pts, detail = self._points(src, px, counts)
        for p in pts:
            p.update(source=src.id, label=src.label, kind=src.kind)
        with self._lock:
            self._status["points"].extend(pts)
            dx, dy = decimate(px, counts, 1500)
            self._status["spectra"][src.id] = {
                "px": dx.tolist(), "counts": dy.tolist(), **roi_info,
                "peaks": [{"px": p["px"], "counts": p.get("heightCounts")} for p in pts]}
        self._step(i, status="done" if pts else "failed", detail=detail, points=len(pts))
        if pts:
            self._refit_provisional()
        if src.control == "manual":
            self._hint(f"You can switch {src.label} off now.")

    def _spectrum(self, m: Dict, src: CalibrationSource):
        roi = np.asarray(m["roi"], dtype=np.float32)
        r0, r1 = m["rows"]
        c0, c1 = m["cols"]
        narrow = src.wavelengthNm is not None or bool(src.lines)
        smile = self._status["smile"]
        info = {}
        if self.settings.measureSmile and smile is None and narrow:
            try:
                smile = measure_smile(roi, r0)
                self._set(smile=smile)
                info["smileBowPx"] = smile["bowPx"]
            except ValueError as exc:
                self._hint(f"smile not measured with {src.label}: {exc}")
        smile = smile or self.initial.smile
        if smile:
            roi = apply_shifts(roi, smile_shifts(np.arange(r0, r1), smile))
        return np.arange(c0, c1, dtype=float), collapse_rows(roi), info

    def _points(self, src: CalibrationSource, px, counts):
        if src.wavelengthNm is not None:
            return self._laser_point(src, px, counts)
        if src.lines:
            return self._lamp_points(src, px, counts)
        if src.reference:
            return self._led_point(src, px, counts)
        return [], "source has no wavelength, line list or reference spectrum"

    @staticmethod
    def _cut_off(f: Dict, n: int, margin: float = 2.0) -> bool:
        """True if a feature's half-height edges touch the end of the spectrum."""
        return f["left"] <= margin or f["right"] >= n - 1 - margin

    def _laser_point(self, src, px, counts):
        peaks = find_peaks(counts, max_peaks=8)
        if not peaks:
            return [], "no line found (is the source on and aimed at the slit?)"
        pred = float(np.asarray(self._current.nm_to_px(src.wavelengthNm)))
        best = peaks[0]
        if self._current.calibrated and np.isfinite(pred):
            near = [p for p in peaks[:5] if p["prominence"] > 0.25 * peaks[0]["prominence"]]
            best = min(near, key=lambda p: abs(px[p["index"]] - pred))
        props = peak_properties(counts, best["index"], best["base"])
        if props["snr"] < self.settings.minSnr:
            return [], f"line too weak (SNR {props['snr']:.0f})"
        if self._cut_off(props, counts.size):
            return [], "the line sits on the edge of the sensor; turn the grating a little"
        x = float(px[0] + props["center"])
        pt = {"px": x, "nm": float(src.wavelengthNm), "sigmaNm": float(src.sigmaNm),
              "fwhmPx": props["fwhm"], "snr": props["snr"], "heightCounts": props["height"]}
        return [pt], f"line at column {x:.1f}, FWHM {props['fwhm']:.1f} px"

    def _lamp_peaks(self, px, counts):
        peaks = find_peaks(counts, max_peaks=60, min_distance=4)
        props = [peak_properties(counts, p["index"], p["base"]) for p in peaks]
        props = [p for p in props if p["snr"] >= self.settings.minSnr
                 and not self._cut_off(p, counts.size)]
        props.sort(key=lambda p: p["center"])
        return props, np.array([px[0] + p["center"] for p in props])

    def _lamp_pairs(self, cx, pred, tol: float):
        w = self.design.sensorWidthPx
        pred = np.where((pred > -tol) & (pred < w + tol), pred, np.nan)
        return match_lines(cx, pred, tol)

    def _lamp_points(self, src, px, counts):
        props, cx = self._lamp_peaks(px, counts)
        if len(props) < 2:
            return [], f"{len(props)} line(s) found; a lamp needs at least two"
        lines = [float(v) for v in src.lines]
        width = float(np.median([p["fwhm"] for p in props]))
        c = (self.design.sensorWidthPx - 1) / 2.0
        pred0 = np.asarray(self._current.nm_to_px(lines), dtype=float)
        # coarse: offset and dispersion scale (the design may be several % off)
        off, scale, _ = best_offset_scale(cx, pred0, c, 0.3 * self.design.sensorWidthPx,
                                          max(15.0, 2.0 * width),
                                          0.03 if self._current.calibrated else 0.15)
        pairs = self._lamp_pairs(cx, c + scale * (pred0 - c) + off, max(15.0, 2.0 * width))

        def points(pairs):
            return [{"px": float(cx[k]), "nm": lines[j], "sigmaNm": 0.02,
                     "fwhmPx": props[k]["fwhm"], "snr": props[k]["snr"],
                     "heightCounts": props[k]["height"]} for j, k, _ in pairs]

        pts = points(pairs)
        earlier = list(self._status["points"])
        for tol in (1.5 * width, 1.0 * width, max(3.0, 0.5 * width)):
            if len(pts) < 3:     # refit, re-predict, re-match tighter: sorts out close pairs
                break
            model = make_calibration(earlier + pts, self.design, model="grating",
                                     start=self.initial.grating)
            pairs = self._lamp_pairs(cx, np.asarray(model.nm_to_px(lines), dtype=float), tol)
            pts = points(pairs)
        if len(pts) < 2:
            return [], f"could not match the {len(props)} lines to the {src.lineList or 'list'}"
        return pts, (f"{len(pts)} of {len(props)} lines matched (coarse: offset {off:+.0f} px, "
                     f"dispersion x{scale:.3f})")

    def _led_point(self, src, px, counts):
        f = dominant_feature(counts, self.settings.ledFraction)
        if f["snr"] < self.settings.minSnr:
            return [], f"LED too weak (SNR {f['snr']:.0f})"
        if self._cut_off(f, counts.size):
            return [], "the LED band runs off the sensor; it cannot be centred"
        ref = reference_feature(src.reference["wavelengthNm"], src.reference["intensity"],
                                self.settings.ledFraction)
        x = float(px[0] + f["center"])
        hint = ref["fwhmNm"] / f["fwhm"] if f["fwhm"] > 0 else None
        pt = {"px": x, "nm": ref["centerNm"], "sigmaNm": float(src.sigmaNm),
              "fwhmPx": f["fwhm"], "snr": f["snr"], "heightCounts": f["height"],
              "dispersionHintNmPerPx": hint}
        if src.reference.get("placeholder"):
            self._hint(f"{src.label}: placeholder reference spectrum, a few nm off is likely")
        return [pt], (f"band centre at column {x:.0f} -> {ref['centerNm']:.1f} nm; width "
                      f"suggests {hint:.4f} nm/px" if hint else f"band centre at column {x:.0f}")

    def _refit_provisional(self):
        pts = self._status["points"]
        try:
            self._current = make_calibration(pts, self.design, model="grating",
                                              start=self.initial.grating)
        except Exception as exc:
            self._log(f"provisional fit failed: {exc}")

    def _finish(self):
        pts = self._status["points"]
        if not pts:
            raise RuntimeError("no source gave a usable line; nothing to fit")
        s = self.settings
        cal = make_calibration(pts, self.design, model=s.model, order=s.order,
                               start=self.initial.grating, laserNm=self.initial.laserNm,
                               detector=self.initial.detector, frameShape=self.initial.frameShape)
        cal.smile = self._status["smile"] or dict(self.initial.smile)
        if len(pts) == 1:
            self._hint("one line only: the offset is measured, the dispersion is the design's")
        elif len({p["source"] for p in pts}) == 1 and len(pts) < 3:
            self._hint("all points from one source: check the fit with a second source")
        span = (max(p["px"] for p in pts) - min(p["px"] for p in pts)) / self.design.sensorWidthPx
        if len(pts) > 1 and span < 0.2:
            self._hint(f"the lines cover {100 * span:.0f} % of the sensor: the ends are "
                       "extrapolated")
        res = []
        for p in pts:
            if p.get("kind") in ("laser", "lamp") and p.get("fwhmPx"):
                nmpp = float(cal.nm_per_px(p["px"]))
                turn = cal.grating["turnDeg"]
                res.append({"source": p["source"], "nm": p["nm"], "fwhmPx": p["fwhmPx"],
                            "fwhmNm": p["fwhmPx"] * nmpp, "nmPerPx": nmpp,
                            "slitImagePx": self.design.slit_image_px(p["nm"], turn)})
        self.result = cal
        fit = cal.to_dict()
        fit["summary"] = cal.summary()
        self._set(state="review", fit=fit, resolution=res, finishedAt=time.time(),
                  message=f"fit: {cal.model}, rms {cal.rmsNm:.3f} nm over {len(pts)} line(s)")
