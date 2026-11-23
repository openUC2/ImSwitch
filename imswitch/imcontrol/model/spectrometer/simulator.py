"""Synthetic spectrometer frames for tests and the virtual microscope.

A slit image is drawn through the grating model of a "true" instrument (which can
differ from the nominal design), with smile, stripes along the slit (like the blade
diffraction of the first prototype), shot and read noise, hot pixels, an offset and a
12-bit range. Emission is a list of components:

    {"type": "line", "nm": 488.0, "brightness": 1.0}
    {"type": "lines", "nm": [...], "weights": [...], "brightness": 1.0}
    {"type": "spectrum", "wavelengthNm": [...], "intensity": [...], "brightness": 1.0}

brightness 1 gives roughly 3/4 of full scale at 10 ms for a line.
"""
from dataclasses import dataclass, field, fields
from typing import Dict, List, Optional, Sequence

import numpy as np

from .calibration import GratingDesign, grating_nm_to_px, grating_px_to_nm
from .pipeline import apply_shifts


@dataclass
class SimulationSettings:
    rows: int = 512
    cols: int = 3072
    focalLengthMm: float = 30.0       # a short lens: 488/520/635 nm and R/G/B LEDs fit
    centerNm: float = 555.0
    axisOffsetPx: float = 23.0        # camera axis off the sensor centre
    slitRows: Sequence[int] = (56, 456)
    smileBowPx: float = 16.0
    stripeRms: float = 0.10
    stripePeriodRows: float = 72.0
    offsetDn: float = 64.0
    readNoiseDn: float = 2.5
    electronsPerDn: float = 2.0
    bits: int = 12
    hotPixels: int = 40
    spikesPerFrame: float = 2.0
    lineFwhmPx: Optional[float] = None    # None: slit image width from the design
    seed: int = 7

    @classmethod
    def from_dict(cls, d: Optional[Dict]) -> "SimulationSettings":
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in names})


@dataclass
class SpectrometerSimulator:
    design: GratingDesign = field(default_factory=GratingDesign)
    settings: SimulationSettings = field(default_factory=SimulationSettings)
    exposureMs: float = 10.0
    gain: float = 1.0

    def __post_init__(self):
        s = self.settings
        self.design = GratingDesign.from_dict({**vars(self.design), "sensorWidthPx": s.cols})
        self.turnDeg = self.design.turn_for_axis(s.centerNm)
        self.axisPx = (s.cols - 1) / 2.0 + s.axisOffsetPx
        self.frameNumber = 0
        rng = np.random.default_rng(s.seed)
        self._hot = (rng.integers(0, s.rows, s.hotPixels), rng.integers(0, s.cols, s.hotPixels),
                     rng.uniform(200, 1500, s.hotPixels))
        r = np.arange(s.rows, dtype=float)
        r0, r1 = s.slitRows
        edge = 4.0
        rise = 1 / (1 + np.exp(-(r - r0) / edge))
        self._rowProfile = rise / (1 + np.exp((r - r1) / edge))
        phase = rng.uniform(0, 2 * np.pi, 3)
        stripes = sum(np.sin(2 * np.pi * r / (s.stripePeriodRows * m) + p)
                      for m, p in zip((1.0, 0.61, 1.7), phase))
        self._rowProfile *= 1 + s.stripeRms * stripes / np.std(stripes)
        u = (r - 0.5 * (r0 + r1)) / (0.5 * (r1 - r0))
        self._smile = s.smileBowPx * u ** 2      # the line bends towards +column at the ends
        self._rng = np.random.default_rng(s.seed + 1)

    # ------------------------------------------------------------------ truth
    def true_calibration(self) -> Dict:
        return {"turnDeg": self.turnDeg, "axisPx": self.axisPx,
                "focalLengthMm": self.settings.focalLengthMm}

    def px_to_nm(self, px):
        return grating_px_to_nm(px, self.design, self.turnDeg, self.axisPx,
                                self.settings.focalLengthMm)

    def nm_to_px(self, nm):
        return grating_nm_to_px(nm, self.design, self.turnDeg, self.axisPx,
                                self.settings.focalLengthMm)

    def line_fwhm_px(self, nm: float) -> float:
        if self.settings.lineFwhmPx:
            return float(self.settings.lineFwhmPx)
        d = GratingDesign.from_dict({**vars(self.design),
                                     "cameraFocalLengthMm": self.settings.focalLengthMm})
        return max(1.5, d.slit_image_px(nm, self.turnDeg))

    # ------------------------------------------------------------------ rendering
    def spectrum_electrons(self, emission: Sequence[Dict]) -> np.ndarray:
        """Electrons per pixel per row at the slit centre, for the current exposure."""
        cols = self.settings.cols
        px = np.arange(cols, dtype=float)
        out = np.zeros(cols)
        scale = self.exposureMs / 10.0 * 0.75 * (1 << self.settings.bits) \
            * self.settings.electronsPerDn
        for comp in emission:
            b = float(comp.get("brightness", 1.0))
            kind = comp.get("type", "line")
            if kind in ("line", "lines"):
                nms = np.atleast_1d(comp["nm"]).astype(float)
                wts = np.asarray(comp.get("weights", np.ones_like(nms)), dtype=float)
                for nm, wt in zip(nms, wts):
                    c = float(self.nm_to_px(nm))
                    if not np.isfinite(c):
                        continue
                    sig = self.line_fwhm_px(nm) / 2.3548
                    out += b * wt * scale * np.exp(-0.5 * ((px - c) / sig) ** 2)
            elif kind == "spectrum":
                lam = self.px_to_nm(px)
                ref = np.interp(lam, comp["wavelengthNm"], comp["intensity"], left=0, right=0)
                out += b * 0.6 * scale * ref
        return out

    def render(self, emission: Sequence[Dict], ambient: float = 0.0) -> np.ndarray:
        s = self.settings
        e_line = self.spectrum_electrons(emission) + ambient
        img = self._rowProfile[:, None] * e_line[None, :]
        img = apply_shifts(img, -self._smile)          # line at row r sits at c + smile(r)
        img = self._rng.poisson(np.clip(img, 0, None) * self.gain).astype(np.float64)
        img = img / s.electronsPerDn + s.offsetDn
        img += self._rng.normal(0, s.readNoiseDn, img.shape)
        hr, hc, hv = self._hot
        img[hr, hc] += hv * self.exposureMs / 10.0
        n_spk = self._rng.poisson(s.spikesPerFrame)
        if n_spk:
            img[self._rng.integers(0, s.rows, n_spk), self._rng.integers(0, s.cols, n_spk)] += \
                self._rng.uniform(500, 3000, n_spk)
        self.frameNumber += 1
        return np.clip(np.round(img), 0, (1 << s.bits) - 1).astype(np.uint16)


def emission_for_sources(sources: List[Dict]) -> List[Dict]:
    """Simulator components for active calibration sources (dicts with nm / lines /
    reference and a brightness)."""
    comps = []
    for src in sources:
        b = float(src.get("brightness", 1.0))
        if src.get("trueNm") or src.get("wavelengthNm"):
            comps.append({"type": "line", "nm": src.get("trueNm") or src["wavelengthNm"],
                          "brightness": b})
        elif src.get("lines"):
            n = len(src["lines"])
            w = src.get("weights") or list(np.linspace(1.0, 0.35, n))
            comps.append({"type": "lines", "nm": src["lines"], "weights": w, "brightness": b})
        elif src.get("reference"):
            comps.append({"type": "spectrum", **src["reference"], "brightness": b})
    return comps
